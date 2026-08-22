import Foundation
import Metal

struct ProbeCase {
    let name: String
    let function: String
    let bits: Int
    let packedBytesPerGroup: Int
    let qmax: Int
}

let cases = [
    ProbeCase(name: "q2", function: "qx_q2_gemv", bits: 2, packedBytesPerGroup: 8, qmax: 1),
    ProbeCase(name: "q3", function: "qx_q3_gemv", bits: 3, packedBytesPerGroup: 12, qmax: 3),
    ProbeCase(name: "q4", function: "qx_q4_gemv", bits: 4, packedBytesPerGroup: 16, qmax: 7),
]

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let rows = 4
let cols = 1024

func buffer<T>(_ values: [T], on device: MTLDevice) -> MTLBuffer {
    values.withUnsafeBytes { raw in
        device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)!
    }
}

func packedMatrix(rows: Int, cols: Int, item: ProbeCase) -> [UInt8] {
    let groups = (cols + 31) / 32
    var packed = [UInt8](repeating: 0, count: rows * groups * item.packedBytesPerGroup)
    let code = item.qmax + 1
    for row in 0..<rows {
        for column in 0..<cols {
            let group = column / 32
            let bit = (column % 32) * item.bits
            let base = row * groups * item.packedBytesPerGroup + group * item.packedBytesPerGroup
            for offset in 0..<item.bits where ((code >> offset) & 1) != 0 {
                packed[base + (bit + offset) / 8] |= UInt8(1 << ((bit + offset) % 8))
            }
        }
    }
    return packed
}

func runFusedExpert(_ item: ProbeCase, device: MTLDevice, queue: MTLCommandQueue) throws {
    let source = try String(contentsOfFile: "kernels/\(item.name)_expert.metal", encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: nil)
    let function = library.makeFunction(name: "qx_\(item.name)_expert")!
    let pipeline = try device.makeComputePipelineState(function: function)
    guard pipeline.maxTotalThreadsPerThreadgroup >= 1024 else {
        fatalError("fused expert requires 1024 threads, device supports \(pipeline.maxTotalThreadsPerThreadgroup)")
    }

    let gateUpPacked = packedMatrix(rows: 1024, cols: 1024, item: item)
    let downPacked = packedMatrix(rows: 1024, cols: 512, item: item)
    let gateUpScales = [UInt16](repeating: 0x3c00, count: 1024 * 32)
    let downScales = [UInt16](repeating: 0x3c00, count: 1024 * 16)
    let input = [UInt16](repeating: 0x1400, count: 1024) // Float16(1/1024)
    let output = device.makeBuffer(length: 1024 * MemoryLayout<UInt16>.stride,
                                   options: .storageModeShared)!
    let gateUpBuffer = buffer(gateUpPacked, on: device)
    let gateUpScaleBuffer = buffer(gateUpScales, on: device)
    let downBuffer = buffer(downPacked, on: device)
    let downScaleBuffer = buffer(downScales, on: device)
    let inputBuffer = buffer(input, on: device)

    func submit() -> MTLCommandBuffer {
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gateUpBuffer, offset: 0, index: 0)
        encoder.setBuffer(gateUpScaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(downBuffer, offset: 0, index: 2)
        encoder.setBuffer(downScaleBuffer, offset: 0, index: 3)
        encoder.setBuffer(inputBuffer, offset: 0, index: 4)
        encoder.setBuffer(output, offset: 0, index: 5)
        var layout: UInt32 = 0 // RowMajor reference layout
        encoder.setBytes(&layout, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        return command
    }

    let first = submit()
    first.waitUntilCompleted()
    let result = output.contents().assumingMemoryBound(to: UInt16.self)
    let actual = Float(Float16(bitPattern: result[0]))
    let expected = Float(512.0 / (1.0 + Foundation.exp(-1.0)))
    guard abs(actual - expected) < 1.0 else {
        fatalError("fused \(item.name) expert mismatch: got \(actual), expected \(expected)")
    }

    let iterations = 1000
    let start = DispatchTime.now().uptimeNanoseconds
    var last: MTLCommandBuffer?
    for _ in 0..<iterations {
        last = submit()
    }
    last!.waitUntilCompleted()
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    print("\(item.name)_expert: PASS, expected=\(expected), actual=\(actual), \(Double(iterations) / elapsed) experts/s")
}

for item in cases {
    let source = try String(contentsOfFile: "kernels/\(item.name)_expert.metal", encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: nil)
    let function = library.makeFunction(name: item.function)!
    let pipeline = try device.makeComputePipelineState(function: function)

    let packed = packedMatrix(rows: rows, cols: cols, item: item)

    let groups = (cols + 31) / 32
    let scales = [UInt16](repeating: 0x3c00, count: rows * groups)
    let input = [UInt16](repeating: 0x3c00, count: cols)
    let output = device.makeBuffer(length: rows * MemoryLayout<UInt16>.stride,
                                   options: .storageModeShared)!
    var columnCount = UInt32(cols)
    let packedBuffer = buffer(packed, on: device)
    let scaleBuffer = buffer(scales, on: device)
    let inputBuffer = buffer(input, on: device)

    func submit() -> MTLCommandBuffer {
        let command = queue.makeCommandBuffer()!
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(packedBuffer, offset: 0, index: 0)
        encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
        encoder.setBuffer(inputBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&columnCount, length: MemoryLayout<UInt32>.size, index: 4)
        let width = min(rows, pipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        return command
    }

    let first = submit()
    first.waitUntilCompleted()
    let result = output.contents().assumingMemoryBound(to: UInt16.self)
    let values = (0..<rows).map { Float(Float16(bitPattern: result[$0])) }
    guard values.allSatisfy({ abs($0 - Float(cols)) < 0.5 }) else {
        fatalError("\(item.name) output mismatch: \(values)")
    }

    let iterations = 1000
    let start = DispatchTime.now().uptimeNanoseconds
    var last: MTLCommandBuffer?
    for _ in 0..<iterations {
        last = submit()
    }
    last!.waitUntilCompleted()
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
    print("\(item.name): PASS, rows=\(rows), cols=\(cols), \(Double(iterations) / elapsed) dispatches/s")
}

for item in cases {
    try runFusedExpert(item, device: device, queue: queue)
}
print("Metal device: \(device.name)")
