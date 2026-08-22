#include <metal_stdlib>
using namespace metal;

// Intentionally simple FP16-storage GEMV baseline. One thread owns one output
// row and accumulates in FP32. This is a control implementation for the Q4
// kernel experiment, not the optimized kernel design.
kernel void qx_f16_dense_gemv(
    device const half* weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    float sum = 0.0f;
    uint offset = row * columns;
    for (uint column = 0; column < columns; ++column) {
        sum += float(weights[offset + column]) * float(input[column]);
    }
    output[row] = half(sum);
}
