#include <metal_stdlib>
using namespace metal;

inline float qx_ordered_simd_sum(float value) {
    value = value + simd_shuffle_down(value, 16u);
    value = value + simd_shuffle_down(value, 8u);
    value = value + simd_shuffle_down(value, 4u);
    value = value + simd_shuffle_down(value, 2u);
    value = value + simd_shuffle_down(value, 1u);
    return simd_broadcast(value, 0u);
}

inline uint qx_group_offset(uint row, uint group, uint groups, uint bytes_per_group, uint layout) {
    if (layout == 0) {
        return row * groups * bytes_per_group + group * bytes_per_group;
    }
    uint tile = row / 32;
    uint lane = row % 32;
    return tile * 32 * groups * bytes_per_group
        + group * 32 * bytes_per_group
        + lane * bytes_per_group;
}

inline uint qx_scale_offset(uint row, uint group, uint groups, uint layout) {
    if (layout == 0) {
        return row * groups + group;
    }
    return (row / 32) * 32 * groups + group * 32 + row % 32;
}

kernel void qx_q4_gemv(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& layout [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint qmax = 7;
    constexpr uint bytes_per_group = 16;
    uint groups = (cols + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint group = 0; group < groups; ++group) {
        uint base = qx_group_offset(row, group, groups, bytes_per_group, layout);
        float scale = float(scales[qx_scale_offset(row, group, groups, layout)]);
        uint group_column = group * group_size;
        for (uint packet = 0; packet < bytes_per_group; ++packet) {
            uint codes = uint(packed[base + packet]);
            uint column = group_column + packet * 2;
            sum += float(int(codes & 0xf) - int(qmax)) * scale * float(input[column]);
            if (column + 1 < cols) {
                sum += float(int((codes >> 4) & 0xf) - int(qmax))
                    * scale * float(input[column + 1]);
            }
        }
    }
    output[row] = half(sum);
}

// SIMD-cooperative Q4 stage for the benchmark path. One SIMD group reduces
// the K dimension for one output row; eight rows are processed per 256-thread
// threadgroup. The layout helper preserves both RowMajor and MetalTile32.
kernel void qx_q4_simd_gemv(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& layout [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint qmax = 7;
    constexpr uint bytes_per_group = 16;
    constexpr uint rows_per_threadgroup = 8;
    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    if (row >= rows) return;
    uint groups = (cols + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = lane; column < cols; column += 32) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint base = qx_group_offset(row, group, groups, bytes_per_group, layout);
        uchar byte = packed[base + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - int(qmax))
            * float(scales[qx_scale_offset(row, group, groups, layout)]);
        sum += weight * float(input[column]);
    }
    sum = qx_ordered_simd_sum(sum);
    if (lane == 0) output[row] = half(sum);
}

// Fused Granite expert path for the fixed V0 shape:
// gate/up: 1024 -> 1024, SiLU(gate) * up, down: 512 -> 1024.
// A threadgroup owns one expert invocation. Packed weights are decoded only
// at the point of use; no dequantized matrix is written to device memory.
kernel void qx_q4_expert(
    device const uchar* gate_up_packed [[buffer(0)]],
    device const half* gate_up_scales [[buffer(1)]],
    device const uchar* down_packed [[buffer(2)]],
    device const half* down_scales [[buffer(3)]],
    device const half* input [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& layout [[buffer(6)]],
    uint row [[thread_position_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint group_size = 32;
    constexpr uint qmax = 7;
    constexpr uint bytes_per_group = 16;
    constexpr uint groups = hidden / group_size;
    threadgroup half gate_up[hidden];
    threadgroup half activated[intermediate];

    float gate_up_sum = 0.0f;
    for (uint column = 0; column < hidden; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint base = qx_group_offset(row, group, groups, bytes_per_group, layout);
        uchar byte = gate_up_packed[base + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - int(qmax)) * float(gate_up_scales[qx_scale_offset(row, group, groups, layout)]);
        gate_up_sum += weight * float(input[column]);
    }
    gate_up[row] = half(gate_up_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < intermediate) {
        float gate = float(gate_up[row]);
        float up = float(gate_up[intermediate + row]);
        activated[row] = half((gate / (1.0f + qx_consensus_exp(-gate))) * up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float down_sum = 0.0f;
    for (uint column = 0; column < intermediate; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint down_groups = intermediate / group_size;
        uint base = qx_group_offset(row, group, down_groups, bytes_per_group, layout);
        uchar byte = down_packed[base + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - int(qmax)) * float(down_scales[qx_scale_offset(row, group, down_groups, layout)]);
        down_sum += weight * float(activated[column]);
    }
    output[row] = half(down_sum);
}

kernel void qx_moe_accumulate(
    device const half* expert0 [[buffer(0)]],
    device const half* expert1 [[buffer(1)]],
    device const half* expert2 [[buffer(2)]],
    device const half* expert3 [[buffer(3)]],
    device const half* expert4 [[buffer(4)]],
    device const half* expert5 [[buffer(5)]],
    device const half* expert6 [[buffer(6)]],
    device const half* expert7 [[buffer(7)]],
    constant float* weights [[buffer(8)]],
    device half* output [[buffer(9)]],
    uint row [[thread_position_in_grid]]) {
    float value = float(weights[0]) * float(expert0[row]);
    value += float(weights[1]) * float(expert1[row]);
    value += float(weights[2]) * float(expert2[row]);
    value += float(weights[3]) * float(expert3[row]);
    value += float(weights[4]) * float(expert4[row]);
    value += float(weights[5]) * float(expert5[row]);
    value += float(weights[6]) * float(expert6[row]);
    value += float(weights[7]) * float(expert7[row]);
    output[row] = half(value);
}
