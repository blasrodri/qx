#include <metal_stdlib>
using namespace metal;

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

kernel void qx_q3_gemv(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint qmax = 3;
    constexpr uint bytes_per_group = 12;
    uint groups = (cols + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = 0; column < cols; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint bit = local * 3;
        uint byte = bit / 8;
        uint shift = bit % 8;
        uint word = uint(packed[row * groups * bytes_per_group + group * bytes_per_group + byte]);
        if (shift + 3 > 8) {
            word |= uint(packed[row * groups * bytes_per_group + group * bytes_per_group + byte + 1]) << 8;
        }
        uint code = (word >> shift) & 0x7;
        float weight = (float(int(code) - int(qmax))) * float(scales[row * groups + group]);
        sum += weight * float(input[column]);
    }
    output[row] = half(sum);
}

kernel void qx_q3_expert(
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
    constexpr uint qmax = 3;
    constexpr uint bytes_per_group = 12;
    constexpr uint groups = hidden / group_size;
    threadgroup half gate_up[hidden];
    threadgroup half activated[intermediate];

    float gate_up_sum = 0.0f;
    for (uint column = 0; column < hidden; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint bit = local * 3;
        uint byte = bit / 8;
        uint shift = bit % 8;
        uint base = qx_group_offset(row, group, groups, bytes_per_group, layout);
        uint word = uint(gate_up_packed[base + byte]);
        if (shift + 3 > 8) {
            word |= uint(gate_up_packed[base + byte + 1]) << 8;
        }
        uint code = (word >> shift) & 0x7;
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
        uint bit = local * 3;
        uint byte = bit / 8;
        uint shift = bit % 8;
        uint down_groups = intermediate / group_size;
        uint base = qx_group_offset(row, group, down_groups, bytes_per_group, layout);
        uint word = uint(down_packed[base + byte]);
        if (shift + 3 > 8) {
            word |= uint(down_packed[base + byte + 1]) << 8;
        }
        uint code = (word >> shift) & 0x7;
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
