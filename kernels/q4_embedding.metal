#include <metal_stdlib>
using namespace metal;

kernel void qx_q4_embedding_lookup(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const uint* token [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]]) {
    if (token[0] >= rows) return;
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    uint row = token[0];
    uint groups = (columns + group_size - 1) / group_size;
    for (uint column = tid; column < columns; column += 256u) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
        output[column] = weight * multiplier;
    }
}

kernel void qx_q4_embedding_lookup_batch(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const uint* tokens [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint token_index [[threadgroup_position_in_grid]]) {
    if (token_index >= token_count) return;
    uint row = tokens[token_index];
    if (row >= rows) return;
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    uint groups = (columns + group_size - 1) / group_size;
    for (uint column = tid; column < columns; column += 256) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group
            + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
        output[token_index * columns + column] = weight * multiplier;
    }
}

kernel void qx_q4_embedding_superblock144_lookup(
    device const uchar* stream [[buffer(0)]],
    device const uchar* ignored_scales [[buffer(1)]],
    device const uint* token [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]]) {
    uint row = token[0];
    if (row >= rows) return;
    uint blocks = columns / 256u;
    uint row_bytes = blocks * 144u;
    for (uint column = tid; column < columns; column += 256u) {
        uint block = column / 256u;
        uint local = column % 256u;
        uint base = row * row_bytes + block * 144u;
        uchar packed = stream[base + local / 2u];
        uint code = (packed >> ((local & 1u) * 4u)) & 0xfu;
        half scale = reinterpret_cast<device const half*>(stream + base + 128u)[local / 32u];
        output[column] = float(int(code) - 7) * float(scale) * multiplier;
    }
}

kernel void qx_q4_embedding_superblock144_lookup_batch(
    device const uchar* stream [[buffer(0)]],
    device const uchar* ignored_scales [[buffer(1)]],
    device const uint* tokens [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint token_index [[threadgroup_position_in_grid]]) {
    if (token_index >= token_count) return;
    uint row = tokens[token_index];
    if (row >= rows) return;
    uint blocks = columns / 256u;
    uint row_bytes = blocks * 144u;
    for (uint column = tid; column < columns; column += 256u) {
        uint block = column / 256u;
        uint local = column % 256u;
        uint base = row * row_bytes + block * 144u;
        uchar packed = stream[base + local / 2u];
        uint code = (packed >> ((local & 1u) * 4u)) & 0xfu;
        half scale = reinterpret_cast<device const half*>(stream + base + 128u)[local / 32u];
        output[token_index * columns + column] =
            float(int(code) - 7) * float(scale) * multiplier;
    }
}
