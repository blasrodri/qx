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
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0 || token[0] >= rows) return;
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    uint row = token[0];
    uint groups = (columns + group_size - 1) / group_size;
    for (uint column = 0; column < columns; ++column) {
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
