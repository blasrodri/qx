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

inline float qx_q2s4_decode(uint code, float scale) {
    constexpr float inner = 0.32f;
    float level = code == 0 ? -1.0f : (code == 1 ? -inner : (code == 2 ? inner : 1.0f));
    return level * scale;
}

// One output row per thread. Values remain packed until the multiply.
kernel void qx_q2_gemv(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint qmax = 1;
    constexpr uint bytes_per_group = 8;
    uint groups = (cols + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = 0; column < cols; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group + group * bytes_per_group + local / 4];
        uint code = (byte >> ((local % 4) * 2)) & 0x3;
        float weight = (float(int(code) - int(qmax))) * float(scales[row * groups + group]);
        sum += weight * float(input[column]);
    }
    output[row] = half(sum);
}

kernel void qx_q2s4_gemv(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 8;
    uint groups = (cols + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = 0; column < cols; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group + group * bytes_per_group + local / 4];
        uint code = (byte >> ((local % 4) * 2)) & 0x3;
        float weight = qx_q2s4_decode(code, float(scales[row * groups + group]));
        sum += weight * float(input[column]);
    }
    output[row] = half(sum);
}

kernel void qx_q2_expert(
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
    constexpr uint qmax = 1;
    constexpr uint bytes_per_group = 8;
    constexpr uint groups = hidden / group_size;
    threadgroup half gate_up[hidden];
    threadgroup half activated[intermediate];

    float gate_up_sum = 0.0f;
    for (uint column = 0; column < hidden; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint base = qx_group_offset(row, group, groups, bytes_per_group, layout);
        uchar byte = gate_up_packed[base + local / 4];
        uint code = (byte >> ((local % 4) * 2)) & 0x3;
        float weight = float(int(code) - int(qmax)) * float(gate_up_scales[qx_scale_offset(row, group, groups, layout)]);
        gate_up_sum += weight * float(input[column]);
    }
    gate_up[row] = half(gate_up_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < intermediate) {
        float gate = float(gate_up[row]);
        float up = float(gate_up[intermediate + row]);
        activated[row] = half((gate / (1.0f + exp(-gate))) * up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float down_sum = 0.0f;
    for (uint column = 0; column < intermediate; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint down_groups = intermediate / group_size;
        uint base = qx_group_offset(row, group, down_groups, bytes_per_group, layout);
        uchar byte = down_packed[base + local / 4];
        uint code = (byte >> ((local % 4) * 2)) & 0x3;
        float weight = float(int(code) - int(qmax)) * float(down_scales[qx_scale_offset(row, group, down_groups, layout)]);
        down_sum += weight * float(activated[column]);
    }
    output[row] = half(down_sum);
}

kernel void qx_q2s4_expert(
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
    constexpr uint bytes_per_group = 8;
    constexpr uint groups = hidden / group_size;
    threadgroup half gate_up[hidden];
    threadgroup half activated[intermediate];

    float gate_up_sum = 0.0f;
    for (uint column = 0; column < hidden; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint base = qx_group_offset(row, group, groups, bytes_per_group, layout);
        uchar byte = gate_up_packed[base + local / 4];
        uint code = (byte >> ((local % 4) * 2)) & 0x3;
        float weight = qx_q2s4_decode(code, float(gate_up_scales[qx_scale_offset(row, group, groups, layout)]));
        gate_up_sum += weight * float(input[column]);
    }
    gate_up[row] = half(gate_up_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < intermediate) {
        float gate = float(gate_up[row]);
        float up = float(gate_up[intermediate + row]);
        activated[row] = half((gate / (1.0f + exp(-gate))) * up);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float down_sum = 0.0f;
    for (uint column = 0; column < intermediate; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint down_groups = intermediate / group_size;
        uint base = qx_group_offset(row, group, down_groups, bytes_per_group, layout);
        uchar byte = down_packed[base + local / 4];
        uint code = (byte >> ((local % 4) * 2)) & 0x3;
        float weight = qx_q2s4_decode(code, float(down_scales[qx_scale_offset(row, group, down_groups, layout)]));
        down_sum += weight * float(activated[column]);
    }
    output[row] = half(down_sum);
}

// Qwen3-30B-A3B expert correctness path. One 768-thread group computes the
// complete SwiGLU expert: every lane owns one gate/up row, then strides over
// the 2,048 down rows. Packed weights remain group-32 Q2S4 throughout.
kernel void qx_q2s4_qwen_expert(
    device const uchar* gate_up_packed [[buffer(0)]],
    device const half* gate_up_scales [[buffer(1)]],
    device const uchar* down_packed [[buffer(2)]],
    device const half* down_scales [[buffer(3)]],
    device const half* input [[buffer(4)]],
    device half* output [[buffer(5)]],
    uint lane [[thread_position_in_threadgroup]]) {
    constexpr uint hidden = 2048;
    constexpr uint intermediate = 768;
    constexpr uint group_size = 32;
    constexpr uint gate_groups = hidden / group_size;
    constexpr uint down_groups = intermediate / group_size;
    constexpr uint bytes_per_group = 8;
    threadgroup half activated[intermediate];

    float gate = 0.0f;
    float up = 0.0f;
    uint up_row = intermediate + lane;
    for (uint column = 0; column < hidden; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uint gate_base = (lane * gate_groups + group) * bytes_per_group;
        uint up_base = (up_row * gate_groups + group) * bytes_per_group;
        uint shift = (local % 4) * 2;
        uint gate_code = (gate_up_packed[gate_base + local / 4] >> shift) & 0x3;
        uint up_code = (gate_up_packed[up_base + local / 4] >> shift) & 0x3;
        gate += qx_q2s4_decode(gate_code, float(gate_up_scales[lane * gate_groups + group]))
            * float(input[column]);
        up += qx_q2s4_decode(up_code, float(gate_up_scales[up_row * gate_groups + group]))
            * float(input[column]);
    }
    half gate_half = half(gate);
    half up_half = half(up);
    float gate_rounded = float(gate_half);
    activated[lane] = half((gate_rounded / (1.0f + exp(-gate_rounded))) * float(up_half));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = lane; row < hidden; row += intermediate) {
        float down = 0.0f;
        for (uint column = 0; column < intermediate; ++column) {
            uint group = column / group_size;
            uint local = column % group_size;
            uint base = (row * down_groups + group) * bytes_per_group;
            uint code = (down_packed[base + local / 4] >> ((local % 4) * 2)) & 0x3;
            down += qx_q2s4_decode(code, float(down_scales[row * down_groups + group]))
                * float(activated[column]);
        }
        output[row] = half(down);
    }
}

// Codec IDs used by the heterogeneous Qwen directory. They are intentionally
// explicit rather than QuantType enum ordinals, which are not an artifact ABI.
inline float qx_qwen_mixed_weight(
    device const uchar* packed,
    device const half* scales,
    uint row,
    uint column,
    uint groups,
    uchar codec) {
    constexpr uint group_size = 32;
    uint group = column / group_size;
    uint local = column % group_size;
    float scale = float(scales[row * groups + group]);
    if (codec == 2) {
        constexpr uint bytes_per_group = 8;
        uint base = (row * groups + group) * bytes_per_group;
        uint code = (uint(packed[base + local / 4]) >> ((local % 4) * 2)) & 0x3;
        return qx_q2s4_decode(code, scale);
    }
    if (codec == 3) {
        constexpr uint bytes_per_group = 12;
        uint base = (row * groups + group) * bytes_per_group;
        uint bit = local * 3;
        uint byte = bit / 8;
        uint shift = bit % 8;
        uint word = uint(packed[base + byte]);
        if (shift + 3 > 8) {
            word |= uint(packed[base + byte + 1]) << 8;
        }
        uint code = (word >> shift) & 0x7;
        return float(int(code) - 3) * scale;
    }
    constexpr uint bytes_per_group = 16;
    uint base = (row * groups + group) * bytes_per_group;
    uint code = (uint(packed[base + local / 2]) >> ((local % 2) * 4)) & 0xf;
    return float(int(code) - 7) * scale;
}

inline float qx_qwen_q4_dot_device(
    device const uchar* packed,
    device const half* scales,
    uint row,
    uint groups,
    device const half* input) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    float sum = 0.0f;
    for (uint group = 0; group < groups; ++group) {
        uint base = (row * groups + group) * bytes_per_group;
        float scale = float(scales[row * groups + group]);
        uint group_column = group * group_size;
        for (uint packet = 0; packet < bytes_per_group; ++packet) {
            uint codes = uint(packed[base + packet]);
            uint column = group_column + packet * 2;
            sum += float(int(codes & 0xf) - 7) * scale * float(input[column]);
            sum += float(int((codes >> 4) & 0xf) - 7) * scale * float(input[column + 1]);
        }
    }
    return sum;
}

kernel void qx_qwen_moe_dynamic(
    device const uchar* slab [[buffer(0)]],
    device const ulong* gate_data_offsets [[buffer(1)]],
    device const ulong* gate_scale_offsets [[buffer(2)]],
    device const ulong* down_data_offsets [[buffer(3)]],
    device const ulong* down_scale_offsets [[buffer(4)]],
    device const uchar* codecs [[buffer(5)]],
    device const half* input [[buffer(6)]],
    device const uint* expert_ids [[buffer(7)]],
    device half* expert_outputs [[buffer(8)]],
    uint lane [[thread_position_in_threadgroup]],
    uint slot [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 2048;
    constexpr uint intermediate = 768;
    constexpr uint group_size = 32;
    constexpr uint gate_groups = hidden / group_size;
    constexpr uint down_groups = intermediate / group_size;
    uint expert = expert_ids[slot];
    uchar codec = codecs[expert];
    device const uchar* gate_data = slab + gate_data_offsets[expert];
    device const half* gate_scales =
        reinterpret_cast<device const half*>(slab + gate_scale_offsets[expert]);
    device const uchar* down_data = slab + down_data_offsets[expert];
    device const half* down_scales =
        reinterpret_cast<device const half*>(slab + down_scale_offsets[expert]);
    threadgroup half activated[intermediate];

    float gate = 0.0f;
    float up = 0.0f;
    uint up_row = intermediate + lane;
    if (codec == 2) {
        constexpr uint bytes_per_group = 8;
        for (uint group = 0; group < gate_groups; ++group) {
            uint gate_base = (lane * gate_groups + group) * bytes_per_group;
            uint up_base = (up_row * gate_groups + group) * bytes_per_group;
            float gate_scale = float(gate_scales[lane * gate_groups + group]);
            float up_scale = float(gate_scales[up_row * gate_groups + group]);
            uint group_column = group * group_size;
            for (uint packet = 0; packet < bytes_per_group; ++packet) {
                uint gate_packet = uint(gate_data[gate_base + packet]);
                uint up_packet = uint(gate_data[up_base + packet]);
                uint packet_column = group_column + packet * 4;
                for (uint element = 0; element < 4; ++element) {
                    uint shift = element * 2;
                    float value = float(input[packet_column + element]);
                    gate += qx_q2s4_decode((gate_packet >> shift) & 0x3, gate_scale) * value;
                    up += qx_q2s4_decode((up_packet >> shift) & 0x3, up_scale) * value;
                }
            }
        }
    } else {
        for (uint column = 0; column < hidden; ++column) {
            float value = float(input[column]);
            gate += qx_qwen_mixed_weight(
                gate_data, gate_scales, lane, column, gate_groups, codec) * value;
            up += qx_qwen_mixed_weight(
                gate_data, gate_scales, up_row, column, gate_groups, codec) * value;
        }
    }
    half gate_half = half(gate);
    half up_half = half(up);
    float gate_rounded = float(gate_half);
    activated[lane] = half((gate_rounded / (1.0f + exp(-gate_rounded))) * float(up_half));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = lane; row < hidden; row += intermediate) {
        float down = 0.0f;
        if (codec == 2) {
            constexpr uint bytes_per_group = 8;
            for (uint group = 0; group < down_groups; ++group) {
                uint base = (row * down_groups + group) * bytes_per_group;
                float scale = float(down_scales[row * down_groups + group]);
                uint group_column = group * group_size;
                for (uint packet = 0; packet < bytes_per_group; ++packet) {
                    uint codes = uint(down_data[base + packet]);
                    uint packet_column = group_column + packet * 4;
                    for (uint element = 0; element < 4; ++element) {
                        uint code = (codes >> (element * 2)) & 0x3;
                        down += qx_q2s4_decode(code, scale)
                            * float(activated[packet_column + element]);
                    }
                }
            }
        } else {
            for (uint column = 0; column < intermediate; ++column) {
                down += qx_qwen_mixed_weight(
                    down_data, down_scales, row, column, down_groups, codec)
                    * float(activated[column]);
            }
        }
        expert_outputs[slot * hidden + row] = half(down);
    }
}

kernel void qx_qwen_moe_gate_dynamic(
    device const uchar* slab [[buffer(0)]],
    device const ulong* gate_data_offsets [[buffer(1)]],
    device const ulong* gate_scale_offsets [[buffer(2)]],
    device const uchar* codecs [[buffer(3)]],
    device const half* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device half* activations [[buffer(6)]],
    uint lane [[thread_position_in_threadgroup]],
    uint slot [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 2048;
    constexpr uint intermediate = 768;
    constexpr uint group_size = 32;
    constexpr uint groups = hidden / group_size;
    uint expert = expert_ids[slot];
    uchar codec = codecs[expert];
    device const uchar* gate_data = slab + gate_data_offsets[expert];
    device const half* gate_scales =
        reinterpret_cast<device const half*>(slab + gate_scale_offsets[expert]);
    uint up_row = intermediate + lane;
    float gate = 0.0f;
    float up = 0.0f;
    if (codec == 2) {
        constexpr uint bytes_per_group = 8;
        for (uint group = 0; group < groups; ++group) {
            uint gate_base = (lane * groups + group) * bytes_per_group;
            uint up_base = (up_row * groups + group) * bytes_per_group;
            float gate_scale = float(gate_scales[lane * groups + group]);
            float up_scale = float(gate_scales[up_row * groups + group]);
            uint group_column = group * group_size;
            for (uint packet = 0; packet < bytes_per_group; ++packet) {
                uint gate_packet = uint(gate_data[gate_base + packet]);
                uint up_packet = uint(gate_data[up_base + packet]);
                uint packet_column = group_column + packet * 4;
                for (uint element = 0; element < 4; ++element) {
                    uint shift = element * 2;
                    float value = float(input[packet_column + element]);
                    gate += qx_q2s4_decode((gate_packet >> shift) & 0x3, gate_scale) * value;
                    up += qx_q2s4_decode((up_packet >> shift) & 0x3, up_scale) * value;
                }
            }
        }
    } else if (codec == 3) {
        for (uint column = 0; column < hidden; ++column) {
            float value = float(input[column]);
            gate += qx_qwen_mixed_weight(gate_data, gate_scales, lane, column, groups, codec)
                * value;
            up += qx_qwen_mixed_weight(gate_data, gate_scales, up_row, column, groups, codec)
                * value;
        }
    } else {
        gate = qx_qwen_q4_dot_device(gate_data, gate_scales, lane, groups, input);
        up = qx_qwen_q4_dot_device(gate_data, gate_scales, up_row, groups, input);
    }
    half gate_half = half(gate);
    half up_half = half(up);
    float gate_rounded = float(gate_half);
    activations[slot * intermediate + lane] =
        half((gate_rounded / (1.0f + exp(-gate_rounded))) * float(up_half));
}

kernel void qx_qwen_moe_down_tiled_dynamic(
    device const uchar* slab [[buffer(0)]],
    device const ulong* down_data_offsets [[buffer(1)]],
    device const ulong* down_scale_offsets [[buffer(2)]],
    device const uchar* codecs [[buffer(3)]],
    device const half* activations [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device half* expert_outputs [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 2048;
    constexpr uint intermediate = 768;
    constexpr uint group_size = 32;
    constexpr uint groups = intermediate / group_size;
    constexpr uint rows_per_tile = 256;
    constexpr uint tiles_per_expert = hidden / rows_per_tile;
    uint slot = group_id / tiles_per_expert;
    uint tile = group_id % tiles_per_expert;
    uint row = tile * rows_per_tile + tid;
    uint expert = expert_ids[slot];
    uchar codec = codecs[expert];
    device const uchar* down_data = slab + down_data_offsets[expert];
    device const half* down_scales =
        reinterpret_cast<device const half*>(slab + down_scale_offsets[expert]);
    device const half* activated = activations + slot * intermediate;
    float down = 0.0f;
    if (codec == 2) {
        constexpr uint bytes_per_group = 8;
        for (uint group = 0; group < groups; ++group) {
            uint base = (row * groups + group) * bytes_per_group;
            float scale = float(down_scales[row * groups + group]);
            uint group_column = group * group_size;
            for (uint packet = 0; packet < bytes_per_group; ++packet) {
                uint codes = uint(down_data[base + packet]);
                uint packet_column = group_column + packet * 4;
                for (uint element = 0; element < 4; ++element) {
                    uint code = (codes >> (element * 2)) & 0x3;
                    down += qx_q2s4_decode(code, scale)
                        * float(activated[packet_column + element]);
                }
            }
        }
    } else if (codec == 3) {
        for (uint column = 0; column < intermediate; ++column) {
            down += qx_qwen_mixed_weight(down_data, down_scales, row, column, groups, codec)
                * float(activated[column]);
        }
    } else {
        down = qx_qwen_q4_dot_device(down_data, down_scales, row, groups, activated);
    }
    expert_outputs[slot * hidden + row] = half(down);
}

kernel void qx_qwen_moe_accumulate_dynamic(
    device const half* expert_outputs [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    constexpr uint hidden = 2048;
    if (row >= hidden) {
        return;
    }
    float value = 0.0f;
    for (uint slot = 0; slot < 8; ++slot) {
        value += weights[slot] * float(expert_outputs[slot * hidden + row]);
    }
    output[row] = half(value);
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
