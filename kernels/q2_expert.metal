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
        activated[row] = half((gate / (1.0f + qx_consensus_exp(-gate))) * up);
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
        activated[row] = half((gate / (1.0f + qx_consensus_exp(-gate))) * up);
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
    activated[lane] = half((gate_rounded / (1.0f + qx_consensus_exp(-gate_rounded))) * float(up_half));
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

inline float qx_qwen_q3_dot8(uint packet, float4 x0, float4 x1, float scale) {
    uint4 codes0 = (uint4(packet) >> uint4(0u, 3u, 6u, 9u)) & 0x7u;
    uint4 codes1 = (uint4(packet) >> uint4(12u, 15u, 18u, 21u)) & 0x7u;
    float4 weights0 = float4(int4(codes0) - 3) * scale;
    float4 weights1 = float4(int4(codes1) - 3) * scale;
    return dot(weights0, x0) + dot(weights1, x1);
}

inline uint qx_qwen_q3_packet24(device const uchar* stream, uint offset) {
    packed_uchar4 bytes = *reinterpret_cast<device const packed_uchar4*>(stream + offset);
    return as_type<uint>(bytes) & 0x00ffffffu;
}

inline float qx_qwen_q3_112_dot_device(
    device const uchar* stream,
    uint row,
    uint blocks,
    device const half* input) {
    constexpr uint block_values = 256;
    constexpr uint block_bytes = 112;
    constexpr uint code_bytes = 96;
    float sum = 0.0f;
    uint row_base = row * blocks * block_bytes;
    for (uint block = 0; block < blocks; ++block) {
        uint block_base = row_base + block * block_bytes;
        device const half* scales =
            reinterpret_cast<device const half*>(stream + block_base + code_bytes);
        for (uint group = 0; group < 8; ++group) {
            float scale = float(scales[group]);
            uint group_column = block * block_values + group * 32;
            uint group_base = block_base + group * 12;
            for (uint packet_index = 0; packet_index < 4; ++packet_index) {
                uint column = group_column + packet_index * 8;
                float4 x0 = float4(
                    input[column], input[column + 1], input[column + 2], input[column + 3]);
                float4 x1 = float4(input[column + 4], input[column + 5],
                    input[column + 6], input[column + 7]);
                uint packet = qx_qwen_q3_packet24(stream, group_base + packet_index * 3);
                sum += qx_qwen_q3_dot8(packet, x0, x1, scale);
            }
        }
    }
    return sum;
}

inline float qx_qwen_q3_112_dot_threadgroup(
    device const uchar* stream,
    uint row,
    uint blocks,
    threadgroup const half* input) {
    constexpr uint block_values = 256;
    constexpr uint block_bytes = 112;
    constexpr uint code_bytes = 96;
    float sum = 0.0f;
    uint row_base = row * blocks * block_bytes;
    for (uint block = 0; block < blocks; ++block) {
        uint block_base = row_base + block * block_bytes;
        device const half* scales =
            reinterpret_cast<device const half*>(stream + block_base + code_bytes);
        for (uint group = 0; group < 8; ++group) {
            float scale = float(scales[group]);
            uint group_column = block * block_values + group * 32;
            uint group_base = block_base + group * 12;
            for (uint packet_index = 0; packet_index < 4; ++packet_index) {
                uint column = group_column + packet_index * 8;
                float4 x0 = float4(
                    input[column], input[column + 1], input[column + 2], input[column + 3]);
                float4 x1 = float4(input[column + 4], input[column + 5],
                    input[column + 6], input[column + 7]);
                uint packet = qx_qwen_q3_packet24(stream, group_base + packet_index * 3);
                sum += qx_qwen_q3_dot8(packet, x0, x1, scale);
            }
        }
    }
    return sum;
}

constant constexpr uint qx_qwen_q4_gate_scale_bytes = 196608u;
constant constexpr uint qx_qwen_q4_gate_data_bytes = 1572864u;
constant constexpr uint qx_qwen_q4_down_scale_bytes = 98304u;
constant constexpr uint qx_qwen_q4_down_data_bytes = 786432u;
constant constexpr uint qx_qwen_q4_cache_gate_scale_offset = 0u;
constant constexpr uint qx_qwen_q4_cache_gate_data_offset = 196608u;
constant constexpr uint qx_qwen_q4_cache_down_scale_offset = 1769472u;
constant constexpr uint qx_qwen_q4_cache_down_data_offset = 1867776u;
constant constexpr uint qx_qwen_q4_cache_slot_bytes = 2654208u;
constant constexpr uint qx_qwen_uncached_slot = 0xffffffffu;

// Resolve the selected source-exact Q4 experts against a small fully
// associative GPU cache. One thread owns the metadata update, so token graphs
// queued back-to-back preserve exact LRU semantics without atomics or host
// synchronization. One-hit entries bypass storage; only a second touch earns
// admission, preventing the sparse tail from polluting the bounded cache.
// Counters are [hits, misses, evictions, Q4 selections, bypasses].
kernel void qx_qwen_q4_cache_resolve(
    device const uchar* codecs [[buffer(0)]],
    device const uint* expert_ids [[buffer(1)]],
    device uint* keys [[buffer(2)]],
    device uint* ages [[buffer(3)]],
    device uint* clock [[buffer(4)]],
    device uint* resolved_slots [[buffer(5)]],
    device uint* copy_flags [[buffer(6)]],
    device uint* counters [[buffer(7)]],
    device uchar* seen [[buffer(8)]],
    constant uint& layer [[buffer(9)]],
    constant uint& cache_slots [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) {
        return;
    }
    uint now = clock[0] + 1u;
    if (now == 0u) {
        for (uint slot = 0; slot < cache_slots; ++slot) {
            ages[slot] = 0u;
        }
        now = 1u;
    }
    clock[0] = now;
    for (uint route_slot = 0; route_slot < 8u; ++route_slot) {
        uint expert = expert_ids[route_slot];
        resolved_slots[route_slot] = qx_qwen_uncached_slot;
        copy_flags[route_slot] = 0u;
        if (codecs[expert] != 4u) {
            continue;
        }
        counters[3] += 1u;
        uint key = layer * 128u + expert;
        uint hit_slot = qx_qwen_uncached_slot;
        uint empty_slot = qx_qwen_uncached_slot;
        uint oldest_slot = 0u;
        uint oldest_age = 0xffffffffu;
        for (uint slot = 0; slot < cache_slots; ++slot) {
            uint cached_key = keys[slot];
            if (cached_key == key) {
                hit_slot = slot;
                break;
            }
            if (cached_key == qx_qwen_uncached_slot && empty_slot == qx_qwen_uncached_slot) {
                empty_slot = slot;
            }
            if (ages[slot] < oldest_age) {
                oldest_age = ages[slot];
                oldest_slot = slot;
            }
        }
        if (hit_slot != qx_qwen_uncached_slot) {
            counters[0] += 1u;
            ages[hit_slot] = now;
            resolved_slots[route_slot] = hit_slot;
            continue;
        }
        counters[1] += 1u;
        if (seen[key] == 0u) {
            seen[key] = 1u;
            counters[4] += 1u;
            continue;
        }
        uint target = empty_slot != qx_qwen_uncached_slot ? empty_slot : oldest_slot;
        if (empty_slot == qx_qwen_uncached_slot) {
            counters[2] += 1u;
        }
        keys[target] = key;
        ages[target] = now;
        resolved_slots[route_slot] = target;
        copy_flags[route_slot] = 1u;
    }
}

inline void qx_qwen_copy_words(
    device const uchar* source,
    device uchar* destination,
    uint bytes,
    uint tid) {
    device const uint* source_words = reinterpret_cast<device const uint*>(source);
    device uint* destination_words = reinterpret_cast<device uint*>(destination);
    for (uint word = tid; word < bytes / 4u; word += 256u) {
        destination_words[word] = source_words[word];
    }
}

// Eight fixed threadgroups avoid a route readback or indirect dispatch. Cache
// hits exit immediately; a miss copies the source expert's four physical
// ranges into one consumer-native fixed slot.
kernel void qx_qwen_q4_cache_fill(
    device const uchar* slab [[buffer(0)]],
    device const ulong* gate_data_offsets [[buffer(1)]],
    device const ulong* gate_scale_offsets [[buffer(2)]],
    device const ulong* down_data_offsets [[buffer(3)]],
    device const ulong* down_scale_offsets [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device const uint* resolved_slots [[buffer(6)]],
    device const uint* copy_flags [[buffer(7)]],
    device uchar* cache [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint route_slot [[threadgroup_position_in_grid]]) {
    if (copy_flags[route_slot] == 0u) {
        return;
    }
    uint expert = expert_ids[route_slot];
    uint cache_slot = resolved_slots[route_slot];
    device uchar* target = cache + cache_slot * qx_qwen_q4_cache_slot_bytes;
    qx_qwen_copy_words(slab + gate_scale_offsets[expert],
        target + qx_qwen_q4_cache_gate_scale_offset, qx_qwen_q4_gate_scale_bytes, tid);
    qx_qwen_copy_words(slab + gate_data_offsets[expert],
        target + qx_qwen_q4_cache_gate_data_offset, qx_qwen_q4_gate_data_bytes, tid);
    qx_qwen_copy_words(slab + down_scale_offsets[expert],
        target + qx_qwen_q4_cache_down_scale_offset, qx_qwen_q4_down_scale_bytes, tid);
    qx_qwen_copy_words(slab + down_data_offsets[expert],
        target + qx_qwen_q4_cache_down_data_offset, qx_qwen_q4_down_data_bytes, tid);
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
    device const uchar* cache [[buffer(9)]],
    device const uint* resolved_slots [[buffer(10)]],
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
    uint cache_slot = resolved_slots[slot];
    if (codec == 4u && cache_slot != qx_qwen_uncached_slot) {
        device const uchar* cached = cache + cache_slot * qx_qwen_q4_cache_slot_bytes;
        gate_data = cached + qx_qwen_q4_cache_gate_data_offset;
        gate_scales = reinterpret_cast<device const half*>(
            cached + qx_qwen_q4_cache_gate_scale_offset);
        down_data = cached + qx_qwen_q4_cache_down_data_offset;
        down_scales = reinterpret_cast<device const half*>(
            cached + qx_qwen_q4_cache_down_scale_offset);
    }
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
    } else if (codec == 5) {
        gate = qx_qwen_q3_112_dot_device(gate_data, lane, gate_groups / 8, input);
        up = qx_qwen_q3_112_dot_device(gate_data, up_row, gate_groups / 8, input);
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
    activated[lane] = half((gate_rounded / (1.0f + qx_consensus_exp(-gate_rounded))) * float(up_half));
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
        } else if (codec == 5) {
            down = qx_qwen_q3_112_dot_threadgroup(
                down_data, row, down_groups / 8, activated);
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
    device const uchar* cache [[buffer(7)]],
    device const uint* resolved_slots [[buffer(8)]],
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
    uint cache_slot = resolved_slots[slot];
    if (codec == 4u && cache_slot != qx_qwen_uncached_slot) {
        device const uchar* cached = cache + cache_slot * qx_qwen_q4_cache_slot_bytes;
        gate_data = cached + qx_qwen_q4_cache_gate_data_offset;
        gate_scales = reinterpret_cast<device const half*>(
            cached + qx_qwen_q4_cache_gate_scale_offset);
    }
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
    } else if (codec == 5) {
        gate = qx_qwen_q3_112_dot_device(gate_data, lane, groups / 8, input);
        up = qx_qwen_q3_112_dot_device(gate_data, up_row, groups / 8, input);
    } else {
        gate = qx_qwen_q4_dot_device(gate_data, gate_scales, lane, groups, input);
        up = qx_qwen_q4_dot_device(gate_data, gate_scales, up_row, groups, input);
    }
    half gate_half = half(gate);
    half up_half = half(up);
    float gate_rounded = float(gate_half);
    activations[slot * intermediate + lane] =
        half((gate_rounded / (1.0f + qx_consensus_exp(-gate_rounded))) * float(up_half));
}

kernel void qx_qwen_moe_down_tiled_dynamic(
    device const uchar* slab [[buffer(0)]],
    device const ulong* down_data_offsets [[buffer(1)]],
    device const ulong* down_scale_offsets [[buffer(2)]],
    device const uchar* codecs [[buffer(3)]],
    device const half* activations [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device half* expert_outputs [[buffer(6)]],
    device const uchar* cache [[buffer(7)]],
    device const uint* resolved_slots [[buffer(8)]],
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
    uint cache_slot = resolved_slots[slot];
    if (codec == 4u && cache_slot != qx_qwen_uncached_slot) {
        device const uchar* cached = cache + cache_slot * qx_qwen_q4_cache_slot_bytes;
        down_data = cached + qx_qwen_q4_cache_down_data_offset;
        down_scales = reinterpret_cast<device const half*>(
            cached + qx_qwen_q4_cache_down_scale_offset);
    }
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
    } else if (codec == 5) {
        down = qx_qwen_q3_112_dot_device(down_data, row, groups / 8, activated);
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
