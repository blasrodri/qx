#include <metal_stdlib>
using namespace metal;

inline uint qx_dynamic_group_offset(uint row, uint group, uint groups, uint bytes_per_group, uint layout) {
    if (layout == 0) {
        return row * groups * bytes_per_group + group * bytes_per_group;
    }
    if (layout == 7 || layout == 8) {
        // Transaction streams contain eight [16-byte codes + FP16 scale]
        // groups per block. Q4Transaction160 adds a sixteen-byte alignment
        // tail after each block; the tail is never consumed.
        uint block_bytes = layout == 8 ? 160 : 144;
        return row * (groups / 8) * block_bytes
            + (group / 8) * block_bytes
            + (group % 8) * 18;
    }
    // Paired gate/up layout interleaves logical gate row r and up row r.
    // Down matrices use the ordinary tile layout even when the enclosing
    // MoE preparation carries layout=2, hence the groups==32 guard.
    uint physical_row = row;
    if (layout == 2 && groups == 32) {
        physical_row = row < 512 ? row * 2 : (row - 512) * 2 + 1;
    }
    uint tile = physical_row / 32;
    uint lane = physical_row % 32;
    return tile * 32 * groups * bytes_per_group
        + group * 32 * bytes_per_group
        + lane * bytes_per_group;
}

inline uint qx_dynamic_scale_offset(uint row, uint group, uint groups, uint layout) {
    if (layout == 0) {
        return row * groups + group;
    }
    if (layout == 7 || layout == 8) {
        return qx_dynamic_group_offset(row, group, groups, 16, layout) / 2 + 8;
    }
    uint physical_row = row;
    if (layout == 2 && groups == 32) {
        physical_row = row < 512 ? row * 2 : (row - 512) * 2 + 1;
    }
    return (physical_row / 32) * 32 * groups + group * 32 + physical_row % 32;
}

inline uint qx_dynamic_packed_stride(uint rows, uint groups, uint layout) {
    return qx_dynamic_group_offset(rows, 0, groups, 16, layout);
}

inline uint qx_dynamic_scale_stride(uint rows, uint groups, uint layout) {
    if (layout == 7 || layout == 8) {
        return qx_dynamic_group_offset(rows, 0, groups, 16, layout) / 2;
    }
    return rows * groups;
}

inline float qx_q4_inline_scale(device const uchar* stream, uint offset) {
    uint bits = uint(stream[offset]) | (uint(stream[offset + 1]) << 8);
    return float(as_type<half>(ushort(bits)));
}

// Packetized exact-Q4 row-inline body. Unlike the original row-inline stream,
// each input group stores the selected rows together:
//   row 0 codes+scale, row 1 codes+scale, ...
// The total bytes are unchanged. The packet width is a compile-time plan
// choice exposed through separate entry points so the compiler can benchmark
// the representation without passing a per-dispatch control buffer.
inline void qx_dynamic_q4_gate_up_inline_packet_body(
    device const uchar* stream,
    device const uint* expert_ids,
    device float* output,
    threadgroup float* cached_input,
    uint packet_rows,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32 && row < hidden) {
        uint packet = row / packet_rows;
        uint row_in_packet = row % packet_rows;
        uint packet_stride = groups * packet_rows * group_stride;
        uint row_base = expert * expert_stride + packet * packet_stride
            + row_in_packet * group_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * packet_rows * group_stride;
            uint code = (uint(stream[group_base + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7)
                * qx_q4_inline_scale(stream, group_base + 16)
                * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// K-wide verifier primitive. The route compactor presents one expert/token
// pair per entry, so the GPU can consume the selected experts in expert-major
// order without re-running the scalar eight-expert dispatch. Each threadgroup
// owns one route pair and eight output rows; the input row is staged once for
// the token referenced by that pair. This is intentionally gate/up only: it
// is the exact first stage of the eventual K-wide MoE path.
kernel void qx_verifier_q4_moe_gate_up_batch(
    device const uchar* stream [[buffer(0)]],
    device const uint* route_experts [[buffer(1)]],
    device const uint* route_tokens [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& pair_count [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_pair = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];

    uint pair = threadgroup_id / threadgroups_per_pair;
    uint row_group = threadgroup_id % threadgroups_per_pair;
    if (pair >= pair_count) return;
    uint token = route_tokens[pair];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[token * hidden + column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    uint expert = route_experts[pair];
    float sum = 0.0f;
    if (expert < 32 && row < hidden) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2])
                >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7)
                * qx_q4_inline_scale(stream, group_base + 16)
                * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[pair * hidden + row] = sum;
}

// Expert-major K-wide variant. `bucket_offsets` partitions `bucket_pairs`
// by expert; for the K=4 verifier every bucket contains at most four
// token/route pairs. One threadgroup therefore stages up to four activation
// rows, walks each packed weight packet once, and updates all active token
// outputs from that shared weight consumption.
kernel void qx_verifier_q4_moe_gate_up_expert_major(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_experts [[buffer(1)]],
    device const uint* bucket_offsets [[buffer(2)]],
    device const uint* bucket_pairs [[buffer(3)]],
    device const uint* route_tokens [[buffer(4)]],
    device const float* input [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& bucket_count [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_input[max_tokens * hidden];

    uint bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (bucket >= bucket_count) return;
    uint begin = bucket_offsets[bucket];
    uint end = bucket_offsets[bucket + 1];
    uint active = min(end - begin, max_tokens);
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[begin + token_slot];
            uint token = route_tokens[pair];
            for (uint column = thread_index; column < hidden; column += 256) {
                cached_input[token_slot * hidden + column] = input[token * hidden + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    uint expert = bucket_experts[bucket];
    float sums[max_tokens] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (expert < 32 && row < hidden) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2])
                >> ((local % 2) * 4)) & 0xf;
            float weight = float(int(code) - 7)
                * qx_q4_inline_scale(stream, group_base + 16);
            for (uint token_slot = 0; token_slot < active; ++token_slot) {
                sums[token_slot] += weight
                    * cached_input[token_slot * hidden + column];
            }
        }
    }
    for (uint token_slot = 0; token_slot < active; ++token_slot) {
        sums[token_slot] = simd_sum(sums[token_slot]);
        if (lane == 0) {
            uint pair = bucket_pairs[begin + token_slot];
            output[pair * hidden + row] = sums[token_slot];
        }
    }
}

// Same expert-major work unit without the 4 x 1024 threadgroup staging.
// This is a controlled activation-access variant: packed weight ownership,
// bucket order, output ownership, and arithmetic are unchanged.
kernel void qx_verifier_q4_moe_gate_up_expert_major_direct(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_experts [[buffer(1)]],
    device const uint* bucket_offsets [[buffer(2)]],
    device const uint* bucket_pairs [[buffer(3)]],
    device const uint* route_tokens [[buffer(4)]],
    device const float* input [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& bucket_count [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;

    uint bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (bucket >= bucket_count) return;
    uint begin = bucket_offsets[bucket];
    uint end = bucket_offsets[bucket + 1];
    uint active = min(end - begin, max_tokens);
    uint row = row_group * rows_per_threadgroup + simdgroup;
    uint expert = bucket_experts[bucket];
    float sums[max_tokens] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (expert < 32 && row < hidden) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2])
                >> ((local % 2) * 4)) & 0xf;
            float weight = float(int(code) - 7)
                * qx_q4_inline_scale(stream, group_base + 16);
            for (uint token_slot = 0; token_slot < active; ++token_slot) {
                uint pair = bucket_pairs[begin + token_slot];
                uint token = route_tokens[pair];
                sums[token_slot] += weight * input[token * hidden + column];
            }
        }
    }
    for (uint token_slot = 0; token_slot < active; ++token_slot) {
        sums[token_slot] = simd_sum(sums[token_slot]);
        if (lane == 0) {
            uint pair = bucket_pairs[begin + token_slot];
            output[pair * hidden + row] = sums[token_slot];
        }
    }
}

// Expert-major packet-stationary candidate. This keeps the expert-major
// bucket schedule, but uses the same two-group packet ownership as the
// promoted single-token unique-packet consumer: one packed code byte and one
// broadcast scale serve both nibbles, then the decoded weight is reused across
// every token pair in the bucket.
kernel void qx_verifier_q4_moe_gate_up_expert_major_unique_packet(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_experts [[buffer(1)]],
    device const uint* bucket_offsets [[buffer(2)]],
    device const uint* bucket_pairs [[buffer(3)]],
    device const uint* route_tokens [[buffer(4)]],
    device const float* input [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& bucket_count [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_input[max_tokens * hidden];

    uint bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (bucket >= bucket_count) return;
    uint begin = bucket_offsets[bucket];
    uint end = bucket_offsets[bucket + 1];
    uint active = min(end - begin, max_tokens);
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[begin + token_slot];
            uint token = route_tokens[pair];
            for (uint column = thread_index; column < hidden; column += 256) {
                cached_input[token_slot * hidden + column] = input[token * hidden + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    uint expert = bucket_experts[bucket];
    float sums[max_tokens] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (expert < 32 && row < hidden) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) {
                scale = qx_q4_inline_scale(stream, group_base + 16u);
            }
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            float weight0 = float(int(packed & 0xfu) - 7) * selected_scale;
            float weight1 = float(int((packed >> 4u) & 0xfu) - 7) * selected_scale;
            for (uint token_slot = 0; token_slot < active; ++token_slot) {
                uint token_base = token_slot * hidden;
                sums[token_slot] += weight0 * cached_input[token_base + input_base]
                    + weight1 * cached_input[token_base + input_base + 1u];
            }
        }
    }
    for (uint token_slot = 0; token_slot < active; ++token_slot) {
        sums[token_slot] = simd_sum(sums[token_slot]);
        if (lane == 0) {
            uint pair = bucket_pairs[begin + token_slot];
            output[pair * hidden + row] = sums[token_slot];
        }
    }
}

// GPU route compaction for the K-wide verifier. Each expert owns one thread
// and writes its selected pair indices into a fixed-size bucket. Fixed-stride
// buckets avoid an atomic prefix-sum and keep the route schedule GPU-resident;
// at K<=4 an expert can have at most four token pairs.
kernel void qx_verifier_compact_routes_fixed(
    device const uint* route_experts [[buffer(0)]],
    device uint* bucket_counts [[buffer(1)]],
    device uint* bucket_pairs [[buffer(2)]],
    constant uint& pair_count [[buffer(3)]],
    device uint* active_experts [[buffer(4)]],
    device uint* dispatch_args [[buffer(5)]],
    uint expert [[thread_position_in_grid]]) {
    if (expert >= 32) return;
    uint count = 0;
    for (uint pair = 0; pair < pair_count; ++pair) {
        if (route_experts[pair] == expert) {
            bucket_pairs[expert * pair_count + count] = pair;
            ++count;
        }
    }
    bucket_counts[expert] = count;
    threadgroup_barrier(mem_flags::mem_device);
    if (count > 0) {
        uint rank = 0;
        for (uint prior = 0; prior < expert; ++prior) {
            if (bucket_counts[prior] > 0) ++rank;
        }
        active_experts[rank] = expert;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (expert == 0) {
        uint active = 0;
        for (uint e = 0; e < 32; ++e) {
            if (bucket_counts[e] > 0) ++active;
        }
        // MTLDispatchThreadgroupsIndirectArguments: three uint32 values.
        dispatch_args[0] = active * (1024 / 8);
        dispatch_args[1] = 1;
        dispatch_args[2] = 1;
        dispatch_args[3] = active * (1024 / 16);
        dispatch_args[4] = 1;
        dispatch_args[5] = 1;
    }
}

// Packet-stationary consumer for the fixed-stride route buckets emitted by
// qx_verifier_compact_routes_fixed. The expert ID is the bucket index and the
// bucket count determines the active token slots.
kernel void qx_verifier_q4_moe_gate_up_compact_unique_packet(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_counts [[buffer(1)]],
    device const uint* bucket_pairs [[buffer(2)]],
    device const uint* route_tokens [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& pair_count [[buffer(6)]],
    device const uint* active_experts [[buffer(7)]],
    constant uint& active_mode [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_input[max_tokens * hidden];

    uint active_bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (active_bucket >= 32) return;
    uint bucket = active_mode != 0 ? active_experts[active_bucket] : active_bucket;
    uint active = min(bucket_counts[bucket], max_tokens);
    uint bucket_base = bucket * pair_count;
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            uint token = route_tokens[pair];
            for (uint column = thread_index; column < hidden; column += 256) {
                cached_input[token_slot * hidden + column] = input[token * hidden + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    float sums[max_tokens] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active > 0) {
        uint row_base = bucket * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) scale = qx_q4_inline_scale(stream, group_base + 16u);
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            float weight0 = float(int(packed & 0xfu) - 7) * selected_scale;
            float weight1 = float(int((packed >> 4u) & 0xfu) - 7) * selected_scale;
            for (uint token_slot = 0; token_slot < active; ++token_slot) {
                uint token_base = token_slot * hidden;
                sums[token_slot] += weight0 * cached_input[token_base + input_base]
                    + weight1 * cached_input[token_base + input_base + 1u];
            }
        }
    }
    for (uint token_slot = 0; token_slot < active; ++token_slot) {
        sums[token_slot] = simd_sum(sums[token_slot]);
        if (lane == 0) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            output[pair * hidden + row] = sums[token_slot];
        }
    }
}

kernel void qx_verifier_q4_moe_gate_up_compact_unique_packet_k2(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_counts [[buffer(1)]],
    device const uint* bucket_pairs [[buffer(2)]],
    device const uint* route_tokens [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& pair_count [[buffer(6)]],
    device const uint* active_experts [[buffer(7)]],
    constant uint& active_mode [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 2;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_input[max_tokens * hidden];

    uint active_bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (active_bucket >= 32) return;
    uint bucket = active_mode != 0 ? active_experts[active_bucket] : active_bucket;
    uint active = min(bucket_counts[bucket], max_tokens);
    uint bucket_base = bucket * pair_count;
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            uint token = route_tokens[pair];
            for (uint column = thread_index; column < hidden; column += 256) {
                cached_input[token_slot * hidden + column] = input[token * hidden + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    float sums[max_tokens] = {0.0f, 0.0f};
    if (active > 0) {
        uint row_base = bucket * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) scale = qx_q4_inline_scale(stream, group_base + 16u);
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            float weight0 = float(int(packed & 0xfu) - 7) * selected_scale;
            float weight1 = float(int((packed >> 4u) & 0xfu) - 7) * selected_scale;
            for (uint token_slot = 0; token_slot < active; ++token_slot) {
                uint token_base = token_slot * hidden;
                sums[token_slot] += weight0 * cached_input[token_base + input_base]
                    + weight1 * cached_input[token_base + input_base + 1u];
            }
        }
    }
    for (uint token_slot = 0; token_slot < active; ++token_slot) {
        sums[token_slot] = simd_sum(sums[token_slot]);
        if (lane == 0) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            output[pair * hidden + row] = sums[token_slot];
        }
    }
}

// K4 packet consumer. The decoded packet is shared across a float4 of token
// activations so the K dimension is explicit in the dataflow. Production
// selection is carried by the installed verifier execution plan; the legacy
// environment flag remains available only to standalone diagnostics.
kernel void qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_vector(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_counts [[buffer(1)]],
    device const uint* bucket_pairs [[buffer(2)]],
    device const uint* route_tokens [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& pair_count [[buffer(6)]],
    device const uint* active_experts [[buffer(7)]],
    constant uint& active_mode [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_input[max_tokens * hidden];

    uint active_bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (active_bucket >= 32) return;
    uint bucket = active_mode != 0 ? active_experts[active_bucket] : active_bucket;
    uint active = min(bucket_counts[bucket], max_tokens);
    uint bucket_base = bucket * pair_count;
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            uint token = route_tokens[pair];
            for (uint column = thread_index; column < hidden; column += 256) {
                cached_input[token_slot * hidden + column] = input[token * hidden + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    float4 sums = float4(0.0f);
    if (active > 0) {
        uint row_base = bucket * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) scale = qx_q4_inline_scale(stream, group_base + 16u);
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            float weight0 = float(int(packed & 0xfu) - 7) * selected_scale;
            float weight1 = float(int((packed >> 4u) & 0xfu) - 7) * selected_scale;
            float4 x0 = float4(0.0f);
            float4 x1 = float4(0.0f);
            if (active > 0) {
                x0.x = cached_input[input_base];
                x1.x = cached_input[input_base + 1u];
            }
            if (active > 1) {
                x0.y = cached_input[hidden + input_base];
                x1.y = cached_input[hidden + input_base + 1u];
            }
            if (active > 2) {
                x0.z = cached_input[2u * hidden + input_base];
                x1.z = cached_input[2u * hidden + input_base + 1u];
            }
            if (active > 3) {
                x0.w = cached_input[3u * hidden + input_base];
                x1.w = cached_input[3u * hidden + input_base + 1u];
            }
            sums += x0 * weight0 + x1 * weight1;
        }
    }
    if (active > 0) sums.x = simd_sum(sums.x);
    if (active > 1) sums.y = simd_sum(sums.y);
    if (active > 2) sums.z = simd_sum(sums.z);
    if (active > 3) sums.w = simd_sum(sums.w);
    if (lane == 0) {
        if (active > 0) output[bucket_pairs[bucket_base] * hidden + row] = sums.x;
        if (active > 1) output[bucket_pairs[bucket_base + 1u] * hidden + row] = sums.y;
        if (active > 2) output[bucket_pairs[bucket_base + 2u] * hidden + row] = sums.z;
        if (active > 3) output[bucket_pairs[bucket_base + 3u] * hidden + row] = sums.w;
    }
}

// K4 diagnostic consumer.  It preserves packet reuse within pairs of tokens,
// but only keeps two accumulators live at a time.  This is intentionally not a
// production default: it reloads each weight packet for the second token pair
// so that a benchmark can separate accumulator/register pressure from the
// benefit of keeping all four K4 accumulators live.
kernel void qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_split_accum(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_counts [[buffer(1)]],
    device const uint* bucket_pairs [[buffer(2)]],
    device const uint* route_tokens [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& pair_count [[buffer(6)]],
    device const uint* active_experts [[buffer(7)]],
    constant uint& active_mode [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_input[max_tokens * hidden];

    uint active_bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (active_bucket >= 32) return;
    uint bucket = active_mode != 0 ? active_experts[active_bucket] : active_bucket;
    uint active = min(bucket_counts[bucket], max_tokens);
    uint bucket_base = bucket * pair_count;
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            uint token = route_tokens[pair];
            for (uint column = thread_index; column < hidden; column += 256) {
                cached_input[token_slot * hidden + column] = input[token * hidden + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    if (active == 0 || row >= hidden) return;
    uint row_base = bucket * expert_stride + row * row_stride;
    for (uint pass = 0; pass < 2; ++pass) {
        uint token0 = pass * 2;
        uint token1 = token0 + 1;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) scale = qx_q4_inline_scale(stream, group_base + 16u);
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            float weight0 = float(int(packed & 0xfu) - 7) * selected_scale;
            float weight1 = float(int((packed >> 4u) & 0xfu) - 7) * selected_scale;
            if (token0 < active) {
                sum0 += weight0 * cached_input[token0 * hidden + input_base]
                    + weight1 * cached_input[token0 * hidden + input_base + 1u];
            }
            if (token1 < active) {
                sum1 += weight0 * cached_input[token1 * hidden + input_base]
                    + weight1 * cached_input[token1 * hidden + input_base + 1u];
            }
        }
        if (token0 < active) sum0 = simd_sum(sum0);
        if (token1 < active) sum1 = simd_sum(sum1);
        if (lane == 0) {
            if (token0 < active) {
                output[bucket_pairs[bucket_base + token0] * hidden + row] = sum0;
            }
            if (token1 < active) {
                output[bucket_pairs[bucket_base + token1] * hidden + row] = sum1;
            }
        }
    }
}

// K4 packet consumer without threadgroup activation staging. Packet decode
// and four-token reuse remain identical in spirit to the production K4 path;
// only activation placement changes. This isolates staging/barrier cost from
// packet reuse and is intentionally diagnostic-only.
kernel void qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_direct_activation(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_counts [[buffer(1)]],
    device const uint* bucket_pairs [[buffer(2)]],
    device const uint* route_tokens [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& pair_count [[buffer(6)]],
    device const uint* active_experts [[buffer(7)]],
    constant uint& active_mode [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;

    uint active_bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (active_bucket >= 32) return;
    uint bucket = active_mode != 0 ? active_experts[active_bucket] : active_bucket;
    uint active = min(bucket_counts[bucket], max_tokens);
    uint bucket_base = bucket * pair_count;
    uint row = row_group * rows_per_threadgroup + simdgroup;
    float4 sums = float4(0.0f);
    if (active > 0 && row < hidden) {
        uint row_base = bucket * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) scale = qx_q4_inline_scale(stream, group_base + 16u);
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            float weight0 = float(int(packed & 0xfu) - 7) * selected_scale;
            float weight1 = float(int((packed >> 4u) & 0xfu) - 7) * selected_scale;
            if (active > 0) {
                sums.x += weight0 * input[route_tokens[bucket_pairs[bucket_base]] * hidden + input_base]
                    + weight1 * input[route_tokens[bucket_pairs[bucket_base]] * hidden + input_base + 1u];
            }
            if (active > 1) {
                sums.y += weight0 * input[route_tokens[bucket_pairs[bucket_base + 1u]] * hidden + input_base]
                    + weight1 * input[route_tokens[bucket_pairs[bucket_base + 1u]] * hidden + input_base + 1u];
            }
            if (active > 2) {
                sums.z += weight0 * input[route_tokens[bucket_pairs[bucket_base + 2u]] * hidden + input_base]
                    + weight1 * input[route_tokens[bucket_pairs[bucket_base + 2u]] * hidden + input_base + 1u];
            }
            if (active > 3) {
                sums.w += weight0 * input[route_tokens[bucket_pairs[bucket_base + 3u]] * hidden + input_base]
                    + weight1 * input[route_tokens[bucket_pairs[bucket_base + 3u]] * hidden + input_base + 1u];
            }
        }
    }
    if (active > 0) sums.x = simd_sum(sums.x);
    if (active > 1) sums.y = simd_sum(sums.y);
    if (active > 2) sums.z = simd_sum(sums.z);
    if (active > 3) sums.w = simd_sum(sums.w);
    if (lane == 0) {
        if (active > 0) output[bucket_pairs[bucket_base] * hidden + row] = sums.x;
        if (active > 1) output[bucket_pairs[bucket_base + 1u] * hidden + row] = sums.y;
        if (active > 2) output[bucket_pairs[bucket_base + 2u] * hidden + row] = sums.z;
        if (active > 3) output[bucket_pairs[bucket_base + 3u] * hidden + row] = sums.w;
    }
}

// K4 packet consumer with sixteen output rows per threadgroup. This keeps
// four-token packet reuse and activation staging, but stages each routed input
// set across half as many row groups. It is a width-specific mapping probe.
kernel void qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_rows16(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_counts [[buffer(1)]],
    device const uint* bucket_pairs [[buffer(2)]],
    device const uint* route_tokens [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& pair_count [[buffer(6)]],
    device const uint* active_experts [[buffer(7)]],
    constant uint& active_mode [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 16;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_input[max_tokens * hidden];

    uint active_bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (active_bucket >= 32) return;
    uint bucket = active_mode != 0 ? active_experts[active_bucket] : active_bucket;
    uint active = min(bucket_counts[bucket], max_tokens);
    uint bucket_base = bucket * pair_count;
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            uint token = route_tokens[pair];
            for (uint column = thread_index; column < hidden; column += 512) {
                cached_input[token_slot * hidden + column] = input[token * hidden + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    float4 sums = float4(0.0f);
    if (active > 0 && row < hidden) {
        uint row_base = bucket * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) scale = qx_q4_inline_scale(stream, group_base + 16u);
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            float weight0 = float(int(packed & 0xfu) - 7) * selected_scale;
            float weight1 = float(int((packed >> 4u) & 0xfu) - 7) * selected_scale;
            if (active > 0) {
                uint pair = bucket_pairs[bucket_base];
                uint token = route_tokens[pair];
                sums.x += weight0 * cached_input[input_base]
                    + weight1 * cached_input[input_base + 1u];
                if (active > 1) {
                    sums.y += weight0 * cached_input[hidden + input_base]
                        + weight1 * cached_input[hidden + input_base + 1u];
                }
                if (active > 2) {
                    sums.z += weight0 * cached_input[2u * hidden + input_base]
                        + weight1 * cached_input[2u * hidden + input_base + 1u];
                }
                if (active > 3) {
                    sums.w += weight0 * cached_input[3u * hidden + input_base]
                        + weight1 * cached_input[3u * hidden + input_base + 1u];
                }
                (void)token;
            }
        }
    }
    if (active > 0) sums.x = simd_sum(sums.x);
    if (active > 1) sums.y = simd_sum(sums.y);
    if (active > 2) sums.z = simd_sum(sums.z);
    if (active > 3) sums.w = simd_sum(sums.w);
    if (lane == 0 && row < hidden) {
        if (active > 0) output[bucket_pairs[bucket_base] * hidden + row] = sums.x;
        if (active > 1) output[bucket_pairs[bucket_base + 1u] * hidden + row] = sums.y;
        if (active > 2) output[bucket_pairs[bucket_base + 2u] * hidden + row] = sums.z;
        if (active > 3) output[bucket_pairs[bucket_base + 3u] * hidden + row] = sums.w;
    }
}

// K4 packet consumer with two packet pairs in flight. This keeps the
// production K4 ownership, staging, and four accumulators, while exposing
// independent Q4 loads to the compiler. It is a load-issue diagnostic.
kernel void qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_unroll4(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_counts [[buffer(1)]],
    device const uint* bucket_pairs [[buffer(2)]],
    device const uint* route_tokens [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& pair_count [[buffer(6)]],
    device const uint* active_experts [[buffer(7)]],
    constant uint& active_mode [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_input[max_tokens * hidden];

    uint active_bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (active_bucket >= 32) return;
    uint bucket = active_mode != 0 ? active_experts[active_bucket] : active_bucket;
    uint active = min(bucket_counts[bucket], max_tokens);
    uint bucket_base = bucket * pair_count;
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            uint token = route_tokens[pair];
            for (uint column = thread_index; column < hidden; column += 256) {
                cached_input[token_slot * hidden + column] = input[token * hidden + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    float4 sums = float4(0.0f);
    if (active > 0 && row < hidden) {
        uint row_base = bucket * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 4) {
            uint local = lane & 15u;
            uint half_group = lane >= 16u ? 1u : 0u;
            uint selected0 = group + half_group;
            uint selected1 = group + 2u + half_group;
            uint base0 = row_base + selected0 * group_stride;
            uint base1 = row_base + selected1 * group_stride;
            uint packed0 = uint(stream[base0 + local]);
            uint packed1 = uint(stream[base1 + local]);
            float scale0 = 0.0f;
            float scale1 = 0.0f;
            if (local == 0u) {
                scale0 = qx_q4_inline_scale(stream, base0 + 16u);
                scale1 = qx_q4_inline_scale(stream, base1 + 16u);
            }
            float scale00 = simd_broadcast(lane == 0u ? scale0 : 0.0f, 0);
            float scale01 = simd_broadcast(lane == 16u ? scale0 : 0.0f, 16);
            float scale10 = simd_broadcast(lane == 0u ? scale1 : 0.0f, 0);
            float scale11 = simd_broadcast(lane == 16u ? scale1 : 0.0f, 16);
            float selected_scale0 = lane < 16u ? scale00 : scale01;
            float selected_scale1 = lane < 16u ? scale10 : scale11;
            uint input_base0 = selected0 * 32u + local * 2u;
            uint input_base1 = selected1 * 32u + local * 2u;
            float weight00 = float(int(packed0 & 0xfu) - 7) * selected_scale0;
            float weight01 = float(int((packed0 >> 4u) & 0xfu) - 7) * selected_scale0;
            float weight10 = float(int(packed1 & 0xfu) - 7) * selected_scale1;
            float weight11 = float(int((packed1 >> 4u) & 0xfu) - 7) * selected_scale1;
            if (active > 0) {
                sums.x += weight00 * cached_input[input_base0]
                    + weight01 * cached_input[input_base0 + 1u]
                    + weight10 * cached_input[input_base1]
                    + weight11 * cached_input[input_base1 + 1u];
            }
            if (active > 1) {
                sums.y += weight00 * cached_input[hidden + input_base0]
                    + weight01 * cached_input[hidden + input_base0 + 1u]
                    + weight10 * cached_input[hidden + input_base1]
                    + weight11 * cached_input[hidden + input_base1 + 1u];
            }
            if (active > 2) {
                sums.z += weight00 * cached_input[2u * hidden + input_base0]
                    + weight01 * cached_input[2u * hidden + input_base0 + 1u]
                    + weight10 * cached_input[2u * hidden + input_base1]
                    + weight11 * cached_input[2u * hidden + input_base1 + 1u];
            }
            if (active > 3) {
                sums.w += weight00 * cached_input[3u * hidden + input_base0]
                    + weight01 * cached_input[3u * hidden + input_base0 + 1u]
                    + weight10 * cached_input[3u * hidden + input_base1]
                    + weight11 * cached_input[3u * hidden + input_base1 + 1u];
            }
        }
    }
    if (active > 0) sums.x = simd_sum(sums.x);
    if (active > 1) sums.y = simd_sum(sums.y);
    if (active > 2) sums.z = simd_sum(sums.z);
    if (active > 3) sums.w = simd_sum(sums.w);
    if (lane == 0 && row < hidden) {
        if (active > 0) output[bucket_pairs[bucket_base] * hidden + row] = sums.x;
        if (active > 1) output[bucket_pairs[bucket_base + 1u] * hidden + row] = sums.y;
        if (active > 2) output[bucket_pairs[bucket_base + 2u] * hidden + row] = sums.z;
        if (active > 3) output[bucket_pairs[bucket_base + 3u] * hidden + row] = sums.w;
    }
}

kernel void qx_verifier_q4_moe_silu_batch(
    device const float* gate_up [[buffer(0)]],
    device float* activated [[buffer(1)]],
    constant uint& pair_count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint intermediate = 512;
    uint pair = tid / intermediate;
    uint row = tid % intermediate;
    if (pair >= pair_count) return;
    uint base = pair * 2 * intermediate;
    float gate = gate_up[base + row];
    float up = gate_up[base + intermediate + row];
    activated[pair * intermediate + row] = (gate / (1.0f + exp(-gate))) * up;
}

kernel void qx_verifier_q4_moe_down_batch(
    device const uchar* stream [[buffer(0)]],
    device const uint* route_experts [[buffer(1)]],
    device const float* activated [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& pair_count [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_pair = hidden / rows_per_threadgroup;
    uint pair = threadgroup_id / threadgroups_per_pair;
    uint row = (threadgroup_id % threadgroups_per_pair) * rows_per_threadgroup + simdgroup;
    if (pair >= pair_count) return;
    uint expert = route_experts[pair];
    float sum = 0.0f;
    if (expert < 32 && row < hidden) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) {
                scale = qx_q4_inline_scale(stream, group_base + 16u);
            }
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            uint lo = packed & 0xfu;
            uint hi = (packed >> 4u) & 0xfu;
            sum += float(int(lo) - 7) * selected_scale
                * activated[pair * intermediate + input_base];
            sum += float(int(hi) - 7) * selected_scale
                * activated[pair * intermediate + input_base + 1u];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0 && row < hidden) output[pair * hidden + row] = sum;
}

// K-wide packet-stationary Down consumer. The route compactor groups pairs
// by expert; this kernel decodes each Down packet once and applies it to all
// activated rows routed to that expert before discarding the packet.
kernel void qx_verifier_q4_moe_down_compact_unique_packet(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_counts [[buffer(1)]],
    device const uint* bucket_pairs [[buffer(2)]],
    device const float* activated [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& pair_count [[buffer(5)]],
    device const uint* active_experts [[buffer(6)]],
    constant uint& active_mode [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_activated[max_tokens * intermediate];
    uint active_bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (active_bucket >= 32) return;
    uint bucket = active_mode != 0 ? active_experts[active_bucket] : active_bucket;
    uint active = min(bucket_counts[bucket], max_tokens);
    uint bucket_base = bucket * pair_count;
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            for (uint column = thread_index; column < intermediate; column += 256) {
                cached_activated[token_slot * intermediate + column] =
                    activated[pair * intermediate + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row = row_group * rows_per_threadgroup + simdgroup;
    float sums[max_tokens] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active > 0 && row < hidden) {
        uint row_base = bucket * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) scale = qx_q4_inline_scale(stream, group_base + 16u);
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            float weight0 = float(int(packed & 0xfu) - 7) * selected_scale;
            float weight1 = float(int((packed >> 4u) & 0xfu) - 7) * selected_scale;
            for (uint token_slot = 0; token_slot < active; ++token_slot) {
                uint token_base = token_slot * intermediate;
                sums[token_slot] += weight0 * cached_activated[token_base + input_base]
                    + weight1 * cached_activated[token_base + input_base + 1u];
            }
        }
    }
    for (uint token_slot = 0; token_slot < active; ++token_slot) {
        sums[token_slot] = simd_sum(sums[token_slot]);
        if (lane == 0) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            output[pair * hidden + row] = sums[token_slot];
        }
    }
}

// K4 packet-stationary Down consumer with two packet pairs in flight.  Down
// has half as many input groups as Gate/Up, so this variant keeps the same
// four-token activation reuse while increasing independent packed-weight
// loads.  It is deliberately separate from the Gate/Up unroll4 consumer:
// the matrix shape and activation footprint are different.
kernel void qx_verifier_q4_moe_down_compact_unique_packet_k4_unroll4(
    device const uchar* stream [[buffer(0)]],
    device const uint* bucket_counts [[buffer(1)]],
    device const uint* bucket_pairs [[buffer(2)]],
    device const float* activated [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& pair_count [[buffer(5)]],
    device const uint* active_experts [[buffer(6)]],
    constant uint& active_mode [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint max_tokens = 4;
    constexpr uint threadgroups_per_bucket = hidden / rows_per_threadgroup;
    threadgroup float cached_activated[max_tokens * intermediate];

    uint active_bucket = threadgroup_id / threadgroups_per_bucket;
    uint row_group = threadgroup_id % threadgroups_per_bucket;
    if (active_bucket >= 32) return;
    uint bucket = active_mode != 0 ? active_experts[active_bucket] : active_bucket;
    uint active = min(bucket_counts[bucket], max_tokens);
    uint bucket_base = bucket * pair_count;
    for (uint token_slot = 0; token_slot < max_tokens; ++token_slot) {
        if (token_slot < active) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            for (uint column = thread_index; column < intermediate; column += 256) {
                cached_activated[token_slot * intermediate + column] =
                    activated[pair * intermediate + column];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = row_group * rows_per_threadgroup + simdgroup;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (active > 0 && row < hidden) {
        uint row_base = bucket * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 4) {
            uint local = lane & 15u;
            uint half_group = lane >= 16u ? 1u : 0u;
            uint selected0 = group + half_group;
            uint selected1 = group + 2u + half_group;
            uint base0 = row_base + selected0 * group_stride;
            uint base1 = row_base + selected1 * group_stride;
            uint packed0 = uint(stream[base0 + local]);
            uint packed1 = uint(stream[base1 + local]);
            float scale0 = 0.0f;
            float scale1 = 0.0f;
            if (local == 0u) {
                scale0 = qx_q4_inline_scale(stream, base0 + 16u);
                scale1 = qx_q4_inline_scale(stream, base1 + 16u);
            }
            float scale00 = simd_broadcast(lane == 0u ? scale0 : 0.0f, 0);
            float scale01 = simd_broadcast(lane == 16u ? scale0 : 0.0f, 16);
            float scale10 = simd_broadcast(lane == 0u ? scale1 : 0.0f, 0);
            float scale11 = simd_broadcast(lane == 16u ? scale1 : 0.0f, 16);
            float selected_scale0 = lane < 16u ? scale00 : scale01;
            float selected_scale1 = lane < 16u ? scale10 : scale11;
            uint input_base0 = selected0 * 32u + local * 2u;
            uint input_base1 = selected1 * 32u + local * 2u;
            float weight00 = float(int(packed0 & 0xfu) - 7) * selected_scale0;
            float weight01 = float(int((packed0 >> 4u) & 0xfu) - 7) * selected_scale0;
            float weight10 = float(int(packed1 & 0xfu) - 7) * selected_scale1;
            float weight11 = float(int((packed1 >> 4u) & 0xfu) - 7) * selected_scale1;
            for (uint token_slot = 0; token_slot < active; ++token_slot) {
                uint token_base = token_slot * intermediate;
                sums[token_slot] += weight00 * cached_activated[token_base + input_base0]
                    + weight01 * cached_activated[token_base + input_base0 + 1u]
                    + weight10 * cached_activated[token_base + input_base1]
                    + weight11 * cached_activated[token_base + input_base1 + 1u];
            }
        }
    }
    for (uint token_slot = 0; token_slot < active; ++token_slot) {
        sums[token_slot] = simd_sum(sums[token_slot]);
        if (lane == 0) {
            uint pair = bucket_pairs[bucket_base + token_slot];
            output[pair * hidden + row] = sums[token_slot];
        }
    }
}

kernel void qx_verifier_q4_moe_accumulate_batch(
    device const float* expert_output [[buffer(0)]],
    device const uint* route_tokens [[buffer(1)]],
    device const float* route_weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& pair_count [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint hidden = 1024;
    uint token = tid / hidden;
    uint row = tid % hidden;
    if (token >= token_count) return;
    float value = 0.0f;
    for (uint pair = 0; pair < pair_count; ++pair) {
        if (route_tokens[pair] == token) {
            value += route_weights[pair] * expert_output[pair * hidden + row];
        }
    }
    output[token * hidden + row] = value;
}

kernel void qx_dynamic_q4_moe_gate_up_inline_rows2_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_inline_packet_body(stream, expert_ids, output, cached_input, 2, lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_rows4_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_inline_packet_body(stream, expert_ids, output, cached_input, 4, lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_rows8_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_inline_packet_body(stream, expert_ids, output, cached_input, 8, lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_rows2_norm_simd(
    device const float* input [[buffer(4)]], device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]], device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]], device const uchar* stream [[buffer(0)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024]; float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column] * inverse * norm_weights[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_inline_packet_body(stream, expert_ids, output, cached_input, 2, lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_rows4_norm_simd(
    device const float* input [[buffer(4)]], device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]], device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]], device const uchar* stream [[buffer(0)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024]; float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column] * inverse * norm_weights[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_inline_packet_body(stream, expert_ids, output, cached_input, 4, lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_rows8_norm_simd(
    device const float* input [[buffer(4)]], device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]], device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]], device const uchar* stream [[buffer(0)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024]; float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column] * inverse * norm_weights[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_inline_packet_body(stream, expert_ids, output, cached_input, 8, lane, simdgroup, threadgroup_id);
}

// Exact-Q4 row-inline transport candidate. Each group is laid out as
// [16 packed code bytes][FP16 scale], keeping the existing 18 bytes/group
// while removing the separate scale stream from the hot access path.
kernel void qx_dynamic_q4_moe_gate_up_inline_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7)
                * qx_q4_inline_scale(stream, group_base + 16)
                * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_gate_up_inline_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7)
                * qx_q4_inline_scale(stream, group_base + 16)
                * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Exact-Q4 semantic candidate corresponding to the load-unique-packet
// diagnostic. Each SIMD group processes two adjacent 32-weight groups:
// lanes 0..15 load one code byte for group 0, lanes 16..31 load one code byte
// for group 1. Lanes 0 and 16 load the corresponding FP16 scale, which is
// broadcast within each half. Each lane then evaluates both nibbles in its
// byte and participates in the normal SIMD reduction.
inline void qx_dynamic_q4_gate_up_inline_unique_packet_body(
    device const uchar* stream,
    device const uint* expert_ids,
    device float* output,
    threadgroup float* cached_input,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) {
                scale = qx_q4_inline_scale(stream, group_base + 16u);
            }
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            uint lo = packed & 0xfu;
            uint hi = (packed >> 4u) & 0xfu;
            sum += float(int(lo) - 7) * selected_scale * cached_input[input_base];
            sum += float(int(hi) - 7) * selected_scale * cached_input[input_base + 1u];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_gate_up_inline_unique_packet_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_inline_unique_packet_body(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_unique_packet_norm_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]], device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_inline_unique_packet_body(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id);
}

// Llama-derived exact-Q4 dataflow control. This keeps QX row-inline storage
// but makes the execution unit two output rows per SIMD group, with two SIMD
// groups per threadgroup. Within each 32-weight group, each half of the SIMD
// group owns one row's packed bytes; only the half leader loads the scale and
// broadcasts it. This isolates the interaction between llama's N_R0=2 /
// N_SG=2 mapping and QX's proven packet load ownership.
inline void qx_dynamic_q4_gate_up_two_row_unique_packet_body(
    device const uchar* stream,
    device const float* input,
    device const float* norm_weights,
    float inverse_rms,
    uint normalized,
    device const uint* expert_ids,
    device float* output,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_simdgroup = 2;
    constexpr uint simdgroups_per_threadgroup = 2;
    constexpr uint rows_per_threadgroup = rows_per_simdgroup * simdgroups_per_threadgroup;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;

    uint slot = threadgroup_id / threadgroups_per_expert;
    uint tile = threadgroup_id % threadgroups_per_expert;
    uint first_row = tile * rows_per_threadgroup + simdgroup * rows_per_simdgroup;
    uint expert = expert_ids[slot];
    uint row_offset = lane >= 16u ? 1u : 0u;
    uint local = lane & 15u;
    float sum = 0.0f;

    if (expert < 32 && first_row + row_offset < hidden) {
        uint row = first_row + row_offset;
        uint expert_base = expert * expert_stride;
        for (uint group = 0; group < groups; ++group) {
            uint group_base = expert_base + row * row_stride + group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) {
                scale = qx_q4_inline_scale(stream, group_base + 16u);
            }
            float selected_scale = simd_broadcast(
                lane < 16u ? scale : 0.0f, row_offset == 0u ? 0 : 16);
            uint input_base = group * 32u + local * 2u;
            float x0 = input[input_base];
            float x1 = input[input_base + 1u];
            if (normalized != 0) {
                x0 *= inverse_rms * norm_weights[input_base];
                x1 *= inverse_rms * norm_weights[input_base + 1u];
            }
            sum += float(int(packed & 0xfu) - 7) * selected_scale * x0;
            sum += float(int((packed >> 4u) & 0xfu) - 7) * selected_scale * x1;
        }
    }

    sum = simd_sum(sum);
    if (lane == 0 && expert < 32 && first_row < hidden) {
        output[slot * hidden + first_row] = sum;
    }
    if (lane == 16 && expert < 32 && first_row + 1u < hidden) {
        output[slot * hidden + first_row + 1u] = sum;
    }
}

kernel void qx_dynamic_q4_moe_gate_up_two_row_unique_packet_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4_gate_up_two_row_unique_packet_body(
        stream, input, input, 1.0f, 0, expert_ids, output,
        lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_two_row_unique_packet_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4_gate_up_two_row_unique_packet_body(
        stream, input, norm_weights, inverse_rms[0], 1, expert_ids, output,
        lane, simdgroup, threadgroup_id);
}

// Production-path instrumentation variants. These intentionally preserve the
// row-inline address calculation, 256-thread dispatch, row ownership, and
// activation staging of qx_dynamic_q4_moe_gate_up_inline_simd. They are
// diagnostic kernels only; the non-canonical modes do not produce valid MoE
// outputs. `mode` is compile-time constant at each entry point after inlining:
//   0 = load packed codes/scales and sink a checksum
//   1 = decode Q4 values and sink the decoded sum
//   2 = decode and perform the dot loop, but omit SIMD final reduction
//   3 = preserve activation staging, but do not touch expert weights
//   4 = calculate production routed addresses, but do not dereference them
//   5 = load packed code/scale bytes from a contiguous production-shaped stream
//   6 = load the same packed bytes in two independent group streams per lane
//   7 = cooperatively load each 18-byte packet once across a SIMD group
inline void qx_dynamic_q4_gate_up_instrument_core(
    device const uchar* stream,
    device const uint* expert_ids,
    device float* output,
    threadgroup float* cached_input,
    uint lane,
    uint simdgroup,
    uint threadgroup_id,
    uint thread_index,
    uint mode) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;

    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    uint checksum = 0;
    if (expert < 32) {
        uint row_base;
        if (mode == 5) {
            // Keep the production row/threadgroup mapping, but replace the
            // selected-expert base with a contiguous per-threadgroup stream.
            // The host binds the same-sized artifact buffer, so this remains
            // inside the layer's available packed storage.
            row_base = threadgroup_id * rows_per_threadgroup * row_stride
                + row * row_stride;
        } else {
            row_base = expert * expert_stride + row * row_stride;
        }
        if (mode == 3) {
            // Staging-only keeps the exact activation-copy/barrier and row
            // ownership of production, while removing all expert traffic.
            // Consume one staged value so the compiler cannot discard the
            // diagnostic work entirely.
            sum = cached_input[(row * 32u + lane) & (hidden - 1u)];
        } else if (mode == 4) {
            // Address-only follows the production row-inline address
            // arithmetic without reading the resulting bytes. This isolates
            // expert/row/group/byte address generation from transactions.
            for (uint column = lane; column < hidden; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                checksum ^= group_base + local / 2;
                checksum ^= group_base + 16 + (local & 1u);
            }
        } else if (mode == 7) {
            // Each pair of groups is handled by one SIMD group. Lanes 0..15
            // load the 16 code bytes for the first group, lanes 16..31 load
            // the second; lanes 0/1 and 16/17 also load the two scale bytes.
            // This removes the production path's duplicate code-byte and
            // scale reads without changing the physical representation.
            for (uint group = 0; group < groups; group += 2) {
                uint local = lane & 15u;
                uint selected_group = group + (lane >= 16u ? 1u : 0u);
                uint group_base = row_base + selected_group * group_stride;
                checksum ^= uint(stream[group_base + local]);
                if (local < 2u) {
                    checksum ^= uint(stream[group_base + 16u + local]) << (local * 8u);
                }
            }
        } else if (mode == 6) {
            // Keep the same row-inline addresses and byte count as load-only,
            // but issue two independent groups in each loop iteration. This
            // is a diagnostic for load-level parallelism/latency hiding.
            for (uint base = lane; base < hidden; base += 64) {
                for (uint which = 0; which < 2; ++which) {
                    uint column = base + which * 32;
                    if (column >= hidden) continue;
                    uint group = column / 32;
                    uint local = column % 32;
                    uint group_base = row_base + group * group_stride;
                    uint packed = uint(stream[group_base + local / 2]);
                    uint scale_bits = uint(stream[group_base + 16])
                        | (uint(stream[group_base + 17]) << 8);
                    checksum ^= packed + scale_bits + column;
                }
            }
        } else if (mode == 0 || mode == 5) {
            for (uint column = lane; column < hidden; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                uint packed = uint(stream[group_base + local / 2]);
                uint scale_bits = uint(stream[group_base + 16])
                    | (uint(stream[group_base + 17]) << 8);
                checksum ^= packed + scale_bits + column;
            }
        } else if (mode == 1) {
            for (uint column = lane; column < hidden; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                uint packed = uint(stream[group_base + local / 2]);
                uint code = (packed >> ((local % 2) * 4)) & 0xf;
                uint scale_bits = uint(stream[group_base + 16])
                    | (uint(stream[group_base + 17]) << 8);
                sum += float(int(code) - 7)
                    * float(as_type<half>(ushort(scale_bits)));
            }
        } else {
            for (uint column = lane; column < hidden; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                uint packed = uint(stream[group_base + local / 2]);
                uint code = (packed >> ((local % 2) * 4)) & 0xf;
                uint scale_bits = uint(stream[group_base + 16])
                    | (uint(stream[group_base + 17]) << 8);
                sum += float(int(code) - 7)
                    * float(as_type<half>(ushort(scale_bits)))
                    * cached_input[column];
            }
        }
    }
    // The sink is deliberately minimal. Mode 2 writes each lane's partial
    // independently, so no SIMD reduction is introduced into that variant.
    if (mode == 0 || mode == 4 || mode == 5 || mode == 6) {
        if (lane == 0) output[slot * hidden + row] = float(checksum & 0xffffu);
    } else if (mode == 1 || mode == 3) {
        if (lane == 0) output[slot * hidden + row] = sum;
    } else {
        output[slot * hidden + row] = sum;
    }
}

kernel void qx_dynamic_q4_moe_gate_up_inline_load_only_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 0);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_load_decode_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 1);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 2);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_staging_only_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 3);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_address_only_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 4);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 5);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_load_two_groups_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 6);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 7);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_load_only_norm_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]], device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 0);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_load_decode_norm_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]], device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 1);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_norm_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]], device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 2);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_staging_only_norm_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]], device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 3);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_address_only_norm_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]], device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 4);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_norm_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]], device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 5);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_load_two_groups_norm_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]], device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 6);
}

kernel void qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_norm_simd(
    device const uchar* stream [[buffer(0)]], device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]], device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]], device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]], uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_instrument_core(stream, expert_ids, output, cached_input,
        lane, simdgroup, threadgroup_id, thread_index, 7);
}

// Decode-specific batch-1 mapping candidate. This follows the shape of the
// llama.cpp mat-vec work assignment without changing the exact Q4 row-inline
// representation: two SIMD groups (64 threads) per threadgroup, with each
// SIMD group owning two output rows. The two rows are evaluated sequentially
// while the activation vector is staged once for the threadgroup.
inline void qx_dynamic_q4_gate_up_llama_like_body(
    device const uchar* stream,
    device const uint* expert_ids,
    device float* output,
    threadgroup float* cached_input,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_simdgroup = 2;
    constexpr uint simdgroups_per_threadgroup = 2;
    constexpr uint rows_per_threadgroup = rows_per_simdgroup * simdgroups_per_threadgroup;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;

    uint slot = threadgroup_id / threadgroups_per_expert;
    uint tile = threadgroup_id % threadgroups_per_expert;
    uint first_row = tile * rows_per_threadgroup + simdgroup * rows_per_simdgroup;
    uint expert = expert_ids[slot];

    for (uint row_offset = 0; row_offset < rows_per_simdgroup; ++row_offset) {
        uint row = first_row + row_offset;
        float sum = 0.0f;
        if (expert < 32 && row < hidden) {
            uint row_base = expert * expert_stride + row * row_stride;
            for (uint column = lane; column < hidden; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                uint code = (uint(stream[group_base + local / 2])
                    >> ((local % 2) * 4)) & 0xf;
                sum += float(int(code) - 7)
                    * qx_q4_inline_scale(stream, group_base + 16)
                    * cached_input[column];
            }
        }
        sum = simd_sum(sum);
        if (lane == 0 && expert < 32 && row < hidden) {
            output[slot * hidden + row] = sum;
        }
    }
}

kernel void qx_dynamic_q4_moe_gate_up_llama_like_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 64) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_llama_like_body(
        stream, expert_ids, output, cached_input, lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_llama_like_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 64) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_llama_like_body(
        stream, expert_ids, output, cached_input, lane, simdgroup, threadgroup_id);
}

// Decode-direct exact-Q4 row-inline mapping. Each 64-thread threadgroup owns
// two rows: one row is evaluated by each of its two SIMD groups.
// Unlike the older llama-like candidate, this path never stages the complete
// activation vector; every lane reads its QX input element directly while it
// consumes the row-inline [codes][FP16 scale] packets.
inline void qx_dynamic_q4_gate_up_decode_direct_body(
    device const uchar* stream,
    device const float* input,
    device const float* norm_weights,
    float inverse_rms,
    uint normalized,
    device const uint* expert_ids,
    device float* output,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_simdgroup = 1;
    constexpr uint simdgroups_per_threadgroup = 2;
    constexpr uint rows_per_threadgroup = rows_per_simdgroup * simdgroups_per_threadgroup;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;

    uint slot = threadgroup_id / threadgroups_per_expert;
    uint tile = threadgroup_id % threadgroups_per_expert;
    uint first_row = tile * rows_per_threadgroup + simdgroup * rows_per_simdgroup;
    uint expert = expert_ids[slot];

    for (uint row_offset = 0; row_offset < rows_per_simdgroup; ++row_offset) {
        uint row = first_row + row_offset;
        float sum = 0.0f;
        if (expert < 32 && row < hidden) {
            uint row_base = expert * expert_stride + row * row_stride;
            for (uint column = lane; column < hidden; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                uint code = (uint(stream[group_base + local / 2])
                    >> ((local % 2) * 4)) & 0xf;
                float value = input[column];
                if (normalized != 0) {
                    value *= inverse_rms * norm_weights[column];
                }
                sum += float(int(code) - 7)
                    * qx_q4_inline_scale(stream, group_base + 16)
                    * value;
            }
        }
        sum = simd_sum(sum);
        if (lane == 0 && expert < 32 && row < hidden) {
            output[slot * hidden + row] = sum;
        }
    }
}

kernel void qx_dynamic_q4_moe_gate_up_decode_direct_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4_gate_up_decode_direct_body(
        stream, input, input, 1.0f, 0, expert_ids, output,
        lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_decode_direct_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4_gate_up_decode_direct_body(
        stream, input, norm_weights, inverse_rms[0], 1, expert_ids, output,
        lane, simdgroup, threadgroup_id);
}

// Dataflow factorial candidate for exact row-inline Q4. Each SIMD group owns
// two rows and keeps two accumulators live. The activation for a column is
// loaded once and reused for both rows; the canonical path loads/stages the
// same activation independently for each row. This preserves QX's physical
// representation and top-8 work while changing only accumulator ownership,
// input reuse, and the inner Q4 decode schedule.
inline void qx_dynamic_q4_gate_up_two_row_dataflow_body(
    device const uchar* stream,
    device const float* input,
    device const float* norm_weights,
    float inverse_rms,
    uint normalized,
    device const uint* expert_ids,
    device float* output,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_simdgroup = 2;
    constexpr uint simdgroups_per_threadgroup = 2;
    constexpr uint rows_per_threadgroup = rows_per_simdgroup * simdgroups_per_threadgroup;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;

    uint slot = threadgroup_id / threadgroups_per_expert;
    uint tile = threadgroup_id % threadgroups_per_expert;
    uint first_row = tile * rows_per_threadgroup + simdgroup * rows_per_simdgroup;
    uint expert = expert_ids[slot];
    float sums[rows_per_simdgroup] = {0.0f, 0.0f};

    if (expert < 32 && first_row < hidden) {
        uint expert_base = expert * expert_stride;
        for (uint group = 0; group < groups; ++group) {
            uint column = group * 32 + lane;
            float value = input[column];
            if (normalized != 0) {
                value *= inverse_rms * norm_weights[column];
            }
            for (uint row_offset = 0; row_offset < rows_per_simdgroup; ++row_offset) {
                uint row = first_row + row_offset;
                if (row >= hidden) {
                    continue;
                }
                uint group_base = expert_base + row * row_stride + group * group_stride;
                uint code = (uint(stream[group_base + lane / 2])
                    >> ((lane % 2) * 4)) & 0xf;
                sums[row_offset] += float(int(code) - 7)
                    * qx_q4_inline_scale(stream, group_base + 16)
                    * value;
            }
        }
    }

    for (uint row_offset = 0; row_offset < rows_per_simdgroup; ++row_offset) {
        sums[row_offset] = simd_sum(sums[row_offset]);
        uint row = first_row + row_offset;
        if (lane == 0 && expert < 32 && row < hidden) {
            output[slot * hidden + row] = sums[row_offset];
        }
    }
}

kernel void qx_dynamic_q4_moe_gate_up_two_row_dataflow_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4_gate_up_two_row_dataflow_body(
        stream, input, input, 1.0f, 0, expert_ids, output,
        lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_two_row_dataflow_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4_gate_up_two_row_dataflow_body(
        stream, input, norm_weights, inverse_rms[0], 1, expert_ids, output,
        lane, simdgroup, threadgroup_id);
}

// Fused gate/up consumer for exact-Q4 transaction streams. The same body also
// accepts the legacy row-inline stream when its factorial mapping is enabled;
// keeping that path here preserves its existing address contract.
inline uint qx_dynamic_q4_fused_block_group_offset(
    uint row, uint group, uint layout) {
    constexpr uint groups = 32;
    if (layout == 7 || layout == 8) {
        uint block_bytes = layout == 8 ? 160 : 144;
        return row * (groups / 8) * block_bytes
            + (group / 8) * block_bytes
            + (group % 8) * 18;
    }
    constexpr uint group_bytes = 18;
    return row * groups * group_bytes + group * group_bytes;
}

inline void qx_dynamic_q4_gate_up_fused_block_body(
    device const uchar* stream,
    device const uint* expert_ids,
    device float* output,
    threadgroup float* cached_input,
    uint lane,
    uint simdgroup,
    uint threadgroup_id,
    uint rows_per_simdgroup,
    uint simdgroups_per_threadgroup,
    uint layout) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    uint row_stride = (layout == 7 || layout == 8)
        ? (groups / 8) * (layout == 8 ? 160 : 144)
        : groups * 18;
    uint expert_stride = hidden * row_stride;
    uint rows_per_threadgroup = rows_per_simdgroup * simdgroups_per_threadgroup;
    uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint tile = threadgroup_id % threadgroups_per_expert;
    uint first_row = tile * rows_per_threadgroup
        + simdgroup * rows_per_simdgroup;
    uint expert = expert_ids[slot];

    for (uint row_offset = 0; row_offset < rows_per_simdgroup; ++row_offset) {
        uint row = first_row + row_offset;
        float sum = 0.0f;
        if (expert < 32 && row < hidden) {
            uint expert_base = expert * expert_stride;
            for (uint group = 0; group < groups; ++group) {
                uint group_base = expert_base
                    + qx_dynamic_q4_fused_block_group_offset(row, group, layout);
                uint local = lane;
                uint code = (uint(stream[group_base + local / 2])
                    >> ((local % 2) * 4)) & 0xf;
                sum += float(int(code) - 7)
                    * qx_q4_inline_scale(stream, group_base + 16)
                    * cached_input[group * 32 + local];
            }
        }
        sum = simd_sum(sum);
        if (lane == 0 && expert < 32 && row < hidden) {
            output[slot * hidden + row] = sum;
        }
    }
}

// Factorial A: fused transaction-block representation with the canonical QX
// mapping (256 threads, eight SIMD groups, eight rows per threadgroup).
kernel void qx_dynamic_q4_moe_gate_up_fused_block_qx_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_fused_block_body(
        stream, expert_ids, output, cached_input, lane, simdgroup,
        threadgroup_id, 1, 8, layout);
}

// Factorial B: the same transaction-block contract with the decode-specific
// llama-like mapping (64 threads, two SIMD groups, two rows per SIMD group).
kernel void qx_dynamic_q4_moe_gate_up_fused_block_llama_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 64) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_fused_block_body(
        stream, expert_ids, output, cached_input, lane, simdgroup,
        threadgroup_id, 2, 2, layout);
}

kernel void qx_dynamic_q4_moe_gate_up_fused_block_qx_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    constant uint& layout [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_fused_block_body(
        stream, expert_ids, output, cached_input, lane, simdgroup,
        threadgroup_id, 1, 8, layout);
}

kernel void qx_dynamic_q4_moe_gate_up_fused_block_llama_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    constant uint& layout [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 64) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_gate_up_fused_block_body(
        stream, expert_ids, output, cached_input, lane, simdgroup,
        threadgroup_id, 2, 2, layout);
}

// Payload-contiguous exact-Q4 superblock consumer. Each 144-byte block is
// [128 code bytes][8 FP16 scales] for 256 weights. A SIMD lane owns eight
// adjacent weights (one uint payload load), four neighboring lanes share one
// group scale, and each SIMD group accumulates two rows while reusing the same
// activation registers. Four SIMD groups therefore produce eight rows from a
// 256-thread threadgroup.
inline float qx_dynamic_q4_superblock_dot8(
    uint packed, float4 x0, float4 x1, float scale, float sum) {
    uint q0 = packed & 0xffu;
    uint q1 = (packed >> 8u) & 0xffu;
    uint q2 = (packed >> 16u) & 0xffu;
    uint q3 = (packed >> 24u) & 0xffu;
    sum = fma(float(int(q0 & 0xfu) - 7) * scale, x0.x, sum);
    sum = fma(float(int(q0 >> 4u) - 7) * scale, x0.y, sum);
    sum = fma(float(int(q1 & 0xfu) - 7) * scale, x0.z, sum);
    sum = fma(float(int(q1 >> 4u) - 7) * scale, x0.w, sum);
    sum = fma(float(int(q2 & 0xfu) - 7) * scale, x1.x, sum);
    sum = fma(float(int(q2 >> 4u) - 7) * scale, x1.y, sum);
    sum = fma(float(int(q3 & 0xfu) - 7) * scale, x1.z, sum);
    return fma(float(int(q3 >> 4u) - 7) * scale, x1.w, sum);
}

inline void qx_dynamic_q4_superblock_gate_up_body(
    device const uchar* stream,
    device const uint* expert_ids,
    device float* output,
    threadgroup const float* cached_input,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint hidden = 1024;
    constexpr uint blocks = 4;
    constexpr uint block_bytes = 144;
    constexpr uint row_bytes = blocks * block_bytes;
    constexpr uint expert_bytes = hidden * row_bytes;
    constexpr uint rows_per_threadgroup = 32;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint first_row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup
        + simdgroup * 4u;
    uint expert = expert_ids[slot];
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (expert < 32u) {
        uint expert_base = expert * expert_bytes;
        #pragma unroll
        for (uint block = 0; block < blocks; ++block) {
            uint column = block * 256u + lane * 8u;
            threadgroup const float4* values =
                reinterpret_cast<threadgroup const float4*>(cached_input + column);
            float4 x0 = values[0];
            float4 x1 = values[1];
            #pragma unroll
            for (uint row_offset = 0; row_offset < 4u; ++row_offset) {
                uint base = expert_base + (first_row + row_offset) * row_bytes
                    + block * block_bytes;
                uint packed = reinterpret_cast<device const uint*>(stream + base)[lane];
                float owned_scale = lane < 8u
                    ? float(reinterpret_cast<device const half*>(stream + base + 128u)[lane])
                    : 0.0f;
                float scale = simd_shuffle(owned_scale, ushort(lane >> 2u));
                sums[row_offset] = qx_dynamic_q4_superblock_dot8(
                    packed, x0, x1, scale, sums[row_offset]);
            }
        }
    }
    #pragma unroll
    for (uint row_offset = 0; row_offset < 4u; ++row_offset) {
        sums[row_offset] = simd_sum(sums[row_offset]);
    }
    if (lane == 0u && expert < 32u) {
        #pragma unroll
        for (uint row_offset = 0; row_offset < 4u; ++row_offset) {
            output[slot * hidden + first_row + row_offset] = sums[row_offset];
        }
    }
}

kernel void qx_dynamic_q4_moe_gate_up_superblock144_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024u; column += 256u) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_superblock_gate_up_body(
        stream, expert_ids, output, cached_input, lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_gate_up_superblock144_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024u; column += 256u) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_dynamic_q4_superblock_gate_up_body(
        stream, expert_ids, output, cached_input, lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4_moe_down_superblock144_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint blocks = 2;
    constexpr uint block_bytes = 144;
    constexpr uint row_bytes = blocks * block_bytes;
    constexpr uint expert_bytes = hidden * row_bytes;
    constexpr uint rows_per_threadgroup = 32;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint first_row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup
        + simdgroup * 4u;
    uint expert = expert_ids[slot];
    threadgroup float cached_input[intermediate];
    for (uint column = thread_index; column < intermediate; column += 256u) {
        cached_input[column] = activated[slot * intermediate + column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (expert < 32u) {
        uint expert_base = expert * expert_bytes;
        #pragma unroll
        for (uint block = 0; block < blocks; ++block) {
            uint column = block * 256u + lane * 8u;
            threadgroup const float4* values =
                reinterpret_cast<threadgroup const float4*>(cached_input + column);
            float4 x0 = values[0];
            float4 x1 = values[1];
            #pragma unroll
            for (uint row_offset = 0; row_offset < 4u; ++row_offset) {
                uint base = expert_base + (first_row + row_offset) * row_bytes
                    + block * block_bytes;
                uint packed = reinterpret_cast<device const uint*>(stream + base)[lane];
                float owned_scale = lane < 8u
                    ? float(reinterpret_cast<device const half*>(stream + base + 128u)[lane])
                    : 0.0f;
                float scale = simd_shuffle(owned_scale, ushort(lane >> 2u));
                sums[row_offset] = qx_dynamic_q4_superblock_dot8(
                    packed, x0, x1, scale, sums[row_offset]);
            }
        }
    }
    #pragma unroll
    for (uint row_offset = 0; row_offset < 4u; ++row_offset) {
        sums[row_offset] = simd_sum(sums[row_offset]);
    }
    if (lane == 0u && expert < 32u) {
        #pragma unroll
        for (uint row_offset = 0; row_offset < 4u; ++row_offset) {
            output[slot * hidden + first_row + row_offset] = sums[row_offset];
        }
    }
}

// Cross-expert activation-chunk candidate. A threadgroup owns one eight-row
// tile and walks all eight selected experts, so the 4 KiB normalized/input
// vector is staged once for the complete route set instead of once per
// expert tile. The candidate deliberately keeps the same row ownership and
// Q4 accumulation order as the canonical row-inline kernel; only the
// expert/tile scheduling changes. It is valid for the single-row row-inline
// stream (packetized streams use a separate geometry).
kernel void qx_dynamic_q4_moe_gate_up_cross_expert_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    if (row >= hidden) return;
    for (uint slot = 0; slot < 8; ++slot) {
        uint expert = expert_ids[slot];
        float sum = 0.0f;
        if (expert < 32) {
            uint row_base = expert * expert_stride + row * row_stride;
            for (uint column = lane; column < hidden; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                uint code = (uint(stream[group_base + local / 2])
                    >> ((local % 2) * 4)) & 0xf;
                sum += float(int(code) - 7)
                    * qx_q4_inline_scale(stream, group_base + 16)
                    * cached_input[column];
            }
        }
        sum = simd_sum(sum);
        if (lane == 0) output[slot * hidden + row] = sum;
    }
}

kernel void qx_dynamic_q4_moe_gate_up_cross_expert_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    if (row >= hidden) return;
    for (uint slot = 0; slot < 8; ++slot) {
        uint expert = expert_ids[slot];
        float sum = 0.0f;
        if (expert < 32) {
            uint row_base = expert * expert_stride + row * row_stride;
            for (uint column = lane; column < hidden; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                uint code = (uint(stream[group_base + local / 2])
                    >> ((local % 2) * 4)) & 0xf;
                sum += float(int(code) - 7)
                    * qx_q4_inline_scale(stream, group_base + 16)
                    * cached_input[column];
            }
        }
        sum = simd_sum(sum);
        if (lane == 0) output[slot * hidden + row] = sum;
    }
}

// Exact-Q4 cross-expert tile-8 consumer. The compiler artifact stores one
// shared stream in [tile][group][row-in-tile][expert][codes][FP16 scale]
// order. A threadgroup owns one tile and each SIMD group owns one selected
// expert slot, so every group/row frontier is traversed in the artifact's
// physical order. The lane/group accumulation order remains the canonical
// Q4 order: lane n consumes columns n, n+32, ... and reduces with simd_sum.
inline void qx_dynamic_q4_cross_expert_tile8_gate_up_body(
    device const uchar* stream,
    device const uint* expert_ids,
    device float* output,
    threadgroup float* cached_input,
    threadgroup uint* selected_experts,
    uint lane,
    uint simdgroup,
    uint threadgroup_id,
    uint thread_index) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint tile_rows = 8;
    constexpr uint experts = 32;
    constexpr uint group_stride = experts * 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint tile_stride = tile_rows * row_stride;

    if (thread_index < tile_rows) {
        selected_experts[thread_index] = expert_ids[thread_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint expert = selected_experts[simdgroup];
    uint first_row = threadgroup_id * tile_rows;
    for (uint row_offset = 0; row_offset < tile_rows; ++row_offset) {
        uint row = first_row + row_offset;
        float sum = 0.0f;
        if (expert < experts && row < hidden) {
            uint tile_base = threadgroup_id * tile_stride;
            for (uint group = 0; group < groups; ++group) {
                uint group_base = tile_base + group * tile_stride / tile_rows
                    + row_offset * group_stride + expert * 18;
                ushort packed = reinterpret_cast<device const ushort*>(stream + group_base)[lane >> 2];
                uint byte = (uint(packed) >> (((lane >> 1) & 1) * 8)) & 0xff;
                uint code = (byte >> ((lane & 1) * 4)) & 0xf;
                float scale = lane == 0
                    ? qx_q4_inline_scale(stream, group_base + 16)
                    : 0.0f;
                scale = simd_broadcast(scale, 0);
                sum += float(int(code) - 7)
                    * scale
                    * cached_input[group * 32 + lane];
            }
        }
        sum = simd_sum(sum);
        if (lane == 0 && row < hidden) {
            output[simdgroup * hidden + row] = sum;
        }
    }
}

kernel void qx_dynamic_q4_moe_gate_up_cross_expert_tile8_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup uint selected_experts[8];
    qx_dynamic_q4_cross_expert_tile8_gate_up_body(
        stream, expert_ids, output, cached_input, selected_experts,
        lane, simdgroup, threadgroup_id, thread_index);
}

kernel void qx_dynamic_q4_moe_gate_up_cross_expert_tile8_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup uint selected_experts[8];
    qx_dynamic_q4_cross_expert_tile8_gate_up_body(
        stream, expert_ids, output, cached_input, selected_experts,
        lane, simdgroup, threadgroup_id, thread_index);
}

// Fixed Granite-M5 top-8 row-stream candidate. A threadgroup owns one
// eight-row tile and one SIMD group owns one selected expert. The eight
// streams therefore advance through the same row/group frontier in lockstep
// instead of serializing the route inside each SIMD group as cross_expert
// does. Route IDs are staged once per threadgroup, and the row-inline
// [16-byte codes][FP16 scale] address contract is unchanged.
inline void qx_dynamic_q4_gate_up_row_stream_body(
    device const uchar* stream,
    device const uint* expert_ids,
    device float* output,
    threadgroup float* cached_input,
    threadgroup uint* selected_experts,
    uint lane,
    uint simdgroup,
    uint threadgroup_id,
    uint thread_index) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;

    if (thread_index < 8) {
        selected_experts[thread_index] = expert_ids[thread_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint expert = selected_experts[simdgroup];
    uint first_row = threadgroup_id * rows_per_threadgroup;
    for (uint row_offset = 0; row_offset < rows_per_threadgroup; ++row_offset) {
        uint row = first_row + row_offset;
        float sum = 0.0f;
        if (expert < 32 && row < hidden) {
            uint row_base = expert * expert_stride + row * row_stride;
            for (uint group = 0; group < groups; ++group) {
                uint group_base = row_base + group * group_stride;
                // The row-inline packet is 2-byte aligned. Four lanes share
                // each ushort load, covering the 16 code bytes with eight
                // coalesced word transactions instead of 32 byte requests.
                ushort packed = reinterpret_cast<device const ushort*>(stream + group_base)[lane >> 2];
                uint byte = (uint(packed) >> (((lane >> 1) & 1) * 8)) & 0xff;
                uint code = (byte >> ((lane & 1) * 4)) & 0xf;
                float scale = lane == 0
                    ? qx_q4_inline_scale(stream, group_base + 16)
                    : 0.0f;
                scale = simd_broadcast(scale, 0);
                sum += float(int(code) - 7)
                    * scale
                    * cached_input[group * 32 + lane];
            }
        }
        sum = simd_sum(sum);
        if (lane == 0) {
            output[simdgroup * hidden + row] = sum;
        }
    }
}

kernel void qx_dynamic_q4_moe_gate_up_row_stream_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup uint selected_experts[8];
    qx_dynamic_q4_gate_up_row_stream_body(
        stream, expert_ids, output, cached_input, selected_experts,
        lane, simdgroup, threadgroup_id, thread_index);
}

kernel void qx_dynamic_q4_moe_gate_up_row_stream_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < 1024; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup uint selected_experts[8];
    qx_dynamic_q4_gate_up_row_stream_body(
        stream, expert_ids, output, cached_input, selected_experts,
        lane, simdgroup, threadgroup_id, thread_index);
}

// Vectorized exact-Q4 row-inline candidate. Each lane retains the canonical
// row/reduction ownership, but consumes four quantization groups at a time.
// This keeps the numerical reduction order close to the reference kernel
// while exposing float4 activation/code work to the compiler.
kernel void qx_dynamic_q4_moe_gate_up_inline_vec4_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];

    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32 && row < hidden) {
        uint row_base_offset = expert * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 4) {
            for (uint i = 0; i < 4; ++i) {
                uint group_base = row_base_offset + (group + i) * group_stride;
                uchar byte = stream[group_base + (lane >> 1)];
                uint code = (uint(byte) >> ((lane & 1) * 4)) & 0xf;
                uint bits = uint(stream[group_base + 16])
                    | (uint(stream[group_base + 17]) << 8);
                float scale = float(as_type<half>(ushort(bits)));
                sum += float(int(code) - 7) * scale * cached_input[(group + i) * 32 + lane];
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_gate_up_inline_vec4_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];

    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32 && row < hidden) {
        uint row_base_offset = expert * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 4) {
            for (uint i = 0; i < 4; ++i) {
                uint group_base = row_base_offset + (group + i) * group_stride;
                uchar byte = stream[group_base + (lane >> 1)];
                uint code = (uint(byte) >> ((lane & 1) * 4)) & 0xf;
                uint bits = uint(stream[group_base + 16])
                    | (uint(stream[group_base + 17]) << 8);
                float scale = float(as_type<half>(ushort(bits)));
                float normalized = cached_input[(group + i) * 32 + lane];
                sum += float(int(code) - 7) * scale * normalized;
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// The activation vector is only 4 KiB and is normally already hot after the
// preceding router/RMS work. This exact-Q4 candidate avoids copying it into
// threadgroup memory for every one of the 1024 selected-expert row tiles.
kernel void qx_dynamic_q4_moe_gate_up_inline_nocache_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7)
                * qx_q4_inline_scale(stream, group_base + 16)
                * input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_gate_up_inline_nocache_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float inverse = inverse_rms[0];
    float sum = 0.0f;
    if (expert < 32) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2]) >> ((local % 2) * 4)) & 0xf;
            float normalized = input[column] * inverse * norm_weights[column];
            sum += float(int(code) - 7)
                * qx_q4_inline_scale(stream, group_base + 16)
                * normalized;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Dynamic top-8 Q4 MoE. All 32 experts are resident in contiguous packed
// buffers; GPU-produced IDs select the expert stride. One threadgroup
// processes the fixed batch-1 token and accumulates weighted expert outputs.
kernel void qx_dynamic_q4_moe(
    device const uchar* gate_up_packed [[buffer(0)]],
    device const half* gate_up_scales [[buffer(1)]],
    device const uchar* down_packed [[buffer(2)]],
    device const half* down_scales [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device const float* expert_weights [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& layout [[buffer(8)]],
    uint row [[thread_position_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint group_size = 32;
    constexpr uint qmax = 7;
    constexpr uint bytes_per_group = 16;
    constexpr uint groups_up = 32;
    constexpr uint groups_down = 16;
    uint gate_up_packed_stride = qx_dynamic_packed_stride(hidden, groups_up, layout);
    uint gate_up_scale_stride = qx_dynamic_scale_stride(hidden, groups_up, layout);
    uint down_packed_stride = qx_dynamic_packed_stride(hidden, groups_down, layout);
    uint down_scale_stride = qx_dynamic_scale_stride(hidden, groups_down, layout);
    threadgroup float gate_up[hidden];
    threadgroup float activated[intermediate];
    float accumulated = 0.0f;

    for (uint slot = 0; slot < 8; ++slot) {
        uint expert = expert_ids[slot];
        if (expert >= 32) return;
        uint gate_base = expert * gate_up_packed_stride;
        uint gate_scale_base = expert * gate_up_scale_stride;
        float gate_up_sum = 0.0f;
        for (uint column = 0; column < hidden; ++column) {
            uint group = column / group_size;
            uint local = column % group_size;
            uint offset = gate_base + qx_dynamic_group_offset(row, group, groups_up, bytes_per_group, layout);
            uchar byte = gate_up_packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = gate_scale_base + qx_dynamic_scale_offset(row, group, groups_up, layout);
            float weight = float(int(code) - int(qmax)) * float(gate_up_scales[scale]);
            gate_up_sum += weight * input[column];
        }
        gate_up[row] = gate_up_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row < intermediate) {
            float gate = gate_up[row];
            float up = gate_up[intermediate + row];
            activated[row] = (gate / (1.0f + exp(-gate))) * up;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint down_base = expert * down_packed_stride;
        uint down_scale_base = expert * down_scale_stride;
        float down_sum = 0.0f;
        for (uint column = 0; column < intermediate; ++column) {
            uint group = column / group_size;
            uint local = column % group_size;
            uint offset = down_base + qx_dynamic_group_offset(row, group, groups_down, bytes_per_group, layout);
            uchar byte = down_packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = down_scale_base + qx_dynamic_scale_offset(row, group, groups_down, layout);
            float weight = float(int(code) - int(qmax)) * float(down_scales[scale]);
            down_sum += weight * activated[column];
        }
        accumulated += expert_weights[slot] * down_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[row] = accumulated;
}

// SIMD-cooperative dynamic top-8 Q4 MoE. The threadgroup still owns the full
// 1024-row expert invocation, but each SIMD group reduces the K dimension for
// one row. The input is staged once in threadgroup memory and reused by all
// eight selected experts.
kernel void qx_dynamic_q4_moe_simd(
    device const uchar* gate_up_packed [[buffer(0)]],
    device const half* gate_up_scales [[buffer(1)]],
    device const uchar* down_packed [[buffer(2)]],
    device const half* down_scales [[buffer(3)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device const float* expert_weights [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& layout [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint group_size = 32;
    constexpr uint qmax = 7;
    constexpr uint bytes_per_group = 16;
    constexpr uint groups_up = 32;
    constexpr uint groups_down = 16;
    uint gate_up_packed_stride = qx_dynamic_packed_stride(hidden, groups_up, layout);
    uint gate_up_scale_stride = qx_dynamic_scale_stride(hidden, groups_up, layout);
    uint down_packed_stride = qx_dynamic_packed_stride(hidden, groups_down, layout);
    uint down_scale_stride = qx_dynamic_scale_stride(hidden, groups_down, layout);
    threadgroup float cached_input[hidden];
    threadgroup float gate_up[hidden];
    threadgroup float activated[intermediate];
    threadgroup float accumulated_output[hidden];

    cached_input[thread_index] = input[thread_index];
    accumulated_output[thread_index] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint rows_per_simd = 32;
    uint first_row = simdgroup * rows_per_simd;
    for (uint slot = 0; slot < 8; ++slot) {
        uint expert = expert_ids[slot];
        if (expert >= 32) continue;
        uint gate_base = expert * gate_up_packed_stride;
        uint gate_scale_base = expert * gate_up_scale_stride;
        for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
            uint row = first_row + row_offset;
            float gate_up_sum = 0.0f;
            for (uint column = lane; column < hidden; column += 32) {
                uint group = column / group_size;
                uint local = column % group_size;
                uint offset = gate_base
                    + qx_dynamic_group_offset(row, group, groups_up, bytes_per_group, layout);
                uchar byte = gate_up_packed[offset + local / 2];
                uint code = (byte >> ((local % 2) * 4)) & 0xf;
                uint scale = gate_scale_base
                    + qx_dynamic_scale_offset(row, group, groups_up, layout);
                float weight = float(int(code) - int(qmax)) * float(gate_up_scales[scale]);
                gate_up_sum += weight * cached_input[column];
            }
            gate_up_sum = simd_sum(gate_up_sum);
            if (lane == 0) gate_up[row] = gate_up_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0) {
            for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
                uint row = first_row + row_offset;
                if (row < intermediate) {
                    float gate = gate_up[row];
                    float up = gate_up[intermediate + row];
                    activated[row] = (gate / (1.0f + exp(-gate))) * up;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint down_base = expert * down_packed_stride;
        uint down_scale_base = expert * down_scale_stride;
        for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
            uint row = first_row + row_offset;
            float down_sum = 0.0f;
            for (uint column = lane; column < intermediate; column += 32) {
                uint group = column / group_size;
                uint local = column % group_size;
                uint offset = down_base
                    + qx_dynamic_group_offset(row, group, groups_down, bytes_per_group, layout);
                uchar byte = down_packed[offset + local / 2];
                uint code = (byte >> ((local % 2) * 4)) & 0xf;
                uint scale = down_scale_base
                    + qx_dynamic_scale_offset(row, group, groups_down, layout);
                float weight = float(int(code) - int(qmax)) * float(down_scales[scale]);
                down_sum += weight * activated[column];
            }
            down_sum = simd_sum(down_sum);
            if (lane == 0) {
                accumulated_output[row] += expert_weights[slot] * down_sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
            output[first_row + row_offset] = accumulated_output[first_row + row_offset];
        }
    }
}

// Batched SIMD gate/up for the eight GPU-selected experts. One SIMD group
// owns one output row; experts and rows are flattened across threadgroups.
kernel void qx_dynamic_q4_moe_gate_up_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint bytes_per_group = 16;
    uint packed_stride = qx_dynamic_packed_stride(hidden, groups, layout);
    uint scale_stride = qx_dynamic_scale_stride(hidden, groups, layout);
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * scale_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base + qx_dynamic_group_offset(row, group, groups, bytes_per_group, layout);
            uchar byte = packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = scale_base + qx_dynamic_scale_offset(row, group, groups, layout);
            sum += float(int(code) - 7) * float(scales[scale]) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Batched SIMD gate/up with projection-aware RMSNorm. The raw post-attention
// hidden vector is normalized in the shared input cache; no normalized
// 1024-element global buffer is materialized for the router or experts.
kernel void qx_dynamic_q4_moe_gate_up_norm_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint bytes_per_group = 16;
    uint packed_stride = qx_dynamic_packed_stride(hidden, groups, layout);
    uint scale_stride = qx_dynamic_scale_stride(hidden, groups, layout);
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * scale_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base + qx_dynamic_group_offset(row, group, groups, bytes_per_group, layout);
            uchar byte = packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = scale_base + qx_dynamic_scale_offset(row, group, groups, layout);
            sum += float(int(code) - 7) * float(scales[scale]) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Wide-expert gate/up candidate. One 1024-threadgroup owns one selected
// expert, so the normalized/input vector is staged once per expert rather
// than once per eight-row tile. SIMD groups walk disjoint row lanes and each
// group covers the remaining rows serially. This is deliberately a separate
// candidate: it may trade occupancy for less repeated activation traffic.
kernel void qx_dynamic_q4_moe_gate_up_wide_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint bytes_per_group = 16;
    uint packed_stride = qx_dynamic_packed_stride(hidden, groups, layout);
    uint scale_stride = qx_dynamic_scale_stride(hidden, groups, layout);
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 1024) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id;
    uint expert = expert_ids[slot];
    if (expert >= 32) return;
    uint packed_base = expert * packed_stride;
    uint scale_base = expert * scale_stride;
    for (uint row = simdgroup; row < hidden; row += 32) {
        float sum = 0.0f;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base
                + qx_dynamic_group_offset(row, group, groups, bytes_per_group, layout);
            uchar byte = packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = scale_base + qx_dynamic_scale_offset(row, group, groups, layout);
            sum += float(int(code) - 7) * float(scales[scale]) * cached_input[column];
        }
        sum = simd_sum(sum);
        if (lane == 0) output[slot * hidden + row] = sum;
    }
}

kernel void qx_dynamic_q4_moe_gate_up_wide_norm_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint bytes_per_group = 16;
    uint packed_stride = qx_dynamic_packed_stride(hidden, groups, layout);
    uint scale_stride = qx_dynamic_scale_stride(hidden, groups, layout);
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 1024) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id;
    uint expert = expert_ids[slot];
    if (expert >= 32) return;
    uint packed_base = expert * packed_stride;
    uint scale_base = expert * scale_stride;
    for (uint row = simdgroup; row < hidden; row += 32) {
        float sum = 0.0f;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base
                + qx_dynamic_group_offset(row, group, groups, bytes_per_group, layout);
            uchar byte = packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = scale_base + qx_dynamic_scale_offset(row, group, groups, layout);
            sum += float(int(code) - 7) * float(scales[scale]) * cached_input[column];
        }
        sum = simd_sum(sum);
        if (lane == 0) output[slot * hidden + row] = sum;
    }
}

inline void qx_q4k_scale_min(device const uchar* metadata, uint sub, thread uint& scale, thread uint& minimum) {
    if (sub < 4) {
        scale = uint(metadata[sub]) & 63;
        minimum = uint(metadata[sub + 4]) & 63;
    } else {
        scale = (uint(metadata[sub + 4]) & 15) | ((uint(metadata[sub - 4]) >> 6) << 4);
        minimum = (uint(metadata[sub + 4]) >> 4) | ((uint(metadata[sub]) >> 6) << 4);
    }
}

inline float qx_q4k_weight(
    device const uchar* packed,
    device const half* scales,
    uint packed_base,
    uint scale_base,
    uint row,
    uint blocks,
    uint column) {
    constexpr uint block_bytes = 140;
    uint block = column / 256;
    uint within = column % 256;
    uint sub = within / 32;
    uint local = within % 32;
    uint base = packed_base + (row * blocks + block) * block_bytes;
    device const uchar* metadata = packed + base;
    uchar byte = packed[base + 12 + (sub / 2) * 32 + local];
    uint code = (sub & 1) == 0 ? uint(byte & 15) : uint(byte >> 4);
    uint scale_code;
    uint min_code;
    qx_q4k_scale_min(metadata, sub, scale_code, min_code);
    uint dm = scale_base + (row * blocks + block) * 2;
    return float(scales[dm]) * float(scale_code * code)
        - float(scales[dm + 1]) * float(min_code);
}

kernel void qx_dynamic_q4k_moe_gate_up_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint blocks = 4;
    constexpr uint packed_stride = hidden * blocks * 140;
    constexpr uint scale_stride = hidden * blocks * 2;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * scale_stride;
        for (uint column = lane; column < hidden; column += 32) {
            sum += qx_q4k_weight(packed, scales, packed_base, scale_base, row, blocks, column)
                * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4k_moe_gate_up_norm_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint blocks = 4;
    constexpr uint packed_stride = hidden * blocks * 140;
    constexpr uint scale_stride = hidden * blocks * 2;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * scale_stride;
        for (uint column = lane; column < hidden; column += 32) {
            sum += qx_q4k_weight(packed, scales, packed_base, scale_base, row, blocks, column)
                * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Q4_K batch-1 control using the same 2-row/SIMD-group mapping as llama's
// decode mat-vec kernel. QX's artifact keeps the 12-byte Q4_K scale metadata
// in the packed stream and stores d/m as two FP16 values per block in the
// companion scale stream, so this is an execution control rather than an
// artifact conversion. Four SIMD groups per 128-thread threadgroup cover
// eight output rows while every lane reads the activation vector directly.
inline void qx_dynamic_q4k_mature_body_simple(
    device const uchar* packed,
    device const half* scales,
    device const float* input,
    device const float* norm_weights,
    float inverse_rms,
    uint normalized,
    device const uint* expert_ids,
    device float* output,
    uint lane,
    uint simdgroup,
    uint threadgroup_id,
    uint rows,
    uint blocks,
    uint columns) {
    constexpr uint rows_per_simdgroup = 2;
    constexpr uint simdgroups_per_threadgroup = 4;
    constexpr uint rows_per_threadgroup = rows_per_simdgroup * simdgroups_per_threadgroup;
    uint threadgroups_per_expert = rows / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint tile = threadgroup_id % threadgroups_per_expert;
    uint first_row = (tile * simdgroups_per_threadgroup + simdgroup) * rows_per_simdgroup;
    uint expert = expert_ids[slot];
    if (expert >= 32 || first_row >= rows) return;
    uint packed_base = expert * rows * blocks * 140;
    uint scale_base = expert * rows * blocks * 2;
    for (uint row_offset = 0; row_offset < rows_per_simdgroup; ++row_offset) {
        uint row = first_row + row_offset;
        if (row >= rows) continue;
        float sum = 0.0f;
        for (uint column = lane; column < columns; column += 32) {
            float value = input[column];
            if (normalized) {
                value *= inverse_rms * norm_weights[column];
            }
            sum += qx_q4k_weight(
                packed, scales, packed_base, scale_base, row, blocks, column) * value;
        }
        sum = simd_sum(sum);
        if (lane == 0) output[slot * rows + row] = sum;
    }
}

inline void qx_dynamic_q4k_mature_body(
    device const uchar* packed,
    device const half* scales,
    device const float* input,
    device const float* norm_weights,
    float inverse_rms,
    uint normalized,
    device const uint* expert_ids,
    device float* output,
    uint lane,
    uint simdgroup,
    uint threadgroup_id,
    uint rows,
    uint blocks) {
    constexpr uint rows_per_simdgroup = 2;
    constexpr uint simdgroups_per_threadgroup = 4;
    constexpr uint rows_per_threadgroup = rows_per_simdgroup * simdgroups_per_threadgroup;
    constexpr uint block_bytes = 140;
    uint threadgroups_per_expert = rows / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint tile = threadgroup_id % threadgroups_per_expert;
    uint first_row = (tile * simdgroups_per_threadgroup + simdgroup) * rows_per_simdgroup;
    uint expert = expert_ids[slot];
    uint packed_stride = rows * blocks * block_bytes;
    uint scale_stride = rows * blocks * 2;
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    if (expert >= 32 || first_row >= rows) return;
    uint packed_expert = expert * packed_stride;
    uint scale_expert = expert * scale_stride;
    uint ix = lane / 8;
    uint it = lane % 8;
    uint iq = it / 4;
    uint ir = it % 4;
    for (uint row_offset = 0; row_offset < rows_per_simdgroup; ++row_offset) {
        uint row = first_row + row_offset;
        if (row >= rows) continue;
        float sum = 0.0f;
        for (uint ib = ix; ib < blocks; ib += 4) {
            float yl[16];
            float yh[16];
            float4 sumy = float4(0.0f);
            uint input_base = ix * 256 + 64 * iq + 8 * ir;
            for (uint i = 0; i < 8; ++i) {
                float v0 = input[input_base + i];
                float v1 = input[input_base + 32 + i];
                float v2 = input[input_base + 128 + i];
                float v3 = input[input_base + 160 + i];
                if (normalized) {
                    v0 *= inverse_rms * norm_weights[input_base + i];
                    v1 *= inverse_rms * norm_weights[input_base + 32 + i];
                    v2 *= inverse_rms * norm_weights[input_base + 128 + i];
                    v3 *= inverse_rms * norm_weights[input_base + 160 + i];
                }
                yl[i] = v0;
                yl[i + 8] = v1;
                yh[i] = v2;
                yh[i + 8] = v3;
                sumy[0] += v0;
                sumy[1] += v1;
                sumy[2] += v2;
                sumy[3] += v3;
            }

            uint block_base = packed_expert + (row * blocks + ib) * block_bytes;
            device const uint16_t* sc = (device const uint16_t*)(packed + block_base);
            device const uint16_t* q1 = (device const uint16_t*)(packed + block_base + 12)
                + 16 * iq + 4 * ir;
            device const half* dm = scales + scale_expert + (row * blocks + ib) * 2;
            uint16_t sc16[4];
            uint8_t sc8[8];
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            device const uint16_t* q2 = q1 + 32;
            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            for (uint i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000f);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0f00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00f0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xf000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000f);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0f00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00f0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xf000);
            }
            sum += float(dm[0]) * ((acc1[0] + 1.0f / 256.0f * acc1[1]) * sc8[0]
                + (acc1[2] + 1.0f / 256.0f * acc1[3]) * sc8[1] / 16.0f
                + (acc2[0] + 1.0f / 256.0f * acc2[1]) * sc8[4]
                + (acc2[2] + 1.0f / 256.0f * acc2[3]) * sc8[5] / 16.0f)
                - float(dm[1]) * (sumy[0] * sc8[2] + sumy[1] * sc8[3]
                + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
        sum = simd_sum(sum);
        if (lane == 0) output[slot * rows + row] = sum;
    }
}

kernel void qx_dynamic_q4k_moe_gate_up_mature_simd(
    device const uchar* packed [[buffer(0)]], device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]], device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4k_mature_body_simple(packed, scales, input, input, 1.0f, 0, expert_ids, output,
        lane, simdgroup, threadgroup_id, 1024, 4, 1024);
}

kernel void qx_dynamic_q4k_moe_gate_up_mature_norm_simd(
    device const uchar* packed [[buffer(0)]], device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]], device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]], device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4k_mature_body_simple(packed, scales, input, norm_weights, inverse_rms[0], 1, expert_ids, output,
        lane, simdgroup, threadgroup_id, 1024, 4, 1024);
}

inline void qx_dynamic_q4k_mature_down_body(
    device const uchar* packed,
    device const half* scales,
    device const float* input,
    device float* output,
    device const uint* expert_ids,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint rows = 1024;
    constexpr uint blocks = 2;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = rows / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint tile = threadgroup_id % threadgroups_per_expert;
    uint row = tile * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    if (expert >= 32 || row >= rows) return;
    uint packed_base = expert * rows * blocks * 140;
    uint scale_base = expert * rows * blocks * 2;
    float sum = 0.0f;
    for (uint column = lane; column < 512; column += 32) {
        sum += qx_q4k_weight(
            packed, scales, packed_base, scale_base, row, blocks, column) * input[slot * 512 + column];
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * rows + row] = sum;
}

kernel void qx_dynamic_q4k_moe_down_mature_simd(
    device const uchar* packed [[buffer(0)]], device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]], device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4k_mature_down_body(
        packed, scales, input, output, expert_ids, lane, simdgroup, threadgroup_id);
}

kernel void qx_dynamic_q4k_moe_down_mature_norm_simd(
    device const uchar* packed [[buffer(0)]], device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]], device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]], device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]], uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4k_mature_down_body(
        packed, scales, input, output, expert_ids, lane, simdgroup, threadgroup_id);
}

// Experimental Granite gate/up Q3 path.  The logical format is row-major,
// group-32: twelve bytes of packed 3-bit codes followed by an FP16 scale in
// the separate scale buffer.  `block_bytes` is either 12 (dense) or 16
// (transaction-native, with four trailing padding bytes per group).  Keeping
// the two layouts in one kernel makes the comparison about representation,
// not a second scheduling design.
inline uint qx_dynamic_q3_code(device const uchar* packed, uint base, uint local) {
    uint bit = local * 3;
    uint byte = bit / 8;
    uint shift = bit % 8;
    uint code = uint(packed[base + byte]) >> shift;
    if (shift > 5) {
        code |= uint(packed[base + byte + 1]) << (8 - shift);
    }
    return code & 0x7;
}

kernel void qx_dynamic_q3_moe_gate_up_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& block_bytes [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    constexpr uint qmax = 3;
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint packed_stride = hidden * groups * block_bytes;
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * hidden * groups;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = packed_base + row * groups * block_bytes + group * block_bytes;
            uint code = qx_dynamic_q3_code(packed, group_base, local);
            float weight = float(int(code) - int(qmax)) * float(scales[scale_base + row * groups + group]);
            sum += weight * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q3_moe_gate_up_norm_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& block_bytes [[buffer(7)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    constexpr uint qmax = 3;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint packed_stride = hidden * groups * block_bytes;
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * hidden * groups;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = packed_base + row * groups * block_bytes + group * block_bytes;
            uint code = qx_dynamic_q3_code(packed, group_base, local);
            float weight = float(int(code) - int(qmax)) * float(scales[scale_base + row * groups + group]);
            sum += weight * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Q3 down projection for a fully Q3 expert plan. The padded/dense block
// choice is supplied at runtime so the arithmetic kernel remains identical
// while the compiler can compare transaction-friendly physical formats.
kernel void qx_dynamic_q3_moe_down_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& block_bytes [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    constexpr uint qmax = 3;
    threadgroup float cached_input[intermediate];
    for (uint column = thread_index; column < intermediate; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint packed_stride = hidden * groups * block_bytes;
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * hidden * groups;
        for (uint column = lane; column < intermediate; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = packed_base + row * groups * block_bytes + group * block_bytes;
            uint code = qx_dynamic_q3_code(packed, group_base, local);
            float weight = float(int(code) - int(qmax))
                * float(scales[scale_base + row * groups + group]);
            sum += weight * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Expertwise Q3/Q4 candidates. The map is indexed by the logical router
// expert id. Bit 31 marks a Q3 physical slot; the remaining bits are the
// compact slot in that representation's slab. Each representation is
// dispatched separately, so a selected slot is either computed by exactly
// one kernel or skipped by both.
inline bool qx_expert_map_is_q3(uint encoded) {
    return (encoded & 0x80000000u) != 0;
}

inline uint qx_expert_map_slot(uint encoded) {
    return encoded & 0x7fffffffu;
}

kernel void qx_dynamic_q4_moe_gate_up_expertwise_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const uint* expert_map [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint bytes_per_group = 16;
    constexpr uint packed_stride = hidden * groups * bytes_per_group;
    constexpr uint scale_stride = hidden * groups;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint logical = expert_ids[slot];
    uint encoded = logical < 32 ? expert_map[logical] : 0xffffffffu;
    float sum = 0.0f;
    if (logical < 32 && encoded != 0xffffffffu && !qx_expert_map_is_q3(encoded)) {
        uint physical = qx_expert_map_slot(encoded);
        uint packed_base = physical * packed_stride;
        uint scale_base = physical * scale_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base + row * groups * bytes_per_group + group * bytes_per_group;
            uint code = (uint(packed[offset + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7) * float(scales[scale_base + row * groups + group]) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_gate_up_expertwise_norm_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    device const uint* expert_map [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint bytes_per_group = 16;
    constexpr uint packed_stride = hidden * groups * bytes_per_group;
    constexpr uint scale_stride = hidden * groups;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 256) cached_input[column] = input[column] * inverse * norm_weights[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint logical = expert_ids[slot];
    uint encoded = logical < 32 ? expert_map[logical] : 0xffffffffu;
    float sum = 0.0f;
    if (logical < 32 && encoded != 0xffffffffu && !qx_expert_map_is_q3(encoded)) {
        uint physical = qx_expert_map_slot(encoded);
        uint packed_base = physical * packed_stride;
        uint scale_base = physical * scale_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base + row * groups * bytes_per_group + group * bytes_per_group;
            uint code = (uint(packed[offset + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7) * float(scales[scale_base + row * groups + group]) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_gate_up_expertwise_inline_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const uint* expert_map [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint logical = expert_ids[slot];
    uint encoded = logical < 32 ? expert_map[logical] : 0xffffffffu;
    float sum = 0.0f;
    if (logical < 32 && encoded != 0xffffffffu && !qx_expert_map_is_q3(encoded)) {
        uint physical = qx_expert_map_slot(encoded);
        uint row_base = physical * expert_stride + row * row_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7) * qx_q4_inline_scale(stream, group_base + 16) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_gate_up_expertwise_inline_norm_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    device const uint* expert_map [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 256) cached_input[column] = input[column] * inverse * norm_weights[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint logical = expert_ids[slot];
    uint encoded = logical < 32 ? expert_map[logical] : 0xffffffffu;
    float sum = 0.0f;
    if (logical < 32 && encoded != 0xffffffffu && !qx_expert_map_is_q3(encoded)) {
        uint physical = qx_expert_map_slot(encoded);
        uint row_base = physical * expert_stride + row * row_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7) * qx_q4_inline_scale(stream, group_base + 16) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q3_moe_gate_up_expertwise_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& block_bytes [[buffer(7)]],
    device const uint* expert_map [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    constexpr uint qmax = 3;
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 256) cached_input[column] = input[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint logical = expert_ids[slot];
    uint encoded = logical < 32 ? expert_map[logical] : 0xffffffffu;
    float sum = 0.0f;
    if (logical < 32 && encoded != 0xffffffffu && qx_expert_map_is_q3(encoded)) {
        uint physical = qx_expert_map_slot(encoded);
        uint packed_stride = hidden * groups * block_bytes;
        uint packed_base = physical * packed_stride;
        uint scale_base = physical * hidden * groups;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = packed_base + row * groups * block_bytes + group * block_bytes;
            uint code = qx_dynamic_q3_code(packed, group_base, local);
            sum += float(int(code) - int(qmax)) * float(scales[scale_base + row * groups + group]) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q3_moe_gate_up_expertwise_norm_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& block_bytes [[buffer(7)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    device const uint* expert_map [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint groups = 32;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    constexpr uint qmax = 3;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 256) cached_input[column] = input[column] * inverse * norm_weights[column];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint logical = expert_ids[slot];
    uint encoded = logical < 32 ? expert_map[logical] : 0xffffffffu;
    float sum = 0.0f;
    if (logical < 32 && encoded != 0xffffffffu && qx_expert_map_is_q3(encoded)) {
        uint physical = qx_expert_map_slot(encoded);
        uint packed_stride = hidden * groups * block_bytes;
        uint packed_base = physical * packed_stride;
        uint scale_base = physical * hidden * groups;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = packed_base + row * groups * block_bytes + group * block_bytes;
            uint code = qx_dynamic_q3_code(packed, group_base, local);
            sum += float(int(code) - int(qmax)) * float(scales[scale_base + row * groups + group]) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_down_expertwise_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const uint* expert_map [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint bytes_per_group = 16;
    constexpr uint packed_stride = hidden * groups * bytes_per_group;
    constexpr uint scale_stride = hidden * groups;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint logical = expert_ids[slot];
    uint encoded = logical < 32 ? expert_map[logical] : 0xffffffffu;
    float sum = 0.0f;
    if (logical < 32 && encoded != 0xffffffffu && !qx_expert_map_is_q3(encoded)) {
        uint physical = qx_expert_map_slot(encoded);
        uint packed_base = physical * packed_stride;
        uint scale_base = physical * scale_stride;
        for (uint column = lane; column < intermediate; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base + row * groups * bytes_per_group + group * bytes_per_group;
            uint code = (uint(packed[offset + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7) * float(scales[scale_base + row * groups + group]) * activated[slot * intermediate + column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_down_expertwise_inline_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const uint* expert_map [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint logical = expert_ids[slot];
    uint encoded = logical < 32 ? expert_map[logical] : 0xffffffffu;
    float sum = 0.0f;
    if (logical < 32 && encoded != 0xffffffffu && !qx_expert_map_is_q3(encoded)) {
        uint physical = qx_expert_map_slot(encoded);
        uint row_base = physical * expert_stride + row * row_stride;
        for (uint column = lane; column < intermediate; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7) * qx_q4_inline_scale(stream, group_base + 16) * activated[slot * intermediate + column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q3_moe_down_expertwise_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& block_bytes [[buffer(7)]],
    device const uint* expert_map [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    constexpr uint qmax = 3;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint logical = expert_ids[slot];
    uint encoded = logical < 32 ? expert_map[logical] : 0xffffffffu;
    float sum = 0.0f;
    if (logical < 32 && encoded != 0xffffffffu && qx_expert_map_is_q3(encoded)) {
        uint physical = qx_expert_map_slot(encoded);
        uint packed_stride = hidden * groups * block_bytes;
        uint packed_base = physical * packed_stride;
        uint scale_base = physical * hidden * groups;
        for (uint column = lane; column < intermediate; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = packed_base + row * groups * block_bytes + group * block_bytes;
            uint code = qx_dynamic_q3_code(packed, group_base, local);
            sum += float(int(code) - int(qmax)) * float(scales[scale_base + row * groups + group]) * activated[slot * intermediate + column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Benchmark-only gate/up traffic control. It traverses exactly the selected
// experts' packed Q4 payload and scale bytes, but performs no dot products.
// The sink write keeps the loads observable without making this a model path.
kernel void qx_dynamic_q4_moe_gate_up_touch(
    device const uchar* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const uchar* contiguous [[buffer(2)]],
    device const uint* expert_ids [[buffer(5)]],
    device uint* sink [[buffer(6)]],
    constant uint& mode [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint packed_stride = 1024 * 1024 / 2;
    constexpr uint scale_bytes = 1024 * 32 * 2;
    constexpr uint expert_bytes = packed_stride + scale_bytes;
    uint total_bytes = 8 * expert_bytes;
    uint fraction = 4;
    if (mode >= 4 && mode <= 7) fraction = mode - 3;
    if (mode >= 8 && mode <= 11) fraction = mode - 7;
    if (mode == 1 || (mode >= 4 && mode <= 7)) {
        total_bytes = 8 * packed_stride * fraction / 4;
    }
    if (mode == 2) total_bytes = 8 * scale_bytes;
    if (mode >= 8 && mode <= 11) total_bytes = 8 * expert_bytes * fraction / 4;
    if (tid >= total_bytes) return;
    uint value;
    if (mode >= 12 && mode <= 15) {
        uint region_slot;
        uint offset;
        if (mode == 12) {
            region_slot = 0;
            offset = tid % expert_bytes;
        } else if (mode == 13) {
            constexpr uint region_bytes = 2 * expert_bytes;
            region_slot = (tid / region_bytes) & 1;
            offset = (tid % region_bytes) % expert_bytes;
        } else if (mode == 14) {
            constexpr uint region_bytes = 4 * expert_bytes;
            region_slot = (tid / region_bytes) & 3;
            offset = (tid % region_bytes) % expert_bytes;
        } else {
            constexpr uint region_bytes = 8 * expert_bytes;
            region_slot = (tid / region_bytes) & 7;
            offset = (tid % region_bytes) % expert_bytes;
        }
        uint expert = expert_ids[region_slot];
        if (expert >= 32) return;
        if (offset < packed_stride) {
            value = uint(packed[expert * packed_stride + offset]);
        } else {
            uint scale_offset = (offset - packed_stride) / 2;
            value = uint(scales[expert * (scale_bytes / 2) + scale_offset]);
        }
    } else if (mode == 3 || (mode >= 8 && mode <= 11)) {
        value = uint(contiguous[tid]);
    } else {
        uint slot;
        uint offset;
        uint expert;
        if (mode == 1 || (mode >= 4 && mode <= 7)) {
            uint payload_bytes = packed_stride * fraction / 4;
            slot = tid / payload_bytes;
            offset = tid % payload_bytes;
            expert = expert_ids[slot];
            if (expert >= 32) return;
            value = uint(packed[expert * packed_stride + offset]);
        } else if (mode == 2) {
            slot = tid / scale_bytes;
            offset = tid % scale_bytes;
            expert = expert_ids[slot];
            if (expert >= 32) return;
            value = uint(scales[expert * (scale_bytes / 2) + offset / 2]);
        } else {
            slot = tid / expert_bytes;
            offset = tid % expert_bytes;
            expert = expert_ids[slot];
            if (expert >= 32) return;
            if (offset < packed_stride) {
                value = uint(packed[expert * packed_stride + offset]);
            } else {
                uint scale_offset = (offset - packed_stride) / 2;
                value = uint(scales[expert * (scale_bytes / 2) + scale_offset]);
            }
        }
    }
    sink[tid % 1024] = value;
}

// Specialized full selected-expert touch path. Keeping this separate from
// the mode ladder avoids introducing a per-byte mode branch into the control
// used for the primary same-bytes comparison.
kernel void qx_dynamic_q4_moe_gate_up_touch_full(
    device const uchar* packed [[buffer(0)]],
    device const ushort* scales [[buffer(1)]],
    device const uint* expert_ids [[buffer(5)]],
    device uint* sink [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint packed_stride = 1024 * 1024 / 2;
    constexpr uint scale_bytes = 1024 * 32 * 2;
    constexpr uint expert_bytes = packed_stride + scale_bytes;
    constexpr uint total_bytes = 8 * expert_bytes;
    if (tid >= total_bytes) return;
    uint slot = tid / expert_bytes;
    uint offset = tid % expert_bytes;
    uint expert = expert_ids[slot];
    if (expert >= 32) return;
    uint value;
    if (offset < packed_stride) {
        value = uint(packed[expert * packed_stride + offset]);
    } else {
        uint scale_offset = (offset - packed_stride) / 2;
        value = uint(scales[expert * (scale_bytes / 2) + scale_offset]);
    }
    sink[tid % 1024] = value;
}

// Direct-address traffic controls. The selected expert base addresses are
// bound by the host before dispatch, so these kernels do not read expert IDs
// or perform routed-address lookup inside the byte traversal. Each selected
// region is repeated as necessary to preserve the same total eight-expert
// traffic volume as the variable region-count controls.
inline uint qx_dynamic_q4_moe_direct_touch_value(
    device const uchar* packed,
    device const ushort* scales,
    uint offset) {
    constexpr uint packed_stride = 1024 * 1024 / 2;
    if (offset < packed_stride) return uint(packed[offset]);
    return uint(scales[(offset - packed_stride) / 2]);
}

kernel void qx_dynamic_q4_moe_gate_up_touch_direct_1(
    device const uchar* packed0 [[buffer(0)]],
    device const ushort* scales0 [[buffer(1)]],
    device uint* sink [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint expert_bytes = 1024 * 1024 / 2 + 1024 * 32 * 2;
    constexpr uint total_bytes = 8 * expert_bytes;
    if (tid >= total_bytes) return;
    sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(
        packed0, scales0, tid % expert_bytes);
}

kernel void qx_dynamic_q4_moe_gate_up_touch_direct_2(
    device const uchar* packed0 [[buffer(0)]],
    device const ushort* scales0 [[buffer(1)]],
    device const uchar* packed1 [[buffer(2)]],
    device const ushort* scales1 [[buffer(3)]],
    device uint* sink [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint expert_bytes = 1024 * 1024 / 2 + 1024 * 32 * 2;
    constexpr uint region_bytes = 4 * expert_bytes;
    constexpr uint total_bytes = 8 * expert_bytes;
    if (tid >= total_bytes) return;
    uint offset = tid % expert_bytes;
    if (tid / region_bytes == 0) {
        sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed0, scales0, offset);
    } else {
        sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed1, scales1, offset);
    }
}

kernel void qx_dynamic_q4_moe_gate_up_touch_direct_4(
    device const uchar* packed0 [[buffer(0)]],
    device const ushort* scales0 [[buffer(1)]],
    device const uchar* packed1 [[buffer(2)]],
    device const ushort* scales1 [[buffer(3)]],
    device const uchar* packed2 [[buffer(4)]],
    device const ushort* scales2 [[buffer(5)]],
    device const uchar* packed3 [[buffer(6)]],
    device const ushort* scales3 [[buffer(7)]],
    device uint* sink [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint expert_bytes = 1024 * 1024 / 2 + 1024 * 32 * 2;
    constexpr uint region_bytes = 2 * expert_bytes;
    constexpr uint total_bytes = 8 * expert_bytes;
    if (tid >= total_bytes) return;
    uint offset = tid % expert_bytes;
    switch (tid / region_bytes) {
        case 0: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed0, scales0, offset); break;
        case 1: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed1, scales1, offset); break;
        case 2: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed2, scales2, offset); break;
        default: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed3, scales3, offset); break;
    }
}

kernel void qx_dynamic_q4_moe_gate_up_touch_direct_8(
    device const uchar* packed0 [[buffer(0)]], device const ushort* scales0 [[buffer(1)]],
    device const uchar* packed1 [[buffer(2)]], device const ushort* scales1 [[buffer(3)]],
    device const uchar* packed2 [[buffer(4)]], device const ushort* scales2 [[buffer(5)]],
    device const uchar* packed3 [[buffer(6)]], device const ushort* scales3 [[buffer(7)]],
    device const uchar* packed4 [[buffer(8)]], device const ushort* scales4 [[buffer(9)]],
    device const uchar* packed5 [[buffer(10)]], device const ushort* scales5 [[buffer(11)]],
    device const uchar* packed6 [[buffer(12)]], device const ushort* scales6 [[buffer(13)]],
    device const uchar* packed7 [[buffer(14)]], device const ushort* scales7 [[buffer(15)]],
    device uint* sink [[buffer(16)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint expert_bytes = 1024 * 1024 / 2 + 1024 * 32 * 2;
    constexpr uint total_bytes = 8 * expert_bytes;
    if (tid >= total_bytes) return;
    uint offset = tid % expert_bytes;
    switch (tid / expert_bytes) {
        case 0: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed0, scales0, offset); break;
        case 1: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed1, scales1, offset); break;
        case 2: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed2, scales2, offset); break;
        case 3: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed3, scales3, offset); break;
        case 4: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed4, scales4, offset); break;
        case 5: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed5, scales5, offset); break;
        case 6: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed6, scales6, offset); break;
        default: sink[tid % 1024] = qx_dynamic_q4_moe_direct_touch_value(packed7, scales7, offset); break;
    }
}

// Benchmark-only arithmetic control. It preserves the canonical 8-expert ×
// 1024-row output geometry and reduction shape, but replaces Q4 payload loads
// and unpacking with deterministic synthetic weights.
kernel void qx_dynamic_q4_moe_gate_up_synthetic(
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        for (uint column = lane; column < hidden; column += 32) {
            float synthetic_weight = float(int((row + column) & 31) - 16) * 0.03125f;
            sum += synthetic_weight * input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_gate_up_synthetic_norm(
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const float* norm_weights [[buffer(8)]],
    device const float* inverse_rms [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float inverse = inverse_rms[0];
    float sum = 0.0f;
    if (expert < 32) {
        for (uint column = lane; column < hidden; column += 32) {
            float normalized = input[column] * inverse * norm_weights[column];
            float synthetic_weight = float(int((row + column) & 31) - 16) * 0.03125f;
            sum += synthetic_weight * normalized;
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Split gate/up projection used by the topology experiment. Each dispatch
// owns one 512-row projection, which keeps the output-row geometry identical
// for gate and up instead of making one kernel cover 1024 merged rows.
kernel void qx_dynamic_q4_moe_projection_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    constant uint& output_base [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint rows = 512;
    constexpr uint groups = 32;
    constexpr uint bytes_per_group = 16;
    constexpr uint packed_stride = rows * hidden / 2;
    constexpr uint scale_stride = rows * groups;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = rows / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32 && row < rows) {
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * scale_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base
                + qx_dynamic_group_offset(row, group, groups, bytes_per_group, layout);
            uchar byte = packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = scale_base + qx_dynamic_scale_offset(row, group, groups, layout);
            sum += float(int(code) - 7) * float(scales[scale]) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0 && row < rows) {
        output[slot * 1024 + output_base + row] = sum;
    }
}

kernel void qx_dynamic_q4_moe_projection_norm_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    constant uint& output_base [[buffer(8)]],
    device const float* norm_weights [[buffer(9)]],
    device const float* inverse_rms [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint rows = 512;
    constexpr uint groups = 32;
    constexpr uint bytes_per_group = 16;
    constexpr uint packed_stride = rows * hidden / 2;
    constexpr uint scale_stride = rows * groups;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = rows / rows_per_threadgroup;
    threadgroup float cached_input[hidden];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < hidden; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32 && row < rows) {
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * scale_stride;
        for (uint column = lane; column < hidden; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base
                + qx_dynamic_group_offset(row, group, groups, bytes_per_group, layout);
            uchar byte = packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = scale_base + qx_dynamic_scale_offset(row, group, groups, layout);
            sum += float(int(code) - 7) * float(scales[scale]) * cached_input[column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0 && row < rows) {
        output[slot * 1024 + output_base + row] = sum;
    }
}

kernel void qx_dynamic_q4_moe_silu(
    device const float* gate_up [[buffer(0)]],
    device float* activated [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint intermediate = 512;
    uint slot = tid / intermediate;
    uint row = tid % intermediate;
    uint base = slot * 2 * intermediate;
    float gate = gate_up[base + row];
    float up = gate_up[base + intermediate + row];
    activated[slot * intermediate + row] = (gate / (1.0f + exp(-gate))) * up;
}

kernel void qx_dynamic_q4_moe_down_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint bytes_per_group = 16;
    uint packed_stride = qx_dynamic_packed_stride(hidden, groups, layout);
    uint scale_stride = qx_dynamic_scale_stride(hidden, groups, layout);
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * scale_stride;
        for (uint column = lane; column < intermediate; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base + qx_dynamic_group_offset(row, group, groups, bytes_per_group, layout);
            uchar byte = packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = scale_base + qx_dynamic_scale_offset(row, group, groups, layout);
            sum += float(int(code) - 7) * float(scales[scale]) * activated[slot * intermediate + column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_down_inline_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint column = lane; column < intermediate; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint group_base = row_base + group * group_stride;
            uint code = (uint(stream[group_base + local / 2]) >> ((local % 2) * 4)) & 0xf;
            sum += float(int(code) - 7)
                * qx_q4_inline_scale(stream, group_base + 16)
                * activated[slot * intermediate + column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

// Exact-Q4 cooperative packet consumer for the row-inline down projection.
// Down has 512 input values (16 group-32 packets), so each SIMD group owns
// two adjacent packets just as the proven gate/up consumer does.  Lanes 0..15
// load one packed byte from the first packet and lanes 16..31 load the
// corresponding byte from the second; packet scales are loaded once and
// broadcast within each half.  The arithmetic and output ownership remain
// identical to qx_dynamic_q4_moe_down_inline_simd.
inline void qx_dynamic_q4_down_inline_unique_packet_body(
    device const uchar* stream,
    device const float* activated,
    device const uint* expert_ids,
    device float* output,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32 && row < hidden) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            uint group_base = row_base + selected_group * group_stride;
            uint packed = uint(stream[group_base + local]);
            float scale = 0.0f;
            if (local == 0u) {
                scale = qx_q4_inline_scale(stream, group_base + 16u);
            }
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * 32u + local * 2u;
            uint lo = packed & 0xfu;
            uint hi = (packed >> 4u) & 0xfu;
            sum += float(int(lo) - 7) * selected_scale
                * activated[slot * intermediate + input_base];
            sum += float(int(hi) - 7) * selected_scale
                * activated[slot * intermediate + input_base + 1u];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0 && expert < 32 && row < hidden) {
        output[slot * hidden + row] = sum;
    }
}

kernel void qx_dynamic_q4_moe_down_inline_unique_packet_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4_down_inline_unique_packet_body(
        stream, activated, expert_ids, output, lane, simdgroup, threadgroup_id);
}

// Down-side neuron-cluster pre-screen. This keeps the existing row-inline
// storage and output ownership, but changes the accumulation order to consume
// the activated vector in C-wide intermediate-neuron clusters. It is a
// deliberately conservative first test: no new artifact packing or fused
// Gate/Up path is implied. The two widths are compile-time variants so the
// benchmark can reject the Down-side idea before a cluster format is built.
inline void qx_dynamic_q4_down_cluster_body(
    device const uchar* stream,
    device const float* activated,
    device const uint* expert_ids,
    device float* output,
    uint lane,
    uint simdgroup,
    uint threadgroup_id,
    uint cluster_width) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32 && row < hidden) {
        uint row_base = expert * expert_stride + row * row_stride;
        for (uint cluster = 0; cluster < intermediate; cluster += cluster_width) {
            uint cluster_end = min(cluster + cluster_width, intermediate);
            for (uint column = cluster + lane; column < cluster_end; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                uint code = (uint(stream[group_base + local / 2])
                    >> ((local % 2) * 4)) & 0xf;
                sum += float(int(code) - 7)
                    * qx_q4_inline_scale(stream, group_base + 16)
                    * activated[slot * intermediate + column];
            }
        }
    }
    sum = simd_sum(sum);
    if (lane == 0 && expert < 32 && row < hidden) {
        output[slot * hidden + row] = sum;
    }
}

kernel void qx_dynamic_q4_moe_down_cluster16_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4_down_cluster_body(
        stream, activated, expert_ids, output, lane, simdgroup, threadgroup_id, 16);
}

kernel void qx_dynamic_q4_moe_down_cluster32_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    qx_dynamic_q4_down_cluster_body(
        stream, activated, expert_ids, output, lane, simdgroup, threadgroup_id, 32);
}

// Decode-direct exact-Q4 down projection. It mirrors the gate/up geometry:
// two SIMD groups per 64-thread threadgroup and one row per group.
// Activated values and row-inline QX packets are consumed directly; no
// threadgroup activation staging or route compaction is introduced.
kernel void qx_dynamic_q4_moe_down_decode_direct_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint group_stride = 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint expert_stride = hidden * row_stride;
    constexpr uint rows_per_simdgroup = 1;
    constexpr uint simdgroups_per_threadgroup = 2;
    constexpr uint rows_per_threadgroup = rows_per_simdgroup * simdgroups_per_threadgroup;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;

    uint slot = threadgroup_id / threadgroups_per_expert;
    uint tile = threadgroup_id % threadgroups_per_expert;
    uint first_row = tile * rows_per_threadgroup + simdgroup * rows_per_simdgroup;
    uint expert = expert_ids[slot];

    for (uint row_offset = 0; row_offset < rows_per_simdgroup; ++row_offset) {
        uint row = first_row + row_offset;
        float sum = 0.0f;
        if (expert < 32 && row < hidden) {
            uint row_base = expert * expert_stride + row * row_stride;
            for (uint column = lane; column < intermediate; column += 32) {
                uint group = column / 32;
                uint local = column % 32;
                uint group_base = row_base + group * group_stride;
                uint code = (uint(stream[group_base + local / 2])
                    >> ((local % 2) * 4)) & 0xf;
                sum += float(int(code) - 7)
                    * qx_q4_inline_scale(stream, group_base + 16)
                    * activated[slot * intermediate + column];
            }
        }
        sum = simd_sum(sum);
        if (lane == 0 && expert < 32 && row < hidden) {
            output[slot * hidden + row] = sum;
        }
    }
}

// Down-projection consumer for the same shared cross-expert tile-8 stream.
// The only shape change from gate/up is the 512-wide activation and sixteen
// quantization groups per output row.
kernel void qx_dynamic_q4_moe_down_cross_expert_tile8_simd(
    device const uchar* stream [[buffer(0)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint tile_rows = 8;
    constexpr uint experts = 32;
    constexpr uint group_stride = experts * 18;
    constexpr uint row_stride = groups * group_stride;
    constexpr uint tile_stride = tile_rows * row_stride;
    threadgroup uint selected_experts[8];
    if (thread_index < tile_rows) {
        selected_experts[thread_index] = expert_ids[thread_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint expert = selected_experts[simdgroup];
    uint first_row = threadgroup_id * tile_rows;
    for (uint row_offset = 0; row_offset < tile_rows; ++row_offset) {
        uint row = first_row + row_offset;
        float sum = 0.0f;
        if (expert < experts && row < hidden) {
            uint tile_base = threadgroup_id * tile_stride;
            for (uint group = 0; group < groups; ++group) {
                uint group_base = tile_base + group * tile_stride / tile_rows
                    + row_offset * group_stride + expert * 18;
                ushort packed = reinterpret_cast<device const ushort*>(stream + group_base)[lane >> 2];
                uint byte = (uint(packed) >> (((lane >> 1) & 1) * 8)) & 0xff;
                uint code = (byte >> ((lane & 1) * 4)) & 0xf;
                float scale = lane == 0
                    ? qx_q4_inline_scale(stream, group_base + 16)
                    : 0.0f;
                scale = simd_broadcast(scale, 0);
                uint column = group * 32 + lane;
                if (column < intermediate) {
                    sum += float(int(code) - 7)
                        * scale
                        * activated[simdgroup * intermediate + column];
                }
            }
        }
        sum = simd_sum(sum);
        if (lane == 0 && row < hidden) {
            output[simdgroup * hidden + row] = sum;
        }
    }
}

// Wide-expert down projection paired with the gate/up candidate above.
kernel void qx_dynamic_q4_moe_down_wide_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& layout [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint groups = 16;
    constexpr uint bytes_per_group = 16;
    uint packed_stride = qx_dynamic_packed_stride(hidden, groups, layout);
    uint scale_stride = qx_dynamic_scale_stride(hidden, groups, layout);
    threadgroup float cached_activated[intermediate];
    uint slot = threadgroup_id;
    for (uint column = thread_index; column < intermediate; column += 1024) {
        cached_activated[column] = activated[slot * intermediate + column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint expert = expert_ids[slot];
    if (expert >= 32) return;
    uint packed_base = expert * packed_stride;
    uint scale_base = expert * scale_stride;
    for (uint row = simdgroup; row < hidden; row += 32) {
        float sum = 0.0f;
        for (uint column = lane; column < intermediate; column += 32) {
            uint group = column / 32;
            uint local = column % 32;
            uint offset = packed_base
                + qx_dynamic_group_offset(row, group, groups, bytes_per_group, layout);
            uchar byte = packed[offset + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            uint scale = scale_base + qx_dynamic_scale_offset(row, group, groups, layout);
            sum += float(int(code) - 7) * float(scales[scale]) * cached_activated[column];
        }
        sum = simd_sum(sum);
        if (lane == 0) output[slot * hidden + row] = sum;
    }
}

kernel void qx_dynamic_q4k_moe_down_simd(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* activated [[buffer(4)]],
    device const uint* expert_ids [[buffer(5)]],
    device float* output [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint hidden = 1024;
    constexpr uint intermediate = 512;
    constexpr uint blocks = 2;
    constexpr uint packed_stride = hidden * blocks * 140;
    constexpr uint scale_stride = hidden * blocks * 2;
    constexpr uint rows_per_threadgroup = 8;
    constexpr uint threadgroups_per_expert = hidden / rows_per_threadgroup;
    uint slot = threadgroup_id / threadgroups_per_expert;
    uint row = (threadgroup_id % threadgroups_per_expert) * rows_per_threadgroup + simdgroup;
    uint expert = expert_ids[slot];
    float sum = 0.0f;
    if (expert < 32) {
        uint packed_base = expert * packed_stride;
        uint scale_base = expert * scale_stride;
        for (uint column = lane; column < intermediate; column += 32) {
            sum += qx_q4k_weight(packed, scales, packed_base, scale_base, row, blocks, column)
                * activated[slot * intermediate + column];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) output[slot * hidden + row] = sum;
}

kernel void qx_dynamic_q4_moe_accumulate(
    device const float* expert_output [[buffer(0)]],
    device const float* expert_weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    float value = 0.0f;
    for (uint slot = 0; slot < 8; ++slot) {
        value += expert_weights[slot] * expert_output[slot * 1024 + row];
    }
    output[row] = value;
}
