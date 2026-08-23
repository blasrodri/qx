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
