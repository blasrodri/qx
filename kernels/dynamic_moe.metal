#include <metal_stdlib>
using namespace metal;

inline uint qx_dynamic_group_offset(uint row, uint group, uint groups, uint bytes_per_group, uint layout) {
    if (layout == 0) {
        return row * groups * bytes_per_group + group * bytes_per_group;
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
    uint physical_row = row;
    if (layout == 2 && groups == 32) {
        physical_row = row < 512 ? row * 2 : (row - 512) * 2 + 1;
    }
    return (physical_row / 32) * 32 * groups + group * 32 + physical_row % 32;
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
    constexpr uint gate_up_packed_stride = 1024 * 1024 / 2;
    constexpr uint gate_up_scale_stride = 1024 * 32;
    constexpr uint down_packed_stride = 1024 * 512 / 2;
    constexpr uint down_scale_stride = 1024 * 16;
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
    constexpr uint gate_up_packed_stride = 1024 * 1024 / 2;
    constexpr uint gate_up_scale_stride = 1024 * 32;
    constexpr uint down_packed_stride = 1024 * 512 / 2;
    constexpr uint down_scale_stride = 1024 * 16;
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
    constexpr uint packed_stride = 1024 * 1024 / 2;
    constexpr uint scale_stride = 1024 * 32;
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
    constexpr uint packed_stride = 1024 * 1024 / 2;
    constexpr uint scale_stride = 1024 * 32;
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
    constexpr uint packed_stride = 1024 * 512 / 2;
    constexpr uint scale_stride = 1024 * 16;
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
