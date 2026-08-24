#include <metal_stdlib>
using namespace metal;

// Row-major Q4 dense GEMV for the fixed Granite decode shape. The packed
// matrix and FP16 scales remain compressed; no dequantized matrix is written.
kernel void qx_q4_dense_gemv(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant float& multiplier [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    uint groups = (columns + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = 0; column < columns; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
        sum += weight * input[column];
    }
    output[row] = sum * multiplier;
}

// SIMD-cooperative control variant. One SIMD group owns one output row and
// reduces the K dimension across lanes; eight rows run concurrently in the
// fixed 256-thread threadgroup. This is intentionally separate from the
// production path so the microbenchmark can compare parallelization alone.
kernel void qx_q4_dense_simd_gemv(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant float& multiplier [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    constexpr uint rows_per_threadgroup = 8;
    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    if (row >= rows) return;
    uint groups = (columns + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 32) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
        sum += weight * input[column];
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        output[row] = sum * multiplier;
    }
}

// Exact greedy-decode vocabulary projection. Each SIMD group computes one
// vocabulary row, and the threadgroup emits only its best (value, token)
// pair. This avoids materializing and rereading the full logits vector when
// the caller only needs the next greedy token.
struct qx_q4_argmax_partial_t {
    float value;
    uint index;
};

kernel void qx_q4_dense_argmax_partial(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device qx_q4_argmax_partial_t* partials [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float values[rows_per_threadgroup];
    threadgroup uint indices[rows_per_threadgroup];

    if (local_tid < rows_per_threadgroup) {
        values[local_tid] = -3.402823466e+38f;
        indices[local_tid] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    float sum = 0.0f;
    if (row < rows) {
        uint groups = (columns + group_size - 1) / group_size;
        for (uint column = lane; column < columns; column += 32) {
            uint group = column / group_size;
            uint local = column % group_size;
            uchar byte = packed[row * groups * bytes_per_group
                + group * bytes_per_group + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
            sum += weight * input[column];
        }
        sum = simd_sum(sum) * multiplier;
    }
    if (lane == 0) {
        values[simdgroup] = row < rows ? sum : -3.402823466e+38f;
        indices[simdgroup] = row < rows ? row : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_tid == 0) {
        float best_value = values[0];
        uint best_index = indices[0];
        for (uint i = 1; i < rows_per_threadgroup; ++i) {
            float value = values[i];
            uint index = indices[i];
            if (value > best_value || (value == best_value && index < best_index)) {
                best_value = value;
                best_index = index;
            }
        }
        partials[threadgroup_id].value = best_value;
        partials[threadgroup_id].index = best_index;
    }
}

// Projection-aware variant of the exact greedy vocabulary path. The final
// RMS scalar and norm weights are consumed while reading the raw hidden
// vector, so no normalized hidden activation buffer is materialized.
kernel void qx_q4_dense_argmax_norm_partial(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device qx_q4_argmax_partial_t* partials [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    device const float* norm_weights [[buffer(7)]],
    device const float* inverse_rms [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float values[rows_per_threadgroup];
    threadgroup uint indices[rows_per_threadgroup];

    if (local_tid < rows_per_threadgroup) {
        values[local_tid] = -3.402823466e+38f;
        indices[local_tid] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    float sum = 0.0f;
    float inverse = inverse_rms[0];
    if (row < rows) {
        uint groups = (columns + group_size - 1) / group_size;
        for (uint column = lane; column < columns; column += 32) {
            uint group = column / group_size;
            uint local = column % group_size;
            uchar byte = packed[row * groups * bytes_per_group
                + group * bytes_per_group + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
            float activation = input[column] * inverse * norm_weights[column];
            sum += weight * activation;
        }
        sum = simd_sum(sum) * multiplier;
    }
    if (lane == 0) {
        values[simdgroup] = row < rows ? sum : -3.402823466e+38f;
        indices[simdgroup] = row < rows ? row : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_tid == 0) {
        float best_value = values[0];
        uint best_index = indices[0];
        for (uint i = 1; i < rows_per_threadgroup; ++i) {
            float value = values[i];
            uint index = indices[i];
            if (value > best_value || (value == best_value && index < best_index)) {
                best_value = value;
                best_index = index;
            }
        }
        partials[threadgroup_id].value = best_value;
        partials[threadgroup_id].index = best_index;
    }
}

// Exact Q4 row-major vocabulary consumer using cooperative packet ownership.
// Each SIMD group still owns one output row and the threadgroup still emits
// one partial maximum, but two adjacent group-32 packets are consumed per
// loop iteration. Packed bytes are loaded once per lane and each packet scale
// is loaded once per half-group and broadcast, avoiding the repeated scale
// loads in the canonical row-major consumer.
kernel void qx_q4_dense_argmax_unique_packet_partial(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device qx_q4_argmax_partial_t* partials [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float values[rows_per_threadgroup];
    threadgroup uint indices[rows_per_threadgroup];
    if (local_tid < rows_per_threadgroup) {
        values[local_tid] = -3.402823466e+38f;
        indices[local_tid] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    float sum = 0.0f;
    if (row < rows) {
        uint groups = (columns + group_size - 1) / group_size;
        uint row_base = row * groups * bytes_per_group;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            if (selected_group >= groups) continue;
            uint packed_base = row_base + selected_group * bytes_per_group;
            uint packed_byte = uint(packed[packed_base + local]);
            float scale = 0.0f;
            if (local == 0u) {
                scale = float(scales[row * groups + selected_group]);
            }
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * group_size + local * 2u;
            uint lo = packed_byte & 0xfu;
            uint hi = (packed_byte >> 4u) & 0xfu;
            if (input_base < columns) {
                sum += float(int(lo) - qmax) * selected_scale * input[input_base];
            }
            if (input_base + 1u < columns) {
                sum += float(int(hi) - qmax) * selected_scale * input[input_base + 1u];
            }
        }
        sum = simd_sum(sum) * multiplier;
    }
    if (lane == 0) {
        values[simdgroup] = row < rows ? sum : -3.402823466e+38f;
        indices[simdgroup] = row < rows ? row : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_tid == 0) {
        float best_value = values[0];
        uint best_index = indices[0];
        for (uint i = 1; i < rows_per_threadgroup; ++i) {
            float value = values[i];
            uint index = indices[i];
            if (value > best_value || (value == best_value && index < best_index)) {
                best_value = value;
                best_index = index;
            }
        }
        partials[threadgroup_id].value = best_value;
        partials[threadgroup_id].index = best_index;
    }
}

kernel void qx_q4_dense_argmax_unique_packet_norm_partial(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device qx_q4_argmax_partial_t* partials [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    device const float* norm_weights [[buffer(7)]],
    device const float* inverse_rms [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float values[rows_per_threadgroup];
    threadgroup uint indices[rows_per_threadgroup];
    if (local_tid < rows_per_threadgroup) {
        values[local_tid] = -3.402823466e+38f;
        indices[local_tid] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    float sum = 0.0f;
    float inverse = inverse_rms[0];
    if (row < rows) {
        uint groups = (columns + group_size - 1) / group_size;
        uint row_base = row * groups * bytes_per_group;
        for (uint group = 0; group < groups; group += 2) {
            uint local = lane & 15u;
            uint selected_group = group + (lane >= 16u ? 1u : 0u);
            if (selected_group >= groups) continue;
            uint packed_base = row_base + selected_group * bytes_per_group;
            uint packed_byte = uint(packed[packed_base + local]);
            float scale = 0.0f;
            if (local == 0u) {
                scale = float(scales[row * groups + selected_group]);
            }
            float scale0 = simd_broadcast(lane == 0u ? scale : 0.0f, 0);
            float scale1 = simd_broadcast(lane == 16u ? scale : 0.0f, 16);
            float selected_scale = lane < 16u ? scale0 : scale1;
            uint input_base = selected_group * group_size + local * 2u;
            uint lo = packed_byte & 0xfu;
            uint hi = (packed_byte >> 4u) & 0xfu;
            if (input_base < columns) {
                float activation = input[input_base] * inverse * norm_weights[input_base];
                sum += float(int(lo) - qmax) * selected_scale * activation;
            }
            if (input_base + 1u < columns) {
                float activation = input[input_base + 1u] * inverse * norm_weights[input_base + 1u];
                sum += float(int(hi) - qmax) * selected_scale * activation;
            }
        }
        sum = simd_sum(sum) * multiplier;
    }
    if (lane == 0) {
        values[simdgroup] = row < rows ? sum : -3.402823466e+38f;
        indices[simdgroup] = row < rows ? row : 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_tid == 0) {
        float best_value = values[0];
        uint best_index = indices[0];
        for (uint i = 1; i < rows_per_threadgroup; ++i) {
            float value = values[i];
            uint index = indices[i];
            if (value > best_value || (value == best_value && index < best_index)) {
                best_value = value;
                best_index = index;
            }
        }
        partials[threadgroup_id].value = best_value;
        partials[threadgroup_id].index = best_index;
    }
}

// SIMD Q4 GEMV that applies RMSNorm projection-wise while consuming the raw
// activation vector. This is used for the post-attention router boundary.
kernel void qx_q4_simd_norm_gemv(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant float& multiplier [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    device const float* norm_weights [[buffer(7)]],
    device const float* inverse_rms [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < columns; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    if (row >= rows) return;
    uint groups = (columns + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 32) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
        sum += weight * cached_input[column];
    }
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum * multiplier;
}

// Parameterized kernel used by the Q-projection geometry sweep. The compiler
// can later specialize the winning constants into the artifact's native
// layout. This version keeps the existing row-major Q4 bytes so the sweep
// measures execution topology independently of repacking.
kernel void qx_q4_dense_sweep_gemv(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant float& multiplier [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& rows_per_simd [[buffer(7)]],
    constant uint& rows_per_threadgroup [[buffer(8)]],
    constant uint& cache_input [[buffer(9)]],
    constant uint& threads_per_group [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    threadgroup float cached_input[1024];
    if (cache_input != 0) {
        for (uint column = thread_index; column < columns; column += threads_per_group) {
            cached_input[column] = input[column];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint first_row = threadgroup_id * rows_per_threadgroup + simdgroup * rows_per_simd;
    uint groups = (columns + group_size - 1) / group_size;
    for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
        uint row = first_row + row_offset;
        if (row >= rows) continue;
        float sum = 0.0f;
        for (uint column = lane; column < columns; column += 32) {
            uint group = column / group_size;
            uint local = column % group_size;
            uchar byte = packed[row * groups * bytes_per_group + group * bytes_per_group + local / 2];
            uint code = (byte >> ((local % 2) * 4)) & 0xf;
            float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
            float activation = cache_input != 0 ? cached_input[column] : input[column];
            sum += weight * activation;
        }
        sum = simd_sum(sum);
        if (lane == 0) output[row] = sum * multiplier;
    }
}

// Fused Granite Q/K/V GEMV. The three projections consume the same normalized
// hidden vector, so one dispatch computes all 1024 Q rows and both 512-row KV
// projections without returning to the command encoder between matrices.
kernel void qx_q4_qkv_gemv(
    device const uchar* q_packed [[buffer(0)]],
    device const half* q_scales [[buffer(1)]],
    device const uchar* k_packed [[buffer(2)]],
    device const half* k_scales [[buffer(3)]],
    device const uchar* v_packed [[buffer(4)]],
    device const half* v_scales [[buffer(5)]],
    device const float* input [[buffer(6)]],
    device float* query [[buffer(7)]],
    device float* key [[buffer(8)]],
    device float* value [[buffer(9)]],
    constant uint& columns [[buffer(10)]],
    constant float& multiplier [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    constexpr uint q_rows = 1024;
    constexpr uint kv_rows = 512;
    if (tid >= q_rows + 2 * kv_rows) return;

    device const uchar* packed;
    device const half* scales;
    device float* output;
    uint row;
    if (tid < q_rows) {
        packed = q_packed;
        scales = q_scales;
        output = query;
        row = tid;
    } else if (tid < q_rows + kv_rows) {
        packed = k_packed;
        scales = k_scales;
        output = key;
        row = tid - q_rows;
    } else {
        packed = v_packed;
        scales = v_scales;
        output = value;
        row = tid - q_rows - kv_rows;
    }

    uint groups = (columns + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = 0; column < columns; ++column) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group
            + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
        sum += weight * input[column];
    }
    output[row] = sum * multiplier;
}

// SIMD-cooperative fused QKV. Eight output rows are assigned to each
// threadgroup and each SIMD group reduces the shared input vector over K.
kernel void qx_q4_qkv_simd_gemv(
    device const uchar* q_packed [[buffer(0)]],
    device const half* q_scales [[buffer(1)]],
    device const uchar* k_packed [[buffer(2)]],
    device const half* k_scales [[buffer(3)]],
    device const uchar* v_packed [[buffer(4)]],
    device const half* v_scales [[buffer(5)]],
    device const float* input [[buffer(6)]],
    device float* query [[buffer(7)]],
    device float* key [[buffer(8)]],
    device float* value [[buffer(9)]],
    constant uint& columns [[buffer(10)]],
    constant float& multiplier [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint q_rows = 1024;
    constexpr uint kv_rows = 512;
    constexpr uint total_rows = q_rows + 2 * kv_rows;
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < columns; column += 256) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tid = threadgroup_id * rows_per_threadgroup + simdgroup;
    if (tid >= total_rows) return;
    device const uchar* packed;
    device const half* scales;
    device float* output;
    uint row;
    uint rows;
    if (tid < q_rows) {
        packed = q_packed;
        scales = q_scales;
        output = query;
        row = tid;
        rows = q_rows;
    } else if (tid < q_rows + kv_rows) {
        packed = k_packed;
        scales = k_scales;
        output = key;
        row = tid - q_rows;
        rows = kv_rows;
    } else {
        packed = v_packed;
        scales = v_scales;
        output = value;
        row = tid - q_rows - kv_rows;
        rows = kv_rows;
    }
    uint groups = (columns + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 32) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        sum += float(int(code) - 7) * float(scales[row * groups + group]) * cached_input[column];
    }
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum * multiplier;
}

// SIMD fused QKV with topology-B RMSNorm: the original hidden vector is
// cached once, while each lane applies the shared inverse RMS and the
// per-element norm weight before the Q4 dot product.
kernel void qx_q4_qkv_simd_norm_gemv(
    device const uchar* q_packed [[buffer(0)]],
    device const half* q_scales [[buffer(1)]],
    device const uchar* k_packed [[buffer(2)]],
    device const half* k_scales [[buffer(3)]],
    device const uchar* v_packed [[buffer(4)]],
    device const half* v_scales [[buffer(5)]],
    device const float* input [[buffer(6)]],
    device float* query [[buffer(7)]],
    device float* key [[buffer(8)]],
    device float* value [[buffer(9)]],
    constant uint& columns [[buffer(10)]],
    constant float& multiplier [[buffer(11)]],
    device const float* norm_weights [[buffer(12)]],
    device const float* inverse_rms [[buffer(13)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    constexpr uint q_rows = 1024;
    constexpr uint kv_rows = 512;
    constexpr uint total_rows = q_rows + 2 * kv_rows;
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr uint rows_per_threadgroup = 8;
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < columns; column += 256) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint tid = threadgroup_id * rows_per_threadgroup + simdgroup;
    if (tid >= total_rows) return;
    device const uchar* packed;
    device const half* scales;
    device float* output;
    uint row;
    if (tid < q_rows) {
        packed = q_packed;
        scales = q_scales;
        output = query;
        row = tid;
    } else if (tid < q_rows + kv_rows) {
        packed = k_packed;
        scales = k_scales;
        output = key;
        row = tid - q_rows;
    } else {
        packed = v_packed;
        scales = v_scales;
        output = value;
        row = tid - q_rows - kv_rows;
    }
    uint groups = (columns + group_size - 1) / group_size;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 32) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        sum += float(int(code) - 7) * float(scales[row * groups + group]) * cached_input[column];
    }
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum * multiplier;
}

// Benchmark-only fused QKV geometry sweep. One dispatch still covers the
// heterogeneous Q/K/V output regions, while the compiler varies how many
// rows each SIMD group owns in the Q region and in the two KV regions. This
// keeps the candidate inside the fused execution boundary while exposing the
// occupancy choices that the fixed 8-row production kernel hides.
kernel void qx_q4_qkv_sweep_gemv(
    device const uchar* q_packed [[buffer(0)]],
    device const half* q_scales [[buffer(1)]],
    device const uchar* k_packed [[buffer(2)]],
    device const half* k_scales [[buffer(3)]],
    device const uchar* v_packed [[buffer(4)]],
    device const half* v_scales [[buffer(5)]],
    device const float* input [[buffer(6)]],
    device float* query [[buffer(7)]],
    device float* key [[buffer(8)]],
    device float* value [[buffer(9)]],
    constant uint& columns [[buffer(10)]],
    constant float& multiplier [[buffer(11)]],
    constant uint& q_rows_per_simd [[buffer(12)]],
    constant uint& kv_rows_per_simd [[buffer(13)]],
    constant uint& simd_groups [[buffer(14)]],
    constant uint& cache_input [[buffer(15)]],
    constant uint& normalize [[buffer(16)]],
    device const float* norm_weights [[buffer(17)]],
    constant float& inverse_rms [[buffer(18)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint q_rows = 1024;
    constexpr uint kv_rows = 512;
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    uint q_threadgroups = (q_rows + q_rows_per_simd * simd_groups - 1)
        / (q_rows_per_simd * simd_groups);
    uint kv_threadgroups = (kv_rows + kv_rows_per_simd * simd_groups - 1)
        / (kv_rows_per_simd * simd_groups);
    uint region = 0;
    uint local_group = threadgroup_id;
    uint rows_per_simd = q_rows_per_simd;
    if (threadgroup_id >= q_threadgroups) {
        region = 1 + (threadgroup_id - q_threadgroups) / kv_threadgroups;
        local_group = (threadgroup_id - q_threadgroups) % kv_threadgroups;
        rows_per_simd = kv_rows_per_simd;
    }
    uint row_base = local_group * rows_per_simd * simd_groups
        + simdgroup * rows_per_simd;

    threadgroup float cached_input[1024];
    if (cache_input != 0) {
        for (uint column = thread_index; column < columns; column += simd_groups * 32) {
            float activation = input[column];
            if (normalize != 0) activation *= inverse_rms * norm_weights[column];
            cached_input[column] = activation;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint groups = (columns + group_size - 1) / group_size;
    if (region == 0) {
        for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
            uint current_row = row_base + row_offset;
            if (current_row >= q_rows) continue;
            float sum = 0.0f;
            for (uint column = lane; column < columns; column += 32) {
                uint group = column / group_size;
                uint local = column % group_size;
                uchar byte = q_packed[current_row * groups * bytes_per_group
                    + group * bytes_per_group + local / 2];
                uint code = (byte >> ((local % 2) * 4)) & 0xf;
                float activation = cache_input != 0
                    ? cached_input[column]
                    : input[column] * (normalize != 0 ? inverse_rms * norm_weights[column] : 1.0f);
                sum += float(int(code) - qmax) * float(q_scales[current_row * groups + group]) * activation;
            }
            sum = simd_sum(sum);
            if (lane == 0) query[current_row] = sum * multiplier;
        }
    } else if (region == 1) {
        for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
            uint current_row = row_base + row_offset;
            if (current_row >= kv_rows) continue;
            float sum = 0.0f;
            for (uint column = lane; column < columns; column += 32) {
                uint group = column / group_size;
                uint local = column % group_size;
                uchar byte = k_packed[current_row * groups * bytes_per_group
                    + group * bytes_per_group + local / 2];
                uint code = (byte >> ((local % 2) * 4)) & 0xf;
                float activation = cache_input != 0
                    ? cached_input[column]
                    : input[column] * (normalize != 0 ? inverse_rms * norm_weights[column] : 1.0f);
                sum += float(int(code) - qmax) * float(k_scales[current_row * groups + group]) * activation;
            }
            sum = simd_sum(sum);
            if (lane == 0) key[current_row] = sum * multiplier;
        }
    } else {
        for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
            uint current_row = row_base + row_offset;
            if (current_row >= kv_rows) continue;
            float sum = 0.0f;
            for (uint column = lane; column < columns; column += 32) {
                uint group = column / group_size;
                uint local = column % group_size;
                uchar byte = v_packed[current_row * groups * bytes_per_group
                    + group * bytes_per_group + local / 2];
                uint code = (byte >> ((local % 2) * 4)) & 0xf;
                float activation = cache_input != 0
                    ? cached_input[column]
                    : input[column] * (normalize != 0 ? inverse_rms * norm_weights[column] : 1.0f);
                sum += float(int(code) - qmax) * float(v_scales[current_row * groups + group]) * activation;
            }
            sum = simd_sum(sum);
            if (lane == 0) value[current_row] = sum * multiplier;
        }
    }
}
