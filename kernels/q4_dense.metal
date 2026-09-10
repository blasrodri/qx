#include <metal_stdlib>
using namespace metal;

// Consensus reduction for QX_EXEC_METAL_V1. Lane 0 observes the fixed tree
// (((0+16)+(8+24))+...) and broadcasts that exact FP32 result.
inline float qx_ordered_simd_sum(float value) {
    value = value + simd_shuffle_down(value, 16u);
    value = value + simd_shuffle_down(value, 8u);
    value = value + simd_shuffle_down(value, 4u);
    value = value + simd_shuffle_down(value, 2u);
    value = value + simd_shuffle_down(value, 1u);
    return simd_broadcast(value, 0u);
}

inline float qx_ordered_simd_max(float value) {
    value = max(value, simd_shuffle_down(value, 16u));
    value = max(value, simd_shuffle_down(value, 8u));
    value = max(value, simd_shuffle_down(value, 4u));
    value = max(value, simd_shuffle_down(value, 2u));
    value = max(value, simd_shuffle_down(value, 1u));
    return simd_broadcast(value, 0u);
}

kernel void qx_residual_rms_scalar(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    device float* inverse_rms [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8];
    float square_sum = 0.0f;
    for (uint i = tid; i < 1024u; i += 256u) {
        float value = left[i] + scale * right[i];
        output[i] = value;
        square_sum += value * value;
    }
    square_sum = qx_ordered_simd_sum(square_sum);
    if (lane == 0u) partials[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        float total = lane < 8u ? partials[lane] : 0.0f;
        total = qx_ordered_simd_sum(total);
        if (lane == 0u) inverse_rms[0] = qx_consensus_rsqrt(total / 1024.0f + 1.0e-5f);
    }
}

kernel void qx_residual_rms_scalar_snapshot(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    device float* inverse_rms [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    device float* snapshot [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8];
    float square_sum = 0.0f;
    for (uint i = tid; i < 1024u; i += 256u) {
        float value = left[i] + scale * right[i];
        output[i] = value;
        snapshot[i] = value;
        square_sum += value * value;
    }
    square_sum = qx_ordered_simd_sum(square_sum);
    if (lane == 0u) partials[simdgroup] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u) {
        float total = lane < 8u ? partials[lane] : 0.0f;
        total = qx_ordered_simd_sum(total);
        if (lane == 0u) inverse_rms[0] = qx_consensus_rsqrt(total / 1024.0f + 1.0e-5f);
    }
}

kernel void qx_attention_softmax_onepass_simd(
    device float* scores [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint q_head [[threadgroup_position_in_grid]]) {
    constexpr uint max_context = 512;
    threadgroup float reductions[8];
    threadgroup float maximum;
    threadgroup float denominator;
    float local_max = -INFINITY;
    for (uint position = tid; position < length; position += 256u) {
        local_max = max(local_max, scores[q_head * max_context + position]);
    }
    local_max = qx_ordered_simd_max(local_max);
    if (lane == 0u) reductions[simdgroup] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u && lane == 0u) {
        float value = -INFINITY;
        for (uint i = 0; i < 8u; ++i) value = max(value, reductions[i]);
        maximum = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint position = tid; position < length; position += 256u) {
        local_sum += qx_consensus_exp(scores[q_head * max_context + position] - maximum);
    }
    local_sum = qx_ordered_simd_sum(local_sum);
    if (lane == 0u) reductions[simdgroup] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0u && lane == 0u) {
        float value = 0.0f;
        for (uint i = 0; i < 8u; ++i) value += reductions[i];
        denominator = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint position = tid; position < length; position += 256u) {
        scores[q_head * max_context + position] =
            qx_consensus_exp(scores[q_head * max_context + position] - maximum) / denominator;
    }
}

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
    sum = qx_ordered_simd_sum(sum);
    if (lane == 0) {
        output[row] = sum * multiplier;
    }
}

// Exact-Q4 payload-contiguous dense projection. Each row is stored as
// [128 code bytes][8 FP16 scales] for every 256 columns. A lane consumes one
// aligned uint (eight weights), four lanes share one scale, and a SIMD group
// computes two rows while reusing the same activation registers.
inline float qx_q4_dense_superblock_dot8(
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

inline void qx_q4_dense_superblock_body(
    device const uchar* stream,
    device float* output,
    threadgroup const float* cached_input,
    uint columns,
    uint rows,
    float multiplier,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint block_bytes = 144;
    constexpr uint rows_per_threadgroup = 16;
    uint blocks = columns / 256u;
    uint row_bytes = blocks * block_bytes;
    uint row0 = threadgroup_id * rows_per_threadgroup + simdgroup * 2u;
    uint row1 = row0 + 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    if (row0 < rows) {
        for (uint block = 0; block < blocks; ++block) {
            uint column = block * 256u + lane * 8u;
            threadgroup const float4* values =
                reinterpret_cast<threadgroup const float4*>(cached_input + column);
            float4 x0 = values[0];
            float4 x1 = values[1];
            uint base0 = row0 * row_bytes + block * block_bytes;
            uint packed0 = reinterpret_cast<device const uint*>(stream + base0)[lane];
            float owned_scale0 = lane < 8u
                ? float(reinterpret_cast<device const half*>(stream + base0 + 128u)[lane])
                : 0.0f;
            float scale0 = simd_shuffle(owned_scale0, ushort(lane >> 2u));
            sum0 = qx_q4_dense_superblock_dot8(packed0, x0, x1, scale0, sum0);
            if (row1 < rows) {
                uint base1 = base0 + row_bytes;
                uint packed1 = reinterpret_cast<device const uint*>(stream + base1)[lane];
                float owned_scale1 = lane < 8u
                    ? float(reinterpret_cast<device const half*>(stream + base1 + 128u)[lane])
                    : 0.0f;
                float scale1 = simd_shuffle(owned_scale1, ushort(lane >> 2u));
                sum1 = qx_q4_dense_superblock_dot8(packed1, x0, x1, scale1, sum1);
            }
        }
    }
    sum0 = qx_ordered_simd_sum(sum0);
    sum1 = qx_ordered_simd_sum(sum1);
    if (lane == 0u && row0 < rows) {
        output[row0] = sum0 * multiplier;
        if (row1 < rows) output[row1] = sum1 * multiplier;
    }
}

kernel void qx_q4_dense_superblock144_simd_gemv(
    device const uchar* stream [[buffer(0)]],
    device const uchar* ignored_scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant float& multiplier [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < columns; column += 256u) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_q4_dense_superblock_body(
        stream, output, cached_input, columns, rows, multiplier,
        lane, simdgroup, threadgroup_id);
}

kernel void qx_q4_dense_superblock144_norm_simd_gemv(
    device const uchar* stream [[buffer(0)]],
    device const uchar* ignored_scales [[buffer(1)]],
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
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < columns; column += 256u) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_q4_dense_superblock_body(
        stream, output, cached_input, columns, rows, multiplier,
        lane, simdgroup, threadgroup_id);
}

// K-wide verifier projection. Inputs and outputs are token-major:
// input[token * columns + column] and output[token * rows + row]. A
// threadgroup owns eight output rows for one candidate token, so the packed
// weight stream is reused across K threadgroups without calling the scalar
// decode graph K times.
kernel void qx_q4_dense_simd_gemm_k(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant float& multiplier [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& tokens [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    constexpr uint rows_per_threadgroup = 8;
    uint token = threadgroup_id.y;
    uint row = threadgroup_id.x * rows_per_threadgroup + simdgroup;
    if (token >= tokens || row >= rows) return;

    uint groups = (columns + group_size - 1) / group_size;
    uint packed_row_base = row * groups * bytes_per_group;
    uint input_base = token * columns;
    float sum = 0.0f;
    for (uint column = lane; column < columns; column += 32) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[packed_row_base + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local % 2) * 4)) & 0xf;
        float weight = float(int(code) - qmax) * float(scales[row * groups + group]);
        sum += weight * input[input_base + column];
    }
    sum = qx_ordered_simd_sum(sum);
    if (lane == 0) {
        output[token * rows + row] = sum * multiplier;
    }
}

// Weight-stationary K-wide Q4 projection. A threadgroup owns eight output
// rows and each SIMD group owns one row. Each packed byte and scale is loaded
// once for that row, then applied to all resident token activations before
// being discarded. This is used only by the verifier's QKV candidate.
kernel void qx_q4_dense_weight_stationary_gemm_k(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant float& multiplier [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& tokens [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    constexpr uint group_size = 32;
    constexpr uint bytes_per_group = 16;
    constexpr int qmax = 7;
    constexpr uint rows_per_threadgroup = 8;
    uint row = threadgroup_id * rows_per_threadgroup + simdgroup;
    if (row >= rows || tokens == 0) return;

    uint groups = (columns + group_size - 1) / group_size;
    uint row_base = row * groups * bytes_per_group;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint group = 0; group < groups; ++group) {
        float selected_scale = simd_broadcast(
            lane == 0 ? float(scales[row * groups + group]) : 0.0f, 0);
        uint byte_index = lane >> 1u;
        uchar packed_byte = packed[row_base + group * bytes_per_group + byte_index];
        uint code = (packed_byte >> ((lane & 1u) * 4u)) & 0xfu;
        float weight = float(int(code) - qmax) * selected_scale;
        uint input_base = group * group_size + lane;
        for (uint token = 0; token < 4; ++token) {
            if (token >= tokens) break;
            uint token_base = token * columns;
            if (input_base < columns) {
                accum[token] += weight * input[token_base + input_base];
            }
        }
    }
    for (uint token = 0; token < 4; ++token) {
        if (token >= tokens) break;
        float sum = qx_ordered_simd_sum(accum[token]) * multiplier;
        if (lane == 0) output[token * rows + row] = sum;
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
        sum = qx_ordered_simd_sum(sum) * multiplier;
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
        sum = qx_ordered_simd_sum(sum) * multiplier;
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

inline void qx_q4_dense_superblock_argmax_body(
    device const uchar* stream,
    device qx_q4_argmax_partial_t* partials,
    threadgroup const float* cached_input,
    threadgroup float* values,
    threadgroup uint* indices,
    uint columns,
    uint rows,
    float multiplier,
    uint lane,
    uint simdgroup,
    uint local_tid,
    uint threadgroup_id) {
    constexpr uint rows_per_threadgroup = 32;
    constexpr uint block_bytes = 144;
    if (local_tid < rows_per_threadgroup) {
        values[local_tid] = -3.402823466e+38f;
        indices[local_tid] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint blocks = columns / 256u;
    uint row_bytes = blocks * block_bytes;
    uint row0 = threadgroup_id * rows_per_threadgroup + simdgroup * 4u;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (row0 < rows) {
        for (uint block = 0; block < blocks; ++block) {
            uint column = block * 256u + lane * 8u;
            threadgroup const float4* input4 =
                reinterpret_cast<threadgroup const float4*>(cached_input + column);
            float4 x0 = input4[0];
            float4 x1 = input4[1];
            #pragma unroll
            for (uint row_offset = 0; row_offset < 4u; ++row_offset) {
                uint row = row0 + row_offset;
                if (row < rows) {
                    uint base = row * row_bytes + block * block_bytes;
                    uint packed = reinterpret_cast<device const uint*>(stream + base)[lane];
                    float owned_scale = lane < 8u
                        ? float(reinterpret_cast<device const half*>(stream + base + 128u)[lane])
                        : 0.0f;
                    float scale = simd_shuffle(owned_scale, ushort(lane >> 2u));
                    sums[row_offset] = qx_q4_dense_superblock_dot8(
                        packed, x0, x1, scale, sums[row_offset]);
                }
            }
        }
        #pragma unroll
        for (uint row_offset = 0; row_offset < 4u; ++row_offset) {
            sums[row_offset] = qx_ordered_simd_sum(sums[row_offset]) * multiplier;
        }
    }
    if (lane == 0u) {
        uint local0 = simdgroup * 4u;
        #pragma unroll
        for (uint row_offset = 0; row_offset < 4u; ++row_offset) {
            uint row = row0 + row_offset;
            values[local0 + row_offset] = row < rows
                ? sums[row_offset] : -3.402823466e+38f;
            indices[local0 + row_offset] = row < rows ? row : 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_tid == 0u) {
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

kernel void qx_q4_dense_argmax_superblock144_partial(
    device const uchar* stream [[buffer(0)]],
    device const uchar* ignored_scales [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device qx_q4_argmax_partial_t* partials [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant float& multiplier [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint local_tid [[thread_index_in_threadgroup]],
    uint threadgroup_id [[threadgroup_position_in_grid]]) {
    threadgroup float cached_input[1024];
    threadgroup float values[32];
    threadgroup uint indices[32];
    for (uint column = local_tid; column < columns; column += 256u) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_q4_dense_superblock_argmax_body(
        stream, partials, cached_input, values, indices, columns, rows, multiplier,
        lane, simdgroup, local_tid, threadgroup_id);
}

kernel void qx_q4_dense_argmax_superblock144_norm_partial(
    device const uchar* stream [[buffer(0)]],
    device const uchar* ignored_scales [[buffer(1)]],
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
    threadgroup float cached_input[1024];
    threadgroup float values[32];
    threadgroup uint indices[32];
    float inverse = inverse_rms[0];
    for (uint column = local_tid; column < columns; column += 256u) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_q4_dense_superblock_argmax_body(
        stream, partials, cached_input, values, indices, columns, rows, multiplier,
        lane, simdgroup, local_tid, threadgroup_id);
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
        sum = qx_ordered_simd_sum(sum) * multiplier;
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
        sum = qx_ordered_simd_sum(sum) * multiplier;
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
    sum = qx_ordered_simd_sum(sum);
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
        sum = qx_ordered_simd_sum(sum);
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
    sum = qx_ordered_simd_sum(sum);
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
    sum = qx_ordered_simd_sum(sum);
    if (lane == 0) output[row] = sum * multiplier;
}

inline void qx_q4_qkv_superblock_body(
    device const uchar* q_stream,
    device const uchar* k_stream,
    device const uchar* v_stream,
    device float* query,
    device float* key,
    device float* value,
    threadgroup const float* cached_input,
    float multiplier,
    uint lane,
    uint simdgroup,
    uint threadgroup_id) {
    constexpr uint q_rows = 1024;
    constexpr uint kv_rows = 512;
    constexpr uint rows_per_threadgroup = 16;
    constexpr uint blocks = 4;
    constexpr uint row_bytes = blocks * 144;
    uint global_row0 = threadgroup_id * rows_per_threadgroup + simdgroup * 2u;
    device const uchar* stream;
    device float* output;
    uint row0;
    if (global_row0 < q_rows) {
        stream = q_stream;
        output = query;
        row0 = global_row0;
    } else if (global_row0 < q_rows + kv_rows) {
        stream = k_stream;
        output = key;
        row0 = global_row0 - q_rows;
    } else {
        stream = v_stream;
        output = value;
        row0 = global_row0 - q_rows - kv_rows;
    }
    uint row1 = row0 + 1u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    #pragma unroll
    for (uint block = 0; block < blocks; ++block) {
        uint column = block * 256u + lane * 8u;
        threadgroup const float4* values =
            reinterpret_cast<threadgroup const float4*>(cached_input + column);
        float4 x0 = values[0];
        float4 x1 = values[1];
        uint base0 = row0 * row_bytes + block * 144u;
        uint base1 = base0 + row_bytes;
        uint packed0 = reinterpret_cast<device const uint*>(stream + base0)[lane];
        uint packed1 = reinterpret_cast<device const uint*>(stream + base1)[lane];
        float owned_scale0 = lane < 8u
            ? float(reinterpret_cast<device const half*>(stream + base0 + 128u)[lane])
            : 0.0f;
        float owned_scale1 = lane < 8u
            ? float(reinterpret_cast<device const half*>(stream + base1 + 128u)[lane])
            : 0.0f;
        ushort scale_lane = ushort(lane >> 2u);
        float scale0 = simd_shuffle(owned_scale0, scale_lane);
        float scale1 = simd_shuffle(owned_scale1, scale_lane);
        sum0 = qx_q4_dense_superblock_dot8(packed0, x0, x1, scale0, sum0);
        sum1 = qx_q4_dense_superblock_dot8(packed1, x0, x1, scale1, sum1);
    }
    sum0 = qx_ordered_simd_sum(sum0);
    sum1 = qx_ordered_simd_sum(sum1);
    if (lane == 0u) {
        output[row0] = sum0 * multiplier;
        output[row1] = sum1 * multiplier;
    }
}

kernel void qx_q4_qkv_superblock144_simd_gemv(
    device const uchar* q_stream [[buffer(0)]],
    device const uchar* ignored_q_scales [[buffer(1)]],
    device const uchar* k_stream [[buffer(2)]],
    device const uchar* ignored_k_scales [[buffer(3)]],
    device const uchar* v_stream [[buffer(4)]],
    device const uchar* ignored_v_scales [[buffer(5)]],
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
    threadgroup float cached_input[1024];
    for (uint column = thread_index; column < columns; column += 256u) {
        cached_input[column] = input[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_q4_qkv_superblock_body(
        q_stream, k_stream, v_stream, query, key, value, cached_input,
        multiplier, lane, simdgroup, threadgroup_id);
}

kernel void qx_q4_qkv_superblock144_norm_simd_gemv(
    device const uchar* q_stream [[buffer(0)]],
    device const uchar* ignored_q_scales [[buffer(1)]],
    device const uchar* k_stream [[buffer(2)]],
    device const uchar* ignored_k_scales [[buffer(3)]],
    device const uchar* v_stream [[buffer(4)]],
    device const uchar* ignored_v_scales [[buffer(5)]],
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
    threadgroup float cached_input[1024];
    float inverse = inverse_rms[0];
    for (uint column = thread_index; column < columns; column += 256u) {
        cached_input[column] = input[column] * inverse * norm_weights[column];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qx_q4_qkv_superblock_body(
        q_stream, k_stream, v_stream, query, key, value, cached_input,
        multiplier, lane, simdgroup, threadgroup_id);
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
            sum = qx_ordered_simd_sum(sum);
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
            sum = qx_ordered_simd_sum(sum);
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
            sum = qx_ordered_simd_sum(sum);
            if (lane == 0) value[current_row] = sum * multiplier;
        }
    }
}
