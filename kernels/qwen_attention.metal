#include <metal_stdlib>
using namespace metal;

kernel void qx_qwen_rms_norm(
    device const half* input [[buffer(0)]],
    device const float* norm [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 2048;
    constexpr uint group = 256;
    threadgroup float sums[group];
    float local = 0.0f;
    for (uint index = tid; index < hidden; index += group) {
        float value = float(input[index]);
        local += value * value;
    }
    sums[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_rms = rsqrt(sums[0] / float(hidden) + 1.0e-6f);
    for (uint index = tid; index < hidden; index += group) {
        output[index] = half(float(input[index]) * inverse_rms * norm[index]);
    }
}

kernel void qx_qwen_residual_rms_norm(
    device const half* residual [[buffer(0)]],
    device const half* update [[buffer(1)]],
    device const float* norm [[buffer(2)]],
    device half* hidden_output [[buffer(3)]],
    device half* norm_output [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint hidden = 2048;
    constexpr uint group = 256;
    threadgroup float sums[group];
    float local = 0.0f;
    for (uint index = tid; index < hidden; index += group) {
        half value = half(float(residual[index]) + float(update[index]));
        hidden_output[index] = value;
        float quantized = float(value);
        local += quantized * quantized;
    }
    sums[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse_rms = rsqrt(sums[0] / float(hidden) + 1.0e-6f);
    for (uint index = tid; index < hidden; index += group) {
        norm_output[index] = half(float(hidden_output[index]) * inverse_rms * norm[index]);
    }
}

kernel void qx_qwen_store_kv(
    device const half* key [[buffer(0)]],
    device const half* value [[buffer(1)]],
    device half* key_cache [[buffer(2)]],
    device half* value_cache [[buffer(3)]],
    constant uint& position [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint kv_elements = 4 * 128;
    if (index < kv_elements) {
        uint destination = position * kv_elements + index;
        key_cache[destination] = key[index];
        value_cache[destination] = value[index];
    }
}

kernel void qx_qwen_residual(
    device const half* residual [[buffer(0)]],
    device const half* update [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < 2048) {
        output[index] = half(float(residual[index]) + float(update[index]));
    }
}

kernel void qx_qwen_q4_embedding_lookup(
    device const uchar* packed [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const uint* token [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint columns = 2048;
    constexpr uint group_size = 32;
    constexpr uint groups = columns / group_size;
    constexpr uint bytes_per_group = 16;
    uint row = token[0];
    if (row >= rows) {
        return;
    }
    for (uint column = tid; column < columns; column += 256) {
        uint group = column / group_size;
        uint local = column % group_size;
        uchar byte = packed[row * groups * bytes_per_group
            + group * bytes_per_group + local / 2];
        uint code = (byte >> ((local & 1) * 4)) & 0xf;
        output[column] = half(float(int(code) - 7) * float(scales[row * groups + group]));
    }
}

// Correctness-first final selection. This intentionally favors an exact,
// allocation-free GPU dependency over parallel reduction; the production
// head can replace it without changing the token graph contract.
kernel void qx_qwen_argmax(
    device const half* logits [[buffer(0)]],
    device uint* token [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0 || count == 0) {
        return;
    }
    float best = float(logits[0]);
    uint best_index = 0;
    for (uint index = 1; index < count; ++index) {
        float value = float(logits[index]);
        if (value > best) {
            best = value;
            best_index = index;
        }
    }
    token[0] = best_index;
}

// Qwen3 Q/K normalization and split-half RoPE. Query contains 32 x 128
// elements and key contains 4 x 128 elements. One thread owns a complete head
// so the reference mapping has deterministic reduction and write ordering.
kernel void qx_qwen_qk_norm_rope(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    device const float* query_norm [[buffer(2)]],
    device const float* key_norm [[buffer(3)]],
    constant uint& position [[buffer(4)]],
    uint head [[thread_position_in_grid]]) {
    constexpr uint query_heads = 32;
    constexpr uint key_heads = 4;
    constexpr uint head_dim = 128;
    constexpr uint half_dim = 64;
    constexpr float epsilon = 1.0e-6f;
    constexpr float rope_theta = 1000000.0f;
    if (head >= query_heads + key_heads) {
        return;
    }

    bool is_query = head < query_heads;
    uint local_head = is_query ? head : head - query_heads;
    device half* values = is_query ? query : key;
    device const float* norm = is_query ? query_norm : key_norm;
    uint offset = local_head * head_dim;
    float sum_squares = 0.0f;
    for (uint index = 0; index < head_dim; ++index) {
        float value = float(values[offset + index]);
        sum_squares += value * value;
    }
    float inverse_rms = rsqrt(sum_squares / float(head_dim) + epsilon);
    for (uint index = 0; index < half_dim; ++index) {
        float left = float(values[offset + index]) * inverse_rms * norm[index];
        float right = float(values[offset + half_dim + index]) * inverse_rms
            * norm[half_dim + index];
        float inverse_frequency = pow(rope_theta, -float(index) / float(half_dim));
        float angle = float(position) * inverse_frequency;
        float cosine = cos(angle);
        float sine = sin(angle);
        values[offset + index] = half(left * cosine - right * sine);
        values[offset + half_dim + index] = half(right * cosine + left * sine);
    }
}

// Correctness-first grouped-query attention. One thread owns one output
// element. It recomputes the head score while accumulating each value
// dimension, trading arithmetic for a synchronization-free reference path.
kernel void qx_qwen_gqa_attention(
    device const half* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device const half* value_cache [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& context [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    constexpr uint query_heads = 32;
    constexpr uint key_heads = 4;
    constexpr uint head_dim = 128;
    constexpr uint query_elements = query_heads * head_dim;
    if (index >= query_elements || context == 0) {
        return;
    }
    uint query_head = index / head_dim;
    uint dimension = index % head_dim;
    uint key_head = query_head / (query_heads / key_heads);
    uint query_offset = query_head * head_dim;
    float scale = rsqrt(float(head_dim));

    float maximum = -INFINITY;
    for (uint token = 0; token < context; ++token) {
        uint key_offset = (token * key_heads + key_head) * head_dim;
        float score = 0.0f;
        for (uint column = 0; column < head_dim; ++column) {
            score += float(query[query_offset + column])
                * float(key_cache[key_offset + column]);
        }
        maximum = max(maximum, score * scale);
    }

    float denominator = 0.0f;
    float weighted = 0.0f;
    for (uint token = 0; token < context; ++token) {
        uint key_offset = (token * key_heads + key_head) * head_dim;
        float score = 0.0f;
        for (uint column = 0; column < head_dim; ++column) {
            score += float(query[query_offset + column])
                * float(key_cache[key_offset + column]);
        }
        float weight = exp(score * scale - maximum);
        denominator += weight;
        weighted += weight * float(value_cache[key_offset + dimension]);
    }
    output[index] = half(weighted / denominator);
}

kernel void qx_qwen_top8(
    device const half* logits [[buffer(0)]],
    device uint* ids [[buffer(1)]],
    device float* weights [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) {
        return;
    }
    float best_values[8];
    uint best_ids[8];
    for (uint slot = 0; slot < 8; ++slot) {
        best_values[slot] = -INFINITY;
        best_ids[slot] = 0;
    }
    for (uint expert = 0; expert < 128; ++expert) {
        float value = float(logits[expert]);
        for (uint slot = 0; slot < 8; ++slot) {
            if (value > best_values[slot]
                || (value == best_values[slot] && expert < best_ids[slot])) {
                for (uint move = 7; move > slot; --move) {
                    best_values[move] = best_values[move - 1];
                    best_ids[move] = best_ids[move - 1];
                }
                best_values[slot] = value;
                best_ids[slot] = expert;
                break;
            }
        }
    }
    float maximum = best_values[0];
    float denominator = 0.0f;
    for (uint slot = 0; slot < 8; ++slot) {
        weights[slot] = exp(best_values[slot] - maximum);
        denominator += weights[slot];
    }
    for (uint slot = 0; slot < 8; ++slot) {
        ids[slot] = best_ids[slot];
        weights[slot] /= denominator;
    }
}
