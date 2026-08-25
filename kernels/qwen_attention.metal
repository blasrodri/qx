#include <metal_stdlib>
using namespace metal;

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
