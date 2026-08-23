#include <metal_stdlib>
using namespace metal;

struct NormParams {
    uint n;
    float epsilon;
};

struct VectorParams {
    uint n;
};

struct ResidualParams {
    uint n;
    float scale;
};

struct PositionParams {
    uint position;
};

struct AttentionParams {
    uint length;
};

kernel void qx_v2_noop(uint tid [[thread_position_in_grid]]) {
    (void)tid;
}

// These first V2 kernels use one control thread for reductions. They are
// intentionally simple correctness kernels; fixed SIMD reductions can replace
// their bodies without changing the graph ABI.
kernel void qx_v2_rmsnorm(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant NormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < params.n; ++i) sum += input[i] * input[i];
    float inverse = rsqrt(sum / float(params.n) + params.epsilon);
    for (uint i = 0; i < params.n; ++i) output[i] = input[i] * inverse * weights[i];
}

// Fixed Granite RMSNorm path. Eight SIMD groups cooperatively reduce the
// 1024-element vector, then all lanes write the normalized result. Keeping
// the reduction and writeback in one dispatch avoids the serial control
// thread in the reference kernel above.
kernel void qx_v2_rmsnorm_simd(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant NormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8];
    float sum = 0.0f;
    for (uint i = tid; i < params.n; i += 256) {
        sum += input[i] * input[i];
    }
    sum = simd_sum(sum);
    if (lane == 0) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float total = lane < 8 ? partials[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) partials[0] = rsqrt(total / float(params.n) + params.epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < params.n) output[tid] = input[tid] * partials[0] * weights[tid];
}

// Topology-B RMSNorm stage: compute only the shared inverse RMS scalar. The
// following projection consumer applies the per-element norm weights while
// it reads the original hidden vector, avoiding a normalized-vector writeback.
kernel void qx_v2_rmsnorm_scalar(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant NormParams& params [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    (void)weights;
    threadgroup float partials[8];
    float sum = 0.0f;
    for (uint i = tid; i < params.n; i += 256) sum += input[i] * input[i];
    sum = simd_sum(sum);
    if (lane == 0) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float total = lane < 8 ? partials[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) output[0] = rsqrt(total / float(params.n) + params.epsilon);
    }
}

kernel void qx_v2_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant VectorParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid < params.n) output[tid] = half(input[tid]);
}

kernel void qx_v2_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant VectorParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid < params.n) output[tid] = float(input[tid]);
}

kernel void qx_v2_residual(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ResidualParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid < params.n) output[tid] = left[tid] + params.scale * right[tid];
}

kernel void qx_v2_rope_kv(
    device const float* query_input [[buffer(0)]],
    device const float* key_input [[buffer(1)]],
    device const float* value_input [[buffer(2)]],
    device float* query_output [[buffer(3)]],
    device float* key_cache [[buffer(4)]],
    device float* value_cache [[buffer(5)]],
    constant PositionParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    constexpr uint head_dim = 64;
    constexpr uint q_heads = 16;
    constexpr uint kv_heads = 8;
    constexpr float theta = 10000.0f;
    for (uint head = 0; head < q_heads; ++head) {
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            uint offset = head * head_dim + pair;
            uint paired_offset = head * head_dim + head_dim / 2 + pair;
            float inverse = pow(theta, -float(2 * pair) / float(head_dim));
            float angle = float(params.position) * inverse;
            float c = cos(angle);
            float s = sin(angle);
            float x = query_input[offset];
            float y = query_input[paired_offset];
            query_output[offset] = x * c - y * s;
            query_output[paired_offset] = x * s + y * c;
        }
    }
    uint cache_offset = params.position * kv_heads * head_dim;
    for (uint head = 0; head < kv_heads; ++head) {
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            uint offset = head * head_dim + pair;
            uint paired_offset = head * head_dim + head_dim / 2 + pair;
            float inverse = pow(theta, -float(2 * pair) / float(head_dim));
            float angle = float(params.position) * inverse;
            float c = cos(angle);
            float s = sin(angle);
            float x = key_input[offset];
            float y = key_input[paired_offset];
            key_cache[cache_offset + offset] = x * c - y * s;
            key_cache[cache_offset + paired_offset] = x * s + y * c;
            value_cache[cache_offset + offset] = value_input[offset];
            value_cache[cache_offset + paired_offset] = value_input[paired_offset];
        }
    }
}

// Fixed Granite query RoPE. One thread owns one 2-value rotary pair, so the
// expensive angle calculation is performed once per pair rather than by a
// single serial control thread.
kernel void qx_v2_rope_query_simd(
    device const float* query_input [[buffer(0)]],
    device float* query_output [[buffer(1)]],
    constant PositionParams& params [[buffer(2)]],
    uint pair_id [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint q_heads = 16;
    constexpr uint pairs_per_head = head_dim / 2;
    if (pair_id >= q_heads * pairs_per_head) return;
    constexpr float theta = 10000.0f;
    uint head = pair_id / pairs_per_head;
    uint pair = pair_id % pairs_per_head;
    uint offset = head * head_dim + pair;
    uint paired_offset = offset + pairs_per_head;
    float inverse = pow(theta, -float(2 * pair) / float(head_dim));
    float angle = float(params.position) * inverse;
    float c = cos(angle);
    float s = sin(angle);
    float x = query_input[offset];
    float y = query_input[paired_offset];
    query_output[offset] = x * c - y * s;
    query_output[paired_offset] = x * s + y * c;
}

// Fixed Granite K/V path. K is rotated directly into its final cache slot;
// V is copied directly into the matching slot. One thread owns one rotary
// pair, exposing 256 independent pair calculations in the dispatch.
kernel void qx_v2_rope_kv_simd(
    device const float* key_input [[buffer(0)]],
    device const float* value_input [[buffer(1)]],
    device float* key_cache [[buffer(2)]],
    device float* value_cache [[buffer(3)]],
    constant PositionParams& params [[buffer(4)]],
    uint pair_id [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint pairs_per_head = head_dim / 2;
    if (pair_id >= kv_heads * pairs_per_head) return;
    constexpr float theta = 10000.0f;
    uint head = pair_id / pairs_per_head;
    uint pair = pair_id % pairs_per_head;
    uint offset = head * head_dim + pair;
    uint paired_offset = offset + pairs_per_head;
    uint cache_offset = params.position * kv_heads * head_dim;
    float inverse = pow(theta, -float(2 * pair) / float(head_dim));
    float angle = float(params.position) * inverse;
    float c = cos(angle);
    float s = sin(angle);
    float x = key_input[offset];
    float y = key_input[paired_offset];
    key_cache[cache_offset + offset] = x * c - y * s;
    key_cache[cache_offset + paired_offset] = x * s + y * c;
    value_cache[cache_offset + offset] = value_input[offset];
    value_cache[cache_offset + paired_offset] = value_input[paired_offset];
}

kernel void qx_v2_rope_kv_f16(
    device const float* query_input [[buffer(0)]],
    device const float* key_input [[buffer(1)]],
    device const float* value_input [[buffer(2)]],
    device float* query_output [[buffer(3)]],
    device half* key_cache [[buffer(4)]],
    device half* value_cache [[buffer(5)]],
    constant PositionParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    constexpr uint head_dim = 64;
    constexpr uint q_heads = 16;
    constexpr uint kv_heads = 8;
    constexpr float theta = 10000.0f;
    for (uint head = 0; head < q_heads; ++head) {
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            uint offset = head * head_dim + pair;
            uint paired_offset = head * head_dim + head_dim / 2 + pair;
            float inverse = pow(theta, -float(2 * pair) / float(head_dim));
            float angle = float(params.position) * inverse;
            float c = cos(angle);
            float s = sin(angle);
            float x = query_input[offset];
            float y = query_input[paired_offset];
            query_output[offset] = x * c - y * s;
            query_output[paired_offset] = x * s + y * c;
        }
    }
    uint cache_offset = params.position * kv_heads * head_dim;
    for (uint head = 0; head < kv_heads; ++head) {
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            uint offset = head * head_dim + pair;
            uint paired_offset = head * head_dim + head_dim / 2 + pair;
            float inverse = pow(theta, -float(2 * pair) / float(head_dim));
            float angle = float(params.position) * inverse;
            float c = cos(angle);
            float s = sin(angle);
            float x = key_input[offset];
            float y = key_input[paired_offset];
            key_cache[cache_offset + offset] = half(x * c - y * s);
            key_cache[cache_offset + paired_offset] = half(x * s + y * c);
            value_cache[cache_offset + offset] = half(value_input[offset]);
            value_cache[cache_offset + paired_offset] = half(value_input[paired_offset]);
        }
    }
}

kernel void qx_v2_rope_kv_kf32_vf16(
    device const float* query_input [[buffer(0)]],
    device const float* key_input [[buffer(1)]],
    device const float* value_input [[buffer(2)]],
    device float* query_output [[buffer(3)]],
    device float* key_cache [[buffer(4)]],
    device half* value_cache [[buffer(5)]],
    constant PositionParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    constexpr uint head_dim = 64;
    constexpr uint q_heads = 16;
    constexpr uint kv_heads = 8;
    constexpr float theta = 10000.0f;
    for (uint head = 0; head < q_heads; ++head) {
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            uint offset = head * head_dim + pair;
            uint paired_offset = head * head_dim + head_dim / 2 + pair;
            float inverse = pow(theta, -float(2 * pair) / float(head_dim));
            float angle = float(params.position) * inverse;
            float c = cos(angle);
            float s = sin(angle);
            float x = query_input[offset];
            float y = query_input[paired_offset];
            query_output[offset] = x * c - y * s;
            query_output[paired_offset] = x * s + y * c;
        }
    }
    uint cache_offset = params.position * kv_heads * head_dim;
    for (uint head = 0; head < kv_heads; ++head) {
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            uint offset = head * head_dim + pair;
            uint paired_offset = head * head_dim + head_dim / 2 + pair;
            float inverse = pow(theta, -float(2 * pair) / float(head_dim));
            float angle = float(params.position) * inverse;
            float c = cos(angle);
            float s = sin(angle);
            float x = key_input[offset];
            float y = key_input[paired_offset];
            key_cache[cache_offset + offset] = x * c - y * s;
            key_cache[cache_offset + paired_offset] = x * s + y * c;
            value_cache[cache_offset + offset] = half(value_input[offset]);
            value_cache[cache_offset + paired_offset] = half(value_input[paired_offset]);
        }
    }
}

kernel void qx_v2_rope_kv_kf16_vf32(
    device const float* query_input [[buffer(0)]],
    device const float* key_input [[buffer(1)]],
    device const float* value_input [[buffer(2)]],
    device float* query_output [[buffer(3)]],
    device half* key_cache [[buffer(4)]],
    device float* value_cache [[buffer(5)]],
    constant PositionParams& params [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    constexpr uint head_dim = 64;
    constexpr uint q_heads = 16;
    constexpr uint kv_heads = 8;
    constexpr float theta = 10000.0f;
    for (uint head = 0; head < q_heads; ++head) {
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            uint offset = head * head_dim + pair;
            uint paired_offset = head * head_dim + head_dim / 2 + pair;
            float inverse = pow(theta, -float(2 * pair) / float(head_dim));
            float angle = float(params.position) * inverse;
            float c = cos(angle);
            float s = sin(angle);
            float x = query_input[offset];
            float y = query_input[paired_offset];
            query_output[offset] = x * c - y * s;
            query_output[paired_offset] = x * s + y * c;
        }
    }
    uint cache_offset = params.position * kv_heads * head_dim;
    for (uint head = 0; head < kv_heads; ++head) {
        for (uint pair = 0; pair < head_dim / 2; ++pair) {
            uint offset = head * head_dim + pair;
            uint paired_offset = head * head_dim + head_dim / 2 + pair;
            float inverse = pow(theta, -float(2 * pair) / float(head_dim));
            float angle = float(params.position) * inverse;
            float c = cos(angle);
            float s = sin(angle);
            float x = key_input[offset];
            float y = key_input[paired_offset];
            key_cache[cache_offset + offset] = half(x * c - y * s);
            key_cache[cache_offset + paired_offset] = half(x * s + y * c);
            value_cache[cache_offset + offset] = value_input[offset];
            value_cache[cache_offset + paired_offset] = value_input[paired_offset];
        }
    }
}

kernel void qx_v2_attention(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr float scale = 0.015625f;
    if (tid >= 1024) return;
    uint q_head = tid / head_dim;
    uint dimension = tid % head_dim;
    uint kv_head = q_head / 2;
    float maximum = -INFINITY;
    for (uint position = 0; position < params.length; ++position) {
        float score = 0.0f;
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        uint query_offset = q_head * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_offset + d] * key_cache[cache_offset + d];
        }
        maximum = max(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (uint position = 0; position < params.length; ++position) {
        float score = 0.0f;
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        uint query_offset = q_head * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_offset + d] * key_cache[cache_offset + d];
        }
        denominator += exp(score * scale - maximum);
    }
    float value = 0.0f;
    for (uint position = 0; position < params.length; ++position) {
        float score = 0.0f;
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        uint query_offset = q_head * head_dim;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_offset + d] * key_cache[cache_offset + d];
        }
        value += exp(score * scale - maximum) / denominator
            * value_cache[cache_offset + dimension];
    }
    output[tid] = value;
}

// One SIMD group owns one query head. The group computes each key score once,
// normalizes those scores in-place, and then produces two output dimensions
// per lane. The reference kernel recomputed every dot product for every output
// dimension; this layout reuses the score vector across the whole head.
kernel void qx_v2_attention_simd(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint q_heads = 16;
    constexpr uint max_context = 512;
    constexpr float scale = 0.015625f;
    threadgroup float scores[q_heads * max_context];
    if (params.length > max_context || simdgroup >= q_heads) return;

    uint q_head = simdgroup;
    uint kv_head = q_head / 2;
    uint query_offset = q_head * head_dim;
    uint score_offset = q_head * max_context;
    for (uint position = lane; position < params.length; position += 32) {
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            score += query[query_offset + d] * key_cache[cache_offset + d];
        }
        scores[score_offset + position] = score * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        float maximum = -INFINITY;
        for (uint position = 0; position < params.length; ++position) {
            maximum = max(maximum, scores[score_offset + position]);
        }
        float denominator = 0.0f;
        for (uint position = 0; position < params.length; ++position) {
            float weight = exp(scores[score_offset + position] - maximum);
            scores[score_offset + position] = weight;
            denominator += weight;
        }
        for (uint position = 0; position < params.length; ++position) {
            scores[score_offset + position] /= denominator;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dimension = lane; dimension < head_dim; dimension += 32) {
        float value = 0.0f;
        for (uint position = 0; position < params.length; ++position) {
            uint cache_offset = (position * kv_heads + kv_head) * head_dim;
            value += scores[score_offset + position] * value_cache[cache_offset + dimension];
        }
        output[query_offset + dimension] = value;
    }
}

// Benchmark-only attention decomposition. Scores use the same position-major
// KV layout as the live path and occupy [q_head, context] in a GPU scratch
// buffer. These deliberately simple stages establish a cost floor before a
// new fused topology is promoted.
kernel void qx_v2_attention_qk(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint q_heads = 16;
    constexpr uint max_context = 512;
    constexpr float scale = 0.015625f;
    uint total = q_heads * max_context;
    if (tid >= total) return;
    uint q_head = tid / max_context;
    uint position = tid % max_context;
    if (position >= params.length) return;
    uint kv_head = q_head / 2;
    uint query_offset = q_head * head_dim;
    uint cache_offset = (position * kv_heads + kv_head) * head_dim;
    float score = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        score += query[query_offset + d] * key_cache[cache_offset + d];
    }
    scores[tid] = score * scale;
}

kernel void qx_v2_attention_qk_f16(
    device const float* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint q_heads = 16;
    constexpr uint max_context = 512;
    constexpr float scale = 0.015625f;
    uint total = q_heads * max_context;
    if (tid >= total) return;
    uint q_head = tid / max_context;
    uint position = tid % max_context;
    if (position >= params.length) return;
    uint kv_head = q_head / 2;
    uint query_offset = q_head * head_dim;
    uint cache_offset = (position * kv_heads + kv_head) * head_dim;
    float score = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        score += query[query_offset + d] * float(key_cache[cache_offset + d]);
    }
    scores[tid] = score * scale;
}

// Benchmark-only synthetic K-cache layout control. The buffer has the same
// dimensions as production but is consumed as [kv_head][position][dim].
kernel void qx_v2_attention_qk_kmajor(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint q_heads = 16;
    constexpr uint max_context = 512;
    constexpr float scale = 0.015625f;
    uint total = q_heads * max_context;
    if (tid >= total) return;
    uint q_head = tid / max_context;
    uint position = tid % max_context;
    if (position >= params.length) return;
    uint kv_head = q_head / 2;
    uint query_offset = q_head * head_dim;
    uint cache_offset = (kv_head * max_context + position) * head_dim;
    float score = 0.0f;
    for (uint d = 0; d < head_dim; ++d) {
        score += query[query_offset + d] * key_cache[cache_offset + d];
    }
    scores[tid] = score * scale;
}

// Granite decode-specialized qlen=1 streaming attention. One SIMD group owns
// one KV head and serves both query heads that share it. K and V are loaded
// once per lane while online softmax state and two 64-value output vectors are
// maintained without score/probability or partial buffers.
kernel void qx_v2_attention_stream_gqa(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint q_heads = 16;
    constexpr uint max_context = 512;
    constexpr float scale = 0.015625f;
    if (tid >= kv_heads * 32) return;

    uint kv_head = tid / 32;
    uint q_head0 = kv_head * 2;
    uint q_head1 = q_head0 + 1;
    uint dim0 = lane * 2;
    uint dim1 = dim0 + 1;
    uint query0 = q_head0 * head_dim;
    uint query1 = q_head1 * head_dim;

    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (uint position = 0; position < min(params.length, max_context); ++position) {
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        float k0 = key_cache[cache_offset + dim0];
        float k1 = key_cache[cache_offset + dim1];
        float dot0 = simd_sum(query[query0 + dim0] * k0
                            + query[query0 + dim1] * k1);
        float dot1 = simd_sum(query[query1 + dim0] * k0
                            + query[query1 + dim1] * k1);
        dot0 = simd_broadcast(dot0 * scale, 0);
        dot1 = simd_broadcast(dot1 * scale, 0);

        float new_max0 = max(max0, dot0);
        float new_max1 = max(max1, dot1);
        float alpha0 = exp(max0 - new_max0);
        float alpha1 = exp(max1 - new_max1);
        float beta0 = exp(dot0 - new_max0);
        float beta1 = exp(dot1 - new_max1);
        max0 = new_max0;
        max1 = new_max1;
        sum0 = sum0 * alpha0 + beta0;
        sum1 = sum1 * alpha1 + beta1;

        float v0 = value_cache[cache_offset + dim0];
        float v1 = value_cache[cache_offset + dim1];
        acc00 = acc00 * alpha0 + beta0 * v0;
        acc01 = acc01 * alpha0 + beta0 * v1;
        acc10 = acc10 * alpha1 + beta1 * v0;
        acc11 = acc11 * alpha1 + beta1 * v1;
    }

    output[query0 + dim0] = acc00 / sum0;
    output[query0 + dim1] = acc01 / sum0;
    output[query1 + dim0] = acc10 / sum1;
    output[query1 + dim1] = acc11 / sum1;
}

// Chunk-parallel variant: one 256-thread workgroup owns one KV head and its
// eight SIMD groups process independent 64-position context chunks. Each
// group emits online-softmax statistics and unnormalized output partials.
kernel void qx_v2_attention_chunked_gqa(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device float* scratch [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint max_context = 512;
    constexpr uint chunk = 64;
    constexpr uint chunks = 8;
    constexpr uint partial_floats = 16 * chunks * head_dim;
    constexpr float scale = 0.015625f;
    if (tid >= kv_heads * 256) return;

    uint kv_head = tid / 256;
    uint local = tid % 256;
    uint chunk_id = local / 32;
    uint q_head0 = kv_head * 2;
    uint q_head1 = q_head0 + 1;
    uint dim0 = lane * 2;
    uint dim1 = dim0 + 1;
    uint query0 = q_head0 * head_dim;
    uint query1 = q_head1 * head_dim;
    uint position_begin = chunk_id * chunk;

    float max0 = -INFINITY;
    float max1 = -INFINITY;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    for (uint position = position_begin;
         position < min(params.length, position_begin + chunk);
         ++position) {
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        float k0 = key_cache[cache_offset + dim0];
        float k1 = key_cache[cache_offset + dim1];
        float dot0 = simd_broadcast(simd_sum(
            query[query0 + dim0] * k0 + query[query0 + dim1] * k1) * scale, 0);
        float dot1 = simd_broadcast(simd_sum(
            query[query1 + dim0] * k0 + query[query1 + dim1] * k1) * scale, 0);
        float new_max0 = max(max0, dot0);
        float new_max1 = max(max1, dot1);
        float alpha0 = exp(max0 - new_max0);
        float alpha1 = exp(max1 - new_max1);
        float beta0 = exp(dot0 - new_max0);
        float beta1 = exp(dot1 - new_max1);
        max0 = new_max0;
        max1 = new_max1;
        sum0 = sum0 * alpha0 + beta0;
        sum1 = sum1 * alpha1 + beta1;
        float v0 = value_cache[cache_offset + dim0];
        float v1 = value_cache[cache_offset + dim1];
        acc00 = acc00 * alpha0 + beta0 * v0;
        acc01 = acc01 * alpha0 + beta0 * v1;
        acc10 = acc10 * alpha1 + beta1 * v0;
        acc11 = acc11 * alpha1 + beta1 * v1;
    }

    uint base0 = (q_head0 * chunks + chunk_id) * head_dim;
    uint base1 = (q_head1 * chunks + chunk_id) * head_dim;
    scratch[base0 + dim0] = acc00;
    scratch[base0 + dim1] = acc01;
    scratch[base1 + dim0] = acc10;
    scratch[base1 + dim1] = acc11;
    if (lane == 0) {
        uint stats = partial_floats + (q_head0 * chunks + chunk_id) * 2;
        scratch[stats] = max0;
        scratch[stats + 1] = sum0;
        stats = partial_floats + (q_head1 * chunks + chunk_id) * 2;
        scratch[stats] = max1;
        scratch[stats + 1] = sum1;
    }
}

// Merge the eight chunk summaries for each query head and output dimension.
kernel void qx_v2_attention_chunked_merge(
    device const float* scratch [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant AttentionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint chunks = 8;
    constexpr uint partial_floats = 16 * chunks * head_dim;
    constexpr float neg_inf = -INFINITY;
    if (tid >= 16 * head_dim) return;
    uint q_head = tid / head_dim;
    uint dimension = tid % head_dim;
    float global_max = neg_inf;
    for (uint c = 0; c < chunks; ++c) {
        uint stats = partial_floats + (q_head * chunks + c) * 2;
        global_max = max(global_max, scratch[stats]);
    }
    float total = 0.0f;
    float value = 0.0f;
    for (uint c = 0; c < chunks; ++c) {
        uint stats = partial_floats + (q_head * chunks + c) * 2;
        float factor = exp(scratch[stats] - global_max);
        total += scratch[stats + 1] * factor;
        value += scratch[(q_head * chunks + c) * head_dim + dimension] * factor;
    }
    output[q_head * head_dim + dimension] = value / total;
    (void)params;
}

kernel void qx_v2_attention_softmax(
    device float* scores [[buffer(0)]],
    constant AttentionParams& params [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint max_context = 512;
    constexpr uint q_heads = 16;
    if (tid >= q_heads) return;
    uint offset = tid * max_context;
    float maximum = -INFINITY;
    for (uint position = 0; position < params.length; ++position) {
        maximum = max(maximum, scores[offset + position]);
    }
    float denominator = 0.0f;
    for (uint position = 0; position < params.length; ++position) {
        float weight = exp(scores[offset + position] - maximum);
        scores[offset + position] = weight;
        denominator += weight;
    }
    for (uint position = 0; position < params.length; ++position) {
        scores[offset + position] /= denominator;
    }
}

kernel void qx_v2_attention_softmax_max(
    device const float* scores [[buffer(0)]],
    device float* partial_max [[buffer(1)]],
    constant AttentionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint max_context = 512;
    constexpr uint q_heads = 16;
    constexpr uint chunk = 32;
    constexpr uint chunks = max_context / chunk;
    uint group = tid / 32;
    uint q_head = group / chunks;
    uint chunk_id = group % chunks;
    uint position_begin = chunk_id * chunk;
    float maximum = -INFINITY;
    for (uint position = position_begin + lane;
         position < min(params.length, position_begin + chunk);
         position += 32) {
        maximum = max(maximum, scores[q_head * max_context + position]);
    }
    maximum = simd_max(maximum);
    if (lane == 0) partial_max[q_head * chunks + chunk_id] = maximum;
    (void)q_heads;
}

kernel void qx_v2_attention_softmax_simd(
    device float* scores [[buffer(0)]],
    device const float* partial_max [[buffer(1)]],
    constant AttentionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]) {
    constexpr uint max_context = 512;
    constexpr uint q_heads = 16;
    constexpr uint chunks = 16;
    threadgroup float max_reductions[16];
    threadgroup float sum_reductions[8];
    threadgroup float maximum;
    threadgroup float denominator;
    if (simdgroup == 0 && lane < chunks) {
        max_reductions[lane] = partial_max[group_id * chunks + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0 && lane == 0) {
        float value = -INFINITY;
        for (uint i = 0; i < chunks; ++i) value = max(value, max_reductions[i]);
        maximum = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint position = tid; position < params.length; position += 256) {
        local_sum += exp(scores[group_id * max_context + position] - maximum);
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) sum_reductions[simdgroup] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0 && lane == 0) {
        float value = 0.0f;
        for (uint i = 0; i < 8; ++i) value += sum_reductions[i];
        denominator = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint position = tid; position < params.length; position += 256) {
        scores[group_id * max_context + position] =
            exp(scores[group_id * max_context + position] - maximum) / denominator;
    }
    (void)q_heads;
}

kernel void qx_v2_attention_v(
    device const float* scores [[buffer(0)]],
    device const float* value_cache [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint q_heads = 16;
    constexpr uint max_context = 512;
    if (tid >= q_heads * head_dim) return;
    uint q_head = tid / head_dim;
    uint dimension = tid % head_dim;
    uint kv_head = q_head / 2;
    uint score_offset = q_head * max_context;
    float value = 0.0f;
    for (uint position = 0; position < params.length; ++position) {
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        value += scores[score_offset + position] * value_cache[cache_offset + dimension];
    }
    output[tid] = value;
}

kernel void qx_v2_attention_v_f16(
    device const float* scores [[buffer(0)]],
    device const half* value_cache [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint q_heads = 16;
    constexpr uint max_context = 512;
    if (tid >= q_heads * head_dim) return;
    uint q_head = tid / head_dim;
    uint dimension = tid % head_dim;
    uint kv_head = q_head / 2;
    uint score_offset = q_head * max_context;
    float value = 0.0f;
    for (uint position = 0; position < params.length; ++position) {
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        value += scores[score_offset + position]
            * float(value_cache[cache_offset + dimension]);
    }
    output[tid] = value;
}

// GQA-aware V accumulation. One thread owns one output dimension for a
// 32-position chunk and computes both query heads that share the KV head. The
// value load is therefore reused across the pair; a second dispatch reduces
// the chunk partials over context.
kernel void qx_v2_attention_v_partial_gqa(
    device const float* scores [[buffer(0)]],
    device const float* value_cache [[buffer(1)]],
    device float* partials [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint q_heads = 16;
    constexpr uint max_context = 512;
    constexpr uint chunk = 32;
    constexpr uint chunks = max_context / chunk;
    uint group = tid / head_dim;
    uint dimension = tid % head_dim;
    uint kv_head = group / chunks;
    uint chunk_id = group % chunks;
    uint position_begin = chunk_id * chunk;
    uint score_position = chunk_id;
    uint q_head0 = kv_head * 2;
    uint q_head1 = q_head0 + 1;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint position = position_begin;
         position < min(params.length, position_begin + chunk);
         ++position) {
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        float value = value_cache[cache_offset + dimension];
        sum0 += scores[q_head0 * max_context + position] * value;
        sum1 += scores[q_head1 * max_context + position] * value;
    }
    partials[(q_head0 * chunks + chunk_id) * head_dim + dimension] = sum0;
    partials[(q_head1 * chunks + chunk_id) * head_dim + dimension] = sum1;
    (void)q_heads;
    (void)score_position;
}

kernel void qx_v2_attention_v_partial_gqa_f16(
    device const float* scores [[buffer(0)]],
    device const half* value_cache [[buffer(1)]],
    device float* partials [[buffer(2)]],
    constant AttentionParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint kv_heads = 8;
    constexpr uint max_context = 512;
    constexpr uint chunk = 32;
    constexpr uint chunks = max_context / chunk;
    uint group = tid / head_dim;
    uint dimension = tid % head_dim;
    uint kv_head = group / chunks;
    uint chunk_id = group % chunks;
    uint position_begin = chunk_id * chunk;
    uint q_head0 = kv_head * 2;
    uint q_head1 = q_head0 + 1;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint position = position_begin;
         position < min(params.length, position_begin + chunk);
         ++position) {
        uint cache_offset = (position * kv_heads + kv_head) * head_dim;
        float value = float(value_cache[cache_offset + dimension]);
        sum0 += scores[q_head0 * max_context + position] * value;
        sum1 += scores[q_head1 * max_context + position] * value;
    }
    partials[(q_head0 * chunks + chunk_id) * head_dim + dimension] = sum0;
    partials[(q_head1 * chunks + chunk_id) * head_dim + dimension] = sum1;
}

kernel void qx_v2_attention_v_reduce(
    device const float* partials [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant AttentionParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    constexpr uint head_dim = 64;
    constexpr uint q_heads = 16;
    constexpr uint chunk = 32;
    constexpr uint chunks = 512 / chunk;
    if (tid >= q_heads * head_dim) return;
    uint q_head = tid / head_dim;
    uint dimension = tid % head_dim;
    uint active_chunks = (params.length + chunk - 1) / chunk;
    float value = 0.0f;
    for (uint chunk_id = 0; chunk_id < active_chunks; ++chunk_id) {
        value += partials[(q_head * chunks + chunk_id) * head_dim + dimension];
    }
    output[q_head * head_dim + dimension] = value;
}

kernel void qx_v2_topk(
    device const float* logits [[buffer(0)]],
    device uint* ids [[buffer(1)]],
    device float* weights [[buffer(2)]],
    constant VectorParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0 || params.n != 32) return;
    bool used[32];
    for (uint i = 0; i < 32; ++i) used[i] = false;
    float maximum = -INFINITY;
    for (uint slot = 0; slot < 8; ++slot) {
        uint best = 0;
        float best_value = -INFINITY;
        for (uint i = 0; i < 32; ++i) {
            if (!used[i] && logits[i] > best_value) {
                best = i;
                best_value = logits[i];
            }
        }
        used[best] = true;
        ids[slot] = best;
        weights[slot] = best_value;
        maximum = max(maximum, best_value);
    }
    float denominator = 0.0f;
    for (uint slot = 0; slot < 8; ++slot) {
        weights[slot] = exp(weights[slot] - maximum);
        denominator += weights[slot];
    }
    for (uint slot = 0; slot < 8; ++slot) weights[slot] /= denominator;
}

// Opt-in active-k control for MoE experiments. The canonical top-k kernel
// still selects and normalizes eight experts; this follow-up masks the tail,
// renormalizes the surviving weights, and marks inactive IDs invalid so the
// existing GPU-resident MoE stages skip them without changing their topology.
kernel void qx_v2_topk_mask(
    device uint* ids [[buffer(0)]],
    device float* weights [[buffer(1)]],
    constant uint& active_k [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint k = min(active_k, 8u);
    float denominator = 0.0f;
    for (uint slot = 0; slot < 8; ++slot) {
        if (slot < k) {
            denominator += weights[slot];
        } else {
            ids[slot] = 32;
            weights[slot] = 0.0f;
        }
    }
    if (denominator > 0.0f) {
        for (uint slot = 0; slot < k; ++slot) weights[slot] /= denominator;
    }
}
