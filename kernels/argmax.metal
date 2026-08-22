#include <metal_stdlib>
using namespace metal;

struct qx_argmax_partial_t {
    float value;
    uint index;
};

inline bool qx_better(float value, uint index, float best_value, uint best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

// First reduction level: one 256-thread threadgroup produces one partial
// maximum. The fixed vocabulary is small enough that the partial array fits
// in the preallocated V2 MoE scratch region.
kernel void qx_argmax_partial(
    device const float* logits [[buffer(0)]],
    device qx_argmax_partial_t* partials [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];

    uint row = tid;
    if (row < rows) {
        values[local_tid] = logits[row];
        indices[local_tid] = row;
    } else {
        values[local_tid] = -3.402823466e+38f;
        indices[local_tid] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_tid < stride && qx_better(
                values[local_tid + stride],
                indices[local_tid + stride],
                values[local_tid],
                indices[local_tid])) {
            values[local_tid] = values[local_tid + stride];
            indices[local_tid] = indices[local_tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_tid == 0) {
        partials[group_id].value = values[0];
        partials[group_id].index = indices[0];
    }
}

// Second reduction level: the partial maxima are reduced by one compact
// threadgroup and the winning vocabulary index is written to the u32 output.
kernel void qx_argmax_reduce(
    device const qx_argmax_partial_t* partials [[buffer(0)]],
    device uint* token_output [[buffer(1)]],
    constant uint& partial_count [[buffer(2)]],
    uint local_tid [[thread_position_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256];

    if (local_tid < partial_count) {
        values[local_tid] = partials[local_tid].value;
        indices[local_tid] = partials[local_tid].index;
    } else {
        values[local_tid] = -3.402823466e+38f;
        indices[local_tid] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local_tid < stride && qx_better(
                values[local_tid + stride],
                indices[local_tid + stride],
                values[local_tid],
                indices[local_tid])) {
            values[local_tid] = values[local_tid + stride];
            indices[local_tid] = indices[local_tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_tid == 0) {
        token_output[0] = indices[0];
    }
}
