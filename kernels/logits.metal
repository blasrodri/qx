#include <metal_stdlib>
using namespace metal;

// One output row per thread for the tied Granite embedding/LM head.
kernel void qx_logits(
    device const float* embeddings [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant float& scaling [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) {
        return;
    }
    constexpr uint hidden_size = 1024;
    float sum = 0.0f;
    for (uint column = 0; column < hidden_size; ++column) {
        sum += embeddings[row * hidden_size + column] * hidden[column];
    }
    logits[row] = sum / scaling;
}
