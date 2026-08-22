#include <metal_stdlib>
using namespace metal;

// Row-major dense matrix-vector multiply used by Granite attention and router
// projections. One thread owns one output row.
kernel void qx_dense_gemv(
    device const float* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    float sum = 0.0f;
    uint offset = row * columns;
    for (uint column = 0; column < columns; ++column) {
        sum += weights[offset + column] * input[column];
    }
    output[row] = sum;
}
