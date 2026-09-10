// QX fixed-operation scalar math V1. Keep constants and operation order in
// sync with crates/qx-consensus-math. This source is prepended to every model
// kernel that needs exp or reciprocal square root.

#include <metal_stdlib>
using namespace metal;

#if defined(QX_CONSENSUS_FIXED_MATH)
inline float qx_consensus_exp(float x) {
    if (!isfinite(x)) return NAN;
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) return as_type<float>(0x7f7fffffu);

    float scaled = fma(x, as_type<float>(0x3fb8aa3bu), 0.0f);
    float rounded = scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f;
    int exponent = int(rounded);
    float exponent_f32 = float(exponent);
    float reduced = fma(-exponent_f32, as_type<float>(0x3f317200u), x);
    reduced = fma(-exponent_f32, as_type<float>(0x35bfbe8eu), reduced);

    float polynomial = as_type<float>(0x39500d01u);
    polynomial = fma(polynomial, reduced, as_type<float>(0x3ab60b61u));
    polynomial = fma(polynomial, reduced, as_type<float>(0x3c088889u));
    polynomial = fma(polynomial, reduced, as_type<float>(0x3d2aaaabu));
    polynomial = fma(polynomial, reduced, as_type<float>(0x3e2aaaabu));
    polynomial = fma(polynomial, reduced, as_type<float>(0x3f000000u));
    polynomial = fma(polynomial, reduced, 1.0f);
    polynomial = fma(polynomial, reduced, 1.0f);

    float power_of_two = as_type<float>(uint(exponent + 127) << 23);
    return polynomial * power_of_two;
}

inline float qx_consensus_rsqrt(float x) {
    if (!isfinite(x) || x <= 0.0f) return NAN;
    float half_value = x * 0.5f;
    float estimate = as_type<float>(0x5f375a86u - (as_type<uint>(x) >> 1));
    for (uint iteration = 0; iteration < 4; iteration++) {
        float square = estimate * estimate;
        float correction = fma(-half_value, square, 1.5f);
        estimate = estimate * correction;
    }
    return estimate;
}
#else
inline float qx_consensus_exp(float x) { return exp(x); }
inline float qx_consensus_rsqrt(float x) { return rsqrt(x); }
#endif

kernel void qx_consensus_math_conformance(
    device const float* inputs [[buffer(0)]],
    device uint2* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    output[index] = uint2(
        as_type<uint>(qx_consensus_exp(inputs[index])),
        as_type<uint>(qx_consensus_rsqrt(inputs[index])));
}
