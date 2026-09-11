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

constant uint QX_GRANITE_ROPE_INVERSE_BITS[32] = {
    0x3f800000u, 0x3f3ff911u, 0x3f0ff59au, 0x3ed7e89bu,
    0x3ea1e89bu, 0x3e72d424u, 0x3e361887u, 0x3e088d77u,
    0x3dcccccdu, 0x3d99940eu, 0x3d6655c3u, 0x3d2cba15u,
    0x3d0186e2u, 0x3cc24350u, 0x3c91ad39u, 0x3c5a7bf1u,
    0x3c23d70au, 0x3bf5b9b0u, 0x3bb8449cu, 0x3b8a2e77u,
    0x3b4f3e37u, 0x3b1b690du, 0x3ae91528u, 0x3aaec98eu,
    0x3a83126fu, 0x3a44948cu, 0x3a136a16u, 0x39dd1726u,
    0x39a5cb5fu, 0x3978a814u, 0x393a7753u, 0x390bd472u,
};

inline float2 qx_granite_rope_sin_cos(uint position, uint pair) {
    float angle = float(position) * as_type<float>(QX_GRANITE_ROPE_INVERSE_BITS[pair]);
    float scaled = fma(angle, as_type<float>(0x3e22f983u), 0.0f);
    float rounded = scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f;
    int turns = int(rounded);
    float reduced = fma(-float(turns), as_type<float>(0x40c90fdbu), angle);
    reduced = fma(-float(turns), as_type<float>(0xb43bbd2eu), reduced);
    float cosine_sign = 1.0f;
    if (reduced > as_type<float>(0x3fc90fdbu)) {
        reduced = as_type<float>(0x40490fdbu) - reduced;
        cosine_sign = -1.0f;
    } else if (reduced < -as_type<float>(0x3fc90fdbu)) {
        reduced = -as_type<float>(0x40490fdbu) - reduced;
        cosine_sign = -1.0f;
    }

    float square = reduced * reduced;
    float sine_polynomial = as_type<float>(0x2f309231u);
    sine_polynomial = fma(sine_polynomial, square, as_type<float>(0xb2d7322bu));
    sine_polynomial = fma(sine_polynomial, square, as_type<float>(0x3638ef1du));
    sine_polynomial = fma(sine_polynomial, square, as_type<float>(0xb9500d01u));
    sine_polynomial = fma(sine_polynomial, square, as_type<float>(0x3c088889u));
    sine_polynomial = fma(sine_polynomial, square, as_type<float>(0xbe2aaaabu));
    float sine = fma(sine_polynomial * square, reduced, reduced);

    float cosine_polynomial = as_type<float>(0x310f76c7u);
    cosine_polynomial = fma(cosine_polynomial, square, as_type<float>(0xb493f27eu));
    cosine_polynomial = fma(cosine_polynomial, square, as_type<float>(0x37d00d01u));
    cosine_polynomial = fma(cosine_polynomial, square, as_type<float>(0xbab60b61u));
    cosine_polynomial = fma(cosine_polynomial, square, as_type<float>(0x3d2aaaabu));
    cosine_polynomial = fma(cosine_polynomial, square, as_type<float>(0xbf000000u));
    float cosine = fma(cosine_polynomial, square, 1.0f) * cosine_sign;
    return float2(sine, cosine);
}
#else
inline float qx_consensus_exp(float x) { return exp(x); }
inline float qx_consensus_rsqrt(float x) { return rsqrt(x); }
inline float2 qx_granite_rope_sin_cos(uint position, uint pair) {
    constexpr float theta = 10000.0f;
    float inverse = pow(theta, -float(2 * pair) / 64.0f);
    float angle = float(position) * inverse;
    return float2(sin(angle), cos(angle));
}
#endif

kernel void qx_consensus_math_conformance(
    device const float* inputs [[buffer(0)]],
    device uint4* output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    float2 rope = qx_granite_rope_sin_cos(uint(inputs[index]), index & 31u);
    output[index] = uint4(
        as_type<uint>(qx_consensus_exp(inputs[index])),
        as_type<uint>(qx_consensus_rsqrt(inputs[index])),
        as_type<uint>(rope.x),
        as_type<uint>(rope.y));
}
