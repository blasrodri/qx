#![no_std]

//! Bit-exact scalar math used by QX execution and terminal verification.
//!
//! These routines deliberately use only elementary IEEE-754 binary32
//! operations in an explicit order. The matching Metal implementation lives
//! in `kernels/consensus_math.metal`. Changing a constant, clamp, or operation
//! order requires a new execution-spec version.

pub const EXP_MIN_INPUT: f32 = -87.0;
pub const EXP_MAX_INPUT: f32 = 88.0;

/// Pinned Granite inverse frequencies for `theta=10000`, head dimension 64.
/// Storing the binary32 values avoids implementation-defined `pow` in every
/// CPU/GPU RoPE invocation.
pub const GRANITE_ROPE_INVERSE_FREQUENCY_BITS: [u32; 32] = [
    0x3f80_0000,
    0x3f3f_f911,
    0x3f0f_f59a,
    0x3ed7_e89b,
    0x3ea1_e89b,
    0x3e72_d424,
    0x3e36_1887,
    0x3e08_8d77,
    0x3dcc_cccd,
    0x3d99_940e,
    0x3d66_55c3,
    0x3d2c_ba15,
    0x3d01_86e2,
    0x3cc2_4350,
    0x3c91_ad39,
    0x3c5a_7bf1,
    0x3c23_d70a,
    0x3bf5_b9b0,
    0x3bb8_449c,
    0x3b8a_2e77,
    0x3b4f_3e37,
    0x3b1b_690d,
    0x3ae9_1528,
    0x3aae_c98e,
    0x3a83_126f,
    0x3a44_948c,
    0x3a13_6a16,
    0x39dd_1726,
    0x39a5_cb5f,
    0x3978_a814,
    0x393a_7753,
    0x390b_d472,
];

/// Deterministic binary32 approximation to `exp(x)`.
///
/// The input is reduced to `r` in approximately `[-ln(2)/2, ln(2)/2]`, a
/// seventh-degree polynomial is evaluated with explicit FMAs, and the result
/// is scaled by constructing an exact normal power of two. Values below the
/// clamp become zero; values above it become the largest finite binary32.
#[inline(always)]
pub fn exp_f32(x: f32) -> Option<f32> {
    if !x.is_finite() {
        return None;
    }
    if x < EXP_MIN_INPUT {
        return Some(0.0);
    }
    if x > EXP_MAX_INPUT {
        return Some(f32::MAX);
    }

    // Round x/log(2) to the nearest integer. Rust and Metal integer casts both
    // truncate toward zero for this finite, tightly bounded value.
    let scaled = libm::fmaf(x, f32::from_bits(0x3fb8_aa3b), 0.0);
    let rounded = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    let exponent = rounded as i32;
    let exponent_f32 = exponent as f32;

    // Split ln(2) keeps cancellation error small during range reduction.
    let reduced = libm::fmaf(-exponent_f32, f32::from_bits(0x3f31_7200), x);
    let reduced = libm::fmaf(-exponent_f32, f32::from_bits(0x35bf_be8e), reduced);

    // Degree-7 Taylor on the small reduced interval. Its truncation error is
    // below one binary32 ulp throughout that interval.
    let mut polynomial = f32::from_bits(0x3950_0d01); // 1/5040
    polynomial = libm::fmaf(polynomial, reduced, f32::from_bits(0x3ab6_0b61)); // 1/720
    polynomial = libm::fmaf(polynomial, reduced, f32::from_bits(0x3c08_8889)); // 1/120
    polynomial = libm::fmaf(polynomial, reduced, f32::from_bits(0x3d2a_aaab)); // 1/24
    polynomial = libm::fmaf(polynomial, reduced, f32::from_bits(0x3e2a_aaab)); // 1/6
    polynomial = libm::fmaf(polynomial, reduced, f32::from_bits(0x3f00_0000)); // 1/2
    polynomial = libm::fmaf(polynomial, reduced, 1.0);
    polynomial = libm::fmaf(polynomial, reduced, 1.0);

    let power_of_two = f32::from_bits(((exponent + 127) as u32) << 23);
    let result = polynomial * power_of_two;
    result.is_finite().then_some(result)
}

/// Deterministic binary32 approximation to `1/sqrt(x)` for positive inputs.
#[inline(always)]
pub fn rsqrt_f32(x: f32) -> Option<f32> {
    if !x.is_finite() || x <= 0.0 {
        return None;
    }
    let half = x * 0.5;
    let mut estimate = f32::from_bits(0x5f37_5a86_u32.wrapping_sub(x.to_bits() >> 1));
    // Four fixed Newton-Raphson refinements reach binary32 precision from the
    // integer seed. Spell each step identically in Metal.
    for _ in 0..4 {
        let square = estimate * estimate;
        let correction = libm::fmaf(-half, square, 1.5);
        estimate *= correction;
    }
    estimate.is_finite().then_some(estimate)
}

#[inline(always)]
pub fn div_f32(numerator: f32, denominator: f32) -> Option<f32> {
    if !numerator.is_finite() || !denominator.is_finite() || denominator == 0.0 {
        return None;
    }
    let result = numerator / denominator;
    result.is_finite().then_some(result)
}

#[inline(always)]
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    libm::fmaf(a, b, c)
}

/// Fixed-operation sine/cosine pair for finite Granite RoPE angles.
/// Range reduction and both polynomials use only pinned binary32 constants
/// and explicit FMAs; `kernels/consensus_math.metal` mirrors this exactly.
#[inline(always)]
pub fn sin_cos_f32(angle: f32) -> Option<(f32, f32)> {
    if !angle.is_finite() || angle.abs() > 4096.0 {
        return None;
    }
    let scaled = libm::fmaf(angle, f32::from_bits(0x3e22_f983), 0.0);
    let rounded = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    let turns = rounded as i32;
    let turns_f32 = turns as f32;
    let mut reduced = libm::fmaf(-turns_f32, f32::from_bits(0x40c9_0fdb), angle);
    reduced = libm::fmaf(-turns_f32, f32::from_bits(0xb43b_bd2e), reduced);

    let mut cosine_sign = 1.0;
    if reduced > f32::from_bits(0x3fc9_0fdb) {
        reduced = f32::from_bits(0x4049_0fdb) - reduced;
        cosine_sign = -1.0;
    } else if reduced < -f32::from_bits(0x3fc9_0fdb) {
        reduced = -f32::from_bits(0x4049_0fdb) - reduced;
        cosine_sign = -1.0;
    }

    let square = reduced * reduced;
    let mut sine_polynomial = f32::from_bits(0x2f30_9231);
    sine_polynomial = libm::fmaf(sine_polynomial, square, f32::from_bits(0xb2d7_322b));
    sine_polynomial = libm::fmaf(sine_polynomial, square, f32::from_bits(0x3638_ef1d));
    sine_polynomial = libm::fmaf(sine_polynomial, square, f32::from_bits(0xb950_0d01));
    sine_polynomial = libm::fmaf(sine_polynomial, square, f32::from_bits(0x3c08_8889));
    sine_polynomial = libm::fmaf(sine_polynomial, square, f32::from_bits(0xbe2a_aaab));
    let sine = libm::fmaf(sine_polynomial * square, reduced, reduced);

    let mut cosine_polynomial = f32::from_bits(0x310f_76c7);
    cosine_polynomial = libm::fmaf(cosine_polynomial, square, f32::from_bits(0xb493_f27e));
    cosine_polynomial = libm::fmaf(cosine_polynomial, square, f32::from_bits(0x37d0_0d01));
    cosine_polynomial = libm::fmaf(cosine_polynomial, square, f32::from_bits(0xbab6_0b61));
    cosine_polynomial = libm::fmaf(cosine_polynomial, square, f32::from_bits(0x3d2a_aaab));
    cosine_polynomial = libm::fmaf(cosine_polynomial, square, f32::from_bits(0xbf00_0000));
    let cosine = libm::fmaf(cosine_polynomial, square, 1.0) * cosine_sign;
    Some((sine, cosine))
}

#[inline(always)]
pub fn granite_rope_sin_cos(position: u32, pair: u32) -> Option<(f32, f32)> {
    let inverse = f32::from_bits(*GRANITE_ROPE_INVERSE_FREQUENCY_BITS.get(pair as usize)?);
    sin_cos_f32(position as f32 * inverse)
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_is_near_native_across_model_domain() {
        let mut worst_relative = 0.0_f32;
        let mut worst_input = 0.0_f32;
        let mut worst_absolute = 0.0_f32;
        for step in -87_000..=88_000 {
            let input = step as f32 / 1_000.0;
            let actual = exp_f32(input).unwrap();
            let reference = input.exp();
            let relative = ((actual - reference) / reference).abs();
            if relative > worst_relative {
                worst_relative = relative;
                worst_input = input;
            }
            worst_absolute = worst_absolute.max((actual - reference).abs());
        }
        assert!(
            worst_relative <= 3.0e-6,
            "worst relative error {worst_relative} at {worst_input}; absolute {worst_absolute}"
        );
    }

    #[test]
    fn rsqrt_is_near_native_across_rms_domain() {
        let mut worst_relative = 0.0_f32;
        for exponent in -20..=20 {
            for mantissa in 1..=1_000 {
                let input = (mantissa as f32 / 1_000.0) * 2.0_f32.powi(exponent);
                let actual = rsqrt_f32(input).unwrap();
                let reference = 1.0 / input.sqrt();
                let relative = ((actual - reference) / reference).abs();
                worst_relative = worst_relative.max(relative);
            }
        }
        assert!(
            worst_relative <= 3.0e-7,
            "worst relative error {worst_relative}"
        );
    }

    #[test]
    fn special_values_are_rejected_or_clamped() {
        assert_eq!(exp_f32(-100.0), Some(0.0));
        assert_eq!(exp_f32(100.0), Some(f32::MAX));
        assert_eq!(exp_f32(f32::NAN), None);
        assert_eq!(rsqrt_f32(0.0), None);
        assert_eq!(rsqrt_f32(-1.0), None);
        assert_eq!(div_f32(1.0, 0.0), None);
    }

    #[test]
    fn golden_bits_pin_operation_order() {
        let exp_cases = [-87.0_f32, -10.0, -1.0, 0.0, 1.0, 10.0, 88.0];
        let exp_bits = exp_cases.map(|value| exp_f32(value).unwrap().to_bits());
        assert_eq!(
            exp_bits,
            [
                0x00b3_3687,
                0x383e_6bce,
                0x3ebc_5ab2,
                0x3f80_0000,
                0x402d_f854,
                0x46ac_14ef,
                0x7ef8_82b7,
            ]
        );

        let rsqrt_cases = [1.0e-6_f32, 0.25, 1.0, 2.0, 1024.0];
        let rsqrt_bits = rsqrt_cases.map(|value| rsqrt_f32(value).unwrap().to_bits());
        assert_eq!(
            rsqrt_bits,
            [
                0x447a_0000,
                0x3fff_ffff,
                0x3f7f_ffff,
                0x3f35_04f3,
                0x3cff_ffff
            ]
        );
    }

    #[test]
    fn fixed_rope_is_close_to_native_over_the_registered_context() {
        let mut worst = 0.0_f32;
        for position in 0..4096_u32 {
            for pair in 0..32_u32 {
                let inverse = f32::from_bits(GRANITE_ROPE_INVERSE_FREQUENCY_BITS[pair as usize]);
                let angle = position as f32 * inverse;
                let (sine, cosine) = granite_rope_sin_cos(position, pair).unwrap();
                let (native_sine, native_cosine) = angle.sin_cos();
                worst = worst.max((sine - native_sine).abs());
                worst = worst.max((cosine - native_cosine).abs());
            }
        }
        assert!(worst <= 3.0e-4, "worst absolute error {worst}");
    }

    #[test]
    fn fixed_rope_golden_bits_pin_range_reduction() {
        assert_eq!(
            [
                granite_rope_sin_cos(0, 0).unwrap(),
                granite_rope_sin_cos(1, 1).unwrap(),
                granite_rope_sin_cos(4095, 0).unwrap(),
                granite_rope_sin_cos(4095, 31).unwrap(),
            ]
            .map(|(sine, cosine)| (sine.to_bits(), cosine.to_bits())),
            [
                (0x0000_0000, 0x3f80_0000),
                (0x3f2e_7ace, 0x3f3b_54b0),
                (0xbf7f_7136, 0xbd87_1e5e),
                (0x3f04_f364, 0x3f5a_c4ff),
            ]
        );
    }
}
