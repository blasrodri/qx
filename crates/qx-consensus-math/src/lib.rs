#![no_std]

//! Bit-exact scalar math used by QX execution and terminal verification.
//!
//! These routines deliberately use only elementary IEEE-754 binary32
//! operations in an explicit order. The matching Metal implementation lives
//! in `kernels/consensus_math.metal`. Changing a constant, clamp, or operation
//! order requires a new execution-spec version.

pub const EXP_MIN_INPUT: f32 = -87.0;
pub const EXP_MAX_INPUT: f32 = 88.0;

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
}
