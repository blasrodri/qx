use crate::CommitError;

pub const PPM_DENOMINATOR: u32 = 1_000_000;

/// Effective detection probability after accounting for whether an honest
/// auditor actually performs a selected replay.
pub fn effective_detection_probability(
    audit_probability_ppm: u32,
    auditor_availability_ppm: u32,
) -> Result<f64, CommitError> {
    validate_ppm(audit_probability_ppm)?;
    validate_ppm(auditor_availability_ppm)?;
    Ok(
        f64::from(audit_probability_ppm) * f64::from(auditor_availability_ppm)
            / f64::from(PPM_DENOMINATOR).powi(2),
    )
}

/// Probability of detecting at least one of `attempts` independent cheats.
pub fn repeated_cheat_detection_probability(
    audit_probability_ppm: u32,
    auditor_availability_ppm: u32,
    attempts: u64,
) -> Result<f64, CommitError> {
    let probability =
        effective_detection_probability(audit_probability_ppm, auditor_availability_ppm)?;
    Ok(1.0 - (1.0 - probability).powf(attempts as f64))
}

/// Smallest collateral that makes one cheat non-profitable under
/// `(1-p) * gain - p * collateral <= 0`, multiplied by an integer safety
/// factor. Challenger rewards and transaction fees must be added separately.
pub fn minimum_deterrent_collateral(
    cheating_gain: u64,
    audit_probability_ppm: u32,
    auditor_availability_ppm: u32,
    safety_factor: u32,
) -> Result<u64, CommitError> {
    if cheating_gain == 0 || safety_factor == 0 {
        return Err(CommitError::InvalidShape(
            "cheating gain and safety factor must be non-zero".to_owned(),
        ));
    }
    validate_ppm(audit_probability_ppm)?;
    validate_ppm(auditor_availability_ppm)?;
    let denominator = u128::from(audit_probability_ppm)
        .checked_mul(u128::from(auditor_availability_ppm))
        .ok_or(CommitError::ArithmeticOverflow)?;
    if denominator == 0 {
        return Err(CommitError::InvalidShape(
            "effective audit probability must be non-zero".to_owned(),
        ));
    }
    let ppm_squared = u128::from(PPM_DENOMINATOR).pow(2);
    let missed = ppm_squared
        .checked_sub(denominator)
        .ok_or(CommitError::ArithmeticOverflow)?;
    let numerator = u128::from(cheating_gain)
        .checked_mul(u128::from(safety_factor))
        .and_then(|value| value.checked_mul(missed))
        .ok_or(CommitError::ArithmeticOverflow)?;
    let collateral = numerator
        .checked_add(denominator - 1)
        .ok_or(CommitError::ArithmeticOverflow)?
        / denominator;
    u64::try_from(collateral).map_err(|_| CommitError::ArithmeticOverflow)
}

/// Expected fraction of honest inference work spent on mandatory replay.
/// Cost weighting belongs outside this helper when jobs have unequal sizes.
pub fn replay_overhead_fraction(
    audit_probability_ppm: u32,
    auditor_redundancy: u32,
) -> Result<f64, CommitError> {
    validate_ppm(audit_probability_ppm)?;
    if auditor_redundancy == 0 {
        return Err(CommitError::InvalidShape(
            "auditor redundancy must be non-zero".to_owned(),
        ));
    }
    Ok(
        f64::from(audit_probability_ppm) * f64::from(auditor_redundancy)
            / f64::from(PPM_DENOMINATOR),
    )
}

fn validate_ppm(value: u32) -> Result<(), CommitError> {
    if value > PPM_DENOMINATOR {
        return Err(CommitError::InvalidShape(
            "probability exceeds one million ppm".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_percent_requires_99x_collateral_at_unit_safety() {
        assert_eq!(
            minimum_deterrent_collateral(1_000, 10_000, 1_000_000, 1).unwrap(),
            99_000
        );
        assert_eq!(
            minimum_deterrent_collateral(1_000, 10_000, 1_000_000, 4).unwrap(),
            396_000
        );
    }

    #[test]
    fn offline_auditors_multiply_the_required_collateral() {
        let always_online = minimum_deterrent_collateral(1_000, 10_000, 1_000_000, 1).unwrap();
        let half_online = minimum_deterrent_collateral(1_000, 10_000, 500_000, 1).unwrap();
        assert_eq!(always_online, 99_000);
        assert_eq!(half_online, 199_000);
    }

    #[test]
    fn repeated_detection_and_replay_cost_have_expected_boundaries() {
        let detected = repeated_cheat_detection_probability(10_000, 1_000_000, 100).unwrap();
        assert!((detected - 0.633_967_658_7).abs() < 1e-10);
        assert_eq!(replay_overhead_fraction(10_000, 1).unwrap(), 0.01);
        assert_eq!(replay_overhead_fraction(10_000, 2).unwrap(), 0.02);
    }
}
