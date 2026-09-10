use crate::{CommitError, D_AUDIT_DRAW, D_AUDIT_SEED, Digest, HashAlgorithm, hash_domain};

/// Consensus inputs that must be fixed before the future entropy is known.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuditEpochV1 {
    pub epoch: u64,
    pub receipt_root: Digest,
    pub receipt_count: u32,
    pub audit_count: u32,
    pub entropy_slot: u64,
}

impl AuditEpochV1 {
    pub fn validate(&self) -> Result<(), CommitError> {
        if self.receipt_count == 0 || self.audit_count == 0 || self.audit_count > self.receipt_count
        {
            return Err(CommitError::InvalidShape(
                "epoch requires 0 < audit_count <= receipt_count".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Mix a future Solana entropy value with all closed epoch coordinates.
/// `independent_entropy` is an optional aggregate of committed auditor reveals;
/// it may be empty, but must be fixed by the protocol before sampling starts.
pub fn derive_audit_seed_v1(
    epoch: &AuditEpochV1,
    future_solana_entropy: &Digest,
    independent_entropy: &[u8],
) -> Result<Digest, CommitError> {
    epoch.validate()?;
    hash_domain(
        HashAlgorithm::Blake3V1,
        D_AUDIT_SEED,
        &[
            &epoch.epoch.to_le_bytes(),
            &epoch.receipt_root,
            &epoch.receipt_count.to_le_bytes(),
            &epoch.audit_count.to_le_bytes(),
            &epoch.entropy_slot.to_le_bytes(),
            future_solana_entropy,
            independent_entropy,
        ],
    )
}

/// Select exactly `audit_count` distinct positions from the committed receipt
/// tree. Rejection sampling removes modulo bias; duplicate positions are drawn
/// again. An on-chain verifier can reproduce this loop from the epoch account.
pub fn sample_receipt_indices_v1(
    epoch: &AuditEpochV1,
    seed: &Digest,
) -> Result<Vec<u32>, CommitError> {
    epoch.validate()?;
    let count = u64::from(epoch.receipt_count);
    let rejection_floor = count.wrapping_neg() % count;
    let mut selected = Vec::with_capacity(epoch.audit_count as usize);
    let mut draw = 0_u64;
    while selected.len() < epoch.audit_count as usize {
        let digest = hash_domain(
            HashAlgorithm::Blake3V1,
            D_AUDIT_DRAW,
            &[seed, &draw.to_le_bytes()],
        )?;
        draw = draw.checked_add(1).ok_or(CommitError::ArithmeticOverflow)?;
        let candidate = u64::from_le_bytes(
            digest[..8]
                .try_into()
                .map_err(|_| CommitError::InvalidExecution("invalid audit digest".to_owned()))?,
        );
        if candidate < rejection_floor {
            continue;
        }
        let index =
            u32::try_from(candidate % count).map_err(|_| CommitError::ArithmeticOverflow)?;
        if !selected.contains(&index) {
            selected.push(index);
        }
    }
    Ok(selected)
}

/// Probability that a uniform sample without replacement contains at least
/// one of `cheated_receipts` bad receipts.
pub fn audit_detection_probability(
    receipt_count: u32,
    cheated_receipts: u32,
    audit_count: u32,
) -> Result<f64, CommitError> {
    if receipt_count == 0 || cheated_receipts > receipt_count || audit_count > receipt_count {
        return Err(CommitError::InvalidShape(
            "invalid population, cheat count, or audit count".to_owned(),
        ));
    }
    if cheated_receipts == 0 || audit_count == 0 {
        return Ok(0.0);
    }
    if audit_count > receipt_count - cheated_receipts {
        return Ok(1.0);
    }
    let mut miss = 1.0_f64;
    for draw in 0..audit_count {
        miss *=
            f64::from(receipt_count - cheated_receipts - draw) / f64::from(receipt_count - draw);
    }
    Ok(1.0 - miss)
}

pub fn required_audit_count(
    receipt_count: u32,
    cheated_receipts: u32,
    target_detection_probability: f64,
) -> Result<u32, CommitError> {
    if !(0.0..=1.0).contains(&target_detection_probability)
        || receipt_count == 0
        || cheated_receipts == 0
        || cheated_receipts > receipt_count
    {
        return Err(CommitError::InvalidShape(
            "invalid audit sizing inputs".to_owned(),
        ));
    }
    for audits in 0..=receipt_count {
        if audit_detection_probability(receipt_count, cheated_receipts, audits)?
            >= target_detection_probability
        {
            return Ok(audits);
        }
    }
    Ok(receipt_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn epoch() -> AuditEpochV1 {
        AuditEpochV1 {
            epoch: 7,
            receipt_root: [3; 32],
            receipt_count: 1_000,
            audit_count: 25,
            entropy_slot: 99,
        }
    }

    #[test]
    fn selection_is_deterministic_distinct_and_in_range() {
        let epoch = epoch();
        let seed = derive_audit_seed_v1(&epoch, &[9; 32], b"auditor aggregate").unwrap();
        let first = sample_receipt_indices_v1(&epoch, &seed).unwrap();
        let second = sample_receipt_indices_v1(&epoch, &seed).unwrap();
        assert_eq!(
            first,
            vec![
                906, 545, 292, 690, 272, 142, 933, 189, 400, 312, 478, 307, 641, 927, 80, 126, 714,
                925, 93, 23, 184, 449, 960, 444, 61,
            ]
        );
        assert_eq!(first, second);
        assert_eq!(first.len(), 25);
        assert!(first.iter().all(|index| *index < 1_000));
        let mut sorted = first;
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 25);
    }

    #[test]
    fn sampling_probability_matches_boundary_cases() {
        assert_eq!(audit_detection_probability(100, 0, 10).unwrap(), 0.0);
        assert_eq!(audit_detection_probability(100, 91, 10).unwrap(), 1.0);
        let probability = audit_detection_probability(1_000, 10, 100).unwrap();
        assert!((probability - 0.653_072_285_2).abs() < 1e-9);
        assert_eq!(required_audit_count(1_000, 10, probability).unwrap(), 100);
    }

    #[test]
    fn seed_binds_the_closed_epoch() {
        let mut changed = epoch();
        let baseline = derive_audit_seed_v1(&changed, &[9; 32], &[]).unwrap();
        changed.receipt_root[0] ^= 1;
        assert_ne!(
            baseline,
            derive_audit_seed_v1(&changed, &[9; 32], &[]).unwrap()
        );
    }
}
