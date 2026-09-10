use crate::{CommitError, Digest, HashAlgorithm, hash_domain};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EpochBlockSeedV2 {
    pub protocol_instance: Digest,
    pub epoch: u64,
    pub miner: [u8; 32],
    pub result_root: Digest,
    pub result_count: u32,
    pub entropy_slot: u64,
}

impl EpochBlockSeedV2 {
    pub fn derive(&self, canonical_block_hash: &Digest) -> Result<Digest, CommitError> {
        if self.result_count == 0 {
            return Err(CommitError::InvalidShape(
                "an epoch seed requires at least one result".to_owned(),
            ));
        }
        hash_domain(
            HashAlgorithm::Blake3V1,
            crate::D_AUDIT_SEED_V2,
            &[
                &self.protocol_instance,
                &self.epoch.to_le_bytes(),
                &self.miner,
                &self.result_root,
                &self.result_count.to_le_bytes(),
                &self.entropy_slot.to_le_bytes(),
                canonical_block_hash,
            ],
        )
    }
}

/// Canonical preimage for one state in a disputed ordered Q4 FMA trace.
/// `column` is the next column to execute; `accumulator_bits` is the value
/// before that column. Adjacent states therefore differ by one column.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FmaStateV1 {
    pub job_hash: Digest,
    pub tensor_id: u32,
    pub row: u32,
    pub column: u32,
    pub activation_root: Digest,
    pub accumulator_bits: u32,
}

impl FmaStateV1 {
    pub fn hash(&self) -> Result<Digest, CommitError> {
        hash_domain(
            HashAlgorithm::Blake3V1,
            crate::D_FMA_STATE,
            &[
                &self.job_hash,
                &self.tensor_id.to_le_bytes(),
                &self.row.to_le_bytes(),
                &self.column.to_le_bytes(),
                &self.activation_root,
                &self.accumulator_bits.to_le_bytes(),
            ],
        )
    }

    pub fn successor_with_accumulator(&self, accumulator_bits: u32) -> Result<Self, CommitError> {
        Ok(Self {
            column: self
                .column
                .checked_add(1)
                .ok_or(CommitError::ArithmeticOverflow)?,
            accumulator_bits,
            ..*self
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MerkleTree;

    #[test]
    fn block_seed_binds_epoch_operator_and_future_hash() {
        let epoch = EpochBlockSeedV2 {
            protocol_instance: [1; 32],
            epoch: 7,
            miner: [2; 32],
            result_root: [3; 32],
            result_count: 99,
            entropy_slot: 1_000,
        };
        let first = epoch.derive(&[4; 32]).unwrap();
        let mut changed = epoch;
        changed.miner[0] ^= 1;
        assert_ne!(first, changed.derive(&[4; 32]).unwrap());
        assert_ne!(first, epoch.derive(&[5; 32]).unwrap());
    }

    #[test]
    fn fma_trace_states_are_openable_for_bisection() {
        let start = FmaStateV1 {
            job_hash: [1; 32],
            tensor_id: 4,
            row: 9,
            column: 33,
            activation_root: [2; 32],
            accumulator_bits: 1.0_f32.to_bits(),
        };
        let end = start.successor_with_accumulator(2.0_f32.to_bits()).unwrap();
        let leaves = [start.hash().unwrap(), end.hash().unwrap()];
        let tree =
            MerkleTree::new(HashAlgorithm::Blake3V1, crate::D_DISPUTE_TRACE, &leaves).unwrap();
        assert!(tree.verify(leaves[0], &tree.proof(0).unwrap()).unwrap());
        assert!(tree.verify(leaves[1], &tree.proof(1).unwrap()).unwrap());
    }
}
