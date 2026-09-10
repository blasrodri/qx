use crate::{CommitError, Digest, HashAlgorithm, ProtocolVersion, hash_domain};

/// Immutable request data. The requester or scheduling protocol authorizes
/// this payload; a miner does not sign it again per result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobPayloadV1 {
    pub protocol_version: u16,
    pub protocol_instance: Digest,
    pub job_id: Digest,
    pub requester: [u8; 32],
    pub assigned_miner: [u8; 32],
    pub model_id: Digest,
    pub execution_spec: Digest,
    pub input_hash: Digest,
    pub input_token_count: u32,
    pub max_output_tokens: u32,
    pub payment: u64,
    pub slashable_exposure: u64,
    pub deadline_slot: u64,
    pub nonce: u64,
    /// Protocol-set probability in parts per million. It is committed in the
    /// job before the epoch entropy exists and cannot be chosen by the miner.
    pub audit_probability_ppm: u32,
    /// The unique miner epoch in which this job may be committed. Binding it
    /// before entropy prevents replaying one result across many sampling draws.
    pub assigned_epoch: u64,
    /// Exact on-chain model registry account selected by the requester. This
    /// prevents substituting another account that repeats the model ID with a
    /// different weight root.
    pub model_registry: [u8; 32],
}

impl JobPayloadV1 {
    pub const CANONICAL_SIZE: usize = 310;

    pub fn validate(&self) -> Result<(), CommitError> {
        ProtocolVersion::parse(self.protocol_version)?;
        if self.input_token_count == 0
            || self.max_output_tokens == 0
            || self.slashable_exposure == 0
            || self.audit_probability_ppm > 1_000_000
        {
            return Err(CommitError::InvalidShape(
                "invalid job counts, exposure, or audit probability".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CommitError> {
        self.validate()?;
        let mut bytes = Vec::with_capacity(Self::CANONICAL_SIZE);
        bytes.extend_from_slice(&self.protocol_version.to_le_bytes());
        bytes.extend_from_slice(&self.protocol_instance);
        bytes.extend_from_slice(&self.job_id);
        bytes.extend_from_slice(&self.requester);
        bytes.extend_from_slice(&self.assigned_miner);
        bytes.extend_from_slice(&self.model_id);
        bytes.extend_from_slice(&self.execution_spec);
        bytes.extend_from_slice(&self.input_hash);
        bytes.extend_from_slice(&self.input_token_count.to_le_bytes());
        bytes.extend_from_slice(&self.max_output_tokens.to_le_bytes());
        bytes.extend_from_slice(&self.payment.to_le_bytes());
        bytes.extend_from_slice(&self.slashable_exposure.to_le_bytes());
        bytes.extend_from_slice(&self.deadline_slot.to_le_bytes());
        bytes.extend_from_slice(&self.nonce.to_le_bytes());
        bytes.extend_from_slice(&self.audit_probability_ppm.to_le_bytes());
        bytes.extend_from_slice(&self.assigned_epoch.to_le_bytes());
        bytes.extend_from_slice(&self.model_registry);
        debug_assert_eq!(bytes.len(), Self::CANONICAL_SIZE);
        Ok(bytes)
    }

    pub fn hash(&self) -> Result<Digest, CommitError> {
        hash_domain(
            ProtocolVersion::parse(self.protocol_version)?.hash_algorithm(),
            crate::D_JOB_PAYLOAD,
            &[&self.canonical_bytes()?],
        )
    }
}

/// Unsigned result committed below a miner-authenticated epoch root.
/// Attribution comes from the epoch submission, not a per-job signature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultLeafV1 {
    pub job_hash: Digest,
    pub output_hash: Digest,
    pub trace_root: Digest,
    pub output_token_count: u32,
    /// Number of leaves under `trace_root`. The miner commits this so a
    /// challenger cannot choose a proof shape that forces an honest timeout.
    pub trace_length: u32,
}

impl ResultLeafV1 {
    pub const CANONICAL_SIZE: usize = 104;

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::CANONICAL_SIZE);
        bytes.extend_from_slice(&self.job_hash);
        bytes.extend_from_slice(&self.output_hash);
        bytes.extend_from_slice(&self.trace_root);
        bytes.extend_from_slice(&self.output_token_count.to_le_bytes());
        bytes.extend_from_slice(&self.trace_length.to_le_bytes());
        debug_assert_eq!(bytes.len(), Self::CANONICAL_SIZE);
        bytes
    }

    pub fn hash(&self) -> Result<Digest, CommitError> {
        if self.output_token_count == 0 || self.trace_length < 2 {
            return Err(CommitError::InvalidShape(
                "result output token count must be non-zero and trace length at least two"
                    .to_owned(),
            ));
        }
        hash_domain(
            HashAlgorithm::Sha256V1,
            crate::D_RESULT,
            &[&self.canonical_bytes()],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job() -> JobPayloadV1 {
        JobPayloadV1 {
            protocol_version: 1,
            protocol_instance: [1; 32],
            job_id: [2; 32],
            requester: [3; 32],
            assigned_miner: [4; 32],
            model_id: [5; 32],
            execution_spec: [6; 32],
            input_hash: [7; 32],
            input_token_count: 8,
            max_output_tokens: 16,
            payment: 1_000,
            slashable_exposure: 2_000,
            deadline_slot: 99,
            nonce: 3,
            audit_probability_ppm: 10_000,
            assigned_epoch: 7,
            model_registry: [8; 32],
        }
    }

    #[test]
    fn job_and_result_have_fixed_canonical_encodings() {
        let job = job();
        assert_eq!(
            job.canonical_bytes().unwrap().len(),
            JobPayloadV1::CANONICAL_SIZE
        );
        let result = ResultLeafV1 {
            job_hash: job.hash().unwrap(),
            output_hash: [8; 32],
            trace_root: [9; 32],
            output_token_count: 4,
            trace_length: 3,
        };
        assert_eq!(result.canonical_bytes().len(), ResultLeafV1::CANONICAL_SIZE);
        assert_ne!(job.hash().unwrap(), result.hash().unwrap());
    }

    #[test]
    fn invalid_miner_controlled_sampling_probability_is_rejected() {
        let mut invalid = job();
        invalid.audit_probability_ppm = 1_000_001;
        assert!(invalid.hash().is_err());
    }

    #[test]
    fn result_hash_binds_trace_length() {
        let mut result = ResultLeafV1 {
            job_hash: [1; 32],
            output_hash: [2; 32],
            trace_root: [3; 32],
            output_token_count: 4,
            trace_length: 8,
        };
        let original = result.hash().unwrap();
        result.trace_length = 9;
        assert_ne!(original, result.hash().unwrap());
    }
}
