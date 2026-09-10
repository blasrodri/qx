use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};

use crate::{CommitError, D_INPUT, D_OUTPUT, D_RECEIPT, Digest, ProtocolVersion, hash_domain};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InferenceReceiptV1 {
    pub protocol_version: u16,
    pub job_id: Digest,
    pub model_id: Digest,
    pub input_hash: Digest,
    pub output_hash: Digest,
    pub trace_root: Digest,
    pub input_token_count: u32,
    pub output_token_count: u32,
    pub miner_pubkey: [u8; 32],
    pub sequence: u64,
}

impl InferenceReceiptV1 {
    pub const CANONICAL_SIZE: usize = 210;

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CommitError> {
        ProtocolVersion::parse(self.protocol_version)?;
        let mut bytes = Vec::with_capacity(Self::CANONICAL_SIZE);
        bytes.extend_from_slice(&self.protocol_version.to_le_bytes());
        bytes.extend_from_slice(&self.job_id);
        bytes.extend_from_slice(&self.model_id);
        bytes.extend_from_slice(&self.input_hash);
        bytes.extend_from_slice(&self.output_hash);
        bytes.extend_from_slice(&self.trace_root);
        bytes.extend_from_slice(&self.input_token_count.to_le_bytes());
        bytes.extend_from_slice(&self.output_token_count.to_le_bytes());
        bytes.extend_from_slice(&self.miner_pubkey);
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        debug_assert_eq!(bytes.len(), Self::CANONICAL_SIZE);
        Ok(bytes)
    }

    pub fn hash(&self) -> Result<Digest, CommitError> {
        let protocol = ProtocolVersion::parse(self.protocol_version)?;
        hash_domain(
            protocol.hash_algorithm(),
            D_RECEIPT,
            &[&self.canonical_bytes()?],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignedInferenceReceiptV1 {
    pub receipt: InferenceReceiptV1,
    pub signature: [u8; 64],
}

impl SignedInferenceReceiptV1 {
    pub const CANONICAL_SIZE: usize = InferenceReceiptV1::CANONICAL_SIZE + 64;

    pub fn sign(receipt: InferenceReceiptV1, key: &SigningKey) -> Result<Self, CommitError> {
        if receipt.miner_pubkey != key.verifying_key().to_bytes() {
            return Err(CommitError::Signature(
                "receipt pubkey does not match signing key".to_owned(),
            ));
        }
        let signature = key.sign(&receipt.hash()?).to_bytes();
        Ok(Self { receipt, signature })
    }

    pub fn verify(&self) -> Result<(), CommitError> {
        let key = VerifyingKey::from_bytes(&self.receipt.miner_pubkey)
            .map_err(|error| CommitError::Signature(error.to_string()))?;
        let signature = Signature::from_bytes(&self.signature);
        key.verify(&self.receipt.hash()?, &signature)
            .map_err(|error| CommitError::Signature(error.to_string()))
    }

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CommitError> {
        let mut bytes = self.receipt.canonical_bytes()?;
        bytes.extend_from_slice(&self.signature);
        debug_assert_eq!(bytes.len(), Self::CANONICAL_SIZE);
        Ok(bytes)
    }
}

fn token_bytes(tokens: &[u32]) -> Result<Vec<u8>, CommitError> {
    let byte_length = tokens
        .len()
        .checked_mul(4)
        .ok_or(CommitError::ArithmeticOverflow)?;
    let mut bytes = Vec::with_capacity(byte_length);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    Ok(bytes)
}

pub fn input_hash_v1(model_id: &Digest, tokens: &[u32]) -> Result<Digest, CommitError> {
    hash_domain(
        ProtocolVersion::V1.hash_algorithm(),
        D_INPUT,
        &[model_id, &token_bytes(tokens)?],
    )
}

pub fn output_hash_v1(tokens: &[u32]) -> Result<Digest, CommitError> {
    hash_domain(
        ProtocolVersion::V1.hash_algorithm(),
        D_OUTPUT,
        &[&token_bytes(tokens)?],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_receipt_is_compact_and_verifies() {
        let key = SigningKey::from_bytes(&[7; 32]);
        let receipt = InferenceReceiptV1 {
            protocol_version: 1,
            job_id: [1; 32],
            model_id: [2; 32],
            input_hash: [3; 32],
            output_hash: [4; 32],
            trace_root: [5; 32],
            input_token_count: 3,
            output_token_count: 2,
            miner_pubkey: key.verifying_key().to_bytes(),
            sequence: 9,
        };
        let signed = SignedInferenceReceiptV1::sign(receipt, &key).unwrap();
        signed.verify().unwrap();
        assert_eq!(signed.canonical_bytes().unwrap().len(), 274);
    }
}
