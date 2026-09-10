use sha2::{Digest as _, Sha256};

use crate::{CommitError, D_MERKLE_EMPTY};

pub type Digest = [u8; 32];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum HashAlgorithm {
    Sha256V1 = 1,
    Blake3V1 = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum ProtocolVersion {
    V1 = 1,
}

impl ProtocolVersion {
    pub const fn hash_algorithm(self) -> HashAlgorithm {
        match self {
            Self::V1 => HashAlgorithm::Sha256V1,
        }
    }

    pub fn parse(value: u16) -> Result<Self, CommitError> {
        match value {
            1 => Ok(Self::V1),
            _ => Err(CommitError::UnsupportedProtocol(value)),
        }
    }
}

/// Hash a domain and length-framed fields. Field lengths are unsigned u64 LE.
pub fn hash_domain(
    algorithm: HashAlgorithm,
    domain: &'static [u8],
    fields: &[&[u8]],
) -> Result<Digest, CommitError> {
    match algorithm {
        HashAlgorithm::Sha256V1 => {
            let mut hasher = Sha256::new();
            hasher.update(domain);
            for field in fields {
                let length =
                    u64::try_from(field.len()).map_err(|_| CommitError::ArithmeticOverflow)?;
                hasher.update(length.to_le_bytes());
                hasher.update(field);
            }
            Ok(hasher.finalize().into())
        }
        HashAlgorithm::Blake3V1 => {
            let mut hasher = blake3::Hasher::new();
            hasher.update(domain);
            for field in fields {
                let length =
                    u64::try_from(field.len()).map_err(|_| CommitError::ArithmeticOverflow)?;
                hasher.update(&length.to_le_bytes());
                hasher.update(field);
            }
            Ok(*hasher.finalize().as_bytes())
        }
    }
}

pub fn hash_domain_input_len(domain: &[u8], fields: &[&[u8]]) -> Result<u64, CommitError> {
    fields.iter().try_fold(
        u64::try_from(domain.len()).map_err(|_| CommitError::ArithmeticOverflow)?,
        |total, field| {
            total
                .checked_add(8)
                .and_then(|value| value.checked_add(field.len() as u64))
                .ok_or(CommitError::ArithmeticOverflow)
        },
    )
}

pub(crate) fn empty_leaf(
    algorithm: HashAlgorithm,
    tree_domain: &'static [u8],
    _index: u64,
) -> Result<Digest, CommitError> {
    hash_domain(algorithm, D_MERKLE_EMPTY, &[tree_domain])
}
