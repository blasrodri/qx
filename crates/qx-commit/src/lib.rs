//! Consensus commitments for optimistic QX inference verification.
//!
//! All encodings in this crate are manual, versioned, little-endian, and
//! independent of Rust layout. The CPU implementation is the protocol oracle;
//! a future Metal implementation must match its golden vectors byte-for-byte.

mod epoch;
mod execution;
mod hash;
mod merkle;
mod model;
mod receipt;
mod state;
mod weights;

pub use epoch::{
    AuditEpochV1, audit_detection_probability, derive_audit_seed_v1, required_audit_count,
    sample_receipt_indices_v1,
};
pub use execution::{CheckpointV1, CommitmentMetrics, ExecutionCommitterV1, ExecutionTraceV1};
pub use hash::{Digest, HashAlgorithm, ProtocolVersion, hash_domain, hash_domain_input_len};
pub use merkle::{FixedMerkleAccumulator, MerkleProof, MerkleTree, MutableMerkleTree};
pub use model::{
    ModelSpecV1, granite_architecture_root, granite_execution_spec,
    granite_execution_spec_with_checkpoint_interval,
};
pub use receipt::{InferenceReceiptV1, SignedInferenceReceiptV1, input_hash_v1, output_hash_v1};
pub use state::{
    ACTIVATION_CHUNK_BYTES, STATE_CHUNK_BYTES, StateChunkProofV1, StateTreeKindV1,
    activation_hash_v1, kv_leaf_hash_v1, prove_state_chunk_v1, state_hash_v1,
};
pub use weights::{
    WeightCommitmentV1, WeightLeafV1, WeightProofV1, commit_q4_artifact, prove_q4_artifact_leaf,
};

pub const D_MODEL: &[u8] = b"QX:MODEL:V1";
pub const D_WEIGHT_LEAF: &[u8] = b"QX:WEIGHT:V1";
pub const D_MERKLE: &[u8] = b"QX:MERKLE:V1";
pub const D_MERKLE_EMPTY: &[u8] = b"QX:MERKLE:EMPTY:V1";
pub const D_KV: &[u8] = b"QX:KV:V1";
pub const D_STATE: &[u8] = b"QX:STATE:V1";
pub const D_STATE_CHUNK: &[u8] = b"QX:STATE:CHUNK:V1";
pub const D_KV_KEY_CHUNK: &[u8] = b"QX:KV:KEY:CHUNK:V1";
pub const D_KV_KEY_TREE: &[u8] = b"QX:KV:KEY:TREE:V1";
pub const D_KV_VALUE_CHUNK: &[u8] = b"QX:KV:VALUE:CHUNK:V1";
pub const D_KV_VALUE_TREE: &[u8] = b"QX:KV:VALUE:TREE:V1";
pub const D_ACTIVATION_CHUNK: &[u8] = b"QX:ACTIVATION:CHUNK:V1";
pub const D_ACTIVATION_TREE: &[u8] = b"QX:ACTIVATION:TREE:V1";
pub const D_CHECKPOINT: &[u8] = b"QX:CHECKPOINT:V1";
pub const D_TOKEN: &[u8] = b"QX:TOKEN:V1";
pub const D_JOB: &[u8] = b"QX:JOB:V1";
pub const D_RECEIPT: &[u8] = b"QX:RECEIPT:V1";
pub const D_BATCH: &[u8] = b"QX:BATCH:V1";
pub const D_INPUT: &[u8] = b"QX:INPUT:V1";
pub const D_OUTPUT: &[u8] = b"QX:OUTPUT:V1";
pub const D_ARCHITECTURE: &[u8] = b"QX:ARCHITECTURE:V1";
pub const D_EXECUTION_SPEC: &[u8] = b"QX:EXECUTION:V1";
pub const D_WEIGHT_TREE: &[u8] = b"QX:WEIGHT:TREE:V1";
pub const D_AUDIT_SEED: &[u8] = b"QX:AUDIT:SEED:V1";
pub const D_AUDIT_DRAW: &[u8] = b"QX:AUDIT:DRAW:V1";

pub const MAX_MERKLE_DEPTH: usize = 32;
/// Hot per-token commitment trees use the Solana-supported BLAKE3 syscall;
/// cold model, weight, and receipt objects retain their existing algorithms.
pub const TRACE_HASH_ALGORITHM: HashAlgorithm = HashAlgorithm::Blake3V1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommitError {
    UnsupportedProtocol(u16),
    UnsupportedHash(HashAlgorithm),
    EmptyTree,
    TreeCapacityExceeded,
    InvalidMerkleDepth(usize),
    InvalidMerkleProof,
    InvalidIndex { index: usize, length: usize },
    InvalidShape(String),
    InvalidArtifact(String),
    InvalidExecution(String),
    ArithmeticOverflow,
    Signature(String),
}

impl std::fmt::Display for CommitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedProtocol(value) => write!(f, "unsupported protocol version {value}"),
            Self::UnsupportedHash(value) => write!(f, "unsupported hash algorithm {value:?}"),
            Self::EmptyTree => f.write_str("Merkle tree must contain at least one leaf"),
            Self::TreeCapacityExceeded => f.write_str("Merkle tree capacity exceeded"),
            Self::InvalidMerkleDepth(value) => write!(f, "invalid Merkle depth {value}"),
            Self::InvalidMerkleProof => f.write_str("invalid Merkle proof"),
            Self::InvalidIndex { index, length } => {
                write!(f, "index {index} is outside length {length}")
            }
            Self::InvalidShape(value) => write!(f, "invalid shape: {value}"),
            Self::InvalidArtifact(value) => write!(f, "invalid artifact: {value}"),
            Self::InvalidExecution(value) => write!(f, "invalid execution: {value}"),
            Self::ArithmeticOverflow => f.write_str("checked arithmetic overflow"),
            Self::Signature(value) => write!(f, "signature error: {value}"),
        }
    }
}

impl std::error::Error for CommitError {}
