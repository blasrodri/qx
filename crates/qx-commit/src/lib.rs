//! Consensus commitments for optimistic QX inference verification.
//!
//! All encodings in this crate are manual, versioned, little-endian, and
//! independent of Rust layout. The CPU implementation is the protocol oracle;
//! a future Metal implementation must match its golden vectors byte-for-byte.

mod dispute;
mod economics;
mod epoch;
mod execution;
mod granite;
mod hash;
mod job;
mod merkle;
mod model;
mod receipt;
mod state;
mod vm;
mod weights;

pub use dispute::{EpochBlockSeedV2, FmaStateV1};
pub use economics::{
    PPM_DENOMINATOR, effective_detection_probability, minimum_deterrent_collateral,
    repeated_cheat_detection_probability, replay_overhead_fraction,
};
pub use epoch::{
    AuditEpochV1, audit_detection_probability, audit_selected_v1, derive_audit_seed_v1,
    required_audit_count, sample_receipt_indices_v1,
};
pub use execution::{
    CheckpointV1, CommitmentMetrics, ExecutionCommitterV1, ExecutionTraceV1,
    TOKEN_GENESIS_LAYER_V1, TokenResultV1, job_genesis_hash_v1, token_input_state_hash_v1,
};
pub use granite::{
    GRANITE_TENSORS_PER_LAYER_V1, GraniteDenseOutputV1, GraniteLayerTensorIdsV1,
    GraniteMacroOpcodeV1, GraniteMacroOutputV1, GraniteMacroPlanV1, GraniteMacroStateV1,
    GraniteMacroTraceV1, GraniteMacroWorkspaceV1, GraniteMicroLevelV1, GraniteMicroStateV1,
    GraniteMicroTraceV1, GraniteWorkspaceSlotV1, granite_embedding_macro_program_v1,
    granite_embedding_tensor_id_v1, granite_final_head_macro_program_v1,
    granite_final_norm_tensor_id_v1, granite_layer_macro_program_v1, granite_layer_tensor_ids_v1,
    granite_macro_plan_v1, granite_macro_trace_v1, granite_micro_transition_count_v1,
    validate_granite_macro_artifact_v1,
};
pub use hash::{Digest, HashAlgorithm, ProtocolVersion, hash_domain, hash_domain_input_len};
pub use job::{JobEnvelopeV1, JobPayloadV1, MIN_AUDIT_PROBABILITY_PPM_V1, ResultLeafV1};
pub use merkle::{FixedMerkleAccumulator, MerkleProof, MerkleTree, MutableMerkleTree};
pub use model::{
    GraniteMathFunctionsV1, GraniteMathModeV1, ModelSpecV1, granite_architecture_root,
    granite_execution_spec, granite_execution_spec_with_checkpoint_interval,
    granite_execution_spec_with_math_functions, granite_execution_spec_with_math_profile,
};
pub use receipt::{
    InferenceReceiptV1, SignedInferenceReceiptV1, input_hash_v1, input_token_tree_v1,
    output_hash_v1,
};
pub use state::{
    ACTIVATION_CHUNK_BYTES, STATE_CHUNK_BYTES, StateChunkProofV1, StateTreeKindV1,
    activation_hash_v1, kv_leaf_hash_from_roots_v1, kv_leaf_hash_v1, prove_state_chunk_v1,
    state_hash_v1,
};
pub use vm::{
    Q4_SELECTOR_LOCAL_BITS, VmInstructionV1, VmMemoryOpeningV1, VmOutputStateV1,
    VmSegmentProgramV1, VmStateV1, decode_q4_weight_selector_v1, q4_weight_selector_v1,
    vm_program_tree_v1, vm_segment_schedule_v1,
};
pub use weights::{
    SemanticTensorV1, WeightCommitmentV1, WeightLeafV1, WeightProofV1,
    artifact_semantic_tensors_v1, commit_q4_artifact, prove_q4_artifact_leaf,
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
pub const D_TOKEN_RESULT: &[u8] = b"QX:TOKEN:RESULT:V1";
pub const D_TOKEN_INPUT_STATE: &[u8] = b"QX:TOKEN:INPUT:STATE:V1";
pub const D_JOB: &[u8] = b"QX:JOB:V1";
pub const D_JOB_GENESIS: &[u8] = b"QX:JOB:GENESIS:V1";
pub const D_JOB_PAYLOAD: &[u8] = b"QX:JOB:PAYLOAD:V1";
pub const D_RESULT: &[u8] = b"QX:RESULT:V1";
pub const D_RECEIPT: &[u8] = b"QX:RECEIPT:V1";
pub const D_BATCH: &[u8] = b"QX:BATCH:V1";
pub const D_INPUT: &[u8] = b"QX:INPUT:V1";
pub const D_INPUT_TOKEN: &[u8] = b"QX:INPUT:TOKEN:V1";
pub const D_INPUT_TREE: &[u8] = b"QX:INPUT:TREE:V1";
pub const D_OUTPUT: &[u8] = b"QX:OUTPUT:V1";
pub const D_ARCHITECTURE: &[u8] = b"QX:ARCHITECTURE:V1";
pub const D_EXECUTION_SPEC: &[u8] = b"QX:EXECUTION:V1";
pub const D_WEIGHT_TREE: &[u8] = b"QX:WEIGHT:TREE:V1";
pub const D_AUDIT_SEED: &[u8] = b"QX:AUDIT:SEED:V1";
pub const D_AUDIT_SEED_V2: &[u8] = b"QX:AUDIT:SEED:V2";
pub const D_AUDIT_DRAW: &[u8] = b"QX:AUDIT:DRAW:V1";
pub const D_DISPUTE_TRACE: &[u8] = b"QX:DISPUTE:TRACE:V1";
pub const D_FMA_STATE: &[u8] = b"QX:FMA:STATE:V1";
pub const D_VM_STATE: &[u8] = b"QX:VM:STATE:V1";
pub const D_VM_OUTPUT: &[u8] = b"QX:VM:OUTPUT:V1";
pub const D_VM_INSTRUCTION: &[u8] = b"QX:VM:INSTRUCTION:V1";
pub const D_VM_SEGMENT_PROGRAM: &[u8] = b"QX:VM:SEGMENT:V1";
pub const D_PROGRAM_TREE: &[u8] = b"QX:PROGRAM:TREE:V1";
pub const D_SEGMENT_PROGRAM_TREE: &[u8] = b"QX:SEGMENT:PROGRAM:TREE:V1";
pub const D_OPERATOR_TREE: &[u8] = b"QX:OPERATOR:V1";
pub const D_MACRO_WORKSPACE: &[u8] = b"QX:MACRO:WORKSPACE:V1";
pub const D_MACRO_WORKSPACE_SLOT: &[u8] = b"QX:MACRO:WORKSPACE:SLOT:V1";
pub const D_MACRO_STATE: &[u8] = b"QX:MACRO:STATE:V1";
pub const D_GRANITE_MICRO_STATE: &[u8] = b"QX:GRANITE:MICRO:STATE:V1";
pub const D_GRANITE_MICRO_TREE: &[u8] = b"QX:GRANITE:MICRO:TREE:V1";

pub const MAX_MERKLE_DEPTH: usize = 32;
/// Hot per-token commitment trees use BLAKE3. On clusters where the native
/// syscall is unavailable, the rare on-chain dispute path evaluates it in SBF;
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
