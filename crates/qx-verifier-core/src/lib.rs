//! Small deterministic primitives used at the end of an optimistic QX
//! dispute. Model/layer decomposition stays off-chain; this crate checks only
//! authenticated state, one exact arithmetic transition, and ordering claims.

use half::f16;
use qx_commit::{Digest, StateTreeKindV1, WeightProofV1};

pub const Q4_GROUP_SIZE: u32 = 32;
pub const Q4_QUANTIZATION_TYPE: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum QxOpV1 {
    Dot = 1,
    Add = 2,
    Mul = 3,
    QuantizedFma = 4,
    ReduceSum = 5,
    ReduceMax = 6,
    Compare = 7,
    ArgMax = 8,
    Rsqrt = 9,
    ExpApprox = 10,
    KvAppend = 11,
    Div = 12,
    Select = 13,
    RawWeightFma = 14,
}

impl TryFrom<u8> for QxOpV1 {
    type Error = VerificationError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Dot),
            2 => Ok(Self::Add),
            3 => Ok(Self::Mul),
            4 => Ok(Self::QuantizedFma),
            5 => Ok(Self::ReduceSum),
            6 => Ok(Self::ReduceMax),
            7 => Ok(Self::Compare),
            8 => Ok(Self::ArgMax),
            9 => Ok(Self::Rsqrt),
            10 => Ok(Self::ExpApprox),
            11 => Ok(Self::KvAppend),
            12 => Ok(Self::Div),
            13 => Ok(Self::Select),
            14 => Ok(Self::RawWeightFma),
            _ => Err(VerificationError::InvalidWitness("unknown QX operation")),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VerificationError {
    InvalidWitness(&'static str),
    AuthenticationFailed(&'static str),
    ArithmeticClaimFailed,
    Overflow,
    Commitment(String),
}

impl std::fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWitness(message) => write!(formatter, "invalid witness: {message}"),
            Self::AuthenticationFailed(message) => {
                write!(formatter, "authentication failed: {message}")
            }
            Self::ArithmeticClaimFailed => formatter.write_str("arithmetic claim failed"),
            Self::Overflow => formatter.write_str("checked arithmetic overflow"),
            Self::Commitment(message) => write!(formatter, "commitment error: {message}"),
        }
    }
}

impl std::error::Error for VerificationError {}

/// Evaluate a scalar verification-IR transition. Operands are raw binary32
/// bits so the commitment layer never depends on text or host serialization.
/// Vector operations reduce to these scalar steps during a terminal dispute.
pub fn evaluate_scalar_op_v1(op: QxOpV1, operands: [u32; 3]) -> Result<u32, VerificationError> {
    let [a_bits, b_bits, c_bits] = operands;
    let a = finite_f32(a_bits)?;
    let b = finite_f32(b_bits)?;
    let c = finite_f32(c_bits)?;
    let result = match op {
        QxOpV1::Dot | QxOpV1::QuantizedFma => libm_fma(a, b, c),
        QxOpV1::Add | QxOpV1::ReduceSum => a + b,
        QxOpV1::Mul => a * b,
        QxOpV1::ReduceMax => {
            if ordered_finite_ge(a_bits, b_bits) {
                a
            } else {
                b
            }
        }
        QxOpV1::Compare => {
            if ordered_finite_ge(a_bits, b_bits) {
                1.0
            } else {
                0.0
            }
        }
        QxOpV1::Rsqrt => qx_consensus_math::rsqrt_f32(a)
            .ok_or(VerificationError::InvalidWitness("invalid rsqrt input"))?,
        QxOpV1::ExpApprox => qx_consensus_math::exp_f32(a)
            .ok_or(VerificationError::InvalidWitness("invalid exp input"))?,
        QxOpV1::Div => qx_consensus_math::div_f32(a, b)
            .ok_or(VerificationError::InvalidWitness("invalid division"))?,
        QxOpV1::Select => match a_bits {
            0x0000_0000 => c,
            0x3f80_0000 => b,
            _ => {
                return Err(VerificationError::InvalidWitness(
                    "select condition must be exactly 0.0 or 1.0",
                ));
            }
        },
        QxOpV1::ArgMax | QxOpV1::KvAppend | QxOpV1::RawWeightFma => {
            return Err(VerificationError::InvalidWitness(
                "operation requires a structured terminal witness",
            ));
        }
    };
    if !result.is_finite() {
        return Err(VerificationError::InvalidWitness("non-finite result"));
    }
    Ok(result.to_bits())
}

/// Complete witness for one chained verification-VM scalar transition. The
/// registered instruction binds addresses while each value is authenticated
/// against the common pre-state memory root. The destination's old opening is
/// reused to calculate the unique post-state root.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmScalarWitnessV1 {
    pub pre_state: qx_commit::VmStateV1,
    pub instruction: qx_commit::VmInstructionV1,
    pub instruction_proof: qx_commit::MerkleProof,
    pub memory: qx_commit::VmMemoryOpeningV1,
}

pub fn verify_vm_scalar_transition_v1(
    registered_program_root: Digest,
    witness: &VmScalarWitnessV1,
) -> Result<qx_commit::VmStateV1, VerificationError> {
    let pc = witness.pre_state.program_counter;
    if witness.instruction_proof.leaf_index != pc
        || !witness
            .instruction_proof
            .verify(
                qx_commit::HashAlgorithm::Blake3V1,
                qx_commit::D_PROGRAM_TREE,
                witness.instruction.hash(pc).map_err(commitment_error)?,
                registered_program_root,
            )
            .map_err(commitment_error)?
    {
        return Err(VerificationError::AuthenticationFailed(
            "verification instruction",
        ));
    }
    if !witness
        .memory
        .verify(witness.pre_state.residual_root)
        .map_err(commitment_error)?
    {
        return Err(VerificationError::AuthenticationFailed(
            "VM residual memory",
        ));
    }
    let addresses = [
        witness.instruction.source_a,
        witness.instruction.source_b,
        witness.instruction.source_c,
    ];
    let mut operands = [0_u32; 3];
    for (index, address) in addresses.into_iter().enumerate() {
        operands[index] = witness
            .memory
            .scalar_bits(address)
            .map_err(commitment_error)?;
    }
    let result = evaluate_scalar_op_v1(QxOpV1::try_from(witness.instruction.opcode)?, operands)?;
    let residual_root = witness
        .memory
        .root_after_scalar_write(witness.instruction.destination, result)
        .map_err(commitment_error)?;
    witness
        .pre_state
        .successor(residual_root)
        .map_err(commitment_error)
}

/// Complete host-oracle witness for a Q4 FMA in the checkpoint-bound VM.
/// `source_c` packs the canonical weight-leaf index and its five-bit local Q4
/// position; `source_a` is independent dispute-local activation memory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmQ4WitnessV1 {
    pub pre_state: qx_commit::VmStateV1,
    pub instruction: qx_commit::VmInstructionV1,
    pub instruction_proof: qx_commit::MerkleProof,
    pub memory: qx_commit::VmMemoryOpeningV1,
    pub weight: WeightProofV1,
}

pub fn verify_vm_q4_transition_v1(
    registered_program_root: Digest,
    registered_weights_root: Digest,
    witness: &VmQ4WitnessV1,
) -> Result<qx_commit::VmStateV1, VerificationError> {
    let pc = witness.pre_state.program_counter;
    let (weight_leaf_index, local) =
        qx_commit::decode_q4_weight_selector_v1(witness.instruction.source_c);
    if witness.instruction.opcode != QxOpV1::QuantizedFma as u8
        || witness.instruction.source_a & 3 != 0
        || weight_leaf_index != witness.weight.proof.leaf_index
        || witness.instruction_proof.leaf_index != pc
        || !witness
            .instruction_proof
            .verify(
                qx_commit::HashAlgorithm::Blake3V1,
                qx_commit::D_PROGRAM_TREE,
                witness.instruction.hash(pc).map_err(commitment_error)?,
                registered_program_root,
            )
            .map_err(commitment_error)?
    {
        return Err(VerificationError::AuthenticationFailed(
            "Q4 verification instruction",
        ));
    }
    if witness.weight.commitment.root != registered_weights_root
        || !witness.weight.verify().map_err(commitment_error)?
    {
        return Err(VerificationError::AuthenticationFailed("Q4 weight"));
    }
    let leaf = &witness.weight.leaf;
    if leaf.quantization_type != Q4_QUANTIZATION_TYPE || leaf.packed_values.len() != 16 {
        return Err(VerificationError::InvalidWitness("Q4 weight coordinates"));
    }
    if !witness
        .memory
        .verify(witness.pre_state.residual_root)
        .map_err(commitment_error)?
    {
        return Err(VerificationError::AuthenticationFailed(
            "VM residual memory",
        ));
    }
    let activation = finite_f32(
        witness
            .memory
            .scalar_bits(witness.instruction.source_a)
            .map_err(commitment_error)?,
    )?;
    let accumulator = finite_f32(
        witness
            .memory
            .scalar_bits(witness.instruction.source_b)
            .map_err(commitment_error)?,
    )?;
    let local = usize::from(local);
    let packed = leaf.packed_values[local / 2];
    let code = if local & 1 == 0 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let scale = f16::from_bits(leaf.scale_bits).to_f32();
    if !scale.is_finite() {
        return Err(VerificationError::InvalidWitness("non-finite Q4 scale"));
    }
    let weight = f32::from(code as i8 - 7) * scale;
    let result = libm_fma(weight, activation, accumulator);
    if !result.is_finite() {
        return Err(VerificationError::InvalidWitness("non-finite result"));
    }
    let residual_root = witness
        .memory
        .root_after_scalar_write(witness.instruction.destination, result.to_bits())
        .map_err(commitment_error)?;
    witness
        .pre_state
        .successor(residual_root)
        .map_err(commitment_error)
}

/// Structured append witness for one layer's fixed-capacity KV tree and the
/// small global tree of per-layer roots. The opened VM memory authenticates
/// the K/V vector roots produced by the preceding dispute-local computation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmKvAppendWitnessV1 {
    pub pre_state: qx_commit::VmStateV1,
    pub instruction: qx_commit::VmInstructionV1,
    pub instruction_proof: qx_commit::MerkleProof,
    pub memory: qx_commit::VmMemoryOpeningV1,
    pub max_context_tokens: u32,
    pub num_layers: u16,
    pub layer_siblings: Vec<Digest>,
    pub global_siblings: Vec<Digest>,
}

pub fn verify_vm_kv_append_transition_v1(
    registered_program_root: Digest,
    witness: &VmKvAppendWitnessV1,
) -> Result<qx_commit::VmStateV1, VerificationError> {
    let pc = witness.pre_state.program_counter;
    if witness.instruction.opcode != QxOpV1::KvAppend as u8
        || witness.instruction.destination != 0
        || witness.instruction.source_c >= u32::from(witness.num_layers)
        || witness.pre_state.token_index >= witness.max_context_tokens
        || witness.instruction_proof.leaf_index != pc
        || !witness
            .instruction_proof
            .verify(
                qx_commit::HashAlgorithm::Blake3V1,
                qx_commit::D_PROGRAM_TREE,
                witness.instruction.hash(pc).map_err(commitment_error)?,
                registered_program_root,
            )
            .map_err(commitment_error)?
    {
        return Err(VerificationError::AuthenticationFailed(
            "KV append instruction",
        ));
    }
    if !witness
        .memory
        .verify(witness.pre_state.residual_root)
        .map_err(commitment_error)?
    {
        return Err(VerificationError::AuthenticationFailed(
            "VM residual memory",
        ));
    }
    let key_root: Digest = witness
        .memory
        .bytes_at(witness.instruction.source_a, 32)
        .map_err(commitment_error)?
        .try_into()
        .map_err(|_| VerificationError::InvalidWitness("key root"))?;
    let value_root: Digest = witness
        .memory
        .bytes_at(witness.instruction.source_b, 32)
        .map_err(commitment_error)?
        .try_into()
        .map_err(|_| VerificationError::InvalidWitness("value root"))?;

    let context_capacity = usize::try_from(witness.max_context_tokens)
        .map_err(|_| VerificationError::Overflow)?
        .checked_add(1)
        .ok_or(VerificationError::Overflow)?
        .checked_next_power_of_two()
        .ok_or(VerificationError::Overflow)?;
    let context_depth = context_capacity.trailing_zeros() as usize;
    let layer_capacity = usize::from(witness.num_layers)
        .checked_next_power_of_two()
        .ok_or(VerificationError::Overflow)?;
    let global_depth = layer_capacity.trailing_zeros() as usize;
    if witness.layer_siblings.len() != context_depth
        || witness.global_siblings.len() != global_depth
    {
        return Err(VerificationError::InvalidWitness("KV proof depth"));
    }
    let empty = qx_commit::hash_domain(
        qx_commit::HashAlgorithm::Blake3V1,
        qx_commit::D_MERKLE_EMPTY,
        &[qx_commit::D_KV],
    )
    .map_err(commitment_error)?;
    let token_index = witness.pre_state.token_index;
    let old_layer_root =
        domain_merkle_root(empty, token_index, &witness.layer_siblings, qx_commit::D_KV)?;
    let layer_index = witness.instruction.source_c;
    let old_global_root = domain_merkle_root(
        old_layer_root,
        layer_index,
        &witness.global_siblings,
        qx_commit::D_KV,
    )?;
    if old_global_root != witness.pre_state.global_kv_root {
        return Err(VerificationError::AuthenticationFailed("old KV frontier"));
    }
    let kv_leaf = qx_commit::kv_leaf_hash_from_roots_v1(
        u16::try_from(layer_index).map_err(|_| VerificationError::Overflow)?,
        token_index,
        key_root,
        value_root,
    )
    .map_err(commitment_error)?;
    let new_layer_root = domain_merkle_root(
        kv_leaf,
        token_index,
        &witness.layer_siblings,
        qx_commit::D_KV,
    )?;
    let new_global_root = domain_merkle_root(
        new_layer_root,
        layer_index,
        &witness.global_siblings,
        qx_commit::D_KV,
    )?;
    witness
        .pre_state
        .successor_with_roots(witness.pre_state.residual_root, new_global_root)
        .map_err(commitment_error)
}

fn domain_merkle_root(
    mut node: Digest,
    mut index: u32,
    siblings: &[Digest],
    tree_domain: &'static [u8],
) -> Result<Digest, VerificationError> {
    for sibling in siblings {
        node = if index & 1 == 0 {
            qx_commit::hash_domain(
                qx_commit::HashAlgorithm::Blake3V1,
                qx_commit::D_MERKLE,
                &[tree_domain, &node, sibling],
            )
        } else {
            qx_commit::hash_domain(
                qx_commit::HashAlgorithm::Blake3V1,
                qx_commit::D_MERKLE,
                &[tree_domain, sibling, &node],
            )
        }
        .map_err(commitment_error)?;
        index >>= 1;
    }
    Ok(node)
}

/// Pairwise greedy-output terminal. A challenger only needs one token that
/// outranks the miner's claimed token; equal logits select the lower token ID.
/// Each memory record is `[logit_bits, token_id]` and the selected raw token ID
/// is written at `destination`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VmArgMaxWitnessV1 {
    pub pre_state: qx_commit::VmStateV1,
    pub instruction: qx_commit::VmInstructionV1,
    pub instruction_proof: qx_commit::MerkleProof,
    pub memory: qx_commit::VmMemoryOpeningV1,
    pub vocab_size: u32,
}

pub fn verify_vm_argmax_transition_v1(
    registered_program_root: Digest,
    witness: &VmArgMaxWitnessV1,
) -> Result<(qx_commit::VmStateV1, u32), VerificationError> {
    let pc = witness.pre_state.program_counter;
    if witness.instruction.opcode != QxOpV1::ArgMax as u8
        || witness.instruction.source_c != 0
        || witness.instruction_proof.leaf_index != pc
        || !witness
            .instruction_proof
            .verify(
                qx_commit::HashAlgorithm::Blake3V1,
                qx_commit::D_PROGRAM_TREE,
                witness.instruction.hash(pc).map_err(commitment_error)?,
                registered_program_root,
            )
            .map_err(commitment_error)?
    {
        return Err(VerificationError::AuthenticationFailed(
            "argmax instruction",
        ));
    }
    if !witness
        .memory
        .verify(witness.pre_state.residual_root)
        .map_err(commitment_error)?
    {
        return Err(VerificationError::AuthenticationFailed(
            "VM residual memory",
        ));
    }
    let left_logit = witness
        .memory
        .scalar_bits(witness.instruction.source_a)
        .map_err(commitment_error)?;
    let left_token = witness
        .memory
        .scalar_bits(
            witness
                .instruction
                .source_a
                .checked_add(4)
                .ok_or(VerificationError::Overflow)?,
        )
        .map_err(commitment_error)?;
    let right_logit = witness
        .memory
        .scalar_bits(witness.instruction.source_b)
        .map_err(commitment_error)?;
    let right_token = witness
        .memory
        .scalar_bits(
            witness
                .instruction
                .source_b
                .checked_add(4)
                .ok_or(VerificationError::Overflow)?,
        )
        .map_err(commitment_error)?;
    if left_token >= witness.vocab_size
        || right_token >= witness.vocab_size
        || left_token == right_token
    {
        return Err(VerificationError::InvalidWitness("argmax token IDs"));
    }
    let left_key = finite_order_key(left_logit)?;
    let right_key = finite_order_key(right_logit)?;
    let selected = if right_key > left_key || (right_key == left_key && right_token < left_token) {
        right_token
    } else {
        left_token
    };
    let residual_root = witness
        .memory
        .root_after_scalar_write(witness.instruction.destination, selected)
        .map_err(commitment_error)?;
    let post = witness
        .pre_state
        .successor(residual_root)
        .map_err(commitment_error)?;
    Ok((post, selected))
}

fn commitment_error(error: qx_commit::CommitError) -> VerificationError {
    VerificationError::Commitment(error.to_string())
}

#[inline(always)]
fn libm_fma(a: f32, b: f32, c: f32) -> f32 {
    // qx-consensus-math already pins libm's software-exact fmaf dependency;
    // expose it there rather than relying on target-specific contraction.
    qx_consensus_math::fma_f32(a, b, c)
}

fn finite_f32(bits: u32) -> Result<f32, VerificationError> {
    let value = f32::from_bits(bits);
    value
        .is_finite()
        .then_some(value)
        .ok_or(VerificationError::InvalidWitness("non-finite operand"))
}

fn ordered_finite_ge(left: u32, right: u32) -> bool {
    // Values have already been checked finite. Byte-wise key comparison also
    // avoids the SBF f32 comparison miscompile covered by program tests.
    let left = if left & 0x7fff_ffff == 0 { 0 } else { left };
    let right = if right & 0x7fff_ffff == 0 { 0 } else { right };
    let key = |bits: u32| {
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    };
    key(left).to_be_bytes() >= key(right).to_be_bytes()
}

/// The terminal unit for the current Metal Q4 reduction: one lane-local FMA.
/// A dispute first narrows to a lane and then to one `column = lane + 32*step`.
/// The later 32-lane reduction is a separate `ReduceSum` dispute.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedFmaWitnessV1 {
    pub tensor_id: u32,
    pub row: u32,
    pub column: u32,
    pub activation_root: Digest,
    pub activation: qx_commit::StateChunkProofV1,
    pub weight: WeightProofV1,
    pub accumulator_start_bits: u32,
    pub accumulator_claimed_end_bits: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerminalVerificationCost {
    pub sha256_hashes: u32,
    pub blake3_hashes: u32,
    pub compact_witness_bytes: u32,
    pub f32_multiply_adds: u32,
}

impl QuantizedFmaWitnessV1 {
    pub fn cost(&self) -> Result<TerminalVerificationCost, VerificationError> {
        let weight_bytes = self
            .weight
            .compact_encoded_len()
            .map_err(|error| VerificationError::Commitment(error.to_string()))?;
        let activation_bytes = self
            .activation
            .encoded_len()
            .map_err(|error| VerificationError::Commitment(error.to_string()))?;
        let compact_witness_bytes = 1_usize
            .checked_add(12)
            .and_then(|value| value.checked_add(8))
            .and_then(|value| value.checked_add(weight_bytes))
            .and_then(|value| value.checked_add(activation_bytes))
            .ok_or(VerificationError::Overflow)?;
        Ok(TerminalVerificationCost {
            sha256_hashes: u32::try_from(self.weight.proof.siblings.len() + 1)
                .map_err(|_| VerificationError::Overflow)?,
            blake3_hashes: u32::try_from(self.activation.siblings.len() + 1)
                .map_err(|_| VerificationError::Overflow)?,
            compact_witness_bytes: u32::try_from(compact_witness_bytes)
                .map_err(|_| VerificationError::Overflow)?,
            f32_multiply_adds: 1,
        })
    }
}

pub fn verify_quantized_fma_v1(
    registered_weights_root: Digest,
    witness: &QuantizedFmaWitnessV1,
) -> Result<(), VerificationError> {
    if witness.weight.commitment.root != registered_weights_root {
        return Err(VerificationError::AuthenticationFailed("weights root"));
    }
    if !witness
        .weight
        .verify()
        .map_err(|error| VerificationError::Commitment(error.to_string()))?
    {
        return Err(VerificationError::AuthenticationFailed("weight opening"));
    }
    let leaf = &witness.weight.leaf;
    if leaf.tensor_id != witness.tensor_id
        || leaf.row != witness.row
        || leaf.group != witness.column / Q4_GROUP_SIZE
        || leaf.quantization_type != Q4_QUANTIZATION_TYPE
        || leaf.packed_values.len() != 16
    {
        return Err(VerificationError::InvalidWitness("weight coordinates"));
    }
    if witness.activation.kind != StateTreeKindV1::Activation {
        return Err(VerificationError::InvalidWitness("activation tree kind"));
    }
    if !witness
        .activation
        .verify(witness.activation_root)
        .map_err(|error| VerificationError::Commitment(error.to_string()))?
    {
        return Err(VerificationError::AuthenticationFailed(
            "activation opening",
        ));
    }

    let activation_byte = usize::try_from(witness.column)
        .map_err(|_| VerificationError::Overflow)?
        .checked_mul(4)
        .ok_or(VerificationError::Overflow)?;
    let chunk_start = usize::try_from(witness.activation.chunk_index)
        .map_err(|_| VerificationError::Overflow)?
        .checked_mul(qx_commit::ACTIVATION_CHUNK_BYTES)
        .ok_or(VerificationError::Overflow)?;
    let local = activation_byte
        .checked_sub(chunk_start)
        .ok_or(VerificationError::InvalidWitness("activation coordinate"))?;
    let activation_bits = witness
        .activation
        .chunk
        .get(local..local + 4)
        .ok_or(VerificationError::InvalidWitness("activation coordinate"))?;
    let activation = f32::from_bits(u32::from_le_bytes(
        activation_bits
            .try_into()
            .map_err(|_| VerificationError::InvalidWitness("activation bits"))?,
    ));

    let local_weight =
        usize::try_from(witness.column % Q4_GROUP_SIZE).map_err(|_| VerificationError::Overflow)?;
    let packed = leaf.packed_values[local_weight / 2];
    let code = if local_weight & 1 == 0 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let scale = f16::from_bits(leaf.scale_bits).to_f32();
    let accumulator = f32::from_bits(witness.accumulator_start_bits);
    if !activation.is_finite() || !scale.is_finite() || !accumulator.is_finite() {
        return Err(VerificationError::InvalidWitness("non-finite arithmetic"));
    }
    let weight =
        f32::from(i8::try_from(code).map_err(|_| VerificationError::Overflow)? - 7) * scale;
    let expected = weight.mul_add(activation, accumulator);
    if !expected.is_finite() {
        return Err(VerificationError::InvalidWitness("non-finite result"));
    }
    if expected.to_bits() != witness.accumulator_claimed_end_bits {
        return Err(VerificationError::ArithmeticClaimFailed);
    }
    Ok(())
}

/// Returns true exactly when `challenger_expert` outranks a claimed selected
/// expert. Equal finite scores use the lower expert ID, matching QX Top-K.
pub fn proves_router_selection_wrong_v1(
    selected_expert: u16,
    selected_score_bits: u32,
    challenger_expert: u16,
    challenger_score_bits: u32,
) -> Result<bool, VerificationError> {
    if selected_expert == challenger_expert {
        return Err(VerificationError::InvalidWitness("identical expert IDs"));
    }
    let selected = finite_order_key(selected_score_bits)?;
    let challenger = finite_order_key(challenger_score_bits)?;
    Ok(challenger > selected || (challenger == selected && challenger_expert < selected_expert))
}

/// Reproduce the consensus Metal 32-lane shuffle-down reduction exactly.
/// Every stage reads the prior stage's lanes, matching SIMD lockstep.
pub fn ordered_simd_sum_32_v1(lane_bits: [u32; 32]) -> Result<u32, VerificationError> {
    let mut lanes = lane_bits.map(f32::from_bits);
    if lanes.iter().any(|value| !value.is_finite()) {
        return Err(VerificationError::InvalidWitness(
            "non-finite reduction lane",
        ));
    }
    for offset in [16_usize, 8, 4, 2, 1] {
        let prior = lanes;
        for lane in 0..offset {
            lanes[lane] = prior[lane] + prior[lane + offset];
            if !lanes[lane].is_finite() {
                return Err(VerificationError::InvalidWitness(
                    "non-finite reduction result",
                ));
            }
        }
    }
    Ok(lanes[0].to_bits())
}

pub fn verify_ordered_simd_sum_32_v1(
    lane_bits: [u32; 32],
    claimed_sum_bits: u32,
) -> Result<(), VerificationError> {
    if ordered_simd_sum_32_v1(lane_bits)? != claimed_sum_bits {
        return Err(VerificationError::ArithmeticClaimFailed);
    }
    Ok(())
}

fn finite_order_key(bits: u32) -> Result<u32, VerificationError> {
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return Err(VerificationError::InvalidWitness("non-finite router score"));
    }
    // Treat -0 and +0 as the same numerical score before applying total order.
    let bits = if bits & 0x7fff_ffff == 0 { 0 } else { bits };
    Ok(if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DisagreementLocation {
    pub first_differing_index: usize,
    pub matching_predecessor: Option<usize>,
    pub bisection_rounds: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisputePartyV1 {
    Miner,
    Challenger,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BisectionPhaseV1 {
    AwaitMinerBounds,
    AwaitChallengerMidpoint,
    AwaitMinerMidpoint,
    Terminal,
    Resolved,
}

/// Consensus state transitions for an adversarial trace bisection. Merkle
/// proof authentication is performed by the caller before passing a claim to
/// this state machine. No transition requires both parties to sign.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BisectionGameV1 {
    pub phase: BisectionPhaseV1,
    pub low: u32,
    pub high: u32,
    pub trace_length: u32,
    pub common_claim: Digest,
    pub miner_end_claim: Digest,
    pub challenger_end_claim: Digest,
    pub pending_midpoint: u32,
    pub pending_challenger_claim: Digest,
    pub response_deadline_slot: u64,
    pub response_window_slots: u64,
    pub winner: Option<DisputePartyV1>,
}

impl BisectionGameV1 {
    pub fn open(
        trace_length: u32,
        challenger_start_claim: Digest,
        challenger_end_claim: Digest,
        current_slot: u64,
        response_window_slots: u64,
    ) -> Result<Self, VerificationError> {
        if trace_length < 2 || response_window_slots == 0 {
            return Err(VerificationError::InvalidWitness(
                "a dispute needs at least two states and a response window",
            ));
        }
        Ok(Self {
            phase: BisectionPhaseV1::AwaitMinerBounds,
            low: 0,
            high: trace_length - 1,
            trace_length,
            common_claim: challenger_start_claim,
            miner_end_claim: [0; 32],
            challenger_end_claim,
            pending_midpoint: 0,
            pending_challenger_claim: [0; 32],
            response_deadline_slot: next_deadline(current_slot, response_window_slots)?,
            response_window_slots,
            winner: None,
        })
    }

    pub fn submit_miner_bounds(
        &mut self,
        miner_start_claim: Digest,
        miner_end_claim: Digest,
        current_slot: u64,
    ) -> Result<(), VerificationError> {
        self.require_live(BisectionPhaseV1::AwaitMinerBounds, current_slot)?;
        if miner_start_claim != self.common_claim || miner_end_claim == self.challenger_end_claim {
            return Err(VerificationError::InvalidWitness(
                "trace bounds must have a common start and differing end",
            ));
        }
        self.miner_end_claim = miner_end_claim;
        if self.high == 1 {
            self.phase = BisectionPhaseV1::Terminal;
        } else {
            self.phase = BisectionPhaseV1::AwaitChallengerMidpoint;
            self.refresh_deadline(current_slot)?;
        }
        Ok(())
    }

    pub fn submit_challenger_midpoint(
        &mut self,
        index: u32,
        claim: Digest,
        current_slot: u64,
    ) -> Result<(), VerificationError> {
        self.require_live(BisectionPhaseV1::AwaitChallengerMidpoint, current_slot)?;
        let expected = self.low + (self.high - self.low) / 2;
        if index != expected || index <= self.low || index >= self.high {
            return Err(VerificationError::InvalidWitness("invalid midpoint"));
        }
        self.pending_midpoint = index;
        self.pending_challenger_claim = claim;
        self.phase = BisectionPhaseV1::AwaitMinerMidpoint;
        self.refresh_deadline(current_slot)?;
        Ok(())
    }

    pub fn submit_miner_midpoint(
        &mut self,
        index: u32,
        claim: Digest,
        current_slot: u64,
    ) -> Result<(), VerificationError> {
        self.require_live(BisectionPhaseV1::AwaitMinerMidpoint, current_slot)?;
        if index != self.pending_midpoint {
            return Err(VerificationError::InvalidWitness("unexpected midpoint"));
        }
        if claim == self.pending_challenger_claim {
            self.low = index;
            self.common_claim = claim;
        } else {
            self.high = index;
            self.miner_end_claim = claim;
            self.challenger_end_claim = self.pending_challenger_claim;
        }
        self.pending_midpoint = 0;
        self.pending_challenger_claim = [0; 32];
        if self.high - self.low == 1 {
            self.phase = BisectionPhaseV1::Terminal;
        } else {
            self.phase = BisectionPhaseV1::AwaitChallengerMidpoint;
            self.refresh_deadline(current_slot)?;
        }
        Ok(())
    }

    pub fn timeout(&mut self, current_slot: u64) -> Result<DisputePartyV1, VerificationError> {
        if current_slot <= self.response_deadline_slot || self.winner.is_some() {
            return Err(VerificationError::InvalidWitness(
                "timeout is not available",
            ));
        }
        let winner = match self.phase {
            BisectionPhaseV1::AwaitMinerBounds | BisectionPhaseV1::AwaitMinerMidpoint => {
                DisputePartyV1::Challenger
            }
            BisectionPhaseV1::AwaitChallengerMidpoint => DisputePartyV1::Miner,
            BisectionPhaseV1::Terminal | BisectionPhaseV1::Resolved => {
                return Err(VerificationError::InvalidWitness(
                    "terminal resolution does not use a bisection timeout",
                ));
            }
        };
        self.phase = BisectionPhaseV1::Resolved;
        self.winner = Some(winner);
        Ok(winner)
    }

    /// Resolve only the miner's committed transition. A challenger wins only
    /// by supplying the canonical alternative; if neither claim is canonical,
    /// this challenge fails without immunizing the result against others.
    pub fn resolve_terminal_bits(
        &mut self,
        expected_bits: u32,
        miner_bits: u32,
        challenger_bits: u32,
    ) -> Result<DisputePartyV1, VerificationError> {
        if self.phase != BisectionPhaseV1::Terminal || self.winner.is_some() {
            return Err(VerificationError::InvalidWitness(
                "dispute has not reached a live terminal transition",
            ));
        }
        let winner = if expected_bits != miner_bits && expected_bits == challenger_bits {
            DisputePartyV1::Challenger
        } else {
            DisputePartyV1::Miner
        };
        self.phase = BisectionPhaseV1::Resolved;
        self.winner = Some(winner);
        Ok(winner)
    }

    fn require_live(
        &self,
        phase: BisectionPhaseV1,
        current_slot: u64,
    ) -> Result<(), VerificationError> {
        if self.phase != phase || self.winner.is_some() {
            return Err(VerificationError::InvalidWitness("wrong dispute phase"));
        }
        if current_slot > self.response_deadline_slot {
            return Err(VerificationError::InvalidWitness(
                "response deadline passed",
            ));
        }
        Ok(())
    }

    fn refresh_deadline(&mut self, current_slot: u64) -> Result<(), VerificationError> {
        self.response_deadline_slot = next_deadline(current_slot, self.response_window_slots)?;
        Ok(())
    }
}

fn next_deadline(current_slot: u64, response_window: u64) -> Result<u64, VerificationError> {
    current_slot
        .checked_add(response_window)
        .ok_or(VerificationError::Overflow)
}

/// Host oracle for the interactive token/layer bisection. On-chain, each
/// midpoint is authenticated against the parties' existing Merkle roots.
pub fn locate_first_disagreement<T: Eq>(
    defender: &[T],
    challenger: &[T],
) -> Result<Option<DisagreementLocation>, VerificationError> {
    if defender.len() != challenger.len() || defender.is_empty() {
        return Err(VerificationError::InvalidWitness(
            "trace lengths must be equal and non-zero",
        ));
    }
    let Some(first) = defender
        .iter()
        .zip(challenger)
        .position(|(left, right)| left != right)
    else {
        return Ok(None);
    };
    let rounds = usize::BITS - (defender.len() - 1).leading_zeros();
    Ok(Some(DisagreementLocation {
        first_differing_index: first,
        matching_predecessor: first.checked_sub(1),
        bisection_rounds: rounds,
    }))
}

#[cfg(test)]
mod tests {
    use qx_commit::{
        HashAlgorithm, MerkleTree, ProtocolVersion, StateTreeKindV1, VmInstructionV1, VmStateV1,
        WeightCommitmentV1, WeightLeafV1, WeightProofV1, activation_hash_v1, prove_state_chunk_v1,
        state_hash_v1,
    };

    use super::*;

    fn terminal_witness() -> QuantizedFmaWitnessV1 {
        let leaf = WeightLeafV1 {
            tensor_id: 4,
            row: 9,
            group: 1,
            quantization_type: Q4_QUANTIZATION_TYPE,
            packed_values: vec![0x77; 16],
            scale_bits: f16::from_f32(0.5).to_bits(),
        };
        let digest = leaf.hash(ProtocolVersion::V1).unwrap();
        let tree =
            MerkleTree::new(HashAlgorithm::Sha256V1, qx_commit::D_WEIGHT_TREE, &[digest]).unwrap();
        let mut activations = vec![0_u8; 64 * 4];
        activations[33 * 4..34 * 4].copy_from_slice(&2.0_f32.to_bits().to_le_bytes());
        let activation_root = activation_hash_v1(&activations).unwrap();
        QuantizedFmaWitnessV1 {
            tensor_id: 4,
            row: 9,
            column: 33,
            activation_root,
            activation: prove_state_chunk_v1(StateTreeKindV1::Activation, &activations, 1).unwrap(),
            weight: WeightProofV1 {
                commitment: WeightCommitmentV1 {
                    root: tree.root(),
                    leaf_count: 1,
                    tensor_count: 1,
                },
                leaf,
                proof: tree.proof(0).unwrap(),
            },
            accumulator_start_bits: 3.0_f32.to_bits(),
            accumulator_claimed_end_bits: 3.0_f32.to_bits(),
        }
    }

    #[test]
    fn authenticated_fma_accepts_honest_and_rejects_corruption() {
        let mut witness = terminal_witness();
        // Code 7 decodes to zero, so the accumulator stays exactly 3.0.
        verify_quantized_fma_v1(witness.weight.commitment.root, &witness).unwrap();
        witness.accumulator_claimed_end_bits = 4.0_f32.to_bits();
        assert_eq!(
            verify_quantized_fma_v1(witness.weight.commitment.root, &witness),
            Err(VerificationError::ArithmeticClaimFailed)
        );
    }

    #[test]
    fn chained_q4_authenticates_program_weight_and_memory() {
        let mut memory = vec![0_u8; 512];
        memory[4..8].copy_from_slice(&3.0_f32.to_le_bytes());
        memory[12..16].copy_from_slice(&(-1.0_f32).to_le_bytes());
        memory[136..140].copy_from_slice(&2.0_f32.to_le_bytes());
        let memory_root = state_hash_v1(&memory).unwrap();
        let memory_opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();

        let mut packed = vec![0x77; 16];
        packed[1] = 0x78;
        let leaf = WeightLeafV1 {
            tensor_id: 4,
            row: 9,
            group: 1,
            quantization_type: Q4_QUANTIZATION_TYPE,
            packed_values: packed,
            scale_bits: f16::from_f32(0.5).to_bits(),
        };
        let weight_tree = MerkleTree::new(
            HashAlgorithm::Sha256V1,
            qx_commit::D_WEIGHT_TREE,
            &[leaf.hash(ProtocolVersion::V1).unwrap()],
        )
        .unwrap();
        let weight = WeightProofV1 {
            commitment: WeightCommitmentV1 {
                root: weight_tree.root(),
                leaf_count: 1,
                tensor_count: 1,
            },
            leaf,
            proof: weight_tree.proof(0).unwrap(),
        };
        let instruction = VmInstructionV1 {
            opcode: QxOpV1::QuantizedFma as u8,
            source_a: 136,
            source_b: 4,
            source_c: qx_commit::q4_weight_selector_v1(0, 2).unwrap(),
            destination: 12,
        };
        let program = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            qx_commit::D_PROGRAM_TREE,
            &[instruction.hash(0).unwrap()],
        )
        .unwrap();
        let pre_state = VmStateV1 {
            job_hash: [1; 32],
            token_index: 2,
            segment_index: 3,
            program_counter: 0,
            residual_root: memory_root,
            global_kv_root: [2; 32],
        };
        let mut witness = VmQ4WitnessV1 {
            pre_state,
            instruction,
            instruction_proof: program.proof(0).unwrap(),
            memory: memory_opening,
            weight,
        };
        let post =
            verify_vm_q4_transition_v1(program.root(), weight_tree.root(), &witness).unwrap();
        memory[12..16].copy_from_slice(&4.0_f32.to_le_bytes());
        assert_eq!(post.residual_root, state_hash_v1(&memory).unwrap());

        witness.instruction.source_c = 1;
        assert!(verify_vm_q4_transition_v1(program.root(), weight_tree.root(), &witness).is_err());
    }

    #[test]
    fn structured_kv_append_preserves_the_old_frontier() {
        let key_root = [0x31; 32];
        let value_root = [0x52; 32];
        let mut memory = vec![0_u8; 256];
        memory[..32].copy_from_slice(&key_root);
        memory[32..64].copy_from_slice(&value_root);
        let memory_root = state_hash_v1(&memory).unwrap();
        let memory_opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();

        let empty = qx_commit::hash_domain(
            HashAlgorithm::Blake3V1,
            qx_commit::D_MERKLE_EMPTY,
            &[qx_commit::D_KV],
        )
        .unwrap();
        let layer_tree =
            MerkleTree::new(HashAlgorithm::Blake3V1, qx_commit::D_KV, &[empty; 4]).unwrap();
        let global_tree = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            qx_commit::D_KV,
            &[layer_tree.root(); 3],
        )
        .unwrap();
        let instruction = VmInstructionV1 {
            opcode: QxOpV1::KvAppend as u8,
            source_a: 0,
            source_b: 32,
            source_c: 1,
            destination: 0,
        };
        let program = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            qx_commit::D_PROGRAM_TREE,
            &[instruction.hash(0).unwrap()],
        )
        .unwrap();
        let pre_state = VmStateV1 {
            job_hash: [1; 32],
            token_index: 1,
            segment_index: 2,
            program_counter: 0,
            residual_root: memory_root,
            global_kv_root: global_tree.root(),
        };
        let layer_proof = layer_tree.proof(1).unwrap();
        let global_proof = global_tree.proof(1).unwrap();
        let mut witness = VmKvAppendWitnessV1 {
            pre_state,
            instruction,
            instruction_proof: program.proof(0).unwrap(),
            memory: memory_opening,
            max_context_tokens: 3,
            num_layers: 3,
            layer_siblings: layer_proof.siblings,
            global_siblings: global_proof.siblings,
        };
        let post = verify_vm_kv_append_transition_v1(program.root(), &witness).unwrap();
        assert_ne!(post.global_kv_root, pre_state.global_kv_root);
        assert_eq!(post.residual_root, pre_state.residual_root);
        assert_eq!(post.program_counter, 1);

        witness.layer_siblings[0][0] ^= 1;
        assert!(verify_vm_kv_append_transition_v1(program.root(), &witness).is_err());
    }

    #[test]
    fn pairwise_argmax_uses_lowest_token_id_for_ties() {
        let mut memory = vec![0_u8; 256];
        memory[..4].copy_from_slice(&7.0_f32.to_le_bytes());
        memory[4..8].copy_from_slice(&42_u32.to_le_bytes());
        memory[8..12].copy_from_slice(&7.0_f32.to_le_bytes());
        memory[12..16].copy_from_slice(&9_u32.to_le_bytes());
        memory[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        let opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
        let instruction = VmInstructionV1 {
            opcode: QxOpV1::ArgMax as u8,
            source_a: 0,
            source_b: 8,
            source_c: 0,
            destination: 16,
        };
        let program = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            qx_commit::D_PROGRAM_TREE,
            &[instruction.hash(0).unwrap()],
        )
        .unwrap();
        let pre_state = VmStateV1 {
            job_hash: [1; 32],
            token_index: 2,
            segment_index: 4,
            program_counter: 0,
            residual_root: state_hash_v1(&memory).unwrap(),
            global_kv_root: [2; 32],
        };
        let witness = VmArgMaxWitnessV1 {
            pre_state,
            instruction,
            instruction_proof: program.proof(0).unwrap(),
            memory: opening,
            vocab_size: 100,
        };
        let (post, selected) = verify_vm_argmax_transition_v1(program.root(), &witness).unwrap();
        assert_eq!(selected, 9);
        memory[16..20].copy_from_slice(&9_u32.to_le_bytes());
        assert_eq!(post.residual_root, state_hash_v1(&memory).unwrap());
    }

    #[test]
    fn scalar_ir_covers_inference_arithmetic() {
        let bits = |value: f32| value.to_bits();
        assert_eq!(
            evaluate_scalar_op_v1(QxOpV1::Add, [bits(2.0), bits(3.0), bits(0.0)]).unwrap(),
            bits(5.0)
        );
        assert_eq!(
            evaluate_scalar_op_v1(QxOpV1::Mul, [bits(2.0), bits(3.0), bits(0.0)]).unwrap(),
            bits(6.0)
        );
        assert_eq!(
            evaluate_scalar_op_v1(QxOpV1::QuantizedFma, [bits(2.0), bits(3.0), bits(4.0)]).unwrap(),
            bits(10.0)
        );
        assert_eq!(
            evaluate_scalar_op_v1(QxOpV1::Div, [bits(3.0), bits(2.0), bits(0.0)]).unwrap(),
            bits(1.5)
        );
        assert_eq!(
            evaluate_scalar_op_v1(QxOpV1::ReduceMax, [bits(-1.0), bits(-2.0), bits(0.0)]).unwrap(),
            bits(-1.0)
        );
        for op in [QxOpV1::Rsqrt, QxOpV1::ExpApprox] {
            let result = evaluate_scalar_op_v1(op, [bits(2.0), bits(0.0), bits(0.0)]).unwrap();
            assert!(f32::from_bits(result).is_finite());
        }
        assert!(evaluate_scalar_op_v1(QxOpV1::Div, [bits(1.0), bits(0.0), bits(0.0)]).is_err());
        assert_eq!(
            evaluate_scalar_op_v1(QxOpV1::Select, [bits(1.0), bits(7.0), bits(9.0)]).unwrap(),
            bits(7.0)
        );
    }

    #[test]
    fn chained_vm_transition_binds_program_operands_and_memory_update() {
        let mut memory = vec![0_u8; 512];
        for (index, value) in [2.0_f32, 3.0, 0.0, -1.0].iter().enumerate() {
            memory[index * 4..index * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        }
        let memory_root = state_hash_v1(&memory).unwrap();
        let opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
        let instruction = VmInstructionV1 {
            opcode: QxOpV1::Add as u8,
            source_a: 0,
            source_b: 4,
            source_c: 8,
            destination: 12,
        };
        let instruction_leaf = instruction.hash(0).unwrap();
        let program = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            qx_commit::D_PROGRAM_TREE,
            &[instruction_leaf],
        )
        .unwrap();
        let witness = VmScalarWitnessV1 {
            pre_state: VmStateV1 {
                job_hash: [7; 32],
                token_index: 3,
                segment_index: 11,
                program_counter: 0,
                residual_root: memory_root,
                global_kv_root: [8; 32],
            },
            instruction,
            instruction_proof: program.proof(0).unwrap(),
            memory: opening,
        };
        let post = verify_vm_scalar_transition_v1(program.root(), &witness).unwrap();
        memory[12..16].copy_from_slice(&5.0_f32.to_bits().to_le_bytes());
        let expected_memory = state_hash_v1(&memory).unwrap();
        assert_eq!(post.program_counter, 1);
        assert_eq!(post.residual_root, expected_memory);

        let mut forged = witness;
        forged.memory.chunk[0..4].copy_from_slice(&9.0_f32.to_bits().to_le_bytes());
        assert!(verify_vm_scalar_transition_v1(program.root(), &forged).is_err());
    }

    #[test]
    fn false_weight_and_activation_openings_lose() {
        let mut witness = terminal_witness();
        assert!(verify_quantized_fma_v1([8; 32], &witness).is_err());
        witness.activation.chunk[0] ^= 1;
        assert!(verify_quantized_fma_v1(witness.weight.commitment.root, &witness).is_err());
    }

    #[test]
    fn router_order_handles_scores_and_ties() {
        assert!(
            proves_router_selection_wrong_v1(31, 1.0_f32.to_bits(), 21, 2.0_f32.to_bits()).unwrap()
        );
        assert!(
            proves_router_selection_wrong_v1(31, 1.0_f32.to_bits(), 21, 1.0_f32.to_bits()).unwrap()
        );
        assert!(
            !proves_router_selection_wrong_v1(5, 1.0_f32.to_bits(), 21, 1.0_f32.to_bits()).unwrap()
        );
        assert!(
            !proves_router_selection_wrong_v1(31, 2.0_f32.to_bits(), 21, 1.0_f32.to_bits())
                .unwrap()
        );
    }

    #[test]
    fn ordered_reduction_matches_the_registered_tree() {
        let lanes = std::array::from_fn(|index| {
            let value = if index % 3 == 0 {
                1.0e7_f32
            } else {
                index as f32 / 7.0
            };
            value.to_bits()
        });
        let sum = ordered_simd_sum_32_v1(lanes).unwrap();
        verify_ordered_simd_sum_32_v1(lanes, sum).unwrap();
        assert_eq!(
            verify_ordered_simd_sum_32_v1(lanes, sum ^ 1),
            Err(VerificationError::ArithmeticClaimFailed)
        );
    }

    #[test]
    fn disagreement_finds_the_first_bad_transition() {
        let honest = [0, 1, 2, 3, 4, 5, 6, 7];
        let corrupt = [0, 1, 2, 9, 9, 9, 9, 9];
        assert_eq!(
            locate_first_disagreement(&honest, &corrupt).unwrap(),
            Some(DisagreementLocation {
                first_differing_index: 3,
                matching_predecessor: Some(2),
                bisection_rounds: 3,
            })
        );
        assert_eq!(locate_first_disagreement(&honest, &honest).unwrap(), None);
    }

    #[test]
    fn terminal_cost_reports_both_authenticated_paths() {
        let cost = terminal_witness().cost().unwrap();
        assert_eq!(cost.sha256_hashes, 1);
        assert_eq!(cost.blake3_hashes, 2);
        assert!(cost.compact_witness_bytes < 512);
    }

    #[test]
    fn adversarial_bisection_reaches_first_bad_transition_without_cosigning() {
        let correct = [[1; 32], [2; 32], [3; 32], [4; 32], [5; 32]];
        let miner = [[1; 32], [2; 32], [9; 32], [8; 32], [7; 32]];
        let mut game = BisectionGameV1::open(5, correct[0], correct[4], 10, 5).unwrap();
        game.submit_miner_bounds(miner[0], miner[4], 11).unwrap();
        game.submit_challenger_midpoint(2, correct[2], 12).unwrap();
        game.submit_miner_midpoint(2, miner[2], 13).unwrap();
        game.submit_challenger_midpoint(1, correct[1], 14).unwrap();
        game.submit_miner_midpoint(1, miner[1], 15).unwrap();
        assert_eq!(game.phase, BisectionPhaseV1::Terminal);
        assert_eq!((game.low, game.high), (1, 2));
        assert_eq!(game.common_claim, correct[1]);
        assert_eq!(game.miner_end_claim, miner[2]);
        assert_eq!(game.challenger_end_claim, correct[2]);
        assert_eq!(
            game.resolve_terminal_bits(3.0_f32.to_bits(), 9.0_f32.to_bits(), 3.0_f32.to_bits())
                .unwrap(),
            DisputePartyV1::Challenger
        );
    }

    #[test]
    fn timeout_penalizes_the_party_whose_turn_it_is() {
        let mut miner_turn = BisectionGameV1::open(4, [1; 32], [4; 32], 10, 5).unwrap();
        assert_eq!(miner_turn.timeout(16).unwrap(), DisputePartyV1::Challenger);

        let mut challenger_turn = BisectionGameV1::open(4, [1; 32], [4; 32], 10, 5).unwrap();
        challenger_turn
            .submit_miner_bounds([1; 32], [9; 32], 11)
            .unwrap();
        assert_eq!(challenger_turn.timeout(17).unwrap(), DisputePartyV1::Miner);
    }

    #[test]
    fn invalid_challenger_terminal_claim_does_not_slash_the_miner() {
        let mut game = BisectionGameV1::open(2, [1; 32], [2; 32], 10, 5).unwrap();
        game.submit_miner_bounds([1; 32], [3; 32], 11).unwrap();
        assert_eq!(
            game.resolve_terminal_bits(7, 8, 9).unwrap(),
            DisputePartyV1::Miner
        );
    }
}
