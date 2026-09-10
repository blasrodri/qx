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
        HashAlgorithm, MerkleTree, ProtocolVersion, StateTreeKindV1, WeightCommitmentV1,
        WeightLeafV1, WeightProofV1, activation_hash_v1, prove_state_chunk_v1,
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
}
