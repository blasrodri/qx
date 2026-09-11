use crate::{CommitError, Digest, HashAlgorithm, MerkleTree, StateChunkProofV1, hash_domain};

pub const Q4_SELECTOR_LOCAL_BITS: u32 = 5;

/// Pack an exact weight-tree leaf and one of its 32 Q4 values into the VM's
/// `source_c` field, leaving activation memory placement independent of model
/// tensor coordinates.
pub fn q4_weight_selector_v1(leaf_index: u32, local_index: u8) -> Result<u32, CommitError> {
    if local_index >= 32 {
        return Err(CommitError::InvalidIndex {
            index: usize::from(local_index),
            length: 32,
        });
    }
    leaf_index
        .checked_mul(1 << Q4_SELECTOR_LOCAL_BITS)
        .and_then(|value| value.checked_add(u32::from(local_index)))
        .ok_or(CommitError::ArithmeticOverflow)
}

pub fn decode_q4_weight_selector_v1(selector: u32) -> (u32, u8) {
    (selector >> Q4_SELECTOR_LOCAL_BITS, (selector & 31) as u8)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmStateV1 {
    pub job_hash: Digest,
    pub token_index: u32,
    pub segment_index: u32,
    pub program_counter: u32,
    pub residual_root: Digest,
    pub global_kv_root: Digest,
}

/// Final-head endpoint. Binding the selected token into the endpoint prevents
/// a valid logits trace from being paired with an unrelated token result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmOutputStateV1 {
    pub job_hash: Digest,
    pub token_index: u32,
    pub segment_index: u32,
    pub program_counter: u32,
    pub residual_root: Digest,
    pub global_kv_root: Digest,
    pub predicted_token: u32,
}

/// One checkpoint interval's independently committed verification program.
/// Segment zero is reserved; checkpoint transitions use their high checkpoint
/// index and the final head uses `checkpoint_count`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmSegmentProgramV1 {
    pub segment_index: u32,
    pub instruction_root: Digest,
    pub instruction_count: u32,
}

impl VmSegmentProgramV1 {
    pub const CANONICAL_SIZE: usize = 40;

    pub fn canonical_bytes(&self) -> [u8; Self::CANONICAL_SIZE] {
        let mut bytes = [0; Self::CANONICAL_SIZE];
        bytes[..4].copy_from_slice(&self.segment_index.to_le_bytes());
        bytes[4..36].copy_from_slice(&self.instruction_root);
        bytes[36..40].copy_from_slice(&self.instruction_count.to_le_bytes());
        bytes
    }

    pub fn hash(&self) -> Result<Digest, CommitError> {
        if self.instruction_root == [0; 32] || self.instruction_count == 0 {
            return Err(CommitError::InvalidShape(
                "segment program must contain instructions".to_owned(),
            ));
        }
        hash_domain(
            HashAlgorithm::Blake3V1,
            crate::D_VM_SEGMENT_PROGRAM,
            &[&self.canonical_bytes()],
        )
    }
}

impl VmOutputStateV1 {
    pub fn hash(&self) -> Result<Digest, CommitError> {
        hash_domain(
            HashAlgorithm::Blake3V1,
            crate::D_VM_OUTPUT,
            &[
                &self.job_hash,
                &self.token_index.to_le_bytes(),
                &self.segment_index.to_le_bytes(),
                &self.program_counter.to_le_bytes(),
                &self.residual_root,
                &self.global_kv_root,
                &self.predicted_token.to_le_bytes(),
            ],
        )
    }
}

impl VmStateV1 {
    pub fn hash(&self) -> Result<Digest, CommitError> {
        hash_domain(
            HashAlgorithm::Blake3V1,
            crate::D_VM_STATE,
            &[
                &self.job_hash,
                &self.token_index.to_le_bytes(),
                &self.segment_index.to_le_bytes(),
                &self.program_counter.to_le_bytes(),
                &self.residual_root,
                &self.global_kv_root,
            ],
        )
    }

    pub fn successor(&self, residual_root: Digest) -> Result<Self, CommitError> {
        self.successor_with_roots(residual_root, self.global_kv_root)
    }

    pub fn successor_with_roots(
        &self,
        residual_root: Digest,
        global_kv_root: Digest,
    ) -> Result<Self, CommitError> {
        Ok(Self {
            program_counter: self
                .program_counter
                .checked_add(1)
                .ok_or(CommitError::ArithmeticOverflow)?,
            residual_root,
            global_kv_root,
            ..*self
        })
    }
}

/// One instruction in the registered verification VM. Addresses, rather than
/// runtime values, are committed by the model program. Runtime values must be
/// opened from the pre-state memory root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VmInstructionV1 {
    pub opcode: u8,
    pub source_a: u32,
    pub source_b: u32,
    pub source_c: u32,
    pub destination: u32,
}

impl VmInstructionV1 {
    pub const CANONICAL_SIZE: usize = 17;

    pub fn canonical_bytes(&self) -> [u8; Self::CANONICAL_SIZE] {
        let mut bytes = [0; Self::CANONICAL_SIZE];
        bytes[0] = self.opcode;
        bytes[1..5].copy_from_slice(&self.source_a.to_le_bytes());
        bytes[5..9].copy_from_slice(&self.source_b.to_le_bytes());
        bytes[9..13].copy_from_slice(&self.source_c.to_le_bytes());
        bytes[13..17].copy_from_slice(&self.destination.to_le_bytes());
        bytes
    }

    pub fn hash(&self, instruction_index: u32) -> Result<Digest, CommitError> {
        hash_domain(
            HashAlgorithm::Blake3V1,
            crate::D_VM_INSTRUCTION,
            &[&instruction_index.to_le_bytes(), &self.canonical_bytes()],
        )
    }
}

pub fn vm_program_tree_v1(instructions: &[VmInstructionV1]) -> Result<MerkleTree, CommitError> {
    let leaves = instructions
        .iter()
        .enumerate()
        .map(|(index, instruction)| {
            instruction.hash(u32::try_from(index).map_err(|_| CommitError::ArithmeticOverflow)?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    MerkleTree::new(HashAlgorithm::Blake3V1, crate::D_PROGRAM_TREE, &leaves)
}

pub fn vm_segment_schedule_v1(segments: &[VmSegmentProgramV1]) -> Result<MerkleTree, CommitError> {
    let leaves = segments
        .iter()
        .enumerate()
        .map(|(index, segment)| {
            if segment.segment_index
                != u32::try_from(index).map_err(|_| CommitError::ArithmeticOverflow)?
            {
                return Err(CommitError::InvalidShape(
                    "segment programs must be dense and ordered".to_owned(),
                ));
            }
            segment.hash()
        })
        .collect::<Result<Vec<_>, _>>()?;
    MerkleTree::new(
        HashAlgorithm::Blake3V1,
        crate::D_SEGMENT_PROGRAM_TREE,
        &leaves,
    )
}

/// A scalar VM transition opens one canonical residual chunk. The verifier
/// requires all operands and the destination to be in this chunk, avoiding
/// four redundant Merkle paths and keeping the witness transportable.
pub type VmMemoryOpeningV1 = StateChunkProofV1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_opening_authenticates_reads_and_one_write() {
        let mut bytes = vec![0_u8; 512];
        for (index, value) in [1.0_f32, 2.0, 3.0, 0.0].iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        }
        let root = crate::state_hash_v1(&bytes).unwrap();
        let opening =
            crate::prove_state_chunk_v1(crate::StateTreeKindV1::Residual, &bytes, 0).unwrap();
        assert!(opening.verify(root).unwrap());
        assert_eq!(opening.scalar_bits(4).unwrap(), 2.0_f32.to_bits());
        let written = opening
            .root_after_scalar_write(12, 7.0_f32.to_bits())
            .unwrap();
        bytes[12..16].copy_from_slice(&7.0_f32.to_bits().to_le_bytes());
        assert_eq!(written, crate::state_hash_v1(&bytes).unwrap());
    }

    #[test]
    fn instruction_hash_binds_every_address() {
        let instruction = VmInstructionV1 {
            opcode: 2,
            source_a: 0,
            source_b: 1,
            source_c: 2,
            destination: 3,
        };
        let root = instruction.hash(9).unwrap();
        let mut changed = instruction;
        changed.destination = 2;
        assert_ne!(root, changed.hash(9).unwrap());
        assert_ne!(root, instruction.hash(10).unwrap());
    }

    #[test]
    fn q4_selector_keeps_leaf_and_local_coordinate() {
        let selector = q4_weight_selector_v1(41_755_743, 31).unwrap();
        assert_eq!(decode_q4_weight_selector_v1(selector), (41_755_743, 31));
        assert!(q4_weight_selector_v1(u32::MAX, 0).is_err());
        assert!(q4_weight_selector_v1(0, 32).is_err());
    }

    #[test]
    fn output_state_binds_selected_token() {
        let output = VmOutputStateV1 {
            job_hash: [1; 32],
            token_index: 2,
            segment_index: 3,
            program_counter: 4,
            residual_root: [5; 32],
            global_kv_root: [6; 32],
            predicted_token: 7,
        };
        let mut changed = output;
        changed.predicted_token = 8;
        assert_ne!(output.hash().unwrap(), changed.hash().unwrap());
    }

    #[test]
    fn segment_program_binds_index_root_and_length() {
        let segment = VmSegmentProgramV1 {
            segment_index: 3,
            instruction_root: [4; 32],
            instruction_count: 5,
        };
        let mut changed = segment;
        changed.segment_index = 2;
        assert_ne!(segment.hash().unwrap(), changed.hash().unwrap());
        changed = segment;
        changed.instruction_count = 6;
        assert_ne!(segment.hash().unwrap(), changed.hash().unwrap());
        changed.instruction_count = 0;
        assert!(changed.hash().is_err());

        let program = vm_program_tree_v1(&[VmInstructionV1 {
            opcode: 2,
            source_a: 0,
            source_b: 4,
            source_c: 8,
            destination: 12,
        }])
        .unwrap();
        let schedule = vm_segment_schedule_v1(&[
            VmSegmentProgramV1 {
                segment_index: 0,
                instruction_root: program.root(),
                instruction_count: 1,
            },
            VmSegmentProgramV1 {
                segment_index: 1,
                instruction_root: program.root(),
                instruction_count: 1,
            },
        ])
        .unwrap();
        assert!(schedule.proof(1).is_ok());
        let mut unordered = segment;
        unordered.segment_index = 1;
        assert!(vm_segment_schedule_v1(&[unordered]).is_err());
    }
}
