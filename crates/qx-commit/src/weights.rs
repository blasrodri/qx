use std::path::Path;

use qx_format::{MappedArtifact, TensorDirectoryEntry, TensorKind};
use qx_quant::QuantType;

use crate::{
    CommitError, D_WEIGHT_LEAF, D_WEIGHT_TREE, Digest, FixedMerkleAccumulator, HashAlgorithm,
    MerkleProof, ProtocolVersion, hash_domain,
};

const RAW_FP32: u8 = 0;
const Q4_GROUPED: u8 = 4;
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightLeafV1 {
    pub tensor_id: u32,
    pub row: u32,
    pub group: u32,
    pub quantization_type: u8,
    pub packed_values: Vec<u8>,
    pub scale_bits: u16,
}

impl WeightLeafV1 {
    pub fn hash(&self, protocol: ProtocolVersion) -> Result<Digest, CommitError> {
        hash_domain(
            protocol.hash_algorithm(),
            D_WEIGHT_LEAF,
            &[
                &self.tensor_id.to_le_bytes(),
                &self.row.to_le_bytes(),
                &self.group.to_le_bytes(),
                &[self.quantization_type],
                &self.packed_values,
                &self.scale_bits.to_le_bytes(),
            ],
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightCommitmentV1 {
    pub root: Digest,
    pub leaf_count: u32,
    pub tensor_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightProofV1 {
    pub commitment: WeightCommitmentV1,
    pub leaf: WeightLeafV1,
    pub proof: MerkleProof,
}

impl WeightProofV1 {
    pub fn verify(&self) -> Result<bool, CommitError> {
        let leaf = self.leaf.hash(ProtocolVersion::V1)?;
        self.proof.verify(
            HashAlgorithm::Sha256V1,
            D_WEIGHT_TREE,
            leaf,
            self.commitment.root,
        )
    }
}

#[derive(Clone, Copy)]
struct OrderedTensor {
    artifact_index: usize,
    tensor_id: u32,
}

fn semantic_rank(entry: &TensorDirectoryEntry) -> (u8, u16, u8, u8) {
    let global = entry.layer == u16::MAX;
    let stage = match entry.kind {
        TensorKind::Embedding => 0,
        TensorKind::InputNorm => 1,
        TensorKind::QProj => 2,
        TensorKind::KProj => 3,
        TensorKind::VProj => 4,
        TensorKind::OProj => 5,
        TensorKind::PostAttentionNorm => 6,
        TensorKind::Router => 7,
        TensorKind::GateUp => 8,
        TensorKind::Down => 9,
        TensorKind::QueryNorm => 10,
        TensorKind::KeyNorm => 11,
        TensorKind::FinalNorm => 12,
        TensorKind::LmHead => 13,
    };
    if global && entry.kind == TensorKind::Embedding {
        (0, 0, stage, entry.expert)
    } else if global {
        (2, 0, stage, entry.expert)
    } else {
        (1, entry.layer, stage, entry.expert)
    }
}

fn ordered_tensors(artifact: &MappedArtifact) -> Result<Vec<OrderedTensor>, CommitError> {
    let mut indices = (0..artifact.info().entries.len()).collect::<Vec<_>>();
    indices.sort_by_key(|index| semantic_rank(&artifact.info().entries[*index]));
    for pair in indices.windows(2) {
        let left = &artifact.info().entries[pair[0]];
        let right = &artifact.info().entries[pair[1]];
        if semantic_rank(left) == semantic_rank(right) {
            return Err(CommitError::InvalidArtifact(format!(
                "duplicate semantic tensor at layer {}, expert {}, kind {:?}",
                left.layer, left.expert, left.kind
            )));
        }
    }
    indices
        .into_iter()
        .enumerate()
        .map(|(tensor_id, artifact_index)| {
            Ok(OrderedTensor {
                artifact_index,
                tensor_id: u32::try_from(tensor_id).map_err(|_| CommitError::ArithmeticOverflow)?,
            })
        })
        .collect()
}

fn count_leaves(
    artifact: &MappedArtifact,
    ordered: &[OrderedTensor],
) -> Result<usize, CommitError> {
    ordered.iter().try_fold(0_usize, |total, ordered| {
        let entry = &artifact.info().entries[ordered.artifact_index];
        let groups = if entry.group_size == 0 {
            (entry.cols as usize).div_ceil(32)
        } else {
            if entry.quant_type != QuantType::Q4 || entry.group_size != 32 {
                return Err(CommitError::InvalidArtifact(format!(
                    "verification V1 requires Q4/group-32, found {}/{:?} for tensor {}",
                    entry.group_size, entry.quant_type, ordered.tensor_id
                )));
            }
            (entry.cols as usize).div_ceil(32)
        };
        let leaves = (entry.rows as usize)
            .checked_mul(groups)
            .ok_or(CommitError::ArithmeticOverflow)?;
        total
            .checked_add(leaves)
            .ok_or(CommitError::ArithmeticOverflow)
    })
}

fn visit_leaves(
    artifact: &MappedArtifact,
    ordered: &[OrderedTensor],
    mut visitor: impl FnMut(WeightLeafV1) -> Result<(), CommitError>,
) -> Result<(), CommitError> {
    for ordered in ordered {
        let entry = &artifact.info().entries[ordered.artifact_index];
        if entry.group_size == 0 {
            let (_, data) = artifact
                .tensor_bytes(ordered.artifact_index)
                .map_err(|error| CommitError::InvalidArtifact(error.to_string()))?;
            let expected = (entry.rows as usize)
                .checked_mul(entry.cols as usize)
                .and_then(|value| value.checked_mul(4))
                .ok_or(CommitError::ArithmeticOverflow)?;
            if data.len() != expected {
                return Err(CommitError::InvalidArtifact(format!(
                    "raw tensor {} has {} bytes, expected {expected}",
                    ordered.tensor_id,
                    data.len()
                )));
            }
            let row_bytes = (entry.cols as usize)
                .checked_mul(4)
                .ok_or(CommitError::ArithmeticOverflow)?;
            for row in 0..entry.rows as usize {
                for (group, packed_values) in data[row * row_bytes..(row + 1) * row_bytes]
                    .chunks(32 * 4)
                    .enumerate()
                {
                    visitor(WeightLeafV1 {
                        tensor_id: ordered.tensor_id,
                        row: u32::try_from(row).map_err(|_| CommitError::ArithmeticOverflow)?,
                        group: u32::try_from(group).map_err(|_| CommitError::ArithmeticOverflow)?,
                        quantization_type: RAW_FP32,
                        packed_values: packed_values.to_vec(),
                        scale_bits: 0,
                    })?;
                }
            }
            continue;
        }

        let tensor = artifact
            .quantized_tensor(ordered.artifact_index)
            .map_err(|error| CommitError::InvalidArtifact(error.to_string()))?;
        let groups = tensor.cols.div_ceil(tensor.group_size);
        for row in 0..tensor.rows {
            for group in 0..groups {
                let (packed_values, scale_bits) = tensor
                    .logical_group(row, group)
                    .map_err(CommitError::InvalidArtifact)?;
                visitor(WeightLeafV1 {
                    tensor_id: ordered.tensor_id,
                    row: u32::try_from(row).map_err(|_| CommitError::ArithmeticOverflow)?,
                    group: u32::try_from(group).map_err(|_| CommitError::ArithmeticOverflow)?,
                    quantization_type: Q4_GROUPED,
                    packed_values: packed_values.to_vec(),
                    scale_bits,
                })?;
            }
        }
    }
    Ok(())
}

pub fn commit_q4_artifact(path: impl AsRef<Path>) -> Result<WeightCommitmentV1, CommitError> {
    let artifact = MappedArtifact::open(path)
        .map_err(|error| CommitError::InvalidArtifact(error.to_string()))?;
    let ordered = ordered_tensors(&artifact)?;
    let leaf_count = count_leaves(&artifact, &ordered)?;
    if leaf_count == 0 {
        return Err(CommitError::EmptyTree);
    }
    let capacity = leaf_count
        .checked_next_power_of_two()
        .ok_or(CommitError::ArithmeticOverflow)?;
    let depth = capacity.trailing_zeros() as usize;
    let mut tree =
        FixedMerkleAccumulator::new(ProtocolVersion::V1.hash_algorithm(), D_WEIGHT_TREE, depth)?;
    visit_leaves(&artifact, &ordered, |leaf| {
        tree.append(leaf.hash(ProtocolVersion::V1)?)
    })?;
    Ok(WeightCommitmentV1 {
        root: tree.root()?,
        leaf_count: u32::try_from(leaf_count).map_err(|_| CommitError::ArithmeticOverflow)?,
        tensor_count: u32::try_from(ordered.len()).map_err(|_| CommitError::ArithmeticOverflow)?,
    })
}

pub fn prove_q4_artifact_leaf(
    path: impl AsRef<Path>,
    target_index: u32,
) -> Result<WeightProofV1, CommitError> {
    let artifact = MappedArtifact::open(path)
        .map_err(|error| CommitError::InvalidArtifact(error.to_string()))?;
    let ordered = ordered_tensors(&artifact)?;
    let leaf_count = count_leaves(&artifact, &ordered)?;
    if target_index as usize >= leaf_count {
        return Err(CommitError::InvalidIndex {
            index: target_index as usize,
            length: leaf_count,
        });
    }
    let capacity = leaf_count
        .checked_next_power_of_two()
        .ok_or(CommitError::ArithmeticOverflow)?;
    let depth = capacity.trailing_zeros() as usize;
    let mut frontier = vec![None::<(Digest, usize, usize)>; depth + 1];
    let mut siblings = vec![None; depth];
    let mut next_index = 0_usize;

    let mut captured_leaf = None;
    visit_leaves(&artifact, &ordered, |leaf| {
        let index = next_index;
        let digest = leaf.hash(ProtocolVersion::V1)?;
        if index == target_index as usize {
            captured_leaf = Some(leaf);
        }
        append_proof_node(
            &mut frontier,
            &mut siblings,
            target_index as usize,
            &mut next_index,
            digest,
        )
    })?;
    let empty = crate::hash::empty_leaf(HashAlgorithm::Sha256V1, D_WEIGHT_TREE, 0)?;
    while next_index < capacity {
        append_proof_node(
            &mut frontier,
            &mut siblings,
            target_index as usize,
            &mut next_index,
            empty,
        )?;
    }
    let root = frontier[depth]
        .map(|value| value.0)
        .ok_or(CommitError::InvalidMerkleProof)?;
    let proof = MerkleProof {
        leaf_index: target_index,
        leaf_count: u32::try_from(leaf_count).map_err(|_| CommitError::ArithmeticOverflow)?,
        siblings: siblings
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(CommitError::InvalidMerkleProof)?,
    };
    Ok(WeightProofV1 {
        commitment: WeightCommitmentV1 {
            root,
            leaf_count: proof.leaf_count,
            tensor_count: u32::try_from(ordered.len())
                .map_err(|_| CommitError::ArithmeticOverflow)?,
        },
        leaf: captured_leaf.ok_or(CommitError::InvalidMerkleProof)?,
        proof,
    })
}

fn append_proof_node(
    frontier: &mut [Option<(Digest, usize, usize)>],
    siblings: &mut [Option<Digest>],
    target: usize,
    next_index: &mut usize,
    digest: Digest,
) -> Result<(), CommitError> {
    let mut node = (digest, *next_index, 1_usize);
    *next_index = next_index
        .checked_add(1)
        .ok_or(CommitError::ArithmeticOverflow)?;
    let mut level = 0;
    while let Some(left) = frontier
        .get_mut(level)
        .ok_or(CommitError::InvalidMerkleProof)?
        .take()
    {
        if (left.1..left.1 + left.2).contains(&target) {
            siblings[level] = Some(node.0);
        } else if (node.1..node.1 + node.2).contains(&target) {
            siblings[level] = Some(left.0);
        }
        node = (
            hash_domain(
                HashAlgorithm::Sha256V1,
                crate::D_MERKLE,
                &[D_WEIGHT_TREE, &left.0, &node.0],
            )?,
            left.1,
            left.2
                .checked_mul(2)
                .ok_or(CommitError::ArithmeticOverflow)?,
        );
        level += 1;
    }
    let slot = frontier
        .get_mut(level)
        .ok_or(CommitError::InvalidMerkleProof)?;
    *slot = Some(node);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use half::f16;
    use qx_format::{CompiledTensor, MemoryPlan};
    use qx_quant::{PackingLayout, QuantizedTensor};

    use super::*;

    #[test]
    fn artifact_root_ignores_physical_q4_layout_and_proofs_verify() {
        let values = (0..64)
            .map(|value| value as f32 / 10.0 - 3.0)
            .collect::<Vec<_>>();
        let base = QuantizedTensor::quantize(&values, 2, 32, QuantType::Q4, 32);
        let mut tiled = base.clone();
        // Two rows cannot use MetalTile32; row-inline still exercises physical
        // code/scale interleaving while preserving the logical groups.
        tiled.repack(PackingLayout::Q4RowInlineRows2).unwrap();
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let first = std::env::temp_dir().join(format!("qx-commit-{suffix}-a.qx"));
        let second = std::env::temp_dir().join(format!("qx-commit-{suffix}-b.qx"));
        for (path, tensor) in [(&first, base), (&second, tiled)] {
            qx_format::write_artifact(
                path,
                "test",
                "cpu",
                4,
                MemoryPlan::default(),
                &[CompiledTensor {
                    layer: 0,
                    expert: 0,
                    kind: TensorKind::GateUp,
                    tensor,
                }],
            )
            .unwrap();
        }
        let first_root = commit_q4_artifact(&first).unwrap();
        let second_root = commit_q4_artifact(&second).unwrap();
        assert_eq!(first_root, second_root);
        let proof = prove_q4_artifact_leaf(&second, 1).unwrap();
        assert!(proof.verify().unwrap());
        fs::remove_file(first).unwrap();
        fs::remove_file(second).unwrap();

        let _ = f16::from_f32(1.0); // keep dev dependency covered on all targets
    }
}
