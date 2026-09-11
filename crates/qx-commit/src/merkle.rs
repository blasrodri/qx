use crate::{
    CommitError, D_MERKLE, Digest, HashAlgorithm, MAX_MERKLE_DEPTH, hash::empty_leaf, hash_domain,
};

fn parent(
    algorithm: HashAlgorithm,
    tree_domain: &'static [u8],
    left: &Digest,
    right: &Digest,
) -> Result<Digest, CommitError> {
    hash_domain(algorithm, D_MERKLE, &[tree_domain, left, right])
}

fn depth_for_leaves(leaves: usize) -> Result<usize, CommitError> {
    if leaves == 0 {
        return Err(CommitError::EmptyTree);
    }
    let capacity = leaves
        .checked_next_power_of_two()
        .ok_or(CommitError::ArithmeticOverflow)?;
    Ok(capacity.trailing_zeros() as usize)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MerkleProof {
    pub leaf_index: u32,
    pub leaf_count: u32,
    pub siblings: Vec<Digest>,
}

impl MerkleProof {
    pub fn verify(
        &self,
        algorithm: HashAlgorithm,
        tree_domain: &'static [u8],
        leaf: Digest,
        expected_root: Digest,
    ) -> Result<bool, CommitError> {
        if self.leaf_count == 0 || self.leaf_index >= self.leaf_count {
            return Err(CommitError::InvalidMerkleProof);
        }
        let expected_depth = depth_for_leaves(self.leaf_count as usize)?;
        if self.siblings.len() != expected_depth || expected_depth > MAX_MERKLE_DEPTH {
            return Err(CommitError::InvalidMerkleDepth(self.siblings.len()));
        }
        let mut node = leaf;
        let mut index = self.leaf_index as usize;
        for sibling in &self.siblings {
            node = if index & 1 == 0 {
                parent(algorithm, tree_domain, &node, sibling)?
            } else {
                parent(algorithm, tree_domain, sibling, &node)?
            };
            index >>= 1;
        }
        Ok(node == expected_root)
    }
}

#[derive(Clone, Debug)]
pub struct MerkleTree {
    algorithm: HashAlgorithm,
    tree_domain: &'static [u8],
    leaf_count: usize,
    levels: Vec<Vec<Digest>>,
}

impl MerkleTree {
    pub fn new(
        algorithm: HashAlgorithm,
        tree_domain: &'static [u8],
        leaves: &[Digest],
    ) -> Result<Self, CommitError> {
        let depth = depth_for_leaves(leaves.len())?;
        if depth > MAX_MERKLE_DEPTH {
            return Err(CommitError::InvalidMerkleDepth(depth));
        }
        let capacity = 1_usize << depth;
        let mut level = Vec::with_capacity(capacity);
        level.extend_from_slice(leaves);
        for index in leaves.len()..capacity {
            level.push(empty_leaf(algorithm, tree_domain, index as u64)?);
        }
        let mut levels = vec![level];
        while levels.last().map_or(0, Vec::len) > 1 {
            let prior = levels.last().ok_or(CommitError::EmptyTree)?;
            let mut next = Vec::with_capacity(prior.len() / 2);
            for pair in prior.chunks_exact(2) {
                next.push(parent(algorithm, tree_domain, &pair[0], &pair[1])?);
            }
            levels.push(next);
        }
        Ok(Self {
            algorithm,
            tree_domain,
            leaf_count: leaves.len(),
            levels,
        })
    }

    pub fn root(&self) -> Digest {
        self.levels[self.levels.len() - 1][0]
    }

    pub fn proof(&self, leaf_index: usize) -> Result<MerkleProof, CommitError> {
        if leaf_index >= self.leaf_count {
            return Err(CommitError::InvalidIndex {
                index: leaf_index,
                length: self.leaf_count,
            });
        }
        let mut index = leaf_index;
        let mut siblings = Vec::with_capacity(self.levels.len().saturating_sub(1));
        for level in self.levels.iter().take(self.levels.len().saturating_sub(1)) {
            siblings.push(level[index ^ 1]);
            index >>= 1;
        }
        Ok(MerkleProof {
            leaf_index: u32::try_from(leaf_index).map_err(|_| CommitError::ArithmeticOverflow)?,
            leaf_count: u32::try_from(self.leaf_count)
                .map_err(|_| CommitError::ArithmeticOverflow)?,
            siblings,
        })
    }

    pub fn verify(&self, leaf: Digest, proof: &MerkleProof) -> Result<bool, CommitError> {
        proof.verify(self.algorithm, self.tree_domain, leaf, self.root())
    }
}

/// Fixed-capacity streaming Merkle accumulator with O(log N) hash storage.
#[derive(Clone, Debug)]
pub struct FixedMerkleAccumulator {
    algorithm: HashAlgorithm,
    tree_domain: &'static [u8],
    depth: usize,
    count: usize,
    frontier: Vec<Option<Digest>>,
    empty_subtrees: Vec<Digest>,
}

impl FixedMerkleAccumulator {
    pub fn new(
        algorithm: HashAlgorithm,
        tree_domain: &'static [u8],
        depth: usize,
    ) -> Result<Self, CommitError> {
        if depth > MAX_MERKLE_DEPTH {
            return Err(CommitError::InvalidMerkleDepth(depth));
        }
        let mut empty_subtrees = Vec::with_capacity(depth + 1);
        empty_subtrees.push(empty_leaf(algorithm, tree_domain, 0)?);
        for level in 0..depth {
            let value = parent(
                algorithm,
                tree_domain,
                &empty_subtrees[level],
                &empty_subtrees[level],
            )?;
            empty_subtrees.push(value);
        }
        Ok(Self {
            algorithm,
            tree_domain,
            depth,
            count: 0,
            frontier: vec![None; depth + 1],
            empty_subtrees,
        })
    }

    pub fn append(&mut self, leaf: Digest) -> Result<(), CommitError> {
        let capacity = 1_usize
            .checked_shl(self.depth as u32)
            .ok_or(CommitError::ArithmeticOverflow)?;
        if self.count >= capacity {
            return Err(CommitError::TreeCapacityExceeded);
        }
        let mut node = leaf;
        let mut index = self.count;
        for level in 0..=self.depth {
            if index & 1 == 0 {
                self.frontier[level] = Some(node);
                break;
            }
            let left = self.frontier[level]
                .take()
                .ok_or(CommitError::InvalidMerkleProof)?;
            node = parent(self.algorithm, self.tree_domain, &left, &node)?;
            index >>= 1;
        }
        self.count += 1;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn root(&self) -> Result<Digest, CommitError> {
        self.root_at_depth(self.depth)
    }

    /// Root of the smallest complete tree that contains every appended leaf.
    /// This is the root authenticated by a `MerkleProof` whose `leaf_count`
    /// equals `len()`, even when the accumulator was provisioned with a much
    /// larger maximum capacity.
    pub fn compact_root(&self) -> Result<Digest, CommitError> {
        if self.count == 0 {
            return Err(CommitError::EmptyTree);
        }
        self.root_at_depth(depth_for_leaves(self.count)?)
    }

    fn root_at_depth(&self, depth: usize) -> Result<Digest, CommitError> {
        if depth > self.depth {
            return Err(CommitError::InvalidMerkleDepth(depth));
        }
        let capacity = 1_usize << depth;
        if self.count > capacity {
            return Err(CommitError::TreeCapacityExceeded);
        }
        if self.count == capacity {
            return self.frontier[depth].ok_or(CommitError::InvalidMerkleProof);
        }
        let mut node = self.empty_subtrees[0];
        let mut index = self.count;
        for level in 0..depth {
            node = if index & 1 == 1 {
                let left = self.frontier[level].ok_or(CommitError::InvalidMerkleProof)?;
                parent(self.algorithm, self.tree_domain, &left, &node)?
            } else {
                parent(
                    self.algorithm,
                    self.tree_domain,
                    &node,
                    &self.empty_subtrees[level],
                )?
            };
            index >>= 1;
        }
        Ok(node)
    }
}

/// Mutable complete tree used for the 24 per-layer KV roots.
#[derive(Clone, Debug)]
pub struct MutableMerkleTree {
    algorithm: HashAlgorithm,
    tree_domain: &'static [u8],
    leaf_count: usize,
    levels: Vec<Vec<Digest>>,
}

impl MutableMerkleTree {
    pub fn new(
        algorithm: HashAlgorithm,
        tree_domain: &'static [u8],
        leaf_count: usize,
    ) -> Result<Self, CommitError> {
        let leaves = (0..leaf_count)
            .map(|index| empty_leaf(algorithm, tree_domain, index as u64))
            .collect::<Result<Vec<_>, _>>()?;
        let tree = MerkleTree::new(algorithm, tree_domain, &leaves)?;
        Ok(Self {
            algorithm,
            tree_domain,
            leaf_count,
            levels: tree.levels,
        })
    }

    pub fn update(&mut self, leaf_index: usize, leaf: Digest) -> Result<(), CommitError> {
        if leaf_index >= self.leaf_count {
            return Err(CommitError::InvalidIndex {
                index: leaf_index,
                length: self.leaf_count,
            });
        }
        self.levels[0][leaf_index] = leaf;
        let mut index = leaf_index;
        for level in 1..self.levels.len() {
            index >>= 1;
            let prior = &self.levels[level - 1];
            self.levels[level][index] = parent(
                self.algorithm,
                self.tree_domain,
                &prior[index * 2],
                &prior[index * 2 + 1],
            )?;
        }
        Ok(())
    }

    pub fn root(&self) -> Digest {
        self.levels[self.levels.len() - 1][0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D_TEST: &[u8] = b"QX:TEST:TREE:V1";

    #[test]
    fn proofs_round_trip_and_reject_tampering() {
        let leaves = (0_u8..7)
            .map(|value| {
                hash_domain(HashAlgorithm::Sha256V1, b"QX:TEST:LEAF:V1", &[&[value]]).unwrap()
            })
            .collect::<Vec<_>>();
        let tree = MerkleTree::new(HashAlgorithm::Sha256V1, D_TEST, &leaves).unwrap();
        for (index, leaf) in leaves.iter().copied().enumerate() {
            let proof = tree.proof(index).unwrap();
            assert!(tree.verify(leaf, &proof).unwrap());
            let mut corrupt = leaf;
            corrupt[0] ^= 1;
            assert!(!tree.verify(corrupt, &proof).unwrap());
        }
    }

    #[test]
    fn streaming_root_matches_materialized_tree() {
        for count in 1..=8 {
            let leaves = (0..count)
                .map(|value| {
                    hash_domain(
                        HashAlgorithm::Sha256V1,
                        b"QX:TEST:LEAF:V1",
                        &[&(value as u64).to_le_bytes()],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let tree = MerkleTree::new(HashAlgorithm::Sha256V1, D_TEST, &leaves).unwrap();
            let depth = depth_for_leaves(count).unwrap();
            let mut stream =
                FixedMerkleAccumulator::new(HashAlgorithm::Sha256V1, D_TEST, depth).unwrap();
            for leaf in &leaves {
                stream.append(*leaf).unwrap();
            }
            assert_eq!(stream.root().unwrap(), tree.root());
        }
    }

    #[test]
    fn compact_streaming_root_ignores_unused_maximum_capacity() {
        for count in 1..=8 {
            let leaves = (0..count)
                .map(|value| {
                    hash_domain(
                        HashAlgorithm::Blake3V1,
                        b"QX:TEST:LEAF:V1",
                        &[&(value as u64).to_le_bytes()],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let tree = MerkleTree::new(HashAlgorithm::Blake3V1, D_TEST, &leaves).unwrap();
            let mut stream =
                FixedMerkleAccumulator::new(HashAlgorithm::Blake3V1, D_TEST, 12).unwrap();
            for leaf in &leaves {
                stream.append(*leaf).unwrap();
            }
            assert_eq!(stream.compact_root().unwrap(), tree.root());
        }
    }
}
