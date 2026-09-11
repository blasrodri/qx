use crate::{
    CommitError, D_ACTIVATION_CHUNK, D_ACTIVATION_TREE, D_KV, D_KV_KEY_CHUNK, D_KV_KEY_TREE,
    D_KV_VALUE_CHUNK, D_KV_VALUE_TREE, D_MERKLE_EMPTY, D_STATE, D_STATE_CHUNK, Digest,
    HashAlgorithm, MAX_MERKLE_DEPTH, hash_domain,
};

/// Consensus chunk size for accelerator-friendly state commitments.
pub const STATE_CHUNK_BYTES: usize = 256;
/// Smaller dispute-only leaves keep one weight opening plus one activation
/// opening inside Solana's transaction-size envelope.
pub const ACTIVATION_CHUNK_BYTES: usize = 128;

/// Identifies the leaf domain of a V1 state opening. Keeping this value in the
/// proof prevents a residual chunk from being replayed as a K/V or dispute
/// activation chunk.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum StateTreeKindV1 {
    Residual = 1,
    KvKey = 2,
    KvValue = 3,
    Activation = 4,
}

impl StateTreeKindV1 {
    fn domains(self) -> (&'static [u8], &'static [u8]) {
        match self {
            Self::Residual => (D_STATE_CHUNK, D_STATE),
            Self::KvKey => (D_KV_KEY_CHUNK, D_KV_KEY_TREE),
            Self::KvValue => (D_KV_VALUE_CHUNK, D_KV_VALUE_TREE),
            Self::Activation => (D_ACTIVATION_CHUNK, D_ACTIVATION_TREE),
        }
    }

    fn chunk_bytes(self) -> usize {
        match self {
            Self::Activation => ACTIVATION_CHUNK_BYTES,
            Self::Residual | Self::KvKey | Self::KvValue => STATE_CHUNK_BYTES,
        }
    }
}

/// An opening of one 256-byte (or final partial) leaf in a BLAKE3 state tree.
/// `byte_length` binds the number of leaves and the exact final-chunk length.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateChunkProofV1 {
    pub kind: StateTreeKindV1,
    pub byte_length: u32,
    pub chunk_index: u32,
    pub chunk: Vec<u8>,
    pub siblings: Vec<Digest>,
}

impl StateChunkProofV1 {
    pub fn verify(&self, expected_root: Digest) -> Result<bool, CommitError> {
        let byte_length =
            usize::try_from(self.byte_length).map_err(|_| CommitError::ArithmeticOverflow)?;
        if byte_length == 0 {
            return Err(CommitError::InvalidMerkleProof);
        }
        let chunk_bytes = self.kind.chunk_bytes();
        let leaf_count = byte_length.div_ceil(chunk_bytes);
        let index =
            usize::try_from(self.chunk_index).map_err(|_| CommitError::ArithmeticOverflow)?;
        if index >= leaf_count {
            return Err(CommitError::InvalidMerkleProof);
        }
        let expected_chunk_len = if index + 1 == leaf_count {
            byte_length - index * chunk_bytes
        } else {
            chunk_bytes
        };
        if self.chunk.len() != expected_chunk_len {
            return Err(CommitError::InvalidMerkleProof);
        }
        let capacity = leaf_count
            .checked_next_power_of_two()
            .ok_or(CommitError::ArithmeticOverflow)?;
        let depth = capacity.trailing_zeros() as usize;
        if depth > MAX_MERKLE_DEPTH || self.siblings.len() != depth {
            return Err(CommitError::InvalidMerkleDepth(self.siblings.len()));
        }
        let (leaf_domain, _) = self.kind.domains();
        let mut node = hash_domain(
            HashAlgorithm::Blake3V1,
            leaf_domain,
            &[&self.chunk_index.to_le_bytes(), &self.chunk],
        )?;
        let mut position = index;
        for sibling in &self.siblings {
            node = if position & 1 == 0 {
                state_parent_v1(&node, sibling)
            } else {
                state_parent_v1(sibling, &node)
            };
            position >>= 1;
        }
        Ok(node == expected_root)
    }

    /// Exact consensus payload size before an outer instruction discriminator.
    pub fn encoded_len(&self) -> Result<usize, CommitError> {
        1_usize
            .checked_add(4)
            .and_then(|value| value.checked_add(4))
            .and_then(|value| value.checked_add(2))
            .and_then(|value| value.checked_add(self.chunk.len()))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_add(self.siblings.len() * 32))
            .ok_or(CommitError::ArithmeticOverflow)
    }

    /// Read one aligned FP32 word addressed in the canonical byte image.
    /// VM instructions use byte addresses so the same commitment can cover
    /// residual vectors and any execution-spec-defined scratch region.
    pub fn scalar_bits(&self, byte_address: u32) -> Result<u32, CommitError> {
        if self.kind != StateTreeKindV1::Residual || byte_address & 3 != 0 {
            return Err(CommitError::InvalidMerkleProof);
        }
        let address = usize::try_from(byte_address).map_err(|_| CommitError::ArithmeticOverflow)?;
        let image_len =
            usize::try_from(self.byte_length).map_err(|_| CommitError::ArithmeticOverflow)?;
        let chunk_start = usize::try_from(self.chunk_index)
            .map_err(|_| CommitError::ArithmeticOverflow)?
            .checked_mul(STATE_CHUNK_BYTES)
            .ok_or(CommitError::ArithmeticOverflow)?;
        if address
            .checked_add(4)
            .ok_or(CommitError::ArithmeticOverflow)?
            > image_len
            || address < chunk_start
            || address + 4 > chunk_start + self.chunk.len()
        {
            return Err(CommitError::InvalidMerkleProof);
        }
        let offset = address - chunk_start;
        Ok(u32::from_le_bytes(
            self.chunk[offset..offset + 4]
                .try_into()
                .map_err(|_| CommitError::InvalidMerkleProof)?,
        ))
    }

    /// Read authenticated bytes wholly contained in this residual chunk.
    /// Structured VM operations use this for fixed-size roots without
    /// interpreting their bytes as floating-point values.
    pub fn bytes_at(&self, byte_address: u32, length: usize) -> Result<&[u8], CommitError> {
        if self.kind != StateTreeKindV1::Residual {
            return Err(CommitError::InvalidMerkleProof);
        }
        let address = usize::try_from(byte_address).map_err(|_| CommitError::ArithmeticOverflow)?;
        let image_len =
            usize::try_from(self.byte_length).map_err(|_| CommitError::ArithmeticOverflow)?;
        let chunk_start = usize::try_from(self.chunk_index)
            .map_err(|_| CommitError::ArithmeticOverflow)?
            .checked_mul(STATE_CHUNK_BYTES)
            .ok_or(CommitError::ArithmeticOverflow)?;
        let end = address
            .checked_add(length)
            .ok_or(CommitError::ArithmeticOverflow)?;
        if length == 0
            || end > image_len
            || address < chunk_start
            || end > chunk_start + self.chunk.len()
        {
            return Err(CommitError::InvalidMerkleProof);
        }
        Ok(&self.chunk[address - chunk_start..end - chunk_start])
    }

    /// Calculate the unique residual root after replacing one authenticated
    /// FP32 word. Callers must verify this opening against the pre-state root.
    pub fn root_after_scalar_write(
        &self,
        byte_address: u32,
        value_bits: u32,
    ) -> Result<Digest, CommitError> {
        self.scalar_bits(byte_address)?;
        let chunk_start = usize::try_from(self.chunk_index)
            .map_err(|_| CommitError::ArithmeticOverflow)?
            .checked_mul(STATE_CHUNK_BYTES)
            .ok_or(CommitError::ArithmeticOverflow)?;
        let offset = usize::try_from(byte_address).map_err(|_| CommitError::ArithmeticOverflow)?
            - chunk_start;
        let mut chunk = self.chunk.clone();
        chunk[offset..offset + 4].copy_from_slice(&value_bits.to_le_bytes());
        let mut node = hash_domain(
            HashAlgorithm::Blake3V1,
            D_STATE_CHUNK,
            &[&self.chunk_index.to_le_bytes(), &chunk],
        )?;
        let mut position =
            usize::try_from(self.chunk_index).map_err(|_| CommitError::ArithmeticOverflow)?;
        for sibling in &self.siblings {
            node = if position & 1 == 0 {
                state_parent_v1(&node, sibling)
            } else {
                state_parent_v1(sibling, &node)
            };
            position >>= 1;
        }
        Ok(node)
    }
}

fn state_parent_v1(left: &Digest, right: &Digest) -> Digest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

fn chunk_tree_hash(
    bytes: &[u8],
    chunk_bytes: usize,
    leaf_domain: &'static [u8],
    tree_domain: &'static [u8],
) -> Result<Digest, CommitError> {
    if bytes.is_empty() {
        return Err(CommitError::InvalidShape(
            "state commitment input must not be empty".to_owned(),
        ));
    }
    let leaves = bytes
        .chunks(chunk_bytes)
        .enumerate()
        .map(|(index, chunk)| {
            let index = u32::try_from(index).map_err(|_| CommitError::ArithmeticOverflow)?;
            hash_domain(
                HashAlgorithm::Blake3V1,
                leaf_domain,
                &[&index.to_le_bytes(), chunk],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let capacity = leaves
        .len()
        .checked_next_power_of_two()
        .ok_or(CommitError::ArithmeticOverflow)?;
    let mut nodes = leaves;
    let empty = hash_domain(HashAlgorithm::Blake3V1, D_MERKLE_EMPTY, &[tree_domain])?;
    nodes.resize(capacity, empty);
    while nodes.len() > 1 {
        let mut parents = Vec::with_capacity(nodes.len() / 2);
        for pair in nodes.chunks_exact(2) {
            parents.push(state_parent_v1(&pair[0], &pair[1]));
        }
        nodes = parents;
    }
    nodes.pop().ok_or(CommitError::EmptyTree)
}

/// Commit canonical post-layer residual bytes using independently hashable
/// chunks followed by a typed-leaf BLAKE3 Merkle reduction.
pub fn state_hash_v1(bytes: &[u8]) -> Result<Digest, CommitError> {
    chunk_tree_hash(bytes, STATE_CHUNK_BYTES, D_STATE_CHUNK, D_STATE)
}

/// Commit a dispute-local activation vector. This is not emitted during normal
/// inference; it is constructed only after a disputed operator is identified.
pub fn activation_hash_v1(bytes: &[u8]) -> Result<Digest, CommitError> {
    chunk_tree_hash(
        bytes,
        ACTIVATION_CHUNK_BYTES,
        D_ACTIVATION_CHUNK,
        D_ACTIVATION_TREE,
    )
}

pub fn prove_state_chunk_v1(
    kind: StateTreeKindV1,
    bytes: &[u8],
    chunk_index: u32,
) -> Result<StateChunkProofV1, CommitError> {
    if bytes.is_empty() {
        return Err(CommitError::InvalidShape(
            "state commitment input must not be empty".to_owned(),
        ));
    }
    let index = usize::try_from(chunk_index).map_err(|_| CommitError::ArithmeticOverflow)?;
    let chunk_bytes = kind.chunk_bytes();
    let leaf_count = bytes.len().div_ceil(chunk_bytes);
    if index >= leaf_count {
        return Err(CommitError::InvalidIndex {
            index,
            length: leaf_count,
        });
    }
    let (leaf_domain, tree_domain) = kind.domains();
    let mut nodes = bytes
        .chunks(chunk_bytes)
        .enumerate()
        .map(|(leaf_index, chunk)| {
            let leaf_index =
                u32::try_from(leaf_index).map_err(|_| CommitError::ArithmeticOverflow)?;
            hash_domain(
                HashAlgorithm::Blake3V1,
                leaf_domain,
                &[&leaf_index.to_le_bytes(), chunk],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let capacity = leaf_count
        .checked_next_power_of_two()
        .ok_or(CommitError::ArithmeticOverflow)?;
    let empty = hash_domain(HashAlgorithm::Blake3V1, D_MERKLE_EMPTY, &[tree_domain])?;
    nodes.resize(capacity, empty);
    let mut position = index;
    let mut siblings = Vec::with_capacity(capacity.trailing_zeros() as usize);
    while nodes.len() > 1 {
        siblings.push(nodes[position ^ 1]);
        let mut parents = Vec::with_capacity(nodes.len() / 2);
        for pair in nodes.chunks_exact(2) {
            parents.push(state_parent_v1(&pair[0], &pair[1]));
        }
        nodes = parents;
        position >>= 1;
    }
    let start = index
        .checked_mul(chunk_bytes)
        .ok_or(CommitError::ArithmeticOverflow)?;
    let end = bytes.len().min(start + chunk_bytes);
    Ok(StateChunkProofV1 {
        kind,
        byte_length: u32::try_from(bytes.len()).map_err(|_| CommitError::ArithmeticOverflow)?,
        chunk_index,
        chunk: bytes[start..end].to_vec(),
        siblings,
    })
}

/// Commit the newly appended K/V vectors for one layer and token. K and V
/// receive separate chunk trees so an eventual dispute can reveal one path.
pub fn kv_leaf_hash_v1(
    layer_index: u16,
    token_index: u32,
    key_bytes: &[u8],
    value_bytes: &[u8],
) -> Result<Digest, CommitError> {
    let key_root = chunk_tree_hash(key_bytes, STATE_CHUNK_BYTES, D_KV_KEY_CHUNK, D_KV_KEY_TREE)?;
    let value_root = chunk_tree_hash(
        value_bytes,
        STATE_CHUNK_BYTES,
        D_KV_VALUE_CHUNK,
        D_KV_VALUE_TREE,
    )?;
    kv_leaf_hash_from_roots_v1(layer_index, token_index, key_root, value_root)
}

/// Construct the canonical KV leaf from already authenticated K/V chunk-tree
/// roots. This is the form used by a structured append fraud proof.
pub fn kv_leaf_hash_from_roots_v1(
    layer_index: u16,
    token_index: u32,
    key_root: Digest,
    value_root: Digest,
) -> Result<Digest, CommitError> {
    hash_domain(
        HashAlgorithm::Blake3V1,
        D_KV,
        &[
            &layer_index.to_le_bytes(),
            &token_index.to_le_bytes(),
            &key_root,
            &value_root,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_boundaries_and_coordinates_are_committed() {
        let bytes = vec![7; STATE_CHUNK_BYTES + 1];
        assert_ne!(
            state_hash_v1(&bytes).unwrap(),
            state_hash_v1(&bytes[..STATE_CHUNK_BYTES]).unwrap()
        );
        let key = vec![1; STATE_CHUNK_BYTES];
        let value = vec![2; STATE_CHUNK_BYTES];
        assert_ne!(
            kv_leaf_hash_v1(1, 2, &key, &value).unwrap(),
            kv_leaf_hash_v1(2, 2, &key, &value).unwrap()
        );
    }

    #[test]
    fn state_chunk_openings_bind_kind_position_length_and_data() {
        let bytes = (0..777).map(|value| value as u8).collect::<Vec<_>>();
        let root = state_hash_v1(&bytes).unwrap();
        let proof = prove_state_chunk_v1(StateTreeKindV1::Residual, &bytes, 3).unwrap();
        assert!(proof.verify(root).unwrap());
        assert_eq!(proof.chunk.len(), 9);
        assert_eq!(proof.encoded_len().unwrap(), 1 + 4 + 4 + 2 + 9 + 1 + 64);

        let mut corrupt = proof.clone();
        corrupt.chunk[0] ^= 1;
        assert!(!corrupt.verify(root).unwrap());
        let mut wrong_kind = proof.clone();
        wrong_kind.kind = StateTreeKindV1::Activation;
        assert!(wrong_kind.verify(root).is_err());
        let mut wrong_length = proof;
        wrong_length.byte_length -= 1;
        assert!(wrong_length.verify(root).is_err());
    }

    #[test]
    fn activation_openings_use_a_separate_domain() {
        let bytes = vec![3; 256];
        assert_ne!(
            activation_hash_v1(&bytes).unwrap(),
            state_hash_v1(&bytes).unwrap()
        );
        let proof = prove_state_chunk_v1(StateTreeKindV1::Activation, &bytes, 0).unwrap();
        assert!(proof.verify(activation_hash_v1(&bytes).unwrap()).unwrap());
    }
}
