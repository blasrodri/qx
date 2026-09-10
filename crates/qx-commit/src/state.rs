use crate::{
    CommitError, D_KV, D_KV_KEY_CHUNK, D_KV_KEY_TREE, D_KV_VALUE_CHUNK, D_KV_VALUE_TREE, D_STATE,
    D_STATE_CHUNK, Digest, HashAlgorithm, MerkleTree, hash_domain,
};

/// Consensus chunk size for accelerator-friendly state commitments.
pub const STATE_CHUNK_BYTES: usize = 256;

fn chunk_tree_hash(
    bytes: &[u8],
    leaf_domain: &'static [u8],
    tree_domain: &'static [u8],
) -> Result<Digest, CommitError> {
    if bytes.is_empty() {
        return Err(CommitError::InvalidShape(
            "state commitment input must not be empty".to_owned(),
        ));
    }
    let leaves = bytes
        .chunks(STATE_CHUNK_BYTES)
        .enumerate()
        .map(|(index, chunk)| {
            let index = u32::try_from(index).map_err(|_| CommitError::ArithmeticOverflow)?;
            hash_domain(
                HashAlgorithm::Sha256V1,
                leaf_domain,
                &[&index.to_le_bytes(), chunk],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MerkleTree::new(HashAlgorithm::Sha256V1, tree_domain, &leaves)?.root())
}

/// Commit canonical post-layer residual bytes using independently hashable
/// chunks followed by a domain-separated Merkle reduction.
pub fn state_hash_v1(bytes: &[u8]) -> Result<Digest, CommitError> {
    chunk_tree_hash(bytes, D_STATE_CHUNK, D_STATE)
}

/// Commit the newly appended K/V vectors for one layer and token. K and V
/// receive separate chunk trees so an eventual dispute can reveal one path.
pub fn kv_leaf_hash_v1(
    layer_index: u16,
    token_index: u32,
    key_bytes: &[u8],
    value_bytes: &[u8],
) -> Result<Digest, CommitError> {
    let key_root = chunk_tree_hash(key_bytes, D_KV_KEY_CHUNK, D_KV_KEY_TREE)?;
    let value_root = chunk_tree_hash(value_bytes, D_KV_VALUE_CHUNK, D_KV_VALUE_TREE)?;
    hash_domain(
        HashAlgorithm::Sha256V1,
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
}
