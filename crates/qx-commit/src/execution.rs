use crate::{
    CommitError, D_CHECKPOINT, D_JOB, D_KV, D_TOKEN, D_TOKEN_RESULT, Digest,
    FixedMerkleAccumulator, MerkleTree, MutableMerkleTree, TRACE_HASH_ALGORITHM, hash_domain,
    kv_leaf_hash_v1, state_hash_v1,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointV1 {
    pub job_id: Digest,
    pub token_index: u32,
    pub layer_index: u16,
    pub residual_hash: Digest,
    pub global_kv_root: Digest,
}

impl CheckpointV1 {
    pub const CANONICAL_SIZE: usize = 102;

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::CANONICAL_SIZE);
        bytes.extend_from_slice(&self.job_id);
        bytes.extend_from_slice(&self.token_index.to_le_bytes());
        bytes.extend_from_slice(&self.layer_index.to_le_bytes());
        bytes.extend_from_slice(&self.residual_hash);
        bytes.extend_from_slice(&self.global_kv_root);
        debug_assert_eq!(bytes.len(), Self::CANONICAL_SIZE);
        bytes
    }

    pub fn hash(&self) -> Result<Digest, CommitError> {
        hash_domain(
            TRACE_HASH_ALGORITHM,
            D_CHECKPOINT,
            &[&self.canonical_bytes()],
        )
    }
}

/// One executed token's checkpoint root and deterministic next-token result.
/// Binding the prediction prevents a miner from pairing an honest execution
/// trace with a different output sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenResultV1 {
    pub job_id: Digest,
    pub token_index: u32,
    pub checkpoint_root: Digest,
    pub predicted_token: u32,
}

impl TokenResultV1 {
    pub const CANONICAL_SIZE: usize = 72;

    pub fn canonical_bytes(&self) -> [u8; Self::CANONICAL_SIZE] {
        let mut bytes = [0; Self::CANONICAL_SIZE];
        bytes[..32].copy_from_slice(&self.job_id);
        bytes[32..36].copy_from_slice(&self.token_index.to_le_bytes());
        bytes[36..68].copy_from_slice(&self.checkpoint_root);
        bytes[68..72].copy_from_slice(&self.predicted_token.to_le_bytes());
        bytes
    }

    pub fn hash(&self) -> Result<Digest, CommitError> {
        hash_domain(
            TRACE_HASH_ALGORITHM,
            D_TOKEN_RESULT,
            &[&self.canonical_bytes()],
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CommitmentMetrics {
    /// Raw canonical residual + newly appended K/V bytes covered by hashing.
    pub canonical_state_bytes: u64,
    pub checkpoint_count: u64,
    pub executed_tokens: u32,
}

impl CommitmentMetrics {
    pub fn canonical_state_bytes_per_token(&self) -> Option<u64> {
        (self.executed_tokens != 0)
            .then(|| self.canonical_state_bytes / u64::from(self.executed_tokens))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionTraceV1 {
    pub trace_root: Digest,
    pub executed_tokens: u32,
    pub metrics: CommitmentMetrics,
}

/// CPU reference commitment state for one sequential batch-1 execution.
pub struct ExecutionCommitterV1 {
    job_id: Digest,
    layers: usize,
    checkpoint_interval: usize,
    expected_layer: usize,
    expected_token: u32,
    max_context: usize,
    kv_trees: Vec<FixedMerkleAccumulator>,
    global_kv: MutableMerkleTree,
    token_checkpoints: Vec<Digest>,
    pending_token_root: Option<Digest>,
    job_tree: FixedMerkleAccumulator,
    metrics: CommitmentMetrics,
}

impl ExecutionCommitterV1 {
    pub fn new(job_id: Digest, layers: usize, max_context: usize) -> Result<Self, CommitError> {
        Self::new_with_checkpoint_interval(job_id, layers, max_context, 1)
    }

    pub fn new_with_checkpoint_interval(
        job_id: Digest,
        layers: usize,
        max_context: usize,
        checkpoint_interval: usize,
    ) -> Result<Self, CommitError> {
        if layers == 0 || layers > u16::MAX as usize {
            return Err(CommitError::InvalidShape(format!(
                "layer count {layers} is unsupported"
            )));
        }
        if max_context == 0 || max_context > u32::MAX as usize {
            return Err(CommitError::InvalidShape(format!(
                "context {max_context} is unsupported"
            )));
        }
        if checkpoint_interval == 0 || !layers.is_multiple_of(checkpoint_interval) {
            return Err(CommitError::InvalidShape(format!(
                "checkpoint interval {checkpoint_interval} must divide {layers} layers"
            )));
        }
        let context_capacity = max_context
            .checked_next_power_of_two()
            .ok_or(CommitError::ArithmeticOverflow)?;
        let context_depth = context_capacity.trailing_zeros() as usize;
        let kv_trees = (0..layers)
            .map(|_| FixedMerkleAccumulator::new(TRACE_HASH_ALGORITHM, D_KV, context_depth))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            job_id,
            layers,
            checkpoint_interval,
            expected_layer: 0,
            expected_token: 0,
            max_context,
            kv_trees,
            global_kv: MutableMerkleTree::new(TRACE_HASH_ALGORITHM, D_KV, layers)?,
            token_checkpoints: Vec::with_capacity(layers),
            pending_token_root: None,
            job_tree: FixedMerkleAccumulator::new(TRACE_HASH_ALGORITHM, D_JOB, context_depth)?,
            metrics: CommitmentMetrics::default(),
        })
    }

    pub fn commit_layer(
        &mut self,
        token_index: u32,
        layer_index: u16,
        residual: &[f32],
        new_key: &[f32],
        new_value: &[f32],
    ) -> Result<Option<Digest>, CommitError> {
        if token_index != self.expected_token || layer_index as usize != self.expected_layer {
            return Err(CommitError::InvalidExecution(format!(
                "expected token {} layer {}, received token {token_index} layer {layer_index}",
                self.expected_token, self.expected_layer
            )));
        }
        if self.pending_token_root.is_some() {
            return Err(CommitError::InvalidExecution(
                "predicted token must be committed before the next layer".to_owned(),
            ));
        }
        if token_index as usize >= self.max_context {
            return Err(CommitError::TreeCapacityExceeded);
        }
        if residual.is_empty() || new_key.is_empty() || new_key.len() != new_value.len() {
            return Err(CommitError::InvalidShape(
                "residual and equal-sized K/V slices must be non-empty".to_owned(),
            ));
        }

        let residual_bytes = canonical_f32(residual)?;
        let key_bytes = canonical_f32(new_key)?;
        let value_bytes = canonical_f32(new_value)?;
        let residual_hash = state_hash_v1(&residual_bytes)?;
        let kv_leaf = kv_leaf_hash_v1(layer_index, token_index, &key_bytes, &value_bytes)?;
        let checkpoint = self.is_checkpoint_layer(layer_index as usize);
        let canonical_bytes = key_bytes
            .len()
            .checked_add(value_bytes.len())
            .and_then(|value| value.checked_add(if checkpoint { residual_bytes.len() } else { 0 }))
            .ok_or(CommitError::ArithmeticOverflow)?;
        self.commit_layer_hashes(
            token_index,
            layer_index,
            residual_hash,
            kv_leaf,
            canonical_bytes,
        )
    }

    /// Consume state/KV digests produced by a compatible accelerator. The
    /// caller supplies the number of canonical tensor bytes covered so
    /// receipts retain auditable accounting without retaining those tensors.
    pub fn commit_layer_hashes(
        &mut self,
        token_index: u32,
        layer_index: u16,
        residual_hash: Digest,
        kv_leaf: Digest,
        canonical_bytes: usize,
    ) -> Result<Option<Digest>, CommitError> {
        if token_index != self.expected_token || layer_index as usize != self.expected_layer {
            return Err(CommitError::InvalidExecution(format!(
                "expected token {} layer {}, received token {token_index} layer {layer_index}",
                self.expected_token, self.expected_layer
            )));
        }
        if self.pending_token_root.is_some() {
            return Err(CommitError::InvalidExecution(
                "predicted token must be committed before the next layer".to_owned(),
            ));
        }
        if token_index as usize >= self.max_context {
            return Err(CommitError::TreeCapacityExceeded);
        }
        let kv_tree = &mut self.kv_trees[layer_index as usize];
        kv_tree.append(kv_leaf)?;
        self.global_kv
            .update(layer_index as usize, kv_tree.root()?)?;
        let checkpoint = if self.is_checkpoint_layer(layer_index as usize) {
            let checkpoint = CheckpointV1 {
                job_id: self.job_id,
                token_index,
                layer_index,
                residual_hash,
                global_kv_root: self.global_kv.root(),
            }
            .hash()?;
            self.token_checkpoints.push(checkpoint);
            Some(checkpoint)
        } else {
            None
        };
        self.metrics.canonical_state_bytes = self
            .metrics
            .canonical_state_bytes
            .checked_add(
                u64::try_from(canonical_bytes).map_err(|_| CommitError::ArithmeticOverflow)?,
            )
            .ok_or(CommitError::ArithmeticOverflow)?;
        if checkpoint.is_some() {
            self.metrics.checkpoint_count = self
                .metrics
                .checkpoint_count
                .checked_add(1)
                .ok_or(CommitError::ArithmeticOverflow)?;
        }

        self.expected_layer += 1;
        if self.expected_layer == self.layers {
            let token_tree =
                MerkleTree::new(TRACE_HASH_ALGORITHM, D_TOKEN, &self.token_checkpoints)?;
            self.pending_token_root = Some(token_tree.root());
            self.token_checkpoints.clear();
            self.expected_layer = 0;
        }
        Ok(checkpoint)
    }

    /// Complete one token after the runtime has deterministically selected its
    /// next token. This must occur before any layer of the following token.
    pub fn commit_output_token(
        &mut self,
        token_index: u32,
        predicted_token: u32,
    ) -> Result<Digest, CommitError> {
        if token_index != self.expected_token || self.expected_layer != 0 {
            return Err(CommitError::InvalidExecution(format!(
                "expected output for token {}, received token {token_index}",
                self.expected_token
            )));
        }
        let checkpoint_root = self.pending_token_root.take().ok_or_else(|| {
            CommitError::InvalidExecution("token layers are not complete".to_owned())
        })?;
        let leaf = TokenResultV1 {
            job_id: self.job_id,
            token_index,
            checkpoint_root,
            predicted_token,
        }
        .hash()?;
        self.job_tree.append(leaf)?;
        self.expected_token = self
            .expected_token
            .checked_add(1)
            .ok_or(CommitError::ArithmeticOverflow)?;
        self.metrics.executed_tokens = self.expected_token;
        Ok(leaf)
    }

    fn is_checkpoint_layer(&self, layer_index: usize) -> bool {
        (layer_index + 1).is_multiple_of(self.checkpoint_interval) || layer_index + 1 == self.layers
    }

    pub fn finish(&self) -> Result<ExecutionTraceV1, CommitError> {
        if self.expected_layer != 0 || self.pending_token_root.is_some() {
            return Err(CommitError::InvalidExecution(
                "cannot finish a partially committed token or unbound prediction".to_owned(),
            ));
        }
        if self.job_tree.is_empty() {
            return Err(CommitError::InvalidExecution(
                "cannot finish an empty execution".to_owned(),
            ));
        }
        Ok(ExecutionTraceV1 {
            trace_root: self.job_tree.root()?,
            executed_tokens: self.expected_token,
            metrics: self.metrics,
        })
    }
}

fn canonical_f32(values: &[f32]) -> Result<Vec<u8>, CommitError> {
    let capacity = values
        .len()
        .checked_mul(4)
        .ok_or(CommitError::ArithmeticOverflow)?;
    let mut bytes = Vec::with_capacity(capacity);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_commitments_are_deterministic_and_streaming() {
        fn run(corrupt: bool) -> ExecutionTraceV1 {
            let mut state = ExecutionCommitterV1::new([9; 32], 3, 8).unwrap();
            for token in 0..2 {
                for layer in 0..3 {
                    let mut residual = vec![token as f32, layer as f32, 0.25, -0.0];
                    if corrupt && token == 1 && layer == 2 {
                        residual[0] += 1.0;
                    }
                    state
                        .commit_layer(
                            token,
                            layer,
                            &residual,
                            &[layer as f32, token as f32],
                            &[1.0, -1.0],
                        )
                        .unwrap();
                }
                state.commit_output_token(token, 100 + token).unwrap();
            }
            state.finish().unwrap()
        }
        assert_eq!(run(false), run(false));
        assert_ne!(run(false).trace_root, run(true).trace_root);
        assert_eq!(run(false).metrics.canonical_state_bytes, 192);
    }

    #[test]
    fn coarser_checkpoints_keep_every_kv_leaf_and_reduce_residual_hashing() {
        let mut state =
            ExecutionCommitterV1::new_with_checkpoint_interval([4; 32], 4, 8, 2).unwrap();
        for layer in 0..4 {
            state
                .commit_layer(0, layer, &[layer as f32; 4], &[1.0, 2.0], &[3.0, 4.0])
                .unwrap();
        }
        state.commit_output_token(0, 17).unwrap();
        let trace = state.finish().unwrap();
        assert_eq!(trace.metrics.checkpoint_count, 2);
        // Four 16-byte K/V pairs plus two 16-byte residuals.
        assert_eq!(trace.metrics.canonical_state_bytes, 96);
        assert_ne!(
            crate::granite_execution_spec().unwrap(),
            crate::granite_execution_spec_with_checkpoint_interval(2).unwrap()
        );
    }

    #[test]
    fn prediction_is_bound_into_the_job_trace() {
        fn root(predicted_token: u32) -> Digest {
            let mut state = ExecutionCommitterV1::new([6; 32], 1, 2).unwrap();
            state.commit_layer(0, 0, &[1.0], &[2.0], &[3.0]).unwrap();
            state.commit_output_token(0, predicted_token).unwrap();
            state.finish().unwrap().trace_root
        }
        assert_ne!(root(11), root(12));
    }
}
