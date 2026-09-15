use crate::{
    CommitError, D_CHECKPOINT, D_JOB, D_JOB_GENESIS, D_KV, D_TOKEN, D_TOKEN_INPUT_STATE,
    D_TOKEN_RESULT, Digest, FixedMerkleAccumulator, MerkleProof, MerkleTree, MutableMerkleTree,
    TRACE_HASH_ALGORITHM, hash_domain, kv_leaf_hash_v1, state_hash_v1,
};

pub fn job_genesis_hash_v1(job_id: &Digest) -> Result<Digest, CommitError> {
    hash_domain(TRACE_HASH_ALGORITHM, D_JOB_GENESIS, &[job_id])
}

/// Sentinel layer used by the canonical pre-embedding checkpoint. Its
/// residual field is derived from `(job_id, token_index)` on chain, so the
/// first registered macro can objectively verify the embedding lookup.
pub const TOKEN_GENESIS_LAYER_V1: u16 = u16::MAX;

pub fn token_input_state_hash_v1(job_id: &Digest, token_index: u32) -> Result<Digest, CommitError> {
    hash_domain(
        TRACE_HASH_ALGORITHM,
        D_TOKEN_INPUT_STATE,
        &[job_id, &token_index.to_le_bytes()],
    )
}

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
    /// Leaves under `checkpoint_root`, including the canonical token genesis.
    pub checkpoint_count: u32,
}

impl TokenResultV1 {
    pub const CANONICAL_SIZE: usize = 76;

    pub fn canonical_bytes(&self) -> [u8; Self::CANONICAL_SIZE] {
        let mut bytes = [0; Self::CANONICAL_SIZE];
        bytes[..32].copy_from_slice(&self.job_id);
        bytes[32..36].copy_from_slice(&self.token_index.to_le_bytes());
        bytes[36..68].copy_from_slice(&self.checkpoint_root);
        bytes[68..72].copy_from_slice(&self.predicted_token.to_le_bytes());
        bytes[72..76].copy_from_slice(&self.checkpoint_count.to_le_bytes());
        bytes
    }

    pub fn hash(&self) -> Result<Digest, CommitError> {
        if self.checkpoint_count < 2 {
            return Err(CommitError::InvalidShape(
                "token result requires genesis plus at least one checkpoint".to_owned(),
            ));
        }
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
    /// Job-tree leaves, including the canonical genesis at index zero.
    pub trace_length: u32,
    pub executed_tokens: u32,
    pub metrics: CommitmentMetrics,
}

/// Compact preimages needed to open one token leaf during a dispute. This is
/// deliberately limited to hashes and indices; activation tensors remain
/// ephemeral and are only recomputed by the two parties for the disputed
/// terminal operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenWitnessV1 {
    pub result: TokenResultV1,
    pub checkpoints: Vec<CheckpointV1>,
}

/// Durable, canonical witness material for a complete execution trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionWitnessV1 {
    pub job_id: Digest,
    pub tokens: Vec<TokenWitnessV1>,
}

impl ExecutionWitnessV1 {
    pub const MAGIC: [u8; 4] = *b"QXW1";
    pub const HEADER_SIZE: usize = 40;

    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CommitError> {
        self.validate()?;
        let capacity = self
            .tokens
            .iter()
            .try_fold(Self::HEADER_SIZE, |size, token| {
                size.checked_add(TokenResultV1::CANONICAL_SIZE)
                    .and_then(|value| {
                        value.checked_add(
                            token
                                .checkpoints
                                .len()
                                .checked_mul(CheckpointV1::CANONICAL_SIZE)?,
                        )
                    })
                    .ok_or(CommitError::ArithmeticOverflow)
            })?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(&Self::MAGIC);
        bytes.extend_from_slice(
            &u32::try_from(self.tokens.len())
                .map_err(|_| CommitError::ArithmeticOverflow)?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&self.job_id);
        for token in &self.tokens {
            bytes.extend_from_slice(&token.result.canonical_bytes());
            for checkpoint in &token.checkpoints {
                bytes.extend_from_slice(&checkpoint.canonical_bytes());
            }
        }
        debug_assert_eq!(bytes.len(), capacity);
        Ok(bytes)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, CommitError> {
        if bytes.len() < Self::HEADER_SIZE || bytes[..4] != Self::MAGIC {
            return Err(CommitError::InvalidShape(
                "invalid execution witness header".to_owned(),
            ));
        }
        let token_count = read_u32(bytes, 4)? as usize;
        let minimum_token_bytes = TokenResultV1::CANONICAL_SIZE
            .checked_add(2 * CheckpointV1::CANONICAL_SIZE)
            .ok_or(CommitError::ArithmeticOverflow)?;
        if token_count == 0
            || token_count
                > bytes
                    .len()
                    .saturating_sub(Self::HEADER_SIZE)
                    .checked_div(minimum_token_bytes)
                    .unwrap_or(0)
        {
            return Err(CommitError::InvalidShape(
                "execution witness token count exceeds its bytes".to_owned(),
            ));
        }
        let mut job_id = [0; 32];
        job_id.copy_from_slice(&bytes[8..40]);
        let mut offset = Self::HEADER_SIZE;
        let mut tokens = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            let end = offset
                .checked_add(TokenResultV1::CANONICAL_SIZE)
                .ok_or(CommitError::ArithmeticOverflow)?;
            let result = decode_token_result(bytes.get(offset..end).ok_or_else(|| {
                CommitError::InvalidShape("truncated execution witness token".to_owned())
            })?)?;
            offset = end;
            let checkpoint_count = result.checkpoint_count as usize;
            if checkpoint_count < 2
                || checkpoint_count
                    > bytes.len().saturating_sub(offset) / CheckpointV1::CANONICAL_SIZE
            {
                return Err(CommitError::InvalidShape(
                    "execution witness checkpoint count exceeds its bytes".to_owned(),
                ));
            }
            let mut checkpoints = Vec::with_capacity(checkpoint_count);
            for _ in 0..checkpoint_count {
                let end = offset
                    .checked_add(CheckpointV1::CANONICAL_SIZE)
                    .ok_or(CommitError::ArithmeticOverflow)?;
                checkpoints.push(decode_checkpoint(bytes.get(offset..end).ok_or_else(
                    || {
                        CommitError::InvalidShape(
                            "truncated execution witness checkpoint".to_owned(),
                        )
                    },
                )?)?);
                offset = end;
            }
            tokens.push(TokenWitnessV1 {
                result,
                checkpoints,
            });
        }
        if offset != bytes.len() {
            return Err(CommitError::InvalidShape(
                "execution witness has trailing bytes".to_owned(),
            ));
        }
        let witness = Self { job_id, tokens };
        witness.validate()?;
        Ok(witness)
    }

    pub fn validate(&self) -> Result<ExecutionTraceV1, CommitError> {
        if self.tokens.is_empty() {
            return Err(CommitError::InvalidExecution(
                "execution witness cannot be empty".to_owned(),
            ));
        }
        let mut job_leaves = Vec::with_capacity(self.tokens.len() + 1);
        job_leaves.push(job_genesis_hash_v1(&self.job_id)?);
        for (token_index, token) in self.tokens.iter().enumerate() {
            if token.result.job_id != self.job_id
                || token.result.token_index as usize != token_index
                || token.result.checkpoint_count as usize != token.checkpoints.len()
                || token.checkpoints.len() < 2
            {
                return Err(CommitError::InvalidExecution(
                    "execution witness token binding is invalid".to_owned(),
                ));
            }
            let mut checkpoint_leaves = Vec::with_capacity(token.checkpoints.len());
            let mut prior_layer = None;
            for (checkpoint_index, checkpoint) in token.checkpoints.iter().enumerate() {
                if checkpoint.job_id != self.job_id
                    || checkpoint.token_index as usize != token_index
                    || (checkpoint_index == 0 && checkpoint.layer_index != TOKEN_GENESIS_LAYER_V1)
                    || (checkpoint_index != 0 && checkpoint.layer_index == TOKEN_GENESIS_LAYER_V1)
                {
                    return Err(CommitError::InvalidExecution(
                        "execution witness checkpoint binding is invalid".to_owned(),
                    ));
                }
                if checkpoint_index != 0 {
                    if prior_layer.is_some_and(|layer| checkpoint.layer_index <= layer) {
                        return Err(CommitError::InvalidExecution(
                            "execution witness checkpoint layers are not increasing".to_owned(),
                        ));
                    }
                    prior_layer = Some(checkpoint.layer_index);
                }
                checkpoint_leaves.push(checkpoint.hash()?);
            }
            let checkpoint_tree =
                MerkleTree::new(TRACE_HASH_ALGORITHM, D_TOKEN, &checkpoint_leaves)?;
            if checkpoint_tree.root() != token.result.checkpoint_root {
                return Err(CommitError::InvalidExecution(
                    "execution witness checkpoint root mismatch".to_owned(),
                ));
            }
            job_leaves.push(token.result.hash()?);
        }
        let job_tree = MerkleTree::new(TRACE_HASH_ALGORITHM, D_JOB, &job_leaves)?;
        let executed_tokens =
            u32::try_from(self.tokens.len()).map_err(|_| CommitError::ArithmeticOverflow)?;
        Ok(ExecutionTraceV1 {
            trace_root: job_tree.root(),
            trace_length: executed_tokens
                .checked_add(1)
                .ok_or(CommitError::ArithmeticOverflow)?,
            executed_tokens,
            metrics: CommitmentMetrics::default(),
        })
    }

    pub fn token_proof(&self, token_index: usize) -> Result<MerkleProof, CommitError> {
        let mut leaves = Vec::with_capacity(self.tokens.len() + 1);
        leaves.push(job_genesis_hash_v1(&self.job_id)?);
        leaves.extend(
            self.tokens
                .iter()
                .map(|token| token.result.hash())
                .collect::<Result<Vec<_>, _>>()?,
        );
        MerkleTree::new(TRACE_HASH_ALGORITHM, D_JOB, &leaves)?.proof(token_index + 1)
    }

    pub fn checkpoint_proof(
        &self,
        token_index: usize,
        checkpoint_index: usize,
    ) -> Result<MerkleProof, CommitError> {
        let token = self
            .tokens
            .get(token_index)
            .ok_or(CommitError::InvalidIndex {
                index: token_index,
                length: self.tokens.len(),
            })?;
        let leaves = token
            .checkpoints
            .iter()
            .map(CheckpointV1::hash)
            .collect::<Result<Vec<_>, _>>()?;
        MerkleTree::new(TRACE_HASH_ALGORITHM, D_TOKEN, &leaves)?.proof(checkpoint_index)
    }
}

impl ExecutionTraceV1 {
    /// Construct the unsigned result committed under the miner's epoch root.
    /// Keeping this beside the committer prevents callers from accidentally
    /// publishing a fixed-capacity root or a mismatched trace length.
    pub fn result_leaf(
        &self,
        job_hash: Digest,
        output_tokens: &[u32],
    ) -> Result<crate::ResultLeafV1, CommitError> {
        let result = crate::ResultLeafV1 {
            job_hash,
            output_hash: crate::output_hash_v1(output_tokens)?,
            trace_root: self.trace_root,
            output_token_count: u32::try_from(output_tokens.len())
                .map_err(|_| CommitError::ArithmeticOverflow)?,
            trace_length: self.trace_length,
        };
        result.hash()?;
        Ok(result)
    }
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
    token_checkpoint_preimages: Vec<CheckpointV1>,
    witness_tokens: Vec<TokenWitnessV1>,
    pending_token_root: Option<(Digest, u32)>,
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
            .checked_add(1)
            .ok_or(CommitError::ArithmeticOverflow)?
            .checked_next_power_of_two()
            .ok_or(CommitError::ArithmeticOverflow)?;
        let context_depth = context_capacity.trailing_zeros() as usize;
        let kv_trees = (0..layers)
            .map(|_| FixedMerkleAccumulator::new(TRACE_HASH_ALGORITHM, D_KV, context_depth))
            .collect::<Result<Vec<_>, _>>()?;
        let mut job_tree = FixedMerkleAccumulator::new(TRACE_HASH_ALGORITHM, D_JOB, context_depth)?;
        job_tree.append(job_genesis_hash_v1(&job_id)?)?;
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
            token_checkpoint_preimages: Vec::with_capacity(layers),
            witness_tokens: Vec::new(),
            pending_token_root: None,
            job_tree,
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

    pub fn begin_token(&mut self, token_index: u32) -> Result<Digest, CommitError> {
        self.begin_token_hash(token_index)
    }

    pub fn begin_token_hash(&mut self, token_index: u32) -> Result<Digest, CommitError> {
        if token_index != self.expected_token
            || self.expected_layer != 0
            || !self.token_checkpoints.is_empty()
            || self.pending_token_root.is_some()
            || token_index as usize >= self.max_context
        {
            return Err(CommitError::InvalidExecution(
                "token genesis is out of sequence or empty".to_owned(),
            ));
        }
        let checkpoint = CheckpointV1 {
            job_id: self.job_id,
            token_index,
            layer_index: TOKEN_GENESIS_LAYER_V1,
            residual_hash: token_input_state_hash_v1(&self.job_id, token_index)?,
            global_kv_root: self.global_kv.root(),
        };
        let checkpoint_hash = checkpoint.hash()?;
        self.token_checkpoints.push(checkpoint_hash);
        self.token_checkpoint_preimages.push(checkpoint);
        Ok(checkpoint_hash)
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
        if self.expected_layer == 0 && self.token_checkpoints.is_empty() {
            return Err(CommitError::InvalidExecution(
                "begin_token must commit the token-start sentinel first".to_owned(),
            ));
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
            };
            let checkpoint_hash = checkpoint.hash()?;
            self.token_checkpoints.push(checkpoint_hash);
            self.token_checkpoint_preimages.push(checkpoint);
            Some(checkpoint_hash)
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
            self.pending_token_root = Some((
                token_tree.root(),
                u32::try_from(self.token_checkpoints.len())
                    .map_err(|_| CommitError::ArithmeticOverflow)?,
            ));
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
        let (checkpoint_root, checkpoint_count) =
            self.pending_token_root.take().ok_or_else(|| {
                CommitError::InvalidExecution("token layers are not complete".to_owned())
            })?;
        let result = TokenResultV1 {
            job_id: self.job_id,
            token_index,
            checkpoint_root,
            predicted_token,
            checkpoint_count,
        };
        let leaf = result.hash()?;
        self.job_tree.append(leaf)?;
        self.witness_tokens.push(TokenWitnessV1 {
            result,
            checkpoints: std::mem::take(&mut self.token_checkpoint_preimages),
        });
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
        if self.expected_token == 0 {
            return Err(CommitError::InvalidExecution(
                "cannot finish an empty execution".to_owned(),
            ));
        }
        Ok(ExecutionTraceV1 {
            trace_root: self.job_tree.compact_root()?,
            trace_length: self
                .expected_token
                .checked_add(1)
                .ok_or(CommitError::ArithmeticOverflow)?,
            executed_tokens: self.expected_token,
            metrics: self.metrics,
        })
    }

    pub fn witness(&self) -> Result<ExecutionWitnessV1, CommitError> {
        self.finish()?;
        let witness = ExecutionWitnessV1 {
            job_id: self.job_id,
            tokens: self.witness_tokens.clone(),
        };
        let trace = witness.validate()?;
        if trace.trace_root != self.job_tree.compact_root()? {
            return Err(CommitError::InvalidExecution(
                "retained witness does not match streaming trace root".to_owned(),
            ));
        }
        Ok(witness)
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, CommitError> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| CommitError::InvalidShape("truncated u32".to_owned()))?;
    Ok(u32::from_le_bytes(value.try_into().map_err(|_| {
        CommitError::InvalidShape("invalid u32".to_owned())
    })?))
}

fn decode_token_result(bytes: &[u8]) -> Result<TokenResultV1, CommitError> {
    let mut job_id = [0; 32];
    job_id.copy_from_slice(&bytes[..32]);
    let mut checkpoint_root = [0; 32];
    checkpoint_root.copy_from_slice(&bytes[36..68]);
    Ok(TokenResultV1 {
        job_id,
        token_index: read_u32(bytes, 32)?,
        checkpoint_root,
        predicted_token: read_u32(bytes, 68)?,
        checkpoint_count: read_u32(bytes, 72)?,
    })
}

fn decode_checkpoint(bytes: &[u8]) -> Result<CheckpointV1, CommitError> {
    let mut job_id = [0; 32];
    job_id.copy_from_slice(&bytes[..32]);
    let mut residual_hash = [0; 32];
    residual_hash.copy_from_slice(&bytes[38..70]);
    let mut global_kv_root = [0; 32];
    global_kv_root.copy_from_slice(&bytes[70..102]);
    Ok(CheckpointV1 {
        job_id,
        token_index: read_u32(bytes, 32)?,
        layer_index: u16::from_le_bytes([bytes[36], bytes[37]]),
        residual_hash,
        global_kv_root,
    })
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
                state.begin_token(token).unwrap();
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
        assert_eq!(run(false).trace_length, 3);
        assert_eq!(run(false).metrics.canonical_state_bytes, 192);
    }

    #[test]
    fn compact_witness_round_trips_and_opens_every_leaf() {
        let mut state =
            ExecutionCommitterV1::new_with_checkpoint_interval([3; 32], 4, 8, 2).unwrap();
        for token_index in 0..3 {
            state.begin_token(token_index).unwrap();
            for layer_index in 0..4 {
                state
                    .commit_layer_hashes(
                        token_index,
                        layer_index,
                        [layer_index as u8; 32],
                        [token_index as u8; 32],
                        64,
                    )
                    .unwrap();
            }
            state
                .commit_output_token(token_index, 100 + token_index)
                .unwrap();
        }
        let trace = state.finish().unwrap();
        let witness = state.witness().unwrap();
        let encoded = witness.canonical_bytes().unwrap();
        let decoded = ExecutionWitnessV1::from_canonical_bytes(&encoded).unwrap();
        assert_eq!(decoded, witness);
        assert_eq!(decoded.validate().unwrap().trace_root, trace.trace_root);
        for (token_index, token) in decoded.tokens.iter().enumerate() {
            let proof = decoded.token_proof(token_index).unwrap();
            assert!(
                proof
                    .verify(
                        TRACE_HASH_ALGORITHM,
                        D_JOB,
                        token.result.hash().unwrap(),
                        trace.trace_root
                    )
                    .unwrap()
            );
            for (checkpoint_index, checkpoint) in token.checkpoints.iter().enumerate() {
                let proof = decoded
                    .checkpoint_proof(token_index, checkpoint_index)
                    .unwrap();
                assert!(
                    proof
                        .verify(
                            TRACE_HASH_ALGORITHM,
                            D_TOKEN,
                            checkpoint.hash().unwrap(),
                            token.result.checkpoint_root,
                        )
                        .unwrap()
                );
            }
        }
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(ExecutionWitnessV1::from_canonical_bytes(&corrupt).is_err());
    }

    #[test]
    fn malformed_witness_corpus_is_bounded_and_rejected() {
        let corpus = [
            Vec::new(),
            b"QXW1".to_vec(),
            {
                let mut bytes = vec![0; ExecutionWitnessV1::HEADER_SIZE];
                bytes[..4].copy_from_slice(b"QXW1");
                bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
                bytes
            },
            {
                let mut bytes = vec![0; ExecutionWitnessV1::HEADER_SIZE + 1];
                bytes[..4].copy_from_slice(b"QXW1");
                bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
                bytes
            },
        ];
        for bytes in corpus {
            assert!(ExecutionWitnessV1::from_canonical_bytes(&bytes).is_err());
        }
    }

    #[test]
    fn randomized_committer_sequences_fail_closed() {
        let mut seed = 0x9e37_79b9_u32;
        for _ in 0..512 {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let token = seed & 3;
            let layer = ((seed >> 8) & 3) as u16;
            let mut state = ExecutionCommitterV1::new([4; 32], 4, 8).unwrap();
            if token != 0 || layer != 0 {
                assert!(
                    state
                        .commit_layer_hashes(token, layer, [1; 32], [2; 32], 64)
                        .is_err()
                );
            }
            state.begin_token(0).unwrap();
            if layer != 0 {
                assert!(
                    state
                        .commit_layer_hashes(0, layer, [1; 32], [2; 32], 64)
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn coarser_checkpoints_keep_every_kv_leaf_and_reduce_residual_hashing() {
        let mut state =
            ExecutionCommitterV1::new_with_checkpoint_interval([4; 32], 4, 8, 2).unwrap();
        state.begin_token(0).unwrap();
        for layer in 0..4 {
            state
                .commit_layer(0, layer, &[layer as f32; 4], &[1.0, 2.0], &[3.0, 4.0])
                .unwrap();
        }
        state.commit_output_token(0, 17).unwrap();
        let trace = state.finish().unwrap();
        assert_eq!(trace.metrics.checkpoint_count, 2);
        assert_eq!(trace.trace_length, 2);
        // Four 16-byte K/V pairs and two 16-byte retained residuals.
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
            state.begin_token(0).unwrap();
            state.commit_layer(0, 0, &[1.0], &[2.0], &[3.0]).unwrap();
            state.commit_output_token(0, predicted_token).unwrap();
            state.finish().unwrap().trace_root
        }
        assert_ne!(root(11), root(12));
    }

    #[test]
    fn accelerator_digest_path_uses_the_same_genesis_leaves() {
        let job_id = [7; 32];
        let mut cpu = ExecutionCommitterV1::new(job_id, 1, 2).unwrap();
        cpu.begin_token(0).unwrap();
        cpu.commit_layer(0, 0, &[1.0], &[2.0], &[3.0]).unwrap();
        cpu.commit_output_token(0, 9).unwrap();

        let residual_hash = state_hash_v1(&1.0_f32.to_bits().to_le_bytes()).unwrap();
        let kv_leaf = kv_leaf_hash_v1(
            0,
            0,
            &2.0_f32.to_bits().to_le_bytes(),
            &3.0_f32.to_bits().to_le_bytes(),
        )
        .unwrap();
        let mut accelerator = ExecutionCommitterV1::new(job_id, 1, 2).unwrap();
        accelerator.begin_token_hash(0).unwrap();
        accelerator
            .commit_layer_hashes(0, 0, residual_hash, kv_leaf, 12)
            .unwrap();
        accelerator.commit_output_token(0, 9).unwrap();

        assert_eq!(cpu.finish().unwrap(), accelerator.finish().unwrap());
    }

    #[test]
    fn execution_trace_builds_a_shape_consistent_result_leaf() {
        let mut state = ExecutionCommitterV1::new([8; 32], 1, 4).unwrap();
        state.begin_token(0).unwrap();
        state.commit_layer(0, 0, &[1.0], &[2.0], &[3.0]).unwrap();
        state.commit_output_token(0, 9).unwrap();
        let trace = state.finish().unwrap();
        let result = trace.result_leaf([5; 32], &[9]).unwrap();
        assert_eq!(result.trace_root, trace.trace_root);
        assert_eq!(result.trace_length, 2);
        assert_eq!(result.output_token_count, 1);
        assert!(trace.result_leaf([5; 32], &[]).is_err());
    }

    #[test]
    fn token_result_rejects_an_unprovable_checkpoint_shape() {
        let mut result = TokenResultV1 {
            job_id: [1; 32],
            token_index: 0,
            checkpoint_root: [2; 32],
            predicted_token: 3,
            checkpoint_count: 1,
        };
        assert!(result.hash().is_err());
        result.checkpoint_count = 2;
        result.hash().unwrap();
    }
}
