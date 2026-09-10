use crate::{
    CommitError, D_ARCHITECTURE, D_EXECUTION_SPEC, D_MODEL, Digest, HashAlgorithm, ProtocolVersion,
    hash_domain,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelSpecV1 {
    pub model_version: u32,
    pub architecture_root: Digest,
    pub weights_root: Digest,
    pub num_layers: u16,
    pub hidden_size: u16,
    pub num_attention_heads: u16,
    pub num_kv_heads: u16,
    pub num_experts: u16,
    pub experts_per_token: u16,
    pub vocab_size: u32,
    pub max_context_tokens: u32,
    pub quantization_spec: Digest,
    pub execution_spec: Digest,
    pub tokenizer_hash: Digest,
}

impl ModelSpecV1 {
    pub const CANONICAL_SIZE: usize = 184;

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::CANONICAL_SIZE);
        bytes.extend_from_slice(&self.model_version.to_le_bytes());
        bytes.extend_from_slice(&self.architecture_root);
        bytes.extend_from_slice(&self.weights_root);
        bytes.extend_from_slice(&self.num_layers.to_le_bytes());
        bytes.extend_from_slice(&self.hidden_size.to_le_bytes());
        bytes.extend_from_slice(&self.num_attention_heads.to_le_bytes());
        bytes.extend_from_slice(&self.num_kv_heads.to_le_bytes());
        bytes.extend_from_slice(&self.num_experts.to_le_bytes());
        bytes.extend_from_slice(&self.experts_per_token.to_le_bytes());
        bytes.extend_from_slice(&self.vocab_size.to_le_bytes());
        bytes.extend_from_slice(&self.max_context_tokens.to_le_bytes());
        bytes.extend_from_slice(&self.quantization_spec);
        bytes.extend_from_slice(&self.execution_spec);
        bytes.extend_from_slice(&self.tokenizer_hash);
        debug_assert_eq!(bytes.len(), Self::CANONICAL_SIZE);
        bytes
    }

    pub fn model_id(&self, protocol: ProtocolVersion) -> Result<Digest, CommitError> {
        hash_domain(
            protocol.hash_algorithm(),
            D_MODEL,
            &[&self.canonical_bytes()],
        )
    }

    pub fn validate_granite_v1(&self) -> Result<(), CommitError> {
        let shape = qx_model::ModelShape::granite_v0();
        let expected = [
            ("layers", usize::from(self.num_layers), shape.layers),
            (
                "hidden_size",
                usize::from(self.hidden_size),
                shape.hidden_size,
            ),
            (
                "attention_heads",
                usize::from(self.num_attention_heads),
                shape.attention_heads,
            ),
            (
                "key_value_heads",
                usize::from(self.num_kv_heads),
                shape.key_value_heads,
            ),
            ("experts", usize::from(self.num_experts), shape.experts),
            (
                "active_experts",
                usize::from(self.experts_per_token),
                shape.active_experts,
            ),
            ("vocabulary", self.vocab_size as usize, shape.vocabulary),
            (
                "max_context",
                self.max_context_tokens as usize,
                shape.max_context,
            ),
        ];
        let mismatches = expected
            .into_iter()
            .filter_map(|(name, actual, expected)| (actual != expected).then_some(name))
            .collect::<Vec<_>>();
        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(CommitError::InvalidShape(mismatches.join(", ")))
        }
    }
}

pub fn granite_architecture_root() -> Result<Digest, CommitError> {
    let shape = qx_model::ModelShape::granite_v0();
    let mut bytes = Vec::with_capacity(80);
    bytes.push(1); // GraniteMoe
    for value in [
        shape.layers,
        shape.hidden_size,
        shape.attention_heads,
        shape.key_value_heads,
        shape.head_dim,
        shape.experts,
        shape.active_experts,
        shape.expert_intermediate,
        shape.dense_intermediate,
        shape.vocabulary,
        shape.max_context,
    ] {
        bytes.extend_from_slice(
            &u32::try_from(value)
                .map_err(|_| CommitError::ArithmeticOverflow)?
                .to_le_bytes(),
        );
    }
    bytes.extend_from_slice(&shape.rms_epsilon.to_bits().to_le_bytes());
    bytes.extend_from_slice(&shape.rope_theta.to_bits().to_le_bytes());
    bytes.extend_from_slice(&[
        shape.normalize_router_weights as u8,
        shape.tied_word_embeddings as u8,
        shape.query_key_norm as u8,
    ]);
    hash_domain(HashAlgorithm::Sha256V1, D_ARCHITECTURE, &[&bytes])
}

/// Digest of the pinned CPU-oracle semantics used by Phase 3.
pub fn granite_execution_spec() -> Result<Digest, CommitError> {
    granite_execution_spec_with_checkpoint_interval(1)
}

/// V1 defines every-layer checkpoints implicitly. Supported coarser profiles
/// append their interval so each receives a distinct execution-spec digest.
pub fn granite_execution_spec_with_checkpoint_interval(
    checkpoint_interval: u8,
) -> Result<Digest, CommitError> {
    if !matches!(checkpoint_interval, 1 | 2 | 4) {
        return Err(CommitError::InvalidShape(
            "Granite checkpoint interval must be 1, 2, or 4".to_owned(),
        ));
    }
    let mut bytes = Vec::with_capacity(64);
    bytes.extend_from_slice(b"QX_EXEC_GRANITE_CPU_V1");
    bytes.push(1); // activation FP32
    bytes.push(1); // accumulator FP32
    bytes.push(1); // IEEE round-to-nearest-ties-even at f32 operations
    bytes.push(1); // grouped symmetric Q4 decoder
    bytes.extend_from_slice(&32_u16.to_le_bytes());
    bytes.extend_from_slice(&qx_model::RMS_EPSILON.to_bits().to_le_bytes());
    bytes.extend_from_slice(&qx_model::ROPE_THETA.to_bits().to_le_bytes());
    bytes.push(1); // normalized top-k softmax
    bytes.push(1); // top-k ties: lowest expert id
    bytes.push(1); // greedy decoding
    bytes.push(1); // argmax ties: lowest token id
    bytes.push(1); // total_cmp ordering for exceptional values
    bytes.push(HashAlgorithm::Blake3V1 as u8); // residual and new-K/V leaf hash
    bytes.extend_from_slice(&256_u16.to_le_bytes()); // state hash chunk bytes
    bytes.push(1); // raw BLAKE3(left || right) state-tree parent
    bytes.push(HashAlgorithm::Blake3V1 as u8); // hot KV/checkpoint/token/job trees
    if checkpoint_interval != 1 {
        bytes.extend_from_slice(b"CHECKPOINT_INTERVAL");
        bytes.push(checkpoint_interval);
    }
    hash_domain(HashAlgorithm::Sha256V1, D_EXECUTION_SPEC, &[&bytes])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_encoding_is_fixed_width() {
        let spec = ModelSpecV1 {
            model_version: 1,
            architecture_root: [1; 32],
            weights_root: [2; 32],
            num_layers: 24,
            hidden_size: 1024,
            num_attention_heads: 16,
            num_kv_heads: 8,
            num_experts: 32,
            experts_per_token: 8,
            vocab_size: 49_155,
            max_context_tokens: 4096,
            quantization_spec: [3; 32],
            execution_spec: [4; 32],
            tokenizer_hash: [5; 32],
        };
        assert_eq!(spec.canonical_bytes().len(), ModelSpecV1::CANONICAL_SIZE);
        spec.validate_granite_v1().unwrap();
    }
}
