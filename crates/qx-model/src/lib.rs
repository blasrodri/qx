//! Model topology descriptors used at the QX compiler/artifact boundary.
//!
//! Runtime V2 is still deliberately specialized for Granite.  These types
//! make the next specialization (Qwen3-30B-A3B) explicit and auditable instead
//! of scattering another set of shape constants through the compiler.

use serde::{Deserialize, Serialize};

pub const HIDDEN: usize = 1024;
pub const LAYERS: usize = 24;
pub const ATTENTION_HEADS: usize = 16;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 64;
pub const EXPERTS: usize = 32;
pub const TOP_K: usize = 8;
pub const EXPERT_INTERMEDIATE: usize = 512;
pub const MAX_CONTEXT: usize = 4096;
pub const VOCAB: usize = 49_155;

pub const ATTENTION_MULTIPLIER: f32 = 0.015625;
pub const EMBEDDING_MULTIPLIER: f32 = 12.0;
pub const RESIDUAL_MULTIPLIER: f32 = 0.22;
pub const LOGITS_SCALING: f32 = 6.0;
pub const RMS_EPSILON: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10_000.0;

pub const QWEN3_30B_A3B_MODEL_ID: &str = "Qwen/Qwen3-30B-A3B";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    GraniteMoe,
    Qwen3Moe,
}

/// Artifact-sealed semantic model shape.
///
/// Values which alter logits or persistent-state sizing belong here.  Kernel
/// packet widths, command-buffer depth, and quantization layouts are target
/// specializations and deliberately remain in the execution plan.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ModelShape {
    pub architecture: ModelArchitecture,
    pub layers: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub active_experts: usize,
    pub expert_intermediate: usize,
    pub dense_intermediate: usize,
    pub vocabulary: usize,
    pub max_context: usize,
    pub rms_epsilon: f32,
    pub rope_theta: f32,
    pub normalize_router_weights: bool,
    pub tied_word_embeddings: bool,
    pub query_key_norm: bool,
}

impl ModelShape {
    pub const fn granite_v0() -> Self {
        Self {
            architecture: ModelArchitecture::GraniteMoe,
            layers: LAYERS,
            hidden_size: HIDDEN,
            attention_heads: ATTENTION_HEADS,
            key_value_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            experts: EXPERTS,
            active_experts: TOP_K,
            expert_intermediate: EXPERT_INTERMEDIATE,
            dense_intermediate: EXPERT_INTERMEDIATE,
            vocabulary: VOCAB,
            max_context: MAX_CONTEXT,
            rms_epsilon: RMS_EPSILON,
            rope_theta: ROPE_THETA,
            normalize_router_weights: true,
            tied_word_embeddings: true,
            query_key_norm: false,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let positive = [
            ("layers", self.layers),
            ("hidden_size", self.hidden_size),
            ("attention_heads", self.attention_heads),
            ("key_value_heads", self.key_value_heads),
            ("head_dim", self.head_dim),
            ("experts", self.experts),
            ("active_experts", self.active_experts),
            ("expert_intermediate", self.expert_intermediate),
            ("vocabulary", self.vocabulary),
            ("max_context", self.max_context),
        ];
        let mut errors = positive
            .into_iter()
            .filter_map(|(name, value)| (value == 0).then_some(format!("{name} is zero")))
            .collect::<Vec<_>>();
        if self.active_experts > self.experts {
            errors.push("active_experts exceeds experts".to_owned());
        }
        if self.attention_heads % self.key_value_heads != 0 {
            errors.push("attention_heads is not divisible by key_value_heads".to_owned());
        }
        // Do not require hidden_size == attention_heads * head_dim. Qwen3
        // deliberately expands its query projection from 2,048 to 4,096.
        if !self.rms_epsilon.is_finite() || self.rms_epsilon <= 0.0 {
            errors.push("rms_epsilon is not finite and positive".to_owned());
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            errors.push("rope_theta is not finite and positive".to_owned());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join(", "))
        }
    }

    pub const fn expert_weight_elements(&self) -> usize {
        // SwiGLU gate + up + down.
        self.hidden_size * self.expert_intermediate * 3
    }

    pub const fn kv_elements_per_token(&self) -> usize {
        self.layers * self.key_value_heads * self.head_dim * 2
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Granite3A400M;

impl Granite3A400M {
    pub const MODEL_ID: &'static str = "ibm-granite-3.0-1b-a400m-instruct";

    pub const fn expert_input_elements() -> usize {
        HIDDEN * (EXPERT_INTERMEDIATE * 2)
    }

    pub const fn expert_output_elements() -> usize {
        HIDDEN * EXPERT_INTERMEDIATE
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TargetProfile {
    pub architecture: AppleGpuArchitecture,
    pub os_version: String,
    pub physical_memory: u64,
    pub max_workgroup_size: usize,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum AppleGpuArchitecture {
    AppleSilicon,
    A19,
    A19Pro,
    Unknown,
}

impl std::fmt::Display for AppleGpuArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::AppleSilicon => "apple-silicon",
            Self::A19 => "apple-a19",
            Self::A19Pro => "apple-a19-pro",
            Self::Unknown => "unknown",
        };
        f.write_str(name)
    }
}

/// The subset of a Hugging Face config we validate for the V0 inspector.
#[derive(Debug, Deserialize)]
pub struct GraniteConfig {
    pub model_type: Option<String>,
    pub num_hidden_layers: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub num_local_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub vocab_size: Option<usize>,
}

impl GraniteConfig {
    pub fn validate_v0(&self) -> Result<(), String> {
        let checks = [
            (
                "model_type",
                matches!(self.model_type.as_deref(), Some("granite" | "granitemoe")),
            ),
            ("num_hidden_layers", self.num_hidden_layers == Some(LAYERS)),
            ("hidden_size", self.hidden_size == Some(HIDDEN)),
            (
                "num_attention_heads",
                self.num_attention_heads == Some(ATTENTION_HEADS),
            ),
            (
                "num_key_value_heads",
                self.num_key_value_heads == Some(KV_HEADS),
            ),
            ("num_local_experts", self.num_local_experts == Some(EXPERTS)),
            (
                "num_experts_per_tok",
                self.num_experts_per_tok == Some(TOP_K),
            ),
            (
                "intermediate_size",
                self.intermediate_size == Some(EXPERT_INTERMEDIATE),
            ),
        ];
        let mismatches = checks
            .iter()
            .filter_map(|(name, ok)| (!ok).then_some(*name))
            .collect::<Vec<_>>();
        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "config is not Granite V0: {} mismatch",
                mismatches.join(", ")
            ))
        }
    }
}

/// Hugging Face Qwen3 MoE configuration fields which determine execution
/// semantics or artifact capacity.
#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3MoeConfig {
    pub model_type: Option<String>,
    pub num_hidden_layers: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub moe_intermediate_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub vocab_size: Option<usize>,
    pub rms_norm_eps: Option<f32>,
    pub rope_theta: Option<f32>,
    pub norm_topk_prob: Option<bool>,
    pub tie_word_embeddings: Option<bool>,
}

impl Qwen3MoeConfig {
    pub fn model_shape(&self) -> Result<ModelShape, String> {
        if self.model_type.as_deref() != Some("qwen3_moe") {
            return Err("model_type is not qwen3_moe".to_owned());
        }
        let required =
            |name: &str, value: Option<usize>| value.ok_or_else(|| format!("missing {name}"));
        let shape = ModelShape {
            architecture: ModelArchitecture::Qwen3Moe,
            layers: required("num_hidden_layers", self.num_hidden_layers)?,
            hidden_size: required("hidden_size", self.hidden_size)?,
            attention_heads: required("num_attention_heads", self.num_attention_heads)?,
            key_value_heads: required("num_key_value_heads", self.num_key_value_heads)?,
            head_dim: required("head_dim", self.head_dim)?,
            experts: required("num_experts", self.num_experts)?,
            active_experts: required("num_experts_per_tok", self.num_experts_per_tok)?,
            expert_intermediate: required("moe_intermediate_size", self.moe_intermediate_size)?,
            dense_intermediate: required("intermediate_size", self.intermediate_size)?,
            vocabulary: required("vocab_size", self.vocab_size)?,
            max_context: required("max_position_embeddings", self.max_position_embeddings)?,
            rms_epsilon: self
                .rms_norm_eps
                .ok_or_else(|| "missing rms_norm_eps".to_owned())?,
            rope_theta: self
                .rope_theta
                .ok_or_else(|| "missing rope_theta".to_owned())?,
            normalize_router_weights: self.norm_topk_prob.unwrap_or(false),
            tied_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
            query_key_norm: true,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn validate_30b_a3b(&self) -> Result<ModelShape, String> {
        let shape = self.model_shape()?;
        let expected = ModelShape {
            architecture: ModelArchitecture::Qwen3Moe,
            layers: 48,
            hidden_size: 2_048,
            attention_heads: 32,
            key_value_heads: 4,
            head_dim: 128,
            experts: 128,
            active_experts: 8,
            expert_intermediate: 768,
            dense_intermediate: 6_144,
            vocabulary: 151_936,
            max_context: 40_960,
            rms_epsilon: 1e-6,
            rope_theta: 1_000_000.0,
            normalize_router_weights: true,
            tied_word_embeddings: false,
            query_key_norm: true,
        };
        if shape == expected {
            Ok(shape)
        } else {
            Err(format!(
                "config is not the pinned Qwen3-30B-A3B shape: got {shape:?}"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pinned_qwen3_30b_a3b_shape() {
        let config: Qwen3MoeConfig = serde_json::from_str(
            r#"{
                "model_type":"qwen3_moe",
                "num_hidden_layers":48,
                "hidden_size":2048,
                "num_attention_heads":32,
                "num_key_value_heads":4,
                "head_dim":128,
                "num_experts":128,
                "num_experts_per_tok":8,
                "moe_intermediate_size":768,
                "intermediate_size":6144,
                "max_position_embeddings":40960,
                "vocab_size":151936,
                "rms_norm_eps":0.000001,
                "rope_theta":1000000.0,
                "norm_topk_prob":true,
                "tie_word_embeddings":false
            }"#,
        )
        .unwrap();
        let shape = config.validate_30b_a3b().unwrap();
        assert_eq!(shape.expert_weight_elements(), 4_718_592);
        assert_eq!(shape.kv_elements_per_token(), 49_152);
    }

    #[test]
    fn rejects_semantically_invalid_shape() {
        let mut shape = ModelShape::granite_v0();
        shape.key_value_heads = 3;
        assert!(shape.validate().unwrap_err().contains("not divisible"));
    }
}
