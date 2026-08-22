//! The deliberately narrow Granite 3.0 1B-A400M model description used by QX V0.

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
