//! Metal boundary for the specialized batch-1 expert path.
//!
//! The kernels are kept as source assets so the compiler can later embed a
//! target-specific metallib. The Rust context is intentionally small: model
//! orchestration remains outside Metal, while the hot GEMV kernels consume
//! packed weights directly and never allocate a dequantized matrix.

use qx_quant::{PackingLayout, QuantType};

pub const Q2_KERNEL_SOURCE: &str = include_str!("../../../kernels/q2_expert.metal");
pub const Q3_KERNEL_SOURCE: &str = include_str!("../../../kernels/q3_expert.metal");
pub const Q4_KERNEL_SOURCE: &str = include_str!("../../../kernels/q4_expert.metal");
pub const LOGITS_KERNEL_SOURCE: &str = include_str!("../../../kernels/logits.metal");
pub const DENSE_KERNEL_SOURCE: &str = include_str!("../../../kernels/dense.metal");
pub const F16_DENSE_KERNEL_SOURCE: &str = include_str!("../../../kernels/f16_dense.metal");
pub const ARGMAX_KERNEL_SOURCE: &str = include_str!("../../../kernels/argmax.metal");
pub const V2_KERNEL_SOURCE: &str = include_str!("../../../kernels/v2_decode.metal");
pub const Q4_DENSE_KERNEL_SOURCE: &str = include_str!("../../../kernels/q4_dense.metal");
pub const DYNAMIC_MOE_KERNEL_SOURCE: &str = include_str!("../../../kernels/dynamic_moe.metal");
pub const Q4_EMBEDDING_KERNEL_SOURCE: &str = include_str!("../../../kernels/q4_embedding.metal");

pub fn kernel_source(quant_type: QuantType) -> &'static str {
    match quant_type {
        QuantType::Q2 => Q2_KERNEL_SOURCE,
        QuantType::Q3 => Q3_KERNEL_SOURCE,
        QuantType::Q4 => Q4_KERNEL_SOURCE,
        QuantType::Q4K => Q4_KERNEL_SOURCE,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GemvGeometry {
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
}

impl GemvGeometry {
    pub const fn granite_gate_up() -> Self {
        Self {
            rows: 1024,
            cols: 1024,
            group_size: 32,
        }
    }

    pub const fn granite_down() -> Self {
        Self {
            rows: 1024,
            cols: 512,
            group_size: 32,
        }
    }
}

#[cfg(target_os = "macos")]
pub mod platform {
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    use block2::RcBlock;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{NSRange, NSString};
    pub use objc2_metal::MTLSize;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLComputePipelineState,
        MTLCounterResultTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
        MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice,
        MTLLibrary, MTLResourceOptions, MTLStorageMode,
    };

    use super::{
        ARGMAX_KERNEL_SOURCE, DENSE_KERNEL_SOURCE, DYNAMIC_MOE_KERNEL_SOURCE,
        F16_DENSE_KERNEL_SOURCE, LOGITS_KERNEL_SOURCE, PackingLayout, Q4_DENSE_KERNEL_SOURCE,
        Q4_EMBEDDING_KERNEL_SOURCE, QuantType, V2_KERNEL_SOURCE, kernel_source,
    };

    const HIDDEN: usize = 1024;
    const INTERMEDIATE: usize = 512;
    const GROUP_SIZE: usize = 32;

    /// Select the validated three-dispatch attention topology.
    ///
    /// The topology is enabled by default after its full sequential parity
    /// gate. Set `QX_V2_ATTENTION_3_STAGE=0` (or `false`/`no`/`off`) to run
    /// the older one-dispatch reference path for A/B comparisons. It is read
    /// once per process so a token graph cannot change topology halfway
    /// through a run.
    pub fn three_stage_attention_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_ATTENTION_3_STAGE")
                .map(|value| {
                    !matches!(
                        value.to_ascii_lowercase().as_str(),
                        "0" | "false" | "no" | "off"
                    )
                })
                .unwrap_or(true)
        })
    }

    /// Validated V accumulation topology. It splits each GQA KV head into
    /// context chunks, computes paired Q-head partials, then reduces them.
    /// Set `QX_V2_ATTENTION_V_SIMD=0` to use the rollback path.
    pub fn attention_v_simd_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_ATTENTION_V_SIMD")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(true)
        })
    }

    /// Validated hierarchical softmax. The max pass is chunked across
    /// SIMD groups; the normalize pass performs the global max/sum reduction
    /// and writes probabilities with regular parallelism. Set
    /// `QX_V2_ATTENTION_SOFTMAX_SIMD=0` to use the rollback path.
    pub fn attention_softmax_simd_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_ATTENTION_SOFTMAX_SIMD")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(true)
        })
    }

    /// Experimental qlen=1 FlashAttention-style streaming path. Disabled by
    /// default until its long sequential parity gate passes.
    pub fn attention_stream_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_ATTENTION_STREAM")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    }

    /// Streaming attention geometry: 32 gives one SIMD group per workgroup;
    /// 256 puts all eight KV-head groups in one workgroup.
    pub fn attention_stream_threadgroup_size() -> usize {
        if std::env::var("QX_V2_ATTENTION_STREAM_TG").ok().as_deref() == Some("32") {
            32
        } else {
            256
        }
    }

    /// Experimental chunk-parallel online-softmax attention. Disabled by
    /// default; this preserves context parallelism while avoiding score data.
    pub fn attention_chunked_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_ATTENTION_CHUNKED")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    }

    /// Experimental topology B: compute only inverse RMS before QKV, then
    /// let the SIMD QKV consumer apply the norm weights while reading input.
    pub fn rms_scalar_qkv_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_RMS_SCALAR_QKV")
                .map(|value| {
                    !matches!(
                        value.to_ascii_lowercase().as_str(),
                        "0" | "false" | "no" | "off"
                    )
                })
                .unwrap_or(true)
        })
    }

    /// Validated symmetric topology for the post-attention RMS→router and
    /// expert gate/up boundary. Set `QX_V2_RMS_SCALAR_ROUTER=0` (or
    /// `false`/`no`/`off`) to run the materialized-vector rollback path.
    pub fn rms_scalar_router_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_RMS_SCALAR_ROUTER")
                .map(|value| {
                    !matches!(
                        value.to_ascii_lowercase().as_str(),
                        "0" | "false" | "no" | "off"
                    )
                })
                .unwrap_or(true)
        })
    }

    /// Optional final LM-head geometry candidate. The value is parsed once
    /// per process as `rows_per_simd:simd_groups:cache_input`, for example
    /// `2:8:0`. It is intentionally an experiment knob: the planner can
    /// later materialize the selected constants in an artifact plan.
    pub fn final_head_sweep_geometry() -> Option<(usize, usize, bool)> {
        static GEOMETRY: std::sync::OnceLock<Option<(usize, usize, bool)>> =
            std::sync::OnceLock::new();
        *GEOMETRY.get_or_init(|| {
            let value = std::env::var("QX_V2_FINAL_HEAD_SWEEP").ok()?;
            let mut fields = value.split(':');
            let rows = fields.next()?.parse().ok()?;
            let groups = fields.next()?.parse().ok()?;
            let cache = match fields.next()?.to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => true,
                "0" | "false" | "no" | "off" => false,
                _ => return None,
            };
            if !matches!(rows, 1 | 2 | 4 | 8) || !matches!(groups, 1 | 2 | 4 | 8) {
                return None;
            }
            Some((rows, groups, cache))
        })
    }

    /// Exact greedy-decode final-head fusion. It is disabled by default so
    /// quality and diagnostic callers can continue to receive full logits.
    pub fn final_argmax_fused_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_FINAL_ARGMAX_FUSED")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    }

    /// Projection-aware final RMS plus greedy vocabulary reduction.
    pub fn final_argmax_normalized_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_FINAL_ARGMAX_NORMALIZED")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    }

    /// Exact-Q4 row-inline candidate: skip per-threadgroup activation staging
    /// when the 4 KiB vector is already hot from the preceding router/RMS path.
    /// Set `QX_V2_MOE_INLINE_INPUT_CACHE=0` to evaluate the no-cache mapping.
    pub fn moe_inline_input_cache_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            !matches!(
                std::env::var("QX_V2_MOE_INLINE_INPUT_CACHE")
                    .ok()
                    .map(|value| value.to_ascii_lowercase())
                    .as_deref(),
                Some("0") | Some("false") | Some("no") | Some("off")
            )
        })
    }

    /// Experimental vectorized consumer for the exact-Q4 row-inline expert
    /// representation. It is opt-in until a complete-token gate promotes it.
    pub fn moe_inline_vec4_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QX_V2_MOE_INLINE_VEC4")
                    .ok()
                    .map(|value| value.to_ascii_lowercase())
                    .as_deref(),
                Some("1") | Some("true") | Some("yes") | Some("on")
            )
        })
    }

    /// Experimental route-set scheduler for the exact single-row Q4
    /// row-inline representation. One threadgroup owns an output-row tile and
    /// consumes all eight selected experts, staging the activation once. The
    /// candidate is opt-in because it trades eight independent groups for a
    /// serial expert loop and must be judged by complete-token latency.
    pub fn moe_cross_expert_activation_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QX_V2_MOE_CROSS_EXPERT_ACTIVATION")
                    .ok()
                    .map(|value| value.to_ascii_lowercase())
                    .as_deref(),
                Some("1") | Some("true") | Some("yes") | Some("on")
            )
        })
    }

    /// Opt-in consumer for the exact Q4CrossExpertTile8 artifact stream. The
    /// stream is shared by all experts and is addressed with the selected
    /// expert slot IDs; keep this separate from the older row-inline route-set
    /// candidate until artifact-level parity promotes it.
    pub fn moe_cross_expert_tile8_enabled() -> bool {
        matches!(
            std::env::var("QX_V2_MOE_CROSS_EXPERT_TILE8")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    }

    /// Bounded Granite-M5 top-8 row-stream candidate for exact Q4 row-inline
    /// gate/up. One SIMD group owns one selected expert while the threadgroup
    /// advances an eight-row tile in lockstep. It is opt-in until complete
    /// token parity and sustained latency evidence promote it.
    pub fn moe_row_inline_streaming_enabled() -> bool {
        matches!(
            std::env::var("QX_V2_MOE_ROW_INLINE_STREAM8")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    }

    /// Experimental decode-specific Q4 row-inline mapping candidate. It uses
    /// two SIMD groups per 64-thread threadgroup and computes two output rows
    /// per SIMD group, matching the batch-1 work shape used by the llama audit.
    /// It is intentionally opt-in and only applies to the exact single-row
    /// row-inline representation.
    pub fn moe_llama_like_mapping_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QX_V2_MOE_LLAMA_LIKE_MAPPING")
                    .ok()
                    .map(|value| value.to_ascii_lowercase())
                    .as_deref(),
                Some("1") | Some("true") | Some("yes") | Some("on")
            )
        })
    }

    /// Exact-Q4 row-inline decode candidate. Set `QX_V2_MOE_DECODE_DIRECT=1`
    /// to use two SIMD groups per 64-thread threadgroup, with two rows
    /// computed serially per SIMD group and direct activation reads.
    pub fn moe_decode_direct_enabled() -> bool {
        matches!(
            std::env::var("QX_V2_MOE_DECODE_DIRECT")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    }

    /// Exact-Q4 dataflow factorial candidate. Unlike the geometry-only
    /// llama-like mapping, each SIMD group owns two rows and reuses each
    /// activation load across both accumulators. It is opt-in until the full
    /// sequential promotion ladder validates it.
    pub fn moe_two_row_dataflow_enabled() -> bool {
        matches!(
            std::env::var("QX_V2_MOE_TWO_ROW_DATAFLOW")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    }

    pub fn moe_q4k_mature_gate_up_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            match std::env::var("QX_V2_MOE_Q4K_MATURE")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref()
            {
                Some("gate") => true,
                Some("down") => false,
                Some("1") | Some("true") | Some("yes") | Some("on") => true,
                _ => false,
            }
        })
    }

    pub fn moe_q4k_mature_down_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            match std::env::var("QX_V2_MOE_Q4K_MATURE")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref()
            {
                Some("gate") => false,
                Some("down") => true,
                Some("1") | Some("true") | Some("yes") | Some("on") => true,
                _ => false,
            }
        })
    }

    /// Opt-in factorial candidate for the exact-Q4 row-inline stream. The
    /// representation API does not yet expose a distinct fused transaction
    /// block, so this selects temporary kernels that consume the existing
    /// 18-byte group packets in compile-time eight-group (144-byte) blocks.
    /// Values are `qx`/`canonical` for the QX mapping or `llama` for the
    /// two-SIMD-group decode mapping. Unset/false disables the candidate.
    pub fn moe_fused_transaction_block_mapping() -> Option<bool> {
        static MAPPING: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
        MAPPING
            .get_or_init(
                || match std::env::var("QX_V2_MOE_FUSED_TRANSACTION_BLOCK") {
                    Ok(value) => match value.to_ascii_lowercase().as_str() {
                        "qx" | "canonical" | "1" | "true" | "yes" | "on" => Some(false),
                        "llama" | "llama-like" | "llama_like" => Some(true),
                        _ => None,
                    },
                    Err(_) => None,
                },
            )
            .to_owned()
    }

    /// Experimental four-stage MoE topology: one 1024-threadgroup owns each
    /// selected expert projection. It reduces repeated staging of the small
    /// batch-1 activation vector. The canonical path remains the default.
    pub fn moe_wide_expert_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_MOE_WIDE_EXPERT")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    }

    /// Limit the number of routed experts exposed to each exact-Q4 row-inline
    /// dispatch. A value of 1, 2, 4, or 8 is accepted; 8 preserves the
    /// canonical single-dispatch path. The wave path reuses the validated
    /// kernels and offsets the expert/input/output views, so it changes only
    /// memory-stream concurrency.
    pub fn moe_expert_wave_width() -> usize {
        static WIDTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *WIDTH.get_or_init(|| {
            match std::env::var("QX_V2_MOE_WAVE_WIDTH")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
            {
                Some(1) | Some(2) | Some(4) | Some(8) => std::env::var("QX_V2_MOE_WAVE_WIDTH")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap(),
                _ => 8,
            }
        })
    }

    /// Validated fixed-shape RoPE/KV topology. The split Q plus direct-KV
    /// path is canonical after its sequential parity and full-token gates.
    /// Set `QX_V2_ROPE_KV_TOPOLOGY=legacy` (or `0`) for the one-dispatch
    /// rollback path.
    pub fn rope_kv_split_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            !matches!(
                std::env::var("QX_V2_ROPE_KV_TOPOLOGY")
                    .ok()
                    .map(|value| value.to_ascii_lowercase())
                    .as_deref(),
                Some("0")
                    | Some("false")
                    | Some("no")
                    | Some("off")
                    | Some("legacy")
                    | Some("current")
            )
        })
    }

    /// Experimental separate Q/K/V SIMD topology. The fused QKV kernel stays
    /// canonical until this candidate passes the full parity and throughput
    /// gates. Set `QX_V2_QKV_TOPOLOGY=split` (or `separate`) to run three
    /// shape-specialized dense dispatches; set it to `fused`/`legacy` to keep
    /// the current one-dispatch path.
    pub fn qkv_split_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QX_V2_QKV_TOPOLOGY")
                    .ok()
                    .map(|value| value.to_ascii_lowercase())
                    .as_deref(),
                Some("split") | Some("separate") | Some("qkv-split")
            )
        })
    }

    /// Experimental fused-QKV geometry selected by the benchmark sweep. The
    /// candidate keeps one dispatch but gives Q and KV regions different row
    /// ownership. Set `QX_V2_QKV_GEOMETRY=q2-kv1-g8` to enable it.
    pub fn qkv_geometry_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QX_V2_QKV_GEOMETRY")
                    .ok()
                    .map(|value| value.to_ascii_lowercase())
                    .as_deref(),
                Some("q2-kv1-g8")
                    | Some("q2x1x8")
                    | Some("q2-kv1-g8-nocache")
                    | Some("q2x1x8-nocache")
            )
        })
    }

    pub fn qkv_geometry_cache_enabled() -> bool {
        !matches!(
            std::env::var("QX_V2_QKV_GEOMETRY")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("q2-kv1-g8-nocache") | Some("q2x1x8-nocache")
        )
    }

    /// Optional FP16 KV cache experiment. Activations and score scratch remain
    /// FP32; only persisted K/V cache storage and its attention loads change.
    pub fn kv_fp16_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_KV_FP16")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    }

    /// Optional asymmetric KV experiment controls. The all-FP16 switch is a
    /// shorthand for enabling both sides of the cache.
    pub fn kv_key_fp16_enabled() -> bool {
        kv_fp16_enabled()
            || std::env::var("QX_V2_KV_K_FP16")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
    }

    pub fn kv_value_fp16_enabled() -> bool {
        kv_fp16_enabled()
            || std::env::var("QX_V2_KV_V_FP16")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
    }

    fn fp16_layer_list(variable: &str) -> Vec<usize> {
        std::env::var(variable)
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|item| item.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn kv_key_fp16_for_layer(layer: usize) -> bool {
        static LAYERS: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
        kv_key_fp16_enabled()
            || LAYERS
                .get_or_init(|| fp16_layer_list("QX_V2_KV_K_FP16_LAYERS"))
                .contains(&layer)
    }

    pub fn kv_value_fp16_for_layer(layer: usize) -> bool {
        static LAYERS: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
        kv_value_fp16_enabled()
            || LAYERS
                .get_or_init(|| fp16_layer_list("QX_V2_KV_V_FP16_LAYERS"))
                .contains(&layer)
    }

    fn bytes_per_group(quant_type: QuantType) -> usize {
        match quant_type {
            QuantType::Q2 => 8,
            QuantType::Q3 => 12,
            QuantType::Q4 => 16,
            QuantType::Q4K => 140,
        }
    }

    fn buffer_len(rows: usize, cols: usize, quant_type: QuantType) -> usize {
        rows * (cols / GROUP_SIZE) * bytes_per_group(quant_type)
    }

    fn scale_len(rows: usize, cols: usize) -> usize {
        rows * (cols / GROUP_SIZE) * std::mem::size_of::<u16>()
    }

    /// A prepared expert owns persistent Metal buffers for its packed weights.
    type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

    pub struct PreparedFusedExpert {
        pipeline: Pipeline,
        aggregate_pipeline: Pipeline,
        layout: PackingLayout,
        gate_up_packed: RuntimeBuffer,
        gate_up_scales: RuntimeBuffer,
        down_packed: RuntimeBuffer,
        down_scales: RuntimeBuffer,
    }

    /// A single Q4 matrix stage used by the kernel microbenchmarks. It keeps
    /// the artifact's packing layout so expert gate/up and down projections
    /// can be measured independently.
    pub struct PreparedQ4Stage {
        pipeline: Pipeline,
        packed: RuntimeBuffer,
        scales: RuntimeBuffer,
        rows: usize,
        columns: usize,
        layout: PackingLayout,
    }

    pub struct PreparedQ4SimdStage {
        pipeline: Pipeline,
        packed: Retained<ProtocolObject<dyn MTLBuffer>>,
        scales: Retained<ProtocolObject<dyn MTLBuffer>>,
        rows: usize,
        columns: usize,
        layout: PackingLayout,
    }

    impl PreparedQ4SimdStage {
        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn columns(&self) -> usize {
            self.columns
        }
    }

    pub struct PreparedLogits {
        pipeline: Pipeline,
        embeddings: Retained<ProtocolObject<dyn MTLBuffer>>,
        rows: usize,
    }

    pub struct PreparedDense {
        pipeline: Pipeline,
        weights: Retained<ProtocolObject<dyn MTLBuffer>>,
        rows: usize,
        columns: usize,
    }

    /// A simple FP16-storage GEMV used as the control implementation for the
    /// Q4 kernel benchmark. We keep the bit representation as u16 at this
    /// boundary so qx-metal does not need to expose the `half` crate.
    pub struct PreparedF16Dense {
        pipeline: Pipeline,
        weights: Retained<ProtocolObject<dyn MTLBuffer>>,
        rows: usize,
        columns: usize,
    }

    impl PreparedF16Dense {
        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn columns(&self) -> usize {
            self.columns
        }
    }

    pub struct PreparedQuantizedDense {
        pipeline: Pipeline,
        simd_pipeline: Pipeline,
        sweep_pipeline: Pipeline,
        simd_norm_pipeline: Pipeline,
        packed: RuntimeBuffer,
        scales: RuntimeBuffer,
        rows: usize,
        columns: usize,
    }

    /// Dense row-major Q4 weights prepared for the SIMD-cooperative control
    /// kernel used by the kernel-only benchmark.
    pub struct PreparedQuantizedSimdDense {
        pipeline: Pipeline,
        packed: Retained<ProtocolObject<dyn MTLBuffer>>,
        scales: Retained<ProtocolObject<dyn MTLBuffer>>,
        rows: usize,
        columns: usize,
    }

    impl PreparedQuantizedSimdDense {
        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn columns(&self) -> usize {
            self.columns
        }
    }

    impl PreparedQuantizedDense {
        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn columns(&self) -> usize {
            self.columns
        }
    }

    /// Three Q4 projections sharing one input vector. The fused kernel keeps
    /// the projection weights compressed and emits Q, K, and V in one dispatch.
    pub struct PreparedQuantizedQkv {
        pipeline: Pipeline,
        simd_pipeline: Pipeline,
        simd_norm_pipeline: Pipeline,
        split_simd_pipeline: Pipeline,
        split_simd_norm_pipeline: Pipeline,
        sweep_pipeline: Pipeline,
        q_packed: RuntimeBuffer,
        q_scales: RuntimeBuffer,
        k_packed: RuntimeBuffer,
        k_scales: RuntimeBuffer,
        v_packed: RuntimeBuffer,
        v_scales: RuntimeBuffer,
        columns: usize,
    }

    pub struct PreparedDynamicQ4Moe {
        pipeline: Pipeline,
        simd_pipeline: Pipeline,
        batched_gate_up_pipeline: Pipeline,
        batched_gate_up_norm_pipeline: Pipeline,
        inline_gate_up_pipeline: Pipeline,
        inline_gate_up_norm_pipeline: Pipeline,
        inline_gate_up_vec4_pipeline: Pipeline,
        inline_gate_up_vec4_norm_pipeline: Pipeline,
        inline_gate_up_nocache_pipeline: Pipeline,
        inline_gate_up_nocache_norm_pipeline: Pipeline,
        cross_expert_gate_up_pipeline: Pipeline,
        cross_expert_gate_up_norm_pipeline: Pipeline,
        cross_expert_tile8_gate_up_pipeline: Pipeline,
        cross_expert_tile8_gate_up_norm_pipeline: Pipeline,
        row_stream_gate_up_pipeline: Pipeline,
        row_stream_gate_up_norm_pipeline: Pipeline,
        llama_like_gate_up_pipeline: Pipeline,
        llama_like_gate_up_norm_pipeline: Pipeline,
        decode_direct_gate_up_pipeline: Pipeline,
        decode_direct_gate_up_norm_pipeline: Pipeline,
        two_row_dataflow_gate_up_pipeline: Option<Pipeline>,
        two_row_dataflow_gate_up_norm_pipeline: Option<Pipeline>,
        q4k_mature_gate_up_pipeline: Pipeline,
        q4k_mature_gate_up_norm_pipeline: Pipeline,
        fused_block_qx_gate_up_pipeline: Option<Pipeline>,
        fused_block_qx_gate_up_norm_pipeline: Option<Pipeline>,
        fused_block_llama_gate_up_pipeline: Option<Pipeline>,
        fused_block_llama_gate_up_norm_pipeline: Option<Pipeline>,
        inline_gate_up_rows2_pipeline: Option<Pipeline>,
        inline_gate_up_rows2_norm_pipeline: Option<Pipeline>,
        inline_gate_up_rows4_pipeline: Option<Pipeline>,
        inline_gate_up_rows4_norm_pipeline: Option<Pipeline>,
        inline_gate_up_rows8_pipeline: Option<Pipeline>,
        inline_gate_up_rows8_norm_pipeline: Option<Pipeline>,
        wide_gate_up_pipeline: Pipeline,
        wide_gate_up_norm_pipeline: Pipeline,
        wide_supported: bool,
        expertwise: bool,
        expertwise_q4_gate_up_pipeline: Option<Pipeline>,
        expertwise_q4_gate_up_norm_pipeline: Option<Pipeline>,
        expertwise_q3_gate_up_pipeline: Option<Pipeline>,
        expertwise_q3_gate_up_norm_pipeline: Option<Pipeline>,
        expertwise_q4_down_pipeline: Option<Pipeline>,
        expertwise_q3_down_pipeline: Option<Pipeline>,
        expertwise_q4_gate_up_packed: Option<RuntimeBuffer>,
        expertwise_q4_gate_up_scales: Option<RuntimeBuffer>,
        expertwise_q3_gate_up_packed: Option<RuntimeBuffer>,
        expertwise_q3_gate_up_scales: Option<RuntimeBuffer>,
        expertwise_gate_up_map: Option<RuntimeBuffer>,
        expertwise_q4_down_packed: Option<RuntimeBuffer>,
        expertwise_q4_down_scales: Option<RuntimeBuffer>,
        expertwise_q3_down_packed: Option<RuntimeBuffer>,
        expertwise_q3_down_scales: Option<RuntimeBuffer>,
        expertwise_down_map: Option<RuntimeBuffer>,
        expertwise_q3_gate_up_block_bytes: Option<u32>,
        expertwise_q3_down_block_bytes: Option<u32>,
        q3_gate_up_pipeline: Option<Pipeline>,
        q3_gate_up_norm_pipeline: Option<Pipeline>,
        q3_gate_up_packed: Option<RuntimeBuffer>,
        q3_gate_up_scales: Option<RuntimeBuffer>,
        q3_gate_up_block_bytes: Option<u32>,
        q3_down_pipeline: Option<Pipeline>,
        q3_down_packed: Option<RuntimeBuffer>,
        q3_down_scales: Option<RuntimeBuffer>,
        q3_down_block_bytes: Option<u32>,
        batched_gate_up_touch_pipeline: Pipeline,
        batched_gate_up_touch_full_pipeline: Pipeline,
        batched_gate_up_touch_direct_1_pipeline: Pipeline,
        batched_gate_up_touch_direct_2_pipeline: Pipeline,
        batched_gate_up_touch_direct_4_pipeline: Pipeline,
        batched_gate_up_touch_direct_8_pipeline: Pipeline,
        batched_gate_up_synthetic_pipeline: Pipeline,
        batched_gate_up_synthetic_norm_pipeline: Pipeline,
        batched_projection_pipeline: Pipeline,
        batched_projection_norm_pipeline: Pipeline,
        batched_silu_pipeline: Pipeline,
        batched_down_pipeline: Pipeline,
        inline_down_pipeline: Pipeline,
        decode_direct_down_pipeline: Pipeline,
        q4k_mature_down_pipeline: Pipeline,
        cross_expert_tile8_down_pipeline: Pipeline,
        inline_layout: bool,
        down_inline_layout: bool,
        wide_down_pipeline: Pipeline,
        batched_accumulate_pipeline: Pipeline,
        gate_up_packed: RuntimeBuffer,
        gate_up_scales: RuntimeBuffer,
        split_gate_packed: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        split_gate_scales: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        split_up_packed: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        split_up_scales: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        split_gate_up: bool,
        down_packed: RuntimeBuffer,
        down_scales: RuntimeBuffer,
        layout: PackingLayout,
        gate_up_layout: PackingLayout,
        down_layout: PackingLayout,
    }

    pub struct PreparedQ4Embedding {
        pipeline: Pipeline,
        packed: RuntimeBuffer,
        scales: RuntimeBuffer,
        rows: usize,
        columns: usize,
    }

    /// A buffer allocated during V2 runtime initialization and reused by every
    /// token. It intentionally exposes no CPU-side tensor view; callers can
    /// only bind it to an encode graph or read tiny synchronization results
    /// after the graph has completed.
    #[derive(Clone)]
    pub struct RuntimeBuffer {
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        length: usize,
    }

    impl RuntimeBuffer {
        pub fn subregion(&self, offset: usize, length: usize) -> Result<Self, String> {
            let end = offset
                .checked_add(length)
                .ok_or_else(|| "V2 runtime buffer region overflows".to_owned())?;
            if end > self.length {
                return Err("V2 runtime buffer region exceeds its parent".to_owned());
            }
            Ok(Self {
                buffer: self.buffer.clone(),
                offset: self.offset + offset,
                length,
            })
        }
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum DecodeKernel {
        RmsNorm,
        RmsNormSimd,
        RmsNormScalar,
        F32ToF16,
        F16ToF32,
        Residual,
        RopeKv,
        RopeQuerySimd,
        RopeKvSimd,
        Attention,
        AttentionSimd,
        TopK,
        Noop,
        AttentionQk,
        AttentionSoftmax,
        AttentionV,
        RopeKvF16,
        RopeKvKf32Vf16,
        RopeKvKf16Vf32,
        AttentionQkF16,
        AttentionVF16,
        AttentionVPartial,
        AttentionVPartialF16,
        AttentionVReduce,
        AttentionSoftmaxMax,
        AttentionSoftmaxSimd,
        TopKMask,
        AttentionQkKMajor,
        AttentionStreamGqa,
        AttentionChunkedGqa,
        AttentionChunkedMerge,
    }

    fn bytes_of<T>(value: &T) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct MetalMetrics {
        pub command_buffers: u64,
        pub encoders: u64,
        pub dispatches: u64,
        pub gpu_time_seconds: f64,
        pub cpu_encode_seconds: f64,
        pub cpu_wait_seconds: f64,
    }

    /// Timestamp samples recorded at dispatch-boundary markers in one decode
    /// graph. This is intentionally separate from aggregate metrics: enabling
    /// it changes command encoding and is reserved for diagnostic profiles.
    #[derive(Clone, Debug, Default)]
    pub struct StageTimeline {
        /// GPU timestamp ticks, in the order markers were encoded.
        pub timestamps: Vec<u64>,
        /// Aggregate GPU duration for the command buffer sequence. Counter
        /// ticks are normalized against this duration by the benchmark tool,
        /// avoiding any dependency on a private timestamp frequency API.
        pub gpu_duration_seconds: f64,
        /// Fallback stage durations when dispatch counters are unavailable.
        /// These are command-buffer intervals created without CPU waits.
        pub segment_gpu_seconds: Vec<f64>,
    }

    #[derive(Debug)]
    struct CounterTimeline {
        sample_buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
        sample_count: usize,
    }

    impl CounterTimeline {
        fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
            if !device.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary) {
                eprintln!("QX_STAGE_COUNTERS unavailable: dispatch-boundary sampling unsupported");
                return None;
            }
            let Some(sets) = device.counterSets() else {
                eprintln!("QX_STAGE_COUNTERS unavailable: device returned no counter sets");
                return None;
            };
            let mut timestamp_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
            for index in 0..sets.count() {
                let set = sets.objectAtIndex(index);
                let name = set.name().to_string().to_ascii_lowercase();
                if name
                    == (unsafe { MTLCommonCounterSetTimestamp })
                        .to_string()
                        .to_ascii_lowercase()
                    || name.contains("timestamp")
                {
                    timestamp_set = Some(set);
                    break;
                }
            }
            let Some(timestamp_set) = timestamp_set else {
                eprintln!("QX_STAGE_COUNTERS unavailable: no timestamp counter set");
                return None;
            };
            let descriptor = MTLCounterSampleBufferDescriptor::new();
            descriptor.setCounterSet(Some(&timestamp_set));
            descriptor.setStorageMode(MTLStorageMode::Shared);
            // A complete Granite token has several hundred dispatches. The
            // diagnostic timeline uses one marker per logical boundary and is
            // deliberately bounded so it cannot grow with a long run.
            let sample_count = 512;
            unsafe { descriptor.setSampleCount(sample_count) };
            let sample_buffer = match device.newCounterSampleBufferWithDescriptor_error(&descriptor)
            {
                Ok(buffer) => buffer,
                Err(error) => {
                    eprintln!(
                        "QX_STAGE_COUNTERS unavailable: counter buffer allocation failed: {error:?}"
                    );
                    return None;
                }
            };
            Some(Self {
                sample_buffer,
                sample_count,
            })
        }

        fn sample(&mut self, encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>) {
            if self.sample_count == 0 {
                return;
            }
            let index = 512usize.saturating_sub(self.sample_count);
            unsafe {
                encoder.sampleCountersInBuffer_atSampleIndex_withBarrier(
                    &self.sample_buffer,
                    index,
                    true,
                );
            }
            self.sample_count -= 1;
        }

        fn resolve(self, gpu_duration_seconds: f64) -> StageTimeline {
            let count = 512usize.saturating_sub(self.sample_count);
            if count == 0 {
                return StageTimeline {
                    timestamps: Vec::new(),
                    gpu_duration_seconds,
                    segment_gpu_seconds: Vec::new(),
                };
            }
            let Some(data) = (unsafe {
                self.sample_buffer
                    .resolveCounterRange(NSRange::new(0, count))
            }) else {
                return StageTimeline {
                    timestamps: Vec::new(),
                    gpu_duration_seconds,
                    segment_gpu_seconds: Vec::new(),
                };
            };
            let bytes = data.to_vec();
            let stride = std::mem::size_of::<MTLCounterResultTimestamp>();
            let timestamps = bytes
                .chunks_exact(stride)
                .take(count)
                .map(|chunk| {
                    let mut raw = [0_u8; std::mem::size_of::<u64>()];
                    raw.copy_from_slice(&chunk[..std::mem::size_of::<u64>()]);
                    u64::from_ne_bytes(raw)
                })
                .collect();
            StageTimeline {
                timestamps,
                gpu_duration_seconds,
                segment_gpu_seconds: Vec::new(),
            }
        }
    }

    #[derive(Debug, Default)]
    struct MetalMetricsState {
        command_buffers: u64,
        encoders: u64,
        dispatches: u64,
        gpu_time_seconds: f64,
        cpu_encode_seconds: f64,
        cpu_wait_seconds: f64,
    }

    struct MetalScratch {
        input_f32: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        input_u16: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        dense_outputs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
        expert_outputs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
        moe_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        logits_output: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    }

    pub struct MetalContext {
        pub device: Retained<ProtocolObject<dyn MTLDevice>>,
        pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pipelines: Mutex<[Option<Pipeline>; 3]>,
        aggregate_pipelines: Mutex<[Option<Pipeline>; 3]>,
        logits_pipeline: Mutex<Option<Pipeline>>,
        dense_pipeline: Mutex<Option<Pipeline>>,
        f16_dense_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_library: Mutex<Option<Retained<ProtocolObject<dyn MTLLibrary>>>>,
        q4_dense_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_simd_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_sweep_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_argmax_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_argmax_norm_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_simd_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_simd_norm_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_sweep_pipeline: Mutex<Option<Pipeline>>,
        q4_gemv_pipeline: Mutex<Option<Pipeline>>,
        q4_simd_gemv_pipeline: Mutex<Option<Pipeline>>,
        q4_simd_norm_gemv_pipeline: Mutex<Option<Pipeline>>,
        dynamic_moe_pipeline: Mutex<Option<Pipeline>>,
        dynamic_moe_library: Mutex<Option<Retained<ProtocolObject<dyn MTLLibrary>>>>,
        dynamic_moe_simd_pipeline: Mutex<Option<Pipeline>>,
        q4_embedding_pipeline: Mutex<Option<Pipeline>>,
        argmax_partial_pipeline: Mutex<Option<Pipeline>>,
        argmax_reduce_pipeline: Mutex<Option<Pipeline>>,
        argmax_reduce_partials_pipeline: Mutex<Option<Pipeline>>,
        v2_pipelines: Mutex<[Option<Pipeline>; 31]>,
        scratch: Mutex<MetalScratch>,
        metrics: Mutex<MetalMetricsState>,
        stage_counters: Mutex<bool>,
        last_stage_timeline: Mutex<StageTimeline>,
        no_copy_buffers: Mutex<HashMap<(usize, usize), RuntimeBuffer>>,
    }

    /// Token-level command stream for Runtime V2. Operators may encode fixed
    /// shape work through this object; lifecycle operations occur only in
    /// `finish`, which is the single permitted synchronization point.
    pub struct DecodeGraph {
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        dispatches: u64,
        counter_timeline: Option<CounterTimeline>,
        committed: Vec<(Retained<ProtocolObject<dyn MTLCommandBuffer>>, u64, u64)>,
        encoders: u64,
        encode_started_at: Instant,
    }

    impl DecodeGraph {
        pub fn encoder(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
            &self.encoder
        }

        pub fn encode_prepared_dense(
            &mut self,
            prepared: &PreparedDense,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || output.length < prepared.rows * std::mem::size_of::<f32>()
            {
                return Err("V2 dense buffer is smaller than its fixed shape".to_owned());
            }
            self.encoder.setComputePipelineState(&prepared.pipeline);
            let columns = prepared.columns as u32;
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&prepared.weights), 0, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 1);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 2);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    3,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: prepared.rows,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_prepared_q4_dense(
            &mut self,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            self.encode_prepared_q4_dense_scaled(prepared, input, output, 1.0)
        }

        pub fn encode_prepared_q4_dense_scaled(
            &mut self,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
            multiplier: f32,
        ) -> Result<(), String> {
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || output.length < prepared.rows * std::mem::size_of::<f32>()
            {
                return Err("V2 Q4 dense buffer is smaller than its fixed shape".to_owned());
            }
            self.encoder.setComputePipelineState(&prepared.pipeline);
            let columns = prepared.columns as u32;
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.packed.buffer),
                    prepared.packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.scales.buffer),
                    prepared.scales.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 2);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 3);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    4,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    5,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                objc2_metal::MTLSize {
                    width: prepared.rows,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_prepared_q4_dense_simd(
            &mut self,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            self.encode_prepared_q4_dense_simd_scaled(prepared, input, output, 1.0)
        }

        pub fn encode_prepared_q4_dense_simd_scaled(
            &mut self,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
            multiplier: f32,
        ) -> Result<(), String> {
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || output.length < prepared.rows * std::mem::size_of::<f32>()
            {
                return Err("V2 SIMD Q4 dense buffer is smaller than its fixed shape".to_owned());
            }
            if prepared.rows >= 16_384 {
                if let Some((rows_per_simd, simd_groups, cache_input)) = final_head_sweep_geometry()
                {
                    return self.encode_prepared_q4_dense_sweep_scaled(
                        prepared,
                        input,
                        output,
                        multiplier,
                        rows_per_simd,
                        simd_groups,
                        cache_input,
                    );
                }
            }
            self.encoder
                .setComputePipelineState(&prepared.simd_pipeline);
            let columns = prepared.columns as u32;
            let rows = prepared.rows as u32;
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.packed.buffer),
                    prepared.packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.scales.buffer),
                    prepared.scales.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 2);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 3);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    4,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    5,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&rows).cast(),
                    std::mem::size_of_val(&rows),
                    6,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows.div_ceil(8),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        /// Encode an exact vocabulary projection that produces only partial
        /// maxima, then reduce those partials to the greedy token ID.
        pub fn encode_prepared_q4_dense_argmax(
            &mut self,
            context: &MetalContext,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            partials: &RuntimeBuffer,
            token_output: &RuntimeBuffer,
            multiplier: f32,
        ) -> Result<(), String> {
            self.encode_prepared_q4_dense_argmax_impl(
                context,
                prepared,
                input,
                None,
                None,
                partials,
                token_output,
                multiplier,
            )
        }

        /// Projection-aware exact vocabulary projection. The raw input is
        /// normalized inside the consumer using the scalar inverse RMS.
        pub fn encode_prepared_q4_dense_argmax_normalized(
            &mut self,
            context: &MetalContext,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            norm_weights: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            partials: &RuntimeBuffer,
            token_output: &RuntimeBuffer,
            multiplier: f32,
        ) -> Result<(), String> {
            self.encode_prepared_q4_dense_argmax_impl(
                context,
                prepared,
                input,
                Some(norm_weights),
                Some(inverse_rms),
                partials,
                token_output,
                multiplier,
            )
        }

        fn encode_prepared_q4_dense_argmax_impl(
            &mut self,
            context: &MetalContext,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            norm_weights: Option<&RuntimeBuffer>,
            inverse_rms: Option<&RuntimeBuffer>,
            partials: &RuntimeBuffer,
            token_output: &RuntimeBuffer,
            multiplier: f32,
        ) -> Result<(), String> {
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || token_output.length < std::mem::size_of::<u32>()
            {
                return Err("V2 fused final head buffer is smaller than its fixed shape".to_owned());
            }
            let partial_count = prepared.rows.div_ceil(8);
            let partial_bytes = partial_count * 2 * std::mem::size_of::<u32>();
            if partials.length < partial_bytes {
                return Err(
                    "V2 fused final head partial buffer is smaller than its fixed shape".to_owned(),
                );
            }
            let columns = prepared.columns as u32;
            let rows = prepared.rows as u32;
            let partial_count_u32 = partial_count as u32;
            let pipeline = if norm_weights.is_some() {
                context.q4_dense_argmax_norm_pipeline()?
            } else {
                context.q4_dense_argmax_pipeline()?
            };
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.packed.buffer),
                    prepared.packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.scales.buffer),
                    prepared.scales.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 2);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&partials.buffer), partials.offset, 3);
                for (value, index) in [(&columns, 4), (&rows, 5)] {
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(value).cast(),
                        std::mem::size_of_val(value),
                        index,
                    );
                }
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    6,
                );
                if let (Some(norm_weights), Some(inverse_rms)) = (norm_weights, inverse_rms) {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&norm_weights.buffer),
                        norm_weights.offset,
                        7,
                    );
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&inverse_rms.buffer),
                        inverse_rms.offset,
                        8,
                    );
                }
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: partial_count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;

            let reduced =
                partials.subregion(partial_bytes, 256 * 2 * std::mem::size_of::<u32>())?;
            let reduced_count = partial_count.div_ceil(256);
            let reduced_count_u32 = reduced_count as u32;
            let reduce_partials_pipeline = context.argmax_reduce_partials_pipeline()?;
            self.encoder
                .setComputePipelineState(&reduce_partials_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&partials.buffer), partials.offset, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&reduced.buffer), reduced.offset, 1);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&partial_count_u32).cast(),
                    std::mem::size_of_val(&partial_count_u32),
                    2,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: reduced_count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;

            let reduce_pipeline = context.argmax_reduce_pipeline()?;
            self.encoder.setComputePipelineState(&reduce_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&reduced.buffer), reduced.offset, 0);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&token_output.buffer),
                    token_output.offset,
                    1,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&reduced_count_u32).cast(),
                    std::mem::size_of_val(&reduced_count_u32),
                    2,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        fn encode_prepared_q4_dense_sweep_scaled(
            &mut self,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
            multiplier: f32,
            rows_per_simd: usize,
            simd_groups: usize,
            cache_input: bool,
        ) -> Result<(), String> {
            let rows_per_threadgroup = rows_per_simd * simd_groups;
            let threads_per_threadgroup = simd_groups * 32;
            self.encoder
                .setComputePipelineState(&prepared.sweep_pipeline);
            let columns = prepared.columns as u32;
            let rows = prepared.rows as u32;
            let rows_per_simd = rows_per_simd as u32;
            let rows_per_threadgroup = rows_per_threadgroup as u32;
            let cache_input = u32::from(cache_input);
            let threads_per_threadgroup = threads_per_threadgroup as u32;
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.packed.buffer),
                    prepared.packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.scales.buffer),
                    prepared.scales.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 2);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 3);
                for (value, index) in [
                    (&columns, 4),
                    (&rows, 6),
                    (&rows_per_simd, 7),
                    (&rows_per_threadgroup, 8),
                    (&cache_input, 9),
                    (&threads_per_threadgroup, 10),
                ] {
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(value).cast(),
                        std::mem::size_of_val(value),
                        index,
                    );
                }
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    5,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows.div_ceil(rows_per_threadgroup as usize),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_threadgroup as usize,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_prepared_q4_dense_simd_normalized(
            &mut self,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            norm_weights: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || norm_weights.length < prepared.columns * std::mem::size_of::<f32>()
                || inverse_rms.length < std::mem::size_of::<f32>()
                || output.length < prepared.rows * std::mem::size_of::<f32>()
            {
                return Err(
                    "V2 normalized SIMD Q4 dense buffers are smaller than their fixed shape"
                        .to_owned(),
                );
            }
            self.encoder
                .setComputePipelineState(&prepared.simd_norm_pipeline);
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;
            let rows = prepared.rows as u32;
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.packed.buffer),
                    prepared.packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.scales.buffer),
                    prepared.scales.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 2);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 3);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    4,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    5,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&rows).cast(),
                    std::mem::size_of_val(&rows),
                    6,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&norm_weights.buffer),
                    norm_weights.offset,
                    7,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&inverse_rms.buffer),
                    inverse_rms.offset,
                    8,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows.div_ceil(8),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_prepared_q4_qkv(
            &mut self,
            prepared: &PreparedQuantizedQkv,
            input: &RuntimeBuffer,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
        ) -> Result<(), String> {
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || query.length < 1024 * std::mem::size_of::<f32>()
                || key.length < 512 * std::mem::size_of::<f32>()
                || value.length < 512 * std::mem::size_of::<f32>()
            {
                return Err("V2 fused QKV buffers are smaller than their fixed shape".to_owned());
            }
            self.encoder.setComputePipelineState(&prepared.pipeline);
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.q_packed.buffer),
                    prepared.q_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.q_scales.buffer),
                    prepared.q_scales.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.k_packed.buffer),
                    prepared.k_packed.offset,
                    2,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.k_scales.buffer),
                    prepared.k_scales.offset,
                    3,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.v_packed.buffer),
                    prepared.v_packed.offset,
                    4,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.v_scales.buffer),
                    prepared.v_scales.offset,
                    5,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 6);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&query.buffer), query.offset, 7);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&key.buffer), key.offset, 8);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&value.buffer), value.offset, 9);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    10,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    11,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 2048,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_prepared_q4_qkv_simd(
            &mut self,
            prepared: &PreparedQuantizedQkv,
            input: &RuntimeBuffer,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
        ) -> Result<(), String> {
            if qkv_split_enabled() {
                return self.encode_prepared_q4_qkv_split(prepared, input, query, key, value);
            }
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || query.length < 1024 * std::mem::size_of::<f32>()
                || key.length < 512 * std::mem::size_of::<f32>()
                || value.length < 512 * std::mem::size_of::<f32>()
            {
                return Err(
                    "V2 SIMD fused QKV buffers are smaller than their fixed shape".to_owned(),
                );
            }
            self.encoder
                .setComputePipelineState(&prepared.simd_pipeline);
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;
            unsafe {
                for (buffer, index) in [
                    (&prepared.q_packed, 0),
                    (&prepared.q_scales, 1),
                    (&prepared.k_packed, 2),
                    (&prepared.k_scales, 3),
                    (&prepared.v_packed, 4),
                    (&prepared.v_scales, 5),
                ] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 6);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&query.buffer), query.offset, 7);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&key.buffer), key.offset, 8);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&value.buffer), value.offset, 9);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    10,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    11,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 2048 / 8,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        fn encode_prepared_q4_qkv_split(
            &mut self,
            prepared: &PreparedQuantizedQkv,
            input: &RuntimeBuffer,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
        ) -> Result<(), String> {
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || query.length < 1024 * std::mem::size_of::<f32>()
                || key.length < 512 * std::mem::size_of::<f32>()
                || value.length < 512 * std::mem::size_of::<f32>()
            {
                return Err(
                    "V2 split SIMD QKV buffers are smaller than their fixed shape".to_owned(),
                );
            }
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;

            for (packed, scales, output, rows) in [
                (&prepared.q_packed, &prepared.q_scales, query, 1024_usize),
                (&prepared.k_packed, &prepared.k_scales, key, 512_usize),
                (&prepared.v_packed, &prepared.v_scales, value, 512_usize),
            ] {
                self.encoder
                    .setComputePipelineState(&prepared.split_simd_pipeline);
                unsafe {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&packed.buffer), packed.offset, 0);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&scales.buffer), scales.offset, 1);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 2);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 3);
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&columns).cast(),
                        std::mem::size_of_val(&columns),
                        4,
                    );
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&multiplier).cast(),
                        std::mem::size_of_val(&multiplier),
                        5,
                    );
                    let rows = rows as u32;
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&rows).cast(),
                        std::mem::size_of_val(&rows),
                        6,
                    );
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: rows.div_ceil(8),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                self.dispatches += 1;
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_prepared_q4_qkv_simd_normalized(
            &mut self,
            prepared: &PreparedQuantizedQkv,
            input: &RuntimeBuffer,
            norm_weights: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
        ) -> Result<(), String> {
            if qkv_split_enabled() {
                return self.encode_prepared_q4_qkv_split_normalized(
                    prepared,
                    input,
                    norm_weights,
                    inverse_rms,
                    query,
                    key,
                    value,
                );
            }
            if qkv_geometry_enabled() {
                return self.encode_prepared_q4_qkv_geometry_normalized(
                    prepared,
                    input,
                    norm_weights,
                    inverse_rms,
                    query,
                    key,
                    value,
                );
            }
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || norm_weights.length < prepared.columns * std::mem::size_of::<f32>()
                || inverse_rms.length < std::mem::size_of::<f32>()
                || query.length < 1024 * std::mem::size_of::<f32>()
                || key.length < 512 * std::mem::size_of::<f32>()
                || value.length < 512 * std::mem::size_of::<f32>()
            {
                return Err(
                    "V2 normalized SIMD fused QKV buffers are smaller than their fixed shape"
                        .to_owned(),
                );
            }
            self.encoder
                .setComputePipelineState(&prepared.simd_norm_pipeline);
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;
            unsafe {
                for (buffer, index) in [
                    (&prepared.q_packed, 0),
                    (&prepared.q_scales, 1),
                    (&prepared.k_packed, 2),
                    (&prepared.k_scales, 3),
                    (&prepared.v_packed, 4),
                    (&prepared.v_scales, 5),
                ] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 6);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&query.buffer), query.offset, 7);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&key.buffer), key.offset, 8);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&value.buffer), value.offset, 9);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&norm_weights.buffer),
                    norm_weights.offset,
                    12,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&inverse_rms.buffer),
                    inverse_rms.offset,
                    13,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    10,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    11,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 2048 / 8,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_prepared_q4_qkv_geometry_normalized(
            &mut self,
            prepared: &PreparedQuantizedQkv,
            input: &RuntimeBuffer,
            norm_weights: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
        ) -> Result<(), String> {
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || norm_weights.length < prepared.columns * std::mem::size_of::<f32>()
                || inverse_rms.length < std::mem::size_of::<f32>()
                || query.length < 1024 * std::mem::size_of::<f32>()
                || key.length < 512 * std::mem::size_of::<f32>()
                || value.length < 512 * std::mem::size_of::<f32>()
            {
                return Err(
                    "V2 fused-QKV geometry buffers are smaller than their fixed shape".to_owned(),
                );
            }
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;
            let q_rows_per_simd = 2_u32;
            let kv_rows_per_simd = 1_u32;
            let simd_groups = 8_u32;
            let cache_input = u32::from(qkv_geometry_cache_enabled());
            let normalize = 1_u32;
            unsafe {
                for (buffer, index) in [
                    (&prepared.q_packed, 0),
                    (&prepared.q_scales, 1),
                    (&prepared.k_packed, 2),
                    (&prepared.k_scales, 3),
                    (&prepared.v_packed, 4),
                    (&prepared.v_scales, 5),
                ] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
                for (buffer, offset, index) in [
                    (&input.buffer, input.offset, 6),
                    (&query.buffer, query.offset, 7),
                    (&key.buffer, key.offset, 8),
                    (&value.buffer, value.offset, 9),
                    (&norm_weights.buffer, norm_weights.offset, 17),
                    (&inverse_rms.buffer, inverse_rms.offset, 18),
                ] {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(buffer), offset, index);
                }
                for (value, index) in [
                    (NonNull::from(&columns).cast(), 10),
                    (NonNull::from(&multiplier).cast(), 11),
                    (NonNull::from(&q_rows_per_simd).cast(), 12),
                    (NonNull::from(&kv_rows_per_simd).cast(), 13),
                    (NonNull::from(&simd_groups).cast(), 14),
                    (NonNull::from(&cache_input).cast(), 15),
                    (NonNull::from(&normalize).cast(), 16),
                ] {
                    self.encoder
                        .setBytes_length_atIndex(value, std::mem::size_of::<u32>(), index);
                }
            }
            self.encoder
                .setComputePipelineState(&prepared.sweep_pipeline);
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 192,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_prepared_q4_qkv_split_normalized(
            &mut self,
            prepared: &PreparedQuantizedQkv,
            input: &RuntimeBuffer,
            norm_weights: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
        ) -> Result<(), String> {
            if input.length < prepared.columns * std::mem::size_of::<f32>()
                || norm_weights.length < prepared.columns * std::mem::size_of::<f32>()
                || inverse_rms.length < std::mem::size_of::<f32>()
                || query.length < 1024 * std::mem::size_of::<f32>()
                || key.length < 512 * std::mem::size_of::<f32>()
                || value.length < 512 * std::mem::size_of::<f32>()
            {
                return Err(
                    "V2 normalized split SIMD QKV buffers are smaller than their fixed shape"
                        .to_owned(),
                );
            }
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;

            for (packed, scales, output, rows) in [
                (&prepared.q_packed, &prepared.q_scales, query, 1024_usize),
                (&prepared.k_packed, &prepared.k_scales, key, 512_usize),
                (&prepared.v_packed, &prepared.v_scales, value, 512_usize),
            ] {
                self.encoder
                    .setComputePipelineState(&prepared.split_simd_norm_pipeline);
                unsafe {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&packed.buffer), packed.offset, 0);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&scales.buffer), scales.offset, 1);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 2);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 3);
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&columns).cast(),
                        std::mem::size_of_val(&columns),
                        4,
                    );
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&multiplier).cast(),
                        std::mem::size_of_val(&multiplier),
                        5,
                    );
                    let rows = rows as u32;
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&rows).cast(),
                        std::mem::size_of_val(&rows),
                        6,
                    );
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&norm_weights.buffer),
                        norm_weights.offset,
                        7,
                    );
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&inverse_rms.buffer),
                        inverse_rms.offset,
                        8,
                    );
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: rows.div_ceil(8),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                self.dispatches += 1;
            }
            Ok(())
        }

        pub fn encode_dynamic_q4_moe(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            if input.length < 1024 * std::mem::size_of::<f32>()
                || expert_ids.length < 8 * std::mem::size_of::<u32>()
                || expert_weights.length < 8 * std::mem::size_of::<f32>()
                || output.length < 1024 * std::mem::size_of::<f32>()
            {
                return Err("V2 dynamic MoE buffer is smaller than its fixed shape".to_owned());
            }
            self.encoder.setComputePipelineState(&prepared.pipeline);
            let layout = prepared.layout as u32;
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_packed.buffer),
                    prepared.gate_up_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_scales.buffer),
                    prepared.gate_up_scales.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.down_packed.buffer),
                    prepared.down_packed.offset,
                    2,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.down_scales.buffer),
                    prepared.down_scales.offset,
                    3,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_ids.buffer),
                    expert_ids.offset,
                    5,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_weights.buffer),
                    expert_weights.offset,
                    6,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 7);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&layout).cast(),
                    std::mem::size_of_val(&layout),
                    8,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_dynamic_q4_moe_simd(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            if input.length < 1024 * std::mem::size_of::<f32>()
                || expert_ids.length < 8 * std::mem::size_of::<u32>()
                || expert_weights.length < 8 * std::mem::size_of::<f32>()
                || output.length < 1024 * std::mem::size_of::<f32>()
            {
                return Err("V2 SIMD dynamic MoE buffer is smaller than its fixed shape".to_owned());
            }
            self.encoder
                .setComputePipelineState(&prepared.simd_pipeline);
            let layout = prepared.layout as u32;
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_packed.buffer),
                    prepared.gate_up_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_scales.buffer),
                    prepared.gate_up_scales.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.down_packed.buffer),
                    prepared.down_packed.offset,
                    2,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.down_scales.buffer),
                    prepared.down_scales.offset,
                    3,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_ids.buffer),
                    expert_ids.offset,
                    5,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_weights.buffer),
                    expert_weights.offset,
                    6,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 7);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&layout).cast(),
                    std::mem::size_of_val(&layout),
                    8,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        /// GPU-resident batched SIMD MoE. The eight selected experts remain
        /// GPU-selected, while four fixed dispatches perform gate/up, SiLU,
        /// down, and weighted accumulation without eight expert-by-expert
        /// command submissions.
        pub fn encode_dynamic_q4_moe_batched_simd(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            self.encode_dynamic_q4_moe_batched_simd_with_norm(
                prepared,
                input,
                expert_ids,
                expert_weights,
                scratch,
                output,
                None,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_dynamic_q4_moe_batched_simd_normalized(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            norm_weights: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            self.encode_dynamic_q4_moe_batched_simd_with_norm(
                prepared,
                input,
                expert_ids,
                expert_weights,
                scratch,
                output,
                Some((norm_weights, inverse_rms)),
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_dynamic_q4_moe_batched_simd_with_norm(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            norm: Option<(&RuntimeBuffer, &RuntimeBuffer)>,
        ) -> Result<(), String> {
            if prepared.expertwise {
                return self.encode_dynamic_q4_moe_expertwise_with_norm(
                    prepared,
                    input,
                    expert_ids,
                    expert_weights,
                    scratch,
                    output,
                    norm,
                );
            }
            if prepared.split_gate_up {
                return self.encode_dynamic_q4_moe_batched_split_with_norm(
                    prepared,
                    input,
                    expert_ids,
                    expert_weights,
                    scratch,
                    output,
                    norm,
                );
            }
            if input.length < 1024 * std::mem::size_of::<f32>()
                || expert_ids.length < 8 * std::mem::size_of::<u32>()
                || expert_weights.length < 8 * std::mem::size_of::<f32>()
                || output.length < 1024 * std::mem::size_of::<f32>()
                || scratch.length < 8 * 1024 * 4 + 8 * 512 * 4 + 8 * 1024 * 4
            {
                return Err(
                    "V2 batched SIMD MoE buffers are smaller than their fixed shape".to_owned(),
                );
            }
            let gate_up = scratch.subregion(0, 8 * 1024 * 4)?;
            let activated = scratch.subregion(8 * 1024 * 4, 8 * 512 * 4)?;
            let expert_output = scratch.subregion(8 * 1024 * 4 + 8 * 512 * 4, 8 * 1024 * 4)?;
            let gate_up_layout = prepared.gate_up_layout as u32;
            let down_layout = prepared.down_layout as u32;
            let q3_block_bytes = prepared.q3_gate_up_block_bytes;
            let is_q4k = prepared.gate_up_packed.length == 32 * 1024 * 4 * 140
                && prepared.gate_up_scales.length == 32 * 1024 * 4 * 2 * 2
                && prepared.down_packed.length == 32 * 1024 * 2 * 140
                && prepared.down_scales.length == 32 * 1024 * 2 * 2 * 2;
            let q4k_mature_gate_up = is_q4k && moe_q4k_mature_gate_up_enabled();
            let q4k_mature_down = is_q4k && moe_q4k_mature_down_enabled();
            let expert_wave_width = moe_expert_wave_width();
            let wide_expert =
                moe_wide_expert_enabled() && prepared.wide_supported && q3_block_bytes.is_none();
            let inline_input_cache = moe_inline_input_cache_enabled();
            let fused_block_mapping = moe_fused_transaction_block_mapping();
            let transaction_gate_up = prepared.gate_up_layout.is_q4_transaction();
            let fused_block_active = q3_block_bytes.is_none()
                && (transaction_gate_up
                    || (fused_block_mapping.is_some()
                        && prepared.inline_layout
                        && prepared.gate_up_layout == PackingLayout::Q4RowInline));
            let llama_like_mapping = moe_llama_like_mapping_enabled()
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let decode_direct = moe_decode_direct_enabled()
                && prepared.inline_layout
                && prepared.down_inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && prepared.down_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let two_row_dataflow = moe_two_row_dataflow_enabled()
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let cross_expert_activation = moe_cross_expert_activation_enabled()
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let row_inline_streaming = moe_row_inline_streaming_enabled()
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let cross_expert_tile8 = moe_cross_expert_tile8_enabled()
                && prepared.gate_up_layout == PackingLayout::Q4CrossExpertTile8
                && q3_block_bytes.is_none();
            let cross_expert_tile8_down = moe_cross_expert_tile8_enabled()
                && prepared.down_layout == PackingLayout::Q4CrossExpertTile8
                && q3_block_bytes.is_none();
            let expert_wave = expert_wave_width < 8
                && prepared.inline_layout
                && prepared.down_inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && prepared.down_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none()
                && !decode_direct
                && !two_row_dataflow
                && !row_inline_streaming;

            if let Some((norm_weights, inverse_rms)) = norm {
                if let Some(pipeline) = prepared.q3_gate_up_norm_pipeline.as_ref() {
                    self.encoder.setComputePipelineState(pipeline);
                } else if q4k_mature_gate_up {
                    self.encoder
                        .setComputePipelineState(&prepared.q4k_mature_gate_up_norm_pipeline);
                } else if two_row_dataflow {
                    self.encoder.setComputePipelineState(
                        prepared
                            .two_row_dataflow_gate_up_norm_pipeline
                            .as_ref()
                            .ok_or_else(|| "two-row dataflow pipeline is missing".to_owned())?,
                    );
                } else if decode_direct {
                    self.encoder
                        .setComputePipelineState(&prepared.decode_direct_gate_up_norm_pipeline);
                } else if fused_block_active {
                    let pipeline = if fused_block_mapping == Some(true) {
                        prepared.fused_block_llama_gate_up_norm_pipeline.as_ref()
                    } else {
                        prepared.fused_block_qx_gate_up_norm_pipeline.as_ref()
                    }
                    .ok_or_else(|| "fused transaction-block pipeline is missing".to_owned())?;
                    self.encoder.setComputePipelineState(pipeline);
                } else if cross_expert_tile8 {
                    self.encoder.setComputePipelineState(
                        &prepared.cross_expert_tile8_gate_up_norm_pipeline,
                    );
                } else if llama_like_mapping {
                    self.encoder
                        .setComputePipelineState(&prepared.llama_like_gate_up_norm_pipeline);
                } else if cross_expert_activation {
                    self.encoder
                        .setComputePipelineState(&prepared.cross_expert_gate_up_norm_pipeline);
                } else if row_inline_streaming {
                    self.encoder
                        .setComputePipelineState(&prepared.row_stream_gate_up_norm_pipeline);
                } else if prepared
                    .gate_up_layout
                    .row_inline_packet_rows()
                    .is_some_and(|rows| rows > 1)
                {
                    let pipeline = match prepared.gate_up_layout {
                        PackingLayout::Q4RowInlineRows2 => {
                            prepared.inline_gate_up_rows2_norm_pipeline.as_ref()
                        }
                        PackingLayout::Q4RowInlineRows4 => {
                            prepared.inline_gate_up_rows4_norm_pipeline.as_ref()
                        }
                        PackingLayout::Q4RowInlineRows8 => {
                            prepared.inline_gate_up_rows8_norm_pipeline.as_ref()
                        }
                        _ => None,
                    }
                    .ok_or_else(|| "multi-row gate/up pipeline is missing".to_owned())?;
                    self.encoder.setComputePipelineState(pipeline);
                } else if prepared.inline_layout {
                    if inline_input_cache && moe_inline_vec4_enabled() {
                        self.encoder
                            .setComputePipelineState(&prepared.inline_gate_up_vec4_norm_pipeline);
                    } else if inline_input_cache {
                        self.encoder
                            .setComputePipelineState(&prepared.inline_gate_up_norm_pipeline);
                    } else {
                        self.encoder.setComputePipelineState(
                            &prepared.inline_gate_up_nocache_norm_pipeline,
                        );
                    }
                } else if wide_expert {
                    self.encoder
                        .setComputePipelineState(&prepared.wide_gate_up_norm_pipeline);
                } else {
                    self.encoder
                        .setComputePipelineState(&prepared.batched_gate_up_norm_pipeline);
                }
                if norm_weights.length < 1024 * 4 || inverse_rms.length < std::mem::size_of::<f32>()
                {
                    return Err("V2 normalized batched MoE buffers are too small".to_owned());
                }
                unsafe {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&norm_weights.buffer),
                        norm_weights.offset,
                        8,
                    );
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&inverse_rms.buffer),
                        inverse_rms.offset,
                        9,
                    );
                }
            } else {
                if let Some(pipeline) = prepared.q3_gate_up_pipeline.as_ref() {
                    self.encoder.setComputePipelineState(pipeline);
                } else if q4k_mature_gate_up {
                    self.encoder
                        .setComputePipelineState(&prepared.q4k_mature_gate_up_pipeline);
                } else if two_row_dataflow {
                    self.encoder.setComputePipelineState(
                        prepared
                            .two_row_dataflow_gate_up_pipeline
                            .as_ref()
                            .ok_or_else(|| "two-row dataflow pipeline is missing".to_owned())?,
                    );
                } else if decode_direct {
                    self.encoder
                        .setComputePipelineState(&prepared.decode_direct_gate_up_pipeline);
                } else if fused_block_active {
                    let pipeline = if fused_block_mapping == Some(true) {
                        prepared.fused_block_llama_gate_up_pipeline.as_ref()
                    } else {
                        prepared.fused_block_qx_gate_up_pipeline.as_ref()
                    }
                    .ok_or_else(|| "fused transaction-block pipeline is missing".to_owned())?;
                    self.encoder.setComputePipelineState(pipeline);
                } else if cross_expert_tile8 {
                    self.encoder
                        .setComputePipelineState(&prepared.cross_expert_tile8_gate_up_pipeline);
                } else if llama_like_mapping {
                    self.encoder
                        .setComputePipelineState(&prepared.llama_like_gate_up_pipeline);
                } else if cross_expert_activation {
                    self.encoder
                        .setComputePipelineState(&prepared.cross_expert_gate_up_pipeline);
                } else if row_inline_streaming {
                    self.encoder
                        .setComputePipelineState(&prepared.row_stream_gate_up_pipeline);
                } else if prepared
                    .gate_up_layout
                    .row_inline_packet_rows()
                    .is_some_and(|rows| rows > 1)
                {
                    let pipeline = match prepared.gate_up_layout {
                        PackingLayout::Q4RowInlineRows2 => {
                            prepared.inline_gate_up_rows2_pipeline.as_ref()
                        }
                        PackingLayout::Q4RowInlineRows4 => {
                            prepared.inline_gate_up_rows4_pipeline.as_ref()
                        }
                        PackingLayout::Q4RowInlineRows8 => {
                            prepared.inline_gate_up_rows8_pipeline.as_ref()
                        }
                        _ => None,
                    }
                    .ok_or_else(|| "multi-row gate/up pipeline is missing".to_owned())?;
                    self.encoder.setComputePipelineState(pipeline);
                } else if prepared.inline_layout {
                    if inline_input_cache && moe_inline_vec4_enabled() {
                        self.encoder
                            .setComputePipelineState(&prepared.inline_gate_up_vec4_pipeline);
                    } else if inline_input_cache {
                        self.encoder
                            .setComputePipelineState(&prepared.inline_gate_up_pipeline);
                    } else {
                        self.encoder
                            .setComputePipelineState(&prepared.inline_gate_up_nocache_pipeline);
                    }
                } else if wide_expert {
                    self.encoder
                        .setComputePipelineState(&prepared.wide_gate_up_pipeline);
                } else {
                    self.encoder
                        .setComputePipelineState(&prepared.batched_gate_up_pipeline);
                }
            }
            unsafe {
                if let (Some(packed), Some(scales), Some(block_bytes)) = (
                    prepared.q3_gate_up_packed.as_ref(),
                    prepared.q3_gate_up_scales.as_ref(),
                    q3_block_bytes,
                ) {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&packed.buffer), packed.offset, 0);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&scales.buffer), scales.offset, 1);
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&block_bytes).cast(),
                        std::mem::size_of_val(&block_bytes),
                        7,
                    );
                } else {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&prepared.gate_up_packed.buffer),
                        prepared.gate_up_packed.offset,
                        0,
                    );
                    if !decode_direct {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&prepared.gate_up_scales.buffer),
                            prepared.gate_up_scales.offset,
                            1,
                        );
                        self.encoder.setBytes_length_atIndex(
                            NonNull::from(&gate_up_layout).cast(),
                            std::mem::size_of_val(&gate_up_layout),
                            7,
                        );
                    }
                }
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_ids.buffer),
                    expert_ids.offset,
                    5,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&gate_up.buffer), gate_up.offset, 6);
            }
            if expert_wave {
                for slot_base in (0..8).step_by(expert_wave_width) {
                    unsafe {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_ids.buffer),
                            expert_ids.offset + slot_base * std::mem::size_of::<u32>(),
                            5,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&gate_up.buffer),
                            gate_up.offset + slot_base * 1024 * std::mem::size_of::<f32>(),
                            6,
                        );
                    }
                    self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: expert_wave_width * (1024 / 8),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
            } else {
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: if q4k_mature_gate_up {
                            8 * (1024 / 8)
                        } else if fused_block_active && fused_block_mapping == Some(true) {
                            8 * (1024 / 4)
                        } else if fused_block_active {
                            8 * (1024 / 8)
                        } else if llama_like_mapping {
                            8 * (1024 / 4)
                        } else if two_row_dataflow {
                            8 * (1024 / 4)
                        } else if decode_direct {
                            8 * (1024 / 2)
                        } else if cross_expert_tile8 {
                            1024 / 8
                        } else if cross_expert_activation {
                            1024 / 8
                        } else if row_inline_streaming {
                            1024 / 8
                        } else if wide_expert {
                            8
                        } else if prepared.inline_layout
                            && inline_input_cache
                            && moe_inline_vec4_enabled()
                        {
                            8 * (1024 / 8)
                        } else {
                            8 * (1024 / 8)
                        },
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: if q4k_mature_gate_up {
                            128
                        } else if fused_block_active && fused_block_mapping == Some(true) {
                            64
                        } else if fused_block_active {
                            256
                        } else if llama_like_mapping {
                            64
                        } else if two_row_dataflow {
                            64
                        } else if decode_direct {
                            64
                        } else if wide_expert {
                            1024
                        } else {
                            256
                        },
                        height: 1,
                        depth: 1,
                    },
                );
            }
            self.sample_stage_boundary();

            self.encoder
                .setComputePipelineState(&prepared.batched_silu_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&gate_up.buffer), gate_up.offset, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&activated.buffer), activated.offset, 1);
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 8 * 512,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.sample_stage_boundary();

            if let Some(pipeline) = prepared.q3_down_pipeline.as_ref() {
                self.encoder.setComputePipelineState(pipeline);
            } else if q4k_mature_down {
                self.encoder
                    .setComputePipelineState(&prepared.q4k_mature_down_pipeline);
            } else if decode_direct {
                self.encoder
                    .setComputePipelineState(&prepared.decode_direct_down_pipeline);
            } else if cross_expert_tile8_down {
                self.encoder
                    .setComputePipelineState(&prepared.cross_expert_tile8_down_pipeline);
            } else if prepared.down_inline_layout {
                self.encoder
                    .setComputePipelineState(&prepared.inline_down_pipeline);
            } else if wide_expert {
                self.encoder
                    .setComputePipelineState(&prepared.wide_down_pipeline);
            } else {
                self.encoder
                    .setComputePipelineState(&prepared.batched_down_pipeline);
            }
            unsafe {
                if let (Some(packed), Some(scales), Some(block_bytes)) = (
                    prepared.q3_down_packed.as_ref(),
                    prepared.q3_down_scales.as_ref(),
                    prepared.q3_down_block_bytes,
                ) {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&packed.buffer), packed.offset, 0);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&scales.buffer), scales.offset, 1);
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&block_bytes).cast(),
                        std::mem::size_of_val(&block_bytes),
                        7,
                    );
                } else {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&prepared.down_packed.buffer),
                        prepared.down_packed.offset,
                        0,
                    );
                    if !decode_direct {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(if cross_expert_tile8_down {
                                &prepared.down_packed.buffer
                            } else {
                                &prepared.down_scales.buffer
                            }),
                            if cross_expert_tile8_down {
                                prepared.down_packed.offset
                            } else {
                                prepared.down_scales.offset
                            },
                            1,
                        );
                        self.encoder.setBytes_length_atIndex(
                            NonNull::from(&down_layout).cast(),
                            std::mem::size_of_val(&down_layout),
                            7,
                        );
                    }
                }
                self.encoder
                    .setBuffer_offset_atIndex(Some(&activated.buffer), activated.offset, 4);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_ids.buffer),
                    expert_ids.offset,
                    5,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_output.buffer),
                    expert_output.offset,
                    6,
                );
            }
            if expert_wave {
                for slot_base in (0..8).step_by(expert_wave_width) {
                    unsafe {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&activated.buffer),
                            activated.offset + slot_base * 512 * std::mem::size_of::<f32>(),
                            4,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_ids.buffer),
                            expert_ids.offset + slot_base * std::mem::size_of::<u32>(),
                            5,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_output.buffer),
                            expert_output.offset + slot_base * 1024 * std::mem::size_of::<f32>(),
                            6,
                        );
                    }
                    self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: expert_wave_width * (1024 / 8),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
            } else {
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: if q4k_mature_down {
                            8 * (1024 / 8)
                        } else if decode_direct {
                            8 * (1024 / 2)
                        } else if wide_expert {
                            8
                        } else {
                            8 * (1024 / 8)
                        },
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: if q4k_mature_down {
                            256
                        } else if decode_direct {
                            64
                        } else if wide_expert {
                            1024
                        } else {
                            256
                        },
                        height: 1,
                        depth: 1,
                    },
                );
            }
            self.sample_stage_boundary();

            self.encoder
                .setComputePipelineState(&prepared.batched_accumulate_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_output.buffer),
                    expert_output.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_weights.buffer),
                    expert_weights.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 2);
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.sample_stage_boundary();
            self.dispatches += (if expert_wave {
                2 * (8 / expert_wave_width) + 2
            } else {
                4
            }) as u64;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_dynamic_q4_moe_batched_split_with_norm(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            norm: Option<(&RuntimeBuffer, &RuntimeBuffer)>,
        ) -> Result<(), String> {
            if input.length < 1024 * std::mem::size_of::<f32>()
                || expert_ids.length < 8 * std::mem::size_of::<u32>()
                || expert_weights.length < 8 * std::mem::size_of::<f32>()
                || output.length < 1024 * std::mem::size_of::<f32>()
                || scratch.length < 8 * 1024 * 4 + 8 * 512 * 4 + 8 * 1024 * 4
            {
                return Err(
                    "V2 split batched SIMD MoE buffers are smaller than their fixed shape"
                        .to_owned(),
                );
            }
            let gate_up = scratch.subregion(0, 8 * 1024 * 4)?;
            let activated = scratch.subregion(8 * 1024 * 4, 8 * 512 * 4)?;
            let expert_output = scratch.subregion(8 * 1024 * 4 + 8 * 512 * 4, 8 * 1024 * 4)?;
            let gate_layout = prepared.gate_up_layout as u32;
            let down_layout = prepared.down_layout as u32;
            let gate_packed = prepared
                .split_gate_packed
                .as_ref()
                .ok_or_else(|| "split gate buffer is not prepared".to_owned())?;
            let gate_scales = prepared
                .split_gate_scales
                .as_ref()
                .ok_or_else(|| "split gate scales are not prepared".to_owned())?;
            let up_packed = prepared
                .split_up_packed
                .as_ref()
                .ok_or_else(|| "split up buffer is not prepared".to_owned())?;
            let up_scales = prepared
                .split_up_scales
                .as_ref()
                .ok_or_else(|| "split up scales are not prepared".to_owned())?;

            if let Some((norm_weights, inverse_rms)) = norm
                && (norm_weights.length < 1024 * 4
                    || inverse_rms.length < std::mem::size_of::<f32>())
            {
                return Err("V2 normalized split MoE buffers are too small".to_owned());
            }

            for (packed, scales, output_base) in [
                (gate_packed, gate_scales, 0_u32),
                (up_packed, up_scales, 512_u32),
            ] {
                if norm.is_some() {
                    self.encoder
                        .setComputePipelineState(&prepared.batched_projection_norm_pipeline);
                } else {
                    self.encoder
                        .setComputePipelineState(&prepared.batched_projection_pipeline);
                }
                unsafe {
                    self.encoder.setBuffer_offset_atIndex(Some(packed), 0, 0);
                    self.encoder.setBuffer_offset_atIndex(Some(scales), 0, 1);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&expert_ids.buffer),
                        expert_ids.offset,
                        5,
                    );
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&gate_up.buffer), gate_up.offset, 6);
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&gate_layout).cast(),
                        std::mem::size_of_val(&gate_layout),
                        7,
                    );
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&output_base).cast(),
                        std::mem::size_of_val(&output_base),
                        8,
                    );
                    if let Some((norm_weights, inverse_rms)) = norm {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&norm_weights.buffer),
                            norm_weights.offset,
                            9,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&inverse_rms.buffer),
                            inverse_rms.offset,
                            10,
                        );
                    }
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 8 * (512 / 8),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            }

            self.encoder
                .setComputePipelineState(&prepared.batched_silu_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&gate_up.buffer), gate_up.offset, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&activated.buffer), activated.offset, 1);
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 8 * 512,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );

            self.encoder
                .setComputePipelineState(&prepared.batched_down_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.down_packed.buffer),
                    prepared.down_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.down_scales.buffer),
                    prepared.down_scales.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&activated.buffer), activated.offset, 4);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_ids.buffer),
                    expert_ids.offset,
                    5,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_output.buffer),
                    expert_output.offset,
                    6,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&down_layout).cast(),
                    std::mem::size_of_val(&down_layout),
                    7,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 8 * (1024 / 8),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );

            self.encoder
                .setComputePipelineState(&prepared.batched_accumulate_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_output.buffer),
                    expert_output.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_weights.buffer),
                    expert_weights.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 2);
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 5;
            Ok(())
        }

        /// Encode one stage of the batched MoE for isolated profiling. The
        /// runtime uses `encode_dynamic_q4_moe_batched_simd`; this entry point
        /// keeps the four measured stages visible without changing the graph.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_dynamic_q4_moe_batched_stage(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            stage: u8,
        ) -> Result<(), String> {
            self.encode_dynamic_q4_moe_batched_stage_with_norm(
                prepared,
                input,
                expert_ids,
                expert_weights,
                scratch,
                output,
                stage,
                None,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_dynamic_q4_moe_batched_stage_normalized(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            norm_weights: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            stage: u8,
        ) -> Result<(), String> {
            self.encode_dynamic_q4_moe_batched_stage_with_norm(
                prepared,
                input,
                expert_ids,
                expert_weights,
                scratch,
                output,
                stage,
                Some((norm_weights, inverse_rms)),
            )
        }

        /// Benchmark-only control that traverses the selected experts' gate/up
        /// Q4 payload and scales without performing dot products.
        pub fn encode_dynamic_q4_moe_gate_up_touch(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            expert_ids: &RuntimeBuffer,
            contiguous: &RuntimeBuffer,
            sink: &RuntimeBuffer,
            mode: u32,
        ) -> Result<(), String> {
            if mode == 0 {
                self.encoder
                    .setComputePipelineState(&prepared.batched_gate_up_touch_full_pipeline);
            } else {
                self.encoder
                    .setComputePipelineState(&prepared.batched_gate_up_touch_pipeline);
            }
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_packed.buffer),
                    prepared.gate_up_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_scales.buffer),
                    prepared.gate_up_scales.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&contiguous.buffer),
                    contiguous.offset,
                    2,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_ids.buffer),
                    expert_ids.offset,
                    5,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&sink.buffer), sink.offset, 6);
                if mode != 0 {
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&mode).cast(),
                        std::mem::size_of_val(&mode),
                        7,
                    );
                }
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 8 * (1024 * 1024 / 2 + 1024 * 32 * 2),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_dynamic_q4_moe_expertwise_with_norm(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            norm: Option<(&RuntimeBuffer, &RuntimeBuffer)>,
        ) -> Result<(), String> {
            if input.length < 1024 * 4
                || expert_ids.length < 8 * 4
                || expert_weights.length < 8 * 4
                || output.length < 1024 * 4
                || scratch.length < 8 * 1024 * 4 + 8 * 512 * 4 + 8 * 1024 * 4
            {
                return Err("expertwise MoE buffers are smaller than their fixed shape".to_owned());
            }
            let gate_up = scratch.subregion(0, 8 * 1024 * 4)?;
            let activated = scratch.subregion(8 * 1024 * 4, 8 * 512 * 4)?;
            let expert_output = scratch.subregion(8 * 1024 * 4 + 8 * 512 * 4, 8 * 1024 * 4)?;
            let map = prepared
                .expertwise_gate_up_map
                .as_ref()
                .ok_or_else(|| "expertwise gate/up map is missing".to_owned())?;
            let down_map = prepared
                .expertwise_down_map
                .as_ref()
                .ok_or_else(|| "expertwise down map is missing".to_owned())?;

            let q4_gate = prepared
                .expertwise_q4_gate_up_packed
                .as_ref()
                .ok_or_else(|| "expertwise Q4 gate/up slab is missing".to_owned())?;
            let q4_gate_scales = prepared
                .expertwise_q4_gate_up_scales
                .as_ref()
                .ok_or_else(|| "expertwise Q4 gate/up scales are missing".to_owned())?;
            let q4_down = prepared
                .expertwise_q4_down_packed
                .as_ref()
                .ok_or_else(|| "expertwise Q4 down slab is missing".to_owned())?;
            let q4_down_scales = prepared
                .expertwise_q4_down_scales
                .as_ref()
                .ok_or_else(|| "expertwise Q4 down scales are missing".to_owned())?;
            let q3_gate = prepared
                .expertwise_q3_gate_up_packed
                .as_ref()
                .ok_or_else(|| "expertwise Q3 gate/up slab is missing".to_owned())?;
            let q3_gate_scales = prepared
                .expertwise_q3_gate_up_scales
                .as_ref()
                .ok_or_else(|| "expertwise Q3 gate/up scales are missing".to_owned())?;
            let q3_down = prepared
                .expertwise_q3_down_packed
                .as_ref()
                .ok_or_else(|| "expertwise Q3 down slab is missing".to_owned())?;
            let q3_down_scales = prepared
                .expertwise_q3_down_scales
                .as_ref()
                .ok_or_else(|| "expertwise Q3 down scales are missing".to_owned())?;

            let dispatch_projection = |this: &mut Self,
                                       pipeline: &Pipeline,
                                       packed: &RuntimeBuffer,
                                       scales: &RuntimeBuffer,
                                       block_bytes: Option<u32>,
                                       norm: Option<(&RuntimeBuffer, &RuntimeBuffer)>|
             -> Result<(), String> {
                this.encoder.setComputePipelineState(pipeline);
                unsafe {
                    this.encoder
                        .setBuffer_offset_atIndex(Some(&packed.buffer), packed.offset, 0);
                    this.encoder
                        .setBuffer_offset_atIndex(Some(&scales.buffer), scales.offset, 1);
                    this.encoder
                        .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                    this.encoder.setBuffer_offset_atIndex(
                        Some(&expert_ids.buffer),
                        expert_ids.offset,
                        5,
                    );
                    this.encoder
                        .setBuffer_offset_atIndex(Some(&gate_up.buffer), gate_up.offset, 6);
                    this.encoder
                        .setBuffer_offset_atIndex(Some(&map.buffer), map.offset, 10);
                    if let Some(block_bytes) = block_bytes {
                        this.encoder.setBytes_length_atIndex(
                            NonNull::from(&block_bytes).cast(),
                            std::mem::size_of_val(&block_bytes),
                            7,
                        );
                    }
                    if let Some((weights, inverse)) = norm {
                        this.encoder.setBuffer_offset_atIndex(
                            Some(&weights.buffer),
                            weights.offset,
                            8,
                        );
                        this.encoder.setBuffer_offset_atIndex(
                            Some(&inverse.buffer),
                            inverse.offset,
                            9,
                        );
                    }
                }
                this.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 8 * (1024 / 8),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                this.dispatches += 1;
                Ok(())
            };

            if let Some((weights, inverse)) = norm {
                dispatch_projection(
                    self,
                    prepared
                        .expertwise_q4_gate_up_norm_pipeline
                        .as_ref()
                        .unwrap(),
                    q4_gate,
                    q4_gate_scales,
                    None,
                    Some((weights, inverse)),
                )?;
            } else {
                dispatch_projection(
                    self,
                    prepared.expertwise_q4_gate_up_pipeline.as_ref().unwrap(),
                    q4_gate,
                    q4_gate_scales,
                    None,
                    None,
                )?;
            }
            if let Some(block_bytes) = prepared.expertwise_q3_gate_up_block_bytes {
                if let Some((weights, inverse)) = norm {
                    dispatch_projection(
                        self,
                        prepared
                            .expertwise_q3_gate_up_norm_pipeline
                            .as_ref()
                            .unwrap(),
                        q3_gate,
                        q3_gate_scales,
                        Some(block_bytes),
                        Some((weights, inverse)),
                    )?;
                } else {
                    dispatch_projection(
                        self,
                        prepared.expertwise_q3_gate_up_pipeline.as_ref().unwrap(),
                        q3_gate,
                        q3_gate_scales,
                        Some(block_bytes),
                        None,
                    )?;
                }
            }

            self.encoder
                .setComputePipelineState(&prepared.batched_silu_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&gate_up.buffer), gate_up.offset, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&activated.buffer), activated.offset, 1);
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 8 * 512,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;

            let dispatch_down = |this: &mut Self,
                                 pipeline: &Pipeline,
                                 packed: &RuntimeBuffer,
                                 scales: &RuntimeBuffer,
                                 block_bytes: Option<u32>|
             -> Result<(), String> {
                this.encoder.setComputePipelineState(pipeline);
                unsafe {
                    this.encoder
                        .setBuffer_offset_atIndex(Some(&packed.buffer), packed.offset, 0);
                    this.encoder
                        .setBuffer_offset_atIndex(Some(&scales.buffer), scales.offset, 1);
                    this.encoder.setBuffer_offset_atIndex(
                        Some(&activated.buffer),
                        activated.offset,
                        4,
                    );
                    this.encoder.setBuffer_offset_atIndex(
                        Some(&expert_ids.buffer),
                        expert_ids.offset,
                        5,
                    );
                    this.encoder.setBuffer_offset_atIndex(
                        Some(&expert_output.buffer),
                        expert_output.offset,
                        6,
                    );
                    this.encoder.setBuffer_offset_atIndex(
                        Some(&down_map.buffer),
                        down_map.offset,
                        10,
                    );
                    if let Some(block_bytes) = block_bytes {
                        this.encoder.setBytes_length_atIndex(
                            NonNull::from(&block_bytes).cast(),
                            std::mem::size_of_val(&block_bytes),
                            7,
                        );
                    }
                }
                this.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 8 * (1024 / 8),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
                this.dispatches += 1;
                Ok(())
            };
            dispatch_down(
                self,
                prepared.expertwise_q4_down_pipeline.as_ref().unwrap(),
                q4_down,
                q4_down_scales,
                None,
            )?;
            if let Some(block_bytes) = prepared.expertwise_q3_down_block_bytes {
                dispatch_down(
                    self,
                    prepared.expertwise_q3_down_pipeline.as_ref().unwrap(),
                    q3_down,
                    q3_down_scales,
                    Some(block_bytes),
                )?;
            }

            self.encoder
                .setComputePipelineState(&prepared.batched_accumulate_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_output.buffer),
                    expert_output.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_weights.buffer),
                    expert_weights.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 2);
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        /// Benchmark-only direct-address traffic controls. The host resolves
        /// the fixed route once and binds each selected expert's packed and
        /// scale base directly, so the kernel does not read expert IDs or do
        /// routed address lookup during the byte traversal.
        pub fn encode_dynamic_q4_moe_gate_up_touch_direct(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            expert_ids: &[u32; 8],
            sink: &RuntimeBuffer,
            region_count: u32,
        ) -> Result<(), String> {
            const PACKED_STRIDE: usize = 1024 * 1024 / 2;
            const SCALE_STRIDE: usize = 1024 * 32 * 2;
            let (pipeline, regions) = match region_count {
                1 => (&prepared.batched_gate_up_touch_direct_1_pipeline, 1_usize),
                2 => (&prepared.batched_gate_up_touch_direct_2_pipeline, 2_usize),
                4 => (&prepared.batched_gate_up_touch_direct_4_pipeline, 4_usize),
                8 => (&prepared.batched_gate_up_touch_direct_8_pipeline, 8_usize),
                _ => return Err("direct gate/up traffic requires 1, 2, 4, or 8 regions".to_owned()),
            };
            self.encoder.setComputePipelineState(pipeline);
            unsafe {
                for (region, &expert_id) in expert_ids.iter().enumerate().take(regions) {
                    let expert = expert_id as usize;
                    if expert >= 32 {
                        return Err(
                            "direct gate/up traffic route contains an invalid expert".to_owned()
                        );
                    }
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&prepared.gate_up_packed.buffer),
                        prepared.gate_up_packed.offset + expert * PACKED_STRIDE,
                        2 * region,
                    );
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&prepared.gate_up_scales.buffer),
                        prepared.gate_up_scales.offset + expert * SCALE_STRIDE,
                        2 * region + 1,
                    );
                }
                self.encoder
                    .setBuffer_offset_atIndex(Some(&sink.buffer), sink.offset, 16);
            }
            let expert_bytes = PACKED_STRIDE + SCALE_STRIDE;
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 8 * expert_bytes,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        /// Benchmark-only arithmetic control with the same selected-expert ×
        /// output-row geometry as canonical gate/up but synthetic weights.
        pub fn encode_dynamic_q4_moe_gate_up_synthetic(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            self.encoder
                .setComputePipelineState(&prepared.batched_gate_up_synthetic_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_ids.buffer),
                    expert_ids.offset,
                    5,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 6);
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 8 * (1024 / 8),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_dynamic_q4_moe_gate_up_synthetic_normalized(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            norm_weights: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            self.encoder
                .setComputePipelineState(&prepared.batched_gate_up_synthetic_norm_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_ids.buffer),
                    expert_ids.offset,
                    5,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 6);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&norm_weights.buffer),
                    norm_weights.offset,
                    8,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&inverse_rms.buffer),
                    inverse_rms.offset,
                    9,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 8 * (1024 / 8),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_dynamic_q4_moe_batched_stage_with_norm(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            stage: u8,
            norm: Option<(&RuntimeBuffer, &RuntimeBuffer)>,
        ) -> Result<(), String> {
            if prepared.split_gate_up {
                return self.encode_dynamic_q4_moe_batched_split_stage_with_norm(
                    prepared,
                    input,
                    expert_ids,
                    expert_weights,
                    scratch,
                    output,
                    stage,
                    norm,
                );
            }
            if input.length < 1024 * 4
                || expert_ids.length < 8 * 4
                || expert_weights.length < 8 * 4
                || output.length < 1024 * 4
                || scratch.length < 8 * 1024 * 4 + 8 * 512 * 4 + 8 * 1024 * 4
            {
                return Err("V2 batched MoE profiling buffers are too small".to_owned());
            }
            let gate_up = scratch.subregion(0, 8 * 1024 * 4)?;
            let activated = scratch.subregion(8 * 1024 * 4, 8 * 512 * 4)?;
            let expert_output = scratch.subregion(8 * 1024 * 4 + 8 * 512 * 4, 8 * 1024 * 4)?;
            let gate_up_layout = prepared.gate_up_layout as u32;
            let down_layout = prepared.down_layout as u32;
            let q3_block_bytes = prepared.q3_gate_up_block_bytes;
            let is_q4k = prepared.gate_up_packed.length == 32 * 1024 * 4 * 140
                && prepared.gate_up_scales.length == 32 * 1024 * 4 * 2 * 2
                && prepared.down_packed.length == 32 * 1024 * 2 * 140
                && prepared.down_scales.length == 32 * 1024 * 2 * 2 * 2;
            let q4k_mature_gate_up = is_q4k && moe_q4k_mature_gate_up_enabled();
            let q4k_mature_down = is_q4k && moe_q4k_mature_down_enabled();
            let wide_expert =
                moe_wide_expert_enabled() && prepared.wide_supported && q3_block_bytes.is_none();
            let row_inline_streaming = moe_row_inline_streaming_enabled()
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let decode_direct = moe_decode_direct_enabled()
                && prepared.inline_layout
                && prepared.down_inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && prepared.down_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let two_row_dataflow = moe_two_row_dataflow_enabled()
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let cross_expert_tile8 = moe_cross_expert_tile8_enabled()
                && prepared.gate_up_layout == PackingLayout::Q4CrossExpertTile8
                && q3_block_bytes.is_none();
            let cross_expert_tile8_down = moe_cross_expert_tile8_enabled()
                && prepared.down_layout == PackingLayout::Q4CrossExpertTile8
                && q3_block_bytes.is_none();
            match stage {
                0 => {
                    if let Some((norm_weights, inverse_rms)) = norm {
                        if let Some(pipeline) = prepared.q3_gate_up_norm_pipeline.as_ref() {
                            self.encoder.setComputePipelineState(pipeline);
                        } else if q4k_mature_gate_up {
                            self.encoder.setComputePipelineState(
                                &prepared.q4k_mature_gate_up_norm_pipeline,
                            );
                        } else if two_row_dataflow {
                            self.encoder.setComputePipelineState(
                                prepared
                                    .two_row_dataflow_gate_up_norm_pipeline
                                    .as_ref()
                                    .ok_or_else(|| {
                                        "two-row dataflow pipeline is missing".to_owned()
                                    })?,
                            );
                        } else if decode_direct {
                            self.encoder.setComputePipelineState(
                                &prepared.decode_direct_gate_up_norm_pipeline,
                            );
                        } else if cross_expert_tile8 {
                            self.encoder.setComputePipelineState(
                                &prepared.cross_expert_tile8_gate_up_norm_pipeline,
                            );
                        } else if row_inline_streaming {
                            self.encoder.setComputePipelineState(
                                &prepared.row_stream_gate_up_norm_pipeline,
                            );
                        } else if prepared
                            .gate_up_layout
                            .row_inline_packet_rows()
                            .is_some_and(|rows| rows > 1)
                        {
                            let pipeline = match prepared.gate_up_layout {
                                PackingLayout::Q4RowInlineRows2 => {
                                    prepared.inline_gate_up_rows2_norm_pipeline.as_ref()
                                }
                                PackingLayout::Q4RowInlineRows4 => {
                                    prepared.inline_gate_up_rows4_norm_pipeline.as_ref()
                                }
                                PackingLayout::Q4RowInlineRows8 => {
                                    prepared.inline_gate_up_rows8_norm_pipeline.as_ref()
                                }
                                _ => None,
                            }
                            .ok_or_else(|| "multi-row gate/up pipeline is missing".to_owned())?;
                            self.encoder.setComputePipelineState(pipeline);
                        } else if prepared.inline_layout {
                            if moe_inline_input_cache_enabled() && moe_inline_vec4_enabled() {
                                self.encoder.setComputePipelineState(
                                    &prepared.inline_gate_up_vec4_norm_pipeline,
                                );
                            } else {
                                self.encoder.setComputePipelineState(
                                    &prepared.inline_gate_up_norm_pipeline,
                                );
                            }
                        } else if wide_expert {
                            self.encoder
                                .setComputePipelineState(&prepared.wide_gate_up_norm_pipeline);
                        } else {
                            self.encoder
                                .setComputePipelineState(&prepared.batched_gate_up_norm_pipeline);
                        }
                        unsafe {
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&norm_weights.buffer),
                                norm_weights.offset,
                                8,
                            );
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&inverse_rms.buffer),
                                inverse_rms.offset,
                                9,
                            );
                        }
                    } else {
                        if prepared.gate_up_layout.is_q4_transaction() {
                            self.encoder.setComputePipelineState(
                                prepared
                                    .fused_block_qx_gate_up_pipeline
                                    .as_ref()
                                    .ok_or_else(|| {
                                        "fused transaction-block pipeline is missing".to_owned()
                                    })?,
                            );
                        } else if let Some(pipeline) = prepared.q3_gate_up_pipeline.as_ref() {
                            self.encoder.setComputePipelineState(pipeline);
                        } else if q4k_mature_gate_up {
                            self.encoder
                                .setComputePipelineState(&prepared.q4k_mature_gate_up_pipeline);
                        } else if two_row_dataflow {
                            self.encoder.setComputePipelineState(
                                prepared
                                    .two_row_dataflow_gate_up_pipeline
                                    .as_ref()
                                    .ok_or_else(|| {
                                        "two-row dataflow pipeline is missing".to_owned()
                                    })?,
                            );
                        } else if decode_direct {
                            self.encoder
                                .setComputePipelineState(&prepared.decode_direct_gate_up_pipeline);
                        } else if cross_expert_tile8 {
                            self.encoder.setComputePipelineState(
                                &prepared.cross_expert_tile8_gate_up_pipeline,
                            );
                        } else if row_inline_streaming {
                            self.encoder
                                .setComputePipelineState(&prepared.row_stream_gate_up_pipeline);
                        } else if prepared
                            .gate_up_layout
                            .row_inline_packet_rows()
                            .is_some_and(|rows| rows > 1)
                        {
                            let pipeline = match prepared.gate_up_layout {
                                PackingLayout::Q4RowInlineRows2 => {
                                    prepared.inline_gate_up_rows2_pipeline.as_ref()
                                }
                                PackingLayout::Q4RowInlineRows4 => {
                                    prepared.inline_gate_up_rows4_pipeline.as_ref()
                                }
                                PackingLayout::Q4RowInlineRows8 => {
                                    prepared.inline_gate_up_rows8_pipeline.as_ref()
                                }
                                _ => None,
                            }
                            .ok_or_else(|| "multi-row gate/up pipeline is missing".to_owned())?;
                            self.encoder.setComputePipelineState(pipeline);
                        } else if prepared.inline_layout {
                            if moe_inline_input_cache_enabled() && moe_inline_vec4_enabled() {
                                self.encoder.setComputePipelineState(
                                    &prepared.inline_gate_up_vec4_pipeline,
                                );
                            } else {
                                self.encoder
                                    .setComputePipelineState(&prepared.inline_gate_up_pipeline);
                            }
                        } else if wide_expert {
                            self.encoder
                                .setComputePipelineState(&prepared.wide_gate_up_pipeline);
                        } else {
                            self.encoder
                                .setComputePipelineState(&prepared.batched_gate_up_pipeline);
                        }
                    }
                    unsafe {
                        if let (Some(packed), Some(scales), Some(block_bytes)) = (
                            prepared.q3_gate_up_packed.as_ref(),
                            prepared.q3_gate_up_scales.as_ref(),
                            q3_block_bytes,
                        ) {
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&packed.buffer),
                                packed.offset,
                                0,
                            );
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&scales.buffer),
                                scales.offset,
                                1,
                            );
                            self.encoder.setBytes_length_atIndex(
                                NonNull::from(&block_bytes).cast(),
                                std::mem::size_of_val(&block_bytes),
                                7,
                            );
                        } else {
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&prepared.gate_up_packed.buffer),
                                prepared.gate_up_packed.offset,
                                0,
                            );
                            if !decode_direct {
                                self.encoder.setBuffer_offset_atIndex(
                                    Some(&prepared.gate_up_scales.buffer),
                                    prepared.gate_up_scales.offset,
                                    1,
                                );
                                self.encoder.setBytes_length_atIndex(
                                    NonNull::from(&gate_up_layout).cast(),
                                    4,
                                    7,
                                );
                            }
                        }
                        self.encoder
                            .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_ids.buffer),
                            expert_ids.offset,
                            5,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&gate_up.buffer),
                            gate_up.offset,
                            6,
                        );
                    }
                    self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: if q4k_mature_gate_up {
                                8 * (1024 / 8)
                            } else if two_row_dataflow {
                                8 * (1024 / 4)
                            } else if decode_direct {
                                8 * (1024 / 2)
                            } else if cross_expert_tile8 || row_inline_streaming {
                                1024 / 8
                            } else if wide_expert {
                                8
                            } else if prepared.inline_layout
                                && moe_inline_input_cache_enabled()
                                && moe_inline_vec4_enabled()
                            {
                                1024
                            } else {
                                1024
                            },
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: if q4k_mature_gate_up {
                                128
                            } else if two_row_dataflow {
                                64
                            } else if decode_direct {
                                64
                            } else if wide_expert {
                                1024
                            } else {
                                256
                            },
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                1 => {
                    self.encoder
                        .setComputePipelineState(&prepared.batched_silu_pipeline);
                    unsafe {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&gate_up.buffer),
                            gate_up.offset,
                            0,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&activated.buffer),
                            activated.offset,
                            1,
                        );
                    }
                    self.encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize {
                            width: 8 * 512,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                2 => {
                    if let Some(pipeline) = prepared.q3_down_pipeline.as_ref() {
                        self.encoder.setComputePipelineState(pipeline);
                    } else if q4k_mature_down {
                        self.encoder
                            .setComputePipelineState(&prepared.q4k_mature_down_pipeline);
                    } else if decode_direct {
                        self.encoder
                            .setComputePipelineState(&prepared.decode_direct_down_pipeline);
                    } else if cross_expert_tile8_down {
                        self.encoder
                            .setComputePipelineState(&prepared.cross_expert_tile8_down_pipeline);
                    } else if prepared.down_inline_layout {
                        self.encoder
                            .setComputePipelineState(&prepared.inline_down_pipeline);
                    } else if wide_expert {
                        self.encoder
                            .setComputePipelineState(&prepared.wide_down_pipeline);
                    } else {
                        self.encoder
                            .setComputePipelineState(&prepared.batched_down_pipeline);
                    }
                    unsafe {
                        if let (Some(packed), Some(scales), Some(block_bytes)) = (
                            prepared.q3_down_packed.as_ref(),
                            prepared.q3_down_scales.as_ref(),
                            prepared.q3_down_block_bytes,
                        ) {
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&packed.buffer),
                                packed.offset,
                                0,
                            );
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&scales.buffer),
                                scales.offset,
                                1,
                            );
                            self.encoder.setBytes_length_atIndex(
                                NonNull::from(&block_bytes).cast(),
                                4,
                                7,
                            );
                        } else if decode_direct {
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&prepared.down_packed.buffer),
                                prepared.down_packed.offset,
                                0,
                            );
                        } else if cross_expert_tile8_down {
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&prepared.down_packed.buffer),
                                prepared.down_packed.offset,
                                0,
                            );
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&prepared.down_packed.buffer),
                                prepared.down_packed.offset,
                                1,
                            );
                        } else {
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&prepared.down_packed.buffer),
                                prepared.down_packed.offset,
                                0,
                            );
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&prepared.down_scales.buffer),
                                prepared.down_scales.offset,
                                1,
                            );
                            self.encoder.setBytes_length_atIndex(
                                NonNull::from(&down_layout).cast(),
                                4,
                                7,
                            );
                        }
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&activated.buffer),
                            activated.offset,
                            4,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_ids.buffer),
                            expert_ids.offset,
                            5,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_output.buffer),
                            expert_output.offset,
                            6,
                        );
                        self.encoder.setBytes_length_atIndex(
                            NonNull::from(&down_layout).cast(),
                            4,
                            7,
                        );
                    }
                    self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: if q4k_mature_down {
                                8 * (1024 / 8)
                            } else if decode_direct {
                                8 * (1024 / 2)
                            } else if wide_expert {
                                8
                            } else {
                                1024
                            },
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: if q4k_mature_down {
                                256
                            } else if decode_direct {
                                64
                            } else if wide_expert {
                                1024
                            } else {
                                256
                            },
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                3 => {
                    self.encoder
                        .setComputePipelineState(&prepared.batched_accumulate_pipeline);
                    unsafe {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_output.buffer),
                            expert_output.offset,
                            0,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_weights.buffer),
                            expert_weights.offset,
                            1,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&output.buffer),
                            output.offset,
                            2,
                        );
                    }
                    self.encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize {
                            width: 1024,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                _ => return Err("unknown batched MoE profiling stage".to_owned()),
            }
            self.dispatches += 1;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_dynamic_q4_moe_batched_split_stage_with_norm(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            stage: u8,
            norm: Option<(&RuntimeBuffer, &RuntimeBuffer)>,
        ) -> Result<(), String> {
            if input.length < 1024 * 4
                || expert_ids.length < 8 * 4
                || expert_weights.length < 8 * 4
                || output.length < 1024 * 4
                || scratch.length < 8 * 1024 * 4 + 8 * 512 * 4 + 8 * 1024 * 4
            {
                return Err("V2 split batched MoE profiling buffers are too small".to_owned());
            }
            let gate_up = scratch.subregion(0, 8 * 1024 * 4)?;
            let activated = scratch.subregion(8 * 1024 * 4, 8 * 512 * 4)?;
            let expert_output = scratch.subregion(8 * 1024 * 4 + 8 * 512 * 4, 8 * 1024 * 4)?;
            match stage {
                0 | 1 => {
                    let (packed, scales, output_base) =
                        if stage == 0 {
                            (
                                prepared.split_gate_packed.as_ref().ok_or_else(|| {
                                    "split gate buffer is not prepared".to_owned()
                                })?,
                                prepared.split_gate_scales.as_ref().ok_or_else(|| {
                                    "split gate scales are not prepared".to_owned()
                                })?,
                                0_u32,
                            )
                        } else {
                            (
                                prepared
                                    .split_up_packed
                                    .as_ref()
                                    .ok_or_else(|| "split up buffer is not prepared".to_owned())?,
                                prepared
                                    .split_up_scales
                                    .as_ref()
                                    .ok_or_else(|| "split up scales are not prepared".to_owned())?,
                                512_u32,
                            )
                        };
                    let layout = prepared.gate_up_layout as u32;
                    if let Some((norm_weights, inverse_rms)) = norm {
                        self.encoder
                            .setComputePipelineState(&prepared.batched_projection_norm_pipeline);
                        unsafe {
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&norm_weights.buffer),
                                norm_weights.offset,
                                9,
                            );
                            self.encoder.setBuffer_offset_atIndex(
                                Some(&inverse_rms.buffer),
                                inverse_rms.offset,
                                10,
                            );
                        }
                    } else {
                        self.encoder
                            .setComputePipelineState(&prepared.batched_projection_pipeline);
                    }
                    unsafe {
                        self.encoder.setBuffer_offset_atIndex(Some(packed), 0, 0);
                        self.encoder.setBuffer_offset_atIndex(Some(scales), 0, 1);
                        self.encoder
                            .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_ids.buffer),
                            expert_ids.offset,
                            5,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&gate_up.buffer),
                            gate_up.offset,
                            6,
                        );
                        self.encoder
                            .setBytes_length_atIndex(NonNull::from(&layout).cast(), 4, 7);
                        self.encoder.setBytes_length_atIndex(
                            NonNull::from(&output_base).cast(),
                            4,
                            8,
                        );
                    }
                    self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: 8 * (512 / 8),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                2 => {
                    self.encoder
                        .setComputePipelineState(&prepared.batched_silu_pipeline);
                    unsafe {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&gate_up.buffer),
                            gate_up.offset,
                            0,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&activated.buffer),
                            activated.offset,
                            1,
                        );
                    }
                    self.encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize {
                            width: 8 * 512,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                3 => {
                    self.encoder
                        .setComputePipelineState(&prepared.batched_down_pipeline);
                    let layout = prepared.down_layout as u32;
                    unsafe {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&prepared.down_packed.buffer),
                            prepared.down_packed.offset,
                            0,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&prepared.down_scales.buffer),
                            prepared.down_scales.offset,
                            1,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&activated.buffer),
                            activated.offset,
                            4,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_ids.buffer),
                            expert_ids.offset,
                            5,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_output.buffer),
                            expert_output.offset,
                            6,
                        );
                        self.encoder
                            .setBytes_length_atIndex(NonNull::from(&layout).cast(), 4, 7);
                    }
                    self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: 8 * (1024 / 8),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                4 => {
                    self.encoder
                        .setComputePipelineState(&prepared.batched_accumulate_pipeline);
                    unsafe {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_output.buffer),
                            expert_output.offset,
                            0,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&expert_weights.buffer),
                            expert_weights.offset,
                            1,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&output.buffer),
                            output.offset,
                            2,
                        );
                    }
                    self.encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize {
                            width: 1024,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                _ => return Err("unknown split batched MoE profiling stage".to_owned()),
            }
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_q4_embedding_lookup(
            &mut self,
            prepared: &PreparedQ4Embedding,
            token: &RuntimeBuffer,
            output: &RuntimeBuffer,
            multiplier: f32,
        ) -> Result<(), String> {
            if token.length < std::mem::size_of::<u32>()
                || output.length < prepared.columns * std::mem::size_of::<f32>()
            {
                return Err("V2 embedding buffer is smaller than its fixed shape".to_owned());
            }
            self.encoder.setComputePipelineState(&prepared.pipeline);
            let rows = prepared.rows as u32;
            let columns = prepared.columns as u32;
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.packed.buffer),
                    prepared.packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.scales.buffer),
                    prepared.scales.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&token.buffer), token.offset, 2);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 3);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&rows).cast(),
                    std::mem::size_of_val(&rows),
                    4,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    5,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    6,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_prepared_logits(
            &mut self,
            prepared: &PreparedLogits,
            hidden: &RuntimeBuffer,
            logits: &RuntimeBuffer,
            scaling: f32,
        ) -> Result<(), String> {
            if hidden.length < HIDDEN * std::mem::size_of::<f32>()
                || logits.length < prepared.rows * std::mem::size_of::<f32>()
            {
                return Err("V2 logits buffer is smaller than its fixed shape".to_owned());
            }
            self.encoder.setComputePipelineState(&prepared.pipeline);
            let rows = prepared.rows as u32;
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&prepared.embeddings), 0, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&hidden.buffer), hidden.offset, 1);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&logits.buffer), logits.offset, 2);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&rows).cast(),
                    std::mem::size_of_val(&rows),
                    3,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&scaling).cast(),
                    std::mem::size_of_val(&scaling),
                    4,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_argmax(
            &mut self,
            context: &MetalContext,
            logits: &RuntimeBuffer,
            partials: &RuntimeBuffer,
            token_output: &RuntimeBuffer,
            rows: usize,
        ) -> Result<(), String> {
            if logits.length < rows * std::mem::size_of::<f32>()
                || token_output.length < std::mem::size_of::<u32>()
            {
                return Err("V2 argmax buffer is smaller than its fixed shape".to_owned());
            }
            if rows == 0 {
                return Ok(());
            }
            let partial_count = rows.div_ceil(256);
            let partial_bytes = partial_count * 2 * std::mem::size_of::<u32>();
            if partials.length < partial_bytes {
                return Err("V2 argmax partial buffer is smaller than its fixed shape".to_owned());
            }
            let rows = rows as u32;
            let partial_count_u32 = partial_count as u32;

            let partial_pipeline = context.argmax_partial_pipeline()?;
            self.encoder.setComputePipelineState(&partial_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&logits.buffer), logits.offset, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&partials.buffer), partials.offset, 1);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&rows).cast(),
                    std::mem::size_of_val(&rows),
                    2,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: partial_count * 256,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;

            let reduce_pipeline = context.argmax_reduce_pipeline()?;
            self.encoder.setComputePipelineState(&reduce_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&partials.buffer), partials.offset, 0);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&token_output.buffer),
                    token_output.offset,
                    1,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&partial_count_u32).cast(),
                    std::mem::size_of_val(&partial_count_u32),
                    2,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        /// Encode one of the fixed-shape V2 decode kernels. All buffers must
        /// have been allocated before the token graph starts; this method does
        /// not create, resize, commit, or wait on anything.
        pub fn encode_decode_kernel(
            &mut self,
            context: &MetalContext,
            kernel: DecodeKernel,
            buffers: &[&RuntimeBuffer],
            params: &[u8],
            threads: MTLSize,
            group: MTLSize,
        ) -> Result<(), String> {
            let pipeline = context.decode_pipeline(kernel)?;
            if params.is_empty() {
                return Err("V2 decode kernel parameters cannot be empty".to_owned());
            }
            self.encoder.setComputePipelineState(&pipeline);
            for (index, buffer) in buffers.iter().enumerate() {
                unsafe {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
            }
            unsafe {
                self.encoder.setBytes_length_atIndex(
                    NonNull::new(params.as_ptr() as *mut c_void)
                        .ok_or_else(|| "V2 decode parameters cannot be empty".to_owned())?,
                    params.len(),
                    buffers.len(),
                );
            }
            self.encoder
                .dispatchThreads_threadsPerThreadgroup(threads, group);
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_rmsnorm(
            &mut self,
            context: &MetalContext,
            input: &RuntimeBuffer,
            weights: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                n: u32,
                epsilon: f32,
            }
            let params = Params {
                n: 1024,
                epsilon: 1e-5,
            };
            self.encode_decode_kernel(
                context,
                DecodeKernel::RmsNorm,
                &[input, weights, output],
                bytes_of(&params),
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn encode_rmsnorm_scalar(
            &mut self,
            context: &MetalContext,
            input: &RuntimeBuffer,
            weights: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                n: u32,
                epsilon: f32,
            }
            let params = Params {
                n: 1024,
                epsilon: 1e-5,
            };
            self.encode_decode_kernel(
                context,
                DecodeKernel::RmsNormScalar,
                &[input, weights, output],
                bytes_of(&params),
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn encode_f32_to_f16(
            &mut self,
            context: &MetalContext,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
            elements: usize,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                n: u32,
            }
            let params = Params { n: elements as u32 };
            self.encode_decode_kernel(
                context,
                DecodeKernel::F32ToF16,
                &[input, output],
                bytes_of(&params),
                MTLSize {
                    width: elements,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_rope_kv(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
            query_output: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            position: usize,
        ) -> Result<(), String> {
            self.encode_rope_kv_precision(
                context,
                query,
                key,
                value,
                query_output,
                key_cache,
                value_cache,
                position,
                kv_key_fp16_enabled(),
                kv_value_fp16_enabled(),
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_rope_kv_for_layer(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
            query_output: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            layer: usize,
            position: usize,
        ) -> Result<(), String> {
            self.encode_rope_kv_precision(
                context,
                query,
                key,
                value,
                query_output,
                key_cache,
                value_cache,
                position,
                kv_key_fp16_for_layer(layer),
                kv_value_fp16_for_layer(layer),
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_rope_kv_precision(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
            query_output: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            position: usize,
            key_fp16: bool,
            value_fp16: bool,
        ) -> Result<(), String> {
            if rope_kv_split_enabled() && !key_fp16 && !value_fp16 {
                return self.encode_rope_kv_split(
                    context,
                    query,
                    key,
                    value,
                    query_output,
                    key_cache,
                    value_cache,
                    position,
                );
            }
            #[repr(C)]
            struct Params {
                position: u32,
            }
            let params = Params {
                position: position as u32,
            };
            self.encode_decode_kernel(
                context,
                match (key_fp16, value_fp16) {
                    (false, false) => DecodeKernel::RopeKv,
                    (true, true) => DecodeKernel::RopeKvF16,
                    (false, true) => DecodeKernel::RopeKvKf32Vf16,
                    (true, false) => DecodeKernel::RopeKvKf16Vf32,
                },
                &[query, key, value, query_output, key_cache, value_cache],
                bytes_of(&params),
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_rope_kv_split(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
            query_output: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            position: usize,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                position: u32,
            }
            let params = Params {
                position: position as u32,
            };
            self.encode_decode_kernel(
                context,
                DecodeKernel::RopeQuerySimd,
                &[query, query_output],
                bytes_of(&params),
                MTLSize {
                    width: 16 * (64 / 2),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )?;
            self.encode_decode_kernel(
                context,
                DecodeKernel::RopeKvSimd,
                &[key, value, key_cache, value_cache],
                bytes_of(&params),
                MTLSize {
                    width: 8 * (64 / 2),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn encode_attention(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            output: &RuntimeBuffer,
            length: usize,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                length: u32,
            }
            let params = Params {
                length: length as u32,
            };
            self.encode_decode_kernel(
                context,
                DecodeKernel::Attention,
                &[query, key_cache, value_cache, output],
                bytes_of(&params),
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_attention_three_stage(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            scores: &RuntimeBuffer,
            output: &RuntimeBuffer,
            length: usize,
        ) -> Result<(), String> {
            if attention_chunked_enabled() && !kv_key_fp16_enabled() && !kv_value_fp16_enabled() {
                return self.encode_attention_chunked_gqa(
                    context,
                    query,
                    key_cache,
                    value_cache,
                    scores,
                    output,
                    length,
                );
            }
            if attention_stream_enabled() && !kv_key_fp16_enabled() && !kv_value_fp16_enabled() {
                return self.encode_attention_stream_gqa(
                    context,
                    query,
                    key_cache,
                    value_cache,
                    output,
                    length,
                );
            }
            self.encode_attention_qk(context, query, key_cache, scores, length)?;
            self.encode_attention_softmax(context, scores, length)?;
            self.encode_attention_v(context, scores, value_cache, output, length)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_attention_three_stage_for_layer(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            scores: &RuntimeBuffer,
            output: &RuntimeBuffer,
            layer: usize,
            length: usize,
        ) -> Result<(), String> {
            if attention_chunked_enabled()
                && !kv_key_fp16_for_layer(layer)
                && !kv_value_fp16_for_layer(layer)
            {
                return self.encode_attention_chunked_gqa(
                    context,
                    query,
                    key_cache,
                    value_cache,
                    scores,
                    output,
                    length,
                );
            }
            if attention_stream_enabled()
                && !kv_key_fp16_for_layer(layer)
                && !kv_value_fp16_for_layer(layer)
            {
                return self.encode_attention_stream_gqa(
                    context,
                    query,
                    key_cache,
                    value_cache,
                    output,
                    length,
                );
            }
            self.encode_attention_qk_for_layer(context, query, key_cache, scores, layer, length)?;
            self.encode_attention_softmax(context, scores, length)?;
            self.encode_attention_v_for_layer(context, scores, value_cache, output, layer, length)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_attention_stream_gqa(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            output: &RuntimeBuffer,
            length: usize,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                length: u32,
            }
            let params = Params {
                length: length as u32,
            };
            self.encode_decode_kernel(
                context,
                DecodeKernel::AttentionStreamGqa,
                &[query, key_cache, value_cache, output],
                bytes_of(&params),
                MTLSize {
                    width: 8 * 32,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: attention_stream_threadgroup_size(),
                    height: 1,
                    depth: 1,
                },
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_attention_chunked_gqa(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            length: usize,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                length: u32,
            }
            let params = Params {
                length: length as u32,
            };
            self.encode_decode_kernel(
                context,
                DecodeKernel::AttentionChunkedGqa,
                &[query, key_cache, value_cache, scratch],
                bytes_of(&params),
                MTLSize {
                    width: 8 * 256,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )?;
            self.encode_decode_kernel(
                context,
                DecodeKernel::AttentionChunkedMerge,
                &[scratch, output],
                bytes_of(&params),
                MTLSize {
                    width: 16 * 64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn encode_attention_qk(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            scores: &RuntimeBuffer,
            length: usize,
        ) -> Result<(), String> {
            self.encode_attention_qk_precision(
                context,
                query,
                key_cache,
                scores,
                length,
                kv_key_fp16_enabled(),
            )
        }

        /// Benchmark-only QK variant for a synthetic KV-head-major cache:
        /// [kv_head][position][head_dim]. Production remains position-major.
        pub fn encode_attention_qk_kmajor(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            scores: &RuntimeBuffer,
            length: usize,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                length: u32,
            }
            let params = Params {
                length: length as u32,
            };
            self.encode_decode_kernel(
                context,
                DecodeKernel::AttentionQkKMajor,
                &[query, key_cache, scores],
                bytes_of(&params),
                MTLSize {
                    width: 16 * 512,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn encode_attention_qk_for_layer(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            scores: &RuntimeBuffer,
            layer: usize,
            length: usize,
        ) -> Result<(), String> {
            self.encode_attention_qk_precision(
                context,
                query,
                key_cache,
                scores,
                length,
                kv_key_fp16_for_layer(layer),
            )
        }

        fn encode_attention_qk_precision(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            scores: &RuntimeBuffer,
            length: usize,
            key_fp16: bool,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                length: u32,
            }
            let params = Params {
                length: length as u32,
            };
            self.encode_decode_kernel(
                context,
                if key_fp16 {
                    DecodeKernel::AttentionQkF16
                } else {
                    DecodeKernel::AttentionQk
                },
                &[query, key_cache, scores],
                bytes_of(&params),
                MTLSize {
                    width: 16 * 512,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn encode_attention_softmax(
            &mut self,
            context: &MetalContext,
            scores: &RuntimeBuffer,
            length: usize,
        ) -> Result<(), String> {
            if attention_softmax_simd_enabled() {
                return self.encode_attention_softmax_simd(context, scores, length);
            }
            #[repr(C)]
            struct Params {
                length: u32,
            }
            let params = Params {
                length: length as u32,
            };
            self.encode_decode_kernel(
                context,
                DecodeKernel::AttentionSoftmax,
                &[scores],
                bytes_of(&params),
                MTLSize {
                    width: 16,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 16,
                    height: 1,
                    depth: 1,
                },
            )
        }

        fn encode_attention_softmax_simd(
            &mut self,
            context: &MetalContext,
            scores: &RuntimeBuffer,
            length: usize,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                length: u32,
            }
            let params = Params {
                length: length as u32,
            };
            const SCORE_BYTES: usize = 16 * 512 * std::mem::size_of::<f32>();
            let partial_max = scores.subregion(SCORE_BYTES, 16 * 16 * 4)?;
            self.encode_decode_kernel(
                context,
                DecodeKernel::AttentionSoftmaxMax,
                &[scores, &partial_max],
                bytes_of(&params),
                MTLSize {
                    width: 16 * 16 * 32,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            )?;
            self.encode_decode_kernel(
                context,
                DecodeKernel::AttentionSoftmaxSimd,
                &[scores, &partial_max],
                bytes_of(&params),
                MTLSize {
                    width: 16 * 256,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn encode_attention_v(
            &mut self,
            context: &MetalContext,
            scores: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            output: &RuntimeBuffer,
            length: usize,
        ) -> Result<(), String> {
            self.encode_attention_v_precision(
                context,
                scores,
                value_cache,
                output,
                length,
                kv_value_fp16_enabled(),
            )
        }

        pub fn encode_attention_v_for_layer(
            &mut self,
            context: &MetalContext,
            scores: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            output: &RuntimeBuffer,
            layer: usize,
            length: usize,
        ) -> Result<(), String> {
            self.encode_attention_v_precision(
                context,
                scores,
                value_cache,
                output,
                length,
                kv_value_fp16_for_layer(layer),
            )
        }

        fn encode_attention_v_precision(
            &mut self,
            context: &MetalContext,
            scores: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            output: &RuntimeBuffer,
            length: usize,
            value_fp16: bool,
        ) -> Result<(), String> {
            if attention_v_simd_enabled() {
                return self.encode_attention_v_simd(
                    context,
                    scores,
                    value_cache,
                    output,
                    length,
                    value_fp16,
                );
            }
            #[repr(C)]
            struct Params {
                length: u32,
            }
            let params = Params {
                length: length as u32,
            };
            self.encode_decode_kernel(
                context,
                if value_fp16 {
                    DecodeKernel::AttentionVF16
                } else {
                    DecodeKernel::AttentionV
                },
                &[scores, value_cache, output],
                bytes_of(&params),
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        fn encode_attention_v_simd(
            &mut self,
            context: &MetalContext,
            scores: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            output: &RuntimeBuffer,
            length: usize,
            value_fp16: bool,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                length: u32,
            }
            let params = Params {
                length: length as u32,
            };
            const SCORE_BYTES: usize = 16 * 512 * std::mem::size_of::<f32>();
            const PARTIAL_BYTES: usize = 16 * 16 * 64 * std::mem::size_of::<f32>();
            let partials = scores.subregion(SCORE_BYTES, PARTIAL_BYTES)?;
            self.encode_decode_kernel(
                context,
                if value_fp16 {
                    DecodeKernel::AttentionVPartialF16
                } else {
                    DecodeKernel::AttentionVPartial
                },
                &[scores, value_cache, &partials],
                bytes_of(&params),
                MTLSize {
                    width: 8 * 16 * 64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            )?;
            self.encode_decode_kernel(
                context,
                DecodeKernel::AttentionVReduce,
                &[&partials, output],
                bytes_of(&params),
                MTLSize {
                    width: 16 * 64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn encode_residual(
            &mut self,
            context: &MetalContext,
            left: &RuntimeBuffer,
            right: &RuntimeBuffer,
            output: &RuntimeBuffer,
            scale: f32,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                n: u32,
                scale: f32,
            }
            let params = Params { n: 1024, scale };
            self.encode_decode_kernel(
                context,
                DecodeKernel::Residual,
                &[left, right, output],
                bytes_of(&params),
                MTLSize {
                    width: 1024,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn encode_topk(
            &mut self,
            context: &MetalContext,
            logits: &RuntimeBuffer,
            ids: &RuntimeBuffer,
            weights: &RuntimeBuffer,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                n: u32,
            }
            let params = Params { n: 32 };
            self.encode_decode_kernel(
                context,
                DecodeKernel::TopK,
                &[logits, ids, weights],
                bytes_of(&params),
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            )
        }

        /// Mask the tail of a canonical top-8 route for opt-in MoE active-k
        /// experiments. This stays GPU-resident and adds one tiny dispatch.
        pub fn encode_topk_mask(
            &mut self,
            context: &MetalContext,
            ids: &RuntimeBuffer,
            weights: &RuntimeBuffer,
            active_k: u32,
        ) -> Result<(), String> {
            if !(1..=8).contains(&active_k) {
                return Err("active MoE k must be in 1..=8".to_owned());
            }
            self.encode_decode_kernel(
                context,
                DecodeKernel::TopKMask,
                &[ids, weights],
                bytes_of(&active_k),
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            )
        }

        pub fn dispatch_threads(&mut self, threads: MTLSize, group: MTLSize) {
            self.encoder
                .dispatchThreads_threadsPerThreadgroup(threads, group);
            self.dispatches += 1;
        }

        pub fn dispatch_threadgroups(&mut self, groups: MTLSize, group: MTLSize) {
            self.encoder
                .dispatchThreadgroups_threadsPerThreadgroup(groups, group);
            self.dispatches += 1;
        }

        /// Record a GPU timestamp at the current dispatch boundary. Markers
        /// are no-ops unless stage diagnostics were enabled on the context.
        pub fn sample_stage_boundary(&mut self) {
            if let Some(timeline) = self.counter_timeline.as_mut() {
                timeline.sample(&self.encoder);
            }
        }

        /// Mark a logical stage boundary. On devices with Metal counters this
        /// is an in-command-buffer sample. On devices without them, the
        /// diagnostic mode uses an ordered command-buffer boundary instead;
        /// there is still no CPU wait and GPU state remains resident.
        pub fn stage_boundary(&mut self, context: &MetalContext) -> Result<(), String> {
            if self.counter_timeline.is_some() {
                self.sample_stage_boundary();
                return Ok(());
            }
            self.begin_next_command_buffer(context)
        }

        pub fn encode_noop(
            &mut self,
            context: &MetalContext,
            threads: MTLSize,
            group: MTLSize,
        ) -> Result<(), String> {
            let pipeline = context.decode_pipeline(DecodeKernel::Noop)?;
            self.encoder.setComputePipelineState(&pipeline);
            self.encoder
                .dispatchThreads_threadsPerThreadgroup(threads, group);
            self.dispatches += 1;
            Ok(())
        }

        /// End and commit the current command buffer, then continue encoding
        /// on the same queue without waiting on the CPU. Metal command queues
        /// preserve submission order, so the next segment observes all writes
        /// from the previous segment while GPU-resident state remains live.
        pub fn begin_next_command_buffer(&mut self, context: &MetalContext) -> Result<(), String> {
            let next_command_buffer = context
                .queue
                .commandBuffer()
                .ok_or_else(|| "Runtime V2 command buffer allocation failed".to_owned())?;
            let next_encoder = next_command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Runtime V2 encoder allocation failed".to_owned())?;
            let previous_encoder = std::mem::replace(&mut self.encoder, next_encoder);
            previous_encoder.endEncoding();
            let previous_command_buffer =
                std::mem::replace(&mut self.command_buffer, next_command_buffer);
            previous_command_buffer.commit();
            self.committed
                .push((previous_command_buffer, self.dispatches, self.encoders));
            self.dispatches = 0;
            self.encoders = 1;
            Ok(())
        }

        /// End the current compute encoder while keeping the command buffer
        /// open. This is a scheduling-only boundary: all resources remain in
        /// the same GPU command buffer and no host synchronization occurs.
        pub fn begin_next_encoder(&mut self) -> Result<(), String> {
            self.encoder.endEncoding();
            let next_encoder = self
                .command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Runtime V2 encoder allocation failed".to_owned())?;
            let previous_encoder = std::mem::replace(&mut self.encoder, next_encoder);
            drop(previous_encoder);
            self.encoders += 1;
            Ok(())
        }

        pub fn finish(mut self, context: &MetalContext) -> Result<(), String> {
            self.encoder.endEncoding();
            self.command_buffer.commit();
            let wait_start = Instant::now();
            self.command_buffer.waitUntilCompleted();
            self.committed
                .push((self.command_buffer, self.dispatches, self.encoders));
            for (command_buffer, _, _) in &self.committed {
                if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                    return Err("Runtime V2 command buffer execution failed".to_owned());
                }
            }
            let mut metrics = context
                .metrics
                .lock()
                .map_err(|_| "Metal metrics cache is poisoned".to_owned())?;
            let gpu_duration_seconds = self
                .committed
                .iter()
                .map(|(command_buffer, _, _)| {
                    (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()).max(0.0)
                })
                .sum::<f64>();
            let stage_timeline = self
                .counter_timeline
                .take()
                .map(|timeline| timeline.resolve(gpu_duration_seconds))
                .unwrap_or_else(|| StageTimeline {
                    timestamps: Vec::new(),
                    gpu_duration_seconds,
                    segment_gpu_seconds: self
                        .committed
                        .iter()
                        .map(|(command_buffer, _, _)| {
                            (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()).max(0.0)
                        })
                        .collect(),
                });
            let mut stage_timeline = stage_timeline;
            if stage_timeline.segment_gpu_seconds.is_empty() {
                stage_timeline.segment_gpu_seconds = self
                    .committed
                    .iter()
                    .map(|(command_buffer, _, _)| {
                        (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()).max(0.0)
                    })
                    .collect();
            }
            *context
                .last_stage_timeline
                .lock()
                .map_err(|_| "Metal stage timeline cache is poisoned".to_owned())? = stage_timeline;
            metrics.command_buffers += self.committed.len() as u64;
            metrics.encoders += self
                .committed
                .iter()
                .map(|(_, _, encoders)| *encoders)
                .sum::<u64>();
            metrics.dispatches += self
                .committed
                .iter()
                .map(|(_, dispatches, _)| *dispatches)
                .sum::<u64>();
            metrics.gpu_time_seconds += self
                .committed
                .iter()
                .map(|(command_buffer, _, _)| {
                    (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()).max(0.0)
                })
                .sum::<f64>();
            metrics.cpu_encode_seconds += wait_start
                .duration_since(self.encode_started_at)
                .as_secs_f64();
            // There is exactly one CPU wait, on the final command buffer.
            metrics.cpu_wait_seconds += wait_start.elapsed().as_secs_f64();
            Ok(())
        }
    }

    impl MetalContext {
        pub fn system_default() -> Option<Self> {
            let device = MTLCreateSystemDefaultDevice()?;
            let queue = device.newCommandQueue()?;
            Some(Self {
                device,
                queue,
                pipelines: Mutex::new([None, None, None]),
                aggregate_pipelines: Mutex::new([None, None, None]),
                logits_pipeline: Mutex::new(None),
                dense_pipeline: Mutex::new(None),
                f16_dense_pipeline: Mutex::new(None),
                q4_dense_library: Mutex::new(None),
                q4_dense_pipeline: Mutex::new(None),
                q4_dense_simd_pipeline: Mutex::new(None),
                q4_dense_sweep_pipeline: Mutex::new(None),
                q4_dense_argmax_pipeline: Mutex::new(None),
                q4_dense_argmax_norm_pipeline: Mutex::new(None),
                q4_qkv_pipeline: Mutex::new(None),
                q4_qkv_simd_pipeline: Mutex::new(None),
                q4_qkv_simd_norm_pipeline: Mutex::new(None),
                q4_qkv_sweep_pipeline: Mutex::new(None),
                q4_gemv_pipeline: Mutex::new(None),
                q4_simd_gemv_pipeline: Mutex::new(None),
                q4_simd_norm_gemv_pipeline: Mutex::new(None),
                dynamic_moe_pipeline: Mutex::new(None),
                dynamic_moe_library: Mutex::new(None),
                dynamic_moe_simd_pipeline: Mutex::new(None),
                q4_embedding_pipeline: Mutex::new(None),
                argmax_partial_pipeline: Mutex::new(None),
                argmax_reduce_pipeline: Mutex::new(None),
                argmax_reduce_partials_pipeline: Mutex::new(None),
                v2_pipelines: Mutex::new([
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None,
                ]),
                scratch: Mutex::new(MetalScratch {
                    input_f32: None,
                    input_u16: None,
                    dense_outputs: Vec::new(),
                    expert_outputs: Vec::new(),
                    moe_output: None,
                    logits_output: None,
                }),
                metrics: Mutex::new(MetalMetricsState::default()),
                no_copy_buffers: Mutex::new(HashMap::new()),
                stage_counters: Mutex::new(false),
                last_stage_timeline: Mutex::new(StageTimeline::default()),
            })
        }

        pub fn metrics(&self) -> Result<MetalMetrics, String> {
            let metrics = self
                .metrics
                .lock()
                .map_err(|_| "Metal metrics cache is poisoned".to_owned())?;
            Ok(MetalMetrics {
                command_buffers: metrics.command_buffers,
                encoders: metrics.encoders,
                dispatches: metrics.dispatches,
                gpu_time_seconds: metrics.gpu_time_seconds,
                cpu_encode_seconds: metrics.cpu_encode_seconds,
                cpu_wait_seconds: metrics.cpu_wait_seconds,
            })
        }

        /// Enable dispatch-boundary timestamps for subsequently created
        /// decode graphs. This is diagnostic-only and intentionally opt-in.
        pub fn set_stage_counters(&self, enabled: bool) -> Result<(), String> {
            *self
                .stage_counters
                .lock()
                .map_err(|_| "Metal stage-counter cache is poisoned".to_owned())? = enabled;
            Ok(())
        }

        pub fn last_stage_timeline(&self) -> Result<StageTimeline, String> {
            self.last_stage_timeline
                .lock()
                .map_err(|_| "Metal stage timeline cache is poisoned".to_owned())
                .map(|timeline| timeline.clone())
        }

        pub fn current_allocated_bytes(&self) -> u64 {
            self.device.currentAllocatedSize() as u64
        }

        pub fn recommended_working_set_bytes(&self) -> u64 {
            self.device.recommendedMaxWorkingSetSize()
        }

        pub fn begin_decode(&self) -> Result<DecodeGraph, String> {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Runtime V2 command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Runtime V2 encoder allocation failed".to_owned())?;
            Ok(DecodeGraph {
                command_buffer,
                encoder,
                dispatches: 0,
                counter_timeline: if *self
                    .stage_counters
                    .lock()
                    .map_err(|_| "Metal stage-counter cache is poisoned".to_owned())?
                {
                    CounterTimeline::new(&self.device)
                } else {
                    None
                },
                committed: Vec::new(),
                encoders: 1,
                encode_started_at: Instant::now(),
            })
        }

        pub fn allocate_runtime_buffer(&self, length: usize) -> Result<RuntimeBuffer, String> {
            if length == 0 {
                return Err("V2 runtime buffers cannot be empty".to_owned());
            }
            let buffer = self
                .device
                .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| "V2 runtime buffer allocation failed".to_owned())?;
            Ok(RuntimeBuffer {
                buffer,
                offset: 0,
                length,
            })
        }

        /// Wrap a page-aligned immutable mapped slab without copying it. The
        /// captured owner keeps the mapping alive until Metal releases the
        /// buffer. Tensor views should be expressed as `RuntimeBuffer`
        /// subregions of this single slab.
        ///
        /// # Safety
        /// `pointer..pointer + length` must remain valid and immutable for the
        /// lifetime of `owner`, and must satisfy Metal's no-copy alignment
        /// requirements.
        pub unsafe fn input_buffer_no_copy<T>(
            &self,
            pointer: NonNull<c_void>,
            length: usize,
            owner: Arc<T>,
        ) -> Result<RuntimeBuffer, String>
        where
            T: Send + Sync + 'static,
        {
            if length == 0 {
                return Err("no-copy Metal buffer cannot be empty".to_owned());
            }
            let key = (pointer.as_ptr() as usize, length);
            if let Some(existing) = self
                .no_copy_buffers
                .lock()
                .map_err(|_| "Metal no-copy slab cache is poisoned".to_owned())?
                .get(&key)
                .cloned()
            {
                return Ok(existing);
            }
            let deallocator = RcBlock::new(move |_pointer: NonNull<c_void>, _length: usize| {
                // Capturing the owner is the lifetime mechanism. Metal
                // invokes this callback before releasing its copied block.
                let _ = &owner;
            });
            let buffer = unsafe {
                self.device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        pointer,
                        length,
                        MTLResourceOptions::StorageModeShared,
                        Some(&deallocator),
                    )
            }
            .ok_or_else(|| "Metal no-copy mapped-slab allocation failed".to_owned())?;
            let runtime = RuntimeBuffer {
                buffer,
                offset: 0,
                length,
            };
            self.no_copy_buffers
                .lock()
                .map_err(|_| "Metal no-copy slab cache is poisoned".to_owned())?
                .insert(key, runtime.clone());
            Ok(runtime)
        }

        pub fn write_runtime_buffer(
            &self,
            buffer: &RuntimeBuffer,
            bytes: &[u8],
        ) -> Result<(), String> {
            if bytes.len() > buffer.length {
                return Err("V2 initialization data exceeds runtime buffer".to_owned());
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.buffer.contents().as_ptr().add(buffer.offset) as *mut u8,
                    bytes.len(),
                );
            }
            Ok(())
        }

        pub fn read_runtime_u32(&self, buffer: &RuntimeBuffer) -> Result<u32, String> {
            if buffer.length < std::mem::size_of::<u32>() {
                return Err("V2 token buffer is smaller than u32".to_owned());
            }
            Ok(unsafe { *(buffer.buffer.contents().as_ptr().add(buffer.offset) as *const u32) })
        }

        pub fn read_runtime_u32s(
            &self,
            buffer: &RuntimeBuffer,
            elements: usize,
        ) -> Result<Vec<u32>, String> {
            let bytes = elements
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| "V2 u32 read size overflows".to_owned())?;
            if bytes > buffer.length {
                return Err("V2 u32 read exceeds runtime buffer".to_owned());
            }
            let pointer =
                unsafe { buffer.buffer.contents().as_ptr().add(buffer.offset) as *const u32 };
            Ok(unsafe { std::slice::from_raw_parts(pointer, elements).to_vec() })
        }

        pub fn read_runtime_f32(
            &self,
            buffer: &RuntimeBuffer,
            elements: usize,
        ) -> Result<Vec<f32>, String> {
            let bytes = elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| "V2 f32 read size overflows".to_owned())?;
            if bytes > buffer.length {
                return Err("V2 f32 read exceeds runtime buffer".to_owned());
            }
            let pointer =
                unsafe { buffer.buffer.contents().as_ptr().add(buffer.offset) as *const f32 };
            Ok(unsafe { std::slice::from_raw_parts(pointer, elements).to_vec() })
        }

        pub fn read_runtime_f16(
            &self,
            buffer: &RuntimeBuffer,
            elements: usize,
        ) -> Result<Vec<f32>, String> {
            let bytes = elements
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| "V2 f16 read size overflows".to_owned())?;
            if bytes > buffer.length {
                return Err("V2 f16 read exceeds runtime buffer".to_owned());
            }
            let pointer =
                unsafe { buffer.buffer.contents().as_ptr().add(buffer.offset) as *const u16 };
            let bits = unsafe { std::slice::from_raw_parts(pointer, elements) };
            Ok(bits.iter().copied().map(f16_bits_to_f32).collect())
        }

        fn record_command(
            &self,
            command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
            dispatches: u64,
            encode_start: Instant,
            wait_start: Instant,
        ) -> Result<(), String> {
            let gpu_start = command_buffer.GPUStartTime();
            let gpu_end = command_buffer.GPUEndTime();
            let mut metrics = self
                .metrics
                .lock()
                .map_err(|_| "Metal metrics cache is poisoned".to_owned())?;
            metrics.command_buffers += 1;
            metrics.encoders += 1;
            metrics.dispatches += dispatches;
            metrics.gpu_time_seconds += (gpu_end - gpu_start).max(0.0);
            metrics.cpu_encode_seconds += wait_start.duration_since(encode_start).as_secs_f64();
            metrics.cpu_wait_seconds += wait_start.elapsed().as_secs_f64();
            Ok(())
        }

        fn pipeline_index(quant_type: QuantType) -> usize {
            match quant_type {
                QuantType::Q2 => 0,
                QuantType::Q3 => 1,
                QuantType::Q4 => 2,
                QuantType::Q4K => panic!("Q4_K uses the dedicated dynamic MoE pipeline"),
            }
        }

        fn fused_pipeline(
            &self,
            quant_type: QuantType,
        ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
            let index = Self::pipeline_index(quant_type);
            let mut pipelines = self
                .pipelines
                .lock()
                .map_err(|_| "Metal pipeline cache is poisoned".to_owned())?;
            if let Some(pipeline) = pipelines[index].as_ref() {
                return Ok(pipeline.clone());
            }
            let source = NSString::from_str(kernel_source(quant_type));
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal library compilation failed".to_owned())?;
            let name = NSString::from_str(&format!("qx_q{}_expert", quant_type.bits()));
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal fused expert function was not found".to_owned())?;
            let pipeline = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal compute pipeline creation failed".to_owned())?;
            pipelines[index] = Some(pipeline.clone());
            Ok(pipeline)
        }

        fn q4_gemv_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_gemv_pipeline
                .lock()
                .map_err(|_| "Metal Q4 stage pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(kernel_source(QuantType::Q4));
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal Q4 stage library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_q4_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal Q4 stage function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal Q4 stage pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_simd_gemv_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_simd_gemv_pipeline
                .lock()
                .map_err(|_| "Metal SIMD Q4 stage pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(kernel_source(QuantType::Q4));
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal SIMD Q4 stage library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_q4_simd_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal SIMD Q4 stage function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal SIMD Q4 stage pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_simd_norm_gemv_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_simd_norm_gemv_pipeline
                .lock()
                .map_err(|_| "Metal normalized SIMD Q4 stage cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(Q4_DENSE_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| {
                    "Metal normalized SIMD Q4 stage library compilation failed".to_owned()
                })?;
            let name = NSString::from_str("qx_q4_simd_norm_gemv");
            let function = library.newFunctionWithName(&name).ok_or_else(|| {
                "Metal normalized SIMD Q4 stage function was not found".to_owned()
            })?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| {
                    "Metal normalized SIMD Q4 stage pipeline creation failed".to_owned()
                })?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn aggregate_pipeline(
            &self,
            quant_type: QuantType,
        ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
            let index = Self::pipeline_index(quant_type);
            let mut pipelines = self
                .aggregate_pipelines
                .lock()
                .map_err(|_| "Metal aggregate pipeline cache is poisoned".to_owned())?;
            if let Some(pipeline) = pipelines[index].as_ref() {
                return Ok(pipeline.clone());
            }
            let source = NSString::from_str(kernel_source(quant_type));
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal aggregate library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_moe_accumulate");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal aggregate function was not found".to_owned())?;
            let pipeline = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal aggregate pipeline creation failed".to_owned())?;
            pipelines[index] = Some(pipeline.clone());
            Ok(pipeline)
        }

        fn input_buffer(
            &self,
            bytes: &[u8],
        ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
            let pointer = NonNull::new(bytes.as_ptr() as *mut c_void)
                .ok_or_else(|| "Metal input buffer cannot be empty".to_owned())?;
            let shared = unsafe {
                self.device.newBufferWithBytes_length_options(
                    pointer,
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| "Metal input buffer allocation failed".to_owned())?;
            if std::env::var_os("QX_METAL_PRIVATE_STATIC").is_none() {
                return Ok(shared);
            }

            let private = self
                .device
                .newBufferWithLength_options(bytes.len(), MTLResourceOptions::StorageModePrivate)
                .ok_or_else(|| "Metal private input buffer allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal private input copy command buffer failed".to_owned())?;
            let blit = command_buffer
                .blitCommandEncoder()
                .ok_or_else(|| "Metal private input copy encoder failed".to_owned())?;
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &shared,
                    0,
                    &private,
                    0,
                    bytes.len(),
                );
            }
            blit.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal private input copy failed".to_owned());
            }
            Ok(private)
        }

        fn input_runtime_buffer(&self, bytes: &[u8]) -> Result<RuntimeBuffer, String> {
            Ok(RuntimeBuffer {
                buffer: self.input_buffer(bytes)?,
                offset: 0,
                length: bytes.len(),
            })
        }

        fn logits_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .logits_pipeline
                .lock()
                .map_err(|_| "Metal logits pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(LOGITS_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal logits library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_logits");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal logits function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal logits pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn dense_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .dense_pipeline
                .lock()
                .map_err(|_| "Metal dense pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(DENSE_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal dense library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_dense_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal dense function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal dense pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn f16_dense_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .f16_dense_pipeline
                .lock()
                .map_err(|_| "Metal FP16 dense pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(F16_DENSE_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal FP16 dense library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_f16_dense_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal FP16 dense function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal FP16 dense pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_library(&self) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
            let mut library = self
                .q4_dense_library
                .lock()
                .map_err(|_| "Metal Q4 dense library cache is poisoned".to_owned())?;
            if let Some(value) = library.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(Q4_DENSE_KERNEL_SOURCE);
            let value = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal Q4 dense library compilation failed".to_owned())?;
            *library = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_pipeline
                .lock()
                .map_err(|_| "Metal Q4 dense pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal Q4 dense function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal Q4 dense pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_simd_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_simd_pipeline
                .lock()
                .map_err(|_| "Metal SIMD Q4 dense pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_simd_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal SIMD Q4 dense function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal SIMD Q4 dense pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_sweep_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_sweep_pipeline
                .lock()
                .map_err(|_| "Metal Q4 sweep pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_sweep_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal Q4 sweep function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal Q4 sweep pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_argmax_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_argmax_pipeline
                .lock()
                .map_err(|_| "Metal Q4 argmax pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_argmax_partial");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal Q4 argmax function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal Q4 argmax pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_qkv_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_qkv_pipeline
                .lock()
                .map_err(|_| "Metal fused QKV pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_qkv_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal fused QKV function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal fused QKV pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_qkv_simd_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_qkv_simd_pipeline
                .lock()
                .map_err(|_| "Metal SIMD QKV pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_qkv_simd_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal SIMD QKV function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal SIMD QKV pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_qkv_simd_norm_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_qkv_simd_norm_pipeline
                .lock()
                .map_err(|_| "Metal normalized SIMD QKV pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_qkv_simd_norm_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal normalized SIMD QKV function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal normalized SIMD QKV pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_qkv_sweep_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_qkv_sweep_pipeline
                .lock()
                .map_err(|_| "Metal QKV sweep pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_qkv_sweep_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal QKV sweep function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal QKV sweep pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn dynamic_moe_library(&self) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
            let mut library = self
                .dynamic_moe_library
                .lock()
                .map_err(|_| "Metal dynamic MoE library cache is poisoned".to_owned())?;
            if let Some(value) = library.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(DYNAMIC_MOE_KERNEL_SOURCE);
            let value = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal dynamic MoE library compilation failed".to_owned())?;
            *library = Some(value.clone());
            Ok(value)
        }

        fn dynamic_moe_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .dynamic_moe_pipeline
                .lock()
                .map_err(|_| "Metal dynamic MoE pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.dynamic_moe_library()?;
            let name = NSString::from_str("qx_dynamic_q4_moe");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal dynamic MoE function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal dynamic MoE pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn dynamic_moe_simd_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .dynamic_moe_simd_pipeline
                .lock()
                .map_err(|_| "Metal SIMD dynamic MoE pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.dynamic_moe_library()?;
            let name = NSString::from_str("qx_dynamic_q4_moe_simd");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal SIMD dynamic MoE function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal SIMD dynamic MoE pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn dynamic_moe_named_pipeline(&self, name: &str) -> Result<Pipeline, String> {
            let library = self.dynamic_moe_library()?;
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("Metal dynamic MoE function {name} was not found"))?;
            self.device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| format!("Metal dynamic MoE pipeline {name} creation failed"))
        }

        fn q4_embedding_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_embedding_pipeline
                .lock()
                .map_err(|_| "Metal Q4 embedding pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(Q4_EMBEDDING_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal Q4 embedding library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_q4_embedding_lookup");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal Q4 embedding function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal Q4 embedding pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn argmax_partial_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .argmax_partial_pipeline
                .lock()
                .map_err(|_| "Metal argmax pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(ARGMAX_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal argmax library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_argmax_partial");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal argmax partial function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal argmax partial pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn argmax_reduce_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .argmax_reduce_pipeline
                .lock()
                .map_err(|_| "Metal argmax pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(ARGMAX_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal argmax library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_argmax_reduce");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal argmax reduce function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal argmax reduce pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn argmax_reduce_partials_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .argmax_reduce_partials_pipeline
                .lock()
                .map_err(|_| "Metal argmax partial-reduction cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(ARGMAX_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal argmax library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_argmax_reduce_partials");
            let function = library.newFunctionWithName(&name).ok_or_else(|| {
                "Metal argmax partial-reduction function was not found".to_owned()
            })?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| {
                    "Metal argmax partial-reduction pipeline creation failed".to_owned()
                })?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_argmax_norm_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_argmax_norm_pipeline
                .lock()
                .map_err(|_| "Metal normalized Q4 argmax pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_argmax_norm_partial");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal normalized Q4 argmax function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal normalized Q4 argmax pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn decode_pipeline(&self, kernel: DecodeKernel) -> Result<Pipeline, String> {
            let index = match kernel {
                DecodeKernel::RmsNorm => 0,
                DecodeKernel::RmsNormSimd => 1,
                DecodeKernel::RmsNormScalar => 2,
                DecodeKernel::F32ToF16 => 3,
                DecodeKernel::F16ToF32 => 4,
                DecodeKernel::Residual => 5,
                DecodeKernel::RopeKv => 6,
                DecodeKernel::Attention => 7,
                DecodeKernel::AttentionSimd => 8,
                DecodeKernel::TopK => 9,
                DecodeKernel::Noop => 10,
                DecodeKernel::AttentionQk => 11,
                DecodeKernel::AttentionSoftmax => 12,
                DecodeKernel::AttentionV => 13,
                DecodeKernel::RopeKvF16 => 14,
                DecodeKernel::RopeKvKf32Vf16 => 15,
                DecodeKernel::RopeKvKf16Vf32 => 16,
                DecodeKernel::AttentionQkF16 => 17,
                DecodeKernel::AttentionVF16 => 18,
                DecodeKernel::AttentionVPartial => 19,
                DecodeKernel::AttentionVPartialF16 => 20,
                DecodeKernel::AttentionVReduce => 21,
                DecodeKernel::AttentionSoftmaxMax => 22,
                DecodeKernel::AttentionSoftmaxSimd => 23,
                DecodeKernel::RopeQuerySimd => 24,
                DecodeKernel::RopeKvSimd => 25,
                DecodeKernel::TopKMask => 26,
                DecodeKernel::AttentionQkKMajor => 27,
                DecodeKernel::AttentionStreamGqa => 28,
                DecodeKernel::AttentionChunkedGqa => 29,
                DecodeKernel::AttentionChunkedMerge => 30,
            };
            let mut pipelines = self
                .v2_pipelines
                .lock()
                .map_err(|_| "Metal V2 pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipelines[index].as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(V2_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, None)
                .map_err(|_| "Metal V2 decode library compilation failed".to_owned())?;
            let name = NSString::from_str(match kernel {
                DecodeKernel::RmsNorm => "qx_v2_rmsnorm",
                DecodeKernel::RmsNormSimd => "qx_v2_rmsnorm_simd",
                DecodeKernel::RmsNormScalar => "qx_v2_rmsnorm_scalar",
                DecodeKernel::F32ToF16 => "qx_v2_f32_to_f16",
                DecodeKernel::F16ToF32 => "qx_v2_f16_to_f32",
                DecodeKernel::Residual => "qx_v2_residual",
                DecodeKernel::RopeKv => "qx_v2_rope_kv",
                DecodeKernel::Attention => "qx_v2_attention",
                DecodeKernel::AttentionSimd => "qx_v2_attention_simd",
                DecodeKernel::TopK => "qx_v2_topk",
                DecodeKernel::Noop => "qx_v2_noop",
                DecodeKernel::AttentionQk => "qx_v2_attention_qk",
                DecodeKernel::AttentionSoftmax => "qx_v2_attention_softmax",
                DecodeKernel::AttentionV => "qx_v2_attention_v",
                DecodeKernel::RopeKvF16 => "qx_v2_rope_kv_f16",
                DecodeKernel::RopeKvKf32Vf16 => "qx_v2_rope_kv_kf32_vf16",
                DecodeKernel::RopeKvKf16Vf32 => "qx_v2_rope_kv_kf16_vf32",
                DecodeKernel::AttentionQkF16 => "qx_v2_attention_qk_f16",
                DecodeKernel::AttentionVF16 => "qx_v2_attention_v_f16",
                DecodeKernel::AttentionVPartial => "qx_v2_attention_v_partial_gqa",
                DecodeKernel::AttentionVPartialF16 => "qx_v2_attention_v_partial_gqa_f16",
                DecodeKernel::AttentionVReduce => "qx_v2_attention_v_reduce",
                DecodeKernel::AttentionSoftmaxMax => "qx_v2_attention_softmax_max",
                DecodeKernel::AttentionSoftmaxSimd => "qx_v2_attention_softmax_simd",
                DecodeKernel::RopeQuerySimd => "qx_v2_rope_query_simd",
                DecodeKernel::RopeKvSimd => "qx_v2_rope_kv_simd",
                DecodeKernel::TopKMask => "qx_v2_topk_mask",
                DecodeKernel::AttentionQkKMajor => "qx_v2_attention_qk_kmajor",
                DecodeKernel::AttentionStreamGqa => "qx_v2_attention_stream_gqa",
                DecodeKernel::AttentionChunkedGqa => "qx_v2_attention_chunked_gqa",
                DecodeKernel::AttentionChunkedMerge => "qx_v2_attention_chunked_merge",
            });
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal V2 decode function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal V2 decode pipeline creation failed".to_owned())?;
            pipelines[index] = Some(value.clone());
            Ok(value)
        }

        pub fn prepare_dense(
            &self,
            weights: &[f32],
            rows: usize,
            columns: usize,
        ) -> Result<PreparedDense, String> {
            if rows == 0 || columns == 0 || weights.len() != rows * columns {
                return Err(format!(
                    "dense weight shape {rows}x{columns} does not match {} values",
                    weights.len()
                ));
            }
            let weight_bytes = unsafe {
                std::slice::from_raw_parts(
                    weights.as_ptr() as *const u8,
                    std::mem::size_of_val(weights),
                )
            };
            Ok(PreparedDense {
                pipeline: self.dense_pipeline()?,
                weights: self.input_buffer(weight_bytes)?,
                rows,
                columns,
            })
        }

        pub fn prepare_f16_dense(
            &self,
            weights: &[u16],
            rows: usize,
            columns: usize,
        ) -> Result<PreparedF16Dense, String> {
            if rows == 0 || columns == 0 || weights.len() != rows * columns {
                return Err(format!(
                    "FP16 dense weight shape {rows}x{columns} does not match {} values",
                    weights.len()
                ));
            }
            let weight_bytes = unsafe {
                std::slice::from_raw_parts(
                    weights.as_ptr() as *const u8,
                    std::mem::size_of_val(weights),
                )
            };
            Ok(PreparedF16Dense {
                pipeline: self.f16_dense_pipeline()?,
                weights: self.input_buffer(weight_bytes)?,
                rows,
                columns,
            })
        }

        pub fn run_prepared_f16_dense(
            &self,
            prepared: &PreparedF16Dense,
            input: &[u16],
        ) -> Result<Vec<u16>, String> {
            if input.len() != prepared.columns {
                return Err(format!(
                    "FP16 dense input length {}, expected {}",
                    input.len(),
                    prepared.columns
                ));
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let input_buffer = self.input_buffer(input_bytes)?;
            let output_buffer = self
                .device
                .newBufferWithLength_options(
                    prepared.rows * std::mem::size_of::<u16>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal FP16 dense output allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal FP16 dense command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal FP16 dense encoder allocation failed".to_owned())?;
            encoder.setComputePipelineState(&prepared.pipeline);
            let columns = prepared.columns as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&prepared.weights), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    3,
                );
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal FP16 dense command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let contents = output_buffer.contents().as_ptr() as *const u16;
            Ok(unsafe { std::slice::from_raw_parts(contents, prepared.rows) }.to_vec())
        }

        pub fn prepare_q4_dense(
            &self,
            weights: &qx_quant::QuantizedTensor,
        ) -> Result<PreparedQuantizedDense, String> {
            if weights.quant_type != QuantType::Q4
                || weights.layout != PackingLayout::RowMajor
                || weights.group_size != 32
            {
                return Err(
                    "V2 Q4 dense requires row-major Q4 weights with group size 32".to_owned(),
                );
            }
            let packed = self.input_runtime_buffer(&weights.data)?;
            let mut scale_bytes = Vec::with_capacity(weights.scales.len() * 2);
            for scale in &weights.scales {
                scale_bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
            }
            let scales = self.input_runtime_buffer(&scale_bytes)?;
            Ok(PreparedQuantizedDense {
                pipeline: self.q4_dense_pipeline()?,
                simd_pipeline: self.q4_dense_simd_pipeline()?,
                sweep_pipeline: self.q4_dense_sweep_pipeline()?,
                simd_norm_pipeline: self.q4_simd_norm_gemv_pipeline()?,
                packed,
                scales,
                rows: weights.rows,
                columns: weights.cols,
            })
        }

        /// Prepare a row-major group-32 Q4 dense matrix from views into a
        /// static artifact slab without reconstructing a QuantizedTensor.
        pub fn prepare_q4_dense_runtime_views(
            &self,
            packed: RuntimeBuffer,
            scales: RuntimeBuffer,
            rows: usize,
            columns: usize,
        ) -> Result<PreparedQuantizedDense, String> {
            if rows == 0 || columns == 0 || columns % 32 != 0 {
                return Err("no-copy Q4 dense has an invalid shape".to_owned());
            }
            let expected_packed = rows * columns / 2;
            let expected_scales = rows * (columns / 32) * 2;
            if packed.length != expected_packed || scales.length != expected_scales {
                return Err(format!(
                    "no-copy Q4 dense byte lengths are {}/{}, expected {expected_packed}/{expected_scales}",
                    packed.length, scales.length
                ));
            }
            Ok(PreparedQuantizedDense {
                pipeline: self.q4_dense_pipeline()?,
                simd_pipeline: self.q4_dense_simd_pipeline()?,
                sweep_pipeline: self.q4_dense_sweep_pipeline()?,
                simd_norm_pipeline: self.q4_simd_norm_gemv_pipeline()?,
                packed,
                scales,
                rows,
                columns,
            })
        }

        pub fn prepare_q4_dense_simd(
            &self,
            weights: &qx_quant::QuantizedTensor,
        ) -> Result<PreparedQuantizedSimdDense, String> {
            if weights.quant_type != QuantType::Q4
                || weights.layout != PackingLayout::RowMajor
                || weights.group_size != 32
            {
                return Err(
                    "SIMD Q4 dense requires row-major Q4 weights with group size 32".to_owned(),
                );
            }
            let packed = self.input_buffer(&weights.data)?;
            let mut scale_bytes = Vec::with_capacity(weights.scales.len() * 2);
            for scale in &weights.scales {
                scale_bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
            }
            let scales = self.input_buffer(&scale_bytes)?;
            Ok(PreparedQuantizedSimdDense {
                pipeline: self.q4_dense_simd_pipeline()?,
                packed,
                scales,
                rows: weights.rows,
                columns: weights.cols,
            })
        }

        pub fn run_prepared_q4_dense_simd(
            &self,
            prepared: &PreparedQuantizedSimdDense,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            if input.len() != prepared.columns {
                return Err(format!(
                    "SIMD Q4 dense input length {}, expected {}",
                    input.len(),
                    prepared.columns
                ));
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let input_buffer = self.input_buffer(input_bytes)?;
            let output_buffer = self
                .device
                .newBufferWithLength_options(
                    prepared.rows * std::mem::size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal SIMD Q4 dense output allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal SIMD Q4 dense command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal SIMD Q4 dense encoder allocation failed".to_owned())?;
            encoder.setComputePipelineState(&prepared.pipeline);
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;
            let rows = prepared.rows as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&prepared.packed), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&prepared.scales), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 3);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    4,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::from(&multiplier).cast(),
                    std::mem::size_of_val(&multiplier),
                    5,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::from(&rows).cast(),
                    std::mem::size_of_val(&rows),
                    6,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows.div_ceil(8),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal SIMD Q4 dense command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let contents = output_buffer.contents().as_ptr() as *const f32;
            Ok(unsafe { std::slice::from_raw_parts(contents, prepared.rows) }.to_vec())
        }

        pub fn run_prepared_q4_dense_sweep(
            &self,
            prepared: &PreparedQuantizedSimdDense,
            input: &[f32],
            rows_per_simd: usize,
            simd_groups: usize,
            cache_input: bool,
        ) -> Result<Vec<f32>, String> {
            if input.len() != prepared.columns {
                return Err(format!(
                    "Q4 sweep input length {}, expected {}",
                    input.len(),
                    prepared.columns
                ));
            }
            if !matches!(rows_per_simd, 1 | 2 | 4 | 8) || !matches!(simd_groups, 1 | 2 | 4 | 8) {
                return Err("Q4 sweep geometry must use rows/groups 1, 2, 4, or 8".to_owned());
            }
            let rows_per_threadgroup = rows_per_simd * simd_groups;
            let threads_per_threadgroup = simd_groups * 32;
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let input_buffer = self.input_buffer(input_bytes)?;
            let output_buffer = self
                .device
                .newBufferWithLength_options(
                    prepared.rows * std::mem::size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal Q4 sweep output allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal Q4 sweep command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal Q4 sweep encoder allocation failed".to_owned())?;
            let pipeline = self.q4_dense_sweep_pipeline()?;
            encoder.setComputePipelineState(&pipeline);
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;
            let rows = prepared.rows as u32;
            let rows_per_simd = rows_per_simd as u32;
            let rows_per_threadgroup = rows_per_threadgroup as u32;
            let cache_input = u32::from(cache_input);
            let threads_per_threadgroup = threads_per_threadgroup as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&prepared.packed), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&prepared.scales), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 3);
                for (value, index) in [
                    (bytes_of(&columns), 4),
                    (bytes_of(&multiplier), 5),
                    (bytes_of(&rows), 6),
                    (bytes_of(&rows_per_simd), 7),
                    (bytes_of(&rows_per_threadgroup), 8),
                    (bytes_of(&cache_input), 9),
                    (bytes_of(&threads_per_threadgroup), 10),
                ] {
                    encoder.setBytes_length_atIndex(
                        NonNull::new(value.as_ptr() as *mut c_void)
                            .expect("non-empty scalar bytes")
                            .cast(),
                        value.len(),
                        index,
                    );
                }
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows.div_ceil(rows_per_threadgroup as usize),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads_per_threadgroup as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal Q4 sweep command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let contents = output_buffer.contents().as_ptr() as *const f32;
            Ok(unsafe { std::slice::from_raw_parts(contents, prepared.rows) }.to_vec())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn run_prepared_q4_qkv_sweep(
            &self,
            prepared: &PreparedQuantizedQkv,
            input: &[f32],
            norm_weights: &[f32],
            inverse_rms: f32,
            q_rows_per_simd: usize,
            kv_rows_per_simd: usize,
            simd_groups: usize,
            cache_input: bool,
            normalize: bool,
        ) -> Result<Vec<f32>, String> {
            if input.len() != prepared.columns || norm_weights.len() != prepared.columns {
                return Err("QKV sweep input and norm-weight lengths must match columns".to_owned());
            }
            if !matches!(q_rows_per_simd, 1 | 2 | 4 | 8)
                || !matches!(kv_rows_per_simd, 1 | 2 | 4 | 8)
                || !matches!(simd_groups, 1 | 2 | 4 | 8)
            {
                return Err("QKV sweep geometry must use rows/groups 1, 2, 4, or 8".to_owned());
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let norm_bytes = unsafe {
                std::slice::from_raw_parts(
                    norm_weights.as_ptr() as *const u8,
                    std::mem::size_of_val(norm_weights),
                )
            };
            let input_buffer = self.input_buffer(input_bytes)?;
            let norm_buffer = self.input_buffer(norm_bytes)?;
            let query_buffer = self
                .device
                .newBufferWithLength_options(
                    1024 * std::mem::size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal QKV sweep query allocation failed".to_owned())?;
            let key_buffer = self
                .device
                .newBufferWithLength_options(
                    512 * std::mem::size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal QKV sweep key allocation failed".to_owned())?;
            let value_buffer = self
                .device
                .newBufferWithLength_options(
                    512 * std::mem::size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal QKV sweep value allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal QKV sweep command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal QKV sweep encoder allocation failed".to_owned())?;
            let pipeline = self.q4_qkv_sweep_pipeline()?;
            encoder.setComputePipelineState(&pipeline);
            let columns = prepared.columns as u32;
            let multiplier = 1.0_f32;
            let q_rows_per_simd = q_rows_per_simd as u32;
            let kv_rows_per_simd = kv_rows_per_simd as u32;
            let simd_groups = simd_groups as u32;
            let cache_input = u32::from(cache_input);
            let normalize = u32::from(normalize);
            let q_threadgroups =
                1024_usize.div_ceil(q_rows_per_simd as usize * simd_groups as usize);
            let kv_threadgroups =
                512_usize.div_ceil(kv_rows_per_simd as usize * simd_groups as usize);
            unsafe {
                for (buffer, index) in [
                    (&prepared.q_packed, 0),
                    (&prepared.q_scales, 1),
                    (&prepared.k_packed, 2),
                    (&prepared.k_scales, 3),
                    (&prepared.v_packed, 4),
                    (&prepared.v_scales, 5),
                ] {
                    encoder.setBuffer_offset_atIndex(Some(&buffer.buffer), buffer.offset, index);
                }
                for (buffer, index) in [
                    (&input_buffer, 6),
                    (&query_buffer, 7),
                    (&key_buffer, 8),
                    (&value_buffer, 9),
                    (&norm_buffer, 17),
                ] {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
                }
                for (value, index) in [
                    (bytes_of(&columns), 10),
                    (bytes_of(&multiplier), 11),
                    (bytes_of(&q_rows_per_simd), 12),
                    (bytes_of(&kv_rows_per_simd), 13),
                    (bytes_of(&simd_groups), 14),
                    (bytes_of(&cache_input), 15),
                    (bytes_of(&normalize), 16),
                    (bytes_of(&inverse_rms), 18),
                ] {
                    encoder.setBytes_length_atIndex(
                        NonNull::new(value.as_ptr() as *mut c_void)
                            .expect("non-empty QKV sweep scalar bytes")
                            .cast(),
                        value.len(),
                        index,
                    );
                }
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: q_threadgroups + 2 * kv_threadgroups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: simd_groups as usize * 32,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal QKV sweep command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let query = unsafe {
                std::slice::from_raw_parts(query_buffer.contents().as_ptr() as *const f32, 1024)
            };
            let key = unsafe {
                std::slice::from_raw_parts(key_buffer.contents().as_ptr() as *const f32, 512)
            };
            let value = unsafe {
                std::slice::from_raw_parts(value_buffer.contents().as_ptr() as *const f32, 512)
            };
            let mut output = Vec::with_capacity(2048);
            output.extend_from_slice(query);
            output.extend_from_slice(key);
            output.extend_from_slice(value);
            Ok(output)
        }

        pub fn prepare_q4_qkv(
            &self,
            query: &PreparedQuantizedDense,
            key: &PreparedQuantizedDense,
            value: &PreparedQuantizedDense,
        ) -> Result<PreparedQuantizedQkv, String> {
            if query.rows != 1024
                || key.rows != 512
                || value.rows != 512
                || query.columns != 1024
                || key.columns != 1024
                || value.columns != 1024
            {
                return Err("fused QKV requires Granite 1024x1024/512 projections".to_owned());
            }
            Ok(PreparedQuantizedQkv {
                pipeline: self.q4_qkv_pipeline()?,
                simd_pipeline: self.q4_qkv_simd_pipeline()?,
                simd_norm_pipeline: self.q4_qkv_simd_norm_pipeline()?,
                split_simd_pipeline: self.q4_dense_simd_pipeline()?,
                split_simd_norm_pipeline: self.q4_simd_norm_gemv_pipeline()?,
                sweep_pipeline: self.q4_qkv_sweep_pipeline()?,
                q_packed: query.packed.clone(),
                q_scales: query.scales.clone(),
                k_packed: key.packed.clone(),
                k_scales: key.scales.clone(),
                v_packed: value.packed.clone(),
                v_scales: value.scales.clone(),
                columns: query.columns,
            })
        }

        pub fn prepare_dynamic_q4_moe(
            &self,
            gate_up_packed: &[u8],
            gate_up_scales: &[u8],
            down_packed: &[u8],
            down_scales: &[u8],
            layout: PackingLayout,
        ) -> Result<PreparedDynamicQ4Moe, String> {
            self.prepare_dynamic_q4_moe_with_layouts(
                gate_up_packed,
                gate_up_scales,
                down_packed,
                down_scales,
                layout,
                layout,
            )
        }

        pub fn prepare_dynamic_q4_moe_with_layouts(
            &self,
            gate_up_packed: &[u8],
            gate_up_scales: &[u8],
            down_packed: &[u8],
            down_scales: &[u8],
            gate_up_layout: PackingLayout,
            down_layout: PackingLayout,
        ) -> Result<PreparedDynamicQ4Moe, String> {
            const EXPERTS: usize = 32;
            const GATE_UP_PACKED: usize = 1024 * 1024 / 2;
            const GATE_UP_SCALES: usize = 1024 * 32 * 2;
            const DOWN_PACKED: usize = 1024 * 512 / 2;
            const DOWN_SCALES: usize = 1024 * 16 * 2;
            const Q4K_GATE_UP_PACKED: usize = 1024 * 4 * 140;
            const Q4K_GATE_UP_SCALES: usize = 1024 * 4 * 2 * 2;
            const Q4K_DOWN_PACKED: usize = 1024 * 2 * 140;
            const Q4K_DOWN_SCALES: usize = 1024 * 2 * 2 * 2;
            const Q4_TRANSACTION_GATE_UP_144: usize = 1024 * 4 * 144;
            const Q4_TRANSACTION_GATE_UP_160: usize = 1024 * 4 * 160;
            const Q4_TRANSACTION_DOWN_144: usize = 1024 * 2 * 144;
            const Q4_TRANSACTION_DOWN_160: usize = 1024 * 2 * 160;
            const Q4_CROSS_EXPERT_GATE_UP: usize = EXPERTS * 1024 * 32 * 18;
            const Q4_CROSS_EXPERT_DOWN: usize = EXPERTS * 1024 * 16 * 18;
            let is_q4_gate = gate_up_packed.len() == EXPERTS * GATE_UP_PACKED
                && gate_up_scales.len() == EXPERTS * GATE_UP_SCALES
                || gate_up_packed.len() == EXPERTS * 1024 * 32 * 18
                    && gate_up_scales.len() == EXPERTS * 1024 * 32 * 18;
            let is_q4_down = down_packed.len() == EXPERTS * DOWN_PACKED
                && down_scales.len() == EXPERTS * DOWN_SCALES
                || down_packed.len() == EXPERTS * 1024 * 16 * 18
                    && down_scales.len() == EXPERTS * 1024 * 16 * 18;
            let is_q4_transaction_gate = gate_up_layout.is_q4_transaction()
                && gate_up_packed.len()
                    == EXPERTS
                        * match gate_up_layout {
                            PackingLayout::Q4Transaction144 => Q4_TRANSACTION_GATE_UP_144,
                            PackingLayout::Q4Transaction160 => Q4_TRANSACTION_GATE_UP_160,
                            _ => 0,
                        }
                && gate_up_scales.len() == gate_up_packed.len();
            let is_q4_transaction_down = down_layout.is_q4_transaction()
                && down_packed.len()
                    == EXPERTS
                        * match down_layout {
                            PackingLayout::Q4Transaction144 => Q4_TRANSACTION_DOWN_144,
                            PackingLayout::Q4Transaction160 => Q4_TRANSACTION_DOWN_160,
                            _ => 0,
                        }
                && down_scales.len() == down_packed.len();
            let is_q4_cross_expert_gate = gate_up_layout == PackingLayout::Q4CrossExpertTile8
                && gate_up_packed.len() == Q4_CROSS_EXPERT_GATE_UP
                && gate_up_scales.len() == Q4_CROSS_EXPERT_GATE_UP;
            let is_q4_cross_expert_down = down_layout == PackingLayout::Q4CrossExpertTile8
                && down_packed.len() == Q4_CROSS_EXPERT_DOWN
                && down_scales.len() == Q4_CROSS_EXPERT_DOWN;
            let is_q4_gate = is_q4_gate || is_q4_transaction_gate || is_q4_cross_expert_gate;
            let is_q4_down = is_q4_down || is_q4_transaction_down || is_q4_cross_expert_down;
            let is_q4 = is_q4_gate && is_q4_down;
            let is_q4k = gate_up_packed.len() == EXPERTS * Q4K_GATE_UP_PACKED
                && gate_up_scales.len() == EXPERTS * Q4K_GATE_UP_SCALES
                && down_packed.len() == EXPERTS * Q4K_DOWN_PACKED
                && down_scales.len() == EXPERTS * Q4K_DOWN_SCALES;
            if (!is_q4 && !is_q4k)
                || (gate_up_layout.is_q4_transaction() && !is_q4_transaction_gate)
                || (down_layout.is_q4_transaction() && !is_q4_transaction_down)
                || (gate_up_layout == PackingLayout::Q4CrossExpertTile8 && !is_q4_cross_expert_gate)
                || (down_layout == PackingLayout::Q4CrossExpertTile8 && !is_q4_cross_expert_down)
            {
                return Err("dynamic Q4 MoE buffers have an invalid Granite shape".to_owned());
            }
            let gate_up_runtime = self.input_runtime_buffer(gate_up_packed)?;
            let gate_up_scales_runtime = if is_q4_cross_expert_gate {
                gate_up_runtime.clone()
            } else {
                self.input_runtime_buffer(gate_up_scales)?
            };
            let down_runtime = self.input_runtime_buffer(down_packed)?;
            let down_scales_runtime = if is_q4_cross_expert_down {
                down_runtime.clone()
            } else {
                self.input_runtime_buffer(down_scales)?
            };
            let gate_up_pipeline = if is_q4k {
                "qx_dynamic_q4k_moe_gate_up_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_simd"
            };
            let gate_up_norm_pipeline = if is_q4k {
                "qx_dynamic_q4k_moe_gate_up_norm_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_norm_simd"
            };
            let down_pipeline = if is_q4k {
                "qx_dynamic_q4k_moe_down_simd"
            } else {
                "qx_dynamic_q4_moe_down_simd"
            };
            Ok(PreparedDynamicQ4Moe {
                pipeline: self.dynamic_moe_pipeline()?,
                simd_pipeline: self.dynamic_moe_simd_pipeline()?,
                batched_gate_up_pipeline: self.dynamic_moe_named_pipeline(gate_up_pipeline)?,
                batched_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline(gate_up_norm_pipeline)?,
                inline_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_simd")?,
                inline_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_norm_simd")?,
                inline_gate_up_vec4_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_vec4_simd")?,
                inline_gate_up_vec4_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_vec4_norm_simd",
                )?,
                inline_gate_up_nocache_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_nocache_simd")?,
                inline_gate_up_nocache_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_nocache_norm_simd",
                )?,
                cross_expert_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_cross_expert_simd")?,
                cross_expert_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_norm_simd",
                )?,
                cross_expert_tile8_gate_up_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_tile8_simd",
                )?,
                cross_expert_tile8_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_tile8_norm_simd",
                )?,
                row_stream_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_row_stream_simd")?,
                row_stream_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_row_stream_norm_simd")?,
                llama_like_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_llama_like_simd")?,
                llama_like_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_llama_like_norm_simd")?,
                decode_direct_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_decode_direct_simd")?,
                decode_direct_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_decode_direct_norm_simd",
                )?,
                two_row_dataflow_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_dataflow_simd",
                )?),
                two_row_dataflow_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_dataflow_norm_simd",
                )?),
                q4k_mature_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_simd")?,
                q4k_mature_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_norm_simd")?,
                fused_block_qx_gate_up_pipeline: Some(
                    self.dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_fused_block_qx_simd",
                    )?,
                ),
                fused_block_qx_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_fused_block_qx_norm_simd",
                )?),
                fused_block_llama_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_fused_block_llama_simd",
                )?),
                fused_block_llama_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_fused_block_llama_norm_simd",
                )?),
                inline_gate_up_rows2_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_rows2_simd")?,
                ),
                inline_gate_up_rows2_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_rows2_norm_simd",
                )?),
                inline_gate_up_rows4_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_rows4_simd")?,
                ),
                inline_gate_up_rows4_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_rows4_norm_simd",
                )?),
                inline_gate_up_rows8_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_rows8_simd")?,
                ),
                inline_gate_up_rows8_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_rows8_norm_simd",
                )?),
                wide_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_wide_simd")?,
                wide_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_wide_norm_simd")?,
                wide_supported: is_q4,
                expertwise: false,
                expertwise_q4_gate_up_pipeline: None,
                expertwise_q4_gate_up_norm_pipeline: None,
                expertwise_q3_gate_up_pipeline: None,
                expertwise_q3_gate_up_norm_pipeline: None,
                expertwise_q4_down_pipeline: None,
                expertwise_q3_down_pipeline: None,
                expertwise_q4_gate_up_packed: None,
                expertwise_q4_gate_up_scales: None,
                expertwise_q3_gate_up_packed: None,
                expertwise_q3_gate_up_scales: None,
                expertwise_gate_up_map: None,
                expertwise_q4_down_packed: None,
                expertwise_q4_down_scales: None,
                expertwise_q3_down_packed: None,
                expertwise_q3_down_scales: None,
                expertwise_down_map: None,
                expertwise_q3_gate_up_block_bytes: None,
                expertwise_q3_down_block_bytes: None,
                q3_gate_up_pipeline: None,
                q3_gate_up_norm_pipeline: None,
                q3_gate_up_packed: None,
                q3_gate_up_scales: None,
                q3_gate_up_block_bytes: None,
                q3_down_pipeline: None,
                q3_down_packed: None,
                q3_down_scales: None,
                q3_down_block_bytes: None,
                batched_gate_up_touch_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch")?,
                batched_gate_up_touch_full_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_full")?,
                batched_gate_up_touch_direct_1_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_1")?,
                batched_gate_up_touch_direct_2_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_2")?,
                batched_gate_up_touch_direct_4_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_4")?,
                batched_gate_up_touch_direct_8_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_8")?,
                batched_gate_up_synthetic_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_synthetic")?,
                batched_gate_up_synthetic_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_synthetic_norm")?,
                batched_projection_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_projection_simd")?,
                batched_projection_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_projection_norm_simd")?,
                batched_silu_pipeline: self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_silu")?,
                batched_down_pipeline: self.dynamic_moe_named_pipeline(down_pipeline)?,
                inline_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_inline_simd")?,
                decode_direct_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_decode_direct_simd")?,
                q4k_mature_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_down_mature_simd")?,
                cross_expert_tile8_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cross_expert_tile8_simd")?,
                inline_layout: gate_up_layout == PackingLayout::Q4RowInline,
                down_inline_layout: down_layout.is_q4_row_inline(),
                wide_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_wide_simd")?,
                batched_accumulate_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_accumulate")?,
                gate_up_packed: gate_up_runtime,
                gate_up_scales: gate_up_scales_runtime,
                split_gate_packed: None,
                split_gate_scales: None,
                split_up_packed: None,
                split_up_scales: None,
                split_gate_up: false,
                down_packed: down_runtime,
                down_scales: down_scales_runtime,
                layout: gate_up_layout,
                gate_up_layout,
                down_layout,
            })
        }

        /// Prepare a dynamic MoE directly from offset-bearing views into a
        /// static weight slab. No tensor payload is copied or reconstructed.
        pub fn prepare_dynamic_q4_moe_runtime_views(
            &self,
            gate_up_packed: RuntimeBuffer,
            gate_up_scales: RuntimeBuffer,
            down_packed: RuntimeBuffer,
            down_scales: RuntimeBuffer,
            gate_up_layout: PackingLayout,
            down_layout: PackingLayout,
        ) -> Result<PreparedDynamicQ4Moe, String> {
            const EXPERTS: usize = 32;
            const Q4_GATE_UP_PACKED: usize = 1024 * 1024 / 2;
            const Q4_GATE_UP_SCALES: usize = 1024 * 32 * 2;
            const Q4_DOWN_PACKED: usize = 1024 * 512 / 2;
            const Q4_DOWN_SCALES: usize = 1024 * 16 * 2;
            const Q4K_GATE_UP_PACKED: usize = 1024 * 4 * 140;
            const Q4K_GATE_UP_SCALES: usize = 1024 * 4 * 2 * 2;
            const Q4K_DOWN_PACKED: usize = 1024 * 2 * 140;
            const Q4K_DOWN_SCALES: usize = 1024 * 2 * 2 * 2;
            const Q4_INLINE_GATE_UP: usize = 1024 * 32 * 18;
            const Q4_INLINE_DOWN: usize = 1024 * 16 * 18;
            const Q4_TRANSACTION_GATE_UP_144: usize = 1024 * 4 * 144;
            const Q4_TRANSACTION_GATE_UP_160: usize = 1024 * 4 * 160;
            const Q4_TRANSACTION_DOWN_144: usize = 1024 * 2 * 144;
            const Q4_TRANSACTION_DOWN_160: usize = 1024 * 2 * 160;
            const Q4_CROSS_EXPERT_GATE_UP: usize = EXPERTS * 1024 * 32 * 18;
            const Q4_CROSS_EXPERT_DOWN: usize = EXPERTS * 1024 * 16 * 18;
            const Q3_GATE_UP_12: usize = 1024 * 32 * 12;
            const Q3_GATE_UP_16: usize = 1024 * 32 * 16;
            const Q3_DOWN_12: usize = 1024 * 16 * 12;
            const Q3_DOWN_16: usize = 1024 * 16 * 16;
            let is_q4_gate = gate_up_packed.length == EXPERTS * Q4_GATE_UP_PACKED
                && gate_up_scales.length == EXPERTS * Q4_GATE_UP_SCALES
                || gate_up_packed.length == EXPERTS * Q4_INLINE_GATE_UP
                    && gate_up_scales.length == EXPERTS * Q4_INLINE_GATE_UP;
            let is_q4_down = down_packed.length == EXPERTS * Q4_DOWN_PACKED
                && down_scales.length == EXPERTS * Q4_DOWN_SCALES
                || down_packed.length == EXPERTS * Q4_INLINE_DOWN
                    && down_scales.length == EXPERTS * Q4_INLINE_DOWN;
            let is_q4_transaction_gate = gate_up_layout.is_q4_transaction()
                && gate_up_packed.length
                    == EXPERTS
                        * match gate_up_layout {
                            PackingLayout::Q4Transaction144 => Q4_TRANSACTION_GATE_UP_144,
                            PackingLayout::Q4Transaction160 => Q4_TRANSACTION_GATE_UP_160,
                            _ => 0,
                        }
                && gate_up_scales.length == gate_up_packed.length;
            let is_q4_transaction_down = down_layout.is_q4_transaction()
                && down_packed.length
                    == EXPERTS
                        * match down_layout {
                            PackingLayout::Q4Transaction144 => Q4_TRANSACTION_DOWN_144,
                            PackingLayout::Q4Transaction160 => Q4_TRANSACTION_DOWN_160,
                            _ => 0,
                        }
                && down_scales.length == down_packed.length;
            let is_q4_cross_expert_gate = gate_up_layout == PackingLayout::Q4CrossExpertTile8
                && gate_up_packed.length == Q4_CROSS_EXPERT_GATE_UP
                && gate_up_scales.length == Q4_CROSS_EXPERT_GATE_UP;
            let is_q4_cross_expert_down = down_layout == PackingLayout::Q4CrossExpertTile8
                && down_packed.length == Q4_CROSS_EXPERT_DOWN
                && down_scales.length == Q4_CROSS_EXPERT_DOWN;
            let is_q4_gate = is_q4_gate || is_q4_transaction_gate || is_q4_cross_expert_gate;
            let is_q4_down = is_q4_down || is_q4_transaction_down || is_q4_cross_expert_down;
            let is_q4 = is_q4_gate && is_q4_down;
            let is_q4k = gate_up_packed.length == EXPERTS * Q4K_GATE_UP_PACKED
                && gate_up_scales.length == EXPERTS * Q4K_GATE_UP_SCALES
                && down_packed.length == EXPERTS * Q4K_DOWN_PACKED
                && down_scales.length == EXPERTS * Q4K_DOWN_SCALES;
            let is_inline = gate_up_packed.length == EXPERTS * Q4_INLINE_GATE_UP
                && gate_up_scales.length == EXPERTS * Q4_INLINE_GATE_UP
                && down_packed.length == EXPERTS * Q4_INLINE_DOWN
                && down_scales.length == EXPERTS * Q4_INLINE_DOWN;
            let is_q3_gate = (gate_up_packed.length == EXPERTS * Q3_GATE_UP_12
                || gate_up_packed.length == EXPERTS * Q3_GATE_UP_16)
                && gate_up_scales.length == EXPERTS * 1024 * 32 * 2;
            let is_q3_down = (down_packed.length == EXPERTS * Q3_DOWN_12
                || down_packed.length == EXPERTS * Q3_DOWN_16)
                && down_scales.length == EXPERTS * 1024 * 16 * 2;
            let is_q3 = is_q3_gate && is_q3_down;
            let is_q3_gate_q4_down = is_q3_gate && is_q4_down;
            let is_q4_gate_q3_down = is_q4_gate && is_q3_down;
            if (!is_q4
                && !is_q4k
                && !is_inline
                && !is_q3
                && !is_q3_gate_q4_down
                && !is_q4_gate_q3_down)
                || (gate_up_layout.is_q4_transaction() && !is_q4_transaction_gate)
                || (down_layout.is_q4_transaction() && !is_q4_transaction_down)
                || (gate_up_layout == PackingLayout::Q4CrossExpertTile8 && !is_q4_cross_expert_gate)
                || (down_layout == PackingLayout::Q4CrossExpertTile8 && !is_q4_cross_expert_down)
            {
                return Err("dynamic Q4 MoE slab views have an invalid Granite shape".to_owned());
            }
            let gate_up_pipeline = if is_q4k {
                "qx_dynamic_q4k_moe_gate_up_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_simd"
            };
            let gate_up_norm_pipeline = if is_q4k {
                "qx_dynamic_q4k_moe_gate_up_norm_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_norm_simd"
            };
            let down_pipeline = if is_q4k {
                "qx_dynamic_q4k_moe_down_simd"
            } else {
                "qx_dynamic_q4_moe_down_simd"
            };
            let mut prepared = PreparedDynamicQ4Moe {
                pipeline: self.dynamic_moe_pipeline()?,
                simd_pipeline: self.dynamic_moe_simd_pipeline()?,
                batched_gate_up_pipeline: self.dynamic_moe_named_pipeline(gate_up_pipeline)?,
                batched_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline(gate_up_norm_pipeline)?,
                inline_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_simd")?,
                inline_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_norm_simd")?,
                inline_gate_up_vec4_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_vec4_simd")?,
                inline_gate_up_vec4_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_vec4_norm_simd",
                )?,
                inline_gate_up_nocache_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_nocache_simd")?,
                inline_gate_up_nocache_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_nocache_norm_simd",
                )?,
                cross_expert_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_cross_expert_simd")?,
                cross_expert_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_norm_simd",
                )?,
                cross_expert_tile8_gate_up_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_tile8_simd",
                )?,
                cross_expert_tile8_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_tile8_norm_simd",
                )?,
                row_stream_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_row_stream_simd")?,
                row_stream_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_row_stream_norm_simd")?,
                llama_like_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_llama_like_simd")?,
                llama_like_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_llama_like_norm_simd")?,
                decode_direct_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_decode_direct_simd")?,
                decode_direct_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_decode_direct_norm_simd",
                )?,
                two_row_dataflow_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_dataflow_simd",
                )?),
                two_row_dataflow_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_dataflow_norm_simd",
                )?),
                q4k_mature_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_simd")?,
                q4k_mature_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_norm_simd")?,
                fused_block_qx_gate_up_pipeline: Some(
                    self.dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_fused_block_qx_simd",
                    )?,
                ),
                fused_block_qx_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_fused_block_qx_norm_simd",
                )?),
                fused_block_llama_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_fused_block_llama_simd",
                )?),
                fused_block_llama_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_fused_block_llama_norm_simd",
                )?),
                inline_gate_up_rows2_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_rows2_simd")?,
                ),
                inline_gate_up_rows2_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_rows2_norm_simd",
                )?),
                inline_gate_up_rows4_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_rows4_simd")?,
                ),
                inline_gate_up_rows4_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_rows4_norm_simd",
                )?),
                inline_gate_up_rows8_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_rows8_simd")?,
                ),
                inline_gate_up_rows8_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_rows8_norm_simd",
                )?),
                wide_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_wide_simd")?,
                wide_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_wide_norm_simd")?,
                wide_supported: is_q4,
                expertwise: false,
                expertwise_q4_gate_up_pipeline: None,
                expertwise_q4_gate_up_norm_pipeline: None,
                expertwise_q3_gate_up_pipeline: None,
                expertwise_q3_gate_up_norm_pipeline: None,
                expertwise_q4_down_pipeline: None,
                expertwise_q3_down_pipeline: None,
                expertwise_q4_gate_up_packed: None,
                expertwise_q4_gate_up_scales: None,
                expertwise_q3_gate_up_packed: None,
                expertwise_q3_gate_up_scales: None,
                expertwise_gate_up_map: None,
                expertwise_q4_down_packed: None,
                expertwise_q4_down_scales: None,
                expertwise_q3_down_packed: None,
                expertwise_q3_down_scales: None,
                expertwise_down_map: None,
                expertwise_q3_gate_up_block_bytes: None,
                expertwise_q3_down_block_bytes: None,
                q3_gate_up_pipeline: None,
                q3_gate_up_norm_pipeline: None,
                q3_gate_up_packed: None,
                q3_gate_up_scales: None,
                q3_gate_up_block_bytes: None,
                q3_down_pipeline: None,
                q3_down_packed: None,
                q3_down_scales: None,
                q3_down_block_bytes: None,
                batched_gate_up_touch_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch")?,
                batched_gate_up_touch_full_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_full")?,
                batched_gate_up_touch_direct_1_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_1")?,
                batched_gate_up_touch_direct_2_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_2")?,
                batched_gate_up_touch_direct_4_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_4")?,
                batched_gate_up_touch_direct_8_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_8")?,
                batched_gate_up_synthetic_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_synthetic")?,
                batched_gate_up_synthetic_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_synthetic_norm")?,
                batched_projection_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_projection_simd")?,
                batched_projection_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_projection_norm_simd")?,
                batched_silu_pipeline: self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_silu")?,
                batched_down_pipeline: self.dynamic_moe_named_pipeline(down_pipeline)?,
                inline_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_inline_simd")?,
                decode_direct_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_decode_direct_simd")?,
                q4k_mature_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_down_mature_simd")?,
                cross_expert_tile8_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cross_expert_tile8_simd")?,
                inline_layout: gate_up_layout == PackingLayout::Q4RowInline,
                down_inline_layout: down_layout.is_q4_row_inline(),
                wide_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_wide_simd")?,
                batched_accumulate_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_accumulate")?,
                gate_up_packed,
                gate_up_scales,
                split_gate_packed: None,
                split_gate_scales: None,
                split_up_packed: None,
                split_up_scales: None,
                split_gate_up: false,
                down_packed,
                down_scales,
                layout: gate_up_layout,
                gate_up_layout,
                down_layout,
            };
            if is_q3_gate {
                prepared.q3_gate_up_pipeline =
                    Some(self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_gate_up_simd")?);
                prepared.q3_gate_up_norm_pipeline =
                    Some(self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_gate_up_norm_simd")?);
                prepared.q3_gate_up_packed = Some(prepared.gate_up_packed.clone());
                prepared.q3_gate_up_scales = Some(prepared.gate_up_scales.clone());
                prepared.q3_gate_up_block_bytes = Some(
                    (prepared.q3_gate_up_packed.as_ref().unwrap().length / (EXPERTS * 1024 * 32))
                        as u32,
                );
                prepared.inline_layout = false;
            }
            if is_q3_down {
                prepared.q3_down_pipeline =
                    Some(self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_down_simd")?);
                prepared.q3_down_packed = Some(prepared.down_packed.clone());
                prepared.q3_down_scales = Some(prepared.down_scales.clone());
                prepared.q3_down_block_bytes = Some(
                    (prepared.q3_down_packed.as_ref().unwrap().length / (EXPERTS * 1024 * 16))
                        as u32,
                );
            }
            Ok(prepared)
        }

        /// Prepare a Q3 gate/up and Q3 down pair directly from immutable
        /// mapped slab views. This is the no-copy path for a uniform Q3
        /// expert plan; dense Q4 preparation remains available through the
        /// neighboring constructor.
        #[allow(clippy::too_many_arguments)]
        pub fn prepare_dynamic_q4_moe_expertwise_runtime_views(
            &self,
            q4_gate_up_packed: Option<RuntimeBuffer>,
            q4_gate_up_scales: Option<RuntimeBuffer>,
            q3_gate_up_packed: Option<RuntimeBuffer>,
            q3_gate_up_scales: Option<RuntimeBuffer>,
            gate_up_map: RuntimeBuffer,
            q4_down_packed: Option<RuntimeBuffer>,
            q4_down_scales: Option<RuntimeBuffer>,
            q3_down_packed: Option<RuntimeBuffer>,
            q3_down_scales: Option<RuntimeBuffer>,
            down_map: RuntimeBuffer,
            q3_gate_up_block_bytes: Option<usize>,
            q3_down_block_bytes: Option<usize>,
            q4_gate_up_inline: bool,
            q4_down_inline: bool,
        ) -> Result<PreparedDynamicQ4Moe, String> {
            if gate_up_map.length < 32 * std::mem::size_of::<u32>()
                || down_map.length < 32 * std::mem::size_of::<u32>()
            {
                return Err("expertwise MoE maps must contain 32 u32 entries".to_owned());
            }
            for value in [q3_gate_up_block_bytes, q3_down_block_bytes]
                .into_iter()
                .flatten()
            {
                if !matches!(value, 12 | 16) {
                    return Err("expertwise Q3 block size must be 12 or 16 bytes".to_owned());
                }
            }
            let dummy = self.allocate_runtime_buffer(4)?;
            let q4_gate_up_packed = q4_gate_up_packed.unwrap_or_else(|| dummy.clone());
            let q4_gate_up_scales = q4_gate_up_scales.unwrap_or_else(|| dummy.clone());
            let q3_gate_up_packed = q3_gate_up_packed.unwrap_or_else(|| dummy.clone());
            let q3_gate_up_scales = q3_gate_up_scales.unwrap_or_else(|| dummy.clone());
            let q4_down_packed = q4_down_packed.unwrap_or_else(|| dummy.clone());
            let q4_down_scales = q4_down_scales.unwrap_or_else(|| dummy.clone());
            let q3_down_packed = q3_down_packed.unwrap_or_else(|| dummy.clone());
            let q3_down_scales = q3_down_scales.unwrap_or_else(|| dummy.clone());
            let prepared = PreparedDynamicQ4Moe {
                pipeline: self.dynamic_moe_pipeline()?,
                simd_pipeline: self.dynamic_moe_simd_pipeline()?,
                batched_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_simd")?,
                batched_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_norm_simd")?,
                inline_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_simd")?,
                inline_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_norm_simd")?,
                inline_gate_up_vec4_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_vec4_simd")?,
                inline_gate_up_vec4_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_vec4_norm_simd",
                )?,
                inline_gate_up_nocache_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_nocache_simd")?,
                inline_gate_up_nocache_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_nocache_norm_simd",
                )?,
                cross_expert_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_cross_expert_simd")?,
                cross_expert_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_norm_simd",
                )?,
                cross_expert_tile8_gate_up_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_tile8_simd",
                )?,
                cross_expert_tile8_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_tile8_norm_simd",
                )?,
                llama_like_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_llama_like_simd")?,
                llama_like_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_llama_like_norm_simd")?,
                decode_direct_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_decode_direct_simd")?,
                decode_direct_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_decode_direct_norm_simd",
                )?,
                two_row_dataflow_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_dataflow_simd",
                )?),
                two_row_dataflow_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_dataflow_norm_simd",
                )?),
                q4k_mature_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_simd")?,
                q4k_mature_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_norm_simd")?,
                row_stream_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_row_stream_simd")?,
                row_stream_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_row_stream_norm_simd")?,
                fused_block_qx_gate_up_pipeline: None,
                fused_block_qx_gate_up_norm_pipeline: None,
                fused_block_llama_gate_up_pipeline: None,
                fused_block_llama_gate_up_norm_pipeline: None,
                inline_gate_up_rows2_pipeline: None,
                inline_gate_up_rows2_norm_pipeline: None,
                inline_gate_up_rows4_pipeline: None,
                inline_gate_up_rows4_norm_pipeline: None,
                inline_gate_up_rows8_pipeline: None,
                inline_gate_up_rows8_norm_pipeline: None,
                wide_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_wide_simd")?,
                wide_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_wide_norm_simd")?,
                wide_supported: false,
                expertwise: true,
                expertwise_q4_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    if q4_gate_up_inline {
                        "qx_dynamic_q4_moe_gate_up_expertwise_inline_simd"
                    } else {
                        "qx_dynamic_q4_moe_gate_up_expertwise_simd"
                    },
                )?),
                expertwise_q4_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    if q4_gate_up_inline {
                        "qx_dynamic_q4_moe_gate_up_expertwise_inline_norm_simd"
                    } else {
                        "qx_dynamic_q4_moe_gate_up_expertwise_norm_simd"
                    },
                )?),
                expertwise_q3_gate_up_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_gate_up_expertwise_simd")?,
                ),
                expertwise_q3_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q3_moe_gate_up_expertwise_norm_simd",
                )?),
                expertwise_q4_down_pipeline: Some(self.dynamic_moe_named_pipeline(
                    if q4_down_inline {
                        "qx_dynamic_q4_moe_down_expertwise_inline_simd"
                    } else {
                        "qx_dynamic_q4_moe_down_expertwise_simd"
                    },
                )?),
                expertwise_q3_down_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_down_expertwise_simd")?,
                ),
                expertwise_q4_gate_up_packed: Some(q4_gate_up_packed.clone()),
                expertwise_q4_gate_up_scales: Some(q4_gate_up_scales.clone()),
                expertwise_q3_gate_up_packed: Some(q3_gate_up_packed.clone()),
                expertwise_q3_gate_up_scales: Some(q3_gate_up_scales.clone()),
                expertwise_gate_up_map: Some(gate_up_map),
                expertwise_q4_down_packed: Some(q4_down_packed.clone()),
                expertwise_q4_down_scales: Some(q4_down_scales.clone()),
                expertwise_q3_down_packed: Some(q3_down_packed.clone()),
                expertwise_q3_down_scales: Some(q3_down_scales.clone()),
                expertwise_down_map: Some(down_map),
                expertwise_q3_gate_up_block_bytes: q3_gate_up_block_bytes.map(|value| value as u32),
                expertwise_q3_down_block_bytes: q3_down_block_bytes.map(|value| value as u32),
                q3_gate_up_pipeline: None,
                q3_gate_up_norm_pipeline: None,
                q3_gate_up_packed: None,
                q3_gate_up_scales: None,
                q3_gate_up_block_bytes: None,
                q3_down_pipeline: None,
                q3_down_packed: None,
                q3_down_scales: None,
                q3_down_block_bytes: None,
                batched_gate_up_touch_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch")?,
                batched_gate_up_touch_full_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_full")?,
                batched_gate_up_touch_direct_1_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_1")?,
                batched_gate_up_touch_direct_2_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_2")?,
                batched_gate_up_touch_direct_4_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_4")?,
                batched_gate_up_touch_direct_8_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_8")?,
                batched_gate_up_synthetic_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_synthetic")?,
                batched_gate_up_synthetic_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_synthetic_norm")?,
                batched_projection_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_projection_simd")?,
                batched_projection_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_projection_norm_simd")?,
                batched_silu_pipeline: self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_silu")?,
                batched_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_simd")?,
                inline_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_inline_simd")?,
                decode_direct_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_decode_direct_simd")?,
                q4k_mature_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_down_mature_simd")?,
                cross_expert_tile8_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cross_expert_tile8_simd")?,
                inline_layout: false,
                down_inline_layout: false,
                wide_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_wide_simd")?,
                batched_accumulate_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_accumulate")?,
                gate_up_packed: q4_gate_up_packed,
                gate_up_scales: q4_gate_up_scales,
                split_gate_packed: None,
                split_gate_scales: None,
                split_up_packed: None,
                split_up_scales: None,
                split_gate_up: false,
                down_packed: q4_down_packed,
                down_scales: q4_down_scales,
                layout: PackingLayout::RowMajor,
                gate_up_layout: PackingLayout::RowMajor,
                down_layout: PackingLayout::RowMajor,
            };
            // Keep the fields live in debug diagnostics and make the intent
            // explicit: both typed dispatch families are required even when
            // one representation has no selected expert in a particular
            // layer; the map makes those dispatches no-ops.
            if prepared.expertwise_q3_gate_up_block_bytes.is_none()
                && prepared.expertwise_q3_down_block_bytes.is_none()
            {
                return Err(
                    "expertwise preparation requires at least one Q3 tensor family".to_owned(),
                );
            }
            Ok(prepared)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn prepare_dynamic_q3_moe_runtime_views(
            &self,
            gate_up_q3_packed: RuntimeBuffer,
            gate_up_q3_scales: RuntimeBuffer,
            down_q3_packed: RuntimeBuffer,
            down_q3_scales: RuntimeBuffer,
            block_bytes: usize,
        ) -> Result<PreparedDynamicQ4Moe, String> {
            const EXPERTS: usize = 32;
            const HIDDEN: usize = 1024;
            const GATE_GROUPS: usize = 32;
            const DOWN_GROUPS: usize = 16;
            if !matches!(block_bytes, 12 | 16)
                || gate_up_q3_packed.length != EXPERTS * HIDDEN * GATE_GROUPS * block_bytes
                || gate_up_q3_scales.length != EXPERTS * HIDDEN * GATE_GROUPS * 2
                || down_q3_packed.length != EXPERTS * HIDDEN * DOWN_GROUPS * block_bytes
                || down_q3_scales.length != EXPERTS * HIDDEN * DOWN_GROUPS * 2
            {
                return Err("Q3 runtime expert slabs have invalid Granite shapes".to_owned());
            }
            let mut prepared = self.prepare_dynamic_q4_moe_runtime_views(
                // These fields are not consumed by the Q3 pipelines, but
                // retain valid fixed-shape views for the shared struct and
                // diagnostics.
                gate_up_q3_packed.clone(),
                gate_up_q3_scales.clone(),
                down_q3_packed.clone(),
                down_q3_scales.clone(),
                PackingLayout::RowMajor,
                PackingLayout::RowMajor,
            )?;
            prepared.q3_gate_up_pipeline =
                Some(self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_gate_up_simd")?);
            prepared.q3_gate_up_norm_pipeline =
                Some(self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_gate_up_norm_simd")?);
            prepared.q3_gate_up_packed = Some(gate_up_q3_packed);
            prepared.q3_gate_up_scales = Some(gate_up_q3_scales);
            prepared.q3_gate_up_block_bytes = Some(block_bytes as u32);
            prepared.q3_down_pipeline =
                Some(self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_down_simd")?);
            prepared.q3_down_packed = Some(down_q3_packed);
            prepared.q3_down_scales = Some(down_q3_scales);
            prepared.q3_down_block_bytes = Some(block_bytes as u32);
            Ok(prepared)
        }

        /// Add an opt-in Q3 gate/up representation to an otherwise canonical
        /// Q4 MoE preparation.  The down projection and every other runtime
        /// tensor remain Q4. `block_bytes` is 12 for dense Q3 or 16 for the
        /// transaction-native padded representation.
        #[allow(clippy::too_many_arguments)]
        pub fn prepare_dynamic_q4_moe_with_q3_gate_up(
            &self,
            gate_up_q3_packed: &[u8],
            gate_up_q3_scales: &[u8],
            block_bytes: usize,
            gate_up_q4_packed: &[u8],
            gate_up_q4_scales: &[u8],
            down_packed: &[u8],
            down_scales: &[u8],
            down_layout: PackingLayout,
        ) -> Result<PreparedDynamicQ4Moe, String> {
            const EXPERTS: usize = 32;
            const HIDDEN: usize = 1024;
            const GROUPS: usize = 32;
            if !matches!(block_bytes, 12 | 16) {
                return Err("Q3 gate/up block size must be 12 or 16 bytes".to_owned());
            }
            let expected_packed = EXPERTS * HIDDEN * GROUPS * block_bytes;
            let expected_scales = EXPERTS * HIDDEN * GROUPS * 2;
            if gate_up_q3_packed.len() != expected_packed
                || gate_up_q3_scales.len() != expected_scales
            {
                return Err(format!(
                    "Q3 gate/up buffers have invalid lengths: packed {}, expected {}; scales {}, expected {}",
                    gate_up_q3_packed.len(),
                    expected_packed,
                    gate_up_q3_scales.len(),
                    expected_scales
                ));
            }
            let mut prepared = self.prepare_dynamic_q4_moe_with_layouts(
                gate_up_q4_packed,
                gate_up_q4_scales,
                down_packed,
                down_scales,
                PackingLayout::RowMajor,
                down_layout,
            )?;
            prepared.q3_gate_up_pipeline =
                Some(self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_gate_up_simd")?);
            prepared.q3_gate_up_norm_pipeline =
                Some(self.dynamic_moe_named_pipeline("qx_dynamic_q3_moe_gate_up_norm_simd")?);
            prepared.q3_gate_up_packed = Some(self.input_runtime_buffer(gate_up_q3_packed)?);
            prepared.q3_gate_up_scales = Some(self.input_runtime_buffer(gate_up_q3_scales)?);
            prepared.q3_gate_up_block_bytes = Some(block_bytes as u32);
            Ok(prepared)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn prepare_dynamic_q4_moe_with_split_layouts(
            &self,
            gate_up_packed: &[u8],
            gate_up_scales: &[u8],
            split_gate_packed: &[u8],
            split_gate_scales: &[u8],
            split_up_packed: &[u8],
            split_up_scales: &[u8],
            down_packed: &[u8],
            down_scales: &[u8],
            gate_up_layout: PackingLayout,
            down_layout: PackingLayout,
        ) -> Result<PreparedDynamicQ4Moe, String> {
            const EXPERTS: usize = 32;
            const GATE_UP_PACKED: usize = 1024 * 1024 / 2;
            const GATE_UP_SCALES: usize = 1024 * 32 * 2;
            const SPLIT_PROJECTION_PACKED: usize = 512 * 1024 / 2;
            const SPLIT_PROJECTION_SCALES: usize = 512 * 32 * 2;
            const DOWN_PACKED: usize = 1024 * 512 / 2;
            const DOWN_SCALES: usize = 1024 * 16 * 2;
            if gate_up_packed.len() != EXPERTS * GATE_UP_PACKED
                || gate_up_scales.len() != EXPERTS * GATE_UP_SCALES
                || split_gate_packed.len() != EXPERTS * SPLIT_PROJECTION_PACKED
                || split_gate_scales.len() != EXPERTS * SPLIT_PROJECTION_SCALES
                || split_up_packed.len() != EXPERTS * SPLIT_PROJECTION_PACKED
                || split_up_scales.len() != EXPERTS * SPLIT_PROJECTION_SCALES
                || down_packed.len() != EXPERTS * DOWN_PACKED
                || down_scales.len() != EXPERTS * DOWN_SCALES
            {
                return Err("split dynamic Q4 MoE buffers have an invalid Granite shape".to_owned());
            }
            Ok(PreparedDynamicQ4Moe {
                pipeline: self.dynamic_moe_pipeline()?,
                simd_pipeline: self.dynamic_moe_simd_pipeline()?,
                batched_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_simd")?,
                batched_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_norm_simd")?,
                inline_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_simd")?,
                inline_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_norm_simd")?,
                inline_gate_up_vec4_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_vec4_simd")?,
                inline_gate_up_vec4_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_vec4_norm_simd",
                )?,
                inline_gate_up_nocache_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_nocache_simd")?,
                inline_gate_up_nocache_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_nocache_norm_simd",
                )?,
                cross_expert_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_cross_expert_simd")?,
                cross_expert_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_norm_simd",
                )?,
                cross_expert_tile8_gate_up_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_tile8_simd",
                )?,
                cross_expert_tile8_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_cross_expert_tile8_norm_simd",
                )?,
                llama_like_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_llama_like_simd")?,
                llama_like_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_llama_like_norm_simd")?,
                decode_direct_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_decode_direct_simd")?,
                decode_direct_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_decode_direct_norm_simd",
                )?,
                two_row_dataflow_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_dataflow_simd",
                )?),
                two_row_dataflow_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_dataflow_norm_simd",
                )?),
                q4k_mature_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_simd")?,
                q4k_mature_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_norm_simd")?,
                fused_block_qx_gate_up_pipeline: Some(
                    self.dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_fused_block_qx_simd",
                    )?,
                ),
                fused_block_qx_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_fused_block_qx_norm_simd",
                )?),
                fused_block_llama_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_fused_block_llama_simd",
                )?),
                fused_block_llama_gate_up_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_fused_block_llama_norm_simd",
                )?),
                inline_gate_up_rows2_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_rows2_simd")?,
                ),
                inline_gate_up_rows2_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_rows2_norm_simd",
                )?),
                inline_gate_up_rows4_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_rows4_simd")?,
                ),
                inline_gate_up_rows4_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_rows4_norm_simd",
                )?),
                inline_gate_up_rows8_pipeline: Some(
                    self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_inline_rows8_simd")?,
                ),
                inline_gate_up_rows8_norm_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_rows8_norm_simd",
                )?),
                wide_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_wide_simd")?,
                wide_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_wide_norm_simd")?,
                wide_supported: true,
                expertwise: false,
                expertwise_q4_gate_up_pipeline: None,
                expertwise_q4_gate_up_norm_pipeline: None,
                expertwise_q3_gate_up_pipeline: None,
                expertwise_q3_gate_up_norm_pipeline: None,
                expertwise_q4_down_pipeline: None,
                expertwise_q3_down_pipeline: None,
                expertwise_q4_gate_up_packed: None,
                expertwise_q4_gate_up_scales: None,
                expertwise_q3_gate_up_packed: None,
                expertwise_q3_gate_up_scales: None,
                expertwise_gate_up_map: None,
                expertwise_q4_down_packed: None,
                expertwise_q4_down_scales: None,
                expertwise_q3_down_packed: None,
                expertwise_q3_down_scales: None,
                expertwise_down_map: None,
                expertwise_q3_gate_up_block_bytes: None,
                expertwise_q3_down_block_bytes: None,
                q3_gate_up_pipeline: None,
                q3_gate_up_norm_pipeline: None,
                q3_gate_up_packed: None,
                q3_gate_up_scales: None,
                q3_gate_up_block_bytes: None,
                q3_down_pipeline: None,
                q3_down_packed: None,
                q3_down_scales: None,
                q3_down_block_bytes: None,
                batched_gate_up_touch_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch")?,
                batched_gate_up_touch_full_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_full")?,
                batched_gate_up_touch_direct_1_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_1")?,
                batched_gate_up_touch_direct_2_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_2")?,
                batched_gate_up_touch_direct_4_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_4")?,
                batched_gate_up_touch_direct_8_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_touch_direct_8")?,
                batched_gate_up_synthetic_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_synthetic")?,
                batched_gate_up_synthetic_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_synthetic_norm")?,
                batched_projection_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_projection_simd")?,
                batched_projection_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_projection_norm_simd")?,
                batched_silu_pipeline: self.dynamic_moe_named_pipeline("qx_dynamic_q4_moe_silu")?,
                batched_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_simd")?,
                inline_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_inline_simd")?,
                decode_direct_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_decode_direct_simd")?,
                q4k_mature_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_down_mature_simd")?,
                cross_expert_tile8_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cross_expert_tile8_simd")?,
                inline_layout: gate_up_layout == PackingLayout::Q4RowInline,
                down_inline_layout: down_layout.is_q4_row_inline(),
                wide_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_wide_simd")?,
                batched_accumulate_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_accumulate")?,
                gate_up_packed: self.input_runtime_buffer(gate_up_packed)?,
                gate_up_scales: self.input_runtime_buffer(gate_up_scales)?,
                split_gate_packed: Some(self.input_buffer(split_gate_packed)?),
                split_gate_scales: Some(self.input_buffer(split_gate_scales)?),
                split_up_packed: Some(self.input_buffer(split_up_packed)?),
                split_up_scales: Some(self.input_buffer(split_up_scales)?),
                row_stream_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_row_stream_simd")?,
                row_stream_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_gate_up_row_stream_norm_simd")?,
                split_gate_up: true,
                down_packed: self.input_runtime_buffer(down_packed)?,
                down_scales: self.input_runtime_buffer(down_scales)?,
                layout: gate_up_layout,
                gate_up_layout,
                down_layout,
            })
        }

        pub fn prepare_q4_embedding(
            &self,
            weights: &qx_quant::QuantizedTensor,
        ) -> Result<PreparedQ4Embedding, String> {
            if weights.quant_type != QuantType::Q4
                || weights.layout != PackingLayout::RowMajor
                || weights.group_size != 32
            {
                return Err("V2 Q4 embedding requires row-major Q4 group-32 weights".to_owned());
            }
            let packed = self.input_runtime_buffer(&weights.data)?;
            let mut scale_bytes = Vec::with_capacity(weights.scales.len() * 2);
            for scale in &weights.scales {
                scale_bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
            }
            let scales = self.input_runtime_buffer(&scale_bytes)?;
            Ok(PreparedQ4Embedding {
                pipeline: self.q4_embedding_pipeline()?,
                packed,
                scales,
                rows: weights.rows,
                columns: weights.cols,
            })
        }

        pub fn prepare_q4_embedding_from_dense(
            &self,
            weights: &PreparedQuantizedDense,
        ) -> Result<PreparedQ4Embedding, String> {
            Ok(PreparedQ4Embedding {
                pipeline: self.q4_embedding_pipeline()?,
                packed: weights.packed.clone(),
                scales: weights.scales.clone(),
                rows: weights.rows,
                columns: weights.columns,
            })
        }

        pub fn run_prepared_dense(
            &self,
            prepared: &PreparedDense,
            input: &[f32],
        ) -> Result<Vec<f32>, String> {
            let outputs = self.run_prepared_dense_many(std::slice::from_ref(&prepared), input)?;
            Ok(outputs.into_iter().next().expect("one dense output"))
        }

        pub fn run_prepared_dense_many(
            &self,
            prepared: &[&PreparedDense],
            input: &[f32],
        ) -> Result<Vec<Vec<f32>>, String> {
            if prepared.is_empty() {
                return Err("at least one prepared dense matrix is required".to_owned());
            }
            if prepared.iter().any(|matrix| matrix.columns != input.len()) {
                return Err("dense input width does not match prepared matrix".to_owned());
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let mut scratch = self
                .scratch
                .lock()
                .map_err(|_| "Metal scratch cache is poisoned".to_owned())?;
            if scratch
                .input_f32
                .as_ref()
                .is_none_or(|buffer| buffer.length() < input_bytes.len())
            {
                scratch.input_f32 = Some(
                    self.device
                        .newBufferWithLength_options(
                            input_bytes.len(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal dense input buffer allocation failed".to_owned())?,
                );
            }
            while scratch.dense_outputs.len() < prepared.len() {
                let rows = prepared[scratch.dense_outputs.len()].rows;
                scratch.dense_outputs.push(
                    self.device
                        .newBufferWithLength_options(
                            rows * std::mem::size_of::<f32>(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal dense output buffer allocation failed".to_owned())?,
                );
            }
            for (slot, matrix) in scratch.dense_outputs.iter_mut().zip(prepared) {
                let required = matrix.rows * std::mem::size_of::<f32>();
                if slot.length() < required {
                    *slot = self
                        .device
                        .newBufferWithLength_options(
                            required,
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal dense output buffer allocation failed".to_owned())?;
                }
            }
            let input_buffer = scratch.input_f32.as_ref().expect("dense input buffer");
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input_bytes.as_ptr(),
                    input_buffer.contents().as_ptr() as *mut u8,
                    input_bytes.len(),
                );
            }
            let outputs = &scratch.dense_outputs[..prepared.len()];
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal dense command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal dense encoder allocation failed".to_owned())?;
            for (matrix, output) in prepared.iter().zip(outputs) {
                encoder.setComputePipelineState(&matrix.pipeline);
                let columns = matrix.columns as u32;
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&matrix.weights), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(input_buffer), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
                    encoder.setBytes_length_atIndex(
                        NonNull::from(&columns).cast(),
                        std::mem::size_of_val(&columns),
                        3,
                    );
                }
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: matrix.rows,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 256,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal dense command buffer execution failed".to_owned());
            }
            self.record_command(
                &command_buffer,
                prepared.len() as u64,
                encode_start,
                wait_start,
            )?;
            Ok(outputs
                .iter()
                .map(|output| {
                    let contents = output.contents().as_ptr() as *const f32;
                    unsafe { std::slice::from_raw_parts(contents, output.length() / 4).to_vec() }
                })
                .collect())
        }

        pub fn prepare_logits(&self, embeddings: &[f32]) -> Result<PreparedLogits, String> {
            if embeddings.is_empty() || !embeddings.len().is_multiple_of(HIDDEN) {
                return Err(format!(
                    "embedding length {} is not a non-empty multiple of {HIDDEN}",
                    embeddings.len()
                ));
            }
            let embedding_bytes = unsafe {
                std::slice::from_raw_parts(
                    embeddings.as_ptr() as *const u8,
                    std::mem::size_of_val(embeddings),
                )
            };
            Ok(PreparedLogits {
                pipeline: self.logits_pipeline()?,
                embeddings: self.input_buffer(embedding_bytes)?,
                rows: embeddings.len() / HIDDEN,
            })
        }

        pub fn run_prepared_logits(
            &self,
            prepared: &PreparedLogits,
            hidden: &[f32; HIDDEN],
            scaling: f32,
        ) -> Result<Vec<f32>, String> {
            let hidden_bytes = unsafe {
                std::slice::from_raw_parts(
                    hidden.as_ptr() as *const u8,
                    std::mem::size_of_val(hidden),
                )
            };
            let mut scratch = self
                .scratch
                .lock()
                .map_err(|_| "Metal scratch cache is poisoned".to_owned())?;
            if scratch
                .input_f32
                .as_ref()
                .is_none_or(|buffer| buffer.length() < hidden_bytes.len())
            {
                scratch.input_f32 = Some(
                    self.device
                        .newBufferWithLength_options(
                            hidden_bytes.len(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal logits input allocation failed".to_owned())?,
                );
            }
            if scratch
                .logits_output
                .as_ref()
                .is_none_or(|buffer| buffer.length() < prepared.rows * 4)
            {
                scratch.logits_output = Some(
                    self.device
                        .newBufferWithLength_options(
                            prepared.rows * std::mem::size_of::<f32>(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal logits output allocation failed".to_owned())?,
                );
            }
            let output = scratch
                .logits_output
                .as_ref()
                .expect("logits output buffer");
            let hidden_buffer = scratch.input_f32.as_ref().expect("logits input buffer");
            unsafe {
                std::ptr::copy_nonoverlapping(
                    hidden_bytes.as_ptr(),
                    hidden_buffer.contents().as_ptr() as *mut u8,
                    hidden_bytes.len(),
                );
            }
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal logits command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal logits encoder allocation failed".to_owned())?;
            encoder.setComputePipelineState(&prepared.pipeline);
            let rows = prepared.rows as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&prepared.embeddings), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(hidden_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(output), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&rows).cast(),
                    std::mem::size_of_val(&rows),
                    3,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::from(&scaling).cast(),
                    std::mem::size_of_val(&scaling),
                    4,
                );
            }
            let threads = 256;
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal logits command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let contents = output.contents().as_ptr() as *const f32;
            Ok(unsafe { std::slice::from_raw_parts(contents, prepared.rows) }.to_vec())
        }

        fn validate_fused_weights(
            quant_type: QuantType,
            gate_up_packed: &[u8],
            gate_up_scales: &[u8],
            down_packed: &[u8],
            down_scales: &[u8],
        ) -> Result<(), String> {
            let expected_gate_up = buffer_len(HIDDEN, HIDDEN, quant_type);
            let expected_down = buffer_len(HIDDEN, INTERMEDIATE, quant_type);
            let expected_gate_up_scales = scale_len(HIDDEN, HIDDEN);
            let expected_down_scales = scale_len(HIDDEN, INTERMEDIATE);
            if gate_up_packed.len() != expected_gate_up {
                return Err(format!(
                    "gate/up packed length {}, expected {expected_gate_up}",
                    gate_up_packed.len()
                ));
            }
            if down_packed.len() != expected_down {
                return Err(format!(
                    "down packed length {}, expected {expected_down}",
                    down_packed.len()
                ));
            }
            if gate_up_scales.len() != expected_gate_up_scales {
                return Err(format!(
                    "gate/up scale byte length {}, expected {expected_gate_up_scales}",
                    gate_up_scales.len()
                ));
            }
            if down_scales.len() != expected_down_scales {
                return Err(format!(
                    "down scale byte length {}, expected {expected_down_scales}",
                    down_scales.len()
                ));
            }
            Ok(())
        }

        pub fn prepare_q4_stage(
            &self,
            weights: &qx_quant::QuantizedTensor,
        ) -> Result<PreparedQ4Stage, String> {
            if weights.quant_type != QuantType::Q4 || weights.group_size != GROUP_SIZE {
                return Err("Q4 stage benchmark requires group-32 Q4 weights".to_owned());
            }
            let scale_bytes = weights
                .scales
                .iter()
                .flat_map(|scale| scale.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            Ok(PreparedQ4Stage {
                pipeline: self.q4_gemv_pipeline()?,
                packed: self.input_runtime_buffer(&weights.data)?,
                scales: self.input_runtime_buffer(&scale_bytes)?,
                rows: weights.rows,
                columns: weights.cols,
                layout: weights.layout,
            })
        }

        pub fn prepare_q4_simd_stage(
            &self,
            weights: &qx_quant::QuantizedTensor,
        ) -> Result<PreparedQ4SimdStage, String> {
            if weights.quant_type != QuantType::Q4
                || weights.group_size != GROUP_SIZE
                || !weights.rows.is_multiple_of(8)
            {
                return Err(
                    "SIMD Q4 stage requires group-32 Q4 weights with rows divisible by 8"
                        .to_owned(),
                );
            }
            let scale_bytes = weights
                .scales
                .iter()
                .flat_map(|scale| scale.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            Ok(PreparedQ4SimdStage {
                pipeline: self.q4_simd_gemv_pipeline()?,
                packed: self.input_buffer(&weights.data)?,
                scales: self.input_buffer(&scale_bytes)?,
                rows: weights.rows,
                columns: weights.cols,
                layout: weights.layout,
            })
        }

        pub fn run_prepared_q4_stage(
            &self,
            prepared: &PreparedQ4Stage,
            input: &[u16],
        ) -> Result<Vec<u16>, String> {
            if input.len() != prepared.columns {
                return Err(format!(
                    "Q4 stage input length {}, expected {}",
                    input.len(),
                    prepared.columns
                ));
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let input_buffer = self.input_buffer(input_bytes)?;
            let output_buffer = self
                .device
                .newBufferWithLength_options(
                    prepared.rows * std::mem::size_of::<u16>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal Q4 stage output allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal Q4 stage command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal Q4 stage encoder allocation failed".to_owned())?;
            encoder.setComputePipelineState(&prepared.pipeline);
            let columns = prepared.columns as u32;
            let layout = prepared.layout as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(&prepared.packed.buffer),
                    prepared.packed.offset,
                    0,
                );
                encoder.setBuffer_offset_atIndex(
                    Some(&prepared.scales.buffer),
                    prepared.scales.offset,
                    1,
                );
                encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 3);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&columns).cast(),
                    std::mem::size_of_val(&columns),
                    4,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::from(&layout).cast(),
                    std::mem::size_of_val(&layout),
                    5,
                );
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal Q4 stage command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let contents = output_buffer.contents().as_ptr() as *const u16;
            Ok(unsafe { std::slice::from_raw_parts(contents, prepared.rows) }.to_vec())
        }

        pub fn run_prepared_q4_simd_stage(
            &self,
            prepared: &PreparedQ4SimdStage,
            input: &[u16],
        ) -> Result<Vec<u16>, String> {
            if input.len() != prepared.columns {
                return Err(format!(
                    "SIMD Q4 stage input length {}, expected {}",
                    input.len(),
                    prepared.columns
                ));
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let input_buffer = self.input_buffer(input_bytes)?;
            let output_buffer = self
                .device
                .newBufferWithLength_options(
                    prepared.rows * std::mem::size_of::<u16>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal SIMD Q4 stage output allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal SIMD Q4 stage command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal SIMD Q4 stage encoder allocation failed".to_owned())?;
            encoder.setComputePipelineState(&prepared.pipeline);
            let columns = prepared.columns as u32;
            let layout = prepared.layout as u32;
            let rows = prepared.rows as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&prepared.packed), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&prepared.scales), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 3);
                for (value, index) in [(&columns, 4), (&layout, 5), (&rows, 6)] {
                    encoder.setBytes_length_atIndex(
                        NonNull::from(value).cast(),
                        std::mem::size_of_val(value),
                        index,
                    );
                }
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows / 8,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal SIMD Q4 stage command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let contents = output_buffer.contents().as_ptr() as *const u16;
            Ok(unsafe { std::slice::from_raw_parts(contents, prepared.rows) }.to_vec())
        }

        /// Prepare one fixed-shape Granite V0 expert without materializing
        /// weights. The packed buffers use the qx row-major format.
        pub fn prepare_fused_expert_bytes(
            &self,
            quant_type: QuantType,
            gate_up_packed: &[u8],
            gate_up_scales: &[u8],
            down_packed: &[u8],
            down_scales: &[u8],
        ) -> Result<PreparedFusedExpert, String> {
            self.prepare_fused_expert_bytes_with_layout(
                quant_type,
                gate_up_packed,
                gate_up_scales,
                down_packed,
                down_scales,
                PackingLayout::RowMajor,
            )
        }

        pub fn prepare_fused_expert_bytes_with_layout(
            &self,
            quant_type: QuantType,
            gate_up_packed: &[u8],
            gate_up_scales: &[u8],
            down_packed: &[u8],
            down_scales: &[u8],
            layout: PackingLayout,
        ) -> Result<PreparedFusedExpert, String> {
            Self::validate_fused_weights(
                quant_type,
                gate_up_packed,
                gate_up_scales,
                down_packed,
                down_scales,
            )?;
            Ok(PreparedFusedExpert {
                pipeline: self.fused_pipeline(quant_type)?,
                aggregate_pipeline: self.aggregate_pipeline(quant_type)?,
                layout,
                gate_up_packed: self.input_runtime_buffer(gate_up_packed)?,
                gate_up_scales: self.input_runtime_buffer(gate_up_scales)?,
                down_packed: self.input_runtime_buffer(down_packed)?,
                down_scales: self.input_runtime_buffer(down_scales)?,
            })
        }

        /// Dispatch one prepared expert. Only the hidden input and output are
        /// transient; packed weights and the compiled pipeline are reused.
        pub fn run_prepared_expert(
            &self,
            prepared: &PreparedFusedExpert,
            input: &[u16],
        ) -> Result<Vec<u16>, String> {
            let outputs = self.run_prepared_experts(std::slice::from_ref(&prepared), input)?;
            Ok(outputs.into_iter().next().expect("one expert output"))
        }

        /// Dispatch several prepared experts from the same input in one command
        /// buffer. The kernels are independent, but batching their submissions
        /// avoids a GPU/CPU synchronization for every routed expert.
        pub fn run_prepared_experts(
            &self,
            prepared: &[&PreparedFusedExpert],
            input: &[u16],
        ) -> Result<Vec<Vec<u16>>, String> {
            if input.len() != HIDDEN {
                return Err(format!("input length {}, expected {HIDDEN}", input.len()));
            }
            if prepared.is_empty() {
                return Err("at least one prepared expert is required".to_owned());
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let mut scratch = self
                .scratch
                .lock()
                .map_err(|_| "Metal scratch cache is poisoned".to_owned())?;
            if scratch
                .input_u16
                .as_ref()
                .is_none_or(|buffer| buffer.length() < input_bytes.len())
            {
                scratch.input_u16 = Some(
                    self.device
                        .newBufferWithLength_options(
                            input_bytes.len(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal expert input allocation failed".to_owned())?,
                );
            }
            while scratch.expert_outputs.len() < prepared.len() {
                scratch.expert_outputs.push(
                    self.device
                        .newBufferWithLength_options(
                            HIDDEN * std::mem::size_of::<u16>(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal output buffer allocation failed".to_owned())?,
                );
            }
            let input_buffer = scratch.input_u16.as_ref().expect("expert input buffer");
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input_bytes.as_ptr(),
                    input_buffer.contents().as_ptr() as *mut u8,
                    input_bytes.len(),
                );
            }
            let outputs = &scratch.expert_outputs[..prepared.len()];

            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal compute encoder allocation failed".to_owned())?;
            for (prepared, output) in prepared.iter().zip(outputs) {
                encoder.setComputePipelineState(&prepared.pipeline);
                unsafe {
                    encoder.setBuffer_offset_atIndex(
                        Some(&prepared.gate_up_packed.buffer),
                        prepared.gate_up_packed.offset,
                        0,
                    );
                    encoder.setBuffer_offset_atIndex(
                        Some(&prepared.gate_up_scales.buffer),
                        prepared.gate_up_scales.offset,
                        1,
                    );
                    encoder.setBuffer_offset_atIndex(
                        Some(&prepared.down_packed.buffer),
                        prepared.down_packed.offset,
                        2,
                    );
                    encoder.setBuffer_offset_atIndex(
                        Some(&prepared.down_scales.buffer),
                        prepared.down_scales.offset,
                        3,
                    );
                    encoder.setBuffer_offset_atIndex(Some(input_buffer), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(output), 0, 5);
                    let layout = prepared.layout as u32;
                    encoder.setBytes_length_atIndex(
                        NonNull::from(&layout).cast(),
                        std::mem::size_of_val(&layout),
                        6,
                    );
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: HIDDEN,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal command buffer execution failed".to_owned());
            }
            self.record_command(
                &command_buffer,
                prepared.len() as u64,
                encode_start,
                wait_start,
            )?;

            Ok(outputs
                .iter()
                .map(|output| {
                    let contents = output.contents().as_ptr() as *const u16;
                    unsafe { std::slice::from_raw_parts(contents, HIDDEN).to_vec() }
                })
                .collect())
        }

        /// Execute the fixed top-8 expert set and accumulate router-weighted
        /// outputs on the GPU before a single readback.
        pub fn run_prepared_experts_weighted(
            &self,
            prepared: &[&PreparedFusedExpert],
            input: &[u16],
            weights: &[f32; 8],
        ) -> Result<Vec<u16>, String> {
            if prepared.len() != 8 {
                return Err(format!(
                    "expected 8 prepared experts, got {}",
                    prepared.len()
                ));
            }
            if input.len() != HIDDEN {
                return Err(format!("input length {}, expected {HIDDEN}", input.len()));
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let mut scratch = self
                .scratch
                .lock()
                .map_err(|_| "Metal scratch cache is poisoned".to_owned())?;
            if scratch
                .input_u16
                .as_ref()
                .is_none_or(|buffer| buffer.length() < input_bytes.len())
            {
                scratch.input_u16 = Some(
                    self.device
                        .newBufferWithLength_options(
                            input_bytes.len(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal expert input allocation failed".to_owned())?,
                );
            }
            while scratch.expert_outputs.len() < 8 {
                scratch.expert_outputs.push(
                    self.device
                        .newBufferWithLength_options(
                            HIDDEN * std::mem::size_of::<u16>(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal output buffer allocation failed".to_owned())?,
                );
            }
            if scratch.moe_output.is_none() {
                scratch.moe_output = Some(
                    self.device
                        .newBufferWithLength_options(
                            HIDDEN * std::mem::size_of::<u16>(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal MoE output allocation failed".to_owned())?,
                );
            }
            let input_buffer = scratch.input_u16.as_ref().expect("expert input buffer");
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input_bytes.as_ptr(),
                    input_buffer.contents().as_ptr() as *mut u8,
                    input_bytes.len(),
                );
            }
            let outputs = &scratch.expert_outputs[..8];
            let moe_output = scratch.moe_output.as_ref().expect("MoE output buffer");
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal compute encoder allocation failed".to_owned())?;
            for (prepared, output) in prepared.iter().zip(outputs) {
                encoder.setComputePipelineState(&prepared.pipeline);
                unsafe {
                    encoder.setBuffer_offset_atIndex(
                        Some(&prepared.gate_up_packed.buffer),
                        prepared.gate_up_packed.offset,
                        0,
                    );
                    encoder.setBuffer_offset_atIndex(
                        Some(&prepared.gate_up_scales.buffer),
                        prepared.gate_up_scales.offset,
                        1,
                    );
                    encoder.setBuffer_offset_atIndex(
                        Some(&prepared.down_packed.buffer),
                        prepared.down_packed.offset,
                        2,
                    );
                    encoder.setBuffer_offset_atIndex(
                        Some(&prepared.down_scales.buffer),
                        prepared.down_scales.offset,
                        3,
                    );
                    encoder.setBuffer_offset_atIndex(Some(input_buffer), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(output), 0, 5);
                    let layout = prepared.layout as u32;
                    encoder.setBytes_length_atIndex(
                        NonNull::from(&layout).cast(),
                        std::mem::size_of_val(&layout),
                        6,
                    );
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: HIDDEN,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.setComputePipelineState(&prepared[0].aggregate_pipeline);
            unsafe {
                for (index, output) in outputs.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(output), 0, index);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::from(weights).cast(),
                    std::mem::size_of_val(weights),
                    8,
                );
                encoder.setBuffer_offset_atIndex(Some(moe_output), 0, 9);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: HIDDEN,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            let encode_start = Instant::now();
            command_buffer.commit();
            let wait_start = Instant::now();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal weighted MoE command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 9, encode_start, wait_start)?;
            let contents = moe_output.contents().as_ptr() as *const u16;
            Ok(unsafe { std::slice::from_raw_parts(contents, HIDDEN).to_vec() })
        }

        /// Dispatch one fixed-shape Granite V0 expert without materializing weights.
        /// The returned vector contains 1024 FP16 bit patterns.
        pub fn run_fused_expert(
            &self,
            quant_type: QuantType,
            gate_up_packed: &[u8],
            gate_up_scales: &[u16],
            down_packed: &[u8],
            down_scales: &[u16],
            input: &[u16],
        ) -> Result<Vec<u16>, String> {
            let gate_up_scales_bytes = unsafe {
                std::slice::from_raw_parts(
                    gate_up_scales.as_ptr() as *const u8,
                    std::mem::size_of_val(gate_up_scales),
                )
            };
            let down_scales_bytes = unsafe {
                std::slice::from_raw_parts(
                    down_scales.as_ptr() as *const u8,
                    std::mem::size_of_val(down_scales),
                )
            };
            self.run_fused_expert_bytes(
                quant_type,
                gate_up_packed,
                gate_up_scales_bytes,
                down_packed,
                down_scales_bytes,
                input,
            )
        }

        /// Dispatch a fused expert when FP16 scales are already stored as bytes.
        /// This is the zero-copy scale interface used by memory-mapped `.qx`
        /// artifacts; Metal receives the mapped scale bytes directly.
        pub fn run_fused_expert_bytes(
            &self,
            quant_type: QuantType,
            gate_up_packed: &[u8],
            gate_up_scales_bytes: &[u8],
            down_packed: &[u8],
            down_scales_bytes: &[u8],
            input: &[u16],
        ) -> Result<Vec<u16>, String> {
            let prepared = self.prepare_fused_expert_bytes(
                quant_type,
                gate_up_packed,
                gate_up_scales_bytes,
                down_packed,
                down_scales_bytes,
            )?;
            self.run_prepared_expert(&prepared, input)
        }
    }

    fn f16_bits_to_f32(bits: u16) -> f32 {
        let sign = u32::from(bits & 0x8000) << 16;
        let exponent = (bits >> 10) & 0x1f;
        let fraction = u32::from(bits & 0x03ff);
        let value = match exponent {
            0 if fraction == 0 => sign,
            0 => {
                let mut fraction = fraction;
                let mut exponent = -14_i32;
                while fraction & 0x0400 == 0 {
                    fraction <<= 1;
                    exponent -= 1;
                }
                let fraction = fraction & 0x03ff;
                sign | (u32::try_from(exponent + 127).unwrap_or(0) << 23) | (fraction << 13)
            }
            0x1f if fraction == 0 => sign | 0x7f80_0000,
            0x1f => sign | 0x7f80_0000 | (fraction << 13),
            exponent => sign | (u32::from(exponent) + 112) << 23 | (fraction << 13),
        };
        f32::from_bits(value)
    }
}

#[cfg(not(target_os = "macos"))]
pub mod platform {
    pub struct MetalContext;

    impl MetalContext {
        pub fn system_default() -> Option<Self> {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_quant_type_has_a_kernel() {
        for quant_type in [QuantType::Q2, QuantType::Q3, QuantType::Q4] {
            assert!(kernel_source(quant_type).contains("kernel"));
        }
    }

    #[test]
    fn fused_transaction_block_factorial_kernels_are_present() {
        for name in [
            "qx_dynamic_q4_moe_gate_up_fused_block_qx_simd",
            "qx_dynamic_q4_moe_gate_up_fused_block_qx_norm_simd",
            "qx_dynamic_q4_moe_gate_up_fused_block_llama_simd",
            "qx_dynamic_q4_moe_gate_up_fused_block_llama_norm_simd",
        ] {
            assert!(
                DYNAMIC_MOE_KERNEL_SOURCE.contains(name),
                "missing Metal entry point: {name}"
            );
        }
        assert!(DYNAMIC_MOE_KERNEL_SOURCE.contains("layout == 7 || layout == 8"));
        assert!(DYNAMIC_MOE_KERNEL_SOURCE.contains("layout == 8 ? 160 : 144"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_expert_dispatch_matches_uniform_synthetic_fixture() {
        use half::f16;

        fn packed(rows: usize, cols: usize, quant_type: QuantType, code: u8) -> Vec<u8> {
            let bits = quant_type.bits();
            let bytes_per_row = cols * bits / 8;
            let mut output = vec![0_u8; rows * bytes_per_row];
            for row in 0..rows {
                for column in 0..cols {
                    let bit = column * bits;
                    let byte = row * bytes_per_row + bit / 8;
                    let shift = bit % 8;
                    output[byte] |= code << shift;
                    if shift + bits > 8 {
                        output[byte + 1] |= code >> (8 - shift);
                    }
                }
            }
            output
        }

        let context = platform::MetalContext::system_default().expect("Metal device");
        let input = vec![f16::from_f32(1.0 / 1024.0).to_bits(); 1024];
        for quant_type in [QuantType::Q2, QuantType::Q3, QuantType::Q4] {
            let qmax = (1_u8 << (quant_type.bits() - 1)) - 1;
            let gate_up_packed = packed(1024, 1024, quant_type, qmax + 1);
            let down_packed = packed(1024, 512, quant_type, qmax + 1);
            let gate_up_scales = vec![f16::from_f32(1.0).to_bits(); 1024 * 32];
            let down_scales = vec![f16::from_f32(1.0).to_bits(); 1024 * 16];
            let output = context
                .run_fused_expert(
                    quant_type,
                    &gate_up_packed,
                    &gate_up_scales,
                    &down_packed,
                    &down_scales,
                    &input,
                )
                .expect("fused Metal dispatch");
            let expected = f16::from_f32(374.3019).to_f32();
            for value in output {
                assert!((f16::from_bits(value).to_f32() - expected).abs() < 0.5);
            }
        }

        let quant_type = QuantType::Q4;
        let qmax = (1_u8 << (quant_type.bits() - 1)) - 1;
        let gate_up_packed = packed(1024, 1024, quant_type, qmax + 1);
        let down_packed = packed(1024, 512, quant_type, qmax + 1);
        let gate_up_scales = vec![f16::from_f32(1.0).to_bits(); 1024 * 32];
        let down_scales = vec![f16::from_f32(1.0).to_bits(); 1024 * 16];
        let gate_up_scale_bytes = unsafe {
            std::slice::from_raw_parts(
                gate_up_scales.as_ptr() as *const u8,
                std::mem::size_of_val(gate_up_scales.as_slice()),
            )
        };
        let down_scale_bytes = unsafe {
            std::slice::from_raw_parts(
                down_scales.as_ptr() as *const u8,
                std::mem::size_of_val(down_scales.as_slice()),
            )
        };
        let prepared = (0..8)
            .map(|_| {
                context
                    .prepare_fused_expert_bytes(
                        quant_type,
                        &gate_up_packed,
                        gate_up_scale_bytes,
                        &down_packed,
                        down_scale_bytes,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let selected = prepared.iter().collect::<Vec<_>>();
        let weighted = context
            .run_prepared_experts_weighted(&selected, &input, &[0.125; 8])
            .expect("weighted MoE dispatch");
        let expected = f16::from_f32(374.3019).to_f32();
        assert!(
            weighted
                .iter()
                .all(|value| (f16::from_bits(*value).to_f32() - expected).abs() < 0.5)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn logits_dispatch_matches_tied_embedding_fixture() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let embeddings = [vec![1.0_f32; 1024], vec![2.0_f32; 1024]].concat();
        let prepared = context.prepare_logits(&embeddings).unwrap();
        let hidden = [1.0_f32; 1024];
        let logits = context
            .run_prepared_logits(&prepared, &hidden, 2.0)
            .unwrap();
        assert_eq!(logits, [512.0, 1024.0]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn v2_graph_encodes_argmax_with_one_final_wait() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let logits = context
            .allocate_runtime_buffer(4 * std::mem::size_of::<f32>())
            .unwrap();
        let token = context
            .allocate_runtime_buffer(std::mem::size_of::<u32>())
            .unwrap();
        let values = [1.0_f32, 9.0, 3.0, 8.0];
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(&values))
        };
        context.write_runtime_buffer(&logits, bytes).unwrap();

        let before = context.metrics().unwrap();
        let mut graph = context.begin_decode().unwrap();
        graph
            .encode_argmax(&context, &logits, &logits, &token, values.len())
            .unwrap();
        graph.finish(&context).unwrap();
        let after = context.metrics().unwrap();

        assert_eq!(context.read_runtime_u32(&token).unwrap(), 1);
        assert_eq!(after.command_buffers - before.command_buffers, 1);
        assert_eq!(after.dispatches - before.dispatches, 2);
        assert!(after.cpu_wait_seconds >= before.cpu_wait_seconds);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn v2_graph_can_submit_two_gpu_ordered_command_buffers_with_one_final_wait() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let before = context.metrics().unwrap();
        let mut graph = context.begin_decode().unwrap();
        graph
            .encode_noop(
                &context,
                platform::MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
                platform::MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            )
            .unwrap();
        graph.begin_next_command_buffer(&context).unwrap();
        graph
            .encode_noop(
                &context,
                platform::MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
                platform::MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            )
            .unwrap();
        graph.finish(&context).unwrap();
        let after = context.metrics().unwrap();
        assert_eq!(after.command_buffers - before.command_buffers, 2);
        assert_eq!(after.encoders - before.encoders, 2);
        assert_eq!(after.dispatches - before.dispatches, 2);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn v2_q4_dense_graph_matches_quantized_reference() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let weights = (0..32).map(|value| value as f32 / 32.0).collect::<Vec<_>>();
        let quantized = qx_quant::QuantizedTensor::quantize(&weights, 1, 32, QuantType::Q4, 32);
        let prepared = context.prepare_q4_dense(&quantized).unwrap();
        let input = context.allocate_runtime_buffer(32 * 4).unwrap();
        let output = context.allocate_runtime_buffer(4).unwrap();
        let input_values = vec![1.0_f32; 32];
        let input_bytes = unsafe {
            std::slice::from_raw_parts(
                input_values.as_ptr() as *const u8,
                std::mem::size_of_val(input_values.as_slice()),
            )
        };
        context.write_runtime_buffer(&input, input_bytes).unwrap();
        let mut graph = context.begin_decode().unwrap();
        graph
            .encode_prepared_q4_dense(&prepared, &input, &output)
            .unwrap();
        graph.finish(&context).unwrap();
        let expected = weights.iter().sum::<f32>();
        let observed = context.read_runtime_f32(&output, 1).unwrap()[0];
        assert!((observed - expected).abs() < 0.2);

        let simd_prepared = context.prepare_q4_dense_simd(&quantized).unwrap();
        let simd_observed = context
            .run_prepared_q4_dense_simd(&simd_prepared, &input_values)
            .unwrap()[0];
        assert!((simd_observed - expected).abs() < 0.2);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn v2_fused_qkv_matches_three_q4_dense_dispatches() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let input_values = vec![0.25_f32; 1024];
        let q_values = (0..1024 * 1024)
            .map(|index| (index % 31) as f32 / 31.0)
            .collect::<Vec<_>>();
        let kv_values = (0..512 * 1024)
            .map(|index| (index % 17) as f32 / 17.0)
            .collect::<Vec<_>>();
        let q = context
            .prepare_q4_dense(&qx_quant::QuantizedTensor::quantize(
                &q_values,
                1024,
                1024,
                QuantType::Q4,
                32,
            ))
            .unwrap();
        let k = context
            .prepare_q4_dense(&qx_quant::QuantizedTensor::quantize(
                &kv_values,
                512,
                1024,
                QuantType::Q4,
                32,
            ))
            .unwrap();
        let v = context
            .prepare_q4_dense(&qx_quant::QuantizedTensor::quantize(
                &kv_values,
                512,
                1024,
                QuantType::Q4,
                32,
            ))
            .unwrap();
        let qkv = context.prepare_q4_qkv(&q, &k, &v).unwrap();
        let input = context.allocate_runtime_buffer(1024 * 4).unwrap();
        let fused_q = context.allocate_runtime_buffer(1024 * 4).unwrap();
        let fused_k = context.allocate_runtime_buffer(512 * 4).unwrap();
        let fused_v = context.allocate_runtime_buffer(512 * 4).unwrap();
        let separate_q = context.allocate_runtime_buffer(1024 * 4).unwrap();
        let separate_k = context.allocate_runtime_buffer(512 * 4).unwrap();
        let separate_v = context.allocate_runtime_buffer(512 * 4).unwrap();
        let input_bytes = unsafe {
            std::slice::from_raw_parts(
                input_values.as_ptr() as *const u8,
                std::mem::size_of_val(input_values.as_slice()),
            )
        };
        context.write_runtime_buffer(&input, input_bytes).unwrap();
        let mut graph = context.begin_decode().unwrap();
        graph
            .encode_prepared_q4_qkv(&qkv, &input, &fused_q, &fused_k, &fused_v)
            .unwrap();
        graph
            .encode_prepared_q4_dense(&q, &input, &separate_q)
            .unwrap();
        graph
            .encode_prepared_q4_dense(&k, &input, &separate_k)
            .unwrap();
        graph
            .encode_prepared_q4_dense(&v, &input, &separate_v)
            .unwrap();
        graph.finish(&context).unwrap();
        for (fused, separate, count) in [
            (&fused_q, &separate_q, 1024),
            (&fused_k, &separate_k, 512),
            (&fused_v, &separate_v, 512),
        ] {
            let fused = context.read_runtime_f32(fused, count).unwrap();
            let separate = context.read_runtime_f32(separate, count).unwrap();
            let max_error = fused
                .iter()
                .zip(separate)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_error < 1.0e-4, "fused QKV error: {max_error}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn v2_dynamic_q4_moe_consumes_gpu_routing_buffers() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let gate_up_packed = vec![0_u8; 32 * 1024 * 1024 / 2];
        let gate_up_scales = vec![0_u8; 32 * 1024 * 32 * 2];
        let down_packed = vec![0_u8; 32 * 1024 * 512 / 2];
        let down_scales = vec![0_u8; 32 * 1024 * 16 * 2];
        let prepared = context
            .prepare_dynamic_q4_moe(
                &gate_up_packed,
                &gate_up_scales,
                &down_packed,
                &down_scales,
                PackingLayout::RowMajor,
            )
            .unwrap();
        let input = context.allocate_runtime_buffer(1024 * 4).unwrap();
        let ids = context.allocate_runtime_buffer(8 * 4).unwrap();
        let weights = context.allocate_runtime_buffer(8 * 4).unwrap();
        let output = context.allocate_runtime_buffer(1024 * 4).unwrap();
        let ids_values = [0_u32, 1, 2, 3, 4, 5, 6, 7];
        let weight_values = [0.125_f32; 8];
        let input_values = [0.0_f32; 1024];
        for (buffer, values) in [
            (&input, unsafe {
                std::slice::from_raw_parts(
                    input_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&input_values),
                )
            }),
            (&ids, unsafe {
                std::slice::from_raw_parts(
                    ids_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&ids_values),
                )
            }),
            (&weights, unsafe {
                std::slice::from_raw_parts(
                    weight_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&weight_values),
                )
            }),
        ] {
            context.write_runtime_buffer(buffer, values).unwrap();
        }
        let mut graph = context.begin_decode().unwrap();
        graph
            .encode_dynamic_q4_moe(&prepared, &input, &ids, &weights, &output)
            .unwrap();
        graph.finish(&context).unwrap();
        assert!(
            context
                .read_runtime_f32(&output, 1024)
                .unwrap()
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn row_inline_stream8_matches_canonical_gate_up_kernel() {
        const EXPERTS: usize = 32;
        const HIDDEN: usize = 1024;
        const GROUPS: usize = 32;
        const GROUP_BYTES: usize = 18;
        let context = platform::MetalContext::system_default().expect("Metal device");
        let mut gate_up = vec![0_u8; EXPERTS * HIDDEN * GROUPS * GROUP_BYTES];
        for expert in 0..EXPERTS {
            for row in 0..HIDDEN {
                for group in 0..GROUPS {
                    let offset = (expert * HIDDEN * GROUPS + row * GROUPS + group) * GROUP_BYTES;
                    let nibble = ((expert + row + group) & 0xf) as u8;
                    gate_up[offset..offset + 16].fill(nibble | (nibble << 4));
                    gate_up[offset + 16..offset + 18].copy_from_slice(&[0x00, 0x3c]);
                }
            }
        }
        let down = vec![0_u8; EXPERTS * HIDDEN * 16 * GROUP_BYTES];
        let prepared = context
            .prepare_dynamic_q4_moe_with_layouts(
                &gate_up,
                &gate_up,
                &down,
                &down,
                PackingLayout::Q4RowInline,
                PackingLayout::Q4RowInline,
            )
            .unwrap();
        let input = context.allocate_runtime_buffer(HIDDEN * 4).unwrap();
        let ids = context.allocate_runtime_buffer(8 * 4).unwrap();
        let weights = context.allocate_runtime_buffer(8 * 4).unwrap();
        let scratch = context
            .allocate_runtime_buffer(8 * HIDDEN * 4 + 8 * 512 * 4 + 8 * HIDDEN * 4)
            .unwrap();
        let output = context.allocate_runtime_buffer(HIDDEN * 4).unwrap();
        let input_values = [1.0_f32; HIDDEN];
        let ids_values = [31_u32, 7, 13, 2, 24, 5, 18, 0];
        let weights_values = [0.125_f32; 8];
        for (buffer, bytes) in [
            (&input, unsafe {
                std::slice::from_raw_parts(
                    input_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&input_values),
                )
            }),
            (&ids, unsafe {
                std::slice::from_raw_parts(
                    ids_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&ids_values),
                )
            }),
            (&weights, unsafe {
                std::slice::from_raw_parts(
                    weights_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&weights_values),
                )
            }),
        ] {
            context.write_runtime_buffer(buffer, bytes).unwrap();
        }

        let run = |streaming: bool| {
            if streaming {
                unsafe { std::env::set_var("QX_V2_MOE_ROW_INLINE_STREAM8", "1") };
            } else {
                unsafe { std::env::remove_var("QX_V2_MOE_ROW_INLINE_STREAM8") };
            }
            let mut graph = context.begin_decode().unwrap();
            graph
                .encode_dynamic_q4_moe_batched_stage(
                    &prepared, &input, &ids, &weights, &scratch, &output, 0,
                )
                .unwrap();
            graph.finish(&context).unwrap();
            context.read_runtime_f32(&scratch, 8 * HIDDEN).unwrap()
        };
        let canonical = run(false);
        let streaming = run(true);
        unsafe { std::env::remove_var("QX_V2_MOE_ROW_INLINE_STREAM8") };
        let max_error = canonical
            .iter()
            .zip(&streaming)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_error < 1.0e-5, "row-stream gate/up error: {max_error}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn decode_direct_matches_canonical_gate_up_and_down() {
        const EXPERTS: usize = 32;
        const HIDDEN: usize = 1024;
        const GATE_GROUPS: usize = 32;
        const DOWN_GROUPS: usize = 16;
        const GROUP_BYTES: usize = 18;
        const SCRATCH_FLOATS: usize = 8 * HIDDEN + 8 * 512 + 8 * HIDDEN;
        let context = platform::MetalContext::system_default().expect("Metal device");
        let mut gate_up = vec![0_u8; EXPERTS * HIDDEN * GATE_GROUPS * GROUP_BYTES];
        for expert in 0..EXPERTS {
            for row in 0..HIDDEN {
                for group in 0..GATE_GROUPS {
                    let offset =
                        (expert * HIDDEN * GATE_GROUPS + row * GATE_GROUPS + group) * GROUP_BYTES;
                    let nibble = ((expert * 3 + row + group * 5) & 0xf) as u8;
                    gate_up[offset..offset + 16].fill(nibble | (nibble << 4));
                    gate_up[offset + 16..offset + 18].copy_from_slice(&[0x00, 0x3c]);
                }
            }
        }
        let mut down = vec![0_u8; EXPERTS * HIDDEN * DOWN_GROUPS * GROUP_BYTES];
        for expert in 0..EXPERTS {
            for row in 0..HIDDEN {
                for group in 0..DOWN_GROUPS {
                    let offset =
                        (expert * HIDDEN * DOWN_GROUPS + row * DOWN_GROUPS + group) * GROUP_BYTES;
                    let nibble = ((expert + row * 3 + group * 7) & 0xf) as u8;
                    down[offset..offset + 16].fill(nibble | (nibble << 4));
                    down[offset + 16..offset + 18].copy_from_slice(&[0x00, 0x3c]);
                }
            }
        }
        let prepared = context
            .prepare_dynamic_q4_moe_with_layouts(
                &gate_up,
                &gate_up,
                &down,
                &down,
                PackingLayout::Q4RowInline,
                PackingLayout::Q4RowInline,
            )
            .unwrap();
        let input = context.allocate_runtime_buffer(HIDDEN * 4).unwrap();
        let ids = context.allocate_runtime_buffer(8 * 4).unwrap();
        let weights = context.allocate_runtime_buffer(8 * 4).unwrap();
        let input_values: Vec<f32> = (0..HIDDEN)
            .map(|index| (index as f32 % 19.0 - 9.0) * 0.125)
            .collect();
        let ids_values = [31_u32, 7, 13, 2, 24, 5, 18, 0];
        let weights_values = [0.125_f32; 8];
        for (buffer, bytes) in [
            (&input, unsafe {
                std::slice::from_raw_parts(
                    input_values.as_ptr() as *const u8,
                    std::mem::size_of_val(input_values.as_slice()),
                )
            }),
            (&ids, unsafe {
                std::slice::from_raw_parts(
                    ids_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&ids_values),
                )
            }),
            (&weights, unsafe {
                std::slice::from_raw_parts(
                    weights_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&weights_values),
                )
            }),
        ] {
            context.write_runtime_buffer(buffer, bytes).unwrap();
        }

        let run_stage = |stage: u8, direct: bool| {
            if direct {
                unsafe { std::env::set_var("QX_V2_MOE_DECODE_DIRECT", "1") };
            } else {
                unsafe { std::env::remove_var("QX_V2_MOE_DECODE_DIRECT") };
            }
            let scratch = context.allocate_runtime_buffer(SCRATCH_FLOATS * 4).unwrap();
            let output = context.allocate_runtime_buffer(HIDDEN * 4).unwrap();
            let mut graph = context.begin_decode().unwrap();
            graph
                .encode_dynamic_q4_moe_batched_stage(
                    &prepared, &input, &ids, &weights, &scratch, &output, stage,
                )
                .unwrap();
            graph.finish(&context).unwrap();
            context.read_runtime_f32(&scratch, SCRATCH_FLOATS).unwrap()
        };

        let canonical_gate_up = run_stage(0, false);
        let direct_gate_up = run_stage(0, true);
        let gate_up_error = canonical_gate_up
            .iter()
            .zip(&direct_gate_up)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            gate_up_error < 1.0e-5,
            "decode-direct gate/up error: {gate_up_error}"
        );

        let run_down = |direct: bool| {
            unsafe { std::env::remove_var("QX_V2_MOE_DECODE_DIRECT") };
            let scratch = context.allocate_runtime_buffer(SCRATCH_FLOATS * 4).unwrap();
            let output = context.allocate_runtime_buffer(HIDDEN * 4).unwrap();
            let mut graph = context.begin_decode().unwrap();
            graph
                .encode_dynamic_q4_moe_batched_stage(
                    &prepared, &input, &ids, &weights, &scratch, &output, 1,
                )
                .unwrap();
            if direct {
                unsafe { std::env::set_var("QX_V2_MOE_DECODE_DIRECT", "1") };
            }
            graph
                .encode_dynamic_q4_moe_batched_stage(
                    &prepared, &input, &ids, &weights, &scratch, &output, 2,
                )
                .unwrap();
            graph.finish(&context).unwrap();
            context.read_runtime_f32(&scratch, SCRATCH_FLOATS).unwrap()
        };
        let canonical_down = run_down(false);
        let direct_down = run_down(true);
        let down_offset = 8 * HIDDEN + 8 * 512;
        let down_error = canonical_down[down_offset..down_offset + 8 * HIDDEN]
            .iter()
            .zip(&direct_down[down_offset..down_offset + 8 * HIDDEN])
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            down_error < 1.0e-5,
            "decode-direct down error: {down_error}"
        );
        unsafe { std::env::remove_var("QX_V2_MOE_DECODE_DIRECT") };
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "quarantined: cross-expert tile8 fails live parity despite isolated-kernel checks"]
    fn cross_expert_tile8_matches_row_inline_gate_up_kernel() {
        const EXPERTS: usize = 32;
        const HIDDEN: usize = 1024;
        const GROUPS: usize = 32;
        const GROUP_BYTES: usize = 18;
        let context = platform::MetalContext::system_default().expect("Metal device");
        let mut row_inline = vec![0_u8; EXPERTS * HIDDEN * GROUPS * GROUP_BYTES];
        for expert in 0..EXPERTS {
            for row in 0..HIDDEN {
                for group in 0..GROUPS {
                    let offset = (expert * HIDDEN * GROUPS + row * GROUPS + group) * GROUP_BYTES;
                    let nibble = ((expert + row + group) & 0xf) as u8;
                    row_inline[offset..offset + 16].fill(nibble | (nibble << 4));
                    row_inline[offset + 16..offset + 18].copy_from_slice(&[0x00, 0x3c]);
                }
            }
        }
        let mut cross = vec![0_u8; row_inline.len()];
        for tile in 0..(HIDDEN / 8) {
            for group in 0..GROUPS {
                for row_in_tile in 0..8 {
                    for expert in 0..EXPERTS {
                        let source =
                            (expert * HIDDEN * GROUPS + (tile * 8 + row_in_tile) * GROUPS + group)
                                * GROUP_BYTES;
                        let destination = (((tile * GROUPS + group) * 8 + row_in_tile) * EXPERTS
                            + expert)
                            * GROUP_BYTES;
                        cross[destination..destination + GROUP_BYTES]
                            .copy_from_slice(&row_inline[source..source + GROUP_BYTES]);
                    }
                }
            }
        }
        const DOWN_GROUPS: usize = 16;
        let mut down = vec![0_u8; EXPERTS * HIDDEN * DOWN_GROUPS * GROUP_BYTES];
        for expert in 0..EXPERTS {
            for row in 0..HIDDEN {
                for group in 0..DOWN_GROUPS {
                    let offset =
                        (expert * HIDDEN * DOWN_GROUPS + row * DOWN_GROUPS + group) * GROUP_BYTES;
                    let nibble = ((expert + row + group * 3) & 0xf) as u8;
                    down[offset..offset + 16].fill(nibble | (nibble << 4));
                    down[offset + 16..offset + 18].copy_from_slice(&[0x00, 0x3c]);
                }
            }
        }
        let mut cross_down = vec![0_u8; down.len()];
        for tile in 0..(HIDDEN / 8) {
            for group in 0..DOWN_GROUPS {
                for row_in_tile in 0..8 {
                    for expert in 0..EXPERTS {
                        let source = (expert * HIDDEN * DOWN_GROUPS
                            + (tile * 8 + row_in_tile) * DOWN_GROUPS
                            + group)
                            * GROUP_BYTES;
                        let destination =
                            (((tile * DOWN_GROUPS + group) * 8 + row_in_tile) * EXPERTS + expert)
                                * GROUP_BYTES;
                        cross_down[destination..destination + GROUP_BYTES]
                            .copy_from_slice(&down[source..source + GROUP_BYTES]);
                    }
                }
            }
        }
        let row_prepared = context
            .prepare_dynamic_q4_moe_with_layouts(
                &row_inline,
                &row_inline,
                &down,
                &down,
                PackingLayout::Q4RowInline,
                PackingLayout::Q4RowInline,
            )
            .unwrap();
        let cross_prepared = context
            .prepare_dynamic_q4_moe_with_layouts(
                &cross,
                &cross,
                &cross_down,
                &cross_down,
                PackingLayout::Q4CrossExpertTile8,
                PackingLayout::Q4CrossExpertTile8,
            )
            .unwrap();
        let input = context.allocate_runtime_buffer(HIDDEN * 4).unwrap();
        let ids = context.allocate_runtime_buffer(8 * 4).unwrap();
        let weights = context.allocate_runtime_buffer(8 * 4).unwrap();
        let scratch = context
            .allocate_runtime_buffer(8 * HIDDEN * 4 + 8 * 512 * 4 + 8 * HIDDEN * 4)
            .unwrap();
        let output = context.allocate_runtime_buffer(HIDDEN * 4).unwrap();
        let input_values = [1.0_f32; HIDDEN];
        let ids_values = [31_u32, 7, 13, 2, 24, 5, 18, 0];
        let weights_values = [0.125_f32; 8];
        for (buffer, bytes) in [
            (&input, unsafe {
                std::slice::from_raw_parts(
                    input_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&input_values),
                )
            }),
            (&ids, unsafe {
                std::slice::from_raw_parts(
                    ids_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&ids_values),
                )
            }),
            (&weights, unsafe {
                std::slice::from_raw_parts(
                    weights_values.as_ptr() as *const u8,
                    std::mem::size_of_val(&weights_values),
                )
            }),
        ] {
            context.write_runtime_buffer(buffer, bytes).unwrap();
        }
        let run = |prepared: &platform::PreparedDynamicQ4Moe, cross_enabled: bool| {
            if cross_enabled {
                unsafe { std::env::set_var("QX_V2_MOE_CROSS_EXPERT_TILE8", "1") };
            } else {
                unsafe { std::env::remove_var("QX_V2_MOE_CROSS_EXPERT_TILE8") };
            }
            let mut graph = context.begin_decode().unwrap();
            graph
                .encode_dynamic_q4_moe_batched_stage(
                    prepared, &input, &ids, &weights, &scratch, &output, 0,
                )
                .unwrap();
            graph.finish(&context).unwrap();
            context.read_runtime_f32(&scratch, 8 * HIDDEN).unwrap()
        };
        let row_result = run(&row_prepared, false);
        let cross_result = run(&cross_prepared, true);
        unsafe { std::env::remove_var("QX_V2_MOE_CROSS_EXPERT_TILE8") };
        let max_error = row_result
            .iter()
            .zip(&cross_result)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error < 1.0e-5,
            "cross-expert tile8 gate/up error: {max_error}"
        );

        let activated = scratch.subregion(8 * HIDDEN * 4, 8 * 512 * 4).unwrap();
        let activated_values = [1.0_f32; 8 * 512];
        let activated_bytes = unsafe {
            std::slice::from_raw_parts(
                activated_values.as_ptr() as *const u8,
                std::mem::size_of_val(&activated_values),
            )
        };
        context
            .write_runtime_buffer(&activated, activated_bytes)
            .unwrap();
        let run_down = |prepared: &platform::PreparedDynamicQ4Moe, cross_enabled: bool| {
            if cross_enabled {
                unsafe { std::env::set_var("QX_V2_MOE_CROSS_EXPERT_TILE8", "1") };
            } else {
                unsafe { std::env::remove_var("QX_V2_MOE_CROSS_EXPERT_TILE8") };
            }
            let mut graph = context.begin_decode().unwrap();
            graph
                .encode_dynamic_q4_moe_batched_stage(
                    prepared, &input, &ids, &weights, &scratch, &output, 2,
                )
                .unwrap();
            graph.finish(&context).unwrap();
            context
                .read_runtime_f32(
                    &scratch
                        .subregion(8 * HIDDEN * 4 + 8 * 512 * 4, 8 * HIDDEN * 4)
                        .unwrap(),
                    8 * HIDDEN,
                )
                .unwrap()
        };
        let row_down = run_down(&row_prepared, false);
        let cross_down_result = run_down(&cross_prepared, true);
        unsafe { std::env::remove_var("QX_V2_MOE_CROSS_EXPERT_TILE8") };
        let down_error = row_down
            .iter()
            .zip(&cross_down_result)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            down_error < 1.0e-5,
            "cross-expert tile8 down error: {down_error}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn transaction_144_and_160_dispatches_have_identical_projection_results() {
        use half::f16;

        fn transaction_stream(
            experts: usize,
            rows: usize,
            groups: usize,
            block_bytes: usize,
        ) -> Vec<u8> {
            let blocks = groups / 8;
            let expert_bytes = rows * blocks * block_bytes;
            let mut stream = vec![0_u8; experts * expert_bytes];
            for expert in 0..experts {
                for row in 0..rows {
                    for block in 0..blocks {
                        let block_start =
                            expert * expert_bytes + (row * blocks + block) * block_bytes;
                        for group in 0..8 {
                            let group_start = block_start + group * 18;
                            stream[group_start..group_start + 16].fill(0x88);
                            stream[group_start + 16..group_start + 18]
                                .copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
                        }
                    }
                }
            }
            stream
        }

        let context = platform::MetalContext::system_default().expect("Metal device");
        let input_values = [1.0_f32; 1024];
        let ids_values = [0_u32, 1, 2, 3, 4, 5, 6, 7];
        let weights_values = [0.125_f32; 8];
        let run = |layout, block_bytes| {
            let gate_up = transaction_stream(32, 1024, 32, block_bytes);
            let down = transaction_stream(32, 1024, 16, block_bytes);
            let prepared = context
                .prepare_dynamic_q4_moe_with_layouts(
                    &gate_up, &gate_up, &down, &down, layout, layout,
                )
                .unwrap();
            let input = context.allocate_runtime_buffer(1024 * 4).unwrap();
            let ids = context.allocate_runtime_buffer(8 * 4).unwrap();
            let weights = context.allocate_runtime_buffer(8 * 4).unwrap();
            let output = context.allocate_runtime_buffer(1024 * 4).unwrap();
            for (buffer, values) in [
                (&input, unsafe {
                    std::slice::from_raw_parts(
                        input_values.as_ptr() as *const u8,
                        std::mem::size_of_val(&input_values),
                    )
                }),
                (&ids, unsafe {
                    std::slice::from_raw_parts(
                        ids_values.as_ptr() as *const u8,
                        std::mem::size_of_val(&ids_values),
                    )
                }),
                (&weights, unsafe {
                    std::slice::from_raw_parts(
                        weights_values.as_ptr() as *const u8,
                        std::mem::size_of_val(&weights_values),
                    )
                }),
            ] {
                context.write_runtime_buffer(buffer, values).unwrap();
            }
            let scratch = context
                .allocate_runtime_buffer(8 * 1024 * 4 + 8 * 512 * 4 + 8 * 1024 * 4)
                .unwrap();
            let mut stage_graph = context.begin_decode().unwrap();
            stage_graph
                .encode_dynamic_q4_moe_batched_stage(
                    &prepared, &input, &ids, &weights, &scratch, &output, 0,
                )
                .unwrap();
            stage_graph
                .encode_dynamic_q4_moe_batched_stage(
                    &prepared, &input, &ids, &weights, &scratch, &output, 1,
                )
                .unwrap();
            stage_graph
                .encode_dynamic_q4_moe_batched_stage(
                    &prepared, &input, &ids, &weights, &scratch, &output, 2,
                )
                .unwrap();
            stage_graph.finish(&context).unwrap();
            let stage = context.read_runtime_f32(&scratch, 8 * 1024).unwrap();
            let down_stage = context
                .read_runtime_f32(
                    &scratch
                        .subregion(8 * 1024 * 4 + 8 * 512 * 4, 8 * 1024 * 4)
                        .unwrap(),
                    8 * 1024,
                )
                .unwrap();
            let mut graph = context.begin_decode().unwrap();
            graph
                .encode_dynamic_q4_moe(&prepared, &input, &ids, &weights, &output)
                .unwrap();
            graph.finish(&context).unwrap();
            (
                stage,
                down_stage,
                context.read_runtime_f32(&output, 1024).unwrap(),
            )
        };

        let (dense_stage, dense_down, dense) = run(PackingLayout::Q4Transaction144, 144);
        let (padded_stage, padded_down, padded) = run(PackingLayout::Q4Transaction160, 160);
        let stage_error = dense_stage
            .iter()
            .zip(&padded_stage)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let max_error = dense
            .iter()
            .zip(&padded)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let down_error = dense_down
            .iter()
            .zip(&padded_down)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(dense.iter().all(|value| value.is_finite() && *value > 0.0));
        assert!(
            max_error < 1.0e-2,
            "transaction layout error: {max_error}, stage error: {stage_error}, down error: {down_error}, dense[0]={}, padded[0]={}, down dense[0]={}, down padded[0]={}, stage dense[0]={}, stage padded[0]={}",
            dense[0],
            padded[0],
            dense_down[0],
            padded_down[0],
            dense_stage[0],
            padded_stage[0]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn v2_decode_kernels_compile_and_execute_in_metal_graphs() {
        use objc2_metal::MTLSize;
        use platform::DecodeKernel;

        fn bytes_of<T>(value: &T) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
            }
        }

        #[repr(C)]
        struct NormParams {
            n: u32,
            epsilon: f32,
        }
        #[repr(C)]
        struct VectorParams {
            n: u32,
        }
        #[repr(C)]
        struct ResidualParams {
            n: u32,
            scale: f32,
        }
        #[repr(C)]
        struct PositionParams {
            position: u32,
        }
        #[repr(C)]
        struct AttentionParams {
            length: u32,
        }

        let context = platform::MetalContext::system_default().expect("Metal device");
        let vector = context.allocate_runtime_buffer(4096).unwrap();
        let vector_b = context.allocate_runtime_buffer(4096).unwrap();
        let vector_c = context.allocate_runtime_buffer(4096).unwrap();
        let cache = context.allocate_runtime_buffer(512 * 8 * 64 * 4).unwrap();

        let run = |kernel, buffers, params, threads, group| {
            let mut graph = context.begin_decode().unwrap();
            graph
                .encode_decode_kernel(&context, kernel, buffers, params, threads, group)
                .unwrap();
            graph.finish(&context).unwrap();
        };
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let vector_threads = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        let vector_group = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let norm_params = NormParams {
            n: 1024,
            epsilon: 1e-5,
        };
        let vector_params = VectorParams { n: 1024 };
        let residual_params = ResidualParams {
            n: 1024,
            scale: 1.0,
        };
        let position_params = PositionParams { position: 0 };
        let attention_params = AttentionParams { length: 1 };
        let topk_params = VectorParams { n: 32 };
        let norm_buffers = [&vector, &vector_b, &vector_c];
        let conversion_buffers = [&vector, &vector_b];
        let residual_buffers = [&vector, &vector_b, &vector_c];
        let rope_buffers = [&vector, &vector_b, &vector_c, &vector, &cache, &cache];
        let attention_buffers = [&vector, &cache, &cache, &vector_b];
        let topk_buffers = [&vector, &vector_b, &vector_c];
        run(
            DecodeKernel::RmsNorm,
            &norm_buffers,
            bytes_of(&norm_params),
            one,
            one,
        );
        run(
            DecodeKernel::F32ToF16,
            &conversion_buffers,
            bytes_of(&vector_params),
            vector_threads,
            vector_group,
        );
        run(
            DecodeKernel::F16ToF32,
            &conversion_buffers,
            bytes_of(&vector_params),
            vector_threads,
            vector_group,
        );
        run(
            DecodeKernel::Residual,
            &residual_buffers,
            bytes_of(&residual_params),
            vector_threads,
            vector_group,
        );
        run(
            DecodeKernel::RopeKv,
            &rope_buffers,
            bytes_of(&position_params),
            one,
            one,
        );
        let rope_query_simd_buffers = [&vector, &vector];
        run(
            DecodeKernel::RopeQuerySimd,
            &rope_query_simd_buffers,
            bytes_of(&position_params),
            MTLSize {
                width: 512,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        let rope_kv_simd_buffers = [&vector, &vector, &cache, &cache];
        run(
            DecodeKernel::RopeKvSimd,
            &rope_kv_simd_buffers,
            bytes_of(&position_params),
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        run(
            DecodeKernel::Attention,
            &attention_buffers,
            bytes_of(&attention_params),
            vector_threads,
            vector_group,
        );
        run(
            DecodeKernel::TopK,
            &topk_buffers,
            bytes_of(&topk_params),
            one,
            one,
        );
    }
}
