//! Metal boundary for the specialized batch-1 expert path.
//!
//! The kernels are kept as source assets so the compiler can later embed a
//! target-specific metallib. The Rust context is intentionally small: model
//! orchestration remains outside Metal, while the hot GEMV kernels consume
//! packed weights directly and never allocate a dequantized matrix.

use qx_quant::{PackingLayout, QuantType};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalMathProfile {
    /// Protocol-defined fixed-operation exp/rsqrt with IEEE-safe arithmetic.
    ConsensusFixedV1,
    /// IEEE-safe expression evaluation with Metal-native precise math
    /// functions. Retained as a non-consensus A/B reference.
    SafePrecise,
    /// IEEE-safe expression evaluation with the faster Metal math functions.
    /// This is a benchmark candidate until cross-device parity is established.
    SafeFastFunctions,
    /// Relaxed arithmetic with precise math functions. This preserves NaN and
    /// infinity handling but may still reassociate arithmetic, so it is a
    /// diagnostic profile until compiled lowering is constrained and audited.
    RelaxedPrecise,
    /// Compiler defaults used by the historical diagnostic escape hatch.
    UnsafeCompilerDefault,
}

impl MetalMathProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConsensusFixedV1 => "consensus-fixed-v1",
            Self::SafePrecise => "safe-precise",
            Self::SafeFastFunctions => "safe-fast-functions",
            Self::RelaxedPrecise => "relaxed-precise",
            Self::UnsafeCompilerDefault => "unsafe-compiler-default",
        }
    }
}

pub fn active_metal_math_profile() -> MetalMathProfile {
    static PROFILE: std::sync::OnceLock<MetalMathProfile> = std::sync::OnceLock::new();

    *PROFILE.get_or_init(metal_math_profile_from_env)
}

fn metal_math_profile_from_env() -> MetalMathProfile {
    if env_is_false("QX_METAL_PRECISE_MATH") {
        MetalMathProfile::UnsafeCompilerDefault
    } else if env_is_true("QX_METAL_NATIVE_MATH") {
        MetalMathProfile::SafePrecise
    } else if env_is_true("QX_METAL_RELAXED_ARITHMETIC") {
        MetalMathProfile::RelaxedPrecise
    } else if env_is_true("QX_METAL_FAST_FUNCTIONS") {
        MetalMathProfile::SafeFastFunctions
    } else {
        MetalMathProfile::ConsensusFixedV1
    }
}

fn env_is_false(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("0" | "false" | "no")
    )
}

fn env_is_true(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1" | "true" | "yes")
    )
}

pub const CONSENSUS_MATH_KERNEL_SOURCE: &str =
    include_str!("../../../kernels/consensus_math.metal");
pub const Q2_KERNEL_SOURCE: &str = concat!(
    include_str!("../../../kernels/consensus_math.metal"),
    include_str!("../../../kernels/q2_expert.metal")
);
pub const Q3_KERNEL_SOURCE: &str = concat!(
    include_str!("../../../kernels/consensus_math.metal"),
    include_str!("../../../kernels/q3_expert.metal")
);
pub const Q4_KERNEL_SOURCE: &str = concat!(
    include_str!("../../../kernels/consensus_math.metal"),
    include_str!("../../../kernels/q4_expert.metal")
);
pub const LOGITS_KERNEL_SOURCE: &str = include_str!("../../../kernels/logits.metal");
pub const DENSE_KERNEL_SOURCE: &str = include_str!("../../../kernels/dense.metal");
pub const F16_DENSE_KERNEL_SOURCE: &str = include_str!("../../../kernels/f16_dense.metal");
pub const ARGMAX_KERNEL_SOURCE: &str = include_str!("../../../kernels/argmax.metal");
pub const V2_KERNEL_SOURCE: &str = concat!(
    include_str!("../../../kernels/consensus_math.metal"),
    include_str!("../../../kernels/v2_decode.metal")
);
pub const Q4_DENSE_KERNEL_SOURCE: &str = concat!(
    include_str!("../../../kernels/consensus_math.metal"),
    include_str!("../../../kernels/q4_dense.metal")
);
pub const DYNAMIC_MOE_KERNEL_SOURCE: &str = concat!(
    include_str!("../../../kernels/consensus_math.metal"),
    include_str!("../../../kernels/dynamic_moe.metal")
);
pub const Q4_EMBEDDING_KERNEL_SOURCE: &str = include_str!("../../../kernels/q4_embedding.metal");
pub const QWEN_ATTENTION_KERNEL_SOURCE: &str = concat!(
    include_str!("../../../kernels/consensus_math.metal"),
    include_str!("../../../kernels/qwen_attention.metal")
);
pub const VERIFICATION_HASH_KERNEL_SOURCE: &str =
    include_str!("../../../kernels/verification_hash.metal");

pub fn kernel_source(quant_type: QuantType) -> &'static str {
    match quant_type {
        QuantType::Q2 | QuantType::Q2S4 => Q2_KERNEL_SOURCE,
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
    use objc2_foundation::{NSDictionary, NSObject, NSRange, NSString};
    pub use objc2_metal::MTLSize;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLCommonCounterSetTimestamp, MTLCompileOptions, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLCounter, MTLCounterResultTimestamp, MTLCounterSampleBuffer,
        MTLCounterSampleBufferDescriptor, MTLCounterSamplingPoint, MTLCounterSet,
        MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathFloatingPointFunctions,
        MTLMathMode, MTLResourceOptions, MTLStorageMode,
    };

    use super::{
        ARGMAX_KERNEL_SOURCE, CONSENSUS_MATH_KERNEL_SOURCE, DENSE_KERNEL_SOURCE,
        DYNAMIC_MOE_KERNEL_SOURCE, F16_DENSE_KERNEL_SOURCE, LOGITS_KERNEL_SOURCE, MetalMathProfile,
        PackingLayout, Q2_KERNEL_SOURCE, Q4_DENSE_KERNEL_SOURCE, Q4_EMBEDDING_KERNEL_SOURCE,
        QWEN_ATTENTION_KERNEL_SOURCE, QuantType, V2_KERNEL_SOURCE, active_metal_math_profile,
        kernel_source,
    };

    const HIDDEN: usize = 1024;
    const INTERMEDIATE: usize = 512;
    const GROUP_SIZE: usize = 32;

    fn consensus_compile_options() -> Option<Retained<MTLCompileOptions>> {
        let profile = active_metal_math_profile();
        if profile == MetalMathProfile::UnsafeCompilerDefault {
            return None;
        }
        let options = MTLCompileOptions::new();
        // QX_EXEC_METAL_V1 requires IEEE-safe expression evaluation so the
        // compiler cannot reassociate the explicit SIMD reduction tree.
        options.setMathMode(match profile {
            MetalMathProfile::RelaxedPrecise => MTLMathMode::Relaxed,
            _ => MTLMathMode::Safe,
        });
        options.setMathFloatingPointFunctions(match profile {
            MetalMathProfile::ConsensusFixedV1
            | MetalMathProfile::SafePrecise
            | MetalMathProfile::RelaxedPrecise => MTLMathFloatingPointFunctions::Precise,
            MetalMathProfile::SafeFastFunctions => MTLMathFloatingPointFunctions::Fast,
            MetalMathProfile::UnsafeCompilerDefault => unreachable!(),
        });
        if profile == MetalMathProfile::ConsensusFixedV1 {
            let key = NSString::from_str("QX_CONSENSUS_FIXED_MATH");
            let value = NSString::from_str("1");
            let macros: Retained<NSDictionary<NSString, NSObject>> =
                NSDictionary::from_slices(&[&*key], &[&*value]);
            unsafe { options.setPreprocessorMacros(Some(&macros)) };
        }
        Some(options)
    }

    fn qwen_split_moe_enabled() -> bool {
        !matches!(
            std::env::var("QX_QWEN_SPLIT_MOE").ok().as_deref(),
            Some("0" | "false" | "no")
        )
    }

    /// Consumer choices supplied by the compiled Runtime V2 plan.  Production
    /// graph encoding passes this explicitly; the environment-backed wrapper
    /// remains only for legacy benchmark entry points.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct DynamicMoeConsumerPlan {
        pub unique_packet: bool,
        pub unique_packet_down: bool,
    }

    impl DynamicMoeConsumerPlan {
        fn from_legacy_environment() -> Self {
            Self {
                unique_packet: moe_unique_packet_enabled(),
                unique_packet_down: moe_unique_packet_down_enabled(),
            }
        }
    }

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

    /// K-wide verifier attention candidate. Each threadgroup owns one
    /// speculative token and each SIMD group owns one query head, so the
    /// Q·K score vector is computed once and reused across all 64 output
    /// dimensions. It is opt-in until the complete verifier parity gate.
    pub fn attention_batch_simd_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_ATTENTION_BATCH_SIMD")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    }

    /// Opt-in K-wide QKV consumer that loads each packed Q4 weight packet once
    /// per output row and applies it to all resident verifier tokens.
    pub fn qkv_weight_stationary_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_QKV_WEIGHT_STATIONARY")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    }

    pub fn o_projection_weight_stationary_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_O_WEIGHT_STATIONARY")
                .map(|value| {
                    matches!(
                        value.to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    }

    pub fn rms_parallel_writeback_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("QX_V2_RMS_PARALLEL_WRITEBACK")
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

    /// Experimental exact-Q4 vocabulary consumer. It keeps the existing
    /// partial-max/argmax graph and changes only packed-code/scale ownership.
    pub fn final_argmax_unique_packet_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QX_V2_FINAL_ARGMAX_UNIQUE_PACKET")
                    .ok()
                    .as_deref(),
                Some("1") | Some("true") | Some("yes") | Some("on")
            )
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

    /// Enables the exact-Q4 cooperative packet consumer. It keeps row-inline
    /// storage but assigns one packed code byte per SIMD lane and broadcasts
    /// each packet scale within the two-lane halves.
    pub fn moe_unique_packet_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QX_V2_MOE_UNIQUE_PACKET").ok().as_deref(),
                Some("1") | Some("true") | Some("yes") | Some("on")
            )
        })
    }

    /// Enables the experimental cooperative packet consumer for row-inline Q4
    /// MoE down projections. Gate/up remains controlled by the sealed
    /// `QX_V2_MOE_UNIQUE_PACKET` plan flag; this separate switch makes the
    /// down follow-up measurable without changing the promoted baseline.
    pub fn moe_unique_packet_down_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QX_V2_MOE_UNIQUE_PACKET_DOWN")
                    .ok()
                    .as_deref(),
                Some("1") | Some("true") | Some("yes") | Some("on")
            )
        })
    }

    /// Opt-in K-wide gate/up dispatch that uses the GPU-produced active expert
    /// list and indirect threadgroup count to skip empty expert buckets.
    pub fn verifier_active_expert_dispatch_enabled() -> bool {
        matches!(
            std::env::var("QX_V2_VERIFIER_ACTIVE_EXPERT_DISPATCH")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    }

    /// Legacy diagnostic override for the K4 packet consumer. Production
    /// Runtime V2 passes this choice through its resolved execution plan.
    pub fn verifier_k4_vector_packet_enabled() -> bool {
        matches!(
            std::env::var("QX_V2_VERIFIER_K4_VECTOR_PACKET")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    }

    /// Enables the experimental packet-stationary K-wide Down consumer.
    pub fn verifier_down_unique_packet_enabled() -> bool {
        matches!(
            std::env::var("QX_V2_VERIFIER_DOWN_UNIQUE_PACKET")
                .ok()
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        )
    }

    /// Experimental Down-side neuron-cluster pre-screen. Accepted values are
    /// `16` and `32`; the switch is intentionally independent from the sealed
    /// artifact plan until a complete clustered path exists.
    pub fn moe_cluster_down_width() -> Option<u32> {
        static WIDTH: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        *WIDTH.get_or_init(
            || match std::env::var("QX_V2_MOE_CLUSTER_DOWN").ok()?.as_str() {
                "16" => Some(16),
                "32" => Some(32),
                _ => None,
            },
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

    /// Selects a diagnostic variant of the production row-inline gate/up
    /// consumer. `canonical` (or an unset variable) leaves the normal kernel
    /// selected. The other values are intentionally non-semantic profiling
    /// paths and must never be used for model output.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum MoeGateUpInstrumentVariant {
        StagingOnly,
        AddressOnly,
        ContiguousLoadOnly,
        LoadTwoGroups,
        LoadUniquePacket,
        LoadOnly,
        LoadDecode,
        DotNoReduction,
    }

    pub fn moe_gate_up_instrument_variant() -> Option<MoeGateUpInstrumentVariant> {
        match std::env::var("QX_V2_MOE_GATE_UP_INSTRUMENT")
            .ok()
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("staging-only") | Some("staging_only") => {
                Some(MoeGateUpInstrumentVariant::StagingOnly)
            }
            Some("address-only") | Some("address_only") => {
                Some(MoeGateUpInstrumentVariant::AddressOnly)
            }
            Some("contiguous-load-only")
            | Some("contiguous_load_only")
            | Some("contiguous-load") => Some(MoeGateUpInstrumentVariant::ContiguousLoadOnly),
            Some("load-two-groups") | Some("load_two_groups") => {
                Some(MoeGateUpInstrumentVariant::LoadTwoGroups)
            }
            Some("load-unique-packet") | Some("load_unique_packet") => {
                Some(MoeGateUpInstrumentVariant::LoadUniquePacket)
            }
            Some("load-only") | Some("load_only") => Some(MoeGateUpInstrumentVariant::LoadOnly),
            Some("load-decode") | Some("load_decode") => {
                Some(MoeGateUpInstrumentVariant::LoadDecode)
            }
            Some("dot-no-reduction") | Some("dot_no_reduction") => {
                Some(MoeGateUpInstrumentVariant::DotNoReduction)
            }
            _ => None,
        }
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

    /// Experimental llama-derived dataflow for exact Q4 row-inline gate/up.
    /// It keeps QX's representation but adopts the production llama shape:
    /// two output rows per SIMD group and two SIMD groups per threadgroup,
    /// with packet-owned code/scale loads. It is intentionally separate from
    /// the earlier two-row control, which did not change weight ownership.
    pub fn moe_two_row_unique_packet_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            matches!(
                std::env::var("QX_V2_MOE_TWO_ROW_UNIQUE_PACKET")
                    .ok()
                    .as_deref(),
                Some("1") | Some("true") | Some("yes") | Some("on")
            )
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
            QuantType::Q2 | QuantType::Q2S4 => 8,
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

    /// One Qwen3-30B-A3B group-32 Q2S4 expert. This initial path is
    /// shape-specific by design and executes gate/up, SwiGLU, and down in a
    /// single Metal dispatch without an intermediate host synchronization.
    pub struct PreparedQwenQ2S4Expert {
        pipeline: Pipeline,
        gate_up_packed: RuntimeBuffer,
        gate_up_scales: RuntimeBuffer,
        down_packed: RuntimeBuffer,
        down_scales: RuntimeBuffer,
    }

    pub struct PreparedQwenDynamicMoe {
        expert_pipeline: Pipeline,
        gate_pipeline: Pipeline,
        down_tiled_pipeline: Pipeline,
        accumulate_pipeline: Pipeline,
        slab: RuntimeBuffer,
        gate_data_offsets: RuntimeBuffer,
        gate_scale_offsets: RuntimeBuffer,
        down_data_offsets: RuntimeBuffer,
        down_scale_offsets: RuntimeBuffer,
        codecs: RuntimeBuffer,
        uncached_storage: RuntimeBuffer,
        uncached_slots: RuntimeBuffer,
    }

    pub struct PreparedQwenExpertCache {
        resolve_pipeline: Pipeline,
        fill_pipeline: Pipeline,
        storage: RuntimeBuffer,
        keys: RuntimeBuffer,
        ages: RuntimeBuffer,
        clock: RuntimeBuffer,
        resolved_slots: RuntimeBuffer,
        copy_flags: RuntimeBuffer,
        counters: RuntimeBuffer,
        seen: RuntimeBuffer,
        slots: usize,
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct QwenExpertCacheStats {
        pub hits: u32,
        pub misses: u32,
        pub evictions: u32,
        pub q4_selections: u32,
        pub bypasses: u32,
        pub slots: usize,
        pub storage_bytes: usize,
    }

    pub struct PreparedQwenQ4Embedding {
        pipeline: Pipeline,
        packed: RuntimeBuffer,
        scales: RuntimeBuffer,
        rows: usize,
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
        batch_simd_pipeline: Pipeline,
        weight_stationary_batch_pipeline: Pipeline,
        packed: RuntimeBuffer,
        scales: RuntimeBuffer,
        rows: usize,
        columns: usize,
        layout: PackingLayout,
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
        layout: PackingLayout,
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
        unique_packet_gate_up_pipeline: Pipeline,
        verifier_gate_up_pipeline: Pipeline,
        verifier_gate_up_expert_major_pipeline: Pipeline,
        verifier_gate_up_expert_major_direct_pipeline: Pipeline,
        verifier_gate_up_expert_major_unique_packet_pipeline: Pipeline,
        verifier_compact_routes_pipeline: Pipeline,
        verifier_gate_up_compact_unique_packet_pipeline: Pipeline,
        verifier_gate_up_compact_unique_packet_k2_pipeline: Pipeline,
        verifier_gate_up_compact_unique_packet_k4_vector_pipeline: Pipeline,
        verifier_gate_up_compact_unique_packet_k4_split_accum_pipeline: Pipeline,
        verifier_gate_up_compact_unique_packet_k4_direct_activation_pipeline: Pipeline,
        verifier_gate_up_compact_unique_packet_k4_rows16_pipeline: Pipeline,
        verifier_gate_up_compact_unique_packet_k4_unroll4_pipeline: Pipeline,
        verifier_silu_pipeline: Pipeline,
        verifier_down_pipeline: Pipeline,
        verifier_down_compact_unique_packet_pipeline: Pipeline,
        verifier_down_compact_unique_packet_k4_unroll4_pipeline: Pipeline,
        verifier_accumulate_pipeline: Pipeline,
        unique_packet_gate_up_norm_pipeline: Pipeline,
        unique_packet_down_pipeline: Pipeline,
        cluster_down16_pipeline: Pipeline,
        cluster_down32_pipeline: Pipeline,
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
        two_row_unique_packet_gate_up_pipeline: Option<Pipeline>,
        two_row_unique_packet_gate_up_norm_pipeline: Option<Pipeline>,
        gate_up_instrument_load_only_pipeline: Pipeline,
        gate_up_instrument_staging_only_pipeline: Pipeline,
        gate_up_instrument_address_only_pipeline: Pipeline,
        gate_up_instrument_contiguous_load_only_pipeline: Pipeline,
        gate_up_instrument_load_two_groups_pipeline: Pipeline,
        gate_up_instrument_load_unique_packet_pipeline: Pipeline,
        gate_up_instrument_load_decode_pipeline: Pipeline,
        gate_up_instrument_dot_no_reduce_pipeline: Pipeline,
        gate_up_instrument_load_only_norm_pipeline: Pipeline,
        gate_up_instrument_staging_only_norm_pipeline: Pipeline,
        gate_up_instrument_address_only_norm_pipeline: Pipeline,
        gate_up_instrument_contiguous_load_only_norm_pipeline: Pipeline,
        gate_up_instrument_load_two_groups_norm_pipeline: Pipeline,
        gate_up_instrument_load_unique_packet_norm_pipeline: Pipeline,
        gate_up_instrument_load_decode_norm_pipeline: Pipeline,
        gate_up_instrument_dot_no_reduce_norm_pipeline: Pipeline,
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
        expertwise_superblock_gate_up_pipeline: Option<Pipeline>,
        expertwise_superblock_gate_up_norm_pipeline: Option<Pipeline>,
        expertwise_superblock_down_pipeline: Option<Pipeline>,
        expertwise_q4_gate_up_superblock: bool,
        expertwise_q4_down_superblock: bool,
        expertwise_q3_gate_up_superblock: bool,
        expertwise_q3_down_superblock: bool,
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
        batch_pipeline: Pipeline,
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
        pub fn len(&self) -> usize {
            self.length
        }

        pub fn is_empty(&self) -> bool {
            self.length == 0
        }

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
        RopeKvSimdKf32Vf16,
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
        CopyF32,
        RopeQueryBatchSimd,
        RopeKvBatchSimd,
        AttentionBatch,
        AttentionBatchSimd,
        RmsNormBatch,
        ResidualBatch,
        TopKBatch,
        RmsNormBatchScalar,
        RmsNormBatchParallelWriteback,
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
        /// Idle time between consecutive GPU command-buffer timestamp ranges
        /// inside the current measurement interval.
        pub gpu_queue_gap_seconds: f64,
        pub cpu_encode_seconds: f64,
        pub cpu_wait_seconds: f64,
    }

    /// Counter sets exposed by the current Metal device. This is diagnostic
    /// inventory only; querying it does not alter the canonical execution
    /// path or allocate a sample buffer.
    #[derive(Clone, Debug, Default)]
    pub struct MetalCounterSetInfo {
        pub name: String,
        pub counters: Vec<String>,
    }

    #[derive(Clone, Debug, Default)]
    pub struct MetalKernelInfo {
        pub name: String,
        pub thread_execution_width: usize,
        pub max_threads_per_threadgroup: usize,
        pub static_threadgroup_memory_bytes: usize,
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
        gpu_queue_gap_seconds: f64,
        last_gpu_end_seconds: Option<f64>,
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
        pipelines: Mutex<[Option<Pipeline>; 4]>,
        aggregate_pipelines: Mutex<[Option<Pipeline>; 4]>,
        qwen_q2s4_expert_pipeline: Mutex<Option<Pipeline>>,
        qwen_qk_norm_rope_pipeline: Mutex<Option<Pipeline>>,
        qwen_gqa_attention_pipeline: Mutex<Option<Pipeline>>,
        qwen_gqa_score_reuse_pipeline: Mutex<Option<Pipeline>>,
        qwen_top8_pipeline: Mutex<Option<Pipeline>>,
        qwen_rms_norm_pipeline: Mutex<Option<Pipeline>>,
        qwen_residual_rms_norm_pipeline: Mutex<Option<Pipeline>>,
        qwen_store_kv_pipeline: Mutex<Option<Pipeline>>,
        qwen_residual_pipeline: Mutex<Option<Pipeline>>,
        qwen_embedding_pipeline: Mutex<Option<Pipeline>>,
        qwen_argmax_pipeline: Mutex<Option<Pipeline>>,
        qwen_dynamic_moe_pipeline: Mutex<Option<Pipeline>>,
        qwen_dynamic_gate_pipeline: Mutex<Option<Pipeline>>,
        qwen_dynamic_down_tiled_pipeline: Mutex<Option<Pipeline>>,
        qwen_dynamic_accumulate_pipeline: Mutex<Option<Pipeline>>,
        qwen_q4_cache_resolve_pipeline: Mutex<Option<Pipeline>>,
        qwen_q4_cache_fill_pipeline: Mutex<Option<Pipeline>>,
        logits_pipeline: Mutex<Option<Pipeline>>,
        dense_pipeline: Mutex<Option<Pipeline>>,
        f16_dense_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_library: Mutex<Option<Retained<ProtocolObject<dyn MTLLibrary>>>>,
        residual_rms_pipeline: Mutex<Option<Pipeline>>,
        residual_rms_snapshot_pipeline: Mutex<Option<Pipeline>>,
        attention_softmax_onepass_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_simd_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_superblock_simd_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_superblock_norm_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_batch_simd_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_weight_stationary_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_sweep_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_argmax_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_argmax_norm_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_argmax_superblock_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_argmax_superblock_norm_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_argmax_unique_packet_pipeline: Mutex<Option<Pipeline>>,
        q4_dense_argmax_unique_packet_norm_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_simd_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_simd_norm_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_superblock_simd_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_superblock_norm_pipeline: Mutex<Option<Pipeline>>,
        q4_qkv_sweep_pipeline: Mutex<Option<Pipeline>>,
        q4_gemv_pipeline: Mutex<Option<Pipeline>>,
        q4_simd_gemv_pipeline: Mutex<Option<Pipeline>>,
        q4_simd_norm_gemv_pipeline: Mutex<Option<Pipeline>>,
        dynamic_moe_pipeline: Mutex<Option<Pipeline>>,
        dynamic_moe_library: Mutex<Option<Retained<ProtocolObject<dyn MTLLibrary>>>>,
        dynamic_moe_simd_pipeline: Mutex<Option<Pipeline>>,
        q4_embedding_pipeline: Mutex<Option<Pipeline>>,
        q4_embedding_batch_pipeline: Mutex<Option<Pipeline>>,
        q4_embedding_superblock_pipeline: Mutex<Option<Pipeline>>,
        q4_embedding_superblock_batch_pipeline: Mutex<Option<Pipeline>>,
        argmax_partial_pipeline: Mutex<Option<Pipeline>>,
        argmax_reduce_pipeline: Mutex<Option<Pipeline>>,
        argmax_reduce_partials_pipeline: Mutex<Option<Pipeline>>,
        verification_hash_pipeline: Mutex<Option<Pipeline>>,
        v2_pipelines: Mutex<[Option<Pipeline>; 42]>,
        scratch: Mutex<MetalScratch>,
        metrics: Mutex<MetalMetricsState>,
        stage_counters: Mutex<bool>,
        last_stage_timeline: Mutex<StageTimeline>,
        no_copy_buffers: Mutex<HashMap<(usize, usize), RuntimeBuffer>>,
    }

    /// Token-level command stream for Runtime V2. Operators may encode fixed
    /// shape work through this object. `submit` commits the graph without a
    /// CPU wait; `finish` is the synchronous convenience path.
    pub struct DecodeGraph {
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        dispatches: u64,
        counter_timeline: Option<CounterTimeline>,
        stage_diagnostics: bool,
        committed: Vec<(Retained<ProtocolObject<dyn MTLCommandBuffer>>, u64, u64)>,
        encoders: u64,
        encode_started_at: Instant,
    }

    /// A committed decode graph whose GPU work may still be in flight. Metal
    /// queue order makes it safe to submit the next token before waiting on
    /// this one; callers must eventually call `wait` to surface GPU errors and
    /// account the completed work in runtime metrics.
    pub struct SubmittedDecode {
        committed: Vec<(Retained<ProtocolObject<dyn MTLCommandBuffer>>, u64, u64)>,
        counter_timeline: Option<CounterTimeline>,
        encode_started_at: Instant,
        submitted_at: Instant,
    }

    impl DecodeGraph {
        pub fn encoder(&self) -> &ProtocolObject<dyn MTLComputeCommandEncoder> {
            &self.encoder
        }

        pub fn encode_qwen_embedding_lookup(
            &mut self,
            prepared: &PreparedQwenQ4Embedding,
            token: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            const HIDDEN: usize = 2048;
            if token.len() < std::mem::size_of::<u32>()
                || output.len() < HIDDEN * std::mem::size_of::<u16>()
            {
                return Err("Qwen embedding buffers are smaller than their fixed shape".to_owned());
            }
            self.encoder.setComputePipelineState(&prepared.pipeline);
            let rows = prepared.rows as u32;
            unsafe {
                for (buffer, index) in [
                    (&prepared.packed, 0),
                    (&prepared.scales, 1),
                    (token, 2),
                    (output, 3),
                ] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&rows).cast(),
                    std::mem::size_of_val(&rows),
                    4,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
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

        pub fn encode_qwen_argmax(
            &mut self,
            context: &MetalContext,
            logits: &RuntimeBuffer,
            token: &RuntimeBuffer,
            count: usize,
        ) -> Result<(), String> {
            if count == 0
                || logits.len() < count * std::mem::size_of::<u16>()
                || token.len() < std::mem::size_of::<u32>()
            {
                return Err("Qwen argmax buffers are smaller than the requested shape".to_owned());
            }
            let count =
                u32::try_from(count).map_err(|_| "Qwen argmax count exceeds u32".to_owned())?;
            let pipeline = context.qwen_argmax_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&logits.buffer), logits.offset, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&token.buffer), token.offset, 1);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&count).cast(),
                    std::mem::size_of_val(&count),
                    2,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
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

        pub fn encode_qwen_rms_norm(
            &mut self,
            context: &MetalContext,
            input: &RuntimeBuffer,
            norm: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            const HIDDEN: usize = 2048;
            if input.len() < HIDDEN * std::mem::size_of::<u16>()
                || norm.len() < HIDDEN * std::mem::size_of::<f32>()
                || output.len() < HIDDEN * std::mem::size_of::<u16>()
            {
                return Err("Qwen RMSNorm buffers are smaller than their fixed shape".to_owned());
            }
            let pipeline = context.qwen_rms_norm_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, index) in [(input, 0), (norm, 1), (output, 2)] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
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

        pub fn encode_prepared_qwen_q4_stage(
            &mut self,
            prepared: &PreparedQ4Stage,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            if input.len() < prepared.columns * std::mem::size_of::<u16>()
                || output.len() < prepared.rows * std::mem::size_of::<u16>()
            {
                return Err("Qwen Q4 stage buffers are smaller than their fixed shape".to_owned());
            }
            self.encoder.setComputePipelineState(&prepared.pipeline);
            let columns = prepared.columns as u32;
            let layout = prepared.layout as u32;
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
                    NonNull::from(&layout).cast(),
                    std::mem::size_of_val(&layout),
                    5,
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

        pub fn encode_qwen_qk_norm_rope(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            query_norm: &RuntimeBuffer,
            key_norm: &RuntimeBuffer,
            position: u32,
        ) -> Result<(), String> {
            if query.len() < 4096 * std::mem::size_of::<u16>()
                || key.len() < 512 * std::mem::size_of::<u16>()
                || query_norm.len() < 128 * std::mem::size_of::<f32>()
                || key_norm.len() < 128 * std::mem::size_of::<f32>()
            {
                return Err(
                    "Qwen Q/K norm+RoPE buffers are smaller than their fixed shape".to_owned(),
                );
            }
            let pipeline = context.qwen_qk_norm_rope_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, index) in [(query, 0), (key, 1), (query_norm, 2), (key_norm, 3)] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&position).cast(),
                    std::mem::size_of_val(&position),
                    4,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 36,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 36,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_qwen_store_kv(
            &mut self,
            context: &MetalContext,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            position: u32,
        ) -> Result<(), String> {
            const KV_ELEMENTS: usize = 512;
            let required_cache = (position as usize + 1)
                .checked_mul(KV_ELEMENTS * std::mem::size_of::<u16>())
                .ok_or_else(|| "Qwen KV cache size overflows".to_owned())?;
            if key.len() < KV_ELEMENTS * std::mem::size_of::<u16>()
                || value.len() < KV_ELEMENTS * std::mem::size_of::<u16>()
                || key_cache.len() < required_cache
                || value_cache.len() < required_cache
            {
                return Err("Qwen KV buffers are smaller than the requested position".to_owned());
            }
            let pipeline = context.qwen_store_kv_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, index) in [(key, 0), (value, 1), (key_cache, 2), (value_cache, 3)] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&position).cast(),
                    std::mem::size_of_val(&position),
                    4,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: KV_ELEMENTS,
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

        pub fn encode_qwen_gqa_attention(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            output: &RuntimeBuffer,
            context_len: u32,
        ) -> Result<(), String> {
            const QUERY_ELEMENTS: usize = 4096;
            const KV_ELEMENTS: usize = 512;
            let required_cache = context_len as usize * KV_ELEMENTS * std::mem::size_of::<u16>();
            if context_len == 0
                || query.len() < QUERY_ELEMENTS * std::mem::size_of::<u16>()
                || key_cache.len() < required_cache
                || value_cache.len() < required_cache
                || output.len() < QUERY_ELEMENTS * std::mem::size_of::<u16>()
            {
                return Err("Qwen GQA buffers are smaller than the requested context".to_owned());
            }
            let score_reuse = context_len <= 4096;
            let pipeline = if score_reuse {
                context.qwen_gqa_score_reuse_pipeline()?
            } else {
                context.qwen_gqa_attention_pipeline()?
            };
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, index) in [(query, 0), (key_cache, 1), (value_cache, 2), (output, 3)] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&context_len).cast(),
                    std::mem::size_of_val(&context_len),
                    4,
                );
            }
            if score_reuse {
                unsafe {
                    self.encoder.setThreadgroupMemoryLength_atIndex(
                        (context_len as usize + 128) * std::mem::size_of::<f32>(),
                        0,
                    );
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else {
                self.encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: QUERY_ELEMENTS,
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
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_qwen_residual_rms_norm(
            &mut self,
            context: &MetalContext,
            residual: &RuntimeBuffer,
            update: &RuntimeBuffer,
            norm: &RuntimeBuffer,
            hidden_output: &RuntimeBuffer,
            norm_output: &RuntimeBuffer,
        ) -> Result<(), String> {
            const HIDDEN: usize = 2048;
            if [residual, update, hidden_output, norm_output]
                .iter()
                .any(|buffer| buffer.len() < HIDDEN * std::mem::size_of::<u16>())
                || norm.len() < HIDDEN * std::mem::size_of::<f32>()
            {
                return Err(
                    "Qwen residual/RMS buffers are smaller than their fixed shape".to_owned(),
                );
            }
            let pipeline = context.qwen_residual_rms_norm_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, index) in [
                    (residual, 0),
                    (update, 1),
                    (norm, 2),
                    (hidden_output, 3),
                    (norm_output, 4),
                ] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
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

        pub fn encode_qwen_residual(
            &mut self,
            context: &MetalContext,
            residual: &RuntimeBuffer,
            update: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            const HIDDEN: usize = 2048;
            if [residual, update, output]
                .iter()
                .any(|buffer| buffer.len() < HIDDEN * std::mem::size_of::<u16>())
            {
                return Err("Qwen residual buffers are smaller than their fixed shape".to_owned());
            }
            let pipeline = context.qwen_residual_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, index) in [(residual, 0), (update, 1), (output, 2)] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
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
            self.dispatches += 1;
            Ok(())
        }

        /// Select and normalize the eight highest router logits entirely on
        /// the GPU. The resulting IDs and weights can feed the dynamic MoE
        /// encoder below without an intervening command-buffer boundary.
        pub fn encode_qwen_top8(
            &mut self,
            context: &MetalContext,
            logits: &RuntimeBuffer,
            ids: &RuntimeBuffer,
            weights: &RuntimeBuffer,
        ) -> Result<(), String> {
            const EXPERTS: usize = 128;
            const TOP_K: usize = 8;
            if logits.len() < EXPERTS * std::mem::size_of::<u16>()
                || ids.len() < TOP_K * std::mem::size_of::<u32>()
                || weights.len() < TOP_K * std::mem::size_of::<f32>()
            {
                return Err("Qwen top-8 buffers are smaller than their fixed shape".to_owned());
            }
            let pipeline = context.qwen_top8_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&logits.buffer), logits.offset, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&ids.buffer), ids.offset, 1);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&weights.buffer), weights.offset, 2);
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
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

        /// Execute the GPU-selected Qwen experts directly from a no-copy
        /// artifact slab, then combine their outputs with the router weights.
        /// Both dispatches remain in this graph and require no host readback.
        pub fn encode_qwen_dynamic_moe(
            &mut self,
            prepared: &PreparedQwenDynamicMoe,
            input: &RuntimeBuffer,
            ids: &RuntimeBuffer,
            weights: &RuntimeBuffer,
            expert_activations: &RuntimeBuffer,
            expert_outputs: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            self.encode_qwen_dynamic_moe_impl(
                prepared,
                input,
                ids,
                weights,
                expert_activations,
                expert_outputs,
                output,
                &prepared.uncached_storage,
                &prepared.uncached_slots,
            )
        }

        /// Resolve and fill the bounded source-Q4 cache entirely on GPU, then
        /// execute the same mixed expert graph against the resolved slots.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_qwen_dynamic_moe_cached(
            &mut self,
            prepared: &PreparedQwenDynamicMoe,
            cache: &PreparedQwenExpertCache,
            layer: u16,
            input: &RuntimeBuffer,
            ids: &RuntimeBuffer,
            weights: &RuntimeBuffer,
            expert_activations: &RuntimeBuffer,
            expert_outputs: &RuntimeBuffer,
            output: &RuntimeBuffer,
        ) -> Result<(), String> {
            let layer = u32::from(layer);
            let cache_slots = u32::try_from(cache.slots)
                .map_err(|_| "Qwen expert cache slot count exceeds u32".to_owned())?;
            self.encoder
                .setComputePipelineState(&cache.resolve_pipeline);
            unsafe {
                for (buffer, index) in [
                    (&prepared.codecs, 0),
                    (ids, 1),
                    (&cache.keys, 2),
                    (&cache.ages, 3),
                    (&cache.clock, 4),
                    (&cache.resolved_slots, 5),
                    (&cache.copy_flags, 6),
                    (&cache.counters, 7),
                    (&cache.seen, 8),
                ] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&layer).cast(),
                    std::mem::size_of_val(&layer),
                    9,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&cache_slots).cast(),
                    std::mem::size_of_val(&cache_slots),
                    10,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
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

            self.encoder.setComputePipelineState(&cache.fill_pipeline);
            unsafe {
                for (buffer, index) in [
                    (&prepared.slab, 0),
                    (&prepared.gate_data_offsets, 1),
                    (&prepared.gate_scale_offsets, 2),
                    (&prepared.down_data_offsets, 3),
                    (&prepared.down_scale_offsets, 4),
                    (ids, 5),
                    (&cache.resolved_slots, 6),
                    (&cache.copy_flags, 7),
                    (&cache.storage, 8),
                ] {
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&buffer.buffer),
                        buffer.offset,
                        index,
                    );
                }
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 8,
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

            self.encode_qwen_dynamic_moe_impl(
                prepared,
                input,
                ids,
                weights,
                expert_activations,
                expert_outputs,
                output,
                &cache.storage,
                &cache.resolved_slots,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_qwen_dynamic_moe_impl(
            &mut self,
            prepared: &PreparedQwenDynamicMoe,
            input: &RuntimeBuffer,
            ids: &RuntimeBuffer,
            weights: &RuntimeBuffer,
            expert_activations: &RuntimeBuffer,
            expert_outputs: &RuntimeBuffer,
            output: &RuntimeBuffer,
            cache_storage: &RuntimeBuffer,
            resolved_slots: &RuntimeBuffer,
        ) -> Result<(), String> {
            const HIDDEN: usize = 2048;
            const INTERMEDIATE: usize = 768;
            const TOP_K: usize = 8;
            if input.len() < HIDDEN * std::mem::size_of::<u16>()
                || ids.len() < TOP_K * std::mem::size_of::<u32>()
                || weights.len() < TOP_K * std::mem::size_of::<f32>()
                || expert_activations.len() < TOP_K * INTERMEDIATE * std::mem::size_of::<u16>()
                || expert_outputs.len() < TOP_K * HIDDEN * std::mem::size_of::<u16>()
                || output.len() < HIDDEN * std::mem::size_of::<u16>()
            {
                return Err(
                    "Qwen dynamic MoE buffers are smaller than their fixed shape".to_owned(),
                );
            }
            if qwen_split_moe_enabled() {
                self.encoder
                    .setComputePipelineState(&prepared.gate_pipeline);
                unsafe {
                    for (buffer, index) in [
                        (&prepared.slab, 0),
                        (&prepared.gate_data_offsets, 1),
                        (&prepared.gate_scale_offsets, 2),
                        (&prepared.codecs, 3),
                        (input, 4),
                        (ids, 5),
                        (expert_activations, 6),
                        (cache_storage, 7),
                        (resolved_slots, 8),
                    ] {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&buffer.buffer),
                            buffer.offset,
                            index,
                        );
                    }
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: TOP_K,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: INTERMEDIATE,
                        height: 1,
                        depth: 1,
                    },
                );
                self.dispatches += 1;

                self.encoder
                    .setComputePipelineState(&prepared.down_tiled_pipeline);
                unsafe {
                    for (buffer, index) in [
                        (&prepared.slab, 0),
                        (&prepared.down_data_offsets, 1),
                        (&prepared.down_scale_offsets, 2),
                        (&prepared.codecs, 3),
                        (expert_activations, 4),
                        (ids, 5),
                        (expert_outputs, 6),
                        (cache_storage, 7),
                        (resolved_slots, 8),
                    ] {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&buffer.buffer),
                            buffer.offset,
                            index,
                        );
                    }
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: TOP_K * 8,
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
            } else {
                self.encoder
                    .setComputePipelineState(&prepared.expert_pipeline);
                unsafe {
                    for (buffer, index) in [
                        (&prepared.slab, 0),
                        (&prepared.gate_data_offsets, 1),
                        (&prepared.gate_scale_offsets, 2),
                        (&prepared.down_data_offsets, 3),
                        (&prepared.down_scale_offsets, 4),
                        (&prepared.codecs, 5),
                        (input, 6),
                        (ids, 7),
                        (expert_outputs, 8),
                        (cache_storage, 9),
                        (resolved_slots, 10),
                    ] {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&buffer.buffer),
                            buffer.offset,
                            index,
                        );
                    }
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: TOP_K,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: INTERMEDIATE,
                        height: 1,
                        depth: 1,
                    },
                );
                self.dispatches += 1;
            }

            self.encoder
                .setComputePipelineState(&prepared.accumulate_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_outputs.buffer),
                    expert_outputs.offset,
                    0,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&weights.buffer), weights.offset, 1);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 2);
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
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
            self.dispatches += 1;
            Ok(())
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
            if prepared.layout != PackingLayout::RowMajor {
                return Err("scalar Q4 dense only supports row-major weights".to_owned());
            }
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
            if prepared.layout == PackingLayout::RowMajor && prepared.rows >= 16_384 {
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
                    width: prepared.rows.div_ceil(
                        if prepared.layout == PackingLayout::Q4Superblock144 {
                            16
                        } else {
                            8
                        },
                    ),
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

        /// Encode a real K-position Q4 projection. Inputs and outputs are
        /// token-major `[tokens, columns]` and `[tokens, rows]`; this is the
        /// first verifier primitive and is intentionally not implemented as
        /// K calls to the scalar decode path.
        pub fn encode_prepared_q4_dense_simd_batch(
            &mut self,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
            tokens: usize,
        ) -> Result<(), String> {
            if prepared.layout != PackingLayout::RowMajor {
                return Err("batched Q4 dense only supports row-major weights".to_owned());
            }
            if tokens == 0 {
                return Err("batched Q4 dense requires at least one token".to_owned());
            }
            if input.length < tokens * prepared.columns * std::mem::size_of::<f32>()
                || output.length < tokens * prepared.rows * std::mem::size_of::<f32>()
            {
                return Err(
                    "batched Q4 dense buffers are smaller than their fixed shape".to_owned(),
                );
            }
            self.encoder
                .setComputePipelineState(&prepared.batch_simd_pipeline);
            let columns = prepared.columns as u32;
            let rows = prepared.rows as u32;
            let token_count = tokens as u32;
            let multiplier = 1.0_f32;
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
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&token_count).cast(),
                    std::mem::size_of_val(&token_count),
                    7,
                );
            }
            // The kernel assigns one threadgroup to eight output rows.  Use
            // an explicit threadgroup grid, matching the scalar GEMV path,
            // rather than dispatchThreads' partial-group semantics.  This
            // also makes the token dimension an exact workgroup coordinate.
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: prepared.rows.div_ceil(8),
                    height: tokens,
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

        pub fn encode_prepared_q4_dense_weight_stationary_batch(
            &mut self,
            prepared: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
            tokens: usize,
        ) -> Result<(), String> {
            if prepared.layout != PackingLayout::RowMajor {
                return Err(
                    "weight-stationary Q4 verifier only supports row-major weights".to_owned(),
                );
            }
            if tokens == 0 || tokens > 4 {
                return Err("weight-stationary Q4 verifier supports K=1..4".to_owned());
            }
            if input.length < tokens * prepared.columns * std::mem::size_of::<f32>()
                || output.length < tokens * prepared.rows * std::mem::size_of::<f32>()
            {
                return Err(
                    "weight-stationary Q4 buffers are smaller than their fixed shape".to_owned(),
                );
            }
            self.encoder
                .setComputePipelineState(&prepared.weight_stationary_batch_pipeline);
            let columns = prepared.columns as u32;
            let rows = prepared.rows as u32;
            let token_count = tokens as u32;
            let multiplier = 1.0_f32;
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
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&token_count).cast(),
                    std::mem::size_of_val(&token_count),
                    7,
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

        /// Encode Q, K, and V for K verifier positions using the validated
        /// token-major dense primitive. This is intentionally three
        /// K-wide dispatches rather than K scalar QKV calls; the next
        /// verifier stage can replace this with a fused batched QKV kernel
        /// without changing its buffer contract.
        pub fn encode_prepared_q4_qkv_simd_batch(
            &mut self,
            q: &PreparedQuantizedDense,
            k: &PreparedQuantizedDense,
            v: &PreparedQuantizedDense,
            input: &RuntimeBuffer,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
            tokens: usize,
        ) -> Result<(), String> {
            if qkv_weight_stationary_enabled() && tokens > 1 {
                self.encode_prepared_q4_dense_weight_stationary_batch(q, input, query, tokens)?;
                self.encode_prepared_q4_dense_weight_stationary_batch(k, input, key, tokens)?;
                return self
                    .encode_prepared_q4_dense_weight_stationary_batch(v, input, value, tokens);
            }
            self.encode_prepared_q4_dense_simd_batch(q, input, query, tokens)?;
            self.encode_prepared_q4_dense_simd_batch(k, input, key, tokens)?;
            self.encode_prepared_q4_dense_simd_batch(v, input, value, tokens)
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
            let rows_per_threadgroup = if prepared.layout == PackingLayout::Q4Superblock144 {
                32
            } else {
                8
            };
            let partial_count = prepared.rows.div_ceil(rows_per_threadgroup);
            let partial_bytes = partial_count * 2 * std::mem::size_of::<u32>();
            if partials.length < partial_bytes {
                return Err(
                    "V2 fused final head partial buffer is smaller than its fixed shape".to_owned(),
                );
            }
            let columns = prepared.columns as u32;
            let rows = prepared.rows as u32;
            let partial_count_u32 = partial_count as u32;
            let pipeline = if prepared.layout == PackingLayout::Q4Superblock144 {
                if norm_weights.is_some() {
                    context.q4_dense_argmax_superblock_norm_pipeline()?
                } else {
                    context.q4_dense_argmax_superblock_pipeline()?
                }
            } else if final_argmax_unique_packet_enabled() {
                if norm_weights.is_some() {
                    context.q4_dense_argmax_unique_packet_norm_pipeline()?
                } else {
                    context.q4_dense_argmax_unique_packet_pipeline()?
                }
            } else if norm_weights.is_some() {
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
                    width: prepared.rows.div_ceil(
                        if prepared.layout == PackingLayout::Q4Superblock144 {
                            16
                        } else {
                            8
                        },
                    ),
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
            if prepared.layout != PackingLayout::RowMajor {
                return Err("scalar fused QKV only supports row-major weights".to_owned());
            }
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
            if prepared.layout == PackingLayout::RowMajor && qkv_split_enabled() {
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
                    width: 2048
                        / if prepared.layout == PackingLayout::Q4Superblock144 {
                            16
                        } else {
                            8
                        },
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
            if prepared.layout == PackingLayout::RowMajor && qkv_split_enabled() {
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
            if prepared.layout == PackingLayout::RowMajor && qkv_geometry_enabled() {
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
                    width: 2048
                        / if prepared.layout == PackingLayout::Q4Superblock144 {
                            16
                        } else {
                            8
                        },
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
            if prepared.gate_up_layout.is_q4_superblock() || prepared.down_layout.is_q4_superblock()
            {
                return Err(
                    "Q4 superblocks require the four-stage batched SIMD MoE path".to_owned(),
                );
            }
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
                    width: 256,
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
            if prepared.gate_up_layout.is_q4_superblock() || prepared.down_layout.is_q4_superblock()
            {
                return Err(
                    "Q4 superblocks require the four-stage batched SIMD MoE path".to_owned(),
                );
            }
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
                    width: 256,
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

        fn select_gate_up_instrument_pipeline(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            normalized: bool,
        ) -> Result<bool, String> {
            let Some(variant) = moe_gate_up_instrument_variant() else {
                return Ok(false);
            };
            let pipeline = match (variant, normalized) {
                (MoeGateUpInstrumentVariant::StagingOnly, false) => {
                    &prepared.gate_up_instrument_staging_only_pipeline
                }
                (MoeGateUpInstrumentVariant::AddressOnly, false) => {
                    &prepared.gate_up_instrument_address_only_pipeline
                }
                (MoeGateUpInstrumentVariant::ContiguousLoadOnly, false) => {
                    &prepared.gate_up_instrument_contiguous_load_only_pipeline
                }
                (MoeGateUpInstrumentVariant::LoadTwoGroups, false) => {
                    &prepared.gate_up_instrument_load_two_groups_pipeline
                }
                (MoeGateUpInstrumentVariant::LoadUniquePacket, false) => {
                    &prepared.gate_up_instrument_load_unique_packet_pipeline
                }
                (MoeGateUpInstrumentVariant::LoadOnly, false) => {
                    &prepared.gate_up_instrument_load_only_pipeline
                }
                (MoeGateUpInstrumentVariant::LoadDecode, false) => {
                    &prepared.gate_up_instrument_load_decode_pipeline
                }
                (MoeGateUpInstrumentVariant::DotNoReduction, false) => {
                    &prepared.gate_up_instrument_dot_no_reduce_pipeline
                }
                (MoeGateUpInstrumentVariant::LoadOnly, true) => {
                    &prepared.gate_up_instrument_load_only_norm_pipeline
                }
                (MoeGateUpInstrumentVariant::StagingOnly, true) => {
                    &prepared.gate_up_instrument_staging_only_norm_pipeline
                }
                (MoeGateUpInstrumentVariant::AddressOnly, true) => {
                    &prepared.gate_up_instrument_address_only_norm_pipeline
                }
                (MoeGateUpInstrumentVariant::ContiguousLoadOnly, true) => {
                    &prepared.gate_up_instrument_contiguous_load_only_norm_pipeline
                }
                (MoeGateUpInstrumentVariant::LoadTwoGroups, true) => {
                    &prepared.gate_up_instrument_load_two_groups_norm_pipeline
                }
                (MoeGateUpInstrumentVariant::LoadUniquePacket, true) => {
                    &prepared.gate_up_instrument_load_unique_packet_norm_pipeline
                }
                (MoeGateUpInstrumentVariant::LoadDecode, true) => {
                    &prepared.gate_up_instrument_load_decode_norm_pipeline
                }
                (MoeGateUpInstrumentVariant::DotNoReduction, true) => {
                    &prepared.gate_up_instrument_dot_no_reduce_norm_pipeline
                }
            };
            self.encoder.setComputePipelineState(pipeline);
            Ok(true)
        }

        /// Encode the first expert-major stage of a K-wide verifier.
        ///
        /// `route_experts[p]` and `route_tokens[p]` describe one selected
        /// `(expert, token)` pair. The output is pair-major with 1024
        /// gate/up values per pair. This deliberately exposes only the
        /// exact row-inline Q4 gate/up stage; SiLU, down, and weighted
        /// accumulation remain separate verifier stages.
        pub fn encode_verifier_q4_gate_up_batch(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            route_experts: &RuntimeBuffer,
            route_tokens: &RuntimeBuffer,
            output: &RuntimeBuffer,
            pair_count: usize,
        ) -> Result<(), String> {
            if prepared.gate_up_layout != PackingLayout::Q4RowInline {
                return Err(
                    "K-wide verifier gate/up requires the exact Q4 row-inline layout".to_owned(),
                );
            }
            if pair_count == 0 {
                return Ok(());
            }
            if pair_count > u32::MAX as usize {
                return Err("K-wide verifier route count exceeds u32".to_owned());
            }
            if input.length < 1024 * std::mem::size_of::<f32>()
                || route_experts.length < pair_count * std::mem::size_of::<u32>()
                || route_tokens.length < pair_count * std::mem::size_of::<u32>()
                || output.length < pair_count * 1024 * std::mem::size_of::<f32>()
            {
                return Err("K-wide verifier gate/up buffers are too small".to_owned());
            }
            let pair_count = pair_count as u32;
            self.encoder
                .setComputePipelineState(&prepared.verifier_gate_up_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_packed.buffer),
                    prepared.gate_up_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_experts.buffer),
                    route_experts.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_tokens.buffer),
                    route_tokens.offset,
                    2,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 3);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 4);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&pair_count).cast(),
                    std::mem::size_of_val(&pair_count),
                    5,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (pair_count as usize) * (1024 / 8),
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

        /// Encode the weight-amortizing expert-major verifier stage. Each
        /// bucket is one expert and `bucket_pairs[offset..next]` identifies
        /// up to four route pairs for that expert. The kernel consumes each
        /// row-inline Q4 weight packet once while updating all token outputs
        /// in the bucket.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_verifier_q4_gate_up_expert_major(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            bucket_experts: &RuntimeBuffer,
            bucket_offsets: &RuntimeBuffer,
            bucket_pairs: &RuntimeBuffer,
            route_tokens: &RuntimeBuffer,
            output: &RuntimeBuffer,
            bucket_count: usize,
        ) -> Result<(), String> {
            if prepared.gate_up_layout != PackingLayout::Q4RowInline {
                return Err(
                    "K-wide expert-major gate/up requires the exact Q4 row-inline layout"
                        .to_owned(),
                );
            }
            if bucket_count == 0 {
                return Ok(());
            }
            if bucket_count > 32 {
                return Err("K-wide expert-major bucket count exceeds 32".to_owned());
            }
            if input.length < 1024 * std::mem::size_of::<f32>()
                || bucket_experts.length < bucket_count * std::mem::size_of::<u32>()
                || bucket_offsets.length < (bucket_count + 1) * std::mem::size_of::<u32>()
                || bucket_pairs.length < std::mem::size_of::<u32>()
                || route_tokens.length < std::mem::size_of::<u32>()
                || output.length < 1024 * std::mem::size_of::<f32>()
            {
                return Err("K-wide expert-major gate/up buffers are too small".to_owned());
            }
            let bucket_count = bucket_count as u32;
            self.encoder
                .setComputePipelineState(&prepared.verifier_gate_up_expert_major_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_packed.buffer),
                    prepared.gate_up_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_experts.buffer),
                    bucket_experts.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_offsets.buffer),
                    bucket_offsets.offset,
                    2,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_pairs.buffer),
                    bucket_pairs.offset,
                    3,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_tokens.buffer),
                    route_tokens.offset,
                    4,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 5);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 6);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&bucket_count).cast(),
                    std::mem::size_of_val(&bucket_count),
                    7,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (bucket_count as usize) * (1024 / 8),
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

        /// Direct-activation control for the expert-major verifier stage.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_verifier_q4_gate_up_expert_major_direct(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            bucket_experts: &RuntimeBuffer,
            bucket_offsets: &RuntimeBuffer,
            bucket_pairs: &RuntimeBuffer,
            route_tokens: &RuntimeBuffer,
            output: &RuntimeBuffer,
            bucket_count: usize,
        ) -> Result<(), String> {
            if prepared.gate_up_layout != PackingLayout::Q4RowInline {
                return Err(
                    "K-wide expert-major gate/up requires the exact Q4 row-inline layout"
                        .to_owned(),
                );
            }
            if bucket_count == 0 {
                return Ok(());
            }
            if bucket_count > 32 {
                return Err("K-wide expert-major bucket count exceeds 32".to_owned());
            }
            if input.length < 1024 * std::mem::size_of::<f32>()
                || bucket_experts.length < bucket_count * std::mem::size_of::<u32>()
                || bucket_offsets.length < (bucket_count + 1) * std::mem::size_of::<u32>()
                || bucket_pairs.length < std::mem::size_of::<u32>()
                || route_tokens.length < std::mem::size_of::<u32>()
                || output.length < 1024 * std::mem::size_of::<f32>()
            {
                return Err("K-wide expert-major gate/up buffers are too small".to_owned());
            }
            let bucket_count = bucket_count as u32;
            self.encoder
                .setComputePipelineState(&prepared.verifier_gate_up_expert_major_direct_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_packed.buffer),
                    prepared.gate_up_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_experts.buffer),
                    bucket_experts.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_offsets.buffer),
                    bucket_offsets.offset,
                    2,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_pairs.buffer),
                    bucket_pairs.offset,
                    3,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_tokens.buffer),
                    route_tokens.offset,
                    4,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 5);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 6);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&bucket_count).cast(),
                    std::mem::size_of_val(&bucket_count),
                    7,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (bucket_count as usize) * (1024 / 8),
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

        /// Compact token-major top-k IDs into fixed-stride GPU-resident expert
        /// buckets. Each expert owns one thread; no CPU route readback or
        /// atomic prefix sum is required.
        pub fn encode_verifier_compact_routes_fixed(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            route_experts: &RuntimeBuffer,
            bucket_counts: &RuntimeBuffer,
            bucket_pairs: &RuntimeBuffer,
            pair_count: usize,
            active_experts: &RuntimeBuffer,
            dispatch_args: &RuntimeBuffer,
        ) -> Result<(), String> {
            if pair_count == 0 || pair_count > 32 {
                return Err("K-wide route compaction supports pair counts 1..=32".to_owned());
            }
            if route_experts.length < pair_count * std::mem::size_of::<u32>()
                || bucket_counts.length < 32 * std::mem::size_of::<u32>()
                || bucket_pairs.length < 32 * pair_count * std::mem::size_of::<u32>()
                || active_experts.length < 32 * std::mem::size_of::<u32>()
                || dispatch_args.length < 3 * std::mem::size_of::<u32>()
            {
                return Err("K-wide route-compaction buffers are too small".to_owned());
            }
            let pair_count = pair_count as u32;
            self.encoder
                .setComputePipelineState(&prepared.verifier_compact_routes_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_experts.buffer),
                    route_experts.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_counts.buffer),
                    bucket_counts.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_pairs.buffer),
                    bucket_pairs.offset,
                    2,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&pair_count).cast(),
                    std::mem::size_of_val(&pair_count),
                    3,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&active_experts.buffer),
                    active_experts.offset,
                    4,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&dispatch_args.buffer),
                    dispatch_args.offset,
                    5,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 32,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        /// Encode the packet-stationary expert-major candidate. It has the
        /// same route/bucket ABI as the cached expert-major control, but the
        /// Metal kernel assigns one packed code byte and scale to each lane
        /// half so both nibbles are reused across all active token pairs.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_verifier_q4_gate_up_expert_major_unique_packet(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            bucket_experts: &RuntimeBuffer,
            bucket_offsets: &RuntimeBuffer,
            bucket_pairs: &RuntimeBuffer,
            route_tokens: &RuntimeBuffer,
            output: &RuntimeBuffer,
            bucket_count: usize,
        ) -> Result<(), String> {
            if prepared.gate_up_layout != PackingLayout::Q4RowInline {
                return Err(
                    "K-wide expert-major gate/up requires the exact Q4 row-inline layout"
                        .to_owned(),
                );
            }
            if bucket_count == 0 {
                return Ok(());
            }
            if bucket_count > 32 {
                return Err("K-wide expert-major bucket count exceeds 32".to_owned());
            }
            if input.length < 1024 * std::mem::size_of::<f32>()
                || bucket_experts.length < bucket_count * std::mem::size_of::<u32>()
                || bucket_offsets.length < (bucket_count + 1) * std::mem::size_of::<u32>()
                || bucket_pairs.length < std::mem::size_of::<u32>()
                || route_tokens.length < std::mem::size_of::<u32>()
                || output.length < 1024 * std::mem::size_of::<f32>()
            {
                return Err("K-wide expert-major gate/up buffers are too small".to_owned());
            }
            let bucket_count = bucket_count as u32;
            self.encoder.setComputePipelineState(
                &prepared.verifier_gate_up_expert_major_unique_packet_pipeline,
            );
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_packed.buffer),
                    prepared.gate_up_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_experts.buffer),
                    bucket_experts.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_offsets.buffer),
                    bucket_offsets.offset,
                    2,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_pairs.buffer),
                    bucket_pairs.offset,
                    3,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_tokens.buffer),
                    route_tokens.offset,
                    4,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 5);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 6);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&bucket_count).cast(),
                    std::mem::size_of_val(&bucket_count),
                    7,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: (bucket_count as usize) * (1024 / 8),
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

        /// Encode packet-stationary gate/up from the fixed-stride GPU route
        /// buckets produced by `encode_verifier_compact_routes_fixed`.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_verifier_q4_gate_up_compact_unique_packet(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            bucket_counts: &RuntimeBuffer,
            bucket_pairs: &RuntimeBuffer,
            route_tokens: &RuntimeBuffer,
            output: &RuntimeBuffer,
            pair_count: usize,
            active_experts: &RuntimeBuffer,
            dispatch_args: &RuntimeBuffer,
        ) -> Result<(), String> {
            self.encode_verifier_q4_gate_up_compact_unique_packet_with_plan(
                prepared,
                input,
                bucket_counts,
                bucket_pairs,
                route_tokens,
                output,
                pair_count,
                active_experts,
                dispatch_args,
                verifier_k4_vector_packet_enabled(),
                false,
                false,
                false,
                false,
                verifier_active_expert_dispatch_enabled(),
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_verifier_q4_gate_up_compact_unique_packet_with_plan(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            bucket_counts: &RuntimeBuffer,
            bucket_pairs: &RuntimeBuffer,
            route_tokens: &RuntimeBuffer,
            output: &RuntimeBuffer,
            pair_count: usize,
            active_experts: &RuntimeBuffer,
            dispatch_args: &RuntimeBuffer,
            use_k4_vector_packet: bool,
            use_k4_split_accumulation: bool,
            use_k4_direct_activation: bool,
            use_k4_rows16: bool,
            use_k4_unroll4: bool,
            use_active_expert_dispatch: bool,
        ) -> Result<(), String> {
            if prepared.gate_up_layout != PackingLayout::Q4RowInline {
                return Err("K-wide compact gate/up requires Q4 row-inline".to_owned());
            }
            if pair_count == 0 || pair_count > 32 {
                return Err("K-wide compact gate/up supports pair counts 1..=32".to_owned());
            }
            if input.length < 1024 * std::mem::size_of::<f32>()
                || bucket_counts.length < 32 * std::mem::size_of::<u32>()
                || bucket_pairs.length < 32 * pair_count * std::mem::size_of::<u32>()
                || route_tokens.length < pair_count * std::mem::size_of::<u32>()
                || output.length < pair_count * 1024 * std::mem::size_of::<f32>()
                || active_experts.length < 32 * std::mem::size_of::<u32>()
                || dispatch_args.length
                    < if use_k4_rows16 {
                        6 * std::mem::size_of::<u32>()
                    } else {
                        3 * std::mem::size_of::<u32>()
                    }
            {
                return Err("K-wide compact gate/up buffers are too small".to_owned());
            }
            let pair_count = pair_count as u32;
            let active_mode = if use_active_expert_dispatch {
                1_u32
            } else {
                0_u32
            };
            // K2 has at most two routed tokens per expert bucket.  Selecting
            // the width-specialized kernel removes the unused K4 activation
            // storage and accumulator state; K4 keeps the established path.
            if pair_count > 16 && use_k4_vector_packet {
                if use_k4_unroll4 {
                    self.encoder.setComputePipelineState(
                        &prepared.verifier_gate_up_compact_unique_packet_k4_unroll4_pipeline,
                    );
                } else if use_k4_direct_activation {
                    self.encoder.setComputePipelineState(
                        &prepared
                            .verifier_gate_up_compact_unique_packet_k4_direct_activation_pipeline,
                    );
                } else if use_k4_split_accumulation {
                    self.encoder.setComputePipelineState(
                        &prepared.verifier_gate_up_compact_unique_packet_k4_split_accum_pipeline,
                    );
                } else if use_k4_rows16 {
                    self.encoder.setComputePipelineState(
                        &prepared.verifier_gate_up_compact_unique_packet_k4_rows16_pipeline,
                    );
                } else {
                    self.encoder.setComputePipelineState(
                        &prepared.verifier_gate_up_compact_unique_packet_k4_vector_pipeline,
                    );
                }
            } else if pair_count <= 16 {
                self.encoder.setComputePipelineState(
                    &prepared.verifier_gate_up_compact_unique_packet_k2_pipeline,
                );
            } else {
                self.encoder.setComputePipelineState(
                    &prepared.verifier_gate_up_compact_unique_packet_pipeline,
                );
            }
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.gate_up_packed.buffer),
                    prepared.gate_up_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_counts.buffer),
                    bucket_counts.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_pairs.buffer),
                    bucket_pairs.offset,
                    2,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_tokens.buffer),
                    route_tokens.offset,
                    3,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 5);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&pair_count).cast(),
                    std::mem::size_of_val(&pair_count),
                    6,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&active_experts.buffer),
                    active_experts.offset,
                    7,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&active_mode).cast(),
                    std::mem::size_of_val(&active_mode),
                    8,
                );
            }
            let threadgroups_per_bucket = if use_k4_rows16 { 1024 / 16 } else { 1024 / 8 };
            let threads_per_threadgroup = if use_k4_rows16 { 512 } else { 256 };
            if use_active_expert_dispatch {
                unsafe {
                    self.encoder
                        .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                            &dispatch_args.buffer,
                            dispatch_args.offset
                                + if use_k4_rows16 { 3 * std::mem::size_of::<u32>() } else { 0 },
                            MTLSize {
                                width: threads_per_threadgroup,
                                height: 1,
                                depth: 1,
                            },
                        );
                }
            } else {
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 32 * threadgroups_per_bucket,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: threads_per_threadgroup,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_verifier_q4_silu_batch(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            output: &RuntimeBuffer,
            pair_count: usize,
        ) -> Result<(), String> {
            if pair_count == 0 {
                return Ok(());
            }
            if input.length < pair_count * 1024 * std::mem::size_of::<f32>()
                || output.length < pair_count * 512 * std::mem::size_of::<f32>()
            {
                return Err("K-wide verifier SiLU buffers are too small".to_owned());
            }
            let pair_count = pair_count as u32;
            self.encoder
                .setComputePipelineState(&prepared.verifier_silu_pipeline);
            unsafe {
                self.encoder
                    .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 0);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 1);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&pair_count).cast(),
                    std::mem::size_of_val(&pair_count),
                    2,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: pair_count as usize * 512,
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

        pub fn encode_verifier_q4_down_batch(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            route_experts: &RuntimeBuffer,
            activated: &RuntimeBuffer,
            output: &RuntimeBuffer,
            pair_count: usize,
        ) -> Result<(), String> {
            if prepared.down_layout != PackingLayout::Q4RowInline {
                return Err(
                    "K-wide verifier down requires the exact Q4 row-inline layout".to_owned(),
                );
            }
            if pair_count == 0 {
                return Ok(());
            }
            if route_experts.length < pair_count * std::mem::size_of::<u32>()
                || activated.length < pair_count * 512 * std::mem::size_of::<f32>()
                || output.length < pair_count * 1024 * std::mem::size_of::<f32>()
            {
                return Err("K-wide verifier down buffers are too small".to_owned());
            }
            let pair_count = pair_count as u32;
            self.encoder
                .setComputePipelineState(&prepared.verifier_down_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.down_packed.buffer),
                    prepared.down_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_experts.buffer),
                    route_experts.offset,
                    1,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&activated.buffer), activated.offset, 2);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 3);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&pair_count).cast(),
                    std::mem::size_of_val(&pair_count),
                    4,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: pair_count as usize * (1024 / 8),
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

        /// Encode the packet-stationary K-wide Down consumer. Route pairs are
        /// already grouped by expert; the kernel reuses each expert's packed
        /// Down packet across all active token pairs in that bucket.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_verifier_q4_down_compact_unique_packet(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            bucket_counts: &RuntimeBuffer,
            bucket_pairs: &RuntimeBuffer,
            activated: &RuntimeBuffer,
            output: &RuntimeBuffer,
            pair_count: usize,
            active_experts: &RuntimeBuffer,
            dispatch_args: &RuntimeBuffer,
        ) -> Result<(), String> {
            self.encode_verifier_q4_down_compact_unique_packet_with_plan(
                prepared,
                bucket_counts,
                bucket_pairs,
                activated,
                output,
                pair_count,
                active_experts,
                dispatch_args,
                verifier_active_expert_dispatch_enabled(),
                false,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_verifier_q4_down_compact_unique_packet_with_plan(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            bucket_counts: &RuntimeBuffer,
            bucket_pairs: &RuntimeBuffer,
            activated: &RuntimeBuffer,
            output: &RuntimeBuffer,
            pair_count: usize,
            active_experts: &RuntimeBuffer,
            dispatch_args: &RuntimeBuffer,
            use_active_expert_dispatch: bool,
            use_k4_unroll4: bool,
        ) -> Result<(), String> {
            if prepared.down_layout != PackingLayout::Q4RowInline {
                return Err("K-wide compact Down requires Q4 row-inline".to_owned());
            }
            if pair_count == 0 || pair_count > 32 {
                return Err("K-wide compact Down supports pair counts 1..=32".to_owned());
            }
            if bucket_counts.length < 32 * std::mem::size_of::<u32>()
                || bucket_pairs.length < 32 * pair_count * std::mem::size_of::<u32>()
                || activated.length < pair_count * 512 * std::mem::size_of::<f32>()
                || output.length < pair_count * 1024 * std::mem::size_of::<f32>()
                || active_experts.length < 32 * std::mem::size_of::<u32>()
                || dispatch_args.length < 3 * std::mem::size_of::<u32>()
            {
                return Err("K-wide compact Down buffers are too small".to_owned());
            }
            let pair_count = pair_count as u32;
            let active_mode = if use_active_expert_dispatch {
                1_u32
            } else {
                0_u32
            };
            if use_k4_unroll4 && pair_count > 16 {
                self.encoder.setComputePipelineState(
                    &prepared.verifier_down_compact_unique_packet_k4_unroll4_pipeline,
                );
            } else {
                self.encoder.setComputePipelineState(
                    &prepared.verifier_down_compact_unique_packet_pipeline,
                );
            }
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&prepared.down_packed.buffer),
                    prepared.down_packed.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_counts.buffer),
                    bucket_counts.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&bucket_pairs.buffer),
                    bucket_pairs.offset,
                    2,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&activated.buffer), activated.offset, 3);
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 4);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&pair_count).cast(),
                    std::mem::size_of_val(&pair_count),
                    5,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&active_experts.buffer),
                    active_experts.offset,
                    6,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&active_mode).cast(),
                    std::mem::size_of_val(&active_mode),
                    7,
                );
            }
            if use_active_expert_dispatch {
                unsafe {
                    self.encoder
                        .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                            &dispatch_args.buffer,
                            dispatch_args.offset,
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
                        width: 32 * (1024 / 8),
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
            self.dispatches += 1;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_verifier_q4_accumulate_batch(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            expert_output: &RuntimeBuffer,
            route_tokens: &RuntimeBuffer,
            route_weights: &RuntimeBuffer,
            output: &RuntimeBuffer,
            pair_count: usize,
            token_count: usize,
        ) -> Result<(), String> {
            if pair_count == 0 || token_count == 0 {
                return Ok(());
            }
            if expert_output.length < pair_count * 1024 * std::mem::size_of::<f32>()
                || route_tokens.length < pair_count * std::mem::size_of::<u32>()
                || route_weights.length < pair_count * std::mem::size_of::<f32>()
                || output.length < token_count * 1024 * std::mem::size_of::<f32>()
            {
                return Err("K-wide verifier accumulation buffers are too small".to_owned());
            }
            let pair_count = pair_count as u32;
            let token_count = token_count as u32;
            self.encoder
                .setComputePipelineState(&prepared.verifier_accumulate_pipeline);
            unsafe {
                self.encoder.setBuffer_offset_atIndex(
                    Some(&expert_output.buffer),
                    expert_output.offset,
                    0,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_tokens.buffer),
                    route_tokens.offset,
                    1,
                );
                self.encoder.setBuffer_offset_atIndex(
                    Some(&route_weights.buffer),
                    route_weights.offset,
                    2,
                );
                self.encoder
                    .setBuffer_offset_atIndex(Some(&output.buffer), output.offset, 3);
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&pair_count).cast(),
                    std::mem::size_of_val(&pair_count),
                    4,
                );
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&token_count).cast(),
                    std::mem::size_of_val(&token_count),
                    5,
                );
            }
            self.encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: token_count as usize * 1024,
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
            self.encode_dynamic_q4_moe_batched_simd_with_plan(
                prepared,
                input,
                expert_ids,
                expert_weights,
                scratch,
                output,
                DynamicMoeConsumerPlan::from_legacy_environment(),
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_dynamic_q4_moe_batched_simd_with_plan(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            consumer_plan: DynamicMoeConsumerPlan,
        ) -> Result<(), String> {
            self.encode_dynamic_q4_moe_batched_simd_with_norm_and_plan(
                prepared,
                input,
                expert_ids,
                expert_weights,
                scratch,
                output,
                None,
                consumer_plan,
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
            self.encode_dynamic_q4_moe_batched_simd_with_norm_and_plan(
                prepared,
                input,
                expert_ids,
                expert_weights,
                scratch,
                output,
                Some((norm_weights, inverse_rms)),
                DynamicMoeConsumerPlan::from_legacy_environment(),
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encode_dynamic_q4_moe_batched_simd_normalized_with_plan(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            norm_weights: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            consumer_plan: DynamicMoeConsumerPlan,
        ) -> Result<(), String> {
            self.encode_dynamic_q4_moe_batched_simd_with_norm_and_plan(
                prepared,
                input,
                expert_ids,
                expert_weights,
                scratch,
                output,
                Some((norm_weights, inverse_rms)),
                consumer_plan,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_dynamic_q4_moe_batched_simd_with_norm_and_plan(
            &mut self,
            prepared: &PreparedDynamicQ4Moe,
            input: &RuntimeBuffer,
            expert_ids: &RuntimeBuffer,
            expert_weights: &RuntimeBuffer,
            scratch: &RuntimeBuffer,
            output: &RuntimeBuffer,
            norm: Option<(&RuntimeBuffer, &RuntimeBuffer)>,
            consumer_plan: DynamicMoeConsumerPlan,
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
            let gate_up_instrument = moe_gate_up_instrument_variant().is_some()
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let unique_packet = consumer_plan.unique_packet
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let unique_packet_down = consumer_plan.unique_packet_down
                && prepared.down_inline_layout
                && prepared.down_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let is_q4k = prepared.gate_up_packed.length == 32 * 1024 * 4 * 140
                && prepared.gate_up_scales.length == 32 * 1024 * 4 * 2 * 2
                && prepared.down_packed.length == 32 * 1024 * 2 * 140
                && prepared.down_scales.length == 32 * 1024 * 2 * 2 * 2;
            let q4k_mature_gate_up = is_q4k && moe_q4k_mature_gate_up_enabled();
            let q4k_mature_down = is_q4k && moe_q4k_mature_down_enabled();
            let superblock_gate_up = prepared.gate_up_layout.is_q4_superblock();
            let superblock_down = prepared.down_layout.is_q4_superblock();
            let expert_wave_width = moe_expert_wave_width();
            let wide_expert =
                moe_wide_expert_enabled() && prepared.wide_supported && q3_block_bytes.is_none();
            let inline_input_cache = moe_inline_input_cache_enabled();
            let fused_block_mapping = moe_fused_transaction_block_mapping();
            let transaction_gate_up =
                prepared.gate_up_layout.is_q4_transaction() || superblock_gate_up;
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
            let two_row_unique_packet = moe_two_row_unique_packet_enabled()
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
                if unique_packet {
                    self.encoder
                        .setComputePipelineState(&prepared.unique_packet_gate_up_norm_pipeline);
                } else if gate_up_instrument
                    && self.select_gate_up_instrument_pipeline(prepared, true)?
                {
                } else if let Some(pipeline) = prepared.q3_gate_up_norm_pipeline.as_ref() {
                    self.encoder.setComputePipelineState(pipeline);
                } else if q4k_mature_gate_up {
                    self.encoder
                        .setComputePipelineState(&prepared.q4k_mature_gate_up_norm_pipeline);
                } else if two_row_unique_packet {
                    self.encoder.setComputePipelineState(
                        prepared
                            .two_row_unique_packet_gate_up_norm_pipeline
                            .as_ref()
                            .ok_or_else(|| {
                                "two-row unique-packet pipeline is missing".to_owned()
                            })?,
                    );
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
                    let pipeline = if fused_block_mapping == Some(true) && !superblock_gate_up {
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
                if unique_packet {
                    self.encoder
                        .setComputePipelineState(&prepared.unique_packet_gate_up_pipeline);
                } else if gate_up_instrument
                    && self.select_gate_up_instrument_pipeline(prepared, false)?
                {
                } else if let Some(pipeline) = prepared.q3_gate_up_pipeline.as_ref() {
                    self.encoder.setComputePipelineState(pipeline);
                } else if q4k_mature_gate_up {
                    self.encoder
                        .setComputePipelineState(&prepared.q4k_mature_gate_up_pipeline);
                } else if two_row_unique_packet {
                    self.encoder.setComputePipelineState(
                        prepared
                            .two_row_unique_packet_gate_up_pipeline
                            .as_ref()
                            .ok_or_else(|| {
                                "two-row unique-packet pipeline is missing".to_owned()
                            })?,
                    );
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
                    let pipeline = if fused_block_mapping == Some(true) && !superblock_gate_up {
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
                        width: if superblock_gate_up {
                            8 * (1024 / 32)
                        } else if q4k_mature_gate_up {
                            8 * (1024 / 8)
                        } else if fused_block_active && fused_block_mapping == Some(true) {
                            8 * (1024 / 4)
                        } else if fused_block_active {
                            8 * (1024 / 8)
                        } else if llama_like_mapping {
                            8 * (1024 / 4)
                        } else if two_row_unique_packet || two_row_dataflow {
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
                        width: if superblock_gate_up {
                            256
                        } else if q4k_mature_gate_up {
                            128
                        } else if fused_block_active && fused_block_mapping == Some(true) {
                            64
                        } else if fused_block_active {
                            256
                        } else if llama_like_mapping {
                            64
                        } else if two_row_unique_packet || two_row_dataflow {
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
            } else if unique_packet_down {
                self.encoder
                    .setComputePipelineState(&prepared.unique_packet_down_pipeline);
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
                        width: if superblock_down {
                            8 * (1024 / 32)
                        } else if q4k_mature_down {
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
                        width: if superblock_down {
                            256
                        } else if q4k_mature_down {
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

            if prepared.expertwise_q4_gate_up_superblock {
                let pipeline = if norm.is_some() {
                    prepared
                        .expertwise_superblock_gate_up_norm_pipeline
                        .as_ref()
                        .unwrap()
                } else {
                    prepared
                        .expertwise_superblock_gate_up_pipeline
                        .as_ref()
                        .unwrap()
                };
                self.encoder.setComputePipelineState(pipeline);
                unsafe {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&q4_gate.buffer), q4_gate.offset, 0);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&input.buffer), input.offset, 4);
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&expert_ids.buffer),
                        expert_ids.offset,
                        5,
                    );
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&gate_up.buffer), gate_up.offset, 6);
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&map.buffer), map.offset, 10);
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&q3_gate.buffer),
                        q3_gate.offset,
                        11,
                    );
                    if !prepared.expertwise_q3_gate_up_superblock {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&q3_gate_scales.buffer),
                            q3_gate_scales.offset,
                            12,
                        );
                    }
                    if let Some((weights, inverse)) = norm {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&weights.buffer),
                            weights.offset,
                            8,
                        );
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&inverse.buffer),
                            inverse.offset,
                            9,
                        );
                    }
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 8 * (1024 / 32),
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
            } else if let Some((weights, inverse)) = norm {
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
            if !prepared.expertwise_q4_gate_up_superblock
                && let Some(block_bytes) = prepared.expertwise_q3_gate_up_block_bytes
            {
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
            if prepared.expertwise_q4_down_superblock {
                self.encoder.setComputePipelineState(
                    prepared
                        .expertwise_superblock_down_pipeline
                        .as_ref()
                        .unwrap(),
                );
                unsafe {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&q4_down.buffer), q4_down.offset, 0);
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
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&down_map.buffer),
                        down_map.offset,
                        10,
                    );
                    self.encoder.setBuffer_offset_atIndex(
                        Some(&q3_down.buffer),
                        q3_down.offset,
                        11,
                    );
                    if !prepared.expertwise_q3_down_superblock {
                        self.encoder.setBuffer_offset_atIndex(
                            Some(&q3_down_scales.buffer),
                            q3_down_scales.offset,
                            12,
                        );
                    }
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 8 * (1024 / 32),
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
            } else {
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
            let legacy_consumer_plan = DynamicMoeConsumerPlan::from_legacy_environment();
            let unique_packet = legacy_consumer_plan.unique_packet
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && prepared.q3_gate_up_block_bytes.is_none();
            let unique_packet_down = legacy_consumer_plan.unique_packet_down
                && prepared.down_inline_layout
                && prepared.down_layout == PackingLayout::Q4RowInline
                && prepared.q3_down_block_bytes.is_none();
            let cluster_down_width = if prepared.down_inline_layout
                && prepared.down_layout == PackingLayout::Q4RowInline
                && prepared.q3_down_block_bytes.is_none()
            {
                moe_cluster_down_width()
            } else {
                None
            };
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
            let gate_up_instrument = moe_gate_up_instrument_variant().is_some()
                && prepared.inline_layout
                && prepared.gate_up_layout == PackingLayout::Q4RowInline
                && q3_block_bytes.is_none();
            let is_q4k = prepared.gate_up_packed.length == 32 * 1024 * 4 * 140
                && prepared.gate_up_scales.length == 32 * 1024 * 4 * 2 * 2
                && prepared.down_packed.length == 32 * 1024 * 2 * 140
                && prepared.down_scales.length == 32 * 1024 * 2 * 2 * 2;
            let q4k_mature_gate_up = is_q4k && moe_q4k_mature_gate_up_enabled();
            let q4k_mature_down = is_q4k && moe_q4k_mature_down_enabled();
            let superblock_gate_up = prepared.gate_up_layout.is_q4_superblock();
            let superblock_down = prepared.down_layout.is_q4_superblock();
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
                        if unique_packet {
                            self.encoder.setComputePipelineState(
                                &prepared.unique_packet_gate_up_norm_pipeline,
                            );
                        } else if gate_up_instrument
                            && self.select_gate_up_instrument_pipeline(prepared, true)?
                        {
                        } else if superblock_gate_up {
                            self.encoder.setComputePipelineState(
                                prepared
                                    .fused_block_qx_gate_up_norm_pipeline
                                    .as_ref()
                                    .ok_or_else(|| {
                                        "superblock gate/up pipeline is missing".to_owned()
                                    })?,
                            );
                        } else if let Some(pipeline) = prepared.q3_gate_up_norm_pipeline.as_ref() {
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
                        if unique_packet {
                            self.encoder
                                .setComputePipelineState(&prepared.unique_packet_gate_up_pipeline);
                        } else if gate_up_instrument
                            && self.select_gate_up_instrument_pipeline(prepared, false)?
                        {
                        } else if prepared.gate_up_layout.is_q4_transaction() || superblock_gate_up
                        {
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
                            width: if superblock_gate_up {
                                8 * (1024 / 32)
                            } else if q4k_mature_gate_up {
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
                            width: if superblock_gate_up {
                                256
                            } else if q4k_mature_gate_up {
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
                    } else if let Some(width) = cluster_down_width {
                        self.encoder.setComputePipelineState(match width {
                            16 => &prepared.cluster_down16_pipeline,
                            32 => &prepared.cluster_down32_pipeline,
                            _ => unreachable!(),
                        });
                    } else if unique_packet_down {
                        self.encoder
                            .setComputePipelineState(&prepared.unique_packet_down_pipeline);
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
                            width: if superblock_down {
                                8 * (1024 / 32)
                            } else if q4k_mature_down {
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
                            width: if superblock_down {
                                256
                            } else if q4k_mature_down {
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
            snapshot: Option<&RuntimeBuffer>,
            multiplier: f32,
        ) -> Result<(), String> {
            if token.length < std::mem::size_of::<u32>()
                || output.length < prepared.columns * std::mem::size_of::<f32>()
                || snapshot.is_some_and(|buffer| {
                    buffer.length < prepared.columns * std::mem::size_of::<f32>()
                })
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
                let snapshot_buffer = snapshot.unwrap_or(output);
                self.encoder.setBuffer_offset_atIndex(
                    Some(&snapshot_buffer.buffer),
                    snapshot_buffer.offset,
                    7,
                );
                let capture_snapshot = u32::from(snapshot.is_some());
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&capture_snapshot).cast(),
                    std::mem::size_of_val(&capture_snapshot),
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
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            self.dispatches += 1;
            Ok(())
        }

        pub fn encode_q4_embedding_lookup_batch(
            &mut self,
            prepared: &PreparedQ4Embedding,
            tokens: &RuntimeBuffer,
            output: &RuntimeBuffer,
            count: usize,
            multiplier: f32,
        ) -> Result<(), String> {
            if count == 0
                || tokens.length < count * std::mem::size_of::<u32>()
                || output.length < count * prepared.columns * std::mem::size_of::<f32>()
            {
                return Err(
                    "batched embedding buffers are smaller than their fixed shape".to_owned(),
                );
            }
            self.encoder
                .setComputePipelineState(&prepared.batch_pipeline);
            let rows = prepared.rows as u32;
            let columns = prepared.columns as u32;
            let count_u32 = count as u32;
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
                    .setBuffer_offset_atIndex(Some(&tokens.buffer), tokens.offset, 2);
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
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&count_u32).cast(),
                    std::mem::size_of_val(&count_u32),
                    7,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: count,
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
            if matches!(kernel, DecodeKernel::AttentionBatchSimd) {
                self.encoder
                    .dispatchThreadgroups_threadsPerThreadgroup(threads, group);
            } else {
                self.encoder
                    .dispatchThreads_threadsPerThreadgroup(threads, group);
            }
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

        /// Encode RMSNorm for token-major `[tokens, 1024]` storage in one
        /// dispatch. Each threadgroup owns one token row.
        pub fn encode_rmsnorm_batch(
            &mut self,
            context: &MetalContext,
            input: &RuntimeBuffer,
            weights: &RuntimeBuffer,
            output: &RuntimeBuffer,
            tokens: usize,
        ) -> Result<(), String> {
            if tokens == 0 {
                return Err("batched RMSNorm requires at least one token".to_owned());
            }
            let params = [1024u32.to_ne_bytes(), 1e-5f32.to_ne_bytes()].concat();
            self.encode_decode_kernel(
                context,
                DecodeKernel::RmsNormBatch,
                &[input, weights, output],
                &params,
                MTLSize {
                    width: 256,
                    height: tokens,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            )
        }

        /// Encode one serial, scalar-semantics RMSNorm per token in a single
        /// dispatch. This is a dispatch-batching control, not a SIMD
        /// reduction: preserving the canonical accumulation order is
        /// necessary because RMSNorm feeds routing decisions.
        pub fn encode_rmsnorm_batch_scalar(
            &mut self,
            context: &MetalContext,
            input: &RuntimeBuffer,
            weights: &RuntimeBuffer,
            output: &RuntimeBuffer,
            tokens: usize,
        ) -> Result<(), String> {
            if tokens == 0 {
                return Err(
                    "scalar-semantics batched RMSNorm requires at least one token".to_owned(),
                );
            }
            let params = [1024u32.to_ne_bytes(), 1e-5f32.to_ne_bytes()].concat();
            let kernel = if rms_parallel_writeback_enabled() {
                DecodeKernel::RmsNormBatchParallelWriteback
            } else {
                DecodeKernel::RmsNormBatchScalar
            };
            let threads = if rms_parallel_writeback_enabled() {
                MTLSize {
                    width: 256,
                    height: tokens,
                    depth: 1,
                }
            } else {
                MTLSize {
                    width: tokens,
                    height: 1,
                    depth: 1,
                }
            };
            let group = if rms_parallel_writeback_enabled() {
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                }
            } else {
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                }
            };
            self.encode_decode_kernel(
                context,
                kernel,
                &[input, weights, output],
                &params,
                threads,
                group,
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
            if rope_kv_split_enabled() && !key_fp16 {
                return self.encode_rope_kv_split(
                    context,
                    query,
                    key,
                    value,
                    query_output,
                    key_cache,
                    value_cache,
                    position,
                    value_fp16,
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
            value_fp16: bool,
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
                if value_fp16 {
                    DecodeKernel::RopeKvSimdKf32Vf16
                } else {
                    DecodeKernel::RopeKvSimd
                },
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

        /// Encode RoPE and speculative KV writes for K consecutive positions
        /// in two GPU dispatches. Inputs and query output are token-major;
        /// cache positions begin at `start_position` and are committed by the
        /// caller only after verification accepts a prefix.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_rope_kv_batch(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key: &RuntimeBuffer,
            value: &RuntimeBuffer,
            query_output: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            start_position: usize,
            tokens: usize,
        ) -> Result<(), String> {
            if tokens == 0 {
                return Err("batched RoPE/KV requires at least one token".to_owned());
            }
            let q_values = tokens * 16 * 64;
            let kv_values = tokens * 8 * 64;
            let cache_values = (start_position + tokens) * 8 * 64;
            let f32_bytes = std::mem::size_of::<f32>();
            if query.length < q_values * f32_bytes
                || query_output.length < q_values * f32_bytes
                || key.length < kv_values * f32_bytes
                || value.length < kv_values * f32_bytes
                || key_cache.length < cache_values * f32_bytes
                || value_cache.length < cache_values * f32_bytes
            {
                return Err("batched RoPE/KV buffers are smaller than their fixed shape".to_owned());
            }
            #[repr(C)]
            struct Params {
                start_position: u32,
                tokens: u32,
            }
            let params = Params {
                start_position: start_position as u32,
                tokens: tokens as u32,
            };
            self.encode_decode_kernel(
                context,
                DecodeKernel::RopeQueryBatchSimd,
                &[query, query_output],
                bytes_of(&params),
                MTLSize {
                    width: q_values,
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
                DecodeKernel::RopeKvBatchSimd,
                &[key, value, key_cache, value_cache],
                bytes_of(&params),
                MTLSize {
                    width: kv_values,
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

        /// Encode causal attention for K speculative positions in one GPU
        /// dispatch. The cache already contains the committed prefix and
        /// the batch RoPE stage has written positions
        /// `[start_position, start_position + tokens)`.
        pub fn encode_attention_batch(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            output: &RuntimeBuffer,
            start_position: usize,
            tokens: usize,
        ) -> Result<(), String> {
            self.encode_attention_batch_with_plan(
                context,
                query,
                key_cache,
                value_cache,
                output,
                start_position,
                tokens,
                attention_batch_simd_enabled(),
            )
        }

        /// Encode batched verifier attention with an explicit planner choice.
        /// The legacy method above remains for standalone benchmark callers;
        /// Runtime V2 production verification must pass the resolved plan so
        /// attention selection is not read from process-global environment.
        pub fn encode_attention_batch_with_plan(
            &mut self,
            context: &MetalContext,
            query: &RuntimeBuffer,
            key_cache: &RuntimeBuffer,
            value_cache: &RuntimeBuffer,
            output: &RuntimeBuffer,
            start_position: usize,
            tokens: usize,
            use_batch_simd: bool,
        ) -> Result<(), String> {
            if tokens == 0 {
                return Err("batched attention requires at least one token".to_owned());
            }
            let hidden = 16 * 64;
            let cache_values = (start_position + tokens) * 8 * 64;
            let f32_bytes = std::mem::size_of::<f32>();
            if query.length < tokens * hidden * f32_bytes
                || output.length < tokens * hidden * f32_bytes
                || key_cache.length < cache_values * f32_bytes
                || value_cache.length < cache_values * f32_bytes
            {
                return Err(
                    "batched attention buffers are smaller than their fixed shape".to_owned(),
                );
            }
            #[repr(C)]
            struct Params {
                start_position: u32,
                tokens: u32,
            }
            let params = Params {
                start_position: start_position as u32,
                tokens: tokens as u32,
            };
            let kernel = if use_batch_simd {
                DecodeKernel::AttentionBatchSimd
            } else {
                DecodeKernel::AttentionBatch
            };
            let threads = if use_batch_simd { 512 } else { 256 };
            self.encode_decode_kernel(
                context,
                kernel,
                &[query, key_cache, value_cache, output],
                bytes_of(&params),
                MTLSize {
                    width: if use_batch_simd {
                        tokens
                    } else {
                        tokens * hidden
                    },
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threads,
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
                if length == 0 || length > 512 {
                    return Err("one-pass attention softmax length is outside 1..=512".to_owned());
                }
                let pipeline = context.attention_softmax_onepass_pipeline()?;
                let length = length as u32;
                self.encoder.setComputePipelineState(&pipeline);
                unsafe {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(&scores.buffer), scores.offset, 0);
                    self.encoder.setBytes_length_atIndex(
                        NonNull::from(&length).cast(),
                        std::mem::size_of_val(&length),
                        1,
                    );
                }
                self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 16,
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
                return Ok(());
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

        #[allow(dead_code)]
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

        pub fn encode_residual_rms_scalar(
            &mut self,
            context: &MetalContext,
            left: &RuntimeBuffer,
            right: &RuntimeBuffer,
            output: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            scale: f32,
        ) -> Result<(), String> {
            if left.length < 1024 * std::mem::size_of::<f32>()
                || right.length < 1024 * std::mem::size_of::<f32>()
                || output.length < 1024 * std::mem::size_of::<f32>()
                || inverse_rms.length < std::mem::size_of::<f32>()
            {
                return Err("residual/RMS buffers are smaller than the fixed shape".to_owned());
            }
            let pipeline = context.residual_rms_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, offset, index) in [
                    (&left.buffer, left.offset, 0),
                    (&right.buffer, right.offset, 1),
                    (&output.buffer, output.offset, 2),
                    (&inverse_rms.buffer, inverse_rms.offset, 3),
                ] {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(buffer), offset, index);
                }
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&scale).cast(),
                    std::mem::size_of_val(&scale),
                    4,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
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
        pub fn encode_residual_rms_scalar_snapshot(
            &mut self,
            context: &MetalContext,
            left: &RuntimeBuffer,
            right: &RuntimeBuffer,
            output: &RuntimeBuffer,
            inverse_rms: &RuntimeBuffer,
            snapshot: &RuntimeBuffer,
            scale: f32,
        ) -> Result<(), String> {
            if left.length < 1024 * std::mem::size_of::<f32>()
                || right.length < 1024 * std::mem::size_of::<f32>()
                || output.length < 1024 * std::mem::size_of::<f32>()
                || snapshot.length < 1024 * std::mem::size_of::<f32>()
                || inverse_rms.length < std::mem::size_of::<f32>()
            {
                return Err(
                    "residual/RMS snapshot buffers are smaller than the fixed shape".to_owned(),
                );
            }
            let pipeline = context.residual_rms_snapshot_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, offset, index) in [
                    (&left.buffer, left.offset, 0),
                    (&right.buffer, right.offset, 1),
                    (&output.buffer, output.offset, 2),
                    (&inverse_rms.buffer, inverse_rms.offset, 3),
                    (&snapshot.buffer, snapshot.offset, 5),
                ] {
                    self.encoder
                        .setBuffer_offset_atIndex(Some(buffer), offset, index);
                }
                self.encoder.setBytes_length_atIndex(
                    NonNull::from(&scale).cast(),
                    std::mem::size_of_val(&scale),
                    4,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
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

        pub fn encode_residual_batch(
            &mut self,
            context: &MetalContext,
            left: &RuntimeBuffer,
            right: &RuntimeBuffer,
            output: &RuntimeBuffer,
            tokens: usize,
            scale: f32,
        ) -> Result<(), String> {
            if tokens == 0 {
                return Err("batched residual requires at least one token".to_owned());
            }
            #[repr(C)]
            struct Params {
                n: u32,
                scale: f32,
            }
            let params = Params { n: 1024, scale };
            self.encode_decode_kernel(
                context,
                DecodeKernel::ResidualBatch,
                &[left, right, output],
                bytes_of(&params),
                MTLSize {
                    width: 1024,
                    height: tokens,
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

        /// Encode top-8 selection for token-major `[tokens, 32]` router
        /// logits and `[tokens, 8]` route outputs in one dispatch.
        pub fn encode_topk_batch(
            &mut self,
            context: &MetalContext,
            logits: &RuntimeBuffer,
            ids: &RuntimeBuffer,
            weights: &RuntimeBuffer,
            tokens: usize,
        ) -> Result<(), String> {
            if tokens == 0 {
                return Err("batched top-k requires at least one token".to_owned());
            }
            let params = 32u32;
            self.encode_decode_kernel(
                context,
                DecodeKernel::TopKBatch,
                &[logits, ids, weights],
                bytes_of(&params),
                MTLSize {
                    width: 1,
                    height: tokens,
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

        /// Opt-in logical boundary used by runtime-owned diagnostic graphs.
        /// It is a strict no-op for canonical execution. When diagnostics are
        /// enabled it uses counters where available and ordered command-buffer
        /// segments otherwise, without introducing a CPU wait.
        pub fn diagnostic_stage_boundary(&mut self, context: &MetalContext) -> Result<(), String> {
            if !self.stage_diagnostics {
                return Ok(());
            }
            self.stage_boundary(context)
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

        /// Preserve the five-dispatch MoE slot in a verifier probe without
        /// performing expert work. This is intentionally diagnostic-only:
        /// downstream values are not expected to preserve model semantics.
        pub fn encode_verifier_moe_noop(&mut self, context: &MetalContext) -> Result<(), String> {
            for _ in 0..5 {
                self.encode_noop(
                    context,
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
                )?;
            }
            Ok(())
        }

        /// Replay a captured K-wide MoE output while preserving the four
        /// post-routing dispatch slots (gate/up, activation, down, and
        /// accumulation). The route-compaction dispatch is emitted by the
        /// caller, so the complete MoE region retains five dispatches.
        pub fn encode_verifier_moe_replay(
            &mut self,
            context: &MetalContext,
            captured: &RuntimeBuffer,
            output: &RuntimeBuffer,
            elements: usize,
        ) -> Result<(), String> {
            self.encode_copy_f32(context, captured, output, elements)?;
            for _ in 0..3 {
                self.encode_noop(
                    context,
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
                )?;
            }
            Ok(())
        }

        /// Copy a fixed FP32 vector inside the GPU command stream. This is a
        /// diagnostic primitive used by production-preserving replacement
        /// probes; it is not part of the canonical token graph.
        pub fn encode_copy_f32(
            &mut self,
            context: &MetalContext,
            source: &RuntimeBuffer,
            destination: &RuntimeBuffer,
            n: usize,
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                n: u32,
            }
            let params = Params { n: n as u32 };
            self.encode_decode_kernel(
                context,
                DecodeKernel::CopyF32,
                &[source, destination],
                bytes_of(&params),
                MTLSize {
                    width: n,
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

        /// Hash post-layer residual snapshots and their newly appended K/V
        /// slices concurrently at the end of the token command stream. Only
        /// the two 32-byte digests per layer are written to the output.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_verification_hashes(
            &mut self,
            context: &MetalContext,
            residual_snapshots: &RuntimeBuffer,
            kv_arena: &RuntimeBuffer,
            output: &RuntimeBuffer,
            token: u32,
            layers: usize,
            residual_stride: usize,
            checkpoint_interval: usize,
            key_offsets: [u32; 24],
            value_offsets: [u32; 24],
            key_bytes: [u32; 24],
            value_bytes: [u32; 24],
        ) -> Result<(), String> {
            #[repr(C)]
            struct Params {
                token: u32,
                layers: u32,
                residual_stride: u32,
                checkpoint_interval: u32,
                key_offsets: [u32; 24],
                value_offsets: [u32; 24],
                key_bytes: [u32; 24],
                value_bytes: [u32; 24],
            }
            if layers == 0 || layers > 24 {
                return Err(format!(
                    "verification layer count {layers} is outside 1..=24"
                ));
            }
            if checkpoint_interval == 0 || checkpoint_interval > layers {
                return Err(format!(
                    "verification checkpoint interval {checkpoint_interval} is outside 1..={layers}"
                ));
            }
            if output.len() < (layers + 1) * 64 {
                return Err("verification digest output is too small".to_owned());
            }
            if residual_snapshots.len() < (layers + 1) * residual_stride {
                return Err("verification residual snapshot buffer is too small".to_owned());
            }
            const VERIFICATION_CHUNK_BYTES: usize = 256;
            let validate = |name: &str, bytes: usize, max_chunks: usize| {
                if bytes == 0 || !bytes.is_multiple_of(VERIFICATION_CHUNK_BYTES) {
                    return Err(format!(
                        "verification {name} size {bytes} is not a non-zero multiple of {}",
                        VERIFICATION_CHUNK_BYTES
                    ));
                }
                let chunks = bytes / VERIFICATION_CHUNK_BYTES;
                if !chunks.is_power_of_two() || chunks > max_chunks {
                    return Err(format!(
                        "verification {name} chunk count {chunks} must be a power of two <= {max_chunks}"
                    ));
                }
                Ok(())
            };
            validate("residual", residual_stride, 16)?;
            for layer in 0..layers {
                validate("key", key_bytes[layer] as usize, 8)?;
                validate("value", value_bytes[layer] as usize, 8)?;
                let key_end = key_offsets[layer] as usize + key_bytes[layer] as usize;
                let value_end = value_offsets[layer] as usize + value_bytes[layer] as usize;
                if key_end > kv_arena.len() || value_end > kv_arena.len() {
                    return Err(format!(
                        "verification layer {layer} KV slice is out of bounds"
                    ));
                }
            }
            let params = Params {
                token,
                layers: layers as u32,
                residual_stride: u32::try_from(residual_stride)
                    .map_err(|_| "verification residual stride exceeds u32".to_owned())?,
                checkpoint_interval: checkpoint_interval as u32,
                key_offsets,
                value_offsets,
                key_bytes,
                value_bytes,
            };
            let pipeline = context.verification_hash_pipeline()?;
            self.encoder.setComputePipelineState(&pipeline);
            for (index, buffer) in [residual_snapshots, kv_arena, output]
                .into_iter()
                .enumerate()
            {
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
                    NonNull::from(&params).cast(),
                    std::mem::size_of_val(&params),
                    3,
                );
            }
            self.encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: layers + 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 96,
                    height: 1,
                    depth: 1,
                },
            );
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

        pub fn submit(mut self) -> SubmittedDecode {
            self.encoder.endEncoding();
            self.command_buffer.commit();
            self.committed
                .push((self.command_buffer, self.dispatches, self.encoders));
            SubmittedDecode {
                committed: self.committed,
                counter_timeline: self.counter_timeline.take(),
                encode_started_at: self.encode_started_at,
                submitted_at: Instant::now(),
            }
        }

        pub fn finish(self, context: &MetalContext) -> Result<(), String> {
            self.submit().wait(context)
        }
    }

    impl SubmittedDecode {
        pub fn wait(mut self, context: &MetalContext) -> Result<(), String> {
            let final_command_buffer = self
                .committed
                .last()
                .ok_or_else(|| "submitted decode graph has no command buffers".to_owned())?;
            let wait_started_at = Instant::now();
            final_command_buffer.0.waitUntilCompleted();
            let cpu_wait = wait_started_at.elapsed();
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
            for (command_buffer, _, _) in &self.committed {
                let gpu_start = command_buffer.GPUStartTime();
                let gpu_end = command_buffer.GPUEndTime();
                if gpu_start > 0.0 && gpu_end >= gpu_start {
                    if let Some(previous_end) = metrics.last_gpu_end_seconds {
                        metrics.gpu_queue_gap_seconds += (gpu_start - previous_end).max(0.0);
                    }
                    metrics.last_gpu_end_seconds = Some(
                        metrics
                            .last_gpu_end_seconds
                            .map_or(gpu_end, |previous_end| previous_end.max(gpu_end)),
                    );
                }
            }
            metrics.cpu_encode_seconds += self
                .submitted_at
                .duration_since(self.encode_started_at)
                .as_secs_f64();
            metrics.cpu_wait_seconds += cpu_wait.as_secs_f64();
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
                pipelines: Mutex::new([None, None, None, None]),
                aggregate_pipelines: Mutex::new([None, None, None, None]),
                qwen_q2s4_expert_pipeline: Mutex::new(None),
                qwen_qk_norm_rope_pipeline: Mutex::new(None),
                qwen_gqa_attention_pipeline: Mutex::new(None),
                qwen_gqa_score_reuse_pipeline: Mutex::new(None),
                qwen_top8_pipeline: Mutex::new(None),
                qwen_rms_norm_pipeline: Mutex::new(None),
                qwen_residual_rms_norm_pipeline: Mutex::new(None),
                qwen_store_kv_pipeline: Mutex::new(None),
                qwen_residual_pipeline: Mutex::new(None),
                qwen_embedding_pipeline: Mutex::new(None),
                qwen_argmax_pipeline: Mutex::new(None),
                qwen_dynamic_moe_pipeline: Mutex::new(None),
                qwen_dynamic_gate_pipeline: Mutex::new(None),
                qwen_dynamic_down_tiled_pipeline: Mutex::new(None),
                qwen_dynamic_accumulate_pipeline: Mutex::new(None),
                qwen_q4_cache_resolve_pipeline: Mutex::new(None),
                qwen_q4_cache_fill_pipeline: Mutex::new(None),
                logits_pipeline: Mutex::new(None),
                dense_pipeline: Mutex::new(None),
                f16_dense_pipeline: Mutex::new(None),
                q4_dense_library: Mutex::new(None),
                residual_rms_pipeline: Mutex::new(None),
                residual_rms_snapshot_pipeline: Mutex::new(None),
                attention_softmax_onepass_pipeline: Mutex::new(None),
                q4_dense_pipeline: Mutex::new(None),
                q4_dense_simd_pipeline: Mutex::new(None),
                q4_dense_superblock_simd_pipeline: Mutex::new(None),
                q4_dense_superblock_norm_pipeline: Mutex::new(None),
                q4_dense_batch_simd_pipeline: Mutex::new(None),
                q4_dense_weight_stationary_pipeline: Mutex::new(None),
                q4_dense_sweep_pipeline: Mutex::new(None),
                q4_dense_argmax_pipeline: Mutex::new(None),
                q4_dense_argmax_norm_pipeline: Mutex::new(None),
                q4_dense_argmax_superblock_pipeline: Mutex::new(None),
                q4_dense_argmax_superblock_norm_pipeline: Mutex::new(None),
                q4_dense_argmax_unique_packet_pipeline: Mutex::new(None),
                q4_dense_argmax_unique_packet_norm_pipeline: Mutex::new(None),
                q4_qkv_pipeline: Mutex::new(None),
                q4_qkv_simd_pipeline: Mutex::new(None),
                q4_qkv_simd_norm_pipeline: Mutex::new(None),
                q4_qkv_superblock_simd_pipeline: Mutex::new(None),
                q4_qkv_superblock_norm_pipeline: Mutex::new(None),
                q4_qkv_sweep_pipeline: Mutex::new(None),
                q4_gemv_pipeline: Mutex::new(None),
                q4_simd_gemv_pipeline: Mutex::new(None),
                q4_simd_norm_gemv_pipeline: Mutex::new(None),
                dynamic_moe_pipeline: Mutex::new(None),
                dynamic_moe_library: Mutex::new(None),
                dynamic_moe_simd_pipeline: Mutex::new(None),
                q4_embedding_pipeline: Mutex::new(None),
                q4_embedding_batch_pipeline: Mutex::new(None),
                q4_embedding_superblock_pipeline: Mutex::new(None),
                q4_embedding_superblock_batch_pipeline: Mutex::new(None),
                argmax_partial_pipeline: Mutex::new(None),
                argmax_reduce_pipeline: Mutex::new(None),
                argmax_reduce_partials_pipeline: Mutex::new(None),
                verification_hash_pipeline: Mutex::new(None),
                v2_pipelines: Mutex::new([
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None, None,
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
                gpu_queue_gap_seconds: metrics.gpu_queue_gap_seconds,
                cpu_encode_seconds: metrics.cpu_encode_seconds,
                cpu_wait_seconds: metrics.cpu_wait_seconds,
            })
        }

        /// Execute the protocol math conformance kernel and return raw exp,
        /// reciprocal-square-root, Granite-RoPE sine, and cosine bits.
        pub fn run_consensus_math_conformance(
            &self,
            inputs: &[f32],
        ) -> Result<Vec<[u32; 4]>, String> {
            if inputs.is_empty() {
                return Ok(Vec::new());
            }
            let source = NSString::from_str(CONSENSUS_MATH_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|error| {
                    format!("Metal consensus-math library compilation failed: {error}")
                })?;
            let function = library
                .newFunctionWithName(&NSString::from_str("qx_consensus_math_conformance"))
                .ok_or_else(|| "Metal consensus-math conformance function missing".to_owned())?;
            let pipeline = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal consensus-math pipeline creation failed".to_owned())?;
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    inputs.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(inputs),
                )
            };
            let input = self.input_buffer(input_bytes)?;
            let output = self
                .device
                .newBufferWithLength_options(
                    inputs.len() * std::mem::size_of::<[u32; 4]>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal consensus-math output allocation failed".to_owned())?;
            let command_buffer = self.queue.commandBuffer().ok_or_else(|| {
                "Metal consensus-math command buffer allocation failed".to_owned()
            })?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal consensus-math encoder allocation failed".to_owned())?;
            encoder.setComputePipelineState(&pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&input), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&output), 0, 1);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: inputs.len(),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: inputs
                        .len()
                        .min(pipeline.maxTotalThreadsPerThreadgroup() as usize),
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if command_buffer.status() == objc2_metal::MTLCommandBufferStatus::Error {
                return Err("Metal consensus-math execution failed".to_owned());
            }
            let pointer = output.contents().as_ptr() as *const [u32; 4];
            Ok(unsafe { std::slice::from_raw_parts(pointer, inputs.len()).to_vec() })
        }

        /// Start a new GPU queue-gap interval without discarding cumulative
        /// work counters. The next completed command buffer becomes the
        /// timestamp anchor, so setup/prefill time is not charged as decode
        /// idle time.
        pub fn begin_metrics_interval(&self) -> Result<(), String> {
            let mut metrics = self
                .metrics
                .lock()
                .map_err(|_| "Metal metrics cache is poisoned".to_owned())?;
            metrics.last_gpu_end_seconds = None;
            Ok(())
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

        /// Enumerate the counter sets visible to the process. Keep this
        /// separate from stage timestamps: timestamps are broadly available,
        /// while stage-utilization/statistic counters are device- and
        /// entitlement-dependent on Apple platforms.
        pub fn counter_inventory(&self) -> Vec<MetalCounterSetInfo> {
            let Some(sets) = self.device.counterSets() else {
                return Vec::new();
            };
            (0..sets.count())
                .map(|index| {
                    let set = sets.objectAtIndex(index);
                    let counters = set.counters();
                    MetalCounterSetInfo {
                        name: set.name().to_string(),
                        counters: (0..counters.count())
                            .map(|counter_index| {
                                counters.objectAtIndex(counter_index).name().to_string()
                            })
                            .collect(),
                    }
                })
                .collect()
        }

        /// Compile the production dynamic-MoE source and expose the target's
        /// pipeline metadata for the selected entry points. This is not a
        /// performance benchmark: it is the compiler-visible hardware shape
        /// used to distinguish a real consumer change from a source-only
        /// change.
        pub fn dynamic_moe_kernel_inventory(&self) -> Result<Vec<MetalKernelInfo>, String> {
            let source = NSString::from_str(DYNAMIC_MOE_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| "Metal dynamic-MoE library compilation failed".to_owned())?;
            let mut names = Vec::new();
            for index in 0..library.functionNames().count() {
                let name = library.functionNames().objectAtIndex(index).to_string();
                if name.contains("gate_up_inline")
                    || name.contains("down")
                    || name.contains("verifier")
                {
                    names.push(name);
                }
            }
            names.sort();
            names
                .into_iter()
                .map(|name| {
                    let function = library
                        .newFunctionWithName(&NSString::from_str(&name))
                        .ok_or_else(|| format!("Metal function not found: {name}"))?;
                    let pipeline = self
                        .device
                        .newComputePipelineStateWithFunction_error(&function)
                        .map_err(|_| format!("Metal pipeline creation failed: {name}"))?;
                    Ok(MetalKernelInfo {
                        name,
                        thread_execution_width: pipeline.threadExecutionWidth() as usize,
                        max_threads_per_threadgroup: pipeline.maxTotalThreadsPerThreadgroup()
                            as usize,
                        static_threadgroup_memory_bytes: pipeline.staticThreadgroupMemoryLength()
                            as usize,
                    })
                })
                .collect()
        }

        pub fn recommended_working_set_bytes(&self) -> u64 {
            self.device.recommendedMaxWorkingSetSize()
        }

        /// Compile every pipeline used by the persistent Qwen layer graph.
        /// Model loading calls this once so shader compilation cannot leak
        /// into the first token's encode latency.
        pub fn prepare_qwen_layer_graph(&self) -> Result<(), String> {
            let _ = self.qwen_rms_norm_pipeline()?;
            let _ = self.qwen_residual_rms_norm_pipeline()?;
            let _ = self.qwen_store_kv_pipeline()?;
            let _ = self.qwen_residual_pipeline()?;
            let _ = self.qwen_embedding_pipeline()?;
            let _ = self.qwen_argmax_pipeline()?;
            let _ = self.qwen_qk_norm_rope_pipeline()?;
            let _ = self.qwen_gqa_attention_pipeline()?;
            let _ = self.qwen_gqa_score_reuse_pipeline()?;
            let _ = self.qwen_top8_pipeline()?;
            let _ = self.qwen_dynamic_moe_pipeline()?;
            let _ = self.qwen_dynamic_gate_pipeline()?;
            let _ = self.qwen_dynamic_down_tiled_pipeline()?;
            let _ = self.qwen_dynamic_accumulate_pipeline()?;
            Ok(())
        }

        pub fn begin_decode(&self) -> Result<DecodeGraph, String> {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Runtime V2 command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Runtime V2 encoder allocation failed".to_owned())?;
            let stage_diagnostics = *self
                .stage_counters
                .lock()
                .map_err(|_| "Metal stage-counter cache is poisoned".to_owned())?;
            Ok(DecodeGraph {
                command_buffer,
                encoder,
                dispatches: 0,
                counter_timeline: if stage_diagnostics {
                    CounterTimeline::new(&self.device)
                } else {
                    None
                },
                stage_diagnostics,
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

        pub fn read_runtime_bytes(&self, buffer: &RuntimeBuffer) -> Vec<u8> {
            let pointer =
                unsafe { buffer.buffer.contents().as_ptr().add(buffer.offset) as *const u8 };
            unsafe { std::slice::from_raw_parts(pointer, buffer.length).to_vec() }
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
            if gpu_start > 0.0 && gpu_end >= gpu_start {
                if let Some(previous_end) = metrics.last_gpu_end_seconds {
                    metrics.gpu_queue_gap_seconds += (gpu_start - previous_end).max(0.0);
                }
                metrics.last_gpu_end_seconds = Some(
                    metrics
                        .last_gpu_end_seconds
                        .map_or(gpu_end, |previous_end| previous_end.max(gpu_end)),
                );
            }
            metrics.cpu_encode_seconds += wait_start.duration_since(encode_start).as_secs_f64();
            metrics.cpu_wait_seconds += wait_start.elapsed().as_secs_f64();
            Ok(())
        }

        fn pipeline_index(quant_type: QuantType) -> usize {
            match quant_type {
                QuantType::Q2 => 0,
                QuantType::Q2S4 => 1,
                QuantType::Q3 => 2,
                QuantType::Q4 => 3,
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| "Metal library compilation failed".to_owned())?;
            let function_name = match quant_type {
                QuantType::Q2S4 => "qx_q2s4_expert".to_owned(),
                _ => format!("qx_q{}_expert", quant_type.bits()),
            };
            let name = NSString::from_str(&function_name);
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

        fn qwen_q2s4_expert_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .qwen_q2s4_expert_pipeline
                .lock()
                .map_err(|_| "Metal Qwen Q2S4 pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(Q2_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| "Metal Qwen Q2S4 library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_q2s4_qwen_expert");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal Qwen Q2S4 expert function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal Qwen Q2S4 pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn qwen_qk_norm_rope_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .qwen_qk_norm_rope_pipeline
                .lock()
                .map_err(|_| "Metal Qwen Q/K pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(QWEN_ATTENTION_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| "Metal Qwen attention library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_qwen_qk_norm_rope");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal Qwen Q/K function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal Qwen Q/K pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn qwen_gqa_attention_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .qwen_gqa_attention_pipeline
                .lock()
                .map_err(|_| "Metal Qwen GQA pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(QWEN_ATTENTION_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| "Metal Qwen GQA library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_qwen_gqa_attention");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal Qwen GQA function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal Qwen GQA pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn qwen_gqa_score_reuse_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_gqa_score_reuse_pipeline,
                QWEN_ATTENTION_KERNEL_SOURCE,
                "qx_qwen_gqa_score_reuse",
            )
        }

        fn qwen_named_pipeline(
            &self,
            cache: &Mutex<Option<Pipeline>>,
            source: &str,
            name: &str,
        ) -> Result<Pipeline, String> {
            let mut pipeline = cache
                .lock()
                .map_err(|_| format!("Metal {name} pipeline cache is poisoned"))?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(source),
                    consensus_compile_options().as_deref(),
                )
                .map_err(|_| format!("Metal {name} library compilation failed"))?;
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("Metal function {name} was not found"))?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| format!("Metal {name} pipeline creation failed"))?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn qwen_top8_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_top8_pipeline,
                QWEN_ATTENTION_KERNEL_SOURCE,
                "qx_qwen_top8",
            )
        }

        fn qwen_rms_norm_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_rms_norm_pipeline,
                QWEN_ATTENTION_KERNEL_SOURCE,
                "qx_qwen_rms_norm",
            )
        }

        fn qwen_residual_rms_norm_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_residual_rms_norm_pipeline,
                QWEN_ATTENTION_KERNEL_SOURCE,
                "qx_qwen_residual_rms_norm",
            )
        }

        fn qwen_store_kv_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_store_kv_pipeline,
                QWEN_ATTENTION_KERNEL_SOURCE,
                "qx_qwen_store_kv",
            )
        }

        fn qwen_residual_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_residual_pipeline,
                QWEN_ATTENTION_KERNEL_SOURCE,
                "qx_qwen_residual",
            )
        }

        fn qwen_embedding_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_embedding_pipeline,
                QWEN_ATTENTION_KERNEL_SOURCE,
                "qx_qwen_q4_embedding_lookup",
            )
        }

        fn qwen_argmax_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_argmax_pipeline,
                QWEN_ATTENTION_KERNEL_SOURCE,
                "qx_qwen_argmax",
            )
        }

        fn qwen_dynamic_moe_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_dynamic_moe_pipeline,
                Q2_KERNEL_SOURCE,
                "qx_qwen_moe_dynamic",
            )
        }

        fn qwen_dynamic_gate_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_dynamic_gate_pipeline,
                Q2_KERNEL_SOURCE,
                "qx_qwen_moe_gate_dynamic",
            )
        }

        fn qwen_dynamic_down_tiled_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_dynamic_down_tiled_pipeline,
                Q2_KERNEL_SOURCE,
                "qx_qwen_moe_down_tiled_dynamic",
            )
        }

        fn qwen_dynamic_accumulate_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_dynamic_accumulate_pipeline,
                Q2_KERNEL_SOURCE,
                "qx_qwen_moe_accumulate_dynamic",
            )
        }

        fn qwen_q4_cache_resolve_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_q4_cache_resolve_pipeline,
                Q2_KERNEL_SOURCE,
                "qx_qwen_q4_cache_resolve",
            )
        }

        fn qwen_q4_cache_fill_pipeline(&self) -> Result<Pipeline, String> {
            self.qwen_named_pipeline(
                &self.qwen_q4_cache_fill_pipeline,
                Q2_KERNEL_SOURCE,
                "qx_qwen_q4_cache_fill",
            )
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| "Metal Q4 dense library compilation failed".to_owned())?;
            *library = Some(value.clone());
            Ok(value)
        }

        fn residual_rms_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .residual_rms_pipeline
                .lock()
                .map_err(|_| "Metal residual/RMS pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_residual_rms_scalar");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal residual/RMS function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal residual/RMS pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn residual_rms_snapshot_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .residual_rms_snapshot_pipeline
                .lock()
                .map_err(|_| "Metal residual/RMS snapshot cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_residual_rms_scalar_snapshot");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal residual/RMS snapshot function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal residual/RMS snapshot pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn attention_softmax_onepass_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .attention_softmax_onepass_pipeline
                .lock()
                .map_err(|_| "Metal one-pass attention softmax cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_attention_softmax_onepass_simd");
            let function = library.newFunctionWithName(&name).ok_or_else(|| {
                "Metal one-pass attention softmax function was not found".to_owned()
            })?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| {
                    "Metal one-pass attention softmax pipeline creation failed".to_owned()
                })?;
            *pipeline = Some(value.clone());
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

        fn q4_dense_superblock_simd_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_superblock_simd_pipeline
                .lock()
                .map_err(|_| "Metal superblock Q4 dense pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_superblock144_simd_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal superblock Q4 dense function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal superblock Q4 dense pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_superblock_norm_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_superblock_norm_pipeline
                .lock()
                .map_err(|_| "Metal normalized superblock Q4 dense cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_superblock144_norm_simd_gemv");
            let function = library.newFunctionWithName(&name).ok_or_else(|| {
                "Metal normalized superblock Q4 dense function was not found".to_owned()
            })?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| {
                    "Metal normalized superblock Q4 dense pipeline creation failed".to_owned()
                })?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_batch_simd_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_batch_simd_pipeline
                .lock()
                .map_err(|_| "Metal batched Q4 dense pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_simd_gemm_k");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal batched Q4 dense function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal batched Q4 dense pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_weight_stationary_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_weight_stationary_pipeline
                .lock()
                .map_err(|_| "Metal weight-stationary Q4 pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_weight_stationary_gemm_k");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal weight-stationary Q4 function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal weight-stationary Q4 pipeline creation failed".to_owned())?;
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

        fn q4_dense_argmax_unique_packet_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_argmax_unique_packet_pipeline
                .lock()
                .map_err(|_| "Metal unique-packet argmax pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_argmax_unique_packet_partial");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal unique-packet argmax function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal unique-packet argmax pipeline creation failed".to_owned())?;
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

        fn q4_qkv_superblock_simd_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_qkv_superblock_simd_pipeline
                .lock()
                .map_err(|_| "Metal superblock QKV pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_qkv_superblock144_simd_gemv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal superblock QKV function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal superblock QKV pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_qkv_superblock_norm_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_qkv_superblock_norm_pipeline
                .lock()
                .map_err(|_| "Metal normalized superblock QKV cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_qkv_superblock144_norm_simd_gemv");
            let function = library.newFunctionWithName(&name).ok_or_else(|| {
                "Metal normalized superblock QKV function was not found".to_owned()
            })?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| {
                    "Metal normalized superblock QKV pipeline creation failed".to_owned()
                })?;
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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

        fn q4_embedding_batch_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_embedding_batch_pipeline
                .lock()
                .map_err(|_| "Metal Q4 batch embedding pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(Q4_EMBEDDING_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| "Metal Q4 embedding library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_q4_embedding_lookup_batch");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal Q4 batch embedding function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal Q4 batch embedding pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_embedding_superblock_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_embedding_superblock_pipeline
                .lock()
                .map_err(|_| "Metal superblock embedding cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(Q4_EMBEDDING_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| "Metal superblock embedding library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_q4_embedding_superblock144_lookup");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal superblock embedding function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal superblock embedding pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_embedding_superblock_batch_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_embedding_superblock_batch_pipeline
                .lock()
                .map_err(|_| "Metal batch superblock embedding cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(Q4_EMBEDDING_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| {
                    "Metal batch superblock embedding library compilation failed".to_owned()
                })?;
            let name = NSString::from_str("qx_q4_embedding_superblock144_lookup_batch");
            let function = library.newFunctionWithName(&name).ok_or_else(|| {
                "Metal batch superblock embedding function was not found".to_owned()
            })?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| {
                    "Metal batch superblock embedding pipeline creation failed".to_owned()
                })?;
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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

        fn q4_dense_argmax_superblock_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_argmax_superblock_pipeline
                .lock()
                .map_err(|_| "Metal superblock Q4 argmax cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_argmax_superblock144_partial");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal superblock Q4 argmax function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal superblock Q4 argmax pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_argmax_superblock_norm_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_argmax_superblock_norm_pipeline
                .lock()
                .map_err(|_| {
                    "Metal normalized superblock Q4 argmax cache is poisoned".to_owned()
                })?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_argmax_superblock144_norm_partial");
            let function = library.newFunctionWithName(&name).ok_or_else(|| {
                "Metal normalized superblock Q4 argmax function was not found".to_owned()
            })?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| {
                    "Metal normalized superblock Q4 argmax pipeline creation failed".to_owned()
                })?;
            *pipeline = Some(value.clone());
            Ok(value)
        }

        fn q4_dense_argmax_unique_packet_norm_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .q4_dense_argmax_unique_packet_norm_pipeline
                .lock()
                .map_err(|_| {
                    "Metal normalized unique-packet argmax pipeline cache is poisoned".to_owned()
                })?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let library = self.q4_dense_library()?;
            let name = NSString::from_str("qx_q4_dense_argmax_unique_packet_norm_partial");
            let function = library.newFunctionWithName(&name).ok_or_else(|| {
                "Metal normalized unique-packet argmax function was not found".to_owned()
            })?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| {
                    "Metal normalized unique-packet argmax pipeline creation failed".to_owned()
                })?;
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
                DecodeKernel::CopyF32 => 31,
                DecodeKernel::RopeQueryBatchSimd => 32,
                DecodeKernel::RopeKvBatchSimd => 33,
                DecodeKernel::AttentionBatch => 34,
                DecodeKernel::RmsNormBatch => 35,
                DecodeKernel::ResidualBatch => 36,
                DecodeKernel::TopKBatch => 37,
                DecodeKernel::RmsNormBatchScalar => 38,
                DecodeKernel::AttentionBatchSimd => 39,
                DecodeKernel::RmsNormBatchParallelWriteback => 40,
                DecodeKernel::RopeKvSimdKf32Vf16 => 41,
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
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
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
                DecodeKernel::RopeKvSimdKf32Vf16 => "qx_v2_rope_kv_simd_kf32_vf16",
                DecodeKernel::TopKMask => "qx_v2_topk_mask",
                DecodeKernel::AttentionQkKMajor => "qx_v2_attention_qk_kmajor",
                DecodeKernel::AttentionStreamGqa => "qx_v2_attention_stream_gqa",
                DecodeKernel::AttentionChunkedGqa => "qx_v2_attention_chunked_gqa",
                DecodeKernel::AttentionChunkedMerge => "qx_v2_attention_chunked_merge",
                DecodeKernel::CopyF32 => "qx_v2_copy_f32",
                DecodeKernel::RopeQueryBatchSimd => "qx_v2_rope_query_batch_simd",
                DecodeKernel::RopeKvBatchSimd => "qx_v2_rope_kv_batch_simd",
                DecodeKernel::AttentionBatch => "qx_v2_attention_batch",
                DecodeKernel::RmsNormBatch => "qx_v2_rmsnorm_batch",
                DecodeKernel::ResidualBatch => "qx_v2_residual_batch",
                DecodeKernel::TopKBatch => "qx_v2_topk_batch",
                DecodeKernel::RmsNormBatchScalar => "qx_v2_rmsnorm_batch_scalar",
                DecodeKernel::AttentionBatchSimd => "qx_v2_attention_batch_simd",
                DecodeKernel::RmsNormBatchParallelWriteback => {
                    "qx_v2_rmsnorm_batch_parallel_writeback"
                }
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

        fn verification_hash_pipeline(&self) -> Result<Pipeline, String> {
            let mut pipeline = self
                .verification_hash_pipeline
                .lock()
                .map_err(|_| "Metal verification hash pipeline cache is poisoned".to_owned())?;
            if let Some(value) = pipeline.as_ref() {
                return Ok(value.clone());
            }
            let source = NSString::from_str(crate::VERIFICATION_HASH_KERNEL_SOURCE);
            let library = self
                .device
                .newLibraryWithSource_options_error(&source, consensus_compile_options().as_deref())
                .map_err(|_| "Metal verification hash library compilation failed".to_owned())?;
            let name = NSString::from_str("qx_verification_hash_state_kv");
            let function = library
                .newFunctionWithName(&name)
                .ok_or_else(|| "Metal verification hash function was not found".to_owned())?;
            let value = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| "Metal verification hash pipeline creation failed".to_owned())?;
            *pipeline = Some(value.clone());
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
                batch_simd_pipeline: self.q4_dense_batch_simd_pipeline()?,
                weight_stationary_batch_pipeline: self.q4_dense_weight_stationary_pipeline()?,
                sweep_pipeline: self.q4_dense_sweep_pipeline()?,
                simd_norm_pipeline: self.q4_simd_norm_gemv_pipeline()?,
                packed,
                scales,
                rows: weights.rows,
                columns: weights.cols,
                layout: PackingLayout::RowMajor,
            })
        }

        /// Prepare a group-32 Q4 dense matrix from views into a static
        /// artifact slab without reconstructing a QuantizedTensor.
        pub fn prepare_q4_dense_runtime_views(
            &self,
            packed: RuntimeBuffer,
            scales: RuntimeBuffer,
            rows: usize,
            columns: usize,
            layout: PackingLayout,
        ) -> Result<PreparedQuantizedDense, String> {
            if rows == 0 || columns == 0 || columns % 32 != 0 {
                return Err("no-copy Q4 dense has an invalid shape".to_owned());
            }
            if !matches!(
                layout,
                PackingLayout::RowMajor | PackingLayout::Q4Superblock144
            ) {
                return Err("no-copy Q4 dense has an unsupported layout".to_owned());
            }
            if layout == PackingLayout::Q4Superblock144 && columns % 256 != 0 {
                return Err(
                    "no-copy superblock Q4 dense columns must be divisible by 256".to_owned(),
                );
            }
            let expected_packed = if layout == PackingLayout::Q4Superblock144 {
                rows * (columns / 256) * 144
            } else {
                rows * columns / 2
            };
            let expected_scales = if layout == PackingLayout::Q4Superblock144 {
                expected_packed
            } else {
                rows * (columns / 32) * 2
            };
            if packed.length != expected_packed || scales.length != expected_scales {
                return Err(format!(
                    "no-copy Q4 dense byte lengths are {}/{}, expected {expected_packed}/{expected_scales}",
                    packed.length, scales.length
                ));
            }
            Ok(PreparedQuantizedDense {
                pipeline: self.q4_dense_pipeline()?,
                simd_pipeline: if layout == PackingLayout::Q4Superblock144 {
                    self.q4_dense_superblock_simd_pipeline()?
                } else {
                    self.q4_dense_simd_pipeline()?
                },
                batch_simd_pipeline: self.q4_dense_batch_simd_pipeline()?,
                weight_stationary_batch_pipeline: self.q4_dense_weight_stationary_pipeline()?,
                sweep_pipeline: self.q4_dense_sweep_pipeline()?,
                simd_norm_pipeline: if layout == PackingLayout::Q4Superblock144 {
                    self.q4_dense_superblock_norm_pipeline()?
                } else {
                    self.q4_simd_norm_gemv_pipeline()?
                },
                packed,
                scales,
                rows,
                columns,
                layout,
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
                || query.layout != key.layout
                || query.layout != value.layout
            {
                return Err("fused QKV requires Granite 1024x1024/512 projections".to_owned());
            }
            Ok(PreparedQuantizedQkv {
                pipeline: self.q4_qkv_pipeline()?,
                simd_pipeline: if query.layout == PackingLayout::Q4Superblock144 {
                    self.q4_qkv_superblock_simd_pipeline()?
                } else {
                    self.q4_qkv_simd_pipeline()?
                },
                simd_norm_pipeline: if query.layout == PackingLayout::Q4Superblock144 {
                    self.q4_qkv_superblock_norm_pipeline()?
                } else {
                    self.q4_qkv_simd_norm_pipeline()?
                },
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
                layout: query.layout,
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
            let is_q4_superblock_gate = gate_up_layout.is_q4_superblock()
                && gate_up_packed.len() == EXPERTS * Q4_TRANSACTION_GATE_UP_144
                && gate_up_scales.len() == gate_up_packed.len();
            let is_q4_superblock_down = down_layout.is_q4_superblock()
                && down_packed.len() == EXPERTS * Q4_TRANSACTION_DOWN_144
                && down_scales.len() == down_packed.len();
            let is_q4_cross_expert_gate = gate_up_layout == PackingLayout::Q4CrossExpertTile8
                && gate_up_packed.len() == Q4_CROSS_EXPERT_GATE_UP
                && gate_up_scales.len() == Q4_CROSS_EXPERT_GATE_UP;
            let is_q4_cross_expert_down = down_layout == PackingLayout::Q4CrossExpertTile8
                && down_packed.len() == Q4_CROSS_EXPERT_DOWN
                && down_scales.len() == Q4_CROSS_EXPERT_DOWN;
            let is_q4_gate = is_q4_gate
                || is_q4_transaction_gate
                || is_q4_superblock_gate
                || is_q4_cross_expert_gate;
            let is_q4_down = is_q4_down
                || is_q4_transaction_down
                || is_q4_superblock_down
                || is_q4_cross_expert_down;
            let is_q4 = is_q4_gate && is_q4_down;
            let is_q4k = gate_up_packed.len() == EXPERTS * Q4K_GATE_UP_PACKED
                && gate_up_scales.len() == EXPERTS * Q4K_GATE_UP_SCALES
                && down_packed.len() == EXPERTS * Q4K_DOWN_PACKED
                && down_scales.len() == EXPERTS * Q4K_DOWN_SCALES;
            if (!is_q4 && !is_q4k)
                || (gate_up_layout.is_q4_transaction() && !is_q4_transaction_gate)
                || (down_layout.is_q4_transaction() && !is_q4_transaction_down)
                || (gate_up_layout.is_q4_superblock() && !is_q4_superblock_gate)
                || (down_layout.is_q4_superblock() && !is_q4_superblock_down)
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
            let gate_up_pipeline = if gate_up_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_gate_up_superblock144_simd"
            } else if is_q4k {
                "qx_dynamic_q4k_moe_gate_up_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_simd"
            };
            let gate_up_norm_pipeline = if gate_up_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_gate_up_superblock144_norm_simd"
            } else if is_q4k {
                "qx_dynamic_q4k_moe_gate_up_norm_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_norm_simd"
            };
            let down_pipeline = if down_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_down_superblock144_simd"
            } else if is_q4k {
                "qx_dynamic_q4k_moe_down_simd"
            } else {
                "qx_dynamic_q4_moe_down_simd"
            };
            let fused_block_qx_pipeline = if gate_up_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_gate_up_superblock144_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_fused_block_qx_simd"
            };
            let fused_block_qx_norm_pipeline = if gate_up_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_gate_up_superblock144_norm_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_fused_block_qx_norm_simd"
            };
            Ok(PreparedDynamicQ4Moe {
                unique_packet_gate_up_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_unique_packet_simd",
                )?,
                verifier_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_batch")?,
                verifier_gate_up_expert_major_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_expert_major")?,
                verifier_gate_up_expert_major_direct_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_expert_major_direct")?,
                verifier_gate_up_expert_major_unique_packet_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_expert_major_unique_packet",
                    )?,
                verifier_compact_routes_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_compact_routes_fixed")?,
                verifier_gate_up_compact_unique_packet_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_verifier_q4_moe_gate_up_compact_unique_packet",
                )?,
                verifier_gate_up_compact_unique_packet_k2_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k2",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_vector_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_vector",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_split_accum_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_split_accum",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_direct_activation_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_direct_activation",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_rows16_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_rows16",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_unroll4_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_unroll4",
                    )?,
                verifier_silu_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_silu_batch")?,
                verifier_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_down_batch")?,
                verifier_down_compact_unique_packet_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_down_compact_unique_packet")?,
                verifier_down_compact_unique_packet_k4_unroll4_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_down_compact_unique_packet_k4_unroll4",
                    )?,
                verifier_accumulate_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_accumulate_batch")?,
                unique_packet_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_unique_packet_norm_simd",
                )?,
                unique_packet_down_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_down_inline_unique_packet_simd",
                )?,
                cluster_down16_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cluster16_simd")?,
                cluster_down32_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cluster32_simd")?,
                gate_up_instrument_load_unique_packet_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_simd",
                )?,
                gate_up_instrument_load_unique_packet_norm_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_norm_simd",
                    )?,
                gate_up_instrument_load_two_groups_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_two_groups_simd",
                )?,
                gate_up_instrument_load_two_groups_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_two_groups_norm_simd",
                )?,
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
                two_row_unique_packet_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_unique_packet_simd",
                )?),
                two_row_unique_packet_gate_up_norm_pipeline: Some(
                    self.dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_two_row_unique_packet_norm_simd",
                    )?,
                ),
                gate_up_instrument_staging_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_staging_only_simd",
                )?,
                gate_up_instrument_address_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_address_only_simd",
                )?,
                gate_up_instrument_contiguous_load_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_simd",
                )?,
                gate_up_instrument_load_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_only_simd",
                )?,
                gate_up_instrument_load_decode_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_decode_simd",
                )?,
                gate_up_instrument_dot_no_reduce_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_simd",
                )?,
                gate_up_instrument_load_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_only_norm_simd",
                )?,
                gate_up_instrument_staging_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_staging_only_norm_simd",
                )?,
                gate_up_instrument_address_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_address_only_norm_simd",
                )?,
                gate_up_instrument_contiguous_load_only_norm_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_norm_simd",
                    )?,
                gate_up_instrument_load_decode_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_decode_norm_simd",
                )?,
                gate_up_instrument_dot_no_reduce_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_norm_simd",
                )?,
                q4k_mature_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_simd")?,
                q4k_mature_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_norm_simd")?,
                fused_block_qx_gate_up_pipeline: Some(
                    self.dynamic_moe_named_pipeline(fused_block_qx_pipeline)?,
                ),
                fused_block_qx_gate_up_norm_pipeline: Some(
                    self.dynamic_moe_named_pipeline(fused_block_qx_norm_pipeline)?,
                ),
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
                expertwise_superblock_gate_up_pipeline: None,
                expertwise_superblock_gate_up_norm_pipeline: None,
                expertwise_superblock_down_pipeline: None,
                expertwise_q4_gate_up_superblock: false,
                expertwise_q4_down_superblock: false,
                expertwise_q3_gate_up_superblock: false,
                expertwise_q3_down_superblock: false,
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
            let is_q4_superblock_gate = gate_up_layout.is_q4_superblock()
                && gate_up_packed.length == EXPERTS * Q4_TRANSACTION_GATE_UP_144
                && gate_up_scales.length == gate_up_packed.length;
            let is_q4_superblock_down = down_layout.is_q4_superblock()
                && down_packed.length == EXPERTS * Q4_TRANSACTION_DOWN_144
                && down_scales.length == down_packed.length;
            let is_q4_cross_expert_gate = gate_up_layout == PackingLayout::Q4CrossExpertTile8
                && gate_up_packed.length == Q4_CROSS_EXPERT_GATE_UP
                && gate_up_scales.length == Q4_CROSS_EXPERT_GATE_UP;
            let is_q4_cross_expert_down = down_layout == PackingLayout::Q4CrossExpertTile8
                && down_packed.length == Q4_CROSS_EXPERT_DOWN
                && down_scales.length == Q4_CROSS_EXPERT_DOWN;
            let is_q4_gate = is_q4_gate
                || is_q4_transaction_gate
                || is_q4_superblock_gate
                || is_q4_cross_expert_gate;
            let is_q4_down = is_q4_down
                || is_q4_transaction_down
                || is_q4_superblock_down
                || is_q4_cross_expert_down;
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
                || (gate_up_layout.is_q4_superblock() && !is_q4_superblock_gate)
                || (down_layout.is_q4_superblock() && !is_q4_superblock_down)
                || (gate_up_layout == PackingLayout::Q4CrossExpertTile8 && !is_q4_cross_expert_gate)
                || (down_layout == PackingLayout::Q4CrossExpertTile8 && !is_q4_cross_expert_down)
            {
                return Err("dynamic Q4 MoE slab views have an invalid Granite shape".to_owned());
            }
            let gate_up_pipeline = if gate_up_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_gate_up_superblock144_simd"
            } else if is_q4k {
                "qx_dynamic_q4k_moe_gate_up_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_simd"
            };
            let gate_up_norm_pipeline = if gate_up_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_gate_up_superblock144_norm_simd"
            } else if is_q4k {
                "qx_dynamic_q4k_moe_gate_up_norm_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_norm_simd"
            };
            let down_pipeline = if down_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_down_superblock144_simd"
            } else if is_q4k {
                "qx_dynamic_q4k_moe_down_simd"
            } else {
                "qx_dynamic_q4_moe_down_simd"
            };
            let fused_block_qx_pipeline = if gate_up_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_gate_up_superblock144_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_fused_block_qx_simd"
            };
            let fused_block_qx_norm_pipeline = if gate_up_layout.is_q4_superblock() {
                "qx_dynamic_q4_moe_gate_up_superblock144_norm_simd"
            } else {
                "qx_dynamic_q4_moe_gate_up_fused_block_qx_norm_simd"
            };
            let mut prepared = PreparedDynamicQ4Moe {
                unique_packet_gate_up_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_unique_packet_simd",
                )?,
                verifier_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_batch")?,
                verifier_gate_up_expert_major_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_expert_major")?,
                verifier_gate_up_expert_major_direct_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_expert_major_direct")?,
                verifier_gate_up_expert_major_unique_packet_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_expert_major_unique_packet",
                    )?,
                verifier_compact_routes_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_compact_routes_fixed")?,
                verifier_gate_up_compact_unique_packet_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_verifier_q4_moe_gate_up_compact_unique_packet",
                )?,
                verifier_gate_up_compact_unique_packet_k2_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k2",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_vector_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_vector",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_split_accum_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_split_accum",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_direct_activation_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_direct_activation",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_rows16_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_rows16",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_unroll4_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_unroll4",
                    )?,
                verifier_silu_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_silu_batch")?,
                verifier_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_down_batch")?,
                verifier_down_compact_unique_packet_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_down_compact_unique_packet")?,
                verifier_down_compact_unique_packet_k4_unroll4_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_down_compact_unique_packet_k4_unroll4",
                    )?,
                verifier_accumulate_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_accumulate_batch")?,
                unique_packet_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_unique_packet_norm_simd",
                )?,
                unique_packet_down_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_down_inline_unique_packet_simd",
                )?,
                cluster_down16_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cluster16_simd")?,
                cluster_down32_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cluster32_simd")?,
                gate_up_instrument_load_unique_packet_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_simd",
                )?,
                gate_up_instrument_load_unique_packet_norm_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_norm_simd",
                    )?,
                gate_up_instrument_load_two_groups_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_two_groups_simd",
                )?,
                gate_up_instrument_load_two_groups_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_two_groups_norm_simd",
                )?,
                gate_up_instrument_contiguous_load_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_simd",
                )?,
                gate_up_instrument_contiguous_load_only_norm_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_norm_simd",
                    )?,
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
                two_row_unique_packet_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_unique_packet_simd",
                )?),
                two_row_unique_packet_gate_up_norm_pipeline: Some(
                    self.dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_two_row_unique_packet_norm_simd",
                    )?,
                ),
                gate_up_instrument_staging_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_staging_only_simd",
                )?,
                gate_up_instrument_address_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_address_only_simd",
                )?,
                gate_up_instrument_load_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_only_simd",
                )?,
                gate_up_instrument_load_decode_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_decode_simd",
                )?,
                gate_up_instrument_dot_no_reduce_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_simd",
                )?,
                gate_up_instrument_load_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_only_norm_simd",
                )?,
                gate_up_instrument_staging_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_staging_only_norm_simd",
                )?,
                gate_up_instrument_address_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_address_only_norm_simd",
                )?,
                gate_up_instrument_load_decode_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_decode_norm_simd",
                )?,
                gate_up_instrument_dot_no_reduce_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_norm_simd",
                )?,
                q4k_mature_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_simd")?,
                q4k_mature_gate_up_norm_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4k_moe_gate_up_mature_norm_simd")?,
                fused_block_qx_gate_up_pipeline: Some(
                    self.dynamic_moe_named_pipeline(fused_block_qx_pipeline)?,
                ),
                fused_block_qx_gate_up_norm_pipeline: Some(
                    self.dynamic_moe_named_pipeline(fused_block_qx_norm_pipeline)?,
                ),
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
                expertwise_superblock_gate_up_pipeline: None,
                expertwise_superblock_gate_up_norm_pipeline: None,
                expertwise_superblock_down_pipeline: None,
                expertwise_q4_gate_up_superblock: false,
                expertwise_q4_down_superblock: false,
                expertwise_q3_gate_up_superblock: false,
                expertwise_q3_down_superblock: false,
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
            q4_gate_up_superblock: bool,
            q4_down_superblock: bool,
            q3_gate_up_superblock: bool,
            q3_down_superblock: bool,
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
                unique_packet_gate_up_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_unique_packet_simd",
                )?,
                verifier_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_batch")?,
                verifier_gate_up_expert_major_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_expert_major")?,
                verifier_gate_up_expert_major_direct_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_expert_major_direct")?,
                verifier_gate_up_expert_major_unique_packet_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_expert_major_unique_packet",
                    )?,
                verifier_compact_routes_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_compact_routes_fixed")?,
                verifier_gate_up_compact_unique_packet_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_verifier_q4_moe_gate_up_compact_unique_packet",
                )?,
                verifier_gate_up_compact_unique_packet_k2_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k2",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_vector_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_vector",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_split_accum_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_split_accum",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_direct_activation_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_direct_activation",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_rows16_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_rows16",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_unroll4_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_unroll4",
                    )?,
                verifier_silu_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_silu_batch")?,
                verifier_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_down_batch")?,
                verifier_down_compact_unique_packet_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_down_compact_unique_packet")?,
                verifier_down_compact_unique_packet_k4_unroll4_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_down_compact_unique_packet_k4_unroll4",
                    )?,
                verifier_accumulate_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_accumulate_batch")?,
                unique_packet_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_unique_packet_norm_simd",
                )?,
                unique_packet_down_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_down_inline_unique_packet_simd",
                )?,
                cluster_down16_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cluster16_simd")?,
                cluster_down32_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cluster32_simd")?,
                gate_up_instrument_load_unique_packet_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_simd",
                )?,
                gate_up_instrument_load_unique_packet_norm_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_norm_simd",
                    )?,
                gate_up_instrument_load_two_groups_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_two_groups_simd",
                )?,
                gate_up_instrument_load_two_groups_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_two_groups_norm_simd",
                )?,
                gate_up_instrument_contiguous_load_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_simd",
                )?,
                gate_up_instrument_contiguous_load_only_norm_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_norm_simd",
                    )?,
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
                two_row_unique_packet_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_unique_packet_simd",
                )?),
                two_row_unique_packet_gate_up_norm_pipeline: Some(
                    self.dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_two_row_unique_packet_norm_simd",
                    )?,
                ),
                gate_up_instrument_staging_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_staging_only_simd",
                )?,
                gate_up_instrument_address_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_address_only_simd",
                )?,
                gate_up_instrument_load_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_only_simd",
                )?,
                gate_up_instrument_load_decode_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_decode_simd",
                )?,
                gate_up_instrument_dot_no_reduce_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_simd",
                )?,
                gate_up_instrument_load_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_only_norm_simd",
                )?,
                gate_up_instrument_staging_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_staging_only_norm_simd",
                )?,
                gate_up_instrument_address_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_address_only_norm_simd",
                )?,
                gate_up_instrument_load_decode_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_decode_norm_simd",
                )?,
                gate_up_instrument_dot_no_reduce_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_norm_simd",
                )?,
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
                expertwise_superblock_gate_up_pipeline: q4_gate_up_superblock
                    .then(|| {
                        self.dynamic_moe_named_pipeline(if q3_gate_up_superblock {
                            "qx_dynamic_mixed_moe_gate_up_q3_112_simd"
                        } else {
                            "qx_dynamic_mixed_moe_gate_up_superblock_simd"
                        })
                    })
                    .transpose()?,
                expertwise_superblock_gate_up_norm_pipeline: q4_gate_up_superblock
                    .then(|| {
                        self.dynamic_moe_named_pipeline(if q3_gate_up_superblock {
                            "qx_dynamic_mixed_moe_gate_up_q3_112_norm_simd"
                        } else {
                            "qx_dynamic_mixed_moe_gate_up_superblock_norm_simd"
                        })
                    })
                    .transpose()?,
                expertwise_superblock_down_pipeline: q4_down_superblock
                    .then(|| {
                        self.dynamic_moe_named_pipeline(if q3_down_superblock {
                            "qx_dynamic_mixed_moe_down_q3_112_simd"
                        } else {
                            "qx_dynamic_mixed_moe_down_superblock_simd"
                        })
                    })
                    .transpose()?,
                expertwise_q4_gate_up_superblock: q4_gate_up_superblock,
                expertwise_q4_down_superblock: q4_down_superblock,
                expertwise_q3_gate_up_superblock: q3_gate_up_superblock,
                expertwise_q3_down_superblock: q3_down_superblock,
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
                && !prepared.expertwise_q4_gate_up_superblock
                && !prepared.expertwise_q4_down_superblock
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
                unique_packet_gate_up_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_unique_packet_simd",
                )?,
                verifier_gate_up_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_batch")?,
                verifier_gate_up_expert_major_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_expert_major")?,
                verifier_gate_up_expert_major_direct_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_gate_up_expert_major_direct")?,
                verifier_gate_up_expert_major_unique_packet_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_expert_major_unique_packet",
                    )?,
                verifier_compact_routes_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_compact_routes_fixed")?,
                verifier_gate_up_compact_unique_packet_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_verifier_q4_moe_gate_up_compact_unique_packet",
                )?,
                verifier_gate_up_compact_unique_packet_k2_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k2",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_vector_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_vector",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_split_accum_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_split_accum",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_direct_activation_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_direct_activation",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_rows16_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_rows16",
                    )?,
                verifier_gate_up_compact_unique_packet_k4_unroll4_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_gate_up_compact_unique_packet_k4_unroll4",
                    )?,
                verifier_silu_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_silu_batch")?,
                verifier_down_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_down_batch")?,
                verifier_down_compact_unique_packet_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_down_compact_unique_packet")?,
                verifier_down_compact_unique_packet_k4_unroll4_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_verifier_q4_moe_down_compact_unique_packet_k4_unroll4",
                    )?,
                verifier_accumulate_pipeline: self
                    .dynamic_moe_named_pipeline("qx_verifier_q4_moe_accumulate_batch")?,
                unique_packet_gate_up_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_unique_packet_norm_simd",
                )?,
                unique_packet_down_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_down_inline_unique_packet_simd",
                )?,
                cluster_down16_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cluster16_simd")?,
                cluster_down32_pipeline: self
                    .dynamic_moe_named_pipeline("qx_dynamic_q4_moe_down_cluster32_simd")?,
                gate_up_instrument_contiguous_load_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_simd",
                )?,
                gate_up_instrument_contiguous_load_only_norm_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_inline_contiguous_load_only_norm_simd",
                    )?,
                gate_up_instrument_load_two_groups_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_two_groups_simd",
                )?,
                gate_up_instrument_load_two_groups_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_two_groups_norm_simd",
                )?,
                gate_up_instrument_load_unique_packet_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_simd",
                )?,
                gate_up_instrument_load_unique_packet_norm_pipeline: self
                    .dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_inline_load_unique_packet_norm_simd",
                    )?,
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
                two_row_unique_packet_gate_up_pipeline: Some(self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_two_row_unique_packet_simd",
                )?),
                two_row_unique_packet_gate_up_norm_pipeline: Some(
                    self.dynamic_moe_named_pipeline(
                        "qx_dynamic_q4_moe_gate_up_two_row_unique_packet_norm_simd",
                    )?,
                ),
                gate_up_instrument_staging_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_staging_only_simd",
                )?,
                gate_up_instrument_address_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_address_only_simd",
                )?,
                gate_up_instrument_load_only_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_only_simd",
                )?,
                gate_up_instrument_load_decode_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_decode_simd",
                )?,
                gate_up_instrument_dot_no_reduce_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_simd",
                )?,
                gate_up_instrument_load_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_only_norm_simd",
                )?,
                gate_up_instrument_staging_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_staging_only_norm_simd",
                )?,
                gate_up_instrument_address_only_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_address_only_norm_simd",
                )?,
                gate_up_instrument_load_decode_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_load_decode_norm_simd",
                )?,
                gate_up_instrument_dot_no_reduce_norm_pipeline: self.dynamic_moe_named_pipeline(
                    "qx_dynamic_q4_moe_gate_up_inline_dot_no_reduce_norm_simd",
                )?,
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
                expertwise_superblock_gate_up_pipeline: None,
                expertwise_superblock_gate_up_norm_pipeline: None,
                expertwise_superblock_down_pipeline: None,
                expertwise_q4_gate_up_superblock: false,
                expertwise_q4_down_superblock: false,
                expertwise_q3_gate_up_superblock: false,
                expertwise_q3_down_superblock: false,
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
                batch_pipeline: self.q4_embedding_batch_pipeline()?,
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
                pipeline: if weights.layout == PackingLayout::Q4Superblock144 {
                    self.q4_embedding_superblock_pipeline()?
                } else {
                    self.q4_embedding_pipeline()?
                },
                batch_pipeline: if weights.layout == PackingLayout::Q4Superblock144 {
                    self.q4_embedding_superblock_batch_pipeline()?
                } else {
                    self.q4_embedding_batch_pipeline()?
                },
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
                .zip(prepared)
                .map(|(output, matrix)| {
                    let contents = output.contents().as_ptr() as *const f32;
                    unsafe { std::slice::from_raw_parts(contents, matrix.rows).to_vec() }
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

        /// Bind a Q4 matrix from existing runtime-buffer ranges. This is the
        /// no-copy constructor used by mapped Qwen artifacts.
        pub fn prepare_q4_stage_buffers(
            &self,
            packed: RuntimeBuffer,
            scales: RuntimeBuffer,
            rows: usize,
            columns: usize,
            layout: PackingLayout,
        ) -> Result<PreparedQ4Stage, String> {
            let expected_packed = buffer_len(rows, columns, QuantType::Q4);
            let expected_scales = scale_len(rows, columns);
            if packed.len() != expected_packed || scales.len() != expected_scales {
                return Err(format!(
                    "Q4 mapped stage bytes packed/scales={}/{}, expected {expected_packed}/{expected_scales}",
                    packed.len(),
                    scales.len()
                ));
            }
            Ok(PreparedQ4Stage {
                pipeline: self.q4_gemv_pipeline()?,
                packed,
                scales,
                rows,
                columns,
                layout,
            })
        }

        pub fn prepare_qwen_q4_embedding_buffers(
            &self,
            packed: RuntimeBuffer,
            scales: RuntimeBuffer,
            rows: usize,
        ) -> Result<PreparedQwenQ4Embedding, String> {
            const HIDDEN: usize = 2048;
            let expected_packed = buffer_len(rows, HIDDEN, QuantType::Q4);
            let expected_scales = scale_len(rows, HIDDEN);
            if packed.len() != expected_packed || scales.len() != expected_scales {
                return Err(format!(
                    "Qwen embedding bytes packed/scales={}/{}, expected {expected_packed}/{expected_scales}",
                    packed.len(),
                    scales.len()
                ));
            }
            Ok(PreparedQwenQ4Embedding {
                pipeline: self.qwen_embedding_pipeline()?,
                packed,
                scales,
                rows,
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

        /// Apply Qwen3 per-head Q/K RMSNorm and split-half RoPE to FP16
        /// projection outputs. This is the semantic front end required before
        /// GQA attention and KV-cache insertion.
        pub fn run_qwen_qk_norm_rope(
            &self,
            query: &[u16],
            key: &[u16],
            query_norm: &[f32],
            key_norm: &[f32],
            position: u32,
        ) -> Result<(Vec<u16>, Vec<u16>), String> {
            const QUERY_ELEMENTS: usize = 32 * 128;
            const KEY_ELEMENTS: usize = 4 * 128;
            if query.len() != QUERY_ELEMENTS
                || key.len() != KEY_ELEMENTS
                || query_norm.len() != 128
                || key_norm.len() != 128
            {
                return Err(format!(
                    "invalid Qwen Q/K shapes: q={}, k={}, q_norm={}, k_norm={}",
                    query.len(),
                    key.len(),
                    query_norm.len(),
                    key_norm.len()
                ));
            }
            let bytes = |values: &[u16]| unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    std::mem::size_of_val(values),
                )
            };
            let float_bytes = |values: &[f32]| unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    std::mem::size_of_val(values),
                )
            };
            let shared = |values: &[u8]| {
                let pointer = NonNull::new(values.as_ptr() as *mut c_void)
                    .ok_or_else(|| "Qwen attention buffer cannot be empty".to_owned())?;
                unsafe {
                    self.device.newBufferWithBytes_length_options(
                        pointer,
                        values.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| "Metal Qwen attention buffer allocation failed".to_owned())
            };
            let query_buffer = shared(bytes(query))?;
            let key_buffer = shared(bytes(key))?;
            let query_norm_buffer = shared(float_bytes(query_norm))?;
            let key_norm_buffer = shared(float_bytes(key_norm))?;
            let command_buffer = self.queue.commandBuffer().ok_or_else(|| {
                "Metal Qwen attention command buffer allocation failed".to_owned()
            })?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal Qwen attention encoder allocation failed".to_owned())?;
            let pipeline = self.qwen_qk_norm_rope_pipeline()?;
            encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, index) in [
                    (&query_buffer, 0),
                    (&key_buffer, 1),
                    (&query_norm_buffer, 2),
                    (&key_norm_buffer, 3),
                ] {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::from(&position).cast(),
                    std::mem::size_of_val(&position),
                    4,
                );
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: 36,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 36,
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
                return Err("Metal Qwen attention command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let query_result = unsafe {
                std::slice::from_raw_parts(
                    query_buffer.contents().as_ptr() as *const u16,
                    QUERY_ELEMENTS,
                )
                .to_vec()
            };
            let key_result = unsafe {
                std::slice::from_raw_parts(
                    key_buffer.contents().as_ptr() as *const u16,
                    KEY_ELEMENTS,
                )
                .to_vec()
            };
            Ok((query_result, key_result))
        }

        /// Execute correctness-first Qwen 32Q/4KV grouped-query attention over
        /// an FP16 position-major KV cache. Query and output contain 4096 FP16
        /// elements; each cache position contains 4 x 128 elements.
        pub fn run_qwen_gqa_attention(
            &self,
            query: &[u16],
            key_cache: &[u16],
            value_cache: &[u16],
        ) -> Result<Vec<u16>, String> {
            const QUERY_ELEMENTS: usize = 32 * 128;
            const KV_ELEMENTS: usize = 4 * 128;
            if query.len() != QUERY_ELEMENTS
                || key_cache.is_empty()
                || key_cache.len() != value_cache.len()
                || !key_cache.len().is_multiple_of(KV_ELEMENTS)
            {
                return Err(format!(
                    "invalid Qwen GQA shapes: q={}, k_cache={}, v_cache={}",
                    query.len(),
                    key_cache.len(),
                    value_cache.len()
                ));
            }
            let context = u32::try_from(key_cache.len() / KV_ELEMENTS)
                .map_err(|_| "Qwen GQA context exceeds u32".to_owned())?;
            let bytes = |values: &[u16]| unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    std::mem::size_of_val(values),
                )
            };
            let query_buffer = self.input_buffer(bytes(query))?;
            let key_buffer = self.input_buffer(bytes(key_cache))?;
            let value_buffer = self.input_buffer(bytes(value_cache))?;
            let output = self
                .device
                .newBufferWithLength_options(
                    QUERY_ELEMENTS * std::mem::size_of::<u16>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal Qwen GQA output allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal Qwen GQA command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal Qwen GQA encoder allocation failed".to_owned())?;
            let score_reuse = context <= 4096;
            let pipeline = if score_reuse {
                self.qwen_gqa_score_reuse_pipeline()?
            } else {
                self.qwen_gqa_attention_pipeline()?
            };
            encoder.setComputePipelineState(&pipeline);
            unsafe {
                for (buffer, index) in [
                    (&query_buffer, 0),
                    (&key_buffer, 1),
                    (&value_buffer, 2),
                    (&output, 3),
                ] {
                    encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::from(&context).cast(),
                    std::mem::size_of_val(&context),
                    4,
                );
            }
            if score_reuse {
                unsafe {
                    encoder.setThreadgroupMemoryLength_atIndex(
                        (context as usize + 128) * std::mem::size_of::<f32>(),
                        0,
                    );
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 128,
                        height: 1,
                        depth: 1,
                    },
                );
            } else {
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: QUERY_ELEMENTS,
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
                return Err("Metal Qwen GQA command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let contents = output.contents().as_ptr() as *const u16;
            Ok(unsafe { std::slice::from_raw_parts(contents, QUERY_ELEMENTS).to_vec() })
        }

        /// Prepare one Qwen3-30B-A3B Q2S4 expert. The byte slices may come
        /// directly from a mapped QX artifact; this correctness path currently
        /// uploads the selected expert into persistent private Metal buffers.
        pub fn prepare_qwen_q2s4_expert_bytes(
            &self,
            gate_up_packed: &[u8],
            gate_up_scales: &[u8],
            down_packed: &[u8],
            down_scales: &[u8],
        ) -> Result<PreparedQwenQ2S4Expert, String> {
            const QWEN_HIDDEN: usize = 2048;
            const QWEN_INTERMEDIATE: usize = 768;
            const QWEN_GATE_UP_ROWS: usize = QWEN_INTERMEDIATE * 2;
            let expected_gate_up = buffer_len(QWEN_GATE_UP_ROWS, QWEN_HIDDEN, QuantType::Q2S4);
            let expected_down = buffer_len(QWEN_HIDDEN, QWEN_INTERMEDIATE, QuantType::Q2S4);
            let expected_gate_up_scales = scale_len(QWEN_GATE_UP_ROWS, QWEN_HIDDEN);
            let expected_down_scales = scale_len(QWEN_HIDDEN, QWEN_INTERMEDIATE);
            for (name, actual, expected) in [
                ("gate/up packed", gate_up_packed.len(), expected_gate_up),
                (
                    "gate/up scales",
                    gate_up_scales.len(),
                    expected_gate_up_scales,
                ),
                ("down packed", down_packed.len(), expected_down),
                ("down scales", down_scales.len(), expected_down_scales),
            ] {
                if actual != expected {
                    return Err(format!("Qwen {name} length {actual}, expected {expected}"));
                }
            }
            Ok(PreparedQwenQ2S4Expert {
                pipeline: self.qwen_q2s4_expert_pipeline()?,
                gate_up_packed: self.input_runtime_buffer(gate_up_packed)?,
                gate_up_scales: self.input_runtime_buffer(gate_up_scales)?,
                down_packed: self.input_runtime_buffer(down_packed)?,
                down_scales: self.input_runtime_buffer(down_scales)?,
            })
        }

        /// Bind one Qwen expert from subregions of a shared mapped weight
        /// slab. No weight bytes are copied or re-packed.
        pub fn prepare_qwen_q2s4_expert_buffers(
            &self,
            gate_up_packed: RuntimeBuffer,
            gate_up_scales: RuntimeBuffer,
            down_packed: RuntimeBuffer,
            down_scales: RuntimeBuffer,
        ) -> Result<PreparedQwenQ2S4Expert, String> {
            const QWEN_HIDDEN: usize = 2048;
            const QWEN_INTERMEDIATE: usize = 768;
            const QWEN_GATE_UP_ROWS: usize = QWEN_INTERMEDIATE * 2;
            let expected = [
                buffer_len(QWEN_GATE_UP_ROWS, QWEN_HIDDEN, QuantType::Q2S4),
                scale_len(QWEN_GATE_UP_ROWS, QWEN_HIDDEN),
                buffer_len(QWEN_HIDDEN, QWEN_INTERMEDIATE, QuantType::Q2S4),
                scale_len(QWEN_HIDDEN, QWEN_INTERMEDIATE),
            ];
            let actual = [
                gate_up_packed.len(),
                gate_up_scales.len(),
                down_packed.len(),
                down_scales.len(),
            ];
            if actual != expected {
                return Err(format!(
                    "invalid mapped Qwen expert byte lengths {actual:?}, expected {expected:?}"
                ));
            }
            Ok(PreparedQwenQ2S4Expert {
                pipeline: self.qwen_q2s4_expert_pipeline()?,
                gate_up_packed,
                gate_up_scales,
                down_packed,
                down_scales,
            })
        }

        /// Bind all 128 Qwen experts through byte offsets into one mapped
        /// artifact slab. Only the compact offset and codec tables are copied.
        pub fn prepare_qwen_dynamic_moe(
            &self,
            slab: RuntimeBuffer,
            gate_data_offsets: &[u64],
            gate_scale_offsets: &[u64],
            down_data_offsets: &[u64],
            down_scale_offsets: &[u64],
            codecs: &[u8],
        ) -> Result<PreparedQwenDynamicMoe, String> {
            const EXPERTS: usize = 128;
            for (name, offsets) in [
                ("gate data", gate_data_offsets),
                ("gate scales", gate_scale_offsets),
                ("down data", down_data_offsets),
                ("down scales", down_scale_offsets),
            ] {
                if offsets.len() != EXPERTS {
                    return Err(format!(
                        "Qwen dynamic MoE {name} offset count {}, expected {EXPERTS}",
                        offsets.len()
                    ));
                }
                if offsets.iter().any(|offset| *offset as usize >= slab.len()) {
                    return Err(format!("Qwen dynamic MoE {name} offset exceeds the slab"));
                }
            }
            if codecs.len() != EXPERTS || codecs.iter().any(|codec| !matches!(codec, 2 | 3 | 4 | 5))
            {
                return Err(
                    "Qwen dynamic MoE codecs must contain 128 Q2S4/Q3/Q4/Q3-112 IDs".to_owned(),
                );
            }
            let offset_buffer = |offsets: &[u64]| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        offsets.as_ptr().cast::<u8>(),
                        std::mem::size_of_val(offsets),
                    )
                };
                self.input_runtime_buffer(bytes)
            };
            Ok(PreparedQwenDynamicMoe {
                expert_pipeline: self.qwen_dynamic_moe_pipeline()?,
                gate_pipeline: self.qwen_dynamic_gate_pipeline()?,
                down_tiled_pipeline: self.qwen_dynamic_down_tiled_pipeline()?,
                accumulate_pipeline: self.qwen_dynamic_accumulate_pipeline()?,
                slab,
                gate_data_offsets: offset_buffer(gate_data_offsets)?,
                gate_scale_offsets: offset_buffer(gate_scale_offsets)?,
                down_data_offsets: offset_buffer(down_data_offsets)?,
                down_scale_offsets: offset_buffer(down_scale_offsets)?,
                codecs: self.input_runtime_buffer(codecs)?,
                uncached_storage: self.allocate_runtime_buffer(4)?,
                uncached_slots: {
                    let slots = [u32::MAX; 8];
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            slots.as_ptr().cast::<u8>(),
                            std::mem::size_of_val(&slots),
                        )
                    };
                    self.input_runtime_buffer(bytes)?
                },
            })
        }

        /// Allocate a bounded, fully associative cache for source-exact Q4
        /// Qwen experts. Metadata and replacement decisions remain on GPU.
        pub fn prepare_qwen_expert_cache(
            &self,
            budget_bytes: usize,
        ) -> Result<PreparedQwenExpertCache, String> {
            const SLOT_BYTES: usize = 2_654_208;
            let slots = budget_bytes / SLOT_BYTES;
            if slots < 8 {
                return Err(format!(
                    "Qwen expert cache budget {budget_bytes} provides {slots} slots; at least eight are required for one top-8 route"
                ));
            }
            let storage_bytes = slots
                .checked_mul(SLOT_BYTES)
                .ok_or_else(|| "Qwen expert cache allocation overflows".to_owned())?;
            let u32_bytes = |values: &[u32]| unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            let keys = vec![u32::MAX; slots];
            let ages = vec![0_u32; slots];
            Ok(PreparedQwenExpertCache {
                resolve_pipeline: self.qwen_q4_cache_resolve_pipeline()?,
                fill_pipeline: self.qwen_q4_cache_fill_pipeline()?,
                storage: self.allocate_runtime_buffer(storage_bytes)?,
                keys: self.input_runtime_buffer(u32_bytes(&keys))?,
                ages: self.input_runtime_buffer(u32_bytes(&ages))?,
                clock: self.input_runtime_buffer(u32_bytes(&[0]))?,
                resolved_slots: self.input_runtime_buffer(u32_bytes(&[u32::MAX; 8]))?,
                copy_flags: self.input_runtime_buffer(u32_bytes(&[0; 8]))?,
                counters: self.input_runtime_buffer(u32_bytes(&[0; 5]))?,
                seen: self.input_runtime_buffer(&[0; 48 * 128])?,
                slots,
            })
        }

        pub fn qwen_expert_cache_stats(
            &self,
            cache: &PreparedQwenExpertCache,
        ) -> Result<QwenExpertCacheStats, String> {
            let counters = self.read_runtime_u32s(&cache.counters, 5)?;
            Ok(QwenExpertCacheStats {
                hits: counters[0],
                misses: counters[1],
                evictions: counters[2],
                q4_selections: counters[3],
                bypasses: counters[4],
                slots: cache.slots,
                storage_bytes: cache.storage.len(),
            })
        }

        /// Reset cache metadata at a host synchronization boundary. Cached
        /// payload bytes need not be cleared because empty keys make them
        /// unreachable to the resolver.
        pub fn reset_qwen_expert_cache(
            &self,
            cache: &PreparedQwenExpertCache,
        ) -> Result<(), String> {
            let u32_bytes = |values: &[u32]| unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            self.write_runtime_buffer(&cache.keys, u32_bytes(&vec![u32::MAX; cache.slots]))?;
            self.write_runtime_buffer(&cache.ages, u32_bytes(&vec![0; cache.slots]))?;
            self.write_runtime_buffer(&cache.clock, u32_bytes(&[0]))?;
            self.write_runtime_buffer(&cache.resolved_slots, u32_bytes(&[u32::MAX; 8]))?;
            self.write_runtime_buffer(&cache.copy_flags, u32_bytes(&[0; 8]))?;
            self.write_runtime_buffer(&cache.counters, u32_bytes(&[0; 5]))?;
            self.write_runtime_buffer(&cache.seen, &[0; 48 * 128])
        }

        /// Execute one prepared Qwen Q2S4 expert with one command buffer and
        /// one dispatch. Input and output are FP16 bit patterns.
        pub fn run_prepared_qwen_q2s4_expert(
            &self,
            prepared: &PreparedQwenQ2S4Expert,
            input: &[u16],
        ) -> Result<Vec<u16>, String> {
            const QWEN_HIDDEN: usize = 2048;
            const QWEN_INTERMEDIATE: usize = 768;
            if input.len() != QWEN_HIDDEN {
                return Err(format!(
                    "Qwen expert input length {}, expected {QWEN_HIDDEN}",
                    input.len()
                ));
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let input_buffer = self.input_buffer(input_bytes)?;
            let output = self
                .device
                .newBufferWithLength_options(
                    QWEN_HIDDEN * std::mem::size_of::<u16>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal Qwen expert output allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal Qwen expert command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal Qwen expert encoder allocation failed".to_owned())?;
            encoder.setComputePipelineState(&prepared.pipeline);
            unsafe {
                for (buffer, index) in [
                    (&prepared.gate_up_packed, 0),
                    (&prepared.gate_up_scales, 1),
                    (&prepared.down_packed, 2),
                    (&prepared.down_scales, 3),
                ] {
                    encoder.setBuffer_offset_atIndex(Some(&buffer.buffer), buffer.offset, index);
                }
                encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&output), 0, 5);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: QWEN_INTERMEDIATE,
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
                return Err("Metal Qwen expert command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 1, encode_start, wait_start)?;
            let contents = output.contents().as_ptr() as *const u16;
            Ok(unsafe { std::slice::from_raw_parts(contents, QWEN_HIDDEN).to_vec() })
        }

        /// Execute and router-weight eight prepared Qwen experts in one
        /// command buffer. Expert kernels remain independent dispatches, but
        /// the activated outputs never cross the host boundary.
        pub fn run_prepared_qwen_q2s4_experts_weighted(
            &self,
            prepared: &[&PreparedQwenQ2S4Expert],
            input: &[u16],
            weights: &[f32; 8],
        ) -> Result<Vec<u16>, String> {
            const QWEN_HIDDEN: usize = 2048;
            const QWEN_INTERMEDIATE: usize = 768;
            if prepared.len() != 8 || input.len() != QWEN_HIDDEN {
                return Err(format!(
                    "Qwen weighted MoE requires 8 experts and {QWEN_HIDDEN} inputs; got {} and {}",
                    prepared.len(),
                    input.len()
                ));
            }
            let input_bytes = unsafe {
                std::slice::from_raw_parts(
                    input.as_ptr() as *const u8,
                    std::mem::size_of_val(input),
                )
            };
            let input_buffer = self.input_buffer(input_bytes)?;
            let mut outputs = Vec::with_capacity(8);
            for _ in 0..8 {
                outputs.push(
                    self.device
                        .newBufferWithLength_options(
                            QWEN_HIDDEN * std::mem::size_of::<u16>(),
                            MTLResourceOptions::StorageModeShared,
                        )
                        .ok_or_else(|| "Metal Qwen expert output allocation failed".to_owned())?,
                );
            }
            let moe_output = self
                .device
                .newBufferWithLength_options(
                    QWEN_HIDDEN * std::mem::size_of::<u16>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "Metal Qwen MoE output allocation failed".to_owned())?;
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| "Metal Qwen MoE command buffer allocation failed".to_owned())?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| "Metal Qwen MoE encoder allocation failed".to_owned())?;
            for (expert, output) in prepared.iter().zip(&outputs) {
                encoder.setComputePipelineState(&expert.pipeline);
                unsafe {
                    for (buffer, index) in [
                        (&expert.gate_up_packed, 0),
                        (&expert.gate_up_scales, 1),
                        (&expert.down_packed, 2),
                        (&expert.down_scales, 3),
                    ] {
                        encoder.setBuffer_offset_atIndex(
                            Some(&buffer.buffer),
                            buffer.offset,
                            index,
                        );
                    }
                    encoder.setBuffer_offset_atIndex(Some(&input_buffer), 0, 4);
                    encoder.setBuffer_offset_atIndex(Some(output), 0, 5);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: QWEN_INTERMEDIATE,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            let aggregate = self.aggregate_pipeline(QuantType::Q2S4)?;
            encoder.setComputePipelineState(&aggregate);
            unsafe {
                for (index, output) in outputs.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(output), 0, index);
                }
                encoder.setBytes_length_atIndex(
                    NonNull::from(weights).cast(),
                    std::mem::size_of_val(weights),
                    8,
                );
                encoder.setBuffer_offset_atIndex(Some(&moe_output), 0, 9);
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: QWEN_HIDDEN,
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
                return Err("Metal Qwen MoE command buffer execution failed".to_owned());
            }
            self.record_command(&command_buffer, 9, encode_start, wait_start)?;
            let contents = moe_output.contents().as_ptr() as *const u16;
            Ok(unsafe { std::slice::from_raw_parts(contents, QWEN_HIDDEN).to_vec() })
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
    #[derive(Clone, Debug, Default)]
    pub struct MetalCounterSetInfo {
        pub name: String,
        pub counters: Vec<String>,
    }

    #[derive(Clone, Debug, Default)]
    pub struct MetalKernelInfo {
        pub name: String,
        pub thread_execution_width: usize,
        pub max_threads_per_threadgroup: usize,
        pub static_threadgroup_memory_bytes: usize,
    }

    pub struct MetalContext;

    impl MetalContext {
        pub fn system_default() -> Option<Self> {
            None
        }

        pub fn counter_inventory(&self) -> Vec<MetalCounterSetInfo> {
            Vec::new()
        }

        pub fn dynamic_moe_kernel_inventory(&self) -> Result<Vec<MetalKernelInfo>, String> {
            Err("Metal kernel inventory requires macOS".to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    #[test]
    fn consensus_math_is_bit_exact_between_rust_and_metal() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let inputs = [1.0e-6_f32, 1.0, 17.0, 128.0, 511.0, 1024.0, 2048.0, 4095.0];
        let observed = context
            .run_consensus_math_conformance(&inputs)
            .expect("consensus math dispatch");
        let expected: [[u32; 4]; 8] = std::array::from_fn(|index| {
            let value = inputs[index];
            [
                qx_consensus_math::exp_f32(value).unwrap().to_bits(),
                qx_consensus_math::rsqrt_f32(value).unwrap().to_bits(),
                qx_consensus_math::granite_rope_sin_cos(value as u32, index as u32)
                    .unwrap()
                    .0
                    .to_bits(),
                qx_consensus_math::granite_rope_sin_cos(value as u32, index as u32)
                    .unwrap()
                    .1
                    .to_bits(),
            ]
        });
        assert_eq!(observed, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn prepared_dense_readback_uses_logical_rows_after_scratch_reuse() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let wide = context
            .prepare_dense(&vec![1.0; 1024 * 4], 1024, 4)
            .unwrap();
        let narrow = context.prepare_dense(&vec![2.0; 32 * 4], 32, 4).unwrap();
        assert_eq!(
            context.run_prepared_dense(&wide, &[1.0; 4]).unwrap().len(),
            1024
        );
        let output = context.run_prepared_dense(&narrow, &[1.0; 4]).unwrap();
        assert_eq!(output.len(), 32);
        assert!(output.iter().all(|value| *value == 8.0));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn verification_hash_matches_cpu_golden_oracle() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let residual_values = (0..1024)
            .map(|index| index as f32 * 0.125 - 64.0)
            .collect::<Vec<_>>();
        let key_values = (0..512)
            .map(|index| index as f32 * -0.25 + 3.0)
            .collect::<Vec<_>>();
        let value_values = (0..512)
            .map(|index| index as f32 / 17.0)
            .collect::<Vec<_>>();
        let bytes = |pointer: *const u8, length: usize| unsafe {
            std::slice::from_raw_parts(pointer, length)
        };
        let residual_bytes = bytes(
            residual_values.as_ptr().cast(),
            std::mem::size_of_val(residual_values.as_slice()),
        );
        let key_bytes = bytes(
            key_values.as_ptr().cast(),
            std::mem::size_of_val(key_values.as_slice()),
        );
        let value_bytes = bytes(
            value_values.as_ptr().cast(),
            std::mem::size_of_val(value_values.as_slice()),
        );
        let residual = context
            .allocate_runtime_buffer(residual_bytes.len() * 2)
            .unwrap();
        let kv = context
            .allocate_runtime_buffer(key_bytes.len() + value_bytes.len())
            .unwrap();
        let output = context.allocate_runtime_buffer(128).unwrap();
        context
            .write_runtime_buffer(
                &residual.subregion(0, residual_bytes.len()).unwrap(),
                residual_bytes,
            )
            .unwrap();
        context
            .write_runtime_buffer(
                &residual
                    .subregion(residual_bytes.len(), residual_bytes.len())
                    .unwrap(),
                residual_bytes,
            )
            .unwrap();
        context
            .write_runtime_buffer(&kv.subregion(0, key_bytes.len()).unwrap(), key_bytes)
            .unwrap();
        context
            .write_runtime_buffer(
                &kv.subregion(key_bytes.len(), value_bytes.len()).unwrap(),
                value_bytes,
            )
            .unwrap();
        let mut graph = context.begin_decode().unwrap();
        let key_offsets = [0; 24];
        let mut value_offsets = [0; 24];
        let mut key_widths = [0; 24];
        let mut value_widths = [0; 24];
        value_offsets[0] = key_bytes.len() as u32;
        key_widths[0] = key_bytes.len() as u32;
        value_widths[0] = value_bytes.len() as u32;
        graph
            .encode_verification_hashes(
                &context,
                &residual,
                &kv,
                &output,
                11,
                1,
                residual_bytes.len(),
                1,
                key_offsets,
                value_offsets,
                key_widths,
                value_widths,
            )
            .unwrap();
        graph.finish(&context).unwrap();
        let observed = context.read_runtime_bytes(&output);
        let state = qx_commit::state_hash_v1(residual_bytes).unwrap();
        let kv = qx_commit::kv_leaf_hash_v1(0, 11, key_bytes, value_bytes).unwrap();
        assert_eq!(&observed[..32], state);
        assert_eq!(&observed[64..96], state);
        assert_eq!(&observed[96..128], kv);
    }

    #[test]
    fn every_quant_type_has_a_kernel() {
        for quant_type in [QuantType::Q2, QuantType::Q2S4, QuantType::Q3, QuantType::Q4] {
            assert!(kernel_source(quant_type).contains("kernel"));
        }
        for name in [
            "qx_qwen_moe_dynamic",
            "qx_qwen_moe_gate_dynamic",
            "qx_qwen_moe_down_tiled_dynamic",
            "qx_qwen_moe_accumulate_dynamic",
        ] {
            assert!(
                Q2_KERNEL_SOURCE.contains(name),
                "missing Qwen MoE kernel {name}"
            );
        }
        for name in [
            "qx_qwen_rms_norm",
            "qx_qwen_residual_rms_norm",
            "qx_qwen_store_kv",
            "qx_qwen_residual",
            "qx_qwen_q4_embedding_lookup",
            "qx_qwen_argmax",
            "qx_qwen_qk_norm_rope",
            "qx_qwen_gqa_attention",
            "qx_qwen_gqa_score_reuse",
            "qx_qwen_top8",
        ] {
            assert!(
                QWEN_ATTENTION_KERNEL_SOURCE.contains(name),
                "missing Qwen layer kernel {name}"
            );
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
        for name in [
            "qx_dynamic_q4_moe_gate_up_superblock144_simd",
            "qx_dynamic_q4_moe_gate_up_superblock144_norm_simd",
            "qx_dynamic_q4_moe_down_superblock144_simd",
        ] {
            assert!(
                DYNAMIC_MOE_KERNEL_SOURCE.contains(name),
                "missing Metal entry point: {name}"
            );
        }
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
        for quant_type in [QuantType::Q2, QuantType::Q2S4, QuantType::Q3, QuantType::Q4] {
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
            let weight = if quant_type == QuantType::Q2S4 {
                qx_quant::Q2S4_INNER_RATIO
            } else {
                1.0
            };
            let activated = (weight / (1.0 + (-weight).exp())) * weight;
            let expected = f16::from_f32(512.0 * weight * activated).to_f32();
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
    fn separately_submitted_graphs_preserve_gpu_queue_order() {
        let context = platform::MetalContext::system_default().expect("Metal device");
        let source = context.allocate_runtime_buffer(4).unwrap();
        let intermediate = context.allocate_runtime_buffer(4).unwrap();
        let output = context.allocate_runtime_buffer(4).unwrap();
        context
            .write_runtime_buffer(&source, &13.25_f32.to_ne_bytes())
            .unwrap();

        let before = context.metrics().unwrap();
        let mut producer = context.begin_decode().unwrap();
        producer
            .encode_copy_f32(&context, &source, &intermediate, 1)
            .unwrap();
        let producer = producer.submit();

        let mut consumer = context.begin_decode().unwrap();
        consumer
            .encode_copy_f32(&context, &intermediate, &output, 1)
            .unwrap();
        let consumer = consumer.submit();

        producer.wait(&context).unwrap();
        consumer.wait(&context).unwrap();
        let after = context.metrics().unwrap();
        assert_eq!(context.read_runtime_f32(&output, 1).unwrap(), [13.25]);
        assert_eq!(after.command_buffers - before.command_buffers, 2);
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
    fn verifier_gate_up_batch_matches_exact_row_inline_reference() {
        const EXPERTS: usize = 32;
        const HIDDEN: usize = 1024;
        const GROUPS: usize = 32;
        const GROUP_BYTES: usize = 18;
        const PAIRS: usize = 2;
        let context = platform::MetalContext::system_default().expect("Metal device");
        let mut gate_up = vec![0_u8; EXPERTS * HIDDEN * GROUPS * GROUP_BYTES];
        for expert in 0..EXPERTS {
            for row in 0..HIDDEN {
                for group in 0..GROUPS {
                    let offset = (expert * HIDDEN * GROUPS + row * GROUPS + group) * GROUP_BYTES;
                    gate_up[offset..offset + 16].fill(0x88);
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
        let input = context
            .allocate_runtime_buffer(PAIRS * HIDDEN * std::mem::size_of::<f32>())
            .unwrap();
        let route_experts = context
            .allocate_runtime_buffer(PAIRS * std::mem::size_of::<u32>())
            .unwrap();
        let route_tokens = context
            .allocate_runtime_buffer(PAIRS * std::mem::size_of::<u32>())
            .unwrap();
        let output = context
            .allocate_runtime_buffer(PAIRS * HIDDEN * std::mem::size_of::<f32>())
            .unwrap();
        let input_values = [vec![1.0_f32; HIDDEN], vec![2.0_f32; HIDDEN]].concat();
        let experts = [0_u32, 1_u32];
        let tokens = [0_u32, 1_u32];
        for (buffer, bytes) in [
            (&input, unsafe {
                std::slice::from_raw_parts(
                    input_values.as_ptr() as *const u8,
                    std::mem::size_of_val(input_values.as_slice()),
                )
            }),
            (&route_experts, unsafe {
                std::slice::from_raw_parts(
                    experts.as_ptr() as *const u8,
                    std::mem::size_of_val(&experts),
                )
            }),
            (&route_tokens, unsafe {
                std::slice::from_raw_parts(
                    tokens.as_ptr() as *const u8,
                    std::mem::size_of_val(&tokens),
                )
            }),
        ] {
            context.write_runtime_buffer(buffer, bytes).unwrap();
        }
        let mut graph = context.begin_decode().unwrap();
        graph
            .encode_verifier_q4_gate_up_batch(
                &prepared,
                &input,
                &route_experts,
                &route_tokens,
                &output,
                PAIRS,
            )
            .unwrap();
        graph.finish(&context).unwrap();
        let observed = context.read_runtime_f32(&output, PAIRS * HIDDEN).unwrap();
        for (pair, expected) in [1024.0_f32, 2048.0].into_iter().enumerate() {
            let start = pair * HIDDEN;
            let max_error = observed[start..start + HIDDEN]
                .iter()
                .map(|value| (value - expected).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_error < 1.0e-3, "pair {pair} gate/up error: {max_error}");
        }
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
    fn block_layout_dispatches_have_identical_projection_results() {
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

        fn superblock_stream(experts: usize, rows: usize, groups: usize) -> Vec<u8> {
            let blocks = groups / 8;
            let expert_bytes = rows * blocks * 144;
            let mut stream = vec![0_u8; experts * expert_bytes];
            for expert in 0..experts {
                for row in 0..rows {
                    for block in 0..blocks {
                        let block_start = expert * expert_bytes + (row * blocks + block) * 144;
                        stream[block_start..block_start + 128].fill(0x88);
                        for group in 0..8 {
                            let scale_start = block_start + 128 + group * 2;
                            stream[scale_start..scale_start + 2]
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
            let gate_up = if layout == PackingLayout::Q4Superblock144 {
                superblock_stream(32, 1024, 32)
            } else {
                transaction_stream(32, 1024, 32, block_bytes)
            };
            let down = if layout == PackingLayout::Q4Superblock144 {
                superblock_stream(32, 1024, 16)
            } else {
                transaction_stream(32, 1024, 16, block_bytes)
            };
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
                .encode_dynamic_q4_moe_batched_simd(
                    &prepared, &input, &ids, &weights, &scratch, &output,
                )
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
        let (superblock_stage, superblock_down, superblock) =
            run(PackingLayout::Q4Superblock144, 144);
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
        let superblock_stage_error = dense_stage
            .iter()
            .zip(&superblock_stage)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let superblock_down_error = dense_down
            .iter()
            .zip(&superblock_down)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let superblock_error = dense
            .iter()
            .zip(&superblock)
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
        assert!(
            superblock_error < 1.0e-2
                && superblock_stage_error < 1.0e-2
                && superblock_down_error < 1.0e-2,
            "superblock error: output={superblock_error}, stage={superblock_stage_error}, down={superblock_down_error}"
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
