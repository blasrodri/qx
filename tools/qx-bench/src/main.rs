use anyhow::{Result, bail};
use clap::Parser;
use half::f16;
use qx_compiler::QuantizedExpert;
use qx_format::{MappedArtifact, TensorKind};
use qx_model::{
    EMBEDDING_MULTIPLIER, HEAD_DIM, HIDDEN, KV_HEADS, LAYERS, LOGITS_SCALING, RESIDUAL_MULTIPLIER,
    VOCAB,
};
use qx_runtime::{GenerationStats, GenerationTrace, GraniteModel, RuntimeProfile};
use qx_runtime_v2::{BufferId, ExecutionPlan};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Debug, Parser)]
#[command(
    name = "qx-bench",
    about = "Benchmark/report the compiled QX CPU reference path"
)]
struct Args {
    artifact: Option<PathBuf>,
    /// Run the full Granite runtime benchmark instead of the CPU expert check.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Compare the positional artifact against this artifact in full runtime mode.
    #[arg(long, requires = "model")]
    compare_artifact: Option<PathBuf>,
    /// Score this text prefix for perplexity in comparison mode.
    #[arg(long, requires = "model")]
    quality_text: Option<PathBuf>,
    /// JSON fixture containing canonical prompt token IDs and benchmark settings.
    #[arg(long)]
    fixture: Option<PathBuf>,
    /// Write a machine-readable benchmark or memory-attribution record.
    #[arg(long)]
    result_json: Option<PathBuf>,
    #[arg(long, default_value_t = 512)]
    quality_tokens: usize,
    /// Print per-layer/component timing for one warmed generation.
    #[arg(long)]
    profile: bool,
    #[arg(short, long, default_value = "Hello")]
    prompt: String,
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    /// Continue to the requested length even if greedy decoding emits EOS.
    #[arg(long)]
    ignore_eos: bool,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 512)]
    context: usize,
    #[arg(long, default_value_t = 100)]
    iterations: usize,
    #[arg(long, default_value_t = 0)]
    layer: u16,
    #[arg(long, default_value_t = 0)]
    expert: u8,
    /// Print the first expert coordinate using this format and exit.
    #[arg(long)]
    find_quant: Option<String>,
    /// Print artifact format counts and exit.
    #[arg(long)]
    format_summary: bool,
    /// Print the V2 static memory/dispatch audit and exit.
    #[arg(long)]
    v2_plan: bool,
    /// Attribute V2 memory ownership across load and steady-state phases.
    #[arg(long)]
    v2_memory_profile: bool,
    /// Benchmark the real 24-layer V2 attention graph without MoE/logits.
    #[arg(long, requires = "model")]
    v2_attention_stack: bool,
    /// Benchmark the real 24-layer V2 attention+MoE graph without embedding/logits.
    #[arg(long)]
    v2_full_stack: bool,
    /// Run a complete V2 token graph including embedding, LM head, and argmax.
    #[arg(long)]
    v2_token: bool,
    /// Prefill a prompt through V2, then continue with artifact-only greedy decode.
    #[arg(long)]
    v2_generate: bool,
    /// Compare V2 GPU logits against the CPU oracle using the same QX artifact.
    #[arg(long)]
    v2_parity: bool,
    /// Compare canonical V2 top-8 against the opt-in active-k V2 path on text.
    #[arg(long)]
    v2_quality: bool,
    /// Benchmark representative artifact Q4 GEMV stages on Metal.
    #[arg(long)]
    v2_kernel_bench: bool,
    /// Profile each major fixed-shape V2 GPU kernel independently.
    #[arg(long)]
    v2_kernel_profile: bool,
    /// Sweep Q-projection SIMD geometry and input-cache choices.
    #[arg(long)]
    v2_kernel_sweep: bool,
    /// Sweep fused Q/K/V SIMD geometry while retaining one dispatch.
    #[arg(long)]
    v2_qkv_sweep: bool,
    /// Replay scalar-RMS -> QKV -> RoPE/KV with real layer tensors.
    #[arg(long)]
    v2_qkv_boundary: bool,
    /// Reconstruct the transformer layer around QKV and measure interference.
    #[arg(long)]
    v2_qkv_interference: bool,
    /// Measure marginal GPU cost by ablating major blocks in the complete V2 token graph.
    #[arg(long)]
    v2_differential_profile: bool,
    /// Measure marginal GPU cost of each canonical GPU-resident MoE stage.
    #[arg(long)]
    v2_moe_differential_profile: bool,
    /// Capture contextual gate/up, fixed-route MoE, and complete-token timings.
    #[arg(long)]
    v2_moe_contextual_profile: bool,
    /// Measure fixed-route MoE after controlled GPU predecessor workloads.
    #[arg(long)]
    v2_moe_predecessor_profile: bool,
    /// Label for the candidate being measured in a contextual profile.
    #[arg(long, default_value = "canonical")]
    contextual_candidate: String,
    /// Measure top-8 expert-set overlap across sequential artifact-only tokens.
    #[arg(long)]
    v2_route_overlap: bool,
    /// Replay hot, realistic, and low-overlap route sequences against the live MoE graph.
    #[arg(long)]
    v2_route_churn_profile: bool,
    /// Capture dispatch-boundary GPU timestamps during a sequential decode.
    #[arg(long)]
    v2_sequential_timeline: bool,
    /// Compare previous-route and oracle expert traffic prefetch on the live token graph.
    #[arg(long)]
    v2_moe_prefetch_profile: bool,
    /// Compare only fixed direct-address 1/2/4/8-region gate/up controls.
    #[arg(long)]
    v2_moe_direct_region_profile: bool,
    /// Compare original, frequency-ordered, and co-selection expert permutations.
    #[arg(long)]
    v2_moe_permutation_profile: bool,
    /// Measure the fixed Metal command-buffer dispatch floor with no-op kernels.
    #[arg(long)]
    v2_dispatch_floor: bool,
    /// Sweep decode-attention context lengths on the real artifact path.
    #[arg(long)]
    v2_attention_sweep: bool,
    /// Measure real V2 layer-block sizes without final logits/sampling.
    #[arg(long)]
    v2_layer_block: bool,
    /// Split attention into benchmark-only QK, softmax, and V stages.
    #[arg(long)]
    v2_attention_stages: bool,
    /// Emit an all-token NLL plus sampled-logit quality trace from V2.
    #[arg(long)]
    v2_quality_trace: bool,
    /// Exact QXTOKN token corpus consumed by --v2-quality-trace.
    #[arg(long, requires = "v2_quality_trace")]
    quality_token_file: Option<PathBuf>,
    /// Binary QXQUAL output produced by --v2-quality-trace.
    #[arg(long, requires = "v2_quality_trace")]
    quality_trace: Option<PathBuf>,
    /// Keep full logits at every Nth position; NLL is computed at every position.
    #[arg(long, default_value_t = 256)]
    quality_sample_stride: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct KernelStats {
    gpu_median_us: f64,
    gpu_p95_us: f64,
    gpu_cv: f64,
    wall_median_us: f64,
}

struct RuntimeOptions<'a> {
    prompt: &'a str,
    max_new_tokens: usize,
    context: usize,
    warmup: usize,
    iterations: usize,
    compare_artifact: Option<&'a std::path::Path>,
    quality_text: Option<&'a std::path::Path>,
    quality_tokens: usize,
    fixture_tokens: Option<Vec<u32>>,
    profile: bool,
    ignore_eos: bool,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum MeasurementMode {
    #[default]
    SeedInclusive,
    PostPrefill,
}

#[derive(Debug, Deserialize)]
struct BenchmarkFixture {
    #[serde(default = "unnamed_fixture")]
    name: String,
    #[serde(default = "unknown_domain")]
    domain: String,
    tokens: Vec<u32>,
    context: Option<usize>,
    generation: Option<usize>,
    #[serde(default)]
    measurement: MeasurementMode,
}

fn unnamed_fixture() -> String {
    "unnamed".to_owned()
}

fn unknown_domain() -> String {
    "unknown".to_owned()
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.iterations == 0 || args.max_new_tokens == 0 {
        bail!("iterations must be positive");
    }
    if args.v2_quality_trace {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-quality-trace requires an artifact"))?;
        return emit_v2_quality_trace(
            artifact,
            args.quality_token_file
                .as_deref()
                .expect("clap quality token requirement"),
            args.quality_trace
                .as_deref()
                .expect("clap quality trace requirement"),
            args.context,
            args.quality_sample_stride,
        );
    }
    if args.v2_full_stack {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-full-stack requires an artifact path"))?;
        return benchmark_v2_full_stack(artifact, args.context, args.warmup, args.iterations);
    }
    if args.v2_memory_profile {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-memory-profile requires an artifact path"))?;
        return benchmark_v2_memory_profile(
            artifact,
            args.context,
            args.warmup,
            args.iterations,
            args.result_json.as_deref(),
        );
    }
    if args.v2_token {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-token requires an artifact path"))?;
        let fixture = args.fixture.as_deref().map(load_fixture).transpose()?;
        let context = fixture
            .as_ref()
            .and_then(|fixture| fixture.context)
            .unwrap_or(args.context);
        let iterations = fixture
            .as_ref()
            .and_then(|fixture| fixture.generation)
            .unwrap_or(args.iterations);
        return benchmark_v2_token(
            artifact,
            context,
            args.warmup,
            iterations,
            fixture.as_ref(),
            args.result_json.as_deref(),
        );
    }
    if args.v2_generate {
        let model = args
            .model
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-generate requires --model for tokenizer files"))?;
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-generate requires an artifact path"))?;
        return benchmark_v2_generate(
            model,
            artifact,
            args.prompt.as_str(),
            args.max_new_tokens,
            args.context,
        );
    }
    if args.v2_parity {
        let model = args
            .model
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-parity requires --model for tokenizer files"))?;
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-parity requires an artifact path"))?;
        return benchmark_v2_parity(model, artifact, args.prompt.as_str(), args.context);
    }
    if args.v2_quality {
        let model = args
            .model
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-quality requires --model for tokenizer files"))?;
        let quality_text = args
            .quality_text
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-quality requires --quality-text"))?;
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-quality requires an artifact path"))?;
        let active_k = std::env::var("QX_V2_MOE_ACTIVE_K")
            .ok()
            .ok_or_else(|| anyhow::anyhow!("--v2-quality requires QX_V2_MOE_ACTIVE_K=1..8"))?
            .parse::<usize>()?;
        if !(1..=8).contains(&active_k) {
            bail!("QX_V2_MOE_ACTIVE_K must be in 1..=8");
        }
        return benchmark_v2_quality(
            model,
            artifact,
            quality_text,
            args.quality_tokens,
            args.context,
            active_k,
        );
    }
    if args.v2_kernel_bench {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-kernel-bench requires an artifact path"))?;
        return benchmark_v2_kernel(artifact, args.iterations);
    }
    if args.v2_kernel_profile {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-kernel-profile requires an artifact path"))?;
        return benchmark_v2_kernel_profile(artifact, args.context, args.iterations);
    }
    if args.v2_kernel_sweep {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-kernel-sweep requires an artifact path"))?;
        return benchmark_v2_kernel_sweep(artifact, args.iterations);
    }
    if args.v2_qkv_sweep {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-qkv-sweep requires an artifact path"))?;
        return benchmark_v2_qkv_sweep(artifact, args.iterations);
    }
    if args.v2_qkv_boundary {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-qkv-boundary requires an artifact path"))?;
        return benchmark_v2_qkv_boundary(artifact, args.context, args.warmup, args.iterations);
    }
    if args.v2_qkv_interference {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-qkv-interference requires an artifact path"))?;
        return benchmark_v2_qkv_interference(artifact, args.context, args.warmup, args.iterations);
    }
    if args.v2_differential_profile {
        let artifact = args.artifact.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--v2-differential-profile requires an artifact path")
        })?;
        let fixture = args.fixture.as_deref().map(load_fixture).transpose()?;
        let context = fixture
            .as_ref()
            .and_then(|fixture| fixture.context)
            .unwrap_or(args.context);
        return benchmark_v2_differential_profile(
            artifact,
            context,
            args.warmup,
            args.iterations,
            fixture.as_ref(),
            args.result_json.as_deref(),
        );
    }
    if args.v2_moe_differential_profile {
        let artifact = args.artifact.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--v2-moe-differential-profile requires an artifact path")
        })?;
        let fixture = args.fixture.as_deref().map(load_fixture).transpose()?;
        let context = fixture
            .as_ref()
            .and_then(|fixture| fixture.context)
            .unwrap_or(args.context);
        return benchmark_v2_moe_differential_profile(
            artifact,
            context,
            args.warmup,
            args.iterations,
            fixture.as_ref(),
            args.result_json.as_deref(),
        );
    }
    if args.v2_moe_contextual_profile {
        let artifact = args.artifact.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--v2-moe-contextual-profile requires an artifact path")
        })?;
        let fixture = args.fixture.as_deref().map(load_fixture).transpose()?;
        let context = fixture
            .as_ref()
            .and_then(|fixture| fixture.context)
            .unwrap_or(args.context);
        return benchmark_v2_moe_contextual_profile(
            artifact,
            context,
            args.warmup,
            args.iterations,
            fixture.as_ref(),
            &args.contextual_candidate,
            args.result_json.as_deref(),
        );
    }
    if args.v2_moe_predecessor_profile {
        let artifact = args.artifact.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--v2-moe-predecessor-profile requires an artifact path")
        })?;
        let fixture = args.fixture.as_deref().map(load_fixture).transpose()?;
        let context = fixture
            .as_ref()
            .and_then(|fixture| fixture.context)
            .unwrap_or(args.context);
        return benchmark_v2_moe_predecessor_profile(
            artifact,
            context,
            args.warmup,
            args.iterations,
            fixture.as_ref(),
            args.result_json.as_deref(),
        );
    }
    if args.v2_route_overlap {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-route-overlap requires an artifact path"))?;
        return benchmark_v2_route_overlap(artifact, args.context, args.iterations);
    }
    if args.v2_route_churn_profile {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-route-churn-profile requires an artifact path"))?;
        return benchmark_v2_route_churn_profile(
            artifact,
            args.context,
            args.warmup,
            args.iterations,
            args.result_json.as_deref(),
        );
    }
    if args.v2_sequential_timeline {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-sequential-timeline requires an artifact path"))?;
        let fixture = args.fixture.as_deref().map(load_fixture).transpose()?;
        let context = fixture
            .as_ref()
            .and_then(|fixture| fixture.context)
            .unwrap_or(args.context);
        return benchmark_v2_sequential_timeline(
            artifact,
            context,
            args.warmup,
            args.iterations,
            fixture.as_ref(),
            args.result_json.as_deref(),
        );
    }
    if args.v2_moe_prefetch_profile {
        let artifact = args.artifact.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--v2-moe-prefetch-profile requires an artifact path")
        })?;
        return benchmark_v2_moe_prefetch_profile(
            artifact,
            args.context,
            args.warmup,
            args.iterations,
        );
    }
    if args.v2_moe_direct_region_profile {
        let artifact = args.artifact.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--v2-moe-direct-region-profile requires an artifact path")
        })?;
        return benchmark_v2_moe_direct_region_profile(
            artifact,
            args.context,
            args.warmup,
            args.iterations,
        );
    }
    if args.v2_moe_permutation_profile {
        let artifact = args.artifact.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--v2-moe-permutation-profile requires an artifact path")
        })?;
        return benchmark_v2_moe_permutation_profile(
            artifact,
            args.context,
            args.warmup,
            args.iterations,
        );
    }
    if args.v2_dispatch_floor {
        return benchmark_v2_dispatch_floor(args.iterations);
    }
    if args.v2_attention_sweep {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-attention-sweep requires an artifact path"))?;
        return benchmark_v2_attention_sweep(artifact, args.iterations);
    }
    if args.v2_layer_block {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-layer-block requires an artifact path"))?;
        return benchmark_v2_layer_block(artifact, args.context, args.iterations);
    }
    if args.v2_attention_stages {
        let artifact = args
            .artifact
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--v2-attention-stages requires an artifact path"))?;
        return benchmark_v2_attention_stages(artifact, args.context, args.iterations);
    }
    if let Some(model) = args.model {
        if args.v2_attention_stack {
            return benchmark_v2_attention_stack(
                &model,
                args.context,
                args.warmup,
                args.iterations,
            );
        }
        let artifact = args
            .artifact
            .ok_or_else(|| anyhow::anyhow!("full runtime mode requires an artifact path"))?;
        let fixture = args.fixture.as_deref().map(load_fixture).transpose()?;
        let fixture_tokens = fixture.as_ref().map(|fixture| fixture.tokens.clone());
        return benchmark_runtime(
            &model,
            &artifact,
            RuntimeOptions {
                prompt: &args.prompt,
                max_new_tokens: fixture
                    .as_ref()
                    .and_then(|fixture| fixture.generation)
                    .unwrap_or(args.max_new_tokens),
                context: fixture
                    .as_ref()
                    .and_then(|fixture| fixture.context)
                    .unwrap_or(args.context),
                warmup: args.warmup,
                iterations: args.iterations,
                compare_artifact: args.compare_artifact.as_deref(),
                quality_text: args.quality_text.as_deref(),
                quality_tokens: args.quality_tokens,
                fixture_tokens,
                profile: args.profile,
                ignore_eos: args.ignore_eos,
            },
        );
    }
    let artifact = args
        .artifact
        .ok_or_else(|| anyhow::anyhow!("an artifact path is required"))?;
    let artifact = MappedArtifact::open(&artifact)?;
    let info = artifact.info();
    if args.v2_plan {
        print_v2_plan(&artifact, args.context)?;
        return Ok(());
    }
    if args.format_summary {
        let mut counts = std::collections::BTreeMap::<String, usize>::new();
        let mut bytes = std::collections::BTreeMap::<String, usize>::new();
        for (index, entry) in info
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.is_expert())
        {
            let key = entry.quant_type.to_string();
            *counts.entry(key.clone()).or_default() += 1;
            *bytes.entry(key).or_default() += artifact.quantized_tensor(index)?.bytes();
        }
        println!("format counts: {:?}", counts);
        println!("format bytes:  {:?}", bytes);
        return Ok(());
    }
    if let Some(format) = args.find_quant.as_deref() {
        let quant_type = qx_quant::QuantType::parse(format)
            .ok_or_else(|| anyhow::anyhow!("unknown format {format}; use q2, q3, or q4"))?;
        let entry = info
            .entries
            .iter()
            .find(|entry| entry.is_expert() && entry.quant_type == quant_type)
            .ok_or_else(|| anyhow::anyhow!("no expert uses format {quant_type}"))?;
        println!(
            "first {quant_type} expert: layer {} expert {}",
            entry.layer, entry.expert
        );
        return Ok(());
    }
    let gate_up = info
        .entries
        .iter()
        .position(|entry| {
            entry.layer == args.layer
                && entry.expert == args.expert
                && entry.kind == TensorKind::GateUp
        })
        .ok_or_else(|| anyhow::anyhow!("gate/up tensor not found"))?;
    let down = info
        .entries
        .iter()
        .position(|entry| {
            entry.layer == args.layer
                && entry.expert == args.expert
                && entry.kind == TensorKind::Down
        })
        .ok_or_else(|| anyhow::anyhow!("down tensor not found"))?;
    let expert = QuantizedExpert {
        gate_up: artifact.quantized_tensor(gate_up)?,
        down: artifact.quantized_tensor(down)?,
    };
    if expert.gate_up.cols != HIDDEN || expert.down.rows != HIDDEN {
        bail!("artifact is not Granite V0 shaped");
    }
    let hidden = vec![0.125_f32; expert.gate_up.cols];
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..args.iterations {
        checksum += expert.run(&hidden).into_iter().sum::<f32>();
    }
    let elapsed = start.elapsed();
    let micros = elapsed.as_secs_f64() * 1_000_000.0 / args.iterations as f64;
    println!("QX benchmark report");
    println!("artifact:       {}", artifact.path().display());
    println!("model:          {}", info.header.model_id);
    println!("target:         {}", info.header.target_id);
    println!("layer/expert:   {}/{}", args.layer, args.expert);
    println!("iterations:     {}", args.iterations);
    println!("CPU ref µs/op:  {:.2}", micros);
    println!("CPU ref op/s:   {:.2}", 1_000_000.0 / micros);
    println!("checksum:       {:.6}", checksum);
    #[cfg(target_os = "macos")]
    benchmark_gpu_expert(&expert, args.iterations, info)?;
    println!("note:           CPU reference and, on macOS, real Metal dispatch");
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_kernel(artifact_path: &std::path::Path, iterations: usize) -> Result<()> {
    use std::rc::Rc;

    if iterations == 0 {
        bail!("kernel benchmark iterations must be positive");
    }
    let artifact = MappedArtifact::open(artifact_path)?;
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let find = |layer: u16, expert: u8, kind: TensorKind| {
        artifact
            .info()
            .entries
            .iter()
            .position(|entry| entry.layer == layer && entry.expert == expert && entry.kind == kind)
            .ok_or_else(|| anyhow::anyhow!("artifact tensor {kind:?} not found"))
    };

    let q_tensor = artifact.quantized_tensor(find(0, u8::MAX, TensorKind::QProj)?)?;
    let o_tensor = artifact.quantized_tensor(find(0, u8::MAX, TensorKind::OProj)?)?;
    let gate_up = artifact.quantized_tensor(find(0, 0, TensorKind::GateUp)?)?;
    let down = artifact.quantized_tensor(find(0, 0, TensorKind::Down)?)?;
    if q_tensor.rows != 1024
        || q_tensor.cols != 1024
        || o_tensor.rows != 1024
        || o_tensor.cols != 1024
        || gate_up.rows != 1024
        || gate_up.cols != 1024
        || down.rows != 1024
        || down.cols != 512
    {
        bail!("artifact does not contain the expected Granite Q4 benchmark shapes");
    }

    let q_prepared = context
        .prepare_q4_dense(&q_tensor)
        .map_err(anyhow::Error::msg)?;
    let o_prepared = context
        .prepare_q4_dense(&o_tensor)
        .map_err(anyhow::Error::msg)?;
    let q_simd_prepared = context
        .prepare_q4_dense_simd(&q_tensor)
        .map_err(anyhow::Error::msg)?;
    let o_simd_prepared = context
        .prepare_q4_dense_simd(&o_tensor)
        .map_err(anyhow::Error::msg)?;
    let gate_prepared = context
        .prepare_q4_stage(&gate_up)
        .map_err(anyhow::Error::msg)?;
    let down_prepared = context
        .prepare_q4_stage(&down)
        .map_err(anyhow::Error::msg)?;
    let gate_simd_prepared = context
        .prepare_q4_simd_stage(&gate_up)
        .map_err(anyhow::Error::msg)?;
    let down_simd_prepared = context
        .prepare_q4_simd_stage(&down)
        .map_err(anyhow::Error::msg)?;

    // The FP16 control uses the same dequantized QX weights as the Q4 path.
    // This isolates storage/unpack cost from model-shape and activation cost;
    // it is not a comparison against the original dense FP32 checkpoint.
    let q_f16 = context
        .prepare_f16_dense(
            &to_f16_bits(&q_tensor.dequantize()),
            q_tensor.rows,
            q_tensor.cols,
        )
        .map_err(anyhow::Error::msg)?;
    let o_f16 = context
        .prepare_f16_dense(
            &to_f16_bits(&o_tensor.dequantize()),
            o_tensor.rows,
            o_tensor.cols,
        )
        .map_err(anyhow::Error::msg)?;
    let gate_f16 = context
        .prepare_f16_dense(
            &to_f16_bits(&gate_up.dequantize()),
            gate_up.rows,
            gate_up.cols,
        )
        .map_err(anyhow::Error::msg)?;
    let down_f16 = context
        .prepare_f16_dense(&to_f16_bits(&down.dequantize()), down.rows, down.cols)
        .map_err(anyhow::Error::msg)?;

    let dense_input = vec![1.0_f32; 1024];
    let gate_input = vec![f16::from_f32(1.0).to_bits(); 1024];
    let down_input = vec![f16::from_f32(1.0).to_bits(); 512];
    let gate_reference = context
        .run_prepared_q4_stage(&gate_prepared, &gate_input)
        .map_err(anyhow::Error::msg)?;
    let gate_simd_reference = context
        .run_prepared_q4_simd_stage(&gate_simd_prepared, &gate_input)
        .map_err(anyhow::Error::msg)?;
    let down_reference = context
        .run_prepared_q4_stage(&down_prepared, &down_input)
        .map_err(anyhow::Error::msg)?;
    let down_simd_reference = context
        .run_prepared_q4_simd_stage(&down_simd_prepared, &down_input)
        .map_err(anyhow::Error::msg)?;
    let warmup = 10.min(iterations);
    let q_result =
        run_q4_dense_microbench(&context, &q_prepared, &dense_input, warmup, iterations)?;
    let o_result =
        run_q4_dense_microbench(&context, &o_prepared, &dense_input, warmup, iterations)?;
    let q_simd_result =
        run_q4_simd_microbench(&context, &q_simd_prepared, &dense_input, warmup, iterations)?;
    let o_simd_result =
        run_q4_simd_microbench(&context, &o_simd_prepared, &dense_input, warmup, iterations)?;
    let gate_result =
        run_q4_stage_microbench(&context, &gate_prepared, &gate_input, warmup, iterations)?;
    let down_result =
        run_q4_stage_microbench(&context, &down_prepared, &down_input, warmup, iterations)?;
    let gate_simd_result = run_q4_simd_stage_microbench(
        &context,
        &gate_simd_prepared,
        &gate_input,
        warmup,
        iterations,
    )?;
    let down_simd_result = run_q4_simd_stage_microbench(
        &context,
        &down_simd_prepared,
        &down_input,
        warmup,
        iterations,
    )?;
    let f16_input = vec![f16::from_f32(1.0).to_bits(); 1024];
    let f16_down_input = vec![f16::from_f32(1.0).to_bits(); 512];
    let q_f16_result = run_f16_dense_microbench(&context, &q_f16, &f16_input, warmup, iterations)?;
    let o_f16_result = run_f16_dense_microbench(&context, &o_f16, &f16_input, warmup, iterations)?;
    let gate_f16_result =
        run_f16_dense_microbench(&context, &gate_f16, &f16_input, warmup, iterations)?;
    let down_f16_result =
        run_f16_dense_microbench(&context, &down_f16, &f16_down_input, warmup, iterations)?;

    println!("QX V2 representative GEMV benchmark (median per dispatch)");
    println!("artifact:       {}", artifact_path.display());
    println!("iterations:     {iterations} (warmup {warmup})");
    println!(
        "configuration:  scalar Q4/FP16 = 256 threads/threadgroup, 8 SIMD groups, 1 row/thread; SIMD Q4 = 256 threads, 8 SIMD groups, 1 row/SIMD group (8 rows), 0 B threadgroup memory"
    );
    println!("                register pressure: not exposed by current build");
    println!(
        "expert SIMD correctness: gate/up max abs {:.6}, down max abs {:.6}",
        max_half_abs_diff(&gate_reference, &gate_simd_reference),
        max_half_abs_diff(&down_reference, &down_simd_reference),
    );
    println!(
        "shape              fmt  weight B   input B  output B   total B   GPU med   p95    CV   wall med  effective GB/s"
    );
    print_kernel_row(
        "dense Q projection",
        "Q4",
        q_tensor.bytes(),
        q_tensor.cols * 4,
        q_tensor.rows * 4,
        q_result,
    );
    print_kernel_row(
        "dense O projection",
        "Q4",
        o_tensor.bytes(),
        o_tensor.cols * 4,
        o_tensor.rows * 4,
        o_result,
    );
    print_kernel_row(
        "dense Q projection",
        "Q4-SIMD",
        q_tensor.bytes(),
        q_tensor.cols * 4,
        q_tensor.rows * 4,
        q_simd_result,
    );
    print_kernel_row(
        "dense O projection",
        "Q4-SIMD",
        o_tensor.bytes(),
        o_tensor.cols * 4,
        o_tensor.rows * 4,
        o_simd_result,
    );
    print_kernel_row(
        "expert gate/up",
        "Q4",
        gate_up.bytes(),
        gate_up.cols * 2,
        gate_up.rows * 2,
        gate_result,
    );
    print_kernel_row(
        "expert down",
        "Q4",
        down.bytes(),
        down.cols * 2,
        down.rows * 2,
        down_result,
    );
    print_kernel_row(
        "expert gate/up",
        "Q4-SIMD",
        gate_up.bytes(),
        gate_up.cols * 2,
        gate_up.rows * 2,
        gate_simd_result,
    );
    print_kernel_row(
        "expert down",
        "Q4-SIMD",
        down.bytes(),
        down.cols * 2,
        down.rows * 2,
        down_simd_result,
    );
    print_kernel_row(
        "dense Q projection",
        "F16",
        q_tensor.rows * q_tensor.cols * 2,
        q_tensor.cols * 2,
        q_tensor.rows * 2,
        q_f16_result,
    );
    print_kernel_row(
        "dense O projection",
        "F16",
        o_tensor.rows * o_tensor.cols * 2,
        o_tensor.cols * 2,
        o_tensor.rows * 2,
        o_f16_result,
    );
    print_kernel_row(
        "expert gate/up",
        "F16",
        gate_up.rows * gate_up.cols * 2,
        gate_up.cols * 2,
        gate_up.rows * 2,
        gate_f16_result,
    );
    print_kernel_row(
        "expert down",
        "F16",
        down.rows * down.cols * 2,
        down.cols * 2,
        down.rows * 2,
        down_f16_result,
    );
    println!(
        "llama equivalent:  not yet instrumented per kernel; whole-model baseline is separate"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_kernel_sweep(artifact_path: &std::path::Path, iterations: usize) -> Result<()> {
    use std::rc::Rc;

    if iterations == 0 {
        bail!("kernel sweep iterations must be positive");
    }
    let artifact = MappedArtifact::open(artifact_path)?;
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let q_index = artifact
        .info()
        .entries
        .iter()
        .position(|entry| {
            entry.layer == 0 && entry.expert == u8::MAX && entry.kind == TensorKind::QProj
        })
        .ok_or_else(|| anyhow::anyhow!("artifact Q projection not found"))?;
    let q_tensor = artifact.quantized_tensor(q_index)?;
    if q_tensor.rows != 1024 || q_tensor.cols != 1024 {
        bail!("artifact does not contain the expected 1024x1024 Q projection");
    }
    let prepared = context
        .prepare_q4_dense_simd(&q_tensor)
        .map_err(anyhow::Error::msg)?;
    let input = vec![1.0_f32; q_tensor.cols];
    let warmup = 5.min(iterations);
    let scalar = run_q4_dense_microbench(
        &context,
        &context
            .prepare_q4_dense(&q_tensor)
            .map_err(anyhow::Error::msg)?,
        &input,
        warmup,
        iterations,
    )?;
    let reference_output = context
        .run_prepared_q4_dense_simd(&prepared, &input)
        .map_err(anyhow::Error::msg)?;
    let sweep_output = context
        .run_prepared_q4_dense_sweep(&prepared, &input, 4, 8, true)
        .map_err(anyhow::Error::msg)?;
    let max_abs = reference_output
        .iter()
        .zip(&sweep_output)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let mut results = Vec::new();
    for rows_per_simd in [1_usize, 2, 4, 8] {
        for simd_groups in [1_usize, 2, 4, 8] {
            for cache_input in [false, true] {
                let result = run_q4_sweep_microbench(
                    &context,
                    &prepared,
                    &input,
                    rows_per_simd,
                    simd_groups,
                    cache_input,
                    warmup,
                    iterations,
                )?;
                results.push((rows_per_simd, simd_groups, cache_input, result));
            }
        }
    }
    results.sort_by(|left, right| left.3.gpu_median_us.total_cmp(&right.3.gpu_median_us));

    println!("QX Q-projection SIMD geometry sweep (release, median GPU dispatch)");
    println!("artifact: {}", artifact_path.display());
    println!("iterations: {iterations} (warmup {warmup})");
    println!(
        "scalar baseline: {:.2} us, p95 {:.2} us, CV {:.2}",
        scalar.gpu_median_us, scalar.gpu_p95_us, scalar.gpu_cv
    );
    println!("sweep correctness check (rows/SIMD=4, groups=8, cached): max abs {max_abs:.6}");
    println!("rows/SIMD  SIMD groups  cache input  threads/tg  GPU med us  p95 us  CV");
    for (rows_per_simd, simd_groups, cache_input, result) in results {
        println!(
            "{rows_per_simd:>9}  {simd_groups:>11}  {cache:>11}  {threads:>10}  {median:>10.2}  {p95:>6.2}  {cv:>4.2}",
            cache = if cache_input { "yes" } else { "no" },
            threads = simd_groups * 32,
            median = result.gpu_median_us,
            p95 = result.gpu_p95_us,
            cv = result.gpu_cv,
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_qkv_boundary_layer(
    graph: &mut qx_metal::platform::DecodeGraph,
    buffers: &qx_runtime_v2::GpuRuntimeBuffers,
    stack: &qx_runtime_v2::PreparedAttentionStack,
    layer: usize,
    position: usize,
    with_rope: bool,
) -> Result<()> {
    let context = buffers.context().as_ref();
    let prepared = &stack.layers[layer];
    let inverse = buffers.buffer(qx_runtime_v2::BufferId::MoeScratch);
    graph
        .encode_rmsnorm_scalar(
            context,
            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
            &prepared.input_norm,
            inverse,
        )
        .map_err(anyhow::Error::msg)?;
    graph
        .encode_prepared_q4_qkv_simd_normalized(
            &prepared.qkv,
            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
            &prepared.input_norm,
            inverse,
            buffers.buffer(qx_runtime_v2::BufferId::Query),
            buffers.buffer(qx_runtime_v2::BufferId::Key),
            buffers.buffer(qx_runtime_v2::BufferId::Value),
        )
        .map_err(anyhow::Error::msg)?;
    if with_rope {
        let key_cache = buffers.kv_key_buffer(layer).map_err(anyhow::Error::msg)?;
        let value_cache = buffers.kv_value_buffer(layer).map_err(anyhow::Error::msg)?;
        graph
            .encode_rope_kv_for_layer(
                context,
                buffers.buffer(qx_runtime_v2::BufferId::Query),
                buffers.buffer(qx_runtime_v2::BufferId::Key),
                buffers.buffer(qx_runtime_v2::BufferId::Value),
                buffers.buffer(qx_runtime_v2::BufferId::Query),
                &key_cache,
                &value_cache,
                layer,
                position,
            )
            .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_qkv_boundary(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;

    if iterations == 0 {
        bail!("QKV boundary iterations must be positive");
    }
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let hidden = vec![1.0_f32; qx_model::HIDDEN];
    let hidden_bytes = unsafe {
        std::slice::from_raw_parts(
            hidden.as_ptr() as *const u8,
            std::mem::size_of_val(hidden.as_slice()),
        )
    };
    context
        .write_runtime_buffer(
            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
            hidden_bytes,
        )
        .map_err(anyhow::Error::msg)?;
    let position = context_length.saturating_sub(1);
    let warmup = warmup.min(iterations);
    let qkv_one_layer = profile_graph_stage(context.as_ref(), warmup, iterations, |graph| {
        encode_qkv_boundary_layer(graph, &buffers, &stack, 0, position, false)
    })?;
    let qkv_all_layers = profile_graph_stage(context.as_ref(), warmup, iterations, |graph| {
        for layer in 0..stack.layers.len() {
            encode_qkv_boundary_layer(graph, &buffers, &stack, layer, position, false)?;
        }
        Ok(())
    })?;
    let one_layer = profile_graph_stage(context.as_ref(), warmup, iterations, |graph| {
        encode_qkv_boundary_layer(graph, &buffers, &stack, 0, position, true)
    })?;
    let all_layers = profile_graph_stage(context.as_ref(), warmup, iterations, |graph| {
        for layer in 0..stack.layers.len() {
            encode_qkv_boundary_layer(graph, &buffers, &stack, layer, position, true)?;
        }
        Ok(())
    })?;
    println!("QX QKV boundary replay (real artifact tensors)");
    println!("artifact:        {}", artifact_path.display());
    println!("context/position: {context_length}/{position}");
    println!(
        "geometry:        {}",
        if qx_metal::platform::qkv_geometry_enabled() {
            std::env::var("QX_V2_QKV_GEOMETRY").unwrap_or_else(|_| "enabled".to_owned())
        } else {
            "fused-control".to_owned()
        }
    );
    if qx_metal::platform::qkv_geometry_enabled() {
        println!(
            "input cache:     {}",
            if qx_metal::platform::qkv_geometry_cache_enabled() {
                "enabled"
            } else {
                "disabled"
            }
        );
    }
    println!("iterations:       {iterations} (warmup {warmup})");
    println!("RMS -> QKV, one layer:              {qkv_one_layer:.2} us GPU median");
    println!(
        "RMS -> QKV, 24 layers:              {qkv_all_layers:.2} us GPU median ({:.2} us/layer)",
        qkv_all_layers / stack.layers.len() as f64
    );
    println!("RMS -> QKV -> RoPE/KV, one layer:  {one_layer:.2} us GPU median");
    println!(
        "RMS -> QKV -> RoPE/KV, 24 layers:  {all_layers:.2} us GPU median ({:.2} us/layer)",
        all_layers / stack.layers.len() as f64
    );
    println!("dispatches:       4/layer (RMS, QKV, split RoPE/KV)");
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_interference_variant(
    graph: &mut qx_metal::platform::DecodeGraph,
    buffers: &qx_runtime_v2::GpuRuntimeBuffers,
    stack: &qx_runtime_v2::PreparedAttentionStack,
    moe: &qx_runtime_v2::PreparedMoeStack,
    position: usize,
    rung: usize,
) -> Result<()> {
    if rung == 0 {
        for layer in 0..stack.layers.len() {
            encode_qkv_boundary_layer(graph, buffers, stack, layer, position, true)?;
        }
        return Ok(());
    }
    if rung == 1 {
        return stack
            .encode(graph, buffers, position)
            .map_err(anyhow::Error::msg);
    }
    if rung == 3 {
        return stack
            .encode_full_layers(graph, buffers, moe, position, stack.layers.len())
            .map_err(anyhow::Error::msg);
    }

    let context = buffers.context().as_ref();
    for (layer, weights) in stack.layers.iter().enumerate() {
        let input = if layer.is_multiple_of(2) {
            qx_runtime_v2::BufferId::HiddenA
        } else {
            qx_runtime_v2::BufferId::HiddenB
        };
        let output = if layer.is_multiple_of(2) {
            qx_runtime_v2::BufferId::HiddenB
        } else {
            qx_runtime_v2::BufferId::HiddenA
        };
        buffers
            .encode_attention_layer(
                graph,
                layer,
                input,
                output,
                &weights.input_norm,
                &weights.qkv,
                &weights.o_proj,
                position,
            )
            .map_err(anyhow::Error::msg)?;
        let inverse = buffers
            .buffer(qx_runtime_v2::BufferId::MoeScratch)
            .subregion(81_916, std::mem::size_of::<f32>())
            .map_err(anyhow::Error::msg)?;
        if qx_metal::platform::rms_scalar_router_enabled() {
            graph
                .encode_rmsnorm_scalar(
                    context,
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    &inverse,
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_prepared_q4_dense_simd_normalized(
                    &weights.router,
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    &inverse,
                    buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                )
                .map_err(anyhow::Error::msg)?;
        } else {
            graph
                .encode_rmsnorm(
                    context,
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_prepared_q4_dense_simd(
                    &weights.router,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                )
                .map_err(anyhow::Error::msg)?;
        }
        graph
            .encode_topk(
                context,
                buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                buffers.buffer(qx_runtime_v2::BufferId::TopKIds),
                buffers.buffer(qx_runtime_v2::BufferId::TopKWeights),
            )
            .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_qkv_interference(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;

    if iterations == 0 {
        bail!("QKV interference iterations must be positive");
    }
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    let hidden = vec![1.0_f32; qx_model::HIDDEN];
    let hidden_bytes = unsafe {
        std::slice::from_raw_parts(
            hidden.as_ptr() as *const u8,
            std::mem::size_of_val(hidden.as_slice()),
        )
    };
    context
        .write_runtime_buffer(
            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
            hidden_bytes,
        )
        .map_err(anyhow::Error::msg)?;
    let position = context_length.saturating_sub(1);
    let warmup = warmup.min(iterations);
    println!("QX QKV interference ladder (real 24-layer graph)");
    println!("artifact:        {}", artifact_path.display());
    println!("context/position: {context_length}/{position}");
    println!("iterations:       {iterations} (warmup {warmup})");
    println!("rung                         graph GPU us   probe GPU us   probe delta/layer");
    let names = [
        "RMS -> QKV -> RoPE/KV",
        "+ attention + O + residual",
        "+ router + top-k",
        "+ complete MoE",
    ];
    for (rung, name) in names.iter().enumerate() {
        let base = profile_graph_stage(context.as_ref(), warmup, iterations, |graph| {
            encode_interference_variant(graph, &buffers, &stack, &moe, position, rung)
        })?;
        let probed = profile_graph_stage(context.as_ref(), warmup, iterations, |graph| {
            encode_interference_variant(graph, &buffers, &stack, &moe, position, rung)?;
            for layer in 0..stack.layers.len() {
                encode_qkv_boundary_layer(graph, &buffers, &stack, layer, position, false)?;
            }
            Ok(())
        })?;
        println!(
            "{name:<30} {base:>13.2}  {probed:>13.2}  {:>17.2}",
            (probed - base) / stack.layers.len() as f64
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DifferentialBlock {
    InputQkv,
    RopeKv,
    Attention,
    OutputResidual,
    RouterTopK,
    Moe,
    FinalHead,
}

#[cfg(target_os = "macos")]
struct RouteTrace {
    ids: qx_metal::platform::RuntimeBuffer,
    weights: qx_metal::platform::RuntimeBuffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct HostRouteStep {
    ids: [[u32; 8]; LAYERS],
    weights: [[f32; 8]; LAYERS],
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GateUpControl {
    WeightTouch,
    PayloadFraction(u32),
    ContiguousFraction(u32),
    RegionCount(u32),
    DirectRegionCount(u32),
    ScalesTouch,
    SyntheticArithmetic,
}

#[cfg(target_os = "macos")]
impl RouteTrace {
    fn ids_for(&self, layer: usize) -> Result<qx_metal::platform::RuntimeBuffer> {
        self.ids
            .subregion(
                layer * 8 * std::mem::size_of::<u32>(),
                8 * std::mem::size_of::<u32>(),
            )
            .map_err(anyhow::Error::msg)
    }

    fn weights_for(&self, layer: usize) -> Result<qx_metal::platform::RuntimeBuffer> {
        self.weights
            .subregion(
                layer * 8 * std::mem::size_of::<f32>(),
                8 * std::mem::size_of::<f32>(),
            )
            .map_err(anyhow::Error::msg)
    }
}

#[cfg(target_os = "macos")]
fn differential_noops(
    graph: &mut qx_metal::platform::DecodeGraph,
    context: &qx_metal::platform::MetalContext,
    count: usize,
) -> Result<()> {
    let threads = qx_metal::platform::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    for _ in 0..count {
        graph
            .encode_noop(context, threads, threads)
            .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn differential_attention_dispatches() -> usize {
    if !qx_metal::platform::three_stage_attention_enabled() {
        return 1;
    }
    1 + if qx_metal::platform::attention_softmax_simd_enabled() {
        2
    } else {
        1
    } + if qx_metal::platform::attention_v_simd_enabled() {
        2
    } else {
        1
    }
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_differential_token(
    graph: &mut qx_metal::platform::DecodeGraph,
    buffers: &qx_runtime_v2::GpuRuntimeBuffers,
    stack: &qx_runtime_v2::PreparedAttentionStack,
    moe: &qx_runtime_v2::PreparedMoeStack,
    position: usize,
    skip: Option<DifferentialBlock>,
    route_trace: Option<&RouteTrace>,
    direct_route_ids: Option<&[[u32; 8]]>,
    moe_stage_skip: Option<u8>,
    gate_up_control: Option<GateUpControl>,
    traffic_buffer: Option<&qx_metal::platform::RuntimeBuffer>,
    prefetch_route_ids: Option<&[[u32; 8]]>,
) -> Result<()> {
    let context = buffers.context().as_ref();
    let noop = |graph: &mut qx_metal::platform::DecodeGraph, count: usize| {
        differential_noops(graph, context, count)
    };

    // Every variant keeps the embedding lookup live. The named ablations are
    // transformer/final-head blocks; keeping this producer fixed makes their
    // deltas easier to compare.
    graph
        .encode_q4_embedding_lookup(
            &stack.embedding_lookup,
            buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
            EMBEDDING_MULTIPLIER,
        )
        .map_err(anyhow::Error::msg)?;
    // Diagnostic marker: the next interval is input RMS/QKV for layer 0.
    graph.stage_boundary(context).map_err(anyhow::Error::msg)?;

    for (layer, weights) in stack.layers.iter().enumerate() {
        let input = if layer.is_multiple_of(2) {
            qx_runtime_v2::BufferId::HiddenA
        } else {
            qx_runtime_v2::BufferId::HiddenB
        };
        let output = if layer.is_multiple_of(2) {
            qx_runtime_v2::BufferId::HiddenB
        } else {
            qx_runtime_v2::BufferId::HiddenA
        };

        if skip == Some(DifferentialBlock::InputQkv) {
            // Canonical path is scalar RMS + one normalized fused QKV dispatch.
            noop(graph, 2)?;
        } else if qx_metal::platform::rms_scalar_qkv_enabled() {
            let inverse_rms = buffers.buffer(qx_runtime_v2::BufferId::MoeScratch);
            graph
                .encode_rmsnorm_scalar(
                    context,
                    buffers.buffer(input),
                    &weights.input_norm,
                    inverse_rms,
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_prepared_q4_qkv_simd_normalized(
                    &weights.qkv,
                    buffers.buffer(input),
                    &weights.input_norm,
                    inverse_rms,
                    buffers.buffer(qx_runtime_v2::BufferId::Query),
                    buffers.buffer(qx_runtime_v2::BufferId::Key),
                    buffers.buffer(qx_runtime_v2::BufferId::Value),
                )
                .map_err(anyhow::Error::msg)?;
        } else {
            graph
                .encode_rmsnorm(
                    context,
                    buffers.buffer(input),
                    &weights.input_norm,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_prepared_q4_qkv_simd(
                    &weights.qkv,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    buffers.buffer(qx_runtime_v2::BufferId::Query),
                    buffers.buffer(qx_runtime_v2::BufferId::Key),
                    buffers.buffer(qx_runtime_v2::BufferId::Value),
                )
                .map_err(anyhow::Error::msg)?;
        }
        graph.stage_boundary(context).map_err(anyhow::Error::msg)?;

        let key_cache = buffers.kv_key_buffer(layer).map_err(anyhow::Error::msg)?;
        let value_cache = buffers.kv_value_buffer(layer).map_err(anyhow::Error::msg)?;
        if skip == Some(DifferentialBlock::RopeKv) {
            noop(
                graph,
                if qx_metal::platform::rope_kv_split_enabled() {
                    2
                } else {
                    1
                },
            )?;
        } else {
            graph
                .encode_rope_kv_for_layer(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::Query),
                    buffers.buffer(qx_runtime_v2::BufferId::Key),
                    buffers.buffer(qx_runtime_v2::BufferId::Value),
                    buffers.buffer(qx_runtime_v2::BufferId::Query),
                    &key_cache,
                    &value_cache,
                    layer,
                    position,
                )
                .map_err(anyhow::Error::msg)?;
        }
        graph.stage_boundary(context).map_err(anyhow::Error::msg)?;

        if skip == Some(DifferentialBlock::Attention) {
            noop(graph, differential_attention_dispatches())?;
        } else if qx_metal::platform::three_stage_attention_enabled() {
            graph
                .encode_attention_three_stage_for_layer(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::Query),
                    &key_cache,
                    &value_cache,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                    buffers.buffer(qx_runtime_v2::BufferId::Attention),
                    layer,
                    position + 1,
                )
                .map_err(anyhow::Error::msg)?;
        } else {
            graph
                .encode_attention(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::Query),
                    &key_cache,
                    &value_cache,
                    buffers.buffer(qx_runtime_v2::BufferId::Attention),
                    position + 1,
                )
                .map_err(anyhow::Error::msg)?;
        }
        graph.stage_boundary(context).map_err(anyhow::Error::msg)?;

        if skip == Some(DifferentialBlock::OutputResidual) {
            noop(graph, 2)?;
        } else {
            graph
                .encode_prepared_q4_dense_simd(
                    &weights.o_proj,
                    buffers.buffer(qx_runtime_v2::BufferId::Attention),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_residual(
                    context,
                    buffers.buffer(input),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    buffers.buffer(output),
                    RESIDUAL_MULTIPLIER,
                )
                .map_err(anyhow::Error::msg)?;
        }
        graph.stage_boundary(context).map_err(anyhow::Error::msg)?;

        let inverse_rms = buffers
            .buffer(qx_runtime_v2::BufferId::MoeScratch)
            .subregion(81_916, std::mem::size_of::<f32>())
            .map_err(anyhow::Error::msg)?;
        if let Some(prefetch_routes) = prefetch_route_ids {
            graph
                .encode_dynamic_q4_moe_gate_up_touch_direct(
                    &moe.layers[layer],
                    &prefetch_routes[layer],
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    8,
                )
                .map_err(anyhow::Error::msg)?;
        }
        if skip == Some(DifferentialBlock::RouterTopK) {
            noop(graph, 3)?;
        } else if qx_metal::platform::rms_scalar_router_enabled() {
            graph
                .encode_rmsnorm_scalar(
                    context,
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    &inverse_rms,
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_prepared_q4_dense_simd_normalized(
                    &weights.router,
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    &inverse_rms,
                    buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_topk(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                    buffers.buffer(qx_runtime_v2::BufferId::TopKIds),
                    buffers.buffer(qx_runtime_v2::BufferId::TopKWeights),
                )
                .map_err(anyhow::Error::msg)?;
        } else {
            graph
                .encode_rmsnorm(
                    context,
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_topk(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                    buffers.buffer(qx_runtime_v2::BufferId::TopKIds),
                    buffers.buffer(qx_runtime_v2::BufferId::TopKWeights),
                )
                .map_err(anyhow::Error::msg)?;
        }
        graph.stage_boundary(context).map_err(anyhow::Error::msg)?;

        let expert_ids = route_trace
            .map(|trace| trace.ids_for(layer))
            .transpose()?
            .unwrap_or_else(|| buffers.buffer(qx_runtime_v2::BufferId::TopKIds).clone());
        let expert_weights = route_trace
            .map(|trace| trace.weights_for(layer))
            .transpose()?
            .unwrap_or_else(|| buffers.buffer(qx_runtime_v2::BufferId::TopKWeights).clone());

        if skip == Some(DifferentialBlock::Moe) {
            noop(graph, 4)?;
        } else if let Some(stage_skip) = moe_stage_skip {
            for stage in 0_u8..4 {
                if stage == stage_skip {
                    match gate_up_control {
                        Some(
                            GateUpControl::WeightTouch
                            | GateUpControl::PayloadFraction(_)
                            | GateUpControl::ContiguousFraction(_)
                            | GateUpControl::RegionCount(_)
                            | GateUpControl::ScalesTouch,
                        ) if stage == 0 => {
                            let mode = match gate_up_control {
                                Some(GateUpControl::WeightTouch) => 0,
                                Some(GateUpControl::PayloadFraction(25)) => 4,
                                Some(GateUpControl::PayloadFraction(50)) => 5,
                                Some(GateUpControl::PayloadFraction(75)) => 6,
                                Some(GateUpControl::PayloadFraction(100)) => 7,
                                Some(GateUpControl::ContiguousFraction(25)) => 8,
                                Some(GateUpControl::ContiguousFraction(50)) => 9,
                                Some(GateUpControl::ContiguousFraction(75)) => 10,
                                Some(GateUpControl::ContiguousFraction(100)) => 11,
                                Some(GateUpControl::RegionCount(1)) => 12,
                                Some(GateUpControl::RegionCount(2)) => 13,
                                Some(GateUpControl::RegionCount(4)) => 14,
                                Some(GateUpControl::RegionCount(8)) => 15,
                                Some(GateUpControl::ScalesTouch) => 2,
                                _ => unreachable!(),
                            };
                            graph
                                .encode_dynamic_q4_moe_gate_up_touch(
                                    &moe.layers[layer],
                                    &expert_ids,
                                    traffic_buffer.ok_or_else(|| {
                                        anyhow::anyhow!("gate/up traffic buffer is missing")
                                    })?,
                                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                                    mode,
                                )
                                .map_err(anyhow::Error::msg)?;
                        }
                        Some(GateUpControl::DirectRegionCount(region_count)) if stage == 0 => {
                            let routes = direct_route_ids.ok_or_else(|| {
                                anyhow::anyhow!(
                                    "direct gate/up traffic requires host-resolved route IDs"
                                )
                            })?;
                            graph
                                .encode_dynamic_q4_moe_gate_up_touch_direct(
                                    &moe.layers[layer],
                                    &routes[layer],
                                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                                    region_count,
                                )
                                .map_err(anyhow::Error::msg)?;
                        }
                        Some(GateUpControl::SyntheticArithmetic) if stage == 0 => {
                            if qx_metal::platform::rms_scalar_router_enabled() {
                                graph
                                    .encode_dynamic_q4_moe_gate_up_synthetic_normalized(
                                        &moe.layers[layer],
                                        buffers.buffer(output),
                                        &weights.post_attention_norm,
                                        &inverse_rms,
                                        &expert_ids,
                                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                                    )
                                    .map_err(anyhow::Error::msg)?;
                            } else {
                                graph
                                    .encode_dynamic_q4_moe_gate_up_synthetic(
                                        &moe.layers[layer],
                                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                                        &expert_ids,
                                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                                    )
                                    .map_err(anyhow::Error::msg)?;
                            }
                        }
                        _ => noop(graph, 1)?,
                    }
                } else if qx_metal::platform::rms_scalar_router_enabled() {
                    graph
                        .encode_dynamic_q4_moe_batched_stage_normalized(
                            &moe.layers[layer],
                            buffers.buffer(output),
                            &weights.post_attention_norm,
                            &inverse_rms,
                            &expert_ids,
                            &expert_weights,
                            buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                            buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                            stage,
                        )
                        .map_err(anyhow::Error::msg)?;
                } else {
                    graph
                        .encode_dynamic_q4_moe_batched_stage(
                            &moe.layers[layer],
                            buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                            &expert_ids,
                            &expert_weights,
                            buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                            buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                            stage,
                        )
                        .map_err(anyhow::Error::msg)?;
                }
            }
        } else if qx_metal::platform::rms_scalar_router_enabled() {
            graph
                .encode_dynamic_q4_moe_batched_simd_normalized(
                    &moe.layers[layer],
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    &inverse_rms,
                    &expert_ids,
                    &expert_weights,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                )
                .map_err(anyhow::Error::msg)?;
        } else {
            graph
                .encode_dynamic_q4_moe_batched_simd(
                    &moe.layers[layer],
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    &expert_ids,
                    &expert_weights,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                )
                .map_err(anyhow::Error::msg)?;
        }

        if skip == Some(DifferentialBlock::Moe) {
            // The ablated path already replaced the MoE block; preserve the
            // residual write so later layers retain the canonical dependency shape.
        }
        graph
            .encode_residual(
                context,
                buffers.buffer(output),
                buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                buffers.buffer(output),
                RESIDUAL_MULTIPLIER,
            )
            .map_err(anyhow::Error::msg)?;
        graph.stage_boundary(context).map_err(anyhow::Error::msg)?;
    }

    let final_hidden = if LAYERS.is_multiple_of(2) {
        qx_runtime_v2::BufferId::HiddenA
    } else {
        qx_runtime_v2::BufferId::HiddenB
    };
    if skip == Some(DifferentialBlock::FinalHead) {
        // final RMSNorm + LM head + the two-pass argmax reduction
        noop(graph, 4)?;
    } else {
        graph
            .encode_rmsnorm(
                context,
                buffers.buffer(final_hidden),
                &stack.final_norm,
                buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
            )
            .map_err(anyhow::Error::msg)?;
        graph
            .encode_prepared_q4_dense_simd_scaled(
                &stack.embedding,
                buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                buffers.buffer(qx_runtime_v2::BufferId::Logits),
                1.0 / LOGITS_SCALING,
            )
            .map_err(anyhow::Error::msg)?;
        graph
            .encode_argmax(
                context,
                buffers.buffer(qx_runtime_v2::BufferId::Logits),
                buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                VOCAB,
            )
            .map_err(anyhow::Error::msg)?;
    }
    graph.stage_boundary(context).map_err(anyhow::Error::msg)?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn capture_route_trace(
    context: &qx_metal::platform::MetalContext,
    buffers: &qx_runtime_v2::GpuRuntimeBuffers,
    stack: &qx_runtime_v2::PreparedAttentionStack,
    moe: &qx_runtime_v2::PreparedMoeStack,
    position: usize,
) -> Result<RouteTrace> {
    let ids = context
        .allocate_runtime_buffer(LAYERS * 8 * std::mem::size_of::<u32>())
        .map_err(anyhow::Error::msg)?;
    let weights = context
        .allocate_runtime_buffer(LAYERS * 8 * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;

    // Re-run prefixes one layer at a time. This is intentionally a setup-only
    // operation: it makes every layer's canonical GPU top-8 route persistent
    // in separate storage, without adding CPU waits to the measured graphs.
    for target_layer in 0..LAYERS {
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        graph
            .encode_q4_embedding_lookup(
                &stack.embedding_lookup,
                buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
                EMBEDDING_MULTIPLIER,
            )
            .map_err(anyhow::Error::msg)?;
        for (layer, layer_weights) in stack.layers.iter().enumerate().take(target_layer + 1) {
            let input = if layer.is_multiple_of(2) {
                qx_runtime_v2::BufferId::HiddenA
            } else {
                qx_runtime_v2::BufferId::HiddenB
            };
            let output = if layer.is_multiple_of(2) {
                qx_runtime_v2::BufferId::HiddenB
            } else {
                qx_runtime_v2::BufferId::HiddenA
            };
            buffers
                .encode_attention_layer(
                    &mut graph,
                    layer,
                    input,
                    output,
                    &layer_weights.input_norm,
                    &layer_weights.qkv,
                    &layer_weights.o_proj,
                    position,
                )
                .map_err(anyhow::Error::msg)?;
            let inverse_rms = buffers
                .buffer(qx_runtime_v2::BufferId::MoeScratch)
                .subregion(81_916, std::mem::size_of::<f32>())
                .map_err(anyhow::Error::msg)?;
            if qx_metal::platform::rms_scalar_router_enabled() {
                graph
                    .encode_rmsnorm_scalar(
                        context,
                        buffers.buffer(output),
                        &layer_weights.post_attention_norm,
                        &inverse_rms,
                    )
                    .map_err(anyhow::Error::msg)?;
                graph
                    .encode_prepared_q4_dense_simd_normalized(
                        &layer_weights.router,
                        buffers.buffer(output),
                        &layer_weights.post_attention_norm,
                        &inverse_rms,
                        buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                    )
                    .map_err(anyhow::Error::msg)?;
            } else {
                graph
                    .encode_rmsnorm(
                        context,
                        buffers.buffer(output),
                        &layer_weights.post_attention_norm,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    )
                    .map_err(anyhow::Error::msg)?;
                graph
                    .encode_prepared_q4_dense_simd(
                        &layer_weights.router,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                    )
                    .map_err(anyhow::Error::msg)?;
            }
            let route_ids = if layer == target_layer {
                ids.subregion(layer * 8 * 4, 8 * 4)
                    .map_err(anyhow::Error::msg)?
            } else {
                buffers.buffer(qx_runtime_v2::BufferId::TopKIds).clone()
            };
            let route_weights = if layer == target_layer {
                weights
                    .subregion(layer * 8 * 4, 8 * 4)
                    .map_err(anyhow::Error::msg)?
            } else {
                buffers.buffer(qx_runtime_v2::BufferId::TopKWeights).clone()
            };
            graph
                .encode_topk(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                    &route_ids,
                    &route_weights,
                )
                .map_err(anyhow::Error::msg)?;
            if qx_metal::platform::rms_scalar_router_enabled() {
                graph
                    .encode_dynamic_q4_moe_batched_simd_normalized(
                        &moe.layers[layer],
                        buffers.buffer(output),
                        &layer_weights.post_attention_norm,
                        &inverse_rms,
                        &route_ids,
                        &route_weights,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    )
                    .map_err(anyhow::Error::msg)?;
            } else {
                graph
                    .encode_dynamic_q4_moe_batched_simd(
                        &moe.layers[layer],
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        &route_ids,
                        &route_weights,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    )
                    .map_err(anyhow::Error::msg)?;
            }
            graph
                .encode_residual(
                    context,
                    buffers.buffer(output),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    buffers.buffer(output),
                    RESIDUAL_MULTIPLIER,
                )
                .map_err(anyhow::Error::msg)?;
        }
        graph.finish(context).map_err(anyhow::Error::msg)?;
    }
    Ok(RouteTrace { ids, weights })
}

#[cfg(target_os = "macos")]
fn sort_route_trace_by_expert_id(
    context: &qx_metal::platform::MetalContext,
    trace: &RouteTrace,
) -> Result<RouteTrace> {
    let ids = context
        .read_runtime_u32s(&trace.ids, LAYERS * 8)
        .map_err(anyhow::Error::msg)?;
    let weights = context
        .read_runtime_f32(&trace.weights, LAYERS * 8)
        .map_err(anyhow::Error::msg)?;
    let mut sorted_ids = Vec::with_capacity(LAYERS * 8);
    let mut sorted_weights = Vec::with_capacity(LAYERS * 8);
    for layer in 0..LAYERS {
        let mut route = (0..8)
            .map(|slot| (ids[layer * 8 + slot], weights[layer * 8 + slot]))
            .collect::<Vec<_>>();
        route.sort_unstable_by_key(|(expert, _)| *expert);
        sorted_ids.extend(route.iter().map(|(expert, _)| *expert));
        sorted_weights.extend(route.iter().map(|(_, weight)| *weight));
    }

    let sorted_id_buffer = context
        .allocate_runtime_buffer(sorted_ids.len() * std::mem::size_of::<u32>())
        .map_err(anyhow::Error::msg)?;
    let sorted_weight_buffer = context
        .allocate_runtime_buffer(sorted_weights.len() * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let id_bytes = sorted_ids
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let weight_bytes = sorted_weights
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    context
        .write_runtime_buffer(&sorted_id_buffer, &id_bytes)
        .map_err(anyhow::Error::msg)?;
    context
        .write_runtime_buffer(&sorted_weight_buffer, &weight_bytes)
        .map_err(anyhow::Error::msg)?;
    Ok(RouteTrace {
        ids: sorted_id_buffer,
        weights: sorted_weight_buffer,
    })
}

#[cfg(target_os = "macos")]
fn benchmark_v2_differential_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
    fixture: Option<&BenchmarkFixture>,
    result_json: Option<&std::path::Path>,
) -> Result<()> {
    use std::rc::Rc;

    let (_artifact_bytes, artifact_hash) = artifact_fingerprint(artifact_path)?;
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    let position = if let Some(fixture) = fixture {
        if fixture.measurement != MeasurementMode::PostPrefill {
            bail!("differential fixtures must use post_prefill measurement");
        }
        if fixture.tokens.len() >= context_length {
            bail!("differential fixture prefill must leave one decode position");
        }
        for (position, token) in fixture.tokens.iter().enumerate() {
            context
                .write_runtime_buffer(
                    buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                    &token.to_le_bytes(),
                )
                .map_err(anyhow::Error::msg)?;
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            stack
                .encode_token(&mut graph, &buffers, &moe, position)
                .map_err(anyhow::Error::msg)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        }
        fixture.tokens.len()
    } else {
        context_length.saturating_sub(1)
    };
    let warmup = warmup.min(iterations);
    let variants = [
        ("baseline", None),
        ("without input RMS/QKV", Some(DifferentialBlock::InputQkv)),
        ("without RoPE/KV", Some(DifferentialBlock::RopeKv)),
        ("without attention", Some(DifferentialBlock::Attention)),
        (
            "without O/residual",
            Some(DifferentialBlock::OutputResidual),
        ),
        ("without router/top-k", Some(DifferentialBlock::RouterTopK)),
        ("without MoE", Some(DifferentialBlock::Moe)),
        ("without final head", Some(DifferentialBlock::FinalHead)),
    ];

    println!("QX V2 realistic differential profile");
    println!("artifact:          {}", artifact_path.display());
    println!("context/position:  {context_length}/{position}");
    println!("iterations:        {iterations} (warmup {warmup})");
    println!(
        "baseline and ablations replay one frozen canonical top-8 route trace; no-op dispatch shape is retained where possible"
    );
    let route_trace = capture_route_trace(context.as_ref(), &buffers, &stack, &moe, position)?;
    println!("variant                       GPU us/token   dispatches   marginal saved us");
    let mut baseline_gpu = 0.0;
    let mut results = Vec::with_capacity(variants.len());
    for (name, skip) in variants {
        let detailed =
            profile_graph_stage_detailed(context.as_ref(), warmup, iterations, |graph| {
                encode_differential_token(
                    graph,
                    &buffers,
                    &stack,
                    &moe,
                    position,
                    skip,
                    Some(&route_trace),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            })?;
        if skip.is_none() {
            baseline_gpu = detailed.0;
        }
        results.push((name, detailed));
    }
    let report_rows = results
        .iter()
        .map(|(name, (gpu, dispatches))| DifferentialProfileRow {
            name: (*name).to_owned(),
            gpu_us_per_token: *gpu,
            dispatches: *dispatches,
            marginal_saved_us: baseline_gpu - *gpu,
        })
        .collect::<Vec<_>>();
    for row in &report_rows {
        println!(
            "{:<30} {:>13.2}  {:>11}  {:>18.2}",
            row.name, row.gpu_us_per_token, row.dispatches, row.marginal_saved_us
        );
    }
    if let Some(path) = result_json {
        let mapped = MappedArtifact::open(artifact_path)?;
        let report = DifferentialProfileReport {
            schema_version: 2,
            qx_revision: option_env!("QX_GIT_REVISION").unwrap_or("unknown"),
            artifact: artifact_path.display().to_string(),
            artifact_hash: format!("{artifact_hash:016x}"),
            artifact_version: mapped.info().header.version,
            plan_id: mapped
                .info()
                .execution_plan
                .as_ref()
                .map(|plan| plan.plan_id.clone())
                .unwrap_or_else(|| "legacy-v1-unsealed".to_owned()),
            target_id: mapped.info().header.target_id.clone(),
            fixture_name: fixture.map(|value| value.name.clone()),
            fixture_domain: fixture.map(|value| value.domain.clone()),
            context: context_length,
            position,
            routes_frozen: true,
            iterations,
            warmup,
            rows: report_rows,
        };
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("result JSON:       {}", path.display());
    }
    println!();
    println!(
        "Interpretation: ablation deltas are marginal full-graph savings, not additive operator costs."
    );
    println!(
        "Ablations can change cache residency and dependency scheduling; use them to rank targets, then validate any replacement on the complete graph and parity gate."
    );
    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Serialize)]
struct DifferentialProfileRow {
    name: String,
    gpu_us_per_token: f64,
    dispatches: u64,
    marginal_saved_us: f64,
}

#[cfg(target_os = "macos")]
#[derive(Serialize)]
struct DifferentialProfileReport {
    schema_version: u32,
    qx_revision: &'static str,
    artifact: String,
    artifact_hash: String,
    artifact_version: u32,
    plan_id: String,
    target_id: String,
    fixture_name: Option<String>,
    fixture_domain: Option<String>,
    context: usize,
    position: usize,
    routes_frozen: bool,
    iterations: usize,
    warmup: usize,
    rows: Vec<DifferentialProfileRow>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Serialize)]
struct ContextualTiming {
    gpu_us_per_sample: f64,
    wall_us_per_sample: f64,
    dispatches_per_sample: f64,
    command_buffers_per_sample: f64,
    encoders_per_sample: f64,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Serialize)]
struct TwoCommandBufferTiming {
    predecessor_gpu_us: f64,
    moe_gpu_us: f64,
    total_gpu_us: f64,
    wall_us: f64,
    dispatches: f64,
    command_buffers: f64,
    encoders: f64,
}

#[cfg(target_os = "macos")]
fn profile_two_command_buffers<P, M>(
    context: &qx_metal::platform::MetalContext,
    warmup: usize,
    iterations: usize,
    mut encode_predecessor: P,
    mut encode_moe: M,
) -> Result<TwoCommandBufferTiming>
where
    P: FnMut(&mut qx_metal::platform::DecodeGraph) -> Result<()>,
    M: FnMut(&mut qx_metal::platform::DecodeGraph) -> Result<()>,
{
    let run = |encode_predecessor: &mut P,
               encode_moe: &mut M|
     -> Result<(f64, f64, f64, f64, f64)> {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        encode_predecessor(&mut graph)?;
        // This commits CB1 without waiting and starts CB2 on the same Metal
        // queue. The final wait happens only in finish(), after CB2.
        graph
            .begin_next_command_buffer(context)
            .map_err(anyhow::Error::msg)?;
        encode_moe(&mut graph)?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        let timeline = context.last_stage_timeline().map_err(anyhow::Error::msg)?;
        let predecessor_gpu_us = timeline.segment_gpu_seconds.first().copied().unwrap_or(0.0) * 1e6;
        let moe_gpu_us = timeline.segment_gpu_seconds.get(1).copied().unwrap_or(0.0) * 1e6;
        Ok((
            predecessor_gpu_us,
            moe_gpu_us,
            (after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6,
            start.elapsed().as_secs_f64() * 1e6,
            (after.dispatches - before.dispatches) as f64,
        ))
    };

    for _ in 0..warmup {
        run(&mut encode_predecessor, &mut encode_moe)?;
    }
    let mut predecessor = Vec::with_capacity(iterations);
    let mut moe = Vec::with_capacity(iterations);
    let mut total = Vec::with_capacity(iterations);
    let mut wall = Vec::with_capacity(iterations);
    let mut dispatches = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (pred, moe_time, total_time, wall_time, dispatch_count) =
            run(&mut encode_predecessor, &mut encode_moe)?;
        predecessor.push(pred);
        moe.push(moe_time);
        total.push(total_time);
        wall.push(wall_time);
        dispatches.push(dispatch_count);
    }
    Ok(TwoCommandBufferTiming {
        predecessor_gpu_us: median_sample(&mut predecessor),
        moe_gpu_us: median_sample(&mut moe),
        total_gpu_us: median_sample(&mut total),
        wall_us: median_sample(&mut wall),
        dispatches: median_sample(&mut dispatches),
        command_buffers: 2.0,
        encoders: 2.0,
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Serialize)]
struct ContextualMemory {
    artifact_bytes: u64,
    mapped_artifact_bytes: u64,
    known_metal_allocated_bytes: u64,
    recommended_metal_working_set_bytes: u64,
    current_rss_bytes: u64,
    peak_rss_bytes: u64,
    current_physical_footprint_bytes: u64,
    peak_physical_footprint_bytes: u64,
}

#[cfg(target_os = "macos")]
#[derive(Serialize)]
struct ContextualLevelReport {
    level: String,
    timing: ContextualTiming,
    memory: ContextualMemory,
}

#[cfg(target_os = "macos")]
#[derive(Serialize)]
struct ContextualProfileReport {
    schema_version: u32,
    mode: &'static str,
    candidate: String,
    artifact: String,
    artifact_hash: String,
    fixture_name: Option<String>,
    fixture_domain: Option<String>,
    context: usize,
    position: usize,
    iterations: usize,
    warmup: usize,
    routes_frozen: bool,
    route_source: &'static str,
    topology: Vec<String>,
    levels: Vec<ContextualLevelReport>,
}

#[cfg(target_os = "macos")]
fn profile_graph_stage_contextual<F>(
    context: &qx_metal::platform::MetalContext,
    warmup: usize,
    iterations: usize,
    mut encode: F,
) -> Result<ContextualTiming>
where
    F: FnMut(&mut qx_metal::platform::DecodeGraph) -> Result<()>,
{
    for _ in 0..warmup {
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        encode(&mut graph)?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    let mut dispatch_samples = Vec::with_capacity(iterations);
    let mut command_buffer_samples = Vec::with_capacity(iterations);
    let mut encoder_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        encode(&mut graph)?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6);
        wall_samples.push(start.elapsed().as_secs_f64() * 1e6);
        dispatch_samples.push((after.dispatches - before.dispatches) as f64);
        command_buffer_samples.push((after.command_buffers - before.command_buffers) as f64);
        encoder_samples.push((after.encoders - before.encoders) as f64);
    }
    Ok(ContextualTiming {
        gpu_us_per_sample: median_sample(&mut gpu_samples),
        wall_us_per_sample: median_sample(&mut wall_samples),
        dispatches_per_sample: median_sample(&mut dispatch_samples),
        command_buffers_per_sample: median_sample(&mut command_buffer_samples),
        encoders_per_sample: median_sample(&mut encoder_samples),
    })
}

#[cfg(target_os = "macos")]
fn contextual_memory(
    context: &qx_metal::platform::MetalContext,
    artifact_bytes: u64,
) -> ContextualMemory {
    let (current_physical_footprint_bytes, peak_physical_footprint_bytes) = physical_footprint();
    ContextualMemory {
        artifact_bytes,
        mapped_artifact_bytes: artifact_bytes,
        known_metal_allocated_bytes: context.current_allocated_bytes(),
        recommended_metal_working_set_bytes: context.recommended_working_set_bytes(),
        current_rss_bytes: current_rss_bytes(),
        peak_rss_bytes: peak_rss_bytes(),
        current_physical_footprint_bytes,
        peak_physical_footprint_bytes,
    }
}

#[cfg(target_os = "macos")]
fn encode_fixed_route_moe(
    graph: &mut qx_metal::platform::DecodeGraph,
    buffers: &qx_runtime_v2::GpuRuntimeBuffers,
    stack: &qx_runtime_v2::PreparedAttentionStack,
    moe: &qx_runtime_v2::PreparedMoeStack,
    route_trace: &RouteTrace,
    gate_up_only: bool,
) -> Result<()> {
    let context = buffers.context().as_ref();
    for (layer, weights) in stack.layers.iter().enumerate() {
        let ids = route_trace.ids_for(layer)?;
        let route_weights = route_trace.weights_for(layer)?;
        if qx_metal::platform::rms_scalar_router_enabled() {
            let inverse_rms = buffers
                .buffer(qx_runtime_v2::BufferId::MoeScratch)
                .subregion(81_916, std::mem::size_of::<f32>())
                .map_err(anyhow::Error::msg)?;
            if gate_up_only {
                graph
                    .encode_dynamic_q4_moe_batched_stage_normalized(
                        &moe.layers[layer],
                        buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                        &weights.post_attention_norm,
                        &inverse_rms,
                        &ids,
                        &route_weights,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                        0,
                    )
                    .map_err(anyhow::Error::msg)?;
            } else {
                graph
                    .encode_dynamic_q4_moe_batched_simd_normalized(
                        &moe.layers[layer],
                        buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                        &weights.post_attention_norm,
                        &inverse_rms,
                        &ids,
                        &route_weights,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    )
                    .map_err(anyhow::Error::msg)?;
            }
        } else if gate_up_only {
            graph
                .encode_dynamic_q4_moe_batched_stage(
                    &moe.layers[layer],
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    &ids,
                    &route_weights,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    0,
                )
                .map_err(anyhow::Error::msg)?;
        } else {
            graph
                .encode_dynamic_q4_moe_batched_simd(
                    &moe.layers[layer],
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    &ids,
                    &route_weights,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                )
                .map_err(anyhow::Error::msg)?;
        }
    }
    let _ = context;
    Ok(())
}

#[cfg(target_os = "macos")]
fn replay_contextual_fixture(
    context: &qx_metal::platform::MetalContext,
    buffers: &qx_runtime_v2::GpuRuntimeBuffers,
    stack: &qx_runtime_v2::PreparedAttentionStack,
    moe: &qx_runtime_v2::PreparedMoeStack,
    fixture: Option<&BenchmarkFixture>,
    context_length: usize,
) -> Result<usize> {
    let Some(fixture) = fixture else {
        return Ok(context_length.saturating_sub(1));
    };
    if fixture.measurement != MeasurementMode::PostPrefill {
        bail!("contextual profiles require post_prefill fixtures");
    }
    if fixture.tokens.len() >= context_length {
        bail!("fixture prefill must leave one decode position");
    }
    for (position, token) in fixture.tokens.iter().enumerate() {
        context
            .write_runtime_buffer(
                buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                &token.to_le_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, buffers, moe, position)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    Ok(fixture.tokens.len())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_moe_contextual_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
    fixture: Option<&BenchmarkFixture>,
    candidate: &str,
    result_json: Option<&std::path::Path>,
) -> Result<()> {
    use std::rc::Rc;
    if iterations == 0 {
        bail!("contextual profile iterations must be positive");
    }
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let mapped_artifact = std::sync::Arc::new(MappedArtifact::open(artifact_path)?);
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_mapped_artifact(
        context.as_ref(),
        mapped_artifact.clone(),
    )
    .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load_from_mapped_artifact(
        context.as_ref(),
        mapped_artifact,
    )
    .map_err(anyhow::Error::msg)?;
    let position = replay_contextual_fixture(
        context.as_ref(),
        &buffers,
        &stack,
        &moe,
        fixture,
        context_length,
    )?;
    let route_trace = capture_route_trace(context.as_ref(), &buffers, &stack, &moe, position)?;
    let (artifact_bytes, artifact_hash) = artifact_fingerprint(artifact_path)?;
    let topology = vec![
        format!(
            "qkv={}",
            std::env::var("QX_V2_QKV_TOPOLOGY").unwrap_or_else(|_| "fused".to_owned())
        ),
        format!(
            "rope_kv_split={}",
            qx_metal::platform::rope_kv_split_enabled()
        ),
        format!(
            "attention_3stage={}",
            qx_metal::platform::three_stage_attention_enabled()
        ),
        format!(
            "rms_scalar_qkv={}",
            qx_metal::platform::rms_scalar_qkv_enabled()
        ),
        format!(
            "rms_scalar_router={}",
            qx_metal::platform::rms_scalar_router_enabled()
        ),
        format!(
            "moe_layout={}",
            std::env::var("QX_V2_MOE_LAYOUT").unwrap_or_else(|_| "artifact".to_owned())
        ),
    ];
    let memory = || contextual_memory(context.as_ref(), artifact_bytes);
    let gate_up_timing =
        profile_graph_stage_contextual(context.as_ref(), warmup, iterations, |graph| {
            encode_fixed_route_moe(graph, &buffers, &stack, &moe, &route_trace, true)
        })?;
    let fixed_moe_timing =
        profile_graph_stage_contextual(context.as_ref(), warmup, iterations, |graph| {
            encode_fixed_route_moe(graph, &buffers, &stack, &moe, &route_trace, false)
        })?;
    let complete_timing =
        profile_graph_stage_contextual(context.as_ref(), warmup, iterations, |graph| {
            encode_differential_token(
                graph,
                &buffers,
                &stack,
                &moe,
                position,
                None,
                Some(&route_trace),
                None,
                None,
                None,
                None,
                None,
            )
        })?;
    let levels = vec![
        ContextualLevelReport {
            level: "gate_up_boundary_24_layers".to_owned(),
            timing: gate_up_timing,
            memory: memory(),
        },
        ContextualLevelReport {
            level: "fixed_route_moe_24_layers".to_owned(),
            timing: fixed_moe_timing,
            memory: memory(),
        },
        ContextualLevelReport {
            level: "complete_token_fixed_route".to_owned(),
            timing: complete_timing,
            memory: memory(),
        },
    ];
    println!("QX V2 contextual MoE profile");
    println!("candidate:          {candidate}");
    println!("artifact:           {}", artifact_path.display());
    println!("artifact bytes:     {artifact_bytes}");
    println!("artifact hash:      {artifact_hash:016x}");
    println!("context/position:   {context_length}/{position}");
    println!("iterations/warmup:  {iterations}/{warmup}");
    println!("routes frozen:      true (captured once before all measurements)");
    println!("level                              GPU us     wall us  dispatches  CBs");
    for level in &levels {
        let timing = level.timing;
        println!(
            "{:<34} {:>9.2}  {:>9.2}  {:>10.2}  {:>4.2}",
            level.level,
            timing.gpu_us_per_sample,
            timing.wall_us_per_sample,
            timing.dispatches_per_sample,
            timing.command_buffers_per_sample,
        );
        println!(
            "  memory: metal={} physical={}/{} RSS={}/{}",
            level.memory.known_metal_allocated_bytes,
            level.memory.current_physical_footprint_bytes,
            level.memory.peak_physical_footprint_bytes,
            level.memory.current_rss_bytes,
            level.memory.peak_rss_bytes,
        );
    }
    if let Some(path) = result_json {
        let report = ContextualProfileReport {
            schema_version: 1,
            mode: "v2_moe_contextual_profile",
            candidate: candidate.to_owned(),
            artifact: artifact_path.display().to_string(),
            artifact_hash: format!("{artifact_hash:016x}"),
            fixture_name: fixture.map(|value| value.name.clone()),
            fixture_domain: fixture.map(|value| value.domain.clone()),
            context: context_length,
            position,
            iterations,
            warmup,
            routes_frozen: true,
            route_source: "canonical top-8 captured once after fixture prefill",
            topology,
            levels,
        };
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("contextual result JSON: {}", path.display());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_canonical_pre_moe(
    graph: &mut qx_metal::platform::DecodeGraph,
    context: &qx_metal::platform::MetalContext,
    buffers: &qx_runtime_v2::GpuRuntimeBuffers,
    stack: &qx_runtime_v2::PreparedAttentionStack,
    position: usize,
) -> Result<()> {
    graph
        .encode_q4_embedding_lookup(
            &stack.embedding_lookup,
            buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
            EMBEDDING_MULTIPLIER,
        )
        .map_err(anyhow::Error::msg)?;

    // This is the real pre-MoE workload: embedding, attention-side layer
    // work, post-attention normalization, router, and top-k. MoE itself is
    // deliberately omitted so CB2 can contain the exact frozen-route MoE.
    // The hidden state after each layer is therefore a workload carrier, not
    // a semantic model output; this probe is about GPU interference only.
    for (layer, weights) in stack.layers.iter().enumerate() {
        let input = if layer.is_multiple_of(2) {
            qx_runtime_v2::BufferId::HiddenA
        } else {
            qx_runtime_v2::BufferId::HiddenB
        };
        let output = if layer.is_multiple_of(2) {
            qx_runtime_v2::BufferId::HiddenB
        } else {
            qx_runtime_v2::BufferId::HiddenA
        };
        buffers
            .encode_attention_layer(
                graph,
                layer,
                input,
                output,
                &weights.input_norm,
                &weights.qkv,
                &weights.o_proj,
                position,
            )
            .map_err(anyhow::Error::msg)?;
        let inverse_rms = buffers
            .buffer(qx_runtime_v2::BufferId::MoeScratch)
            .subregion(81_916, std::mem::size_of::<f32>())
            .map_err(anyhow::Error::msg)?;
        if qx_metal::platform::rms_scalar_router_enabled() {
            graph
                .encode_rmsnorm_scalar(
                    context,
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    &inverse_rms,
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_prepared_q4_dense_simd_normalized(
                    &weights.router,
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    &inverse_rms,
                    buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                )
                .map_err(anyhow::Error::msg)?;
        } else {
            graph
                .encode_rmsnorm(
                    context,
                    buffers.buffer(output),
                    &weights.post_attention_norm,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                )
                .map_err(anyhow::Error::msg)?;
            graph
                .encode_prepared_q4_dense_simd(
                    &weights.router,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                )
                .map_err(anyhow::Error::msg)?;
        }
        graph
            .encode_topk(
                context,
                buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                buffers.buffer(qx_runtime_v2::BufferId::TopKIds),
                buffers.buffer(qx_runtime_v2::BufferId::TopKWeights),
            )
            .map_err(anyhow::Error::msg)?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_moe_predecessor(
    graph: &mut qx_metal::platform::DecodeGraph,
    context: &qx_metal::platform::MetalContext,
    buffers: &qx_runtime_v2::GpuRuntimeBuffers,
    stack: &qx_runtime_v2::PreparedAttentionStack,
    moe: &qx_runtime_v2::PreparedMoeStack,
    route_trace: &RouteTrace,
    position: usize,
    mode: &str,
) -> Result<()> {
    match mode {
        "none" => Ok(()),
        "noop-24" => differential_noops(graph, context, LAYERS),
        "canonical-24" => encode_canonical_pre_moe(graph, context, buffers, stack, position),
        "compute-24" => {
            for layer in 0..LAYERS {
                let ids = route_trace.ids_for(layer)?;
                graph
                    .encode_dynamic_q4_moe_gate_up_synthetic(
                        &moe.layers[layer],
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        &ids,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    )
                    .map_err(anyhow::Error::msg)?;
            }
            Ok(())
        }
        "memory-24" => {
            for layer in 0..LAYERS {
                let ids = route_trace.ids_for(layer)?;
                graph
                    .encode_dynamic_q4_moe_gate_up_touch(
                        &moe.layers[layer],
                        &ids,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        0,
                    )
                    .map_err(anyhow::Error::msg)?;
            }
            Ok(())
        }
        "contiguous-memory-24" => {
            for layer in 0..LAYERS {
                let ids = route_trace.ids_for(layer)?;
                graph
                    .encode_dynamic_q4_moe_gate_up_touch(
                        &moe.layers[layer],
                        &ids,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        11,
                    )
                    .map_err(anyhow::Error::msg)?;
            }
            Ok(())
        }
        other => bail!(
            "unknown MoE predecessor {other}; expected none, noop-24, canonical-24, compute-24, contiguous-memory-24, or memory-24"
        ),
    }
}

#[cfg(target_os = "macos")]
fn benchmark_v2_moe_predecessor_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
    fixture: Option<&BenchmarkFixture>,
    result_json: Option<&std::path::Path>,
) -> Result<()> {
    use std::rc::Rc;
    if iterations == 0 {
        bail!("MoE predecessor profile iterations must be positive");
    }
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let mapped_artifact = std::sync::Arc::new(MappedArtifact::open(artifact_path)?);
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_mapped_artifact(
        context.as_ref(),
        mapped_artifact.clone(),
    )
    .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load_from_mapped_artifact(
        context.as_ref(),
        mapped_artifact,
    )
    .map_err(anyhow::Error::msg)?;
    let position = replay_contextual_fixture(
        context.as_ref(),
        &buffers,
        &stack,
        &moe,
        fixture,
        context_length,
    )?;
    let route_trace = capture_route_trace(context.as_ref(), &buffers, &stack, &moe, position)?;
    let modes = [
        ("none/control", "none"),
        ("24 no-ops", "noop-24"),
        ("canonical pre-MoE graph", "canonical-24"),
        ("compute-heavy", "compute-24"),
        ("contiguous-memory-heavy", "contiguous-memory-24"),
        ("selected-expert-memory-heavy", "memory-24"),
    ];
    let mut rows = Vec::with_capacity(modes.len());
    println!("QX two-command-buffer fixed-route MoE attribution");
    println!("artifact:         {}", artifact_path.display());
    println!("context/position: {context_length}/{position}");
    println!("routes frozen:    true");
    println!("CB1 predecessor is committed without a CPU wait; CB2 contains MoE only.");
    println!(
        "predecessor                         CB1 GPU us  CB2 MoE us  total GPU us   wall us  dispatches"
    );
    for (label, mode) in modes {
        let timing = profile_two_command_buffers(
            context.as_ref(),
            warmup,
            iterations,
            |graph| {
                encode_moe_predecessor(
                    graph,
                    context.as_ref(),
                    &buffers,
                    &stack,
                    &moe,
                    &route_trace,
                    position,
                    mode,
                )
            },
            |graph| encode_fixed_route_moe(graph, &buffers, &stack, &moe, &route_trace, false),
        )?;
        println!(
            "{label:<34} {:>11.2}  {:>10.2}  {:>12.2}  {:>8.2}  {:>10.2}",
            timing.predecessor_gpu_us,
            timing.moe_gpu_us,
            timing.total_gpu_us,
            timing.wall_us,
            timing.dispatches,
        );
        rows.push(serde_json::json!({
            "predecessor": mode,
            "label": label,
            "timing": timing,
        }));
    }
    if let Some(path) = result_json {
        let (artifact_bytes, artifact_hash) = artifact_fingerprint(artifact_path)?;
        let report = serde_json::json!({
            "schema_version": 1,
            "mode": "v2_moe_predecessor_cb2_profile",
            "artifact": artifact_path.display().to_string(),
            "artifact_bytes": artifact_bytes,
            "artifact_hash": format!("{artifact_hash:016x}"),
            "fixture": fixture.map(|value| value.name.clone()),
            "context": context_length,
            "position": position,
            "iterations": iterations,
            "warmup": warmup,
            "routes_frozen": true,
            "submission": "CB1 predecessor -> CB2 frozen-route MoE, one final CPU wait",
            "rows": rows,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("result JSON: {}", path.display());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn artifact_fingerprint(path: &std::path::Path) -> Result<(u64, u64)> {
    use std::hash::Hasher;
    use std::io::Read;

    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        hasher.write(&chunk[..read]);
    }
    Ok((size, hasher.finish()))
}

#[cfg(target_os = "macos")]
fn benchmark_v2_moe_differential_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
    fixture: Option<&BenchmarkFixture>,
    result_json: Option<&std::path::Path>,
) -> Result<()> {
    use std::rc::Rc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    let position = if let Some(fixture) = fixture {
        if fixture.measurement != MeasurementMode::PostPrefill {
            bail!("MoE differential fixtures must use post_prefill measurement");
        }
        if fixture.tokens.len() >= context_length {
            bail!("MoE differential fixture prefill must leave one decode position");
        }
        for (position, token) in fixture.tokens.iter().enumerate() {
            context
                .write_runtime_buffer(
                    buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                    &token.to_le_bytes(),
                )
                .map_err(anyhow::Error::msg)?;
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            stack
                .encode_token(&mut graph, &buffers, &moe, position)
                .map_err(anyhow::Error::msg)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        }
        fixture.tokens.len()
    } else {
        context_length.saturating_sub(1)
    };
    let (artifact_bytes, artifact_hash) = artifact_fingerprint(artifact_path)?;

    println!("QX V2 live MoE differential profile");
    println!("artifact:          {}", artifact_path.display());
    println!("artifact bytes:    {artifact_bytes}");
    println!("artifact hash:     {artifact_hash:016x} (streaming DefaultHasher fingerprint)");
    println!("context/position:  {context_length}/{position}");
    println!(
        "iterations:        {iterations} (warmup {})",
        warmup.min(iterations)
    );
    println!(
        "topology:          qkv={} rope_split={} attention_3stage={} rms_qkv={} rms_router={} v_simd={} softmax_simd={}",
        std::env::var("QX_V2_QKV_TOPOLOGY").unwrap_or_else(|_| "fused".to_owned()),
        qx_metal::platform::rope_kv_split_enabled(),
        qx_metal::platform::three_stage_attention_enabled(),
        qx_metal::platform::rms_scalar_qkv_enabled(),
        qx_metal::platform::rms_scalar_router_enabled(),
        qx_metal::platform::attention_v_simd_enabled(),
        qx_metal::platform::attention_softmax_simd_enabled(),
    );
    if let Some(source) = std::env::var_os("QX_V2_MOE_Q3_ARTIFACT") {
        let block_bytes = match std::env::var("QX_V2_MOE_Q3_LAYOUT")
            .ok()
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("native") | Some("padded") | Some("transaction") => 16,
            _ => 12,
        };
        let per_expert = 1024 * 32 * block_bytes + 1024 * 32 * 2;
        println!(
            "gate/up format:    Q3 block={} B, source={}, selected-top8 logical bytes/layer={} ({:.2} MiB)",
            block_bytes,
            std::path::PathBuf::from(source).display(),
            8 * per_expert,
            8.0 * per_expert as f64 / (1024.0 * 1024.0),
        );
    } else {
        println!(
            "gate/up format:    canonical Q4, selected-top8 logical bytes/layer={} ({:.2} MiB)",
            8 * (1024 * 1024 / 2 + 1024 * 32 * 2),
            8.0 * (1024 * 1024 / 2 + 1024 * 32 * 2) as f64 / (1024.0 * 1024.0),
        );
    }
    println!("capturing canonical top-8 routes for all 24 layers...");
    let route_trace = capture_route_trace(context.as_ref(), &buffers, &stack, &moe, position)?;
    let sorted_route_trace = sort_route_trace_by_expert_id(context.as_ref(), &route_trace)?;
    let direct_route_ids = context
        .read_runtime_u32s(&route_trace.ids, LAYERS * 8)
        .map_err(anyhow::Error::msg)?
        .chunks_exact(8)
        .map(|ids| ids.try_into().expect("top-k route has eight experts"))
        .collect::<Vec<[u32; 8]>>();
    const GATE_UP_CONTROL_BYTES: usize = 8 * (1024 * 1024 / 2 + 1024 * 32 * 2);
    let traffic_buffer = context
        .allocate_runtime_buffer(GATE_UP_CONTROL_BYTES)
        .map_err(anyhow::Error::msg)?;
    context
        .write_runtime_buffer(&traffic_buffer, &vec![0_u8; GATE_UP_CONTROL_BYTES])
        .map_err(anyhow::Error::msg)?;
    let variants = [
        ("canonical MoE", None),
        ("without gate/up", Some(0_u8)),
        ("without SiLU", Some(1_u8)),
        ("without down", Some(2_u8)),
        ("without accumulation", Some(3_u8)),
    ];
    println!("variant                       GPU us/token   dispatches   marginal saved us");
    let mut baseline_gpu = 0.0;
    let mut results = Vec::with_capacity(variants.len());
    for (name, stage_skip) in variants {
        let detailed =
            profile_graph_stage_detailed(context.as_ref(), warmup, iterations, |graph| {
                encode_differential_token(
                    graph,
                    &buffers,
                    &stack,
                    &moe,
                    position,
                    None,
                    Some(&route_trace),
                    Some(&direct_route_ids),
                    stage_skip,
                    None,
                    None,
                    None,
                )
            })?;
        if stage_skip.is_none() {
            baseline_gpu = detailed.0;
        }
        results.push((name, detailed));
    }
    let stage_rows = results
        .iter()
        .map(|(name, (gpu, dispatches))| DifferentialProfileRow {
            name: (*name).to_owned(),
            gpu_us_per_token: *gpu,
            dispatches: *dispatches,
            marginal_saved_us: baseline_gpu - *gpu,
        })
        .collect::<Vec<_>>();
    for row in &stage_rows {
        println!(
            "{:<30} {:>13.2}  {:>11}  {:>18.2}",
            row.name, row.gpu_us_per_token, row.dispatches, row.marginal_saved_us
        );
    }
    if let Some(path) = result_json {
        let mapped = MappedArtifact::open(artifact_path)?;
        let report = DifferentialProfileReport {
            schema_version: 2,
            qx_revision: option_env!("QX_GIT_REVISION").unwrap_or("unknown"),
            artifact: artifact_path.display().to_string(),
            artifact_hash: format!("{artifact_hash:016x}"),
            artifact_version: mapped.info().header.version,
            plan_id: mapped
                .info()
                .execution_plan
                .as_ref()
                .map(|plan| plan.plan_id.clone())
                .unwrap_or_else(|| "legacy-v1-unsealed".to_owned()),
            target_id: mapped.info().header.target_id.clone(),
            fixture_name: fixture.map(|value| value.name.clone()),
            fixture_domain: fixture.map(|value| value.domain.clone()),
            context: context_length,
            position,
            routes_frozen: true,
            iterations,
            warmup,
            rows: stage_rows,
        };
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("MoE stage result JSON: {}", path.display());
    }
    let controls = [
        ("canonical gate/up", None, None),
        (
            "gate/up weight-touch",
            Some(GateUpControl::WeightTouch),
            Some(0_u8),
        ),
        (
            "gate/up payload 25%",
            Some(GateUpControl::PayloadFraction(25)),
            Some(0_u8),
        ),
        (
            "gate/up payload 50%",
            Some(GateUpControl::PayloadFraction(50)),
            Some(0_u8),
        ),
        (
            "gate/up payload 75%",
            Some(GateUpControl::PayloadFraction(75)),
            Some(0_u8),
        ),
        (
            "gate/up payload 100%",
            Some(GateUpControl::PayloadFraction(100)),
            Some(0_u8),
        ),
        (
            "gate/up scales-only",
            Some(GateUpControl::ScalesTouch),
            Some(0_u8),
        ),
        (
            "gate/up contiguous 25%",
            Some(GateUpControl::ContiguousFraction(25)),
            Some(0_u8),
        ),
        (
            "gate/up contiguous 50%",
            Some(GateUpControl::ContiguousFraction(50)),
            Some(0_u8),
        ),
        (
            "gate/up contiguous 75%",
            Some(GateUpControl::ContiguousFraction(75)),
            Some(0_u8),
        ),
        (
            "gate/up contiguous 100%",
            Some(GateUpControl::ContiguousFraction(100)),
            Some(0_u8),
        ),
        (
            "gate/up synthetic arithmetic",
            Some(GateUpControl::SyntheticArithmetic),
            Some(0_u8),
        ),
        ("gate/up no-op", None, Some(0_u8)),
        (
            "gate/up 1 routed region",
            Some(GateUpControl::RegionCount(1)),
            Some(0_u8),
        ),
        (
            "gate/up 2 routed regions",
            Some(GateUpControl::RegionCount(2)),
            Some(0_u8),
        ),
        (
            "gate/up 4 routed regions",
            Some(GateUpControl::RegionCount(4)),
            Some(0_u8),
        ),
        (
            "gate/up 8 routed regions",
            Some(GateUpControl::RegionCount(8)),
            Some(0_u8),
        ),
        (
            "gate/up 1 direct region",
            Some(GateUpControl::DirectRegionCount(1)),
            Some(0_u8),
        ),
        (
            "gate/up 2 direct regions",
            Some(GateUpControl::DirectRegionCount(2)),
            Some(0_u8),
        ),
        (
            "gate/up 4 direct regions",
            Some(GateUpControl::DirectRegionCount(4)),
            Some(0_u8),
        ),
        (
            "gate/up 8 direct regions",
            Some(GateUpControl::DirectRegionCount(8)),
            Some(0_u8),
        ),
    ];
    println!();
    println!("gate/up control                 GPU us/token   dispatches   marginal saved us");
    let mut control_baseline = 0.0;
    let mut control_results = Vec::with_capacity(controls.len());
    for (name, control, stage_skip) in controls {
        let detailed =
            profile_graph_stage_detailed(context.as_ref(), warmup, iterations, |graph| {
                encode_differential_token(
                    graph,
                    &buffers,
                    &stack,
                    &moe,
                    position,
                    None,
                    Some(&route_trace),
                    Some(&direct_route_ids),
                    stage_skip,
                    control,
                    Some(&traffic_buffer),
                    None,
                )
            })?;
        if control.is_none() && stage_skip.is_none() {
            control_baseline = detailed.0;
        }
        control_results.push((name, detailed));
    }
    for (name, (gpu, dispatches)) in control_results {
        println!(
            "{name:<30} {gpu:>13.2}  {dispatches:>11}  {:>18.2}",
            control_baseline - gpu
        );
    }
    let route_orders = [
        ("canonical expert order", &route_trace),
        ("expert-id sorted order", &sorted_route_trace),
    ];
    println!();
    println!("full MoE route order            GPU us/token   dispatches   delta vs canonical");
    let mut canonical_order_gpu = 0.0;
    let mut order_results = Vec::with_capacity(route_orders.len());
    for (name, trace) in route_orders {
        let detailed =
            profile_graph_stage_detailed(context.as_ref(), warmup, iterations, |graph| {
                encode_differential_token(
                    graph,
                    &buffers,
                    &stack,
                    &moe,
                    position,
                    None,
                    Some(trace),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            })?;
        if name == "canonical expert order" {
            canonical_order_gpu = detailed.0;
        }
        order_results.push((name, detailed));
    }
    for (name, (gpu, dispatches)) in order_results {
        println!(
            "{name:<30} {gpu:>13.2}  {dispatches:>11}  {:>17.2}",
            gpu - canonical_order_gpu
        );
    }
    println!();
    println!("Routes are captured once from the canonical graph and reused by every variant.");
    println!(
        "Stage deltas are marginal full-token savings; they are not additive and must be followed by parity and sustained promotion gates."
    );
    println!(
        "The weight-touch control traverses selected Q4 payload/scales without dot products; the synthetic control preserves output geometry and reductions without Q4 weight traffic."
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_moe_prefetch_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;

    if context_length < 2 {
        bail!("--v2-moe-prefetch-profile requires context >= 2");
    }
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    let previous_position = context_length - 2;
    let current_position = context_length - 1;

    println!("capturing previous and current fixed routes...");
    let previous_trace =
        capture_route_trace(context.as_ref(), &buffers, &stack, &moe, previous_position)?;
    let mut advance = buffers.begin_token().map_err(anyhow::Error::msg)?;
    stack
        .encode_token(&mut advance, &buffers, &moe, previous_position)
        .map_err(anyhow::Error::msg)?;
    buffers.finish_token(advance).map_err(anyhow::Error::msg)?;
    let current_trace =
        capture_route_trace(context.as_ref(), &buffers, &stack, &moe, current_position)?;
    let previous_ids = context
        .read_runtime_u32s(&previous_trace.ids, LAYERS * 8)
        .map_err(anyhow::Error::msg)?
        .chunks_exact(8)
        .map(|ids| ids.try_into().expect("top-k route has eight experts"))
        .collect::<Vec<[u32; 8]>>();
    let current_ids = context
        .read_runtime_u32s(&current_trace.ids, LAYERS * 8)
        .map_err(anyhow::Error::msg)?
        .chunks_exact(8)
        .map(|ids| ids.try_into().expect("top-k route has eight experts"))
        .collect::<Vec<[u32; 8]>>();

    let variants = [
        ("canonical", None),
        ("previous-route prefetch", Some(previous_ids.as_slice())),
        ("oracle-route prefetch", Some(current_ids.as_slice())),
    ];
    println!("QX V2 MoE route-prefetch profile");
    println!("artifact:          {}", artifact_path.display());
    println!("context/positions:  {context_length}/{previous_position}->{current_position}");
    println!("iterations:         {iterations} (warmup {warmup})");
    println!("variant                       GPU us/token   dispatches   delta vs canonical");
    let mut canonical_gpu = 0.0;
    let mut results = Vec::with_capacity(variants.len());
    for (name, prefetch) in variants {
        let detailed =
            profile_graph_stage_detailed(context.as_ref(), warmup, iterations, |graph| {
                encode_differential_token(
                    graph,
                    &buffers,
                    &stack,
                    &moe,
                    current_position,
                    None,
                    Some(&current_trace),
                    None,
                    None,
                    None,
                    None,
                    prefetch,
                )
            })?;
        if name == "canonical" {
            canonical_gpu = detailed.0;
        }
        results.push((name, detailed));
    }
    for (name, (gpu, dispatches)) in results {
        println!(
            "{name:<30} {gpu:>13.2}  {dispatches:>11}  {:>18.2}",
            gpu - canonical_gpu
        );
    }
    println!(
        "Prefetch timing includes all 24 direct-address touch dispatches; no variant is a runtime promotion."
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_moe_direct_region_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    let position = context_length.saturating_sub(1);
    let route_trace = capture_route_trace(context.as_ref(), &buffers, &stack, &moe, position)?;
    let direct_route_ids = context
        .read_runtime_u32s(&route_trace.ids, LAYERS * 8)
        .map_err(anyhow::Error::msg)?
        .chunks_exact(8)
        .map(|ids| ids.try_into().expect("top-k route has eight experts"))
        .collect::<Vec<[u32; 8]>>();
    let variants = [
        ("gate/up no-op", None),
        ("direct 1 region", Some(GateUpControl::DirectRegionCount(1))),
        (
            "direct 2 regions",
            Some(GateUpControl::DirectRegionCount(2)),
        ),
        (
            "direct 4 regions",
            Some(GateUpControl::DirectRegionCount(4)),
        ),
        (
            "direct 8 regions",
            Some(GateUpControl::DirectRegionCount(8)),
        ),
    ];
    println!("QX V2 direct-address region profile");
    println!("artifact:          {}", artifact_path.display());
    println!("context/position:  {context_length}/{position}");
    println!("iterations:        {iterations} (warmup {warmup})");
    println!("variant                       GPU us/token   dispatches   cost vs no-op");
    let mut no_op_gpu = 0.0;
    let mut results = Vec::with_capacity(variants.len());
    for (name, control) in variants {
        let detailed =
            profile_graph_stage_detailed(context.as_ref(), warmup, iterations, |graph| {
                encode_differential_token(
                    graph,
                    &buffers,
                    &stack,
                    &moe,
                    position,
                    None,
                    Some(&route_trace),
                    Some(&direct_route_ids),
                    Some(0),
                    control,
                    None,
                    None,
                )
            })?;
        if control.is_none() {
            no_op_gpu = detailed.0;
        }
        results.push((name, detailed));
    }
    for (name, (gpu, dispatches)) in results {
        println!(
            "{name:<30} {gpu:>13.2}  {dispatches:>11}  {:>14.2}",
            gpu - no_op_gpu
        );
    }
    println!(
        "All variants use compile-time direct-address pipelines and the same eight-expert byte volume."
    );
    Ok(())
}

#[cfg(target_os = "macos")]
struct RouteCorpusStats {
    frequency: [[u32; 32]; LAYERS],
    cooccurrence: [[[u32; 32]; 32]; LAYERS],
}

#[cfg(target_os = "macos")]
fn collect_route_corpus_stats(
    context: &qx_metal::platform::MetalContext,
    buffers: &qx_runtime_v2::GpuRuntimeBuffers,
    stack: &qx_runtime_v2::PreparedAttentionStack,
    moe: &qx_runtime_v2::PreparedMoeStack,
    positions: usize,
) -> Result<RouteCorpusStats> {
    let mut previous: Option<Vec<u32>> = None;
    let mut frequency = [[0_u32; 32]; LAYERS];
    let mut cooccurrence = [[[0_u32; 32]; 32]; LAYERS];
    for position in 0..positions {
        let trace = capture_route_trace(context, buffers, stack, moe, position)?;
        let current = context
            .read_runtime_u32s(&trace.ids, LAYERS * 8)
            .map_err(anyhow::Error::msg)?;
        for (layer, experts) in current.chunks_exact(8).enumerate() {
            for &expert in experts {
                let expert = expert as usize;
                if expert < 32 {
                    frequency[layer][expert] += 1;
                }
            }
            for (left, &a) in experts.iter().enumerate() {
                for &b in experts.iter().skip(left + 1) {
                    let a = a as usize;
                    let b = b as usize;
                    if a < 32 && b < 32 {
                        cooccurrence[layer][a][b] += 1;
                        cooccurrence[layer][b][a] += 1;
                    }
                }
            }
        }
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, buffers, moe, position)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        previous = Some(current);
    }
    let _ = previous;
    Ok(RouteCorpusStats {
        frequency,
        cooccurrence,
    })
}

#[cfg(target_os = "macos")]
fn identity_expert_permutation() -> [[u8; 32]; LAYERS] {
    let mut permutation = [[0_u8; 32]; LAYERS];
    for layer in &mut permutation {
        for (expert, slot) in layer.iter_mut().enumerate() {
            *slot = expert as u8;
        }
    }
    permutation
}

#[cfg(target_os = "macos")]
fn frequency_expert_permutation(frequency: &[[u32; 32]; LAYERS]) -> [[u8; 32]; LAYERS] {
    let mut permutation = identity_expert_permutation();
    for (layer, counts) in frequency.iter().enumerate() {
        let mut experts = (0..32).collect::<Vec<_>>();
        experts.sort_unstable_by(|left, right| {
            counts[*right]
                .cmp(&counts[*left])
                .then_with(|| left.cmp(right))
        });
        for (slot, &expert) in experts.iter().enumerate() {
            permutation[layer][slot] = expert as u8;
        }
    }
    permutation
}

#[cfg(target_os = "macos")]
fn cooccurrence_expert_permutation(
    frequency: &[[u32; 32]; LAYERS],
    cooccurrence: &[[[u32; 32]; 32]; LAYERS],
) -> [[u8; 32]; LAYERS] {
    let mut permutation = identity_expert_permutation();
    for layer in 0..LAYERS {
        let mut remaining = (0..32).collect::<Vec<_>>();
        remaining.sort_unstable_by(|left, right| {
            frequency[layer][*right]
                .cmp(&frequency[layer][*left])
                .then_with(|| left.cmp(right))
        });
        let first = remaining.remove(0);
        let mut ordered = vec![first];
        while !remaining.is_empty() {
            let next_index = remaining
                .iter()
                .enumerate()
                .max_by_key(|(_, candidate)| {
                    (
                        ordered
                            .iter()
                            .map(|chosen| cooccurrence[layer][**candidate][*chosen])
                            .sum::<u32>(),
                        frequency[layer][**candidate],
                        std::cmp::Reverse(**candidate),
                    )
                })
                .map(|(index, _)| index)
                .expect("remaining expert set is non-empty");
            ordered.push(remaining.remove(next_index));
        }
        for (slot, &expert) in ordered.iter().enumerate() {
            permutation[layer][slot] = expert as u8;
        }
    }
    permutation
}

#[cfg(target_os = "macos")]
fn inverse_expert_permutation(permutation: &[[u8; 32]; LAYERS]) -> [[u8; 32]; LAYERS] {
    let mut inverse = [[0_u8; 32]; LAYERS];
    for layer in 0..LAYERS {
        for (slot, &logical) in permutation[layer].iter().enumerate() {
            inverse[layer][logical as usize] = slot as u8;
        }
    }
    inverse
}

#[cfg(target_os = "macos")]
fn remap_route_trace(
    context: &qx_metal::platform::MetalContext,
    trace: &RouteTrace,
    inverse: &[[u8; 32]; LAYERS],
) -> Result<RouteTrace> {
    let ids = context
        .read_runtime_u32s(&trace.ids, LAYERS * 8)
        .map_err(anyhow::Error::msg)?;
    let weights = context
        .read_runtime_f32(&trace.weights, LAYERS * 8)
        .map_err(anyhow::Error::msg)?;
    let remapped_ids = ids
        .chunks_exact(8)
        .enumerate()
        .flat_map(|(layer, route)| {
            route
                .iter()
                .map(move |&expert| inverse[layer][expert as usize] as u32)
        })
        .collect::<Vec<_>>();
    let id_buffer = context
        .allocate_runtime_buffer(remapped_ids.len() * std::mem::size_of::<u32>())
        .map_err(anyhow::Error::msg)?;
    let weight_buffer = context
        .allocate_runtime_buffer(weights.len() * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let id_bytes = remapped_ids
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let weight_bytes = weights
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    context
        .write_runtime_buffer(&id_buffer, &id_bytes)
        .map_err(anyhow::Error::msg)?;
    context
        .write_runtime_buffer(&weight_buffer, &weight_bytes)
        .map_err(anyhow::Error::msg)?;
    Ok(RouteTrace {
        ids: id_buffer,
        weights: weight_buffer,
    })
}

#[cfg(target_os = "macos")]
fn benchmark_v2_moe_permutation_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let corpus_moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    let corpus_plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let corpus_buffers = qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), corpus_plan)
        .map_err(anyhow::Error::msg)?;
    let corpus_positions = 32.min(context_length);
    let stats = collect_route_corpus_stats(
        context.as_ref(),
        &corpus_buffers,
        &stack,
        &corpus_moe,
        corpus_positions,
    )?;
    drop(corpus_buffers);
    drop(corpus_moe);

    let profile_plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let profile_buffers = qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), profile_plan)
        .map_err(anyhow::Error::msg)?;
    let base_moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    let position = context_length.saturating_sub(1);
    let logical_trace = capture_route_trace(
        context.as_ref(),
        &profile_buffers,
        &stack,
        &base_moe,
        position,
    )?;
    drop(base_moe);

    let permutations = [
        ("original", identity_expert_permutation()),
        (
            "frequency ordered",
            frequency_expert_permutation(&stats.frequency),
        ),
        (
            "co-selection clustered",
            cooccurrence_expert_permutation(&stats.frequency, &stats.cooccurrence),
        ),
    ];
    println!("QX V2 physical expert permutation profile");
    println!("artifact:          {}", artifact_path.display());
    println!("context/position:  {context_length}/{position}");
    println!("route corpus:      {corpus_positions} positions");
    println!("iterations:        {iterations} (warmup {warmup})");
    println!("variant                     full us/token  gate/up us  down us  dispatches");
    for (name, permutation) in permutations {
        let inverse = inverse_expert_permutation(&permutation);
        let remapped_trace = remap_route_trace(context.as_ref(), &logical_trace, &inverse)?;
        let moe = qx_runtime_v2::PreparedMoeStack::load_with_permutation(
            context.as_ref(),
            artifact_path,
            &permutation,
        )
        .map_err(anyhow::Error::msg)?;
        let mut timings = [0.0_f64; 3];
        let mut dispatches = 0_u64;
        for (index, stage_skip) in [None, Some(0_u8), Some(2_u8)].into_iter().enumerate() {
            let detailed =
                profile_graph_stage_detailed(context.as_ref(), warmup, iterations, |graph| {
                    encode_differential_token(
                        graph,
                        &profile_buffers,
                        &stack,
                        &moe,
                        position,
                        None,
                        Some(&remapped_trace),
                        None,
                        stage_skip,
                        None,
                        None,
                        None,
                    )
                })?;
            timings[index] = detailed.0;
            dispatches = detailed.1;
        }
        println!(
            "{name:<27} {:>13.2}  {:>10.2}  {:>7.2}  {dispatches:>10}",
            timings[0],
            timings[0] - timings[1],
            timings[0] - timings[2],
        );
    }
    println!(
        "Permutation profiles use fixed routes remapped to physical slots; promote only after full sequential parity and sustained generation validation."
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_route_overlap(
    artifact_path: &std::path::Path,
    context_length: usize,
    requested_positions: usize,
) -> Result<()> {
    use std::rc::Rc;

    if requested_positions < 2 {
        bail!("--v2-route-overlap requires at least two positions");
    }
    let positions = requested_positions.min(context_length);
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    context
        .write_runtime_buffer(
            buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
            &0_u32.to_le_bytes(),
        )
        .map_err(anyhow::Error::msg)?;

    let mut previous: Option<Vec<u32>> = None;
    let mut per_layer = vec![0.0_f64; LAYERS];
    let mut frequency = [[0_u32; 32]; LAYERS];
    let mut cooccurrence = vec![[[0_u32; 32]; 32]; LAYERS];
    let mut persistence = [[0_u32; 32]; LAYERS];
    let mut cumulative_mass = [[0.0_f64; 8]; LAYERS];
    let mut transitions = 0_usize;
    for position in 0..positions {
        let trace = capture_route_trace(context.as_ref(), &buffers, &stack, &moe, position)?;
        let current = context
            .read_runtime_u32s(&trace.ids, LAYERS * 8)
            .map_err(anyhow::Error::msg)?;
        let current_weights = context
            .read_runtime_f32(&trace.weights, LAYERS * 8)
            .map_err(anyhow::Error::msg)?;
        if let Some(previous) = previous.as_ref() {
            for (layer, (old, new)) in previous
                .chunks_exact(8)
                .zip(current.chunks_exact(8))
                .enumerate()
            {
                let overlap = new
                    .iter()
                    .filter(|expert| old.iter().any(|candidate| candidate == *expert))
                    .count();
                per_layer[layer] += overlap as f64 / 8.0;
                for &old_expert in old {
                    let old_index = old_expert as usize;
                    if old_index < 32 && new.contains(&old_expert) {
                        persistence[layer][old_index] += 1;
                    }
                }
            }
            transitions += 1;
        }
        for (layer, experts) in current.chunks_exact(8).enumerate() {
            let mut mass = 0.0;
            for (rank, &weight) in current_weights[layer * 8..layer * 8 + 8].iter().enumerate() {
                mass += weight as f64;
                cumulative_mass[layer][rank] += mass;
            }
            for &expert in experts {
                let index = expert as usize;
                if index < 32 {
                    frequency[layer][index] += 1;
                }
            }
            for left in 0..experts.len() {
                for right in left + 1..experts.len() {
                    let a = experts[left] as usize;
                    let b = experts[right] as usize;
                    if a < 32 && b < 32 {
                        cooccurrence[layer][a][b] += 1;
                        cooccurrence[layer][b][a] += 1;
                    }
                }
            }
        }
        // Advance the artifact-only token stream so the next capture sees the
        // next greedy token and the true sequential KV state.
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, position)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        previous = Some(current);
    }

    let mut total = 0.0;
    let mut min: f64 = 1.0;
    let mut max: f64 = 0.0;
    for value in &mut per_layer {
        *value /= transitions as f64;
        total += *value;
        min = min.min(*value);
        max = max.max(*value);
    }
    println!("QX V2 route-overlap diagnostic");
    println!("artifact:          {}", artifact_path.display());
    println!("context:           {context_length}");
    println!("positions:         {positions} ({transitions} transitions)");
    println!(
        "overlap definition: set intersection of consecutive top-8 expert IDs / 8; order ignored"
    );
    println!("mean top-8 overlap: {:.2}%", 100.0 * total / LAYERS as f64);
    println!(
        "layer range:       {:.2}% .. {:.2}%",
        100.0 * min,
        100.0 * max
    );
    println!("layer  overlap");
    for (layer, value) in per_layer.iter().enumerate() {
        println!("{layer:>5}  {:>7.2}%", 100.0 * value);
    }
    println!("mean cumulative normalized router mass by rank");
    println!("rank  cumulative mass");
    let mut mean_mass = [0.0_f64; 8];
    for layer_mass in cumulative_mass.iter().take(LAYERS) {
        for (rank, mass) in layer_mass.iter().enumerate() {
            mean_mass[rank] += mass / positions as f64;
        }
    }
    for (rank, mass) in mean_mass.iter_mut().enumerate() {
        *mass /= LAYERS as f64;
        println!("{:>4}  {:>16.4}", rank + 1, *mass);
    }
    println!("mean minimum active k for cumulative mass threshold");
    println!("threshold  active k");
    for threshold in [0.80_f64, 0.90, 0.95, 0.99] {
        let active_k = mean_mass
            .iter()
            .position(|&mass| mass >= threshold)
            .map(|rank| rank + 1)
            .unwrap_or(8);
        println!("{:>8.0}%  {:>8}", threshold * 100.0, active_k);
    }
    let mut pairs = Vec::with_capacity(LAYERS * 32 * 31 / 2);
    for (layer, layer_counts) in cooccurrence.iter().enumerate() {
        for (a, counts) in layer_counts.iter().enumerate() {
            for (b, &count) in counts.iter().enumerate().skip(a + 1) {
                pairs.push((count, layer, a, b));
            }
        }
    }
    pairs.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
    println!("top co-selected expert pairs across layers");
    println!("count  layer  experts");
    for (count, layer, a, b) in pairs.into_iter().take(10) {
        println!("{count:>5}  {layer:>5}  {a:>3} + {b:<3}");
    }
    let mut hot_experts = Vec::with_capacity(LAYERS * 32);
    for (layer, counts) in frequency.iter().enumerate() {
        for (expert, &count) in counts.iter().enumerate() {
            hot_experts.push((count, layer, expert));
        }
    }
    hot_experts.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
    println!("top selected experts across layers");
    println!("count  layer  expert");
    for (count, layer, expert) in hot_experts.into_iter().take(10) {
        println!("{count:>5}  {layer:>5}  {expert:>6}");
    }
    let mut route_persistence = Vec::with_capacity(LAYERS * 32);
    for (layer, counts) in persistence.iter().enumerate() {
        for (expert, &count) in counts.iter().enumerate() {
            if frequency[layer][expert] > 0 {
                route_persistence.push((
                    count as f64 / frequency[layer][expert] as f64,
                    count,
                    frequency[layer][expert],
                    layer,
                    expert,
                ));
            }
        }
    }
    route_persistence.sort_unstable_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| right.1.cmp(&left.1))
    });
    println!("top same-expert persistence across transitions");
    println!("probability  hits/selected  layer  expert");
    for (probability, hits, selected, layer, expert) in route_persistence.into_iter().take(10) {
        println!(
            "{:>10.2}%  {hits:>4}/{selected:<8}  {layer:>5}  {expert:>6}",
            100.0 * probability
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn capture_sequential_route_steps(
    artifact_path: &std::path::Path,
    context_length: usize,
    positions: usize,
) -> Result<Vec<HostRouteStep>> {
    use std::rc::Rc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    context
        .write_runtime_buffer(
            buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
            &0_u32.to_le_bytes(),
        )
        .map_err(anyhow::Error::msg)?;

    let mut steps = Vec::with_capacity(positions);
    for position in 0..positions {
        let trace = capture_route_trace(context.as_ref(), &buffers, &stack, &moe, position)?;
        let ids = context
            .read_runtime_u32s(&trace.ids, LAYERS * 8)
            .map_err(anyhow::Error::msg)?;
        let weights = context
            .read_runtime_f32(&trace.weights, LAYERS * 8)
            .map_err(anyhow::Error::msg)?;
        let mut step = HostRouteStep {
            ids: [[0; 8]; LAYERS],
            weights: [[0.0; 8]; LAYERS],
        };
        for layer in 0..LAYERS {
            step.ids[layer].copy_from_slice(&ids[layer * 8..layer * 8 + 8]);
            step.weights[layer].copy_from_slice(&weights[layer * 8..layer * 8 + 8]);
        }
        steps.push(step);

        // Advance the real artifact-only stream so the next capture has the
        // next token's KV state and router decisions.
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, position)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    Ok(steps)
}

#[cfg(target_os = "macos")]
fn route_overlap(a: &[u32; 8], b: &[u32; 8]) -> usize {
    b.iter()
        .filter(|expert| a.iter().any(|candidate| candidate == *expert))
        .count()
}

#[cfg(target_os = "macos")]
fn cold_route_sequence(realistic: &[HostRouteStep]) -> Vec<HostRouteStep> {
    let positions = realistic.len();
    let mut cold = vec![
        HostRouteStep {
            ids: [[0; 8]; LAYERS],
            weights: [[0.0; 8]; LAYERS],
        };
        positions
    ];
    for layer in 0..LAYERS {
        let mut remaining: Vec<usize> = (0..positions).collect();
        let mut previous = 0;
        for output in 0..positions {
            let chosen_offset = if output == 0 {
                remaining.iter().position(|&index| index == 0).unwrap_or(0)
            } else {
                remaining
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &index)| {
                        (
                            route_overlap(
                                &realistic[previous].ids[layer],
                                &realistic[index].ids[layer],
                            ),
                            index,
                        )
                    })
                    .map(|(offset, _)| offset)
                    .unwrap_or(0)
            };
            let source = remaining.remove(chosen_offset);
            cold[output].ids[layer] = realistic[source].ids[layer];
            cold[output].weights[layer] = realistic[source].weights[layer];
            previous = source;
        }
    }
    cold
}

#[cfg(target_os = "macos")]
fn route_sequence_bytes(sequence: &HostRouteStep) -> (Vec<u8>, Vec<u8>) {
    let ids: Vec<u32> = sequence
        .ids
        .iter()
        .flat_map(|layer| layer.iter())
        .copied()
        .collect();
    let weights: Vec<f32> = sequence
        .weights
        .iter()
        .flat_map(|layer| layer.iter())
        .copied()
        .collect();
    let ids_bytes = unsafe {
        std::slice::from_raw_parts(
            ids.as_ptr() as *const u8,
            std::mem::size_of_val(ids.as_slice()),
        )
        .to_vec()
    };
    let weights_bytes = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            std::mem::size_of_val(weights.as_slice()),
        )
        .to_vec()
    };
    (ids_bytes, weights_bytes)
}

#[cfg(target_os = "macos")]
fn profile_route_sequence(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    sequence: &[HostRouteStep],
) -> Result<(f64, f64, f64)> {
    use std::rc::Rc;
    use std::sync::Arc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let mapped = Arc::new(MappedArtifact::open(artifact_path)?);
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_mapped_artifact(
        context.as_ref(),
        mapped.clone(),
    )
    .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load_from_mapped_artifact(context.as_ref(), mapped)
        .map_err(anyhow::Error::msg)?;
    context
        .write_runtime_buffer(
            buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
            &0_u32.to_le_bytes(),
        )
        .map_err(anyhow::Error::msg)?;
    let route_ids = context
        .allocate_runtime_buffer(LAYERS * 8 * std::mem::size_of::<u32>())
        .map_err(anyhow::Error::msg)?;
    let route_weights = context
        .allocate_runtime_buffer(LAYERS * 8 * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let trace = RouteTrace {
        ids: route_ids,
        weights: route_weights,
    };
    let (ids_bytes, weights_bytes) = route_sequence_bytes(&sequence[0]);
    for _ in 0..warmup {
        context
            .write_runtime_buffer(&trace.ids, &ids_bytes)
            .map_err(anyhow::Error::msg)?;
        context
            .write_runtime_buffer(&trace.weights, &weights_bytes)
            .map_err(anyhow::Error::msg)?;
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        encode_fixed_route_moe(&mut graph, &buffers, &stack, &moe, &trace, false)?;
        graph.finish(context.as_ref()).map_err(anyhow::Error::msg)?;
    }
    let before = context.metrics().map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    for step in sequence {
        let (ids_bytes, weights_bytes) = route_sequence_bytes(step);
        context
            .write_runtime_buffer(&trace.ids, &ids_bytes)
            .map_err(anyhow::Error::msg)?;
        context
            .write_runtime_buffer(&trace.weights, &weights_bytes)
            .map_err(anyhow::Error::msg)?;
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        encode_fixed_route_moe(&mut graph, &buffers, &stack, &moe, &trace, false)?;
        graph.finish(context.as_ref()).map_err(anyhow::Error::msg)?;
    }
    let wall_us = start.elapsed().as_secs_f64() * 1e6 / sequence.len() as f64;
    let after = context.metrics().map_err(anyhow::Error::msg)?;
    let gpu_us =
        (after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6 / sequence.len() as f64;
    let dispatches = (after.dispatches - before.dispatches) as f64 / sequence.len() as f64;
    Ok((gpu_us, wall_us, dispatches))
}

#[cfg(target_os = "macos")]
fn profile_full_route_sequence(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    sequence: &[HostRouteStep],
) -> Result<(f64, f64, f64)> {
    use std::rc::Rc;
    use std::sync::Arc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let mapped = Arc::new(MappedArtifact::open(artifact_path)?);
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_mapped_artifact(
        context.as_ref(),
        mapped.clone(),
    )
    .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load_from_mapped_artifact(context.as_ref(), mapped)
        .map_err(anyhow::Error::msg)?;
    context
        .write_runtime_buffer(
            buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
            &0_u32.to_le_bytes(),
        )
        .map_err(anyhow::Error::msg)?;
    let trace = RouteTrace {
        ids: context
            .allocate_runtime_buffer(LAYERS * 8 * std::mem::size_of::<u32>())
            .map_err(anyhow::Error::msg)?,
        weights: context
            .allocate_runtime_buffer(LAYERS * 8 * std::mem::size_of::<f32>())
            .map_err(anyhow::Error::msg)?,
    };
    let run_one = |context: &qx_metal::platform::MetalContext,
                   buffers: &qx_runtime_v2::GpuRuntimeBuffers,
                   stack: &qx_runtime_v2::PreparedAttentionStack,
                   moe: &qx_runtime_v2::PreparedMoeStack,
                   trace: &RouteTrace,
                   step: &HostRouteStep,
                   position: usize|
     -> Result<()> {
        let (ids_bytes, weights_bytes) = route_sequence_bytes(step);
        context
            .write_runtime_buffer(&trace.ids, &ids_bytes)
            .map_err(anyhow::Error::msg)?;
        context
            .write_runtime_buffer(&trace.weights, &weights_bytes)
            .map_err(anyhow::Error::msg)?;
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        encode_differential_token(
            &mut graph,
            buffers,
            stack,
            moe,
            position,
            None,
            Some(trace),
            None,
            None,
            None,
            None,
            None,
        )?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
        Ok(())
    };
    for _ in 0..warmup {
        run_one(
            context.as_ref(),
            &buffers,
            &stack,
            &moe,
            &trace,
            &sequence[0],
            0,
        )?;
    }
    let before = context.metrics().map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    for (position, step) in sequence.iter().enumerate() {
        run_one(
            context.as_ref(),
            &buffers,
            &stack,
            &moe,
            &trace,
            step,
            position,
        )?;
    }
    let wall_us = start.elapsed().as_secs_f64() * 1e6 / sequence.len() as f64;
    let after = context.metrics().map_err(anyhow::Error::msg)?;
    let gpu_us =
        (after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6 / sequence.len() as f64;
    let dispatches = (after.dispatches - before.dispatches) as f64 / sequence.len() as f64;
    Ok((gpu_us, wall_us, dispatches))
}

#[cfg(target_os = "macos")]
fn benchmark_v2_route_churn_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    requested_positions: usize,
    result_json: Option<&std::path::Path>,
) -> Result<()> {
    if requested_positions < 2 {
        bail!("--v2-route-churn-profile requires at least two positions");
    }
    let positions = requested_positions.min(128).min(context_length);
    let realistic = capture_sequential_route_steps(artifact_path, context_length, positions)?;
    let hot = vec![realistic[0].clone(); positions];
    let cold = cold_route_sequence(&realistic);
    let sequences = [("hot", hot), ("realistic", realistic), ("cold", cold)];
    let mut reports = Vec::new();
    for (name, sequence) in sequences {
        let mut overlaps = Vec::new();
        let mut new_experts = Vec::new();
        let mut unique_regions = Vec::new();
        for (index, step) in sequence.iter().enumerate() {
            let mut overlap_sum = 0.0;
            let mut new_sum = 0.0;
            let mut unique_sum = 0.0;
            for layer in 0..LAYERS {
                let current = &step.ids[layer];
                let previous = if index == 0 {
                    current
                } else {
                    &sequence[index - 1].ids[layer]
                };
                let overlap = route_overlap(previous, current);
                overlap_sum += overlap as f64 / 8.0;
                new_sum += (8 - overlap) as f64;
                unique_sum += current
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len() as f64;
            }
            overlaps.push(overlap_sum / LAYERS as f64);
            new_experts.push(new_sum / LAYERS as f64);
            unique_regions.push(unique_sum / LAYERS as f64);
        }
        let (gpu_us, wall_us, dispatches) =
            profile_route_sequence(artifact_path, context_length, warmup, &sequence)?;
        let (full_gpu_us, full_wall_us, full_dispatches) =
            profile_full_route_sequence(artifact_path, context_length, warmup, &sequence)?;
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        println!(
            "{name:<10} GPU {:>8.2} us/token  wall {:>8.2} us/token  dispatches {:>6.1}  overlap {:>6.2}%  new {:>5.2}  regions {:>5.2}",
            gpu_us,
            wall_us,
            dispatches,
            100.0 * mean(&overlaps),
            mean(&new_experts),
            mean(&unique_regions),
        );
        println!(
            "{name:<10} full GPU {:>8.2} us/token  full wall {:>8.2} us/token  full dispatches {:>6.1}",
            full_gpu_us, full_wall_us, full_dispatches,
        );
        reports.push(serde_json::json!({
            "name": name,
            "gpu_us_per_token": gpu_us,
            "wall_us_per_token": wall_us,
            "dispatches_per_token": dispatches,
            "mean_top8_overlap": mean(&overlaps),
            "mean_new_experts": mean(&new_experts),
            "mean_unique_regions": mean(&unique_regions),
            "full_gpu_us_per_token": full_gpu_us,
            "full_wall_us_per_token": full_wall_us,
            "full_dispatches_per_token": full_dispatches,
        }));
    }
    println!("QX V2 route-churn MoE replay");
    println!("artifact: {}", artifact_path.display());
    println!("positions: {positions} (captured sequential route sets)");
    println!(
        "candidate: {}",
        if qx_metal::platform::moe_decode_direct_enabled() {
            "decode-direct"
        } else {
            "canonical"
        }
    );
    if let Some(path) = result_json {
        let report = serde_json::json!({
            "schema_version": 1,
            "mode": "v2_route_churn_profile",
            "artifact": artifact_path.display().to_string(),
            "context": context_length,
            "positions": positions,
            "warmup": warmup,
            "routes": reports,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("result JSON: {}", path.display());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_sequential_timeline(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
    fixture: Option<&BenchmarkFixture>,
    result_json: Option<&std::path::Path>,
) -> Result<()> {
    use std::rc::Rc;
    use std::sync::Arc;

    if iterations == 0 {
        bail!("--v2-sequential-timeline requires positive iterations");
    }
    let measurement = fixture
        .map(|value| value.measurement)
        .unwrap_or(MeasurementMode::SeedInclusive);
    let tokens = fixture
        .map(|value| value.tokens.as_slice())
        .unwrap_or(&[0_u32]);
    if measurement == MeasurementMode::SeedInclusive && tokens.len() != 1 {
        bail!("seed_inclusive timeline fixtures require one seed token");
    }
    let prefill_tokens = if measurement == MeasurementMode::PostPrefill {
        tokens.len()
    } else {
        0
    };
    if prefill_tokens.saturating_add(iterations) > context_length {
        bail!("prefill plus timeline positions exceed the context limit");
    }

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let mapped_artifact = Arc::new(MappedArtifact::open(artifact_path)?);
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_mapped_artifact(
        context.as_ref(),
        mapped_artifact.clone(),
    )
    .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load_from_mapped_artifact(
        context.as_ref(),
        mapped_artifact,
    )
    .map_err(anyhow::Error::msg)?;

    let seed = tokens[0].to_le_bytes();
    // Warm and prefill with counters disabled. The measured sequence must not
    // include pipeline compilation, fixture replay, or counter setup.
    for _ in 0..warmup {
        context
            .write_runtime_buffer(buffers.buffer(qx_runtime_v2::BufferId::TokenOutput), &seed)
            .map_err(anyhow::Error::msg)?;
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, 0)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    if measurement == MeasurementMode::PostPrefill {
        for (position, token) in tokens.iter().enumerate() {
            context
                .write_runtime_buffer(
                    buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                    &token.to_le_bytes(),
                )
                .map_err(anyhow::Error::msg)?;
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            stack
                .encode_token(&mut graph, &buffers, &moe, position)
                .map_err(anyhow::Error::msg)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        }
    } else {
        context
            .write_runtime_buffer(buffers.buffer(qx_runtime_v2::BufferId::TokenOutput), &seed)
            .map_err(anyhow::Error::msg)?;
    }

    context
        .set_stage_counters(true)
        .map_err(anyhow::Error::msg)?;
    let mut per_stage: Vec<Vec<f64>> = vec![Vec::with_capacity(iterations); 11];
    let expected_markers = LAYERS * 10 + 2;
    let expected_fallback_segments = LAYERS * 6 + 2;
    let mut counter_mode = None;
    let mut total_gpu_us_samples = Vec::with_capacity(iterations);
    let start = Instant::now();
    for offset in 0..iterations {
        let position = prefill_tokens + offset;
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        encode_differential_token(
            &mut graph, &buffers, &stack, &moe, position, None, None, None, None, None, None, None,
        )?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        let timeline = context.last_stage_timeline().map_err(anyhow::Error::msg)?;
        total_gpu_us_samples.push(timeline.gpu_duration_seconds * 1e6);
        let has_counters = !timeline.timestamps.is_empty();
        if let Some(previous) = counter_mode {
            if previous != has_counters {
                bail!("Metal timeline mode changed during the measured sequence");
            }
        } else {
            counter_mode = Some(has_counters);
        }
        if has_counters {
            if timeline.timestamps.len() != expected_markers {
                bail!(
                    "Metal timestamp timeline returned {} markers; expected {}",
                    timeline.timestamps.len(),
                    expected_markers
                );
            }
            let first = *timeline.timestamps.first().unwrap() as f64;
            let last = *timeline.timestamps.last().unwrap() as f64;
            let tick_span = last - first;
            if tick_span <= 0.0 || timeline.gpu_duration_seconds <= 0.0 {
                bail!("Metal timestamp timeline is invalid for position {position}");
            }
            let us_per_tick = timeline.gpu_duration_seconds * 1e6 / tick_span;
            let interval = |a: usize, b: usize| {
                (timeline.timestamps[b] as f64 - timeline.timestamps[a] as f64) * us_per_tick
            };
            for layer in 0..LAYERS {
                let base = 1 + layer * 10;
                for (stage, (a, b)) in [
                    (0, (base - 1, base)),
                    (1, (base, base + 1)),
                    (2, (base + 1, base + 2)),
                    (3, (base + 2, base + 3)),
                    (4, (base + 3, base + 4)),
                    (5, (base + 4, base + 5)),
                    (6, (base + 5, base + 6)),
                    (7, (base + 6, base + 7)),
                    (8, (base + 7, base + 8)),
                    (9, (base + 8, base + 9)),
                ] {
                    per_stage[stage].push(interval(a, b));
                }
            }
            let final_start = LAYERS * 10;
            per_stage[10].push(interval(final_start, final_start + 1));
        } else {
            if timeline.segment_gpu_seconds.len() < expected_fallback_segments {
                bail!(
                    "Metal fallback timeline returned {} command-buffer segments; expected at least {}",
                    timeline.segment_gpu_seconds.len(),
                    expected_fallback_segments
                );
            }
            // Segment zero is embedding and the final segment is the empty
            // command buffer created after the last marker. The middle
            // segments are the real logical intervals.
            for layer in 0..LAYERS {
                let base = 1 + layer * 6;
                for (stage, segment) in [
                    (0, base),
                    (1, base + 1),
                    (2, base + 2),
                    (3, base + 3),
                    (4, base + 4),
                    (5, base + 5),
                ] {
                    per_stage[stage].push(timeline.segment_gpu_seconds[segment] * 1e6);
                }
            }
            per_stage[6].push(timeline.segment_gpu_seconds[1 + LAYERS * 6] * 1e6);
        }
    }
    context
        .set_stage_counters(false)
        .map_err(anyhow::Error::msg)?;

    let detailed_names = [
        "input_rms_qkv",
        "rope_kv",
        "attention",
        "o_residual",
        "router_topk",
        "moe_gate_up",
        "moe_silu",
        "moe_down",
        "moe_accumulation",
        "moe_residual",
        "final_head",
    ];
    let fallback_names = [
        "input_rms_qkv",
        "rope_kv",
        "attention",
        "o_residual",
        "router_topk",
        "moe_plus_residual",
        "final_head",
    ];
    let names: &[&str] = if counter_mode == Some(true) {
        &detailed_names
    } else {
        &fallback_names
    };
    let mut rows = Vec::with_capacity(names.len());
    println!("QX V2 sequential in-graph timeline");
    println!(
        "candidate: {}",
        if qx_metal::platform::moe_decode_direct_enabled() {
            "decode-direct"
        } else {
            "canonical"
        }
    );
    println!(
        "fixture: {}",
        fixture.map(|value| value.name.as_str()).unwrap_or("seed")
    );
    println!(
        "positions: {}..{}",
        prefill_tokens,
        prefill_tokens + iterations - 1
    );
    println!("stage                         median us/token   median us/layer");
    for (index, name) in names.iter().enumerate() {
        let median = median_sample(&mut per_stage[index]);
        let per_layer = if (counter_mode == Some(true) && index < 10)
            || (counter_mode != Some(true) && index < 6)
        {
            median / LAYERS as f64
        } else {
            median
        };
        println!("{name:<28} {median:>16.2} {per_layer:>18.2}");
        rows.push(serde_json::json!({
            "stage": name,
            "gpu_us_per_token_median": median,
            "gpu_us_per_layer_median": per_layer,
        }));
    }
    let wall_us_per_token = start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
    let total_gpu_us_per_token = median_sample(&mut total_gpu_us_samples);
    let (artifact_bytes, artifact_hash) = artifact_fingerprint(artifact_path)?;
    println!("wall us/token: {:.2}", wall_us_per_token);
    println!(
        "GPU us/token (all command buffers): {:.2}",
        total_gpu_us_per_token
    );
    println!("artifact hash: {artifact_hash:016x}");
    if let Some(path) = result_json {
        let report = serde_json::json!({
            "schema_version": 1,
            "mode": "v2_sequential_in_graph_timeline",
            "candidate": if qx_metal::platform::moe_decode_direct_enabled() { "decode-direct" } else { "canonical" },
            "artifact": artifact_path.display().to_string(),
            "artifact_bytes": artifact_bytes,
            "artifact_hash": format!("{artifact_hash:016x}"),
            "fixture": fixture.map(|value| value.name.clone()),
            "measurement": "artifact_only_sequential_after_prefill",
            "context": context_length,
            "prefill_tokens": prefill_tokens,
            "iterations": iterations,
            "expected_markers_per_token": expected_markers,
            "stages": rows,
            "wall_us_per_token": wall_us_per_token,
            "gpu_us_per_token_all_command_buffers": total_gpu_us_per_token,
            "timeline_mode": if counter_mode == Some(true) { "dispatch_counters" } else { "command_buffer_fallback" },
            "warning": "stage intervals are diagnostic; the fallback changes command-buffer boundaries and is not a production schedule",
        });
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("result JSON: {}", path.display());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_sequential_timeline(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
    _fixture: Option<&BenchmarkFixture>,
    _result_json: Option<&std::path::Path>,
) -> Result<()> {
    bail!("--v2-sequential-timeline requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_differential_profile(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
    _fixture: Option<&BenchmarkFixture>,
    _result_json: Option<&std::path::Path>,
) -> Result<()> {
    bail!("--v2-differential-profile requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_moe_differential_profile(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
    _fixture: Option<&BenchmarkFixture>,
    _result_json: Option<&std::path::Path>,
) -> Result<()> {
    bail!("--v2-moe-differential-profile requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_moe_contextual_profile(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
    _fixture: Option<&BenchmarkFixture>,
    _candidate: &str,
    _result_json: Option<&std::path::Path>,
) -> Result<()> {
    bail!("--v2-moe-contextual-profile requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_moe_predecessor_profile(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
    _fixture: Option<&BenchmarkFixture>,
    _result_json: Option<&std::path::Path>,
) -> Result<()> {
    bail!("--v2-moe-predecessor-profile requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_moe_prefetch_profile(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-moe-prefetch-profile requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_moe_direct_region_profile(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-moe-direct-region-profile requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_moe_permutation_profile(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-moe-permutation-profile requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_route_overlap(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _requested_positions: usize,
) -> Result<()> {
    bail!("--v2-route-overlap requires macOS Metal")
}

#[cfg(target_os = "macos")]
fn benchmark_v2_qkv_sweep(artifact_path: &std::path::Path, iterations: usize) -> Result<()> {
    use std::rc::Rc;

    if iterations == 0 {
        bail!("QKV sweep iterations must be positive");
    }
    let artifact = MappedArtifact::open(artifact_path)?;
    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let find = |kind: TensorKind| {
        artifact
            .info()
            .entries
            .iter()
            .position(|entry| entry.layer == 0 && entry.expert == u8::MAX && entry.kind == kind)
            .ok_or_else(|| anyhow::anyhow!("artifact tensor {kind:?} not found"))
    };
    let q = artifact.quantized_tensor(find(TensorKind::QProj)?)?;
    let k = artifact.quantized_tensor(find(TensorKind::KProj)?)?;
    let v = artifact.quantized_tensor(find(TensorKind::VProj)?)?;
    if (q.rows, q.cols, k.rows, k.cols, v.rows, v.cols) != (1024, 1024, 512, 1024, 512, 1024) {
        bail!("artifact does not contain the expected Granite QKV shapes");
    }
    let q_prepared = context.prepare_q4_dense(&q).map_err(anyhow::Error::msg)?;
    let k_prepared = context.prepare_q4_dense(&k).map_err(anyhow::Error::msg)?;
    let v_prepared = context.prepare_q4_dense(&v).map_err(anyhow::Error::msg)?;
    let qkv = context
        .prepare_q4_qkv(&q_prepared, &k_prepared, &v_prepared)
        .map_err(anyhow::Error::msg)?;
    let input = vec![1.0_f32; 1024];
    let norm_weights = vec![1.0_f32; 1024];
    let warmup = 5.min(iterations);
    let fused =
        run_qkv_fused_microbench(&context, &qkv, &input, &norm_weights, warmup, iterations)?;
    let fused_output = run_qkv_fused_output(&context, &qkv, &input, &norm_weights)?;
    let reference = context
        .run_prepared_q4_qkv_sweep(&qkv, &input, &norm_weights, 1.0, 2, 1, 8, true, true)
        .map_err(anyhow::Error::msg)?;
    let mut results = Vec::new();
    for q_rows_per_simd in [1_usize, 2, 4, 8] {
        for kv_rows_per_simd in [1_usize, 2, 4, 8] {
            for simd_groups in [1_usize, 2, 4, 8] {
                for cache_input in [false, true] {
                    let result = run_qkv_sweep_microbench(
                        &context,
                        &qkv,
                        &input,
                        &norm_weights,
                        q_rows_per_simd,
                        kv_rows_per_simd,
                        simd_groups,
                        cache_input,
                        warmup,
                        iterations,
                    )?;
                    results.push((
                        q_rows_per_simd,
                        kv_rows_per_simd,
                        simd_groups,
                        cache_input,
                        result,
                    ));
                }
            }
        }
    }
    results.sort_by(|left, right| left.4.gpu_median_us.total_cmp(&right.4.gpu_median_us));
    println!("QX fused QKV SIMD geometry sweep (one dispatch, normalized input)");
    println!("artifact: {}", artifact_path.display());
    println!("iterations: {iterations} (warmup {warmup})");
    println!(
        "fused production control: {:.2} us GPU median, p95 {:.2} us, CV {:.2}",
        fused.gpu_median_us, fused.gpu_p95_us, fused.gpu_cv
    );
    let max_abs = fused_output
        .iter()
        .zip(&reference)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    println!(
        "sweep correctness check (Q/SIMD=2, KV/SIMD=1, groups=8, cached): max abs {max_abs:.6}"
    );
    println!(
        "Q rows/SIMD  KV rows/SIMD  SIMD groups  cache input  threads/tg  threadgroups  GPU med us  p95 us  CV"
    );
    for (q_rows_per_simd, kv_rows_per_simd, simd_groups, cache_input, result) in results {
        let q_groups = 1024_usize.div_ceil(q_rows_per_simd * simd_groups);
        let kv_groups = 512_usize.div_ceil(kv_rows_per_simd * simd_groups);
        println!(
            "{q_rows_per_simd:>11}  {kv_rows_per_simd:>12}  {simd_groups:>11}  {cache:>11}  {threads:>10}  {threadgroups:>12}  {median:>10.2}  {p95:>6.2}  {cv:>4.2}",
            cache = if cache_input { "yes" } else { "no" },
            threads = simd_groups * 32,
            threadgroups = q_groups + 2 * kv_groups,
            median = result.gpu_median_us,
            p95 = result.gpu_p95_us,
            cv = result.gpu_cv,
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_qkv_fused_output(
    context: &std::rc::Rc<qx_metal::platform::MetalContext>,
    prepared: &qx_metal::platform::PreparedQuantizedQkv,
    input_values: &[f32],
    norm_weights_values: &[f32],
) -> Result<Vec<f32>> {
    let input = context
        .allocate_runtime_buffer(std::mem::size_of_val(input_values))
        .map_err(anyhow::Error::msg)?;
    let norm_weights = context
        .allocate_runtime_buffer(std::mem::size_of_val(norm_weights_values))
        .map_err(anyhow::Error::msg)?;
    let inverse = context
        .allocate_runtime_buffer(std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let query = context
        .allocate_runtime_buffer(1024 * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let key = context
        .allocate_runtime_buffer(512 * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let value = context
        .allocate_runtime_buffer(512 * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    for (buffer, values) in [(&input, input_values), (&norm_weights, norm_weights_values)] {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        };
        context
            .write_runtime_buffer(buffer, bytes)
            .map_err(anyhow::Error::msg)?;
    }
    context
        .write_runtime_buffer(&inverse, &1.0_f32.to_ne_bytes())
        .map_err(anyhow::Error::msg)?;
    let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
    graph
        .encode_prepared_q4_qkv_simd_normalized(
            prepared,
            &input,
            &norm_weights,
            &inverse,
            &query,
            &key,
            &value,
        )
        .map_err(anyhow::Error::msg)?;
    graph.finish(context.as_ref()).map_err(anyhow::Error::msg)?;
    let mut output = context
        .read_runtime_f32(&query, 1024)
        .map_err(anyhow::Error::msg)?;
    output.extend(
        context
            .read_runtime_f32(&key, 512)
            .map_err(anyhow::Error::msg)?,
    );
    output.extend(
        context
            .read_runtime_f32(&value, 512)
            .map_err(anyhow::Error::msg)?,
    );
    Ok(output)
}

#[cfg(target_os = "macos")]
fn run_qkv_fused_microbench(
    context: &std::rc::Rc<qx_metal::platform::MetalContext>,
    prepared: &qx_metal::platform::PreparedQuantizedQkv,
    input_values: &[f32],
    norm_weights_values: &[f32],
    warmup: usize,
    iterations: usize,
) -> Result<KernelStats> {
    let input = context
        .allocate_runtime_buffer(std::mem::size_of_val(input_values))
        .map_err(anyhow::Error::msg)?;
    let norm_weights = context
        .allocate_runtime_buffer(std::mem::size_of_val(norm_weights_values))
        .map_err(anyhow::Error::msg)?;
    let inverse = context
        .allocate_runtime_buffer(std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let query = context
        .allocate_runtime_buffer(1024 * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let key = context
        .allocate_runtime_buffer(512 * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let value = context
        .allocate_runtime_buffer(512 * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    for (buffer, values) in [(&input, input_values), (&norm_weights, norm_weights_values)] {
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        };
        context
            .write_runtime_buffer(buffer, bytes)
            .map_err(anyhow::Error::msg)?;
    }
    let inverse_value = 1.0_f32.to_ne_bytes();
    context
        .write_runtime_buffer(&inverse, &inverse_value)
        .map_err(anyhow::Error::msg)?;
    for _ in 0..warmup {
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        graph
            .encode_prepared_q4_qkv_simd_normalized(
                prepared,
                &input,
                &norm_weights,
                &inverse,
                &query,
                &key,
                &value,
            )
            .map_err(anyhow::Error::msg)?;
        graph.finish(context.as_ref()).map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        graph
            .encode_prepared_q4_qkv_simd_normalized(
                prepared,
                &input,
                &norm_weights,
                &inverse,
                &query,
                &key,
                &value,
            )
            .map_err(anyhow::Error::msg)?;
        graph.finish(context.as_ref()).map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        wall_samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1_000_000.0);
    }
    Ok(summarize_kernel_samples(
        &mut gpu_samples,
        &mut wall_samples,
    ))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_qkv_sweep_microbench(
    context: &std::rc::Rc<qx_metal::platform::MetalContext>,
    prepared: &qx_metal::platform::PreparedQuantizedQkv,
    input: &[f32],
    norm_weights: &[f32],
    q_rows_per_simd: usize,
    kv_rows_per_simd: usize,
    simd_groups: usize,
    cache_input: bool,
    warmup: usize,
    iterations: usize,
) -> Result<KernelStats> {
    for _ in 0..warmup {
        context
            .run_prepared_q4_qkv_sweep(
                prepared,
                input,
                norm_weights,
                1.0,
                q_rows_per_simd,
                kv_rows_per_simd,
                simd_groups,
                cache_input,
                true,
            )
            .map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        context
            .run_prepared_q4_qkv_sweep(
                prepared,
                input,
                norm_weights,
                1.0,
                q_rows_per_simd,
                kv_rows_per_simd,
                simd_groups,
                cache_input,
                true,
            )
            .map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        wall_samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1_000_000.0);
    }
    Ok(summarize_kernel_samples(
        &mut gpu_samples,
        &mut wall_samples,
    ))
}

#[cfg(target_os = "macos")]
fn run_q4_dense_microbench(
    context: &qx_metal::platform::MetalContext,
    prepared: &qx_metal::platform::PreparedQuantizedDense,
    input_values: &[f32],
    warmup: usize,
    iterations: usize,
) -> Result<KernelStats> {
    let input = context
        .allocate_runtime_buffer(std::mem::size_of_val(input_values))
        .map_err(anyhow::Error::msg)?;
    let output = context
        .allocate_runtime_buffer(prepared.rows() * std::mem::size_of::<f32>())
        .map_err(anyhow::Error::msg)?;
    let input_bytes = unsafe {
        std::slice::from_raw_parts(
            input_values.as_ptr() as *const u8,
            std::mem::size_of_val(input_values),
        )
    };
    context
        .write_runtime_buffer(&input, input_bytes)
        .map_err(anyhow::Error::msg)?;
    for _ in 0..warmup {
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        graph
            .encode_prepared_q4_dense(prepared, &input, &output)
            .map_err(anyhow::Error::msg)?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        graph
            .encode_prepared_q4_dense(prepared, &input, &output)
            .map_err(anyhow::Error::msg)?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        wall_samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1_000_000.0);
    }
    Ok(summarize_kernel_samples(
        &mut gpu_samples,
        &mut wall_samples,
    ))
}

#[cfg(target_os = "macos")]
fn run_q4_stage_microbench(
    context: &qx_metal::platform::MetalContext,
    prepared: &qx_metal::platform::PreparedQ4Stage,
    input: &[u16],
    warmup: usize,
    iterations: usize,
) -> Result<KernelStats> {
    for _ in 0..warmup {
        context
            .run_prepared_q4_stage(prepared, input)
            .map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        context
            .run_prepared_q4_stage(prepared, input)
            .map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        wall_samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1_000_000.0);
    }
    Ok(summarize_kernel_samples(
        &mut gpu_samples,
        &mut wall_samples,
    ))
}

#[cfg(target_os = "macos")]
fn run_q4_simd_stage_microbench(
    context: &qx_metal::platform::MetalContext,
    prepared: &qx_metal::platform::PreparedQ4SimdStage,
    input: &[u16],
    warmup: usize,
    iterations: usize,
) -> Result<KernelStats> {
    for _ in 0..warmup {
        context
            .run_prepared_q4_simd_stage(prepared, input)
            .map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        context
            .run_prepared_q4_simd_stage(prepared, input)
            .map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        wall_samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1_000_000.0);
    }
    Ok(summarize_kernel_samples(
        &mut gpu_samples,
        &mut wall_samples,
    ))
}

#[cfg(target_os = "macos")]
fn run_q4_simd_microbench(
    context: &qx_metal::platform::MetalContext,
    prepared: &qx_metal::platform::PreparedQuantizedSimdDense,
    input: &[f32],
    warmup: usize,
    iterations: usize,
) -> Result<KernelStats> {
    for _ in 0..warmup {
        context
            .run_prepared_q4_dense_simd(prepared, input)
            .map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        context
            .run_prepared_q4_dense_simd(prepared, input)
            .map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        wall_samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1_000_000.0);
    }
    Ok(summarize_kernel_samples(
        &mut gpu_samples,
        &mut wall_samples,
    ))
}

#[cfg(target_os = "macos")]
fn run_f16_dense_microbench(
    context: &qx_metal::platform::MetalContext,
    prepared: &qx_metal::platform::PreparedF16Dense,
    input: &[u16],
    warmup: usize,
    iterations: usize,
) -> Result<KernelStats> {
    for _ in 0..warmup {
        context
            .run_prepared_f16_dense(prepared, input)
            .map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        context
            .run_prepared_f16_dense(prepared, input)
            .map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        wall_samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1_000_000.0);
    }
    Ok(summarize_kernel_samples(
        &mut gpu_samples,
        &mut wall_samples,
    ))
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn run_q4_sweep_microbench(
    context: &qx_metal::platform::MetalContext,
    prepared: &qx_metal::platform::PreparedQuantizedSimdDense,
    input: &[f32],
    rows_per_simd: usize,
    simd_groups: usize,
    cache_input: bool,
    warmup: usize,
    iterations: usize,
) -> Result<KernelStats> {
    for _ in 0..warmup {
        context
            .run_prepared_q4_dense_sweep(prepared, input, rows_per_simd, simd_groups, cache_input)
            .map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut wall_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let start = Instant::now();
        context
            .run_prepared_q4_dense_sweep(prepared, input, rows_per_simd, simd_groups, cache_input)
            .map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        wall_samples.push(start.elapsed().as_secs_f64() * 1_000_000.0);
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1_000_000.0);
    }
    Ok(summarize_kernel_samples(
        &mut gpu_samples,
        &mut wall_samples,
    ))
}

#[cfg(target_os = "macos")]
fn print_kernel_row(
    label: &str,
    format: &str,
    weight_bytes: usize,
    input_bytes: usize,
    output_bytes: usize,
    result: KernelStats,
) {
    let total_bytes = weight_bytes + input_bytes + output_bytes;
    let effective_gbps =
        total_bytes as f64 / (result.gpu_median_us * 1_000.0).max(f64::MIN_POSITIVE);
    println!(
        "{label:<20} {format:>4} {weight_bytes:>10} {input_bytes:>9} {output_bytes:>9} {total_bytes:>9} {gpu:>8.2} {p95:>6.2} {cv:>5.2} {wall:>9.2} {effective:>14.2}",
        gpu = result.gpu_median_us,
        p95 = result.gpu_p95_us,
        cv = result.gpu_cv,
        wall = result.wall_median_us,
        effective = effective_gbps,
    );
}

#[cfg(target_os = "macos")]
fn summarize_kernel_samples(gpu_samples: &mut [f64], wall_samples: &mut [f64]) -> KernelStats {
    let gpu_median_us = median_sample(gpu_samples);
    let gpu_p95_us = percentile_sample(gpu_samples, 0.95);
    let mean = gpu_samples.iter().sum::<f64>() / gpu_samples.len() as f64;
    let variance = gpu_samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / gpu_samples.len() as f64;
    KernelStats {
        gpu_median_us,
        gpu_p95_us,
        gpu_cv: variance.sqrt() / mean.max(f64::MIN_POSITIVE),
        wall_median_us: median_sample(wall_samples),
    }
}

#[cfg(target_os = "macos")]
fn to_f16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|value| f16::from_f32(*value).to_bits())
        .collect()
}

#[cfg(target_os = "macos")]
fn max_half_abs_diff(left: &[u16], right: &[u16]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            (f16::from_bits(*left).to_f32() - f16::from_bits(*right).to_f32()).abs()
        })
        .fold(0.0, f32::max)
}

#[cfg(target_os = "macos")]
fn median_sample(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[cfg(target_os = "macos")]
fn percentile_sample(values: &[f64], percentile: f64) -> f64 {
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index.min(values.len() - 1)]
}

#[cfg(target_os = "macos")]
fn benchmark_v2_kernel_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;
    use std::sync::Arc;

    if iterations == 0 {
        bail!("kernel profile iterations must be positive");
    }
    let metal = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(metal.clone(), plan).map_err(anyhow::Error::msg)?;
    let mapped_artifact = Arc::new(MappedArtifact::open(artifact_path)?);
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_mapped_artifact(
        &metal,
        mapped_artifact.clone(),
    )
    .map_err(anyhow::Error::msg)?;
    let moe =
        qx_runtime_v2::PreparedMoeStack::load_from_mapped_artifact(&metal, mapped_artifact.clone())
            .map_err(anyhow::Error::msg)?;
    let token = 0_u32.to_le_bytes();
    metal
        .write_runtime_buffer(buffers.buffer(qx_runtime_v2::BufferId::TokenOutput), &token)
        .map_err(anyhow::Error::msg)?;
    let ids = [0_u32, 1, 2, 3, 4, 5, 6, 7];
    let weights = [0.125_f32; 8];
    metal
        .write_runtime_buffer(buffers.buffer(qx_runtime_v2::BufferId::TopKIds), unsafe {
            std::slice::from_raw_parts(ids.as_ptr() as *const u8, std::mem::size_of_val(&ids))
        })
        .map_err(anyhow::Error::msg)?;
    metal
        .write_runtime_buffer(
            buffers.buffer(qx_runtime_v2::BufferId::TopKWeights),
            unsafe {
                std::slice::from_raw_parts(
                    weights.as_ptr() as *const u8,
                    std::mem::size_of_val(&weights),
                )
            },
        )
        .map_err(anyhow::Error::msg)?;

    let layer = &stack.layers[0];
    let context = buffers.context().as_ref();
    let key_cache = buffers.kv_key_buffer(0).map_err(anyhow::Error::msg)?;
    let value_cache = buffers.kv_value_buffer(0).map_err(anyhow::Error::msg)?;
    let profile_position = context_length.saturating_sub(1);
    let mut profile_entries = Vec::<(String, f64)>::new();
    let mut profile = |label: &str, gpu_us: f64| {
        println!("{label:<24} {gpu_us:>9.2}");
        profile_entries.push((label.to_owned(), gpu_us));
    };
    let warmup = 3.min(iterations);

    println!("QX V2 major-kernel GPU profile (median per dispatch)");
    println!("artifact:       {}", artifact_path.display());
    println!("context:        {context_length}");
    println!("iterations:     {iterations} (warmup {warmup})");
    println!("kernel                    GPU us");
    profile(
        "embedding lookup",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            graph
                .encode_q4_embedding_lookup(
                    &stack.embedding_lookup,
                    buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                    buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
                    qx_model::EMBEDDING_MULTIPLIER,
                )
                .map_err(anyhow::Error::msg)
        })?,
    );
    profile(
        "RMSNorm",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            if qx_metal::platform::rms_scalar_qkv_enabled() {
                graph
                    .encode_rmsnorm_scalar(
                        context,
                        buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
                        &layer.input_norm,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                    )
                    .map_err(anyhow::Error::msg)
            } else {
                graph
                    .encode_rmsnorm(
                        context,
                        buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
                        &layer.input_norm,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    )
                    .map_err(anyhow::Error::msg)
            }
        })?,
    );
    profile(
        "fused QKV GEMV",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            if qx_metal::platform::rms_scalar_qkv_enabled() {
                graph
                    .encode_prepared_q4_qkv_simd_normalized(
                        &layer.qkv,
                        buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
                        &layer.input_norm,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::Query),
                        buffers.buffer(qx_runtime_v2::BufferId::Key),
                        buffers.buffer(qx_runtime_v2::BufferId::Value),
                    )
                    .map_err(anyhow::Error::msg)
            } else {
                graph
                    .encode_prepared_q4_qkv_simd(
                        &layer.qkv,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        buffers.buffer(qx_runtime_v2::BufferId::Query),
                        buffers.buffer(qx_runtime_v2::BufferId::Key),
                        buffers.buffer(qx_runtime_v2::BufferId::Value),
                    )
                    .map_err(anyhow::Error::msg)
            }
        })?,
    );
    profile(
        "RoPE + KV write",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            graph
                .encode_rope_kv(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::Query),
                    buffers.buffer(qx_runtime_v2::BufferId::Key),
                    buffers.buffer(qx_runtime_v2::BufferId::Value),
                    buffers.buffer(qx_runtime_v2::BufferId::Query),
                    &key_cache,
                    &value_cache,
                    profile_position,
                )
                .map_err(anyhow::Error::msg)
        })?,
    );
    profile(
        "decode attention",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            if qx_metal::platform::three_stage_attention_enabled() {
                graph
                    .encode_attention_three_stage(
                        context,
                        buffers.buffer(qx_runtime_v2::BufferId::Query),
                        &key_cache,
                        &value_cache,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::Attention),
                        profile_position + 1,
                    )
                    .map_err(anyhow::Error::msg)
            } else {
                graph
                    .encode_attention(
                        context,
                        buffers.buffer(qx_runtime_v2::BufferId::Query),
                        &key_cache,
                        &value_cache,
                        buffers.buffer(qx_runtime_v2::BufferId::Attention),
                        profile_position + 1,
                    )
                    .map_err(anyhow::Error::msg)
            }
        })?,
    );
    profile(
        "O projection GEMV",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            graph
                .encode_prepared_q4_dense_simd(
                    &layer.o_proj,
                    buffers.buffer(qx_runtime_v2::BufferId::Attention),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                )
                .map_err(anyhow::Error::msg)
        })?,
    );
    profile(
        "attention residual",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            graph
                .encode_residual(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                    qx_model::RESIDUAL_MULTIPLIER,
                )
                .map_err(anyhow::Error::msg)
        })?,
    );
    profile(
        "post-attention RMSNorm",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            if qx_metal::platform::rms_scalar_router_enabled() {
                graph
                    .encode_rmsnorm_scalar(
                        context,
                        buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                        &layer.post_attention_norm,
                        &buffers
                            .buffer(qx_runtime_v2::BufferId::MoeScratch)
                            .subregion(81_916, std::mem::size_of::<f32>())
                            .map_err(anyhow::Error::msg)?,
                    )
                    .map_err(anyhow::Error::msg)
            } else {
                graph
                    .encode_rmsnorm(
                        context,
                        buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                        &layer.post_attention_norm,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    )
                    .map_err(anyhow::Error::msg)
            }
        })?,
    );
    profile(
        "router GEMV",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            if qx_metal::platform::rms_scalar_router_enabled() {
                graph
                    .encode_prepared_q4_dense_simd_normalized(
                        &layer.router,
                        buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                        &layer.post_attention_norm,
                        &buffers
                            .buffer(qx_runtime_v2::BufferId::MoeScratch)
                            .subregion(81_916, std::mem::size_of::<f32>())
                            .map_err(anyhow::Error::msg)?,
                        buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                    )
                    .map_err(anyhow::Error::msg)
            } else {
                graph
                    .encode_prepared_q4_dense_simd(
                        &layer.router,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                    )
                    .map_err(anyhow::Error::msg)
            }
        })?,
    );
    profile(
        "top-k routing",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            graph
                .encode_topk(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                    buffers.buffer(qx_runtime_v2::BufferId::TopKIds),
                    buffers.buffer(qx_runtime_v2::BufferId::TopKWeights),
                )
                .map_err(anyhow::Error::msg)
        })?,
    );
    let split_gate_up = matches!(
        std::env::var("QX_V2_MOE_SPLIT_GATE_UP")
            .ok()
            .map(|value| value.to_ascii_lowercase())
            .as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    );
    let moe_stage_names: Vec<&str> = if split_gate_up {
        vec![
            "MoE gate",
            "MoE up",
            "MoE SiLU",
            "MoE down",
            "MoE weighted accumulation",
        ]
    } else {
        vec![
            "MoE gate/up",
            "MoE SiLU",
            "MoE down",
            "MoE weighted accumulation",
        ]
    };
    let mut moe_total = 0.0;
    let mut moe_stage_micros = vec![0.0_f64; moe_stage_names.len()];
    for (stage, name) in moe_stage_names.iter().enumerate() {
        let micros = profile_graph_stage(&metal, warmup, iterations, |graph| {
            if qx_metal::platform::rms_scalar_router_enabled() {
                graph
                    .encode_dynamic_q4_moe_batched_stage_normalized(
                        &moe.layers[0],
                        buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                        &layer.post_attention_norm,
                        &buffers
                            .buffer(qx_runtime_v2::BufferId::MoeScratch)
                            .subregion(81_916, std::mem::size_of::<f32>())
                            .map_err(anyhow::Error::msg)?,
                        buffers.buffer(qx_runtime_v2::BufferId::TopKIds),
                        buffers.buffer(qx_runtime_v2::BufferId::TopKWeights),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                        stage as u8,
                    )
                    .map_err(anyhow::Error::msg)
            } else {
                graph
                    .encode_dynamic_q4_moe_batched_stage(
                        &moe.layers[0],
                        buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        buffers.buffer(qx_runtime_v2::BufferId::TopKIds),
                        buffers.buffer(qx_runtime_v2::BufferId::TopKWeights),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                        stage as u8,
                    )
                    .map_err(anyhow::Error::msg)
            }
        })?;
        moe_total += micros;
        moe_stage_micros[stage] = micros;
        profile(name, micros);
    }
    profile(
        if split_gate_up {
            "dynamic Q4 MoE (5-stage total)"
        } else {
            "dynamic Q4 MoE (4-stage total)"
        },
        moe_total,
    );
    // Granite-specific byte model for the selected top-8 path.  This counts
    // Q4 payload, FP16 scales, and stage-local activation I/O; it does not
    // pretend that all 32 resident experts are read for one token.
    let moe_stage_bytes = if split_gate_up {
        vec![
            8 * (512 * 1024 / 2 + 512 * 32 * 2) + 1024 * 4 + 8 * 512 * 4,
            8 * (512 * 1024 / 2 + 512 * 32 * 2) + 1024 * 4 + 8 * 512 * 4,
            8 * 2 * 512 * 4 + 8 * 512 * 4,
            8 * (1024 * 512 / 2 + 1024 * 16 * 2) + 8 * 512 * 4 + 8 * 1024 * 4,
            8 * 1024 * 4 + 8 * 4 + 1024 * 4,
        ]
    } else {
        vec![
            8 * (1024 * 1024 / 2 + 1024 * 32 * 2) + 1024 * 4 + 8 * 1024 * 4,
            8 * 2 * 512 * 4 + 8 * 512 * 4,
            8 * (1024 * 512 / 2 + 1024 * 16 * 2) + 8 * 512 * 4 + 8 * 1024 * 4,
            8 * 1024 * 4 + 8 * 4 + 1024 * 4,
        ]
    };
    println!();
    println!("MoE selected-top8 traffic model (weights + scales + stage I/O)");
    println!("stage                         bytes       GPU us   effective GB/s");
    for (index, name) in moe_stage_names.iter().enumerate() {
        let bytes = moe_stage_bytes[index];
        let micros = moe_stage_micros[index];
        let bandwidth = bytes as f64 / micros.max(f64::MIN_POSITIVE) / 1_000.0;
        println!("{name:<29} {bytes:>10} {micros:>10.2} {bandwidth:>15.2}");
    }
    profile(
        "MoE residual",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            graph
                .encode_residual(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                    buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                    qx_model::RESIDUAL_MULTIPLIER,
                )
                .map_err(anyhow::Error::msg)
        })?,
    );
    profile(
        "final RMSNorm",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            graph
                .encode_rmsnorm(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
                    &stack.final_norm,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                )
                .map_err(anyhow::Error::msg)
        })?,
    );
    profile(
        "Q4 LM head",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            graph
                .encode_prepared_q4_dense_simd_scaled(
                    &stack.embedding,
                    buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                    buffers.buffer(qx_runtime_v2::BufferId::Logits),
                    1.0 / qx_model::LOGITS_SCALING,
                )
                .map_err(anyhow::Error::msg)
        })?,
    );
    profile(
        "GPU argmax",
        profile_graph_stage(&metal, warmup, iterations, |graph| {
            graph
                .encode_argmax(
                    context,
                    buffers.buffer(qx_runtime_v2::BufferId::Logits),
                    buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                    buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                    VOCAB,
                )
                .map_err(anyhow::Error::msg)
        })?,
    );
    let mut contribution = profile_entries
        .iter()
        .filter(|(label, _)| !label.starts_with("dynamic Q4 MoE ("))
        .map(|(label, gpu_us)| {
            let once = matches!(
                label.as_str(),
                "embedding lookup" | "final RMSNorm" | "Q4 LM head" | "GPU argmax"
            );
            (label.clone(), *gpu_us, if once { 1.0 } else { 24.0 })
        })
        .map(|(label, gpu_us, multiplicity)| (label, gpu_us * multiplicity / 1000.0))
        .collect::<Vec<_>>();
    contribution.sort_by(|left, right| right.1.total_cmp(&left.1));
    let isolated_total_ms = contribution.iter().map(|(_, total)| total).sum::<f64>();
    println!();
    println!("Estimated per-token contribution from isolated GPU medians");
    println!("component                         ms/token   share of isolated sum");
    for (label, total_ms) in &contribution {
        println!(
            "{label:<32} {total_ms:>9.3}   {:>8.1}%",
            100.0 * total_ms / isolated_total_ms.max(f64::MIN_POSITIVE)
        );
    }
    println!("isolated-dispatch estimate total: {isolated_total_ms:.3} ms/token");
    println!(
        "Note: this estimate intentionally excludes command overlap and graph scheduling; compare it with the measured --v2-token GPU ms/token."
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_dispatch_floor(iterations: usize) -> Result<()> {
    use qx_metal::platform::MTLSize;
    use std::rc::Rc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let counts = [1_usize, 24, 50, 100, 200, 340];
    let threads = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    println!("QX V2 Metal dispatch-floor benchmark");
    println!("iterations: {iterations} (warmup 3)");
    println!("dispatches  GPU median us  wall median us  GPU us/dispatch  wall us/dispatch");
    for count in counts {
        for _ in 0..3 {
            let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
            for _ in 0..count {
                graph
                    .encode_noop(context.as_ref(), threads, threads)
                    .map_err(anyhow::Error::msg)?;
            }
            graph.finish(&context).map_err(anyhow::Error::msg)?;
        }
        let mut gpu_samples = Vec::with_capacity(iterations);
        let mut wall_samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let before = context.metrics().map_err(anyhow::Error::msg)?;
            let start = Instant::now();
            let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
            for _ in 0..count {
                graph
                    .encode_noop(context.as_ref(), threads, threads)
                    .map_err(anyhow::Error::msg)?;
            }
            graph.finish(&context).map_err(anyhow::Error::msg)?;
            let after = context.metrics().map_err(anyhow::Error::msg)?;
            gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6);
            wall_samples.push(start.elapsed().as_secs_f64() * 1e6);
        }
        let gpu = median_sample(&mut gpu_samples);
        let wall = median_sample(&mut wall_samples);
        println!(
            "{count:9}  {gpu:14.2}  {wall:14.2}  {gpu_per:16.3}  {wall_per:16.3}",
            gpu_per = gpu / count as f64,
            wall_per = wall / count as f64
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_attention_sweep(artifact_path: &std::path::Path, iterations: usize) -> Result<()> {
    use std::rc::Rc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let lengths = [32_usize, 64, 128, 256, 512];
    println!("QX V2 decode-attention context sweep");
    println!("artifact: {}", artifact_path.display());
    println!("iterations: {iterations} (warmup 3)");
    println!("context  GPU median us  wall median us  GPU ms/token");
    for length in lengths {
        let plan = ExecutionPlan::granite(length).map_err(anyhow::Error::msg)?;
        let buffers = qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan)
            .map_err(anyhow::Error::msg)?;
        let stack = qx_runtime_v2::PreparedAttentionStack::load_from_artifact(
            context.as_ref(),
            artifact_path,
        )
        .map_err(anyhow::Error::msg)?;
        let key_cache = buffers.kv_key_buffer(0).map_err(anyhow::Error::msg)?;
        let value_cache = buffers.kv_value_buffer(0).map_err(anyhow::Error::msg)?;
        let decode_context = buffers.context().as_ref();
        for _ in 0..3 {
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            if qx_metal::platform::three_stage_attention_enabled() {
                graph
                    .encode_attention_three_stage(
                        decode_context,
                        buffers.buffer(qx_runtime_v2::BufferId::Query),
                        &key_cache,
                        &value_cache,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::Attention),
                        length,
                    )
                    .map_err(anyhow::Error::msg)?;
            } else {
                graph
                    .encode_attention(
                        decode_context,
                        buffers.buffer(qx_runtime_v2::BufferId::Query),
                        &key_cache,
                        &value_cache,
                        buffers.buffer(qx_runtime_v2::BufferId::Attention),
                        length,
                    )
                    .map_err(anyhow::Error::msg)?;
            }
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        }
        let mut gpu_samples = Vec::with_capacity(iterations);
        let mut wall_samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let before = context.metrics().map_err(anyhow::Error::msg)?;
            let start = Instant::now();
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            if qx_metal::platform::three_stage_attention_enabled() {
                graph
                    .encode_attention_three_stage(
                        decode_context,
                        buffers.buffer(qx_runtime_v2::BufferId::Query),
                        &key_cache,
                        &value_cache,
                        buffers.buffer(qx_runtime_v2::BufferId::MoeScratch),
                        buffers.buffer(qx_runtime_v2::BufferId::Attention),
                        length,
                    )
                    .map_err(anyhow::Error::msg)?;
            } else {
                graph
                    .encode_attention(
                        decode_context,
                        buffers.buffer(qx_runtime_v2::BufferId::Query),
                        &key_cache,
                        &value_cache,
                        buffers.buffer(qx_runtime_v2::BufferId::Attention),
                        length,
                    )
                    .map_err(anyhow::Error::msg)?;
            }
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
            let after = context.metrics().map_err(anyhow::Error::msg)?;
            gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6);
            wall_samples.push(start.elapsed().as_secs_f64() * 1e6);
        }
        let gpu = median_sample(&mut gpu_samples);
        let wall = median_sample(&mut wall_samples);
        println!("{length:7}  {gpu:14.2}  {wall:14.2}  {:14.3}", gpu / 1000.0);
        // Keep the prepared stack alive until all samples have completed; its
        // pipelines are also representative of the real attention path.
        let _ = stack.layers.len();
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_layer_block(
    artifact_path: &std::path::Path,
    context_length: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(context.as_ref(), artifact_path)
            .map_err(anyhow::Error::msg)?;
    let moe = qx_runtime_v2::PreparedMoeStack::load(context.as_ref(), artifact_path)
        .map_err(anyhow::Error::msg)?;
    let blocks = [1_usize, 2, 4, 8, 24];
    println!("QX V2 layer-block topology benchmark");
    println!("context: {context_length}, iterations: {iterations} (warmup 3)");
    println!("layers  dispatches  GPU median ms  wall median ms  GPU ms/layer");
    for block in blocks {
        for _ in 0..3 {
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            stack
                .encode_full_layers(&mut graph, &buffers, &moe, 0, block)
                .map_err(anyhow::Error::msg)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        }
        let mut gpu_samples = Vec::with_capacity(iterations);
        let mut wall_samples = Vec::with_capacity(iterations);
        let before_all = context.metrics().map_err(anyhow::Error::msg)?;
        for _ in 0..iterations {
            let before = context.metrics().map_err(anyhow::Error::msg)?;
            let start = Instant::now();
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            stack
                .encode_full_layers(&mut graph, &buffers, &moe, 0, block)
                .map_err(anyhow::Error::msg)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
            let after = context.metrics().map_err(anyhow::Error::msg)?;
            gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1000.0);
            wall_samples.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        let gpu = median_sample(&mut gpu_samples);
        let wall = median_sample(&mut wall_samples);
        let after_all = context.metrics().map_err(anyhow::Error::msg)?;
        let dispatches = (after_all.dispatches - before_all.dispatches) / iterations as u64;
        println!(
            "{block:6}  {dispatches:10}  {gpu:14.3}  {wall:14.3}  {:14.3}",
            gpu / block as f64
        );
    }

    let decode_context = buffers.context().as_ref();
    let key_cache = buffers.kv_key_buffer(0).map_err(anyhow::Error::msg)?;
    let value_cache = buffers.kv_value_buffer(0).map_err(anyhow::Error::msg)?;
    let layer = &stack.layers[0];
    let bundles = [
        ("A current one-layer graph", 14_usize),
        ("B norm + QKV + RoPE/KV", 3),
        ("C attention + O + residual", 3),
        ("D post-norm + router + top-k", 3),
    ];
    println!();
    println!("one-layer controlled bundles (same command-buffer discipline)");
    println!("bundle                              dispatches  GPU median us  wall median us");
    for (bundle, expected_dispatches) in bundles {
        let encode_bundle = |graph: &mut qx_metal::platform::DecodeGraph| -> Result<()> {
            match bundle.chars().next().unwrap_or('A') {
                'A' => stack
                    .encode_full_layers(graph, &buffers, &moe, 0, 1)
                    .map_err(anyhow::Error::msg)?,
                'B' => {
                    graph
                        .encode_rmsnorm(
                            decode_context,
                            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
                            &layer.input_norm,
                            buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        )
                        .map_err(anyhow::Error::msg)?;
                    graph
                        .encode_prepared_q4_qkv_simd(
                            &layer.qkv,
                            buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                            buffers.buffer(qx_runtime_v2::BufferId::Query),
                            buffers.buffer(qx_runtime_v2::BufferId::Key),
                            buffers.buffer(qx_runtime_v2::BufferId::Value),
                        )
                        .map_err(anyhow::Error::msg)?;
                    graph
                        .encode_rope_kv(
                            decode_context,
                            buffers.buffer(qx_runtime_v2::BufferId::Query),
                            buffers.buffer(qx_runtime_v2::BufferId::Key),
                            buffers.buffer(qx_runtime_v2::BufferId::Value),
                            buffers.buffer(qx_runtime_v2::BufferId::Query),
                            &key_cache,
                            &value_cache,
                            0,
                        )
                        .map_err(anyhow::Error::msg)?;
                }
                'C' => {
                    graph
                        .encode_attention(
                            decode_context,
                            buffers.buffer(qx_runtime_v2::BufferId::Query),
                            &key_cache,
                            &value_cache,
                            buffers.buffer(qx_runtime_v2::BufferId::Attention),
                            1,
                        )
                        .map_err(anyhow::Error::msg)?;
                    graph
                        .encode_prepared_q4_dense_simd(
                            &layer.o_proj,
                            buffers.buffer(qx_runtime_v2::BufferId::Attention),
                            buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                        )
                        .map_err(anyhow::Error::msg)?;
                    graph
                        .encode_residual(
                            decode_context,
                            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
                            buffers.buffer(qx_runtime_v2::BufferId::MoeOutput),
                            buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                            qx_model::RESIDUAL_MULTIPLIER,
                        )
                        .map_err(anyhow::Error::msg)?;
                }
                'D' => {
                    graph
                        .encode_rmsnorm(
                            decode_context,
                            buffers.buffer(qx_runtime_v2::BufferId::HiddenB),
                            &layer.post_attention_norm,
                            buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                        )
                        .map_err(anyhow::Error::msg)?;
                    graph
                        .encode_prepared_q4_dense_simd(
                            &layer.router,
                            buffers.buffer(qx_runtime_v2::BufferId::MoeInput),
                            buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                        )
                        .map_err(anyhow::Error::msg)?;
                    graph
                        .encode_topk(
                            decode_context,
                            buffers.buffer(qx_runtime_v2::BufferId::RouterScores),
                            buffers.buffer(qx_runtime_v2::BufferId::TopKIds),
                            buffers.buffer(qx_runtime_v2::BufferId::TopKWeights),
                        )
                        .map_err(anyhow::Error::msg)?;
                }
                _ => unreachable!(),
            }
            Ok(())
        };
        for _ in 0..3 {
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            encode_bundle(&mut graph)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        }
        let mut gpu_samples = Vec::with_capacity(iterations);
        let mut wall_samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let before = context.metrics().map_err(anyhow::Error::msg)?;
            let start = Instant::now();
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            encode_bundle(&mut graph)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
            let after = context.metrics().map_err(anyhow::Error::msg)?;
            gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6);
            wall_samples.push(start.elapsed().as_secs_f64() * 1e6);
        }
        let gpu = median_sample(&mut gpu_samples);
        let wall = median_sample(&mut wall_samples);
        println!("{bundle:<34} {expected_dispatches:10}  {gpu:14.2}  {wall:14.2}");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_attention_stages(
    _artifact_path: &std::path::Path,
    context_length: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;

    let context = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(context.clone(), plan).map_err(anyhow::Error::msg)?;
    let decode_context = buffers.context().as_ref();
    let key_cache = buffers.kv_key_buffer(0).map_err(anyhow::Error::msg)?;
    let value_cache = buffers.kv_value_buffer(0).map_err(anyhow::Error::msg)?;
    let query = buffers.buffer(qx_runtime_v2::BufferId::Query);
    let scores = buffers.buffer(qx_runtime_v2::BufferId::MoeScratch);
    let value_output = buffers.buffer(qx_runtime_v2::BufferId::Attention);

    let stages = [
        ("QK dot products", 0_u8),
        ("max/sum/softmax", 1),
        ("V accumulation", 2),
        ("QK + softmax + V", 3),
        ("live attention", 4),
    ];
    println!("QX V2 attention-stage decomposition");
    println!("context: {context_length}, iterations: {iterations} (warmup 3)");
    println!(
        "stage                         GPU median us  wall median us  ns/context  effective GB/s"
    );
    for (name, stage) in stages {
        let encode = |graph: &mut qx_metal::platform::DecodeGraph| -> Result<()> {
            if stage == 0 || stage == 3 {
                graph
                    .encode_attention_qk(decode_context, query, &key_cache, scores, context_length)
                    .map_err(anyhow::Error::msg)?;
            }
            if stage == 1 || stage == 3 {
                graph
                    .encode_attention_softmax(decode_context, scores, context_length)
                    .map_err(anyhow::Error::msg)?;
            }
            if stage == 2 || stage == 3 {
                graph
                    .encode_attention_v(
                        decode_context,
                        scores,
                        &value_cache,
                        value_output,
                        context_length,
                    )
                    .map_err(anyhow::Error::msg)?;
            }
            if stage == 4 {
                if qx_metal::platform::three_stage_attention_enabled() {
                    graph
                        .encode_attention_three_stage(
                            decode_context,
                            query,
                            &key_cache,
                            &value_cache,
                            scores,
                            value_output,
                            context_length,
                        )
                        .map_err(anyhow::Error::msg)?;
                } else {
                    graph
                        .encode_attention(
                            decode_context,
                            query,
                            &key_cache,
                            &value_cache,
                            value_output,
                            context_length,
                        )
                        .map_err(anyhow::Error::msg)?;
                }
            }
            Ok(())
        };
        for _ in 0..3 {
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            encode(&mut graph)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        }
        let mut gpu_samples = Vec::with_capacity(iterations);
        let mut wall_samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let before = context.metrics().map_err(anyhow::Error::msg)?;
            let start = Instant::now();
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            encode(&mut graph)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
            let after = context.metrics().map_err(anyhow::Error::msg)?;
            gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6);
            wall_samples.push(start.elapsed().as_secs_f64() * 1e6);
        }
        let gpu = median_sample(&mut gpu_samples);
        let wall = median_sample(&mut wall_samples);
        let logical_bytes_per_context = match stage {
            0 => 16 * 64 * std::mem::size_of::<f32>() * 2,
            1 => 16 * std::mem::size_of::<f32>() * 2,
            2 => 16 * 64 * std::mem::size_of::<f32>(),
            3 => (16 * 64 * std::mem::size_of::<f32>() * 3) + 16 * 8,
            _ => 0,
        } as f64;
        let ns_per_context = gpu * 1000.0 / context_length as f64;
        let effective_gbps = if logical_bytes_per_context == 0.0 {
            0.0
        } else {
            logical_bytes_per_context * context_length as f64 / (gpu / 1e6) / 1e9
        };
        println!(
            "{name:<28} {gpu:14.2}  {wall:14.2}  {ns_per_context:10.2}  {effective_gbps:15.2}"
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn profile_graph_stage<F>(
    context: &qx_metal::platform::MetalContext,
    warmup: usize,
    iterations: usize,
    mut encode: F,
) -> Result<f64>
where
    F: FnMut(&mut qx_metal::platform::DecodeGraph) -> Result<()>,
{
    for _ in 0..warmup {
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        encode(&mut graph)?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        encode(&mut graph)?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6);
    }
    Ok(median_sample(&mut samples))
}

#[cfg(target_os = "macos")]
fn profile_graph_stage_detailed<F>(
    context: &qx_metal::platform::MetalContext,
    warmup: usize,
    iterations: usize,
    mut encode: F,
) -> Result<(f64, u64)>
where
    F: FnMut(&mut qx_metal::platform::DecodeGraph) -> Result<()>,
{
    for _ in 0..warmup {
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        encode(&mut graph)?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
    }
    let mut gpu_samples = Vec::with_capacity(iterations);
    let mut dispatch_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let before = context.metrics().map_err(anyhow::Error::msg)?;
        let mut graph = context.begin_decode().map_err(anyhow::Error::msg)?;
        encode(&mut graph)?;
        graph.finish(context).map_err(anyhow::Error::msg)?;
        let after = context.metrics().map_err(anyhow::Error::msg)?;
        gpu_samples.push((after.gpu_time_seconds - before.gpu_time_seconds).max(0.0) * 1e6);
        dispatch_samples.push(after.dispatches - before.dispatches);
    }
    gpu_samples.sort_by(f64::total_cmp);
    dispatch_samples.sort_unstable();
    Ok((
        gpu_samples[gpu_samples.len() / 2],
        dispatch_samples[dispatch_samples.len() / 2],
    ))
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_kernel_profile(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-kernel-profile requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_kernel(_artifact_path: &std::path::Path, _iterations: usize) -> Result<()> {
    bail!("--v2-kernel-bench requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_kernel_sweep(_artifact_path: &std::path::Path, _iterations: usize) -> Result<()> {
    bail!("--v2-kernel-sweep requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_qkv_sweep(_artifact_path: &std::path::Path, _iterations: usize) -> Result<()> {
    bail!("--v2-qkv-sweep requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_qkv_boundary(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-qkv-boundary requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_qkv_interference(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-qkv-interference requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_dispatch_floor(_iterations: usize) -> Result<()> {
    bail!("--v2-dispatch-floor requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_attention_sweep(
    _artifact_path: &std::path::Path,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-attention-sweep requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_layer_block(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-layer-block requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_attention_stages(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-attention-stages requires macOS Metal")
}

#[cfg(target_os = "macos")]
fn benchmark_v2_attention_stack(
    model_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;
    let metal = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(metal.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack = qx_runtime_v2::PreparedAttentionStack::load(&metal, model_path)
        .map_err(anyhow::Error::msg)?;
    let hidden = vec![1.0_f32; qx_model::HIDDEN];
    let hidden_bytes = unsafe {
        std::slice::from_raw_parts(
            hidden.as_ptr() as *const u8,
            std::mem::size_of_val(hidden.as_slice()),
        )
    };
    metal
        .write_runtime_buffer(
            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
            hidden_bytes,
        )
        .map_err(anyhow::Error::msg)?;
    for _ in 0..warmup {
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode(&mut graph, &buffers, 0)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    let before = metal.metrics().map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    for _ in 0..iterations {
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode(&mut graph, &buffers, 0)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    let wall = start.elapsed();
    let after = metal.metrics().map_err(anyhow::Error::msg)?;
    let gpu_seconds = after.gpu_time_seconds - before.gpu_time_seconds;
    let count = iterations.max(1) as f64;
    println!("QX V2 attention-stack benchmark (not full generation)");
    println!("layers:                 {}", stack.layers.len());
    println!("iterations:             {iterations}");
    println!(
        "wall ms/token:          {:.3}",
        1000.0 * wall.as_secs_f64() / count
    );
    println!(
        "GPU ms/token:           {:.3}",
        1000.0 * gpu_seconds / count
    );
    println!(
        "command buffers/token:  {:.2}",
        (after.command_buffers - before.command_buffers) as f64 / count
    );
    println!(
        "encoders/token:         {:.2}",
        (after.encoders - before.encoders) as f64 / count
    );
    println!(
        "dispatches/token:       {:.2}",
        (after.dispatches - before.dispatches) as f64 / count
    );
    println!(
        "equivalent attention-only tok/s: {:.2}",
        count / wall.as_secs_f64()
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_attention_stack(
    _model_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-attention-stack requires macOS Metal")
}

#[cfg(target_os = "macos")]
fn benchmark_v2_full_stack(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
) -> Result<()> {
    use std::rc::Rc;
    let metal = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(metal.clone(), plan).map_err(anyhow::Error::msg)?;
    let attention =
        qx_runtime_v2::PreparedAttentionStack::load_from_artifact(&metal, artifact_path)
            .map_err(anyhow::Error::msg)?;
    let moe =
        qx_runtime_v2::PreparedMoeStack::load(&metal, artifact_path).map_err(anyhow::Error::msg)?;
    let hidden = vec![1.0_f32; qx_model::HIDDEN];
    let hidden_bytes = unsafe {
        std::slice::from_raw_parts(
            hidden.as_ptr() as *const u8,
            std::mem::size_of_val(hidden.as_slice()),
        )
    };
    metal
        .write_runtime_buffer(
            buffers.buffer(qx_runtime_v2::BufferId::HiddenA),
            hidden_bytes,
        )
        .map_err(anyhow::Error::msg)?;
    for _ in 0..warmup {
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        attention
            .encode_full(&mut graph, &buffers, &moe, 0)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    let before = metal.metrics().map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    for _ in 0..iterations {
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        attention
            .encode_full(&mut graph, &buffers, &moe, 0)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    let wall = start.elapsed();
    let after = metal.metrics().map_err(anyhow::Error::msg)?;
    let count = iterations.max(1) as f64;
    let gpu_seconds = after.gpu_time_seconds - before.gpu_time_seconds;
    println!("QX V2 attention+MoE benchmark (no embedding/logits)");
    println!("layers:                 {}", attention.layers.len());
    println!("iterations:             {iterations}");
    println!(
        "wall ms/token:          {:.3}",
        1000.0 * wall.as_secs_f64() / count
    );
    println!(
        "GPU ms/token:           {:.3}",
        1000.0 * gpu_seconds / count
    );
    println!(
        "command buffers/token:  {:.2}",
        (after.command_buffers - before.command_buffers) as f64 / count
    );
    println!(
        "encoders/token:         {:.2}",
        (after.encoders - before.encoders) as f64 / count
    );
    println!(
        "dispatches/token:       {:.2}",
        (after.dispatches - before.dispatches) as f64 / count
    );
    println!("equivalent core tok/s:  {:.2}", count / wall.as_secs_f64());
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_full_stack(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
) -> Result<()> {
    bail!("--v2-full-stack requires macOS Metal")
}

#[cfg(target_os = "macos")]
#[derive(Serialize)]
struct V2TokenBenchmarkRecord {
    schema_version: u32,
    implementation: &'static str,
    qx_revision: &'static str,
    artifact: String,
    artifact_bytes: u64,
    artifact_hash: String,
    artifact_version: u32,
    plan_id: String,
    target_id: String,
    fixture_name: String,
    fixture_domain: String,
    fixture_token_hash: String,
    measurement: MeasurementMode,
    context: usize,
    prefill_tokens: usize,
    measured_position_start: usize,
    measured_tokens: usize,
    warmup: usize,
    wall_ms_per_token: f64,
    gpu_ms_per_token: f64,
    tokens_per_second: f64,
    command_buffers_per_token: f64,
    encoders_per_token: f64,
    dispatches_per_token: f64,
    internal_cpu_waits: u64,
    current_rss_bytes: u64,
    peak_rss_bytes: u64,
    current_physical_footprint_bytes: u64,
    peak_physical_footprint_bytes: u64,
    mapped_artifact_bytes: u64,
    known_metal_allocated_bytes: u64,
    recommended_metal_working_set_bytes: u64,
    os_version: String,
    power_state: String,
    thermal_state: String,
    memory_pressure: String,
    last_token_id: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Serialize)]
struct MemorySnapshot {
    label: String,
    current_rss_bytes: u64,
    peak_rss_bytes: u64,
    current_physical_footprint_bytes: u64,
    peak_physical_footprint_bytes: u64,
    known_metal_allocated_bytes: u64,
    recommended_metal_working_set_bytes: u64,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Serialize)]
struct MemoryOwnership {
    artifact_file_bytes: u64,
    mapped_artifact_bytes: u64,
    arena_bytes: u64,
    arena_region_bytes: u64,
    arena_alignment_bytes: u64,
    activation_and_small_buffers_bytes: u64,
    attention_scratch_bytes: u64,
    moe_scratch_bytes: u64,
    logits_bytes: u64,
    kv_cache_bytes: u64,
    pipeline_state_bytes: u64,
    pipeline_state_bytes_known: bool,
    duplicate_cpu_weight_bytes: u64,
    duplicate_cpu_weight_bytes_known: bool,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Serialize)]
struct V2MemoryProfileReport {
    schema_version: u32,
    implementation: &'static str,
    qx_revision: &'static str,
    artifact: String,
    artifact_hash: String,
    artifact_version: u32,
    plan_id: String,
    target_id: String,
    context: usize,
    warmup: usize,
    iterations: usize,
    ownership: MemoryOwnership,
    phases: Vec<MemorySnapshot>,
    os_version: String,
    power_state: String,
    thermal_state: String,
    memory_pressure: String,
    limitations: Vec<&'static str>,
}

#[cfg(target_os = "macos")]
fn benchmark_v2_memory_profile(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
    result_json: Option<&std::path::Path>,
) -> Result<()> {
    use std::rc::Rc;
    use std::sync::Arc;

    let sealed_plan = qx_runtime_v2::activate_artifact_execution_plan(artifact_path)
        .map_err(anyhow::Error::msg)?;
    if let Some(plan) = &sealed_plan {
        if plan.workload.context_limit as usize != context_length {
            bail!(
                "context {context_length} does not match sealed plan context {}",
                plan.workload.context_limit
            );
        }
    }
    let (artifact_bytes, artifact_hash) = artifact_fingerprint(artifact_path)?;
    let mapped_artifact = Arc::new(MappedArtifact::open(artifact_path)?);
    let target_id = mapped_artifact.info().header.target_id.clone();
    let plan_id = sealed_plan
        .as_ref()
        .map(|plan| plan.plan_id.clone())
        .unwrap_or_else(|| "legacy-v1-unsealed".to_owned());
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let metal = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let mut phases = vec![memory_snapshot("baseline", &metal)];
    phases.push(memory_snapshot("artifact_mapped", &metal));

    let buffers = qx_runtime_v2::GpuRuntimeBuffers::new(metal.clone(), plan.clone())
        .map_err(anyhow::Error::msg)?;
    phases.push(memory_snapshot("runtime_arena_and_kv", &metal));
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_mapped_artifact(
        &metal,
        mapped_artifact.clone(),
    )
    .map_err(anyhow::Error::msg)?;
    phases.push(memory_snapshot("attention_weights_and_pipelines", &metal));
    let moe = qx_runtime_v2::PreparedMoeStack::load_from_mapped_artifact(&metal, mapped_artifact)
        .map_err(anyhow::Error::msg)?;
    phases.push(memory_snapshot("all_weights_and_pipelines", &metal));

    let token_bytes = 0_u32.to_le_bytes();
    for _ in 0..warmup {
        metal
            .write_runtime_buffer(
                buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                &token_bytes,
            )
            .map_err(anyhow::Error::msg)?;
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, 0)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    phases.push(memory_snapshot("steady_state_after_warmup", &metal));
    for position in 0..iterations {
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, position % context_length)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    phases.push(memory_snapshot("steady_state_after_measurement", &metal));

    let ownership = memory_ownership(&plan, artifact_bytes);
    let final_metal_allocated_bytes = phases
        .last()
        .map(|phase| phase.known_metal_allocated_bytes)
        .unwrap_or(0);
    let report = V2MemoryProfileReport {
        schema_version: 1,
        implementation: "qx-v2-memory-profile",
        qx_revision: option_env!("QX_GIT_REVISION").unwrap_or("unknown"),
        artifact: artifact_path.display().to_string(),
        artifact_hash: format!("{artifact_hash:016x}"),
        artifact_version: mapped_artifact_version(artifact_path)?,
        plan_id,
        target_id,
        context: context_length,
        warmup,
        iterations,
        ownership,
        phases,
        os_version: command_output("sw_vers", &["-productVersion"]),
        power_state: command_output("pmset", &["-g", "batt"]),
        thermal_state: command_output("pmset", &["-g", "therm"]),
        memory_pressure: command_output("memory_pressure", &["-Q"]),
        limitations: vec![
            "Metal currentAllocatedSize is aggregate; Metal does not expose per-buffer ownership through the current qx-metal API.",
            "mapped_artifact_bytes is the file-backed mapping extent, not resident mapped pages.",
            "pipeline_state_bytes and duplicate_cpu_weight_bytes are unknown because their allocations are not exposed by the current runtime API.",
            "peak RSS/physical footprint are process lifetime maxima and cannot be reset between phases.",
        ],
    };
    println!("QX V2 memory ownership profile");
    println!("artifact:                 {}", report.artifact);
    println!(
        "artifact bytes/mapped:    {} / {}",
        artifact_bytes, ownership.mapped_artifact_bytes
    );
    println!("arena bytes:              {}", ownership.arena_bytes);
    println!("KV cache bytes:           {}", ownership.kv_cache_bytes);
    println!(
        "attention scratch bytes:  {}",
        ownership.attention_scratch_bytes
    );
    println!("MoE scratch bytes:        {}", ownership.moe_scratch_bytes);
    println!("known Metal allocations:  {final_metal_allocated_bytes}");
    for phase in &report.phases {
        println!(
            "phase {:<34} rss {:>12} physical {:>12} metal {:>12}",
            phase.label,
            phase.current_rss_bytes,
            phase.current_physical_footprint_bytes,
            phase.known_metal_allocated_bytes
        );
    }
    if let Some(path) = result_json {
        std::fs::write(path, serde_json::to_vec_pretty(&report)?)?;
        println!("memory report JSON:       {}", path.display());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn memory_snapshot(label: &str, metal: &qx_metal::platform::MetalContext) -> MemorySnapshot {
    let (current_physical_footprint_bytes, peak_physical_footprint_bytes) = physical_footprint();
    MemorySnapshot {
        label: label.to_owned(),
        current_rss_bytes: current_rss_bytes(),
        peak_rss_bytes: peak_rss_bytes(),
        current_physical_footprint_bytes,
        peak_physical_footprint_bytes,
        known_metal_allocated_bytes: metal.current_allocated_bytes(),
        recommended_metal_working_set_bytes: metal.recommended_working_set_bytes(),
    }
}

#[cfg(target_os = "macos")]
fn memory_ownership(plan: &ExecutionPlan, artifact_bytes: u64) -> MemoryOwnership {
    let region_bytes = |id: BufferId| {
        plan.regions
            .iter()
            .find(|region| region.id == id)
            .map(|region| region.length as u64)
            .unwrap_or(0)
    };
    let arena_region_bytes = plan.regions.iter().map(|region| region.length as u64).sum();
    let activation_and_small_buffers_bytes = [
        BufferId::HiddenA,
        BufferId::HiddenB,
        BufferId::Query,
        BufferId::Key,
        BufferId::Value,
        BufferId::Attention,
        BufferId::MoeInput,
        BufferId::RouterScores,
        BufferId::TopKIds,
        BufferId::TopKWeights,
        BufferId::MoeOutput,
        BufferId::TokenOutput,
    ]
    .into_iter()
    .map(region_bytes)
    .sum();
    let attention_scratch_bytes = region_bytes(BufferId::AttentionScratch);
    let moe_scratch_bytes = region_bytes(BufferId::MoeScratch);
    let logits_bytes = region_bytes(BufferId::Logits);
    let kv_cache_bytes = (0..LAYERS)
        .map(|layer| {
            let key_bytes = if qx_metal::platform::kv_key_fp16_for_layer(layer) {
                std::mem::size_of::<u16>()
            } else {
                std::mem::size_of::<f32>()
            };
            let value_bytes = if qx_metal::platform::kv_value_fp16_for_layer(layer) {
                std::mem::size_of::<u16>()
            } else {
                std::mem::size_of::<f32>()
            };
            context_kv_bytes(plan.context, key_bytes + value_bytes)
        })
        .sum();
    MemoryOwnership {
        artifact_file_bytes: artifact_bytes,
        mapped_artifact_bytes: artifact_bytes,
        arena_bytes: plan.arena_bytes as u64,
        arena_region_bytes,
        arena_alignment_bytes: plan.arena_bytes as u64 - arena_region_bytes,
        activation_and_small_buffers_bytes,
        attention_scratch_bytes,
        moe_scratch_bytes,
        logits_bytes,
        kv_cache_bytes,
        pipeline_state_bytes: 0,
        pipeline_state_bytes_known: false,
        duplicate_cpu_weight_bytes: 0,
        duplicate_cpu_weight_bytes_known: false,
    }
}

#[cfg(target_os = "macos")]
fn context_kv_bytes(context: usize, element_bytes: usize) -> u64 {
    (context * KV_HEADS * HEAD_DIM * element_bytes) as u64
}

#[cfg(target_os = "macos")]
fn mapped_artifact_version(path: &std::path::Path) -> Result<u32> {
    Ok(MappedArtifact::open(path)?.info().header.version)
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_memory_profile(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
    _result_json: Option<&std::path::Path>,
) -> Result<()> {
    bail!("--v2-memory-profile requires macOS Metal")
}

#[cfg(target_os = "macos")]
fn benchmark_v2_token(
    artifact_path: &std::path::Path,
    context_length: usize,
    warmup: usize,
    iterations: usize,
    fixture: Option<&BenchmarkFixture>,
    result_json: Option<&std::path::Path>,
) -> Result<()> {
    use std::rc::Rc;
    use std::sync::Arc;
    let default_tokens = [0_u32];
    let tokens = fixture
        .map(|fixture| fixture.tokens.as_slice())
        .unwrap_or(default_tokens.as_slice());
    let measurement = fixture
        .map(|fixture| fixture.measurement)
        .unwrap_or_default();
    if measurement == MeasurementMode::SeedInclusive && tokens.len() != 1 {
        bail!("seed_inclusive fixtures require exactly one seed token");
    }
    let prefill_tokens = if measurement == MeasurementMode::PostPrefill {
        tokens.len()
    } else {
        0
    };
    if prefill_tokens.saturating_add(iterations) > context_length {
        bail!("prefill plus measured positions must fit within the V2 context");
    }
    let initial_token = tokens[0];
    let sealed_plan = qx_runtime_v2::activate_artifact_execution_plan(artifact_path)
        .map_err(anyhow::Error::msg)?;
    if let Some(plan) = &sealed_plan {
        if plan.workload.context_limit as usize != context_length {
            bail!(
                "fixture context {context_length} does not match sealed plan context {}",
                plan.workload.context_limit
            );
        }
    }
    let metal = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(metal.clone(), plan).map_err(anyhow::Error::msg)?;
    let mapped_artifact = Arc::new(MappedArtifact::open(artifact_path)?);
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_mapped_artifact(
        &metal,
        mapped_artifact.clone(),
    )
    .map_err(anyhow::Error::msg)?;
    let moe =
        qx_runtime_v2::PreparedMoeStack::load_from_mapped_artifact(&metal, mapped_artifact.clone())
            .map_err(anyhow::Error::msg)?;
    let initial_token_bytes = initial_token.to_le_bytes();
    if iterations > context_length {
        bail!("iterations must fit within the V2 context");
    }
    // Warm pipeline state without advancing the measured sequential context.
    // Position zero is overwritten when the measured run starts.
    for _ in 0..warmup {
        metal
            .write_runtime_buffer(
                buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                &initial_token_bytes,
            )
            .map_err(anyhow::Error::msg)?;
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, 0)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    if measurement == MeasurementMode::PostPrefill {
        for (position, token) in tokens.iter().enumerate() {
            metal
                .write_runtime_buffer(
                    buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                    &token.to_le_bytes(),
                )
                .map_err(anyhow::Error::msg)?;
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            stack
                .encode_token(&mut graph, &buffers, &moe, position)
                .map_err(anyhow::Error::msg)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        }
    } else {
        metal
            .write_runtime_buffer(
                buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                &initial_token_bytes,
            )
            .map_err(anyhow::Error::msg)?;
    }
    let before = metal.metrics().map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    let measured_position_start = prefill_tokens;
    for position in measured_position_start..measured_position_start + iterations {
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, position)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    let wall = start.elapsed();
    let after = metal.metrics().map_err(anyhow::Error::msg)?;
    let count = iterations.max(1) as f64;
    let gpu_seconds = after.gpu_time_seconds - before.gpu_time_seconds;
    let token = metal
        .read_runtime_u32(buffers.buffer(qx_runtime_v2::BufferId::TokenOutput))
        .map_err(anyhow::Error::msg)?;
    let (artifact_bytes, artifact_hash) = artifact_fingerprint(artifact_path)?;
    let plan_id = sealed_plan
        .as_ref()
        .map(|plan| plan.plan_id.clone())
        .unwrap_or_else(|| "legacy-v1-unsealed".to_owned());
    let target_id = mapped_artifact.info().header.target_id.clone();
    let wall_ms_per_token = 1000.0 * wall.as_secs_f64() / count;
    let gpu_ms_per_token = 1000.0 * gpu_seconds / count;
    let command_buffers_per_token = (after.command_buffers - before.command_buffers) as f64 / count;
    let encoders_per_token = (after.encoders - before.encoders) as f64 / count;
    let dispatches_per_token = (after.dispatches - before.dispatches) as f64 / count;
    let (current_physical_footprint_bytes, peak_physical_footprint_bytes) = physical_footprint();
    let record = V2TokenBenchmarkRecord {
        schema_version: 1,
        implementation: "qx-v2",
        qx_revision: option_env!("QX_GIT_REVISION").unwrap_or("unknown"),
        artifact: artifact_path.display().to_string(),
        artifact_bytes,
        artifact_hash: format!("{artifact_hash:016x}"),
        artifact_version: mapped_artifact.info().header.version,
        plan_id: plan_id.clone(),
        target_id,
        fixture_name: fixture
            .map(|fixture| fixture.name.clone())
            .unwrap_or_else(unnamed_fixture),
        fixture_domain: fixture
            .map(|fixture| fixture.domain.clone())
            .unwrap_or_else(unknown_domain),
        fixture_token_hash: format!("{:016x}", token_fixture_hash(tokens)),
        measurement,
        context: context_length,
        prefill_tokens,
        measured_position_start,
        measured_tokens: iterations,
        warmup,
        wall_ms_per_token,
        gpu_ms_per_token,
        tokens_per_second: count / wall.as_secs_f64(),
        command_buffers_per_token,
        encoders_per_token,
        dispatches_per_token,
        internal_cpu_waits: 0,
        current_rss_bytes: current_rss_bytes(),
        peak_rss_bytes: peak_rss_bytes(),
        current_physical_footprint_bytes,
        peak_physical_footprint_bytes,
        mapped_artifact_bytes: artifact_bytes,
        known_metal_allocated_bytes: metal.current_allocated_bytes(),
        recommended_metal_working_set_bytes: metal.recommended_working_set_bytes(),
        os_version: command_output("sw_vers", &["-productVersion"]),
        power_state: command_output("pmset", &["-g", "batt"]),
        thermal_state: command_output("pmset", &["-g", "therm"]),
        memory_pressure: command_output("memory_pressure", &["-Q"]),
        last_token_id: token,
    };
    println!("QX V2 complete token graph (artifact-only, sequential positions)");
    println!("artifact:               {}", artifact_path.display());
    println!("artifact bytes:         {artifact_bytes}");
    println!("artifact hash:          {artifact_hash:016x}");
    println!("artifact plan:          {}", plan_id);
    println!("context:                {context_length}");
    println!("initial token ID:       {initial_token}");
    println!("layers:                 {}", stack.layers.len());
    println!("warmup:                {warmup}");
    println!("iterations:             {iterations}");
    println!("last GPU token ID:      {token}");
    println!("process peak RSS bytes:  {}", record.peak_rss_bytes);
    println!("process current RSS bytes: {}", record.current_rss_bytes);
    println!(
        "physical footprint current/peak: {} / {}",
        record.current_physical_footprint_bytes, record.peak_physical_footprint_bytes
    );
    println!(
        "Metal allocated bytes:   {}",
        record.known_metal_allocated_bytes
    );
    println!("wall ms/token:          {:.3}", wall_ms_per_token);
    println!("GPU ms/token:           {:.3}", gpu_ms_per_token);
    println!("command buffers/token:  {:.2}", command_buffers_per_token);
    println!("encoders/token:         {:.2}", encoders_per_token);
    println!("dispatches/token:       {:.2}", dispatches_per_token);
    println!("equivalent complete tok/s: {:.2}", record.tokens_per_second);
    if let Some(path) = result_json {
        std::fs::write(path, serde_json::to_vec_pretty(&record)?)?;
        println!("result JSON:            {}", path.display());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_generate(
    model_path: &std::path::Path,
    artifact_path: &std::path::Path,
    prompt: &str,
    max_new_tokens: usize,
    context_length: usize,
) -> Result<()> {
    use std::rc::Rc;
    let tokenizer = qx_tokenizer::Tokenizer::from_dir(model_path)?;
    let prompt_tokens = tokenizer.encode(prompt)?;
    if prompt_tokens.is_empty() {
        bail!("V2 prompt must produce at least one token");
    }
    if prompt_tokens.len().saturating_add(max_new_tokens) > context_length {
        bail!("prompt plus requested generation exceeds the V2 context");
    }
    let metal = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(metal.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_artifact(&metal, artifact_path)
        .map_err(anyhow::Error::msg)?;
    let moe =
        qx_runtime_v2::PreparedMoeStack::load(&metal, artifact_path).map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    for (position, token) in prompt_tokens.iter().enumerate() {
        metal
            .write_runtime_buffer(
                buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                &token.to_le_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, position)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
    }
    let prefill_wall = start.elapsed();
    let decode_start = Instant::now();
    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(
        metal
            .read_runtime_u32(buffers.buffer(qx_runtime_v2::BufferId::TokenOutput))
            .map_err(anyhow::Error::msg)?,
    );
    for position in prompt_tokens.len()..prompt_tokens.len() + max_new_tokens.saturating_sub(1) {
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, position)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        generated.push(
            metal
                .read_runtime_u32(buffers.buffer(qx_runtime_v2::BufferId::TokenOutput))
                .map_err(anyhow::Error::msg)?,
        );
    }
    let decode_wall = decode_start.elapsed();
    let token = *generated.last().expect("positive generation length");
    println!("QX V2 artifact-only generation");
    println!("prompt tokens:           {}", prompt_tokens.len());
    println!("generated tokens:        {max_new_tokens}");
    println!("last GPU token ID:       {token}");
    println!("decoded tail:            {}", tokenizer.decode(&generated)?);
    println!(
        "TTFT/prefill ms:         {:.3}",
        1000.0 * prefill_wall.as_secs_f64()
    );
    let evaluated_decode_tokens = generated.len().saturating_sub(1);
    if evaluated_decode_tokens == 0 {
        println!("decode wall ms/token:    n/a (first token is produced by prefill)");
        println!("decode tok/s:            n/a (no post-prefill token evaluation)");
    } else {
        println!(
            "decode wall ms/token:    {:.3}",
            1000.0 * decode_wall.as_secs_f64() / evaluated_decode_tokens as f64
        );
        println!(
            "decode tok/s:            {:.2}",
            evaluated_decode_tokens as f64 / decode_wall.as_secs_f64()
        );
    }
    println!("process peak RSS bytes:   {}", peak_rss_bytes());
    println!("process current RSS bytes: {}", current_rss_bytes());
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_v2_teacher_forced(
    artifact_path: &std::path::Path,
    tokens: &[u32],
    context_length: usize,
    active_k: Option<usize>,
) -> Result<Vec<Vec<f32>>> {
    use std::rc::Rc;

    let previous_k = std::env::var_os("QX_V2_MOE_ACTIVE_K");
    let previous_by_layer = std::env::var_os("QX_V2_MOE_ACTIVE_K_BY_LAYER");
    if active_k.is_none() {
        unsafe { std::env::remove_var("QX_V2_MOE_ACTIVE_K_BY_LAYER") };
    }
    match active_k {
        Some(k) => unsafe { std::env::set_var("QX_V2_MOE_ACTIVE_K", k.to_string()) },
        None => unsafe { std::env::remove_var("QX_V2_MOE_ACTIVE_K") },
    }
    let result = (|| {
        let metal = Rc::new(
            qx_metal::platform::MetalContext::system_default()
                .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
        );
        let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
        let buffers = qx_runtime_v2::GpuRuntimeBuffers::new(metal.clone(), plan)
            .map_err(anyhow::Error::msg)?;
        let stack =
            qx_runtime_v2::PreparedAttentionStack::load_from_artifact(&metal, artifact_path)
                .map_err(anyhow::Error::msg)?;
        let moe = qx_runtime_v2::PreparedMoeStack::load(&metal, artifact_path)
            .map_err(anyhow::Error::msg)?;
        let mut logits = Vec::with_capacity(tokens.len().saturating_sub(1));
        for (position, &token) in tokens
            .iter()
            .take(tokens.len().saturating_sub(1))
            .enumerate()
        {
            metal
                .write_runtime_buffer(
                    buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                    &token.to_le_bytes(),
                )
                .map_err(anyhow::Error::msg)?;
            let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
            stack
                .encode_token(&mut graph, &buffers, &moe, position)
                .map_err(anyhow::Error::msg)?;
            buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
            logits.push(
                metal
                    .read_runtime_f32(
                        buffers.buffer(qx_runtime_v2::BufferId::Logits),
                        qx_model::VOCAB,
                    )
                    .map_err(anyhow::Error::msg)?,
            );
        }
        Ok(logits)
    })();
    match previous_k {
        Some(value) => unsafe { std::env::set_var("QX_V2_MOE_ACTIVE_K", value) },
        None => unsafe { std::env::remove_var("QX_V2_MOE_ACTIVE_K") },
    }
    match previous_by_layer {
        Some(value) => unsafe { std::env::set_var("QX_V2_MOE_ACTIVE_K_BY_LAYER", value) },
        None => unsafe { std::env::remove_var("QX_V2_MOE_ACTIVE_K_BY_LAYER") },
    }
    result
}

#[cfg(target_os = "macos")]
fn emit_v2_quality_trace(
    artifact_path: &std::path::Path,
    token_path: &std::path::Path,
    output_path: &std::path::Path,
    context_length: usize,
    sample_stride: u32,
) -> Result<()> {
    if context_length < 2 || sample_stride == 0 {
        bail!("quality tracing requires context >= 2 and a positive sample stride");
    }
    let corpus = qx_eval::read_token_corpus(token_path)?;
    if corpus.tokens.len() < 2 {
        bail!("quality corpus requires at least two tokens");
    }
    let mut generator = qx_runtime_v2::GraniteV2Generator::load(artifact_path, context_length)
        .map_err(anyhow::Error::msg)?;
    let mut blocks = Vec::new();
    let mut samples = Vec::new();
    let mut start = 0_usize;
    while start + 1 < corpus.tokens.len() {
        let end = (start + context_length).min(corpus.tokens.len());
        let block = &corpus.tokens[start..end];
        generator.reset();
        let mut nll_sum = 0.0;
        for local_position in 0..block.len() - 1 {
            let logits = generator
                .step_logits(block[local_position])
                .map_err(anyhow::Error::msg)?;
            let target = block[local_position + 1];
            let nll = qx_eval::token_nll(&logits, target as usize);
            let global_position = start + local_position;
            nll_sum += nll;
            if global_position % sample_stride as usize == 0 {
                samples.push(qx_eval::LogitSample {
                    position: global_position as u64,
                    target,
                    nll,
                    logits,
                });
            }
        }
        blocks.push(qx_eval::QualityBlock {
            start_position: start as u64,
            positions: (block.len() - 1) as u32,
            nll_sum,
        });
        if end == corpus.tokens.len() {
            break;
        }
        start = end - 1;
    }
    let trace = qx_eval::QualityTrace {
        vocab: qx_model::VOCAB as u32,
        total_positions: (corpus.tokens.len() - 1) as u64,
        block_tokens: context_length as u32,
        sample_stride,
        blocks,
        samples,
    };
    qx_eval::write_quality_trace(output_path, &trace)?;
    let nll =
        trace.blocks.iter().map(|block| block.nll_sum).sum::<f64>() / trace.total_positions as f64;
    println!("QX V2 triangular quality trace");
    println!("domain:       {}", corpus.domain);
    println!("split:        {}", corpus.split);
    println!("positions:    {}", trace.total_positions);
    println!("blocks:       {}", trace.blocks.len());
    println!("logit samples: {}", trace.samples.len());
    println!("NLL/PPL:      {:.6} / {:.6}", nll, nll.exp());
    println!("trace:        {}", output_path.display());
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn emit_v2_quality_trace(
    _artifact_path: &std::path::Path,
    _token_path: &std::path::Path,
    _output_path: &std::path::Path,
    _context_length: usize,
    _sample_stride: u32,
) -> Result<()> {
    bail!("V2 quality tracing requires macOS Metal")
}

#[cfg(target_os = "macos")]
fn benchmark_v2_quality(
    model_path: &std::path::Path,
    artifact_path: &std::path::Path,
    quality_path: &std::path::Path,
    quality_tokens: usize,
    context_length: usize,
    active_k: usize,
) -> Result<()> {
    let tokenizer = qx_tokenizer::Tokenizer::from_dir(model_path)?;
    let text = std::fs::read_to_string(quality_path)?;
    let encoded = tokenizer.encode(&text)?;
    let count = encoded.len().min(quality_tokens.saturating_add(1));
    if count < 2 {
        bail!("quality text must provide at least two tokens");
    }
    if count > context_length {
        bail!("quality token count {count} exceeds context {context_length}");
    }
    let tokens = &encoded[..count];
    let baseline = run_v2_teacher_forced(artifact_path, tokens, context_length, None)?;
    let candidate = run_v2_teacher_forced(artifact_path, tokens, context_length, Some(active_k))?;
    let metrics = compare_teacher_forced(tokens, &baseline, &candidate);
    println!("QX V2 teacher-forced active-k quality");
    println!("artifact:       {}", artifact_path.display());
    println!("quality text:    {}", quality_path.display());
    println!("positions:       {}", metrics.positions);
    println!("canonical k:     8");
    println!("candidate k:     {active_k}");
    println!("canonical NLL:   {:.6}", metrics.baseline_nll);
    println!("candidate NLL:   {:.6}", metrics.candidate_nll);
    println!("delta NLL:       {:+.6}", metrics.delta_nll);
    println!("canonical PPL:   {:.6}", metrics.baseline_nll.exp());
    println!("candidate PPL:   {:.6}", metrics.candidate_nll.exp());
    println!(
        "PPL delta:       {:+.2}%",
        100.0 * (metrics.candidate_nll.exp() / metrics.baseline_nll.exp() - 1.0)
    );
    println!(
        "KL mean/median/p95: {:.6} / {:.6} / {:.6}",
        metrics.mean_kl, metrics.median_kl, metrics.p95_kl
    );
    println!("logit cosine:    {:.6}", metrics.logit_cosine);
    println!(
        "top-1/top-5/top-10: {:.2}% / {:.2}% / {:.2}%",
        100.0 * metrics.top1,
        100.0 * metrics.top5,
        100.0 * metrics.top10
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn benchmark_v2_parity(
    model_path: &std::path::Path,
    artifact_path: &std::path::Path,
    prompt: &str,
    context_length: usize,
) -> Result<()> {
    use std::rc::Rc;
    qx_runtime_v2::activate_artifact_execution_plan(artifact_path).map_err(anyhow::Error::msg)?;
    let tokenizer = qx_tokenizer::Tokenizer::from_dir(model_path)?;
    let tokens = tokenizer.encode(prompt)?;
    const CHECKPOINTS: [usize; 8] = [0, 1, 7, 31, 63, 127, 255, 511];
    if tokens.is_empty() || tokens.len() > context_length || context_length <= CHECKPOINTS[7] {
        bail!("multi-position parity requires a non-empty prompt and --context 512");
    }
    let mut cpu = qx_runtime_v2::CpuQ4Reference::load(artifact_path).map_err(anyhow::Error::msg)?;
    let metal = Rc::new(
        qx_metal::platform::MetalContext::system_default()
            .ok_or_else(|| anyhow::anyhow!("no Metal device is available"))?,
    );
    let plan = ExecutionPlan::granite(context_length).map_err(anyhow::Error::msg)?;
    let buffers =
        qx_runtime_v2::GpuRuntimeBuffers::new(metal.clone(), plan).map_err(anyhow::Error::msg)?;
    let stack = qx_runtime_v2::PreparedAttentionStack::load_from_artifact(&metal, artifact_path)
        .map_err(anyhow::Error::msg)?;
    let moe =
        qx_runtime_v2::PreparedMoeStack::load(&metal, artifact_path).map_err(anyhow::Error::msg)?;
    let fill_token = *tokens.last().expect("non-empty parity prompt");
    let mut max_logit_abs = 0.0_f32;
    let mut min_cosine = 1.0_f64;
    let mut max_key_abs = 0.0_f32;
    let mut max_value_abs = 0.0_f32;
    let mut max_attention_abs = 0.0_f32;
    let mut matching_argmax = 0;
    let mut checkpoint_count = 0;
    for position in 0..=CHECKPOINTS[7] {
        let token = tokens.get(position).copied().unwrap_or(fill_token);
        let cpu_logits = cpu.step(token, position).map_err(anyhow::Error::msg)?;
        metal
            .write_runtime_buffer(
                buffers.buffer(qx_runtime_v2::BufferId::TokenOutput),
                &token.to_le_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
        let mut graph = buffers.begin_token().map_err(anyhow::Error::msg)?;
        stack
            .encode_token(&mut graph, &buffers, &moe, position)
            .map_err(anyhow::Error::msg)?;
        buffers.finish_token(graph).map_err(anyhow::Error::msg)?;
        let gpu_logits = metal
            .read_runtime_f32(
                buffers.buffer(qx_runtime_v2::BufferId::Logits),
                qx_model::VOCAB,
            )
            .map_err(anyhow::Error::msg)?;
        if CHECKPOINTS.contains(&position) {
            let max_abs_diff = |left: &[f32], right: &[f32]| {
                left.iter()
                    .zip(right)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0_f32, f32::max)
            };
            let token_max_abs = max_abs_diff(&cpu_logits, &gpu_logits);
            let token_cosine = cosine_similarity(&cpu_logits, &gpu_logits);
            let cpu_token = argmax(&cpu_logits) as u32;
            let gpu_token = metal
                .read_runtime_u32(buffers.buffer(qx_runtime_v2::BufferId::TokenOutput))
                .map_err(anyhow::Error::msg)?;
            let mut key_abs = 0.0_f32;
            let mut value_abs = 0.0_f32;
            let cache_elements = (position + 1) * qx_model::KV_HEADS * qx_model::HEAD_DIM;
            for layer in 0..qx_model::LAYERS {
                let cpu_cache = cpu.cache_at(layer, position).map_err(anyhow::Error::msg)?;
                let key_cache = buffers.kv_key_buffer(layer).map_err(anyhow::Error::msg)?;
                let value_cache = buffers.kv_value_buffer(layer).map_err(anyhow::Error::msg)?;
                let gpu_keys = if qx_metal::platform::kv_key_fp16_for_layer(layer) {
                    metal
                        .read_runtime_f16(&key_cache, cache_elements)
                        .map_err(anyhow::Error::msg)?
                } else {
                    metal
                        .read_runtime_f32(&key_cache, cache_elements)
                        .map_err(anyhow::Error::msg)?
                };
                let gpu_values = if qx_metal::platform::kv_value_fp16_for_layer(layer) {
                    metal
                        .read_runtime_f16(&value_cache, cache_elements)
                        .map_err(anyhow::Error::msg)?
                } else {
                    metal
                        .read_runtime_f32(&value_cache, cache_elements)
                        .map_err(anyhow::Error::msg)?
                };
                let slot_start = position * qx_model::KV_HEADS * qx_model::HEAD_DIM;
                key_abs = key_abs.max(max_abs_diff(
                    cpu_cache.0,
                    &gpu_keys[slot_start..slot_start + cpu_cache.0.len()],
                ));
                value_abs = value_abs.max(max_abs_diff(
                    cpu_cache.1,
                    &gpu_values[slot_start..slot_start + cpu_cache.1.len()],
                ));
            }
            let gpu_attention = metal
                .read_runtime_f32(
                    buffers.buffer(qx_runtime_v2::BufferId::Attention),
                    qx_model::HIDDEN,
                )
                .map_err(anyhow::Error::msg)?;
            let attention_abs = max_abs_diff(cpu.last_attention(), &gpu_attention);
            max_logit_abs = max_logit_abs.max(token_max_abs);
            min_cosine = min_cosine.min(token_cosine);
            max_key_abs = max_key_abs.max(key_abs);
            max_value_abs = max_value_abs.max(value_abs);
            max_attention_abs = max_attention_abs.max(attention_abs);
            if cpu_token == gpu_token {
                matching_argmax += 1;
            }
            checkpoint_count += 1;
            println!(
                "position {} token {}: KV key={:.6} value={:.6} attention={:.6} logits={:.6} cosine={:.6} cpu_argmax={} gpu_argmax={}",
                position,
                token,
                key_abs,
                value_abs,
                attention_abs,
                token_max_abs,
                token_cosine,
                cpu_token,
                gpu_token
            );
        }
    }
    println!("QX V2 Q4 artifact parity");
    println!("checkpoint positions:   {:?}", CHECKPOINTS);
    println!("max KV key error:        {max_key_abs:.6}");
    println!("max KV value error:      {max_value_abs:.6}");
    println!("max attention error:     {max_attention_abs:.6}");
    println!("max absolute logit:      {max_logit_abs:.6}");
    println!("minimum logit cosine:    {min_cosine:.6}");
    println!(
        "argmax agreement:       {}/{}",
        matching_argmax, checkpoint_count
    );
    if checkpoint_count != CHECKPOINTS.len()
        || matching_argmax != checkpoint_count
        || min_cosine < 0.99999
        || max_logit_abs > 0.25
        || max_key_abs > 0.01
        || max_value_abs > 0.01
        || max_attention_abs > 0.01
    {
        bail!("V2 GPU failed the multi-position KV/attention/logit parity gate");
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_parity(
    _model_path: &std::path::Path,
    _artifact_path: &std::path::Path,
    _prompt: &str,
    _context_length: usize,
) -> Result<()> {
    bail!("--v2-parity requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_quality(
    _model_path: &std::path::Path,
    _artifact_path: &std::path::Path,
    _quality_path: &std::path::Path,
    _quality_tokens: usize,
    _context_length: usize,
    _active_k: usize,
) -> Result<()> {
    bail!("--v2-quality requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_generate(
    _model_path: &std::path::Path,
    _artifact_path: &std::path::Path,
    _prompt: &str,
    _max_new_tokens: usize,
    _context_length: usize,
) -> Result<()> {
    bail!("--v2-generate requires macOS Metal")
}

#[cfg(not(target_os = "macos"))]
fn benchmark_v2_token(
    _artifact_path: &std::path::Path,
    _context_length: usize,
    _warmup: usize,
    _iterations: usize,
    _fixture: Option<&BenchmarkFixture>,
    _result_json: Option<&std::path::Path>,
) -> Result<()> {
    bail!("--v2-token requires macOS Metal")
}

#[cfg(target_os = "macos")]
fn benchmark_gpu_expert(
    expert: &QuantizedExpert,
    iterations: usize,
    info: &qx_format::QxArtifactInfo,
) -> Result<()> {
    let context = qx_metal::platform::MetalContext::system_default()
        .ok_or_else(|| anyhow::anyhow!("Metal device unavailable"))?;
    let gate_up_scales = expert
        .gate_up
        .scales
        .iter()
        .map(|scale| scale.to_bits())
        .collect::<Vec<_>>();
    let down_scales = expert
        .down
        .scales
        .iter()
        .map(|scale| scale.to_bits())
        .collect::<Vec<_>>();
    let prepared = context
        .prepare_fused_expert_bytes_with_layout(
            expert.gate_up.quant_type,
            &expert.gate_up.data,
            unsafe {
                std::slice::from_raw_parts(
                    gate_up_scales.as_ptr() as *const u8,
                    gate_up_scales.len() * std::mem::size_of::<u16>(),
                )
            },
            &expert.down.data,
            unsafe {
                std::slice::from_raw_parts(
                    down_scales.as_ptr() as *const u8,
                    down_scales.len() * std::mem::size_of::<u16>(),
                )
            },
            expert.gate_up.layout,
        )
        .map_err(anyhow::Error::msg)?;
    let input = vec![f16::from_f32(0.125).to_bits(); qx_model::HIDDEN];
    for _ in 0..iterations.min(10) {
        context
            .run_fused_expert(
                expert.gate_up.quant_type,
                &expert.gate_up.data,
                &gate_up_scales,
                &expert.down.data,
                &down_scales,
                &input,
            )
            .map_err(anyhow::Error::msg)?;
    }
    let before = context.metrics().map_err(anyhow::Error::msg)?;
    let start = Instant::now();
    for _ in 0..iterations {
        context
            .run_prepared_expert(&prepared, &input)
            .map_err(anyhow::Error::msg)?;
    }
    let elapsed = start.elapsed();
    let after = context.metrics().map_err(anyhow::Error::msg)?;
    let gpu_seconds = (after.gpu_time_seconds - before.gpu_time_seconds).max(0.0);
    let packed_bytes = (expert.gate_up.data.len()
        + expert.down.data.len()
        + (gate_up_scales.len() + down_scales.len()) * std::mem::size_of::<u16>())
        as f64;
    println!("GPU format:      {:?}", expert.gate_up.quant_type);
    println!("GPU artifact:    {}", info.header.target_id);
    println!(
        "GPU wall µs/op:  {:.2}",
        elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64
    );
    println!(
        "GPU time µs/op:  {:.2}",
        gpu_seconds * 1_000_000.0 / iterations as f64
    );
    println!(
        "effective GB/s:  {:.2}",
        packed_bytes / gpu_seconds.max(f64::MIN_POSITIVE) / 1e9
    );
    println!(
        "GPU dispatches:  {}",
        after.dispatches.saturating_sub(before.dispatches)
    );
    Ok(())
}

fn benchmark_runtime(
    model_path: &std::path::Path,
    artifact_path: &std::path::Path,
    options: RuntimeOptions<'_>,
) -> Result<()> {
    let load_start = Instant::now();
    let mut model = GraniteModel::from_paths(model_path, artifact_path, options.context)
        .map_err(anyhow::Error::msg)?;
    let load = load_start.elapsed();
    for _ in 0..options.warmup {
        run_stats(&mut model, &options)?;
    }
    #[cfg(target_os = "macos")]
    let metrics_before = model.metal_metrics();
    let mut stats = Vec::with_capacity(options.iterations);
    for _ in 0..options.iterations {
        stats.push(run_stats(&mut model, &options)?);
    }
    println!("QX full runtime benchmark");
    println!("model:              {}", model_path.display());
    println!("artifact:           {}", artifact_path.display());
    print_memory_plan(artifact_path)?;
    println!("prompt tokens:      {}", stats[0].prompt_tokens);
    println!("generated tokens:   {}", stats[0].generated_tokens);
    println!("load ms:            {:.2}", millis(load));
    println!("warmup:             {}", options.warmup);
    println!("iterations:         {}", options.iterations);
    println!(
        "TTFT p50 ms:        {:.2}",
        percentile(&stats, |s| s.ttft())
    );
    println!(
        "TTFT min/max ms:    {:.2} / {:.2}",
        minimum(&stats, |s| s.ttft()),
        maximum(&stats, |s| s.ttft())
    );
    println!(
        "decode p50 ms:      {:.2}",
        percentile(&stats, |s| s.decode)
    );
    println!(
        "decode p95 ms:      {:.2}",
        percentile95(&stats, |s| s.decode)
    );
    println!(
        "decode min/max ms:  {:.2} / {:.2}",
        minimum(&stats, |s| s.decode),
        maximum(&stats, |s| s.decode)
    );
    println!(
        "decode tok/s p50:   {:.2}",
        median_f64(&stats, |s| s.tokens_per_second())
    );
    println!("total p50 ms:       {:.2}", percentile(&stats, |s| s.total));
    println!(
        "total min/max ms:   {:.2} / {:.2}",
        minimum(&stats, |s| s.total),
        maximum(&stats, |s| s.total)
    );
    #[cfg(target_os = "macos")]
    if let (Some(before), Some(after)) = (metrics_before, model.metal_metrics()) {
        let command_buffers = after.command_buffers.saturating_sub(before.command_buffers);
        let encoders = after.encoders.saturating_sub(before.encoders);
        let dispatches = after.dispatches.saturating_sub(before.dispatches);
        let gpu_seconds = (after.gpu_time_seconds - before.gpu_time_seconds).max(0.0);
        let encode_seconds = (after.cpu_encode_seconds - before.cpu_encode_seconds).max(0.0);
        let wait_seconds = (after.cpu_wait_seconds - before.cpu_wait_seconds).max(0.0);
        let generated = stats
            .iter()
            .map(|stat| stat.generated_tokens)
            .sum::<usize>()
            .max(1);
        println!("Metal execution metrics");
        println!(
            "command buffers:    {command_buffers} ({:.2}/token)",
            command_buffers as f64 / generated as f64
        );
        println!(
            "encoders:           {encoders} ({:.2}/token)",
            encoders as f64 / generated as f64
        );
        println!(
            "dispatches:         {dispatches} ({:.2}/token)",
            dispatches as f64 / generated as f64
        );
        println!(
            "GPU time ms:        {:.2} ({:.3}/token)",
            1000.0 * gpu_seconds,
            1000.0 * gpu_seconds / generated as f64
        );
        println!(
            "CPU encode ms:      {:.2} ({:.3}/token)",
            1000.0 * encode_seconds,
            1000.0 * encode_seconds / generated as f64
        );
        println!(
            "CPU wait ms:        {:.2} ({:.3}/token)",
            1000.0 * wait_seconds,
            1000.0 * wait_seconds / generated as f64
        );
    }
    if options.profile {
        let (_, profile) = run_profile(&mut model, &options)?;
        print_profile(&profile);
    }
    if let Some(compare_path) = options.compare_artifact {
        compare_runtime_quality(model_path, artifact_path, compare_path, &options)?;
    }
    Ok(())
}

fn load_fixture(path: &std::path::Path) -> Result<BenchmarkFixture> {
    let fixture: BenchmarkFixture = serde_json::from_slice(&std::fs::read(path)?)?;
    if fixture.tokens.is_empty() {
        bail!("benchmark fixture contains no prompt tokens");
    }
    println!("fixture:             {}", path.display());
    if let Some(context) = fixture.context {
        println!("fixture context:     {context}");
    }
    if let Some(generation) = fixture.generation {
        println!("fixture generation:  {generation}");
    }
    Ok(fixture)
}

fn run_stats(model: &mut GraniteModel, options: &RuntimeOptions<'_>) -> Result<GenerationStats> {
    if let Some(tokens) = options.fixture_tokens.as_deref() {
        Ok(model
            .generate_tokens_with_trace_mode(tokens, options.max_new_tokens, !options.ignore_eos)
            .map_err(anyhow::Error::msg)?
            .stats)
    } else {
        Ok(model
            .generate_with_stats(options.prompt, options.max_new_tokens)
            .map_err(anyhow::Error::msg)?
            .1)
    }
}

fn run_trace(model: &mut GraniteModel, options: &RuntimeOptions<'_>) -> Result<GenerationTrace> {
    if let Some(tokens) = options.fixture_tokens.as_deref() {
        model
            .generate_tokens_with_trace_mode(tokens, options.max_new_tokens, !options.ignore_eos)
            .map_err(anyhow::Error::msg)
    } else {
        model
            .generate_with_trace(options.prompt, options.max_new_tokens)
            .map_err(anyhow::Error::msg)
    }
}

fn run_profile(
    model: &mut GraniteModel,
    options: &RuntimeOptions<'_>,
) -> Result<(GenerationStats, RuntimeProfile)> {
    if let Some(tokens) = options.fixture_tokens.as_deref() {
        model
            .generate_tokens_with_profile_mode(tokens, options.max_new_tokens, !options.ignore_eos)
            .map_err(anyhow::Error::msg)
    } else {
        model
            .generate_with_profile(options.prompt, options.max_new_tokens)
            .map_err(anyhow::Error::msg)
    }
}

fn print_profile(profile: &RuntimeProfile) {
    println!();
    println!("QX decode profile ({} profiled tokens)", profile.tokens);
    println!("component totals ms");
    println!(
        "  attention:         {:.2}",
        millis(profile.total_attention())
    );
    println!("  router:            {:.2}", millis(profile.total_router()));
    println!("  MoE:               {:.2}", millis(profile.total_moe()));
    println!(
        "  residual:          {:.2}",
        millis(profile.total_residual())
    );
    println!("  LM head:           {:.2}", millis(profile.lm_head));
    println!("per-layer ms: layer attention router moe residual total");
    for (layer, timing) in profile.layers.iter().enumerate() {
        println!(
            "  {layer:02}: {:.3} {:.3} {:.3} {:.3} {:.3}",
            millis(timing.attention),
            millis(timing.router),
            millis(timing.moe),
            millis(timing.residual),
            millis(timing.total)
        );
    }
}

fn print_memory_plan(artifact_path: &std::path::Path) -> Result<()> {
    let artifact = MappedArtifact::open(artifact_path)?;
    let info = artifact.info();
    let mut layouts = std::collections::BTreeMap::<String, usize>::new();
    for entry in info.entries.iter().filter(|entry| entry.is_expert()) {
        *layouts.entry(format!("{:?}", entry.layout)).or_default() += 1;
    }
    println!("planned weights bytes: {}", info.memory_plan.weights_bytes);
    println!("planned total bytes:   {}", info.memory_plan.total_bytes);
    println!("artifact version:      {}", info.header.version);
    println!("weight slab offset:    {}", info.header.weights_offset);
    println!("weight slab bytes:     {}", info.header.weights_length);
    println!(
        "execution plan:        {}",
        info.execution_plan
            .as_ref()
            .map(|plan| plan.plan_id.as_str())
            .unwrap_or("legacy-v1-unsealed")
    );
    println!("expert packing layouts: {:?}", layouts);
    println!("process peak RSS bytes: {}", peak_rss_bytes());
    Ok(())
}

fn print_v2_plan(artifact: &MappedArtifact, context: usize) -> Result<()> {
    let plan = ExecutionPlan::granite(context).map_err(anyhow::Error::msg)?;
    let sealed = artifact.info().execution_plan.as_ref();
    if let Some(sealed) = sealed
        && sealed.workload.context_limit as usize != context
    {
        bail!(
            "requested context {context} differs from sealed context {}",
            sealed.workload.context_limit
        );
    }
    let (expert_bytes, dense_bytes) =
        artifact
            .info()
            .entries
            .iter()
            .fold((0_u64, 0_u64), |(expert, dense), entry| {
                let bytes = entry.scale_length + entry.data_length;
                if entry.is_expert() {
                    (expert + bytes, dense)
                } else {
                    (expert, dense + bytes)
                }
            });
    let audit = plan.memory_audit_with_dense_artifact(expert_bytes as usize, dense_bytes as usize);
    let key_kv_bytes = if qx_metal::platform::kv_key_fp16_enabled() {
        plan.kv_bytes / 4
    } else {
        plan.kv_bytes / 2
    };
    let value_kv_bytes = if qx_metal::platform::kv_value_fp16_enabled() {
        plan.kv_bytes / 4
    } else {
        plan.kv_bytes / 2
    };
    println!("QX Runtime V2 static audit");
    println!(
        "sealed plan:                {}",
        sealed
            .map(|plan| plan.plan_id.as_str())
            .unwrap_or("legacy-v1")
    );
    println!(
        "weight ownership:           {}",
        sealed
            .map(|plan| format!("{:?}", plan.memory.weight_storage))
            .unwrap_or_else(|| "legacy-copied".to_owned())
    );
    println!(
        "compiler revision:          {}",
        sealed
            .map(|plan| plan.provenance.compiler_revision.as_str())
            .unwrap_or("unsealed")
    );
    println!(
        "benchmark contract hash:    {}",
        sealed
            .map(|plan| plan.provenance.benchmark_contract_hash.as_str())
            .filter(|value| !value.is_empty())
            .unwrap_or("missing")
    );
    println!(
        "predicted peak bytes:       {}",
        sealed
            .map(|plan| plan.memory.predicted_peak_bytes)
            .unwrap_or(0)
    );
    println!("context:                    {}", plan.context);
    println!(
        "arena bytes:                {}",
        sealed
            .filter(|plan| plan.arena.arena_bytes != 0)
            .map(|plan| plan.arena.arena_bytes as usize)
            .unwrap_or(audit.arena_bytes)
    );
    println!("KV bytes (FP32 validation): {}", audit.kv_bytes);
    println!(
        "KV mode:                    K={} V={}",
        if qx_metal::platform::kv_key_fp16_enabled() {
            "FP16"
        } else {
            "FP32"
        },
        if qx_metal::platform::kv_value_fp16_enabled() {
            "FP16"
        } else {
            "FP32"
        }
    );
    println!("K cache bytes (runtime):     {}", key_kv_bytes);
    println!("V cache bytes (runtime):     {}", value_kv_bytes);
    println!(
        "KV bytes (runtime):          {}",
        key_kv_bytes + value_kv_bytes
    );
    println!(
        "expert artifact bytes:      {}",
        audit.expert_artifact_bytes
    );
    println!(
        "dense Q4 artifact bytes:     {}",
        audit.dense_artifact_bytes
    );
    println!(
        "dense FP32 reference:       {}",
        audit.dense_fp32_reference_bytes
    );
    println!("dense Q4 full target:       {}", audit.dense_q4_full_bytes);
    println!("attention Q4 currently:     {}", audit.attention_q4_bytes);
    println!(
        "FP32 reference total:       {}",
        audit.estimated_fp32_reference_bytes
    );
    println!(
        "Q4 full target total:       {}",
        audit.estimated_q4_full_resident_bytes
    );
    println!(
        "Q4 full target MiB:         {:.1}",
        audit.estimated_q4_full_resident_mib()
    );
    println!(
        "command buffers/token:      {}",
        sealed
            .map(|plan| plan.schedule.command_buffers_per_token as usize)
            .unwrap_or(plan.command_buffers_per_token)
    );
    println!(
        "encoders/token:             {}",
        sealed
            .map(|plan| plan.schedule.encoders_per_token as usize)
            .unwrap_or(plan.encoders_per_token)
    );
    println!(
        "planned dispatches/token:   {}",
        sealed
            .map(|plan| plan.schedule.expected_dispatches_per_token as usize)
            .unwrap_or_else(|| plan.expected_dispatches_per_token())
    );
    Ok(())
}

fn peak_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage initializes the caller-provided rusage structure.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return 0;
    }
    // macOS reports ru_maxrss in bytes; Linux reports it in KiB.
    let bytes = unsafe { usage.assume_init().ru_maxrss } as u64;
    if cfg!(target_os = "macos") {
        bytes
    } else {
        bytes.saturating_mul(1024)
    }
}

#[cfg(target_os = "macos")]
fn current_rss_bytes() -> u64 {
    let mut task = std::mem::MaybeUninit::<libc::proc_taskinfo>::zeroed();
    let size = std::mem::size_of::<libc::proc_taskinfo>() as i32;
    // SAFETY: proc_pidinfo writes at most `size` bytes to the valid task-info
    // allocation supplied for the current process.
    let written = unsafe {
        libc::proc_pidinfo(
            std::process::id() as i32,
            libc::PROC_PIDTASKINFO,
            0,
            task.as_mut_ptr().cast(),
            size,
        )
    };
    if written != size {
        return 0;
    }
    // SAFETY: a full proc_taskinfo was written when `written == size`.
    unsafe { task.assume_init().pti_resident_size }
}

#[cfg(not(target_os = "macos"))]
fn current_rss_bytes() -> u64 {
    0
}

#[cfg(target_os = "macos")]
fn physical_footprint() -> (u64, u64) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: proc_pid_rusage initializes the supplied rusage_info_v4 buffer
    // for the current process when it returns zero.
    let status = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as i32,
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast(),
        )
    };
    if status != 0 {
        return (0, 0);
    }
    // SAFETY: a successful call initialized the complete structure.
    let usage = unsafe { usage.assume_init() };
    (
        usage.ri_phys_footprint,
        usage.ri_lifetime_max_phys_footprint,
    )
}

#[cfg(not(target_os = "macos"))]
fn physical_footprint() -> (u64, u64) {
    (0, 0)
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    std::process::Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn token_fixture_hash(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in tokens.iter().flat_map(|token| token.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn compare_runtime_quality(
    model_path: &std::path::Path,
    baseline_artifact: &std::path::Path,
    candidate_artifact: &std::path::Path,
    options: &RuntimeOptions<'_>,
) -> Result<()> {
    let load_start = Instant::now();
    let mut candidate = GraniteModel::from_paths(model_path, candidate_artifact, options.context)
        .map_err(anyhow::Error::msg)?;
    let candidate_load = load_start.elapsed();
    for _ in 0..options.warmup {
        run_stats(&mut candidate, options)?;
    }
    let mut baseline = GraniteModel::from_paths(model_path, baseline_artifact, options.context)
        .map_err(anyhow::Error::msg)?;
    let baseline_trace = run_trace(&mut baseline, options)?;
    let candidate_trace = run_trace(&mut candidate, options)?;
    let metrics = compare_traces(&baseline_trace, &candidate_trace);
    println!();
    println!("QX end-to-end quality comparison");
    println!("baseline artifact:  {}", baseline_artifact.display());
    println!("candidate artifact: {}", candidate_artifact.display());
    println!("candidate load ms:  {:.2}", millis(candidate_load));
    println!("baseline text:      {:?}", baseline_trace.text);
    println!("candidate text:     {:?}", candidate_trace.text);
    println!("compared steps:     {}", metrics.steps);
    println!("top-1 agreement:    {:.4}", metrics.top1_agreement);
    println!("logit cosine mean:  {:.6}", metrics.logit_cosine);
    println!("KL baseline||cand:  {:.6}", metrics.kl_divergence);
    println!(
        "candidate TTFT ms:  {:.2}",
        millis(candidate_trace.stats.ttft())
    );
    println!(
        "candidate decode:   {:.2} tok/s",
        candidate_trace.stats.tokens_per_second()
    );
    if let Some(quality_path) = options.quality_text {
        let quality_text = std::fs::read_to_string(quality_path)?;
        let baseline_ppl = baseline
            .perplexity(&quality_text, options.quality_tokens)
            .map_err(anyhow::Error::msg)?;
        let candidate_ppl = candidate
            .perplexity(&quality_text, options.quality_tokens)
            .map_err(anyhow::Error::msg)?;
        println!("quality text:       {}", quality_path.display());
        println!("quality tokens:     {}", options.quality_tokens);
        println!("baseline PPL:       {:.6}", baseline_ppl);
        println!("candidate PPL:      {:.6}", candidate_ppl);
        println!(
            "PPL delta:          {:+.6} ({:+.2}%)",
            candidate_ppl - baseline_ppl,
            100.0 * (candidate_ppl / baseline_ppl - 1.0)
        );
        let tokens = baseline
            .encode_text(&quality_text)
            .map_err(anyhow::Error::msg)?;
        let teacher_limit = options.quality_tokens.saturating_add(1);
        let baseline_logits = baseline
            .teacher_forced_logits(&tokens, teacher_limit)
            .map_err(anyhow::Error::msg)?;
        let candidate_logits = candidate
            .teacher_forced_logits(&tokens, teacher_limit)
            .map_err(anyhow::Error::msg)?;
        let teacher = compare_teacher_forced(&tokens, &baseline_logits, &candidate_logits);
        println!("teacher-forced positions: {}", teacher.positions);
        println!("teacher baseline NLL:     {:.6}", teacher.baseline_nll);
        println!("teacher candidate NLL:    {:.6}", teacher.candidate_nll);
        println!("teacher delta NLL:        {:+.6}", teacher.delta_nll);
        println!(
            "teacher KL mean/median/p95: {:.6} / {:.6} / {:.6}",
            teacher.mean_kl, teacher.median_kl, teacher.p95_kl
        );
        println!("teacher logit cosine:     {:.6}", teacher.logit_cosine);
        println!(
            "teacher top-1/top-5/top-10: {:.2}% / {:.2}% / {:.2}%",
            100.0 * teacher.top1,
            100.0 * teacher.top5,
            100.0 * teacher.top10
        );
    }
    print_memory_plan(candidate_artifact)?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct QualityMetrics {
    steps: usize,
    top1_agreement: f64,
    logit_cosine: f64,
    kl_divergence: f64,
}

#[derive(Clone, Copy, Debug)]
struct TeacherForcedMetrics {
    positions: usize,
    baseline_nll: f64,
    candidate_nll: f64,
    delta_nll: f64,
    mean_kl: f64,
    median_kl: f64,
    p95_kl: f64,
    logit_cosine: f64,
    top1: f64,
    top5: f64,
    top10: f64,
}

fn compare_teacher_forced(
    tokens: &[u32],
    baseline: &[Vec<f32>],
    candidate: &[Vec<f32>],
) -> TeacherForcedMetrics {
    let positions = baseline
        .len()
        .min(candidate.len())
        .min(tokens.len().saturating_sub(1));
    if positions == 0 {
        return TeacherForcedMetrics {
            positions: 0,
            baseline_nll: 0.0,
            candidate_nll: 0.0,
            delta_nll: 0.0,
            mean_kl: 0.0,
            median_kl: 0.0,
            p95_kl: 0.0,
            logit_cosine: 0.0,
            top1: 0.0,
            top5: 0.0,
            top10: 0.0,
        };
    }
    let mut baseline_nll = 0.0;
    let mut candidate_nll = 0.0;
    let mut kl_values = Vec::with_capacity(positions);
    let mut cosine = 0.0;
    let mut top1 = 0usize;
    let mut top5 = 0usize;
    let mut top10 = 0usize;
    for position in 0..positions {
        let target = tokens[position + 1] as usize;
        baseline_nll += token_nll(&baseline[position], target);
        candidate_nll += token_nll(&candidate[position], target);
        kl_values.push(kl_divergence(&baseline[position], &candidate[position]));
        cosine += cosine_similarity(&baseline[position], &candidate[position]);
        if argmax(&baseline[position]) == argmax(&candidate[position]) {
            top1 += 1;
        }
        top5 += top_k_overlap(&baseline[position], &candidate[position], 5);
        top10 += top_k_overlap(&baseline[position], &candidate[position], 10);
    }
    kl_values.sort_by(f64::total_cmp);
    let mean_kl = kl_values.iter().sum::<f64>() / positions as f64;
    let median_kl = kl_values[positions / 2];
    let p95_kl = kl_values[((positions - 1) * 95 / 100).min(positions - 1)];
    TeacherForcedMetrics {
        positions,
        baseline_nll: baseline_nll / positions as f64,
        candidate_nll: candidate_nll / positions as f64,
        delta_nll: (candidate_nll - baseline_nll) / positions as f64,
        mean_kl,
        median_kl,
        p95_kl,
        logit_cosine: cosine / positions as f64,
        top1: top1 as f64 / positions as f64,
        top5: top5 as f64 / (positions * 5) as f64,
        top10: top10 as f64 / (positions * 10) as f64,
    }
}

fn token_nll(logits: &[f32], target: usize) -> f64 {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp() as f64)
        .sum::<f64>()
        .ln();
    (log_sum + max_logit as f64 - logits.get(target).copied().unwrap_or(max_logit) as f64).max(0.0)
}

fn top_k_overlap(left: &[f32], right: &[f32], k: usize) -> usize {
    let mut left_indices = left.iter().enumerate().collect::<Vec<_>>();
    let mut right_indices = right.iter().enumerate().collect::<Vec<_>>();
    left_indices.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    right_indices.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    let right_top = right_indices
        .iter()
        .take(k)
        .map(|(index, _)| *index)
        .collect::<std::collections::HashSet<_>>();
    left_indices
        .iter()
        .take(k)
        .filter(|(index, _)| right_top.contains(index))
        .count()
}

fn compare_traces(baseline: &GenerationTrace, candidate: &GenerationTrace) -> QualityMetrics {
    let steps = baseline.logits.len().min(candidate.logits.len());
    if steps == 0 {
        return QualityMetrics {
            steps: 0,
            top1_agreement: 0.0,
            logit_cosine: 0.0,
            kl_divergence: 0.0,
        };
    }
    let mut top1 = 0usize;
    let mut cosine = 0.0;
    let mut kl = 0.0;
    for index in 0..steps {
        let left = &baseline.logits[index];
        let right = &candidate.logits[index];
        if argmax(left) == argmax(right) {
            top1 += 1;
        }
        cosine += cosine_similarity(left, right);
        kl += kl_divergence(left, right);
    }
    QualityMetrics {
        steps,
        top1_agreement: top1 as f64 / steps as f64,
        logit_cosine: cosine / steps as f64,
        kl_divergence: kl / steps as f64,
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    let (dot, left_norm, right_norm) = left.iter().zip(right).fold(
        (0.0_f64, 0.0_f64, 0.0_f64),
        |(dot, left_norm, right_norm), (&left, &right)| {
            let left = left as f64;
            let right = right as f64;
            (
                dot + left * right,
                left_norm + left * left,
                right_norm + right * right,
            )
        },
    );
    dot / (left_norm.sqrt() * right_norm.sqrt()).max(f64::MIN_POSITIVE)
}

fn kl_divergence(left: &[f32], right: &[f32]) -> f64 {
    let left_max = left.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let right_max = right.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let left_sum = left
        .iter()
        .map(|value| (*value - left_max).exp() as f64)
        .sum::<f64>();
    let right_sum = right
        .iter()
        .map(|value| (*value - right_max).exp() as f64)
        .sum::<f64>();
    left.iter().zip(right).fold(0.0, |kl, (&left, &right)| {
        let p = (left - left_max).exp() as f64 / left_sum;
        let q = (right - right_max).exp() as f64 / right_sum;
        kl + p * (p / q.max(f64::MIN_POSITIVE)).ln()
    })
}

fn millis(value: Duration) -> f64 {
    value.as_secs_f64() * 1_000.0
}

fn percentile<F>(stats: &[GenerationStats], selector: F) -> f64
where
    F: Fn(&GenerationStats) -> Duration,
{
    let mut values = stats.iter().map(selector).collect::<Vec<_>>();
    values.sort_unstable();
    millis(values[values.len() / 2])
}

fn percentile95<F>(stats: &[GenerationStats], selector: F) -> f64
where
    F: Fn(&GenerationStats) -> Duration,
{
    let mut values = stats.iter().map(selector).collect::<Vec<_>>();
    values.sort_unstable();
    let index = ((values.len() - 1) * 95 / 100).min(values.len() - 1);
    millis(values[index])
}

fn minimum<F>(stats: &[GenerationStats], selector: F) -> f64
where
    F: Fn(&GenerationStats) -> Duration,
{
    stats.iter().map(selector).min().map_or(0.0, millis)
}

fn maximum<F>(stats: &[GenerationStats], selector: F) -> f64
where
    F: Fn(&GenerationStats) -> Duration,
{
    stats.iter().map(selector).max().map_or(0.0, millis)
}

fn median_f64<F>(stats: &[GenerationStats], selector: F) -> f64
where
    F: Fn(&GenerationStats) -> f64,
{
    let mut values = stats.iter().map(selector).collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
