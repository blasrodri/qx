use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use qx_compiler::{
    ExpertSensitivity, ExpertWeights, QxTrace, allocate_with_minimum_options, granite_memory_plan,
    measure_sensitivity, quantized_bytes,
};
use qx_format::{
    CompiledTensor, ExecutionPlanManifest, ExpertRepresentation, KvPrecision, RawTensor,
    TensorKind, VerifierMoeConsumerTopology, WeightStorage, write_artifact_with_raw_and_plan,
};
use qx_model::{EXPERTS, Granite3A400M, GraniteConfig, LAYERS, MAX_CONTEXT, TOP_K};
use qx_quant::{PackingLayout, QuantType, QuantizedTensor};
use qx_safetensors::Checkpoint;
use std::{collections::BTreeMap, fmt::Write as _, fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "qx-compile",
    about = "Compile Granite MoE expert weights to a target-specific QX artifact"
)]
struct Args {
    #[arg(value_name = "MODEL")]
    model: PathBuf,
    #[arg(long)]
    target: String,
    #[arg(long, value_parser = parse_budget)]
    memory_budget: usize,
    #[arg(long, default_value_t = 0.05)]
    quality_loss: f32,
    /// Lowest precision allowed by the allocator (q2, q3, or q4).
    #[arg(long = "min-bits", default_value = "q2", value_parser = parse_quant_type)]
    minimum_bits: QuantType,
    /// Exclude Q3 to measure the hardware-friendly Q2/Q4 alternative.
    #[arg(long)]
    disable_q3: bool,
    /// Force every expert tensor to one quantization type. This is a
    /// representation control used by the runtime's uniform native-slab
    /// experiments; traced mixed allocation remains the default.
    #[arg(long, value_parser = parse_quant_type)]
    uniform_expert_bits: Option<QuantType>,
    /// Select whole layers for native Q3 only when all experts in the layer
    /// have finite traced sensitivity and the weighted layer error is within
    /// `--layerwise-q3-threshold`. Other layers remain native Q4.
    #[arg(long)]
    layerwise_q3: bool,
    #[arg(long, default_value_t = 0.01)]
    layerwise_q3_threshold: f32,
    /// Select Q3 independently for complete layer gate/up and down tensor
    /// families. This is the first executable mixed-precision candidate;
    /// expert-level routing selection remains a later slab-format step.
    #[arg(long)]
    tensorwise_q3: bool,
    /// Target expert bytes to remove with layer/tensor-family Q3 choices,
    /// ranked by traced incremental quality cost per byte. This preserves one
    /// native projection dispatch per family and requires `--tensorwise-q3`.
    #[arg(long, value_parser = parse_budget, requires = "tensorwise_q3")]
    tensorwise_q3_savings: Option<usize>,
    /// Select Q3 independently for every (layer, expert, gate/up|down)
    /// tensor. This emits compact typed slabs and requires the expertwise
    /// GPU-map runtime path.
    #[arg(long)]
    expertwise_q3: bool,
    #[arg(long, default_value_t = 0.01)]
    expertwise_q3_threshold: f32,
    /// Target expert bytes to remove with Q3, ranked by traced incremental
    /// quality cost per byte. This is an experimental alternative to the raw
    /// reconstruction-error threshold and requires `--expertwise-q3`.
    #[arg(long, value_parser = parse_budget, requires = "expertwise_q3")]
    expertwise_q3_savings: Option<usize>,
    #[arg(long, default_value_t = 32)]
    group_size: usize,
    /// Expert-only scientific control; dense tensors remain QX group-32 Q4.
    #[arg(long, value_enum, default_value_t = ExpertFormat::QxQ4)]
    expert_format: ExpertFormat,
    /// Physical expert layout candidate. Whole-model evidence currently makes
    /// row-major the Granite/M5 control even though Tile32 wins in isolation.
    #[arg(long, value_enum, default_value_t = ExpertLayout::RowMajor)]
    expert_layout: ExpertLayout,
    /// Physical layout for attention, router, and tied embedding/LM-head Q4
    /// projections.
    #[arg(long, value_enum, default_value_t = DenseLayout::RowMajor)]
    dense_layout: DenseLayout,
    /// Four-stage MoE execution candidate. `wide-expert` changes only the
    /// threadgroup mapping; it is never selected without complete-model
    /// evidence by qx-autotune.
    #[arg(long, value_enum, default_value_t = MoeVariant::FourStage)]
    moe_variant: MoeVariant,
    /// Compiler-selected mapping for the dynamic MoE transaction candidate.
    /// The value is persisted in the artifact plan for reproducible runtime
    /// activation without a process-wide environment override.
    #[arg(long, value_enum, default_value_t = MoeMappingArg::Qx)]
    moe_mapping: MoeMappingArg,
    /// Exact-Q4 MoE consumer dataflow selected by the compiler. The
    /// unique-packet variant is semantically identical to canonical Q4 but
    /// changes packed code/scale load ownership on Metal.
    #[arg(long, value_enum, default_value_t = MoeConsumerArg::Canonical)]
    moe_consumer: MoeConsumerArg,
    /// Exact-Q4 consumer for the row-inline MoE down projection. Kept
    /// independent from gate/up because the shapes and best packet mapping
    /// need not be identical.
    #[arg(long, value_enum, default_value_t = MoeConsumerArg::Canonical)]
    moe_down_consumer: MoeConsumerArg,
    /// Width-specific exact verifier gate/up consumer. K4 vector-packet
    /// reuses decoded Q4 packets across token activations; K1/K2 retain the
    /// normal unique-packet path.
    #[arg(long, value_enum, default_value_t = VerifierMoeConsumerArg::UniquePacket)]
    verifier_moe_consumer: VerifierMoeConsumerArg,
    /// Diagnostic K4 consumer that keeps two token accumulators live at a
    /// time. It trades packet reloads for lower accumulator/resource pressure.
    #[arg(long, default_value_t = false)]
    verifier_k4_split_accumulation: bool,
    /// Diagnostic K4 consumer that bypasses threadgroup activation staging.
    #[arg(long, default_value_t = false)]
    verifier_k4_direct_activation: bool,
    /// Diagnostic K4 mapping using sixteen output rows per threadgroup.
    #[arg(long, default_value_t = false)]
    verifier_k4_rows16: bool,
    /// Diagnostic K4 packet consumer with two packed packet pairs in flight.
    #[arg(long, default_value_t = false)]
    verifier_k4_unroll4: bool,
    /// Maximum decode context at which the unroll4 K4 consumer is eligible.
    /// Omit to make it eligible at every context; the planner may retain the
    /// baseline consumer above this bound.
    #[arg(long)]
    verifier_k4_unroll4_max_context: Option<u32>,
    /// Enable the independent K4 Down packet consumer.
    #[arg(long, default_value_t = false)]
    verifier_k4_down_unroll4: bool,
    /// Use active-expert indirect dispatch for the K-wide verifier gate/up
    /// path. This is a target/workload-specific execution-plan choice.
    #[arg(long, default_value_t = false)]
    verifier_active_expert_dispatch: bool,
    #[arg(long, default_value_t = MAX_CONTEXT)]
    context: usize,
    /// Stable identifier embedded in artifact v2 for compiler/autotuner provenance.
    #[arg(long, default_value = "granite-m5-top8-canonical")]
    plan_id: String,
    /// Matched-quality aggregate PPL budget carried by the compiled plan.
    #[arg(long, default_value_t = 0.01)]
    max_relative_ppl_delta: f64,
    /// Per-domain matched-quality PPL budget carried by the compiled plan.
    #[arg(long, default_value_t = 0.02)]
    max_domain_relative_ppl_delta: f64,
    #[arg(long, default_value = "")]
    calibration_hash: String,
    #[arg(long, default_value = "")]
    evaluation_hash: String,
    #[arg(long, default_value = "")]
    benchmark_contract_hash: String,
    /// Static-weight ownership plan embedded into artifact v2.
    #[arg(long, value_enum, default_value_t = WeightStorageArg::MmapNoCopy)]
    weight_storage: WeightStorageArg,
    /// Persistent KV representation sealed into the artifact. The canonical
    /// plan keeps attention keys in FP32 and stores values in FP16.
    #[arg(long, value_enum, default_value_t = KvPrecisionArg::Fp32KFp16V)]
    kv_precision: KvPrecisionArg,
    /// GPU-ordered command buffers per decode token. No intermediate CPU wait
    /// is permitted; layer boundaries are divided evenly between buffers.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..=24))]
    command_buffers_per_token: u16,
    /// Compiler-selected number of queue-ordered greedy token graphs kept in
    /// flight before a host synchronization boundary.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u16).range(1..=64))]
    rollout_depth: u16,
    /// QXTRACE produced by the llama.cpp calibration bridge. Without this,
    /// compilation is conservative and emits uniform Q4 experts.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// Write per-expert sensitivity and allocation evidence as CSV.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Emit Q4 row-major attention, router, and embedding tensors into the
    /// artifact. This is enabled by default so V2 does not benchmark against
    /// dense FP32 checkpoint weights.
    #[arg(long)]
    experts_only: bool,
    #[arg(short = 'o', long)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ExpertFormat {
    QxQ4,
    Q4K,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ExpertLayout {
    RowMajor,
    Tile32,
    RowInline,
    RowInline2,
    RowInline4,
    RowInline8,
    #[value(name = "transaction-144", alias = "transaction144")]
    Transaction144,
    #[value(name = "transaction-160", alias = "transaction160")]
    Transaction160,
    #[value(name = "superblock-144", alias = "superblock144")]
    Superblock144,
    #[value(name = "cross-expert-tile8", alias = "cross-expert")]
    CrossExpertTile8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DenseLayout {
    RowMajor,
    #[value(name = "superblock-144", alias = "superblock144")]
    Superblock144,
}

impl From<DenseLayout> for PackingLayout {
    fn from(value: DenseLayout) -> Self {
        match value {
            DenseLayout::RowMajor => Self::RowMajor,
            DenseLayout::Superblock144 => Self::Q4Superblock144,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MoeVariant {
    FourStage,
    WideExpert,
    CrossExpertActivation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MoeMappingArg {
    Qx,
    #[value(name = "llama-like", alias = "llama")]
    LlamaLike,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MoeConsumerArg {
    Canonical,
    #[value(name = "unique-packet", alias = "unique_packet")]
    UniquePacket,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum VerifierMoeConsumerArg {
    #[value(name = "unique-packet", alias = "unique_packet")]
    UniquePacket,
    #[value(name = "k4-vector-packet", alias = "k4_vector_packet")]
    K4VectorPacket,
}

impl From<ExpertLayout> for PackingLayout {
    fn from(value: ExpertLayout) -> Self {
        match value {
            ExpertLayout::RowMajor => Self::RowMajor,
            ExpertLayout::Tile32 => Self::MetalTile32,
            ExpertLayout::RowInline => Self::Q4RowInline,
            ExpertLayout::RowInline2 => Self::Q4RowInlineRows2,
            ExpertLayout::RowInline4 => Self::Q4RowInlineRows4,
            ExpertLayout::RowInline8 => Self::Q4RowInlineRows8,
            ExpertLayout::Transaction144 => Self::Q4Transaction144,
            ExpertLayout::Transaction160 => Self::Q4Transaction160,
            ExpertLayout::CrossExpertTile8 => Self::Q4CrossExpertTile8,
            ExpertLayout::Superblock144 => Self::Q4Superblock144,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum WeightStorageArg {
    CopiedShared,
    MmapNoCopy,
    Hybrid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum KvPrecisionArg {
    Fp32,
    Fp32KFp16V,
    Fp16KFp32V,
    Fp16,
}

impl From<KvPrecisionArg> for KvPrecision {
    fn from(value: KvPrecisionArg) -> Self {
        match value {
            KvPrecisionArg::Fp32 => Self::Fp32,
            KvPrecisionArg::Fp32KFp16V => Self::Fp32KFp16V,
            KvPrecisionArg::Fp16KFp32V => Self::Fp16KFp32V,
            KvPrecisionArg::Fp16 => Self::Fp16,
        }
    }
}

impl From<WeightStorageArg> for WeightStorage {
    fn from(value: WeightStorageArg) -> Self {
        match value {
            WeightStorageArg::CopiedShared => Self::CopiedShared,
            WeightStorageArg::MmapNoCopy => Self::MmapNoCopy,
            WeightStorageArg::Hybrid => Self::Hybrid,
        }
    }
}

fn parse_budget(value: &str) -> Result<usize, String> {
    let lower = value.to_ascii_lowercase();
    let (number, multiplier) = if let Some(value) = lower.strip_suffix("gb") {
        (value, 1usize << 30)
    } else if let Some(value) = lower.strip_suffix("mb") {
        (value, 1usize << 20)
    } else if let Some(value) = lower.strip_suffix("kb") {
        (value, 1usize << 10)
    } else {
        (lower.as_str(), 1)
    };
    number
        .parse::<usize>()
        .map(|value| value.saturating_mul(multiplier))
        .map_err(|_| "expected bytes or a suffix such as 512mb".to_owned())
}

fn parse_quant_type(value: &str) -> Result<QuantType, String> {
    QuantType::parse(value).ok_or_else(|| "expected q2, q3, or q4".to_owned())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.group_size == 0 || args.context == 0 {
        bail!("group size and context must be positive");
    }
    let kv_precision: KvPrecision = args.kv_precision.into();
    let runtime_plan =
        qx_runtime_v2::ExecutionPlan::granite_with_kv_precision(args.context, kv_precision)
            .map_err(anyhow::Error::msg)?;
    let fixed_dense_bytes = if args.experts_only {
        0
    } else {
        qx_runtime_v2::granite_dense_q4_bytes()
    };
    let fixed_plan_bytes = fixed_dense_bytes
        .saturating_add(runtime_plan.kv_bytes)
        .saturating_add(runtime_plan.arena_bytes);
    let expert_memory_budget = args
        .memory_budget
        .checked_sub(fixed_plan_bytes)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "memory budget {} cannot hold {} bytes of fixed dense/runtime state",
                args.memory_budget,
                fixed_plan_bytes
            )
        })?;
    let config_path = args.model.join("config.json");
    let config = serde_json::from_slice::<GraniteConfig>(
        &fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?,
    )
    .with_context(|| format!("parse {}", config_path.display()))?;
    config.validate_v0().map_err(anyhow::Error::msg)?;
    let checkpoint = Checkpoint::open(&args.model)?;
    let trace = args.trace.as_ref().map(QxTrace::read).transpose()?;
    if let Some(trace) = &trace {
        if trace.hidden_size != config.hidden_size.unwrap_or_default() {
            bail!(
                "trace hidden size {} does not match model",
                trace.hidden_size
            );
        }
    } else {
        let format = args.uniform_expert_bits.unwrap_or(QuantType::Q4);
        eprintln!(
            "warning: no --trace supplied; emitting uniform {format:?} experts without quality claims"
        );
    }
    if args.layerwise_q3 {
        if trace.is_none() {
            bail!("--layerwise-q3 requires --trace");
        }
        if args.uniform_expert_bits.is_some()
            || args.expert_format != ExpertFormat::QxQ4
            || !matches!(
                args.expert_layout,
                ExpertLayout::RowMajor | ExpertLayout::RowInline
            )
        {
            bail!(
                "--layerwise-q3 requires traced QX Q4 row-major experts without --uniform-expert-bits"
            );
        }
        if !(0.0..=1.0).contains(&args.layerwise_q3_threshold) {
            bail!("--layerwise-q3-threshold must be between 0 and 1");
        }
    }
    if args.tensorwise_q3 {
        if trace.is_none() {
            bail!("--tensorwise-q3 requires --trace");
        }
        if args.uniform_expert_bits.is_some()
            || args.expert_format != ExpertFormat::QxQ4
            || !matches!(
                args.expert_layout,
                ExpertLayout::RowMajor | ExpertLayout::RowInline
            )
        {
            bail!("--tensorwise-q3 requires traced QX Q4 row-major or row-inline experts");
        }
        if args.layerwise_q3 {
            bail!("--tensorwise-q3 cannot be combined with --layerwise-q3");
        }
    }
    if args.expertwise_q3 {
        if trace.is_none() {
            bail!("--expertwise-q3 requires --trace");
        }
        if args.uniform_expert_bits.is_some()
            || args.expert_format != ExpertFormat::QxQ4
            || args.layerwise_q3
            || args.tensorwise_q3
            || !matches!(
                args.expert_layout,
                ExpertLayout::RowMajor | ExpertLayout::RowInline | ExpertLayout::Superblock144
            )
        {
            bail!(
                "--expertwise-q3 requires traced QX Q4 row-major, row-inline, or superblock-144 experts and cannot be combined with layerwise/tensorwise Q3"
            );
        }
        if !(0.0..=1.0).contains(&args.expertwise_q3_threshold) {
            bail!("--expertwise-q3-threshold must be between 0 and 1");
        }
    }
    if matches!(
        args.expert_layout,
        ExpertLayout::RowInline2 | ExpertLayout::RowInline4 | ExpertLayout::RowInline8
    ) && (args.expert_format != ExpertFormat::QxQ4
        || args.uniform_expert_bits.is_some()
        || args.layerwise_q3
        || args.tensorwise_q3
        || args.expertwise_q3)
    {
        bail!("multi-row row-inline candidates require uniform exact group-32 Q4 experts");
    }
    if matches!(
        args.expert_layout,
        ExpertLayout::Transaction144 | ExpertLayout::Transaction160 | ExpertLayout::Superblock144
    ) && (args.expert_format != ExpertFormat::QxQ4
        || args.uniform_expert_bits.is_some()
        || args.layerwise_q3
        || args.tensorwise_q3
        || (args.expertwise_q3 && args.expert_layout != ExpertLayout::Superblock144)
        || args.group_size != 32)
    {
        bail!(
            "block-layout candidates require uniform exact group-32 Q4 experts and --group-size 32"
        );
    }
    if matches!(args.expert_layout, ExpertLayout::CrossExpertTile8)
        && (args.expert_format != ExpertFormat::QxQ4
            || args.uniform_expert_bits.is_some()
            || args.layerwise_q3
            || args.tensorwise_q3
            || args.expertwise_q3
            || args.group_size != 32)
    {
        bail!(
            "cross-expert tile candidates require uniform exact group-32 Q4 experts and --group-size 32"
        );
    }

    let grouped = trace.as_ref().map(QxTrace::samples).unwrap_or_default();
    // Each calibration token contributes TOP_K records per layer. Keep the
    // denominator per layer because llama.cpp may split a prompt into
    // different graph batches while tracing.
    let layer_token_counts = trace.as_ref().map(|value| {
        let mut counts = BTreeMap::<u16, usize>::new();
        for record in &value.records {
            *counts.entry(record.layer).or_default() += 1;
        }
        counts
            .into_iter()
            .map(|(layer, count)| (layer, count.div_ceil(TOP_K)))
            .collect::<BTreeMap<_, _>>()
    });
    let mut sensitivities = Vec::with_capacity(LAYERS * EXPERTS);
    let mut tensors = Vec::with_capacity(LAYERS * EXPERTS * 2);
    for layer in 0..LAYERS {
        let input_name = format!("model.layers.{layer}.block_sparse_moe.input_linear.weight");
        let output_name = format!("model.layers.{layer}.block_sparse_moe.output_linear.weight");
        let (input_shape, input_values) = checkpoint
            .tensor_f32(&input_name)
            .with_context(|| format!("load {input_name}"))?;
        let (output_shape, output_values) = checkpoint
            .tensor_f32(&output_name)
            .with_context(|| format!("load {output_name}"))?;
        validate_expert_tensor(&input_shape, &input_values, &input_name)?;
        validate_expert_tensor(&output_shape, &output_values, &output_name)?;
        for expert in 0..EXPERTS {
            let input_elements = input_shape[1] * input_shape[2];
            let output_elements = output_shape[1] * output_shape[2];
            let input = first_expert(&input_values, expert, input_elements);
            let output = first_expert(&output_values, expert, output_elements);
            let weights = ExpertWeights::new(
                &input_shape[1..],
                input.clone(),
                &output_shape[1..],
                output.clone(),
            )?;
            let key = (layer as u16, expert as u8);
            let samples = grouped.get(&key).map(Vec::as_slice).unwrap_or(&[]);
            let sensitivity = if trace.is_some() && !samples.is_empty() {
                let total_tokens = layer_token_counts
                    .as_ref()
                    .and_then(|counts| counts.get(&(layer as u16)).copied())
                    .unwrap_or(0);
                measure_sensitivity(
                    layer as u16,
                    expert as u8,
                    &weights,
                    samples,
                    total_tokens,
                    args.group_size,
                )
            } else {
                // Unmeasured experts are deliberately forced to Q4 below.
                ExpertSensitivity {
                    layer: layer as u16,
                    expert: expert as u8,
                    q2_error: f32::INFINITY,
                    q3_error: f32::INFINITY,
                    q4_error: 0.0,
                    q2_mse: f32::INFINITY,
                    q3_mse: f32::INFINITY,
                    q4_mse: 0.0,
                    q2_cosine: 0.0,
                    q3_cosine: 0.0,
                    q4_cosine: 1.0,
                    gate_up_q3_error: f32::INFINITY,
                    down_q3_error: f32::INFINITY,
                    gate_up_q4_error: 0.0,
                    down_q4_error: 0.0,
                    activation_frequency: 0.0,
                    mean_router_weight: 0.0,
                    gate_up_rows: weights.gate_up_rows,
                    gate_up_cols: weights.gate_up_cols,
                    down_rows: weights.down_rows,
                    down_cols: weights.down_cols,
                }
            };
            sensitivities.push(sensitivity);
            tensors.push((
                layer as u16,
                expert as u8,
                input,
                output,
                weights.gate_up_rows,
                weights.gate_up_cols,
                weights.down_rows,
                weights.down_cols,
            ));
        }
    }

    let q4_bytes = sensitivities
        .iter()
        .map(|item| item.bytes_for(QuantType::Q4, args.group_size))
        .sum::<usize>();
    let plan = if trace.is_some() {
        allocate_with_minimum_options(
            &sensitivities,
            expert_memory_budget,
            args.quality_loss,
            args.group_size,
            args.minimum_bits,
            !args.disable_q3,
        )
    } else {
        if expert_memory_budget < q4_bytes {
            bail!(
                "uniform Q4 experts need {q4_bytes} bytes after reserving {fixed_plan_bytes} bytes for dense/runtime state; provide --trace or a larger budget"
            );
        }
        allocate_with_minimum_options(
            &sensitivities
                .iter()
                .map(|item| ExpertSensitivity {
                    q2_error: 0.0,
                    q3_error: 0.0,
                    q4_error: 0.0,
                    ..item.clone()
                })
                .collect::<Vec<_>>(),
            expert_memory_budget,
            f32::NEG_INFINITY,
            args.group_size,
            args.minimum_bits,
            !args.disable_q3,
        )
    };
    if trace.is_some() && plan.total_bytes > expert_memory_budget {
        bail!("allocator exceeded memory budget");
    }
    let mut selected = plan
        .entries
        .iter()
        .map(|entry| ((entry.layer, entry.expert), entry.quant_type))
        .collect::<BTreeMap<_, _>>();
    if trace.is_some() {
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                if !grouped.contains_key(&(layer as u16, expert as u8)) {
                    selected.insert((layer as u16, expert as u8), QuantType::Q4);
                }
            }
        }
        let selected_bytes = sensitivities
            .iter()
            .map(|item| item.bytes_for(selected[&(item.layer, item.expert)], args.group_size))
            .sum::<usize>();
        if selected_bytes > expert_memory_budget {
            bail!(
                "conservative Q4 fallback for unmeasured experts needs {selected_bytes} bytes, over --memory-budget {}",
                expert_memory_budget
            );
        }
        let total_importance = sensitivities
            .iter()
            .map(ExpertSensitivity::importance)
            .sum::<f32>();
        let weighted_error = sensitivities
            .iter()
            .map(|item| {
                let importance = item.importance();
                if importance == 0.0 {
                    0.0
                } else {
                    importance * item.error_for(selected[&(item.layer, item.expert)])
                }
            })
            .sum::<f32>()
            / total_importance.max(f32::EPSILON);
        if weighted_error > args.quality_loss {
            bail!(
                "no allocation satisfies weighted --quality-loss {} (measured {}) within --memory-budget {}",
                args.quality_loss,
                weighted_error,
                expert_memory_budget
            );
        }
    }
    let layerwise_q3_layers = if args.layerwise_q3 {
        let mut layers = [false; LAYERS];
        for layer in 0..LAYERS {
            let items = sensitivities
                .iter()
                .filter(|item| item.layer == layer as u16)
                .collect::<Vec<_>>();
            let total_importance = items.iter().map(|item| item.importance()).sum::<f32>();
            let weighted_error = items
                .iter()
                .map(|item| item.importance() * item.q3_error)
                .sum::<f32>()
                / total_importance.max(f32::EPSILON);
            layers[layer] = items.len() == EXPERTS
                && items.iter().all(|item| item.q3_error.is_finite())
                && weighted_error <= args.layerwise_q3_threshold;
        }
        layers
    } else {
        [false; LAYERS]
    };
    let tensorwise_q3_layers = if args.tensorwise_q3 {
        if let Some(target_savings) = args.tensorwise_q3_savings {
            select_tensorwise_q3_for_savings(&sensitivities, args.group_size, target_savings)
        } else {
            let mut gate = [false; LAYERS];
            let mut down = [false; LAYERS];
            for layer in 0..LAYERS {
                let items = sensitivities
                    .iter()
                    .filter(|item| item.layer == layer as u16)
                    .collect::<Vec<_>>();
                let total_importance = items.iter().map(|item| item.importance()).sum::<f32>();
                let gate_error = items
                    .iter()
                    .map(|item| item.importance() * item.gate_up_q3_error)
                    .sum::<f32>()
                    / total_importance.max(f32::EPSILON);
                let down_error = items
                    .iter()
                    .map(|item| item.importance() * item.down_q3_error)
                    .sum::<f32>()
                    / total_importance.max(f32::EPSILON);
                let measured = items.len() == EXPERTS
                    && items.iter().all(|item| {
                        item.gate_up_q3_error.is_finite() && item.down_q3_error.is_finite()
                    });
                gate[layer] = measured && gate_error <= args.layerwise_q3_threshold;
                down[layer] = measured && down_error <= args.layerwise_q3_threshold;
            }
            (gate, down)
        }
    } else {
        ([false; LAYERS], [false; LAYERS])
    };
    let expertwise_q3_tensors = if args.expertwise_q3 {
        if let Some(target_savings) = args.expertwise_q3_savings {
            select_expertwise_q3_for_savings(&sensitivities, args.group_size, target_savings)
        } else {
            let threshold = args.expertwise_q3_threshold;
            sensitivities
                .iter()
                .map(|item| {
                    (
                        item.gate_up_q3_error.is_finite() && item.gate_up_q3_error <= threshold,
                        item.down_q3_error.is_finite() && item.down_q3_error <= threshold,
                    )
                })
                .collect::<Vec<_>>()
        }
    } else {
        vec![(false, false); sensitivities.len()]
    };
    if args.expertwise_q3 {
        let total_importance = sensitivities
            .iter()
            .map(ExpertSensitivity::importance)
            .sum::<f32>();
        let weighted_error = sensitivities
            .iter()
            .zip(&expertwise_q3_tensors)
            .map(|(item, &(gate_q3, down_q3))| {
                let gate = if gate_q3 { item.gate_up_q3_error } else { 0.0 };
                let down = if down_q3 { item.down_q3_error } else { 0.0 };
                item.importance() * (gate + down) * 0.5
            })
            .sum::<f32>()
            / total_importance.max(f32::EPSILON);
        if weighted_error > args.quality_loss {
            bail!(
                "expertwise Q3/Q4 allocation has weighted error {weighted_error}, above --quality-loss {}",
                args.quality_loss
            );
        }
        let selected_bytes = sensitivities
            .iter()
            .zip(&expertwise_q3_tensors)
            .map(|(item, &(gate_q3, down_q3))| {
                quantized_bytes(
                    item.gate_up_rows,
                    item.gate_up_cols,
                    if gate_q3 {
                        QuantType::Q3
                    } else {
                        QuantType::Q4
                    },
                    args.group_size,
                ) + quantized_bytes(
                    item.down_rows,
                    item.down_cols,
                    if down_q3 {
                        QuantType::Q3
                    } else {
                        QuantType::Q4
                    },
                    args.group_size,
                )
            })
            .sum::<usize>();
        if selected_bytes > expert_memory_budget {
            bail!(
                "expertwise Q3/Q4 allocation needs {selected_bytes} bytes after reserving {fixed_plan_bytes} bytes for dense/runtime state"
            );
        }
        eprintln!(
            "expertwise Q3/Q4 candidate: {} gate/up and {} down tensors selected for Q3; weighted error {weighted_error:.6}",
            expertwise_q3_tensors.iter().filter(|item| item.0).count(),
            expertwise_q3_tensors.iter().filter(|item| item.1).count(),
        );
    }
    if args.tensorwise_q3 {
        let total_importance = sensitivities
            .iter()
            .map(ExpertSensitivity::importance)
            .sum::<f32>();
        let weighted_error = sensitivities
            .iter()
            .map(|item| {
                let gate = if tensorwise_q3_layers.0[item.layer as usize] {
                    item.gate_up_q3_error
                } else {
                    0.0
                };
                let down = if tensorwise_q3_layers.1[item.layer as usize] {
                    item.down_q3_error
                } else {
                    0.0
                };
                item.importance() * (gate + down) * 0.5
            })
            .sum::<f32>()
            / total_importance.max(f32::EPSILON);
        if weighted_error > args.quality_loss {
            bail!(
                "tensorwise Q3/Q4 allocation has weighted error {weighted_error}, above --quality-loss {}",
                args.quality_loss
            );
        }
    }
    if args.layerwise_q3 {
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                selected.insert(
                    (layer as u16, expert as u8),
                    if layerwise_q3_layers[layer] {
                        QuantType::Q3
                    } else {
                        QuantType::Q4
                    },
                );
            }
        }
        let selected_bytes = sensitivities
            .iter()
            .map(|item| item.bytes_for(selected[&(item.layer, item.expert)], args.group_size))
            .sum::<usize>();
        let total_importance = sensitivities
            .iter()
            .map(ExpertSensitivity::importance)
            .sum::<f32>();
        let weighted_error = sensitivities
            .iter()
            .map(|item| item.importance() * item.error_for(selected[&(item.layer, item.expert)]))
            .sum::<f32>()
            / total_importance.max(f32::EPSILON);
        if selected_bytes > expert_memory_budget {
            bail!(
                "layerwise Q3/Q4 experts need {selected_bytes} bytes after reserving {fixed_plan_bytes} bytes for dense/runtime state"
            );
        }
        if weighted_error > args.quality_loss {
            bail!(
                "layerwise Q3/Q4 allocation has weighted error {weighted_error}, above --quality-loss {}",
                args.quality_loss
            );
        }
    }
    let mut compiled = Vec::with_capacity(tensors.len() * 2 + LAYERS * 5 + 1);
    let mut raw_tensors = Vec::with_capacity(LAYERS * 2 + 1);
    let mut actual_weight_bytes = 0usize;
    let packing_layout: PackingLayout = args.expert_layout.into();
    for (layer, expert, input, output, input_rows, input_cols, output_rows, output_cols) in tensors
    {
        let tensor_index = layer as usize * EXPERTS + expert as usize;
        let quant_type = if args.expert_format == ExpertFormat::Q4K {
            QuantType::Q4K
        } else if let Some(uniform) = args.uniform_expert_bits {
            uniform
        } else if args.layerwise_q3 && layerwise_q3_layers[layer as usize] {
            QuantType::Q3
        } else if args.tensorwise_q3 {
            // Gate/up and down are overridden independently below. Keep the
            // family-local fallback at Q4 when only its sibling selected Q3.
            QuantType::Q4
        } else if args.expertwise_q3 {
            // Expertwise mode is deliberately a Q3/Q4 experiment. Do not
            // let the general allocator silently introduce Q2 tensors that
            // the typed GPU-map path does not support.
            QuantType::Q4
        } else if trace.is_some() {
            if args.layerwise_q3 {
                QuantType::Q4
            } else {
                selected[&(layer, expert)]
            }
        } else {
            QuantType::Q4
        };
        let gate_quant_type = if args.expertwise_q3 && expertwise_q3_tensors[tensor_index].0 {
            QuantType::Q3
        } else if args.tensorwise_q3 && tensorwise_q3_layers.0[layer as usize] {
            QuantType::Q3
        } else {
            quant_type
        };
        let down_quant_type = if args.expertwise_q3 && expertwise_q3_tensors[tensor_index].1 {
            QuantType::Q3
        } else if args.tensorwise_q3 && tensorwise_q3_layers.1[layer as usize] {
            QuantType::Q3
        } else {
            quant_type
        };
        let gate_group_size = if gate_quant_type == QuantType::Q4K {
            256
        } else {
            args.group_size
        };
        let down_group_size = if down_quant_type == QuantType::Q4K {
            256
        } else {
            args.group_size
        };
        let gate_layout = if gate_quant_type == QuantType::Q4K {
            PackingLayout::RowMajor
        } else if args.expertwise_q3
            && gate_quant_type == QuantType::Q3
            && packing_layout == PackingLayout::Q4Superblock144
        {
            PackingLayout::Q3Superblock112
        } else if (args.layerwise_q3 || args.tensorwise_q3 || args.expertwise_q3)
            && gate_quant_type == QuantType::Q3
        {
            // Q3 has no row-inline transport; Q4 layers may still use the
            // proven row-inline representation in the same native slab.
            PackingLayout::RowMajor
        } else {
            packing_layout
        };
        let down_layout = if down_quant_type == QuantType::Q4K {
            PackingLayout::RowMajor
        } else if args.expertwise_q3
            && down_quant_type == QuantType::Q3
            && packing_layout == PackingLayout::Q4Superblock144
        {
            PackingLayout::Q3Superblock112
        } else if (args.layerwise_q3 || args.tensorwise_q3 || args.expertwise_q3)
            && down_quant_type == QuantType::Q3
        {
            PackingLayout::RowMajor
        } else if packing_layout
            .row_inline_packet_rows()
            .is_some_and(|rows| rows > 1)
        {
            // The first experiment is deliberately asymmetric: packetize
            // gate/up rows while keeping down on the proven row-inline-1
            // stream. This isolates the representation change to the
            // dominant gate/up traffic.
            PackingLayout::Q4RowInline
        } else {
            packing_layout
        };
        let gate_up = QuantizedTensor::quantize_for_layout(
            &input,
            input_rows,
            input_cols,
            gate_quant_type,
            gate_group_size,
            gate_layout,
        )
        .map_err(anyhow::Error::msg)?;
        let down = QuantizedTensor::quantize_for_layout(
            &output,
            output_rows,
            output_cols,
            down_quant_type,
            down_group_size,
            down_layout,
        )
        .map_err(anyhow::Error::msg)?;
        actual_weight_bytes += gate_up.serialized_bytes() + down.serialized_bytes();
        compiled.push(CompiledTensor {
            layer,
            expert,
            kind: TensorKind::GateUp,
            tensor: gate_up,
        });
        compiled.push(CompiledTensor {
            layer,
            expert,
            kind: TensorKind::Down,
            tensor: down,
        });
    }
    if !args.experts_only {
        for layer in 0..LAYERS {
            for (kind, suffix) in [
                (TensorKind::InputNorm, "input_layernorm.weight"),
                (
                    TensorKind::PostAttentionNorm,
                    "post_attention_layernorm.weight",
                ),
            ] {
                let name = format!("model.layers.{layer}.{suffix}");
                let (shape, values) = checkpoint
                    .tensor_f32(&name)
                    .with_context(|| format!("load {name}"))?;
                if shape != [qx_model::HIDDEN] {
                    bail!(
                        "tensor {name} has shape {shape:?}, expected [{}]",
                        qx_model::HIDDEN
                    );
                }
                actual_weight_bytes += values.len() * std::mem::size_of::<f32>();
                raw_tensors.push(RawTensor {
                    layer: layer as u16,
                    expert: u8::MAX,
                    kind,
                    rows: qx_model::HIDDEN,
                    cols: 1,
                    values,
                });
            }
            for (kind, suffix, rows, columns) in [
                (
                    TensorKind::QProj,
                    "self_attn.q_proj.weight",
                    qx_model::HIDDEN,
                    qx_model::HIDDEN,
                ),
                (
                    TensorKind::KProj,
                    "self_attn.k_proj.weight",
                    qx_model::KV_HEADS * qx_model::HEAD_DIM,
                    qx_model::HIDDEN,
                ),
                (
                    TensorKind::VProj,
                    "self_attn.v_proj.weight",
                    qx_model::KV_HEADS * qx_model::HEAD_DIM,
                    qx_model::HIDDEN,
                ),
                (
                    TensorKind::OProj,
                    "self_attn.o_proj.weight",
                    qx_model::HIDDEN,
                    qx_model::HIDDEN,
                ),
                (
                    TensorKind::Router,
                    "block_sparse_moe.router.layer.weight",
                    qx_model::EXPERTS,
                    qx_model::HIDDEN,
                ),
            ] {
                let name = format!("model.layers.{layer}.{suffix}");
                let (shape, values) = checkpoint
                    .tensor_f32(&name)
                    .with_context(|| format!("load {name}"))?;
                if shape != [rows, columns] {
                    bail!("tensor {name} has shape {shape:?}, expected [{rows}, {columns}]");
                }
                let tensor = QuantizedTensor::quantize_for_layout(
                    &values,
                    rows,
                    columns,
                    QuantType::Q4,
                    args.group_size,
                    args.dense_layout.into(),
                )
                .map_err(anyhow::Error::msg)?;
                actual_weight_bytes += tensor.bytes();
                compiled.push(CompiledTensor {
                    layer: layer as u16,
                    expert: u8::MAX,
                    kind,
                    tensor,
                });
            }
        }
        let (shape, values) = checkpoint
            .tensor_f32("model.embed_tokens.weight")
            .context("load model.embed_tokens.weight")?;
        if shape != [qx_model::VOCAB, qx_model::HIDDEN] {
            bail!(
                "tensor model.embed_tokens.weight has shape {shape:?}, expected [{}, {}]",
                qx_model::VOCAB,
                qx_model::HIDDEN
            );
        }
        let tensor = QuantizedTensor::quantize_for_layout(
            &values,
            qx_model::VOCAB,
            qx_model::HIDDEN,
            QuantType::Q4,
            args.group_size,
            args.dense_layout.into(),
        )
        .map_err(anyhow::Error::msg)?;
        actual_weight_bytes += tensor.bytes();
        compiled.push(CompiledTensor {
            layer: u16::MAX,
            expert: u8::MAX,
            kind: TensorKind::Embedding,
            tensor,
        });
        let (shape, values) = checkpoint
            .tensor_f32("model.norm.weight")
            .context("load model.norm.weight")?;
        if shape != [qx_model::HIDDEN] {
            bail!(
                "tensor model.norm.weight has shape {shape:?}, expected [{}]",
                qx_model::HIDDEN
            );
        }
        actual_weight_bytes += values.len() * std::mem::size_of::<f32>();
        raw_tensors.push(RawTensor {
            layer: u16::MAX,
            expert: u8::MAX,
            kind: TensorKind::FinalNorm,
            rows: qx_model::HIDDEN,
            cols: 1,
            values,
        });
    }
    let mut memory_plan = granite_memory_plan(actual_weight_bytes, args.context);
    memory_plan.kv_cache_bytes = runtime_plan.kv_bytes as u64;
    memory_plan.total_bytes =
        memory_plan.weights_bytes + runtime_plan.kv_bytes as u64 + runtime_plan.arena_bytes as u64;
    if memory_plan.total_bytes > args.memory_budget as u64 {
        bail!(
            "compiled plan predicts {} bytes at decode, exceeding the {}-byte memory budget",
            memory_plan.total_bytes,
            args.memory_budget
        );
    }
    let mut execution_plan = ExecutionPlanManifest::granite_m5_canonical(
        Granite3A400M::MODEL_ID,
        &args.target,
        args.context as u32,
        memory_plan,
    );
    execution_plan.plan_id.clone_from(&args.plan_id);
    execution_plan.memory.budget_bytes = args.memory_budget as u64;
    execution_plan.memory.weight_storage = args.weight_storage.into();
    execution_plan.topology.kv_precision = kv_precision;
    execution_plan.schedule.command_buffers_per_token = args.command_buffers_per_token;
    execution_plan.schedule.encoders_per_token = args.command_buffers_per_token;
    execution_plan.schedule.expected_dispatches_per_token = 390;
    execution_plan.schedule.rollout = if args.rollout_depth == 1 {
        qx_format::RolloutPlan::default()
    } else {
        qx_format::RolloutPlan {
            dependency: qx_format::RolloutDependency::GpuTokenChain,
            inflight_depth: args.rollout_depth,
            host_sync: qx_format::RolloutHostSync::ChunkBoundary,
        }
    };
    let last_dispatch = execution_plan
        .schedule
        .expected_dispatches_per_token
        .saturating_sub(1);
    execution_plan.arena = qx_format::ArenaPlan {
        arena_bytes: runtime_plan.arena_bytes as u64,
        kv_bytes: runtime_plan.kv_bytes as u64,
        no_decode_allocations: true,
        regions: runtime_plan
            .regions
            .iter()
            .map(|region| {
                let (live_from_dispatch, live_through_dispatch) = match region.id {
                    qx_runtime_v2::BufferId::HiddenA
                    | qx_runtime_v2::BufferId::HiddenB
                    | qx_runtime_v2::BufferId::TokenOutput => (0, last_dispatch),
                    qx_runtime_v2::BufferId::Logits => {
                        (last_dispatch.saturating_sub(2), last_dispatch)
                    }
                    _ => (1, last_dispatch.saturating_sub(3)),
                };
                qx_format::PlannedBufferRegion {
                    name: format!("{:?}", region.id),
                    offset: region.offset as u64,
                    length: region.length as u64,
                    live_from_dispatch,
                    live_through_dispatch,
                }
            })
            .collect(),
    };
    execution_plan.quality_budget.max_relative_ppl_delta = args.max_relative_ppl_delta;
    execution_plan.quality_budget.max_domain_relative_ppl_delta =
        args.max_domain_relative_ppl_delta;
    execution_plan.provenance.compiler_revision = option_env!("QX_GIT_REVISION")
        .unwrap_or("unknown")
        .to_owned();
    execution_plan
        .provenance
        .calibration_hash
        .clone_from(&args.calibration_hash);
    execution_plan
        .provenance
        .evaluation_hash
        .clone_from(&args.evaluation_hash);
    execution_plan
        .provenance
        .benchmark_contract_hash
        .clone_from(&args.benchmark_contract_hash);
    execution_plan.representations.gate_up = if args.expertwise_q3 {
        ExpertRepresentation::PerExpertTensorQ3Q4
    } else if args.tensorwise_q3 {
        tensorwise_representation_for(&compiled, TensorKind::GateUp)
    } else if args.layerwise_q3 {
        layerwise_representation_for(&compiled, TensorKind::GateUp)
    } else {
        representation_for(&compiled, TensorKind::GateUp)
    };
    execution_plan.representations.down = if args.expertwise_q3 {
        ExpertRepresentation::PerExpertTensorQ3Q4
    } else if args.tensorwise_q3 {
        tensorwise_representation_for(&compiled, TensorKind::Down)
    } else if args.layerwise_q3 {
        layerwise_representation_for(&compiled, TensorKind::Down)
    } else {
        representation_for(&compiled, TensorKind::Down)
    };
    execution_plan.representations.dense = if args.experts_only {
        ExpertRepresentation::MixedQ2Q3Q4
    } else {
        match args.dense_layout {
            DenseLayout::RowMajor => ExpertRepresentation::Group32Q4RowMajor,
            DenseLayout::Superblock144 => ExpertRepresentation::Group32Q4Superblock144,
        }
    };
    execution_plan.topology.moe = match args.moe_variant {
        MoeVariant::FourStage => qx_format::MoeTopology::FourStage,
        MoeVariant::WideExpert => qx_format::MoeTopology::FourStageWideExpert,
        MoeVariant::CrossExpertActivation => qx_format::MoeTopology::FourStageCrossExpertActivation,
    };
    execution_plan.moe_mapping = Some(match args.moe_mapping {
        MoeMappingArg::Qx => "qx".to_owned(),
        MoeMappingArg::LlamaLike => "llama-like".to_owned(),
    });
    execution_plan.moe_consumer = Some(match args.moe_consumer {
        MoeConsumerArg::Canonical => "canonical".to_owned(),
        MoeConsumerArg::UniquePacket => "unique-packet".to_owned(),
    });
    execution_plan.moe_down_consumer = Some(match args.moe_down_consumer {
        MoeConsumerArg::Canonical => "canonical".to_owned(),
        MoeConsumerArg::UniquePacket => "unique-packet".to_owned(),
    });
    execution_plan.verifier_moe_consumer = Some(match args.verifier_moe_consumer {
        VerifierMoeConsumerArg::UniquePacket => VerifierMoeConsumerTopology::UniquePacket,
        VerifierMoeConsumerArg::K4VectorPacket => VerifierMoeConsumerTopology::K4VectorPacket,
    });
    execution_plan.verifier_k4_split_accumulation = Some(args.verifier_k4_split_accumulation);
    execution_plan.verifier_k4_direct_activation = Some(args.verifier_k4_direct_activation);
    execution_plan.verifier_k4_rows16 = Some(args.verifier_k4_rows16);
    execution_plan.verifier_k4_unroll4 = Some(args.verifier_k4_unroll4);
    execution_plan.verifier_k4_unroll4_max_context = args.verifier_k4_unroll4_max_context;
    execution_plan.verifier_k4_down_unroll4 = Some(args.verifier_k4_down_unroll4);
    execution_plan.verifier_active_expert_dispatch = Some(args.verifier_active_expert_dispatch);
    execution_plan.expert_traversal = if args.expert_layout == ExpertLayout::CrossExpertTile8 {
        Some(qx_format::ExpertTraversalPlan {
            kind: qx_format::ExpertTraversalKind::CrossExpertTile,
            tile_rows: 8,
            expert_count: EXPERTS as u16,
            active_k: TOP_K as u8,
        })
    } else {
        None
    };
    write_artifact_with_raw_and_plan(
        &args.output,
        Granite3A400M::MODEL_ID,
        &args.target,
        args.context as u32,
        memory_plan,
        &compiled,
        &raw_tensors,
        &execution_plan,
    )
    .with_context(|| format!("write {}", args.output.display()))?;
    if let Some(report) = &args.report {
        write_sensitivity_report(report, &sensitivities, &selected, &plan)?;
    }
    let mut counts = BTreeMap::<QuantType, usize>::new();
    for tensor in &compiled {
        *counts.entry(tensor.tensor.quant_type).or_default() += 1;
    }
    println!("QX compilation complete");
    println!("model:              {}", Granite3A400M::MODEL_ID);
    println!("target:             {}", args.target);
    println!("plan id:            {}", execution_plan.plan_id);
    println!("artifact version:   {}", qx_format::VERSION);
    println!("compiled tensors:   {}", compiled.len());
    println!(
        "dense weights:      {}",
        if args.experts_only { "omitted" } else { "Q4" }
    );
    println!("uniform Q4 bytes:   {q4_bytes}");
    println!("compiled weights:   {actual_weight_bytes}");
    println!("memory plan total:  {}", memory_plan.total_bytes);
    println!(
        "precision tensors:  Q2={} Q3={} Q4={} Q4_K={}",
        counts.get(&QuantType::Q2).copied().unwrap_or(0),
        counts.get(&QuantType::Q3).copied().unwrap_or(0),
        counts.get(&QuantType::Q4).copied().unwrap_or(0),
        counts.get(&QuantType::Q4K).copied().unwrap_or(0)
    );
    println!("artifact:            {}", args.output.display());
    if let Some(report) = &args.report {
        println!("report:              {}", report.display());
    }
    Ok(())
}

fn select_expertwise_q3_for_savings(
    sensitivities: &[ExpertSensitivity],
    group_size: usize,
    target_savings: usize,
) -> Vec<(bool, bool)> {
    #[derive(Clone, Copy)]
    struct Candidate {
        sensitivity_index: usize,
        gate_up: bool,
        bytes_saved: usize,
        incremental_cost_per_byte: f32,
    }

    let mut candidates = Vec::with_capacity(sensitivities.len() * 2);
    for (sensitivity_index, item) in sensitivities.iter().enumerate() {
        for (gate_up, rows, cols, q3_error, q4_error) in [
            (
                true,
                item.gate_up_rows,
                item.gate_up_cols,
                item.gate_up_q3_error,
                item.gate_up_q4_error,
            ),
            (
                false,
                item.down_rows,
                item.down_cols,
                item.down_q3_error,
                item.down_q4_error,
            ),
        ] {
            if !q3_error.is_finite() || !q4_error.is_finite() {
                continue;
            }
            let q4_bytes = quantized_bytes(rows, cols, QuantType::Q4, group_size);
            let q3_bytes = quantized_bytes(rows, cols, QuantType::Q3, group_size);
            let bytes_saved = q4_bytes.saturating_sub(q3_bytes);
            if bytes_saved == 0 {
                continue;
            }
            let incremental_error = (q3_error - q4_error).max(0.0);
            candidates.push(Candidate {
                sensitivity_index,
                gate_up,
                bytes_saved,
                incremental_cost_per_byte: item.importance() * incremental_error
                    / bytes_saved as f32,
            });
        }
    }
    candidates.sort_by(|left, right| {
        left.incremental_cost_per_byte
            .total_cmp(&right.incremental_cost_per_byte)
            .then_with(|| right.bytes_saved.cmp(&left.bytes_saved))
            .then_with(|| left.sensitivity_index.cmp(&right.sensitivity_index))
            .then_with(|| right.gate_up.cmp(&left.gate_up))
    });

    let mut selected = vec![(false, false); sensitivities.len()];
    let mut saved = 0usize;
    for candidate in candidates {
        if saved >= target_savings {
            break;
        }
        if candidate.gate_up {
            selected[candidate.sensitivity_index].0 = true;
        } else {
            selected[candidate.sensitivity_index].1 = true;
        }
        saved += candidate.bytes_saved;
    }
    selected
}

fn select_tensorwise_q3_for_savings(
    sensitivities: &[ExpertSensitivity],
    group_size: usize,
    target_savings: usize,
) -> ([bool; LAYERS], [bool; LAYERS]) {
    #[derive(Clone, Copy)]
    struct Candidate {
        layer: usize,
        gate_up: bool,
        bytes_saved: usize,
        incremental_cost_per_byte: f32,
    }

    let mut candidates = Vec::with_capacity(LAYERS * 2);
    for layer in 0..LAYERS {
        let items = sensitivities
            .iter()
            .filter(|item| item.layer == layer as u16)
            .collect::<Vec<_>>();
        if items.len() != EXPERTS {
            continue;
        }
        for gate_up in [true, false] {
            let mut bytes_saved = 0usize;
            let mut incremental_cost = 0.0f32;
            let mut measured = true;
            for item in &items {
                let (rows, cols, q3_error, q4_error) = if gate_up {
                    (
                        item.gate_up_rows,
                        item.gate_up_cols,
                        item.gate_up_q3_error,
                        item.gate_up_q4_error,
                    )
                } else {
                    (
                        item.down_rows,
                        item.down_cols,
                        item.down_q3_error,
                        item.down_q4_error,
                    )
                };
                if !q3_error.is_finite() || !q4_error.is_finite() {
                    measured = false;
                    break;
                }
                bytes_saved += quantized_bytes(rows, cols, QuantType::Q4, group_size)
                    .saturating_sub(quantized_bytes(rows, cols, QuantType::Q3, group_size));
                incremental_cost += item.importance() * (q3_error - q4_error).max(0.0);
            }
            if measured && bytes_saved > 0 {
                candidates.push(Candidate {
                    layer,
                    gate_up,
                    bytes_saved,
                    incremental_cost_per_byte: incremental_cost / bytes_saved as f32,
                });
            }
        }
    }
    candidates.sort_by(|left, right| {
        left.incremental_cost_per_byte
            .total_cmp(&right.incremental_cost_per_byte)
            .then_with(|| right.bytes_saved.cmp(&left.bytes_saved))
            .then_with(|| left.layer.cmp(&right.layer))
            .then_with(|| right.gate_up.cmp(&left.gate_up))
    });

    let mut gate = [false; LAYERS];
    let mut down = [false; LAYERS];
    let mut saved = 0usize;
    for candidate in candidates {
        if saved >= target_savings {
            break;
        }
        if candidate.gate_up {
            gate[candidate.layer] = true;
        } else {
            down[candidate.layer] = true;
        }
        saved += candidate.bytes_saved;
    }
    (gate, down)
}

fn representation_for(tensors: &[CompiledTensor], kind: TensorKind) -> ExpertRepresentation {
    let matching = tensors
        .iter()
        .filter(|tensor| tensor.kind == kind)
        .collect::<Vec<_>>();
    if matching
        .iter()
        .all(|tensor| tensor.tensor.quant_type == QuantType::Q4K)
    {
        ExpertRepresentation::Q4K
    } else if matching.iter().all(|tensor| {
        tensor.tensor.quant_type == QuantType::Q4 && tensor.tensor.layout == PackingLayout::RowMajor
    }) {
        ExpertRepresentation::Group32Q4RowMajor
    } else if matching.iter().all(|tensor| {
        tensor.tensor.quant_type == QuantType::Q3 && tensor.tensor.layout == PackingLayout::RowMajor
    }) {
        ExpertRepresentation::DenseQ3
    } else if matching.iter().all(|tensor| {
        tensor.tensor.quant_type == QuantType::Q4
            && tensor.tensor.layout == PackingLayout::MetalTile32
    }) {
        ExpertRepresentation::Group32Q4MetalTile32
    } else if matching.iter().all(|tensor| {
        tensor.tensor.quant_type == QuantType::Q4
            && tensor.tensor.layout == PackingLayout::Q4RowInline
    }) {
        ExpertRepresentation::Group32Q4RowInline
    } else if kind == TensorKind::GateUp
        && matching.iter().all(|tensor| {
            tensor.tensor.quant_type == QuantType::Q4
                && tensor.tensor.layout == PackingLayout::Q4RowInlineRows2
        })
    {
        ExpertRepresentation::Group32Q4RowInlineRows2
    } else if kind == TensorKind::GateUp
        && matching.iter().all(|tensor| {
            tensor.tensor.quant_type == QuantType::Q4
                && tensor.tensor.layout == PackingLayout::Q4RowInlineRows4
        })
    {
        ExpertRepresentation::Group32Q4RowInlineRows4
    } else if kind == TensorKind::GateUp
        && matching.iter().all(|tensor| {
            tensor.tensor.quant_type == QuantType::Q4
                && tensor.tensor.layout == PackingLayout::Q4RowInlineRows8
        })
    {
        ExpertRepresentation::Group32Q4RowInlineRows8
    } else if matching.iter().all(|tensor| {
        tensor.tensor.quant_type == QuantType::Q4
            && tensor.tensor.layout == PackingLayout::Q4Transaction144
    }) {
        ExpertRepresentation::Group32Q4Transaction144
    } else if matching.iter().all(|tensor| {
        tensor.tensor.quant_type == QuantType::Q4
            && tensor.tensor.layout == PackingLayout::Q4Transaction160
    }) {
        ExpertRepresentation::Group32Q4Transaction160
    } else if matching.iter().all(|tensor| {
        tensor.tensor.quant_type == QuantType::Q4
            && tensor.tensor.layout == PackingLayout::Q4Superblock144
    }) {
        ExpertRepresentation::Group32Q4Superblock144
    } else if matching.iter().all(|tensor| {
        tensor.tensor.quant_type == QuantType::Q4
            && tensor.tensor.layout == PackingLayout::Q4CrossExpertTile8
    }) {
        // Keep the legacy logical representation enum exhaustive for the
        // existing runtime crates. The physical traversal contract is
        // carried by `expert_traversal` in the execution-plan manifest.
        ExpertRepresentation::Group32Q4RowMajor
    } else {
        ExpertRepresentation::MixedQ2Q3Q4
    }
}

fn layerwise_representation_for(
    tensors: &[CompiledTensor],
    kind: TensorKind,
) -> ExpertRepresentation {
    let matching = tensors
        .iter()
        .filter(|tensor| tensor.kind == kind)
        .collect::<Vec<_>>();
    let valid = matching.iter().all(|tensor| {
        matches!(tensor.tensor.quant_type, QuantType::Q3 | QuantType::Q4)
            && ((tensor.tensor.quant_type == QuantType::Q3
                && tensor.tensor.layout == PackingLayout::RowMajor)
                || (tensor.tensor.quant_type == QuantType::Q4
                    && matches!(
                        tensor.tensor.layout,
                        PackingLayout::RowMajor | PackingLayout::Q4RowInline
                    )))
    });
    let layerwise = (0..LAYERS).all(|layer| {
        let layer_tensors = matching
            .iter()
            .filter(|tensor| tensor.layer == layer as u16)
            .collect::<Vec<_>>();
        layer_tensors.len() == EXPERTS
            && layer_tensors
                .iter()
                .all(|tensor| tensor.tensor.quant_type == layer_tensors[0].tensor.quant_type)
    });
    if valid && layerwise {
        if matching
            .iter()
            .any(|tensor| tensor.tensor.layout == PackingLayout::Q4RowInline)
        {
            ExpertRepresentation::LayerwiseQ3Q4RowInline
        } else {
            ExpertRepresentation::LayerwiseQ3Q4
        }
    } else {
        ExpertRepresentation::MixedQ2Q3Q4
    }
}

fn tensorwise_representation_for(
    tensors: &[CompiledTensor],
    kind: TensorKind,
) -> ExpertRepresentation {
    let matching = tensors
        .iter()
        .filter(|tensor| tensor.kind == kind)
        .collect::<Vec<_>>();
    let valid = matching.iter().all(|tensor| {
        matches!(tensor.tensor.quant_type, QuantType::Q3 | QuantType::Q4)
            && ((tensor.tensor.quant_type == QuantType::Q3
                && tensor.tensor.layout == PackingLayout::RowMajor)
                || (tensor.tensor.quant_type == QuantType::Q4
                    && matches!(
                        tensor.tensor.layout,
                        PackingLayout::RowMajor | PackingLayout::Q4RowInline
                    )))
    });
    let uniform_per_layer = (0..LAYERS).all(|layer| {
        let layer_tensors = matching
            .iter()
            .filter(|tensor| tensor.layer == layer as u16)
            .collect::<Vec<_>>();
        layer_tensors.len() == EXPERTS
            && layer_tensors
                .iter()
                .all(|tensor| tensor.tensor.quant_type == layer_tensors[0].tensor.quant_type)
    });
    if valid && uniform_per_layer {
        ExpertRepresentation::LayerwiseTensorQ3Q4
    } else {
        ExpertRepresentation::MixedQ2Q3Q4
    }
}

fn write_sensitivity_report(
    path: &PathBuf,
    sensitivities: &[ExpertSensitivity],
    selected: &BTreeMap<(u16, u8), QuantType>,
    plan: &qx_compiler::AllocationPlan,
) -> Result<()> {
    let mut csv = "layer,expert,selected,bits,bytes,importance,frequency,mean_router_weight,"
        .to_owned()
        + "q2_relative_l2,q3_relative_l2,q4_relative_l2,q2_mse,q3_mse,q4_mse,"
        + "q2_cosine,q3_cosine,q4_cosine,gate_up_q3_relative_l2,"
        + "down_q3_relative_l2,gate_up_q4_relative_l2,down_q4_relative_l2,"
        + "gate_up_q3_bytes_saved,down_q3_bytes_saved,gate_up_route_value,down_route_value\n";
    for item in sensitivities {
        let quant_type = selected[&(item.layer, item.expert)];
        let bytes = item.bytes_for(quant_type, plan.group_size);
        let gate_up_bytes_saved = quantized_bytes(
            item.gate_up_rows,
            item.gate_up_cols,
            QuantType::Q4,
            plan.group_size,
        )
        .saturating_sub(quantized_bytes(
            item.gate_up_rows,
            item.gate_up_cols,
            QuantType::Q3,
            plan.group_size,
        ));
        let down_bytes_saved = quantized_bytes(
            item.down_rows,
            item.down_cols,
            QuantType::Q4,
            plan.group_size,
        )
        .saturating_sub(quantized_bytes(
            item.down_rows,
            item.down_cols,
            QuantType::Q3,
            plan.group_size,
        ));
        let fields = [
            item.layer.to_string(),
            item.expert.to_string(),
            quant_type.to_string(),
            quant_type.bits().to_string(),
            bytes.to_string(),
            format!("{:.8}", item.importance()),
            format!("{:.8}", item.activation_frequency),
            format!("{:.8}", item.mean_router_weight),
            format!("{:.8}", item.q2_error),
            format!("{:.8}", item.q3_error),
            format!("{:.8}", item.q4_error),
            format!("{:.8}", item.q2_mse),
            format!("{:.8}", item.q3_mse),
            format!("{:.8}", item.q4_mse),
            format!("{:.8}", item.q2_cosine),
            format!("{:.8}", item.q3_cosine),
            format!("{:.8}", item.q4_cosine),
            format!("{:.8}", item.gate_up_q3_error),
            format!("{:.8}", item.down_q3_error),
            format!("{:.8}", item.gate_up_q4_error),
            format!("{:.8}", item.down_q4_error),
            gate_up_bytes_saved.to_string(),
            down_bytes_saved.to_string(),
            format!("{:.8}", item.importance() * gate_up_bytes_saved as f32),
            format!("{:.8}", item.importance() * down_bytes_saved as f32),
        ];
        writeln!(csv, "{}", fields.join(","))?;
    }
    fs::write(path, csv).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn validate_expert_tensor(shape: &[usize], values: &[f32], name: &str) -> Result<()> {
    if shape.len() != 3 || shape[0] != EXPERTS || values.len() != shape.iter().product::<usize>() {
        bail!("{name} must have shape [32, rows, cols], got {shape:?}");
    }
    Ok(())
}

fn first_expert(values: &[f32], expert: usize, elements: usize) -> Vec<f32> {
    values[expert * elements..(expert + 1) * elements].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sensitivity(layer: u16, expert: u8, importance: f32) -> ExpertSensitivity {
        ExpertSensitivity {
            layer,
            expert,
            q2_error: 1.0,
            q3_error: 0.2,
            q4_error: 0.1,
            q2_mse: 1.0,
            q3_mse: 0.2,
            q4_mse: 0.1,
            q2_cosine: 0.0,
            q3_cosine: 0.9,
            q4_cosine: 0.99,
            gate_up_q3_error: 0.2,
            down_q3_error: 0.3,
            gate_up_q4_error: 0.1,
            down_q4_error: 0.1,
            activation_frequency: importance,
            mean_router_weight: 1.0,
            gate_up_rows: 1,
            gate_up_cols: 32,
            down_rows: 1,
            down_cols: 32,
        }
    }

    #[test]
    fn expertwise_savings_prefers_lower_incremental_cost() {
        let sensitivities = vec![sensitivity(0, 0, 0.01), sensitivity(0, 1, 1.0)];
        let one_tensor_saving =
            quantized_bytes(1, 32, QuantType::Q4, 32) - quantized_bytes(1, 32, QuantType::Q3, 32);
        let selected = select_expertwise_q3_for_savings(&sensitivities, 32, one_tensor_saving);
        assert_eq!(selected, vec![(true, false), (false, false)]);
    }

    #[test]
    fn tensorwise_savings_preserves_family_granularity() {
        let sensitivities = (0..EXPERTS)
            .map(|expert| sensitivity(0, expert as u8, 0.01))
            .collect::<Vec<_>>();
        let family_saving = EXPERTS
            * (quantized_bytes(1, 32, QuantType::Q4, 32)
                - quantized_bytes(1, 32, QuantType::Q3, 32));
        let (gate, down) = select_tensorwise_q3_for_savings(&sensitivities, 32, family_saving);
        assert!(gate[0]);
        assert!(!down[0]);
        assert_eq!(gate.iter().filter(|&&selected| selected).count(), 1);
    }
}
