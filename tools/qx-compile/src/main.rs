use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use qx_compiler::{
    ExpertSensitivity, ExpertWeights, QxTrace, allocate_with_minimum_options, granite_memory_plan,
    measure_sensitivity,
};
use qx_format::{
    CompiledTensor, ExecutionPlanManifest, ExpertRepresentation, RawTensor, TensorKind,
    WeightStorage, write_artifact_with_raw_and_plan,
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
    #[arg(long, default_value_t = 32)]
    group_size: usize,
    /// Expert-only scientific control; dense tensors remain QX group-32 Q4.
    #[arg(long, value_enum, default_value_t = ExpertFormat::QxQ4)]
    expert_format: ExpertFormat,
    /// Physical expert layout candidate. Whole-model evidence currently makes
    /// row-major the Granite/M5 control even though Tile32 wins in isolation.
    #[arg(long, value_enum, default_value_t = ExpertLayout::RowMajor)]
    expert_layout: ExpertLayout,
    /// Four-stage MoE execution candidate. `wide-expert` changes only the
    /// threadgroup mapping; it is never selected without complete-model
    /// evidence by qx-autotune.
    #[arg(long, value_enum, default_value_t = MoeVariant::FourStage)]
    moe_variant: MoeVariant,
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
    #[arg(long, value_enum, default_value_t = WeightStorageArg::CopiedShared)]
    weight_storage: WeightStorageArg,
    /// GPU-ordered command buffers per decode token. No intermediate CPU wait
    /// is permitted; layer boundaries are divided evenly between buffers.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..=24))]
    command_buffers_per_token: u16,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MoeVariant {
    FourStage,
    WideExpert,
}

impl From<ExpertLayout> for PackingLayout {
    fn from(value: ExpertLayout) -> Self {
        match value {
            ExpertLayout::RowMajor => Self::RowMajor,
            ExpertLayout::Tile32 => Self::MetalTile32,
            ExpertLayout::RowInline => Self::Q4RowInline,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum WeightStorageArg {
    CopiedShared,
    MmapNoCopy,
    Hybrid,
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
    let runtime_plan =
        qx_runtime_v2::ExecutionPlan::granite(args.context).map_err(anyhow::Error::msg)?;
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
        eprintln!(
            "warning: no --trace supplied; emitting uniform {:?} experts without quality claims",
            args.expert_format
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
    let mut compiled = Vec::with_capacity(tensors.len() * 2 + LAYERS * 5 + 1);
    let mut raw_tensors = Vec::with_capacity(LAYERS * 2 + 1);
    let mut actual_weight_bytes = 0usize;
    let packing_layout: PackingLayout = args.expert_layout.into();
    for (layer, expert, input, output, input_rows, input_cols, output_rows, output_cols) in tensors
    {
        let quant_type = if args.expert_format == ExpertFormat::Q4K {
            QuantType::Q4K
        } else if trace.is_some() {
            selected[&(layer, expert)]
        } else {
            QuantType::Q4
        };
        let expert_group_size = if quant_type == QuantType::Q4K {
            256
        } else {
            args.group_size
        };
        let expert_layout = if quant_type == QuantType::Q4K {
            PackingLayout::RowMajor
        } else {
            packing_layout
        };
        let gate_up = QuantizedTensor::quantize_for_layout(
            &input,
            input_rows,
            input_cols,
            quant_type,
            expert_group_size,
            expert_layout,
        )
        .map_err(anyhow::Error::msg)?;
        let down = QuantizedTensor::quantize_for_layout(
            &output,
            output_rows,
            output_cols,
            quant_type,
            expert_group_size,
            expert_layout,
        )
        .map_err(anyhow::Error::msg)?;
        actual_weight_bytes += gate_up.bytes() + down.bytes();
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
                let tensor = QuantizedTensor::quantize(
                    &values,
                    rows,
                    columns,
                    QuantType::Q4,
                    args.group_size,
                );
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
        let tensor = QuantizedTensor::quantize(
            &values,
            qx_model::VOCAB,
            qx_model::HIDDEN,
            QuantType::Q4,
            args.group_size,
        );
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
    execution_plan.schedule.command_buffers_per_token = args.command_buffers_per_token;
    execution_plan.schedule.encoders_per_token = args.command_buffers_per_token;
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
    execution_plan.representations.gate_up = representation_for(&compiled, TensorKind::GateUp);
    execution_plan.representations.down = representation_for(&compiled, TensorKind::Down);
    execution_plan.representations.dense = if args.experts_only {
        ExpertRepresentation::MixedQ2Q3Q4
    } else {
        ExpertRepresentation::Group32Q4RowMajor
    };
    execution_plan.topology.moe = match args.moe_variant {
        MoeVariant::FourStage => qx_format::MoeTopology::FourStage,
        MoeVariant::WideExpert => qx_format::MoeTopology::FourStageWideExpert,
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
        tensor.tensor.quant_type == QuantType::Q4
            && tensor.tensor.layout == PackingLayout::MetalTile32
    }) {
        ExpertRepresentation::Group32Q4MetalTile32
    } else if matching.iter().all(|tensor| {
        tensor.tensor.quant_type == QuantType::Q4
            && tensor.tensor.layout == PackingLayout::Q4RowInline
    }) {
        ExpertRepresentation::Group32Q4RowInline
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
        + "q2_cosine,q3_cosine,q4_cosine\n";
    for item in sensitivities {
        let quant_type = selected[&(item.layer, item.expert)];
        let bytes = item.bytes_for(quant_type, plan.group_size);
        writeln!(
            csv,
            "{},{},{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
            item.layer,
            item.expert,
            quant_type,
            quant_type.bits(),
            bytes,
            item.importance(),
            item.activation_frequency,
            item.mean_router_weight,
            item.q2_error,
            item.q3_error,
            item.q4_error,
            item.q2_mse,
            item.q3_mse,
            item.q4_mse,
            item.q2_cosine,
            item.q3_cosine,
            item.q4_cosine,
        )?;
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
