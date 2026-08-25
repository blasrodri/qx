use anyhow::{Context, Result, bail};
use clap::Parser;
use qx_format::{
    AttentionTopology, CompiledTensor, ExecutionPlanManifest, ExpertRepresentation, KvPrecision,
    MemoryPlan, MoeTopology, QkvTopology, RawTensor, RepresentationPlan, RmsTopology,
    RopeKvTopology, SparseExpertPolicy, TensorKind, WeightStorage,
    write_artifact_with_raw_and_plan,
};
use qx_model::{ModelShape, QWEN3_30B_A3B_MODEL_ID, Qwen3MoeConfig};
use qx_quant::{PackingLayout, QuantType, QuantizedTensor};
use qx_safetensors::{Checkpoint, TensorInfo};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "qx-compile-qwen",
    about = "Compile the pinned Qwen3-30B-A3B BF16 checkpoint into a provisional QX artifact"
)]
struct Args {
    /// Pinned Hugging Face BF16 checkpoint directory.
    model: PathBuf,
    /// Successful qx-inspect release audit for the exact source directory.
    #[arg(long)]
    source_audit: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 4096)]
    context: usize,
    #[arg(long, value_parser = parse_expert_bits, default_value = "q2")]
    expert_bits: QuantType,
    #[arg(long, default_value = "qwen3-a19pro-provisional")]
    plan_id: String,
}

#[derive(Debug, Deserialize)]
struct SourceAudit {
    ready_for_qx_compilation: bool,
    model_id: String,
    model_revision: Option<String>,
    source_files_verified: bool,
    shape: ModelShape,
}

#[derive(Default)]
struct PendingGateUp {
    gate: BTreeMap<(u16, u8), QuantizedTensor>,
    up: BTreeMap<(u16, u8), QuantizedTensor>,
}

fn parse_expert_bits(value: &str) -> Result<QuantType, String> {
    match QuantType::parse(value) {
        Some(bits @ (QuantType::Q2 | QuantType::Q3 | QuantType::Q4)) => Ok(bits),
        _ => Err("expected q2, q3, or q4".to_owned()),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.context == 0 || args.context > 4096 {
        bail!("the first mobile artifact requires context in 1..=4096");
    }
    let config_path = args.model.join("config.json");
    let config: Qwen3MoeConfig = serde_json::from_slice(
        &fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?,
    )?;
    let shape = config.validate_30b_a3b().map_err(anyhow::Error::msg)?;
    validate_source_audit(&args.source_audit, &shape)?;

    println!("compiling {QWEN3_30B_A3B_MODEL_ID}");
    println!("expert representation: {}", args.expert_bits);
    println!("context: {}", args.context);

    let checkpoint = Checkpoint::open(&args.model)?;
    let mut compiled = Vec::<CompiledTensor>::with_capacity(18_700);
    let mut raw = Vec::<RawTensor>::with_capacity(256);
    let mut pending = PendingGateUp::default();
    let mut visited = 0usize;
    checkpoint.visit_tensors_f32(|info, values| {
        compile_tensor(
            info,
            values,
            &shape,
            args.expert_bits,
            &mut compiled,
            &mut raw,
            &mut pending,
        )?;
        visited += 1;
        if visited.is_multiple_of(512) {
            println!("source tensors: {visited}/18867");
        }
        Ok(())
    })?;
    if !pending.gate.is_empty() || !pending.up.is_empty() {
        bail!(
            "unpaired gate/up projections remain after source traversal: {}/{}",
            pending.gate.len(),
            pending.up.len()
        );
    }
    if visited != 18_867 {
        bail!("expected 18867 source tensors, visited {visited}");
    }

    let weight_bytes = compiled
        .iter()
        .map(|tensor| tensor.tensor.bytes() as u64)
        .sum::<u64>()
        + raw
            .iter()
            .map(|tensor| (tensor.values.len() * 4) as u64)
            .sum::<u64>();
    let memory_plan = qwen_memory_plan(weight_bytes, args.context, &shape);
    let mut execution_plan = ExecutionPlanManifest::granite_m5_canonical(
        QWEN3_30B_A3B_MODEL_ID,
        "apple-a19-pro",
        args.context as u32,
        memory_plan,
    );
    execution_plan.plan_id = args.plan_id;
    execution_plan.model_shape = Some(shape.clone());
    execution_plan.target.gpu_family = "apple-a19-pro".to_owned();
    execution_plan.target.physical_memory_bytes = 12 * 1024 * 1024 * 1024;
    execution_plan.memory.weight_storage = WeightStorage::MmapNoCopy;
    let expert_representation = match args.expert_bits {
        QuantType::Q2 => ExpertRepresentation::Group32Q2RowMajor,
        QuantType::Q3 => ExpertRepresentation::Group32Q3RowMajor,
        QuantType::Q4 => ExpertRepresentation::Group32Q4RowMajor,
        _ => unreachable!(),
    };
    execution_plan.representations = RepresentationPlan {
        gate_up: expert_representation,
        down: expert_representation,
        dense: ExpertRepresentation::Group32Q4RowMajor,
    };
    execution_plan.topology.qkv = QkvTopology::Split;
    execution_plan.topology.input_rms = RmsTopology::Materialized;
    execution_plan.topology.post_attention_rms = RmsTopology::Materialized;
    execution_plan.topology.rope_kv = RopeKvTopology::Canonical;
    execution_plan.topology.attention = AttentionTopology::HierarchicalGqa;
    execution_plan.topology.moe = MoeTopology::FourStage;
    execution_plan.topology.kv_precision = KvPrecision::Fp16;
    execution_plan.sparse_experts = SparseExpertPolicy::Top8;
    // The phone backend must replace this provisional schedule after device
    // measurement; zero dispatches prevents this source artifact from being
    // mistaken for a performance-promoted execution plan.
    execution_plan.schedule.expected_dispatches_per_token = 0;
    execution_plan.provenance.compiler_revision = option_env!("GIT_COMMIT")
        .unwrap_or("working-tree")
        .to_owned();

    println!("quantized tensors: {}", compiled.len());
    println!("raw vectors: {}", raw.len());
    println!("logical weight bytes: {weight_bytes}");
    println!(
        "predicted logical bytes at context {}: {}",
        args.context, memory_plan.total_bytes
    );
    println!("writing {}", args.output.display());
    write_artifact_with_raw_and_plan(
        &args.output,
        QWEN3_30B_A3B_MODEL_ID,
        "apple-a19-pro",
        args.context as u32,
        memory_plan,
        &compiled,
        &raw,
        &execution_plan,
    )?;
    println!("artifact ready: {}", args.output.display());
    Ok(())
}

fn validate_source_audit(path: &std::path::Path, shape: &ModelShape) -> Result<()> {
    let audit: SourceAudit = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    if !audit.ready_for_qx_compilation || !audit.source_files_verified {
        bail!("source audit is not release-ready and hash-verified");
    }
    if audit.model_id != QWEN3_30B_A3B_MODEL_ID || audit.shape != *shape {
        bail!("source audit model identity or topology does not match checkpoint");
    }
    if audit.model_revision.as_deref() != Some("ad44e777bcd18fa416d9da3bd8f70d33ebb85d39") {
        bail!("source audit does not pin the promoted upstream revision");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compile_tensor(
    info: &TensorInfo,
    values: &[f32],
    shape: &ModelShape,
    expert_bits: QuantType,
    compiled: &mut Vec<CompiledTensor>,
    raw: &mut Vec<RawTensor>,
    pending: &mut PendingGateUp,
) -> Result<()> {
    if let Some((layer, expert, projection)) = parse_expert_name(&info.name) {
        let expected = match projection {
            "gate_proj" | "up_proj" => vec![shape.expert_intermediate, shape.hidden_size],
            "down_proj" => vec![shape.hidden_size, shape.expert_intermediate],
            _ => bail!("unknown expert projection {projection}"),
        };
        require_shape(info, &expected)?;
        let tensor = QuantizedTensor::quantize(values, expected[0], expected[1], expert_bits, 32);
        let key = (layer, expert);
        match projection {
            "gate_proj" => {
                pending.gate.insert(key, tensor);
                finish_gate_up(key, compiled, pending)?;
            }
            "up_proj" => {
                pending.up.insert(key, tensor);
                finish_gate_up(key, compiled, pending)?;
            }
            "down_proj" => compiled.push(CompiledTensor {
                layer,
                expert,
                kind: TensorKind::Down,
                tensor,
            }),
            _ => unreachable!(),
        }
        return Ok(());
    }

    if info.name == "model.embed_tokens.weight" {
        return push_quantized(info, values, u16::MAX, TensorKind::Embedding, compiled);
    }
    if info.name == "lm_head.weight" {
        return push_quantized(info, values, u16::MAX, TensorKind::LmHead, compiled);
    }
    if info.name == "model.norm.weight" {
        return push_raw(info, values, u16::MAX, TensorKind::FinalNorm, raw);
    }
    let (layer, suffix) = parse_layer_name(&info.name)?;
    match suffix {
        "input_layernorm.weight" => push_raw(info, values, layer, TensorKind::InputNorm, raw),
        "post_attention_layernorm.weight" => {
            push_raw(info, values, layer, TensorKind::PostAttentionNorm, raw)
        }
        "self_attn.q_norm.weight" => push_raw(info, values, layer, TensorKind::QueryNorm, raw),
        "self_attn.k_norm.weight" => push_raw(info, values, layer, TensorKind::KeyNorm, raw),
        "self_attn.q_proj.weight" => {
            push_quantized(info, values, layer, TensorKind::QProj, compiled)
        }
        "self_attn.k_proj.weight" => {
            push_quantized(info, values, layer, TensorKind::KProj, compiled)
        }
        "self_attn.v_proj.weight" => {
            push_quantized(info, values, layer, TensorKind::VProj, compiled)
        }
        "self_attn.o_proj.weight" => {
            push_quantized(info, values, layer, TensorKind::OProj, compiled)
        }
        "mlp.gate.weight" => push_quantized(info, values, layer, TensorKind::Router, compiled),
        _ => bail!("unsupported Qwen tensor {}", info.name),
    }
}

fn finish_gate_up(
    key: (u16, u8),
    compiled: &mut Vec<CompiledTensor>,
    pending: &mut PendingGateUp,
) -> Result<()> {
    let (Some(gate), Some(up)) = (pending.gate.remove(&key), pending.up.remove(&key)) else {
        return Ok(());
    };
    if gate.quant_type != up.quant_type
        || gate.layout != up.layout
        || gate.cols != up.cols
        || gate.group_size != up.group_size
    {
        bail!(
            "gate/up quantization contract differs at {}:{}",
            key.0,
            key.1
        );
    }
    let mut scales = gate.scales;
    scales.extend(up.scales);
    let mut data = gate.data;
    data.extend(up.data);
    compiled.push(CompiledTensor {
        layer: key.0,
        expert: key.1,
        kind: TensorKind::GateUp,
        tensor: QuantizedTensor {
            quant_type: gate.quant_type,
            layout: PackingLayout::RowMajor,
            rows: gate.rows + up.rows,
            cols: gate.cols,
            group_size: gate.group_size,
            scales,
            data,
        },
    });
    Ok(())
}

fn push_quantized(
    info: &TensorInfo,
    values: &[f32],
    layer: u16,
    kind: TensorKind,
    output: &mut Vec<CompiledTensor>,
) -> Result<()> {
    if info.shape.len() != 2 {
        bail!("{} is not a matrix", info.name);
    }
    output.push(CompiledTensor {
        layer,
        expert: u8::MAX,
        kind,
        tensor: QuantizedTensor::quantize(values, info.shape[0], info.shape[1], QuantType::Q4, 32),
    });
    Ok(())
}

fn push_raw(
    info: &TensorInfo,
    values: &[f32],
    layer: u16,
    kind: TensorKind,
    output: &mut Vec<RawTensor>,
) -> Result<()> {
    if info.shape.len() != 1 {
        bail!("{} is not a vector", info.name);
    }
    output.push(RawTensor {
        layer,
        expert: u8::MAX,
        kind,
        rows: 1,
        cols: info.shape[0],
        values: values.to_vec(),
    });
    Ok(())
}

fn require_shape(info: &TensorInfo, expected: &[usize]) -> Result<()> {
    if info.shape != expected {
        bail!(
            "{} has shape {:?}, expected {:?}",
            info.name,
            info.shape,
            expected
        );
    }
    Ok(())
}

fn parse_expert_name(name: &str) -> Option<(u16, u8, &str)> {
    let parts = name.split('.').collect::<Vec<_>>();
    if parts.len() == 8
        && parts[0] == "model"
        && parts[1] == "layers"
        && parts[3] == "mlp"
        && parts[4] == "experts"
        && parts[7] == "weight"
    {
        Some((parts[2].parse().ok()?, parts[5].parse().ok()?, parts[6]))
    } else {
        None
    }
}

fn parse_layer_name(name: &str) -> Result<(u16, &str)> {
    let prefix = "model.layers.";
    let rest = name
        .strip_prefix(prefix)
        .with_context(|| format!("unsupported global tensor {name}"))?;
    let (layer, suffix) = rest
        .split_once('.')
        .with_context(|| format!("malformed layer tensor {name}"))?;
    Ok((layer.parse().context("invalid layer index")?, suffix))
}

fn qwen_memory_plan(weight_bytes: u64, context: usize, shape: &ModelShape) -> MemoryPlan {
    let kv_cache_bytes = (shape.kv_elements_per_token() * context * 2) as u64;
    let hidden_bytes = (shape.hidden_size * 4) as u64;
    let attention_scratch_bytes =
        ((shape.attention_heads * context + shape.attention_heads * shape.head_dim) * 4) as u64;
    let moe_scratch_bytes = (shape.active_experts * shape.expert_intermediate * 4 * 3) as u64;
    let logits_bytes = (shape.vocabulary * 4) as u64;
    MemoryPlan {
        weights_bytes: weight_bytes,
        kv_cache_bytes,
        hidden_a_bytes: hidden_bytes,
        hidden_b_bytes: hidden_bytes,
        attention_scratch_bytes,
        moe_scratch_bytes,
        logits_bytes,
        total_bytes: weight_bytes
            + kv_cache_bytes
            + hidden_bytes * 2
            + attention_scratch_bytes
            + moe_scratch_bytes
            + logits_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_qwen_expert_and_layer_names() {
        assert_eq!(
            parse_expert_name("model.layers.47.mlp.experts.127.down_proj.weight"),
            Some((47, 127, "down_proj"))
        );
        assert_eq!(
            parse_layer_name("model.layers.3.self_attn.q_proj.weight").unwrap(),
            (3, "self_attn.q_proj.weight")
        );
    }

    #[test]
    fn gate_up_quantized_rows_compose_without_requantization() {
        let gate = QuantizedTensor::quantize(&[1.0; 64], 2, 32, QuantType::Q2, 32);
        let up = QuantizedTensor::quantize(&[2.0; 64], 2, 32, QuantType::Q2, 32);
        let mut pending = PendingGateUp::default();
        pending.gate.insert((0, 0), gate);
        pending.up.insert((0, 0), up);
        let mut output = Vec::new();
        finish_gate_up((0, 0), &mut output, &mut pending).unwrap();
        assert_eq!(output[0].tensor.rows, 4);
        assert_eq!(output[0].tensor.cols, 32);
        assert_eq!(output[0].tensor.dequantize()[0], 1.0);
        assert_eq!(output[0].tensor.dequantize()[64], 2.0);
    }
}
