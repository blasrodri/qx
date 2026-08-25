use anyhow::{Context, Result, bail};
use clap::Parser;
use qx_model::{
    Granite3A400M, ModelArchitecture, ModelShape, QWEN3_30B_A3B_MODEL_ID, Qwen3MoeConfig,
};
use qx_quant::{QuantType, QuantizedTensor};
use qx_safetensors::{Checkpoint, TensorInfo};
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "qx-inspect",
    about = "Inspect and release-audit a supported Hugging Face checkpoint"
)]
struct Args {
    /// Hugging Face model directory containing config.json and safetensors.
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    layer: Option<usize>,
    #[arg(long)]
    expert: Option<usize>,
    #[arg(long, value_parser = parse_quant_type)]
    quantize: Option<QuantType>,
    /// Write a complete tensor/shape/capacity preflight report. This maps each
    /// safetensor shard once but does not decode its full payload.
    #[arg(long)]
    audit_output: Option<PathBuf>,
    /// Immutable upstream revision recorded in the audit report.
    #[arg(long)]
    model_revision: Option<String>,
}

#[derive(Debug)]
struct InspectedModel {
    model_id: &'static str,
    shape: ModelShape,
}

#[derive(Debug, Serialize)]
struct ShapeMismatch {
    tensor: String,
    expected: Vec<usize>,
    actual: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct CapacityProjection {
    expert_width: usize,
    expert_q2_bytes: u64,
    expert_q3_bytes: u64,
    expert_q4_bytes: u64,
    non_expert_matrix_q4_bytes: u64,
    vector_fp16_bytes: u64,
    q2_expert_q4_shared_payload_bytes: u64,
    q3_expert_q4_shared_payload_bytes: u64,
    all_q4_payload_bytes: u64,
    fp16_kv_bytes_context_4096: u64,
}

#[derive(Debug, Serialize)]
struct CheckpointAudit {
    schema_version: u32,
    classification: &'static str,
    ready_for_qx_compilation: bool,
    model_id: String,
    model_revision: Option<String>,
    shape: ModelShape,
    tensor_count: usize,
    source_tensor_bytes: u64,
    source_expert_bytes: u64,
    source_non_expert_bytes: u64,
    missing_tensors: Vec<String>,
    unexpected_tensors: Vec<String>,
    shape_mismatches: Vec<ShapeMismatch>,
    non_bf16_tensors: Vec<String>,
    unmodified: CapacityProjection,
    structured_width_672: CapacityProjection,
    limitations: Vec<&'static str>,
}

fn parse_quant_type(value: &str) -> Result<QuantType, String> {
    QuantType::parse(value).ok_or_else(|| "expected q2, q3, or q4".to_owned())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.expert.is_some() != args.layer.is_some() {
        bail!("--layer and --expert must be supplied together");
    }
    let model = inspect_config(&args.model)?;
    if args.layer.is_some_and(|layer| layer >= model.shape.layers) {
        bail!("layer must be in 0..{}", model.shape.layers);
    }
    if args
        .expert
        .is_some_and(|expert| expert >= model.shape.experts)
    {
        bail!("expert must be in 0..{}", model.shape.experts);
    }

    print_model(&model);
    let checkpoint = Checkpoint::open(&args.model)?;
    if let (Some(layer), Some(expert)) = (args.layer, args.expert) {
        inspect_expert(&checkpoint, &model, layer, expert, args.quantize)?;
    } else if args.audit_output.is_none() {
        print_layer_overview(&checkpoint, &model, 0)?;
        println!("\ntensors indexed: {}", checkpoint.tensor_names().count());
    }
    if let Some(output) = args.audit_output {
        let audit = audit_checkpoint(&checkpoint, &model, args.model_revision)?;
        let bytes = serde_json::to_vec_pretty(&audit)?;
        fs::write(&output, [bytes.as_slice(), b"\n"].concat())
            .with_context(|| format!("write {}", output.display()))?;
        println!("\ncheckpoint audit: {}", output.display());
        println!(
            "ready for QX compilation: {}",
            audit.ready_for_qx_compilation
        );
        if !audit.ready_for_qx_compilation {
            bail!("checkpoint failed the QX release preflight");
        }
    }
    Ok(())
}

fn inspect_config(root: &std::path::Path) -> Result<InspectedModel> {
    let config_path = root.join("config.json");
    let bytes =
        fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?;
    let generic: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", config_path.display()))?;
    match generic.get("model_type").and_then(|value| value.as_str()) {
        Some("granite" | "granitemoe") => {
            let config = serde_json::from_slice::<qx_model::GraniteConfig>(&bytes)?;
            config.validate_v0().map_err(anyhow::Error::msg)?;
            Ok(InspectedModel {
                model_id: Granite3A400M::MODEL_ID,
                shape: ModelShape::granite_v0(),
            })
        }
        Some("qwen3_moe") => {
            let config = serde_json::from_slice::<Qwen3MoeConfig>(&bytes)?;
            let shape = config.validate_30b_a3b().map_err(anyhow::Error::msg)?;
            Ok(InspectedModel {
                model_id: QWEN3_30B_A3B_MODEL_ID,
                shape,
            })
        }
        other => bail!("unsupported model_type {other:?}"),
    }
}

fn print_model(model: &InspectedModel) {
    let shape = &model.shape;
    println!("QX Model Inspector\n");
    println!("model:        {}", model.model_id);
    println!("architecture: {:?}", shape.architecture);
    println!("layers:       {}", shape.layers);
    println!("hidden:       {}", shape.hidden_size);
    println!(
        "attention:    {} heads / {} KV heads / dim {}",
        shape.attention_heads, shape.key_value_heads, shape.head_dim
    );
    println!("experts:      {}", shape.experts);
    println!("top-k:        {}", shape.active_experts);
    println!("intermediate: {}", shape.expert_intermediate);
    println!("vocabulary:   {}", shape.vocabulary);
    println!("max context:  {}", shape.max_context);
}

fn print_layer_overview(
    checkpoint: &Checkpoint,
    model: &InspectedModel,
    layer: usize,
) -> Result<()> {
    println!("\nLayer {layer}:");
    let names = match model.shape.architecture {
        ModelArchitecture::GraniteMoe => vec![
            "block_sparse_moe.router.layer.weight".to_owned(),
            "block_sparse_moe.input_linear.weight".to_owned(),
            "block_sparse_moe.output_linear.weight".to_owned(),
        ],
        ModelArchitecture::Qwen3Moe => vec![
            "mlp.gate.weight".to_owned(),
            "mlp.experts.0.gate_proj.weight".to_owned(),
            "mlp.experts.0.up_proj.weight".to_owned(),
            "mlp.experts.0.down_proj.weight".to_owned(),
            "self_attn.q_proj.weight".to_owned(),
            "self_attn.k_proj.weight".to_owned(),
            "self_attn.v_proj.weight".to_owned(),
            "self_attn.o_proj.weight".to_owned(),
        ],
    };
    for suffix in names {
        let name = format!("model.layers.{layer}.{suffix}");
        match checkpoint.info(&name) {
            Ok(info) => println!(
                "{suffix}\n    shape: {:?}\n    dtype: {}\n    bytes: {}",
                info.shape, info.dtype, info.byte_len
            ),
            Err(_) => println!("{suffix}\n    not found"),
        }
    }
    println!("\nestimated group-32 payload per expert:");
    for quant_type in [QuantType::Q4, QuantType::Q3, QuantType::Q2] {
        println!(
            "    {quant_type}: {} bytes",
            estimate_quantized_bytes(model.shape.expert_weight_elements(), quant_type)
        );
    }
    Ok(())
}

fn inspect_expert(
    checkpoint: &Checkpoint,
    model: &InspectedModel,
    layer: usize,
    expert: usize,
    requested: Option<QuantType>,
) -> Result<()> {
    println!("\nExpert {expert} / Layer {layer}");
    match model.shape.architecture {
        ModelArchitecture::GraniteMoe => {
            inspect_granite_expert(checkpoint, &model.shape, layer, expert, requested)
        }
        ModelArchitecture::Qwen3Moe => {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let name = format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight");
                inspect_matrix(checkpoint, &name, requested)?;
            }
            Ok(())
        }
    }
}

fn inspect_granite_expert(
    checkpoint: &Checkpoint,
    shape: &ModelShape,
    layer: usize,
    expert: usize,
    requested: Option<QuantType>,
) -> Result<()> {
    for name in [
        format!("model.layers.{layer}.block_sparse_moe.input_linear.weight"),
        format!("model.layers.{layer}.block_sparse_moe.output_linear.weight"),
    ] {
        let info = checkpoint.info(&name)?;
        if info.shape.len() != 3 || info.shape[0] != shape.experts {
            bail!("unexpected Granite expert tensor shape: {:?}", info.shape);
        }
        let rows = info.shape[1];
        let cols = info.shape[2];
        let (tensor_shape, values) = checkpoint.tensor_f32(&name)?;
        let per_expert = rows * cols;
        if tensor_shape[0] != shape.experts
            || values.len() != tensor_shape.iter().product::<usize>()
        {
            bail!("unexpected Granite expert tensor data");
        }
        let start = expert * per_expert;
        print_quantization(
            &name,
            rows,
            cols,
            &values[start..start + per_expert],
            requested,
        );
    }
    Ok(())
}

fn inspect_matrix(checkpoint: &Checkpoint, name: &str, requested: Option<QuantType>) -> Result<()> {
    let (shape, values) = checkpoint.tensor_f32(name)?;
    if shape.len() != 2 {
        bail!("{name} must be a matrix, got {shape:?}");
    }
    print_quantization(name, shape[0], shape[1], &values, requested);
    Ok(())
}

fn print_quantization(
    name: &str,
    rows: usize,
    cols: usize,
    values: &[f32],
    requested: Option<QuantType>,
) {
    println!("\n{name}:\n    shape: [{rows}, {cols}]");
    println!("    source bytes (FP16/BF16): {}", values.len() * 2);
    for quant_type in requested
        .into_iter()
        .chain([QuantType::Q4, QuantType::Q3, QuantType::Q2])
    {
        let q = QuantizedTensor::quantize(values, rows, cols, quant_type, 32);
        println!(
            "    {quant_type}: {} bytes, compression: {:.2}x, MSE: {:.6}, relative L2: {:.6}",
            q.bytes(),
            values.len() as f32 * 2.0 / q.bytes() as f32,
            q.mse(values),
            q.relative_l2(values)
        );
    }
}

fn audit_checkpoint(
    checkpoint: &Checkpoint,
    model: &InspectedModel,
    revision: Option<String>,
) -> Result<CheckpointAudit> {
    let infos = checkpoint.tensor_infos()?;
    let actual = infos
        .iter()
        .map(|info| (info.name.clone(), info))
        .collect::<BTreeMap<_, _>>();
    let expected = expected_tensors(&model.shape);
    let missing_tensors = expected
        .keys()
        .filter(|name| !actual.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    let unexpected_tensors = actual
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    let shape_mismatches = expected
        .iter()
        .filter_map(|(name, expected_shape)| {
            actual.get(name).and_then(|info| {
                (info.shape != *expected_shape).then(|| ShapeMismatch {
                    tensor: name.clone(),
                    expected: expected_shape.clone(),
                    actual: info.shape.clone(),
                })
            })
        })
        .collect::<Vec<_>>();
    let non_bf16_tensors = infos
        .iter()
        .filter(|info| info.dtype != "BF16")
        .map(|info| format!("{} ({})", info.name, info.dtype))
        .collect::<Vec<_>>();
    let source_tensor_bytes = infos.iter().map(|info| info.byte_len as u64).sum();
    let source_expert_bytes = infos
        .iter()
        .filter(|info| is_expert_tensor(info))
        .map(|info| info.byte_len as u64)
        .sum();
    let source_non_expert_bytes = source_tensor_bytes - source_expert_bytes;
    let ready = missing_tensors.is_empty()
        && unexpected_tensors.is_empty()
        && shape_mismatches.is_empty()
        && non_bf16_tensors.is_empty();
    Ok(CheckpointAudit {
        schema_version: 1,
        classification: "qx-source-checkpoint-release-preflight",
        ready_for_qx_compilation: ready,
        model_id: model.model_id.to_owned(),
        model_revision: revision,
        shape: model.shape.clone(),
        tensor_count: infos.len(),
        source_tensor_bytes,
        source_expert_bytes,
        source_non_expert_bytes,
        missing_tensors,
        unexpected_tensors,
        shape_mismatches,
        non_bf16_tensors,
        unmodified: capacity_projection(&infos, &model.shape, model.shape.expert_intermediate),
        structured_width_672: capacity_projection(&infos, &model.shape, 672),
        limitations: vec![
            "capacity values are packed-payload estimates, not measured iOS physical memory",
            "the width-672 projection requires a trained joint gate/up/down student",
            "source readiness does not imply runtime semantic parity or quality promotion",
        ],
    })
}

fn is_expert_tensor(info: &TensorInfo) -> bool {
    info.name.contains(".mlp.experts.")
        || info.name.contains(".block_sparse_moe.input_linear")
        || info.name.contains(".block_sparse_moe.output_linear")
}

fn capacity_projection(
    infos: &[TensorInfo],
    shape: &ModelShape,
    expert_width: usize,
) -> CapacityProjection {
    let width_ratio = expert_width as f64 / shape.expert_intermediate as f64;
    let expert_elements = infos
        .iter()
        .filter(|info| is_expert_tensor(info))
        .map(|info| info.shape.iter().product::<usize>())
        .sum::<usize>();
    let projected_expert_elements = (expert_elements as f64 * width_ratio).round() as usize;
    let matrix_elements = infos
        .iter()
        .filter(|info| !is_expert_tensor(info) && info.shape.len() >= 2)
        .map(|info| info.shape.iter().product::<usize>())
        .collect::<Vec<_>>();
    let vector_elements = infos
        .iter()
        .filter(|info| !is_expert_tensor(info) && info.shape.len() == 1)
        .map(|info| info.shape.iter().product::<usize>())
        .sum::<usize>();
    let expert_q2_bytes = estimate_quantized_bytes(projected_expert_elements, QuantType::Q2) as u64;
    let expert_q3_bytes = estimate_quantized_bytes(projected_expert_elements, QuantType::Q3) as u64;
    let expert_q4_bytes = estimate_quantized_bytes(projected_expert_elements, QuantType::Q4) as u64;
    let non_expert_matrix_q4_bytes = matrix_elements
        .into_iter()
        .map(|elements| estimate_quantized_bytes(elements, QuantType::Q4) as u64)
        .sum::<u64>();
    let vector_fp16_bytes = (vector_elements * 2) as u64;
    let fp16_kv_bytes_context_4096 = (shape.kv_elements_per_token() * 4_096 * 2) as u64;
    CapacityProjection {
        expert_width,
        expert_q2_bytes,
        expert_q3_bytes,
        expert_q4_bytes,
        non_expert_matrix_q4_bytes,
        vector_fp16_bytes,
        q2_expert_q4_shared_payload_bytes: expert_q2_bytes
            + non_expert_matrix_q4_bytes
            + vector_fp16_bytes,
        q3_expert_q4_shared_payload_bytes: expert_q3_bytes
            + non_expert_matrix_q4_bytes
            + vector_fp16_bytes,
        all_q4_payload_bytes: expert_q4_bytes + non_expert_matrix_q4_bytes + vector_fp16_bytes,
        fp16_kv_bytes_context_4096,
    }
}

fn expected_tensors(shape: &ModelShape) -> BTreeMap<String, Vec<usize>> {
    let mut tensors = BTreeMap::new();
    if shape.architecture != ModelArchitecture::Qwen3Moe {
        return tensors;
    }
    tensors.insert(
        "model.embed_tokens.weight".to_owned(),
        vec![shape.vocabulary, shape.hidden_size],
    );
    tensors.insert("model.norm.weight".to_owned(), vec![shape.hidden_size]);
    tensors.insert(
        "lm_head.weight".to_owned(),
        vec![shape.vocabulary, shape.hidden_size],
    );
    let query_width = shape.attention_heads * shape.head_dim;
    let kv_width = shape.key_value_heads * shape.head_dim;
    for layer in 0..shape.layers {
        let prefix = format!("model.layers.{layer}");
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            vec![shape.hidden_size],
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![shape.hidden_size],
        );
        tensors.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![query_width, shape.hidden_size],
        );
        tensors.insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_width, shape.hidden_size],
        );
        tensors.insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_width, shape.hidden_size],
        );
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![shape.hidden_size, query_width],
        );
        if shape.query_key_norm {
            tensors.insert(
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![shape.head_dim],
            );
            tensors.insert(
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![shape.head_dim],
            );
        }
        tensors.insert(
            format!("{prefix}.mlp.gate.weight"),
            vec![shape.experts, shape.hidden_size],
        );
        for expert in 0..shape.experts {
            let expert_prefix = format!("{prefix}.mlp.experts.{expert}");
            tensors.insert(
                format!("{expert_prefix}.gate_proj.weight"),
                vec![shape.expert_intermediate, shape.hidden_size],
            );
            tensors.insert(
                format!("{expert_prefix}.up_proj.weight"),
                vec![shape.expert_intermediate, shape.hidden_size],
            );
            tensors.insert(
                format!("{expert_prefix}.down_proj.weight"),
                vec![shape.hidden_size, shape.expert_intermediate],
            );
        }
    }
    tensors
}

fn estimate_quantized_bytes(elements: usize, quant_type: QuantType) -> usize {
    let groups = elements.div_ceil(32);
    (elements * quant_type.bits()).div_ceil(8) + groups * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_tensor_contract_covers_complete_checkpoint() {
        let config: Qwen3MoeConfig = serde_json::from_str(
            r#"{
              "model_type":"qwen3_moe","num_hidden_layers":48,"hidden_size":2048,
              "num_attention_heads":32,"num_key_value_heads":4,"head_dim":128,
              "num_experts":128,"num_experts_per_tok":8,"moe_intermediate_size":768,
              "intermediate_size":6144,"max_position_embeddings":40960,"vocab_size":151936,
              "rms_norm_eps":0.000001,"rope_theta":1000000.0,"norm_topk_prob":true,
              "tie_word_embeddings":false
            }"#,
        )
        .unwrap();
        let shape = config.validate_30b_a3b().unwrap();
        let expected = expected_tensors(&shape);
        assert_eq!(expected.len(), 18_867);
        assert_eq!(
            expected["model.layers.47.self_attn.q_proj.weight"],
            vec![4096, 2048]
        );
        assert_eq!(
            expected["model.layers.47.mlp.experts.127.down_proj.weight"],
            vec![2048, 768]
        );
    }
}
