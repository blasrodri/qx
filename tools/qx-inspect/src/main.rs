use anyhow::{Context, Result, bail};
use clap::Parser;
use qx_model::{
    Granite3A400M, ModelArchitecture, ModelShape, QWEN3_30B_A3B_MODEL_ID, Qwen3MoeConfig,
};
use qx_quant::{QuantType, QuantizedTensor};
use qx_safetensors::{Checkpoint, TensorInfo};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

#[derive(Debug, Parser)]
#[command(
    name = "qx-inspect",
    about = "Inspect and release-audit a supported Hugging Face checkpoint"
)]
struct Args {
    /// Hugging Face model directory containing config.json and safetensors.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Inspect a compiled QX artifact instead of a source checkpoint.
    #[arg(long)]
    artifact: Option<PathBuf>,
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
    /// Pinned source manifest containing immutable shard sizes and SHA-256.
    /// Required for a release-ready audit.
    #[arg(long)]
    reference_manifest: Option<PathBuf>,
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
    source_files_verified: bool,
    shard_verification: Vec<ShardVerification>,
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

#[derive(Debug, Deserialize)]
struct SourceReference {
    model: SourceModelReference,
}

#[derive(Debug, Deserialize)]
struct SourceModelReference {
    repository: String,
    revision: String,
    shards: Vec<SourceShardReference>,
}

#[derive(Debug, Deserialize)]
struct SourceShardReference {
    filename: String,
    size_bytes: u64,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct ShardVerification {
    filename: String,
    expected_size_bytes: u64,
    actual_size_bytes: u64,
    expected_sha256: String,
    actual_sha256: String,
    verified: bool,
}

#[derive(Debug, Serialize)]
struct ArtifactAudit {
    schema_version: u32,
    classification: &'static str,
    capacity_baseline_complete: bool,
    execution_promoted: bool,
    artifact_sha256: String,
    file_bytes: u64,
    model_id: String,
    target_id: String,
    context_limit: u32,
    tensor_entries: usize,
    q2_entries: usize,
    q4_entries: usize,
    raw_entries: usize,
    gate_up_entries: usize,
    down_entries: usize,
    memory_plan: ArtifactMemoryAudit,
    model_shape: Option<ModelShape>,
    plan_id: Option<String>,
    limitations: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct ArtifactMemoryAudit {
    weights_bytes: u64,
    kv_cache_bytes: u64,
    runtime_bytes: u64,
    total_bytes: u64,
}

fn parse_quant_type(value: &str) -> Result<QuantType, String> {
    QuantType::parse(value).ok_or_else(|| "expected q2, q3, or q4".to_owned())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(artifact) = args.artifact.as_deref() {
        if args.model.is_some()
            || args.layer.is_some()
            || args.expert.is_some()
            || args.quantize.is_some()
            || args.reference_manifest.is_some()
            || args.model_revision.is_some()
        {
            bail!("--artifact cannot be combined with source-checkpoint options");
        }
        let audit = inspect_artifact(artifact)?;
        print_artifact_audit(&audit);
        if let Some(output) = args.audit_output {
            let bytes = serde_json::to_vec_pretty(&audit)?;
            fs::write(&output, [bytes.as_slice(), b"\n"].concat())
                .with_context(|| format!("write {}", output.display()))?;
            println!("artifact audit: {}", output.display());
        }
        if !audit.capacity_baseline_complete {
            bail!("artifact failed the Qwen capacity-baseline contract");
        }
        return Ok(());
    }
    let model_root = args
        .model
        .as_deref()
        .context("either --model or --artifact is required")?;
    if args.expert.is_some() != args.layer.is_some() {
        bail!("--layer and --expert must be supplied together");
    }
    let model = inspect_config(model_root)?;
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
    let checkpoint = Checkpoint::open(model_root)?;
    if let (Some(layer), Some(expert)) = (args.layer, args.expert) {
        inspect_expert(&checkpoint, &model, layer, expert, args.quantize)?;
    } else if args.audit_output.is_none() {
        print_layer_overview(&checkpoint, &model, 0)?;
        println!("\ntensors indexed: {}", checkpoint.tensor_names().count());
    }
    if let Some(output) = args.audit_output {
        let (revision, shard_verification) = match args.reference_manifest.as_deref() {
            Some(reference) => {
                let (reference_revision, verification) =
                    verify_source_reference(model_root, reference)?;
                if args
                    .model_revision
                    .as_ref()
                    .is_some_and(|revision| revision != &reference_revision)
                {
                    bail!("--model-revision disagrees with the source reference manifest");
                }
                (Some(reference_revision), verification)
            }
            None => (args.model_revision, Vec::new()),
        };
        let audit = audit_checkpoint(&checkpoint, &model, revision, shard_verification)?;
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

fn inspect_artifact(path: &Path) -> Result<ArtifactAudit> {
    let artifact = qx_format::MappedArtifact::open(path)
        .with_context(|| format!("open QX artifact {}", path.display()))?;
    let info = artifact.info();
    let q2_entries = info
        .entries
        .iter()
        .filter(|entry| entry.group_size != 0 && entry.quant_type == QuantType::Q2)
        .count();
    let q4_entries = info
        .entries
        .iter()
        .filter(|entry| entry.group_size != 0 && entry.quant_type == QuantType::Q4)
        .count();
    let raw_entries = info
        .entries
        .iter()
        .filter(|entry| entry.group_size == 0)
        .count();
    let gate_up_entries = info
        .entries
        .iter()
        .filter(|entry| entry.kind == qx_format::TensorKind::GateUp)
        .count();
    let down_entries = info
        .entries
        .iter()
        .filter(|entry| entry.kind == qx_format::TensorKind::Down)
        .count();
    let plan = info.execution_plan.as_ref();
    let complete = info.header.model_id == QWEN3_30B_A3B_MODEL_ID
        && info.header.target_id == "apple-a19-pro"
        && info.header.context_limit == 4096
        && info.entries.len() == 12_723
        && q2_entries == 12_288
        && q4_entries == 242
        && raw_entries == 193
        && gate_up_entries == 6_144
        && down_entries == 6_144
        && plan
            .and_then(|plan| plan.model_shape.as_ref())
            .is_some_and(|shape| {
                shape.architecture == ModelArchitecture::Qwen3Moe
                    && shape.expert_intermediate == 768
            })
        && plan.is_some_and(|plan| plan.schedule.expected_dispatches_per_token == 0);
    let memory = info.memory_plan;
    let runtime_bytes = memory.total_bytes - memory.weights_bytes - memory.kv_cache_bytes;
    Ok(ArtifactAudit {
        schema_version: 1,
        classification: "qwen3-a19pro-q2-capacity-baseline-artifact",
        capacity_baseline_complete: complete,
        execution_promoted: false,
        artifact_sha256: sha256_file(path)?,
        file_bytes: fs::metadata(path)?.len(),
        model_id: info.header.model_id.clone(),
        target_id: info.header.target_id.clone(),
        context_limit: info.header.context_limit,
        tensor_entries: info.entries.len(),
        q2_entries,
        q4_entries,
        raw_entries,
        gate_up_entries,
        down_entries,
        memory_plan: ArtifactMemoryAudit {
            weights_bytes: memory.weights_bytes,
            kv_cache_bytes: memory.kv_cache_bytes,
            runtime_bytes,
            total_bytes: memory.total_bytes,
        },
        model_shape: plan.and_then(|plan| plan.model_shape.clone()),
        plan_id: plan.map(|plan| plan.plan_id.clone()),
        limitations: vec![
            "uniform symmetric Q2 has not passed the QX quality gates",
            "the Qwen Metal runtime and physical iPhone memory gate are not implemented",
            "this artifact validates capacity and conversion only; it is not an app release model",
        ],
    })
}

fn print_artifact_audit(audit: &ArtifactAudit) {
    println!("QX Artifact Inspector\n");
    println!("model:              {}", audit.model_id);
    println!("target:             {}", audit.target_id);
    println!("file bytes:         {}", audit.file_bytes);
    println!("tensor entries:     {}", audit.tensor_entries);
    println!(
        "Q2/Q4/raw entries:  {}/{}/{}",
        audit.q2_entries, audit.q4_entries, audit.raw_entries
    );
    println!(
        "gate-up/down:       {}/{}",
        audit.gate_up_entries, audit.down_entries
    );
    println!("logical weights:    {}", audit.memory_plan.weights_bytes);
    println!("logical total:      {}", audit.memory_plan.total_bytes);
    println!("capacity complete:  {}", audit.capacity_baseline_complete);
    println!("execution promoted: {}", audit.execution_promoted);
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
    let mut quant_types = requested.into_iter().collect::<Vec<_>>();
    for quant_type in [QuantType::Q4, QuantType::Q3, QuantType::Q2] {
        if !quant_types.contains(&quant_type) {
            quant_types.push(quant_type);
        }
    }
    for quant_type in quant_types {
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
    shard_verification: Vec<ShardVerification>,
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
    let source_files_verified =
        !shard_verification.is_empty() && shard_verification.iter().all(|shard| shard.verified);
    let ready = source_files_verified
        && missing_tensors.is_empty()
        && unexpected_tensors.is_empty()
        && shape_mismatches.is_empty()
        && non_bf16_tensors.is_empty();
    Ok(CheckpointAudit {
        schema_version: 1,
        classification: "qx-source-checkpoint-release-preflight",
        ready_for_qx_compilation: ready,
        model_id: model.model_id.to_owned(),
        model_revision: revision,
        source_files_verified,
        shard_verification,
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

fn verify_source_reference(
    model_root: &Path,
    manifest_path: &Path,
) -> Result<(String, Vec<ShardVerification>)> {
    let reference: SourceReference = serde_json::from_slice(
        &fs::read(manifest_path).with_context(|| format!("read {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse {}", manifest_path.display()))?;
    if reference.model.repository != QWEN3_30B_A3B_MODEL_ID {
        bail!("source reference repository does not match the inspected model");
    }
    if reference.model.shards.is_empty() {
        bail!("source reference contains no checkpoint shards");
    }
    let mut verification = Vec::with_capacity(reference.model.shards.len());
    for expected in reference.model.shards {
        if expected.filename.contains('/') || expected.filename.contains("..") {
            bail!("unsafe source shard filename {}", expected.filename);
        }
        let path = model_root.join(&expected.filename);
        let actual_size_bytes = fs::metadata(&path)
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        let actual_sha256 = sha256_file(&path)?;
        let verified = actual_size_bytes == expected.size_bytes
            && actual_sha256.eq_ignore_ascii_case(&expected.sha256);
        verification.push(ShardVerification {
            filename: expected.filename,
            expected_size_bytes: expected.size_bytes,
            actual_size_bytes,
            expected_sha256: expected.sha256,
            actual_sha256,
            verified,
        });
    }
    Ok((reference.model.revision, verification))
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, file);
    let mut buffer = vec![0_u8; 4 * 1024 * 1024];
    let mut digest = Sha256::new();
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
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
