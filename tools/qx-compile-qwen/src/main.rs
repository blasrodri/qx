use anyhow::{Context, Result, bail};
use clap::Parser;
use qx_format::{
    AttentionTopology, CompiledTensor, ExecutionPlanManifest, ExpertLayerWeightPlan,
    ExpertRepresentation, ExpertWeightPlan, KvPrecision, MemoryPlan, MoeTopology, QkvTopology,
    RawTensor, RepresentationPlan, RmsTopology, RopeKvTopology, SparseExpertPolicy, TensorKind,
    WeightStorage, write_artifact_with_raw_and_plan,
};
use qx_model::{ModelShape, QWEN3_30B_A3B_MODEL_ID, Qwen3MoeConfig};
use qx_quant::{PackingLayout, QuantType, QuantizedTensor};
use qx_safetensors::{Checkpoint, TensorInfo};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

const JOINT_EXPERT_STUDENT_MAGIC: &[u8; 8] = b"QXEJS001";

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
    #[arg(long, required_unless_present = "check_expert_policy")]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 4096)]
    context: usize,
    #[arg(long, value_parser = parse_expert_bits, default_value = "q2_s4")]
    expert_bits: QuantType,
    /// Checkpoint-specific Qwen expert policy. Requires --expert-build-dir.
    #[arg(long, requires = "expert_build_dir")]
    expert_policy: Option<PathBuf>,
    /// Completed full-layer student build containing manifests and sidecars.
    #[arg(long, requires = "expert_policy")]
    expert_build_dir: Option<PathBuf>,
    /// Validate and summarize a heterogeneous expert policy without compiling weights.
    #[arg(long, requires = "expert_policy", conflicts_with = "output")]
    check_expert_policy: bool,
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

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd)]
struct ExpertTarget {
    layer: u16,
    expert: u8,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenExpertPolicy {
    schema_version: u32,
    classification: String,
    policy_id: String,
    model_id: String,
    screening_reference_sha256: String,
    selection_evidence: PathBuf,
    selection_evidence_sha256: String,
    frozen_evidence: PathBuf,
    frozen_evidence_sha256: String,
    student_width: u16,
    default_student_codec: String,
    q3_layers: Vec<u16>,
    validated_layers: Vec<u16>,
    causal_anchors: Vec<ExpertTarget>,
}

#[derive(Debug, Deserialize)]
struct ProjectionCodecs {
    gate: String,
    up: String,
    down: String,
}

#[derive(Debug, Deserialize)]
struct StudentManifestEntry {
    layer: u16,
    expert: u8,
    sidecar: PathBuf,
}

#[derive(Debug, Deserialize)]
struct LayerStudentManifest {
    classification: String,
    layer: u16,
    student_width: u16,
    teacher_target: String,
    projection_codecs: ProjectionCodecs,
    students: Vec<StudentManifestEntry>,
}

#[derive(Debug, Deserialize)]
struct CausalEvidenceStudent {
    layer: u16,
    expert: u8,
}

#[derive(Debug, Deserialize)]
struct CausalEvidence {
    model_sha256: String,
    scored_tokens: usize,
    relative_perplexity_delta: f64,
    students: Vec<CausalEvidenceStudent>,
}

#[derive(Clone, Debug)]
struct StudentSpec {
    codec: QuantType,
    sidecar: PathBuf,
}

#[derive(Debug)]
struct ResolvedExpertPolicy {
    plan: ExpertWeightPlan,
    students: BTreeMap<(u16, u8), StudentSpec>,
}

struct JointExpertStudent {
    layer: u16,
    expert: u8,
    hidden: usize,
    width: usize,
    teacher_width: usize,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

#[derive(Default)]
struct PendingGateUp {
    gate: BTreeMap<(u16, u8), QuantizedTensor>,
    up: BTreeMap<(u16, u8), QuantizedTensor>,
}

fn parse_expert_bits(value: &str) -> Result<QuantType, String> {
    match QuantType::parse(value) {
        Some(bits @ (QuantType::Q2 | QuantType::Q2S4 | QuantType::Q3 | QuantType::Q4)) => Ok(bits),
        _ => Err("expected q2, q2_s4, q3, or q4".to_owned()),
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
    let expert_policy = match (&args.expert_policy, &args.expert_build_dir) {
        (Some(policy), Some(build_dir)) => Some(resolve_expert_policy(policy, build_dir, &shape)?),
        (None, None) => None,
        _ => unreachable!("clap enforces paired expert policy arguments"),
    };

    println!("compiling {QWEN3_30B_A3B_MODEL_ID}");
    if let Some(policy) = &expert_policy {
        println!("expert policy: {}", policy.plan.policy_id);
        println!("student experts: {}", policy.students.len());
    } else {
        println!("expert representation: {}", args.expert_bits);
    }
    println!("context: {}", args.context);
    if args.check_expert_policy {
        let policy = expert_policy
            .as_ref()
            .expect("clap requires an expert policy for policy checking");
        let q3_students = policy
            .students
            .values()
            .filter(|student| student.codec == QuantType::Q3)
            .count();
        let q2s4_students = policy.students.len() - q3_students;
        let source_experts = policy
            .plan
            .layers
            .iter()
            .map(|layer| layer.source_experts.len())
            .sum::<usize>();
        let causal_anchors = policy
            .plan
            .layers
            .iter()
            .map(|layer| layer.causal_anchors.len())
            .sum::<usize>();
        println!("Q2S4 students: {q2s4_students}");
        println!("Q3 students: {q3_students}");
        println!("source experts: {source_experts}");
        println!("causal anchors: {causal_anchors}");
        println!("validated layers: {:?}", policy.plan.validated_layers);
        println!("expert policy is internally consistent");
        return Ok(());
    }
    let output = args
        .output
        .as_ref()
        .expect("clap requires output unless policy checking");

    let checkpoint = Checkpoint::open(&args.model)?;
    let mut compiled = Vec::<CompiledTensor>::with_capacity(18_700);
    let mut raw = Vec::<RawTensor>::with_capacity(256);
    let mut pending = PendingGateUp::default();
    let mut compiled_students = BTreeSet::new();
    let mut visited = 0usize;
    checkpoint.visit_tensors_f32(|info, values| {
        compile_tensor(
            info,
            values,
            &shape,
            args.expert_bits,
            expert_policy.as_ref(),
            &mut compiled,
            &mut raw,
            &mut pending,
            &mut compiled_students,
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
    if let Some(policy) = &expert_policy
        && compiled_students.len() != policy.students.len()
    {
        bail!(
            "compiled {} of {} policy students",
            compiled_students.len(),
            policy.students.len()
        );
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
        QuantType::Q2S4 => ExpertRepresentation::Group32Q2S4RowMajor,
        QuantType::Q3 => ExpertRepresentation::Group32Q3RowMajor,
        QuantType::Q4 => ExpertRepresentation::Group32Q4RowMajor,
        _ => unreachable!(),
    };
    execution_plan.representations = RepresentationPlan {
        gate_up: if expert_policy.is_some() {
            ExpertRepresentation::MixedQ2Q3Q4
        } else {
            expert_representation
        },
        down: if expert_policy.is_some() {
            ExpertRepresentation::MixedQ2Q3Q4
        } else {
            expert_representation
        },
        dense: ExpertRepresentation::Group32Q4RowMajor,
    };
    execution_plan.expert_weights = expert_policy.as_ref().map(|policy| policy.plan.clone());
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
    println!("writing {}", output.display());
    write_artifact_with_raw_and_plan(
        output,
        QWEN3_30B_A3B_MODEL_ID,
        "apple-a19-pro",
        args.context as u32,
        memory_plan,
        &compiled,
        &raw,
        &execution_plan,
    )?;
    println!("artifact ready: {}", output.display());
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

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} must contain exactly 64 hexadecimal characters");
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn resolve_build_path(build_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        build_dir.join(path)
    }
}

fn resolve_sidecar_path(
    build_dir: &Path,
    manifest_path: &Path,
    declared: &Path,
) -> Result<PathBuf> {
    if declared.is_file() {
        return Ok(declared.to_path_buf());
    }
    if let Some(file_name) = declared.file_name() {
        let local = manifest_path
            .parent()
            .and_then(Path::parent)
            .unwrap_or(build_dir)
            .join(
                if manifest_path
                    .parent()
                    .is_some_and(|path| path.ends_with("manifests-q3"))
                {
                    "students-q3"
                } else {
                    "students"
                },
            )
            .join(
                declared
                    .parent()
                    .and_then(Path::file_name)
                    .unwrap_or_default(),
            )
            .join(file_name);
        if local.is_file() {
            return Ok(local);
        }
    }
    bail!("student sidecar does not exist: {}", declared.display())
}

fn validate_causal_evidence(
    path: &Path,
    expected_sha256: &str,
    reference_sha256: &str,
    expected_students: &BTreeSet<(u16, u8)>,
    label: &str,
) -> Result<()> {
    validate_sha256(expected_sha256, &format!("{label} evidence SHA-256"))?;
    let actual_sha256 = sha256_file(path)?;
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        bail!("{label} evidence SHA-256 does not match {}", path.display());
    }
    let evidence: CausalEvidence = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    if !evidence.model_sha256.eq_ignore_ascii_case(reference_sha256) {
        bail!("{label} evidence uses a different screening reference");
    }
    if evidence.scored_tokens == 0 || evidence.relative_perplexity_delta > 0.01 {
        bail!("{label} evidence does not pass the 1% screening gate");
    }
    let actual_students = evidence
        .students
        .into_iter()
        .map(|student| (student.layer, student.expert))
        .collect::<BTreeSet<_>>();
    if actual_students != *expected_students {
        bail!("{label} evidence student set does not match the validated policy scope");
    }
    Ok(())
}

fn resolve_expert_policy(
    path: &Path,
    build_dir: &Path,
    shape: &ModelShape,
) -> Result<ResolvedExpertPolicy> {
    let policy: QwenExpertPolicy = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    if policy.schema_version != 1
        || policy.classification != "qx-qwen-expert-weight-policy"
        || policy.model_id != QWEN3_30B_A3B_MODEL_ID
    {
        bail!("expert policy schema, classification, or model identity is invalid");
    }
    if policy.policy_id.trim().is_empty() {
        bail!("expert policy ID cannot be empty");
    }
    validate_sha256(
        &policy.screening_reference_sha256,
        "screening reference SHA-256",
    )?;
    if usize::from(policy.student_width) != shape.expert_intermediate {
        bail!("expert policy student width does not match the model");
    }
    let default_codec = QuantType::parse(&policy.default_student_codec)
        .filter(|codec| *codec == QuantType::Q2S4)
        .ok_or_else(|| anyhow::anyhow!("expert policy default codec must be q2_s4"))?;
    let q3_layers = policy.q3_layers.iter().copied().collect::<BTreeSet<_>>();
    if q3_layers.len() != policy.q3_layers.len()
        || q3_layers
            .iter()
            .any(|layer| usize::from(*layer) >= shape.layers)
    {
        bail!("expert policy Q3 layers must be unique and within the model");
    }
    let validated_layers = policy
        .validated_layers
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if validated_layers.len() != policy.validated_layers.len()
        || validated_layers.is_empty()
        || validated_layers
            .iter()
            .any(|layer| usize::from(*layer) >= shape.layers)
    {
        bail!("expert policy validated layers must be unique and within the model");
    }
    if !validated_layers.is_subset(&q3_layers) {
        bail!("validated layers must be a subset of Q3 layers");
    }
    let anchors = policy
        .causal_anchors
        .iter()
        .map(|target| (target.layer, target.expert))
        .collect::<BTreeSet<_>>();
    if anchors.len() != policy.causal_anchors.len()
        || anchors.iter().any(|(layer, expert)| {
            usize::from(*layer) >= shape.layers || usize::from(*expert) >= shape.experts
        })
    {
        bail!("expert policy causal anchors must be unique and within the model");
    }

    let mut students = BTreeMap::new();
    let mut layer_plans = Vec::with_capacity(shape.layers);
    for layer_index in 0..shape.layers {
        let layer = u16::try_from(layer_index).context("Qwen layer exceeds u16")?;
        let codec = if q3_layers.contains(&layer) {
            QuantType::Q3
        } else {
            default_codec
        };
        let manifest_dir = if codec == QuantType::Q3 {
            "manifests-q3"
        } else {
            "manifests"
        };
        let manifest_path = build_dir
            .join(manifest_dir)
            .join(format!("layer{layer}.json"));
        let manifest: LayerStudentManifest = serde_json::from_slice(
            &fs::read(&manifest_path)
                .with_context(|| format!("read {}", manifest_path.display()))?,
        )?;
        let expected_codec = if codec == QuantType::Q3 { "q3" } else { "q2s4" };
        if manifest.classification != "qwen-layer-batched-width-transformation"
            || manifest.layer != layer
            || manifest.student_width != policy.student_width
            || manifest.teacher_target != "captured-q4"
            || manifest.projection_codecs.gate != expected_codec
            || manifest.projection_codecs.up != expected_codec
            || manifest.projection_codecs.down != expected_codec
        {
            bail!("incompatible student manifest {}", manifest_path.display());
        }
        let layer_anchors = anchors
            .iter()
            .filter_map(|(anchor_layer, expert)| (*anchor_layer == layer).then_some(*expert))
            .collect::<BTreeSet<_>>();
        let mut manifest_students = BTreeMap::new();
        for entry in manifest.students {
            if entry.layer != layer || usize::from(entry.expert) >= shape.experts {
                bail!("student manifest contains an invalid target");
            }
            let sidecar = resolve_sidecar_path(build_dir, &manifest_path, &entry.sidecar)?;
            if manifest_students.insert(entry.expert, sidecar).is_some() {
                bail!(
                    "student manifest contains duplicate expert {}",
                    entry.expert
                );
            }
        }
        if !layer_anchors
            .iter()
            .all(|expert| manifest_students.contains_key(expert))
        {
            bail!("causal anchor is not a transformed student in layer {layer}");
        }
        for expert in &layer_anchors {
            manifest_students.remove(expert);
        }
        let student_experts = manifest_students.keys().copied().collect::<Vec<_>>();
        let student_set = student_experts.iter().copied().collect::<BTreeSet<_>>();
        let source_experts = (0..shape.experts)
            .map(|expert| u8::try_from(expert).expect("validated Qwen expert count fits u8"))
            .filter(|expert| !student_set.contains(expert))
            .collect::<Vec<_>>();
        for (expert, sidecar) in manifest_students {
            students.insert((layer, expert), StudentSpec { codec, sidecar });
        }
        layer_plans.push(ExpertLayerWeightPlan {
            layer,
            student_representation: match codec {
                QuantType::Q2S4 => ExpertRepresentation::Group32Q2S4RowMajor,
                QuantType::Q3 => ExpertRepresentation::Group32Q3Superblock112,
                _ => unreachable!(),
            },
            source_representation: ExpertRepresentation::Group32Q4RowMajor,
            student_experts,
            source_experts,
            causal_anchors: layer_anchors.into_iter().collect(),
            training_manifest_sha256: sha256_file(&manifest_path)?,
        });
    }
    if !anchors.iter().all(|(layer, expert)| {
        layer_plans[usize::from(*layer)]
            .causal_anchors
            .contains(expert)
    }) {
        bail!("expert policy contains an unresolved causal anchor");
    }
    let expected_evidence_students = students
        .keys()
        .filter(|(layer, _)| validated_layers.contains(layer))
        .copied()
        .collect::<BTreeSet<_>>();
    let selection_path = resolve_build_path(build_dir, &policy.selection_evidence);
    let frozen_path = resolve_build_path(build_dir, &policy.frozen_evidence);
    validate_causal_evidence(
        &selection_path,
        &policy.selection_evidence_sha256,
        &policy.screening_reference_sha256,
        &expected_evidence_students,
        "selection",
    )?;
    validate_causal_evidence(
        &frozen_path,
        &policy.frozen_evidence_sha256,
        &policy.screening_reference_sha256,
        &expected_evidence_students,
        "frozen",
    )?;

    Ok(ResolvedExpertPolicy {
        plan: ExpertWeightPlan {
            policy_id: policy.policy_id,
            screening_reference_sha256: policy.screening_reference_sha256.to_ascii_lowercase(),
            selection_evidence_sha256: policy.selection_evidence_sha256.to_ascii_lowercase(),
            frozen_evidence_sha256: policy.frozen_evidence_sha256.to_ascii_lowercase(),
            student_width: policy.student_width,
            validated_layers: validated_layers.into_iter().collect(),
            layers: layer_plans,
        },
        students,
    })
}

fn read_joint_expert_student(path: &Path) -> Result<JointExpertStudent> {
    let mut reader =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != JOINT_EXPERT_STUDENT_MAGIC {
        bail!("joint expert student has invalid magic: {}", path.display());
    }
    let mut dimensions = [0_u32; 5];
    for value in &mut dimensions {
        let mut bytes = [0_u8; 4];
        reader.read_exact(&mut bytes)?;
        *value = u32::from_le_bytes(bytes);
    }
    let [layer, expert, hidden, width, teacher_width] = dimensions.map(|value| value as usize);
    let values = hidden
        .checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("joint expert student dimensions overflow"))?;
    let mut read_matrix = || -> Result<Vec<f32>> {
        let mut bytes = vec![0_u8; values * std::mem::size_of::<f32>()];
        reader.read_exact(&mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    };
    let gate = read_matrix()?;
    let up = read_matrix()?;
    let down = read_matrix()?;
    let mut trailing = [0_u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!(
            "joint expert student contains trailing bytes: {}",
            path.display()
        );
    }
    Ok(JointExpertStudent {
        layer: u16::try_from(layer).context("student layer exceeds u16")?,
        expert: u8::try_from(expert).context("student expert exceeds u8")?,
        hidden,
        width,
        teacher_width,
        gate,
        up,
        down,
    })
}

fn quantize_student_matrix(
    values: &[f32],
    rows: usize,
    cols: usize,
    codec: QuantType,
) -> Result<QuantizedTensor> {
    if values.iter().any(|value| !value.is_finite()) {
        bail!("student matrix contains a non-finite value");
    }
    let mut tensor = QuantizedTensor::quantize(values, rows, cols, codec, 32);
    let decoded = tensor.dequantize();
    let scale = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let tolerance = scale.max(1.0) * 2.0e-3;
    if values
        .iter()
        .zip(decoded.iter())
        .any(|(source, packed)| (source - packed).abs() > tolerance)
    {
        bail!("student matrix does not lie on the declared {codec} codec");
    }
    if codec == QuantType::Q3 {
        tensor
            .repack(PackingLayout::Q3Superblock112)
            .map_err(anyhow::Error::msg)?;
    }
    Ok(tensor)
}

fn compile_joint_expert_student(
    student: &JointExpertStudent,
    key: (u16, u8),
    codec: QuantType,
    shape: &ModelShape,
    compiled: &mut Vec<CompiledTensor>,
    pending: &mut PendingGateUp,
) -> Result<()> {
    if (student.layer, student.expert) != key
        || student.hidden != shape.hidden_size
        || student.width != shape.expert_intermediate
        || student.teacher_width != shape.expert_intermediate
    {
        bail!("joint expert student dimensions or target do not match the policy");
    }
    pending.gate.insert(
        key,
        quantize_student_matrix(&student.gate, student.width, student.hidden, codec)?,
    );
    pending.up.insert(
        key,
        quantize_student_matrix(&student.up, student.width, student.hidden, codec)?,
    );
    finish_gate_up(key, compiled, pending)?;
    compiled.push(CompiledTensor {
        layer: key.0,
        expert: key.1,
        kind: TensorKind::Down,
        tensor: quantize_student_matrix(&student.down, student.hidden, student.width, codec)?,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compile_tensor(
    info: &TensorInfo,
    values: &[f32],
    shape: &ModelShape,
    expert_bits: QuantType,
    expert_policy: Option<&ResolvedExpertPolicy>,
    compiled: &mut Vec<CompiledTensor>,
    raw: &mut Vec<RawTensor>,
    pending: &mut PendingGateUp,
    compiled_students: &mut BTreeSet<(u16, u8)>,
) -> Result<()> {
    if let Some((layer, expert, projection)) = parse_expert_name(&info.name) {
        let expected = match projection {
            "gate_proj" | "up_proj" => vec![shape.expert_intermediate, shape.hidden_size],
            "down_proj" => vec![shape.hidden_size, shape.expert_intermediate],
            _ => bail!("unknown expert projection {projection}"),
        };
        require_shape(info, &expected)?;
        let key = (layer, expert);
        if let Some(spec) = expert_policy.and_then(|policy| policy.students.get(&key)) {
            if compiled_students.insert(key) {
                let student = read_joint_expert_student(&spec.sidecar)?;
                compile_joint_expert_student(&student, key, spec.codec, shape, compiled, pending)?;
            }
            return Ok(());
        }
        let expert_bits = if expert_policy.is_some() {
            QuantType::Q4
        } else {
            expert_bits
        };
        let tensor = QuantizedTensor::quantize(values, expected[0], expected[1], expert_bits, 32);
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
    if !pending.gate.contains_key(&key) || !pending.up.contains_key(&key) {
        return Ok(());
    }
    let gate = pending.gate.remove(&key).unwrap();
    let up = pending.up.remove(&key).unwrap();
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
            layout: gate.layout,
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
        assert_eq!(parse_expert_bits("q2_s4").unwrap(), QuantType::Q2S4);
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
        let gate = QuantizedTensor::quantize(&[1.0; 64], 2, 32, QuantType::Q2S4, 32);
        let up = QuantizedTensor::quantize(&[2.0; 64], 2, 32, QuantType::Q2S4, 32);
        let mut pending = PendingGateUp::default();
        pending.gate.insert((0, 0), gate);
        let mut output = Vec::new();
        finish_gate_up((0, 0), &mut output, &mut pending).unwrap();
        assert!(output.is_empty());
        assert_eq!(pending.gate.len(), 1);
        pending.up.insert((0, 0), up);
        finish_gate_up((0, 0), &mut output, &mut pending).unwrap();
        assert_eq!(output[0].tensor.rows, 4);
        assert_eq!(output[0].tensor.cols, 32);
        assert_eq!(output[0].tensor.dequantize()[0], 1.0);
        assert_eq!(output[0].tensor.dequantize()[64], 2.0);
    }

    #[test]
    fn canonical_policy_fixture_seals_the_four_causal_anchors() {
        let policy: QwenExpertPolicy = serde_json::from_str(include_str!(
            "../../../benchmarks/qwen3-30b-a3b-four-layer-q3-policy.json"
        ))
        .unwrap();
        assert_eq!(policy.q3_layers, vec![0, 1, 2, 3]);
        assert_eq!(policy.validated_layers, vec![0, 1, 2, 3]);
        assert_eq!(
            policy.causal_anchors,
            vec![
                ExpertTarget {
                    layer: 1,
                    expert: 68
                },
                ExpertTarget {
                    layer: 2,
                    expert: 76
                },
                ExpertTarget {
                    layer: 2,
                    expert: 92
                },
                ExpertTarget {
                    layer: 3,
                    expert: 82
                },
            ]
        );
    }

    #[test]
    fn dequantized_student_values_survive_codec_ingestion() {
        for codec in [QuantType::Q2S4, QuantType::Q3] {
            let source = (0..512)
                .map(|index| ((index as f32 * 0.37).sin() * 3.0) + 0.25)
                .collect::<Vec<_>>();
            let trained = QuantizedTensor::quantize(&source, 2, 256, codec, 32).dequantize();
            let compiled = quantize_student_matrix(&trained, 2, 256, codec).unwrap();
            assert_eq!(compiled.dequantize(), trained);
            assert_eq!(
                compiled.layout,
                if codec == QuantType::Q3 {
                    PackingLayout::Q3Superblock112
                } else {
                    PackingLayout::RowMajor
                }
            );
        }
    }
}
