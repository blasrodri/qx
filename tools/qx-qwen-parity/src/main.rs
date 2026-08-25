use anyhow::{Context, Result, bail};
use clap::Parser;
use half::f16;
use qx_format::{MappedArtifact, TensorKind};
use qx_model::{ModelArchitecture, QWEN3_30B_A3B_MODEL_ID};
use qx_quant::QuantType;
use std::path::PathBuf;

const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 768;

#[derive(Debug, Parser)]
#[command(
    name = "qx-qwen-parity",
    about = "Compare one mapped Qwen Q2S4 expert between the CPU oracle and Metal"
)]
struct Args {
    artifact: PathBuf,
    #[arg(long, default_value_t = 0)]
    layer: u16,
    #[arg(long, default_value_t = 0)]
    expert: u8,
    #[arg(long, default_value_t = 0.01)]
    max_relative_l2: f32,
    #[arg(long, default_value_t = 37)]
    position: u32,
}

fn main() -> Result<()> {
    #[cfg(not(target_os = "macos"))]
    bail!("Qwen Metal parity requires macOS");
    #[cfg(target_os = "macos")]
    run()
}

#[cfg(target_os = "macos")]
fn run() -> Result<()> {
    let args = Args::parse();
    let artifact = MappedArtifact::open(&args.artifact)
        .with_context(|| format!("open {}", args.artifact.display()))?;
    validate_artifact(&artifact, &args)?;
    let gate_index = find_expert(&artifact, args.layer, args.expert, TensorKind::GateUp)?;
    let down_index = find_expert(&artifact, args.layer, args.expert, TensorKind::Down)?;
    let gate = artifact.quantized_tensor(gate_index)?;
    let down = artifact.quantized_tensor(down_index)?;

    let input_bits = (0..HIDDEN)
        .map(|index| {
            let x = index as f32;
            f16::from_f32((x * 0.0137).sin() * 0.35 + (x * 0.0031).cos() * 0.15).to_bits()
        })
        .collect::<Vec<_>>();
    let input = input_bits
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let expected = cpu_expert(&gate, &down, &input);

    let (gate_scales, gate_data) = artifact.tensor_bytes(gate_index)?;
    let (down_scales, down_data) = artifact.tensor_bytes(down_index)?;
    let context = qx_metal::platform::MetalContext::system_default()
        .context("no Metal device is available")?;
    let prepared = context
        .prepare_qwen_q2s4_expert_bytes(gate_data, gate_scales, down_data, down_scales)
        .map_err(anyhow::Error::msg)?;
    let actual_bits = context
        .run_prepared_qwen_q2s4_expert(&prepared, &input_bits)
        .map_err(anyhow::Error::msg)?;
    let actual = actual_bits
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let metrics = ErrorMetrics::between(&expected, &actual);

    let router_index = find_layer_matrix(&artifact, args.layer, TensorKind::Router, 128, HIDDEN)?;
    let router = artifact.quantized_tensor(router_index)?;
    let router_expected = router
        .mat_vec(&input)
        .into_iter()
        .map(|value| f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let router_prepared = context
        .prepare_q4_stage(&router)
        .map_err(anyhow::Error::msg)?;
    let router_actual = context
        .run_prepared_q4_stage(&router_prepared, &input_bits)
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .map(|bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let router_metrics = ErrorMetrics::between(&router_expected, &router_actual);
    let expected_route = normalized_top8(&router_expected);
    let actual_route = normalized_top8(&router_actual);

    let input_norm = raw_layer_vector(&artifact, args.layer, TensorKind::InputNorm, HIDDEN)?;
    let attention_input = rms_norm(&input, &input_norm, 1.0e-6);
    let attention_input_bits = attention_input
        .iter()
        .map(|&value| f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let attention_input = attention_input_bits
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let query = q4_projection(
        &artifact,
        &context,
        args.layer,
        TensorKind::QProj,
        4096,
        HIDDEN,
        &attention_input,
        &attention_input_bits,
    )?;
    let key = q4_projection(
        &artifact,
        &context,
        args.layer,
        TensorKind::KProj,
        512,
        HIDDEN,
        &attention_input,
        &attention_input_bits,
    )?;
    let value = q4_projection(
        &artifact,
        &context,
        args.layer,
        TensorKind::VProj,
        512,
        HIDDEN,
        &attention_input,
        &attention_input_bits,
    )?;
    let query_norm = raw_layer_vector(&artifact, args.layer, TensorKind::QueryNorm, 128)?;
    let key_norm = raw_layer_vector(&artifact, args.layer, TensorKind::KeyNorm, 128)?;
    let (expected_query, expected_key) = cpu_qk_norm_rope(
        &query.actual_bits,
        &key.actual_bits,
        &query_norm,
        &key_norm,
        args.position,
    );
    let (actual_query, actual_key) = context
        .run_qwen_qk_norm_rope(
            &query.actual_bits,
            &key.actual_bits,
            &query_norm,
            &key_norm,
            args.position,
        )
        .map_err(anyhow::Error::msg)?;
    let qk_expected = expected_query
        .iter()
        .chain(&expected_key)
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let qk_actual = actual_query
        .iter()
        .chain(&actual_key)
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let qk_metrics = ErrorMetrics::between(&qk_expected, &qk_actual);
    let context_len = args.position as usize + 1;
    let mut key_cache = deterministic_kv_cache(context_len, 0.017, 0.11);
    let mut value_cache = deterministic_kv_cache(context_len, 0.009, -0.07);
    let current_offset = args.position as usize * 512;
    key_cache[current_offset..current_offset + 512].copy_from_slice(&actual_key);
    value_cache[current_offset..current_offset + 512].copy_from_slice(&value.actual_bits);
    let attention_expected = cpu_gqa_attention(&actual_query, &key_cache, &value_cache);
    let attention_actual = context
        .run_qwen_gqa_attention(&actual_query, &key_cache, &value_cache)
        .map_err(anyhow::Error::msg)?;
    let attention_expected_f32 = attention_expected
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let attention_actual_f32 = attention_actual
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let attention_metrics = ErrorMetrics::between(&attention_expected_f32, &attention_actual_f32);
    let output_projection = q4_projection(
        &artifact,
        &context,
        args.layer,
        TensorKind::OProj,
        HIDDEN,
        4096,
        &attention_actual_f32,
        &attention_actual,
    )?;
    let hidden_after_attention_bits = input
        .iter()
        .zip(&output_projection.actual_bits)
        .map(|(&residual, &projected)| {
            f16::from_f32(residual + f16::from_bits(projected).to_f32()).to_bits()
        })
        .collect::<Vec<_>>();
    let hidden_after_attention = hidden_after_attention_bits
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let post_attention_norm =
        raw_layer_vector(&artifact, args.layer, TensorKind::PostAttentionNorm, HIDDEN)?;
    let moe_input = rms_norm(&hidden_after_attention, &post_attention_norm, 1.0e-6);
    let moe_input_bits = moe_input
        .iter()
        .map(|&value| f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    let moe_input = moe_input_bits
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let post_router_expected = router
        .mat_vec(&moe_input)
        .into_iter()
        .map(|value| f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let post_router_actual_bits = context
        .run_prepared_q4_stage(&router_prepared, &moe_input_bits)
        .map_err(anyhow::Error::msg)?;
    let post_router_actual = post_router_actual_bits
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let post_router_metrics = ErrorMetrics::between(&post_router_expected, &post_router_actual);
    let post_expected_route = normalized_top8(&post_router_expected);
    let post_actual_route = normalized_top8(&post_router_actual);
    if post_expected_route
        .iter()
        .map(|item| item.0)
        .collect::<Vec<_>>()
        != post_actual_route
            .iter()
            .map(|item| item.0)
            .collect::<Vec<_>>()
    {
        bail!("post-attention router selected different CPU and Metal experts");
    }

    let mut prepared_experts = Vec::with_capacity(8);
    let mut cpu_expert_outputs = Vec::with_capacity(8);
    for &(expert, _) in &post_actual_route {
        let expert = u8::try_from(expert).expect("Qwen expert ID fits u8");
        let gate_index = find_expert(&artifact, args.layer, expert, TensorKind::GateUp)?;
        let down_index = find_expert(&artifact, args.layer, expert, TensorKind::Down)?;
        let gate = artifact.quantized_tensor(gate_index)?;
        let down = artifact.quantized_tensor(down_index)?;
        cpu_expert_outputs.push(cpu_expert(&gate, &down, &moe_input));
        let (gate_scales, gate_data) = artifact.tensor_bytes(gate_index)?;
        let (down_scales, down_data) = artifact.tensor_bytes(down_index)?;
        prepared_experts.push(
            context
                .prepare_qwen_q2s4_expert_bytes(gate_data, gate_scales, down_data, down_scales)
                .map_err(anyhow::Error::msg)?,
        );
    }
    let route_weights: [f32; 8] = post_actual_route
        .iter()
        .map(|item| item.1)
        .collect::<Vec<_>>()
        .try_into()
        .expect("top-8 route has eight weights");
    let prepared_refs = prepared_experts.iter().collect::<Vec<_>>();
    let moe_actual_bits = context
        .run_prepared_qwen_q2s4_experts_weighted(&prepared_refs, &moe_input_bits, &route_weights)
        .map_err(anyhow::Error::msg)?;
    let moe_expected = (0..HIDDEN)
        .map(|row| {
            let value = cpu_expert_outputs
                .iter()
                .zip(route_weights)
                .map(|(output, weight)| output[row] * weight)
                .sum::<f32>();
            f16::from_f32(value).to_f32()
        })
        .collect::<Vec<_>>();
    let moe_actual = moe_actual_bits
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    let moe_metrics = ErrorMetrics::between(&moe_expected, &moe_actual);

    println!("QX Qwen expert parity\n");
    println!("artifact:      {}", args.artifact.display());
    println!("layer/expert:  {}/{}", args.layer, args.expert);
    println!("gate/up:       {} x {}", gate.rows, gate.cols);
    println!("down:          {} x {}", down.rows, down.cols);
    println!("max abs error: {:.8}", metrics.max_abs);
    println!("RMSE:          {:.8}", metrics.rmse);
    println!("relative L2:   {:.8}", metrics.relative_l2);
    println!("cosine:        {:.8}", metrics.cosine);
    if metrics.relative_l2 > args.max_relative_l2 {
        bail!(
            "Qwen expert parity failed: relative L2 {:.8} exceeds {:.8}",
            metrics.relative_l2,
            args.max_relative_l2
        );
    }
    println!("expert parity: pass\n");
    println!("router relative L2: {:.8}", router_metrics.relative_l2);
    println!("router cosine:      {:.8}", router_metrics.cosine);
    println!(
        "CPU top-8:          {:?}",
        expected_route.iter().map(|item| item.0).collect::<Vec<_>>()
    );
    println!(
        "Metal top-8:        {:?}",
        actual_route.iter().map(|item| item.0).collect::<Vec<_>>()
    );
    if expected_route.iter().map(|item| item.0).collect::<Vec<_>>()
        != actual_route.iter().map(|item| item.0).collect::<Vec<_>>()
    {
        bail!("Qwen router parity failed: top-8 expert IDs differ");
    }
    let route_weight_error = expected_route
        .iter()
        .zip(&actual_route)
        .map(|(expected, actual)| (expected.1 - actual.1).abs())
        .fold(0.0_f32, f32::max);
    println!("route weight max Δ: {:.8}", route_weight_error);
    if router_metrics.relative_l2 > args.max_relative_l2 || route_weight_error > 0.001 {
        bail!("Qwen router numerical parity failed");
    }
    println!("router parity:       pass");
    println!();
    println!("Q projection L2:     {:.8}", query.metrics.relative_l2);
    println!("K projection L2:     {:.8}", key.metrics.relative_l2);
    println!("V projection L2:     {:.8}", value.metrics.relative_l2);
    println!("Q/K norm+RoPE pos:   {}", args.position);
    println!("Q/K norm+RoPE L2:    {:.8}", qk_metrics.relative_l2);
    println!("Q/K norm+RoPE cosine:{:.8}", qk_metrics.cosine);
    println!("GQA context:          {}", context_len);
    println!("GQA relative L2:      {:.8}", attention_metrics.relative_l2);
    println!("GQA cosine:           {:.8}", attention_metrics.cosine);
    println!(
        "O projection L2:     {:.8}",
        output_projection.metrics.relative_l2
    );
    if [
        query.metrics.relative_l2,
        key.metrics.relative_l2,
        value.metrics.relative_l2,
        qk_metrics.relative_l2,
        attention_metrics.relative_l2,
        output_projection.metrics.relative_l2,
    ]
    .into_iter()
    .any(|error| error > args.max_relative_l2)
    {
        bail!("Qwen attention front-end parity failed");
    }
    println!("attention front end: pass");
    println!();
    println!(
        "post-attn router L2: {:.8}",
        post_router_metrics.relative_l2
    );
    println!(
        "post-attn top-8:      {:?}",
        post_actual_route
            .iter()
            .map(|item| item.0)
            .collect::<Vec<_>>()
    );
    println!("weighted MoE L2:     {:.8}", moe_metrics.relative_l2);
    println!("weighted MoE cosine: {:.8}", moe_metrics.cosine);
    if post_router_metrics.relative_l2 > args.max_relative_l2
        || moe_metrics.relative_l2 > args.max_relative_l2
    {
        bail!("Qwen post-attention MoE parity failed");
    }
    println!("layer-{} components: pass", args.layer);
    Ok(())
}

fn validate_artifact(artifact: &MappedArtifact, args: &Args) -> Result<()> {
    let info = artifact.info();
    if info.header.model_id != QWEN3_30B_A3B_MODEL_ID {
        bail!("artifact is not {QWEN3_30B_A3B_MODEL_ID}");
    }
    let shape = info
        .execution_plan
        .as_ref()
        .and_then(|plan| plan.model_shape.as_ref())
        .context("artifact does not seal a model shape")?;
    if shape.architecture != ModelArchitecture::Qwen3Moe
        || shape.hidden_size != HIDDEN
        || shape.expert_intermediate != INTERMEDIATE
    {
        bail!("artifact has unsupported Qwen shape {shape:?}");
    }
    if usize::from(args.layer) >= shape.layers || usize::from(args.expert) >= shape.experts {
        bail!("requested layer/expert is outside the sealed model shape");
    }
    Ok(())
}

fn find_expert(
    artifact: &MappedArtifact,
    layer: u16,
    expert: u8,
    kind: TensorKind,
) -> Result<usize> {
    let index = artifact
        .info()
        .entries
        .iter()
        .position(|entry| entry.layer == layer && entry.expert == expert && entry.kind == kind)
        .with_context(|| format!("missing {kind:?} tensor for layer {layer}, expert {expert}"))?;
    let entry = &artifact.info().entries[index];
    let expected = match kind {
        TensorKind::GateUp => (INTERMEDIATE * 2, HIDDEN),
        TensorKind::Down => (HIDDEN, INTERMEDIATE),
        _ => unreachable!(),
    };
    if entry.quant_type != QuantType::Q2S4
        || entry.group_size != 32
        || entry.layout != qx_quant::PackingLayout::RowMajor
        || (entry.rows as usize, entry.cols as usize) != expected
    {
        bail!("unsupported Qwen expert entry {entry:?}");
    }
    Ok(index)
}

fn find_layer_matrix(
    artifact: &MappedArtifact,
    layer: u16,
    kind: TensorKind,
    rows: usize,
    cols: usize,
) -> Result<usize> {
    let index = artifact
        .info()
        .entries
        .iter()
        .position(|entry| entry.layer == layer && entry.expert == u8::MAX && entry.kind == kind)
        .with_context(|| format!("missing {kind:?} tensor for layer {layer}"))?;
    let entry = &artifact.info().entries[index];
    if entry.quant_type != QuantType::Q4
        || entry.group_size != 32
        || entry.layout != qx_quant::PackingLayout::RowMajor
        || (entry.rows as usize, entry.cols as usize) != (rows, cols)
    {
        bail!("unsupported Qwen layer matrix entry {entry:?}");
    }
    Ok(index)
}

fn raw_layer_vector(
    artifact: &MappedArtifact,
    layer: u16,
    kind: TensorKind,
    elements: usize,
) -> Result<Vec<f32>> {
    let index = artifact
        .info()
        .entries
        .iter()
        .position(|entry| entry.layer == layer && entry.expert == u8::MAX && entry.kind == kind)
        .with_context(|| format!("missing {kind:?} vector for layer {layer}"))?;
    let values = artifact.raw_tensor_f32(index)?;
    if values.len() != elements {
        bail!(
            "{kind:?} vector has {} elements, expected {elements}",
            values.len()
        );
    }
    Ok(values)
}

struct ProjectionParity {
    actual_bits: Vec<u16>,
    metrics: ErrorMetrics,
}

fn q4_projection(
    artifact: &MappedArtifact,
    context: &qx_metal::platform::MetalContext,
    layer: u16,
    kind: TensorKind,
    rows: usize,
    cols: usize,
    input: &[f32],
    input_bits: &[u16],
) -> Result<ProjectionParity> {
    let index = find_layer_matrix(artifact, layer, kind, rows, cols)?;
    let weights = artifact.quantized_tensor(index)?;
    let expected = weights
        .mat_vec(input)
        .into_iter()
        .map(|value| f16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let prepared = context
        .prepare_q4_stage(&weights)
        .map_err(anyhow::Error::msg)?;
    let actual_bits = context
        .run_prepared_q4_stage(&prepared, input_bits)
        .map_err(anyhow::Error::msg)?;
    let actual = actual_bits
        .iter()
        .map(|&bits| f16::from_bits(bits).to_f32())
        .collect::<Vec<_>>();
    Ok(ProjectionParity {
        actual_bits,
        metrics: ErrorMetrics::between(&expected, &actual),
    })
}

fn rms_norm(input: &[f32], weights: &[f32], epsilon: f32) -> Vec<f32> {
    assert_eq!(input.len(), weights.len());
    let inverse_rms = (input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32
        + epsilon)
        .sqrt()
        .recip();
    input
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| value * inverse_rms * weight)
        .collect()
}

fn cpu_qk_norm_rope(
    query: &[u16],
    key: &[u16],
    query_norm: &[f32],
    key_norm: &[f32],
    position: u32,
) -> (Vec<u16>, Vec<u16>) {
    fn transform(values: &[u16], heads: usize, norm: &[f32], position: u32) -> Vec<u16> {
        let mut output = values.to_vec();
        for head in 0..heads {
            let offset = head * 128;
            let inverse_rms = (values[offset..offset + 128]
                .iter()
                .map(|&bits| {
                    let value = f16::from_bits(bits).to_f32();
                    value * value
                })
                .sum::<f32>()
                / 128.0
                + 1.0e-6)
                .sqrt()
                .recip();
            for index in 0..64 {
                let left =
                    f16::from_bits(values[offset + index]).to_f32() * inverse_rms * norm[index];
                let right = f16::from_bits(values[offset + 64 + index]).to_f32()
                    * inverse_rms
                    * norm[64 + index];
                let angle = position as f32 * 1_000_000.0_f32.powf(-(index as f32) / 64.0);
                let (sine, cosine) = angle.sin_cos();
                output[offset + index] = f16::from_f32(left * cosine - right * sine).to_bits();
                output[offset + 64 + index] = f16::from_f32(right * cosine + left * sine).to_bits();
            }
        }
        output
    }
    (
        transform(query, 32, query_norm, position),
        transform(key, 4, key_norm, position),
    )
}

fn deterministic_kv_cache(context: usize, frequency: f32, bias: f32) -> Vec<u16> {
    (0..context * 512)
        .map(|index| f16::from_f32((index as f32 * frequency).sin() * 0.2 + bias).to_bits())
        .collect()
}

fn cpu_gqa_attention(query: &[u16], key_cache: &[u16], value_cache: &[u16]) -> Vec<u16> {
    const HEAD_DIM: usize = 128;
    const KV_HEADS: usize = 4;
    let context = key_cache.len() / (KV_HEADS * HEAD_DIM);
    let mut output = vec![0_u16; 32 * HEAD_DIM];
    for query_head in 0..32 {
        let key_head = query_head / 8;
        let query_offset = query_head * HEAD_DIM;
        let mut scores = Vec::with_capacity(context);
        for token in 0..context {
            let key_offset = (token * KV_HEADS + key_head) * HEAD_DIM;
            let score = (0..HEAD_DIM)
                .map(|column| {
                    f16::from_bits(query[query_offset + column]).to_f32()
                        * f16::from_bits(key_cache[key_offset + column]).to_f32()
                })
                .sum::<f32>()
                / (HEAD_DIM as f32).sqrt();
            scores.push(score);
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f32>();
        for dimension in 0..HEAD_DIM {
            let value = weights
                .iter()
                .enumerate()
                .map(|(token, weight)| {
                    let offset = (token * KV_HEADS + key_head) * HEAD_DIM + dimension;
                    weight * f16::from_bits(value_cache[offset]).to_f32()
                })
                .sum::<f32>()
                / denominator;
            output[query_offset + dimension] = f16::from_f32(value).to_bits();
        }
    }
    output
}

fn normalized_top8(logits: &[f32]) -> Vec<(usize, f32)> {
    assert_eq!(logits.len(), 128);
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let probabilities = logits
        .iter()
        .map(|&value| (value - maximum).exp())
        .collect::<Vec<_>>();
    let mut selected = (0..logits.len()).collect::<Vec<_>>();
    selected.sort_unstable_by(|&left, &right| {
        probabilities[right]
            .total_cmp(&probabilities[left])
            .then_with(|| left.cmp(&right))
    });
    selected.truncate(8);
    let selected_sum = selected
        .iter()
        .map(|&index| probabilities[index])
        .sum::<f32>();
    selected
        .into_iter()
        .map(|index| (index, probabilities[index] / selected_sum))
        .collect()
}

fn cpu_expert(
    gate: &qx_quant::QuantizedTensor,
    down: &qx_quant::QuantizedTensor,
    input: &[f32],
) -> Vec<f32> {
    let gate_up = gate.mat_vec(input);
    let activated = (0..INTERMEDIATE)
        .map(|index| {
            let gate = f16::from_f32(gate_up[index]).to_f32();
            let up = f16::from_f32(gate_up[INTERMEDIATE + index]).to_f32();
            f16::from_f32((gate / (1.0 + (-gate).exp())) * up).to_f32()
        })
        .collect::<Vec<_>>();
    down.mat_vec(&activated)
        .into_iter()
        .map(|value| f16::from_f32(value).to_f32())
        .collect()
}

#[derive(Debug)]
struct ErrorMetrics {
    max_abs: f32,
    rmse: f32,
    relative_l2: f32,
    cosine: f32,
}

impl ErrorMetrics {
    fn between(expected: &[f32], actual: &[f32]) -> Self {
        assert_eq!(expected.len(), actual.len());
        let mut max_abs = 0.0_f32;
        let mut squared_error = 0.0_f64;
        let mut expected_energy = 0.0_f64;
        let mut actual_energy = 0.0_f64;
        let mut dot = 0.0_f64;
        for (&expected, &actual) in expected.iter().zip(actual) {
            let error = f64::from(expected - actual);
            max_abs = max_abs.max((expected - actual).abs());
            squared_error += error * error;
            expected_energy += f64::from(expected) * f64::from(expected);
            actual_energy += f64::from(actual) * f64::from(actual);
            dot += f64::from(expected) * f64::from(actual);
        }
        Self {
            max_abs,
            rmse: (squared_error / expected.len() as f64).sqrt() as f32,
            relative_l2: (squared_error / expected_energy).sqrt() as f32,
            cosine: (dot / (expected_energy * actual_energy).sqrt()) as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_have_exact_metrics() {
        let metrics = ErrorMetrics::between(&[1.0, -2.0], &[1.0, -2.0]);
        assert_eq!(metrics.max_abs, 0.0);
        assert_eq!(metrics.relative_l2, 0.0);
        assert!((metrics.cosine - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalized_top8_is_stable_and_sums_to_one() {
        let logits = (0..128).map(|index| index as f32).collect::<Vec<_>>();
        let route = normalized_top8(&logits);
        assert_eq!(
            route.iter().map(|item| item.0).collect::<Vec<_>>(),
            (120..128).rev().collect::<Vec<_>>()
        );
        assert!((route.iter().map(|item| item.1).sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
