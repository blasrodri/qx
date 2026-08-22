use anyhow::{Context, Result, bail};
use clap::Parser;
use qx_compiler::QxTrace;
use qx_runtime::{ForwardTrace, GraniteModel};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "qx-parity",
    about = "Compare QX layer routing against a Granite reference trace"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    artifact: Option<PathBuf>,
    /// Use the original checkpoint's unquantized expert weights.
    #[arg(long)]
    float_reference: bool,
    #[arg(long)]
    fixture: PathBuf,
    /// QXTRACE emitted by llama.cpp with the same canonical token IDs.
    #[arg(long)]
    reference: PathBuf,
    /// Optional raw QXLOGIT file emitted by the llama.cpp parity bridge.
    #[arg(long)]
    reference_logits: Option<PathBuf>,
    #[arg(long, default_value_t = 512)]
    context: usize,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    tokens: Vec<u32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let fixture: Fixture = serde_json::from_slice(
        &std::fs::read(&args.fixture)
            .with_context(|| format!("read {}", args.fixture.display()))?,
    )?;
    if fixture.tokens.is_empty() {
        bail!("fixture has no tokens");
    }
    let reference = QxTrace::read(&args.reference)
        .with_context(|| format!("read reference {}", args.reference.display()))?;
    if reference.hidden_size != 1024 {
        bail!(
            "expected Granite hidden size 1024, got {}",
            reference.hidden_size
        );
    }
    let mut model = if args.float_reference {
        GraniteModel::from_paths_float(&args.model, args.context)
    } else {
        let artifact = args
            .artifact
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--artifact is required without --float-reference"))?;
        GraniteModel::from_paths(&args.model, artifact, args.context)
    }
    .map_err(anyhow::Error::msg)?;
    let actual = model
        .forward_trace(&fixture.tokens)
        .map_err(anyhow::Error::msg)?;
    let result = compare(&actual, &reference);
    println!("QX parity report");
    println!("fixture tokens:     {}", fixture.tokens.len());
    println!("reference records:  {}", reference.records.len());
    println!("layers compared:    {}", result.layers);
    println!("MoE input cosine:   {:.6}", result.moe_input_cosine);
    println!(
        "router agreement:   {:.2}%",
        100.0 * result.router_agreement
    );
    println!("weight cosine:      {:.6}", result.weight_cosine);
    if let Some(path) = &args.reference_logits {
        let reference_logits = read_logits(path)?;
        let cosine = cosine(&actual.logits, &reference_logits);
        let top1 = argmax(&actual.logits) == argmax(&reference_logits);
        println!("logit cosine:       {:.6}", cosine);
        println!(
            "logit top-1:        {}",
            if top1 { "MATCH" } else { "MISMATCH" }
        );
        if cosine < 0.9999 && result.first_failure.is_none() {
            // Keep the existing layer failure reporting stable while making a
            // final-logit-only mismatch fail the command as well.
            println!("first divergence:   final logits");
            println!("divergence cosine:  {:.6}", cosine);
            println!("status:             FAIL");
            std::process::exit(1);
        }
    }
    if let Some(failure) = result.first_failure {
        println!(
            "first divergence:   layer {} ({})",
            failure.layer, failure.component
        );
        println!("divergence cosine:  {:.6}", failure.cosine);
        println!("status:             FAIL");
        std::process::exit(1);
    }
    println!("status:             PASS");
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Failure {
    layer: u16,
    component: &'static str,
    cosine: f64,
}

#[derive(Clone, Copy, Debug)]
struct ResultSummary {
    layers: usize,
    moe_input_cosine: f64,
    router_agreement: f64,
    weight_cosine: f64,
    first_failure: Option<Failure>,
}

fn compare(actual: &ForwardTrace, reference: &QxTrace) -> ResultSummary {
    let mut first_failure = None;
    let mut input_cosine = 0.0;
    let mut weight_cosine = 0.0;
    let mut router_matches = 0usize;
    let mut router_total = 0usize;
    let mut layers = 0usize;
    for layer in &actual.layers {
        let records = reference
            .records
            .iter()
            .filter(|record| record.layer == layer.layer)
            .collect::<Vec<_>>();
        let target_token = (actual.tokens.len() - 1) as u16;
        let token_records = records
            .iter()
            .copied()
            .filter(|record| record.token == target_token)
            .collect::<Vec<_>>();
        // Version-1 traces have no token position. Preserve the old fallback
        // so existing one-token calibration traces remain usable.
        let prompt_records = if token_records.len() >= 8 {
            token_records
        } else {
            records
                .into_iter()
                .take(8 * actual.tokens.len())
                .collect::<Vec<_>>()
        };
        if prompt_records.len() < 8 {
            continue;
        }
        let records = &prompt_records[prompt_records.len() - 8..];
        layers += 1;
        let reference_hidden = &records[0].hidden;
        let hidden_cosine = cosine(&layer.moe_input, reference_hidden);
        input_cosine += hidden_cosine;
        if hidden_cosine < 0.9999 && first_failure.is_none() {
            first_failure = Some(Failure {
                layer: layer.layer,
                component: "moe_input",
                cosine: hidden_cosine,
            });
        }
        let reference_experts = records
            .iter()
            .map(|record| record.expert)
            .collect::<Vec<_>>();
        for (actual_expert, reference_expert) in layer.routed_experts.iter().zip(reference_experts)
        {
            router_total += 1;
            if *actual_expert == reference_expert {
                router_matches += 1;
            }
        }
        let reference_weights = records
            .iter()
            .map(|record| record.router_weight)
            .collect::<Vec<_>>();
        weight_cosine += cosine(&layer.router_weights, &reference_weights);
        if layer.routed_experts
            != [
                records[0].expert,
                records[1].expert,
                records[2].expert,
                records[3].expert,
                records[4].expert,
                records[5].expert,
                records[6].expert,
                records[7].expert,
            ]
            && first_failure.is_none()
        {
            first_failure = Some(Failure {
                layer: layer.layer,
                component: "router_top8",
                cosine: 0.0,
            });
        }
    }
    let denominator = layers.max(1) as f64;
    ResultSummary {
        layers,
        moe_input_cosine: input_cosine / denominator,
        router_agreement: router_matches as f64 / router_total.max(1) as f64,
        weight_cosine: weight_cosine / denominator,
        first_failure,
    }
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
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

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

fn read_logits(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read logits {}", path.display()))?;
    if bytes.len() < 12 || &bytes[..8] != b"QXLOGIT\0" {
        bail!("invalid QXLOGIT file");
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if bytes.len() != 12 + count * 4 {
        bail!("QXLOGIT length does not match count");
    }
    Ok(bytes[12..]
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect())
}

#[allow(dead_code)]
fn _assert_paths_exist(paths: &[&Path]) -> Result<()> {
    for path in paths {
        if !path.exists() {
            bail!("missing {}", path.display());
        }
    }
    Ok(())
}
