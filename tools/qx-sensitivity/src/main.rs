use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use qx_compiler::granite_memory_plan;
use qx_format::{CompiledTensor, MappedArtifact};
use qx_quant::{QuantType, QuantizedTensor};
use qx_runtime::GraniteModel;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "qx-sensitivity",
    about = "Measure downstream PPL cost of individual Q3 downgrades"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long)]
    quality_text: PathBuf,
    #[arg(long, default_value_t = 512)]
    quality_tokens: usize,
    #[arg(long, default_value_t = 512)]
    context: usize,
    /// Candidate expert coordinates such as 0:3. Repeat for multiple candidates.
    #[arg(long = "candidate", value_parser = parse_candidate)]
    candidates: Vec<(u16, u8)>,
    /// Which expert matrix to downgrade for each candidate.
    #[arg(long, value_enum, default_value_t = Matrix::Both)]
    matrix: Matrix,
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Matrix {
    GateUp,
    Down,
    Both,
}

impl Matrix {
    fn matches(self, kind: qx_format::TensorKind) -> bool {
        matches!(
            (self, kind),
            (Self::GateUp, qx_format::TensorKind::GateUp)
                | (Self::Down, qx_format::TensorKind::Down)
                | (Self::Both, _)
        )
    }

    fn label(self) -> &'static str {
        match self {
            Self::GateUp => "gate_up",
            Self::Down => "down",
            Self::Both => "both",
        }
    }
}

fn parse_candidate(value: &str) -> Result<(u16, u8), String> {
    let (layer, expert) = value
        .split_once(':')
        .ok_or_else(|| "candidate must be layer:expert".to_owned())?;
    Ok((
        layer.parse().map_err(|_| "invalid layer".to_owned())?,
        expert.parse().map_err(|_| "invalid expert".to_owned())?,
    ))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.candidates.is_empty() {
        bail!("provide at least one --candidate layer:expert");
    }
    std::fs::create_dir_all(
        args.output
            .parent()
            .ok_or_else(|| anyhow::anyhow!("output must include a directory"))?,
    )?;
    let quality_text = std::fs::read_to_string(&args.quality_text)
        .with_context(|| format!("read {}", args.quality_text.display()))?;
    let baseline_artifact = MappedArtifact::open(&args.baseline)?;
    let baseline_bytes = baseline_artifact.info().memory_plan.weights_bytes;
    let mut baseline_model = GraniteModel::from_paths(&args.model, &args.baseline, args.context)
        .map_err(anyhow::Error::msg)?;
    let baseline_ppl = baseline_model
        .perplexity(&quality_text, args.quality_tokens)
        .map_err(anyhow::Error::msg)?;
    let mut csv = String::from(
        "layer,expert,matrix,baseline_ppl,candidate_ppl,delta_ppl,delta_nll,bytes_saved,artifact\n",
    );
    println!("QX model-level sensitivity");
    println!("baseline PPL: {:.6}", baseline_ppl);
    for (layer, expert) in args.candidates {
        let candidate_path = candidate_path(&args.output, layer, expert);
        write_candidate(
            &baseline_artifact,
            layer,
            expert,
            args.matrix,
            &candidate_path,
        )?;
        let mut candidate_model =
            GraniteModel::from_paths(&args.model, &candidate_path, args.context)
                .map_err(anyhow::Error::msg)?;
        let candidate_ppl = candidate_model
            .perplexity(&quality_text, args.quality_tokens)
            .map_err(anyhow::Error::msg)?;
        let candidate_artifact = MappedArtifact::open(&candidate_path)?;
        let candidate_bytes = candidate_artifact.info().memory_plan.weights_bytes;
        let delta_ppl = candidate_ppl - baseline_ppl;
        let delta_nll = (candidate_ppl / baseline_ppl).ln();
        println!(
            "layer {:02} expert {:02} {}: PPL {:.6} delta {:+.6} ({:+.2}%), saved {} bytes",
            layer,
            expert,
            args.matrix.label(),
            candidate_ppl,
            delta_ppl,
            100.0 * delta_ppl / baseline_ppl,
            baseline_bytes.saturating_sub(candidate_bytes),
        );
        writeln!(
            csv,
            "{layer},{expert},{},{baseline_ppl:.8},{candidate_ppl:.8},{delta_ppl:.8},{delta_nll:.8},{},{}",
            args.matrix.label(),
            baseline_bytes.saturating_sub(candidate_bytes),
            candidate_path.display(),
        )?;
    }
    std::fs::write(&args.output, csv)?;
    println!("report: {}", args.output.display());
    Ok(())
}

fn candidate_path(output: &Path, layer: u16, expert: u8) -> PathBuf {
    output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("candidate-q3-layer-{layer}-expert-{expert}.qx"))
}

fn write_candidate(
    baseline: &MappedArtifact,
    layer: u16,
    expert: u8,
    matrix: Matrix,
    output: &Path,
) -> Result<()> {
    let mut tensors = Vec::with_capacity(baseline.info().entries.len());
    for (index, entry) in baseline.info().entries.iter().enumerate() {
        let mut tensor = baseline.quantized_tensor(index)?;
        if entry.layer == layer && entry.expert == expert && matrix.matches(entry.kind) {
            if tensor.quant_type != QuantType::Q4 {
                bail!("candidate {}:{} is not Q4 in baseline", layer, expert);
            }
            let values = tensor.dequantize();
            tensor = QuantizedTensor::quantize_for_layout(
                &values,
                tensor.rows,
                tensor.cols,
                QuantType::Q3,
                tensor.group_size,
                tensor.layout,
            )
            .map_err(anyhow::Error::msg)?;
        }
        tensors.push(CompiledTensor {
            layer: entry.layer,
            expert: entry.expert,
            kind: entry.kind,
            tensor,
        });
    }
    let bytes = tensors
        .iter()
        .map(|tensor| tensor.tensor.bytes())
        .sum::<usize>();
    let plan = granite_memory_plan(bytes, baseline.info().header.context_limit as usize);
    qx_format::write_artifact(
        output,
        &baseline.info().header.model_id,
        &baseline.info().header.target_id,
        baseline.info().header.context_limit,
        plan,
        &tensors,
    )?;
    Ok(())
}
