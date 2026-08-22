use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use qx_eval::{QualityTrace, TokenCorpus, read_quality_trace, write_token_corpus};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "qx-quality",
    about = "Prepare and evaluate triangular QX quality traces"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Prepare {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        domain: String,
        #[arg(long, value_enum)]
        split: Split,
        #[arg(long, default_value_t = 32_769)]
        tokens: usize,
        #[arg(long)]
        output: PathBuf,
    },
    Compare {
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        llama: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Gate {
        #[arg(long = "report", required = true)]
        reports: Vec<PathBuf>,
        #[arg(long)]
        tasks: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Split {
    Calibration,
    Heldout,
}

impl Split {
    fn label(self) -> &'static str {
        match self {
            Self::Calibration => "calibration",
            Self::Heldout => "heldout",
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TraceSummary {
    positions: u64,
    nll: f64,
    ppl: f64,
    relative_ppl_vs_reference: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct DistributionComparison {
    samples: usize,
    mean_kl: f64,
    mean_cosine: f64,
    top1_agreement: f64,
    top5_overlap: f64,
    top10_overlap: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BootstrapComparison {
    relative_ppl_delta: f64,
    lower_95: f64,
    upper_95: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct DomainQualityReport {
    schema_version: u32,
    domain: String,
    split: String,
    source_hash: String,
    tokenizer_hash: String,
    reference: TraceSummary,
    llama: TraceSummary,
    candidate: TraceSummary,
    candidate_vs_llama: BootstrapComparison,
    llama_vs_reference_logits: DistributionComparison,
    candidate_vs_reference_logits: DistributionComparison,
    candidate_vs_llama_logits: DistributionComparison,
}

#[derive(Debug, Deserialize)]
struct TaskMetrics {
    human_eval_llama: f64,
    human_eval_candidate: f64,
    human_eval_delta_lower_95: f64,
    gsm8k_llama: f64,
    gsm8k_candidate: f64,
    gsm8k_delta_lower_95: f64,
}

#[derive(Debug, Serialize)]
struct QualityGateReport {
    schema_version: u32,
    domains: usize,
    ppl_domains: Vec<String>,
    task_domains: Vec<String>,
    positions: u64,
    aggregate_relative_ppl_delta: f64,
    worst_domain_relative_ppl_delta: f64,
    worst_domain_interval_width: f64,
    confidence_resolves_one_point: bool,
    minimum_domain_positions: u64,
    recommended_tokens_per_domain: u64,
    aggregate_ppl_gate: bool,
    per_domain_ppl_gate: bool,
    human_eval_gate: Option<bool>,
    gsm8k_gate: Option<bool>,
    human_eval_delta_points_vs_llama: Option<f64>,
    gsm8k_delta_points_vs_llama: Option<f64>,
    accepted: bool,
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Prepare {
            model,
            source,
            domain,
            split,
            tokens,
            output,
        } => prepare(model, source, domain, split, tokens, output),
        Command::Compare {
            corpus,
            reference,
            llama,
            candidate,
            output,
        } => compare(corpus, reference, llama, candidate, output),
        Command::Gate {
            reports,
            tasks,
            output,
        } => gate(reports, tasks, output),
    }
}

fn prepare(
    model: PathBuf,
    source: PathBuf,
    domain: String,
    split: Split,
    tokens: usize,
    output: PathBuf,
) -> Result<()> {
    if tokens < 513 {
        bail!("quality corpora require at least 513 tokens");
    }
    let source_bytes = fs::read(&source).with_context(|| format!("read {}", source.display()))?;
    let text = String::from_utf8(source_bytes.clone())?;
    let tokenizer = qx_tokenizer::Tokenizer::from_dir(&model)?;
    let mut token_ids = tokenizer.encode(&text)?;
    if token_ids.len() < tokens {
        bail!(
            "source has {} tokens, fewer than requested {tokens}",
            token_ids.len()
        );
    }
    token_ids.truncate(tokens);
    let corpus = TokenCorpus {
        domain,
        split: split.label().to_owned(),
        source_hash: sha256(&source_bytes),
        tokenizer_hash: tokenizer_hash(&model)?,
        tokens: token_ids,
    };
    write_token_corpus(&output, &corpus)?;
    println!(
        "wrote {} {} tokens to {}",
        corpus.domain,
        corpus.tokens.len(),
        output.display()
    );
    Ok(())
}

fn compare(
    corpus_path: PathBuf,
    reference_path: PathBuf,
    llama_path: PathBuf,
    candidate_path: PathBuf,
    output: PathBuf,
) -> Result<()> {
    let corpus = qx_eval::read_token_corpus(&corpus_path)?;
    let reference = read_quality_trace(&reference_path)?;
    let llama = read_quality_trace(&llama_path)?;
    let candidate = read_quality_trace(&candidate_path)?;
    validate_aligned(&reference, &llama, &candidate, corpus.tokens.len() - 1)?;
    let reference_nll = mean_nll(&reference);
    let llama_nll = mean_nll(&llama);
    let candidate_nll = mean_nll(&candidate);
    let report = DomainQualityReport {
        schema_version: 1,
        domain: corpus.domain,
        split: corpus.split,
        source_hash: corpus.source_hash,
        tokenizer_hash: corpus.tokenizer_hash,
        reference: summary(&reference, reference_nll, reference_nll),
        llama: summary(&llama, llama_nll, reference_nll),
        candidate: summary(&candidate, candidate_nll, reference_nll),
        candidate_vs_llama: bootstrap_relative_ppl(&llama, &candidate, 10_000),
        llama_vs_reference_logits: compare_samples(&reference, &llama)?,
        candidate_vs_reference_logits: compare_samples(&reference, &candidate)?,
        candidate_vs_llama_logits: compare_samples(&llama, &candidate)?,
    };
    fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}: QX vs llama PPL {:+.3}% [{:+.3}%, {:+.3}%]",
        report.domain,
        report.candidate_vs_llama.relative_ppl_delta * 100.0,
        report.candidate_vs_llama.lower_95 * 100.0,
        report.candidate_vs_llama.upper_95 * 100.0
    );
    Ok(())
}

fn gate(reports: Vec<PathBuf>, tasks: Option<PathBuf>, output: PathBuf) -> Result<()> {
    let reports = reports
        .iter()
        .map(|path| {
            serde_json::from_slice::<DomainQualityReport>(&fs::read(path)?).map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    if reports.is_empty() || reports.iter().any(|report| report.split != "heldout") {
        bail!("quality acceptance requires heldout domain reports");
    }
    let required_domains = ["wikitext-2", "dolly", "humaneval", "gsm8k"]
        .into_iter()
        .collect::<BTreeSet<_>>();
    let actual_domains = reports
        .iter()
        .map(|report| report.domain.to_ascii_lowercase())
        .collect::<BTreeSet<_>>();
    if !required_domains
        .iter()
        .all(|required| actual_domains.contains(*required))
    {
        bail!(
            "quality gate requires held-out WikiText-2, Dolly, HumanEval, and GSM8K token streams"
        );
    }
    let text_reports = reports
        .iter()
        .filter(|report| {
            matches!(
                report.domain.to_ascii_lowercase().as_str(),
                "wikitext-2" | "dolly"
            )
        })
        .collect::<Vec<_>>();
    let task_domains = reports
        .iter()
        .filter(|report| {
            matches!(
                report.domain.to_ascii_lowercase().as_str(),
                "humaneval" | "gsm8k"
            )
        })
        .collect::<Vec<_>>();
    let minimum_domain_positions = text_reports
        .iter()
        .map(|report| report.candidate.positions)
        .min()
        .unwrap_or(0);
    if minimum_domain_positions < 32_768 {
        bail!("every held-out quality domain requires at least 32,768 scored positions");
    }
    let positions = text_reports
        .iter()
        .map(|report| report.candidate.positions)
        .sum::<u64>();
    let llama_nll = text_reports
        .iter()
        .map(|report| report.llama.nll * report.llama.positions as f64)
        .sum::<f64>()
        / positions as f64;
    let candidate_nll = text_reports
        .iter()
        .map(|report| report.candidate.nll * report.candidate.positions as f64)
        .sum::<f64>()
        / positions as f64;
    let aggregate_delta = (candidate_nll - llama_nll).exp() - 1.0;
    let worst_domain = text_reports
        .iter()
        .map(|report| report.candidate_vs_llama.relative_ppl_delta)
        .fold(f64::NEG_INFINITY, f64::max);
    let worst_domain_interval_width = reports
        .iter()
        .map(|report| report.candidate_vs_llama.upper_95 - report.candidate_vs_llama.lower_95)
        .fold(0.0_f64, f64::max);
    // A two-sided interval no wider than two percentage points resolves a
    // one-point effect in either direction around its estimate.
    let confidence_resolves_one_point = worst_domain_interval_width <= 0.02;
    let recommended_tokens_per_domain = if confidence_resolves_one_point {
        minimum_domain_positions
    } else {
        (minimum_domain_positions.saturating_mul(2)).min(131_072)
    };
    let task_metrics = tasks
        .map(|path| -> Result<TaskMetrics> {
            Ok(serde_json::from_slice::<TaskMetrics>(&fs::read(path)?)?)
        })
        .transpose()?;
    let human_eval_gate = task_metrics.as_ref().map(|metrics| {
        metrics.human_eval_candidate >= metrics.human_eval_llama - 0.02
            && metrics.human_eval_delta_lower_95 >= -2.0
    });
    let gsm8k_gate = task_metrics.as_ref().map(|metrics| {
        metrics.gsm8k_candidate >= metrics.gsm8k_llama - 0.02
            && metrics.gsm8k_delta_lower_95 >= -2.0
    });
    let aggregate_ppl_gate = aggregate_delta <= 0.01;
    let per_domain_ppl_gate = worst_domain <= 0.02;
    let accepted = aggregate_ppl_gate
        && per_domain_ppl_gate
        && confidence_resolves_one_point
        && human_eval_gate.unwrap_or(false)
        && gsm8k_gate.unwrap_or(false);
    let gate = QualityGateReport {
        schema_version: 1,
        domains: reports.len(),
        ppl_domains: text_reports
            .iter()
            .map(|report| report.domain.clone())
            .collect(),
        task_domains: task_domains
            .iter()
            .map(|report| report.domain.clone())
            .collect(),
        positions,
        aggregate_relative_ppl_delta: aggregate_delta,
        worst_domain_relative_ppl_delta: worst_domain,
        worst_domain_interval_width,
        confidence_resolves_one_point,
        minimum_domain_positions,
        recommended_tokens_per_domain,
        aggregate_ppl_gate,
        per_domain_ppl_gate,
        human_eval_gate,
        gsm8k_gate,
        human_eval_delta_points_vs_llama: task_metrics
            .as_ref()
            .map(|metrics| (metrics.human_eval_candidate - metrics.human_eval_llama) * 100.0),
        gsm8k_delta_points_vs_llama: task_metrics
            .as_ref()
            .map(|metrics| (metrics.gsm8k_candidate - metrics.gsm8k_llama) * 100.0),
        accepted,
    };
    fs::write(&output, serde_json::to_vec_pretty(&gate)?)?;
    println!("quality gate accepted: {}", gate.accepted);
    Ok(())
}

fn validate_aligned(
    a: &QualityTrace,
    b: &QualityTrace,
    c: &QualityTrace,
    positions: usize,
) -> Result<()> {
    if a.total_positions != positions as u64
        || b.total_positions != a.total_positions
        || c.total_positions != a.total_positions
        || a.vocab != b.vocab
        || a.vocab != c.vocab
        || a.blocks.len() != b.blocks.len()
        || a.blocks.len() != c.blocks.len()
    {
        bail!("quality traces are not aligned to the same corpus");
    }
    Ok(())
}

fn mean_nll(trace: &QualityTrace) -> f64 {
    trace.blocks.iter().map(|block| block.nll_sum).sum::<f64>() / trace.total_positions as f64
}

fn summary(trace: &QualityTrace, nll: f64, reference_nll: f64) -> TraceSummary {
    TraceSummary {
        positions: trace.total_positions,
        nll,
        ppl: nll.exp(),
        relative_ppl_vs_reference: (nll - reference_nll).exp() - 1.0,
    }
}

fn bootstrap_relative_ppl(
    baseline: &QualityTrace,
    candidate: &QualityTrace,
    iterations: usize,
) -> BootstrapComparison {
    let mut random = Lcg(0x7151_5eed_d15c_a11e);
    let deltas = baseline
        .blocks
        .iter()
        .zip(&candidate.blocks)
        .map(|(left, right)| (right.nll_sum - left.nll_sum, left.positions as u64))
        .collect::<Vec<_>>();
    let total_delta = deltas.iter().map(|value| value.0).sum::<f64>();
    let total_positions = deltas.iter().map(|value| value.1).sum::<u64>();
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut delta = 0.0;
        let mut positions = 0_u64;
        for _ in 0..deltas.len() {
            let value = deltas[random.next() as usize % deltas.len()];
            delta += value.0;
            positions += value.1;
        }
        samples.push((delta / positions as f64).exp() - 1.0);
    }
    samples.sort_by(f64::total_cmp);
    BootstrapComparison {
        relative_ppl_delta: (total_delta / total_positions as f64).exp() - 1.0,
        lower_95: samples[samples.len() * 25 / 1000],
        upper_95: samples[samples.len() * 975 / 1000],
    }
}

fn compare_samples(left: &QualityTrace, right: &QualityTrace) -> Result<DistributionComparison> {
    if left.samples.len() != right.samples.len() || left.samples.is_empty() {
        bail!("quality traces have different or empty logit samples");
    }
    let mut kl = 0.0;
    let mut cosine = 0.0;
    let mut top1 = 0;
    let mut top5 = 0;
    let mut top10 = 0;
    for (left, right) in left.samples.iter().zip(&right.samples) {
        if left.position != right.position || left.target != right.target {
            bail!("quality logit samples are not position-aligned");
        }
        kl += kl_divergence(&left.logits, &right.logits);
        cosine += cosine_similarity(&left.logits, &right.logits);
        top1 += usize::from(argmax(&left.logits) == argmax(&right.logits));
        top5 += top_k_overlap(&left.logits, &right.logits, 5);
        top10 += top_k_overlap(&left.logits, &right.logits, 10);
    }
    let count = left.samples.len();
    Ok(DistributionComparison {
        samples: count,
        mean_kl: kl / count as f64,
        mean_cosine: cosine / count as f64,
        top1_agreement: top1 as f64 / count as f64,
        top5_overlap: top5 as f64 / (count * 5) as f64,
        top10_overlap: top10 as f64 / (count * 10) as f64,
    })
}

fn kl_divergence(left: &[f32], right: &[f32]) -> f64 {
    let left_max = left.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let right_max = right.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let left_sum = left
        .iter()
        .map(|x| (*x - left_max).exp() as f64)
        .sum::<f64>();
    let right_sum = right
        .iter()
        .map(|x| (*x - right_max).exp() as f64)
        .sum::<f64>();
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let p = (*a - left_max).exp() as f64 / left_sum;
            let log_p = *a as f64 - left_max as f64 - left_sum.ln();
            let log_q = *b as f64 - right_max as f64 - right_sum.ln();
            p * (log_p - log_q)
        })
        .sum()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    let (dot, l2, r2) = left.iter().zip(right).fold((0.0, 0.0, 0.0), |acc, (a, b)| {
        let a = *a as f64;
        let b = *b as f64;
        (acc.0 + a * b, acc.1 + a * a, acc.2 + b * b)
    });
    dot / (l2.sqrt() * r2.sqrt()).max(f64::MIN_POSITIVE)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|value| value.0)
        .unwrap_or(0)
}

fn top_k_overlap(left: &[f32], right: &[f32], k: usize) -> usize {
    let left = top_k(left, k);
    let right = top_k(right, k);
    left.intersection(&right).count()
}

fn top_k(values: &[f32], k: usize) -> BTreeSet<usize> {
    let mut best = Vec::<(usize, f32)>::with_capacity(k);
    for (index, value) in values.iter().copied().enumerate() {
        let at = best.partition_point(|(_, current)| current >= &value);
        if at < k {
            best.insert(at, (index, value));
            best.truncate(k);
        }
    }
    best.into_iter().map(|value| value.0).collect()
}

fn tokenizer_hash(model: &std::path::Path) -> Result<String> {
    let mut digest = Sha256::new();
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ] {
        digest.update(name.as_bytes());
        digest.update(fs::read(model.join(name))?);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
}
