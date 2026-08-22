use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, io::BufRead, path::PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "qx-task-eval",
    about = "Compare paired greedy HumanEval/GSM8K task-result records"
)]
struct Args {
    #[arg(long)]
    human_eval_llama: PathBuf,
    #[arg(long)]
    human_eval_candidate: PathBuf,
    #[arg(long)]
    gsm8k_llama: PathBuf,
    #[arg(long)]
    gsm8k_candidate: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 10_000)]
    bootstrap_iterations: usize,
}

#[derive(Debug, Deserialize)]
struct TaskResult {
    task_id: String,
    /// HumanEval adapters should populate this from the canonical execution
    /// harness. GSM8K adapters may provide it directly or provide prediction
    /// and reference strings for deterministic numeric normalization here.
    passed: Option<bool>,
    prediction: Option<String>,
    reference: Option<String>,
}

#[derive(Debug, Serialize)]
struct TaskMetrics {
    schema_version: u32,
    human_eval_tasks: usize,
    human_eval_llama: f64,
    human_eval_candidate: f64,
    human_eval_llama_lower_95: f64,
    human_eval_llama_upper_95: f64,
    human_eval_candidate_lower_95: f64,
    human_eval_candidate_upper_95: f64,
    human_eval_delta_points: f64,
    human_eval_delta_lower_95: f64,
    human_eval_delta_upper_95: f64,
    gsm8k_tasks: usize,
    gsm8k_llama: f64,
    gsm8k_candidate: f64,
    gsm8k_llama_lower_95: f64,
    gsm8k_llama_upper_95: f64,
    gsm8k_candidate_lower_95: f64,
    gsm8k_candidate_upper_95: f64,
    gsm8k_delta_points: f64,
    gsm8k_delta_lower_95: f64,
    gsm8k_delta_upper_95: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.bootstrap_iterations < 1000 {
        bail!("task confidence intervals require at least 1000 bootstrap iterations");
    }
    let human = compare_task(
        &args.human_eval_llama,
        &args.human_eval_candidate,
        false,
        args.bootstrap_iterations,
        0x4855_4d41_4e45_5641,
    )?;
    let gsm = compare_task(
        &args.gsm8k_llama,
        &args.gsm8k_candidate,
        true,
        args.bootstrap_iterations,
        0x4753_4d38_4b45_5641,
    )?;
    let report = TaskMetrics {
        schema_version: 1,
        human_eval_tasks: human.tasks,
        human_eval_llama: human.baseline,
        human_eval_candidate: human.candidate,
        human_eval_llama_lower_95: human.baseline_lower,
        human_eval_llama_upper_95: human.baseline_upper,
        human_eval_candidate_lower_95: human.candidate_lower,
        human_eval_candidate_upper_95: human.candidate_upper,
        human_eval_delta_points: 100.0 * (human.candidate - human.baseline),
        human_eval_delta_lower_95: 100.0 * human.delta_lower,
        human_eval_delta_upper_95: 100.0 * human.delta_upper,
        gsm8k_tasks: gsm.tasks,
        gsm8k_llama: gsm.baseline,
        gsm8k_candidate: gsm.candidate,
        gsm8k_llama_lower_95: gsm.baseline_lower,
        gsm8k_llama_upper_95: gsm.baseline_upper,
        gsm8k_candidate_lower_95: gsm.candidate_lower,
        gsm8k_candidate_upper_95: gsm.candidate_upper,
        gsm8k_delta_points: 100.0 * (gsm.candidate - gsm.baseline),
        gsm8k_delta_lower_95: 100.0 * gsm.delta_lower,
        gsm8k_delta_upper_95: 100.0 * gsm.delta_upper,
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", args.output.display()))?;
    println!(
        "HumanEval QX-llama {:+.2} points; GSM8K {:+.2} points",
        report.human_eval_delta_points, report.gsm8k_delta_points
    );
    Ok(())
}

struct Comparison {
    tasks: usize,
    baseline: f64,
    candidate: f64,
    baseline_lower: f64,
    baseline_upper: f64,
    candidate_lower: f64,
    candidate_upper: f64,
    delta_lower: f64,
    delta_upper: f64,
}

fn compare_task(
    baseline_path: &PathBuf,
    candidate_path: &PathBuf,
    allow_numeric_scoring: bool,
    iterations: usize,
    seed: u64,
) -> Result<Comparison> {
    let baseline = read_results(baseline_path, allow_numeric_scoring)?;
    let candidate = read_results(candidate_path, allow_numeric_scoring)?;
    if baseline.keys().ne(candidate.keys()) || baseline.is_empty() {
        bail!(
            "paired task files {} and {} have different or empty task IDs",
            baseline_path.display(),
            candidate_path.display()
        );
    }
    let pairs = baseline
        .iter()
        .map(|(task, baseline)| (*baseline, candidate[task]))
        .collect::<Vec<_>>();
    let baseline_accuracy = mean(pairs.iter().map(|pair| pair.0));
    let candidate_accuracy = mean(pairs.iter().map(|pair| pair.1));
    let mut random = Lcg(seed);
    let mut baseline_samples = Vec::with_capacity(iterations);
    let mut candidate_samples = Vec::with_capacity(iterations);
    let mut delta_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let sample = (0..pairs.len())
            .map(|_| pairs[random.next() as usize % pairs.len()])
            .collect::<Vec<_>>();
        let left = mean(sample.iter().map(|pair| pair.0));
        let right = mean(sample.iter().map(|pair| pair.1));
        baseline_samples.push(left);
        candidate_samples.push(right);
        delta_samples.push(right - left);
    }
    baseline_samples.sort_by(f64::total_cmp);
    candidate_samples.sort_by(f64::total_cmp);
    delta_samples.sort_by(f64::total_cmp);
    let low = iterations * 25 / 1000;
    let high = (iterations * 975 / 1000).min(iterations - 1);
    Ok(Comparison {
        tasks: pairs.len(),
        baseline: baseline_accuracy,
        candidate: candidate_accuracy,
        baseline_lower: baseline_samples[low],
        baseline_upper: baseline_samples[high],
        candidate_lower: candidate_samples[low],
        candidate_upper: candidate_samples[high],
        delta_lower: delta_samples[low],
        delta_upper: delta_samples[high],
    })
}

fn read_results(path: &PathBuf, allow_numeric_scoring: bool) -> Result<BTreeMap<String, f64>> {
    let file = fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut results = BTreeMap::new();
    for (line, value) in std::io::BufReader::new(file).lines().enumerate() {
        let value = value?;
        if value.trim().is_empty() {
            continue;
        }
        let record: TaskResult = serde_json::from_str(&value)
            .with_context(|| format!("parse {} line {}", path.display(), line + 1))?;
        let passed = if let Some(passed) = record.passed {
            passed
        } else if allow_numeric_scoring {
            match (
                normalize_number(record.prediction.as_deref().unwrap_or("")),
                normalize_number(record.reference.as_deref().unwrap_or("")),
            ) {
                (Some(prediction), Some(reference)) => prediction == reference,
                _ => false,
            }
        } else {
            bail!(
                "HumanEval record {} lacks canonical passed result",
                record.task_id
            );
        };
        if results
            .insert(record.task_id.clone(), f64::from(passed))
            .is_some()
        {
            bail!("duplicate task ID {} in {}", record.task_id, path.display());
        }
    }
    Ok(results)
}

fn normalize_number(value: &str) -> Option<String> {
    let marker = value
        .rsplit_once("####")
        .map(|(_, value)| value)
        .unwrap_or(value);
    marker
        .split_whitespace()
        .rev()
        .map(|value| {
            value
                .trim_matches(|character: char| {
                    !character.is_ascii_digit() && !matches!(character, '-' | '.' | ',')
                })
                .replace(',', "")
        })
        .find(|value| value.parse::<f64>().is_ok())
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    values.iter().sum::<f64>() / values.len() as f64
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::normalize_number;

    #[test]
    fn gsm8k_numeric_normalization_requires_an_actual_number() {
        assert_eq!(normalize_number("there is no answer"), None);
        assert_eq!(normalize_number("work #### 1,024"), Some("1024".to_owned()));
        assert_eq!(normalize_number("answer: -3.5"), Some("-3.5".to_owned()));
    }
}
