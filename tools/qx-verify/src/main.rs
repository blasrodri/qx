use std::path::PathBuf;
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "qx-verify",
    about = "Build and inspect QX verification commitments"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Stream the canonical V1 weight root without retaining model weights.
    ArtifactRoot {
        #[arg(value_name = "ARTIFACT")]
        artifact: PathBuf,
    },
    /// Measure the production Metal graph with and without commitments.
    MetalBenchmark {
        #[arg(value_name = "ARTIFACT")]
        artifact: PathBuf,
        #[arg(long, default_value_t = 512)]
        context: usize,
        #[arg(long, default_value_t = 16)]
        new_tokens: usize,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        repeats: usize,
        #[arg(long, default_value_t = 1)]
        checkpoint_interval: usize,
    },
    /// Size epoch audits and show the probability that cheating is sampled.
    SamplingCost {
        #[arg(long, default_value_t = 10_000)]
        receipts: u32,
        #[arg(long, default_value_t = 100)]
        cheated: u32,
        #[arg(long, value_delimiter = ',', default_value = "10,25,50,100,250,500")]
        audits: Vec<u32>,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::ArtifactRoot { artifact } => {
            let commitment = qx_commit::commit_q4_artifact(artifact)?;
            println!("weights_root={}", hex(&commitment.root));
            println!("weight_leaves={}", commitment.leaf_count);
            println!("tensors={}", commitment.tensor_count);
        }
        Command::MetalBenchmark {
            artifact,
            context,
            new_tokens,
            warmup,
            repeats,
            checkpoint_interval,
        } => metal_benchmark(
            &artifact,
            context,
            new_tokens,
            warmup,
            repeats,
            checkpoint_interval,
        )?,
        Command::SamplingCost {
            receipts,
            cheated,
            audits,
        } => sampling_cost(receipts, cheated, &audits)?,
    }
    Ok(())
}

fn sampling_cost(receipts: u32, cheated: u32, audits: &[u32]) -> Result<()> {
    use anyhow::bail;

    if receipts == 0 || cheated == 0 || cheated > receipts {
        bail!("require 0 < cheated <= receipts");
    }
    println!("epoch_receipts={receipts}");
    println!("cheated_receipts={cheated}");
    println!(
        "cheated_fraction_percent={:.6}",
        f64::from(cheated) * 100.0 / f64::from(receipts)
    );
    for &audit_count in audits {
        let probability = qx_commit::audit_detection_probability(receipts, cheated, audit_count)?;
        println!(
            "audits={audit_count} audit_fraction_percent={:.4} detection_percent={:.6} two_seed_grind_detection_percent={:.6}",
            f64::from(audit_count) * 100.0 / f64::from(receipts),
            probability * 100.0,
            probability.powi(2) * 100.0,
        );
    }
    for target in [0.9, 0.99, 0.999] {
        let required = qx_commit::required_audit_count(receipts, cheated, target)?;
        println!(
            "target_detection_percent={:.1} required_audits={required} required_fraction_percent={:.4}",
            target * 100.0,
            f64::from(required) * 100.0 / f64::from(receipts),
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn metal_benchmark(
    _artifact: &std::path::Path,
    _context: usize,
    _new_tokens: usize,
    _warmup: usize,
    _repeats: usize,
    _checkpoint_interval: usize,
) -> Result<()> {
    anyhow::bail!("Metal verification benchmarks require macOS")
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct BenchmarkSample {
    prefill: Duration,
    decode: Duration,
    total: Duration,
    executed_tokens: usize,
    generated: Vec<u32>,
    trace_root: Option<[u8; 32]>,
    metrics: qx_metal_metrics::Delta,
}

#[cfg(target_os = "macos")]
mod qx_metal_metrics {
    use qx_runtime_v2::GraniteV2Generator;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Snapshot {
        pub command_buffers: u64,
        pub dispatches: u64,
        pub gpu_seconds: f64,
        pub cpu_encode_seconds: f64,
        pub cpu_wait_seconds: f64,
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Delta {
        pub command_buffers: u64,
        pub dispatches: u64,
        pub gpu_seconds: f64,
        pub cpu_encode_seconds: f64,
        pub cpu_wait_seconds: f64,
    }

    pub fn snapshot(generator: &GraniteV2Generator) -> Result<Snapshot, String> {
        let value = generator.metrics()?;
        Ok(Snapshot {
            command_buffers: value.command_buffers,
            dispatches: value.dispatches,
            gpu_seconds: value.gpu_time_seconds,
            cpu_encode_seconds: value.cpu_encode_seconds,
            cpu_wait_seconds: value.cpu_wait_seconds,
        })
    }

    pub fn delta(before: Snapshot, after: Snapshot) -> Delta {
        Delta {
            command_buffers: after.command_buffers - before.command_buffers,
            dispatches: after.dispatches - before.dispatches,
            gpu_seconds: after.gpu_seconds - before.gpu_seconds,
            cpu_encode_seconds: after.cpu_encode_seconds - before.cpu_encode_seconds,
            cpu_wait_seconds: after.cpu_wait_seconds - before.cpu_wait_seconds,
        }
    }
}

#[cfg(target_os = "macos")]
fn metal_benchmark(
    artifact: &std::path::Path,
    context: usize,
    new_tokens: usize,
    warmup: usize,
    repeats: usize,
    checkpoint_interval: usize,
) -> Result<()> {
    use anyhow::{Context, bail};
    use qx_runtime_v2::GraniteV2Generator;

    if new_tokens == 0 || repeats == 0 {
        bail!("new-tokens and repeats must be positive");
    }
    let prompt = [1_u32, 42, 314, 2718, 17, 19, 23, 29];
    if prompt.len() + new_tokens > context {
        bail!("prompt plus generated tokens exceeds context");
    }
    let mut generator = GraniteV2Generator::load(artifact, context)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("load {}", artifact.display()))?;

    for index in 0..warmup {
        let verified = index % 2 == 1;
        let _ = run_sample(
            &mut generator,
            &prompt,
            2,
            verified,
            index as u8,
            checkpoint_interval,
        )?;
    }

    let mut baseline = Vec::with_capacity(repeats);
    let mut verified = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let baseline_sample = run_sample(
            &mut generator,
            &prompt,
            new_tokens,
            false,
            7,
            checkpoint_interval,
        )?;
        let verified_sample = run_sample(
            &mut generator,
            &prompt,
            new_tokens,
            true,
            7,
            checkpoint_interval,
        )?;
        if baseline_sample.generated != verified_sample.generated {
            bail!("verified execution changed generated tokens at repeat {repeat}");
        }
        if verified
            .first()
            .and_then(|sample: &BenchmarkSample| sample.trace_root)
            .is_some_and(|root| Some(root) != verified_sample.trace_root)
        {
            bail!("verified trace root changed at repeat {repeat}");
        }
        baseline.push(baseline_sample);
        verified.push(verified_sample);
    }

    let baseline_total = median_duration(&baseline, |sample| sample.total);
    let verified_total = median_duration(&verified, |sample| sample.total);
    let overhead = (verified_total.as_secs_f64() / baseline_total.as_secs_f64() - 1.0) * 100.0;
    println!("artifact={}", artifact.display());
    println!("context={context}");
    println!("prompt_tokens={}", prompt.len());
    println!("generated_tokens={new_tokens}");
    println!("repeats={repeats}");
    println!("checkpoint_interval={checkpoint_interval}");
    print_mode("baseline", &baseline);
    print_mode("verified", &verified);
    println!("verification_overhead_percent={overhead:.3}");
    if let Some(trace_root) = verified[0].trace_root {
        println!("trace_root={}", hex(&trace_root));
    } else {
        println!("trace_root=cost-only");
    }
    println!("output_tokens={:?}", baseline[0].generated);
    println!(
        "performance_gate={}",
        if overhead < 5.0 { "pass" } else { "fail" }
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_sample(
    generator: &mut qx_runtime_v2::GraniteV2Generator,
    prompt: &[u32],
    new_tokens: usize,
    verified: bool,
    job_tag: u8,
    checkpoint_interval: usize,
) -> Result<BenchmarkSample> {
    generator.reset();
    if verified {
        generator
            .set_verification_checkpoint_interval(checkpoint_interval)
            .map_err(anyhow::Error::msg)?;
    } else {
        generator
            .set_verification_enabled(false)
            .map_err(anyhow::Error::msg)?;
    }
    generator
        .begin_metrics_interval()
        .map_err(anyhow::Error::msg)?;
    let before = qx_metal_metrics::snapshot(generator).map_err(anyhow::Error::msg)?;
    let mut committer = verified
        .then(|| {
            qx_commit::ExecutionCommitterV1::new_with_checkpoint_interval(
                [job_tag; 32],
                24,
                generator.context_length(),
                checkpoint_interval,
            )
        })
        .transpose()?;
    let total_start = Instant::now();
    let prefill_start = Instant::now();
    let mut next = 0_u32;
    for &token in prompt {
        next = if let Some(committer) = committer.as_mut() {
            generator.step_verified(token, committer)
        } else {
            generator.step(token)
        }
        .map_err(anyhow::Error::msg)?;
    }
    let prefill = prefill_start.elapsed();
    let decode_start = Instant::now();
    let mut generated = Vec::with_capacity(new_tokens);
    for index in 0..new_tokens {
        generated.push(next);
        if index + 1 < new_tokens {
            next = if let Some(committer) = committer.as_mut() {
                generator.step_verified(next, committer)
            } else {
                generator.step(next)
            }
            .map_err(anyhow::Error::msg)?;
        }
    }
    let decode = decode_start.elapsed();
    let total = total_start.elapsed();
    let after = qx_metal_metrics::snapshot(generator).map_err(anyhow::Error::msg)?;
    let trace_root = committer
        .as_ref()
        .map(qx_commit::ExecutionCommitterV1::finish)
        .transpose()?
        .map(|trace| trace.trace_root);
    Ok(BenchmarkSample {
        prefill,
        decode,
        total,
        executed_tokens: prompt.len() + new_tokens - 1,
        generated,
        trace_root,
        metrics: qx_metal_metrics::delta(before, after),
    })
}

#[cfg(target_os = "macos")]
fn median_duration(
    samples: &[BenchmarkSample],
    field: impl Fn(&BenchmarkSample) -> Duration,
) -> Duration {
    let mut values = samples.iter().map(field).collect::<Vec<_>>();
    values.sort_unstable();
    values[values.len() / 2]
}

#[cfg(target_os = "macos")]
fn print_mode(name: &str, samples: &[BenchmarkSample]) {
    let total = median_duration(samples, |sample| sample.total);
    let prefill = median_duration(samples, |sample| sample.prefill);
    let decode = median_duration(samples, |sample| sample.decode);
    let executed = samples[0].executed_tokens;
    let median_metric = |field: fn(&qx_metal_metrics::Delta) -> f64| {
        let mut values = samples
            .iter()
            .map(|sample| field(&sample.metrics))
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let median_u64 = |field: fn(&qx_metal_metrics::Delta) -> u64| {
        let mut values = samples
            .iter()
            .map(|sample| field(&sample.metrics))
            .collect::<Vec<_>>();
        values.sort_unstable();
        values[values.len() / 2]
    };
    println!("{name}_total_ms={:.3}", total.as_secs_f64() * 1000.0);
    println!("{name}_prefill_ms={:.3}", prefill.as_secs_f64() * 1000.0);
    println!("{name}_decode_ms={:.3}", decode.as_secs_f64() * 1000.0);
    println!(
        "{name}_executed_tokens_per_second={:.3}",
        executed as f64 / total.as_secs_f64()
    );
    println!(
        "{name}_gpu_ms={:.3}",
        median_metric(|metrics| metrics.gpu_seconds) * 1000.0
    );
    println!(
        "{name}_cpu_encode_ms={:.3}",
        median_metric(|metrics| metrics.cpu_encode_seconds) * 1000.0
    );
    println!(
        "{name}_cpu_wait_ms={:.3}",
        median_metric(|metrics| metrics.cpu_wait_seconds) * 1000.0
    );
    println!(
        "{name}_command_buffers={}",
        median_u64(|metrics| metrics.command_buffers)
    );
    println!(
        "{name}_dispatches={}",
        median_u64(|metrics| metrics.dispatches)
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
