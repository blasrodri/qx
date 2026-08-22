use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};

#[derive(Debug, Parser)]
#[command(
    name = "qx-validate",
    about = "Run randomized paired QX/llama validation blocks"
)]
struct Args {
    #[arg(long)]
    contract: PathBuf,
    #[arg(long, default_value = "target/release/qx-bench")]
    qx_bench: PathBuf,
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    llama_simple: PathBuf,
    #[arg(long)]
    gguf: PathBuf,
    #[arg(long, default_value = "external/models/granite-3.0-1b-a400m-instruct")]
    reference_model: PathBuf,
    #[arg(long, default_value = "kernels")]
    metal_sources: PathBuf,
    #[arg(long, default_value = "patches/llama-1719747-qx-validation.patch")]
    llama_patch: PathBuf,
    #[arg(long, default_value = "benchmarks/granite-m5-pro-workload-v1.json")]
    workload_contract: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Verify every pinned input and emit provenance without running decode.
    #[arg(long)]
    verify_only: bool,
    /// Override the contract pair count for smoke testing.
    #[arg(long)]
    pairs: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Contract {
    schema_version: u32,
    name: String,
    pairs: usize,
    warmup: usize,
    fixtures: Vec<PathBuf>,
    provenance: ContractProvenance,
}

#[derive(Debug, Deserialize)]
struct ContractProvenance {
    qx_revision: String,
    llama_revision: String,
    artifact_sha256: String,
    gguf_sha256: String,
    tokenizer_sha256: String,
    #[serde(default)]
    reference_model_manifest_sha256: String,
    #[serde(default)]
    metal_sources_sha256: String,
    #[serde(default)]
    llama_patch_sha256: String,
    workload_contract_sha256: String,
    os_build: String,
    hardware: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Fixture {
    name: String,
    domain: String,
    tokens: Vec<u32>,
    context: usize,
    generation: usize,
    measurement: String,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum DecodeBand {
    Early,
    Late,
}

#[derive(Debug, Serialize)]
struct EngineObservation {
    engine: &'static str,
    tokens_per_second: f64,
    wall_ms_per_token: f64,
    current_rss_bytes: u64,
    peak_rss_bytes: u64,
    current_physical_footprint_bytes: u64,
    peak_physical_footprint_bytes: u64,
    model_bytes: u64,
    mapped_model_bytes: u64,
    known_allocation_bytes: u64,
}

#[derive(Debug, Serialize)]
struct PairedObservation {
    pair: usize,
    fixture: String,
    domain: String,
    band: DecodeBand,
    qx_first: bool,
    thermal_before: String,
    thermal_after: String,
    qx: EngineObservation,
    llama: EngineObservation,
    throughput_advantage: f64,
    wall_ms_saved: f64,
}

#[derive(Debug, Serialize)]
struct ConfidenceInterval {
    median: f64,
    lower_95: f64,
    upper_95: f64,
}

#[derive(Debug, Serialize)]
struct ValidationReport {
    schema_version: u32,
    contract: String,
    contract_hash: String,
    pairs: usize,
    observations: Vec<PairedObservation>,
    throughput_advantage: ConfidenceInterval,
    wall_ms_saved: ConfidenceInterval,
    median_qx_tokens_per_second: f64,
    median_llama_tokens_per_second: f64,
    three_percent_speed_gate: bool,
    paired_parity_gate: bool,
    no_fixture_regression_gate: bool,
    accepted: bool,
    fixture_intervals: Vec<FixtureInterval>,
    provenance: RunProvenance,
}

#[derive(Debug, Serialize)]
struct FixtureInterval {
    fixture: String,
    throughput_advantage: ConfidenceInterval,
    statistically_significant_regression: bool,
}

#[derive(Debug, Serialize)]
struct RunProvenance {
    qx_revision: String,
    qx_source_revision: String,
    llama_revision: String,
    qx_worktree_dirty: bool,
    llama_worktree_dirty: bool,
    qx_binary_sha256: String,
    llama_binary_sha256: String,
    artifact_sha256: String,
    gguf_sha256: String,
    tokenizer_sha256: String,
    reference_model_manifest_sha256: String,
    metal_sources_sha256: String,
    llama_patch_sha256: String,
    workload_contract_sha256: String,
    os_build: String,
    hardware: String,
    power_source: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let contract_bytes =
        fs::read(&args.contract).with_context(|| format!("read {}", args.contract.display()))?;
    let contract: Contract = serde_json::from_slice(&contract_bytes)?;
    if contract.schema_version != 1 || contract.fixtures.len() < 5 {
        bail!("validation contract must be schema v1 with at least five fixtures");
    }
    let provenance = verify_provenance(&args, &contract)?;
    if args.verify_only {
        fs::write(&args.output, serde_json::to_vec_pretty(&provenance)?)
            .with_context(|| format!("write {}", args.output.display()))?;
        println!("provenance verified: {}", args.output.display());
        return Ok(());
    }
    let pairs = args.pairs.unwrap_or(contract.pairs);
    if !(1..=30).contains(&pairs) {
        bail!("paired blocks must be in 1..=30");
    }
    let fixtures = contract
        .fixtures
        .iter()
        .map(|path| {
            let resolved = args
                .contract
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(path);
            serde_json::from_slice::<Fixture>(&fs::read(&resolved)?).map_err(anyhow::Error::from)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut random = Lcg::new(hash_bytes(&contract_bytes));
    let mut observations = Vec::new();
    for pair in 0..pairs {
        for fixture in &fixtures {
            for band in [DecodeBand::Early, DecodeBand::Late] {
                let fixture = fixture_for_band(fixture, band)?;
                let mut stable = None;
                for attempt in 0..3 {
                    let thermal_before = command_output("pmset", &["-g", "therm"]);
                    let qx_first = random.next_u64() & 1 == 0;
                    let (qx, llama) = if qx_first {
                        (
                            run_qx(&args, &contract, pair, &fixture, band)?,
                            run_llama(&args, &fixture)?,
                        )
                    } else {
                        let llama = run_llama(&args, &fixture)?;
                        let qx = run_qx(&args, &contract, pair, &fixture, band)?;
                        (qx, llama)
                    };
                    let thermal_after = command_output("pmset", &["-g", "therm"]);
                    if thermal_before == thermal_after {
                        stable = Some((thermal_before, thermal_after, qx_first, qx, llama));
                        break;
                    }
                    eprintln!(
                        "repeating pair {pair} fixture {} {:?} after thermal transition (attempt {})",
                        fixture.name,
                        band,
                        attempt + 1
                    );
                }
                let Some((thermal_before, thermal_after, qx_first, qx, llama)) = stable else {
                    eprintln!(
                        "discarding pair {pair} fixture {} {:?} after three thermal transitions",
                        fixture.name, band
                    );
                    continue;
                };
                observations.push(PairedObservation {
                    pair,
                    fixture: fixture.name.clone(),
                    domain: fixture.domain.clone(),
                    band,
                    qx_first,
                    thermal_before,
                    thermal_after,
                    throughput_advantage: qx.tokens_per_second / llama.tokens_per_second - 1.0,
                    wall_ms_saved: llama.wall_ms_per_token - qx.wall_ms_per_token,
                    qx,
                    llama,
                });
            }
        }
    }
    if observations.is_empty() {
        bail!("no thermally stable paired observations completed");
    }
    let pair_advantages = pair_means(&observations, |observation| {
        observation.throughput_advantage
    });
    let pair_wall = pair_means(&observations, |observation| observation.wall_ms_saved);
    let throughput_advantage = bootstrap_interval(&pair_advantages, random.next_u64());
    let wall_ms_saved = bootstrap_interval(&pair_wall, random.next_u64());
    let qx_rates = observations
        .iter()
        .map(|observation| observation.qx.tokens_per_second)
        .collect::<Vec<_>>();
    let llama_rates = observations
        .iter()
        .map(|observation| observation.llama.tokens_per_second)
        .collect::<Vec<_>>();
    let fixture_intervals = fixtures
        .iter()
        .map(|fixture| {
            let values = pair_means_filtered(&observations, |observation| {
                observation.fixture == fixture.name
            });
            let interval = bootstrap_interval(&values, random.next_u64());
            FixtureInterval {
                fixture: fixture.name.clone(),
                statistically_significant_regression: interval.upper_95 < 0.0,
                throughput_advantage: interval,
            }
        })
        .collect::<Vec<_>>();
    let three_percent_speed_gate = throughput_advantage.median >= 0.03;
    let paired_parity_gate = throughput_advantage.lower_95 > 0.0;
    let no_fixture_regression_gate = fixture_intervals
        .iter()
        .all(|fixture| !fixture.statistically_significant_regression);
    let report = ValidationReport {
        schema_version: 1,
        contract: contract.name,
        contract_hash: format!("{:016x}", hash_bytes(&contract_bytes)),
        pairs,
        three_percent_speed_gate,
        paired_parity_gate,
        no_fixture_regression_gate,
        accepted: three_percent_speed_gate && paired_parity_gate && no_fixture_regression_gate,
        fixture_intervals,
        provenance,
        throughput_advantage,
        wall_ms_saved,
        median_qx_tokens_per_second: median(qx_rates),
        median_llama_tokens_per_second: median(llama_rates),
        observations,
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", args.output.display()))?;
    println!("paired report: {}", args.output.display());
    println!(
        "QX/llama throughput advantage median/95%: {:.2}% [{:.2}%, {:.2}%]",
        100.0 * report.throughput_advantage.median,
        100.0 * report.throughput_advantage.lower_95,
        100.0 * report.throughput_advantage.upper_95
    );
    Ok(())
}

fn verify_provenance(args: &Args, contract: &Contract) -> Result<RunProvenance> {
    let qx_revision = command_output("git", &["rev-parse", "HEAD"]);
    let llama_revision = command_output("git", &["-C", "external/llama.cpp", "rev-parse", "HEAD"]);
    let artifact_sha256 = file_sha256(&args.artifact)?;
    let gguf_sha256 = file_sha256(&args.gguf)?;
    let tokenizer_path = args.reference_model.join("tokenizer.json");
    let tokenizer_sha256 = file_sha256(&tokenizer_path)?;
    let reference_model_manifest_sha256 =
        directory_manifest_sha256(&args.reference_model, &["json", "safetensors"])?;
    let metal_sources_sha256 = directory_manifest_sha256(&args.metal_sources, &["metal"])?;
    let llama_patch_sha256 = file_sha256(&args.llama_patch)?;
    let workload_contract_sha256 = file_sha256(&args.workload_contract)?;
    let os_build = command_output("sw_vers", &["-buildVersion"]);
    let hardware = command_output("sysctl", &["-n", "machdep.cpu.brand_string"]);
    if qx_revision != contract.provenance.qx_revision
        && !validation_metadata_descendant(&contract.provenance.qx_revision, &qx_revision)?
    {
        bail!(
            "QX revision mismatch: expected compiled source {} or a validation-metadata-only descendant, found {}",
            contract.provenance.qx_revision,
            qx_revision
        );
    }
    for (name, actual, expected) in [
        (
            "llama revision",
            &llama_revision,
            &contract.provenance.llama_revision,
        ),
        (
            "artifact SHA-256",
            &artifact_sha256,
            &contract.provenance.artifact_sha256,
        ),
        (
            "GGUF SHA-256",
            &gguf_sha256,
            &contract.provenance.gguf_sha256,
        ),
        (
            "tokenizer SHA-256",
            &tokenizer_sha256,
            &contract.provenance.tokenizer_sha256,
        ),
        (
            "reference model manifest SHA-256",
            &reference_model_manifest_sha256,
            &contract.provenance.reference_model_manifest_sha256,
        ),
        (
            "Metal sources SHA-256",
            &metal_sources_sha256,
            &contract.provenance.metal_sources_sha256,
        ),
        (
            "llama instrumentation patch SHA-256",
            &llama_patch_sha256,
            &contract.provenance.llama_patch_sha256,
        ),
        (
            "workload contract SHA-256",
            &workload_contract_sha256,
            &contract.provenance.workload_contract_sha256,
        ),
        ("OS build", &os_build, &contract.provenance.os_build),
        ("hardware", &hardware, &contract.provenance.hardware),
    ] {
        if actual != expected {
            bail!("provenance mismatch for {name}: expected {expected}, found {actual}");
        }
    }
    Ok(RunProvenance {
        qx_revision,
        qx_source_revision: contract.provenance.qx_revision.clone(),
        llama_revision,
        qx_worktree_dirty: !command_output("git", &["status", "--porcelain"]).is_empty(),
        llama_worktree_dirty: !command_output(
            "git",
            &["-C", "external/llama.cpp", "status", "--porcelain"],
        )
        .is_empty(),
        qx_binary_sha256: file_sha256(&args.qx_bench)?,
        llama_binary_sha256: file_sha256(&args.llama_simple)?,
        artifact_sha256,
        gguf_sha256,
        tokenizer_sha256,
        reference_model_manifest_sha256,
        metal_sources_sha256,
        llama_patch_sha256,
        workload_contract_sha256,
        os_build,
        hardware,
        power_source: command_output("pmset", &["-g", "batt"]),
    })
}

fn validation_metadata_descendant(expected: &str, actual: &str) -> Result<bool> {
    let ancestor = Command::new("git")
        .args(["merge-base", "--is-ancestor", expected, actual])
        .status()
        .context("check QX source ancestry")?;
    if !ancestor.success() {
        return Ok(false);
    }
    let output = Command::new("git")
        .args(["diff", "--name-only", &format!("{expected}..{actual}")])
        .output()
        .context("inspect post-source provenance changes")?;
    if !output.status.success() {
        return Ok(false);
    }
    let paths = String::from_utf8(output.stdout)?;
    Ok(!paths.trim().is_empty()
        && paths.lines().all(|path| {
            path.starts_with("benchmarks/granite-m5-pro-validation-") && path.ends_with(".json")
        }))
}

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {} for SHA-256", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn directory_manifest_sha256(root: &Path, extensions: &[&str]) -> Result<String> {
    let mut files = Vec::new();
    collect_manifest_files(root, extensions, &mut files)?;
    files.sort();
    let mut digest = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(root).unwrap_or(&path);
        digest.update(relative.to_string_lossy().as_bytes());
        digest.update([0]);
        digest.update(fs::read(&path).with_context(|| format!("read {}", path.display()))?);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn collect_manifest_files(
    directory: &Path,
    extensions: &[&str],
    output: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("read directory {}", directory.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            collect_manifest_files(&path, extensions, output)?;
        } else if path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|extension| extensions.contains(&extension))
        {
            output.push(path);
        }
    }
    Ok(())
}

fn fixture_for_band(source: &Fixture, band: DecodeBand) -> Result<Fixture> {
    let mut fixture = source.clone();
    fixture.tokens = match band {
        DecodeBand::Early => vec![
            *source
                .tokens
                .first()
                .ok_or_else(|| anyhow::anyhow!("empty fixture"))?,
        ],
        DecodeBand::Late => {
            if source.tokens.len() < 384 {
                bail!("late fixture {} has fewer than 384 tokens", source.name);
            }
            source.tokens[..384].to_vec()
        }
    };
    fixture.measurement = "post_prefill".to_owned();
    Ok(fixture)
}

fn run_qx(
    args: &Args,
    contract: &Contract,
    pair: usize,
    fixture: &Fixture,
    band: DecodeBand,
) -> Result<EngineObservation> {
    let stem = format!(
        "qx-validation-{}-{pair}-{}-{}",
        std::process::id(),
        fixture.name,
        match band {
            DecodeBand::Early => "early",
            DecodeBand::Late => "late",
        }
    );
    let fixture_path = std::env::temp_dir().join(format!("{stem}.fixture.json"));
    let result_path = std::env::temp_dir().join(format!("{stem}.result.json"));
    fs::write(&fixture_path, serde_json::to_vec(fixture)?)?;
    let output = Command::new(&args.qx_bench)
        .arg(&args.artifact)
        .arg("--v2-token")
        .arg("--fixture")
        .arg(&fixture_path)
        .arg("--warmup")
        .arg(contract.warmup.to_string())
        .arg("--result-json")
        .arg(&result_path)
        .output()
        .with_context(|| format!("run {}", args.qx_bench.display()))?;
    if !output.status.success() {
        bail!("QX failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let record: Value = serde_json::from_slice(&fs::read(&result_path)?)?;
    let _ = fs::remove_file(fixture_path);
    let _ = fs::remove_file(result_path);
    Ok(EngineObservation {
        engine: "qx",
        tokens_per_second: json_f64(&record, "tokens_per_second")?,
        wall_ms_per_token: json_f64(&record, "wall_ms_per_token")?,
        current_rss_bytes: json_u64(&record, "current_rss_bytes")?,
        peak_rss_bytes: json_u64(&record, "peak_rss_bytes")?,
        current_physical_footprint_bytes: json_u64(&record, "current_physical_footprint_bytes")?,
        peak_physical_footprint_bytes: json_u64(&record, "peak_physical_footprint_bytes")?,
        model_bytes: json_u64(&record, "artifact_bytes")?,
        mapped_model_bytes: json_u64(&record, "mapped_artifact_bytes")?,
        known_allocation_bytes: json_u64(&record, "known_metal_allocated_bytes")?,
    })
}

fn run_llama(args: &Args, fixture: &Fixture) -> Result<EngineObservation> {
    let token_ids = fixture
        .tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut child = Command::new(&args.llama_simple)
        .arg("-m")
        .arg(&args.gguf)
        .arg("-ngl")
        .arg("99")
        .arg("-n")
        .arg((fixture.generation + 1).to_string())
        .arg("-c")
        .arg(fixture.context.to_string())
        .arg("--log-disable")
        .arg("qx-validation")
        .env("QX_TOKEN_IDS", token_ids)
        .env("QX_MEASURE_POST_PREFILL", "1")
        .env("QX_BENCH_SILENT", "1")
        .env("QX_IGNORE_EOG", "1")
        .env("QX_REPORT_MEMORY", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("run {}", args.llama_simple.display()))?;
    let mut child_stdout = child.stdout.take().expect("captured llama stdout");
    let mut child_stderr = child.stderr.take().expect("captured llama stderr");
    let stdout_reader = thread::spawn(move || {
        let mut output = Vec::new();
        child_stdout.read_to_end(&mut output).map(|_| output)
    });
    let stderr_reader = thread::spawn(move || {
        let mut output = String::new();
        child_stderr.read_to_string(&mut output).map(|_| output)
    });
    let mut peak_rss = 0_u64;
    let mut current_rss = 0_u64;
    let mut peak_footprint = 0_u64;
    let mut current_footprint = 0_u64;
    while child.try_wait()?.is_none() {
        current_rss = process_rss(child.id());
        peak_rss = peak_rss.max(current_rss);
        let footprint = process_physical_footprint(child.id());
        current_footprint = footprint;
        peak_footprint = peak_footprint.max(footprint);
        thread::sleep(Duration::from_millis(10));
    }
    let status = child.wait()?;
    let _stdout = stdout_reader
        .join()
        .map_err(|_| anyhow::anyhow!("llama stdout reader panicked"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| anyhow::anyhow!("llama stderr reader panicked"))??;
    if !status.success() {
        bail!("llama failed: {stderr}");
    }
    let (tokens, elapsed_ms, tokens_per_second) = parse_llama_measurement(&stderr)?;
    if tokens != fixture.generation {
        bail!(
            "llama measured {tokens} tokens, expected {}",
            fixture.generation
        );
    }
    let model_tensor_bytes = parse_llama_model_tensor_bytes(&stderr)?;
    let gguf_bytes = fs::metadata(&args.gguf)?.len();
    Ok(EngineObservation {
        engine: "llama",
        tokens_per_second,
        wall_ms_per_token: elapsed_ms / tokens as f64,
        current_rss_bytes: current_rss,
        peak_rss_bytes: peak_rss,
        current_physical_footprint_bytes: current_footprint,
        peak_physical_footprint_bytes: peak_footprint,
        model_bytes: gguf_bytes,
        mapped_model_bytes: gguf_bytes,
        known_allocation_bytes: model_tensor_bytes,
    })
}

fn parse_llama_model_tensor_bytes(stderr: &str) -> Result<u64> {
    let line = stderr
        .lines()
        .find(|line| line.starts_with("QX llama model tensor bytes:"))
        .ok_or_else(|| anyhow::anyhow!("llama output lacks model tensor byte accounting"))?;
    line.rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("invalid llama model tensor accounting"))?
        .1
        .trim()
        .parse()
        .map_err(Into::into)
}

fn parse_llama_measurement(stderr: &str) -> Result<(usize, f64, f64)> {
    let line = stderr
        .lines()
        .find(|line| line.starts_with("QX measured post-prefill:"))
        .ok_or_else(|| anyhow::anyhow!("llama output lacks QX post-prefill measurement"))?;
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 11 {
        bail!("invalid llama QX measurement: {line}");
    }
    Ok((fields[3].parse()?, fields[6].parse()?, fields[9].parse()?))
}

fn pair_means<F>(observations: &[PairedObservation], value: F) -> Vec<f64>
where
    F: Fn(&PairedObservation) -> f64,
{
    let max_pair = observations
        .iter()
        .map(|observation| observation.pair)
        .max()
        .unwrap_or(0);
    (0..=max_pair)
        .filter_map(|pair| {
            let values = observations
                .iter()
                .filter(|observation| observation.pair == pair)
                .map(&value)
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
        })
        .collect()
}

fn pair_means_filtered<F>(observations: &[PairedObservation], include: F) -> Vec<f64>
where
    F: Fn(&PairedObservation) -> bool,
{
    let max_pair = observations
        .iter()
        .map(|observation| observation.pair)
        .max()
        .unwrap_or(0);
    (0..=max_pair)
        .filter_map(|pair| {
            let values = observations
                .iter()
                .filter(|observation| observation.pair == pair && include(observation))
                .map(|observation| observation.throughput_advantage)
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
        })
        .collect()
}

fn bootstrap_interval(values: &[f64], seed: u64) -> ConfidenceInterval {
    let mut random = Lcg::new(seed);
    let mut means = Vec::with_capacity(10_000);
    for _ in 0..10_000 {
        let mean = (0..values.len())
            .map(|_| values[random.next_u64() as usize % values.len()])
            .sum::<f64>()
            / values.len() as f64;
        means.push(mean);
    }
    means.sort_by(f64::total_cmp);
    ConfidenceInterval {
        median: median(values.to_vec()),
        lower_95: means[means.len() * 25 / 1000],
        upper_95: means[means.len() * 975 / 1000],
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn json_f64(value: &Value, field: &str) -> Result<f64> {
    value[field]
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("benchmark JSON lacks numeric {field}"))
}

fn json_u64(value: &Value, field: &str) -> Result<u64> {
    value[field]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("benchmark JSON lacks integer {field}"))
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

#[cfg(target_os = "macos")]
fn process_rss(pid: u32) -> u64 {
    let mut task = std::mem::MaybeUninit::<libc::proc_taskinfo>::zeroed();
    let size = std::mem::size_of::<libc::proc_taskinfo>() as i32;
    let written = unsafe {
        libc::proc_pidinfo(
            pid as i32,
            libc::PROC_PIDTASKINFO,
            0,
            task.as_mut_ptr().cast(),
            size,
        )
    };
    if written == size {
        unsafe { task.assume_init().pti_resident_size }
    } else {
        0
    }
}

#[cfg(target_os = "macos")]
fn process_physical_footprint(pid: u32) -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    let status = unsafe {
        libc::proc_pid_rusage(pid as i32, libc::RUSAGE_INFO_V4, usage.as_mut_ptr().cast())
    };
    if status == 0 {
        unsafe { usage.assume_init().ri_phys_footprint }
    } else {
        0
    }
}

#[cfg(not(target_os = "macos"))]
fn process_rss(_pid: u32) -> u64 {
    0
}

#[cfg(not(target_os = "macos"))]
fn process_physical_footprint(_pid: u32) -> u64 {
    0
}
