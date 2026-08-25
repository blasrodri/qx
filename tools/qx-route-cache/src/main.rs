use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: u32 = 1;
const SIMULATION_SCHEMA_VERSION: u32 = 2;
const DEFAULT_Q3_EXPERT_BYTES: u64 = 2_064_384;
const DEFAULT_EXPERT_NEURONS: usize = 768;

#[derive(Debug, Parser)]
#[command(
    name = "qx-route-cache",
    about = "Capture sparse-model routes and simulate bounded expert residency"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Capture one-token Qwen MoE routes through llama.cpp's eval callback.
    Capture {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "smoke")]
        request_id: String,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 4096)]
        context: usize,
        #[arg(long, default_value_t = 48)]
        layers: usize,
        #[arg(long, default_value_t = 128)]
        experts: usize,
        #[arg(long, default_value_t = 8)]
        top_k: usize,
        #[arg(long, default_value_t = -1)]
        gpu_layers: i32,
        #[arg(long, default_value_t = 4)]
        threads: i32,
        /// Capture post-SwiGLU energy summaries for these neuron packet sizes.
        #[arg(long, value_delimiter = ',')]
        activation_packet_neurons: Vec<usize>,
        /// Capture exact down-output contribution summaries for these packet sizes.
        #[arg(long, value_delimiter = ',')]
        contribution_packet_neurons: Vec<usize>,
        /// Expert intermediate width used by activation capture.
        #[arg(long, default_value_t = DEFAULT_EXPERT_NEURONS)]
        expert_neurons: usize,
        /// Model hidden width used by down-output contribution capture.
        #[arg(long, default_value_t = 2048)]
        hidden_size: usize,
        /// Trusted model SHA-256. If omitted, the complete GGUF is hashed.
        #[arg(long)]
        model_sha256: Option<String>,
    },
    /// Replay a route trace through one or more bounded expert caches.
    Simulate {
        #[arg(long)]
        trace: PathBuf,
        /// Disjoint calibration trace used only to select static hot experts.
        #[arg(long)]
        calibration: Option<PathBuf>,
        #[arg(long, value_delimiter = ',', default_value = "lru")]
        policies: Vec<CachePolicy>,
        #[arg(long, value_delimiter = ',', value_parser = parse_bytes)]
        budgets: Vec<u64>,
        #[arg(long, default_value_t = DEFAULT_Q3_EXPERT_BYTES)]
        expert_bytes: u64,
        /// Intermediate width of one complete expert.
        #[arg(long, default_value_t = DEFAULT_EXPERT_NEURONS)]
        expert_neurons: usize,
        /// Neurons stored in one independently resident packet.
        #[arg(long, value_delimiter = ',', default_value = "768")]
        packet_neurons: Vec<usize>,
        /// Physical packet order used to turn misses into contiguous fetches.
        #[arg(long, value_delimiter = ',', default_value = "expert-major")]
        layouts: Vec<PhysicalLayout>,
        /// Synthetic sustained storage bandwidth used for I/O timing estimates.
        #[arg(long)]
        io_bandwidth_mib_s: Option<f64>,
        /// Synthetic fixed latency for each contiguous missing-packet run.
        #[arg(long, default_value_t = 0.0)]
        io_fixed_us: f64,
        /// Maximum concurrent contiguous fetches in the synthetic I/O shaper.
        #[arg(long, default_value_t = 8)]
        io_parallelism: usize,
        #[arg(long)]
        reset_between_requests: bool,
        #[arg(long)]
        output: PathBuf,
    },
    /// Analyze perfect-information packet selection from captured activations.
    Oracle {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "0.9,0.95,0.99,0.999")]
        thresholds: Vec<f64>,
        #[arg(long, default_value_t = DEFAULT_Q3_EXPERT_BYTES)]
        expert_bytes: u64,
        #[arg(long)]
        output: PathBuf,
    },
    /// Analyze perfect-knowledge greedy packet selection by down-output error.
    ContributionOracle {
        #[arg(long)]
        trace: PathBuf,
        /// Per-expert relative L2 error targets after the down projection.
        #[arg(long, value_delimiter = ',', default_value = "0.1,0.05,0.01,0.001")]
        relative_errors: Vec<f64>,
        /// Fixed packet fractions whose resulting down-output error is reported.
        #[arg(long, value_delimiter = ',', default_value = "0.25,0.5,0.75,0.9")]
        selected_fractions: Vec<f64>,
        #[arg(long, default_value_t = DEFAULT_Q3_EXPERT_BYTES)]
        expert_bytes: u64,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum CachePolicy {
    Lru,
    StaticHot,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum PhysicalLayout {
    #[value(name = "expert-major")]
    ExpertContiguous,
    #[value(name = "neuron-major")]
    NeuronInterleaved,
    #[value(name = "co-selection-expert-major")]
    CoSelectedExpertContiguous,
    #[value(name = "co-selection-neuron-major")]
    CoSelectedNeuronInterleaved,
}

impl PhysicalLayout {
    fn label(self) -> &'static str {
        match self {
            Self::ExpertContiguous => "expert-major",
            Self::NeuronInterleaved => "neuron-major",
            Self::CoSelectedExpertContiguous => "co-selection-expert-major",
            Self::CoSelectedNeuronInterleaved => "co-selection-neuron-major",
        }
    }

    fn uses_co_selection(self) -> bool {
        matches!(
            self,
            Self::CoSelectedExpertContiguous | Self::CoSelectedNeuronInterleaved
        )
    }

    fn is_neuron_major(self) -> bool {
        matches!(
            self,
            Self::NeuronInterleaved | Self::CoSelectedNeuronInterleaved
        )
    }
}

impl CachePolicy {
    fn label(self) -> &'static str {
        match self {
            Self::Lru => "lru",
            Self::StaticHot => "static-hot",
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TraceLine {
    Header(TraceHeader),
    Token(TokenRoutes),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TraceHeader {
    schema_version: u32,
    model_sha256: String,
    llama_revision: String,
    layers: usize,
    experts: usize,
    top_k: usize,
    source: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TokenRoutes {
    request_id: String,
    token_position: usize,
    phase: String,
    input_token: i32,
    predicted_token: i32,
    routes: Vec<Vec<u16>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    packet_activation_energy: Option<Vec<PacketActivationEnergy>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    packet_down_contribution: Option<PacketDownContribution>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PacketActivationEnergy {
    packet_neurons: usize,
    l1: Vec<Vec<Vec<f32>>>,
    squared_l2: Vec<Vec<Vec<f32>>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PacketDownContribution {
    reconstruction_relative_squared_error: Vec<Vec<f32>>,
    packets: Vec<PacketContributionMetrics>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PacketContributionMetrics {
    packet_neurons: usize,
    squared_l2: Vec<Vec<Vec<f32>>>,
    /// Relative squared error after each greedy selection, from one packet through all packets.
    greedy_relative_squared_error: Vec<Vec<Vec<f32>>>,
}

#[derive(Debug, Serialize)]
struct SimulationReport {
    schema_version: u32,
    trace: String,
    calibration: Option<String>,
    trace_header: TraceHeader,
    expert_bytes: u64,
    expert_neurons: usize,
    packet_neurons: Vec<usize>,
    layouts: Vec<PhysicalLayout>,
    io_shape: Option<IoShape>,
    results: Vec<SimulationResult>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct IoShape {
    bandwidth_mib_s: f64,
    fixed_latency_us: f64,
    max_inflight_fetches: usize,
}

#[derive(Debug, Serialize)]
struct SimulationResult {
    policy: String,
    physical_layout: String,
    budget_bytes: u64,
    packet_neurons: usize,
    packets_per_expert: usize,
    packet_bytes: u64,
    resident_packet_capacity: usize,
    token_records: usize,
    expert_route_lookups: u64,
    route_lookups: u64,
    hits: u64,
    misses: u64,
    byte_weighted_hit_rate: f64,
    compulsory_misses: u64,
    capacity_misses: u64,
    total_missing_bytes: u64,
    mean_missing_bytes_per_token: f64,
    p50_missing_bytes_per_token: u64,
    p95_missing_bytes_per_token: u64,
    p99_missing_bytes_per_token: u64,
    p50_missing_bytes_per_layer: u64,
    p95_missing_bytes_per_layer: u64,
    p99_missing_bytes_per_layer: u64,
    maximum_missing_bytes_per_layer: u64,
    maximum_contiguous_burst_bytes: u64,
    mean_transfer_runs_per_token: f64,
    p95_transfer_runs_per_token: u64,
    p99_transfer_runs_per_token: u64,
    mean_contiguous_bytes_per_fetch: f64,
    p50_contiguous_bytes_per_fetch: u64,
    p95_contiguous_bytes_per_fetch: u64,
    p99_contiguous_bytes_per_fetch: u64,
    p50_reuse_distance_packet_lookups: u64,
    p95_reuse_distance_packet_lookups: u64,
    p99_reuse_distance_packet_lookups: u64,
    mean_layers_with_miss_per_token: f64,
    p95_layers_with_miss_per_token: u64,
    p99_layers_with_miss_per_token: u64,
    evictions: u64,
    mean_evictions_per_token: f64,
    mean_adjacent_route_overlap: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_serial_io_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p95_serial_io_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p99_serial_io_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_ideal_layer_parallel_io_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p95_ideal_layer_parallel_io_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p99_ideal_layer_parallel_io_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mean_shaped_io_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p95_shaped_io_ms_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p99_shaped_io_ms_per_token: Option<f64>,
}

#[derive(Debug, Serialize)]
struct OracleReport {
    schema_version: u32,
    classification: &'static str,
    quality_semantics: &'static str,
    trace: String,
    trace_header: TraceHeader,
    expert_bytes: u64,
    token_records: usize,
    results: Vec<OracleResult>,
}

#[derive(Debug, Serialize)]
struct OracleResult {
    packet_neurons: usize,
    packets_per_expert: usize,
    threshold: f64,
    expert_observations: usize,
    mean_selected_packets_per_expert: f64,
    p50_selected_packets_per_expert: u64,
    p95_selected_packets_per_expert: u64,
    p99_selected_packets_per_expert: u64,
    mean_selected_fraction: f64,
    mean_retained_activation_energy: f64,
    mean_active_packet_bytes_per_token: f64,
    p50_active_packet_bytes_per_token: u64,
    p95_active_packet_bytes_per_token: u64,
    p99_active_packet_bytes_per_token: u64,
}

#[derive(Default)]
struct OracleAccumulator {
    selected_packets: Vec<u64>,
    retained_energy: Vec<f64>,
    active_bytes_per_token: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct ContributionOracleReport {
    schema_version: u32,
    classification: &'static str,
    selection_semantics: &'static str,
    quality_semantics: &'static str,
    trace: String,
    trace_header: TraceHeader,
    expert_bytes: u64,
    token_records: usize,
    reconstruction_relative_l2_error_p95: f64,
    reconstruction_relative_l2_error_p99: f64,
    reconstruction_relative_l2_error_max: f64,
    results: Vec<ContributionOracleResult>,
    fixed_budget_results: Vec<FixedBudgetContributionResult>,
}

#[derive(Debug, Serialize)]
struct ContributionOracleResult {
    packet_neurons: usize,
    packets_per_expert: usize,
    relative_l2_error_target: f64,
    expert_observations: usize,
    unmet_target_observations: usize,
    mean_selected_packets_per_expert: f64,
    p50_selected_packets_per_expert: u64,
    p95_selected_packets_per_expert: u64,
    p99_selected_packets_per_expert: u64,
    mean_selected_fraction: f64,
    mean_achieved_relative_l2_error: f64,
    mean_active_packet_bytes_per_token: f64,
    p50_active_packet_bytes_per_token: u64,
    p95_active_packet_bytes_per_token: u64,
    p99_active_packet_bytes_per_token: u64,
}

#[derive(Debug, Serialize)]
struct FixedBudgetContributionResult {
    packet_neurons: usize,
    packets_per_expert: usize,
    requested_selected_fraction: f64,
    selected_packets_per_expert: usize,
    actual_selected_fraction: f64,
    expert_observations: usize,
    mean_relative_l2_error: f64,
    p50_relative_l2_error: f64,
    p95_relative_l2_error: f64,
    p99_relative_l2_error: f64,
    active_packet_bytes_per_token: u64,
}

#[derive(Default)]
struct ContributionOracleAccumulator {
    selected_packets: Vec<u64>,
    achieved_relative_l2_error: Vec<f64>,
    active_bytes_per_token: Vec<u64>,
    unmet_target_observations: usize,
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Capture {
            model,
            output,
            prompt,
            request_id,
            max_new_tokens,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            activation_packet_neurons,
            contribution_packet_neurons,
            expert_neurons,
            hidden_size,
            model_sha256,
        } => capture(CaptureArgs {
            model,
            output,
            prompt,
            request_id,
            max_new_tokens,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            activation_packet_neurons,
            contribution_packet_neurons,
            expert_neurons,
            hidden_size,
            model_sha256,
        }),
        Command::Simulate {
            trace,
            calibration,
            policies,
            budgets,
            expert_bytes,
            expert_neurons,
            packet_neurons,
            layouts,
            io_bandwidth_mib_s,
            io_fixed_us,
            io_parallelism,
            reset_between_requests,
            output,
        } => simulate(SimulateArgs {
            trace,
            calibration,
            policies,
            budgets,
            expert_bytes,
            expert_neurons,
            packet_neurons,
            layouts,
            io_bandwidth_mib_s,
            io_fixed_us,
            io_parallelism,
            reset_between_requests,
            output,
        }),
        Command::Oracle {
            trace,
            thresholds,
            expert_bytes,
            output,
        } => oracle(&trace, &thresholds, expert_bytes, &output),
        Command::ContributionOracle {
            trace,
            relative_errors,
            selected_fractions,
            expert_bytes,
            output,
        } => contribution_oracle(
            &trace,
            &relative_errors,
            &selected_fractions,
            expert_bytes,
            &output,
        ),
    }
}

struct CaptureArgs {
    model: PathBuf,
    output: PathBuf,
    prompt: String,
    request_id: String,
    max_new_tokens: usize,
    context: usize,
    layers: usize,
    experts: usize,
    top_k: usize,
    gpu_layers: i32,
    threads: i32,
    activation_packet_neurons: Vec<usize>,
    contribution_packet_neurons: Vec<usize>,
    expert_neurons: usize,
    hidden_size: usize,
    model_sha256: Option<String>,
}

struct SimulateArgs {
    trace: PathBuf,
    calibration: Option<PathBuf>,
    policies: Vec<CachePolicy>,
    budgets: Vec<u64>,
    expert_bytes: u64,
    expert_neurons: usize,
    packet_neurons: Vec<usize>,
    layouts: Vec<PhysicalLayout>,
    io_bandwidth_mib_s: Option<f64>,
    io_fixed_us: f64,
    io_parallelism: usize,
    reset_between_requests: bool,
    output: PathBuf,
}

#[cfg(target_os = "macos")]
fn capture(args: CaptureArgs) -> Result<()> {
    if !args.model.is_file() {
        bail!("model does not exist: {}", args.model.display());
    }
    if args.layers == 0 || args.experts == 0 || args.top_k == 0 {
        bail!("layers, experts, and top-k must be positive");
    }
    if args.max_new_tokens == 0 {
        bail!("max-new-tokens must be positive");
    }
    for &packet_neurons in args
        .activation_packet_neurons
        .iter()
        .chain(&args.contribution_packet_neurons)
    {
        if packet_neurons == 0 || !args.expert_neurons.is_multiple_of(packet_neurons) {
            bail!(
                "capture packet size {packet_neurons} must divide expert-neurons {}",
                args.expert_neurons
            );
        }
    }
    if args.hidden_size == 0 {
        bail!("hidden-size must be positive");
    }
    if let Some(&finest) = args.contribution_packet_neurons.iter().min()
        && args
            .contribution_packet_neurons
            .iter()
            .any(|size| !size.is_multiple_of(finest))
    {
        bail!("every contribution packet size must be a multiple of the finest size");
    }
    if args
        .contribution_packet_neurons
        .iter()
        .copied()
        .collect::<HashSet<_>>()
        .len()
        != args.contribution_packet_neurons.len()
    {
        bail!("contribution packet sizes must be unique");
    }
    let model_hash = match args.model_sha256.as_ref() {
        Some(value) => {
            validate_sha256(value)?;
            value.clone()
        }
        None => sha256_file(&args.model)?,
    };
    let mut llama = llama_capture::Capture::open(
        &args.model,
        llama_capture::OpenConfig {
            context: args.context,
            layers: args.layers,
            top_k: args.top_k,
            gpu_layers: args.gpu_layers,
            threads: args.threads,
            intermediate: (!args.activation_packet_neurons.is_empty()
                || !args.contribution_packet_neurons.is_empty())
            .then_some(args.expert_neurons),
            hidden: args.hidden_size,
            contribution_packet_neurons: &args.contribution_packet_neurons,
        },
    )?;
    let prompt_tokens = llama.tokenize(&args.prompt)?;
    if prompt_tokens.is_empty() {
        bail!("prompt produced no tokens");
    }
    if prompt_tokens.len() + args.max_new_tokens > args.context {
        bail!("prompt plus generation exceeds context {}", args.context);
    }

    let file =
        File::create(&args.output).with_context(|| format!("create {}", args.output.display()))?;
    let mut writer = BufWriter::new(file);
    write_trace_line(
        &mut writer,
        &TraceLine::Header(TraceHeader {
            schema_version: SCHEMA_VERSION,
            model_sha256: model_hash,
            llama_revision: option_env!("QX_LLAMA_REVISION")
                .unwrap_or("unknown")
                .to_owned(),
            layers: args.layers,
            experts: args.experts,
            top_k: args.top_k,
            source: "llama.cpp-eval-callback".to_owned(),
        }),
    )?;

    let mut position = 0usize;
    let mut next = 0i32;
    for token in prompt_tokens {
        let llama_capture::StepOutput {
            predicted,
            routes,
            activations,
            down_contribution,
        } = llama.step(token)?;
        validate_routes(&routes, args.layers, args.top_k, args.experts)?;
        write_trace_line(
            &mut writer,
            &TraceLine::Token(TokenRoutes {
                request_id: args.request_id.clone(),
                token_position: position,
                phase: "prompt".to_owned(),
                input_token: token,
                predicted_token: predicted,
                routes,
                packet_activation_energy: (!args.activation_packet_neurons.is_empty()).then(|| {
                    summarize_activations(
                        activations
                            .as_deref()
                            .expect("activation buffer exists when energy capture is enabled"),
                        &args,
                    )
                }),
                packet_down_contribution: down_contribution,
            }),
        )?;
        position += 1;
        next = predicted;
    }
    let eos = llama.eos();
    for _ in 0..args.max_new_tokens {
        if next == eos {
            break;
        }
        let llama_capture::StepOutput {
            predicted,
            routes,
            activations,
            down_contribution,
        } = llama.step(next)?;
        validate_routes(&routes, args.layers, args.top_k, args.experts)?;
        write_trace_line(
            &mut writer,
            &TraceLine::Token(TokenRoutes {
                request_id: args.request_id.clone(),
                token_position: position,
                phase: "generation".to_owned(),
                input_token: next,
                predicted_token: predicted,
                routes,
                packet_activation_energy: (!args.activation_packet_neurons.is_empty()).then(|| {
                    summarize_activations(
                        activations
                            .as_deref()
                            .expect("activation buffer exists when energy capture is enabled"),
                        &args,
                    )
                }),
                packet_down_contribution: down_contribution,
            }),
        )?;
        position += 1;
        next = predicted;
    }
    writer.flush().context("flush route trace")?;
    println!("trace: {}", args.output.display());
    println!("routed token positions: {position}");
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn capture(_args: CaptureArgs) -> Result<()> {
    bail!("route capture currently requires the macOS llama.cpp bridge")
}

fn summarize_activations(values: &[f32], args: &CaptureArgs) -> Vec<PacketActivationEnergy> {
    debug_assert_eq!(values.len(), args.layers * args.top_k * args.expert_neurons);
    args.activation_packet_neurons
        .iter()
        .map(|&packet_neurons| {
            let packets = args.expert_neurons / packet_neurons;
            let mut l1 = vec![vec![vec![0.0f32; packets]; args.top_k]; args.layers];
            let mut squared_l2 = vec![vec![vec![0.0f32; packets]; args.top_k]; args.layers];
            for layer in 0..args.layers {
                for slot in 0..args.top_k {
                    let base = (layer * args.top_k + slot) * args.expert_neurons;
                    for packet in 0..packets {
                        let range =
                            base + packet * packet_neurons..base + (packet + 1) * packet_neurons;
                        for &value in &values[range] {
                            l1[layer][slot][packet] += value.abs();
                            squared_l2[layer][slot][packet] += value * value;
                        }
                    }
                }
            }
            PacketActivationEnergy {
                packet_neurons,
                l1,
                squared_l2,
            }
        })
        .collect()
}

fn oracle(trace: &Path, thresholds: &[f64], expert_bytes: u64, output: &Path) -> Result<()> {
    if thresholds.is_empty()
        || thresholds
            .iter()
            .any(|threshold| !threshold.is_finite() || *threshold <= 0.0 || *threshold > 1.0)
    {
        bail!("oracle thresholds must be finite values in (0, 1]");
    }
    let header = read_trace_header(trace)?;
    let mut accumulators = HashMap::<(usize, usize), OracleAccumulator>::new();
    let mut packet_counts = HashMap::<usize, usize>::new();
    let mut token_records = 0usize;
    visit_trace(trace, &header, |token| {
        let energy_sets = token
            .packet_activation_energy
            .ok_or_else(|| anyhow::anyhow!("trace token lacks packet activation energy"))?;
        for energy in energy_sets {
            validate_activation_energy(&energy, &header)?;
            let packets_per_expert = energy.squared_l2[0][0].len();
            if !expert_bytes.is_multiple_of(packets_per_expert as u64) {
                bail!(
                    "expert bytes {expert_bytes} are not divisible by {packets_per_expert} packets"
                );
            }
            if packet_counts
                .insert(energy.packet_neurons, packets_per_expert)
                .is_some_and(|previous| previous != packets_per_expert)
            {
                bail!(
                    "inconsistent packet count for {} neurons",
                    energy.packet_neurons
                );
            }
            let packet_bytes = expert_bytes / packets_per_expert as u64;
            let mut token_selected = vec![0u64; thresholds.len()];
            for layer in &energy.squared_l2 {
                for expert in layer {
                    for (threshold_index, &threshold) in thresholds.iter().enumerate() {
                        let (selected, retained) = select_activation_packets(expert, threshold)?;
                        let accumulator = accumulators
                            .entry((energy.packet_neurons, threshold_index))
                            .or_default();
                        accumulator.selected_packets.push(selected as u64);
                        accumulator.retained_energy.push(retained);
                        token_selected[threshold_index] += selected as u64;
                    }
                }
            }
            for (threshold_index, selected) in token_selected.into_iter().enumerate() {
                accumulators
                    .entry((energy.packet_neurons, threshold_index))
                    .or_default()
                    .active_bytes_per_token
                    .push(selected.saturating_mul(packet_bytes));
            }
        }
        token_records += 1;
        Ok(())
    })?;
    if token_records == 0 || packet_counts.is_empty() {
        bail!("activation trace contains no token records");
    }
    let mut packet_sizes = packet_counts.keys().copied().collect::<Vec<_>>();
    packet_sizes.sort_unstable_by(|left, right| right.cmp(left));
    let mut results = Vec::new();
    for packet_neurons in packet_sizes {
        let packets_per_expert = packet_counts[&packet_neurons];
        for (threshold_index, &threshold) in thresholds.iter().enumerate() {
            let accumulator = accumulators
                .remove(&(packet_neurons, threshold_index))
                .ok_or_else(|| anyhow::anyhow!("missing oracle accumulator"))?;
            let expert_observations = accumulator.selected_packets.len();
            let selected_total = accumulator.selected_packets.iter().sum::<u64>();
            results.push(OracleResult {
                packet_neurons,
                packets_per_expert,
                threshold,
                expert_observations,
                mean_selected_packets_per_expert: selected_total as f64
                    / expert_observations.max(1) as f64,
                p50_selected_packets_per_expert: percentile(&accumulator.selected_packets, 0.50),
                p95_selected_packets_per_expert: percentile(&accumulator.selected_packets, 0.95),
                p99_selected_packets_per_expert: percentile(&accumulator.selected_packets, 0.99),
                mean_selected_fraction: selected_total as f64
                    / (expert_observations.max(1) * packets_per_expert) as f64,
                mean_retained_activation_energy: accumulator.retained_energy.iter().sum::<f64>()
                    / accumulator.retained_energy.len().max(1) as f64,
                mean_active_packet_bytes_per_token: accumulator
                    .active_bytes_per_token
                    .iter()
                    .sum::<u64>() as f64
                    / accumulator.active_bytes_per_token.len().max(1) as f64,
                p50_active_packet_bytes_per_token: percentile(
                    &accumulator.active_bytes_per_token,
                    0.50,
                ),
                p95_active_packet_bytes_per_token: percentile(
                    &accumulator.active_bytes_per_token,
                    0.95,
                ),
                p99_active_packet_bytes_per_token: percentile(
                    &accumulator.active_bytes_per_token,
                    0.99,
                ),
            });
        }
    }
    let report = OracleReport {
        schema_version: 1,
        classification: "activation-energy-oracle-smoke",
        quality_semantics: "post-swiglu-squared-l2-proxy-not-down-output-or-quality",
        trace: trace.display().to_string(),
        trace_header: header,
        expert_bytes,
        token_records,
        results,
    };
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("create {}", output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("oracle report: {}", output.display());
    for result in &report.results {
        println!(
            "{} neurons {:.3}% energy: mean {:.2}/{} packets, p95 {}, {:.2} MiB/token",
            result.packet_neurons,
            result.threshold * 100.0,
            result.mean_selected_packets_per_expert,
            result.packets_per_expert,
            result.p95_selected_packets_per_expert,
            result.mean_active_packet_bytes_per_token / (1024.0 * 1024.0),
        );
    }
    Ok(())
}

fn validate_activation_energy(energy: &PacketActivationEnergy, header: &TraceHeader) -> Result<()> {
    for (label, values) in [("l1", &energy.l1), ("squared-l2", &energy.squared_l2)] {
        if values.len() != header.layers {
            bail!("{label} activation energy has the wrong layer count");
        }
        for layer in values {
            if layer.len() != header.top_k || layer.iter().any(Vec::is_empty) {
                bail!("{label} activation energy has an invalid expert-slot shape");
            }
            let packets = layer[0].len();
            if layer.iter().any(|expert| expert.len() != packets) {
                bail!("{label} activation energy has inconsistent packet counts");
            }
        }
    }
    Ok(())
}

fn select_activation_packets(energy: &[f32], threshold: f64) -> Result<(usize, f64)> {
    if energy
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        bail!("activation energy contains a non-finite or negative value");
    }
    let total = energy.iter().map(|&value| value as f64).sum::<f64>();
    if total == 0.0 {
        return Ok((0, 1.0));
    }
    let mut ranked = energy.to_vec();
    ranked.sort_by(|left, right| right.total_cmp(left));
    let mut retained = 0.0f64;
    for (index, value) in ranked.into_iter().enumerate() {
        retained += value as f64;
        if retained / total >= threshold {
            return Ok((index + 1, retained / total));
        }
    }
    Ok((energy.len(), 1.0))
}

fn contribution_oracle(
    trace: &Path,
    relative_errors: &[f64],
    selected_fractions: &[f64],
    expert_bytes: u64,
    output: &Path,
) -> Result<()> {
    if relative_errors.is_empty()
        || relative_errors
            .iter()
            .any(|error| !error.is_finite() || *error <= 0.0 || *error >= 1.0)
    {
        bail!("relative error targets must be finite values in (0, 1)");
    }
    if selected_fractions.is_empty()
        || selected_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || *fraction <= 0.0 || *fraction > 1.0)
    {
        bail!("selected fractions must be finite values in (0, 1]");
    }
    let header = read_trace_header(trace)?;
    let mut accumulators = HashMap::<(usize, usize), ContributionOracleAccumulator>::new();
    let mut packet_counts = HashMap::<usize, usize>::new();
    let mut fixed_budget_errors = HashMap::<(usize, usize), Vec<f64>>::new();
    let mut reconstruction_errors = Vec::<f64>::new();
    let mut token_records = 0usize;
    visit_trace(trace, &header, |token| {
        let contribution = token
            .packet_down_contribution
            .ok_or_else(|| anyhow::anyhow!("trace token lacks packet down contribution"))?;
        validate_matrix_shape(
            "down reconstruction error",
            &contribution.reconstruction_relative_squared_error,
            header.layers,
            header.top_k,
        )?;
        for layer in &contribution.reconstruction_relative_squared_error {
            for &error in layer {
                if !error.is_finite() || error < 0.0 {
                    bail!("down reconstruction error is non-finite or negative");
                }
                reconstruction_errors.push((error as f64).sqrt());
            }
        }
        for packet_metrics in contribution.packets {
            validate_contribution_metrics(&packet_metrics, &header)?;
            let packets_per_expert = packet_metrics.greedy_relative_squared_error[0][0].len();
            if !expert_bytes.is_multiple_of(packets_per_expert as u64) {
                bail!(
                    "expert bytes {expert_bytes} are not divisible by {packets_per_expert} packets"
                );
            }
            if packet_counts
                .insert(packet_metrics.packet_neurons, packets_per_expert)
                .is_some_and(|previous| previous != packets_per_expert)
            {
                bail!(
                    "inconsistent packet count for {} neurons",
                    packet_metrics.packet_neurons
                );
            }
            let packet_bytes = expert_bytes / packets_per_expert as u64;
            let mut token_selected = vec![0u64; relative_errors.len()];
            for layer in &packet_metrics.greedy_relative_squared_error {
                for curve in layer {
                    for (target_index, &target) in relative_errors.iter().enumerate() {
                        let (selected, achieved, met) = select_contribution_packets(curve, target)?;
                        let accumulator = accumulators
                            .entry((packet_metrics.packet_neurons, target_index))
                            .or_default();
                        accumulator.selected_packets.push(selected as u64);
                        accumulator.achieved_relative_l2_error.push(achieved);
                        accumulator.unmet_target_observations += usize::from(!met);
                        token_selected[target_index] += selected as u64;
                    }
                    for (fraction_index, &fraction) in selected_fractions.iter().enumerate() {
                        let selected = ((packets_per_expert as f64 * fraction).ceil() as usize)
                            .clamp(1, packets_per_expert);
                        fixed_budget_errors
                            .entry((packet_metrics.packet_neurons, fraction_index))
                            .or_default()
                            .push((curve[selected - 1] as f64).sqrt());
                    }
                }
            }
            for (target_index, selected) in token_selected.into_iter().enumerate() {
                accumulators
                    .entry((packet_metrics.packet_neurons, target_index))
                    .or_default()
                    .active_bytes_per_token
                    .push(selected.saturating_mul(packet_bytes));
            }
        }
        token_records += 1;
        Ok(())
    })?;
    if token_records == 0 || packet_counts.is_empty() || reconstruction_errors.is_empty() {
        bail!("contribution trace contains no token records");
    }

    let mut packet_sizes = packet_counts.keys().copied().collect::<Vec<_>>();
    packet_sizes.sort_unstable_by(|left, right| right.cmp(left));
    let mut results = Vec::new();
    let mut fixed_budget_results = Vec::new();
    for packet_neurons in packet_sizes {
        let packets_per_expert = packet_counts[&packet_neurons];
        for (target_index, &target) in relative_errors.iter().enumerate() {
            let accumulator = accumulators
                .remove(&(packet_neurons, target_index))
                .ok_or_else(|| anyhow::anyhow!("missing contribution oracle accumulator"))?;
            let expert_observations = accumulator.selected_packets.len();
            let selected_total = accumulator.selected_packets.iter().sum::<u64>();
            results.push(ContributionOracleResult {
                packet_neurons,
                packets_per_expert,
                relative_l2_error_target: target,
                expert_observations,
                unmet_target_observations: accumulator.unmet_target_observations,
                mean_selected_packets_per_expert: selected_total as f64
                    / expert_observations.max(1) as f64,
                p50_selected_packets_per_expert: percentile(&accumulator.selected_packets, 0.50),
                p95_selected_packets_per_expert: percentile(&accumulator.selected_packets, 0.95),
                p99_selected_packets_per_expert: percentile(&accumulator.selected_packets, 0.99),
                mean_selected_fraction: selected_total as f64
                    / (expert_observations.max(1) * packets_per_expert) as f64,
                mean_achieved_relative_l2_error: accumulator
                    .achieved_relative_l2_error
                    .iter()
                    .sum::<f64>()
                    / accumulator.achieved_relative_l2_error.len().max(1) as f64,
                mean_active_packet_bytes_per_token: accumulator
                    .active_bytes_per_token
                    .iter()
                    .sum::<u64>() as f64
                    / accumulator.active_bytes_per_token.len().max(1) as f64,
                p50_active_packet_bytes_per_token: percentile(
                    &accumulator.active_bytes_per_token,
                    0.50,
                ),
                p95_active_packet_bytes_per_token: percentile(
                    &accumulator.active_bytes_per_token,
                    0.95,
                ),
                p99_active_packet_bytes_per_token: percentile(
                    &accumulator.active_bytes_per_token,
                    0.99,
                ),
            });
        }
        for (fraction_index, &fraction) in selected_fractions.iter().enumerate() {
            let errors = fixed_budget_errors
                .remove(&(packet_neurons, fraction_index))
                .ok_or_else(|| anyhow::anyhow!("missing fixed-budget contribution errors"))?;
            let selected = ((packets_per_expert as f64 * fraction).ceil() as usize)
                .clamp(1, packets_per_expert);
            let packet_bytes = expert_bytes / packets_per_expert as u64;
            fixed_budget_results.push(FixedBudgetContributionResult {
                packet_neurons,
                packets_per_expert,
                requested_selected_fraction: fraction,
                selected_packets_per_expert: selected,
                actual_selected_fraction: selected as f64 / packets_per_expert as f64,
                expert_observations: errors.len(),
                mean_relative_l2_error: errors.iter().sum::<f64>() / errors.len().max(1) as f64,
                p50_relative_l2_error: percentile_f64(&errors, 0.50).unwrap_or(0.0),
                p95_relative_l2_error: percentile_f64(&errors, 0.95).unwrap_or(0.0),
                p99_relative_l2_error: percentile_f64(&errors, 0.99).unwrap_or(0.0),
                active_packet_bytes_per_token: (header.layers as u64)
                    .saturating_mul(header.top_k as u64)
                    .saturating_mul(selected as u64)
                    .saturating_mul(packet_bytes),
            });
        }
    }
    let reconstruction_max = reconstruction_errors
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(0.0);
    let report = ContributionOracleReport {
        schema_version: 1,
        classification: "down-contribution-greedy-oracle-smoke",
        selection_semantics: "perfect-knowledge-greedy-minimum-next-residual",
        quality_semantics: "per-expert-quantized-down-output-relative-l2-not-logit-or-nll",
        trace: trace.display().to_string(),
        trace_header: header,
        expert_bytes,
        token_records,
        reconstruction_relative_l2_error_p95: percentile_f64(&reconstruction_errors, 0.95)
            .unwrap_or(0.0),
        reconstruction_relative_l2_error_p99: percentile_f64(&reconstruction_errors, 0.99)
            .unwrap_or(0.0),
        reconstruction_relative_l2_error_max: reconstruction_max,
        results,
        fixed_budget_results,
    };
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("create {}", output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("contribution oracle report: {}", output.display());
    println!(
        "down reconstruction relative L2: p95 {:.6}, p99 {:.6}, max {:.6}",
        report.reconstruction_relative_l2_error_p95,
        report.reconstruction_relative_l2_error_p99,
        report.reconstruction_relative_l2_error_max,
    );
    for result in &report.results {
        println!(
            "{} neurons {:.3}% relative L2: mean {:.2}/{}, p95 {}, {:.2} MiB/token, unmet {}",
            result.packet_neurons,
            result.relative_l2_error_target * 100.0,
            result.mean_selected_packets_per_expert,
            result.packets_per_expert,
            result.p95_selected_packets_per_expert,
            result.mean_active_packet_bytes_per_token / (1024.0 * 1024.0),
            result.unmet_target_observations,
        );
    }
    for result in &report.fixed_budget_results {
        println!(
            "{} neurons keep {:.1}% ({}/{}): mean error {:.2}%, p95 {:.2}%, {:.2} MiB/token",
            result.packet_neurons,
            result.actual_selected_fraction * 100.0,
            result.selected_packets_per_expert,
            result.packets_per_expert,
            result.mean_relative_l2_error * 100.0,
            result.p95_relative_l2_error * 100.0,
            result.active_packet_bytes_per_token as f64 / (1024.0 * 1024.0),
        );
    }
    Ok(())
}

fn validate_matrix_shape(
    label: &str,
    values: &[Vec<f32>],
    layers: usize,
    top_k: usize,
) -> Result<()> {
    if values.len() != layers || values.iter().any(|layer| layer.len() != top_k) {
        bail!("{label} has an invalid layer or expert-slot shape");
    }
    Ok(())
}

fn validate_contribution_metrics(
    metrics: &PacketContributionMetrics,
    header: &TraceHeader,
) -> Result<()> {
    for (label, values) in [
        ("down contribution squared-l2", &metrics.squared_l2),
        (
            "down contribution greedy residual",
            &metrics.greedy_relative_squared_error,
        ),
    ] {
        if values.len() != header.layers {
            bail!("{label} has the wrong layer count");
        }
        for layer in values {
            if layer.len() != header.top_k || layer.iter().any(Vec::is_empty) {
                bail!("{label} has an invalid expert-slot shape");
            }
            let packets = layer[0].len();
            if layer.iter().any(|expert| expert.len() != packets) {
                bail!("{label} has inconsistent packet counts");
            }
            if layer
                .iter()
                .flatten()
                .any(|value| !value.is_finite() || *value < 0.0)
            {
                bail!("{label} contains a non-finite or negative value");
            }
        }
    }
    Ok(())
}

fn select_contribution_packets(curve: &[f32], target: f64) -> Result<(usize, f64, bool)> {
    if curve.is_empty() || curve.iter().any(|value| !value.is_finite() || *value < 0.0) {
        bail!("contribution residual curve is empty, non-finite, or negative");
    }
    for (index, &relative_squared_error) in curve.iter().enumerate() {
        let relative_error = (relative_squared_error as f64).sqrt();
        if relative_error <= target {
            return Ok((index + 1, relative_error, true));
        }
    }
    Ok((
        curve.len(),
        (*curve.last().expect("curve is non-empty") as f64).sqrt(),
        false,
    ))
}

fn simulate(args: SimulateArgs) -> Result<()> {
    let SimulateArgs {
        trace,
        calibration,
        policies,
        budgets,
        expert_bytes,
        expert_neurons,
        packet_neurons,
        layouts,
        io_bandwidth_mib_s,
        io_fixed_us,
        io_parallelism,
        reset_between_requests,
        output,
    } = args;
    let calibration = calibration.as_deref();
    if policies.is_empty() || budgets.is_empty() || packet_neurons.is_empty() || layouts.is_empty()
    {
        bail!("provide at least one cache policy, budget, packet size, and layout");
    }
    if expert_bytes == 0 || expert_neurons == 0 {
        bail!("expert-bytes and expert-neurons must be positive");
    }
    if io_fixed_us < 0.0 || !io_fixed_us.is_finite() || io_parallelism == 0 {
        bail!("io-fixed-us must be finite and non-negative and io-parallelism must be positive");
    }
    let io_shape = io_bandwidth_mib_s
        .map(|bandwidth_mib_s| {
            if bandwidth_mib_s <= 0.0 || !bandwidth_mib_s.is_finite() {
                bail!("io-bandwidth-mib-s must be finite and positive");
            }
            Ok(IoShape {
                bandwidth_mib_s,
                fixed_latency_us: io_fixed_us,
                max_inflight_fetches: io_parallelism,
            })
        })
        .transpose()?;
    if io_shape.is_none() && io_fixed_us != 0.0 {
        bail!("io-fixed-us requires io-bandwidth-mib-s");
    }
    let trace_header = read_trace_header(&trace)?;
    let hot_frequency = calibration
        .map(|path| collect_frequency(path, &trace_header))
        .transpose()?;
    let co_selection_ranks = calibration
        .map(|path| collect_co_selection_ranks(path, &trace_header))
        .transpose()?;
    let mut results = Vec::new();
    for &neurons in &packet_neurons {
        if neurons == 0
            || !expert_neurons.is_multiple_of(neurons)
            || !expert_bytes.is_multiple_of((expert_neurons / neurons) as u64)
        {
            bail!(
                "packet-neurons {neurons} must divide expert-neurons {expert_neurons} and expert-bytes {expert_bytes} exactly"
            );
        }
        let packets_per_expert = expert_neurons / neurons;
        let packet_bytes = expert_bytes / packets_per_expert as u64;
        for &layout in &layouts {
            if layout.uses_co_selection() && co_selection_ranks.is_none() {
                bail!("co-selection layouts require a disjoint --calibration trace");
            }
            for &policy in &policies {
                if matches!(policy, CachePolicy::StaticHot) && hot_frequency.is_none() {
                    bail!("static-hot requires a disjoint --calibration trace");
                }
                for &budget in &budgets {
                    let capacity = usize::try_from(budget / packet_bytes)
                        .context("resident packet capacity does not fit usize")?;
                    if capacity == 0 {
                        bail!("budget {budget} is smaller than one neuron packet");
                    }
                    let hot = hot_frequency.as_ref().map(|frequency| {
                        hottest_packet_keys(
                            frequency,
                            capacity,
                            trace_header.layers,
                            trace_header.experts,
                            packets_per_expert,
                        )
                    });
                    results.push(simulate_cell(SimulationCellConfig {
                        trace: &trace,
                        expected_header: &trace_header,
                        policy,
                        layout,
                        co_selection_ranks: co_selection_ranks.as_deref(),
                        budget_bytes: budget,
                        capacity,
                        packet_neurons: neurons,
                        packets_per_expert,
                        packet_bytes,
                        io_shape,
                        reset_between_requests,
                        static_hot: hot.as_ref(),
                    })?);
                }
            }
        }
    }
    let report = SimulationReport {
        schema_version: SIMULATION_SCHEMA_VERSION,
        trace: trace.display().to_string(),
        calibration: calibration.map(|path| path.display().to_string()),
        trace_header,
        expert_bytes,
        expert_neurons,
        packet_neurons,
        layouts,
        io_shape,
        results,
    };
    let mut writer = BufWriter::new(
        File::create(&output).with_context(|| format!("create {}", output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report).context("write simulation report")?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("report: {}", output.display());
    for result in &report.results {
        println!(
            "{} {} {} neurons {} bytes: hit {:.3}% p95 miss {:.2} MiB p99 miss layers {}",
            result.policy,
            result.physical_layout,
            result.packet_neurons,
            result.budget_bytes,
            100.0 * result.byte_weighted_hit_rate,
            result.p95_missing_bytes_per_token as f64 / (1024.0 * 1024.0),
            result.p99_layers_with_miss_per_token
        );
    }
    Ok(())
}

type ExpertKey = (u16, u16);
type PacketKey = (u16, u16, u16);

struct SimulationCellConfig<'a> {
    trace: &'a Path,
    expected_header: &'a TraceHeader,
    policy: CachePolicy,
    layout: PhysicalLayout,
    co_selection_ranks: Option<&'a [Vec<u16>]>,
    budget_bytes: u64,
    capacity: usize,
    packet_neurons: usize,
    packets_per_expert: usize,
    packet_bytes: u64,
    io_shape: Option<IoShape>,
    reset_between_requests: bool,
    static_hot: Option<&'a HashSet<PacketKey>>,
}

fn simulate_cell(config: SimulationCellConfig<'_>) -> Result<SimulationResult> {
    let mut state = match config.policy {
        CachePolicy::Lru => CacheState::Lru(LruCache::new(config.capacity)),
        CachePolicy::StaticHot => CacheState::Static(
            config
                .static_hot
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing static hot set"))?,
        ),
    };
    let mut seen_ever = HashSet::new();
    let mut token_records = 0usize;
    let mut hits = 0u64;
    let mut misses = 0u64;
    let mut compulsory_misses = 0u64;
    let mut evictions = 0u64;
    let mut missing_bytes = Vec::new();
    let mut missing_bytes_per_layer = Vec::new();
    let mut contiguous_bytes_per_fetch = Vec::new();
    let mut maximum_contiguous_burst_bytes = 0u64;
    let mut transfer_runs = Vec::new();
    let mut miss_layers = Vec::new();
    let mut reuse_distances = Vec::new();
    let mut last_access = HashMap::<PacketKey, u64>::new();
    let mut access_clock = 0u64;
    let mut serial_io_ms = Vec::new();
    let mut ideal_layer_parallel_io_ms = Vec::new();
    let mut shaped_io_ms = Vec::new();
    let mut overlap_sum = 0.0f64;
    let mut overlap_count = 0u64;
    let mut previous_request = None::<String>;
    let mut previous_keys = HashSet::new();

    visit_trace(config.trace, config.expected_header, |token| {
        if config.reset_between_requests
            && previous_request
                .as_ref()
                .is_some_and(|previous| previous != &token.request_id)
        {
            state.clear();
            previous_keys.clear();
            last_access.clear();
        }
        let mut current_keys =
            HashSet::with_capacity(config.expected_header.layers * config.expected_header.top_k);
        let mut token_misses = 0u64;
        let mut token_transfer_runs = 0u64;
        let mut token_miss_layers = 0u64;
        let mut token_serial_io_ms = 0.0f64;
        let mut token_parallel_io_ms = 0.0f64;
        let mut token_shaped_io_ms = 0.0f64;
        for (layer, experts) in token.routes.iter().enumerate() {
            let mut layer_missed = false;
            let mut layer_missing_ordinals = Vec::new();
            let mut layer_missing_packets = 0u64;
            for &expert in experts {
                current_keys.insert((layer as u16, expert));
                for packet in 0..config.packets_per_expert {
                    let key = (layer as u16, expert, packet as u16);
                    if let Some(previous) = last_access.insert(key, access_clock) {
                        reuse_distances.push(access_clock.saturating_sub(previous + 1));
                    }
                    access_clock += 1;
                    let hit = state.access(key, &mut evictions);
                    if hit {
                        hits += 1;
                    } else {
                        misses += 1;
                        token_misses += 1;
                        layer_missing_packets += 1;
                        layer_missed = true;
                        layer_missing_ordinals.push(physical_packet_ordinal(
                            config.layout,
                            layer,
                            expert,
                            packet,
                            config.expected_header.experts,
                            config.packets_per_expert,
                            config.co_selection_ranks,
                        ));
                        if seen_ever.insert(key) {
                            compulsory_misses += 1;
                        }
                    }
                }
            }
            let run_lengths = contiguous_run_lengths(&mut layer_missing_ordinals);
            let layer_runs = run_lengths.len() as u64;
            token_transfer_runs += layer_runs;
            let layer_missing_bytes = layer_missing_packets.saturating_mul(config.packet_bytes);
            missing_bytes_per_layer.push(layer_missing_bytes);
            let layer_run_bytes = run_lengths
                .into_iter()
                .map(|run_packets| (run_packets as u64).saturating_mul(config.packet_bytes))
                .collect::<Vec<_>>();
            for &run_bytes in &layer_run_bytes {
                maximum_contiguous_burst_bytes = maximum_contiguous_burst_bytes.max(run_bytes);
                contiguous_bytes_per_fetch.push(run_bytes);
            }
            if let Some(io_shape) = config.io_shape {
                token_serial_io_ms += transfer_time_ms(layer_missing_bytes, layer_runs, io_shape);
                token_parallel_io_ms +=
                    transfer_time_ms(layer_missing_bytes, u64::from(layer_runs > 0), io_shape);
                token_shaped_io_ms +=
                    shaped_transfer_time_ms(layer_missing_bytes, layer_runs, io_shape);
            }
            token_miss_layers += u64::from(layer_missed);
        }
        if previous_request.as_deref() == Some(token.request_id.as_str())
            && !previous_keys.is_empty()
        {
            let shared = current_keys.intersection(&previous_keys).count();
            overlap_sum += shared as f64 / current_keys.len().max(1) as f64;
            overlap_count += 1;
        }
        previous_request = Some(token.request_id.clone());
        previous_keys = current_keys;
        missing_bytes.push(token_misses.saturating_mul(config.packet_bytes));
        transfer_runs.push(token_transfer_runs);
        miss_layers.push(token_miss_layers);
        if config.io_shape.is_some() {
            serial_io_ms.push(token_serial_io_ms);
            ideal_layer_parallel_io_ms.push(token_parallel_io_ms);
            shaped_io_ms.push(token_shaped_io_ms);
        }
        token_records += 1;
        Ok(())
    })?;
    if token_records == 0 {
        bail!("trace contains no token records");
    }
    let route_lookups = hits + misses;
    let total_missing_bytes = misses.saturating_mul(config.packet_bytes);
    Ok(SimulationResult {
        policy: config.policy.label().to_owned(),
        physical_layout: config.layout.label().to_owned(),
        budget_bytes: config.budget_bytes,
        packet_neurons: config.packet_neurons,
        packets_per_expert: config.packets_per_expert,
        packet_bytes: config.packet_bytes,
        resident_packet_capacity: config.capacity,
        token_records,
        expert_route_lookups: token_records as u64
            * config.expected_header.layers as u64
            * config.expected_header.top_k as u64,
        route_lookups,
        hits,
        misses,
        byte_weighted_hit_rate: hits as f64 / route_lookups.max(1) as f64,
        compulsory_misses,
        capacity_misses: misses.saturating_sub(compulsory_misses),
        total_missing_bytes,
        mean_missing_bytes_per_token: total_missing_bytes as f64 / token_records as f64,
        p50_missing_bytes_per_token: percentile(&missing_bytes, 0.50),
        p95_missing_bytes_per_token: percentile(&missing_bytes, 0.95),
        p99_missing_bytes_per_token: percentile(&missing_bytes, 0.99),
        p50_missing_bytes_per_layer: percentile(&missing_bytes_per_layer, 0.50),
        p95_missing_bytes_per_layer: percentile(&missing_bytes_per_layer, 0.95),
        p99_missing_bytes_per_layer: percentile(&missing_bytes_per_layer, 0.99),
        maximum_missing_bytes_per_layer: missing_bytes_per_layer.iter().copied().max().unwrap_or(0),
        maximum_contiguous_burst_bytes,
        mean_transfer_runs_per_token: transfer_runs.iter().sum::<u64>() as f64
            / token_records as f64,
        p95_transfer_runs_per_token: percentile(&transfer_runs, 0.95),
        p99_transfer_runs_per_token: percentile(&transfer_runs, 0.99),
        mean_contiguous_bytes_per_fetch: contiguous_bytes_per_fetch.iter().sum::<u64>() as f64
            / contiguous_bytes_per_fetch.len().max(1) as f64,
        p50_contiguous_bytes_per_fetch: percentile(&contiguous_bytes_per_fetch, 0.50),
        p95_contiguous_bytes_per_fetch: percentile(&contiguous_bytes_per_fetch, 0.95),
        p99_contiguous_bytes_per_fetch: percentile(&contiguous_bytes_per_fetch, 0.99),
        p50_reuse_distance_packet_lookups: percentile(&reuse_distances, 0.50),
        p95_reuse_distance_packet_lookups: percentile(&reuse_distances, 0.95),
        p99_reuse_distance_packet_lookups: percentile(&reuse_distances, 0.99),
        mean_layers_with_miss_per_token: miss_layers.iter().sum::<u64>() as f64
            / token_records as f64,
        p95_layers_with_miss_per_token: percentile(&miss_layers, 0.95),
        p99_layers_with_miss_per_token: percentile(&miss_layers, 0.99),
        evictions,
        mean_evictions_per_token: evictions as f64 / token_records as f64,
        mean_adjacent_route_overlap: overlap_sum / overlap_count.max(1) as f64,
        mean_serial_io_ms_per_token: mean_f64(&serial_io_ms),
        p95_serial_io_ms_per_token: percentile_f64(&serial_io_ms, 0.95),
        p99_serial_io_ms_per_token: percentile_f64(&serial_io_ms, 0.99),
        mean_ideal_layer_parallel_io_ms_per_token: mean_f64(&ideal_layer_parallel_io_ms),
        p95_ideal_layer_parallel_io_ms_per_token: percentile_f64(&ideal_layer_parallel_io_ms, 0.95),
        p99_ideal_layer_parallel_io_ms_per_token: percentile_f64(&ideal_layer_parallel_io_ms, 0.99),
        mean_shaped_io_ms_per_token: mean_f64(&shaped_io_ms),
        p95_shaped_io_ms_per_token: percentile_f64(&shaped_io_ms, 0.95),
        p99_shaped_io_ms_per_token: percentile_f64(&shaped_io_ms, 0.99),
    })
}

enum CacheState {
    Lru(LruCache),
    Static(HashSet<PacketKey>),
}

impl CacheState {
    fn access(&mut self, key: PacketKey, evictions: &mut u64) -> bool {
        match self {
            Self::Lru(cache) => cache.access(key, evictions),
            Self::Static(keys) => keys.contains(&key),
        }
    }

    fn clear(&mut self) {
        if let Self::Lru(cache) = self {
            cache.clear();
        }
    }
}

struct LruCache {
    capacity: usize,
    clock: u64,
    resident: HashMap<PacketKey, u64>,
    oldest: BinaryHeap<Reverse<(u64, PacketKey)>>,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            clock: 0,
            resident: HashMap::with_capacity(capacity),
            oldest: BinaryHeap::new(),
        }
    }

    fn access(&mut self, key: PacketKey, evictions: &mut u64) -> bool {
        self.clock += 1;
        let hit = self.resident.contains_key(&key);
        if !hit && self.resident.len() >= self.capacity {
            while let Some(Reverse((timestamp, candidate))) = self.oldest.pop() {
                if self.resident.get(&candidate) == Some(&timestamp) {
                    self.resident.remove(&candidate);
                    *evictions += 1;
                    break;
                }
            }
        }
        self.resident.insert(key, self.clock);
        self.oldest.push(Reverse((self.clock, key)));
        hit
    }

    fn clear(&mut self) {
        self.resident.clear();
        self.oldest.clear();
    }
}

fn collect_frequency(
    path: &Path,
    expected_header: &TraceHeader,
) -> Result<HashMap<ExpertKey, u64>> {
    let mut frequency = HashMap::new();
    visit_trace(path, expected_header, |token| {
        for (layer, experts) in token.routes.iter().enumerate() {
            for &expert in experts {
                *frequency.entry((layer as u16, expert)).or_insert(0) += 1;
            }
        }
        Ok(())
    })?;
    Ok(frequency)
}

fn collect_co_selection_ranks(path: &Path, expected_header: &TraceHeader) -> Result<Vec<Vec<u16>>> {
    let experts = expected_header.experts;
    let mut node_frequency = vec![vec![0u64; experts]; expected_header.layers];
    let mut edge_frequency = vec![vec![0u64; experts * experts]; expected_header.layers];
    visit_trace(path, expected_header, |token| {
        for (layer, selected) in token.routes.iter().enumerate() {
            for &expert in selected {
                node_frequency[layer][expert as usize] += 1;
            }
            for (left_index, &left) in selected.iter().enumerate() {
                for &right in &selected[left_index + 1..] {
                    edge_frequency[layer][left as usize * experts + right as usize] += 1;
                    edge_frequency[layer][right as usize * experts + left as usize] += 1;
                }
            }
        }
        Ok(())
    })?;
    Ok((0..expected_header.layers)
        .map(|layer| {
            greedy_co_selection_ranks(&node_frequency[layer], &edge_frequency[layer], experts)
        })
        .collect())
}

fn greedy_co_selection_ranks(
    node_frequency: &[u64],
    edge_frequency: &[u64],
    experts: usize,
) -> Vec<u16> {
    let mut order = Vec::with_capacity(experts);
    let mut unused = vec![true; experts];
    let first = (0..experts)
        .max_by_key(|&expert| (node_frequency[expert], Reverse(expert)))
        .unwrap_or(0);
    order.push(first);
    unused[first] = false;
    while order.len() < experts {
        let previous = *order.last().expect("co-selection order is non-empty");
        let next = (0..experts)
            .filter(|&expert| unused[expert])
            .max_by_key(|&expert| {
                (
                    edge_frequency[previous * experts + expert],
                    node_frequency[expert],
                    Reverse(expert),
                )
            })
            .expect("at least one expert remains");
        unused[next] = false;
        order.push(next);
    }
    let mut ranks = vec![0u16; experts];
    for (rank, expert) in order.into_iter().enumerate() {
        ranks[expert] = rank as u16;
    }
    ranks
}

fn physical_packet_ordinal(
    layout: PhysicalLayout,
    layer: usize,
    expert: u16,
    packet: usize,
    experts: usize,
    packets_per_expert: usize,
    co_selection_ranks: Option<&[Vec<u16>]>,
) -> usize {
    let rank = if layout.uses_co_selection() {
        co_selection_ranks.expect("co-selection ranks were validated")[layer][expert as usize]
            as usize
    } else {
        expert as usize
    };
    if layout.is_neuron_major() {
        packet * experts + rank
    } else {
        rank * packets_per_expert + packet
    }
}

fn contiguous_run_lengths(ordinals: &mut Vec<usize>) -> Vec<usize> {
    if ordinals.is_empty() {
        return Vec::new();
    }
    ordinals.sort_unstable();
    ordinals.dedup();
    let mut runs = Vec::new();
    let mut run_length = 1usize;
    for pair in ordinals.windows(2) {
        if pair[1] == pair[0] + 1 {
            run_length += 1;
        } else {
            runs.push(run_length);
            run_length = 1;
        }
    }
    runs.push(run_length);
    runs
}

fn hottest_packet_keys(
    frequency: &HashMap<ExpertKey, u64>,
    capacity: usize,
    layers: usize,
    experts: usize,
    packets_per_expert: usize,
) -> HashSet<PacketKey> {
    let mut ranked = frequency
        .iter()
        .flat_map(|(&(layer, expert), &count)| {
            (0..packets_per_expert)
                .map(move |packet| (Reverse(count), (layer, expert, packet as u16)))
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    let mut selected = ranked
        .into_iter()
        .take(capacity)
        .map(|(_, key)| key)
        .collect::<HashSet<_>>();
    for layer in 0..layers {
        for expert in 0..experts {
            for packet in 0..packets_per_expert {
                if selected.len() >= capacity {
                    return selected;
                }
                selected.insert((layer as u16, expert as u16, packet as u16));
            }
        }
    }
    selected
}

fn transfer_time_ms(bytes: u64, runs: u64, shape: IoShape) -> f64 {
    let bandwidth_bytes_s = shape.bandwidth_mib_s * 1024.0 * 1024.0;
    bytes as f64 / bandwidth_bytes_s * 1000.0 + runs as f64 * shape.fixed_latency_us / 1000.0
}

fn shaped_transfer_time_ms(bytes: u64, runs: u64, shape: IoShape) -> f64 {
    if runs == 0 {
        return 0.0;
    }
    let latency_rounds = runs.div_ceil(shape.max_inflight_fetches as u64);
    transfer_time_ms(bytes, latency_rounds, shape)
}

fn visit_trace(
    path: &Path,
    expected_header: &TraceHeader,
    mut visitor: impl FnMut(TokenRoutes) -> Result<()>,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut lines = BufReader::new(file).lines();
    let first = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty trace {}", path.display()))??;
    let TraceLine::Header(header) = serde_json::from_str::<TraceLine>(&first)
        .with_context(|| format!("parse header from {}", path.display()))?
    else {
        bail!("first trace line is not a header");
    };
    validate_compatible_headers(expected_header, &header)?;
    for (index, line) in lines.enumerate() {
        let line = line.with_context(|| format!("read line {}", index + 2))?;
        let TraceLine::Token(token) = serde_json::from_str::<TraceLine>(&line)
            .with_context(|| format!("parse line {} from {}", index + 2, path.display()))?
        else {
            bail!("unexpected repeated header at line {}", index + 2);
        };
        validate_routes(&token.routes, header.layers, header.top_k, header.experts)?;
        visitor(token)?;
    }
    Ok(())
}

fn read_trace_header(path: &Path) -> Result<TraceHeader> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let line = BufReader::new(file)
        .lines()
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty trace {}", path.display()))??;
    let TraceLine::Header(header) = serde_json::from_str::<TraceLine>(&line)? else {
        bail!("first trace line is not a header");
    };
    if header.schema_version != SCHEMA_VERSION {
        bail!("unsupported route trace schema {}", header.schema_version);
    }
    Ok(header)
}

fn validate_compatible_headers(expected: &TraceHeader, actual: &TraceHeader) -> Result<()> {
    if expected.schema_version != actual.schema_version
        || expected.model_sha256 != actual.model_sha256
        || expected.layers != actual.layers
        || expected.experts != actual.experts
        || expected.top_k != actual.top_k
    {
        bail!("route traces have incompatible headers");
    }
    Ok(())
}

fn validate_routes(routes: &[Vec<u16>], layers: usize, top_k: usize, experts: usize) -> Result<()> {
    if routes.len() != layers {
        bail!(
            "route record has {} layers; expected {layers}",
            routes.len()
        );
    }
    for (layer, ids) in routes.iter().enumerate() {
        if ids.len() != top_k {
            bail!(
                "route layer {layer} has {} experts; expected {top_k}",
                ids.len()
            );
        }
        let mut unique = HashSet::with_capacity(top_k);
        for &expert in ids {
            if expert as usize >= experts {
                bail!("route layer {layer} expert {expert} is outside 0..{experts}");
            }
            if !unique.insert(expert) {
                bail!("route layer {layer} repeats expert {expert}");
            }
        }
    }
    Ok(())
}

fn write_trace_line(writer: &mut impl Write, line: &TraceLine) -> Result<()> {
    serde_json::to_writer(&mut *writer, line)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn percentile(values: &[u64], quantile: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn percentile_f64(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    Some(sorted[index.min(sorted.len() - 1)])
}

fn parse_bytes(value: &str) -> std::result::Result<u64, String> {
    let normalized = value.trim().to_ascii_lowercase();
    let suffixes = [
        ("gib", 1024u64.pow(3)),
        ("gb", 1_000_000_000u64),
        ("mib", 1024u64.pow(2)),
        ("mb", 1_000_000u64),
        ("kib", 1024u64),
        ("kb", 1_000u64),
    ];
    for (suffix, multiplier) in suffixes {
        if let Some(number) = normalized.strip_suffix(suffix) {
            let parsed = number
                .trim()
                .parse::<u64>()
                .map_err(|error| error.to_string())?;
            return parsed
                .checked_mul(multiplier)
                .ok_or_else(|| "byte quantity overflow".to_owned());
        }
    }
    normalized.parse::<u64>().map_err(|error| error.to_string())
}

fn validate_sha256(value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("model-sha256 must contain 64 hexadecimal characters");
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 4 * 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(target_os = "macos")]
mod llama_capture {
    use super::*;

    unsafe extern "C" {
        fn qx_llama_route_open(
            path: *const c_char,
            context_length: u32,
            layers: c_int,
            top_k: c_int,
            gpu_layers: c_int,
            threads: c_int,
            intermediate: c_int,
            hidden: c_int,
            contribution_packet_neurons: *const c_int,
            contribution_size_count: c_int,
        ) -> *mut c_void;
        fn qx_llama_route_close(route: *mut c_void);
        fn qx_llama_route_error(route: *const c_void) -> *const c_char;
        fn qx_llama_route_tokenize(
            route: *const c_void,
            text: *const c_char,
            text_length: c_int,
            tokens: *mut c_int,
            capacity: c_int,
        ) -> c_int;
        fn qx_llama_route_eos(route: *const c_void) -> c_int;
        fn qx_llama_route_step(
            route: *mut c_void,
            input: c_int,
            output: *mut c_int,
            routes: *mut c_int,
            route_count: c_int,
            activations: *mut f32,
            activation_count: c_int,
            contribution_norms: *mut f32,
            contribution_residuals: *mut f32,
            contribution_metric_count: c_int,
            contribution_reconstruction: *mut f32,
            contribution_reconstruction_count: c_int,
        ) -> c_int;
    }

    pub struct Capture {
        raw: *mut c_void,
        layers: usize,
        top_k: usize,
        intermediate: Option<usize>,
        contribution_packet_neurons: Vec<usize>,
        contribution_total_packets: usize,
    }

    pub struct StepOutput {
        pub predicted: i32,
        pub routes: Vec<Vec<u16>>,
        pub activations: Option<Vec<f32>>,
        pub down_contribution: Option<PacketDownContribution>,
    }

    pub struct OpenConfig<'a> {
        pub context: usize,
        pub layers: usize,
        pub top_k: usize,
        pub gpu_layers: i32,
        pub threads: i32,
        pub intermediate: Option<usize>,
        pub hidden: usize,
        pub contribution_packet_neurons: &'a [usize],
    }

    impl Capture {
        pub fn open(path: &Path, config: OpenConfig<'_>) -> Result<Self> {
            let OpenConfig {
                context,
                layers,
                top_k,
                gpu_layers,
                threads,
                intermediate,
                hidden,
                contribution_packet_neurons,
            } = config;
            let path = CString::new(path.to_string_lossy().as_bytes())?;
            let contribution_packet_neurons_i32 = contribution_packet_neurons
                .iter()
                .map(|&value| {
                    value
                        .try_into()
                        .context("contribution packet size exceeds i32")
                })
                .collect::<Result<Vec<c_int>>>()?;
            let raw = unsafe {
                qx_llama_route_open(
                    path.as_ptr(),
                    context.try_into()?,
                    layers.try_into()?,
                    top_k.try_into()?,
                    gpu_layers,
                    threads,
                    intermediate.unwrap_or(0).try_into()?,
                    hidden.try_into()?,
                    contribution_packet_neurons_i32.as_ptr(),
                    contribution_packet_neurons_i32.len().try_into()?,
                )
            };
            if raw.is_null() {
                bail!("failed to initialize llama.cpp route capture");
            }
            Ok(Self {
                raw,
                layers,
                top_k,
                intermediate,
                contribution_total_packets: intermediate
                    .map(|width| {
                        contribution_packet_neurons
                            .iter()
                            .map(|packet| width / packet)
                            .sum()
                    })
                    .unwrap_or(0),
                contribution_packet_neurons: contribution_packet_neurons.to_vec(),
            })
        }

        pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
            let text = CString::new(text)?;
            let required = unsafe {
                qx_llama_route_tokenize(
                    self.raw,
                    text.as_ptr(),
                    text.as_bytes().len().try_into()?,
                    std::ptr::null_mut(),
                    0,
                )
            };
            let capacity = if required < 0 { -required } else { required };
            if capacity <= 0 {
                bail!("llama tokenizer returned no tokens");
            }
            let mut tokens = vec![0i32; capacity as usize];
            let count = unsafe {
                qx_llama_route_tokenize(
                    self.raw,
                    text.as_ptr(),
                    text.as_bytes().len().try_into()?,
                    tokens.as_mut_ptr(),
                    capacity,
                )
            };
            if count < 0 {
                bail!("llama tokenizer capacity negotiation failed");
            }
            tokens.truncate(count as usize);
            Ok(tokens)
        }

        pub fn eos(&self) -> i32 {
            unsafe { qx_llama_route_eos(self.raw) }
        }

        pub fn step(&mut self, input: i32) -> Result<StepOutput> {
            let mut output = 0i32;
            let mut flat = vec![-1i32; self.layers * self.top_k];
            let mut activations = self
                .intermediate
                .map(|width| vec![0.0f32; self.layers * self.top_k * width]);
            let (activation_pointer, activation_count) = activations
                .as_mut()
                .map(|values| {
                    Ok::<_, anyhow::Error>((values.as_mut_ptr(), values.len().try_into()?))
                })
                .transpose()?
                .unwrap_or((std::ptr::null_mut(), 0));
            let metric_count = self.layers * self.top_k * self.contribution_total_packets;
            let mut contribution_norms =
                (!self.contribution_packet_neurons.is_empty()).then(|| vec![0.0f32; metric_count]);
            let mut contribution_residuals =
                (!self.contribution_packet_neurons.is_empty()).then(|| vec![0.0f32; metric_count]);
            let mut contribution_reconstruction = (!self.contribution_packet_neurons.is_empty())
                .then(|| vec![0.0f32; self.layers * self.top_k]);
            let contribution_metric_count: c_int = metric_count.try_into()?;
            let contribution_reconstruction_count: c_int = contribution_reconstruction
                .as_ref()
                .map_or(0, Vec::len)
                .try_into()?;
            let status = unsafe {
                qx_llama_route_step(
                    self.raw,
                    input,
                    &mut output,
                    flat.as_mut_ptr(),
                    flat.len().try_into()?,
                    activation_pointer,
                    activation_count,
                    contribution_norms
                        .as_mut()
                        .map_or(std::ptr::null_mut(), Vec::as_mut_ptr),
                    contribution_residuals
                        .as_mut()
                        .map_or(std::ptr::null_mut(), Vec::as_mut_ptr),
                    contribution_metric_count,
                    contribution_reconstruction
                        .as_mut()
                        .map_or(std::ptr::null_mut(), Vec::as_mut_ptr),
                    contribution_reconstruction_count,
                )
            };
            if status != 0 {
                bail!("llama route step failed ({status}): {}", self.error());
            }
            let routes = flat
                .chunks_exact(self.top_k)
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|&value| u16::try_from(value).context("negative expert ID"))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            let down_contribution = contribution_norms
                .zip(contribution_residuals)
                .zip(contribution_reconstruction)
                .map(|((norms, residuals), reconstruction)| {
                    self.structure_contributions(norms, residuals, reconstruction)
                })
                .transpose()?;
            Ok(StepOutput {
                predicted: output,
                routes,
                activations,
                down_contribution,
            })
        }

        fn structure_contributions(
            &self,
            norms: Vec<f32>,
            residuals: Vec<f32>,
            reconstruction: Vec<f32>,
        ) -> Result<PacketDownContribution> {
            let mut reconstruction_relative_squared_error =
                vec![vec![0.0f32; self.top_k]; self.layers];
            for (index, value) in reconstruction.into_iter().enumerate() {
                reconstruction_relative_squared_error[index / self.top_k][index % self.top_k] =
                    value;
            }
            let mut packets = self
                .contribution_packet_neurons
                .iter()
                .map(|&packet_neurons| {
                    let count =
                        self.intermediate.expect("contribution capture has width") / packet_neurons;
                    PacketContributionMetrics {
                        packet_neurons,
                        squared_l2: vec![vec![vec![0.0; count]; self.top_k]; self.layers],
                        greedy_relative_squared_error: vec![
                            vec![vec![0.0; count]; self.top_k];
                            self.layers
                        ],
                    }
                })
                .collect::<Vec<_>>();
            for layer in 0..self.layers {
                for slot in 0..self.top_k {
                    let expert_base = (layer * self.top_k + slot) * self.contribution_total_packets;
                    let mut packet_offset = 0usize;
                    for metrics in &mut packets {
                        let count = metrics.squared_l2[layer][slot].len();
                        let range =
                            expert_base + packet_offset..expert_base + packet_offset + count;
                        metrics.squared_l2[layer][slot].copy_from_slice(&norms[range.clone()]);
                        metrics.greedy_relative_squared_error[layer][slot]
                            .copy_from_slice(&residuals[range]);
                        packet_offset += count;
                    }
                }
            }
            Ok(PacketDownContribution {
                reconstruction_relative_squared_error,
                packets,
            })
        }

        fn error(&self) -> String {
            unsafe {
                let pointer = qx_llama_route_error(self.raw);
                if pointer.is_null() {
                    return "unknown error".to_owned();
                }
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        }
    }

    impl Drop for Capture {
        fn drop(&mut self) {
            unsafe { qx_llama_route_close(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_parser_distinguishes_decimal_and_binary_units() {
        assert_eq!(parse_bytes("3GiB").unwrap(), 3 * 1024u64.pow(3));
        assert_eq!(parse_bytes("3gb").unwrap(), 3_000_000_000);
        assert_eq!(parse_bytes("2048").unwrap(), 2048);
    }

    #[test]
    fn lru_evicts_the_oldest_expert() {
        let mut cache = LruCache::new(2);
        let mut evictions = 0;
        assert!(!cache.access((0, 1, 0), &mut evictions));
        assert!(!cache.access((0, 2, 0), &mut evictions));
        assert!(cache.access((0, 1, 0), &mut evictions));
        assert!(!cache.access((0, 3, 0), &mut evictions));
        assert!(cache.access((0, 1, 0), &mut evictions));
        assert!(!cache.access((0, 2, 0), &mut evictions));
        assert_eq!(evictions, 2);
    }

    #[test]
    fn hottest_keys_are_selected_by_frequency() {
        let frequency = HashMap::from([((0, 1), 4), ((0, 2), 9), ((1, 0), 6)]);
        let hot = hottest_packet_keys(&frequency, 2, 2, 3, 1);
        assert_eq!(hot, HashSet::from([(0, 2, 0), (1, 0, 0)]));
    }

    #[test]
    fn hottest_keys_fill_unused_capacity_with_cold_experts() {
        let hot = hottest_packet_keys(&HashMap::from([((1, 1), 9)]), 4, 2, 2, 1);
        assert_eq!(hot.len(), 4);
        assert!(hot.contains(&(1, 1, 0)));
    }

    #[test]
    fn percentile_uses_an_upper_rank() {
        assert_eq!(percentile(&[0, 1, 2, 3, 4], 0.50), 2);
        assert_eq!(percentile(&[0, 1, 2, 3, 4], 0.95), 4);
    }

    #[test]
    fn activation_oracle_selects_the_minimum_energy_prefix() {
        let (selected, retained) = select_activation_packets(&[9.0, 0.5, 0.5], 0.90).unwrap();
        assert_eq!(selected, 1);
        assert!((retained - 0.9).abs() < 1e-6);
        let (selected, retained) = select_activation_packets(&[0.0, 0.0], 0.99).unwrap();
        assert_eq!(selected, 0);
        assert_eq!(retained, 1.0);
    }

    #[test]
    fn contribution_oracle_selects_the_first_greedy_prefix_below_error() {
        let (selected, achieved, met) =
            select_contribution_packets(&[0.25, 0.04, 0.0001], 0.05).unwrap();
        assert_eq!(selected, 3);
        assert!((achieved - 0.01).abs() < 1e-6);
        assert!(met);
        let (selected, achieved, met) = select_contribution_packets(&[0.25, 0.04], 0.1).unwrap();
        assert_eq!(selected, 2);
        assert!((achieved - 0.2).abs() < 1e-6);
        assert!(!met);
    }

    #[test]
    fn synthetic_trace_reports_hits_misses_and_tail_bytes() {
        let path = std::env::temp_dir().join(format!(
            "qx-route-cache-{}-{}.jsonl",
            std::process::id(),
            SCHEMA_VERSION
        ));
        let header = TraceHeader {
            schema_version: SCHEMA_VERSION,
            model_sha256: "0".repeat(64),
            llama_revision: "test".to_owned(),
            layers: 2,
            experts: 2,
            top_k: 1,
            source: "synthetic".to_owned(),
        };
        let file = File::create(&path).unwrap();
        let mut writer = BufWriter::new(file);
        write_trace_line(&mut writer, &TraceLine::Header(header.clone())).unwrap();
        for (position, routes) in [
            vec![vec![0], vec![0]],
            vec![vec![0], vec![0]],
            vec![vec![1], vec![0]],
        ]
        .into_iter()
        .enumerate()
        {
            write_trace_line(
                &mut writer,
                &TraceLine::Token(TokenRoutes {
                    request_id: "test".to_owned(),
                    token_position: position,
                    phase: "test".to_owned(),
                    input_token: position as i32,
                    predicted_token: position as i32 + 1,
                    routes,
                    packet_activation_energy: None,
                    packet_down_contribution: None,
                }),
            )
            .unwrap();
        }
        writer.flush().unwrap();
        let result = simulate_cell(SimulationCellConfig {
            trace: &path,
            expected_header: &header,
            policy: CachePolicy::Lru,
            layout: PhysicalLayout::ExpertContiguous,
            co_selection_ranks: None,
            budget_bytes: 200,
            capacity: 2,
            packet_neurons: 1,
            packets_per_expert: 1,
            packet_bytes: 100,
            io_shape: None,
            reset_between_requests: false,
            static_hot: None,
        })
        .unwrap();
        assert_eq!(result.token_records, 3);
        assert_eq!(result.hits, 3);
        assert_eq!(result.misses, 3);
        assert_eq!(result.p95_missing_bytes_per_token, 200);
        assert_eq!(result.p99_layers_with_miss_per_token, 2);
        let packetized = simulate_cell(SimulationCellConfig {
            trace: &path,
            expected_header: &header,
            policy: CachePolicy::Lru,
            layout: PhysicalLayout::ExpertContiguous,
            co_selection_ranks: None,
            budget_bytes: 200,
            capacity: 4,
            packet_neurons: 1,
            packets_per_expert: 2,
            packet_bytes: 50,
            io_shape: Some(IoShape {
                bandwidth_mib_s: 1000.0,
                fixed_latency_us: 100.0,
                max_inflight_fetches: 2,
            }),
            reset_between_requests: false,
            static_hot: None,
        })
        .unwrap();
        assert_eq!(packetized.total_missing_bytes, result.total_missing_bytes);
        assert_eq!(
            packetized.p95_missing_bytes_per_token,
            result.p95_missing_bytes_per_token
        );
        assert_eq!(
            packetized.byte_weighted_hit_rate,
            result.byte_weighted_hit_rate
        );
        assert_eq!(packetized.maximum_contiguous_burst_bytes, 100);
        assert!(packetized.p95_serial_io_ms_per_token.is_some());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn physical_layout_changes_fetch_runs_without_changing_packets() {
        let expert_major = [
            physical_packet_ordinal(PhysicalLayout::ExpertContiguous, 0, 0, 0, 4, 2, None),
            physical_packet_ordinal(PhysicalLayout::ExpertContiguous, 0, 0, 1, 4, 2, None),
            physical_packet_ordinal(PhysicalLayout::ExpertContiguous, 0, 2, 0, 4, 2, None),
            physical_packet_ordinal(PhysicalLayout::ExpertContiguous, 0, 2, 1, 4, 2, None),
        ];
        let neuron_major = [0, 1]
            .into_iter()
            .flat_map(|packet| {
                [0, 2].map(|expert| {
                    physical_packet_ordinal(
                        PhysicalLayout::NeuronInterleaved,
                        0,
                        expert,
                        packet,
                        4,
                        2,
                        None,
                    )
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(contiguous_run_lengths(&mut expert_major.to_vec()), [2, 2]);
        assert_eq!(
            contiguous_run_lengths(&mut neuron_major.clone()),
            [1, 1, 1, 1]
        );
    }
}
