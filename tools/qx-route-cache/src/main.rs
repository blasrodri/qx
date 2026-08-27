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
const MAX_GREEDY_CONTRIBUTION_PACKETS: usize = 64;
const EXPERT_DOWN_WEIGHTS_MAGIC: &[u8; 8] = b"QXEDW001";
const JOINT_EXPERT_STUDENT_MAGIC: &[u8; 8] = b"QXEJS001";

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
        /// Capture post-SwiGLU input and teacher down output when this layer uses sample-expert.
        #[arg(long, requires = "sample_expert")]
        sample_layer: Option<usize>,
        #[arg(long, requires = "sample_layer")]
        sample_expert: Option<u16>,
        /// Binary dequantized teacher down matrix written on the first sampled hit.
        #[arg(long, requires = "sample_layer")]
        sample_weights_output: Option<PathBuf>,
        /// Trusted model SHA-256. If omitted, the complete GGUF is hashed.
        #[arg(long)]
        model_sha256: Option<String>,
    },
    /// Capture selected routed experts in one layer across reset corpus paragraphs.
    CaptureCorpus {
        #[arg(long)]
        model: PathBuf,
        #[arg(long = "corpus", required = true)]
        corpora: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        sample_weights_output: Option<PathBuf>,
        /// Layer to sample. Omit for a compact all-layer route-only trace.
        #[arg(long)]
        sample_layer: Option<usize>,
        /// Selected experts to retain, repeated or comma-delimited. Empty captures every routed expert.
        #[arg(
            long = "sample-expert",
            value_delimiter = ',',
            requires = "sample_layer"
        )]
        sample_experts: Vec<u16>,
        /// Maximum retained samples per expert. Zero retains every routed hit.
        #[arg(long, default_value_t = 0)]
        max_samples_per_expert: usize,
        #[arg(long, default_value_t = 64)]
        max_sequences: usize,
        #[arg(long, default_value_t = 96)]
        max_tokens_per_sequence: usize,
        #[arg(long, default_value_t = 128)]
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
        #[arg(long, default_value_t = DEFAULT_EXPERT_NEURONS)]
        expert_neurons: usize,
        #[arg(long, default_value_t = 2048)]
        hidden_size: usize,
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
        /// Fractions pinned by hot-lru; the remainder is an adaptive LRU cache.
        #[arg(long, value_delimiter = ',', default_value = "0.25,0.5,0.75")]
        hot_fractions: Vec<f64>,
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
        /// Only score these phases while still using earlier phases to warm the cache.
        #[arg(long = "measure-phase", value_delimiter = ',')]
        measure_phases: Vec<String>,
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
    /// Learn fixed non-contiguous neuron groups and measure held-out packet coverage.
    GroupOracle {
        #[arg(long)]
        calibration: PathBuf,
        #[arg(long)]
        trace: PathBuf,
        #[arg(long, default_value_t = 32)]
        group_neurons: usize,
        #[arg(long, value_delimiter = ',', default_value = "0.25,0.1,0.05")]
        relative_errors: Vec<f64>,
        #[arg(long, default_value_t = DEFAULT_Q3_EXPERT_BYTES)]
        expert_bytes: u64,
        #[arg(long)]
        output: PathBuf,
    },
    /// Apply calibration-trained fixed neuron groups and measure teacher-forced NLL.
    StructuredNll {
        #[arg(long)]
        model: PathBuf,
        /// Exact individual-neuron contribution traces used only for mask training.
        #[arg(long = "calibration", required = true)]
        calibrations: Vec<PathBuf>,
        /// Disjoint teacher-forced text used for the quality smoke.
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 8)]
        group_neurons: usize,
        /// Local per-expert contribution error used to label calibration neurons.
        #[arg(long, default_value_t = 0.1)]
        relative_error: f64,
        /// Fixed group fractions retained in every expert. One is an all-one-mask control.
        #[arg(long, value_delimiter = ',', default_value = "1,0.75,0.625,0.5")]
        retained_fractions: Vec<f64>,
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
        #[arg(long, default_value_t = DEFAULT_EXPERT_NEURONS)]
        expert_neurons: usize,
        #[arg(long, default_value_t = 2048)]
        hidden_size: usize,
        /// Trusted model SHA-256. If omitted, the complete GGUF is hashed.
        #[arg(long)]
        model_sha256: Option<String>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Fit and evaluate one reduced expert down projection from captured teacher samples.
    DistillExpert {
        #[arg(long = "trace", required = true)]
        traces: Vec<PathBuf>,
        /// Fully disjoint traces used only for evaluation. When omitted, the
        /// training traces are split chronologically using --train-fraction.
        #[arg(long = "heldout-trace")]
        heldout_traces: Vec<PathBuf>,
        #[arg(long)]
        teacher_down_weights: PathBuf,
        #[arg(long = "calibration", required = true)]
        calibrations: Vec<PathBuf>,
        #[arg(long)]
        layer: usize,
        #[arg(long)]
        expert: u16,
        #[arg(long, default_value_t = 8)]
        group_neurons: usize,
        #[arg(long, default_value_t = 0.5)]
        retained_fraction: f64,
        #[arg(long, default_value_t = 0.1)]
        relative_error: f64,
        #[arg(long, default_value_t = 0.75)]
        train_fraction: f64,
        /// Limit explicit-corpus training hits for learning-curve experiments.
        #[arg(long)]
        max_train_samples: Option<usize>,
        /// Export the folded student as a dense-shape F32 sidecar with zeroed
        /// pruned columns for direct execution-callback evaluation.
        #[arg(long)]
        student_down_weights_output: Option<PathBuf>,
        /// Ridge penalty relative to the mean train-kernel diagonal.
        #[arg(long, default_value_t = 1e-3)]
        ridge: f64,
        #[arg(long)]
        output: PathBuf,
    },
    /// Compare one folded expert student with the exact teacher on a corpus.
    ExpertStudentNll {
        #[arg(long)]
        model: PathBuf,
        #[arg(long = "corpus", required = true)]
        corpora: Vec<PathBuf>,
        #[arg(long)]
        student_down_weights: PathBuf,
        #[arg(long)]
        layer: usize,
        #[arg(long)]
        expert: u16,
        #[arg(long, default_value_t = 64)]
        max_sequences: usize,
        #[arg(long, default_value_t = 96)]
        max_tokens_per_sequence: usize,
        #[arg(long, default_value_t = 128)]
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
        #[arg(long, default_value_t = DEFAULT_EXPERT_NEURONS)]
        expert_neurons: usize,
        #[arg(long, default_value_t = 2048)]
        hidden_size: usize,
        #[arg(long)]
        model_sha256: Option<String>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Compare one jointly distilled gate/up/down expert student with the teacher.
    JointExpertStudentNll {
        #[arg(long)]
        model: PathBuf,
        #[arg(long = "corpus", required = true)]
        corpora: Vec<PathBuf>,
        #[arg(long)]
        student: PathBuf,
        #[arg(long, default_value_t = 64)]
        max_sequences: usize,
        #[arg(long, default_value_t = 96)]
        max_tokens_per_sequence: usize,
        #[arg(long, default_value_t = 128)]
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
        #[arg(long, default_value_t = DEFAULT_EXPERT_NEURONS)]
        expert_neurons: usize,
        #[arg(long, default_value_t = 2048)]
        hidden_size: usize,
        #[arg(long)]
        model_sha256: Option<String>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Compare several jointly distilled gate/up/down students simultaneously.
    MultiJointExpertStudentNll {
        #[arg(long)]
        model: PathBuf,
        #[arg(long = "corpus", required = true)]
        corpora: Vec<PathBuf>,
        #[arg(long = "student", required = true)]
        students: Vec<PathBuf>,
        /// Cumulatively replace matching baseline students, freeing the old matrices in place.
        #[arg(long = "replacement-student")]
        replacement_students: Vec<PathBuf>,
        /// Measure every replacement against the same complete baseline instead of cumulatively.
        #[arg(long)]
        independent_replacements: bool,
        /// Optional low-rank correction injected after aggregated MoE output.
        #[arg(long)]
        layer_residual: Option<PathBuf>,
        /// Evaluate sorted student prefixes in one loaded-model session. Empty evaluates all.
        #[arg(long, value_delimiter = ',')]
        active_prefixes: Vec<usize>,
        /// Skip this many scoreable corpus paragraphs before evaluation.
        #[arg(long, default_value_t = 0)]
        skip_sequences: usize,
        #[arg(long, default_value_t = 64)]
        max_sequences: usize,
        #[arg(long, default_value_t = 96)]
        max_tokens_per_sequence: usize,
        #[arg(long, default_value_t = 128)]
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
        #[arg(long, default_value_t = DEFAULT_EXPERT_NEURONS)]
        expert_neurons: usize,
        #[arg(long, default_value_t = 2048)]
        hidden_size: usize,
        #[arg(long)]
        model_sha256: Option<String>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Compare several simultaneously active folded students with the teacher.
    MultiExpertStudentNll {
        #[arg(long)]
        model: PathBuf,
        #[arg(long = "corpus", required = true)]
        corpora: Vec<PathBuf>,
        /// Repeated `layer:expert:path` folded student specifications.
        #[arg(long = "student", required = true)]
        students: Vec<String>,
        #[arg(long, default_value_t = 64)]
        max_sequences: usize,
        #[arg(long, default_value_t = 96)]
        max_tokens_per_sequence: usize,
        #[arg(long, default_value_t = 128)]
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
        #[arg(long, default_value_t = DEFAULT_EXPERT_NEURONS)]
        expert_neurons: usize,
        #[arg(long, default_value_t = 2048)]
        hidden_size: usize,
        #[arg(long)]
        model_sha256: Option<String>,
        #[arg(long)]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum CachePolicy {
    Lru,
    StaticHot,
    HotLru,
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
            Self::HotLru => "hot-lru",
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expert_distill_sample: Option<ExpertDistillSample>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    expert_distill_samples: Vec<ExpertDistillSample>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ExpertDistillSample {
    layer: usize,
    expert: u16,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    input: Vec<f32>,
    activation: Vec<f32>,
    teacher_down_output: Vec<f32>,
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
    #[serde(default = "default_greedy_selection_semantics")]
    selection_semantics: String,
    squared_l2: Vec<Vec<Vec<f32>>>,
    /// Relative squared error after each strategy-ranked selection, from one packet through all.
    #[serde(alias = "greedy_relative_squared_error")]
    ranked_relative_squared_error: Vec<Vec<Vec<f32>>>,
    /// Packet IDs in selection order. Absent in traces captured before rank identity support.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ranked_packet_indices: Option<Vec<Vec<Vec<u16>>>>,
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
    hot_fractions: Vec<f64>,
    measure_phases: Vec<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    hot_fraction: Option<f64>,
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
    selection_semantics: String,
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

#[derive(Debug, Serialize)]
struct GroupOracleReport {
    schema_version: u32,
    classification: &'static str,
    layout_semantics: &'static str,
    quality_semantics: &'static str,
    calibration: String,
    trace: String,
    trace_header: TraceHeader,
    expert_bytes: u64,
    expert_neurons: usize,
    group_neurons: usize,
    results: Vec<GroupOracleResult>,
}

#[derive(Debug, Serialize)]
struct GroupOracleResult {
    relative_l2_error_target: f64,
    heldout_expert_observations: usize,
    fallback_identity_observations: usize,
    mean_individual_neurons: f64,
    mean_individual_fraction: f64,
    mean_ideal_groups: f64,
    mean_frequency_packed_groups: f64,
    p50_frequency_packed_groups: u64,
    p95_frequency_packed_groups: u64,
    p99_frequency_packed_groups: u64,
    mean_frequency_group_load_amplification: f64,
    mean_frequency_resident_neuron_fraction: f64,
    mean_signature_packed_groups: f64,
    p50_signature_packed_groups: u64,
    p95_signature_packed_groups: u64,
    p99_signature_packed_groups: u64,
    mean_signature_group_load_amplification: f64,
    mean_signature_resident_neuron_fraction: f64,
    calibrated_only_observations: usize,
    calibrated_only_mean_frequency_group_load_amplification: f64,
    calibrated_only_mean_frequency_resident_neuron_fraction: f64,
    calibrated_only_p95_frequency_groups: u64,
    calibrated_only_mean_signature_group_load_amplification: f64,
    calibrated_only_mean_signature_resident_neuron_fraction: f64,
    calibrated_only_p95_signature_groups: u64,
    mean_active_packet_bytes_per_token: f64,
    p95_active_packet_bytes_per_token: u64,
    p99_active_packet_bytes_per_token: u64,
}

#[derive(Debug, Serialize)]
struct StructuredNllReport {
    schema_version: u32,
    classification: &'static str,
    promotion_evidence: bool,
    training_semantics: &'static str,
    masking_semantics: &'static str,
    quality_semantics: &'static str,
    model: String,
    model_sha256: String,
    llama_revision: String,
    calibrations: Vec<String>,
    calibration_token_positions: usize,
    calibration_expert_observations: usize,
    directly_calibrated_layer_experts: usize,
    fallback_layer_experts: usize,
    prompt_sha256: String,
    prompt_tokens: usize,
    scored_positions: usize,
    group_neurons: usize,
    groups_per_expert: usize,
    relative_l2_training_target: f64,
    exact: StructuredNllMeasurement,
    results: Vec<StructuredNllResult>,
}

#[derive(Clone, Debug, Serialize)]
struct StructuredNllMeasurement {
    mean_nll: f64,
    perplexity: f64,
    top1_tokens: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct StructuredNllResult {
    requested_retained_fraction: f64,
    retained_groups_per_expert: usize,
    actual_retained_fraction: f64,
    mean_nll: f64,
    perplexity: f64,
    nll_delta_vs_exact: f64,
    relative_perplexity_vs_exact: f64,
    top1_agreement_vs_exact: f64,
}

#[derive(Debug, Serialize)]
struct ExpertStudentNllReport {
    schema_version: u32,
    classification: &'static str,
    promotion_evidence: bool,
    quality_semantics: &'static str,
    model_sha256: String,
    corpora: Vec<String>,
    student_down_weights: String,
    layer: usize,
    expert: u16,
    sequences: usize,
    scored_tokens: usize,
    expert_hits: usize,
    exact_mean_nll: f64,
    student_mean_nll: f64,
    nll_delta: f64,
    relative_perplexity_delta: f64,
    top1_agreement: f64,
}

#[derive(Debug, Serialize)]
struct JointExpertStudentNllReport {
    schema_version: u32,
    classification: &'static str,
    promotion_evidence: bool,
    quality_semantics: &'static str,
    model_sha256: String,
    corpora: Vec<String>,
    student: String,
    layer: usize,
    expert: u16,
    teacher_width: usize,
    student_width: usize,
    sequences: usize,
    scored_tokens: usize,
    expert_hits: usize,
    exact_mean_nll: f64,
    student_mean_nll: f64,
    nll_delta: f64,
    relative_perplexity_delta: f64,
    top1_agreement: f64,
}

struct JointExpertStudent {
    layer: usize,
    expert: u16,
    hidden: usize,
    width: usize,
    teacher_width: usize,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

struct LayerResidual {
    layer: usize,
    rank: usize,
    hidden: usize,
    down: Vec<f32>,
    up: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct MultiExpertStudentNllReport {
    schema_version: u32,
    classification: &'static str,
    promotion_evidence: bool,
    quality_semantics: &'static str,
    model_sha256: String,
    corpora: Vec<String>,
    students: Vec<MultiExpertStudentResult>,
    sequences: usize,
    scored_tokens: usize,
    exact_mean_nll: f64,
    student_mean_nll: f64,
    nll_delta: f64,
    relative_perplexity_delta: f64,
    top1_agreement: f64,
}

#[derive(Debug, Serialize)]
struct MultiExpertStudentResult {
    layer: usize,
    expert: u16,
    weights: String,
    exact_route_hits: usize,
    student_route_hits: usize,
}

#[derive(Debug, Serialize)]
struct MultiJointExpertStudentNllReport {
    schema_version: u32,
    classification: &'static str,
    promotion_evidence: bool,
    quality_semantics: &'static str,
    model_sha256: String,
    corpora: Vec<String>,
    students: Vec<MultiJointExpertStudentResult>,
    layer_residual: Option<String>,
    skipped_sequences: usize,
    sequences: usize,
    scored_tokens: usize,
    exact_mean_nll: f64,
    student_mean_nll: f64,
    nll_delta: f64,
    relative_perplexity_delta: f64,
    top1_agreement: f64,
    replacement_semantics: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    prefix_results: Vec<MultiJointExpertStudentPrefixResult>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    replacement_results: Vec<MultiJointExpertStudentReplacementResult>,
}

#[derive(Debug, Serialize)]
struct MultiJointExpertStudentPrefixResult {
    active_students: usize,
    scored_tokens: usize,
    student_mean_nll: f64,
    nll_delta: f64,
    relative_perplexity_delta: f64,
    top1_agreement: f64,
}

#[derive(Debug, Serialize)]
struct MultiJointExpertStudentResult {
    layer: usize,
    expert: u16,
    width: usize,
    sidecar: String,
    exact_route_hits: usize,
    student_route_hits: usize,
}

#[derive(Debug, Serialize)]
struct MultiJointExpertStudentReplacementResult {
    active_replacements: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    cumulative_replacements: Option<usize>,
    layer: usize,
    expert: u16,
    width: usize,
    sidecar: String,
    scored_tokens: usize,
    student_mean_nll: f64,
    nll_delta: f64,
    relative_perplexity_delta: f64,
    top1_agreement: f64,
}

#[derive(Default)]
struct StructuredCalibration {
    token_positions: usize,
    expert_observations: usize,
    counts: HashMap<(usize, u16), Vec<u32>>,
    observations: HashMap<(usize, u16), Vec<Vec<u16>>>,
    layer_counts: HashMap<usize, Vec<u32>>,
    layer_observations: HashMap<usize, Vec<Vec<u16>>>,
}

#[derive(Debug, Serialize)]
struct ExpertDistillReport {
    schema_version: u32,
    classification: &'static str,
    promotion_evidence: bool,
    training_semantics: &'static str,
    inference_semantics: &'static str,
    quality_semantics: &'static str,
    split_semantics: &'static str,
    traces: Vec<String>,
    heldout_traces: Vec<String>,
    teacher_down_weights: String,
    student_down_weights: Option<String>,
    calibrations: Vec<String>,
    trace_header: TraceHeader,
    layer: usize,
    expert: u16,
    group_neurons: usize,
    retained_groups: usize,
    retained_neurons: usize,
    retained_fraction: f64,
    available_train_samples: usize,
    train_samples: usize,
    heldout_samples: usize,
    ridge_relative_to_mean_kernel_diagonal: f64,
    ridge_absolute: f64,
    student_down_parameters: usize,
    student_down_f16_bytes: usize,
    student_down_q4_payload_bytes: usize,
    dual_support_f32_bytes_avoided: usize,
    primal_dual_max_relative_l2: f64,
    heldout_mean_relative_l2: f64,
    heldout_p50_relative_l2: f64,
    heldout_p95_relative_l2: f64,
    heldout_mean_cosine: f64,
    masked_teacher_mean_relative_l2: f64,
    masked_teacher_p95_relative_l2: f64,
    masked_teacher_mean_cosine: f64,
    heldout_zero_baseline_relative_l2: f64,
}

fn default_greedy_selection_semantics() -> String {
    "perfect-knowledge-greedy-minimum-next-residual".to_owned()
}

fn contribution_selection_semantics(packets: usize) -> &'static str {
    if packets <= MAX_GREEDY_CONTRIBUTION_PACKETS {
        "perfect-knowledge-greedy-minimum-next-residual"
    } else {
        "perfect-knowledge-descending-packet-output-norm"
    }
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
            sample_layer,
            sample_expert,
            sample_weights_output,
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
            sample: sample_layer.zip(sample_expert),
            sample_weights_output,
            model_sha256,
        }),
        Command::CaptureCorpus {
            model,
            corpora,
            output,
            sample_weights_output,
            sample_layer,
            sample_experts,
            max_samples_per_expert,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
        } => capture_corpus(CaptureCorpusArgs {
            model,
            corpora,
            output,
            sample_weights_output,
            sample_layer,
            sample_experts,
            max_samples_per_expert,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
        }),
        Command::Simulate {
            trace,
            calibration,
            policies,
            hot_fractions,
            budgets,
            expert_bytes,
            expert_neurons,
            packet_neurons,
            layouts,
            io_bandwidth_mib_s,
            io_fixed_us,
            io_parallelism,
            reset_between_requests,
            measure_phases,
            output,
        } => simulate(SimulateArgs {
            trace,
            calibration,
            policies,
            hot_fractions,
            budgets,
            expert_bytes,
            expert_neurons,
            packet_neurons,
            layouts,
            io_bandwidth_mib_s,
            io_fixed_us,
            io_parallelism,
            reset_between_requests,
            measure_phases,
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
        Command::GroupOracle {
            calibration,
            trace,
            group_neurons,
            relative_errors,
            expert_bytes,
            output,
        } => group_oracle(
            &calibration,
            &trace,
            group_neurons,
            &relative_errors,
            expert_bytes,
            &output,
        ),
        Command::StructuredNll {
            model,
            calibrations,
            prompt,
            group_neurons,
            relative_error,
            retained_fractions,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        } => structured_nll(StructuredNllArgs {
            model,
            calibrations,
            prompt,
            group_neurons,
            relative_error,
            retained_fractions,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        }),
        Command::DistillExpert {
            traces,
            heldout_traces,
            teacher_down_weights,
            calibrations,
            layer,
            expert,
            group_neurons,
            retained_fraction,
            relative_error,
            train_fraction,
            max_train_samples,
            student_down_weights_output,
            ridge,
            output,
        } => distill_expert(
            &traces,
            &heldout_traces,
            &teacher_down_weights,
            &calibrations,
            layer,
            expert,
            group_neurons,
            retained_fraction,
            relative_error,
            train_fraction,
            max_train_samples,
            student_down_weights_output.as_deref(),
            ridge,
            &output,
        ),
        Command::ExpertStudentNll {
            model,
            corpora,
            student_down_weights,
            layer,
            expert,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        } => expert_student_nll(ExpertStudentNllArgs {
            model,
            corpora,
            student_down_weights,
            layer,
            expert,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        }),
        Command::JointExpertStudentNll {
            model,
            corpora,
            student,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        } => joint_expert_student_nll(JointExpertStudentNllArgs {
            model,
            corpora,
            student,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        }),
        Command::MultiJointExpertStudentNll {
            model,
            corpora,
            students,
            replacement_students,
            independent_replacements,
            layer_residual,
            active_prefixes,
            skip_sequences,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        } => multi_joint_expert_student_nll(MultiJointExpertStudentNllArgs {
            model,
            corpora,
            students,
            replacement_students,
            independent_replacements,
            layer_residual,
            active_prefixes,
            skip_sequences,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        }),
        Command::MultiExpertStudentNll {
            model,
            corpora,
            students,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        } => multi_expert_student_nll(MultiExpertStudentNllArgs {
            model,
            corpora,
            students,
            max_sequences,
            max_tokens_per_sequence,
            context,
            layers,
            experts,
            top_k,
            gpu_layers,
            threads,
            expert_neurons,
            hidden_size,
            model_sha256,
            output,
        }),
    }
}

struct StructuredNllArgs {
    model: PathBuf,
    calibrations: Vec<PathBuf>,
    prompt: String,
    group_neurons: usize,
    relative_error: f64,
    retained_fractions: Vec<f64>,
    context: usize,
    layers: usize,
    experts: usize,
    top_k: usize,
    gpu_layers: i32,
    threads: i32,
    expert_neurons: usize,
    hidden_size: usize,
    model_sha256: Option<String>,
    output: PathBuf,
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
    sample: Option<(usize, u16)>,
    sample_weights_output: Option<PathBuf>,
    model_sha256: Option<String>,
}

struct CaptureCorpusArgs {
    model: PathBuf,
    corpora: Vec<PathBuf>,
    output: PathBuf,
    sample_weights_output: Option<PathBuf>,
    sample_layer: Option<usize>,
    sample_experts: Vec<u16>,
    max_samples_per_expert: usize,
    max_sequences: usize,
    max_tokens_per_sequence: usize,
    context: usize,
    layers: usize,
    experts: usize,
    top_k: usize,
    gpu_layers: i32,
    threads: i32,
    expert_neurons: usize,
    hidden_size: usize,
    model_sha256: Option<String>,
}

fn corpus_sample_modes(
    layer: Option<usize>,
    experts: &[u16],
) -> (Option<(usize, u16)>, Option<usize>) {
    match (layer, experts) {
        (Some(layer), [expert]) => (Some((layer, *expert)), None),
        (Some(layer), _) => (None, Some(layer)),
        (None, _) => (None, None),
    }
}

struct ExpertStudentNllArgs {
    model: PathBuf,
    corpora: Vec<PathBuf>,
    student_down_weights: PathBuf,
    layer: usize,
    expert: u16,
    max_sequences: usize,
    max_tokens_per_sequence: usize,
    context: usize,
    layers: usize,
    experts: usize,
    top_k: usize,
    gpu_layers: i32,
    threads: i32,
    expert_neurons: usize,
    hidden_size: usize,
    model_sha256: Option<String>,
    output: PathBuf,
}

struct JointExpertStudentNllArgs {
    model: PathBuf,
    corpora: Vec<PathBuf>,
    student: PathBuf,
    max_sequences: usize,
    max_tokens_per_sequence: usize,
    context: usize,
    layers: usize,
    experts: usize,
    top_k: usize,
    gpu_layers: i32,
    threads: i32,
    expert_neurons: usize,
    hidden_size: usize,
    model_sha256: Option<String>,
    output: PathBuf,
}

struct MultiExpertStudentNllArgs {
    model: PathBuf,
    corpora: Vec<PathBuf>,
    students: Vec<String>,
    max_sequences: usize,
    max_tokens_per_sequence: usize,
    context: usize,
    layers: usize,
    experts: usize,
    top_k: usize,
    gpu_layers: i32,
    threads: i32,
    expert_neurons: usize,
    hidden_size: usize,
    model_sha256: Option<String>,
    output: PathBuf,
}

struct MultiJointExpertStudentNllArgs {
    model: PathBuf,
    corpora: Vec<PathBuf>,
    students: Vec<PathBuf>,
    replacement_students: Vec<PathBuf>,
    independent_replacements: bool,
    layer_residual: Option<PathBuf>,
    active_prefixes: Vec<usize>,
    skip_sequences: usize,
    max_sequences: usize,
    max_tokens_per_sequence: usize,
    context: usize,
    layers: usize,
    experts: usize,
    top_k: usize,
    gpu_layers: i32,
    threads: i32,
    expert_neurons: usize,
    hidden_size: usize,
    model_sha256: Option<String>,
    output: PathBuf,
}

struct ExpertStudentSpec {
    layer: usize,
    expert: u16,
    path: PathBuf,
    weights: Vec<f32>,
}

struct SimulateArgs {
    trace: PathBuf,
    calibration: Option<PathBuf>,
    policies: Vec<CachePolicy>,
    hot_fractions: Vec<f64>,
    budgets: Vec<u64>,
    expert_bytes: u64,
    expert_neurons: usize,
    packet_neurons: Vec<usize>,
    layouts: Vec<PhysicalLayout>,
    io_bandwidth_mib_s: Option<f64>,
    io_fixed_us: f64,
    io_parallelism: usize,
    reset_between_requests: bool,
    measure_phases: Vec<String>,
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
    if let Some((layer, expert)) = args.sample
        && (layer >= args.layers || usize::from(expert) >= args.experts)
    {
        bail!("expert distillation sample target is outside the model shape");
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
            experts: args.experts,
            top_k: args.top_k,
            gpu_layers: args.gpu_layers,
            threads: args.threads,
            intermediate: (!args.activation_packet_neurons.is_empty()
                || !args.contribution_packet_neurons.is_empty())
            .then_some(args.expert_neurons)
            .or(args.sample.map(|_| args.expert_neurons)),
            hidden: args.hidden_size,
            sample: args.sample,
            sample_all_layer: None,
            capture_all_routes: true,
            capture_all_activations: !args.activation_packet_neurons.is_empty()
                || !args.contribution_packet_neurons.is_empty(),
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
            target_nll: _,
            routes,
            activations,
            down_contribution,
            expert_sample,
            expert_samples: _,
            expert_down_weights,
        } = llama.step(token)?;
        if let Some(weights) = expert_down_weights
            && let Some(path) = &args.sample_weights_output
        {
            write_expert_down_weights(path, args.expert_neurons, args.hidden_size, &weights)?;
        }
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
                expert_distill_sample: expert_sample.map(
                    |(input, activation, teacher_down_output)| {
                        let (layer, expert) =
                            args.sample.expect("sample exists when bridge returns one");
                        ExpertDistillSample {
                            layer,
                            expert,
                            input,
                            activation,
                            teacher_down_output,
                        }
                    },
                ),
                expert_distill_samples: Vec::new(),
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
            target_nll: _,
            routes,
            activations,
            down_contribution,
            expert_sample,
            expert_samples: _,
            expert_down_weights,
        } = llama.step(next)?;
        if let Some(weights) = expert_down_weights
            && let Some(path) = &args.sample_weights_output
        {
            write_expert_down_weights(path, args.expert_neurons, args.hidden_size, &weights)?;
        }
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
                expert_distill_sample: expert_sample.map(
                    |(input, activation, teacher_down_output)| {
                        let (layer, expert) =
                            args.sample.expect("sample exists when bridge returns one");
                        ExpertDistillSample {
                            layer,
                            expert,
                            input,
                            activation,
                            teacher_down_output,
                        }
                    },
                ),
                expert_distill_samples: Vec::new(),
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

#[cfg(target_os = "macos")]
fn capture_corpus(args: CaptureCorpusArgs) -> Result<()> {
    if !args.model.is_file() {
        bail!("model does not exist: {}", args.model.display());
    }
    if args.corpora.is_empty() || args.corpora.iter().any(|path| !path.is_file()) {
        bail!("every corpus input must be an existing file");
    }
    if args.max_sequences == 0
        || args.max_tokens_per_sequence < 2
        || args.max_tokens_per_sequence > args.context
    {
        bail!("corpus capture requires positive sequences and 2..=context tokens per sequence");
    }
    if args.sample_layer.is_some_and(|layer| layer >= args.layers)
        || args
            .sample_experts
            .iter()
            .any(|&expert| usize::from(expert) >= args.experts)
    {
        bail!("expert sample target is outside the model shape");
    }
    let selected_experts: HashSet<u16> = args.sample_experts.iter().copied().collect();
    if selected_experts.len() != args.sample_experts.len() {
        bail!("sample expert selection contains duplicates");
    }
    if args.sample_weights_output.is_some()
        && (args.sample_layer.is_none() || args.sample_experts.len() != 1)
    {
        bail!("sample weights output requires exactly one selected expert");
    }
    let model_hash = match args.model_sha256.as_ref() {
        Some(value) => {
            validate_sha256(value)?;
            value.clone()
        }
        None => sha256_file(&args.model)?,
    };
    let routes_only = args.sample_layer.is_none();
    let (sample, sample_all_layer) = corpus_sample_modes(args.sample_layer, &args.sample_experts);
    let mut llama = llama_capture::Capture::open(
        &args.model,
        llama_capture::OpenConfig {
            context: args.context,
            layers: args.layers,
            experts: args.experts,
            top_k: args.top_k,
            gpu_layers: args.gpu_layers,
            threads: args.threads,
            intermediate: (!routes_only).then_some(args.expert_neurons),
            hidden: args.hidden_size,
            sample,
            sample_all_layer,
            capture_all_routes: routes_only,
            capture_all_activations: false,
            contribution_packet_neurons: &[],
        },
    )?;
    let mut paragraphs = Vec::new();
    for path in &args.corpora {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("read corpus {}", path.display()))?;
        paragraphs.extend(split_corpus_paragraphs(&contents));
    }
    if paragraphs.is_empty() {
        bail!("corpus files contain no non-empty paragraphs");
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
            source: if routes_only {
                "llama.cpp-eval-callback-route-only-corpus-reset".to_owned()
            } else {
                "llama.cpp-eval-callback-selected-expert-batch-corpus-reset".to_owned()
            },
        }),
    )?;

    let mut sequences = 0usize;
    let mut token_positions = 0usize;
    let mut expert_hits = 0usize;
    let mut observed_hit_counts = vec![0usize; args.experts];
    let mut retained_hit_counts = vec![0usize; args.experts];
    let mut wrote_weights = args.sample_weights_output.is_none();
    for paragraph in paragraphs {
        if sequences >= args.max_sequences {
            break;
        }
        let mut tokens = llama.tokenize(&paragraph)?;
        tokens.truncate(args.max_tokens_per_sequence);
        if tokens.len() < 2 {
            continue;
        }
        if sequences > 0 {
            llama.reset()?;
        }
        let request_id = format!("corpus-{sequences:04}");
        for (position, token) in tokens.into_iter().enumerate() {
            let llama_capture::StepOutput {
                predicted,
                target_nll: _,
                routes,
                activations: _,
                down_contribution: _,
                expert_sample,
                expert_samples,
                expert_down_weights,
            } = llama.step(token)?;
            if let Some(sample_layer) = args.sample_layer {
                validate_sample_route(
                    &routes,
                    args.layers,
                    args.top_k,
                    args.experts,
                    sample_layer,
                )?;
            } else {
                validate_routes(&routes, args.layers, args.top_k, args.experts)?;
            }
            if let Some(weights) = expert_down_weights
                && let Some(path) = &args.sample_weights_output
            {
                write_expert_down_weights(path, args.expert_neurons, args.hidden_size, &weights)?;
                wrote_weights = true;
            }
            let sample_layer = args.sample_layer.unwrap_or(0);
            let mut samples = expert_samples
                .into_iter()
                .map(
                    |(expert, input, activation, teacher_down_output)| ExpertDistillSample {
                        layer: sample_layer,
                        expert,
                        input,
                        activation,
                        teacher_down_output,
                    },
                )
                .collect::<Vec<_>>();
            if let Some((input, activation, teacher_down_output)) = expert_sample {
                samples.push(ExpertDistillSample {
                    layer: sample_layer,
                    expert: args.sample_experts[0],
                    input,
                    activation,
                    teacher_down_output,
                });
            }
            samples.retain(|sample| {
                selected_experts.is_empty() || selected_experts.contains(&sample.expert)
            });
            for sample in &samples {
                observed_hit_counts[usize::from(sample.expert)] += 1;
            }
            samples.retain(|sample| {
                let expert = usize::from(sample.expert);
                if args.max_samples_per_expert > 0
                    && retained_hit_counts[expert] >= args.max_samples_per_expert
                {
                    false
                } else {
                    retained_hit_counts[expert] += 1;
                    true
                }
            });
            expert_hits += samples.len();
            write_trace_line(
                &mut writer,
                &TraceLine::Token(TokenRoutes {
                    request_id: request_id.clone(),
                    token_position: position,
                    phase: "corpus".to_owned(),
                    input_token: token,
                    predicted_token: predicted,
                    routes,
                    packet_activation_energy: None,
                    packet_down_contribution: None,
                    expert_distill_sample: None,
                    expert_distill_samples: samples,
                }),
            )?;
            token_positions += 1;
        }
        sequences += 1;
    }
    writer.flush()?;
    if sequences == 0 || (!routes_only && expert_hits == 0) || !wrote_weights {
        bail!(
            "corpus capture produced sequences {sequences}, expert hits {expert_hits}, weights {wrote_weights}"
        );
    }
    println!("corpus trace: {}", args.output.display());
    println!(
        "sequences: {sequences}, routed token positions: {token_positions}, expert hits: {expert_hits}"
    );
    let populated = retained_hit_counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(expert, count)| format!("{expert}:{count}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("retained expert hit counts: {populated}");
    if args.max_samples_per_expert > 0 {
        let observed = observed_hit_counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(expert, count)| format!("{expert}:{count}"))
            .collect::<Vec<_>>()
            .join(",");
        println!("observed expert hit counts: {observed}");
    }
    Ok(())
}

fn split_corpus_paragraphs(contents: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut paragraph = String::new();
    for line in contents.lines() {
        if line.trim().is_empty() {
            if !paragraph.is_empty() {
                paragraphs.push(std::mem::take(&mut paragraph));
            }
        } else {
            if !paragraph.is_empty() {
                paragraph.push('\n');
            }
            paragraph.push_str(line.trim());
        }
    }
    if !paragraph.is_empty() {
        paragraphs.push(paragraph);
    }
    paragraphs
}

#[derive(Default)]
struct CorpusNllMeasurement {
    nll: f64,
    scored_tokens: usize,
    expert_hits: usize,
    route_hits: Vec<usize>,
    top1_tokens: Vec<i32>,
}

impl CorpusNllMeasurement {
    fn mean_nll(&self) -> f64 {
        self.nll / self.scored_tokens.max(1) as f64
    }
}

#[cfg(target_os = "macos")]
fn expert_student_nll(args: ExpertStudentNllArgs) -> Result<()> {
    if !args.model.is_file()
        || args.corpora.is_empty()
        || args.corpora.iter().any(|path| !path.is_file())
        || !args.student_down_weights.is_file()
    {
        bail!("model, corpus, and student weight inputs must exist");
    }
    if args.max_sequences == 0
        || args.max_tokens_per_sequence < 2
        || args.max_tokens_per_sequence > args.context
        || args.layer >= args.layers
        || usize::from(args.expert) >= args.experts
    {
        bail!("expert student corpus or model shape arguments are invalid");
    }
    let model_hash = match args.model_sha256.as_ref() {
        Some(value) => {
            validate_sha256(value)?;
            value.clone()
        }
        None => sha256_file(&args.model)?,
    };
    let (student_neurons, student_hidden, student_weights) =
        read_expert_down_weights(&args.student_down_weights)?;
    if (student_neurons, student_hidden) != (args.expert_neurons, args.hidden_size) {
        bail!("student down weight dimensions do not match the requested model shape");
    }
    let mut capture = llama_capture::Capture::open(
        &args.model,
        llama_capture::OpenConfig {
            context: args.context,
            layers: args.layers,
            experts: args.experts,
            top_k: args.top_k,
            gpu_layers: args.gpu_layers,
            threads: args.threads,
            intermediate: Some(args.expert_neurons),
            hidden: args.hidden_size,
            sample: Some((args.layer, args.expert)),
            sample_all_layer: None,
            capture_all_routes: false,
            capture_all_activations: false,
            contribution_packet_neurons: &[],
        },
    )?;
    let mut sequences = Vec::new();
    for path in &args.corpora {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("read corpus {}", path.display()))?;
        for paragraph in split_corpus_paragraphs(&contents) {
            if sequences.len() >= args.max_sequences {
                break;
            }
            let mut tokens = capture.tokenize(&paragraph)?;
            tokens.truncate(args.max_tokens_per_sequence);
            if tokens.len() >= 2 {
                sequences.push(tokens);
            }
        }
        if sequences.len() >= args.max_sequences {
            break;
        }
    }
    if sequences.is_empty() {
        bail!("corpus produced no scoreable token sequences");
    }
    let exact = measure_corpus_nll(&mut capture, &sequences)?;
    capture.set_expert_student(&student_weights)?;
    let student = measure_corpus_nll(&mut capture, &sequences)?;
    if exact.scored_tokens != student.scored_tokens
        || exact.top1_tokens.len() != student.top1_tokens.len()
    {
        bail!("exact and student corpus passes produced different token counts");
    }
    let exact_mean_nll = exact.mean_nll();
    let student_mean_nll = student.mean_nll();
    let top1_matches = exact
        .top1_tokens
        .iter()
        .zip(&student.top1_tokens)
        .filter(|(exact, student)| exact == student)
        .count();
    let report = ExpertStudentNllReport {
        schema_version: 1,
        classification: "single-expert-folded-student-corpus-nll-control",
        promotion_evidence: false,
        quality_semantics: "teacher-forced-corpus-nll-with-one-expert-down-projection-replaced",
        model_sha256: model_hash,
        corpora: args
            .corpora
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        student_down_weights: args.student_down_weights.display().to_string(),
        layer: args.layer,
        expert: args.expert,
        sequences: sequences.len(),
        scored_tokens: exact.scored_tokens,
        expert_hits: exact.expert_hits,
        exact_mean_nll,
        student_mean_nll,
        nll_delta: student_mean_nll - exact_mean_nll,
        relative_perplexity_delta: (student_mean_nll - exact_mean_nll).exp() - 1.0,
        top1_agreement: top1_matches as f64 / exact.top1_tokens.len().max(1) as f64,
    };
    let mut writer = BufWriter::new(
        File::create(&args.output).with_context(|| format!("create {}", args.output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("expert student NLL report: {}", args.output.display());
    println!(
        "layer {} expert {}: sequences {}, scored {}, hits {}, NLL {:.6} -> {:.6} ({:+.4}% PPL), top1 {:.2}%",
        args.layer,
        args.expert,
        report.sequences,
        report.scored_tokens,
        report.expert_hits,
        exact_mean_nll,
        student_mean_nll,
        report.relative_perplexity_delta * 100.0,
        report.top1_agreement * 100.0,
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn joint_expert_student_nll(args: JointExpertStudentNllArgs) -> Result<()> {
    if !args.model.is_file()
        || args.corpora.is_empty()
        || args.corpora.iter().any(|path| !path.is_file())
        || !args.student.is_file()
    {
        bail!("model, corpus, and joint student inputs must exist");
    }
    if args.max_sequences == 0
        || args.max_tokens_per_sequence < 2
        || args.max_tokens_per_sequence > args.context
    {
        bail!("joint student corpus arguments are invalid");
    }
    let joint = read_joint_expert_student(&args.student)?;
    if joint.layer >= args.layers
        || usize::from(joint.expert) >= args.experts
        || joint.hidden != args.hidden_size
        || joint.teacher_width != args.expert_neurons
    {
        bail!("joint student dimensions or target do not match the requested model");
    }
    let model_hash = match args.model_sha256.as_ref() {
        Some(value) => {
            validate_sha256(value)?;
            value.clone()
        }
        None => sha256_file(&args.model)?,
    };
    let mut capture = llama_capture::Capture::open(
        &args.model,
        llama_capture::OpenConfig {
            context: args.context,
            layers: args.layers,
            experts: args.experts,
            top_k: args.top_k,
            gpu_layers: args.gpu_layers,
            threads: args.threads,
            intermediate: Some(args.expert_neurons),
            hidden: args.hidden_size,
            sample: Some((joint.layer, joint.expert)),
            sample_all_layer: None,
            capture_all_routes: false,
            capture_all_activations: false,
            contribution_packet_neurons: &[],
        },
    )?;
    let mut sequences = Vec::new();
    for path in &args.corpora {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("read corpus {}", path.display()))?;
        for paragraph in split_corpus_paragraphs(&contents) {
            if sequences.len() >= args.max_sequences {
                break;
            }
            let mut tokens = capture.tokenize(&paragraph)?;
            tokens.truncate(args.max_tokens_per_sequence);
            if tokens.len() >= 2 {
                sequences.push(tokens);
            }
        }
        if sequences.len() >= args.max_sequences {
            break;
        }
    }
    if sequences.is_empty() {
        bail!("corpus produced no scoreable token sequences");
    }
    let exact = measure_corpus_nll(&mut capture, &sequences)?;
    capture.set_joint_expert_student(&joint)?;
    let student = measure_corpus_nll(&mut capture, &sequences)?;
    if exact.scored_tokens != student.scored_tokens
        || exact.top1_tokens.len() != student.top1_tokens.len()
    {
        bail!("exact and joint-student corpus passes produced different token counts");
    }
    let exact_mean_nll = exact.mean_nll();
    let student_mean_nll = student.mean_nll();
    let top1_matches = exact
        .top1_tokens
        .iter()
        .zip(&student.top1_tokens)
        .filter(|(exact, student)| exact == student)
        .count();
    let report = JointExpertStudentNllReport {
        schema_version: 1,
        classification: "single-joint-expert-student-corpus-nll-control",
        promotion_evidence: false,
        quality_semantics: "teacher-forced-corpus-nll-with-one-gate-up-down-expert-replaced",
        model_sha256: model_hash,
        corpora: args
            .corpora
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        student: args.student.display().to_string(),
        layer: joint.layer,
        expert: joint.expert,
        teacher_width: joint.teacher_width,
        student_width: joint.width,
        sequences: sequences.len(),
        scored_tokens: exact.scored_tokens,
        expert_hits: exact.expert_hits,
        exact_mean_nll,
        student_mean_nll,
        nll_delta: student_mean_nll - exact_mean_nll,
        relative_perplexity_delta: (student_mean_nll - exact_mean_nll).exp() - 1.0,
        top1_agreement: top1_matches as f64 / exact.top1_tokens.len().max(1) as f64,
    };
    let mut writer = BufWriter::new(
        File::create(&args.output).with_context(|| format!("create {}", args.output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("joint expert student NLL report: {}", args.output.display());
    println!(
        "layer {} expert {} width {}: sequences {}, scored {}, hits {}, NLL {:.6} -> {:.6} ({:+.4}% PPL), top1 {:.2}%",
        joint.layer,
        joint.expert,
        joint.width,
        report.sequences,
        report.scored_tokens,
        report.expert_hits,
        exact_mean_nll,
        student_mean_nll,
        report.relative_perplexity_delta * 100.0,
        report.top1_agreement * 100.0,
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn measure_corpus_nll(
    capture: &mut llama_capture::Capture,
    sequences: &[Vec<i32>],
) -> Result<CorpusNllMeasurement> {
    let mut measurement = CorpusNllMeasurement::default();
    for tokens in sequences {
        capture.reset()?;
        for pair in tokens.windows(2) {
            let output = capture.step_target(pair[0], pair[1])?;
            measurement.nll += output
                .target_nll
                .ok_or_else(|| anyhow::anyhow!("llama bridge omitted target NLL"))?
                as f64;
            measurement.scored_tokens += 1;
            measurement.expert_hits += usize::from(output.expert_sample.is_some());
            measurement.top1_tokens.push(output.predicted);
        }
    }
    Ok(measurement)
}

#[cfg(target_os = "macos")]
fn multi_joint_expert_student_nll(args: MultiJointExpertStudentNllArgs) -> Result<()> {
    if !args.model.is_file()
        || args.corpora.is_empty()
        || args.corpora.iter().any(|path| !path.is_file())
        || args.students.is_empty()
        || args.students.iter().any(|path| !path.is_file())
        || args.replacement_students.iter().any(|path| !path.is_file())
        || args
            .layer_residual
            .as_ref()
            .is_some_and(|path| !path.is_file())
    {
        bail!("model, corpus, and joint student inputs must exist");
    }
    if args.max_sequences == 0
        || args.max_tokens_per_sequence < 2
        || args.max_tokens_per_sequence > args.context
    {
        bail!("multi-joint-student corpus arguments are invalid");
    }
    let model_hash = match args.model_sha256.as_ref() {
        Some(value) => {
            validate_sha256(value)?;
            value.clone()
        }
        None => sha256_file(&args.model)?,
    };
    let mut targets = HashSet::new();
    let mut students = Vec::new();
    for path in &args.students {
        let student = read_joint_expert_student(path)?;
        if student.layer >= args.layers
            || usize::from(student.expert) >= args.experts
            || student.hidden != args.hidden_size
            || student.teacher_width != args.expert_neurons
        {
            bail!(
                "joint student {} does not match the requested model",
                path.display()
            );
        }
        if !targets.insert((student.layer, student.expert)) {
            bail!(
                "duplicate joint student target {}:{}",
                student.layer,
                student.expert
            );
        }
        students.push((path.clone(), student.layer, student.expert, student.width));
    }
    let mut replacements = Vec::new();
    let mut replaced_targets = HashSet::new();
    for path in &args.replacement_students {
        let student = read_joint_expert_student(path)?;
        let target = (student.layer, student.expert);
        if student.layer >= args.layers
            || usize::from(student.expert) >= args.experts
            || student.hidden != args.hidden_size
            || student.teacher_width != args.expert_neurons
            || !targets.contains(&target)
        {
            bail!(
                "replacement student {} does not match one baseline model target",
                path.display()
            );
        }
        if !replaced_targets.insert(target) {
            bail!(
                "replacement target {}:{} occurs more than once",
                student.layer,
                student.expert
            );
        }
        replacements.push((path.clone(), student.layer, student.expert, student.width));
    }
    if !replacements.is_empty() && !args.active_prefixes.is_empty() {
        bail!("replacement students require the complete baseline, without --active-prefixes");
    }
    if args.independent_replacements && replacements.is_empty() {
        bail!("--independent-replacements requires at least one --replacement-student");
    }
    let active_prefixes = if args.active_prefixes.is_empty() {
        vec![students.len()]
    } else {
        let mut previous = 0usize;
        for &prefix in &args.active_prefixes {
            if prefix == 0 || prefix > students.len() || prefix <= previous {
                bail!(
                    "active prefixes must be strictly increasing within 1..={}",
                    students.len()
                );
            }
            previous = prefix;
        }
        args.active_prefixes.clone()
    };
    let mut capture = llama_capture::Capture::open(
        &args.model,
        llama_capture::OpenConfig {
            context: args.context,
            layers: args.layers,
            experts: args.experts,
            top_k: args.top_k,
            gpu_layers: args.gpu_layers,
            threads: args.threads,
            intermediate: Some(args.expert_neurons),
            hidden: args.hidden_size,
            sample: None,
            sample_all_layer: None,
            capture_all_routes: true,
            capture_all_activations: true,
            contribution_packet_neurons: &[],
        },
    )?;
    let mut sequences = Vec::new();
    let mut scoreable_seen = 0usize;
    for path in &args.corpora {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("read corpus {}", path.display()))?;
        for paragraph in split_corpus_paragraphs(&contents) {
            if sequences.len() >= args.max_sequences {
                break;
            }
            let mut tokens = capture.tokenize(&paragraph)?;
            tokens.truncate(args.max_tokens_per_sequence);
            if tokens.len() >= 2 {
                if scoreable_seen < args.skip_sequences {
                    scoreable_seen += 1;
                    continue;
                }
                sequences.push(tokens);
            }
        }
        if sequences.len() >= args.max_sequences {
            break;
        }
    }
    if sequences.is_empty() {
        bail!("corpus produced no scoreable token sequences");
    }
    let target_pairs = students
        .iter()
        .map(|(_, layer, expert, _)| (*layer, *expert))
        .collect::<Vec<_>>();
    let exact = measure_multi_corpus_nll(&mut capture, &sequences, &target_pairs)?;
    for (path, layer, expert, width) in &students {
        let student = read_joint_expert_student(path)?;
        if student.layer != *layer || student.expert != *expert || student.width != *width {
            bail!("joint student changed while evaluating: {}", path.display());
        }
        capture.add_joint_expert_student(&student)?;
    }
    if let Some(path) = &args.layer_residual {
        let residual = read_layer_residual(path)?;
        if residual.layer >= args.layers || residual.hidden != args.hidden_size {
            bail!(
                "layer residual {} does not match the requested model",
                path.display()
            );
        }
        capture.set_layer_residual(&residual)?;
    }
    let mut prefix_measurements = Vec::new();
    for &prefix in &active_prefixes {
        capture.set_joint_active_prefix(prefix)?;
        let measurement = measure_multi_corpus_nll(&mut capture, &sequences, &target_pairs)?;
        if exact.scored_tokens != measurement.scored_tokens
            || exact.top1_tokens.len() != measurement.top1_tokens.len()
        {
            bail!("exact and multi-joint-student passes produced incompatible measurements");
        }
        prefix_measurements.push((prefix, measurement));
    }
    let exact_mean_nll = exact.mean_nll();
    let mut replacement_measurements = Vec::new();
    for (path, layer, expert, width) in &replacements {
        let replacement = read_joint_expert_student(path)?;
        if replacement.layer != *layer
            || replacement.expert != *expert
            || replacement.width != *width
        {
            bail!(
                "replacement student changed while evaluating: {}",
                path.display()
            );
        }
        capture.replace_joint_expert_student(&replacement)?;
        let measurement = measure_multi_corpus_nll(&mut capture, &sequences, &target_pairs)?;
        if exact.scored_tokens != measurement.scored_tokens
            || exact.top1_tokens.len() != measurement.top1_tokens.len()
        {
            bail!("exact and replacement-student passes produced incompatible measurements");
        }
        replacement_measurements.push((path, layer, expert, width, measurement));
        if args.independent_replacements {
            let baseline_path = students
                .iter()
                .find(|(_, baseline_layer, baseline_expert, _)| {
                    baseline_layer == layer && baseline_expert == expert
                })
                .map(|(baseline_path, _, _, _)| baseline_path)
                .expect("replacement targets were validated against baseline students");
            let baseline = read_joint_expert_student(baseline_path)?;
            capture.replace_joint_expert_student(&baseline)?;
        }
    }
    let student_measurement = if args.independent_replacements {
        &prefix_measurements
            .last()
            .expect("at least one active prefix is always evaluated")
            .1
    } else {
        replacement_measurements
            .last()
            .map(|(_, _, _, _, measurement)| measurement)
            .unwrap_or_else(|| {
                &prefix_measurements
                    .last()
                    .expect("at least one active prefix is always evaluated")
                    .1
            })
    };
    let student_mean_nll = student_measurement.mean_nll();
    let top1_matches = exact
        .top1_tokens
        .iter()
        .zip(&student_measurement.top1_tokens)
        .filter(|(exact, student)| exact == student)
        .count();
    let report = MultiJointExpertStudentNllReport {
        schema_version: 1,
        classification: "simultaneous-joint-expert-student-corpus-nll-control",
        promotion_evidence: false,
        quality_semantics: "teacher-forced-corpus-nll-with-multiple-gate-up-down-experts-replaced",
        model_sha256: model_hash,
        corpora: args
            .corpora
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        students: students
            .iter()
            .zip(exact.route_hits.iter().zip(&student_measurement.route_hits))
            .map(
                |((path, layer, expert, width), (&exact_route_hits, &student_route_hits))| {
                    MultiJointExpertStudentResult {
                        layer: *layer,
                        expert: *expert,
                        width: *width,
                        sidecar: path.display().to_string(),
                        exact_route_hits,
                        student_route_hits,
                    }
                },
            )
            .collect(),
        layer_residual: args
            .layer_residual
            .as_ref()
            .map(|path| path.display().to_string()),
        skipped_sequences: args.skip_sequences,
        sequences: sequences.len(),
        scored_tokens: exact.scored_tokens,
        exact_mean_nll,
        student_mean_nll,
        nll_delta: student_mean_nll - exact_mean_nll,
        relative_perplexity_delta: (student_mean_nll - exact_mean_nll).exp() - 1.0,
        top1_agreement: top1_matches as f64 / exact.top1_tokens.len().max(1) as f64,
        replacement_semantics: if replacements.is_empty() {
            "none"
        } else if args.independent_replacements {
            "independent-against-complete-baseline"
        } else {
            "cumulative-in-command-line-order"
        },
        prefix_results: prefix_measurements
            .iter()
            .map(|(active_students, measurement)| {
                let mean_nll = measurement.mean_nll();
                let matches = exact
                    .top1_tokens
                    .iter()
                    .zip(&measurement.top1_tokens)
                    .filter(|(exact, student)| exact == student)
                    .count();
                MultiJointExpertStudentPrefixResult {
                    active_students: *active_students,
                    scored_tokens: measurement.scored_tokens,
                    student_mean_nll: mean_nll,
                    nll_delta: mean_nll - exact_mean_nll,
                    relative_perplexity_delta: (mean_nll - exact_mean_nll).exp() - 1.0,
                    top1_agreement: matches as f64 / exact.top1_tokens.len().max(1) as f64,
                }
            })
            .collect(),
        replacement_results: replacement_measurements
            .iter()
            .enumerate()
            .map(|(index, (path, layer, expert, width, measurement))| {
                let mean_nll = measurement.mean_nll();
                let matches = exact
                    .top1_tokens
                    .iter()
                    .zip(&measurement.top1_tokens)
                    .filter(|(exact, student)| exact == student)
                    .count();
                MultiJointExpertStudentReplacementResult {
                    active_replacements: if args.independent_replacements {
                        1
                    } else {
                        index + 1
                    },
                    cumulative_replacements: (!args.independent_replacements).then_some(index + 1),
                    layer: **layer,
                    expert: **expert,
                    width: **width,
                    sidecar: path.display().to_string(),
                    scored_tokens: measurement.scored_tokens,
                    student_mean_nll: mean_nll,
                    nll_delta: mean_nll - exact_mean_nll,
                    relative_perplexity_delta: (mean_nll - exact_mean_nll).exp() - 1.0,
                    top1_agreement: matches as f64 / exact.top1_tokens.len().max(1) as f64,
                }
            })
            .collect(),
    };
    let mut writer = BufWriter::new(
        File::create(&args.output).with_context(|| format!("create {}", args.output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!(
        "multi-joint students {}, sequences {}, scored {}, NLL {:.6} -> {:.6} ({:+.4}% PPL), top1 {:.2}%",
        report.students.len(),
        report.sequences,
        report.scored_tokens,
        exact_mean_nll,
        student_mean_nll,
        report.relative_perplexity_delta * 100.0,
        report.top1_agreement * 100.0,
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn multi_expert_student_nll(args: MultiExpertStudentNllArgs) -> Result<()> {
    if !args.model.is_file()
        || args.corpora.is_empty()
        || args.corpora.iter().any(|path| !path.is_file())
        || args.students.is_empty()
    {
        bail!("model, corpus, and student inputs must exist");
    }
    if args.max_sequences == 0
        || args.max_tokens_per_sequence < 2
        || args.max_tokens_per_sequence > args.context
    {
        bail!("multi-student corpus arguments are invalid");
    }
    let model_hash = match args.model_sha256.as_ref() {
        Some(value) => {
            validate_sha256(value)?;
            value.clone()
        }
        None => sha256_file(&args.model)?,
    };
    let mut targets = HashSet::new();
    let mut specs = Vec::new();
    for value in &args.students {
        let mut fields = value.splitn(3, ':');
        let layer = fields
            .next()
            .ok_or_else(|| anyhow::anyhow!("student lacks layer"))?
            .parse::<usize>()
            .with_context(|| format!("invalid student layer in {value}"))?;
        let expert = fields
            .next()
            .ok_or_else(|| anyhow::anyhow!("student lacks expert"))?
            .parse::<u16>()
            .with_context(|| format!("invalid student expert in {value}"))?;
        let path = PathBuf::from(
            fields
                .next()
                .ok_or_else(|| anyhow::anyhow!("student lacks weight path"))?,
        );
        if layer >= args.layers || usize::from(expert) >= args.experts || !path.is_file() {
            bail!("student target or path is invalid: {value}");
        }
        if !targets.insert((layer, expert)) {
            bail!("duplicate student target {layer}:{expert}");
        }
        let (neurons, hidden, weights) = read_expert_down_weights(&path)?;
        if (neurons, hidden) != (args.expert_neurons, args.hidden_size) {
            bail!("student {value} dimensions do not match the requested model shape");
        }
        specs.push(ExpertStudentSpec {
            layer,
            expert,
            path,
            weights,
        });
    }
    let mut capture = llama_capture::Capture::open(
        &args.model,
        llama_capture::OpenConfig {
            context: args.context,
            layers: args.layers,
            experts: args.experts,
            top_k: args.top_k,
            gpu_layers: args.gpu_layers,
            threads: args.threads,
            intermediate: Some(args.expert_neurons),
            hidden: args.hidden_size,
            sample: None,
            sample_all_layer: None,
            capture_all_routes: true,
            capture_all_activations: true,
            contribution_packet_neurons: &[],
        },
    )?;
    let mut sequences = Vec::new();
    for path in &args.corpora {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("read corpus {}", path.display()))?;
        for paragraph in split_corpus_paragraphs(&contents) {
            if sequences.len() >= args.max_sequences {
                break;
            }
            let mut tokens = capture.tokenize(&paragraph)?;
            tokens.truncate(args.max_tokens_per_sequence);
            if tokens.len() >= 2 {
                sequences.push(tokens);
            }
        }
        if sequences.len() >= args.max_sequences {
            break;
        }
    }
    if sequences.is_empty() {
        bail!("corpus produced no scoreable token sequences");
    }
    let target_pairs = specs
        .iter()
        .map(|spec| (spec.layer, spec.expert))
        .collect::<Vec<_>>();
    let exact = measure_multi_corpus_nll(&mut capture, &sequences, &target_pairs)?;
    for spec in &specs {
        capture.add_expert_student(spec.layer, spec.expert, &spec.weights)?;
    }
    let student = measure_multi_corpus_nll(&mut capture, &sequences, &target_pairs)?;
    if exact.scored_tokens != student.scored_tokens
        || exact.top1_tokens.len() != student.top1_tokens.len()
    {
        bail!("exact and multi-student passes produced incompatible measurements");
    }
    let exact_mean_nll = exact.mean_nll();
    let student_mean_nll = student.mean_nll();
    let top1_matches = exact
        .top1_tokens
        .iter()
        .zip(&student.top1_tokens)
        .filter(|(exact, student)| exact == student)
        .count();
    let report = MultiExpertStudentNllReport {
        schema_version: 1,
        classification: "simultaneous-folded-expert-student-corpus-nll-control",
        promotion_evidence: false,
        quality_semantics: "teacher-forced-corpus-nll-with-multiple-expert-down-projections-replaced",
        model_sha256: model_hash,
        corpora: args
            .corpora
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        students: specs
            .iter()
            .zip(exact.route_hits.iter().zip(&student.route_hits))
            .map(
                |(spec, (&exact_route_hits, &student_route_hits))| MultiExpertStudentResult {
                    layer: spec.layer,
                    expert: spec.expert,
                    weights: spec.path.display().to_string(),
                    exact_route_hits,
                    student_route_hits,
                },
            )
            .collect(),
        sequences: sequences.len(),
        scored_tokens: exact.scored_tokens,
        exact_mean_nll,
        student_mean_nll,
        nll_delta: student_mean_nll - exact_mean_nll,
        relative_perplexity_delta: (student_mean_nll - exact_mean_nll).exp() - 1.0,
        top1_agreement: top1_matches as f64 / exact.top1_tokens.len().max(1) as f64,
    };
    let mut writer = BufWriter::new(
        File::create(&args.output).with_context(|| format!("create {}", args.output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("multi-expert student NLL report: {}", args.output.display());
    println!(
        "students {}, sequences {}, scored {}, NLL {:.6} -> {:.6} ({:+.4}% PPL), top1 {:.2}%",
        report.students.len(),
        report.sequences,
        report.scored_tokens,
        exact_mean_nll,
        student_mean_nll,
        report.relative_perplexity_delta * 100.0,
        report.top1_agreement * 100.0,
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn measure_multi_corpus_nll(
    capture: &mut llama_capture::Capture,
    sequences: &[Vec<i32>],
    targets: &[(usize, u16)],
) -> Result<CorpusNllMeasurement> {
    let mut measurement = CorpusNllMeasurement {
        route_hits: vec![0; targets.len()],
        ..CorpusNllMeasurement::default()
    };
    for tokens in sequences {
        capture.reset()?;
        for pair in tokens.windows(2) {
            let output = capture.step_target(pair[0], pair[1])?;
            measurement.nll += output
                .target_nll
                .ok_or_else(|| anyhow::anyhow!("llama bridge omitted target NLL"))?
                as f64;
            measurement.scored_tokens += 1;
            measurement.top1_tokens.push(output.predicted);
            for (target_index, &(layer, expert)) in targets.iter().enumerate() {
                if output.routes[layer].contains(&expert) {
                    measurement.route_hits[target_index] += 1;
                }
            }
        }
    }
    Ok(measurement)
}

#[cfg(not(target_os = "macos"))]
fn multi_expert_student_nll(_args: MultiExpertStudentNllArgs) -> Result<()> {
    bail!("multi-expert student NLL currently requires the macOS llama.cpp bridge")
}

#[cfg(not(target_os = "macos"))]
fn multi_joint_expert_student_nll(_args: MultiJointExpertStudentNllArgs) -> Result<()> {
    bail!("multi-joint expert student NLL currently requires the macOS llama.cpp bridge")
}

#[cfg(not(target_os = "macos"))]
fn expert_student_nll(_args: ExpertStudentNllArgs) -> Result<()> {
    bail!("expert student NLL currently requires the macOS llama.cpp bridge")
}

#[cfg(not(target_os = "macos"))]
fn joint_expert_student_nll(_args: JointExpertStudentNllArgs) -> Result<()> {
    bail!("joint expert student NLL currently requires the macOS llama.cpp bridge")
}

#[cfg(not(target_os = "macos"))]
fn capture_corpus(_args: CaptureCorpusArgs) -> Result<()> {
    bail!("corpus capture currently requires the macOS llama.cpp bridge")
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
    let mut packet_selection_semantics = HashMap::<usize, String>::new();
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
            let packets_per_expert = packet_metrics.ranked_relative_squared_error[0][0].len();
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
            if packet_selection_semantics
                .insert(
                    packet_metrics.packet_neurons,
                    packet_metrics.selection_semantics.clone(),
                )
                .is_some_and(|previous| previous != packet_metrics.selection_semantics)
            {
                bail!(
                    "inconsistent selection semantics for {} neurons",
                    packet_metrics.packet_neurons
                );
            }
            let packet_bytes = expert_bytes / packets_per_expert as u64;
            let mut token_selected = vec![0u64; relative_errors.len()];
            for layer in &packet_metrics.ranked_relative_squared_error {
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
                selection_semantics: packet_selection_semantics[&packet_neurons].clone(),
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
        classification: "down-contribution-ranked-oracle-smoke",
        selection_semantics: "per-packet-size-see-results",
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
            "{} neurons {:.3}% relative L2: mean {:.2}/{}, p95 {}, {:.2} MiB/token, unmet {}, {}",
            result.packet_neurons,
            result.relative_l2_error_target * 100.0,
            result.mean_selected_packets_per_expert,
            result.packets_per_expert,
            result.p95_selected_packets_per_expert,
            result.mean_active_packet_bytes_per_token / (1024.0 * 1024.0),
            result.unmet_target_observations,
            result.selection_semantics,
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

#[derive(Default)]
struct GroupOracleAccumulator {
    individual_neurons: Vec<u64>,
    ideal_groups: Vec<u64>,
    learned_groups: Vec<u64>,
    signature_groups: Vec<u64>,
    calibrated_individual_neurons: Vec<u64>,
    calibrated_learned_groups: Vec<u64>,
    calibrated_signature_groups: Vec<u64>,
    active_bytes_per_token: Vec<u64>,
    fallback_identity_observations: usize,
}

fn group_oracle(
    calibration: &Path,
    trace: &Path,
    group_neurons: usize,
    relative_errors: &[f64],
    expert_bytes: u64,
    output: &Path,
) -> Result<()> {
    if group_neurons == 0 {
        bail!("group-neurons must be positive");
    }
    if relative_errors.is_empty()
        || relative_errors
            .iter()
            .any(|error| !error.is_finite() || *error <= 0.0 || *error >= 1.0)
    {
        bail!("relative error targets must be finite values in (0, 1)");
    }
    let trace_header = read_trace_header(trace)?;
    let calibration_header = read_trace_header(calibration)?;
    validate_compatible_headers(&trace_header, &calibration_header)?;

    let mut expert_neurons = None::<usize>;
    let mut frequencies = HashMap::<(usize, u16, usize), Vec<u32>>::new();
    let mut selection_observations = HashMap::<(usize, u16, usize), Vec<Vec<u16>>>::new();
    visit_trace(calibration, &trace_header, |token| {
        let routes = token.routes;
        let contribution = token
            .packet_down_contribution
            .ok_or_else(|| anyhow::anyhow!("calibration token lacks packet down contribution"))?;
        let metrics = contribution
            .packets
            .into_iter()
            .find(|metrics| metrics.packet_neurons == 1)
            .ok_or_else(|| anyhow::anyhow!("calibration trace lacks individual-neuron metrics"))?;
        validate_contribution_metrics(&metrics, &trace_header)?;
        let orders = metrics.ranked_packet_indices.as_ref().ok_or_else(|| {
            anyhow::anyhow!("calibration trace lacks ranked individual-neuron identities")
        })?;
        let width = metrics.ranked_relative_squared_error[0][0].len();
        if expert_neurons
            .replace(width)
            .is_some_and(|previous| previous != width)
        {
            bail!("calibration trace has inconsistent expert width");
        }
        for layer in 0..trace_header.layers {
            for slot in 0..trace_header.top_k {
                let expert = routes[layer][slot];
                let curve = &metrics.ranked_relative_squared_error[layer][slot];
                let order = &orders[layer][slot];
                for (target_index, &target) in relative_errors.iter().enumerate() {
                    let (selected, _, _) = select_contribution_packets(curve, target)?;
                    let counts = frequencies
                        .entry((layer, expert, target_index))
                        .or_insert_with(|| vec![0u32; width]);
                    for &neuron in &order[..selected] {
                        counts[neuron as usize] += 1;
                    }
                    selection_observations
                        .entry((layer, expert, target_index))
                        .or_default()
                        .push(order[..selected].to_vec());
                }
            }
        }
        Ok(())
    })?;
    let expert_neurons =
        expert_neurons.ok_or_else(|| anyhow::anyhow!("empty calibration trace"))?;
    if !expert_neurons.is_multiple_of(group_neurons) {
        bail!("group-neurons must divide the expert neuron count exactly");
    }
    let groups_per_expert = expert_neurons / group_neurons;
    if !expert_bytes.is_multiple_of(groups_per_expert as u64) {
        bail!("expert-bytes must divide evenly across learned groups");
    }
    let packet_bytes = expert_bytes / groups_per_expert as u64;
    let layouts = frequencies
        .iter()
        .map(|(&key, counts)| (key, frequency_packed_groups(counts, group_neurons)))
        .collect::<HashMap<_, _>>();
    let signature_layouts = selection_observations
        .iter()
        .map(|(&key, observations)| {
            (
                key,
                signature_packed_groups(observations, expert_neurons, group_neurons),
            )
        })
        .collect::<HashMap<_, _>>();

    let mut accumulators = (0..relative_errors.len())
        .map(|_| GroupOracleAccumulator::default())
        .collect::<Vec<_>>();
    visit_trace(trace, &trace_header, |token| {
        let routes = token.routes;
        let contribution = token
            .packet_down_contribution
            .ok_or_else(|| anyhow::anyhow!("held-out token lacks packet down contribution"))?;
        let metrics = contribution
            .packets
            .into_iter()
            .find(|metrics| metrics.packet_neurons == 1)
            .ok_or_else(|| anyhow::anyhow!("held-out trace lacks individual-neuron metrics"))?;
        validate_contribution_metrics(&metrics, &trace_header)?;
        let orders = metrics.ranked_packet_indices.as_ref().ok_or_else(|| {
            anyhow::anyhow!("held-out trace lacks ranked individual-neuron identities")
        })?;
        if metrics.ranked_relative_squared_error[0][0].len() != expert_neurons {
            bail!("held-out trace expert width does not match calibration");
        }
        let mut token_groups = vec![0u64; relative_errors.len()];
        for layer in 0..trace_header.layers {
            for slot in 0..trace_header.top_k {
                let expert = routes[layer][slot];
                let curve = &metrics.ranked_relative_squared_error[layer][slot];
                let order = &orders[layer][slot];
                for (target_index, &target) in relative_errors.iter().enumerate() {
                    let (selected, _, _) = select_contribution_packets(curve, target)?;
                    let identity;
                    let (groups, signature_groups, calibrated) =
                        if let (Some(groups), Some(signature_groups)) = (
                            layouts.get(&(layer, expert, target_index)),
                            signature_layouts.get(&(layer, expert, target_index)),
                        ) {
                            (groups, signature_groups, true)
                        } else {
                            identity = (0..expert_neurons)
                                .map(|neuron| neuron / group_neurons)
                                .collect::<Vec<_>>();
                            accumulators[target_index].fallback_identity_observations += 1;
                            (&identity, &identity, false)
                        };
                    let mut touched = vec![false; groups_per_expert];
                    let mut signature_touched = vec![false; groups_per_expert];
                    for &neuron in &order[..selected] {
                        touched[groups[neuron as usize]] = true;
                        signature_touched[signature_groups[neuron as usize]] = true;
                    }
                    let learned = touched.into_iter().filter(|value| *value).count();
                    let signature = signature_touched.into_iter().filter(|value| *value).count();
                    accumulators[target_index]
                        .individual_neurons
                        .push(selected as u64);
                    accumulators[target_index]
                        .ideal_groups
                        .push(selected.div_ceil(group_neurons) as u64);
                    accumulators[target_index]
                        .learned_groups
                        .push(learned as u64);
                    accumulators[target_index]
                        .signature_groups
                        .push(signature as u64);
                    if calibrated {
                        accumulators[target_index]
                            .calibrated_individual_neurons
                            .push(selected as u64);
                        accumulators[target_index]
                            .calibrated_learned_groups
                            .push(learned as u64);
                        accumulators[target_index]
                            .calibrated_signature_groups
                            .push(signature as u64);
                    }
                    token_groups[target_index] += signature as u64;
                }
            }
        }
        for (target_index, groups) in token_groups.into_iter().enumerate() {
            accumulators[target_index]
                .active_bytes_per_token
                .push(groups.saturating_mul(packet_bytes));
        }
        Ok(())
    })?;

    let mut results = Vec::new();
    for (target_index, &target) in relative_errors.iter().enumerate() {
        let accumulator = &accumulators[target_index];
        if accumulator.individual_neurons.is_empty() {
            bail!("held-out trace contains no expert observations");
        }
        let observations = accumulator.individual_neurons.len();
        let individual_total = accumulator.individual_neurons.iter().sum::<u64>();
        let ideal_total = accumulator.ideal_groups.iter().sum::<u64>();
        let learned_total = accumulator.learned_groups.iter().sum::<u64>();
        let signature_total = accumulator.signature_groups.iter().sum::<u64>();
        let calibrated_individual_total = accumulator
            .calibrated_individual_neurons
            .iter()
            .sum::<u64>();
        let calibrated_learned_total = accumulator.calibrated_learned_groups.iter().sum::<u64>();
        let calibrated_signature_total =
            accumulator.calibrated_signature_groups.iter().sum::<u64>();
        let calibrated_observations = accumulator.calibrated_learned_groups.len();
        results.push(GroupOracleResult {
            relative_l2_error_target: target,
            heldout_expert_observations: observations,
            fallback_identity_observations: accumulator.fallback_identity_observations,
            mean_individual_neurons: individual_total as f64 / observations as f64,
            mean_individual_fraction: individual_total as f64
                / (observations * expert_neurons) as f64,
            mean_ideal_groups: ideal_total as f64 / observations as f64,
            mean_frequency_packed_groups: learned_total as f64 / observations as f64,
            p50_frequency_packed_groups: percentile(&accumulator.learned_groups, 0.50),
            p95_frequency_packed_groups: percentile(&accumulator.learned_groups, 0.95),
            p99_frequency_packed_groups: percentile(&accumulator.learned_groups, 0.99),
            mean_frequency_group_load_amplification: learned_total as f64 * group_neurons as f64
                / individual_total.max(1) as f64,
            mean_frequency_resident_neuron_fraction: learned_total as f64
                / (observations * groups_per_expert) as f64,
            mean_signature_packed_groups: signature_total as f64 / observations as f64,
            p50_signature_packed_groups: percentile(&accumulator.signature_groups, 0.50),
            p95_signature_packed_groups: percentile(&accumulator.signature_groups, 0.95),
            p99_signature_packed_groups: percentile(&accumulator.signature_groups, 0.99),
            mean_signature_group_load_amplification: signature_total as f64 * group_neurons as f64
                / individual_total.max(1) as f64,
            mean_signature_resident_neuron_fraction: signature_total as f64
                / (observations * groups_per_expert) as f64,
            calibrated_only_observations: calibrated_observations,
            calibrated_only_mean_frequency_group_load_amplification: calibrated_learned_total
                as f64
                * group_neurons as f64
                / calibrated_individual_total.max(1) as f64,
            calibrated_only_mean_frequency_resident_neuron_fraction: calibrated_learned_total
                as f64
                / (calibrated_observations.max(1) * groups_per_expert) as f64,
            calibrated_only_p95_frequency_groups: percentile(
                &accumulator.calibrated_learned_groups,
                0.95,
            ),
            calibrated_only_mean_signature_group_load_amplification: calibrated_signature_total
                as f64
                * group_neurons as f64
                / calibrated_individual_total.max(1) as f64,
            calibrated_only_mean_signature_resident_neuron_fraction: calibrated_signature_total
                as f64
                / (calibrated_observations.max(1) * groups_per_expert) as f64,
            calibrated_only_p95_signature_groups: percentile(
                &accumulator.calibrated_signature_groups,
                0.95,
            ),
            mean_active_packet_bytes_per_token: accumulator
                .active_bytes_per_token
                .iter()
                .sum::<u64>() as f64
                / accumulator.active_bytes_per_token.len().max(1) as f64,
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

    let report = GroupOracleReport {
        schema_version: 1,
        classification: "learned-neuron-group-coverage-smoke",
        layout_semantics: "per-layer-expert-target-frequency-and-co-selection-signature-fixed-groups",
        quality_semantics: "group-coverage-of-individual-contribution-prefix-not-grouped-logit-or-nll",
        calibration: calibration.display().to_string(),
        trace: trace.display().to_string(),
        trace_header,
        expert_bytes,
        expert_neurons,
        group_neurons,
        results,
    };
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("create {}", output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("group oracle report: {}", output.display());
    for result in &report.results {
        println!(
            "{:.1}% error: individual {:.1}%, frequency {:.1}%/{:.1}% calibrated, signature {:.1}%/{:.1}% calibrated, signature amplification {:.2}x/{:.2}x calibrated, signature p95 {}/{}, {:.2} MiB/token, fallback {}",
            result.relative_l2_error_target * 100.0,
            result.mean_individual_fraction * 100.0,
            result.mean_frequency_resident_neuron_fraction * 100.0,
            result.calibrated_only_mean_frequency_resident_neuron_fraction * 100.0,
            result.mean_signature_resident_neuron_fraction * 100.0,
            result.calibrated_only_mean_signature_resident_neuron_fraction * 100.0,
            result.mean_signature_group_load_amplification,
            result.calibrated_only_mean_signature_group_load_amplification,
            result.p95_signature_packed_groups,
            result.calibrated_only_p95_signature_groups,
            result.mean_active_packet_bytes_per_token / (1024.0 * 1024.0),
            result.fallback_identity_observations,
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn structured_nll(args: StructuredNllArgs) -> Result<()> {
    if !args.model.is_file() {
        bail!("model does not exist: {}", args.model.display());
    }
    if args.calibrations.is_empty() {
        bail!("at least one calibration trace is required");
    }
    if args.layers == 0 || args.experts == 0 || args.top_k == 0 || args.expert_neurons == 0 {
        bail!("layers, experts, top-k, and expert-neurons must be positive");
    }
    if args.group_neurons == 0 || !args.expert_neurons.is_multiple_of(args.group_neurons) {
        bail!("group-neurons must divide expert-neurons exactly");
    }
    if !args.relative_error.is_finite() || args.relative_error <= 0.0 || args.relative_error >= 1.0
    {
        bail!("relative-error must be a finite value in (0, 1)");
    }
    if args.retained_fractions.is_empty()
        || args
            .retained_fractions
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
    {
        bail!("retained-fractions must be finite values in (0, 1]");
    }
    let model_hash = match args.model_sha256.as_ref() {
        Some(value) => {
            validate_sha256(value)?;
            value.clone()
        }
        None => sha256_file(&args.model)?,
    };
    let (header, calibration) =
        train_structured_calibration(&args.calibrations, args.relative_error, args.expert_neurons)?;
    if header.model_sha256 != model_hash {
        bail!("calibration model SHA-256 does not match the requested model");
    }
    if (header.layers, header.experts, header.top_k) != (args.layers, args.experts, args.top_k) {
        bail!("calibration shape does not match structured-NLL arguments");
    }

    let mut exact_capture = open_structured_capture(&args)?;
    let tokens = exact_capture.tokenize(&args.prompt)?;
    if tokens.len() < 2 {
        bail!("structured-NLL prompt must tokenize to at least two tokens");
    }
    if tokens.len() > args.context {
        bail!(
            "structured-NLL prompt has {} tokens, exceeding context {}",
            tokens.len(),
            args.context
        );
    }
    let exact = measure_teacher_forced_nll(&mut exact_capture, &tokens)?;
    drop(exact_capture);

    let groups_per_expert = args.expert_neurons / args.group_neurons;
    let directly_calibrated_layer_experts = calibration.counts.len();
    let total_layer_experts = args.layers * args.experts;
    let mut results = Vec::new();
    for &fraction in &args.retained_fractions {
        let retained_groups =
            ((groups_per_expert as f64 * fraction).ceil() as usize).clamp(1, groups_per_expert);
        let mask = build_structured_mask(
            &calibration,
            args.layers,
            args.experts,
            args.expert_neurons,
            args.group_neurons,
            retained_groups,
        )?;
        let mut capture = open_structured_capture(&args)?;
        capture.set_neuron_mask(&mask)?;
        let measurement = measure_teacher_forced_nll(&mut capture, &tokens)?;
        let agreements = measurement
            .top1_tokens
            .iter()
            .zip(&exact.top1_tokens)
            .filter(|(candidate, reference)| candidate == reference)
            .count();
        results.push(StructuredNllResult {
            requested_retained_fraction: fraction,
            retained_groups_per_expert: retained_groups,
            actual_retained_fraction: retained_groups as f64 / groups_per_expert as f64,
            mean_nll: measurement.mean_nll,
            perplexity: measurement.perplexity,
            nll_delta_vs_exact: measurement.mean_nll - exact.mean_nll,
            relative_perplexity_vs_exact: (measurement.mean_nll - exact.mean_nll).exp() - 1.0,
            top1_agreement_vs_exact: agreements as f64 / exact.top1_tokens.len() as f64,
        });
    }

    let report = StructuredNllReport {
        schema_version: 1,
        classification: "post-training-fixed-group-mask-nll-smoke",
        promotion_evidence: false,
        training_semantics: "calibration-contribution-frequency-and-co-selection-no-weight-updates",
        masking_semantics: "fixed-per-layer-expert-neuron-mask-applied-before-down-projection",
        quality_semantics: "teacher-forced-next-token-nll-on-one-disjoint-prompt",
        model: args.model.display().to_string(),
        model_sha256: model_hash,
        llama_revision: header.llama_revision,
        calibrations: args
            .calibrations
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        calibration_token_positions: calibration.token_positions,
        calibration_expert_observations: calibration.expert_observations,
        directly_calibrated_layer_experts,
        fallback_layer_experts: total_layer_experts - directly_calibrated_layer_experts,
        prompt_sha256: format!("{:x}", Sha256::digest(args.prompt.as_bytes())),
        prompt_tokens: tokens.len(),
        scored_positions: tokens.len() - 1,
        group_neurons: args.group_neurons,
        groups_per_expert,
        relative_l2_training_target: args.relative_error,
        exact,
        results,
    };
    let mut writer = BufWriter::new(
        File::create(&args.output).with_context(|| format!("create {}", args.output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("structured NLL report: {}", args.output.display());
    println!(
        "exact: NLL {:.6}, PPL {:.4}, scored positions {}",
        report.exact.mean_nll, report.exact.perplexity, report.scored_positions
    );
    for result in &report.results {
        println!(
            "retain {:.1}%: NLL {:.6} ({:+.6}), relative PPL {:+.2}%, top-1 {:.1}%",
            result.actual_retained_fraction * 100.0,
            result.mean_nll,
            result.nll_delta_vs_exact,
            result.relative_perplexity_vs_exact * 100.0,
            result.top1_agreement_vs_exact * 100.0,
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn structured_nll(_args: StructuredNllArgs) -> Result<()> {
    bail!("structured NLL currently requires the macOS llama.cpp bridge")
}

fn train_structured_calibration(
    paths: &[PathBuf],
    relative_error: f64,
    expert_neurons: usize,
) -> Result<(TraceHeader, StructuredCalibration)> {
    let header = read_trace_header(&paths[0])?;
    let mut calibration = StructuredCalibration::default();
    for path in paths {
        let candidate_header = read_trace_header(path)?;
        validate_compatible_headers(&header, &candidate_header)?;
        visit_trace(path, &header, |token| {
            let routes = token.routes;
            let contribution = token.packet_down_contribution.ok_or_else(|| {
                anyhow::anyhow!("calibration token lacks packet down contribution")
            })?;
            let metrics = contribution
                .packets
                .into_iter()
                .find(|metrics| metrics.packet_neurons == 1)
                .ok_or_else(|| {
                    anyhow::anyhow!("calibration trace lacks individual-neuron metrics")
                })?;
            validate_contribution_metrics(&metrics, &header)?;
            let orders = metrics.ranked_packet_indices.as_ref().ok_or_else(|| {
                anyhow::anyhow!("calibration trace lacks ranked individual-neuron identities")
            })?;
            if metrics.ranked_relative_squared_error[0][0].len() != expert_neurons {
                bail!("calibration expert width does not match expert-neurons");
            }
            calibration.token_positions += 1;
            for layer in 0..header.layers {
                for slot in 0..header.top_k {
                    let expert = routes[layer][slot];
                    let (selected, _, _) = select_contribution_packets(
                        &metrics.ranked_relative_squared_error[layer][slot],
                        relative_error,
                    )?;
                    let selected_neurons = orders[layer][slot][..selected].to_vec();
                    let counts = calibration
                        .counts
                        .entry((layer, expert))
                        .or_insert_with(|| vec![0u32; expert_neurons]);
                    let layer_counts = calibration
                        .layer_counts
                        .entry(layer)
                        .or_insert_with(|| vec![0u32; expert_neurons]);
                    for &neuron in &selected_neurons {
                        counts[neuron as usize] += 1;
                        layer_counts[neuron as usize] += 1;
                    }
                    calibration
                        .observations
                        .entry((layer, expert))
                        .or_default()
                        .push(selected_neurons.clone());
                    calibration
                        .layer_observations
                        .entry(layer)
                        .or_default()
                        .push(selected_neurons);
                    calibration.expert_observations += 1;
                }
            }
            Ok(())
        })?;
    }
    if calibration.token_positions == 0 {
        bail!("calibration traces contain no token positions");
    }
    Ok((header, calibration))
}

fn build_structured_mask(
    calibration: &StructuredCalibration,
    layers: usize,
    experts: usize,
    expert_neurons: usize,
    group_neurons: usize,
    retained_groups: usize,
) -> Result<Vec<u8>> {
    let groups_per_expert = expert_neurons / group_neurons;
    if retained_groups == 0 || retained_groups > groups_per_expert {
        bail!("retained group count is outside the valid range");
    }
    let mut mask = vec![0u8; layers * experts * expert_neurons];
    for layer in 0..layers {
        let fallback_counts = calibration
            .layer_counts
            .get(&layer)
            .ok_or_else(|| anyhow::anyhow!("calibration has no observations for layer {layer}"))?;
        let fallback_observations =
            calibration.layer_observations.get(&layer).ok_or_else(|| {
                anyhow::anyhow!("calibration has no selection signatures for layer {layer}")
            })?;
        for expert in 0..experts {
            let key = (layer, expert as u16);
            let counts = calibration.counts.get(&key).unwrap_or(fallback_counts);
            let observations = calibration
                .observations
                .get(&key)
                .unwrap_or(fallback_observations);
            let layout = signature_packed_groups(observations, expert_neurons, group_neurons);
            let mut scores = vec![0u64; groups_per_expert];
            for neuron in 0..expert_neurons {
                scores[layout[neuron]] += counts[neuron] as u64;
            }
            let mut group_order = (0..groups_per_expert).collect::<Vec<_>>();
            group_order.sort_unstable_by_key(|&group| (Reverse(scores[group]), group));
            let mut retained = vec![false; groups_per_expert];
            for group in group_order.into_iter().take(retained_groups) {
                retained[group] = true;
            }
            let base = (layer * experts + expert) * expert_neurons;
            for neuron in 0..expert_neurons {
                mask[base + neuron] = u8::from(retained[layout[neuron]]);
            }
        }
    }
    Ok(mask)
}

#[cfg(target_os = "macos")]
fn open_structured_capture(args: &StructuredNllArgs) -> Result<llama_capture::Capture> {
    llama_capture::Capture::open(
        &args.model,
        llama_capture::OpenConfig {
            context: args.context,
            layers: args.layers,
            experts: args.experts,
            top_k: args.top_k,
            gpu_layers: args.gpu_layers,
            threads: args.threads,
            intermediate: Some(args.expert_neurons),
            hidden: args.hidden_size,
            sample: None,
            sample_all_layer: None,
            capture_all_routes: true,
            capture_all_activations: true,
            contribution_packet_neurons: &[],
        },
    )
}

#[cfg(target_os = "macos")]
fn measure_teacher_forced_nll(
    capture: &mut llama_capture::Capture,
    tokens: &[i32],
) -> Result<StructuredNllMeasurement> {
    let mut nll = 0.0f64;
    let mut top1_tokens = Vec::with_capacity(tokens.len() - 1);
    for pair in tokens.windows(2) {
        let output = capture.step_target(pair[0], pair[1])?;
        nll += output
            .target_nll
            .ok_or_else(|| anyhow::anyhow!("llama bridge omitted target NLL"))?
            as f64;
        top1_tokens.push(output.predicted);
    }
    let mean_nll = nll / top1_tokens.len() as f64;
    Ok(StructuredNllMeasurement {
        mean_nll,
        perplexity: mean_nll.exp(),
        top1_tokens,
    })
}

#[allow(clippy::too_many_arguments)]
fn distill_expert(
    traces: &[PathBuf],
    heldout_traces: &[PathBuf],
    teacher_down_weights: &Path,
    calibration_paths: &[PathBuf],
    layer: usize,
    expert: u16,
    group_neurons: usize,
    retained_fraction: f64,
    relative_error: f64,
    train_fraction: f64,
    max_train_samples: Option<usize>,
    student_down_weights_output: Option<&Path>,
    ridge: f64,
    output: &Path,
) -> Result<()> {
    if traces.is_empty() {
        bail!("at least one expert sample trace is required");
    }
    if group_neurons == 0 {
        bail!("group-neurons must be positive");
    }
    if !retained_fraction.is_finite() || retained_fraction <= 0.0 || retained_fraction > 1.0 {
        bail!("retained-fraction must be a finite value in (0, 1]");
    }
    if heldout_traces.is_empty()
        && (!train_fraction.is_finite() || train_fraction <= 0.0 || train_fraction >= 1.0)
    {
        bail!("train-fraction must be a finite value in (0, 1)");
    }
    if heldout_traces.is_empty() && max_train_samples.is_some() {
        bail!("max-train-samples requires explicit heldout traces");
    }
    if max_train_samples.is_some_and(|limit| limit < 2) {
        bail!("max-train-samples must be at least two");
    }
    if !ridge.is_finite() || ridge <= 0.0 {
        bail!("ridge must be finite and positive");
    }
    let trace_header = read_trace_header(&traces[0])?;
    if layer >= trace_header.layers || usize::from(expert) >= trace_header.experts {
        bail!("distillation target is outside the trace model shape");
    }
    let mut samples = read_expert_samples(traces, &trace_header, layer, expert)?;
    let available_train_samples = samples.len();
    if let Some(limit) = max_train_samples {
        samples.truncate(limit);
    }
    let (train, heldout, split_semantics) = if heldout_traces.is_empty() {
        if samples.len() < 4 {
            bail!(
                "chronological distillation split requires at least four expert hits, found {}",
                samples.len()
            );
        }
        let train_samples =
            ((samples.len() as f64 * train_fraction).floor() as usize).clamp(2, samples.len() - 1);
        let heldout = samples.split_off(train_samples);
        (samples, heldout, "chronological-within-training-traces")
    } else {
        let heldout = read_expert_samples(heldout_traces, &trace_header, layer, expert)?;
        if samples.len() < 2 || heldout.is_empty() {
            bail!(
                "explicit corpus split requires at least two train hits and one held-out hit; found {}/{}",
                samples.len(),
                heldout.len()
            );
        }
        (samples, heldout, "explicit-disjoint-corpus-traces")
    };
    let expert_neurons = train[0].activation.len();
    let hidden = train[0].teacher_down_output.len();
    if expert_neurons == 0
        || hidden == 0
        || train.iter().chain(&heldout).any(|sample| {
            sample.activation.len() != expert_neurons
                || sample.teacher_down_output.len() != hidden
                || sample
                    .activation
                    .iter()
                    .chain(&sample.teacher_down_output)
                    .any(|value| !value.is_finite())
        })
    {
        bail!("expert samples have inconsistent, empty, or non-finite vectors");
    }
    if !expert_neurons.is_multiple_of(group_neurons) {
        bail!("group-neurons must divide the sampled expert width");
    }
    let (calibration_header, calibration) =
        train_structured_calibration(calibration_paths, relative_error, expert_neurons)?;
    validate_compatible_headers(&trace_header, &calibration_header)?;
    let groups_per_expert = expert_neurons / group_neurons;
    let retained_groups = ((groups_per_expert as f64 * retained_fraction).ceil() as usize)
        .clamp(1, groups_per_expert);
    let full_mask = build_structured_mask(
        &calibration,
        trace_header.layers,
        trace_header.experts,
        expert_neurons,
        group_neurons,
        retained_groups,
    )?;
    let mask_base = (layer * trace_header.experts + usize::from(expert)) * expert_neurons;
    let retained_indices = full_mask[mask_base..mask_base + expert_neurons]
        .iter()
        .enumerate()
        .filter_map(|(neuron, &retained)| (retained != 0).then_some(neuron))
        .collect::<Vec<_>>();
    let (weight_neurons, weight_hidden, teacher_weights) =
        read_expert_down_weights(teacher_down_weights)?;
    if (weight_neurons, weight_hidden) != (expert_neurons, hidden) {
        bail!("teacher down weight dimensions do not match captured samples");
    }
    let train_samples = train.len();
    let train_residuals = train
        .iter()
        .map(|sample| {
            let base = project_down_subset(
                &teacher_weights,
                &sample.activation,
                &retained_indices,
                expert_neurons,
                hidden,
            );
            sample
                .teacher_down_output
                .iter()
                .zip(base)
                .map(|(&target, predicted)| target - predicted)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut kernel = vec![0.0f64; train_samples * train_samples];
    for row in 0..train_samples {
        for column in 0..=row {
            let value = retained_indices
                .iter()
                .map(|&neuron| {
                    train[row].activation[neuron] as f64 * train[column].activation[neuron] as f64
                })
                .sum::<f64>();
            kernel[row * train_samples + column] = value;
            kernel[column * train_samples + row] = value;
        }
    }
    let mean_diagonal = (0..train_samples)
        .map(|index| kernel[index * train_samples + index])
        .sum::<f64>()
        / train_samples as f64;
    let ridge_absolute = ridge * mean_diagonal.max(f64::EPSILON);
    for index in 0..train_samples {
        kernel[index * train_samples + index] += ridge_absolute;
    }
    cholesky_in_place(&mut kernel, train_samples)?;

    // The linear-kernel dual is exactly a reduced primal projection. Fold the
    // learned residual into the retained teacher columns so runtime needs one
    // compact down matrix, not a support-vector bank.
    let retained_neurons = retained_indices.len();
    let mut student_weights = vec![0.0f32; hidden * retained_neurons];
    for output_index in 0..hidden {
        let mut coefficients = train_residuals
            .iter()
            .map(|residual| residual[output_index] as f64)
            .collect::<Vec<_>>();
        cholesky_solve(&kernel, train_samples, &mut coefficients);
        for (compact_neuron, &teacher_neuron) in retained_indices.iter().enumerate() {
            let correction = train
                .iter()
                .zip(&coefficients)
                .map(|(sample, coefficient)| sample.activation[teacher_neuron] as f64 * coefficient)
                .sum::<f64>();
            student_weights[output_index * retained_neurons + compact_neuron] =
                teacher_weights[output_index * expert_neurons + teacher_neuron] + correction as f32;
        }
    }
    if let Some(path) = student_down_weights_output {
        let mut dense_student_weights = vec![0.0f32; hidden * expert_neurons];
        for output_index in 0..hidden {
            for (compact_neuron, &teacher_neuron) in retained_indices.iter().enumerate() {
                dense_student_weights[output_index * expert_neurons + teacher_neuron] =
                    student_weights[output_index * retained_neurons + compact_neuron];
            }
        }
        write_expert_down_weights(path, expert_neurons, hidden, &dense_student_weights)?;
    }

    let mut relative_l2 = Vec::with_capacity(heldout.len());
    let mut cosines = Vec::with_capacity(heldout.len());
    let mut masked_relative_l2 = Vec::with_capacity(heldout.len());
    let mut masked_cosines = Vec::with_capacity(heldout.len());
    let mut primal_dual_relative_l2 = Vec::with_capacity(heldout.len());
    for sample in &heldout {
        let mut coefficients = train
            .iter()
            .map(|training| {
                retained_indices
                    .iter()
                    .map(|&neuron| {
                        training.activation[neuron] as f64 * sample.activation[neuron] as f64
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        cholesky_solve(&kernel, train_samples, &mut coefficients);
        let mut squared_error = 0.0f64;
        let mut target_norm = 0.0f64;
        let mut predicted_norm = 0.0f64;
        let mut dot = 0.0f64;
        let base = project_down_subset(
            &teacher_weights,
            &sample.activation,
            &retained_indices,
            expert_neurons,
            hidden,
        );
        let mut masked_squared_error = 0.0f64;
        let mut masked_norm = 0.0f64;
        let mut masked_dot = 0.0f64;
        let mut primal_dual_squared_error = 0.0f64;
        let mut dual_norm = 0.0f64;
        for output_index in 0..hidden {
            let residual = train_residuals
                .iter()
                .zip(&coefficients)
                .map(|(training_residual, coefficient)| {
                    *coefficient * training_residual[output_index] as f64
                })
                .sum::<f64>();
            let masked = base[output_index] as f64;
            let predicted = masked + residual;
            let primal = retained_indices
                .iter()
                .enumerate()
                .map(|(compact_neuron, &teacher_neuron)| {
                    student_weights[output_index * retained_neurons + compact_neuron]
                        * sample.activation[teacher_neuron]
                })
                .sum::<f32>() as f64;
            let target = sample.teacher_down_output[output_index] as f64;
            squared_error += (predicted - target).powi(2);
            masked_squared_error += (masked - target).powi(2);
            target_norm += target * target;
            predicted_norm += predicted * predicted;
            masked_norm += masked * masked;
            dot += predicted * target;
            masked_dot += masked * target;
            primal_dual_squared_error += (primal - predicted).powi(2);
            dual_norm += predicted * predicted;
        }
        relative_l2.push((squared_error / target_norm.max(f64::MIN_POSITIVE)).sqrt());
        cosines.push(dot / (predicted_norm * target_norm).sqrt().max(f64::MIN_POSITIVE));
        masked_relative_l2.push((masked_squared_error / target_norm.max(f64::MIN_POSITIVE)).sqrt());
        masked_cosines.push(masked_dot / (masked_norm * target_norm).sqrt().max(f64::MIN_POSITIVE));
        primal_dual_relative_l2
            .push((primal_dual_squared_error / dual_norm.max(f64::MIN_POSITIVE)).sqrt());
    }
    let student_down_parameters = retained_neurons * hidden;
    let report = ExpertDistillReport {
        schema_version: 2,
        classification: "expert-local-fixed-group-ridge-distillation-smoke",
        promotion_evidence: false,
        training_semantics: "kernel-ridge-fit-of-reduced-down-projection-from-real-teacher-activations",
        inference_semantics: "linear-kernel-dual-folded-into-one-compact-primal-down-matrix",
        quality_semantics: "heldout-local-expert-down-output-not-logit-or-nll",
        split_semantics,
        traces: traces
            .iter()
            .map(|trace| trace.display().to_string())
            .collect(),
        heldout_traces: heldout_traces
            .iter()
            .map(|trace| trace.display().to_string())
            .collect(),
        teacher_down_weights: teacher_down_weights.display().to_string(),
        student_down_weights: student_down_weights_output.map(|path| path.display().to_string()),
        calibrations: calibration_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        trace_header,
        layer,
        expert,
        group_neurons,
        retained_groups,
        retained_neurons,
        retained_fraction: retained_neurons as f64 / expert_neurons as f64,
        available_train_samples,
        train_samples,
        heldout_samples: heldout.len(),
        ridge_relative_to_mean_kernel_diagonal: ridge,
        ridge_absolute,
        student_down_parameters,
        student_down_f16_bytes: student_down_parameters * 2,
        student_down_q4_payload_bytes: student_down_parameters.div_ceil(2),
        dual_support_f32_bytes_avoided: train_samples * (retained_neurons + hidden) * 4,
        primal_dual_max_relative_l2: primal_dual_relative_l2
            .into_iter()
            .reduce(f64::max)
            .unwrap_or(0.0),
        heldout_mean_relative_l2: relative_l2.iter().sum::<f64>() / relative_l2.len() as f64,
        heldout_p50_relative_l2: percentile_f64(&relative_l2, 0.50)
            .expect("heldout split is non-empty"),
        heldout_p95_relative_l2: percentile_f64(&relative_l2, 0.95)
            .expect("heldout split is non-empty"),
        heldout_mean_cosine: cosines.iter().sum::<f64>() / cosines.len() as f64,
        masked_teacher_mean_relative_l2: masked_relative_l2.iter().sum::<f64>()
            / masked_relative_l2.len() as f64,
        masked_teacher_p95_relative_l2: percentile_f64(&masked_relative_l2, 0.95)
            .expect("heldout split is non-empty"),
        masked_teacher_mean_cosine: masked_cosines.iter().sum::<f64>()
            / masked_cosines.len() as f64,
        heldout_zero_baseline_relative_l2: 1.0,
    };
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("create {}", output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("expert distillation report: {}", output.display());
    println!(
        "layer {} expert {}: train {}, heldout {}, retain {:.1}%, masked/distilled relative L2 {:.4}/{:.4}, distilled p95 {:.4}, cosine {:.4}",
        layer,
        expert,
        train_samples,
        heldout.len(),
        report.retained_fraction * 100.0,
        report.masked_teacher_mean_relative_l2,
        report.heldout_mean_relative_l2,
        report.heldout_p95_relative_l2,
        report.heldout_mean_cosine,
    );
    Ok(())
}

fn read_expert_samples(
    traces: &[PathBuf],
    expected_header: &TraceHeader,
    layer: usize,
    expert: u16,
) -> Result<Vec<ExpertDistillSample>> {
    let mut samples = Vec::new();
    for trace in traces {
        let candidate_header = read_trace_header(trace)?;
        validate_compatible_headers(expected_header, &candidate_header)?;
        visit_trace(trace, expected_header, |token| {
            if let Some(sample) = token.expert_distill_sample {
                if sample.layer != layer || sample.expert != expert {
                    bail!("trace contains an expert sample for a different target");
                }
                samples.push(sample);
            }
            for sample in token.expert_distill_samples {
                if sample.layer == layer && sample.expert == expert {
                    samples.push(sample);
                }
            }
            Ok(())
        })?;
    }
    Ok(samples)
}

fn project_down_subset(
    weights: &[f32],
    activation: &[f32],
    retained_indices: &[usize],
    neurons: usize,
    hidden: usize,
) -> Vec<f32> {
    debug_assert_eq!(weights.len(), neurons * hidden);
    debug_assert_eq!(activation.len(), neurons);
    (0..hidden)
        .map(|output| {
            retained_indices
                .iter()
                .map(|&neuron| weights[output * neurons + neuron] * activation[neuron])
                .sum()
        })
        .collect()
}

fn cholesky_in_place(matrix: &mut [f64], size: usize) -> Result<()> {
    if matrix.len() != size * size || size == 0 {
        bail!("Cholesky matrix has an invalid shape");
    }
    for row in 0..size {
        for column in 0..=row {
            let mut value = matrix[row * size + column];
            for inner in 0..column {
                value -= matrix[row * size + inner] * matrix[column * size + inner];
            }
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    bail!("ridge kernel is not positive definite");
                }
                matrix[row * size + column] = value.sqrt();
            } else {
                matrix[row * size + column] = value / matrix[column * size + column];
            }
        }
        for column in row + 1..size {
            matrix[row * size + column] = 0.0;
        }
    }
    Ok(())
}

fn cholesky_solve(cholesky: &[f64], size: usize, values: &mut [f64]) {
    debug_assert_eq!(cholesky.len(), size * size);
    debug_assert_eq!(values.len(), size);
    for row in 0..size {
        let correction = (0..row)
            .map(|column| cholesky[row * size + column] * values[column])
            .sum::<f64>();
        values[row] = (values[row] - correction) / cholesky[row * size + row];
    }
    for row in (0..size).rev() {
        let correction = (row + 1..size)
            .map(|column| cholesky[column * size + row] * values[column])
            .sum::<f64>();
        values[row] = (values[row] - correction) / cholesky[row * size + row];
    }
}

fn frequency_packed_groups(frequency: &[u32], group_neurons: usize) -> Vec<usize> {
    let mut neurons = (0..frequency.len()).collect::<Vec<_>>();
    neurons.sort_unstable_by_key(|&neuron| (Reverse(frequency[neuron]), neuron));
    let mut groups = vec![0usize; frequency.len()];
    for (rank, neuron) in neurons.into_iter().enumerate() {
        groups[neuron] = rank / group_neurons;
    }
    groups
}

fn signature_packed_groups(
    observations: &[Vec<u16>],
    neurons: usize,
    group_neurons: usize,
) -> Vec<usize> {
    let words = observations.len().div_ceil(64);
    let mut signatures = vec![vec![0u64; words]; neurons];
    for (observation, selected) in observations.iter().enumerate() {
        for &neuron in selected {
            signatures[neuron as usize][observation / 64] |= 1u64 << (observation % 64);
        }
    }
    let mut order = (0..neurons).collect::<Vec<_>>();
    order.sort_unstable_by(|left, right| {
        signatures[*right]
            .cmp(&signatures[*left])
            .then_with(|| left.cmp(right))
    });
    let mut groups = vec![0usize; neurons];
    for (rank, neuron) in order.into_iter().enumerate() {
        groups[neuron] = rank / group_neurons;
    }
    groups
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
    let packet_count = metrics
        .ranked_relative_squared_error
        .first()
        .and_then(|layer| layer.first())
        .map_or(0, Vec::len);
    if metrics.selection_semantics != contribution_selection_semantics(packet_count) {
        bail!("down contribution selection semantics do not match packet count");
    }
    for (label, values) in [
        ("down contribution squared-l2", &metrics.squared_l2),
        (
            "down contribution ranked residual",
            &metrics.ranked_relative_squared_error,
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
    if let Some(orders) = &metrics.ranked_packet_indices {
        if orders.len() != header.layers {
            bail!("down contribution rank identities have the wrong layer count");
        }
        for layer in orders {
            if layer.len() != header.top_k {
                bail!("down contribution rank identities have the wrong expert-slot count");
            }
            for order in layer {
                if order.len() != packet_count {
                    bail!("down contribution rank identity length does not match packets");
                }
                let mut seen = vec![false; packet_count];
                for &packet in order {
                    let packet = packet as usize;
                    if packet >= packet_count || seen[packet] {
                        bail!("down contribution rank identities are not a permutation");
                    }
                    seen[packet] = true;
                }
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
        hot_fractions,
        budgets,
        expert_bytes,
        expert_neurons,
        packet_neurons,
        layouts,
        io_bandwidth_mib_s,
        io_fixed_us,
        io_parallelism,
        reset_between_requests,
        measure_phases,
        output,
    } = args;
    let measured_phase_set = measure_phases.iter().cloned().collect::<HashSet<_>>();
    let calibration = calibration.as_deref();
    if policies.is_empty() || budgets.is_empty() || packet_neurons.is_empty() || layouts.is_empty()
    {
        bail!("provide at least one cache policy, budget, packet size, and layout");
    }
    if hot_fractions.is_empty()
        || hot_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || *fraction <= 0.0 || *fraction >= 1.0)
    {
        bail!("hot fractions must be finite and strictly between zero and one");
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
    let tokens = load_simulation_tokens(&trace, &trace_header)?;
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
                if matches!(policy, CachePolicy::StaticHot | CachePolicy::HotLru)
                    && hot_frequency.is_none()
                {
                    bail!("static-hot and hot-lru require a disjoint --calibration trace");
                }
                for &budget in &budgets {
                    let capacity = usize::try_from(budget / packet_bytes)
                        .context("resident packet capacity does not fit usize")?;
                    if capacity == 0 {
                        bail!("budget {budget} is smaller than one neuron packet");
                    }
                    let fractions = match policy {
                        CachePolicy::Lru => vec![None],
                        CachePolicy::StaticHot => vec![Some(1.0)],
                        CachePolicy::HotLru => hot_fractions.iter().copied().map(Some).collect(),
                    };
                    for hot_fraction in fractions {
                        let hot_capacity = hot_fraction
                            .map(|fraction| (capacity as f64 * fraction).floor() as usize)
                            .unwrap_or(0);
                        let hot = hot_frequency.as_ref().map(|frequency| {
                            hottest_packet_keys(
                                frequency,
                                hot_capacity,
                                trace_header.layers,
                                trace_header.experts,
                                packets_per_expert,
                            )
                        });
                        results.push(simulate_cell(SimulationCellConfig {
                            tokens: &tokens,
                            expected_header: &trace_header,
                            policy,
                            hot_fraction: hot_fraction.filter(|fraction| *fraction < 1.0),
                            layout,
                            co_selection_ranks: co_selection_ranks.as_deref(),
                            budget_bytes: budget,
                            capacity,
                            packet_neurons: neurons,
                            packets_per_expert,
                            packet_bytes,
                            io_shape,
                            reset_between_requests,
                            measure_phases: &measured_phase_set,
                            static_hot: hot.as_ref(),
                        })?);
                    }
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
        hot_fractions,
        measure_phases,
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
    tokens: &'a [SimulationToken],
    expected_header: &'a TraceHeader,
    policy: CachePolicy,
    hot_fraction: Option<f64>,
    layout: PhysicalLayout,
    co_selection_ranks: Option<&'a [Vec<u16>]>,
    budget_bytes: u64,
    capacity: usize,
    packet_neurons: usize,
    packets_per_expert: usize,
    packet_bytes: u64,
    io_shape: Option<IoShape>,
    reset_between_requests: bool,
    measure_phases: &'a HashSet<String>,
    static_hot: Option<&'a HashSet<PacketKey>>,
}

struct SimulationToken {
    request_id: String,
    phase: String,
    routes: Vec<Vec<u16>>,
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
        CachePolicy::HotLru => {
            let hot = config
                .static_hot
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing hybrid hot set"))?;
            CacheState::HotLru {
                dynamic: LruCache::new(config.capacity.saturating_sub(hot.len())),
                hot,
            }
        }
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

    for token in config.tokens {
        let measured = config.measure_phases.is_empty()
            || config.measure_phases.contains(token.phase.as_str());
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
        let mut token_hits = 0u64;
        let mut token_misses = 0u64;
        let mut token_compulsory_misses = 0u64;
        let mut token_evictions = 0u64;
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
                        if measured {
                            reuse_distances.push(access_clock.saturating_sub(previous + 1));
                        }
                    }
                    access_clock += 1;
                    let hit = state.access(key, &mut token_evictions);
                    if hit {
                        token_hits += 1;
                    } else {
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
                            token_compulsory_misses += 1;
                        }
                    }
                }
            }
            let run_lengths = contiguous_run_lengths(&mut layer_missing_ordinals);
            let layer_runs = run_lengths.len() as u64;
            token_transfer_runs += layer_runs;
            let layer_missing_bytes = layer_missing_packets.saturating_mul(config.packet_bytes);
            if measured {
                missing_bytes_per_layer.push(layer_missing_bytes);
            }
            let layer_run_bytes = run_lengths
                .into_iter()
                .map(|run_packets| (run_packets as u64).saturating_mul(config.packet_bytes))
                .collect::<Vec<_>>();
            for &run_bytes in &layer_run_bytes {
                if measured {
                    maximum_contiguous_burst_bytes = maximum_contiguous_burst_bytes.max(run_bytes);
                    contiguous_bytes_per_fetch.push(run_bytes);
                }
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
        if measured
            && previous_request.as_deref() == Some(token.request_id.as_str())
            && !previous_keys.is_empty()
        {
            let shared = current_keys.intersection(&previous_keys).count();
            overlap_sum += shared as f64 / current_keys.len().max(1) as f64;
            overlap_count += 1;
        }
        previous_request = Some(token.request_id.clone());
        previous_keys = current_keys;
        if measured {
            hits += token_hits;
            misses += token_misses;
            compulsory_misses += token_compulsory_misses;
            evictions += token_evictions;
            missing_bytes.push(token_misses.saturating_mul(config.packet_bytes));
            transfer_runs.push(token_transfer_runs);
            miss_layers.push(token_miss_layers);
            if config.io_shape.is_some() {
                serial_io_ms.push(token_serial_io_ms);
                ideal_layer_parallel_io_ms.push(token_parallel_io_ms);
                shaped_io_ms.push(token_shaped_io_ms);
            }
            token_records += 1;
        }
    }
    if token_records == 0 {
        bail!("trace contains no token records");
    }
    let route_lookups = hits + misses;
    let total_missing_bytes = misses.saturating_mul(config.packet_bytes);
    Ok(SimulationResult {
        policy: config.policy.label().to_owned(),
        hot_fraction: config.hot_fraction,
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

fn load_simulation_tokens(
    path: &Path,
    expected_header: &TraceHeader,
) -> Result<Vec<SimulationToken>> {
    let mut tokens = Vec::new();
    visit_trace(path, expected_header, |token| {
        tokens.push(SimulationToken {
            request_id: token.request_id,
            phase: token.phase,
            routes: token.routes,
        });
        Ok(())
    })?;
    Ok(tokens)
}

enum CacheState {
    Lru(LruCache),
    Static(HashSet<PacketKey>),
    HotLru {
        hot: HashSet<PacketKey>,
        dynamic: LruCache,
    },
}

impl CacheState {
    fn access(&mut self, key: PacketKey, evictions: &mut u64) -> bool {
        match self {
            Self::Lru(cache) => cache.access(key, evictions),
            Self::Static(keys) => keys.contains(&key),
            Self::HotLru { hot, dynamic } => hot.contains(&key) || dynamic.access(key, evictions),
        }
    }

    fn clear(&mut self) {
        if let Self::Lru(cache) = self {
            cache.clear();
        }
        if let Self::HotLru { dynamic, .. } = self {
            dynamic.clear();
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

fn validate_sample_route(
    routes: &[Vec<u16>],
    layers: usize,
    top_k: usize,
    experts: usize,
    sample_layer: usize,
) -> Result<()> {
    if routes.len() != layers || sample_layer >= routes.len() {
        bail!("partial route record does not match its model shape");
    }
    let ids = &routes[sample_layer];
    if ids.len() != top_k {
        bail!("sample route has {} experts; expected {top_k}", ids.len());
    }
    let mut unique = HashSet::with_capacity(top_k);
    for &expert in ids {
        if expert as usize >= experts || !unique.insert(expert) {
            bail!("sample route contains an invalid or repeated expert {expert}");
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

fn write_expert_down_weights(
    path: &Path,
    neurons: usize,
    hidden: usize,
    weights: &[f32],
) -> Result<()> {
    if weights.len() != neurons * hidden {
        bail!("expert down weight buffer does not match its dimensions");
    }
    let mut writer =
        BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    writer.write_all(EXPERT_DOWN_WEIGHTS_MAGIC)?;
    writer.write_all(&u32::try_from(neurons)?.to_le_bytes())?;
    writer.write_all(&u32::try_from(hidden)?.to_le_bytes())?;
    for &value in weights {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    Ok(())
}

fn read_expert_down_weights(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let mut reader =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != EXPERT_DOWN_WEIGHTS_MAGIC {
        bail!("expert down weights have an invalid magic");
    }
    let mut dimension = [0u8; 4];
    reader.read_exact(&mut dimension)?;
    let neurons = u32::from_le_bytes(dimension) as usize;
    reader.read_exact(&mut dimension)?;
    let hidden = u32::from_le_bytes(dimension) as usize;
    let values = neurons
        .checked_mul(hidden)
        .ok_or_else(|| anyhow::anyhow!("expert down weight dimensions overflow"))?;
    let mut bytes = vec![0u8; values * std::mem::size_of::<f32>()];
    reader.read_exact(&mut bytes)?;
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!("expert down weights contain trailing bytes");
    }
    let weights = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    Ok((neurons, hidden, weights))
}

fn read_joint_expert_student(path: &Path) -> Result<JointExpertStudent> {
    let mut reader =
        BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != JOINT_EXPERT_STUDENT_MAGIC {
        bail!("joint expert student has an invalid magic");
    }
    let mut dimensions = [0u32; 5];
    for value in &mut dimensions {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        *value = u32::from_le_bytes(bytes);
    }
    let [layer, expert, hidden, width, teacher_width] = dimensions.map(|value| value as usize);
    if hidden == 0 || width == 0 || teacher_width == 0 || width > teacher_width {
        bail!("joint expert student has invalid dimensions");
    }
    let matrix_values = hidden
        .checked_mul(width)
        .ok_or_else(|| anyhow::anyhow!("joint expert student dimensions overflow"))?;
    let mut read_matrix = || -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; matrix_values * std::mem::size_of::<f32>()];
        reader.read_exact(&mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect())
    };
    let gate = read_matrix()?;
    let up = read_matrix()?;
    let down = read_matrix()?;
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!("joint expert student contains trailing bytes");
    }
    Ok(JointExpertStudent {
        layer,
        expert: u16::try_from(expert).context("joint student expert exceeds u16")?,
        hidden,
        width,
        teacher_width,
        gate,
        up,
        down,
    })
}

fn read_layer_residual(path: &Path) -> Result<LayerResidual> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"QXLRA001" {
        bail!("layer residual has an invalid magic");
    }
    let mut read_u32 = || -> Result<usize> {
        let mut bytes = [0u8; 4];
        reader.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes) as usize)
    };
    let layer = read_u32()?;
    let rank = read_u32()?;
    let hidden = read_u32()?;
    if rank == 0 || hidden == 0 {
        bail!("layer residual has invalid dimensions");
    }
    let values = rank
        .checked_mul(hidden)
        .ok_or_else(|| anyhow::anyhow!("layer residual dimensions overflow"))?;
    let mut read_values = || -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; values * std::mem::size_of::<f32>()];
        reader.read_exact(&mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect())
    };
    let down = read_values()?;
    let up = read_values()?;
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        bail!("layer residual contains trailing bytes");
    }
    Ok(LayerResidual {
        layer,
        rank,
        hidden,
        down,
        up,
    })
}

#[cfg(target_os = "macos")]
mod llama_capture {
    use super::*;

    unsafe extern "C" {
        fn qx_llama_route_open(
            path: *const c_char,
            context_length: u32,
            layers: c_int,
            experts: c_int,
            top_k: c_int,
            gpu_layers: c_int,
            threads: c_int,
            intermediate: c_int,
            hidden: c_int,
            sample_layer: c_int,
            sample_expert: c_int,
            capture_all_routes: c_int,
            capture_all_activations: c_int,
            contribution_packet_neurons: *const c_int,
            contribution_size_count: c_int,
        ) -> *mut c_void;
        fn qx_llama_route_set_neuron_mask(
            route: *mut c_void,
            mask: *const u8,
            mask_count: c_int,
        ) -> c_int;
        fn qx_llama_route_set_expert_student(
            route: *mut c_void,
            weights: *const f32,
            weight_count: c_int,
        ) -> c_int;
        fn qx_llama_route_set_joint_expert_student(
            route: *mut c_void,
            width: c_int,
            gate_weights: *const f32,
            gate_count: c_int,
            up_weights: *const f32,
            up_count: c_int,
            down_weights: *const f32,
            down_count: c_int,
        ) -> c_int;
        fn qx_llama_route_add_joint_expert_student(
            route: *mut c_void,
            layer: c_int,
            expert: c_int,
            width: c_int,
            gate_weights: *const f32,
            gate_count: c_int,
            up_weights: *const f32,
            up_count: c_int,
            down_weights: *const f32,
            down_count: c_int,
        ) -> c_int;
        fn qx_llama_route_replace_joint_expert_student(
            route: *mut c_void,
            layer: c_int,
            expert: c_int,
            width: c_int,
            gate_weights: *const f32,
            gate_count: c_int,
            up_weights: *const f32,
            up_count: c_int,
            down_weights: *const f32,
            down_count: c_int,
        ) -> c_int;
        fn qx_llama_route_set_joint_active_prefix(route: *mut c_void, count: c_int) -> c_int;
        fn qx_llama_route_set_layer_residual(
            route: *mut c_void,
            layer: c_int,
            rank: c_int,
            down: *const f32,
            down_count: c_int,
            up: *const f32,
            up_count: c_int,
        ) -> c_int;
        fn qx_llama_route_add_expert_student(
            route: *mut c_void,
            layer: c_int,
            expert: c_int,
            weights: *const f32,
            weight_count: c_int,
        ) -> c_int;
        fn qx_llama_route_take_expert_sample(
            route: *mut c_void,
            input: *mut f32,
            input_count: c_int,
            activation: *mut f32,
            activation_count: c_int,
            down_output: *mut f32,
            down_count: c_int,
        ) -> c_int;
        fn qx_llama_route_take_expert_samples(
            route: *mut c_void,
            input: *mut f32,
            input_count: c_int,
            down_outputs: *mut f32,
            down_output_count: c_int,
        ) -> c_int;
        fn qx_llama_route_take_expert_weights(
            route: *mut c_void,
            weights: *mut f32,
            weight_count: c_int,
        ) -> c_int;
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
        fn qx_llama_route_reset(route: *mut c_void) -> c_int;
        fn qx_llama_route_step(
            route: *mut c_void,
            input: c_int,
            target: c_int,
            target_nll: *mut f32,
            output: *mut c_int,
            routes: *mut c_int,
            route_count: c_int,
            activations: *mut f32,
            activation_count: c_int,
            contribution_norms: *mut f32,
            contribution_residuals: *mut f32,
            contribution_order: *mut u16,
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
        sample: Option<(usize, u16)>,
        sample_all_layer: Option<usize>,
        capture_all_routes: bool,
        hidden: usize,
    }

    pub struct StepOutput {
        pub predicted: i32,
        pub target_nll: Option<f32>,
        pub routes: Vec<Vec<u16>>,
        pub activations: Option<Vec<f32>>,
        pub down_contribution: Option<PacketDownContribution>,
        pub expert_sample: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
        pub expert_samples: Vec<(u16, Vec<f32>, Vec<f32>, Vec<f32>)>,
        pub expert_down_weights: Option<Vec<f32>>,
    }

    pub struct OpenConfig<'a> {
        pub context: usize,
        pub layers: usize,
        pub experts: usize,
        pub top_k: usize,
        pub gpu_layers: i32,
        pub threads: i32,
        pub intermediate: Option<usize>,
        pub hidden: usize,
        pub sample: Option<(usize, u16)>,
        pub sample_all_layer: Option<usize>,
        pub capture_all_routes: bool,
        pub capture_all_activations: bool,
        pub contribution_packet_neurons: &'a [usize],
    }

    impl Capture {
        pub fn open(path: &Path, config: OpenConfig<'_>) -> Result<Self> {
            let OpenConfig {
                context,
                layers,
                experts,
                top_k,
                gpu_layers,
                threads,
                intermediate,
                hidden,
                sample,
                sample_all_layer,
                capture_all_routes,
                capture_all_activations,
                contribution_packet_neurons,
            } = config;
            if sample.is_some() && sample_all_layer.is_some() {
                bail!("single-expert and all-expert capture modes are mutually exclusive");
            }
            let sample_layer = sample.map(|(layer, _)| layer).or(sample_all_layer);
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
                    experts.try_into()?,
                    top_k.try_into()?,
                    gpu_layers,
                    threads,
                    intermediate.unwrap_or(0).try_into()?,
                    hidden.try_into()?,
                    sample_layer.map_or(-1, |layer| layer as c_int),
                    sample.map_or(-1, |(_, expert)| c_int::from(expert)),
                    c_int::from(capture_all_routes),
                    c_int::from(capture_all_activations),
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
                sample,
                sample_all_layer,
                capture_all_routes,
                hidden,
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

        pub fn reset(&mut self) -> Result<()> {
            let status = unsafe { qx_llama_route_reset(self.raw) };
            if status != 0 {
                bail!("llama route reset failed ({status}): {}", self.error());
            }
            Ok(())
        }

        pub fn set_neuron_mask(&mut self, mask: &[u8]) -> Result<()> {
            let status = unsafe {
                qx_llama_route_set_neuron_mask(self.raw, mask.as_ptr(), mask.len().try_into()?)
            };
            if status != 0 {
                bail!("llama route mask setup failed ({status}): {}", self.error());
            }
            Ok(())
        }

        pub fn set_expert_student(&mut self, weights: &[f32]) -> Result<()> {
            let count =
                c_int::try_from(weights.len()).context("student weight count exceeds C int")?;
            let status =
                unsafe { qx_llama_route_set_expert_student(self.raw, weights.as_ptr(), count) };
            if status != 0 {
                bail!("llama bridge rejected expert student weights with status {status}");
            }
            Ok(())
        }

        pub fn set_joint_expert_student(&mut self, student: &JointExpertStudent) -> Result<()> {
            let status = unsafe {
                qx_llama_route_set_joint_expert_student(
                    self.raw,
                    student.width.try_into()?,
                    student.gate.as_ptr(),
                    student.gate.len().try_into()?,
                    student.up.as_ptr(),
                    student.up.len().try_into()?,
                    student.down.as_ptr(),
                    student.down.len().try_into()?,
                )
            };
            if status != 0 {
                bail!("llama bridge rejected joint expert student with status {status}");
            }
            Ok(())
        }

        pub fn add_joint_expert_student(&mut self, student: &JointExpertStudent) -> Result<()> {
            let status = unsafe {
                qx_llama_route_add_joint_expert_student(
                    self.raw,
                    student.layer.try_into()?,
                    c_int::from(student.expert),
                    student.width.try_into()?,
                    student.gate.as_ptr(),
                    student.gate.len().try_into()?,
                    student.up.as_ptr(),
                    student.up.len().try_into()?,
                    student.down.as_ptr(),
                    student.down.len().try_into()?,
                )
            };
            if status != 0 {
                bail!("llama bridge rejected joint expert student with status {status}");
            }
            Ok(())
        }

        pub fn replace_joint_expert_student(&mut self, student: &JointExpertStudent) -> Result<()> {
            let status = unsafe {
                qx_llama_route_replace_joint_expert_student(
                    self.raw,
                    student.layer.try_into()?,
                    c_int::from(student.expert),
                    student.width.try_into()?,
                    student.gate.as_ptr(),
                    student.gate.len().try_into()?,
                    student.up.as_ptr(),
                    student.up.len().try_into()?,
                    student.down.as_ptr(),
                    student.down.len().try_into()?,
                )
            };
            if status != 0 {
                bail!("llama bridge rejected joint expert replacement with status {status}");
            }
            Ok(())
        }

        pub fn set_joint_active_prefix(&mut self, count: usize) -> Result<()> {
            let status =
                unsafe { qx_llama_route_set_joint_active_prefix(self.raw, count.try_into()?) };
            if status != 0 {
                bail!("llama bridge rejected joint active prefix with status {status}");
            }
            Ok(())
        }

        pub fn set_layer_residual(&mut self, residual: &LayerResidual) -> Result<()> {
            let status = unsafe {
                qx_llama_route_set_layer_residual(
                    self.raw,
                    residual.layer.try_into()?,
                    residual.rank.try_into()?,
                    residual.down.as_ptr(),
                    residual.down.len().try_into()?,
                    residual.up.as_ptr(),
                    residual.up.len().try_into()?,
                )
            };
            if status != 0 {
                bail!("llama bridge rejected layer residual with status {status}");
            }
            Ok(())
        }

        pub fn add_expert_student(
            &mut self,
            layer: usize,
            expert: u16,
            weights: &[f32],
        ) -> Result<()> {
            let count =
                c_int::try_from(weights.len()).context("student weight count exceeds C int")?;
            let layer = c_int::try_from(layer).context("student layer exceeds C int")?;
            let status = unsafe {
                qx_llama_route_add_expert_student(
                    self.raw,
                    layer,
                    c_int::from(expert),
                    weights.as_ptr(),
                    count,
                )
            };
            if status != 0 {
                bail!("llama bridge rejected expert student with status {status}");
            }
            Ok(())
        }

        pub fn step(&mut self, input: i32) -> Result<StepOutput> {
            self.step_optional_target(input, None)
        }

        pub fn step_target(&mut self, input: i32, target: i32) -> Result<StepOutput> {
            self.step_optional_target(input, Some(target))
        }

        fn step_optional_target(&mut self, input: i32, target: Option<i32>) -> Result<StepOutput> {
            let mut output = 0i32;
            let mut target_nll = 0.0f32;
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
            let mut contribution_order = (!self.contribution_packet_neurons.is_empty())
                .then(|| vec![u16::MAX; metric_count]);
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
                    target.unwrap_or(-1),
                    target
                        .map(|_| &mut target_nll as *mut f32)
                        .unwrap_or(std::ptr::null_mut()),
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
                    contribution_order
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
                        .map(|&value| {
                            if value < 0 && !self.capture_all_routes {
                                Ok(u16::MAX)
                            } else {
                                u16::try_from(value).context("negative expert ID")
                            }
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            let down_contribution = contribution_norms
                .zip(contribution_residuals)
                .zip(contribution_order)
                .zip(contribution_reconstruction)
                .map(|(((norms, residuals), order), reconstruction)| {
                    self.structure_contributions(norms, residuals, order, reconstruction)
                })
                .transpose()?;
            let expert_sample = if self.sample.is_some() {
                let intermediate = self
                    .intermediate
                    .expect("expert sample has activation width");
                let mut input = vec![0.0f32; self.hidden];
                let mut activation = vec![0.0f32; intermediate];
                let mut down_output = vec![0.0f32; self.hidden];
                let status = unsafe {
                    qx_llama_route_take_expert_sample(
                        self.raw,
                        input.as_mut_ptr(),
                        input.len().try_into()?,
                        activation.as_mut_ptr(),
                        activation.len().try_into()?,
                        down_output.as_mut_ptr(),
                        down_output.len().try_into()?,
                    )
                };
                match status {
                    0 => None,
                    1 => Some((input, activation, down_output)),
                    _ => bail!(
                        "llama expert sample read failed ({status}): {}",
                        self.error()
                    ),
                }
            } else {
                None
            };
            let mut expert_samples = Vec::new();
            if let Some(layer) = self.sample_all_layer {
                let intermediate = self
                    .intermediate
                    .expect("expert samples have activation width");
                let mut input = vec![0.0f32; self.hidden];
                let mut down_outputs = vec![0.0f32; self.top_k * self.hidden];
                let status = unsafe {
                    qx_llama_route_take_expert_samples(
                        self.raw,
                        input.as_mut_ptr(),
                        input.len().try_into()?,
                        down_outputs.as_mut_ptr(),
                        down_outputs.len().try_into()?,
                    )
                };
                if status < 0 {
                    bail!(
                        "llama expert sample batch read failed ({status}): {}",
                        self.error()
                    );
                }
                if status > 0 {
                    let count = usize::try_from(status)?;
                    if count != self.top_k {
                        bail!(
                            "llama expert sample batch returned {count} of {} slots",
                            self.top_k
                        );
                    }
                    let all_activations = activations
                        .as_ref()
                        .expect("expert sample batch has activation storage");
                    for slot in 0..count {
                        let activation_start = (layer * self.top_k + slot) * intermediate;
                        let output_start = slot * self.hidden;
                        expert_samples.push((
                            routes[layer][slot],
                            input.clone(),
                            all_activations[activation_start..activation_start + intermediate]
                                .to_vec(),
                            down_outputs[output_start..output_start + self.hidden].to_vec(),
                        ));
                    }
                }
            }
            let expert_down_weights = if self.sample.is_some() {
                let intermediate = self
                    .intermediate
                    .expect("expert sample has activation width");
                let mut weights = vec![0.0f32; self.hidden * intermediate];
                let status = unsafe {
                    qx_llama_route_take_expert_weights(
                        self.raw,
                        weights.as_mut_ptr(),
                        weights.len().try_into()?,
                    )
                };
                match status {
                    0 => None,
                    1 => Some(weights),
                    _ => bail!(
                        "llama expert weight read failed ({status}): {}",
                        self.error()
                    ),
                }
            } else {
                None
            };
            Ok(StepOutput {
                predicted: output,
                target_nll: target.map(|_| target_nll),
                routes,
                activations,
                down_contribution,
                expert_sample,
                expert_samples,
                expert_down_weights,
            })
        }

        fn structure_contributions(
            &self,
            norms: Vec<f32>,
            residuals: Vec<f32>,
            order: Vec<u16>,
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
                        selection_semantics: contribution_selection_semantics(count).to_owned(),
                        squared_l2: vec![vec![vec![0.0; count]; self.top_k]; self.layers],
                        ranked_relative_squared_error: vec![
                            vec![vec![0.0; count]; self.top_k];
                            self.layers
                        ],
                        ranked_packet_indices: Some(vec![
                            vec![vec![u16::MAX; count]; self.top_k];
                            self.layers
                        ]),
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
                        metrics.ranked_relative_squared_error[layer][slot]
                            .copy_from_slice(&residuals[range]);
                        metrics
                            .ranked_packet_indices
                            .as_mut()
                            .expect("new contribution capture includes rank identities")[layer]
                            [slot]
                            .copy_from_slice(
                                &order[expert_base + packet_offset
                                    ..expert_base + packet_offset + count],
                            );
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
    fn corpus_split_treats_whitespace_only_lines_as_boundaries() {
        assert_eq!(
            split_corpus_paragraphs(" first line\nsecond line\n \t\n third \n\n"),
            vec!["first line\nsecond line", "third"]
        );
    }

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
    fn hot_lru_pins_calibrated_packets_and_adapts_the_remainder() {
        let mut cache = CacheState::HotLru {
            hot: HashSet::from([(0, 1, 0)]),
            dynamic: LruCache::new(1),
        };
        let mut evictions = 0;
        assert!(cache.access((0, 1, 0), &mut evictions));
        assert!(!cache.access((0, 2, 0), &mut evictions));
        assert!(cache.access((0, 2, 0), &mut evictions));
        assert!(!cache.access((0, 3, 0), &mut evictions));
        assert!(cache.access((0, 1, 0), &mut evictions));
        assert_eq!(evictions, 1);
    }

    #[test]
    fn phase_filter_uses_prompt_to_warm_generation_cache() {
        let header = TraceHeader {
            schema_version: SCHEMA_VERSION,
            model_sha256: "0".repeat(64),
            llama_revision: "test".to_owned(),
            layers: 1,
            experts: 2,
            top_k: 1,
            source: "synthetic".to_owned(),
        };
        let tokens = [
            SimulationToken {
                request_id: "test".to_owned(),
                phase: "prompt".to_owned(),
                routes: vec![vec![0]],
            },
            SimulationToken {
                request_id: "test".to_owned(),
                phase: "generation".to_owned(),
                routes: vec![vec![0]],
            },
        ];
        let measured = HashSet::from(["generation".to_owned()]);
        let result = simulate_cell(SimulationCellConfig {
            tokens: &tokens,
            expected_header: &header,
            policy: CachePolicy::Lru,
            hot_fraction: None,
            layout: PhysicalLayout::ExpertContiguous,
            co_selection_ranks: None,
            budget_bytes: 100,
            capacity: 1,
            packet_neurons: 1,
            packets_per_expert: 1,
            packet_bytes: 100,
            io_shape: None,
            reset_between_requests: false,
            measure_phases: &measured,
            static_hot: None,
        })
        .unwrap();
        assert_eq!(result.token_records, 1);
        assert_eq!(result.hits, 1);
        assert_eq!(result.misses, 0);
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
    fn corpus_capture_uses_batch_mode_for_zero_or_multiple_experts() {
        assert_eq!(corpus_sample_modes(None, &[]), (None, None));
        assert_eq!(corpus_sample_modes(Some(8), &[]), (None, Some(8)));
        assert_eq!(corpus_sample_modes(Some(8), &[11]), (Some((8, 11)), None));
        assert_eq!(corpus_sample_modes(Some(8), &[11, 38]), (None, Some(8)));
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
    fn signature_packing_keeps_co_selected_neurons_together() {
        let groups = signature_packed_groups(&[vec![0, 1], vec![0, 1], vec![2, 3]], 4, 2);
        assert_eq!(groups[0], groups[1]);
        assert_eq!(groups[2], groups[3]);
        assert_ne!(groups[0], groups[2]);
    }

    #[test]
    fn structured_mask_enforces_the_same_group_budget_for_fallback_experts() {
        let mut calibration = StructuredCalibration::default();
        calibration
            .counts
            .insert((0, 0), vec![3, 3, 0, 0, 1, 1, 0, 0]);
        calibration
            .observations
            .insert((0, 0), vec![vec![0, 1], vec![0, 1, 4, 5]]);
        calibration
            .layer_counts
            .insert(0, vec![2, 2, 0, 0, 1, 1, 0, 0]);
        calibration
            .layer_observations
            .insert(0, vec![vec![0, 1], vec![0, 1, 4, 5]]);

        let mask = build_structured_mask(&calibration, 1, 2, 8, 2, 2).unwrap();
        assert_eq!(mask.len(), 16);
        assert_eq!(
            mask[..8]
                .iter()
                .map(|&value| usize::from(value))
                .sum::<usize>(),
            4
        );
        assert_eq!(
            mask[8..]
                .iter()
                .map(|&value| usize::from(value))
                .sum::<usize>(),
            4
        );

        let all = build_structured_mask(&calibration, 1, 2, 8, 2, 4).unwrap();
        assert!(all.into_iter().all(|value| value == 1));
    }

    #[test]
    fn cholesky_solver_recovers_a_known_solution() {
        let mut matrix = vec![4.0, 1.0, 1.0, 3.0];
        cholesky_in_place(&mut matrix, 2).unwrap();
        let mut values = vec![1.0, 2.0];
        cholesky_solve(&matrix, 2, &mut values);
        assert!((values[0] - 1.0 / 11.0).abs() < 1e-12);
        assert!((values[1] - 7.0 / 11.0).abs() < 1e-12);
    }

    #[test]
    fn expert_down_weight_sidecar_round_trips_and_projects_a_subset() {
        let path = std::env::temp_dir().join(format!(
            "qx-expert-down-{}-{}.qxedw",
            std::process::id(),
            SCHEMA_VERSION
        ));
        let weights = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        write_expert_down_weights(&path, 4, 2, &weights).unwrap();
        let (neurons, hidden, decoded) = read_expert_down_weights(&path).unwrap();
        assert_eq!((neurons, hidden), (4, 2));
        assert_eq!(decoded, weights);
        assert_eq!(
            project_down_subset(&decoded, &[1.0, 10.0, 100.0, 1000.0], &[0, 2], 4, 2),
            vec![301.0, -301.0]
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn joint_expert_student_sidecar_round_trips() {
        let path = std::env::temp_dir().join(format!(
            "qx-joint-expert-student-{}-{}.qxejs",
            std::process::id(),
            SCHEMA_VERSION
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(JOINT_EXPERT_STUDENT_MAGIC).unwrap();
        for dimension in [7_u32, 11, 3, 2, 4] {
            file.write_all(&dimension.to_le_bytes()).unwrap();
        }
        let values: Vec<f32> = (0..18).map(|value| value as f32 * 0.25 - 1.0).collect();
        for value in &values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        drop(file);

        let student = read_joint_expert_student(&path).unwrap();
        assert_eq!(student.layer, 7);
        assert_eq!(student.expert, 11);
        assert_eq!(student.hidden, 3);
        assert_eq!(student.width, 2);
        assert_eq!(student.teacher_width, 4);
        assert_eq!(student.gate, values[0..6]);
        assert_eq!(student.up, values[6..12]);
        assert_eq!(student.down, values[12..18]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn layer_residual_sidecar_round_trips() {
        let path = std::env::temp_dir().join(format!(
            "qx-layer-residual-{}-{}.qxlra",
            std::process::id(),
            SCHEMA_VERSION
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(b"QXLRA001").unwrap();
        for dimension in [7_u32, 2, 3] {
            file.write_all(&dimension.to_le_bytes()).unwrap();
        }
        let values: Vec<f32> = (0..12).map(|value| value as f32 * 0.25 - 1.0).collect();
        for value in &values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        drop(file);

        let residual = read_layer_residual(&path).unwrap();
        assert_eq!(residual.layer, 7);
        assert_eq!(residual.rank, 2);
        assert_eq!(residual.hidden, 3);
        assert_eq!(residual.down, values[0..6]);
        assert_eq!(residual.up, values[6..12]);
        std::fs::remove_file(path).unwrap();
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
                    expert_distill_sample: None,
                    expert_distill_samples: Vec::new(),
                }),
            )
            .unwrap();
        }
        writer.flush().unwrap();
        let tokens = load_simulation_tokens(&path, &header).unwrap();
        let result = simulate_cell(SimulationCellConfig {
            tokens: &tokens,
            expected_header: &header,
            policy: CachePolicy::Lru,
            hot_fraction: None,
            layout: PhysicalLayout::ExpertContiguous,
            co_selection_ranks: None,
            budget_bytes: 200,
            capacity: 2,
            packet_neurons: 1,
            packets_per_expert: 1,
            packet_bytes: 100,
            io_shape: None,
            reset_between_requests: false,
            measure_phases: &HashSet::new(),
            static_hot: None,
        })
        .unwrap();
        assert_eq!(result.token_records, 3);
        assert_eq!(result.hits, 3);
        assert_eq!(result.misses, 3);
        assert_eq!(result.p95_missing_bytes_per_token, 200);
        assert_eq!(result.p99_layers_with_miss_per_token, 2);
        let packetized = simulate_cell(SimulationCellConfig {
            tokens: &tokens,
            expected_header: &header,
            policy: CachePolicy::Lru,
            hot_fraction: None,
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
            measure_phases: &HashSet::new(),
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
