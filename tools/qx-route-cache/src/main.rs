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
const DEFAULT_Q3_EXPERT_BYTES: u64 = 2_064_384;

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
        #[arg(long)]
        reset_between_requests: bool,
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
}

#[derive(Debug, Serialize)]
struct SimulationReport {
    schema_version: u32,
    trace: String,
    calibration: Option<String>,
    trace_header: TraceHeader,
    expert_bytes: u64,
    results: Vec<SimulationResult>,
}

#[derive(Debug, Serialize)]
struct SimulationResult {
    policy: String,
    budget_bytes: u64,
    resident_expert_capacity: usize,
    token_records: usize,
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
    mean_layers_with_miss_per_token: f64,
    p95_layers_with_miss_per_token: u64,
    p99_layers_with_miss_per_token: u64,
    evictions: u64,
    mean_adjacent_route_overlap: f64,
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
            model_sha256,
        }),
        Command::Simulate {
            trace,
            calibration,
            policies,
            budgets,
            expert_bytes,
            reset_between_requests,
            output,
        } => simulate(
            &trace,
            calibration.as_deref(),
            &policies,
            &budgets,
            expert_bytes,
            reset_between_requests,
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
    model_sha256: Option<String>,
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
    let model_hash = match args.model_sha256 {
        Some(value) => {
            validate_sha256(&value)?;
            value
        }
        None => sha256_file(&args.model)?,
    };
    let mut llama = llama_capture::Capture::open(
        &args.model,
        args.context,
        args.layers,
        args.top_k,
        args.gpu_layers,
        args.threads,
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
        let (predicted, routes) = llama.step(token)?;
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
        let (predicted, routes) = llama.step(next)?;
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

fn simulate(
    trace: &Path,
    calibration: Option<&Path>,
    policies: &[CachePolicy],
    budgets: &[u64],
    expert_bytes: u64,
    reset_between_requests: bool,
    output: &Path,
) -> Result<()> {
    if policies.is_empty() || budgets.is_empty() {
        bail!("provide at least one cache policy and budget");
    }
    if expert_bytes == 0 {
        bail!("expert-bytes must be positive");
    }
    let trace_header = read_trace_header(trace)?;
    let hot_frequency = calibration
        .map(|path| collect_frequency(path, &trace_header))
        .transpose()?;
    let mut results = Vec::new();
    for &policy in policies {
        if matches!(policy, CachePolicy::StaticHot) && hot_frequency.is_none() {
            bail!("static-hot requires a disjoint --calibration trace");
        }
        for &budget in budgets {
            let capacity = usize::try_from(budget / expert_bytes)
                .context("resident expert capacity does not fit usize")?;
            if capacity == 0 {
                bail!("budget {budget} is smaller than one expert packet");
            }
            let hot = hot_frequency
                .as_ref()
                .map(|frequency| hottest_keys(frequency, capacity));
            results.push(simulate_cell(SimulationCellConfig {
                trace,
                expected_header: &trace_header,
                policy,
                budget_bytes: budget,
                capacity,
                expert_bytes,
                reset_between_requests,
                static_hot: hot.as_ref(),
            })?);
        }
    }
    let report = SimulationReport {
        schema_version: SCHEMA_VERSION,
        trace: trace.display().to_string(),
        calibration: calibration.map(|path| path.display().to_string()),
        trace_header,
        expert_bytes,
        results,
    };
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("create {}", output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report).context("write simulation report")?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    println!("report: {}", output.display());
    for result in &report.results {
        println!(
            "{} {} bytes: hit {:.3}% p95 miss {:.2} MiB p99 miss layers {}",
            result.policy,
            result.budget_bytes,
            100.0 * result.byte_weighted_hit_rate,
            result.p95_missing_bytes_per_token as f64 / (1024.0 * 1024.0),
            result.p99_layers_with_miss_per_token
        );
    }
    Ok(())
}

type ExpertKey = (u16, u16);

struct SimulationCellConfig<'a> {
    trace: &'a Path,
    expected_header: &'a TraceHeader,
    policy: CachePolicy,
    budget_bytes: u64,
    capacity: usize,
    expert_bytes: u64,
    reset_between_requests: bool,
    static_hot: Option<&'a HashSet<ExpertKey>>,
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
    let mut miss_layers = Vec::new();
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
        }
        let mut current_keys =
            HashSet::with_capacity(config.expected_header.layers * config.expected_header.top_k);
        let mut token_misses = 0u64;
        let mut token_miss_layers = 0u64;
        for (layer, experts) in token.routes.iter().enumerate() {
            let mut layer_missed = false;
            for &expert in experts {
                let key = (layer as u16, expert);
                current_keys.insert(key);
                let hit = state.access(key, &mut evictions);
                if hit {
                    hits += 1;
                } else {
                    misses += 1;
                    token_misses += 1;
                    layer_missed = true;
                    if seen_ever.insert(key) {
                        compulsory_misses += 1;
                    }
                }
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
        missing_bytes.push(token_misses.saturating_mul(config.expert_bytes));
        miss_layers.push(token_miss_layers);
        token_records += 1;
        Ok(())
    })?;
    if token_records == 0 {
        bail!("trace contains no token records");
    }
    let route_lookups = hits + misses;
    let total_missing_bytes = misses.saturating_mul(config.expert_bytes);
    Ok(SimulationResult {
        policy: config.policy.label().to_owned(),
        budget_bytes: config.budget_bytes,
        resident_expert_capacity: config.capacity,
        token_records,
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
        mean_layers_with_miss_per_token: miss_layers.iter().sum::<u64>() as f64
            / token_records as f64,
        p95_layers_with_miss_per_token: percentile(&miss_layers, 0.95),
        p99_layers_with_miss_per_token: percentile(&miss_layers, 0.99),
        evictions,
        mean_adjacent_route_overlap: overlap_sum / overlap_count.max(1) as f64,
    })
}

enum CacheState {
    Lru(LruCache),
    Static(HashSet<ExpertKey>),
}

impl CacheState {
    fn access(&mut self, key: ExpertKey, evictions: &mut u64) -> bool {
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
    resident: HashMap<ExpertKey, u64>,
    oldest: BinaryHeap<Reverse<(u64, ExpertKey)>>,
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

    fn access(&mut self, key: ExpertKey, evictions: &mut u64) -> bool {
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

fn hottest_keys(frequency: &HashMap<ExpertKey, u64>, capacity: usize) -> HashSet<ExpertKey> {
    let mut ranked = frequency
        .iter()
        .map(|(&key, &count)| (Reverse(count), key))
        .collect::<Vec<_>>();
    ranked.sort_unstable();
    ranked
        .into_iter()
        .take(capacity)
        .map(|(_, key)| key)
        .collect()
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
        ) -> c_int;
    }

    pub struct Capture {
        raw: *mut c_void,
        layers: usize,
        top_k: usize,
    }

    impl Capture {
        pub fn open(
            path: &Path,
            context: usize,
            layers: usize,
            top_k: usize,
            gpu_layers: i32,
            threads: i32,
        ) -> Result<Self> {
            let path = CString::new(path.to_string_lossy().as_bytes())?;
            let raw = unsafe {
                qx_llama_route_open(
                    path.as_ptr(),
                    context.try_into()?,
                    layers.try_into()?,
                    top_k.try_into()?,
                    gpu_layers,
                    threads,
                )
            };
            if raw.is_null() {
                bail!("failed to initialize llama.cpp route capture");
            }
            Ok(Self { raw, layers, top_k })
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

        pub fn step(&mut self, input: i32) -> Result<(i32, Vec<Vec<u16>>)> {
            let mut output = 0i32;
            let mut flat = vec![-1i32; self.layers * self.top_k];
            let status = unsafe {
                qx_llama_route_step(
                    self.raw,
                    input,
                    &mut output,
                    flat.as_mut_ptr(),
                    flat.len().try_into()?,
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
            Ok((output, routes))
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
        assert!(!cache.access((0, 1), &mut evictions));
        assert!(!cache.access((0, 2), &mut evictions));
        assert!(cache.access((0, 1), &mut evictions));
        assert!(!cache.access((0, 3), &mut evictions));
        assert!(cache.access((0, 1), &mut evictions));
        assert!(!cache.access((0, 2), &mut evictions));
        assert_eq!(evictions, 2);
    }

    #[test]
    fn hottest_keys_are_selected_by_frequency() {
        let frequency = HashMap::from([((0, 1), 4), ((0, 2), 9), ((1, 0), 6)]);
        let hot = hottest_keys(&frequency, 2);
        assert_eq!(hot, HashSet::from([(0, 2), (1, 0)]));
    }

    #[test]
    fn percentile_uses_an_upper_rank() {
        assert_eq!(percentile(&[0, 1, 2, 3, 4], 0.50), 2);
        assert_eq!(percentile(&[0, 1, 2, 3, 4], 0.95), 4);
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
                }),
            )
            .unwrap();
        }
        writer.flush().unwrap();
        let result = simulate_cell(SimulationCellConfig {
            trace: &path,
            expected_header: &header,
            policy: CachePolicy::Lru,
            budget_bytes: 200,
            capacity: 2,
            expert_bytes: 100,
            reset_between_requests: false,
            static_hot: None,
        })
        .unwrap();
        assert_eq!(result.token_records, 3);
        assert_eq!(result.hits, 3);
        assert_eq!(result.misses, 3);
        assert_eq!(result.p95_missing_bytes_per_token, 200);
        assert_eq!(result.p99_layers_with_miss_per_token, 2);
        std::fs::remove_file(path).unwrap();
    }
}
