use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use qx_runtime::GraniteModel;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "qx-generate",
    about = "Greedy batch-1 Granite generation through the QX runtime"
)]
struct Args {
    /// Hugging Face Granite model directory.
    #[arg(long)]
    model: PathBuf,
    /// QX artifact containing the quantized MoE expert tensors.
    #[arg(long)]
    artifact: PathBuf,
    /// Prompt text to tokenize and complete.
    #[arg(short, long, required_unless_present = "interactive")]
    prompt: Option<String>,
    /// Keep the model loaded and generate one response per stdin line.
    #[arg(long)]
    interactive: bool,
    /// Number of new tokens to generate.
    #[arg(short = 'n', long, default_value_t = 32)]
    max_new_tokens: usize,
    /// Maximum sequence length for the KV caches.
    #[arg(long, default_value_t = 512)]
    context: usize,
    /// Runtime implementation. V2 is the canonical artifact-only Metal path.
    #[arg(long, value_enum, default_value_t = RuntimeKind::V2)]
    runtime: RuntimeKind,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum RuntimeKind {
    V2,
    V1,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.model.is_dir() {
        bail!("model directory does not exist: {}", args.model.display());
    }
    if !args.artifact.is_file() {
        bail!("QX artifact does not exist: {}", args.artifact.display());
    }
    if cfg!(debug_assertions) {
        eprintln!("warning: qx-generate is a debug build; use --release for performance");
    }
    match args.runtime {
        RuntimeKind::V2 => run_v2(&args),
        RuntimeKind::V1 => run_v1(&args),
    }
}

fn run_v1(args: &Args) -> Result<()> {
    let mut model = GraniteModel::from_paths(&args.model, &args.artifact, args.context)
        .map_err(anyhow::Error::msg)
        .with_context(|| "load Granite checkpoint and QX artifact")?;
    if args.interactive {
        let stdin = io::stdin();
        let mut stdout = io::stdout().lock();
        for line in stdin.lock().lines() {
            let prompt = line.context("read prompt from stdin")?;
            if prompt.trim().is_empty() {
                continue;
            }
            let output = model
                .generate(&prompt, args.max_new_tokens)
                .map_err(anyhow::Error::msg)
                .with_context(|| "run greedy generation")?;
            writeln!(stdout, "{output}").context("write generated output")?;
            stdout.flush().context("flush generated output")?;
        }
    } else {
        let prompt = args.prompt.as_deref().expect("clap requires --prompt");
        let output = model
            .generate(prompt, args.max_new_tokens)
            .map_err(anyhow::Error::msg)
            .with_context(|| "run greedy generation")?;
        println!("{output}");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_v2(args: &Args) -> Result<()> {
    let tokenizer =
        qx_tokenizer::Tokenizer::from_dir(&args.model).with_context(|| "load Granite tokenizer")?;
    let mut generator = qx_runtime_v2::GraniteV2Generator::load(&args.artifact, args.context)
        .map_err(anyhow::Error::msg)
        .with_context(|| "load canonical V2 artifact runtime")?;
    let mut generate = |prompt: &str| -> Result<String> {
        let prompt_tokens = tokenizer.encode(prompt)?;
        let generated = generator
            .generate_tokens(&prompt_tokens, args.max_new_tokens, tokenizer.eos_id())
            .map_err(anyhow::Error::msg)?;
        let mut all_tokens = prompt_tokens;
        all_tokens.extend_from_slice(&generated);
        tokenizer.decode(&all_tokens)
    };
    if args.interactive {
        let stdin = io::stdin();
        let mut stdout = io::stdout().lock();
        for line in stdin.lock().lines() {
            let prompt = line.context("read prompt from stdin")?;
            if prompt.trim().is_empty() {
                continue;
            }
            writeln!(stdout, "{}", generate(&prompt)?).context("write generated output")?;
            stdout.flush().context("flush generated output")?;
        }
    } else {
        let prompt = args.prompt.as_deref().expect("clap requires --prompt");
        println!("{}", generate(prompt)?);
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn run_v2(_args: &Args) -> Result<()> {
    bail!("Runtime V2 generation requires macOS Metal; use --runtime v1 for the reference path")
}
