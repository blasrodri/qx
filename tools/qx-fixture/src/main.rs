use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "qx-fixture",
    about = "Create a fixed token-ID QX benchmark fixture"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    name: String,
    #[arg(long)]
    domain: String,
    #[arg(long, conflicts_with = "text_file")]
    text: Option<String>,
    #[arg(long, conflicts_with = "text")]
    text_file: Option<PathBuf>,
    #[arg(long, default_value_t = 384)]
    prefix_tokens: usize,
    #[arg(long, default_value_t = 512)]
    context: usize,
    #[arg(long, default_value_t = 128)]
    generation: usize,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Serialize)]
struct Fixture<'a> {
    name: &'a str,
    domain: &'a str,
    model: &'static str,
    target: &'static str,
    source_hash: String,
    tokenizer_hash: String,
    tokens: Vec<u32>,
    context: usize,
    generation: usize,
    batch: usize,
    sampling: &'static str,
    measurement: &'static str,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.prefix_tokens == 0 || args.generation == 0 {
        bail!("prefix and generation must be positive");
    }
    if args.prefix_tokens.saturating_add(args.generation) > args.context {
        bail!("prefix plus generation exceeds context");
    }
    let text = match (&args.text, &args.text_file) {
        (Some(text), None) => text.clone(),
        (None, Some(path)) => {
            fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?
        }
        _ => bail!("provide exactly one of --text or --text-file"),
    };
    let tokenizer = qx_tokenizer::Tokenizer::from_dir(&args.model)?;
    let mut tokens = tokenizer.encode(&text)?;
    if tokens.len() < args.prefix_tokens {
        bail!(
            "source produced {} tokens, fewer than requested prefix {}",
            tokens.len(),
            args.prefix_tokens
        );
    }
    tokens.truncate(args.prefix_tokens);
    let fixture = Fixture {
        name: &args.name,
        domain: &args.domain,
        model: "ibm-granite-3.0-1b-a400m-instruct",
        target: "apple-m5-pro",
        source_hash: format!("fnv1a64:{:016x}", hash_bytes(text.as_bytes())),
        tokenizer_hash: format!("fnv1a64:{:016x}", hash_tokenizer(&args.model)?),
        tokens,
        context: args.context,
        generation: args.generation,
        batch: 1,
        sampling: "greedy",
        measurement: "post_prefill",
    };
    fs::write(&args.output, serde_json::to_vec_pretty(&fixture)?)
        .with_context(|| format!("write {}", args.output.display()))?;
    println!("wrote {}", args.output.display());
    Ok(())
}

fn hash_tokenizer(model: &std::path::Path) -> Result<u64> {
    let mut hash = 0xcbf29ce484222325_u64;
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ] {
        let path = model.join(name);
        if path.exists() {
            hash = hash_bytes_seeded(hash, name.as_bytes());
            hash = hash_bytes_seeded(hash, &fs::read(&path)?);
        }
    }
    Ok(hash)
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    hash_bytes_seeded(0xcbf29ce484222325_u64, bytes)
}

fn hash_bytes_seeded(seed: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(seed, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}
