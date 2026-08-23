use anyhow::{Context, Result, bail};
use clap::Parser;
use qx_eval::{
    QUALITY_CORPUS_MANIFEST_SCHEMA_VERSION, QualityCorpusManifest, QualityCorpusManifestEntry,
    TokenCorpus, write_token_corpus,
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Parser)]
#[command(
    name = "qx-quality-corpus",
    about = "Build reproducible triangular-quality corpus files and a sealed manifest"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    wikitext: PathBuf,
    #[arg(long)]
    dolly: PathBuf,
    #[arg(long)]
    humaneval: PathBuf,
    #[arg(long)]
    gsm8k: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    source_revision: String,
    #[arg(long, default_value = "heldout")]
    split: String,
    #[arg(long, default_value_t = 0x51_58_51_43_4f_52_50_u64)]
    sampling_seed: u64,
    #[arg(long, default_value_t = 32_769)]
    text_tokens: usize,
    #[arg(long, default_value_t = 512)]
    bootstrap_block_tokens: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.text_tokens < 2 {
        bail!("--text-tokens must be at least 2");
    }
    if args.bootstrap_block_tokens == 0 {
        bail!("--bootstrap-block-tokens must be positive");
    }
    if args.bootstrap_block_tokens as usize >= args.text_tokens {
        bail!(
            "--bootstrap-block-tokens must be smaller than --text-tokens so held-out traces contain multiple positions"
        );
    }
    if args.split.trim().is_empty() || args.source_revision.trim().is_empty() {
        bail!("--split and --source-revision must be non-empty");
    }
    let missing = [
        ("model tokenizer directory", args.model.clone()),
        ("WikiText source", args.wikitext.clone()),
        ("Dolly source", args.dolly.clone()),
        ("HumanEval source", args.humaneval.clone()),
        ("GSM8K source", args.gsm8k.clone()),
    ]
    .into_iter()
    .filter(|(_, path)| !path.exists())
    .map(|(kind, path)| format!("{kind}: {}", path.display()))
    .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "quality corpus inputs unavailable:\n  {}",
            missing.join("\n  ")
        );
    }
    fs::create_dir_all(&args.output)
        .with_context(|| format!("create output directory {}", args.output.display()))?;

    let tokenizer = qx_tokenizer::Tokenizer::from_dir(&args.model)?;
    let tokenizer_hash = tokenizer_hash(&args.model)?;
    let mut entries = Vec::with_capacity(4);

    entries.push(build_text_entry(
        "wikitext-2",
        &args.wikitext,
        &args.output.join("wikitext.tokens"),
        &args.split,
        args.sampling_seed,
        args.text_tokens,
        &tokenizer,
        &tokenizer_hash,
    )?);
    entries.push(build_text_entry(
        "dolly",
        &args.dolly,
        &args.output.join("dolly.tokens"),
        &args.split,
        args.sampling_seed,
        args.text_tokens,
        &tokenizer,
        &tokenizer_hash,
    )?);
    entries.push(build_jsonl_entry(
        "humaneval",
        &args.humaneval,
        &args.output.join("humaneval.jsonl"),
        &args.split,
    )?);
    entries.push(build_jsonl_entry(
        "gsm8k",
        &args.gsm8k,
        &args.output.join("gsm8k.jsonl"),
        &args.split,
    )?);

    let manifest = QualityCorpusManifest {
        schema_version: QUALITY_CORPUS_MANIFEST_SCHEMA_VERSION,
        source_revision: args.source_revision,
        split: args.split,
        sampling_seed: args.sampling_seed,
        text_tokens_requested: args.text_tokens,
        bootstrap_block_tokens: args.bootstrap_block_tokens,
        tokenizer_hash,
        entries,
    };
    let manifest_path = args.output.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    println!(
        "wrote reproducible quality manifest: {}",
        manifest_path.display()
    );
    Ok(())
}

fn build_text_entry(
    domain: &str,
    source: &Path,
    output: &Path,
    split: &str,
    seed: u64,
    requested_tokens: usize,
    tokenizer: &qx_tokenizer::Tokenizer,
    tokenizer_hash: &str,
) -> Result<QualityCorpusManifestEntry> {
    let raw = read_required(source, "text corpus")?;
    let text = String::from_utf8(raw.clone())
        .with_context(|| format!("decode {} as UTF-8", source.display()))?;
    let tokens = tokenizer.encode(&text)?;
    if tokens.len() < requested_tokens {
        bail!(
            "{} contains {} tokens, fewer than requested {}",
            source.display(),
            tokens.len(),
            requested_tokens
        );
    }
    let span = tokens.len() - requested_tokens + 1;
    let offset = (seed % span as u64) as usize;
    let selected = tokens[offset..offset + requested_tokens].to_vec();
    let corpus = TokenCorpus {
        domain: domain.to_owned(),
        split: split.to_owned(),
        source_hash: sha256(&raw),
        tokenizer_hash: tokenizer_hash.to_owned(),
        tokens: selected,
    };
    write_token_corpus(output, &corpus)?;
    Ok(QualityCorpusManifestEntry {
        domain: domain.to_owned(),
        kind: "token-corpus".to_owned(),
        source_path: source.display().to_string(),
        output_path: output.display().to_string(),
        source_split: split.to_owned(),
        sampling_strategy: "deterministic-contiguous-token-window".to_owned(),
        sampling_offset: Some(offset),
        raw_source_sha256: sha256(&raw),
        processed_corpus_sha256: sha256_file(output)?,
        records_or_tokens: requested_tokens as u64,
        scored_positions: Some((requested_tokens - 1) as u64),
    })
}

fn build_jsonl_entry(
    domain: &str,
    source: &Path,
    output: &Path,
    split: &str,
) -> Result<QualityCorpusManifestEntry> {
    let raw = read_required(source, "task corpus")?;
    let text = String::from_utf8(raw.clone())
        .with_context(|| format!("decode {} as UTF-8", source.display()))?;
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut records = 0_u64;
    for (line_number, line) in normalized.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        serde_json::from_str::<serde_json::Value>(line).with_context(|| {
            format!(
                "parse {} JSONL record {}",
                source.display(),
                line_number + 1
            )
        })?;
        records += 1;
    }
    if records == 0 {
        bail!("{} contains no JSONL records", source.display());
    }
    fs::write(output, normalized.as_bytes())
        .with_context(|| format!("write {}", output.display()))?;
    Ok(QualityCorpusManifestEntry {
        domain: domain.to_owned(),
        kind: "task-jsonl".to_owned(),
        source_path: source.display().to_string(),
        output_path: output.display().to_string(),
        source_split: split.to_owned(),
        sampling_strategy: "canonical-jsonl-order".to_owned(),
        sampling_offset: None,
        raw_source_sha256: sha256(&raw),
        processed_corpus_sha256: sha256_file(output)?,
        records_or_tokens: records,
        scored_positions: None,
    })
}

fn tokenizer_hash(model: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ] {
        digest.update(name.as_bytes());
        let path = model.join(name);
        let bytes = read_required(&path, "tokenizer input")?;
        digest.update(bytes);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn read_required(path: &Path, kind: &str) -> Result<Vec<u8>> {
    if !path.exists() {
        bail!(
            "quality corpus input unavailable: missing {kind} {}",
            path.display()
        );
    }
    fs::read(path).with_context(|| format!("read {kind} {}", path.display()))
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256(&fs::read(path)?))
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
