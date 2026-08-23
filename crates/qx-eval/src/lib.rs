use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

const TOKEN_MAGIC: &[u8; 8] = b"QXTOKN\0\0";
const TRACE_MAGIC: &[u8; 8] = b"QXQUAL\0\0";
const VERSION: u32 = 1;
pub const QUALITY_CORPUS_MANIFEST_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityCorpusManifest {
    pub schema_version: u32,
    pub source_revision: String,
    pub split: String,
    pub sampling_seed: u64,
    pub text_tokens_requested: usize,
    pub bootstrap_block_tokens: u32,
    pub tokenizer_hash: String,
    pub entries: Vec<QualityCorpusManifestEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityCorpusManifestEntry {
    pub domain: String,
    pub kind: String,
    pub source_path: String,
    pub output_path: String,
    pub source_split: String,
    pub sampling_strategy: String,
    pub sampling_offset: Option<usize>,
    pub raw_source_sha256: String,
    pub processed_corpus_sha256: String,
    pub records_or_tokens: u64,
    pub scored_positions: Option<u64>,
}

pub fn read_quality_corpus_manifest(path: &Path) -> Result<QualityCorpusManifest> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let manifest = serde_json::from_slice::<QualityCorpusManifest>(&bytes)
        .with_context(|| format!("parse quality corpus manifest {}", path.display()))?;
    if manifest.schema_version != QUALITY_CORPUS_MANIFEST_SCHEMA_VERSION {
        bail!(
            "unsupported quality corpus manifest schema {} (expected {})",
            manifest.schema_version,
            QUALITY_CORPUS_MANIFEST_SCHEMA_VERSION
        );
    }
    if manifest.source_revision.trim().is_empty()
        || manifest.split.trim().is_empty()
        || manifest.tokenizer_hash.trim().is_empty()
        || manifest.bootstrap_block_tokens == 0
    {
        bail!("quality corpus manifest has incomplete provenance metadata");
    }
    if manifest.entries.is_empty() {
        bail!("quality corpus manifest contains no entries");
    }
    Ok(manifest)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenCorpus {
    pub domain: String,
    pub split: String,
    pub source_hash: String,
    pub tokenizer_hash: String,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityBlock {
    pub start_position: u64,
    pub positions: u32,
    pub nll_sum: f64,
}

#[derive(Clone, Debug)]
pub struct LogitSample {
    pub position: u64,
    pub target: u32,
    pub nll: f64,
    pub logits: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct QualityTrace {
    pub vocab: u32,
    pub total_positions: u64,
    pub block_tokens: u32,
    pub sample_stride: u32,
    pub blocks: Vec<QualityBlock>,
    pub samples: Vec<LogitSample>,
}

pub fn write_token_corpus(path: &Path, corpus: &TokenCorpus) -> Result<()> {
    let mut bytes = Vec::with_capacity(64 + corpus.tokens.len() * 4);
    bytes.extend_from_slice(TOKEN_MAGIC);
    put_u32(&mut bytes, VERSION);
    put_string(&mut bytes, &corpus.domain);
    put_string(&mut bytes, &corpus.split);
    put_string(&mut bytes, &corpus.source_hash);
    put_string(&mut bytes, &corpus.tokenizer_hash);
    put_u64(&mut bytes, corpus.tokens.len() as u64);
    for token in &corpus.tokens {
        put_u32(&mut bytes, *token);
    }
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

pub fn read_token_corpus(path: &Path) -> Result<TokenCorpus> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut cursor = Cursor::new(&bytes);
    cursor.expect_magic(TOKEN_MAGIC)?;
    expect_version(cursor.u32()?)?;
    let domain = cursor.string()?;
    let split = cursor.string()?;
    let source_hash = cursor.string()?;
    let tokenizer_hash = cursor.string()?;
    let count = usize::try_from(cursor.u64()?)?;
    let mut tokens = Vec::with_capacity(count);
    for _ in 0..count {
        tokens.push(cursor.u32()?);
    }
    cursor.finish()?;
    Ok(TokenCorpus {
        domain,
        split,
        source_hash,
        tokenizer_hash,
        tokens,
    })
}

pub fn write_quality_trace(path: &Path, trace: &QualityTrace) -> Result<()> {
    if trace.vocab == 0 || trace.sample_stride == 0 {
        bail!("quality trace requires a vocabulary and positive sample stride");
    }
    let sample_bytes = trace.samples.len() * (8 + 4 + 8 + trace.vocab as usize * 4);
    let mut bytes = Vec::with_capacity(48 + trace.blocks.len() * 20 + sample_bytes);
    bytes.extend_from_slice(TRACE_MAGIC);
    put_u32(&mut bytes, VERSION);
    put_u32(&mut bytes, trace.vocab);
    put_u64(&mut bytes, trace.total_positions);
    put_u32(&mut bytes, trace.block_tokens);
    put_u32(&mut bytes, trace.sample_stride);
    put_u32(&mut bytes, trace.blocks.len() as u32);
    put_u32(&mut bytes, trace.samples.len() as u32);
    for block in &trace.blocks {
        put_u64(&mut bytes, block.start_position);
        put_u32(&mut bytes, block.positions);
        put_f64(&mut bytes, block.nll_sum);
    }
    for sample in &trace.samples {
        if sample.logits.len() != trace.vocab as usize {
            bail!("sample {} has the wrong vocabulary", sample.position);
        }
        put_u64(&mut bytes, sample.position);
        put_u32(&mut bytes, sample.target);
        put_f64(&mut bytes, sample.nll);
        for logit in &sample.logits {
            bytes.extend_from_slice(&logit.to_le_bytes());
        }
    }
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

pub fn read_quality_trace(path: &Path) -> Result<QualityTrace> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut cursor = Cursor::new(&bytes);
    cursor.expect_magic(TRACE_MAGIC)?;
    expect_version(cursor.u32()?)?;
    let vocab = cursor.u32()?;
    let total_positions = cursor.u64()?;
    let block_tokens = cursor.u32()?;
    let sample_stride = cursor.u32()?;
    let block_count = cursor.u32()? as usize;
    let sample_count = cursor.u32()? as usize;
    let mut blocks = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        blocks.push(QualityBlock {
            start_position: cursor.u64()?,
            positions: cursor.u32()?,
            nll_sum: cursor.f64()?,
        });
    }
    let mut samples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        let position = cursor.u64()?;
        let target = cursor.u32()?;
        let nll = cursor.f64()?;
        let mut logits = Vec::with_capacity(vocab as usize);
        for _ in 0..vocab {
            logits.push(cursor.f32()?);
        }
        samples.push(LogitSample {
            position,
            target,
            nll,
            logits,
        });
    }
    cursor.finish()?;
    Ok(QualityTrace {
        vocab,
        total_positions,
        block_tokens,
        sample_stride,
        blocks,
        samples,
    })
}

pub fn token_nll(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits
        .iter()
        .map(|value| (*value - max).exp() as f64)
        .sum::<f64>()
        .ln();
    log_sum + max as f64 - logits[target] as f64
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BlockBootstrapEstimate {
    pub relative_ppl_delta: f64,
    pub lower_95: f64,
    pub upper_95: f64,
    pub iterations: usize,
    pub blocks: usize,
    pub block_positions: u32,
}

/// Estimate a relative PPL delta by resampling complete contiguous trace
/// blocks. Blocks are the unit of resampling so positions within a block are
/// not treated as independent observations.
pub fn bootstrap_relative_ppl(
    baseline: &QualityTrace,
    candidate: &QualityTrace,
    iterations: usize,
    seed: u64,
) -> Result<BlockBootstrapEstimate> {
    validate_bootstrap_traces(baseline, candidate)?;
    if iterations < 1_000 {
        bail!("block bootstrap requires at least 1,000 iterations");
    }
    let deltas = baseline
        .blocks
        .iter()
        .zip(&candidate.blocks)
        .map(|(left, right)| (right.nll_sum - left.nll_sum, left.positions))
        .collect::<Vec<_>>();
    let total_delta = deltas.iter().map(|value| value.0).sum::<f64>();
    let total_positions = deltas.iter().map(|value| value.1 as u64).sum::<u64>();
    let mut random = Lcg(seed);
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut delta = 0.0;
        let mut positions = 0_u64;
        for _ in 0..deltas.len() {
            let value = deltas[random.next() as usize % deltas.len()];
            delta += value.0;
            positions += value.1 as u64;
        }
        samples.push((delta / positions as f64).exp() - 1.0);
    }
    samples.sort_by(f64::total_cmp);
    let low = iterations * 25 / 1_000;
    let high = (iterations * 975 / 1_000).min(iterations - 1);
    Ok(BlockBootstrapEstimate {
        relative_ppl_delta: (total_delta / total_positions as f64).exp() - 1.0,
        lower_95: samples[low],
        upper_95: samples[high],
        iterations,
        blocks: deltas.len(),
        block_positions: baseline.block_tokens,
    })
}

fn validate_bootstrap_traces(baseline: &QualityTrace, candidate: &QualityTrace) -> Result<()> {
    if baseline.blocks.is_empty()
        || baseline.blocks.len() != candidate.blocks.len()
        || baseline.total_positions != candidate.total_positions
        || baseline.block_tokens != candidate.block_tokens
    {
        bail!("quality traces do not have aligned non-empty bootstrap blocks");
    }
    if baseline.blocks.iter().any(|block| block.positions == 0)
        || baseline
            .blocks
            .iter()
            .map(|block| block.positions as u64)
            .sum::<u64>()
            != baseline.total_positions
    {
        bail!("quality trace bootstrap blocks do not cover scored positions");
    }
    for (left, right) in baseline.blocks.iter().zip(&candidate.blocks) {
        if left.start_position != right.start_position || left.positions != right.positions {
            bail!("quality traces have different contiguous bootstrap blocks");
        }
    }
    Ok(())
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }
}

fn expect_version(version: u32) -> Result<()> {
    if version != VERSION {
        bail!("unsupported QX evaluation format version {version}");
    }
    Ok(())
}

fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn put_string(bytes: &mut Vec<u8>, value: &str) {
    put_u32(bytes, value.len() as u32);
    bytes.extend_from_slice(value.as_bytes());
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .context("evaluation file overflow")?;
        let value = self
            .bytes
            .get(self.offset..end)
            .context("truncated evaluation file")?;
        self.offset = end;
        Ok(value)
    }

    fn expect_magic(&mut self, magic: &[u8; 8]) -> Result<()> {
        if self.take(8)? != magic {
            bail!("invalid QX evaluation file magic");
        }
        Ok(())
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn string(&mut self) -> Result<String> {
        let count = self.u32()? as usize;
        Ok(String::from_utf8(self.take(count)?.to_vec())?)
    }

    fn finish(&self) -> Result<()> {
        if self.offset != self.bytes.len() {
            bail!("evaluation file has trailing bytes");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nll_is_low_for_target_maximum() {
        assert!(token_nll(&[0.0, 10.0, -2.0], 1) < 0.001);
    }

    #[test]
    fn block_bootstrap_is_reproducible_and_uses_contiguous_blocks() {
        let trace = |values: &[f64]| QualityTrace {
            vocab: 2,
            total_positions: values.len() as u64 * 4,
            block_tokens: 4,
            sample_stride: 4,
            blocks: values
                .iter()
                .enumerate()
                .map(|(index, value)| QualityBlock {
                    start_position: index as u64 * 4,
                    positions: 4,
                    nll_sum: *value,
                })
                .collect(),
            samples: Vec::new(),
        };
        let baseline = trace(&[4.0, 8.0, 12.0]);
        let candidate = trace(&[5.0, 9.0, 13.0]);
        let left = bootstrap_relative_ppl(&baseline, &candidate, 1_000, 7).unwrap();
        let right = bootstrap_relative_ppl(&baseline, &candidate, 1_000, 7).unwrap();
        assert_eq!(left.relative_ppl_delta, right.relative_ppl_delta);
        assert_eq!(left.lower_95, right.lower_95);
        assert_eq!(left.blocks, 3);
        assert!((left.relative_ppl_delta - (0.25_f64).exp_m1()).abs() < 1e-12);
    }
}
