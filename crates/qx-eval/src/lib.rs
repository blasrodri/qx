use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

const TOKEN_MAGIC: &[u8; 8] = b"QXTOKN\0\0";
const TRACE_MAGIC: &[u8; 8] = b"QXQUAL\0\0";
const VERSION: u32 = 1;

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
}
