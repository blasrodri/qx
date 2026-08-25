//! Minimal Hugging Face safetensors index/shard reader for the V0 inspector.
//!
//! Shards are memory mapped and only the requested tensor is decoded. The
//! returned `Vec<f32>` is deliberate: the quantizer is the first consumer and
//! owns the conversion boundary.

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: String,
    pub shard: PathBuf,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub byte_len: usize,
}

#[derive(Debug, Deserialize)]
struct IndexFile {
    weight_map: BTreeMap<String, String>,
}

pub struct Checkpoint {
    root: PathBuf,
    weight_map: BTreeMap<String, String>,
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        if !root.is_dir() {
            bail!("model path is not a directory: {}", root.display());
        }
        let index_path = root.join("model.safetensors.index.json");
        let weight_map = if index_path.exists() {
            let bytes =
                fs::read(&index_path).with_context(|| format!("read {}", index_path.display()))?;
            serde_json::from_slice::<IndexFile>(&bytes)
                .with_context(|| format!("parse {}", index_path.display()))?
                .weight_map
        } else {
            let shard = root.join("model.safetensors");
            if !shard.exists() {
                bail!(
                    "no model.safetensors.index.json or model.safetensors in {}",
                    root.display()
                );
            }
            let names = tensor_names(&shard)?;
            names
                .into_iter()
                .map(|name| {
                    (
                        name,
                        shard.file_name().unwrap().to_string_lossy().into_owned(),
                    )
                })
                .collect()
        };
        Ok(Self { root, weight_map })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    /// Inspect every tensor while mapping each shard only once. This is the
    /// release-preflight path for large sharded checkpoints; repeatedly
    /// calling `info` would deserialize a 3--4 GiB shard for every tensor.
    pub fn tensor_infos(&self) -> Result<Vec<TensorInfo>> {
        let mut by_shard = BTreeMap::<&str, Vec<&str>>::new();
        for (name, shard) in &self.weight_map {
            by_shard.entry(shard).or_default().push(name);
        }
        let mut infos = Vec::with_capacity(self.weight_map.len());
        for (shard_name, names) in by_shard {
            let shard = self.root.join(shard_name);
            let mmap = map_file(&shard)?;
            let tensors = SafeTensors::deserialize(&mmap)
                .with_context(|| format!("parse {}", shard.display()))?;
            for name in names {
                let tensor = tensors
                    .tensor(name)
                    .with_context(|| format!("tensor {name} missing from {}", shard.display()))?;
                infos.push(TensorInfo {
                    name: name.to_owned(),
                    shard: shard.clone(),
                    dtype: format!("{:?}", tensor.dtype()),
                    shape: tensor.shape().to_vec(),
                    byte_len: tensor.data().len(),
                });
            }
        }
        infos.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(infos)
    }

    pub fn info(&self, name: &str) -> Result<TensorInfo> {
        let shard_name = self
            .weight_map
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;
        let shard = self.root.join(shard_name);
        let mmap = map_file(&shard)?;
        let tensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("parse {}", shard.display()))?;
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("tensor {name} missing from {}", shard.display()))?;
        Ok(TensorInfo {
            name: name.to_owned(),
            shard,
            dtype: format!("{:?}", tensor.dtype()),
            shape: tensor.shape().to_vec(),
            byte_len: tensor.data().len(),
        })
    }

    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let shard_name = self
            .weight_map
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;
        let shard = self.root.join(shard_name);
        let mmap = map_file(&shard)?;
        let tensors = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("parse {}", shard.display()))?;
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("tensor {name} missing from {}", shard.display()))?;
        let values = decode_f32(tensor.dtype(), tensor.data())?;
        Ok((tensor.shape().to_vec(), values))
    }
}

fn map_file(path: &Path) -> Result<Mmap> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: the file descriptor remains open for the duration of the map
    // construction, and Mmap owns the mapping after this call returns.
    unsafe { Mmap::map(&file).with_context(|| format!("mmap {}", path.display())) }
}

fn tensor_names(path: &Path) -> Result<Vec<String>> {
    let mmap = map_file(path)?;
    let tensors =
        SafeTensors::deserialize(&mmap).with_context(|| format!("parse {}", path.display()))?;
    Ok(tensors
        .names()
        .iter()
        .map(|name| (*name).to_owned())
        .collect())
}

fn decode_f32(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>> {
    let values = match dtype {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        Dtype::F64 => data
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect(),
        other => bail!("tensor dtype {other:?} is not supported by qx-inspect"),
    };
    Ok(values)
}
