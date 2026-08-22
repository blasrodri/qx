//! Fixed-layout metadata and weight primitives for the memory-mappable `.qx`
//! artifact.

use anyhow::{Result, bail};
use half::f16;
use memmap2::{Mmap, UncheckedAdvice};
use qx_quant::{PackingLayout, QuantType, QuantizedTensor};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const MAGIC: [u8; 8] = *b"QXMODL\0\0";
pub const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TensorKind {
    GateUp = 0,
    Down = 1,
    QProj = 2,
    KProj = 3,
    VProj = 4,
    OProj = 5,
    Router = 6,
    Embedding = 7,
    InputNorm = 8,
    PostAttentionNorm = 9,
    FinalNorm = 10,
}

impl TensorKind {
    fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::GateUp),
            1 => Ok(Self::Down),
            2 => Ok(Self::QProj),
            3 => Ok(Self::KProj),
            4 => Ok(Self::VProj),
            5 => Ok(Self::OProj),
            6 => Ok(Self::Router),
            7 => Ok(Self::Embedding),
            8 => Ok(Self::InputNorm),
            9 => Ok(Self::PostAttentionNorm),
            10 => Ok(Self::FinalNorm),
            _ => bail!("unknown QX tensor kind {value}"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QxHeader {
    pub model_id: String,
    pub target_id: String,
    pub context_limit: u32,
    pub directory_offset: u64,
    pub directory_entries: u32,
    pub memory_offset: u64,
    pub memory_length: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorDirectoryEntry {
    pub layer: u16,
    pub expert: u8,
    pub quant_type: QuantType,
    pub layout: PackingLayout,
    pub kind: TensorKind,
    pub rows: u32,
    pub cols: u32,
    pub group_size: u16,
    pub scale_offset: u64,
    pub scale_length: u64,
    pub data_offset: u64,
    pub data_length: u64,
}

impl TensorDirectoryEntry {
    pub const fn is_expert(&self) -> bool {
        self.expert != u8::MAX
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryPlan {
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub hidden_a_bytes: u64,
    pub hidden_b_bytes: u64,
    pub attention_scratch_bytes: u64,
    pub moe_scratch_bytes: u64,
    pub logits_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct CompiledTensor {
    pub layer: u16,
    pub expert: u8,
    pub kind: TensorKind,
    pub tensor: QuantizedTensor,
}

#[derive(Clone, Debug)]
pub struct RawTensor {
    pub layer: u16,
    pub expert: u8,
    pub kind: TensorKind,
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct QxArtifactInfo {
    pub header: QxHeader,
    pub entries: Vec<TensorDirectoryEntry>,
    pub memory_plan: MemoryPlan,
}

pub struct MappedArtifact {
    path: PathBuf,
    mmap: Mmap,
    info: QxArtifactInfo,
}

impl MappedArtifact {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = fs::File::open(&path)?;
        // SAFETY: Mmap owns the mapping and the file remains logically
        // immutable for the lifetime of this read-only artifact view.
        let mmap = unsafe { Mmap::map(&file)? };
        let info = read_artifact_bytes(&mmap)?;
        for entry in &info.entries {
            let scale_end = entry.scale_offset.saturating_add(entry.scale_length);
            let data_end = entry.data_offset.saturating_add(entry.data_length);
            if scale_end > mmap.len() as u64 || data_end > mmap.len() as u64 {
                bail!("QX tensor directory points outside artifact");
            }
        }
        Ok(Self { path, mmap, info })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn info(&self) -> &QxArtifactInfo {
        &self.info
    }

    /// Ask the OS to reclaim file-backed pages already copied into Metal
    /// buffers. The mapping remains valid and will fault pages back in if a
    /// later CPU inspection needs them.
    pub fn discard_resident_pages(&self) -> Result<()> {
        #[cfg(unix)]
        // SAFETY: this is a read-only file mapping and no borrowed slice is
        // retained across the advisory call.
        unsafe {
            self.mmap.unchecked_advise(UncheckedAdvice::DontNeed)?;
        }
        Ok(())
    }

    pub fn tensor_bytes(&self, index: usize) -> Result<(&[u8], &[u8])> {
        let entry = self
            .info
            .entries
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("QX tensor index out of range"))?;
        let scales = &self.mmap
            [entry.scale_offset as usize..(entry.scale_offset + entry.scale_length) as usize];
        let data = &self.mmap
            [entry.data_offset as usize..(entry.data_offset + entry.data_length) as usize];
        Ok((scales, data))
    }

    pub fn quantized_tensor(&self, index: usize) -> Result<QuantizedTensor> {
        let entry = self
            .info
            .entries
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("QX tensor index out of range"))?;
        let (scales, data) = self.tensor_bytes(index)?;
        if scales.len() % 2 != 0 {
            bail!("QX scale section is not FP16 aligned");
        }
        let scale_values = scales
            .chunks_exact(2)
            .map(|value| f16::from_bits(u16::from_le_bytes(value.try_into().unwrap())))
            .collect();
        Ok(QuantizedTensor {
            quant_type: entry.quant_type,
            layout: entry.layout,
            rows: entry.rows as usize,
            cols: entry.cols as usize,
            group_size: entry.group_size as usize,
            scales: scale_values,
            data: data.to_vec(),
        })
    }

    pub fn raw_tensor_f32(&self, index: usize) -> Result<Vec<f32>> {
        let entry = self
            .info
            .entries
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("QX tensor index out of range"))?;
        if !matches!(
            entry.kind,
            TensorKind::InputNorm | TensorKind::PostAttentionNorm | TensorKind::FinalNorm
        ) || entry.scale_length != 0
        {
            bail!("QX tensor is not a raw FP32 tensor");
        }
        let data = &self.mmap
            [entry.data_offset as usize..(entry.data_offset + entry.data_length) as usize];
        if !data.len().is_multiple_of(4)
            || data.len() / 4 != entry.rows as usize * entry.cols as usize
        {
            bail!("QX raw FP32 tensor has invalid dimensions");
        }
        Ok(data
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect())
    }
}

pub fn write_artifact(
    path: impl AsRef<Path>,
    model_id: &str,
    target_id: &str,
    context_limit: u32,
    memory_plan: MemoryPlan,
    tensors: &[CompiledTensor],
) -> Result<()> {
    write_artifact_internal(
        path,
        model_id,
        target_id,
        context_limit,
        memory_plan,
        tensors,
        &[],
    )
}

pub fn write_artifact_with_raw(
    path: impl AsRef<Path>,
    model_id: &str,
    target_id: &str,
    context_limit: u32,
    memory_plan: MemoryPlan,
    tensors: &[CompiledTensor],
    raw_tensors: &[RawTensor],
) -> Result<()> {
    write_artifact_internal(
        path,
        model_id,
        target_id,
        context_limit,
        memory_plan,
        tensors,
        raw_tensors,
    )
}

fn write_artifact_internal(
    path: impl AsRef<Path>,
    model_id: &str,
    target_id: &str,
    context_limit: u32,
    memory_plan: MemoryPlan,
    tensors: &[CompiledTensor],
    raw_tensors: &[RawTensor],
) -> Result<()> {
    let model = model_id.as_bytes();
    let target = target_id.as_bytes();
    if model.len() > u16::MAX as usize || target.len() > u16::MAX as usize {
        bail!("QX model or target id is too long");
    }
    let fixed_header = 52_u64;
    let directory_offset = fixed_header + model.len() as u64 + target.len() as u64;
    let directory_size = (tensors.len() + raw_tensors.len()) as u64 * 48;
    let memory_offset = directory_offset + directory_size;
    let memory_length = 64_u64;
    let data_offset = memory_offset + memory_length;
    let mut entries = Vec::with_capacity(tensors.len());
    let mut blobs = Vec::new();
    for compiled in tensors {
        let scale_offset = data_offset + blobs.len() as u64;
        for scale in &compiled.tensor.scales {
            blobs.extend_from_slice(&scale.to_bits().to_le_bytes());
        }
        let scale_length = blobs.len() as u64 - (scale_offset - data_offset);
        let tensor_data_offset = data_offset + blobs.len() as u64;
        blobs.extend_from_slice(&compiled.tensor.data);
        entries.push(TensorDirectoryEntry {
            layer: compiled.layer,
            expert: compiled.expert,
            quant_type: compiled.tensor.quant_type,
            layout: compiled.tensor.layout,
            kind: compiled.kind,
            rows: compiled.tensor.rows as u32,
            cols: compiled.tensor.cols as u32,
            group_size: compiled.tensor.group_size as u16,
            scale_offset,
            scale_length,
            data_offset: tensor_data_offset,
            data_length: compiled.tensor.data.len() as u64,
        });
    }
    for raw in raw_tensors {
        if raw.values.len() != raw.rows.saturating_mul(raw.cols) {
            bail!("raw tensor dimensions do not match values");
        }
        let blob_start = blobs.len();
        let data_offset = data_offset + blobs.len() as u64;
        for value in &raw.values {
            blobs.extend_from_slice(&value.to_le_bytes());
        }
        entries.push(TensorDirectoryEntry {
            layer: raw.layer,
            expert: raw.expert,
            quant_type: QuantType::Q4,
            layout: PackingLayout::RowMajor,
            kind: raw.kind,
            rows: raw.rows as u32,
            cols: raw.cols as u32,
            group_size: 0,
            scale_offset: data_offset,
            scale_length: 0,
            data_offset,
            data_length: (blobs.len() - blob_start) as u64,
        });
    }
    let mut output = Vec::with_capacity(data_offset as usize + blobs.len());
    output.extend_from_slice(&MAGIC);
    output.extend_from_slice(&VERSION.to_le_bytes());
    output.extend_from_slice(&(model.len() as u16).to_le_bytes());
    output.extend_from_slice(&(target.len() as u16).to_le_bytes());
    output.extend_from_slice(&context_limit.to_le_bytes());
    output.extend_from_slice(&directory_offset.to_le_bytes());
    output.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    output.extend_from_slice(&memory_offset.to_le_bytes());
    output.extend_from_slice(&memory_length.to_le_bytes());
    output.extend_from_slice(&0_u32.to_le_bytes());
    output.extend_from_slice(model);
    output.extend_from_slice(target);
    for entry in &entries {
        output.extend_from_slice(&entry.layer.to_le_bytes());
        output.push(entry.expert);
        output.push(entry.kind as u8);
        output.push(quant_code(entry.quant_type));
        output.push(entry.layout as u8);
        output.extend_from_slice(&entry.group_size.to_le_bytes());
        output.extend_from_slice(&entry.rows.to_le_bytes());
        output.extend_from_slice(&entry.cols.to_le_bytes());
        output.extend_from_slice(&entry.scale_offset.to_le_bytes());
        output.extend_from_slice(&entry.scale_length.to_le_bytes());
        output.extend_from_slice(&entry.data_offset.to_le_bytes());
        output.extend_from_slice(&entry.data_length.to_le_bytes());
    }
    for value in [
        memory_plan.weights_bytes,
        memory_plan.kv_cache_bytes,
        memory_plan.hidden_a_bytes,
        memory_plan.hidden_b_bytes,
        memory_plan.attention_scratch_bytes,
        memory_plan.moe_scratch_bytes,
        memory_plan.logits_bytes,
        memory_plan.total_bytes,
    ] {
        output.extend_from_slice(&value.to_le_bytes());
    }
    output.extend_from_slice(&blobs);
    debug_assert_eq!(output.len(), data_offset as usize + blobs.len());
    fs::write(path, output)?;
    Ok(())
}

pub fn read_artifact(path: impl AsRef<Path>) -> Result<QxArtifactInfo> {
    let bytes = fs::read(path)?;
    read_artifact_bytes(&bytes)
}

fn read_artifact_bytes(bytes: &[u8]) -> Result<QxArtifactInfo> {
    if bytes.len() < 52 || bytes[..8] != MAGIC {
        bail!("invalid QX artifact header");
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != VERSION {
        bail!("unsupported QX artifact version {version}");
    }
    let model_len = u16::from_le_bytes(bytes[12..14].try_into().unwrap()) as usize;
    let target_len = u16::from_le_bytes(bytes[14..16].try_into().unwrap()) as usize;
    let context_limit = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let directory_offset = u64::from_le_bytes(bytes[20..28].try_into().unwrap());
    let directory_entries = u32::from_le_bytes(bytes[28..32].try_into().unwrap());
    let memory_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    let memory_length = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
    let string_start = 52;
    let target_start = string_start + model_len;
    let directory_start = directory_offset as usize;
    if target_start + target_len > bytes.len()
        || directory_start + directory_entries as usize * 48 > bytes.len()
    {
        bail!("QX artifact offsets exceed file length");
    }
    let model_id = String::from_utf8(bytes[string_start..target_start].to_vec())?;
    let target_id = String::from_utf8(bytes[target_start..target_start + target_len].to_vec())?;
    let mut entries = Vec::with_capacity(directory_entries as usize);
    for index in 0..directory_entries as usize {
        let start = directory_start + index * 48;
        entries.push(TensorDirectoryEntry {
            layer: u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap()),
            expert: bytes[start + 2],
            kind: TensorKind::from_u8(bytes[start + 3])?,
            quant_type: quant_from_code(bytes[start + 4])?,
            layout: PackingLayout::from_u8(bytes[start + 5])
                .ok_or_else(|| anyhow::anyhow!("unknown QX packing layout"))?,
            rows: u32::from_le_bytes(bytes[start + 8..start + 12].try_into().unwrap()),
            cols: u32::from_le_bytes(bytes[start + 12..start + 16].try_into().unwrap()),
            group_size: u16::from_le_bytes(bytes[start + 6..start + 8].try_into().unwrap()),
            scale_offset: u64::from_le_bytes(bytes[start + 16..start + 24].try_into().unwrap()),
            scale_length: u64::from_le_bytes(bytes[start + 24..start + 32].try_into().unwrap()),
            data_offset: u64::from_le_bytes(bytes[start + 32..start + 40].try_into().unwrap()),
            data_length: u64::from_le_bytes(bytes[start + 40..start + 48].try_into().unwrap()),
        });
    }
    if memory_length != 64 || memory_offset as usize + 64 > bytes.len() {
        bail!("invalid QX memory plan");
    }
    let mut values = [0_u64; 8];
    for (index, value) in values.iter_mut().enumerate() {
        let start = memory_offset as usize + index * 8;
        *value = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
    }
    Ok(QxArtifactInfo {
        header: QxHeader {
            model_id,
            target_id,
            context_limit,
            directory_offset,
            directory_entries,
            memory_offset,
            memory_length,
        },
        entries,
        memory_plan: MemoryPlan {
            weights_bytes: values[0],
            kv_cache_bytes: values[1],
            hidden_a_bytes: values[2],
            hidden_b_bytes: values[3],
            attention_scratch_bytes: values[4],
            moe_scratch_bytes: values[5],
            logits_bytes: values[6],
            total_bytes: values[7],
        },
    })
}

fn quant_code(value: QuantType) -> u8 {
    match value {
        QuantType::Q2 => 2,
        QuantType::Q3 => 3,
        QuantType::Q4 => 4,
    }
}

fn quant_from_code(value: u8) -> Result<QuantType> {
    match value {
        2 => Ok(QuantType::Q2),
        3 => Ok(QuantType::Q3),
        4 => Ok(QuantType::Q4),
        _ => bail!("unknown QX quantization code {value}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_magic_is_eight_bytes() {
        assert_eq!(MAGIC.len(), 8);
        assert_eq!(&MAGIC[..6], b"QXMODL");
    }

    #[test]
    fn artifact_round_trip_preserves_directory_and_memory_plan() {
        let tensor = QuantizedTensor::quantize(&[1.0, -0.5, 0.25, 0.0], 2, 2, QuantType::Q3, 2);
        let path = std::env::temp_dir().join(format!("qx-artifact-{}.qx", std::process::id()));
        write_artifact_with_raw(
            &path,
            "granite",
            "apple-a19",
            4096,
            MemoryPlan {
                weights_bytes: 9,
                total_bytes: 10,
                ..Default::default()
            },
            &[CompiledTensor {
                layer: 0,
                expert: 1,
                kind: TensorKind::GateUp,
                tensor,
            }],
            &[RawTensor {
                layer: u16::MAX,
                expert: u8::MAX,
                kind: TensorKind::FinalNorm,
                rows: 2,
                cols: 1,
                values: vec![1.0, -2.0],
            }],
        )
        .unwrap();
        let artifact = MappedArtifact::open(&path).unwrap();
        let raw_index = artifact
            .info()
            .entries
            .iter()
            .position(|entry| entry.kind == TensorKind::FinalNorm)
            .unwrap();
        assert_eq!(artifact.raw_tensor_f32(raw_index).unwrap(), vec![1.0, -2.0]);
        std::fs::remove_file(path).unwrap();
        assert_eq!(artifact.info.header.target_id, "apple-a19");
        assert_eq!(artifact.info.entries[0].quant_type, QuantType::Q3);
        assert_eq!(artifact.info.memory_plan.total_bytes, 10);
    }
}
