//! Fixed-layout metadata and weight primitives for the memory-mappable `.qx`
//! artifact.

use anyhow::{Result, bail};
use half::f16;
use memmap2::{Mmap, UncheckedAdvice};
use qx_quant::{PackingLayout, QuantType, QuantizedTensor};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const MAGIC: [u8; 8] = *b"QXMODL\0\0";
pub const LEGACY_VERSION: u32 = 1;
pub const VERSION: u32 = 2;
pub const WEIGHT_SLAB_ALIGNMENT: u64 = 16 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertRepresentation {
    Group32Q4RowMajor,
    Group32Q4MetalTile32,
    Group32Q4RowInline,
    MixedQ2Q3Q4,
    DenseQ3,
    PaddedQ3,
    Q4K,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QkvTopology {
    Fused,
    Split,
    FusedExperimental,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RmsTopology {
    Materialized,
    ScalarConsumer,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeKvTopology {
    Canonical,
    SplitDirect,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionTopology {
    SingleDispatch,
    ThreeStage,
    HierarchicalGqa,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeTopology {
    SingleDispatch,
    FourStage,
    /// Four-stage MoE with one 1024-threadgroup per selected expert. This is
    /// an install-time candidate; it is intentionally distinct from the
    /// canonical 8-rows-per-threadgroup mapping.
    FourStageWideExpert,
    SplitGateUp,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvPrecision {
    Fp32,
    Fp32KFp16V,
    Fp16KFp32V,
    Fp16,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightStorage {
    CopiedShared,
    MmapNoCopy,
    Hybrid,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SparseExpertPolicy {
    Top8,
    FixedK {
        active_k: u8,
        renormalize: bool,
    },
    PerLayer {
        active_k: Vec<u8>,
        renormalize: bool,
    },
    CumulativeMass {
        threshold: f32,
        min_k: u8,
        max_k: u8,
        renormalize: bool,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TargetProfile {
    pub target_id: String,
    pub gpu_family: String,
    pub os_version: String,
    pub physical_memory_bytes: u64,
    pub max_threads_per_threadgroup: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkloadProfile {
    pub batch: u32,
    pub context_limit: u32,
    pub generation: u32,
    pub sampling: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct QualityBudget {
    pub max_relative_ppl_delta: f64,
    pub max_domain_relative_ppl_delta: f64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompiledMemoryPlan {
    pub budget_bytes: u64,
    pub predicted_peak_bytes: u64,
    pub weight_storage: WeightStorage,
    pub page_alignment: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RepresentationPlan {
    pub gate_up: ExpertRepresentation,
    pub down: ExpertRepresentation,
    pub dense: ExpertRepresentation,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TopologyPlan {
    pub qkv: QkvTopology,
    pub input_rms: RmsTopology,
    pub post_attention_rms: RmsTopology,
    pub rope_kv: RopeKvTopology,
    pub attention: AttentionTopology,
    pub moe: MoeTopology,
    pub kv_precision: KvPrecision,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DispatchSchedule {
    pub command_buffers_per_token: u16,
    pub encoders_per_token: u16,
    pub expected_dispatches_per_token: u32,
    pub internal_cpu_waits: u16,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PlannedBufferRegion {
    pub name: String,
    pub offset: u64,
    pub length: u64,
    pub live_from_dispatch: u32,
    pub live_through_dispatch: u32,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArenaPlan {
    pub arena_bytes: u64,
    pub kv_bytes: u64,
    pub no_decode_allocations: bool,
    pub regions: Vec<PlannedBufferRegion>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PlanProvenance {
    pub compiler_revision: String,
    pub calibration_hash: String,
    pub evaluation_hash: String,
    pub benchmark_contract_hash: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ExecutionPlanManifest {
    pub schema_version: u32,
    pub plan_id: String,
    pub model_id: String,
    pub target: TargetProfile,
    pub workload: WorkloadProfile,
    pub quality_budget: QualityBudget,
    pub memory: CompiledMemoryPlan,
    pub representations: RepresentationPlan,
    pub topology: TopologyPlan,
    pub schedule: DispatchSchedule,
    #[serde(default)]
    pub arena: ArenaPlan,
    pub sparse_experts: SparseExpertPolicy,
    pub provenance: PlanProvenance,
}

impl ExecutionPlanManifest {
    pub fn granite_m5_canonical(
        model_id: &str,
        target_id: &str,
        context_limit: u32,
        memory_plan: MemoryPlan,
    ) -> Self {
        Self {
            schema_version: 1,
            plan_id: "granite-m5-top8-canonical".to_owned(),
            model_id: model_id.to_owned(),
            target: TargetProfile {
                target_id: target_id.to_owned(),
                gpu_family: "apple-m5-pro".to_owned(),
                os_version: String::new(),
                physical_memory_bytes: 0,
                max_threads_per_threadgroup: 1024,
            },
            workload: WorkloadProfile {
                batch: 1,
                context_limit,
                generation: 128,
                sampling: "greedy".to_owned(),
            },
            quality_budget: QualityBudget {
                max_relative_ppl_delta: 0.01,
                max_domain_relative_ppl_delta: 0.02,
            },
            memory: CompiledMemoryPlan {
                budget_bytes: memory_plan.total_bytes,
                predicted_peak_bytes: memory_plan.total_bytes,
                weight_storage: WeightStorage::CopiedShared,
                page_alignment: WEIGHT_SLAB_ALIGNMENT,
            },
            representations: RepresentationPlan {
                gate_up: ExpertRepresentation::Group32Q4RowMajor,
                down: ExpertRepresentation::Group32Q4RowMajor,
                dense: ExpertRepresentation::Group32Q4RowMajor,
            },
            topology: TopologyPlan {
                qkv: QkvTopology::Fused,
                input_rms: RmsTopology::ScalarConsumer,
                post_attention_rms: RmsTopology::ScalarConsumer,
                rope_kv: RopeKvTopology::SplitDirect,
                attention: AttentionTopology::HierarchicalGqa,
                moe: MoeTopology::FourStage,
                kv_precision: KvPrecision::Fp32,
            },
            schedule: DispatchSchedule {
                command_buffers_per_token: 1,
                encoders_per_token: 1,
                expected_dispatches_per_token: 461,
                internal_cpu_waits: 0,
            },
            arena: ArenaPlan::default(),
            sparse_experts: SparseExpertPolicy::Top8,
            provenance: PlanProvenance::default(),
        }
    }
}

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
    pub version: u32,
    pub model_id: String,
    pub target_id: String,
    pub context_limit: u32,
    pub directory_offset: u64,
    pub directory_entries: u32,
    pub memory_offset: u64,
    pub memory_length: u64,
    pub plan_offset: u64,
    pub plan_length: u64,
    pub weights_offset: u64,
    pub weights_length: u64,
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
    pub execution_plan: Option<ExecutionPlanManifest>,
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

    /// Entire page-aligned mapped weight slab, including trailing alignment
    /// padding. It can be wrapped once by Metal and sliced using the relative
    /// ranges returned by `tensor_slab_ranges`.
    pub fn weight_slab(&self) -> Result<&[u8]> {
        let start = self.info.header.weights_offset as usize;
        if start % WEIGHT_SLAB_ALIGNMENT as usize != 0 || start >= self.mmap.len() {
            bail!("QX weight slab is not page aligned");
        }
        Ok(&self.mmap[start..])
    }

    pub fn tensor_slab_ranges(
        &self,
        index: usize,
    ) -> Result<(std::ops::Range<usize>, std::ops::Range<usize>)> {
        let entry = self
            .info
            .entries
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("QX tensor index out of range"))?;
        let base = self.info.header.weights_offset;
        if entry.scale_offset < base || entry.data_offset < base {
            bail!("QX tensor precedes its declared weight slab");
        }
        let scale = (entry.scale_offset - base) as usize
            ..(entry.scale_offset + entry.scale_length - base) as usize;
        let data = (entry.data_offset - base) as usize
            ..(entry.data_offset + entry.data_length - base) as usize;
        Ok((scale, data))
    }

    pub fn quantized_tensor(&self, index: usize) -> Result<QuantizedTensor> {
        let entry = self
            .info
            .entries
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("QX tensor index out of range"))?;
        let (scales, data) = self.tensor_bytes(index)?;
        if entry.layout == PackingLayout::Q4RowInline {
            if entry.quant_type != QuantType::Q4 || entry.group_size != 32 {
                bail!("inline artifact tensor is not group-32 Q4");
            }
            let groups = (entry.cols as usize).div_ceil(entry.group_size as usize);
            let group_bytes = 16_usize;
            let stride = groups * (group_bytes + 2);
            if data.len() != entry.rows as usize * stride {
                bail!("inline artifact tensor has invalid stream length");
            }
            let mut split_data = Vec::with_capacity(entry.rows as usize * groups * group_bytes);
            let mut split_scales = Vec::with_capacity(entry.rows as usize * groups);
            for row in 0..entry.rows as usize {
                for group in 0..groups {
                    let start = row * stride + group * (group_bytes + 2);
                    split_data.extend_from_slice(&data[start..start + group_bytes]);
                    split_scales.push(f16::from_bits(u16::from_le_bytes(
                        data[start + group_bytes..start + group_bytes + 2]
                            .try_into()
                            .unwrap(),
                    )));
                }
            }
            return Ok(QuantizedTensor {
                quant_type: entry.quant_type,
                layout: PackingLayout::RowMajor,
                rows: entry.rows as usize,
                cols: entry.cols as usize,
                group_size: entry.group_size as usize,
                scales: split_scales,
                data: split_data,
            });
        }
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
    let execution_plan = ExecutionPlanManifest::granite_m5_canonical(
        model_id,
        target_id,
        context_limit,
        memory_plan,
    );
    write_artifact_internal(
        path,
        model_id,
        target_id,
        context_limit,
        memory_plan,
        tensors,
        &[],
        &execution_plan,
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
    let execution_plan = ExecutionPlanManifest::granite_m5_canonical(
        model_id,
        target_id,
        context_limit,
        memory_plan,
    );
    write_artifact_internal(
        path,
        model_id,
        target_id,
        context_limit,
        memory_plan,
        tensors,
        raw_tensors,
        &execution_plan,
    )
}

pub fn write_artifact_with_raw_and_plan(
    path: impl AsRef<Path>,
    model_id: &str,
    target_id: &str,
    context_limit: u32,
    memory_plan: MemoryPlan,
    tensors: &[CompiledTensor],
    raw_tensors: &[RawTensor],
    execution_plan: &ExecutionPlanManifest,
) -> Result<()> {
    if execution_plan.model_id != model_id {
        bail!("execution plan model does not match artifact model");
    }
    if execution_plan.target.target_id != target_id {
        bail!("execution plan target does not match artifact target");
    }
    if execution_plan.workload.context_limit != context_limit {
        bail!("execution plan context does not match artifact context");
    }
    write_artifact_internal(
        path,
        model_id,
        target_id,
        context_limit,
        memory_plan,
        tensors,
        raw_tensors,
        execution_plan,
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
    execution_plan: &ExecutionPlanManifest,
) -> Result<()> {
    let model = model_id.as_bytes();
    let target = target_id.as_bytes();
    if model.len() > u16::MAX as usize || target.len() > u16::MAX as usize {
        bail!("QX model or target id is too long");
    }
    let fixed_header = 84_u64;
    let directory_offset = fixed_header + model.len() as u64 + target.len() as u64;
    let directory_size = (tensors.len() + raw_tensors.len()) as u64 * 48;
    let memory_offset = directory_offset + directory_size;
    let memory_length = 64_u64;
    let execution_plan_bytes = serde_json::to_vec(execution_plan)?;
    let plan_offset = memory_offset + memory_length;
    let plan_length = execution_plan_bytes.len() as u64;
    let data_offset = align_u64(plan_offset + plan_length, WEIGHT_SLAB_ALIGNMENT);
    let mut quantized_entries = vec![None; tensors.len()];
    let mut blobs = Vec::new();
    let mut physical_order = (0..tensors.len()).collect::<Vec<_>>();
    physical_order.sort_by_key(|index| {
        let tensor = &tensors[*index];
        (tensor.layer, tensor.kind as u8, tensor.expert)
    });
    let mut group_start = 0;
    while group_start < physical_order.len() {
        let first = &tensors[physical_order[group_start]];
        let mut group_end = group_start + 1;
        while group_end < physical_order.len() {
            let next = &tensors[physical_order[group_end]];
            if next.layer != first.layer || next.kind != first.kind {
                break;
            }
            group_end += 1;
        }
        let group = &physical_order[group_start..group_end];
        if first.tensor.layout == PackingLayout::Q4RowInline {
            for index in group {
                let compiled = &tensors[*index];
                let inline = compiled
                    .tensor
                    .row_inline_bytes()
                    .map_err(|error| anyhow::anyhow!(error))?;
                let offset = data_offset + blobs.len() as u64;
                blobs.extend_from_slice(&inline);
                quantized_entries[*index] = Some(TensorDirectoryEntry {
                    layer: compiled.layer,
                    expert: compiled.expert,
                    quant_type: compiled.tensor.quant_type,
                    layout: compiled.tensor.layout,
                    kind: compiled.kind,
                    rows: compiled.tensor.rows as u32,
                    cols: compiled.tensor.cols as u32,
                    group_size: compiled.tensor.group_size as u16,
                    // Inline entries deliberately point both views at the
                    // same exact byte stream. Runtime preparation selects
                    // the inline Metal kernel from the layout tag.
                    scale_offset: offset,
                    scale_length: inline.len() as u64,
                    data_offset: offset,
                    data_length: inline.len() as u64,
                });
            }
            group_start = group_end;
            continue;
        }
        let mut scale_locations = Vec::with_capacity(group.len());
        for index in group {
            let tensor = &tensors[*index].tensor;
            let offset = data_offset + blobs.len() as u64;
            for scale in &tensor.scales {
                blobs.extend_from_slice(&scale.to_bits().to_le_bytes());
            }
            scale_locations.push((offset, tensor.scales.len() as u64 * 2));
        }
        for (slot, index) in group.iter().enumerate() {
            let compiled = &tensors[*index];
            let tensor_data_offset = data_offset + blobs.len() as u64;
            blobs.extend_from_slice(&compiled.tensor.data);
            let (scale_offset, scale_length) = scale_locations[slot];
            quantized_entries[*index] = Some(TensorDirectoryEntry {
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
        group_start = group_end;
    }
    let mut entries = quantized_entries
        .into_iter()
        .map(|entry| entry.expect("every quantized tensor receives a physical location"))
        .collect::<Vec<_>>();
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
    let weights_length = blobs.len() as u64;
    let padded_file_length = align_u64(data_offset + weights_length, WEIGHT_SLAB_ALIGNMENT);
    let mut output = Vec::with_capacity(padded_file_length as usize);
    output.extend_from_slice(&MAGIC);
    output.extend_from_slice(&VERSION.to_le_bytes());
    output.extend_from_slice(&(model.len() as u16).to_le_bytes());
    output.extend_from_slice(&(target.len() as u16).to_le_bytes());
    output.extend_from_slice(&context_limit.to_le_bytes());
    output.extend_from_slice(&directory_offset.to_le_bytes());
    output.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    output.extend_from_slice(&memory_offset.to_le_bytes());
    output.extend_from_slice(&memory_length.to_le_bytes());
    output.extend_from_slice(&(fixed_header as u32).to_le_bytes());
    output.extend_from_slice(&plan_offset.to_le_bytes());
    output.extend_from_slice(&plan_length.to_le_bytes());
    output.extend_from_slice(&data_offset.to_le_bytes());
    output.extend_from_slice(&weights_length.to_le_bytes());
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
    output.extend_from_slice(&execution_plan_bytes);
    output.resize(data_offset as usize, 0);
    output.extend_from_slice(&blobs);
    output.resize(padded_file_length as usize, 0);
    fs::write(path, output)?;
    Ok(())
}

const fn align_u64(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
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
    if version != LEGACY_VERSION && version != VERSION {
        bail!("unsupported QX artifact version {version}");
    }
    let model_len = u16::from_le_bytes(bytes[12..14].try_into().unwrap()) as usize;
    let target_len = u16::from_le_bytes(bytes[14..16].try_into().unwrap()) as usize;
    let context_limit = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    let directory_offset = u64::from_le_bytes(bytes[20..28].try_into().unwrap());
    let directory_entries = u32::from_le_bytes(bytes[28..32].try_into().unwrap());
    let memory_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
    let memory_length = u64::from_le_bytes(bytes[40..48].try_into().unwrap());
    let (string_start, plan_offset, plan_length, declared_weights_offset, declared_weights_length) =
        if version == VERSION {
            if bytes.len() < 84 {
                bail!("truncated QX v2 artifact header");
            }
            let header_length = u32::from_le_bytes(bytes[48..52].try_into().unwrap()) as usize;
            if header_length != 84 {
                bail!("unsupported QX v2 header length {header_length}");
            }
            (
                header_length,
                u64::from_le_bytes(bytes[52..60].try_into().unwrap()),
                u64::from_le_bytes(bytes[60..68].try_into().unwrap()),
                u64::from_le_bytes(bytes[68..76].try_into().unwrap()),
                u64::from_le_bytes(bytes[76..84].try_into().unwrap()),
            )
        } else {
            (52, 0, 0, 0, 0)
        };
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
    let execution_plan = if version == VERSION {
        let start = plan_offset as usize;
        let end = start.saturating_add(plan_length as usize);
        if start < memory_offset as usize + memory_length as usize || end > bytes.len() {
            bail!("invalid QX execution plan offsets");
        }
        Some(serde_json::from_slice::<ExecutionPlanManifest>(
            &bytes[start..end],
        )?)
    } else {
        None
    };
    let (weights_offset, weights_length) = if version == VERSION {
        let end = declared_weights_offset.saturating_add(declared_weights_length);
        if declared_weights_offset % WEIGHT_SLAB_ALIGNMENT != 0 || end > bytes.len() as u64 {
            bail!("invalid QX aligned weight slab");
        }
        (declared_weights_offset, declared_weights_length)
    } else {
        let start = entries
            .iter()
            .flat_map(|entry| [entry.scale_offset, entry.data_offset])
            .min()
            .unwrap_or(memory_offset + memory_length);
        let end = entries
            .iter()
            .flat_map(|entry| {
                [
                    entry.scale_offset.saturating_add(entry.scale_length),
                    entry.data_offset.saturating_add(entry.data_length),
                ]
            })
            .max()
            .unwrap_or(start);
        (start, end.saturating_sub(start))
    };
    Ok(QxArtifactInfo {
        header: QxHeader {
            version,
            model_id,
            target_id,
            context_limit,
            directory_offset,
            directory_entries,
            memory_offset,
            memory_length,
            plan_offset,
            plan_length,
            weights_offset,
            weights_length,
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
        execution_plan,
    })
}

fn quant_code(value: QuantType) -> u8 {
    match value {
        QuantType::Q2 => 2,
        QuantType::Q3 => 3,
        QuantType::Q4 => 4,
        QuantType::Q4K => 5,
    }
}

fn quant_from_code(value: u8) -> Result<QuantType> {
    match value {
        2 => Ok(QuantType::Q2),
        3 => Ok(QuantType::Q3),
        4 => Ok(QuantType::Q4),
        5 => Ok(QuantType::Q4K),
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
        assert_eq!(artifact.info.header.version, VERSION);
        assert_eq!(
            artifact.info.header.weights_offset % WEIGHT_SLAB_ALIGNMENT,
            0
        );
        assert_eq!(
            artifact.info.execution_plan.as_ref().unwrap().plan_id,
            "granite-m5-top8-canonical"
        );
        assert_eq!(artifact.info.entries[0].quant_type, QuantType::Q3);
        assert_eq!(artifact.info.memory_plan.total_bytes, 10);
    }

    #[test]
    fn artifact_round_trip_decodes_row_inline_q4_stream() {
        let values = (0..64)
            .map(|index| (index as f32 - 12.0) / 7.0)
            .collect::<Vec<_>>();
        let mut tensor = QuantizedTensor::quantize(&values, 2, 32, QuantType::Q4, 32);
        tensor.layout = PackingLayout::Q4RowInline;
        let path = std::env::temp_dir().join(format!("qx-inline-{}.qx", std::process::id()));
        write_artifact_with_raw(
            &path,
            "granite",
            "apple-m5-pro",
            512,
            MemoryPlan {
                weights_bytes: tensor.bytes() as u64,
                total_bytes: tensor.bytes() as u64,
                ..Default::default()
            },
            &[CompiledTensor {
                layer: 0,
                expert: 0,
                kind: TensorKind::GateUp,
                tensor: tensor.clone(),
            }],
            &[],
        )
        .unwrap();
        let artifact = MappedArtifact::open(&path).unwrap();
        let decoded = artifact.quantized_tensor(0).unwrap();
        assert_eq!(decoded.layout, PackingLayout::RowMajor);
        assert_eq!(decoded.data, tensor.data);
        assert_eq!(decoded.scales, tensor.scales);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn legacy_v1_header_remains_readable() {
        let model = b"granite";
        let target = b"apple-m5-pro";
        let directory_offset = 52 + model.len() + target.len();
        let memory_offset = directory_offset;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&LEGACY_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(model.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&(target.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&512_u32.to_le_bytes());
        bytes.extend_from_slice(&(directory_offset as u64).to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&(memory_offset as u64).to_le_bytes());
        bytes.extend_from_slice(&64_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(model);
        bytes.extend_from_slice(target);
        bytes.resize(memory_offset + 64, 0);
        let info = read_artifact_bytes(&bytes).unwrap();
        assert_eq!(info.header.version, LEGACY_VERSION);
        assert_eq!(info.header.model_id, "granite");
        assert!(info.execution_plan.is_none());
    }

    #[test]
    fn artifact_v2_groups_expert_scales_and_payloads_into_contiguous_slabs() {
        let path = std::env::temp_dir().join(format!(
            "qx-artifact-grouped-{}-{}.qx",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let tensors = [0_u8, 1_u8]
            .into_iter()
            .map(|expert| CompiledTensor {
                layer: 0,
                expert,
                kind: TensorKind::GateUp,
                tensor: QuantizedTensor::quantize(
                    &[1.0 + expert as f32, -0.5, 0.25, 0.0],
                    2,
                    2,
                    QuantType::Q4,
                    2,
                ),
            })
            .collect::<Vec<_>>();
        write_artifact(
            &path,
            "granite",
            "apple-m5-pro",
            512,
            MemoryPlan::default(),
            &tensors,
        )
        .unwrap();
        let artifact = MappedArtifact::open(&path).unwrap();
        let first = artifact.tensor_slab_ranges(0).unwrap();
        let second = artifact.tensor_slab_ranges(1).unwrap();
        assert_eq!(first.0.end, second.0.start, "scale slabs must be adjacent");
        assert_eq!(
            first.1.end, second.1.start,
            "payload slabs must be adjacent"
        );
        assert!(second.0.end <= first.1.start, "all scales precede payloads");
        std::fs::remove_file(path).unwrap();
    }
}
