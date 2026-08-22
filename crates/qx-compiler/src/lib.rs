//! CPU reference compiler logic. It deliberately mirrors the eventual GPU
//! boundary: quantize expert weights, execute an expert, measure sensitivity,
//! then choose precision under a byte budget.

use anyhow::{Result, bail};
use qx_format::MemoryPlan;
use qx_model::{EXPERTS, HEAD_DIM, HIDDEN, KV_HEADS, LAYERS, VOCAB};
use qx_quant::{QuantType, QuantizedTensor};
use rayon::prelude::*;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Clone, Debug)]
pub struct ExpertWeights {
    pub gate_up_rows: usize,
    pub gate_up_cols: usize,
    pub down_rows: usize,
    pub down_cols: usize,
    pub gate_up: Vec<f32>,
    pub down: Vec<f32>,
}

impl ExpertWeights {
    pub fn new(
        gate_up_shape: &[usize],
        gate_up: Vec<f32>,
        down_shape: &[usize],
        down: Vec<f32>,
    ) -> Result<Self> {
        if gate_up_shape.len() != 2 || down_shape.len() != 2 {
            bail!("expert projections must be rank-2 matrices");
        }
        let (gate_up_rows, gate_up_cols) = (gate_up_shape[0], gate_up_shape[1]);
        let (down_rows, down_cols) = (down_shape[0], down_shape[1]);
        if gate_up.len() != gate_up_rows * gate_up_cols || down.len() != down_rows * down_cols {
            bail!("expert projection data does not match its shape");
        }
        if gate_up_rows != down_cols * 2 {
            bail!(
                "expected gate/up rows to be twice down columns, got {gate_up_rows} and {down_cols}"
            );
        }
        Ok(Self {
            gate_up_rows,
            gate_up_cols,
            down_rows,
            down_cols,
            gate_up,
            down,
        })
    }

    pub fn run(&self, hidden: &[f32]) -> Vec<f32> {
        assert_eq!(hidden.len(), self.gate_up_cols);
        let gate_up = mat_vec(&self.gate_up, self.gate_up_rows, self.gate_up_cols, hidden);
        let intermediate = self.down_cols;
        let mut activated = vec![0.0; intermediate];
        for index in 0..intermediate {
            activated[index] = silu(gate_up[index]) * gate_up[intermediate + index];
        }
        mat_vec(&self.down, self.down_rows, self.down_cols, &activated)
    }

    pub fn quantize(&self, quant_type: QuantType, group_size: usize) -> QuantizedExpert {
        QuantizedExpert {
            gate_up: QuantizedTensor::quantize(
                &self.gate_up,
                self.gate_up_rows,
                self.gate_up_cols,
                quant_type,
                group_size,
            ),
            down: QuantizedTensor::quantize(
                &self.down,
                self.down_rows,
                self.down_cols,
                quant_type,
                group_size,
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedExpert {
    pub gate_up: QuantizedTensor,
    pub down: QuantizedTensor,
}

impl QuantizedExpert {
    pub fn run(&self, hidden: &[f32]) -> Vec<f32> {
        let gate_up = self.gate_up.mat_vec(hidden);
        let intermediate = self.down.cols;
        let mut activated = vec![0.0; intermediate];
        for index in 0..intermediate {
            activated[index] = silu(gate_up[index]) * gate_up[intermediate + index];
        }
        self.down.mat_vec(&activated)
    }

    pub fn bytes(&self) -> usize {
        self.gate_up.bytes() + self.down.bytes()
    }
}

fn mat_vec(matrix: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    matrix
        .chunks_exact(cols)
        .take(rows)
        .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
        .collect()
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[derive(Clone, Debug)]
pub struct CalibrationSample {
    pub hidden: Vec<f32>,
    pub router_weight: f32,
}

#[derive(Clone, Debug)]
pub struct ExpertSensitivity {
    pub layer: u16,
    pub expert: u8,
    pub q2_error: f32,
    pub q3_error: f32,
    pub q4_error: f32,
    pub q2_mse: f32,
    pub q3_mse: f32,
    pub q4_mse: f32,
    pub q2_cosine: f32,
    pub q3_cosine: f32,
    pub q4_cosine: f32,
    pub activation_frequency: f32,
    pub mean_router_weight: f32,
    pub gate_up_rows: usize,
    pub gate_up_cols: usize,
    pub down_rows: usize,
    pub down_cols: usize,
}

impl ExpertSensitivity {
    pub fn importance(&self) -> f32 {
        self.activation_frequency * self.mean_router_weight
    }

    pub fn bytes_for(&self, quant_type: QuantType, group_size: usize) -> usize {
        quantized_bytes(self.gate_up_rows, self.gate_up_cols, quant_type, group_size)
            + quantized_bytes(self.down_rows, self.down_cols, quant_type, group_size)
    }

    pub fn error_for(&self, quant_type: QuantType) -> f32 {
        match quant_type {
            QuantType::Q2 => self.q2_error,
            QuantType::Q3 => self.q3_error,
            QuantType::Q4 => self.q4_error,
        }
    }
}

pub fn measure_sensitivity(
    layer: u16,
    expert: u8,
    weights: &ExpertWeights,
    samples: &[CalibrationSample],
    total_tokens: usize,
    group_size: usize,
) -> ExpertSensitivity {
    let quantized = [
        weights.quantize(QuantType::Q2, group_size),
        weights.quantize(QuantType::Q3, group_size),
        weights.quantize(QuantType::Q4, group_size),
    ];
    let (errors, mses, cosines, weight_sum) = samples
        .par_iter()
        .map(|sample| {
            let reference = weights.run(&sample.hidden);
            let reference_norm = reference
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let weight = sample.router_weight.max(0.0);
            let mut errors = [0.0; 3];
            let mut mses = [0.0; 3];
            let mut cosines = [0.0; 3];
            for (index, candidate) in quantized.iter().enumerate() {
                let output = candidate.run(&sample.hidden);
                let squared_error = output
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>();
                let error_norm = squared_error.sqrt();
                let mse = squared_error / output.len() as f32;
                let dot = output
                    .iter()
                    .zip(&reference)
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                let candidate_norm = output.iter().map(|value| value * value).sum::<f32>().sqrt();
                let cosine = if candidate_norm == 0.0 || reference_norm == 0.0 {
                    0.0
                } else {
                    dot / (candidate_norm * reference_norm)
                };
                errors[index] = weight
                    * if reference_norm == 0.0 {
                        0.0
                    } else {
                        error_norm / reference_norm
                    };
                mses[index] = weight * mse;
                cosines[index] = weight * cosine;
            }
            (errors, mses, cosines, weight)
        })
        .reduce(
            || ([0.0; 3], [0.0; 3], [0.0; 3], 0.0),
            |left, right| {
                let errors = std::array::from_fn(|index| left.0[index] + right.0[index]);
                let mses = std::array::from_fn(|index| left.1[index] + right.1[index]);
                let cosines = std::array::from_fn(|index| left.2[index] + right.2[index]);
                (errors, mses, cosines, left.3 + right.3)
            },
        );
    let divisor = weight_sum.max(f32::EPSILON);
    let frequency = if total_tokens == 0 {
        0.0
    } else {
        samples.len() as f32 / total_tokens as f32
    };
    ExpertSensitivity {
        layer,
        expert,
        q2_error: errors[0] / divisor,
        q3_error: errors[1] / divisor,
        q4_error: errors[2] / divisor,
        q2_mse: mses[0] / divisor,
        q3_mse: mses[1] / divisor,
        q4_mse: mses[2] / divisor,
        q2_cosine: cosines[0] / divisor,
        q3_cosine: cosines[1] / divisor,
        q4_cosine: cosines[2] / divisor,
        activation_frequency: frequency,
        mean_router_weight: if samples.is_empty() {
            0.0
        } else {
            weight_sum / samples.len() as f32
        },
        gate_up_rows: weights.gate_up_rows,
        gate_up_cols: weights.gate_up_cols,
        down_rows: weights.down_rows,
        down_cols: weights.down_cols,
    }
}

#[derive(Clone, Debug)]
pub struct AllocationEntry {
    pub layer: u16,
    pub expert: u8,
    pub quant_type: QuantType,
    pub error: f32,
    pub importance: f32,
    pub bytes: usize,
}

#[derive(Clone, Debug)]
pub struct AllocationPlan {
    pub entries: Vec<AllocationEntry>,
    pub group_size: usize,
    pub budget: usize,
    pub total_bytes: usize,
    pub weighted_error: f32,
    pub quality_constraint_met: bool,
}

pub fn allocate(
    sensitivities: &[ExpertSensitivity],
    memory_budget: usize,
    quality_loss: f32,
    group_size: usize,
) -> AllocationPlan {
    allocate_with_minimum(
        sensitivities,
        memory_budget,
        quality_loss,
        group_size,
        QuantType::Q2,
    )
}

pub fn allocate_with_minimum(
    sensitivities: &[ExpertSensitivity],
    memory_budget: usize,
    quality_loss: f32,
    group_size: usize,
    minimum: QuantType,
) -> AllocationPlan {
    allocate_with_minimum_options(
        sensitivities,
        memory_budget,
        quality_loss,
        group_size,
        minimum,
        true,
    )
}

pub fn allocate_with_minimum_options(
    sensitivities: &[ExpertSensitivity],
    memory_budget: usize,
    quality_loss: f32,
    group_size: usize,
    minimum: QuantType,
    allow_q3: bool,
) -> AllocationPlan {
    let mut selected = sensitivities
        .iter()
        .map(|item| match minimum {
            QuantType::Q2 if item.q2_error.is_finite() => QuantType::Q2,
            QuantType::Q2 | QuantType::Q3 if allow_q3 && item.q3_error.is_finite() => QuantType::Q3,
            QuantType::Q4 | QuantType::Q3 | QuantType::Q2 => QuantType::Q4,
        })
        .collect::<Vec<_>>();
    let mut total_bytes = sensitivities
        .iter()
        .zip(&selected)
        .map(|(item, quant_type)| item.bytes_for(*quant_type, group_size))
        .sum::<usize>();
    loop {
        let current_error = weighted_error_for(sensitivities, &selected);
        if current_error <= quality_loss {
            break;
        }
        let candidate = sensitivities
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                let next = match selected[index] {
                    QuantType::Q2 if allow_q3 && minimum <= QuantType::Q3 => QuantType::Q3,
                    QuantType::Q2 => QuantType::Q4,
                    QuantType::Q3 if allow_q3 => QuantType::Q4,
                    QuantType::Q3 => return None,
                    QuantType::Q4 => return None,
                };
                let old_bytes = item.bytes_for(selected[index], group_size);
                let new_bytes = item.bytes_for(next, group_size);
                let cost = new_bytes.saturating_sub(old_bytes);
                if total_bytes + cost > memory_budget {
                    return None;
                }
                let benefit = item.importance().max(f32::EPSILON)
                    * (item.error_for(selected[index]) - item.error_for(next));
                if !benefit.is_finite() {
                    return None;
                }
                Some((index, next, cost, benefit))
            })
            .max_by(|a, b| a.3.total_cmp(&b.3));
        let Some((index, next, cost, benefit)) = candidate else {
            break;
        };
        if benefit <= 0.0 {
            break;
        }
        selected[index] = next;
        total_bytes += cost;
    }

    let entries = sensitivities
        .iter()
        .enumerate()
        .map(|(index, item)| AllocationEntry {
            layer: item.layer,
            expert: item.expert,
            quant_type: selected[index],
            error: item.error_for(selected[index]),
            importance: item.importance(),
            bytes: item.bytes_for(selected[index], group_size),
        })
        .collect::<Vec<_>>();
    let total_importance = entries.iter().map(|entry| entry.importance).sum::<f32>();
    let weighted_error = entries
        .iter()
        .map(|entry| {
            if entry.importance == 0.0 {
                0.0
            } else {
                entry.error * entry.importance
            }
        })
        .sum::<f32>()
        / total_importance.max(f32::EPSILON);
    let quality_constraint_met = weighted_error <= quality_loss;
    AllocationPlan {
        entries,
        group_size,
        budget: memory_budget,
        total_bytes,
        weighted_error,
        quality_constraint_met,
    }
}

fn weighted_error_for(sensitivities: &[ExpertSensitivity], selected: &[QuantType]) -> f32 {
    let total_importance = sensitivities
        .iter()
        .map(ExpertSensitivity::importance)
        .sum::<f32>();
    if total_importance == 0.0 {
        return 0.0;
    }
    sensitivities
        .iter()
        .zip(selected)
        .map(|(item, quant_type)| {
            let importance = item.importance();
            if importance == 0.0 {
                0.0
            } else {
                importance * item.error_for(*quant_type)
            }
        })
        .sum::<f32>()
        / total_importance
}

pub fn quantized_bytes(
    rows: usize,
    cols: usize,
    quant_type: QuantType,
    group_size: usize,
) -> usize {
    let groups = rows * cols.div_ceil(group_size);
    (rows * cols * quant_type.bits()).div_ceil(8) + groups * 2
}

pub fn granite_memory_plan(weights_bytes: usize, context_limit: usize) -> MemoryPlan {
    let kv_cache_bytes = LAYERS * 2 * KV_HEADS * HEAD_DIM * context_limit * 2;
    let hidden_bytes = HIDDEN * 2;
    let attention_scratch_bytes = HIDDEN * HIDDEN * 2;
    let moe_scratch_bytes = EXPERTS.max(8) * HIDDEN * 2;
    let logits_bytes = VOCAB * 2;
    let total_bytes = weights_bytes
        + kv_cache_bytes
        + hidden_bytes * 2
        + attention_scratch_bytes
        + moe_scratch_bytes
        + logits_bytes;
    MemoryPlan {
        weights_bytes: weights_bytes as u64,
        kv_cache_bytes: kv_cache_bytes as u64,
        hidden_a_bytes: hidden_bytes as u64,
        hidden_b_bytes: hidden_bytes as u64,
        attention_scratch_bytes: attention_scratch_bytes as u64,
        moe_scratch_bytes: moe_scratch_bytes as u64,
        logits_bytes: logits_bytes as u64,
        total_bytes: total_bytes as u64,
    }
}

/// A compact, versioned binary trace. The llama.cpp instrumentation bridge can
/// emit this without depending on any QX runtime types.
#[derive(Clone, Debug)]
pub struct TraceRecord {
    pub layer: u16,
    pub expert: u8,
    pub router_weight: f32,
    /// Token position within the captured graph. Version-1 traces did not
    /// carry this field and are read with a synthetic sequential position.
    pub token: u16,
    pub hidden: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct QxTrace {
    pub hidden_size: usize,
    pub records: Vec<TraceRecord>,
}

const TRACE_MAGIC: &[u8; 8] = b"QXTRACE\0";

impl QxTrace {
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(TRACE_MAGIC);
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&(self.hidden_size as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.records.len() as u64).to_le_bytes());
        for record in &self.records {
            if record.hidden.len() != self.hidden_size {
                bail!("trace record hidden size mismatch");
            }
            bytes.extend_from_slice(&record.layer.to_le_bytes());
            bytes.push(record.expert);
            bytes.push(0);
            bytes.extend_from_slice(&record.router_weight.to_le_bytes());
            bytes.extend_from_slice(&record.token.to_le_bytes());
            for value in &record.hidden {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() < 24 || &bytes[..8] != TRACE_MAGIC {
            bail!("invalid QXTRACE header");
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != 1 && version != 2 {
            bail!("unsupported QXTRACE version {version}");
        }
        let hidden_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let record_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let record_size = if version == 1 { 8 } else { 10 } + hidden_size * 4;
        if bytes.len() != 24 + record_count * record_size {
            bail!("truncated or oversized QXTRACE");
        }
        let mut records = Vec::with_capacity(record_count);
        for index in 0..record_count {
            let start = 24 + index * record_size;
            let layer = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
            let expert = bytes[start + 2];
            let router_weight = f32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
            let token = if version == 1 {
                (index as u16) / 8
            } else {
                u16::from_le_bytes(bytes[start + 8..start + 10].try_into().unwrap())
            };
            let hidden_start = if version == 1 { 8 } else { 10 };
            let hidden = bytes[start + hidden_start..start + record_size]
                .chunks_exact(4)
                .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                .collect();
            records.push(TraceRecord {
                layer,
                expert,
                router_weight,
                token,
                hidden,
            });
        }
        Ok(Self {
            hidden_size,
            records,
        })
    }

    pub fn samples(&self) -> BTreeMap<(u16, u8), Vec<CalibrationSample>> {
        let mut grouped = BTreeMap::new();
        for record in &self.records {
            grouped
                .entry((record.layer, record.expert))
                .or_insert_with(Vec::new)
                .push(CalibrationSample {
                    hidden: record.hidden.clone(),
                    router_weight: record.router_weight,
                });
        }
        grouped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights() -> ExpertWeights {
        ExpertWeights::new(
            &[4, 2],
            vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8],
            &[2, 2],
            vec![0.2, 0.1, -0.4, 0.3],
        )
        .unwrap()
    }

    #[test]
    fn quantized_expert_tracks_reference() {
        let expert = weights();
        let input = [0.25, -0.75];
        let reference = expert.run(&input);
        let candidate = expert.quantize(QuantType::Q4, 2).run(&input);
        assert_eq!(reference.len(), 2);
        assert!(
            reference
                .iter()
                .zip(candidate)
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
                < 0.2
        );
    }

    #[test]
    fn allocator_spends_budget_on_high_importance_expert() {
        let base = weights();
        let mut a = measure_sensitivity(
            0,
            0,
            &base,
            &[CalibrationSample {
                hidden: vec![1.0, 0.0],
                router_weight: 1.0,
            }],
            1,
            2,
        );
        let mut b = a.clone();
        b.expert = 1;
        a.activation_frequency = 1.0;
        b.activation_frequency = 0.1;
        let q2 = a.bytes_for(QuantType::Q2, 2);
        let q3 = a.bytes_for(QuantType::Q3, 2);
        let plan = allocate(&[a, b], q2 * 2 + (q3 - q2), 1.0, 2);
        assert_eq!(plan.entries[0].quant_type, QuantType::Q3);
    }

    #[test]
    fn trace_round_trip() {
        let trace = QxTrace {
            hidden_size: 2,
            records: vec![TraceRecord {
                layer: 1,
                expert: 2,
                router_weight: 0.5,
                token: 0,
                hidden: vec![1.0, -2.0],
            }],
        };
        let path = std::env::temp_dir().join(format!("qx-trace-{}.bin", std::process::id()));
        trace.write(&path).unwrap();
        let decoded = QxTrace::read(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(decoded.records[0].hidden, vec![1.0, -2.0]);
    }
}
