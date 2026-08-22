//! Small, deterministic, CPU reference quantizers for the QX V0 experiment.
//!
//! The layout is row-major. Each row is split into groups; every group stores
//! one FP16 scale followed by its packed integer values. Q3 uses exactly three
//! bytes for every eight values.

use half::f16;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum QuantType {
    Q2,
    Q3,
    Q4,
    Q4K,
}

/// Physical layout of packed quantized matrix data.
///
/// RowMajor is the portable CPU/reference layout. MetalTile32 groups rows in
/// 32-row tiles so a fixed Granite Metal kernel can walk scales and packed
/// values in the order consumed by its threadgroup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PackingLayout {
    RowMajor = 0,
    MetalTile32 = 1,
    MetalTile32Paired = 2,
    /// Exact-Q4 row-major stream with each group's FP16 scale immediately
    /// following its packed codes. The logical values and byte count are
    /// unchanged; artifact/runtime consumers must use the inline kernel.
    Q4RowInline = 3,
}

impl PackingLayout {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::RowMajor),
            1 => Some(Self::MetalTile32),
            2 => Some(Self::MetalTile32Paired),
            3 => Some(Self::Q4RowInline),
            _ => None,
        }
    }
}

impl QuantType {
    pub const fn bits(self) -> usize {
        match self {
            Self::Q2 => 2,
            Self::Q3 => 3,
            Self::Q4 => 4,
            Self::Q4K => 4,
        }
    }

    const fn qmax(self) -> i32 {
        (1_i32 << (self.bits() - 1)) - 1
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "q2" => Some(Self::Q2),
            "q3" => Some(Self::Q3),
            "q4" => Some(Self::Q4),
            "q4_k" | "q4k" => Some(Self::Q4K),
            _ => None,
        }
    }
}

impl std::fmt::Display for QuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4K => f.write_str("q4_k"),
            _ => write!(f, "q{}", self.bits()),
        }
    }
}

pub trait QuantCodec {
    fn quantize(src: &[f32], dst: &mut [u8], bits: QuantType);
    fn dequantize(src: &[u8], dst: &mut [f32], bits: QuantType);
}

pub struct PackedCodec;

impl QuantCodec for PackedCodec {
    fn quantize(src: &[f32], dst: &mut [u8], bits: QuantType) {
        pack_values(src, dst, bits);
    }

    fn dequantize(src: &[u8], dst: &mut [f32], bits: QuantType) {
        unpack_values(src, dst, bits);
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedTensor {
    pub quant_type: QuantType,
    pub layout: PackingLayout,
    pub rows: usize,
    pub cols: usize,
    pub group_size: usize,
    pub scales: Vec<f16>,
    pub data: Vec<u8>,
}

impl QuantizedTensor {
    pub fn quantize(
        src: &[f32],
        rows: usize,
        cols: usize,
        quant_type: QuantType,
        group_size: usize,
    ) -> Self {
        assert!(rows > 0 && cols > 0 && group_size > 0);
        assert_eq!(src.len(), rows * cols);
        if quant_type == QuantType::Q4K {
            assert_eq!(group_size, 256);
            assert!(cols.is_multiple_of(256));
            return quantize_q4k(src, rows, cols);
        }
        let groups_per_row = cols.div_ceil(group_size);
        let mut scales = Vec::with_capacity(rows * groups_per_row);
        let mut data = Vec::new();

        for row in src.chunks_exact(cols) {
            for group in row.chunks(group_size) {
                let max_abs = group.iter().copied().map(f32::abs).fold(0.0, f32::max);
                let scale = if max_abs == 0.0 {
                    1.0
                } else {
                    max_abs / quant_type.qmax() as f32
                };
                scales.push(f16::from_f32(scale));
                let start = data.len();
                data.resize(start + packed_len(group.len(), quant_type), 0);
                pack_scaled(group, scale, &mut data[start..], quant_type);
            }
        }

        Self {
            quant_type,
            layout: PackingLayout::RowMajor,
            rows,
            cols,
            group_size,
            scales,
            data,
        }
    }

    /// Quantize and arrange the result for a concrete execution target.
    pub fn quantize_for_layout(
        src: &[f32],
        rows: usize,
        cols: usize,
        quant_type: QuantType,
        group_size: usize,
        layout: PackingLayout,
    ) -> Result<Self, String> {
        let mut tensor = Self::quantize(src, rows, cols, quant_type, group_size);
        tensor.repack(layout)?;
        Ok(tensor)
    }

    /// Reorder packed values and scales without changing their logical matrix.
    pub fn repack(&mut self, layout: PackingLayout) -> Result<(), String> {
        if self.layout == layout {
            return Ok(());
        }
        if self.quant_type == QuantType::Q4K {
            return Err(
                "Q4_K currently supports only its canonical row-major block layout".to_owned(),
            );
        }
        match layout {
            PackingLayout::RowMajor => {
                let values = self.dequantize();
                *self = Self::quantize(
                    &values,
                    self.rows,
                    self.cols,
                    self.quant_type,
                    self.group_size,
                );
                Ok(())
            }
            PackingLayout::MetalTile32 | PackingLayout::MetalTile32Paired => {
                if !self.rows.is_multiple_of(32) {
                    return Err(format!(
                        "MetalTile32 requires rows divisible by 32, got {}",
                        self.rows
                    ));
                }
                let groups = self.cols.div_ceil(self.group_size);
                let row_bytes = (0..groups)
                    .map(|group| {
                        packed_len(
                            (self.cols - group * self.group_size).min(self.group_size),
                            self.quant_type,
                        )
                    })
                    .sum::<usize>();
                let mut data = Vec::with_capacity(self.data.len());
                let mut scales = Vec::with_capacity(self.scales.len());
                for tile in (0..self.rows).step_by(32) {
                    for group in 0..groups {
                        let group_len = (self.cols - group * self.group_size).min(self.group_size);
                        let group_bytes = packed_len(group_len, self.quant_type);
                        for lane in 0..32 {
                            let row = tile + lane;
                            let source = row * row_bytes
                                + (0..group)
                                    .map(|prior| {
                                        packed_len(
                                            (self.cols - prior * self.group_size)
                                                .min(self.group_size),
                                            self.quant_type,
                                        )
                                    })
                                    .sum::<usize>();
                            data.extend_from_slice(&self.data[source..source + group_bytes]);
                            scales.push(self.scales[row * groups + group]);
                        }
                    }
                }
                self.data = data;
                self.scales = scales;
                self.layout = PackingLayout::MetalTile32;
                Ok(())
            }
            PackingLayout::Q4RowInline => {
                if self.quant_type != QuantType::Q4 || self.group_size != 32 {
                    return Err("Q4RowInline requires group-32 Q4".to_owned());
                }
                self.layout = PackingLayout::Q4RowInline;
                Ok(())
            }
        }
    }

    /// Return the exact-Q4 row-inline stream used by the artifact-v2
    /// transport candidate: 16 code bytes followed by one FP16 scale for
    /// every group-32 block. No quantized value is changed.
    pub fn row_inline_bytes(&self) -> Result<Vec<u8>, String> {
        if self.quant_type != QuantType::Q4 || self.group_size != 32 {
            return Err("row-inline transport requires group-32 Q4".to_owned());
        }
        let groups = self.cols.div_ceil(self.group_size);
        let group_bytes = packed_len(self.group_size, self.quant_type);
        let row_bytes = groups * (group_bytes + 2);
        let mut output = Vec::with_capacity(self.rows * row_bytes);
        for row in 0..self.rows {
            for group in 0..groups {
                // `data` and `scales` remain in their normal row-major
                // split form until the artifact writer emits this stream.
                let data_offset = row * groups * group_bytes + group * group_bytes;
                let scale_offset = row * groups + group;
                output.extend_from_slice(&self.data[data_offset..data_offset + group_bytes]);
                output.extend_from_slice(&self.scales[scale_offset].to_bits().to_le_bytes());
            }
        }
        Ok(output)
    }

    fn group_location(&self, row: usize, group: usize) -> (usize, usize) {
        let groups = self.cols.div_ceil(self.group_size);
        let group_bytes = packed_len(
            (self.cols - group * self.group_size).min(self.group_size),
            self.quant_type,
        );
        match self.layout {
            PackingLayout::RowMajor => {
                let row_bytes = (0..groups)
                    .map(|index| {
                        packed_len(
                            (self.cols - index * self.group_size).min(self.group_size),
                            self.quant_type,
                        )
                    })
                    .sum::<usize>();
                let group_offset = (0..group)
                    .map(|index| {
                        packed_len(
                            (self.cols - index * self.group_size).min(self.group_size),
                            self.quant_type,
                        )
                    })
                    .sum::<usize>();
                (row * row_bytes + group_offset, row * groups + group)
            }
            PackingLayout::MetalTile32 => {
                let tile = row / 32;
                let lane = row % 32;
                let tile_bytes = 32
                    * (0..groups)
                        .map(|index| {
                            packed_len(
                                (self.cols - index * self.group_size).min(self.group_size),
                                self.quant_type,
                            )
                        })
                        .sum::<usize>();
                let group_offset = 32
                    * (0..group)
                        .map(|index| {
                            packed_len(
                                (self.cols - index * self.group_size).min(self.group_size),
                                self.quant_type,
                            )
                        })
                        .sum::<usize>();
                (
                    tile * tile_bytes + group_offset + lane * group_bytes,
                    tile * 32 * groups + group * 32 + lane,
                )
            }
            PackingLayout::MetalTile32Paired => {
                if self.rows != 1024 || groups != 32 {
                    return self.group_location_for_tile32(row, group);
                }
                let physical_row = if row < 512 {
                    row * 2
                } else {
                    (row - 512) * 2 + 1
                };
                self.group_location_for_tile32(physical_row, group)
            }
            PackingLayout::Q4RowInline => {
                let row_bytes = groups * (group_bytes + 2);
                (
                    row * row_bytes + group * (group_bytes + 2),
                    row * groups + group,
                )
            }
        }
    }

    fn group_location_for_tile32(&self, row: usize, group: usize) -> (usize, usize) {
        let groups = self.cols.div_ceil(self.group_size);
        let group_bytes = packed_len(
            (self.cols - group * self.group_size).min(self.group_size),
            self.quant_type,
        );
        let tile = row / 32;
        let lane = row % 32;
        let tile_bytes = 32
            * (0..groups)
                .map(|index| {
                    packed_len(
                        (self.cols - index * self.group_size).min(self.group_size),
                        self.quant_type,
                    )
                })
                .sum::<usize>();
        let group_offset = 32
            * (0..group)
                .map(|index| {
                    packed_len(
                        (self.cols - index * self.group_size).min(self.group_size),
                        self.quant_type,
                    )
                })
                .sum::<usize>();
        (
            tile * tile_bytes + group_offset + lane * group_bytes,
            tile * 32 * groups + group * 32 + lane,
        )
    }

    pub fn dequantize(&self) -> Vec<f32> {
        if self.quant_type == QuantType::Q4K {
            return dequantize_q4k(self);
        }
        let groups_per_row = self.cols.div_ceil(self.group_size);
        let mut output = Vec::with_capacity(self.rows * self.cols);
        for row in 0..self.rows {
            let mut remaining = self.cols;
            let mut group = 0;
            while remaining > 0 {
                let len = remaining.min(self.group_size);
                let bytes = packed_len(len, self.quant_type);
                let (data_offset, scale_offset) = self.group_location(row, group);
                let scale = self.scales[scale_offset].to_f32();
                let mut values = vec![0.0; len];
                unpack_scaled(
                    &self.data[data_offset..data_offset + bytes],
                    &mut values,
                    scale,
                    self.quant_type,
                );
                output.extend(values);
                remaining -= len;
                group += 1;
            }
        }
        debug_assert_eq!(self.scales.len(), self.rows * groups_per_row);
        output
    }

    /// Matrix-vector multiplication without constructing a dequantized matrix.
    /// This is the CPU reference for the eventual Metal GEMV kernels.
    pub fn mat_vec(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cols);
        if self.quant_type == QuantType::Q4K {
            return mat_vec_q4k(self, input);
        }
        let groups_per_row = self.cols.div_ceil(self.group_size);
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let mut accumulator = 0.0;
            let mut remaining = self.cols;
            let mut column = 0;
            let mut group = 0;
            while remaining > 0 {
                let len = remaining.min(self.group_size);
                let bytes = packed_len(len, self.quant_type);
                let (data_offset, scale_offset) = self.group_location(row, group);
                let scale = self.scales[scale_offset].to_f32();
                let packed = &self.data[data_offset..data_offset + bytes];
                for offset in 0..len {
                    let code = unpack_code(packed, offset, self.quant_type);
                    let value = (code as i32 - self.quant_type.qmax()) as f32 * scale;
                    accumulator += value * input[column + offset];
                }
                remaining -= len;
                group += 1;
                column += len;
            }
            output.push(accumulator);
            debug_assert_eq!(column, self.cols);
            debug_assert_eq!(group, groups_per_row);
        }
        output
    }

    /// Decode one logical row without materializing the complete matrix.
    pub fn row(&self, row: usize) -> Vec<f32> {
        assert!(row < self.rows);
        if self.quant_type == QuantType::Q4K {
            let all = dequantize_q4k(self);
            return all[row * self.cols..(row + 1) * self.cols].to_vec();
        }
        let mut output = Vec::with_capacity(self.cols);
        let mut remaining = self.cols;
        let mut group = 0;
        while remaining > 0 {
            let len = remaining.min(self.group_size);
            let bytes = packed_len(len, self.quant_type);
            let (data_offset, scale_offset) = self.group_location(row, group);
            let scale = self.scales[scale_offset].to_f32();
            let packed = &self.data[data_offset..data_offset + bytes];
            for offset in 0..len {
                let code = unpack_code(packed, offset, self.quant_type);
                output.push((code as i32 - self.quant_type.qmax()) as f32 * scale);
            }
            remaining -= len;
            group += 1;
        }
        output
    }

    pub fn bytes(&self) -> usize {
        self.data.len() + self.scales.len() * std::mem::size_of::<f16>()
    }

    /// Return a Granite gate/up matrix in the paired physical order used by
    /// the experimental Metal MoE kernel.  Logical rows 0..512 are gate and
    /// rows 512..1024 are up; the packed stream instead stores
    /// gate[0], up[0], gate[1], up[1], ... inside each 32-row tile.
    ///
    /// This deliberately returns raw buffers rather than a `QuantizedTensor`:
    /// the paired order is an execution-specific representation, not a
    /// general-purpose matrix layout for CPU dequantization.
    pub fn repack_gate_up_paired(&self) -> Result<(Vec<f16>, Vec<u8>), String> {
        if self.quant_type != QuantType::Q4
            || self.rows != 1024
            || self.cols != 1024
            || self.group_size != 32
        {
            return Err(format!(
                "paired Granite gate/up packing requires Q4 [1024, 1024] group-32, got {:?} [{}x{}] group-{}",
                self.quant_type, self.rows, self.cols, self.group_size
            ));
        }
        let groups = self.cols / self.group_size;
        let group_bytes = packed_len(self.group_size, self.quant_type);
        let row_bytes = groups * group_bytes;
        let mut data = Vec::with_capacity(self.data.len());
        let mut scales = Vec::with_capacity(self.scales.len());
        for tile in (0..self.rows).step_by(32) {
            for group in 0..groups {
                for lane in 0..32 {
                    let physical_row = tile + lane;
                    let logical_row = if physical_row % 2 == 0 {
                        physical_row / 2
                    } else {
                        512 + physical_row / 2
                    };
                    let (source, scale) = self.group_location(logical_row, group);
                    debug_assert_eq!(source, logical_row * row_bytes + group * group_bytes);
                    data.extend_from_slice(&self.data[source..source + group_bytes]);
                    scales.push(self.scales[scale]);
                }
            }
        }
        Ok((scales, data))
    }

    /// Copy a logical row range into a standalone row-major quantized tensor.
    /// This preserves the original packed codes/scales exactly and is used to
    /// split Granite's merged gate/up matrix for topology experiments.
    pub fn row_range(&self, start: usize, count: usize) -> Result<Self, String> {
        if start.checked_add(count).is_none() || start + count > self.rows {
            return Err(format!(
                "row range [{start}, {}) exceeds tensor rows {}",
                start + count,
                self.rows
            ));
        }
        let groups = self.cols.div_ceil(self.group_size);
        let mut data =
            Vec::with_capacity(count * groups * packed_len(self.group_size, self.quant_type));
        let mut scales = Vec::with_capacity(count * groups);
        for row in start..start + count {
            for group in 0..groups {
                let bytes = packed_len(
                    (self.cols - group * self.group_size).min(self.group_size),
                    self.quant_type,
                );
                let (offset, scale) = self.group_location(row, group);
                data.extend_from_slice(&self.data[offset..offset + bytes]);
                scales.push(self.scales[scale]);
            }
        }
        Ok(Self {
            quant_type: self.quant_type,
            layout: PackingLayout::RowMajor,
            rows: count,
            cols: self.cols,
            group_size: self.group_size,
            scales,
            data,
        })
    }

    pub fn mse(&self, original: &[f32]) -> f32 {
        assert_eq!(original.len(), self.rows * self.cols);
        let reconstructed = self.dequantize();
        original
            .iter()
            .zip(reconstructed)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / original.len() as f32
    }

    pub fn relative_l2(&self, original: &[f32]) -> f32 {
        let reconstructed = self.dequantize();
        let numerator = original
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let denominator = original.iter().map(|v| v * v).sum::<f32>().sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

fn packed_len(values: usize, quant_type: QuantType) -> usize {
    if quant_type == QuantType::Q4K {
        return values.div_ceil(256) * 140;
    }
    (values * quant_type.bits()).div_ceil(8)
}

fn quantize_q4k(src: &[f32], rows: usize, cols: usize) -> QuantizedTensor {
    let blocks_per_row = cols / 256;
    let mut scales = Vec::with_capacity(rows * blocks_per_row * 2);
    let mut data = Vec::with_capacity(rows * blocks_per_row * 140);
    for row in src.chunks_exact(cols) {
        for block in row.chunks_exact(256) {
            let mut levels = [0_u8; 256];
            let mut aux = [0_u8; 32];
            let mut sub_scales = [0.0_f32; 8];
            let mut sub_mins = [0.0_f32; 8];
            for sub in 0..8 {
                let values = &block[sub * 32..(sub + 1) * 32];
                let average = (values.iter().map(|value| value * value).sum::<f32>() / 32.0).sqrt();
                let mut weights = [0.0_f32; 32];
                for index in 0..32 {
                    weights[index] = average + values[index].abs();
                }
                sub_scales[sub] = make_qkx2_quants(
                    values,
                    &weights,
                    &mut levels[sub * 32..(sub + 1) * 32],
                    &mut sub_mins[sub],
                    &mut aux,
                );
            }
            let max_scale = sub_scales.iter().copied().fold(0.0_f32, f32::max);
            let max_min = sub_mins.iter().copied().fold(0.0_f32, f32::max);
            let mut metadata = [0_u8; 12];
            for sub in 0..8 {
                let scale = if max_scale > 0.0 {
                    (63.0 * sub_scales[sub] / max_scale)
                        .round()
                        .clamp(0.0, 63.0) as u8
                } else {
                    0
                };
                let minimum = if max_min > 0.0 {
                    (63.0 * sub_mins[sub] / max_min).round().clamp(0.0, 63.0) as u8
                } else {
                    0
                };
                if sub < 4 {
                    metadata[sub] = scale;
                    metadata[sub + 4] = minimum;
                } else {
                    metadata[sub + 4] = (scale & 0x0f) | ((minimum & 0x0f) << 4);
                    metadata[sub - 4] |= (scale >> 4) << 6;
                    metadata[sub] |= (minimum >> 4) << 6;
                }
            }
            let d = f16::from_f32(max_scale / 63.0);
            let dmin = f16::from_f32(max_min / 63.0);
            scales.push(d);
            scales.push(dmin);
            for sub in 0..8 {
                let (scale_code, min_code) = q4k_scale_min(&metadata, sub);
                let step = d.to_f32() * scale_code as f32;
                let minimum = dmin.to_f32() * min_code as f32;
                if step != 0.0 {
                    for index in 0..32 {
                        levels[sub * 32 + index] = ((block[sub * 32 + index] + minimum) / step)
                            .round()
                            .clamp(0.0, 15.0)
                            as u8;
                    }
                }
            }
            data.extend_from_slice(&metadata);
            for pair in 0..4 {
                for index in 0..32 {
                    data.push(levels[pair * 64 + index] | (levels[pair * 64 + 32 + index] << 4));
                }
            }
        }
    }
    QuantizedTensor {
        quant_type: QuantType::Q4K,
        layout: PackingLayout::RowMajor,
        rows,
        cols,
        group_size: 256,
        scales,
        data,
    }
}

fn make_qkx2_quants(
    values: &[f32],
    weights: &[f32],
    levels: &mut [u8],
    minimum_out: &mut f32,
    aux: &mut [u8],
) -> f32 {
    let mut minimum = values
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .min(0.0);
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_weight = weights.iter().sum::<f32>();
    let sum_x = values.iter().zip(weights).map(|(x, w)| x * w).sum::<f32>();
    if maximum == minimum {
        levels.fill(0);
        *minimum_out = -minimum;
        return 0.0;
    }
    let mut inverse = 15.0 / (maximum - minimum);
    let mut scale = 1.0 / inverse;
    let mut best_error = 0.0;
    for index in 0..values.len() {
        levels[index] = (inverse * (values[index] - minimum))
            .round()
            .clamp(0.0, 15.0) as u8;
        let difference = scale * levels[index] as f32 + minimum - values[index];
        best_error += weights[index] * difference * difference;
    }
    for step in 0..=20 {
        inverse = (-1.0 + 0.1 * step as f32 + 15.0) / (maximum - minimum);
        let mut sum_l = 0.0;
        let mut sum_l2 = 0.0;
        let mut sum_xl = 0.0;
        for index in 0..values.len() {
            let level = (inverse * (values[index] - minimum))
                .round()
                .clamp(0.0, 15.0) as u8;
            aux[index] = level;
            let weighted = weights[index];
            sum_l += weighted * level as f32;
            sum_l2 += weighted * (level as f32) * (level as f32);
            sum_xl += weighted * level as f32 * values[index];
        }
        let determinant = sum_weight * sum_l2 - sum_l * sum_l;
        if determinant > 0.0 {
            let mut candidate_scale = (sum_weight * sum_xl - sum_x * sum_l) / determinant;
            let mut candidate_minimum = (sum_l2 * sum_x - sum_l * sum_xl) / determinant;
            if candidate_minimum > 0.0 {
                candidate_minimum = 0.0;
                candidate_scale = sum_xl / sum_l2;
            }
            let error = values
                .iter()
                .zip(weights)
                .zip(aux.iter())
                .map(|((value, weight), level)| {
                    let difference = candidate_scale * *level as f32 + candidate_minimum - value;
                    weight * difference * difference
                })
                .sum::<f32>();
            if error < best_error {
                levels.copy_from_slice(&aux[..levels.len()]);
                best_error = error;
                scale = candidate_scale;
                minimum = candidate_minimum;
            }
        }
    }
    *minimum_out = -minimum;
    scale
}

fn q4k_scale_min(metadata: &[u8], sub: usize) -> (u8, u8) {
    if sub < 4 {
        (metadata[sub] & 63, metadata[sub + 4] & 63)
    } else {
        (
            (metadata[sub + 4] & 0x0f) | ((metadata[sub - 4] >> 6) << 4),
            (metadata[sub + 4] >> 4) | ((metadata[sub] >> 6) << 4),
        )
    }
}

fn q4k_value(tensor: &QuantizedTensor, row: usize, column: usize) -> f32 {
    let blocks_per_row = tensor.cols / 256;
    let block = column / 256;
    let within = column % 256;
    let sub = within / 32;
    let local = within % 32;
    let block_index = row * blocks_per_row + block;
    let data_base = block_index * 140;
    let metadata = &tensor.data[data_base..data_base + 12];
    let packed = tensor.data[data_base + 12 + (sub / 2) * 32 + local];
    let code = if sub.is_multiple_of(2) {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let (scale_code, min_code) = q4k_scale_min(metadata, sub);
    let d = tensor.scales[block_index * 2].to_f32();
    let dmin = tensor.scales[block_index * 2 + 1].to_f32();
    d * scale_code as f32 * code as f32 - dmin * min_code as f32
}

fn dequantize_q4k(tensor: &QuantizedTensor) -> Vec<f32> {
    let mut values = Vec::with_capacity(tensor.rows * tensor.cols);
    for row in 0..tensor.rows {
        for column in 0..tensor.cols {
            values.push(q4k_value(tensor, row, column));
        }
    }
    values
}

fn mat_vec_q4k(tensor: &QuantizedTensor, input: &[f32]) -> Vec<f32> {
    (0..tensor.rows)
        .map(|row| {
            input
                .iter()
                .enumerate()
                .map(|(column, value)| q4k_value(tensor, row, column) * value)
                .sum()
        })
        .collect()
}

fn pack_values(values: &[f32], dst: &mut [u8], bits: QuantType) {
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0, f32::max);
    let scale = if max_abs == 0.0 {
        1.0
    } else {
        max_abs / bits.qmax() as f32
    };
    pack_scaled(values, scale, dst, bits);
}

fn pack_scaled(values: &[f32], scale: f32, dst: &mut [u8], bits: QuantType) {
    dst.fill(0);
    let mask = (1_u8 << bits.bits()) - 1;
    for (index, value) in values.iter().enumerate() {
        let quantized = (value / scale)
            .round()
            .clamp(-(bits.qmax() as f32), bits.qmax() as f32) as i32;
        let code = (quantized + bits.qmax()) as u8;
        let bit = index * bits.bits();
        let byte = bit / 8;
        let shift = bit % 8;
        dst[byte] |= (code & mask) << shift;
        if shift + bits.bits() > 8 {
            dst[byte + 1] |= code >> (8 - shift);
        }
    }
}

fn unpack_values(src: &[u8], dst: &mut [f32], bits: QuantType) {
    unpack_scaled(src, dst, 1.0, bits);
}

fn unpack_code(src: &[u8], index: usize, bits: QuantType) -> u8 {
    let bit = index * bits.bits();
    let byte = bit / 8;
    let shift = bit % 8;
    let mask = (1_u8 << bits.bits()) - 1;
    let mut code = (src[byte] >> shift) & mask;
    if shift + bits.bits() > 8 {
        code |= (src[byte + 1] << (8 - shift)) & mask;
    }
    code
}

fn unpack_scaled(src: &[u8], dst: &mut [f32], scale: f32, bits: QuantType) {
    let mask = (1_u8 << bits.bits()) - 1;
    for (index, value) in dst.iter_mut().enumerate() {
        let bit = index * bits.bits();
        let byte = bit / 8;
        let shift = bit % 8;
        let mut code = (src[byte] >> shift) & mask;
        if shift + bits.bits() > 8 {
            code |= (src[byte + 1] << (8 - shift)) & mask;
        }
        *value = (code as i32 - bits.qmax()) as f32 * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_codecs_round_trip_with_bounded_error() {
        let input = (0..32).map(|i| (i as f32 - 16.0) / 7.0).collect::<Vec<_>>();
        for quant_type in [QuantType::Q2, QuantType::Q3, QuantType::Q4] {
            let q = QuantizedTensor::quantize(&input, 1, input.len(), quant_type, 32);
            assert_eq!(
                q.data.len(),
                match quant_type {
                    QuantType::Q2 => 8,
                    QuantType::Q3 => 12,
                    QuantType::Q4 => 16,
                    QuantType::Q4K => unreachable!(),
                }
            );
            assert!(
                q.relative_l2(&input) < 0.6,
                "{quant_type}: {}",
                q.relative_l2(&input)
            );
        }
    }

    #[test]
    fn q3_uses_three_bytes_per_eight_values() {
        let values = vec![1.0; 8];
        let q = QuantizedTensor::quantize(&values, 1, 8, QuantType::Q3, 8);
        assert_eq!(q.data.len(), 3);
        assert!(q.dequantize().iter().all(|v| (*v - 1.0).abs() < 0.01));
    }

    #[test]
    fn q4k_matches_pinned_llama_reference_block() {
        let values = (0..256)
            .map(|index| 0.6 * (index as f32 * 0.17).sin() + 0.2 * (index as f32 * 0.031).cos())
            .collect::<Vec<_>>();
        let tensor = QuantizedTensor::quantize(&values, 1, 256, QuantType::Q4K, 256);
        let mut actual = Vec::with_capacity(144);
        for scale in &tensor.scales {
            actual.extend_from_slice(&scale.to_bits().to_le_bytes());
        }
        actual.extend_from_slice(&tensor.data);
        let expected_hex = "ca157522fefff5f9e4b2b6bf5d471ccd37495a6b7c8d9daebecededeededecebead9d8c6b5a4837261514030201000013020101001020304151627394a5b6c8d9eaebfcfdfeefefdfcfbfae8e7d6c4b3d8d7c6b4a392726150403020100101020304151628394a6b7c8daebecfdfefffbecedeeeeefdfdfcfbeae9d8c6b5a49382615140302010110102030405161728";
        let expected = expected_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(tensor.relative_l2(&values) < 0.1);
    }

    #[test]
    fn groups_handle_tail_values() {
        let values = (0..37).map(|v| v as f32).collect::<Vec<_>>();
        let q = QuantizedTensor::quantize(&values, 1, values.len(), QuantType::Q4, 32);
        assert_eq!(q.dequantize().len(), values.len());
    }

    #[test]
    fn row_range_preserves_quantized_rows() {
        let rows = 16;
        let cols = 64;
        let values = (0..rows * cols)
            .map(|index| (index as f32 - 32.0) / 11.0)
            .collect::<Vec<_>>();
        let tensor = QuantizedTensor::quantize(&values, rows, cols, QuantType::Q4, 32);
        let subset = tensor.row_range(5, 7).unwrap();
        assert_eq!(subset.layout, PackingLayout::RowMajor);
        assert_eq!(subset.rows, 7);
        assert_eq!(subset.cols, cols);
        let decoded = tensor.dequantize();
        assert_eq!(subset.dequantize(), decoded[5 * cols..12 * cols].to_vec());
        let input = vec![0.25; cols];
        assert_eq!(
            subset.mat_vec(&input),
            tensor.mat_vec(&input)[5..12].to_vec()
        );
    }

    #[test]
    fn metal_tile_layout_preserves_logical_values_and_mat_vec() {
        let rows = 64;
        let cols = 64;
        let values = (0..rows * cols)
            .map(|index| (index as f32 - 100.0) / 17.0)
            .collect::<Vec<_>>();
        let row_major = QuantizedTensor::quantize(&values, rows, cols, QuantType::Q3, 32);
        let tiled = QuantizedTensor::quantize_for_layout(
            &values,
            rows,
            cols,
            QuantType::Q3,
            32,
            PackingLayout::MetalTile32,
        )
        .unwrap();
        assert_ne!(row_major.data, tiled.data);
        assert_eq!(tiled.layout, PackingLayout::MetalTile32);
        assert_eq!(row_major.dequantize(), tiled.dequantize());
        let input = vec![0.25; cols];
        assert_eq!(row_major.mat_vec(&input), tiled.mat_vec(&input));
    }

    #[test]
    fn paired_gate_up_layout_preserves_logical_values_and_mat_vec() {
        let rows = 1024;
        let cols = 1024;
        let values = (0..rows * cols)
            .map(|index| (index as f32 - 100.0) / 17.0)
            .collect::<Vec<_>>();
        let row_major = QuantizedTensor::quantize(&values, rows, cols, QuantType::Q4, 32);
        let (scales, data) = row_major.repack_gate_up_paired().unwrap();
        let paired = QuantizedTensor {
            quant_type: QuantType::Q4,
            layout: PackingLayout::MetalTile32Paired,
            rows,
            cols,
            group_size: 32,
            scales,
            data,
        };
        assert_eq!(row_major.dequantize(), paired.dequantize());
        let input = vec![0.25; cols];
        assert_eq!(row_major.mat_vec(&input), paired.mat_vec(&input));
    }

    #[test]
    fn row_inline_q4_preserves_codes_scales_and_byte_count() {
        let rows = 64;
        let cols = 1024;
        let values = (0..rows * cols)
            .map(|index| (index as f32 - 100.0) / 17.0)
            .collect::<Vec<_>>();
        let split = QuantizedTensor::quantize(&values, rows, cols, QuantType::Q4, 32);
        let mut inline = split.clone();
        inline.layout = PackingLayout::Q4RowInline;
        let stream = inline.row_inline_bytes().unwrap();
        assert_eq!(stream.len(), split.bytes());
        assert_eq!(stream.len(), rows * (32 * 18));
        for row in 0..rows {
            for group in 0..32 {
                let start = row * 32 * 18 + group * 18;
                let data_start = row * 32 * 16 + group * 16;
                assert_eq!(
                    &stream[start..start + 16],
                    &split.data[data_start..data_start + 16]
                );
                assert_eq!(
                    &stream[start + 16..start + 18],
                    &split.scales[row * 32 + group].to_bits().to_le_bytes()
                );
            }
        }
    }
}
