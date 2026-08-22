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
}

impl PackingLayout {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::RowMajor),
            1 => Some(Self::MetalTile32),
            2 => Some(Self::MetalTile32Paired),
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
            _ => None,
        }
    }
}

impl std::fmt::Display for QuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "q{}", self.bits())
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
        }
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
    (values * quant_type.bits()).div_ceil(8)
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
}
