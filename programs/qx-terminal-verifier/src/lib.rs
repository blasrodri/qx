#![allow(unexpected_cfgs)]

use half::f16;
use solana_program::{
    account_info::AccountInfo, entrypoint, entrypoint::ProgramResult, program_error::ProgramError,
    pubkey::Pubkey,
};

entrypoint!(process_instruction);

const OP_STAGE_Q4_WEIGHT_V1: u8 = 1;
const OP_VERIFY_STAGED_Q4_FMA_V1: u8 = 2;
const Q4_GROUP_SIZE: u32 = 32;
const Q4_TYPE: u8 = 4;
const ACTIVATION_CHUNK_BYTES: usize = 128;
const D_WEIGHT_LEAF: &[u8] = b"QX:WEIGHT:V1";
const D_MERKLE: &[u8] = b"QX:MERKLE:V1";
const D_WEIGHT_TREE: &[u8] = b"QX:WEIGHT:TREE:V1";
const D_ACTIVATION_CHUNK: &[u8] = b"QX:ACTIVATION:CHUNK:V1";
const MODEL_ACCOUNT_BYTES: usize = 36;
const STAGED_WEIGHT_BYTES: usize = 20;
const STAGED_WEIGHT_MAGIC: &[u8; 4] = b"QXW1";

#[repr(u32)]
enum QxError {
    InvalidEncoding = 1,
    InvalidWeightProof = 2,
    InvalidActivationProof = 3,
    InvalidArithmetic = 4,
    NonFinite = 5,
}

impl From<QxError> for ProgramError {
    fn from(value: QxError) -> Self {
        Self::Custom(value as u32)
    }
}

pub fn process_instruction(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    data: &[u8],
) -> ProgramResult {
    let mut cursor = Cursor::new(data);
    match cursor.u8()? {
        OP_STAGE_Q4_WEIGHT_V1 => stage_q4_weight(program_id, accounts, &mut cursor),
        OP_VERIFY_STAGED_Q4_FMA_V1 => verify_staged_q4_fma(program_id, accounts, &mut cursor),
        _ => Err(QxError::InvalidEncoding.into()),
    }
}

fn stage_q4_weight(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || staged.owner != program_id
        || !staged.is_writable
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || staged.data_len() != STAGED_WEIGHT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let model_data = model.try_borrow_data()?;
    let expected_root: [u8; 32] = model_data[..32]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let leaf_count = u32::from_le_bytes(
        model_data[32..36]
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    );
    drop(model_data);

    let column = cursor.u32()?;
    let leaf_index = cursor.u32()?;
    let tensor_id = cursor.u32()?;
    let row = cursor.u32()?;
    let group = cursor.u32()?;
    let quantization_type = cursor.u8()?;
    let scale_bits = cursor.u16()?;
    let packed = cursor.take(16)?;
    let depth = usize::from(cursor.u8()?);
    let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    if !cursor.is_empty()
        || leaf_count == 0
        || leaf_index >= leaf_count
        || merkle_depth(leaf_count)? != depth
        || group != column / Q4_GROUP_SIZE
        || quantization_type != Q4_TYPE
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut node = weight_leaf_hash(tensor_id, row, group, quantization_type, packed, scale_bits);
    let mut position = leaf_index;
    for sibling in siblings.chunks_exact(32) {
        let sibling: &[u8; 32] = sibling.try_into().map_err(|_| QxError::InvalidEncoding)?;
        node = if position & 1 == 0 {
            weight_parent(&node, sibling)
        } else {
            weight_parent(sibling, &node)
        };
        position >>= 1;
    }
    if node != expected_root {
        return Err(QxError::InvalidWeightProof.into());
    }
    let local_weight =
        usize::try_from(column % Q4_GROUP_SIZE).map_err(|_| QxError::InvalidEncoding)?;
    let byte = packed[local_weight / 2];
    let code = if local_weight & 1 == 0 {
        byte & 0x0f
    } else {
        byte >> 4
    };
    let scale = f16::from_bits(scale_bits).to_f32();
    if !scale.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let weight_bits = ((code as i8 - 7) as f32 * scale).to_bits();
    let mut staged_data = staged.try_borrow_mut_data()?;
    staged_data[..4].copy_from_slice(STAGED_WEIGHT_MAGIC);
    staged_data[4..8].copy_from_slice(&tensor_id.to_le_bytes());
    staged_data[8..12].copy_from_slice(&row.to_le_bytes());
    staged_data[12..16].copy_from_slice(&column.to_le_bytes());
    staged_data[16..20].copy_from_slice(&weight_bits.to_le_bytes());
    Ok(())
}

fn verify_staged_q4_fma(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if staged.owner != program_id || staged.data_len() != STAGED_WEIGHT_BYTES {
        return Err(QxError::InvalidEncoding.into());
    }
    let staged_data = staged.try_borrow_data()?;
    if &staged_data[..4] != STAGED_WEIGHT_MAGIC {
        return Err(QxError::InvalidEncoding.into());
    }
    let column = u32::from_le_bytes(
        staged_data[12..16]
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    );
    let weight = f32::from_bits(u32::from_le_bytes(
        staged_data[16..20]
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    ));
    drop(staged_data);

    let expected_activation_root = cursor.array_32()?;
    let accumulator_start_bits = cursor.u32()?;
    let accumulator_end_bits = cursor.u32()?;
    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let depth = usize::from(cursor.u8()?);
    let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    if !cursor.is_empty() || byte_length == 0 {
        return Err(QxError::InvalidEncoding.into());
    }
    let leaf_count = byte_length.div_ceil(ACTIVATION_CHUNK_BYTES as u32);
    if chunk_index >= leaf_count || merkle_depth(leaf_count)? != depth {
        return Err(QxError::InvalidEncoding.into());
    }
    let expected_len = if chunk_index + 1 == leaf_count {
        usize::try_from(byte_length - chunk_index * ACTIVATION_CHUNK_BYTES as u32)
            .map_err(|_| QxError::InvalidEncoding)?
    } else {
        ACTIVATION_CHUNK_BYTES
    };
    if chunk_len != expected_len {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut node = activation_leaf_hash(chunk_index, chunk);
    let mut position = chunk_index;
    for sibling in siblings.chunks_exact(32) {
        let sibling: &[u8; 32] = sibling.try_into().map_err(|_| QxError::InvalidEncoding)?;
        node = if position & 1 == 0 {
            raw_blake3_parent(&node, sibling)
        } else {
            raw_blake3_parent(sibling, &node)
        };
        position >>= 1;
    }
    if node != expected_activation_root {
        return Err(QxError::InvalidActivationProof.into());
    }
    let activation_byte = usize::try_from(column)
        .map_err(|_| QxError::InvalidEncoding)?
        .checked_mul(4)
        .ok_or(QxError::InvalidEncoding)?;
    let chunk_start = usize::try_from(chunk_index)
        .map_err(|_| QxError::InvalidEncoding)?
        .checked_mul(ACTIVATION_CHUNK_BYTES)
        .ok_or(QxError::InvalidEncoding)?;
    let local = activation_byte
        .checked_sub(chunk_start)
        .ok_or(QxError::InvalidEncoding)?;
    let activation = f32::from_bits(u32::from_le_bytes(
        chunk
            .get(local..local + 4)
            .ok_or(QxError::InvalidEncoding)?
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    ));
    let accumulator = f32::from_bits(accumulator_start_bits);
    if !activation.is_finite() || !accumulator.is_finite() || !weight.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let result = weight.mul_add(activation, accumulator);
    if !result.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    if result.to_bits() != accumulator_end_bits {
        return Err(QxError::InvalidArithmetic.into());
    }
    Ok(())
}

fn weight_leaf_hash(
    tensor_id: u32,
    row: u32,
    group: u32,
    quantization_type: u8,
    packed: &[u8],
    scale_bits: u16,
) -> [u8; 32] {
    let tensor_id = tensor_id.to_le_bytes();
    let row = row.to_le_bytes();
    let group = group.to_le_bytes();
    let quantization_type = [quantization_type];
    let scale_bits = scale_bits.to_le_bytes();
    let len4 = 4_u64.to_le_bytes();
    let len1 = 1_u64.to_le_bytes();
    let len_packed = (packed.len() as u64).to_le_bytes();
    let len2 = 2_u64.to_le_bytes();
    solana_sha256_hasher::hashv(&[
        D_WEIGHT_LEAF,
        &len4,
        &tensor_id,
        &len4,
        &row,
        &len4,
        &group,
        &len1,
        &quantization_type,
        &len_packed,
        packed,
        &len2,
        &scale_bits,
    ])
    .to_bytes()
}

fn weight_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let tree_len = (D_WEIGHT_TREE.len() as u64).to_le_bytes();
    let digest_len = 32_u64.to_le_bytes();
    solana_sha256_hasher::hashv(&[
        D_MERKLE,
        &tree_len,
        D_WEIGHT_TREE,
        &digest_len,
        left,
        &digest_len,
        right,
    ])
    .to_bytes()
}

fn activation_leaf_hash(index: u32, chunk: &[u8]) -> [u8; 32] {
    let index = index.to_le_bytes();
    let index_len = 4_u64.to_le_bytes();
    let chunk_len = (chunk.len() as u64).to_le_bytes();
    solana_blake3_hasher::hashv(&[D_ACTIVATION_CHUNK, &index_len, &index, &chunk_len, chunk])
        .to_bytes()
}

fn raw_blake3_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    solana_blake3_hasher::hashv(&[left, right]).to_bytes()
}

fn merkle_depth(leaf_count: u32) -> Result<usize, ProgramError> {
    if leaf_count == 0 {
        return Err(QxError::InvalidEncoding.into());
    }
    let capacity = leaf_count
        .checked_next_power_of_two()
        .ok_or(QxError::InvalidEncoding)?;
    Ok(capacity.trailing_zeros() as usize)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], ProgramError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(QxError::InvalidEncoding)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(QxError::InvalidEncoding)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, ProgramError> {
        Ok(*self.take(1)?.first().ok_or(QxError::InvalidEncoding)?)
    }

    fn u16(&mut self) -> Result<u16, ProgramError> {
        Ok(u16::from_le_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| QxError::InvalidEncoding)?,
        ))
    }

    fn u32(&mut self) -> Result<u32, ProgramError> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| QxError::InvalidEncoding)?,
        ))
    }

    fn array_32(&mut self) -> Result<[u8; 32], ProgramError> {
        self.take(32)?
            .try_into()
            .map_err(|_| QxError::InvalidEncoding.into())
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
