#![allow(unexpected_cfgs)]

mod v2;

use half::f16;
use solana_program::{
    account_info::AccountInfo,
    clock::Clock,
    ed25519_program, entrypoint,
    entrypoint::ProgramResult,
    program_error::ProgramError,
    pubkey::Pubkey,
    sysvar::{instructions as instructions_sysvar, rent::Rent, Sysvar},
};

entrypoint!(process_instruction);

const OP_STAGE_Q4_WEIGHT_V1: u8 = 1;
const OP_VERIFY_STAGED_Q4_FMA_V1: u8 = 2;
const OP_OPEN_RECEIPT_DISPUTE_V1: u8 = 3;
const OP_AGREE_TERMINAL_V1: u8 = 4;
const OP_TIMEOUT_V1: u8 = 5;
const OP_REGISTER_MODEL_V1: u8 = 6;
const OP_CLOSE_EPOCH_V1: u8 = 7;

const Q4_GROUP_SIZE: u32 = 32;
const Q4_TYPE: u8 = 4;
const ACTIVATION_CHUNK_BYTES: usize = 128;
const RECEIPT_BYTES: usize = 210;
const MODEL_ACCOUNT_BYTES: usize = 156;
const EPOCH_ACCOUNT_BYTES: usize = 56;
const DISPUTE_ACCOUNT_BYTES: usize = 304;
const STAGED_WEIGHT_BYTES: usize = 64;
const STAGED_KV_BYTES: usize = 144;

const D_RECEIPT: &[u8] = b"QX:RECEIPT:V1";
const D_BATCH: &[u8] = b"QX:BATCH:V1";
const D_WEIGHT_LEAF: &[u8] = b"QX:WEIGHT:V1";
const D_MERKLE: &[u8] = b"QX:MERKLE:V1";
const D_WEIGHT_TREE: &[u8] = b"QX:WEIGHT:TREE:V1";
const D_ACTIVATION_CHUNK: &[u8] = b"QX:ACTIVATION:CHUNK:V1";

const EPOCH_MAGIC: &[u8; 4] = b"QXE1";
const MODEL_MAGIC: &[u8; 4] = b"QXM1";
const DISPUTE_MAGIC: &[u8; 4] = b"QXD1";
const STAGED_WEIGHT_MAGIC: &[u8; 4] = b"QXW1";
const PHASE_RECEIPT: u8 = 1;
const PHASE_TERMINAL: u8 = 2;
const PHASE_RESOLVED: u8 = 3;
const WINNER_NONE: u8 = 0;
const WINNER_MINER: u8 = 1;
const WINNER_CHALLENGER: u8 = 2;

const RECEIPT_MODEL_OFFSET: usize = 34;
const RECEIPT_TRACE_OFFSET: usize = 130;
const RECEIPT_MINER_OFFSET: usize = 170;
const DISPUTE_PHASE_OFFSET: usize = 4;
const DISPUTE_WINNER_OFFSET: usize = 5;
const DISPUTE_EPOCH_OFFSET: usize = 8;
const DISPUTE_MODEL_OFFSET: usize = 40;
const DISPUTE_RECEIPT_OFFSET: usize = 72;
const DISPUTE_MINER_OFFSET: usize = 104;
const DISPUTE_CHALLENGER_OFFSET: usize = 136;
const DISPUTE_DEFENDER_TRACE_OFFSET: usize = 168;
const DISPUTE_CHALLENGER_TRACE_OFFSET: usize = 200;
const DISPUTE_DEADLINE_OFFSET: usize = 232;
const DISPUTE_WINDOW_OFFSET: usize = 240;
const DISPUTE_ACTIVATION_OFFSET: usize = 248;
const DISPUTE_TENSOR_OFFSET: usize = 280;
const DISPUTE_ROW_OFFSET: usize = 284;
const DISPUTE_COLUMN_OFFSET: usize = 288;
const DISPUTE_ACC_START_OFFSET: usize = 292;
const DISPUTE_DEFENDER_END_OFFSET: usize = 296;
const DISPUTE_CHALLENGER_END_OFFSET: usize = 300;

#[repr(u32)]
enum QxError {
    InvalidEncoding = 1,
    InvalidWeightProof = 2,
    InvalidActivationProof = 3,
    InvalidArithmetic = 4,
    NonFinite = 5,
    InvalidState = 6,
    InvalidReceiptProof = 7,
    InvalidReceiptSignature = 8,
    Deadline = 9,
    Unauthorized = 10,
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
        OP_OPEN_RECEIPT_DISPUTE_V1 => open_receipt_dispute(program_id, accounts, &mut cursor),
        OP_AGREE_TERMINAL_V1 => agree_terminal(program_id, accounts, &mut cursor),
        OP_TIMEOUT_V1 => timeout_dispute(program_id, accounts, &mut cursor),
        OP_REGISTER_MODEL_V1 => register_model(program_id, accounts, &mut cursor),
        OP_CLOSE_EPOCH_V1 => close_epoch(program_id, accounts, &mut cursor),
        v2::OP_REGISTER_JOB_V2 => v2::register_job(program_id, accounts, &mut cursor),
        v2::OP_SUBMIT_EPOCH_V2 => v2::submit_epoch(program_id, accounts, &mut cursor),
        v2::OP_FINALIZE_EPOCH_V2 => v2::finalize_epoch(program_id, accounts, &mut cursor),
        v2::OP_OPEN_CHALLENGE_V2 => v2::open_challenge(program_id, accounts, &mut cursor),
        v2::OP_CHALLENGER_BOUNDS_V2 => {
            v2::submit_challenger_bounds(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_BOUNDS_V2 => v2::submit_miner_bounds(program_id, accounts, &mut cursor),
        v2::OP_CHALLENGER_MIDPOINT_V2 => {
            v2::submit_challenger_midpoint(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_MIDPOINT_V2 => v2::submit_miner_midpoint(program_id, accounts, &mut cursor),
        v2::OP_CHALLENGER_TERMINAL_V2 => {
            v2::submit_challenger_terminal(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_TERMINAL_V2 => v2::submit_miner_terminal(program_id, accounts, &mut cursor),
        v2::OP_STAGE_Q4_WEIGHT_V2 => v2::stage_q4_weight(program_id, accounts, &mut cursor),
        v2::OP_RESOLVE_Q4_FMA_V2 => v2::resolve_q4_fma(program_id, accounts, &mut cursor),
        v2::OP_TIMEOUT_V2 => v2::timeout(program_id, accounts, &mut cursor),
        v2::OP_CHALLENGER_SCALAR_V2 => {
            v2::submit_challenger_scalar(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_SCALAR_V2 => v2::submit_miner_scalar(program_id, accounts, &mut cursor),
        v2::OP_RESOLVE_SCALAR_V2 => v2::resolve_scalar(program_id, accounts, &mut cursor),
        v2::OP_RESOLVE_VM_SCALAR_V2 => v2::resolve_vm_scalar(program_id, accounts, &mut cursor),
        v2::OP_CHALLENGER_TOKEN_V2 => {
            v2::submit_challenger_token(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_TOKEN_V2 => v2::submit_miner_token(program_id, accounts, &mut cursor),
        v2::OP_CHALLENGER_CHECKPOINT_V2 => {
            v2::submit_challenger_checkpoint(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_CHECKPOINT_V2 => {
            v2::submit_miner_checkpoint(program_id, accounts, &mut cursor)
        }
        v2::OP_CHALLENGER_OPERATOR_V2 => {
            v2::submit_challenger_operator(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_OPERATOR_V2 => v2::submit_miner_operator(program_id, accounts, &mut cursor),
        v2::OP_STAGE_VM_Q4_V2 => v2::stage_vm_q4(program_id, accounts, &mut cursor),
        v2::OP_RESOLVE_VM_Q4_V2 => v2::resolve_vm_q4(program_id, accounts, &mut cursor),
        v2::OP_STAGE_VM_KV_V2 => v2::stage_vm_kv(program_id, accounts, &mut cursor),
        v2::OP_RESOLVE_VM_KV_V2 => v2::resolve_vm_kv(program_id, accounts, &mut cursor),
        v2::OP_RESOLVE_VM_ARGMAX_V2 => v2::resolve_vm_argmax(program_id, accounts, &mut cursor),
        v2::OP_CHALLENGER_FINAL_HEAD_V2 => {
            v2::submit_challenger_final_head(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_FINAL_HEAD_V2 => {
            v2::submit_miner_final_head(program_id, accounts, &mut cursor)
        }
        v2::OP_STAGE_VM_SCALAR_V2 => v2::stage_vm_scalar(program_id, accounts, &mut cursor),
        v2::OP_RESOLVE_STAGED_VM_SCALAR_V2 => {
            v2::resolve_staged_vm_scalar(program_id, accounts, &mut cursor)
        }
        v2::OP_RESOLVE_STAGED_VM_Q4_V2 => {
            v2::resolve_staged_vm_q4(program_id, accounts, &mut cursor)
        }
        v2::OP_STAGE_VM_ARGMAX_V2 => v2::stage_vm_argmax(program_id, accounts, &mut cursor),
        v2::OP_RESOLVE_STAGED_VM_ARGMAX_V2 => {
            v2::resolve_staged_vm_argmax(program_id, accounts, &mut cursor)
        }
        v2::OP_STAGE_VM_RAW_WEIGHT_V2 => v2::stage_vm_raw_weight(program_id, accounts, &mut cursor),
        v2::OP_CHALLENGER_MACRO_V2 => {
            v2::submit_challenger_macro(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_MACRO_V2 => v2::submit_miner_macro(program_id, accounts, &mut cursor),
        v2::OP_STAGE_MACRO_RESIDUAL_V2 => {
            v2::stage_macro_residual(program_id, accounts, &mut cursor)
        }
        v2::OP_RESOLVE_MACRO_RESIDUAL_V2 => {
            v2::resolve_macro_residual(program_id, accounts, &mut cursor)
        }
        v2::OP_TESTNET_LOAD_PROBE_V2 => v2::testnet_load_probe(program_id, accounts, &mut cursor),
        v2::OP_CHALLENGER_GRANITE_MICRO_V2 => {
            v2::submit_challenger_granite_micro(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_GRANITE_MICRO_V2 => {
            v2::submit_miner_granite_micro(program_id, accounts, &mut cursor)
        }
        v2::OP_RESOLVE_GRANITE_MICRO_V2 => {
            v2::resolve_granite_micro(program_id, accounts, &mut cursor)
        }
        v2::OP_STAGE_GRANITE_WEIGHT_V2 => {
            v2::stage_granite_weight(program_id, accounts, &mut cursor)
        }
        v2::OP_RESOLVE_MACRO_KV_V2 => v2::resolve_macro_kv(program_id, accounts, &mut cursor),
        v2::OP_RESOLVE_MACRO_TOPK_V2 => v2::resolve_macro_topk(program_id, accounts, &mut cursor),
        v2::OP_STAGE_GRANITE_KV_SCALAR_V2 => {
            v2::stage_granite_kv_scalar(program_id, accounts, &mut cursor)
        }
        v2::OP_CHALLENGER_NESTED_MICRO_V2 => {
            v2::submit_challenger_nested_micro(program_id, accounts, &mut cursor)
        }
        v2::OP_MINER_NESTED_MICRO_V2 => {
            v2::submit_miner_nested_micro(program_id, accounts, &mut cursor)
        }
        v2::OP_STAGE_GRANITE_OUTPUT_V2 => {
            v2::stage_granite_output(program_id, accounts, &mut cursor)
        }
        v2::OP_STAGE_GRANITE_BOUND_V2 => v2::stage_granite_bound(program_id, accounts, &mut cursor),
        v2::OP_EXPIRE_EPOCH_V2 => v2::expire_epoch(program_id, accounts, &mut cursor),
        v2::OP_SETTLE_DISPUTE_V2 => v2::settle_dispute(program_id, accounts, &mut cursor),
        v2::OP_SETTLE_UNCHALLENGED_V2 => v2::settle_unchallenged(program_id, accounts, &mut cursor),
        v2::OP_WITHDRAW_MINER_V2 => v2::withdraw_miner(program_id, accounts, &mut cursor),
        _ => Err(QxError::InvalidEncoding.into()),
    }
}

fn register_model(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || !model.is_signer
        || !model.is_writable
        || model.data_len() != MODEL_ACCOUNT_BYTES
    {
        return Err(QxError::Unauthorized.into());
    }
    let model_id = cursor.array_32()?;
    let weights_root = cursor.array_32()?;
    let weight_leaves = cursor.u32()?;
    let execution_spec = cursor.array_32()?;
    let verification_schedule_root = cursor.array_32()?;
    let verification_schedule_leaves = cursor.u32()?;
    let checkpoint_count = cursor.u32()?;
    let model_layers = cursor.u16()?;
    let checkpoint_interval = cursor.u16()?;
    let max_context_tokens = cursor.u32()?;
    let vocab_size = cursor.u32()?;
    if !cursor.is_empty()
        || weight_leaves == 0
        || execution_spec == [0; 32]
        || verification_schedule_root == [0; 32]
        || verification_schedule_leaves == 0
        || model_layers == 0
        || max_context_tokens == 0
        || vocab_size == 0
        || checkpoint_interval == 0
        || !model_layers.is_multiple_of(checkpoint_interval)
        || checkpoint_count != u32::from(model_layers / checkpoint_interval) + 1
        || verification_schedule_leaves
            != checkpoint_count
                .checked_add(1)
                .ok_or(QxError::InvalidEncoding)?
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = model.try_borrow_mut_data()?;
    if state.iter().any(|byte| *byte != 0) {
        return Err(QxError::InvalidState.into());
    }
    state[..4].copy_from_slice(MODEL_MAGIC);
    put(&mut state, 4, &model_id)?;
    put(&mut state, 36, &weights_root)?;
    put(&mut state, 68, &weight_leaves.to_le_bytes())?;
    put(&mut state, 72, &execution_spec)?;
    put(&mut state, 104, &verification_schedule_root)?;
    put(&mut state, 136, &verification_schedule_leaves.to_le_bytes())?;
    put(&mut state, 140, &checkpoint_count.to_le_bytes())?;
    put(&mut state, 144, &model_layers.to_le_bytes())?;
    put(&mut state, 146, &checkpoint_interval.to_le_bytes())?;
    put(&mut state, 148, &max_context_tokens.to_le_bytes())?;
    put(&mut state, 152, &vocab_size.to_le_bytes())?;
    Ok(())
}

fn close_epoch(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [epoch] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if epoch.owner != program_id
        || !epoch.is_signer
        || !epoch.is_writable
        || epoch.data_len() != EPOCH_ACCOUNT_BYTES
    {
        return Err(QxError::Unauthorized.into());
    }
    let receipt_root = cursor.array_32()?;
    let receipt_count = cursor.u32()?;
    let challenge_deadline = cursor.u64()?;
    let response_window = cursor.u64()?;
    if !cursor.is_empty()
        || receipt_count == 0
        || response_window == 0
        || challenge_deadline <= Clock::get()?.slot
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = epoch.try_borrow_mut_data()?;
    if state.iter().any(|byte| *byte != 0) {
        return Err(QxError::InvalidState.into());
    }
    state[..4].copy_from_slice(EPOCH_MAGIC);
    put(&mut state, 4, &receipt_root)?;
    put(&mut state, 36, &receipt_count.to_le_bytes())?;
    put(&mut state, 40, &challenge_deadline.to_le_bytes())?;
    put(&mut state, 48, &response_window.to_le_bytes())?;
    Ok(())
}

/// Opens a challenge only for a miner-signed receipt included in a live epoch.
/// The Ed25519 precompile must immediately precede this instruction.
fn open_receipt_dispute(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [epoch, model, dispute, challenger, instruction_sysvar] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if epoch.owner != program_id
        || model.owner != program_id
        || dispute.owner != program_id
        || !dispute.is_writable
        || !challenger.is_signer
        || epoch.data_len() != EPOCH_ACCOUNT_BYTES
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.data_len() != DISPUTE_ACCOUNT_BYTES
        || instruction_sysvar.key != &instructions_sysvar::ID
    {
        return Err(QxError::InvalidEncoding.into());
    }
    if dispute.try_borrow_data()?[..4] != [0; 4] {
        return Err(QxError::InvalidState.into());
    }

    let receipt = cursor.take(RECEIPT_BYTES)?;
    let leaf_index = cursor.u32()?;
    let leaf_count = cursor.u32()?;
    let depth = usize::from(cursor.u8()?);
    let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    let challenger_trace = cursor.array_32()?;
    if !cursor.is_empty()
        || u16::from_le_bytes(
            receipt[..2]
                .try_into()
                .map_err(|_| QxError::InvalidEncoding)?,
        ) != 1
        || leaf_count == 0
        || leaf_index >= leaf_count
        || merkle_depth(leaf_count)? != depth
    {
        return Err(QxError::InvalidEncoding.into());
    }

    let epoch_data = epoch.try_borrow_data()?;
    if &epoch_data[..4] != EPOCH_MAGIC {
        return Err(QxError::InvalidState.into());
    }
    let receipt_root = array_at(&epoch_data, 4)?;
    let epoch_receipt_count = u32_at(&epoch_data, 36)?;
    let challenge_deadline = u64_at(&epoch_data, 40)?;
    let response_window = u64_at(&epoch_data, 48)?;
    if epoch_receipt_count != leaf_count || response_window == 0 {
        return Err(QxError::InvalidState.into());
    }
    let slot = Clock::get()?.slot;
    if slot > challenge_deadline {
        return Err(QxError::Deadline.into());
    }
    drop(epoch_data);

    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC
        || receipt[RECEIPT_MODEL_OFFSET..RECEIPT_MODEL_OFFSET + 32] != model_data[4..36]
    {
        return Err(QxError::InvalidState.into());
    }
    drop(model_data);

    let receipt_hash = receipt_hash(receipt);
    let mut node = receipt_hash;
    let mut position = leaf_index;
    for sibling in siblings.chunks_exact(32) {
        let sibling: &[u8; 32] = sibling.try_into().map_err(|_| QxError::InvalidEncoding)?;
        node = if position & 1 == 0 {
            merkle_parent(D_BATCH, &node, sibling)
        } else {
            merkle_parent(D_BATCH, sibling, &node)
        };
        position >>= 1;
    }
    if node != receipt_root {
        return Err(QxError::InvalidReceiptProof.into());
    }

    let miner: [u8; 32] = receipt[RECEIPT_MINER_OFFSET..RECEIPT_MINER_OFFSET + 32]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    if miner == challenger.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    verify_prior_ed25519(instruction_sysvar, &miner, &receipt_hash)?;
    let defender_trace: [u8; 32] = receipt[RECEIPT_TRACE_OFFSET..RECEIPT_TRACE_OFFSET + 32]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    if defender_trace == challenger_trace {
        return Err(QxError::InvalidState.into());
    }
    let deadline = slot
        .checked_add(response_window)
        .ok_or(QxError::InvalidEncoding)?;

    let mut state = dispute.try_borrow_mut_data()?;
    state.fill(0);
    state[..4].copy_from_slice(DISPUTE_MAGIC);
    state[DISPUTE_PHASE_OFFSET] = PHASE_RECEIPT;
    state[DISPUTE_WINNER_OFFSET] = WINNER_NONE;
    put(&mut state, DISPUTE_EPOCH_OFFSET, epoch.key.as_ref())?;
    put(&mut state, DISPUTE_MODEL_OFFSET, model.key.as_ref())?;
    put(&mut state, DISPUTE_RECEIPT_OFFSET, &receipt_hash)?;
    put(&mut state, DISPUTE_MINER_OFFSET, &miner)?;
    put(
        &mut state,
        DISPUTE_CHALLENGER_OFFSET,
        challenger.key.as_ref(),
    )?;
    put(&mut state, DISPUTE_DEFENDER_TRACE_OFFSET, &defender_trace)?;
    put(
        &mut state,
        DISPUTE_CHALLENGER_TRACE_OFFSET,
        &challenger_trace,
    )?;
    put(&mut state, DISPUTE_DEADLINE_OFFSET, &deadline.to_le_bytes())?;
    put(
        &mut state,
        DISPUTE_WINDOW_OFFSET,
        &response_window.to_le_bytes(),
    )?;
    Ok(())
}

/// Records the terminal boundary after both parties agree that narrowing
/// reached these exact coordinates.
fn agree_terminal(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, miner, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_ACCOUNT_BYTES
        || !miner.is_signer
        || !challenger.is_signer
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let activation_root = cursor.array_32()?;
    let tensor_id = cursor.u32()?;
    let row = cursor.u32()?;
    let column = cursor.u32()?;
    let accumulator_start = cursor.u32()?;
    let defender_end = cursor.u32()?;
    let challenger_end = cursor.u32()?;
    if !cursor.is_empty() || defender_end == challenger_end {
        return Err(QxError::InvalidEncoding.into());
    }

    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute_phase(&state, PHASE_RECEIPT)?;
    if state[DISPUTE_MINER_OFFSET..DISPUTE_MINER_OFFSET + 32] != miner.key.to_bytes()
        || state[DISPUTE_CHALLENGER_OFFSET..DISPUTE_CHALLENGER_OFFSET + 32]
            != challenger.key.to_bytes()
    {
        return Err(QxError::Unauthorized.into());
    }
    let slot = Clock::get()?.slot;
    if slot > u64_at(&state, DISPUTE_DEADLINE_OFFSET)? {
        return Err(QxError::Deadline.into());
    }
    let deadline = slot
        .checked_add(u64_at(&state, DISPUTE_WINDOW_OFFSET)?)
        .ok_or(QxError::InvalidEncoding)?;
    state[DISPUTE_PHASE_OFFSET] = PHASE_TERMINAL;
    put(&mut state, DISPUTE_DEADLINE_OFFSET, &deadline.to_le_bytes())?;
    put(&mut state, DISPUTE_ACTIVATION_OFFSET, &activation_root)?;
    put(&mut state, DISPUTE_TENSOR_OFFSET, &tensor_id.to_le_bytes())?;
    put(&mut state, DISPUTE_ROW_OFFSET, &row.to_le_bytes())?;
    put(&mut state, DISPUTE_COLUMN_OFFSET, &column.to_le_bytes())?;
    put(
        &mut state,
        DISPUTE_ACC_START_OFFSET,
        &accumulator_start.to_le_bytes(),
    )?;
    put(
        &mut state,
        DISPUTE_DEFENDER_END_OFFSET,
        &defender_end.to_le_bytes(),
    )?;
    put(
        &mut state,
        DISPUTE_CHALLENGER_END_OFFSET,
        &challenger_end.to_le_bytes(),
    )?;
    Ok(())
}

fn stage_q4_weight(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || dispute.owner != program_id
        || staged.owner != program_id
        || !staged.is_writable
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.data_len() != DISPUTE_ACCOUNT_BYTES
        || staged.data_len() != STAGED_WEIGHT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let dispute_data = dispute.try_borrow_data()?;
    require_dispute_phase(&dispute_data, PHASE_TERMINAL)?;
    if dispute_data[DISPUTE_MODEL_OFFSET..DISPUTE_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let expected_tensor = u32_at(&dispute_data, DISPUTE_TENSOR_OFFSET)?;
    let expected_row = u32_at(&dispute_data, DISPUTE_ROW_OFFSET)?;
    let expected_column = u32_at(&dispute_data, DISPUTE_COLUMN_OFFSET)?;
    drop(dispute_data);

    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC {
        return Err(QxError::InvalidState.into());
    }
    let expected_root = array_at(&model_data, 36)?;
    let leaf_count = u32_at(&model_data, 68)?;
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
        || tensor_id != expected_tensor
        || row != expected_row
        || column != expected_column
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut node = weight_leaf_hash(tensor_id, row, group, quantization_type, packed, scale_bits);
    let mut position = leaf_index;
    for sibling in siblings.chunks_exact(32) {
        let sibling: &[u8; 32] = sibling.try_into().map_err(|_| QxError::InvalidEncoding)?;
        node = if position & 1 == 0 {
            merkle_parent(D_WEIGHT_TREE, &node, sibling)
        } else {
            merkle_parent(D_WEIGHT_TREE, sibling, &node)
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
    staged_data.fill(0);
    staged_data[..4].copy_from_slice(STAGED_WEIGHT_MAGIC);
    put(&mut staged_data, 4, dispute.key.as_ref())?;
    put(&mut staged_data, 36, &tensor_id.to_le_bytes())?;
    put(&mut staged_data, 40, &row.to_le_bytes())?;
    put(&mut staged_data, 44, &column.to_le_bytes())?;
    put(&mut staged_data, 48, &weight_bits.to_le_bytes())?;
    Ok(())
}

fn verify_staged_q4_fma(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if dispute.owner != program_id
        || staged.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_ACCOUNT_BYTES
        || staged.data_len() != STAGED_WEIGHT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let staged_data = staged.try_borrow_data()?;
    if &staged_data[..4] != STAGED_WEIGHT_MAGIC || staged_data[4..36] != dispute.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let column = u32_at(&staged_data, 44)?;
    let weight = f32::from_bits(u32_at(&staged_data, 48)?);
    drop(staged_data);

    let mut dispute_data = dispute.try_borrow_mut_data()?;
    require_dispute_phase(&dispute_data, PHASE_TERMINAL)?;
    if Clock::get()?.slot > u64_at(&dispute_data, DISPUTE_DEADLINE_OFFSET)? {
        return Err(QxError::Deadline.into());
    }
    let expected_activation_root = array_at(&dispute_data, DISPUTE_ACTIVATION_OFFSET)?;
    if column != u32_at(&dispute_data, DISPUTE_COLUMN_OFFSET)? {
        return Err(QxError::InvalidState.into());
    }

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
    let accumulator = f32::from_bits(u32_at(&dispute_data, DISPUTE_ACC_START_OFFSET)?);
    if !activation.is_finite() || !accumulator.is_finite() || !weight.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let result = weight.mul_add(activation, accumulator);
    if !result.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let defender_end = u32_at(&dispute_data, DISPUTE_DEFENDER_END_OFFSET)?;
    let challenger_end = u32_at(&dispute_data, DISPUTE_CHALLENGER_END_OFFSET)?;
    let agrees_with_defender = result == f32::from_bits(defender_end);
    let agrees_with_challenger = result == f32::from_bits(challenger_end);
    let winner = match (agrees_with_defender, agrees_with_challenger) {
        (true, false) => WINNER_MINER,
        (false, true) => WINNER_CHALLENGER,
        _ => return Err(QxError::InvalidArithmetic.into()),
    };
    dispute_data[DISPUTE_PHASE_OFFSET] = PHASE_RESOLVED;
    dispute_data[DISPUTE_WINNER_OFFSET] = winner;
    Ok(())
}

fn timeout_dispute(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if !cursor.is_empty()
        || dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = dispute.try_borrow_mut_data()?;
    if &state[..4] != DISPUTE_MAGIC
        || !matches!(state[DISPUTE_PHASE_OFFSET], PHASE_RECEIPT | PHASE_TERMINAL)
    {
        return Err(QxError::InvalidState.into());
    }
    if Clock::get()?.slot <= u64_at(&state, DISPUTE_DEADLINE_OFFSET)? {
        return Err(QxError::Deadline.into());
    }
    // Until interactive bisection is added, the challenger bears the burden
    // of reaching and proving a terminal claim before the deadline.
    state[DISPUTE_PHASE_OFFSET] = PHASE_RESOLVED;
    state[DISPUTE_WINNER_OFFSET] = WINNER_MINER;
    Ok(())
}

fn verify_prior_ed25519(
    instructions: &AccountInfo,
    expected_pubkey: &[u8; 32],
    expected_message: &[u8; 32],
) -> ProgramResult {
    let current = instructions_sysvar::load_current_index_checked(instructions)?;
    let prior_index = current
        .checked_sub(1)
        .ok_or(QxError::InvalidReceiptSignature)?;
    let prior =
        instructions_sysvar::load_instruction_at_checked(prior_index as usize, instructions)?;
    if prior.program_id != ed25519_program::ID || prior.data.len() < 16 || prior.data[0] != 1 {
        return Err(QxError::InvalidReceiptSignature.into());
    }
    let signature_instruction = u16_from(&prior.data, 4)?;
    let public_key_offset = usize::from(u16_from(&prior.data, 6)?);
    let public_key_instruction = u16_from(&prior.data, 8)?;
    let message_offset = usize::from(u16_from(&prior.data, 10)?);
    let message_size = usize::from(u16_from(&prior.data, 12)?);
    let message_instruction = u16_from(&prior.data, 14)?;
    if signature_instruction != u16::MAX
        || public_key_instruction != u16::MAX
        || message_instruction != u16::MAX
        || message_size != 32
        || prior.data.get(public_key_offset..public_key_offset + 32) != Some(expected_pubkey)
        || prior.data.get(message_offset..message_offset + 32) != Some(expected_message)
    {
        return Err(QxError::InvalidReceiptSignature.into());
    }
    Ok(())
}

fn receipt_hash(receipt: &[u8]) -> [u8; 32] {
    let length = (receipt.len() as u64).to_le_bytes();
    solana_sha256_hasher::hashv(&[D_RECEIPT, &length, receipt]).to_bytes()
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

fn merkle_parent(tree_domain: &[u8], left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let tree_len = (tree_domain.len() as u64).to_le_bytes();
    let digest_len = 32_u64.to_le_bytes();
    solana_sha256_hasher::hashv(&[
        D_MERKLE,
        &tree_len,
        tree_domain,
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
    stable_hashv(&[D_ACTIVATION_CHUNK, &index_len, &index, &chunk_len, chunk])
}

fn raw_blake3_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    stable_hashv(&[left, right])
}

/// Evaluate BLAKE3 in SBF until its syscall is active on every target cluster.
/// Disputes pay this cost; normal inference retains fast GPU commitments.
fn stable_hashv(parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    *hasher.finalize().as_bytes()
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

fn require_dispute_phase(state: &[u8], phase: u8) -> ProgramResult {
    if state.len() != DISPUTE_ACCOUNT_BYTES
        || &state[..4] != DISPUTE_MAGIC
        || state[DISPUTE_PHASE_OFFSET] != phase
        || state[DISPUTE_WINNER_OFFSET] != WINNER_NONE
    {
        return Err(QxError::InvalidState.into());
    }
    Ok(())
}

fn put(target: &mut [u8], offset: usize, value: &[u8]) -> ProgramResult {
    target
        .get_mut(offset..offset + value.len())
        .ok_or(QxError::InvalidEncoding)?
        .copy_from_slice(value);
    Ok(())
}

fn array_at(bytes: &[u8], offset: usize) -> Result<[u8; 32], ProgramError> {
    bytes
        .get(offset..offset + 32)
        .ok_or(QxError::InvalidEncoding)?
        .try_into()
        .map_err(|_| QxError::InvalidEncoding.into())
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, ProgramError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or(QxError::InvalidEncoding)?
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    ))
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, ProgramError> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or(QxError::InvalidEncoding)?
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    ))
}

fn u16_from(bytes: &[u8], offset: usize) -> Result<u16, ProgramError> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or(QxError::InvalidReceiptSignature)?
            .try_into()
            .map_err(|_| QxError::InvalidReceiptSignature)?,
    ))
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

    fn u64(&mut self) -> Result<u64, ProgramError> {
        Ok(u64::from_le_bytes(
            self.take(8)?
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
