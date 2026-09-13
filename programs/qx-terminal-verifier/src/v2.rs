//! Adversarial V2 protocol path. Results are attributed by one miner-signed
//! epoch submission; individual result leaves are unsigned. Dispute moves are
//! unilateral and a timeout penalizes the party whose response is pending.

use super::*;
use solana_program::sysvar::slot_hashes::PodSlotHashes;

pub(super) const OP_REGISTER_JOB_V2: u8 = 8;
pub(super) const OP_SUBMIT_EPOCH_V2: u8 = 9;
pub(super) const OP_FINALIZE_EPOCH_V2: u8 = 10;
pub(super) const OP_OPEN_CHALLENGE_V2: u8 = 11;
pub(super) const OP_CHALLENGER_BOUNDS_V2: u8 = 12;
pub(super) const OP_MINER_BOUNDS_V2: u8 = 13;
pub(super) const OP_CHALLENGER_MIDPOINT_V2: u8 = 14;
pub(super) const OP_MINER_MIDPOINT_V2: u8 = 15;
pub(super) const OP_CHALLENGER_TERMINAL_V2: u8 = 16;
pub(super) const OP_MINER_TERMINAL_V2: u8 = 17;
pub(super) const OP_STAGE_Q4_WEIGHT_V2: u8 = 18;
pub(super) const OP_RESOLVE_Q4_FMA_V2: u8 = 19;
pub(super) const OP_TIMEOUT_V2: u8 = 20;
pub(super) const OP_CHALLENGER_SCALAR_V2: u8 = 21;
pub(super) const OP_MINER_SCALAR_V2: u8 = 22;
pub(super) const OP_RESOLVE_SCALAR_V2: u8 = 23;
pub(super) const OP_RESOLVE_VM_SCALAR_V2: u8 = 24;
pub(super) const OP_CHALLENGER_TOKEN_V2: u8 = 25;
pub(super) const OP_MINER_TOKEN_V2: u8 = 26;
pub(super) const OP_CHALLENGER_CHECKPOINT_V2: u8 = 27;
pub(super) const OP_MINER_CHECKPOINT_V2: u8 = 28;
pub(super) const OP_CHALLENGER_OPERATOR_V2: u8 = 29;
pub(super) const OP_MINER_OPERATOR_V2: u8 = 30;
pub(super) const OP_STAGE_VM_Q4_V2: u8 = 31;
pub(super) const OP_RESOLVE_VM_Q4_V2: u8 = 32;
pub(super) const OP_STAGE_VM_KV_V2: u8 = 33;
pub(super) const OP_RESOLVE_VM_KV_V2: u8 = 34;
pub(super) const OP_RESOLVE_VM_ARGMAX_V2: u8 = 35;
pub(super) const OP_CHALLENGER_FINAL_HEAD_V2: u8 = 36;
pub(super) const OP_MINER_FINAL_HEAD_V2: u8 = 37;
pub(super) const OP_STAGE_VM_SCALAR_V2: u8 = 38;
pub(super) const OP_RESOLVE_STAGED_VM_SCALAR_V2: u8 = 39;
pub(super) const OP_RESOLVE_STAGED_VM_Q4_V2: u8 = 40;
pub(super) const OP_STAGE_VM_ARGMAX_V2: u8 = 41;
pub(super) const OP_RESOLVE_STAGED_VM_ARGMAX_V2: u8 = 42;
pub(super) const OP_STAGE_VM_RAW_WEIGHT_V2: u8 = 43;
pub(super) const OP_CHALLENGER_MACRO_V2: u8 = 44;
pub(super) const OP_MINER_MACRO_V2: u8 = 45;
pub(super) const OP_STAGE_MACRO_RESIDUAL_V2: u8 = 46;
pub(super) const OP_RESOLVE_MACRO_RESIDUAL_V2: u8 = 47;
pub(super) const OP_TESTNET_LOAD_PROBE_V2: u8 = 48;

pub const JOB_PAYLOAD_BYTES: usize = 310;
pub const JOB_ACCOUNT_BYTES: usize = 314;
pub const RESULT_LEAF_BYTES: usize = 104;
pub const EPOCH_V2_ACCOUNT_BYTES: usize = 152;
pub const DISPUTE_V2_ACCOUNT_BYTES: usize = 768;
pub const MINER_V2_ACCOUNT_BYTES: usize = 16;
pub const STAGED_SCALAR_V2_ACCOUNT_BYTES: usize = 104;
pub const STAGED_ARGMAX_V2_ACCOUNT_BYTES: usize = 108;
pub const STAGED_MACRO_SCALAR_V2_ACCOUNT_BYTES: usize = 108;
/// Roughly one day at a 400 ms slot. This is a consensus parameter, not a
/// value the submitting miner may shorten.
pub const CHALLENGE_WINDOW_SLOTS: u64 = 216_000;
/// Roughly eight minutes at a 400 ms slot for each interactive response.
pub const RESPONSE_WINDOW_SLOTS: u64 = 1_200;
pub const MIN_AUDIT_PROBABILITY_PPM: u32 = 10_000;

const JOB_MAGIC: &[u8; 4] = b"QXJ2";
const EPOCH_MAGIC_V2: &[u8; 4] = b"QXE2";
const DISPUTE_MAGIC_V2: &[u8; 4] = b"QXD2";
const MINER_MAGIC_V2: &[u8; 4] = b"QXR2";

const EPOCH_PHASE_OFFSET: usize = 4;
const EPOCH_NUMBER_OFFSET: usize = 8;
const EPOCH_MINER_OFFSET: usize = 16;
const EPOCH_ROOT_OFFSET: usize = 48;
const EPOCH_COUNT_OFFSET: usize = 80;
const EPOCH_CLOSE_SLOT_OFFSET: usize = 88;
const EPOCH_ENTROPY_SLOT_OFFSET: usize = 96;
const EPOCH_SEED_OFFSET: usize = 104;
const EPOCH_CHALLENGE_DEADLINE_OFFSET: usize = 136;
const EPOCH_WINDOW_OFFSET: usize = 144;
const EPOCH_CLOSED: u8 = 1;
const EPOCH_SEEDED: u8 = 2;

const JOB_INSTANCE_OFFSET: usize = 2;
const JOB_ID_OFFSET: usize = 34;
const JOB_REQUESTER_OFFSET: usize = 66;
const JOB_MINER_OFFSET: usize = 98;
const JOB_MODEL_OFFSET: usize = 130;
const JOB_EXECUTION_SPEC_OFFSET: usize = 162;
const JOB_INPUT_HASH_OFFSET: usize = 194;
const JOB_INPUT_COUNT_OFFSET: usize = 226;
const JOB_MAX_OUTPUT_OFFSET: usize = 230;
const JOB_EXPOSURE_OFFSET: usize = 242;
const JOB_DEADLINE_OFFSET: usize = 250;
const JOB_AUDIT_PPM_OFFSET: usize = 266;
const JOB_EPOCH_OFFSET: usize = 270;
const JOB_MODEL_REGISTRY_OFFSET: usize = 278;

const RESULT_JOB_OFFSET: usize = 0;
const RESULT_TRACE_OFFSET: usize = 64;
const RESULT_OUTPUT_COUNT_OFFSET: usize = 96;
const RESULT_TRACE_LENGTH_OFFSET: usize = 100;

const D_JOB_PAYLOAD: &[u8] = b"QX:JOB:PAYLOAD:V1";
const D_RESULT: &[u8] = b"QX:RESULT:V1";
const D_AUDIT_SEED: &[u8] = b"QX:AUDIT:SEED:V2";
const D_AUDIT_DRAW: &[u8] = b"QX:AUDIT:DRAW:V1";
// The outer bisection opens the production job tree directly. Nested token
// and operator trees use explicit domains in their descent instructions.
const D_TRACE_TREE: &[u8] = b"QX:JOB:V1";
const D_JOB_GENESIS: &[u8] = b"QX:JOB:GENESIS:V1";
const D_TOKEN_TREE: &[u8] = b"QX:TOKEN:V1";
const D_OPERATOR_TREE: &[u8] = b"QX:OPERATOR:V1";
const D_TOKEN_RESULT: &[u8] = b"QX:TOKEN:RESULT:V1";
const D_TOKEN_INPUT_STATE: &[u8] = b"QX:TOKEN:INPUT:STATE:V1";
const D_CHECKPOINT: &[u8] = b"QX:CHECKPOINT:V1";
const D_KV: &[u8] = b"QX:KV:V1";
const D_FMA_STATE: &[u8] = b"QX:FMA:STATE:V1";
const D_SCALAR_STATE: &[u8] = b"QX:SCALAR:STATE:V1";
const D_PROGRAM_TREE: &[u8] = b"QX:PROGRAM:TREE:V1";
const D_PROGRAM_INSTRUCTION: &[u8] = b"QX:PROGRAM:INSTRUCTION:V1";
const D_VM_STATE: &[u8] = b"QX:VM:STATE:V1";
const D_VM_OUTPUT: &[u8] = b"QX:VM:OUTPUT:V1";
const D_VM_INSTRUCTION: &[u8] = b"QX:VM:INSTRUCTION:V1";
const D_VM_SEGMENT_PROGRAM: &[u8] = b"QX:VM:SEGMENT:V1";
const D_SEGMENT_PROGRAM_TREE: &[u8] = b"QX:SEGMENT:PROGRAM:TREE:V1";
const D_STATE_CHUNK: &[u8] = b"QX:STATE:CHUNK:V1";
const D_MERKLE_EMPTY: &[u8] = b"QX:MERKLE:EMPTY:V1";
const D_INPUT_TOKEN: &[u8] = b"QX:INPUT:TOKEN:V1";
const D_INPUT_TREE: &[u8] = b"QX:INPUT:TREE:V1";
const D_MACRO_STATE: &[u8] = b"QX:MACRO:STATE:V1";
const D_MACRO_WORKSPACE: &[u8] = b"QX:MACRO:WORKSPACE:V1";
const D_MACRO_WORKSPACE_SLOT: &[u8] = b"QX:MACRO:WORKSPACE:SLOT:V1";
const D_TESTNET_LOAD: &[u8] = b"QX:TESTNET:LOAD:V1";
const STATE_CHUNK_BYTES: usize = 256;
const STAGED_KV_MAGIC: &[u8; 4] = b"QXK2";
const STAGED_SCALAR_MAGIC: &[u8; 4] = b"QXS2";
const STAGED_ARGMAX_MAGIC: &[u8; 4] = b"QXA2";
const STAGED_MACRO_SCALAR_MAGIC: &[u8; 4] = b"QXM2";

const PHASE_CHALLENGER_BOUNDS: u8 = 1;
const PHASE_MINER_BOUNDS: u8 = 2;
const PHASE_CHALLENGER_MIDPOINT: u8 = 3;
const PHASE_MINER_MIDPOINT: u8 = 4;
const PHASE_CHALLENGER_TERMINAL: u8 = 5;
const PHASE_MINER_TERMINAL: u8 = 6;
const PHASE_TERMINAL_READY: u8 = 7;
const PHASE_RESOLVED_V2: u8 = 8;
const PHASE_CHALLENGER_TOKEN: u8 = 9;
const PHASE_MINER_TOKEN: u8 = 10;
const PHASE_CHALLENGER_CHECKPOINT: u8 = 11;
const PHASE_MINER_CHECKPOINT: u8 = 12;
const PHASE_CHALLENGER_OPERATOR_ROOT: u8 = 13;
const PHASE_MINER_OPERATOR_ROOT: u8 = 14;
const PHASE_CHALLENGER_FINAL_HEAD: u8 = 16;
const PHASE_MINER_FINAL_HEAD: u8 = 17;
const PHASE_MINER_MACRO: u8 = 18;
const PHASE_MACRO_READY: u8 = 19;
const PARTY_NONE: u8 = 0;
const PARTY_MINER: u8 = 1;
const PARTY_CHALLENGER: u8 = 2;

const D_PHASE_OFFSET: usize = 4;
const D_WINNER_OFFSET: usize = 5;
const D_EXPECTED_PARTY_OFFSET: usize = 6;
const D_RESOLUTION_FLAGS_OFFSET: usize = 7;
const D_EPOCH_OFFSET: usize = 8;
const D_MODEL_OFFSET: usize = 40;
const D_JOB_HASH_OFFSET: usize = 72;
const D_MINER_OFFSET: usize = 104;
const D_CHALLENGER_OFFSET: usize = 136;
const D_DEFENDER_ROOT_OFFSET: usize = 168;
const D_CHALLENGER_ROOT_OFFSET: usize = 200;
const D_COMMON_ROOT_OFFSET: usize = 232;
const D_DEFENDER_END_OFFSET: usize = 264;
const D_CHALLENGER_END_OFFSET: usize = 296;
const D_PENDING_CHALLENGER_OFFSET: usize = 328;
const D_DEADLINE_OFFSET: usize = 360;
const D_WINDOW_OFFSET: usize = 368;
const D_LOW_OFFSET: usize = 376;
const D_HIGH_OFFSET: usize = 380;
const D_TRACE_LENGTH_OFFSET: usize = 384;
const D_MIDPOINT_OFFSET: usize = 388;
const D_ACTIVATION_OFFSET: usize = 392;
const D_TENSOR_OFFSET: usize = 424;
const D_ROW_OFFSET: usize = 428;
const D_COLUMN_OFFSET: usize = 432;
const D_ACC_START_OFFSET: usize = 436;
const D_DEFENDER_BITS_OFFSET: usize = 440;
const D_CHALLENGER_BITS_OFFSET: usize = 444;
const D_JOB_ID_OFFSET: usize = 448;
const D_CHALLENGER_TOKEN_ROOT_OFFSET: usize = 480;
const D_TOKEN_COUNT_OFFSET: usize = 512;
const D_TOKEN_INDEX_OFFSET: usize = 516;
const D_CHALLENGER_PREDICTION_OFFSET: usize = 520;
const D_DEFENDER_PREDICTION_OFFSET: usize = 524;
const D_CHECKPOINT_LAYER_OFFSET: usize = 528;
const D_CHALLENGER_RESIDUAL_OFFSET: usize = 536;
const D_CHALLENGER_KV_OFFSET: usize = 568;
const D_DEFENDER_RESIDUAL_OFFSET: usize = 600;
const D_DEFENDER_KV_OFFSET: usize = 632;
const D_COMMON_RESIDUAL_OFFSET: usize = 664;
const D_COMMON_KV_OFFSET: usize = 696;
const D_SEGMENT_PROGRAM_ROOT_OFFSET: usize = 728;
const D_SEGMENT_INSTRUCTION_COUNT_OFFSET: usize = 760;

const TRACE_KIND_JOB: u8 = 1;
const TRACE_KIND_TOKEN: u8 = 2;
const TRACE_KIND_OPERATOR: u8 = 3;
const TRACE_KIND_FINAL_HEAD: u8 = 4;

/// Stateless testnet probe used to measure RPC/leader throughput and actual
/// SBF compute without creating fake protocol jobs. It has no production
/// state transition and cannot affect a dispute.
pub(super) fn testnet_load_probe(
    _program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    if !accounts.is_empty() {
        return Err(QxError::InvalidEncoding.into());
    }
    let rounds = cursor.u16()?;
    let payload_len = usize::from(cursor.u16()?);
    let payload = cursor.take(payload_len)?;
    let expected = cursor.array_32()?;
    if !cursor.is_empty() || !(1..=64).contains(&rounds) || payload_len > 512 {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut digest = stable_hash_domain(D_TESTNET_LOAD, &[payload]);
    for round in 1..rounds {
        digest = stable_hash_domain(D_TESTNET_LOAD, &[&round.to_le_bytes(), &digest]);
    }
    if digest != expected {
        return Err(QxError::InvalidArithmetic.into());
    }
    Ok(())
}

pub(super) fn register_job(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [job, requester, model] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if job.owner != program_id
        || !job.is_signer
        || !job.is_writable
        || !requester.is_signer
        || model.owner != program_id
        || job.data_len() != JOB_ACCOUNT_BYTES
        || model.data_len() != MODEL_ACCOUNT_BYTES
    {
        return Err(QxError::Unauthorized.into());
    }
    let payload = cursor.take(JOB_PAYLOAD_BYTES)?;
    let model_data = model.try_borrow_data()?;
    let input_count = u32_from(payload, JOB_INPUT_COUNT_OFFSET)?;
    let max_output = u32_from(payload, JOB_MAX_OUTPUT_OFFSET)?;
    if !cursor.is_empty()
        || &model_data[..4] != MODEL_MAGIC
        || u16::from_le_bytes(
            payload[..2]
                .try_into()
                .map_err(|_| QxError::InvalidEncoding)?,
        ) != 1
        || payload[JOB_INSTANCE_OFFSET..JOB_INSTANCE_OFFSET + 32] != program_id.to_bytes()
        || payload[JOB_REQUESTER_OFFSET..JOB_REQUESTER_OFFSET + 32] != requester.key.to_bytes()
        || payload[JOB_MODEL_OFFSET..JOB_MODEL_OFFSET + 32] != model_data[4..36]
        || payload[JOB_MODEL_REGISTRY_OFFSET..JOB_MODEL_REGISTRY_OFFSET + 32]
            != model.key.to_bytes()
        || input_count == 0
        || max_output == 0
        || input_count
            .checked_add(max_output)
            .ok_or(QxError::InvalidEncoding)?
            > u32_at(&model_data, 148)?
        || u64_from(payload, JOB_EXPOSURE_OFFSET)? == 0
        || u64_from(payload, JOB_DEADLINE_OFFSET)? <= Clock::get()?.slot
        || !(MIN_AUDIT_PROBABILITY_PPM..=1_000_000)
            .contains(&u32_from(payload, JOB_AUDIT_PPM_OFFSET)?)
    {
        return Err(QxError::InvalidEncoding.into());
    }
    drop(model_data);
    let mut data = job.try_borrow_mut_data()?;
    if data.iter().any(|byte| *byte != 0) {
        return Err(QxError::InvalidState.into());
    }
    data[..4].copy_from_slice(JOB_MAGIC);
    data[4..].copy_from_slice(payload);
    Ok(())
}

pub(super) fn submit_epoch(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [epoch, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let epoch_number = cursor.u64()?;
    if epoch.owner != program_id
        || !epoch.is_signer
        || !epoch.is_writable
        || !miner.is_signer
        || !miner.is_writable
        || miner.owner != program_id
        || miner.data_len() != MINER_V2_ACCOUNT_BYTES
        || epoch.data_len() != EPOCH_V2_ACCOUNT_BYTES
    {
        return Err(QxError::Unauthorized.into());
    }
    let result_root = cursor.array_32()?;
    let result_count = cursor.u32()?;
    let current_slot = Clock::get()?.slot;
    if !cursor.is_empty() || result_count == 0 {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut miner_data = miner.try_borrow_mut_data()?;
    if miner_data.iter().all(|byte| *byte == 0) {
        miner_data[..4].copy_from_slice(MINER_MAGIC_V2);
    } else if &miner_data[..4] != MINER_MAGIC_V2 || epoch_number <= u64_at(&miner_data, 8)? {
        return Err(QxError::InvalidState.into());
    }
    put(&mut miner_data, 8, &epoch_number.to_le_bytes())?;
    drop(miner_data);
    let mut data = epoch.try_borrow_mut_data()?;
    if data.iter().any(|byte| *byte != 0) {
        return Err(QxError::InvalidState.into());
    }
    data[..4].copy_from_slice(EPOCH_MAGIC_V2);
    data[EPOCH_PHASE_OFFSET] = EPOCH_CLOSED;
    put(&mut data, EPOCH_NUMBER_OFFSET, &epoch_number.to_le_bytes())?;
    put(&mut data, EPOCH_MINER_OFFSET, miner.key.as_ref())?;
    put(&mut data, EPOCH_ROOT_OFFSET, &result_root)?;
    put(&mut data, EPOCH_COUNT_OFFSET, &result_count.to_le_bytes())?;
    put(
        &mut data,
        EPOCH_CLOSE_SLOT_OFFSET,
        &current_slot.to_le_bytes(),
    )?;
    put(&mut data, EPOCH_ENTROPY_SLOT_OFFSET, &0_u64.to_le_bytes())?;
    Ok(())
}

pub(super) fn finalize_epoch(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [epoch] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if !cursor.is_empty()
        || epoch.owner != program_id
        || !epoch.is_writable
        || epoch.data_len() != EPOCH_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut data = epoch.try_borrow_mut_data()?;
    require_epoch(&data, EPOCH_CLOSED)?;
    let close_slot = u64_at(&data, EPOCH_CLOSE_SLOT_OFFSET)?;
    if Clock::get()?.slot <= close_slot {
        return Err(QxError::Deadline.into());
    }
    let slot_hashes = PodSlotHashes::fetch()?;
    // The first produced Solana block after closure is deterministic even if
    // close_slot + 1 was skipped. Neither miner nor finalizer chooses among
    // multiple candidate hashes.
    let entries = slot_hashes.as_slice()?;
    // SlotHashes is descending. Binary partitioning finds the final entry
    // above close_slot (therefore the earliest produced successor) without a
    // 512-entry on-chain scan.
    let successor_count = entries.partition_point(|entry| entry.slot > close_slot);
    let successor = successor_count
        .checked_sub(1)
        .and_then(|index| entries.get(index))
        .ok_or(QxError::InvalidState)?;
    let entropy_slot = successor.slot;
    let block_hash = successor.hash.to_bytes();
    let seed = audit_seed(
        program_id,
        u64_at(&data, EPOCH_NUMBER_OFFSET)?,
        &array_at(&data, EPOCH_MINER_OFFSET)?,
        &array_at(&data, EPOCH_ROOT_OFFSET)?,
        u32_at(&data, EPOCH_COUNT_OFFSET)?,
        entropy_slot,
        &block_hash,
    );
    put(&mut data, EPOCH_SEED_OFFSET, &seed)?;
    put(
        &mut data,
        EPOCH_ENTROPY_SLOT_OFFSET,
        &entropy_slot.to_le_bytes(),
    )?;
    let challenge_deadline = entropy_slot
        .checked_add(CHALLENGE_WINDOW_SLOTS)
        .ok_or(QxError::InvalidEncoding)?;
    put(
        &mut data,
        EPOCH_CHALLENGE_DEADLINE_OFFSET,
        &challenge_deadline.to_le_bytes(),
    )?;
    put(
        &mut data,
        EPOCH_WINDOW_OFFSET,
        &RESPONSE_WINDOW_SLOTS.to_le_bytes(),
    )?;
    data[EPOCH_PHASE_OFFSET] = EPOCH_SEEDED;
    Ok(())
}

pub(super) fn open_challenge(
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
        || !dispute.is_signer
        || !dispute.is_writable
        || !challenger.is_signer
        || epoch.data_len() != EPOCH_V2_ACCOUNT_BYTES
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || instruction_sysvar.key != &instructions_sysvar::ID
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let job_payload = cursor.take(JOB_PAYLOAD_BYTES)?;
    let result = cursor.take(RESULT_LEAF_BYTES)?;
    let leaf_index = cursor.u32()?;
    let leaf_count = cursor.u32()?;
    let depth = usize::from(cursor.u8()?);
    let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    let challenger_trace_root = cursor.array_32()?;
    let trace_length = u32_from(result, RESULT_TRACE_LENGTH_OFFSET)?;
    if !cursor.is_empty()
        || trace_length < 2
        || result[RESULT_TRACE_OFFSET..RESULT_TRACE_OFFSET + 32] == challenger_trace_root
    {
        return Err(QxError::InvalidEncoding.into());
    }

    let epoch_data = epoch.try_borrow_data()?;
    require_epoch(&epoch_data, EPOCH_SEEDED)?;
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC {
        return Err(QxError::InvalidState.into());
    }
    let slot = Clock::get()?.slot;
    if slot > u64_at(&epoch_data, EPOCH_CHALLENGE_DEADLINE_OFFSET)?
        || leaf_count != u32_at(&epoch_data, EPOCH_COUNT_OFFSET)?
    {
        return Err(QxError::Deadline.into());
    }
    let miner = array_at(&epoch_data, EPOCH_MINER_OFFSET)?;
    let result_root = array_at(&epoch_data, EPOCH_ROOT_OFFSET)?;
    let epoch_number = u64_at(&epoch_data, EPOCH_NUMBER_OFFSET)?;
    let audit_seed = array_at(&epoch_data, EPOCH_SEED_OFFSET)?;
    let epoch_close_slot = u64_at(&epoch_data, EPOCH_CLOSE_SLOT_OFFSET)?;
    let response_window = u64_at(&epoch_data, EPOCH_WINDOW_OFFSET)?;
    drop(epoch_data);

    if u16::from_le_bytes(
        job_payload[..2]
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    ) != 1
        || job_payload[JOB_INSTANCE_OFFSET..JOB_INSTANCE_OFFSET + 32] != program_id.to_bytes()
        || job_payload[JOB_MINER_OFFSET..JOB_MINER_OFFSET + 32] != miner
        || job_payload[JOB_MODEL_OFFSET..JOB_MODEL_OFFSET + 32] != model.try_borrow_data()?[4..36]
        || job_payload[JOB_EXECUTION_SPEC_OFFSET..JOB_EXECUTION_SPEC_OFFSET + 32]
            != model.try_borrow_data()?[72..104]
        || job_payload[JOB_MODEL_REGISTRY_OFFSET..JOB_MODEL_REGISTRY_OFFSET + 32]
            != model.key.to_bytes()
        || u64_from(job_payload, JOB_EPOCH_OFFSET)? != epoch_number
        || u32_from(job_payload, JOB_INPUT_COUNT_OFFSET)? == 0
        || u32_from(job_payload, JOB_MAX_OUTPUT_OFFSET)? == 0
        || u64_from(job_payload, JOB_EXPOSURE_OFFSET)? == 0
        || !(MIN_AUDIT_PROBABILITY_PPM..=1_000_000)
            .contains(&u32_from(job_payload, JOB_AUDIT_PPM_OFFSET)?)
        || u64_from(job_payload, JOB_DEADLINE_OFFSET)? < epoch_close_slot
    {
        return Err(QxError::InvalidState.into());
    }
    let job_hash = sha_domain(D_JOB_PAYLOAD, &[job_payload]);
    let requester: [u8; 32] = job_payload[JOB_REQUESTER_OFFSET..JOB_REQUESTER_OFFSET + 32]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    verify_prior_ed25519(instruction_sysvar, &requester, &job_hash)?;
    let max_output = u32_from(job_payload, JOB_MAX_OUTPUT_OFFSET)?;
    let input_count = u32_from(job_payload, JOB_INPUT_COUNT_OFFSET)?;
    let output_count = u32_from(result, RESULT_OUTPUT_COUNT_OFFSET)?;
    let audit_probability_ppm = u32_from(job_payload, JOB_AUDIT_PPM_OFFSET)?;
    if result[RESULT_JOB_OFFSET..RESULT_JOB_OFFSET + 32] != job_hash
        || output_count == 0
        || output_count > max_output
        // Executed steps are input + output - 1, and the compact job tree has
        // one additional genesis leaf, hence input + output exactly.
        || input_count
            .checked_add(output_count)
            .ok_or(QxError::InvalidEncoding)?
            != trace_length
    {
        return Err(QxError::InvalidState.into());
    }
    let result_hash = sha_domain(D_RESULT, &[result]);
    verify_merkle_sha(
        D_BATCH,
        result_hash,
        leaf_index,
        leaf_count,
        depth,
        siblings,
        result_root,
    )?;
    if !audit_selected(
        &audit_seed,
        epoch_number,
        leaf_index,
        &job_hash,
        audit_probability_ppm,
    )? {
        return Err(QxError::Unauthorized.into());
    }
    if miner == challenger.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }

    let mut state = dispute.try_borrow_mut_data()?;
    if state.iter().any(|byte| *byte != 0) {
        return Err(QxError::InvalidState.into());
    }
    state[..4].copy_from_slice(DISPUTE_MAGIC_V2);
    state[D_PHASE_OFFSET] = PHASE_CHALLENGER_BOUNDS;
    state[D_WINNER_OFFSET] = PARTY_NONE;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_CHALLENGER;
    state[D_RESOLUTION_FLAGS_OFFSET] = TRACE_KIND_JOB;
    put(&mut state, D_EPOCH_OFFSET, epoch.key.as_ref())?;
    put(&mut state, D_MODEL_OFFSET, model.key.as_ref())?;
    put(&mut state, D_JOB_HASH_OFFSET, &job_hash)?;
    put(
        &mut state,
        D_JOB_ID_OFFSET,
        &job_payload[JOB_ID_OFFSET..JOB_ID_OFFSET + 32],
    )?;
    put(
        &mut state,
        D_ACTIVATION_OFFSET,
        &job_payload[JOB_INPUT_HASH_OFFSET..JOB_INPUT_HASH_OFFSET + 32],
    )?;
    put(&mut state, D_TENSOR_OFFSET, &input_count.to_le_bytes())?;
    put(&mut state, D_MINER_OFFSET, &miner)?;
    put(&mut state, D_CHALLENGER_OFFSET, challenger.key.as_ref())?;
    put(
        &mut state,
        D_DEFENDER_ROOT_OFFSET,
        result[RESULT_TRACE_OFFSET..RESULT_TRACE_OFFSET + 32]
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    )?;
    put(&mut state, D_CHALLENGER_ROOT_OFFSET, &challenger_trace_root)?;
    put(&mut state, D_LOW_OFFSET, &0_u32.to_le_bytes())?;
    put(&mut state, D_HIGH_OFFSET, &(trace_length - 1).to_le_bytes())?;
    put(
        &mut state,
        D_TRACE_LENGTH_OFFSET,
        &trace_length.to_le_bytes(),
    )?;
    put(&mut state, D_WINDOW_OFFSET, &response_window.to_le_bytes())?;
    refresh_deadline(&mut state, slot)?;
    Ok(())
}

pub(super) fn submit_challenger_bounds(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state = checked_party_state(
        program_id,
        dispute,
        challenger,
        PHASE_CHALLENGER_BOUNDS,
        PARTY_CHALLENGER,
    )?;
    let trace_length = u32_at(&state, D_TRACE_LENGTH_OFFSET)?;
    let (start, end) = parse_and_verify_bounds(
        cursor,
        trace_length,
        array_at(&state, D_CHALLENGER_ROOT_OFFSET)?,
        trace_domain(&state)?,
    )?;
    put(&mut state, D_COMMON_ROOT_OFFSET, &start)?;
    put(&mut state, D_CHALLENGER_END_OFFSET, &end)?;
    state[D_PHASE_OFFSET] = PHASE_MINER_BOUNDS;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_MINER;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_miner_bounds(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state =
        checked_party_state(program_id, dispute, miner, PHASE_MINER_BOUNDS, PARTY_MINER)?;
    let trace_length = u32_at(&state, D_TRACE_LENGTH_OFFSET)?;
    let (start, end) = parse_and_verify_bounds(
        cursor,
        trace_length,
        array_at(&state, D_DEFENDER_ROOT_OFFSET)?,
        trace_domain(&state)?,
    )?;
    if start != array_at(&state, D_COMMON_ROOT_OFFSET)?
        || end == array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_DEFENDER_END_OFFSET, &end)?;
    if trace_length == 2 {
        set_terminal_phase(&mut state)?;
    } else {
        set_phase(&mut state, PHASE_CHALLENGER_MIDPOINT, PARTY_CHALLENGER)?;
    }
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_challenger_midpoint(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state = checked_party_state(
        program_id,
        dispute,
        challenger,
        PHASE_CHALLENGER_MIDPOINT,
        PARTY_CHALLENGER,
    )?;
    let low = u32_at(&state, D_LOW_OFFSET)?;
    let high = u32_at(&state, D_HIGH_OFFSET)?;
    let midpoint = low + (high - low) / 2;
    let claim = parse_and_verify_trace_opening(
        cursor,
        midpoint,
        u32_at(&state, D_TRACE_LENGTH_OFFSET)?,
        array_at(&state, D_CHALLENGER_ROOT_OFFSET)?,
        trace_domain(&state)?,
    )?;
    put(&mut state, D_MIDPOINT_OFFSET, &midpoint.to_le_bytes())?;
    put(&mut state, D_PENDING_CHALLENGER_OFFSET, &claim)?;
    set_phase(&mut state, PHASE_MINER_MIDPOINT, PARTY_MINER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_miner_midpoint(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state = checked_party_state(
        program_id,
        dispute,
        miner,
        PHASE_MINER_MIDPOINT,
        PARTY_MINER,
    )?;
    let midpoint = u32_at(&state, D_MIDPOINT_OFFSET)?;
    let claim = parse_and_verify_trace_opening(
        cursor,
        midpoint,
        u32_at(&state, D_TRACE_LENGTH_OFFSET)?,
        array_at(&state, D_DEFENDER_ROOT_OFFSET)?,
        trace_domain(&state)?,
    )?;
    let challenger_claim = array_at(&state, D_PENDING_CHALLENGER_OFFSET)?;
    if claim == challenger_claim {
        put(&mut state, D_LOW_OFFSET, &midpoint.to_le_bytes())?;
        put(&mut state, D_COMMON_ROOT_OFFSET, &claim)?;
    } else {
        put(&mut state, D_HIGH_OFFSET, &midpoint.to_le_bytes())?;
        put(&mut state, D_DEFENDER_END_OFFSET, &claim)?;
        put(&mut state, D_CHALLENGER_END_OFFSET, &challenger_claim)?;
    }
    put(&mut state, D_MIDPOINT_OFFSET, &0_u32.to_le_bytes())?;
    put(&mut state, D_PENDING_CHALLENGER_OFFSET, &[0; 32])?;
    if u32_at(&state, D_HIGH_OFFSET)? - u32_at(&state, D_LOW_OFFSET)? == 1 {
        set_terminal_phase(&mut state)?;
    } else {
        set_phase(&mut state, PHASE_CHALLENGER_MIDPOINT, PARTY_CHALLENGER)?;
    }
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_challenger_terminal(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state = checked_party_state(
        program_id,
        dispute,
        challenger,
        PHASE_CHALLENGER_TERMINAL,
        PARTY_CHALLENGER,
    )?;
    let tensor = cursor.u32()?;
    let row = cursor.u32()?;
    let column = cursor.u32()?;
    let activation_root = cursor.array_32()?;
    let accumulator_start = cursor.u32()?;
    let challenger_end = cursor.u32()?;
    if !cursor.is_empty() || column == u32::MAX {
        return Err(QxError::InvalidEncoding.into());
    }
    let job_hash = array_at(&state, D_JOB_HASH_OFFSET)?;
    let common = fma_state_hash(
        &job_hash,
        tensor,
        row,
        column,
        &activation_root,
        accumulator_start,
    );
    let challenger_output = fma_state_hash(
        &job_hash,
        tensor,
        row,
        column + 1,
        &activation_root,
        challenger_end,
    );
    if common != array_at(&state, D_COMMON_ROOT_OFFSET)?
        || challenger_output != array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_ACTIVATION_OFFSET, &activation_root)?;
    put(&mut state, D_TENSOR_OFFSET, &tensor.to_le_bytes())?;
    put(&mut state, D_ROW_OFFSET, &row.to_le_bytes())?;
    put(&mut state, D_COLUMN_OFFSET, &column.to_le_bytes())?;
    put(
        &mut state,
        D_ACC_START_OFFSET,
        &accumulator_start.to_le_bytes(),
    )?;
    put(
        &mut state,
        D_CHALLENGER_BITS_OFFSET,
        &challenger_end.to_le_bytes(),
    )?;
    set_phase(&mut state, PHASE_MINER_TERMINAL, PARTY_MINER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_miner_terminal(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state = checked_party_state(
        program_id,
        dispute,
        miner,
        PHASE_MINER_TERMINAL,
        PARTY_MINER,
    )?;
    let defender_end = cursor.u32()?;
    if !cursor.is_empty() {
        return Err(QxError::InvalidEncoding.into());
    }
    let output = fma_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TENSOR_OFFSET)?,
        u32_at(&state, D_ROW_OFFSET)?,
        u32_at(&state, D_COLUMN_OFFSET)?
            .checked_add(1)
            .ok_or(QxError::InvalidEncoding)?,
        &array_at(&state, D_ACTIVATION_OFFSET)?,
        defender_end,
    );
    if output != array_at(&state, D_DEFENDER_END_OFFSET)? {
        return Err(QxError::InvalidState.into());
    }
    put(
        &mut state,
        D_DEFENDER_BITS_OFFSET,
        &defender_end.to_le_bytes(),
    )?;
    set_phase(&mut state, PHASE_TERMINAL_READY, PARTY_CHALLENGER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_challenger_token(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id || model.data_len() != MODEL_ACCOUNT_BYTES {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = checked_party_state(
        program_id,
        dispute,
        challenger,
        PHASE_CHALLENGER_TOKEN,
        PARTY_CHALLENGER,
    )?;
    let token = cursor.take(76)?;
    let checkpoint_count = u32_from(token, 72)?;
    let token_index = u32_from(token, 32)?;
    let expected_token = u32_at(&state, D_HIGH_OFFSET)?
        .checked_sub(1)
        .ok_or(QxError::InvalidState)?;
    let model_data = model.try_borrow_data()?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes()
        || &model_data[..4] != MODEL_MAGIC
        || checkpoint_count != u32_at(&model_data, 140)?
        || token_index != expected_token
        || token[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
        || stable_hash_domain(D_TOKEN_RESULT, &[token])
            != array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let input_count = u32_at(&state, D_TENSOR_OFFSET)?;
    let consumed_token = cursor.u32()?;
    if token_index < input_count {
        let depth = usize::from(cursor.u8()?);
        let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
        let input_leaf = sha_domain(
            D_INPUT_TOKEN,
            &[
                &model_data[4..36],
                &input_count.to_le_bytes(),
                &token_index.to_le_bytes(),
                &consumed_token.to_le_bytes(),
            ],
        );
        verify_merkle_sha(
            D_INPUT_TREE,
            input_leaf,
            token_index,
            input_count,
            depth,
            siblings,
            array_at(&state, D_ACTIVATION_OFFSET)?,
        )?;
    }

    let expected_genesis_kv = if token_index == 0 {
        if array_at(&state, D_COMMON_ROOT_OFFSET)?
            != stable_hash_domain(D_JOB_GENESIS, &[&array_at(&state, D_JOB_ID_OFFSET)?])
        {
            return Err(QxError::InvalidState.into());
        }
        empty_global_kv_root(u32::from(u16_from(&model_data, 144)?))?
    } else {
        let prior = cursor.take(76)?;
        let prior_index = token_index.checked_sub(1).ok_or(QxError::InvalidState)?;
        if prior[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
            || u32_from(prior, 32)? != prior_index
            || u32_from(prior, 72)? != checkpoint_count
            || stable_hash_domain(D_TOKEN_RESULT, &[prior])
                != array_at(&state, D_COMMON_ROOT_OFFSET)?
            || (token_index >= input_count && u32_from(prior, 68)? != consumed_token)
        {
            return Err(QxError::InvalidState.into());
        }
        let final_checkpoint = cursor.take(102)?;
        let depth = usize::from(cursor.u8()?);
        let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
        let final_layer = u32::from(u16_from(&model_data, 144)?)
            .checked_sub(1)
            .ok_or(QxError::InvalidState)?;
        if final_checkpoint[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
            || u32_from(final_checkpoint, 32)? != prior_index
            || u32::from(u16_from(final_checkpoint, 36)?) != final_layer
        {
            return Err(QxError::InvalidState.into());
        }
        verify_merkle_blake3(
            stable_hash_domain(D_CHECKPOINT, &[final_checkpoint]),
            checkpoint_count - 1,
            checkpoint_count,
            depth,
            siblings,
            prior[36..68]
                .try_into()
                .map_err(|_| QxError::InvalidEncoding)?,
            D_TOKEN_TREE,
        )?;
        final_checkpoint[70..102]
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?
    };
    if !cursor.is_empty() {
        return Err(QxError::InvalidEncoding.into());
    }
    drop(model_data);
    put(&mut state, D_CHALLENGER_TOKEN_ROOT_OFFSET, &token[36..68])?;
    put(
        &mut state,
        D_TOKEN_COUNT_OFFSET,
        &checkpoint_count.to_le_bytes(),
    )?;
    put(&mut state, D_TOKEN_INDEX_OFFSET, &token_index.to_le_bytes())?;
    put(&mut state, D_CHALLENGER_PREDICTION_OFFSET, &token[68..72])?;
    put(&mut state, D_ACTIVATION_OFFSET, &expected_genesis_kv)?;
    put(&mut state, D_ROW_OFFSET, &consumed_token.to_le_bytes())?;
    set_phase(&mut state, PHASE_MINER_TOKEN, PARTY_MINER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_miner_token(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state =
        checked_party_state(program_id, dispute, miner, PHASE_MINER_TOKEN, PARTY_MINER)?;
    let token = cursor.take(76)?;
    let checkpoint_count = u32_from(token, 72)?;
    let token_index = u32_from(token, 32)?;
    if !cursor.is_empty()
        || checkpoint_count != u32_at(&state, D_TOKEN_COUNT_OFFSET)?
        || token_index != u32_at(&state, D_TOKEN_INDEX_OFFSET)?
        || token[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
        || stable_hash_domain(D_TOKEN_RESULT, &[token]) != array_at(&state, D_DEFENDER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_DEFENDER_PREDICTION_OFFSET, &token[68..72])?;
    let defender_root: [u8; 32] = token[36..68]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let challenger_root = array_at(&state, D_CHALLENGER_TOKEN_ROOT_OFFSET)?;
    if defender_root == challenger_root {
        if token[68..72]
            == state[D_CHALLENGER_PREDICTION_OFFSET..D_CHALLENGER_PREDICTION_OFFSET + 4]
        {
            return Err(QxError::InvalidState.into());
        }
        put(&mut state, D_DEFENDER_ROOT_OFFSET, &defender_root)?;
        put(&mut state, D_CHALLENGER_ROOT_OFFSET, &challenger_root)?;
        put(&mut state, D_LOW_OFFSET, &0_u32.to_le_bytes())?;
        put(&mut state, D_HIGH_OFFSET, &checkpoint_count.to_le_bytes())?;
        state[D_RESOLUTION_FLAGS_OFFSET] = TRACE_KIND_FINAL_HEAD;
        set_phase(&mut state, PHASE_CHALLENGER_FINAL_HEAD, PARTY_CHALLENGER)?;
        return refresh_deadline(&mut state, Clock::get()?.slot);
    }
    put(&mut state, D_DEFENDER_ROOT_OFFSET, &defender_root)?;
    put(&mut state, D_CHALLENGER_ROOT_OFFSET, &challenger_root)?;
    put(&mut state, D_LOW_OFFSET, &0_u32.to_le_bytes())?;
    put(
        &mut state,
        D_HIGH_OFFSET,
        &(checkpoint_count - 1).to_le_bytes(),
    )?;
    put(
        &mut state,
        D_TRACE_LENGTH_OFFSET,
        &checkpoint_count.to_le_bytes(),
    )?;
    state[D_RESOLUTION_FLAGS_OFFSET] = TRACE_KIND_TOKEN;
    set_phase(&mut state, PHASE_CHALLENGER_BOUNDS, PARTY_CHALLENGER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

/// When checkpoint roots agree but predicted tokens differ, authenticate the
/// shared final checkpoint and bind the challenger's final-head trace to that
/// checkpoint and its claimed output token.
pub(super) fn submit_challenger_final_head(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id || model.data_len() != MODEL_ACCOUNT_BYTES {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = checked_party_state(
        program_id,
        dispute,
        challenger,
        PHASE_CHALLENGER_FINAL_HEAD,
        PARTY_CHALLENGER,
    )?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let checkpoint = cursor.take(102)?;
    let checkpoint_depth = usize::from(cursor.u8()?);
    let checkpoint_siblings = cursor.take(
        checkpoint_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC {
        return Err(QxError::InvalidState.into());
    }
    let checkpoint_count = u32_at(&state, D_TOKEN_COUNT_OFFSET)?;
    let checkpoint_index = checkpoint_count
        .checked_sub(1)
        .ok_or(QxError::InvalidState)?;
    let model_layers = u32::from(u16_from(&model_data, 144)?);
    if checkpoint[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
        || u32_from(checkpoint, 32)? != u32_at(&state, D_TOKEN_INDEX_OFFSET)?
        || u32::from(u16_from(checkpoint, 36)?)
            != model_layers.checked_sub(1).ok_or(QxError::InvalidState)?
    {
        return Err(QxError::InvalidState.into());
    }
    verify_merkle_blake3(
        stable_hash_domain(D_CHECKPOINT, &[checkpoint]),
        checkpoint_index,
        checkpoint_count,
        checkpoint_depth,
        checkpoint_siblings,
        array_at(&state, D_CHALLENGER_ROOT_OFFSET)?,
        D_TOKEN_TREE,
    )?;
    let segment_index = u32_at(&state, D_HIGH_OFFSET)?;
    let (instruction_root, instruction_count) =
        parse_segment_program(cursor, &model_data, segment_index)?;
    let trace_length = instruction_count
        .checked_add(1)
        .ok_or(QxError::InvalidEncoding)?;
    drop(model_data);
    let end_residual = cursor.array_32()?;
    let end_kv = cursor.array_32()?;
    let trace_root = cursor.array_32()?;
    let (start, end) = parse_and_verify_bounds(cursor, trace_length, trace_root, D_OPERATOR_TREE)?;
    let checkpoint_residual: [u8; 32] = checkpoint[38..70]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let checkpoint_kv: [u8; 32] = checkpoint[70..102]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let expected_start = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        0,
        &checkpoint_residual,
        &checkpoint_kv,
    );
    let expected_end = vm_output_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        instruction_count,
        &end_residual,
        &end_kv,
        u32_at(&state, D_CHALLENGER_PREDICTION_OFFSET)?,
    );
    if start != expected_start || end != expected_end {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_COMMON_RESIDUAL_OFFSET, &checkpoint[38..70])?;
    put(&mut state, D_COMMON_KV_OFFSET, &checkpoint[70..102])?;
    put(&mut state, D_CHALLENGER_RESIDUAL_OFFSET, &end_residual)?;
    put(&mut state, D_CHALLENGER_KV_OFFSET, &end_kv)?;
    put(&mut state, D_CHALLENGER_ROOT_OFFSET, &trace_root)?;
    put(&mut state, D_COMMON_ROOT_OFFSET, &start)?;
    put(&mut state, D_CHALLENGER_END_OFFSET, &end)?;
    put(&mut state, D_SEGMENT_PROGRAM_ROOT_OFFSET, &instruction_root)?;
    put(
        &mut state,
        D_SEGMENT_INSTRUCTION_COUNT_OFFSET,
        &instruction_count.to_le_bytes(),
    )?;
    put(
        &mut state,
        D_TRACE_LENGTH_OFFSET,
        &trace_length.to_le_bytes(),
    )?;
    set_phase(&mut state, PHASE_MINER_FINAL_HEAD, PARTY_MINER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

/// Bind the miner's final-head trace to the same shared checkpoint and its
/// different predicted token, then enter the common operator bisection game.
pub(super) fn submit_miner_final_head(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id || model.data_len() != MODEL_ACCOUNT_BYTES {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = checked_party_state(
        program_id,
        dispute,
        miner,
        PHASE_MINER_FINAL_HEAD,
        PARTY_MINER,
    )?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC {
        return Err(QxError::InvalidState.into());
    }
    let instruction_count = u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?;
    let trace_length = instruction_count
        .checked_add(1)
        .ok_or(QxError::InvalidEncoding)?;
    drop(model_data);
    let end_residual = cursor.array_32()?;
    let end_kv = cursor.array_32()?;
    let trace_root = cursor.array_32()?;
    let (start, end) = parse_and_verify_bounds(cursor, trace_length, trace_root, D_OPERATOR_TREE)?;
    let segment_index = u32_at(&state, D_HIGH_OFFSET)?;
    let expected_start = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        0,
        &array_at(&state, D_COMMON_RESIDUAL_OFFSET)?,
        &array_at(&state, D_COMMON_KV_OFFSET)?,
    );
    let expected_end = vm_output_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        instruction_count,
        &end_residual,
        &end_kv,
        u32_at(&state, D_DEFENDER_PREDICTION_OFFSET)?,
    );
    if start != expected_start
        || start != array_at(&state, D_COMMON_ROOT_OFFSET)?
        || end != expected_end
        || end == array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_DEFENDER_RESIDUAL_OFFSET, &end_residual)?;
    put(&mut state, D_DEFENDER_KV_OFFSET, &end_kv)?;
    put(&mut state, D_DEFENDER_ROOT_OFFSET, &trace_root)?;
    put(&mut state, D_DEFENDER_END_OFFSET, &end)?;
    put(&mut state, D_LOW_OFFSET, &0_u32.to_le_bytes())?;
    put(&mut state, D_HIGH_OFFSET, &instruction_count.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] = TRACE_KIND_FINAL_HEAD;
    if trace_length == 2 {
        set_terminal_phase(&mut state)?;
    } else {
        set_phase(&mut state, PHASE_CHALLENGER_MIDPOINT, PARTY_CHALLENGER)?;
    }
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_challenger_checkpoint(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id || model.data_len() != MODEL_ACCOUNT_BYTES {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = checked_party_state(
        program_id,
        dispute,
        challenger,
        PHASE_CHALLENGER_CHECKPOINT,
        PARTY_CHALLENGER,
    )?;
    let common_checkpoint = cursor.take(102)?;
    let checkpoint = cursor.take(102)?;
    let model_data = model.try_borrow_data()?;
    let model_layers = u32::from(u16_from(&model_data, 144)?);
    let checkpoint_interval = u32::from(u16_from(&model_data, 146)?);
    let checkpoint_position = u32_at(&state, D_HIGH_OFFSET)?;
    let expected_layer = checkpoint_position
        .checked_mul(checkpoint_interval)
        .and_then(|value| value.checked_sub(1))
        .ok_or(QxError::InvalidState)?;
    let expected_common_layer = if checkpoint_position == 1 {
        u32::from(u16::MAX)
    } else {
        expected_layer
            .checked_sub(checkpoint_interval)
            .ok_or(QxError::InvalidState)?
    };
    let expected_input_state = stable_hash_domain(
        D_TOKEN_INPUT_STATE,
        &[
            &array_at(&state, D_JOB_ID_OFFSET)?,
            &u32_at(&state, D_TOKEN_INDEX_OFFSET)?.to_le_bytes(),
        ],
    );
    if !cursor.is_empty()
        || state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes()
        || expected_layer >= model_layers
        || common_checkpoint[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
        || u32_from(common_checkpoint, 32)? != u32_at(&state, D_TOKEN_INDEX_OFFSET)?
        || u32::from(u16_from(common_checkpoint, 36)?) != expected_common_layer
        || (model_layers == 24
            && expected_common_layer == u32::from(u16::MAX)
            && common_checkpoint[38..70] != expected_input_state)
        || (expected_common_layer == u32::from(u16::MAX)
            && common_checkpoint[70..102] != state[D_ACTIVATION_OFFSET..D_ACTIVATION_OFFSET + 32])
        || stable_hash_domain(D_CHECKPOINT, &[common_checkpoint])
            != array_at(&state, D_COMMON_ROOT_OFFSET)?
        || checkpoint[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
        || u32_from(checkpoint, 32)? != u32_at(&state, D_TOKEN_INDEX_OFFSET)?
        || u32::from(u16_from(checkpoint, 36)?) != expected_layer
        || stable_hash_domain(D_CHECKPOINT, &[checkpoint])
            != array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    drop(model_data);
    put(
        &mut state,
        D_COMMON_RESIDUAL_OFFSET,
        &common_checkpoint[38..70],
    )?;
    put(&mut state, D_COMMON_KV_OFFSET, &common_checkpoint[70..102])?;
    put(&mut state, D_CHECKPOINT_LAYER_OFFSET, &checkpoint[36..38])?;
    put(
        &mut state,
        D_CHALLENGER_RESIDUAL_OFFSET,
        &checkpoint[38..70],
    )?;
    put(&mut state, D_CHALLENGER_KV_OFFSET, &checkpoint[70..102])?;
    set_phase(&mut state, PHASE_MINER_CHECKPOINT, PARTY_MINER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_miner_checkpoint(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state = checked_party_state(
        program_id,
        dispute,
        miner,
        PHASE_MINER_CHECKPOINT,
        PARTY_MINER,
    )?;
    let checkpoint = cursor.take(102)?;
    if !cursor.is_empty()
        || checkpoint[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
        || u32_from(checkpoint, 32)? != u32_at(&state, D_TOKEN_INDEX_OFFSET)?
        || u16_from(checkpoint, 36)? != u16_from(&state, D_CHECKPOINT_LAYER_OFFSET)?
        || stable_hash_domain(D_CHECKPOINT, &[checkpoint])
            != array_at(&state, D_DEFENDER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_DEFENDER_RESIDUAL_OFFSET, &checkpoint[38..70])?;
    put(&mut state, D_DEFENDER_KV_OFFSET, &checkpoint[70..102])?;
    state[D_RESOLUTION_FLAGS_OFFSET] = TRACE_KIND_OPERATOR;
    set_phase(&mut state, PHASE_CHALLENGER_OPERATOR_ROOT, PARTY_CHALLENGER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

/// Bind the challenger operator trace to the agreed predecessor checkpoint
/// and the challenger's disputed checkpoint. Neither party may choose the VM
/// boundary states or the trace length.
pub(super) fn submit_challenger_operator(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id || model.data_len() != MODEL_ACCOUNT_BYTES {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = checked_party_state(
        program_id,
        dispute,
        challenger,
        PHASE_CHALLENGER_OPERATOR_ROOT,
        PARTY_CHALLENGER,
    )?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC {
        return Err(QxError::InvalidState.into());
    }
    let segment_index = u32_at(&state, D_HIGH_OFFSET)?;
    let (instruction_root, instruction_count) =
        parse_segment_program(cursor, &model_data, segment_index)?;
    let trace_length = instruction_count
        .checked_add(1)
        .ok_or(QxError::InvalidEncoding)?;
    drop(model_data);
    let trace_root = cursor.array_32()?;
    let (start, end) = parse_and_verify_bounds(cursor, trace_length, trace_root, D_OPERATOR_TREE)?;
    let expected_start = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        0,
        &array_at(&state, D_COMMON_RESIDUAL_OFFSET)?,
        &array_at(&state, D_COMMON_KV_OFFSET)?,
    );
    let expected_end = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        instruction_count,
        &array_at(&state, D_CHALLENGER_RESIDUAL_OFFSET)?,
        &array_at(&state, D_CHALLENGER_KV_OFFSET)?,
    );
    if start != expected_start || end != expected_end {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_CHALLENGER_ROOT_OFFSET, &trace_root)?;
    put(&mut state, D_COMMON_ROOT_OFFSET, &start)?;
    put(&mut state, D_CHALLENGER_END_OFFSET, &end)?;
    put(&mut state, D_SEGMENT_PROGRAM_ROOT_OFFSET, &instruction_root)?;
    put(
        &mut state,
        D_SEGMENT_INSTRUCTION_COUNT_OFFSET,
        &instruction_count.to_le_bytes(),
    )?;
    put(
        &mut state,
        D_TRACE_LENGTH_OFFSET,
        &trace_length.to_le_bytes(),
    )?;
    set_phase(&mut state, PHASE_MINER_OPERATOR_ROOT, PARTY_MINER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

/// Bind the miner operator trace to the same agreed predecessor and the
/// miner's checkpoint, then enter the shared logarithmic bisection engine.
pub(super) fn submit_miner_operator(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id || model.data_len() != MODEL_ACCOUNT_BYTES {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = checked_party_state(
        program_id,
        dispute,
        miner,
        PHASE_MINER_OPERATOR_ROOT,
        PARTY_MINER,
    )?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC {
        return Err(QxError::InvalidState.into());
    }
    let instruction_count = u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?;
    let trace_length = instruction_count
        .checked_add(1)
        .ok_or(QxError::InvalidEncoding)?;
    drop(model_data);
    let trace_root = cursor.array_32()?;
    let (start, end) = parse_and_verify_bounds(cursor, trace_length, trace_root, D_OPERATOR_TREE)?;
    let segment_index = u32_at(&state, D_HIGH_OFFSET)?;
    let expected_start = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        0,
        &array_at(&state, D_COMMON_RESIDUAL_OFFSET)?,
        &array_at(&state, D_COMMON_KV_OFFSET)?,
    );
    let expected_end = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        instruction_count,
        &array_at(&state, D_DEFENDER_RESIDUAL_OFFSET)?,
        &array_at(&state, D_DEFENDER_KV_OFFSET)?,
    );
    if start != expected_start
        || start != array_at(&state, D_COMMON_ROOT_OFFSET)?
        || end != expected_end
        || end == array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_DEFENDER_ROOT_OFFSET, &trace_root)?;
    put(&mut state, D_DEFENDER_END_OFFSET, &end)?;
    put(&mut state, D_LOW_OFFSET, &0_u32.to_le_bytes())?;
    put(&mut state, D_HIGH_OFFSET, &instruction_count.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] = TRACE_KIND_OPERATOR;
    if trace_length == 2 {
        set_terminal_phase(&mut state)?;
    } else {
        set_phase(&mut state, PHASE_CHALLENGER_MIDPOINT, PARTY_CHALLENGER)?;
    }
    refresh_deadline(&mut state, Clock::get()?.slot)
}

/// Authenticate the challenger's isolated Granite macro boundary. The
/// surrounding bisection commits only rolling workspace roots; this opening
/// proves which registered macro was selected and that it changes exactly its
/// designated workspace slot. The arithmetic behind the new value is handled
/// by the following macro-specific scalar descent.
pub(super) fn submit_challenger_macro(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state = checked_party_state(
        program_id,
        dispute,
        challenger,
        PHASE_CHALLENGER_TERMINAL,
        PARTY_CHALLENGER,
    )?;
    if !matches!(
        state[D_RESOLUTION_FLAGS_OFFSET],
        TRACE_KIND_OPERATOR | TRACE_KIND_FINAL_HEAD
    ) {
        return Err(QxError::InvalidState.into());
    }
    let segment_index = cursor.u32()?;
    let program_counter = u32_at(&state, D_LOW_OFFSET)?;
    let instruction = cursor.take(17)?;
    let instruction_count = cursor.u32()?;
    let instruction_depth = usize::from(cursor.u8()?);
    let instruction_siblings = cursor.take(
        instruction_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if instruction_count != u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?
        || program_counter >= instruction_count
    {
        return Err(QxError::InvalidState.into());
    }
    verify_merkle_blake3(
        stable_hash_domain(
            D_VM_INSTRUCTION,
            &[&program_counter.to_le_bytes(), instruction],
        ),
        program_counter,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&state, D_SEGMENT_PROGRAM_ROOT_OFFSET)?,
        D_PROGRAM_TREE,
    )?;
    let opcode = instruction[0];
    let slot = granite_macro_slot(instruction)?;
    let old_value_root = cursor.array_32()?;
    let new_value_root = cursor.array_32()?;
    let workspace_siblings = cursor.take(4 * 32)?;
    let pre_kv_root = cursor.array_32()?;
    let post_kv_root = cursor.array_32()?;
    if !cursor.is_empty() || old_value_root == new_value_root {
        return Err(QxError::InvalidEncoding.into());
    }
    validate_macro_kv_transition(opcode, &pre_kv_root, &post_kv_root)?;
    let pre_workspace_root = macro_workspace_root(slot, &old_value_root, workspace_siblings);
    let post_workspace_root = macro_workspace_root(slot, &new_value_root, workspace_siblings);
    validate_macro_pre_boundary(
        &state,
        segment_index,
        program_counter,
        &pre_workspace_root,
        &pre_kv_root,
    )?;
    let challenger_post = macro_post_boundary_hash(
        &state,
        segment_index,
        program_counter,
        instruction_count,
        slot,
        &new_value_root,
        &post_workspace_root,
        &post_kv_root,
        true,
    )?;
    if challenger_post != array_at(&state, D_CHALLENGER_END_OFFSET)? {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_COMMON_RESIDUAL_OFFSET, &pre_workspace_root)?;
    put(&mut state, D_COMMON_KV_OFFSET, &pre_kv_root)?;
    put(
        &mut state,
        D_CHALLENGER_RESIDUAL_OFFSET,
        &post_workspace_root,
    )?;
    put(&mut state, D_CHALLENGER_KV_OFFSET, &post_kv_root)?;
    put(&mut state, D_PENDING_CHALLENGER_OFFSET, &old_value_root)?;
    put(&mut state, D_CHALLENGER_END_OFFSET, &new_value_root)?;
    put(&mut state, D_ACTIVATION_OFFSET, instruction)?;
    put(&mut state, D_TENSOR_OFFSET, &u32::from(slot).to_le_bytes())?;
    set_phase(&mut state, PHASE_MINER_MACRO, PARTY_MINER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

/// Authenticate the miner's endpoint for the same isolated macro and retain
/// both disputed output roots for the macro-specific scalar proof.
pub(super) fn submit_miner_macro(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state =
        checked_party_state(program_id, dispute, miner, PHASE_MINER_MACRO, PARTY_MINER)?;
    let segment_index = cursor.u32()?;
    let program_counter = u32_at(&state, D_LOW_OFFSET)?;
    let instruction_count = u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?;
    let instruction: [u8; 17] = state[D_ACTIVATION_OFFSET..D_ACTIVATION_OFFSET + 17]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let slot =
        u8::try_from(u32_at(&state, D_TENSOR_OFFSET)?).map_err(|_| QxError::InvalidEncoding)?;
    if granite_macro_slot(&instruction)? != slot {
        return Err(QxError::InvalidState.into());
    }
    let old_value_root = cursor.array_32()?;
    let new_value_root = cursor.array_32()?;
    let workspace_siblings = cursor.take(4 * 32)?;
    let pre_kv_root = cursor.array_32()?;
    let post_kv_root = cursor.array_32()?;
    if !cursor.is_empty()
        || old_value_root != array_at(&state, D_PENDING_CHALLENGER_OFFSET)?
        || new_value_root == old_value_root
        || pre_kv_root != array_at(&state, D_COMMON_KV_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    validate_macro_kv_transition(instruction[0], &pre_kv_root, &post_kv_root)?;
    let pre_workspace_root = macro_workspace_root(slot, &old_value_root, workspace_siblings);
    if pre_workspace_root != array_at(&state, D_COMMON_RESIDUAL_OFFSET)? {
        return Err(QxError::InvalidState.into());
    }
    let post_workspace_root = macro_workspace_root(slot, &new_value_root, workspace_siblings);
    let defender_post = macro_post_boundary_hash(
        &state,
        segment_index,
        program_counter,
        instruction_count,
        slot,
        &new_value_root,
        &post_workspace_root,
        &post_kv_root,
        false,
    )?;
    if defender_post != array_at(&state, D_DEFENDER_END_OFFSET)?
        || new_value_root == array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_DEFENDER_RESIDUAL_OFFSET, &post_workspace_root)?;
    put(&mut state, D_DEFENDER_KV_OFFSET, &post_kv_root)?;
    put(&mut state, D_DEFENDER_END_OFFSET, &new_value_root)?;
    set_phase(&mut state, PHASE_MACRO_READY, PARTY_CHALLENGER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

/// Stage one of the two authenticated inputs or two competing outputs for an
/// isolated Granite residual macro. Keeping each vector opening in its own
/// instruction bounds transaction size even for 256-byte residual chunks.
pub(super) fn stage_macro_residual(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if dispute.owner != program_id
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || !staged.is_writable
        || staged.data_len() != STAGED_MACRO_SCALAR_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let state = dispute.try_borrow_data()?;
    require_dispute(&state, PHASE_MACRO_READY)?;
    require_deadline(&state)?;
    let role = cursor.u8()?;
    if role > 3 {
        return Err(QxError::InvalidEncoding.into());
    }
    let element = cursor.u32()?;
    let instruction: [u8; 17] = state[D_ACTIVATION_OFFSET..D_ACTIVATION_OFFSET + 17]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    if instruction[0] != 70 || element >= u32_from(&instruction, 13)? {
        return Err(QxError::InvalidState.into());
    }
    let residual_stage = u32_from(&instruction, 9)?;
    let (expected_root, workspace_slot, state_chunks) = match (residual_stage, role) {
        (0, 0) => (None, 0, true),
        (0, 1) => (None, 8, false),
        (0, 2) => (Some(array_at(&state, D_CHALLENGER_END_OFFSET)?), 0, false),
        (0, 3) => (Some(array_at(&state, D_DEFENDER_END_OFFSET)?), 0, false),
        (1, 0) => (None, 9, false),
        (1, 1) => (None, 13, false),
        (1, 2) => (Some(array_at(&state, D_CHALLENGER_END_OFFSET)?), 0, true),
        (1, 3) => (Some(array_at(&state, D_DEFENDER_END_OFFSET)?), 0, true),
        _ => return Err(QxError::InvalidState.into()),
    };
    let root = if let Some(root) = expected_root {
        root
    } else {
        parse_macro_workspace_value(
            cursor,
            workspace_slot,
            array_at(&state, D_COMMON_RESIDUAL_OFFSET)?,
        )?
    };
    let bits = parse_vector_scalar(cursor, element, root, state_chunks)?;
    if !cursor.is_empty() || !f32::from_bits(bits).is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let common_workspace = array_at(&state, D_COMMON_RESIDUAL_OFFSET)?;
    let program_counter = u32_at(&state, D_LOW_OFFSET)?;
    drop(state);

    let mut staged_data = staged.try_borrow_mut_data()?;
    if staged_data.iter().all(|byte| *byte == 0) {
        staged_data[..4].copy_from_slice(STAGED_MACRO_SCALAR_MAGIC);
        put(&mut staged_data, 4, dispute.key.as_ref())?;
        put(&mut staged_data, 36, &common_workspace)?;
        put(&mut staged_data, 68, &program_counter.to_le_bytes())?;
        put(&mut staged_data, 72, &element.to_le_bytes())?;
    } else if &staged_data[..4] != STAGED_MACRO_SCALAR_MAGIC
        || staged_data[4..36] != dispute.key.to_bytes()
        || staged_data[36..68] != common_workspace
        || u32_at(&staged_data, 68)? != program_counter
        || u32_at(&staged_data, 72)? != element
    {
        return Err(QxError::Unauthorized.into());
    }
    let mask = 1_u8 << role;
    let offset = 80 + usize::from(role) * 4;
    if staged_data[76] & mask != 0 && u32_at(&staged_data, offset)? != bits {
        return Err(QxError::InvalidState.into());
    }
    put(&mut staged_data, offset, &bits.to_le_bytes())?;
    staged_data[76] |= mask;
    Ok(())
}

pub(super) fn resolve_macro_residual(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if !cursor.is_empty()
        || dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || staged.data_len() != STAGED_MACRO_SCALAR_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let staged_data = staged.try_borrow_data()?;
    if &staged_data[..4] != STAGED_MACRO_SCALAR_MAGIC
        || staged_data[4..36] != dispute.key.to_bytes()
        || staged_data[76] != 15
    {
        return Err(QxError::Unauthorized.into());
    }
    let common_workspace: [u8; 32] = staged_data[36..68]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let program_counter = u32_at(&staged_data, 68)?;
    let left = u32_at(&staged_data, 80)?;
    let right = u32_at(&staged_data, 84)?;
    let challenger = u32_at(&staged_data, 88)?;
    let defender = u32_at(&staged_data, 92)?;
    drop(staged_data);

    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_MACRO_READY)?;
    require_deadline(&state)?;
    let instruction = &state[D_ACTIVATION_OFFSET..D_ACTIVATION_OFFSET + 17];
    if instruction[0] != 70
        || common_workspace != array_at(&state, D_COMMON_RESIDUAL_OFFSET)?
        || program_counter != u32_at(&state, D_LOW_OFFSET)?
        || challenger == defender
    {
        return Err(QxError::InvalidState.into());
    }
    let scale = f32::from_bits(u32_from(instruction, 5)?);
    if !scale.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let expected =
        qx_consensus_math::fma_f32(scale, f32::from_bits(right), f32::from_bits(left)).to_bits();
    let agrees_with_challenger = expected == challenger;
    let disagrees_with_miner = expected != defender;
    let winner = if agrees_with_challenger && disagrees_with_miner {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    put(&mut state, D_ACC_START_OFFSET, &expected.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

pub(super) fn stage_q4_weight(
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
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.data_len() != STAGED_WEIGHT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let state = dispute.try_borrow_data()?;
    require_dispute(&state, PHASE_TERMINAL_READY)?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let tensor = cursor.u32()?;
    let row = cursor.u32()?;
    let column = cursor.u32()?;
    let leaf_index = cursor.u32()?;
    let group = cursor.u32()?;
    let quantization_type = cursor.u8()?;
    let scale_bits = cursor.u16()?;
    let packed = cursor.take(16)?;
    let depth = usize::from(cursor.u8()?);
    let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    if !cursor.is_empty()
        || tensor != u32_at(&state, D_TENSOR_OFFSET)?
        || row != u32_at(&state, D_ROW_OFFSET)?
        || column != u32_at(&state, D_COLUMN_OFFSET)?
        || group != column / Q4_GROUP_SIZE
        || quantization_type != Q4_TYPE
    {
        return Err(QxError::InvalidEncoding.into());
    }
    drop(state);
    let model_data = model.try_borrow_data()?;
    let leaf_count = u32_at(&model_data, 68)?;
    let leaf = weight_leaf_hash(tensor, row, group, quantization_type, packed, scale_bits);
    verify_merkle_sha(
        D_WEIGHT_TREE,
        leaf,
        leaf_index,
        leaf_count,
        depth,
        siblings,
        array_at(&model_data, 36)?,
    )?;
    drop(model_data);
    let local = usize::try_from(column % Q4_GROUP_SIZE).map_err(|_| QxError::InvalidEncoding)?;
    let code = if local & 1 == 0 {
        packed[local / 2] & 0x0f
    } else {
        packed[local / 2] >> 4
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
    put(&mut staged_data, 36, &tensor.to_le_bytes())?;
    put(&mut staged_data, 40, &row.to_le_bytes())?;
    put(&mut staged_data, 44, &column.to_le_bytes())?;
    put(&mut staged_data, 48, &weight_bits.to_le_bytes())?;
    Ok(())
}

pub(super) fn resolve_q4_fma(
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
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
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
    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_TERMINAL_READY)?;
    require_deadline(&state)?;
    if column != u32_at(&state, D_COLUMN_OFFSET)? {
        return Err(QxError::InvalidState.into());
    }
    let activation = parse_activation(cursor, column, array_at(&state, D_ACTIVATION_OFFSET)?)?;
    let accumulator = f32::from_bits(u32_at(&state, D_ACC_START_OFFSET)?);
    if !activation.is_finite() || !accumulator.is_finite() || !weight.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let result = weight.mul_add(activation, accumulator);
    if !result.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let defender = u32_at(&state, D_DEFENDER_BITS_OFFSET)?;
    let challenger = u32_at(&state, D_CHALLENGER_BITS_OFFSET)?;
    let result_bytes = result.to_le_bytes();
    let result_bits = u32::from_le_bytes(result_bytes);
    let agrees_with_challenger = result_bytes == challenger.to_le_bytes();
    let disagrees_with_miner = result_bytes != defender.to_le_bytes();
    let winner = if disagrees_with_miner && agrees_with_challenger {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    // Once resolved, the start accumulator slot is no longer needed for
    // progression. Retain the verifier-computed result there so settlement
    // and indexers can independently inspect the objective terminal value.
    put(&mut state, D_ACC_START_OFFSET, &result_bits.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

/// Reveal an entire two-state scalar primitive after the surrounding trace
/// has already narrowed to one transition. The agreed predecessor binds all
/// operands; the challenger endpoint binds its claimed result.
pub(super) fn submit_challenger_scalar(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, challenger] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id || model.data_len() != MODEL_ACCOUNT_BYTES {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = checked_party_state(
        program_id,
        dispute,
        challenger,
        PHASE_CHALLENGER_TERMINAL,
        PARTY_CHALLENGER,
    )?;
    let op = cursor.u8()?;
    let a = cursor.u32()?;
    let b = cursor.u32()?;
    let c = cursor.u32()?;
    let challenger_result = cursor.u32()?;
    let instruction_index = cursor.u32()?;
    let instruction_count = cursor.u32()?;
    let instruction_depth = usize::from(cursor.u8()?);
    let instruction_siblings = cursor.take(
        instruction_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if !cursor.is_empty() || !scalar_op_supported(op) {
        return Err(QxError::InvalidEncoding.into());
    }
    let job_hash = array_at(&state, D_JOB_HASH_OFFSET)?;
    let transition = u32_at(&state, D_LOW_OFFSET)?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes()
        || instruction_index != transition
    {
        return Err(QxError::Unauthorized.into());
    }
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC || instruction_count != u32_at(&model_data, 136)? {
        return Err(QxError::InvalidState.into());
    }
    let instruction_hash = stable_hash_domain(
        D_PROGRAM_INSTRUCTION,
        &[&instruction_index.to_le_bytes(), &[op]],
    );
    verify_merkle_blake3(
        instruction_hash,
        instruction_index,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&model_data, 104)?,
        D_PROGRAM_TREE,
    )?;
    drop(model_data);
    let common = scalar_state_hash(&job_hash, transition, op, a, b, c, None);
    let challenger_output =
        scalar_state_hash(&job_hash, transition, op, a, b, c, Some(challenger_result));
    if common != array_at(&state, D_COMMON_ROOT_OFFSET)?
        || challenger_output != array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    state[D_RESOLUTION_FLAGS_OFFSET] = op;
    put(&mut state, D_TENSOR_OFFSET, &a.to_le_bytes())?;
    put(&mut state, D_ROW_OFFSET, &b.to_le_bytes())?;
    put(&mut state, D_COLUMN_OFFSET, &c.to_le_bytes())?;
    put(
        &mut state,
        D_CHALLENGER_BITS_OFFSET,
        &challenger_result.to_le_bytes(),
    )?;
    set_phase(&mut state, PHASE_MINER_TERMINAL, PARTY_MINER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn submit_miner_scalar(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, miner] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    let mut state = checked_party_state(
        program_id,
        dispute,
        miner,
        PHASE_MINER_TERMINAL,
        PARTY_MINER,
    )?;
    let defender_result = cursor.u32()?;
    if !cursor.is_empty() {
        return Err(QxError::InvalidEncoding.into());
    }
    let output = scalar_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_LOW_OFFSET)?,
        state[D_RESOLUTION_FLAGS_OFFSET],
        u32_at(&state, D_TENSOR_OFFSET)?,
        u32_at(&state, D_ROW_OFFSET)?,
        u32_at(&state, D_COLUMN_OFFSET)?,
        Some(defender_result),
    );
    if output != array_at(&state, D_DEFENDER_END_OFFSET)? {
        return Err(QxError::InvalidState.into());
    }
    put(
        &mut state,
        D_DEFENDER_BITS_OFFSET,
        &defender_result.to_le_bytes(),
    )?;
    set_phase(&mut state, PHASE_TERMINAL_READY, PARTY_CHALLENGER)?;
    refresh_deadline(&mut state, Clock::get()?.slot)
}

pub(super) fn resolve_scalar(
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
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_TERMINAL_READY)?;
    require_deadline(&state)?;
    let expected = evaluate_scalar(
        state[D_RESOLUTION_FLAGS_OFFSET],
        u32_at(&state, D_TENSOR_OFFSET)?,
        u32_at(&state, D_ROW_OFFSET)?,
        u32_at(&state, D_COLUMN_OFFSET)?,
    )?;
    let defender = u32_at(&state, D_DEFENDER_BITS_OFFSET)?;
    let challenger = u32_at(&state, D_CHALLENGER_BITS_OFFSET)?;
    let expected_bytes = expected.to_le_bytes();
    let agrees_with_challenger = expected_bytes == challenger.to_le_bytes();
    let disagrees_with_miner = expected_bytes != defender.to_le_bytes();
    let winner = if disagrees_with_miner && agrees_with_challenger {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    put(&mut state, D_ACC_START_OFFSET, &expected.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

/// Resolve one transition of the chained verification VM. Unlike the older
/// scalar terminal, operands are authenticated from the common memory root
/// and their addresses are bound by the registered instruction leaf.
pub(super) fn resolve_vm_scalar(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }

    let segment_index = cursor.u32()?;
    let program_counter = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    let opcode = cursor.u8()?;
    let source_a = cursor.u32()?;
    let source_b = cursor.u32()?;
    let source_c = cursor.u32()?;
    let destination = cursor.u32()?;
    let instruction_count = cursor.u32()?;
    let instruction_depth = usize::from(cursor.u8()?);
    let instruction_siblings = cursor.take(
        instruction_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if !scalar_op_supported(opcode)
        || program_counter != u32_at(&state, D_LOW_OFFSET)?
        || vm_state_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            program_counter,
            &residual_root,
            &global_kv_root,
        ) != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }

    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC
        || instruction_count != u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let instruction_bytes = vm_instruction_bytes(opcode, source_a, source_b, source_c, destination);
    let instruction_hash = stable_hash_domain(
        D_VM_INSTRUCTION,
        &[&program_counter.to_le_bytes(), &instruction_bytes],
    );
    verify_merkle_blake3(
        instruction_hash,
        program_counter,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&state, D_SEGMENT_PROGRAM_ROOT_OFFSET)?,
        D_PROGRAM_TREE,
    )?;
    drop(model_data);

    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let memory_depth = usize::from(cursor.u8()?);
    let memory_siblings = cursor.take(
        memory_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let memory_count = byte_length.div_ceil(STATE_CHUNK_BYTES as u32);
    if byte_length == 0
        || memory_count == 0
        || merkle_depth(memory_count)? != memory_depth
        || chunk_index >= memory_count
        || chunk_len
            != if chunk_index + 1 == memory_count {
                usize::try_from(byte_length - chunk_index * STATE_CHUNK_BYTES as u32)
                    .map_err(|_| QxError::InvalidEncoding)?
            } else {
                STATE_CHUNK_BYTES
            }
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let memory_leaf = stable_hash_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]);
    let opened_root = state_merkle_root(memory_leaf, chunk_index, memory_siblings);
    if opened_root != residual_root {
        return Err(QxError::InvalidReceiptProof.into());
    }
    let a = scalar_from_chunk(source_a, byte_length, chunk_index, chunk)?;
    let b = scalar_from_chunk(source_b, byte_length, chunk_index, chunk)?;
    let c = scalar_from_chunk(source_c, byte_length, chunk_index, chunk)?;
    let expected = evaluate_scalar(opcode, a, b, c)?;
    scalar_from_chunk(destination, byte_length, chunk_index, chunk)?;
    if !cursor.is_empty() {
        return Err(QxError::InvalidEncoding.into());
    }
    let chunk_start = chunk_index as usize * STATE_CHUNK_BYTES;
    let destination_offset = destination as usize - chunk_start;
    let mut updated_chunk = [0_u8; STATE_CHUNK_BYTES];
    updated_chunk[..chunk_len].copy_from_slice(chunk);
    updated_chunk[destination_offset..destination_offset + 4]
        .copy_from_slice(&expected.to_le_bytes());
    let updated_leaf = stable_hash_domain(
        D_STATE_CHUNK,
        &[&chunk_index.to_le_bytes(), &updated_chunk[..chunk_len]],
    );
    let post_residual_root = state_merkle_root(updated_leaf, chunk_index, memory_siblings);
    let expected_end = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        program_counter
            .checked_add(1)
            .ok_or(QxError::InvalidEncoding)?,
        &post_residual_root,
        &global_kv_root,
    );
    let agrees_with_challenger = expected_end == array_at(&state, D_CHALLENGER_END_OFFSET)?;
    let disagrees_with_miner = expected_end != array_at(&state, D_DEFENDER_END_OFFSET)?;
    let winner = if agrees_with_challenger && disagrees_with_miner {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    put(&mut state, D_ACC_START_OFFSET, &expected.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

/// Authenticate any subset of a scalar instruction's three operands from one
/// residual-memory chunk. Repeated calls may fill the same dispute-bound
/// staging account, allowing operands to reside in different chunks.
pub(super) fn stage_vm_scalar(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if dispute.owner != program_id
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || !staged.is_writable
        || staged.data_len() != STAGED_SCALAR_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let state = dispute.try_borrow_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    let role_mask = cursor.u8()?;
    let segment_index = cursor.u32()?;
    let program_counter = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    if role_mask == 0
        || role_mask & !7 != 0
        || program_counter != u32_at(&state, D_LOW_OFFSET)?
        || vm_state_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            program_counter,
            &residual_root,
            &global_kv_root,
        ) != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let opcode = cursor.u8()?;
    let source_a = cursor.u32()?;
    let source_b = cursor.u32()?;
    let source_c = cursor.u32()?;
    let destination = cursor.u32()?;
    let instruction_count = cursor.u32()?;
    let instruction_depth = usize::from(cursor.u8()?);
    let instruction_siblings = cursor.take(
        instruction_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if (!scalar_op_supported(opcode) && opcode != 14)
        || (matches!(opcode, 4 | 14) && role_mask & !3 != 0)
        || instruction_count != u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let instruction = vm_instruction_bytes(opcode, source_a, source_b, source_c, destination);
    verify_merkle_blake3(
        stable_hash_domain(
            D_VM_INSTRUCTION,
            &[&program_counter.to_le_bytes(), &instruction],
        ),
        program_counter,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&state, D_SEGMENT_PROGRAM_ROOT_OFFSET)?,
        D_PROGRAM_TREE,
    )?;

    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let memory_depth = usize::from(cursor.u8()?);
    let memory_siblings = cursor.take(
        memory_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let memory_count = byte_length.div_ceil(STATE_CHUNK_BYTES as u32);
    if memory_count == 0 || chunk_index >= memory_count {
        return Err(QxError::InvalidEncoding.into());
    }
    let expected_len = if chunk_index.checked_add(1) == Some(memory_count) {
        let chunk_start = chunk_index
            .checked_mul(STATE_CHUNK_BYTES as u32)
            .ok_or(QxError::InvalidEncoding)?;
        usize::try_from(
            byte_length
                .checked_sub(chunk_start)
                .ok_or(QxError::InvalidEncoding)?,
        )
        .map_err(|_| QxError::InvalidEncoding)?
    } else {
        STATE_CHUNK_BYTES
    };
    if !cursor.is_empty()
        || byte_length == 0
        || chunk_len != expected_len
        || merkle_depth(memory_count)? != memory_depth
        || state_merkle_root(
            stable_hash_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]),
            chunk_index,
            memory_siblings,
        ) != residual_root
    {
        return Err(QxError::InvalidReceiptProof.into());
    }
    let addresses = [source_a, source_b, source_c];
    let mut opened = [0_u32; 3];
    for role in 0..3 {
        if role_mask & (1 << role) != 0 {
            opened[role] = scalar_from_chunk(addresses[role], byte_length, chunk_index, chunk)?;
        }
    }
    let common_state = array_at(&state, D_COMMON_ROOT_OFFSET)?;
    drop(state);

    let mut staged_data = staged.try_borrow_mut_data()?;
    if staged_data.iter().all(|byte| *byte == 0) {
        staged_data[..4].copy_from_slice(STAGED_SCALAR_MAGIC);
        put(&mut staged_data, 4, dispute.key.as_ref())?;
        put(&mut staged_data, 36, &common_state)?;
        put(&mut staged_data, 68, &program_counter.to_le_bytes())?;
        put(&mut staged_data, 72, &instruction)?;
    } else if &staged_data[..4] != STAGED_SCALAR_MAGIC
        || staged_data[4..36] != dispute.key.to_bytes()
        || staged_data[36..68] != common_state
        || u32_at(&staged_data, 68)? != program_counter
        || staged_data[72..89] != instruction
    {
        return Err(QxError::Unauthorized.into());
    }
    for role in 0..3 {
        if role_mask & (1 << role) == 0 {
            continue;
        }
        let offset = 92 + role * 4;
        if staged_data[89] & (1 << role) != 0 && u32_at(&staged_data, offset)? != opened[role] {
            return Err(QxError::InvalidState.into());
        }
        put(&mut staged_data, offset, &opened[role].to_le_bytes())?;
    }
    staged_data[89] |= role_mask;
    Ok(())
}

/// Resolve a scalar instruction after all three operands were staged. Only
/// the destination chunk is transported here; its path derives the post-root.
pub(super) fn resolve_staged_vm_scalar(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || staged.data_len() != STAGED_SCALAR_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let staged_data = staged.try_borrow_data()?;
    if &staged_data[..4] != STAGED_SCALAR_MAGIC
        || staged_data[4..36] != dispute.key.to_bytes()
        || staged_data[89] != 7
        || matches!(staged_data[72], 4 | 14)
    {
        return Err(QxError::Unauthorized.into());
    }
    let common_state: [u8; 32] = staged_data[36..68]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let program_counter = u32_at(&staged_data, 68)?;
    let instruction: [u8; 17] = staged_data[72..89]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let operands = [
        u32_at(&staged_data, 92)?,
        u32_at(&staged_data, 96)?,
        u32_at(&staged_data, 100)?,
    ];
    drop(staged_data);

    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    if program_counter != u32_at(&state, D_LOW_OFFSET)?
        || common_state != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let segment_index = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    if vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        program_counter,
        &residual_root,
        &global_kv_root,
    ) != common_state
    {
        return Err(QxError::InvalidState.into());
    }
    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let memory_depth = usize::from(cursor.u8()?);
    let memory_siblings = cursor.take(
        memory_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let memory_count = byte_length.div_ceil(STATE_CHUNK_BYTES as u32);
    if memory_count == 0 || chunk_index >= memory_count {
        return Err(QxError::InvalidEncoding.into());
    }
    let expected_len = if chunk_index.checked_add(1) == Some(memory_count) {
        let chunk_start = chunk_index
            .checked_mul(STATE_CHUNK_BYTES as u32)
            .ok_or(QxError::InvalidEncoding)?;
        usize::try_from(
            byte_length
                .checked_sub(chunk_start)
                .ok_or(QxError::InvalidEncoding)?,
        )
        .map_err(|_| QxError::InvalidEncoding)?
    } else {
        STATE_CHUNK_BYTES
    };
    let destination = u32_from(&instruction, 13)?;
    if !cursor.is_empty()
        || byte_length == 0
        || chunk_len != expected_len
        || merkle_depth(memory_count)? != memory_depth
        || state_merkle_root(
            stable_hash_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]),
            chunk_index,
            memory_siblings,
        ) != residual_root
    {
        return Err(QxError::InvalidReceiptProof.into());
    }
    scalar_from_chunk(destination, byte_length, chunk_index, chunk)?;
    let expected = evaluate_scalar(instruction[0], operands[0], operands[1], operands[2])?;
    let chunk_start = chunk_index as usize * STATE_CHUNK_BYTES;
    let destination_offset = destination as usize - chunk_start;
    let mut updated_chunk = [0_u8; STATE_CHUNK_BYTES];
    updated_chunk[..chunk_len].copy_from_slice(chunk);
    updated_chunk[destination_offset..destination_offset + 4]
        .copy_from_slice(&expected.to_le_bytes());
    let post_residual_root = state_merkle_root(
        stable_hash_domain(
            D_STATE_CHUNK,
            &[&chunk_index.to_le_bytes(), &updated_chunk[..chunk_len]],
        ),
        chunk_index,
        memory_siblings,
    );
    let expected_end = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        program_counter
            .checked_add(1)
            .ok_or(QxError::InvalidEncoding)?,
        &post_residual_root,
        &global_kv_root,
    );
    let agrees_with_challenger = expected_end == array_at(&state, D_CHALLENGER_END_OFFSET)?;
    let disagrees_with_miner = expected_end != array_at(&state, D_DEFENDER_END_OFFSET)?;
    let winner = if agrees_with_challenger && disagrees_with_miner {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    put(&mut state, D_ACC_START_OFFSET, &expected.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

/// Authenticate the registered Q4 instruction and its model weight separately
/// from the state transition. A Granite weight proof is 26 hashes, so staging
/// keeps both transactions comfortably below Solana's transport limit while
/// binding the decoded value to this dispute and program counter.
pub(super) fn stage_vm_q4(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.owner != program_id
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || !staged.is_writable
        || staged.data_len() != STAGED_WEIGHT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let state = dispute.try_borrow_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }

    let program_counter = cursor.u32()?;
    let opcode = cursor.u8()?;
    let source_a = cursor.u32()?;
    let source_b = cursor.u32()?;
    let source_c = cursor.u32()?;
    let destination = cursor.u32()?;
    let instruction_count = cursor.u32()?;
    let instruction_depth = usize::from(cursor.u8()?);
    let instruction_siblings = cursor.take(
        instruction_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if opcode != 4 || program_counter != u32_at(&state, D_LOW_OFFSET)? || source_a & 3 != 0 {
        return Err(QxError::InvalidState.into());
    }
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC
        || instruction_count != u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let instruction_bytes = vm_instruction_bytes(opcode, source_a, source_b, source_c, destination);
    let instruction_hash = stable_hash_domain(
        D_VM_INSTRUCTION,
        &[&program_counter.to_le_bytes(), &instruction_bytes],
    );
    verify_merkle_blake3(
        instruction_hash,
        program_counter,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&state, D_SEGMENT_PROGRAM_ROOT_OFFSET)?,
        D_PROGRAM_TREE,
    )?;

    let leaf_index = cursor.u32()?;
    let tensor = cursor.u32()?;
    let row = cursor.u32()?;
    let group = cursor.u32()?;
    let quantization_type = cursor.u8()?;
    let scale_bits = cursor.u16()?;
    let packed = cursor.take(16)?;
    let weight_depth = usize::from(cursor.u8()?);
    let weight_siblings = cursor.take(
        weight_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let selected_leaf = source_c >> 5;
    let local = usize::try_from(source_c & 31).map_err(|_| QxError::InvalidEncoding)?;
    if !cursor.is_empty() || selected_leaf != leaf_index || quantization_type != Q4_TYPE {
        return Err(QxError::InvalidEncoding.into());
    }
    let leaf = weight_leaf_hash(tensor, row, group, quantization_type, packed, scale_bits);
    verify_merkle_sha(
        D_WEIGHT_TREE,
        leaf,
        leaf_index,
        u32_at(&model_data, 68)?,
        weight_depth,
        weight_siblings,
        array_at(&model_data, 36)?,
    )?;
    drop(model_data);
    drop(state);

    let code = if local & 1 == 0 {
        packed[local / 2] & 0x0f
    } else {
        packed[local / 2] >> 4
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
    put(&mut staged_data, 36, &program_counter.to_le_bytes())?;
    put(&mut staged_data, 40, &source_a.to_le_bytes())?;
    put(&mut staged_data, 44, &source_b.to_le_bytes())?;
    put(&mut staged_data, 48, &destination.to_le_bytes())?;
    put(&mut staged_data, 52, &weight_bits.to_le_bytes())?;
    put(&mut staged_data, 56, &leaf_index.to_le_bytes())?;
    Ok(())
}

/// Authenticate one FP32 value from a canonical raw-weight leaf. Raw leaves
/// contain up to 32 values (128 bytes) and cover Granite's norm vectors.
pub(super) fn stage_vm_raw_weight(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.owner != program_id
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || !staged.is_writable
        || staged.data_len() != STAGED_WEIGHT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let state = dispute.try_borrow_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let program_counter = cursor.u32()?;
    let opcode = cursor.u8()?;
    let source_a = cursor.u32()?;
    let source_b = cursor.u32()?;
    let source_c = cursor.u32()?;
    let destination = cursor.u32()?;
    let instruction_count = cursor.u32()?;
    let instruction_depth = usize::from(cursor.u8()?);
    let instruction_siblings = cursor.take(
        instruction_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if opcode != 14 || program_counter != u32_at(&state, D_LOW_OFFSET)? || source_a & 3 != 0 {
        return Err(QxError::InvalidState.into());
    }
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC
        || instruction_count != u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let instruction = vm_instruction_bytes(opcode, source_a, source_b, source_c, destination);
    verify_merkle_blake3(
        stable_hash_domain(
            D_VM_INSTRUCTION,
            &[&program_counter.to_le_bytes(), &instruction],
        ),
        program_counter,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&state, D_SEGMENT_PROGRAM_ROOT_OFFSET)?,
        D_PROGRAM_TREE,
    )?;

    let leaf_index = cursor.u32()?;
    let tensor = cursor.u32()?;
    let row = cursor.u32()?;
    let group = cursor.u32()?;
    let quantization_type = cursor.u8()?;
    let scale_bits = cursor.u16()?;
    let packed_len = usize::from(cursor.u16()?);
    let packed = cursor.take(packed_len)?;
    let weight_depth = usize::from(cursor.u8()?);
    let weight_siblings = cursor.take(
        weight_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let selected_leaf = source_c >> 5;
    let local = usize::try_from(source_c & 31).map_err(|_| QxError::InvalidEncoding)?;
    if !cursor.is_empty()
        || selected_leaf != leaf_index
        || quantization_type != 0
        || scale_bits != 0
        || packed_len == 0
        || packed_len > 128
        || !packed_len.is_multiple_of(4)
        || local >= packed_len / 4
    {
        return Err(QxError::InvalidEncoding.into());
    }
    verify_merkle_sha(
        D_WEIGHT_TREE,
        weight_leaf_hash(tensor, row, group, quantization_type, packed, scale_bits),
        leaf_index,
        u32_at(&model_data, 68)?,
        weight_depth,
        weight_siblings,
        array_at(&model_data, 36)?,
    )?;
    drop(model_data);
    drop(state);
    let weight_bits = u32::from_le_bytes(
        packed[local * 4..local * 4 + 4]
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    );
    if !f32::from_bits(weight_bits).is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let mut staged_data = staged.try_borrow_mut_data()?;
    staged_data.fill(0);
    staged_data[..4].copy_from_slice(STAGED_WEIGHT_MAGIC);
    put(&mut staged_data, 4, dispute.key.as_ref())?;
    put(&mut staged_data, 36, &program_counter.to_le_bytes())?;
    put(&mut staged_data, 40, &source_a.to_le_bytes())?;
    put(&mut staged_data, 44, &source_b.to_le_bytes())?;
    put(&mut staged_data, 48, &destination.to_le_bytes())?;
    put(&mut staged_data, 52, &weight_bits.to_le_bytes())?;
    put(&mut staged_data, 56, &leaf_index.to_le_bytes())?;
    Ok(())
}

/// Resolve one checkpoint-bound VM Q4 FMA using the staged authenticated
/// model weight and one opening of the common runtime state chunk.
pub(super) fn resolve_vm_q4(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || staged.data_len() != STAGED_WEIGHT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let staged_data = staged.try_borrow_data()?;
    if &staged_data[..4] != STAGED_WEIGHT_MAGIC || staged_data[4..36] != dispute.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let program_counter = u32_at(&staged_data, 36)?;
    let source_a = u32_at(&staged_data, 40)?;
    let source_b = u32_at(&staged_data, 44)?;
    let destination = u32_at(&staged_data, 48)?;
    let weight = f32::from_bits(u32_at(&staged_data, 52)?);
    drop(staged_data);

    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    let segment_index = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    if program_counter != u32_at(&state, D_LOW_OFFSET)?
        || vm_state_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            program_counter,
            &residual_root,
            &global_kv_root,
        ) != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let memory_depth = usize::from(cursor.u8()?);
    let memory_siblings = cursor.take(
        memory_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let memory_count = byte_length.div_ceil(STATE_CHUNK_BYTES as u32);
    if !cursor.is_empty()
        || byte_length == 0
        || memory_count == 0
        || merkle_depth(memory_count)? != memory_depth
        || chunk_index >= memory_count
        || chunk_len
            != if chunk_index + 1 == memory_count {
                usize::try_from(byte_length - chunk_index * STATE_CHUNK_BYTES as u32)
                    .map_err(|_| QxError::InvalidEncoding)?
            } else {
                STATE_CHUNK_BYTES
            }
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let memory_leaf = stable_hash_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]);
    if state_merkle_root(memory_leaf, chunk_index, memory_siblings) != residual_root {
        return Err(QxError::InvalidReceiptProof.into());
    }
    let activation = f32::from_bits(scalar_from_chunk(
        source_a,
        byte_length,
        chunk_index,
        chunk,
    )?);
    let accumulator = f32::from_bits(scalar_from_chunk(
        source_b,
        byte_length,
        chunk_index,
        chunk,
    )?);
    scalar_from_chunk(destination, byte_length, chunk_index, chunk)?;
    if !activation.is_finite() || !accumulator.is_finite() || !weight.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let expected = qx_consensus_math::fma_f32(weight, activation, accumulator);
    if !expected.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let chunk_start = chunk_index as usize * STATE_CHUNK_BYTES;
    let destination_offset = destination as usize - chunk_start;
    let mut updated_chunk = [0_u8; STATE_CHUNK_BYTES];
    updated_chunk[..chunk_len].copy_from_slice(chunk);
    updated_chunk[destination_offset..destination_offset + 4]
        .copy_from_slice(&expected.to_le_bytes());
    let updated_leaf = stable_hash_domain(
        D_STATE_CHUNK,
        &[&chunk_index.to_le_bytes(), &updated_chunk[..chunk_len]],
    );
    let post_residual_root = state_merkle_root(updated_leaf, chunk_index, memory_siblings);
    let expected_end = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        program_counter
            .checked_add(1)
            .ok_or(QxError::InvalidEncoding)?,
        &post_residual_root,
        &global_kv_root,
    );
    let agrees_with_challenger = expected_end == array_at(&state, D_CHALLENGER_END_OFFSET)?;
    let disagrees_with_miner = expected_end != array_at(&state, D_DEFENDER_END_OFFSET)?;
    let winner = if agrees_with_challenger && disagrees_with_miner {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    put(&mut state, D_ACC_START_OFFSET, &expected.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

/// Resolve a Q4 FMA whose activation, accumulator, and destination need not
/// share one residual chunk. The weight and scalar operands were independently
/// authenticated and bound to this dispute and registered instruction.
pub(super) fn resolve_staged_vm_q4(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, staged_weight, staged_operands] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged_weight.owner != program_id
        || staged_weight.data_len() != STAGED_WEIGHT_BYTES
        || staged_operands.owner != program_id
        || staged_operands.data_len() != STAGED_SCALAR_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let weight_data = staged_weight.try_borrow_data()?;
    if &weight_data[..4] != STAGED_WEIGHT_MAGIC || weight_data[4..36] != dispute.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let program_counter = u32_at(&weight_data, 36)?;
    let source_a = u32_at(&weight_data, 40)?;
    let source_b = u32_at(&weight_data, 44)?;
    let destination = u32_at(&weight_data, 48)?;
    let weight = f32::from_bits(u32_at(&weight_data, 52)?);
    drop(weight_data);

    let operand_data = staged_operands.try_borrow_data()?;
    if &operand_data[..4] != STAGED_SCALAR_MAGIC
        || operand_data[4..36] != dispute.key.to_bytes()
        || operand_data[89] != 3
        || u32_at(&operand_data, 68)? != program_counter
    {
        return Err(QxError::Unauthorized.into());
    }
    let common_state: [u8; 32] = operand_data[36..68]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let instruction: [u8; 17] = operand_data[72..89]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    if !matches!(instruction[0], 4 | 14)
        || u32_from(&instruction, 1)? != source_a
        || u32_from(&instruction, 5)? != source_b
        || u32_from(&instruction, 13)? != destination
    {
        return Err(QxError::InvalidState.into());
    }
    let activation = f32::from_bits(u32_at(&operand_data, 92)?);
    let accumulator = f32::from_bits(u32_at(&operand_data, 96)?);
    drop(operand_data);
    if !activation.is_finite() || !accumulator.is_finite() || !weight.is_finite() {
        return Err(QxError::NonFinite.into());
    }

    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    if program_counter != u32_at(&state, D_LOW_OFFSET)?
        || common_state != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let segment_index = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    if vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        program_counter,
        &residual_root,
        &global_kv_root,
    ) != common_state
    {
        return Err(QxError::InvalidState.into());
    }
    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let memory_depth = usize::from(cursor.u8()?);
    let memory_siblings = cursor.take(
        memory_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let memory_count = byte_length.div_ceil(STATE_CHUNK_BYTES as u32);
    if memory_count == 0 || chunk_index >= memory_count {
        return Err(QxError::InvalidEncoding.into());
    }
    let expected_len = if chunk_index.checked_add(1) == Some(memory_count) {
        let chunk_start = chunk_index
            .checked_mul(STATE_CHUNK_BYTES as u32)
            .ok_or(QxError::InvalidEncoding)?;
        usize::try_from(
            byte_length
                .checked_sub(chunk_start)
                .ok_or(QxError::InvalidEncoding)?,
        )
        .map_err(|_| QxError::InvalidEncoding)?
    } else {
        STATE_CHUNK_BYTES
    };
    if !cursor.is_empty()
        || byte_length == 0
        || chunk_len != expected_len
        || merkle_depth(memory_count)? != memory_depth
        || state_merkle_root(
            stable_hash_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]),
            chunk_index,
            memory_siblings,
        ) != residual_root
    {
        return Err(QxError::InvalidReceiptProof.into());
    }
    scalar_from_chunk(destination, byte_length, chunk_index, chunk)?;
    let expected = qx_consensus_math::fma_f32(weight, activation, accumulator);
    if !expected.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let chunk_start = chunk_index as usize * STATE_CHUNK_BYTES;
    let destination_offset = destination as usize - chunk_start;
    let mut updated_chunk = [0_u8; STATE_CHUNK_BYTES];
    updated_chunk[..chunk_len].copy_from_slice(chunk);
    updated_chunk[destination_offset..destination_offset + 4]
        .copy_from_slice(&expected.to_le_bytes());
    let post_residual_root = state_merkle_root(
        stable_hash_domain(
            D_STATE_CHUNK,
            &[&chunk_index.to_le_bytes(), &updated_chunk[..chunk_len]],
        ),
        chunk_index,
        memory_siblings,
    );
    let expected_end = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        program_counter
            .checked_add(1)
            .ok_or(QxError::InvalidEncoding)?,
        &post_residual_root,
        &global_kv_root,
    );
    let agrees_with_challenger = expected_end == array_at(&state, D_CHALLENGER_END_OFFSET)?;
    let disagrees_with_miner = expected_end != array_at(&state, D_DEFENDER_END_OFFSET)?;
    let winner = if agrees_with_challenger && disagrees_with_miner {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    put(
        &mut state,
        D_ACC_START_OFFSET,
        &expected.to_bits().to_le_bytes(),
    )?;
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

/// Prove that the current token position is empty in one layer's KV tree,
/// authenticate that layer root under the common global root, and stage the
/// unique roots after appending the supplied K/V vector roots.
pub(super) fn stage_vm_kv(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.owner != program_id
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || !staged.is_writable
        || staged.data_len() != STAGED_KV_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let state = dispute.try_borrow_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let segment_index = cursor.u32()?;
    let program_counter = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    let layer = cursor.u16()?;
    let key_root = cursor.array_32()?;
    let value_root = cursor.array_32()?;
    if program_counter != u32_at(&state, D_LOW_OFFSET)?
        || vm_state_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            program_counter,
            &residual_root,
            &global_kv_root,
        ) != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC {
        return Err(QxError::InvalidState.into());
    }
    let model_layers = u32::from(u16_from(&model_data, 144)?);
    let max_context = u32_at(&model_data, 148)?;
    let token_index = u32_at(&state, D_TOKEN_INDEX_OFFSET)?;
    if u32::from(layer) >= model_layers || token_index >= max_context {
        return Err(QxError::InvalidEncoding.into());
    }
    let context_leaves = max_context.checked_add(1).ok_or(QxError::InvalidEncoding)?;
    let layer_depth = usize::from(cursor.u8()?);
    let layer_siblings = cursor.take(
        layer_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let global_depth = usize::from(cursor.u8()?);
    let global_siblings = cursor.take(
        global_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if !cursor.is_empty()
        || layer_depth != merkle_depth(context_leaves)?
        || global_depth != merkle_depth(model_layers)?
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let empty = stable_hash_domain(D_MERKLE_EMPTY, &[D_KV]);
    let old_layer_root = domain_merkle_root(empty, token_index, layer_siblings, D_KV);
    let opened_global = domain_merkle_root(old_layer_root, u32::from(layer), global_siblings, D_KV);
    if opened_global != global_kv_root {
        return Err(QxError::InvalidReceiptProof.into());
    }
    let kv_leaf = stable_hash_domain(
        D_KV,
        &[
            &layer.to_le_bytes(),
            &token_index.to_le_bytes(),
            &key_root,
            &value_root,
        ],
    );
    let new_layer_root = domain_merkle_root(kv_leaf, token_index, layer_siblings, D_KV);
    let new_global_root =
        domain_merkle_root(new_layer_root, u32::from(layer), global_siblings, D_KV);
    drop(model_data);
    drop(state);

    let mut staged_data = staged.try_borrow_mut_data()?;
    staged_data.fill(0);
    staged_data[..4].copy_from_slice(STAGED_KV_MAGIC);
    put(&mut staged_data, 4, dispute.key.as_ref())?;
    put(&mut staged_data, 36, &program_counter.to_le_bytes())?;
    put(&mut staged_data, 40, &layer.to_le_bytes())?;
    put(&mut staged_data, 44, &key_root)?;
    put(&mut staged_data, 76, &value_root)?;
    put(&mut staged_data, 108, &new_global_root)?;
    Ok(())
}

/// Authenticate the KV instruction and the staged K/V roots against common
/// VM memory, then compare the unique post-append state with both endpoints.
pub(super) fn resolve_vm_kv(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || staged.data_len() != STAGED_KV_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let staged_data = staged.try_borrow_data()?;
    if &staged_data[..4] != STAGED_KV_MAGIC || staged_data[4..36] != dispute.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let staged_pc = u32_at(&staged_data, 36)?;
    let staged_layer = u32::from(u16_from(&staged_data, 40)?);
    let staged_key = array_at(&staged_data, 44)?;
    let staged_value = array_at(&staged_data, 76)?;
    let new_global_root = array_at(&staged_data, 108)?;
    drop(staged_data);

    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes()
        || staged_pc != u32_at(&state, D_LOW_OFFSET)?
    {
        return Err(QxError::Unauthorized.into());
    }
    let segment_index = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    if vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        staged_pc,
        &residual_root,
        &global_kv_root,
    ) != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let opcode = cursor.u8()?;
    let source_a = cursor.u32()?;
    let source_b = cursor.u32()?;
    let source_c = cursor.u32()?;
    let destination = cursor.u32()?;
    let instruction_count = cursor.u32()?;
    let instruction_depth = usize::from(cursor.u8()?);
    let instruction_siblings = cursor.take(
        instruction_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if opcode != 11 || source_c != staged_layer || destination != 0 {
        return Err(QxError::InvalidState.into());
    }
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC
        || instruction_count != u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let instruction = vm_instruction_bytes(opcode, source_a, source_b, source_c, destination);
    let instruction_hash =
        stable_hash_domain(D_VM_INSTRUCTION, &[&staged_pc.to_le_bytes(), &instruction]);
    verify_merkle_blake3(
        instruction_hash,
        staged_pc,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&state, D_SEGMENT_PROGRAM_ROOT_OFFSET)?,
        D_PROGRAM_TREE,
    )?;
    drop(model_data);

    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let memory_depth = usize::from(cursor.u8()?);
    let memory_siblings = cursor.take(
        memory_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let memory_count = byte_length.div_ceil(STATE_CHUNK_BYTES as u32);
    if !cursor.is_empty()
        || byte_length == 0
        || memory_count == 0
        || merkle_depth(memory_count)? != memory_depth
        || chunk_index >= memory_count
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let expected_len = if chunk_index + 1 == memory_count {
        usize::try_from(byte_length - chunk_index * STATE_CHUNK_BYTES as u32)
            .map_err(|_| QxError::InvalidEncoding)?
    } else {
        STATE_CHUNK_BYTES
    };
    if chunk_len != expected_len {
        return Err(QxError::InvalidEncoding.into());
    }
    let memory_leaf = stable_hash_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]);
    if state_merkle_root(memory_leaf, chunk_index, memory_siblings) != residual_root
        || bytes32_from_chunk(source_a, byte_length, chunk_index, chunk)? != staged_key
        || bytes32_from_chunk(source_b, byte_length, chunk_index, chunk)? != staged_value
    {
        return Err(QxError::InvalidReceiptProof.into());
    }
    let expected_end = vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        staged_pc.checked_add(1).ok_or(QxError::InvalidEncoding)?,
        &residual_root,
        &new_global_root,
    );
    let agrees_with_challenger = expected_end == array_at(&state, D_CHALLENGER_END_OFFSET)?;
    let disagrees_with_miner = expected_end != array_at(&state, D_DEFENDER_END_OFFSET)?;
    let winner = if agrees_with_challenger && disagrees_with_miner {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

/// Authenticate either source `[logit_bits, token_id]` record for an argmax
/// instruction. Two calls permit arbitrary source chunks while keeping each
/// witness below Solana's transaction transport limit.
pub(super) fn stage_vm_argmax(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if dispute.owner != program_id
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || !staged.is_writable
        || staged.data_len() != STAGED_ARGMAX_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let state = dispute.try_borrow_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    let record_mask = cursor.u8()?;
    let segment_index = cursor.u32()?;
    let program_counter = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    if record_mask == 0
        || record_mask & !3 != 0
        || program_counter != u32_at(&state, D_LOW_OFFSET)?
        || vm_state_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            program_counter,
            &residual_root,
            &global_kv_root,
        ) != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let opcode = cursor.u8()?;
    let source_a = cursor.u32()?;
    let source_b = cursor.u32()?;
    let source_c = cursor.u32()?;
    let destination = cursor.u32()?;
    let instruction_count = cursor.u32()?;
    let instruction_depth = usize::from(cursor.u8()?);
    let instruction_siblings = cursor.take(
        instruction_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if opcode != 8
        || source_c != 0
        || instruction_count != u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let instruction = vm_instruction_bytes(opcode, source_a, source_b, source_c, destination);
    verify_merkle_blake3(
        stable_hash_domain(
            D_VM_INSTRUCTION,
            &[&program_counter.to_le_bytes(), &instruction],
        ),
        program_counter,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&state, D_SEGMENT_PROGRAM_ROOT_OFFSET)?,
        D_PROGRAM_TREE,
    )?;

    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let memory_depth = usize::from(cursor.u8()?);
    let memory_siblings = cursor.take(
        memory_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let memory_count = byte_length.div_ceil(STATE_CHUNK_BYTES as u32);
    if !cursor.is_empty()
        || byte_length == 0
        || memory_count == 0
        || chunk_index >= memory_count
        || merkle_depth(memory_count)? != memory_depth
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let expected_len = if chunk_index.checked_add(1) == Some(memory_count) {
        usize::try_from(
            byte_length
                .checked_sub(
                    chunk_index
                        .checked_mul(STATE_CHUNK_BYTES as u32)
                        .ok_or(QxError::InvalidEncoding)?,
                )
                .ok_or(QxError::InvalidEncoding)?,
        )
        .map_err(|_| QxError::InvalidEncoding)?
    } else {
        STATE_CHUNK_BYTES
    };
    if chunk_len != expected_len
        || state_merkle_root(
            stable_hash_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]),
            chunk_index,
            memory_siblings,
        ) != residual_root
    {
        return Err(QxError::InvalidReceiptProof.into());
    }
    let addresses = [source_a, source_b];
    let mut records = [[0_u32; 2]; 2];
    for role in 0..2 {
        if record_mask & (1 << role) == 0 {
            continue;
        }
        records[role][0] = scalar_from_chunk(addresses[role], byte_length, chunk_index, chunk)?;
        records[role][1] = scalar_from_chunk(
            addresses[role]
                .checked_add(4)
                .ok_or(QxError::InvalidEncoding)?,
            byte_length,
            chunk_index,
            chunk,
        )?;
    }
    let common_state = array_at(&state, D_COMMON_ROOT_OFFSET)?;
    drop(state);

    let mut staged_data = staged.try_borrow_mut_data()?;
    if staged_data.iter().all(|byte| *byte == 0) {
        staged_data[..4].copy_from_slice(STAGED_ARGMAX_MAGIC);
        put(&mut staged_data, 4, dispute.key.as_ref())?;
        put(&mut staged_data, 36, &common_state)?;
        put(&mut staged_data, 68, &program_counter.to_le_bytes())?;
        put(&mut staged_data, 72, &instruction)?;
    } else if &staged_data[..4] != STAGED_ARGMAX_MAGIC
        || staged_data[4..36] != dispute.key.to_bytes()
        || staged_data[36..68] != common_state
        || u32_at(&staged_data, 68)? != program_counter
        || staged_data[72..89] != instruction
    {
        return Err(QxError::Unauthorized.into());
    }
    for role in 0..2 {
        if record_mask & (1 << role) == 0 {
            continue;
        }
        let offset = 92 + role * 8;
        if staged_data[89] & (1 << role) != 0
            && (u32_at(&staged_data, offset)? != records[role][0]
                || u32_at(&staged_data, offset + 4)? != records[role][1])
        {
            return Err(QxError::InvalidState.into());
        }
        put(&mut staged_data, offset, &records[role][0].to_le_bytes())?;
        put(
            &mut staged_data,
            offset + 4,
            &records[role][1].to_le_bytes(),
        )?;
    }
    staged_data[89] |= record_mask;
    Ok(())
}

/// Resolve an argmax comparison after both source records were staged. The
/// destination record may be in a third chunk and the last comparison binds
/// its selected token into the final-head output-state domain.
pub(super) fn resolve_staged_vm_argmax(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute, staged] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
        || staged.owner != program_id
        || staged.data_len() != STAGED_ARGMAX_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let staged_data = staged.try_borrow_data()?;
    if &staged_data[..4] != STAGED_ARGMAX_MAGIC
        || staged_data[4..36] != dispute.key.to_bytes()
        || staged_data[89] != 3
        || staged_data[72] != 8
    {
        return Err(QxError::Unauthorized.into());
    }
    let common_state: [u8; 32] = staged_data[36..68]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let program_counter = u32_at(&staged_data, 68)?;
    let destination = u32_at(&staged_data, 85)?;
    let left_logit = u32_at(&staged_data, 92)?;
    let left_token = u32_at(&staged_data, 96)?;
    let right_logit = u32_at(&staged_data, 100)?;
    let right_token = u32_at(&staged_data, 104)?;
    drop(staged_data);

    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC {
        return Err(QxError::InvalidState.into());
    }
    let vocab_size = u32_at(&model_data, 152)?;
    drop(model_data);
    if !f32::from_bits(left_logit).is_finite()
        || !f32::from_bits(right_logit).is_finite()
        || left_token >= vocab_size
        || right_token >= vocab_size
        || left_token == right_token
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let (selected_logit, selected) = if finite_key(right_logit) > finite_key(left_logit)
        || (finite_key(right_logit) == finite_key(left_logit) && right_token < left_token)
    {
        (right_logit, right_token)
    } else {
        (left_logit, left_token)
    };

    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes()
        || program_counter != u32_at(&state, D_LOW_OFFSET)?
        || common_state != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let segment_index = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    if vm_state_hash(
        &array_at(&state, D_JOB_HASH_OFFSET)?,
        u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
        segment_index,
        program_counter,
        &residual_root,
        &global_kv_root,
    ) != common_state
    {
        return Err(QxError::InvalidState.into());
    }
    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let memory_depth = usize::from(cursor.u8()?);
    let memory_siblings = cursor.take(
        memory_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let memory_count = byte_length.div_ceil(STATE_CHUNK_BYTES as u32);
    if !cursor.is_empty()
        || byte_length == 0
        || memory_count == 0
        || chunk_index >= memory_count
        || merkle_depth(memory_count)? != memory_depth
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let expected_len = if chunk_index.checked_add(1) == Some(memory_count) {
        usize::try_from(
            byte_length
                .checked_sub(
                    chunk_index
                        .checked_mul(STATE_CHUNK_BYTES as u32)
                        .ok_or(QxError::InvalidEncoding)?,
                )
                .ok_or(QxError::InvalidEncoding)?,
        )
        .map_err(|_| QxError::InvalidEncoding)?
    } else {
        STATE_CHUNK_BYTES
    };
    if chunk_len != expected_len
        || state_merkle_root(
            stable_hash_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]),
            chunk_index,
            memory_siblings,
        ) != residual_root
    {
        return Err(QxError::InvalidReceiptProof.into());
    }
    scalar_from_chunk(destination, byte_length, chunk_index, chunk)?;
    scalar_from_chunk(
        destination.checked_add(4).ok_or(QxError::InvalidEncoding)?,
        byte_length,
        chunk_index,
        chunk,
    )?;
    let chunk_start = chunk_index as usize * STATE_CHUNK_BYTES;
    let destination_offset = destination as usize - chunk_start;
    let mut updated_chunk = [0_u8; STATE_CHUNK_BYTES];
    updated_chunk[..chunk_len].copy_from_slice(chunk);
    updated_chunk[destination_offset..destination_offset + 4]
        .copy_from_slice(&selected_logit.to_le_bytes());
    updated_chunk[destination_offset + 4..destination_offset + 8]
        .copy_from_slice(&selected.to_le_bytes());
    let post_residual_root = state_merkle_root(
        stable_hash_domain(
            D_STATE_CHUNK,
            &[&chunk_index.to_le_bytes(), &updated_chunk[..chunk_len]],
        ),
        chunk_index,
        memory_siblings,
    );
    let next_program_counter = program_counter
        .checked_add(1)
        .ok_or(QxError::InvalidEncoding)?;
    let expected_end = if state[D_RESOLUTION_FLAGS_OFFSET] == TRACE_KIND_FINAL_HEAD
        && next_program_counter == u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?
    {
        vm_output_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            next_program_counter,
            &post_residual_root,
            &global_kv_root,
            selected,
        )
    } else {
        vm_state_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            next_program_counter,
            &post_residual_root,
            &global_kv_root,
        )
    };
    let agrees_with_challenger = expected_end == array_at(&state, D_CHALLENGER_END_OFFSET)?;
    let disagrees_with_miner = expected_end != array_at(&state, D_DEFENDER_END_OFFSET)?;
    let winner = if agrees_with_challenger && disagrees_with_miner {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    put(&mut state, D_ACC_START_OFFSET, &selected.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

/// Resolve one pairwise greedy-output reduction. Source and destination
/// records are `[logit_bits, token_id]`; copying both fields lets a registered
/// program reduce the entire vocabulary as a deterministic tournament. A
/// higher finite logit wins and equal logits select the lower token ID.
pub(super) fn resolve_vm_argmax(
    program_id: &Pubkey,
    accounts: &[AccountInfo],
    cursor: &mut Cursor<'_>,
) -> ProgramResult {
    let [model, dispute] = accounts else {
        return Err(QxError::InvalidEncoding.into());
    };
    if model.owner != program_id
        || model.data_len() != MODEL_ACCOUNT_BYTES
        || dispute.owner != program_id
        || !dispute.is_writable
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, PHASE_CHALLENGER_TERMINAL)?;
    require_deadline(&state)?;
    if state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes() {
        return Err(QxError::Unauthorized.into());
    }
    let segment_index = cursor.u32()?;
    let program_counter = cursor.u32()?;
    let residual_root = cursor.array_32()?;
    let global_kv_root = cursor.array_32()?;
    if program_counter != u32_at(&state, D_LOW_OFFSET)?
        || vm_state_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            program_counter,
            &residual_root,
            &global_kv_root,
        ) != array_at(&state, D_COMMON_ROOT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let opcode = cursor.u8()?;
    let source_a = cursor.u32()?;
    let source_b = cursor.u32()?;
    let source_c = cursor.u32()?;
    let destination = cursor.u32()?;
    let instruction_count = cursor.u32()?;
    let instruction_depth = usize::from(cursor.u8()?);
    let instruction_siblings = cursor.take(
        instruction_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    if opcode != 8 || source_c != 0 {
        return Err(QxError::InvalidState.into());
    }
    let model_data = model.try_borrow_data()?;
    if &model_data[..4] != MODEL_MAGIC
        || instruction_count != u32_at(&state, D_SEGMENT_INSTRUCTION_COUNT_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    let instruction = vm_instruction_bytes(opcode, source_a, source_b, source_c, destination);
    let instruction_hash = stable_hash_domain(
        D_VM_INSTRUCTION,
        &[&program_counter.to_le_bytes(), &instruction],
    );
    verify_merkle_blake3(
        instruction_hash,
        program_counter,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&state, D_SEGMENT_PROGRAM_ROOT_OFFSET)?,
        D_PROGRAM_TREE,
    )?;
    let vocab_size = u32_at(&model_data, 152)?;
    drop(model_data);

    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let memory_depth = usize::from(cursor.u8()?);
    let memory_siblings = cursor.take(
        memory_depth
            .checked_mul(32)
            .ok_or(QxError::InvalidEncoding)?,
    )?;
    let memory_count = byte_length.div_ceil(STATE_CHUNK_BYTES as u32);
    if !cursor.is_empty()
        || byte_length == 0
        || memory_count == 0
        || merkle_depth(memory_count)? != memory_depth
        || chunk_index >= memory_count
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let expected_len = if chunk_index + 1 == memory_count {
        usize::try_from(byte_length - chunk_index * STATE_CHUNK_BYTES as u32)
            .map_err(|_| QxError::InvalidEncoding)?
    } else {
        STATE_CHUNK_BYTES
    };
    if chunk_len != expected_len {
        return Err(QxError::InvalidEncoding.into());
    }
    let memory_leaf = stable_hash_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]);
    if state_merkle_root(memory_leaf, chunk_index, memory_siblings) != residual_root {
        return Err(QxError::InvalidReceiptProof.into());
    }
    let left_logit = scalar_from_chunk(source_a, byte_length, chunk_index, chunk)?;
    let left_token = scalar_from_chunk(
        source_a.checked_add(4).ok_or(QxError::InvalidEncoding)?,
        byte_length,
        chunk_index,
        chunk,
    )?;
    let right_logit = scalar_from_chunk(source_b, byte_length, chunk_index, chunk)?;
    let right_token = scalar_from_chunk(
        source_b.checked_add(4).ok_or(QxError::InvalidEncoding)?,
        byte_length,
        chunk_index,
        chunk,
    )?;
    scalar_from_chunk(destination, byte_length, chunk_index, chunk)?;
    scalar_from_chunk(
        destination.checked_add(4).ok_or(QxError::InvalidEncoding)?,
        byte_length,
        chunk_index,
        chunk,
    )?;
    if !f32::from_bits(left_logit).is_finite()
        || !f32::from_bits(right_logit).is_finite()
        || left_token >= vocab_size
        || right_token >= vocab_size
        || left_token == right_token
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let left_key = finite_key(left_logit);
    let right_key = finite_key(right_logit);
    let (selected_logit, selected) =
        if right_key > left_key || (right_key == left_key && right_token < left_token) {
            (right_logit, right_token)
        } else {
            (left_logit, left_token)
        };
    let chunk_start = chunk_index as usize * STATE_CHUNK_BYTES;
    let destination_offset = destination as usize - chunk_start;
    let mut updated_chunk = [0_u8; STATE_CHUNK_BYTES];
    updated_chunk[..chunk_len].copy_from_slice(chunk);
    updated_chunk[destination_offset..destination_offset + 4]
        .copy_from_slice(&selected_logit.to_le_bytes());
    updated_chunk[destination_offset + 4..destination_offset + 8]
        .copy_from_slice(&selected.to_le_bytes());
    let updated_leaf = stable_hash_domain(
        D_STATE_CHUNK,
        &[&chunk_index.to_le_bytes(), &updated_chunk[..chunk_len]],
    );
    let post_residual_root = state_merkle_root(updated_leaf, chunk_index, memory_siblings);
    let next_program_counter = program_counter
        .checked_add(1)
        .ok_or(QxError::InvalidEncoding)?;
    let expected_end = if state[D_RESOLUTION_FLAGS_OFFSET] == TRACE_KIND_FINAL_HEAD
        && next_program_counter == instruction_count
    {
        vm_output_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            next_program_counter,
            &post_residual_root,
            &global_kv_root,
            selected,
        )
    } else {
        vm_state_hash(
            &array_at(&state, D_JOB_HASH_OFFSET)?,
            u32_at(&state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            next_program_counter,
            &post_residual_root,
            &global_kv_root,
        )
    };
    let agrees_with_challenger = expected_end == array_at(&state, D_CHALLENGER_END_OFFSET)?;
    let disagrees_with_miner = expected_end != array_at(&state, D_DEFENDER_END_OFFSET)?;
    let winner = if agrees_with_challenger && disagrees_with_miner {
        PARTY_CHALLENGER
    } else {
        PARTY_MINER
    };
    put(&mut state, D_ACC_START_OFFSET, &selected.to_le_bytes())?;
    state[D_RESOLUTION_FLAGS_OFFSET] =
        u8::from(agrees_with_challenger) | (u8::from(disagrees_with_miner) << 1);
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

pub(super) fn timeout(
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
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let mut state = dispute.try_borrow_mut_data()?;
    if &state[..4] != DISPUTE_MAGIC_V2
        || state[D_PHASE_OFFSET] == PHASE_RESOLVED_V2
        || Clock::get()?.slot <= u64_at(&state, D_DEADLINE_OFFSET)?
    {
        return Err(QxError::Deadline.into());
    }
    let winner = match state[D_EXPECTED_PARTY_OFFSET] {
        PARTY_MINER => PARTY_CHALLENGER,
        PARTY_CHALLENGER => PARTY_MINER,
        _ => return Err(QxError::InvalidState.into()),
    };
    state[D_PHASE_OFFSET] = PHASE_RESOLVED_V2;
    state[D_WINNER_OFFSET] = winner;
    state[D_EXPECTED_PARTY_OFFSET] = PARTY_NONE;
    Ok(())
}

fn vm_instruction_bytes(
    opcode: u8,
    source_a: u32,
    source_b: u32,
    source_c: u32,
    destination: u32,
) -> [u8; 17] {
    let mut bytes = [0_u8; 17];
    bytes[0] = opcode;
    bytes[1..5].copy_from_slice(&source_a.to_le_bytes());
    bytes[5..9].copy_from_slice(&source_b.to_le_bytes());
    bytes[9..13].copy_from_slice(&source_c.to_le_bytes());
    bytes[13..17].copy_from_slice(&destination.to_le_bytes());
    bytes
}

fn vm_state_hash(
    job_hash: &[u8; 32],
    token_index: u32,
    segment_index: u32,
    program_counter: u32,
    residual_root: &[u8; 32],
    global_kv_root: &[u8; 32],
) -> [u8; 32] {
    stable_hash_domain(
        D_VM_STATE,
        &[
            job_hash,
            &token_index.to_le_bytes(),
            &segment_index.to_le_bytes(),
            &program_counter.to_le_bytes(),
            residual_root,
            global_kv_root,
        ],
    )
}

fn vm_output_hash(
    job_hash: &[u8; 32],
    token_index: u32,
    segment_index: u32,
    program_counter: u32,
    residual_root: &[u8; 32],
    global_kv_root: &[u8; 32],
    predicted_token: u32,
) -> [u8; 32] {
    stable_hash_domain(
        D_VM_OUTPUT,
        &[
            job_hash,
            &token_index.to_le_bytes(),
            &segment_index.to_le_bytes(),
            &program_counter.to_le_bytes(),
            residual_root,
            global_kv_root,
            &predicted_token.to_le_bytes(),
        ],
    )
}

fn granite_macro_slot(instruction: &[u8]) -> Result<u8, ProgramError> {
    if instruction.len() != 17 {
        return Err(QxError::InvalidEncoding.into());
    }
    let slot = match instruction[0] {
        65 => {
            if u32_from(instruction, 1)? == u32::MAX {
                1
            } else if u32_from(instruction, 5)? % 71 == 6 {
                10
            } else {
                1
            }
        }
        66 => match u32_from(instruction, 13)? {
            1 => 2,
            2 => 3,
            3 => 4,
            4 => 8,
            5 => 11,
            _ => return Err(QxError::InvalidEncoding.into()),
        },
        67 => 5,
        68 => 6,
        69 => 7,
        70 => match u32_from(instruction, 9)? {
            0 => 9,
            1 => 0,
            _ => return Err(QxError::InvalidEncoding.into()),
        },
        71 => 12,
        72 => 13,
        73 => 14,
        74 => 15,
        75 => 0,
        _ => return Err(QxError::InvalidEncoding.into()),
    };
    Ok(slot)
}

fn macro_workspace_leaf(slot: u8, value_root: &[u8; 32]) -> [u8; 32] {
    stable_hash_domain(
        D_MACRO_WORKSPACE_SLOT,
        &[&u32::from(slot).to_le_bytes(), value_root],
    )
}

fn empty_global_kv_root(layer_count: u32) -> Result<[u8; 32], ProgramError> {
    let depth = merkle_depth(layer_count)?;
    let mut root = stable_hash_domain(D_MERKLE_EMPTY, &[D_KV]);
    for _ in 0..depth {
        root = blake3_merkle_parent(D_KV, &root, &root);
    }
    Ok(root)
}

fn macro_workspace_root(slot: u8, value_root: &[u8; 32], siblings: &[u8]) -> [u8; 32] {
    domain_merkle_root(
        macro_workspace_leaf(slot, value_root),
        u32::from(slot),
        siblings,
        D_MACRO_WORKSPACE,
    )
}

fn parse_macro_workspace_value(
    cursor: &mut Cursor<'_>,
    slot: u8,
    expected_workspace_root: [u8; 32],
) -> Result<[u8; 32], ProgramError> {
    let value_root = cursor.array_32()?;
    let siblings = cursor.take(4 * 32)?;
    if macro_workspace_root(slot, &value_root, siblings) != expected_workspace_root {
        return Err(QxError::InvalidReceiptProof.into());
    }
    Ok(value_root)
}

fn parse_vector_scalar(
    cursor: &mut Cursor<'_>,
    element: u32,
    expected_root: [u8; 32],
    state_chunks: bool,
) -> Result<u32, ProgramError> {
    let byte_length = cursor.u32()?;
    let chunk_index = cursor.u32()?;
    let chunk_len = usize::from(cursor.u16()?);
    let chunk = cursor.take(chunk_len)?;
    let depth = usize::from(cursor.u8()?);
    let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    let chunk_bytes = if state_chunks {
        STATE_CHUNK_BYTES
    } else {
        ACTIVATION_CHUNK_BYTES
    };
    let leaf_count = byte_length.div_ceil(chunk_bytes as u32);
    let expected_chunk_len = if chunk_index.checked_add(1) == Some(leaf_count) {
        usize::try_from(
            byte_length
                .checked_sub(
                    chunk_index
                        .checked_mul(chunk_bytes as u32)
                        .ok_or(QxError::InvalidEncoding)?,
                )
                .ok_or(QxError::InvalidEncoding)?,
        )
        .map_err(|_| QxError::InvalidEncoding)?
    } else {
        chunk_bytes
    };
    let byte_address = element.checked_mul(4).ok_or(QxError::InvalidEncoding)?;
    let chunk_start = chunk_index
        .checked_mul(chunk_bytes as u32)
        .ok_or(QxError::InvalidEncoding)?;
    let offset = byte_address
        .checked_sub(chunk_start)
        .ok_or(QxError::InvalidEncoding)? as usize;
    if byte_length != 1024 * 4
        || chunk_index >= leaf_count
        || chunk_len != expected_chunk_len
        || merkle_depth(leaf_count)? != depth
        || offset.checked_add(4).ok_or(QxError::InvalidEncoding)? > chunk_len
    {
        return Err(QxError::InvalidEncoding.into());
    }
    let leaf_domain = if state_chunks {
        D_STATE_CHUNK
    } else {
        D_ACTIVATION_CHUNK
    };
    let leaf = stable_hash_domain(leaf_domain, &[&chunk_index.to_le_bytes(), chunk]);
    if state_merkle_root(leaf, chunk_index, siblings) != expected_root {
        return Err(QxError::InvalidActivationProof.into());
    }
    Ok(u32::from_le_bytes(
        chunk[offset..offset + 4]
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    ))
}

fn initial_macro_workspace_root(residual_root: &[u8; 32]) -> [u8; 32] {
    let zero = [0_u8; 32];
    let mut nodes = [[0_u8; 32]; 16];
    for (slot, node) in nodes.iter_mut().enumerate() {
        *node = macro_workspace_leaf(slot as u8, if slot == 0 { residual_root } else { &zero });
    }
    let mut width = nodes.len();
    while width > 1 {
        for index in 0..width / 2 {
            nodes[index] =
                blake3_merkle_parent(D_MACRO_WORKSPACE, &nodes[index * 2], &nodes[index * 2 + 1]);
        }
        width /= 2;
    }
    nodes[0]
}

fn macro_state_hash(
    job_hash: &[u8; 32],
    token_index: u32,
    segment_index: u32,
    program_counter: u32,
    workspace_root: &[u8; 32],
    global_kv_root: &[u8; 32],
) -> [u8; 32] {
    stable_hash_domain(
        D_MACRO_STATE,
        &[
            job_hash,
            &token_index.to_le_bytes(),
            &segment_index.to_le_bytes(),
            &program_counter.to_le_bytes(),
            workspace_root,
            global_kv_root,
        ],
    )
}

fn validate_macro_kv_transition(
    opcode: u8,
    pre_kv_root: &[u8; 32],
    post_kv_root: &[u8; 32],
) -> ProgramResult {
    if (opcode == 68) == (pre_kv_root == post_kv_root) {
        return Err(QxError::InvalidState.into());
    }
    Ok(())
}

fn validate_macro_pre_boundary(
    state: &[u8],
    segment_index: u32,
    program_counter: u32,
    workspace_root: &[u8; 32],
    global_kv_root: &[u8; 32],
) -> ProgramResult {
    let expected = if program_counter == 0 {
        let residual_root = array_at(state, D_COMMON_RESIDUAL_OFFSET)?;
        if *workspace_root != initial_macro_workspace_root(&residual_root)
            || *global_kv_root != array_at(state, D_COMMON_KV_OFFSET)?
        {
            return Err(QxError::InvalidState.into());
        }
        vm_state_hash(
            &array_at(state, D_JOB_HASH_OFFSET)?,
            u32_at(state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            0,
            &residual_root,
            global_kv_root,
        )
    } else {
        macro_state_hash(
            &array_at(state, D_JOB_HASH_OFFSET)?,
            u32_at(state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            program_counter,
            workspace_root,
            global_kv_root,
        )
    };
    if expected != array_at(state, D_COMMON_ROOT_OFFSET)? {
        return Err(QxError::InvalidState.into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn macro_post_boundary_hash(
    state: &[u8],
    segment_index: u32,
    program_counter: u32,
    instruction_count: u32,
    slot: u8,
    output_root: &[u8; 32],
    workspace_root: &[u8; 32],
    global_kv_root: &[u8; 32],
    challenger: bool,
) -> Result<[u8; 32], ProgramError> {
    let next = program_counter
        .checked_add(1)
        .ok_or(QxError::InvalidEncoding)?;
    if next < instruction_count {
        return Ok(macro_state_hash(
            &array_at(state, D_JOB_HASH_OFFSET)?,
            u32_at(state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            next,
            workspace_root,
            global_kv_root,
        ));
    }
    if next != instruction_count {
        return Err(QxError::InvalidState.into());
    }
    match state[D_RESOLUTION_FLAGS_OFFSET] {
        TRACE_KIND_OPERATOR => {
            if slot != 0 {
                return Err(QxError::InvalidState.into());
            }
            Ok(vm_state_hash(
                &array_at(state, D_JOB_HASH_OFFSET)?,
                u32_at(state, D_TOKEN_INDEX_OFFSET)?,
                segment_index,
                next,
                output_root,
                global_kv_root,
            ))
        }
        TRACE_KIND_FINAL_HEAD => Ok(vm_output_hash(
            &array_at(state, D_JOB_HASH_OFFSET)?,
            u32_at(state, D_TOKEN_INDEX_OFFSET)?,
            segment_index,
            next,
            workspace_root,
            global_kv_root,
            u32_at(
                state,
                if challenger {
                    D_CHALLENGER_PREDICTION_OFFSET
                } else {
                    D_DEFENDER_PREDICTION_OFFSET
                },
            )?,
        )),
        _ => Err(QxError::InvalidState.into()),
    }
}

fn scalar_from_chunk(
    address: u32,
    byte_length: u32,
    chunk_index: u32,
    chunk: &[u8],
) -> Result<u32, ProgramError> {
    if address & 3 != 0 || address.checked_add(4).ok_or(QxError::InvalidEncoding)? > byte_length {
        return Err(QxError::InvalidEncoding.into());
    }
    let chunk_start = chunk_index
        .checked_mul(STATE_CHUNK_BYTES as u32)
        .ok_or(QxError::InvalidEncoding)?;
    let offset = address
        .checked_sub(chunk_start)
        .ok_or(QxError::InvalidEncoding)? as usize;
    Ok(u32::from_le_bytes(
        chunk
            .get(offset..offset + 4)
            .ok_or(QxError::InvalidEncoding)?
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    ))
}

fn state_merkle_root(mut node: [u8; 32], mut index: u32, siblings: &[u8]) -> [u8; 32] {
    for sibling in siblings.chunks_exact(32) {
        let sibling: &[u8; 32] = sibling.try_into().expect("exact chunks");
        node = if index & 1 == 0 {
            crate::stable_hashv(&[&node, sibling])
        } else {
            crate::stable_hashv(&[sibling, &node])
        };
        index >>= 1;
    }
    node
}

fn domain_merkle_root(
    mut node: [u8; 32],
    mut index: u32,
    siblings: &[u8],
    tree_domain: &[u8],
) -> [u8; 32] {
    for sibling in siblings.chunks_exact(32) {
        let sibling: &[u8; 32] = sibling.try_into().expect("exact chunks");
        node = if index & 1 == 0 {
            blake3_merkle_parent(tree_domain, &node, sibling)
        } else {
            blake3_merkle_parent(tree_domain, sibling, &node)
        };
        index >>= 1;
    }
    node
}

fn bytes32_from_chunk(
    address: u32,
    byte_length: u32,
    chunk_index: u32,
    chunk: &[u8],
) -> Result<[u8; 32], ProgramError> {
    if address.checked_add(32).ok_or(QxError::InvalidEncoding)? > byte_length {
        return Err(QxError::InvalidEncoding.into());
    }
    let chunk_start = chunk_index
        .checked_mul(STATE_CHUNK_BYTES as u32)
        .ok_or(QxError::InvalidEncoding)?;
    let offset = address
        .checked_sub(chunk_start)
        .ok_or(QxError::InvalidEncoding)? as usize;
    chunk
        .get(offset..offset + 32)
        .ok_or(QxError::InvalidEncoding)?
        .try_into()
        .map_err(|_| QxError::InvalidEncoding.into())
}

fn checked_party_state<'a, 'b>(
    program_id: &Pubkey,
    dispute: &'a AccountInfo<'b>,
    party: &AccountInfo,
    phase: u8,
    expected_party: u8,
) -> Result<std::cell::RefMut<'a, &'b mut [u8]>, ProgramError> {
    if dispute.owner != program_id
        || !dispute.is_writable
        || !party.is_signer
        || dispute.data_len() != DISPUTE_V2_ACCOUNT_BYTES
    {
        return Err(QxError::Unauthorized.into());
    }
    let state = dispute.try_borrow_mut_data()?;
    require_dispute(&state, phase)?;
    require_deadline(&state)?;
    let party_offset = if expected_party == PARTY_MINER {
        D_MINER_OFFSET
    } else {
        D_CHALLENGER_OFFSET
    };
    if state[D_EXPECTED_PARTY_OFFSET] != expected_party
        || state[party_offset..party_offset + 32] != party.key.to_bytes()
    {
        return Err(QxError::Unauthorized.into());
    }
    Ok(state)
}

fn trace_domain(state: &[u8]) -> Result<&'static [u8], ProgramError> {
    match state[D_RESOLUTION_FLAGS_OFFSET] {
        TRACE_KIND_JOB => Ok(D_TRACE_TREE),
        TRACE_KIND_TOKEN => Ok(D_TOKEN_TREE),
        TRACE_KIND_OPERATOR | TRACE_KIND_FINAL_HEAD => Ok(D_OPERATOR_TREE),
        _ => Err(QxError::InvalidState.into()),
    }
}

fn set_terminal_phase(state: &mut [u8]) -> ProgramResult {
    let phase = match state[D_RESOLUTION_FLAGS_OFFSET] {
        TRACE_KIND_JOB => PHASE_CHALLENGER_TOKEN,
        TRACE_KIND_TOKEN => PHASE_CHALLENGER_CHECKPOINT,
        TRACE_KIND_OPERATOR | TRACE_KIND_FINAL_HEAD => PHASE_CHALLENGER_TERMINAL,
        _ => return Err(QxError::InvalidState.into()),
    };
    set_phase(state, phase, PARTY_CHALLENGER)
}

fn parse_segment_program(
    cursor: &mut Cursor<'_>,
    model_data: &[u8],
    expected_segment: u32,
) -> Result<([u8; 32], u32), ProgramError> {
    let descriptor = cursor.take(40)?;
    let segment_index = u32_from(descriptor, 0)?;
    let instruction_root: [u8; 32] = descriptor[4..36]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let instruction_count = u32_from(descriptor, 36)?;
    let schedule_count = u32_at(model_data, 136)?;
    let depth = usize::from(cursor.u8()?);
    let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    if segment_index != expected_segment || instruction_root == [0; 32] || instruction_count == 0 {
        return Err(QxError::InvalidState.into());
    }
    let descriptor_hash = stable_hash_domain(D_VM_SEGMENT_PROGRAM, &[descriptor]);
    verify_merkle_blake3(
        descriptor_hash,
        segment_index,
        schedule_count,
        depth,
        siblings,
        array_at(model_data, 104)?,
        D_SEGMENT_PROGRAM_TREE,
    )?;
    Ok((instruction_root, instruction_count))
}

fn parse_and_verify_bounds(
    cursor: &mut Cursor<'_>,
    trace_length: u32,
    root: [u8; 32],
    tree_domain: &[u8],
) -> Result<([u8; 32], [u8; 32]), ProgramError> {
    let start = cursor.array_32()?;
    let end = cursor.array_32()?;
    let depth = usize::from(cursor.u8()?);
    let start_siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    let end_siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    if !cursor.is_empty() || merkle_depth(trace_length)? != depth {
        return Err(QxError::InvalidEncoding.into());
    }
    verify_merkle_blake3(
        start,
        0,
        trace_length,
        depth,
        start_siblings,
        root,
        tree_domain,
    )?;
    verify_merkle_blake3(
        end,
        trace_length - 1,
        trace_length,
        depth,
        end_siblings,
        root,
        tree_domain,
    )?;
    Ok((start, end))
}

fn parse_and_verify_trace_opening(
    cursor: &mut Cursor<'_>,
    index: u32,
    trace_length: u32,
    root: [u8; 32],
    tree_domain: &[u8],
) -> Result<[u8; 32], ProgramError> {
    let claim = cursor.array_32()?;
    let depth = usize::from(cursor.u8()?);
    let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    if !cursor.is_empty() || merkle_depth(trace_length)? != depth {
        return Err(QxError::InvalidEncoding.into());
    }
    verify_merkle_blake3(
        claim,
        index,
        trace_length,
        depth,
        siblings,
        root,
        tree_domain,
    )?;
    Ok(claim)
}

fn verify_merkle_sha(
    domain: &[u8],
    mut node: [u8; 32],
    mut index: u32,
    leaf_count: u32,
    depth: usize,
    siblings: &[u8],
    root: [u8; 32],
) -> ProgramResult {
    if leaf_count == 0
        || index >= leaf_count
        || merkle_depth(leaf_count)? != depth
        || siblings.len() != depth * 32
    {
        return Err(QxError::InvalidEncoding.into());
    }
    for sibling in siblings.chunks_exact(32) {
        let sibling: &[u8; 32] = sibling.try_into().map_err(|_| QxError::InvalidEncoding)?;
        node = if index & 1 == 0 {
            merkle_parent(domain, &node, sibling)
        } else {
            merkle_parent(domain, sibling, &node)
        };
        index >>= 1;
    }
    if node != root {
        return Err(QxError::InvalidReceiptProof.into());
    }
    Ok(())
}

fn verify_merkle_blake3(
    mut node: [u8; 32],
    mut index: u32,
    leaf_count: u32,
    depth: usize,
    siblings: &[u8],
    root: [u8; 32],
    tree_domain: &[u8],
) -> ProgramResult {
    if leaf_count == 0
        || index >= leaf_count
        || siblings.len() != depth * 32
        || merkle_depth(leaf_count)? != depth
    {
        return Err(QxError::InvalidEncoding.into());
    }
    for sibling in siblings.chunks_exact(32) {
        let sibling: &[u8; 32] = sibling.try_into().map_err(|_| QxError::InvalidEncoding)?;
        node = if index & 1 == 0 {
            blake3_merkle_parent(tree_domain, &node, sibling)
        } else {
            blake3_merkle_parent(tree_domain, sibling, &node)
        };
        index >>= 1;
    }
    if node != root {
        return Err(QxError::InvalidReceiptProof.into());
    }
    Ok(())
}

fn blake3_merkle_parent(tree_domain: &[u8], left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    stable_hash_domain(D_MERKLE, &[tree_domain, left, right])
}

fn parse_activation(
    cursor: &mut Cursor<'_>,
    column: u32,
    expected_root: [u8; 32],
) -> Result<f32, ProgramError> {
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
    if node != expected_root {
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
    Ok(f32::from_bits(u32::from_le_bytes(
        chunk
            .get(local..local + 4)
            .ok_or(QxError::InvalidEncoding)?
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    )))
}

fn fma_state_hash(
    job_hash: &[u8; 32],
    tensor: u32,
    row: u32,
    column: u32,
    activation_root: &[u8; 32],
    accumulator_bits: u32,
) -> [u8; 32] {
    stable_hash_domain(
        D_FMA_STATE,
        &[
            job_hash,
            &tensor.to_le_bytes(),
            &row.to_le_bytes(),
            &column.to_le_bytes(),
            activation_root,
            &accumulator_bits.to_le_bytes(),
        ],
    )
}

fn scalar_state_hash(
    job_hash: &[u8; 32],
    transition: u32,
    op: u8,
    a: u32,
    b: u32,
    c: u32,
    result: Option<u32>,
) -> [u8; 32] {
    let phase = u8::from(result.is_some());
    let result = result.unwrap_or(0);
    stable_hash_domain(
        D_SCALAR_STATE,
        &[
            job_hash,
            &transition.to_le_bytes(),
            &[op],
            &[phase],
            &a.to_le_bytes(),
            &b.to_le_bytes(),
            &c.to_le_bytes(),
            &result.to_le_bytes(),
        ],
    )
}

fn scalar_op_supported(op: u8) -> bool {
    matches!(op, 1..=7 | 9 | 10 | 12 | 13 | 15 | 16)
}

fn evaluate_scalar(op: u8, a_bits: u32, b_bits: u32, c_bits: u32) -> Result<u32, ProgramError> {
    if !scalar_op_supported(op) {
        return Err(QxError::InvalidEncoding.into());
    }
    let a = f32::from_bits(a_bits);
    let b = f32::from_bits(b_bits);
    let c = f32::from_bits(c_bits);
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    let result = match op {
        1 | 4 => qx_consensus_math::fma_f32(a, b, c),
        2 | 5 => a + b,
        3 => a * b,
        6 => {
            if finite_ge(a_bits, b_bits) {
                a
            } else {
                b
            }
        }
        7 => {
            if finite_ge(a_bits, b_bits) {
                1.0
            } else {
                0.0
            }
        }
        9 => qx_consensus_math::rsqrt_f32(a).ok_or(QxError::InvalidEncoding)?,
        10 => qx_consensus_math::exp_f32(a).ok_or(QxError::InvalidEncoding)?,
        15 => qx_consensus_math::sin_cos_f32(a)
            .map(|value| value.0)
            .ok_or(QxError::InvalidEncoding)?,
        16 => qx_consensus_math::sin_cos_f32(a)
            .map(|value| value.1)
            .ok_or(QxError::InvalidEncoding)?,
        12 => qx_consensus_math::div_f32(a, b).ok_or(QxError::InvalidEncoding)?,
        13 => match a_bits {
            0x0000_0000 => c,
            0x3f80_0000 => b,
            _ => return Err(QxError::InvalidEncoding.into()),
        },
        _ => return Err(QxError::InvalidEncoding.into()),
    };
    if !result.is_finite() {
        return Err(QxError::NonFinite.into());
    }
    Ok(result.to_bits())
}

fn finite_ge(left: u32, right: u32) -> bool {
    finite_key(left) >= finite_key(right)
}

fn finite_key(bits: u32) -> [u8; 4] {
    let bits = if bits & 0x7fff_ffff == 0 { 0 } else { bits };
    if bits & 0x8000_0000 != 0 {
        (!bits).to_be_bytes()
    } else {
        (bits ^ 0x8000_0000).to_be_bytes()
    }
}

fn audit_seed(
    program_id: &Pubkey,
    epoch: u64,
    miner: &[u8; 32],
    root: &[u8; 32],
    count: u32,
    entropy_slot: u64,
    block_hash: &[u8; 32],
) -> [u8; 32] {
    stable_hash_domain(
        D_AUDIT_SEED,
        &[
            program_id.as_ref(),
            &epoch.to_le_bytes(),
            miner,
            root,
            &count.to_le_bytes(),
            &entropy_slot.to_le_bytes(),
            block_hash,
        ],
    )
}

fn audit_selected(
    seed: &[u8; 32],
    epoch: u64,
    leaf_index: u32,
    job_hash: &[u8; 32],
    probability_ppm: u32,
) -> Result<bool, ProgramError> {
    if probability_ppm > 1_000_000 {
        return Err(QxError::InvalidEncoding.into());
    }
    if probability_ppm == 0 {
        return Ok(false);
    }
    if probability_ppm == 1_000_000 {
        return Ok(true);
    }
    let digest = stable_hash_domain(
        D_AUDIT_DRAW,
        &[
            seed,
            &epoch.to_le_bytes(),
            &leaf_index.to_le_bytes(),
            job_hash,
        ],
    );
    let draw = u64::from_le_bytes(
        digest[..8]
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    );
    let threshold = ((u128::from(probability_ppm) << 64) / 1_000_000_u128) as u64;
    Ok(draw < threshold)
}

fn sha_domain(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut parts = Vec::with_capacity(1 + fields.len() * 2);
    parts.push(domain);
    let lengths = fields
        .iter()
        .map(|field| (field.len() as u64).to_le_bytes())
        .collect::<Vec<_>>();
    for (field, length) in fields.iter().zip(&lengths) {
        parts.push(length);
        parts.push(field);
    }
    solana_sha256_hasher::hashv(&parts).to_bytes()
}

fn stable_hash_domain(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut parts = Vec::with_capacity(1 + fields.len() * 2);
    parts.push(domain);
    let lengths = fields
        .iter()
        .map(|field| (field.len() as u64).to_le_bytes())
        .collect::<Vec<_>>();
    for (field, length) in fields.iter().zip(&lengths) {
        parts.push(length);
        parts.push(field);
    }
    crate::stable_hashv(&parts)
}

fn require_epoch(data: &[u8], phase: u8) -> ProgramResult {
    if data.len() != EPOCH_V2_ACCOUNT_BYTES
        || &data[..4] != EPOCH_MAGIC_V2
        || data[EPOCH_PHASE_OFFSET] != phase
    {
        return Err(QxError::InvalidState.into());
    }
    Ok(())
}

fn require_dispute(data: &[u8], phase: u8) -> ProgramResult {
    if data.len() != DISPUTE_V2_ACCOUNT_BYTES
        || &data[..4] != DISPUTE_MAGIC_V2
        || data[D_PHASE_OFFSET] != phase
        || data[D_WINNER_OFFSET] != PARTY_NONE
    {
        return Err(QxError::InvalidState.into());
    }
    Ok(())
}

fn require_deadline(data: &[u8]) -> ProgramResult {
    if Clock::get()?.slot > u64_at(data, D_DEADLINE_OFFSET)? {
        return Err(QxError::Deadline.into());
    }
    Ok(())
}

fn set_phase(data: &mut [u8], phase: u8, expected_party: u8) -> ProgramResult {
    data[D_PHASE_OFFSET] = phase;
    data[D_EXPECTED_PARTY_OFFSET] = expected_party;
    Ok(())
}

fn refresh_deadline(data: &mut [u8], current_slot: u64) -> ProgramResult {
    let deadline = current_slot
        .checked_add(u64_at(data, D_WINDOW_OFFSET)?)
        .ok_or(QxError::InvalidEncoding)?;
    put(data, D_DEADLINE_OFFSET, &deadline.to_le_bytes())
}

fn u32_from(bytes: &[u8], offset: usize) -> Result<u32, ProgramError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or(QxError::InvalidEncoding)?
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    ))
}

fn u64_from(bytes: &[u8], offset: usize) -> Result<u64, ProgramError> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or(QxError::InvalidEncoding)?
            .try_into()
            .map_err(|_| QxError::InvalidEncoding)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn onchain_sampling_matches_canonical_commitment_code() {
        let seed = [17; 32];
        let job_hash = [91; 32];
        for probability in [0, 1, 10_000, 500_000, 999_999, 1_000_000] {
            assert_eq!(
                audit_selected(&seed, 42, 7, &job_hash, probability).unwrap(),
                qx_commit::audit_selected_v1(&seed, 42, 7, &job_hash, probability).unwrap()
            );
        }
    }

    #[test]
    fn first_segment_reaches_checkpoint_reveal() {
        let mut state = [0_u8; DISPUTE_V2_ACCOUNT_BYTES];
        state[..4].copy_from_slice(DISPUTE_MAGIC_V2);
        state[D_RESOLUTION_FLAGS_OFFSET] = TRACE_KIND_TOKEN;
        state[D_HIGH_OFFSET..D_HIGH_OFFSET + 4].copy_from_slice(&1_u32.to_le_bytes());
        set_terminal_phase(&mut state).unwrap();
        assert_eq!(state[D_PHASE_OFFSET], PHASE_CHALLENGER_CHECKPOINT);
        assert_eq!(state[D_EXPECTED_PARTY_OFFSET], PARTY_CHALLENGER);
        assert_eq!(state[D_WINNER_OFFSET], PARTY_NONE);
    }
}
