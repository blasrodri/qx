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

pub const JOB_PAYLOAD_BYTES: usize = 310;
pub const JOB_ACCOUNT_BYTES: usize = 314;
pub const RESULT_LEAF_BYTES: usize = 104;
pub const EPOCH_V2_ACCOUNT_BYTES: usize = 152;
pub const DISPUTE_V2_ACCOUNT_BYTES: usize = 728;
pub const MINER_V2_ACCOUNT_BYTES: usize = 16;
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
const D_TOKEN_TREE: &[u8] = b"QX:TOKEN:V1";
const D_OPERATOR_TREE: &[u8] = b"QX:OPERATOR:V1";
const D_TOKEN_RESULT: &[u8] = b"QX:TOKEN:RESULT:V1";
const D_CHECKPOINT: &[u8] = b"QX:CHECKPOINT:V1";
const D_FMA_STATE: &[u8] = b"QX:FMA:STATE:V1";
const D_SCALAR_STATE: &[u8] = b"QX:SCALAR:STATE:V1";
const D_PROGRAM_TREE: &[u8] = b"QX:PROGRAM:TREE:V1";
const D_PROGRAM_INSTRUCTION: &[u8] = b"QX:PROGRAM:INSTRUCTION:V1";
const D_VM_STATE: &[u8] = b"QX:VM:STATE:V1";
const D_VM_INSTRUCTION: &[u8] = b"QX:VM:INSTRUCTION:V1";
const D_STATE_CHUNK: &[u8] = b"QX:STATE:CHUNK:V1";
const STATE_CHUNK_BYTES: usize = 256;

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
/// A valid disagreement reached a transition family that this version cannot
/// decide. No timeout or winner is possible from this fail-closed state.
const PHASE_UNSUPPORTED: u8 = 15;
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

const TRACE_KIND_JOB: u8 = 1;
const TRACE_KIND_TOKEN: u8 = 2;
const TRACE_KIND_OPERATOR: u8 = 3;

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
    if !cursor.is_empty()
        || u16::from_le_bytes(
            payload[..2]
                .try_into()
                .map_err(|_| QxError::InvalidEncoding)?,
        ) != 1
        || payload[JOB_INSTANCE_OFFSET..JOB_INSTANCE_OFFSET + 32] != program_id.to_bytes()
        || payload[JOB_REQUESTER_OFFSET..JOB_REQUESTER_OFFSET + 32] != requester.key.to_bytes()
        || payload[JOB_MODEL_OFFSET..JOB_MODEL_OFFSET + 32] != model.try_borrow_data()?[4..36]
        || payload[JOB_MODEL_REGISTRY_OFFSET..JOB_MODEL_REGISTRY_OFFSET + 32]
            != model.key.to_bytes()
        || u32_from(payload, JOB_INPUT_COUNT_OFFSET)? == 0
        || u32_from(payload, JOB_MAX_OUTPUT_OFFSET)? == 0
        || u64_from(payload, JOB_EXPOSURE_OFFSET)? == 0
        || u64_from(payload, JOB_DEADLINE_OFFSET)? <= Clock::get()?.slot
        || !(MIN_AUDIT_PROBABILITY_PPM..=1_000_000)
            .contains(&u32_from(payload, JOB_AUDIT_PPM_OFFSET)?)
    {
        return Err(QxError::InvalidEncoding.into());
    }
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
    if !cursor.is_empty()
        || state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes()
        || checkpoint_count != u32_at(&model.try_borrow_data()?, 140)?
        || token_index != expected_token
        || token[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
        || blake3_domain(D_TOKEN_RESULT, &[token]) != array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_CHALLENGER_TOKEN_ROOT_OFFSET, &token[36..68])?;
    put(
        &mut state,
        D_TOKEN_COUNT_OFFSET,
        &checkpoint_count.to_le_bytes(),
    )?;
    put(&mut state, D_TOKEN_INDEX_OFFSET, &token_index.to_le_bytes())?;
    put(&mut state, D_CHALLENGER_PREDICTION_OFFSET, &token[68..72])?;
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
        || blake3_domain(D_TOKEN_RESULT, &[token]) != array_at(&state, D_DEFENDER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_DEFENDER_PREDICTION_OFFSET, &token[68..72])?;
    let defender_root: [u8; 32] = token[36..68]
        .try_into()
        .map_err(|_| QxError::InvalidEncoding)?;
    let challenger_root = array_at(&state, D_CHALLENGER_TOKEN_ROOT_OFFSET)?;
    if defender_root == challenger_root {
        // The checkpoint execution agrees and only the final head/argmax is
        // disputed. Freeze until that registered transition is implemented;
        // allowing a timeout here would let one side win without proof.
        return set_phase(&mut state, PHASE_UNSUPPORTED, PARTY_NONE);
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
    if !cursor.is_empty()
        || state[D_MODEL_OFFSET..D_MODEL_OFFSET + 32] != model.key.to_bytes()
        || expected_layer >= model_layers
        || common_checkpoint[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
        || u32_from(common_checkpoint, 32)? != u32_at(&state, D_TOKEN_INDEX_OFFSET)?
        || u32::from(u16_from(common_checkpoint, 36)?) != expected_common_layer
        || blake3_domain(D_CHECKPOINT, &[common_checkpoint])
            != array_at(&state, D_COMMON_ROOT_OFFSET)?
        || checkpoint[..32] != state[D_JOB_ID_OFFSET..D_JOB_ID_OFFSET + 32]
        || u32_from(checkpoint, 32)? != u32_at(&state, D_TOKEN_INDEX_OFFSET)?
        || u32::from(u16_from(checkpoint, 36)?) != expected_layer
        || blake3_domain(D_CHECKPOINT, &[checkpoint]) != array_at(&state, D_CHALLENGER_END_OFFSET)?
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
        || blake3_domain(D_CHECKPOINT, &[checkpoint]) != array_at(&state, D_DEFENDER_END_OFFSET)?
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
    let instruction_count = u32_at(&model_data, 136)?;
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
        &array_at(&state, D_CHALLENGER_RESIDUAL_OFFSET)?,
        &array_at(&state, D_CHALLENGER_KV_OFFSET)?,
    );
    if start != expected_start || end != expected_end {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_CHALLENGER_ROOT_OFFSET, &trace_root)?;
    put(&mut state, D_COMMON_ROOT_OFFSET, &start)?;
    put(&mut state, D_CHALLENGER_END_OFFSET, &end)?;
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
    let instruction_count = u32_at(&model_data, 136)?;
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
    let instruction_hash = blake3_domain(
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
    if &model_data[..4] != MODEL_MAGIC || instruction_count != u32_at(&model_data, 136)? {
        return Err(QxError::InvalidState.into());
    }
    let instruction_bytes = vm_instruction_bytes(opcode, source_a, source_b, source_c, destination);
    let instruction_hash = blake3_domain(
        D_VM_INSTRUCTION,
        &[&program_counter.to_le_bytes(), &instruction_bytes],
    );
    verify_merkle_blake3(
        instruction_hash,
        program_counter,
        instruction_count,
        instruction_depth,
        instruction_siblings,
        array_at(&model_data, 104)?,
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
    let memory_leaf = blake3_domain(D_STATE_CHUNK, &[&chunk_index.to_le_bytes(), chunk]);
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
    let updated_leaf = blake3_domain(
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
    blake3_domain(
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
            solana_blake3_hasher::hashv(&[&node, sibling]).to_bytes()
        } else {
            solana_blake3_hasher::hashv(&[sibling, &node]).to_bytes()
        };
        index >>= 1;
    }
    node
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
        TRACE_KIND_OPERATOR => Ok(D_OPERATOR_TREE),
        _ => Err(QxError::InvalidState.into()),
    }
}

fn set_terminal_phase(state: &mut [u8]) -> ProgramResult {
    let phase = match state[D_RESOLUTION_FLAGS_OFFSET] {
        TRACE_KIND_JOB => PHASE_CHALLENGER_TOKEN,
        TRACE_KIND_TOKEN => PHASE_CHALLENGER_CHECKPOINT,
        TRACE_KIND_OPERATOR => PHASE_CHALLENGER_TERMINAL,
        _ => return Err(QxError::InvalidState.into()),
    };
    set_phase(state, phase, PARTY_CHALLENGER)
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
    blake3_domain(D_MERKLE, &[tree_domain, left, right])
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
    blake3_domain(
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
    blake3_domain(
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
    matches!(op, 1..=7 | 9 | 10 | 12 | 13)
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
    let left = if left & 0x7fff_ffff == 0 { 0 } else { left };
    let right = if right & 0x7fff_ffff == 0 { 0 } else { right };
    let key = |bits: u32| {
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits ^ 0x8000_0000
        }
    };
    key(left).to_be_bytes() >= key(right).to_be_bytes()
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
    blake3_domain(
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
    let digest = blake3_domain(
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

fn blake3_domain(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
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
    solana_blake3_hasher::hashv(&parts).to_bytes()
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
