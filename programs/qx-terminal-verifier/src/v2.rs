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

pub const JOB_PAYLOAD_BYTES: usize = 310;
pub const JOB_ACCOUNT_BYTES: usize = 314;
pub const RESULT_LEAF_BYTES: usize = 104;
pub const EPOCH_V2_ACCOUNT_BYTES: usize = 152;
pub const DISPUTE_V2_ACCOUNT_BYTES: usize = 448;
pub const MINER_V2_ACCOUNT_BYTES: usize = 16;
/// Roughly one day at a 400 ms slot. This is a consensus parameter, not a
/// value the submitting miner may shorten.
pub const CHALLENGE_WINDOW_SLOTS: u64 = 216_000;
/// Roughly eight minutes at a 400 ms slot for each interactive response.
pub const RESPONSE_WINDOW_SLOTS: u64 = 1_200;

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
const JOB_REQUESTER_OFFSET: usize = 66;
const JOB_MINER_OFFSET: usize = 98;
const JOB_MODEL_OFFSET: usize = 130;
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
const D_TRACE_TREE: &[u8] = b"QX:DISPUTE:TRACE:V1";
const D_FMA_STATE: &[u8] = b"QX:FMA:STATE:V1";

const PHASE_CHALLENGER_BOUNDS: u8 = 1;
const PHASE_MINER_BOUNDS: u8 = 2;
const PHASE_CHALLENGER_MIDPOINT: u8 = 3;
const PHASE_MINER_MIDPOINT: u8 = 4;
const PHASE_CHALLENGER_TERMINAL: u8 = 5;
const PHASE_MINER_TERMINAL: u8 = 6;
const PHASE_TERMINAL_READY: u8 = 7;
const PHASE_RESOLVED_V2: u8 = 8;
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
        || u32_from(payload, JOB_AUDIT_PPM_OFFSET)? > 1_000_000
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
        || job_payload[JOB_MODEL_REGISTRY_OFFSET..JOB_MODEL_REGISTRY_OFFSET + 32]
            != model.key.to_bytes()
        || u64_from(job_payload, JOB_EPOCH_OFFSET)? != epoch_number
        || u32_from(job_payload, JOB_INPUT_COUNT_OFFSET)? == 0
        || u32_from(job_payload, JOB_MAX_OUTPUT_OFFSET)? == 0
        || u64_from(job_payload, JOB_EXPOSURE_OFFSET)? == 0
        || u32_from(job_payload, JOB_AUDIT_PPM_OFFSET)? > 1_000_000
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
    let audit_probability_ppm = u32_from(job_payload, JOB_AUDIT_PPM_OFFSET)?;
    if result[RESULT_JOB_OFFSET..RESULT_JOB_OFFSET + 32] != job_hash
        || u32_from(result, RESULT_OUTPUT_COUNT_OFFSET)? == 0
        || u32_from(result, RESULT_OUTPUT_COUNT_OFFSET)? > max_output
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
    put(&mut state, D_EPOCH_OFFSET, epoch.key.as_ref())?;
    put(&mut state, D_MODEL_OFFSET, model.key.as_ref())?;
    put(&mut state, D_JOB_HASH_OFFSET, &job_hash)?;
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
    )?;
    if start != array_at(&state, D_COMMON_ROOT_OFFSET)?
        || end == array_at(&state, D_CHALLENGER_END_OFFSET)?
    {
        return Err(QxError::InvalidState.into());
    }
    put(&mut state, D_DEFENDER_END_OFFSET, &end)?;
    if trace_length == 2 {
        set_phase(&mut state, PHASE_CHALLENGER_TERMINAL, PARTY_CHALLENGER)?;
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
        set_phase(&mut state, PHASE_CHALLENGER_TERMINAL, PARTY_CHALLENGER)?;
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

fn parse_and_verify_bounds(
    cursor: &mut Cursor<'_>,
    trace_length: u32,
    root: [u8; 32],
) -> Result<([u8; 32], [u8; 32]), ProgramError> {
    let start = cursor.array_32()?;
    let end = cursor.array_32()?;
    let depth = usize::from(cursor.u8()?);
    let start_siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    let end_siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    if !cursor.is_empty() || merkle_depth(trace_length)? != depth {
        return Err(QxError::InvalidEncoding.into());
    }
    verify_merkle_blake3(start, 0, trace_length, depth, start_siblings, root)?;
    verify_merkle_blake3(
        end,
        trace_length - 1,
        trace_length,
        depth,
        end_siblings,
        root,
    )?;
    Ok((start, end))
}

fn parse_and_verify_trace_opening(
    cursor: &mut Cursor<'_>,
    index: u32,
    trace_length: u32,
    root: [u8; 32],
) -> Result<[u8; 32], ProgramError> {
    let claim = cursor.array_32()?;
    let depth = usize::from(cursor.u8()?);
    let siblings = cursor.take(depth.checked_mul(32).ok_or(QxError::InvalidEncoding)?)?;
    if !cursor.is_empty() || merkle_depth(trace_length)? != depth {
        return Err(QxError::InvalidEncoding.into());
    }
    verify_merkle_blake3(claim, index, trace_length, depth, siblings, root)?;
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
            blake3_merkle_parent(&node, sibling)
        } else {
            blake3_merkle_parent(sibling, &node)
        };
        index >>= 1;
    }
    if node != root {
        return Err(QxError::InvalidReceiptProof.into());
    }
    Ok(())
}

fn blake3_merkle_parent(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    blake3_domain(D_MERKLE, &[D_TRACE_TREE, left, right])
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
}
