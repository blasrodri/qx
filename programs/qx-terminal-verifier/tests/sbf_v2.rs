use ed25519_dalek::{Signer, SigningKey};
use half::f16;
use mollusk_svm::Mollusk;
use qx_commit::{
    activation_hash_v1, hash_domain, job_genesis_hash_v1, prove_state_chunk_v1, state_hash_v1,
    vm_program_tree_v1, vm_segment_schedule_v1, CheckpointV1, EpochBlockSeedV2, FmaStateV1,
    GraniteMacroStateV1, GraniteMacroWorkspaceV1, GraniteMicroLevelV1, GraniteMicroStateV1,
    GraniteWorkspaceSlotV1, HashAlgorithm, JobPayloadV1, MerkleProof, MerkleTree, ProtocolVersion,
    ResultLeafV1, StateTreeKindV1, TokenResultV1, VmInstructionV1, VmOutputStateV1,
    VmSegmentProgramV1, VmStateV1, WeightLeafV1, D_BATCH, D_GRANITE_MICRO_TREE, D_JOB, D_MERKLE,
    D_OPERATOR_TREE, D_PROGRAM_TREE, D_TOKEN, D_WEIGHT_TREE,
};
use solana_account::Account;
use solana_hash::Hash;
use solana_instruction::{AccountMeta, Instruction};
use solana_pubkey::Pubkey;

const EPOCH_BYTES: usize = 152;
const DISPUTE_BYTES: usize = 768;
const STAGED_BYTES: usize = 64;
const STAGED_KV_BYTES: usize = 144;
const STAGED_SCALAR_BYTES: usize = 104;
const STAGED_ARGMAX_BYTES: usize = 108;
const STAGED_GRANITE_BOUND_BYTES: usize = 152;
const MINER_BYTES: usize = 16;
const MODEL_BYTES: usize = 156;
const WEIGHT_LEAVES: u32 = 41_755_744;

#[test]
fn testnet_load_probe_executes_real_sbf_hash_work() {
    let f = Fixture::new();
    let vm = f.vm();
    let rounds = 32_u16;
    let payload = vec![0x5a; 256];
    let mut digest =
        hash_domain(HashAlgorithm::Blake3V1, b"QX:TESTNET:LOAD:V1", &[&payload]).unwrap();
    for round in 1..rounds {
        digest = hash_domain(
            HashAlgorithm::Blake3V1,
            b"QX:TESTNET:LOAD:V1",
            &[&round.to_le_bytes(), &digest],
        )
        .unwrap();
    }
    let mut body = Vec::new();
    body.extend_from_slice(&rounds.to_le_bytes());
    body.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    body.extend_from_slice(&payload);
    body.extend_from_slice(&digest);
    let valid = vm.process_instruction(&f.ix(48, vec![], &body), &[]);
    assert_ok(&valid.program_result);
    let mut corrupt = body;
    *corrupt.last_mut().unwrap() ^= 1;
    let invalid = vm.process_instruction(&f.ix(48, vec![], &corrupt), &[]);
    assert!(invalid.program_result.is_err());
    println!("v2_testnet_load_probe_cu={}", valid.compute_units_consumed);
}

#[test]
fn granite_output_and_trace_bounds_are_packet_safe_and_admitted_on_sbf() {
    let f = Fixture::new();
    let mut vm = f.vm();
    vm.sysvars.clock.slot = 100;
    let job_hash = [0x51; 32];
    let token_index = 7_u32;
    let segment_index = 3_u32;
    let macro_pc = 4_u32;
    let level = GraniteMicroLevelV1::DenseDot;
    let coordinates = [0_u32, 0, 0];
    let instruction = VmInstructionV1 {
        opcode: 66,
        source_a: 0,
        source_b: 0,
        source_c: 1_024,
        destination: 0,
    };
    let output = vec![1.25_f32; 1_024];
    let output_bytes = output
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let output_root = state_hash_v1(&output_bytes).unwrap();
    let output_opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &output_bytes, 0).unwrap();
    let miner_output = vec![2.25_f32; 1_024];
    let miner_output_bytes = miner_output
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let miner_output_root = state_hash_v1(&miner_output_bytes).unwrap();
    let miner_output_opening =
        prove_state_chunk_v1(StateTreeKindV1::Residual, &miner_output_bytes, 0).unwrap();

    let context = |step, registers| GraniteMicroStateV1 {
        job_hash,
        token_index,
        segment_index,
        macro_pc,
        level,
        coordinates,
        step,
        registers,
    };
    let start_registers = [0_u32; 12];
    let mut end_registers = [0_u32; 12];
    end_registers[8] = output[0].to_bits();
    let transition_count = 1_086_u32;
    let start = context(0, start_registers).hash().unwrap();
    let end = context(transition_count, end_registers).hash().unwrap();
    let mut leaves = vec![[0x73; 32]; transition_count as usize + 1];
    leaves[0] = start;
    leaves[transition_count as usize] = end;
    let trace = MerkleTree::new(HashAlgorithm::Blake3V1, D_GRANITE_MICRO_TREE, &leaves).unwrap();
    let mut miner_end_registers = [0_u32; 12];
    miner_end_registers[8] = miner_output[0].to_bits();
    let miner_end = context(transition_count, miner_end_registers)
        .hash()
        .unwrap();
    let mut miner_leaves = vec![[0x74; 32]; transition_count as usize + 1];
    miner_leaves[0] = start;
    miner_leaves[transition_count as usize] = miner_end;
    let miner_trace =
        MerkleTree::new(HashAlgorithm::Blake3V1, D_GRANITE_MICRO_TREE, &miner_leaves).unwrap();

    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 19;
    dispute.data[6] = 2;
    dispute.data[72..104].copy_from_slice(&job_hash);
    dispute.data[104..136].copy_from_slice(f.miner.as_ref());
    dispute.data[136..168].copy_from_slice(f.challenger.as_ref());
    dispute.data[264..296].copy_from_slice(&miner_output_root);
    dispute.data[296..328].copy_from_slice(&output_root);
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[392..409].copy_from_slice(&instruction.canonical_bytes());
    dispute.data[516..520].copy_from_slice(&token_index.to_le_bytes());
    dispute.data[520..524].copy_from_slice(&segment_index.to_le_bytes());
    dispute.data[524..528].copy_from_slice(&macro_pc.to_le_bytes());

    let mut output_body = vec![level as u8];
    for coordinate in coordinates {
        output_body.extend_from_slice(&coordinate.to_le_bytes());
    }
    output_body.extend_from_slice(&output_opening.byte_length.to_le_bytes());
    output_body.extend_from_slice(&output_opening.chunk_index.to_le_bytes());
    output_body.extend_from_slice(&(output_opening.chunk.len() as u16).to_le_bytes());
    output_body.extend_from_slice(&output_opening.chunk);
    output_body.push(output_opening.siblings.len() as u8);
    for sibling in &output_opening.siblings {
        output_body.extend_from_slice(sibling);
    }
    assert!(output_body.len() + 1 < 900);
    let output_stage = vm.process_instruction(
        &f.ix(
            58,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
                AccountMeta::new(f.staged, false),
            ],
            &output_body,
        ),
        &[
            (f.dispute, dispute.clone()),
            (f.challenger, Account::default()),
            (f.staged, owned(STAGED_SCALAR_BYTES, f.program)),
        ],
    );
    assert_ok(&output_stage.program_result);
    let staged_output = account(&output_stage.resulting_accounts, f.staged);

    let staged_bound_key = Pubkey::new_unique();
    let mut staged_bound = owned(STAGED_GRANITE_BOUND_BYTES, f.program);
    let mut bound_cu = 0;
    for (endpoint, registers, proof) in [
        (0_u8, start_registers, trace.proof(0).unwrap()),
        (
            1_u8,
            end_registers,
            trace.proof(transition_count as usize).unwrap(),
        ),
    ] {
        let mut body = vec![level as u8];
        for coordinate in coordinates {
            body.extend_from_slice(&coordinate.to_le_bytes());
        }
        body.extend_from_slice(&trace.root());
        body.push(endpoint);
        for register in registers {
            body.extend_from_slice(&register.to_le_bytes());
        }
        body.push(proof.siblings.len() as u8);
        append_siblings(&mut body, &proof);
        assert!(body.len() + 1 < 900);
        let staged = vm.process_instruction(
            &f.ix(
                59,
                vec![
                    AccountMeta::new(f.dispute, false),
                    AccountMeta::new_readonly(f.challenger, true),
                    AccountMeta::new(staged_bound_key, false),
                ],
                &body,
            ),
            &[
                (f.dispute, dispute.clone()),
                (f.challenger, Account::default()),
                (staged_bound_key, staged_bound),
            ],
        );
        assert_ok(&staged.program_result);
        bound_cu += staged.compute_units_consumed;
        staged_bound = account(&staged.resulting_accounts, staged_bound_key);
    }

    let mut submit_body = vec![level as u8];
    for coordinate in coordinates {
        submit_body.extend_from_slice(&coordinate.to_le_bytes());
    }
    let submit = vm.process_instruction(
        &f.ix(
            49,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
                AccountMeta::new_readonly(f.staged, false),
                AccountMeta::new_readonly(staged_bound_key, false),
            ],
            &submit_body,
        ),
        &[
            (f.dispute, dispute),
            (f.challenger, Account::default()),
            (f.staged, staged_output),
            (staged_bound_key, staged_bound),
        ],
    );
    assert_ok(&submit.program_result);
    let dispute = account(&submit.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[6]), (20, 1));
    assert_eq!(&dispute.data[200..232], &trace.root());
    assert_eq!(u32_at(&dispute.data, 444), output[0].to_bits());

    let mut miner_output_body = vec![level as u8];
    for coordinate in coordinates {
        miner_output_body.extend_from_slice(&coordinate.to_le_bytes());
    }
    miner_output_body.extend_from_slice(&miner_output_opening.byte_length.to_le_bytes());
    miner_output_body.extend_from_slice(&miner_output_opening.chunk_index.to_le_bytes());
    miner_output_body.extend_from_slice(&(miner_output_opening.chunk.len() as u16).to_le_bytes());
    miner_output_body.extend_from_slice(&miner_output_opening.chunk);
    miner_output_body.push(miner_output_opening.siblings.len() as u8);
    for sibling in &miner_output_opening.siblings {
        miner_output_body.extend_from_slice(sibling);
    }
    let miner_output_stage = vm.process_instruction(
        &f.ix(
            58,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
                AccountMeta::new(f.staged, false),
            ],
            &miner_output_body,
        ),
        &[
            (f.dispute, dispute.clone()),
            (f.miner, Account::default()),
            (f.staged, account(&submit.resulting_accounts, f.staged)),
        ],
    );
    assert_ok(&miner_output_stage.program_result);
    let staged_output = account(&miner_output_stage.resulting_accounts, f.staged);
    let mut staged_bound = account(&submit.resulting_accounts, staged_bound_key);
    let mut miner_bound_cu = 0;
    for (endpoint, registers, proof) in [
        (0_u8, start_registers, miner_trace.proof(0).unwrap()),
        (
            1_u8,
            miner_end_registers,
            miner_trace.proof(transition_count as usize).unwrap(),
        ),
    ] {
        let mut body = vec![level as u8];
        for coordinate in coordinates {
            body.extend_from_slice(&coordinate.to_le_bytes());
        }
        body.extend_from_slice(&miner_trace.root());
        body.push(endpoint);
        for register in registers {
            body.extend_from_slice(&register.to_le_bytes());
        }
        body.push(proof.siblings.len() as u8);
        append_siblings(&mut body, &proof);
        let staged = vm.process_instruction(
            &f.ix(
                59,
                vec![
                    AccountMeta::new(f.dispute, false),
                    AccountMeta::new_readonly(f.miner, true),
                    AccountMeta::new(staged_bound_key, false),
                ],
                &body,
            ),
            &[
                (f.dispute, dispute.clone()),
                (f.miner, Account::default()),
                (staged_bound_key, staged_bound),
            ],
        );
        assert_ok(&staged.program_result);
        miner_bound_cu += staged.compute_units_consumed;
        staged_bound = account(&staged.resulting_accounts, staged_bound_key);
    }
    let miner_submit = vm.process_instruction(
        &f.ix(
            50,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
                AccountMeta::new_readonly(f.staged, false),
                AccountMeta::new_readonly(staged_bound_key, false),
            ],
            &[],
        ),
        &[
            (f.dispute, dispute),
            (f.miner, Account::default()),
            (f.staged, staged_output),
            (staged_bound_key, staged_bound),
        ],
    );
    assert_ok(&miner_submit.program_result);
    let dispute = account(&miner_submit.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[6]), (3, 2));
    assert_eq!(&dispute.data[168..200], &miner_trace.root());
    assert_eq!(u32_at(&dispute.data, 440), miner_output[0].to_bits());
    println!(
        "v2_granite_challenger_staged_micro_admission_cu={}",
        output_stage.compute_units_consumed + bound_cu + submit.compute_units_consumed
    );
    println!(
        "v2_granite_miner_staged_micro_admission_cu={}",
        miner_output_stage.compute_units_consumed
            + miner_bound_cu
            + miner_submit.compute_units_consumed
    );
}

#[test]
fn granite_macro_boundary_authenticates_the_single_workspace_update_on_sbf() {
    let f = Fixture::new();
    let vm = f.vm();
    let scale = 0.22_f32;
    let instruction = VmInstructionV1 {
        opcode: 70,
        source_a: 0,
        source_b: scale.to_bits(),
        source_c: 1,
        destination: 1024,
    };
    let reserved = VmInstructionV1 {
        opcode: 64,
        source_a: 0,
        source_b: 0,
        source_c: 0,
        destination: 0,
    };
    let mut instructions = vec![reserved; 13];
    instructions.push(instruction);
    let program = vm_program_tree_v1(&instructions).unwrap();
    let proof = program.proof(13).unwrap();
    let left = vec![1.0_f32; 1024];
    let right = vec![2.0_f32; 1024];
    let expected = qx_consensus_math::fma_f32(scale, right[17], left[17]);
    let honest_output = vec![expected; 1024];
    let mut miner_output = honest_output.clone();
    miner_output[17] = 9.0;
    let bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>()
    };
    let left_root = activation_hash_v1(&bytes(&left)).unwrap();
    let right_root = activation_hash_v1(&bytes(&right)).unwrap();
    let honest_residual = state_hash_v1(&bytes(&honest_output)).unwrap();
    let miner_residual = state_hash_v1(&bytes(&miner_output)).unwrap();
    let pre_residual = [0x11; 32];
    let kv_root = [0x44; 32];
    let mut workspace = GraniteMacroWorkspaceV1::new(pre_residual);
    workspace.set(GraniteWorkspaceSlotV1::PostAttention, left_root);
    workspace.set(GraniteWorkspaceSlotV1::Moe, right_root);
    let pre = GraniteMacroStateV1 {
        job_hash: [0x55; 32],
        token_index: 7,
        segment_index: 3,
        program_counter: 13,
        workspace_root: workspace.root().unwrap(),
        global_kv_root: kv_root,
    };
    let honest_post = VmStateV1 {
        job_hash: pre.job_hash,
        token_index: pre.token_index,
        segment_index: pre.segment_index,
        program_counter: 14,
        residual_root: honest_residual,
        global_kv_root: kv_root,
    };
    let miner_post = VmStateV1 {
        residual_root: miner_residual,
        ..honest_post
    };
    let workspace_proof = workspace.proof(GraniteWorkspaceSlotV1::Residual).unwrap();

    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[7] = 3;
    dispute.data[72..104].copy_from_slice(&pre.job_hash);
    dispute.data[104..136].copy_from_slice(f.miner.as_ref());
    dispute.data[136..168].copy_from_slice(f.challenger.as_ref());
    dispute.data[232..264].copy_from_slice(&pre.hash().unwrap());
    dispute.data[264..296].copy_from_slice(&miner_post.hash().unwrap());
    dispute.data[296..328].copy_from_slice(&honest_post.hash().unwrap());
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[376..380].copy_from_slice(&13_u32.to_le_bytes());
    dispute.data[380..384].copy_from_slice(&14_u32.to_le_bytes());
    dispute.data[516..520].copy_from_slice(&pre.token_index.to_le_bytes());
    bind_segment_program(&mut dispute, &program);

    let macro_body = |new_root: [u8; 32]| {
        let mut body = Vec::new();
        body.extend_from_slice(&pre.segment_index.to_le_bytes());
        body.extend_from_slice(&instruction.canonical_bytes());
        body.extend_from_slice(&14_u32.to_le_bytes());
        body.push(proof.siblings.len() as u8);
        append_siblings(&mut body, &proof);
        body.extend_from_slice(&pre_residual);
        body.extend_from_slice(&new_root);
        append_siblings(&mut body, &workspace_proof);
        body.extend_from_slice(&kv_root);
        body.extend_from_slice(&kv_root);
        body
    };
    let miner_macro_body = |new_root: [u8; 32]| {
        let mut body = Vec::new();
        body.extend_from_slice(&pre.segment_index.to_le_bytes());
        body.extend_from_slice(&pre_residual);
        body.extend_from_slice(&new_root);
        append_siblings(&mut body, &workspace_proof);
        body.extend_from_slice(&kv_root);
        body.extend_from_slice(&kv_root);
        body
    };
    let challenger = vm.process_instruction(
        &f.ix(
            44,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &macro_body(honest_residual),
        ),
        &[
            (f.dispute, dispute.clone()),
            (f.challenger, Account::default()),
        ],
    );
    assert_ok(&challenger.program_result);
    let dispute = account(&challenger.resulting_accounts, f.dispute);
    assert_eq!(dispute.data[4], 18);

    let mut forged = miner_macro_body(miner_residual);
    forged[4 + 32 + 32] ^= 1;
    let rejected = vm.process_instruction(
        &f.ix(
            45,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
            ],
            &forged,
        ),
        &[(f.dispute, dispute.clone()), (f.miner, Account::default())],
    );
    assert!(rejected.program_result.is_err());

    let miner_body = miner_macro_body(miner_residual);
    let miner = vm.process_instruction(
        &f.ix(
            45,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
            ],
            &miner_body,
        ),
        &[(f.dispute, dispute), (f.miner, Account::default())],
    );
    assert_ok(&miner.program_result);
    let dispute = account(&miner.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[6]), (19, 2));
    assert_eq!(&dispute.data[264..296], &miner_residual);
    assert_eq!(&dispute.data[296..328], &honest_residual);
    let openings = [
        (
            0_u8,
            Some((
                GraniteWorkspaceSlotV1::PostAttention,
                left_root,
                workspace
                    .proof(GraniteWorkspaceSlotV1::PostAttention)
                    .unwrap(),
            )),
            prove_state_chunk_v1(StateTreeKindV1::Activation, &bytes(&left), 0).unwrap(),
        ),
        (
            1_u8,
            Some((
                GraniteWorkspaceSlotV1::Moe,
                right_root,
                workspace.proof(GraniteWorkspaceSlotV1::Moe).unwrap(),
            )),
            prove_state_chunk_v1(StateTreeKindV1::Activation, &bytes(&right), 0).unwrap(),
        ),
        (
            2_u8,
            None,
            prove_state_chunk_v1(StateTreeKindV1::Residual, &bytes(&honest_output), 0).unwrap(),
        ),
        (
            3_u8,
            None,
            prove_state_chunk_v1(StateTreeKindV1::Residual, &bytes(&miner_output), 0).unwrap(),
        ),
    ];
    let mut staged = owned(STAGED_ARGMAX_BYTES, f.program);
    let mut stage_cu = 0;
    for (role, workspace_opening, opening) in openings {
        let mut body = vec![role];
        body.extend_from_slice(&17_u32.to_le_bytes());
        if let Some((_slot, root, proof)) = workspace_opening {
            body.extend_from_slice(&root);
            append_siblings(&mut body, &proof);
        }
        body.extend_from_slice(&opening.byte_length.to_le_bytes());
        body.extend_from_slice(&opening.chunk_index.to_le_bytes());
        body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
        body.extend_from_slice(&opening.chunk);
        body.push(opening.siblings.len() as u8);
        for sibling in &opening.siblings {
            body.extend_from_slice(sibling);
        }
        let result = vm.process_instruction(
            &f.ix(
                46,
                vec![
                    AccountMeta::new_readonly(f.dispute, false),
                    AccountMeta::new(f.staged, false),
                ],
                &body,
            ),
            &[(f.dispute, dispute.clone()), (f.staged, staged)],
        );
        assert_ok(&result.program_result);
        stage_cu += result.compute_units_consumed;
        staged = account(&result.resulting_accounts, f.staged);
    }
    let resolved = vm.process_instruction(
        &f.ix(
            47,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.staged, false),
            ],
            &[],
        ),
        &[(f.dispute, dispute), (f.staged, staged)],
    );
    assert_ok(&resolved.program_result);
    let resolved_dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((resolved_dispute.data[4], resolved_dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&resolved_dispute.data, 436), expected.to_bits());
    println!(
        "v2_granite_macro_boundary_cu={}",
        challenger.compute_units_consumed + miner.compute_units_consumed
    );
    println!(
        "v2_granite_residual_terminal_cu={}",
        stage_cu + resolved.compute_units_consumed
    );
}

#[test]
fn block_seeded_unsigned_result_is_bisected_and_convicted_on_sbf() {
    let f = Fixture::new();
    let mut vm = f.vm();
    let (model, epoch, finalize_cu) = f.seeded_epoch(&mut vm);
    let expected_seed = EpochBlockSeedV2 {
        protocol_instance: f.program.to_bytes(),
        epoch: 7,
        miner: f.miner.to_bytes(),
        result_root: f.results.root(),
        result_count: 2,
        entropy_slot: 101,
    }
    .derive(&f.entropy)
    .unwrap();
    assert_eq!(&epoch.data[104..136], &expected_seed);

    let (mut dispute, open_cu) = f.open(&vm, epoch, model.clone());
    // This compatibility fixture starts directly with an operator/FMA trace.
    // Production roots use the token/checkpoint descent tested separately.
    dispute.data[7] = 3;
    let (next, challenger_bounds_cu) = f.bounds(&vm, dispute, true);
    dispute = next;
    let (next, miner_bounds_cu) = f.bounds(&vm, dispute, false);
    dispute = next;

    // The sole midpoint matches, leaving the adjacent first bad transition.
    let (next, challenger_midpoint_cu) = f.midpoint(&vm, dispute, true);
    dispute = next;
    let (next, miner_midpoint_cu) = f.midpoint(&vm, dispute, false);
    dispute = next;
    assert_eq!(
        (u32_at(&dispute.data, 376), u32_at(&dispute.data, 380)),
        (1, 2)
    );
    assert_eq!(dispute.data[4], 5);

    let challenger_terminal = vm.process_instruction(
        &f.challenger_terminal_ix(),
        &[(f.dispute, dispute), (f.challenger, Account::default())],
    );
    assert_ok(&challenger_terminal.program_result);
    let dispute = account(&challenger_terminal.resulting_accounts, f.dispute);
    let miner_terminal = vm.process_instruction(
        &f.ix(
            17,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
            ],
            &f.miner_states[2].accumulator_bits.to_le_bytes(),
        ),
        &[(f.dispute, dispute), (f.miner, Account::default())],
    );
    assert_ok(&miner_terminal.program_result);
    let dispute = account(&miner_terminal.resulting_accounts, f.dispute);
    let stage = vm.process_instruction(
        &f.stage_ix(),
        &[
            (f.model, model),
            (f.dispute, dispute.clone()),
            (f.staged, owned(STAGED_BYTES, f.program)),
        ],
    );
    assert_ok(&stage.program_result);
    let staged = account(&stage.resulting_accounts, f.staged);
    assert_eq!(f32::from_bits(u32_at(&staged.data, 48)), 0.5);
    assert_eq!(f32::from_bits(u32_at(&dispute.data, 436)), 1.0);
    assert_eq!(f32::from_bits(u32_at(&dispute.data, 440)), 3.0);
    assert_eq!(f32::from_bits(u32_at(&dispute.data, 444)), 2.0);
    assert_eq!(
        f32::from_le_bytes(f.activation.chunk[8..12].try_into().unwrap()),
        2.0
    );
    let resolved =
        vm.process_instruction(&f.resolve_ix(), &[(f.dispute, dispute), (f.staged, staged)]);
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!(f32::from_bits(u32_at(&dispute.data, 436)), 2.0);
    assert_eq!(f32::from_bits(u32_at(&dispute.data, 440)), 3.0);
    assert_eq!(f32::from_bits(u32_at(&dispute.data, 444)), 2.0);
    assert_eq!(u32_at(&dispute.data, 436), u32_at(&dispute.data, 444));
    assert_eq!(dispute.data[7], 3);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));

    println!("v2_finalize_entropy_cu={finalize_cu}");
    println!("v2_open_challenge_cu={open_cu}");
    println!(
        "v2_bisection_round_cu={}",
        challenger_bounds_cu + miner_bounds_cu + challenger_midpoint_cu + miner_midpoint_cu
    );
    println!(
        "v2_terminal_cu={}",
        stage.compute_units_consumed + resolved.compute_units_consumed
    );
    println!(
        "v2_dispute_total_cu={}",
        open_cu
            + challenger_bounds_cu
            + miner_bounds_cu
            + challenger_midpoint_cu
            + miner_midpoint_cu
            + challenger_terminal.compute_units_consumed
            + miner_terminal.compute_units_consumed
            + stage.compute_units_consumed
            + resolved.compute_units_consumed
    );
}

#[test]
fn fixed_exp_scalar_transition_convicts_wrong_miner_on_sbf() {
    let f = Fixture::new();
    let mut vm = f.vm();
    vm.sysvars.clock.slot = 100;
    let job_hash = [0x42; 32];
    let transition = 17_u32;
    let op = 10_u8;
    let a = 2.0_f32.to_bits();
    let b = 0.0_f32.to_bits();
    let c = 0.0_f32.to_bits();
    let honest = qx_consensus_math::exp_f32(2.0).unwrap().to_bits();
    let dishonest = honest ^ 1;
    let common = scalar_state_hash(job_hash, transition, op, a, b, c, None);
    let challenger_end = scalar_state_hash(job_hash, transition, op, a, b, c, Some(honest));
    let defender_end = scalar_state_hash(job_hash, transition, op, a, b, c, Some(dishonest));
    let program_leaves = (0..=transition)
        .map(|index| {
            let instruction_op = if index == transition { op } else { 2 };
            hash_domain(
                HashAlgorithm::Blake3V1,
                b"QX:PROGRAM:INSTRUCTION:V1",
                &[&index.to_le_bytes(), &[instruction_op]],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let program_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        b"QX:PROGRAM:TREE:V1",
        &program_leaves,
    )
    .unwrap();
    let program_proof = program_tree.proof(transition as usize).unwrap();
    let mut model = owned(MODEL_BYTES, f.program);
    model.data[..4].copy_from_slice(b"QXM1");
    model.data[104..136].copy_from_slice(&program_tree.root());
    model.data[136..140].copy_from_slice(&(program_leaves.len() as u32).to_le_bytes());

    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[72..104].copy_from_slice(&job_hash);
    dispute.data[40..72].copy_from_slice(f.model.as_ref());
    dispute.data[104..136].copy_from_slice(f.miner.as_ref());
    dispute.data[136..168].copy_from_slice(f.challenger.as_ref());
    dispute.data[232..264].copy_from_slice(&common);
    dispute.data[264..296].copy_from_slice(&defender_end);
    dispute.data[296..328].copy_from_slice(&challenger_end);
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[376..380].copy_from_slice(&transition.to_le_bytes());
    dispute.data[380..384].copy_from_slice(&(transition + 1).to_le_bytes());

    let mut challenger_body = vec![op];
    challenger_body.extend_from_slice(&a.to_le_bytes());
    challenger_body.extend_from_slice(&b.to_le_bytes());
    challenger_body.extend_from_slice(&c.to_le_bytes());
    challenger_body.extend_from_slice(&honest.to_le_bytes());
    challenger_body.extend_from_slice(&transition.to_le_bytes());
    challenger_body.extend_from_slice(&(program_leaves.len() as u32).to_le_bytes());
    challenger_body.push(program_proof.siblings.len() as u8);
    for sibling in &program_proof.siblings {
        challenger_body.extend_from_slice(sibling);
    }
    let challenger = vm.process_instruction(
        &f.ix(
            21,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &challenger_body,
        ),
        &[
            (f.model, model),
            (f.dispute, dispute),
            (f.challenger, Account::default()),
        ],
    );
    assert_ok(&challenger.program_result);
    let dispute = account(&challenger.resulting_accounts, f.dispute);

    let miner = vm.process_instruction(
        &f.ix(
            22,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
            ],
            &dishonest.to_le_bytes(),
        ),
        &[(f.dispute, dispute), (f.miner, Account::default())],
    );
    assert_ok(&miner.program_result);
    let dispute = account(&miner.resulting_accounts, f.dispute);

    let resolved = vm.process_instruction(
        &f.ix(23, vec![AccountMeta::new(f.dispute, false)], &[]),
        &[(f.dispute, dispute)],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&dispute.data, 436), honest);
    println!(
        "v2_fixed_exp_terminal_cu={}",
        challenger.compute_units_consumed
            + miner.compute_units_consumed
            + resolved.compute_units_consumed
    );
}

#[test]
fn fixed_sine_terminal_matches_the_registered_rope_math_on_sbf() {
    let f = Fixture::new();
    let mut vm = f.vm();
    vm.sysvars.clock.slot = 100;
    let angle = 1234.5_f32;
    let honest = qx_consensus_math::sin_cos_f32(angle).unwrap().0.to_bits();
    let dishonest = honest ^ 1;
    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 7;
    dispute.data[7] = 15;
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[424..428].copy_from_slice(&angle.to_bits().to_le_bytes());
    dispute.data[428..432].copy_from_slice(&0.0_f32.to_bits().to_le_bytes());
    dispute.data[432..436].copy_from_slice(&0.0_f32.to_bits().to_le_bytes());
    dispute.data[440..444].copy_from_slice(&dishonest.to_le_bytes());
    dispute.data[444..448].copy_from_slice(&honest.to_le_bytes());

    let resolved = vm.process_instruction(
        &f.ix(23, vec![AccountMeta::new(f.dispute, false)], &[]),
        &[(f.dispute, dispute)],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&dispute.data, 436), honest);
    println!(
        "v2_fixed_sine_terminal_cu={}",
        resolved.compute_units_consumed
    );
}

#[test]
fn chained_vm_scalar_transition_convicts_wrong_miner_on_sbf() {
    let f = Fixture::new();
    let mut vm = f.vm();
    vm.sysvars.clock.slot = 100;
    let mut memory = vec![0_u8; 512];
    for (index, value) in [2.0_f32, 3.0, 0.0, -1.0].iter().enumerate() {
        memory[index * 4..index * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
    }
    let memory_root = state_hash_v1(&memory).unwrap();
    let opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
    let instruction = VmInstructionV1 {
        opcode: 2,
        source_a: 0,
        source_b: 4,
        source_c: 8,
        destination: 12,
    };
    let program = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_PROGRAM_TREE,
        &[instruction.hash(0).unwrap()],
    )
    .unwrap();
    let program_proof = program.proof(0).unwrap();
    let honest_memory = opening
        .root_after_scalar_write(12, 5.0_f32.to_bits())
        .unwrap();
    let dishonest_memory = opening
        .root_after_scalar_write(12, 6.0_f32.to_bits())
        .unwrap();
    let job_hash = [0x51; 32];
    let pre = VmStateV1 {
        job_hash,
        token_index: 1,
        segment_index: 7,
        program_counter: 0,
        residual_root: memory_root,
        global_kv_root: [0x61; 32],
    };
    let honest_post = pre.successor(honest_memory).unwrap();
    let dishonest_post = pre.successor(dishonest_memory).unwrap();

    let mut model = owned(MODEL_BYTES, f.program);
    model.data[..4].copy_from_slice(b"QXM1");
    model.data[104..136].copy_from_slice(&program.root());
    model.data[136..140].copy_from_slice(&1_u32.to_le_bytes());
    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[40..72].copy_from_slice(f.model.as_ref());
    dispute.data[72..104].copy_from_slice(&job_hash);
    dispute.data[232..264].copy_from_slice(&pre.hash().unwrap());
    dispute.data[264..296].copy_from_slice(&dishonest_post.hash().unwrap());
    dispute.data[296..328].copy_from_slice(&honest_post.hash().unwrap());
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[376..380].copy_from_slice(&0_u32.to_le_bytes());
    dispute.data[380..384].copy_from_slice(&1_u32.to_le_bytes());
    dispute.data[516..520].copy_from_slice(&pre.token_index.to_le_bytes());
    bind_segment_program(&mut dispute, &program);

    let mut body = Vec::new();
    body.extend_from_slice(&pre.segment_index.to_le_bytes());
    body.extend_from_slice(&pre.program_counter.to_le_bytes());
    body.extend_from_slice(&pre.residual_root);
    body.extend_from_slice(&pre.global_kv_root);
    body.extend_from_slice(&instruction.canonical_bytes());
    body.extend_from_slice(&1_u32.to_le_bytes());
    body.push(program_proof.siblings.len() as u8);
    append_siblings(&mut body, &program_proof);
    body.extend_from_slice(&opening.byte_length.to_le_bytes());
    body.extend_from_slice(&opening.chunk_index.to_le_bytes());
    body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
    body.extend_from_slice(&opening.chunk);
    body.push(opening.siblings.len() as u8);
    for sibling in &opening.siblings {
        body.extend_from_slice(sibling);
    }

    let mut forged_body = body.clone();
    let first_memory_value_offset = 4 + 4 + 32 + 32 + 17 + 4 + 1 + 4 + 4 + 2;
    forged_body[first_memory_value_offset] ^= 1;
    let forged = vm.process_instruction(
        &f.ix(
            24,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
            ],
            &forged_body,
        ),
        &[(f.model, model.clone()), (f.dispute, dispute.clone())],
    );
    assert!(forged.program_result.is_err());

    let resolved = vm.process_instruction(
        &f.ix(
            24,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
            ],
            &body,
        ),
        &[(f.model, model), (f.dispute, dispute)],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&dispute.data, 436), 5.0_f32.to_bits());
    println!(
        "v2_chained_vm_scalar_cu={}",
        resolved.compute_units_consumed
    );
}

#[test]
fn staged_vm_scalar_reads_cross_chunk_operands_on_sbf() {
    let f = Fixture::new();
    let vm = f.vm();
    let mut memory = vec![0_u8; 512];
    memory[..4].copy_from_slice(&2.0_f32.to_le_bytes());
    memory[260..264].copy_from_slice(&3.0_f32.to_le_bytes());
    memory[264..268].copy_from_slice(&4.0_f32.to_le_bytes());
    let residual_root = state_hash_v1(&memory).unwrap();
    let opening_0 = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
    let opening_1 = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 1).unwrap();
    let instruction = VmInstructionV1 {
        opcode: 1,
        source_a: 0,
        source_b: 260,
        source_c: 264,
        destination: 268,
    };
    let program = vm_program_tree_v1(&[instruction]).unwrap();
    let pre = VmStateV1 {
        job_hash: [0x51; 32],
        token_index: 7,
        segment_index: 2,
        program_counter: 0,
        residual_root,
        global_kv_root: [0x52; 32],
    };
    let expected = qx_consensus_math::fma_f32(2.0, 3.0, 4.0);
    let mut honest_memory = memory.clone();
    honest_memory[268..272].copy_from_slice(&expected.to_le_bytes());
    let honest_post = pre
        .successor(state_hash_v1(&honest_memory).unwrap())
        .unwrap();
    let mut miner_memory = memory.clone();
    miner_memory[268..272].copy_from_slice(&11.0_f32.to_le_bytes());
    let miner_post = pre
        .successor(state_hash_v1(&miner_memory).unwrap())
        .unwrap();
    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[7] = 3;
    dispute.data[72..104].copy_from_slice(&pre.job_hash);
    dispute.data[232..264].copy_from_slice(&pre.hash().unwrap());
    dispute.data[264..296].copy_from_slice(&miner_post.hash().unwrap());
    dispute.data[296..328].copy_from_slice(&honest_post.hash().unwrap());
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[376..380].copy_from_slice(&0_u32.to_le_bytes());
    dispute.data[516..520].copy_from_slice(&pre.token_index.to_le_bytes());
    bind_segment_program(&mut dispute, &program);
    let proof = program.proof(0).unwrap();
    let stage_body = |mask: u8, opening: &qx_commit::StateChunkProofV1| {
        let mut body = vec![mask];
        body.extend_from_slice(&pre.segment_index.to_le_bytes());
        body.extend_from_slice(&pre.program_counter.to_le_bytes());
        body.extend_from_slice(&pre.residual_root);
        body.extend_from_slice(&pre.global_kv_root);
        body.extend_from_slice(&instruction.canonical_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.push(proof.siblings.len() as u8);
        append_siblings(&mut body, &proof);
        body.extend_from_slice(&opening.byte_length.to_le_bytes());
        body.extend_from_slice(&opening.chunk_index.to_le_bytes());
        body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
        body.extend_from_slice(&opening.chunk);
        body.push(opening.siblings.len() as u8);
        for sibling in &opening.siblings {
            body.extend_from_slice(sibling);
        }
        body
    };
    let staged_a = vm.process_instruction(
        &f.ix(
            38,
            vec![
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(f.staged, false),
            ],
            &stage_body(1, &opening_0),
        ),
        &[
            (f.dispute, dispute.clone()),
            (f.staged, owned(STAGED_SCALAR_BYTES, f.program)),
        ],
    );
    assert_ok(&staged_a.program_result);
    let staged = account(&staged_a.resulting_accounts, f.staged);
    let staged_bc = vm.process_instruction(
        &f.ix(
            38,
            vec![
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(f.staged, false),
            ],
            &stage_body(6, &opening_1),
        ),
        &[(f.dispute, dispute.clone()), (f.staged, staged)],
    );
    assert_ok(&staged_bc.program_result);
    let staged = account(&staged_bc.resulting_accounts, f.staged);
    let mut resolve_body = Vec::new();
    resolve_body.extend_from_slice(&pre.segment_index.to_le_bytes());
    resolve_body.extend_from_slice(&pre.residual_root);
    resolve_body.extend_from_slice(&pre.global_kv_root);
    resolve_body.extend_from_slice(&opening_1.byte_length.to_le_bytes());
    resolve_body.extend_from_slice(&opening_1.chunk_index.to_le_bytes());
    resolve_body.extend_from_slice(&(opening_1.chunk.len() as u16).to_le_bytes());
    resolve_body.extend_from_slice(&opening_1.chunk);
    resolve_body.push(opening_1.siblings.len() as u8);
    for sibling in &opening_1.siblings {
        resolve_body.extend_from_slice(sibling);
    }
    let resolved = vm.process_instruction(
        &f.ix(
            39,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.staged, false),
            ],
            &resolve_body,
        ),
        &[(f.dispute, dispute), (f.staged, staged)],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&dispute.data, 436), expected.to_bits());
    println!(
        "v2_cross_chunk_scalar_cu={}",
        staged_a.compute_units_consumed
            + staged_bc.compute_units_consumed
            + resolved.compute_units_consumed
    );
}

#[test]
fn staged_vm_q4_reads_cross_chunk_operands_on_sbf() {
    let f = Fixture::new();
    let vm = f.vm();
    let staged_operands = Pubkey::new_unique();
    let mut memory = vec![0_u8; 512];
    memory[..4].copy_from_slice(&2.0_f32.to_le_bytes());
    memory[260..264].copy_from_slice(&3.0_f32.to_le_bytes());
    let residual_root = state_hash_v1(&memory).unwrap();
    let opening_0 = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
    let opening_1 = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 1).unwrap();
    let instruction = VmInstructionV1 {
        opcode: 4,
        source_a: 0,
        source_b: 260,
        source_c: qx_commit::q4_weight_selector_v1(f.weight_index, 2).unwrap(),
        destination: 264,
    };
    let program = vm_program_tree_v1(&[instruction]).unwrap();
    let pre = VmStateV1 {
        job_hash: [0x61; 32],
        token_index: 8,
        segment_index: 2,
        program_counter: 0,
        residual_root,
        global_kv_root: [0x62; 32],
    };
    let expected = qx_consensus_math::fma_f32(0.5, 2.0, 3.0);
    let mut honest_memory = memory.clone();
    honest_memory[264..268].copy_from_slice(&expected.to_le_bytes());
    let honest_post = pre
        .successor(state_hash_v1(&honest_memory).unwrap())
        .unwrap();
    let mut miner_memory = memory.clone();
    miner_memory[264..268].copy_from_slice(&6.0_f32.to_le_bytes());
    let miner_post = pre
        .successor(state_hash_v1(&miner_memory).unwrap())
        .unwrap();
    let mut model = owned(MODEL_BYTES, f.program);
    model.data[..4].copy_from_slice(b"QXM1");
    model.data[36..68].copy_from_slice(&f.weight_root);
    model.data[68..72].copy_from_slice(&WEIGHT_LEAVES.to_le_bytes());
    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[7] = 3;
    dispute.data[40..72].copy_from_slice(f.model.as_ref());
    dispute.data[72..104].copy_from_slice(&pre.job_hash);
    dispute.data[232..264].copy_from_slice(&pre.hash().unwrap());
    dispute.data[264..296].copy_from_slice(&miner_post.hash().unwrap());
    dispute.data[296..328].copy_from_slice(&honest_post.hash().unwrap());
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[376..380].copy_from_slice(&0_u32.to_le_bytes());
    dispute.data[516..520].copy_from_slice(&pre.token_index.to_le_bytes());
    bind_segment_program(&mut dispute, &program);
    let proof = program.proof(0).unwrap();

    let mut weight_body = Vec::new();
    weight_body.extend_from_slice(&pre.program_counter.to_le_bytes());
    weight_body.extend_from_slice(&instruction.canonical_bytes());
    weight_body.extend_from_slice(&1_u32.to_le_bytes());
    weight_body.push(proof.siblings.len() as u8);
    append_siblings(&mut weight_body, &proof);
    weight_body.extend_from_slice(&f.weight_index.to_le_bytes());
    weight_body.extend_from_slice(&f.weight.tensor_id.to_le_bytes());
    weight_body.extend_from_slice(&f.weight.row.to_le_bytes());
    weight_body.extend_from_slice(&f.weight.group.to_le_bytes());
    weight_body.push(f.weight.quantization_type);
    weight_body.extend_from_slice(&f.weight.scale_bits.to_le_bytes());
    weight_body.extend_from_slice(&f.weight.packed_values);
    weight_body.push(f.weight_siblings.len() as u8);
    for sibling in &f.weight_siblings {
        weight_body.extend_from_slice(sibling);
    }
    let staged_weight = vm.process_instruction(
        &f.ix(
            31,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(f.staged, false),
            ],
            &weight_body,
        ),
        &[
            (f.model, model),
            (f.dispute, dispute.clone()),
            (f.staged, owned(STAGED_BYTES, f.program)),
        ],
    );
    assert_ok(&staged_weight.program_result);
    let staged_weight_account = account(&staged_weight.resulting_accounts, f.staged);

    let stage_body = |mask: u8, opening: &qx_commit::StateChunkProofV1| {
        let mut body = vec![mask];
        body.extend_from_slice(&pre.segment_index.to_le_bytes());
        body.extend_from_slice(&pre.program_counter.to_le_bytes());
        body.extend_from_slice(&pre.residual_root);
        body.extend_from_slice(&pre.global_kv_root);
        body.extend_from_slice(&instruction.canonical_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.push(proof.siblings.len() as u8);
        append_siblings(&mut body, &proof);
        body.extend_from_slice(&opening.byte_length.to_le_bytes());
        body.extend_from_slice(&opening.chunk_index.to_le_bytes());
        body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
        body.extend_from_slice(&opening.chunk);
        body.push(opening.siblings.len() as u8);
        for sibling in &opening.siblings {
            body.extend_from_slice(sibling);
        }
        body
    };
    let staged_a = vm.process_instruction(
        &f.ix(
            38,
            vec![
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(staged_operands, false),
            ],
            &stage_body(1, &opening_0),
        ),
        &[
            (f.dispute, dispute.clone()),
            (staged_operands, owned(STAGED_SCALAR_BYTES, f.program)),
        ],
    );
    assert_ok(&staged_a.program_result);
    let operand_account = account(&staged_a.resulting_accounts, staged_operands);
    let staged_b = vm.process_instruction(
        &f.ix(
            38,
            vec![
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(staged_operands, false),
            ],
            &stage_body(2, &opening_1),
        ),
        &[
            (f.dispute, dispute.clone()),
            (staged_operands, operand_account),
        ],
    );
    assert_ok(&staged_b.program_result);
    let operand_account = account(&staged_b.resulting_accounts, staged_operands);

    let mut resolve_body = Vec::new();
    resolve_body.extend_from_slice(&pre.segment_index.to_le_bytes());
    resolve_body.extend_from_slice(&pre.residual_root);
    resolve_body.extend_from_slice(&pre.global_kv_root);
    resolve_body.extend_from_slice(&opening_1.byte_length.to_le_bytes());
    resolve_body.extend_from_slice(&opening_1.chunk_index.to_le_bytes());
    resolve_body.extend_from_slice(&(opening_1.chunk.len() as u16).to_le_bytes());
    resolve_body.extend_from_slice(&opening_1.chunk);
    resolve_body.push(opening_1.siblings.len() as u8);
    for sibling in &opening_1.siblings {
        resolve_body.extend_from_slice(sibling);
    }
    let resolved = vm.process_instruction(
        &f.ix(
            40,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.staged, false),
                AccountMeta::new_readonly(staged_operands, false),
            ],
            &resolve_body,
        ),
        &[
            (f.dispute, dispute),
            (f.staged, staged_weight_account),
            (staged_operands, operand_account),
        ],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&dispute.data, 436), expected.to_bits());
    println!(
        "v2_cross_chunk_q4_cu={}",
        staged_weight.compute_units_consumed
            + staged_a.compute_units_consumed
            + staged_b.compute_units_consumed
            + resolved.compute_units_consumed
    );
}

#[test]
fn staged_raw_weight_fma_authenticates_granite_norm_value_on_sbf() {
    let f = Fixture::new();
    let vm = f.vm();
    let staged_operands = Pubkey::new_unique();
    let mut packed = vec![0_u8; 128];
    packed[8..12].copy_from_slice(&0.25_f32.to_le_bytes());
    let raw_leaf = WeightLeafV1 {
        tensor_id: 77,
        row: 0,
        group: 0,
        quantization_type: 0,
        packed_values: packed,
        scale_bits: 0,
    };
    let leaf_index = f.weight_index;
    let mut weight_root = raw_leaf.hash(ProtocolVersion::V1).unwrap();
    let mut position = leaf_index;
    for sibling in &f.weight_siblings {
        weight_root = if position & 1 == 0 {
            hash_domain(
                HashAlgorithm::Sha256V1,
                D_MERKLE,
                &[D_WEIGHT_TREE, &weight_root, sibling],
            )
            .unwrap()
        } else {
            hash_domain(
                HashAlgorithm::Sha256V1,
                D_MERKLE,
                &[D_WEIGHT_TREE, sibling, &weight_root],
            )
            .unwrap()
        };
        position >>= 1;
    }
    let mut memory = vec![0_u8; 256];
    memory[..4].copy_from_slice(&2.0_f32.to_le_bytes());
    memory[4..8].copy_from_slice(&3.0_f32.to_le_bytes());
    let residual_root = state_hash_v1(&memory).unwrap();
    let opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
    let instruction = VmInstructionV1 {
        opcode: 14,
        source_a: 0,
        source_b: 4,
        source_c: qx_commit::q4_weight_selector_v1(leaf_index, 2).unwrap(),
        destination: 8,
    };
    let program = vm_program_tree_v1(&[instruction]).unwrap();
    let pre = VmStateV1 {
        job_hash: [0x81; 32],
        token_index: 10,
        segment_index: 2,
        program_counter: 0,
        residual_root,
        global_kv_root: [0x82; 32],
    };
    let expected = qx_consensus_math::fma_f32(0.25, 2.0, 3.0);
    let mut honest_memory = memory.clone();
    honest_memory[8..12].copy_from_slice(&expected.to_le_bytes());
    let honest_post = pre
        .successor(state_hash_v1(&honest_memory).unwrap())
        .unwrap();
    let mut miner_memory = memory.clone();
    miner_memory[8..12].copy_from_slice(&4.0_f32.to_le_bytes());
    let miner_post = pre
        .successor(state_hash_v1(&miner_memory).unwrap())
        .unwrap();
    let mut model = owned(MODEL_BYTES, f.program);
    model.data[..4].copy_from_slice(b"QXM1");
    model.data[36..68].copy_from_slice(&weight_root);
    model.data[68..72].copy_from_slice(&WEIGHT_LEAVES.to_le_bytes());
    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[7] = 3;
    dispute.data[40..72].copy_from_slice(f.model.as_ref());
    dispute.data[72..104].copy_from_slice(&pre.job_hash);
    dispute.data[232..264].copy_from_slice(&pre.hash().unwrap());
    dispute.data[264..296].copy_from_slice(&miner_post.hash().unwrap());
    dispute.data[296..328].copy_from_slice(&honest_post.hash().unwrap());
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[376..380].copy_from_slice(&0_u32.to_le_bytes());
    dispute.data[516..520].copy_from_slice(&pre.token_index.to_le_bytes());
    bind_segment_program(&mut dispute, &program);
    let proof = program.proof(0).unwrap();
    let mut weight_body = Vec::new();
    weight_body.extend_from_slice(&pre.program_counter.to_le_bytes());
    weight_body.extend_from_slice(&instruction.canonical_bytes());
    weight_body.extend_from_slice(&1_u32.to_le_bytes());
    weight_body.push(proof.siblings.len() as u8);
    append_siblings(&mut weight_body, &proof);
    weight_body.extend_from_slice(&leaf_index.to_le_bytes());
    weight_body.extend_from_slice(&raw_leaf.tensor_id.to_le_bytes());
    weight_body.extend_from_slice(&raw_leaf.row.to_le_bytes());
    weight_body.extend_from_slice(&raw_leaf.group.to_le_bytes());
    weight_body.push(raw_leaf.quantization_type);
    weight_body.extend_from_slice(&raw_leaf.scale_bits.to_le_bytes());
    weight_body.extend_from_slice(&(raw_leaf.packed_values.len() as u16).to_le_bytes());
    weight_body.extend_from_slice(&raw_leaf.packed_values);
    weight_body.push(f.weight_siblings.len() as u8);
    for sibling in &f.weight_siblings {
        weight_body.extend_from_slice(sibling);
    }
    let staged_weight = vm.process_instruction(
        &f.ix(
            43,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(f.staged, false),
            ],
            &weight_body,
        ),
        &[
            (f.model, model),
            (f.dispute, dispute.clone()),
            (f.staged, owned(STAGED_BYTES, f.program)),
        ],
    );
    assert_ok(&staged_weight.program_result);
    let staged_weight_account = account(&staged_weight.resulting_accounts, f.staged);
    let mut operand_body = vec![3];
    operand_body.extend_from_slice(&pre.segment_index.to_le_bytes());
    operand_body.extend_from_slice(&pre.program_counter.to_le_bytes());
    operand_body.extend_from_slice(&pre.residual_root);
    operand_body.extend_from_slice(&pre.global_kv_root);
    operand_body.extend_from_slice(&instruction.canonical_bytes());
    operand_body.extend_from_slice(&1_u32.to_le_bytes());
    operand_body.push(proof.siblings.len() as u8);
    append_siblings(&mut operand_body, &proof);
    operand_body.extend_from_slice(&opening.byte_length.to_le_bytes());
    operand_body.extend_from_slice(&opening.chunk_index.to_le_bytes());
    operand_body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
    operand_body.extend_from_slice(&opening.chunk);
    operand_body.push(opening.siblings.len() as u8);
    for sibling in &opening.siblings {
        operand_body.extend_from_slice(sibling);
    }
    let staged_values = vm.process_instruction(
        &f.ix(
            38,
            vec![
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(staged_operands, false),
            ],
            &operand_body,
        ),
        &[
            (f.dispute, dispute.clone()),
            (staged_operands, owned(STAGED_SCALAR_BYTES, f.program)),
        ],
    );
    assert_ok(&staged_values.program_result);
    let operand_account = account(&staged_values.resulting_accounts, staged_operands);
    let mut resolve_body = Vec::new();
    resolve_body.extend_from_slice(&pre.segment_index.to_le_bytes());
    resolve_body.extend_from_slice(&pre.residual_root);
    resolve_body.extend_from_slice(&pre.global_kv_root);
    resolve_body.extend_from_slice(&opening.byte_length.to_le_bytes());
    resolve_body.extend_from_slice(&opening.chunk_index.to_le_bytes());
    resolve_body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
    resolve_body.extend_from_slice(&opening.chunk);
    resolve_body.push(opening.siblings.len() as u8);
    for sibling in &opening.siblings {
        resolve_body.extend_from_slice(sibling);
    }
    let resolved = vm.process_instruction(
        &f.ix(
            40,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.staged, false),
                AccountMeta::new_readonly(staged_operands, false),
            ],
            &resolve_body,
        ),
        &[
            (f.dispute, dispute),
            (f.staged, staged_weight_account),
            (staged_operands, operand_account),
        ],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&dispute.data, 436), expected.to_bits());
    println!(
        "v2_raw_weight_fma_cu={}",
        staged_weight.compute_units_consumed
            + staged_values.compute_units_consumed
            + resolved.compute_units_consumed
    );
}

#[test]
fn chained_vm_kv_append_convicts_wrong_frontier_on_sbf() {
    let f = Fixture::new();
    let vm = f.vm();
    let key_root = [0x31; 32];
    let value_root = [0x52; 32];
    let mut memory = vec![0_u8; 256];
    memory[..32].copy_from_slice(&key_root);
    memory[32..64].copy_from_slice(&value_root);
    let memory_root = state_hash_v1(&memory).unwrap();
    let opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
    let empty = hash_domain(
        HashAlgorithm::Blake3V1,
        qx_commit::D_MERKLE_EMPTY,
        &[qx_commit::D_KV],
    )
    .unwrap();
    let layer_tree =
        MerkleTree::new(HashAlgorithm::Blake3V1, qx_commit::D_KV, &[empty; 4]).unwrap();
    let global_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        qx_commit::D_KV,
        &[layer_tree.root(); 3],
    )
    .unwrap();
    let kv_leaf = qx_commit::kv_leaf_hash_from_roots_v1(1, 1, key_root, value_root).unwrap();
    let new_layer_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        qx_commit::D_KV,
        &[empty, kv_leaf, empty, empty],
    )
    .unwrap();
    let new_global_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        qx_commit::D_KV,
        &[layer_tree.root(), new_layer_tree.root(), layer_tree.root()],
    )
    .unwrap();
    let instruction = VmInstructionV1 {
        opcode: 11,
        source_a: 0,
        source_b: 32,
        source_c: 1,
        destination: 0,
    };
    let program = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_PROGRAM_TREE,
        &[instruction.hash(0).unwrap()],
    )
    .unwrap();
    let pre = VmStateV1 {
        job_hash: [0x71; 32],
        token_index: 1,
        segment_index: 7,
        program_counter: 0,
        residual_root: memory_root,
        global_kv_root: global_tree.root(),
    };
    let honest_post = pre
        .successor_with_roots(memory_root, new_global_tree.root())
        .unwrap();
    let miner_post = pre.successor_with_roots(memory_root, [0x99; 32]).unwrap();

    let mut model = owned(MODEL_BYTES, f.program);
    model.data[..4].copy_from_slice(b"QXM1");
    model.data[104..136].copy_from_slice(&program.root());
    model.data[136..140].copy_from_slice(&1_u32.to_le_bytes());
    model.data[144..146].copy_from_slice(&3_u16.to_le_bytes());
    model.data[148..152].copy_from_slice(&3_u32.to_le_bytes());
    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[40..72].copy_from_slice(f.model.as_ref());
    dispute.data[72..104].copy_from_slice(&pre.job_hash);
    dispute.data[232..264].copy_from_slice(&pre.hash().unwrap());
    dispute.data[264..296].copy_from_slice(&miner_post.hash().unwrap());
    dispute.data[296..328].copy_from_slice(&honest_post.hash().unwrap());
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[376..380].copy_from_slice(&0_u32.to_le_bytes());
    dispute.data[516..520].copy_from_slice(&pre.token_index.to_le_bytes());
    bind_segment_program(&mut dispute, &program);

    let layer_proof = layer_tree.proof(1).unwrap();
    let global_proof = global_tree.proof(1).unwrap();
    let mut stage_body = Vec::new();
    stage_body.extend_from_slice(&pre.segment_index.to_le_bytes());
    stage_body.extend_from_slice(&pre.program_counter.to_le_bytes());
    stage_body.extend_from_slice(&pre.residual_root);
    stage_body.extend_from_slice(&pre.global_kv_root);
    stage_body.extend_from_slice(&1_u16.to_le_bytes());
    stage_body.extend_from_slice(&key_root);
    stage_body.extend_from_slice(&value_root);
    stage_body.push(layer_proof.siblings.len() as u8);
    append_siblings(&mut stage_body, &layer_proof);
    stage_body.push(global_proof.siblings.len() as u8);
    append_siblings(&mut stage_body, &global_proof);
    let staged = vm.process_instruction(
        &f.ix(
            33,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(f.staged, false),
            ],
            &stage_body,
        ),
        &[
            (f.model, model.clone()),
            (f.dispute, dispute.clone()),
            (f.staged, owned(STAGED_KV_BYTES, f.program)),
        ],
    );
    assert_ok(&staged.program_result);
    let stage_cu = staged.compute_units_consumed;
    let staged_kv = account(&staged.resulting_accounts, f.staged);

    let instruction_proof = program.proof(0).unwrap();
    let mut resolve_body = Vec::new();
    resolve_body.extend_from_slice(&pre.segment_index.to_le_bytes());
    resolve_body.extend_from_slice(&pre.residual_root);
    resolve_body.extend_from_slice(&pre.global_kv_root);
    resolve_body.extend_from_slice(&instruction.canonical_bytes());
    resolve_body.extend_from_slice(&1_u32.to_le_bytes());
    resolve_body.push(instruction_proof.siblings.len() as u8);
    append_siblings(&mut resolve_body, &instruction_proof);
    resolve_body.extend_from_slice(&opening.byte_length.to_le_bytes());
    resolve_body.extend_from_slice(&opening.chunk_index.to_le_bytes());
    resolve_body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
    resolve_body.extend_from_slice(&opening.chunk);
    resolve_body.push(opening.siblings.len() as u8);
    for sibling in &opening.siblings {
        resolve_body.extend_from_slice(sibling);
    }
    let resolved = vm.process_instruction(
        &f.ix(
            34,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.staged, false),
            ],
            &resolve_body,
        ),
        &[
            (f.model, model),
            (f.dispute, dispute),
            (f.staged, staged_kv),
        ],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    println!(
        "v2_chained_vm_kv_append_cu={}",
        stage_cu + resolved.compute_units_consumed
    );
}

#[test]
fn chained_vm_argmax_enforces_lowest_token_tie_on_sbf() {
    let f = Fixture::new();
    let vm = f.vm();
    let mut memory = vec![0_u8; 256];
    memory[..4].copy_from_slice(&7.0_f32.to_le_bytes());
    memory[4..8].copy_from_slice(&42_u32.to_le_bytes());
    memory[8..12].copy_from_slice(&7.0_f32.to_le_bytes());
    memory[12..16].copy_from_slice(&9_u32.to_le_bytes());
    memory[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    let memory_root = state_hash_v1(&memory).unwrap();
    let opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
    let instruction = VmInstructionV1 {
        opcode: 8,
        source_a: 0,
        source_b: 8,
        source_c: 0,
        destination: 16,
    };
    let program = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_PROGRAM_TREE,
        &[instruction.hash(0).unwrap(), instruction.hash(1).unwrap()],
    )
    .unwrap();
    let pre = VmStateV1 {
        job_hash: [0x81; 32],
        token_index: 2,
        segment_index: 8,
        program_counter: 0,
        residual_root: memory_root,
        global_kv_root: [0x91; 32],
    };
    let mut honest_memory = memory.clone();
    honest_memory[16..20].copy_from_slice(&7.0_f32.to_bits().to_le_bytes());
    honest_memory[20..24].copy_from_slice(&9_u32.to_le_bytes());
    let honest_root = state_hash_v1(&honest_memory).unwrap();
    let mut miner_memory = memory.clone();
    miner_memory[16..20].copy_from_slice(&7.0_f32.to_bits().to_le_bytes());
    miner_memory[20..24].copy_from_slice(&42_u32.to_le_bytes());
    let miner_root = state_hash_v1(&miner_memory).unwrap();
    let honest_post = pre.successor(honest_root).unwrap();
    let miner_post = pre.successor(miner_root).unwrap();
    let mut model = owned(MODEL_BYTES, f.program);
    model.data[..4].copy_from_slice(b"QXM1");
    model.data[104..136].copy_from_slice(&program.root());
    model.data[136..140].copy_from_slice(&2_u32.to_le_bytes());
    model.data[152..156].copy_from_slice(&49_155_u32.to_le_bytes());
    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[7] = 4;
    dispute.data[40..72].copy_from_slice(f.model.as_ref());
    dispute.data[72..104].copy_from_slice(&pre.job_hash);
    dispute.data[232..264].copy_from_slice(&pre.hash().unwrap());
    dispute.data[264..296].copy_from_slice(&miner_post.hash().unwrap());
    dispute.data[296..328].copy_from_slice(&honest_post.hash().unwrap());
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[376..380].copy_from_slice(&0_u32.to_le_bytes());
    dispute.data[516..520].copy_from_slice(&pre.token_index.to_le_bytes());
    bind_segment_program(&mut dispute, &program);

    let instruction_proof = program.proof(0).unwrap();
    let mut body = Vec::new();
    body.extend_from_slice(&pre.segment_index.to_le_bytes());
    body.extend_from_slice(&pre.program_counter.to_le_bytes());
    body.extend_from_slice(&pre.residual_root);
    body.extend_from_slice(&pre.global_kv_root);
    body.extend_from_slice(&instruction.canonical_bytes());
    body.extend_from_slice(&2_u32.to_le_bytes());
    body.push(instruction_proof.siblings.len() as u8);
    append_siblings(&mut body, &instruction_proof);
    body.extend_from_slice(&opening.byte_length.to_le_bytes());
    body.extend_from_slice(&opening.chunk_index.to_le_bytes());
    body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
    body.extend_from_slice(&opening.chunk);
    body.push(opening.siblings.len() as u8);
    for sibling in &opening.siblings {
        body.extend_from_slice(sibling);
    }
    let resolved = vm.process_instruction(
        &f.ix(
            35,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
            ],
            &body,
        ),
        &[(f.model, model), (f.dispute, dispute)],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&dispute.data, 436), 9);
    println!(
        "v2_chained_vm_argmax_cu={}",
        resolved.compute_units_consumed
    );
}

#[test]
fn staged_vm_argmax_resolves_cross_chunk_final_output_on_sbf() {
    let f = Fixture::new();
    let vm = f.vm();
    let mut memory = vec![0_u8; 512];
    memory[..4].copy_from_slice(&7.0_f32.to_le_bytes());
    memory[4..8].copy_from_slice(&42_u32.to_le_bytes());
    memory[260..264].copy_from_slice(&7.0_f32.to_le_bytes());
    memory[264..268].copy_from_slice(&9_u32.to_le_bytes());
    memory[268..272].copy_from_slice(&f32::NEG_INFINITY.to_le_bytes());
    memory[272..276].copy_from_slice(&u32::MAX.to_le_bytes());
    let residual_root = state_hash_v1(&memory).unwrap();
    let opening_0 = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
    let opening_1 = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 1).unwrap();
    let instruction = VmInstructionV1 {
        opcode: 8,
        source_a: 0,
        source_b: 260,
        source_c: 0,
        destination: 268,
    };
    let program = vm_program_tree_v1(&[instruction]).unwrap();
    let pre = VmStateV1 {
        job_hash: [0x71; 32],
        token_index: 9,
        segment_index: 3,
        program_counter: 0,
        residual_root,
        global_kv_root: [0x72; 32],
    };
    let mut honest_memory = memory.clone();
    honest_memory[268..272].copy_from_slice(&7.0_f32.to_le_bytes());
    honest_memory[272..276].copy_from_slice(&9_u32.to_le_bytes());
    let honest_output = VmOutputStateV1 {
        job_hash: pre.job_hash,
        token_index: pre.token_index,
        segment_index: pre.segment_index,
        program_counter: 1,
        residual_root: state_hash_v1(&honest_memory).unwrap(),
        global_kv_root: pre.global_kv_root,
        predicted_token: 9,
    };
    let mut miner_memory = memory.clone();
    miner_memory[268..272].copy_from_slice(&7.0_f32.to_le_bytes());
    miner_memory[272..276].copy_from_slice(&42_u32.to_le_bytes());
    let miner_output = VmOutputStateV1 {
        residual_root: state_hash_v1(&miner_memory).unwrap(),
        predicted_token: 42,
        ..honest_output
    };
    let mut model = owned(MODEL_BYTES, f.program);
    model.data[..4].copy_from_slice(b"QXM1");
    model.data[152..156].copy_from_slice(&49_155_u32.to_le_bytes());
    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[7] = 4;
    dispute.data[40..72].copy_from_slice(f.model.as_ref());
    dispute.data[72..104].copy_from_slice(&pre.job_hash);
    dispute.data[232..264].copy_from_slice(&pre.hash().unwrap());
    dispute.data[264..296].copy_from_slice(&miner_output.hash().unwrap());
    dispute.data[296..328].copy_from_slice(&honest_output.hash().unwrap());
    dispute.data[360..368].copy_from_slice(&1_000_u64.to_le_bytes());
    dispute.data[368..376].copy_from_slice(&1_200_u64.to_le_bytes());
    dispute.data[376..380].copy_from_slice(&0_u32.to_le_bytes());
    dispute.data[516..520].copy_from_slice(&pre.token_index.to_le_bytes());
    bind_segment_program(&mut dispute, &program);
    let proof = program.proof(0).unwrap();
    let stage_body = |mask: u8, opening: &qx_commit::StateChunkProofV1| {
        let mut body = vec![mask];
        body.extend_from_slice(&pre.segment_index.to_le_bytes());
        body.extend_from_slice(&pre.program_counter.to_le_bytes());
        body.extend_from_slice(&pre.residual_root);
        body.extend_from_slice(&pre.global_kv_root);
        body.extend_from_slice(&instruction.canonical_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.push(proof.siblings.len() as u8);
        append_siblings(&mut body, &proof);
        body.extend_from_slice(&opening.byte_length.to_le_bytes());
        body.extend_from_slice(&opening.chunk_index.to_le_bytes());
        body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
        body.extend_from_slice(&opening.chunk);
        body.push(opening.siblings.len() as u8);
        for sibling in &opening.siblings {
            body.extend_from_slice(sibling);
        }
        body
    };
    let staged_left = vm.process_instruction(
        &f.ix(
            41,
            vec![
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(f.staged, false),
            ],
            &stage_body(1, &opening_0),
        ),
        &[
            (f.dispute, dispute.clone()),
            (f.staged, owned(STAGED_ARGMAX_BYTES, f.program)),
        ],
    );
    assert_ok(&staged_left.program_result);
    let staged = account(&staged_left.resulting_accounts, f.staged);
    let staged_right = vm.process_instruction(
        &f.ix(
            41,
            vec![
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(f.staged, false),
            ],
            &stage_body(2, &opening_1),
        ),
        &[(f.dispute, dispute.clone()), (f.staged, staged)],
    );
    assert_ok(&staged_right.program_result);
    let staged = account(&staged_right.resulting_accounts, f.staged);
    let mut resolve_body = Vec::new();
    resolve_body.extend_from_slice(&pre.segment_index.to_le_bytes());
    resolve_body.extend_from_slice(&pre.residual_root);
    resolve_body.extend_from_slice(&pre.global_kv_root);
    resolve_body.extend_from_slice(&opening_1.byte_length.to_le_bytes());
    resolve_body.extend_from_slice(&opening_1.chunk_index.to_le_bytes());
    resolve_body.extend_from_slice(&(opening_1.chunk.len() as u16).to_le_bytes());
    resolve_body.extend_from_slice(&opening_1.chunk);
    resolve_body.push(opening_1.siblings.len() as u8);
    for sibling in &opening_1.siblings {
        resolve_body.extend_from_slice(sibling);
    }
    let resolved = vm.process_instruction(
        &f.ix(
            42,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.staged, false),
            ],
            &resolve_body,
        ),
        &[(f.model, model), (f.dispute, dispute), (f.staged, staged)],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&dispute.data, 436), 9);
    println!(
        "v2_cross_chunk_argmax_cu={}",
        staged_left.compute_units_consumed
            + staged_right.compute_units_consumed
            + resolved.compute_units_consumed
    );
}

#[test]
fn pure_state_machine_timeout_penalizes_each_non_responder() {
    let mut miner_due =
        qx_verifier_core::BisectionGameV1::open(3, [1; 32], [2; 32], 10, 3).unwrap();
    assert_eq!(
        miner_due.timeout(14).unwrap(),
        qx_verifier_core::DisputePartyV1::Challenger
    );
    let mut challenger_due =
        qx_verifier_core::BisectionGameV1::open(3, [1; 32], [2; 32], 10, 3).unwrap();
    challenger_due
        .submit_miner_bounds([1; 32], [3; 32], 11)
        .unwrap();
    assert_eq!(
        challenger_due.timeout(15).unwrap(),
        qx_verifier_core::DisputePartyV1::Miner
    );
}

#[test]
fn sbf_timeout_awards_the_other_party_for_each_turn() {
    let challenger_due = Fixture::new();
    let mut vm = challenger_due.vm();
    let (model, epoch, _) = challenger_due.seeded_epoch(&mut vm);
    let (dispute, _) = challenger_due.open(&vm, epoch, model);
    vm.sysvars.clock.slot = 1_303;
    let timed_out = vm.process_instruction(
        &challenger_due.ix(
            20,
            vec![AccountMeta::new(challenger_due.dispute, false)],
            &[],
        ),
        &[(challenger_due.dispute, dispute)],
    );
    assert_ok(&timed_out.program_result);
    assert_eq!(
        account(&timed_out.resulting_accounts, challenger_due.dispute).data[5],
        1
    );

    let miner_due = Fixture::new();
    let mut vm = miner_due.vm();
    let (model, epoch, _) = miner_due.seeded_epoch(&mut vm);
    let (mut dispute, _) = miner_due.open(&vm, epoch, model);
    dispute.data[7] = 3;
    let (dispute, _) = miner_due.bounds(&vm, dispute, true);
    vm.sysvars.clock.slot = 1_303;
    let timed_out = vm.process_instruction(
        &miner_due.ix(20, vec![AccountMeta::new(miner_due.dispute, false)], &[]),
        &[(miner_due.dispute, dispute)],
    );
    assert_ok(&timed_out.program_result);
    assert_eq!(
        account(&timed_out.resulting_accounts, miner_due.dispute).data[5],
        2
    );
}

#[test]
fn sbf_rejects_a_result_that_epoch_sampling_did_not_select() {
    let f = Fixture::with_audit_probability(10_000);
    let mut vm = f.vm();
    let (model, epoch, _) = f.seeded_epoch(&mut vm);
    let opened = vm.process_transaction_instructions(
        &[f.job_signature_ix(), f.open_ix()],
        &[
            (f.epoch, epoch),
            (f.model, model),
            (f.dispute, owned(DISPUTE_BYTES, f.program)),
            (f.challenger, Account::default()),
            ed25519_program_account(),
        ],
    );
    assert!(opened.program_result.is_err());
}

#[test]
fn sbf_rejects_forged_job_authorization() {
    let f = Fixture::new();
    let mut vm = f.vm();
    let (model, epoch, _) = f.seeded_epoch(&mut vm);
    let wrong_message = [88; 32];
    let wrong_signature = f.requester_signing.sign(&wrong_message).to_bytes();
    let forged = solana_ed25519_program::new_ed25519_instruction_with_signature(
        &wrong_message,
        &wrong_signature,
        &f.requester_signing.verifying_key().to_bytes(),
    );
    let opened = vm.process_transaction_instructions(
        &[forged, f.open_ix()],
        &[
            (f.epoch, epoch),
            (f.model, model),
            (f.dispute, owned(DISPUTE_BYTES, f.program)),
            (f.challenger, Account::default()),
            ed25519_program_account(),
        ],
    );
    assert!(opened.program_result.is_err());
}

#[test]
fn sbf_miner_identity_rejects_duplicate_epoch_grinding() {
    let f = Fixture::new();
    let vm = f.vm();
    let (_, _, miner_state) = f.register(&vm);
    let alternate_epoch = Pubkey::new_unique();
    let mut body = Vec::new();
    body.extend_from_slice(&7_u64.to_le_bytes());
    body.extend_from_slice(&f.results.root());
    body.extend_from_slice(&2_u32.to_le_bytes());
    let duplicate = vm.process_instruction(
        &f.ix(
            9,
            vec![
                AccountMeta::new(alternate_epoch, true),
                AccountMeta::new(f.miner, true),
            ],
            &body,
        ),
        &[
            (alternate_epoch, owned(EPOCH_BYTES, f.program)),
            (f.miner, miner_state),
        ],
    );
    assert!(duplicate.program_result.is_err());
}

#[test]
fn sbf_bisects_the_actual_runtime_job_leaf_format() {
    let mut f = Fixture::new();
    let instruction = VmInstructionV1 {
        opcode: 4,
        source_a: 136,
        source_b: 4,
        source_c: qx_commit::q4_weight_selector_v1(f.weight_index, 2).unwrap(),
        destination: 12,
    };
    f.verification_instruction = instruction;
    let job_genesis = job_genesis_hash_v1(&f.job.job_id).unwrap();
    let mut common_memory = vec![0_u8; 512];
    for (index, value) in [2.0_f32, 3.0, 0.0, -1.0].iter().enumerate() {
        common_memory[index * 4..index * 4 + 4].copy_from_slice(&value.to_bits().to_le_bytes());
    }
    common_memory[136..140].copy_from_slice(&2.0_f32.to_bits().to_le_bytes());
    let common_residual = state_hash_v1(&common_memory).unwrap();
    let common_opening =
        prove_state_chunk_v1(StateTreeKindV1::Residual, &common_memory, 0).unwrap();
    let honest_residual = common_opening
        .root_after_scalar_write(12, 4.0_f32.to_bits())
        .unwrap();
    let miner_residual = common_opening
        .root_after_scalar_write(12, 6.0_f32.to_bits())
        .unwrap();
    let common_kv = [12; 32];
    let common_checkpoint = CheckpointV1 {
        job_id: f.job.job_id,
        token_index: 1,
        layer_index: u16::MAX,
        residual_hash: common_residual,
        global_kv_root: common_kv,
    };
    let honest_checkpoint = CheckpointV1 {
        layer_index: 1,
        residual_hash: honest_residual,
        ..common_checkpoint
    };
    let miner_checkpoint = CheckpointV1 {
        residual_hash: miner_residual,
        ..honest_checkpoint
    };
    let common_checkpoint_hash = common_checkpoint.hash().unwrap();
    let honest_checkpoint_hash = honest_checkpoint.hash().unwrap();
    let miner_checkpoint_hash = miner_checkpoint.hash().unwrap();
    let honest_final_hash = CheckpointV1 {
        layer_index: 3,
        residual_hash: [21; 32],
        ..honest_checkpoint
    }
    .hash()
    .unwrap();
    let miner_final_hash = CheckpointV1 {
        layer_index: 3,
        residual_hash: [22; 32],
        ..miner_checkpoint
    }
    .hash()
    .unwrap();
    let honest_checkpoint_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_TOKEN,
        &[
            common_checkpoint_hash,
            honest_checkpoint_hash,
            honest_final_hash,
        ],
    )
    .unwrap();
    let miner_checkpoint_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_TOKEN,
        &[
            common_checkpoint_hash,
            miner_checkpoint_hash,
            miner_final_hash,
        ],
    )
    .unwrap();
    let prior_final_checkpoint = CheckpointV1 {
        job_id: f.job.job_id,
        token_index: 0,
        layer_index: 3,
        residual_hash: [7; 32],
        global_kv_root: common_kv,
    };
    let prior_checkpoint_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_TOKEN,
        &[
            CheckpointV1 {
                layer_index: u16::MAX,
                ..prior_final_checkpoint
            }
            .hash()
            .unwrap(),
            CheckpointV1 {
                layer_index: 1,
                ..prior_final_checkpoint
            }
            .hash()
            .unwrap(),
            prior_final_checkpoint.hash().unwrap(),
        ],
    )
    .unwrap();
    let common_token = TokenResultV1 {
        job_id: f.job.job_id,
        token_index: 0,
        checkpoint_root: prior_checkpoint_tree.root(),
        predicted_token: 41,
        checkpoint_count: 3,
    };
    let honest_token = TokenResultV1 {
        job_id: f.job.job_id,
        token_index: 1,
        checkpoint_root: honest_checkpoint_tree.root(),
        predicted_token: 42,
        checkpoint_count: 3,
    };
    let miner_token = TokenResultV1 {
        checkpoint_root: miner_checkpoint_tree.root(),
        predicted_token: 99,
        ..honest_token
    };
    let common_token_hash = common_token.hash().unwrap();
    let honest_token_hash = honest_token.hash().unwrap();
    let miner_token_hash = miner_token.hash().unwrap();
    f.honest_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_JOB,
        &[job_genesis, common_token_hash, honest_token_hash],
    )
    .unwrap();
    f.miner_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_JOB,
        &[job_genesis, common_token_hash, miner_token_hash],
    )
    .unwrap();
    f.result.trace_root = f.miner_tree.root();
    f.results = MerkleTree::new(
        HashAlgorithm::Sha256V1,
        D_BATCH,
        &[f.result.hash().unwrap(), [99; 32]],
    )
    .unwrap();

    let mut vm = f.vm();
    let (model, epoch, _) = f.seeded_epoch(&mut vm);
    let (dispute, _) = f.open(&vm, epoch, model.clone());
    let (dispute, _) = f.bounds_hashes(&vm, dispute, true, job_genesis, honest_token_hash);
    let (dispute, _) = f.bounds_hashes(&vm, dispute, false, job_genesis, miner_token_hash);
    let (dispute, _) = f.midpoint_hash(&vm, dispute, true, common_token_hash);
    let (dispute, _) = f.midpoint_hash(&vm, dispute, false, common_token_hash);
    assert_eq!(dispute.data[4], 9);
    let mut challenger_token_body = honest_token.canonical_bytes().to_vec();
    challenger_token_body.extend_from_slice(&common_token.predicted_token.to_le_bytes());
    challenger_token_body.extend_from_slice(&common_token.canonical_bytes());
    challenger_token_body.extend_from_slice(&prior_final_checkpoint.canonical_bytes());
    let prior_final_proof = prior_checkpoint_tree.proof(2).unwrap();
    challenger_token_body.push(prior_final_proof.siblings.len() as u8);
    append_siblings(&mut challenger_token_body, &prior_final_proof);
    let challenger_token = vm.process_instruction(
        &f.ix(
            25,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &challenger_token_body,
        ),
        &[
            (f.model, model.clone()),
            (f.dispute, dispute),
            (f.challenger, Account::default()),
        ],
    );
    assert_ok(&challenger_token.program_result);
    let dispute = account(&challenger_token.resulting_accounts, f.dispute);
    let miner_token_reveal = vm.process_instruction(
        &f.ix(
            26,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
            ],
            &miner_token.canonical_bytes(),
        ),
        &[(f.dispute, dispute), (f.miner, Account::default())],
    );
    assert_ok(&miner_token_reveal.program_result);
    let dispute = account(&miner_token_reveal.resulting_accounts, f.dispute);
    assert_eq!(dispute.data[4], 1);
    assert_eq!(dispute.data[7], 2);

    let (dispute, _) = f.bounds_tree(
        &vm,
        dispute,
        true,
        &honest_checkpoint_tree,
        common_checkpoint_hash,
        honest_final_hash,
    );
    let (dispute, _) = f.bounds_tree(
        &vm,
        dispute,
        false,
        &miner_checkpoint_tree,
        common_checkpoint_hash,
        miner_final_hash,
    );
    let (dispute, _) = f.midpoint_tree(
        &vm,
        dispute,
        true,
        &honest_checkpoint_tree,
        honest_checkpoint_hash,
    );
    let (dispute, _) = f.midpoint_tree(
        &vm,
        dispute,
        false,
        &miner_checkpoint_tree,
        miner_checkpoint_hash,
    );
    assert_eq!(dispute.data[4], 11);

    let challenger_checkpoint = vm.process_instruction(
        &f.ix(
            27,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &[
                common_checkpoint.canonical_bytes(),
                honest_checkpoint.canonical_bytes(),
            ]
            .concat(),
        ),
        &[
            (f.model, model),
            (f.dispute, dispute),
            (f.challenger, Account::default()),
        ],
    );
    assert_ok(&challenger_checkpoint.program_result);
    let model = account(&challenger_checkpoint.resulting_accounts, f.model);
    let dispute = account(&challenger_checkpoint.resulting_accounts, f.dispute);
    let miner_checkpoint_reveal = vm.process_instruction(
        &f.ix(
            28,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
            ],
            &miner_checkpoint.canonical_bytes(),
        ),
        &[(f.dispute, dispute), (f.miner, Account::default())],
    );
    assert_ok(&miner_checkpoint_reveal.program_result);
    let dispute = account(&miner_checkpoint_reveal.resulting_accounts, f.dispute);
    assert_eq!(dispute.data[4], 13);
    assert_eq!(&dispute.data[536..568], &honest_checkpoint.residual_hash);
    assert_eq!(&dispute.data[600..632], &miner_checkpoint.residual_hash);

    let program = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_PROGRAM_TREE,
        &[instruction.hash(0).unwrap()],
    )
    .unwrap();
    let pre = VmStateV1 {
        job_hash: f.job.hash().unwrap(),
        token_index: 1,
        segment_index: 1,
        program_counter: 0,
        residual_root: common_residual,
        global_kv_root: common_kv,
    };
    let honest_post = pre.successor(honest_residual).unwrap();
    let miner_post = pre.successor(miner_residual).unwrap();
    let pre_hash = pre.hash().unwrap();
    let honest_post_hash = honest_post.hash().unwrap();
    let miner_post_hash = miner_post.hash().unwrap();
    let honest_operator = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_OPERATOR_TREE,
        &[pre_hash, honest_post_hash],
    )
    .unwrap();
    let miner_operator = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_OPERATOR_TREE,
        &[pre_hash, miner_post_hash],
    )
    .unwrap();
    let operator_bounds = |tree: &MerkleTree, start_hash: [u8; 32], end_hash: [u8; 32]| {
        let start = tree.proof(0).unwrap();
        let end = tree.proof(1).unwrap();
        let mut body = Vec::new();
        body.extend_from_slice(&tree.root());
        body.extend_from_slice(&start_hash);
        body.extend_from_slice(&end_hash);
        body.push(start.siblings.len() as u8);
        append_siblings(&mut body, &start);
        append_siblings(&mut body, &end);
        body
    };
    let forged_pre = VmStateV1 {
        residual_root: [0x99; 32],
        ..pre
    };
    let forged_pre_hash = forged_pre.hash().unwrap();
    let forged_operator = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_OPERATOR_TREE,
        &[forged_pre_hash, honest_post_hash],
    )
    .unwrap();
    let mut wrong_segment_body = Vec::new();
    f.append_segment_program(&mut wrong_segment_body, 3);
    wrong_segment_body.extend_from_slice(&operator_bounds(
        &honest_operator,
        pre_hash,
        honest_post_hash,
    ));
    let wrong_segment = vm.process_instruction(
        &f.ix(
            29,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &wrong_segment_body,
        ),
        &[
            (f.model, model.clone()),
            (f.dispute, dispute.clone()),
            (f.challenger, Account::default()),
        ],
    );
    assert!(wrong_segment.program_result.is_err());
    let mut forged_operator_body = Vec::new();
    f.append_segment_program(&mut forged_operator_body, 1);
    forged_operator_body.extend_from_slice(&operator_bounds(
        &forged_operator,
        forged_pre_hash,
        honest_post_hash,
    ));
    let forged_boundary = vm.process_instruction(
        &f.ix(
            29,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &forged_operator_body,
        ),
        &[
            (f.model, model.clone()),
            (f.dispute, dispute.clone()),
            (f.challenger, Account::default()),
        ],
    );
    assert!(forged_boundary.program_result.is_err());
    let mut challenger_operator_body = Vec::new();
    f.append_segment_program(&mut challenger_operator_body, 1);
    challenger_operator_body.extend_from_slice(&operator_bounds(
        &honest_operator,
        pre_hash,
        honest_post_hash,
    ));
    let challenger_operator = vm.process_instruction(
        &f.ix(
            29,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &challenger_operator_body,
        ),
        &[
            (f.model, model.clone()),
            (f.dispute, dispute),
            (f.challenger, Account::default()),
        ],
    );
    assert_ok(&challenger_operator.program_result);
    let challenger_operator_cu = challenger_operator.compute_units_consumed;
    let dispute = account(&challenger_operator.resulting_accounts, f.dispute);
    let miner_operator_reveal = vm.process_instruction(
        &f.ix(
            30,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
            ],
            &operator_bounds(&miner_operator, pre_hash, miner_post_hash),
        ),
        &[
            (f.model, model.clone()),
            (f.dispute, dispute),
            (f.miner, Account::default()),
        ],
    );
    assert_ok(&miner_operator_reveal.program_result);
    let miner_operator_cu = miner_operator_reveal.compute_units_consumed;
    let dispute = account(&miner_operator_reveal.resulting_accounts, f.dispute);
    assert_eq!(dispute.data[4], 5);

    let instruction_proof = program.proof(0).unwrap();
    let mut stage_body = Vec::new();
    stage_body.extend_from_slice(&pre.program_counter.to_le_bytes());
    stage_body.extend_from_slice(&instruction.canonical_bytes());
    stage_body.extend_from_slice(&1_u32.to_le_bytes());
    stage_body.push(instruction_proof.siblings.len() as u8);
    append_siblings(&mut stage_body, &instruction_proof);
    stage_body.extend_from_slice(&f.weight_index.to_le_bytes());
    stage_body.extend_from_slice(&f.weight.tensor_id.to_le_bytes());
    stage_body.extend_from_slice(&f.weight.row.to_le_bytes());
    stage_body.extend_from_slice(&f.weight.group.to_le_bytes());
    stage_body.push(f.weight.quantization_type);
    stage_body.extend_from_slice(&f.weight.scale_bits.to_le_bytes());
    stage_body.extend_from_slice(&f.weight.packed_values);
    stage_body.push(f.weight_siblings.len() as u8);
    for sibling in &f.weight_siblings {
        stage_body.extend_from_slice(sibling);
    }
    let staged = vm.process_instruction(
        &f.ix(
            31,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new_readonly(f.dispute, false),
                AccountMeta::new(f.staged, false),
            ],
            &stage_body,
        ),
        &[
            (f.model, model),
            (f.dispute, dispute.clone()),
            (f.staged, owned(STAGED_BYTES, f.program)),
        ],
    );
    assert_ok(&staged.program_result);
    let staged_cu = staged.compute_units_consumed;
    let staged_weight = account(&staged.resulting_accounts, f.staged);
    let mut terminal = Vec::new();
    terminal.extend_from_slice(&pre.segment_index.to_le_bytes());
    terminal.extend_from_slice(&pre.residual_root);
    terminal.extend_from_slice(&pre.global_kv_root);
    terminal.extend_from_slice(&common_opening.byte_length.to_le_bytes());
    terminal.extend_from_slice(&common_opening.chunk_index.to_le_bytes());
    terminal.extend_from_slice(&(common_opening.chunk.len() as u16).to_le_bytes());
    terminal.extend_from_slice(&common_opening.chunk);
    terminal.push(common_opening.siblings.len() as u8);
    for sibling in &common_opening.siblings {
        terminal.extend_from_slice(sibling);
    }
    let resolved = vm.process_instruction(
        &f.ix(
            32,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.staged, false),
            ],
            &terminal,
        ),
        &[(f.dispute, dispute), (f.staged, staged_weight)],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    println!(
        "v2_production_hierarchy_q4_terminal_cu={}",
        staged_cu + resolved.compute_units_consumed
    );
    println!(
        "v2_operator_boundary_cu={}",
        challenger_operator_cu + miner_operator_cu
    );
}

#[test]
fn equal_checkpoints_descend_into_final_head_and_convict_wrong_token_on_sbf() {
    let mut f = Fixture::new();
    let instruction = VmInstructionV1 {
        opcode: 8,
        source_a: 0,
        source_b: 8,
        source_c: 0,
        destination: 16,
    };
    f.verification_instruction = instruction;

    let mut memory = vec![0_u8; 256];
    memory[..4].copy_from_slice(&7.0_f32.to_le_bytes());
    memory[4..8].copy_from_slice(&42_u32.to_le_bytes());
    memory[8..12].copy_from_slice(&7.0_f32.to_le_bytes());
    memory[12..16].copy_from_slice(&9_u32.to_le_bytes());
    memory[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
    let memory_root = state_hash_v1(&memory).unwrap();
    let opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0).unwrap();
    let common_kv = [0x72; 32];

    let genesis_checkpoint = CheckpointV1 {
        job_id: f.job.job_id,
        token_index: 1,
        layer_index: u16::MAX,
        residual_hash: [0x30; 32],
        global_kv_root: [0x31; 32],
    };
    let middle_checkpoint = CheckpointV1 {
        layer_index: 1,
        residual_hash: [0x32; 32],
        global_kv_root: [0x33; 32],
        ..genesis_checkpoint
    };
    let final_checkpoint = CheckpointV1 {
        layer_index: 3,
        residual_hash: memory_root,
        global_kv_root: common_kv,
        ..genesis_checkpoint
    };
    let checkpoint_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_TOKEN,
        &[
            genesis_checkpoint.hash().unwrap(),
            middle_checkpoint.hash().unwrap(),
            final_checkpoint.hash().unwrap(),
        ],
    )
    .unwrap();

    let prior_final_checkpoint = CheckpointV1 {
        job_id: f.job.job_id,
        token_index: 0,
        layer_index: 3,
        residual_hash: [0x28; 32],
        global_kv_root: genesis_checkpoint.global_kv_root,
    };
    let prior_checkpoint_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_TOKEN,
        &[
            CheckpointV1 {
                layer_index: u16::MAX,
                ..prior_final_checkpoint
            }
            .hash()
            .unwrap(),
            CheckpointV1 {
                layer_index: 1,
                ..prior_final_checkpoint
            }
            .hash()
            .unwrap(),
            prior_final_checkpoint.hash().unwrap(),
        ],
    )
    .unwrap();
    let common_token = TokenResultV1 {
        job_id: f.job.job_id,
        token_index: 0,
        checkpoint_root: prior_checkpoint_tree.root(),
        predicted_token: 41,
        checkpoint_count: 3,
    };
    let challenger_token = TokenResultV1 {
        job_id: f.job.job_id,
        token_index: 1,
        checkpoint_root: checkpoint_tree.root(),
        predicted_token: 9,
        checkpoint_count: 3,
    };
    let miner_token = TokenResultV1 {
        predicted_token: 42,
        ..challenger_token
    };
    let job_genesis = job_genesis_hash_v1(&f.job.job_id).unwrap();
    let common_token_hash = common_token.hash().unwrap();
    let challenger_token_hash = challenger_token.hash().unwrap();
    let miner_token_hash = miner_token.hash().unwrap();
    f.honest_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_JOB,
        &[job_genesis, common_token_hash, challenger_token_hash],
    )
    .unwrap();
    f.miner_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_JOB,
        &[job_genesis, common_token_hash, miner_token_hash],
    )
    .unwrap();
    f.result.trace_root = f.miner_tree.root();
    f.results = MerkleTree::new(
        HashAlgorithm::Sha256V1,
        D_BATCH,
        &[f.result.hash().unwrap(), [99; 32]],
    )
    .unwrap();

    let mut vm = f.vm();
    let (model, epoch, _) = f.seeded_epoch(&mut vm);
    let (dispute, _) = f.open(&vm, epoch, model.clone());
    let (dispute, _) = f.bounds_hashes(&vm, dispute, true, job_genesis, challenger_token_hash);
    let (dispute, _) = f.bounds_hashes(&vm, dispute, false, job_genesis, miner_token_hash);
    let (dispute, _) = f.midpoint_hash(&vm, dispute, true, common_token_hash);
    let (dispute, _) = f.midpoint_hash(&vm, dispute, false, common_token_hash);

    let mut challenger_reveal_body = challenger_token.canonical_bytes().to_vec();
    challenger_reveal_body.extend_from_slice(&common_token.predicted_token.to_le_bytes());
    challenger_reveal_body.extend_from_slice(&common_token.canonical_bytes());
    challenger_reveal_body.extend_from_slice(&prior_final_checkpoint.canonical_bytes());
    let prior_final_proof = prior_checkpoint_tree.proof(2).unwrap();
    challenger_reveal_body.push(prior_final_proof.siblings.len() as u8);
    append_siblings(&mut challenger_reveal_body, &prior_final_proof);
    let challenger_reveal = vm.process_instruction(
        &f.ix(
            25,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &challenger_reveal_body,
        ),
        &[
            (f.model, model.clone()),
            (f.dispute, dispute),
            (f.challenger, Account::default()),
        ],
    );
    assert_ok(&challenger_reveal.program_result);
    let dispute = account(&challenger_reveal.resulting_accounts, f.dispute);
    let miner_reveal = vm.process_instruction(
        &f.ix(
            26,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
            ],
            &miner_token.canonical_bytes(),
        ),
        &[(f.dispute, dispute), (f.miner, Account::default())],
    );
    assert_ok(&miner_reveal.program_result);
    let dispute = account(&miner_reveal.resulting_accounts, f.dispute);
    assert_eq!(dispute.data[4], 16);

    let pre = VmStateV1 {
        job_hash: f.job.hash().unwrap(),
        token_index: 1,
        segment_index: 3,
        program_counter: 0,
        residual_root: memory_root,
        global_kv_root: common_kv,
    };
    let mut challenger_memory = memory.clone();
    challenger_memory[16..20].copy_from_slice(&7.0_f32.to_bits().to_le_bytes());
    challenger_memory[20..24].copy_from_slice(&9_u32.to_le_bytes());
    let challenger_residual = state_hash_v1(&challenger_memory).unwrap();
    let mut miner_memory = memory.clone();
    miner_memory[16..20].copy_from_slice(&7.0_f32.to_bits().to_le_bytes());
    miner_memory[20..24].copy_from_slice(&42_u32.to_le_bytes());
    let miner_residual = state_hash_v1(&miner_memory).unwrap();
    let challenger_output = VmOutputStateV1 {
        job_hash: pre.job_hash,
        token_index: pre.token_index,
        segment_index: pre.segment_index,
        program_counter: 1,
        residual_root: challenger_residual,
        global_kv_root: common_kv,
        predicted_token: 9,
    };
    let miner_output = VmOutputStateV1 {
        residual_root: miner_residual,
        predicted_token: 42,
        ..challenger_output
    };
    let challenger_operator = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_OPERATOR_TREE,
        &[pre.hash().unwrap(), challenger_output.hash().unwrap()],
    )
    .unwrap();
    let miner_operator = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_OPERATOR_TREE,
        &[pre.hash().unwrap(), miner_output.hash().unwrap()],
    )
    .unwrap();
    let operator_bounds = |tree: &MerkleTree, end: [u8; 32]| {
        let start_proof = tree.proof(0).unwrap();
        let end_proof = tree.proof(1).unwrap();
        let mut body = Vec::new();
        body.extend_from_slice(&tree.root());
        body.extend_from_slice(&pre.hash().unwrap());
        body.extend_from_slice(&end);
        body.push(start_proof.siblings.len() as u8);
        append_siblings(&mut body, &start_proof);
        append_siblings(&mut body, &end_proof);
        body
    };

    let checkpoint_proof = checkpoint_tree.proof(2).unwrap();
    let mut challenger_head_body = final_checkpoint.canonical_bytes().to_vec();
    challenger_head_body.push(checkpoint_proof.siblings.len() as u8);
    append_siblings(&mut challenger_head_body, &checkpoint_proof);
    f.append_segment_program(&mut challenger_head_body, 3);
    challenger_head_body.extend_from_slice(&challenger_residual);
    challenger_head_body.extend_from_slice(&common_kv);
    challenger_head_body.extend_from_slice(&operator_bounds(
        &challenger_operator,
        challenger_output.hash().unwrap(),
    ));
    let challenger_head = vm.process_instruction(
        &f.ix(
            36,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &challenger_head_body,
        ),
        &[
            (f.model, model.clone()),
            (f.dispute, dispute),
            (f.challenger, Account::default()),
        ],
    );
    assert_ok(&challenger_head.program_result);
    let dispute = account(&challenger_head.resulting_accounts, f.dispute);

    let mut miner_head_body = Vec::new();
    miner_head_body.extend_from_slice(&miner_residual);
    miner_head_body.extend_from_slice(&common_kv);
    miner_head_body.extend_from_slice(&operator_bounds(
        &miner_operator,
        miner_output.hash().unwrap(),
    ));
    let miner_head = vm.process_instruction(
        &f.ix(
            37,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.miner, true),
            ],
            &miner_head_body,
        ),
        &[
            (f.model, model.clone()),
            (f.dispute, dispute),
            (f.miner, Account::default()),
        ],
    );
    assert_ok(&miner_head.program_result);
    let dispute = account(&miner_head.resulting_accounts, f.dispute);
    assert_eq!(dispute.data[4], 5);

    let instruction_tree = MerkleTree::new(
        HashAlgorithm::Blake3V1,
        D_PROGRAM_TREE,
        &[instruction.hash(0).unwrap()],
    )
    .unwrap();
    let instruction_proof = instruction_tree.proof(0).unwrap();
    let mut terminal_body = Vec::new();
    terminal_body.extend_from_slice(&pre.segment_index.to_le_bytes());
    terminal_body.extend_from_slice(&pre.program_counter.to_le_bytes());
    terminal_body.extend_from_slice(&pre.residual_root);
    terminal_body.extend_from_slice(&pre.global_kv_root);
    terminal_body.extend_from_slice(&instruction.canonical_bytes());
    terminal_body.extend_from_slice(&1_u32.to_le_bytes());
    terminal_body.push(instruction_proof.siblings.len() as u8);
    append_siblings(&mut terminal_body, &instruction_proof);
    terminal_body.extend_from_slice(&opening.byte_length.to_le_bytes());
    terminal_body.extend_from_slice(&opening.chunk_index.to_le_bytes());
    terminal_body.extend_from_slice(&(opening.chunk.len() as u16).to_le_bytes());
    terminal_body.extend_from_slice(&opening.chunk);
    terminal_body.push(opening.siblings.len() as u8);
    for sibling in &opening.siblings {
        terminal_body.extend_from_slice(sibling);
    }
    let resolved = vm.process_instruction(
        &f.ix(
            35,
            vec![
                AccountMeta::new_readonly(f.model, false),
                AccountMeta::new(f.dispute, false),
            ],
            &terminal_body,
        ),
        &[(f.model, model), (f.dispute, dispute)],
    );
    assert_ok(&resolved.program_result);
    let dispute = account(&resolved.resulting_accounts, f.dispute);
    assert_eq!((dispute.data[4], dispute.data[5]), (8, 2));
    assert_eq!(u32_at(&dispute.data, 436), 9);
    println!(
        "v2_final_head_boundary_cu={}",
        challenger_head.compute_units_consumed + miner_head.compute_units_consumed
    );
    println!(
        "v2_final_head_argmax_cu={}",
        resolved.compute_units_consumed
    );
}

#[test]
fn sbf_rejects_result_trace_length_inconsistent_with_job_shape() {
    let mut f = Fixture::new();
    f.result.trace_length = 4;
    f.results = MerkleTree::new(
        HashAlgorithm::Sha256V1,
        D_BATCH,
        &[f.result.hash().unwrap(), [99; 32]],
    )
    .unwrap();
    let mut vm = f.vm();
    let (model, epoch, _) = f.seeded_epoch(&mut vm);
    let opened = vm.process_transaction_instructions(
        &[f.job_signature_ix(), f.open_ix()],
        &[
            (f.epoch, epoch),
            (f.model, model),
            (f.dispute, owned(DISPUTE_BYTES, f.program)),
            (f.challenger, Account::default()),
            ed25519_program_account(),
        ],
    );
    assert!(opened.program_result.is_err());
}

struct Fixture {
    program: Pubkey,
    model: Pubkey,
    epoch: Pubkey,
    dispute: Pubkey,
    staged: Pubkey,
    requester_signing: SigningKey,
    miner: Pubkey,
    challenger: Pubkey,
    entropy: [u8; 32],
    job: JobPayloadV1,
    result: ResultLeafV1,
    results: MerkleTree,
    honest_states: Vec<FmaStateV1>,
    miner_states: Vec<FmaStateV1>,
    honest_tree: MerkleTree,
    miner_tree: MerkleTree,
    activation: qx_commit::StateChunkProofV1,
    weight: WeightLeafV1,
    weight_index: u32,
    weight_siblings: Vec<[u8; 32]>,
    weight_root: [u8; 32],
    verification_instruction: VmInstructionV1,
}

impl Fixture {
    fn new() -> Self {
        Self::with_audit_probability(1_000_000)
    }

    fn with_audit_probability(audit_probability_ppm: u32) -> Self {
        let program = Pubkey::new_unique();
        let model = Pubkey::new_unique();
        let dispute = Pubkey::new_unique();
        let staged = Pubkey::new_unique();
        let requester_signing = SigningKey::from_bytes(&[23; 32]);
        let requester = Pubkey::new_from_array(requester_signing.verifying_key().to_bytes());
        let miner = Pubkey::new_unique();
        let epoch = Pubkey::new_unique();
        let challenger = Pubkey::new_unique();
        let mut activation_bytes = vec![0; 4096];
        activation_bytes[136..140].copy_from_slice(&2.0_f32.to_bits().to_le_bytes());
        let activation_root = activation_hash_v1(&activation_bytes).unwrap();
        let activation =
            prove_state_chunk_v1(StateTreeKindV1::Activation, &activation_bytes, 1).unwrap();
        let job = JobPayloadV1 {
            protocol_version: 1,
            protocol_instance: program.to_bytes(),
            job_id: [1; 32],
            requester: requester.to_bytes(),
            assigned_miner: miner.to_bytes(),
            model_id: [2; 32],
            execution_spec: [3; 32],
            input_hash: [4; 32],
            input_token_count: 1,
            max_output_tokens: 16,
            payment: 1_000,
            slashable_exposure: 2_000,
            deadline_slot: 2_000,
            nonce: 1,
            audit_probability_ppm,
            assigned_epoch: 7,
            model_registry: model.to_bytes(),
        };
        let job_hash = job.hash().unwrap();
        let state = |column, value: f32| FmaStateV1 {
            job_hash,
            tensor_id: 4,
            row: 9,
            column,
            activation_root,
            accumulator_bits: value.to_bits(),
        };
        let honest_states = vec![state(33, 0.5), state(34, 1.0), state(35, 2.0)];
        let miner_states = vec![state(33, 0.5), state(34, 1.0), state(35, 3.0)];
        let make_tree = |states: &[FmaStateV1]| {
            MerkleTree::new(
                HashAlgorithm::Blake3V1,
                D_OPERATOR_TREE,
                &states
                    .iter()
                    .map(FmaStateV1::hash)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap(),
            )
            .unwrap()
        };
        let honest_tree = make_tree(&honest_states);
        let miner_tree = make_tree(&miner_states);
        let result = ResultLeafV1 {
            job_hash,
            output_hash: [5; 32],
            trace_root: miner_tree.root(),
            output_token_count: 2,
            trace_length: 3,
        };
        let results = MerkleTree::new(
            HashAlgorithm::Sha256V1,
            D_BATCH,
            &[result.hash().unwrap(), [99; 32]],
        )
        .unwrap();
        let mut packed = [0x77; 16];
        packed[1] = 0x78;
        let weight = WeightLeafV1 {
            tensor_id: 4,
            row: 9,
            group: 1,
            quantization_type: 4,
            packed_values: packed.to_vec(),
            scale_bits: f16::from_f32(0.5).to_bits(),
        };
        let weight_index = 123_457;
        let weight_siblings = (0..26).map(|n| [n + 1; 32]).collect::<Vec<_>>();
        let mut weight_root = weight.hash(ProtocolVersion::V1).unwrap();
        let mut pos = weight_index;
        for sibling in &weight_siblings {
            weight_root = if pos & 1 == 0 {
                hash_domain(
                    HashAlgorithm::Sha256V1,
                    D_MERKLE,
                    &[D_WEIGHT_TREE, &weight_root, sibling],
                )
                .unwrap()
            } else {
                hash_domain(
                    HashAlgorithm::Sha256V1,
                    D_MERKLE,
                    &[D_WEIGHT_TREE, sibling, &weight_root],
                )
                .unwrap()
            };
            pos >>= 1;
        }
        let verification_instruction = VmInstructionV1 {
            opcode: 2,
            source_a: 0,
            source_b: 4,
            source_c: 8,
            destination: 12,
        };
        Self {
            program,
            model,
            epoch,
            dispute,
            staged,
            requester_signing,
            miner,
            challenger,
            entropy: [7; 32],
            job,
            result,
            results,
            honest_states,
            miner_states,
            honest_tree,
            miner_tree,
            activation,
            weight,
            weight_index,
            weight_siblings,
            weight_root,
            verification_instruction,
        }
    }

    fn vm(&self) -> Mollusk {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/deploy/qx_terminal_verifier");
        Mollusk::new(&self.program, path.to_str().unwrap())
    }

    fn seeded_epoch(&self, vm: &mut Mollusk) -> (Account, Account, u64) {
        vm.sysvars.clock.slot = 100;
        let (model, epoch, _) = self.register(vm);
        vm.sysvars.clock.slot = 102;
        // Mainnet keeps the slot-hashes sysvar at its full fixed capacity. The
        // zero-copy syscall deliberately fetches that full buffer, so mirror
        // the production layout instead of a one-entry test sysvar.
        let mut slot_hashes = (0..solana_slot_hashes::MAX_ENTRIES as u64)
            .map(|slot| (slot, Hash::default()))
            .collect::<Vec<_>>();
        slot_hashes[101] = (101, Hash::new_from_array(self.entropy));
        vm.sysvars.slot_hashes = solana_slot_hashes::SlotHashes::new(&slot_hashes);
        let finalized = vm.process_instruction(
            &self.ix(10, vec![AccountMeta::new(self.epoch, false)], &[]),
            &[(self.epoch, epoch)],
        );
        assert_ok(&finalized.program_result);
        (
            model,
            account(&finalized.resulting_accounts, self.epoch),
            finalized.compute_units_consumed,
        )
    }

    fn ix(&self, op: u8, accounts: Vec<AccountMeta>, body: &[u8]) -> Instruction {
        let mut data = vec![op];
        data.extend_from_slice(body);
        Instruction::new_with_bytes(self.program, &data, accounts)
    }

    fn verification_program(&self) -> MerkleTree {
        vm_program_tree_v1(&[self.verification_instruction]).unwrap()
    }

    fn verification_schedule(&self) -> (MerkleTree, Vec<VmSegmentProgramV1>) {
        let program = self.verification_program();
        let segments = (0..4)
            .map(|segment_index| VmSegmentProgramV1 {
                segment_index,
                instruction_root: program.root(),
                instruction_count: 1,
            })
            .collect::<Vec<_>>();
        let schedule = vm_segment_schedule_v1(&segments).unwrap();
        (schedule, segments)
    }

    fn append_segment_program(&self, body: &mut Vec<u8>, segment_index: usize) {
        let (schedule, segments) = self.verification_schedule();
        let proof = schedule.proof(segment_index).unwrap();
        body.extend_from_slice(&segments[segment_index].canonical_bytes());
        body.push(proof.siblings.len() as u8);
        append_siblings(body, &proof);
    }

    fn register(&self, vm: &Mollusk) -> (Account, Account, Account) {
        let (verification_schedule, segments) = self.verification_schedule();
        let mut model_body = Vec::new();
        model_body.extend_from_slice(&self.job.model_id);
        model_body.extend_from_slice(&self.weight_root);
        model_body.extend_from_slice(&WEIGHT_LEAVES.to_le_bytes());
        model_body.extend_from_slice(&self.job.execution_spec);
        model_body.extend_from_slice(&verification_schedule.root());
        model_body.extend_from_slice(&(segments.len() as u32).to_le_bytes());
        model_body.extend_from_slice(&3_u32.to_le_bytes());
        model_body.extend_from_slice(&4_u16.to_le_bytes());
        model_body.extend_from_slice(&2_u16.to_le_bytes());
        model_body.extend_from_slice(&4096_u32.to_le_bytes());
        model_body.extend_from_slice(&49_155_u32.to_le_bytes());
        let r = vm.process_instruction(
            &self.ix(6, vec![AccountMeta::new(self.model, true)], &model_body),
            &[(self.model, owned(MODEL_BYTES, self.program))],
        );
        assert_ok(&r.program_result);
        let model = account(&r.resulting_accounts, self.model);
        let mut body = Vec::new();
        body.extend_from_slice(&7_u64.to_le_bytes());
        body.extend_from_slice(&self.results.root());
        body.extend_from_slice(&2_u32.to_le_bytes());
        let r = vm.process_instruction(
            &self.ix(
                9,
                vec![
                    AccountMeta::new(self.epoch, true),
                    AccountMeta::new(self.miner, true),
                ],
                &body,
            ),
            &[
                (self.epoch, owned(EPOCH_BYTES, self.program)),
                (self.miner, owned(MINER_BYTES, self.program)),
            ],
        );
        assert_ok(&r.program_result);
        (
            model,
            account(&r.resulting_accounts, self.epoch),
            account(&r.resulting_accounts, self.miner),
        )
    }

    fn open(&self, vm: &Mollusk, epoch: Account, model: Account) -> (Account, u64) {
        let opened = vm.process_transaction_instructions(
            &[self.job_signature_ix(), self.open_ix()],
            &[
                (self.epoch, epoch),
                (self.model, model),
                (self.dispute, owned(DISPUTE_BYTES, self.program)),
                (self.challenger, Account::default()),
                ed25519_program_account(),
            ],
        );
        assert!(opened.program_result.is_ok(), "{:?}", opened.program_result);
        (
            opened.get_account(&self.dispute).unwrap().clone(),
            opened.compute_units_consumed,
        )
    }

    fn job_signature_ix(&self) -> Instruction {
        let job_hash = self.job.hash().unwrap();
        let signature = self.requester_signing.sign(&job_hash).to_bytes();
        solana_ed25519_program::new_ed25519_instruction_with_signature(
            &job_hash,
            &signature,
            &self.requester_signing.verifying_key().to_bytes(),
        )
    }

    fn open_ix(&self) -> Instruction {
        let proof = self.results.proof(0).unwrap();
        let mut body = self.job.canonical_bytes().unwrap();
        body.extend_from_slice(&self.result.canonical_bytes());
        append_proof(&mut body, &proof);
        body.extend_from_slice(&self.honest_tree.root());
        self.ix(
            11,
            vec![
                AccountMeta::new_readonly(self.epoch, false),
                AccountMeta::new_readonly(self.model, false),
                AccountMeta::new(self.dispute, true),
                AccountMeta::new_readonly(self.challenger, true),
                AccountMeta::new_readonly(solana_sdk_ids::sysvar::instructions::ID, false),
            ],
            &body,
        )
    }

    fn bounds(&self, vm: &Mollusk, dispute: Account, honest: bool) -> (Account, u64) {
        let (start, end) = if honest {
            (
                self.honest_states[0].hash().unwrap(),
                self.honest_states[2].hash().unwrap(),
            )
        } else {
            (
                self.miner_states[0].hash().unwrap(),
                self.miner_states[2].hash().unwrap(),
            )
        };
        self.bounds_hashes(vm, dispute, honest, start, end)
    }

    fn bounds_hashes(
        &self,
        vm: &Mollusk,
        dispute: Account,
        honest: bool,
        start_hash: [u8; 32],
        end_hash: [u8; 32],
    ) -> (Account, u64) {
        let tree = if honest {
            &self.honest_tree
        } else {
            &self.miner_tree
        };
        self.bounds_tree(vm, dispute, honest, tree, start_hash, end_hash)
    }

    fn bounds_tree(
        &self,
        vm: &Mollusk,
        dispute: Account,
        honest: bool,
        tree: &MerkleTree,
        start_hash: [u8; 32],
        end_hash: [u8; 32],
    ) -> (Account, u64) {
        let (op, party) = if honest {
            (12, self.challenger)
        } else {
            (13, self.miner)
        };
        let start = tree.proof(0).unwrap();
        let end = tree.proof(2).unwrap();
        let mut body = Vec::new();
        body.extend_from_slice(&start_hash);
        body.extend_from_slice(&end_hash);
        body.push(start.siblings.len() as u8);
        append_siblings(&mut body, &start);
        append_siblings(&mut body, &end);
        let r = vm.process_instruction(
            &self.ix(
                op,
                vec![
                    AccountMeta::new(self.dispute, false),
                    AccountMeta::new_readonly(party, true),
                ],
                &body,
            ),
            &[(self.dispute, dispute), (party, Account::default())],
        );
        assert_ok(&r.program_result);
        (
            account(&r.resulting_accounts, self.dispute),
            r.compute_units_consumed,
        )
    }

    fn midpoint(&self, vm: &Mollusk, dispute: Account, honest: bool) -> (Account, u64) {
        let claim = if honest {
            self.honest_states[1].hash().unwrap()
        } else {
            self.miner_states[1].hash().unwrap()
        };
        self.midpoint_hash(vm, dispute, honest, claim)
    }

    fn midpoint_hash(
        &self,
        vm: &Mollusk,
        dispute: Account,
        honest: bool,
        claim: [u8; 32],
    ) -> (Account, u64) {
        let tree = if honest {
            &self.honest_tree
        } else {
            &self.miner_tree
        };
        self.midpoint_tree(vm, dispute, honest, tree, claim)
    }

    fn midpoint_tree(
        &self,
        vm: &Mollusk,
        dispute: Account,
        honest: bool,
        tree: &MerkleTree,
        claim: [u8; 32],
    ) -> (Account, u64) {
        let (op, party) = if honest {
            (14, self.challenger)
        } else {
            (15, self.miner)
        };
        let proof = tree.proof(1).unwrap();
        let mut body = claim.to_vec();
        body.push(proof.siblings.len() as u8);
        append_siblings(&mut body, &proof);
        let r = vm.process_instruction(
            &self.ix(
                op,
                vec![
                    AccountMeta::new(self.dispute, false),
                    AccountMeta::new_readonly(party, true),
                ],
                &body,
            ),
            &[(self.dispute, dispute), (party, Account::default())],
        );
        assert_ok(&r.program_result);
        (
            account(&r.resulting_accounts, self.dispute),
            r.compute_units_consumed,
        )
    }

    fn challenger_terminal_ix(&self) -> Instruction {
        let start = self.honest_states[1];
        let mut body = Vec::new();
        body.extend_from_slice(&start.tensor_id.to_le_bytes());
        body.extend_from_slice(&start.row.to_le_bytes());
        body.extend_from_slice(&start.column.to_le_bytes());
        body.extend_from_slice(&start.activation_root);
        body.extend_from_slice(&start.accumulator_bits.to_le_bytes());
        body.extend_from_slice(&self.honest_states[2].accumulator_bits.to_le_bytes());
        self.ix(
            16,
            vec![
                AccountMeta::new(self.dispute, false),
                AccountMeta::new_readonly(self.challenger, true),
            ],
            &body,
        )
    }

    fn stage_ix(&self) -> Instruction {
        let mut body = Vec::new();
        body.extend_from_slice(&self.weight.tensor_id.to_le_bytes());
        body.extend_from_slice(&self.weight.row.to_le_bytes());
        body.extend_from_slice(&34_u32.to_le_bytes());
        body.extend_from_slice(&self.weight_index.to_le_bytes());
        body.extend_from_slice(&self.weight.group.to_le_bytes());
        body.push(4);
        body.extend_from_slice(&self.weight.scale_bits.to_le_bytes());
        body.extend_from_slice(&self.weight.packed_values);
        body.push(self.weight_siblings.len() as u8);
        for sibling in &self.weight_siblings {
            body.extend_from_slice(sibling);
        }
        self.ix(
            18,
            vec![
                AccountMeta::new_readonly(self.model, false),
                AccountMeta::new_readonly(self.dispute, false),
                AccountMeta::new(self.staged, false),
            ],
            &body,
        )
    }

    fn resolve_ix(&self) -> Instruction {
        let mut body = Vec::new();
        body.extend_from_slice(&self.activation.byte_length.to_le_bytes());
        body.extend_from_slice(&self.activation.chunk_index.to_le_bytes());
        body.extend_from_slice(&(self.activation.chunk.len() as u16).to_le_bytes());
        body.extend_from_slice(&self.activation.chunk);
        body.push(self.activation.siblings.len() as u8);
        for sibling in &self.activation.siblings {
            body.extend_from_slice(sibling);
        }
        self.ix(
            19,
            vec![
                AccountMeta::new(self.dispute, false),
                AccountMeta::new_readonly(self.staged, false),
            ],
            &body,
        )
    }
}

fn append_proof(body: &mut Vec<u8>, proof: &MerkleProof) {
    body.extend_from_slice(&proof.leaf_index.to_le_bytes());
    body.extend_from_slice(&proof.leaf_count.to_le_bytes());
    body.push(proof.siblings.len() as u8);
    append_siblings(body, proof);
}

fn scalar_state_hash(
    job_hash: [u8; 32],
    transition: u32,
    op: u8,
    a: u32,
    b: u32,
    c: u32,
    result: Option<u32>,
) -> [u8; 32] {
    let phase = u8::from(result.is_some());
    let result = result.unwrap_or(0);
    hash_domain(
        HashAlgorithm::Blake3V1,
        b"QX:SCALAR:STATE:V1",
        &[
            &job_hash,
            &transition.to_le_bytes(),
            &[op],
            &[phase],
            &a.to_le_bytes(),
            &b.to_le_bytes(),
            &c.to_le_bytes(),
            &result.to_le_bytes(),
        ],
    )
    .unwrap()
}

fn append_siblings(body: &mut Vec<u8>, proof: &MerkleProof) {
    for sibling in &proof.siblings {
        body.extend_from_slice(sibling);
    }
}

fn bind_segment_program(dispute: &mut Account, program: &MerkleTree) {
    dispute.data[728..760].copy_from_slice(&program.root());
    dispute.data[760..764].copy_from_slice(&program.proof(0).unwrap().leaf_count.to_le_bytes());
}

fn owned(size: usize, owner: Pubkey) -> Account {
    Account {
        data: vec![0; size],
        owner,
        ..Account::default()
    }
}

fn ed25519_program_account() -> (Pubkey, Account) {
    (
        solana_sdk_ids::ed25519_program::ID,
        Account {
            lamports: 1,
            owner: solana_sdk_ids::native_loader::ID,
            executable: true,
            ..Account::default()
        },
    )
}

fn account(accounts: &[(Pubkey, Account)], key: Pubkey) -> Account {
    accounts
        .iter()
        .find(|(candidate, _)| *candidate == key)
        .unwrap()
        .1
        .clone()
}

fn u32_at(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

#[track_caller]
fn assert_ok(result: &mollusk_svm::result::ProgramResult) {
    assert!(result.is_ok(), "{result:?}");
}
