use ed25519_dalek::{Signer, SigningKey};
use half::f16;
use mollusk_svm::Mollusk;
use qx_commit::{
    activation_hash_v1, hash_domain, prove_state_chunk_v1, EpochBlockSeedV2, FmaStateV1,
    HashAlgorithm, JobPayloadV1, MerkleProof, MerkleTree, ProtocolVersion, ResultLeafV1,
    StateTreeKindV1, WeightLeafV1, D_BATCH, D_DISPUTE_TRACE, D_MERKLE, D_WEIGHT_TREE,
};
use solana_account::Account;
use solana_hash::Hash;
use solana_instruction::{AccountMeta, Instruction};
use solana_pubkey::Pubkey;

const EPOCH_BYTES: usize = 152;
const DISPUTE_BYTES: usize = 448;
const STAGED_BYTES: usize = 52;
const MINER_BYTES: usize = 16;
const WEIGHT_LEAVES: u32 = 41_755_744;

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

    let mut dispute = owned(DISPUTE_BYTES, f.program);
    dispute.data[..4].copy_from_slice(b"QXD2");
    dispute.data[4] = 5;
    dispute.data[6] = 2;
    dispute.data[72..104].copy_from_slice(&job_hash);
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
    let challenger = vm.process_instruction(
        &f.ix(
            21,
            vec![
                AccountMeta::new(f.dispute, false),
                AccountMeta::new_readonly(f.challenger, true),
            ],
            &challenger_body,
        ),
        &[(f.dispute, dispute), (f.challenger, Account::default())],
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
    let (dispute, _) = miner_due.open(&vm, epoch, model);
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
    let f = Fixture::with_audit_probability(0);
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
            input_token_count: 8,
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
                D_DISPUTE_TRACE,
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
            output_token_count: 4,
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

    fn register(&self, vm: &Mollusk) -> (Account, Account, Account) {
        let mut model_body = Vec::new();
        model_body.extend_from_slice(&self.job.model_id);
        model_body.extend_from_slice(&self.weight_root);
        model_body.extend_from_slice(&WEIGHT_LEAVES.to_le_bytes());
        let r = vm.process_instruction(
            &self.ix(6, vec![AccountMeta::new(self.model, true)], &model_body),
            &[(self.model, owned(72, self.program))],
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
        let (op, party, states, tree) = if honest {
            (12, self.challenger, &self.honest_states, &self.honest_tree)
        } else {
            (13, self.miner, &self.miner_states, &self.miner_tree)
        };
        let start = tree.proof(0).unwrap();
        let end = tree.proof(2).unwrap();
        let mut body = Vec::new();
        body.extend_from_slice(&states[0].hash().unwrap());
        body.extend_from_slice(&states[2].hash().unwrap());
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
        let (op, party, state, proof) = if honest {
            (
                14,
                self.challenger,
                self.honest_states[1],
                self.honest_tree.proof(1).unwrap(),
            )
        } else {
            (
                15,
                self.miner,
                self.miner_states[1],
                self.miner_tree.proof(1).unwrap(),
            )
        };
        let mut body = state.hash().unwrap().to_vec();
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
