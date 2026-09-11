use ed25519_dalek::{Signer, SigningKey};
use half::f16;
use mollusk_svm::Mollusk;
use qx_commit::{
    activation_hash_v1, hash_domain, prove_state_chunk_v1, HashAlgorithm, InferenceReceiptV1,
    MerkleTree, ProtocolVersion, StateTreeKindV1, WeightLeafV1, D_BATCH, D_MERKLE, D_WEIGHT_TREE,
};
use solana_account::Account;
use solana_instruction::{AccountMeta, Instruction};
use solana_pubkey::Pubkey;

const PRODUCTION_WEIGHT_LEAVES: u32 = 41_755_744;
const DISPUTE_BYTES: usize = 304;
const STAGED_BYTES: usize = 52;

#[test]
fn signed_epoch_receipt_reaches_objective_sbf_terminal_resolution() {
    let fixture = Fixture::new();
    let mut mollusk = fixture.mollusk();
    mollusk.sysvars.clock.slot = 100;
    let (epoch_account, model_account, registration_cu) = fixture.register_accounts(&mollusk);

    let open = mollusk.process_transaction_instructions(
        &[fixture.signature_instruction(), fixture.open_instruction()],
        &fixture.open_accounts(epoch_account, model_account.clone()),
    );
    assert!(open.program_result.is_ok(), "{:?}", open.program_result);
    let dispute = open.get_account(&fixture.dispute_key).unwrap().clone();
    assert_eq!(&dispute.data[..4], b"QXD1");
    assert_eq!(dispute.data[4], 1);

    let agree = mollusk.process_instruction(
        &fixture.agree_instruction(5.0, 4.0),
        &[
            (fixture.dispute_key, dispute),
            (fixture.miner_key(), Account::default()),
            (fixture.challenger_key, Account::default()),
        ],
    );
    assert!(agree.program_result.is_ok(), "{:?}", agree.program_result);
    let dispute = account_from(&agree.resulting_accounts, fixture.dispute_key);

    let stage = mollusk.process_instruction(
        &fixture.stage_instruction(),
        &[
            (fixture.model_key, model_account),
            (fixture.dispute_key, dispute.clone()),
            (fixture.staged_key, fixture.staged_account()),
        ],
    );
    assert!(stage.program_result.is_ok(), "{:?}", stage.program_result);
    let staged = account_from(&stage.resulting_accounts, fixture.staged_key);

    // A weight staged for one dispute cannot be replayed into another.
    let other_dispute_key = Pubkey::new_unique();
    let replay = mollusk.process_instruction(
        &Instruction::new_with_bytes(
            fixture.program_id,
            &fixture.verify_fma,
            vec![
                AccountMeta::new(other_dispute_key, false),
                AccountMeta::new_readonly(fixture.staged_key, false),
            ],
        ),
        &[
            (other_dispute_key, dispute.clone()),
            (fixture.staged_key, staged.clone()),
        ],
    );
    assert!(replay.program_result.is_err());

    let terminal = mollusk.process_instruction(
        &fixture.verify_instruction(&fixture.verify_fma),
        &[(fixture.dispute_key, dispute), (fixture.staged_key, staged)],
    );
    assert!(
        terminal.program_result.is_ok(),
        "{:?}",
        terminal.program_result
    );
    let resolved = account_from(&terminal.resulting_accounts, fixture.dispute_key);
    assert_eq!(resolved.data[4], 3);
    assert_eq!(
        resolved.data[5], 2,
        "challenger must win the false miner FMA"
    );

    println!("open_receipt_compute_units={}", open.compute_units_consumed);
    println!("registration_compute_units={registration_cu}");
    println!(
        "agree_terminal_compute_units={}",
        agree.compute_units_consumed
    );
    println!(
        "stage_weight_instruction_bytes={}",
        fixture.stage_weight.len()
    );
    println!(
        "stage_weight_compute_units={}",
        stage.compute_units_consumed
    );
    println!(
        "terminal_fma_instruction_bytes={}",
        fixture.verify_fma.len()
    );
    println!(
        "terminal_fma_compute_units={}",
        terminal.compute_units_consumed
    );
    println!(
        "bound_dispute_compute_units_total={}",
        open.compute_units_consumed
            + agree.compute_units_consumed
            + stage.compute_units_consumed
            + terminal.compute_units_consumed
    );
    println!("persistent_dispute_account_bytes={DISPUTE_BYTES}");
    println!("temporary_staged_account_bytes={STAGED_BYTES}");
}

#[test]
fn dishonest_challenger_and_bad_activation_lose() {
    let fixture = Fixture::new();
    let mut mollusk = fixture.mollusk();
    mollusk.sysvars.clock.slot = 100;
    let (epoch_account, model_account, _) = fixture.register_accounts(&mollusk);
    let open = mollusk.process_transaction_instructions(
        &[fixture.signature_instruction(), fixture.open_instruction()],
        &fixture.open_accounts(epoch_account, model_account.clone()),
    );
    assert!(open.program_result.is_ok(), "{:?}", open.program_result);
    let dispute = open.get_account(&fixture.dispute_key).unwrap().clone();
    let agree = mollusk.process_instruction(
        &fixture.agree_instruction(4.0, 5.0),
        &[
            (fixture.dispute_key, dispute),
            (fixture.miner_key(), Account::default()),
            (fixture.challenger_key, Account::default()),
        ],
    );
    assert!(agree.program_result.is_ok(), "{:?}", agree.program_result);
    let dispute = account_from(&agree.resulting_accounts, fixture.dispute_key);
    let stage = mollusk.process_instruction(
        &fixture.stage_instruction(),
        &[
            (fixture.model_key, model_account),
            (fixture.dispute_key, dispute.clone()),
            (fixture.staged_key, fixture.staged_account()),
        ],
    );
    assert!(stage.program_result.is_ok(), "{:?}", stage.program_result);
    let staged = account_from(&stage.resulting_accounts, fixture.staged_key);

    let mut corrupt = fixture.verify_fma.clone();
    // discriminator + byte length + chunk index + chunk length
    corrupt[11] ^= 1;
    let bad = mollusk.process_instruction(
        &fixture.verify_instruction(&corrupt),
        &[
            (fixture.dispute_key, dispute.clone()),
            (fixture.staged_key, staged.clone()),
        ],
    );
    assert!(bad.program_result.is_err());

    let honest = mollusk.process_instruction(
        &fixture.verify_instruction(&fixture.verify_fma),
        &[(fixture.dispute_key, dispute), (fixture.staged_key, staged)],
    );
    assert!(honest.program_result.is_ok(), "{:?}", honest.program_result);
    let resolved = account_from(&honest.resulting_accounts, fixture.dispute_key);
    assert_eq!(resolved.data[5], 1, "miner must defeat a false challenge");
}

#[test]
fn forged_receipt_signature_and_expired_challenge_are_rejected() {
    let fixture = Fixture::new();
    let mut mollusk = fixture.mollusk();
    mollusk.sysvars.clock.slot = 100;
    let (epoch_account, model_account, _) = fixture.register_accounts(&mollusk);

    let forged_message = [88; 32];
    let forged_signature = fixture.signing_key.sign(&forged_message).to_bytes();
    let forged_precompile = solana_ed25519_program::new_ed25519_instruction_with_signature(
        &forged_message,
        &forged_signature,
        &fixture.signing_key.verifying_key().to_bytes(),
    );
    let forged = mollusk.process_transaction_instructions(
        &[forged_precompile, fixture.open_instruction()],
        &fixture.open_accounts(epoch_account.clone(), model_account.clone()),
    );
    assert!(forged.program_result.is_err());

    let open = mollusk.process_transaction_instructions(
        &[fixture.signature_instruction(), fixture.open_instruction()],
        &fixture.open_accounts(epoch_account, model_account),
    );
    assert!(open.program_result.is_ok(), "{:?}", open.program_result);
    let dispute = open.get_account(&fixture.dispute_key).unwrap().clone();
    let early_timeout = mollusk.process_instruction(
        &Instruction::new_with_bytes(
            fixture.program_id,
            &[5],
            vec![AccountMeta::new(fixture.dispute_key, false)],
        ),
        &[(fixture.dispute_key, dispute.clone())],
    );
    assert!(early_timeout.program_result.is_err());

    mollusk.sysvars.clock.slot = 151;
    let timeout = mollusk.process_instruction(
        &Instruction::new_with_bytes(
            fixture.program_id,
            &[5],
            vec![AccountMeta::new(fixture.dispute_key, false)],
        ),
        &[(fixture.dispute_key, dispute)],
    );
    assert!(
        timeout.program_result.is_ok(),
        "{:?}",
        timeout.program_result
    );
    let resolved = account_from(&timeout.resulting_accounts, fixture.dispute_key);
    assert_eq!(resolved.data[4], 3);
    assert_eq!(resolved.data[5], 1);
}

fn account_from(accounts: &[(Pubkey, Account)], key: Pubkey) -> Account {
    accounts
        .iter()
        .find(|(candidate, _)| candidate == &key)
        .unwrap()
        .1
        .clone()
}

struct Fixture {
    program_id: Pubkey,
    epoch_key: Pubkey,
    model_key: Pubkey,
    dispute_key: Pubkey,
    staged_key: Pubkey,
    challenger_key: Pubkey,
    signing_key: SigningKey,
    receipt_bytes: Vec<u8>,
    receipt_hash: [u8; 32],
    epoch_data: Vec<u8>,
    model_data: Vec<u8>,
    receipt_proof: qx_commit::MerkleProof,
    activation_root: [u8; 32],
    stage_weight: Vec<u8>,
    verify_fma: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let program_id = Pubkey::new_unique();
        let epoch_key = Pubkey::new_unique();
        let model_key = Pubkey::new_unique();
        let dispute_key = Pubkey::new_unique();
        let staged_key = Pubkey::new_unique();
        let challenger_key = Pubkey::new_unique();
        let signing_key = SigningKey::from_bytes(&[17; 32]);
        let model_id = [2; 32];
        let receipt = InferenceReceiptV1 {
            protocol_version: 1,
            job_id: [1; 32],
            model_id,
            input_hash: [3; 32],
            output_hash: [4; 32],
            trace_root: [5; 32],
            input_token_count: 8,
            output_token_count: 16,
            miner_pubkey: signing_key.verifying_key().to_bytes(),
            sequence: 9,
        };
        let receipt_bytes = receipt.canonical_bytes().unwrap();
        let receipt_hash = receipt.hash().unwrap();
        let receipt_tree =
            MerkleTree::new(HashAlgorithm::Sha256V1, D_BATCH, &[receipt_hash, [99; 32]]).unwrap();
        let receipt_proof = receipt_tree.proof(0).unwrap();
        let mut epoch_data = Vec::with_capacity(56);
        epoch_data.extend_from_slice(b"QXE1");
        epoch_data.extend_from_slice(&receipt_tree.root());
        epoch_data.extend_from_slice(&2_u32.to_le_bytes());
        epoch_data.extend_from_slice(&1_000_u64.to_le_bytes());
        epoch_data.extend_from_slice(&50_u64.to_le_bytes());
        let tensor_id = 4_u32;
        let row = 9_u32;
        let column = 33_u32;
        let mut packed = [0x77_u8; 16];
        packed[0] = 0x87;
        let scale_bits = f16::from_f32(0.5).to_bits();
        let leaf = WeightLeafV1 {
            tensor_id,
            row,
            group: 1,
            quantization_type: 4,
            packed_values: packed.to_vec(),
            scale_bits,
        };
        let weight_siblings = (0..26)
            .map(|level| [level as u8 + 1; 32])
            .collect::<Vec<_>>();
        let weight_leaf_index = 123_457_u32;
        let mut weight_root = leaf.hash(ProtocolVersion::V1).unwrap();
        let mut position = weight_leaf_index;
        for sibling in &weight_siblings {
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
        let mut model_data = Vec::with_capacity(140);
        model_data.extend_from_slice(b"QXM1");
        model_data.extend_from_slice(&model_id);
        model_data.extend_from_slice(&weight_root);
        model_data.extend_from_slice(&PRODUCTION_WEIGHT_LEAVES.to_le_bytes());
        model_data.extend_from_slice(&[7; 32]);
        model_data.extend_from_slice(&[8; 32]);
        model_data.extend_from_slice(&1_u32.to_le_bytes());

        let mut stage_weight = vec![1];
        stage_weight.extend_from_slice(&column.to_le_bytes());
        stage_weight.extend_from_slice(&weight_leaf_index.to_le_bytes());
        stage_weight.extend_from_slice(&tensor_id.to_le_bytes());
        stage_weight.extend_from_slice(&row.to_le_bytes());
        stage_weight.extend_from_slice(&1_u32.to_le_bytes());
        stage_weight.push(4);
        stage_weight.extend_from_slice(&scale_bits.to_le_bytes());
        stage_weight.extend_from_slice(&packed);
        stage_weight.push(weight_siblings.len() as u8);
        for sibling in &weight_siblings {
            stage_weight.extend_from_slice(sibling);
        }

        let mut activation_bytes = vec![0_u8; 1_024 * 4];
        activation_bytes[column as usize * 4..column as usize * 4 + 4]
            .copy_from_slice(&2.0_f32.to_bits().to_le_bytes());
        let activation_root = activation_hash_v1(&activation_bytes).unwrap();
        let activation = prove_state_chunk_v1(
            StateTreeKindV1::Activation,
            &activation_bytes,
            column * 4 / qx_commit::ACTIVATION_CHUNK_BYTES as u32,
        )
        .unwrap();
        let mut verify_fma = vec![2];
        verify_fma.extend_from_slice(&activation.byte_length.to_le_bytes());
        verify_fma.extend_from_slice(&activation.chunk_index.to_le_bytes());
        verify_fma.extend_from_slice(&(activation.chunk.len() as u16).to_le_bytes());
        verify_fma.extend_from_slice(&activation.chunk);
        verify_fma.push(activation.siblings.len() as u8);
        for sibling in &activation.siblings {
            verify_fma.extend_from_slice(sibling);
        }

        Self {
            program_id,
            epoch_key,
            model_key,
            dispute_key,
            staged_key,
            challenger_key,
            signing_key,
            receipt_bytes,
            receipt_hash,
            epoch_data,
            model_data,
            receipt_proof,
            activation_root,
            stage_weight,
            verify_fma,
        }
    }

    fn mollusk(&self) -> Mollusk {
        let program_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/deploy/qx_terminal_verifier");
        Mollusk::new(&self.program_id, program_path.to_str().unwrap())
    }

    fn miner_key(&self) -> Pubkey {
        Pubkey::new_from_array(self.signing_key.verifying_key().to_bytes())
    }

    fn signature_instruction(&self) -> Instruction {
        let signature = self.signing_key.sign(&self.receipt_hash).to_bytes();
        solana_ed25519_program::new_ed25519_instruction_with_signature(
            &self.receipt_hash,
            &signature,
            &self.signing_key.verifying_key().to_bytes(),
        )
    }

    fn open_instruction(&self) -> Instruction {
        let mut data = vec![3];
        data.extend_from_slice(&self.receipt_bytes);
        data.extend_from_slice(&self.receipt_proof.leaf_index.to_le_bytes());
        data.extend_from_slice(&self.receipt_proof.leaf_count.to_le_bytes());
        data.push(self.receipt_proof.siblings.len() as u8);
        for sibling in &self.receipt_proof.siblings {
            data.extend_from_slice(sibling);
        }
        data.extend_from_slice(&[6; 32]);
        Instruction::new_with_bytes(
            self.program_id,
            &data,
            vec![
                AccountMeta::new_readonly(self.epoch_key, false),
                AccountMeta::new_readonly(self.model_key, false),
                AccountMeta::new(self.dispute_key, false),
                AccountMeta::new_readonly(self.challenger_key, true),
                AccountMeta::new_readonly(solana_sdk_ids::sysvar::instructions::ID, false),
            ],
        )
    }

    fn register_accounts(&self, mollusk: &Mollusk) -> (Account, Account, u64) {
        let mut model_ix_data = vec![6];
        model_ix_data.extend_from_slice(&self.model_data[4..]);
        let model_result = mollusk.process_instruction(
            &Instruction::new_with_bytes(
                self.program_id,
                &model_ix_data,
                vec![AccountMeta::new(self.model_key, true)],
            ),
            &[(self.model_key, owned_account(vec![0; 140], self.program_id))],
        );
        assert!(
            model_result.program_result.is_ok(),
            "{:?}",
            model_result.program_result
        );
        let model = account_from(&model_result.resulting_accounts, self.model_key);

        let mut epoch_ix_data = vec![7];
        epoch_ix_data.extend_from_slice(&self.epoch_data[4..]);
        let epoch_result = mollusk.process_instruction(
            &Instruction::new_with_bytes(
                self.program_id,
                &epoch_ix_data,
                vec![AccountMeta::new(self.epoch_key, true)],
            ),
            &[(self.epoch_key, owned_account(vec![0; 56], self.program_id))],
        );
        assert!(
            epoch_result.program_result.is_ok(),
            "{:?}",
            epoch_result.program_result
        );
        let epoch = account_from(&epoch_result.resulting_accounts, self.epoch_key);
        (
            epoch,
            model,
            model_result.compute_units_consumed + epoch_result.compute_units_consumed,
        )
    }

    fn open_accounts(
        &self,
        epoch_account: Account,
        model_account: Account,
    ) -> Vec<(Pubkey, Account)> {
        vec![
            (self.epoch_key, epoch_account),
            (self.model_key, model_account),
            (
                self.dispute_key,
                owned_account(vec![0; DISPUTE_BYTES], self.program_id),
            ),
            (self.challenger_key, Account::default()),
            (
                solana_sdk_ids::ed25519_program::ID,
                Account {
                    lamports: 1,
                    data: vec![],
                    owner: solana_sdk_ids::native_loader::ID,
                    executable: true,
                    rent_epoch: 0,
                },
            ),
        ]
    }

    fn staged_account(&self) -> Account {
        owned_account(vec![0; STAGED_BYTES], self.program_id)
    }

    fn agree_instruction(&self, defender_end: f32, challenger_end: f32) -> Instruction {
        let mut data = vec![4];
        data.extend_from_slice(&self.activation_root);
        data.extend_from_slice(&4_u32.to_le_bytes());
        data.extend_from_slice(&9_u32.to_le_bytes());
        data.extend_from_slice(&33_u32.to_le_bytes());
        data.extend_from_slice(&3.0_f32.to_bits().to_le_bytes());
        data.extend_from_slice(&defender_end.to_bits().to_le_bytes());
        data.extend_from_slice(&challenger_end.to_bits().to_le_bytes());
        Instruction::new_with_bytes(
            self.program_id,
            &data,
            vec![
                AccountMeta::new(self.dispute_key, false),
                AccountMeta::new_readonly(self.miner_key(), true),
                AccountMeta::new_readonly(self.challenger_key, true),
            ],
        )
    }

    fn stage_instruction(&self) -> Instruction {
        Instruction::new_with_bytes(
            self.program_id,
            &self.stage_weight,
            vec![
                AccountMeta::new_readonly(self.model_key, false),
                AccountMeta::new_readonly(self.dispute_key, false),
                AccountMeta::new(self.staged_key, false),
            ],
        )
    }

    fn verify_instruction(&self, data: &[u8]) -> Instruction {
        Instruction::new_with_bytes(
            self.program_id,
            data,
            vec![
                AccountMeta::new(self.dispute_key, false),
                AccountMeta::new_readonly(self.staged_key, false),
            ],
        )
    }
}

fn owned_account(data: Vec<u8>, owner: Pubkey) -> Account {
    Account {
        lamports: 1_000_000,
        data,
        owner,
        executable: false,
        rent_epoch: 0,
    }
}
