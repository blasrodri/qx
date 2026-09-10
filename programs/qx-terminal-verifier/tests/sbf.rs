use half::f16;
use mollusk_svm::Mollusk;
use qx_commit::{
    activation_hash_v1, hash_domain, prove_state_chunk_v1, HashAlgorithm, ProtocolVersion,
    StateTreeKindV1, WeightLeafV1, D_MERKLE, D_WEIGHT_TREE,
};
use solana_account::Account;
use solana_instruction::{AccountMeta, Instruction};
use solana_pubkey::Pubkey;

const PRODUCTION_WEIGHT_LEAVES: u32 = 41_755_744;

#[test]
fn benchmark_production_depth_staged_terminal_witness() {
    let witness = terminal_instructions();
    let program_id = Pubkey::new_unique();
    let model_key = Pubkey::new_unique();
    let staged_key = Pubkey::new_unique();
    let program_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/deploy/qx_terminal_verifier");
    let mollusk = Mollusk::new(&program_id, program_path.to_str().unwrap());
    let model_account = Account {
        lamports: 1_000_000,
        data: witness.model_account,
        owner: program_id,
        executable: false,
        rent_epoch: 0,
    };
    let staged_account = Account {
        lamports: 1_000_000,
        data: vec![0; 20],
        owner: program_id,
        executable: false,
        rent_epoch: 0,
    };
    let stage_instruction = Instruction::new_with_bytes(
        program_id,
        &witness.stage_weight,
        vec![
            AccountMeta::new_readonly(model_key, false),
            AccountMeta::new(staged_key, false),
        ],
    );
    let stage_result = mollusk.process_instruction(
        &stage_instruction,
        &[(model_key, model_account), (staged_key, staged_account)],
    );
    assert!(
        stage_result.program_result.is_ok(),
        "{:?}",
        stage_result.program_result
    );
    let staged_account = stage_result
        .resulting_accounts
        .iter()
        .find(|(key, _)| key == &staged_key)
        .unwrap()
        .1
        .clone();

    let terminal_instruction = Instruction::new_with_bytes(
        program_id,
        &witness.verify_fma,
        vec![AccountMeta::new_readonly(staged_key, false)],
    );
    let terminal_result = mollusk.process_instruction(
        &terminal_instruction,
        &[(staged_key, staged_account.clone())],
    );
    assert!(
        terminal_result.program_result.is_ok(),
        "{:?}",
        terminal_result.program_result
    );
    println!(
        "stage_weight_instruction_bytes={}",
        witness.stage_weight.len()
    );
    println!(
        "stage_weight_compute_units={}",
        stage_result.compute_units_consumed
    );
    println!(
        "terminal_fma_instruction_bytes={}",
        witness.verify_fma.len()
    );
    println!(
        "terminal_fma_compute_units={}",
        terminal_result.compute_units_consumed
    );
    println!(
        "dispute_compute_units_total={}",
        stage_result.compute_units_consumed + terminal_result.compute_units_consumed
    );
    println!("account_data_touched_bytes={}", 36 + 20);

    let bad_instruction = Instruction::new_with_bytes(
        program_id,
        &witness.bad_fma,
        vec![AccountMeta::new_readonly(staged_key, false)],
    );
    let bad_result = mollusk.process_instruction(&bad_instruction, &[(staged_key, staged_account)]);
    assert!(bad_result.program_result.is_err());
    println!(
        "rejected_arithmetic_compute_units={}",
        bad_result.compute_units_consumed
    );
}

struct TerminalInstructions {
    model_account: Vec<u8>,
    stage_weight: Vec<u8>,
    verify_fma: Vec<u8>,
    bad_fma: Vec<u8>,
}

fn terminal_instructions() -> TerminalInstructions {
    let tensor_id = 4_u32;
    let row = 9_u32;
    let column = 33_u32;
    let mut packed = [0x77_u8; 16];
    packed[0] = 0x87; // Column 33 is local nibble 1: code 8 => +1.
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
    let mut model_account = Vec::with_capacity(36);
    model_account.extend_from_slice(&weight_root);
    model_account.extend_from_slice(&PRODUCTION_WEIGHT_LEAVES.to_le_bytes());

    let mut stage_weight = Vec::new();
    stage_weight.push(1);
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
    let mut verify_fma = Vec::new();
    verify_fma.push(2);
    verify_fma.extend_from_slice(&activation_root);
    verify_fma.extend_from_slice(&3.0_f32.to_bits().to_le_bytes());
    verify_fma.extend_from_slice(&4.0_f32.to_bits().to_le_bytes());
    verify_fma.extend_from_slice(&activation.byte_length.to_le_bytes());
    verify_fma.extend_from_slice(&activation.chunk_index.to_le_bytes());
    verify_fma.extend_from_slice(&(activation.chunk.len() as u16).to_le_bytes());
    verify_fma.extend_from_slice(&activation.chunk);
    verify_fma.push(activation.siblings.len() as u8);
    for sibling in &activation.siblings {
        verify_fma.extend_from_slice(sibling);
    }
    let mut bad_fma = verify_fma.clone();
    // Discriminator + activation root + accumulator start.
    let claimed_end_offset = 1 + 32 + 4;
    bad_fma[claimed_end_offset..claimed_end_offset + 4]
        .copy_from_slice(&5.0_f32.to_bits().to_le_bytes());
    TerminalInstructions {
        model_account,
        stage_weight,
        verify_fma,
        bad_fma,
    }
}
