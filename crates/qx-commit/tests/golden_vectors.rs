use qx_commit::{
    CheckpointV1, D_BATCH, D_JOB, D_KV, D_TOKEN, D_WEIGHT_TREE, HashAlgorithm, InferenceReceiptV1,
    MerkleTree, ModelSpecV1, ProtocolVersion, TokenResultV1, WeightLeafV1,
    granite_architecture_root, granite_execution_spec, hash_domain, input_hash_v1, kv_leaf_hash_v1,
    output_hash_v1, state_hash_v1,
};

fn hex(value: [u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn protocol_v1_golden_vectors() {
    let algorithm = HashAlgorithm::Sha256V1;
    let trace_algorithm = HashAlgorithm::Blake3V1;
    let weight_a = WeightLeafV1 {
        tensor_id: 7,
        row: 3,
        group: 2,
        quantization_type: 4,
        packed_values: (0_u8..16).collect(),
        scale_bits: 0x3555,
    }
    .hash(ProtocolVersion::V1)
    .unwrap();
    let weight_b = WeightLeafV1 {
        tensor_id: 7,
        row: 3,
        group: 3,
        quantization_type: 4,
        packed_values: (16_u8..32).collect(),
        scale_bits: 0xb800,
    }
    .hash(ProtocolVersion::V1)
    .unwrap();
    let weight_root = MerkleTree::new(algorithm, D_WEIGHT_TREE, &[weight_a, weight_b])
        .unwrap()
        .root();
    let quantization_spec = hash_domain(
        algorithm,
        b"QX:QUANTIZATION:V1",
        &[b"symmetric-q4", &32_u16.to_le_bytes(), b"fp16-scale"],
    )
    .unwrap();
    let model = ModelSpecV1 {
        model_version: 1,
        architecture_root: granite_architecture_root().unwrap(),
        weights_root: weight_root,
        num_layers: 24,
        hidden_size: 1024,
        num_attention_heads: 16,
        num_kv_heads: 8,
        num_experts: 32,
        experts_per_token: 8,
        vocab_size: 49_155,
        max_context_tokens: 4096,
        quantization_spec,
        execution_spec: granite_execution_spec().unwrap(),
        tokenizer_hash: [0x5a; 32],
    };
    let model_id = model.model_id(ProtocolVersion::V1).unwrap();

    let f32_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>()
    };
    let key = f32_bytes(&[1.0, -2.0, 0.5, -0.0]);
    let value = f32_bytes(&[3.0, 4.0, -5.0, 6.0]);
    let kv_leaf = kv_leaf_hash_v1(2, 9, &key, &value).unwrap();
    let kv_other = hash_domain(algorithm, D_KV, &[b"other-kv-leaf"]).unwrap();
    let kv_root = MerkleTree::new(trace_algorithm, D_KV, &[kv_leaf, kv_other])
        .unwrap()
        .root();
    let checkpoint = CheckpointV1 {
        job_id: [0x11; 32],
        token_index: 9,
        layer_index: 2,
        residual_hash: state_hash_v1(&f32_bytes(&[0.25, -1.5])).unwrap(),
        global_kv_root: kv_root,
    }
    .hash()
    .unwrap();
    let checkpoint_other = hash_domain(algorithm, qx_commit::D_CHECKPOINT, &[b"other"]).unwrap();
    let token_root = MerkleTree::new(trace_algorithm, D_TOKEN, &[checkpoint, checkpoint_other])
        .unwrap()
        .root();
    let token_result = TokenResultV1 {
        job_id: [0x11; 32],
        token_index: 9,
        checkpoint_root: token_root,
        predicted_token: 42,
        checkpoint_count: 2,
    }
    .hash()
    .unwrap();
    let job_other = hash_domain(algorithm, D_TOKEN, &[b"other-token"]).unwrap();
    let job_root = MerkleTree::new(trace_algorithm, D_JOB, &[token_result, job_other])
        .unwrap()
        .root();
    let input_hash = input_hash_v1(&model_id, &[1, 2, 49_154]).unwrap();
    let output_hash = output_hash_v1(&[42, 7]).unwrap();
    let receipt = InferenceReceiptV1 {
        protocol_version: 1,
        job_id: [0x11; 32],
        model_id,
        input_hash,
        output_hash,
        trace_root: job_root,
        input_token_count: 3,
        output_token_count: 2,
        miner_pubkey: [0x22; 32],
        sequence: 17,
    }
    .hash()
    .unwrap();
    let receipt_other = hash_domain(algorithm, qx_commit::D_RECEIPT, &[b"other-receipt"]).unwrap();
    let batch_root = MerkleTree::new(algorithm, D_BATCH, &[receipt, receipt_other])
        .unwrap()
        .root();

    let vectors = [
        (
            "model_id",
            model_id,
            "e361af482a0ee1bd2d8942437bb2fb5175fdd14e014d87967ea1f67dacbf41cb",
        ),
        (
            "weight_leaf",
            weight_a,
            "d8a53e482a1e5fdcdcc70b8d21a93619645ed2fa98178081fd3bf6293cb558a7",
        ),
        (
            "weight_root",
            weight_root,
            "efa0c68195aee790063a52891485a556b83f9b9fdad70e75754ea7cc60b34e59",
        ),
        (
            "kv_leaf",
            kv_leaf,
            "3516c5116950fd510fd4c2c65b63454c29739d9030e3a83e1944391137b8dac1",
        ),
        (
            "kv_root",
            kv_root,
            "ac3cb8d1ef765a4b2e71c4adc61921ab2105650d8a3685e445df7c27afbd3c6c",
        ),
        (
            "checkpoint",
            checkpoint,
            "73eabcfc5497772b03f0473ae0054c66a93b5b83b877ba3e5f5640dfa8b1a826",
        ),
        (
            "token_root",
            token_root,
            "a4fe82012dbe3226056e3292f4c8aaa86ca00056f3200b329f3a4a8fb39f1203",
        ),
        (
            "token_result",
            token_result,
            "ac0f9c304fcee19fb42743cbf0458a778dc706255736cc62f5c84625c50d2d6a",
        ),
        (
            "job_root",
            job_root,
            "ae1dc5b1f6871042973bbb1585da16264b7247a030e6d31e648c51a55040dca2",
        ),
        (
            "input_hash",
            input_hash,
            "4dac0324a4d8a43e7becbcd2935eba1d6c088ae3057fed06cf60d7be70fafaad",
        ),
        (
            "output_hash",
            output_hash,
            "88e599929cdefc6107f75e0fffdcac8e43cd957875740a9a60fd93daefc479b5",
        ),
        (
            "receipt_hash",
            receipt,
            "7df29d85cd4bf4f1f497d37d968bd83a57fd9caec1babe681cf5ff6408c91b5b",
        ),
        (
            "batch_root",
            batch_root,
            "f8e6e0497187f154f072319a37c8f847ba19113c4e3ad0fc8ba86900f3c10c4a",
        ),
    ];
    if std::env::var_os("QX_PRINT_GOLDEN").is_some() {
        for (name, digest, _) in vectors {
            println!("{name}={}", hex(digest));
        }
        return;
    }
    for (name, digest, expected) in vectors {
        assert_eq!(hex(digest), expected, "{name}");
    }
}
