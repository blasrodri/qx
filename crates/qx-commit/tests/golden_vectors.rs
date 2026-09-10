use qx_commit::{
    CheckpointV1, D_BATCH, D_JOB, D_KV, D_TOKEN, D_WEIGHT_TREE, HashAlgorithm, InferenceReceiptV1,
    MerkleTree, ModelSpecV1, ProtocolVersion, WeightLeafV1, granite_architecture_root,
    granite_execution_spec, hash_domain, input_hash_v1, output_hash_v1,
};

fn hex(value: [u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn protocol_v1_golden_vectors() {
    let algorithm = HashAlgorithm::Sha256V1;
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
    let kv_leaf = hash_domain(
        algorithm,
        D_KV,
        &[&2_u16.to_le_bytes(), &9_u32.to_le_bytes(), &key, &value],
    )
    .unwrap();
    let kv_other = hash_domain(algorithm, D_KV, &[b"other-kv-leaf"]).unwrap();
    let kv_root = MerkleTree::new(algorithm, D_KV, &[kv_leaf, kv_other])
        .unwrap()
        .root();
    let checkpoint = CheckpointV1 {
        job_id: [0x11; 32],
        token_index: 9,
        layer_index: 2,
        residual_hash: hash_domain(algorithm, qx_commit::D_STATE, &[&f32_bytes(&[0.25, -1.5])])
            .unwrap(),
        global_kv_root: kv_root,
    }
    .hash()
    .unwrap();
    let checkpoint_other = hash_domain(algorithm, qx_commit::D_CHECKPOINT, &[b"other"]).unwrap();
    let token_root = MerkleTree::new(algorithm, D_TOKEN, &[checkpoint, checkpoint_other])
        .unwrap()
        .root();
    let job_other = hash_domain(algorithm, D_TOKEN, &[b"other-token"]).unwrap();
    let job_root = MerkleTree::new(algorithm, D_JOB, &[token_root, job_other])
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
            "3028b0108d882f938eca5f94d09fa8c7b08a8161787bbb9703af9bec3b055161",
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
            "7cf0943ec8e37d6c10f02a97d79277b7ecca104d066edf774cacbffb88be6382",
        ),
        (
            "kv_root",
            kv_root,
            "9397d4bdaa9c15117bbc393e5749e381eabf240287b489a7cada2ff3aefc4492",
        ),
        (
            "checkpoint",
            checkpoint,
            "3f6d49a11fb729ab12869749cf5b77b39ed024df8fc7035334a80d3f64ea2b94",
        ),
        (
            "token_root",
            token_root,
            "f03ad52d23aacb1ed03683a190d09e90ecca9ee242e2df59509403d5ced772c2",
        ),
        (
            "job_root",
            job_root,
            "19a0b3ba62fddb16f77625296952189d32d959f4a98e1412f3b26454ee0826e3",
        ),
        (
            "input_hash",
            input_hash,
            "54b91ab41c81270d7089b09766881456931f64345e4f2251a36c6f3f1bc5d0ae",
        ),
        (
            "output_hash",
            output_hash,
            "88e599929cdefc6107f75e0fffdcac8e43cd957875740a9a60fd93daefc479b5",
        ),
        (
            "receipt_hash",
            receipt,
            "14179fd10e467c391cd01668736d849ddb6083235c95a4e17a6ba809f2f80e3b",
        ),
        (
            "batch_root",
            batch_root,
            "d8a672dd01fa236d86869b199418361a9baa0d0c1fdf2da1145a9faa98f5e288",
        ),
    ];
    for (name, digest, expected) in vectors {
        assert_eq!(hex(digest), expected, "{name}");
    }
}
