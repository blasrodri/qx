# QX verification model profile (protocol V1)

This profile was locked from the repository constants, the installed Granite
`config.json`, and the current full QX artifact before consensus code was
introduced. Runtime loaders reject metadata that differs from these values.

## Registered model

| Field | Exact value |
|---|---:|
| Upstream model | `ibm-granite-3.0-1b-a400m-instruct` |
| Architecture | `GraniteMoeForCausalLM` / `granitemoe` |
| Transformer layers | 24 |
| Hidden size | 1,024 |
| Attention heads | 16 |
| KV heads | 8 |
| Head dimension | 64 |
| Local experts per layer | 32 |
| Selected experts per token | 8 |
| Expert intermediate size | 512 |
| Vocabulary | 49,155 |
| Maximum context | 4,096 tokens |
| Tied input/output embedding | yes |
| Router weights normalized | yes, softmax over selected top-8 |
| RMS epsilon | `1e-6` (FP32 bits `0x358637bd`) |
| RoPE theta | `10000.0` |
| Attention multiplier | `0.015625` |
| Embedding multiplier | `12.0` |
| Residual multiplier | `0.22` |
| Logits scaling | `6.0` |
| Generation | batch 1, greedy |
| Top-K tie break | lowest expert ID |
| Argmax tie break | lowest token ID |

Sources in the repository are `qx_model::ModelShape::granite_v0`,
`GraniteConfig::validate_v0`, the V1 `GraniteModel` loader, and the V2 artifact
execution-plan validation. The V1 loader now reads and validates `config.json`
before loading any tensor. V2 rejects a sealed `model_shape` other than the
same `granite_v0` shape.

The inspected production-control artifact is
`external/models/granite-m5-pro-q4-rowmajor-nocopy-v2.qx` (not committed). Its
observed metadata is:

| Field | Exact value |
|---|---:|
| Target | `apple-m5-pro` |
| File bytes | 751,009,792 |
| Tensor entries | 1,706 |
| Q4 entries | 1,657 |
| Raw FP32 entries | 49 |
| Gate-up / down entries | 768 / 768 |
| Expert payload | 679,477,248 bytes |
| Dense quantized payload | 71,222,976 bytes |
| Raw norm vectors | 200,704 bytes |
| Logical weight payload | 750,900,928 bytes |

The Phase-2 canonical commitment for that artifact is:

```text
weights_root = c31f3758efc5be3b410c0747bea880128a026d3bb7f38a1fd51e1a8949072d43
weight_leaves = 41,755,744
tensor_count = 1,706
```

## Existing runtime map

- Model loading: V1 `GraniteModel::from_paths` reads the Hugging Face
  checkpoint for dense tensors and a mapped QX artifact for experts. V2
  `GraniteV2Generator::load` loads the complete mapped artifact and activates
  its sealed execution plan.
- Quantization: `qx-quant::QuantizedTensor`; artifact mapping and physical
  layout decoding are in `qx-format::MappedArtifact`.
- Metal entrypoint: V2 `GraniteV2Generator::step` and
  `PreparedAttentionStack::encode_token_with_plan`.
- Transformer loop: V1 `GraniteModel::step_internal`; V2
  `PreparedAttentionStack::encode_full_layer_range_with_plan`.
- KV cache: V1 `KvCache` uses per-layer contiguous FP32 K and V arrays. V2
  `GpuRuntimeBuffers` uses one GPU arena with a sealed `KvPrecision`.
- MoE routing/execution: V1 `GraniteLayerWeights::route_from_logits` and
  `granite_moe_decode`; V2 uses `encode_topk` and the dynamic Q4 MoE kernels.
- Token generation: V1 `generate_tokens_with_trace_mode`; V2
  `generate_tokens_host_synchronized` and GPU-resident rollout variants.
- Greedy selection: V1 `argmax` uses `f32::total_cmp` and lowest-ID tie
  breaking; V2 uses the Metal argmax reduction.
- Tests: crate-local Rust unit tests, Metal parity tests in `qx-metal` and
  `qx-runtime-v2`, benchmark contracts under `benchmarks/`, and C Q4_K
  reference vectors under `tests/`.

## Quantization and weight commitment

Protocol V1 accepts only QX symmetric Q4 with group size 32 for quantized
matrices. For each logical row/group:

- scale = FP16(`max(abs(group)) / 7`), with zero groups using `1.0`;
- 32 signed values are rounded and clamped to `[-7, 7]`;
- stored code = quantized value + 7;
- two 4-bit codes share one byte, least-significant nibble first;
- a full group is 16 packed-code bytes plus one 2-byte FP16 scale.

Physical row-major, MetalTile32, row-inline, transaction, cross-expert, and
superblock arrangements are transport details. `WeightLeafV1` always commits
the same logical `(tensor_id, row, group, Q4, packed codes, scale bits)`.
Raw FP32 norm vectors are committed in logical 32-value chunks with type 0 and
scale bits 0. The SBF verifier can authenticate an exact finite FP32 value from
such a leaf and apply it through the same staged weighted-FMA transition used
for cross-chunk Q4 operands.

Tensor IDs are assigned after sorting in this semantic order:

1. tied embedding;
2. each layer in ascending order: input norm, Q, K, V, O, post-attention norm,
   router, all gate/up experts by ID, all down experts by ID, query norm, key
   norm;
3. final norm and an explicit LM head if the model is not tied.

Leaves are ordered by tensor ID, row, then group. Their already-domain-separated
hashes form a binary tree padded to the next power of two with
`QX:MERKLE:EMPTY:V1` leaves. Internal nodes use `QX:MERKLE:V1` and include the
tree domain `QX:WEIGHT:TREE:V1`. Construction keeps only O(log N) hashes;
inclusion proofs can be regenerated by replaying the immutable artifact.

## Phase-3 checkpoint state

The default profile can commit after every transformer layer. The selected
lower-overhead profile commits residuals after each two-layer span while still
committing every layer's new K/V. At each retained checkpoint, the CPU oracle
commits:

1. the exact 1,024 FP32 values of the post-MoE residual stream;
2. the exact rotated 8 × 64 FP32 key values appended by that layer;
3. the exact 8 × 64 FP32 value values appended by that layer;
4. token position, layer index, job ID, and the incrementally updated global
   per-layer KV root.

The every-layer profile covers **196,608 canonical activation/KV bytes per
executed token** (`24 × (1024 + 512 + 512) × 4`). The selected two-layer
profile covers every K/V append, 12 post-layer residuals, and the pre-layer
embedding residual, or **151,552 bytes/token**.
Only digests and Merkle frontiers remain after a token; no activation tensor is
retained.
Each completed token adds a 76-byte canonical token-result preimage binding the
job ID, token index, checkpoint root, checkpoint leaf count, and greedy
predicted token. Its hash,
rather than the bare checkpoint root, is the job-tree leaf. This prevents
attaching a different output sequence to an honest execution trace and adds no
permanent activation storage.
The accelerator-facing encoding divides each canonical tensor into 256-byte
leaves, binds each leaf's index, and reduces it under separate residual, key,
and value tree domains. The CPU oracle and Metal implementation produce
identical commitments and the structure permits independent chunk proofs
during replay.

The verifier schedule has `checkpoint_count + 1` leaves. For the
selected two-layer profile this is 14 leaves: index 0 is reserved, indices
1–12 bind the distinct instruction root and length for each two-layer
checkpoint interval, and index 13 binds the final norm/LM-head/argmax program.
Each 40-byte segment descriptor commits `(segment_index, instruction_root,
instruction_count)`. This prevents a proof for one layer's otherwise
well-formed arithmetic from substituting another layer's weights or schedule.
The canonical macro compiler emits 28 instructions per two-layer interval
(14 per layer) and three for the final head. It validates its tensor mapping
against the installed 1,706-entry artifact without reading the weight payload.
The challenge-only runtime emits matching roots for normalized/Q/K/V,
rotated-QK, K/V leaf, attention, router, selected routes, MoE, residual,
final-hidden, logits, and the winning `[logit, token]` record. The remaining
implementation gate is a rolling commitment to all simultaneously live macro
values and descent of the one disputed macro into scalar terminals on chain.
Generation executes and commits `input_token_count + output_token_count - 1`
tokens in the usual non-EOS case because the final output is the prediction of
the preceding step.

`InferenceReceiptV1` is 210 canonical bytes. Adding an Ed25519 signature makes
the signed receipt exactly **274 bytes**, below the 512-byte target.

## CPU/Metal compatibility and Phase-4 gate

The current sealed M5 V2 artifact stores both K and V as FP32. Metal now hashes
the exact GPU-resident bytes and matches the CPU BLAKE3 oracle in a golden
parity test. Pre/post-layer residual snapshots use 102,400 transient bytes per
runtime instance and are written by the existing final residual kernel; the
snapshots are overwritten each token and are never receipt or permanent trace
storage. A single token-end hash dispatch processes all 24 layers and returns
only 1,536 digest bytes after the runtime's existing token synchronization.

This closes the byte-level commitment gap. Consensus arithmetic now uses an
explicit 32-lane shuffle-down reduction tree, protocol-defined fixed-operation
`exp`, `rsqrt`, `sin`, and `cos`, a pinned 32-value RoPE inverse-frequency
table, and safe Metal arithmetic; CPU, Metal, and SBF primitives reproduce the
same operation order. Two fixed-RoPE 31-repeat runs measured 3.510% and 3.003%
commitment overhead. Its
512-position PTB PPL differed from native precise math by only
`4.8735e-8` relatively, with complete sampled top-k agreement. The selected
interval-2 execution-spec is
`f5ff8f542c09108d080e22b90f4d8d7e0335c6b2e4750b3627959aac13e0eec9`.
It binds the hash design, reduction tree, math mode, fixed functions,
checkpoint interval, canonical genesis leaves, checkpoint-tree length, and
compact job-root semantics. See `docs/verification-benchmarks.md` for exact
measurements.

A distinct relaxed-arithmetic/precise-functions candidate now measures 3.734%
total overhead using the means of two independent 31-repeat candidate runs and
two bracketing fast controls. It is repeatable on this M5 Pro but remains
non-slashable until cross-device parity and compiler/metallib pinning are
demonstrated. It is retained as historical performance evidence rather than
the default protocol profile.
