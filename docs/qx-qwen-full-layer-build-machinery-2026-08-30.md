# Qwen full-layer build machinery

Date: 2026-08-30  
Status: capture path validated; first of five evidence shards complete

## Outcome

The complete 48-layer mixed-bit experiment no longer requires one full GGUF
replay per layer. `capture-corpus --sample-all-layers` captures the routed
input and exact Q4 teacher down-output for all `48 × 8` active layer/slot pairs
during one replay. The output can be split into one trainer-compatible artifact
per layer.

A two-token live control proved that the new all-layer path produced all 768
expected samples. Layer 24's inputs, activations, and teacher outputs hashed
identically to a separate legacy single-layer capture. The routed expert IDs
also matched exactly.

## Compact evidence format

The first JSON implementation was correct but operationally unusable. It
repeated the same 2,048-float layer input eight times, serialized an unused
768-float activation for every route, and formatted every F32 value as text.
The two-token, 48-layer smoke artifact occupied 43 MiB, and a bounded shard
advanced by only about 12 tokens in 135 seconds.

The `QXDS0001` binary format stores, per layer and token:

- one shared F32 expert input;
- zero to eight expert IDs after the evidence cap;
- one F32 Q4 teacher down-output per retained expert;
- sequence and token position;
- the source GGUF SHA-256 in every file header.

The trainer reads this format directly. It does not reconstruct or quantize the
capture values. The same two-token artifact occupies 6.9 MiB, a 6.2× reduction
from JSON. A separate-run comparison observed only normal Metal replay jitter:
maximum absolute difference was `4.44e-7` for inputs and `4.53e-8` for teacher
outputs. The byte-level writer test verifies exact F32 little-endian storage.

The first compact implementation still retained 144 callback tensors per token.
That fragmented llama's graph execution and reduced a longer run to 68 tokens
in 308 seconds (0.22 captured tok/s). The canonical collector now lives at
llama.cpp commit `4f48604`: it marks `ffn_norm`, `ffn_moe_topk`, and
`ffn_moe_down` as graph outputs, executes the production graph once, performs
one post-graph synchronization, and then writes all layers. It does not install
sampling callbacks.

The post-graph and callback paths matched bit-for-bit across all eight test
tokens for layer 24, including routes, inputs, and teacher outputs. The first
real eight-prompt shard captured 227 tokens and retained 31,645 samples in 65.9
seconds, including model load, raw conversion, and completion marking. That is
3.44 captured tok/s and about 15.6× the observed callback rate.

## Resumable build

`tools/build_qwen_full_experts.py` makes both capture and training bounded and
resumable:

- four eight-sequence training shards cover 32 reset sequences;
- one eight-sequence held-out shard supplies the promotion evidence;
- every capture shard gets an atomic completion marker tied to its full input
  and shape contract;
- every trained layer is a resume boundary through its validated manifest;
- the default invocation performs at most one missing work unit;
- layers 2, 24, and 44 reuse the already validated full-width Q2S4 manifests,
  leaving 45 layers to train.

The first shard is complete and occupies 316 MiB. Its 48 files passed direct
trainer reads at representative layers 0, 24, and 47. The transient combined
post-graph file is removed only after all per-layer artifacts and route records
are written successfully.

The training corpus is the 12 curated general prompts followed by WikiText
paragraphs. The driver now rejects a requested sequence count larger than the
available reset paragraphs, preventing a late empty-shard failure. Held-out
evidence uses eight distinct curated general prompts.

The canonical invocation uses the existing Torch/MPS environment:

```sh
external/venv314/bin/python tools/build_qwen_full_experts.py \
  --model external/models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf \
  --model-sha256 0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5 \
  --model-root external/models/qwen3-30b-a3b-hf-ad44e777 \
  --train-corpus external/corpora/qwen-residency-general-calibration.txt \
  --train-corpus external/corpora/wikitext-2/train.txt \
  --heldout-corpus external/corpora/qwen-residency-general-heldout.txt \
  --work-dir external/results/qwen3-full-q2s4-build \
  --reuse-manifest external/results/qwen3-width768q2s4-layer2-manifest.json \
  --reuse-manifest external/results/qwen3-width768q2s4-layer24-manifest.json \
  --reuse-manifest external/results/qwen3-width768q2s4-layer44-manifest.json
```

## Remaining gate

Completion of capture and local training is not promotion. The resulting 48
layer policy still requires model-level exact-anchor selection, consumer-native
packing, full replay quality, installed-byte accounting, peak iPhone residency,
and physical iPhone 17 Pro Max throughput measurements.
