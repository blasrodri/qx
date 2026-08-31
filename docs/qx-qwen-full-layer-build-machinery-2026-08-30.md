# Qwen full-layer build machinery

Date: 2026-08-31
Status: five base evidence shards complete; six of 48 layers trained or reused

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

All five base shards are complete: four training shards and one held-out shard
contain exactly 240 non-empty QXDS files and occupy 1.6 GiB. Their transient
combined post-graph files were removed only after all per-layer artifacts and
route records were written successfully. Layers 0, 1, and 3 have since been
trained locally; layers 2, 24, and 44 are recognized through reused manifests.

The training corpus is the 12 curated general prompts followed by WikiText
paragraphs. The driver now rejects a requested sequence count larger than the
available reset paragraphs, preventing a late empty-shard failure. Held-out
evidence uses eight distinct curated general prompts.

Those eight held-out prompts cover roughly 98–99% of routed positions in the
first newly trained layers, but leave 28–40 experts per early layer without the
minimum independent evidence. Extrapolating that exact-retained density would
erase several GiB of installed-weight savings. The driver therefore supports
append-only held-out extensions. An extension uses a separate corpus and
directory, preserves every base-shard fingerprint, and becomes an additional
trainer input only after its own atomic completion marker exists:

```sh
  --heldout-extension-corpus external/corpora/wikitext-2/test.txt \
  --heldout-extension-sequences 8 \
  --heldout-extension-samples-per-expert 8 \
  --heldout-gpu-layers 40
```

The first extension shard captured 335 routed positions and 29,265 expert hits
from eight WikiText test sequences in 350.9 seconds. The completed build now
contains six atomic markers, 288 non-empty QXDS files, and no transient raw
traces. The reduced held-out GPU residency avoided the Metal out-of-memory
failure observed with all 48 layers offloaded.

An isolated layer-3 retrain measured whether the extension merely added data or
actually recovered deployable capacity:

| Layer-3 evidence | Transformed | Exact retained | Route coverage | Mean relative L2 | Mean cosine |
|---|---:|---:|---:|---:|---:|
| curated held-out | 88 | 40 | 98.38% | 0.475 | 0.879 |
| + WikiText extension | 112 | 16 | 99.40% | 0.462 | 0.884 |

The extension recovered 24 experts while also improving aggregate error and
cosine. At the measured 1,585,152-byte premium of an exact Q4/Q6 expert over
Q2S4, that avoids 36.3 MiB of installed payload for layer 3.

## Layer-3 causal promotion

Full-model replay on four selection prompts showed that the evidence extension
was storage-positive but not itself a quality fix: the 88- and 112-student
policies both retained 80% top-1 agreement, with respective perplexity deltas
of +10.99% and +10.83%. An independent 12-expert exact-fallback panel found one
dominant causal anchor, `3:82`. Restoring only that original GGUF expert raised
the selection shard to 100% top-1 and reduced the perplexity delta to +1.21%.

The anchor was then frozen and evaluated without reselection on the remaining
eight general prompts. It was neutral on both shards because those prompts did
not route through the damaging path. Across all 343 scored tokens:

| Layer-3 policy | PPL delta | Top-1 agreement | Exact premium |
|---|---:|---:|---:|
| 112 Q2S4 students | +4.77% | 90.67% | 0 |
| + exact `3:82` | **+1.36%** | **97.96%** | 1.512 MiB |

The broader evidence therefore removes 36.3 MiB of avoidable exact payload and
adds back only 1.512 MiB for the causally demonstrated anchor, a net layer-3
saving of 34.77 MiB. The 112-student manifest is now canonical; the previous
88-student manifest and sidecars remain in the external build archive for
rollback. This promotes the layer-3 policy, not the unfinished 48-layer model.

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
