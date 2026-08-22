# QX current status

Date: 2026-08-22  
Target: Apple M5 Pro  
Model: IBM Granite 3.0 1B-A400M Instruct  
Canonical workload: batch-1 greedy decode, context 512, 128 sequential token
steps from `benchmarks/granite_decode_512.json`

## Executive status

QX has a correct, complete, artifact-only GPU decode runtime. Runtime V2 keeps
the transformer state, routing, expert intermediates, KV cache, logits, and
argmax on the GPU. Each token uses one command buffer, one compute encoder, no
internal CPU wait, and one final synchronization to return the selected token.

The canonical release run after fixing fixture and warmup semantics is:

| Metric | QX V2 | llama.cpp reference |
|---|---:|---:|
| Artifact/model bytes | 750,982,977 | 820,107,736 |
| GPU ms/token | 4.987 median | not exposed by normal run |
| Wall/eval ms/token | 5.418 median | 3.18 |
| Decode tok/s | 184.58 median | 310.56 |
| Command buffers/token | 1 | not instrumented |
| Internal waits/token | 0 | not instrumented |
| Dispatches/token | 461 | not instrumented |
| Current process RSS | 1,214,414,848 B median | not exposed |
| Peak process RSS | 1,891,237,888 B median | 916,668,416 B |

The llama number is a fresh same-session `llama-simple` run from seed token
8279 with 128 generated tokens, build commit `1719747`. QX currently reaches
about 59.4% of its end-to-end throughput. The historical 205 tok/s QX result
remains a valid observed run,
but it is not the baseline to optimize against until reproduced under this
canonical fixture and reporting path.

The eight-position sequential same-artifact gate passes through position 511:
maximum KV-key error is below 0.01, logit cosine is 1.000000, and argmax agrees
8/8. The release `qx-generate` command now uses Runtime V2 by default and emits
coherent text; `--runtime v1` retains the reference runtime.

## Canonical architecture

The compiler emits a complete `.qx` artifact containing row-major group-32 Q4
dense, router, embedding, gate/up, and down matrices plus exact FP32 norm
vectors. Runtime V2 prepares persistent Metal resources from that artifact and
uses a static 24-layer execution plan with a small reusable arena and dedicated
KV allocation.

The promoted per-layer graph is:

```text
inverse-RMS scalar -> normalized fused QKV
Q RoPE + direct K/V cache writes
three-stage decode attention (QK, hierarchical softmax, GQA-aware V)
Q4 O projection + residual
inverse-RMS scalar -> normalized router -> top-8
four-stage dynamic MoE (gate/up, SiLU product, down, weighted accumulation)
residual
```

The token tail is final RMSNorm, tied Q4 LM head, hierarchical GPU argmax, then
one `u32` readback.

## What the experiments established

### Execution topology is a first-class optimization

Minimizing dispatches was repeatedly the wrong objective. Four-stage MoE beat
the single fused MoE kernel, three-stage attention beat one-dispatch attention,
and split/direct RoPE-KV improved the full model while increasing dispatches.
The useful compiler objective is end-to-end latency subject to occupancy,
register, memory, and quality constraints.

### Local winners do not reliably compose

Tile32 expert packing, split gate/up, and the `q2-kv1-g8` QKV geometry all won
or looked promising in isolation and lost in the complete graph. Candidate
selection therefore needs four tiers: kernel microbenchmark, producer/consumer
replay, multi-layer replay, and complete-model promotion.

### Correctness and quality are different gates

Kernel/topology rewrites must pass the strict same-computation gate. Reduced
precision or fewer active experts intentionally change the model and must be
judged by teacher-forced NLL/PPL, KL, logit similarity, top-k behavior, and
held-out generation. FP16 KV and active-k results made this distinction
explicit.

### MoE remains the dominant full-program opportunity

Full-graph differential profiling attributes about 3.13 ms/token of removable
cost to MoE, reconstructed coherently as roughly 2.02 ms gate/up, 1.09 ms down,
and negligible SiLU/accumulation. Traffic controls show that selected-expert
weight access dominates arithmetic, but simple prefetch, private resources,
expert ordering, physical permutation, and uniform Q3 did not improve the
complete model.

### Active expert count has leverage but currently loses quality

Global active-k improved the same-run GPU curve from 4.814 ms at k=8 to 4.448
ms at k=6 and 3.576 ms at k=4. Teacher forcing rejected global k=7 (+3.59%
PPL) and k=6 (+8.87% PPL). A layer-12-only k=6 policy stayed within the short
quality budget (-0.33% PPL) and passed the implementation parity oracle, but
its throughput was neutral. Canonical execution therefore remains top-8.

## Promoted and rejected branches

Promoted:

- one-command-buffer Runtime V2;
- cooperative Q4 SIMD kernels;
- scalar RMS consumers for QKV and router/MoE;
- fused QKV with canonical geometry;
- split/direct RoPE-KV;
- three-stage attention, hierarchical softmax, and paired-GQA V reduction;
- four-stage dynamic top-8 MoE;
- row-major expert representation at whole-model scope;
- hierarchical GPU argmax;
- canonical V2 user-facing generation.

Retained only as diagnostics or rollback paths:

- split Q/K/V and alternative fused-QKV geometry;
- single-dispatch attention and single-dispatch MoE;
- split gate/up;
- Tile32, mixed, paired, reordered, and permuted expert plans;
- serialized route prefetch and private static Metal buffers;
- uniform dense or padded Q3 gate/up;
- full FP16 KV under the strict reference contract;
- global and current layerwise active-k policies.

## What is still missing

### 1. A quality-matched competitive baseline

The largest scientific gap is a long, same-token, teacher-forced comparison of
QX uniform Q4 against llama.cpp Q4_K_M. It must cover held-out general text,
instruction/chat, code, and reasoning-like fixtures, with calibration and
evaluation kept separate. Current active-k screening uses only 128 PTB
positions and cannot support a product-quality claim.

### 2. Steady-state and peak memory closure

The logical Q4 target is about 765 MiB, current RSS is about 1.21 GB, and load
peak is about 1.89 GB. The ~677 MB peak-to-current drop is close to the expert
artifact size, strongly suggesting that faulted mmap pages coexist with copied
Metal weight buffers during preparation. The next memory experiment should be
mmap-backed/no-copy static Metal buffers or another streaming preparation path,
plus allocation attribution for dense weights, expert weights, KV, pipelines,
and scratch. Peak RSS alone previously overstated steady-state residency.

### 3. A reproducible fresh llama comparison

The canonical QX command is now fixture-stamped and the same-session llama run
is recorded, but this must become an automated benchmark record containing the
exact binary revision, flags, model hash, context, and token count. Per-kernel
replay numbers are useful for hypotheses but are not additive live costs.

### 4. Automated execution-plan selection

QX has manually discovered several model/hardware-specific plans, but the
compiler does not yet search, benchmark, store, and select them. The strongest
thesis demonstrated so far is the search procedure, not an automated compiler.
A plan database must treat full-program latency and quality as promotion gates.

### 5. Sparse quality/latency allocation

Uniform Q3 and global active-k are rejected, but the infrastructure supports a
future optimizer over layer/expert precision and active count. This should not
resume until the multi-domain quality harness exists. Router mass alone is not
enough; actual weighted expert-output contribution and held-out NLL are needed.

### 6. Repository reproducibility

The workspace currently has no tracked baseline commit: every project file is
reported as untracked. That makes artifact/result provenance fragile even
though benchmark output now includes an artifact fingerprint. Establishing a
real Git baseline is required before another large optimization phase.

## Recommended order

1. Freeze canonical top-8 V2 and record repeated same-session QX/llama runs.
2. Build the matched multi-domain teacher-forced QX-Q4 versus llama-Q4 quality
   gate.
3. Attribute current and peak memory; test no-copy/mmap-backed static weights.
4. Re-run the live differential profile under the canonical fixture and attack
   only components with a large measured full-program delta and llama gap.
5. Return to sparse specialization through a quality-constrained optimizer,
   starting with actual expert contributions and layerwise choices.
6. Encode successful/rejected plans into compiler-search data rather than
   accumulating more environment-variable experiments.

## Canonical commands

```bash
cargo build --release --workspace

target/release/qx-bench \
  external/models/granite-m5-pro-q4-dense.qx \
  --v2-token --fixture benchmarks/granite_decode_512.json --warmup 3

target/release/qx-bench \
  external/models/granite-m5-pro-q4-dense.qx \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --v2-parity --context 512

target/release/qx-generate \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --artifact external/models/granite-m5-pro-q4-dense.qx \
  --prompt Hello --max-new-tokens 16 --context 512
```
