# QX PoC — Iteration 4 Report

Date: 2026-08-21  
Target: Apple M5 Pro  
Model: IBM Granite 3.0 1B-A400M Instruct  
Workload: batch 1, greedy decode

## Executive result

Iteration 4 now has a separate V2 execution-plan crate, a reusable Metal
encode graph boundary, and a complete artifact-only V2 token graph. Same-
artifact Q4 logits parity and FP32 cross-implementation parity pass. QKV
projection fusion is now implemented and measured; output/residual fusion
remains open.

| Item | Status |
|---|---|
| Separate `qx-runtime-v2` crate | Implemented |
| Static 512-token Granite arena plan | Implemented and unit-tested |
| Hidden ping-pong plan for 24 layers | Implemented and unit-tested |
| One command buffer / one encoder lifecycle | Implemented at Metal boundary |
| Preallocated V2 runtime arena and KV storage | Implemented and tested |
| Encode-only dense and logits operators | Implemented at Metal boundary |
| Fixed-shape RMSNorm/conversion/residual/RoPE/attention/top-k kernels | Implemented and Metal-tested |
| One-layer GPU attention subgraph | Implemented and Metal-tested |
| GPU greedy argmax into a 4-byte token buffer | Implemented and Metal-tested |
| Complete GPU-resident 24-layer decoder | Implemented for sequential artifact-only token decode |
| GPU router/top-k and GPU KV cache | Implemented and Metal-tested |
| V2 uniform-Q4 generation | Tokenizer-backed prompt prefill and greedy continuation implemented |
| Fused QKV Q4 projection | Implemented, parity-validated, and measured |

## What changed

`crates/qx-runtime-v2` defines the fixed Granite topology instead of
discovering tensor shapes during decode. At the 512-token target it allocates
aligned regions for hidden-state ping-pong, Q/K/V, attention scratch, router
outputs, top-k metadata, MoE scratch/output, logits, and the selected token.
KV capacity is accounted for separately. The original fused plan targeted one
command buffer, one encoder, and 268 dispatches per token. The live path now
uses 437 dispatches per token: 18 per transformer layer with hierarchical
three-stage attention and four-stage batched MoE, plus embedding, final norm, LM head, and
two-stage argmax. Scalar-only RMS→QKV is the canonical norm topology. The older
paths remain available as explicit rollbacks.
Every layer also records the planned
kernel variant for norm/QKV, RoPE/KV, attention, output residual, router/top-k,
and MoE.

`qx-metal` now exposes a `DecodeGraph` whose operators only encode work. The
graph owns the command buffer and encoder; `finish` is the only place that
ends encoding, commits, waits, and records timing. V2 runtime buffers are
allocated through `MetalContext` before decode and can be reused across
tokens. Prepared dense and logits pipelines can bind those buffers without
creating an operation-specific command buffer. Metal metrics now report
command buffers, encoders, dispatches, GPU time, CPU encode time, and CPU wait
time.

The fixed-shape V2 kernel family is implemented in
`kernels/v2_decode.metal`: RMSNorm, f32/f16 conversion, residual, RoPE plus KV
write, query-length-1 attention, and fixed top-8-of-32 routing. GPU argmax is
implemented in `kernels/argmax.metal`. It writes only the final
`u32` token result to a tiny shared buffer. The macOS regression test verifies
that the kernel family compiles and executes in Metal graphs. Separate tests
verify that a graph containing argmax records exactly one command buffer and
one dispatch, and that the result is correct after the final synchronization.

The V2 owner composes the attention half of one layer as
RMSNorm → fused QKV → RoPE/KV → decode attention → O projection → residual. A
target test executes this chain in one command buffer using real prepared Metal
pipelines and arena/KV views; the fused attention chain records six dispatches
instead of eight. The fused path is bound to all 24 layers, GPU routing,
dynamic Q4 MoE, final norm, tied Q4 LM head, and GPU argmax.

## V1 measurements retained as the correctness baseline

Historical V1 and intermediate V2 measurements remain useful baselines:

| Runtime | Format | Decode | Command buffers/token | CPU wait/token |
|---|---|---:|---:|---:|
| QX V1 | uniform Q4 | ~28.3 tok/s | ~100 | ~35.0 ms |
| QX V1 diagnostic | CPU dense + GPU MoE | ~37.6 tok/s | ~24.8 | ~22.4 ms |
| llama.cpp | Q4_K_M Metal | 157.93 tok/s* | captured | captured |
| QX V2 unfused | full artifact Q4 | 33.89 tok/s | 1 | 1 |
| QX V2 fused QKV + batched SIMD MoE | full artifact Q4 | ~59 tok/s sustained | 1 | 1 |
| QX V2 promoted scalar RMS/QKV + RMS/router + hierarchical attention | full artifact Q4 | 160.83 tok/s sustained | 1 | 1 |

\* Direct short-run comparison: same `Hello`, four generated tokens, and
context 512. llama.cpp reported 157.93 decode tok/s, 6.56 ms TTFT, and
904,396,800 bytes peak RSS; the unfused QX V2 run reported 33.93 decode tok/s,
81.94 ms prefill, and 1,904,115,712 bytes peak RSS. The fused QX V2 run now
reports 37.56 decode tok/s and 1,589,903,360 bytes peak RSS. The earlier 310.7
tok/s figure is the longer historical llama.cpp fixture and is retained below
for context.

### Matched 128-token decode comparison

For the direct comparison below, both runtimes use batch 1, one prompt token,
context 512, and 128 generated tokens in release/Metal builds. The QX number
uses the sustained context-growing run, not the shorter three-token smoke
measurement above.

| | llama.cpp | QX V2 |
|---|---:|---:|
| model bytes | 821,844,576 B (Q4_K_M) | 750,982,977 B artifact; 801,774,528 B logical resident target |
| weight format | Q4_K_M | row-major Q4 |
| command buffers/token | not instrumented | 1 |
| internal waits | not instrumented | 0; one final wait per token |
| GPU ms/token | 3.09 | 5.81 |
| tok/s | 323.58 | 160.83 |

Relative to the fresh llama.cpp run, QX V2 reaches 49.7% of decode throughput
and uses about 1.9× the GPU time per token. Relative to the historical 310.7
tok/s fixture, QX V2 is 51.7%.

The previous scalar-RMS/QKV result was 113.81 tok/s at context 512; the
promoted V and hierarchical softmax attention topologies raise the matched
result to 160.83 tok/s. Roll back with `QX_V2_ATTENTION_V_SIMD=0`,
`QX_V2_ATTENTION_SOFTMAX_SIMD=0`, `QX_V2_RMS_SCALAR_ROUTER=0`,
`QX_V2_ATTENTION_3_STAGE=0`, or `QX_V2_RMS_SCALAR_QKV=0` as needed.

V2 now has a reproducible static audit:

```text
context:                    512
    arena:                      341,248 bytes
KV (FP32 validation):       50,331,648 bytes
uniform-Q4 expert artifact: 679,477,248 bytes
dense artifact (Q4 + raw norms): 71,423,680 bytes
dense FP32 reference:        506,675,200 bytes
full dense Q4 target:        71,521,984 bytes
estimated FP32 reference:    1,236,759,808 bytes = 1,179.5 MiB
    estimated full-Q4 target:     801,774,528 bytes = 764.6 MiB
```

Run it with:

```bash
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-q4.qx --v2-plan --context 512
```

This is an allocation estimate, not measured process RSS. The old FP32
reference is retained to expose the confounder that motivated this iteration;
the dense Q4 matrices and exact FP32 norms are now emitted by the compiler and
loaded from the `.qx` artifact. The FP32 KV figure is deliberately
conservative for validation and should become FP16 after parity is
established. The latest release-build token-graph run observes a macOS process
peak of about 1.89 GB; that includes Metal shared-buffer allocations and
staging copies, so it is materially above the logical resident estimate and
remains a memory-optimization gate.

The real 24-layer attention-only graph, using the Q4 projections, currently
measures:

```text
wall:       45.543 ms/token
GPU:        18.728 ms/token
dispatches: 192/token
```

That is approximately 21.96 attention-only graph tokens/sec, not full model
generation. The retained unfused measurements were 38.25 tok/s for the
attention+MoE core and 33.89 tok/s for the complete token graph. The current
Scalar-only RMS→QKV and RMS→router, SIMD QKV, hierarchical attention, and
four-stage batched MoE graph measures 160.83 sustained tok/s at 437 dispatches.
Same-artifact Q4 parity passes on the current path.

The first complete V2 token graph now executes:

```text
Q4 embedding lookup
→ 24 × (attention + GPU router/top-k + dynamic Q4 MoE)
→ final RMSNorm
→ Q4 tied LM head
→ GPU argmax
→ u32 token ID
```

With seed token 0, the artifact-only loop advances through sequential KV
positions and produced token ID 34 after the measured run:

```text
wall:       ~6.22 ms/token
GPU:        ~5.81 ms/token
dispatches: 437/token
CB/encoder: 1/1 per token
complete rate: ~152.4 tok/s
peak RSS:   ~1.89 GB
```

This is a complete sequential decode graph; the seeded benchmark is synthetic,
while the artifact parity command below covers multi-position logits and
greedy-token agreement.

### Ranked per-token component profile

The release profile ranks fresh context-512 isolated GPU medians by their live
multiplicity; the promoted normalized MoE path is reported as a four-stage
total because its gate/up stage is projection-aware:

```bash
target/release/qx-bench \
  external/models/granite-m5-pro-q4-dense.qx \
  --v2-kernel-profile --context 512 --iterations 50
```

| component | estimated ms/token | share |
|---|---:|---:|
| RoPE + KV write | 0.857 | 29.9% of listed isolated sum |
| decode attention | 0.403 | 14.0% |
| fused QKV GEMV (normalized consumer) | 0.402 | 14.0% |
| Q4 LM head | 0.276 | 9.6% |
| O projection GEMV | 0.237 | 8.3% |
| router GEMV (normalized consumer) | 0.138 | 4.8% |
| top-k routing | 0.123 | 4.3% |
| final RMSNorm | 0.101 | 3.5% |
| embedding lookup | 0.096 | 3.4% |
| RMSNorm | 0.075 | 2.6% |
| post-attention RMS scalar | 0.063 | 2.2% |
| dynamic Q4 MoE (4-stage total) | 2.633 | separate 24-layer contribution |
| MoE/attention residuals | 0.092 | 3.2% |
| GPU argmax | 0.007 | 0.3% |

The latest listed isolated sum is 2.870 ms/token, while the complete graph
measures 5.809 ms/token GPU time. The MoE total is reported separately because
the stage profiler does not include it in the additive list; it contributes
2.633 ms/token at 24 layers. The bottleneck has moved away from attention
softmax/V and is now primarily MoE, RoPE/KV write, and the dense projections.

The canonical normalized MoE stage profile is:

| stage | GPU µs/layer | isolated ×24 |
|---|---:|---:|
| gate/up | 145.87 | 3.501 ms/token |
| SiLU | 3.33 | 0.080 ms/token |
| down | 73.17 | 1.756 ms/token |
| weighted accumulation | 3.79 | 0.091 ms/token |
| four-stage total | 226.17 | 5.428 ms/token |

These isolated multiplicities are intentionally non-additive with the complete
5.809 ms/token graph. They do establish that gate/up and down dominate the
MoE implementation; SiLU and accumulation are not worthwhile standalone
targets. The profiler now measures these stages through the same normalized
GPU-resident input path used by canonical V2.

### MoE weight-layout experiment

The dynamic MoE topology was held constant while the runtime repacked the
selected-expert weights into three representations: the artifact's current
row-major layout, ordinary 32-row Metal tiles, and a paired gate/up layout
that interleaves gate row `j` with up row `j`. The profiler counts the selected
top-8 Q4 payload, FP16 scales, and stage-local activation I/O.

| candidate | gate/up µs | gate/up GB/s | down µs | down GB/s | parity |
|---|---:|---:|---:|---:|---|
| current row-major | 230.04 | 20.67 | 64.75 | 37.20 | — |
| MetalTile32 / lane-native | 145.12 | 32.77 | 72.96 | 33.01 | PASS |
| paired gate/up | 235.00 | 20.24 | 65.75 | 36.63 | PASS |

The lane-native layout improves the isolated gate/up stage by 1.58x in this
sample, but the complete graph reverses that ranking. Three matched
context-512 runs produced:

| candidate | median complete tok/s | median GPU ms/token | CB/encoder/dispatch |
|---|---:|---:|---:|
| row-major | 177.91 | 5.173 | 1 / 1 / 437 |
| MetalTile32 | 163.93 | 5.662 | 1 / 1 / 437 |
| mixed: Tile32 gate/up + row down | 167.85 | 5.507 | 1 / 1 / 437 |

The live attention+MoE-only diagnostic showed the same direction in its
single matched sample: 196.78 core tok/s row-major, 182.84 mixed, and 174.03
MetalTile32. All three configurations pass the eight-position gate with
maximum logit error `0.009907`, minimum cosine `1.000000`, and argmax `8/8`.
The paired representation also passes the gate but is slower in the current
kernel because its physical interleave is not matched by a paired compute
mapping.

This makes row-major the measured whole-program winner and the canonical
choice. MetalTile32, paired, and mixed remain selectable with
`QX_V2_MOE_LAYOUT=tile32`, `paired`, and `mixed` for controlled experiments.
The isolated gate/up result is retained as a candidate-generation signal, not
as a promotion criterion. The complete-token run is still below 200 tok/s;
the 196.78 result is the attention+MoE-only core diagnostic and must not be
reported as complete-model throughput.

### Fresh canonical profile and llama.cpp MoE instrumentation

After selecting row-major at whole-program scope, three fresh release control
runs measured 177.08, 177.76, and 177.37 complete tok/s (median 177.37),
approximately 5.18 GPU ms/token, one command buffer, one encoder, and 437
dispatches/token. The newest isolated canonical profile ranked the remaining
work as follows:

| component | GPU µs/layer or dispatch | isolated ×24/token |
|---|---:|---:|
| MoE gate/up | 69.67 | 1.672 ms |
| RoPE + KV write | 104.71 | 2.513 ms |
| MoE down | 35.88 | 0.861 ms |
| fused QKV | 79.21 | 1.901 ms |
| decode attention | 36.92 | 0.886 ms |
| O projection | 17.00 | 0.408 ms |

The isolated contribution estimate is 9.697 ms/token and is intentionally
larger than the measured 5.184 ms/token because these are independent dispatch
medians without graph overlap or scheduling effects. The live ranking now
puts RoPE/KV, QKV, gate/up, attention, and down above the 0.5 ms/token
threshold.

An opt-in llama.cpp replay profiler was added to the Metal backend. Enable it
only for profiling with:

```bash
GGML_METAL_PROFILE_MOE=1 \
  external/llama.cpp/build-metal/bin/llama-bench \
  -m external/models/granite-3.0-1b-a400m-instruct-q4_k_m.gguf \
  -p 0 -n 1 -r 10 -b 1 -ub 1 -ngl -1
```

On the M5 Pro, the last ten decode samples had median replayed GPU timings of:

| llama.cpp path | median GPU time |
|---|---:|
| one gate/up matrix (`MUL_MAT_ID`) | ~13.3 µs |
| one down matrix (`MUL_MAT_ID`) | ~18.8 µs |
| all 48 gate/up + 24 down matrix nodes | ~1.09 ms/token |
| replayed `ffn_moe_*` node sum | ~1.91 ms/token |

llama.cpp uses two separate gate/up matrix nodes per layer, so its combined
gate/up matrix time is approximately 26.6 µs/layer. Against the fresh QX
profile, QX is about 2.6x slower on merged gate/up and 1.9x slower on down in
this isolated comparison. The full-node number is a replay sum with one
command buffer per tagged graph node, so it is a diagnostic upper-bound-style
comparison rather than llama.cpp's normal fused graph latency.

The same opt-in replay harness now covers the decode attention path. It was
run after a 511-token prompt with one generated token, using batch and
microbatch 512 for the prompt and 1 for decode:

```bash
GGML_METAL_PROFILE_DECODE=1 \
  external/llama.cpp/build-metal/bin/llama-bench \
  -m external/models/granite-3.0-1b-a400m-instruct-q4_k_m.gguf \
  -p 511 -n 1 -r 5 -b 512 -ub 512 -ngl -1
```

The steady-state medians over the five reported decode samples were:

| llama.cpp replay bucket | total/token | per-layer |
|---|---:|---:|
| QKV (`Qcur` + `Kcur` + `Vcur`) | ~362.9 µs | ~15.12 µs |
| RoPE + KV `SET_ROWS` writes | ~321.3 µs | ~13.39 µs |
| `FLASH_ATTN_EXT` | ~228.6 µs | ~9.52 µs |
| O projection | ~131.1 µs | ~5.46 µs |
| complete replayed transformer block | ~2.70 ms | ~112.2 µs |

Against the newest QX isolated profile, the directly comparable ratios are:

| component | QX/layer | llama/layer | QX/llama |
|---|---:|---:|---:|
| QKV | 79.21 µs | ~15.12 µs | ~5.2x |
| RoPE + KV | 104.71 µs | ~13.39 µs | ~7.8x |
| attention | 36.92 µs | ~9.52 µs | ~3.9x |
| O projection | 17.00 µs | ~5.46 µs | ~3.1x |

These are replay diagnostics rather than normal graph totals, but the ranking
is now clear: RoPE/KV is the largest measured primitive gap, followed by QKV;
attention is materially behind too, while the whole-block replay indicates
that composition and non-targeted operations still account for a significant
part of the end-to-end difference.

### RoPE/KV split topology promotion

The first RoPE/KV candidate replaces the serial one-dispatch kernel with two
fixed-shape dispatches: 512 threads compute the 16 Q heads' rotary pairs, and
256 threads rotate K directly into its cache slot while copying V directly to
the V cache. It is enabled canonically; set
`QX_V2_ROPE_KV_TOPOLOGY=legacy` to restore the old path.

The isolated profile improved from 104.71 to 6.46 GPU µs/layer. The full
0/1/7/31/63/127/255/511 gate remained unchanged, with max KV key error
0.009850, value error 0.001048, attention error 0.000127, logit error
0.009907, cosine 1.000000, and argmax agreement 8/8.

Repeated context-512 complete-token runs selected the split path:

| topology | sustained tok/s | GPU ms/token | dispatches/token |
|---|---:|---:|---:|
| legacy one-dispatch | 177.44, 177.12, 177.28 | 5.191, 5.186, 5.169 | 437 |
| split/direct-KV | 204.33, 205.46, 205.19 | 4.423, 4.431, 4.425 | 461 |

The split median is 205.19 tok/s, a roughly 15.7% end-to-end improvement,
while retaining one command buffer and one encoder. This is the first
validated path to clear the 200 tok/s milestone.

The first post-promotion release profile at 203.95 tok/s / 4.424 GPU
ms/token now ranks the remaining isolated contribution estimate as:

| component | GPU µs/layer | isolated contribution |
|---|---:|---:|
| fused QKV | 79.13 | 1.899 ms/token |
| MoE gate/up | 71.46 | 1.715 ms/token |
| decode attention | 47.88 | 1.149 ms/token |
| MoE down | 35.92 | 0.862 ms/token |
| O projection | 19.00 | 0.456 ms/token |
| RoPE + KV write | 14.25 | 0.342 ms/token |

Using the llama replay ratios, QKV is now the highest-value measured target;
attention and MoE remain significant, but QKV combines the largest remaining
primitive gap with a large live contribution. No QKV code has been changed by
the RoPE/KV promotion.

The resulting compiler rule is explicit: numerical validity gates a candidate,
microbenchmarks generate candidates, and complete-workload latency decides
promotion.

### Split gate/up topology experiment

The merged QX gate/up projection was tested as two separate 512-row SIMD
dispatches under the opt-in `QX_V2_MOE_SPLIT_GATE_UP=1` flag. The eight-position
sequential parity gate passed unchanged:

| metric | split result |
|---|---:|
| max KV key error | 0.009850 |
| max KV value error | 0.001048 |
| max attention error | 0.000127 |
| max absolute logit error | 0.009907 |
| minimum logit cosine | 1.000000 |
| argmax agreement | 8/8 |

The isolated MoE profile improved from 122.33 to 112.88 GPU µs/layer. The
complete context-512 workload nevertheless regressed consistently:

| topology | sustained tok/s | GPU ms/token | dispatches/token |
|---|---:|---:|---:|
| merged control | 177.08, 177.76, 177.37 | 5.177, 5.174, 5.177 | 437 |
| split gate/up | 175.36, 175.59, 175.89 | 5.224, 5.227, 5.227 | 461 |

The split path is therefore rejected as canonical despite its isolated kernel
win. It remains available as a diagnostic candidate; this is another direct
example that whole-program latency, not isolated GEMV time, selects a QX
execution topology.

### Topology experiment results

The canonical V2 graph uses scalar-only RMS→QKV, hierarchical argmax, and the
validated three-stage attention topology by default. Set
`QX_V2_RMS_SCALAR_QKV=0` or `QX_V2_ATTENTION_3_STAGE=0` to compare the older
paths. The topology harness measured:

### Asymmetric KV precision experiment

The strict 0/1/7/31/63/127/255/511 gate was run for all four storage choices.
The canonical FP32/FP32 cache is 50,331,648 logical bytes at context 512.
Each asymmetric variant is 37,748,736 bytes and all-FP16 is 25,165,824 bytes.

| K/V storage | strict parity | GPU ms/token | wall ms/token | tok/s |
|---|---|---:|---:|---:|
| FP32 / FP32 | PASS | 6.231 | 6.564 | 152.35 |
| FP32 / FP16 | PASS | 6.196 | 6.585 | 151.86 |
| FP16 / FP32 | PASS | 6.173 | 6.563 | 152.37 |
| FP16 / FP16 | FAIL: K max error 0.020439 | 6.210 | 6.584 | 151.88 |

The asymmetric modes save 25% of KV storage without changing the strict parity
result, but do not provide a meaningful end-to-end speedup. Process peak RSS
was allocator-sensitive across these short-lived benchmark processes, so it is
not treated as evidence of a clean RSS reduction; the logical KV-byte
reduction is the reliable memory result. The canonical runtime remains
FP32/FP32, while the asymmetric modes are opt-in via `QX_V2_KV_V_FP16=1` or
`QX_V2_KV_K_FP16=1`.

The promoted attention decomposition then measured, with the GQA-aware V and
hierarchical softmax enabled, approximately 4–16 µs for each isolated QK and
softmax stage, 12–44 µs for V, and 18–48 µs for live attention across repeated
context-512 release samples. These measurements are hardware-scheduler
sensitive; the promotion decision is based on the full graph result of 160.83
tok/s and the 0–511 parity gate, not on isolated timings alone.

| experiment | result |
|---|---:|
| 340 no-op dispatches, one command buffer | 450.62 GPU µs; 645.62 wall µs |
| attention, context 32 | 76.25 GPU µs/layer |
| attention, context 64 | 213.79 GPU µs/layer |
| attention, context 128 | 322.25 GPU µs/layer |
| attention, context 256 | 674.62 GPU µs/layer |
| attention, context 512 | 1,480.87 GPU µs/layer |
| 24-layer block, one command buffer | 9.639 GPU ms; 0.402 ms/layer |

The short-context attention samples are noisy, but the 256→512 increase is
clear enough to establish real KV-length scaling. The dispatch floor is not
large enough to explain the full token gap by itself. The layer-block curve
shows substantial amortization: a one-layer block measured 1.208 ms, while the
24-layer block measured 0.402 ms/layer. The controlled one-layer bundles were:

```text
A current one-layer graph       14 dispatches  375.33 GPU us
B norm + QKV + RoPE/KV           3 dispatches  150.42 GPU us
C attention + O + residual       3 dispatches   14.29 GPU us
D post-norm + router + top-k      3 dispatches  110.46 GPU us
```

These are topology measurements, not fused-kernel claims. The stage split below
retains this multi-position parity gate and is benchmark-only.

The benchmark-only attention decomposition gives the following latest release
sample (20 iterations per stage) for the old isolated reference-vs-stage
comparison:

| context | QK µs | softmax µs | V µs | isolated combined µs | live µs |
|---:|---:|---:|---:|---:|---:|
| 32 | 11.96 | 19.50 | 9.71 | 19.63 | 76.25 |
| 64 | 12.25 | 49.25 | 52.17 | 35.79 | 213.79 |
| 128 | 12.79 | 91.21 | 28.75 | 49.75 | 322.25 |
| 256 | 15.71 | 175.42 | 191.88 | 106.42 | 674.62 |
| 512 | 15.79 | 345.75 | 377.87 | 320.33 | 241.13 |

QK is not the dominant measured cost in this decomposition. Softmax and V
accumulation grow with context, while the combined isolated graph is much
cheaper than the old live reference attention. The latest context-512 split
also shows that the reduction and V stages are the useful optimization targets;
the promoted three-stage live attention remains 241.13 GPU µs in this sample.
The earlier promoted three-stage live sweep measured 62.92, 107.71, 196.00, 176.50, and 241.63 GPU µs at contexts
32, 64, 128, 256, and 512 respectively. QK logical traffic is 8,192
bytes/context-token and V traffic is 4,096 bytes/context-token; the harness
also reports normalized ns/context-token and effective GB/s.

The promoted eight-position same-artifact parity run reports:

```text
max absolute logit: 0.009914
minimum logit cosine: 1.000000
argmax agreement:    8/8
```

The tokenizer-backed generation path also runs prompt prefill and then
continues autoregressively. For `Hello` with two generated tokens it measured:

```text
prompt tokens:        1
TTFT/prefill:         54.577 ms
decode:               29.448 ms/token = 33.96 tok/s
last token:           30
```

The generated tail was `,,`; this is an execution result, not a quality claim.

V1 GPU accounting on the development fixture was approximately 20.25 ms
GPU time per token, 0.18 ms CPU encode time per token, and 34.97 ms CPU wait
time per token. The diagnostic CPU-dense mode is useful attribution evidence,
not a production runtime.

## Quality and representation evidence

Teacher-forced comparison is operational and remains V1-backed. On the 64-token
PTB development slice, Q3/Q4 versus uniform Q4 produced:

```text
delta NLL:       +0.057962
KL mean/median:  0.447214 / 0.272131
logit cosine:    0.934368
top-1 overlap:   68.75%
```

The Q2/Q4 680 MB artifact selected 518 Q2 and 1018 Q4 experts, reducing expert
storage from 679,477,248 to 577,634,304 bytes (14.99%). On the same short PTB
slice it produced a +57.42% PPL delta, +0.453768 teacher-forced NLL delta,
0.875607 logit cosine, and 51.56% top-1 overlap. It is therefore not a
quality-preserving V2 baseline.

## Gate status

| Iteration 4 gate | Result |
|---|---|
| Separate V2 ownership | PASS |
| Static buffer/dispatch plan | PASS |
| One-buffer encode boundary | PASS for the tested graph primitive |
| Fixed-shape V2 kernel family | PASS at Metal kernel level |
| One-layer attention graph with persistent KV | PASS |
| Real 24-layer Q4 attention graph | PASS |
| GPU router/top-k → dynamic Q4 MoE | PASS at core-stack level |
| Complete token graph through GPU argmax | PASS for artifact-only sequential seeded tokens |
| Hidden state GPU-resident through all 24 layers | PASS for seeded V2 token |
| Router/top-k GPU-resident | PASS in V2 core stack |
| KV permanently GPU-accessible | PASS in V2 core stack |
| Zero internal waits in V2 token graph | PASS; V1 still waits per operator |
| GPU greedy argmax | PASS as an isolated V2 primitive |
| V2 same-artifact Q4 logits parity | PASS: 8 positions through 511, 0.009914 max error, 8/8 argmax |
| Original FP32 cross-implementation parity | PASS: 24/24 layers, 100% router agreement, 1.000000 logit cosine, top-1 match |
| SIMD fused QKV | PASS: one QKV dispatch; parity preserved; ~16–32 µs isolated median |
| Kernel-only Q4 versus FP16 control | IN PROGRESS: FP16 is 1.2–5.9× faster in the current simple implementation; Q4 remains noisy and far below the FP16 bandwidth level |
| SIMD Q4 dense/expert controls | PASS: dense Q/O and MetalTile32 expert stages match scalar outputs; SIMD reduces isolated GPU time by ~4–8× |
| SIMD dense GEMV integrated into V2 | PASS: O/router/LM head use SIMD path; parity preserved |
| Multi-position KV/attention/logit parity | PASS: positions 0, 1, 7, 31, 63, 127, 255, 511; 8/8 argmax |
| Cooperative RMSNorm/attention controls | RETAINED AS EXPERIMENTS: full-model gate exposed regressions; live V2 uses validated scalar RMS, reference post-RMS, and three-stage attention |
| Hierarchical GPU argmax | PASS: two-dispatch reduction, isolated median ~7 µs |
| Four-stage batched SIMD MoE integrated into V2 | PASS: GPU-selected experts, four dispatches/layer, parity preserved |
| Three-stage attention integrated | PASS: parity-clean; 1,481→242 µs isolated context-512 attention |
| GQA-aware V partial/reduce integrated | PASS: 377.87→43.50 µs isolated V; full 0–511 parity; canonical |
| Hierarchical softmax integrated | PASS: 345.75→15.58 µs isolated softmax; full 0–511 parity; canonical |
| Scalar-only RMS→QKV integrated | PASS: parity-clean; normalized-vector writeback removed before QKV |
| Scalar-only RMS→router/MoE integrated | PASS: parity-clean; normalized-vector writeback removed before router and gate/up |
| FP16 KV experiment | RETAINED OFF: all-FP16 KV reaches 151.88 tok/s but fails the strict KV-key gate at 0.020439 |
| K-FP32/V-FP16 KV | PASS: 37,748,736 logical KV bytes; 151.86 tok/s; 0.009850 max K, 0.001049 max V, 0.009914 max logit, 8/8 argmax |
| K-FP16/V-FP32 KV | PASS: 37,748,736 logical KV bytes; 152.37 tok/s; same strict parity gate and 8/8 argmax |
| >=100 tok/s | PASS: stable context-512 complete graph is 160.83 tok/s |
| Matched llama.cpp comparison | PASS: direct release comparison captured; QX is about 50% of llama decode rate |

## Kernel-only viability gate

The next experiment is now isolated from the transformer runtime. The release
benchmark loads real layer-0 Q/O and expert gate/up/down weights from the QX
artifact and measures one dispatch at a time. The FP16 control stores the same
QX-dequantized weights in FP16, so it isolates Q4 packing/unpack overhead rather
than changing the model weights. Effective bandwidth includes weights, input,
and output bytes. The current implementation uses 256 threads per threadgroup,
8 SIMD groups, one output row per thread, and no threadgroup memory; register
pressure is not exposed by the current build.

```bash
target/release/qx-bench \
  external/models/granite-m5-pro-q4-dense.qx \
  --v2-kernel-bench --iterations 200
```

The Q-projection geometry sweep is now available separately:

```bash
target/release/qx-bench \
  external/models/granite-m5-pro-q4-dense.qx \
  --v2-kernel-sweep --iterations 20
```

It tests 32 combinations of 1/2/4/8 rows per SIMD group, 1/2/4/8 SIMD
groups per threadgroup, and direct versus threadgroup-cached input. The
release sweep's scalar baseline was 332.63 us. The best measured parameterized
mapping was 1 row/SIMD, 8 SIMD groups, cached input: 21.12 us median, 21.29 us
p95, CV 0.00, with a zero max-absolute-difference check against the validated
SIMD reference. The hand-specialized 1-row/8-group kernel remains slightly
faster at 18.83 us because it removes the sweep's runtime geometry loop.

The sweep result is strong evidence that SIMD mapping, not M5 DRAM bandwidth,
is the first-order issue. It clears the <60 us credibility gate and nearly
clears the <20–30 us serious-kernel target on Q projection without changing
the QX file layout. Input caching is not uniformly beneficial, so it must be
kept as a measured per-shape choice rather than assumed globally.

Current M5 Pro release result (median GPU time; p95 and CV show variability):

| shape | format | weight B | total B | GPU median us | p95 us | CV | effective GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| dense Q | Q4 | 589,824 | 598,016 | 143.79 | 333.50 | 0.33 | 4.16 |
| dense Q | Q4-SIMD control | 589,824 | 598,016 | 18.79 | 19.17 | 0.04 | 31.82 |
| dense Q | FP16 control | 2,097,152 | 2,101,248 | 57.96 | 58.38 | 0.01 | 36.25 |
| dense O | Q4 | 589,824 | 598,016 | 143.54 | 144.58 | 0.01 | 4.17 |
| dense O | Q4-SIMD control | 589,824 | 598,016 | 18.62 | 19.04 | 0.15 | 32.11 |
| dense O | FP16 control | 2,097,152 | 2,101,248 | 56.21 | 56.62 | 0.01 | 37.38 |
| expert gate/up | Q4 | 589,824 | 593,920 | 92.38 | 109.75 | 0.10 | 6.43 |
| expert gate/up | Q4-SIMD control | 589,824 | 593,920 | 12.13 | 12.67 | 0.06 | 48.98 |
| expert gate/up | FP16 control | 2,097,152 | 2,101,248 | 57.88 | 58.38 | 0.01 | 36.31 |
| expert down | Q4 | 294,912 | 297,984 | 38.83 | 41.37 | 0.07 | 7.67 |
| expert down | Q4-SIMD control | 294,912 | 297,984 | 7.75 | 7.96 | 0.05 | 38.45 |
| expert down | FP16 control | 1,048,576 | 1,051,648 | 29.63 | 30.08 | 0.01 | 35.50 |

This answers the first control question: simple FP16 is not 10× faster in
these measurements, but it is materially faster despite moving 3.5× more
weight bytes. The SIMD-cooperative Q4 control is the first meaningful result:
it reduces dense Q/O GPU time to about 18.8 us, roughly 7.6× faster than the
scalar Q4 path in the same run, using the same packed bytes. The Q4 kernel is
therefore compute/instruction/layout bound, not bandwidth-bound in the useful
The same SIMD mapping transfers to the `MetalTile32` expert stages: gate/up
falls from 92.38 us to 12.13 us and down from 38.83 us to 7.75 us, with both
outputs matching the scalar Q4 stage exactly in the benchmark correctness
check. The scalar Q projection also has high variance and needs a more stable
measurement before small comparisons are trusted. No per-kernel llama.cpp or
MPS measurement is claimed yet; the current llama number is still the
whole-model 323.58 tok/s baseline.

The immediate optimization gate is consequently to extend the SIMD-group
cooperation to expert gate/up, expert down, QKV, router, and LM head, then add
activation reuse and a lane-derived weight layout. Transformer fusion remains
frozen until those isolated shapes are measured.

## Remaining implementation order

The next required work is:

1. Add an optimized fixed-shape Q4 SIMD GEMV and compare it with the current
   one-row-per-thread kernel using the same four shapes.
2. Instrument llama.cpp's equivalent Metal Q4 kernels and add an MPS/Apple
   baseline if a supported bridge is worthwhile.
3. Propagate the winning Q4 kernel architecture to fused QKV and the dynamic
   MoE/LM execution paths, then reprofile full-model GPU time toward the 100
   tok/s gate.
4. Attribute RSS, convert KV FP32 to FP16, and run the matched long-fixture
   quality comparison.
5. Investigate output/residual fusion only where the new profile justifies it;
   promote V2 to `qx-generate` last.

Until those steps are complete, mixed-precision Pareto claims should remain
paused. The current data supports a complete, quality-validated fused-QKV V2
token graph and a logical full-Q4 resident estimate. It does not meet the 100
tok/s performance target, and the fused path currently has a 1.60 GB token-
graph peak RSS.

## When throughput becomes meaningful

We can already compare V1 and llama.cpp: QX V1 is ~28.3 tok/s, the
CPU-dense/GPU-MoE diagnostic is ~37.6 tok/s, and llama.cpp is ~310.7 tok/s on
the matched fixture. V2 now measures 152.35 complete release-build tok/s on
the promoted scalar-RMS/router/three-stage artifact-only token graph. The direct release comparison above provides the
matched V2-versus-llama.cpp throughput baseline; the V2 benchmark also
reports tok/s, TTFT, GPU ms/token, command buffers/token, and peak RSS.

## Verification

The workspace currently passes:

```text
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo test --release --workspace --all-targets
```

On the target Mac, the V2 Metal argmax test passes alongside the existing Q2,
Q3, Q4 expert and logits dispatch tests.

## Latest canonical result: split/direct RoPE-KV and QKV topology test

The validated canonical context-512 artifact-only V2 path is now:

```text
throughput:       205.19 tok/s representative run
GPU time:         4.425 ms/token representative run
wall time:        ~4.90 ms/token
command buffers:  1/token
encoders:         1/token
dispatches:       461/token
RSS:              ~1.91 GB peak
```

The gain from the promoted RoPE/KV topology came from two dispatches per layer:
parallel Q RoPE, plus K RoPE directly into the final K-cache destination and V
directly into the final V-cache destination. It passed the complete checkpoint
gate at positions `0, 1, 7, 31, 63, 127, 255, 511` with maximum KV-key error
`0.009850`, maximum KV-value error `0.001048`, maximum attention error
`0.000127`, maximum logit error `0.009907`, cosine `1.000000`, and argmax
agreement `8/8`.

The next controlled experiment tested separate SIMD Q, K, and V projections
against the fused QKV control. It reused the existing validated row-major SIMD
Q4 GEMV ABI, with exact Granite shapes `1024`, `512`, and `512`, and was kept
behind `QX_V2_QKV_TOPOLOGY=split`.

| QKV topology | QKV isolated GPU median | full context-512 result | dispatches/token | decision |
|---|---:|---:|---:|---|
| fused control | ~79 us/layer | 204.44 / 204.17 tok/s, 4.423 / 4.430 ms GPU | 461 | canonical |
| separate Q/K/V | 99.54 us/layer | 198.47 / 197.64 tok/s, 4.558 / 4.577 ms GPU | 509 | reject |

The split path passed the same full sequential parity gate (`0.009850` max
KV-key error, `0.001048` max KV-value error, `0.000127` max attention error,
`0.009907` max logit error, cosine `1.000000`, argmax `8/8`), but it was about
`3.1%` slower end-to-end on the repeated pair and about `3.2%` slower in GPU
time. It remains useful as an experiment and rollback comparison, but it is
not promoted. This is
another whole-program rejection of a locally plausible topology: three
shape-specialized GEMVs did not compensate for their added scheduling and
dispatch cost.

## Fused-QKV geometry sweep

The next experiment stayed inside one dispatch and varied Q-region rows per
SIMD group, KV-region rows per SIMD group, SIMD groups per threadgroup, and
input caching. The benchmark-only kernel was checked against the fused
production output before timing candidates.

The best isolated result was:

```text
Q rows/SIMD:       2
KV rows/SIMD:      1
SIMD groups:       8
threads/threadgroup: 256
input cache:       enabled
GPU median:        20.54 us
fused control:     ~78.96 us
max output error:  0.000000
```

This candidate was enabled experimentally with
`QX_V2_QKV_GEOMETRY=q2-kv1-g8`. It passed the complete 0–511 parity gate and
kept the canonical `461 dispatches/token`, but the live graph rejected it:

```text
geometry:  204.25 / 201.25 tok/s, 4.498 / 4.497 ms GPU/token
control:   ~204.17 tok/s, ~4.430 ms GPU/token
```

The live component profile also measured the geometry candidate's QKV stage at
`93.25 us`, versus the earlier fused control near `79 us`. It is therefore
retained as benchmark training data but not promoted. This is a particularly
clear instance of the QX compiler rule: an isolated kernel win is only a
candidate; the complete compiled graph remains the optimization objective.

## QKV isolation/live boundary replay

To explain that inversion, `qx-bench --v2-qkv-boundary` now replays the actual
scalar-RMS → normalized-QKV → canonical split/direct-RoPE-KV boundary using
real layer tensors. It measures both one-layer and 24-layer replays; the latter
is the meaningful scheduling/cache comparison. The geometry candidate also
supports `q2-kv1-g8-nocache` for an apples-to-apples cache test.

At context 512, the 24-layer results were:

| Replay | fused control | q2-kv1-g8 cache on | q2-kv1-g8 cache off |
|---|---:|---:|---:|
| RMS → QKV | 18.25 us/layer | 20.98 us/layer | 26.10 us/layer |
| RMS → QKV → RoPE/KV | 20.53 us/layer | 23.45 us/layer | 28.54 us/layer |

The cache is beneficial for the geometry candidate, but does not recover its
live loss. The candidate therefore remains rejected. This establishes the next
autotuning requirement: cheap standalone sweeps may generate candidates, but
QKV candidates must also pass a producer/consumer replay and a multi-layer
replay before any full-model promotion attempt.

## QKV interference ladder

The next replay reconstructed the real layer progressively and appended a
24-layer RMS→QKV probe in the same command buffer. The probe delta is the
incremental GPU time attributable to that appended QKV workload after the
preceding rung, divided by 24 layers.

At context 512, a 30-iteration release run produced:

| Rung | Graph GPU us | Probe GPU us | Probe delta/layer |
|---|---:|---:|---:|
| RMS → QKV → RoPE/KV | 640.04 | 931.25 | 12.13 us |
| + attention + O + residual | 1301.21 | 1722.87 | 17.57 us |
| + router + top-k | 1555.00 | 1978.00 | 17.62 us |
| + complete MoE | 4292.50 | 4730.08 | 18.23 us |

The 10-iteration confirmation was similar (`18.09`, `18.50`, `24.69`, and
`19.22 us/layer` respectively, with expected short-run variance). Complete
MoE does not inflate QKV to the previously attributed `79–93 us/layer`.
Therefore those isolated “QKV stage” values are not direct live QKV costs;
they are command-buffer attribution measurements whose execution context is
not comparable to the in-graph probe. Further QKV geometry search is paused.

The new benchmark is:

```bash
target/release/qx-bench artifact.qx \
  --v2-qkv-interference --context 512 --warmup 3 --iterations 30
```

## Realistic differential profile

Because the QKV interference ladder showed that isolated stage attribution can
be misleading, `qx-bench --v2-differential-profile` now measures marginal GPU
time in the complete token graph. The control is the canonical
`encode_token` graph. Each candidate replaces one major block with no-op
dispatches while retaining the surrounding graph and, where possible, the
same dispatch count. The reported saving is therefore a full-program
ablation delta, not an additive isolated-kernel estimate.

The first release run used the dense Q4 artifact at context/position `512/511`
with 5 warmups and 40 measured iterations:

| Variant | GPU us/token | Dispatches | Marginal saving |
|---|---:|---:|---:|
| canonical complete graph | 5,189.12 | 461 | — |
| without input RMS/QKV | 4,762.58 | 461 | 426.54 us |
| without RoPE/KV | 5,149.25 | 461 | 39.87 us |
| without attention | 4,680.46 | 461 | 508.67 us |
| without O/residual | 4,985.08 | 461 | 204.04 us |
| without router/top-k | 4,883.96 | 461 | 305.17 us |
| without MoE | 2,061.75 | 461 | 3,127.38 us |
| without final head | 4,806.87 | 461 | 382.25 us |

This changes the immediate priority: the live MoE block is the dominant
marginal target, followed by attention, input RMS/QKV, and the final head.
RoPE/KV is now a small marginal target in this full-graph measurement despite
its earlier isolated competitive gap. The deltas are intentionally not added
together: ablation changes cache residency and dependency scheduling. A
replacement must still be measured on the complete graph and pass the
sequential parity gate.

The command is:

```bash
target/release/qx-bench external/models/granite-m5-pro-q4-dense.qx \
  --v2-differential-profile --context 512 --warmup 5 --iterations 40
```

## Live MoE stage differential profile

The full-graph MoE ablation was decomposed without changing routing. A setup
pass captures each layer's canonical top-8 IDs and router weights into separate
GPU-resident trace storage. Every measured variant then reuses those exact
routes, so a gate/up or down ablation cannot change later-layer expert
selection. The measured graph still runs the router/top-k work, but the
captured trace is the input to MoE.

At context/position `512/511`, with the same dense Q4 artifact and 30 release
iterations:

| Variant | GPU us/token | Dispatches | Marginal saving |
|---|---:|---:|---:|
| canonical MoE | 5,172.54 | 461 | — |
| without gate/up | 3,152.50 | 461 | **2,020.04 us** |
| without SiLU | 5,139.12 | 461 | 33.42 us |
| without down | 4,087.46 | 461 | **1,085.08 us** |
| without accumulation | 5,169.67 | 461 | 2.87 us |

The stage savings sum to approximately `3.14 ms`, consistent with the earlier
full-MoE marginal result while retaining the full graph's routing and dispatch
shape. Gate/up is the dominant live MoE target, down is second, and SiLU plus
weighted accumulation should be frozen.

The reproducible command is:

```bash
target/release/qx-bench external/models/granite-m5-pro-q4-dense.qx \
  --v2-moe-differential-profile --context 512 --warmup 5 --iterations 30
```

The report prints artifact byte size, a streaming artifact fingerprint,
context/position, and the active QKV/RoPE/attention/RMS topology flags. The
fingerprint is a reproducibility stamp, not a cryptographic SHA-256 digest.

The same fixed-route run also decomposes gate/up into traffic and arithmetic
controls:

| Gate/up variant | GPU us/token | Dispatches | Saving vs control |
|---|---:|---:|---:|
| canonical gate/up | 5,194.88 | 461 | — |
| weight-touch only | 4,998.50 | 461 | 196.37 us |
| synthetic arithmetic | 3,905.37 | 461 | 1,289.50 us |
| gate/up no-op | 3,106.96 | 461 | 2,087.92 us |

The correct comparison for the controls is against the gate/up no-op. The
weight-touch control costs approximately `4.99850 - 3.10696 = 1.89 ms` above
the no-op, despite doing no dot products. The synthetic arithmetic control
costs approximately `3.90537 - 3.10696 = 0.80 ms` above the no-op. Therefore
selected-expert Q4 payload/scale traffic and its cache/residency behavior are
the dominant resource, with arithmetic and unpack/reduction work secondary.
The next gate/up branch should focus on expert representation, residency,
placement, and reducing bytes touched before inventing another GEMV geometry.
The matched live llama gate/up differential measurement remains the next
competitive comparison.

The control kernels are diagnostic only and are not runtime candidates. Their
results are not additive; they isolate resource classes approximately and
must be confirmed with a real replacement plus the full parity and sustained
throughput gates.

The longer 30-iteration traffic ladder was:

| Gate/up control | GPU us/token | Cost above gate/up no-op |
|---|---:|---:|
| selected full payload + scales | 5,004.00 | 1,929.25 us |
| selected payload only | 4,976.71 | 1,901.96 us |
| selected scales only | 4,681.75 | 1,607.00 us |
| contiguous equal-byte stream | 4,740.58 | 1,665.83 us |
| synthetic arithmetic | 3,909.08 | 834.33 us |
| gate/up no-op | 3,074.75 | — |

The equal-byte contiguous stream is about `0.26 ms` cheaper than selected
expert traversal, quantifying a locality/placement penalty. The larger result
is that both are expensive relative to no-op: byte volume is the primary
resource, with scattered expert locality as a secondary penalty. Payload-only
is much more expensive than scales-only, so reducing Q4 payload bytes is the
first representation lever; scales are not free but are not the dominant
traffic component.

The route-reuse diagnostic ran eight sequential artifact-only positions. Its
mean consecutive top-8 set overlap was `77.68%`, ranging from `69.64%` to
`89.29%` across layers. This is high enough that route-aware placement or
hot-expert residency is worth testing, especially in layers 22–23, while the
lower-overlap layers may benefit more directly from payload compression.

## Fractional gate/up traffic sweep

The traffic controls were extended to touch 25%, 50%, 75%, or 100% of each
selected expert's payload while keeping the full dispatch geometry unchanged.
The same fractions were tested against a contiguous equal-byte buffer. The
30-iteration release run produced the following marginal costs above the
gate/up no-op (`3,060.62 us/token`):

| Fraction | Selected payload | Contiguous equal-byte |
|---:|---:|---:|
| 25% | 1,755.30 us | 1,523.13 us |
| 50% | 1,997.84 us | 1,569.46 us |
| 75% | 2,221.88 us | 1,607.75 us |
| 100% | 2,456.13 us | 1,646.17 us |

A linear fit over the fractional controls gives approximate models:

```text
selected payload     ≈ 1.53 ms fixed + 0.93 ms × fraction
contiguous stream    ≈ 1.48 ms fixed + 0.16 ms × fraction
```

The large intercept means this is not simply a bytes × bandwidth problem:
routed memory-region setup, cache-line behavior, and stream/scheduling costs
are substantial. The selected-vs-contiguous gap also grows with the fraction,
so physical locality remains a real secondary opportunity. The 100% selected
point in this fractional kernel is diagnostic; the specialized full-touch
control remains the preferred same-bytes selected-expert reference.

```bash
target/release/qx-bench external/models/granite-m5-pro-q4-dense.qx \
  --v2-route-overlap --context 512 --iterations 8
```

## Constant-byte routed-region sweep

The next diagnostic held the nominal traffic volume and dispatch geometry
constant while varying the number of routed expert regions touched by the
control kernel. The release run used 15 measured iterations and compared each
case with the same gate/up no-op baseline (`3,077.29 us/token`):

| Routed regions | GPU us/token | Cost above no-op |
|---:|---:|---:|
| 1 | 5,257.71 | 2,180.42 us |
| 2 | 5,489.71 | 2,412.42 us |
| 4 | 5,666.17 | 2,588.88 us |
| 8 | 5,666.13 | 2,588.84 us |

The directional result is a fragmentation penalty: spreading the same
nominal traffic over more routed regions increases the cost, with most of the
increase appearing by four regions. These controls still share a diagnostic
address-selection path, so this is not yet a final artifact-layout decision.
The next clean version should use compile-time-specialized kernels for each
region count before promoting a placement or co-location strategy.

The reproducible command is:

```bash
target/release/qx-bench external/models/granite-m5-pro-q4-dense.qx \
  --v2-moe-differential-profile --context 512 --warmup 5 --iterations 15
```

## Larger route co-selection sample

The route-overlap diagnostic was extended to 32 sequential artifact-only
positions. Across the 31 transitions, the mean consecutive top-8 set overlap
was `90.41%`, with a per-layer range of `83.06%` to `96.37%`. This confirms
strong route stability, but the reuse distance is still approximately one
24-layer token, so overlap alone does not prove hardware-cache residency.

The most frequently observed co-selected pairs in this sample included:

| Count | Layer | Expert pair |
|---:|---:|---:|
| 32 | 23 | 20 + 30 |
| 32 | 1 | 8 + 13 |
| 32 | 1 | 8 + 19 |
| 32 | 5 | 8 + 30 |
| 32 | 2 | 24 + 26 |

This makes co-selection-based physical permutation a reasonable secondary
experiment after the constant-byte control is cleaned up. The compiler should
use the co-selection matrix to propose expert permutations, then validate
them with the fixed-route differential profile and the complete context-512
parity/throughput gates.

```bash
target/release/qx-bench external/models/granite-m5-pro-q4-dense.qx \
  --v2-route-overlap --context 512 --iterations 32
```

No Q3 representation or expert placement has been promoted from these
diagnostics. The next representation experiment should first establish the
traffic slope with clean region-specialized controls, then compare a
hardware-native Q3 candidate against the canonical Q4 artifact. A matched
local llama.cpp gate/up measurement is still unavailable in this workspace.

## Direct-address region diagnostic

The region-count control was followed by four compile-time-specialized Metal
pipelines for `1`, `2`, `4`, and `8` regions. The host resolves the captured
fixed routes once and binds each selected expert's packed and scale base
directly; the kernels do not read expert IDs or perform routed address lookup.
Each region is repeated as needed to preserve the same total eight-expert byte
volume.

The Q3 physical encoding was also checked against the CPU `QuantizedTensor`
reference for both dense and padded forms. Removing the four padding bytes
from the native form reproduces the dense code stream and identical CPU
matvec results, so the large model-level Q3 error is not caused by the padded
representation itself.

## Active-k MoE sweep

Uniform Q3 was not the next lever. The route corpus instead showed a useful
router-confidence signal: mean cumulative normalized mass was approximately
`0.853` at rank 6, `0.931` at rank 7, and `1.000` at rank 8. An opt-in
GPU-resident `qx_v2_topk_mask` stage now masks the tail of the canonical top-8
selection, renormalizes the surviving weights, and marks inactive expert IDs
invalid. The existing four-stage MoE graph then skips those experts without
changing its topology. Enable it with `QX_V2_MOE_ACTIVE_K=4..8`.

The initial context-512 complete-token sweep used the canonical Q4 artifact:

| active k | GPU ms/token | equivalent tok/s | dispatches/token |
|---:|---:|---:|---:|
| 8 | 4.814 | 191.14 | 461 |
| 7 | 4.651 | 194.68 | 485 |
| 6 | 4.448 | 204.09 | 485 |
| 5 | 4.056 | 217.11 | 485 |
| 4 | 3.576 | 249.36 | 485 |

This confirms that active expert count has materially more leverage than the
Q3 representation branch. It is not yet a quality result: the k=6 strict
comparison produced cosine `0.998867`, logits max error about `0.598`, and
argmax parity through the sampled positions `0, 1, 7, 31, 63, 127, 255`.
The argmax remained token 34 in that trace, but the exact Q4 reference gate
does not pass. k=4 also produced visibly unstable generated output and is
rejected pending quality measurement.

The route-overlap profiler now reports cumulative router mass as well as route
reuse. The next promotion decision is active-k quality measurement:
teacher-forced NLL/PPL and long held-out argmax/top-k behavior for k=7 and k=6,
followed by a layer-specific or cumulative-mass policy. No active-k value is
canonical yet; the default remains top-8.

The first teacher-forced V2 comparison used 128 PTB positions against the
canonical top-8 GPU graph:

| candidate | NLL delta | PPL delta | KL mean | logit cosine | top-1 agreement |
|---:|---:|---:|---:|---:|---:|
| k=7 | +0.035291 | +3.59% | 0.111240 | 0.983322 | 82.03% |
| k=6 | +0.084964 | +8.87% | 0.359865 | 0.949744 | 67.19% |

These short results do not meet the proposed quality bands. The GPU active-k
implementation itself is correct: when the CPU oracle is configured with the
same active-k policy, k=6 passed all eight sequential checkpoints through
position 511 with cosine `1.000000`, max logit error `0.008861`, and 8/8
argmax agreement. The remaining difference is therefore intentional model
quality loss, not a top-k mask bug.

The runtime also accepts
`QX_V2_MOE_ACTIVE_K_BY_LAYER=k0,k1,...,k23`, which applies a static
layer-specific active-k policy and takes precedence over the global setting.
The first 128-token sensitivity sweep used `k=6` in one layer at a time, with
all other layers left at `k=8`:

| reduced layer | NLL delta | PPL delta | logit cosine |
|---:|---:|---:|---:|
| 8  | -0.038829 | -3.81% | 0.996714 |
| 12 | -0.003324 | -0.33% | 0.997532 |
| 20 | +0.012377 | +1.25% | 0.998193 |
| 23 | +0.042148 | +4.30% | 0.994280 |

These are screening measurements rather than a final quality claim, but layer
12 is the only tested candidate inside the proposed `<=0.5%` PPL budget. A
layer-12-only policy passed the full sequential GPU-vs-active-k CPU oracle gate
at positions `0, 1, 7, 31, 63, 127, 255, 511`: max KV key error `0.009850`,
max attention error `0.000126`, max logit error `0.009882`, cosine `1.000000`,
and 8/8 argmax agreement.

The corrected layerwise implementation emits the mask dispatch only for layers
whose configured k is below 8. Thus the layer-12 policy costs 462 dispatches
per token versus 461 for canonical top-8, rather than adding 24 no-op
dispatches. Five 50-iteration context-512 runs measured approximately
`185.18 tok/s` canonical versus `184.72 tok/s` for the layer-12 policy, with
overlapping run variance. It is therefore a quality-qualified experiment, not
a promoted performance win; canonical top-8 remains the default.

A 30-iteration release run gave the following costs above the same gate/up
no-op (`3,052.29 us/token`):

| Direct regions | GPU us/token | Cost above no-op |
|---:|---:|---:|
| 1 | 4,573.17 | 1,520.88 us |
| 2 | 4,800.08 | 1,747.79 us |
| 4 | 4,885.08 | 1,832.79 us |
| 8 | 5,043.63 | 1,991.34 us |

This produces a direct-address `8 - 1` spread of `470.46 us/token`, much
closer to a real fragmentation/locality signal than the old dynamic-ID
controls. Repeated 15- and 60-iteration runs measured `277 us` and `152 us`
respectively, so the full-token profiler still has substantial GPU timing
variance. The result is therefore directional rather than a promoted layout
decision. The direct kernels also use different numbers of bound expert
resources, which remains a smaller possible confound.

The direct controls are now exposed alongside the original controls by:

```bash
target/release/qx-bench external/models/granite-m5-pro-q4-dense.qx \
  --v2-moe-differential-profile --context 512 --warmup 8 --iterations 30
```

The next measurement should repeat the direct-only comparison with a stable
sampling order and, if the spread persists near `0.3--0.4 ms/token`, test the
zero-quality-risk co-selection permutation. No expert permutation or Q3
artifact has been promoted yet.

## Expert execution-order experiment

Before changing physical expert IDs in the artifact, the fixed routes were
replayed twice: once in canonical top-k order and once sorted by physical
expert ID, with each expert's router weight kept attached to its ID. This is a
zero-quality-risk locality probe because it changes only the order of the
commutative expert contributions.

| Run | Canonical order | ID-sorted order | Sorted minus canonical |
|---|---:|---:|---:|
| 20 iterations | 5,045.12 us | 5,062.62 us | +17.50 us |
| 40 iterations | 5,064.12 us | 5,049.29 us | -14.83 us |

The result is neutral within measurement variance; there is no evidence that
sorting the selected experts by their current physical IDs improves the live
MoE path. It is therefore rejected as a canonical change. Co-selection-aware
physical permutation remains a separate experiment because it changes the
expert-to-address mapping itself rather than only the traversal order.

The route-overlap command now also reports selection frequency and same-expert
next-transition persistence. In the 32-position sample, the mean set overlap
remained `90.41%`; several frequently selected layer/expert pairs were present
in all 32 positions, and the highest same-expert persistence entries were
`31/32 = 96.88%`. These statistics are useful inputs to a future placement
search, but the sample is still too short to justify artifact replication or
permanent physical reordering.

## Route prefetch experiment

The first residency experiment compared three complete-token variants:

| Variant | GPU us/token | Dispatches | Delta vs canonical |
|---|---:|---:|---:|
| canonical | 5,065.50 | 461 | — |
| previous-route prefetch | 6,733.17 | 485 | +1,667.67 us |
| oracle-route prefetch | 6,545.12 | 485 | +1,479.63 us |

The prefetch dispatches were placed before each layer's router work and the
measurement includes their full cost. Even a perfect current-route oracle is
substantially slower, so serialized direct-address touching is not providing
useful residency on this command stream. Previous-token prediction is rejected
as a runtime optimization; no further prefetch work is justified until a
genuinely overlapping Metal mechanism is available.

```bash
target/release/qx-bench external/models/granite-m5-pro-q4-dense.qx \
  --v2-moe-prefetch-profile --context 512 --warmup 5 --iterations 15
```

## Static Metal resource experiment

Static model inputs normally use `StorageModeShared`. An opt-in diagnostic,
`QX_METAL_PRIVATE_STATIC=1`, allocates those inputs as `StorageModePrivate`
and performs a one-time shared-to-private blit during preparation. The live
graph remains unchanged and still uses one command buffer per token.

At context/position `512/511`, the 10-iteration samples were:

| Resource mode | Canonical GPU us/token |
|---|---:|
| Shared | 5,041.12 us |
| Private via initialization blit | 5,084.00 us |

This is not an improvement and is within the broader timing variance, with the
private mode slightly slower in this sample. The private-storage option is
therefore diagnostic only and the canonical shared representation remains
active. Heaps and cluster-sized buffers are not justified yet without a
stronger signal from this basic resource-mode comparison.

```bash
QX_METAL_PRIVATE_STATIC=1 target/release/qx-bench \
  external/models/granite-m5-pro-q4-dense.qx \
  --v2-moe-differential-profile --context 512 --warmup 4 --iterations 10
```

## Dedicated direct-address fragmentation measurement

The broader differential profile interleaved many controls and showed timing
variance, so the direct-address comparison was rerun in a dedicated profiler
containing only a gate/up no-op and the four fixed pipelines. The 60-iteration
release run was:

| Variant | GPU us/token | Cost above no-op |
|---|---:|---:|
| gate/up no-op | 3,053.50 us | — |
| direct 1 region | 4,567.75 us | 1,514.25 us |
| direct 2 regions | 4,591.79 us | 1,538.29 us |
| direct 4 regions | 4,661.75 us | 1,608.25 us |
| direct 8 regions | 4,826.96 us | 1,773.46 us |

The clean direct-address spread is therefore `259.21 us/token` from one to
eight regions. This is smaller than the contaminated dynamic-region result,
but it survives after removing expert-ID lookup and variable region selection.
Physical expert placement is worth testing, though the expected gain is now
bounded and should not be confused with the main byte-volume cost.

```bash
target/release/qx-bench external/models/granite-m5-pro-q4-dense.qx \
  --v2-moe-direct-region-profile --context 512 --warmup 10 --iterations 60
```

## Physical expert permutation measurement

The runtime can now load a per-layer physical permutation while the benchmark
remaps fixed logical route IDs through the inverse permutation. Three plans
were tested using a 32-position route corpus: original order, descending
selection-frequency order, and greedy co-selection clustering. This changes
actual expert storage order rather than just sorting the execution sequence.

At context/position `512/511`, with 15 measured iterations:

| Layout | Full GPU us/token | Gate/up marginal | Down marginal |
|---|---:|---:|---:|
| original | 5,069.50 us | 2,011.08 us | 979.54 us |
| frequency ordered | 5,074.54 us | 2,012.00 us | 983.37 us |
| co-selection clustered | 5,070.46 us | 2,011.58 us | 983.00 us |

Neither permutation produced a repeatable improvement. The co-selection plan
is effectively neutral, while frequency ordering is slightly worse in this
sample. These are fixed-route full-graph measurements; a production
permutation still requires GPU router-ID remapping plus the full sequential
parity gate before promotion.

No physical permutation has been promoted. Q3 remains the next high-upside
branch, but the existing Q3 artifacts and kernels are not treated as the new
traffic-native design; the next candidate must be measured on live gate/up,
down, full-MoE, full-token, and quality metrics together.

## Traffic-native Q3 gate/up experiment

The next representation branch kept the canonical Q4 artifact and all
non-gate/up tensors unchanged. An opt-in source artifact was supplied through
`QX_V2_MOE_Q3_ARTIFACT`, with `QX_V2_MOE_Q3_LAYOUT=dense|native` selecting the
physical block size:

| representation | Q3 code bytes/group | scale bytes/group | physical block | selected top-8 bytes/layer |
|---|---:|---:|---:|---:|
| canonical Q4 | 16 | 2 | 18 logical bytes | 4.50 MiB |
| dense Q3 | 12 | 2 | 14 logical bytes | 3.50 MiB |
| padded/native Q3 | 12 + 4 padding | 2 | 18 physical bytes | 4.50 MiB |

The implementation is a gate/up-only dynamic SIMD kernel. It consumes the
Q3 codes directly, supports both 12-byte and 16-byte group strides, and keeps
Q4 down projections, routing, attention, dense projections, KV, and the
canonical one-command-buffer graph unchanged. The default loader remains Q4.

At context/position `512/511`, five-iteration complete-token measurements were:

| mode | GPU ms/token | wall ms/token | equivalent tok/s | dispatches | peak RSS |
|---|---:|---:|---:|---:|---:|
| Q4 control | 5.021 | 5.484 | 182.35 | 461 | 1.91 GB |
| dense Q3 gate/up | 5.335 | 5.785 | 172.87 | 461 | 2.56 GB |
| padded/native Q3 gate/up | 4.958 | 5.554 | 180.04 | 461 | 2.76 GB |

These are short candidate runs and should not replace the established
canonical benchmark, but they are sufficient to reject dense Q3 as an
immediate performance promotion. The padded representation has no byte-volume
advantage over Q4 and did not produce a repeatable end-to-end win.

The strict V2 parity gate also failed for dense Q3 immediately:

```text
position 0: KV key max error 3.054429
            attention max error 0.111626
            logits max error 7.186561
            cosine 0.903171
            argmax mismatch (CPU 34, GPU 641)
```

The same failure persisted at the sampled early positions. This is a
quality-preserving optimization failure, not a reference-parity pass, so no
Q3 mode is promoted. The Q3 branch remains useful infrastructure for a future
route-weighted or selectively allocated format, but the next step should not
be a uniform Q3 rollout. In particular, the padded/native form should be
closed unless a future kernel demonstrates a benefit independent of byte
volume.

## Canonical-path audit corrections

A final architecture audit found that the implementation had outrun the
user-facing and benchmark entry points. Three corrections are now canonical:

1. `qx-generate` selects artifact-only Runtime V2 by default; `--runtime v1`
   retains the original reference path.
2. V2 prompt generation now returns the prediction produced by the final
   prefill token as the first generated token. The previous benchmark-only
   loop evaluated that prediction before recording it, shifting the reported
   continuation by one token.
3. `--v2-token` consumes `benchmarks/granite_decode_512.json`, uses its seed,
   context, and generation count, and warms position zero without advancing
   the measured sequential context. It also stamps artifact path/size/hash and
   reports current RSS separately from peak RSS.

The corrected release command is:

```bash
target/release/qx-bench \
  external/models/granite-m5-pro-q4-dense.qx \
  --v2-token --fixture benchmarks/granite_decode_512.json --warmup 3
```

Three audit runs measured a median `184.58 tok/s`, `4.987 ms GPU/token`, and
`5.418 ms wall/token`, with one command buffer, one encoder, and 461
dispatches/token. Median current RSS was `1,214,414,848` bytes while median
peak RSS was `1,891,237,888` bytes. The
large difference confirms that peak RSS is not a steady-state residency
measurement; it likely includes artifact pages faulted while persistent Metal
weight buffers are prepared. Historical 205 tok/s measurements remain useful
observations but are not the current canonical control until reproduced under
this fixture-corrected path.

The fresh same-session llama.cpp control used build commit `1719747`, seed
token `8279`, context 512, and 128 generated tokens. It reported `310.56 tok/s`
end-to-end (`314.57 tok/s` model eval), `3.18 ms/eval`, and `916,668,416` bytes
peak RSS. Under these exact runs QX reaches 59.4% of llama's end-to-end decode
rate. Retained 319--324 tok/s llama-bench runs demonstrate normal benchmark
variation and remain useful as the upper comparison band.
