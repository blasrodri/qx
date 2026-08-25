# Qwen component-runtime parity

Date: 2026-08-25  
Artifact: `qwen3-30b-a3b-a19pro-q2s4-capacity-candidate.qx`  
Artifact SHA-256: `7262a69d8230f0f78dcd12266035fedff5e9145dcace504c93fc27ccadef91c4`

## Result

QX now executes every mathematical component needed by one
Qwen3-30B-A3B decoder layer on Metal from the sealed artifact:

- Q4 Q/K/V projection with the 2,048-to-4,096 query expansion;
- per-head Q/K RMSNorm and split-half RoPE at nonzero positions;
- 32-query-head / 4-KV-head grouped-query attention over an FP16 KV history;
- Q4 output projection;
- post-attention RMSNorm and 128-way Q4 router projection;
- normalized top-8 expert selection;
- eight Q2S4 gate/up-SwiGLU-down experts and GPU weighted accumulation.

The Q2S4 expert kernel is Qwen-shaped rather than a renamed Granite kernel.
Eight 768-thread groups execute the GPU-selected experts directly from 64-bit
offsets into the mapped weight slab, then one dispatch accumulates their
outputs. Routing and expert weights never cross the host boundary.

`PreparedQwenLayer::encode` now composes RMSNorm, Q/K/V, Q/K norm + RoPE, KV
insertion, GQA, O projection, residual + RMSNorm, router, top-8, dynamic MoE,
and the final residual into a caller-owned graph. It performs no allocation,
commit, wait, or readback. One layer is 14 dispatches in one command buffer.

## Real-artifact parity

The harness uses deterministic FP16 hidden input and synthetic prior FP16 KV
history, but all model matrices and norm vectors come from the complete mapped
artifact. CPU is the dequantized-QX numerical oracle, including FP16 stage
boundaries. These tests establish CPU/Metal artifact semantics, not BF16 model
quality.

| Layer / position | Direct expert L2 | Dynamic top-8 MoE L2 | Complete layer L2 | Warm layer GPU |
|---|---:|---:|---:|---:|
| 0 / 37 | 0.00007621 | 0.00012654 | 0.00009075 | 1.799 ms |
| 24 / 37 | 0.00005610 | 0.00007323 | 0.00050762 | 2.216 ms |
| 47 / 37 | 0.00002716 | 0.00016700 | 0.00077815 | 1.661 ms |

Every CPU, separately submitted Metal, dynamic-MoE, and complete-layer replay
selected identical expert IDs. Complete-layer cosine was at least
`0.9999997`. Timings are same-process warm means on the M5 Pro at context 38;
they are correctness-kernel measurements, not a full-model token-rate claim.

## Persistent mapped bindings

`PreparedQwenLayer` now binds Q/K/V/O/router, four norm vectors, and all 128
Q2S4 experts as subregions of one no-copy Metal buffer backed by the artifact
mmap. On the M5 Pro:

| Measurement | Result |
|---|---:|
| Mapped weight-slab buffer | 9,927,294,976 bytes |
| Experts bound | 128 |
| Layer-0 binding time after pipeline preparation | 2.9-5.0 ms |
| Weight upload/repack | none |
| Reusable layer activation workspace | 72,000 bytes |
| Dynamic expert offset tables | 4,096 bytes per prepared layer |

Metal's `currentAllocatedSize` increases by the logical buffer length because
the complete mapping is exposed to Metal. That is not evidence of a second
resident copy: `newBufferWithBytesNoCopy` owns the mmap lifetime and every
prepared tensor is only a range of that same buffer. The complete parity replay
above passes while executing directly from those mapped ranges.

## Reproduce

```bash
cargo run --release -p qx-qwen-parity -- \
  /Users/blas/projects/qx/external/models/qwen3-30b-a3b-a19pro-q2s4-capacity-candidate.qx \
  --layer 0 --expert 0 --position 37
```

## Complete token graph baseline

That milestone now executes. `PreparedQwenModel` binds the Q4 embedding, all
48 layers, final norm, untied Q4 LM head, and GPU argmax. `QwenModelState`
allocates context-bounded per-layer FP16 KV caches plus one shared layer
workspace. One input token traverses the complete model in a single Metal
command buffer without a host decision inside the graph.

M5 Pro, input token `151643`:

| Measurement | Result |
|---|---:|
| Prepared layers | 48 |
| KV growth per allocated context token | 98,304 bytes |
| Command buffers per token | 1 |
| Dispatches per token | 676 |
| Position 0 / capacity 64 | 6,675,528-byte state; 85.868 ms GPU; 11.65 tok/s |
| Position 63 / capacity 64 | 6,675,528-byte state; 92.382 ms GPU; 10.82 tok/s |
| Position 511 / capacity 512 | 50,715,720-byte state; 141.913 ms GPU; 7.05 tok/s |
| Deterministic position-0 next token | `58` |

This changes the optimization question. Host routing and token-boundary waits
are no longer on the critical path inside a token. The remaining 85.7 ms is
dominated by intentionally simple Q4 projection, GQA, and Q2S4 expert
consumers. These figures are a structural baseline, not the expected final
iPhone rate and not yet a BF16-teacher parity claim.

## What is not promoted yet

- multi-token GPU-resident rollout is not connected to this Qwen model state;
- GQA deliberately recomputes scores for every output dimension, which grows
  from a small fraction of token time at position 0 to roughly 56 ms of added
  GPU time at position 511;
- Q4 projections and Q2S4 experts are shape-correct scalar consumers, not the
  final SIMD/tiled performance mappings;
- the serial GPU argmax is a dependency-correct placeholder;
- parity is against quantized QX component math, not the BF16 teacher or
  downstream quality evaluations.

## First optimization from the complete baseline

The K-wide score-reuse GQA mapping assigns one 128-thread group to each query
head. It computes every Q.K score once into dynamic threadgroup memory and
reuses those scores across all 128 value dimensions. Contexts through 4,096
select this mapping; larger contexts retain the low-threadgroup-memory
correctness fallback.

At position 511 on the same M5 Pro and 512-token state allocation:

| Mapping | Warm GPU ms/token | tok/s |
|---|---:|---:|
| Per-output score recomputation | 141.913 | 7.05 |
| K-wide score reuse | 110.501 | 9.05 |

This is a 22.1% latency reduction with unchanged selected token and exact GQA
parity in the context-38 component replay. Position-0 time remained effectively
flat (`85.701 ms` versus `85.868 ms`), so the score-reuse specialization removes
context growth without taxing the short-context path.

The next context-independent change packetizes the Q2S4 expert decoder. The
original scalar loop loaded each packed byte four times and each FP16 group
scale up to 32 times. The packet consumer traverses the same columns in the
same order while loading one scale per 32 weights and one byte per four
weights. No artifact or quantization change is required.

| Full-token case | Before packet decoder | After packet decoder | Gain |
|---|---:|---:|---:|
| Position 0 | 85.701 ms GPU / 11.67 tok/s | 75.817 ms / 13.19 tok/s | 11.5% latency |
| Position 511, score reuse | 110.501 ms / 9.05 tok/s | 101.097 ms / 9.89 tok/s | 8.5% latency |

The position-511 result is 28.8% lower latency than the original complete
token baseline (`141.913 ms`). Dynamic-MoE parity remains below `1.6e-4`
relative L2 on the late-layer replay and selected tokens remain unchanged.

Q4 projection packetization applies the same physical-consumer correction to
Q/K/V/O/router and the untied LM head: one FP16 scale load per 32 weights and
one packed-byte load per two weights, with division/modulo removed from the
per-weight loop. Projection parity remains bit-exact or below `6.7e-7`
relative L2.

| Full-token case | Q2S4 packet baseline | + Q4 packet consumer | Gain |
|---|---:|---:|---:|
| Position 0 | 75.817 ms GPU / 13.19 tok/s | 54.714 ms / 18.28 tok/s | 27.8% latency |
| Position 511 | 101.097 ms / 9.89 tok/s | 80.552 ms / 12.41 tok/s | 20.3% latency |

Against the first complete graph, the combined consumer and attention changes
reduce position-0 latency by 36.2% and position-511 latency by 43.2%, while
preserving deterministic selected tokens (`58` and `106094`, respectively).
