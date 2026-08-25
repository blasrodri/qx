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

## What is not promoted yet

The persistent single-layer graph is now real, but it is still a correctness
baseline rather than a complete token runtime:

- all 48 prepared layers are not yet chained in one command buffer;
- per-layer persistent KV caches, embedding, final norm, untied LM head,
  argmax, and GPU-resident rollout are not connected;
- the GQA kernel deliberately recomputes scores for every output dimension;
- the Q4 projections and Q2S4 expert kernel are shape-correct scalar consumers,
  not the final SIMD/tiled performance mappings;
- parity is against quantized QX math, not the BF16 teacher or downstream
  quality evaluations.

The next executable milestone is one reference token through all 48 layers.
That will expose the actual working-set/page-residency behavior and establish
the baseline from which attention and expert consumers should be optimized.
