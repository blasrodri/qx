# Canonical GPU-resident greedy rollout

## Result

GPU-resident greedy rollout is now part of the sealed compiler/runtime plan,
including chunked EOS handling. The depth sweep does **not** support the prior
assumption that depth 4 is the M5 Pro optimum. It supports depth **2**.

Across 90 runs (six depths, five contexts, three repetitions), depth 1 left a
mean 0.255 ms/token gap between GPU command buffers. Depth 2 reduced that to
0.0012 ms/token, a 99.5% reduction. Depths 3, 4, 6, and 8 did not reduce idle
time further or produce a repeatable throughput gain. They only retained more
submitted work and raised peak physical footprint.

The promoted M5 Pro greedy plan is therefore:

```text
RolloutPlan {
    dependency: GpuTokenChain,
    inflight_depth: 2,
    host_sync: ChunkBoundary,
}
```

This remains a compiler-selected specialization. The artifact carries the
policy, the runtime validates it, and `--rollout-depth` plus
`QX_V2_ROLLOUT_DEPTH_OVERRIDE` retain an explicit autotuning/diagnostic path.
Artifacts that predate the field deserialize to the synchronous depth-1 plan.

## Why two is enough

CPU graph encoding averages roughly 0.14--0.19 ms/token while GPU execution is
roughly 2.3--2.5 ms/token. One queued successor is therefore enough to hide CPU
encoding and the token-boundary submission delay behind the current GPU token.
Depth 1 cannot do that because it waits before encoding/submitting the next
token. Once depth 2 keeps a successor queued, deeper windows expose no second
independent dependency: every token still consumes the preceding GPU argmax.

CPU wait time remains around 2.2 ms/token at depth 2 because the host eventually
waits for the oldest submitted graph. It is not an additive cost. The important
timeline facts are that the GPU queue gap and wall-minus-GPU idle both collapse
to about 0.005 ms/token.

Mean across the five contexts:

| Depth | tok/s | Wall ms/tok | GPU ms/tok | CB gap ms/tok | Idle ms/tok | Peak physical MB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 373.48 | 2.680 | 2.421 | 0.2553 | 0.2596 | 92.52 |
| 2 | **410.95** | 2.436 | 2.432 | 0.0012 | 0.0046 | 93.18 |
| 3 | 410.30 | 2.441 | 2.436 | 0.0013 | 0.0050 | 93.97 |
| 4 | 411.08 | 2.436 | 2.430 | 0.0012 | 0.0056 | 94.78 |
| 6 | 411.32 | 2.434 | 2.429 | 0.0011 | 0.0048 | 96.53 |
| 8 | 409.95 | 2.443 | 2.438 | 0.0013 | 0.0050 | 98.17 |

The complete 30-cell metric table, including CPU encode/wait and peak memory,
is in `benchmarks/results/gpu-rollout-depth-sweep-2026-08-24.json`.

## Context sweep

Each cell is a three-run mean. Context 64 measured positions 0--63. Contexts
128, 256, and 384 used enough of the canonical prefix for 64 measured tokens
to end at the context bound. Context 512 used the full 384-token prefix and
measured all 128 positions through 511.

| Context | d1 | d2 | d3 | d4 | d6 | d8 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 391.90 | 430.20 | 432.25 | 426.30 | 429.54 | 433.46 |
| 128 | 373.94 | 411.08 | 420.75 | 426.04 | 415.50 | 414.86 |
| 256 | 374.60 | 410.00 | 410.89 | 408.63 | 408.78 | 406.39 |
| 384 | 370.52 | 412.91 | 396.80 | 404.08 | 412.02 | 404.80 |
| 512 | 356.43 | 390.56 | 390.81 | 390.36 | 390.76 | 390.24 |

The variation among depths 2--8 tracks GPU busy-time variation rather than
queue idle. This is why selecting the single largest cell would incorrectly
promote depth 8 at context 64 or depth 4 at context 128. Depth 2 is the Pareto
choice across the measured workload.

## Fresh llama.cpp comparison

The same-session context-512 matrix alternated engines and used a 384-token
prefill plus 128 measured decode evaluations:

| Engine | Samples tok/s | Mean tok/s |
|---|---|---:|
| QX synchronous | 379.62, 357.25, 369.89 | 368.92 |
| QX canonical depth 2 | 413.30, 414.19, 412.47 | **413.32** |
| llama-bench Q4_K_M | 334.88, 333.68, 314.98 | 327.85 |
| llama.cpp exact-fixture greedy | 314.30, 313.52, 312.33 | 313.38 |

Canonical QX is 26.1% faster than the fresh llama-bench generation control and
31.9% faster than fixture-driven llama greedy including synchronization and
CPU argmax. The exact-fixture harness feeds llama.cpp the same 384 raw token
IDs; llama-bench is retained as the standard, more favorable predetermined-token
upper-bound control. QX uses its exact group-32 Q4 representation while the
llama control is Q4_K_M, so this is a model/device/workload comparison rather
than a byte-identical quantization comparison.

Synchronous and depth-2 QX produced the same 128-token history hash
`19b4967c2513510e`, KV hash `5f18e9cd28097e0b`, and final token 34 in every
comparison run.

## Canonical runtime behavior

`GraniteV2Generator::generate_tokens` now follows the artifact rollout plan.
The explicit synchronous method remains available for control measurements,
and fixed-length GPU-resident rollout remains available for exact-count jobs.

EOS-aware generation submits a dependency-ordered chunk, reads the produced
IDs once at the boundary, and returns through the first EOS. If EOS occurs
inside the chunk, later work from that final chunk is discarded by rewinding
the logical KV cursor. Those suffix KV slots cannot be observed and are
overwritten before any future attention use.

## New kernel baseline

With queue idle removed, GPU kernel time is effectively wall time. Fresh
context-384 differential profiling ranks the exact blocks as:

- MoE: 0.99 ms/token marginal in the differential graph and 1.12 ms/token in
  the production-preserving block probe;
- attention plus output projection: 0.62 ms/token in the production probe;
- final head: 0.25 ms/token; and
- router/top-k: 0.21 ms/token.

The first new-baseline kernel screen doubled superblock rows per threadgroup to
halve input staging. It preserved the exact token and KV hashes but fell to
350.88 tok/s from the 413.32 tok/s canonical mean. The additional live row
accumulators hurt occupancy more than reduced staging helped, so the change was
reverted. Wider superblock geometry is closed; the next kernel work should
target MoE packet decode/load ownership or a lower-register mapping.

Speculation remains closed. No shared/early-exit draft measured here has a
credible delivered-throughput path above 400 tok/s, while exact canonical
greedy rollout already exceeds that threshold.
