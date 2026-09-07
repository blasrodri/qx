# Qwen heterogeneous tiered residency

Date: 2026-09-07
Status: trace-emulated candidate; physical runtime and iPhone measurement pending

## Outcome

The compiled heterogeneous Qwen artifact exposes a much better physical-memory
policy than treating all 10.746 GiB of installed weights as permanently
resident. Keep the 5,006 trained Q2S4/Q3 students resident and make the 1,138
source-exact Q4 experts an mmap-backed, bounded LRU tier.

This is structurally different from the rejected all-expert paging experiment.
The source-exact experts are predominantly the under-evidenced tail selected by
the compiler policy. Across 567 disjoint held-out route positions, only 1,830
of 217,728 routed expert lookups (0.8405%) target that tier.

The artifact-derived memory split is:

| Tier | Bytes | GiB |
|---|---:|---:|
| resident student and dense weights | 8,517,017,600 | 7.932 |
| mmap-backed source-exact weights | 3,020,488,704 | 2.813 |
| 4K FP16 KV plus modeled scratch | 403,891,712 | 0.376 |
| resident total before cold cache | 8,920,909,312 | 8.308 |

The installed artifact remains 10.746 GiB. The improvement is physical
residency: source-exact pages remain evictable until their expert is routed.

## Cache sweep

`qx-route-cache simulate --artifact-tier` now reads the expert policy and real
tensor-directory lengths directly from the `.qx` artifact. Resident expert
lookups are removed from the cache simulation; source-exact Q4 lookups use the
artifact's measured 2,654,208 bytes per expert. Multiple disjoint traces can be
evaluated as one ordered set.

The screen combines 232 curated-general and 335 WikiText held-out positions.
The I/O column is shaped with the earlier conservative synthetic profile of
1,000 MiB/s shared bandwidth, 100 microseconds fixed latency, and eight
in-flight fetches. It is not an iPhone storage measurement.

| LRU cache | Modeled residency | Hit rate | Mean missing | p50 missing | p95 missing | p99 missing | Mean I/O | p95 I/O | p99 I/O |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 MiB | 8.371 GiB | 53.28% | 3.82 MiB | 0 | 17.72 MiB | 48.09 MiB | 3.96 ms | 18.42 ms | 49.59 ms |
| **128 MiB** | **8.433 GiB** | **65.08%** | **2.85 MiB** | **0** | **12.66 MiB** | **27.84 MiB** | **2.96 ms** | **13.16 ms** | **28.94 ms** |
| 256 MiB | 8.558 GiB | 69.62% | 2.48 MiB | 0 | 12.66 MiB | 27.84 MiB | 2.58 ms | 13.16 ms | 28.94 ms |
| 512 MiB | 8.808 GiB | 75.25% | 2.02 MiB | 0 | 10.12 MiB | 25.31 MiB | 2.10 ms | 10.53 ms | 26.31 ms |

The 128 MiB point is the first useful knee. Doubling it does not improve p95,
and 512 MiB buys only 2.63 ms at p95 while consuming another 384 MiB.

## Cold-request control

Resetting the LRU between every corpus paragraph prevents cross-request warming.
At 128 MiB, the combined result remains zero median missing bytes, 12.66 MiB
p95, 32.91 MiB p99, 3.27 ms mean shaped I/O, 13.16 ms p95, and 34.21 ms p99.
Larger caches are effectively flat in this control because each short request
cannot populate them before reset.

This supports the following runtime candidate:

```text
ExpertResidencyPlan {
    resident: Q2S4AndQ3Students,
    streamed: SourceExactQ4,
    cache: WholeExpertLru(128 MiB),
    storage: MmapNoCopy,
    prefetch: AfterRouterBeforeExpertConsumer,
}
```

The cache size is a measured workload/device candidate, not a universal model
constant. The compiler should select it from the artifact policy, device memory
budget, route replay, and physical storage profile.

## Why this reopens the phone lane

The old exact-packet study paged every routed expert and observed 311--480 MiB
p95 missing bytes per token with 4--6 GiB caches. Packetization could not fix
that because every expert packet was required. The new policy permanently
resides the common transformed tier and streams only the compiler-selected
source-exact tail. It produces 12.66 MiB p95 with a 128 MiB cache on the current
held-out screen—a different memory architecture, not a better packet size.

This does not yet prove that an iPhone can sustain the target token rate. It
does show a credible route to roughly 8.43 GiB modeled model residency without
removing more Qwen parameters or reopening the failed uniform width-512 model.

## Reproduction

```sh
target/release/qx-route-cache simulate \
  --trace external/results/qwen3-full-q2s4-build/captures/heldout/000/routes.jsonl \
  --trace external/results/qwen3-full-q2s4-build/captures/heldout-extension/000/routes.jsonl \
  --artifact-tier external/models/qwen3-30b-a3b-a19pro-q2s4-q3-early4-v1.qx \
  --policies lru \
  --budgets 64MiB,128MiB,256MiB,512MiB \
  --packet-neurons 768 \
  --layouts expert-major \
  --io-bandwidth-mib-s 1000 \
  --io-fixed-us 100 \
  --io-parallelism 8 \
  --output external/results/qwen3-full-q2s4-build/tiered-residency-heldout-combined.json
```

Add `--reset-between-requests` for the cold-request control. The committed
aggregate is
`benchmarks/results/qwen3-30b-a3b-tiered-residency-screen-2026-09-07.json`.

## Remaining gates

1. Measure actual random and concurrent reads from the artifact on the target
   iPhone, including the current split scale/data and gate-up/down slabs.
2. Implement source-exact cache buffers and router-triggered prefetch in the
   Qwen runtime, with page-fault and GPU-wait instrumentation.
3. Replace the tiny provisional scratch ledger with measured peak Metal,
   allocator, tokenizer, and application memory.
4. Expand route and complete-model quality evidence beyond the current 567
   tokens and four causally validated layers.
5. Run physical iPhone jetsam, throughput, energy, and thermal gates before any
   product claim.

The candidate should remain trace-emulated until those gates pass.
