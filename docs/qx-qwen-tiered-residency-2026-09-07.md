# Qwen heterogeneous tiered residency

Date: 2026-09-07
Status: heterogeneous execution proven on Mac; physical cache runtime and iPhone measurement pending

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

## Physical artifact-read control

`qx-route-cache replay-artifact-io` turns the candidate into a real bounded
RAM cache and reads the existing artifact at its actual tensor-directory
offsets. A source-exact miss performs four positional reads: scale and data for
gate-up, then scale and data for down. The macOS control enables `F_NOCACHE` so
the replay descriptor does not retain newly read payloads in the unified file
cache.

Five persistent-cache passes on the 24 GiB M5 Pro produced the following
median pass:

| Metric | Physical result |
|---|---:|
| cold-tier cache hit rate | 65.08% |
| bytes read per pass | 1,696,038,912 |
| mean storage work per token | 0.367 ms |
| p95 storage work per token | 1.772 ms |
| p99 storage work per token | 3.950 ms |
| mean time per missed expert | 0.322 ms |
| p95 time per missed expert | 0.602 ms |

Resetting the cache at every request boundary remains close: across three
passes the median p95 is 1.813 ms/token and p99 is 4.004 ms/token. Therefore
the current four-range physical layout is not a typical-read blocker on this
machine, even without coalescing or concurrent reads.

One persistent pass did contain a single 503.998 ms token stall; the other
four passes had maxima at or below 7.143 ms. This is important architecture
evidence, not noise to discard. Storage cannot synchronously gate every token.
Router-triggered asynchronous prefetch, GPU work overlap, and a bounded stall
fallback are required even though ordinary reads are fast.

This remains a Mac storage result. It does not measure iPhone storage, Metal
buffer binding, GPU contention, or delivered model throughput. The committed
aggregate is
`benchmarks/results/qwen3-30b-a3b-tiered-residency-physical-io-2026-09-07.json`.

## Layer-local Metal mapping

The runtime previously wrapped the complete 9--11 GiB artifact weight slab in
one `MTLBuffer`, then expressed every tensor as a subregion. That preserves
GPU-resident routing but creates the wrong resource shape for a memory-limited
phone. QX now creates one page-aligned no-copy buffer for each layer's
quantized extent and small tensor-local buffers for raw norms, embedding, and
LM head. Every buffer retains the same immutable artifact mmap owner.

On the uniform-Q2S4 control, layer 0's Metal resource fell from 9,926,475,776
bytes to 199,507,968 bytes (49.76x smaller). Expert, router, attention, dynamic
MoE, and complete-layer parity all pass. A complete 48-layer model still
constructs and executes in one command buffer: its warm token measured 51.386
ms wall / 50.713 ms GPU on the M5 Pro, or 19.46 tok/s, with a context-2 state.
This does not reduce total weight bytes by itself; it removes the giant-buffer
constraint and gives the OS/Metal layer-sized file-backed residency units.

The committed control is
`benchmarks/results/qwen3-30b-a3b-layer-local-metal-mapping-2026-09-07.json`.

## Cold-request control

Resetting the LRU between every corpus paragraph prevents cross-request warming.
At 128 MiB, the combined result remains zero median missing bytes, 12.66 MiB
p95, 32.91 MiB p99, 3.27 ms mean shaped I/O, 13.16 ms p95, and 34.21 ms p99.
Larger caches are effectively flat in this control because each short request
cannot populate them before reset.

The trace model initially supported the following runtime candidate:

```text
ExpertResidencyPlan {
    resident: Q2S4AndQ3Students,
    streamed: SourceExactQ4,
    cache: WholeExpertLru(128 MiB),
    storage: MmapNoCopy,
    prefetch: AfterRouterBeforeExpertConsumer,
}
```

The cache size was a measured workload/device candidate, not a universal model
constant. The physical runtime experiment below supersedes this candidate on
Apple unified memory: native file-backed residency is faster and consumes less
process-charged memory. A bounded explicit cache remains a selectable backend
for devices where source storage is not directly GPU-addressable.

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

The physical Mac control is:

```sh
target/release/qx-route-cache replay-artifact-io \
  --trace external/results/qwen3-full-q2s4-build/captures/heldout/000/routes.jsonl \
  --trace external/results/qwen3-full-q2s4-build/captures/heldout-extension/000/routes.jsonl \
  --artifact external/models/qwen3-30b-a3b-a19pro-q2s4-q3-early4-v1.qx \
  --budget 128MiB \
  --uncached-reads \
  --output external/results/qwen3-full-q2s4-build/tiered-residency-physical.json
```

## Remaining gates

1. Port the physical read replay to the target iPhone and repeat under storage,
   thermal, and application-memory contention.
2. Implement source-exact cache buffers and router-triggered prefetch in the
   Qwen runtime, with page-fault and GPU-wait instrumentation.
3. Replace the tiny provisional scratch ledger with measured peak Metal,
   allocator, tokenizer, and application memory.
4. Expand route and complete-model quality evidence beyond the current 567
   tokens and four causally validated layers.
5. Run physical iPhone jetsam, throughput, energy, and thermal gates before any
   product claim.

The bounded-residency policy should remain trace-emulated until those gates
pass; the heterogeneous Metal execution path itself is now physically proven
on the Mac control.

## Heterogeneous GPU execution

The canonical Qwen runtime can now execute the artifact's Q2S4, Q3, and
source-exact Q4 experts directly from each layer-local file-backed slab. The
router selects top-8 experts on the GPU, and a 128-byte codec table selects the
decoder without CPU route readback or per-expert buffer rebinding. Q2S4 keeps
its original packet path; Q4 has a codec-specialized group decoder.

Mixed-codec CPU/GPU parity passes on sampled layer-0 and layer-5 routes, while
a Q2S4-only route through a heterogeneous layer confirms the original compact
path remains fast. A complete 48-layer deterministic replay measured 55.147 ms
GPU per warm token, or 18.13 tok/s, on the 24 GiB M5 Pro. Before the dedicated
Q4 decoder the same heterogeneous execution took 68.577 ms GPU, or 14.58
tok/s. The specialization therefore improved delivered throughput by 24.36%
and reduced measured MoE time from 39.061 to 25.787 ms/token.

A same-build revalidation produced 53.787 ms GPU/token, or 18.59 tok/s, with
mixed-layer relative L2 unchanged at 0.00005393. First-token wall time moved
from 2.936 s to 2.098 s as the mmap page state changed, so cold wall latency is
reported separately and is not treated as kernel throughput.

The remaining heterogeneous penalty is concentrated in the four early Q3
layers: the uniform-Q2S4 control is 51.308 ms GPU/token, while the mixed model
is 55.147 ms. The next representation/runtime experiment is the existing
lossless Q3Superblock112 transport: 96 contiguous code bytes plus eight FP16
scales per 256 weights, consumed with packet-vector decoding. It changes no Q3
quantized values and requires no student retraining.

That experiment now passes. The compiler emits all 908 Q3 tensors in the four
early layers as Q3Superblock112, while legacy row-major Q3 remains readable.
The new artifact has the same 11,537,506,304 logical weight bytes and the Q3
directory views alias each single 112-byte-block stream. Its SHA-256 is
`9ddc646bcffa05626073ff46c0bc0adf2d18e16680e5f6a69c47e2f0f1d33711`.

In a three-pair same-session matrix of ten warm complete tokens per process,
legacy Q3 averaged 54.124 ms GPU/token (18.48 tok/s) and Q3-112 averaged
50.346 ms (19.86 tok/s). That is 6.98% lower complete-token latency and 7.50%
higher throughput. MoE time fell by 3.725 ms/token, from 24.991 to 21.266 ms;
attention and head time stayed effectively flat. Layer-0 dynamic parity passes
at relative L2 0.00004507. Sampled dynamic and complete-layer parity also pass
for layers 1, 2, and 3, with worst relative L2 0.00024783. Q3-112 is therefore
promoted for Qwen Q3 students, not merely retained as an experiment.

Machine-readable evidence is in
`benchmarks/results/qwen3-30b-a3b-q3-superblock112-runtime-2026-09-07.json`.

## Physical runtime policy: native VM versus explicit GPU LRU

QX now has an executable GPU-resident source-Q4 LRU for testing the residency
model without reintroducing a CPU token boundary. A resolver dispatch owns a
fully associative 50-slot table, second-touch admission rejects one-off tail
experts, and a conditional fill dispatch copies admitted 2,654,208-byte Q4
experts into a 126.56 MiB Metal buffer. Hit, miss, eviction, bypass, and Q4
selection counters remain on GPU until the chunk boundary.

The explicit cache is correct but is not the Apple winner. Three paired runs
of 64 causally chained tokens in one command buffer produced:

| Runtime residency | GPU ms/token | tok/s | dispatches/token | final token |
|---|---:|---:|---:|---:|
| native mmap/no-copy | 51.933 | 19.26 | 724 | 15 |
| 128 MiB GPU LRU, second-touch | 55.506 | 18.02 | 820 | 15 |

The LRU recorded 1,024 hits, 549 misses, 300 evictions, and 199 first-touch
bypasses, for a 65.10% hit rate. It nevertheless increased token latency by
6.88% and reduced throughput by 6.44%, because unified memory already makes
clean file-backed pages directly GPU-readable; copying admitted experts adds
another read/write traversal and 96 dispatches per token.

Process accounting makes the structural distinction explicit. The native path
mapped 11,259,314,176 bytes (10.486 GiB) of logical Metal weight resources, but
a 64-token run peaked at 98,534,192 bytes of process physical footprint. The
explicit cache peaked at 242,811,768 bytes. These figures exclude globally
reclaimable clean file-cache pages. `mincore` reported the complete artifact resident because it
had just been encoded and audited; `MADV_DONTNEED` did not purge that global
cache, so file-cache presence is not reported as QX process footprint.

The compiler/runtime now expresses this as `ExpertResidencyPlan`. Apple
unified-memory artifacts select `NativeFileBacked`; `GpuLru` with second-touch
admission remains an experimental backend for a device whose physical storage
and fault measurements justify the copy. This is the stronger result: the
planner can choose the residency mechanism per target rather than assuming
that an application-owned cache is universally better.

This Mac result does not prove iPhone jetsam safety. iOS may account Metal and
file-backed GPU pages differently, and storage pressure, thermals, and a real
application add memory not represented here. The exact next gate is the same
instrumented 64-token run inside the iPhone app, recording jetsam footprint,
page-ins, GPU time, and sustained throughput. Machine-readable evidence is in
`benchmarks/results/qwen3-30b-a3b-expert-residency-runtime-2026-09-07.json`.

The committed runtime result is
`benchmarks/results/qwen3-30b-a3b-heterogeneous-gpu-runtime-2026-09-07.json`.
Its claim is deliberately narrow: sampled mixed-layer semantic parity and
complete-model deterministic execution on this Mac. It does not yet prove
complete-model CPU parity, bounded physical residency, iPhone throughput, or
quality.
