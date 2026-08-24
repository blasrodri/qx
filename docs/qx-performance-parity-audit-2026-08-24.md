# QX performance and llama-parity audit

Date: 2026-08-24  
Audit branch: `codex/perf-parity-audit`  
QX revision: `5b32612`  
llama.cpp runner revision: `436ea36` (relevant Q4_K Metal source matches the
previously pinned `1719747`)  
Artifact: `granite-m5-pro-q4-rowinline-unique-packet.qx`  
Artifact hash: `e0891fd697f49774`  
Machine: Apple M5 Pro, 16 GPU cores, 24 GB unified memory

## Executive conclusion

QX is not primarily losing to llama.cpp because it launches too many kernels,
allocates buffers during decode, or uses too many command buffers. The clean,
matched measurements put QX at approximately **211--221 tok/s** and llama.cpp
at approximately **310--320 tok/s** after the same 384-token prefix. QX needs
to remove roughly **1.3--1.6 ms of wall time** and **1.47 ms of GPU time** per
token for parity.

The dominant problem is effective weight-streaming throughput in the quantized
matrix-vector kernels, especially MoE. QX reads approximately 241 MB of active
Q4 weights per token and sustains only about **60 GB/s** of useful weight
traffic at its current 4.02 ms GPU time. The parity target is approximately
**86 GB/s for within 10%** and **95 GB/s for parity**. Llama's matched
production interval already lands near the latter target.

The most important untried design is a **joint storage-and-kernel rewrite**:
payload-contiguous 256-weight superblocks, vectorized loads, compile-time
unrolling, activation reuse, and two or more output rows accumulated in
parallel. Prior “transaction” and “llama-like” experiments did not implement
that combination, so their negative results do not falsify it.

## Corrected benchmark contract

The earlier comparison in `qx-live-gap-ledger-2026-08-24.md` is not actually a
matched context-512 decode comparison. Its llama result records
`n_prompt = 0`, `n_gen = 128`, and `n_depth = 0`. In llama-bench, `-p 384 -n
128` creates separate prompt and generation test instances; it does not make
the measured generation follow that prompt. The matched command is `-p 0 -n
128 -d 384`.

Clean, idle-GPU runs gave:

| Engine | Contract | Wall result | Production GPU result |
|---|---|---:|---:|
| QX exact Q4 | fixture with 384 prefill + 128 measured decode | 4.53--4.74 ms/token, 211--221 tok/s | 3.96--4.07 ms/token |
| llama.cpp Q4_K_M | `-p 0 -n 128 -d 384`, FA, FP16 KV, full offload | 3.13--3.23 ms/token, 310--320 tok/s | 2.546 ms/token median |

The llama production GPU median excludes the cold first sample and uses
samples 2--10: 2633.67, 2563.29, 2546.29, 2556.75, 2540.08, 2549.75,
2540.87, 2546.12, and 2545.67 microseconds.

The context correction only makes llama roughly 2--3% slower, so it does not
erase the gap. It does, however, lower the honest parity target from the
previously quoted 326 tok/s to roughly 310--320 tok/s on this setup.

All measurements taken while another GPU benchmark was active were discarded.
This matters on unified-memory Apple hardware: the contaminated run inverted
the ranking and was not representative of either engine.

## Where the time goes

The current execution graph uses one command buffer and 461 dispatches per
token: 19 per transformer layer plus five final operations. It allocates the
model arena and KV storage before decode and performs one CPU wait at token
completion. There are no per-token Metal buffer allocations in the normal
decode path.

Llama's matched graph reports 1,471 graph nodes, 72 `mul_mat_id` operations,
and two command buffers per token, yet it is faster. Therefore dispatch count
or command-buffer count by itself cannot explain the 1.47 ms GPU gap.

QX's exact active Q4 weight traffic is:

| Component | Bytes per token |
|---|---:|
| Per-layer QKV | 1,179,648 |
| Per-layer O projection | 589,824 |
| Per-layer router | 18,432 |
| Per-layer selected top-8 gate/up | 4,718,592 |
| Per-layer selected top-8 down | 2,359,296 |
| All 24 layers | 212,779,008 |
| LM head | 28,313,280 |
| **Total** | **241,092,288 bytes (229.9 MiB)** |

MoE accounts for about 170 MB, or **70.5%**, of this mandatory weight stream.
This agrees with the live differential ledger: disabling the MoE boundary
saved about 2.21 ms of complete-graph GPU time, with gate/up and down the clear
hot stages. That differential is not an additive operator timer, but it is
strong target-ranking evidence.

## The fundamental kernel mismatch

QX's promoted gate/up kernel is a meaningful improvement: it cooperatively
loads unique packed bytes and broadcasts scales within a SIMD group. Its host
mapping launches 1,024 threadgroups per layer for the selected experts, with
256 threads and eight SIMD groups per threadgroup. Each SIMD group computes
one output row.

However, the stored row remains a series of 18-byte groups:

```text
[16 packed-code bytes][2-byte scale]
[16 packed-code bytes][2-byte scale]
...
```

This repeatedly interleaves metadata with payload and gives awkward alignment
for wide vector loads. The archived 144-byte and 160-byte “transaction” layouts
still use this same `group * 18` organization inside each 256-weight block.
Their consumer also retains the generic group loop and scalar-lane decode.

Llama's Q4_K decode kernel instead consumes a deliberately structured
256-weight block. It uses explicitly unrolled vector operations, keeps a large
activation slice in registers, reuses that input across multiple output rows,
and accumulates multiple row results concurrently. The important difference is
not merely 144-byte block size or threadgroup width; it is the whole producer /
consumer contract.

Similarly, QX's archived “two-row” and “llama-like” mappings do not combine all
of the following:

1. payload-contiguous superblocks;
2. unique cooperative byte ownership;
3. vector loads and vector arithmetic;
4. two-or-more output rows per activation load;
5. compile-time unrolling for the fixed 1,024-wide hidden dimension; and
6. direct expert-base addressing without generic layout branches.

Consequently, the most promising exact-parity experiment remains open.

## Recommended implementation sequence

### P0: make benchmark truth non-negotiable

- Change all llama decode controls to use `-d 384`, not a separate `-p 384`
  test.
- Record QX revision, llama revision, artifact hash, plan name, depth, KV type,
  Flash Attention setting, raw samples, thermal/power state, and competing GPU
  processes in every result.
- Randomize engine order over at least 20 paired runs; reject runs with other
  GPU workloads.
- Treat same-artifact CPU/GPU parity as implementation correctness only. Add
  held-out NLL and task-quality gates before calling QX's exact Q4 result
  equivalent to llama Q4_K_M.

### P1: build a true superblock MoE kernel

Create an exact QX representation for each 256 input weights, for example:

```text
[128 contiguous packed-code bytes][8 FP16 scales = 16 bytes]
```

Keep the existing group-32 values and scales so this is a layout change, not a
quality change. Align row starts and blocks for vector access. Build a
specialized Metal consumer with a fixed hidden-size specialization, unrolled
four-block loop, register-resident activation fragments, vector payload loads,
and at least two output rows per activation load. Apply the same kernel family
to gate/up and down rather than optimizing only one side of the MoE.

First test it as a gate/up microbenchmark, but promotion must be based on the
full token graph. The experiment should be rejected unless it saves at least
0.15 ms/token; the useful goal is **0.7--1.1 ms/token** from MoE. That range is
a target derived from the gap ledger, not a measured promise.

### P2: reuse the kernel family for dense projections

The current dense Q4 and fused QKV kernels still issue duplicate packed-byte
and scale references across lanes. Port QKV, O, and then the LM head to the
same payload-contiguous, multi-row kernel family. They represent roughly 71 MB
of active weight traffic per token. Expected opportunity is approximately
0.15--0.35 ms/token, subject to complete-graph measurement.

### P3: replace decode attention with an online-softmax kernel

At query length one, map work by KV head, process the paired query heads
together, update online softmax statistics while streaming K/V, and avoid the
materialized score and partial buffers. Also evaluate folding RoPE/KV writes
into the attention boundary. This should reduce five attention dispatches per
layer to one or two and eliminate scratch traffic. Existing evidence makes
this a likely 0.1--0.3 ms opportunity, not the primary 1.47 ms explanation.

### P4: remove CPU submission friction

After GPU work improves, precompute buffer subregions, replace linear buffer
lookups with indexed handles, avoid per-dispatch pipeline mutex/refcount work,
reuse command-buffer bookkeeping, and provide an asynchronous multi-token API.
The current best QX wall-minus-GPU gap is about 0.46 ms, so CPU cleanup cannot
deliver scalar parity alone.

### P5: keep speculative decoding as a separate product lane

The current K=4 verifier result amortizes to roughly 3.72 ms per proposed
token before draft-model cost and real acceptance effects. It is useful system
work, but it is not scalar-kernel parity. Compare end-to-end accepted tokens
per second against llama using the same draft model and workload; do not use
teacher-forced verifier throughput as the exact-parity headline.

## Lower-priority and disproven explanations

- **Decode-time GPU allocations:** not present in the normal path.
- **One versus two command buffers:** prior splitting changed wall scheduling,
  not the missing GPU work.
- **Dispatch count alone:** llama executes a larger graph and remains faster.
- **Artifact size:** QX is smaller than the comparison GGUF, so model bulk is
  not the reason QX loses.
- **Another isolated packet-layout tweak:** packetized down and LM-head
  variants have already measured below the promotion threshold. The next
  experiment needs a different storage/compute contract.
- **Speculation as proof of kernel parity:** it changes the product contract
  and must be reported separately.

## Repository and measurement defects found

1. `qx-bench --v2-plan` double-counted row-inline expert storage by summing
   `scale_length + data_length` even when both fields alias the same stream. It
   reported 1,358,954,496 expert bytes for a 751,009,792-byte artifact. The
   audit-branch fix counts the union of those ranges and now reports the
   correct 679,477,248 expert bytes.
2. `tools/qx-bench/build.rs` assumes `../../external/llama.cpp`, which breaks
   normal Git worktrees because the ignored `external` directory is not
   populated there. The audit branch adds `QX_LLAMA_ROOT` while preserving the
   existing relative-path fallback.
3. Plan activation mutates environment variables while qx-metal caches many
   choices in `OnceLock`. Switching plans or artifacts within one process is
   therefore fragile and can silently retain a prior configuration. Pass an
   immutable compiled plan/config object into graph construction instead.
4. Several benchmark artifacts record `qx_revision: unknown` or were produced
   from a dirty tree. Those results are useful for exploration but inadequate
   for a parity claim.

## Promotion gates

A parity candidate should satisfy all of these before promotion:

- exact full-token benchmark after a 384-token prefix;
- at least 20 randomized paired runs against the pinned llama control;
- production GPU interval and wall-clock throughput both improve;
- long CPU/GPU numeric parity, not only a few positions;
- no quality regression on held-out NLL and representative tasks;
- no hidden memory-footprint regression;
- reproducible source revision, artifact hash, compile flags, and raw samples.

The practical milestone is first **under 3.5 ms/token** (within roughly 10% of
the matched llama run), then 3.1--3.2 ms/token parity. The superblock MoE
kernel is the only currently identified exact-path change with enough
plausible headroom to move the first milestone by itself.
