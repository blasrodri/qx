# QX current status

Date: 2026-08-22  
Target: Apple M5 Pro  
Model: IBM Granite 3.0 1B-A400M Instruct  
Workload: batch-1 greedy decode, position 384 through 511, five fixed domains

## Executive status

QX now has the machinery needed to test three separate claims: runtime parity,
specialization alpha, and automatic compiler selection. It does **not** yet
pass the final Pareto gate against llama.cpp.

The current quality-neutral top-8 runtime is about 200 tok/s in release builds,
versus roughly 326 tok/s for the pinned llama control in the latest paired
smoke. The remaining runtime gap is real. The new mmap/no-copy artifact path,
however, removes the large duplicated host-weight allocation and gives QX a
smaller artifact than the pinned GGUF.

| Current measurement | QX | pinned llama control |
|---|---:|---:|
| Model/artifact bytes | 751,009,792 | 821,844,576 |
| Complete decode | ~200 tok/s | ~326 tok/s |
| Wall latency | ~5.0 ms/token | ~3.1 ms/token |
| Static model representation | group-32 Q4 | Q4_K_M |
| Internal CPU waits | 0 | not yet equivalently exposed |
| QX full no-copy RSS smoke | ~21 MB | ~914–922 MB process RSS |
| Known QX Metal allocation | ~802 MB | 820,107,736 B model tensors before KV/scratch |

RSS and macOS physical footprint are supporting measurements only: file-backed
and unified-memory pages are not charged symmetrically. The defensible current
memory facts are the smaller QX artifact, removal of copied host weights, and
the explicit QX Metal allocation. A same-semantics total-allocation comparison
is still required for the final memory proof.

## Canonical execution

Artifact v2 carries a sealed execution plan with representations, topology,
schedule, liveness/memory policy, sparse policy, target/workload constraints,
and provenance. Runtime activation is driven by that plan rather than by a
manually selected winner.

The current top-8 graph is:

```text
inverse-RMS scalar -> normalized fused QKV
Q RoPE + direct K/V cache writes
three-stage attention (QK, hierarchical softmax, paired-GQA V)
Q4 O + residual
inverse-RMS scalar -> normalized router -> GPU top-8
four-stage MoE (gate/up, SiLU product, down, weighted accumulation)
residual
final norm -> tied Q4 head -> GPU argmax -> u32 readback
```

Every token has GPU-resident intermediates, no decode-time allocation, no
intermediate readback, and one final CPU wait. Command-buffer count is now a
compiler choice. A 15-pair, five-fixture diagnostic found that two GPU-ordered
buffers reduced wall time by 0.137 ms/token (bootstrap 95% CI -0.151 to -0.121)
without changing dispatch count or GPU work. It remains provisional until its
full parity and quality/promotion records are complete.

Both the full mmap/no-copy one-buffer control and provisional two-buffer winner
passed the hard sequential gate at
positions `0,1,7,31,63,127,255,511`:

```text
max KV key error       0.009850
max KV value error     0.001048
max attention error    0.000127
max logit error        0.009907
minimum logit cosine   1.000000
argmax agreement       8/8
```

## Corrected live attribution

The profiler now prefills the real fixture to position 384 and freezes the
canonical route trace for substitutions. Across all five fixtures, the
complete-token baseline was about 4.69 ms GPU and the marginal ranges were:

| Complete-token substitution | Marginal saving |
|---|---:|
| full MoE | 2.67–2.69 ms |
| attention | 0.45–0.46 ms |
| input RMS/QKV | 0.41 ms |
| final norm/head/argmax | 0.37–0.46 ms |
| router/top-k | 0.23–0.25 ms |
| O/residual | 0.21–0.26 ms |
| RoPE/KV | 0.04–0.05 ms |

The fixed-route MoE decomposition is coherent: gate/up contributes roughly
1.72–1.80 ms, down 0.83–0.89 ms, while SiLU and accumulation are negligible.
These are marginal A/B measurements and are not added together to predict
throughput. MoE is the next quality-neutral implementation target.

## Implemented validation/compiler infrastructure

- tracked benchmark provenance contract and five exact-token fixtures;
- randomized paired QX/llama runner with thermal-transition retries and paired
  bootstrap intervals;
- early- and late-decode fixture support;
- artifact-v2 sealed plan IDs and automatic evidence-based plan selector;
- four promotion tiers: kernel, producer/consumer, multi-layer, complete model;
- copied, hybrid, and full mmap/no-copy static-weight ownership candidates;
- Q4_K expert representation and Metal kernel as an early scientific control;
- triangular FP reference / llama / QX teacher-forced trace comparison;
- confidence-width and 32,768-token/domain minimum enforcement;
- paired HumanEval and GSM8K task-result scorer;
- corrected full-graph and fixed-route MoE differential profilers;
- a tracked patch reproducing the pinned llama instrumentation.

The selector rejects missing task evidence, failed quality gates, missing
promotion tiers, non-finite timings, internal CPU waits, and memory-budget
violations. A numerically plausible but underpowered quality run cannot be
promoted.

## Controls and negative results

The Q4_K expert control is implemented, byte-checked against a pinned llama C
reference, and executable in the full model. On the current small calibration
fixture it was both slower (~147 tok/s) and worse in PPL than QX group-32 Q4,
so it is not canonical. The fixture is too small for final quality claims, but
the control already separates representation from the rest of the runtime.

Earlier Tile32, split gate/up, alternate fused-QKV geometry, uniform Q3,
prefetch, physical expert permutation, and active-k candidates remain useful
search evidence. None may be promoted from isolated timing alone.

## What remains before the three proofs pass

1. Run the held-out triangular gate on at least 32,768 tokens for each of
   WikiText-2, Dolly, HumanEval, and GSM8K, increasing to 131,072 where needed.
2. Supply paired greedy HumanEval execution results and GSM8K exact-match
   results; perplexity is not a substitute for task accuracy.
3. Run 20–30 randomized QX/llama pairs on AC power with a clean revision and
   repeat any thermal-transition blocks.
4. Close the current ~39% throughput deficit, beginning with the corrected
   live top-8 MoE gate/up and down paths.
5. Measure llama memory with semantics comparable to QX Metal/KV/scratch
   allocation, then establish lower steady and peak physical footprint.
6. Re-run clean install-time tuning and show it repeatedly selects the same
   winner or a statistically equivalent Pareto plan.
7. Only after top-8 is within 10% of llama may active-k, mixed precision, or
   shared-plus-residual expert approximations enter the promotion frontier.

## Latest MoE candidate screening

The first implementation candidate after the corrected live attribution was a
`four_stage_wide_expert` plan. It kept the canonical top-8 Q4 representation,
four GPU stages, GPU-resident routes, and numerical operations, but changed the
Metal mapping to one 1024-threadgroup per selected expert so the input vector
would be staged once per expert instead of once per eight-row tile.

It was rejected during screening. On the live layer profile it measured about
342 us for gate/up and 178 us for down, and the long sequential parity replay
was already abnormally slow by position 127. The canonical row-major mapping
remains the control. This is not a promotion result; it is a valid negative
compiler candidate showing that reducing activation staging does not overcome
the occupancy/register cost of a 1024-threadgroup on this workload.

The candidate is retained as an artifact-v2 plan ID and runtime variant so the
automatic selector can represent and reject it from evidence, rather than
silently deleting a failed search branch.

The next exact-Q4 transport candidate is now implemented as
`group32_q4_row_inline`. Each group stores its existing 16 Q4 code bytes and
2-byte FP16 scale contiguously. It has identical logical values, identical
artifact size, and uses dedicated live Metal kernels for gate/up and down.
The first five-fixture release screen is consistently positive:

| Plan | Median-ish wall range | Throughput range |
|---|---:|---:|
| row-major control | 4.942--4.978 ms/token | 200.88--202.34 tok/s |
| row-inline candidate | 4.607--4.647 ms/token | 215.20--217.06 tok/s |

The candidate also passed the complete sequential gate at
`0,1,7,31,63,127,255,511` with cosine `1.000000` and `8/8` argmax agreement.
This is the first representation candidate with a repeatable complete-model
win, but it is not canonical yet: it still needs randomized paired suite
evidence and the triangular held-out quality gate. The result is recorded in
`benchmarks/results/row-inline-q4-candidate-2026-08-22.json`.

The compiler-selection experiment is now also real rather than descriptive.
`qx-autotune` ran row-major and row-inline as two artifact-v2 candidates,
verified every quantized tensor decodes to identical codes and scales, and
selected row-inline from complete-model measurements:

| Plan | Complete-model wall | Peak physical footprint |
|---|---:|---:|
| row-major | 4.8592 ms/token | 111,559,280 B |
| row-inline | 4.5404 ms/token | 111,592,072 B |

The selection report is
`benchmarks/results/rowinline-selection-report-2026-08-22.json`. This is the
first compiler-proof artifact for a model/hardware/workload-specific physical
representation choice. It does not yet claim the final triangular quality
gate or parity with llama.cpp.

See [qx-validation-plan-implementation.md](qx-validation-plan-implementation.md)
for commands, schemas, and the proof checklist.
