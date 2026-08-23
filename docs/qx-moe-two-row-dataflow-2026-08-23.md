# QX MoE two-row dataflow experiment

Date: 2026-08-23  
Target: Granite 3.0 1B, Apple M5 Pro, batch-1 decode, context 512  
Artifact: `granite-m5-pro-q4-factorial-rowinline.qx`  
Artifact hash: `e3f2940482d5999d`  
Plan: `factorial-rowinline-qx`

## Hypothesis

The canonical row-inline Q4 gate/up consumer gives one SIMD group one output
row. The candidate kept the exact same row-inline representation, Q4 values,
routes, and four-stage MoE graph, but assigned two rows to each of two SIMD
groups in a 64-thread workgroup. Each group reused one activation load across
two row accumulators and performed the Q4 unpack/reduction locally.

This isolates inner dataflow from the previously tested geometry-only and
llama-like mapping candidates.

The candidate is enabled only with:

```text
QX_V2_MOE_TWO_ROW_DATAFLOW=1
```

## Correctness gate

The candidate passed the complete sequential V2 parity run at positions
`0, 1, 7, 31, 63, 127, 255, 511`:

```text
max KV key error:        0.009853
max KV value error:      0.001048
max attention error:     0.000127
max absolute logit:      0.009898
minimum logit cosine:    1.000000
argmax agreement:        8/8
```

The candidate is therefore numerically valid for the exact-Q4 promotion
ladder.

## Complete-token results

All runs used the release binary, the same artifact and `general.json` fixture,
one warmup, 128 measured sequential positions, one command buffer/token, and
461 dispatches/token.

| Run | Variant | Wall ms/token | GPU ms/token | tok/s |
|---|---|---:|---:|---:|
| 1 | canonical control | 4.7395 | 4.2930 | 210.99 |
| 1 | two-row dataflow | 4.7456 | 4.2973 | 210.72 |
| 2 | canonical control | 4.7538 | 4.2985 | 210.36 |
| 2 | two-row dataflow | 4.7304 | 4.2839 | 211.40 |
| 3 | canonical control | 4.7488 | 4.2980 | 210.58 |
| 3 | two-row dataflow | 4.7294 | 4.2876 | 211.44 |

Across the three paired runs:

```text
canonical mean:       210.64 tok/s, 4.7474 ms wall/token
candidate mean:       211.19 tok/s, 4.7351 ms wall/token
throughput change:    +0.26%
wall change:          -0.012 ms/token
```

The candidate's small mean advantage is below run-to-run noise and far below
the `0.15–0.20 ms/token` promotion threshold. It is not a meaningful
complete-model improvement.

Memory was unchanged at the allocation level:

```text
known Metal allocation: 802,373,632 bytes for both variants
physical footprint:     approximately 112.9–113.7 MB
```

## Decision

**Reject for canonical promotion.** Keep the implementation behind its feature
flag as a compiler candidate, but retain the existing row-inline Q4 path as the
canonical plan.

The result is important: changing activation reuse, accumulator ownership, and
Q4 unpack scheduling inside a 2-row/2-SIMD-group work unit does not recover a
material amount of latency when the row-inline representation and full
sequential graph are held constant. The next exact-Q4 search should therefore
change the representation and decoder jointly, or instrument the production
matrix path at a lower level, rather than repeat another nominally similar
threadgroup geometry sweep.

Result records:

- `benchmarks/results/two-row-dataflow-control-general-2026-08-23.json`
- `benchmarks/results/two-row-dataflow-candidate-general-2026-08-23.json`
- `benchmarks/results/two-row-dataflow-control-general-r2-2026-08-23.json`
- `benchmarks/results/two-row-dataflow-candidate-general-r2-2026-08-23.json`
- `benchmarks/results/two-row-dataflow-control-general-r3-2026-08-23.json`
- `benchmarks/results/two-row-dataflow-candidate-general-r3-2026-08-23.json`
