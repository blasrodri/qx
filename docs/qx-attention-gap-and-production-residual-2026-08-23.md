# QX attention gap and production residual — 2026-08-23

Workload: Granite 1B, batch 1, context 512, decode positions 384–511.
Hardware: Apple M5 Pro.

## Attention substages

The QX stage benchmark used the current canonical three-stage attention
implementation at context 512. Values are GPU medians over 20 iterations.

| Stage | QX GPU µs | QX wall µs | ns/context | Effective GB/s |
| --- | ---: | ---: | ---: | ---: |
| QK dot products | 17.21 | 190.17 | 33.61 | 243.74 |
| max/sum/softmax | 11.00 | 177.25 | 21.48 | 5.96 |
| V accumulation | 30.79 | 190.33 | 60.14 | 68.11 |
| QK + softmax + V | 47.75 | 221.04 | 93.26 | 133.13 |
| live attention | 47.62 | 203.38 | 93.02 | — |

The stage sum and live result agree within measurement noise. This is useful:
there is no hidden large composition cost inside this isolated attention
measurement. The V accumulation is the largest QX substage, followed by QK;
softmax is not the dominant substage in this particular benchmark.

llama’s production attention is emitted as a fused FlashAttention kernel. Its
QK, softmax, and V substages are therefore **not separately measurable** with
the current instrumentation. The matched complete-attention boundary from
the grouped replay is 10.68 µs/layer for llama versus 21.92 µs/layer for QX.
That is a 11.24 µs/layer grouped deficit, or approximately 0.270 ms across
24 layers, but it does not identify which fused FlashAttention sub-operation
causes the difference.

## Production versus grouped replay

Grouped boundaries are diagnostic command-buffer replays, not production
execution. The following totals make that distinction explicit.

### QX

Using the sequential timeline medians:

```text
(19.50 + 4.00 + 21.92 + 10.87 + 12.21 + 95.42) × 24 + 382.25
= 4.316 ms grouped replay
```

The same-session production control is 4.225 ms GPU/token. Therefore:

```text
production − grouped replay = −0.091 ms/token
```

The grouped QX replay slightly overstates production time; this is not a
missing-work claim.

### llama

Using the grouped boundary medians:

```text
(21.32 + 10.08 + 10.68 + 10.67 + 18.08 + 62.09 + 12.42) × 24 + 153.50
= 3.642 ms grouped replay
```

The pinned production reference is approximately 2.590 ms GPU/token:

```text
production − grouped replay = −1.052 ms/token
```

This large negative value demonstrates that llama’s grouped replay has much
more diagnostic overhead or different overlap than its production graph. It
must not be interpreted as a CPU/cache deficit in llama, nor subtracted from
the QX production time.

The grouped ledger difference is therefore approximately 0.674 ms/token,
while the production GPU difference is approximately 1.635 ms/token. The
remaining ~0.961 ms is a schedule/replay-semantics discrepancy, not an
unattributed kernel that can be safely optimized from this table.

## Decision

1. Keep final-head structural work closed; all tested complete-token variants
   were slower than the canonical path.
2. Keep the current three-stage attention topology; it is already internally
   coherent and materially better than the former fused implementation.
3. Do not implement a softmax-only candidate: it is not the dominant QX
   substage, and llama’s fused kernel provides no matching substage evidence.
4. Treat V accumulation as the only plausible next attention candidate, but
   require a complete sequential A/B result of at least 0.10 ms/token before
   promotion. A candidate below 0.05 ms/token is archived.
5. Before selecting that candidate, continue the MoE production dataflow
   audit and improve the llama boundary instrumentation so the production
   residual is reduced rather than mistaken for an optimization target.

The current ledger supports attention as an open lane, but it does not support
the claim that attention alone explains the QX/llama production gap.
