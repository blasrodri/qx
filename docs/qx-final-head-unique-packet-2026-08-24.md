# Final-head unique-packet consumer

## Scope

This was an exact-Q4 consumer experiment for the final vocabulary projection.
It preserved the existing logits-partial/argmax graph and changed only the
packed-code and scale ownership inside the vocabulary GEMV. It was not a new
LM-head/argmax fusion or a change to the final reduction topology.

## Evidence

The release build was measured on the general context-512 fixture after
prefill, with the same artifact, dispatch graph, and one-command-buffer
contract:

| Plan | Wall ms/token | GPU ms/token | tok/s |
|---|---:|---:|---:|
| Canonical final head | 4.374 | 3.903 | 228.60 |
| Unique-packet final head | 4.389 | 3.904 | 227.84 |

The candidate was exact-equivalent through the full sequential parity gate:

```text
positions:       0, 1, 7, 31, 63, 127, 255, 511
max KV key:      0.009853
max KV value:    0.001048
max attention:   0.000127
max logits:      0.009898
min cosine:      1.000000
argmax:          8/8
```

## Decision

Reject for promotion. The candidate is numerically valid, but the first
complete-token measurement is approximately 0.015 ms/token slower. That is
consistent with the final-head lane's earlier structural-fusion rejections:
the final head has not shown a profitable replacement family. Keep the
implementation available as a diagnostic candidate, but retain the canonical
final-head path and do not spend further optimization cycles here until a
new, materially different hypothesis appears.

The result is still useful compiler evidence: exact cooperative packet
consumption is not automatically profitable for every Q4 consumer. Promotion
must remain a complete-model decision, even when the consumer is exact and
the local dataflow resembles the winning MoE gate/up mechanism.
