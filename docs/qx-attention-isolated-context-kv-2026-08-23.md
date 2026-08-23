# Isolated attention context/KV results — 2026-08-23

The attention benchmark now supports `--v2-attention-stage qk|softmax|v|combined|live`.
Each invocation creates one fresh process and Metal context. This removes the
stage-history contamination that made the earlier in-process timings unstable.

## Context scaling

GPU median microseconds, 50 iterations per fresh process. Columns are in
context order 64, 128, 256, 512.

| Stage | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: |
| QK / K-cache reads | 12.25 | 9.17 | 11.08 | 15.83 |
| softmax | 14.62 | 14.42 | 14.62 | 15.50 |
| V-cache accumulation | 36.04 | 37.25 | 39.17 | 43.50 |
| QK + softmax + V | 54.63 | 56.62 | 62.29 | 67.50 |
| full attention | 54.63 | 56.50 | 62.25 | 67.75 |

The complete attention path is now monotonic. Softmax is effectively fixed
cost over this range. V grows by approximately 7.5 µs from context 64 to 512;
QK grows by approximately 3.6 µs. The full path grows by approximately 13.1
µs.

## K/V precision diagnostic at context 512

These are diagnostic-only runs; the canonical artifact remains FP32 K/V.

| Mode | QK µs | V µs | Full attention median µs |
| --- | ---: | ---: | ---: |
| FP32 K / FP32 V | 15.79 | 43.37 | 67.83 |
| FP16 K / FP32 V | 14.04 | 43.38 | 65.87 |
| FP32 K / FP16 V | 15.88 | 43.13 | 67.25 |

K FP16 reduces QK time by approximately 11% and full attention by roughly 3%.
V FP16 does not produce a meaningful V or full-attention improvement. This is
evidence that K-cache reads/addressing are more traffic-sensitive than V
accumulation on this workload. It is not by itself a promotion decision: KV
precision remains governed by the existing numerical-quality contract.

## Decision

The next attention implementation should target K-cache access/layout. Do not
write another V reduction kernel yet. The next diagnostic is a benchmark-only
K-cache layout control with identical dimensions and arithmetic:

```text
current:  [position][kv_head][head_dim]
control:  [kv_head][position][head_dim]
```

The control must be measured with the same QK work and direct GPU timestamps.
Only if the layout control improves complete attention by at least 0.10
ms/token in the full decode graph should it become a production candidate.
