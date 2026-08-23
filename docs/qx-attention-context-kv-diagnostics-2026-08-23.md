# Attention context and KV diagnostics — 2026-08-23

Workload: Granite 1B, batch 1, Apple M5 Pro, release QX binary.

## Complete-attention context sweep

Three serialized 50-iteration QX sweeps produced the following GPU medians
(µs). The runs were deliberately serialized; parallel Metal processes were
shown to contaminate the timing.

| Context | Run 1 | Run 2 | Run 3 | Observation |
| ---: | ---: | ---: | ---: | --- |
| 32 | 53.88 | 37.63 | 37.75 | high variance |
| 64 | 55.21 | 30.46 | 39.00 | high variance |
| 128 | 40.25 | 25.50 | 40.25 | high variance |
| 256 | 34.37 | 23.00 | 44.12 | high variance |
| 512 | 30.92 | 24.87 | 47.83 | high variance |

The medians are not monotonic in context. Consequently, this harness cannot
currently support a meaningful fixed-cost-plus-context-slope fit from GPU
timestamps. The wall-time medians are more stable but still do not establish
the requested slope. This is a measurement limitation, not evidence that
attention gets faster with longer context.

The earlier single-run stage split at context 512 showed QK 17.21 µs,
softmax 11.00 µs, and V 30.79 µs. Repeated stage runs varied from roughly
17–68 µs for the same complete attention measurement. That instability is too
large for selecting a new kernel.

## KV precision diagnostic

The existing diagnostic switches were tested with the complete-attention
sweep:

```text
QX_V2_KV_K_FP16=1   K FP16, V FP32
QX_V2_KV_V_FP16=1   K FP32, V FP16
```

At context 512, the observed GPU medians were not stable enough to establish
a speed benefit. The stage split likewise varied materially between repeated
baseline and asymmetric runs. These modes remain quality diagnostics only;
neither is a performance promotion candidate.

## llama comparison status

llama’s production attention is fused FlashAttention, so QK, softmax, and V
cannot be timed independently without replacing the production kernel. The
existing decode instrumentation reports complete attention near 10–12 µs per
layer across the useful context points, but the current `llama-bench` profiling
invocation emits multiple internal benchmark configurations. Those output
streams are not yet a clean matched context-sweep contract and are not used to
claim a slope.

## Decision

The V-cache path remains the most plausible attention substage target based on
the prior validated decomposition, but the current timing harness must be
stabilized before implementing another V kernel. The next instrumentation fix
should:

1. run one isolated configuration per process;
2. record command-buffer GPU start/end directly for that process;
3. discard samples with missing/zero GPU timestamps;
4. report median and p95 over independent process runs;
5. use the same context and decode position for QX and llama.

Until that exists, attention remains an open measured gap, not a safely
actionable K-vs-V optimization target.
