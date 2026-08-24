# QX payload-contiguous superblock implementation

## Result

The exact-Q4 MoE storage and Metal consumer proposed by the performance audit
is implemented. On the matched 384-prefill / 128-decode fixture it improves
complete-model throughput by **27.8%**, from 226.6 to 289.6 tok/s, while
reducing production GPU time by **24.4%**, from 3.935 to 2.975 ms/token.

| Metric | Promoted row-inline control | Superblock144 | Change |
|---|---:|---:|---:|
| Wall ms/token, three-run mean | 4.413 | 3.454 | -21.7% |
| GPU ms/token, three-run mean | 3.935 | 2.975 | -24.4% |
| Throughput, three-run mean | 226.6 tok/s | 289.6 tok/s | +27.8% |
| Artifact bytes | 751,009,792 | 751,009,792 | unchanged |
| Known Metal allocation | 802,373,632 | 802,373,632 | unchanged |

The matched llama.cpp control remains approximately 310--320 tok/s at depth
384. The candidate is therefore about 6.6--9.5% behind that throughput range,
versus roughly 29--34% for the previous QX path. This reaches the audit's
“within 10%” milestone on the measured three-run mean, but it is not yet exact
wall/GPU parity.

## What changed

The new `Q4Superblock144` physical representation stores each 256 weights as:

```text
[128 contiguous Q4 payload bytes][8 FP16 scales = 16 bytes]
```

This is an exact byte relayout of the existing group-32 Q4 values and scales.
It does not requantize weights and does not increase artifact size.

The specialized Metal gate/up and down consumers:

- issue one aligned 32-bit payload load per lane for eight adjacent weights;
- load eight scales once per row/block and distribute them with a SIMD shuffle;
- retain the corresponding eight activations in two `float4` registers;
- accumulate two output rows concurrently from the same activation registers;
- unroll the fixed four gate/up blocks and two down blocks; and
- produce sixteen rows per 256-thread group, halving threadgroup count and
  activation staging relative to the initial eight-row implementation.

## Attribution

The context-384 fixed-route profile confirms that the speedup comes from the
intended boundary:

| 24-layer boundary | Control GPU time | Superblock GPU time | Saving |
|---|---:|---:|---:|
| Gate/up | 1.269 ms | 0.597 ms | 0.672 ms |
| Full MoE | 1.970 ms | 1.043 ms | 0.927 ms |
| Complete fixed-route token | 4.320 ms | 3.370 ms | 0.950 ms |

The approximately 0.96 ms production-token saving recovers about two thirds
of the previously measured 1.47 ms QX/llama GPU gap.

## Correctness

Quantizer and artifact round-trip tests verify that payload bytes and FP16
scales are unchanged. Full CPU/GPU artifact parity passes at positions 0, 1,
7, 31, 63, 127, 255, and 511 with minimum logit cosine 1.0 and 8/8 argmax
agreement.

## Reproduce

Build the candidate artifact:

```bash
target/release/qx-compile \
  /Users/blas/projects/qx/external/models/granite-3.0-1b-a400m-instruct \
  --target apple-m5-pro --memory-budget 900mb --context 512 \
  --disable-q3 --expert-layout superblock-144 \
  --weight-storage mmap-no-copy \
  --moe-consumer canonical --moe-down-consumer canonical \
  --plan-id granite-m5-top8-superblock144 \
  --output /tmp/granite-m5-pro-q4-superblock144.qx
```

Run the production-shaped decode fixture:

```bash
target/release/qx-bench \
  /tmp/granite-m5-pro-q4-superblock144.qx \
  --v2-token --fixture benchmarks/fixtures/general.json --context 512
```

Run CPU/GPU parity:

```bash
target/release/qx-bench \
  /tmp/granite-m5-pro-q4-superblock144.qx \
  --model /Users/blas/projects/qx/external/models/granite-3.0-1b-a400m-instruct \
  --v2-parity --fixture benchmarks/fixtures/general.json --context 512
```

Raw samples are recorded in
`benchmarks/results/superblock144-implementation-2026-08-24.json`.

## Remaining parity work

MoE is no longer the obvious first bottleneck: its contextual boundary is now
roughly half the control time. The remaining production GPU difference to the
matched llama run is approximately 0.43 ms/token. The next implementation
should apply the same payload-contiguous, multi-row consumer family to dense
QKV and O projection weights, then remeasure before attempting an attention
rewrite. A 20-pair randomized promotion suite and held-out quality evaluation
remain required before changing the promoted artifact.
