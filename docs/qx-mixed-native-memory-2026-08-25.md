# Consumer-native mixed-bit weight experiment

Date: 2026-08-25  
Target: Apple M5 Pro  
Model: IBM Granite 3.0 1B-A400M Instruct  
Workload: batch-1 greedy, context 512, 384-token prefill, 128-token decode

## Result

QX can remove **42,008,576 bytes (40.06 MiB)** from the canonical mapped
weight payload while retaining one consumer dispatch per MoE stage. The tested
capacity tier uses 375 compact Q3 expert tensors and 1,161 Q4 expert tensors;
dense tensors remain Q4 superblock-144.

| Metric | canonical Q4 | mixed Q3/Q4 40 MiB tier | change |
|---|---:|---:|---:|
| Mapped artifact | 751,009,792 | 709,001,216 | -42,008,576 |
| Unique weight payload | 750,900,928 | 708,892,352 | -5.59% |
| Known logical bytes at context 512 | 789,165,312 | 747,156,736 | -5.32% |
| Dispatches/token | 390 | 390 | unchanged |
| Paired throughput mean | 395.27 tok/s | 354.44 tok/s | -10.33% |
| GPU mean | 2.527 ms/token | 2.818 ms/token | +11.52% |

The throughput matrix is a same-session AB/BA-style control after the machine
had reached a lower sustained performance state. An earlier paired 32 MiB
matrix measured 414.14 versus 366.31 tok/s; both matrices put the mixed tier
roughly 10--12% behind canonical QX.

## Quality frontier

The allocator was tested on two disjoint 1,023-position PTB windows. These are
local frontier probes, not the complete multi-domain promotion suite.

| Q3 byte target | window A PPL change | window B PPL change | aggregate PPL change | gate |
|---:|---:|---:|---:|---|
| 32 MiB | -0.85% | +1.48% | +0.31% | pass |
| 40 MiB | +0.35% | +1.37% | +0.86% | pass |
| 48 MiB | +0.35% | +2.62% | +1.48% | fail |
| 64 MiB | not retained | not retained | first-window +6.20% | fail |

The existing plan contract allows at most 1% aggregate and 2% per-domain
relative PPL change. Forty MiB is therefore the largest tested passing tier;
48 MiB is a measured rejection.

## Structural changes

- The compiler can select Q3 tensors by a target byte saving, ranked by traced
  incremental Q3-over-Q4 error times route importance per byte.
- Q4 experts remain in the promoted payload-contiguous 144-byte superblocks.
- Q3 experts remain compact: 12 code bytes plus a separate FP16 scale for each
  group of 32 weights.
- A single Metal consumer reads the expert representation tag once per
  threadgroup and consumes either Q4 or Q3 directly. The tag is uniform for
  every lane in that expert threadgroup.
- Gate/up and down each retain one projection dispatch. No conversion or
  expanded GPU weight buffer is created.

The experiment also fixed a correctness bug in the older dual-dispatch path:
the second representation kernel wrote zeros into slots owned by the first
kernel. Before the fix, a 64 MiB candidate reached PPL 2.68 million; after the
fix its first-window PPL was 23.63 versus 22.25 for Q4.

## Interpretation

This is a real model-capacity tier, but not yet a change in hardware class for
this 751 MB model. It removes 5.6% of weight bytes and 5.3% of the context-512
logical footprint. It establishes that consumer-native mixed precision can be
both executable and quality-gated without dispatch inflation.

The Q3 decoder inside the unified kernel was the next performance target. That
follow-up is implemented as Q3-112 plus a packed vector decoder and is recorded
in `docs/qx-q3-superblock112-2026-08-25.md`. It recovers 0.187 ms/token of the
retained pre-change mixed-tier GPU time and raises its measured throughput by
7.1% without changing artifact size or the selected tensors.

## Reproduce the passing frontier

```bash
target/release/qx-compile \
  /Users/blas/projects/qx/external/models/granite-3.0-1b-a400m-instruct \
  --target apple-m5-pro \
  --trace /Users/blas/projects/qx/external/results/granite-calibration.qxtrace \
  --memory-budget 900mb --context 512 --quality-loss 1.0 \
  --expertwise-q3 --expertwise-q3-savings 40mb \
  --expert-layout superblock-144 --dense-layout superblock-144 \
  --command-buffers-per-token 2 --rollout-depth 2 \
  --kv-precision fp32-k-fp16-v --weight-storage mmap-no-copy \
  --plan-id granite-m5-mixed-superblock-q3-s40m \
  --output /tmp/qx-mixed-superblock-q3-s40m.qx

target/release/qx-bench \
  /tmp/qx-mixed-superblock-q3-s40m.qx \
  --v2-token --fixture benchmarks/fixtures/general.json --context 512
```
