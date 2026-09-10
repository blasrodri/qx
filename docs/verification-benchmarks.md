# QX verification benchmark report

- Date: 2026-09-10
- Machine/runtime: local Apple Metal device, release build
- Artifact: `granite-m5-pro-q4-rowinline-k4-vector-unroll4-down-short-context-active.qx`
- Artifact bytes: 751,009,792
- Context capacity: 512
Workload: 8 prompt tokens, 8 returned tokens, 15 executed tokens

All reported comparisons alternate baseline and verified execution on the same
loaded generator. Outputs are required to match. Timings are medians after four
warmups. The final every-layer and frequency runs use seven or nine repeats.

## Final implementation

The verified graph writes each 4 KiB post-layer residual into a 96 KiB
transient snapshot arena as part of the existing residual/RMS kernel. K/V stay
in the existing GPU cache. One token-end Metal dispatch hashes all 24 layers
concurrently into 1,536 bytes. The CPU reads those digests only after the
existing token wait and incrementally maintains the KV, checkpoint, token, and
job Merkle frontiers.

SHA-256 state, key, and value leaves are 256-byte indexed chunks with separate
domains. Metal and the CPU oracle match byte-for-byte. Normal inference output
is unchanged.

| Checkpoint interval | Baseline total | Verified total | Overhead | Baseline tok/s | Verified tok/s | Status |
|---:|---:|---:|---:|---:|---:|:---|
| 1 layer | 63.450 ms | 69.530 ms | 9.582% | 236.407 | 215.736 | fail |
| 2 layers | 63.385 ms | 67.586 ms | 6.628% | 236.650 | 221.939 | fail |
| 4 layers | 62.819 ms | 67.199 ms | 6.973% | 238.782 | 223.217 | fail |

Intervals 2 and 4 are cost probes: K/V is still hashed after every layer, while
the residual hash is produced at the selected checkpoint frequency. They do
not currently emit a production trace root. Interval 1 is the specified V1
path and produced the deterministic trace root
`83131d5f0acb8d9e7c80680d7796465cde1e5b9533d54f080cdfc26b921d05b0`.

The final every-layer GPU counters were 58.773 ms baseline versus 63.386 ms
verified, with 5,850 versus 5,865 dispatches over the workload. CPU encode time
decreased slightly (1.976 ms to 1.799 ms), while CPU wait tracked GPU time. No
per-layer command buffers or waits were added.

## Optimization history and conclusion

Profiling deliberately retained failed intermediate measurements:

| Design | Overhead |
|---|---:|
| One serial SHA-256 thread per layer | 1,142.533% |
| Parallel 256-byte chunks, one layer dispatch | 411.618% |
| Three SIMD groups per layer dispatch | 161.229% |
| Batched 24-layer hash with separate copy dispatches | 10.535% |
| Batched hash with residual snapshot fused into existing kernel | 9.582% |

Batching and the narrowly scoped snapshot fusion removed 23 extra dispatches
per layer sequence and reduced the original overhead by more than two orders
of magnitude. The frequency sweep is almost flat at intervals 2 and 4, showing
that incremental K/V SHA-256 is now the limiting work rather than checkpoint
frequency, CPU work, synchronization, or residual copies.

The required `<5%` gate is **not achieved**. Advancing the normal path requires
a measured change to the commitment hash strategy (for example a
protocol-versioned, GPU-efficient hash with an acceptable Solana verification
cost) or a materially faster SHA-256 Metal implementation. Later gateway and
Solana protocol work cannot be assumed to improve this normal-path cost.

Reproduction command:

```text
cargo run --release -q -p qx-verify -- metal-benchmark \
  external/models/granite-m5-pro-q4-rowinline-k4-vector-unroll4-down-short-context-active.qx \
  --context 512 --new-tokens 8 --warmup 4 --repeats 9 --checkpoint-interval 1
```
