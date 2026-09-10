# QX verification benchmark report

For epoch sampling economics and the terminal Solana fraud-proof benchmark,
see `docs/verification-protocol.md`. The current executed SBF costs are 7,108
CU for a production-depth weight opening and 1,877 CU for an activation opening
plus one exact Q4 FP32 FMA (8,985 CU total over two transportable instructions).

- Date: 2026-09-10
- Machine/runtime: local Apple Metal device, release build
- Artifact: `granite-m5-pro-q4-rowinline-k4-vector-unroll4-down-short-context-active.qx`
- Artifact bytes: 751,009,792
- Context capacity: 512
Workload: 8 prompt tokens, 8 returned tokens, 15 executed tokens

All reported comparisons alternate baseline and verified execution on the same
loaded generator. Outputs are required to match. Timings are medians after four
or more warmups. The final selected-profile runs use 31 alternating repeats.

## Final implementation

The verified graph writes selected 4 KiB post-layer residuals into a 96 KiB
transient snapshot arena as part of the existing residual/RMS kernel. K/V stay
in the existing GPU cache. One token-end Metal dispatch hashes all 24 layers'
new K/V and the selected residuals concurrently into 1,536 bytes. The CPU reads
those digests only after the existing token wait and incrementally maintains
the KV, checkpoint, token, and job Merkle frontiers.

BLAKE3 state, key, and value leaves are 256-byte indexed chunks with separate
domains. State-tree parents use `BLAKE3(left || right)`; their typed leaf
domains prevent cross-tree ambiguity. Hot CPU-side KV, checkpoint, token, and
job trees also use domain-framed BLAKE3; immutable model, weight, and receipt
objects remain SHA-256. The execution-spec digest binds these choices and the
checkpoint interval. Metal and the CPU oracle match byte-for-byte, and normal
inference output is unchanged.

## Selected configuration

| Checkpoint interval | Baseline total | Verified total | Overhead | Baseline tok/s | Verified tok/s | Status |
|---:|---:|---:|---:|---:|---:|:---|
| 1 layer | 97.631 ms | 100.994 ms | 3.445% | 235.582 | 227.737 | pass |
| **2 layers (selected)** | **99.723 ms** | **102.917 ms** | **3.203%** | **230.639** | **223.481** | **pass** |
| 4 layers | 98.208 ms | 101.844 ms | 3.703% | 234.198 | 225.836 | pass |

These final runs used eight warmups, 31 alternating repeats, 8 prompt tokens,
and 16 returned tokens. The selected profile commits residuals after layers
1, 3, ..., 23 while retaining every layer's append-only K/V commitment. It
therefore widens initial dispute replay from one layer to two without sampling
or omitting any K/V history. Its deterministic trace root was
`63caefd734814b80e90ebfc78888a61a89c16141853b8c8777850e0e4d382dcb`.

## SHA-256 frequency experiment

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

The SHA-256 every-layer GPU counters were 58.773 ms baseline versus 63.386 ms
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
| Batched SHA-256 with residual snapshot fused into existing kernel | 9.582% |
| BLAKE3 leaves, domain-framed state-tree parents | 5.065% |
| BLAKE3 leaves, single-block typed-tree parents | 4.368% |
| BLAKE3 hot CPU trace trees, every-layer checkpoints | 3.445% |
| BLAKE3 hot trees, two-layer checkpoints | **3.203%** |

Batching and the narrowly scoped snapshot fusion removed 23 extra dispatches
per layer sequence and reduced the original overhead by more than two orders
of magnitude. The frequency sweep is almost flat at intervals 2 and 4, showing
that incremental K/V SHA-256 is now the limiting work rather than checkpoint
frequency, CPU work, synchronization, or residual copies.

The required `<5%` gate is **achieved**. The `<=3%` desired target is not yet
achieved; the best robust run is 3.203%, and the four-layer profile did not
improve it. Solana exposes `sol_blake3`, and the actual terminal program is now
measured at 8,985 CU across two transportable instructions; see the protocol
document for the proof and account sizes.

Reproduction command:

```text
cargo run --release -q -p qx-verify -- metal-benchmark \
  external/models/granite-m5-pro-q4-rowinline-k4-vector-unroll4-down-short-context-active.qx \
  --context 512 --new-tokens 16 --warmup 8 --repeats 31 --checkpoint-interval 2
```
