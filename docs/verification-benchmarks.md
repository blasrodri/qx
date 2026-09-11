# QX verification benchmark report

## Current hardened commitment format

The canonical token genesis is now a real pre-layer checkpoint containing the
embedding residual and prior-token KV root. Its snapshot is fused into the
embedding lookup, so verified execution still adds only 23 dispatches. Two
independent 31-repeat runs measured **2.939% and 4.201%** overhead (mean
**3.570%**) with identical trace root
`651e0f2f9e4b236b1add34d98bf69c35dd79ca610771c43e25375921d26cc2bd`.
The new interval-2 execution spec is
`01e47ff99355d9eca97bde30253739fe38fdfb70049871d4307daa89ffb9cd1d`.

On 2026-09-11, after adding canonical job/token genesis leaves, compact job
roots, and committed checkpoint-tree lengths, two independent 31-repeat runs
measured 103.794/107.176 ms and 101.832/105.019 ms uncommitted/committed:
**3.258% and 3.130% commitment overhead** (mean 3.194%). The interval-2
execution spec was
`2f8ab3141938f906004637107276c18ef73601a8467944ef0736646ace9e3246`
and the trace root was
`306b3c3df074aebfdf8fbc8ef94d356a05798b6c34106c9b63c6c137e1e6e59b`.
The 23-command-buffer run used 8 prompt tokens, 16 returned tokens, 23
executed tokens, 8 warmups, and the fixed consensus math profile. This passes
the 5% ceiling but not the desired 3% target.

For epoch sampling economics and the terminal Solana fraud-proof benchmark,
see `docs/verification-protocol.md`. The adversarial V2 path currently uses
1,784 CU once per epoch to finalize entropy, 8,397 CU to authenticate and open
a challenged result, 10,329 CU for both boundary proofs plus one midpoint
round, 7,656 CU to bind both operator traces to checkpoint endpoints, and
4,342 CU for the chained runtime-state-chunk scalar decision. Each further
binary-search level adds another two-party midpoint round. The compatibility
three-state Q4/FMA dispute is 32,075 CU.

- Date: 2026-09-10
- Machine/runtime: local Apple Metal device, release build
- Artifact: `granite-m5-pro-q4-rowinline-k4-vector-unroll4-down-short-context-active.qx`
- Artifact bytes: 751,009,792
- Context capacity: 512
Workload: 8 prompt tokens, 8 returned tokens, 15 executed tokens

After binding each predicted token into its token-trace leaf, a fresh 31-repeat
safe-precise run (8 prompt, 16 returned, 23 executed) measured 103.778 ms
uncommitted and 106.675 ms committed: **2.792% commitment overhead**. The new
execution-spec digest is
`f19851a93e31d52a7ac5a99f810cf8d9137ee46330428a68ebd8e4cd1d1ad45d`
and the trace root was
`3503482caef298f02192c140cc5b968e0e2e6b1de0e07173b55c98a2d22d5ba8`.

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

## Consensus arithmetic cost

The original measurements above used Metal's native `simd_sum`. The registered
execution profile now replaces every SIMD sum/max in the supported kernels with
a fixed shuffle-down tree (`16, 8, 4, 2, 1`) and disables Metal fast-math
reassociation. A matching CPU reduction oracle checks the same stage order.

Fresh 31-repeat measurements for the selected two-layer checkpoint profile:

| Arithmetic mode | Baseline | Verified | Commitment overhead | Trace root |
|:---|---:|---:|---:|:---|
| explicit tree, fast compiler | 98.400 ms | 101.570 ms | 3.222% | `144df0fb...e779f28` |
| explicit tree, precise compiler | 104.057 ms | 107.079 ms | **2.904%** | `e70da1d9...429606` |

Precise arithmetic itself costs 5.749% relative to the same explicit tree under
the fast compiler. Comparing the deployable precise+commitment path (107.079
ms) with the fast uncommitted path (98.400 ms) gives **8.820% total migration
cost**. Thus commitment overhead now meets the desired 3% target, while total
consensus-mode overhead does not yet meet the original 5% ceiling. The
`QX_METAL_PRECISE_MATH=0` escape hatch is diagnostic only and must not identify
a slashable execution profile.

### Relaxed-arithmetic candidate

Metal's modern compile controls let QX keep precise single-precision math
functions while selecting relaxed expression optimization independently. The
runtime exposes this candidate with `QX_METAL_RELAXED_ARITHMETIC=1`; it receives
a distinct execution-spec digest and does not replace the consensus-fixed default.
The runtime fixes the selected profile at first use so lazy kernel compilation
cannot mix arithmetic profiles within one process.

Two independent 31-repeat processes produced:

| Run | Uncommitted | Committed | Commitment overhead | Trace root |
|---:|---:|---:|---:|:---|
| 1 | 97.551 ms | 100.651 ms | 3.178% | `9b7ba037...9b88f59` |
| 2 | 97.774 ms | 100.923 ms | 3.221% | `9b7ba037...9b88f59` |

Fast-uncommitted control runs bracketing this experiment measured 96.189 and
98.128 ms. Comparing the means (97.159 ms fast control and 100.787 ms relaxed
committed) gives **3.734% total migration overhead** on this machine, versus
8.820% in the earlier safe-precise experiment. Every one of the 62 relaxed
repeats produced the same trace root and output tokens.

Safe arithmetic with only fast transcendental functions did not help: it
measured 102.241 ms uncommitted and 105.532 ms committed, so the former tax is
primarily expression optimization rather than `exp`/`rsqrt` function choice.

The selected K4 gate/up, down, and expert-accumulation kernels now spell their
dot steps as ordered explicit `fma` calls, matching the terminal verifier's
primitive. Their safe-precise and compiler-default trace roots did not change.

The relaxed profile is **not slashable yet**. Its recorded AIR retains unsafe
floating-point flags (`reassoc`, `nsz`, `arcp`, `contract`, and `afn`) even
though the five-stage SIMD shuffle dependency and explicit FMA calls remain in
the lowering. Promotion requires identical traces across supported Apple GPU
families, a pinned compiler/metallib identity, and terminal semantics for every
remaining relaxed operation.

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

The commitment-only `<5%` gate is achieved and its `<=3%` desired target is
achieved in safe-precise mode. The safe-precise comparison against the old fast
execution path remains 8.820%. The relaxed-precise candidate reduces the
measured total to 3.734%, below 5%, but the complete production gate remains
open until its cross-device consensus requirements pass.
See the protocol document for the receipt-bound Solana measurements.

Reproduction command:

```text
cargo run --release -q -p qx-verify -- metal-benchmark \
  external/models/granite-m5-pro-q4-rowinline-k4-vector-unroll4-down-short-context-active.qx \
  --context 512 --new-tokens 16 --warmup 8 --repeats 31 --checkpoint-interval 2
```

Prefix the command with `QX_METAL_RELAXED_ARITHMETIC=1` to reproduce the
reduced-overhead candidate. The benchmark prints the active profile and its
execution-spec digest before the timing results.

## Fixed consensus transcendental profile

The default profile now uses a range-reduced degree-7 binary32 exponential and
a fixed integer seed plus four Newton refinements for reciprocal square root.
Every constant is specified as raw binary32 bits. A Metal conformance kernel
checks the GPU outputs bit-for-bit against the Rust/SBF implementation.
`QX_METAL_NATIVE_MATH=1` retains native precise functions only as an A/B
diagnostic; it produces a distinct execution-spec digest and is not the default.

On the same Granite artifact, a 9-repeat, 23-executed-token fixed-math run
measured 103.099 ms without commitments and 106.016 ms with commitments:
**2.829% commitment overhead**. The corresponding native run generated the
same 16 tokens. A 512-position held-out PTB teacher-forced comparison measured:

| Profile | NLL | PPL |
|:---|---:|---:|
| Native precise | 3.4243704835 | 30.703310526 |
| Consensus fixed V1 | 3.4243705322 | 30.703312022 |

The relative PPL change was `4.8735e-8`; eight sampled logit vectors retained
100% top-1, top-5, and top-10 agreement and cosine
`0.9999999999997112`. This is a strong numerical regression check, not a
replacement for the sealed multi-domain/task promotion gate.
