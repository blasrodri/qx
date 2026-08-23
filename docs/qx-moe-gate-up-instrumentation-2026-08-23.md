# QX gate/up load/decode/dot instrumentation

Date: 2026-08-23  
Target: Granite 3.0 1B, Apple M5 Pro, batch-1 decode, context 512  
Artifact: `granite-m5-pro-q4-factorial-rowinline.qx`  
Artifact hash: `e3f2940482d5999d`

## What was added

The existing row-inline Q4 gate/up dispatch now has five diagnostic variants,
selected with `QX_V2_MOE_GATE_UP_INSTRUMENT`:

```text
unset / canonical     existing production kernel
load-only             read packed Q4 code bytes and inline FP16 scales
load-decode           additionally unpack nibbles and convert/scale Q4 values
dot-no-reduction      additionally multiply by the staged activation, with no
                      final SIMD reduction
staging-only          preserve activation staging/barrier, but do not touch
                      expert weights
address-only          calculate production routed row/group/byte addresses,
                      but do not dereference expert weights
```

The variants preserve the production path's:

- row-inline Q4 addresses;
- 8 selected experts and frozen route IDs;
- 1,024 threadgroups per layer;
- 256 threads per threadgroup;
- activation staging shape;
- buffer bindings and dispatch count;
- normalized and non-normalized entry points.

The three diagnostic paths are non-semantic: they must only be used through
the gate/up boundary profiler because they do not produce valid gate/up
outputs for the following SiLU/down stages.

All five variants retain the same 24-layer fixed-route boundary, dispatch
geometry, activation staging shape, and buffer bindings. The new staging and
address controls are deliberately not alternate production kernels; they are
cost-isolation instruments.

## Static QX work count

For one output row and one group of 32 weights, the production row-inline
format is:

| Quantity | QX row-inline |
|---|---:|
| packed code bytes | 16 |
| inline scale bytes | 2 FP16 |
| logical weights | 32 |
| unique code-byte addresses | 16 |
| source code-byte reads | 32 lanes read `local / 2` |
| source scale reads | 32 lanes call the 2-byte scale helper |
| nibble unpack operations | 32 shift/mask pairs |
| input values consumed | 32 |
| Q4 scale conversions | 32 |
| dot products | 32 |
| SIMD reductions | 1 `simd_sum` per output row |
| output stores | 1 per output row |

For one selected top-8 layer this expands to:

```text
weight traffic:       8 × 1024 × 32 × 18 = 589,824 bytes
                      (576 KiB: 512 KiB codes + 64 KiB scales)
output rows:           8 × 1024
row/group units:       8 × 1024 × 32
threadgroups:          8 × 128 = 1,024
threads:               1,024 × 256
activation staging:    1,024 × 4 KiB = 4 MiB of repeated reads/writes
```

The activation vector is small, but the current production mapping stages it
once for every 8-row threadgroup. The two-row experiment showed that changing
this reuse pattern alone did not improve complete-token latency.

The corresponding llama instruction/transaction counts are not filled in:
the pinned llama commit and its Metal source are not present in this workspace,
so an exact `MUL_MAT_ID` source-level count would be fabricated. The existing
source audit establishes its batch-1 mapping (`N_R0_Q4_K=2`, `N_SG_Q4_K=2`),
but not the complete instruction stream.

## Exploratory contextual measurements

These are 24-layer frozen-route gate/up-boundary measurements. They use the
same artifact, context, 24 dispatches, and one command buffer. The diagnostic
paths are not valid complete-model executions, so their fixed-MoE and complete
token rows are intentionally not used for performance decisions.

| Variant | Gate/up GPU µs / 24 layers | µs / layer | Dispatches |
|---|---:|---:|---:|
| canonical control | 1,436.17 | 59.84 | 24 |
| load-only, first exploratory run | 1,448.42 | 60.35 | 24 |
| load-only, later run | 5,509–6,195 | 229.5–258.1 | 24 |
| load+decode, corrected wrapper | 4,827 | 201.1 | 24 |
| dot/no-final-reduction | 5,133 | 213.9 | 24 |

The large spread in repeated load-only runs means the current data is not a
usable quantitative decomposition yet. It is a diagnostic warning, not proof
that Q4 decoding costs 4 ms. The next measurement must randomize canonical and
variant runs in paired blocks while monitoring thermal/power state, and should
capture at least three controls for every variant in the same session.

The useful conclusion at this stage is narrower:

> The instrumentation is now at the production address/dispatch boundary, but
> the diagnostic kernels must be stabilized and paired before attributing the
> canonical-vs-llama gap to decoding, dot products, or reduction.

## llama instrumentation status

No local llama Metal source was available to add equivalent `MUL_MAT_ID`
instrumentation. The workspace therefore does not claim a QX-vs-llama
load/decode/dot attribution. The next valid comparison requires either the
pinned llama source checkout or an equivalent instrumentable build.

## Paired-run follow-up

The randomized follow-up confirmed that machine state must be part of the
measurement contract. In one interleaved sequence, canonical gate/up moved
from `1,446.75 GPU µs / 24 layers` to `3,521.79 GPU µs / 24 layers` without a
source or artifact change. The load+decode variant measured `6,025.71 GPU µs`
in the same sequence.

The host reported AC power, no recorded thermal warning, and no recorded
performance warning, but that does not establish a constant GPU power state.
The contextual profiler now records `sw_vers`, `pmset -g batt`,
`pmset -g therm`, and `memory_pressure -Q` in every JSON report and prints
them with the measurement.

Until randomized paired blocks remain stable across the recorded state, these
numbers are classified as instrumentation validation only. The profiler must
discard or repeat pairs spanning a power/thermal transition before deriving a
decoder-vs-dot cost.

## Averaged follow-up

Using two warmup samples and ten measured samples per process reduced the
single-sample swings enough to expose a useful provisional ordering:

| Variant | 24-layer gate/up GPU µs | Difference vs canonical |
|---|---:|---:|
| canonical | 1,471.37 | baseline |
| load-only | 1,263.42 | -207.95 |
| load+decode | 1,324.29 | -147.08 |
| dot/no-final-reduction | 1,419.37 | -52.00 |

These are still diagnostic, nonsemantic kernels and were run in separate
processes, so they are not a promotion result. Within this instrumented
mapping, decode, FMA work, and final SIMD reduction are not individually large
enough to explain the roughly 0.8 ms QX-vs-llama gate/up gap. However, the
load-only path is not a pure raw-byte cost: it still includes the shared
dispatch, activation staging, routed expert selection, address generation,
packed loads, and diagnostic sink. Therefore it would be too strong to say
that raw weight traffic is unimportant.

## Staging/address/load split

The next surgical split keeps the exact production boundary and adds controls
for the shared pre-load work:

| Variant | 24-layer gate/up GPU µs | µs / layer | Interpretation |
|---|---:|---:|---|
| staging-only | 843.38 | 35.14 | staging/barrier plus minimal sink |
| address-only | 642.46 | 26.77 | staging plus routed row/group/byte address arithmetic |
| load-only, averaged follow-up | 1,263.42 | 52.64 | real packed code/scale dereferences plus shared diagnostic work |
| canonical, averaged follow-up | 1,471.37 | 61.31 | production decode/dot/reduction path |

The measurements are separate release processes and are not additive cost
components. The surprising ordering—address-only below staging-only—is itself
useful: staging is not the dominant explanation, and address generation is not
expensive enough in this control to account for the load-only/canonical gap.
The load-only control is about 621 µs above address-only over 24 layers, while
canonical is about 208 µs above load-only. This is evidence that actual packed
code/scale transactions remain a major live cost, but not a clean bandwidth
measurement because each control has different compiler-visible work and sink
behavior.

The next valid representation experiment is therefore a same-byte contiguous
fake-stream control using the production mapping. If that is much cheaper than
load-only, physical placement/transaction shape remains open. If it is close,
the remaining cost is more likely coupled address/decoder/vectorization
behavior. Neither result justifies a new format until it is confirmed in the
complete semantic graph.

Result records:

- `benchmarks/results/gateup-instrument-canonical-general-2026-08-23.json`
- `benchmarks/results/gateup-instrument-canonical-after-general-2026-08-23.json`
- `benchmarks/results/gateup-instrument-load-only-v4-general-2026-08-23.json`
- `benchmarks/results/gateup-instrument-load-decode-v3-general-2026-08-23.json`
- `benchmarks/results/gateup-instrument-dot-no-reduction-v2-general-2026-08-23.json`
- `benchmarks/results/gateup-instrument-paired-canonical-a-2026-08-23.json`
- `benchmarks/results/gateup-instrument-paired-canonical-b-2026-08-23.json`
- `benchmarks/results/gateup-instrument-paired-load-decode-a-2026-08-23.json`
- `benchmarks/results/gateup-instrument-averaged-canonical-2026-08-23.json`
- `benchmarks/results/gateup-instrument-averaged-load-only-2026-08-23.json`
- `benchmarks/results/gateup-instrument-averaged-load-decode-2026-08-23.json`
- `benchmarks/results/gateup-instrument-averaged-dot-no-reduction-2026-08-23.json`
- `benchmarks/results/gateup-instrument-averaged-staging-only-2026-08-23.json`
- `benchmarks/results/gateup-instrument-averaged-address-only-2026-08-23.json`
