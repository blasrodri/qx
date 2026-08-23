# QX gate/up load/decode/dot instrumentation

Date: 2026-08-23  
Target: Granite 3.0 1B, Apple M5 Pro, batch-1 decode, context 512  
Artifact: `granite-m5-pro-q4-factorial-rowinline.qx`  
Artifact hash: `e3f2940482d5999d`

## What was added

The existing row-inline Q4 gate/up dispatch now has eight diagnostic variants,
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
contiguous-load-only  keep the production mapping and load count, but use a
                      unique contiguous per-threadgroup stream
load-two-groups       keep the real routed stream but interleave two
                      independent group-load sequences per lane
load-unique-packet    cooperatively load each packet's 16 code bytes and
                      2 scale bytes once across the SIMD group
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
weight traffic:       8 × 1024 × 32 × 18 = 4,718,592 bytes
                      (4.5 MiB: 4 MiB codes + 512 KiB scales)
                      (576 KiB per selected expert)
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

## Unique contiguous production-shaped control

The contiguous control keeps the 1,024 threadgroups/layer, 256 threads per
threadgroup, activation staging, lane-to-weight mapping, 32 source code reads,
32 scale-helper calls, and diagnostic sink. It changes only the expert base:
each threadgroup consumes a unique sequential 18-byte-group stream region
instead of an expert/row-derived region. The stream covers the full 4.5 MiB
top-8 encoded traffic for the layer; groups do not alias a small repeated
buffer.

| Variant | 24-layer gate/up GPU µs | µs / layer |
|---|---:|---:|
| real row-inline load-only | 1,263.42 | 52.64 |
| unique contiguous load-only | 1,259.58 | 52.48 |

The difference is only 3.84 µs over 24 layers, within the scale of this
separate-process diagnostic. Therefore removing the routed expert/row physical
placement does not currently expose a material locality win in this mapping.
The result does not prove that raw traffic is cheap; it says that simple
contiguity is not the missing mechanism.

The exact search now moves toward load issue width, vector/scalar transaction
shape, independent outstanding loads, and dependency-chain/ILP behavior in the
production decoder. A new representation is justified only if it changes one
of those properties, not merely because it makes expert regions adjacent.

## Load-issue interleaving control

`load-two-groups` preserves the real routed row-inline addresses, activation
staging, 256-thread dispatch, and total code/scale reads. It changes only the
inner diagnostic loop to process two independent 32-weight groups per
iteration, testing whether a wider in-flight load window hides memory latency.

| Variant | 24-layer gate/up GPU µs | µs / layer |
|---|---:|---:|
| load-only | 1,263.42 | 52.64 |
| load-two-groups | 1,434.50 | 59.77 |

The candidate is slower by 171.08 µs over 24 layers. The simple interleaving
strategy therefore does not improve load-level parallelism on this production
mapping; its extra loop/control/register work outweighs any latency hiding.
This closes the low-cost “process two groups per lane” hypothesis without
changing the canonical runtime.

The remaining open low-level work is a more fundamental load/decode dataflow:
vector transaction width, pointer progression, independent rows in flight, or
a representation whose decoder can consume multiple Q4 packets with fewer
address and conversion dependencies. Those should be tested with complete
production-shaped controls, not inferred from this rejected diagnostic.

## Cooperative packet-load diagnostic

The static count showed that the canonical load-only diagnostic repeatedly
reads the same 16 code bytes and 2-byte scale across the 32 lanes of each
group. `load-unique-packet` keeps the same expert/row/group addresses and
activation staging, but assigns the 16 code bytes of two adjacent groups to
lanes 0..31 and loads each group's scale from only two lanes. It still sinks a
checksum and performs no model arithmetic, so it is diagnostic only.

| Variant | 24-layer gate/up GPU µs | µs / layer |
|---|---:|---:|
| load-only | 1,263.42 | 52.64 |
| load-two-groups | 1,434.50 | 59.77 |
| load-unique-packet | **92.21** | **3.84** |

This is the first result that isolates a large, concrete low-level mechanism:
the current production-shaped load path is issuing redundant code/scale reads
across lanes. The result is not directly convertible into end-to-end speedup,
because the candidate omits Q4 decode, FMA, SIMD reduction, and model stores.
It does justify building an exact semantic cooperative packet consumer next.
That kernel must decode both nibbles from each uniquely loaded code byte,
broadcast one scale per packet, accumulate the corresponding activation pair,
and pass the full sequential parity gate before any promotion decision.

## Exact semantic cooperative packet consumer

The feature-flagged semantic candidate is enabled with
`QX_V2_MOE_UNIQUE_PACKET=1`. It uses the same row-inline Q4 bytes, expert IDs,
256-thread dispatch, activation staging, and one output row per SIMD group.
For every pair of 32-weight groups, each lane loads one unique packed code
byte; lanes 0 and 16 load the two packet scales, which are broadcast within
their respective 16-lane halves. Each lane decodes both nibbles and performs
the corresponding two activation products before the normal SIMD reduction.

The long sequential parity gate passed:

```text
checkpoint positions:   0, 1, 7, 31, 63, 127, 255, 511
max KV key error:        0.009853
max KV value error:      0.001048
max attention error:     0.000127
max absolute logit:      0.009898
minimum logit cosine:    1.000000
argmax agreement:        8/8
```

Same-workload release throughput measurements were:

| Plan | Wall ms/token | GPU ms/token | Equivalent tok/s | CBs | Dispatches |
|---|---:|---:|---:|---:|---:|
| canonical row-inline | 4.830 | 4.340 | 207.02 | 1 | 461 |
| unique packet | 4.437 | 3.977 | 225.38 | 1 | 461 |
| unique packet, repeat | 4.374 | 3.908 | 228.64 | 1 | 461 |

The two candidate runs are approximately 9–10% faster than the immediately
paired canonical run. This is the first load-level result that survives into
the complete semantic graph. It is still not canonical: the feature flag must
be represented as a compiler plan candidate and selected through the normal
multi-tier benchmark/quality/memory promotion process.

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
- `benchmarks/results/gateup-instrument-averaged-contiguous-load-only-2026-08-23.json`
- `benchmarks/results/gateup-instrument-averaged-load-two-groups-2026-08-23.json`
- `benchmarks/results/gateup-instrument-averaged-load-unique-packet-2026-08-23.json`
- `benchmarks/results/unique-packet-v2-token-2026-08-23.json`
- `benchmarks/results/unique-packet-v2-token-r2-2026-08-23.json`
- `benchmarks/results/canonical-v2-token-after-unique-packet-2026-08-23.json`
