# QX gate/up load/decode/dot instrumentation

Date: 2026-08-23  
Target: Granite 3.0 1B, Apple M5 Pro, batch-1 decode, context 512  
Artifact: `granite-m5-pro-q4-factorial-rowinline.qx`  
Artifact hash: `e3f2940482d5999d`

## What was added

The existing row-inline Q4 gate/up dispatch now has three diagnostic variants,
selected with `QX_V2_MOE_GATE_UP_INSTRUMENT`:

```text
unset / canonical     existing production kernel
load-only             read packed Q4 code bytes and inline FP16 scales
load-decode           additionally unpack nibbles and convert/scale Q4 values
dot-no-reduction      additionally multiply by the staged activation, with no
                      final SIMD reduction
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

Result records:

- `benchmarks/results/gateup-instrument-canonical-general-2026-08-23.json`
- `benchmarks/results/gateup-instrument-canonical-after-general-2026-08-23.json`
- `benchmarks/results/gateup-instrument-load-only-v4-general-2026-08-23.json`
- `benchmarks/results/gateup-instrument-load-decode-v3-general-2026-08-23.json`
- `benchmarks/results/gateup-instrument-dot-no-reduction-v2-general-2026-08-23.json`
