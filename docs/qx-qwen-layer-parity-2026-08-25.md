# Qwen component-runtime parity

Date: 2026-08-25  
Artifact: `qwen3-30b-a3b-a19pro-q2s4-capacity-candidate.qx`  
Artifact SHA-256: `7262a69d8230f0f78dcd12266035fedff5e9145dcace504c93fc27ccadef91c4`

## Result

QX now executes every mathematical component needed by one
Qwen3-30B-A3B decoder layer on Metal from the sealed artifact:

- Q4 Q/K/V projection with the 2,048-to-4,096 query expansion;
- per-head Q/K RMSNorm and split-half RoPE at nonzero positions;
- 32-query-head / 4-KV-head grouped-query attention over an FP16 KV history;
- Q4 output projection;
- post-attention RMSNorm and 128-way Q4 router projection;
- normalized top-8 expert selection;
- eight Q2S4 gate/up-SwiGLU-down experts and GPU weighted accumulation.

The Q2S4 expert kernel is Qwen-shaped rather than a renamed Granite kernel.
One 768-thread group computes gate/up and retains the activated vector in
threadgroup memory; each lane then strides over the 2,048 down rows. The eight
selected experts and accumulation execute in one command buffer without
expert-output readback.

## Real-artifact parity

The harness uses deterministic FP16 hidden input and synthetic prior FP16 KV
history, but all model matrices and norm vectors come from the complete mapped
artifact. CPU is the dequantized-QX numerical oracle, including FP16 stage
boundaries. These tests establish CPU/Metal artifact semantics, not BF16 model
quality.

| Layer / direct expert / position | Direct expert relative L2 | Q/K norm + RoPE relative L2 | Weighted top-8 MoE relative L2 | Result |
|---|---:|---:|---:|---|
| 0 / 0 / 37 | 0.00007621 | 0.00001500 | 0.00012655 | pass |
| 24 / 64 / 63 | 0.00005271 | 0.00000419 | 0.00011630 | pass |
| 47 / 127 / 7 | 0.00004916 | 0.00000028 | 0.00017482 | pass |

All projection and GQA comparisons were bit-exact or below `2.1e-7` relative
L2. Every router comparison selected identical expert IDs. Reported cosine was
`1.0` for every component above.

## Reproduce

```bash
cargo run --release -p qx-qwen-parity -- \
  /Users/blas/projects/qx/external/models/qwen3-30b-a3b-a19pro-q2s4-capacity-candidate.qx \
  --layer 0 --expert 0 --position 37
```

## What is not promoted yet

This is a correctness graph, not the production token graph:

- Q/K/V and output projections are separately submitted reference stages;
- RMSNorm, residual composition, top-8 selection, and KV insertion are still
  orchestrated by the harness;
- the GQA kernel deliberately recomputes scores for each output dimension;
- selected expert weights are uploaded instead of bound as persistent mapped
  no-copy slab ranges;
- embedding, persistent KV state, all-48-layer scheduling, final norm, untied
  LM head, argmax, and GPU-resident rollout are not connected;
- parity is against quantized QX math, not the BF16 teacher.

The next implementation milestone is a persistent `PreparedQwenLayer` which
binds mapped artifact ranges once and encodes all stages into a caller-owned
command graph. Only after 48 such layers produce a reference token should the
correctness kernels be replaced with performance mappings.
