# Llama production probe factorial — 2026-08-24

This experiment attempted the symmetric four-cell probe required to separate
attention, MoE, and their interaction in llama.cpp:

```text
T00 = real attention + real MoE
T10 = captured attention + real MoE
T01 = real attention + captured MoE
T11 = captured attention + captured MoE
```

The capture path and GPU injection path are implemented in the nested llama
checkout at commit `436ea36`. The capture file contains 24 layers of 1024-wide
FP32 attention and MoE boundary outputs.

The run used prefill 384, 128 decode tokens, batch/ubatch 512, 99 GPU layers,
and flash attention. The observed medians were:

| Cell | GPU us/token median |
| --- | ---: |
| T00 | 2530.92 |
| T10 | 2355.75 |
| T01 | 2654.83 |
| T11 | 1139.90 |

These numbers are **not a valid Shapley ledger**. T11 was compiled into a
different scheduled graph and alternated between 805 and 690 graph nodes,
while T00/T10/T01 were 1471/1471/1495 nodes. The apparent T11 win is therefore
graph pruning, not a measured interaction effect.

The probe now preserves replacement input names, assigns replacement buffers
to the surrounding GPU backend, retains captured production boundaries, and
offers zero-copy graph scaffolding (`QX_PROBE_FULL_PADDING=1`). Those controls
are useful diagnostics, but they do not yet make T11 production-equivalent.

Decision: do not use these values to authorize an exact kernel change. The
next implementation is a true production-preserving boundary instrument that
keeps the graph/scheduler shape invariant across all four cells.
