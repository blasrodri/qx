# MoE neuron-cluster preflight

## Why this is the next exact lane

The current exact engine is approximately 228 tok/s on the latest release
smoke run, with the promoted row-inline Q4 representation and cooperative
unique-packet gate/up consumer. The exact down unique-packet consumer and the
final-head unique-packet consumer were both rejected by complete-model
evidence. Further local consumer variants therefore have diminishing expected
return.

The remaining structural exact hypothesis is to change the sparse execution
unit from a whole expert projection to an intermediate-neuron cluster:

```text
cluster j..j+C
  Gate rows: [j:j+C, :]
  Up rows:   [j:j+C, :]
  Down cols: [:, j:j+C]
```

The first candidates are `C=16` and `C=32`. They are small enough to keep
temporary activation state bounded while exposing a possible Gate/Up/Down
co-layout. This is not permission to create one giant fused MoE kernel. The
candidate must preserve GPU-resident state and may retain multiple dispatches
if that is faster.

## Implemented semantic boundary

`qx-compiler::ExpertWeights::run_neuron_clusters` is an exact CPU reference
oracle. It computes the same full Gate/Up, SiLU product, and Down result by
walking intermediate clusters. Unit tests compare widths 16, 32, 1, 3, and 5
against the existing whole-matrix reference, including a partial final
cluster.

Artifact-v2 now has an optional typed `MoeExecutionUnit` descriptor. The
descriptor can represent `MoeNeuronCluster { width }`, but no cluster artifact
is emitted and the runtime does not select it yet. This prevents an untested
packing format from entering the canonical path.

## GPU implementation gate

Before adding a cluster artifact section, the candidate must answer all of the
following with a small fixed-route pre-screen:

1. Does cluster packing reduce selected expert bytes or only rearrange them?
2. Does it preserve exact Q4 codes/scales and long-position parity?
3. Does a C=16/32 cluster consumer beat the current complete-token control by
   at least 0.15 ms/token, not merely win a fixed-route microbenchmark?
4. Does it avoid increasing peak physical footprint or decode-time allocation?

If the pre-screen cannot show plausible headroom, the branch is closed before
building a full artifact family. If it does, the next implementation should
add explicit cluster slabs and bounded Gate/Up/Down consumers, with planner
selection only after complete-model evidence.

## Product lane kept separate

Speculative decoding remains a separate product-throughput lane. It may turn
batch-1 routed matvecs into verifier work with higher activation reuse, but it
changes the user-visible execution contract and must be benchmarked against a
like-for-like llama speculative configuration. It should not be used to claim
exact autoregressive kernel parity.
