# Clustered shared-basis MoE analysis — 2026-08-23

This is an offline candidate-generation study against the original Granite
BF16 safetensors checkpoint. It does not change the canonical QX artifact and
does not establish teacher-forced quality.

The 32 experts in each layer were clustered with deterministic greedy
initialization followed by a short k-means refinement. Within each cluster,
the analysis measures the Frobenius energy explained by a cluster mean plus a
rank-r basis of the centered expert residuals.

## Aggregate result

| Tensor | Clusters | Rank 4 | Rank 8 | Rank 16 |
| --- | ---: | ---: | ---: | ---: |
| gate/up | 1 | 17.30% | 30.55% | 55.98% |
| gate/up | 2 | 20.65% | 33.79% | 59.06% |
| gate/up | 4 | 27.24% | 40.20% | 65.16% |
| down | 1 | 17.31% | 30.52% | 55.72% |
| down | 2 | 20.66% | 33.75% | 58.79% |
| down | 4 | 27.25% | 40.11% | 64.88% |

Clustering helps, but not enough at practical low ranks. Even four bases with
rank 16 leave roughly 35% of total weight energy unexplained. Rank 4 and rank
8 leave roughly 60–73% unexplained, before accounting for activation-weighted
error, quantization, routing, and accumulation effects.

## Decision

Do not build the shared-base plus low-rank-residual runtime yet. This scan does
not meet the pre-screen for a credible traffic reduction, and the apparent
compression would require a quality-sensitive teacher-forced validation before
it could be considered for the artifact.

The branch remains useful as a compiler research direction. A future attempt
would need activation-weighted fitting from real routed hidden-state traces,
more expressive block-sparse residuals, or a trained/shared representation.
The next runtime work should remain on exact top-8 execution and matched
QX-versus-llama attribution; no canonical QX weights or routing policy change
is justified by this result.

Reproduce with:

```sh
python3 tools/analyze_clustered_expert_basis.py \
  external/models/granite-3.0-1b-a400m-instruct \
  --output benchmarks/results/expert-clustered-basis-analysis-2026-08-23.json
```
