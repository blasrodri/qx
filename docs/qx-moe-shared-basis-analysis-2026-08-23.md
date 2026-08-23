# Cross-expert shared-basis analysis — 2026-08-23

This is an offline candidate-generation study against the original Granite
BF16 safetensors checkpoint. It does not change the canonical QX artifact and
does not establish model quality for any approximation.

For each layer and tensor family, the 32 expert matrices were flattened across
their output/input dimensions. The analysis computes the mean expert matrix,
then the eigen-spectrum of the centered expert covariance. The reported
fractions measure total Frobenius energy explained by:

```text
shared base + rank-r basis across the expert dimension
```

## Aggregate result

| Tensor | Shared base | Shared + rank 4 | Shared + rank 8 | Shared + rank 16 | Rank-4 residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up | 3.48% | 17.30% | 30.53% | 55.95% | 82.70% |
| down | 3.12% | 17.30% | 30.52% | 55.71% | 82.70% |

The result is consistent across layers. A single shared base explains very
little, and rank 4 leaves approximately 83% of total matrix energy
unexplained. Rank 16 still leaves approximately 44% unexplained.

## Decision

The simple global architecture:

```text
expert = shared_base + low_rank_expert_residual
```

is not a credible first runtime prototype at ranks 4–16. It would likely
introduce unacceptable approximation error unless the compiler adds one or
more of:

- per-layer expert clustering;
- activation-weighted rather than Frobenius-weighted fitting;
- higher residual ranks;
- block-sparse residual selection;
- quality-aware retraining/fine-tuning.

The result does not disprove a clustered QX-native MoE representation, but it
does reject the assumption that Granite’s 32 experts share a strong global
low-dimensional basis. Before runtime work, the next offline experiment should
measure clustered bases and activation-weighted reconstruction on routed
hidden-state traces. Exact top-8 QX remains unchanged.
