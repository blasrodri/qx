# Qwen codec-tagged packed causal validation

Date: 2026-09-02

## Outcome

`qx-route-cache multi-joint-expert-student-nll` can now retain Q2S4 and Q3
students in the same loaded-model session. Each input carries its storage
codec through `--q2s4-student` or `--q3-student`; the existing `--student`
plus `--compact-q2s4-students` and `--compact-q3-students` interfaces remain
available for uniform policies.

The route shim already carries a format tag per student. The command now
preserves that tag while loading each sidecar, verifies and packs it with the
matching codebook, and expands only the routed expert into the one shared F32
scratch allocation. A mixed report identifies itself as
`codec-tagged-packed-q2s4-q3-with-routed-scratch-expansion`.

The release build and all 25 route-cache tests pass. A live Qwen3-30B-A3B
replay loaded 119 layer-0 Q3 students and 111 layer-3 Q2S4 students together
and completed exact and student passes at GPU24. A tagged-Q3-only control at
the same residency produced the identical exact mean NLL (`7.074597230979374`),
confirming that the new input classification does not alter the teacher
control.

## Policy result

The mixed mechanism passes, but the old layer-3 Q2S4 policy does not deserve a
four-layer promotion run. On the seven-token structural smoke:

| Policy | Exact mean NLL | Student mean NLL | PPL delta | Top-1 |
|---|---:|---:|---:|---:|
| layer 0 Q3 | 7.074597 | 7.127143 | +5.3951% | 85.71% |
| layer 0 Q3 + layer 3 Q2S4 (`3:82` exact) | 7.074597 | 7.206759 | +14.1293% | 100.00% |

Seven tokens are intentionally insufficient for a promotion claim, but the
large regression is sufficient to reject spending a long composed replay on
this candidate. Earlier full slices already placed layer 3 Q2S4 at +1.36%
aggregate perplexity, above the current 1% gate. Layers 2 and 3 should receive
the same Q3 bounded escalation that repaired layers 0 and 1 before composition
is retried.

## Evidence

- tagged Q3 isolation smoke:
  `external/results/qwen3-full-q2s4-build/causal-tagged-layer00-q3-equivalence-smoke-gpu24.json`
- mixed Q3/Q2S4 smoke:
  `external/results/qwen3-full-q2s4-build/causal-mixed-layer00-q3-layer03-q2s4-smoke-gpu24.json`

The large evidence and sidecar files remain outside Git. Their model identity
is `0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Next constraint

Retrain the evidenced experts in layers 2 and 3 with Q3 gate/up/down
projections, run isolated packed causal selection, select exact anchors only
when the causal replay requires them, and then rerun the layers 0-3 composed
gate using codec-tagged inputs if any Q2S4 students remain.
