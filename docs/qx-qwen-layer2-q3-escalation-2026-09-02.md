# Qwen layer-2 Q3 escalation

Date: 2026-09-02
Status: isolated gate not met; later promoted by composition-aware selection

## Outcome

Qwen3-30B-A3B layer 2 was retrained at full width with Q3 gate, up, and down
projections using the same four training captures and two held-out captures as
the promoted early-layer policies. The evidence policy transformed 108 of 128
experts, retained 20 under-evidenced experts on their source GGUF payloads, and
covered 98.93% of held-out routes.

Q3 materially repaired the expert-local representation: weighted held-out
relative L2 fell from 0.4439 under Q2S4 to 0.3271, while mean cosine rose from
0.8966 to 0.9447. Full-model causal replay nevertheless remained above the
strict 1% perplexity gate:

| Layer-2 policy | Q3 students | Exact evidenced students | PPL delta | Top-1 |
|---|---:|---:|---:|---:|
| uniform evidenced Q3 | 108 | 0 | +4.4680% | 93.60% |
| Q3 + exact `2:92` | 107 | 1 | **+3.0801%** | **98.40%** |
| Q3 + exact `2:{4,7,44,92,117}` | 103 | 5 | +3.3039% | 98.40% |

Every row scored the same 125 tokens from the first four general held-out
prompts at context 128, GPU32, and four CPU threads. All exact controls were
bit-identical at mean NLL `3.843945774087131` despite run-time variation.

Exact `2:92` remains a strong causal anchor after the codec change, recovering
4.8 percentage points of top-1 agreement and 1.39 percentage points of
relative perplexity. The four additional experts came from the previous Q2S4
independent causal ranking. Together they did not change a top-1 decision and
worsened NLL relative to retaining only `2:92`. Causal rankings therefore do
not transfer compositionally across representations.

## Capacity consequence

A full-width product Q3 expert occupies 2,064,384 bytes versus 1,474,560 bytes
for Q2S4. Moving the 107 non-anchor students in the selected Pareto policy from
Q2S4 to Q3 costs 60.19 MiB. Retaining `2:92` at its 3,059,712-byte source
payload costs 0.95 MiB over Q3, or the already documented 1.51 MiB over Q2S4.

This byte cost is acceptable for an isolated sensitive layer, but the policy
is not promoted because capacity cannot substitute for the causal quality
gate.

## Evidence

- Q3 manifest:
  `external/results/qwen3-full-q2s4-build/manifests-q3/layer2.json`
- uniform Q3 selection:
  `external/results/qwen3-full-q2s4-build/causal-layer02-q3-general4-packed-gpu32.json`
- exact `2:92` selection:
  `external/results/qwen3-full-q2s4-build/causal-layer02-q3-anchor92-general4-packed-gpu32.json`
- five-exact panel:
  `external/results/qwen3-full-q2s4-build/causal-layer02-q3-exact-4-7-44-92-117-general4-packed-gpu32.json`

The large evidence and sidecar files remain outside Git. Their model identity
is `0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Decision

Do not add more exact experts from the stale Q2S4 ranking and do not include
layer 2 in the composed promotion policy yet. Validate layer 3 Q3 independently.
For layer 2, the next bounded escalation must use downstream/logits-aware
attribution under the Q3 baseline or a higher-precision layer policy; local
reconstruction error and inherited codec rankings are insufficient.

## Follow-up

The required composition-aware experiment was completed on 2026-09-03.
Independent exact fallbacks inside the promoted layers-0/1/3 composition found
`2:76`, which was not selected by the isolated Q3 policy, as the missing causal
anchor. Layer 2 is promoted only as part of the four-layer policy documented in
`qx-qwen-layers0-1-2-3-q3-composition-2026-09-03.md`; this document's isolated
rejection remains valid.
