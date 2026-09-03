# Qwen layer-3 Q3 promotion

Date: 2026-09-03
Status: promoted subject to composed-model and task-quality gates

## Outcome

Qwen3-30B-A3B layer 3 was retrained at full width with Q3 gate, up, and down
projections using the same four training captures and two held-out captures as
layers 0-2. The evidence policy transformed 112 of 128 experts, retained 16
under-evidenced experts on their source GGUF payloads, and covered 99.40% of
held-out routes.

Weighted expert-local held-out relative L2 improved from approximately 0.5331
under Q2S4 to 0.3487 under Q3; mean cosine improved from 0.8471 to 0.9408.

The unanchored Q3 policy passed both causal corpus partitions at GPU24:

| Corpus slice | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|
| selection prompts 1-4 | 125 | -0.2803% | 98.40% |
| frozen prompts 5-12 | 218 | +1.0202% | 93.12% |
| **aggregate** | **343** | **+0.5443%** | **95.04%** |

The aggregate mean-NLL delta is `0.005427857276522209`; exponentiating that
paired delta gives the reported +0.5443% perplexity. Both partitions used the
same model, context 128, GPU24 residency, four CPU threads, and an exact
control from the same loaded-model session.

The old Q2S4 policy required exact expert `3:82`. Q3 clears the complete gate
without it, so the source expert is not retained. This saves approximately
0.95 MiB relative to a Q3-plus-anchor policy.

## Residency boundary

The untouched eight-prompt replay was first attempted at GPU32. Metal exhausted
its command-buffer allocation during the exact control pass before producing a
result. This is the same validator-host residency boundary previously observed
for the larger layers-0/1 composition; it is not a quality result and it does
not reflect product iPhone residency.

The frozen replay completed at GPU24 with no policy reselection. The selection
slice was then repeated at GPU24 rather than aggregating across residency
configurations. It reached -0.2803% perplexity and 98.40% top-1, independently
replicating the earlier GPU32 selection result of -0.7052% and 96.80%.

## Capacity consequence

The 112 product Q3 students project to 220.50 MiB, versus 157.50 MiB at Q2S4,
for a 63.00 MiB precision premium. The validator retains F32 scales and uses
252.00 MiB plus one shared 18 MiB route-local expansion scratch.

## Evidence

- Q3 manifest:
  `external/results/qwen3-full-q2s4-build/manifests-q3/layer3.json`
- selection replay:
  `external/results/qwen3-full-q2s4-build/causal-layer03-q3-general4-packed-gpu24.json`
- frozen replay:
  `external/results/qwen3-full-q2s4-build/causal-layer03-q3-untouched8-packed-gpu24.json`
- cross-configuration selection replication:
  `external/results/qwen3-full-q2s4-build/causal-layer03-q3-general4-packed-gpu32.json`
- failed GPU32 frozen attempt: no JSON result was produced

The large evidence and sidecar files remain outside Git. Their model identity
is `0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Decision and next step

Promote layer 3's unanchored Q3 policy subject to composed-model and task
quality. Compose promoted layers 0, 1, and 3 next. Layer 2 remains excluded
until its Q3-baseline causal policy clears the strict gate.
