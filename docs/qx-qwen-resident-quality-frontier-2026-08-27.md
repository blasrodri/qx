# Qwen resident quality frontier

Date: 2026-08-27
Status: layer-2 heterogeneous policy passes provisionally; whole model pending

## Result

The bounded-residency design still projects to 6.43–6.93 GiB resident memory
with a 4.5 GiB expert cache. Quality, not capacity, is now the blocker.

The live gate replaced 79 of 128 experts in Qwen3-30B-A3B layer 2 and scored
47 teacher-forced tokens. Uniform full-width Q2S4 produced acceptable aggregate
NLL but only 65.96% top-1 agreement. That is too much behavioral movement to
justify building the remaining 47 layers.

| Layer-2 representation | Student NLL | PPL change | Top-1 agreement | Decision |
|---|---:|---:|---:|---|
| Uniform Q2S4 | 3.6174 | -2.72% | 65.96% | reject quality |
| Q2S4 + selective Q3 up | 3.6524 | +0.75% | 61.70% | reject selector |
| Q2S4 + rank-64 shared residual | 10.2207 | +71,650% | 4.26% | catastrophic reject |
| Q2S4 + exact GGUF expert 92 | 3.6276 | -1.72% | **93.62%** | provisional pass |

The exact control NLL was 3.6450 in all three runs.

## What the two repairs proved

The selective format promoted the up projection of 29 experts chosen by
calibration route frequency multiplied by expert-local Q2 error. Those experts
covered 57.0% of layer-2 calibration routes. The extra precision cost would be
about 261 MiB across 48 layers, taking installed weights from 9.246 GiB to
about 9.500 GiB. It made live token agreement worse, from 65.96% to 61.70%.
Frequency and local reconstruction error therefore do not identify the experts
that matter to downstream model behavior.

The rank-64 shared residual was the cheapest coordinated repair: 0.5 MiB per
layer, or 24 MiB over the complete model. Its offline aggregate normalized MSE
fell from 0.686 to 0.028, but cosine similarity and per-token relative error
warned that the fit was unstable. Live injection confirmed the failure with a
catastrophic NLL and top-1 regression. Offline output MSE is not a promotion
oracle.

## Causal precision selection

QX added an independent replacement mode and measured every one of the 29 Q3
up-projection candidates against the same complete Q2S4 baseline. Nine improved
NLL independently; three improved both NLL and top-1 agreement. The best joint
WikiText prefix, experts 122, 76, and 40, raised top-1 agreement from 65.96% to
70.21%. Its projected whole-model storage premium is only 27 MiB.

That gain did not survive strongly enough on disjoint general prompts. Across
four prompts and 125 scored tokens, all-Q2S4 reached 77.60% top-1 with a 9.74%
perplexity penalty. The best causal Q3 prefix, experts 122 and 76, reached
78.40% top-1 with a 9.50% penalty. This is directional evidence for downstream
impact selection, but not a quality pass and not justification for more Q3
tuning.

## Exact fallback breakthrough

QX then scanned exact fallbacks by disabling one Q2S4 student at a time while
leaving every other replacement active. This executes the original GGUF expert
for that target. Of 66 routed candidates on the WikiText gate, expert 92 alone
raised top-1 agreement from 65.96% to 93.62% while retaining better NLL than
the exact control.

The result survived two broader 12-prompt corpora without reselection:

| Corpus | Tokens | Q2S4 top-1 | + exact expert 92 | Q2S4 PPL penalty | Fallback penalty |
|---|---:|---:|---:|---:|---:|
| General holdout | 343 | 81.05% | **96.50%** | +3.42% | +2.15% |
| Second general set | 348 | 81.32% | **94.83%** | +4.30% | -0.16% |
| Combined | 691 | 81.19% | **95.66%** | +3.86% | **+0.98%** |

This is a provisional layer-2 quality pass. It proves that the damaging Q2S4
error is highly concentrated and that local reconstruction error was the wrong
selection signal.

The evaluator uses the original Q4_K_M payload, not a generic uniform Q4
estimate. Layer-2 expert 92 occupies 2.918 MiB: Q4_K gate/up plus Q6_K down.
Replacing its 1.406 MiB Q2S4 payload costs only 1.512 MiB. If the eventual
whole-model policy averaged one original expert per layer, the installed
premium would be 63.28 MiB. Reserving that above the 4.5 GiB cache would move
the conservative resident range from 6.433–6.933 GiB to 6.495–6.995 GiB.

## Decision

Stop tuning this artifact with expert-local reconstruction objectives. Do not
expand it to all 48 layers. The result avoids a costly false-positive build:
small NLL movement on one teacher-forced slice did not mean behavior was
preserved.

The downstream-impact machinery has rejected Q3-up and validated sparse exact
fallback for layer 2. The next bounded experiment is a multi-layer frontier:

1. Transform representative early, middle, and late layers to Q2S4.
2. Independently rank original-GGUF exceptions within each transformed layer.
3. Replay the selected exceptions jointly, then feed their actual mixed
   Q4_K/Q6_K byte sizes into the residency simulator.
4. Require at least 90% top-1 agreement with no material NLL regression on a
   larger, disjoint multi-domain replay before building all 48 layers.

If a small precision exception set cannot pass that gate, the remaining viable
route is logits-aware global distillation/QAT. That is a model-training project,
not another runtime codec tweak.

## Product interpretation

QX now has both a credible sub-7 GiB residency mechanism and a quality-preserving
heterogeneous policy for one transformed layer. It has **not** yet demonstrated
a complete 48-layer Qwen3-30B-A3B mobile artifact or iPhone token rate. The
layer-2 result must be reproduced across the network and then measured on-device.
