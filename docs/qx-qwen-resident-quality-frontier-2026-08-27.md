# Qwen resident quality frontier

Date: 2026-08-27
Status: memory architecture plausible; current compressed artifact rejected

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

## Decision

Stop tuning this artifact with expert-local reconstruction objectives. Do not
expand it to all 48 layers. The result avoids a costly false-positive build:
small NLL movement on one teacher-forced slice did not mean behavior was
preserved.

The next bounded experiment should rank precision by *downstream logit impact*:

1. Replay exact routes and replace one expert or projection at a time.
2. Measure the resulting KL divergence and top-logit changes at the model
   output, not local activation MSE.
3. Keep bulk Q2S4, but page exact Q4 or Q3 payloads for the small set with the
   highest causal impact.
4. Require at least 90% top-1 agreement with no material NLL regression on a
   larger, disjoint multi-domain replay before expanding beyond a few layers.

If a small precision exception set cannot pass that gate, the remaining viable
route is logits-aware global distillation/QAT. That is a model-training project,
not another runtime codec tweak.

## Product interpretation

QX has demonstrated a credible storage/residency mechanism for fitting the
working set under 7 GiB. It has **not** yet demonstrated a quality-preserving
Qwen3-30B-A3B mobile artifact or iPhone token rate. The capacity result and the
model-quality result must remain separate until both pass on-device.
