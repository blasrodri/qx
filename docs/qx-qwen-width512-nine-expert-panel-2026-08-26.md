# Qwen width-512 nine-expert composition panel

Date: 2026-08-26  
Status: structural composition smoke; not promotion evidence

## Outcome

The fixed width-512 mixed-bit policy generalized from one expert to a panel of
nine experts across early, middle, and late layers. Every layer contains a
high-, median-, and low-frequency expert selected from disjoint route counts.
All students use Q2S4 gate/down and Q3 up projections with codec-aware QAT.

The decisive untouched replay skipped the four WikiText-2 test paragraphs used
for local held-out reporting and evaluated the next four:

| Metric | Result |
|---|---:|
| Simultaneous students | 9 |
| Scored tokens | 143 |
| Relative perplexity delta | **-1.5716%** |
| Top-1 agreement | 93.01% |
| Minimum exact target routes | 6 |
| Maximum exact target routes | 117 |

The earlier four-paragraph composition replay produced -3.4167% relative
perplexity and 92.05% top-1 agreement across 88 tokens. The untouched result
shows that the favorable NLL direction was not confined to the development
slice. It does not establish that the students improve Qwen; the corpus is too
small and narrow, and top-1 behavior changes materially.

## Machinery added

`capture-corpus` now captures all top-k expert inputs, activations, and down
outputs for a selected layer in one teacher replay, then retains any requested
expert subset. This changes collection cost from one model replay per expert to
one replay per layer. Existing single-expert traces remain readable.

`multi-joint-expert-student-nll` loads several gate/up/down sidecars, injects
them simultaneously at their real llama.cpp MoE boundaries, and records exact
and student route hits. A sequence offset supports a final untouched corpus
slice. A one-student parity replay exactly matched the freshly rebuilt legacy
single-student path.

## Panel

| Layer | Tier | Expert | Train hits | Held-out hits | Local relative L2 | Cosine |
|---:|---|---:|---:|---:|---:|---:|
| 2 | high | 125 | 81 | 34 | 0.5750 | 0.8124 |
| 2 | median | 16 | 20 | 15 | 0.5046 | 0.8536 |
| 2 | low | 124 | 5 | 4 | 0.4298 | 0.8983 |
| 24 | high | 2 | 164 | 61 | 0.6341 | 0.7716 |
| 24 | median | 73 | 36 | 10 | 0.6164 | 0.7842 |
| 24 | low | 96 | 12 | 5 | 0.4652 | 0.8806 |
| 44 | high | 6 | 229 | 72 | 0.6196 | 0.7696 |
| 44 | median | 56 | 19 | 24 | 0.6786 | 0.7363 |
| 44 | low | 46 | 6 | 11 | 0.5408 | 0.8399 |

Expert-local vector error is a poor promotion oracle here: it looks much worse
than the end-to-end NLL effect after route weighting and residual composition.
QX should optimize and select candidates using model-level loss, with local
error retained as a diagnostic and outlier detector.

## Capacity interpretation

If the fixed policy generalized to all 6,144 experts, projected weights remain
7.183 GiB and weights plus a 4K FP16 KV allowance remain 7.558 GiB. These are
logical capacity figures, not measured iPhone peak memory.

## Next gates

1. Add genuinely multi-domain calibration and untouched evaluation corpora.
2. Expand simultaneous replacement to dozens of experts and more layer points.
3. Track task accuracy, generation agreement, and safety—not only corpus NLL.
4. Emit a packed partial width-512 artifact and run native Q2S4/Q3 consumers.
5. Distil the full model only after the broader panel freezes width/codec policy.

The panel supports continuing the width-512 program. It does not yet justify a
full-model quality claim or commercial release.
