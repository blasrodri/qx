# Qwen layers 0-1 Q3 composition

Date: 2026-09-02

## Outcome

The promoted isolated policies for Qwen3-30B-A3B layers 0 and 1 compose
without quality accumulation:

- layer 0: 119 full-width Q3 students and nine evidence-retained source
  experts;
- layer 1: 118 full-width Q3 students, nine evidence-retained source experts,
  and causally retained source expert `1:68`.

The combined policy therefore activates 237 packed Q3 students while retaining
19 source experts. It passed both the selection and untouched corpus slices:

| Corpus slice | GPU layers | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|---:|
| selection prompts 1-4 | 32 | 125 | -0.5399% | 98.40% |
| frozen prompts 5-12 | 24 | 218 | -0.4393% | 95.41% |

Each result compares the student policy only with the exact teacher pass from
the same loaded-model session and GPU residency. No cross-residency aggregate
is claimed. Both slices independently pass the 1% perplexity gate, and both
slightly improve teacher-forced NLL relative to their exact controls.

The result falsifies the assumption that isolated compression errors simply
add across early layers. Under these Q3 policies, their full-model interaction
is mildly compensating on both corpus partitions.

## Validation memory boundary

The four-prompt screen completed at GPU32 using the packed-Q3 validator. The
first frozen attempt at GPU32 crossed the Metal memory boundary during the
student pass despite packed storage. Replaying at GPU24 completed normally.
This is a validation-host constraint: the session simultaneously holds the
17 GiB GGUF teacher and 237 diagnostic students. It is not the projected
iPhone runtime residency, where only the compiled artifact and bounded expert
cache are present.

The two-layer validator payload is approximately 533.25 MiB with F32 scales,
versus 4,266 MiB for resident-F32 sidecars. One reusable 18 MiB expansion
scratch is shared across routed experts.

Relative to the rejected Q2S4 policies for the same two layers, the projected
product payload premium is 134.82 MiB: 66.94 MiB for layer 0 and 67.89 MiB for
layer 1 including its additional exact anchor.

## Evidence

- selection slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-q3-anchor68-general4-packed-gpu32.json`
- frozen slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-q3-anchor68-untouched8-packed-gpu24.json`

The large evidence and sidecar files remain outside Git. Their model identity is
`0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Next constraint

Codec-tagged validator inputs are now implemented and have executed Q3 and
Q2S4 students in one live session. The structural smoke rejected the old
layer-3 Q2S4 candidate before a long layers 0-3 replay: adding it to layer 0
Q3 raised the seven-token probe from +5.40% to +14.13% perplexity. The next
step is the same bounded Q3 escalation for layers 2 and 3 that repaired layers
0 and 1, followed by isolated and composed causal gates.
