# Qwen layer-3 Q3 selection

Date: 2026-09-03
Status: selection gate passes; frozen GPU24 replay pending

## Outcome

Qwen3-30B-A3B layer 3 was retrained at full width with Q3 gate, up, and down
projections using the same four training captures and two held-out captures as
layers 0-2. The evidence policy transformed 112 of 128 experts, retained 16
under-evidenced experts on their source GGUF payloads, and covered 99.40% of
held-out routes.

Weighted expert-local held-out relative L2 improved from approximately 0.5331
under Q2S4 to 0.3487 under Q3; mean cosine improved from 0.8471 to 0.9408.

The unanchored Q3 policy passed its four-prompt causal selection slice:

| Policy | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|
| 112 Q3 students, 16 evidence exceptions | 125 | **-0.7052%** | **96.80%** |

The old Q2S4 policy required exact expert `3:82`. Q3 clears the selection gate
without it, so the source expert is not retained absent contrary frozen
evidence. This saves approximately 0.95 MiB relative to a Q3-plus-anchor policy.

## Frozen replay boundary

The untouched eight-prompt replay was first attempted at GPU32. Metal exhausted
its command-buffer allocation during the exact control pass before producing a
result. This is the same validator-host residency boundary previously observed
for the larger layers-0/1 composition; it is not a quality result and it does
not reflect product iPhone residency.

The frozen replay must be restarted at GPU24, scoring prompts 5-12 with no
policy reselection. Heavy compute was parked after the failed attempt because
the host was on battery. Layer 3 remains a selection candidate rather than a
promoted policy until that replay completes.

## Capacity consequence

The 112 product Q3 students project to 220.50 MiB, versus 157.50 MiB at Q2S4,
for a 63.00 MiB precision premium. The validator retains F32 scales and uses
252.00 MiB plus one shared 18 MiB route-local expansion scratch.

## Evidence

- Q3 manifest:
  `external/results/qwen3-full-q2s4-build/manifests-q3/layer3.json`
- selection replay:
  `external/results/qwen3-full-q2s4-build/causal-layer03-q3-general4-packed-gpu32.json`
- failed GPU32 frozen attempt: no JSON result was produced

The large evidence and sidecar files remain outside Git. Their model identity
is `0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Next step

Run the frozen prompts at GPU24. If the same unanchored policy remains within
the 1% perplexity gate, promote layer 3 and then compose promoted layers 0, 1,
and 3. Layer 2 remains excluded until its Q3-baseline causal policy clears the
strict gate.
