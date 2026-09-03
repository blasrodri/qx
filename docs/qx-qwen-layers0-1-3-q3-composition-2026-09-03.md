# Qwen layers 0, 1, and 3 Q3 composition

Date: 2026-09-03
Status: promoted with exact causal anchors `1:68` and `3:82`

## Outcome

The composed Qwen3-30B-A3B policy for layers 0, 1, and 3 clears the strict
1% perplexity gate when expert `3:82` remains exact. Unresolved layer 2 also
remains exact. The promoted policy contains:

- layer 0: 119 Q3 students and nine evidence-retained source experts;
- layer 1: 118 Q3 students, nine evidence-retained source experts, and exact
  causal anchor `1:68`;
- layer 3: 111 Q3 students, 16 evidence-retained source experts, and exact
  causal anchor `3:82`.

All 348 packed Q3 students executed in one GPU20 validator session. Both
corpus partitions improved mean NLL relative to the exact control, and their
343-token aggregate passes comfortably:

| Corpus slice | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|
| selection prompts 1-4 | 125 | -1.0324% | 97.60% |
| frozen prompts 5-12 | 218 | -0.5587% | 93.58% |
| **aggregate** | **343** | **-0.7316%** | **95.04%** |

The paired aggregate exact mean NLL is `3.8784066855846877`, student mean NLL
is `3.8710638037796676`, and mean-NLL delta is
`-0.007342881805020029`. Both partitions used context 128, GPU20 residency,
four CPU threads, and an exact control from the same loaded-model session.

## Why the anchor is required

The otherwise identical 349-student composition, with `3:82` encoded as Q3,
failed narrowly:

| Corpus slice | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|
| selection prompts 1-4 | 125 | +0.5613% | 98.40% |
| frozen prompts 5-12 | 218 | +1.6419% | 95.41% |
| **aggregate** | **343** | **+1.2468%** | **96.50%** |

That unanchored aggregate had mean-NLL delta `0.012390558403607326`.
Restoring only `3:82` exact changes the aggregate by about -1.98 percentage
points of relative perplexity. This validates `3:82` as a composition-specific
causal exception: local reconstruction quality and isolated layer-3 validation
were not sufficient to expose its downstream interaction with compressed
layers 0 and 1.

## Memory result

The validation representation retains 2.25 MiB per full-width Q3 student, or
783.00 MiB for 348 students, plus one shared 18 MiB route-local F32 expansion
scratch. GPU20 leaves sufficient unified-memory headroom for that payload and
the approximately 17 GiB teacher GGUF.

At product FP16-scale Q3 density, the 348 transformed experts occupy about
685.13 MiB. Moving layer 3 from Q2S4 to this anchored Q3 policy adds about
63.95 MiB, including the approximately 0.95 MiB source-weight premium for
`3:82`. Together with the measured layers-0/1 premium, the three-layer Q3
precision premium is approximately 198.77 MiB. This remains compatible with
the 7-8 GiB installed-weight target.

## Evidence

- promoted selection slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-03-q3-anchors68-82-general4-packed-gpu20.json`
- promoted frozen slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-03-q3-anchors68-82-untouched8-packed-gpu20.json`
- rejected unanchored selection slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-03-q3-anchor68-general4-packed-gpu20.json`
- rejected unanchored frozen slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-03-q3-anchor68-untouched8-packed-gpu20.json`

The large evidence and sidecar files remain outside Git. Their model identity
is `0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Promotion decision

Promote the anchored layers-0/1/3 Q3 composition as the current compact Qwen
frontier. Preserve exact `1:68`, exact `3:82`, all evidence-retained source
experts, and exact layer 2. Do not substitute the independently valid
unanchored layer-3 policy into a composed build.

The next quality work should address layer 2 with composition-aware,
logits-aware selection rather than another reconstruction-only expert scan.
