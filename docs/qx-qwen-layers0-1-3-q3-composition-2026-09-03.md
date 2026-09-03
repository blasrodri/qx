# Qwen layers 0, 1, and 3 Q3 composition

Date: 2026-09-03
Status: strict composition gate not met; bounded anchor test pending

## Outcome

The individually promoted Q3 policies for Qwen3-30B-A3B layers 0, 1, and 3
were composed while unresolved layer 2 remained exact. The policy contains:

- layer 0: 119 Q3 students and nine evidence-retained source experts;
- layer 1: 118 Q3 students, nine evidence-retained source experts, and exact
  causal anchor `1:68`;
- layer 3: 112 Q3 students and 16 evidence-retained source experts.

All 349 packed Q3 students executed in one GPU20 validator session without
out-of-memory failure. The selection slice passed, but the untouched slice and
343-token aggregate exceeded the strict 1% perplexity gate:

| Corpus slice | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|
| selection prompts 1-4 | 125 | +0.5613% | 98.40% |
| frozen prompts 5-12 | 218 | +1.6419% | 95.41% |
| **aggregate** | **343** | **+1.2468%** | **96.50%** |

The paired aggregate mean-NLL delta is `0.012390558403607326`. Both partitions
used context 128, GPU20 residency, four CPU threads, and an exact control from
the same loaded-model session.

This is a narrow quality miss, not a structural collapse. It nevertheless
means individually passing layer policies cannot be promoted compositionally
without another causal gate. The frozen regression is larger than the simple
sum suggested by the isolated results, demonstrating a real downstream
interaction among compressed layers.

## Memory result

The validation representation retains 2.25 MiB per full-width Q3 student, or
785.25 MiB for 349 students, plus one shared 18 MiB route-local F32 expansion
scratch. GPU20 leaves sufficient unified-memory headroom for that payload and
the approximately 17 GiB teacher GGUF.

At product FP16-scale Q3 density, the 349 transformed experts occupy about
687.09 MiB. Moving layer 3 from Q2S4 to Q3 adds 63.00 MiB; together with the
previously measured layers-0/1 premium, the three-layer Q3 precision premium
is approximately 197.82 MiB. This remains compatible with the 7-8 GiB
installed-weight target, but the policy is rejected until quality clears.

## Evidence

- selection slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-03-q3-anchor68-general4-packed-gpu20.json`
- frozen slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-03-q3-anchor68-untouched8-packed-gpu20.json`

The large evidence and sidecar files remain outside Git. Their model identity
is `0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Next bounded experiment

Do not reopen broad expert scans. Test one composition-specific exact anchor:
`3:82`, the dominant causal exception under the prior layer-3 Q2S4 policy.
Its source payload costs approximately 0.95 MiB over Q3. Run both GPU20 corpus
partitions with the otherwise unchanged 348-student policy. Promote only if
the same-residency aggregate clears 1%; otherwise keep the proven layers-0/1
frontier and treat layer 3 as individually valid but composition-incompatible
until logits-aware selection is available.
