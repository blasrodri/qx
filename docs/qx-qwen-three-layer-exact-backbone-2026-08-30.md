# Qwen three-layer exact backbone

Date: 2026-08-30  
Status: representative-layer quality gate passes; complete 48-layer artifact pending

## Result

The mixed-bit consumer-native storage thesis survives a joint early/middle/late
layer test.

QX replaced 206 experts across layers 2, 24, and 44 with full-width Q2S4
gate/up/down students. On 343 teacher-forced tokens from 12 general prompts,
the uniform three-layer policy retained only 79.88% of exact next-token
decisions. Restoring one original GGUF expert, layer 2 expert 92, raised
agreement to **94.75%** (325/343) with a **3.28% perplexity penalty**.

The exact exception costs 1.512 MiB above its Q2S4 payload. The provisional
resident projection therefore moves only from 6.433–6.933 GiB to
**6.434–6.934 GiB**. This is a structural capacity result, not a smaller
distilled architecture: the Qwen3-30B-A3B topology and full expert widths are
unchanged.

| Joint policy | Tokens | PPL change | Top-1 agreement | Exact premium |
|---|---:|---:|---:|---:|
| 206 Q2S4 students | 343 | +5.03% | 79.88% | 0 |
| Q2S4 + exact `2:92` | 343 | **+3.28%** | **94.75%** | **1.512 MiB** |

The one exact anchor recovers 51 next-token decisions.

## Representative layers

Layers 24 and 44 were trained with the same full-width Q2S4 policy previously
used for layer 2. The held-out route coverage of transformed experts was 98.10%
for layer 24 and 98.64% for layer 44. Experts without enough evidence remained
on their original GGUF payload; they are not silently counted as successful Q2
transformations.

The isolated live results were substantially better than layer 2:

| Layer | Transformed experts | Tokens | Q2S4 PPL change | Q2S4 top-1 | Exact selection |
|---|---:|---:|---:|---:|---|
| 2 | 79 | 125 | +9.74% in the earlier isolated control | 77.60% | `2:92` is causal anchor |
| 24 | 63 | 125 | +1.14% | 99.20% | optional; `24:79` reaches 100% |
| 44 | 64 | 125 | +0.36% | 97.60% | none improves top-1 |

Layer 24's exhaustive independent scan tested 53 routed candidates. Five exact
fallbacks improved top-1 and four improved both top-1 and NLL. Layer 44 used a
bounded 12-candidate panel: seven improved NLL, but none changed a top-1
decision. This confirms that exact exceptions are layer-specific causal
interventions, not a uniform quota.

## Joint exact-backbone selection

The first joint replay used all 206 students and restored the best isolated
candidates cumulatively:

| Cumulative policy | PPL change | Top-1 agreement |
|---|---:|---:|
| all Q2S4 | +13.32% | 78.40% |
| + exact `2:92` | +3.42% | **96.00%** |
| + exact `24:79` | +3.52% | **96.00%** |
| + exact `44:98` | +2.55% | 95.20% |

The Pareto policy is therefore just `2:92`. Layer `24:79` adds bytes without a
joint decision gain. Layer `44:98` improves aggregate NLL but changes one more
top-1 decision away from exact. If batch-1 behavioral fidelity is the primary
gate, neither is justified.

The selected policy then ran without reselection on all 12 general prompts in
three independent four-prompt shards:

| Prompt shard | Tokens | PPL change | Top-1 agreement |
|---|---:|---:|---:|
| 1–4 | 125 | +3.42% | 96.00% |
| 5–8 | 99 | -0.51% | 95.96% |
| 9–12 | 119 | +6.41% | 92.44% |
| **Aggregate** | **343** | **+3.28%** | **94.75%** |

This passes the previously stated representative-layer gate of at least 90%
top-1 agreement without material NLL regression.

## Memory-safe validation machinery

The original causal evaluator retained every dequantized student as three F32
matrices. At width 768 and hidden size 2048, that is 18 MiB per expert and
3.621 GiB for 206 experts, beside a 17 GiB GGUF. The machine became unusable,
and a broad all-at-once run eventually failed with a Metal out-of-memory error.

The evaluator now has an explicit `--compact-q2s4-students` mode. It:

1. verifies that each sidecar lies exactly on the Q2S4 codebook;
2. repacks four 2-bit values per byte with one F32 scale per group of 32;
3. retains 1.688 MiB per expert instead of 18 MiB;
4. expands only a routed expert into one reusable 18 MiB scratch area before
   the existing BLAS calculation;
5. reports progress for every exact, baseline, replacement, and fallback pass.

For 206 students, retained validation payload falls from 3.621 GiB to
0.339 GiB, avoiding **3.282 GiB**. A layer-44 equivalence replay produced
identical full-precision JSON metrics: NLL 3.845015366067928 and top-1 0.976
on both paths.

This scratch-expansion path is deliberately a validation mechanism. Product
inference uses the native packed Q2S4 Metal consumer and does not expand every
routed matrix to F32.

## Capacity interpretation

Layer 2 expert 92 uses Q4_K gate/up and Q6_K down in the source GGUF:

- original payload: 3,059,712 bytes;
- Q2S4 payload: 1,474,560 bytes;
- retained-exact premium: 1,585,152 bytes (1.512 MiB).

Adding only this demonstrated anchor projects to 9.247 GiB installed weights
and 6.434–6.934 GiB resident under the existing 4.5 GiB expert-cache plan.
Even an illustrative density of one layer-2-sized anchor per three layers would
add only about 24.2 MiB. That extrapolation is not promotion evidence; the full
48-layer causal scan must determine the real density.

## Decision and next gate

Proceed to the complete 48-layer artifact, but do not blindly transform cold
experts. The current manifests transform only experts with sufficient capture
evidence, so a production build needs broader capture or a principled cold
expert policy.

The next bounded product iteration is:

1. train the remaining layers with enough route evidence to minimize automatic
   exact exceptions;
2. run causal exact-anchor selection layer-wise, using bounded candidate panels
   except where a layer fails the live gate;
3. pack one consumer-native mixed Q2S4/original-GGUF artifact;
4. validate full-model logits and task quality;
5. measure installed bytes, peak iPhone residency, prefill, and decode tok/s on
   iPhone 17 Pro Max.

This result authorizes that build. It does not yet claim a complete sub-7 GiB
Qwen3-30B-A3B artifact or an on-device token rate.

