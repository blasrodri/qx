# Qwen layer-1 mixed-bit promotion

Date: 2026-09-02

## Outcome

Uniform Q2S4 is not an acceptable representation for Qwen3-30B-A3B layer 1.
Across the first four general prompts, replacing all 119 evidenced experts with
their Q2S4 students increased perplexity by 25.56%. Restoring the causally
dominant expert `1:68` reduced that result to +0.78% on the selection slice, but
the frozen eight-prompt result was still +1.82% and the 343-token aggregate was
+1.44%.

Retraining the same 119 evidenced experts at full width with Q3 for gate, up,
and down reduced the weighted expert-local held-out relative L2 from 0.4357 to
0.3190 and raised mean cosine from 0.8968 to 0.9484. Q3 alone reduced the
four-prompt causal perplexity delta to +2.75%, confirming that layer 1 is
representation-sensitive rather than irreducibly uncompressible.

The promoted layer-1 policy is:

- 118 full-width Q3 expert students;
- the original nine under-evidenced experts retained from the GGUF teacher;
- the original GGUF expert `1:68` retained as a causally demonstrated anchor.

All quality comparisons used context 128, four CPU threads, 32 GPU-resident
layers, at most 96 tokens per reset sequence, and an exact pass from the same
loaded-model session. Q3 sidecars were resident F32 diagnostic tensors during
validation; this timing is not a product-runtime measurement.

| Corpus slice | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|
| selection prompts 1-4 | 125 | +0.7001% | 98.40% |
| frozen prompts 5-12 | 218 | +1.0225% | 95.41% |
| **aggregate** | **343** | **+0.9049%** | **96.50%** |

The aggregate clears the predeclared 1% perplexity gate. The frozen slice is
0.0225 percentage points above 1%, so promotion is intentionally described as
an aggregate pass, not a per-slice pass. Broader task and domain gates remain
mandatory before promoting the complete model.

## Capacity cost

With the canonical group-32 layouts, a full-width all-Q3 expert is projected at
2,064,384 bytes and a full-width Q2S4 expert at 1,474,560 bytes. Promoting 118
students from Q2S4 to Q3 and restoring one additional source expert therefore
costs approximately 67.9 MiB relative to the rejected layer-1 Q2S4 policy.
This is less than 1% of the 8 GiB device envelope and is a favorable exchange
for reducing the aggregate perplexity delta from +1.44% to +0.90%.

These are packed-payload projections. The `.qxejs` training sidecars store
dequantized F32 diagnostics and are not the release artifact.

## Evidence

- Q3 manifest:
  `external/results/qwen3-full-q2s4-build/manifests-q3/layer1.json`
- unanchored Q3 screen:
  `external/results/qwen3-full-q2s4-build/causal-layer01-q3-general4-gpu32.json`
- anchored selection slice:
  `external/results/qwen3-full-q2s4-build/causal-layer01-q3-anchor68-general4-gpu32.json`
- anchored untouched slice:
  `external/results/qwen3-full-q2s4-build/causal-layer01-q3-anchor68-untouched8-gpu32.json`

The large evidence and sidecar files remain outside Git. Their model identity is
`0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Compiler consequence

Layer 1 must not inherit a model-wide expert codec. Its representation is a
profile-guided selection result: Q3 by default for evidenced experts, exact for
evidence exceptions and the causal anchor. The future manifest-driven packer
and runtime must consume that policy directly and account for the exact packed
bytes rather than treating Q2S4 as a global model flag.

The next experiment applies the same bounded escalation to layer 0, which also
missed the isolated quality gate under Q2S4. Composition of layers 0-3 should
resume only after each member has a viable isolated policy and the validator can
hold mixed Q2S4/Q3 students without resident-F32 expansion.
