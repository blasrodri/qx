# Qwen layer-0 Q3 promotion and packed validation

Date: 2026-09-02

## Outcome

Qwen3-30B-A3B layer 0 is representation-sensitive but does not require an
exact causal anchor. Its 119 evidenced full-width experts were retrained with
Q3 gate, up, and down projections while the same nine under-evidenced experts
remained on their original GGUF payloads.

Q3 reduced weighted expert-local held-out relative L2 from 0.3450 to 0.2260 and
raised mean cosine from 0.9393 to 0.9721. Full-model causal replay then passed
both the selection and frozen validation slices:

| Corpus slice | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|
| selection prompts 1-4 | 125 | +0.5318% | 98.40% |
| frozen prompts 5-12 | 218 | -0.4493% | 96.33% |
| **aggregate** | **343** | **-0.0929%** | **97.08%** |

All comparisons used context 128, four CPU threads, 32 GPU-resident layers,
at most 96 tokens per reset sequence, and an exact control from the same
loaded-model session. The unanchored Q3 policy is promoted for layer 0 subject
to the later composed-model and task-quality gates.

Relative to its rejected Q2S4 layer policy, promoting all 119 students to the
canonical packed Q3 representation costs 66.94 MiB. No extra exact-expert
premium is required.

## Packed-Q3 validator

The first full Q3 selection replay kept all dequantized sidecars resident as
F32 matrices and failed with a Metal out-of-memory error during the student
pass. At full width, 119 diagnostic students require 2,142 MiB of resident F32
weights in addition to the 17 GiB GGUF teacher and runtime allocations.

`qx-route-cache multi-joint-expert-student-nll` now accepts
`--compact-q3-students`. The mode:

1. verifies every sidecar value against the exact symmetric Q3 group-32
   codebook;
2. retains 12 packed code bytes and one F32 validation scale per group;
3. expands only a routed expert into the existing reusable 18 MiB F32 scratch;
4. identifies the storage mode in the JSON report as
   `packed-q3-with-routed-scratch-expansion`.

The validation representation occupies 2.25 MiB per full-width expert, or
267.75 MiB for layer 0's 119 students. It therefore removes 1,874.25 MiB
(1.83 GiB) of retained student memory. Product Q3 uses FP16 scales and projects
to 234.28 MiB for the same students; the validation path deliberately retains
F32 scales to reproduce the diagnostic sidecars exactly.

A seven-token, all-119-student equivalence replay matched the resident-F32
path exactly for exact NLL, student NLL, perplexity delta, and top-1 decisions.
The subsequent 343-token selection/frozen replay completed without OOM.

The mode is validation machinery, not a native inference performance result.
Product inference continues to require the existing consumer-native Q3 Metal
path and manifest-driven mixed representation packing.

## Evidence

- Q3 manifest:
  `external/results/qwen3-full-q2s4-build/manifests-q3/layer0.json`
- resident-F32 equivalence smoke:
  `external/results/qwen3-full-q2s4-build/causal-layer00-q3-f32-equivalence-smoke.json`
- packed-Q3 equivalence smoke:
  `external/results/qwen3-full-q2s4-build/causal-layer00-q3-packed-equivalence-smoke.json`
- packed selection slice:
  `external/results/qwen3-full-q2s4-build/causal-layer00-q3-general4-packed-gpu32.json`
- packed untouched slice:
  `external/results/qwen3-full-q2s4-build/causal-layer00-q3-untouched8-packed-gpu32.json`

The large evidence and sidecar files remain outside Git. Their model identity is
`0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Next constraint

Layers 0 and 1 now have isolated mixed-bit policies that clear the aggregate
quality gate. Layer composition still needs one validator session containing
both packed Q3 and packed Q2S4 students. The current command-level flags select
one compact codec for the complete input list; the next machinery change is a
codec-tagged student input that can retain both formats simultaneously without
resident-F32 expansion.
