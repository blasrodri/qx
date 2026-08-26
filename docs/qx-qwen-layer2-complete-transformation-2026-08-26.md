# Qwen layer-2 complete transformation gate

Date: 2026-08-26  
Status: iteration 1–3 evidence; independent expert replacement rejected

## Outcome

The nine-expert panel did not generalize to near-complete replacement of one
MoE layer. Seventy-nine evidenced width-512 students covered 97.37% of held-out
layer-2 routes, but simultaneous replacement increased perplexity by 90.85%
and retained only 56.76% top-1 agreement.

Changing the distillation target from recomputed BF16 expert output to the
captured Q4 control output did not fix the collapse. Increasing the student to
width 640 and every projection to Q3 reduced the failure substantially, but
still produced +17.48% perplexity and 87.23% top-1 on the kill-test slice. That
uniform policy also misses the intended mobile capacity tier.

| Policy | Route coverage | PPL delta | Top-1 |
|---|---:|---:|---:|
| width 512, Q2S4/Q3/Q2S4, BF16 target | 97.37% | +90.85% | 56.76% |
| width 512, Q2S4/Q3/Q2S4, captured-Q4 target | 97.37% | +92.30% | 57.63% |
| width 640, all Q3, captured-Q4 target | 97.37% | +17.48% | 87.23% |

These slices differ in length because the later policies were kill tests after
the first decisive failure. They are directional controls, not promotion
benchmarks.

## Heterogeneous frontier

Students were sorted by held-out expert-local relative L2 and enabled as
prefixes in one loaded-model session:

| Active students | PPL delta | Top-1 |
|---:|---:|---:|
| 7 | -5.23% | 97.87% |
| 13 | +9.40% | 85.11% |
| 24 | +11.48% | 85.11% |
| 28 | +12.50% | 85.11% |
| 43 | +9.52% | 82.98% |

The cliff between seven and thirteen students proves that expert-local error
is not a sufficient allocation oracle. The first seven are below the replay's
resolution; adding six apparently good local students changes model behavior
materially. A useful heterogeneous policy therefore needs model-level marginal
selection or coordinated correction.

## Machinery

- `capture-corpus` can cap retained samples per expert while continuing to
  observe rare routes.
- `train_qwen_expert_layer.py` reads each trace and safetensors index once,
  transforms an entire evidenced layer in one MPS process, and records explicit
  teacher-width exceptions.
- Distillation target semantics are explicit: BF16 recomputation or captured
  Q4 control output.
- Multi-student evaluation streams sidecars into the bridge to avoid a second
  complete F32 copy and can evaluate several active prefixes after one exact
  pass.

The layer-2 transformation itself took 2.7–5.5 minutes depending on policy.
Scalar F32 callback evaluation took much longer and encountered an avoidable
OOM when all 79 width-640 sidecars were loaded. Loading only the largest tested
43-student prefix avoided the fault. Packed native evaluation is now required
before broad search.

## Decision

Do not capture layers 24 and 44 under the rejected independent width-512
policy. Iteration 4 should capture the exact and transformed `ffn_moe_out` for
the same token states, train a small layer-level residual correction, and
inject that correction after routed expert aggregation. If a low-rank residual
cannot return the 79-student layer to the quality envelope, the structural
width-reduction program is rejected for the product model.
