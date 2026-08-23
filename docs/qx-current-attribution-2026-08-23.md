# QX current attribution — 2026-08-23

Artifact: `granite-m5-pro-q4-factorial-rowinline.qx`  
Artifact hash: `e3f2940482d5999d`  
Workload: `general-systems`, post-prefill position 384, context 512  
Hardware: Apple M5 Pro

## QX complete-token differential

Release `qx-bench --v2-differential-profile`, five measured iterations after
two warmups. These are marginal complete-graph savings with frozen routes;
they are not additive operator costs.

| Ablation | Baseline/ablated GPU µs | Marginal saved µs |
| --- | ---: | ---: |
| Baseline | 4576.79 | 0.00 |
| Input RMS/QKV | 4278.46 | 298.33 |
| RoPE/KV | 4691.29 | -114.50 |
| Attention | 4296.88 | 279.92 |
| O/residual | 4512.25 | 64.54 |
| Router/top-k | 4400.83 | 175.96 |
| MoE | 2324.00 | 2252.79 |
| Final RMSNorm | 4450.87 | 125.92 |
| LM head | 4102.17 | 474.62 |
| Argmax | 4354.21 | 222.58 |
| Final head combined | 4003.67 | 573.12 |

The negative RoPE/KV delta is a scheduling/non-additivity signal, not a
claim that removing RoPE is slower. The largest actionable non-MoE boundary
is the final head, with the LM projection larger than final RMSNorm or
argmax in the split measurement.

## Final-head geometry experiment

The existing Q4 sweep kernel was exposed as an experimental final-vocabulary
projection candidate using `QX_V2_FINAL_HEAD_SWEEP=rows:simd_groups:cache`.
This changes only execution geometry and keeps the Q4 representation and
arithmetic unchanged.

The same-session control measured 213.76 tok/s and 4.225 ms GPU/token. The
tested geometries were 205–211 tok/s, including `2:8:0` at 210.09 tok/s.
No candidate met the 0.10 ms/token promotion threshold; the branch is
rejected and remains non-canonical.

## Exact fused final-head candidate

Candidate A, enabled with `QX_V2_FINAL_ARGMAX_FUSED=1`, computes the Q4
vocabulary projection and emits only per-threadgroup maxima before a
hierarchical reduction. The reduction was extended for the 6,144 partial
maxima produced by the 49k vocabulary; the original 256-entry reducer cannot
consume that output directly.

The 128-token generated stream was identical to the control. The candidate
measured 212.03 tok/s and 4.239 ms GPU/token, versus the same-session control
at 213.76 tok/s and 4.225 ms GPU/token. It is therefore rejected for
performance. Because it intentionally does not materialize logits, full-logit
parity and quality tools must continue to run with the default path; this
candidate is currently generation-only.

## Matched llama reference replay

Using the instrumented llama Metal replay at `-p 384 -n 128 -b 1 -ub 1
-t 5 -ngl -1 -sm none -fa on`, the stable medians were:

| Component | llama GPU µs/layer |
| --- | ---: |
| QKV | 15.1 |
| RoPE/KV | 13.0 |
| Attention | 10.0 |
| O projection | 5.5 |
| Complete transformer block replay | 112.6 |

The grouped MoE control remains approximately 1.455 ms for llama versus
approximately 2.245 ms for QX fixed-route MoE. Therefore MoE is a real
deficit, but it accounts for only about half of the approximately 1.65 ms
QX-versus-llama full-GPU gap measured under the current comparison contract.

## Decision

Do not promote the final-head geometry candidate or infer a new throughput
milestone from isolated timings. The next performance experiment must be a
complete-token candidate selected against the corrected profile, with MoE
and the largest non-MoE boundary evaluated as separate lanes.
