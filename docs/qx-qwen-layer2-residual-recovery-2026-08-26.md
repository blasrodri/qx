# Qwen layer-2 coordinated residual recovery

Date: 2026-08-26  
Status: iteration 4 conditional success; iteration 5 capacity tier rejected;
not promotion evidence

Independent replacement of 79 layer-2 experts with width-640 all-Q3 students
increased perplexity by 17.48%. A shared rank-64 linear residual applied after
the aggregated MoE output restored paired model loss on two subsequent slices:

| Slice | Tokens | PPL delta | Top-1 |
|---|---:|---:|---:|
| First untouched paragraph | 47 | -2.51% | 89.36% |
| Next two untouched paragraphs | 59 | +1.97% | 93.22% |
| Combined | 106 | **-0.039%** | **91.51%** |

The adapter contains 262,144 parameters and requires 512 KiB per layer in
FP16, or 24 MiB across all 48 layers. Before live injection, the adapter
reduced reconstructed held-out aggregate-output normalized MSE from 0.00305 to
0.0000869, approximately 35x.

This validates the missing specialization dimension: experts cannot be
compressed solely as independent functions. Their weighted sum needs a small
layer-coordinated recovery path. Aggregate NLL passes this small replay, while
top-1 agreement remains below the 95% behavior target.

The result does not rescue uniform width-640/all-Q3 as a product policy because
that representation is too large.

## Iteration 5: width-512 capacity tier

The same rank-64 residual was then fitted to the width-512
Q2S4-gate/Q3-up/Q2S4-down layer. Offline reconstructed normalized MSE fell from
0.5812 to 0.01986, but this did **not** transfer to the live model. On the first
47-token kill slice, mean NLL rose from 3.6450 to 10.3636, perplexity was
827.7x the exact control, and top-1 agreement fell to 4.26%.

This rejects the uniform 7.183 GiB width-512 policy. It also proves that the
offline aggregate proxy is not a sufficient promotion gate: only paired live
model NLL can promote a representation. The next product experiment is a
heterogeneous allocation: retain width-512 mixed-bit students only where live
marginal replacement is safe, assign width-640/Q3 or teacher-width exceptions
to hard experts, and keep the small shared residual. That optimizer must target
the whole-model byte budget directly rather than forcing every expert through
one width and codec.
