# Qwen layer-2 heterogeneous width frontier

Date: 2026-08-26  
Status: iteration 6 rejected for iPhone capacity; not promotion evidence

## Outcome

The live model-level allocator started from 79 width-640/all-Q3 layer-2
students plus the rank-64 shared residual. Replacing two locally promising
students with width-512 mixed-bit variants moved the 47-token replay to the
edge of the loss gate. Four replacements failed it and reduced top-1
agreement.

| Width-512 replacements | PPL delta | Top-1 |
|---:|---:|---:|
| 0 | -1.92% | 89.36% |
| 2: experts 63, 18 | +1.98% | 89.36% |
| 4: plus 118, 113 | +2.16% | 87.23% |

The Accelerate callback is not bit-identical to the earlier scalar callback,
so all points above are interpreted only relative to the exact control in this
self-contained evaluator.

## Capacity consequence

A packed width-640/all-Q3 expert requires 1,720,320 bytes. A width-512
Q2S4/Q3/Q2S4 expert requires 1,114,112 bytes, saving 606,208 bytes. If the
observed two-swap boundary generalized to every layer, it would save only 55.5
MiB across 48 layers:

| Policy | Projected weights |
|---|---:|
| Width 640, all Q3 | 10.652 GiB |
| Two width-512 swaps per layer | 10.598 GiB |

This cannot produce the iPhone tier after KV, runtime buffers, and application
headroom. The width-640-to-512 heterogeneous frontier is therefore rejected,
even though the live allocator itself is the correct selection machinery.

## Evaluator findings

- Expert and residual F32 GEMVs now use Apple Accelerate instead of scalar C.
- A two-pass 47-token baseline completed in 123 seconds.
- Replacement sidecars can overwrite and free matching baseline matrices in
  place.
- Repeated Metal decode/reset passes in one process still accumulate enough
  pressure to OOM. Durable search currently requires a fresh process for each
  checkpoint; packed native evaluation remains necessary.

## Next structural target

The width failure came from deleting one-third of every expert's neurons. The
next candidate keeps the complete 768-neuron expert and applies QX Q2S4 to all
three projections. It projects to approximately 9.246 GiB of weights and 9.621
GiB with a 4K FP16 KV allowance. This is tighter than desired but potentially
fits a 12 GB iPhone while preserving the model's full expert structure.
