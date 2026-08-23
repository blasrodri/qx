# QX MoE control results — 2026-08-23

## Scope

These experiments use release binaries, Granite 1B, batch-1 decode, the
general.json fixture, context 512, 384-token prefill, and 128 measured
tokens. The canonical exact-Q4 artifact is
granite-m5-pro-q4-factorial-rowinline.qx.

## Results

| Plan | GPU ms/token | wall ms/token | tok/s | dispatches | parity |
|---|---:|---:|---:|---:|---|
| Row-inline canonical, wave 8 | 4.235 | 4.683 | 213.52 | 461 | pass |
| Row-inline expert wave 4 | 4.357 | 4.843 | 206.50 | 509 | pass |
| Row-inline expert wave 2 | 4.651 | 5.164 | 193.64 | 557 | pass |
| Row-inline expert wave 1 | 5.175 | 5.684 | 175.93 | 653 | pass |
| Q4_K current control | 6.173 | 6.473 | 154.48 | 461 | pass |
| Q4_K mature two-row gate/up mapping | 6.194 | 6.481 | 154.30 | 461 | pass |

The Q4_K mature control was validated separately for gate/up and down before
combining. The first combined implementation exposed a down-projection
dispatch bug; it was corrected and the final combined control passed the
eight-position gate. It is nevertheless performance-neutral and is rejected.

## Conclusions

1. Limiting the number of selected expert streams does not improve this
   row-inline execution path. The extra wave dispatches and repeated activation
   staging cost more than the predecessor-pressure reduction.
2. The current evidence does not support a CPU-cache diagnosis. QX's measured
   remaining gap is GPU-side resource/memory scheduling around routed experts.
3. Q4_K representation alone is not the explanation for llama.cpp's advantage.
   Under QX's execution architecture it is about 28% slower than exact Q4
   row-inline, and the llama-shaped Q4_K mapping does not recover that loss.
4. Keep both branches as planner evidence, but do not promote either:
   wave_width < 8 is rejected for this workload and
   Q4_K + mature_mapping is a control only.

The next performance work should therefore target a structural difference in
the routed-weight execution program, not another threadgroup-width sweep:
matched live llama MoE attribution, expert-major versus layer-major traversal,
and a route-aware schedule that avoids adding speculative traffic.

## Two-command-buffer predecessor attribution

The predecessor experiment was then corrected so it no longer conflates
predecessor work with MoE work. Each sample commits CB1 on the Metal queue,
starts CB2 immediately, encodes the exact frozen-route MoE in CB2, and waits
only after CB2. The general fixture was rerun with five warmups and twenty
release iterations:

| CB1 predecessor | CB1 GPU us | CB2 MoE GPU us | total GPU us | wall us |
|---|---:|---:|---:|---:|
| none/control | 0.17 | 2245.63 | 2245.67 | 2478.83 |
| 24 no-ops | 61.75 | 2244.17 | 2305.75 | 2443.00 |
| canonical pre-MoE workload | 1558.00 | 2244.87 | 3802.83 | 4122.96 |
| compute-heavy | 623.13 | 2244.00 | 2867.33 | 3069.83 |
| contiguous memory-heavy | 1882.83 | 2256.58 | 4139.46 | 4336.58 |
| selected-expert memory-heavy | 1885.71 | 2244.92 | 4130.17 | 4335.42 |

The CB2 interval remains approximately 2.244 ms in every case. Therefore the
earlier 2.244 → 4.132 ms result was additive predecessor-plus-MoE timing, not
evidence that a preceding workload poisons the MoE command. The small
contiguous-memory variation is not large enough to justify an architecture
change without a larger controlled sample.

This closes the predecessor/cache-poisoning branch. The next comparison must
be a matched live llama MoE measurement or a complete-program attribution;
more predecessor, wave-width, or speculative-residency experiments are not
currently justified.
