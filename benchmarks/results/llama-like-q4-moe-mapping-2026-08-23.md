# Decode-specific two-row Q4 MoE mapping

Date: 2026-08-23  
Worktree: `qx-llama-audit`  
Scope: exact single-row Q4 row-inline gate/up only

## Candidate

`QX_V2_MOE_LLAMA_LIKE_MAPPING=1` selects a new Metal entry point with:

- two SIMD groups per 64-thread threadgroup;
- two output rows owned by each SIMD group;
- direct single-row row-inline code/scale addressing;
- the existing normalized-input variant;
- unchanged eight selected-expert output slots and output ordering.

The default path is unchanged. Packetized row-inline, Q4_K, Q3, split, and
expertwise paths do not select this candidate.

## Validation

Release build and `qx-metal` tests passed: 9/9. The eight-position V2 parity
gate passed:

```text
positions:       0, 1, 7, 31, 63, 127, 255, 511
max KV key:      0.009853
max KV value:    0.001048
max attention:   0.000127
max logits:      0.009898
min cosine:      1.000000
argmax:          8/8
```

## Complete-token smoke comparison

Same artifact, model, context 512, release binary, warmup 1, three measured
iterations, 16 generated tokens. The order-reversed pair was intentionally
run to expose scheduling/thermal bias:

| Order | Plan | Wall ms/token | GPU ms/token | tok/s |
|---|---|---:|---:|---:|
| candidate then control | candidate | 4.843 | 4.184 | 206.47 |
| candidate then control | control | 5.569 | 5.029 | 179.58 |
| control then candidate | control | 4.477 | 4.151 | 223.38 |
| control then candidate | candidate | 4.666 | 4.343 | 214.30 |

Both plans used 2 command buffers, 2 encoders, 461 dispatches/token, the
same 751,009,792-byte artifact, and 802,373,632 known Metal allocation bytes.
The paired smoke result is therefore inconclusive; the candidate is **not
promoted** and requires randomized paired repetitions before any performance
claim.

## Caveats

The candidate evaluates the two rows serially within each SIMD group. It is a
mapping experiment, not a fused gate/up or representation change. A proper
promotion gate still requires randomized complete-model repetitions and the
same long fixture/quality checks used by the canonical plan.
