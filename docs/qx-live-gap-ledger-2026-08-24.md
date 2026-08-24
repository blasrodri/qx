# QX live performance gap ledger

## Contract

This refresh uses the promoted exact artifact
`granite-m5-top8-rowinline-unique-packet`, artifact hash
`e0891fd697f49774`, on the Apple M5 Pro. The QX fixture is context 512 with
384 prefill tokens and 128 measured decode tokens. QX differential runs use a
frozen canonical top-8 route trace and preserve dispatch shape where possible.

The llama control is the same-session `llama-bench` Q4_K_M Granite run,
commit `1719747`, 128 decode tokens, GPU offload, Flash Attention, and FP16
KV. Its whole-token result is a competitive control; its internal component
rows are not yet semantically matched and are intentionally left unfilled.

## Whole-token comparison

| Engine | Decode throughput | Time/token | GPU/graph notes |
|---|---:|---:|---|
| QX exact Q4 | 223.33 tok/s | 4.478 ms wall | 4.012 ms GPU, 461 dispatches, 1 CB |
| llama.cpp Q4_K_M | 326.45 tok/s | 3.063 ms equivalent | `llama-bench` decode timing |

QX remains approximately 31.6% slower in this matched workload. The QX
artifact is 751,009,792 bytes; known Metal allocation is 802,373,632 bytes;
measured QX process physical footprint is approximately 114.6 MB. This is not
yet a total unified-memory comparison against llama's resident footprint.

## QX live differential ledger

These are marginal complete-graph savings, not additive operator costs:

| Block | QX marginal GPU us/token |
|---|---:|
| Input RMS + QKV | 286.83 |
| RoPE/KV | negative / non-actionable in this ablation |
| Attention | 47.58 |
| O/residual | negative / non-actionable in this ablation |
| Router/top-k | negative / non-actionable in this ablation |
| MoE | 2,212.63 |
| Final RMSNorm | 100.42 |
| LM head | 288.38 |
| Argmax | 21.58 |
| Final head combined | 380.08 |

Within the live MoE profile:

| MoE stage | Marginal GPU us/token |
|---|---:|
| Gate/up | 1,428.79 |
| SiLU | negative / noise at this boundary |
| Down | 748.25 |
| Accumulation | 11.46 |

The numbers above do not sum to the whole-token gap and must not be converted
directly into predicted throughput. They are target-ranking evidence only.

## Attention context

The current benchmark-only attention decomposition at context 512 is:

| Stage | GPU median us/layer | Wall median us/layer |
|---|---:|---:|
| QK | 15.87 | 199.88 |
| Softmax | 16.50 | 208.12 |
| V accumulation | 47.87 | 264.62 |
| Combined | 67.75 | 239.62 |
| Live attention | 67.96 | 256.96 |

This is useful for local diagnosis, but it is not a llama component
comparison. The current evidence does not justify another exact attention
rewrite.

## Decision

The exact search is paused pending a genuinely matched internal llama ledger.
The next exact candidate must have a concrete mechanism with at least
approximately 0.15 ms/token complete-model headroom before implementation.
The existing gate/up unique-packet consumer remains canonical. Down packet,
LM-head packet, and C16/C32 Down clustering remain archived negative or
below-threshold records.

The next high-upside system lane is speculative decoding. It must be measured
as a separate product-throughput claim, with a multi-token verifier and a
like-for-like llama speculative baseline; it must not be used to claim exact
single-token kernel parity.
