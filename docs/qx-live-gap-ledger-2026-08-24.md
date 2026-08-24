# QX live performance gap ledger

## Contract

This refresh uses the promoted exact artifact
`granite-m5-top8-rowinline-unique-packet`, artifact hash
`e0891fd697f49774`, on the Apple M5 Pro. The QX fixture is context 512 with
384 prefill tokens and 128 measured decode tokens. QX differential runs use a
frozen canonical top-8 route trace and preserve dispatch shape where possible.

The llama control is the same-session `llama-bench` Q4_K_M Granite run,
commit `1719747`, 128 decode tokens, GPU offload, Flash Attention, and FP16
KV. A new environment-gated Metal ledger records real production command
buffer intervals separately from one-shot semantic replays. The first decode
sample is treated as warm-up; the production median below uses samples 2--10.

## Whole-token comparison

| Engine | Decode throughput | Time/token | GPU/graph notes |
|---|---:|---:|---|
| QX exact Q4 | 223.33 tok/s | 4.478 ms wall | 4.012 ms GPU, 461 dispatches, 1 CB |
| llama.cpp Q4_K_M | 326.45 tok/s | 3.063 ms wall benchmark | 2.519 ms production GPU median, 2 CB |

The measured production GPU gap is approximately **1.493 ms/token**
(`4.012 - 2.519`). This is the number the exact lane must explain; it is not
the sum of isolated node timings. The QX artifact is 751,009,792 bytes; known
Metal allocation is 802,373,632 bytes; measured QX process physical footprint
is approximately 114.6 MB. This is not yet a total unified-memory comparison
against llama's resident footprint.

The new llama ledger is recorded in
`benchmarks/results/llama-production-ledger-2026-08-24.json`.

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

## Llama production ledger

The following rows are from the same decode contract, but the component rows
are diagnostic GPU replays rather than slices of the production command
buffers. They are therefore explanatory evidence, not additive costs:

| Boundary | Llama diagnostic GPU time |
|---|---:|
| Gate/up | 0.642 ms/token |
| Down | 0.397 ms/token |
| Whole MoE | 1.489 ms/token |
| Input RMS + QKV | 15.90 us/layer |
| RoPE/KV | 9.93 us/layer |
| Attention | 10.43 us/layer |
| O/residual | 8.13 us/layer |
| Router/top-k | 16.97 us/layer |
| MoE boundary | 62.52 us/layer |
| MoE residual | 12.31 us/layer |
| Final head boundary | 0.169 ms |

The production-vs-replay residual remains explicit. For llama, the replay
rows are not expected to reconstruct the 2.519 ms production interval; replay
changes command-buffer composition and can overlap or duplicate semantic
work. The ledger now answers the first binary question—whether QX's 4.012 ms
production graph is still materially behind llama's live GPU execution—without
turning diagnostic rows into false precision.

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

The exact search is paused pending a genuinely matched internal llama ledger
for the remaining non-MoE boundaries and a production-vs-replay residual
analysis. We now have the first decisive production number: approximately
1.493 ms/token of GPU gap. No new exact candidate should be implemented unless
its mechanism is tied to at least approximately 0.15--0.20 ms/token of this
measured gap.
The next exact candidate must have a concrete mechanism with at least
approximately 0.15 ms/token complete-model headroom before implementation.
The existing gate/up unique-packet consumer remains canonical. Down packet,
LM-head packet, and C16/C32 Down clustering remain archived negative or
below-threshold records.

The next high-upside system lane is speculative decoding. It must be measured
as a separate product-throughput claim, with a multi-token verifier and a
like-for-like llama speculative baseline; it must not be used to claim exact
single-token kernel parity.
