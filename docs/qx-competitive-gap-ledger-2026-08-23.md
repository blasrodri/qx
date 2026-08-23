# QX competitive gap ledger — 2026-08-23

Workload: Granite 1B, batch 1, context 512, post-prefill position 384.  
QX artifact: `e3f2940482d5999d`.  
Hardware: Apple M5 Pro.

This ledger is a grouped-boundary diagnostic. QX uses the existing sequential
boundary replay fallback because dispatch counters are unavailable on this
device. llama uses the new grouped semantic-boundary replay. The two paths
have equivalent boundary definitions, but their command-buffer schedules are
diagnostic replays rather than headline production timings. Complete-token
differential measurements remain authoritative for promotion.

## Boundary comparison

Values are GPU microseconds per layer, except final head, which is once per
token. QX values are medians over 32 positions; llama values are the stable
median of the grouped replay samples.

| Boundary | QX | llama | QX − llama | Approx. token gap |
| --- | ---: | ---: | ---: | ---: |
| Input norm + QKV | 19.50 | 21.32 | -1.82 | -0.044 ms |
| RoPE/KV | 4.00 | 10.08 | -6.08 | -0.146 ms |
| Attention | 21.92 | 10.68 | **11.24** | **+0.270 ms** |
| O + residual | 10.87 | 10.67 | 0.20 | +0.005 ms |
| Router/top-k | 12.21 | 18.08 | -5.87 | -0.141 ms |
| MoE + residual | 95.42 | 74.51* | **20.91** | **+0.502 ms** |
| Final head | 382.25 | 153.50 | **228.75** | **+0.229 ms** |
| **Ledger total** |  |  |  | **~+0.77 ms** |

\* llama MoE + residual is grouped MoE 62.09 µs plus grouped MoE residual
12.42 µs. The earlier complete-token differential measured a larger but
non-additive recoverable MoE gap of approximately 0.79 ms; both measurements
agree that MoE is a real deficit, but neither should be treated as an
additive prediction of throughput.

## Decisions

The ledger accounts for most of the previously unexplained ~0.85 ms without
inventing a CPU/cache theory:

1. Keep final-head structural candidates closed; both fused variants were
   exact but slower on the complete sequential fixture.
2. Keep MoE as the largest confirmed deficit, but do not invent another
   representation until its production dataflow is compared against llama.
3. Make attention the next non-MoE implementation lane: its grouped deficit
   is approximately 0.27 ms/token and its QX boundary is about 2x llama’s.
4. Preserve the current QX and llama controls while testing the next
   attention candidate; promotion still requires complete-token latency and
   the long sequential correctness gate.

The QX production reference remains approximately 4.23 ms GPU/token and the
llama reference approximately 2.59 ms GPU/token. The ledger is an attribution
tool, not a claim that replacing every listed boundary would subtract its
rows linearly.

## Attention substage and production-residual follow-up

The QX attention-stage benchmark measured 17.21 µs QK, 11.00 µs max/sum/softmax,
30.79 µs V accumulation, and 47.62 µs live attention at context 512. The
three-stage sum agrees with the live result. llama emits fused FlashAttention,
so its QK/softmax/V substages are unavailable; its matched grouped complete
attention boundary is 10.68 µs/layer. The detailed result is recorded in
[`qx-attention-gap-and-production-residual-2026-08-23.md`](qx-attention-gap-and-production-residual-2026-08-23.md).

The grouped replay totals are approximately 4.316 ms/token for QX and 3.642
ms/token for llama, versus production references of 4.225 and 2.590 ms/token.
Thus production minus grouped replay is -0.092 ms/token for QX and -1.052
ms/token for llama. This is replay overhead/overlap mismatch, not an
unattributed kernel cost. The production-vs-grouped residual must not be
converted directly into an optimization target.
