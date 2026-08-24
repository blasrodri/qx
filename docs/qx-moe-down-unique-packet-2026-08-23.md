# MoE down unique-packet follow-up

## Scope

This is an exact-Q4 candidate. It keeps the promoted row-inline artifact,
top-8 routing, four-stage MoE graph, and 461-dispatch/one-command-buffer
schedule unchanged. It changes only the row-inline down consumer:

```text
gate/up: existing compiler-selected unique-packet consumer
down:   experimental cooperative unique-packet consumer
```

The down matrix has 16 group-32 packets per output row. Each SIMD group owns
two adjacent packets, loads each packed code byte once, loads each FP16 scale
once per half-group, broadcasts the scale, and performs the same exact Q4 dot
product as the canonical row-inline kernel.

## Implementation

- Metal entry point: `qx_dynamic_q4_moe_down_inline_unique_packet_simd`
- Runtime switch: `QX_V2_MOE_UNIQUE_PACKET_DOWN=1`
- Sealed artifact-plan field: `moe_down_consumer: "unique-packet"`
- Compiler option: `--moe-down-consumer canonical|unique-packet`

The down consumer is deliberately independent from `moe_consumer`; gate/up
and down have different shapes and may select different consumer dataflows.

## Initial evidence

Same general context-512 fixture, release binary, 128 decode iterations:

| Plan | Wall ms/token | GPU ms/token | tok/s | Artifact | Dispatches |
|---|---:|---:|---:|---:|---:|
| Gate/up unique, down canonical | 4.365 | 3.911 | 229.07 | 751,009,792 B | 461 |
| Gate/up unique, down unique (external flag) | 4.137 | 3.689 | 241.72 | 751,009,792 B | 461 |
| Sealed down-consumer artifact, one run | 4.450 | 4.001 | 224.71 | 751,009,792 B | 461 |

The one-run results are a candidate signal only; the sealed result is noisy
and does not yet establish promotion. The external-flag run passed the full
long-position parity gate:

```text
positions:       0, 1, 7, 31, 63, 127, 255, 511
max KV key:      0.009850
max KV value:    0.001048
max attention:   0.000127
max logits:      0.009905
min cosine:      1.000000
argmax:          8/8
```

## Promotion status

Not promoted. The next measurement is a randomized paired complete-model
comparison of the sealed canonical and sealed down-consumer plans across all
five context-512 fixtures. Promotion requires a repeatable complete-token
win and the same 0–511 parity gate; fixed-route or single-run speedups are
diagnostic only.

## 20-round paired result (2026-08-24)

The full five-fixture suite was run for 20 randomized pairs on battery power.
The machine stayed within the normal performance regime, but this remains a
power-source-qualified result rather than an AC reference benchmark.

```text
selected plan:        granite-m5-top8-rowinline-unique-packet
control median wall:  4.467985 ms/token
candidate median:     4.473212 ms/token
control median GPU:   3.998893 ms/token
candidate median GPU: 4.005573 ms/token
```

The candidate was exact-equivalent, but did not win consistently:

| Fixture | Pairs | Median candidate-control wall delta | Bootstrap 95% CI | Candidate win rate |
|---|---:|---:|---:|---:|
| General | 20 | -0.0018 ms | [-0.0679, 0.0238] | 50% |
| Instruction | 20 | +0.0107 ms | [-0.0153, 0.0598] | 40% |
| Code | 20 | +0.0070 ms | [-0.0142, 0.0586] | 35% |
| Reasoning | 20 | +0.0044 ms | [-0.0373, 0.0359] | 40% |
| Chat | 20 | -0.0066 ms | [-0.0407, 0.0182] | 65% |

Decision: **reject for promotion and retain the gate/up-only unique-packet
plan**. The down consumer is not materially harmful, but its complete-model
effect is statistically indistinguishable from noise and does not clear the
0.10 ms promotion threshold. Its exact implementation remains useful as a
negative compiler record.

The generated down-consumer artifact is intentionally not checked into Git;
the reproducible compiler command is:

```bash
target/release/qx-compile \
  external/models/granite-3.0-1b-a400m-instruct \
  --target apple-m5-pro --memory-budget 900mb --context 512 \
  --disable-q3 --uniform-expert-bits q4 \
  --expert-layout row-inline --weight-storage mmap-no-copy \
  --moe-consumer unique-packet \
  --moe-down-consumer unique-packet \
  --plan-id granite-m5-top8-rowinline-unique-packet-down \
  --output external/models/granite-m5-pro-q4-rowinline-unique-packet-down.qx
```
