# QX MoE scheduling perturbation — 2026-08-23

## Result

The same exact row-inline Q4 graph was measured on the `general-systems`
post-prefill fixture (`context=512`, 128 sequential positions, release
binary, 461 dispatches/token):

| Schedule | Wall ms/token | GPU ms/token | CB/token | Throughput |
| --- | ---: | ---: | ---: | ---: |
| sealed canonical, 1 CB | 4.699–4.717 | 4.236–4.242 | 1 | 212.0–212.8 tok/s |
| layer-sliced, 2 CB | 4.537–4.541 | 4.237–4.245 | 2 | 220.2–220.4 tok/s |
| CB before every MoE | 4.460–4.474 | 4.274–4.284 | 25 | 223.5–224.2 tok/s |
| CB after every MoE | 4.473 | 4.271 | 24 | 223.6 tok/s |
| encoder before every MoE | 4.795 | 4.311 | 1 / 25 encoders | 208.6 tok/s |
| encoder after every MoE | 4.785 | 4.322 | 1 / 24 encoders | 209.0 tok/s |
| encoder before + after | 4.867 | 4.319 | 1 / 48 encoders | 205.5 tok/s |

The command-buffer variants improve wall throughput, but not GPU time. The
encoder-only variants regress. Therefore the first scheduling win is a
GPU-ordered command-buffer submission effect, not a faster MoE kernel and not
an encoder-boundary optimization. It is useful, but insufficient to close the
llama gap by itself.

The schedule is still diagnostic-only. The artifact remains the one-CB
canonical plan until the candidate is represented in the compiler plan and
passes the full promotion/quality gate.

## Predecessor perturbation

With routes and MoE inputs frozen at position 384, the fixed-route MoE block
was run after controlled predecessor work:

| Predecessor | GPU us/token | Dispatches |
| --- | ---: | ---: |
| none | 2,244 | 96 |
| 24 no-op dispatches | 2,256 | 120 |
| 24 synthetic compute dispatches | 2,866 | 120 |
| 24 selected-expert memory-touch dispatches | 4,132 | 120 |

The no-op control is neutral, while preceding memory traffic adds about 1.9
ms to the same MoE computation. This demonstrates that dynamic MoE latency is
strongly dependent on the preceding GPU memory/scheduling state. It does not
prove a cache-only cause: outstanding memory work, translation state,
resource pressure, and scheduler state remain possible contributors.

## llama live control

The normal llama Metal graph was profiled over a 128-token decode after a
384-token prompt. The steady tail was approximately:

| llama command-buffer range | GPU time |
| --- | ---: |
| main graph, including MoE nodes | 2.33–2.35 ms |
| tail graph | 0.25 ms |
| total normal GPU command-buffer time | ~2.58–2.60 ms/token |

This is a system control, not a per-MoE attribution: the main range contains
other graph work as well. It establishes that QX's remaining ~1.6 ms GPU gap
versus llama is real and cannot be explained by CPU submission alone.

## Consequence

The next performance branch is not another isolated GEMV layout. The compiler
needs to search the joint plan:

```text
expert representation × preceding workload state × GPU command boundary
```

The immediate production candidate is the 2-CB schedule, because it gives a
repeatable ~3.5% wall improvement with only two ordered command buffers. The
25-CB MoE-boundary schedule is a diagnostic lead, not yet a product plan: its
GPU time is slightly worse and it adds substantial submission complexity.
