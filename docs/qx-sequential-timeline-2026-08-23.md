# QX sequential MoE timeline diagnostic

This diagnostic compares the exact canonical row-inline path with the
`QX_V2_MOE_DECODE_DIRECT=1` path after a 384-token prefill on the 512-position
`general-systems` fixture.

The release, no-instrumentation decode controls were:

| Plan | Wall ms/token | GPU ms/token | tok/s | CB/token | dispatch/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | 4.702 | 4.239 | 212.68 | 1 | 461 |
| direct | 4.662 | 4.219 | 214.52 | 1 | 461 |

This remains a statistically neutral single-pair result; the direct plan is
not promoted from it.

## Timeline result

Metal on the target M5 Pro reports dispatch-boundary counter sampling as
unsupported. The tool therefore falls back to ordered GPU command-buffer
boundaries, with no CPU waits and GPU-resident buffers. This is a localization
probe, not a production schedule: the fallback creates many command buffers
and exposes queue gaps that are not attributable to one logical stage.

Repeated fallback samples (8 measured positions) were:

| Stage | Canonical us/token | Direct us/token |
| --- | ---: | ---: |
| input RMS/QKV | 19.54 | 19.58 |
| RoPE/KV | 3.92 | 3.88 |
| attention | 21.08 | 21.50 |
| O/residual | 10.75 | 10.75 |
| router/top-k | 12.42 | 12.50 |
| MoE + residual | 95.25 | 94.75 |
| final head | 381.75 | 381.71 |
| all command-buffer GPU time | 4364.25 | 4365.25 |

The direct candidate’s fixed-route advantage is not present in the sequential
MoE interval. The normal one-command-buffer controls also remain close:
canonical 212.68 tok/s versus direct 214.52 tok/s in this run. The result
rules out a simple downstream-only explanation, but does not yet distinguish
cache/resource state from scheduler/queue effects because the target cannot
provide in-command-buffer dispatch timestamps.

The next measurement should therefore be a smaller boundary experiment around
the MoE entry/exit (rather than another kernel redesign), followed by a matched
sequential llama measurement. Do not use the fallback stage values as
production latency or add them to predict throughput.

## Live llama command-buffer control

The existing llama replay profiler was not used for attribution. A separate
`GGML_METAL_PROFILE_LIVE=1` hook was added locally to report the normal graph's
actual command buffers after completion. With `-p 384 -n 1 -b 1 -ub 1 -ngl -1`,
the steady decode tail had a median of approximately:

| llama command-buffer range | GPU us | MoE nodes | gate/up | down |
| --- | ---: | ---: | ---: | ---: |
| nodes 147..1471 | 2,370 | 594 | 44 | 22 |
| nodes 0..147 | 257 | 54 | 4 | 2 |
| total normal graph | **~2,627** | 648 | 48 | 24 |

The benchmark reported 311.09 tok/s for the single measured decode token in
that run. This is a live command-buffer control, not a per-kernel MoE timing:
the large command buffer contains other transformer work as well. It does,
however, establish a real scheduling difference versus QX's ~4.22 ms GPU/token
on the same context-512 class of workload. The next useful experiment is to
split the llama graph only at its existing node-range boundaries or obtain a
hardware trace, rather than compare replayed node sums to QX stages.
