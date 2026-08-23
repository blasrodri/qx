# Streaming attention candidate — 2026-08-23

Candidate: Granite-fixed qlen=1 GQA streaming attention with online softmax.
The candidate keeps both query heads associated with each KV head in one SIMD
group and removes score, probability, and V-partial buffers.

Controls:

```text
QX_V2_ATTENTION_STREAM=1
QX_V2_ATTENTION_STREAM_TG=32   # one SIMD group per workgroup
QX_V2_ATTENTION_STREAM_TG=256  # eight SIMD groups per workgroup
```

The candidate is FP32 K/V only and is disabled by default. Lower-precision KV
automatically falls back to the canonical path.

## Result at context 512

Fresh-process live-attention GPU medians, 50 iterations per process:

| Path | GPU µs |
| --- | ---: |
| canonical three-stage | ~67.8 |
| streaming, threadgroup 32 | ~363 |
| streaming, threadgroup 256 | ~224–274 |

The streaming design is decisively slower. The likely cause is the cost of
performing two SIMD dot reductions and multiple online-softmax broadcasts for
every context position, combined with four persistent accumulators per lane.
Removing intermediate buffers does not compensate for that instruction and
occupancy cost on this mapping.

The candidate is therefore rejected for promotion. It remains useful as a
compiler-search record: fusion and GQA reuse alone do not imply a faster
attention topology on this GPU.

## Chunk-parallel follow-up

The final requested attention architecture was also implemented as a
benchmark-gated candidate: eight 64-position chunks, one 256-thread
workgroup per KV head, online softmax statistics and V partials, followed by a
small merge dispatch. It matched the observed checkpoint errors, but repeated
context-512 live measurements were approximately 75.5 µs versus 67.8 µs for
canonical attention. It is therefore also rejected for promotion.

## Follow-up

Do not continue tuning this kernel or change canonical attention. The exact
attention lane is closed pending a materially different mapping, and the
larger confirmed MoE matrix deficit remains the higher-value production target.
