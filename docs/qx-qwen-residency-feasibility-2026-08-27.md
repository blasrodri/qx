# Qwen Q2S4 bounded residency feasibility

Date: 2026-08-27  
Status: conditional product pass; native iPhone validation pending

## Result

QX replayed exact Qwen routes against bounded full-expert caches using the
1,474,560-byte packed size of one full-width Q2S4 expert. Pure LRU and pure
static-hot both missed the target. A hybrid policy that pins calibrated hot
experts and uses the remaining capacity as LRU produced the first plausible
mobile decode result.

The generation trace contains 14 prompt tokens followed by 64 generated
tokens. Prompt routes warm the cache; the table scores generation only.

| Expert cache | Pinned | Byte hit | Mean fetch/token | p95 fetch/token | Mean synthetic I/O | p95 synthetic I/O |
|---:|---:|---:|---:|---:|---:|---:|
| 4.0 GiB | 25% | 94.68% | 28.72 MiB | 67.50 MiB | 19.92 ms | 46.80 ms |
| 4.5 GiB | 50% | **95.60%** | **23.75 MiB** | **61.88 MiB** | **16.50 ms** | **43.00 ms** |

Synthetic I/O assumes 1,500 MiB/s, 50 microseconds per contiguous run, and
four concurrent reads. These numbers exclude GPU compute and are not iPhone
throughput claims. They establish that storage is no longer obviously fatal
during warmed decode.

## Memory envelope

With a 4.5 GiB expert cache, 0.808 GiB of shared weights, 0.375 GiB of 4K FP16
KV, and 0.75–1.25 GiB for runtime and application buffers, projected resident
memory is 6.43–6.93 GiB. The complete 9.246 GiB artifact remains on storage;
it is never uploaded as one Metal allocation.

## What does not pass yet

- Full-request cold traffic is substantially worse than warmed decode.
- A 22-token code/math stress trace with WikiText-derived pinning averaged
  86.8 MiB/token. The trace is too small for promotion, but it proves that a
  narrow hot calibration is not general-purpose.
- Full Q3 experts averaged 112 MiB/token with the 4.5 GiB cache. Q3 cannot be
  the bulk resident representation; it must be selective.

## Disjoint general-purpose validation

QX then captured compact all-layer routes from 12 calibration and 12 held-out
prompts spanning explanations, business writing, Rust/Python/TypeScript, math,
science, history, planning, security, and data analysis. The texts and requests
were disjoint.

At 4.5 GiB with 50% pinned, the held-out 355-token trace reached 94.80% byte
hit rate, 28.09 MiB/token mean fetch, and 81.56 MiB/token p95. Synthetic I/O
was 19.44 ms/token mean and 56.22 ms/token p95. This narrowly misses a 95% hit
threshold but passes the mean traffic target on broader data.

## Next gate

Residency advances as a conditional product path. The next experiment keeps
Q2S4 as the bulk format and promotes only sensitive, frequently routed up
projections to Q3. The cache planner must account for their larger byte sizes;
bulk Q3 is already rejected by the replay.
