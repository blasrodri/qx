# GPU-resident greedy rollout

> Historical discovery report. The instrumented multi-context sweep in
> `qx-canonical-gpu-rollout-2026-08-24.md` supersedes the depth-4 selection:
> canonical M5 Pro greedy rollout is depth 2, and EOS now synchronizes at chunk
> boundaries.

## Result

QX can now enqueue multiple exact greedy-decode token graphs without waiting
for the CPU after every token. A four-token in-flight window improves the
conservative context-512 three-run mean from **349.35 to 388.91 tok/s**, an
**11.3%** throughput gain. In a higher-performance machine state, the paired
means were **367.51 versus 414.24 tok/s**, a **12.7%** gain.

The conservative result is approximately 22--25% faster than the local
310--320 tok/s llama.cpp control. The graph still executes the same 390
dispatches and two command buffers per token; the optimization removes idle
wall time between GPU submissions rather than changing model arithmetic.

| Context-512 mode | Mean tok/s | Mean wall ms/token | Mean GPU ms/token |
|---|---:|---:|---:|
| Synchronous, window 1 | 349.35 | 2.862 | 2.574 |
| GPU-resident, window 4 | **388.91** | **2.571** | 2.568 |

The higher-performance three-run window-4 result was tightly grouped at
413.59--414.63 tok/s. Both performance states are retained in the result JSON
rather than selecting only the faster sample.

## Mechanism

The fused LM-head argmax already leaves the predicted token in GPU memory and
the next embedding lookup consumes a GPU token buffer. The old runtime still
committed and waited after every complete token graph. The new path:

1. gives every generated token a distinct four-byte slot in a GPU history
   buffer;
2. binds slot `n` as the next embedding input and slot `n + 1` as the argmax
   output;
3. commits the two command buffers for that token without waiting;
4. relies on Metal command-queue ordering for hidden-state, KV, and token
   dependencies; and
5. retains a bounded number of submitted graphs, waiting on the oldest only
   when the configured window is full.

`DecodeGraph::submit` and `SubmittedDecode::wait` separate command submission
from error/metric collection. `GraniteV2Generator::generate_tokens_gpu_resident`
exposes the fixed-length runtime path and returns the complete generated token
history. Existing synchronous and EOS-aware generation remain unchanged.

## Queue depth

Window 4 is the smallest stable promoted depth:

| Window | Mean tok/s in the higher-performance sweep |
|---:|---:|
| 1 | 367.51 |
| 2 | 408.43 |
| 4 | **414.24** |
| 8 | 414.26 |
| 16 | 410.96 |

Window 8 provides no material gain over four, while window 16 increases
command-buffer retention and variance. In the conservative paired sample,
window 4 increased process physical footprint by approximately 2.35 MB; known
Metal model allocation remained unchanged.

The mechanism also holds at shorter contexts: context 128 improved from
379.00 to 437.42 tok/s and context 256 from 380.13 to 430.65 tok/s in the
screening runs.

## Correctness

Synchronous and window-4 execution over the complete 384-prefill/128-decode
fixture produced:

- identical 128-token history hash `19b4967c2513510e`;
- identical complete KV-arena hash `5f18e9cd28097e0b`;
- identical final token 34; and
- identical decoded text in the real tokenizer-backed generation command.

A Metal unit test separately submits a producer and dependent consumer graph
and verifies queue-ordered data visibility. The complete release workspace
test suite passes.

## Scope

The asynchronous API is exact for fixed-length greedy generation because no
host decision is needed between tokens. EOS-aware generation remains
synchronous: stopping early is token-dependent and therefore needs readback.
Windowed streaming can be added later by copying completed history ranges to
the host, but it is not part of this promotion.

## Reproduce

```bash
target/release/qx-bench \
  /tmp/granite-m5-pro-q4-all-superblock144-cb2-final.qx \
  --v2-token --fixture benchmarks/fixtures/general.json --context 512 \
  --v2-rollout-window 4

target/release/qx-bench \
  /tmp/granite-m5-pro-q4-all-superblock144-cb2-final.qx \
  --v2-token --fixture benchmarks/fixtures/general.json --context 512 \
  --v2-rollout-window 4 --v2-rollout-verify
```

Machine-readable evidence is in
`benchmarks/results/gpu-resident-rollout-2026-08-24.json`.
