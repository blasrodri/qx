# QX 350+ tok/s implementation

## Result

The next complete-model optimization is implemented and reaches a conservative
three-run mean of **350.81 tok/s** on the M5 Pro, measured over sequential
positions 384--511. Wall time is **2.851 ms/token** and GPU time is **2.583
ms/token**. A prior five-run sample of the identical payload and graph measured
356.18--376.42 tok/s, or 367.51 tok/s mean; the lower later result is retained
as the headline because it better represents observed system-state variance.

This is a 21.1% throughput improvement over the 289.59 tok/s MoE-superblock
candidate and is beyond the locally measured 310--320 tok/s llama.cpp control.
The final artifact has 390 dispatches, two encoders, two command buffers, and no
CPU waits per token.

| Complete token graph | tok/s | wall ms/token | GPU ms/token |
|---|---:|---:|---:|
| MoE-only Superblock144 | 289.59 | 3.454 | 2.975 |
| All-dense superblocks, one command buffer | 341.48--344.75 | 2.901--2.928 | 2.426--2.459 |
| Final, two command buffers (three-run mean) | **350.81** | **2.851** | **2.583** |
| Final, earlier five-run mean | 367.51 | 2.722 | 2.426 |

## What changed

The payload-contiguous 144-byte Q4 superblock is now used by every large dense
decode projection, not only MoE gate/up and down: fused QKV, attention output,
router, tied embedding lookup, and fused normalized LM-head argmax. The Metal
consumers use aligned 32-bit payload loads, SIMD scale shuffles, and multi-row
work ownership. Normal embedding lookup was also changed from one serial GPU
thread to 256 threads.

Two dataflow changes remove redundant graph work:

- Attention and MoE residual kernels calculate the inverse RMS needed by the
  following projection, so the next RMS reduction does not reread the hidden
  state in a separate dispatch.
- Decode attention softmax performs max, exponentiation, sum, and normalization
  in one threadgroup dispatch per query head instead of two dependent passes.

Finally, the MoE superblock consumer processes 32 rows per threadgroup, and a
two-command-buffer schedule creates a queue boundary before MoE without a CPU
wait. On short contexts the second buffer primarily improves wall scheduling;
it is not presented as a reduction in raw GPU work.

## Context dependence

The Q4 representation, dense kernels, embedding/head kernels, and cross-layer
RMS reuse are sequence-length independent: they perform the same work per
token at positions 0 through 511. Attention itself is not context independent;
QK and value aggregation grow with the KV length. The one-threadgroup softmax
is valid throughout the artifact's sealed maximum context of 512, but no claim
is made beyond that supported window.

The measured wall-time gain survives the context sweep:

| Context / measured positions | 1 CB tok/s | 2 CB tok/s |
|---|---:|---:|
| 128 / 0--127 | 349.46 | 364.32 |
| 256 / 0--255 | 346.89 | 365.73 |
| 512 / 384--511 | 341.48--344.75 | 350.81 conservative mean |

## Correctness and rejected candidates

Full CPU/GPU parity passes at positions 0, 1, 7, 31, 63, 127, 255, and 511:
minimum logit cosine is 1.0, greedy argmax agrees 8/8, maximum absolute logit
error is 0.009906, and maximum attention error is 0.000127. The full release
workspace test suite passes.

The following locally faster-looking microkernel ideas were rejected after
complete-token measurement: fused router/top-k, 64-position value-attention
chunks, four-row generic dense geometry, and eight-row LM-head geometry. None
is enabled in the final graph.

Machine-readable measurements are in
`benchmarks/results/all-superblock-rms-softmax-cb2-2026-08-24.json`.

## Reproduce

```bash
target/release/qx-compile \
  /Users/blas/projects/qx/external/models/granite-3.0-1b-a400m-instruct \
  --target apple-m5-pro --memory-budget 900mb --context 512 --disable-q3 \
  --expert-layout superblock-144 --dense-layout superblock-144 \
  --weight-storage mmap-no-copy --moe-consumer canonical \
  --moe-down-consumer canonical --command-buffers-per-token 2 \
  --plan-id granite-m5-top8-all-superblock144-cb2 \
  --output /tmp/granite-m5-pro-q4-all-superblock144-cb2-final.qx

target/release/qx-bench \
  /tmp/granite-m5-pro-q4-all-superblock144-cb2-final.qx \
  --v2-token --fixture benchmarks/fixtures/general.json --context 512
```
