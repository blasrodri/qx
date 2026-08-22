# QX PoC — Iteration 2 Report

Date: 2026-08-21  
Target: Apple M5 Pro  
Model: IBM Granite 3.0 1B-A400M Instruct  
Workload: batch 1, greedy decode

## Executive result

Iteration 2 established a reproducible correctness and benchmark harness, but
the technical success gate is not met yet.

| Question | Result |
|---|---|
| Unquantized QX reproduces Granite? | Yes for the 8-token parity gate |
| Uniform Q4 reaches 80% of llama.cpp? | No: 28.33 vs 310.72 tok/s |
| Mixed Q3/Q4 saves at least 10% expert storage? | Yes: 16.90% |
| Mixed Q3/Q4 stays within 3% PTB PPL? | Exploratory result: +2.51% |
| Mixed Q3/Q4 preserves generation behavior? | No: 6.25% top-1 agreement, logit cosine 0.6488 |
| Mixed Q3/Q4 is faster than uniform Q4? | No: 23.57 vs 28.57 tok/s in the quality run |

The result supports continuing runtime redesign and broader quality
measurement. It does not yet support claiming a competitive specialized
artifact.

## A. Numerical parity

Canonical fixtures are stored in `benchmarks/`. The llama.cpp bridge accepts
the same integer IDs through `QX_TOKEN_IDS`, and `QX_TRACE_ONCE` limits capture
to the prompt graph. QXTRACE version 2 records the token position, avoiding
selection of padded or generated rows in multi-token graphs.

The 8-token gate passed with an unquantized QX model:

```text
layers compared:    24
MoE input cosine:   1.000000
router agreement:   100.00%
weight cosine:      1.000000
logit cosine:       1.000000
logit top-1:        MATCH
status:             PASS
```

The two-token fixture also passes with the same metrics. The parity runtime
captures layer input, attention, post-attention residual, normalized MoE input,
router logits, top-8 IDs and weights, MoE output, layer output, final hidden,
and logits. The current llama bridge exports the MoE input/routing records and
final logits as the trusted cross-implementation comparison surface; richer
attention/residual tensor export remains a follow-up if exact tensor-by-tensor
reference diagnostics are required.

Uniform Q4 is intentionally not expected to pass unquantized parity. Against
the same single-token reference its first observed downstream divergence was
layer 1 MoE input, with cosine 0.993290; this is the expected quantization
error signal rather than a parity-harness failure.

Parity command:

```bash
QX_TOKEN_IDS=8279,5788,8279,5788,8279,5788,8279,5788 \
QX_TRACE_ONCE=1 \
QX_TRACE_PATH=/tmp/granite-parity.qxtrace \
QX_LOGITS_PATH=/tmp/granite-parity.qxlogits \
external/llama.cpp/build-metal/bin/llama-simple \
  -m external/models/granite-3.0-1b-a400m-instruct-f16.gguf \
  -ngl 99 -n 1 -c 64

cargo run --release -p qx-parity -- \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --float-reference --fixture benchmarks/granite_parity_8.json \
  --reference /tmp/granite-parity.qxtrace \
  --reference-logits /tmp/granite-parity.qxlogits --context 64
```

## B. Matched benchmark

The primary fixture is `benchmarks/granite_decode_512.json`: one canonical
prompt token, context 512, 128 generated tokens, batch 1, greedy sampling.
The llama bridge now parses `-c` and the canonical token environment; passing
`-c` no longer changes the prompt.

| Configuration | Expert/model bytes | Peak RSS | TTFT p50 | Decode p50 | Decode p95 | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| llama Q4_K_M | 820,107,736 | 909,590,528 | 3.79 ms | 408.25 ms | — | 310.72 |
| QX uniform Q4 | 679,477,248 expert | 2,909,634,560 | 35.55 ms | 4,517.69 ms | 4,517.69 ms | 28.33 |

The llama measurement used `llama_perf_context_print` for decode timing, an
explicit first-decode timestamp for TTFT, and the bridge’s `getrusage` peak RSS
line. QX used three measured runs after one warmup. The QX/llama decode ratio
is 9.1%, far below the 80% gate.

QX command:

```bash
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-tile-q4.qx \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --fixture benchmarks/granite_decode_512.json \
  --iterations 3 --warmup 1 --ignore-eos
```

llama command:

```bash
QX_TOKEN_IDS=8279 \
external/llama.cpp/build-metal/bin/llama-simple \
  -m external/models/granite-3.0-1b-a400m-instruct-q4_k_m.gguf \
  -ngl 99 -n 128 -c 512
```

The comparison is protocol-matched, but QX and llama use different Q4
representations and execution implementations. It is therefore a runtime
baseline, not evidence that the quantizers have identical numerical quality.

## C. Uniform-Q4 profile

The development fixture profiled 33 tokens including the one-token prefill.
Aggregate component time was:

| Component | Time |
|---|---:|
| Attention path (RMSNorm, Q/K/V, RoPE, attention, O) | 451.07 ms |
| Router and top-k | 167.10 ms |
| Fused Q4 MoE expert path | 495.27 ms |
| Residual writeback | 0.09 ms |
| LM head | 36.72 ms |

The per-layer profile is available directly from `qx-bench --profile`.
The QX Metal boundary currently commits and waits for each prepared dense,
logits, and fused-expert command buffer. That creates repeated CPU/GPU
synchronization across the 24-layer decode path. This is the identified,
isolated architectural limitation behind the large gap; it is a runtime
execution issue that can be redesigned independently of the compiler’s
packing and allocation logic.

The current profile does not split the fused expert kernel into separate
gate/up, activation, down, and accumulation GPU timings. Those operations are
inside one fused dispatch and require GPU timestamp instrumentation for the
next runtime pass.

## D. Quality-aware Q3/Q4 experiment

The conservative artifact starts from all Q4 and selectively uses Q3. The
compiler output was:

```text
expert tensors: 1536
uniform Q4 bytes: 679,477,248
compiled expert bytes: 564,658,176
Q2/Q3/Q4 assignments: 0 / 1168 / 368
expert storage reduction: 16.8981%
```

On the 512-token PTB slice:

```text
uniform Q4 PPL: 30.956153
Q3/Q4 PPL:     31.733565
delta:         +0.777412 (+2.51%)
```

However, the 16-token generation comparison showed:

```text
top-1 agreement: 0.0625
logit cosine:    0.648800
KL:              13.726995
QX Q4 speed:     28.57 tok/s
Q3/Q4 speed:     23.57 tok/s
```

Therefore the artifact meets the exploratory PTB PPL threshold but does not
meet a general quality-preservation or speed gate. PTB is an evaluation set
for this experiment, not a sufficient multi-domain calibration/evaluation
corpus.

The model-level sensitivity tool now supports expert-level and matrix-level
(`gate-up`, `down`, or `both`) Q4→Q3 candidates. Two measured candidates were:

| Layer/expert | Matrix | ΔPPL | ΔNLL | Bytes saved |
|---|---|---:|---:|---:|
| 0/0 | both | -0.207744 (-0.67%) | -0.006734 | 196,608 |
| 23/31 | both | +0.099173 (+0.32%) | +0.003199 | 196,608 |

This confirms that downstream sensitivity varies and local reconstruction
error cannot be the sole allocation objective. A full model-level allocator
still needs a separated multi-domain calibration/evaluation corpus and a
larger candidate sweep.

Sensitivity command:

```bash
cargo run --release -p qx-sensitivity -- \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --baseline external/models/granite-m5-pro-tile-q4.qx \
  --quality-text external/data/ptb.test.txt --quality-tokens 512 \
  --context 512 --matrix both \
  --candidate 0:0 --candidate 23:31 \
  --output external/results/qx-model-sensitivity-v2.csv
```

## Gate decision

| Gate | Status | Reason |
|---|---|---|
| Minimum: parity, matched benchmark, full runtime profile, gap source | Achieved | 8-token FP reference parity passes; benchmark and profile are reproducible; synchronization/launch overhead is identified |
| Technical: QX Q4 ≥80% llama, ≥10% mixed savings, ≤3% PPL, no slowdown | Not achieved | Throughput is 9.1% of llama; generation quality and mixed speed also fail |
| Strong: QX ≥llama and ≥15% quality-preserving mixed savings | Not achieved | Same reasons |

## Next work, in priority order

1. Remove per-operation `waitUntilCompleted` from the normal decode path and
   submit a larger fused layer graph or asynchronous dependency chain.
2. Add Metal GPU timestamps and split the fused expert path into gate/up,
   activation, down, accumulation, submission, and wait time.
3. Export the remaining trusted-reference tensors for attention/residual/final
   hidden comparison.
4. Build separate general-English, instruction, code, and reasoning
   calibration/evaluation slices; sweep the Q3/Q4 Pareto frontier.
5. Re-run Q2 only after Q3/Q4 preserves model-level quality and its kernel
   economics are measured.

Specifically frozen: speculative decoding, expert caching/offload, iPhone SDK,
Swift bindings, new model support, multi-batch, long context, server work,
and generic autotuning.
