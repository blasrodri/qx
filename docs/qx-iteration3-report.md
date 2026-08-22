# QX PoC — Iteration 3 Report

Date: 2026-08-21  
Target: Apple M5 Pro  
Model: IBM Granite 3.0 1B-A400M Instruct  
Workload: batch 1, greedy decode

## Executive result

Iteration 3 produced the required measurements and a controlled execution
experiment, but the one-command-buffer redesign is not complete and the
performance gate is not met.

| Question | Result |
|---|---|
| Current orchestration cost | ~100 command buffers/token, ~348 dispatches/token |
| CPU wait cost | ~35.0 ms/token on the development fixture |
| CPU-dense/GPU-MoE experiment | 28.5 → 37.6 tok/s |
| Q3 expert latency | ~282 µs vs ~238–239 µs for Q2/Q4 samples |
| Q2/Q4 storage reduction | 14.99% |
| Q2/Q4 short-slice PPL delta | +57.4% |
| Teacher-forced comparison | Operational: NLL, PPL, KL, cosine, top-k |

The measurements reject the idea that dense-GEMV synchronization alone explains
the full 11× gap. The remaining boundary is inside the transformer layer and
MoE path: QX still reads intermediate results to the CPU before continuing
attention, routing, residual, and the next layer.

## Runtime instrumentation

`qx-metal` now records command buffers, compute dispatches, GPU start/end time,
CPU encode time, and CPU wait time. The development fixture measured:

```text
command buffers: 3201 (100.03/token)
dispatches:      11121 (347.53/token)
GPU time:        647.95 ms / 33 tokens = 20.249 ms/token
CPU encode:      5.78 ms / 33 tokens = 0.181 ms/token
CPU wait:        1119.04 ms / 33 tokens = 34.970 ms/token
```

The current operators still own command-buffer lifecycle and synchronously
read outputs. This is now measurable, but it is not yet the target execution
context. A true one-buffer token requires GPU-resident activation, routing, KV,
attention, and residual operations; merely moving `commit()` into a shared
Rust wrapper would be incorrect because the next CPU operation needs the prior
GPU result.

## Controlled dense-path experiment

`QX_CPU_DENSE=1` disables Metal Q/K/V/O/router/LM-head dense dispatches while
retaining the GPU expert path. This is an attribution mode, not production
architecture.

| Mode | Decode tok/s | Command buffers/token | CPU wait/token |
|---|---:|---:|---:|
| Existing all-Metal dense path | 28.52 | 100.03 | 34.97 ms |
| CPU dense + GPU MoE | 37.55 | 24.75 | 22.37 ms |

The experiment confirms dense-GEMV orchestration is real cost, but removing it
does not reach the 100 tok/s minimum. The next redesign must keep complete
layer state on the GPU or use an optimized graph backend.

The corrected matched long-fixture baseline remains:

```text
QX uniform Q4:   28.33 tok/s
llama.cpp Q4_K_M: 310.72 tok/s
ratio:            9.1%
```

## Expert GPU economics

`qx-bench` now supports real Metal expert timing in addition to the CPU
reference measurement. Sampled real Granite experts produced:

| Format | Sample | GPU time/op | GPU wall/op |
|---|---|---:|---:|
| Q2 | mixed artifact layer 0/expert 1 | 237.92 µs | 422.96 µs |
| Q3 | Q3/Q4 artifact layer 0/expert 0 | 282.04 µs | 439.14 µs |
| Q4 | uniform Q4 layer 12/expert 17 | 239.10 µs | 411.00 µs |

These are single-expert synchronous-dispatch measurements, not full-token
latency. Q3 has measurable extraction overhead and is not automatically a
performance format.

## Q2/Q4 alternative

The compiler now accepts `--disable-q3`, allowing an explicit Q2/Q4 search:

```text
uniform Q4 bytes: 679,477,248
Q2/Q4 bytes:     577,634,304
storage saving:  14.9884%
distribution:    Q2=518, Q3=0, Q4=1018
```

On the 64-token PTB development slice:

```text
candidate PPL:       151.414494
PPL delta:           +57.42%
teacher delta NLL:   +0.453768
teacher KL mean:      0.973624
teacher cosine:       0.875607
teacher top-1:        51.56%
```

Regular runtime speed was 18.28 tok/s versus 28.81 tok/s for uniform Q4 in
the same short comparison. Q2/Q4 is not quality-preserving at this allocation.

## Teacher-forced quality

`GraniteModel::teacher_forced_logits` consumes a fixed prefix and emits the
logits for each next-token prediction. `qx-bench` reports baseline/candidate
NLL, delta NLL, PPL, KL mean/median/p95, logit cosine, and top-1/top-5/top-10
overlap. For Q3/Q4 on the 64-token PTB development slice:

```text
baseline NLL:        4.566337
candidate NLL:       4.624298
delta NLL:           +0.057962
KL mean/median/p95:  0.447214 / 0.272131 / 1.266690
logit cosine:        0.934368
top-1/top-5/top-10:  68.75% / 64.06% / 67.50%
```

This is more interpretable than free-generation agreement because both models
receive exactly the same prefix at every position. The requested multi-domain
corpus is not available locally; these are PTB-only development results.

## Gate status

| Requirement | Status |
|---|---|
| No CPU wait inside a transformer token | Not achieved |
| Approximately one command buffer/token | Not achieved; current path ~100 |
| Uniform Q4 ≥100 tok/s | Not achieved; 28.3 baseline, 37.6 diagnostic |
| GPU-level timing | Achieved |
| Q2/Q3/Q4 expert benchmark | Achieved |
| Teacher-forced quality comparison | Achieved |
| Q3/Q4 and Q2/Q4 evaluated | Achieved; both fail quality/speed gates |
| Automated Pareto frontier | Partial; measurements exist, sweep remains |

## Decision

Do not move to a larger model yet. The next iteration should implement a real
GPU-resident token graph or adopt optimized Metal/MPS primitives for dense and
attention operations, then remeasure. The compiler specialization remains
measurable, but the current execution engine is not yet credible enough to
attribute a hardware-level advantage to it.
