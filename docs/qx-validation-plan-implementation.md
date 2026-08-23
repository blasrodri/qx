# QX specialization compiler validation implementation

This document maps the revised validation plan to executable repository
components. It deliberately distinguishes implemented infrastructure from
completed scientific proof.

## Proof model

- **Runtime proof:** quality-neutral top-8 QX reaches or beats pinned llama.
- **Specialization proof:** a QX-specific plan improves a mature control by at
  least 5% in latency or physical footprint without violating quality.
- **Compiler proof:** install-time measured selection chooses that plan from
  model, target, workload, quality, and memory constraints.

The final claim is a Pareto claim. Passing one proof does not imply the others.

## Release/provenance contract

The provisional winning-plan validation contract is
`benchmarks/granite-m5-pro-validation-nocopy-2cb-v2.json`. It pins source/model
hashes, five exact-token fixtures, 20 paired blocks, and the M5 Pro target.
`qx-validate` randomizes candidate order within each pair, retries blocks with
thermal-state transitions, reports paired bootstrap intervals per fixture and
globally, and requires no statistically significant fixture regression.
The artifact separately embeds the SHA-256 of
`benchmarks/granite-m5-pro-workload-v1.json`, avoiding a circular artifact hash
while sealing the exact workload into the compiled plan.

```bash
cargo build --release --workspace

target/release/qx-validate \
  --contract benchmarks/granite-m5-pro-validation-nocopy-2cb-v2.json \
  --artifact external/models/granite-m5-pro-q4-rowmajor-nocopy-2cb-v2.qx \
  --llama-simple external/llama.cpp/build-metal/bin/llama-simple \
  --gguf external/models/granite-3.0-1b-a400m-instruct-q4_k_m.gguf \
  --output validation-paired.json
```

Use `--verify-only` to hash-check all pinned inputs without running decode.

Headline runs must use release binaries without profiler substitutions. Final
runs require AC power, stable thermal state, clean Git provenance, and 20–30
pairs. `--pairs 1` is smoke-only.

## Triangular quality gate

`qx-eval` defines exact corpus and logit-trace formats. `qx-quality` compares
FP16/BF16 reference, llama Q4_K_M, and QX traces. WikiText-2 and Dolly drive
the NLL/PPL gates; HumanEval and GSM8K drive task-accuracy gates while retaining
teacher-forced NLL, KL, cosine, and top-k diagnostics. Text domains require at
least 32,768 scored positions and grow to 131,072 where the interval cannot
resolve one percentage point of relative PPL.

Build a reproducible corpus manifest before generating traces:

```bash
target/release/qx-quality-corpus \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --wikitext quality/raw/wikitext-2-heldout.txt \
  --dolly quality/raw/dolly-heldout.txt \
  --humaneval quality/raw/humaneval-heldout.jsonl \
  --gsm8k quality/raw/gsm8k-heldout.jsonl \
  --source-revision DATASET_REVISION \
  --output quality/granite-heldout
```

The manifest seals the source revision, split, sampling seed and strategy,
tokenizer hash, raw-source hash, processed-corpus hash, and token/record
counts. It writes `wikitext.tokens`, `dolly.tokens`, `humaneval.jsonl`, and
`gsm8k.jsonl` beside `manifest.json`.

```bash
target/release/qx-quality prepare \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --source DATA.txt --domain wikitext-2 --split heldout \
  --tokens 32769 --output wikitext-heldout.qxcorpus

target/release/qx-quality compare \
  --corpus wikitext-heldout.qxcorpus \
  --reference fp16.qxlogits --llama llama-q4km.qxlogits \
  --candidate qx.qxlogits --output wikitext-report.json
```

HumanEval and GSM8K are paired task gates. The external HumanEval sandbox must
emit JSONL records containing `task_id` and `passed`; GSM8K records may contain
`passed` or numeric-normalizable `prediction` and `reference` fields.

```bash
target/release/qx-task-eval \
  --human-eval-llama humaneval-llama.jsonl \
  --human-eval-candidate humaneval-qx.jsonl \
  --gsm8k-llama gsm8k-llama.jsonl \
  --gsm8k-candidate gsm8k-qx.jsonl \
  --output task-gates.json

target/release/qx-quality gate \
  --report wikitext-report.json --report dolly-report.json \
  --report humaneval-report.json --report gsm8k-report.json \
  --tasks task-gates.json --output quality-gate.json
```

Calibration and held-out corpora are separate. A failed or underpowered gate
is carried into autotuning as `accepted=false` and is unconditionally rejected.

## Corrected differential attribution

The full and MoE profilers prefill the selected fixture through position 384.
They capture the canonical 24-layer route trace and replay it for substitutions
so removing a stage does not silently change later expert selection. Results
are complete-token marginal A/B savings and must not be added to predict tok/s.

```bash
target/release/qx-bench ARTIFACT.qx \
  --v2-differential-profile --fixture benchmarks/fixtures/general.json \
  --result-json differential.json

target/release/qx-bench ARTIFACT.qx \
  --v2-moe-differential-profile --fixture benchmarks/fixtures/general.json \
  --result-json moe-differential.json
```

## Artifact-v2 candidates and automatic selection

Every candidate artifact contains a plan ID plus representation, topology,
command-buffer schedule, memory ownership, KV/sparse policy, target/workload,
budgets, and provenance. Existing candidate families are represented by enums;
diagnostic-only plans cannot be sealed as production plans.

Exact MoE consumer dataflow is also sealed as `moe_consumer` in the plan. The
current compiler-selectable values are `canonical` and `unique-packet`; the
runtime derives the Metal feature selection from the artifact, so a candidate
cannot win only because a benchmark process happened to carry an environment
override.

The compiler can emit copied, hybrid, or full no-copy weights and one or more
GPU-ordered command buffers. Multiple buffers preserve queue ordering and use
one CPU wait on the final buffer.

```bash
target/release/qx-compile MODEL \
  --target apple-m5-pro --memory-budget 8589934592 --context 512 \
  --expert-format qx-q4 --expert-layout row-major \
  --weight-storage mmap-no-copy --command-buffers-per-token 2 \
  --plan-id granite-m5-top8-rowmajor-nocopy-2cb \
  --output candidate.qx
```

`qx-autotune --suite` randomizes complete-model candidate execution across all
fixtures, combines those measurements with kernel, producer/consumer, and
multi-layer screening evidence, reads each sealed plan from its artifact, and
selects by measured complete-model latency subject to quality and memory.

For exact transport/layout candidates, a suite entry may instead provide
`exact_equivalent_to`. The runner then compares every decoded quantized tensor
against the reference artifact, including codes and FP16 scales. This is a
strict quality-neutral path for representation candidates such as row-major
versus row-inline Q4; lossy candidates still require the triangular quality
gate. The selection report records the equivalence result, complete-model
latency for every candidate, and the measured reason for the winner.

The current search also includes the first explicit live-MoE topology
candidate, `four_stage_wide_expert`. It is deliberately feature-flagged and
not canonical. Its initial screening was negative: a 1024-threadgroup-per-
expert mapping increased live gate/up and down latency and failed to remain
usable through the long parity replay. The canonical four-stage mapping is
therefore still selected by default. The result is retained as evidence that
the compiler must optimize occupancy and full execution context, not merely
reduce repeated activation staging.

```bash
target/release/qx-autotune \
  --suite autotune-suite.json --write-evidence autotune-evidence.json \
  --output selected-plan.json --report selection-report.json
```

The human supplies implementations, constraints, and screening measurements;
the selector—not a hard-coded default—chooses the winner. Complete-model
latency always outranks a microbenchmark result.

The first sealed consumer selection is recorded in
`benchmarks/results/unique-packet-selection-report-2026-08-23.json`: across
five context-512 fixtures and three randomized interleaved rounds, the
selector chose `granite-m5-top8-rowinline-unique-packet` over the canonical
row-inline plan. Both artifacts passed exact Q4 code/scale equivalence; both
kept one command buffer, 461 dispatches/token, and the same artifact size.

## Memory ownership controls

Artifact-v2 weight data is 16 KiB page aligned. Expert scale and payload slabs
are contiguous per layer/kind. Full no-copy creates one cached
`newBufferWithBytesNoCopy` view over the mmap and represents tensors as buffer
offsets. Hybrid keeps MoE no-copy while copying dense/QKV/router/embedding
weights. Copied mode retains the old ownership control. The arena and KV cache
remain explicit persistent Metal allocations; decode performs no allocation.

On one same-fixture smoke:

| Ownership | RSS | physical footprint | Metal allocated | wall ms/token |
|---|---:|---:|---:|---:|
| copied | 1,966,702,592 | 1,306,691,504 | 802,406,400 | 5.173 |
| hybrid | 196,067,328 | 214,648,008 | 873,840,640 | 5.143 |
| full no-copy | 20,824,064 | 110,609,008 | 802,373,632 | 5.111 |

These are diagnostics, not the final memory claim. Final comparison uses
current/peak RSS, current/peak physical footprint, mapped bytes, known Metal
allocations, KV/arena/scratch, and load versus steady state on both engines.

## Q4_K control

`qx-quant` implements Q4_K with a C-reference byte test derived from pinned
llama code. `qx-compile --expert-format q4k` changes only expert gate/up and
down tensors; dense families remain QX Q4. Dedicated Metal kernels consume the
format. This distinguishes representation, kernel mapping, routed access, and
whole-program scheduling. Q4_K is a control, not a presumed winner.

## Current blockers to final acceptance

- Full held-out domain data and paired task outputs are not present locally.
- The latest machine state is battery-powered, unsuitable for final 20–30 pair
  superiority claims.
- Top-8 QX remains materially slower than llama; corrected attribution points
  to live MoE gate/up and down.
- macOS accounting is not yet semantically symmetric across the two engines.
- Approximate specialization remains frozen until top-8 is within 10% of llama.

The implementation therefore supports the revised experiment, but the runtime,
specialization, compiler-repeatability, and Pareto proofs must still be earned
by the gated measurements above.
