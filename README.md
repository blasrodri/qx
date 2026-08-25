# QX PoC

QX is a deliberately narrow model-by-hardware compiler experiment for IBM
Granite 3.0 1B-A400M Instruct on Apple GPUs. Runtime V2 is now a complete,
artifact-only, batch-1 Metal decode path; it is not a general inference
runtime.

The current canonical status, accepted/rejected experiments, unresolved gates,
and next work are summarized in
[docs/qx-current-status.md](docs/qx-current-status.md). Iteration reports are
retained as the chronological experiment log and contain historical numbers.

The next model/device expansion is planned around sparse Qwen3-30B-A3B on the
M5 Pro followed by a measured iPhone residency tier. The staged implementation,
M5 cache emulation, quality gates, and physical-device acceptance criteria are
defined in
[docs/qx-qwen3-30b-a3b-mobile-plan.md](docs/qx-qwen3-30b-a3b-mobile-plan.md).

The current PoC provides:

- a hard-coded Granite V0 model description;
- memory-mapped safetensors index/shard inspection;
- deterministic, groupwise symmetric Q2/Q3/Q4 CPU quantization;
- a native Granite ByteLevel BPE tokenizer with role/tool special tokens;
- CPU grouped-query attention, RoPE, and a bounded single-sequence KV cache;
- safetensors-backed Granite decoder-layer loading for Q/K/V/O, RMSNorm, and routing;
- a batch-1 greedy `GraniteModel` wrapper covering all 24 decoder layers and tied logits;
- Q3 packing at 8 values per 3 bytes;
- a memory-mappable `.qx` artifact with expert directory metadata;
- a static Granite memory plan and a greedy Q2/Q3/Q4 allocator;
- compiler-selectable row-major, MetalTile32, mixed, Q3, and Q4_K expert
  representations with layout metadata in artifact v2;
- compiled Q2/Q3/Q4 Metal kernels, plus Apple GPU dispatch tests from both Swift
  and the Rust `qx-metal` boundary.
- a fused top-8 GPU MoE dispatch and runtime benchmark/quality comparison
  harness.

The workspace has been exercised against the real IBM Granite checkpoint on an
Apple M5 Pro. External fixtures live under `external/` and are intentionally
ignored by Git because they are several gigabytes.

The current Iteration 2 correctness, matched-benchmark, profiling, and
quality-aware quantization results are documented in
[docs/qx-iteration2-report.md](docs/qx-iteration2-report.md). The headline
result is that the harness and 8-token FP parity gate pass, while uniform Q4
remains far behind llama.cpp and the Q3/Q4 quality experiment is not yet
generation-preserving.

Iteration 3 instrumentation and execution experiments are documented in
[docs/qx-iteration3-report.md](docs/qx-iteration3-report.md). The current
runtime measures roughly 100 Metal command buffers per token; the controlled
`QX_CPU_DENSE=1` mode reaches 37.6 tok/s but does not yet provide a GPU-resident
one-command-buffer transformer path.

Iteration 4 adds the separate `qx-runtime-v2` static Granite execution plan,
preallocated Metal graph buffers, encode-only dense/logits primitives, fixed
shape decode kernels, GPU argmax, and an artifact-only 24-layer token graph.
Prompt prefill, same-artifact Q4 logits parity, and FP32 cross-implementation
parity are implemented. The status is recorded in
[docs/qx-iteration4-report.md](docs/qx-iteration4-report.md).

The V2 static memory audit can be inspected with:

```bash
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-q4-dense.qx --v2-plan --context 512
```

The real-weight 24-layer attention-only V2 graph can be measured with:

```bash
cargo run --release -p qx-bench -- \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --v2-attention-stack --context 512 --warmup 1 --iterations 3
```

This attention-only diagnostic is not full-model generation throughput.

The complete artifact-only V2 token graph, including GPU MoE, Q4 LM head, GPU
argmax, and sequential KV positions, can be exercised with:

```bash
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-q4-dense.qx \
  --v2-token --fixture benchmarks/granite_decode_512.json --warmup 3
```

The current release control uses scalar-only RMS consumers, SIMD fused QKV,
split/direct RoPE-KV, three-stage attention, hierarchical GPU argmax, and
four-stage batched SIMD MoE. Artifact v2 can map every immutable weight through
a cached no-copy Metal buffer, eliminating the former ~1.9 GB copied-weight RSS
peak. Complete context-512 decode is currently about 200 tok/s, with 461
dispatches/token, zero internal waits, and roughly 4.7 ms of measured GPU work.
Command-buffer count is now compiler-selectable rather than an invariant.

On the matched context-512 release decode, the release numbers are:

| | llama.cpp | QX V2 |
|---|---:|---:|
| model bytes | 821,844,576 B | 751,009,792 B artifact |
| weight format | Q4_K_M | row-major Q4 |
| CB/token | — | 1 control; 2-buffer candidate |
| internal waits | — | 0; one final wait |
| GPU ms/token | not exposed by normal run | ~4.7 |
| wall/eval ms/token | ~3.1 | ~5.0 |
| tok/s | ~326 latest smoke | ~200 latest smoke |

These are development measurements, not a final superiority claim. The pinned
20-pair contract, triangular quality gate, current proof status, and exact
acceptance criteria are documented in
[docs/qx-validation-plan-implementation.md](docs/qx-validation-plan-implementation.md).

The same-artifact CPU/GPU parity gate can be run with:

```bash
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-q4-dense.qx \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --v2-parity --prompt 'Hello world, how are you?' --context 512
```

The validated eight-position run through position 511 reaches 0.009914
maximum logit error, 1.000000 minimum cosine, and 8/8 argmax agreement.

The current kernel-only control benchmark compares the real QX Q4 GEMV with a
simple FP16-storage GEMV over the same dequantized weights. It reports weight,
activation, and output bytes, median/p95 GPU time, variance, and effective
bandwidth:

```bash
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-q4-dense.qx \
  --v2-kernel-bench --iterations 200
```

The current M5 Pro scalar result is 1.80–8.42 GB/s for QX Q4 versus
18.49–19.44 GB/s for the simple FP16 control. The first SIMD-cooperative Q4
control reaches 31.75 GB/s and 18.83 us on the dense Q/O shapes, about 7.6×
faster than scalar Q4 in that run. This is still a kernel-only diagnostic. A
local llama.cpp replay profiler covers selected MoE and decode nodes, but those
replay timings are not additive live costs; the whole-model 323.58 tok/s result
remains the competitive reference.

The Q-projection geometry sweep covers rows per SIMD group, SIMD groups per
threadgroup, and direct versus threadgroup-cached input:

```bash
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-q4-dense.qx \
  --v2-kernel-sweep --iterations 20
```

The best parameterized mapping in the current release run was 1 row/SIMD, 8
SIMD groups, cached input: 21.12 us median versus a 332.63 us scalar baseline,
with zero max-absolute-difference against the validated SIMD reference.

The same one-row-per-SIMD mapping now also covers `MetalTile32` expert gate/up
and down stages. In the latest release run they measured 12.13 us and 7.75 us
respectively, versus 92.38 us and 38.83 us for the scalar Q4 stages; both
matched the scalar outputs exactly.

`qx-generate` now uses canonical Runtime V2 by default; pass `--runtime v1` for
the original correctness runtime:

```bash
cargo run --release -p qx-generate -- \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --artifact external/models/granite-m5-pro-q4-dense.qx \
  --prompt Hello --max-new-tokens 16 --context 512
```

The model directory is used for tokenizer files only; V2 weights come from
the artifact. Same-artifact Q4 logits parity passes. The release FP32
cross-implementation parity gate also passes with 24/24 layers, 100% router
agreement, 1.000000 logit cosine, and matching top-1.

## Fetch and build the fixtures

The local validation fixture contains the original Hugging Face safetensors
checkpoint, tokenizer files, llama.cpp, and a PTB test slice. The model is kept
in safetensors for QX; GGUF files are generated only for the llama.cpp baseline.

The already-prepared fixture paths are:

```text
external/models/granite-3.0-1b-a400m-instruct/
external/models/granite-3.0-1b-a400m-instruct-f16.gguf
external/models/granite-3.0-1b-a400m-instruct-q3_k_m.gguf
external/models/granite-3.0-1b-a400m-instruct-q4_k_m.gguf
external/llama.cpp/
external/data/ptb.test.txt
```

On a new macOS machine, install the Metal toolchain if `xcrun metal` is not
available, then build llama.cpp with Metal enabled:

```bash
xcodebuild -downloadComponent MetalToolchain
cmake -S external/llama.cpp -B external/llama.cpp/build-metal \
  -DGGML_METAL=ON -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build external/llama.cpp/build-metal --config Release \
  --target llama llama-bench llama-quantize llama-perplexity --parallel 8
```

## Inspect a checkpoint

The inspector expects the original Hugging Face directory, including
`config.json` and either `model.safetensors.index.json` or
`model.safetensors`:

```bash
cargo run --release -p qx-inspect -- \
  --model ./granite-3.0-1b-a400m-instruct
```

Inspect and quantize one real expert:

```bash
cargo run --release -p qx-inspect -- \
  --model ./granite-3.0-1b-a400m-instruct \
  --layer 0 --expert 0 --quantize q3
```

The quantizer is a CPU reference implementation. The Metal kernel sources are
compiled and dispatched through the Rust `qx-metal` boundary. The runtime now
has a complete batch-1 greedy model path; optimized batching, asynchronous
execution, and sampling strategies remain outside V0.

## Compile an artifact

Without a calibration trace, compilation is intentionally conservative and
emits Q4 for every expert plus Q4 row-major dense projections, router, and
embedding weights. Normalization vectors are stored losslessly as raw FP32:

```bash
cargo run --release -p qx-compile -- \
  external/models/granite-3.0-1b-a400m-instruct \
  --target apple-m5-pro \
  --memory-budget 800mb \
  --context 512 \
  --weight-storage mmap-no-copy \
  --command-buffers-per-token 2 \
  --benchmark-contract-hash 7ca84dd6e2113e1f3828d2dde69f66060f4c87a65b3d125ae4e8b3e2f7ab318c \
  -o external/models/granite-m5-pro-q4-rowmajor-nocopy-2cb-v2.qx
```

With a `QXTRACE` generated by the llama.cpp instrumentation bridge, the
compiler measures Q2/Q3/Q4 expert output error and applies the greedy mixed-bit
allocator:

```bash
cargo run --release -p qx-compile -- \
  external/models/granite-3.0-1b-a400m-instruct \
  --target apple-m5-pro \
  --trace granite.qxtrace \
  --memory-budget 512mb \
  --quality-loss 0.05 \
  -o granite-a19.qx
```

Use `--experts-only` only for the legacy expert-only artifact. The complete
artifact contains packed expert/dense weights, lossless norm entries,
per-tensor directory entries, the selected target ID, and a static memory plan.
`qx-format::MappedArtifact` opens it with memory-mapped views. Apple targets
use `MetalTile32` for expert MoE tensors; dense GEMV tensors use row-major Q4,
with the attention Q/K/V projections dispatched through the fused QKV kernel.

## Reference benchmark

This measures the CPU reference expert path and is useful for parity checks;
it is not a GPU performance claim:

```bash
cargo run --release -p qx-bench -- granite-a19.qx --iterations 100
```

`qx-tokenizer` owns the native Granite tokenizer, and `qx-runtime` exposes
`RuntimeTokenizer`, `KvCache`, Granite RoPE/GQA attention primitives, and the
fixed-shape `granite_moe_decode` top-8 API. `GraniteLayerWeights::load` and
`GraniteLayerWeights::decode` now execute one real checkpoint decoder layer by
combining safetensors attention/router weights with the Metal expert path, while
`qx-metal` owns the Q2/Q3/Q4 Metal GEMV sources and the modern objc2-metal
device/queue boundary. On macOS, `MetalContext::run_fused_expert` compiles the
embedded MSL, binds shared packed-weight/FP16 buffers, dispatches one 1024-thread
Granite expert, and returns the FP16 output vector. `CompiledMoeLayer::from_artifact`
now maps a QX layer and routes top-8 experts through that path; callers that use
`CompiledMoeLayer::new` retain the CPU reference implementation. Metal pipelines
and each used expert's packed weight buffers are cached after first use, so the
runtime no longer recompiles MSL or reuploads weights for every routed token.

## Baseline and kernel checks

The QX CPU artifact benchmark is a parity check, not a GPU claim:

```bash
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-q4.qx --iterations 3
```

Compile the Metal sources and run the numerical GPU dispatch check. The probe
now covers isolated GEMV and the fused full-expert path for all three formats:

```bash
xcrun -sdk macosx metal -c kernels/q2_expert.metal -o /tmp/qx-q2.air
xcrun -sdk macosx metal -c kernels/q3_expert.metal -o /tmp/qx-q3.air
xcrun -sdk macosx metal -c kernels/q4_expert.metal -o /tmp/qx-q4.air
swiftc tools/qx-metal-probe.swift -o /tmp/qx-metal-probe \
  -framework Metal -framework Foundation
/tmp/qx-metal-probe
```

On the Apple M5 Pro, the fused probes produced the expected 374.302 output
(observed 374.25): Q2 15,323 experts/s, Q3 3,915 experts/s, and Q4 10,154
experts/s. These are kernel smoke-throughput figures for one synthetic expert,
not end-to-end model decode results.

For the prepared M5 Pro fixture, the matched llama.cpp Metal results are:

| representation | model bytes | decode tok/s | PTB PPL* |
|---|---:|---:|---:|
| Q4_K_M | 820,107,736 | 254.75 | 16.62 |
| Q3_K_M | 656,841,176 | 234.59 | 19.89 |
| QX uniform Q4 | 679,477,248 expert bytes | 28.18* | 30.956** |
| QX mixed Q2/Q3/Q4 | 552,665,088 expert bytes | 17.69* | 38.954** |

`*` PPL is a one-chunk, 512-token PTB test-slice measurement and is only a
sanity baseline, not a full benchmark claim. `**` QX figures are warm
batch-1 runtime measurements from `qx-bench`; the llama.cpp figures above are
the separately captured llama benchmark fixture and are not a final matched
quality claim. The QX mixed result is intentionally shown because it is the
negative end-to-end result of this PoC.

The real activation trace is `external/results/granite-calibration.qxtrace`.
With a 620 MB expert budget and a 0.30 weighted per-expert relative-L2 target,
the compiler selected 244 Q2, 802 Q3, and 490 Q4 tensors. Expert storage fell
from 679,477,248 bytes to 552,665,088 bytes (18.66%). The sensitivity report
is [qx-sensitivity-report-620mb.csv](external/results/qx-sensitivity-report-620mb.csv).

The reproducible full comparison command is:

```bash
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-tile-q4.qx \
  --compare-artifact external/models/granite-m5-pro-mixed-620mb.qx \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --quality-text external/data/ptb.test.txt --quality-tokens 512 \
  --iterations 3 --warmup 1 --max-new-tokens 16 --context 512
```

Against uniform tiled Q4, the mixed artifact produced 0.0000 top-1
agreement, 0.509673 mean logit cosine, KL 13.077152, and a +25.84% PPL delta
on the 512-token PTB slice. Its generated text also diverged. Uniform Q4 was
28.18 tok/s after warmup; mixed Q2/Q3/Q4 was 17.69 tok/s in the quality run.
These measurements mean the local storage gate is met, but the end-to-end
quality and speed gates are not.

## Current gate status

| gate | result | evidence |
|---|---|---|
| Gate 1: Q2/Q3/Q4 correctness and differing sensitivity | PASS | pack/unpack tests, Metal dispatch tests, real-trace sensitivity report |
| Gate 2: >=15% mixed storage reduction at matched quality | PARTIAL / FAIL end-to-end | 18.66% reduction at the compiler's local 0.30 error target, but +25.84% downstream PPL delta |
| Gate 3: identical-Q4 specialized kernel >=10% faster | NOT PASSED | kernel smoke tests pass; no valid identical-quantization generic end-to-end speed advantage has been demonstrated |
| Gate 4: smaller and faster quality-matched baseline | FAIL | mixed QX is slower than the QX uniform reference and materially worse in measured quality |

The exact physical development target for these measurements is Apple M5 Pro.
No M5 result is being represented as an A19/iPhone result. The current PoC is
therefore a completed vertical slice with a measured negative result for the
full thesis, not evidence of a Pareto advantage.

The actual GPU probe currently reports PASS for Q2/Q3/Q4 on `Apple M5 Pro`.
The Rust-side macOS test also reports PASS for Q2/Q3/Q4, including the Q3
three-byte-per-eight-values packing path:

```bash
cargo test -p qx-metal --lib
```

The local llama.cpp fixture now includes a calibration-only bridge enabled by
`QX_TRACE_PATH`. It pins the Granite MoE input, top-k expert IDs, and normalized
router weights, then writes the real hidden vectors after Metal execution:

```bash
QX_TRACE_PATH=external/results/granite-calibration.qxtrace \
  external/llama.cpp/build-metal/bin/llama-simple \
  -m external/models/granite-3.0-1b-a400m-instruct-f16.gguf \
  -ngl 99 -n 16 -c 1024 \
  "calibration prompt text"
```

The prepared trace is 184,778,520 bytes and contains real Granite routing
records. It is accepted by `qx-compile` and produces the measured 620 MB mixed
artifact described above. The stricter 0.05 and 0.15 bounds remain infeasible
at 800 MiB. The compiler-side 0.30 target is therefore useful for testing the
allocation thesis, but it does not by itself guarantee downstream model
quality; the end-to-end PPL and logit comparison is the deciding measurement.

The complete fixture-backed greedy generation smoke test loads the full
checkpoint and runs all 24 decoder layers through the mapped QX expert path:

```bash
cargo test -p qx-runtime --lib \
  tests::complete_granite_model_generates_from_a_prompt -- --ignored --nocapture
```

The same path is available as a command-line tool:

```bash
cargo run --release -p qx-generate -- \
  --model external/models/granite-3.0-1b-a400m-instruct \
  --artifact external/models/granite-m5-pro-q4.qx \
  --prompt "Hello" \
  --max-new-tokens 16
```

## Test

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
