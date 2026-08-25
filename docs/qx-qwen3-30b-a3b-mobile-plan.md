# Qwen3-30B-A3B local and mobile execution plan

Date: 2026-08-25  
Primary bring-up target: Apple M5 Pro, 24 GiB unified memory  
Second target: iPhone, exact device and usable application-memory ceiling TBD  
Model family: sparse Qwen3-30B-A3B only; dense 30B is out of scope

## Objective

Build a QX execution plan for Qwen3-30B-A3B that:

1. runs fully resident on the M5 Pro with quality-gated mixed-bit weights;
2. sustains interactive batch-1 generation rather than merely fitting;
3. measures whether an iPhone can run the complete model fully resident;
4. if it cannot, determines whether explicit expert residency can provide a
   predictable mobile tier without uncontrolled operating-system paging; and
5. only investigates PowerInfer-style intra-expert neuron skipping if vanilla
   Qwen exhibits enough predictable sparsity to justify an approximate path.

The reference checkpoint must be pinned by repository, revision, configuration
hash, tokenizer hash, and tensor hashes. The initial candidate is
`Qwen/Qwen3-30B-A3B`; an updated instruct checkpoint is a separate validation
target rather than an implicit replacement.

The published model topology is:

| Dimension | Value |
|---|---:|
| Total parameters | 30.5B |
| Active parameters/token | 3.3B |
| Layers | 48 |
| Hidden width | 2,048 |
| Query heads / KV heads | 32 / 4 |
| Head dimension | 128 |
| Experts/layer | 128 |
| Selected experts/token/layer | 8 |
| Expert intermediate width | 768 |
| Vocabulary | 151,936 |

## Non-negotiable distinctions

Three different claims must remain separate:

- **Capacity:** the artifact and runtime state fit inside a declared budget.
- **Residency:** the bytes needed by the next operation are already in unified
  memory, rather than merely represented by a virtual-memory mapping.
- **Interactivity:** delivered token latency remains bounded at sustained
  temperature and under cache misses.

Likewise, explicit MoE sparsity and intra-expert neuron sparsity are different.
Qwen already selects 8 of 128 experts. A baseline such as llama.cpp also avoids
executing unselected experts, so QX cannot count that 16x reduction as a novel
speedup. Skipping neurons inside a selected SwiGLU expert is approximate unless
the model is modified or retrained to create predictable zeros.

## Initial memory model

QX group-32 payload budgets include one FP16 scale per group:

| Weight plan | Idealized 30.5B payload |
|---|---:|
| all Q4 | 15.98 GiB |
| all Q3 | 12.43 GiB |
| all Q2 | 8.88 GiB |
| Q3 experts + Q4 shared/dense | approximately 12.6 GiB |
| Q2 experts + Q4 shared/dense | approximately 9.2 GiB |

These are lower-bound payload calculations, not application-memory promises.
Alignment, sensitive higher-precision tensors, tokenizer state, pipelines,
scratch, KV, and operating-system headroom must be added and measured.

With 48 layers, four KV heads, and head dimension 128, the per-sequence KV
budget is:

| Context | FP16 K/V | FP32 K + FP16 V |
|---:|---:|---:|
| 4,096 | 384 MiB | 576 MiB |
| 8,192 | 768 MiB | 1.125 GiB |
| 16,384 | 1.5 GiB | 2.25 GiB |
| 32,768 | 3.0 GiB | 4.5 GiB |

FP16 K/V is the first Qwen candidate. It must pass quality; Granite's
asymmetric result is not evidence that Qwen requires FP32 K.

## Work package 0: pin the reference and controls

The official Q4_K_M route-capture control is present under the ignored external
model directory and pinned by the committed reference manifest. Original
safetensors are still required before QX compilation and quality claims begin.

Acquire and pin:

- original safetensors and tokenizer files;
- an FP16/BF16 reference runner;
- pinned llama.cpp source and Q4/Q3 GGUF controls;
- a reproducible prompt/routing corpus;
- exact non-thinking and thinking sampling configurations where applicable.

Before implementation, record a manifest containing model revision, file
hashes, configuration, chat template, tokenizer, llama revision, compiler
revision, macOS version, and machine identity.

Exit criteria:

- reference logits and generated text are reproducible;
- llama control measurements run from exact token fixtures;
- model parameter accounting agrees with the checkpoint directory.

## Work package 1: turn fixed Granite shapes into compiled Qwen shapes

The current runtime hard-codes Granite constants in Rust and Metal. Qwen is not
an artifact swap. Introduce an artifact-sealed model descriptor covering:

```rust
ModelShape {
    layers,
    hidden,
    query_heads,
    kv_heads,
    head_dim,
    experts,
    active_experts,
    expert_intermediate,
    vocab,
}
```

Specialize kernels and buffer plans from this descriptor at compile/install
time. Do not turn every inner-loop dimension into a runtime branch. The first
Qwen kernels may remain shape-specific while the descriptor establishes a
general boundary.

Required semantic coverage:

- Qwen RMSNorm, RoPE, GQA, attention and residual ordering;
- router normalization and top-8 behavior;
- gate/up/down expert projections at 2,048 x 768 shapes;
- untied embedding and LM head behavior;
- Qwen tokenizer and chat templates;
- deterministic seeded sampling state.

Exit criteria:

- CPU/reference tensor-name and shape audit passes;
- one-layer FP16 parity passes;
- complete-model FP16/BF16 logit parity passes on fixed positions;
- router top-8 IDs and weights agree before quantization.

## Work package 2: establish the fully resident M5 baseline

Bring up representations in this order:

1. Q4 everywhere as the correctness and llama comparison control;
2. Q3-112 experts with Q4 attention, embedding, router and LM head;
3. sensitivity-selected Q2/Q3 experts only after the Q3 tier passes;
4. FP16 KV, followed by alternatives only when quality evidence requires them.

Q3-112 must remain consumer-native: no expanded expert tensor, no copied scale
slab, one selected-expert consumer, and no dispatch proportional to the number
of representations.

Generation needs GPU-resident sampling, not only argmax:

```text
logits -> temperature -> top-k/top-p -> seeded RNG -> token -> embedding
```

Prefill must use token-wide projection kernels. Sequentially invoking the
decode graph for every prompt token is not a promotable Qwen implementation.

M5 promotion gates:

- mapped/no-copy weights have no full CPU or Metal duplicate;
- Q3/Q4 weight payload is at most 15 GiB;
- total known logical memory at context 4K is at most 17 GiB;
- QX is at least 20% faster than same-session llama at matched quality, or an
  absolute target of 40 delivered tok/s at context 4K is reached;
- stretch target: 50 delivered tok/s at short context and 40 at context 4K;
- seeded token history, task quality, and NLL/PPL gates pass;
- 10-minute sustained throughput loses no more than 15% from the first minute;
- time to first token is reported separately from decode throughput.

## Work package 3: routing trace and cache simulation

This package can run on the M5 before an iPhone backend exists, and most of it
can run before QX has a complete Qwen Metal graph. Instrument the reference
runner to emit, for every token and layer:

```text
request_id, token_position, layer, expert_ids[8], router_weights[8]
```

Use calibration prompts only to select policies. Keep disjoint held-out prompts
for the final cache study. The trace set must cover chat, code, mathematics,
multilingual text, tool use, short prompts, long prompts, and multi-turn reuse.

Replay the trace through an offline cache simulator. Evaluate:

- static per-layer hot experts;
- LRU and segmented LRU;
- frequency plus recency;
- conversation-pinned working sets;
- expert co-selection packing;
- full-expert and 32/64/128-neuron packet units;
- resident budgets of 2, 3, 4, 6, 8, 10, and 12 GiB;
- Q2, Q3, and mixed packet byte sizes.

Record:

- byte-weighted hit rate;
- p50/p95/p99 missing bytes per token;
- p50/p95/p99 layers with at least one miss;
- expert churn and evictions/token;
- adjacent-token and conversation-level expert overlap;
- compulsory misses versus capacity misses;
- flash bytes read per generated token;
- cache warm-up length and prompt-to-prompt retention.

Go/no-go gate for partial residency:

- a phone-sized cache must achieve at least 98% byte-weighted hits on held-out
  prompts;
- p99 miss traffic must fit the device's measured I/O and latency budget;
- misses cannot be concentrated into enough sequential layers to violate the
  inter-token latency target.

An average hit rate alone is insufficient. One cold request that misses across
48 layers can be unusable despite a good corpus-wide mean.

## Work package 4: emulate constrained residency on the M5

Yes, the core design can be tested directly on this machine.

Build an explicit `ExpertCache` whose capacity is independent of host RAM. It
owns a fixed pool of Metal-visible staging regions and a block table mapping
`(layer, expert, packet)` to cache slots. Never describe normal `mmap` page
faulting as the experiment: the operating system may retain the entire file and
produce a false success.

Two test modes are required:

### Trace-only I/O replay

- Pack real Q2/Q3 expert packets into a file.
- Replay recorded routes through the explicit cache.
- Use uncached reads where the platform permits, or evict/recycle a bounded
  user-space buffer pool.
- Parameterize read latency and bandwidth rather than assuming the M5 SSD is
  an iPhone.
- Apply an I/O shaper using later phone measurements.

This validates cache policy, byte traffic, packing, metadata overhead and the
maximum possible overlap without needing a complete Qwen GPU runtime.

### End-to-end staged execution

- Execute resident expert packets immediately.
- Stage missing packets into fixed buffers.
- Overlap the eight independent experts within a layer where dependencies
  permit it.
- Preserve the layer boundary: routing for a future layer cannot be assumed
  before the preceding layer produces its activation.
- Measure actual token latency, not the sum of isolated read and kernel times.

Sweep cache sizes matching plausible phone budgets and synthetic device
profiles. For each cell record compute time, unhidden I/O time, overlap ratio,
cache hits, missing bytes, Metal allocation, physical footprint and output
parity.

Promotion requires:

```text
p95(compute + unhidden I/O) <= 100 ms/token
p99 inter-token latency      <= 200 ms
sustained throughput         >= 10 tok/s target, >= 6 tok/s floor
```

These are product gates, not claims that the M5 emulator reproduces iPhone
thermals.

## Work package 5: decide whether neuron clusters are credible

PowerInfer-2 is inspiration, not evidence that unmodified Qwen neurons can be
skipped. First measure the selected experts' gate activations:

- exact zeros;
- magnitude distribution;
- cluster-level near-zero rates;
- predictability from the layer input;
- stability across domains and positions;
- predictor cost and false-negative rate.

Only open an approximate neuron-cluster implementation if a held-out study
shows all of:

- at least 35% of selected-expert packet bytes can be skipped;
- predictor plus compaction consumes less than 20% of the saved latency;
- aggregate relative PPL change is at most 1%;
- per-domain relative PPL change is at most 2%;
- task gates show no statistically resolved regression;
- the end-to-end result improves over expert-only caching by at least 10%.

Otherwise retain exact expert-level sparsity and consider model fine-tuning as
a separate model-system co-design project.

## Work package 6: physical iPhone validation

The M5 can validate algorithms, representations, trace locality and constrained
cache behavior. It cannot validate:

- the per-process iOS memory ceiling or jetsam behavior;
- actual iPhone flash random-read latency and sustained bandwidth;
- Metal compiler/code-generation differences on the phone GPU;
- Core ML/ANE feasibility and cross-engine synchronization;
- sustained phone thermal throttling and battery drain;
- application lifecycle effects and background pressure.

The first iPhone harness should therefore measure capabilities before running
the model:

- maximum stable Metal-visible allocation;
- no-copy mapped-buffer behavior and physical footprint;
- sequential and random reads at Q3 packet sizes;
- CPU-to-Metal staging bandwidth;
- representative Q3/Q2 kernel throughput;
- 10-, 20-, and 30-minute thermal curves;
- power draw and memory-pressure termination threshold.

Then repeat the M5 experiment matrix on-device with the exact same artifact,
trace IDs, cache policies, seeds and result schema.

iPhone promotion gates:

- total committed memory remains at least 20% below the measured stable app
  ceiling;
- no jetsam or allocation failure across a 30-minute run;
- at least 10 tok/s target or 6 tok/s minimum at context 4K;
- p95/p99 inter-token latency satisfies the staged-execution gate;
- throughput after 20 minutes remains at least 80% of minute-one throughput;
- output and quality gates match the promoted M5 artifact family;
- energy/token and battery drain are reported, not hidden.

## Experiment matrix

Every performance cell uses the same tokenized fixtures and an interleaved
same-session control.

| Axis | Values |
|---|---|
| Weight plan | Q4; Q3 experts/Q4 shared; Q2 experts/Q4 shared; selected mixed |
| Residency | full; 12 GiB; 10 GiB; 8 GiB; 6 GiB; 4 GiB; 3 GiB |
| Cache policy | static-hot; LRU; segmented-LRU; frequency+recency |
| Packet unit | expert; 128 neurons; 64 neurons; 32 neurons |
| Context | 512; 2K; 4K; 8K; 16K where capacity permits |
| Workload | chat; code; math; multilingual; tools; multi-turn |
| Sampling | pinned non-thinking; pinned thinking; deterministic diagnostic |
| Duration | 128-token smoke; 1K-token sustained; 10/30-minute thermal |

Required metrics:

- delivered output tok/s, wall/GPU/I/O ms/token;
- time to first token and prefill tok/s;
- p50/p95/p99 inter-token latency;
- resident, mapped, Metal-allocated and physical bytes;
- byte-weighted cache hits and miss bytes/token;
- cache churn, flash read amplification and overlap efficiency;
- energy/token and thermal state;
- NLL/PPL, task accuracy and seeded generation agreement;
- same-session llama control where the model fits its execution mode.

## Execution order

Do not begin with neuron predictors or an iPhone application shell.

1. Pin checkpoint and llama/reference controls.
2. Produce routing traces and the cache working-set curve.
3. Generalize the artifact descriptor and obtain Qwen FP parity.
4. Promote a fully resident M5 Q4 baseline.
5. Promote Q3-112 experts/Q4 shared weights through quality and memory gates.
6. Add GPU sampling and wide prefill.
7. Measure the fully resident performance frontier.
8. Build trace-only constrained-cache replay on the M5.
9. Build end-to-end staged expert execution only if the cache curve passes.
10. Run the device capability probe and repeat the matrix on iPhone.
11. Open intra-expert neuron sparsity only if its measurement gate passes.

## Immediate experiment on this machine

The fastest informative next step does not require the QX Qwen runtime:

1. download and pin Qwen3-30B-A3B plus a working reference runner;
2. capture top-8 routes for at least 100,000 held-out routed token positions,
   combining teacher-forced corpora with representative generated sessions;
3. implement the offline expert-cache simulator;
4. produce hit-rate and miss-byte curves for 3/4/6/8/10/12 GiB budgets;
5. reject partial residency immediately if no phone-sized budget reaches the
   98% byte-weighted held-out gate.

This answers the central iPhone question before paying for Qwen Metal kernels:
whether explicit MoE locality is strong enough to make a sub-model-sized
resident working set predictable.

### Implemented route-capture harness

`qx-route-cache` implements the first executable slice of this experiment. On
macOS it links to the pinned llama.cpp control and observes the selected-expert
tensor for every MoE layer. It emits a streaming JSONL trace, so the corpus does
not have to reside in memory. Its simulator likewise streams the trace for each
policy/budget cell and measures real `(layer, expert)` packets rather than VM
pages.

Build it against the pinned llama checkout:

```sh
QX_LLAMA_ROOT=/Users/blas/projects/qx/external/llama.cpp \
  cargo build --release -p qx-route-cache
```

Capture a small correctness trace from the pinned GGUF:

```sh
target/release/qx-route-cache capture \
  --model /Users/blas/projects/qx/external/models/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf \
  --model-sha256 0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5 \
  --prompt "Explain why expert routing locality matters." \
  --request-id smoke-001 \
  --max-new-tokens 32 \
  --context 512 \
  --output /Users/blas/projects/qx/external/results/qwen3-route-smoke.jsonl
```

Replay Q3-112 full-expert packets through bounded LRU caches:

```sh
target/release/qx-route-cache simulate \
  --trace /Users/blas/projects/qx/external/results/qwen3-route-smoke.jsonl \
  --policies lru \
  --budgets 3GiB,4GiB,6GiB,8GiB,10GiB,12GiB \
  --output /Users/blas/projects/qx/external/results/qwen3-route-smoke-lru.json
```

`static-hot` requires a separate `--calibration` trace. It is deliberately not
permitted to learn frequencies from the held-out trace. The default
`2,064,384` bytes per expert is the exact group-32 Q3 payload for Qwen's three
2,048 by 768 expert matrices; Q2 and packetized studies must override
`--expert-bytes` or extend the trace schema before their results are compared.

The current capture path records route IDs, which are sufficient for residency
simulation. Router weights and activation magnitudes remain part of the later
neuron-cluster study and must not be inferred from this trace.

### First target-model smoke result

The pinned 18,556,685,824-byte GGUF was verified at SHA-256
`0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`.
It loaded on the 24 GiB M5 Pro and the harness captured valid top-8 routes from
all 48 layers. The first disjoint-prompt smoke used 80 calibration positions
and 78 held-out positions, or 29,952 held-out expert lookups.

For complete Q3 expert packets, the held-out curve was:

| Budget | LRU hit rate | Static-hot hit rate | LRU p95 miss bytes | Static p95 miss bytes |
|---:|---:|---:|---:|---:|
| 2 GiB | 57.42% | 36.57% | 594.6 MiB | 616.2 MiB |
| 3 GiB | 68.40% | 49.06% | 504.0 MiB | 553.2 MiB |
| 4 GiB | 78.73% | 59.70% | 391.8 MiB | 480.4 MiB |
| 6 GiB | 88.38% | 78.02% | 391.8 MiB | 311.1 MiB |
| 8 GiB | 88.46% | 87.77% | 391.8 MiB | 196.9 MiB |
| 10 GiB | 88.46% | 94.47% | 391.8 MiB | 86.6 MiB |
| 12 GiB | 88.46% | 100.00% | 391.8 MiB | 0 MiB |

At 8 GiB and above, cold LRU had zero capacity misses but still recorded 3,457
compulsory misses. Its p99 token had misses in all 48 layers. Static-hot needed
the complete 12 GiB expert tier to cross the 98% gate; at 10 GiB it reached
94.47% and its p99 token still missed in 18 layers. Thus naive full-expert
paging below the complete-expert payload fails the gate in this smoke. The
result does not yet reject partial residency: two prompts are far too small for
a locality claim, the cache started cold, and neuron packets may have
materially different locality. It does reject treating average MoE sparsity as
evidence that whole-expert paging will work automatically.

The committed aggregate is
`benchmarks/results/qwen3-30b-a3b-route-cache-smoke-2026-08-25.json`.
Raw traces remain ignored under `external/results`.

### Exact neuron-packet replay

The second replay implements physical execution packets rather than arbitrary
byte pages. A packet owns one matching intermediate-neuron range through the
entire expert: gate rows, up rows, down columns, and their quantization
metadata. The exact replay always touches every packet of every routed expert.
It therefore tests transfer granularity, placement and scheduling—not
approximate neuron skipping.

The simulator now sweeps complete experts and 128/64/32-neuron packets across
four layouts: expert-major, neuron-major, and calibration-derived co-selection
orders for both traversals. It records missing bytes per layer and token,
contiguous fetch runs, maximum bursts, reuse distance, churn, and serial,
unlimited-latency-overlap, and finite-inflight I/O bounds. The shaped model
uses one shared bandwidth budget; parallel requests never multiply storage
bandwidth.

On the current two-prompt smoke, exact packetization left capacity effectively
unchanged. At 4 GiB, static-hot moved only from 59.699% for complete experts to
59.719% for 32-neuron packets due to final-slot rounding. Both had a 480.375
MiB p95 missing-byte tail. At 6 GiB, every exact packet size had a 311.063 MiB
p95 tail. This is the expected invariant: all 768 neurons are still consumed.

Neuron-major placement did change burst shape. At 4 GiB, the largest observed
fetch fell from 5.906 MiB for expert-major to 0.984/0.492/0.246 MiB for
128/64/32-neuron identity ordering. But p95 fetch count grew from 241 to
1,446/2,892/5,784. Co-selection ordering reduced those neuron-major counts by
about 8%, to 1,326/2,652/5,304, without changing missing bytes.

Under a deliberately synthetic shared 1,000 MiB/s, 100 us/fetch, eight-inflight
profile, the best 4 GiB p95 I/O estimate remained approximately 397 ms for
cold LRU; the 6 GiB static-hot result remained approximately 316 ms. These are
not device measurements, and both exclude compute. They demonstrate the
structural limit: concurrency can amortize request latency, but hundreds of
missing MiB divided by shared bandwidth remains hundreds of milliseconds.

The exact packet lane therefore fails the 100 ms token gate on this smoke. Do
not continue tuning exact packet sizes for capacity. The next experiment is
activation-energy capture followed by a perfect-information packet oracle at
90/95/99/99.9% retained activation energy. Only if that oracle reaches a 3--4 GiB
working set at negligible quality loss should a learned predictor be built.

The committed exact-replay aggregate is
`benchmarks/results/qwen3-30b-a3b-exact-packet-replay-smoke-2026-08-25.json`.

### Activation-energy oracle smoke

Because the exact lane failed, the llama bridge now optionally captures the
post-SwiGLU intermediate for every selected expert. It reduces the raw tensor
immediately into L1 and squared-L2 energy for 128/64/32-neuron packets. The
`oracle` command sorts those packets with perfect knowledge and asks how many
are required to cross 90/95/99/99.9% squared-L2 energy.

This is deliberately a proxy. Post-SwiGLU energy is not the packet's weighted
down-projection contribution, logit error, NLL, or task quality. It establishes
whether obvious contiguous activation sparsity exists before paying for a
masked down-projection and quality harness.

For 29,952 selected-expert observations from the 78-position Qwen smoke:

| Packet | 90% energy | 95% energy | 99% energy | Full |
|---:|---:|---:|---:|---:|
| 128 neurons | 5.47 | 5.93 | 5.99 | 6 |
| 64 neurons | 9.86 | 10.94 | 11.96 | 12 |
| 32 neurons | 18.09 | 20.69 | 23.53 | 24 |

At 99% energy, the perfect selector retained 98.06% of 32-neuron packets and
99.67% of 64-neuron packets. At the much looser 90% target, 32-neuron packets
still retained 75.37%, corresponding to approximately 569.8 MiB of active Q3
expert payload per token instead of 756 MiB. The p95 selected count was 20 of
24 packets. This is not a 2x--10x sparsity opportunity.

Therefore a learned predictor for the current contiguous packet definition is
not justified: it cannot beat the perfect-information oracle. The remaining
credible research step is contribution-aware measurement on a much larger,
disjoint corpus—ideally packet-wise down-output norm and then masked-logit/NLL
curves. If that remains dense, the unmodified-Qwen mobile lane is a model
co-design problem, not a cache-policy problem.

The committed oracle aggregate is
`benchmarks/results/qwen3-30b-a3b-activation-energy-oracle-smoke-2026-08-25.json`.

### Down-contribution oracle smoke

The next control replaces activation magnitude with the packet's actual vector
after the selected expert's quantized down projection. The llama callback reads
the selected Q6_K down rows, dequantizes them with llama's own type traits, and
computes one output vector per contiguous neuron packet. Their sum reconstructs
llama's unweighted selected-expert output with a `2.64e-7` p95 and `5.10e-7`
maximum relative L2 error on the current run.

For each expert, a perfect-knowledge greedy selector repeatedly chooses the
packet that minimizes the next exact residual. This is stronger than ranking
post-SwiGLU energy, although it is not a globally optimal subset search. Router
weighting is a scalar per expert and therefore does not change its relative
error. Cross-expert cancellation, residual-stream effects, logits, NLL and task
quality remain outside this control.

The expanded smoke contains 22 routed positions from a disjoint code/math
prompt, or 8,448 selected-expert observations:

| Packet | 10% relative L2 | 5% relative L2 | 1% relative L2 | Full |
|---:|---:|---:|---:|---:|
| 128 neurons | 5.98 | 5.99 | 6.00 | 6 |
| 64 neurons | 11.92 | 11.98 | 11.99 | 12 |
| 32 neurons | 23.33 | 23.90 | 23.98 | 24 |

For 32-neuron packets, even the 10% error target retains 97.20% of packets and
approximately 734.8 MiB of the 756 MiB exact active payload per token. The
p50, p95 and p99 expert all require 24 of 24 packets. At a 5% error target the
mean retained fraction is 99.59%; at 1% it is 99.93%.

The inverse budget curve is equally decisive:

| 32-neuron packets retained | Active payload | Mean relative L2 error | p95 error |
|---:|---:|---:|---:|
| 25% | 189 MiB/token | 66.24% | 76.82% |
| 50% | 378 MiB/token | 47.31% | 57.48% |
| 75% | 567 MiB/token | 29.38% | 37.15% |
| 91.7% | 693 MiB/token | 14.99% | 19.77% |

This remains smoke evidence, not a quality study or promotion result. But the
gap is too large to justify a learned predictor or masked-logit harness for the
same contiguous packet definition: a predictor cannot manufacture the missing
oracle sparsity, and the 2x capacity point already injects roughly 50% local
output error. Reopen masked NLL only with a different structural hypothesis,
such as learned non-contiguous neuron groups, structured pruning or a
co-designed sparse checkpoint. The unmodified checkpoint's current mobile
lane is model co-design, not cache-policy tuning.

The committed aggregate is
`benchmarks/results/qwen3-30b-a3b-down-contribution-oracle-smoke-2026-08-25.json`.

## Evidence layout

Keep generated model files and raw traces under ignored `external/` paths.
Commit reproducible manifests, aggregate traces and reports under:

```text
benchmarks/qwen3-30b-a3b-*.json
benchmarks/results/qwen3-30b-a3b-*.json
docs/qx-qwen3-30b-a3b-*.md
```

Every report must label results as `reference`, `full-resident`,
`trace-emulated`, `io-shaped`, or `physical-iphone`. Trace emulation is never
reported as measured iPhone throughput.
