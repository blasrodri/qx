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

### Free-neuron and learned-group controls

The contiguous-packet rejection does not imply that every individual neuron is
equally important. The capture path now records the exact contribution rank
identity for every neuron. For 768-way selection it uses perfect knowledge of
each neuron's down-output norm, then measures the exact residual after adding
neurons in that order. This norm-ranked screen is scalable, but unlike the
32/64/128-packet control it is not greedy residual minimization and is not a
globally optimal subset oracle.

On four Qwen positions and 1,536 expert observations, individual-neuron
selection is materially sparser than contiguous packets:

| Relative L2 target | Mean neurons | Fraction | p95 neurons | Ideal Q3 payload |
|---:|---:|---:|---:|---:|
| 25% | 218.3 / 768 | 28.42% | 309 | 214.9 MiB/token |
| 10% | 397.4 / 768 | 51.75% | 482 | 391.2 MiB/token |
| 5% | 500.5 / 768 | 65.17% | 573 | 492.7 MiB/token |
| 1% | 650.2 / 768 | 84.67% | 693 | 640.1 MiB/token |

A fixed 50% neuron budget produces 11.81% mean and 17.53% p95 per-expert
relative L2 error. This is not a quality result, but unlike contiguous packets
it is close enough to a 2x theoretical payload tier to justify testing whether
those neurons can be packed into fixed physical groups.

The grouping control learns a target-specific permutation independently for
each observed `(layer, expert)`. It packs neurons with matching calibration
selection signatures together, then evaluates disjoint held-out positions. A
group is resident if any desired individual neuron belongs to it; compute may
still mask unwanted neurons. This is a capacity-coverage result, not grouped
output error, logits or NLL.

At a 10% individual-output error target, the two-position held-out smoke gives:

| Fixed group | Individual fraction | Resident fraction | Calibrated-only | Load amplification | p95 groups |
|---:|---:|---:|---:|---:|---:|
| 8 neurons | 50.31% | 79.15% | 72.22% | 1.57x | 96 / 96 |
| 16 neurons | 50.31% | 81.52% | 75.16% | 1.62x | 48 / 48 |
| 32 neurons | 50.31% | 84.51% | 79.12% | 1.68x | 24 / 24 |

Of 768 held-out expert observations, 570 had a calibration layout and 198 used
identity fallback. The calibrated-only columns show that insufficient expert
coverage is not the whole failure. Co-selection signatures improve simple
frequency packing by only about one percentage point, and p95 still touches
every fixed group even at eight-neuron granularity.

Therefore the unmodified checkpoint exposes a dynamic individual-neuron signal
but not a deployable fixed-group residency plan. Do not build an irregular
single-neuron storage runtime from this smoke: metadata, fetch granularity and
gathered matvec cost would erase much of the ideal payload win. Use the result
as a model-co-design target—structured pruning, group-lasso/distillation or
router-conditioned training must make importance stable and group-aligned.
Only then rerun the same held-out coverage and masked-NLL gates before opening
a grouped Metal kernel.

The committed aggregate is
`benchmarks/results/qwen3-30b-a3b-free-neuron-grouping-smoke-2026-08-25.json`.

### Fixed eight-neuron mask quality control

The next control applies a physically realizable fixed neuron mask to the real
Qwen execution. Calibration ranks are packed into eight-neuron co-selection
groups independently per observed `(layer, expert)`. Every expert then retains
the same fixed number of groups; layer-pooled statistics provide a deterministic
fallback for experts absent from calibration. The llama.cpp evaluation callback
reads the post-SwiGLU tensor, zeros pruned neurons, writes it back before the down
projection, and computes teacher-forced target NLL from the resulting logits.

This is post-training structured pruning, not structured training: weights are
not updated. Arbitrary learned groups are nevertheless physically realizable by
permuting the corresponding gate/up rows and down columns together when an
artifact is compiled.

The all-one-mask control exactly reproduces the unmodified run over 17 scored
held-out positions: mean NLL `4.7713409224`, zero NLL delta, and 100% top-1
agreement. The callback mutation path is therefore semantically valid for this
smoke.

The fixed-group capacity curve is poor:

| Retained expert neurons | NLL delta | Relative PPL delta | Top-1 agreement |
|---:|---:|---:|---:|
| 93.75% | +0.1081 | +11.42% | 88.24% |
| 87.50% | +0.2264 | +25.40% | 70.59% |
| 81.25% | +0.2292 | +25.76% | 64.71% |
| 75.00% | +0.2837 | +32.80% | 47.06% |
| 62.50% | +0.9703 | +163.87% | 35.29% |
| 50.00% | +1.2657 | +254.55% | 41.18% |

The two sweeps use four and six calibration positions respectively. The larger
one observes 1,314 of 6,144 layer/expert pairs directly; all other masks use the
layer fallback. The corpus is far too small for promotion, but the result is
already sufficient to reject naive pruning as the capacity mechanism. It cannot
remove even 6.25% of expert neurons without an order-of-magnitude miss against
QX's 1% relative-PPL quality gate.

The next credible capacity experiment requires an unquantized training
checkpoint and a training stack. Distill or fine-tune with an explicit
eight-neuron group penalty, freeze the chosen groups into gate/up-row and
down-column permutations, quantize, and rerun this exact tensor-mask harness on
the sealed quality corpus. Do not build the grouped storage kernel until a
candidate reaches all of these gates:

1. at least 50% expert-neuron removal;
2. aggregate and worst-domain relative-PPL delta at or below 1%;
3. no material HumanEval or GSM8K regression;
4. stable group identities on held-out prompts rather than oracle-selected
   groups;
5. capacity accounting over the complete artifact, including dense weights and
   KV state.

The committed aggregate is
`benchmarks/results/qwen3-30b-a3b-structured-nll-smoke-2026-08-25.json`.

### Expert-local residual distillation control

End-to-end BF16 training is not physically possible on the first target
machine. The upstream checkpoint index reports 61,064,245,248 bytes of model
weights, while this M5 Pro has 24 GiB unified memory; optimizer and activation
state would increase the gap substantially. QX therefore implements a serial
expert-local control that keeps the 17 GiB quantized teacher resident and trains
only one reduced down projection at a time.

The capture path records real post-SwiGLU activations and unweighted teacher
down outputs whenever a specified `(layer, expert)` is routed. It also exports
that expert's quantized down matrix once in dequantized row-major form. The
trainer retains fixed eight-neuron groups, uses the retained teacher columns as
the student initialization, and fits only the missing output residual with a
dual ridge solve. This is equivalent to a linear down-projection correction;
gate/up weights remain fixed.

Layer 8, expert 38 was chosen because it appeared on all six earlier calibration
positions. A systems-oriented passage produced 25 routed samples; a disjoint
ecology/statistics passage produced 31 held-out samples. The full-width control
reconstructs the quantized teacher below `5.5e-7` p95 relative L2, validating
activation capture, down-weight export, and projection orientation.

The disjoint result rejects the tiny-data fit:

| Retained neurons | Masked teacher mean L2 | Distilled mean L2 | Distilled p95 | Cosine |
|---:|---:|---:|---:|---:|
| 93.75% | 0.1570 | 0.1634 | 0.2455 | 0.9838 |
| 75.00% | 0.3718 | 0.3812 | 0.5145 | 0.9177 |
| 50.00% | 0.5981 | 0.6146 | 0.7969 | 0.7697 |

The residual fit worsens mean held-out error at every pruned width. Do not feed
this student into full-model NLL: it fails the cheaper local gate. The earlier
adjacent-position split showed a small apparent improvement, so the disjoint
failure is also direct evidence that same-passage expert distillation can give
misleading results.

The machine can execute a larger serial expert experiment, but data—not the
linear solve—is now the bottleneck. The next attempt needs hundreds to thousands
of diverse hits per layer/expert and must optimize gate/up/down jointly or use
group-aware pretraining. A 1,024-token capture context failed with a Metal
out-of-memory error at position 9; context 128 completed. Capture should
therefore stream independent short sequences and reset KV state between them.

The committed aggregate is
`benchmarks/results/qwen3-30b-a3b-expert-local-distillation-smoke-2026-08-25.json`.

### Corpus-scaled expert-local distillation

The bounded-context capture now resets llama.cpp KV state in place and streams
many independent sequences through one loaded model. WikiText-2 train and test
were captured in separate model sessions. The training trace contains 135
layer-8/expert-38 hits from 238 positions across 16 sequences; the held-out
trace contains 114 hits from 144 positions across 12 sequences. Every sequence
starts at position zero. Independently exported teacher matrices are
byte-identical, which checks that reset and session boundaries do not alter the
sampled expert.

This larger, explicitly disjoint control reverses the 25-sample result. At
93.75% down width and ridge 0.1, the learning curve is:

| Training hits | Masked teacher mean L2 | Distilled mean L2 | Distilled p95 |
|---:|---:|---:|---:|
| 25 | 0.1491 | 0.1277 | 0.2684 |
| 50 | 0.1491 | 0.1246 | 0.2504 |
| 100 | 0.1491 | 0.1194 | 0.2411 |
| 135 | 0.1491 | 0.1181 | 0.2268 |

The best bracketed regularizer is 0.3, reaching 0.1164 mean relative L2: a
21.9% relative reduction from the fixed masked teacher. The signal also
survives larger reductions at ridge 0.1:

| Retained down width | Masked teacher mean L2 | Distilled mean L2 | Distilled p95 |
|---:|---:|---:|---:|
| 93.75% | 0.1491 | 0.1181 | 0.2268 |
| 87.50% | 0.2124 | 0.1752 | 0.3279 |
| 75.00% | 0.3310 | 0.2588 | 0.5048 |
| 50.00% | 0.5309 | 0.3973 | 0.6941 |

The ridge evaluator uses a linear kernel. It now explicitly folds the dual
solution into one compact primal down matrix and compares both paths on every
held-out sample. The maximum relative disagreement is `7.41e-7`; the 1.49 MiB
support bank is therefore a training representation, not an inference cost.
At 93.75% width the resulting raw Q4 down payload is 737,280 bytes instead of
786,432 bytes per expert.

The folded matrix was also injected into live llama.cpp execution for this one
expert. On all 12 held-out sequences, 132 scored tokens included 106 target
expert hits. Mean NLL changed from `3.791721` to `3.776882`, or -1.473% relative
perplexity, with 97.73% top-1 agreement. A paired four-sequence full-width
reconstruction control changed perplexity by only +0.0816% with 100% top-1;
the 93.75%-width student changed it by -1.786% with 93.75% top-1 on the same
48 tokens and 34 hits. Thus the single-expert regression gate passes and the
callback is not merely a no-op or generic reconstruction improvement.

This is positive model-co-design evidence, but it is not promotion evidence or
a claim that pruning improves Qwen quality. The held-out corpus is small and
single-domain, only one of 6,144 expert down projections is replaced, and
gate/up storage is untouched. The next gate is representative coverage across
layers and experts on a larger sealed multi-domain corpus. Do not build a Metal
kernel until that aggregate stays within the 1% relative-PPL budget. A hardware
tier reduction still requires joint gate/up/down group-aware training rather
than down-only post-hoc fitting.

The committed aggregate is
`benchmarks/results/qwen3-30b-a3b-expert-local-distillation-corpus-2026-08-25.json`.

### Routing-preselected depth screen

To test whether the layer-8 result was hand-picked, the next targets were fixed
from training-route frequency before looking at quality: the most frequently
routed experts at layers 2, 24, and 44. Each target used the same WikiText
train/test partitions, 93.75% retained down width, and an independent ridge
grid. The best ridge is therefore an expert specialization parameter rather
than a transferred global constant.

| Target | Train/test hits | Ridge | Masked/distilled mean L2 | PPL delta | Top-1 agreement |
|---|---:|---:|---:|---:|---:|
| L2 / E125 | 96 / 40 | 1.0 | 0.1779 / 0.1701 | -0.096% | 100.00% |
| L24 / E2 | 161 / 59 | 3.0 | 0.2047 / 0.1989 | -0.206% | 98.48% |
| L44 / E6 | 179 / 92 | 3.0 | 0.2053 / 0.1927 | -0.212% | 99.24% |

All three separately injected students remain non-regressing over the same 132
held-out target tokens, so the layer-8 pass is not isolated to one depth. The
local benefit is much smaller, however: 2.8% to 6.1% relative improvement
versus 22% for layer 8, and low regularization materially overfits the late
expert. A compiler cannot assign one universal distillation policy.

This screen still replaces only one expert per run. It does not prove that
errors compose when multiple students are active in one forward pass. The next
gate is simultaneous injection of these four preselected students on the same
sealed corpus. Promotion and kernel work remain blocked until the combined
relative-PPL delta stays inside 1%.

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
