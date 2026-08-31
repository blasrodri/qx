# QX profile-guided model compilation

Status: implementation plan

## Product contract

QX should compile a model checkpoint into a target-specific, quality-gated
artifact. The compiler machinery is reusable; learned compact weights, exact
exceptions, and measured execution plans belong to one checkpoint and target
tier.

```text
model checkpoint + calibration profile + target profile + quality gate
                                |
                                v
                     QX model compiler
                                |
                                v
packed artifact + exact map + runtime plan + evidence bundle
```

The intended product entry point is:

```sh
qx compile-model \
  --model Qwen3-30B-A3B-Q4_K_M.gguf \
  --target iphone17-pro-max \
  --memory-budget 7GiB \
  --workload general-local-assistant-v1 \
  --quality-gate qwen-general-v1 \
  --output Qwen3-30B-A3B-iphone17.qx
```

This command runs during artifact production, not on an end user's phone. A
publisher compiles and validates a checkpoint once for a supported target
class, signs the result, and distributes it as a normal model asset.

## What is reusable and what is checkpoint-specific

QX owns the reusable machinery:

- model inspection and semantic validation;
- routed or dense activation capture;
- calibration/held-out separation and provenance;
- codec-aware reconstruction and quantization-aware training;
- mixed-precision candidate generation;
- under-evidenced fallback policy;
- full-model causal exact-anchor selection;
- consumer-native packing;
- memory/residency planning;
- target execution autotuning;
- release quality, performance, and physical-memory gates;
- resumable state, content hashes, and failure explanations.

Every checkpoint produces its own:

- reconstructed compact weights;
- tensor/expert precision map;
- exact exception and causal-anchor map;
- packed artifact slices and integrity hashes;
- measured target runtime plan;
- quality and device evidence bundle.

Even two checkpoints with the same Qwen shape can have different router
behavior and sensitive experts. An anchor such as `3:82` must never be copied
to another checkpoint without replay. Compilation results are reusable only
when the checkpoint, tokenizer, adapter, compiler, codec, gate, and relevant
target fingerprints match.

## Three independent contracts

### Model adapter

A `ModelAdapter` describes semantics, not optimization policy. It must expose:

- architecture and tokenizer identity;
- tensor families, shapes, layer ownership, and weight tying;
- attention, normalization, RoPE, router, expert, and sampling semantics;
- legal capture points and reconstruction units;
- reference execution and trace generation;
- artifact packing names and runtime binding rules;
- applicable quality/task suites.

The first adapter is Qwen3 MoE. Later adapters should include other sparse MoE
families and dense Llama-like models. Dense adapters do not pretend to have
experts: their reconstruction unit can be a projection, block, channel group,
or other independently packable tensor region. The compiler state machine is
shared even when an adapter emits different candidate families.

### Target profile

The existing typed target profile should grow into a measured capability
contract containing:

- GPU family, OS/backend version, workgroup and feature limits;
- physical-memory and stable app-memory ceilings;
- supported codecs and consumer kernels;
- no-copy/mmap behavior and alignment;
- safe validation and inference residency;
- context/KV budget;
- thermal, energy, and sustained-throughput constraints;
- permitted execution schedules, including rollout in-flight depth.

The target selects execution and residency. It must not silently change model
quality. A packed weight family may be shared across compatible Apple devices,
while the sealed runtime plan can still differ between M5, A19 Pro, and later
GPUs.

### Quality gate

A quality gate is a versioned acceptance contract. It answers: "Is this lossy
artifact close enough to the declared control to ship for this use?" It is not
a training corpus, a benchmark score, or an optimizer hint.

A gate seals:

- the high-precision reference and shipping control;
- held-out corpus/task manifests and tokenizer hashes;
- required domains, tasks, sample counts, and sampling seeds;
- aggregate and worst-domain PPL limits;
- task-accuracy non-regression limits;
- confidence-interval requirements;
- behavioral diagnostics such as top-1/top-k agreement, KL, and logit cosine;
- chat-template, stop-token, tool-format, and deterministic-generation checks;
- the rule for missing, contaminated, or statistically underpowered evidence.

Calibration data may choose weights, bit widths, or anchors. It can never
satisfy the release quality gate. All gate inputs must be disjoint and sealed
before candidate selection.

The current `qx-quality` triangular gate is the starting implementation. It
compares FP16/BF16 reference, llama Q4_K_M, and QX on held-out WikiText-2,
Dolly, HumanEval, and GSM8K evidence. Its current acceptance rules include:

- at least 32,768 scored text positions per domain;
- at most +1% aggregate relative PPL versus llama;
- at most +2% relative PPL in the worst text domain;
- a confidence interval narrow enough to resolve a one-point PPL effect;
- HumanEval and GSM8K no more than two accuracy points below llama, including
  their paired confidence checks.

`qwen-general-v1` should be a named extension of this contract, adding Qwen's
declared general-purpose surface: chat/instruction following, code, math,
multilingual text, tool-call formatting, multi-turn state, and long-context
samples. Thresholds and corpus revisions live in the gate manifest rather than
in adapter code.

The current 343-token causal replays are compiler screening gates. They are
excellent for selecting exact anchors, but they are far too small and too
selection-adjacent to be the release quality gate.

## Compilation state machine

`qx compile-model` should execute these resumable stages:

1. **Inspect**: hash and validate model, tokenizer, template, and adapter.
2. **Establish controls**: record high-precision and shipping-control quality,
   memory, and performance.
3. **Seal datasets**: produce disjoint calibration, selection, and release-gate
   manifests.
4. **Capture**: collect adapter-defined activations/routes once and shard them
   into content-addressed evidence.
5. **Synthesize candidates**: generate legal precision, representation,
   fallback, and residency plans under the requested budget.
6. **Transform**: reconstruct or quantization-train independent units.
7. **Screen locally**: reject malformed, under-evidenced, or clearly dominated
   units without treating local metrics as promotion proof.
8. **Select causally**: replay the complete model, test bounded exact or
   higher-precision candidates, and checkpoint every pass.
9. **Pack**: emit one consumer-native mixed artifact with no diagnostic F32
   payloads.
10. **Validate jointly**: evaluate the complete combined policy; isolated layer
    passes do not authorize the artifact.
11. **Autotune target execution**: select kernels, storage ownership, cache,
    command-buffer topology, and rollout schedule using measured evidence.
12. **Run release gates**: quality, physical memory, performance, thermal,
    energy, and sustained execution.
13. **Seal or reject**: sign the artifact and evidence bundle, or produce a
    structured failure report with the next actionable constraint.

Every stage key includes all upstream fingerprints. A completed stage is
reused only on an exact key match. Outputs are written atomically; interruption
never converts a partial shard, layer, candidate pass, or artifact into a
resume boundary.

## Compiler data model

The first implementation should add versioned records resembling:

```rust
struct ModelCompileRequest {
    model: ModelSource,
    adapter_id: String,
    target: TargetProfile,
    workload: WorkloadProfile,
    memory_budget_bytes: u64,
    quality_gate_id: String,
}

struct ModelCompileState {
    request_fingerprint: Sha256,
    completed_stages: Vec<StageEvidence>,
    candidates: Vec<ModelCandidate>,
    selected_model_plan: Option<ModelPlan>,
    selected_execution_plan: Option<ExecutionPlan>,
}

struct ModelPlan {
    representations: Vec<TensorRegionPlan>,
    exact_exceptions: Vec<TensorRegionId>,
    causal_anchors: Vec<TensorRegionId>,
    memory: CompiledMemoryPlan,
    provenance: ModelPlanProvenance,
}
```

The serialized artifact should reference the selected model plan and execution
plan by fixed-width IDs. Runtime activation resolves those IDs once. Decode
must not parse manifests, select codecs, or make per-token planner decisions.

## Mapping onto the current repository

Use existing components rather than replacing them:

- `qx-model`: adapter-facing semantic model description and source identity;
- `qx-compiler`: compile request, stage DAG, candidate legality, and selection;
- `qx-format`: versioned model-plan, mixed-region, target, and provenance
  sections;
- `qx-quality`: named gate manifests and release acceptance;
- `qx-autotune`: measured target execution-plan selection;
- `qx-route-cache`: reference capture and causal replay backend;
- Qwen capture/training scripts: first adapter implementation, later moved
  behind stable stage interfaces.

The Qwen experiment already supplies working prototypes for capture, compact
QXDS evidence, Q2S4 reconstruction, conservative evidence fallbacks, compact
causal replay, exact-anchor selection, and resumable manifests. These should be
wrapped, not rewritten before the first compiler command works.

## Implementation sequence

### Phase 1: orchestration and provenance

1. Add versioned `ModelCompileRequest`, `ModelCompileState`, stage keys, and
   atomic stage-result records.
2. Add `qx compile-model` with `inspect`, `advance`, `status`, `resume`, and
   `explain` surfaces.
3. Wrap the current Qwen capture and layer trainer as adapter stages.
4. Import the active Qwen build directory instead of recapturing completed
   evidence.

### Phase 2: automatic causal planning

1. Rank candidates from route frequency, local error, representation cost, and
   previous causal evidence.
2. Make exact/higher-precision panels bounded and compiler-generated.
3. Write each causal pass immediately so interruption does not lose a panel.
4. Select the Pareto frontier by full-model quality recovery per added byte.
5. Require disjoint selection validation before adding an anchor to the plan.

### Phase 3: consumer-native artifact

1. Pack reconstructed Q2S4/Q3 weights directly rather than retaining F32
   diagnostic sidecars.
2. Copy or map original GGUF regions only for declared exact exceptions.
3. Seal tensor-region maps, hashes, target compatibility, and runtime bindings.
4. Load the mixed artifact through the native Metal consumer and compare it
   against validation scratch expansion.

### Phase 4: release gates and target sealing

1. Make named quality gates data-driven and versioned.
2. Run the complete combined policy through the release gate.
3. Measure installed bytes, mapped bytes, physical footprint, cache/KV/scratch,
   prefill, decode, thermal behavior, and energy.
4. Let `qx-autotune` seal the fastest plan satisfying every quality and memory
   constraint.

### Phase 5: additional model families

1. Extract Qwen-specific tensor naming and capture semantics into the first
   `ModelAdapter`.
2. Add a dense Llama-family adapter using tensor/block sensitivity rather than
   expert routing.
3. Add another sparse MoE adapter to prove that the interface is not
   Qwen-shaped.
4. Require each adapter to pass semantic parity, artifact round-trip, and one
   complete release gate before declaring support.

## Definition of done

The architecture is real when a clean workspace can compile a supported model
with one command; resume after interruption; explain every retained high-bit
region; reject contaminated or incomplete quality evidence; emit one native
mixed artifact; select a target runtime plan from measurements; and reproduce
the sealed quality, memory, and performance reports without manual command
construction.

For the current Qwen program, the immediate handoff into this compiler is:

- import all six completed evidence shards;
- import trained/reused layer manifests and the `3:82` causal anchor;
- continue remaining layer transformations under the extended evidence policy;
- rerun the five older manifests with extended evidence before final packing;
- perform complete 48-layer joint selection and release validation;
- pack and measure the first iPhone 17 Pro Max artifact.
