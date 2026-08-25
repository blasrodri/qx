# Qwen3 mobile model artifact bring-up

Date: 2026-08-25  
Source: `Qwen/Qwen3-30B-A3B` at
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`  
Target: iPhone 17 Pro-class / Apple A19 Pro  
Status: source and capacity artifact complete; quality and execution not promoted

## Result

QX now has a hash-gated source checkpoint and a complete, reproducible
Q2-expert/Q4-shared capacity artifact for the unmodified Qwen shape. This is
the first complete Qwen `.qx` file, but it is deliberately marked as a capacity
baseline rather than an application model.

The source audit passed:

- all 16 BF16 shard SHA-256 values match the pinned upstream revision;
- all 18,867 expected tensors are present;
- no unexpected tensors, shape mismatches, or non-BF16 tensors were found;
- Qwen's 2,048-to-4,096 query expansion, Q/K normalization, normalized top-8
  routing, and untied LM head are sealed into the model descriptor.

The generated baseline is stored outside Git at:

```text
/Users/blas/projects/qx/external/models/
  qwen3-30b-a3b-a19pro-q2-capacity-baseline.qx
```

Its immutable identity and layout are:

| Field | Value |
|---|---:|
| File bytes | 9,927,917,568 (9.246 GiB) |
| SHA-256 | `a9ed2b8f13d9b3facead70207cdb735ab3519797e1807732a313b445528dd377` |
| Tensor entries | 12,723 |
| Native Q2 expert entries | 12,288 |
| Q4 shared/dense entries | 242 |
| Raw norm vectors | 193 |
| Gate/up expert entries | 6,144 |
| Down expert entries | 6,144 |
| Logical weight bytes | 9,927,286,784 |
| FP16 KV bytes at context 4K | 402,653,184 |
| Known logical total | 10,331,178,496 (9.622 GiB) |

The artifact writer was changed to stream planned tensor slabs directly to the
file. It no longer constructs a complete payload blob and then a second
complete output buffer. This removes roughly 20 GiB of avoidable transient
copies for this build and makes conversion viable on the 24 GiB M5 Pro.

## Structural target

The audited projection for a jointly trained width-672 student is:

| Payload | Bytes | GiB |
|---|---:|---:|
| Width-672 Q2 experts + Q4 shared | 8,794,402,816 | 8.190 |
| Plus FP16 KV at context 4K | 9,197,056,000 | 8.565 |

That is the first tier with credible room for runtime allocations and iOS
headroom. The unmodified-width baseline at 9.622 GiB before complete runtime
overhead remains too close to a 12 GB phone's practical application ceiling.

## Why this is not yet the release model

The sampled BF16 expert has approximately 0.66 relative weight L2 error under
plain group-32 symmetric Q2. That metric is not equivalent to NLL, but it is
far too large to assume acceptable model quality. The artifact therefore seals
zero expected dispatches and Granite Runtime V2 explicitly rejects its Qwen
topology.

Promotion still requires:

1. joint gate/up/down structured distillation or quantization-aware training;
2. a width-672 model whose aggregate and worst-domain relative PPL deltas pass
   the 1% / 2% gates;
3. Qwen semantic parity for Q/K norm, RoPE, GQA, router normalization, untied
   logits, tokenizer, and chat template;
4. consumer-native Q2/Q3 mixed-bit Metal kernels at Qwen shapes;
5. measured iPhone physical footprint, jetsam, thermal, energy, and sustained
   throughput gates.

Do not distribute the capacity baseline as a user-facing model.

## Reproduce

```bash
target/release/qx-inspect \
  --model /Users/blas/projects/qx/external/models/qwen3-30b-a3b-hf-ad44e777 \
  --reference-manifest benchmarks/qwen3-30b-a3b-bf16-reference.json \
  --audit-output benchmarks/results/qwen3-30b-a3b-bf16-source-audit-2026-08-25.json

target/release/qx-compile-qwen \
  /Users/blas/projects/qx/external/models/qwen3-30b-a3b-hf-ad44e777 \
  --source-audit benchmarks/results/qwen3-30b-a3b-bf16-source-audit-2026-08-25.json \
  --expert-bits q2 --context 4096 \
  --plan-id qwen3-30b-a3b-a19pro-q2-capacity-baseline \
  --output /Users/blas/projects/qx/external/models/qwen3-30b-a3b-a19pro-q2-capacity-baseline.qx

target/release/qx-inspect \
  --artifact /Users/blas/projects/qx/external/models/qwen3-30b-a3b-a19pro-q2-capacity-baseline.qx \
  --audit-output benchmarks/results/qwen3-30b-a3b-a19pro-q2-capacity-artifact-2026-08-25.json
```
