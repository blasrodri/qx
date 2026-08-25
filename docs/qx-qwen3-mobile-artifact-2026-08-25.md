# Qwen3 mobile model artifact bring-up

Date: 2026-08-25  
Source: `Qwen/Qwen3-30B-A3B` at
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`  
Target: iPhone 17 Pro-class / Apple A19 Pro  
Status: source and Q2S4 capacity candidate complete; quality and execution not promoted

## Result

QX now has a hash-gated source checkpoint and a complete, reproducible
Q2S4-expert/Q4-shared capacity artifact for the unmodified Qwen shape. Q2S4 is
a symmetric four-level 2-bit codec with levels `[-1, -0.32, 0.32, 1]`. It uses
the same payload and FP16 scale bytes as Q2, while the offline compiler refines
each group scale with a least-squares/Lloyd iteration. The file is deliberately
marked as a capacity/quality candidate rather than an application model.

The source audit passed:

- all 16 BF16 shard SHA-256 values match the pinned upstream revision;
- all 18,867 expected tensors are present;
- no unexpected tensors, shape mismatches, or non-BF16 tensors were found;
- Qwen's 2,048-to-4,096 query expansion, Q/K normalization, normalized top-8
  routing, and untied LM head are sealed into the model descriptor.

The generated baseline is stored outside Git at:

```text
/Users/blas/projects/qx/external/models/
  qwen3-30b-a3b-a19pro-q2s4-capacity-candidate.qx
```

Its immutable identity and layout are:

| Field | Value |
|---|---:|
| File bytes | 9,927,917,568 (9.246 GiB) |
| SHA-256 | `7262a69d8230f0f78dcd12266035fedff5e9145dcace504c93fc27ccadef91c4` |
| Tensor entries | 12,723 |
| Native Q2S4 expert entries | 12,288 |
| Q4 shared/dense entries | 242 |
| Raw norm vectors | 193 |
| Gate/up expert entries | 6,144 |
| Down expert entries | 6,144 |
| Logical weight bytes | 9,927,286,784 |
| FP16 KV bytes at context 4K | 402,653,184 |
| Known logical total | 10,331,178,496 (9.622 GiB) |

The artifact inspector now reports the physical payload ledger rather than
leaving the total as a single number:

| Payload | Bytes | Share of logical weights |
|---|---:|---:|
| Q2S4 experts | 9,059,696,640 | 91.26% |
| Q4 shared/dense | 866,746,368 | 8.73% |
| Raw vectors | 843,776 | 0.01% |
| Directory, plan and leading alignment | 622,592 | file overhead |
| Trailing slab padding | 8,192 | file overhead |

This rules out metadata, tensor-directory packing and runtime scratch as useful
capacity targets. Sparse top-8 routing reduces the active computation but does
not remove any of the 128 experts per layer from resident model storage.

The artifact writer was changed to stream planned tensor slabs directly to the
file. It no longer constructs a complete payload blob and then a second
complete output buffer. This removes roughly 20 GiB of avoidable transient
copies for this build and makes conversion viable on the 24 GiB M5 Pro.

## Same-byte Q2 correction

The legacy Q2 encoder only emitted codes for -1, 0, and +1 even though two bits
carry four codes; its fourth code was unreachable. It also decoded the unused
code asymmetrically as +2. Q2S4 is a separately tagged codec so existing Q2
artifacts keep their original semantics.

On sampled early, middle, and final Qwen experts, the best four-level inner
ratio stayed within 0.30-0.32. The fixed 0.32 production codec produced:

| Sample | Legacy Q2 relative L2 | Q2S4 relative L2 |
|---|---:|---:|
| Layer 0, expert 0, gate/up/down | 0.659-0.662 | 0.336-0.339 |
| Layer 24, expert 64, gate/up/down oracle | 0.662-0.663 | 0.339-0.341 |
| Layer 47, expert 127, gate/up/down oracle | 0.662-0.667 | 0.340-0.357 |

This roughly halves weight reconstruction error at zero capacity cost. It is a
necessary correction, not proof that 2-bit end-to-end model quality is good.

## Structural target

The earlier first-step projection for a jointly trained width-672 student is:

| Payload | Bytes | GiB |
|---|---:|---:|
| Width-672 Q2 experts + Q4 shared | 8,794,402,816 | 8.190 |
| Plus FP16 KV at context 4K | 9,197,056,000 | 8.565 |

That is the first tier with credible room for runtime allocations and iOS
headroom. The unmodified-width baseline at 9.622 GiB before complete runtime
overhead remains too close to a 12 GB phone's practical application ceiling.

It is not, however, "way under" a 12 GiB physical ceiling. The stronger product
target is width 512, with width 576 as an intermediate quality bracket. These
projections preserve the current Q2S4 expert and Q4 dense codecs and scale only
the joint gate/up/down expert width:

| Expert width | Logical weights | Plus FP16 KV at 4K | Device headroom from 12 GiB |
|---:|---:|---:|---:|
| 768 (current) | 9.246 GiB | 9.621 GiB | 2.379 GiB |
| 672 | 8.191 GiB | 8.566 GiB | 3.434 GiB |
| 576 | 7.136 GiB | 7.511 GiB | 4.489 GiB |
| 512 | 6.433 GiB | 6.808 GiB | 5.192 GiB |

Width 512 removes one third of every expert's intermediate neurons and cuts the
complete weight payload by 3.020 GB without relying on expert paging. It is a
new trained model, not a file-format conversion: gate, up and down must be
jointly distilled or fine-tuned, then the resulting student must pass the
sealed perplexity and task gates before QX emits its deployable artifact.

## Why this is not yet the release model

The improved codec still has approximately 0.34 relative weight L2 error. That
metric is not equivalent to NLL, and no teacher-forced perplexity or task suite
has promoted it. The artifact therefore seals zero expected dispatches and
Granite Runtime V2 explicitly rejects its Qwen topology.

Promotion still requires:

1. joint gate/up/down structured distillation or quantization-aware training;
2. a width-672 model whose aggregate and worst-domain relative PPL deltas pass
   the 1% / 2% gates;
3. Qwen semantic parity for Q/K norm, RoPE, GQA, router normalization, untied
   logits, tokenizer, and chat template;
4. consumer-native Q2S4/Q3 mixed-bit Metal kernels at Qwen shapes;
5. measured iPhone physical footprint, jetsam, thermal, energy, and sustained
   throughput gates.

Do not distribute the capacity candidate as a user-facing model.

## Reproduce

```bash
target/release/qx-inspect \
  --model /Users/blas/projects/qx/external/models/qwen3-30b-a3b-hf-ad44e777 \
  --reference-manifest benchmarks/qwen3-30b-a3b-bf16-reference.json \
  --audit-output benchmarks/results/qwen3-30b-a3b-bf16-source-audit-2026-08-25.json

target/release/qx-compile-qwen \
  /Users/blas/projects/qx/external/models/qwen3-30b-a3b-hf-ad44e777 \
  --source-audit benchmarks/results/qwen3-30b-a3b-bf16-source-audit-2026-08-25.json \
  --expert-bits q2_s4 --context 4096 \
  --plan-id qwen3-a19pro-q2s4-capacity-candidate \
  --output /Users/blas/projects/qx/external/models/qwen3-30b-a3b-a19pro-q2s4-capacity-candidate.qx

target/release/qx-inspect \
  --artifact /Users/blas/projects/qx/external/models/qwen3-30b-a3b-a19pro-q2s4-capacity-candidate.qx \
  --audit-output benchmarks/results/qwen3-30b-a3b-a19pro-q2s4-capacity-artifact-2026-08-25.json
```
