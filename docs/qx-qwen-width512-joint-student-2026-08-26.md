# Qwen3-30B-A3B width-512 joint expert student

Date: 2026-08-26  
Status: single-expert structural-memory smoke; not promotion evidence

## Result

QX can train a 512-neuron gate/up/down expert on normalized inputs captured at
the real llama.cpp MoE boundary and replace one 768-neuron expert during a
teacher-forced full-model replay. The strongest storage candidate uses Q2S4
gate/down weights and Q3 up weights, with the exact QX group-32 codecs included
in the training forward pass through a straight-through estimator.

On the untouched expanded WikiText-2 test replay, replacing layer 8 expert 38
produced:

| Metric | Q4 control | Width-512 mixed-bit student |
|---|---:|---:|
| Sequences | 8 | 8 |
| Scored tokens | 183 | 183 |
| Target-expert hits | 92 | 92 |
| Mean NLL | 3.253714 | 3.247323 |
| Relative perplexity delta | - | -0.6370% |
| Top-1 agreement | - | 95.08% |

The negative delta is not a claim that the student improves Qwen. It shows that
this one replacement is below the resolution of a small replay and, more
importantly, that a narrowed consumer-native expert can pass the initial
quality gate.

## Capacity projection

The exact storage ledger for a complete width-512 model is:

| Expert policy | Projected weights | + 4K KV allowance |
|---|---:|---:|
| gate/up/down Q2S4 | 6.433 GiB | 6.808 GiB |
| gate/down Q2S4, up Q3 | 7.183 GiB | 7.558 GiB |

Promoting one projection from Q2S4 to Q3 adds 805,306,368 bytes across 48 x 128
experts. The mixed policy therefore remains a genuine iPhone 17 Pro Max
capacity tier while buying materially better expert fidelity.

## What changed

- `qx-route-cache` can capture the normalized 2048-wide expert input together
  with post-SwiGLU activation and down output. Single-expert capture reads only
  the requested layer's route and activation buffers.
- `train_qwen_expert_student.py` maps only the selected BF16 gate/up/down
  tensors, chooses a width-512 initialization, and jointly distils all three
  projections on MPS or CPU.
- The trainer mirrors QX Q2S4 and Q3 group-32 scale rounding and supports
  projection-specific straight-through quantization-aware training.
- The llama.cpp diagnostic callback can inject a complete compact expert at the
  down-projection boundary for paired exact/student NLL measurement.

The injection sidecar contains dequantized F32 matrices. It validates model
semantics only; it is not the packed runtime format or a throughput path.

## Ablation

For the same 48-token, 34-hit replay:

| Student | Relative perplexity delta |
|---|---:|
| Width-512 float | +0.47% on the earlier 88-token replay |
| All-Q2S4 post-training quantization | +2.10% |
| All-Q2S4 QAT | +1.75% |
| Q2S4 gate/down + Q3 up QAT | -0.78% |

Expert-local held-out mean relative L2 follows the same ordering: 0.548 for
all-Q2S4 PTQ, 0.447 for all-Q2S4 QAT, and 0.417 for mixed-bit QAT. QAT is
necessary, and projection precision is a useful compiler specialization
dimension.

## Reproduction

Capture is produced by `qx-route-cache expert-distill-corpus`. A representative
training invocation is:

```sh
external/venv314/bin/python tools/train_qwen_expert_student.py \
  --model-root external/models/qwen3-30b-a3b-hf-ad44e777 \
  --train-trace external/results/qwen3-width512-layer8-expert38-train.jsonl \
  --heldout-trace external/results/qwen3-width512-layer8-expert38-heldout.jsonl \
  --layer 8 --expert 38 --width 512 \
  --epochs 200 --patience 30 --learning-rate 3e-5 \
  --qat-epochs 400 --qat-patience 80 --qat-learning-rate 2e-5 \
  --gate-codec q2s4 --up-codec q3 --down-codec q2s4 \
  --output external/results/qwen3-width512-layer8-expert38-upq3-qat-master.qxejs \
  --quantized-output external/results/qwen3-width512-layer8-expert38-upq3-qat.qxejs \
  --report external/results/qwen3-width512-layer8-expert38-upq3-qat-report.json
```

The expanded semantic replay uses `joint-expert-student-nll` with eight
sequences, 48 tokens per sequence, context 128, and the WikiText-2 test split.

## Promotion gates

This candidate must not become the canonical model plan until all of these pass:

1. capture a multi-domain routed corpus with substantially more hits per expert;
2. repeat across early/middle/late layers and common/rare experts;
3. inject many students simultaneously to measure accumulated NLL loss;
4. train and validate every expert, then emit real packed mixed-bit entries;
5. run device-resident iPhone memory, thermal, and tokens/second measurements.

The immediate next experiment is a stratified panel, not more epochs on expert
38. At minimum it should cover three layers and high/median/low-frequency
experts with a frozen quality threshold of less than 1% relative perplexity
increase per panel replay.
