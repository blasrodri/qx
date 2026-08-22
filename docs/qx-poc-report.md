# QX PoC evaluation report

Date: 2026-08-21  
Physical target: Apple M5 Pro  
Model: IBM Granite 3.0 1B-A400M Instruct  
Workload: batch 1, greedy decode, context 512

## Implemented

- Q2/Q3/Q4 groupwise quantization and pack/unpack tests.
- Real Granite routing trace ingestion and sensitivity measurement.
- Greedy Q2/Q3/Q4 allocation under an expert-byte budget.
- Target-specific `MetalTile32` packing serialized in the `.qx` directory.
- Fused Q2/Q3/Q4 Metal expert execution and one-command-buffer weighted top-8
  accumulation.
- Persistent Metal input/output scratch buffers and persistent attention score
  storage in the KV cache.
- Runtime timing, generation-logit capture, perplexity scoring, and artifact
  comparison in `qx-bench`.

## Measurements

The mixed artifact was compiled with:

```text
qx-compile external/models/granite-3.0-1b-a400m-instruct \
  --target apple-m5-pro \
  --trace external/results/granite-calibration.qxtrace \
  --memory-budget 620mb --context 512 --quality-loss 0.30 \
  --report external/results/qx-sensitivity-report-620mb.csv \
  -o external/models/granite-m5-pro-mixed-620mb.qx
```

| artifact | expert bytes | planned total bytes | selection |
|---|---:|---:|---|
| uniform Q4 | 679,477,248 | 706,908,166 | 1,536 Q4 |
| mixed | 552,665,088 | 580,096,006 | 244 Q2 / 802 Q3 / 490 Q4 |

The mixed artifact reduces expert storage by 18.6632% and planned memory by
17.9390%. Its weighted local relative-L2 error is 0.299689, below the 0.30
allocator target.

After one warmup generation of 16 tokens:

| artifact | TTFT | decode | decode tok/s |
|---|---:|---:|---:|
| uniform Q4 | 34.26 ms | 567.77 ms | 28.18 |
| mixed | 162.02 ms | — | 17.69* |

`*` The mixed tok/s value is from the post-comparison trace run; repeated
steady-state runs ranged from approximately 17.7 to 20.8 tok/s. The benchmark
harness reports p50/p95 with multiple iterations.

Uniform-Q4 versus mixed quality on the same `Hello` prompt:

```text
top-1 agreement:   0.000000
mean logit cosine: 0.509673
KL baseline||cand: 13.077152
uniform text:      Hellohttps, how are you? ...
mixed text:        Helloitchen ...
```

On the first 512 tokens of `external/data/ptb.test.txt`:

```text
uniform Q4 PPL: 30.956153
mixed PPL:      38.953792
delta:          +7.997639 (+25.84%)
```

The target packing itself is validated separately: row-major Q4 and
MetalTile32 Q4 produced 1.000000 top-1 agreement, 1.000000 mean logit cosine,
and zero KL on the same generation. The mixed quality loss is therefore not a
packing-layout serialization bug.

For context, the existing llama.cpp Metal fixture reports 254.75 tok/s and
PPL 16.62 for Q4_K_M, and 234.59 tok/s and PPL 19.89 for Q3_K_M. Those runs
remain separate baseline fixtures and are not claimed as a perfectly matched
benchmark because their prompt/benchmark harness differs from the QX command.

## Gate conclusion

- Gate 1: pass.
- Gate 2: local allocator/storage target passes; matched end-to-end quality
  requirement fails.
- Gate 3: not demonstrated. Kernel numerical smoke tests pass, but a valid
  identical-Q4 generic-vs-specialized end-to-end comparison is still absent.
- Gate 4: fail. The mixed artifact is smaller, but it is slower and materially
  worse in measured quality than uniform Q4.

The PoC has answered the thesis question for this implementation and fixture:
the compiler can produce target-packed mixed artifacts and save bytes, but the
current allocation/error model does not preserve downstream Granite quality,
and the specialized runtime is not yet competitive with llama.cpp Metal.
