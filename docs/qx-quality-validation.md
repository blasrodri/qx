# QX triangular quality validation

The quality lane compares three traces generated from the same sealed token
corpus:

```text
high precision reference (FP16/BF16)
             /             \
       llama Q4_K_M       QX Q4
```

`qx-quality-corpus` writes a versioned `manifest.json` beside the processed
corpora. The manifest records the source revision, split, sampling seed,
tokenizer hash, raw and processed hashes, and the intended contiguous bootstrap
block size. Missing model or source inputs are reported together before any
corpus is generated.

Build a held-out manifest with at least 32,769 tokens for each text domain (so
there are 32,768 scored positions):

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

Compare traces using the manifest. The existing invocation remains valid when
`--manifest` is omitted; supplying it turns on provenance and processed-file
hash validation:

```bash
target/release/qx-quality compare \
  --corpus quality/granite-heldout/wikitext.tokens \
  --manifest quality/granite-heldout/manifest.json \
  --reference fp16.qxlogits \
  --llama llama-q4km.qxlogits \
  --candidate qx-q4.qxlogits \
  --bootstrap-iterations 10000 \
  --output wikitext-report.json
```

The report contains NLL/PPL for all three models, PPL deltas and contiguous
512-position block-bootstrap intervals for each pair, plus KL, logit cosine,
top-1, top-5, and top-10 agreement. The seed and iteration count are recorded
in every bootstrap result. HumanEval and GSM8K task accuracy remain separate
greedy task gates; their teacher-forced trace diagnostics can use the same
report format.

No quality claim is produced when the model, corpus, tokenizer, or trace files
are unavailable. The tools return an actionable missing-input error instead of
creating placeholder results.
