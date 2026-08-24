# QX production-preserving block probes

The `--v2-production-probe` benchmark adds a causal diagnostic for the
remaining QX/llama GPU gap. It is deliberately not a promotion benchmark and
does not alter the canonical runtime.

For one fixed post-prefill decode position it first runs the real QX graph and
captures two exact GPU tensors:

- the post-attention/O-projection output for every layer;
- the post-MoE output for every layer.

It then reruns the same explicit decode position with frozen top-8 routes and
replaces either block with an identical GPU FP32 copy from the captured tensor.
The surrounding embedding, QKV, RoPE/KV, residual, router, final head, and
command-buffer path remain live. This is materially different from the older
no-op ablations: the downstream graph receives the exact production value.

Example:

```text
cargo run --release -p qx-bench -- \
  external/models/granite-m5-pro-q4-rowinline-unique-packet.qx \
  --v2-production-probe \
  --fixture benchmarks/fixtures/general.json \
  --context 512 --warmup 3 --iterations 20 \
  --result-json benchmarks/results/production-preserving-probe-general.json
```

The output reports:

```text
canonical_capture_control
replace_attention_and_o
replace_moe
```

The control and replacement each include one GPU copy at the probed boundary,
so the comparison does not hide copy/injection overhead. Routes are captured
once after the canonical capture and reused for the timed runs. The explicit
position overwrites the same KV slot on each run, avoiding a CPU KV snapshot
while keeping the prefill state fixed.

Interpretation is intentionally conservative. The saved GPU time is a
production-shaped upper bound on the work attributable to the replaced block,
not an additive operator budget. It should be used to decide whether a block
has at least `0.15–0.20 ms/token` of demonstrated end-to-end headroom before
implementing a new exact kernel or representation.

Initial general/context-512 diagnostic, 3 samples:

| Path | GPU µs/token |
| --- | ---: |
| canonical capture control | 4082.12 |
| attention + O replacement | 3603.50 |
| MoE replacement | 2134.63 |

The corresponding coarse marginal savings were approximately `0.479 ms` for
attention+O and `1.948 ms` for MoE under this probe contract. These values are
diagnostic and should be rerun with the sealed benchmark sample count and
paired fixtures before drawing competitive conclusions.
