# QX production-preserving block probes

The `--v2-production-probe` benchmark adds a causal diagnostic for the
remaining QX/llama GPU gap. It is deliberately not a promotion benchmark and
does not alter the canonical runtime.

For one fixed post-prefill decode position it first runs the real QX graph and
captures two exact GPU tensors:

- the post-attention/O-projection output for every layer;
- the post-MoE output for every layer.

It then reruns the same explicit decode position with frozen top-8 routes and
replaces either block, or both blocks, with an identical GPU FP32 copy from the
captured tensor.
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
replace_attention_and_o_and_moe
```

The report computes average marginal effects and keeps the interaction term
explicit:

```text
A = ((T00 - T10) + (T01 - T11)) / 2
M = ((T00 - T01) + (T10 - T11)) / 2
I = (T00 - T10) - (T01 - T11)
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

General/context-512 diagnostic, 10 samples:

| Path | GPU µs/token |
| --- | ---: |
| canonical capture control | 4088.87 |
| attention + O replacement | 3629.87 |
| MoE replacement | 2140.75 |
| attention + O + MoE replacement | 1529.25 |

The resulting average marginal savings were approximately `0.535 ms` for
attention+O and `2.024 ms` for MoE, with a `-0.153 ms` interaction term under
this probe contract. These values are diagnostic and should be rerun with the
sealed benchmark sample count and paired fixtures before drawing competitive
conclusions. They are not additive operator costs.
