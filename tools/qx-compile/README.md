# Cross-expert exact-Q4 candidate

`qx-compile --expert-layout cross-expert-tile8 --group-size 32` emits a
lossless structural candidate for dynamic top-8 MoE expert weights.

The physical stream is ordered as:

```text
[row tile][group-32 block][row in tile][expert][16 Q4 codes][FP16 scale]
```

Each layer/tensor family shares one stream across its experts. This changes
selected-expert traversal: an implementation can hold one activation group
while walking the selected expert lanes. The artifact execution-plan manifest
records `expert_traversal.kind = "cross_expert_tile"`, `tile_rows = 8`,
`expert_count = 32`, and `active_k = 8`.

The packer copies existing Q4 codes and FP16 scales exactly. The directory
still contains one logical entry per expert, and the format reader can recover
each expert's normal row-major quantized tensor from the shared stream.

This candidate is format/compiler/reference support only. Existing runtime and
Metal consumers do not select this layout until a corresponding traversal
kernel is added outside this scoped change.

Suggested artifact build:

```sh
cargo run -p qx-compile --release -- \
  /path/to/granite-model \
  --target apple-m5-pro \
  --memory-budget 16gb \
  --expert-layout cross-expert-tile8 \
  --group-size 32 \
  --output /tmp/granite-cross-expert-tile8.qx
```

For the existing runtime benchmark harness, use the normal artifact benchmark
command with the generated artifact after traversal support is available; the
current runtime is intentionally a control and must not be used as evidence
for a speedup from this representation.
