# Payload-contiguous Q3-112 vector decoder

Date: 2026-08-25  
Target: Apple M5 Pro  
Model: IBM Granite 3.0 1B-A400M Instruct  
Workload: batch-1 greedy, context 512, 384-token prefill, 128-token decode

## Result

The 40 MiB mixed-bit capacity tier now stores every selected Q3 expert tensor
as exact 112-byte blocks and consumes those blocks without expansion:

```text
Q3-112 block (256 weights)
  bytes  0..95   eight group-32 code payloads
  bytes 96..111  eight FP16 scales
```

Each SIMD lane performs one unaligned four-byte packed load, masks the valid
24 bits, extracts eight Q3 codes with two vector shifts, and evaluates two
`float4` dot products. The fourth loaded byte is safe lookahead and is masked;
the artifact remains exactly three bits per weight plus one FP16 scale per 32
weights. Gate/up and down retain one mixed Q3/Q4 dispatch each.

| Metric | retained general extractor | Q3-112 vector | change |
|---|---:|---:|---:|
| Throughput mean | 354.44 tok/s | 379.66 tok/s | +7.11% |
| GPU mean | 2.818 ms/token | 2.631 ms/token | -0.187 ms (-6.63%) |
| Artifact bytes | 709,001,216 | 709,001,216 | unchanged |
| Unique weight payload | 708,892,352 | 708,892,352 | unchanged |
| Dispatches/token | 390 | 390 | unchanged |

The retained extractor numbers are the three-run results committed with the
original 40 MiB tier. The final same-session matrix measured canonical Q4 at
409.78 tok/s and Q3-112 at 379.66 tok/s, leaving this smaller tier 7.35% behind
canonical speed. Q3-112 runs were tightly grouped at 377.54--381.61 tok/s.

An additional same-binary legacy split-Q3 control averaged 374.03 tok/s, but
one of its three runs was a low 363.53 tok/s outlier. Its other two runs were
378.68 and 379.89 tok/s. Therefore scale colocation is not credited with a
material speedup: the packed vector decoder is the demonstrated gain, while
Q3-112 is the canonical no-copy storage form that keeps payload and scales in
one mapped range.

## Quality

The final vector accumulation was rescored on the two disjoint 1,023-position
PTB windows used to promote the 40 MiB capacity tier.

| Window | canonical Q4 NLL | Q3-112 NLL | PPL change |
|---|---:|---:|---:|
| A | 3.102323 | 3.105383 | +0.3065% |
| B | 3.384970 | 3.398301 | +1.3420% |
| Aggregate | - | - | +0.8229% |

This remains inside the existing gate of at most 1% aggregate and 2% for any
individual window. The canonical fixture still ends at token ID 26684, matching
the previous mixed artifact.

## Structural implementation

- `PackingLayout::Q3Superblock112` is an artifact-level specialization with an
  exact byte-count contract and round-trip coverage.
- The compiler emits Q3-112 automatically for expertwise Q3 tensors when the
  requested expert layout is `superblock-144`; Q4 tensors stay Q4-144.
- Artifact scale and data directory views alias one fused mapped range. Runtime
  slabs therefore remain no-copy and do not create an expanded GPU buffer.
- Metal selects separate Q3-112 or legacy-split pipeline functions at load
  time. There is no representation branch in the hot Q3 dot-product path.
- Legacy split-Q3 artifacts remain readable and also receive the packed vector
  decoder improvement.

## Reproduce

```bash
target/release/qx-compile \
  /Users/blas/projects/qx/external/models/granite-3.0-1b-a400m-instruct \
  --target apple-m5-pro \
  --trace /Users/blas/projects/qx/external/results/granite-calibration.qxtrace \
  --memory-budget 900mb --context 512 --quality-loss 1.0 \
  --expertwise-q3 --expertwise-q3-savings 40mb \
  --expert-layout superblock-144 --dense-layout superblock-144 \
  --command-buffers-per-token 2 --rollout-depth 2 \
  --kv-precision fp32-k-fp16-v --weight-storage mmap-no-copy \
  --plan-id granite-m5-mixed-q3-superblock112-s40m \
  --output /tmp/qx-mixed-q3-superblock112-s40m.qx

target/release/qx-bench \
  /tmp/qx-mixed-q3-superblock112-s40m.qx \
  --v2-token --fixture benchmarks/fixtures/general.json --context 512
```

Machine-readable evidence is in
`benchmarks/results/q3-superblock112-2026-08-25.json`.
