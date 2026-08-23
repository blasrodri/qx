# Row-inline vec4 revalidation

The historical `row-inline-vec4-control-general-2026-08-22.json` reported
`261.14 tok/s`, so the existing vec4 consumer was re-run before designing a
new kernel.

## Results

All runs used release `qx-bench`, context 512, the general fixture, one
warmup, and 128 measured sequential tokens.

| Artifact / path | Wall ms/token | GPU ms/token | tok/s | CB/token |
|---|---:|---:|---:|---:|
| old 2-CB artifact, control | 4.513 | 4.234 | 221.59 | 2 |
| old 2-CB artifact, vec4 | 4.655 | 4.331 | 214.84 | 2 |
| current factorial artifact, vec4 | 4.878 | 4.385 | 205.02 | 1 |

The historical 261 tok/s result is not reproducible. The existing vec4 path
is slower than its same-artifact control in the revalidation and slower than
the current canonical row-inline baseline. It is therefore rejected as the
next performance lane; no vec4 promotion or redesign is justified from the
available evidence.

The discrepancy reinforces the benchmark rule already established by the
instrumentation work: a single complete-token result without a same-session
control and power/thermal metadata is not promotion evidence.

Result records:

- `benchmarks/results/revalidate-rowinline-vec4-current-2026-08-23.json`
- `benchmarks/results/revalidate-old-rowinline-control-2026-08-23.json`
- `benchmarks/results/revalidate-old-rowinline-vec4-2026-08-23.json`
