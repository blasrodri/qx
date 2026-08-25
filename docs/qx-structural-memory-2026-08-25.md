# Structural memory promotion: asymmetric KV

## Result

The canonical M5 Pro greedy plan now stores K in FP32 and V in FP16. This
removes 12,582,912 bytes at context 512 while preserving the canonical
390-dispatch GPU-resident graph and exact greedy token history.

The compiler seals both KV precision and mmap/no-copy weight ownership into
the artifact. Runtime allocation, memory-budget enforcement, static audit,
and the live memory profiler now derive their byte counts from that sealed
plan.

## Memory

| Ownership | FP32 K/V | FP32 K + FP16 V | Delta |
|---|---:|---:|---:|
| Artifact mapping | 751,009,792 | 751,009,792 | 0 |
| KV cache | 50,331,648 | 37,748,736 | -12,582,912 |
| Arena | 406,784 | 406,784 | 0 |
| Known unique logical | 801,748,224 | 789,165,312 | -12,582,912 |
| Live Metal allocation | 802,373,632 | 789,790,720 | -12,582,912 |

The memory profiler separately reports artifact extent, unique tensor-payload
bytes, K and V cache bytes, arena ownership, current/peak physical footprint,
and Metal's aggregate allocation. File-backed no-copy weight views are counted
once in the unique logical ledger.

Greedy generator construction also no longer allocates the K-wide verifier.
Its 515,264 logical bytes are created only when a verifier or exit-probe API is
first used. Diagnostic replay captures remain independently lazy.

## Performance and correctness

The first candidate exposed an important runtime bug: any FP16 cache component
disabled split SIMD RoPE and selected a single-thread compatibility kernel.
That path fell from roughly 417 to 333 tok/s. It was not promoted.

The promoted implementation adds a split-SIMD FP32-K/FP16-V writer. A balanced
four-run AB/BA matrix used the 384-token general fixture plus 128 greedy decode
positions at context 512:

| Plan | Samples tok/s | Mean | Median |
|---|---|---:|---:|
| FP32 K/V | 390.73, 389.34, 390.12, 388.72 | 389.73 | 390.12 |
| FP32 K + FP16 V | 392.05, 393.90, 389.82, 392.99 | 392.19 | 392.99 |

The candidate was 0.63% faster by mean in this matrix. Both plans used two
command buffers and 390 dispatches per token. Every repetition produced token
history hash `19b4967c2513510e` and final token 34. KV hashes intentionally
differ because the persistent V representation differs.

The stage microbenchmark measured FP16-V accumulation at 43.00 microseconds
versus 43.37 microseconds for FP32 V, confirming that the narrowed consumer is
not adding a kernel-level penalty.

## Structural invariants

- New canonical artifacts default to mmap/no-copy weights.
- New canonical artifacts default to `Fp32KFp16V`.
- Artifact memory metadata must agree with context and KV precision or runtime
  activation fails.
- `ExecutionPlan`, `GpuRuntimeBuffers`, the compiler budget, and both memory
  reports use the same explicit KV byte plan.
- Greedy-only runtime features allocate no verifier storage.

## Next memory specialization

The next credible long-context step is a tiered value cache: recent V in FP16,
older V in block-scaled INT8, with K retained in FP32 initially. At context
4096, the promoted asymmetric plan saves about 96 MiB relative to the old FP32
cache; moving old V from FP16 to INT8 can save roughly another 48 MiB before
scale metadata. That work needs a matched-quality long-context gate and a
consumer that decodes INT8 without leaving the hierarchical attention path.
