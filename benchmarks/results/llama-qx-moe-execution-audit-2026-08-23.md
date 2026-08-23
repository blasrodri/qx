# llama.cpp × QX MoE execution audit

Date: 2026-08-23  
QX source: `4828c21` (`Expose cross-expert MoE schedule as compiler candidate`)  
llama.cpp source: `17197474510622a3b4ea7d0909d70b606f542b96`  
Target: Apple M5 Pro, Granite 3.0 1B A400M, decode batch 1, context 512

This is a read-only source and execution audit. No QX runtime or Metal source was changed in this lane.

## Bottom line

The remaining QX gap is not explained by Q4 versus Q4_K alone. The relevant decode paths are materially different:

- llama’s batch-1 MoE path is `kernel_mul_mv_id_q4_K_f32`, a small matrix-vector kernel instantiated with `N_R0_Q4_K = 2` and `N_SG_Q4_K = 2`.
- llama’s 64×32 simdgroup-matrix MoE path is selected only when the token dimension is at least 32 (`ne21_mm_id_min = 32`); it is not the path used by single-token decode.
- QX’s canonical gate/up path uses 1,024 threadgroups per layer (8 experts × 128 row tiles), 256 threads per group, and stages the 1,024-element activation in every group.
- QX’s existing cross-expert candidate reduces the gate/up launch to 128 threadgroups and stages the activation once per route set, but serially visits all eight selected experts for each row tile.

The existing cross-expert candidate is therefore the right *experiment*, but not yet a win. A matched release run produced:

| Plan | Wall ms/token | GPU ms/token | Throughput | Dispatches/token |
|---|---:|---:|---:|---:|
| row-inline-1 control | 5.160 | 4.793 | 193.79 tok/s | 461 |
| cross-expert candidate | 5.318 | 5.021 | 188.05 tok/s | 461 |

This short run rejects promotion for now: the candidate is 3.0% slower in wall time and 4.8% slower in GPU time, with identical artifact size, allocation size, command-buffer count, and dispatch count. It does not prove the idea is useless; it identifies the tradeoff precisely: activation-staging savings do not compensate for serializing eight expert traversals in the current mapping.

## llama.cpp mapping

### Batch-1 path

Host dispatch selection is in:

`/Users/blas/projects/qx/external/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp`

- `ggml_metal_op_mul_mat_id()` at lines 2560–2710.
- The matrix-matrix branch is guarded by `ne21 >= 32` at lines 2600–2603.
- Otherwise the host selects `ggml_metal_library_get_pipeline_mul_mv_id()` and dispatches the matrix-vector kernel at lines 2638–2709.

The batch-1 kernel wrapper is:

`/Users/blas/projects/qx/external/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:11209-11270`

`kernel_mul_mv_id()`:

1. Uses the route ID directly to select `src0_cur = src0s + i02*args.nb02`.
2. Uses the selected expert slot to select the activation and output slice.
3. Calls the ordinary Q4_K matrix-vector implementation with `ne02 = 1`, `ne11 = 1`.
4. Launches one matrix-vector grid over the selected expert/output slice.

The Q4_K implementation is:

`/Users/blas/projects/qx/external/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:8498-8615`

`kernel_mul_mv_q4_K_f32_impl()`:

- `N_R0_Q4_K = 2` and `N_SG_Q4_K = 2` come from `ggml-metal-impl.h:54-55`.
- Each SIMD lane loads structured Q4_K blocks and accumulates `nr0` rows.
- The activation vector is read in the Q4_K block’s native access order (`y4`, four super-block chunks at a time), rather than copied into a 1,024-element threadgroup array for every row tile.
- Q4_K scale/min metadata is decoded in the same loop as the payload (`ggml-metal.metal:8555-8585`), so there is no separate scale stream in the execution path.

### Larger-batch path, not the immediate decode target

For `ne21 >= 32`, llama first compacts route IDs with:

- `ggml_metal_op_mul_mat_id()` lines 2608–2640;
- `kernel_mul_mm_id_map0<ne20>()` at `ggml-metal.metal:10664-10740`.

It then uses `kernel_mul_mm_id()` at `ggml-metal.metal:10757-10980`, with:

- `NR0 = 64`, `NR1 = 32`, `NK = 32`;
- threadgroup dimensions `128` threads and one SIMD group;
- route-compacted expert slices;
- 64×32 tiles and simdgroup matrix accumulation.

This is important for a future prefill/batched workload, but copying this path into batch-1 decode would be a category error. QX should maintain separate decode and prefill plan families.

## QX mapping

The canonical QX gate/up kernel is:

`/Users/blas/projects/qx/kernels/dynamic_moe.metal:695-737`

`qx_dynamic_q4_moe_gate_up_simd()`:

- 8 selected experts are flattened across `threadgroup_id`.
- Each 256-thread group owns eight output rows (`simdgroup` owns one row).
- The group copies the full 1,024-element input into `threadgroup float cached_input[1024]` at lines 713–717.
- Each selected expert is then traversed independently at lines 718–736.

The encoder is:

`/Users/blas/projects/qx/crates/qx-metal/src/lib.rs:1725-1949`

For the canonical row-inline path it dispatches `8 * (1024 / 8) = 1024` threadgroups with 256 threads each. The same method then dispatches SiLU, down, and accumulation as separate stages.

The cross-expert candidate is:

`/Users/blas/projects/qx/kernels/dynamic_moe.metal:248-337`

`qx_dynamic_q4_moe_gate_up_cross_expert_simd()`:

- dispatches 128 groups;
- stages the activation once per group;
- assigns one eight-row tile to each group;
- loops over all eight expert slots serially at lines 271–289.

That explains the measured result: it removes repeated activation staging and reduces the number of resident groups, but it replaces eight independent expert row traversals with an eight-iteration serial loop inside each row tile. On this M5 Pro shape, that exchange is currently unfavorable.

## What QX is leaving on the table

The highest-confidence opportunities are not another whole-expert permutation or another isolated Q4 row layout.

### 1. Decode kernel mapping is not yet matched to llama’s batch-1 structure

The most concrete difference is:

| Concern | llama decode | QX canonical |
|---|---|---|
| Expert selection | direct route-indexed base pointer | route-indexed base plus generic layout arithmetic |
| Output work | `nr0=2`, `nsg=2`, small matrix-vector groups | 8 rows/group, 256 threads/group |
| Activation handling | native Q4_K block-order reads | full 4 KiB threadgroup staging per group |
| Quant metadata | payload and Q4_K scale/min co-decoded | Q4 code stream plus separate FP16 scale lookup |
| Selected experts | independent grids | independent grids, unless cross candidate enabled |

This is enough to justify one narrowly scoped experiment: implement a QX decode-only Q4 control with llama-like `2 rows × 2 SIMD groups` geometry and direct per-expert row-base addressing, while retaining QX’s exact Q4 row-inline representation. Do not transplant Q4_K or the larger-batch matrix-matrix path. The experiment should answer whether the loss is primarily mapping/occupancy or representation/traffic.

### 2. Scale-stream separation is a credible representation cost

The QX row-inline representation stores 16 code bytes plus a separate FP16 scale per 32-value group. The canonical kernel performs a scale-buffer lookup for every group (`dynamic_moe.metal:728-732`). llama’s Q4_K block keeps scale/min metadata with the block and decodes it while consuming the payload (`ggml-metal.metal:8555-8585`).

The next representation experiment should therefore be a *QX-native fused-block control*: keep QX’s exact values and artifact semantics, but colocate the scale with the code bytes in the same block and use one vector/block load path. This is different from Q4_K and should be measured against row-inline-1 with the same full-model candidate gate.

### 3. Full route compaction is a prefill feature, not a decode fix

llama’s route-compaction map pass only serves its `ne21 >= 32` matrix-matrix branch. QX should not add a decode-time route-compaction dispatch merely because it exists in llama. For batch 1, the correct test is direct addressing and a smaller decode workgroup; for prefill, the 64×32 route-compacted plan should be evaluated separately.

### 4. Missing evidence: per-dispatch GPU timeline

The current QX benchmark reports complete-token GPU time, but not a timestamped live GPU interval for each MoE stage/kernel. Until that exists, it is impossible to tell whether the 2.0–2.7 ms live MoE marginal is dominated by Q4 arithmetic, expert-weight transactions, pipeline bubbles, or resource transitions. A Metal counter-sample-buffer or equivalent per-encoder/per-stage timing path is the highest-value instrumentation work before a third MoE geometry search.

## Recommended next experiment

Run this two-stage, exact-Q4 experiment in the compiler candidate system:

1. **Decode-direct QX kernel:** `2 rows × 2 SIMD groups`, one selected expert per grid slice, no full threadgroup activation staging; read QX row-inline blocks directly in the Q4 group order.
2. **Decode-direct + fused QX block:** same mapping, but codes and FP16 scale are fetched from one 18-byte logical block stream rather than separate payload/scale buffers.

Promotion gates:

- complete-token benchmark on all existing fixtures;
- long sequential parity through position 511;
- same artifact and memory accounting;
- live MoE differential, not isolated kernel time;
- compiler selection record showing the candidate is selected from measured complete-model evidence.

Interpretation:

- If direct mapping wins while separate-scale remains slow, mapping/occupancy is the main gap.
- If fused-block wins substantially, representation/transaction economics is the main gap.
- If neither wins, the remaining gap is likely broader QX graph scheduling or expert-weight residency; stop kernel guessing and add the per-dispatch GPU timeline.

Do not promote the current cross-expert candidate based on its activation-reuse rationale; the measured full-token result currently rejects it.

## Provenance

- QX control artifact: 751,009,792 bytes, SHA-256 `ec08132b08f7d9d330def113a82408fc922f9b1fd9ec6a1a12e8ac24fb798299`.
- QX cross-expert artifact: 751,009,792 bytes, SHA-256 `d79d26f2a19f8ea356af003e02f3309893fa210c908fa79970a51d97cfe3c674`.
- llama Granite Q4_K_M GGUF: 821,844,576 bytes, SHA-256 `46a2a757a579446d95d900085a5daaf59d78bb753302f656ec01a7192b4928a6`.
- Existing llama batch-1 benchmark record: `/Users/blas/projects/qx/external/results/llama-q4-batch1.json`, pinned build `1719747`, 128-token sample mean `254.745 tok/s` in that historical record.
- Fresh QX comparison commands used the release `target/release/qx-bench` with `--v2-token --context 512 --warmup 1 --iterations 3 --max-new-tokens 16 --ignore-eos`.
