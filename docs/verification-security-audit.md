# QX verification security audit

Date: 2026-09-15

## Verdict

The complete Granite inference program now has an exact hierarchical verifier,
but the protocol is **not ready for automatic settlement or high-throughput
permissionless production inference**.
A real runtime-format job root descends through the token and checkpoint trees,
binds both dispute-local operator traces to the agreed checkpoint and each
party's endpoint, and lets SBF convict the incorrect state update using the
same 256-byte residual commitment emitted by normal execution.

The job, token, checkpoint, macro, micro, and terminal layers are connected.
The on-chain bridge authenticates the isolated Granite macro, its registered
instruction, both rolling 16-slot workspace roots, its K/V boundary, and the
fixed-length micro program. Objective terminals cover embedding, RMSNorm,
dense projections, RoPE, attention, Top-K, dynamic MoE, residual, logits, K/V
append, and argmax. An artifact-native CPU verifier independently reproduces
the slashable Metal path bit-for-bit through the final receipt root.

The implementation now persists compact trace openings, rejects results whose
witness does not reconstruct the committed root, independently replays
selected jobs, reserves collateral per registered job, escrows payment and an
equal challenger bond, and settles either outcome exactly once. Gateway
objects are immutable and have an explicit retention deadline. Epoch entropy
waits 32 slots for rooting and has a permissionless expired state rather than
a selectable fallback seed.

The latest real-artifact interval-2 run measured **2.572%** verifier overhead
including witness serialization, below the 5% miner gate, with exact output
agreement. Independent implementation review and additional hardware/OS
parity remain deployment gates; they cannot be self-certified by this codebase.

## What is implemented and checked

| Property | Evidence | Status |
|---|---|---|
| Exact input is bound | the input hash is a position-bound token Merkle root; prompt tokens open against it and generated inputs must equal the preceding authenticated prediction | Implemented on SBF; durable availability is outside this milestone |
| No miner signature per result | unsigned result leaf is attributed through one miner-signed epoch root | Implemented |
| Job authorization | requester Ed25519 signature over the immutable job hash is checked only when challenged | Implemented on SBF |
| Model and execution semantics | job binds the exact model registry; registry stores model ID, weights root, execution-spec hash, and a tree of checkpoint-segment program descriptors | Implemented on SBF |
| Sampling cannot be disabled | host and SBF reject probabilities below 10,000 ppm or above 1,000,000 ppm | Implemented; 1% is a provisional policy |
| Epoch/root grinding | one monotonically increasing epoch per miner identity; job binds its assigned epoch | Implemented for duplicate submissions |
| Entropy choice | seed uses the first produced Solana block strictly after epoch close, after 32 confirmation slots | Implemented; an aged-out successor makes the epoch terminally expired |
| Compact proof availability | QXW1 stores token/checkpoint preimages, validates every root, and exposes bounded Merkle openings | Implemented; 1,403.7 bytes/executed token at interval 2 |
| Economic settlement | registration locks exposure, job holds payment, challenge holds equal bond, and settlement is one-shot | Implemented and bank-level tested on SBF |
| Runtime job root is provable | canonical job genesis, compact streaming root, and exact trace length | Implemented and regression-tested |
| First-token/checkpoint faults are bisectable | canonical genesis leaves exist in both job and token trees | Implemented in commitments |
| Trace shape cannot be shortened | SBF enforces `trace_length = input_count + output_count` | Implemented and tested |
| Token tree shape is committed | `TokenResultV1` commits its checkpoint leaf count | Implemented in the runtime commitment |
| Token/checkpoint descent | exact token and checkpoint preimages are authenticated; model-fixed counts/layers prevent challenger-chosen geometry | Implemented and tested on SBF, including the genesis segment |
| Checkpoint-to-operator binding | both operator traces must open the agreed start checkpoint and their respective end checkpoint under `QX:OPERATOR:V1` | Implemented and tested on SBF |
| Verification opcode is registered | the disputed segment descriptor authenticates its own instruction root and length; the terminal opcode opens against that root | Implemented on SBF |
| Chained objective scalar update | instruction, runtime-format residual chunk, old/new root, token, segment, PC, and unchanged KV root are bound | Implemented end to end on SBF |
| Cross-chunk scalar update | operand subsets are authenticated over multiple bounded transactions and tied to one dispute/state/instruction; destination opening derives the post-root | Implemented on SBF; 10,365 CU for two source chunks plus destination |
| Chained authenticated Q4 update | registered instruction selects an exact model-weight leaf; its 26-level proof is staged and the VM memory/root update is objective | Implemented end to end on SBF; 11,885 CU |
| Cross-chunk Q4 update | weight, activation, and accumulator are independently authenticated; a separate destination opening derives the post-root | Implemented on SBF; 18,473 CU across four bounded instructions |
| Raw FP32 weight update | one finite value is selected from an authenticated canonical raw-weight leaf and combined with staged operands | Implemented on SBF for Granite norm tensors; 15,014 CU |
| Structured KV append | proves the exact token slot was empty, preserves the layer frontier, and updates the authenticated global layer root | Implemented on SBF and bound to the Granite RoPE/K/V macro output |
| Pairwise greedy argmax | authenticated `[logit, token]` records select the higher score, exact ties select the lower token ID, and both fields are copied for tournament reduction | Implemented on SBF and reached from an equal-checkpoint prediction dispute |
| Cross-chunk argmax | two source records are staged independently and the destination can occupy a third chunk; the last reduction binds the selected token into the output state | Implemented on SBF; 10,554 CU |
| Final-head output binding | shared final checkpoint starts both registered traces; distinct output-state hashes bind each trace to its claimed prediction | Implemented end to end on SBF |
| Objective arithmetic library | Q4 FMA plus fixed add/mul/div/exp/rsqrt/sin/cos/reduction primitives | Implemented and wired through fixed-length Granite micro traces |
| Granite macro schedule | 14 steps per layer, 29 for the first selected interval including embedding, 28 thereafter, and three for the final head; exact tensor map validated against the 1,706-entry production artifact | Implemented and registered |
| Dispute-only macro trace | real runtime captures normalized/Q/K/V/rotated-QK/attention/router/routing/MoE/residual roots, the derived K/V leaf, and final hidden/logits/winner in compiler order; a 16-slot Merkle workspace retains every live root | Implemented and complete-model tested |
| Macro boundary bridge | isolated instruction, pre/post workspace roots, changed slot, and K/V transition are authenticated for both parties | Implemented on SBF; 193,264 CU with software BLAKE3 |
| Granite micro terminals | embedding, RMSNorm, dense, RoPE, attention, Top-K, MoE, residual, logits, K/V, and argmax have objective terminal rules | Implemented on SBF; full branch-by-branch adversarial transaction coverage remains a production gate |
| Cross-token continuity | generated input is the prior agreed prediction and each token's genesis K/V root is the prior authenticated final K/V root | Implemented on SBF |
| Devnet deployability | SBF artifact uses supported syscalls and packet-safe instructions | Program and canonical model registry deployed; 10/10 public-devnet probes and one block-seeded gateway epoch completed |
| CPU verified generation | complete checkpoint+artifact execution emits a nonzero receipt root | Implemented and complete-model tested; a missing `begin_token` call was fixed by this audit |
| Turn liveness | every unilateral move has a deadline and either non-responder loses | Implemented and tested |

Two bugs found by this audit were deployment-critical:

1. the runtime used the `QX:JOB:V1` tree domain while on-chain bisection used a
   synthetic dispute domain; and
2. the runtime published a fixed-capacity streaming root while on-chain proofs
   interpreted the committed length as a compact tree.

Both are fixed, with an actual `TokenResultV1` job-tree SBF bisection test.
The execution-spec digest now binds genesis leaves, token checkpoint counts,
and compact-root semantics so old and new interpretations cannot share a
registered spec.

## Cost at massive scale

The registered interval-2 SIMD verifier measured **2.572%** normal-path
overhead on the tested M5 Pro (114.895 ms verified versus 112.014 ms baseline,
nine alternating repeats), including QXW1 serialization. At the configured 5%
sampling rate, one auditor replaying the same verified path adds about 5.129%
of baseline compute. The
approximate total network compute overhead is therefore:

```text
0.02572 + 0.05 * 1.02572 = 7.7006%
```

This excludes duplicate auditors, data distribution, failed jobs, and actual
dispute transactions. Ordinary miners do not generate micro traces; the wider
12-register SIMD state and its Merkle proofs are paid only after disagreement.
For packet safety, each party authenticates the output opening and two trace
endpoints in three dispute-bound staging transactions before a small admission
transaction. The maximum instruction payload is 640 bytes before the Solana
transaction envelope. Measured SBF cost is 216,566 CU for the challenger
sequence and 216,554 CU for the miner sequence, each spread across four
transactions.

The current live three-leaf synthetic SBF dispute consumes 159,352 CU. Epoch
entropy finalization consumes 14,285 CU once per epoch; opening the challenged
job costs 8,894 CU. A two-party segment-program/operator-boundary submission
costs 87,209 CU, the runtime-state-chunk scalar terminal costs 61,033 CU, the
chained Q4 terminal costs 68,631 CU, structured KV append costs 114,436 CU,
and an intermediate pairwise argmax costs 61,436 CU. Submitting both sides of
the final-head boundary costs 107,227 CU; its final argmax costs 55,075 CU.
These are rare-dispute costs. Software BLAKE3 avoids a testnet-inactive syscall
while preserving the low normal-inference commitment path.
Larger traces add a
challenger/miner midpoint pair for every binary-search level (10,332 CU in the
three-leaf test). These costs occur only after a sampled replay disagrees.

Sampling is detection, not certainty. With one fully available auditor:

| Scenario | Detection probability | Replay overhead | Break-even collateral |
|---|---:|---:|---:|
| One targeted cheat, 1% sampling | 1% | 1% | 99× cheating gain |
| One targeted cheat, 1%, 4× safety factor | 1% | 1% | 396× cheating gain |
| Same, auditor available 90% | 0.9% effective | 1% scheduled | 440.445× cheating gain |
| 100 independent cheats, 1% sampling | 63.397% at least one detection | 1% | exposure still must cover the whole unsettled strategy |

The `qx-verify sampling-cost` command calculates these values. Audit
probability can only be reduced safely when locked collateral rises, maximum
unsettled exposure falls, or multiple independent detection mechanisms are
added. A low sampling rate never gives cryptographic certainty for a targeted
high-value job.

## What remains before confidence

### Completed: full Granite trace hierarchy

The job→token→checkpoint→segment-program→operator→micro chain plus scalar,
authenticated-Q4, cross-chunk scalar, and KV append terminals is implemented. Each checkpoint
interval now authenticates its own program root, so layer-specific Granite
weights cannot be swapped through one generic program. The deterministic
macro compiler now emits RMSNorm, Q/K/V/O/router projection, RoPE, KV append,
attention, Top-K, MoE, residual, final-logit, and argmax steps with exact
artifact tensor IDs. A rolling 16-slot workspace binds every simultaneously
live macro value while retaining the existing checkpoint hashes as trace
endpoints. The on-chain macro bridge and objective terminal traces for
embedding, RMSNorm, dense projections, RoPE, attention, Top-K, dynamic MoE,
residual, final logits, and full-vocabulary argmax are complete. The canonical
pre-layer checkpoint is an on-chain-derivable token-start sentinel; embedding
is the first registered macro in segment one. The SBF hierarchy test exercises
a disagreement in the first checkpoint interval.

The final-head path and token-bound output state are inside the chained dispute
path, including the real LM head and full-vocabulary argmax tournament.

### Completed: bind K/V root production

The structured witness now proves that a new key/value leaf is appended at
exactly `(layer, token)`, preserves the old frontier, and derives the committed
global root. The registered Granite macro and micro traces derive the
authenticated K/V vector roots from Q/K/V projection and RoPE outputs.

### Completed: collateral and settlement

The program reserves miner collateral at job registration, prevents withdrawal
of locked exposure, escrows the requester payment, requires an equal challenger
bond, and uses the registered job account as the one-shot settlement record.
SBF balance tests cover challenger victory, undisputed payment, collateral
release, and replayed-settlement rejection.

### Completed for the gateway-backed testnet profile: input and witness availability

`JobEnvelopeV1` binds the exact input. The gateway stores immutable input,
output, result and QXW1 witness objects, refuses incomplete witnesses, and
publishes a retention deadline covering the challenge window plus 64 response
windows. Once opened, failure to reveal is handled by the existing unilateral
turn timeout. Production should run more than one gateway replica; scaling is
outside this change by explicit scope.

### Completed: entropy recovery and fork policy

Anyone can finalize the unique first successor only after 32 additional slots.
If it leaves SlotHashes first, anyone can mark the epoch expired; it cannot be
challenged or paid and must be retried. No fallback hash exists.

### P1: independent implementation parity

Run golden inference traces on every supported Apple GPU family and OS/compiler
combination, plus the CPU oracle. Pin or register all compiler/metallib inputs.
A single-device repeatability result is not enough for slashing.

### P1: independent review and wider deployment matrix

Randomized fail-closed commitment sequences, malformed/truncated witness
corpora, duplicate settlement, both timeout parties, non-finite arithmetic,
real hierarchy terminals and bank-level balances are automated. Before real
value, obtain an independent Solana/program review and reproduce golden traces
on every supported Apple GPU/OS/compiler tuple.

## Production gate

Devnet payment/slashing is enabled for testing. Do not enable real value until
an independent reviewer reproduces the real-artifact replay and SBF terminal
suite, reviews account aliasing and balance transitions, and signs off on the
deployment hash. Upgrade authority and production gateway scaling are
deliberately outside this change.

The ignored complete-model gate was also run locally against the installed
Granite checkpoint and Q4 artifact. It executes real model tensors, builds all
layer and final-head macro/micro traces, then corrupts every registered macro
output position in turn and requires the slashable state sequence to change.

## Live economic conviction

Commit `4d15a5a` was deployed to program
`QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB` at devnet slot 498734508. A
fresh live run then registered and funded a job, rooted epoch entropy, opened
and completed the false-final-token dispute, recorded the challenger winner,
and settled the bond/payment/collateral state in 15 confirmed transactions.
It used 322,620 simulated CU total (299,996 after setup), 130,000 lamports in
quoted transaction fees, and 10,480,040 lamports of recoverable account rent.
