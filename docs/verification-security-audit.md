# QX verification security audit

Date: 2026-09-11

## Verdict

The generic fraud-proof mechanism is now demonstrated end to end for an
in-place scalar transition, but the complete Granite inference program is
**not ready for automatic settlement or permissionless production inference**.
A real runtime-format job root descends through the token and checkpoint trees,
binds both dispute-local operator traces to the agreed checkpoint and each
party's endpoint, and lets SBF convict the incorrect state update using the
same 256-byte residual commitment emitted by normal execution.

The remaining boundary is model coverage rather than the bisection primitive:
the registered VM still needs authenticated Q4 weight loads, KV append, and
final-head/argmax transitions. Without those transition
families, the test proves the mechanism and one real commitment update, not the
entire model's `Execute(M, E, X)` claim.

## What is implemented and checked

| Property | Evidence | Status |
|---|---|---|
| Exact input is available to both executors | `JobEnvelopeV1` carries raw token IDs and validates their signed header hash | Implemented transport format; no durable data-availability layer |
| No miner signature per result | unsigned result leaf is attributed through one miner-signed epoch root | Implemented |
| Job authorization | requester Ed25519 signature over the immutable job hash is checked only when challenged | Implemented on SBF |
| Model and execution semantics | job binds the exact model registry; registry stores model ID, weights root, execution-spec hash, and verification-program root | Implemented on SBF |
| Sampling cannot be disabled | host and SBF reject probabilities below 10,000 ppm or above 1,000,000 ppm | Implemented; 1% is a provisional policy |
| Epoch/root grinding | one monotonically increasing epoch per miner identity; job binds its assigned epoch | Implemented for duplicate submissions |
| Entropy choice | seed uses the first produced Solana block strictly after epoch close | Implemented; timely finalization is still an operational assumption |
| Runtime job root is provable | canonical job genesis, compact streaming root, and exact trace length | Implemented and regression-tested |
| First-token/checkpoint faults are bisectable | canonical genesis leaves exist in both job and token trees | Implemented in commitments |
| Trace shape cannot be shortened | SBF enforces `trace_length = input_count + output_count` | Implemented and tested |
| Token tree shape is committed | `TokenResultV1` commits its checkpoint leaf count | Implemented in the runtime commitment |
| Token/checkpoint descent | exact token and checkpoint preimages are authenticated; model-fixed counts/layers prevent challenger-chosen geometry | Implemented and tested on SBF for non-genesis segments |
| Checkpoint-to-operator binding | both operator traces must open the agreed start checkpoint and their respective end checkpoint under `QX:OPERATOR:V1` | Implemented and tested on SBF |
| Verification opcode is registered | scalar terminal opcode is opened against the model's verification-program root | Implemented on SBF |
| Chained objective scalar update | instruction, runtime-format residual chunk, old/new root, token, segment, PC, and unchanged KV root are bound | Implemented end to end on SBF |
| Objective arithmetic library | Q4 FMA plus fixed add/mul/div/exp/rsqrt/reduction primitives | Implemented; not every family is wired into the chained VM |
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

Two independent 31-repeat fixed-math runs of the genesis-complete format add
**2.939–4.201%** to inference on the tested M5 Pro (mean **3.570%**). At the mandatory 1% sampling floor with one auditor, replay adds
1% of baseline inference work. If auditors use the same committed path, total
network compute is approximately:

```text
(1 + 0.03570) * (1 + 0.01) - 1 = 4.6057%
```

This excludes the one-time migration from a fast, non-consensus math profile
to the fixed consensus profile. The previous same-machine measurement for
that arithmetic migration plus commitments was 8.820%. It also excludes
duplicate auditors, data distribution, failed jobs, and dispute replay.

The current live three-leaf synthetic SBF dispute consumes 32,075 CU. Epoch
entropy finalization consumes 1,784 CU once per epoch; opening the challenged
job costs 8,397 CU. A two-party operator-boundary submission costs 7,656 CU,
and the runtime-state-chunk scalar terminal costs 4,342 CU. Larger traces add a
challenger/miner midpoint pair for every binary-search level (10,329 CU in the
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

### P0: complete registered VM lowering

The job→token→checkpoint→operator→scalar chain is implemented. Complete the
registered Granite program with terminal witnesses for authenticated Q4 weight
loads/FMA, cross-chunk moves, attention/RoPE/RMSNorm/router/MoE state updates,
and the final head. The canonical pre-layer checkpoint now binds the embedding
residual and prior-token global KV root, and the SBF hierarchy test exercises
a disagreement in the first checkpoint interval.

The final-head argmax and its lowest-token-ID tie rule must be inside this same
chained path, not merely stored as `predicted_token`. Equal checkpoint roots
with different predicted tokens are therefore not yet settlement-safe.

### P0: KV append proof

The verifier needs a structured witness proving that a new key/value leaf was
appended at exactly `(layer, token)`, the old frontier was preserved, and the
per-layer root update produced the committed global KV root. A plain hash
comparison is insufficient.

### P0: collateral and settlement

The program must reserve miner collateral for every unsettled job, prevent the
same collateral from backing concurrent exposure twice, require a challenger
bond, make each dispute unique for `(epoch, result_index)`, and transfer funds
exactly once after arithmetic resolution or timeout. Until this exists,
`slashable_exposure` is only signed metadata.

### P0: input data availability

`JobEnvelopeV1` answers which input is correct, but not whether auditors can
retrieve it throughout the challenge window. Settlement must require a
durable availability commitment/retention rule, or unavailability must cause
the responsible party to lose. Otherwise a miner/requester can make selected
jobs unauditable.

### P1: entropy recovery and fork policy

Anyone can finalize entropy, but the first successor block leaves Solana's
slot-hash history quickly. Define an expiry outcome if nobody finalizes in
time, and specify commitment/finality depth and long-fork behavior. The miner
must never choose among multiple valid epoch seeds.

### P1: independent implementation parity

Run golden inference traces on every supported Apple GPU family and OS/compiler
combination, plus the CPU oracle. Pin or register all compiler/metallib inputs.
A single-device repeatability result is not enough for slashing.

### P1: adversarial testing

Add randomized state-machine sequences, malformed proof/length corpora,
duplicate-dispute tests, maximum-depth job/token/operator traces, every timeout
phase, non-finite arithmetic cases, and bank-level settlement invariants.
Then obtain an independent Solana/program review before enabling value.

## Production gate

Do not enable automatic payment or slashing until all P0 items pass an SBF
test that begins with a real `ExecutionCommitterV1` root, corrupts at least one
operation in each operation family (dense, attention, RoPE, RMSNorm, router,
MoE, KV append, logits, argmax), and ends in correct one-time settlement.
