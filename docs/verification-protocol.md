# QX sampled optimistic verification protocol

## Current conclusion

The outer protocol should sample inference receipts by epoch. An epoch closes a
Merkle root and receipt count first; only then does future entropy select exact
receipt positions. Every selected receipt is fully replayed off-chain. Tokens
and layers are not independently sampled in V1: once a receipt is selected,
sampling internal work would give a miner another place to hide corruption.
The token/layer/operator trees are used to locate a detected disagreement, not
to decide whether the selected inference is checked.

This separates three costs:

1. every inference pays 2.904% measured commitment cost inside the selected
   precise two-layer checkpoint profile; migrating from fast uncommitted
   execution to precise committed execution currently costs 8.820%;
2. the network pays approximately `audited receipts / epoch receipts` in extra
   off-chain inference work, adjusted for the lengths of the selected jobs;
3. Solana pays only when a selected replay disagrees and a bonded dispute is
   opened.

## Epoch selection

An `AuditEpochV1` fixes:

```text
epoch
receipt_root
receipt_count
audit_count
entropy_slot
```

After the receipt root is immutable, derive:

```text
seed = BLAKE3(
    QX:AUDIT:SEED:V1,
    epoch,
    receipt_root,
    receipt_count,
    audit_count,
    entropy_slot,
    future_solana_entropy,
    independent_entropy
)
```

`sample_receipt_indices_v1` uses rejection sampling over this seed to produce
exactly `audit_count` distinct indices without modulo bias. A selected receipt
is authenticated at that position against `receipt_root`. The selection code
and probability calculator live in `qx-commit`; `qx-verify sampling-cost`
reproduces the economics.

The entropy rule must not give the miner a choice among several valid seeds.
For approximately independent alternatives, if one seed detects cheating with
probability `p`, a miner allowed to choose the most favorable of `g` seeds is
detected only with probability `p^g`. Commit/reveal schemes need a fixed rule
for missing reveals; otherwise withholding itself becomes a second seed.
A fixed future Solana entropy slot mixed with at least one independently
committed value is the minimum sensible construction. The production design
must specify the exact Solana entropy source and its availability/fallback
rule before deployment.

## What sampling buys

For 10,000 equal-cost receipts where a miner corrupts 100 receipts (1%), exact
sampling without replacement gives:

| Audits | Audit work | Detect at least one | Detect with two-seed grinding |
|---:|---:|---:|---:|
| 10 | 0.10% | 9.5659% | 0.9151% |
| 25 | 0.25% | 22.2415% | 4.9468% |
| 50 | 0.50% | 39.5745% | 15.6614% |
| 100 | 1.00% | 63.5805% | 40.4249% |
| 250 | 2.50% | 92.1492% | 84.9148% |
| 500 | 5.00% | 99.4233% | 98.8499% |

At this population and cheating rate, 448 audits (4.48%) reach 99% detection.
But a miner corrupting one specially chosen receipt is detected with exactly
the audit fraction: 1% auditing gives only 1% detection. Sampling therefore
cannot stand alone for high-value or targeted requests.

Use value-weighted strata:

- audit every receipt above a value/risk threshold;
- sample ordinary receipts within buckets of similar compute cost and value;
- cap the unsettled value attributable to one miner in each epoch;
- allow permissionless bonded challenges in addition to mandatory sampling;
- size collateral against the worst permitted epoch exposure, not the average
  job value.

If cheating yields gain `G`, detection probability is `p`, and a successful
proof loses bond `B`, the bare rational-deterrence condition is `p * B > G`.
Real parameters need a safety multiplier and must include the chance that no
honest auditor is online. For a single targeted receipt audited at 1%, the
bare bond already exceeds `100 * G`; this is why exposure caps and mandatory
audits for valuable work matter more than a large undifferentiated bond.

## Disagreement and terminal proof

For a selected receipt:

```text
full independent replay
  -> first differing token root
  -> first differing layer checkpoint
  -> replay that layer into a dispute-only operator trace
  -> first differing operator transition
  -> Q4 dot lane and step
  -> one authenticated Q4 floating-point FMA on Solana
```

Normal execution stores no operator activations. A dispute regenerates them
and commits dispute-local activations with 128-byte BLAKE3 leaves. The smaller
leaf is deliberate: it keeps the activation opening transportable alongside
the 26-level production weight proof.

The deployable SBF prototype uses two instructions:

| Instruction | Data | CU | Purpose |
|---|---:|---:|---|
| `stage_q4_weight_v1` | 873 bytes | 7,444 | Verify 26 SHA-256 Merkle levels and stage a dispute-bound decoded weight |
| `verify_staged_q4_fma_v1` | 300 bytes | 2,336 | Verify the dispute-bound activation and select the objectively correct claim |
| Terminal total | 1,173 bytes over two transactions | 9,780 | Complete terminal arithmetic check |

Opening a miner-signed receipt against its epoch root costs 2,772 CU and binding
the mutually agreed terminal boundary costs 1,100 CU. The complete measured
path is 13,652 CU, plus 1,842 CU once per model+epoch registration in this test.

The program uses a 56-byte epoch account, 72-byte model account, 304-byte active
dispute account, and 52-byte temporary staged-weight account. A staged weight
cannot be replayed into another dispute; forged receipt signatures, bad receipt
openings, bad activation openings, early timeouts, and false arithmetic are
rejected. Measurements used `cargo-build-sbf 4.1.0`,
Solana CLI 4.2.2, and Mollusk's SVM executor; they are executed CU measurements,
not host estimates. The test command is:

```text
cargo build-sbf --manifest-path programs/qx-terminal-verifier/Cargo.toml
cargo test -p qx-terminal-verifier --test sbf -- --nocapture
```

A monolithic proof was 1,256 instruction bytes. That does not fit a Solana
transaction once signatures and account keys are included, even though its
execution used only 8,070 CU. Splitting authentication and arithmetic solves
the transport limit at a small CU cost. The staged account must ultimately be
the active dispute PDA, so a proof cannot be replayed into another dispute.

## Cheating analysis

| Attempt | Result | Remaining assumption |
|---|---|---|
| Change a committed weight or activation | Merkle opening fails | SHA-256/BLAKE3 binding security |
| Claim a bad lane-local accumulator | SBF FMA rejects it | FP32/FMA semantics match the execution profile |
| Claim a higher router score that is lower | canonical score ordering rejects the challenger | the two score endpoints were reached by valid dot disputes |
| Exploit an equal router score | lower expert ID wins | router tie rule remains registered and immutable |
| Lie as challenger | authenticated openings or arithmetic fail; challenger bond should be lost | outer dispute state machine enforces settlement |
| Corrupt an unsampled ordinary receipt | escapes mandatory audit for that epoch | permissionless audits, collateral, and future sampling deter it economically |
| Grind or choose entropy | can sharply reduce detection | entropy is fixed, delayed, and not miner-selectable |
| Withhold a dispute response | cannot be decided by arithmetic alone | outer on-chain deadline must award the loss automatically |

## Consensus arithmetic status

`QX_EXEC_METAL_V1` now replaces native `simd_sum`/`simd_max` with an explicit
32-lane shuffle-down tree in the Q4, dense, decode, and dynamic-MoE kernels.
Metal fast math is disabled by default, and the host verifier contains the
matching FP32 sum oracle. Live Metal tests pass, and the execution-spec digest
binds the tree and precise-math requirement.

This removes the previously unspecified reduction order, but costs 5.749% over
the explicit tree compiled in fast mode. Precise execution plus commitments is
8.820% slower than fast uncommitted execution. Deterministic decompositions and
terminal instructions are still needed for rsqrt, exp/softmax, RoPE, KV append,
and the remaining verification IR before a complete inference can be slashed.

Accordingly, the current code proves these pieces:

- deterministic epoch receipt selection and exact detection probabilities;
- state and weight openings;
- token/layer disagreement location oracle;
- router ordering and tie breaking;
- authenticated terminal Q4 FMA on actual SBF with measured CU;
- rejection of corrupted arithmetic, openings, and false ordering claims.

The SBF path now authenticates the receipt signature and epoch inclusion and
binds the model, parties, activation root, arithmetic claims, and staged weight
to a single dispute account. It still uses a two-party terminal-boundary
agreement as the handoff from off-chain narrowing. Automatic slashing remains
disabled until on-chain token/layer/operator bisection can reach that same
terminal phase without miner cooperation and the remaining IR operations are
deterministically covered.

## Platform references

- [Solana compute budget](https://solana.com/docs/core/fees/compute-budget)
- [Solana program limitations, including software-emulated floating point](https://solana.com/docs/programs/limitations)
- [Solana syscall reference](https://github.com/solana-foundation/solana-com/blob/main/apps/docs/content/docs/en/core/programs/syscall-reference.mdx)
- [Solana Mollusk testing guide](https://github.com/solana-foundation/solana-com/blob/main/apps/docs/content/docs/en/programs/testing/mollusk.mdx)
