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

1. every inference pays the 3.203% measured commitment cost in the selected
   two-layer checkpoint profile (3.445% for every-layer checkpoints);
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
| `stage_q4_weight_v1` | 873 bytes | 7,108 | Verify 26 SHA-256 Merkle levels against the registered 41,755,744-leaf weight root and store the decoded weight |
| `verify_staged_q4_fma_v1` | 340 bytes | 1,877 | Verify 5 BLAKE3 activation levels and execute one exact FP32 fused multiply-add |
| Total | 1,213 bytes over two transactions | 8,985 | Complete terminal arithmetic check |

The test touched 36 bytes of model-root data and a 20-byte temporary staged
weight account. A deliberately false accumulator was rejected after 1,875 CU.
The SBF binary was 30,864 bytes. Measurements used `cargo-build-sbf 4.1.0`,
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

## The remaining consensus blocker

The current production Metal kernels use `simd_sum` for 32-lane floating-point
reductions. Apple specifies that the operation sums active lanes, but does not
specify a consensus-visible rounding tree. The lane-local FMA verified here is
exact and executable on Solana; the final connection from 32 lane accumulators
to the committed matrix output is not yet protocol-safe.

Before slashing can be enabled, `QX_EXEC_METAL_V1` must replace those
consensus-relevant reductions with an explicit shuffle/add tree (and prevent
compiler reassociation), then re-run inference parity and the <5% overhead
benchmark. The same treatment is required for reduction-heavy primitives such
as RMSNorm and softmax. Empirical repeatability on one M5 is necessary but is
not a substitute for specifying the arithmetic tree.

Accordingly, the current code proves these pieces:

- deterministic epoch receipt selection and exact detection probabilities;
- state and weight openings;
- token/layer disagreement location oracle;
- router ordering and tie breaking;
- authenticated terminal Q4 FMA on actual SBF with measured CU;
- rejection of corrupted arithmetic, openings, and false ordering claims.

It does not yet justify automatic slashing for a complete Metal inference.
That line should remain disabled until the explicit reduction profile and the
outer dispute-PDA transition checks are implemented and tested end to end.

## Platform references

- [Solana compute budget](https://solana.com/docs/core/fees/compute-budget)
- [Solana program limitations, including software-emulated floating point](https://solana.com/docs/programs/limitations)
- [Solana syscall reference](https://github.com/solana-foundation/solana-com/blob/main/apps/docs/content/docs/en/core/programs/syscall-reference.mdx)
- [Solana Mollusk testing guide](https://github.com/solana-foundation/solana-com/blob/main/apps/docs/content/docs/en/programs/testing/mollusk.mdx)
