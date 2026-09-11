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

1. every inference pays 2.939–4.201% measured commitment cost inside the selected
   fixed-math two-layer checkpoint profile, including output-token binding; the
   earlier fast-uncommitted to precise-committed migration measurement was
   8.820%, so arithmetic mode remains a separate deployment cost;
2. the network pays approximately `audited receipts / epoch receipts` in extra
   off-chain inference work, adjusted for the lengths of the selected jobs;
3. Solana pays only when a selected replay disagrees and a bonded dispute is
   opened.

## Adversarial V2 implementation status (2026-09-10)

The implemented live SBF path is:

```text
requester-signed immutable job payload (310 bytes, off chain)
  -> unsigned result leaf (104 bytes)
  -> one miner-signed epoch root
  -> first produced Solana block after epoch close
  -> per-job Bernoulli audit draw
  -> independent challenger/miner Merkle openings
  -> job-token and checkpoint bisection with exact preimage reveals
  -> checkpoint-bound dispute-local operator traces
  -> unilateral logarithmic operator bisection with turn-specific timeouts
  -> registered instruction + runtime residual-chunk opening
  -> objective state-root update
```

There is no miner signature per job. The requester's job signature is checked
only if a challenge opens, so normal operation has no per-job Solana
transaction. A program-owned miner identity records the last submitted epoch;
the job payload binds its one assigned epoch. Together these reject duplicate
epoch/root grinding. The miner does not choose the entropy slot, challenge
window, or response timeout. The constants are 216,000 slots for challenges
and 1,200 slots per dispute move.

The current live-SBF measurements are:

| Operation | Compute units |
|---|---:|
| Finalize epoch seed (once/epoch) | 1,784 |
| Authenticate job/result and open challenge | 8,677 |
| Both boundary openings + one midpoint round | 10,329 |
| Authenticate segment program and bind both operator traces | 8,914 |
| Chained runtime-chunk scalar terminal | 4,344 |
| Chained authenticated Q4 terminal (two instructions) | 11,883 |
| Chained structured KV append (two instructions) | 9,412 |
| Intermediate pairwise argmax in a two-leaf registered program | 4,471 |
| Authenticate head program and bind both final-head traces | 10,409 |
| Final-head token-deciding argmax | 4,110 |
| Compatibility staged Q4/FMA terminal | 9,534 |

Each additional binary-search level costs one challenger and one miner
midpoint instruction. The complete three-state synthetic dispute is 32,359 CU,
including both terminal-state reveals. The full test also exercises both
timeout directions, rejects an unselected result, rejects duplicate epoch
submission, and convicts an incorrect accumulator without a miner co-signature.

The SBF production-format test now traverses a `QX:JOB:V1` root, reveals the
differing `TokenResultV1`, traverses its `QX:TOKEN:V1` checkpoint root,
authenticates that interval's program descriptor from the model schedule,
binds both `QX:OPERATOR:V1` traces to the agreed runtime checkpoint roots, and
convicts an incorrect update to the same 256-byte chunk tree used by normal
execution. Compact roots and exact lengths prevent trace shortening.

Automatic slashing remains disabled. The generic hierarchy, canonical
first-segment checkpoint, chained scalar/Q4/KV terminals, and token-bound
final-head argmax path work. Equal-checkpoint prediction disagreements now
descend through a registered operator trace and resolve objectively. The real
Granite computation is not lowered into the registered dispute VM yet, and
logit/K/V vector roots are not yet produced through that lowering. Collateral
transfer and durable input availability are also outside this proof milestone. See
`docs/verification-security-audit.md` for the exact readiness gate.

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

V2 uses a per-job Bernoulli draw instead of global k-of-n. The immutable job
payload commits `audit_probability_ppm`; the on-chain check hashes the closed
epoch seed, epoch number, leaf index, and job hash. Adding cheap results cannot
dilute another job's probability. Zero and one million ppm are exact boundary
cases. Entropy is the earliest produced Solana slot strictly after the epoch
root closes, found by binary search in the canonical slot-hashes sysvar;
skipped slots do not create a finalizer-selectable fallback.

The entropy rule must not give the miner a choice among several valid seeds.
For approximately independent alternatives, if one seed detects cheating with
probability `p`, a miner allowed to choose the most favorable of `g` seeds is
detected only with probability `p^g`. Commit/reveal schemes need a fixed rule
for missing reveals; otherwise withholding itself becomes a second seed.
V2 implements this as the first produced successor block and binds the miner,
epoch, result root, result count, entropy slot, program instance, and block
hash into the seed. Long forks and slot-hash retention still need deployment
monitoring, but neither miner nor finalizer chooses among valid hashes.

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

The following measurements describe the older compatibility V1 terminal path.
It requires a per-result miner signature and mutual terminal agreement, so it
is retained for regression comparison and must not drive settlement. That
prototype uses two instructions:

| Instruction | Data | CU | Purpose |
|---|---:|---:|---|
| `stage_q4_weight_v1` | 873 bytes | 7,420 | Verify 26 SHA-256 Merkle levels and stage a dispute-bound decoded weight |
| `verify_staged_q4_fma_v1` | 300 bytes | 2,351 | Verify the dispute-bound activation and select the objectively correct claim |
| Terminal total | 1,173 bytes over two transactions | 9,771 | Complete terminal arithmetic check |

Opening a miner-signed receipt against its epoch root costs 2,746 CU and binding
the mutually agreed terminal boundary costs 1,117 CU. The complete measured
path is 13,634 CU, plus 2,507 CU once per model+epoch registration in this test.

The compatibility path uses a 56-byte epoch account, the shared registry now
uses a 156-byte model account, and the path uses a 304-byte active dispute
account and 64-byte temporary staged-weight account. A staged weight
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
matching FP32 sum oracle. It also replaces native `exp` and `rsqrt` throughout
the model kernels with protocol-defined fixed-operation implementations. Rust,
Metal, and the SBF terminal evaluator share identical constants and operation
order. Live Metal bit-parity tests pass. The selected interval-2 execution-spec
`01e47ff99355d9eca97bde30253739fe38fdfb70049871d4307daa89ffb9cd1d`
binds the fixed-math and commitment-shape profile.

The scalar terminal IR now objectively evaluates FMA/dot steps, add, multiply,
ordered sum/max, compare, select, division, fixed exp, and fixed rsqrt. An
actual SBF fixed-exp dispute, including verification-program authentication,
consumes 10,803 CU across both terminal reveals and resolution. RoPE
decomposes into multiply/add steps; softmax and RMSNorm use the
fixed exp/rsqrt/div primitives. Argmax and Top-K decompose into compare/select.
KV leaf hashing exists, but its append/frontier-update witness and the nested
layer-to-operator trace bridge remain required before complete inference is
slashable end to end.

An experimental Metal profile now combines relaxed arithmetic optimization
with precise math functions. It reduced the mean fast-uncommitted to
relaxed-committed migration cost to 3.734% in two independent 31-repeat runs,
and all 62 repeats produced the same trace root. Recorded AIR still carries
unsafe reassociation, signed-zero, reciprocal, contraction, and approximate
function flags, so this result is a performance candidate rather than a
slashable consensus profile. It must not be promoted solely because it is
repeatable on one GPU.

Accordingly, the current code proves these pieces:

- deterministic epoch result selection and exact detection probabilities;
- one aggregate miner epoch signature, requester-authenticated job payloads,
  and no normal per-job Solana registration;
- binding of every executed token's predicted token into the runtime trace;
- state and weight openings;
- token/layer disagreement location oracle;
- router ordering and tie breaking;
- unilateral on-chain bisection and authenticated terminal Q4 FMA on actual
  SBF with measured CU;
- rejection of corrupted arithmetic, openings, and false ordering claims.

The V2 path authenticates the requester job signature lazily, attributes an
unsigned result through the miner's epoch, and binds the model, parties,
activation root, arithmetic claims, and staged weight to one dispute account.
It no longer needs mutual agreement or a miner signature per result.
Hierarchical bisection now reaches a chained scalar terminal from a
production-format runtime root. Automatic slashing remains disabled until the
registered VM covers genesis, every Granite operation family, KV append, and
final argmax, and collateral/bond settlement is implemented.

## Platform references

- [Solana compute budget](https://solana.com/docs/core/fees/compute-budget)
- [Solana program limitations, including software-emulated floating point](https://solana.com/docs/programs/limitations)
- [Solana syscall reference](https://github.com/solana-foundation/solana-com/blob/main/apps/docs/content/docs/en/core/programs/syscall-reference.mdx)
- [Solana Mollusk testing guide](https://github.com/solana-foundation/solana-com/blob/main/apps/docs/content/docs/en/programs/testing/mollusk.mdx)
