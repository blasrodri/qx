# QX two-person devnet demo

This demo runs a requester and an Apple Silicon miner against the public QX
gateway and the verifier deployed on Solana devnet. The requester submits a
signed job containing the input token IDs, assigned model, execution spec,
miner, epoch, and 5% audit probability. The miner needs no protocol signing
key: it pulls the job, runs the committed Metal inference and trace, and posts
the result through its authenticated gateway connection.

Devnet only: this software is experimental and must not be used to settle
value.

## What the demo proves

The one-command proof demo runs requester, gateway, miner, challenger, devnet,
and SBF verification checks. It fails unless all of these complete:

1. requester-signed job creation and authenticated miner delivery;
2. full Granite inference and committed output/trace generation;
3. immutable 30-second local epochs with no late-result inclusion race;
4. epoch-root submission and successor-block seeding on Solana devnet;
5. independent validation of every selected package against its devnet epoch,
   registered model, requester signature, sampling rule, and Merkle proof;
6. deterministic full-model replay of every job selected by the 5% rule; and
7. SBF execution of adversarial bisection, arithmetic conviction, and the
   packet-safe production Granite trace admission path.

The adversarial dispute is executed against the deployable SBF binary by the
local test harness. Automating every interactive dispute transaction between
remote challenger and miner processes on public devnet remains separate work;
the demo labels this evidence `adversarial_sbf` rather than claiming it is a
live remote dispute.

## Shared deployment

```text
gateway:       https://qx-gateway-devnet-sitoula-8080.ignite.dodil.cloud
Solana RPC:    https://api.devnet.solana.com
program:       QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB
model registry: CsUgpxaXfofqfEVS1DbTR8pcGqC4mCWXhR15kgiDoCZe
miner label:   4vikCFiyndM6KfpK6EezrpLPUjgUs9eZKCDxUhpuXCRa
```

The miner label is routing and attribution data, not a signing identity.

Check that the gateway and program are available:

```bash
curl --fail --silent --show-error \
  https://qx-gateway-devnet-sitoula-8080.ignite.dodil.cloud/health

solana program show QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB \
  --url https://api.devnet.solana.com
```

## Run the 30-minute proof demo

From a clean clone on an Apple Silicon Mac, run:

```bash
./scripts/run-demo.sh
```

It builds the artifact when needed, creates temporary demo identities, uses
the configured Solana payer (or funds a temporary one from the devnet faucet),
and runs 60 consecutive 30-second job epochs. Collection windows are closed
locally every 30 seconds; their devnet submission and replay happen afterward
so RPC confirmation latency cannot stretch the epoch cadence. Every job prints
its input and output token IDs, job hash, result leaf, and trace root.

Every epoch is then committed and seeded on devnet. Every 5%-selected package
is independently verified and replayed. The command also builds the SBF
program and executes the adversarial bisection-to-conviction tests. Missing SBF
tooling, zero selected audits, a count mismatch, an invalid package, or any
replay disagreement makes the command fail.

Public evidence is retained under `.qx-demo-runs/<UTC timestamp>/`, including
the complete log, closed and seeded epoch records, selected packages, replay
results, and `summary.json`. Temporary keys and transport credentials are
deleted and are never copied into the evidence directory.

For a funded existing payer, avoiding the devnet faucet:

```bash
QX_PAYER_KEYPAIR=../qx-devnet-keys/payer.json ./scripts/run-demo.sh
```

For a one-epoch smoke run, select every job so the audit requirement is still
exercised:

```bash
QX_DEMO_DURATION_SECONDS=30 \
QX_AUDIT_PROBABILITY_PPM=1000000 \
./scripts/run-demo.sh
```

The final `event=demo_complete status=passed` line reports epochs, submitted
and committed jobs, failures, selected audits, independently verified
packages, matching replays, and adversarial SBF status. A full 60-epoch run
creates approximately 0.085 SOL of devnet epoch-account rent plus transaction
fees; use a funded devnet payer if the faucet is rate-limited.

## Requirements

The miner needs a macOS Apple Silicon machine, Xcode/Metal, stable Rust,
`curl`, `jq`, and a `.qx` artifact built from the current QX source. Both
people should use the same QX Git commit. The requester also needs `openssl`;
the Solana CLI is needed for the on-chain checks and verifier tests.

Both people run:

```bash
git clone https://github.com/blasrodri/qx.git
cd qx
cargo build --release -p qx-gateway
```

Never create or store private keys or gateway tokens inside the repository.

## 1. Build the artifact

From the repository root, run:

```bash
./scripts/build-artifact.sh
```

The command downloads and verifies the Granite checkpoint, compiles it using
the current checkout, and prints the new artifact's path and SHA-256. The
output is intentionally excluded from Git.

## 2. Give the miner its transport credential

The gateway operator sends the miner transport token through an encrypted
private channel. This credential protects gateway write routes; it is not a
miner signature and must never appear in a job payload or Git commit.

The miner saves it outside the repository:

```bash
mkdir -p ../qx-devnet-keys
chmod 700 ../qx-devnet-keys

# Paste the token into this file without committing it.
read -rs QX_TOKEN
printf '%s\n' "$QX_TOKEN" > ../qx-devnet-keys/gateway.token
unset QX_TOKEN
chmod 600 ../qx-devnet-keys/gateway.token
```

## 3. Start the miner

On the friend's Apple Silicon machine:

```bash
./scripts/devnet-miner-worker.sh \
  4vikCFiyndM6KfpK6EezrpLPUjgUs9eZKCDxUhpuXCRa \
  external/models/granite-m5-pro-q4-main.qx \
  ../qx-devnet-keys/gateway.token
```

Leave this terminal running. The expected initial status is:

```text
worker_status=connected
```

The test worker uses reconnecting polling. A production dispatch plane should
replace it with durable streaming, leases, acknowledgements, and backpressure;
that transport change does not alter the signed job payload or protocol.

## 4. Submit a job

On the requester's machine, create a requester keypair outside the repository.
It signs jobs but does not need devnet SOL merely to submit one to the gateway:

```bash
mkdir -p ../qx-devnet-keys
chmod 700 ../qx-devnet-keys
solana-keygen new \
  --no-bip39-passphrase \
  --outfile ../qx-devnet-keys/requester.json
```

Submit input token IDs `1,2,3`, request at most eight output tokens, and wait up
to five minutes:

```bash
./scripts/devnet-client-test.sh \
  ../qx-devnet-keys/requester.json \
  4vikCFiyndM6KfpK6EezrpLPUjgUs9eZKCDxUhpuXCRa \
  1,2,3 \
  8 \
  300
```

The requester prints a job hash immediately. Once the miner completes the
job, both terminals report completion and the requester prints values like:

```text
job_status=completed
output_tokens=[...]
result_leaf_hex=...
```

The script fixes `audit_probability_ppm=50000`, so each job has a 5% sampling
probability after its epoch is committed and seeded. A returned inference is
still provisional until that epoch and its challenge window complete.

## 5. Exercise the verifier

Run these on a machine with `cargo-build-sbf` installed:

```bash
cargo build-sbf --manifest-path programs/qx-terminal-verifier/Cargo.toml

cargo test -p qx-terminal-verifier --test sbf_v2 \
  block_seeded_unsigned_result_is_bisected_and_convicted_on_sbf \
  -- --nocapture

cargo test -p qx-terminal-verifier --test sbf_v2 \
  granite_output_and_trace_bounds_are_packet_safe_and_admitted_on_sbf \
  -- --nocapture
```

The first test covers the seeded sampling-to-conviction flow. The second checks
that the complete Granite trace protocol fits Solana's transaction/account
bounds and is admitted by the deployed-program build.

To generate real devnet load as well, fund the requester with faucet SOL and
run ten program probes:

```bash
REQUESTER=$(solana-keygen pubkey ../qx-devnet-keys/requester.json)
solana airdrop 1 "$REQUESTER" --url https://api.devnet.solana.com

./scripts/devnet-probe.sh \
  QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB \
  ../qx-devnet-keys/requester.json \
  10
```

The probe measures transaction compute units, fees, throughput, failures, and
confirmation latency. It is a load probe, not a model inference or challenge.

For gateway hosting, epoch operation, API details, and manual package
inspection, continue with
[the full devnet quickstart](docs/devnet-two-person-quickstart.md).
