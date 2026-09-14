# Two-person Solana testnet quickstart

This guide lets one person deploy the QX verifier and a second person confirm
that the deployed SBF program executes real hash work on Solana testnet. It
also explains how to run the complete adversarial dispute suite locally.

## Scope

The public-testnet command currently exercises the deployed program's bounded
load-probe instruction. The complete epoch, sampling, bisection, Granite micro
trace, and objective terminal flows execute as SBF integration tests through
Mollusk. A network client that coordinates those stateful instructions between
two remote keypairs is not yet implemented. Do not describe the load probe as
an end-to-end public-testnet challenge.

## Requirements

- macOS on Apple Silicon for full Granite/Metal replay;
- a current stable Rust toolchain;
- the Agave/Solana CLI and `cargo-build-sbf`;
- Xcode and the Metal toolchain for model execution;
- testnet SOL for each transaction payer;
- the same QX Granite artifact and tokenizer files for full-model replay.

Use the official Solana installation and cluster documentation:

- <https://solana.com/docs/intro/installation>
- <https://solana.com/docs/references/clusters>

Check the tools before continuing:

```bash
rustc --version
cargo --version
solana --version
cargo build-sbf --version
```

Never create keypairs inside the repository. The Git ignore rules are a final
safety net, not a substitute for keeping credentials elsewhere.

## 1. Clone and verify the source

Both people run:

```bash
git clone https://github.com/blasrodri/qx.git
cd qx
cargo test -p qx-commit -p qx-consensus-math -p qx-verifier-core
cargo build-sbf --manifest-path programs/qx-terminal-verifier/Cargo.toml
cargo test -p qx-terminal-verifier --test sbf --test sbf_v2
```

The SBF tests require the `cargo build-sbf` output in `target/deploy`.

## 2. Create separate testnet identities

Each person creates a keypair outside the repository:

```bash
mkdir -p ../qx-testnet-keys
solana-keygen new --outfile ../qx-testnet-keys/payer.json
solana-keygen pubkey ../qx-testnet-keys/payer.json
solana airdrop 2 "$(solana-keygen pubkey ../qx-testnet-keys/payer.json)" \
  --url https://api.testnet.solana.com
```

The testnet faucet may rate-limit requests. Neither keypair nor its seed phrase
should be sent to the other person or committed to Git.

## 3. Deploy the verifier

The deployer runs:

```bash
./scripts/testnet-deploy.sh ../qx-testnet-keys/payer.json
```

The final line prints:

```text
PROGRAM_ID=<public key>
```

Only share that public program ID. Keep
`target/deploy/qx_terminal_verifier-keypair.json` private because it controls
the program address and may control future upgrades.

To deploy an already assigned program address, pass its keypair explicitly:

```bash
./scripts/testnet-deploy.sh \
  ../qx-testnet-keys/payer.json \
  ../qx-testnet-keys/qx-program.json
```

## 4. Let the second person exercise the deployment

The friend clones the same commit and runs:

```bash
./scripts/testnet-probe.sh \
  <PROGRAM_ID> \
  ../qx-testnet-keys/payer.json \
  10
```

The command first simulates a representative transaction, then sends and
confirms the requested probes. It reports simulation compute units, fees,
successes, failures, throughput, and mean confirmation latency. A nonzero
failure count exits unsuccessfully.

Both people can independently verify the program account:

```bash
solana program show <PROGRAM_ID> --url https://api.testnet.solana.com
```

## 5. Exercise the complete dispute implementation

Until the distributed client exists, run the actual stateful verifier flow
through the SBF test harness:

```bash
cargo build-sbf --manifest-path programs/qx-terminal-verifier/Cargo.toml
cargo test -p qx-terminal-verifier --test sbf_v2 \
  block_seeded_unsigned_result_is_bisected_and_convicted_on_sbf \
  -- --nocapture

cargo test -p qx-terminal-verifier --test sbf_v2 \
  granite_output_and_trace_bounds_are_packet_safe_and_admitted_on_sbf \
  -- --nocapture
```

The first test covers block-seeded selection, unsigned result attribution,
bisection, and objective conviction. The second executes packet-safe Granite
micro-trace admission for both challenger and miner.

For the complete artifact parity tests, place the externally distributed model
under `external/models/` as documented in
`docs/verification-model-profile.md`. Model artifacts are intentionally not
stored in Git.

## One-hour epoch operating rule

For a one-hour epoch, the gateway retains every signed job envelope and ordered
result leaf until the successor-block seed is finalized. It sends every
selected replay package to challengers. The gateway and miner retain selected
packages, the epoch Merkle tree, and any open-dispute data through the current
216,000-slot challenge window plus dispute completion.

The gateway transports data but does not decide correctness. Challengers
verify requester signatures, input hashes, epoch inclusion, sampling, model
identity, and execution-spec identity before replay. Solana resolves an opened
disagreement.

## Current limitations

- collateral reservation, challenger bonds, and automatic fund transfers are
  not enabled;
- gateway-to-challenger package transport is a protocol interface, not a
  shipped network service;
- the distributed public-testnet client for the interactive dispute turns is
  not implemented;
- cross-device consensus validation is not complete.

Use testnet accounts only. Do not attach economic value to this version.
