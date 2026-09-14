# Two-person Solana devnet quickstart

This guide lets one person run the QX gateway and a second person retrieve the
self-authenticating jobs selected for replay. The verifier is already deployed
on Solana devnet. It also explains how to run the complete adversarial dispute
suite locally.

## Scope

The public-devnet command exercises the deployed program's bounded load-probe
instruction. `qx-gateway` now validates and durably stores requester-signed
jobs, accepts unsigned results over an authenticated miner transport, builds
epoch roots, submits/finalizes epoch accounts, syncs the on-chain seed, and
serves selected challenge packages. The client that automates every turn of a
live interactive Granite dispute is still separate work. Do not describe the
load probe as an end-to-end challenge.

## Requirements

- macOS on Apple Silicon for full Granite/Metal replay;
- a current stable Rust toolchain;
- the Agave/Solana CLI and `cargo-build-sbf`;
- Xcode and the Metal toolchain for model execution;
- devnet SOL for each transaction payer;
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

## 2. Create separate devnet identities

Each person creates a keypair outside the repository:

```bash
mkdir -p ../qx-devnet-keys
solana-keygen new --outfile ../qx-devnet-keys/payer.json
solana-keygen pubkey ../qx-devnet-keys/payer.json
solana airdrop 2 "$(solana-keygen pubkey ../qx-devnet-keys/payer.json)" \
  --url https://api.devnet.solana.com
```

The devnet faucet may rate-limit requests. Neither keypair nor its seed phrase
should be sent to the other person or committed to Git.

## 3. Confirm the verifier deployment

The shared devnet deployment is:

```bash
solana program show QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB \
  --url https://api.devnet.solana.com
```

The registered canonical interval-2 Granite profile is:

```text
model_registry = CsUgpxaXfofqfEVS1DbTR8pcGqC4mCWXhR15kgiDoCZe
model_id = bc205dbf40be32428fa653aff4c690b2c8a379a2d3fedbf617bb831a6264a17d
weights_root = c31f3758efc5be3b410c0747bea880128a026d3bb7f38a1fd51e1a8949072d43
execution_spec = 2485240f3be98d158f5fca212e98e6776254f7279e772c8b12daa6ee9bec6f5f
verification_schedule_root = 6d979a60646bb7d84fb4c8670d3bc5c7cd4a568284ddc9d8338af1e73ad5a092
```

The first live gateway epoch is retained on devnet as a proof of life:

```text
miner = 4vikCFiyndM6KfpK6EezrpLPUjgUs9eZKCDxUhpuXCRa
epoch_account = 9EbvoLYUBCjLaPrCPjBj5wbgH4QDUf4rHgsHgWT8uJTg
result_root = f002cec072a476f755ac7c795a8e127ee6c1b7cfb240dd6203225693cb8ffa5f
close_slot = 498207688
entropy_slot = 498207689
audit_seed = 4a8590bd692c9a2fd024eb4665739506eeca6ce0c09ac0cf83467c98ff94e640
```

That epoch contains a real interval-2 Metal execution. Its one-token output
was token `504`, its compact job trace had two leaves, and the gateway returned
the selected job with its exact signed payload and valid one-leaf epoch proof.

To deploy a separate copy, run `scripts/devnet-deploy.sh`. Keep
`target/deploy/qx_terminal_verifier-keypair.json` private because it controls
the program address and may control future upgrades.

To deploy an already assigned program address, pass its keypair explicitly:

```bash
./scripts/devnet-deploy.sh \
  ../qx-devnet-keys/payer.json \
  ../qx-devnet-keys/qx-program.json
```

## 4. Let the second person exercise the deployment

The friend clones the same commit and runs:

```bash
./scripts/devnet-probe.sh \
  QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB \
  ../qx-devnet-keys/payer.json \
  10
```

The command first simulates a representative transaction, then sends and
confirms the requested probes. It reports simulation compute units, fees,
successes, failures, throughput, and mean confirmation latency. A nonzero
failure count exits unsuccessfully.

Both people can independently verify the program account:

```bash
solana program show QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB \
  --url https://api.devnet.solana.com
```

## 5. Start the gateway

The canonical model above is already registered. To register a separate exact
model artifact, keep the registry
keypair outside Git. The command deterministically derives the model ID from
the logical weight root, registered execution spec, verifier schedule, and
fixed Granite dimensions. All participants record the printed model ID, weight
root, execution spec, and registry address:

```bash
cargo run --release -p qx-gateway -- register-model \
  --artifact external/models/granite-m5-pro-q4-rowmajor-nocopy-v2.qx \
  --checkpoint-interval 2 \
  --program QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB \
  --payer-keypair ../qx-devnet-keys/payer.json \
  --model-keypair ../qx-devnet-keys/model-registry.json
```

Registration streams and commits the artifact's logical weights, validates its
complete Granite tensor map, derives the interval-2 execution spec and macro
schedule, and submits those commitments to devnet. Physical Q4 layouts that
decode to the same logical weights produce the same weight root.

Create a miner transport token outside the repository, then start the service:

```bash
openssl rand -hex 32 > ../qx-devnet-keys/gateway.token
chmod 600 ../qx-devnet-keys/gateway.token
./scripts/devnet-gateway.sh \
  ../qx-gateway-data \
  ../qx-devnet-keys/gateway.token
```

Confirm it from the same machine:

```bash
curl http://127.0.0.1:8787/health
```

Bind to `0.0.0.0:8787` only behind a firewall, VPN, or TLS reverse proxy. The
read endpoints intentionally let challengers fetch selected packages. Miner
writes require `Authorization: Bearer <token>`. This transport credential is
not a per-job miner signature and has no protocol or slashing authority.

The gateway accepts these JSON resources:

| Method and path | Purpose |
|---|---|
| `POST /v1/jobs` | Validate requester signature and exact input-token hash |
| `POST /v1/results` | Store an unsigned result and its output tokens (miner auth) |
| `POST /v1/epochs/:epoch/close` | Build the deterministic ordered result root (miner auth) |
| `POST /v1/epochs/:epoch/sync` | Verify and copy the seeded epoch account from devnet (miner auth) |
| `GET /v1/epochs/:epoch/audits` | Return all selected replay packages |
| `GET /v1/challenges/:job_hash` | Return one selected package and epoch Merkle proof |

Use `qx-gateway create-job` and `create-result` to produce canonical request
bodies; `--help` lists their exact arguments. Submit the resulting files with
`curl --data-binary @file.json`. The job endpoint is requester-authenticated by
its Ed25519 signature. Result and epoch writes use the bearer token because V2
deliberately has no miner signature per result.

The normal miner path does not require constructing the result manually. Run
the signed job through the production Metal graph and emit its result package:

```bash
cargo run --release -p qx-gateway -- run-job \
  --job ../qx-gateway-work/job.json \
  --artifact external/models/granite-m5-pro-q4-factorial-rowinline.qx \
  --checkpoint-interval 2 \
  --context 512 \
  --output ../qx-gateway-work/result.json
```

`run-job` rejects a bad requester signature, mismatched input-token hash,
wrong execution spec, oversized context, and an artifact whose sealed runtime
plan is incompatible. It generates the output tokens and trace root from the
same committed inference execution. `create-result` exists for integrations
where another miner process already produced those values.

After results arrive, the miner/operator closes the local epoch, then runs:

```bash
cargo run --release -p qx-gateway -- submit-epoch \
  --data-dir ../qx-gateway-data \
  --epoch <EPOCH> \
  --program QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB \
  --payer-keypair ../qx-devnet-keys/payer.json \
  --miner-keypair ../qx-devnet-keys/miner.json \
  --epoch-keypair ../qx-devnet-keys/epoch-<EPOCH>.json

cargo run --release -p qx-gateway -- finalize-epoch \
  --program QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB \
  --payer-keypair ../qx-devnet-keys/payer.json \
  --epoch-account <EPOCH_ACCOUNT>
```

Finalization must happen after a successor block exists. Then call the gateway
`sync` endpoint with `{"epoch_account":"<EPOCH_ACCOUNT>"}`. It rejects any
account whose program owner, epoch number, miner, result root, or count differs
from the local manifest. Only after this sync can the audit endpoints return a
package.

## 6. Exercise the complete dispute implementation

Run the actual stateful verifier flow through the SBF test harness:

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
- the distributed devnet client for automating all interactive dispute turns is
  not implemented;
- cross-device consensus validation is not complete.

Use devnet accounts only. Do not attach economic value to this version.
