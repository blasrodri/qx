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

The live two-machine path demonstrates:

1. requester-signed job creation and gateway validation;
2. authenticated job delivery to the assigned miner;
3. full Granite inference from the committed `.qx` artifact;
4. generation of output token IDs and the verification trace root; and
5. retrieval of the miner's provisional result by the requester.

The repository's SBF tests separately execute the complete adversarial path:
successor-block epoch entropy, sampling, unsigned-result attribution,
interactive bisection, and objective arithmetic conviction. The remote demo
does not yet automate all challenger turns, collateral, or rewards.

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

## Requirements

The miner needs a macOS Apple Silicon machine, Xcode/Metal, stable Rust,
`curl`, `jq`, and the exact `.qx` artifact. Both people should use the same QX
Git commit. The requester also needs `openssl`; the Solana CLI is needed for
the on-chain checks and verifier tests.

Both people run:

```bash
git clone https://github.com/blasrodri/qx.git
cd qx
cargo build --release -p qx-gateway
```

Never create or store private keys or gateway tokens inside the repository.

## 1. Prepare the artifact

The quickest option is to transfer this file to the miner over a trusted
channel:

```text
external/models/granite-m5-pro-q4-factorial-rowinline.qx
```

The miner verifies it before running:

```bash
shasum -a 256 \
  external/models/granite-m5-pro-q4-factorial-rowinline.qx
```

The required SHA-256 is:

```text
b9c17197dc3014ae446188abf79a2b3e73af3e7ff78b68f80c89e5a546b54c84
```

The artifact is deliberately excluded from Git. See
[Reproduce the artifact](#reproduce-the-artifact) to build it from the pinned
IBM Granite checkpoint instead of transferring it.

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
  external/models/granite-m5-pro-q4-factorial-rowinline.qx \
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

## Reproduce the artifact

Run this once from a clean QX clone. It downloads the pinned IBM Granite
checkpoint and uses the historical compiler revision recorded in the released
artifact. The harmless marker is required because compiler provenance,
including its `+dirty` state, is embedded in the artifact.

```bash
git lfs install

QX_REPO_ROOT="$(pwd)"
QX_COMPILER_TREE="${QX_REPO_ROOT}-artifact-compiler"
QX_MODEL_DIR="$QX_REPO_ROOT/external/models/granite-3.0-1b-a400m-instruct"
QX_OUTPUT="$QX_REPO_ROOT/external/models/granite-m5-pro-q4-factorial-rowinline.qx"

mkdir -p "$QX_REPO_ROOT/external/models"

git clone --no-checkout \
  https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct \
  "$QX_MODEL_DIR"
git -C "$QX_MODEL_DIR" checkout \
  ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445

test "$(shasum -a 256 "$QX_MODEL_DIR/model.safetensors" | awk '{print $1}')" = \
  "f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc" ||
  { echo "Model checksum mismatch"; exit 1; }

git fetch origin 464424e7428ac048c9cba88de13db79aeca5c7c6
git worktree add --detach \
  "$QX_COMPILER_TREE" \
  464424e7428ac048c9cba88de13db79aeca5c7c6
touch "$QX_COMPILER_TREE/.qx-reproduction-marker"

cd "$QX_COMPILER_TREE"
cargo run --release -p qx-compile -- \
  "$QX_MODEL_DIR" \
  --target m5-pro \
  --memory-budget 2gb \
  --context 512 \
  --disable-q3 \
  --uniform-expert-bits q4 \
  --expert-layout row-inline \
  --moe-mapping qx \
  --weight-storage mmap-no-copy \
  --command-buffers-per-token 1 \
  --plan-id factorial-rowinline-qx \
  --output "$QX_OUTPUT"

test "$(shasum -a 256 "$QX_OUTPUT" | awk '{print $1}')" = \
  "b9c17197dc3014ae446188abf79a2b3e73af3e7ff78b68f80c89e5a546b54c84" ||
  { echo "Artifact checksum mismatch"; exit 1; }

echo "Artifact reproduced and verified: $QX_OUTPUT"
cd "$QX_REPO_ROOT"
git worktree remove --force "$QX_COMPILER_TREE"
```

Do not compile this artifact from current `main`: the planner has changed, so
the result is valid but not byte-identical to the registered demo artifact.

For gateway hosting, epoch operation, API details, and manual package
inspection, continue with
[the full devnet quickstart](docs/devnet-two-person-quickstart.md).
