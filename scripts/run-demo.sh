#!/usr/bin/env bash
set -euo pipefail

QX_DURATION_SECONDS=${QX_DEMO_DURATION_SECONDS:-1800}
QX_JOB_DELAY_SECONDS=${QX_DEMO_JOB_DELAY_SECONDS:-0}
QX_MAX_OUTPUT_TOKENS=${QX_DEMO_MAX_OUTPUT_TOKENS:-4}
QX_AUDIT_PPM=${QX_AUDIT_PROBABILITY_PPM:-50000}
QX_PORT=${QX_DEMO_PORT:-8787}
QX_RPC_ENDPOINT=${QX_RPC_URL:-https://api.devnet.solana.com}
QX_PROGRAM_ID=${QX_PROGRAM_ID:-QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB}

QX_SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
QX_REPO_ROOT=$(cd "$QX_SCRIPT_DIRECTORY/.." && pwd)
QX_ARTIFACT="$QX_REPO_ROOT/external/models/granite-m5-pro-q4-main.qx"
QX_GATEWAY_URL="http://127.0.0.1:$QX_PORT"

if [[ ! "$QX_DURATION_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "QX_DEMO_DURATION_SECONDS must be a positive integer" >&2
  exit 1
fi
if [[ ! "$QX_JOB_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "QX_DEMO_JOB_DELAY_SECONDS must be a nonnegative integer" >&2
  exit 1
fi
if [[ ! "$QX_MAX_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "QX_DEMO_MAX_OUTPUT_TOKENS must be a positive integer" >&2
  exit 1
fi
if [[ ! "$QX_AUDIT_PPM" =~ ^[0-9]+$ ]] \
  || (( QX_AUDIT_PPM < 10000 || QX_AUDIT_PPM > 1000000 )); then
  echo "QX_AUDIT_PROBABILITY_PPM must be between 10000 and 1000000" >&2
  exit 1
fi
if [[ "$QX_RPC_ENDPOINT" != "https://api.devnet.solana.com" ]]; then
  echo "the demo only permits Solana devnet: $QX_RPC_ENDPOINT" >&2
  exit 1
fi

for QX_COMMAND in cargo curl jq openssl; do
  if ! command -v "$QX_COMMAND" >/dev/null 2>&1; then
    echo "missing required command: $QX_COMMAND" >&2
    exit 1
  fi
done

QX_SOLANA=${QX_SOLANA:-$(command -v solana || true)}
QX_SOLANA_KEYGEN=${QX_SOLANA_KEYGEN:-$(command -v solana-keygen || true)}
if [[ -z "$QX_SOLANA" && -x /Users/blas/.local/share/solana/install/active_release/bin/solana ]]; then
  QX_SOLANA=/Users/blas/.local/share/solana/install/active_release/bin/solana
fi
if [[ -z "$QX_SOLANA_KEYGEN" && -x /Users/blas/.local/share/solana/install/active_release/bin/solana-keygen ]]; then
  QX_SOLANA_KEYGEN=/Users/blas/.local/share/solana/install/active_release/bin/solana-keygen
fi
if [[ -z "$QX_SOLANA" || -z "$QX_SOLANA_KEYGEN" ]]; then
  echo "Solana CLI is unavailable; set QX_SOLANA and QX_SOLANA_KEYGEN" >&2
  exit 1
fi
QX_CARGO_BUILD_SBF=$(command -v cargo-build-sbf || true)
if [[ -z "$QX_CARGO_BUILD_SBF" && -x "$(dirname "$QX_SOLANA")/cargo-build-sbf" ]]; then
  QX_CARGO_BUILD_SBF="$(dirname "$QX_SOLANA")/cargo-build-sbf"
fi

QX_RUN_DIRECTORY=$(mktemp -d "${TMPDIR:-/tmp}/qx-demo.XXXXXX")
QX_GATEWAY_PID=
QX_MINER_PID=

cleanup() {
  if [[ -n "$QX_MINER_PID" ]]; then
    kill "$QX_MINER_PID" 2>/dev/null || true
    wait "$QX_MINER_PID" 2>/dev/null || true
  fi
  if [[ -n "$QX_GATEWAY_PID" ]]; then
    kill "$QX_GATEWAY_PID" 2>/dev/null || true
    wait "$QX_GATEWAY_PID" 2>/dev/null || true
  fi
  if [[ "${QX_DEMO_KEEP_RUN:-0}" == 1 ]]; then
    echo "run_directory=$QX_RUN_DIRECTORY"
  elif [[ -d "$QX_RUN_DIRECTORY" ]]; then
    find "$QX_RUN_DIRECTORY" -type f -delete
    find "$QX_RUN_DIRECTORY" -depth -type d -exec rmdir {} \;
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$QX_RUN_DIRECTORY/gateway-data"
openssl rand -hex 32 -out "$QX_RUN_DIRECTORY/miner.token"
chmod 600 "$QX_RUN_DIRECTORY/miner.token"
"$QX_SOLANA_KEYGEN" new --no-bip39-passphrase --silent \
  --outfile "$QX_RUN_DIRECTORY/requester.json"
"$QX_SOLANA_KEYGEN" new --no-bip39-passphrase --silent \
  --outfile "$QX_RUN_DIRECTORY/miner.json"

QX_GENERATED_PAYER=0
if [[ -n "${QX_PAYER_KEYPAIR:-}" ]]; then
  if [[ ! -f "$QX_PAYER_KEYPAIR" ]]; then
    echo "QX_PAYER_KEYPAIR does not exist: $QX_PAYER_KEYPAIR" >&2
    exit 1
  fi
  QX_PAYER=$(cd "$(dirname "$QX_PAYER_KEYPAIR")" && pwd)/$(basename "$QX_PAYER_KEYPAIR")
else
  QX_CONFIGURED_PAYER=$("$QX_SOLANA" config get 2>/dev/null \
    | sed -n 's/^Keypair Path:[[:space:]]*//p' \
    | sed 's/[[:space:]]*$//')
  if [[ -n "$QX_CONFIGURED_PAYER" && -f "$QX_CONFIGURED_PAYER" ]]; then
    QX_PAYER="$QX_CONFIGURED_PAYER"
    echo "event=using_configured_devnet_payer pubkey=$("$QX_SOLANA_KEYGEN" pubkey "$QX_PAYER")"
  else
    QX_PAYER="$QX_RUN_DIRECTORY/payer.json"
    QX_GENERATED_PAYER=1
    "$QX_SOLANA_KEYGEN" new --no-bip39-passphrase --silent --outfile "$QX_PAYER"
  fi
fi

if [[ "$QX_GENERATED_PAYER" == 1 ]]; then
  QX_PAYER_PUBKEY=$("$QX_SOLANA_KEYGEN" pubkey "$QX_PAYER")
  echo "event=funding_devnet_payer pubkey=$QX_PAYER_PUBKEY"
  QX_FUNDED=0
  for QX_ATTEMPT in $(seq 1 5); do
    if "$QX_SOLANA" airdrop 1 "$QX_PAYER_PUBKEY" --url "$QX_RPC_ENDPOINT"; then
      QX_FUNDED=1
      break
    fi
    echo "event=faucet_retry attempt=$QX_ATTEMPT" >&2
    sleep $((QX_ATTEMPT * 2))
  done
  if [[ "$QX_FUNDED" != 1 ]]; then
    echo "devnet faucet unavailable; set QX_PAYER_KEYPAIR to a funded devnet keypair" >&2
    exit 1
  fi
fi

QX_MINER_PUBKEY=$("$QX_SOLANA_KEYGEN" pubkey "$QX_RUN_DIRECTORY/miner.json")
QX_EPOCH=$(date +%s)

cd "$QX_REPO_ROOT"
if [[ ! -f "$QX_ARTIFACT" ]]; then
  ./scripts/build-artifact.sh
fi
cargo build --release -p qx-gateway

echo "event=demo_start duration_seconds=$QX_DURATION_SECONDS audit_probability_ppm=$QX_AUDIT_PPM"
echo "event=epoch_open epoch=$QX_EPOCH miner=$QX_MINER_PUBKEY"

target/release/qx-gateway serve \
  --listen "127.0.0.1:$QX_PORT" \
  --data-dir "$QX_RUN_DIRECTORY/gateway-data" \
  --program "$QX_PROGRAM_ID" \
  --rpc-url "$QX_RPC_ENDPOINT" \
  --miner-token-file "$QX_RUN_DIRECTORY/miner.token" &
QX_GATEWAY_PID=$!

for _ in $(seq 1 60); do
  if curl --fail --silent "$QX_GATEWAY_URL/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --fail --silent "$QX_GATEWAY_URL/health" >/dev/null

QX_GATEWAY_URL="$QX_GATEWAY_URL" ./scripts/devnet-miner-worker.sh \
  "$QX_MINER_PUBKEY" \
  "$QX_ARTIFACT" \
  "$QX_RUN_DIRECTORY/miner.token" \
  1 &
QX_MINER_PID=$!

QX_STARTED_AT=$(date +%s)
QX_STOP_AT=$((QX_STARTED_AT + QX_DURATION_SECONDS))
QX_JOBS=0
QX_COMPLETED=0
QX_FAILED=0

while (( $(date +%s) < QX_STOP_AT )); do
  QX_JOBS=$((QX_JOBS + 1))
  QX_INPUT_TOKENS="1,2,$((3 + QX_JOBS % 100))"
  echo "event=job_start sequence=$QX_JOBS input_tokens=$QX_INPUT_TOKENS"
  if QX_GATEWAY_URL="$QX_GATEWAY_URL" \
    QX_ASSIGNED_EPOCH="$QX_EPOCH" \
    QX_AUDIT_PROBABILITY_PPM="$QX_AUDIT_PPM" \
    ./scripts/devnet-client-test.sh \
      "$QX_RUN_DIRECTORY/requester.json" \
      "$QX_MINER_PUBKEY" \
      "$QX_INPUT_TOKENS" \
      "$QX_MAX_OUTPUT_TOKENS" \
      120; then
    QX_COMPLETED=$((QX_COMPLETED + 1))
  else
    QX_FAILED=$((QX_FAILED + 1))
  fi
  if (( QX_JOB_DELAY_SECONDS > 0 )); then
    sleep "$QX_JOB_DELAY_SECONDS"
  fi
done

QX_TOKEN=$(tr -d '\r\n' < "$QX_RUN_DIRECTORY/miner.token")
curl --fail --silent --show-error \
  -H "X-QX-Miner-Token: $QX_TOKEN" \
  -H 'Content-Type: application/json' \
  --data "{\"miner_pubkey\":\"$QX_MINER_PUBKEY\"}" \
  "$QX_GATEWAY_URL/v1/epochs/$QX_EPOCH/close" \
  > "$QX_RUN_DIRECTORY/epoch.json"
echo "event=epoch_closed $(jq -r '"result_count=\(.result_count) result_root=\(.result_root_hex)"' "$QX_RUN_DIRECTORY/epoch.json")"

QX_SUBMIT_OUTPUT=$(target/release/qx-gateway submit-epoch \
  --data-dir "$QX_RUN_DIRECTORY/gateway-data" \
  --epoch "$QX_EPOCH" \
  --program "$QX_PROGRAM_ID" \
  --rpc-url "$QX_RPC_ENDPOINT" \
  --payer-keypair "$QX_PAYER" \
  --miner-keypair "$QX_RUN_DIRECTORY/miner.json" \
  --epoch-keypair "$QX_RUN_DIRECTORY/epoch-keypair.json")
echo "$QX_SUBMIT_OUTPUT"
QX_EPOCH_ACCOUNT=$(sed -n 's/^epoch_account=//p' <<< "$QX_SUBMIT_OUTPUT")
if [[ -z "$QX_EPOCH_ACCOUNT" ]]; then
  echo "could not read submitted epoch account" >&2
  exit 1
fi

QX_FINALIZED=0
for QX_ATTEMPT in $(seq 1 30); do
  if QX_FINALIZE_OUTPUT=$(target/release/qx-gateway finalize-epoch \
    --program "$QX_PROGRAM_ID" \
    --rpc-url "$QX_RPC_ENDPOINT" \
    --payer-keypair "$QX_PAYER" \
    --epoch-account "$QX_EPOCH_ACCOUNT" 2>&1); then
    echo "$QX_FINALIZE_OUTPUT"
    QX_FINALIZED=1
    break
  fi
  echo "event=waiting_for_successor_block attempt=$QX_ATTEMPT"
  sleep 1
done
if [[ "$QX_FINALIZED" != 1 ]]; then
  echo "epoch did not finalize after 30 attempts" >&2
  exit 1
fi

curl --fail --silent --show-error \
  -H "X-QX-Miner-Token: $QX_TOKEN" \
  -H 'Content-Type: application/json' \
  --data "{\"epoch_account\":\"$QX_EPOCH_ACCOUNT\"}" \
  "$QX_GATEWAY_URL/v1/epochs/$QX_EPOCH/sync" \
  > "$QX_RUN_DIRECTORY/synced-epoch.json"
echo "event=epoch_seeded $(jq -r '"audit_seed=\(.onchain.audit_seed_hex) challenge_deadline_slot=\(.onchain.challenge_deadline_slot)"' "$QX_RUN_DIRECTORY/synced-epoch.json")"

curl --fail --silent --show-error \
  "$QX_GATEWAY_URL/v1/epochs/$QX_EPOCH/audits" \
  > "$QX_RUN_DIRECTORY/audits.json"
QX_CHALLENGES=$(jq 'length' "$QX_RUN_DIRECTORY/audits.json")
QX_REPLAY_MATCHES=0

for ((QX_INDEX = 0; QX_INDEX < QX_CHALLENGES; QX_INDEX++)); do
  QX_JOB_FILE="$QX_RUN_DIRECTORY/audit-$QX_INDEX.job.json"
  QX_REPLAY_FILE="$QX_RUN_DIRECTORY/audit-$QX_INDEX.replay.json"
  jq ".[$QX_INDEX] | {payload_hex, input_tokens, requester_signature_hex}" \
    "$QX_RUN_DIRECTORY/audits.json" > "$QX_JOB_FILE"
  QX_JOB_HASH=$(jq -r ".[$QX_INDEX].result_hex[0:64]" "$QX_RUN_DIRECTORY/audits.json")
  QX_PROOF_NODES=$(jq ".[$QX_INDEX].result_proof_siblings_hex | length" "$QX_RUN_DIRECTORY/audits.json")
  echo "event=audit_selected index=$QX_INDEX job_hash=$QX_JOB_HASH proof_nodes=$QX_PROOF_NODES"
  target/release/qx-gateway run-job \
    --job "$QX_JOB_FILE" \
    --artifact "$QX_ARTIFACT" \
    --checkpoint-interval 2 \
    --context 512 \
    --output "$QX_REPLAY_FILE"
  if [[ "$(jq -r '.result_hex' "$QX_REPLAY_FILE")" \
      == "$(jq -r ".[$QX_INDEX].result_hex" "$QX_RUN_DIRECTORY/audits.json")" ]]; then
    QX_REPLAY_MATCHES=$((QX_REPLAY_MATCHES + 1))
    echo "event=audit_replay status=match job_hash=$QX_JOB_HASH output_tokens=$(jq -c '.output_tokens' "$QX_REPLAY_FILE")"
  else
    echo "event=audit_replay status=disagreement job_hash=$QX_JOB_HASH" >&2
  fi
done

QX_SBF_STATUS=skipped
if [[ -n "$QX_CARGO_BUILD_SBF" ]]; then
  "$QX_CARGO_BUILD_SBF" --manifest-path programs/qx-terminal-verifier/Cargo.toml
  cargo test -p qx-terminal-verifier --test sbf_v2 \
    block_seeded_unsigned_result_is_bisected_and_convicted_on_sbf \
    -- --nocapture
  QX_SBF_STATUS=passed
fi

echo "event=demo_complete jobs=$QX_JOBS completed=$QX_COMPLETED failed=$QX_FAILED selected_audits=$QX_CHALLENGES replay_matches=$QX_REPLAY_MATCHES adversarial_sbf=$QX_SBF_STATUS"
