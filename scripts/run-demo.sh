#!/usr/bin/env bash
set -euo pipefail

QX_DURATION_SECONDS=${QX_DEMO_DURATION_SECONDS:-1800}
QX_EPOCH_SECONDS=${QX_DEMO_EPOCH_SECONDS:-30}
QX_SUBMISSION_GUARD_SECONDS=${QX_DEMO_SUBMISSION_GUARD_SECONDS:-5}
QX_JOB_DELAY_SECONDS=${QX_DEMO_JOB_DELAY_SECONDS:-0}
QX_MAX_OUTPUT_TOKENS=${QX_DEMO_MAX_OUTPUT_TOKENS:-4}
QX_AUDIT_PPM=${QX_AUDIT_PROBABILITY_PPM:-50000}
QX_PORT=${QX_DEMO_PORT:-8787}
QX_RPC_ENDPOINT=${QX_RPC_URL:-https://api.devnet.solana.com}
QX_PROGRAM_ID=${QX_PROGRAM_ID:-QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB}
QX_REQUIRE_SBF=${QX_DEMO_REQUIRE_SBF:-1}
QX_REQUIRE_AUDIT=${QX_DEMO_REQUIRE_AUDIT:-1}
QX_REQUIRE_DEVNET_ADVERSARY=${QX_DEMO_REQUIRE_DEVNET_ADVERSARY:-1}

QX_SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
QX_REPO_ROOT=$(cd "$QX_SCRIPT_DIRECTORY/.." && pwd)
QX_ARTIFACT="$QX_REPO_ROOT/external/models/granite-m5-pro-q4-main.qx"
QX_GATEWAY_URL="http://127.0.0.1:$QX_PORT"
QX_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
QX_EVIDENCE_DIRECTORY=${QX_DEMO_EVIDENCE_DIR:-$QX_REPO_ROOT/.qx-demo-runs/$QX_RUN_ID}

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

for QX_VALUE in "$QX_DURATION_SECONDS" "$QX_EPOCH_SECONDS" \
  "$QX_SUBMISSION_GUARD_SECONDS" "$QX_MAX_OUTPUT_TOKENS"; do
  if ! positive_integer "$QX_VALUE"; then
    echo "duration, epoch, guard, and output-token values must be positive integers" >&2
    exit 1
  fi
done
if [[ ! "$QX_JOB_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "QX_DEMO_JOB_DELAY_SECONDS must be a nonnegative integer" >&2
  exit 1
fi
if (( QX_DURATION_SECONDS < QX_EPOCH_SECONDS \
  || QX_DURATION_SECONDS % QX_EPOCH_SECONDS != 0 )); then
  echo "QX_DEMO_DURATION_SECONDS must be a positive multiple of QX_DEMO_EPOCH_SECONDS" >&2
  exit 1
fi
if (( QX_SUBMISSION_GUARD_SECONDS >= QX_EPOCH_SECONDS )); then
  echo "the submission guard must be shorter than an epoch" >&2
  exit 1
fi
if [[ ! "$QX_AUDIT_PPM" =~ ^[0-9]+$ ]] \
  || (( QX_AUDIT_PPM < 10000 || QX_AUDIT_PPM > 1000000 )); then
  echo "QX_AUDIT_PROBABILITY_PPM must be between 10000 and 1000000" >&2
  exit 1
fi
if [[ ! "$QX_REQUIRE_SBF" =~ ^[01]$ \
  || ! "$QX_REQUIRE_AUDIT" =~ ^[01]$ \
  || ! "$QX_REQUIRE_DEVNET_ADVERSARY" =~ ^[01]$ ]]; then
  echo "QX_DEMO_REQUIRE_SBF, QX_DEMO_REQUIRE_AUDIT, and QX_DEMO_REQUIRE_DEVNET_ADVERSARY must be 0 or 1" >&2
  exit 1
fi
if [[ "$QX_RPC_ENDPOINT" != "https://api.devnet.solana.com" ]]; then
  echo "the demo only permits Solana devnet: $QX_RPC_ENDPOINT" >&2
  exit 1
fi

for QX_COMMAND in cargo curl jq openssl tee; do
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
if [[ "$QX_REQUIRE_SBF" == 1 && -z "$QX_CARGO_BUILD_SBF" ]]; then
  echo "cargo-build-sbf is required for the complete adversarial demo" >&2
  exit 1
fi

mkdir -p "$QX_EVIDENCE_DIRECTORY"
exec > >(tee "$QX_EVIDENCE_DIRECTORY/demo.log") 2>&1

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
  if [[ "${QX_DEMO_KEEP_PRIVATE_WORK:-0}" == 1 ]]; then
    echo "private_run_directory=$QX_RUN_DIRECTORY"
  elif [[ -d "$QX_RUN_DIRECTORY" ]]; then
    find "$QX_RUN_DIRECTORY" -type f -delete
    find "$QX_RUN_DIRECTORY" -depth -type d -exec rmdir {} \;
  fi
  echo "evidence_directory=$QX_EVIDENCE_DIRECTORY"
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
QX_EPOCH_BASE=$(date +%s)
QX_EPOCH_COUNT=$((QX_DURATION_SECONDS / QX_EPOCH_SECONDS))

cd "$QX_REPO_ROOT"
if [[ ! -f "$QX_ARTIFACT" ]]; then
  ./scripts/build-artifact.sh
fi
cargo build --release -p qx-gateway -p qx-devnet-adversary
cargo test -p qx-commit -p qx-consensus-math -p qx-verifier-core -p qx-gateway
echo "event=protocol_unit_tests status=passed"

if [[ -n "$QX_CARGO_BUILD_SBF" ]]; then
  "$QX_CARGO_BUILD_SBF" --manifest-path programs/qx-terminal-verifier/Cargo.toml
  echo "event=sbf_build status=passed"
fi

echo "event=demo_start duration_seconds=$QX_DURATION_SECONDS epoch_seconds=$QX_EPOCH_SECONDS epochs=$QX_EPOCH_COUNT audit_probability_ppm=$QX_AUDIT_PPM"

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

QX_GATEWAY_URL="$QX_GATEWAY_URL" QX_SKIP_GATEWAY_BUILD=1 \
  ./scripts/devnet-miner-worker.sh \
    "$QX_MINER_PUBKEY" \
    "$QX_ARTIFACT" \
    "$QX_RUN_DIRECTORY/miner.token" \
    1 &
QX_MINER_PID=$!

QX_TOKEN=$(tr -d '\r\n' < "$QX_RUN_DIRECTORY/miner.token")
QX_DEMO_STARTED_AT=$(date +%s)
QX_JOBS=0
QX_CLIENT_COMPLETED=0
QX_FAILED=0
QX_COMMITTED=0
QX_EPOCHS=()

for ((QX_EPOCH_INDEX = 0; QX_EPOCH_INDEX < QX_EPOCH_COUNT; QX_EPOCH_INDEX++)); do
  QX_EPOCH=$((QX_EPOCH_BASE + QX_EPOCH_INDEX))
  QX_EPOCHS+=("$QX_EPOCH")
  QX_EPOCH_STOP=$((QX_DEMO_STARTED_AT + (QX_EPOCH_INDEX + 1) * QX_EPOCH_SECONDS))
  QX_EPOCH_JOBS=0
  QX_EPOCH_COMPLETED=0
  echo "event=epoch_open index=$QX_EPOCH_INDEX epoch=$QX_EPOCH miner=$QX_MINER_PUBKEY"

  while true; do
    QX_NOW=$(date +%s)
    QX_REMAINING=$((QX_EPOCH_STOP - QX_NOW))
    if (( QX_REMAINING <= QX_SUBMISSION_GUARD_SECONDS )); then
      break
    fi
    QX_JOBS=$((QX_JOBS + 1))
    QX_EPOCH_JOBS=$((QX_EPOCH_JOBS + 1))
    QX_INPUT_TOKENS="1,2,$((3 + QX_JOBS % 100))"
    QX_WAIT_SECONDS=$((QX_REMAINING - 1))
    echo "event=job_start sequence=$QX_JOBS epoch=$QX_EPOCH input_tokens=$QX_INPUT_TOKENS"
    if QX_CLIENT_OUTPUT=$(QX_GATEWAY_URL="$QX_GATEWAY_URL" \
      QX_ASSIGNED_EPOCH="$QX_EPOCH" \
      QX_AUDIT_PROBABILITY_PPM="$QX_AUDIT_PPM" \
      QX_SKIP_GATEWAY_BUILD=1 \
      ./scripts/devnet-client-test.sh \
        "$QX_RUN_DIRECTORY/requester.json" \
        "$QX_MINER_PUBKEY" \
        "$QX_INPUT_TOKENS" \
        "$QX_MAX_OUTPUT_TOKENS" \
        "$QX_WAIT_SECONDS" 2>&1); then
      echo "$QX_CLIENT_OUTPUT"
      QX_CLIENT_COMPLETED=$((QX_CLIENT_COMPLETED + 1))
      QX_EPOCH_COMPLETED=$((QX_EPOCH_COMPLETED + 1))
    else
      echo "$QX_CLIENT_OUTPUT" >&2
      QX_FAILED=$((QX_FAILED + 1))
      break
    fi
    if (( QX_JOB_DELAY_SECONDS > 0 )); then
      sleep "$QX_JOB_DELAY_SECONDS"
    fi
  done

  QX_NOW=$(date +%s)
  if (( QX_NOW < QX_EPOCH_STOP )); then
    sleep $((QX_EPOCH_STOP - QX_NOW))
  fi

  QX_EPOCH_FILE="$QX_RUN_DIRECTORY/epoch-$QX_EPOCH.json"
  curl --fail --silent --show-error \
    -H "X-QX-Miner-Token: $QX_TOKEN" \
    -H 'Content-Type: application/json' \
    --data "{\"miner_pubkey\":\"$QX_MINER_PUBKEY\"}" \
    "$QX_GATEWAY_URL/v1/epochs/$QX_EPOCH/close" > "$QX_EPOCH_FILE"
  cp "$QX_EPOCH_FILE" "$QX_EVIDENCE_DIRECTORY/epoch-$QX_EPOCH.closed.json"
  QX_EPOCH_COMMITTED=$(jq -er '.result_count' "$QX_EPOCH_FILE")
  QX_COMMITTED=$((QX_COMMITTED + QX_EPOCH_COMMITTED))
  echo "event=epoch_closed index=$QX_EPOCH_INDEX epoch=$QX_EPOCH submitted_jobs=$QX_EPOCH_JOBS client_completed=$QX_EPOCH_COMPLETED result_count=$QX_EPOCH_COMMITTED result_root=$(jq -r '.result_root_hex' "$QX_EPOCH_FILE")"
done

QX_CHALLENGES=0
QX_PACKAGES_VERIFIED=0
QX_REPLAY_MATCHES=0
QX_SETTLED_EPOCHS=0

for QX_EPOCH in "${QX_EPOCHS[@]}"; do
  QX_SUBMITTED=0
  for QX_ATTEMPT in $(seq 1 8); do
    if QX_SUBMIT_OUTPUT=$(target/release/qx-gateway submit-epoch \
      --data-dir "$QX_RUN_DIRECTORY/gateway-data" \
      --epoch "$QX_EPOCH" \
      --program "$QX_PROGRAM_ID" \
      --rpc-url "$QX_RPC_ENDPOINT" \
      --payer-keypair "$QX_PAYER" \
      --miner-keypair "$QX_RUN_DIRECTORY/miner.json" \
      --epoch-keypair "$QX_RUN_DIRECTORY/epoch-$QX_EPOCH-keypair.json" 2>&1); then
      echo "$QX_SUBMIT_OUTPUT"
      QX_SUBMITTED=1
      break
    fi
    echo "$QX_SUBMIT_OUTPUT" >&2
    echo "event=epoch_submit_retry epoch=$QX_EPOCH attempt=$QX_ATTEMPT" >&2
    sleep "$QX_ATTEMPT"
  done
  if [[ "$QX_SUBMITTED" != 1 ]]; then
    echo "epoch submission failed: $QX_EPOCH" >&2
    exit 1
  fi
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
    echo "$QX_FINALIZE_OUTPUT" >&2
    echo "event=epoch_finalize_retry epoch=$QX_EPOCH attempt=$QX_ATTEMPT" >&2
    sleep 1
  done
  if [[ "$QX_FINALIZED" != 1 ]]; then
    echo "epoch did not finalize after 30 attempts: $QX_EPOCH" >&2
    exit 1
  fi

  QX_RESUBMIT_OUTPUT=$(target/release/qx-gateway submit-epoch \
    --data-dir "$QX_RUN_DIRECTORY/gateway-data" \
    --epoch "$QX_EPOCH" \
    --program "$QX_PROGRAM_ID" \
    --rpc-url "$QX_RPC_ENDPOINT" \
    --payer-keypair "$QX_PAYER" \
    --miner-keypair "$QX_RUN_DIRECTORY/miner.json" \
    --epoch-keypair "$QX_RUN_DIRECTORY/epoch-$QX_EPOCH-keypair.json")
  if ! grep -q '^submission_status=already_seeded$' <<< "$QX_RESUBMIT_OUTPUT"; then
    echo "idempotent epoch resubmission did not observe the seeded account" >&2
    echo "$QX_RESUBMIT_OUTPUT" >&2
    exit 1
  fi
  QX_REFINALIZE_OUTPUT=$(target/release/qx-gateway finalize-epoch \
    --program "$QX_PROGRAM_ID" \
    --rpc-url "$QX_RPC_ENDPOINT" \
    --payer-keypair "$QX_PAYER" \
    --epoch-account "$QX_EPOCH_ACCOUNT")
  if ! grep -q '^finalization_status=already_seeded$' <<< "$QX_REFINALIZE_OUTPUT"; then
    echo "idempotent epoch refinalization did not observe the seeded account" >&2
    echo "$QX_REFINALIZE_OUTPUT" >&2
    exit 1
  fi
  echo "event=epoch_idempotence epoch=$QX_EPOCH status=passed"

  QX_SYNCED_FILE="$QX_RUN_DIRECTORY/epoch-$QX_EPOCH.synced.json"
  curl --fail --silent --show-error \
    -H "X-QX-Miner-Token: $QX_TOKEN" \
    -H 'Content-Type: application/json' \
    --data "{\"epoch_account\":\"$QX_EPOCH_ACCOUNT\"}" \
    "$QX_GATEWAY_URL/v1/epochs/$QX_EPOCH/sync" > "$QX_SYNCED_FILE"
  cp "$QX_SYNCED_FILE" "$QX_EVIDENCE_DIRECTORY/epoch-$QX_EPOCH.synced.json"
  echo "event=epoch_seeded epoch=$QX_EPOCH epoch_account=$QX_EPOCH_ACCOUNT $(jq -r '"audit_seed=\(.onchain.audit_seed_hex) entropy_slot=\(.onchain.entropy_slot) challenge_deadline_slot=\(.onchain.challenge_deadline_slot)"' "$QX_SYNCED_FILE")"
  QX_SETTLED_EPOCHS=$((QX_SETTLED_EPOCHS + 1))

  QX_AUDITS_FILE="$QX_RUN_DIRECTORY/epoch-$QX_EPOCH.audits.json"
  curl --fail --silent --show-error \
    "$QX_GATEWAY_URL/v1/epochs/$QX_EPOCH/audits" > "$QX_AUDITS_FILE"
  cp "$QX_AUDITS_FILE" "$QX_EVIDENCE_DIRECTORY/epoch-$QX_EPOCH.audits.json"
  QX_EPOCH_CHALLENGES=$(jq -er 'length' "$QX_AUDITS_FILE")
  echo "event=epoch_audits epoch=$QX_EPOCH selected=$QX_EPOCH_CHALLENGES"

  for ((QX_INDEX = 0; QX_INDEX < QX_EPOCH_CHALLENGES; QX_INDEX++)); do
    QX_PACKAGE_FILE="$QX_RUN_DIRECTORY/epoch-$QX_EPOCH-audit-$QX_INDEX.package.json"
    QX_JOB_FILE="$QX_RUN_DIRECTORY/epoch-$QX_EPOCH-audit-$QX_INDEX.job.json"
    QX_REPLAY_FILE="$QX_RUN_DIRECTORY/epoch-$QX_EPOCH-audit-$QX_INDEX.replay.json"
    jq ".[$QX_INDEX]" "$QX_AUDITS_FILE" > "$QX_PACKAGE_FILE"
    jq '{payload_hex, input_tokens, requester_signature_hex}' \
      "$QX_PACKAGE_FILE" > "$QX_JOB_FILE"
    QX_JOB_HASH=$(jq -r '.result_hex[0:64]' "$QX_PACKAGE_FILE")
    QX_PROOF_NODES=$(jq -r '.result_proof_siblings_hex | length' "$QX_PACKAGE_FILE")
    echo "event=audit_selected epoch=$QX_EPOCH index=$QX_INDEX job_hash=$QX_JOB_HASH proof_nodes=$QX_PROOF_NODES"

    target/release/qx-gateway verify-challenge \
      --package "$QX_PACKAGE_FILE" \
      --rpc-url "$QX_RPC_ENDPOINT"
    QX_PACKAGES_VERIFIED=$((QX_PACKAGES_VERIFIED + 1))

    target/release/qx-gateway run-job \
      --job "$QX_JOB_FILE" \
      --artifact "$QX_ARTIFACT" \
      --checkpoint-interval 2 \
      --context 512 \
      --output "$QX_REPLAY_FILE"
    if [[ "$(jq -r '.result_hex' "$QX_REPLAY_FILE")" \
        == "$(jq -r '.result_hex' "$QX_PACKAGE_FILE")" ]]; then
      QX_REPLAY_MATCHES=$((QX_REPLAY_MATCHES + 1))
      echo "event=audit_replay status=match epoch=$QX_EPOCH job_hash=$QX_JOB_HASH output_tokens=$(jq -c '.output_tokens' "$QX_REPLAY_FILE")"
    else
      echo "event=audit_replay status=disagreement epoch=$QX_EPOCH job_hash=$QX_JOB_HASH" >&2
    fi
    cp "$QX_PACKAGE_FILE" "$QX_EVIDENCE_DIRECTORY/epoch-$QX_EPOCH-audit-$QX_INDEX.package.json"
    cp "$QX_REPLAY_FILE" "$QX_EVIDENCE_DIRECTORY/epoch-$QX_EPOCH-audit-$QX_INDEX.replay.json"
    QX_CHALLENGES=$((QX_CHALLENGES + 1))
  done
done

QX_SBF_STATUS=not_requested
if [[ -n "$QX_CARGO_BUILD_SBF" ]]; then
  cargo test -p qx-terminal-verifier --test sbf_v2 \
    block_seeded_unsigned_result_is_bisected_and_convicted_on_sbf \
    -- --nocapture
  cargo test -p qx-terminal-verifier --test sbf_v2 \
    granite_output_and_trace_bounds_are_packet_safe_and_admitted_on_sbf \
    -- --nocapture
  QX_SBF_STATUS=passed
  echo "event=adversarial_sbf status=passed"
fi

QX_DEVNET_ADVERSARY_STATUS=not_requested
QX_DEVNET_ADVERSARY_CU=0
if [[ "$QX_REQUIRE_DEVNET_ADVERSARY" == 1 ]]; then
  QX_DEVNET_ADVERSARY_EVIDENCE="$QX_EVIDENCE_DIRECTORY/devnet-adversary.json"
  target/release/qx-devnet-adversary \
    --program "$QX_PROGRAM_ID" \
    --rpc-url "$QX_RPC_ENDPOINT" \
    --payer-keypair "$QX_PAYER" \
    --output "$QX_DEVNET_ADVERSARY_EVIDENCE"
  QX_DEVNET_ADVERSARY_STATUS=$(jq -er '.status' "$QX_DEVNET_ADVERSARY_EVIDENCE")
  QX_DEVNET_ADVERSARY_CU=$(jq -er '.total_simulated_compute_units' "$QX_DEVNET_ADVERSARY_EVIDENCE")
  if [[ "$(jq -er '.winner' "$QX_DEVNET_ADVERSARY_EVIDENCE")" != challenger ]]; then
    echo "live devnet dispute did not convict the dishonest miner" >&2
    exit 1
  fi
  echo "event=adversarial_devnet status=$QX_DEVNET_ADVERSARY_STATUS winner=challenger total_simulated_compute_units=$QX_DEVNET_ADVERSARY_CU"
fi

QX_STATUS=passed
if (( QX_FAILED != 0 \
  || QX_CLIENT_COMPLETED != QX_COMMITTED \
  || QX_SETTLED_EPOCHS != QX_EPOCH_COUNT \
  || QX_PACKAGES_VERIFIED != QX_CHALLENGES \
  || QX_REPLAY_MATCHES != QX_CHALLENGES )); then
  QX_STATUS=failed
fi
if [[ "$QX_REQUIRE_AUDIT" == 1 && "$QX_CHALLENGES" == 0 ]]; then
  QX_STATUS=failed
fi
if [[ "$QX_REQUIRE_SBF" == 1 && "$QX_SBF_STATUS" != passed ]]; then
  QX_STATUS=failed
fi
if [[ "$QX_REQUIRE_DEVNET_ADVERSARY" == 1 \
  && "$QX_DEVNET_ADVERSARY_STATUS" != passed ]]; then
  QX_STATUS=failed
fi

jq -n \
  --arg status "$QX_STATUS" \
  --arg sbf_status "$QX_SBF_STATUS" \
  --arg devnet_adversary_status "$QX_DEVNET_ADVERSARY_STATUS" \
  --argjson devnet_adversary_cu "$QX_DEVNET_ADVERSARY_CU" \
  --argjson epoch_seconds "$QX_EPOCH_SECONDS" \
  --argjson epochs "$QX_EPOCH_COUNT" \
  --argjson jobs "$QX_JOBS" \
  --argjson completed "$QX_CLIENT_COMPLETED" \
  --argjson failed "$QX_FAILED" \
  --argjson committed "$QX_COMMITTED" \
  --argjson selected_audits "$QX_CHALLENGES" \
  --argjson packages_verified "$QX_PACKAGES_VERIFIED" \
  --argjson replay_matches "$QX_REPLAY_MATCHES" \
  '{status:$status, epoch_seconds:$epoch_seconds, epochs:$epochs, jobs:$jobs, completed:$completed, failed:$failed, committed:$committed, selected_audits:$selected_audits, packages_verified:$packages_verified, replay_matches:$replay_matches, adversarial_sbf:$sbf_status, adversarial_devnet:$devnet_adversary_status, adversarial_devnet_simulated_cu:$devnet_adversary_cu}' \
  > "$QX_EVIDENCE_DIRECTORY/summary.json"

echo "event=demo_complete status=$QX_STATUS epochs=$QX_EPOCH_COUNT epoch_seconds=$QX_EPOCH_SECONDS jobs=$QX_JOBS completed=$QX_CLIENT_COMPLETED failed=$QX_FAILED committed=$QX_COMMITTED selected_audits=$QX_CHALLENGES packages_verified=$QX_PACKAGES_VERIFIED replay_matches=$QX_REPLAY_MATCHES adversarial_sbf=$QX_SBF_STATUS adversarial_devnet=$QX_DEVNET_ADVERSARY_STATUS adversarial_devnet_simulated_cu=$QX_DEVNET_ADVERSARY_CU"
if [[ "$QX_STATUS" != passed ]]; then
  exit 1
fi
