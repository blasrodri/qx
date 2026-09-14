#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <requester-keypair.json> <miner-pubkey> <input-token-csv> [max-output-tokens] [wait-seconds]" >&2
  echo "set QX_GATEWAY_URL to override the public devnet gateway" >&2
}

if [[ $# -lt 3 || $# -gt 5 ]]; then
  usage
  exit 2
fi

QX_REQUESTER_KEYPAIR=$1
QX_MINER_PUBKEY=$2
QX_INPUT_TOKENS=$3
QX_MAX_OUTPUT_TOKENS=${4:-8}
QX_WAIT_SECONDS=${5:-300}
QX_GATEWAY_ENDPOINT=${QX_GATEWAY_URL:-https://qx-gateway-devnet-sitoula-8080.ignite.dodil.cloud}
QX_RPC_ENDPOINT=${QX_RPC_URL:-https://api.devnet.solana.com}
QX_PROGRAM_ID=${QX_PROGRAM_ID:-QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB}
QX_MODEL_ID=${QX_MODEL_ID:-bc205dbf40be32428fa653aff4c690b2c8a379a2d3fedbf617bb831a6264a17d}
QX_EXECUTION_SPEC=${QX_EXECUTION_SPEC:-2485240f3be98d158f5fca212e98e6776254f7279e772c8b12daa6ee9bec6f5f}
QX_MODEL_REGISTRY=${QX_MODEL_REGISTRY:-CsUgpxaXfofqfEVS1DbTR8pcGqC4mCWXhR15kgiDoCZe}
QX_AUDIT_PROBABILITY_PPM=${QX_AUDIT_PROBABILITY_PPM:-50000}

for QX_REQUIRED_COMMAND in cargo curl jq openssl; do
  if ! command -v "$QX_REQUIRED_COMMAND" >/dev/null 2>&1; then
    echo "required command is unavailable: $QX_REQUIRED_COMMAND" >&2
    exit 1
  fi
done
if [[ ! -f "$QX_REQUESTER_KEYPAIR" ]]; then
  echo "requester keypair does not exist: $QX_REQUESTER_KEYPAIR" >&2
  exit 1
fi
if [[ ! "$QX_INPUT_TOKENS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "input-token-csv must contain comma-separated unsigned integers" >&2
  exit 1
fi
if [[ ! "$QX_MAX_OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ || ! "$QX_WAIT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "max-output-tokens must be positive and wait-seconds must be nonnegative" >&2
  exit 1
fi
if [[ ! "$QX_AUDIT_PROBABILITY_PPM" =~ ^[0-9]+$ ]] \
  || (( QX_AUDIT_PROBABILITY_PPM < 10000 || QX_AUDIT_PROBABILITY_PPM > 1000000 )); then
  echo "QX_AUDIT_PROBABILITY_PPM must be between 10000 and 1000000" >&2
  exit 1
fi

QX_WORK_DIRECTORY=$(mktemp -d "${TMPDIR:-/tmp}/qx-client-test.XXXXXX")
cleanup() {
  find "$QX_WORK_DIRECTORY" -type f -delete
  rmdir "$QX_WORK_DIRECTORY"
}
trap cleanup EXIT
umask 077

QX_JOB_FILE="$QX_WORK_DIRECTORY/job.json"
QX_RESULT_FILE="$QX_WORK_DIRECTORY/result.json"
QX_JOB_ID=$(openssl rand -hex 32)
QX_NOW=$(date +%s)
QX_EPOCH=${QX_ASSIGNED_EPOCH:-$((QX_NOW / 3600))}
if [[ ! "$QX_EPOCH" =~ ^[0-9]+$ ]]; then
  echo "QX_ASSIGNED_EPOCH must be an unsigned integer" >&2
  exit 1
fi
QX_NONCE=$QX_NOW
QX_CURRENT_SLOT=$(curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  --data '{"jsonrpc":"2.0","id":1,"method":"getSlot","params":[{"commitment":"confirmed"}]}' \
  "$QX_RPC_ENDPOINT" | jq -er '.result')
QX_DEADLINE_SLOT=$((QX_CURRENT_SLOT + 18000))

cargo build --release -p qx-gateway
QX_CREATE_OUTPUT=$(target/release/qx-gateway create-job \
  --program "$QX_PROGRAM_ID" \
  --requester-keypair "$QX_REQUESTER_KEYPAIR" \
  --job-id-hex "$QX_JOB_ID" \
  --assigned-miner "$QX_MINER_PUBKEY" \
  --model-id-hex "$QX_MODEL_ID" \
  --execution-spec-hex "$QX_EXECUTION_SPEC" \
  --model-registry "$QX_MODEL_REGISTRY" \
  --input-tokens "$QX_INPUT_TOKENS" \
  --max-output-tokens "$QX_MAX_OUTPUT_TOKENS" \
  --slashable-exposure 1 \
  --deadline-slot "$QX_DEADLINE_SLOT" \
  --nonce "$QX_NONCE" \
  --audit-probability-ppm "$QX_AUDIT_PROBABILITY_PPM" \
  --assigned-epoch "$QX_EPOCH" \
  --output "$QX_JOB_FILE")
QX_JOB_HASH=$(sed -n 's/^job_hash_hex=//p' <<< "$QX_CREATE_OUTPUT")
if [[ ! "$QX_JOB_HASH" =~ ^[0-9a-f]{64}$ ]]; then
  echo "could not read the generated job hash" >&2
  exit 1
fi

curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary "@$QX_JOB_FILE" \
  "$QX_GATEWAY_ENDPOINT/v1/jobs" >/dev/null

echo "gateway=$QX_GATEWAY_ENDPOINT"
echo "epoch=$QX_EPOCH"
echo "job_hash=$QX_JOB_HASH"
echo "job_status=submitted"

if [[ "$QX_WAIT_SECONDS" == 0 ]]; then
  exit 0
fi

QX_STOP_AT=$((QX_NOW + QX_WAIT_SECONDS))
while (( $(date +%s) < QX_STOP_AT )); do
  if ! QX_HTTP_STATUS=$(curl --silent --show-error \
      --output "$QX_RESULT_FILE" \
      --write-out '%{http_code}' \
      "$QX_GATEWAY_ENDPOINT/v1/results/$QX_JOB_HASH"); then
    sleep 2
    continue
  fi
  if [[ "$QX_HTTP_STATUS" == 200 ]]; then
    echo "job_status=completed"
    echo "output_tokens=$(jq -c '.output_tokens' "$QX_RESULT_FILE")"
    echo "result_leaf_hex=$(jq -r '.result_hex' "$QX_RESULT_FILE")"
    exit 0
  fi
  if [[ "$QX_HTTP_STATUS" != 404 ]]; then
    echo "gateway returned HTTP $QX_HTTP_STATUS while waiting for the result" >&2
    jq -c . "$QX_RESULT_FILE" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "timed out waiting for result; job_hash=$QX_JOB_HASH" >&2
exit 1
