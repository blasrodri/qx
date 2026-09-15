#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <miner-pubkey> <model.qx> <miner-token-file> [poll-seconds]" >&2
  echo "set QX_GATEWAY_URL to override the public devnet gateway" >&2
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
  usage
  exit 2
fi

QX_MINER_PUBKEY=$1
QX_MODEL_ARTIFACT=$2
QX_MINER_TOKEN_FILE=$3
QX_POLL_SECONDS=${4:-10}
QX_GATEWAY_ENDPOINT=${QX_GATEWAY_URL:-https://qx-gateway-devnet-sitoula-8080.ignite.dodil.cloud}

for QX_REQUIRED_COMMAND in cargo curl jq; do
  if ! command -v "$QX_REQUIRED_COMMAND" >/dev/null 2>&1; then
    echo "required command is unavailable: $QX_REQUIRED_COMMAND" >&2
    exit 1
  fi
done
if [[ ! -f "$QX_MODEL_ARTIFACT" ]]; then
  echo "model artifact does not exist: $QX_MODEL_ARTIFACT" >&2
  exit 1
fi
if [[ ! -f "$QX_MINER_TOKEN_FILE" ]]; then
  echo "miner transport token does not exist: $QX_MINER_TOKEN_FILE" >&2
  exit 1
fi
if [[ ! "$QX_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "poll-seconds must be a positive integer" >&2
  exit 1
fi

QX_WORK_DIRECTORY=$(mktemp -d "${TMPDIR:-/tmp}/qx-miner-worker.XXXXXX")
cleanup() {
  find "$QX_WORK_DIRECTORY" -type f -delete
  rmdir "$QX_WORK_DIRECTORY"
}
trap cleanup EXIT

umask 077
QX_AUTH_HEADER_FILE="$QX_WORK_DIRECTORY/miner-header"
printf 'X-QX-Miner-Token: %s\n' "$(tr -d '\r\n' < "$QX_MINER_TOKEN_FILE")" > "$QX_AUTH_HEADER_FILE"
if [[ $(wc -c < "$QX_AUTH_HEADER_FILE") -lt 54 ]]; then
  echo "miner transport token must contain at least 32 characters" >&2
  exit 1
fi

if [[ "${QX_SKIP_GATEWAY_BUILD:-0}" != 1 ]]; then
  cargo build --release -p qx-gateway
fi
QX_GATEWAY_BINARY=target/release/qx-gateway
if [[ ! -x "$QX_GATEWAY_BINARY" ]]; then
  echo "$QX_GATEWAY_BINARY is unavailable" >&2
  exit 1
fi

echo "gateway=$QX_GATEWAY_ENDPOINT"
echo "miner=$QX_MINER_PUBKEY"
echo "poll_seconds=$QX_POLL_SECONDS"
echo "worker_status=connected"

while true; do
  if ! QX_PENDING_JSON=$(curl --fail --silent --show-error --retry 3 \
    -H "@$QX_AUTH_HEADER_FILE" \
    "$QX_GATEWAY_ENDPOINT/v1/miners/$QX_MINER_PUBKEY/jobs"); then
    echo "worker_status=reconnecting" >&2
    sleep "$QX_POLL_SECONDS"
    continue
  fi

  while IFS= read -r QX_JOB_RECORD; do
    [[ -n "$QX_JOB_RECORD" ]] || continue
    QX_JOB_HASH=$(jq -er '.job_hash_hex' <<< "$QX_JOB_RECORD")
    QX_JOB_FILE="$QX_WORK_DIRECTORY/$QX_JOB_HASH.job.json"
    QX_RESULT_FILE="$QX_WORK_DIRECTORY/$QX_JOB_HASH.result.json"
    jq '{payload_hex, input_tokens, requester_signature_hex}' \
      <<< "$QX_JOB_RECORD" > "$QX_JOB_FILE"

    echo "job_status=running job_hash=$QX_JOB_HASH"
    if "$QX_GATEWAY_BINARY" run-job \
      --job "$QX_JOB_FILE" \
      --artifact "$QX_MODEL_ARTIFACT" \
      --checkpoint-interval 2 \
      --context 512 \
      --output "$QX_RESULT_FILE" \
      && curl --fail --silent --show-error --retry 3 \
        -H "@$QX_AUTH_HEADER_FILE" \
        -H 'Content-Type: application/json' \
        --data-binary "@$QX_RESULT_FILE" \
        "$QX_GATEWAY_ENDPOINT/v1/results" >/dev/null; then
      echo "job_status=completed job_hash=$QX_JOB_HASH output_tokens=$(jq -c '.output_tokens' "$QX_RESULT_FILE")"
    else
      echo "job_status=failed job_hash=$QX_JOB_HASH" >&2
    fi
    rm "$QX_JOB_FILE"
    [[ ! -f "$QX_RESULT_FILE" ]] || rm "$QX_RESULT_FILE"
  done < <(jq -c '.jobs[]' <<< "$QX_PENDING_JSON")

  sleep "$QX_POLL_SECONDS"
done
