#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <data-directory> <miner-token-file> [listen-address]" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

QX_GATEWAY_DATA=$1
QX_GATEWAY_TOKEN=$2
QX_GATEWAY_LISTEN=${3:-127.0.0.1:8787}
QX_RPC_ENDPOINT=${QX_RPC_URL:-https://api.devnet.solana.com}
QX_VERIFIER_PROGRAM=${QX_PROGRAM_ID:-QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB}

if [[ ! -f "$QX_GATEWAY_TOKEN" ]]; then
  echo "miner token file does not exist: $QX_GATEWAY_TOKEN" >&2
  exit 1
fi

exec cargo run --release -p qx-gateway -- serve \
  --listen "$QX_GATEWAY_LISTEN" \
  --data-dir "$QX_GATEWAY_DATA" \
  --program "$QX_VERIFIER_PROGRAM" \
  --rpc-url "$QX_RPC_ENDPOINT" \
  --miner-token-file "$QX_GATEWAY_TOKEN"
