#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <program-id> <payer-keypair.json> [transactions]" >&2
  echo "set QX_RPC_URL to override https://api.devnet.solana.com" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

QX_PROGRAM_ID=$1
QX_PAYER_KEYPAIR=$2
QX_TRANSACTION_COUNT=${3:-10}
QX_RPC_ENDPOINT=${QX_RPC_URL:-https://api.devnet.solana.com}

if [[ ! -f "$QX_PAYER_KEYPAIR" ]]; then
  echo "payer keypair does not exist: $QX_PAYER_KEYPAIR" >&2
  exit 1
fi

cargo run --release -p qx-testnet-load -- \
  --rpc-url "$QX_RPC_ENDPOINT" \
  --program "$QX_PROGRAM_ID" \
  --keypair "$QX_PAYER_KEYPAIR" \
  --transactions "$QX_TRANSACTION_COUNT" \
  --concurrency 2
