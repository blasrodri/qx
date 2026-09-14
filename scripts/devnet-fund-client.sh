#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <payer-keypair.json> <client-pubkey> [devnet-sol]" >&2
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

QX_PAYER_KEYPAIR=$1
QX_CLIENT_PUBKEY=$2
QX_DEVNET_SOL=${3:-0.1}
QX_RPC_ENDPOINT=${QX_RPC_URL:-https://api.devnet.solana.com}
QX_SOLANA_BINARY=${QX_SOLANA:-}

if [[ "$QX_RPC_ENDPOINT" != "https://api.devnet.solana.com" ]]; then
  echo "refusing to fund through a non-devnet RPC: $QX_RPC_ENDPOINT" >&2
  exit 1
fi
if [[ ! -f "$QX_PAYER_KEYPAIR" ]]; then
  echo "payer keypair does not exist: $QX_PAYER_KEYPAIR" >&2
  exit 1
fi
if [[ -z "$QX_SOLANA_BINARY" ]]; then
  QX_SOLANA_BINARY=$(command -v solana || true)
fi
if [[ -z "$QX_SOLANA_BINARY" && -x /Users/blas/.local/share/solana/install/active_release/bin/solana ]]; then
  QX_SOLANA_BINARY=/Users/blas/.local/share/solana/install/active_release/bin/solana
fi
if [[ -z "$QX_SOLANA_BINARY" ]]; then
  echo "solana CLI is unavailable; set QX_SOLANA to its absolute path" >&2
  exit 1
fi
if ! awk -v amount="$QX_DEVNET_SOL" 'BEGIN { exit !(amount ~ /^[0-9]+([.][0-9]+)?$/ && amount > 0 && amount <= 2) }'; then
  echo "devnet-sol must be greater than zero and no more than 2" >&2
  exit 1
fi

"$QX_SOLANA_BINARY" transfer "$QX_CLIENT_PUBKEY" "$QX_DEVNET_SOL" \
  --url "$QX_RPC_ENDPOINT" \
  --keypair "$QX_PAYER_KEYPAIR" \
  --fee-payer "$QX_PAYER_KEYPAIR" \
  --allow-unfunded-recipient

echo "client_balance=$("$QX_SOLANA_BINARY" balance "$QX_CLIENT_PUBKEY" --url "$QX_RPC_ENDPOINT")"
