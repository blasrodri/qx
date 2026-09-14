#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <payer-keypair.json> [program-keypair.json]" >&2
  echo "set QX_RPC_URL to override https://api.devnet.solana.com" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

QX_PAYER_KEYPAIR=$1
QX_PROGRAM_KEYPAIR=${2:-target/deploy/qx_terminal_verifier-keypair.json}
QX_RPC_ENDPOINT=${QX_RPC_URL:-https://api.devnet.solana.com}
QX_PROGRAM_SO=target/deploy/qx_terminal_verifier.so

for QX_COMMAND in cargo solana solana-keygen; do
  if ! command -v "$QX_COMMAND" >/dev/null 2>&1; then
    echo "missing required command: $QX_COMMAND" >&2
    exit 1
  fi
done

if [[ ! -f "$QX_PAYER_KEYPAIR" ]]; then
  echo "payer keypair does not exist: $QX_PAYER_KEYPAIR" >&2
  exit 1
fi

cargo build-sbf --manifest-path programs/qx-terminal-verifier/Cargo.toml

if [[ ! -f "$QX_PROGRAM_SO" || ! -f "$QX_PROGRAM_KEYPAIR" ]]; then
  echo "cargo build-sbf did not produce the expected program artifacts" >&2
  exit 1
fi

QX_PAYER_ADDRESS=$(solana-keygen pubkey "$QX_PAYER_KEYPAIR")
QX_PROGRAM_ID=$(solana-keygen pubkey "$QX_PROGRAM_KEYPAIR")

echo "rpc_url=$QX_RPC_ENDPOINT"
echo "payer=$QX_PAYER_ADDRESS"
echo "program_id=$QX_PROGRAM_ID"
solana balance "$QX_PAYER_ADDRESS" --url "$QX_RPC_ENDPOINT"

solana program deploy "$QX_PROGRAM_SO" \
  --program-id "$QX_PROGRAM_KEYPAIR" \
  --keypair "$QX_PAYER_KEYPAIR" \
  --url "$QX_RPC_ENDPOINT"

solana program show "$QX_PROGRAM_ID" --url "$QX_RPC_ENDPOINT"
echo "PROGRAM_ID=$QX_PROGRAM_ID"
