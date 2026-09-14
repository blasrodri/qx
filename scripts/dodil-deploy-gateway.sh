#!/usr/bin/env bash
set -euo pipefail

QX_DODIL_APP=${QX_DODIL_APP:-qx-gateway-devnet}
QX_DODIL_BUCKET=${QX_DODIL_BUCKET:-qx-gateway-devnet}
QX_VERIFIER_PROGRAM=${QX_PROGRAM_ID:-QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB}
QX_RPC_ENDPOINT=${QX_RPC_URL:-https://api.devnet.solana.com}
QX_SOURCE_COMMIT=${QX_SOURCE_COMMIT:-$(git rev-parse HEAD)}
QX_ARCHIVE_DIRECTORY=$(mktemp -d)
QX_SOURCE_ARCHIVE="$QX_ARCHIVE_DIRECTORY/qx-gateway.tar.gz"

cleanup() {
  rm -f "$QX_SOURCE_ARCHIVE"
  rmdir "$QX_ARCHIVE_DIRECTORY"
}
trap cleanup EXIT

git archive --format=tar.gz --output "$QX_SOURCE_ARCHIVE" "$QX_SOURCE_COMMIT"

dodil ignite app deploy "$QX_DODIL_APP" \
  --code "$QX_SOURCE_ARCHIVE" \
  --dockerfile-path Dockerfile.gateway \
  --compile-timeout-secs 1800 \
  --tier small \
  --memory 512 \
  --port 8080 \
  --health-path /health \
  --max-replicas 2 \
  --auto-min-instances 0 \
  --allow-unauthenticated \
  --env "QX_DODIL_BUCKET=$QX_DODIL_BUCKET" \
  --env "QX_PROGRAM_ID=$QX_VERIFIER_PROGRAM" \
  --env "QX_RPC_URL=$QX_RPC_ENDPOINT" \
  --env "QX_MINER_TOKEN=secret://qx-gateway-miner-token" \
  --env "DODIL_TOKEN=secret://qx-gateway-datak3-token" \
  --description "QX permissionless inference verification gateway (Solana devnet)" \
  --follow
