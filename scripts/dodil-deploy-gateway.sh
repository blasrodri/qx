#!/usr/bin/env bash
set -euo pipefail

QX_DODIL_APP=${QX_DODIL_APP:-qx-gateway-devnet}
QX_DODIL_BUCKET=${QX_DODIL_BUCKET:-qx-gateway-devnet}
QX_VERIFIER_PROGRAM=${QX_PROGRAM_ID:-QJPMJrcmJEeLcVZdqaofF9ioCpvXYJXLVZhFhea2EiB}
QX_RPC_ENDPOINT=${QX_RPC_URL:-https://api.devnet.solana.com}
QX_DODIL_ORG=${QX_DODIL_ORG:-sitoula}
QX_IMAGE_TAG=${QX_IMAGE_TAG:-$(git rev-parse HEAD)}
QX_IMAGE_REF="registry.dodil.io/$QX_DODIL_ORG/qx-gateway:$QX_IMAGE_TAG"

QX_REGISTRY_KEY_JSON=$(dodil auth apikey issue \
  --name qx-gateway-deploy \
  --service registry \
  --role registry.developer \
  --ttl 1h \
  --output json)
QX_REGISTRY_TOKEN=$(printf '%s' "$QX_REGISTRY_KEY_JSON" | jq -er '.secret')
QX_REGISTRY_LOOKUP=$(printf '%s' "$QX_REGISTRY_KEY_JSON" | jq -er '.lookup')

cleanup() {
  docker logout registry.dodil.io >/dev/null 2>&1 || true
  dodil auth apikey revoke "$QX_REGISTRY_LOOKUP" >/dev/null 2>&1 || true
}
trap cleanup EXIT

printf '%s' "$QX_REGISTRY_TOKEN" | \
  docker login registry.dodil.io --username ci --password-stdin >/dev/null

docker buildx build \
  --platform linux/amd64 \
  --file Dockerfile.gateway \
  --tag "$QX_IMAGE_REF" \
  --push \
  .

dodil ignite app deploy "$QX_DODIL_APP" \
  --image-ref "$QX_IMAGE_REF" \
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
