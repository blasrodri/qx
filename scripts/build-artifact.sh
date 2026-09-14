#!/usr/bin/env bash
set -euo pipefail

QX_MODEL_REVISION=ffec3c35bdfd97a06f0b4cd5fcc92cd9b1584445
QX_MODEL_SHA256=f7ae1cee56a9ea6c5360437b1c0407f8d84816b2cc75470f4e7e5236fa2a07dc
QX_CONFIG_SHA256=ca4bb3a5c1bdef988ab413e0d731640446da65316e4ed16de3666cd96ecc3a0b

QX_SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
QX_REPO_ROOT=$(cd "$QX_SCRIPT_DIRECTORY/.." && pwd)
QX_MODEL_DIRECTORY="$QX_REPO_ROOT/external/models/granite-3.0-1b-a400m-instruct"
QX_OUTPUT="$QX_REPO_ROOT/external/models/granite-m5-pro-q4-main.qx"
QX_STAGING_OUTPUT="${QX_OUTPUT}.building"

for QX_COMMAND in cargo curl git shasum awk; do
  if ! command -v "$QX_COMMAND" >/dev/null 2>&1; then
    echo "missing required command: $QX_COMMAND" >&2
    exit 1
  fi
done

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

download() {
  local url=$1
  local destination=$2
  local expected_sha=$3
  local partial="${destination}.partial"

  if [[ -f "$destination" ]]; then
    if [[ "$(sha256 "$destination")" != "$expected_sha" ]]; then
      echo "existing file has the wrong checksum: $destination" >&2
      exit 1
    fi
    return
  fi

  curl --location --fail --retry 3 --output "$partial" "$url"
  if [[ "$(sha256 "$partial")" != "$expected_sha" ]]; then
    echo "downloaded file has the wrong checksum: $destination" >&2
    exit 1
  fi
  mv "$partial" "$destination"
}

mkdir -p "$QX_MODEL_DIRECTORY"

download \
  "https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct/resolve/$QX_MODEL_REVISION/config.json" \
  "$QX_MODEL_DIRECTORY/config.json" \
  "$QX_CONFIG_SHA256"
download \
  "https://huggingface.co/ibm-granite/granite-3.0-1b-a400m-instruct/resolve/$QX_MODEL_REVISION/model.safetensors" \
  "$QX_MODEL_DIRECTORY/model.safetensors" \
  "$QX_MODEL_SHA256"

cleanup() {
  rm -f "$QX_STAGING_OUTPUT"
}
trap cleanup EXIT

cd "$QX_REPO_ROOT"
cargo run --release -p qx-compile -- \
  "$QX_MODEL_DIRECTORY" \
  --target m5-pro \
  --memory-budget 2gb \
  --context 512 \
  --disable-q3 \
  --uniform-expert-bits q4 \
  --expert-layout row-inline \
  --moe-mapping qx \
  --weight-storage mmap-no-copy \
  --kv-precision fp32-k-fp16-v \
  --command-buffers-per-token 1 \
  --rollout-depth 2 \
  --plan-id granite-m5-pro-q4-main \
  --output "$QX_STAGING_OUTPUT"

mv "$QX_STAGING_OUTPUT" "$QX_OUTPUT"
trap - EXIT

echo "artifact=$QX_OUTPUT"
echo "sha256=$(sha256 "$QX_OUTPUT")"
echo "source_commit=$(git rev-parse HEAD)"
