#!/usr/bin/env bash
# Run on Slurm compute node (host has docker). Extracts ROCm vLLM from rocm/vllm-dev into repo cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE="${ROOT}/integrations/primus_amd/.vllm_rocm_cache"
IMAGE="${VLLM_IMAGE:-rocm/vllm-dev:nightly_cdna4}"

mkdir -p "$CACHE"
if [ -d "$CACHE/vllm" ] && [ -n "$(ls -A "$CACHE/vllm" 2>/dev/null)" ]; then
  echo "vLLM ROCm cache already present at $CACHE/vllm"
  exit 0
fi

echo "Extracting ROCm vLLM from ${IMAGE} -> ${CACHE}"
cid="$(docker create "$IMAGE")"
docker cp "${cid}:/usr/local/lib/python3.12/dist-packages/vllm" "$CACHE/"
docker cp "${cid}:/usr/local/lib/python3.12/dist-packages/" "$CACHE/_all/"
mv "$CACHE/_all/vllm-"*.dist-info "$CACHE/" 2>/dev/null || true
rm -rf "$CACHE/_all"
docker rm "$cid" >/dev/null
echo "Done. Cached $(du -sh "$CACHE/vllm" | awk '{print $1}') at $CACHE/vllm"
