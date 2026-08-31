#!/usr/bin/env bash
# Build the SkyRL + Primus exploration image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-skyrl-primus-amd}"
PRIMUS_IMAGE="${PRIMUS_IMAGE:-rocm/primus:v26.4}"
BUILDER="${BUILDER:-docker}"

if ! command -v "$BUILDER" >/dev/null 2>&1; then
  if command -v podman >/dev/null 2>&1; then
    BUILDER=podman
  else
    echo "Neither docker nor podman found. Install one to build the image." >&2
    exit 1
  fi
fi

echo "Building $IMAGE from $PRIMUS_IMAGE using $BUILDER ..."
"$BUILDER" build \
  -f docker/Dockerfile.primus-amd \
  --build-arg "PRIMUS_IMAGE=${PRIMUS_IMAGE}" \
  -t "$IMAGE" \
  .

echo
echo "Built $IMAGE"
echo "Run:"
echo "  $BUILDER run --rm -it --network host --ipc=host \\"
echo "    --device=/dev/kfd --device=/dev/dri --group-add video \\"
echo "    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \\"
echo "    $IMAGE"
