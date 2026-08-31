#!/usr/bin/env bash
# Attempt megatron-bridge install on primus:latest (exploratory).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${REPORT_DIR:-$ROOT/integrations/primus_amd/reports}"
mkdir -p "$REPORT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/bridge_install_${STAMP}.log"

srun -p "${SLURM_PARTITION:-rad}" --gres=gpu:1 -c 8 --mem=64G -t 00:45:00 bash -lc "
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0 \
  -v '${ROOT}:/workspace/SkyRL:ro' \
  primus:latest bash -lc '
    export PYTHONPATH=/workspace/Megatron-LM:/workspace/Primus
    python3 --version
    python3 -m pip install --no-cache-dir megatron-bridge 2>&1 || true
    python3 -c \"import megatron.bridge; print(megatron.bridge.__file__)\" 2>&1 || true
    python3 -c \"from megatron.bridge import AutoBridge; print(AutoBridge)\" 2>&1 || true
  ' | tee '$LOG'
"
