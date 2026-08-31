#!/usr/bin/env bash
# Minimal Megatron forward on Primus (no SkyRL, no bridge) — Phase 1.5 probe.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${REPORT_DIR:-$ROOT/integrations/primus_amd/reports}"
mkdir -p "$REPORT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/megatron_forward_${STAMP}.log"

srun -p "${SLURM_PARTITION:-rad}" --gres=gpu:1 -c 8 --mem=64G -t 00:30:00 bash -lc "
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video -e HIP_VISIBLE_DEVICES=0 \
  -v '${ROOT}:/workspace/SkyRL:ro' \
  primus:latest bash -lc '
    export PYTHONPATH=/workspace/Megatron-LM:/workspace/Primus
    export NVTE_USE_ROCM=1
    python3 /workspace/SkyRL/integrations/primus_amd/megatron_forward_probe.py
  ' | tee '$LOG'
"
