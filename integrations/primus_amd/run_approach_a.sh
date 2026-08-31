#!/usr/bin/env bash
# Exhaust Approach A: install megatron-bridge on rocm/primus:v26.4 for SkyRL megatron_worker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${REPORT_DIR:-$ROOT/integrations/primus_amd/reports}"
mkdir -p "$REPORT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/approach_a_${STAMP}.log"
IMAGE="${PRIMUS_IMAGE:-rocm/primus:v26.4}"
BRIDGE_REV="${BRIDGE_REV:-91a15142a4b4442a8d46ab539d1b923bd08570d0}"
MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "Approach A bridge install -> $LOG"

srun -p "${SLURM_PARTITION:-rad}" --gres=gpu:1 -c 8 --mem=64G -t 01:00:00 bash -lc "
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 \
  -e NVTE_USE_ROCM=1 \
  -e NVTE_USE_HIPBLASLT=1 \
  -e NVTE_FUSED_ATTN=0 \
  -v '${ROOT}:/workspace/SkyRL' \
  -w /workspace/SkyRL \
  '${IMAGE}' \
  bash integrations/primus_amd/approach_a_bridge_install.sh '${BRIDGE_REV}' '${MODEL}' \
  2>&1 | tee '${LOG}'
"
