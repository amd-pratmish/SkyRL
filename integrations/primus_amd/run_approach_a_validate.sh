#!/usr/bin/env bash
# Continue Approach A validation after bridge+mcore install (steps 4-7).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

srun -p "${SLURM_PARTITION:-rad}" --gres=gpu:1 -c 8 --mem=64G -t 00:45:00 bash -lc "
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 \
  -e NVTE_USE_ROCM=1 \
  -v '${ROOT}:/workspace/SkyRL' -w /workspace/SkyRL rocm/primus:v26.4 \
  bash integrations/primus_amd/approach_a_validate.sh '${MODEL}'
"
