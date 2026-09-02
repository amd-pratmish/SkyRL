#!/usr/bin/env bash
# Slurm: build vLLM 0.20.2 ROCm wheel (long-running; run before GRPO if cache is stale).
set -euo pipefail
export BUILD_ONLY=1
export SLURM_TIME="${SLURM_TIME:-06:00:00}"
export SLURM_GPUS="${SLURM_GPUS:-1}"
export SLURM_PARTITION="${SLURM_PARTITION:-rad-burst}"
exec "$(dirname "$0")/run_grpo_amd_cluster.sh"
