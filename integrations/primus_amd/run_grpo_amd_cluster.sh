#!/usr/bin/env bash
# Slurm job: full Megatron GRPO cycle on AMD (training + generator + vLLM inference).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${PRIMUS_IMAGE:-rocm/primus:v26.4}"
VLLM_IMAGE="${VLLM_IMAGE:-rocm/vllm-dev:nightly_cdna4}"
VLLM_CACHE="${ROOT}/integrations/primus_amd/.vllm_rocm_cache"
GPUS="${SLURM_GPUS:-2}"
GPU_DEVS="$(python3 -c "print(','.join(str(i) for i in range(${GPUS})))")"
CPUS="${SLURM_CPUS:-16}"
MEM="${SLURM_MEM:-128G}"
TIME="${SLURM_TIME:-02:00:00}"
PARTITION="${SLURM_PARTITION:-rad}"
LOG="${ROOT}/integrations/primus_amd/reports/grpo_amd_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "${ROOT}/integrations/primus_amd/reports"

echo "GRPO AMD log -> $LOG"

srun -p "$PARTITION" \
  --gres="gpu:${GPUS}" \
  -c "$CPUS" \
  --mem="$MEM" \
  -t "$TIME" \
  bash -lc "
set -euo pipefail
export ROCR_VISIBLE_DEVICES=${GPU_DEVS}
export HIP_VISIBLE_DEVICES=${GPU_DEVS}
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export NUM_GPUS=${GPUS}
export DATA_DIR=/tmp/gsm8k_amd_\${USER}
export HOME=/tmp/skyrl_home_\${USER}
mkdir -p \"\$HOME\" \"\$DATA_DIR\"

docker run --rm --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=${GPU_DEVS} \
  -e ROCR_VISIBLE_DEVICES=${GPU_DEVS} \
  -e RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 \
  -e RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
  -e _SKYRL_USE_NEW_INFERENCE=0 \
  -e NUM_GPUS -e DATA_DIR -e HOME \
  -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -v '${ROOT}:/workspace/SkyRL' \
  -w /workspace/SkyRL \
  '${IMAGE}' \
  bash -lc '
    set -euo pipefail
    bash integrations/primus_amd/approach_a_full_install.sh
    bash integrations/primus_amd/run_grpo_amd_megatron.sh
  ' 2>&1 | tee '${LOG}'
"

echo "Done. Log: ${LOG}"
