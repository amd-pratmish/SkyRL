#!/usr/bin/env bash
# Slurm job: full Megatron GRPO cycle on AMD (training + generator + vLLM inference).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${PRIMUS_IMAGE:-rocm/primus:v26.4}"
GPUS="${SLURM_GPUS:-2}"
GPU_DEVS="$(python3 -c "print(','.join(str(i) for i in range(${GPUS})))")"
CPUS="${SLURM_CPUS:-16}"
MEM="${SLURM_MEM:-128G}"
TIME="${SLURM_TIME:-04:00:00}"
PARTITION="${SLURM_PARTITION:-rad}"
LOG="${ROOT}/integrations/primus_amd/reports/grpo_amd_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "${ROOT}/integrations/primus_amd/reports"

# BUILD_ONLY=1: install stack + build vLLM 0.20.2 wheel (no GRPO).
# INSTALL_ONLY=1: full install only (uses cached wheel if valid).
RUN_GRPO=1
if [[ "${BUILD_ONLY:-0}" == "1" || "${INSTALL_ONLY:-0}" == "1" ]]; then
  RUN_GRPO=0
fi

INNER=""
if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
  INNER='bash integrations/primus_amd/approach_a_install.sh && bash integrations/primus_amd/build_vllm_rocm.sh'
elif [[ "${INSTALL_ONLY:-0}" == "1" ]]; then
  INNER='bash integrations/primus_amd/approach_a_full_install.sh'
else
  INNER='bash integrations/primus_amd/approach_a_full_install.sh && bash integrations/primus_amd/run_grpo_amd_megatron.sh'
fi

echo "GRPO AMD log -> $LOG (partition=$PARTITION gpus=$GPUS run_grpo=$RUN_GRPO)"

srun -p "$PARTITION" \
  --gres="gpu:${GPUS}" \
  -c "$CPUS" \
  --mem="$MEM" \
  -t "$TIME" \
  bash -lc "
set -euo pipefail
export ROCR_VISIBLE_DEVICES=${GPU_DEVS}
export HIP_VISIBLE_DEVICES=${GPU_DEVS}
unset CUDA_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
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
  -e RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
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
    unset CUDA_VISIBLE_DEVICES
    ${INNER}
  ' 2>&1 | tee '${LOG}'
"

echo "Done. Log: ${LOG}"
