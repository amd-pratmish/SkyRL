#!/usr/bin/env bash
# SkyRL-side smoke on rocm/vllm-dev (Python 3.12 + vLLM, no Megatron).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${VLLM_IMAGE:-rocm/vllm-dev:nightly_cdna4}"
REPORT_DIR="${REPORT_DIR:-$ROOT/integrations/primus_amd/reports}"
mkdir -p "$REPORT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

srun -p "${SLURM_PARTITION:-rad}" --gres=gpu:1 -c 8 --mem=64G -t 00:45:00 bash -lc "
set -euo pipefail
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 \
  -v '${ROOT}:/workspace/SkyRL' \
  -w /workspace/SkyRL \
  '${IMAGE}' bash -lc '
    python3 --version
    python3 -c \"import torch,vllm; print(torch.__version__, torch.cuda.device_count(), vllm.__version__)\"
    cp docker/pyproject.primus.toml pyproject.toml
    python3 -m pip install --no-cache-dir ray==2.57.0
    python3 -m pip install --no-cache-dir -e \".[primus-megatron]\"
    python3 integrations/primus_amd/smoke_test.py --phase env --phase skyrl --phase vllm \
      --report /workspace/SkyRL/integrations/primus_amd/reports/vllmdev_${STAMP}.json
  '
"
