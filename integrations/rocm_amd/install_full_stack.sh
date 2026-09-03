#!/usr/bin/env bash
# Full ROCm stack: Megatron-Bridge + vLLM (ROCm build) + SkyRL for training + inference on AMD GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WHEEL_CACHE="${SKYRL_ROOT}/integrations/rocm_amd/.vllm_rocm_cache/wheels"

bash "${SCRIPT_DIR}/install_megatron_bridge.sh"

if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
  echo "--- verify supported Instinct GPU (MI300X/MI325X/MI350X/MI355X) ---"
  python3 "${SCRIPT_DIR}/gpu_support.py" || {
    echo "WARN: GPU not supported or not detected; see integrations/rocm_amd/README.md#supported-gpus"
    exit 1
  }
else
  echo "SKIP: no visible GPU during install (OK for image build without devices)"
fi

echo "--- install vLLM (ROCm, SkyRL 0.20.x API) ---"
WHEEL=""
if compgen -G "${WHEEL_CACHE}/vllm-0.20*.whl" >/dev/null; then
  WHEEL="$(ls -1t "${WHEEL_CACHE}"/vllm-0.20*.whl | head -1)"
elif compgen -G "${WHEEL_CACHE}/vllm-*.whl" >/dev/null; then
  WHEEL="$(ls -1t "${WHEEL_CACHE}"/vllm-*.whl | head -1)"
fi

if [ -n "${WHEEL}" ]; then
  echo "Using cached wheel ${WHEEL}"
  python3 -m pip install --no-cache-dir --force-reinstall --no-deps "${WHEEL}"
  python3 -m pip install --no-cache-dir -q \
    msgspec cbor2 xgrammar outlines_core llguidance compressed-tensors fastsafetensors \
    partial-json-parser openai openai-harmony pybase64 pyzmq watchfiles blake3 depyf mistral_common \
    opencv-python-headless prometheus-fastapi-instrumentator setproctitle \
    lm-format-enforcer lark ijson uvloop httptools httpx2 httpcore2 supervisor \
    opentelemetry-exporter-otlp opentelemetry-semantic-conventions-ai model-hosting-container-standards \
    anthropic grpcio-reflection timm tensorizer pytest-asyncio \
    "transformers>=5.5.3" "numba==0.65.0" 2>/dev/null || true
  python3 -m pip install --no-cache-dir -q amdsmi 2>/dev/null || true
  python3 "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"
elif ! python3 "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py" 2>/dev/null; then
  bash "${SCRIPT_DIR}/build_vllm_rocm.sh"
else
  python3 "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"
fi

echo "--- install skyrl-gym deps for gsm8k ---"
python3 -m pip install --no-cache-dir -q datasets 2>/dev/null || true

echo "--- ensure Ray 2.57 (SkyRL pin) ---"
python3 -m pip install --no-cache-dir --force-reinstall "ray[default]==2.57.0"

echo "--- re-pin transformers for SkyRL after vLLM deps ---"
python3 -m pip install --no-cache-dir --force-reinstall "transformers>=5.6.1,<=5.8.0"

echo "--- verify Ray scheduling API ---"
python3 -c "
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
print('ray', ray.__version__, 'PlacementGroupSchedulingStrategy OK')
"

echo "=== Full ROCm AMD stack ready ==="
python3 -c "
import torch
from megatron.bridge import AutoBridge
import megatron.core as mc
print('torch', torch.__version__, 'hip', getattr(torch.version,'hip',None))
print('megatron-core', mc.__version__)
print('AutoBridge', AutoBridge)
import importlib.util
print('vllm', bool(importlib.util.find_spec('vllm')))
"
python3 "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"
