#!/usr/bin/env bash
# Full AMD stack: Primus + Megatron-Bridge + vLLM ROCm + SkyRL for end-to-end GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/approach_a_install.sh"

echo "--- install vLLM (ROCm) ---"
WHEEL_CACHE="/workspace/SkyRL/integrations/primus_amd/.vllm_rocm_cache/wheels"
if compgen -G "${WHEEL_CACHE}/vllm-*.whl" >/dev/null; then
  WHEEL="$(ls -1 "${WHEEL_CACHE}"/vllm-*.whl | tail -1)"
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
elif ! python3 -c "import vllm" 2>/dev/null; then
  bash "${SCRIPT_DIR}/build_vllm_rocm.sh"
else
  python3 -c "import vllm; print('vllm', vllm.__version__, '(preinstalled)')"
fi

echo "--- install skyrl-gym deps for gsm8k ---"
python3 -m pip install --no-cache-dir -q datasets 2>/dev/null || true

echo "--- ensure Ray 2.57 (SkyRL pin; vLLM deps must not downgrade) ---"
python3 -m pip install --no-cache-dir --force-reinstall "ray[default]==2.57.0"

echo "--- verify Ray scheduling API ---"
python3 -c "
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
print('ray', ray.__version__, 'PlacementGroupSchedulingStrategy OK')
"

echo "=== Full AMD stack ready ==="
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
