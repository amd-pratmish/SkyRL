#!/usr/bin/env bash
# Install ROCm vLLM into rocm/primus:v26.4 without upgrading PyTorch.
# Requires .vllm_rocm_cache (see extract_vllm_rocm_cache.sh) mounted at /opt/vllm_rocm or in repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CACHE="${VLLM_ROCM_CACHE:-${SCRIPT_DIR}/.vllm_rocm_cache}"
MOUNT="${VLLM_ROCM_MOUNT:-/opt/vllm_rocm}"

if [ ! -d "${MOUNT}/vllm" ] && [ -d "${CACHE}/vllm" ]; then
  MOUNT="${CACHE}"
fi

if [ ! -d "${MOUNT}/vllm" ]; then
  echo "ERROR: ROCm vLLM cache missing. Run extract_vllm_rocm_cache.sh on a GPU node first."
  exit 1
fi

TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
TV_VER="$(python3 -c 'import torchvision; print(torchvision.__version__)')"
echo "Pinning torch==${TORCH_VER} torchvision==${TV_VER}"

SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
rm -rf "${SITE}/vllm" "${SITE}/vllm-"*.dist-info
cp -a "${MOUNT}/vllm" "${SITE}/" || cp -r "${MOUNT}/vllm" "${SITE}/"
cp -a "${MOUNT}/vllm-"*.dist-info "${SITE}/" 2>/dev/null || true

# Pure-Python + ROCm-safe runtime deps for vLLM 0.9.2rc (pinned from rocm/vllm-dev freeze; no torch).
python3 -m pip install --no-cache-dir -q \
  "transformers>=5.5.3" \
  "numba==0.65.0" "llvmlite==0.47.0" \
  "msgspec==0.21.1" "cbor2==6.1.4" "xgrammar==0.2.3" "outlines_core==0.2.14" \
  "llguidance==1.7.6" "lm-format-enforcer==0.11.3" \
  "compressed-tensors==0.17.0" "fastsafetensors==0.3.3" \
  "partial-json-parser==0.2.1.1.post7" "prometheus-fastapi-instrumentator==8.1.0" \
  "openai==3.3.1" "openai-harmony==0.0.8" "opencv-python-headless==5.0.0.93" \
  "pybase64==1.5.0" "pyzmq==27.2.0" "watchfiles==1.2.0" "blake3==1.0.9" \
  "depyf==0.20.0" "ijson==3.5.1" "lark==1.2.2" "model-hosting-container-standards==0.1.16" \
  "opentelemetry-exporter-otlp==1.44.0" "opentelemetry-semantic-conventions-ai==0.5.1" \
  "setproctitle==1.3.7" "interegular==0.3.3" "astor==0.8.1" "sniffio==1.3.1" \
  "python-json-logger==4.2.0" "supervisor==4.3.0" "uvloop==0.22.1" \
  "httptools==0.8.0" "httpx2==2.12.0" "httpcore2==2.12.0" \
  "anthropic>=0.71.0" "grpcio-reflection==1.78.0" "mistral_common>=1.11.6" \
  "tilelang==0.1.10" "timm>=1.0.17"

# Guard: pip must not replace ROCm torch.
AFTER="$(python3 -c 'import torch; print(torch.__version__)')"
if [ "$AFTER" != "$TORCH_VER" ]; then
  echo "ERROR: torch changed ${TORCH_VER} -> ${AFTER}; refusing to continue"
  exit 1
fi

python3 -c "import vllm; print('vllm', vllm.__version__)"
echo "ROCm vLLM installed (torch unchanged: ${TORCH_VER})"
