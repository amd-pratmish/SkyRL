#!/usr/bin/env bash
# Install Megatron-Bridge + SkyRL on rocm/vllm-dev (has working vLLM for MI355X/cdnd4).
set -euo pipefail

MCORE_REV="${MCORE_REV:-71e418ea7d7b3a6c9a53238c543c3e0b43e11026}"
BRIDGE_REV="${BRIDGE_REV:-91a15142a4b4442a8d46ab539d1b923bd08570d0}"

export NVTE_USE_ROCM="${NVTE_USE_ROCM:-1}"
export NVTE_USE_HIPBLASLT="${NVTE_USE_HIPBLASLT:-1}"

echo "Installing megatron-core @ ${MCORE_REV}"
python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
  "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM@${MCORE_REV}"

echo "Installing bridge deps + nvidia-modelopt"
python3 -m pip install --no-cache-dir \
  nvidia-modelopt \
  "transformers>=5.8.1,<5.9.0" \
  "mistral-common>=1.10.0" \
  "peft>=0.18.1" \
  "datasets>=4.0.0" \
  accelerate diffusers einops imageio imageio-ffmpeg \
  "omegaconf>=2.3.0" tensorboard typing-extensions rich wandb six regex pyyaml tqdm \
  "hydra-core>=1.3,<=1.3.2" \
  qwen-vl-utils flash-linear-attention timm \
  "open-clip-torch>=3.2.0"

echo "Installing megatron-bridge @ ${BRIDGE_REV} (--no-deps)"
python3 -m pip install --no-cache-dir --no-deps \
  "megatron-bridge @ git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@${BRIDGE_REV}"

if [ -d /workspace/SkyRL ]; then
  cd /workspace/SkyRL
  cp docker/pyproject.primus.toml pyproject.toml
  python3 -m pip install --no-cache-dir "ray[default]==2.57.0"
  python3 -m pip install --no-cache-dir -e ".[primus-megatron]"
  python3 -m pip install --no-cache-dir "transformers>=5.8.1,<5.9.0" -q
fi

python3 -c "from megatron.bridge import AutoBridge; import megatron.core as mc; print('OK bridge+mcore', mc.__version__)"
python3 -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)"
