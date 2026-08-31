#!/usr/bin/env bash
# Runs inside rocm/primus:v26.4 container.
set -euo pipefail

BRIDGE_REV="${1:-91a15142a4b4442a8d46ab539d1b923bd08570d0}"
MODEL="${2:-Qwen/Qwen2.5-0.5B-Instruct}"
BRIDGE_URL="https://github.com/NVIDIA-NeMo/Megatron-Bridge/archive/${BRIDGE_REV}.tar.gz"

bootstrap_paths() {
  # Do not prepend Primus Megatron-LM source — it shadows pip megatron-core 0.19.
  :
}
bootstrap_paths
unset PYTHONPATH || true

echo "=== Approach A: megatron-bridge on Primus/ROCm ==="
python3 --version
python3 -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpus', torch.cuda.device_count())"

MCORE_SRC=""
for candidate in /workspace/Primus/third_party/Megatron-LM /workspace/Megatron-LM; do
  if [ -f "$candidate/pyproject.toml" ] || [ -f "$candidate/setup.py" ]; then
    MCORE_SRC="$candidate"
    break
  fi
done
echo "megatron-core source: ${MCORE_SRC:-NOT FOUND}"

MCORE_REV="${MCORE_REV:-71e418ea7d7b3a6c9a53238c543c3e0b43e11026}"

echo
echo "--- Step 1: install megatron-core from SkyRL git pin (${MCORE_REV}) ---"
python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
  "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM@${MCORE_REV}" \
  2>&1 | tail -8
# Drop bundled Primus Megatron-LM from default container PYTHONPATH if set.
export PYTHONPATH="$(python3 - <<'PY'
import os
parts = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p and "Megatron-LM" not in p]
print(":".join(parts))
PY
)"
python3 -c "import megatron.core; import megatron.core._rank_utils as ru; print('megatron.core', getattr(megatron.core,'__version__','?'), megatron.core.__file__); print('safe_get_world_size', hasattr(ru,'safe_get_world_size'))"

echo
echo "--- Step 2: install bridge Python deps (skip CUDA-only packages) ---"
python3 -m pip install --no-cache-dir \
  "transformers>=5.8.1,<5.9.0" \
  "mistral-common>=1.10.0" \
  "peft>=0.18.1" \
  "datasets>=2.20.0" \
  accelerate diffusers einops imageio imageio-ffmpeg \
  "omegaconf>=2.3.0" tensorboard typing-extensions rich wandb six regex pyyaml tqdm \
  "hydra-core>=1.3,<=1.3.2" \
  qwen-vl-utils flash-linear-attention timm \
  "open-clip-torch>=3.2.0" \
  nvidia-modelopt \
  2>&1 | tail -8

echo
echo "--- Step 3: install megatron-bridge --no-deps from SkyRL pin ---"
python3 -m pip install --no-cache-dir --no-deps \
  "megatron-bridge @ git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@${BRIDGE_REV}" \
  2>&1 | tail -8

echo
echo "--- Step 4: import megatron.bridge ---"
python3 - <<'PY'
import traceback
try:
    from megatron.bridge import AutoBridge
    print("PASS: from megatron.bridge import AutoBridge", AutoBridge)
except Exception as e:
    print("FAIL: import AutoBridge:", type(e).__name__, e)
    traceback.print_exc()
PY

echo
echo "--- Step 5: AutoBridge.from_hf_pretrained (small model) ---"
python3 - <<PY
import traceback
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
model = "${MODEL}"
try:
    from megatron.bridge import AutoBridge
    bridge = AutoBridge.from_hf_pretrained(model, trust_remote_code=True)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    print("PASS: AutoBridge.from_hf_pretrained", model)
    print("provider type:", type(provider).__name__)
except Exception as e:
    print("FAIL: from_hf_pretrained:", type(e).__name__, e)
    traceback.print_exc()
PY

echo
echo "--- Step 6: SkyRL megatron_worker import ---"
cp docker/pyproject.primus.toml pyproject.toml
python3 -m pip install --no-cache-dir -e ".[primus-megatron]" 2>&1 | tail -5 || true
python3 - <<'PY'
import traceback
try:
    import skyrl.backends.skyrl_train.workers.megatron.model_bridges  # noqa: F401
    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker
    print("PASS: megatron_worker imported", megatron_worker)
except Exception as e:
    print("FAIL: megatron_worker:", type(e).__name__, e)
    traceback.print_exc()
PY

echo
echo "--- Step 7: optional materialize model on GPU (1 GPU) ---"
python3 - <<PY
import os, traceback
import torch
import torch.distributed as dist
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29503")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
model = "${MODEL}"
try:
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state as mpu
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    bridge = AutoBridge.from_hf_pretrained(model, trust_remote_code=True)
    provider = bridge.to_megatron_provider()
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.finalize()
    models = provider.provide_distributed_model(wrap_with_ddp=False)
    print("PASS: provide_distributed_model", type(models), "n=", len(models) if hasattr(models,'__len__') else '?')
    if dist.is_initialized():
        dist.destroy_process_group()
except Exception as e:
    print("FAIL: materialize:", type(e).__name__, e)
    traceback.print_exc()
PY

echo
echo "=== Approach A install script finished ==="
