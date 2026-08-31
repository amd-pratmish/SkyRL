#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
MCORE_REV="${MCORE_REV:-71e418ea7d7b3a6c9a53238c543c3e0b43e11026}"
BRIDGE_REV="${BRIDGE_REV:-91a15142a4b4442a8d46ab539d1b923bd08570d0}"

export PYTHONPATH="$(python3 - <<'PY'
import os
print(":".join(p for p in os.environ.get("PYTHONPATH","").split(":") if p and "Megatron-LM" not in p))
PY
)"

install_stack() {
  python3 -m pip install -q --no-cache-dir --no-deps --force-reinstall \
    "megatron-core @ git+https://github.com/NVIDIA/Megatron-LM@${MCORE_REV}"
  python3 -m pip install -q --no-cache-dir nvidia-modelopt \
    "transformers>=5.8.1,<5.9.0" peft accelerate datasets omegaconf hydra-core einops rich wandb
  python3 -m pip install -q --no-cache-dir --no-deps \
    "megatron-bridge @ git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@${BRIDGE_REV}"
  cp docker/pyproject.primus.toml pyproject.toml
  python3 -m pip install -q --no-cache-dir -e ".[primus-megatron]" || true
}

echo "=== Approach A validate ==="
install_stack

python3 -c "import megatron.core as mc; import megatron.core._rank_utils as ru; print('mcore', mc.__version__, mc.__file__); assert hasattr(ru,'safe_get_world_size')"

python3 - <<'PY'
from megatron.bridge import AutoBridge
print("PASS: AutoBridge import")
PY

python3 - <<PY
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from megatron.bridge import AutoBridge
bridge = AutoBridge.from_hf_pretrained("${MODEL}", trust_remote_code=True)
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
print("PASS: from_hf_pretrained ${MODEL}", type(provider).__name__)
PY

python3 - <<'PY'
import skyrl.backends.skyrl_train.workers.megatron.model_bridges  # noqa: F401
from skyrl.backends.skyrl_train.workers.megatron import megatron_worker
print("PASS: megatron_worker", megatron_worker)
PY

python3 - <<PY
import os, torch, torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core import parallel_state as mpu
model = "${MODEL}"
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29504", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
torch.cuda.set_device(0)
dist.init_process_group(backend="nccl")
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
provider.attention_backend = "flash"
provider.finalize()
models = provider.provide_distributed_model(wrap_with_ddp=False)
print("PASS: provide_distributed_model", len(models))
dist.destroy_process_group()
PY

echo "=== ALL Approach A validation steps PASSED ==="
