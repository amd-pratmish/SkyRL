#!/usr/bin/env bash
# Executed inside primus container with /workspace/SkyRL mounted.
set -euo pipefail

cd /workspace/SkyRL

# Primus containers ship Megatron-LM as source under /workspace.
bootstrap_megatron_paths() {
  for path in \
    /workspace/Megatron-LM \
    /workspace/Primus/third_party/Megatron-LM \
    /workspace/Primus; do
    if [ -d "$path" ]; then
      export PYTHONPATH="${path}:${PYTHONPATH:-}"
    fi
  done
}
bootstrap_megatron_paths
export NVTE_USE_ROCM="${NVTE_USE_ROCM:-1}"
export NVTE_USE_HIPBLASLT="${NVTE_USE_HIPBLASLT:-1}"

echo "--- container env ---"
python3 - <<'PY'
import importlib.util
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
print("megatron.core", bool(importlib.util.find_spec("megatron.core")))
print("megatron.bridge", bool(importlib.util.find_spec("megatron.bridge")))
print("transformer_engine", bool(importlib.util.find_spec("transformer_engine")))
print("ray", bool(importlib.util.find_spec("ray")))
print("vllm", bool(importlib.util.find_spec("vllm")))
print("primus_cli", __import__("os").path.exists("/workspace/Primus/runner/primus-cli"))
PY

echo
echo "--- install runtime deps (ray) ---"
python3 -m pip install --no-cache-dir "ray[default]==2.57.0" 2>/dev/null || true

PY_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
SKYRL_OK=0
if [ "$PY_MAJOR" -gt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 11 ]; }; then
  echo
  echo "--- install SkyRL (Python ${PY_MAJOR}.${PY_MINOR}) ---"
  if command -v uv >/dev/null 2>&1; then
    cp docker/pyproject.primus.toml pyproject.toml
    uv pip install --system --no-cache-dir -e ".[primus-megatron]" || \
      uv pip install --system --no-cache-dir -e ".[skyrl-train]"
  else
    cp docker/pyproject.primus.toml pyproject.toml
    python3 -m pip install --no-cache-dir -e ".[primus-megatron]" || \
      python3 -m pip install --no-cache-dir -e ".[skyrl-train]"
  fi
  SKYRL_OK=1
else
  echo
  echo "--- skip SkyRL install: Python ${PY_MAJOR}.${PY_MINOR} < 3.11 (need py3.11+ image for SkyRL package) ---"
fi

echo
echo "--- smoke test ---"
if [ "$SKYRL_OK" -eq 1 ]; then
  python3 integrations/primus_amd/smoke_test.py "$@"
else
  python3 integrations/primus_amd/smoke_test.py --phase env --phase megatron_core --phase transformer_engine --phase megatron_bridge --phase distributed "$@"
fi
