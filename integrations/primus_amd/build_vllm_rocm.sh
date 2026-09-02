#!/usr/bin/env bash
# Build vLLM from source against the Primus ROCm PyTorch (do NOT pip install vllm — it pulls CUDA torch).
set -euo pipefail

# SkyRL pins vllm==0.20.2 API; build vLLM main (v0.20.2 tag fails GPTQ compile on ROCm 7.14).
VLLM_TAG="${VLLM_TAG:-main}"
VLLM_SRC="${VLLM_SRC:-/tmp/vllm-rocm-build-${VLLM_TAG}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKYRL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERIFY_PY="${SKYRL_ROOT}/integrations/primus_amd/verify_vllm_skyrl_compat.py"
CACHE_WHEEL_DIR="${SKYRL_ROOT}/integrations/primus_amd/.vllm_rocm_cache/wheels"

TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
echo "Building vLLM ${VLLM_TAG} for torch ${TORCH_VER}"
mkdir -p "${CACHE_WHEEL_DIR}"

wheel_is_skyrl_compatible() {
  local wheel="$1"
  python3 -m pip install --no-cache-dir --force-reinstall --no-deps "${wheel}" -q
  python3 "${VERIFY_PY}"
}

if compgen -G "${CACHE_WHEEL_DIR}/vllm-0.20*.whl" >/dev/null; then
  WHEEL="$(ls -1t "${CACHE_WHEEL_DIR}"/vllm-0.20*.whl | head -1)"
elif compgen -G "${CACHE_WHEEL_DIR}/vllm-*.whl" >/dev/null; then
  WHEEL="$(ls -1t "${CACHE_WHEEL_DIR}"/vllm-*.whl | head -1)"
  echo "Checking cached wheel ${WHEEL}"
  if wheel_is_skyrl_compatible "${WHEEL}"; then
    python3 -c "
import glob, os, vllm
d = os.path.dirname(vllm.__file__)
so = glob.glob(os.path.join(d, '*.so')) + glob.glob(os.path.join(d, '**', '*.so'), recursive=True)
print('vllm', vllm.__version__, 'native_exts', len(so))
"
    exit 0
  fi
  echo "Stale/incompatible cached wheel; rebuilding at ${VLLM_TAG}"
fi

export VLLM_TARGET_DEVICE=rocm
# MI355X = gfx950 (cdna4); include gfx942 for compatibility.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950;gfx942}"
export MAX_JOBS="${MAX_JOBS:-16}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.20.2+rocm.rocm714}"

if [ ! -d "${VLLM_SRC}/.git" ]; then
  git clone --depth 1 --branch "${VLLM_TAG}" https://github.com/vllm-project/vllm.git "${VLLM_SRC}"
fi
cd "${VLLM_SRC}"
git fetch --depth 1 origin "${VLLM_TAG}" 2>/dev/null || git fetch origin "${VLLM_TAG}"
git checkout "${VLLM_TAG}"

python3 -m pip install --no-cache-dir -q \
  "setuptools==79.0.1" "setuptools-scm==10.2.1" setuptools-rust \
  ninja cmake wheel "packaging<26" pybind11 numba numpy psutil

mkdir -p /tmp/vllm-wheels
python3 -m pip wheel --no-deps --no-build-isolation -w /tmp/vllm-wheels .

WHEEL="$(ls -1 /tmp/vllm-wheels/vllm-*.whl | tail -1)"
echo "Installing ${WHEEL}"
python3 -m pip install --no-cache-dir --force-reinstall --no-deps "${WHEEL}"

# Cache wheel in repo for faster re-runs.
cp -f "${WHEEL}" "${CACHE_WHEEL_DIR}/"

# Runtime deps (pinned subset; no torch).
python3 -m pip install --no-cache-dir -q \
  "transformers>=5.10.4" "numba==0.65.0" "msgspec==0.21.1" "cbor2==6.1.4" \
  "xgrammar==0.2.3" "outlines_core==0.2.14" "llguidance==1.7.6" \
  "compressed-tensors==0.17.0" "fastsafetensors==0.3.3" \
  "partial-json-parser==0.2.1.1.post7" "openai==3.3.1" "pybase64==1.5.0" \
  "pyzmq==27.2.0" "watchfiles==1.2.0" "blake3==1.0.9" "depyf==0.20.0" \
  "mistral_common>=1.11.6" "opencv-python-headless==5.0.0.93" \
  "prometheus-fastapi-instrumentator==8.1.0" "setproctitle==1.3.7" \
  "lm-format-enforcer==0.11.3" "lark==1.2.2" "ijson==3.5.1"

AFTER="$(python3 -c 'import torch; print(torch.__version__)')"
if [ "$AFTER" != "$TORCH_VER" ]; then
  echo "ERROR: torch changed during vLLM build ${TORCH_VER} -> ${AFTER}"
  exit 1
fi

cd "${SKYRL_ROOT}"
python3 "${VERIFY_PY}"
python3 -c "
import glob, os, vllm
d = os.path.dirname(vllm.__file__)
so = glob.glob(os.path.join(d, '*.so')) + glob.glob(os.path.join(d, '**', '*.so'), recursive=True)
if not so:
    raise SystemExit(f'vLLM installed but no native extensions under {d}')
print('built vllm', vllm.__version__, 'native_exts', len(so))
"
