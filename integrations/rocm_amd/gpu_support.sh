#!/usr/bin/env bash
# AMD Instinct GPU support helpers (MI300X, MI325X, MI350X, MI355X).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default vLLM build targets all supported Instinct ISAs (CDNA3 + CDNA4).
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942;gfx950}"

rocm_amd_detect_gpu() {
  python3 "${SCRIPT_DIR}/gpu_support.py"
}

rocm_amd_check_gpu_supported() {
  python3 "${SCRIPT_DIR}/gpu_support.py" >/dev/null
}

rocm_amd_print_gpu_matrix() {
  cat <<'EOF'
Supported AMD Instinct GPUs:
  MI300X, MI325X  -> gfx942 (CDNA3)
  MI350X, MI355X  -> gfx950 (CDNA4)

Use a ROCm base image whose PyTorch build matches your GPU ISA.
Set PYTORCH_ROCM_ARCH when building vLLM (default: gfx942;gfx950).
EOF
}
