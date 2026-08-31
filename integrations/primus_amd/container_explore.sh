#!/usr/bin/env bash
set -euo pipefail
cd /workspace/SkyRL
echo "=== explore primus:latest layout ==="
python3 -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'gpus', torch.cuda.device_count())"
echo "--- python paths ---"
python3 -c "import sys; print('\n'.join(sys.path))"
echo "--- find megatron/primus ---"
for d in /workspace /opt /usr/local /root; do
  find "$d" -maxdepth 4 -type d \( -name Megatron-LM -o -name Primus -o -name megatron \) 2>/dev/null | head -20
done
echo "--- pip packages (megatron/primus/te) ---"
python3 -m pip list 2>/dev/null | grep -iE "megatron|primus|transformer|flash|vllm|ray" || true
echo "--- env ---"
env | grep -iE "MEGATRON|PRIMUS|PYTHON|ROCM|HIP" | sort
echo "--- primus cli ---"
ls -la /workspace/Primus/runner/primus-cli 2>/dev/null || ls -la /Primus/runner/primus-cli 2>/dev/null || true
command -v primus-cli || true
