#!/usr/bin/env bash
# Quick vLLM LLM init smoke inside Primus container (sync engine, no Ray).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${MODEL_NAME:=Qwen/Qwen2.5-0.5B-Instruct}"
: "${HIP_VISIBLE_DEVICES:=0}"

export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
unset CUDA_VISIBLE_DEVICES

python3 - <<'PY'
import os
import torch
import vllm

print("torch", torch.__version__, "hip", torch.version.hip)
print("vllm", vllm.__version__)
print("VLLM_USE_V1", os.environ.get("VLLM_USE_V1"))
print("HIP_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES"))
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))

from vllm import LLM, SamplingParams

model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
llm = LLM(
    model=model,
    tensor_parallel_size=1,
    enforce_eager=True,
    gpu_memory_utilization=0.35,
    trust_remote_code=True,
)
out = llm.generate(["Hello"], SamplingParams(max_tokens=8, temperature=0))
print("generated:", out[0].outputs[0].text)
print("PASS: vLLM LLM smoke")
PY
