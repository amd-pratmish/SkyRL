#!/usr/bin/env python3
"""Quick vLLM ROCm platform diagnostic (run inside ROCm container with GPU)."""
import os
import sys

import torch

print("torch", torch.__version__, "avail", torch.cuda.is_available(), "count", torch.cuda.device_count())
print("VLLM_TARGET_DEVICE", os.environ.get("VLLM_TARGET_DEVICE"))
print("VLLM_USE_V1", os.environ.get("VLLM_USE_V1"))
print("HIP_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES"))

import vllm

print("vllm", vllm.__version__)
from vllm.platforms import current_platform

print(
    "platform",
    current_platform.__class__.__name__,
    "device_type",
    repr(getattr(current_platform, "device_type", None)),
    "dispatch",
    getattr(current_platform, "dispatch_key", None),
)

if os.environ.get("TRY_LLM") == "1":
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.35,
        trust_remote_code=True,
    )
    out = llm.generate(["Hello"], SamplingParams(max_tokens=4, temperature=0))
    print("generated", out[0].outputs[0].text)
    print("PASS: LLM init")
