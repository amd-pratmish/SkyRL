#!/usr/bin/env python3
"""Single-GPU Megatron-Core sanity forward on Primus (no Megatron-Bridge)."""

from __future__ import annotations

import os
import sys

for path in ("/workspace/Megatron-LM", "/workspace/Primus/third_party/Megatron-LM", "/workspace/Primus"):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec


def main() -> int:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not torch.cuda.is_available():
        print("FAIL: no HIP/CUDA device")
        return 1

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)

    config = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=False,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        fp16=False,
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(num_experts=None),
        vocab_size=1024,
        max_sequence_length=128,
    ).cuda().bfloat16()

    batch, seqlen = 2, 32
    tokens = torch.randint(0, 1024, (batch, seqlen), device="cuda")
    positions = torch.arange(seqlen, device="cuda").unsqueeze(0).expand(batch, -1)
    attention_mask = torch.ones((batch, 1, seqlen, seqlen), device="cuda", dtype=torch.bool)

    out = model(tokens, positions, attention_mask)
    loss = out.sum()
    loss.backward()
    print("PASS: megatron forward/backward", float(loss.detach().cpu()))

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
