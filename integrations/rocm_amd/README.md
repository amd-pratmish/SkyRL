# SkyRL on AMD ROCm GPUs

Run SkyRL **Megatron training** (`trainer.strategy=megatron`) and **vLLM inference** on AMD Instinct GPUs (MI300X / MI325X / MI355X).

Uses the existing SkyRL `megatron_worker` + NVIDIA Megatron-Bridge on ROCm—no separate training backend.

| Doc | Purpose |
|-----|---------|
| [WORKFLOW.md](WORKFLOW.md) | Architecture: GRPO loop, stack components, vLLM integration |
| [UPSTREAM_PR_DRAFT.md](UPSTREAM_PR_DRAFT.md) | Draft PR summary for upstreaming to NovaSky-AI/SkyRL |

---

## Requirements

- AMD Instinct GPU with ROCm (MI300X class or newer recommended)
- Docker or Podman with `/dev/kfd` and `/dev/dri` access
- Base image with ROCm PyTorch + Transformer Engine, e.g. **`rocm/primus:v26.4`**

The GPU architecture in the image must match your hardware (e.g. gfx950 for MI355X). Mismatched ROCm ISA builds can fail at runtime with `register fat binary failed`.

---

## Quick start

### 1. Enter a ROCm container with this repo mounted

```bash
bash integrations/rocm_amd/run_in_container.sh
```

Override GPU count: `ROCM_GPUS=2 bash integrations/rocm_amd/run_in_container.sh`

### 2. Install the full stack (inside container)

```bash
bash integrations/rocm_amd/install_full_stack.sh
```

This installs Megatron-Bridge, builds or caches vLLM for ROCm, and verifies SkyRL ↔ vLLM API compatibility.

### 3. Validate Megatron on one GPU

```bash
bash integrations/rocm_amd/validate_megatron_rocm.sh
```

### 4. Run Megatron GRPO smoke (GSM8K)

```bash
bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh
```

Or from the host:

```bash
ROCM_GPUS=2 bash integrations/rocm_amd/run_in_container.sh \
  bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh
```

---

## Optional: pre-built image

```bash
bash integrations/rocm_amd/build_image.sh
# IMAGE=skyrl-rocm-megatron ROCM_BASE_IMAGE=rocm/primus:v26.4
```

---

## Smoke tests

```bash
bash integrations/rocm_amd/run_smoke_test.sh
# or: python integrations/rocm_amd/smoke_test.py --all
```

---

## Key scripts

| Script | Purpose |
|--------|---------|
| `run_in_container.sh` | Docker/Podman launcher with ROCm devices |
| `install_megatron_bridge.sh` | megatron-core 0.19 + Megatron-Bridge + SkyRL |
| `install_full_stack.sh` | Above + vLLM ROCm wheel + verification |
| `build_vllm_rocm.sh` | Build vLLM from source on container torch |
| `verify_vllm_skyrl_compat.py` | Check vLLM 0.20 API imports used by SkyRL |
| `validate_megatron_rocm.sh` | Single-GPU Megatron-Bridge validation |
| `ray_preflight.sh` | Ray GPU visibility + placement group check |
| `vllm_llm_smoke.sh` | Single-GPU vLLM `LLM` init (no Ray) |

---

## vLLM on ROCm (important)

1. Do **not** `pip install vllm` from PyPI—it pulls CUDA PyTorch and breaks the stack.
2. Build with `build_vllm_rocm.sh` against your container’s torch (cached under `.vllm_rocm_cache/`).
3. Use `verify_vllm_skyrl_compat.py` before training.

---

## Megatron pins

| Package | Git revision |
|---------|----------------|
| megatron-core | `71e418ea7d7b3a6c9a53238c543c3e0b43e11026` |
| megatron-bridge | `91a15142a4b4442a8d46ab539d1b923bd08570d0` |

---

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `No module named vllm.entrypoints.openai.engine` | Rebuild vLLM; run `verify_vllm_skyrl_compat.py` |
| `register fat binary failed` | ROCm ISA mismatch—use a base image built for your GPU arch |
| Ray reports 0 GPUs | Set `NUM_GPUS`; run `ray_preflight.sh`; check `HIP_VISIBLE_DEVICES` |
| HIP agent crash / empty device | Unset `CUDA_VISIBLE_DEVICES`; use HIP/ROCR only |
| vLLM engine core init failed | `VLLM_USE_V1=0`, `async_engine=false` (see example script) |
| Worker `finish_weight_update` clash | Use SkyRL with `skyrl_*` weight-sync RPC names (vLLM 0.20+) |

Weight sync uses `weight_sync_backend=nccl` in config; on ROCm PyTorch routes collectives through **RCCL** (NCCL-compatible API).
