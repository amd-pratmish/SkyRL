# Draft PR: AMD ROCm support for Megatron training + vLLM inference

**Target:** [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)  
**Status:** Draft — not opened upstream yet. Validate on fork first.

---

## Title

`feat: AMD ROCm support for Megatron GRPO training and vLLM inference`

---

## Summary

- Enable end-to-end **GRPO** on AMD Instinct GPUs using the existing Megatron path (`trainer.strategy=megatron`) and **vLLM** for rollouts—no new training backend.
- Add `integrations/rocm_amd/` with install scripts, smoke tests, Docker recipe, and GSM8K example (`run_gsm8k_megatron_rocm.sh`).
- Patch SkyRL for **vLLM 0.20** API moves, **Ray 2.57** scheduling/collective imports, and **ROCm** device/runtime defaults (HIP env, `VLLM_USE_V1=0`, weight-sync RPC name clash).

Validated: colocated Megatron GRPO on 2 MI355X GPUs (Qwen2.5-0.5B, GSM8K subset)—rollout, collective-based (RCCL/NCCL) weight-sync, policy update. Integration targets MI300X, MI325X, MI350X, and MI355X (`gfx942` / `gfx950`).

---

## SkyRL code changes (core)

| Area | Files | Change |
|------|-------|--------|
| Ray 2.57 | `ray_wrapped_inference_engine.py`, `worker.py`, `utils.py`, tests | `PlacementGroupSchedulingStrategy` import fallback |
| ROCm runtime | `skyrl/train/utils/utils.py` | Skip `NVTE_FUSED_ATTN=0` on HIP; `ray.init(num_gpus=...)`; default `VLLM_USE_V1=0` + `VLLM_TARGET_DEVICE=rocm` on ROCm |
| vLLM compat | `vllm_import_compat.py` (new) | Shims for vLLM 0.20 `serve.engine.protocol` imports |
| vLLM devices | `vllm_engine.py`, `vllm_server_actor.py` | HIP/ROCR env; do not use `CUDA_VISIBLE_DEVICES` on ROCm |
| Weight sync | `layerwise_reload.py`, `vllm_engine.py`, `remote_inference_client.py`, tests | `skyrl_start_weight_update` / `skyrl_finish_weight_update` (vLLM 0.20 Worker clash) |
| Ray collective | `inference_engines/utils.py` | Fallback import for Ray 2.57 `get_address_and_port` |

---

## Integration additions

| Path | Purpose |
|------|---------|
| `integrations/rocm_amd/` | Install, validate, smoke test, container runner |
| `docker/Dockerfile.rocm-megatron` | Optional image extending ROCm base + Megatron-Bridge |
| `docker/pyproject.rocm.toml` | `rocm-megatron` optional dependency set |
| `examples/train/gsm8k/run_gsm8k_megatron_rocm.sh` | Colocated Megatron GRPO smoke |

---

## Requirements for users

- AMD Instinct GPU: MI300X, MI325X (`gfx942`), MI350X, or MI355X (`gfx950`)
- ROCm Docker/Podman (`/dev/kfd`, `/dev/dri`) with a base image whose PyTorch build matches your GPU ISA
- vLLM **built from source** on that torch (`build_vllm_rocm.sh`, default `PYTORCH_ROCM_ARCH=gfx942;gfx950`) — PyPI `vllm` pulls CUDA torch

Megatron pins: megatron-core `71e418ea…`, megatron-bridge `91a15142…` (see `install_megatron_bridge.sh`).

---

## Test plan

- [ ] `bash integrations/rocm_amd/run_in_container.sh`
- [ ] `python3 integrations/rocm_amd/gpu_support.py` (MI300X / MI325X / MI350X / MI355X)
- [ ] `bash integrations/rocm_amd/install_full_stack.sh`
- [ ] `bash integrations/rocm_amd/validate_megatron_rocm.sh`
- [ ] `bash integrations/rocm_amd/run_smoke_test.sh`
- [ ] `bash examples/train/gsm8k/run_gsm8k_megatron_rocm.sh` (2 GPUs, `NUM_GPUS=2`)
- [ ] Confirm training loop: vLLM generate → collective-based (RCCL/NCCL) weight-sync → Megatron policy step
