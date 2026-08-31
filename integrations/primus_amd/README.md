# SkyRL + Primus Megatron on AMD — Exploration

Phase-0/1 scaffold for validating whether SkyRL's existing Megatron path (Megatron-Bridge workers) can run on AMD Instinct GPUs via the [Primus](https://github.com/AMD-AGI/Primus) ROCm stack.

**Status:** exploration — not yet integrated into `trainer.strategy`.

## Goal

Determine blockers for `trainer.strategy=megatron` (or a future `primus` strategy) on MI300X/MI325X/MI355X, before wiring RL training loops.

## Quick start (AMD node with Docker/Podman)

```bash
cd /path/to/SkyRL

# 1. Build image (uses rocm/primus:v26.4 base)
bash integrations/primus_amd/build_image.sh

# 2. Run container with ROCm devices
docker run --rm -it --network host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  skyrl-primus-amd

# 3. Inside container — run smoke tests
bash integrations/primus_amd/run_smoke_test.sh
```

Override the Primus base tag if your cluster pins a different release:

```bash
PRIMUS_IMAGE=rocm/primus:v26.2 bash integrations/primus_amd/build_image.sh
```

## What the smoke test checks

| Phase | Checks |
|-------|--------|
| `env` | ROCm PyTorch (`torch.version.hip`), GPU visibility, Ray, Primus CLI |
| `megatron_core` | `megatron.core` imports from Primus image |
| `transformer_engine` | TE + `transformer_engine.pytorch` on ROCm |
| `megatron_bridge` | **Critical:** `megatron.bridge.AutoBridge` import + optional HF model bridge |
| `skyrl` | SkyRL worker/strategy imports; full `megatron_worker` import |
| `distributed` | Single-GPU `nccl` (RCCL) process group init |
| `vllm` | vLLM import (needed for rollouts) |

Run individual phases:

```bash
python integrations/primus_amd/smoke_test.py --phase megatron_bridge --model Qwen/Qwen2.5-0.5B-Instruct
```

Reports land in `/tmp/skyrl-primus-exploration/smoke_*.json` by default.

## Files added by this exploration

| File | Purpose |
|------|---------|
| `docker/Dockerfile.primus-amd` | SkyRL on `rocm/primus` base |
| `docker/pyproject.primus.toml` | Deps without NVIDIA CUDA pins |
| `integrations/primus_amd/smoke_test.py` | Compatibility checker |
| `integrations/primus_amd/run_smoke_test.sh` | One-shot test runner |
| `integrations/primus_amd/build_image.sh` | Image build helper |
| `integrations/primus_amd/findings.md` | Record results per run |

## Decision tree after smoke test

```
rocm/primus:v26.4 + approach_a_install.sh
  → megatron-bridge imports?
      yes → AutoBridge.from_hf + megatron_worker + provide_distributed_model
              → Approach A WORKS (validated 2026-08-28 on MI355X)
              → next: GRPO smoke with vLLM ROCm + nccl weight sync
      no  → see approach_a_install.sh troubleshooting / Approach B
```

## Approach A install (recommended)

On **`rocm/primus:v26.4`** (Python 3.12):

```bash
bash integrations/primus_amd/approach_a_install.sh
bash integrations/primus_amd/run_approach_a_validate.sh
```

See `findings.md` for the full recipe and pins.

## Known CUDA assumptions in SkyRL Megatron path

These may need ROCm guards before training works:

- `megatron_worker.py`: `torch.cuda.current_device()` for weight export
- `worker.py`: `backend="cpu:gloo,cuda:nccl"` — usually works via RCCL on HIP
- `prepare_runtime_environment()`: `CUDA_DEVICE_MAX_CONNECTIONS`, `NVTE_FUSED_ATTN`
- Weight sync: CUDA IPC for colocated paths; use `weight_sync_backend=nccl` or `delta` on AMD
- Triton fused LM-head backend

## Next phases (after smoke test passes)

1. **Phase 2 — SFT step:** single GPU `sft_trainer.py` with `trainer.strategy=megatron` on a 0.5B–1.5B model
2. **Phase 3 — Weight sync:** policy → vLLM ROCm with `weight_sync_backend=nccl`
3. **Phase 4 — GRPO:** minimal GSM8K run mirroring `examples/train/gsm8k/run_gsm8k.sh`

Record outcomes in `findings.md`.
