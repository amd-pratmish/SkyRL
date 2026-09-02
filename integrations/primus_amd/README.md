# SkyRL + Megatron GRPO on AMD (Approach A)

Run SkyRL's existing **Megatron-Bridge** training path (`trainer.strategy=megatron`) on AMD Instinct MI300X / MI325X / MI355X using the **`rocm/primus:v26.4`** ROCm container as the runtime.

**Status:** Megatron GRPO validated end-to-end on MI355X (rollout + NCCL weight sync + policy update). See `findings.md` and **`WORKFLOW.md`** for architecture and how Primus/Megatron/vLLM fit together.

**Branch:** [feat/primus-amd-approach-a](https://github.com/amd-pratmish/SkyRL/tree/feat/primus-amd-approach-a)

---

## Read first

| Doc | Purpose |
|-----|---------|
| **[WORKFLOW.md](WORKFLOW.md)** | Full architecture: GRPO loop, which Megatron, Primus vs image, vLLM changes |
| **[findings.md](findings.md)** | Install recipe, pins, error fixes, cluster run commands |

---

## Quick start (cluster)

```bash
# Full GRPO (2 GPUs, rad partition, max 2h)
SLURM_GPUS=2 SLURM_PARTITION=rad SLURM_TIME=02:00:00 SLURM_MEM=128G \
  bash integrations/primus_amd/run_grpo_amd_cluster.sh
```

Inside the container (or for manual steps):

```bash
# Megatron-Bridge + SkyRL
bash integrations/primus_amd/approach_a_install.sh

# vLLM ROCm wheel (build once)
bash integrations/primus_amd/build_vllm_rocm.sh

# Full stack + verify
bash integrations/primus_amd/approach_a_full_install.sh

# Validate bridge on GPU
bash integrations/primus_amd/run_approach_a_validate.sh
```

Use **`rad-vultr-mi355x-*`** nodes. `lux-mi355x` / `rad-burst` may hit ROCm ISA mismatch (`register fat binary failed`).

---

## What Approach A is (and is not)

| | Approach A (shipped) |
|--|----------------------|
| **Training** | SkyRL `megatron_worker` + NVIDIA Megatron-Bridge + megatron-core 0.19 |
| **Rollouts** | vLLM built from `main` on Primus ROCm torch |
| **Weight sync** | NCCL (`weight_sync_backend=nccl`) |
| **Runtime** | Docker image `rocm/primus:v26.4` (torch + Transformer Engine) |
| **Not used** | Primus framework as a separate `trainer.strategy` |

---

## Smoke tests (Phase 0/1)

```bash
python integrations/primus_amd/smoke_test.py --all
python integrations/primus_amd/smoke_test.py --phase bridge --model Qwen/Qwen2.5-0.5B-Instruct
bash integrations/primus_amd/run_smoke_test.sh
```

| Phase | Checks |
|-------|--------|
| `env` | ROCm PyTorch, GPU visibility, Ray |
| `megatron_bridge` | `AutoBridge` import + optional HF bridge |
| `skyrl` | `megatron_worker` import |
| `vllm` | vLLM import (rollouts) |

---

## Key integration files

| File | Purpose |
|------|---------|
| `WORKFLOW.md` | Architecture and GRPO workflow |
| `findings.md` | Runbook and troubleshooting |
| `approach_a_install.sh` | Megatron-Bridge + SkyRL install |
| `approach_a_full_install.sh` | + vLLM wheel + Ray + verify |
| `build_vllm_rocm.sh` | Build vLLM on Primus torch |
| `verify_vllm_skyrl_compat.py` | SkyRL ↔ vLLM 0.20 API check |
| `run_grpo_amd_cluster.sh` | Slurm + Docker GRPO launcher |
| `run_grpo_amd_megatron.sh` | GSM8K Megatron GRPO config |
| `ray_amd_preflight.py` | Ray GPU + placement group check |
| `vllm_llm_smoke.sh` | Single-GPU vLLM LLM smoke (no Ray) |

---

## Docker image (optional)

Build a SkyRL layer on Primus:

```bash
PRIMUS_IMAGE=rocm/primus:v26.4 bash integrations/primus_amd/build_image.sh
```

Most cluster runs use `rocm/primus:v26.4` directly with the repo bind-mounted (see `run_grpo_amd_cluster.sh`).
