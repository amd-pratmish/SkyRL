# SkyRL Megatron GRPO on AMD — Architecture & Workflow

This document explains how the **Approach A** port works: which Megatron stack is used, how Primus fits in, what vLLM changes were required, and how a full GRPO step runs on MI355X.

**Status:** validated end-to-end on `rad-vultr-mi355x-*` (see `findings.md` and log `grpo_amd_20260901T234529Z.log`).

**Branch:** `feat/primus-amd-approach-a` on [amd-pratmish/SkyRL](https://github.com/amd-pratmish/SkyRL).

---

## Big picture: one GRPO step

SkyRL reuses the existing **CUDA Megatron path** on AMD. There is **no** `trainer.strategy=primus`. The port is **Approach A**: `trainer.strategy=megatron` + `megatron_worker`, running on a ROCm runtime delivered by the **Primus container image** (`rocm/primus:v26.4`).

```
┌─────────────────────────────────────────────────────────────────┐
│  Ray (skyrl_entrypoint on MI355X)                               │
│                                                                 │
│  1. ROLLOUT          vLLM Ray actors (sync LLM, 1 per GPU)      │
│     prompts → completions                                       │
│                                                                 │
│  2. FORWARD          Megatron policy + ref workers              │
│     logprobs, values, rewards → GRPO advantages                 │
│                                                                 │
│  3. WEIGHT SYNC      NCCL broadcast: policy → vLLM workers      │
│                                                                 │
│  4. POLICY UPDATE    Megatron GRPO loss + optimizer step        │
└─────────────────────────────────────────────────────────────────┘
```

With `colocate_all=true` on 2 GPUs, policy, ref, and vLLM share the same devices (fractional Ray GPU scheduling + sleep/wake on vLLM).

**Validated smoke timings (step 4, Qwen2.5-0.5B, GSM8K 64 prompts):**

| Phase | Component | ~time |
|-------|-----------|-------|
| Generate | vLLM sync `LLM` | ~1.1s |
| Sync weights | NCCL policy → vLLM | ~3.7s |
| Policy train | Megatron GRPO | ~3.3s |

Key config: `generator.inference_engine.weight_sync_backend=nccl`, `async_engine=false`, `VLLM_USE_V1=0` on ROCm.

---

## Which Megatron?

This is **NVIDIA’s Megatron stack**, not a separate AMD Megatron fork or Primus training API.

| Component | Source | Role |
|-----------|--------|------|
| **megatron-core 0.19** | [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) at pinned commit | Distributed training (TP/PP, TE attention) |
| **Megatron-Bridge** | [NVIDIA-NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) at pinned commit | HF ↔ Megatron (`AutoBridge.from_hf`), model provider |
| **SkyRL `megatron_worker`** | `skyrl/backends/skyrl_train/workers/megatron/` | RL worker: forward, GRPO loss, optimizer, weight export for vLLM |
| **Transformer Engine** | Bundled in Primus image (`NVTE_USE_ROCM=1`) | Fused ops on ROCm |

Training entry is the standard SkyRL path:

```python
trainer.strategy=megatron  # → megatron_worker + MegatronStrategy
```

The worker imports `AutoBridge` from `megatron.bridge` — same code path as CUDA Megatron training, with ROCm-specific env and device handling.

### Pins (Approach A)

| Package | Revision |
|---------|----------|
| megatron-core | `71e418ea7d7b3a6c9a53238c543c3e0b43e11026` |
| megatron-bridge | `91a15142a4b4442a8d46ab539d1b923bd08570d0` |

Install bridge with `--no-deps` + `nvidia-modelopt`. Remove legacy `Megatron-LM` from `PYTHONPATH` (Primus image ships 0.16 on path; SkyRL needs 0.19 from git).

---

## Primus: what it is and whether you need it

Primus appears in **two different senses** in this project.

### 1. Primus **container image** (`rocm/primus:v26.4`) — **required (or equivalent)**

What you run in Docker on Slurm compute nodes. It provides:

- ROCm PyTorch `2.12.0+rocm7.14` matched to MI355X
- Transformer Engine built for ROCm
- ROCm user-space / HIP stack aligned with the node

You should not substitute arbitrary `rocm/vllm-dev` images without rebuilding vLLM against that exact torch (ABI breaks). Integration scripts assume **Primus torch + TE**.

### 2. Primus **framework** ([AMD-AGI/Primus](https://github.com/AMD-AGI/Primus)) — **not used for Approach A**

Approach A does **not** add a Primus training backend or Primus-specific worker. The folder `integrations/primus_amd` means “SkyRL on AMD **via the Primus ROCm image**,” not “SkyRL calls Primus training APIs.”

A future **Approach B** could wire Primus’s own Megatron runtime as a new `trainer.strategy`; that was explored early and is **not** what shipped.

**Summary:** Primus image = **runtime environment**. Megatron-Bridge + SkyRL `megatron_worker` = **training logic**.

---

## End-to-end run workflow

### 1. Cluster launch

```bash
SLURM_GPUS=2 SLURM_PARTITION=rad SLURM_TIME=02:00:00 SLURM_MEM=128G \
  bash integrations/primus_amd/run_grpo_amd_cluster.sh
```

- Slurm allocates MI355X GPUs on `rad-vultr-mi355x-*` nodes (avoid `lux-mi355x` / `rad-burst` ISA mismatch).
- Docker runs `rocm/primus:v26.4` with repo mounted at `/workspace/SkyRL`.
- Sets `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`; unsets `CUDA_VISIBLE_DEVICES`.

### 2. Install stack (`approach_a_full_install.sh`)

1. `approach_a_install.sh` — megatron-core, Megatron-Bridge, SkyRL `[primus-megatron]`
2. `build_vllm_rocm.sh` or cached wheel — vLLM built on Primus torch
3. Ray 2.57, GSM8K deps, `verify_vllm_skyrl_compat.py`

### 3. GRPO (`run_grpo_amd_megatron.sh`)

1. Ray + vLLM preflight (`ray_amd_preflight.sh`)
2. Prepare tiny GSM8K parquet subset
3. `python -m skyrl.train.entrypoints.main_base` with Megatron + vLLM config

### 4. Inside `main_base`

1. **Ray init** with explicit `num_gpus`
2. **Create vLLM inference engines** — `VLLMRayActor` + sync `vllm.LLM` (one engine per GPU)
3. **Sleep vLLM** (colocate memory pool)
4. **Create Megatron policy + ref** worker groups on same placement group
5. **GRPO loop** per batch:
   - `generate()` via vLLM
   - Postprocess rewards (GSM8K env)
   - Policy/ref forward for logprobs
   - GRPO advantage + loss
   - **Weight sync** (NCCL) policy → vLLM `WorkerWrap.load_weights`
   - Megatron policy `train_step`

---

## vLLM: build vs SkyRL code changes

There are **no vLLM source forks** in this repo. vLLM is **built from `main`** inside the Primus container so HIP extensions link against the same torch as Megatron-TE.

### Build / install rules

| Rule | Reason |
|------|--------|
| Never `pip install vllm` from PyPI | Pulls CUDA torch; breaks TE |
| Build from vLLM `main`, not tag `v0.20.2` | GPTQ compile fails on ROCm 7.14 |
| Do not copy wheels from `rocm/vllm-dev` | `vllm._C` ABI mismatch |
| Cache wheel under `.vllm_rocm_cache/wheels/` | Faster reruns |
| Run `verify_vllm_skyrl_compat.py` before GRPO | Confirms SkyRL 0.20 API surface |

```bash
bash integrations/primus_amd/build_vllm_rocm.sh   # once
python3 integrations/primus_amd/verify_vllm_skyrl_compat.py
```

### SkyRL patches for vLLM 0.20 + ROCm

| File / setting | Change | Why |
|----------------|--------|-----|
| `vllm_import_compat.py` | Import shims for `serve.engine.protocol`, `OpenAIServingRender` | vLLM 0.20 moved OpenAI entrypoints |
| `vllm_engine.py` | `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`; pop `CUDA_VISIBLE_DEVICES` on ROCm | HIP agent crashes / empty device string |
| `utils.py` | `VLLM_USE_V1=0` default on HIP; `VLLM_TARGET_DEVICE=rocm` | v1 multiprocess engine core failed on colocated GRPO |
| GRPO smoke config | `async_engine=false` | Sync `vllm.LLM` instead of `AsyncLLMEngine` |
| `layerwise_reload.py` | `skyrl_start_weight_update` / `skyrl_finish_weight_update` | vLLM 0.20 Worker already has `finish_weight_update` |
| `WorkerWrap` | NCCL weight receiver + `load_weights` collective RPC | Policy → inference weight transfer |

vLLM remains stock vLLM built for ROCm; SkyRL changes are **integration and configuration** only.

---

## Weight sync path (policy → vLLM)

Legacy path (`_SKYRL_USE_NEW_INFERENCE=0`, used in AMD GRPO smoke):

1. Policy Megatron worker exports weight chunks
2. **NCCL broadcast** (`weight_sync_backend=nccl`) to inference ranks
3. vLLM `WorkerWrap` receives via `init_weight_update_communicator`
4. Bracket: `skyrl_start_weight_update` → `load_weights` (per chunk) → `skyrl_finish_weight_update`
5. Layerwise reload finalizes checkpoint-format weights once per sync (not per chunk)

CUDA IPC weight sync is not the default on AMD; use NCCL.

---

## Mental model (one line each)

| Piece | Role |
|-------|------|
| **SkyRL** | RL orchestration (Ray, GRPO, placement, weight sync API) |
| **Megatron-Bridge + megatron-core** | Policy/ref **training** |
| **vLLM (ROCm build)** | Rollout **inference** |
| **Primus image** | ROCm **environment** (torch + TE), not the training algorithm |
| **NCCL / RCCL** | Copies updated policy weights into vLLM after each train step |

---

## Related docs

| File | Contents |
|------|----------|
| `findings.md` | Install recipe, error table, validated run notes |
| `README.md` | Quick start, smoke tests, file index |
| `approach_a_install.sh` | Megatron-Bridge install script |
| `run_grpo_amd_cluster.sh` | Slurm + Docker launcher |
