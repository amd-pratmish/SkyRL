# SkyRL Megatron GRPO on AMD ROCm — Architecture & Workflow

How SkyRL runs **Megatron training** and **vLLM inference** on AMD Instinct GPUs using the existing CUDA Megatron code path (`trainer.strategy=megatron`), without a separate AMD training backend.

---

## Big picture: one GRPO step

```
┌─────────────────────────────────────────────────────────────────┐
│  Ray (skyrl_entrypoint on AMD GPU)                              │
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

With `colocate_all=true` on multiple GPUs, policy, ref, and vLLM share the same devices (fractional Ray GPU scheduling + sleep/wake on vLLM).

Typical smoke config (Qwen2.5-0.5B, GSM8K subset, 2 GPUs):

| Phase | Component | Notes |
|-------|-----------|-------|
| Generate | vLLM sync `LLM` | `async_engine=false` |
| Sync weights | NCCL policy → vLLM | `weight_sync_backend=nccl` |
| Policy train | Megatron GRPO | `trainer.strategy=megatron` |

Key ROCm settings: `VLLM_USE_V1=0`, `VLLM_TARGET_DEVICE=rocm`, unset `CUDA_VISIBLE_DEVICES` (use HIP/ROCR only).

---

## Which Megatron?

This uses **NVIDIA’s Megatron stack**, not a separate AMD Megatron fork.

| Component | Source | Role |
|-----------|--------|------|
| **megatron-core 0.19** | [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (pinned commit) | Distributed training (TP/PP, TE attention) |
| **Megatron-Bridge** | [NVIDIA-NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) (pinned commit) | HF ↔ Megatron (`AutoBridge.from_hf`), model provider |
| **SkyRL `megatron_worker`** | `skyrl/backends/skyrl_train/workers/megatron/` | RL worker: forward, GRPO loss, optimizer, weight export for vLLM |
| **Transformer Engine** | ROCm build in base image (`NVTE_USE_ROCM=1`) | Fused ops on ROCm |

Training entry:

```python
trainer.strategy=megatron  # → megatron_worker + MegatronStrategy
```

### Pins

| Package | Revision |
|---------|----------|
| megatron-core | `71e418ea7d7b3a6c9a53238c543c3e0b43e11026` |
| megatron-bridge | `91a15142a4b4442a8d46ab539d1b923bd08570d0` |

Install Megatron-Bridge with `--no-deps` + `nvidia-modelopt`. Remove legacy `Megatron-LM` from `PYTHONPATH` if your base image ships an older tree on path (SkyRL needs 0.19 from git). See `install_megatron_bridge.sh`.

---

## Runtime environment

Supported Instinct GPUs: **MI300X**, **MI325X** (`gfx942`, CDNA3) and **MI350X**, **MI355X** (`gfx950`, CDNA4).

Use a ROCm PyTorch image whose **GPU ISA matches your hardware**. A commonly tested CDNA4 base is `rocm/primus:v26.4`; for MI300X/MI325X use a CDNA3 (`gfx942`) ROCm PyTorch image. Equivalent images work if you rebuild vLLM against the same torch ABI.

Run `python3 integrations/rocm_amd/gpu_support.py` on the target node before GRPO.

Do not substitute arbitrary third-party vLLM images without rebuilding vLLM against that exact torch — `vllm._C` ABI mismatches are common.

---

## End-to-end recipe

### 1. Container

```bash
bash integrations/rocm_amd/run_in_container.sh
```

Bind-mounts the repo, sets `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`, unsets `CUDA_VISIBLE_DEVICES`.

### 2. Install stack

```bash
bash integrations/rocm_amd/install_full_stack.sh
```

1. `install_megatron_bridge.sh` — megatron-core, Megatron-Bridge, SkyRL `[rocm-megatron]`
2. `build_vllm_rocm.sh` or cached wheel under `.vllm_rocm_cache/wheels/`
3. Ray 2.57, GSM8K deps, `verify_vllm_skyrl_compat.py`

### 3. GRPO smoke (`examples/train/gsm8k/run_gsm8k_megatron_rocm.sh`)

1. Ray + vLLM preflight (`ray_preflight.sh`)
2. Prepare tiny GSM8K parquet subset
3. `python -m skyrl.train.entrypoints.main_base` with Megatron + vLLM config

### 4. Inside `main_base`

1. **Ray init** with explicit `num_gpus`
2. **Create vLLM inference engines** — `VLLMRayActor` + sync `vllm.LLM` (one engine per GPU)
3. **Sleep vLLM** (colocate memory pool)
4. **Create Megatron policy + ref** worker groups on same placement group
5. **GRPO loop** per batch: generate → reward → forward → advantage → weight sync → train step

---

## vLLM: build vs SkyRL code changes

There are **no vLLM source forks** in this repo. vLLM is **built from `main`** inside the ROCm container so HIP extensions link against the same torch as Megatron-TE.

### Build rules

| Rule | Reason |
|------|--------|
| Never `pip install vllm` from PyPI | Pulls CUDA torch; breaks TE |
| Build from vLLM `main`, not tag `v0.20.2` | GPTQ compile fails on some ROCm 7.14 stacks |
| Build on your container’s torch | `vllm._C` ABI must match |
| Cache wheel under `.vllm_rocm_cache/wheels/` | Faster reruns |
| Run `verify_vllm_skyrl_compat.py` before GRPO | Confirms SkyRL 0.20 API surface |

```bash
bash integrations/rocm_amd/build_vllm_rocm.sh
python3 integrations/rocm_amd/verify_vllm_skyrl_compat.py
```

### SkyRL patches for vLLM 0.20 + ROCm

| Area | Change | Why |
|------|--------|-----|
| `vllm_import_compat.py` | Import shims for moved OpenAI entrypoints | vLLM 0.20 API moves |
| `vllm_engine.py` | HIP/ROCR device env; pop `CUDA_VISIBLE_DEVICES` on ROCm | HIP agent crashes |
| `utils.py` | Default `VLLM_USE_V1=0` on HIP; `VLLM_TARGET_DEVICE=rocm` | v1 engine core failed on colocated GRPO |
| GRPO config | `async_engine=false` | Sync `vllm.LLM` instead of `AsyncLLMEngine` |
| `layerwise_reload.py` | `skyrl_start_weight_update` / `skyrl_finish_weight_update` | vLLM 0.20 Worker already has `finish_weight_update` |
| Ray imports | `PlacementGroupSchedulingStrategy` fallback | Ray 2.57 module move |
| `inference_engines/utils.py` | `get_address_and_port` import fallback | Ray 2.57 collective API |

vLLM remains stock vLLM built for ROCm; SkyRL changes are **integration and configuration** only.

---

## Weight sync (policy → vLLM)

Legacy path (`_SKYRL_USE_NEW_INFERENCE=0`, used in ROCm GRPO smoke):

1. Policy Megatron worker exports weight chunks
2. **NCCL broadcast** (`weight_sync_backend=nccl`) to inference ranks — on ROCm, PyTorch uses **RCCL** (NCCL-compatible API)
3. vLLM `WorkerWrap` receives via `init_weight_update_communicator`
4. Bracket: `skyrl_start_weight_update` → `load_weights` (per chunk) → `skyrl_finish_weight_update`
5. Layerwise reload finalizes checkpoint-format weights once per sync

CUDA IPC weight sync is not the default on AMD; use NCCL.

---

## Mental model

| Piece | Role |
|-------|------|
| **SkyRL** | RL orchestration (Ray, GRPO, placement, weight sync API) |
| **Megatron-Bridge + megatron-core** | Policy/ref **training** |
| **vLLM (ROCm build)** | Rollout **inference** |
| **ROCm base image** | Torch + TE **environment** |
| **NCCL / RCCL** | Policy weights copied into vLLM after each train step |

---

## Related docs

| File | Contents |
|------|----------|
| [README.md](README.md) | Quick start, scripts index, troubleshooting |
| [UPSTREAM_PR_DRAFT.md](UPSTREAM_PR_DRAFT.md) | Draft PR summary for upstreaming |
