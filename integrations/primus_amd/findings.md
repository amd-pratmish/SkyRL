# Primus + SkyRL AMD port findings

## Status (2026-09-01)

| Layer | Status | Notes |
|-------|--------|-------|
| Megatron-Bridge + mcore 0.19 on `rocm/primus:v26.4` | **PASS** | `approach_a_validate.sh` |
| vLLM ROCm 0.20.2+rocm (built from `main`) | **PASS** | Cached wheel under `.vllm_rocm_cache/wheels/` |
| SkyRL + Ray 2.57 | **PASS** | Placement group + scheduling compat |
| Full install (`approach_a_full_install.sh`) | **PASS** | Bridge + vLLM wheel + deps |
| Megatron GRPO on MI355X (2 GPU colocate) | **PASS** | `grpo_amd_20260901T234529Z.log` — 4 steps, rollout + weight sync + policy update |

**Use `rad-vultr-mi355x-*` nodes** — `rad-burst` / `lux-mi355x-*` may show `register fat binary failed` (ROCm ISA mismatch with Primus torch).

---

## Working install (Approach A + vLLM)

Inside **`rocm/primus:v26.4`** with SkyRL mounted at `/workspace/SkyRL`:

```bash
# 1) Megatron-Bridge stack
bash integrations/primus_amd/approach_a_install.sh

# 2) vLLM — build once (or use cached wheel under .vllm_rocm_cache/wheels/)
bash integrations/primus_amd/build_vllm_rocm.sh
# Requires: VLLM_TARGET_DEVICE=rocm, PYTORCH_ROCM_ARCH=gfx950 (MI355X)

# 3) Full stack + GSM8K deps
bash integrations/primus_amd/approach_a_full_install.sh
```

### vLLM critical rules

1. **Never `pip install vllm`** from PyPI — pulls CUDA torch and breaks Transformer Engine.
2. **Build from vLLM `main`** against Primus ROCm torch (`build_vllm_rocm.sh`). Tag `v0.20.2` fails GPTQ compile on ROCm 7.14.
3. **Verify API** before GRPO: `python3 integrations/primus_amd/verify_vllm_skyrl_compat.py`.
4. Do not copy vLLM from `rocm/vllm-dev` — torch ABI mismatch.

### Megatron-Bridge pins

| Package | Rev |
|---------|-----|
| megatron-core | `71e418ea7d7b3a6c9a53238c543c3e0b43e11026` |
| megatron-bridge | `91a15142a4b4442a8d46ab539d1b923bd08570d0` |

Install bridge with `--no-deps` + `nvidia-modelopt`. Strip `Megatron-LM` from `PYTHONPATH`.

### SkyRL code patches (AMD)

- `skyrl/train/utils/utils.py` — skip `NVTE_FUSED_ATTN=0` on HIP; `ray.init(num_gpus=...)`; ROCm defaults `VLLM_USE_V1=0`
- Ray 2.57: `PlacementGroupSchedulingStrategy` from `ray.util.scheduling_strategies`
- `vllm_import_compat.py` — vLLM 0.20 OpenAI/serve API shims
- `vllm_engine.py` — HIP/ROCR visible devices (not `CUDA_VISIBLE_DEVICES`); pop CUDA on ROCm at worker init
- `layerwise_reload.py` — `skyrl_start_weight_update` / `skyrl_finish_weight_update` (avoid clash with vLLM 0.20 native Worker methods)
- GRPO smoke: `generator.inference_engine.async_engine=false` (sync `vllm.LLM` Ray actors)

---

## Run full GRPO cycle

```bash
# Build vLLM wheel (long; 1 GPU enough)
BUILD_ONLY=1 SLURM_PARTITION=rad-burst bash integrations/primus_amd/run_build_vllm_amd_cluster.sh

# Full GRPO (2 GPUs for colocated megatron+ref+vLLM; rad partition max 2h)
SLURM_GPUS=2 SLURM_PARTITION=rad SLURM_TIME=02:00:00 SLURM_MEM=128G \
  bash integrations/primus_amd/run_grpo_amd_cluster.sh
```

Log: `integrations/primus_amd/reports/grpo_amd_*.log`

**Env:** `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` must match `SLURM_GPUS`. Script unsets `CUDA_VISIBLE_DEVICES` in the container.

**Preflight:** `run_grpo_amd_megatron.sh` runs Ray GPU + vLLM API checks before training.

**Validated run (2026-09-01):** 4 training batches on `Qwen/Qwen2.5-0.5B-Instruct`, GSM8K 64-sample subset, `trainer/global_step` 1–4, `timing/sync_weights` ~3.7s/step, checkpoint at `global_step_4`.

---

## Errors hit during port

| Error | Fix |
|-------|-----|
| pip vllm upgrades torch → TE break | Build vLLM from source on Primus torch |
| `No module named vllm.entrypoints.openai.engine` | Rebuild vLLM 0.20 + `verify_vllm_skyrl_compat.py` |
| v0.20.2 tag GPTQ compile fail | Build `main` instead of tag |
| Ray placement group 0 GPUs | `ray.init(num_gpus=NUM_GPUS)` + preflight |
| HIP vs CUDA visible devices → SIGABRT | Set HIP/ROCR only in `setup_envvars_for_vllm` |
| `Device string must not be empty` | `VLLM_TARGET_DEVICE=rocm`, `amdsmi` |
| Async v1 `Engine core initialization failed` | `async_engine=false` + `VLLM_USE_V1=0` on ROCm |
| `finish_weight_update` Worker extension clash (vLLM 0.20) | Rename to `skyrl_finish_weight_update` in mixin + collective_rpc |
| lux-mi355x `register fat binary failed` | Use `rad-vultr-mi355x` nodes |
| `enforce_eager` TypeError | Set only via `generator.inference_engine.enforce_eager` |

---

## Related docs

| File | Contents |
|------|----------|
| **[WORKFLOW.md](WORKFLOW.md)** | Architecture: GRPO loop, Megatron stack, Primus role, vLLM changes |
| `README.md` | Quick start and file index |

## Quick vLLM smoke (no Ray)

```bash
bash integrations/primus_amd/vllm_llm_smoke.sh
```
