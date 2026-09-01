# Primus + SkyRL AMD port findings

## Status (2026-08-31)

| Layer | Status | Notes |
|-------|--------|-------|
| Megatron-Bridge + mcore 0.19 on `rocm/primus:v26.4` | **PASS** | `approach_a_validate.sh` |
| vLLM ROCm on same torch (2.12+rocm7.14) | **PASS (build)** | Build from vLLM `main` → wheel `0.9.2+rocm.rocm714`, 14 native `.so` |
| SkyRL + Ray 2.57 import | **PASS** | Ray `PlacementGroupSchedulingStrategy` compat patch |
| Full install (`approach_a_full_install.sh`) | **PASS** | Bridge + cached vLLM wheel + deps |
| GRPO `main_base` launch | **PARTIAL** | Reaches Ray entrypoint + dataset prep; full train step not completed |

**Use `rad-vultr-mi355x-*` nodes** — `rad-burst` / `lux-mi355x-*` showed mass `register fat binary failed` (ROCm ISA mismatch with Primus torch build).

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

1. **Never `pip install vllm`** from PyPI — pulls CUDA torch 2.13 and breaks Transformer Engine.
2. **Build from vLLM `main`** against Primus ROCm torch (`build_vllm_rocm.sh`).
3. Cached wheel: `integrations/primus_amd/.vllm_rocm_cache/wheels/vllm-0.9.2+rocm.rocm714-*.whl`
4. Do not copy vLLM from `rocm/vllm-dev` — torch ABI mismatch (`vllm._C` undefined symbols).

### Megatron-Bridge pins

| Package | Rev |
|---------|-----|
| megatron-core | `71e418ea7d7b3a6c9a53238c543c3e0b43e11026` |
| megatron-bridge | `91a15142a4b4442a8d46ab539d1b923bd08570d0` |

Install bridge with `--no-deps` + `nvidia-modelopt`. Strip `Megatron-LM` from `PYTHONPATH`.

### SkyRL code patches (AMD)

- `skyrl/train/utils/utils.py` — skip `NVTE_FUSED_ATTN=0` on HIP
- Ray 2.57: import `PlacementGroupSchedulingStrategy` from `ray.util.scheduling_strategies` (4 files)

---

## Run full GRPO cycle

```bash
# Prefer rad partition (vultr MI355X). 2 GPUs recommended for colocated megatron+ref+vLLM.
SLURM_GPUS=2 SLURM_PARTITION=rad bash integrations/primus_amd/run_grpo_amd_cluster.sh
```

Log: `integrations/primus_amd/reports/grpo_amd_*.log`

**Env:** `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` must match `SLURM_GPUS` (script sets `0` or `0,1`).

---

## Errors hit during port

| Error | Fix |
|-------|-----|
| pip vllm upgrades torch → TE break | Build vLLM from source; pin torch |
| Copied vllm-dev → `_C` ABI mismatch | Build on Primus torch |
| `PlacementGroupSchedulingStrategy` import | Ray 2.57 scheduling_strategies path |
| HIP vs ROCR device count mismatch | Set both from `SLURM_GPUS` in docker `-e` |
| lux-mi355x `register fat binary failed` | Use `rad-vultr-mi355x` nodes |

---

## Next: complete one GRPO step on rad-vultr

When `rad` partition has free MI355X nodes:

```bash
SLURM_GPUS=2 SLURM_PARTITION=rad SLURM_MEM=128G \
  bash integrations/primus_amd/run_grpo_amd_cluster.sh
```

If OOM on 1 GPU colocate, keep `NUM_GPUS=2` or lower `gpu_memory_utilization` in `run_grpo_amd_megatron.sh`.
