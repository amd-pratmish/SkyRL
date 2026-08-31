# Primus + SkyRL exploration findings

## Run metadata

| Field | Value |
|-------|-------|
| Date | 2026-08-28 |
| Host / cluster | RAD Vultr Slurm (`rad-vultr-mi355x-*`) |
| GPU model | AMD Instinct **MI355X** |
| Image | **`rocm/primus:v26.4`** (Python 3.12) |
| SkyRL commit | NovaSky-AI/SkyRL clone under `/home/pratmish/SkyRL` |

---

## ✅ Approach A: WORKING (2026-08-28)

SkyRL's existing **`trainer.strategy=megatron`** path can run on AMD via Megatron-Bridge with a custom install recipe — no new `primus` strategy required for the training backend itself.

### Validated on MI355X

| Step | Result |
|------|--------|
| megatron-core **0.19** (SkyRL git pin) | PASS |
| megatron-bridge **0.6** (SkyRL git pin, `--no-deps`) | PASS |
| nvidia-modelopt on ROCm | PASS |
| `AutoBridge.from_hf_pretrained(Qwen/Qwen2.5-0.5B-Instruct)` | PASS |
| `skyrl...megatron_worker` import | PASS |
| `provide_distributed_model` on GPU (494M params) | PASS |

Log: `integrations/primus_amd/reports/approach_a_validate2.log`

### Install recipe (inside `rocm/primus:v26.4`)

```bash
bash integrations/primus_amd/approach_a_install.sh
# or full validation:
bash integrations/primus_amd/run_approach_a_validate.sh
```

**Critical details:**

1. **Override Primus bundled Megatron-LM 0.16** — install megatron-core 0.19 from SkyRL's git pin; strip `Megatron-LM` from `PYTHONPATH`.
2. **Install megatron-bridge with `--no-deps`** — avoids CUDA-only wheels (`mamba-ssm`, `flashinfer`).
3. **Add `nvidia-modelopt`** — required by bridge `auto_bridge.py`.
4. **Do not set `NVTE_FUSED_ATTN=0` on ROCm** — breaks model materialize; patched in `skyrl/train/utils/utils.py` for HIP.
5. Use **`provider.attention_backend = "flash"`** (ROCm flash-attn) when materializing via bridge.

### Pins (match SkyRL `uv.lock`)

| Package | Rev |
|---------|-----|
| megatron-core | `71e418ea7d7b3a6c9a53238c543c3e0b43e11026` |
| megatron-bridge | `91a15142a4b4442a8d46ab539d1b923bd08570d0` |

### Still open for full GRPO

| Item | Status |
|------|--------|
| vLLM ROCm rollouts | Use `rocm/vllm-dev` or separate inference container |
| Weight sync policy→vLLM | Not tested; use `weight_sync_backend=nccl`, not CUDA IPC |
| Docker image build | `docker/Dockerfile.primus-amd` updated with Approach A install |
| End-to-end GSM8K GRPO | Phase 2 — next |

---

## Earlier exploration (superseded)

<details>
<summary>Legacy `primus:latest` (py3.10) results</summary>

- 9/10 smoke checks; megatron-bridge missing; SkyRL cannot install (py3.10).
- Megatron forward/backward via raw Megatron-Core: PASS.

</details>

## Decision

- [x] **Approach A works** — use existing `megatron_worker` + install recipe on `rocm/primus:v26.4`
- [ ] Approach B (`trainer.strategy=primus`) — defer unless Approach A hits RL-loop blockers

## Next action

1. Build `skyrl-primus-amd` image on cluster (`bash integrations/primus_amd/build_image.sh`)
2. Run minimal GRPO smoke with `trainer.strategy=megatron`, `weight_sync_backend=nccl`, vLLM on `rocm/vllm-dev`
