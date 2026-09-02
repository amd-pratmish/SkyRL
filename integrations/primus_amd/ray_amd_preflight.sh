#!/usr/bin/env bash
# Run inside Primus container on a GPU node before GRPO.
set -euo pipefail
: "${NUM_GPUS:=2}"
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
python3 integrations/primus_amd/ray_amd_preflight.py
