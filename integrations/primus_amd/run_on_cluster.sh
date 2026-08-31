#!/usr/bin/env bash
# Run on Slurm GPU node: bash integrations/primus_amd/run_on_cluster.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${REPORT_DIR:-$ROOT/integrations/primus_amd/reports}"
mkdir -p "$REPORT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="/workspace/SkyRL/integrations/primus_amd/reports/smoke_${STAMP}.json"
INSPECT_HOST="${REPORT_DIR}/inspect_${STAMP}.log"

PRIMUS_IMAGE="${PRIMUS_IMAGE:-primus:latest}"
SLURM_PARTITION="${SLURM_PARTITION:-rad}"
SLURM_GPUS="${SLURM_GPUS:-1}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_TIME="${SLURM_TIME:-00:45:00}"

submit_job() {
  local script="$1"
  srun -p "$SLURM_PARTITION" \
    --gres="gpu:${SLURM_GPUS}" \
    -c "$SLURM_CPUS" \
    --mem="$SLURM_MEM" \
    -t "$SLURM_TIME" \
    bash "$script"
}

WRAPPER="$ROOT/integrations/primus_amd/.cluster_job.sh"
cat > "$WRAPPER" <<EOF
set -euo pipefail
export ROCR_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES=\${HIP_VISIBLE_DEVICES:-0}

echo "=== SkyRL Primus cluster smoke ==="
echo "node: \$(hostname)"
echo "job: \${SLURM_JOB_ID:-n/a}"
echo "image: ${PRIMUS_IMAGE}"
echo "skyrl: ${ROOT}"
echo

REPORT="${REPORT}"
INSPECT_HOST="${REPORT_DIR}/inspect_${STAMP}.log"

docker run --rm \\
  --device=/dev/kfd \\
  --device=/dev/dri \\
  --group-add video \\
  --ipc=host \\
  -e HIP_VISIBLE_DEVICES=\${HIP_VISIBLE_DEVICES} \\
  -e ROCR_VISIBLE_DEVICES=\${ROCR_VISIBLE_DEVICES} \\
  -v "${ROOT}:/workspace/SkyRL" \\
  -w /workspace/SkyRL \\
  "${PRIMUS_IMAGE}" \\
  bash -lc 'mkdir -p /workspace/SkyRL/integrations/primus_amd/reports && bash integrations/primus_amd/container_smoke.sh --report "${REPORT}"' \\
  2>&1 | tee "${INSPECT_HOST}"

echo
echo "Report: \$REPORT"
echo "Log:    ${INSPECT_HOST}"
EOF
chmod +x "$WRAPPER"

echo "Submitting to partition=${SLURM_PARTITION} image=${PRIMUS_IMAGE}"
submit_job "$WRAPPER"
