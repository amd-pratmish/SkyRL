#!/usr/bin/env bash
# Run Phase-0/1 smoke tests inside skyrl-primus-amd on an AMD GPU node.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REPORT_DIR="${REPORT_DIR:-/tmp/skyrl-primus-exploration}"
mkdir -p "$REPORT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${REPORT_DIR}/smoke_${STAMP}.json"

echo "SkyRL Primus AMD smoke test"
echo "  repo:   $ROOT"
echo "  report: $REPORT"
echo

# Small model for bridge test; override with SMOKE_MODEL=...
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

python integrations/primus_amd/smoke_test.py --all --model "$SMOKE_MODEL" --report "$REPORT"
EXIT=$?

echo
if [[ $EXIT -eq 0 ]]; then
  echo "All checks passed. See $REPORT"
else
  echo "Some checks failed. Review $REPORT and update integrations/primus_amd/findings.md"
fi

exit $EXIT
