#!/bin/bash
#SBATCH -J aoi_mappo_service_power_audit
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/service-power-audit-v1/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/service-power-audit-v1/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
SOURCE_ROOT="${AOI_SOURCE_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/service-power-audit-v1/P5_N4_gap25}"
OUTPUT_ROOT="$RESULT_ROOT/analysis"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/service-power-audit-v1/P5_N4_gap25" ]]; then
  echo "Refusing to write outside the dedicated audit root: $RESULT_ROOT" >&2
  exit 2
fi
if [[ "$SOURCE_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25" ]]; then
  echo "Refusing to audit a source other than tdec-ab-v1: $SOURCE_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/pycache"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache"

echo "source_root=$SOURCE_ROOT"
echo "result_root=$RESULT_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current)"
test -z "$(git status --porcelain --untracked-files=all)"

"$PYTHON_BIN" -u analysis/audit_mappo_service_power.py \
  --source-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT"

"$PYTHON_BIN" - "$OUTPUT_ROOT/audit_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["status"] == "PASS"
assert summary["training_runs"] == 12
assert summary["evaluation_runs"] == 24
assert summary["episode_agent_rows"] == 102000
PY
test -f "$OUTPUT_ROOT/inventory.csv"
test -f "$OUTPUT_ROOT/cohort_seed_summary.csv"
test -f "$OUTPUT_ROOT/intervals.csv"
test -f "$OUTPUT_ROOT/service_power_audit.md"
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds)"
