#!/bin/bash
#SBATCH -J aoi_mappo_power_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/mode-power-intervention-v1/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/mode-power-intervention-v1/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
SOURCE_ROOT="${AOI_SOURCE_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/mode-power-intervention-v1/P5_N4_gap25}"
OUTPUT_ROOT="$RESULT_ROOT/analysis"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/analysis_pycache"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/analysis_pycache"

echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current) started=$(date --iso-8601=seconds)"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -u -m analysis.summarize_mappo_power_intervention \
  --source-root "$SOURCE_ROOT" \
  --result-root "$RESULT_ROOT" \
  --output-root "$OUTPUT_ROOT"

"$PYTHON_BIN" - "$OUTPUT_ROOT/intervention_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["status"] == "PASS"
assert summary["completed_cells"] == 18
assert summary["baseline_common_arrays_exact"] is True
assert summary["per_heldout_seed_agent_rows"] == 540
assert summary["special_case_rows"] == 3
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds)"
