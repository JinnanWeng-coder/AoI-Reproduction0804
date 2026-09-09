#!/bin/bash
#SBATCH -J aoi_mappo_feas_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/feasibility-reward-audit-v1/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/feasibility-reward-audit-v1/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
SOURCE_ROOT="${AOI_SOURCE_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25}"
OLD_RESULT_ROOT="${AOI_OLD_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/mode-power-intervention-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/feasibility-reward-audit-v1/P5_N4_gap25}"
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
"$PYTHON_BIN" -u -m analysis.summarize_mappo_feasibility_reward \
  --source-root "$SOURCE_ROOT" \
  --old-result-root "$OLD_RESULT_ROOT" \
  --result-root "$RESULT_ROOT" \
  --output-root "$OUTPUT_ROOT"

"$PYTHON_BIN" - "$OUTPUT_ROOT/feasibility_reward_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["status"] == "PASS"
assert summary["completed_cells"] == 30
assert summary["training_seed_rows"] == 30
assert summary["eval_seed_agent_rows"] == 900
assert summary["per_episode_rows"] == 18000
assert summary["service_opportunity_rows"] == 2700
assert summary["paired_eval_seed_agent_rows"] == 720
assert summary["special_case_rows"] == 5
assert summary["equivalence_rows"] == 12
assert summary["physical_reconstruction_rows"] == 30
assert summary["baseline_plus3_common_arrays_exact"] is True
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds)"
