#!/bin/bash
#SBATCH -J aoi_mappo_shared_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/shared-actor-v1/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/shared-actor-v1/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
INDEPENDENT_ROOT="${AOI_INDEPENDENT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/shared-actor-v1/P5_N4_gap25}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/shared-actor-v1/P5_N4_gap25" ]]; then
  echo "Unexpected result root: $RESULT_ROOT" >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT/analysis" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/analysis_pycache"
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
"$PYTHON_BIN" -u -m analysis.summarize_mappo_shared_actor \
  --independent-root "$INDEPENDENT_ROOT" \
  --result-root "$RESULT_ROOT"

"$PYTHON_BIN" - "$RESULT_ROOT/analysis/shared_actor_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["status"] == "PASS"
assert summary["shared_training_cells"] == 6
assert summary["evaluation_cells"] == 12
assert summary["training_per_seed_rows"] == 12
assert summary["training_per_episode_rows"] == 6000
assert summary["evaluation_per_seed_rows"] == 12
assert summary["eval_world_agent_rows"] == 360
assert summary["paired_training_seed_rows"] == 12
assert summary["paired_eval_world_agent_rows"] == 180
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds)"
