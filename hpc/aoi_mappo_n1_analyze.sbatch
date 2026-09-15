#!/bin/bash
#SBATCH -J aoi_mappo_n1_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/N1-training-world-revisit/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/N1-training-world-revisit/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
STUDY_ROOT="${AOI_STUDY_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics/actor-sharing-study}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$STUDY_ROOT/N1-training-world-revisit/P5_N4_gap25}"
PYTHON_BIN="${PYTHON_BIN:-$AOI_ENV_DIR/bin/python}"
EXPECTED_RESULT_ROOT="$STUDY_ROOT/N1-training-world-revisit/P5_N4_gap25"

[[ -x "$PYTHON_BIN" ]] || { echo "Python missing: $PYTHON_BIN" >&2; exit 2; }
[[ "$RESULT_ROOT" == "$EXPECTED_RESULT_ROOT" ]] || { echo "Unexpected N1 result root" >&2; exit 2; }

job_token="${SLURM_JOB_ID:-manual}"
mkdir -p "$RESULT_ROOT/analysis" "$RESULT_ROOT/figures" "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/analysis_pycache/$job_token"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/analysis_pycache/$job_token"

echo "N1 analysis commit=$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -u -m analysis.summarize_mappo_n1_revisit \
  --study-root "$STUDY_ROOT" \
  --n1-root "$RESULT_ROOT"

"$PYTHON_BIN" - "$RESULT_ROOT/analysis/n1_summary.json" <<'PY'
import json, sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "new_training_cells": 0,
    "new_revisit_evaluation_cells": 24,
    "reused_development_evaluation_cells": 24,
    "total_policy_evaluation_cells": 48,
    "condition_count": 4,
    "training_seed_count_per_condition": 6,
    "revisit_world_agent_rows": 120,
    "development_world_agent_rows": 720,
    "combined_world_agent_rows": 840,
    "paired_condition_seed_rows": 24,
}
assert summary["status"] == "PASS" and summary["experiment"] == "N1"
assert summary["counts"] == expected
PY
test -z "$(git status --porcelain --untracked-files=all)"
