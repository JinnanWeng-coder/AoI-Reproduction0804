#!/bin/bash
#SBATCH -J aoi_mappo_w1_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/W1-service-intervals/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/W1-service-intervals/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail
AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
STUDY_ROOT="${AOI_STUDY_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics/actor-sharing-study}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$STUDY_ROOT/W1-service-intervals/P5_N4_gap25}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="${PYTHON_BIN:-$AOI_ENV_DIR/bin/python}"
EXPECTED_COMMIT="${AOI_EXPECTED_COMMIT:?Set AOI_EXPECTED_COMMIT to the pushed W1 commit}"

[[ -d "$PROJECT_DIR/.git" ]] || { echo "Missing project checkout: $PROJECT_DIR" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ "$RESULT_ROOT" == "$STUDY_ROOT/W1-service-intervals/P5_N4_gap25" ]] || {
    echo "Unexpected W1 output root: $RESULT_ROOT" >&2; exit 2;
}
mkdir -p "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/analysis_pycache/${SLURM_JOB_ID:-manual}"
cd "$PROJECT_DIR"
[[ "$(git branch --show-current)" == research/mappo ]] || { echo "Wrong branch" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "Wrong commit" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Dirty checkout" >&2; exit 2; }
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/analysis_pycache/${SLURM_JOB_ID:-manual}"
echo "W1 CPU post-processing commit=$EXPECTED_COMMIT output=$RESULT_ROOT"
"$PYTHON_BIN" -u -m analysis.audit_mappo_w1_service_intervals --study-root "$STUDY_ROOT"
"$PYTHON_BIN" - "$RESULT_ROOT/verification.json" <<'PY'
import json
import sys
from pathlib import Path

v = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert v["status"] == "PASS" and v["scope"] == "technical_source_and_calculation_only"
assert (v["evaluation_cells"], v["policy_world_runs"], v["flow_rows"],
        v["agent_slots"], v["condition_seed_rows"]) == (24, 144, 720, 7200000, 24)
assert v["historical_reconciliation"]["status"] == "PASS"
PY
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Checkout changed" >&2; exit 2; }
