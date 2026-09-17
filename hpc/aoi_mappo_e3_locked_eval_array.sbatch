#!/bin/bash
#SBATCH -J aoi_mappo_e3_eval
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --time=03:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E3-locked-test/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E3-locked-test/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail
BASE="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
PROJECT_ROOT="${AOI_PROJECT_ROOT:-$BASE/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT/AoI-Reproduction0804}"
STUDY_ROOT="${AOI_STUDY_ROOT:-$PROJECT_ROOT/AoI-Reproduction-diagnostics/actor-sharing-study}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$STUDY_ROOT/E3-locked-test/P5_N4_gap25}"
PYTHON_BIN="${PYTHON_BIN:-$BASE/conda_envs/aoi_cuda/bin/python}"
EXPECTED_COMMIT="${AOI_EXPECTED_COMMIT:?Set AOI_EXPECTED_COMMIT to locked E3 commit}"
[[ "$RESULT_ROOT" == "$STUDY_ROOT/E3-locked-test/P5_N4_gap25" ]] || { echo "Unexpected E3 root" >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Submit with --array" >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 24 )) || { echo "Cell id outside 0..23" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python missing" >&2; exit 2; }
job_token="${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RESULT_ROOT/evaluations" "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/pycache/$job_token"
cd "$PROJECT_DIR"
[[ "$(git branch --show-current)" == research/mappo ]] || { echo "Wrong branch" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "Wrong commit" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Dirty checkout" >&2; exit 2; }
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/$job_token"
"$PYTHON_BIN" -m analysis.mappo_e3_contract check-lock --study-root "$STUDY_ROOT"
read -r policy eval_dir < <("$PYTHON_BIN" - "$STUDY_ROOT" "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys
from pathlib import Path
from analysis.mappo_e3_contract import cell, eval_dir, run_dir
root, item = Path(sys.argv[1]), cell(int(sys.argv[2]))
print(run_dir(root, item) / "policy_final.pt", eval_dir(root, item))
PY
)
[[ -f "$policy" ]] || { echo "Missing source policy: $policy" >&2; exit 2; }
if [[ -f "$eval_dir/E3_COMPLETE.json" ]]; then
  "$PYTHON_BIN" -m analysis.mappo_e3_contract check-cell --study-root "$STUDY_ROOT" --cell-id "$SLURM_ARRAY_TASK_ID"
  echo "SKIP validated E3 cell $SLURM_ARRAY_TASK_ID"
  exit 0
fi
[[ ! -e "$eval_dir" ]] || { echo "Incomplete/conflicting cell; preserve and report: $eval_dir" >&2; exit 3; }
worlds="$("$PYTHON_BIN" -c 'from analysis.mappo_e3_contract import protocol; print(",".join(map(str, protocol()["candidate_worlds"])))')"
"$PYTHON_BIN" -u -m analysis.evaluate_mappo_feasibility_reward \
  --policy "$policy" --output-root "$RESULT_ROOT/evaluations" --arm baseline \
  --device cuda:0 --eval-seeds "$worlds" --eval-episodes 100
"$PYTHON_BIN" -m analysis.mappo_e3_contract mark-cell --study-root "$STUDY_ROOT" --cell-id "$SLURM_ARRAY_TASK_ID"
[[ -f "$eval_dir/E3_COMPLETE.json" ]] || { echo "Missing E3 completion marker" >&2; exit 4; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Checkout changed" >&2; exit 5; }
