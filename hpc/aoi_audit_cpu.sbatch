#!/bin/bash
#SBATCH -J aoi_audit
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/Modified_MADDPG_with_TDec_results/heldout-formal-matrix/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/Modified_MADDPG_with_TDec_results/heldout-formal-matrix/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
EXPECTED_RESULT_ROOT="$DIAGNOSTICS_ROOT/Modified_MADDPG_with_TDec_results/heldout-formal-matrix"
: "${AOI_RESULT_ROOT:?Set AOI_RESULT_ROOT=$EXPECTED_RESULT_ROOT explicitly before submitting this deferred held-out job}"
RESULT_ROOT="$AOI_RESULT_ROOT"
if [[ "$RESULT_ROOT" != "$EXPECTED_RESULT_ROOT" ]]; then
  echo "AOI_RESULT_ROOT must equal $EXPECTED_RESULT_ROOT; got $RESULT_ROOT" >&2
  exit 2
fi
RUN_ROOT="$RESULT_ROOT/runs"
HANDOFF_ROOT="$RESULT_ROOT/handoff"
AUDIT_SCOPE="${AOI_AUDIT_SCOPE:-train}"

case "$AUDIT_SCOPE" in
  train|validation) ;;
  *)
    echo "AOI_AUDIT_SCOPE must be 'train' or 'validation'; got '$AUDIT_SCOPE'." >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TMPDIR="$RESULT_ROOT/tmp"
export XDG_CACHE_HOME="$RESULT_ROOT/cache/xdg"
export MPLCONFIGDIR="$RESULT_ROOT/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/audit_${SLURM_JOB_ID}"

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"

AUDIT_ROOT="$HANDOFF_ROOT/audits/${AUDIT_SCOPE}_job${SLURM_JOB_ID}"
mkdir -p "$AUDIT_ROOT" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"
mapfile -t run_dirs < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'reproduction_baseline_*_seed0[2-7]' -print | sort)
if [[ ${#run_dirs[@]} -ne 48 ]]; then
  echo "Expected exactly 48 formal run directories, found ${#run_dirs[@]}." >&2
  exit 3
fi

for run_dir in "${run_dirs[@]}"; do
  run_name="$(basename "$run_dir")"
  if [[ "$AUDIT_SCOPE" == "validation" ]]; then
    "$PYTHON_BIN" analysis/audit_results.py "$run_dir" --scope validation --require-eval \
      > "$AUDIT_ROOT/${run_name}.json"
  else
    "$PYTHON_BIN" analysis/audit_results.py "$run_dir" --scope train \
      > "$AUDIT_ROOT/${run_name}.json"
  fi
done

test -z "$(git status --porcelain --untracked-files=all)"
echo "All 48 $AUDIT_SCOPE audits passed. Reports: $AUDIT_ROOT"
