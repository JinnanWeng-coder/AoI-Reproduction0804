#!/bin/bash
#SBATCH -J aoi_audit
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -o /eeedata/sgxyl2/AoI-Reproduction0804-results/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxyl2/AoI-Reproduction0804-results/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/eeedata/sgxyl2/AoI-Reproduction0804}"
CONDA_ROOT="${CONDA_ROOT:-/share/home/sgxyl2/miniconda3}"
AOI_ENV_NAME="${AOI_ENV_NAME:-aoi_cuda}"
RESULT_ROOT="${AOI_RESULT_ROOT:-/eeedata/sgxyl2/AoI-Reproduction0804-results}"
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

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$AOI_ENV_NAME"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"

AUDIT_ROOT="$HANDOFF_ROOT/audits/${AUDIT_SCOPE}_job${SLURM_JOB_ID}"
mkdir -p "$AUDIT_ROOT"
mapfile -t run_dirs < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'paper_faithful_*_seed0[2-7]' -print | sort)
if [[ ${#run_dirs[@]} -ne 48 ]]; then
  echo "Expected exactly 48 formal run directories, found ${#run_dirs[@]}." >&2
  exit 3
fi

for run_dir in "${run_dirs[@]}"; do
  run_name="$(basename "$run_dir")"
  if [[ "$AUDIT_SCOPE" == "validation" ]]; then
    python analysis/audit_results.py "$run_dir" --scope validation --require-eval \
      > "$AUDIT_ROOT/${run_name}.json"
  else
    python analysis/audit_results.py "$run_dir" --scope train \
      > "$AUDIT_ROOT/${run_name}.json"
  fi
done

test -z "$(git status --porcelain --untracked-files=all)"
echo "All 48 $AUDIT_SCOPE audits passed. Reports: $AUDIT_ROOT"
