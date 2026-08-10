#!/bin/bash
#SBATCH -J aoi_matrix_array
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-47%8
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/Modified_MADDPG_with_TDec_results/heldout-formal-matrix/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/Modified_MADDPG_with_TDec_results/heldout-formal-matrix/slurm_logs/%x_%A_%a.err

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
STAGE="${AOI_STAGE:-train}"
CELL_COUNT=48

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "This script must be submitted as a Slurm job array." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= CELL_COUNT )); then
  echo "SLURM_ARRAY_TASK_ID must satisfy 0 <= id < $CELL_COUNT; got $SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi

case "$STAGE" in
  train)
    approval_marker="$HANDOFF_ROOT/PILOT_APPROVED"
    ;;
  eval)
    approval_marker="$HANDOFF_ROOT/TRAIN_APPROVED"
    ;;
  *)
    echo "AOI_STAGE must be 'train' or 'eval'; got '$STAGE'." >&2
    exit 2
    ;;
esac

if [[ ! -f "$approval_marker" ]]; then
  echo "Required reviewed gate is missing: $approval_marker" >&2
  exit 3
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export XDG_CACHE_HOME="$RESULT_ROOT/cache/xdg"
export MPLCONFIGDIR="$RESULT_ROOT/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/matrix_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

REPORT_ROOT="$HANDOFF_ROOT/matrix_array_reports"
RUN_LOG_DIR="$HANDOFF_ROOT/run_logs/$STAGE/array_${SLURM_ARRAY_JOB_ID}/task_${SLURM_ARRAY_TASK_ID}"
REPORT="$REPORT_ROOT/${STAGE}_array${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.json"
mkdir -p "$RUN_ROOT" "$REPORT_ROOT" "$RUN_LOG_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"

echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID of $CELL_COUNT"
echo "Stage: $STAGE"
echo "Host: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $PYTHON_BIN"
nvidia-smi

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
git rev-parse HEAD
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "visible_gpu_count": torch.cuda.device_count()})'

common_args=(
  --device cuda:0
  --output-root "$RUN_ROOT"
  --checkpoint-every 5
  --stage "$STAGE"
  --eval-purpose validation
  --eval-episodes 100
  --eval-seeds 201,202,203,204,205,206
  --shard-count "$CELL_COUNT"
  --shard-index "$SLURM_ARRAY_TASK_ID"
  --log-dir "$RUN_LOG_DIR"
  --report "$REPORT"
  --execute
)

stage_args=()
if [[ "$STAGE" == "train" ]]; then
  stage_args=(--recover-empty-run)
fi

"$PYTHON_BIN" -u scripts/matrix_runner.py \
  "${common_args[@]}" \
  "${stage_args[@]}"

test -z "$(git status --porcelain --untracked-files=all)"
echo "Array task $SLURM_ARRAY_TASK_ID completed at $(date --iso-8601=seconds). Report: $REPORT"
