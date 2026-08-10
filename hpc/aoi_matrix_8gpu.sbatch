#!/bin/bash
#SBATCH -J aoi_matrix_8gpu
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l20:8
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/AoI-Reproduction0804-results/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/AoI-Reproduction0804-results/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_BASE_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
RESULT_ROOT="${AOI_RESULT_ROOT:-$AOI_BASE_ROOT/AoI-Reproduction0804-results}"
RUN_ROOT="$RESULT_ROOT/runs"
HANDOFF_ROOT="$RESULT_ROOT/handoff"
STAGE="${AOI_STAGE:-train}"
SHARD_COUNT=8

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
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TMPDIR="$RESULT_ROOT/tmp"
export XDG_CACHE_HOME="$RESULT_ROOT/cache/xdg"
export MPLCONFIGDIR="$RESULT_ROOT/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/matrix_${SLURM_JOB_ID}"

STEP_ROOT="$HANDOFF_ROOT/slurm_steps/${STAGE}_job${SLURM_JOB_ID}"
REPORT_ROOT="$HANDOFF_ROOT/matrix_reports"
RUN_LOG_ROOT="$HANDOFF_ROOT/run_logs/$STAGE"
mkdir -p "$RUN_ROOT" "$STEP_ROOT" "$REPORT_ROOT" "$RUN_LOG_ROOT" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"

echo "Host: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "Stage: $STAGE"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
echo "Python: $PYTHON_BIN"
test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8; print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "visible_gpu_count": torch.cuda.device_count()})'

common_args=(
  --device cuda:0
  --output-root "$RUN_ROOT"
  --checkpoint-every 5
  --stage "$STAGE"
  --eval-purpose validation
  --eval-episodes 100
  --eval-seeds 201,202,203,204,205,206
  --shard-count "$SHARD_COUNT"
  --execute
)

stage_args=()
if [[ "$STAGE" == "train" ]]; then
  stage_args=(--recover-empty-run)
fi

declare -a worker_pids=()
for shard_index in $(seq 0 7); do
  shard_log_dir="$RUN_LOG_ROOT/shard_${shard_index}"
  mkdir -p "$shard_log_dir"
  report="$REPORT_ROOT/${STAGE}_job${SLURM_JOB_ID}_shard${shard_index}.json"
  echo "Launching shard $shard_index on one Slurm-managed L20 GPU"
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:l20:1 \
    --output="$STEP_ROOT/shard${shard_index}.out" \
    --error="$STEP_ROOT/shard${shard_index}.err" \
    "$PYTHON_BIN" -u scripts/matrix_runner.py \
      "${common_args[@]}" \
      "${stage_args[@]}" \
      --shard-index "$shard_index" \
      --log-dir "$shard_log_dir" \
      --report "$report" &
  worker_pids[$shard_index]=$!
done

worker_failure=0
for shard_index in $(seq 0 7); do
  if ! wait "${worker_pids[$shard_index]}"; then
    echo "Shard $shard_index failed; inspect $STEP_ROOT/shard${shard_index}.err" >&2
    worker_failure=1
  fi
done

if [[ $worker_failure -ne 0 ]]; then
  exit 1
fi

test -z "$(git status --porcelain --untracked-files=all)"
echo "All eight $STAGE shards completed at $(date --iso-8601=seconds)."
