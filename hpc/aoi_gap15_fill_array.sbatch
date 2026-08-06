#!/bin/bash
#SBATCH -J aoi_gap15_fill
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-5%6
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/AoI-Reproduction-diagnostics/gap15-fill-v1/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/AoI-Reproduction-diagnostics/gap15-fill-v1/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_BASE_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
DIAGNOSTIC_ROOT="${AOI_DIAGNOSTIC_ROOT:-$AOI_BASE_ROOT/AoI-Reproduction-diagnostics/gap15-fill-v1}"
RUN_ROOT="$DIAGNOSTIC_ROOT/runs"
SCENARIO="p05_n04_g15"
TAU="0.005"
SLOW_INTERVAL="1"
TRAIN_NOISE="0.3"

seeds=(2 3 4 5 6 7)

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this file with sbatch as a Slurm job array." >&2
  exit 2
fi

task_count=${#seeds[@]}
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= task_count )); then
  echo "Training requires 0 <= array id < $task_count; got $SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi

seed=${seeds[$SLURM_ARRAY_TASK_ID]}
printf -v seed_token "%02d" "$seed"
run_name="gap15_fill_${SCENARIO}_seed${seed_token}"
run_dir="$RUN_ROOT/$run_name"
latest="$run_dir/checkpoints/latest.pt"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$DIAGNOSTIC_ROOT" == "$AOI_BASE_ROOT/AoI-Reproduction0804-results"* ]]; then
  echo "Refusing to use the formal result root for this bounded experiment: $DIAGNOSTIC_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$DIAGNOSTIC_ROOT/tmp" \
  "$DIAGNOSTIC_ROOT/cache/xdg" "$DIAGNOSTIC_ROOT/cache/matplotlib" \
  "$DIAGNOSTIC_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$DIAGNOSTIC_ROOT/tmp"
export XDG_CACHE_HOME="$DIAGNOSTIC_ROOT/cache/xdg"
export MPLCONFIGDIR="$DIAGNOSTIC_ROOT/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$DIAGNOSTIC_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "task=$SLURM_ARRAY_TASK_ID/$task_count scenario=$SCENARIO seed=$seed tau=$TAU slow=$SLOW_INTERVAL train_noise=$TRAIN_NOISE primary_metric_window=last100"
echo "diagnostic_root=$DIAGNOSTIC_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'
nvidia-smi

if [[ -f "$run_dir/COMPLETE.json" ]]; then
  echo "SKIP completed training cell: $run_name"
  exit 0
fi

recovery_args=()
if [[ -f "$latest" ]]; then
  recovery_args=(--resume "$latest")
elif [[ -d "$run_dir" ]]; then
  recovery_args=(--recover-empty-run)
fi

"$PYTHON_BIN" -u Main.py \
  --profile paper_faithful \
  --scenario "$SCENARIO" \
  --seed "$seed" \
  --device cuda:0 \
  --run-name "$run_name" \
  --output-root "$RUN_ROOT" \
  --episodes 500 \
  --checkpoint-every 50 \
  --tau "$TAU" \
  --slow-update-every-episodes "$SLOW_INTERVAL" \
  --global-actor-mode synchronous_joint \
  --diagnostics \
  --scope train \
  "${recovery_args[@]}"

test -f "$run_dir/COMPLETE.json"
test -f "$run_dir/train_metrics.npz"
test -f "$run_dir/train_metrics_summary.json"
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) run=$run_name primary_metric_window=last100"
