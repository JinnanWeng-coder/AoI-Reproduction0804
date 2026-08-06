#!/bin/bash
#SBATCH -J aoi_gap_gs42
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-23%8
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/AoI-Reproduction-diagnostics/gap-global-slow-42-v1/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/AoI-Reproduction-diagnostics/gap-global-slow-42-v1/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_BASE_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
DIAGNOSTIC_ROOT="${AOI_DIAGNOSTIC_ROOT:-$AOI_BASE_ROOT/AoI-Reproduction-diagnostics/gap-global-slow-42-v1}"
RUN_ROOT="$DIAGNOSTIC_ROOT/runs"
PHASE="${AOI_PHASE:-A}"

TAU="0.005"
TRAIN_NOISE="0.3"
PHASE_A_CELL_COUNT=24
PHASE_B_CELL_COUNT=18
TOTAL_UNIQUE_TRAINING_CELLS=42

seeds=(8 9 10 11 12 13)
phase_a_scenarios=(p05_n04_g05 p05_n04_g15 p05_n04_g25 p05_n04_g35)
phase_b_arm_labels=(sync_slow20 detached_slow01 detached_slow20)
phase_b_slow_intervals=(20 1 20)
phase_b_global_modes=(synchronous_joint detached_actor detached_actor)

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this file with sbatch as a Slurm job array." >&2
  exit 2
fi

seed_index=$((SLURM_ARRAY_TASK_ID % 6))
seed=${seeds[$seed_index]}

case "$PHASE" in
  A|a)
    task_count=$PHASE_A_CELL_COUNT
    scenario_index=$((SLURM_ARRAY_TASK_ID / 6))
    scenario=${phase_a_scenarios[$scenario_index]:-}
    arm_label="sync_slow01"
    slow_interval=1
    global_mode="synchronous_joint"
    ;;
  B|b)
    task_count=$PHASE_B_CELL_COUNT
    scenario="p05_n04_g25"
    arm_index=$((SLURM_ARRAY_TASK_ID / 6))
    arm_label=${phase_b_arm_labels[$arm_index]:-}
    slow_interval=${phase_b_slow_intervals[$arm_index]:-}
    global_mode=${phase_b_global_modes[$arm_index]:-}
    ;;
  *)
    echo "AOI_PHASE must be A or B; got '$PHASE'." >&2
    exit 2
    ;;
esac

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= task_count )); then
  echo "AOI_PHASE=$PHASE requires 0 <= array id < $task_count; got $SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi
if [[ -z "$scenario" || -z "$arm_label" || -z "$slow_interval" || -z "$global_mode" ]]; then
  echo "Internal array mapping failed for AOI_PHASE=$PHASE task=$SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi

printf -v seed_token "%02d" "$seed"
run_name="gap_global_slow_${arm_label}_${scenario}_seed${seed_token}"
run_dir="$RUN_ROOT/$run_name"
latest="$run_dir/checkpoints/latest.pt"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$DIAGNOSTIC_ROOT" == "$AOI_BASE_ROOT/AoI-Reproduction0804-results"* ]]; then
  echo "Refusing to use the formal result root for this diagnostic: $DIAGNOSTIC_ROOT" >&2
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

echo "phase=$PHASE task=$SLURM_ARRAY_TASK_ID/$task_count design_total=$TOTAL_UNIQUE_TRAINING_CELLS scenario=$scenario seed=$seed tau=$TAU slow=$slow_interval global=$global_mode train_noise=$TRAIN_NOISE"
echo "primary_metric_window=last100 secondary_metric_window=last50"
echo "diagnostic_root=$DIAGNOSTIC_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -c 'from config import resolve_config; c=resolve_config(profile="paper_faithful", scenario="p05_n04_g25"); assert c.exploration_noise == 0.3; print({"profile": "paper_faithful", "training_noise": c.exploration_noise})'
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
  --scenario "$scenario" \
  --seed "$seed" \
  --device cuda:0 \
  --run-name "$run_name" \
  --output-root "$RUN_ROOT" \
  --episodes 500 \
  --checkpoint-every 50 \
  --tau "$TAU" \
  --slow-update-every-episodes "$slow_interval" \
  --global-actor-mode "$global_mode" \
  --diagnostics \
  --scope train \
  "${recovery_args[@]}"

test -f "$run_dir/COMPLETE.json"
test -f "$run_dir/train_metrics.npz"
test -f "$run_dir/train_metrics_summary.json"
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) run=$run_name primary_metric_window=last100 secondary_metric_window=last50"
