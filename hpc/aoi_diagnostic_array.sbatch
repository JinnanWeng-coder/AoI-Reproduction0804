#!/bin/bash
#SBATCH -J aoi_diag_array
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-8%8
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/AoI-Reproduction-diagnostics/global-causal-v1/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/AoI-Reproduction-diagnostics/global-causal-v1/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_BASE_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
DIAGNOSTIC_ROOT="${AOI_DIAGNOSTIC_ROOT:-$AOI_BASE_ROOT/AoI-Reproduction-diagnostics/global-causal-v1}"
RUN_ROOT="$DIAGNOSTIC_ROOT/runs"
STAGE="${AOI_STAGE:-train}"
SCENARIO="p05_n04_g25"
VALIDATION_SEEDS="201,202,203,204,205,206"
CHECKPOINT_NAME="best.pt"

arms=(current global_detached legacy_release)
seeds=(3 5 7)
noises=(0 0.05 0.1 0.3)
noise_tokens=(0p0 0p05 0p1 0p3)

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this file with sbatch as a Slurm job array." >&2
  exit 2
fi

case "$STAGE" in
  train)
    task_count=9
    train_index=$SLURM_ARRAY_TASK_ID
    noise=""
    ;;
  eval)
    task_count=36
    train_index=$((SLURM_ARRAY_TASK_ID / 4))
    noise_index=$((SLURM_ARRAY_TASK_ID % 4))
    noise=${noises[$noise_index]}
    noise_token=${noise_tokens[$noise_index]}
    ;;
  *)
    echo "AOI_STAGE must be 'train' or 'eval'; got '$STAGE'." >&2
    exit 2
    ;;
esac

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= task_count )); then
  echo "AOI_STAGE=$STAGE requires 0 <= array id < $task_count; got $SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi

arm=${arms[$((train_index / 3))]}
seed=${seeds[$((train_index % 3))]}
printf -v seed_token "%02d" "$seed"
run_name="diagnostic_${arm}_${SCENARIO}_seed${seed_token}"
run_dir="$RUN_ROOT/$run_name"
latest="$run_dir/checkpoints/latest.pt"
checkpoint="$run_dir/checkpoints/$CHECKPOINT_NAME"

case "$arm" in
  current)
    profile="paper_faithful"
    arm_args=(--global-actor-mode synchronous_joint)
    ;;
  global_detached)
    profile="paper_faithful"
    arm_args=(--global-actor-mode detached_actor)
    ;;
  legacy_release)
    profile="legacy_release"
    arm_args=()
    ;;
  *)
    echo "Unknown diagnostic arm: $arm" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$DIAGNOSTIC_ROOT" == "$AOI_BASE_ROOT/AoI-Reproduction0804-results"* ]]; then
  echo "Refusing to use the formal 48-cell result root for diagnostics: $DIAGNOSTIC_ROOT" >&2
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

echo "stage=$STAGE task=$SLURM_ARRAY_TASK_ID/$task_count arm=$arm seed=$seed noise=${noise:-n/a}"
echo "diagnostic_root=$DIAGNOSTIC_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'
nvidia-smi

common_args=(
  --profile "$profile"
  --scenario "$SCENARIO"
  --seed "$seed"
  --device cuda:0
  --run-name "$run_name"
  --output-root "$RUN_ROOT"
  --checkpoint-every 50
  --diagnostics
  "${arm_args[@]}"
)

if [[ "$STAGE" == "train" ]]; then
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
    "${common_args[@]}" \
    --scope train \
    "${recovery_args[@]}"
else
  if [[ ! -f "$run_dir/COMPLETE.json" || ! -f "$checkpoint" ]]; then
    echo "Training cell or selected checkpoint is incomplete: $run_name ($checkpoint)" >&2
    exit 4
  fi

  shopt -s nullglob
  completed_evals=("$run_dir"/eval/*_noise"$noise_token"/EVAL_COMPLETE.json)
  shopt -u nullglob
  if (( ${#completed_evals[@]} > 0 )); then
    echo "SKIP completed validation: $run_name noise=$noise"
    exit 0
  fi

  "$PYTHON_BIN" -u Main.py \
    "${common_args[@]}" \
    --scope validation \
    --eval-only \
    --eval-purpose validation \
    --eval-episodes 100 \
    --eval-seeds "$VALIDATION_SEEDS" \
    --eval-noise "$noise" \
    --resume "$checkpoint"
fi

test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) run=$run_name stage=$STAGE noise=${noise:-n/a}"
