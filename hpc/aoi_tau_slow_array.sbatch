#!/bin/bash
#SBATCH -J aoi_tau_slow
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-11%8
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/AoI-Reproduction-diagnostics/tau-slow-v1/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/AoI-Reproduction-diagnostics/tau-slow-v1/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_BASE_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
DIAGNOSTIC_ROOT="${AOI_DIAGNOSTIC_ROOT:-$AOI_BASE_ROOT/AoI-Reproduction-diagnostics/tau-slow-v1}"
RUN_ROOT="$DIAGNOSTIC_ROOT/runs"
STAGE="${AOI_STAGE:-train}"
SCENARIO="p05_n04_g25"
VALIDATION_SEEDS="201,202,203,204,205,206"

taus=(0.0005 0.005)
tau_tokens=(0p0005 0p005)
slow_intervals=(1 20)
seeds=(3 5 7)
checkpoint_names=(best.pt latest.pt)
checkpoint_labels=(best episode500)
noises=(0 0.3)
noise_tokens=(0p0 0p3)

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this file with sbatch as a Slurm job array." >&2
  exit 2
fi

case "$STAGE" in
  train)
    task_count=12
    train_index=$SLURM_ARRAY_TASK_ID
    checkpoint_name=""
    checkpoint_label=""
    noise=""
    ;;
  eval)
    task_count=48
    train_index=$((SLURM_ARRAY_TASK_ID / 4))
    eval_index=$((SLURM_ARRAY_TASK_ID % 4))
    checkpoint_index=$((eval_index / 2))
    noise_index=$((eval_index % 2))
    checkpoint_name=${checkpoint_names[$checkpoint_index]}
    checkpoint_label=${checkpoint_labels[$checkpoint_index]}
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

factor_index=$((train_index / 3))
tau_index=$((factor_index / 2))
slow_index=$((factor_index % 2))
seed_index=$((train_index % 3))
tau=${taus[$tau_index]}
tau_token=${tau_tokens[$tau_index]}
slow_interval=${slow_intervals[$slow_index]}
seed=${seeds[$seed_index]}
printf -v seed_token "%02d" "$seed"
printf -v slow_token "%02d" "$slow_interval"

run_name="tau_slow_tau${tau_token}_slow${slow_token}_${SCENARIO}_seed${seed_token}"
run_dir="$RUN_ROOT/$run_name"
latest="$run_dir/checkpoints/latest.pt"

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

echo "stage=$STAGE task=$SLURM_ARRAY_TASK_ID/$task_count tau=$tau slow_interval=$slow_interval seed=$seed checkpoint=${checkpoint_label:-n/a} noise=${noise:-n/a}"
echo "diagnostic_root=$DIAGNOSTIC_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'
nvidia-smi

common_args=(
  --profile paper_faithful
  --scenario "$SCENARIO"
  --seed "$seed"
  --device cuda:0
  --run-name "$run_name"
  --output-root "$RUN_ROOT"
  --checkpoint-every 50
  --tau "$tau"
  --slow-update-every-episodes "$slow_interval"
  --global-actor-mode synchronous_joint
  --diagnostics
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
  checkpoint="$run_dir/checkpoints/$checkpoint_name"
  if [[ ! -f "$run_dir/COMPLETE.json" || ! -f "$checkpoint" ]]; then
    echo "Training cell or checkpoint is incomplete: $run_name ($checkpoint)" >&2
    exit 4
  fi

  if "$PYTHON_BIN" -c 'import json, pathlib, sys; root=pathlib.Path(sys.argv[1]); name=sys.argv[2]; noise=float(sys.argv[3]); wanted=[201,202,203,204,205,206]; records=[]; [(records.append(json.loads(p.read_text()))) for p in root.glob("eval/*/EVAL_COMPLETE.json") if p.is_file()]; raise SystemExit(0 if any(r.get("checkpoint_name")==name and float(r.get("eval_noise",-1))==noise and r.get("eval_purpose")=="validation" and r.get("eval_seeds")==wanted and int(r.get("eval_episodes",-1))==100 for r in records) else 1)' "$run_dir" "$checkpoint_name" "$noise"; then
    echo "SKIP completed validation: $run_name checkpoint=$checkpoint_label noise=$noise"
    exit 0
  fi

  "$PYTHON_BIN" -u Main.py \
    "${common_args[@]}" \
    --scope validation \
    --eval-only \
    --eval-purpose validation \
    --diagnostic-eval \
    --eval-episodes 100 \
    --eval-seeds "$VALIDATION_SEEDS" \
    --eval-noise "$noise" \
    --resume "$checkpoint"
fi

test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) run=$run_name stage=$STAGE checkpoint=${checkpoint_label:-n/a} noise=${noise:-n/a}"
