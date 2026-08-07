#!/bin/bash
#SBATCH -J aoi_mm_size
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-17%6
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/Modified_MADDPG_results/platoon-size-extension/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Modified_MADDPG_results/platoon-size-extension/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_BASE_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
RESULT_ROOT="${AOI_RESULT_ROOT:-$AOI_BASE_ROOT/Modified_MADDPG_results/platoon-size-extension}"
RUN_ROOT="$RESULT_ROOT/runs"

ALGORITHM="modified_maddpg"
TAU="0.005"
SLOW_INTERVAL="1"
GLOBAL_MODE="synchronous_joint"
TRAIN_NOISE="0.3"
TASK_COUNT=18
scenarios=(p05_n06_g25 p05_n08_g25 p05_n10_g25)
seeds=(8 9 10 11 12 13)

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this file with sbatch as a Slurm job array." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TASK_COUNT )); then
  echo "Array id must satisfy 0 <= id < $TASK_COUNT; got $SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi

scenario_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))
scenario=${scenarios[$scenario_index]}
seed=${seeds[$seed_index]}
printf -v seed_token "%02d" "$seed"
run_name="modified_maddpg_platoon_${scenario}_seed${seed_token}"
run_dir="$RUN_ROOT/$run_name"
latest="$run_dir/checkpoints/latest.pt"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$AOI_BASE_ROOT/Modified_MADDPG_results/platoon-size-extension" ]]; then
  echo "Refusing to write outside the requested Algorithm 1 platoon-size-extension root: $RESULT_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$RESULT_ROOT/tmp" \
  "$RESULT_ROOT/cache/xdg" "$RESULT_ROOT/cache/matplotlib" \
  "$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export XDG_CACHE_HOME="$RESULT_ROOT/cache/xdg"
export MPLCONFIGDIR="$RESULT_ROOT/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "algorithm=$ALGORITHM task=$SLURM_ARRAY_TASK_ID/$TASK_COUNT scenario=$scenario seed=$seed tau=$TAU slow=$SLOW_INTERVAL global=$GLOBAL_MODE train_noise=$TRAIN_NOISE"
echo "primary_metric_window=last100 secondary_metric_window=last50"
echo "result_root=$RESULT_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"
test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -c 'import sys; from config import resolve_config; c=resolve_config(profile="paper_faithful", scenario=sys.argv[1], algorithm="modified_maddpg", seed=int(sys.argv[2]), tau=0.005, slow_update_every_episodes=1, global_update_mode="synchronous_joint"); assert c.algorithm == "modified_maddpg"; assert c.scenario.number_platoons == 5; assert c.scenario.platoon_size in (6, 8, 10); assert c.scenario.gap_m == 25.0; assert c.exploration_noise == 0.3; assert not c.is_formal_result; print({"algorithm": c.algorithm, "scenario": c.scenario.id, "platoon_size": c.scenario.platoon_size, "seed": c.seed, "training_noise": c.exploration_noise})' "$scenario" "$seed"
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
  --algorithm "$ALGORITHM" \
  --profile paper_faithful \
  --scenario "$scenario" \
  --seed "$seed" \
  --device cuda:0 \
  --run-name "$run_name" \
  --output-root "$RUN_ROOT" \
  --episodes 500 \
  --checkpoint-every 50 \
  --tau "$TAU" \
  --slow-update-every-episodes "$SLOW_INTERVAL" \
  --global-actor-mode "$GLOBAL_MODE" \
  --diagnostics \
  --scope train \
  "${recovery_args[@]}"

test -f "$run_dir/COMPLETE.json"
test -f "$run_dir/train_metrics.npz"
test -f "$run_dir/train_metrics_summary.json"
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) run=$run_name primary_metric_window=last100 secondary_metric_window=last50"
