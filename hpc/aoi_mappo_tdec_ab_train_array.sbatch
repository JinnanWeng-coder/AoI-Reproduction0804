#!/bin/bash
#SBATCH -J aoi_mappo_tdec_ab_train
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-11%12
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/tdec-ab-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/tdec-ab-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25}"
RUN_ROOT="$RESULT_ROOT/training/runs"

TASK_COUNT=12
SCENARIO="p05_n04_g25"
seeds=(8 9 10 11 12 13)
variants=(combined tdec)
actor_lr="0.0005"
entropy_rb="0.02"
entropy_mode="0.02"
entropy_power="0.002"
value_clip_mode="normalized"

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this file with sbatch as a Slurm job array." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TASK_COUNT )); then
  echo "Array id must satisfy 0 <= id < $TASK_COUNT; got $SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi

variant_index=$((SLURM_ARRAY_TASK_ID / 6))
seed_index=$((SLURM_ARRAY_TASK_ID % 6))
variant=${variants[$variant_index]}
seed=${seeds[$seed_index]}
printf -v seed_token "%02d" "$seed"
run_name="mappo_tdec_ab_${variant}_${SCENARIO}_seed${seed_token}"
run_dir="$RUN_ROOT/$run_name"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25" ]]; then
  echo "Refusing to write outside the requested MAPPO TDec A/B root: $RESULT_ROOT" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$RESULT_ROOT/tmp" \
  "$RESULT_ROOT/cache/xdg" "$RESULT_ROOT/cache/matplotlib" \
  "$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="$RESULT_ROOT/tmp"
export XDG_CACHE_HOME="$RESULT_ROOT/cache/xdg"
export MPLCONFIGDIR="$RESULT_ROOT/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "task=$SLURM_ARRAY_TASK_ID/$TASK_COUNT variant=$variant scenario=$SCENARIO seed=$seed"
echo "frozen_arm=entropy2x actor_lr=$actor_lr entropies=$entropy_rb/$entropy_mode/$entropy_power value_clip=$value_clip_mode"
echo "result_root=$RESULT_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current)"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" - "$variant" "$actor_lr" "$entropy_rb" "$entropy_mode" "$entropy_power" "$value_clip_mode" <<'PY'
import sys
from aoi_v2x_reproduction.config import resolve_config

variant, actor_lr, entropy_rb, entropy_mode, entropy_power, clip_mode = sys.argv[1:]
config = resolve_config(
    scenario="p05_n04_g25",
    algorithm="mappo",
    seed=8,
    mappo_variant=variant,
    mappo_actor_lr=float(actor_lr),
    mappo_entropy_coef_rb=float(entropy_rb),
    mappo_entropy_coef_mode=float(entropy_mode),
    mappo_entropy_coef_power=float(entropy_power),
    mappo_value_clip_mode=clip_mode,
)
assert config.profile == "reproduction_baseline"
assert config.tau == 0.005 and config.slow_update_every_episodes == 1
assert config.mappo_variant == variant and config.mappo_value_clip_mode == "normalized"
assert config.mappo_rollout_episodes == 5 and config.mappo_ppo_epochs == 10
assert config.mappo_actor_lr == float(actor_lr) and config.mappo_critic_lr == 0.0005
assert config.mappo_entropy_coef_rb == float(entropy_rb)
assert config.mappo_entropy_coef_mode == float(entropy_mode)
assert config.mappo_entropy_coef_power == float(entropy_power)
assert config.checkpoint_mode == "policy_only"
print(config.to_dict())
PY
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'
nvidia-smi

if [[ -f "$run_dir/COMPLETE.json" ]]; then
  echo "SKIP completed training cell: $run_name"
  exit 0
fi

"$PYTHON_BIN" -u Main.py \
  --algorithm mappo \
  --scenario "$SCENARIO" \
  --seed "$seed" \
  --device cuda:0 \
  --run-name "$run_name" \
  --output-root "$RUN_ROOT" \
  --episodes 500 \
  --mappo-variant "$variant" \
  --mappo-actor-lr "$actor_lr" \
  --mappo-entropy-coef-rb "$entropy_rb" \
  --mappo-entropy-coef-mode "$entropy_mode" \
  --mappo-entropy-coef-power "$entropy_power" \
  --mappo-value-clip-mode "$value_clip_mode" \
  --diagnostics \
  --scope train

test -f "$run_dir/COMPLETE.json"
test -f "$run_dir/config.resolved.json"
test -f "$run_dir/train_metrics.npz"
test -f "$run_dir/train_metrics_summary.json"
test -f "$run_dir/learning_diagnostics.json"
test -f "$run_dir/policy_final.pt"
test ! -d "$run_dir/checkpoints"
"$PYTHON_BIN" - "$run_dir" "$variant" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir, variant, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
config = json.loads((run_dir / "config.resolved.json").read_text())
complete = json.loads((run_dir / "COMPLETE.json").read_text())
diagnostics = json.loads((run_dir / "learning_diagnostics.json").read_text())
assert config["algorithm"] == complete["algorithm"] == "mappo"
assert config["scenario"]["id"] == complete["scenario"] == "p05_n04_g25"
assert int(config["seed"]) == int(complete["seed"]) == seed
assert config["mappo_variant"] == complete["mappo_variant"] == variant
assert config["mappo_value_clip_mode"] == "normalized"
assert math.isclose(float(config["mappo_actor_lr"]), 0.0005, abs_tol=1e-12)
assert math.isclose(float(config["mappo_entropy_coef_rb"]), 0.02, abs_tol=1e-12)
assert math.isclose(float(config["mappo_entropy_coef_mode"]), 0.02, abs_tol=1e-12)
assert math.isclose(float(config["mappo_entropy_coef_power"]), 0.002, abs_tol=1e-12)
assert complete["status"] == "complete" and complete["update_count"] == 100
assert len(diagnostics) == 100 and all(item["mappo_variant"] == variant for item in diagnostics)
if variant == "tdec":
    assert complete["central_critic_output"] == "scalar_global_and_independent_per_agent_task1_task2_values"
    assert all("global_explained_variance" in item for item in diagnostics)
else:
    assert complete["central_critic_output"] == "per_agent_state_value"
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) variant=$variant seed=$seed run=$run_name"
