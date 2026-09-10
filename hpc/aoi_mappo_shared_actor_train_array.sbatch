#!/bin/bash
#SBATCH -J aoi_mappo_shared_train
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-5%6
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/shared-actor-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/shared-actor-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/shared-actor-v1/P5_N4_gap25}"
RUN_ROOT="$RESULT_ROOT/training/runs"

TASK_COUNT=6
SCENARIO="p05_n04_g25"
seeds=(8 9 10 11 12 13)

if [[ -z "${SLURM_ARRAY_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this file with sbatch as a Slurm job array." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TASK_COUNT )); then
  echo "Array id must satisfy 0 <= id < $TASK_COUNT; got $SLURM_ARRAY_TASK_ID." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/shared-actor-v1/P5_N4_gap25" ]]; then
  echo "Refusing to write outside the requested shared-actor result root: $RESULT_ROOT" >&2
  exit 2
fi

seed=${seeds[$SLURM_ARRAY_TASK_ID]}
printf -v seed_token "%02d" "$seed"
run_name="mappo_shared_actor_tdec_${SCENARIO}_seed${seed_token}"
run_dir="$RUN_ROOT/$run_name"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT/tmp" \
  "$RESULT_ROOT/cache/xdg" "$RESULT_ROOT/cache/matplotlib" \
  "$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export XDG_CACHE_HOME="$RESULT_ROOT/cache/xdg"
export MPLCONFIGDIR="$RESULT_ROOT/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "task=$SLURM_ARRAY_TASK_ID/$TASK_COUNT structure=shared scenario=$SCENARIO seed=$seed"
echo "result_root=$RESULT_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current)"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" - <<'PY'
from aoi_v2x_reproduction.config import resolve_config

c = resolve_config(
    scenario="p05_n04_g25",
    algorithm="mappo",
    seed=8,
    mappo_variant="tdec",
    mappo_actor_sharing=True,
    mappo_actor_lr=0.0005,
    mappo_entropy_coef_rb=0.02,
    mappo_entropy_coef_mode=0.02,
    mappo_entropy_coef_power=0.002,
    mappo_value_clip_mode="normalized",
)
assert c.profile == "reproduction_baseline"
assert c.episodes == 500 and c.steps_per_episode == 100
assert c.mappo_variant == "tdec" and c.mappo_actor_sharing is True
assert c.mappo_actor_lr == c.mappo_critic_lr == 0.0005
assert (c.mappo_entropy_coef_rb, c.mappo_entropy_coef_mode, c.mappo_entropy_coef_power) == (0.02, 0.02, 0.002)
assert c.mappo_value_clip_mode == "normalized"
assert c.mappo_rollout_episodes == 5 and c.mappo_ppo_epochs == 10
assert c.tau == 0.005 and c.checkpoint_mode == "policy_only"
print(c.to_dict())
PY
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'

if [[ -f "$run_dir/COMPLETE.json" ]]; then
  echo "SKIP completed shared training cell: $run_name"
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
  --mappo-variant tdec \
  --mappo-actor-sharing shared \
  --mappo-actor-lr 0.0005 \
  --mappo-entropy-coef-rb 0.02 \
  --mappo-entropy-coef-mode 0.02 \
  --mappo-entropy-coef-power 0.002 \
  --mappo-value-clip-mode normalized \
  --diagnostics \
  --scope train

test -f "$run_dir/COMPLETE.json"
test -f "$run_dir/config.resolved.json"
test -f "$run_dir/train_metrics.npz"
test -f "$run_dir/train_metrics_summary.json"
test -f "$run_dir/learning_diagnostics.json"
test -f "$run_dir/policy_final.pt"
test ! -d "$run_dir/checkpoints"
"$PYTHON_BIN" - "$run_dir" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir, seed = Path(sys.argv[1]), int(sys.argv[2])
config = json.loads((run_dir / "config.resolved.json").read_text())
complete = json.loads((run_dir / "COMPLETE.json").read_text())
diagnostics = json.loads((run_dir / "learning_diagnostics.json").read_text())
assert config["algorithm"] == complete["algorithm"] == "mappo"
assert config["scenario"]["id"] == complete["scenario"] == "p05_n04_g25"
assert int(config["seed"]) == int(complete["seed"]) == seed
assert config["mappo_variant"] == complete["mappo_variant"] == "tdec"
assert config["mappo_actor_sharing"] is complete["actor_sharing"] is True
assert config["mappo_value_clip_mode"] == "normalized"
assert math.isclose(float(config["mappo_actor_lr"]), 0.0005, abs_tol=1e-12)
assert math.isclose(float(config["mappo_critic_lr"]), 0.0005, abs_tol=1e-12)
assert complete["status"] == "complete" and complete["update_count"] == 100
assert complete["environment_steps"] == 50000
assert complete["actor_network_count"] == 1
assert complete["parameter_counts"]["actor_network_count"] == 1
assert complete["actor_optimizer_step_count"] == 1000
assert len(diagnostics) == 100
assert all(row["actor_sharing"] is True and row["actor_network_count"] == 1 for row in diagnostics)
assert all(row["actor_optimizer_steps_this_update"] == 10 for row in diagnostics)
for row in diagnostics:
    for key in ("critic_loss", "critic_grad_norm", "explained_variance"):
        assert math.isfinite(float(row[key]))
    for key in (
        "actor_loss_per_agent", "approx_kl_per_agent", "clip_fraction_per_agent",
        "entropy_rb_per_agent", "entropy_mode_per_agent", "entropy_power_per_agent",
        "actor_grad_norm_per_agent",
    ):
        assert len(row[key]) == 5 and all(math.isfinite(float(value)) for value in row[key])
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) structure=shared seed=$seed run=$run_name"
