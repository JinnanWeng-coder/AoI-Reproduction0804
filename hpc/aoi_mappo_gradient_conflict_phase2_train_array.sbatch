#!/bin/bash
#SBATCH -J aoi_mappo_gc_p2_train
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-17%6
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/gradient-conflict-phase2-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/gradient-conflict-phase2-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/gradient-conflict-phase2-v1/P5_N4_gap25}"
RUN_ROOT="$RESULT_ROOT/training/runs"

TASK_COUNT=18
SCENARIO="p05_n04_g25"
arms=(composed_clip separate_sum_clip pcgrad)
seeds=(8 9 10 11 12 13)
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

arm_index=$((SLURM_ARRAY_TASK_ID / 6))
seed_index=$((SLURM_ARRAY_TASK_ID % 6))
arm=${arms[$arm_index]}
seed=${seeds[$seed_index]}
printf -v seed_token "%02d" "$seed"
run_name="mappo_gradient_phase2_${arm}_${SCENARIO}_seed${seed_token}"
run_dir="$RUN_ROOT/$run_name"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/gradient-conflict-phase2-v1/P5_N4_gap25" ]]; then
  echo "Refusing to write outside the Phase 2 result root: $RESULT_ROOT" >&2
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
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export XDG_CACHE_HOME="$RESULT_ROOT/cache/xdg"
export MPLCONFIGDIR="$RESULT_ROOT/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "task=$SLURM_ARRAY_TASK_ID/$TASK_COUNT arm=$arm scenario=$SCENARIO seed=$seed"
echo "variant=tdec actor_lr=$actor_lr entropies=$entropy_rb/$entropy_mode/$entropy_power value_clip=$value_clip_mode"
echo "result_root=$RESULT_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current)"
test -z "$(git status --porcelain --untracked-files=all)"

"$PYTHON_BIN" - "$arm" <<'PY'
import sys
from aoi_v2x_reproduction.config import resolve_config

arm = sys.argv[1]
config = resolve_config(
    scenario="p05_n04_g25",
    algorithm="mappo",
    seed=8,
    mappo_variant="tdec",
    mappo_actor_update_mode=arm,
    mappo_actor_lr=0.0005,
    mappo_entropy_coef_rb=0.02,
    mappo_entropy_coef_mode=0.02,
    mappo_entropy_coef_power=0.002,
    mappo_value_clip_mode="normalized",
    mappo_objective_gradient_diagnostics=True,
)
assert config.profile == "reproduction_baseline"
assert config.tau == 0.005 and config.slow_update_every_episodes == 1
assert config.mappo_variant == "tdec" and config.mappo_actor_update_mode == arm
assert config.mappo_rollout_episodes == 5 and config.mappo_ppo_epochs == 10
assert config.mappo_actor_lr == config.mappo_critic_lr == 0.0005
assert config.mappo_entropy_coef_rb == config.mappo_entropy_coef_mode == 0.02
assert config.mappo_entropy_coef_power == 0.002
assert config.mappo_value_clip_mode == "normalized"
assert config.mappo_objective_gradient_diagnostics is True
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
  --checkpoint-mode policy_only \
  --mappo-variant tdec \
  --mappo-actor-update-mode "$arm" \
  --mappo-actor-lr "$actor_lr" \
  --mappo-entropy-coef-rb "$entropy_rb" \
  --mappo-entropy-coef-mode "$entropy_mode" \
  --mappo-entropy-coef-power "$entropy_power" \
  --mappo-value-clip-mode "$value_clip_mode" \
  --mappo-objective-gradient-diagnostics \
  --diagnostics \
  --scope train

for file in COMPLETE.json config.resolved.json train_metrics.npz train_metrics_summary.json learning_diagnostics.json policy_final.pt; do
  test -f "$run_dir/$file"
done
test ! -d "$run_dir/checkpoints"
"$PYTHON_BIN" - "$run_dir" "$arm" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir, arm, seed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
config = json.loads((run_dir / "config.resolved.json").read_text())
complete = json.loads((run_dir / "COMPLETE.json").read_text())
diagnostics = json.loads((run_dir / "learning_diagnostics.json").read_text())
assert config["algorithm"] == complete["algorithm"] == "mappo"
assert config["scenario"]["id"] == complete["scenario"] == "p05_n04_g25"
assert int(config["seed"]) == int(complete["seed"]) == seed
assert config["mappo_variant"] == complete["mappo_variant"] == "tdec"
assert config["mappo_actor_update_mode"] == complete["mappo_actor_update_mode"] == arm
assert config["mappo_value_clip_mode"] == "normalized"
assert config["mappo_objective_gradient_diagnostics"] is True
assert math.isclose(float(config["mappo_actor_lr"]), 0.0005, abs_tol=1e-12)
assert math.isclose(float(config["mappo_entropy_coef_rb"]), 0.02, abs_tol=1e-12)
assert math.isclose(float(config["mappo_entropy_coef_mode"]), 0.02, abs_tol=1e-12)
assert math.isclose(float(config["mappo_entropy_coef_power"]), 0.002, abs_tol=1e-12)
assert complete["status"] == "complete" and complete["update_count"] == 100
assert complete["policy_final"] == "policy_final.pt"
assert len(diagnostics) == 100
for update in diagnostics:
    assert update["mappo_variant"] == "tdec" and update["mappo_actor_update_mode"] == arm
    audit = update["objective_gradient_diagnostics"]
    assert audit["actor_update_mode"] == arm and len(audit["records"]) == 10
    for record in audit["records"]:
        if arm == "pcgrad":
            projection = record["pcgrad_projection"]
            assert projection["schema_version"] == "mappo_pcgrad_projection_v1"
            assert all(math.isfinite(float(value)) for value in projection["projection_magnitude"].values())
        else:
            assert "pcgrad_projection" not in record
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) arm=$arm seed=$seed run=$run_name"
