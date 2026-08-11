#!/bin/bash
#SBATCH -J aoi_mappo_policy_eval
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-47%8
#SBATCH --time=02:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/policy-eval-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/policy-eval-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/policy-eval-v1/P5_N4_gap25}"
EVAL_ROOT="$RESULT_ROOT/evaluations"

TASK_COUNT=48
SCENARIO="p05_n04_g25"
EVAL_SEEDS="201,202,203,204,205,206"
seeds=(8 9 10 11 12 13)
arms=(baseline actor_lr1e4 entropy2x actor_lr1e4_entropy2x)
modes=(deterministic stochastic)

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
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/policy-eval-v1/P5_N4_gap25" ]]; then
  echo "Refusing to write outside the requested MAPPO policy-eval root: $RESULT_ROOT" >&2
  exit 2
fi

arm_index=$((SLURM_ARRAY_TASK_ID / 12))
within_arm=$((SLURM_ARRAY_TASK_ID % 12))
seed_index=$((within_arm / 2))
mode_index=$((within_arm % 2))
arm=${arms[$arm_index]}
seed=${seeds[$seed_index]}
mode=${modes[$mode_index]}
printf -v seed_token "%02d" "$seed"

case "$arm" in
  baseline)
    source_root="$DIAGNOSTICS_ROOT/MAPPO_results/default/P5_N4_gap25"
    run_name="mappo_default_${SCENARIO}_seed${seed_token}"
    ;;
  actor_lr1e4|entropy2x)
    source_root="$DIAGNOSTICS_ROOT/MAPPO_results/stability-ablation-v1/P5_N4_gap25"
    run_name="mappo_stability_${arm}_${SCENARIO}_seed${seed_token}"
    ;;
  actor_lr1e4_entropy2x)
    source_root="$DIAGNOSTICS_ROOT/MAPPO_results/combined-confirm-v1/P5_N4_gap25"
    run_name="mappo_combined_${arm}_${SCENARIO}_seed${seed_token}"
    ;;
  *)
    echo "Unknown arm: $arm" >&2
    exit 2
    ;;
esac

policy_path="$source_root/runs/$run_name/policy_final.pt"
eval_id="eval_validation_policy_final_${mode}_sequential_warm_warm5_s201-202-203-204-205-206_ep100"
eval_dir="$EVAL_ROOT/$run_name/$eval_id"

mkdir -p "$EVAL_ROOT" "$RESULT_ROOT/tmp" \
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

echo "task=$SLURM_ARRAY_TASK_ID/$TASK_COUNT arm=$arm seed=$seed mode=$mode"
echo "policy=$policy_path result_root=$RESULT_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current)"
test -z "$(git status --porcelain --untracked-files=all)"
test -f "$policy_path"
test -f "$source_root/runs/$run_name/COMPLETE.json"

if [[ -f "$eval_dir/EVAL_COMPLETE.json" ]]; then
  echo "SKIP completed evaluation: arm=$arm seed=$seed mode=$mode"
  exit 0
fi

"$PYTHON_BIN" - "$source_root/runs/$run_name" "$arm" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
arm = sys.argv[2]
seed = int(sys.argv[3])
specs = {
    "baseline": (0.0005, 0.01, 0.01, 0.001),
    "actor_lr1e4": (0.0001, 0.01, 0.01, 0.001),
    "entropy2x": (0.0005, 0.02, 0.02, 0.002),
    "actor_lr1e4_entropy2x": (0.0001, 0.02, 0.02, 0.002),
}
config = json.loads((run_dir / "config.resolved.json").read_text())
complete = json.loads((run_dir / "COMPLETE.json").read_text())
actual = (
    float(config["mappo_actor_lr"]),
    float(config["mappo_entropy_coef_rb"]),
    float(config["mappo_entropy_coef_mode"]),
    float(config["mappo_entropy_coef_power"]),
)
assert config["algorithm"] == complete["algorithm"] == "mappo"
assert config["scenario"]["id"] == "p05_n04_g25" and int(config["seed"]) == seed
assert complete["status"] == "complete" and complete["policy_final"] == "policy_final.pt"
assert all(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12) for value, expected in zip(actual, specs[arm]))
PY

"$PYTHON_BIN" -u Main.py \
  --algorithm mappo \
  --scenario "$SCENARIO" \
  --seed "$seed" \
  --device cuda:0 \
  --output-root "$EVAL_ROOT" \
  --eval-only \
  --resume "$policy_path" \
  --scope validation \
  --eval-purpose validation \
  --eval-seeds "$EVAL_SEEDS" \
  --eval-episodes 100 \
  --diagnostic-eval \
  --mappo-eval-mode "$mode"

test -f "$eval_dir/EVAL_COMPLETE.json"
test -f "$eval_dir/summary.json"
test -f "$eval_dir/provenance.json"
test -f "$eval_dir/metrics.npz"
"$PYTHON_BIN" - "$eval_dir/EVAL_COMPLETE.json" "$arm" "$seed" "$mode" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
arm, seed, mode = sys.argv[2], int(sys.argv[3]), sys.argv[4]
assert summary["status"] == "complete" and summary["algorithm"] == "mappo"
assert summary["diagnostic_evaluation"] is True and summary["is_formal_result"] is False
assert summary["training_seed"] == seed and summary["mappo_eval_mode"] == mode
assert summary["eval_seeds"] == [201, 202, 203, 204, 205, 206]
assert summary["eval_warmup_episodes"] == 5 and summary["eval_episodes"] == 100
assert summary["external_action_noise_applicable"] is False and summary["eval_noise"] == 0.0
assert 0.0 <= summary["CAM_success_probability"] <= 1.0
assert 0.0 <= summary["payload_completion"] <= 1.0
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) arm=$arm seed=$seed mode=$mode"
