#!/bin/bash
#SBATCH -J aoi_mappo_shared_eval
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-11%6
#SBATCH --time=03:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/shared-actor-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/shared-actor-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
INDEPENDENT_ROOT="${AOI_INDEPENDENT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/shared-actor-v1/P5_N4_gap25}"
SHARED_RUN_ROOT="$RESULT_ROOT/training/runs"
EVAL_ROOT="$RESULT_ROOT/evaluations"

TASK_COUNT=12
seeds=(8 9 10 11 12 13)
structures=(independent shared)
EVAL_SEEDS="213,214,215,216,217,218"

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
if [[ "$INDEPENDENT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25" ]]; then
  echo "Unexpected independent policy root: $INDEPENDENT_ROOT" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/shared-actor-v1/P5_N4_gap25" ]]; then
  echo "Refusing to write outside the requested shared-actor result root: $RESULT_ROOT" >&2
  exit 2
fi

structure_index=$((SLURM_ARRAY_TASK_ID / 6))
seed_index=$((SLURM_ARRAY_TASK_ID % 6))
structure=${structures[$structure_index]}
seed=${seeds[$seed_index]}
printf -v seed_token "%02d" "$seed"
if [[ "$structure" == "independent" ]]; then
  run_name="mappo_tdec_ab_tdec_p05_n04_g25_seed${seed_token}"
  run_dir="$INDEPENDENT_ROOT/training/runs/$run_name"
else
  run_name="mappo_shared_actor_tdec_p05_n04_g25_seed${seed_token}"
  run_dir="$SHARED_RUN_ROOT/$run_name"
fi
policy_path="$run_dir/policy_final.pt"
eval_id="eval_validation_policy_final_stochastic_sequential_warm_warm5_s213-214-215-216-217-218_ep100_feasibility_reward_arm_baseline"
eval_dir="$EVAL_ROOT/$run_name/$eval_id"

mkdir -p "$EVAL_ROOT" "$RESULT_ROOT/tmp" \
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

echo "task=$SLURM_ARRAY_TASK_ID/$TASK_COUNT structure=$structure seed=$seed"
echo "policy=$policy_path result_root=$RESULT_ROOT host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current)"
test -z "$(git status --porcelain --untracked-files=all)"
test -f "$policy_path"
test -f "$run_dir/COMPLETE.json"

if [[ -f "$eval_dir/EVAL_COMPLETE.json" ]]; then
  echo "SKIP completed evaluation: structure=$structure seed=$seed"
  exit 0
fi

"$PYTHON_BIN" -u -m analysis.evaluate_mappo_feasibility_reward \
  --policy "$policy_path" \
  --output-root "$EVAL_ROOT" \
  --arm baseline \
  --device cuda:0 \
  --eval-seeds "$EVAL_SEEDS" \
  --eval-episodes 100

test -f "$eval_dir/EVAL_COMPLETE.json"
test -f "$eval_dir/summary.json"
test -f "$eval_dir/provenance.json"
test -f "$eval_dir/metrics.npz"
"$PYTHON_BIN" - "$eval_dir/EVAL_COMPLETE.json" "$structure" "$seed" "$run_name" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
structure, seed, run_name = sys.argv[2], int(sys.argv[3]), sys.argv[4]
expected_sharing = structure == "shared"
assert summary["status"] == "complete" and summary["algorithm"] == "mappo"
assert summary["mappo_variant"] == "tdec"
assert summary["actor_sharing"] is expected_sharing
assert summary["actor_network_count"] == (1 if expected_sharing else 5)
assert summary["training_seed"] == seed and summary["training_run_name"] == run_name
assert summary["mappo_eval_mode"] == "stochastic"
assert summary["eval_seeds"] == [213, 214, 215, 216, 217, 218]
assert summary["eval_warmup_episodes"] == 5 and summary["eval_episodes"] == 100
assert summary["intervention_arm"] == "baseline" and summary["intervention_db"] == 0.0
assert summary["policy_parameters_unchanged"] is True
assert summary["diagnostic_evaluation"] is True and summary["is_formal_result"] is False
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) structure=$structure seed=$seed"
