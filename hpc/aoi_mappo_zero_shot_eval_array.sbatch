#!/bin/bash
#SBATCH -J aoi_mappo_zeroshot
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-29%6
#SBATCH --time=03:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/zero-shot-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/zero-shot-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
INDEPENDENT_ROOT="${AOI_INDEPENDENT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25}"
SHARED_ROOT="${AOI_SHARED_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/shared-actor-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/zero-shot-v1/P5_N4_gap25}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
EVAL_SEEDS="213,214,215,216,217,218"
seeds=(8 9 10 11 12 13)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this file with sbatch as an array." >&2; exit 2
fi
task_id=$((SLURM_ARRAY_TASK_ID))
if (( task_id < 0 || task_id >= 30 )); then
  echo "Array id must satisfy 0 <= id < 30; got $task_id." >&2; exit 2
fi
if (( task_id < 24 )); then
  gap_block=$((task_id / 12))
  within=$((task_id % 12))
  target_scenario=$([[ $gap_block -eq 0 ]] && echo p05_n04_g05 || echo p05_n04_g35)
  structure=$([[ $((within / 6)) -eq 0 ]] && echo independent || echo shared)
  seed=${seeds[$((within % 6))]}
else
  target_scenario=p07_n04_g25
  structure=shared
  seed=${seeds[$((task_id - 24))]}
fi
printf -v seed_token "%02d" "$seed"
if [[ "$structure" == independent ]]; then
  run_name="mappo_tdec_ab_tdec_p05_n04_g25_seed${seed_token}"
  run_dir="$INDEPENDENT_ROOT/training/runs/$run_name"
else
  run_name="mappo_shared_actor_tdec_p05_n04_g25_seed${seed_token}"
  run_dir="$SHARED_ROOT/training/runs/$run_name"
fi
policy="$run_dir/policy_final.pt"
eval_id="eval_zero_shot_target_${target_scenario}_policy_final_stochastic_sequential_warm_warm5_s213-214-215-216-217-218_ep100"
eval_dir="$RESULT_ROOT/$target_scenario/$structure/$run_name/$eval_id"

[[ -x "$PYTHON_BIN" ]] || { echo "Python missing: $PYTHON_BIN" >&2; exit 2; }
[[ "$RESULT_ROOT" == "$DIAGNOSTICS_ROOT/MAPPO_results/zero-shot-v1/P5_N4_gap25" ]] || { echo "Unexpected result root" >&2; exit 2; }
mkdir -p "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${task_id}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${task_id}"

echo "task=$task_id target=$target_scenario structure=$structure training_seed=$seed commit=$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"
test -f "$policy"; test -f "$run_dir/COMPLETE.json"
if [[ -f "$eval_dir/EVAL_COMPLETE.json" ]]; then
  echo "SKIP completed: $eval_dir"; exit 0
fi
"$PYTHON_BIN" -u -m analysis.evaluate_mappo_zero_shot \
  --policy "$policy" --output-root "$RESULT_ROOT" --target-scenario "$target_scenario" \
  --device cuda:0 --eval-seeds "$EVAL_SEEDS" --eval-episodes 100

test -f "$eval_dir/EVAL_COMPLETE.json"; test -f "$eval_dir/metrics.npz"
"$PYTHON_BIN" - "$eval_dir/EVAL_COMPLETE.json" "$target_scenario" "$structure" "$seed" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
target, structure, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
assert s["status"] == "complete" and s["source_scenario"] == "p05_n04_g25"
assert s["target_scenario"] == target and s["actor_structure"] == structure
assert s["training_seed"] == seed and s["target_number_agents"] == (7 if target.startswith("p07") else 5)
assert s["critic_constructed"] is False and s["optimizer_constructed"] is False
assert s["policy_parameters_unchanged"] is True and s["action_intervention"] == "none"
assert s["eval_seeds"] == [213,214,215,216,217,218]
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds)"
