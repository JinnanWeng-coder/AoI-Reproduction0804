#!/bin/bash
#SBATCH -J aoi_mappo_e5_eval
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-11%6
#SBATCH --time=03:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E5-sharing-critic/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E5-sharing-critic/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail
AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
STUDY_ROOT="${AOI_STUDY_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics/actor-sharing-study}"
PARENT_ROOT="${AOI_PARENT_ROOT:-$STUDY_ROOT/existing-evidence/tdec-ab-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$STUDY_ROOT/E5-sharing-critic/P5_N4_gap25}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"; seeds=(8 9 10 11 12 13)

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Submit as a Slurm array" >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 12 )) || { echo "Array id must be 0-11" >&2; exit 2; }
[[ "$RESULT_ROOT" == "$STUDY_ROOT/E5-sharing-critic/P5_N4_gap25" ]] || { echo "Unexpected E5 result root" >&2; exit 2; }
structure=$([[ $((SLURM_ARRAY_TASK_ID / 6)) -eq 0 ]] && echo independent || echo shared)
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 6))]}; printf -v seed_token "%02d" "$seed"
if [[ "$structure" == independent ]]; then
  run_name="mappo_tdec_ab_combined_p05_n04_g25_seed${seed_token}"
  run_dir="$PARENT_ROOT/training/runs/$run_name"
else
  run_name="mappo_e5_shared_combined_p05_n04_g25_seed${seed_token}"
  run_dir="$RESULT_ROOT/training/runs/$run_name"
fi
policy="$run_dir/policy_final.pt"; eval_root="$RESULT_ROOT/evaluations"
eval_id="eval_validation_policy_final_stochastic_sequential_warm_warm5_s213-214-215-216-217-218_ep100_feasibility_reward_arm_baseline"
eval_dir="$eval_root/$run_name/$eval_id"

mkdir -p "$eval_root" "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$PROJECT_DIR"; export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
test -z "$(git status --porcelain --untracked-files=all)"; test -f "$policy"; test -f "$run_dir/COMPLETE.json"
if [[ -f "$eval_dir/EVAL_COMPLETE.json" ]]; then echo "SKIP complete $structure/$seed"; exit 0; fi
"$PYTHON_BIN" -u -m analysis.evaluate_mappo_feasibility_reward \
  --policy "$policy" --output-root "$eval_root" --arm baseline --device cuda:0 \
  --eval-seeds 213,214,215,216,217,218 --eval-episodes 100
test -f "$eval_dir/EVAL_COMPLETE.json"; test -f "$eval_dir/metrics.npz"
"$PYTHON_BIN" - "$eval_dir/EVAL_COMPLETE.json" "$structure" "$seed" "$run_name" <<'PY'
import json, sys
from pathlib import Path
s=json.loads(Path(sys.argv[1]).read_text()); structure,seed,run=sys.argv[2],int(sys.argv[3]),sys.argv[4]
assert s["status"]=="complete" and s["mappo_variant"]=="combined"
assert s["actor_sharing"] is (structure=="shared") and s["actor_network_count"]==(1 if structure=="shared" else 5)
assert s["training_seed"]==seed and s["training_run_name"]==run
assert s["eval_seeds"]==[213,214,215,216,217,218] and s["mappo_eval_mode"]=="stochastic"
assert s["intervention_arm"]=="baseline" and s["policy_parameters_unchanged"] is True
PY
test -z "$(git status --porcelain --untracked-files=all)"
