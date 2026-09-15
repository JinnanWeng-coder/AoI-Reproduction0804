#!/bin/bash
#SBATCH -J aoi_mappo_n1_revisit
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-23%6
#SBATCH --time=03:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/N1-training-world-revisit/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/N1-training-world-revisit/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
STUDY_ROOT="${AOI_STUDY_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics/actor-sharing-study}"
TDEC_ROOT="${AOI_TDEC_ROOT:-$STUDY_ROOT/existing-evidence/tdec-ab-v1/P5_N4_gap25}"
SHARED_TDEC_ROOT="${AOI_SHARED_TDEC_ROOT:-$STUDY_ROOT/existing-evidence/shared-actor-v1/P5_N4_gap25}"
E5_ROOT="${AOI_E5_ROOT:-$STUDY_ROOT/E5-sharing-critic/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$STUDY_ROOT/N1-training-world-revisit/P5_N4_gap25}"
PYTHON_BIN="${PYTHON_BIN:-$AOI_ENV_DIR/bin/python}"
EXPECTED_RESULT_ROOT="$STUDY_ROOT/N1-training-world-revisit/P5_N4_gap25"
seeds=(8 9 10 11 12 13)
structures=(independent shared independent shared)
variants=(combined combined tdec tdec)

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Submit as a Slurm array" >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 24 )) || { echo "Array id must be 0-23" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python missing: $PYTHON_BIN" >&2; exit 2; }
[[ "$RESULT_ROOT" == "$EXPECTED_RESULT_ROOT" ]] || { echo "Unexpected N1 result root" >&2; exit 2; }

group=$((SLURM_ARRAY_TASK_ID / 6))
seed_index=$((SLURM_ARRAY_TASK_ID % 6))
seed=${seeds[$seed_index]}
structure=${structures[$group]}
variant=${variants[$group]}
printf -v seed_token "%02d" "$seed"

if [[ "$structure" == "independent" ]]; then
  source_root="$TDEC_ROOT"
  run_name="mappo_tdec_ab_${variant}_p05_n04_g25_seed${seed_token}"
elif [[ "$variant" == "tdec" ]]; then
  source_root="$SHARED_TDEC_ROOT"
  run_name="mappo_shared_actor_tdec_p05_n04_g25_seed${seed_token}"
else
  source_root="$E5_ROOT"
  run_name="mappo_e5_shared_combined_p05_n04_g25_seed${seed_token}"
fi

run_dir="$source_root/training/runs/$run_name"
policy="$run_dir/policy_final.pt"
eval_id="eval_validation_policy_final_stochastic_sequential_warm_warm5_s${seed}_ep100_feasibility_reward_arm_baseline"
eval_dir="$RESULT_ROOT/evaluations/$run_name/$eval_id"
job_token="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$RESULT_ROOT/evaluations" "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/pycache/$job_token"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/$job_token"

echo "N1 cell=$SLURM_ARRAY_TASK_ID structure=$structure variant=$variant training_seed=$seed eval_world=$seed commit=$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"
test -f "$run_dir/config.resolved.json"
test -f "$run_dir/COMPLETE.json"
test -f "$policy"

if [[ -f "$eval_dir/N1_COMPLETE.json" ]]; then
  "$PYTHON_BIN" -u -m analysis.mappo_n1_contract \
    --study-root "$STUDY_ROOT" --n1-root "$RESULT_ROOT" --cell-id "$SLURM_ARRAY_TASK_ID"
  echo "SKIP validated N1 cell $SLURM_ARRAY_TASK_ID"
  exit 0
fi
[[ ! -e "$eval_dir" ]] || { echo "Incomplete/conflicting N1 directory exists: $eval_dir" >&2; exit 3; }

"$PYTHON_BIN" -u -m analysis.evaluate_mappo_feasibility_reward \
  --policy "$policy" \
  --output-root "$RESULT_ROOT/evaluations" \
  --arm baseline \
  --device cuda:0 \
  --eval-seeds "$seed" \
  --eval-episodes 100

"$PYTHON_BIN" -u -m analysis.mappo_n1_contract \
  --study-root "$STUDY_ROOT" --n1-root "$RESULT_ROOT" \
  --cell-id "$SLURM_ARRAY_TASK_ID" --write-marker
test -f "$eval_dir/N1_COMPLETE.json"
test -z "$(git status --porcelain --untracked-files=all)"
