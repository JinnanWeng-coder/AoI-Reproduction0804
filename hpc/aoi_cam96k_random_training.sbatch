#!/bin/bash
#SBATCH --job-name=cam96k_random_train
#SBATCH --partition=Q10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${PROJECT_DIR:?}"
test "$(git rev-parse HEAD)" = "${AOI_EXPECTED_COMMIT:?}"
test -z "$(git status --porcelain --untracked-files=all)"
: "${PYTHON_BIN:?}" "${AOI_RANDOM_BATCH:?}"
case "${AOI_RANDOM_STAGE:?}" in
  rollouts)
    id=${SLURM_ARRAY_TASK_ID:?}
    [[ "$id" =~ ^[0-5]$ ]] || { echo 'invalid Random array index' >&2; exit 1; }
    args=(--batch-root "$AOI_RANDOM_BATCH" --seed "$((id + 8))" --attempt "${AOI_RANDOM_ATTEMPT:-a01}")
    if [[ -n "${AOI_RANDOM_REASON:-}" ]]; then args+=(--reason "$AOI_RANDOM_REASON"); fi
    "$PYTHON_BIN" -B -u -m analysis.cam96k_random_training run-unit "${args[@]}"
    ;;
  summary)
    "$PYTHON_BIN" -B -u -m analysis.cam96k_random_training summary --batch-root "$AOI_RANDOM_BATCH" --attempt "${AOI_RANDOM_ANALYSIS_ATTEMPT:-a01}"
    ;;
  *) echo 'invalid Random stage' >&2; exit 1 ;;
esac
