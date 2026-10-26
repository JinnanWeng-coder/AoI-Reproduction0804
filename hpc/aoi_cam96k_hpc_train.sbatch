#!/bin/bash
#SBATCH --job-name=cam96k_train
#SBATCH --partition=Q10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l20:1
source "${PROJECT_DIR:?}/hpc/cam96k_hpc_common.sh"
args=(train-unit --batch-root "$AOI_BATCH_ROOT" --id "${SLURM_ARRAY_TASK_ID:?}" --attempt "${AOI_ATTEMPT:-a01}")
[[ -z "${AOI_RETRY_REASON:-}" ]] || args+=(--reason "$AOI_RETRY_REASON")
[[ -z "${AOI_RESUME_EPISODE:-}" ]] || args+=(--resume-episode "$AOI_RESUME_EPISODE")
"$PYTHON_BIN" -B -u -m analysis.cam96k_hpc "${args[@]}"
