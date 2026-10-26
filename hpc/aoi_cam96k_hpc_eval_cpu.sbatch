#!/bin/bash
#SBATCH --job-name=cam96k_cpu_eval
#SBATCH --partition=Q10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
export CUDA_VISIBLE_DEVICES=''
source "${PROJECT_DIR:?}/hpc/cam96k_hpc_common.sh"
args=(eval-unit --batch-root "$AOI_BATCH_ROOT" --id "${SLURM_ARRAY_TASK_ID:?}" --attempt "${AOI_ATTEMPT:-a01}")
[[ -z "${AOI_RETRY_REASON:-}" ]] || args+=(--reason "$AOI_RETRY_REASON")
"$PYTHON_BIN" -B -u -m analysis.cam96k_hpc "${args[@]}"
