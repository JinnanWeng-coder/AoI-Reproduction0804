#!/bin/bash
#SBATCH --job-name=cam96k_analyze
#SBATCH --partition=Q10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
export CUDA_VISIBLE_DEVICES=''
source "${PROJECT_DIR:?}/hpc/cam96k_hpc_common.sh"
"$PYTHON_BIN" -B -u -m analysis.cam96k_hpc "${AOI_ANALYZE_COMMAND:-summary}" --batch-root "$AOI_BATCH_ROOT"
