#!/bin/bash
#SBATCH -J aoi_training100
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:l20:1
set -euo pipefail
source "${PROJECT_DIR:?}/hpc/training100_common.sh"
[[ -z "${AOI_TASK_OFFSET:-}" ]] || { echo 'Use absolute run IDs; offset forbidden' >&2; exit 1; }
args=(train --batch-root "$BATCH_ROOT" --id "${SLURM_ARRAY_TASK_ID:?}" --attempt "${AOI_ATTEMPT:-a01}")
[[ -z "${AOI_RETRY_REASON:-}" ]] || args+=(--reason "$AOI_RETRY_REASON")
"$PYTHON_BIN" -B -u -m analysis.training100_contract "${args[@]}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
