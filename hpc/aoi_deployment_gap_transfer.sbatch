#!/bin/bash
#SBATCH -J aoi_deploy_gap
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
source "${PROJECT_DIR:?}/hpc/deployment_common.sh"
unit_id=$(( ${SLURM_ARRAY_TASK_ID:?Submit as array} + ${AOI_TASK_OFFSET:-0} ))
args=(--batch-root "$AOI_BATCH_ROOT" --phase gap-transfer --unit-id "$unit_id" --device "${AOI_DEVICE:-cuda:0}" --attempt "${AOI_ATTEMPT:-a01}")
[[ -z "${AOI_RETRY_REASON:-}" ]] || args+=(--reason "$AOI_RETRY_REASON")
"$PYTHON_BIN" -B -u -m analysis.evaluate_frozen_deployment "${args[@]}"
[[ "$(git rev-parse HEAD)" == "$AOI_EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=all)" ]]
