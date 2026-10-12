#!/bin/bash
#SBATCH -J d1_gap15_eval
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
export AOI_CONTRACT_MODULE=analysis.d1_gap15
source "${PROJECT_DIR:?}/hpc/deployment_common.sh"
[[ -z "${AOI_TASK_OFFSET:-}" ]] || { echo 'gap15 uses absolute array IDs, not offsets' >&2; exit 1; }
args=(run --batch-root "$AOI_BATCH_ROOT" --unit-id "${SLURM_ARRAY_TASK_ID:?}" --device "${AOI_DEVICE:-cuda:0}" --attempt "${AOI_ATTEMPT:-a01}")
[[ -z "${AOI_RETRY_REASON:-}" ]] || args+=(--reason "$AOI_RETRY_REASON")
[[ "${AOI_ACCEPT_EXISTING:-0}" != 1 ]] || args+=(--accept-existing)
[[ "${AOI_CHECK_ONLY:-0}" != 1 ]] || args+=(--check-only)
"$PYTHON_BIN" -B -u -m analysis.d1_gap15 "${args[@]}"
[[ "$(git rev-parse HEAD)" == "$AOI_EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=all)" ]]
