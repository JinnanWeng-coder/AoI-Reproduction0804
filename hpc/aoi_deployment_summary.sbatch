#!/bin/bash
#SBATCH -J aoi_deploy_summary
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -euo pipefail
source "${PROJECT_DIR:?}/hpc/deployment_common.sh"
args=(summary --batch-root "$AOI_BATCH_ROOT" --phase "${AOI_PHASE:?Set AOI_PHASE}" --attempt "${AOI_ANALYSIS_ATTEMPT:-a01}")
[[ -z "${AOI_EVIDENCE_PACK:-}" ]] || args+=(--evidence-pack "$AOI_EVIDENCE_PACK")
"$PYTHON_BIN" -B -u -m analysis.summarize_frozen_deployment "${args[@]}"
[[ "$(git rev-parse HEAD)" == "$AOI_EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=all)" ]]
