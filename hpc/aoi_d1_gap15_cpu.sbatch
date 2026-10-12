#!/bin/bash
#SBATCH -J d1_gap15_cpu
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -euo pipefail
export AOI_CONTRACT_MODULE=analysis.d1_gap15
source "${PROJECT_DIR:?}/hpc/deployment_common.sh"
stage="${AOI_GAP15_STAGE:?preflight/first-check/finalize}"
case "$stage" in
  preflight)
    df -PB1 "$AOI_BATCH_ROOT"
    df -Pi "$AOI_BATCH_ROOT"
    if command -v quota >/dev/null 2>&1; then quota -s || echo 'quota query unavailable: review cluster quota separately'; else echo 'quota command unavailable: review cluster quota separately'; fi
    ;;
  first-check|finalize|summary|runtime|export) ;;
  *) echo "unsupported gap15 CPU stage: $stage" >&2; exit 1 ;;
esac
"$PYTHON_BIN" -B -u -m analysis.d1_gap15 "$stage" --batch-root "$AOI_BATCH_ROOT" --attempt "${AOI_ATTEMPT:-a01}"
[[ "$(git rev-parse HEAD)" == "$AOI_EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=all)" ]]
