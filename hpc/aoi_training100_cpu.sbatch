#!/bin/bash
#SBATCH -J aoi_training100_cpu
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail
source "${PROJECT_DIR:?}/hpc/training100_common.sh"
stage="${AOI_TRAIN100_STAGE:?audit/pilot-check/summary/export}"
case "$stage" in audit|pilot-check|summary|export) ;; *) echo 'bad CPU stage' >&2; exit 1 ;; esac
"$PYTHON_BIN" -B -u -m analysis.training100_contract "$stage" --batch-root "$BATCH_ROOT" --attempt "${AOI_ATTEMPT:-a01}"
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
