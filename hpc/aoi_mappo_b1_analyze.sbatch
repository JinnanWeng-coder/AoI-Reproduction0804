#!/bin/bash
#SBATCH -J aoi_b1_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Submit with explicit --output/--error inside the unique B1 batch root.
set -euo pipefail
source "${PROJECT_DIR:-/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804}/hpc/b1_common.sh"
"$PYTHON_BIN" -u -m analysis.summarize_mappo_b1_budget analyze --batch-root "$BATCH_ROOT" \
  --attempt "${AOI_ANALYSIS_ATTEMPT:-a01}"
