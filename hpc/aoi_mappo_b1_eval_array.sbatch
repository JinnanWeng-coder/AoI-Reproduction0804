#!/bin/bash
#SBATCH -J aoi_b1_eval
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l20:1
# Submit with explicit --output/--error inside the unique B1 batch root.
set -euo pipefail
source "${PROJECT_DIR:-/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804}/hpc/b1_common.sh"
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]
"$PYTHON_BIN" -u -m analysis.mappo_b1_contract evaluate --batch-root "$BATCH_ROOT" \
  --id "$SLURM_ARRAY_TASK_ID" --attempt "${AOI_ATTEMPT:-a01}" --reason "${AOI_RETRY_REASON:-}"
