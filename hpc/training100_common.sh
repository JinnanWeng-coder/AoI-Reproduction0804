#!/bin/bash
set -euo pipefail
: "${PROJECT_DIR:?}" "${PYTHON_BIN:?}" "${BATCH_ROOT:?}" "${AOI_EXPECTED_COMMIT:?}"
cd "$PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AOI_EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="$PROJECT_DIR" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
token="${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-cpu}"
export TMPDIR="$BATCH_ROOT/tmp/$token" XDG_CACHE_HOME="$BATCH_ROOT/cache/$token"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PYTHON_BIN" -B -c 'import sys; from analysis.training100_contract import load; m,_=load(sys.argv[1]); assert m["implementation_commit"]==sys.argv[2]' "$BATCH_ROOT" "$AOI_EXPECTED_COMMIT"
