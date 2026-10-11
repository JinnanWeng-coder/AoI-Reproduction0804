#!/bin/bash
set -euo pipefail
: "${PROJECT_DIR:?Set PROJECT_DIR to the pinned clean execution checkout}"
: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${AOI_BATCH_ROOT:?Set AOI_BATCH_ROOT}"
: "${AOI_EXPECTED_COMMIT:?Set AOI_EXPECTED_COMMIT}"
cd "$PROJECT_DIR"
[[ -x "$PYTHON_BIN" && -f "$AOI_BATCH_ROOT/protocol/manifest.json" ]]
[[ "$(git rev-parse HEAD)" == "$AOI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
task_token="${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-summary}"
export TMPDIR="$AOI_BATCH_ROOT/tmp/$task_token" XDG_CACHE_HOME="$AOI_BATCH_ROOT/cache/$task_token"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$CUDA_CACHE_PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$PYTHON_BIN" -B -c 'import sys; from pathlib import Path; from analysis.deployment_contract import load_batch; m,_=load_batch(Path(sys.argv[1])); assert m["implementation_commit"] == sys.argv[2]' "$AOI_BATCH_ROOT" "$AOI_EXPECTED_COMMIT"
