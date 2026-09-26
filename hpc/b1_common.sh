#!/bin/bash
# Sourced only after sbatch exports the fixed batch and implementation commit.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804}"
PYTHON_BIN="${PYTHON_BIN:-/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python}"
BATCH_ROOT="${AOI_BATCH_ROOT:?Set AOI_BATCH_ROOT}"
EXPECTED_COMMIT="${AOI_EXPECTED_COMMIT:?Set AOI_EXPECTED_COMMIT}"
cd "$PROJECT_DIR"
[[ "$(git branch --show-current)" == research/mappo && "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" && -x "$PYTHON_BIN" ]]
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" -m analysis.mappo_b1_contract preflight --batch-root "$BATCH_ROOT"
job_token="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-analysis}"
mkdir -p "$BATCH_ROOT/tmp/$job_token" "$BATCH_ROOT/cache/$job_token"
export TMPDIR="$BATCH_ROOT/tmp/$job_token" XDG_CACHE_HOME="$BATCH_ROOT/cache/$job_token"
export TORCH_HOME="$XDG_CACHE_HOME/torch" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
