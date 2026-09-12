#!/bin/bash
#SBATCH -J aoi_e1_maddpg_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E1-p5-baselines/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E1-p5-baselines/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}}"
RESULT_ROOT="${RESULT_ROOT:-${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/actor-sharing-study/E1-p5-baselines/P5_N4_gap25}}"
PYTHON_BIN="${PYTHON_BIN:-$AOI_ENV_DIR/bin/python}"
EXPECTED_RESULT_ROOT="$DIAGNOSTICS_ROOT/actor-sharing-study/E1-p5-baselines/P5_N4_gap25"

[[ -d "$PROJECT_DIR/.git" ]] || { echo "Git checkout missing: $PROJECT_DIR" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python missing: $PYTHON_BIN" >&2; exit 2; }
[[ "$RESULT_ROOT" == "$EXPECTED_RESULT_ROOT" ]] || { echo "Unexpected E1 result root: $RESULT_ROOT" >&2; exit 2; }

JOB_TOKEN="${SLURM_JOB_ID:-manual}"
mkdir -p "$RESULT_ROOT/analysis" "$RESULT_ROOT/figures" "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/analysis_pycache/$JOB_TOKEN"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/analysis_pycache/$JOB_TOKEN"

echo "E1 analysis branch=$(git rev-parse --abbrev-ref HEAD) commit=$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"

"$PYTHON_BIN" -u -m analysis.summarize_actor_sharing_e1 \
  --diagnostics-root "$DIAGNOSTICS_ROOT" \
  --e1-root "$RESULT_ROOT"

test -z "$(git status --porcelain --untracked-files=all)"
