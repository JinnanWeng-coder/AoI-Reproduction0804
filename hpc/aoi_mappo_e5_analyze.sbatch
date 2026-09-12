#!/bin/bash
#SBATCH -J aoi_mappo_e5_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E5-sharing-critic/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E5-sharing-critic/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail
AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"; AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"; AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
STUDY_ROOT="${AOI_STUDY_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics/actor-sharing-study}"
TDEC_ROOT="${AOI_TDEC_ROOT:-$STUDY_ROOT/existing-evidence/tdec-ab-v1/P5_N4_gap25}"
SHARED_TDEC_ROOT="${AOI_SHARED_TDEC_ROOT:-$STUDY_ROOT/existing-evidence/shared-actor-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$STUDY_ROOT/E5-sharing-critic/P5_N4_gap25}"; PYTHON_BIN="$AOI_ENV_DIR/bin/python"
[[ "$RESULT_ROOT" == "$STUDY_ROOT/E5-sharing-critic/P5_N4_gap25" ]] || { echo "Unexpected E5 result root" >&2; exit 2; }
mkdir -p "$RESULT_ROOT/analysis" "$RESULT_ROOT/figures" "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/analysis_pycache"
cd "$PROJECT_DIR"; export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/analysis_pycache"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -u -m analysis.summarize_mappo_e5_sharing_critic \
  --tdec-ab-root "$TDEC_ROOT" --shared-tdec-root "$SHARED_TDEC_ROOT" --e5-root "$RESULT_ROOT"
"$PYTHON_BIN" - "$RESULT_ROOT/analysis/e5_summary.json" <<'PY'
import json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r["status"]=="PASS" and r["training_cells"]==24 and r["new_training_cells"]==6
assert r["evaluation_cells"]==24 and r["new_evaluation_cells"]==12 and r["reused_evaluation_cells"]==12
assert r["training_episode_rows"]==12000 and r["eval_world_agent_rows"]==720
PY
test -z "$(git status --porcelain --untracked-files=all)"
