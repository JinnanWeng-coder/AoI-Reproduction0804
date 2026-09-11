#!/bin/bash
#SBATCH -J aoi_mappo_zeroshot_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/zero-shot-v1/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/zero-shot-v1/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail
AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
SHARED_ROOT="${AOI_SHARED_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/shared-actor-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/zero-shot-v1/P5_N4_gap25}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"

[[ -x "$PYTHON_BIN" ]] || { echo "Python missing: $PYTHON_BIN" >&2; exit 2; }
mkdir -p "$RESULT_ROOT/analysis/zero_shot" "$RESULT_ROOT/analysis/service_regularity" "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/analysis_pycache"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/analysis_pycache"
test -z "$(git status --porcelain --untracked-files=all)"

"$PYTHON_BIN" -u -m analysis.summarize_mappo_zero_shot \
  --zero-root "$RESULT_ROOT" --shared-reference-root "$SHARED_ROOT"
"$PYTHON_BIN" -u -m analysis.audit_mappo_service_regularity \
  --shared-result-root "$SHARED_ROOT" --output-root "$RESULT_ROOT/analysis/service_regularity"

"$PYTHON_BIN" - "$RESULT_ROOT/analysis/zero_shot/zero_shot_summary.json" "$RESULT_ROOT/analysis/service_regularity/service_regularity_summary.json" <<'PY'
import json, sys
from pathlib import Path
zero = json.loads(Path(sys.argv[1]).read_text())
audit = json.loads(Path(sys.argv[2]).read_text())
assert zero["status"] == "PASS" and zero["evaluation_cells"] == 30 and zero["world_agent_rows"] == 972
assert audit["status"] == "PASS" and audit["source_cells"] == 12 and audit["flow_rows"] == 360
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds)"
