#!/bin/bash
#SBATCH -J aoi_mappo_rb_audit
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/rb-coordination-audit-v1/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/rb-coordination-audit-v1/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
INPUT_ROOT="${AOI_INPUT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/feasibility-reward-audit-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/rb-coordination-audit-v1/P5_N4_gap25}"
OUTPUT_ROOT="$RESULT_ROOT/analysis"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$INPUT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/feasibility-reward-audit-v1/P5_N4_gap25" ]]; then
  echo "Refusing an input other than feasibility-reward-audit-v1: $INPUT_ROOT" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/rb-coordination-audit-v1/P5_N4_gap25" ]]; then
  echo "Refusing to write outside the dedicated RB audit root: $RESULT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/pycache"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache"

echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current) started=$(date --iso-8601=seconds)"
test -z "$(git status --porcelain --untracked-files=all)"

"$PYTHON_BIN" -u -m analysis.audit_mappo_rb_coordination \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT"

"$PYTHON_BIN" - "$OUTPUT_ROOT/rb_coordination_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["status"] == "PASS"
assert summary["input_cells"] == 6
assert summary["per_eval_seed_agent_rows"] == 180
assert summary["per_training_seed_rows"] == 6
assert summary["rb_usage_rows"] == 1620
assert summary["pairwise_reuse_rows"] == 1440
assert summary["oracle_sample_rows"] == 324
assert summary["special_case_rows"] == 1
assert summary["sample_plan"]["joint_assignments_per_sample"] == 243
PY

test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds)"
