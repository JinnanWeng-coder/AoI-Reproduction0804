#!/bin/bash
#SBATCH -J aoi_mappo_e3_analyze
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E3-locked-test/P5_N4_gap25/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E3-locked-test/P5_N4_gap25/slurm_logs/%x_%j.err

set -euo pipefail
BASE="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
PROJECT_ROOT="${AOI_PROJECT_ROOT:-$BASE/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT/AoI-Reproduction0804}"
STUDY_ROOT="${AOI_STUDY_ROOT:-$PROJECT_ROOT/AoI-Reproduction-diagnostics/actor-sharing-study}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$STUDY_ROOT/E3-locked-test/P5_N4_gap25}"
PYTHON_BIN="${PYTHON_BIN:-$BASE/conda_envs/aoi_cuda/bin/python}"
EXPECTED_COMMIT="${AOI_EXPECTED_COMMIT:?Set AOI_EXPECTED_COMMIT}"
[[ "$RESULT_ROOT" == "$STUDY_ROOT/E3-locked-test/P5_N4_gap25" ]] || { echo "Unexpected E3 root" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python missing" >&2; exit 2; }
job_token="${SLURM_JOB_ID:-manual}"
mkdir -p "$RESULT_ROOT/analysis" "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/pycache/$job_token"
cd "$PROJECT_DIR"
[[ "$(git branch --show-current)" == research/mappo && "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || { echo "Wrong Git state" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Dirty checkout" >&2; exit 2; }
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/$job_token"
"$PYTHON_BIN" -m analysis.mappo_e3_contract check-lock --study-root "$STUDY_ROOT"
"$PYTHON_BIN" -u -m analysis.summarize_mappo_e3_locked_test --study-root "$STUDY_ROOT"
"$PYTHON_BIN" - "$RESULT_ROOT/analysis/verification.json" <<'PY'
import json, sys
from pathlib import Path
v=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert v["status"]=="PASS" and v["scope"]=="technical_only"
assert (v["evaluation_cells"],v["policy_world_runs"],v["world_agent_flows"],
        v["scored_agent_slots"],v["cam_agent_episode_events"],v["condition_seed_rows"]) == (24,144,720,7200000,72000,24)
PY
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || { echo "Checkout changed" >&2; exit 2; }
