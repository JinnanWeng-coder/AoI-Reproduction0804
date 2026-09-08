#!/bin/bash
#SBATCH -J aoi_mappo_power_eval
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-17%6
#SBATCH --time=02:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/mode-power-intervention-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/mode-power-intervention-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
SOURCE_ROOT="${AOI_SOURCE_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/mode-power-intervention-v1/P5_N4_gap25}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"

TASK_COUNT=18
seeds=(8 9 10 11 12 13)
arms=(baseline v2i_plus3db v2v_minus3db)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this launcher with sbatch as an array." >&2
  exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TASK_COUNT )); then
  echo "Array id must satisfy 0 <= id < $TASK_COUNT." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$SOURCE_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25" ]]; then
  echo "Refusing a source other than tdec-ab-v1: $SOURCE_ROOT" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/mode-power-intervention-v1/P5_N4_gap25" ]]; then
  echo "Refusing to write outside the dedicated intervention root: $RESULT_ROOT" >&2
  exit 2
fi

arm_index=$((SLURM_ARRAY_TASK_ID / 6))
seed_index=$((SLURM_ARRAY_TASK_ID % 6))
arm=${arms[$arm_index]}
seed=${seeds[$seed_index]}
printf -v seed_token "%02d" "$seed"
run_name="mappo_tdec_ab_tdec_p05_n04_g25_seed${seed_token}"
run_dir="$SOURCE_ROOT/training/runs/$run_name"
policy_path="$run_dir/policy_final.pt"
eval_id="eval_validation_policy_final_stochastic_sequential_warm_warm5_s201-202-203-204-205-206_ep100_arm_${arm}"
eval_dir="$RESULT_ROOT/evaluations/$run_name/$eval_id"

mkdir -p "$RESULT_ROOT/evaluations" "$RESULT_ROOT/tmp" \
  "$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "task=$SLURM_ARRAY_TASK_ID/$TASK_COUNT arm=$arm training_seed=$seed"
echo "policy=$policy_path result=$eval_dir host=$(hostname) started=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current)"
test -z "$(git status --porcelain --untracked-files=all)"
test -f "$policy_path"
test -f "$run_dir/COMPLETE.json"

if [[ -f "$eval_dir/EVAL_COMPLETE.json" ]]; then
  echo "SKIP completed cell: arm=$arm seed=$seed"
  exit 0
fi

"$PYTHON_BIN" - "$run_dir" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir, seed = Path(sys.argv[1]), int(sys.argv[2])
config = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
complete = json.loads((run_dir / "COMPLETE.json").read_text(encoding="utf-8"))
assert config["algorithm"] == complete["algorithm"] == "mappo"
assert config["mappo_variant"] == complete["mappo_variant"] == "tdec"
assert config["scenario"]["id"] == complete["scenario"] == "p05_n04_g25"
assert int(config["seed"]) == int(complete["seed"]) == seed
assert config["mappo_value_clip_mode"] == "normalized"
assert math.isclose(float(config["mappo_actor_lr"]), 0.0005, abs_tol=1e-12)
assert math.isclose(float(config["mappo_entropy_coef_rb"]), 0.02, abs_tol=1e-12)
assert math.isclose(float(config["mappo_entropy_coef_mode"]), 0.02, abs_tol=1e-12)
assert math.isclose(float(config["mappo_entropy_coef_power"]), 0.002, abs_tol=1e-12)
assert complete["status"] == "complete" and complete["policy_final"] == "policy_final.pt"
PY

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})'

"$PYTHON_BIN" -u -m analysis.evaluate_mappo_power_intervention \
  --policy "$policy_path" \
  --output-root "$RESULT_ROOT/evaluations" \
  --arm "$arm" \
  --device cuda:0 \
  --eval-seeds "201,202,203,204,205,206" \
  --eval-episodes 100

test -f "$eval_dir/EVAL_COMPLETE.json"
test -f "$eval_dir/provenance.json"
test -f "$eval_dir/metrics.npz"
"$PYTHON_BIN" - "$eval_dir/EVAL_COMPLETE.json" "$arm" "$seed" <<'PY'
import json
import sys
from pathlib import Path

complete = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
arm, seed = sys.argv[2], int(sys.argv[3])
assert complete["status"] == "complete"
assert complete["algorithm"] == "mappo" and complete["mappo_variant"] == "tdec"
assert complete["intervention_arm"] == arm and complete["training_seed"] == seed
assert complete["mappo_eval_mode"] == "stochastic"
assert complete["eval_seeds"] == [201, 202, 203, 204, 205, 206]
assert complete["eval_warmup_episodes"] == 5 and complete["eval_episodes"] == 100
assert complete["policy_parameters_unchanged"] is True
assert complete["diagnostic_evaluation"] is True and complete["is_formal_result"] is False
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) arm=$arm seed=$seed"
