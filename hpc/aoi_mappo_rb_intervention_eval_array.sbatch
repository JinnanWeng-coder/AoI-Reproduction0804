#!/bin/bash
#SBATCH -J aoi_mappo_rb_int
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-11%6
#SBATCH --time=03:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/rb-intervention-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/rb-intervention-v1/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
SOURCE_ROOT="${AOI_SOURCE_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$DIAGNOSTICS_ROOT/MAPPO_results/rb-intervention-v1/P5_N4_gap25}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"

seeds=(8 9 10 11 12 13)
arms=(baseline joint_rb_strict_improve)
task_id="${SLURM_ARRAY_TASK_ID:?submit as an array}"
if (( task_id < 0 || task_id >= 12 )); then
  echo "Array id must satisfy 0 <= id < 12" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
if [[ "$SOURCE_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/tdec-ab-v1/P5_N4_gap25" ]]; then
  echo "Unexpected policy source: $SOURCE_ROOT" >&2
  exit 2
fi
if [[ "$RESULT_ROOT" != "$DIAGNOSTICS_ROOT/MAPPO_results/rb-intervention-v1/P5_N4_gap25" ]]; then
  echo "Unexpected result root: $RESULT_ROOT" >&2
  exit 2
fi

arm=${arms[$((task_id / 6))]}
seed=${seeds[$((task_id % 6))]}
printf -v seed_token "%02d" "$seed"
run_name="mappo_tdec_ab_tdec_p05_n04_g25_seed${seed_token}"
run_dir="$SOURCE_ROOT/training/runs/$run_name"
policy="$run_dir/policy_final.pt"
eval_id="eval_validation_policy_final_stochastic_sequential_warm_warm5_s207-208-209-210-211-212_ep100_rb_arm_${arm}"
eval_dir="$RESULT_ROOT/evaluations/$run_name/$eval_id"

mkdir -p "$RESULT_ROOT/evaluations" "$RESULT_ROOT/tmp" \
  "$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${task_id}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${task_id}"

echo "task=$task_id arm=$arm seed=$seed commit=$(git rev-parse HEAD) started=$(date --iso-8601=seconds)"
test -z "$(git status --porcelain --untracked-files=all)"
test -f "$policy"
test -f "$run_dir/COMPLETE.json"
if [[ -f "$eval_dir/EVAL_COMPLETE.json" ]]; then
  echo "SKIP completed cell: arm=$arm seed=$seed"
  exit 0
fi

"$PYTHON_BIN" - "$run_dir" "$seed" <<'PY'
import json
import sys
from pathlib import Path

run, seed = Path(sys.argv[1]), int(sys.argv[2])
config = json.loads((run / "config.resolved.json").read_text(encoding="utf-8"))
complete = json.loads((run / "COMPLETE.json").read_text(encoding="utf-8"))
assert config["algorithm"] == complete["algorithm"] == "mappo"
assert config["mappo_variant"] == complete["mappo_variant"] == "tdec"
assert config["scenario"]["id"] == complete["scenario"] == "p05_n04_g25"
assert int(config["seed"]) == int(complete["seed"]) == seed
assert config["mappo_value_clip_mode"] == "normalized"
assert config["eval_protocol"] == "sequential_warm"
assert int(config["eval_warmup_episodes"]) == 5
assert complete["status"] == "complete" and complete["policy_final"] == "policy_final.pt"
PY

"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -u -m analysis.evaluate_mappo_rb_intervention \
  --policy "$policy" \
  --output-root "$RESULT_ROOT/evaluations" \
  --arm "$arm" \
  --device cuda:0 \
  --eval-seeds 207,208,209,210,211,212 \
  --eval-episodes 100

test -f "$eval_dir/EVAL_COMPLETE.json"
test -f "$eval_dir/provenance.json"
test -f "$eval_dir/metrics.npz"
"$PYTHON_BIN" - "$eval_dir/EVAL_COMPLETE.json" "$arm" "$seed" <<'PY'
import json
import sys
from pathlib import Path

item = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
arm, seed = sys.argv[2], int(sys.argv[3])
assert item["status"] == "complete"
assert item["intervention_arm"] == arm and item["training_seed"] == seed
assert item["eval_seeds"] == [207, 208, 209, 210, 211, 212]
assert item["eval_episodes"] == 100 and item["eval_warmup_episodes"] == 5
assert item["policy_parameters_unchanged"] is True
if arm == "baseline":
    assert item["baseline_action_passthrough"] is True
else:
    assert item["baseline_action_passthrough"] is None
assert item["diagnostic_evaluation"] is True and item["is_formal_result"] is False
PY
test -z "$(git status --porcelain --untracked-files=all)"
echo "completed=$(date --iso-8601=seconds) arm=$arm seed=$seed"
