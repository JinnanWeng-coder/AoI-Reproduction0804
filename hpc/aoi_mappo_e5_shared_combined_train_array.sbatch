#!/bin/bash
#SBATCH -J aoi_mappo_e5_train
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l20:1
#SBATCH --array=0-5%6
#SBATCH --time=08:00:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E5-sharing-critic/P5_N4_gap25/slurm_logs/%x_%A_%a.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/E5-sharing-critic/P5_N4_gap25/slurm_logs/%x_%A_%a.err

set -euo pipefail
AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
STUDY_ROOT="${AOI_STUDY_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics/actor-sharing-study}"
PARENT_ROOT="${AOI_PARENT_ROOT:-$STUDY_ROOT/existing-evidence/tdec-ab-v1/P5_N4_gap25}"
RESULT_ROOT="${AOI_RESULT_ROOT:-$STUDY_ROOT/E5-sharing-critic/P5_N4_gap25}"
RUN_ROOT="$RESULT_ROOT/training/runs"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
seeds=(8 9 10 11 12 13)

[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Submit as a Slurm array" >&2; exit 2; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 6 )) || { echo "Array id must be 0-5" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python missing: $PYTHON_BIN" >&2; exit 2; }
[[ "$RESULT_ROOT" == "$STUDY_ROOT/E5-sharing-critic/P5_N4_gap25" ]] || { echo "Unexpected E5 result root" >&2; exit 2; }
seed=${seeds[$SLURM_ARRAY_TASK_ID]}; printf -v seed_token "%02d" "$seed"
parent_name="mappo_tdec_ab_combined_p05_n04_g25_seed${seed_token}"
parent_run="$PARENT_ROOT/training/runs/$parent_name"
run_name="mappo_e5_shared_combined_p05_n04_g25_seed${seed_token}"
run_dir="$RUN_ROOT/$run_name"

mkdir -p "$RUN_ROOT" "$RESULT_ROOT/slurm_logs" "$RESULT_ROOT/tmp" "$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="$RESULT_ROOT/tmp" PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "E5 shared-combined seed=$seed parent=$parent_name commit=$(git rev-parse HEAD)"
test -z "$(git status --porcelain --untracked-files=all)"
test -f "$parent_run/config.resolved.json"; test -f "$parent_run/COMPLETE.json"
test -f "$parent_run/policy_final.pt"; test -f "$parent_run/train_metrics.npz"
if [[ -f "$run_dir/COMPLETE.json" ]]; then echo "SKIP complete $run_name"; exit 0; fi

"$PYTHON_BIN" -u - "$parent_run" "$RUN_ROOT" "$seed" <<'PY'
import json, sys
from pathlib import Path
from analysis.mappo_e5_contract import derive_shared_combined_config, validate_derived_against_parent
from aoi_v2x_reproduction.runtime.runner import train

parent_run, run_root, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
parent = json.loads((parent_run / "config.resolved.json").read_text())
config = derive_shared_combined_config(parent_run, run_root, seed, device="cuda:0")
comparison = validate_derived_against_parent(parent, config.to_dict())
result = train(config)
run_dir = Path(result["run_dir"])
comparison.update({"parent_run": str(parent_run), "derived_run": str(run_dir), "seed": seed})
(run_dir / "parent_config_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
PY

test -f "$run_dir/COMPLETE.json"; test -f "$run_dir/config.resolved.json"
test -f "$run_dir/train_metrics.npz"; test -f "$run_dir/train_metrics_summary.json"
test -f "$run_dir/learning_diagnostics.json"; test -f "$run_dir/policy_final.pt"
test -f "$run_dir/parent_config_comparison.json"; test ! -d "$run_dir/checkpoints"
"$PYTHON_BIN" - "$run_dir" "$seed" <<'PY'
import json, math, sys
from pathlib import Path
run, seed = Path(sys.argv[1]), int(sys.argv[2])
c = json.loads((run / "config.resolved.json").read_text()); done = json.loads((run / "COMPLETE.json").read_text())
d = json.loads((run / "learning_diagnostics.json").read_text()); comparison = json.loads((run / "parent_config_comparison.json").read_text())
assert c["algorithm"] == "mappo" and c["mappo_variant"] == "combined" and c["mappo_actor_sharing"] is True
assert c["scenario"]["id"] == "p05_n04_g25" and c["seed"] == seed
assert c["episodes"] == 500 and c["steps_per_episode"] == 100 and c["checkpoint_mode"] == "policy_only"
assert c["mappo_rollout_episodes"] == 5 and c["mappo_ppo_epochs"] == 10 and c["mappo_value_clip_mode"] == "normalized"
for key, value in (("mappo_actor_lr",.0005),("mappo_critic_lr",.0005),("mappo_entropy_coef_rb",.02),("mappo_entropy_coef_mode",.02),("mappo_entropy_coef_power",.002)):
    assert math.isclose(float(c[key]), value, abs_tol=1e-12)
assert done["status"] == "complete" and done["update_count"] == 100 and done["environment_steps"] == 50000
assert done["mappo_variant"] == "combined" and done["actor_sharing"] is True and done["actor_network_count"] == 1
assert done["actor_optimizer_step_count"] == 1000 and done["central_critic_output"] == "per_agent_state_value"
assert comparison["status"] == "PASS" and len(d) == 100
assert all(x["mappo_variant"] == "combined" and x["actor_sharing"] is True and x["actor_optimizer_steps_this_update"] == 10 for x in d)
PY
test -z "$(git status --porcelain --untracked-files=all)"
