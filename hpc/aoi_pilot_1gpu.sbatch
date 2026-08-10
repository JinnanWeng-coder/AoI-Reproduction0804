#!/bin/bash
#SBATCH -J aoi_pilot_s02
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l20:1
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/Modified_MADDPG_with_TDec_results/heldout-formal-matrix/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/Modified_MADDPG_with_TDec_results/heldout-formal-matrix/slurm_logs/%x_%j.err

set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"
DIAGNOSTICS_ROOT="${AOI_DIAGNOSTICS_ROOT:-$AOI_PROJECT_ROOT/AoI-Reproduction-diagnostics}"
EXPECTED_RESULT_ROOT="$DIAGNOSTICS_ROOT/Modified_MADDPG_with_TDec_results/heldout-formal-matrix"
: "${AOI_RESULT_ROOT:?Set AOI_RESULT_ROOT=$EXPECTED_RESULT_ROOT explicitly before submitting this deferred held-out job}"
RESULT_ROOT="$AOI_RESULT_ROOT"
if [[ "$RESULT_ROOT" != "$EXPECTED_RESULT_ROOT" ]]; then
  echo "AOI_RESULT_ROOT must equal $EXPECTED_RESULT_ROOT; got $RESULT_ROOT" >&2
  exit 2
fi
RUN_ROOT="$RESULT_ROOT/runs"
HANDOFF_ROOT="$RESULT_ROOT/handoff"
PILOT_NAME="reproduction_baseline_p05_n04_g25_seed02"
PILOT_DIR="$RUN_ROOT/$PILOT_NAME"
LATEST="$PILOT_DIR/checkpoints/latest.pt"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 2
fi
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="$RESULT_ROOT/tmp"
export XDG_CACHE_HOME="$RESULT_ROOT/cache/xdg"
export MPLCONFIGDIR="$RESULT_ROOT/cache/matplotlib"
export PYTHONPYCACHEPREFIX="$RESULT_ROOT/cache/pycache/pilot_${SLURM_JOB_ID}"

mkdir -p "$RUN_ROOT" "$HANDOFF_ROOT/pilot_audits" "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"

echo "Host: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "visible_gpu_count": torch.cuda.device_count()})'

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
git rev-parse HEAD
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/preflight_network.py \
  --scenario p05_n10_g25 --batch-size 64 --device cuda:0

if [[ ! -f "$PILOT_DIR/COMPLETE.json" ]]; then
  train_recovery=()
  if [[ -f "$LATEST" ]]; then
    train_recovery=(--resume "$LATEST")
  elif [[ -d "$PILOT_DIR" ]]; then
    train_recovery=(--recover-empty-run)
  fi

  "$PYTHON_BIN" -u Main.py \
    --scenario p05_n04_g25 \
    --seed 2 \
    --device cuda:0 \
    --output-root "$RUN_ROOT" \
    --run-name "$PILOT_NAME" \
    --checkpoint-every 5 \
    --scope train \
    "${train_recovery[@]}"
else
  echo "Pilot training is already complete; preserving and auditing the existing artifact."
fi

"$PYTHON_BIN" analysis/audit_results.py "$PILOT_DIR" --scope train \
  | tee "$HANDOFF_ROOT/pilot_audits/train_job${SLURM_JOB_ID}.json"

if [[ ! -f "$PILOT_DIR/VALIDATION_READY.json" ]]; then
  "$PYTHON_BIN" -u Main.py \
    --scenario p05_n04_g25 \
    --seed 2 \
    --device cuda:0 \
    --output-root "$RUN_ROOT" \
    --run-name "$PILOT_NAME" \
    --checkpoint-every 5 \
    --scope validation \
    --eval-only \
    --eval-purpose validation \
    --eval-episodes 100 \
    --eval-seeds 201,202,203,204,205,206 \
    --resume "$LATEST"
else
  echo "Pilot validation is already complete; preserving and auditing the existing artifact."
fi

"$PYTHON_BIN" analysis/audit_results.py "$PILOT_DIR" --scope validation --require-eval \
  | tee "$HANDOFF_ROOT/pilot_audits/validation_job${SLURM_JOB_ID}.json"

test -z "$(git status --porcelain --untracked-files=all)"
echo "Pilot completed at $(date --iso-8601=seconds). Grok must review it before creating $HANDOFF_ROOT/PILOT_APPROVED."
