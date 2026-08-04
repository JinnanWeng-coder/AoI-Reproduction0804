#!/bin/bash
#SBATCH -J aoi_pilot_s02
#SBATCH -p Q10
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l20:1
#SBATCH --time=6-23:30:00
#SBATCH -o /eeedata/sgxyl2/AoI-Reproduction0804-results/slurm_logs/%x_%j.out
#SBATCH -e /eeedata/sgxyl2/AoI-Reproduction0804-results/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/eeedata/sgxyl2/AoI-Reproduction0804}"
CONDA_ROOT="${CONDA_ROOT:-/share/home/sgxyl2/miniconda3}"
AOI_ENV_NAME="${AOI_ENV_NAME:-aoi_cuda}"
RESULT_ROOT="${AOI_RESULT_ROOT:-/eeedata/sgxyl2/AoI-Reproduction0804-results}"
RUN_ROOT="$RESULT_ROOT/runs"
HANDOFF_ROOT="$RESULT_ROOT/handoff"
PILOT_NAME="paper_faithful_p05_n04_g25_seed02"
PILOT_DIR="$RUN_ROOT/$PILOT_NAME"
LATEST="$PILOT_DIR/checkpoints/latest.pt"

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$AOI_ENV_NAME"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "$RUN_ROOT" "$HANDOFF_ROOT/pilot_audits"

echo "Host: $(hostname)"
echo "Start time: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
which python
python -c 'import torch; assert torch.cuda.is_available(); print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "visible_gpu_count": torch.cuda.device_count()})'

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
git rev-parse HEAD
python -m pytest -q
python scripts/preflight_network.py \
  --scenario p05_n10_g25 --batch-size 64 --device cuda:0

if [[ ! -f "$PILOT_DIR/COMPLETE.json" ]]; then
  train_recovery=()
  if [[ -f "$LATEST" ]]; then
    train_recovery=(--resume "$LATEST")
  elif [[ -d "$PILOT_DIR" ]]; then
    train_recovery=(--recover-empty-run)
  fi

  python -u Main.py \
    --profile paper_faithful \
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

python analysis/audit_results.py "$PILOT_DIR" --scope train \
  | tee "$HANDOFF_ROOT/pilot_audits/train_job${SLURM_JOB_ID}.json"

if [[ ! -f "$PILOT_DIR/VALIDATION_READY.json" ]]; then
  python -u Main.py \
    --profile paper_faithful \
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

python analysis/audit_results.py "$PILOT_DIR" --scope validation --require-eval \
  | tee "$HANDOFF_ROOT/pilot_audits/validation_job${SLURM_JOB_ID}.json"

test -z "$(git status --porcelain --untracked-files=all)"
echo "Pilot completed at $(date --iso-8601=seconds). Grok must review it before creating $HANDOFF_ROOT/PILOT_APPROVED."
