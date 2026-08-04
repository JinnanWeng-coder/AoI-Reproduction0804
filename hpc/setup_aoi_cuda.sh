#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/eeedata/sgxyl2/AoI-Reproduction0804}"
CONDA_ROOT="${CONDA_ROOT:-/share/home/sgxyl2/miniconda3}"
AOI_ENV_NAME="${AOI_ENV_NAME:-aoi_cuda}"

cd "$PROJECT_DIR"

if [[ -r "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  if command -v module >/dev/null 2>&1; then
    set +e
    module load anaconda >/dev/null 2>&1
    module_status=$?
    set -e
    if [[ $module_status -eq 0 ]] && command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
    fi
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found." >&2
  echo "Ask Grok to inspect 'module av' or install Miniconda under $CONDA_ROOT, then rerun this script." >&2
  exit 2
fi

if ! conda run -n "$AOI_ENV_NAME" python --version >/dev/null 2>&1; then
  conda create -n "$AOI_ENV_NAME" python=3.10.20 -y
fi

conda activate "$AOI_ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -r requirements.cuda.lock.txt
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip check

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
python -m pytest -q
python -c 'import torch; print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_visible_on_login": torch.cuda.is_available()})'

echo "Environment '$AOI_ENV_NAME' is ready. CUDA availability is checked again inside the GPU pilot job."
