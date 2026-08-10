#!/usr/bin/env bash
set -euo pipefail

AOI_BASE_ROOT="${AOI_BASE_ROOT:-/eeedata/sgxjw2}"
AOI_PROJECT_ROOT="${AOI_PROJECT_ROOT:-$AOI_BASE_ROOT/Parvini-TVT2023-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$AOI_PROJECT_ROOT/AoI-Reproduction0804}"
CONDA_ROOT="${CONDA_ROOT:-$AOI_BASE_ROOT/miniconda3}"
AOI_ENV_DIR="${AOI_ENV_DIR:-$AOI_BASE_ROOT/conda_envs/aoi_cuda}"
PYTHON_BIN="$AOI_ENV_DIR/bin/python"

export TMPDIR="$AOI_BASE_ROOT/tmp"
export PIP_CACHE_DIR="$AOI_BASE_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$AOI_BASE_ROOT/cache/conda_pkgs"
export XDG_CACHE_HOME="$AOI_BASE_ROOT/cache/xdg"
export MPLCONFIGDIR="$AOI_BASE_ROOT/cache/matplotlib"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$(dirname "$AOI_ENV_DIR")"

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

if [[ ! -x "$PYTHON_BIN" ]]; then
  conda create --prefix "$AOI_ENV_DIR" python=3.10.20 -y
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.cuda.lock.txt
"$PYTHON_BIN" -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu126
"$PYTHON_BIN" -m pip check

test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" -c 'import torch; print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_visible_on_login": torch.cuda.is_available()})'

echo "Environment '$AOI_ENV_DIR' is ready. CUDA availability is checked again inside the GPU pilot job."
