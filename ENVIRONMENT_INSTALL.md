# Environment installation

The formal remote lane is Linux, Python 3.10.20, and an NVIDIA CUDA device.
The locally validated CUDA environment used PyTorch `2.11.0+cu126`. The CPU
environment remains useful for tests and diagnostics, but it is not the
recommended 48-run execution target.

## Clone and establish provenance

```bash
git clone --branch main --single-branch \
  https://github.com/JinnanWeng-coder/AoI-Reproduction0804.git
cd AoI-Reproduction0804

test -n "$(git branch --show-current)"
test -z "$(git status --porcelain --untracked-files=all)"
git rev-parse HEAD
```

Formal train/resume/evaluation requires the same non-detached branch, commit,
tracked-tree digest, clean worktree, and tracked `SOURCE_MANIFEST.json`.
Do not pull, switch branches, edit tracked files, or generate unignored files
inside the repository between these stages. `SOURCE_MANIFEST.json` contains
historical Windows source paths; the remote machine does not need that original
`src` clone and must not regenerate the manifest.

Put runs, matrix reports, study manifests, and figures outside the Git checkout.
The default `experiments/runs` and `scratch` directories are ignored, but an
external high-capacity filesystem is preferred.

## CUDA environment

```bash
conda create -n aoi_cuda python=3.10.20 -y
conda activate aoi_cuda
python -m pip install --upgrade pip
python -m pip install -r requirements.cuda.lock.txt
python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip check
```

The PyTorch command is the official CUDA 12.6 combination for version 2.11.0.
The two requirements files pin the directly used non-PyTorch packages; they do
not pretend to be a cross-platform lock for the CUDA wheel and all transitive
dependencies.

Verify the host before any formal run:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q
python scripts/preflight_network.py \
  --scenario p05_n10_g25 --batch-size 64 --device cuda:0
```

On Linux the read-only test that reopens the historical Windows source checkout
is expected to skip because that separate checkout is intentionally not copied
to the remote host. The tracked manifest schema/digest and all formal runtime
provenance gates remain active; a skip of any other formal-contract test is not
expected on the GPU node.

Expected preflight properties are `status=pass`, `state_dim=46`, two global
target updates, and delayed local/actor target updates `[false, true]`.

## CPU diagnostic environment

```bash
conda create -n aoi_cpu python=3.9.25 -y
conda activate aoi_cpu
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip check
python -m pytest -q
python scripts/preflight_network.py \
  --scenario p05_n10_g25 --batch-size 64 --device cpu
```

Validated local versions were:

- CPU: Python 3.9.25, NumPy 1.23.5, SciPy 1.13.1, Matplotlib 3.9.4,
  PyYAML 6.0.3, pytest 8.4.2, PyTorch 2.8.0+cpu.
- CUDA: Python 3.10.20, NumPy 1.23.5, SciPy 1.15.3, Matplotlib 3.10.8,
  PyYAML 6.0.3, pytest 9.1.1, PyTorch 2.11.0+cu126.

See `REMOTE_RUNBOOK.md` for the staged formal execution and recovery commands.
