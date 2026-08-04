# Environment installation notes

Validated environments:

* `aoi_v2x`: Python 3.9.25, NumPy 1.23.5, PyTorch 2.8.0+cpu,
  PyYAML 6.0.3, pytest 8.4.2.
* `aoi_cuda`: Python 3.10.20, NumPy 1.23.5, PyTorch 2.11.0+cu126,
  PyYAML 6.0.3, pytest 9.1.1, one CUDA device available.

Install the pinned non-PyTorch dependencies with `requirements.lock.txt`, then
select the PyTorch build matching the host:

```text
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest -q
```

The lock file intentionally records CPU and CUDA torch separately; do not
replace one with the other during a formal run.  The validated invocations are:

```text
C:\Users\67497\anaconda3\envs\aoi_v2x\python.exe -m pytest -q
C:\Users\67497\anaconda3\envs\aoi_cuda\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Before remote execution, run `scripts/preflight_network.py --scenario
p05_n10_g25 --batch-size 64` on the selected device.  It performs two learner
updates and verifies global-target-per-step plus delayed local/actor cadence.
Formal execution also requires a clean reproduction Git worktree; smoke and
preflight artifacts are marked non-formal.

Use `--device cpu` for CPU preflight and `--device cuda:0` only when the CUDA
check reports true. This code-completion task does not install packages or
launch formal training.
