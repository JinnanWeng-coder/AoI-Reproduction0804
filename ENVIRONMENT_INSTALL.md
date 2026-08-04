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

The paper-faithful release lane is `paper_faithful_v4` with mobility revision
`lane_graph_exit_safe_v1` and checkpoint schema `checkpoint_v4`. Legacy
historical checkpoints remain confined to `legacy_release_v1`; paper v1/v2/v3
checkpoints are rejected.

Before remote execution, run the largest-network preflight on the selected
device. It performs two learner updates and verifies global-target-per-step
plus delayed local/actor cadence:

```text
C:\Users\67497\anaconda3\envs\aoi_v2x\python.exe scripts/preflight_network.py --scenario p05_n10_g25 --batch-size 64 --device cpu
C:\Users\67497\anaconda3\envs\aoi_cuda\python.exe scripts/preflight_network.py --scenario p05_n10_g25 --batch-size 64 --device cuda:0
```

Formal execution also requires a clean reproduction Git worktree, matching
commit/branch/tree digest, and an explicit lifecycle scope. Smoke and
preflight artifacts are marked non-formal. A validation pilot uses
`--scope validation --eval-purpose validation`; `final_test` is reserved for
the final release lane and is not a pilot substitute.

Use `--device cpu` for CPU preflight and `--device cuda:0` only when the CUDA
check reports true. This code-completion task does not install packages or
launch formal training.
