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

Use `--device cpu` for CPU preflight and `--device cuda:0` only when the CUDA
check reports true. This code-completion task does not install packages or
launch formal training.
