# Environment installation

Use Python 3.10 and install the pinned dependencies from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pytest -q
python -m aoi_v2x_reproduction --scenario p05_n04_g25 --dry-run
```

For the tested CUDA environment, use `requirements.cuda.lock.txt` and verify
that `torch.cuda.is_available()` is true. The cluster helper
`hpc/setup_aoi_cuda.sh` creates the equivalent environment under the path set by
`AOI_ENV_DIR`.

Git commit, branch, and dirty state are recorded in run provenance. There is no
separate source manifest or whole-tree digest; ordinary Git history and the
historical tag provide source traceability.
