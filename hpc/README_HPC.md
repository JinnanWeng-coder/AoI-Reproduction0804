# HPC launchers

All launchers assume the repository is at
`/eeedata/sgxjw2/AoI-Reproduction0804` and the Python environment at
`/eeedata/sgxjw2/conda_envs/aoi_cuda`. Override `PROJECT_DIR`, `AOI_ENV_DIR`, or
result-root variables when required.

Current lightweight Algorithm 1 arrays:

- `aoi_modified_maddpg_default_array.sbatch`: P=5, N=4, gap=25, seeds 8–13
- `aoi_modified_maddpg_gap_array.sbatch`: gaps 5/15/35, seeds 8–13
- `aoi_modified_maddpg_platoon_size_array.sbatch`: N=6/8/10, seeds 8–13

They train only, use the fixed `tau=0.005`, slow-update-1, synchronous-global
baseline, and write one `policy_final.pt` without replay or periodic
checkpoints.

The pilot and matrix launchers are retained for a later held-out stage. Their
matrix helper explicitly uses resumable checkpoints. They should not be started
during early algorithm development without reviewing the storage budget and
evaluation protocol.

Before submission:

```bash
test "$(git branch --show-current)" = main
test -z "$(git status --porcelain --untracked-files=all)"
/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python -m pytest -q
/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python scripts/preflight_network.py --scenario p05_n10_g25 --device cuda:0
```
