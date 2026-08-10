# Remote runbook

## Prepare

```bash
cd /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804
git fetch origin --tags
git checkout main
git pull --ff-only origin main
test -z "$(git status --porcelain --untracked-files=all)"
/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python -m pytest -q
/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python scripts/preflight_network.py --scenario p05_n10_g25 --device cuda:0
```

## Early training

The Modified MADDPG arrays under `hpc/` use the fixed reproduction baseline and
default `policy_only` artifact mode. Submit only the array needed for the
current question. Existing non-complete run directories are not overwritten or
silently resumed.

```bash
sbatch hpc/aoi_modified_maddpg_default_array.sbatch
sbatch hpc/aoi_modified_maddpg_gap_array.sbatch
sbatch hpc/aoi_modified_maddpg_platoon_size_array.sbatch
```

Each completed cell must contain `COMPLETE.json`, `train_metrics.npz`,
`train_metrics_summary.json`, and `policy_final.pt`, with no `checkpoints/`
directory.

## Deferred evaluation workflow

Held-out evaluation is intentionally not part of early exploration. When it is
needed, launch training through the matrix helper, which explicitly requests
`checkpoint_mode=resumable`:

```bash
python scripts/matrix_runner.py --dry-run --stage train --device cuda:0
```

Review the dry-run report and storage budget before using `--execute`. Do not
mix these large resumable runs with lightweight exploratory result roots.
