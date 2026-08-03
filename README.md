# Modified MADDPG with TDec reproduction

This directory is the reproducible implementation workspace for Algorithm 1.
The original source is preserved under `legacy_reference/` and its provenance is
recorded in `SOURCE_MANIFEST.json`. The source Git clone at
`src/AoI-V2X-IEEE-TVT-2023-reimplement` is never modified by this project.

The requested `src/...-main` path was absent after the source repository was
created; the clean clone at commit `974e5f8` is byte-identical for the source
files and the path mapping is recorded in the manifest.

The implementation is split into `legacy_release` and `paper_faithful` profiles.
Smoke runs are written below `scratch/`; formal runs are written below
`experiments/runs/` and are never overwritten.

## Environments

Use a Python environment with a suitable CPU or CUDA PyTorch build. The local
validation used `aoi_v2x` (Python 3.9, CPU torch) and `aoi_cuda` (Python 3.10,
torch `2.11.0+cu126`, one available CUDA device).

```text
python -m pip install -r requirements.txt
```

## CLI and dry-run

```text
python Main.py --profile paper_faithful --scenario p05_n04_g25 --dry-run
python Main.py --profile paper_faithful --dry-run --matrix
```

The first command prints the resolved config, state/action dimensions, and safe
run path without creating output. The second prints exactly 48 unique
`8 scenarios × seeds 2..7` tasks.

## Smoke, resume, eval, and audit

```text
python Main.py --profile paper_faithful --scenario p05_n04_g25 \
  --seed 2 --device cpu --smoke --run-name smoke_paper
python Main.py --profile paper_faithful --device cpu \
  --eval-only --eval-episodes 100 --eval-seeds 102,103,104 \
  --resume scratch/smoke_paper/checkpoints/latest.pt
python analysis/audit_results.py scratch/smoke_paper
python -m analysis.plot_training scratch/smoke_paper
```

Smoke output is always placed under `scratch/` and marked
`is_formal_result=false`. Existing run and eval directories are rejected; only
an explicit `--resume` can continue an incomplete run. `latest.pt` contains
networks, optimizers, replay, environment, metrics, and Python/NumPy/PyTorch RNG
state. `train_metrics.npz` keeps separate task1/task2 arrays, the combined
`local_total_episode_mean`, `global_episode_sum`, and
`training_objective_proxy`; plotting reads these files only.

The restart-safe matrix entry points are:

```text
scripts/run_paper_matrix.sh --dry-run
powershell -File scripts/run_paper_matrix.ps1 -DryRun
```

Use `--execute` only on the remote machine after the formal environment and
storage policy have been confirmed. This code-completion stage does not execute
that matrix.

## Profiles

`legacy_release` preserves the public source behavior, including the detached
global actor term and old environment cadence, for compatibility tracing.
`paper_faithful` is the formal default: continuous `[1,30]` dBm power, full
`750×1299` geometry, centered RSU, per-RB previous interference, remaining time,
current-interference reward, and one synchronized joint actor update with
`global_actor_weight=1.0`.

Formal 500-episode training and the 48-run matrix are intentionally not launched
by this code-completion stage.
