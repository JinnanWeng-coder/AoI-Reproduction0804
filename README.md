# Modified MADDPG with TDec reproduction

The leading `1-` is part of the research-task directory name. `TDec` denotes
the task-decomposition branch corresponding to paper Algorithm 2. This
directory is the Modified MADDPG with TDec reproduction target; it does not
implement a separate "Algorithm 1 training orchestration".

This directory is the reproducible implementation workspace. The original
source is preserved under `legacy_reference/` and its provenance is recorded
in `SOURCE_MANIFEST.json`. The source Git clone used during implementation was
never modified. A remote experiment clone needs only this repository; the
absolute Windows paths retained in the source manifest are historical records.

The requested `src/...-main` path was absent after the source repository was
created; the clean clone at commit `974e5f8` is byte-identical for the source
files and the path mapping is recorded in the manifest.

The implementation is split into `legacy_release` and `paper_faithful` profiles.
Smoke runs are written below `scratch/`; formal runs are written below
`experiments/runs/` and are never overwritten.

`paper_faithful` artifacts use semantic version `paper_faithful_v4`, mobility
revision `lane_graph_exit_safe_v1`, and checkpoint schema `checkpoint_v4`.
Paper-faithful v1/v2/v3 checkpoints are rejected; `legacy_release_v1` retains
its compatibility loader, including historical pre-semantic legacy
checkpoints. The v4 mobility update consumes the exact velocity-times-slow
interval distance along legal lane-graph segments, including the explicit
up->right, down->left, left->up, and right->down exits.

## Environments

See `ENVIRONMENT_INSTALL.md`, `REMOTE_RUNBOOK.md`, and the CPU/CUDA top-level
requirement pins. Validation used
`aoi_v2x` (Python 3.9, CPU torch) and `aoi_cuda` (Python 3.10,
torch `2.11.0+cu126`, one available CUDA device).

## CLI and dry-run

```text
python Main.py --profile paper_faithful --scenario p05_n04_g25 --dry-run
python Main.py --profile paper_faithful --dry-run --matrix
```

The first command prints the resolved config, state/action dimensions, and safe
run path without creating output. The second prints exactly 48 unique
`8 scenarios x seeds 2..7` tasks.

## Smoke, resume, eval, and audit

```text
python Main.py --profile paper_faithful --scenario p05_n04_g25 \
  --seed 2 --device cpu --smoke --run-name smoke_paper
python Main.py --profile paper_faithful --device cpu \
  --eval-only --scope validation --eval-purpose validation --eval-episodes 2 \
  --eval-seeds 201,202 \
  --resume scratch/smoke_paper/checkpoints/latest.pt
python analysis/audit_results.py scratch/smoke_paper --scope validation --require-eval
python -m analysis.plot_training scratch/smoke_paper
python analysis/study_manifest.py experiments/runs --output study_manifest.json
python analysis/build_paper_figures.py study_manifest.json --figure 3
```

Smoke output is marked `is_formal_result=false`. Existing run and eval
directories are rejected; only an explicit `--resume` can continue an
incomplete run. Evaluation accepts only a completed `latest.pt` or a
selection-validation `best.pt` bound to `COMPLETE.json`; `latest.pt` is the
final episode, while `best.pt` may intentionally come from an earlier
checkpoint. `latest.pt`
contains networks, optimizers, replay, environment,
metrics, and Python/NumPy/PyTorch RNG state. `train_metrics.npz` keeps separate
task1/task2 arrays, `local_total_episode_mean`, `global_episode_sum`,
`global_episode_mean`, and `immediate_reward_proxy`. The latter is an immediate
reward aggregation for plotting, not a differentiable actor objective.
Generated metric tensors are stored once as compressed NPZ rather than also
duplicated as MAT files.

Evaluation uses one `reset_world(eval_seed)` per held-out seed, five sequential
warm-up episodes by default, then sequential scored episodes. `validation`
uses 201..206; `final_test` uses 101..106. Raw arrays retain
eval-seed x scored-episode x slot x agent. Within-seed episode SD is
descriptive only; inferential mean/SD/95% CI is computed across independent
training seeds. The CLI requires `--scope validation --eval-purpose validation`
for pilot/validation evaluation. `final_test` is reserved for a formal
checkpoint and requires `--scope final_release`, seeds 101..106, warm-up 5,
and 100 scored episodes. A training command always uses `--scope train` and
does not accept an evaluation purpose. Each evaluation artifact has exactly
one purpose; a study manifest may contain both purposes and filters them
explicitly during analysis.

The restart-safe matrix entry points are:

```text
CHECKPOINT_EVERY=5 scripts/run_paper_matrix.sh --dry-run --device cuda:0 --recover-empty-run
powershell -File scripts/run_paper_matrix.ps1 -DryRun -Device cuda:0 -CheckpointEvery 5 -RecoverEmptyRun
```

`matrix_runner.py --execute` defaults to train-only, a five-episode checkpoint
cadence, and no empty-run recovery unless `--recover-empty-run` is explicit. Use
`--stage all --eval-purpose validation` explicitly for train -> validation
eval -> validation audit; use `--eval-purpose final_test` only for the formal
final-release lane. Existing directories are classified with structured
recovery states and are never deleted or overwritten. A new run is published
from a fully written sibling staging directory with a no-replace atomic rename.
The explicit empty-run path accepts only the exact, provenance-verified
initialization whitelist; ordinary incomplete checkpoints continue through
`--resume`. Completed runs are skipped only after both final checkpoints and
their hashes match `COMPLETE.json`.

Use `--execute` only on the remote machine after the formal environment and
storage policy have been confirmed. Run one formal 500 x 100 training cell and
its held-out validation first, then review it before the sequential 48-run
matrix. Exact Linux commands are in `REMOTE_RUNBOOK.md`.

For the UNNC `Q10` eight-L20 workflow, start with `hpc/README_HPC.md`; the
ready-to-paste Cursor/Grok operating prompt is in `hpc/GROK_HANDOFF.md`.

## Profiles

`legacy_release` preserves the public source behavior, including the detached
global actor term and old environment cadence, for compatibility tracing. Its
adapter also exposes unified `rb`, `mode`, and `power_dbm` info fields.

`paper_faithful` is the formal default: continuous `[1,30]` dBm power, full
`750 x 1299` geometry, centered RSU, per-RB previous interference, remaining time,
current-interference reward, correlated urban-grid mobility, and one
synchronized joint actor update with `global_actor_weight=1.0`.

Fig.4 requires declared baseline artifacts. If they are missing, the figure
command exits nonzero and writes `INCOMPLETE_BASELINES.json`; this prevents
silently presenting the current Algorithm2/TDec curve as a complete comparison.
The required baselines are `Modified_MADDPG`, `MADDPG_FDec`, and `DDPG`;
`DQN` is not a paper Fig.4 baseline. The default Fig.4 stores and draws task1,
task2, global, and combined metrics; raw panel data are saved beside the PNG.
Its complete grid is four algorithms x the selected scenario x training seeds
2..7. Validation and final-test rows from the same training run are deduplicated
by `run_path`; the same cell pointing to different runs is an error. Fig.5 can
produce an explicitly labelled current-algorithm `PARTIAL` validation output
while the baselines are unavailable, and a manifest holding both lifecycle
purposes can be filtered without mixing them.

Formal audit applies hard gates for paper_faithful_v4: 500 episodes, 100 slots,
the resolved full network, training seeds 2..7, final-test seeds 101..106,
warmup 5, scored episodes 100, endpoint-demand consistency, clean Git
provenance, and one unique final-test artifact per training run. It also binds
the final checkpoint hashes and recomputes released AoI/CAM per-agent,
per-evaluation-seed, SD, and overall statistics directly from `metrics.npz`.
`summary.json` and `EVAL_COMPLETE.json` must be identical.

Fig.5 uses only the controlled gap grid
`p05_n04_g05/g15/g25/g35` or size grid
`p05_n04_g25/p05_n06_g25/p05_n08_g25/p05_n10_g25`. Validation may produce a
labelled `PARTIAL` current-algorithm figure; a complete final Fig.5 requires
Modified_MADDPG_with_TDec, Modified_MADDPG, MADDPG_FDec, and DDPG for every
scenario and training seed 2..7. The three baselines are not implemented in
this round.

This repository is ready for a staged remote Algorithm 2 validation pilot and,
after that pilot is reviewed, the Algorithm 2 48-run validation grid. It is not
yet a complete reproduction of the paper's cross-algorithm conclusions: the
three comparison algorithms and their formal artifacts remain future work.
