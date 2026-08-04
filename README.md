# Modified MADDPG with TDec reproduction

The leading `1-` is part of the research-task directory name. `TDec` denotes
the task-decomposition branch corresponding to paper Algorithm 2. This
directory is the Modified MADDPG with TDec reproduction target; it does not
implement a separate "Algorithm 1 training orchestration".

This directory is the reproducible implementation workspace. The original
source is preserved under `legacy_reference/` and its provenance is recorded
in `SOURCE_MANIFEST.json`. The source Git clone at
`src/AoI-V2X-IEEE-TVT-2023-reimplement` is never modified by this project.

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

See `ENVIRONMENT_INSTALL.md` and `requirements.lock.txt`. Validation used
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
incomplete run. `latest.pt` contains networks, optimizers, replay, environment,
metrics, and Python/NumPy/PyTorch RNG state. `train_metrics.npz` keeps separate
task1/task2 arrays, `local_total_episode_mean`, `global_episode_sum`,
`global_episode_mean`, and `immediate_reward_proxy`. The latter is an immediate
reward aggregation for plotting, not a differentiable actor objective.

Evaluation uses one `reset_world(eval_seed)` per held-out seed, five sequential
warm-up episodes by default, then sequential scored episodes. `validation`
uses 201..206; `final_test` uses 101..106. Raw arrays retain
eval-seed x scored-episode x slot x agent. Within-seed episode SD is
descriptive only; inferential mean/SD/95% CI is computed across independent
training seeds. The CLI requires `--scope validation --eval-purpose validation`
for pilot/validation evaluation. `final_test` is reserved for a formal
checkpoint and requires `--scope final_release`, seeds 101..106, warm-up 5,
and 100 scored episodes. A training command always uses `--scope train` and
does not accept an evaluation purpose. Validation and final-test artifacts
cannot be mixed.

The restart-safe matrix entry points are:

```text
scripts/run_paper_matrix.sh --dry-run
powershell -File scripts/run_paper_matrix.ps1 -DryRun
```

`matrix_runner.py --execute` defaults to train-only. Use
`--stage all --eval-purpose validation` explicitly for train -> validation
eval -> validation audit; use `--eval-purpose final_test` only for the formal
final-release lane. Existing directories are classified with structured
recovery states and are never deleted or overwritten.

Use `--execute` only on the remote machine after the formal environment and
storage policy have been confirmed. This code-completion stage does not execute
that matrix.

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
Fig.5 can produce an explicitly labelled current-algorithm `PARTIAL` output
while the baselines are unavailable.

Formal audit applies hard gates for paper_faithful_v4: 500 episodes, 100 slots,
the resolved full network, training seeds 2..7, final-test seeds 101..106,
warmup 5, scored episodes 100, endpoint-demand consistency, clean Git
provenance, and one unique final-test artifact per training run.

Fig.5 uses only the controlled gap grid
`p05_n04_g05/g15/g25/g35` or size grid
`p05_n04_g25/p05_n06_g25/p05_n08_g25/p05_n10_g25`. Validation may produce a
labelled `PARTIAL` current-algorithm figure; a complete final Fig.5 requires
Modified_MADDPG_with_TDec, Modified_MADDPG, MADDPG_FDec, and DDPG for every
scenario and training seed 2..7. The three baselines are not implemented in
this round.

Formal 500-episode training, the final release, and the 48-run matrix are
intentionally not launched by this code-completion stage.
