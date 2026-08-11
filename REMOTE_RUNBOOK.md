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

For the first MAPPO confirmation, create the log directory before submission,
run seed 8 as a pilot, and submit seeds 9--13 only after that cell completes:

```bash
MAPPO_ROOT=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/default/P5_N4_gap25
mkdir -p "$MAPPO_ROOT/slurm_logs"
sbatch --array=0 hpc/aoi_mappo_default_array.sbatch
# after seed 8 passes:
sbatch --array=1-5%5 hpc/aoi_mappo_default_array.sbatch
/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python analysis/summarize_mappo_default.py --result-root "$MAPPO_ROOT"
```

This MAPPO wave is training-only. Do not run held-out evaluation, a formal
matrix, or `final_test` from these policy-only artifacts.

After the first default wave, the compact action/reward audit and two-arm
stability diagnostic are run with:

```bash
BASELINE_ROOT=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/default/P5_N4_gap25
STABILITY_ROOT=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/stability-ablation-v1/P5_N4_gap25
python analysis/audit_mappo_default_actions.py --result-root "$BASELINE_ROOT"
mkdir -p "$STABILITY_ROOT/slurm_logs"
sbatch hpc/aoi_mappo_stability_array.sbatch
python analysis/summarize_mappo_stability.py --stability-root "$STABILITY_ROOT" --baseline-root "$BASELINE_ROOT"
```

This adds 12 trainings; the six-cell default baseline is not rerun.

After reviewing the two single-factor arms, run the six-cell combined
confirmation without rerunning the existing 18 cells:

```bash
COMBINED_ROOT=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/combined-confirm-v1/P5_N4_gap25
mkdir -p "$COMBINED_ROOT/slurm_logs"
sbatch --array=0 hpc/aoi_mappo_combined_array.sbatch
# after seed 8 passes:
sbatch --array=1-5%5 hpc/aoi_mappo_combined_array.sbatch
python analysis/summarize_mappo_stability.py --baseline-root "$BASELINE_ROOT" --stability-root "$STABILITY_ROOT" --combined-root "$COMBINED_ROOT"
```

This stage remains training-only and adds no new scenario variable.

After all four training arms are complete, evaluate their frozen final policies
on the held-out validation split:

```bash
POLICY_EVAL_ROOT=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/MAPPO_results/policy-eval-v1/P5_N4_gap25
mkdir -p "$POLICY_EVAL_ROOT/slurm_logs"
sbatch --array=0,1 hpc/aoi_mappo_policy_eval_array.sbatch
# after the deterministic/stochastic seed-8 pilots pass:
sbatch --array=2-47%8 hpc/aoi_mappo_policy_eval_array.sbatch
python analysis/summarize_mappo_policy_eval.py --result-root "$POLICY_EVAL_ROOT"
```

This is diagnostic evaluation only. It uses the actors in `policy_final.pt`,
does not add external action noise, and does not create validation/final-release
lifecycle markers.

## Deferred evaluation workflow

Held-out evaluation is intentionally not part of early exploration. When it is
needed, launch training through the matrix helper, which explicitly requests
`checkpoint_mode=resumable`:

```bash
python scripts/matrix_runner.py --dry-run --stage train --device cuda:0
```

Review the dry-run report and storage budget before using `--execute`. Do not
mix these large resumable runs with lightweight exploratory result roots.
