# HPC launchers

All launchers assume the repository is at
`/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804`, results
are under the sibling `AoI-Reproduction-diagnostics`, and the Python environment is at
`/eeedata/sgxjw2/conda_envs/aoi_cuda`. Override `PROJECT_DIR`, `AOI_ENV_DIR`, or
result-root variables when required.

Current lightweight Algorithm 1 arrays:

- `aoi_modified_maddpg_default_array.sbatch`: P=5, N=4, gap=25, seeds 8–13
- `aoi_modified_maddpg_gap_array.sbatch`: gaps 5/15/35, seeds 8–13
- `aoi_modified_maddpg_platoon_size_array.sbatch`: N=6/8/10, seeds 8–13

They train only, use the fixed `tau=0.005`, slow-update-1, synchronous-global
baseline, and write one `policy_final.pt` without replay or periodic
checkpoints.

The first MAPPO confirmation uses:

- `aoi_mappo_default_array.sbatch`: P=5, N=4, gap=25, seeds 8–13

It is training-only and writes to
`MAPPO_results/default/P5_N4_gap25`. The six cells use separate local actors,
a centralized per-agent critic, five episodes per on-policy rollout, and ten
PPO epochs. The shared config still records `tau=0.005`, but Polyak tau,
external action noise, and global-actor update semantics are explicitly marked
not applicable in MAPPO completion metadata.

The follow-up stability diagnostic uses
`aoi_mappo_stability_array.sbatch` (12 new cells): six seeds with only actor
learning rate reduced to `1e-4`, and six seeds with only all three entropy
coefficients doubled. The original six default cells are reused for comparison.
`analysis/audit_mappo_default_actions.py` performs read-only action/reward
post-processing, while `analysis/summarize_mappo_stability.py` produces the
three-arm comparison.

The minimal combined confirmation uses `aoi_mappo_combined_array.sbatch` (six
new cells, seeds 8--13). It combines actor learning rate `1e-4` with doubled RB,
mode, and power entropy coefficients and writes to
`MAPPO_results/combined-confirm-v1/P5_N4_gap25`. The summarizer accepts
`--combined-root` to compare all four arms while reusing the existing 18 runs.

The frozen-policy held-out diagnostic uses
`aoi_mappo_policy_eval_array.sbatch` (48 lightweight evaluations): four arms,
six training seeds, and deterministic/stochastic policy action selection. Each
cell uses validation seeds 201--206, five warm-up episodes, and 100 scored
episodes. It reads existing `policy_final.pt` files without retraining and
writes only under `MAPPO_results/policy-eval-v1/P5_N4_gap25`.
`analysis/summarize_mappo_policy_eval.py` validates and compares all 48 cells.

The corrected-value-clipping task-decomposition A/B uses two launchers:

- `aoi_mappo_tdec_ab_train_array.sbatch`: 12 trainings, `combined` versus
  `tdec`, paired over seeds 8--13
- `aoi_mappo_tdec_ab_eval_array.sbatch`: 24 frozen-policy evaluations,
  deterministic and stochastic, held-out seeds 201--206

Both variants use the frozen `entropy2x` setting (actor learning rate `5e-4`),
`mappo_value_clip_mode=normalized`, 500 training episodes, and the default
P=5/N=4/gap=25 scenario. Each array task requests one Slurm `l20` GPU and eight
CPUs. Results live under `MAPPO_results/tdec-ab-v1/P5_N4_gap25`, and
`analysis/summarize_mappo_tdec_ab.py` produces the paired report after both
arrays complete.

The pilot, matrix, and audit launchers are retained for a later held-out stage.
They explicitly use resumable checkpoints and refuse to start unless
`AOI_RESULT_ROOT` is set to the dedicated
`Modified_MADDPG_with_TDec_results/heldout-formal-matrix` directory. Create its
`slurm_logs/` directory before `sbatch`. They should not be started during early
algorithm development without reviewing the storage budget and evaluation
protocol.

Before submission:

```bash
test "$(git branch --show-current)" = main
test -z "$(git status --porcelain --untracked-files=all)"
/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python -m pytest -q
/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python scripts/preflight_network.py --scenario p05_n10_g25 --device cuda:0
```
