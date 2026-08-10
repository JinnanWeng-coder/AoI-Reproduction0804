# Remote handoff

Canonical remote layout:

```text
/eeedata/sgxjw2/Parvini-TVT2023-reproduction/
  AoI-Reproduction0804/            code repository
  AoI-Reproduction-diagnostics/    experiment evidence and comparisons
```

The Python environment remains outside the project parent at
`/eeedata/sgxjw2/conda_envs/aoi_cuda`.

Before any run:

```bash
cd /eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction0804
git checkout main
git pull --ff-only origin main
test -z "$(git status --porcelain --untracked-files=all)"
/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python -m pytest -q
```

The three Algorithm 1 arrays use fixed result locations below
`AoI-Reproduction-diagnostics/Modified_MADDPG_results` and default to the
lightweight `policy_only` artifact mode.

Pilot/matrix/audit scripts are deferred held-out tools. Before submitting one,
create the log directory and explicitly export:

```bash
export AOI_RESULT_ROOT=/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/Modified_MADDPG_with_TDec_results/heldout-formal-matrix
mkdir -p "$AOI_RESULT_ROOT/slurm_logs"
```

Do not run the formal matrix or final test unless the current task explicitly
requests it. Never write new results into the archived pre-tau005 matrix.
