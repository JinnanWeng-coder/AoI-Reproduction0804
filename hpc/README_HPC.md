# UNNC L20 HPC execution

These scripts follow the local `Q10`/L20 Slurm examples and the school HPC
guide. They are intentionally staged: environment, one formal pilot, full
training, training audit, validation, and validation audit. Never run
`final_test` from this workflow.

## Resource model

One `(scenario, training seed)` cell is a single-process, single-GPU job. The
code does not implement distributed data parallelism, so assigning eight GPUs
to one cell would waste seven GPUs. The complete experiment has 48 cells (eight
scenarios x seeds 2..7). `aoi_matrix_8gpu.sbatch` reserves one eight-L20 node
and launches eight exclusive Slurm steps. Each deterministic shard contains six
non-overlapping cells; all shards together cover the 48-cell matrix exactly.

## 1. Put the clean Git checkout on HPC

Preferred, if the login node can reach GitHub:

```bash
ssh sgxjw2@10.179.1.200
mkdir -p /eeedata/sgxjw2
cd /eeedata/sgxjw2
git clone --branch main --single-branch \
  https://github.com/JinnanWeng-coder/AoI-Reproduction0804.git
cd AoI-Reproduction0804
test "$(git branch --show-current)" = main
test -z "$(git status --porcelain --untracked-files=all)"
```

Do not upload only loose source files: formal artifacts require a clean branch,
commit, tracked-tree digest, and `SOURCE_MANIFEST.json`. If GitHub is blocked,
create a Git bundle locally, upload the bundle with MobaXterm/SCP, and clone from
the bundle on HPC so `.git` provenance is preserved.

## 2. Prepare paths and environment

The Slurm log directory must exist before `sbatch` because Slurm opens its
output files before the script starts:

```bash
mkdir -p /eeedata/sgxjw2/AoI-Reproduction0804-results/slurm_logs
cd /eeedata/sgxjw2/AoI-Reproduction0804
bash hpc/setup_aoi_cuda.sh
```

All user-managed work stays below `/eeedata/sgxjw2`: the default Miniconda
root is `/eeedata/sgxjw2/miniconda3`, the environment is
`/eeedata/sgxjw2/conda_envs/aoi_cuda`, and package/cache/temp directories are
also under that base. If Conda is absent, let Grok inspect `module av` for an
Anaconda command that can create the prefix environment, or install Miniconda
at the default root, then rerun setup. The validated environment is Python
3.10.20, PyTorch 2.11.0 with CUDA 12.6 wheels, and
`requirements.cuda.lock.txt`.

## 3. Formal pilot

```bash
pilot_job=$(sbatch --parsable hpc/aoi_pilot_1gpu.sbatch)
echo "$pilot_job"
squeue -j "$pilot_job"
tail -f "/eeedata/sgxjw2/AoI-Reproduction0804-results/slurm_logs/aoi_pilot_s02_${pilot_job}.out"
```

The pilot runs the full paper configuration for scenario `p05_n04_g25`, seed
2, followed by held-out validation and audit. Grok must review the artifacts,
finite metrics, learning behavior, GPU/runtime logs, and audit result. Only when
the pilot is acceptable should Grok create the external approval gate:

```bash
touch /eeedata/sgxjw2/AoI-Reproduction0804-results/handoff/PILOT_APPROVED
```

## 4. Full 8-GPU training

```bash
train_job=$(sbatch --parsable --export=ALL,AOI_STAGE=train hpc/aoi_matrix_8gpu.sbatch)
echo "$train_job"
squeue -j "$train_job"
```

If Slurm preempts the job or the seven-day limit interrupts it, submit the exact
same command again. Completed cells are skipped and matching incomplete
checkpoints are resumed. Never delete or overwrite a conflicting run directory.

After the job completes, audit all training artifacts:

```bash
audit_job=$(sbatch --parsable --export=ALL,AOI_AUDIT_SCOPE=train hpc/aoi_audit_cpu.sbatch)
echo "$audit_job"
```

After Grok confirms all eight shard reports and all 48 audits are complete,
finite, and provenance-clean:

```bash
touch /eeedata/sgxjw2/AoI-Reproduction0804-results/handoff/TRAIN_APPROVED
```

## 5. Validation and audit

```bash
eval_job=$(sbatch --parsable --export=ALL,AOI_STAGE=eval hpc/aoi_matrix_8gpu.sbatch)
echo "$eval_job"

validation_audit_job=$(sbatch --parsable \
  --export=ALL,AOI_AUDIT_SCOPE=validation hpc/aoi_audit_cpu.sbatch)
echo "$validation_audit_job"
```

Submit the validation audit only after the evaluation job is complete. Grok
should then build the validation study manifest and partial figures following
`REMOTE_RUNBOOK.md`. Validation is not `final_test`, and Algorithm 2 alone
cannot establish the paper's four-algorithm comparison conclusions.

## Monitoring

```bash
sinfo
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode,AllocTRES%80
tail -f /eeedata/sgxjw2/AoI-Reproduction0804-results/slurm_logs/JOB_LOG.out
scancel JOB_ID
```

Grok may autonomously handle queue waits, transient scheduler failures,
preemption, exact-command resubmission, log collection, and artifact audits. It
must not change scientific parameters, edit tracked files after artifacts
exist, run `final_test`, delete conflicting runs, or label partial figures as
complete paper reproduction.
