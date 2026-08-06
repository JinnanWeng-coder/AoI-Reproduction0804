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

### Flexible GPU Job Array alternative

When all eight L20 GPUs are not simultaneously available, use the array driver
instead of the full-node driver.  It exposes the same 48 formal cells as 48
independent Slurm array tasks.  Every task requests four CPUs and one L20 GPU;
`%8` is only a concurrency ceiling, so Slurm may start fewer tasks when fewer
GPUs are free.

Do not submit `aoi_matrix_8gpu.sbatch` and `aoi_matrix_array.sbatch` for the
same stage at the same time.  Both drivers target the same immutable run
directories.  The array task ID maps to exactly one deterministic matrix cell,
and completed cells are provenance-checked and skipped.

Default maximum concurrency of eight:

```bash
array_train_job=$(sbatch --parsable \
  --export=ALL,AOI_STAGE=train hpc/aoi_matrix_array.sbatch)
echo "$array_train_job"
```

To cap concurrency below eight, override the array expression when submitting.
For example, use at most four GPUs:

```bash
array_train_job=$(sbatch --parsable --array=0-47%4 \
  --export=ALL,AOI_STAGE=train hpc/aoi_matrix_array.sbatch)
echo "$array_train_job"
```

Monitor the parent array and individual tasks with:

```bash
squeue -j "$array_train_job"
sacct -j "$array_train_job" --format=JobID,JobName,State,Elapsed,ExitCode%12
tail -f "/eeedata/sgxjw2/AoI-Reproduction0804-results/slurm_logs/aoi_matrix_array_${array_train_job}_0.out"
```

After all training array tasks and the CPU training audit pass, create
`TRAIN_APPROVED`.  Validation can then use the same array driver:

```bash
array_eval_job=$(sbatch --parsable --array=0-47%4 \
  --export=ALL,AOI_STAGE=eval hpc/aoi_matrix_array.sbatch)
echo "$array_eval_job"
```

The existing CPU audits are independent of which matrix driver created the
artifacts; they still require exactly 48 valid run directories.  Resubmitting
the same full array is safe after scheduler interruption because each task
validates and skips completed cells or resumes a matching checkpoint.

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

## Small causal diagnostic array (not the formal matrix)

`aoi_diagnostic_array.sbatch` isolates the suspected actor-gradient failure in
one scenario. It trains seeds 3, 5 and 7 for three arms: the current synchronous
global update, the same paper profile with only the global actor gradient
detached, and the exact `legacy_release` profile. Its result root is
`/eeedata/sgxjw2/AoI-Reproduction-diagnostics/global-causal-v1`; it never uses
the formal 48-cell result root. Training enables the gradient/action diagnostics
and selects a real `best.pt` online using the dedicated selection seeds 301 and
302; the noise sweep then uses held-out validation seeds 201..206.

Create the external Slurm log directory before submitting. The train stage has
9 tasks. The eval stage has 36 tasks because each trained cell is evaluated at
noise 0, 0.05, 0.1 and 0.3 on validation seeds 201..206:

```bash
mkdir -p /eeedata/sgxjw2/AoI-Reproduction-diagnostics/global-causal-v1/slurm_logs

diag_train_job=$(sbatch --parsable --array=0-8%8 \
  --export=ALL,AOI_STAGE=train hpc/aoi_diagnostic_array.sbatch)

diag_eval_job=$(sbatch --parsable --dependency=afterok:"$diag_train_job" \
  --array=0-35%8 --export=ALL,AOI_STAGE=eval \
  hpc/aoi_diagnostic_array.sbatch)
```

Evaluation uses the selection-validation winner in `best.pt`. Resubmission skips
completed training and evaluation cells and resumes incomplete training from
`latest.pt`. This diagnostic is evidence for deciding the next code change; it
is not a replacement for the formal matrix and never runs `final_test`.

## Tau x slow-update recovery diagnostic

`aoi_tau_slow_array.sbatch` runs the next small causal experiment entirely
under the current `paper_faithful` environment and synchronous-global learner.
It varies only `tau` (`0.0005`, `0.005`) and
`slow_update_every_episodes` (`1`, `20`) for training seeds 3, 5 and 7 in the
single `p05_n04_g25` scenario. The isolated result root is
`/eeedata/sgxjw2/AoI-Reproduction-diagnostics/tau-slow-v1`.

The 12-cell training array writes gradient/action diagnostics. The dependent
48-task validation array evaluates both the selection winner (`best.pt`) and
the episode-500 policy (`latest.pt`) at noise 0 and 0.3, using held-out seeds
201..206. These full checkpoints contain the exact selected and episode-500
actors, so no separate actor-snapshot evaluator or checkpoint download is
needed. The eval array explicitly uses `--diagnostic-eval`, so its noise-zero
evaluations do not create the single `VALIDATION_READY.json` lifecycle marker.
This allows the two checkpoints to coexist without changing ordinary
validation or training-instrumentation semantics.

```bash
mkdir -p /eeedata/sgxjw2/AoI-Reproduction-diagnostics/tau-slow-v1/slurm_logs

tau_slow_train_job=$(sbatch --parsable --array=0-11%8 \
  --export=ALL,AOI_STAGE=train hpc/aoi_tau_slow_array.sbatch)

tau_slow_eval_job=$(sbatch --parsable \
  --dependency=afterok:"$tau_slow_train_job" --array=0-47%8 \
  --export=ALL,AOI_STAGE=eval hpc/aoi_tau_slow_array.sbatch)
```

Do not submit the formal matrix or `final_test` for this diagnostic. A repeated
submission resumes incomplete training and skips already completed matching
validation tasks.
