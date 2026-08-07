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

## Tau 0.005 six-seed confirmation

`aoi_tau005_confirm_array.sbatch` is the bounded follow-up stability check. It
keeps the `paper_faithful` environment, `p05_n04_g25`, synchronous global actor,
500 episodes, fixed training noise 0.3 and `slow_update_every_episodes=1`. The
only investigated setting is `tau=0.005`, trained with seeds 2..7 under the
isolated result root
`/eeedata/sgxjw2/AoI-Reproduction-diagnostics/tau005-confirm-v1`.

The train stage has six tasks. The dependent 24-task eval stage runs
`best.pt` and `latest.pt` at noise 0 and 0.3 on held-out validation seeds
201..206, with 100 scored episodes and the existing warm-up protocol:

```bash
mkdir -p /eeedata/sgxjw2/AoI-Reproduction-diagnostics/tau005-confirm-v1/slurm_logs

tau005_train_job=$(sbatch --parsable --array=0-5%6 \
  --export=ALL,AOI_STAGE=train hpc/aoi_tau005_confirm_array.sbatch)

tau005_eval_job=$(sbatch --parsable \
  --dependency=afterok:"$tau005_train_job" --array=0-23%8 \
  --export=ALL,AOI_STAGE=eval hpc/aoi_tau005_confirm_array.sbatch)
```

This is a diagnostic confirmation, not the formal matrix. Do not run
`final_test`, add another experiment variable, or treat a partial result as a
paper-wide reproduction.

## Tau 0.005 gap-trend pilot

`aoi_gap_trend_array.sbatch` is the smallest follow-up that tests the Fig. 5
intra-platoon-gap direction without rerunning the completed 25 m anchor. It
adds only `p05_n04_g05` and `p05_n04_g35`; the six `p05_n04_g25` runs and their
noise-0.3 evaluations are reused from `tau005-confirm-v1` during analysis.

Both new scenarios keep `paper_faithful`, `tau=0.005`, synchronous global
actor updates, `slow_update_every_episodes=1`, 500 episodes, fixed training
noise 0.3 and seeds 2..7. The training stage therefore has 12 tasks. The
dependent 24-task validation stage evaluates `best.pt` and `latest.pt` only at
noise 0.3 on held-out seeds 201..206 with 100 scored episodes and the existing
warm-up protocol. It deliberately does not repeat the noise-zero robustness
test.

```bash
mkdir -p /eeedata/sgxjw2/AoI-Reproduction-diagnostics/gap-trend-v1/slurm_logs

gap_train_job=$(sbatch --parsable --array=0-11%8 \
  --export=ALL,AOI_STAGE=train hpc/aoi_gap_trend_array.sbatch)

gap_eval_job=$(sbatch --parsable \
  --dependency=afterok:"$gap_train_job" --array=0-23%8 \
  --export=ALL,AOI_STAGE=eval hpc/aoi_gap_trend_array.sbatch)
```

After both stages complete, combine the new 5 m and 35 m artifacts with the
existing 25 m anchor. Report final-100 training and held-out evaluation values
for AoI, strict binary endpoint CAM and continuous endpoint payload completion;
the continuous audit metric must not replace or relabel the released binary
CAM metric. This pilot is not the formal matrix, does not run `final_test`, and
does not establish the paper-wide comparison against other algorithms.

## Gap 15 source-protocol fill

`aoi_gap15_fill_array.sbatch` adds the one missing Fig. 5(a)/(b) gap point
without changing the learner or rerunning the completed 5 m, 25 m, and 35 m
cells. It trains only `p05_n04_g15` with seeds 2..7, `paper_faithful`,
`tau=0.005`, synchronous global actor updates,
`slow_update_every_episodes=1`, 500 episodes, and the released fixed training
noise 0.3. The primary result is the final 100 training episodes; there is no
held-out eval array and no best/latest checkpoint gate in this bounded run.

```bash
mkdir -p /eeedata/sgxjw2/AoI-Reproduction-diagnostics/gap15-fill-v1/slurm_logs

gap15_train_job=$(sbatch --parsable --array=0-5%6 \
  hpc/aoi_gap15_fill_array.sbatch)
```

After all six cells complete, combine their final-100 training summaries with
the existing gap-trend evidence for 5 m, 25 m, and 35 m. Keep mean AoI, strict
binary endpoint CAM, and continuous endpoint payload completion separate. This
run does not submit validation, `formal matrix`, or `final_test` jobs.

## Seeds 8--13 gap and gap-25 mechanism check (42 training cells)

`aoi_gap_global_slow_42_array.sbatch` extends the training-only evidence with
seeds 8..13 while keeping `paper_faithful`, `tau=0.005`, 500 episodes, and the
released fixed training noise 0.3. It has two non-overlapping phases under the
isolated result root
`/eeedata/sgxjw2/AoI-Reproduction-diagnostics/gap-global-slow-42-v1`:

- Phase A is the paper-facing arm: four gaps (5, 15, 25, and 35 m),
  synchronous global actor, and `slow_update_every_episodes=1`. It contains
  `4 gaps x 6 seeds = 24` training cells.
- Phase B is a mechanism check at the default 25 m anchor only. It adds the
  other three members of the 2 x 2 global-mode/slow-update design:
  synchronous/20, detached/1, and detached/20. The synchronous/1 baseline is
  already present in Phase A, so Phase B contains `3 arms x 6 seeds = 18`
  additional training cells.

The union is therefore exactly `24 + 18 = 42` unique training cells. Submit
Phase B after Phase A so that completion accounting and the shared gap-25
baseline are easy to audit:

```bash
mkdir -p /eeedata/sgxjw2/AoI-Reproduction-diagnostics/gap-global-slow-42-v1/slurm_logs

phase_a_job=$(sbatch --parsable --array=0-23%8 \
  --export=ALL,AOI_PHASE=A hpc/aoi_gap_global_slow_42_array.sbatch)

phase_b_job=$(sbatch --parsable --dependency=afterok:"$phase_a_job" \
  --array=0-17%8 --export=ALL,AOI_PHASE=B \
  hpc/aoi_gap_global_slow_42_array.sbatch)
```

This round is training-only. Use the final 100 episodes as the primary window
and the final 50 as a sensitivity check; keep mean/worst-agent AoI, strict
binary endpoint CAM, and continuous payload completion separate. Phase B can
support conclusions about the mechanism at gap 25 only. Do not submit an eval
array, the formal matrix, or `final_test`.

## Fig. 5(c)/(d) platoon-size trend (18 new training cells)

`aoi_platoon_size_trend_array.sbatch` extends the paper-facing
`paper_faithful`, `tau=0.005`, synchronous-global, slow-update-1 configuration
to platoon sizes 6, 8, and 10 at `P=5` and gap 25 m. It uses training seeds
8..13, 500 episodes, and the released fixed training noise 0.3. The completed
size-4 baseline for the same six seeds is reused from Phase A of
`gap-global-slow-42-v1`; it is not trained again.

The new array therefore contains exactly
`3 new platoon sizes x 6 seeds = 18` training cells under the isolated result
root `/eeedata/sgxjw2/AoI-Reproduction-diagnostics/platoon-size-trend-v1`:

```bash
mkdir -p /eeedata/sgxjw2/AoI-Reproduction-diagnostics/platoon-size-trend-v1/slurm_logs

size_train_job=$(sbatch --parsable --array=0-17%8 \
  hpc/aoi_platoon_size_trend_array.sbatch)
```

This round is training-only. Combine the 18 new cells with the six existing
`p05_n04_g25` baseline runs only after checking the science configuration of
both roots. Use the final 100 episodes as primary and the final 50 as a
sensitivity window. Treat seed as the paired independent unit across sizes;
report every seed, mean and standard deviation, median and interquartile range,
leave-one-seed-out sensitivity, and collapsed-run counts. Keep strict binary
endpoint CAM and continuous payload completion separate. Do not select a
favorable seed subset, submit an eval array, run the formal matrix, or run
`final_test`.

## Modified MADDPG Algorithm 1 default confirmation

`aoi_modified_maddpg_default_array.sbatch` runs the first controlled
Algorithm 1 comparison. The environment and synchronized global actor update
remain the same as the repaired TDec implementation; only the two task critics
are replaced by one holistic local critic trained on `reward_task1 +
reward_task2`. The fixed configuration is `P=5`, `N=4`, gap 25 m,
`tau=0.005`, `slow_update_every_episodes=1`, 500 episodes, fixed training
noise 0.3, and seeds 8..13.

The six training cells are stored under
`/eeedata/sgxjw2/AoI-Reproduction-diagnostics/Modified_MADDPG_results/default/P5_N4_gap25`:

```bash
result_root=/eeedata/sgxjw2/AoI-Reproduction-diagnostics/Modified_MADDPG_results/default/P5_N4_gap25
mkdir -p "$result_root/slurm_logs"

modified_job=$(sbatch --parsable --array=0-5%6 \
  hpc/aoi_modified_maddpg_default_array.sbatch)
```

After all six tasks finish, run the bounded training-only summarizer:

```bash
python analysis/summarize_modified_maddpg_default.py \
  --result-root "$result_root"
```

Use the final 100 training episodes as primary and the final 50 as a
sensitivity window. Keep strict binary endpoint CAM and continuous payload
completion separate. There is no eval array, formal matrix, or `final_test` in
this round. Future gap and platoon-size experiments belong in sibling
subdirectories `gap-extension/` and `platoon-size-extension/` under
`Modified_MADDPG_results/`.
