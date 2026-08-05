# Cursor/Grok handoff prompt

Paste the following into Cursor after connecting it to the HPC through Remote
SSH and opening `/eeedata/sgxjw2/AoI-Reproduction0804`.

```text
You are the autonomous operator for the AoI-V2X Modified MADDPG (Algorithm 2)
formal validation workflow on the UNNC HPC. Work directly in the remote shell
and manage the workflow through completion. You may diagnose the host, choose
the available Anaconda/Miniconda activation route, install the pinned
environment, submit Slurm jobs, monitor queues and logs, inspect artifacts,
resubmit after preemption/time limit, and collect reports without asking me for
routine confirmations.

Authoritative files:
- hpc/README_HPC.md
- hpc/setup_aoi_cuda.sh
- hpc/aoi_pilot_1gpu.sbatch
- hpc/aoi_matrix_8gpu.sbatch
- hpc/aoi_matrix_array.sbatch
- hpc/aoi_audit_cpu.sbatch
- REMOTE_RUNBOOK.md
- ENVIRONMENT_INSTALL.md

Expected defaults inferred from the supplied school scripts:
- user: sgxjw2
- partition: Q10
- GPU resource: gpu:l20
- mandatory work root: /eeedata/sgxjw2
- checkout: /eeedata/sgxjw2/AoI-Reproduction0804
- Miniconda root: /eeedata/sgxjw2/miniconda3
- environment prefix: /eeedata/sgxjw2/conda_envs/aoi_cuda
- results: /eeedata/sgxjw2/AoI-Reproduction0804-results

First inspect `whoami`, `sinfo`, `module av`, available storage, the NVIDIA
driver from a GPU allocation, and the current Git state. Every user-managed
file, including code, environment, package cache, temporary files, logs,
results and handoff artifacts, must remain below `/eeedata/sgxjw2`. Do not use
`/share/home` for this workflow. Preserve the checkout on branch main with a
clean Git status. Record the exact commit and never pull, switch branches, or
edit tracked files once formal artifacts exist.

Execute these gates in order:

1. Obtain the repository as a real Git clone. If GitHub is unavailable, use a
   Git bundle transferred by SCP/MobaXterm; do not use a loose folder without
   `.git`. Create the external Slurm log/results directories.
2. Run `bash hpc/setup_aoi_cuda.sh`. Resolve Conda availability autonomously.
   Do not change pinned scientific dependencies. Confirm pip check and pytest.
3. Submit `hpc/aoi_pilot_1gpu.sbatch`. Monitor with squeue/sacct and tail both
   stdout and stderr. Inspect the train and validation audits, checkpoint and
   completion markers, finite AoI/success/reward arrays, learning curve,
   numerical stability, GPU identity, runtime and memory behavior.
4. Create the external `handoff/PILOT_APPROVED` marker only if the pilot is
   scientifically and operationally acceptable. Otherwise diagnose and report
   the evidence; do not delete or overwrite the run.
5. Submit exactly one matrix driver with `AOI_STAGE=train`. Prefer the 8-GPU
   full-node driver when all eight L20s are available. If a full-node allocation
   is impractical, use `hpc/aoi_matrix_array.sbatch`; each of its 48 array tasks
   requests four CPUs and one GPU, and the submission-time `%N` cap controls
   maximum concurrency. Never run the full-node and array drivers for the same
   stage at the same time. Do not attempt DDP/DataParallel or give multiple GPUs
   to one experiment cell.
6. Monitor all shard logs and reports. After PREEMPTED/TIMEOUT/NODE_FAIL, it is
   safe to resubmit the exact command because recovery is checkpoint-bound. For
   Python errors, provenance mismatches, artifact conflicts, OOM or NaN, stop
   automatic retries, preserve evidence, diagnose, and report. Never reduce
   batch size or alter paper hyperparameters silently.
7. After training ends, submit the CPU audit with AOI_AUDIT_SCOPE=train. Verify
   exactly 48 completed formal runs and eight successful shard reports. Create
   `handoff/TRAIN_APPROVED` only after this review passes.
8. Submit the same 8-GPU script with AOI_STAGE=eval, then submit the CPU audit
   with AOI_AUDIT_SCOPE=validation after evaluation completes. Build the
   validation study manifest and explicitly partial Algorithm-2 figures using
   REMOTE_RUNBOOK.md. Package commit ID, environment versions, Slurm job IDs,
   sacct states, shard reports, audits, figures and concise findings under the
   external handoff directory.

Hard constraints:
- Never run `final_test` or seeds 101..106.
- Never delete, rename, reset or overwrite a formal run directory.
- Never change episodes=500, steps=100, network sizes, batch size, replay,
  evaluation seeds, checkpoint cadence, device identity or paper profile.
- Never deploy or commit a new matrix driver midway through a formal lifecycle;
  the Git commit and tracked-tree digest must continue to match every existing
  checkpoint. A newly committed array driver requires a fresh pilot and fresh
  formal result root.
- Never write result artifacts into the Git checkout.
- Never place user-managed workflow files outside `/eeedata/sgxjw2`.
- Never claim the entire paper is reproduced: this repository covers Algorithm 2 only,
  and the other three comparison algorithms are absent.

Be proactive and use your judgment. Keep me informed with short milestone
updates containing job IDs, status, evidence and the next action. Do not ask me
to run commands you can run yourself. Stop only for an actual permission issue,
an ambiguous scientific choice, a persistent infrastructure failure, or a hard
gate failure that requires code changes.
```
