# Windows laboratory machine: fresh training, no Slurm

This supersedes the HPC-source requirement for the explicitly authorized local
fresh batch. Do NOT download old policies, train_metrics.npz, B1 finals, or whole
HPC batches merely to start training. `configs/training100_fresh_sources.json`
contains four exact config templates plus all 24 seed/run-name/source-commit/config
identities from the accepted D1 ledger. Expansion reproduces every original config
hash; only seed and historical run_name differ within each template. Linux paths
inside original configs are reference strings, NOT local input dependencies.

The new mode explicitly records historical payload audit as NOT_RUN_NOT_REQUIRED;
it does not forge the old HPC audit PASS. There is no legacy weight/state loading,
no resume, no pilot, and no default-config substitution. Runtime, reward, learning
and environment are unchanged. `paper_faithful` source profiles remain unchanged.
It uses the same opt-in selector, config derivation, full-budget/artifact verification,
60000-row summary and compact ZIP as the HPC version.

## Layout and safe relocation (Grok on the lab machine only)

```text
E:/Jinnan_Edisk/Research/Parvini_Reproduction_core/
  AoI-Reproduction0804/          # clean pinned Git checkout
  AoI-Reproduction-diagnostics/
    actor-sharing-study/Training100-best/P5_N4_gap25/<batch_id>/
  .envs/aoi_cuda310/             # dedicated environment
```

Inspect `E:/Jinnan_Edisk/Research/AoI-Reproduction0804` first: clean status, no running
training and `.git` must be a directory (ordinary clone, not managed worktree).
Move only that exact directory to the above target, using native PowerShell
`Move-Item -LiteralPath`. Resolve the absolute source/destination and confirm the
destination is a child of this exact new root. Never overwrite or merge an existing
target; if both old/new copies exist, stop and report. If it was already moved, reuse
the verified target. Move the old diagnostics folder only if one actually exists
and the target does not; otherwise create the new empty diagnostics directory.
No recursive deletion or `git reset`, no main merge. Fetch and safely detach at
the delivered commit; leave that checkout fixed during training.

## Dedicated environment

Historical source COMPLETE records and local validated runtime use Python3.10.20,
torch2.11.0+cu126 and NumPy1.23.5. The lab's existing torch2.6 environments are NOT
the recorded runtime. Create an isolated environment, do not update base/SimuV2X.
Conda executable was reported under `D:/Anaconda3/anaconda3/Scripts/conda.exe`;
verify it exists, or locate the real Conda executable before proceeding.

Run the following in PowerShell, AFTER relocation/safe sync:

```powershell
$root = 'E:/Jinnan_Edisk/Research/Parvini_Reproduction_core'
$repo = Join-Path $root 'AoI-Reproduction0804'
$envDir = Join-Path $root '.envs/aoi_cuda310'
$condaExe = 'D:/Anaconda3/anaconda3/Scripts/conda.exe'
if (!(Test-Path -LiteralPath $condaExe)) { throw 'Locate Conda before proceeding' }
if (!(Test-Path -LiteralPath $envDir)) {
    & $condaExe create --prefix $envDir python=3.10.20 pip -y
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
}
$py = Join-Path $envDir 'python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUNBUFFERED = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'
$env:OPENBLAS_NUM_THREADS = '8'
Set-Location -LiteralPath $repo
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $py -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'Pinned torch installation failed; do not substitute silently' }
& $py -m pip install -r requirements.cuda.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Pinned dependencies installation failed' }
& $py -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts' }
```

If an environment already exists, inspect its ownership/version before installing
into it; never overwrite an unrelated environment. Stop on unavailable pinned wheels,
network or dependency errors and report the exact gap, not silently switch versions.
No driver/system CUDA installation is required by these commands.

## Init, CUDA/config preflight, queue

One RTX4090 was reported: run exactly ONE training child at a time, all 24 queued
serially. No extra pilot. Each child is a formal run, 2500 episodes x100slots. Six
training seeds in each of shared_tdec, independent_tdec, shared_combined, algorithm2;
MADDPG training noise0.3, not two runs for test execution noise0/0.3.

```powershell
$study = Join-Path $root 'AoI-Reproduction-diagnostics/actor-sharing-study'
$batchId = 'lab_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$initText = & $py -B -m analysis.training100_contract init-local --study-root $study --batch-id $batchId
if ($LASTEXITCODE -ne 0) { throw 'init-local failed' }
$batch = (($initText -join "`n") | ConvertFrom-Json).batch_root
& $py -B -m analysis.run_training100_local preflight --batch-root $batch
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; do not train' }
```

Preflight checks pinned code, original config identities, all pinned dependency
versions, CUDA and a real256x256 float32 GPU multiplication. It does NOT train or
load old policies. Report RAM/free disk/GPU usage too; do not share the4090 with
another training job, kill unrelated processes or reduce batch size on OOM.

Foreground execution (keeps the console occupied until completion):

```powershell
& $py -B -u -m analysis.run_training100_local run --batch-root $batch
```

OR start ONCE in the background, not both:

```powershell
$queue = Start-Process -FilePath $py -ArgumentList @('-B','-u','-m','analysis.run_training100_local','run','--batch-root',$batch) -WorkingDirectory $repo -RedirectStandardOutput (Join-Path $batch 'logs/queue.out') -RedirectStandardError (Join-Path $batch 'logs/queue.err') -WindowStyle Hidden -PassThru
$queue.Id
```

The queue records actual interpreter, PID/commands/start-end times/exit codes and
per-run logs. It launches each next child only after the previous one exits0 and
its accepted record exists. A simple study-root launcher lock prevents duplicate
queues. Existing locks are never automatically removed; verify the recorded owner
has exited before requesting recovery. No full resume/recovery framework.
Training buffers metrics until final save; a quiet child log is not by itself a
hang. Before episode100 there is intentionally no best; best is written on strict
improvement afterwards. Inspect PID/GPU activity and artifacts, not log size alone.

## Completion and failure

All24 accepted -> automatic CPU summary/export, no Slurm/sacct. Every new run is
verified from arrays and strict best/final actor loading. Outputs:

- 24 best +24 final; all24 run records and original/derived config provenance.
- `analysis/a01/training_episode_curves.csv`:60000 rows; selected policies:24 rows.
- technical PASS requires2500 episodes/run, 6M accepted training environment steps,
  correct per-seed100-window/tie/version/timing rules and MAPPO optimizer counts.
- `runtime_report.json` contains local process accounting, platform/GPU/dependency
  reports under protocol; MaxRSS is null when unmeasured, never fabricated sacct.
- `analysis/a01/core_evidence.zip`: compactCSV/JSON/docs/completion/source records,
  bounded log tails. ZIP integrity/24+60000 row readback checked. Weights/NPZ/full
  logs/cache remain on this lab machine, not in Git or the compact archive.

This Windows/4090 execution is a new batch, not an assertion of bitwise equality
with Linux/L20 historical trajectories. No frozen policy evaluation is authorized.

Failures stop the queue and preserve the failed run/log/exit record. Do not retry
all24 or change hyperparameters. Accepted runs are strictly reverified before skip.
Selective recovery uses `run --ids <failed_and_unrun_IDs> --attempt a02 --reason
<actual_failure>` after technical review; interrupted runs restart from scratch
in new paths, never fake resume. Extra failed-attempt compute is separate from the
6M accepted steps. Summary/export can also be called directly via the existing
training100_contract commands; failed analysis uses a fresh attempt, no overwrite.
