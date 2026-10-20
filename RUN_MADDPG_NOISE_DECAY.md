# MADDPG-TDec exploration decay: four fresh runs

Scope: Independent and Shared Modified MADDPG-TDec, seeds **10,12**, each 2500 episodes x 100 joint environment steps, P5/N4/K3/gap25. No deployment evaluation, no additional seeds. Existing fixed0.3 runs are read-only configuration sources, never weight/replay initializers.

## Protocol and sources

Before drawing an action, k is the number of completed **joint environment steps**:

`sigma(k) = 0.05 + 0.25 * max(1 - k/150000, 0)`.

At k=0/50000/100000/150000, sigma=0.3/0.2166666667/0.1333333333/0.05. The last1000 episodes retain0.05. Each leader's three normalized action coordinates receive the existing independent Gaussian draw, then the existing +/-0.999 clipping and RB/mode/power decoding. Target smoothing0.2/clip0.5 is unchanged. Frozen eval_noise is independent; no evaluation is scheduled here.

Accepted `records/train_*.json` identify the original run, including Shared seed10's accepted a02. The new contract reads only that record, resolved config, COMPLETE and accepted batch summary. It does not need original raw NPZ/weights to initialize. It preserves each source's networks, actor sharing, optimizer/replay settings, reward, physics, training RNG, profile and semantic_version. Historical missing/null independent sharing means false and is recorded. Historical `paper_faithful_v4` is retained: this dedicated research contract does not invoke or relax the active baseline profile gate.

Only four serialized schedule fields and output_root/run_name change. The inherited `checkpoint_every=50` remains a historical generic field, **not the save interval for this new protocol**: manifest, provenance, COMPLETE and the dedicated runner explicitly set policy and full-state intervals100. Old Shared50 contracts remain unchanged.

Run mapping:

|ID|Structure|Seed|
|---|---|---|
|0|Independent|10|
|1|Independent|12|
|2|Shared|10|
|3|Shared|12|

Shared retains one actor/target/Adam for five local22-dimensional observations; local/global critics and its reviewed mean actor objective are unchanged. No learner, network or environment code is modified.

## Lab Windows commands

Run from a clean **independent checkout pinned to the delivered implementation SHA**, not from an active working tree. Python is the existing `.envs/aoi_cuda310/python.exe`; do not upgrade it. In PowerShell:

```powershell
$base = 'E:/Jinnan_Edisk/Research/Parvini_Reproduction_core'
$py = "$base/.envs/aoi_cuda310/python.exe"
$study = "$base/AoI-Reproduction-diagnostics/actor-sharing-study"
$batch = "$study/MADDPG-NoiseDecay-TDec/P5_N4_gap25/noise03_to005_ep1500_seed10_12_20261021_v1"
$sha = (git rev-parse HEAD).Trim()
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONUNBUFFERED='1'
& $py -B -m analysis.maddpg_noise_decay_contract init --study-root $study --independent-batch "$study/Training100-best/P5_N4_gap25/lab_20261017_012015" --shared-batch "$study/Shared-MADDPG-TDec/P5_N4_gap25/shared2" --expected-commit $sha
& $py -B -m analysis.maddpg_noise_decay_contract preflight --batch-root $batch
& $py -B -m analysis.maddpg_noise_decay_contract smoke --batch-root $batch --device cuda:0
& $py -B -u -m analysis.maddpg_noise_decay_contract run --batch-root $batch --workers 2
```

Check `$LASTEXITCODE` after **each** command and stop on failure. Do not init again when a manifest already exists. `load` checks commit, source identities, matrix and configuration derivation every time.

The technical smoke is separate under batch/tmp, uses tiny networks104x2 slots, a dedicated technical world910010, cut5 and a13-step schedule. It tests real-environment Independent/Shared continuous versus fork-resume bitwise equality across the floor. It is **not** a formal pilot, a formal run or deployment evidence. A matching CUDA smoke PASS is required before formal training.

The reused queue runs at most two training processes on cuda:0, one-thread CPU math per process, isolated temporary paths, study-wide GPU ownership lock. `--workers 1` is available if actual GPU/RAM load requires it; hyperparameters do not change. Do not overlap unrelated training/evaluation. If a child fails, stop launching new runs, let its healthy peer finish, and retain all attempts/logs/checkpoints. The queue automatically summarizes and creates **both ZIPs only after4/4 are accepted**.

## Monitor and resume

```powershell
& $py -B -m analysis.maddpg_noise_decay_contract status --batch-root $batch
Get-ChildItem "$batch/logs/*.out" | Select-Object Name,Length,LastWriteTime
Get-Content "$batch/logs/train_00_a01.out" -Tail 8
nvidia-smi
```

Progress is emitted every10 episodes; full recovery state and compact noise log every100. Full25-policy coverage, best/final, training arrays and counters are rechecked before accepting each run. Native COMPLETE without an accepted record is verified/accepted, never retrained. An accepted complete key is skipped only after re-verification.

On interruption, do not overwrite, extend a finished run, load an old fixed0.3 model or resume from an actor-only snapshot. Wait for owned/peer processes to exit. Inspect this batch's `status` / `checkpoints/latest.json`; take its latest intact **numbered full-state** checkpoint. Resume into the next unused attempt using its run ID; example for ID0's a01:

```powershell
$pointer = Get-Content "$batch/training/a01/runs/noise_decay_algorithm2_seed10/checkpoints/latest.json" -Raw | ConvertFrom-Json
if (-not $pointer.can_resume) { throw 'Not an incomplete resumable state' }
& $py -B -u -m analysis.maddpg_noise_decay_contract train --batch-root $batch --id 0 --attempt a02 --reason interrupted_full_state_recovery --resume $pointer.checkpoint
```

This explicit `train` command is foreground and does not acquire a second queue lock: **only run it after the supervisor/children have exited**. Never bypass a live lock. A stale lock can only be removed after verifying its owner and all recorded children have exited and preserving its JSON in records. Use the actual ID/path/next unused attempt for another interrupted run. Original attempts remain immutable; inherited episodes/noise/selector/metrics are restored once, not concatenated again. The cumulative k never restarts. Snapshot index entries may reference prior attempts within this new batch.

The full state reuses the tested optimizer/target/replay/environment/RNG/metrics/best100 mechanism and requires identical code, config and runtime/device. Fixed0.3 old states, best/final and periodic policies are rejected as recovery points. At most the uncheckpointed tail after the last complete100 boundary is re-executed under a new attempt.

For untouched run IDs after recovery, run only those IDs with unused launcher paths, e.g. `run --batch-root $batch --ids 1,2,3 --attempt a02 --reason remaining_after_interruption --workers 2`. Do not restart the full matrix. Accepted IDs are safe to reverify but do not need new training.

## Finish and evidence

If recovery was run explicitly, or training finished but summary/export was interrupted:

```powershell
& $py -B -m analysis.maddpg_noise_decay_contract summary --batch-root $batch --attempt a01
& $py -B -m analysis.maddpg_noise_decay_contract export --batch-root $batch --attempt a01
```

Existing verified outputs/ZIPs are idempotent. A partial analysis directory or ZIP is retained; use the next unused **analysis** attempt a02, not a training rerun. Do not change tolerance, noise, world/seed or best100 selection to pass.

Expected final verification:4 accepted runs,10000 episode rows,50000 episode-platoon rows,1000000 environment steps,5000000 agent-slots,50000 CAM deadlines,100 periodic policies (25/run),4 best100 and4 final. Technical PASS does not depend on which structure has better reward/business metrics. Two seeds per structure are two training replicates; slots/episodes/platoons are not additional independent seeds.

Files under `analysis/a01/` (or the explicitly selected new analysis attempt):

- `core_evidence.zip`: protocol/source/config/commit/accepted records, complete trainingreward and exploration logs, learning diagnostics, best/final selection metadata, periodic policy index, run/episode summaries and verification/report.
- `training_episode_platoon_metrics.zip`: `training_episode_platoon.csv` (50000 rows), `training_episode_metrics.csv` (10000 rows), window summary, field definitions and verification. Automatically delivered with core evidence.

Raw NPZ, actor policies, numbered full-state checkpoints, technical smoke state and complete logs stay in the lab. No SHA256 list is created. Preserve original sums/counts and float64 reduction precision. Deadline CAM uses final-slot success once/platoon/episode, AoI is strict>50, power is averaged in linear mW. `episode_reward` on platoon rows repeats the run-level episode mean and must not be summed five times. Joint global reward appears once/episode in the episode table, repeated with agent-slot denominators only in the platoon table.

Best100 uses every complete100-episode trailing window and strict improvement (earliest ties), saving the actual window endpoint actor **after that episode's online updates**, not checkpoint quantization. Periodic actor-only policies at100..2500 use the existing strict frozen actor loading path and cannot resume. The2500 snapshot and final may contain identical actor tensors but retain distinct roles. New training's reward changes can reflect the changing execution noise; actor deployment quality requires a later separately authorized noise0 frozen evaluation.
