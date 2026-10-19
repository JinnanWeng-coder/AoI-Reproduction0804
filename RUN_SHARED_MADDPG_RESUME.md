# Shared MADDPG: every-50-episode checkpoints

This is an opt-in extension for Shared Modified MADDPG-TDec + training-reward
best100. It does not retroactively make old `policy_best_training.pt`,
`policy_final.pt`, or interrupted policy-only runs resumable. No extra evaluation,
selection-validation worlds, or training is launched by saving checkpoints.

## Six-seed training batch (no frozen evaluation)

Run IDs 0..5 map to training seeds 8..13. Each run starts from scratch for
2,500 episodes x 100 slots, using its accepted independent Algorithm2 config.
Only actor sharing and output naming change; no old actor weights are loaded.
Five leaders share one actor within each run. Different seeds have independent
models, replay, optimizer state and best100 selection. Critics remain unshared;
the shared actor takes one Adam step on
`(global_loss + sum_leader_task1_task2_losses) / number_agents` per actor update.
This is a new shared-actor treatment, not a claim of equivalent learning dynamics.

From a clean pinned checkout, with the existing lab interpreter and accepted
Training100-best batch (`lab_20261017_012015`):

```powershell
& $py -B -m analysis.shared_maddpg_contract init --training-batch $trainingBatch --batch-id $batchId --expected-commit $sha
if ($LASTEXITCODE -ne 0) { throw 'Initialization failed' }
& $py -B -m analysis.shared_maddpg_contract preflight --batch-root $batchRoot
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
& $py -B -u -m analysis.shared_maddpg_contract run --batch-root $batchRoot
if ($LASTEXITCODE -ne 0) { throw 'Queue failed; preserve the attempt' }
```

`$batchRoot` is `<study>/Shared-MADDPG-TDec/P5_N4_gap25/<batchId>`.
The serial queue runs all six seeds, then automatically summarizes and exports
`analysis/a01/core_evidence.zip`. Do not invoke summary/export again after a
successful full queue. Expected acceptance: six runs, six best and six final
policies, 15,000 episode rows, 1,500,000 training environment steps and zero
new policy evaluations. The original 24-run batch is read-only. Wait until
existing jobs finish; do not delete a live queue lock or share its checkout.

If only packaging failed after all six records were accepted, run the explicit
`summary`/`export` commands with a new unused `--attempt a02`; never retrain an
accepted seed. A failed training attempt instead requires an explicit recovery
decision using the retained checkpoints as described below.

## What is saved

`checkpoints/episode_000050.pt`, `episode_000100.pt`, ... are immutable complete
episode-boundary checkpoints. A 2,500-episode run retains 50 files. An explicit
partial stop additionally saves its last completed episode. `latest.json` is a
small pointer with the actual file size, not a second full copy. Final-episode
checkpoints are marked completed and cannot be used to extend the budget.

Each checkpoint contains:

- One shared online actor, target actor, and actor Adam state (not five copies).
- All local/global critics, their targets and optimizer states.
- Full replay ring, capacity, contents and write counter.
- Environment state and Python/NumPy/PyTorch/CUDA RNG states.
- Learner/update counters, accumulated training/diagnostic metrics.
- Best100's rolling window, complete score history, and the best actor payload
  **as of that checkpoint**, even if a later attempt has replaced its best file.
- Configuration, implementation commit and runtime identity.

Best100 and final remain separate lightweight frozen-policy artifacts. They
cannot by themselves reproduce continued off-policy training. Do not invoke the
old `checkpoint_mode=resumable` selection-validation workflow: the new entry
explicitly keeps the existing policy-only evaluation artifacts and adds its own
full recovery snapshots without validation rollouts.

## Restore semantics

Always restore into a **new attempt directory**. Never rewind or overwrite an
existing attempt, accepted result, metrics, checkpoint, or provenance. The new
run records the source checkpoint and episode and continues at the next episode.
For example, restoring episode 150 starts episode 151, not episode 1, and stops
at the original total budget. Previous metrics/selection history are restored,
not counted twice. Keep one accepted result per seed in the batch ledger.

The same implementation commit, Python/NumPy/PyTorch/CUDA versions, device/GPU
identity and training configuration are required. Only output root and run name
may change. CPU synthetic equality is tested; this is not a promise of bitwise
identity across hardware, software versions or nondeterministic GPU kernels.

## Entrypoints (PowerShell, existing Python environment)

The six-seed `analysis.shared_maddpg_contract train` now enables interval50
automatically. Its explicit recovery form is:

```powershell
& $py -B -m analysis.shared_maddpg_contract train --batch-root $batchRoot --id 0 --attempt a02 --reason interrupted --resume $checkpoint
if ($LASTEXITCODE -ne 0) { throw 'Resume failed; preserve both attempts' }
```

`$checkpoint` is the seed's retained `episode_NNNNNN.pt` within this same batch.
The batch must already have a same-environment preflight. An already accepted
seed cannot be resumed for performance-based reselection. The serial queue does
not automatically choose a checkpoint or retry a failed run.

A standalone runtime entry also supports a resolved parent Algorithm2 config or
an explicit Shared config. Fresh training changes only actor sharing and explicit
output naming/device, keeping the supplied budget and all learning settings:

```powershell
& $py -B -m aoi_v2x_reproduction.runtime.shared_resume --config $configPath --output-root $newOutputRoot --run-name shared_seed08_a01 --device cuda:0 --dry-run
# After inspecting the printed configuration changes, omit --dry-run to train.
```

For a new attempt recovered from an existing checkpoint:

```powershell
& $py -B -m aoi_v2x_reproduction.runtime.shared_resume --resume $checkpoint --output-root $newOutputRoot --run-name shared_seed08_a02 --dry-run
# Omit --dry-run to restore and continue. Do not change device on restore.
```

`--dry-run` does not instantiate the training environment or create output.
Its JSON reports `completed_episodes`, `next_episode`, and `remaining_episodes`:
restoring episode 150 of 2,500 must show 150, 151, and 2,350 respectively.
The standalone entry is a per-run tool, not a substitute for six-seed source
preflight/acceptance. Real launches require clean pinned code; do not update a
checkout while a run is active. Load only trusted local checkpoint files.

## Storage and operational expectations

Using the repository's Algorithm2 configuration (state22, replay50,000), the
raw float32 network/target/Adam state estimate is 54.56 MiB and replay arrays
46.97 MiB per point. Thus these components alone are about 101.53 MiB per point,
or 29.75 GiB for six seeds × 50 checkpoints. **This is not a serialized-file
size estimate**: cumulative raw metrics, best actor, environment, Python object
and serialization overhead are additional and grow during the run. Measure the
first checkpoints and later growth on the laboratory machine. Saving all points
costs disk space and periodic I/O; it does not require keeping 50 models in GPU
memory. Leave room for the temporary atomic-save file and current checkpoint.

Do not package checkpoints, replay, or full raw metrics into compact evidence.
No SHA256 manifest is added. A failure during saving can leave a `.pt.tmp`; do
not treat it as a completed checkpoint. A numbered completed `.pt` can be used
even if interruption prevented updating the latest pointer.

## Local verification

`tests/test_shared_maddpg_resume.py` uses a small synthetic environment with real
small networks, Adam and replay. It checks uninterrupted vs episode50/100 restore,
best100 history on both sides of episode100, exact network/optimizer/target/replay/
metrics/RNG equality on CPU, shared-state deduplication, original-file preservation,
no validation rollout, invalid source/config/commit rejection and CLI dry-run.
It also checks simulated disk-full failure: the preceding checkpoint and latest
pointer stay unchanged, no new completed checkpoint appears, and saving does not
advance RNG state. An incomplete temporary file is never a recovery source.
No formal training or policy evaluation is required. Laboratory GPU recovery,
full-size checkpoint write latency and disk growth remain remote checks.
