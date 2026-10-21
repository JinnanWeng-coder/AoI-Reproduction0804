# Uniform Random frozen evaluation (CPU, no training)

Fixed scope: P5/N4/K3/gap25; worlds1001–1030; action_stream_id12 and13 crossed with worlds, **60 unique units**. Streams are **not training seeds/models** or two identical repeat0 records. These worlds are user-fixed exploratory worlds, not new unseen confirmation.

The source environment is the existing `configs/training100_fresh_sources.json`; all four learning-condition templates must match on environment fields. `run_random_training_baseline.source_bundle/draw_actions` are reused read-only, not its historical Git guard or training lifecycle. The new guard pins this implementation and rejects changes to core/environment/reward/bundle/sampler relative to its reviewed starting commit. No old policy, training NPZ or lab historical directories are required.

Sampling is exactly the existing Random definition: independently each slot/leader, RB uniform{0,1,2}, mode uniform{0,1}, power uniform[1,30] **dBm**, with RB/mode bin-center encoding. Draw order: RB[5], mode[5], power_fraction[5]. No CAM-dependent mode override, neural clipping, actor, checkpoint, optimizer or updates. Theoretical mean linear power is about149.56818536mW, a units/sampler diagnostic only, never a formal-result acceptance threshold.

Policy RNG: PCG64(SeedSequence([action_stream_id, world, 0x52414E44, 1])). It is private and distinct from the environment's `reset_world(world)` generator. Every action draw checks it did not consume physical RNG. Warmup draws are consumed normally before scoring. Both streams must have exactly matching initial/final exogenous probes within each world; closed-loop AoI/demand/observations need not match.

## Shared frozen lifecycle

`analysis.frozen_rollout.sequential_warm` is extracted from the existing MADDPG frozen evaluator, which now calls that same helper. Random also uses it, not a separate behaviorally different loop. Once per world: cold reset;5 warmup episodes;100 scored episodes;100 slots/episode. Native start_episode semantics handle mobility, AoI carryover and CAM reset. Warmup never enters scored arrays. Warmup-end AoI and first scored slot continuity are verified. Raw arrays use the same logged float32 fields, shape(1,100,100,5), and existing `evaluate_frozen_deployment.reduce_arrays` validates/reduces them.

Unit key: (uniform_random,p05_n04_g25,world,action_stream_id). IDs0–29 = stream12/world1001–1030;30–59 = stream13/same worlds. Output config's seed0 is an unused source-template placeholder, not an evaluated/trained seed; every actual world is reset explicitly. Config algorithm fields describe the provenance template only; the runtime policy is always uniform_random and constructs no model.

## Lab Windows commands

Use a clean, independently pinned checkout of the delivered commit. Reuse the existing `.envs/aoi_cuda310/python.exe`, Python3.10/NumPy1.23.5; no environment installation, GPU or sbatch.

```powershell
$base='E:/Jinnan_Edisk/Research/Parvini_Reproduction_core'
$py="$base/.envs/aoi_cuda310/python.exe"
$study="$base/AoI-Reproduction-diagnostics/actor-sharing-study"
$batch="$study/Random-frozen-eval/P5_N4_gap25/random_uniform_stream12_13_world1001_1030_20261021_v1"
$sha=(git rev-parse HEAD).Trim()
& $py -B -m analysis.evaluate_random_frozen preview
& $py -B -m analysis.evaluate_random_frozen init --study-root $study --expected-commit $sha
& $py -B -m analysis.evaluate_random_frozen preflight --batch-root $batch
& $py -B -m analysis.evaluate_random_frozen smoke --batch-root $batch
& $py -B -u -m analysis.evaluate_random_frozen run --batch-root $batch --device cpu --workers 2
& $py -B -m analysis.evaluate_random_frozen status --batch-root $batch
& $py -B -m analysis.evaluate_random_frozen summary --batch-root $batch
& $py -B -m analysis.evaluate_random_frozen export --batch-root $batch
```

Check every external command's exit code. Do not init an existing batch or update a running checkout. `smoke` runs two action streams on nonformal world910001,5 warmup+2 scored episodes, same100slot horizon; it validates the real decoder, reducer, CAM, denominators, warmup/AoI boundary and exogenous probes. It writes only separate tmp and smoke metadata, not accepted formal units. A matching smoke PASS is required before any formal unit.

The existing selected-evaluation queue is reused through a source-free adapter. Default2 CPU child processes, optional `--workers 1` for actual resource contention. RUNNING.json prevents duplicate supervisors. A child failure stops new launches and lets active peers finish. On interruption only owned children are stopped; partial attempts/logs/arrays are preserved. Accepted units are re-verified before skip, not blindly counted again.

Status distinguishes accepted IDs, native EVAL_COMPLETE pending acceptance, failed/partial IDs and unrun IDs. Native-complete post-check failure: run `unit --id ID --attempt ORIGINAL` to inspect/accept existing arrays, never reroll. A truly incomplete unit retries only in a new attempt with `--reason`, e.g. `unit --batch-root $batch --id 7 --attempt a02 --reason interrupted_before_native_complete`. A remaining-ID queue uses `run --ids 7,8 --attempt a02 --reason remaining_after_interruption`; replace example IDs with actual status, never resubmit the full matrix. No checkpoint/resume/partial-window merging exists: the unit is the smallest atomic observation window.

Wait for all recorded processes to exit before handling stale locks. Preserve stale lock JSON and all partial attempts; never remove an active lock or kill unrelated work. Code/source/protocol errors stop for Codex; do not modify sampling, seeds/worlds/tolerances to pass.

After all60 accepted, the queue automatically summarizes/exports. Manual summary/export are idempotent and revalidate all accepted arrays and tables. Partial analysis/ZIP remains; use the next unused analysis `--attempt a02`, not new rollouts.

## Metrics, tables and acceptance

`world_stream_platoon.csv`:300 rows, raw sum/count, AoI/cap/>50, deadline binary CAM, linear mW, combined/global/task1/task2 reward, V2I attempts/resets, RB/mode counts. Nulls remain null. Global reward is one joint[world,episode,slot] array; repeated platoon component columns are common copies, not five independent global observations.

`unit_world_endpoints.csv`:60 rows; global sum/count explicitly counted **once per unit** (10000 joint scored slots),50000 agent-slots and500 CAM deadlines/unit. AoI/CAM extrema are selected separately within each world.

`stream_summary.csv`:both streams separately, each30world mean and **sample world SD(ddof1)** of mean AoI, mean CAM%, power mW, reward/components and scene-first worst AoI/lowest CAM%. No training-seed SD, CI, best-stream selection or60-independent-world interpretation.

`pooled_world_platoon.csv` (150 supplementary rows), `pooled_world_endpoints.csv` (30), `pooled_world_summary.csv`:first pool both streams by sum/count within each world/platoon, then own extrema, then mean/SD over30 worlds. This uses two streams and differs from learned policies' single repeat0. Do not replace primary separate-stream summaries with a favorable pooled result.

`legacy_endpoints.csv` keeps across-world pooled-platoon extrema separately. Existing learning evidence is not required or automatically rerun; comparing those tables later is read-only and must respect stream weighting.

Final technical PASS requires60 accepted completion records,300 raw platoon rows,3000000 scored agent-slots,30000 CAM deadline events,600000 joint global reward slots, exactly30 shared worlds and both fixed streams.150000 warmup agent-slots are excluded. Supplementary150 rows/30worlds are not additional runs. No performance-direction gate.

Deliver `analysis/a01/core_evidence.zip` (or the explicit new analysis attempt), including protocol/actual resolved config/source bundle, all60 completion/provenance records, counts/tables/definitions, process records and short logs. Raw NPZ/full logs stay in the lab; no weights or SHA256 manifests. The intended user-side return path is `tmp/random_baseline/eval/core_evidence.zip`; this is a manual transfer, not an automatic two-machine sync.
