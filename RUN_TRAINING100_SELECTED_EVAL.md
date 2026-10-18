# Training100 selected-seed best100 / final: laboratory Python evaluation

This is the explicitly authorized exploratory **5 conditions × 30 worlds × 2
checkpoint kinds = 300 policy-world-repeat** comparison. No training, no pilot,
no Slurm. It does not change the previous training run or its per-seed selection.

## Fixed selection and protocol

For each of Shared MAPPO-TDec, Independent MAPPO-TDec, Shared MAPPO-Combined and
Modified MADDPG-TDec, retain all six seeds 8–13 as candidates, then independently:

- best100: maximize `best_training100_score`; use that seed's `policy_best_training.pt`.
- final: maximize `last100_training_score` (episodes 2401–2500); use that seed's
  `policy_final.pt`. This is NOT a measured frozen final-policy reward.
- Exact across-seed score ties use the lowest training seed. No selection uses an
  evaluation reward, AoI or CAM. Best and final can select different seeds.
- The accepted `records/train_*.json` govern source attempts, not a directory glob
  or a by_seed shortcut. The source a02 for shared_tdec seed10 remains untouched.
- MADDPG noise0 and noise0.3 share the same selected actor artifact for each kind;
  MAPPO uses stochastic actions. Each condition/world has exactly **repeat0**.
- Fixed worlds1001–1030 are user-specified exploratory worlds. Historical non-use
  is NOT certified; do not describe these as a newly locked unseen confirmation.
- Each world is cold-started once, followed by 5 warmup and 100 scored episodes,
  each100slots, five platoons. Existing evaluator action/environment RNG rules,
  reward, environment, noise/clipping and action mapping are unchanged.
- Actual selected episode/version/role are recorded: best is not relabeled500 or
  final2500. Training budget is250000 environment steps even for an earlier best.
  Row `training_environment_steps` is the saved checkpoint's episode×100;
  `training_budget_environment_steps` is the completed source run's250000 budget.

Expected selection from lab_20261017_012015 (the program recomputes, not hardcodes):

| Model | best100 seed | final seed |
|---|---:|---:|
| shared_tdec |13|9|
| independent_tdec |12|9|
| shared_combined |13|9|
| algorithm2 |12|9|

## Safe sync and run (PowerShell)

Use the delivered actual commit as `$sha`. Verify no training/evaluation process
uses the checkout and `git status --porcelain` is empty. Fetch research/mappo and
`git switch --detach $sha`; verify HEAD equals it. No reset/force push, no source
record rewriting. Leave the checkout fixed while the queue runs. This task only
publishes to research/mappo, not main and not a PR.

Reuse the already working dedicated training environment; do not reinstall or
upgrade it for this evaluation. Replace a path only after locating the real
environment. All failures stop; do not launch the next command after a nonzero exit.

```powershell
$root = 'E:/Jinnan_Edisk/Research/Parvini_Reproduction_core'
$repo = Join-Path $root 'AoI-Reproduction0804'
$py = Join-Path $root '.envs/aoi_cuda310/python.exe'
$study = Join-Path $root 'AoI-Reproduction-diagnostics/actor-sharing-study'
$training = Join-Path $study 'Training100-best/P5_N4_gap25/lab_20261017_012015'
$batchId = 'selected_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$batch = Join-Path $study "Training100-selected-eval/P5_N4_gap25/$batchId"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:CUDA_VISIBLE_DEVICES = '0'
Set-Location -LiteralPath $repo

# Optional read-only dry-run: CSV/metadata only, never loads weights or rolls out.
& $py -B -m analysis.evaluate_training100_selected preview --training-batch $training
if ($LASTEXITCODE -ne 0) { throw 'source preview failed' }

& $py -B -m analysis.evaluate_training100_selected init --training-batch $training --batch-id $batchId --expected-commit $sha
if ($LASTEXITCODE -ne 0) { throw 'init failed' }
& $py -B -m analysis.evaluate_training100_selected preflight --batch-root $batch --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'preflight failed; no evaluation' }

# Run this once; two evaluation children maximum on the existing4090.
& $py -B -u -m analysis.evaluate_training100_selected run --batch-root $batch --device cuda:0 --workers 2
if ($LASTEXITCODE -ne 0) { throw 'evaluation stopped; preserve scene and report failing IDs/logs' }
```

The foreground queue automatically summarizes and exports after all300 accepted
units. For a long background job, replace ONLY the last command with:

```powershell
$queue = Start-Process -FilePath $py -ArgumentList @('-B','-u','-m','analysis.evaluate_training100_selected','run','--batch-root',$batch,'--device','cuda:0','--workers','2') -WorkingDirectory $repo -RedirectStandardOutput (Join-Path $batch 'logs/queue.out') -RedirectStandardError (Join-Path $batch 'logs/queue.err') -WindowStyle Hidden -PassThru
$queue.Id
```

Do not run both foreground and background queues. One/two workers are supported;
do not increase concurrency while running. A batch lock prevents duplicate queues.
If interrupted, preserve partial attempts and inspect the PID/lock; do not delete
a live lock. The driver captures commands, child PIDs, durations and exit codes;
unmeasured MaxRSS stays null. Raw output is under the new batch, not training/a01,
training/a02, existing-evidence, or historical MAPPO_results.

## Acceptance and output

`protocol/candidates.json` stores all24 candidates; `selected_models.csv` and
`sources.json` freeze8 selected artifacts; `units.json` freezes300 units.
Preflight recomputes selection from all60000 episode rows, checks accepted source
identities/configuration and strictly loads8 selected actor payloads, plus a CUDA
tensor probe. It does no rollout. The chosen source policies/metadata remain read-only.

Acceptance requires300 native completions and accepted unit records,1500 per-platoon
rows,15000000 scored agent-slots,150000 CAM deadline events, matching world/process
probes across conditions, actual episode/version/config/provenance, unchanged
actors and correct arrays/denominators. Performance direction is not a PASS gate.

`analysis/a01/` contains:

- `world_repeat_platoon.csv`, world endpoints, selected-seed endpoints, scene-first
  condition summary and separate legacy summaries. Cumulative sums/counts preserve
  AoI, CAM final-slot deadline success, linear-mW power, reward terms, strictAoI>50
  and trusted V2I attempt/reset metrics. Undefined values remain blank, not zero.
- Per-world weakest AoI and CAM are chosen separately. Legacy pools across worlds
  by platoon before extrema; scene-first chooses within each world then averages.
- Each selected condition has n_training_seeds=1. World SD is descriptive variation
  over30 worlds, not training-repeat uncertainty. Training-seed SD/t95 and repeat
  variation are null. Best-vs-final is NOT a same-seed checkpoint effect estimate.
- `verification.json`, `REPORT_CN.md`, and `core_evidence.zip`. ZIP contains only
  CSV/JSON/report/selection/source/completion/process records. No weights, NPZ,
  full logs/trajectories/cache or SHA256 manifests. Originals stay on the lab PC.

## Failure / selective continuation

First failure stops scheduling; already running children finish. Never select a
new seed/model/world after viewing results. Accepted units skip only after complete
source/identity/array/CSV revalidation. Partial outputs are never overwritten.

For a native `EVAL_COMPLETE.json` with a failed post-check, inspect the saved arrays
first, then use `unit --batch-root $batch --id ID --attempt a01 --device cuda:0`
to retry acceptance without rollout (after the cause is understood). A completed
native attempt in another path blocks a fresh rollout.

For genuinely partial/unrun units, after technical review use `run --batch-root
$batch --ids COMMA_SEPARATED_IDS --attempt a02 --reason ACTUAL_REASON --device
cuda:0 --workers 2`; preserve a01. An implementation error requires returning to
engineering, not hotpatching a running checkout or loosening checks.

Independent CPU summary/export, if needed, is one command:
`python -B -m analysis.evaluate_training100_selected summary --batch-root BATCH --attempt a02`.
Use a new analysis attempt when a previous summary exists. This never reruns evaluation.

Local tests use synthetic actor artifacts and fake environments/children. They do
not certify lab GPU performance, real model payload compatibility or the300 actual
rollouts. Those are checked by Grok's preflight and final acceptance.

Local delivery checks (2026-10-18):96 selected-eval/deployment/training100 tests,
two additional native-array acceptance/skip tests and11 isolated legacy tests
passed across the validation invocations. AST syntax and metadata-only CLI preview
passed; the received evidence reproduces the selection table above. One additional
old E1 source-symlink test was blocked by Windows WinError1314 (no symlink privilege),
not repaired or counted as passing. The new path does not require symlinks. Legacy
real-environment rollout tests were not part of this regression selection; no
formal lab model evaluation or experiment was run locally.
