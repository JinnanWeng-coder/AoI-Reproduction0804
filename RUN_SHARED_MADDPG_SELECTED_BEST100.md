# Shared MADDPG: selected best100, deterministic execution

Entry: `python -B -m analysis.evaluate_shared_maddpg_selected`.
This adds one model and 30 rollouts; the original selected300 and all-seed900
matrices retain their original entrypoints and contracts.

## Source and selection

Read the accepted six-run Shared MADDPG batch, not the earlier independent
Algorithm2 training batch. `records/train_00.json` through `train_05.json` are
authoritative accepted run pointers. Read only their complete 2500-row episode
ledgers, including inherited prefixes exactly once. Recompute each best100 using
the existing float64 training-reward rule, then choose the largest score across
seeds8–13; exact ties choose the lowest training seed. No evaluation score enters
selection. A metadata-only preview accepts either the live batch or its compact
ZIP and never loads weights:

```powershell
& $py -B -m analysis.evaluate_shared_maddpg_selected preview --training-batch $trainingBatch
# Offline evidence review, alternative to the preceding command:
& $py -B -m analysis.evaluate_shared_maddpg_selected preview --evidence-zip $trainingEvidence
```

For the accepted `shared2` evidence, the computed winner is seed10, best episode229,
window130–229, version11418. This is a validation observation, not hard-coded
selection logic. Its accepted a02 run inherited1850 episodes, but that does not
change the earlier best checkpoint's episode or version. Preflight validates all
six candidate best payloads, each containing one shared actor. The five leaders
each call that actor with their own22-dimensional local observation. No critics,
replay or optimizer are created for these portable-policy rollouts.

## Protocol and output

Only `policy_best_training.pt`, noise0, P5/N4/K3/gap25, worlds1001–1030, repeat0.
Each world cold-starts once; five warmup episodes precede100 scored episodes,
100 slots each. Mobility, channels, interference history and AoI continue across
episodes under the existing sequential-warm evaluator; CAM demand resets per
episode. Noise placement/clipping, action decoding and RNG use the existing
MADDPG evaluator unchanged. These worlds have been used; they remain
`user_fixed_exploratory_worlds`, `new_unseen_confirmation=false`.

New batch root:
`<study>/Shared-MADDPG-selected-eval/P5_N4_gap25/<unique_batch_id>`.
Source runs and historical evaluations remain read-only. Runtime config,
source sizes/mtimes, selected weight digest, best metadata, before/after parameter
checks and completion records are retained. The original large metrics NPZ stays
in each new unit's native directory on the lab machine.

```powershell
& $py -B -m analysis.evaluate_shared_maddpg_selected init --training-batch $trainingBatch --batch-id $batchId --expected-commit $sha
& $py -B -m analysis.evaluate_shared_maddpg_selected preflight --batch-root $batchRoot --device cpu
& $py -B -u -m analysis.evaluate_shared_maddpg_selected run --batch-root $batchRoot --device cpu --workers 2
& $py -B -m analysis.evaluate_shared_maddpg_selected status --batch-root $batchRoot --verify
```

Check each exit code and stop on failure. Only one supervisor per batch; its
`RUNNING.json` is exclusive. Use a clean pinned independent checkout, the
established lab Python environment, and at most two independent CPU processes.
The `run` command automatically summarizes and exports when all30 records exist.

## Failure, retries, and idempotent finish

`status` reports accepted IDs, incomplete IDs, untouched IDs, and native completions
awaiting acceptance. During a live queue, an incomplete unit may still be running;
never retry it until the supervisor and its children have exited. Retain all logs
and attempts. Completed accepted units are fully verified and skipped.

If an attempt has `EVAL_COMPLETE.json` but lacks its accepted record, invoke
`unit` with that same attempt to verify/reduce existing arrays; no rollout occurs:

```powershell
& $py -B -m analysis.evaluate_shared_maddpg_selected unit --batch-root $batchRoot --id 7 --attempt a01 --device cpu
```

If a rollout is genuinely partial/failed and has no native completion, use a new
attempt and a reason, selecting only failed/unrun IDs. Example syntax:

```powershell
& $py -B -u -m analysis.evaluate_shared_maddpg_selected run --batch-root $batchRoot --ids 7,8 --attempt a02 --reason interrupted --device cpu --workers 2
```

IDs7/8 here are syntax examples, never instructions to rerun successful worlds.
If code/contract errors require a fix, report the exact error; do not patch the
running checkout or change the model, seed, reward, world list or tolerances.

For30 accepted records:

- PASS `analysis/a01/verification.json`: `export --attempt a01` verifies existing
  compact evidence and returns its absolute path and archive SHA256; missing ZIP
  is created. Existing ZIP content is checked against the immutable compact files.
- No `analysis/a01` directory: `summary --attempt a01` reduces all accepted arrays,
  then exports. No policy rollout occurs.
- Partial analysis directory: preserve it; use a new unused analysis attempt,
  e.g. `summary --attempt a02`. Native/accepted unit attempts remain unchanged.
- Repeating `summary` on a PASS attempt delegates to verified export. Repeating
  `run` after completion verifies accepted units and uses this same idempotent finish.

```powershell
& $py -B -m analysis.evaluate_shared_maddpg_selected summary --batch-root $batchRoot --attempt a02
& $py -B -m analysis.evaluate_shared_maddpg_selected export --batch-root $batchRoot --attempt a02
```

The latter export is optional after successful summary; it returns the digest
again without overwriting the archive. Do not create a new analysis attempt solely
because a verified ZIP already exists.

## Acceptance and interpretation

Require30 accepted policy-world-repeat units,150 platoon rows,1,500,000 scored
agent-slots and15,000 CAM deadline events. The75,000 warmup agent-slots are excluded.
`technical_status=PASS` does not depend on favorable performance.

The existing reducers retain raw sums/counts, linear-mW power, strict AoI>50,
CAM deadline successes, reward/task/global components and trusted V2I events.
Scene-first pools each world/platoon before selecting that world's own maximum
AoI and minimum CAM; their worst platoons may differ. World distributions/sample
SD live in `world_endpoints.csv` and `seed_endpoints.csv`. Legacy pooling is exported
separately. Training-seed SD, training-seed t intervals, and action-repeat SD are
undefined for this one selected seed and one repeat; they remain empty.

`analysis/<attempt>/core_evidence.zip` contains protocol, all six candidate
training scores/curves/configs/completions, selected checkpoint digest/metadata,
unit completions/provenance, sum/count CSVs, world/condition summaries, verification,
process records and bounded log excerpts. Weights, replay and NPZ are excluded.
The export command prints the ZIP's SHA256; no checksum manifest is created.
Old four-model best100 results can be compared read-only afterward as selected
model cases, without adding them to this batch's30-unit acceptance counts.
