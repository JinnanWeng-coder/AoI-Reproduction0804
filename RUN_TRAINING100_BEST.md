# Training100 best + final: 2500-episode per-seed selection

This is training-artifact preparation, not a deployment evaluation. No validation,
test-world rollout, new test worlds, or performance-based seed exclusion is authorized.

## Scope and method facts

Four configurations, seeds 8–13: Shared MAPPO-TDec, Independent MAPPO-TDec,
Shared MAPPO-Combined, Modified MADDPG-TDec. IDs 0–5, 6–11, 12–17, 18–23 respectively.
Every run is 2500 episodes ×100 slots =250000 environment steps. Fresh total:
24 runs, 6000000 environment steps. MADDPG test noise 0/0.3 refers to the same six
models, not twelve training runs; training exploration remains 0.3.

For EACH training seed separately, R_e is the float64 mean over slots/platoons of
`task1_step + task2_step + global_step[:,None]`, using the existing float32 logged
streams (global weight 1). S_e is the mean of R_(e-99)..R_e. Start at episode 100;
replace only on strict improvement; ties retain the earliest candidate. Always
finish episode 2500 and keep separate `policy_best_training.pt` / `policy_final.pt`.

MAPPO best is captured after collecting the episode, BEFORE a pending PPO update:
episode e uses version floor((e-1)/5); final is after update 500. Shared total actor
Adam steps=5000; independent=25000 (5000 per actor). MADDPG best is after that
episode's online updates; version counts actual delayed actor-update rounds, not
the sum across its five actors. The selected 100-episode window spans behavior
versions; its reward is NOT a frozen-policy evaluation score. Ledger/artifacts
record window, score, version range and save timing. No across-seed peak selection.

Saving preserves Python/NumPy/Torch/CUDA RNG states and does not change learning
order. Existing runner defaults are unchanged. Package changes relative to
`2ce572b9c7fe487fc7a0898cdd8330fdab04bc4a` are limited to runner hooks and the new
`runtime/training_selection.py`; algorithms, environment, reward, config defaults,
normalization and schedulers are unchanged. Combined/TDec are two value-learning
configurations, not merely two critic network architectures.

`load_training_policy(path)` validates completed artifacts and metadata;
`load_training_actors(path, device="cpu")` additionally loads strict actor
architectures without an environment/rollout and preserves RNG. Import them from
`aoi_v2x_reproduction.runtime.training_selection`. These are NOT resumable
checkpoints. Existing final-500 evaluator contracts have not been repurposed to
accept selected policies: a later explicitly authorized evaluation must use their
real selected episode/version rather than relabeling them as final-500.

## Historical reuse and read-only audit

The authoritative 24 source models/configs come from the accepted D1 batch below.
`init` snapshots its source ledger; `audit` checks payloads and reads only the
three training reward arrays, plus historical B1 final-policy/config metadata.
Source profiles (including historical `paper_faithful`) and all effective settings
are retained. This narrow research wrapper calls the existing runner directly;
it neither relabels profiles nor relaxes `validate_config` for general training.
Missing/None historical MAPPO sharing is recorded and interpreted as false.
Derivation allows only budget, new run name/output root and this false normalization.

Reusable: old configs, curves and final weights as historical references. They are
not copied into new accepted runs. B1's historical MAPPO implementation saved only
final actors: episode CSV cannot restore an earlier best policy, so the 18 MAPPO
models require fresh training unless separately verified extra snapshots exist.
The audit stops on an unexpected B1 training100-best artifact rather than overwrite it.

For MADDPG, `latest.pt` may contain optimizer/replay/environment/RNG state. Presence
of those keys or a periodic actor candidate is NOT equivalent-resume proof. The
earlier best must also be recoverable, configs/state/order must be equivalent, and
the completed-run guard must not be bypassed. This minimal delivery implements no
legacy fork-resume; all six MADDPG models use the authorized conservative fresh-run
fallback. Audit reports full-state-key presence, prefix best and periodic candidate
availability, but never claims continuation is impossible in principle. If new
qualified reuse evidence changes the premise, report it for engineering review;
do not improvise continuation, rename old best/final or change old completion files.

## HPC commands

Use a clean pinned detached worktree at the delivered commit, never update it while
jobs run. Substitute the actual delivered SHA for `IMPLEMENTATION_SHA`. Run all
heavy audit/check/summary/export stages on CPU compute nodes, not the login node.

```bash
export BASE=/eeedata/sgxjw2/Parvini-TVT2023-reproduction
export AOI_EXPECTED_COMMIT=IMPLEMENTATION_SHA
export PROJECT_DIR="$BASE/training-checkouts/$AOI_EXPECTED_COMMIT"
export PYTHON_BIN=/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python
export STUDY_ROOT="$BASE/AoI-Reproduction-diagnostics/actor-sharing-study"
export OLD_BATCH="$STUDY_ROOT/D1-frozen-deployment/P5_N4_gap25/deploy_repair_20261011T092046_29657"
export B1_RUNS="$STUDY_ROOT/B1-budget-sensitivity/P5_N4_gap25/budget_20260926_123219/training/a01/runs"
export BATCH_ID="train100_$(date -u +%Y%m%dT%H%M%S)_$RANDOM"
export BATCH_ROOT="$STUDY_ROOT/Training100-best/P5_N4_gap25/$BATCH_ID"
export AOI_ATTEMPT=a01 PYTHONDONTWRITEBYTECODE=1
unset AOI_TASK_OFFSET AOI_RETRY_REASON AOI_ACCEPT_EXISTING AOI_CHECK_ONLY
cd "$PROJECT_DIR"
test "$(git rev-parse HEAD)" = "$AOI_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
df -h "$STUDY_ROOT"
df -i "$STUDY_ROOT"
bash -n hpc/training100_common.sh hpc/aoi_training100_train.sbatch hpc/aoi_training100_cpu.sbatch
"$PYTHON_BIN" -B -m analysis.training100_contract init \
  --study-root "$STUDY_ROOT" --batch-id "$BATCH_ID" --old-batch "$OLD_BATCH" --b1-runs "$B1_RUNS"
"$PYTHON_BIN" -B -m analysis.submit_training100 --batch-root "$BATCH_ROOT" --stage audit --cpus 4 --mem 32G --time 03:00:00
"$PYTHON_BIN" -B -m analysis.submit_training100 --batch-root "$BATCH_ROOT" --stage audit --cpus 4 --mem 32G --time 03:00:00 --submit
```

Each submit prints JSON including job_id and writes `protocol/submission_*_<stage>.json`.
Record the actual ID. Wait for sacct COMPLETED/0:0, read `historical_audit.json` and
its 24 rows. Missing or incompatible source payloads stop the workflow, not trigger
training substitutes. After audit PASS confirms the conservative fresh-run premise:

```bash
"$PYTHON_BIN" -B -m analysis.training100_contract approve-fresh --batch-root "$BATCH_ROOT"
"$PYTHON_BIN" -B -m analysis.submit_training100 --batch-root "$BATCH_ROOT" --stage all --concurrency 6
"$PYTHON_BIN" -B -m analysis.submit_training100 --batch-root "$BATCH_ROOT" --stage all --concurrency 6 --submit
```

Per the revised authorization, there is NO pilot: submit one `0-23%6` array, all
24 full-budget runs queued together. Each task requests one L20, so at most six GPUs
are occupied by this array. GPU defaults: Q10, 8 CPUs, 32G, 8h/task. Account,
partition/time/memory/CPU/concurrency are CLI options; concurrency must stay ≤6.
`approve-fresh` records `execution_mode=direct24`, `pilot_required=false`; neither
submission nor individual run execution requires pilot_verification in this mode.
Source audit, config/source identity, 2500-episode budget, strict policy loading,
per-seed score recomputation and completion checks remain mandatory.

After all 24 tasks succeed, dry-run and submit CPU `--stage summary`, then
`--stage export`, with the same CPU options as audit. `--dependency
afterok:<actual_job_id>` is supported. Legacy pilot/main entry points are retained
only for compatibility; do not use or combine them with `all` in this batch.
Duplicate or mixed direct24/pilot-main submissions are rejected. Check both job
exit AND JSON verification, never just sbatch acceptance. No result-dependent
restart or hyperparameter adjustment.

## Acceptance, outputs, failures

- `records/train_00..23.json`: 24 unique accepted model×seed runs, all 2500 episodes.
- Separate 24 best +24 final policies; full strict actor loading and selection recheck.
- MAPPO 500 PPO updates, actual shared/independent actor counts/steps; MADDPG delayed
  actor version checked against the unchanged learner schedule/batch size.
- `analysis/a01/training_episode_curves.csv`: 60000 rows; environment_steps=e×100.
- `selected_policies.csv`: 24 rows, independent per-seed windows/scores/versions.
- `verification.json`: technical PASS, accepted=24, fresh=24, training steps=6000000,
  new evaluations=0; claim_support=NOT_ASSESSED.
- `runtime_report.json`, `sacct.csv`: actual commands/job IDs/resources/raw bytes;
  empty MaxRSS stays null. Export-job final state must be checked after snapshot.
- `analysis/a01/core_evidence.zip`: compressed CSV/JSON/README/source/completion
  records and ≤16KiB tail per Slurm log. ZIP integrity and 24/60000 CSV row readback
  are automatically checked. No SHA256 manifest, policies, NPZ, full trajectories,
  full logs, replay, or cache. Raw artifacts remain on HPC.

Accepted run skip requires full key/source/commit/artifact verification. COMPLETE
without the post-check record is re-verified/accepted, never silently retrained.
Partial attempts are preserved; use a new attempt (`a02` etc.) plus reason only for
the failed/missing IDs. The supported runner does NOT resume partial training: a
retry starts from scratch and its extra compute must be reported separately.
For selective recovery call the train sbatch with `--array=<missing IDs>%6`, explicit
exports and fresh attempt/reason; save its real command/job ID as a submission JSON
under protocol so runtime accounting includes it. Do not repeat the full main.
Re-run a failed summary in a new analysis attempt, not by overwriting outputs;
after summary PASS, failed packaging alone does not authorize retraining.
On code/source/equivalence errors preserve the scene and return to engineering;
do not hotpatch a running checkout, loosen checks, or alter old artifacts.

Local synthetic tests use fake environment/learners only, including full-budget
artifact construction, strict loaders, RNG/weight/array equality with the opt-in
off/on, boundary/tie/per-seed rules, gates, summary and ZIP readback. They do not
establish real checkpoint compatibility, GPU equivalence, or HPC completion.
