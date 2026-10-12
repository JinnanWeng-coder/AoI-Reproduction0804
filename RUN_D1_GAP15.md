# D1 gap25 → gap15 frozen deployment extension

Only this extension is executed: no training, original pilot/main/gap5/25/35 rerun,
new worlds, new lock, Combined, or additional action repeats. No figure rendering.
This is observational transfer on old worlds, not a new-world confirmation.

## Fixed matrix and provenance

Four conditions in array order: shared_tdec_stochastic,
independent_tdec_stochastic, algorithm2_noise0, algorithm2_noise0p3.
Within each condition: training seed8–13, then world213–218. All repeat0.
`unit_id = condition_index*36 + (training_seed-8)*6 + (world-213)`.
144 new evaluations, 72 MAPPO + 72 MADDPG, 18 actual gap25-trained final500 models.
The two MADDPG noise conditions share six models. MAPPO uses policy_final.pt;
MADDPG uses the original validated latest.pt. Sources/config/COMPLETE are copied
as metadata from old sources.json, never copied checkpoints or relabelled results.

The old read-only batch must have original implementation commit
75098b00714daf315a1d82298deb563209da9ac8:

`/eeedata/sgxjw2/Parvini-TVT2023-reproduction/AoI-Reproduction-diagnostics/actor-sharing-study/D1-frozen-deployment/P5_N4_gap25/deploy_repair_20261011T092046_29657`

New batches: `actor-sharing-study/D1-gap15-extension/<unique_batch_id>/`.
Initialization refuses an existing directory. New execution remains pinned to its
own clean checkout commit and source identity; only historical reads use historical
code provenance. The algorithm/config/environment package is unchanged.

## Initialize and submit

Use an independent detached checkout of the delivered implementation commit.
Explicitly export variables; keep that checkout fixed until all jobs finish.
`AOI_EXPECTED_COMMIT` must equal its full HEAD. Paths below exist on HPC, not Windows.
The final handoff prompt supplies the actual delivered full SHA.

```bash
export BASE=/eeedata/sgxjw2/Parvini-TVT2023-reproduction
export PYTHON_BIN=/eeedata/sgxjw2/conda_envs/aoi_cuda/bin/python
export DIAGNOSTICS_ROOT="$BASE/AoI-Reproduction-diagnostics"
export AOI_EVIDENCE_PACK="$BASE/tmp/evidence_pack_v0_20261011.zip"
export AOI_METADATA_ADDENDUM="$BASE/tmp/metadata_addendum_v1.zip"
export AOI_OLD_BATCH="$DIAGNOSTICS_ROOT/actor-sharing-study/D1-frozen-deployment/P5_N4_gap25/deploy_repair_20261011T092046_29657"
export AOI_EXPECTED_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
export AOI_ATTEMPT=a01 AOI_DEVICE=cuda:0 AOI_CONTRACT_MODULE=analysis.d1_gap15
unset AOI_TASK_OFFSET AOI_RETRY_REASON AOI_ACCEPT_EXISTING AOI_CHECK_ONLY AOI_GAP15_STAGE
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PROJECT_DIR"
export BATCH_ID="gap15_$(date -u +%Y%m%dT%H%M%S)_$$"
export AOI_BATCH_ROOT="$DIAGNOSTICS_ROOT/actor-sharing-study/D1-gap15-extension/$BATCH_ID"
cd "$PROJECT_DIR"
df -h "$DIAGNOSTICS_ROOT"
df -i "$DIAGNOSTICS_ROOT"
# Review account quota separately if quota is not available; do not assume unlimited.
"$PYTHON_BIN" -B -m analysis.d1_gap15 init --diagnostics-root "$DIAGNOSTICS_ROOT" --batch-id "$BATCH_ID" --old-batch "$AOI_OLD_BATCH" --evidence-pack "$AOI_EVIDENCE_PACK" --metadata-addendum "$AOI_METADATA_ADDENDUM"
"$PYTHON_BIN" -B -m analysis.d1_gap15 dry-run
"$PYTHON_BIN" -B -m analysis.d1_gap15_jobs --batch-root "$AOI_BATCH_ROOT"
"$PYTHON_BIN" -B -m analysis.d1_gap15_jobs --batch-root "$AOI_BATCH_ROOT" --submit
```

The submitter always defaults to dry-run. One submission creates five afterok jobs:

1. CPU preflight: 18 real payloads, gap15 config compatibility, old inputs/identities,
   historical cumulative-table reconciliation, raw NPZ-size disk estimate.
2. GPU formal first16: IDs `0-1,6-7,36-37,42-43,72-73,78-79,108-109,114-115`.
3. CPU first-check: 16/80/800000 slots/8000 CAM deadlines, frozen/source/native
   validation and exact exogenous probes within gap15, raw capacity recheck.
4. GPU remaining128: the exact complement within 0–143. Same seed/world/repeat
   matrix, not conditional on performance. First16 are counted once in final144.
5. CPU finalize: full summary → runtime/sacct → real ZIP → integrity/extract/readback.

GPU arrays are serially dependent: whole-batch GPU concurrency at most6.
Defaults Q10 / gpu:l20:1 / 4 CPU / 16G / 2h; CPU 4 CPU / 16G / 3h, no GPU.
Options: `--partition --account --concurrency --cpus --mem --gpu-time --cpu-time
--gres --device`. An account is not invented; use the actual existing allocation.
Submission commands/job IDs are persisted immediately under protocol/submission_*.
A partly submitted workflow cannot be blindly resubmitted.

## Protocol and aggregation

Per unit: target p05_n04_g15, world cold once, sequential_warm, 5 warmup then100
scored episodes×100slots×5platoons. Same original MAPPO stochastic policy RNG and
MADDPG normalized-action noise0/0.3 RNG, repeat0 historical rule. Frozen parameters.
Warmup is not in scored arrays; do not delete the first five saved episodes again.
Historical MADDPG paper_faithful profile is preserved, active training validation
is not relaxed. Geometry differs across gaps, so probes are compared within gap15,
not between gaps15/25. No new environment or policy implementation.

Scene-first: pool actual sum/count per world/platoon, select own max AoI / min CAM,
then mean over six worlds, then over six training seeds. AoI/CAM may select different
platoons. Legacy endpoints also retained. Single-repeat execution SD is undefined,
not zero. World/slot/platoon are not independent training repetitions.
All seed summaries and paired changes use descriptive t95 df5, sample SD, n=6.
No significance selection, equivalence inference, or favorable-direction PASS gate.
Power uses per-slot mW. CAM fraction, percent, and percentage-point change are separate.
Gap means intra-platoon vehicle gap, bumper-to-bumper (metres).

Historical gaps5/35 use the accepted old gap_alignment CSVs, source identities and
completion records; gap25 comes from the exact same gap25-trained models and worlds
213–218 in the original evidence pack. Relative POSIX policy paths are normalized.
Some old MAPPO slot metadata is blank: 100 slots is verified from10000 slots/100
episodes per platoon and recorded as inferred, never written back to old metadata.
Missing historical reset fields stay missing. Old endpoints are recomputed from
cumulatives and reconciled at existing AoI1e-10/CAM1e-12 precision, no loosened tolerances.
Gap25 Shared reference ~8.94848ms/99.77778%, NOT main's different-world reference.
Those numbers are explanatory checks, never hardcoded computed outputs.

## Outputs and validation

Final `analysis/gap15/a01/verification.json` must PASS with144 new /0 reused units,
720 cumulative rows,7200000 scored agent-slots,72000 CAM deadlines,144 world endpoints,
24 training-seed endpoints,18 models. Historical reused432 policy-worlds /2160 rows
are listed separately and are not part of the new144 count.

CSV levels include world_repeat_platoon, world_platoon_repeat_pooled,
world_endpoints, repeat_endpoints, seed_endpoints, condition_summary, legacy endpoints.
Fig.4 tables: fig4_seed_endpoints96, fig4_condition_means16, paired_seed_changes96,
paired_summary128 (16 groups×8 metrics), power96; gap15-specific paired24/32.
Historical CSVs/inventory allow each reduction to be checked. No plots generated.

CPU commands used by the wrappers, and available for diagnosed partial-stage recovery:

```bash
"$PYTHON_BIN" -B -m analysis.d1_gap15 preflight --batch-root "$AOI_BATCH_ROOT"
"$PYTHON_BIN" -B -m analysis.d1_gap15 first-check --batch-root "$AOI_BATCH_ROOT"
"$PYTHON_BIN" -B -m analysis.d1_gap15 summary --batch-root "$AOI_BATCH_ROOT" --attempt a01
"$PYTHON_BIN" -B -m analysis.d1_gap15 runtime --batch-root "$AOI_BATCH_ROOT" --attempt a01
"$PYTHON_BIN" -B -m analysis.d1_gap15 export --batch-root "$AOI_BATCH_ROOT" --attempt a01
"$PYTHON_BIN" -B -m analysis.d1_gap15 check-zip --zip "$AOI_BATCH_ROOT/analysis/gap15/a01/core_evidence.zip"
```

Do not run these heavy stages on the login node. Normally use the five-job workflow,
not the standalone sequence in addition to finalize. Summary/runtime/archive refuse
overwrites. Re-running preflight/first-check revalidates immutable data and keeps
the original PASS record. A failed summary needs a fresh analysis attempt with all
existing accepted evaluations preserved; no policy rerun. If only runtime/export
failed after summary, run only the missing stages, not summary again.

The ZIP directly includes protocol/config/source snapshots, job submissions,
completion/provenance, compact CSV/JSON/README/fields/runtime/sacct and small log tails.
No NPZ, model, inner tarball, full logs or SHA256 manifest. ZIP testzip and actual
temporary CSV/JSON extraction/readback are performed automatically. Full raw arrays
remain under evaluations/gap15 on HPC. Runtime snapshot is taken inside the packaging
job: its final exit must additionally be checked by Grok, not misreported as completed.
Missing MaxRSS is null; missing quota is unavailable, not zero usage.

## Monitoring and failures

Grok must continue monitoring squeue/sacct and accepted counts through final ZIP,
not stop after submission. Only touch this batch's jobs. Preserve all failures and
submission records. A dependency-blocked descendant is not a reason to resubmit144.
Infrastructure failures: recover exact failed/unrun IDs, same protocol, fresh attempt
(a02 etc) with reason; accepted units are revalidated and skipped across attempts.
Use the same GPU sbatch with absolute unit IDs, no offset. Native EVAL_COMPLETE present:
first use `AOI_ACCEPT_EXISTING=1 AOI_ATTEMPT=<its original attempt>` to revalidate/accept
arrays, never evaluate it again. Retry jobs need submission JSON records with job_id,
stage, actual command and reason, so runtime accounting includes them. Reset these
environment flags before any other stage. After recovered first16, re-run first-check
then submit only the remaining/unrun stage and finalize with afterok dependencies.
If code, source, frozen, protocol or statistical validation fails, stop and report to
Codex; do not edit execution checkout, drop seeds, change worlds/noise, or relax checks.

Local tests use synthetic data/fake environment and read-only evidence tables only.
Real model loading, GPU native arrays, quota, accounting and ZIP from144 rollouts
remain HPC responsibilities. After this extension finishes, stop.
