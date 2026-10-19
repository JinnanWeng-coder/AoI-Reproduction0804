# All-seed best100 laboratory evaluation

This separate contract retains all six training seeds per model. It does not
change the selected-seed 300-unit batch or rerun its 150 accepted best100 units.
Five execution conditions × six seeds × worlds1001–1030 × repeat0 = 900 units:
150 read-only references and 750 new evaluations. There is no new training.
The selected batch must finish all300 units and its summary first.

Use the existing laboratory Python environment, a clean pinned checkout, and no
other active queue. `$sha` must be the exact delivered commit; do not pull while
running. PowerShell commands (stop on every nonzero exit):

```powershell
$root='E:/Jinnan_Edisk/Research/Parvini_Reproduction_core'
$py=Join-Path $root '.envs/aoi_cuda310/python.exe'
$repo=Join-Path $root 'AoI-Reproduction0804'
$study=Join-Path $root 'AoI-Reproduction-diagnostics/actor-sharing-study'
$training=Join-Path $study 'Training100-best/P5_N4_gap25/lab_20261017_012015'
$old=Join-Path $study 'Training100-selected-eval/P5_N4_gap25/selected_20261018_183228'
$batchId='allbest_'+(Get-Date -Format 'yyyyMMdd_HHmmss')
$batch=Join-Path $study "Training100-all-seeds-eval/P5_N4_gap25/$batchId"
Set-Location -LiteralPath $repo
$env:PYTHONDONTWRITEBYTECODE='1'
& $py -B -m analysis.evaluate_training100_all_seeds init --training-batch $training --reuse-batch $old --batch-id $batchId --expected-commit $sha
if($LASTEXITCODE -ne 0){throw 'init failed'}
& $py -B -m analysis.evaluate_training100_all_seeds preflight --batch-root $batch --device cuda:0
if($LASTEXITCODE -ne 0){throw 'preflight failed'}
& $py -B -u -m analysis.evaluate_training100_all_seeds run --batch-root $batch --device cuda:0 --workers 2
if($LASTEXITCODE -ne 0){throw 'queue failed; preserve attempts and report'}
```

Preflight loads24 actor payloads, validates all candidate records and training
curves, and rechecks150 existing native completions/arrays/CSV against the old
commit and complete unit keys. It writes only new reference records. No weights
or NPZ are copied. Each new record retains the source evaluation commit. Reused
rows keep their old implementation commit, with `data_role=reused` and explicit
old batch/unit references. final units are excluded. Accepted a02 training runs
are resolved from records, not guessed from folder listings.

The run command uses the existing bounded child-process queue and automatically
summarizes/exports on completion. Max2 evaluation children. Failures stop further
scheduling; in-flight children finish. Do not delete live locks or partial runs.
For a completed native rollout with post-check failure, `unit --batch-root $batch
--id ID --attempt a01 --device cuda:0` rechecks arrays without rollout. For a truly
incomplete attempt, use `run --ids IDS --attempt a02 --reason REASON ...` only after
technical review. Performance is never a retry criterion. No sbatch/pilot needed.

`summary --batch-root $batch --attempt a02` is CPU postprocessing only and writes a
new analysis attempt. Expected4500 platoon rows,45M agent-slots,450k CAM deadlines,
30 condition×seed endpoints, six training replicates per condition. Scene-first
and legacy remain separate; own AoI/CAM extrema are selected per world. World SD
does not inflate training n. Descriptive95% t intervals use df5 and paired effects
match training seeds. repeat variability is undefined (only repeat0).
Worlds are already used exploratory worlds, not newly unseen confirmation worlds.

`analysis/<attempt>/core_evidence.zip` contains CSV/JSON/report, source metadata,
native completion/provenance and reference records. No NPZ, weights, full logs or
SHA256 manifest. ZIP integrity, unique members and merged CSV row count are checked.
