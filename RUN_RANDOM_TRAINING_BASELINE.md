# Uniform random baseline in the training lifecycle

Independent CPU entry: `python -m analysis.run_random_training_baseline`.
It imports configuration and the existing `PaperEnviron`, not PyTorch, actors,
critics, optimizers, replay, evaluators, or training/resume machinery. It works
from the committed snapshot alone and does not need historical models or NPZ.

## Fixed experiment

- One condition, `uniform_random`; environment seeds **12,13** only.
- Each seed: one cold `reset_world(seed)`, followed by `start_episode(0..2499)`
  and 100 native environment steps per episode. No warmup exclusion. No reset
  of the world/AoI every episode. The environment resets CAM demand/deadline and
  advances mobility/channels exactly as the training runner does.
- Environment/business source: `configs/training100_fresh_sources.json`, the
  same committed configuration bundle used to derive Training100-best. Its four
  model templates are checked for identical `deployment_contract.ENV_FIELDS`,
  and all 24 recorded historical source configuration identities are verified.
  The template budget500 is replaced with2500, matching Training100; this is
  **not** an evaluation of the old D1 50k policies. Algorithm/network fields in
  `config.resolved.json` identify the source template and are inert in this entry.
- Verified P5/N4/K3/gap25, 100slots, 1ms/slot, 32kbit CAM, continuous 1–30dBm,
  unchanged AoI cap, rewards, mobility and channel update logic.
- Each leader and slot: independent discrete-uniform RB and mode; independent
  `u~Uniform[0,1)`; power `1+29*u` dBm. This is **not** uniform mW, fixed max
  power, the PPO initial Beta distribution, or an untrained neural network.
- The native environment accepts normalized actions. Categories are encoded at
  their bin centers, not sampled as arbitrary continuous categories. Power is
  encoded as `2*u-1`. Native decoding and actual returned RB/mode/power are checked
  every slot, preventing categorical bias or a second power mapping.
- Action RNG: `Generator(PCG64(SeedSequence([seed,0x52414E44,1])))`; per slot draw
  `integers(K,size=P)`, `integers(n_modes,size=P)`, then `random(P)`. Never draw
  actions from `environment.rng` or global NumPy state. RNG definition/states and
  compact exogenous probes are saved.

The same environment code/config, seed and lifecycle permit exogenous pairing;
small tests compare different actions and exact mobility/channel/RNG states.
Historical formal training trajectories have **not** been matched. Same seed
does not alone establish such a match, and closed-loop observations/AoI need
not be equal.

## Run on the laboratory Windows machine

Wait for existing training/evaluation tasks to finish. Do not modify their
checkout. Use a clean detached checkout pinned to the delivered commit, and the
existing `.envs/aoi_cuda310/python.exe`. The entry requires Python3.10 and
NumPy1.23.5; it requires no CUDA or GPU and does not alter the environment.

PowerShell, from that checkout (`$sha` is the delivered full commit):

```powershell
$root='E:/Jinnan_Edisk/Research/Parvini_Reproduction_core'
$py=Join-Path $root '.envs/aoi_cuda310/python.exe'
$study=Join-Path $root 'AoI-Reproduction-diagnostics/actor-sharing-study'
$batchId='random_'+(Get-Date -Format 'yyyyMMdd_HHmmss')
$batch=Join-Path $study "Random-training-baseline/P5_N4_gap25/$batchId"
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& $py -B -m analysis.run_random_training_baseline init --study-root $study --batch-id $batchId --expected-commit $sha
if($LASTEXITCODE -ne 0){throw 'init failed'}
& $py -B -m analysis.run_random_training_baseline preflight --batch-root $batch
if($LASTEXITCODE -ne 0){throw 'preflight failed'}
& $py -B -u -m analysis.run_random_training_baseline run --batch-root $batch
if($LASTEXITCODE -ne 0){throw 'run/verification failed; preserve attempts and report'}
```

This runs the two seeds serially in one CPU process, then validates, summarizes
and exports. Record the printed batch path; never repeat `init` on an existing
batch. Keep checking progress/exit status until completion or an explicit failure.
The per-run `run.log` prints progress at episode1 and each100 episodes. No pilot,
training, frozen-world evaluation, or sbatch is involved.

Read-only/preflight commands do not roll out the environment. For a separate
small smoke run, add `--smoke-episodes 3` to `init` and use the automatically
separate `Random-training-baseline-smoke/P5_N4_gap25/<id>` output root. This flag
accepts only1–10 episodes and marks every completion/verification as smoke.
Smoke results cannot satisfy formal counts. No smoke is required before the
formal run if the delivered tests and preflight pass.

## Outputs and definitions

Each seed/attempt has `config.resolved.json`, `STARTED.json`, `COMPLETE.json`,
`episode_metrics.csv`, `episode_platoon_metrics.csv`, compact exogenous probes
and its log. Accepted results are in `records/seed12.json`, `seed13.json`.

`analysis/a01/` contains:

- Combined episode CSV (5,000 rows) and episode×platoon CSV (25,000 rows).
- `summary.csv`: per-seed full-run, first100, episodes401–500, last100 windows.
- `summary_across_seeds.csv`: mean and **sample SD** of the two seed summaries.
- `episode_across_seeds.csv`: episode-wise two-seed means and sample SD, with
  individual values retained in the episode CSV.
- `verification.json`, concise `REPORT_CN.md`, `core_evidence.zip`.

AoI is the native post-step `aoi_ms`, summed with its slot denominator. CAM is
one deadline event per episode/platoon: `success` at the **last slot**, checked
against zero remaining demand. It is not the average of success across slots.
Platoon IDs are zero-based. `cam_completion_percent=100*successes/trials`.

Task1/task2/global reward are cast to the same float32 logged streams as training,
then reduced in float64. Combined reward is `task1+task2+global` for each platoon,
matching the Training100 curve/selection convention (global is included for each
platoon). Power in mW is computed **per slot** as `10^(logged_dBm/10)` before
averaging; it is not the conversion of mean dBm. Reward/power/AoI sums and counts
are all retained. Additional strict AoI>50, mode0 V2I attempts, and post-step AoI≈1
reset counts are descriptive and do not explain CAM completion causally.

Two seeds are two descriptive replicates; slots, platoons and episodes do not
increase independent training n. No significance tests, stars, performance
thresholds, or selection. Missing smoke windows are empty, not zero.

Formal technical acceptance:

| Quantity | Expected |
|---|---:|
| Accepted seeds | 2 (12,13) |
| Seed×episode rows | 5,000 |
| Seed×episode×platoon rows | 25,000 |
| Environment slots | 500,000 |
| Agent slots | 2,500,000 |
| CAM deadline events | 25,000 |
| Optimization updates / frozen evaluations | 0 / 0 |

The ZIP includes CSV, protocol, source snapshots, accepted/completion records,
compact probes and report; no policy, checkpoints, raw slot trajectories, full
logs or SHA256 manifest. ZIP contents, CRC, row counts and correspondence to
the local evidence files are verified.

## Failure and duplicate protection

Only one process can hold `RUNNING.lock`. Do not delete an active or stale lock
without checking the recorded PID and explaining the interruption. A failed
attempt is retained; the entry never resumes partial rollout or silently
overwrites it. Completed accepted units are revalidated and skipped. A complete
native run missing only its accepted record is verified without rerunning it.
Conflicting completed attempts are a blocker, not an opportunity to choose one.

After an understood technical failure, an explicit retry is:

```powershell
& $py -B -u -m analysis.run_random_training_baseline run --batch-root $batch --seeds 12 --attempt a02 --reason 'documented technical failure'
```

Use only the failed seed(s); never retry for better performance. A failed summary
directory is retained; CPU-only regeneration uses:

```powershell
& $py -B -m analysis.run_random_training_baseline summary --batch-root $batch --attempt a02
```

Do not change seeds, rewards, payload, caps, action distributions or tolerance.
An unchanged fully completed run/summary can be invoked again for verification
and is skipped without creating duplicate rollout or rewriting its evidence.

## Local tests

`python -B -m pytest -q tests/test_random_training_baseline.py tests/test_environment.py tests/test_metrics.py`

Tests cover source mapping, action distributions/decoding/RNG, exogenous pairing,
native small-budget lifecycle, training-metric regression, deadline denominators,
failure/locking/retry, idempotence, source/CSV corruption rejection, windows and
two-seed SD, compact ZIP and importing without PyTorch/learning modules. Formal
2×2500 episodes are deliberately not run locally.
