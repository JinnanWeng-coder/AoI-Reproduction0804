# Selected Shared MAPPO-TDec best100: SINR diagnostic

Read-only diagnostic, no training. The previously accepted training-only winner is
seed13, best100 window2010–2109, checkpoint episode2109/version421, training commit
`04606e15f26f316728430e53a0e0d1e225bdb3e2`. The CLI recomputes the selection from
the accepted24 records and episode CSV, then checks the original payload/config.
It does not choose a policy using test SINR or change the old training contract.

Use a clean research/mappo checkout (or detached worktree) pinned to the delivered
commit. Default GPU inference (`cuda:0`, as in the lab selected-eval entry); original
stochastic policy/repeat0 RNG, world1001 by default,
5warmup+100scored episodes ×100slots. No new confirmation worlds. Optional
`--worlds 1001 1002` can select a unique subset of
the previously used1001–1030 only. Do not automatically expand the diagnostic.
Optional `--device cpu` is supported but changes the stochastic sampling stream:
CPU/GPU diagnostics must not be claimed bitwise equivalent to each other. There
is no automatic fallback from GPU to CPU. Same-device logging on/off regression
is tested separately on CPU and CUDA where available.

PowerShell (replace `$sha` with the delivered commit):

```powershell
$base = 'E:/Jinnan_Edisk/Research/Parvini_Reproduction_core'
$py = "$base/.envs/aoi_cuda310/python.exe"
$training = "$base/AoI-Reproduction-diagnostics/actor-sharing-study/Training100-best/P5_N4_gap25/lab_20261017_012015"
$id = 'sinr_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& $py -B -u -m analysis.inspect_selected_mappo_sinr --training-batch $training --batch-id $id --expected-commit $sha --dry-run
if ($LASTEXITCODE -ne 0) { throw 'SINR dry-run failed' }
& $py -B -u -m analysis.inspect_selected_mappo_sinr --training-batch $training --batch-id $id --expected-commit $sha
if ($LASTEXITCODE -ne 0) { throw 'SINR diagnostic failed; keep output, do not overwrite or retrain' }
```

Output: `actor-sharing-study/SINR-diagnostic/P5_N4_gap25/<id>/`.
Only new diagnostic output is written. Existing checkpoint, accepted records and
all historical evaluation directories stay read-only. Failed attempts stay intact;
retries must use a new batchID. No skip/recovery policy is introduced here.

The logger reads executed-slot signal and interference+thermal-noise in mW inside
the native physics call, before `step()` renews fading. `SINR=S/(I+N)` is a linear
ratio; dB=`10log10(SINR)`. The native Shannon bits/slot are checked against this
ratio. There is exactly one physics computation per step, no extra action/RNG call.
Local regression tests check bitwise identical old arrays and external probes.

Each platoon-slot row includes world, phase, episode, slot, RB, mode, executed power,
and active V2I SINR or all three V2V receiver SINRs and their minimum. Inactive
mode fields are null/NA, not zero. V2V SINR is not CAM deadline success. Zero/negative
dB is valid. Episode/slot are1-based, platoon/RB0-based, follower1..3.

`native/sinr.log` and `native/sinr_slots.csv` cover warmup and scored phases separately;
the console shows the first scored episode by default (`--console-episodes 0`
silences it;100 prints all scored episodes). For one world:
2500warmup+50000scored platoon-slot rows; the number of active links of each mode is
policy-dependent. `native/sinr_summary.json` includes scored-only link-slot weighted
count, mean linear, mean dB, dB quantiles and V2I threshold. Mean dB is not dB of mean
linear. Three V2V receivers are counted separately in receiver-weighted statistics.
These slots/worlds are not independent training repetitions.

Successful completion writes `verification.json` and `sinr_logs.zip`, containing
only manifest/verification/definitions, full SINR log/CSV/summary and native
completion/provenance. Return this ZIP and a few console lines to Codex. Weights and
raw `native/metrics.npz` stay on the lab computer. No SHA256 manifest.

Local test command (synthetic policies/short physics only, not lab weights):
`python -B -m pytest -q -p no:cacheprovider tests/test_mappo_sinr.py`.
