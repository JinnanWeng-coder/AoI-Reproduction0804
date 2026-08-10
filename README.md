# AoI-V2X reproduction

This repository contains the maintained reproduction implementation for the
Modified MADDPG algorithms in the AoI-V2X study. Active runs use one baseline:

- Polyak coefficient `tau=0.005`
- synchronous joint global-actor update
- slow fading update every episode
- fixed training exploration noise `0.3`
- `500` episodes and `100` slots per episode for full runs
- lightweight `policy_only` artifacts by default

The old profile split and its duplicated source tree are intentionally absent.
The exact pre-refactor repository remains available at Git tag
`pre-reproduction-baseline-624e84c`.

## Source layout

```text
aoi_v2x_reproduction/
  algorithms/modified_maddpg/  actor, critics, learner, replay
  envs/platoon.py              V2X platoon environment
  runtime/                     training, evaluation, metrics, checkpoints
  cli.py                       command-line entry point
  config.py                    scenarios and the single baseline
analysis/                      result summarizers and plotting utilities
configs/reproduction_baseline.yaml
hpc/                           current Slurm launchers
scripts/                       smoke, preflight, and deferred matrix helpers
tests/                         algorithm and runtime regression tests
```

`Main.py` is a compatibility wrapper. New commands may use either
`python Main.py` or `python -m aoi_v2x_reproduction`.

## Quick checks

```bash
python -m aoi_v2x_reproduction --scenario p05_n04_g25 --dry-run
python scripts/preflight_network.py --scenario p05_n10_g25 --device cpu
python -m pytest -q
bash scripts/run_smoke.sh cpu
```

Algorithm 1 is selected with `--algorithm modified_maddpg`; Algorithm 2 with
task decomposition is the default `modified_maddpg_tdec`.

## Artifact policy

Early training writes metrics, configuration, provenance, completion metadata,
and one final actor-only `policy_final.pt`. It does not create replay snapshots
or periodic/best/latest checkpoints. Use `--checkpoint-mode none` to omit even
the final policy.

`--checkpoint-mode resumable` is reserved for a later, explicitly planned
resume or held-out evaluation stage. It stores optimizer, replay, environment,
RNG, and selection state and therefore uses substantially more disk space.

The baseline values are not exposed as routine CLI switches. A different tau,
slow-update cadence, or global update mode is a new ablation and should be added
explicitly in code rather than silently mixed into baseline runs.

## Historical results

Analysis utilities retain read compatibility with earlier result bundles whose
resolved configuration used older profile names. That compatibility does not
make those profiles available for new training.
