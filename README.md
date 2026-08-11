# AoI-V2X reproduction

This repository contains the maintained reproduction implementation for the
Modified MADDPG algorithms in the AoI-V2X study and an exploratory MAPPO
extension. Modified MADDPG runs use one baseline:

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
  algorithms/mappo/            hybrid actors, central critic, GAE/PPO rollout
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
task decomposition is the default `modified_maddpg_tdec`. The exploratory
extension is selected with `--algorithm mappo`.

The first MAPPO implementation keeps one actor per platoon, uses categorical
RB and transmission-mode heads, a Beta power head, and a centralized critic
that predicts one state value per agent. It uses the same per-agent reward
decomposition and unchanged V2X environment. Its first confirmation wave is
training-only at P=5, N=4, gap=25 with seeds 8--13. Polyak tau, external action
noise, and the MADDPG global-actor update mode do not apply to MAPPO; `tau=0.005`
is retained only in the shared resolved-config baseline metadata.

## Artifact policy

Early training writes metrics, configuration, provenance, completion metadata,
and one final actor-only `policy_final.pt`. It does not create replay snapshots
or periodic/best/latest checkpoints. Use `--checkpoint-mode none` to omit even
the final policy.

The first MAPPO policy artifact follows the same lightweight rule: it contains
only the five actor state dictionaries, not the central critic, optimizers,
rollout data, or RNG state. MAPPO resume and held-out evaluation are deferred.

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
