# Remote execution runbook

This runbook is for the current `Modified_MADDPG_with_TDec` (paper Algorithm 2)
implementation. It deliberately separates validation from final testing. Do
not run `final_test` until the validation artifacts have been reviewed and the
analysis choices are frozen.

The current repository can run the Algorithm 2 grid. It cannot by itself
reproduce the paper's complete cross-algorithm Fig.4/Fig.5 conclusions because
`Modified_MADDPG`, `MADDPG_FDec`, and `DDPG` are not implemented here.

For the UNNC `Q10` eight-L20 Slurm workflow, use `hpc/README_HPC.md`. The matrix
runner supports deterministic `--shard-count` and `--shard-index` operational
partitioning; the default remains the complete, strictly checked 48-cell
matrix.

## 1. Host and checkout gate

Complete `ENVIRONMENT_INSTALL.md`, then verify:

```bash
test "$(git branch --show-current)" = "main"
test -z "$(git status --porcelain --untracked-files=all)"
python -m pytest -q
python scripts/preflight_network.py \
  --scenario p05_n10_g25 --batch-size 64 --device cuda:0
```

Choose external, durable locations:

```bash
export AOI_RUN_ROOT=/data/aoi-v2x/runs
export AOI_HANDOFF_ROOT=/data/aoi-v2x/handoff
mkdir -p "$AOI_RUN_ROOT" "$AOI_HANDOFF_ROOT/logs"
```

Keep the literal training identity stable. Matrix recovery treats `device`,
`output_root`, `run_name`, and `checkpoint_every` as part of the exact saved
configuration. `auto` and `cuda:0` are different identities even if both select
the same GPU. Manual evaluation can use a different runtime device, but the
matrix intentionally keeps one device string throughout.

## 2. One-cell formal validation pilot

This is a real formal cell, not a smoke test: 500 training episodes x 100 slots,
then six held-out validation worlds with five warm-up and 100 scored episodes
each.

```bash
export AOI_PILOT_NAME=paper_faithful_p05_n04_g25_seed02

python Main.py \
  --profile paper_faithful \
  --scenario p05_n04_g25 \
  --seed 2 \
  --device cuda:0 \
  --output-root "$AOI_RUN_ROOT" \
  --run-name "$AOI_PILOT_NAME" \
  --checkpoint-every 5 \
  --scope train
```

If the job stops after `latest.pt` exists, resume with exactly the same identity:

```bash
python Main.py \
  --profile paper_faithful \
  --scenario p05_n04_g25 \
  --seed 2 \
  --device cuda:0 \
  --output-root "$AOI_RUN_ROOT" \
  --run-name "$AOI_PILOT_NAME" \
  --checkpoint-every 5 \
  --scope train \
  --resume "$AOI_RUN_ROOT/$AOI_PILOT_NAME/checkpoints/latest.pt"
```

If the job stops after the run was atomically initialized but before the first
checkpoint, reinitialize only that provenance-verified empty run (no `--resume`):

```bash
python Main.py \
  --profile paper_faithful \
  --scenario p05_n04_g25 \
  --seed 2 \
  --device cuda:0 \
  --output-root "$AOI_RUN_ROOT" \
  --run-name "$AOI_PILOT_NAME" \
  --checkpoint-every 5 \
  --scope train \
  --recover-empty-run
```

The empty-run gate rejects extra files, partial checkpoints, metrics, links,
wrong config hashes, or changed Git/source provenance. It never deletes or
overwrites them.

After training completes:

```bash
python analysis/audit_results.py \
  "$AOI_RUN_ROOT/$AOI_PILOT_NAME" --scope train

python Main.py \
  --profile paper_faithful \
  --scenario p05_n04_g25 \
  --seed 2 \
  --device cuda:0 \
  --output-root "$AOI_RUN_ROOT" \
  --run-name "$AOI_PILOT_NAME" \
  --checkpoint-every 5 \
  --scope validation \
  --eval-only \
  --eval-purpose validation \
  --eval-episodes 100 \
  --eval-seeds 201,202,203,204,205,206 \
  --resume "$AOI_RUN_ROOT/$AOI_PILOT_NAME/checkpoints/latest.pt"

python analysis/audit_results.py \
  "$AOI_RUN_ROOT/$AOI_PILOT_NAME" \
  --scope validation --require-eval
```

Copy the audit JSON and logs to the handoff directory and review learning
curves, raw AoI/success arrays, runtime, GPU memory, and numerical stability.
Proceed to the 48-cell grid only after this pilot passes.

## 3. Dry-run the exact remote matrix

`Main.py --dry-run --matrix` shows the logical 48 cells. The command below is
the authoritative remote dry-run because it resolves the actual CUDA device,
absolute output root, checkpoint cadence, recovery policy, and command hashes.

```bash
python scripts/matrix_runner.py \
  --profile paper_faithful \
  --device cuda:0 \
  --output-root "$AOI_RUN_ROOT" \
  --checkpoint-every 5 \
  --recover-empty-run \
  --stage all \
  --eval-purpose validation \
  --eval-episodes 100 \
  --eval-seeds 201,202,203,204,205,206 \
  --log-dir "$AOI_HANDOFF_ROOT/logs" \
  --report "$AOI_HANDOFF_ROOT/matrix-dry-run.json" \
  --dry-run
```

Confirm `matrix_count=48`, `unique_count=48`, the eight expected scenarios,
training seeds 2..7, and 48 unique train/eval/audit commands. The runner is a
restart-safe, single-process sequential orchestrator; it is not a multi-GPU
scheduler.

## 4. Execute in reviewable stages

Use a new report filename for every invocation; reports are never overwritten.

```bash
python scripts/matrix_runner.py \
  --profile paper_faithful --device cuda:0 \
  --output-root "$AOI_RUN_ROOT" --checkpoint-every 5 \
  --recover-empty-run --stage train \
  --log-dir "$AOI_HANDOFF_ROOT/logs" \
  --report "$AOI_HANDOFF_ROOT/matrix-train.json" --execute

python scripts/matrix_runner.py \
  --profile paper_faithful --device cuda:0 \
  --output-root "$AOI_RUN_ROOT" --checkpoint-every 5 \
  --recover-empty-run --stage eval \
  --eval-purpose validation --eval-episodes 100 \
  --eval-seeds 201,202,203,204,205,206 \
  --log-dir "$AOI_HANDOFF_ROOT/logs" \
  --report "$AOI_HANDOFF_ROOT/matrix-validation.json" --execute

python scripts/matrix_runner.py \
  --profile paper_faithful --device cuda:0 \
  --output-root "$AOI_RUN_ROOT" --checkpoint-every 5 \
  --recover-empty-run --stage audit \
  --eval-purpose validation --eval-episodes 100 \
  --eval-seeds 201,202,203,204,205,206 \
  --log-dir "$AOI_HANDOFF_ROOT/logs" \
  --report "$AOI_HANDOFF_ROOT/matrix-audit.json" --execute
```

Rerunning the same command safely skips completed, hash-bound cells; resumes a
matching incomplete checkpoint; and, only with the explicit flag, reinitializes
a strictly verified pre-checkpoint run. Any conflicting artifact is a hard
error requiring inspection.

## 5. Analysis status

Build a relocatable study manifest outside the checkout:

```bash
python analysis/study_manifest.py "$AOI_RUN_ROOT" \
  --output "$AOI_HANDOFF_ROOT/study-manifest-validation.json"

python analysis/build_paper_figures.py \
  "$AOI_HANDOFF_ROOT/study-manifest-validation.json" \
  --figure 3 --output-dir "$AOI_HANDOFF_ROOT/figures"

python analysis/build_paper_figures.py \
  "$AOI_HANDOFF_ROOT/study-manifest-validation.json" \
  --figure 5 --fig5-x gap_m --eval-purpose validation \
  --output-dir "$AOI_HANDOFF_ROOT/figures"
```

Validation Fig.5 is always labelled `PARTIAL`. Its sidecar distinguishes
`current_missing_cells` from the complete four-algorithm
`paper_missing_cells`. Fig.4 is complete only with all four algorithms for the
selected scenario and all training seeds; incomplete output must remain
explicitly labelled `_PARTIAL`.

Do not start or inspect final-test seeds 101..106 until the validation review is
closed. A final Fig.5 additionally requires all four algorithms across every
controlled scenario and seed 2..7.
