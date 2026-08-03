# Modified MADDPG with TDec reproduction

This directory is the reproducible implementation workspace for Algorithm 1.
The original source is preserved under `legacy_reference/` and its provenance is
recorded in `SOURCE_MANIFEST.json`. The source tree under `src/` is never modified
by this project.

The implementation is split into `legacy_release` and `paper_faithful` profiles.
Smoke runs are written below `scratch/`; formal runs are written below
`experiments/runs/` and are never overwritten.

```text
python Main.py --profile paper_faithful --scenario p05_n04_g25 --dry-run
python Main.py --profile paper_faithful --dry-run --matrix
```

Formal 500-episode training and the 48-run matrix are intentionally not launched
by this code-completion stage.

