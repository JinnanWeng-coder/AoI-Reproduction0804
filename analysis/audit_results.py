"""Result-file audit; this module never runs training or changes metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def audit_run(run_dir: Path, require_complete: bool = True) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    errors = []
    required = [run_dir / "config.resolved.json", run_dir / "provenance.json", run_dir / "train_metrics.npz", run_dir / "checkpoints" / "latest.pt", run_dir / "checkpoints" / "best.pt"]
    for path in required:
        if not path.is_file():
            errors.append(f"missing:{path.name}")
    if require_complete and not (run_dir / "COMPLETE.json").is_file():
        errors.append("missing:COMPLETE.json")
    config = None
    if (run_dir / "config.resolved.json").is_file():
        try:
            config = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"config_parse:{exc}")
    arrays = {}
    metrics_path = run_dir / "train_metrics.npz"
    if metrics_path.is_file():
        try:
            with np.load(metrics_path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            for key, value in arrays.items():
                if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                    errors.append(f"nonfinite:{key}")
        except Exception as exc:
            errors.append(f"metrics_read:{exc}")
    if config is not None and arrays:
        expected_episodes = int(config["episodes"])
        expected_steps = int(config["steps_per_episode"])
        expected_agents = int(config["scenario"]["number_platoons"])
        for key in ("task1_step", "task2_step"):
            if key in arrays and arrays[key].shape != (expected_episodes, expected_steps, expected_agents):
                errors.append(f"shape:{key}:{arrays[key].shape}")
        if "global_step" in arrays and arrays["global_step"].shape != (expected_episodes, expected_steps):
            errors.append(f"shape:global_step:{arrays['global_step'].shape}")
        if "success" in arrays and arrays["success"].shape[:3] != (expected_episodes, expected_steps, expected_agents):
            errors.append(f"shape:success:{arrays['success'].shape}")
    result = {
        "ok": not errors,
        "run_dir": str(run_dir),
        "errors": errors,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "formal_marker": None if config is None else bool(config.get("is_formal_result", True)),
    }
    return result


def audit_eval(eval_dir: Path) -> Dict[str, Any]:
    eval_dir = Path(eval_dir).resolve()
    errors = []
    for name in ("metrics.npz", "summary.json", "EVAL_COMPLETE.json"):
        if not (eval_dir / name).is_file():
            errors.append(f"missing:{name}")
    if (eval_dir / "metrics.npz").is_file():
        with np.load(eval_dir / "metrics.npz", allow_pickle=False) as loaded:
            for key in loaded.files:
                value = loaded[key]
                if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                    errors.append(f"nonfinite:{key}")
    return {"ok": not errors, "eval_dir": str(eval_dir), "errors": errors}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="experiments/runs")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.path)
    if (path / "config.resolved.json").exists():
        result = audit_run(path, require_complete=not args.allow_incomplete)
    else:
        runs = sorted(child for child in path.iterdir() if child.is_dir()) if path.exists() else []
        reports = [audit_run(run, require_complete=not args.allow_incomplete) for run in runs]
        result = {"ok": all(report["ok"] for report in reports), "runs": reports}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

