"""Build a study index that references existing run/eval artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_BASELINES = ["Modified_MADDPG", "MADDPG_FDec", "DDPG"]
EXPECTED_SCENARIOS = [
    "p05_n04_g05", "p07_n04_g05", "p05_n04_g15", "p05_n04_g25",
    "p05_n04_g35", "p05_n06_g25", "p05_n08_g25", "p05_n10_g25",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _relative_reference(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    return Path(os.path.relpath(Path(path).resolve(), base.resolve())).as_posix()


def build_study_manifest(run_root: Path, output: Path, algorithm: str = "Modified_MADDPG_with_TDec", run_paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    run_root = Path(run_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    manifest_base = output.parent.resolve()
    entries: List[Dict[str, Any]] = []
    if run_paths:
        candidate_runs = [Path(path).expanduser().resolve() for path in run_paths]
    elif (run_root / "config.resolved.json").is_file():
        candidate_runs = [run_root]
    else:
        candidate_runs = sorted(path for path in run_root.iterdir() if path.is_dir()) if run_root.is_dir() else []
    for run_dir in candidate_runs:
        config = _json(run_dir / "config.resolved.json")
        if config is None:
            continue
        provenance = _json(run_dir / "provenance.json") or {}
        complete = _json(run_dir / "COMPLETE.json") or {}
        eval_root = run_dir / "eval"
        eval_dirs = sorted(path for path in eval_root.iterdir() if path.is_dir()) if eval_root.is_dir() else []
        if not eval_dirs:
            eval_dirs = [None]
        checkpoint_hashes: Dict[str, Optional[str]] = {}
        for eval_dir in eval_dirs:
            summary = _json(eval_dir / "summary.json") if eval_dir is not None else None
            checkpoint_name = "latest.pt" if summary is None else str(summary.get("checkpoint_name", "latest.pt"))
            if checkpoint_name not in {"latest.pt", "best.pt"}:
                checkpoint_name = "latest.pt"
            checkpoint = run_dir / "checkpoints" / checkpoint_name
            if checkpoint_name not in checkpoint_hashes:
                checkpoint_hashes[checkpoint_name] = _sha256(checkpoint) if checkpoint.is_file() else None
            checkpoint_hash = checkpoint_hashes[checkpoint_name]
            entries.append({
                "algorithm": algorithm,
                "semantic_version": config.get("semantic_version"),
                "mobility_revision": config.get("mobility_revision"),
                "checkpoint_schema_version": "checkpoint_v4",
                "profile": config.get("profile"),
                "scenario": config.get("scenario", {}).get("id"),
                "training_seed": config.get("seed"),
                "run_path": _relative_reference(run_dir, manifest_base),
                "checkpoint_path": _relative_reference(checkpoint, manifest_base) if checkpoint.is_file() else None,
                "checkpoint_sha256": checkpoint_hash,
                "eval_id": None if summary is None else summary.get("eval_id"),
                "eval_path": _relative_reference(eval_dir, manifest_base) if eval_dir is not None else None,
                "status": "complete" if complete.get("status") == "complete" and (summary is None or summary.get("status") == "complete") else "incomplete",
                "is_formal_result": bool(False if summary is None else summary.get("is_formal_result", config.get("is_formal_result", True))),
                "eval_protocol": None if summary is None else summary.get("eval_protocol"),
                "eval_purpose": None if summary is None else summary.get("eval_purpose"),
                "eval_noise": None if summary is None else summary.get("eval_noise", 0.0),
                "scope": None if summary is None else summary.get("scope"),
                "release_status": None if summary is None else summary.get("release_status"),
                "statistics_schema_version": None if summary is None else summary.get("statistics_schema_version", config.get("statistics_schema_version")),
                "config_hash": provenance.get("config_hash"),
                "reproduction_git_commit": provenance.get("reproduction_git_commit"),
                "reproduction_git_branch": provenance.get("reproduction_git_branch"),
                "reproduction_git_dirty": provenance.get("reproduction_git_dirty"),
                "reproduction_tracked_tree_sha256": provenance.get("reproduction_tracked_tree_sha256"),
                "source_manifest_sha256": provenance.get("source_manifest_sha256"),
            })
    manifest = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "root": _relative_reference(run_root, manifest_base),
        "path_base": "manifest_parent",
        "required_baselines": REQUIRED_BASELINES,
        "expected_algorithms": [algorithm],
        "expected_scenarios": EXPECTED_SCENARIOS,
        "expected_training_seeds": list(range(2, 8)),
        "study_identity_fields": ["algorithm", "scenario", "training_seed", "eval_purpose", "eval_noise"],
        "statistics_schema_version": "eval_seed_cluster_v1",
        "mobility_revisions": sorted({entry.get("mobility_revision") for entry in entries if entry.get("mobility_revision")}),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite study manifest: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--output", default="study_manifest.json")
    parser.add_argument("--algorithm", default="Modified_MADDPG_with_TDec")
    parser.add_argument("--run-path", action="append", default=None, help="explicit run path; repeat to exclude unrelated historical artifacts")
    args = parser.parse_args(argv)
    manifest = build_study_manifest(Path(args.run_root), Path(args.output), args.algorithm, [Path(item) for item in args.run_path] if args.run_path else None)
    print(json.dumps({"output": str(Path(args.output).resolve()), "entries": len(manifest["entries"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
