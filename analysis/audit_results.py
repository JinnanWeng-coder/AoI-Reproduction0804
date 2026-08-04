"""Strict, read-only audits for training, evaluation, and study artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_SEMANTICS = {
    "paper_faithful": "paper_faithful_v3",
    "legacy_release": "legacy_release_v1",
}

EXPECTED_EVAL_SEEDS = {
    "validation": [201, 202, 203, 204, 205, 206],
    "final_test": [101, 102, 103, 104, 105, 106],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, errors, label: str) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("expected JSON object")
        return value
    except Exception as exc:
        errors.append(f"{label}_parse:{exc}")
        return None


def _load_resolved_config(data: Dict[str, Any], errors):
    try:
        from config import config_from_dict

        return config_from_dict(data)
    except Exception as exc:
        errors.append(f"config_resolve:{exc}")
        return None


def _audit_checkpoint(path: Path, expected_hash: str, expected_semantic: str, errors, require_completed: bool = False):
    if not path.is_file():
        errors.append(f"missing:{path.name}")
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        errors.append(f"checkpoint_read:{path.name}:{exc}")
        return None
    if payload.get("semantic_version") != expected_semantic:
        errors.append(f"checkpoint_semantic:{path.name}:{payload.get('semantic_version')!r}")
    if payload.get("config_hash") != expected_hash:
        errors.append(f"checkpoint_config_hash:{path.name}")
    embedded = payload.get("config")
    if not isinstance(embedded, dict) or embedded.get("semantic_version") != expected_semantic:
        errors.append(f"checkpoint_config_semantic:{path.name}")
    if require_completed and payload.get("completed") is not True:
        errors.append(f"checkpoint_not_completed:{path.name}")
    return payload


def _finite_arrays(npz_path: Path, errors):
    arrays = {}
    if not npz_path.is_file():
        errors.append(f"missing:{npz_path.name}")
        return arrays
    try:
        with np.load(npz_path, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
    except Exception as exc:
        errors.append(f"metrics_read:{exc}")
        return arrays
    for key, value in arrays.items():
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            errors.append(f"nonfinite:{key}")
    return arrays


def audit_run(run_dir: Path, require_complete: bool = True, require_eval: bool = False) -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    errors = []
    required = [
        run_dir / "config.resolved.json",
        run_dir / "provenance.json",
        run_dir / "train_metrics.npz",
        run_dir / "checkpoints" / "latest.pt",
        run_dir / "checkpoints" / "best.pt",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"missing:{path.name}")
    if require_complete and not (run_dir / "COMPLETE.json").is_file():
        errors.append("missing:COMPLETE.json")

    config = _read_json(run_dir / "config.resolved.json", errors, "config") if (run_dir / "config.resolved.json").is_file() else None
    provenance = _read_json(run_dir / "provenance.json", errors, "provenance") if (run_dir / "provenance.json").is_file() else None
    complete = _read_json(run_dir / "COMPLETE.json", errors, "complete") if (run_dir / "COMPLETE.json").is_file() else None
    resolved = _load_resolved_config(config, errors) if config is not None else None

    expected_hash = resolved.canonical_hash() if resolved is not None else None
    profile = None if config is None else config.get("profile")
    expected_semantic = EXPECTED_SEMANTICS.get(str(profile))
    if expected_semantic is None:
        errors.append(f"profile:{profile!r}")
    elif config.get("semantic_version") != expected_semantic:
        errors.append("config_semantic_version")
    if config is not None and bool(config.get("is_formal_result", True)) and bool(config.get("smoke", False)):
        errors.append("formal_result_marked_smoke")
    if provenance is not None and expected_semantic is not None:
        if provenance.get("semantic_version") != expected_semantic:
            errors.append("provenance_semantic_version")
        manifest = run_dir.parents[1] / "SOURCE_MANIFEST.json"
        if manifest.is_file() and provenance.get("source_manifest_sha256") != _sha256(manifest):
            errors.append("source_manifest_hash")
        if config is not None:
            for key in (
                "global_reward_normalization", "mobility_model", "eval_protocol", "eval_warmup_episodes",
                "gap_definition", "vehicle_length_m", "statistics_schema_version",
            ):
                if key in config and provenance.get(key) != config.get(key):
                    errors.append(f"provenance_{key}")
            if bool(config.get("is_formal_result", True)) and provenance.get("reproduction_git_dirty"):
                errors.append("formal_git_dirty")
            if config.get("is_formal_result", True) and not provenance.get("reproduction_git_commit"):
                errors.append("formal_git_commit_missing")
    if complete is not None:
        if require_complete and complete.get("status") != "complete":
            errors.append("complete_status")
        if expected_semantic is not None and complete.get("semantic_version") != expected_semantic:
            errors.append("complete_semantic_version")
        if expected_hash is not None and complete.get("config_hash") != expected_hash:
            errors.append("complete_config_hash")
        if provenance is not None and complete.get("reproduction_git_commit") != provenance.get("reproduction_git_commit"):
            errors.append("complete_git_commit")

    arrays = _finite_arrays(run_dir / "train_metrics.npz", errors)
    if config is not None and arrays:
        expected_episodes = int(config.get("episodes", -1))
        expected_steps = int(config.get("steps_per_episode", -1))
        expected_agents = int(config.get("scenario", {}).get("number_platoons", -1))
        required_keys = (
            "task1_step", "task2_step", "global_step", "task1_episode_mean", "task2_episode_mean",
            "global_episode_sum", "global_episode_mean", "local_total_episode_mean", "immediate_reward_proxy",
        )
        for key in required_keys:
            if key not in arrays:
                errors.append(f"missing_metric_key:{key}")
        if arrays.get("task1_step", np.empty(0)).shape != (expected_episodes, expected_steps, expected_agents):
            errors.append(f"shape:task1_step:{arrays.get('task1_step', np.empty(0)).shape}")
        if arrays.get("task2_step", np.empty(0)).shape != (expected_episodes, expected_steps, expected_agents):
            errors.append(f"shape:task2_step:{arrays.get('task2_step', np.empty(0)).shape}")
        if arrays.get("global_step", np.empty(0)).shape != (expected_episodes, expected_steps):
            errors.append(f"shape:global_step:{arrays.get('global_step', np.empty(0)).shape}")
        if "success" in arrays and arrays["success"].shape[:3] != (expected_episodes, expected_steps, expected_agents):
            errors.append(f"shape:success:{arrays['success'].shape}")

    checkpoint_payloads = {}
    if expected_hash is not None and expected_semantic is not None:
        for name in ("latest.pt", "best.pt"):
            checkpoint_payloads[name] = _audit_checkpoint(
                run_dir / "checkpoints" / name,
                expected_hash,
                expected_semantic,
                errors,
                require_completed=require_complete,
            )

    eval_reports = []
    eval_root = run_dir / "eval"
    if eval_root.is_dir():
        for child in sorted(eval_root.iterdir()):
            if child.is_dir():
                eval_reports.append(audit_eval(child))
                if not eval_reports[-1]["ok"]:
                    errors.extend(f"eval:{child.name}:{item}" for item in eval_reports[-1]["errors"])
    if require_eval and not eval_reports:
        errors.append("missing:eval")
    if require_eval and any(not report["ok"] for report in eval_reports):
        errors.append("invalid:eval")

    result = {
        "ok": not errors,
        "run_dir": str(run_dir),
        "errors": errors,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "formal_marker": None if config is None else bool(config.get("is_formal_result", True)),
        "semantic_version": None if config is None else config.get("semantic_version"),
        "config_hash": expected_hash,
        "checkpoint_status": {key: None if value is None else bool(value.get("completed", False)) for key, value in checkpoint_payloads.items()},
        "eval_reports": eval_reports,
    }
    return result


def audit_eval(eval_dir: Path) -> Dict[str, Any]:
    eval_dir = Path(eval_dir).resolve()
    errors = []
    metrics_path = eval_dir / "metrics.npz"
    summary_path = eval_dir / "summary.json"
    complete_path = eval_dir / "EVAL_COMPLETE.json"
    provenance_path = eval_dir / "provenance.json"
    for path in (metrics_path, summary_path, complete_path, provenance_path):
        if not path.is_file():
            errors.append(f"missing:{path.name}")
    summary = _read_json(summary_path, errors, "summary") if summary_path.is_file() else None
    complete = _read_json(complete_path, errors, "eval_complete") if complete_path.is_file() else None
    provenance = _read_json(provenance_path, errors, "eval_provenance") if provenance_path.is_file() else None
    arrays = _finite_arrays(metrics_path, errors)
    if complete is not None and complete.get("status") != "complete":
        errors.append("eval_complete_status")
    if summary is not None:
        seeds = [int(seed) for seed in summary.get("eval_seeds", [])]
        episodes = int(summary.get("eval_episodes", -1))
        if not seeds or len(set(seeds)) != len(seeds):
            errors.append("heldout_seed_uniqueness")
        if summary.get("eval_protocol") != "sequential_warm":
            errors.append("eval_protocol")
        purpose = summary.get("eval_purpose")
        if purpose not in EXPECTED_EVAL_SEEDS:
            errors.append("eval_purpose")
        if summary.get("statistics_schema_version") != "eval_seed_cluster_v1":
            errors.append("statistics_schema_version")
        if int(summary.get("eval_warmup_episodes", -1)) < 0:
            errors.append("eval_warmup_episodes")
        if summary.get("is_formal_result", False):
            if seeds != EXPECTED_EVAL_SEEDS.get(purpose):
                errors.append("formal_heldout_seeds")
            if int(summary.get("eval_warmup_episodes", -1)) != 5:
                errors.append("formal_eval_warmup")
            if int(summary.get("eval_episodes", -1)) != 100:
                errors.append("formal_eval_episodes")
        for key in ("mean_AoI_ms_per_seed", "sd_AoI_ms_per_seed", "endpoint_success_probability_per_seed"):
            if key not in summary or len(summary[key]) != len(seeds):
                errors.append(f"summary_length:{key}")
        if "ci95_AoI_ms_per_seed" in summary and len(summary["ci95_AoI_ms_per_seed"]) != len(seeds):
            errors.append("summary_length:ci95_AoI_ms_per_seed")
        if "ci95_CAM_success_probability_per_seed" in summary and len(summary["ci95_CAM_success_probability_per_seed"]) != len(seeds):
            errors.append("summary_length:ci95_CAM_success_probability_per_seed")
        checkpoint = Path(str(summary.get("checkpoint", ""))).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (eval_dir.parent.parent / checkpoint).resolve()
        if not checkpoint.is_file():
            errors.append("checkpoint_missing")
        else:
            actual_hash = _sha256(checkpoint)
            if summary.get("checkpoint_sha256") != actual_hash:
                errors.append("checkpoint_hash")
        if summary.get("semantic_version") not in EXPECTED_SEMANTICS.values():
            errors.append("eval_semantic_version")
        if summary.get("is_frozen_eval") is not True:
            errors.append("not_frozen_eval")
        for key in ("mean_AoI_ms_per_seed", "sd_AoI_ms_per_seed", "endpoint_success_probability_per_seed"):
            try:
                if not np.all(np.isfinite(np.asarray(summary[key], dtype=np.float64))):
                    errors.append(f"summary_nonfinite:{key}")
            except Exception:
                errors.append(f"summary_numeric:{key}")
        for key in ("ci95_AoI_ms_per_seed", "ci95_CAM_success_probability_per_seed"):
            if key in summary:
                values = summary[key]
                if any(value is not None and not np.isfinite(float(value)) for value in values):
                    errors.append(f"summary_nonfinite:{key}")
        if arrays:
            for key in ("aoi_ms", "success", "remaining_demand", "v2i_rate", "v2v_rate", "interference_db", "power_dbm", "rb", "mode"):
                if key not in arrays:
                    errors.append(f"missing_eval_metric_key:{key}")
            if "aoi_ms" in arrays and arrays["aoi_ms"].shape[:2] != (len(seeds), episodes):
                errors.append(f"shape:aoi_ms:{arrays['aoi_ms'].shape}")
            if "success" in arrays and arrays["success"].shape[:2] != (len(seeds), episodes):
                errors.append(f"shape:eval_success:{arrays['success'].shape}")
            if "success" in arrays and arrays["success"].ndim >= 4 and "endpoint_success_probability_per_seed" in summary:
                endpoint = arrays["success"][:, :, -1, :].mean(axis=2).mean(axis=1)
                if not np.allclose(endpoint, np.asarray(summary["endpoint_success_probability_per_seed"], dtype=np.float64), rtol=0.0, atol=1e-6):
                    errors.append("endpoint_success_mismatch")
            if "remaining_demand" in arrays and "success" in arrays:
                demand_endpoint = arrays["remaining_demand"][:, :, -1, :] <= 0.0
                success_endpoint = arrays["success"][:, :, -1, :] > 0.5
                if not np.array_equal(demand_endpoint, success_endpoint):
                    errors.append("endpoint_success_demand_mismatch")
        if summary.get("checkpoint_path_is_relative_to_run") is not True:
            errors.append("checkpoint_path_not_relative_to_run")
        run_dir = eval_dir.parent.parent
        config_path = run_dir / "config.resolved.json"
        run_config = _read_json(config_path, errors, "run_config") if config_path.is_file() else None
        if run_config is not None:
            try:
                from config import config_from_dict

                resolved_run_config = config_from_dict(run_config)
                if summary.get("config_hash") != resolved_run_config.canonical_hash():
                    errors.append("eval_config_hash")
                if summary.get("training_seed") != int(resolved_run_config.seed):
                    errors.append("eval_training_seed")
                if summary.get("scenario") != resolved_run_config.scenario.id:
                    errors.append("eval_scenario")
                if summary.get("gap_definition") != resolved_run_config.gap_definition:
                    errors.append("eval_gap_definition")
            except Exception as exc:
                errors.append(f"eval_config_resolve:{exc}")
        if provenance is not None:
            for key in ("semantic_version", "config_hash", "eval_purpose", "statistics_schema_version", "checkpoint_sha256"):
                if summary.get(key) != provenance.get(key):
                    errors.append(f"eval_provenance_{key}")

    return {
        "ok": not errors,
        "eval_dir": str(eval_dir),
        "errors": errors,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "semantic_version": None if summary is None else summary.get("semantic_version"),
    }


def audit_study_manifest(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    errors = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entries = manifest.get("entries", [])
    except Exception as exc:
        return {"ok": False, "manifest": str(path), "errors": [f"manifest_read:{exc}"]}
    required = {"algorithm", "semantic_version", "scenario", "training_seed", "run_path", "checkpoint_sha256", "eval_id", "status"}
    schema_version = int(manifest.get("schema_version", 1))
    manifest_base = path.parent
    if schema_version >= 2 and manifest.get("path_base") != "manifest_parent":
        errors.append("path_base")
    identities = set()
    for index, entry in enumerate(entries):
        missing = sorted(required.difference(entry))
        if missing:
            errors.append(f"entry{index}:missing:{','.join(missing)}")
            continue
        identity = (entry.get("scenario"), entry.get("training_seed"), entry.get("eval_id"))
        if identity in identities:
            errors.append(f"entry{index}:duplicate:{identity}")
        identities.add(identity)
        if entry.get("semantic_version") not in EXPECTED_SEMANTICS.values():
            errors.append(f"entry{index}:semantic_version")
        run_ref = Path(str(entry.get("run_path")))
        if schema_version >= 2 and run_ref.is_absolute():
            errors.append(f"entry{index}:absolute_run_path")
        run_path = (manifest_base / run_ref).resolve() if not run_ref.is_absolute() else run_ref
        if not run_path.is_dir():
            errors.append(f"entry{index}:run_missing")
        checkpoint_ref = Path(str(entry.get("checkpoint_path", "")))
        checkpoint_path = (manifest_base / checkpoint_ref).resolve() if not checkpoint_ref.is_absolute() else checkpoint_ref
        if entry.get("checkpoint_sha256") is not None:
            if not checkpoint_path.is_file():
                errors.append(f"entry{index}:checkpoint_missing")
            elif _sha256(checkpoint_path) != entry.get("checkpoint_sha256"):
                errors.append(f"entry{index}:checkpoint_hash")
        if entry.get("status") == "complete" and entry.get("eval_path"):
            eval_ref = Path(str(entry["eval_path"]))
            if schema_version >= 2 and eval_ref.is_absolute():
                errors.append(f"entry{index}:absolute_eval_path")
            eval_path = (manifest_base / eval_ref).resolve() if not eval_ref.is_absolute() else eval_ref
            if not (eval_path / "EVAL_COMPLETE.json").is_file():
                errors.append(f"entry{index}:eval_missing")
            else:
                eval_summary = _read_json(eval_path / "summary.json", errors, f"entry{index}:eval_summary")
                if eval_summary is not None:
                    if entry.get("eval_id") != eval_summary.get("eval_id"):
                        errors.append(f"entry{index}:eval_id_mismatch")
                    if entry.get("eval_purpose") != eval_summary.get("eval_purpose"):
                        errors.append(f"entry{index}:eval_purpose_mismatch")
                    if entry.get("semantic_version") != eval_summary.get("semantic_version"):
                        errors.append(f"entry{index}:eval_semantic_mismatch")
    return {"ok": not errors, "manifest": str(path), "errors": errors, "entry_count": len(entries), "unique_count": len(identities)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="experiments/runs")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    path = Path(args.path)
    if path.is_file() and path.name.endswith("manifest.json"):
        result = audit_study_manifest(path)
    elif (path / "EVAL_COMPLETE.json").exists():
        result = audit_eval(path)
    elif (path / "config.resolved.json").exists():
        result = audit_run(path, require_complete=not args.allow_incomplete, require_eval=args.require_eval)
    else:
        runs = sorted(child for child in path.iterdir() if child.is_dir()) if path.exists() else []
        reports = [audit_run(run, require_complete=not args.allow_incomplete, require_eval=args.require_eval) for run in runs]
        result = {"ok": all(report["ok"] for report in reports), "runs": reports}
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite audit report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
