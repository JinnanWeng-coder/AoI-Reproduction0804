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


from config import (
    CHECKPOINT_SCHEMA_VERSION,
    LEGACY_SEMANTIC_VERSION,
    PAPER_MOBILITY_REVISION,
    PAPER_SEMANTIC_VERSION,
    formal_scientific_contract_errors,
)

EXPECTED_SEMANTICS = {
    "paper_faithful": PAPER_SEMANTIC_VERSION,
    "legacy_release": LEGACY_SEMANTIC_VERSION,
}
EXPECTED_MOBILITY_REVISIONS = {
    PAPER_SEMANTIC_VERSION: PAPER_MOBILITY_REVISION,
    LEGACY_SEMANTIC_VERSION: "legacy_source_v1",
}

EXPECTED_EVAL_SEEDS = {
    "validation": [201, 202, 203, 204, 205, 206],
    "final_test": [101, 102, 103, 104, 105, 106],
}
EXPECTED_FIGURE_BASELINES = ["Modified_MADDPG", "MADDPG_FDec", "DDPG"]
FORMAL_TRAINING_SEEDS = set(range(2, 8))
RUNTIME_PROVENANCE_KEYS = (
    "python", "numpy", "torch", "cuda_version", "cuda_device_count", "cuda_driver", "gpu_names",
    "reproduction_git_commit", "reproduction_git_branch", "reproduction_git_dirty",
    "reproduction_tracked_tree_sha256", "source_manifest_sha256",
)
FORMAL_SCENARIOS = {
    "p05_n04_g05", "p07_n04_g05", "p05_n04_g15", "p05_n04_g25",
    "p05_n04_g35", "p05_n06_g25", "p05_n08_g25", "p05_n10_g25",
}
EVAL_BINDING_KEYS = ("eval_id", "eval_purpose", "scope", "checkpoint_sha256")


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


def _audit_checkpoint(
    path: Path,
    expected_hash: str,
    expected_semantic: str,
    errors,
    require_completed: bool = False,
    expected_episode: Optional[int] = None,
):
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
    if expected_semantic == PAPER_SEMANTIC_VERSION and payload.get("checkpoint_version") != 4:
        errors.append(f"checkpoint_schema:{path.name}:{payload.get('checkpoint_version')!r}")
    if expected_semantic == LEGACY_SEMANTIC_VERSION and payload.get("checkpoint_version") not in {1, 2, 3, 4}:
        errors.append(f"checkpoint_schema:{path.name}:{payload.get('checkpoint_version')!r}")
    if int(payload.get("checkpoint_version", 0)) >= 4:
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            errors.append(f"checkpoint_schema_version:{path.name}")
        if payload.get("mobility_revision") != EXPECTED_MOBILITY_REVISIONS.get(expected_semantic):
            errors.append(f"checkpoint_mobility_revision:{path.name}")
    if payload.get("config_hash") != expected_hash:
        errors.append(f"checkpoint_config_hash:{path.name}")
    embedded = payload.get("config")
    if not isinstance(embedded, dict) or embedded.get("semantic_version") != expected_semantic:
        errors.append(f"checkpoint_config_semantic:{path.name}")
    else:
        embedded_config = _load_resolved_config(embedded, errors)
        if embedded_config is not None and embedded_config.canonical_hash() != expected_hash:
            errors.append(f"checkpoint_embedded_config_hash:{path.name}")
    if require_completed and payload.get("completed") is not True:
        errors.append(f"checkpoint_not_completed:{path.name}")
    if require_completed and expected_episode is not None and int(payload.get("episode", -1)) != int(expected_episode):
        errors.append(f"checkpoint_final_episode:{path.name}")
    return payload


def _source_manifest_path() -> Path:
    return ROOT / "SOURCE_MANIFEST.json"


def _check_eval_binding(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]], errors, label: str) -> None:
    if left is None or right is None:
        return
    for key in EVAL_BINDING_KEYS:
        if key not in left:
            errors.append(f"{label}_missing:{key}")
        elif key not in right or left.get(key) != right.get(key):
            errors.append(f"{label}_mismatch:{key}")


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


def _expected_train_shapes(config: Dict[str, Any]) -> Dict[str, tuple]:
    episodes = int(config.get("episodes", -1))
    steps = int(config.get("steps_per_episode", -1))
    agents = int(config.get("scenario", {}).get("number_platoons", -1))
    followers = int(config.get("scenario", {}).get("platoon_size", -1)) - 1
    n_rb = int(config.get("n_rb", 3))
    base = (episodes, steps, agents)
    shapes = {
        "task1_step": base,
        "task2_step": base,
        "global_step": (episodes, steps),
        "task1_episode_mean": (episodes, agents),
        "task2_episode_mean": (episodes, agents),
        "local_total_episode_mean": (episodes, agents),
        "immediate_reward_proxy": (episodes, agents),
        "global_episode_sum": (episodes,),
        "global_episode_mean": (episodes,),
        "aoi_ms": base,
        "remaining_demand": base,
        "success": base,
        "v2i_rate": base,
        "v2v_rate": base,
        "power_dbm": base,
        "rb": base,
        "mode": base,
    }
    if config.get("profile") == "paper_faithful":
        shapes.update({
            "remaining_time_ms": base,
            "interference_db": base + (n_rb,),
            "selected_interference_db": base,
            "interference_linear": base + (n_rb,),
            "v2i_interference_linear": base + (n_rb,),
            "I_mode_db": base + (n_rb,),
            "I_v2i_linear": base + (n_rb,),
            "v2v_interference_linear": base + (followers, n_rb),
            "I_v2v_linear": base + (followers, n_rb),
            "v2v_rate_all": base + (followers,),
        })
    else:
        shapes["interference_db"] = base
        shapes["success_rate"] = (episodes, steps)
    return shapes


def _expected_eval_shapes(config: Dict[str, Any], seeds: int, episodes: int) -> Dict[str, tuple]:
    steps = int(config.get("steps_per_episode", -1))
    agents = int(config.get("scenario", {}).get("number_platoons", -1))
    followers = int(config.get("scenario", {}).get("platoon_size", -1)) - 1
    n_rb = int(config.get("n_rb", 3))
    base = (seeds, episodes, steps, agents)
    shapes = {
        "aoi_ms": base,
        "success": base,
        "remaining_demand": base,
        "v2i_rate": base,
        "v2v_rate": base,
        "power_dbm": base,
        "rb": base,
        "mode": base,
    }
    if config.get("profile") == "paper_faithful":
        shapes.update({
            "interference_db": base + (n_rb,),
            "selected_interference_db": base,
            "interference_linear": base + (n_rb,),
            "v2i_interference_linear": base + (n_rb,),
            "v2v_interference_linear": base + (followers, n_rb),
            "I_v2i_linear": base + (n_rb,),
            "I_v2v_linear": base + (followers, n_rb),
            "I_mode_db": base + (n_rb,),
            "v2v_rate_all": base + (followers,),
        })
    else:
        shapes["interference_db"] = base
    return shapes


def _check_expected_shapes(arrays: Dict[str, np.ndarray], expected: Dict[str, tuple], errors, prefix: str) -> None:
    for key, shape in expected.items():
        if key not in arrays:
            errors.append(f"missing_metric_key:{prefix}{key}")
        elif tuple(arrays[key].shape) != tuple(shape):
            errors.append(f"shape:{prefix}{key}:{arrays[key].shape}:expected={shape}")


def _check_summary_value(summary: Dict[str, Any], key: str, expected: Any, errors) -> None:
    if key not in summary:
        errors.append(f"summary_metric_missing:{key}")
        return
    try:
        claimed = np.asarray(summary[key], dtype=np.float64)
        calculated = np.asarray(expected, dtype=np.float64)
    except Exception:
        errors.append(f"summary_metric_numeric:{key}")
        return
    if claimed.shape != calculated.shape:
        errors.append(f"summary_metric_shape:{key}:{claimed.shape}:expected={calculated.shape}")
    elif not np.allclose(claimed, calculated, rtol=1e-6, atol=1e-6, equal_nan=False):
        errors.append(f"summary_metric_mismatch:{key}")


def _check_eval_summary_statistics(arrays: Dict[str, np.ndarray], summary: Dict[str, Any], errors) -> None:
    """Recompute every released AoI/CAM aggregate from immutable raw arrays."""
    if "aoi_ms" not in arrays or "success" not in arrays:
        return
    aoi = np.asarray(arrays["aoi_ms"], dtype=np.float64)
    success = np.asarray(arrays["success"], dtype=np.float64)
    if aoi.ndim != 4 or success.ndim != 4 or aoi.shape != success.shape:
        errors.append("summary_metric_source_shape")
        return
    if aoi.shape[1] < 1 or aoi.shape[2] < 1 or aoi.shape[3] < 1:
        errors.append("summary_metric_source_empty")
        return

    aoi_episode_seed_agent = aoi.mean(axis=2)
    endpoint_episode_seed_agent = success[:, :, -1, :]
    per_seed_aoi_agent = aoi_episode_seed_agent.mean(axis=1)
    per_seed_success_agent = endpoint_episode_seed_agent.mean(axis=1)
    per_seed_aoi = per_seed_aoi_agent.mean(axis=1)
    per_seed_success = per_seed_success_agent.mean(axis=1)
    if aoi.shape[1] > 1:
        sd_aoi = aoi_episode_seed_agent.mean(axis=2).std(axis=1, ddof=1)
        sd_success = endpoint_episode_seed_agent.mean(axis=2).std(axis=1, ddof=1)
    else:
        sd_aoi = np.zeros(aoi.shape[0], dtype=np.float64)
        sd_success = np.zeros(aoi.shape[0], dtype=np.float64)

    expected = {
        "mean_AoI_ms_per_seed_agent": per_seed_aoi_agent,
        "mean_AoI_ms_per_seed": per_seed_aoi,
        "sd_AoI_ms_per_seed": sd_aoi,
        "CAM_success_probability_per_seed_agent": per_seed_success_agent,
        "CAM_success_probability_per_seed": per_seed_success,
        "endpoint_success_probability_per_seed": per_seed_success,
        "sd_CAM_success_probability_per_seed": sd_success,
        "mean_AoI_ms": float(per_seed_aoi.mean()),
        "CAM_success_probability": float(per_seed_success.mean()),
    }
    for key, calculated in expected.items():
        _check_summary_value(summary, key, calculated, errors)


def audit_run(run_dir: Path, require_complete: bool = True, require_eval: bool = False, scope: str = "train") -> Dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    errors = []
    if scope not in {"train", "validation", "final_release"}:
        return {"ok": False, "run_dir": str(run_dir), "errors": [f"scope:{scope}"], "formal_marker": None}
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
    formal = bool(config.get("is_formal_result", True)) if config is not None else False
    if formal and profile == "paper_faithful":
        if resolved is not None:
            for field_name in formal_scientific_contract_errors(resolved):
                errors.append(f"formal_contract:{field_name}")
        if config.get("semantic_version") != PAPER_SEMANTIC_VERSION:
            errors.append("formal_semantic_version")
        if bool(config.get("smoke", False)):
            errors.append("formal_smoke_override")
        if int(config.get("episodes", -1)) != 500:
            errors.append("formal_episodes")
        if int(config.get("steps_per_episode", -1)) != 100:
            errors.append("formal_steps_per_episode")
        if int(config.get("seed", -1)) not in FORMAL_TRAINING_SEEDS:
            errors.append("formal_training_seed")
        if config.get("scenario", {}).get("id") not in FORMAL_SCENARIOS:
            errors.append("formal_scenario")
        if int(config.get("batch_size", -1)) != 64 or int(config.get("replay_capacity", -1)) != 50000:
            errors.append("formal_replay_or_batch")
        if config.get("actor_hidden") != [1024, 512] or config.get("local_critic_hidden") != [512, 256] or config.get("global_critic_hidden") != [1024, 512, 256]:
            errors.append("formal_network_size")
        if config.get("gap_definition") != "bumper_to_bumper" or float(config.get("vehicle_length_m", -1.0)) != 4.0:
            errors.append("formal_gap_semantics")
        if config.get("mobility_revision") != PAPER_MOBILITY_REVISION:
            errors.append("formal_mobility_revision")
    if provenance is not None and expected_semantic is not None:
        for key in RUNTIME_PROVENANCE_KEYS:
            if key not in provenance:
                errors.append(f"provenance_missing:{key}")
        if formal:
            for key in ("reproduction_git_commit", "reproduction_git_branch", "reproduction_tracked_tree_sha256", "source_manifest_sha256"):
                if not provenance.get(key):
                    errors.append(f"formal_provenance_missing:{key}")
            if not isinstance(provenance.get("reproduction_git_dirty"), bool):
                errors.append("formal_provenance_dirty_flag")
        if provenance.get("semantic_version") != expected_semantic:
            errors.append("provenance_semantic_version")
        manifest = _source_manifest_path()
        if formal and not manifest.is_file():
            errors.append("formal_source_manifest_missing")
        elif manifest.is_file() and provenance.get("source_manifest_sha256") != _sha256(manifest):
            errors.append("source_manifest_hash")
        if config is not None:
            for key in (
                "global_reward_normalization", "mobility_model", "eval_protocol", "eval_warmup_episodes",
                "gap_definition", "vehicle_length_m", "effective_center_spacing_m", "mobility_revision",
                "checkpoint_schema_version", "statistics_schema_version",
            ):
                if key in config and provenance.get(key) != config.get(key):
                    errors.append(f"provenance_{key}")
            if bool(config.get("is_formal_result", True)) and provenance.get("reproduction_git_dirty"):
                errors.append("formal_git_dirty")
            if config.get("is_formal_result", True) and not provenance.get("reproduction_git_commit"):
                errors.append("formal_git_commit_missing")
            if formal and provenance.get("reproduction_git_dirty") is not False:
                errors.append("formal_git_dirty")
            if formal and not provenance.get("reproduction_tracked_tree_sha256"):
                errors.append("formal_tree_digest_missing")
    if complete is not None:
        if require_complete and complete.get("status") != "complete":
            errors.append("complete_status")
        if expected_semantic is not None and complete.get("semantic_version") != expected_semantic:
            errors.append("complete_semantic_version")
        if expected_semantic is not None and complete.get("mobility_revision") != EXPECTED_MOBILITY_REVISIONS.get(expected_semantic):
            errors.append("complete_mobility_revision")
        if expected_hash is not None and complete.get("config_hash") != expected_hash:
            errors.append("complete_config_hash")
        if provenance is not None and complete.get("reproduction_git_commit") != provenance.get("reproduction_git_commit"):
            errors.append("complete_git_commit")
        if provenance is not None and complete.get("source_manifest_sha256") != provenance.get("source_manifest_sha256"):
            errors.append("complete_source_manifest")
        if provenance is not None and complete.get("reproduction_tracked_tree_sha256") != provenance.get("reproduction_tracked_tree_sha256"):
            errors.append("complete_tree_digest")
        if formal and complete.get("reproduction_git_branch") != (None if provenance is None else provenance.get("reproduction_git_branch")):
            errors.append("complete_git_branch")
        if formal:
            for key in ("source_manifest_sha256", "checkpoint_schema_version", "effective_center_spacing_m", "gap_definition", "vehicle_length_m"):
                if key not in complete:
                    errors.append(f"complete_missing:{key}")
            if complete.get("reproduction_git_dirty") is not False:
                errors.append("formal_complete_git_dirty")
            if complete.get("checkpoint_completed") is not True:
                errors.append("formal_complete_checkpoint_completed")
            if config is not None and int(complete.get("final_episode", -1)) != int(config.get("episodes", -1)):
                errors.append("formal_complete_final_episode")
            checkpoint_hashes = complete.get("checkpoint_sha256")
            if not isinstance(checkpoint_hashes, dict):
                errors.append("formal_complete_checkpoint_hashes")
            else:
                for name in ("latest.pt", "best.pt"):
                    path = run_dir / "checkpoints" / name
                    if path.is_file() and checkpoint_hashes.get(name) != _sha256(path):
                        errors.append(f"formal_complete_checkpoint_hash:{name}")

    arrays = _finite_arrays(run_dir / "train_metrics.npz", errors)
    if config is not None and arrays:
        _check_expected_shapes(arrays, _expected_train_shapes(config), errors, "train:")

    checkpoint_payloads = {}
    if expected_hash is not None and expected_semantic is not None:
        for name in ("latest.pt", "best.pt"):
            checkpoint_payloads[name] = _audit_checkpoint(
                run_dir / "checkpoints" / name,
                expected_hash,
                expected_semantic,
                errors,
                require_completed=require_complete,
                expected_episode=None if config is None else int(config.get("episodes", -1)),
            )
    if provenance is not None:
        for name, payload in checkpoint_payloads.items():
            if payload is None:
                continue
            if payload.get("reproduction_git_commit") is not None and payload.get("reproduction_git_commit") != provenance.get("reproduction_git_commit"):
                errors.append(f"checkpoint_git_commit:{name}")
            if payload.get("reproduction_git_branch") is not None and payload.get("reproduction_git_branch") != provenance.get("reproduction_git_branch"):
                errors.append(f"checkpoint_git_branch:{name}")
            if payload.get("reproduction_tracked_tree_sha256") is not None and payload.get("reproduction_tracked_tree_sha256") != provenance.get("reproduction_tracked_tree_sha256"):
                errors.append(f"checkpoint_tree_digest:{name}")
            if payload.get("source_manifest_sha256") is not None and payload.get("source_manifest_sha256") != provenance.get("source_manifest_sha256"):
                errors.append(f"checkpoint_source_manifest:{name}")
            if formal:
                for key in ("reproduction_git_commit", "reproduction_git_branch", "reproduction_git_dirty", "reproduction_tracked_tree_sha256", "source_manifest_sha256"):
                    if key not in payload:
                        errors.append(f"formal_checkpoint_missing:{name}:{key}")
            if payload.get("mobility_revision") is not None and payload.get("mobility_revision") != provenance.get("mobility_revision"):
                errors.append(f"checkpoint_mobility_revision:{name}")
            if payload.get("checkpoint_schema_version") is not None and payload.get("checkpoint_schema_version") != provenance.get("checkpoint_schema_version"):
                errors.append(f"checkpoint_schema_version:{name}")
            if formal and payload.get("reproduction_git_dirty") is not False:
                errors.append(f"formal_checkpoint_git_dirty:{name}")
            if formal and payload.get("episode") != (None if config is None else int(config.get("episodes", -1))) and require_complete:
                errors.append(f"formal_checkpoint_final_episode:{name}")

    eval_reports = []
    eval_root = run_dir / "eval"
    if eval_root.is_dir():
        for child in sorted(eval_root.iterdir()):
            if child.is_dir():
                eval_reports.append(audit_eval(child))
                if not eval_reports[-1]["ok"]:
                    errors.extend(f"eval:{child.name}:{item}" for item in eval_reports[-1]["errors"])
    if (require_eval or scope in {"validation", "final_release"}) and not eval_reports:
        errors.append("missing:eval")
    if (require_eval or scope in {"validation", "final_release"}) and any(not report["ok"] for report in eval_reports):
        errors.append("invalid:eval")
    if scope == "validation":
        validation_reports = [report for report in eval_reports if report.get("eval_purpose") == "validation"]
        if len(validation_reports) != 1:
            errors.append("validation_artifact_not_unique")
        for report in validation_reports:
            if report.get("summary_is_formal_result") is not False:
                errors.append("validation_formal_marker")
            if formal:
                if report.get("eval_seeds") != EXPECTED_EVAL_SEEDS["validation"]:
                    errors.append("formal_validation_seeds")
                if report.get("eval_episodes") != 100:
                    errors.append("formal_validation_episodes")
                if report.get("eval_warmup_episodes") != 5:
                    errors.append("formal_validation_warmup")
                if report.get("summary_is_formal_result") is not False:
                    errors.append("validation_formal_marker")
        marker = _read_json(run_dir / "VALIDATION_READY.json", errors, "validation_ready")
        if marker is None:
            errors.append("missing:VALIDATION_READY.json")
        elif marker.get("status") != "validation_ready" or marker.get("eval_purpose") != "validation":
            errors.append("validation_ready_marker")
        elif len(validation_reports) == 1:
            for key in EVAL_BINDING_KEYS:
                if key not in marker or marker.get(key) != validation_reports[0].get(key):
                    errors.append(f"validation_ready_binding:{key}")
    if scope == "final_release":
        if not (formal and profile == "paper_faithful"):
            errors.append("final_release_requires_formal_paper_training")
        final_reports = [report for report in eval_reports if report.get("eval_purpose") == "final_test"]
        if len(final_reports) != 1:
            errors.append("final_test_artifact_not_unique")
        for report in final_reports:
            if report.get("eval_seeds") != EXPECTED_EVAL_SEEDS["final_test"]:
                errors.append("final_test_seeds")
            if report.get("eval_episodes") != 100:
                errors.append("final_test_episodes")
            if report.get("eval_warmup_episodes") != 5:
                errors.append("final_test_warmup")
            if report.get("summary_is_formal_result") is not True:
                errors.append("final_test_formal_marker")
            if formal:
                if report.get("eval_seeds") != EXPECTED_EVAL_SEEDS["final_test"]:
                    errors.append("formal_final_test_seeds")
                if report.get("eval_episodes") != 100:
                    errors.append("formal_final_test_episodes")
                if report.get("eval_warmup_episodes") != 5:
                    errors.append("formal_final_test_warmup")
                if report.get("summary_is_formal_result") is not True:
                    errors.append("formal_final_test_marker")
        marker = _read_json(run_dir / "FINAL_RELEASE.json", errors, "final_release")
        if marker is None:
            errors.append("missing:FINAL_RELEASE.json")
        elif marker.get("status") != "final_release" or marker.get("eval_purpose") != "final_test":
            errors.append("final_release_marker")
        elif len(final_reports) == 1:
            for key in EVAL_BINDING_KEYS:
                if key not in marker or marker.get(key) != final_reports[0].get(key):
                    errors.append(f"final_release_binding:{key}")

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
        "scope": scope,
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
    run_dir = eval_dir.parent.parent
    config_path = run_dir / "config.resolved.json"
    if not config_path.is_file():
        errors.append("missing:run_config.resolved.json")
    run_config = _read_json(config_path, errors, "run_config") if config_path.is_file() else None
    resolved_run_config = _load_resolved_config(run_config, errors) if run_config is not None else None
    formal_training = bool(resolved_run_config is not None and resolved_run_config.is_formal_result)
    if formal_training and not _source_manifest_path().is_file():
        errors.append("formal_eval_source_manifest_missing")
    if complete is not None and complete.get("status") != "complete":
        errors.append("eval_complete_status")
    if summary is not None and complete is not None and summary != complete:
        errors.append("eval_complete_not_identical_to_summary")
    _check_eval_binding(summary, complete, errors, "eval_complete_binding")
    _check_eval_binding(summary, provenance, errors, "eval_provenance_binding")
    if summary is not None:
        if summary.get("eval_id") != eval_dir.name:
            errors.append("eval_directory_id")
        seeds = [int(seed) for seed in summary.get("eval_seeds", [])]
        episodes = int(summary.get("eval_episodes", -1))
        if not seeds or len(set(seeds)) != len(seeds):
            errors.append("heldout_seed_uniqueness")
        if summary.get("eval_protocol") != "sequential_warm":
            errors.append("eval_protocol")
        purpose = summary.get("eval_purpose")
        if purpose not in EXPECTED_EVAL_SEEDS:
            errors.append("eval_purpose")
        if purpose == "final_test" and not formal_training:
            errors.append("final_test_requires_formal_training")
        expected_scope = "validation" if purpose == "validation" else "final_release"
        if summary.get("scope") != expected_scope:
            errors.append("eval_scope")
        expected_release_status = "validation_ready" if purpose == "validation" else ("final_release" if summary.get("is_formal_result") else "evaluation_complete")
        if summary.get("release_status") != expected_release_status:
            errors.append("eval_release_status")
        if summary.get("statistics_schema_version") != "eval_seed_cluster_v1":
            errors.append("statistics_schema_version")
        if int(summary.get("eval_warmup_episodes", -1)) < 0:
            errors.append("eval_warmup_episodes")
        expected_formal_marker = bool(formal_training and purpose == "final_test")
        if summary.get("is_formal_result") is not expected_formal_marker:
            errors.append("eval_formal_marker_mismatch")
        if formal_training:
            if seeds != EXPECTED_EVAL_SEEDS.get(purpose):
                errors.append("formal_heldout_seeds")
            if int(summary.get("eval_warmup_episodes", -1)) != 5:
                errors.append("formal_eval_warmup")
            if int(summary.get("eval_episodes", -1)) != 100:
                errors.append("formal_eval_episodes")
        for key in ("mean_AoI_ms_per_seed", "sd_AoI_ms_per_seed", "endpoint_success_probability_per_seed"):
            if key not in summary or len(summary[key]) != len(seeds):
                errors.append(f"summary_length:{key}")
        for key in ("CAM_success_probability_per_seed", "sd_CAM_success_probability_per_seed"):
            if key in summary and len(summary[key]) != len(seeds):
                errors.append(f"summary_length:{key}")
        if "ci95_AoI_ms_per_seed" in summary and len(summary["ci95_AoI_ms_per_seed"]) != len(seeds):
            errors.append("summary_length:ci95_AoI_ms_per_seed")
        if "ci95_CAM_success_probability_per_seed" in summary and len(summary["ci95_CAM_success_probability_per_seed"]) != len(seeds):
            errors.append("summary_length:ci95_CAM_success_probability_per_seed")
        checkpoint_payload = None
        checkpoint_ref = Path(str(summary.get("checkpoint", ""))).expanduser()
        if checkpoint_ref.is_absolute():
            errors.append("checkpoint_path_absolute")
            checkpoint = checkpoint_ref.resolve()
        else:
            checkpoint = (run_dir / checkpoint_ref).resolve()
        allowed_checkpoints = {
            (run_dir / "checkpoints" / "latest.pt").resolve(),
            (run_dir / "checkpoints" / "best.pt").resolve(),
        }
        if checkpoint not in allowed_checkpoints or checkpoint.name not in {"latest.pt", "best.pt"}:
            errors.append("checkpoint_outside_run")
        if not checkpoint.is_file():
            errors.append("checkpoint_missing")
        else:
            actual_hash = _sha256(checkpoint)
            if summary.get("checkpoint_sha256") != actual_hash:
                errors.append("checkpoint_hash")
            try:
                checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                for key in ("semantic_version", "config_hash", "mobility_revision", "checkpoint_schema_version"):
                    if summary.get(key) != checkpoint_payload.get(key):
                        errors.append(f"checkpoint_{key}")
                for key in ("reproduction_git_commit", "reproduction_git_branch", "reproduction_tracked_tree_sha256", "source_manifest_sha256"):
                    if checkpoint_payload.get(key) is not None and summary.get(key) != checkpoint_payload.get(key):
                        errors.append(f"checkpoint_{key}")
                if checkpoint_payload.get("completed") is not True:
                    errors.append("checkpoint_not_completed")
                if resolved_run_config is not None and int(checkpoint_payload.get("episode", -1)) != int(resolved_run_config.episodes):
                    errors.append("checkpoint_final_episode")
                if summary.get("checkpoint_name") != checkpoint.name:
                    errors.append("checkpoint_name")
                if summary.get("checkpoint_completed") is not True:
                    errors.append("summary_checkpoint_completed")
                if resolved_run_config is not None and int(summary.get("checkpoint_episode", -1)) != int(resolved_run_config.episodes):
                    errors.append("summary_checkpoint_episode")
                if formal_training and checkpoint_payload.get("reproduction_git_dirty") is not False:
                    errors.append("formal_checkpoint_git_dirty")
                embedded = checkpoint_payload.get("config")
                embedded_config = _load_resolved_config(embedded, errors) if isinstance(embedded, dict) else None
                if embedded_config is None:
                    errors.append("checkpoint_embedded_config")
                elif resolved_run_config is not None and embedded_config.canonical_hash() != resolved_run_config.canonical_hash():
                    errors.append("checkpoint_embedded_config_hash")
            except Exception as exc:
                errors.append(f"checkpoint_read:{exc}")
        if summary.get("semantic_version") not in EXPECTED_SEMANTICS.values():
            errors.append("eval_semantic_version")
        if summary.get("mobility_revision") != EXPECTED_MOBILITY_REVISIONS.get(summary.get("semantic_version")):
            errors.append("eval_mobility_revision")
        if summary.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            errors.append("eval_checkpoint_schema_version")
        if summary.get("is_frozen_eval") is not True:
            errors.append("not_frozen_eval")
        if formal_training and summary.get("reproduction_git_dirty") is not False:
            errors.append("formal_eval_git_dirty")
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
            _check_eval_summary_statistics(arrays, summary, errors)
        if summary.get("checkpoint_path_is_relative_to_run") is not True:
            errors.append("checkpoint_path_not_relative_to_run")
        for key in RUNTIME_PROVENANCE_KEYS:
            if key not in summary:
                errors.append(f"summary_missing_runtime:{key}")
        if run_config is not None:
            try:
                if resolved_run_config is None:
                    raise ValueError("run config did not resolve")
                if summary.get("config_hash") != resolved_run_config.canonical_hash():
                    errors.append("eval_config_hash")
                if summary.get("training_seed") != int(resolved_run_config.seed):
                    errors.append("eval_training_seed")
                if summary.get("scenario") != resolved_run_config.scenario.id:
                    errors.append("eval_scenario")
                if summary.get("gap_definition") != resolved_run_config.gap_definition:
                    errors.append("eval_gap_definition")
                if summary.get("mobility_revision") != resolved_run_config.mobility_revision:
                    errors.append("eval_mobility_revision_config")
                if summary.get("effective_center_spacing_m") != resolved_run_config.effective_center_spacing_m:
                    errors.append("eval_center_spacing")
                if arrays:
                    _check_expected_shapes(
                        arrays,
                        _expected_eval_shapes(run_config, len(seeds), episodes),
                        errors,
                        "eval:",
                    )
            except Exception as exc:
                errors.append(f"eval_config_resolve:{exc}")
        if provenance is not None:
            for key in (
                "semantic_version", "config_hash", "eval_purpose", "scope", "release_status",
                "eval_id",
                "training_device_config", "evaluation_device_requested", "evaluation_device_resolved",
                "statistics_schema_version", "checkpoint_sha256", "checkpoint_schema_version",
                "checkpoint_name", "checkpoint_episode", "checkpoint_completed",
                "mobility_revision", "reproduction_git_commit", "reproduction_git_branch",
                "reproduction_tracked_tree_sha256", "source_manifest_sha256", "python", "numpy", "torch",
                "cuda_available", "cuda_version", "cuda_driver", "gpu_names",
                "cuda_device_count",
            ):
                if summary.get(key) != provenance.get(key):
                    errors.append(f"eval_provenance_{key}")
        run_provenance_path = eval_dir.parent.parent / "provenance.json"
        run_provenance = _read_json(run_provenance_path, errors, "run_provenance") if run_provenance_path.is_file() else None
        if run_provenance is not None:
            manifest_path = _source_manifest_path()
            if manifest_path.is_file() and summary.get("source_manifest_sha256") != _sha256(manifest_path):
                errors.append("eval_source_manifest_hash")
            for key in ("semantic_version", "config_hash", "mobility_revision", "reproduction_git_commit", "reproduction_git_branch", "reproduction_tracked_tree_sha256", "source_manifest_sha256"):
                if summary.get(key) != run_provenance.get(key):
                    errors.append(f"eval_run_provenance_{key}")
            if summary.get("reproduction_git_dirty") != run_provenance.get("reproduction_git_dirty"):
                errors.append("eval_run_provenance_dirty")
            if formal_training:
                for field_name in formal_scientific_contract_errors(resolved_run_config):
                    errors.append(f"formal_eval_contract:{field_name}")
                for key in ("reproduction_git_commit", "reproduction_git_branch", "reproduction_tracked_tree_sha256", "source_manifest_sha256"):
                    if not run_provenance.get(key):
                        errors.append(f"formal_eval_run_provenance_missing:{key}")
                if run_provenance.get("reproduction_git_dirty") is not False:
                    errors.append("formal_eval_run_git_dirty")
                if checkpoint_payload is not None:
                    for key in ("reproduction_git_commit", "reproduction_git_branch", "reproduction_tracked_tree_sha256", "source_manifest_sha256"):
                        if checkpoint_payload.get(key) != run_provenance.get(key):
                            errors.append(f"formal_eval_checkpoint_provenance:{key}")
                    if checkpoint_payload.get("reproduction_git_dirty") is not False:
                        errors.append("formal_eval_checkpoint_git_dirty")
        elif formal_training:
            errors.append("missing:run_provenance.json")

    return {
        "ok": not errors,
        "eval_dir": str(eval_dir),
        "errors": errors,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "semantic_version": None if summary is None else summary.get("semantic_version"),
        "eval_purpose": None if summary is None else summary.get("eval_purpose"),
        "eval_seeds": None if summary is None else summary.get("eval_seeds"),
        "eval_episodes": None if summary is None else summary.get("eval_episodes"),
        "eval_warmup_episodes": None if summary is None else summary.get("eval_warmup_episodes"),
        "summary_is_formal_result": None if summary is None else summary.get("is_formal_result"),
        "eval_id": None if summary is None else summary.get("eval_id"),
        "scope": None if summary is None else summary.get("scope"),
        "checkpoint_sha256": None if summary is None else summary.get("checkpoint_sha256"),
    }


def audit_study_manifest(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    errors = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entries = manifest.get("entries", [])
    except Exception as exc:
        return {"ok": False, "manifest": str(path), "errors": [f"manifest_read:{exc}"]}
    required = {"algorithm", "semantic_version", "mobility_revision", "scenario", "training_seed", "run_path", "checkpoint_sha256", "checkpoint_schema_version", "eval_id", "eval_purpose", "status"}
    schema_version = int(manifest.get("schema_version", 1))
    manifest_base = path.parent
    if schema_version >= 2 and manifest.get("path_base") != "manifest_parent":
        errors.append("path_base")
    if schema_version >= 2 and manifest.get("required_baselines") != EXPECTED_FIGURE_BASELINES:
        errors.append("required_baselines")
    identities = set()
    purpose_identities = set()
    for index, entry in enumerate(entries):
        missing = sorted(required.difference(entry))
        if missing:
            errors.append(f"entry{index}:missing:{','.join(missing)}")
            continue
        identity = (entry.get("algorithm"), entry.get("scenario"), entry.get("training_seed"), entry.get("eval_id"))
        if identity in identities:
            errors.append(f"entry{index}:duplicate:{identity}")
        identities.add(identity)
        purpose_identity = (entry.get("algorithm"), entry.get("scenario"), entry.get("training_seed"), entry.get("eval_purpose"))
        if purpose_identity in purpose_identities:
            errors.append(f"entry{index}:duplicate_training_seed_purpose:{purpose_identity}")
        purpose_identities.add(purpose_identity)
        if entry.get("is_formal_result") and int(entry.get("training_seed", -1)) not in FORMAL_TRAINING_SEEDS:
            errors.append(f"entry{index}:formal_training_seed")
        if entry.get("semantic_version") not in EXPECTED_SEMANTICS.values():
            errors.append(f"entry{index}:semantic_version")
        if entry.get("mobility_revision") != EXPECTED_MOBILITY_REVISIONS.get(entry.get("semantic_version")):
            errors.append(f"entry{index}:mobility_revision")
        if entry.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            errors.append(f"entry{index}:checkpoint_schema_version")
        if entry.get("is_formal_result"):
            for key in ("reproduction_git_commit", "reproduction_git_branch", "reproduction_git_dirty", "reproduction_tracked_tree_sha256", "source_manifest_sha256"):
                if not entry.get(key) and entry.get(key) is not False:
                    errors.append(f"entry{index}:provenance_missing:{key}")
        run_ref = Path(str(entry.get("run_path")))
        if schema_version >= 2 and run_ref.is_absolute():
            errors.append(f"entry{index}:absolute_run_path")
        run_path = (manifest_base / run_ref).resolve() if not run_ref.is_absolute() else run_ref
        if not run_path.is_dir():
            errors.append(f"entry{index}:run_missing")
        else:
            run_provenance = _read_json(run_path / "provenance.json", errors, f"entry{index}:run_provenance")
            if entry.get("config_hash") is not None and run_provenance is not None and entry.get("config_hash") != run_provenance.get("config_hash"):
                errors.append(f"entry{index}:config_hash_mismatch")
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
                    if entry.get("scope") != eval_summary.get("scope"):
                        errors.append(f"entry{index}:scope_mismatch")
                    if entry.get("release_status") != eval_summary.get("release_status"):
                        errors.append(f"entry{index}:release_status_mismatch")
                    if entry.get("semantic_version") != eval_summary.get("semantic_version"):
                        errors.append(f"entry{index}:eval_semantic_mismatch")
    expected_algorithms = [str(item) for item in manifest.get("expected_algorithms", [])]
    expected_scenarios = [str(item) for item in manifest.get("expected_scenarios", [])]
    expected_training_seeds = [int(item) for item in manifest.get("expected_training_seeds", [])]
    observed_cells = {
        (entry.get("algorithm"), entry.get("scenario"), int(entry.get("training_seed")), entry.get("eval_purpose"))
        for entry in entries
        if entry.get("eval_purpose") is not None and entry.get("training_seed") is not None
    }
    expected_purposes = ["validation", "final_test"] if expected_algorithms and expected_scenarios and expected_training_seeds else []
    missing_cells = [
        {"algorithm": algorithm, "scenario": scenario, "training_seed": seed, "eval_purpose": purpose}
        for algorithm in expected_algorithms
        for scenario in expected_scenarios
        for seed in expected_training_seeds
        for purpose in expected_purposes
        if (algorithm, scenario, seed, purpose) not in observed_cells
    ]
    return {
        "ok": not errors,
        "manifest": str(path),
        "errors": errors,
        "entry_count": len(entries),
        "unique_count": len(identities),
        "study_complete": not missing_cells if expected_purposes else None,
        "missing_cells": missing_cells,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="experiments/runs")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-eval", action="store_true")
    parser.add_argument("--scope", choices=("train", "validation", "final_release"), default="train")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    path = Path(args.path)
    if path.is_file() and path.name.endswith("manifest.json"):
        result = audit_study_manifest(path)
    elif (path / "EVAL_COMPLETE.json").exists():
        result = audit_eval(path)
    elif (path / "config.resolved.json").exists():
        result = audit_run(path, require_complete=not args.allow_incomplete, require_eval=args.require_eval, scope=args.scope)
    else:
        runs = sorted(child for child in path.iterdir() if child.is_dir()) if path.exists() else []
        reports = [audit_run(run, require_complete=not args.allow_incomplete, require_eval=args.require_eval, scope=args.scope) for run in runs]
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
