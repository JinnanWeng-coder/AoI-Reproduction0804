"""Contract helpers for the E5 actor-sharing x value-structure experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aoi_v2x_reproduction.config import config_from_dict, validate_config


SEEDS = (8, 9, 10, 11, 12, 13)
SCENARIO = "p05_n04_g25"
EVAL_SEEDS = (213, 214, 215, 216, 217, 218)


def independent_run_name(variant: str, seed: int) -> str:
    if variant not in {"combined", "tdec"}:
        raise ValueError(f"unsupported MAPPO variant: {variant}")
    return f"mappo_tdec_ab_{variant}_{SCENARIO}_seed{int(seed):02d}"


def shared_run_name(variant: str, seed: int) -> str:
    if variant == "tdec":
        return f"mappo_shared_actor_tdec_{SCENARIO}_seed{int(seed):02d}"
    if variant == "combined":
        return f"mappo_e5_shared_combined_{SCENARIO}_seed{int(seed):02d}"
    raise ValueError(f"unsupported MAPPO variant: {variant}")


def eval_task_to_cell(task_id: int) -> tuple[str, int]:
    task_id = int(task_id)
    if task_id < 0 or task_id >= 12:
        raise ValueError("task_id must satisfy 0 <= task_id < 12")
    return ("independent", "shared")[task_id // 6], SEEDS[task_id % 6]


def _read_object(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_parent_combined_run(parent_run: Path, seed: int) -> tuple[dict, dict]:
    parent_run = Path(parent_run).resolve()
    expected_name = independent_run_name("combined", seed)
    if parent_run.name != expected_name:
        raise ValueError(f"parent run must be {expected_name}, got {parent_run.name}")
    config = _read_object(parent_run / "config.resolved.json")
    complete = _read_object(parent_run / "COMPLETE.json")
    checks = {
        "algorithm": (config.get("algorithm"), "mappo"),
        "profile": (config.get("profile"), "reproduction_baseline"),
        "scenario": (config.get("scenario", {}).get("id"), SCENARIO),
        "seed": (int(config.get("seed", -1)), int(seed)),
        "variant": (config.get("mappo_variant", "combined"), "combined"),
        "actor_sharing": (bool(config.get("mappo_actor_sharing", False)), False),
        "episodes": (int(config.get("episodes", -1)), 500),
        "steps": (int(config.get("steps_per_episode", -1)), 100),
        "rollout": (int(config.get("mappo_rollout_episodes", -1)), 5),
        "ppo_epochs": (int(config.get("mappo_ppo_epochs", -1)), 10),
        "value_clip": (config.get("mappo_value_clip_mode"), "normalized"),
        "checkpoint_mode": (config.get("checkpoint_mode"), "policy_only"),
        "diagnostics": (bool(config.get("diagnostics", False)), True),
        "slow_update": (int(config.get("slow_update_every_episodes", -1)), 1),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{expected_name}: parent {label}={actual!r}, expected {expected!r}")
    for key, expected in (
        ("tau", 0.005),
        ("mappo_actor_lr", 0.0005), ("mappo_critic_lr", 0.0005),
        ("mappo_entropy_coef_rb", 0.02), ("mappo_entropy_coef_mode", 0.02),
        ("mappo_entropy_coef_power", 0.002),
    ):
        if not np.isclose(float(config.get(key)), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"{expected_name}: parent {key} mismatch")
    if complete.get("status") != "complete" or complete.get("mappo_variant", "combined") != "combined":
        raise ValueError(f"{expected_name}: parent training is not complete combined MAPPO")
    if bool(complete.get("actor_sharing", False)):
        raise ValueError(f"{expected_name}: parent must use independent actors")
    if not (parent_run / "policy_final.pt").is_file() or not (parent_run / "train_metrics.npz").is_file():
        raise ValueError(f"{expected_name}: parent policy or training metrics are missing")
    return config, complete


def derive_shared_combined_config(parent_run: Path, result_run_root: Path, seed: int, device: str = "cuda:0"):
    """Change only sharing and run/output identity from the seed-matched parent."""
    parent_data, _complete = validate_parent_combined_run(parent_run, seed)
    parent = config_from_dict(parent_data)
    derived = copy.deepcopy(parent)
    derived.mappo_actor_sharing = True
    derived._omit_mappo_actor_sharing_from_serialization = False
    derived.run_name = shared_run_name("combined", seed)
    derived.output_root = str(Path(result_run_root).resolve())
    derived.device = str(device)
    derived.is_formal_result = False
    validate_config(derived)
    return derived


def config_differences(parent: Mapping[str, Any], derived: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return leaf differences; derived metadata is intentionally ignored."""
    differences: dict[str, dict[str, Any]] = {}

    def walk(prefix: str, left: Any, right: Any) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            for key in sorted(set(left) | set(right)):
                if key == "derived":
                    continue
                walk(f"{prefix}.{key}" if prefix else str(key), left.get(key), right.get(key))
        elif left != right:
            differences[prefix] = {"parent": left, "derived": right}

    walk("", parent, derived)
    return differences


ALLOWED_DERIVATION_DIFFERENCES = {
    "device", "mappo_actor_sharing", "output_root", "run_name", "is_formal_result",
}


def validate_derived_against_parent(parent: Mapping[str, Any], derived: Mapping[str, Any]) -> dict:
    differences = config_differences(parent, derived)
    unexpected = sorted(set(differences) - ALLOWED_DERIVATION_DIFFERENCES)
    if unexpected:
        raise ValueError("unexpected E5 parent-config drift: " + ", ".join(unexpected))
    required = {"mappo_actor_sharing", "output_root", "run_name"}
    missing = sorted(required - set(differences))
    if missing:
        raise ValueError("E5 derivation did not change required fields: " + ", ".join(missing))
    return {"status": "PASS", "allowed_fields": sorted(ALLOWED_DERIVATION_DIFFERENCES), "differences": differences}
