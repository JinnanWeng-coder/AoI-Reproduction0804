"""Atomic complete checkpoints and RNG capture/restore."""

from __future__ import annotations

import os
import random
import subprocess
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _git_stamp() -> Dict[str, Any]:
    root = Path(__file__).resolve().parent

    def _git(*args: str):
        try:
            result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    tracked = _git("ls-files", "-z") or ""
    digest = hashlib.sha256()
    if tracked:
        for name in sorted(item for item in tracked.split("\x00") if item):
            path = root / name
            digest.update(name.replace(os.sep, "/").encode("utf-8"))
            digest.update(b"\x00")
            if path.is_file():
                digest.update(path.read_bytes())
    return {
        "reproduction_git_commit": _git("rev-parse", "HEAD"),
        "reproduction_git_branch": _git("branch", "--show-current"),
        "reproduction_git_dirty": bool(_git("status", "--porcelain", "--untracked-files=all")),
        "reproduction_tracked_tree_sha256": digest.hexdigest() if tracked else None,
    }


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _source_manifest_digest() -> Optional[str]:
    path = Path(__file__).resolve().parent / "SOURCE_MANIFEST.json"
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(config, agents, learner, replay, environment, metrics, episode: int, completed: bool = False) -> Dict[str, Any]:
    payload = {
        "checkpoint_version": 4,
        "checkpoint_schema_version": "checkpoint_v4",
        "semantic_version": config.semantic_version,
        "mobility_revision": config.mobility_revision,
        "config_hash": config.canonical_hash(),
        "source_manifest_sha256": _source_manifest_digest(),
        "config": config.to_dict(),
        "episode": int(episode),
        "completed": bool(completed),
        "agents": [agent.state_dict_full() for agent in agents],
        "learner": learner.state_dict_full(),
        "replay": replay.state_dict(),
        "environment": environment.state_dict() if hasattr(environment, "state_dict") else None,
        "metrics": metrics.state_dict(),
        "rng": capture_rng_state(),
        "gap_definition": config.gap_definition,
        "vehicle_length_m": float(config.vehicle_length_m),
        "statistics_schema_version": config.statistics_schema_version,
    }
    payload.update(_git_stamp())
    return payload
