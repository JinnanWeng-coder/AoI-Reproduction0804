"""Atomic complete checkpoints and RNG capture/restore."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


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


def build_payload(config, agents, learner, replay, environment, metrics, episode: int, completed: bool = False) -> Dict[str, Any]:
    return {
        "checkpoint_version": 2,
        "semantic_version": config.semantic_version,
        "config_hash": config.canonical_hash(),
        "config": config.to_dict(),
        "episode": int(episode),
        "completed": bool(completed),
        "agents": [agent.state_dict_full() for agent in agents],
        "learner": learner.state_dict_full(),
        "replay": replay.state_dict(),
        "environment": environment.state_dict() if hasattr(environment, "state_dict") else None,
        "metrics": metrics.state_dict(),
        "rng": capture_rng_state(),
    }
