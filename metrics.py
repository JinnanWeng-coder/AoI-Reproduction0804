"""Raw training/evaluation metrics and deterministic aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class MetricStore:
    def __init__(self, number_agents: int, steps_per_episode: int):
        self.number_agents = int(number_agents)
        self.steps_per_episode = int(steps_per_episode)
        self.episodes: List[Dict[str, Any]] = []
        self.learning: List[Dict[str, Any]] = []

    def append_episode(self, step_records: List[Dict[str, Any]], task1, task2, global_rewards):
        self.episodes.append({
            "task1_step": np.asarray(task1, dtype=np.float32),
            "task2_step": np.asarray(task2, dtype=np.float32),
            "global_step": np.asarray(global_rewards, dtype=np.float32),
            "info": {key: np.asarray([record[key] for record in step_records]) for key in step_records[0]},
        })

    def append_learning(self, diagnostics):
        if diagnostics:
            self.learning.append(diagnostics)

    def state_dict(self):
        return {"episodes": self.episodes, "learning": self.learning, "number_agents": self.number_agents, "steps_per_episode": self.steps_per_episode}

    def load_state_dict(self, state):
        self.episodes = list(state["episodes"])
        self.learning = list(state["learning"])

    def arrays(self) -> Dict[str, np.ndarray]:
        if not self.episodes:
            return {
                "task1_step": np.empty((0, self.steps_per_episode, self.number_agents), dtype=np.float32),
                "task2_step": np.empty((0, self.steps_per_episode, self.number_agents), dtype=np.float32),
                "global_step": np.empty((0, self.steps_per_episode), dtype=np.float32),
            }
        task1 = np.stack([item["task1_step"] for item in self.episodes]).astype(np.float32)
        task2 = np.stack([item["task2_step"] for item in self.episodes]).astype(np.float32)
        global_step = np.stack([item["global_step"] for item in self.episodes]).astype(np.float32)
        arrays = {
            "task1_step": task1,
            "task2_step": task2,
            "global_step": global_step,
            "task1_episode_mean": task1.mean(axis=1),
            "task2_episode_mean": task2.mean(axis=1),
            "global_episode_sum": global_step.sum(axis=1),
            "local_total_episode_mean": (task1 + task2).mean(axis=1),
        }
        info_keys = sorted(self.episodes[0]["info"])
        for key in info_keys:
            arrays[key] = np.stack([item["info"][key] for item in self.episodes]).astype(np.float32)
        return arrays

    def save(self, run_dir: Path, prefix: str = "train_metrics") -> Dict[str, Any]:
        arrays = self.arrays()
        np.savez_compressed(run_dir / f"{prefix}.npz", **arrays)
        try:
            import scipy.io

            scipy.io.savemat(run_dir / f"{prefix}.mat", arrays)
        except ImportError:
            pass
        learning_path = run_dir / "learning_diagnostics.json"
        learning_path.write_text(json.dumps(self.learning, indent=2, default=_json_default) + "\n", encoding="utf-8")
        return {key: list(value.shape) for key, value in arrays.items()}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)

