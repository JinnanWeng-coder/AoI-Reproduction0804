"""Raw training/evaluation metrics and deterministic aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class MetricStore:
    GRADIENT_FIELDS = (
        "task1_grad_l2",
        "task2_grad_l2",
        "local_sum_grad_l2",
        "global_grad_l2",
        "global_to_local_ratio",
        "global_to_task1_ratio",
        "global_to_task2_ratio",
        "task1_to_task2_ratio",
        "global_vs_local_cosine",
        "global_vs_task1_cosine",
        "global_vs_task2_cosine",
        "task1_vs_task2_cosine",
    )

    def __init__(
        self,
        number_agents: int,
        steps_per_episode: int,
        global_actor_weight: float = 1.0,
        n_rb: int = 3,
        n_modes: int = 2,
        power_min_dbm: float = 1.0,
        power_max_dbm: float = 30.0,
        diagnostics: bool = False,
    ):
        self.number_agents = int(number_agents)
        self.steps_per_episode = int(steps_per_episode)
        self.global_actor_weight = float(global_actor_weight)
        self.n_rb = int(n_rb)
        self.n_modes = int(n_modes)
        self.power_min_dbm = float(power_min_dbm)
        self.power_max_dbm = float(power_max_dbm)
        self.diagnostics = bool(diagnostics)
        self.episodes: List[Dict[str, Any]] = []
        self.learning: List[Dict[str, Any]] = []
        self.gradient_episodes: List[Dict[str, Any]] = []

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

    def append_learning_episode(self, records: List[Dict[str, Any]]) -> None:
        """Store compact episode summaries instead of every replay update."""
        episode_index = len(self.episodes) - 1
        actor_records = [record for record in records if record.get("actor_loss") is not None]
        summary: Dict[str, Any] = {
            "episode": int(episode_index + 1),
            "learner_update_count": int(len(records)),
            "actor_update_count": int(len(actor_records)),
        }
        for key in ("global_critic_loss", "actor_loss"):
            values = [float(record[key]) for record in records if record.get(key) is not None and np.isfinite(record[key])]
            summary[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
        for key in ("local_critic_loss", "global_actor_gradient_norms", "actor_parameter_deltas"):
            values = [np.asarray(record[key], dtype=np.float64) for record in records if len(record.get(key, [])) == self.number_agents]
            summary[f"{key}_mean"] = np.mean(np.stack(values), axis=0).tolist() if values else [0.0] * self.number_agents
        if records:
            summary["learn_step_first"] = int(records[0].get("learn_step", 0))
            summary["learn_step_last"] = int(records[-1].get("learn_step", 0))
            summary["global_target_update_fraction"] = float(np.mean([bool(record.get("global_target_update")) for record in records]))
            summary["local_target_update_fraction"] = float(np.mean([bool(record.get("local_target_update")) for record in records]))
        else:
            summary.update({
                "learn_step_first": 0,
                "learn_step_last": 0,
                "global_target_update_fraction": 0.0,
                "local_target_update_fraction": 0.0,
            })
        self.learning.append(summary)

        if not self.diagnostics:
            return
        gradient_records = [
            record["actor_gradient_diagnostics"]
            for record in actor_records
            if isinstance(record.get("actor_gradient_diagnostics"), dict)
        ]
        if len(gradient_records) != len(actor_records) or any(record.get("finite") is not True for record in gradient_records):
            raise FloatingPointError("non-finite or missing actor-gradient diagnostics")
        gradient_summary: Dict[str, Any] = {
            "episode": int(episode_index + 1),
            "actor_update_count": int(len(gradient_records)),
            "finite_fraction": 1.0,
            "all_finite": True,
        }
        for field in self.GRADIENT_FIELDS:
            values = [np.asarray(record[field], dtype=np.float64) for record in gradient_records]
            if values:
                stacked = np.stack(values)
                gradient_summary[f"{field}_mean"] = stacked.mean(axis=0)
                gradient_summary[f"{field}_p50"] = np.median(stacked, axis=0)
                gradient_summary[f"{field}_max"] = stacked.max(axis=0)
            else:
                zeros = np.zeros(self.number_agents, dtype=np.float64)
                gradient_summary[f"{field}_mean"] = zeros.copy()
                gradient_summary[f"{field}_p50"] = zeros.copy()
                gradient_summary[f"{field}_max"] = zeros.copy()
        self.gradient_episodes.append(gradient_summary)

    def state_dict(self):
        return {
            "episodes": self.episodes,
            "learning": self.learning,
            "number_agents": self.number_agents,
            "steps_per_episode": self.steps_per_episode,
            "global_actor_weight": self.global_actor_weight,
            "n_rb": self.n_rb,
            "n_modes": self.n_modes,
            "power_min_dbm": self.power_min_dbm,
            "power_max_dbm": self.power_max_dbm,
            "diagnostics": self.diagnostics,
            "gradient_episodes": self.gradient_episodes,
        }

    def load_state_dict(self, state):
        self.episodes = list(state["episodes"])
        self.learning = list(state["learning"])
        self.global_actor_weight = float(state.get("global_actor_weight", self.global_actor_weight))
        self.n_rb = int(state.get("n_rb", self.n_rb))
        self.n_modes = int(state.get("n_modes", self.n_modes))
        self.power_min_dbm = float(state.get("power_min_dbm", self.power_min_dbm))
        self.power_max_dbm = float(state.get("power_max_dbm", self.power_max_dbm))
        self.diagnostics = bool(state.get("diagnostics", self.diagnostics))
        self.gradient_episodes = list(state.get("gradient_episodes", []))

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
        local_total = (task1 + task2).mean(axis=1)
        global_episode_sum = global_step.sum(axis=1)
        global_episode_mean = global_step.mean(axis=1)
        arrays = {
            "task1_step": task1,
            "task2_step": task2,
            "global_step": global_step,
            "task1_episode_mean": task1.mean(axis=1),
            "task2_episode_mean": task2.mean(axis=1),
            "global_episode_sum": global_episode_sum,
            "global_episode_mean": global_episode_mean,
            "local_total_episode_mean": local_total,
            # This is an immediate reward aggregation for plotting/audit, not
            # the differentiable actor objective used by the learner.
            "immediate_reward_proxy": local_total + self.global_actor_weight * global_episode_mean[:, None],
        }
        info_keys = sorted(self.episodes[0]["info"])
        for key in info_keys:
            arrays[key] = np.stack([item["info"][key] for item in self.episodes]).astype(np.float32)
        if "aoi_ms" in arrays:
            arrays["mean_aoi_ms_episode_agent"] = arrays["aoi_ms"].mean(axis=1)
            arrays["worst_agent_mean_aoi_ms_episode"] = arrays["mean_aoi_ms_episode_agent"].max(axis=1)
        if "success" in arrays:
            arrays["endpoint_cam_episode_agent"] = arrays["success"][:, -1, :]
            arrays["worst_agent_endpoint_cam_episode"] = arrays["endpoint_cam_episode_agent"].min(axis=1)
        if "mode" in arrays:
            mode = arrays["mode"].astype(np.int64)
            fractions = np.stack([(mode == value).mean(axis=1) for value in range(self.n_modes)], axis=-1).astype(np.float32)
            arrays["mode_fraction_episode_agent"] = fractions
            arrays["mode_entropy_normalized_episode_agent"] = self._normalized_entropy(fractions)
            arrays["mode_switch_rate_episode_agent"] = (mode[:, 1:, :] != mode[:, :-1, :]).mean(axis=1).astype(np.float32) if self.steps_per_episode > 1 else np.zeros((mode.shape[0], self.number_agents), dtype=np.float32)
        if "rb" in arrays:
            rb = arrays["rb"].astype(np.int64)
            fractions = np.stack([(rb == value).mean(axis=1) for value in range(self.n_rb)], axis=-1).astype(np.float32)
            arrays["rb_fraction_episode_agent"] = fractions
            arrays["rb_entropy_normalized_episode_agent"] = self._normalized_entropy(fractions)
            arrays["rb_switch_rate_episode_agent"] = (rb[:, 1:, :] != rb[:, :-1, :]).mean(axis=1).astype(np.float32) if self.steps_per_episode > 1 else np.zeros((rb.shape[0], self.number_agents), dtype=np.float32)
        if "power_dbm" in arrays:
            power = arrays["power_dbm"]
            tolerance = max((self.power_max_dbm - self.power_min_dbm) * 0.01, 1e-5)
            arrays["power_post_map_near_min_fraction_episode_agent"] = (power <= self.power_min_dbm + tolerance).mean(axis=1).astype(np.float32)
            arrays["power_post_map_near_max_fraction_episode_agent"] = (power >= self.power_max_dbm - tolerance).mean(axis=1).astype(np.float32)
        if "action_post_clip_normalized" in arrays:
            action = arrays["action_post_clip_normalized"]
            arrays["action_post_clip_abs_ge_0p95_fraction_episode_agent_dim"] = (np.abs(action) >= 0.95).mean(axis=1).astype(np.float32)
            arrays["power_action_post_clip_near_min_fraction_episode_agent"] = (action[..., 2] <= -0.95).mean(axis=1).astype(np.float32)
            arrays["power_action_post_clip_near_max_fraction_episode_agent"] = (action[..., 2] >= 0.95).mean(axis=1).astype(np.float32)
        return arrays

    @staticmethod
    def _normalized_entropy(fractions: np.ndarray) -> np.ndarray:
        category_count = int(fractions.shape[-1])
        if category_count <= 1:
            return np.zeros(fractions.shape[:-1], dtype=np.float32)
        safe = np.where(fractions > 0.0, fractions, 1.0)
        entropy = -np.sum(np.where(fractions > 0.0, fractions * np.log(safe), 0.0), axis=-1)
        return (entropy / np.log(category_count)).astype(np.float32)

    def summary(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if not self.episodes:
            return {"episodes": 0, "status": "empty"}
        aoi = arrays["mean_aoi_ms_episode_agent"]
        cam = arrays["endpoint_cam_episode_agent"]
        final_count = min(100, aoi.shape[0])
        per_agent_aoi = aoi.mean(axis=0)
        per_agent_cam = cam.mean(axis=0)
        final_aoi = aoi[-final_count:].mean(axis=0)
        final_cam = cam[-final_count:].mean(axis=0)
        return {
            "status": "complete",
            "episodes": int(aoi.shape[0]),
            "endpoint_cam_definition": "success at the final slot of each episode",
            "mean_AoI_ms_per_agent": per_agent_aoi.tolist(),
            "endpoint_CAM_probability_per_agent": per_agent_cam.tolist(),
            "mean_AoI_ms": float(per_agent_aoi.mean()),
            "endpoint_CAM_probability": float(per_agent_cam.mean()),
            "worst_agent_mean_AoI_ms": float(per_agent_aoi.max()),
            "worst_agent_endpoint_CAM_probability": float(per_agent_cam.min()),
            "final_window_episodes": int(final_count),
            "final_window_mean_AoI_ms_per_agent": final_aoi.tolist(),
            "final_window_endpoint_CAM_probability_per_agent": final_cam.tolist(),
            "final_window_mean_AoI_ms": float(final_aoi.mean()),
            "final_window_endpoint_CAM_probability": float(final_cam.mean()),
            "final_window_worst_agent_mean_AoI_ms": float(final_aoi.max()),
            "final_window_worst_agent_endpoint_CAM_probability": float(final_cam.min()),
        }

    def save(self, run_dir: Path, prefix: str = "train_metrics") -> Dict[str, Any]:
        arrays = self.arrays()
        np.savez_compressed(run_dir / f"{prefix}.npz", **arrays)
        (run_dir / f"{prefix}_summary.json").write_text(json.dumps(self.summary(arrays), indent=2) + "\n", encoding="utf-8")
        learning_path = run_dir / "learning_diagnostics.json"
        learning_path.write_text(json.dumps(self.learning, indent=2, default=_json_default) + "\n", encoding="utf-8")
        if self.gradient_episodes:
            diagnostic_dir = run_dir / "diagnostics"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            gradient_arrays: Dict[str, np.ndarray] = {
                "episode": np.asarray([item["episode"] for item in self.gradient_episodes], dtype=np.int32),
                "actor_update_count": np.asarray([item["actor_update_count"] for item in self.gradient_episodes], dtype=np.int32),
                "finite_fraction": np.asarray([item["finite_fraction"] for item in self.gradient_episodes], dtype=np.float32),
                "all_finite": np.asarray([item["all_finite"] for item in self.gradient_episodes], dtype=np.bool_),
            }
            for field in self.GRADIENT_FIELDS:
                for statistic in ("mean", "p50", "max"):
                    key = f"{field}_{statistic}"
                    gradient_arrays[key] = np.stack([np.asarray(item[key], dtype=np.float32) for item in self.gradient_episodes])
            np.savez_compressed(diagnostic_dir / "actor_gradient_episode.npz", **gradient_arrays)
            (diagnostic_dir / "actor_gradient_schema.json").write_text(json.dumps({
                "schema_version": "actor_gradient_episode_v1",
                "axes": {"gradient_fields": ["episode", "agent"]},
                "statistics": ["mean", "p50", "max"],
                "global_gradient_semantics": "counterfactual gradient of the weighted live-action global objective; the run config states whether it contributes to the actor update",
                "ratios": "numerator_to_denominator L2 norm ratios with a finite zero-denominator guard",
                "cosines": "zero when either vector has zero L2 norm",
                "finite_gate": "training raises before checkpointing if any enabled actor-gradient diagnostic is missing or non-finite",
            }, indent=2) + "\n", encoding="utf-8")
        return {key: list(value.shape) for key, value in arrays.items()}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value).__name__)
