"""Opt-in training-history selection, never validation/test selection or resume."""
from __future__ import annotations

import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .checkpointing import atomic_torch_save, build_policy_payload, capture_rng_state, restore_rng_state

RULE = {"schema": "training100_v1", "window": 100, "first_eligible_episode": 100,
        "criterion": "maximize_mean_of_last_100_episode_combined_training_rewards",
        "episode_reward": "float64 mean_slot_platoon(task1+task2+1*global) over logged float32 streams",
        "ties": "earliest_strict_improvement", "selection_unit": "one_training_seed",
        "test_or_validation_used": False, "frozen_policy_evaluation": False}
BEST = "policy_best_training.pt"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_request(config, resume=None, max_episodes=None, recover=False):
    require(not resume and not recover and max_episodes is None, "training100 does not support resume/partial/recovery")
    require(config.episodes >= 100 and config.global_actor_weight == 1 and not config.is_formal_result,
            "training100 requires >=100 episodes, weight1, explicit research run")
    require(config.checkpoint_mode in {"policy_only", "resumable"}, "training100 requires saved final policy")
    require(config.algorithm in {"mappo", "modified_maddpg_tdec"}, "training100 unsupported algorithm")


def episode_reward(task1, task2, global_rewards):
    a, b, c = [np.asarray(x, dtype=np.float64) for x in (task1, task2, global_rewards)]
    require(a.ndim == 2 and a.shape == b.shape and c.shape == a.shape[:1] and a.size > 0,
            "training reward shape mismatch")
    require(all(np.isfinite(x).all() for x in (a, b, c)), "nonfinite training reward")
    return float((a + b + c[:, None]).mean(dtype=np.float64))


class TrainingRewardBest:
    def __init__(self, run_dir, config, *, resume_supported=False):
        self.run = Path(run_dir)
        self.config = config
        self.window = deque(maxlen=100)
        self.best = None
        self.episode = 0
        self.rows = []
        self.resume_supported = bool(resume_supported)
        provenance_path = self.run / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance.update(training_reward_selection=RULE, checkpoint_selection=RULE,
                          training100_resume_supported=self.resume_supported)
        from .runner import _write_json
        _write_json(provenance_path, provenance)  # Only the newly initialized run.

    def state_dict(self):
        """Self-contained selection history, including the best actor at this point."""
        return {"rule": RULE, "episode": self.episode, "window": list(self.window),
                "best": self.best, "rows": self.rows,
                "best_payload": torch.load(self.run / BEST, map_location="cpu", weights_only=False)
                if self.best is not None else None}

    def load_state_dict(self, state):
        require(state.get("rule") == RULE and len(state["rows"]) == state["episode"],
                "selector checkpoint rule/history mismatch")
        require(len(state["window"]) == min(100, state["episode"]), "selector window mismatch")
        require([r["episode"] for r in state["rows"]] == list(range(1, state["episode"] + 1)),
                "selector episode sequence mismatch")
        expected_window = [(r["episode_reward"], r["policy_version_before_episode"], r["policy_version_saved"])
                           for r in state["rows"][-100:]]
        require([tuple(x) for x in state["window"]] == expected_window, "selector window/history mismatch")
        self.episode = int(state["episode"])
        self.window = deque(expected_window, maxlen=100)
        self.rows = list(state["rows"])
        self.best = state["best"]
        best = state["best_payload"]
        require((best is None) == (self.best is None) == (self.episode < 100), "missing best state")
        if best is not None:
            require(best["training_reward_selection"] == self.best and
                    best["episode"] == self.best["selected_episode"] <= self.episode,
                    "selector best artifact mismatch")
            # Only runtime naming changes on an explicit fork; preserve all actor tensors.
            best["config"] = self.config.to_dict()
            best["training_completed"] = False
            atomic_torch_save(best, self.run / BEST)

    def observe(self, task1, task2, global_rewards, policy, version_before, version_saved, timing):
        self.episode += 1
        reward = episode_reward(task1, task2, global_rewards)
        self.window.append((reward, int(version_before), int(version_saved)))
        score = float(np.mean([v[0] for v in self.window], dtype=np.float64)) if len(self.window) == 100 else None
        improved = score is not None and (self.best is None or score > self.best["score"])
        row = {"episode": self.episode, "episode_reward": reward, "rolling100": score,
               "policy_version_before_episode": int(version_before), "policy_version_saved": int(version_saved),
               "new_best": improved, "save_timing": timing}
        self.rows.append(row)
        if improved:
            self.best = {"rule": RULE, "selected_episode": self.episode,
                         "window_start": self.episode - 99, "window_end": self.episode, "score": score,
                         "policy_version": int(version_saved), "save_timing": timing,
                         "window_policy_version_min": min(v[1] for v in self.window),
                         "window_policy_version_max": max(v[2] for v in self.window)}
            self._save(policy, BEST, self.episode, self.best, False)

    def _save(self, policy, name, episode, selection, completed):
        # Serialization/building metadata must not consume the training RNG.
        rng = capture_rng_state()
        try:
            payload = build_policy_payload(self.config, policy, episode)
            payload.update(training_reward_selection=selection, training_completed=completed,
                           checkpoint_role="best_training100" if name == BEST else "final_training100",
                           selected_episode=episode, policy_version=selection["policy_version"])
            atomic_torch_save(payload, self.run / name)
        finally:
            restore_rng_state(rng)

    def finish(self, policy, final_version):
        require(self.episode == self.config.episodes and self.best is not None, "incomplete training100 budget")
        self._save(policy, "policy_final.pt", self.episode,
                   {"rule": RULE, "policy_version": int(final_version), "save_timing": "after_all_training_updates"}, True)
        best = torch.load(self.run / BEST, map_location="cpu", weights_only=False)
        require(best["training_reward_selection"] == self.best, "best artifact/selector mismatch")
        best["training_completed"] = True
        atomic_torch_save(best, self.run / BEST)
        with (self.run / "training_reward_selection.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return {"rule": RULE, "best": self.best, "policy_best": BEST, "policy_final": "policy_final.pt",
                "final_episode": self.episode, "final_policy_version": int(final_version), "resume_supported": self.resume_supported}


def load_training_policy(path):
    """Load/validate portable best or final for either family; no rollout/resume."""
    from ..config import config_from_dict
    path = Path(path).resolve()
    require(path.name in {BEST, "policy_final.pt"}, "not a training100 policy artifact")
    run = path.parent
    complete = json.loads((run / "COMPLETE.json").read_text(encoding="utf-8"))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(payload["config"])
    raw = json.loads((run / "config.resolved.json").read_text(encoding="utf-8"))
    selection = complete.get("training_reward_selection", {})
    require(complete.get("status") == "complete" and selection.get("rule") == RULE and
            selection.get("final_episode") == cfg.episodes and payload.get("training_completed") is True and
            complete.get("config_hash") == cfg.canonical_hash() == config_from_dict(raw).canonical_hash(),
            "training100 completion/config mismatch")
    role = "best_training100" if path.name == BEST else "final_training100"
    expected_episode = selection["best"]["selected_episode"] if path.name == BEST else cfg.episodes
    require(payload.get("artifact_type") == "policy_only" and payload.get("policy_schema_version") == "policy_artifact_v1" and
            payload.get("algorithm") == cfg.algorithm == complete.get("algorithm") and payload.get("checkpoint_role") == role and
            payload.get("episode") == payload.get("selected_episode") == expected_episode and
            payload.get("environment_steps") == expected_episode * cfg.steps_per_episode and
            payload.get("reproduction_git_commit") == complete.get("reproduction_git_commit"),
            "training100 role/episode/provenance mismatch")
    require(payload["training_reward_selection"].get("rule") == RULE, "selection rule mismatch")
    if path.name == BEST:
        require(payload["training_reward_selection"] == selection["best"] and 100 <= expected_episode <= cfg.episodes,
                "best selection metadata mismatch")
        require(payload["policy_version"] == selection["best"]["policy_version"], "best policy version mismatch")
    else:
        require(payload["policy_version"] == selection["final_policy_version"], "final policy version mismatch")
    shared = bool(cfg.mappo_actor_sharing) if cfg.algorithm == "mappo" else bool(cfg.maddpg_actor_sharing)
    if cfg.algorithm != "mappo":
        require(bool(payload.get("actor_sharing", False)) == shared and bool(complete.get("actor_sharing",False)) == shared,
                "MADDPG sharing identity mismatch")
        if shared:
            require(payload.get("model_id") == complete.get("model_id") == "shared_maddpg_tdec" and
                    payload.get("sharing_schema") == "shared_maddpg_tdec_v1", "shared MADDPG artifact schema")
    expected = 1 if shared else cfg.number_agents
    require(isinstance(payload.get("actors"), list) and len(payload["actors"]) == expected and all(
        state and all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in state.values()) for state in payload["actors"]),
        "training100 invalid actor states")
    return cfg, payload


def load_training_actors(path, device="cpu"):
    """Strict architecture loading only; no environment, rollout or training."""
    from ..algorithms.mappo.networks import HybridActor
    from ..algorithms.modified_maddpg.networks import ActorNetwork
    cfg, payload = load_training_policy(path)
    rng = capture_rng_state()
    try:
        modules = []
        for state in payload["actors"]:
            actor = (HybridActor(cfg.state_dim, cfg.actor_hidden, cfg.n_rb, cfg.n_modes) if cfg.algorithm == "mappo" else
                     ActorNetwork(cfg.state_dim, cfg.actor_hidden, cfg.action_dim, cfg.actor_lr, "cpu", create_optimizer=False))
            actor.load_state_dict(state, strict=True)
            actor.requires_grad_(False)
            modules.append(actor.to(device).eval())
        return cfg, torch.nn.ModuleList(modules)
    finally:
        restore_rng_state(rng)
