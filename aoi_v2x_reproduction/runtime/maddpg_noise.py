"""Opt-in environment-step exploration, lightweight policies and existing full-state recovery."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..config import config_from_dict
from . import shared_resume as sr
from .checkpointing import (atomic_torch_save, build_policy_payload, capture_rng_state,
                            restore_rng_state, _git_stamp)
from .training_selection import require

SCHEMA = "maddpg_noise_decay_episode_resume_v1"
SCHEDULE = {"name": "linear_environment_steps", "start": .3, "end": .05,
            "decay_environment_steps": 150000, "floor": .05,
            "counter": "completed_joint_environment_steps_before_action"}


def sigma(config, completed_environment_steps):
    require(type(completed_environment_steps) is int and completed_environment_steps >= 0,
            "environment step counter must be a nonnegative integer")
    if config.exploration_noise_schedule == "fixed":
        return float(config.exploration_noise)
    return float(config.exploration_noise_end + (config.exploration_noise - config.exploration_noise_end) *
                 max(1 - completed_environment_steps / config.exploration_noise_decay_environment_steps, 0))


def validate_request(config, training_reward_best, max_episodes=None, recover=False):
    # Accepted research configs carry paper_faithful_v4. Do not rewrite their profile/physics
    # or loosen global active-baseline validation; the new batch contract checks derivation.
    require(np.isfinite([config.exploration_noise, config.exploration_noise_end, config.exploration_noise_floor]).all() and
            0 <= config.exploration_noise_floor == config.exploration_noise_end <= config.exploration_noise and
            type(config.exploration_noise_decay_environment_steps) is int and config.exploration_noise_decay_environment_steps > 0,
            "invalid exploration schedule")
    require(config.algorithm == "modified_maddpg_tdec" and config.global_update_mode == "synchronous_joint" and
            config.checkpoint_mode == "policy_only" and config.exploration_noise_schedule == SCHEDULE["name"] and
            config.episodes >= 100 and config.global_actor_weight == 1 and training_reward_best and
            not config.is_formal_result and not recover, "noise-decay requires research TDec best100/episode recovery")
    require(max_episodes is None or type(max_episodes) is int and 0 < max_episodes <= config.episodes,
            "invalid partial episode limit")


def validate_resume_config(payload, config):
    require(payload.get("resume_schema") == SCHEMA and
            bool(payload.get("actor_sharing", False)) == config.maddpg_actor_sharing,
            "not a matching noise-decay complete-state checkpoint")
    old = config_from_dict(payload["config"])
    require(old.canonical_hash() == payload.get("config_hash"), "source config hash mismatch")
    a, b = old.to_dict(), config.to_dict()
    for key in ("output_root", "run_name"):
        a.pop(key, None)
        b.pop(key, None)
    require(a == b, "resume changes training configuration (only output_root/run_name may change)")


def read_resume(path, config):
    path = Path(path).resolve()
    p = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_config(p, config)
    require(p.get("checkpoint_interval") == 100 and not p.get("completed") and 0 < p["episode"] < config.episodes,
            "checkpoint is final or has wrong interval/episode")
    require(p.get("environment") is not None and isinstance(p.get("training100_selector"), dict) and
            p.get("completed_environment_steps") == p["episode"] * config.steps_per_episode and
            len(p["metrics"]["episodes"]) == p["episode"] == p["training100_selector"]["episode"],
            "missing or inconsistent environment/metrics/selector/step counter")
    require(p.get("runtime_identity") == sr.runtime_identity(config), "resume runtime/device changed")
    require(p.get("reproduction_git_commit") == _git_stamp()["reproduction_git_commit"], "resume implementation commit changed")
    require(path.name == f"episode_{p['episode']:06d}.pt", "checkpoint episode filename mismatch")
    index = p.get("policy_snapshot_index")
    require(isinstance(index, list) and [r["episode"] for r in index] == list(range(100, p["episode"] + 1, 100)),
            "checkpoint snapshot coverage mismatch")
    for row in index:
        check_identity(row)
    return p


def prepare_run(config, resume, interval=100):
    from .runner import _prepare_run, _write_json
    p = read_resume(resume, config) if resume else None
    run, _ = _prepare_run(config, None)
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    provenance.update(episode_resume_schema=SCHEMA, checkpoint_interval=100,
                      resume_source=str(Path(resume).resolve()) if resume else None,
                      resume_episode=p["episode"] if p else 0,
                      exploration_schedule={"start": config.exploration_noise, "end": config.exploration_noise_end,
                          "floor": config.exploration_noise_floor,
                          "decay_environment_steps": config.exploration_noise_decay_environment_steps,
                          "counter": SCHEDULE["counter"]})
    _write_json(run / "provenance.json", provenance)
    return run, p is not None


def check_identity(row):
    path = Path(row["path"])
    require(path.is_file() and path.stat().st_size == row["size"] and path.stat().st_mtime_ns == row["mtime_ns"],
            f"periodic policy source changed: {path}")


def snapshot(run, config, agents, episode, version):
    path = Path(run) / "policies" / f"policy_episode_{episode:06d}.pt"
    require(not path.exists(), "refusing to overwrite periodic policy")
    rng = capture_rng_state()
    try:
        p = build_policy_payload(config, agents, episode)
        p.update(checkpoint_role="periodic_training_policy", policy_version=int(version),
                 save_timing="after_episode_online_actor_updates", resume_supported=False)
        atomic_torch_save(p, path)
    finally:
        restore_rng_state(rng)
    return {"episode": episode, "policy_version": int(version), "path": str(path.resolve()),
            "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
            "actor_network_count": len(p["actors"]), "resumable": False}


def write_logs(run, metrics, index, config, learner):
    from .runner import _write_json
    run = Path(run)
    rows = [{"episode": i + 1, **e["exploration"]} for i, e in enumerate(metrics.episodes)]
    with (run / "exploration_noise.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    _write_json(run / "policy_snapshot_index.json", index)
    _write_json(run / "progress.json", {"episodes_completed": len(rows),
        "environment_steps": len(rows) * config.steps_per_episode,
        "learn_step_count": learner.learn_step_counter, "last_noise": rows[-1],
        "policy_snapshots": len(index), "training_completed": len(rows) == config.episodes})


def load_periodic_policy(path):
    """Policy-only strict loader used by the existing frozen-actor loading path."""
    path = Path(path).resolve()
    require(path.parent.name == "policies", "periodic policy must be in its run/policies directory")
    p = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(p["config"])
    index = json.loads((path.parent.parent / "policy_snapshot_index.json").read_text(encoding="utf-8"))
    matches = [r for r in index if Path(r["path"]).resolve() == path]
    require(len(matches) == 1, "policy not in snapshot index")
    row = matches[0]
    check_identity(row)
    expected = 1 if cfg.maddpg_actor_sharing else cfg.number_agents
    require(p.get("artifact_type") == "policy_only" and p.get("policy_schema_version") == "policy_artifact_v1" and
            p.get("checkpoint_role") == "periodic_training_policy" and p.get("resume_supported") is False and
            p.get("episode") == row["episode"] and p.get("policy_version") == row["policy_version"] and
            path.name == f"policy_episode_{row['episode']:06d}.pt" and
            p.get("environment_steps") == row["episode"] * cfg.steps_per_episode and
            cfg.algorithm == p.get("algorithm") == "modified_maddpg_tdec" and
            cfg.exploration_noise_schedule == SCHEDULE["name"] and len(p.get("actors", [])) == expected and
            bool(p.get("actor_sharing", False)) == cfg.maddpg_actor_sharing and
            all(state and all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in state.values()) for state in p["actors"]),
            "periodic policy metadata/weights mismatch")
    return cfg, p
