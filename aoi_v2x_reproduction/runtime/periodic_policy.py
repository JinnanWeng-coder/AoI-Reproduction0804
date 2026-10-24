"""Atomic, indexed inference-only snapshots, independent of training selection/resume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from .checkpointing import atomic_torch_save, build_policy_payload, capture_rng_state, restore_rng_state
from ..config import config_from_dict

ROLE = "periodic_policy_snapshot"
SCHEMA = "periodic_policy_snapshot_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def shared(config):
    return bool(config.mappo_actor_sharing if config.algorithm == "mappo" else config.maddpg_actor_sharing)


def timing(config):
    return "after_episode_after_scheduled_ppo_update" if config.algorithm == "mappo" else "after_episode_online_actor_updates"


def version_at(config, episode):
    if config.algorithm == "mappo":
        # Only the final boundary consumes a short residual rollout.
        return episode // config.mappo_rollout_episodes + int(episode == config.episodes and episode % config.mappo_rollout_episodes != 0)
    return max(0, episode * config.steps_per_episode - config.batch_size + 1) // config.policy_delay


class PeriodicPolicySnapshots:
    def __init__(self, run, config, interval=500, *, smoke=False):
        require(type(interval) is int and interval > 0 and config.checkpoint_mode == "policy_only", "periodic policies require positive interval and policy_only")
        require((smoke and config.smoke) or (not smoke and interval == 500 and config.episodes == 2500),
                "production snapshot interval/budget locked to 500/2500; short intervals require explicit smoke")
        self.run, self.config, self.interval, self.smoke = Path(run), config, interval, smoke
        self.records = []
        self.folder = self.run / "policies"
        self.folder.mkdir(exist_ok=False)
        atomic_json(self.folder / "index.json", {"schema": SCHEMA, "role": ROLE, "interval": interval, "smoke": smoke, "records": []})

    def observe(self, episode, policy, version):
        if episode % self.interval != 0:
            return
        require(version == version_at(self.config, episode), "snapshot policy version/save timing mismatch")
        path = self.folder / f"policy_episode_{episode:06d}.pt"
        require(not path.exists(), "refusing to overwrite periodic snapshot")
        guard = path.with_suffix(".lock")
        with guard.open("x"):
            pass
        rng = capture_rng_state()
        try:
            payload = build_policy_payload(self.config, policy, episode)
            payload.update(checkpoint_role=ROLE, snapshot_schema=SCHEMA, config_hash=self.config.canonical_hash(),
                           actor_sharing=shared(self.config), actor_network_count=1 if shared(self.config) else self.config.number_agents,
                           training_seed=self.config.seed, policy_version=int(version), save_timing=timing(self.config),
                           periodic_interval=self.interval, smoke=self.smoke, resume_supported=False,
                           training_complete_at_save=(episode == self.config.episodes))
            atomic_torch_save(payload, path)
            record = {k: payload[k] for k in ("checkpoint_role", "snapshot_schema", "algorithm", "actor_sharing", "actor_network_count",
                      "training_seed", "episode", "environment_steps", "policy_version", "save_timing", "config_hash",
                      "reproduction_git_commit", "reproduction_git_branch", "reproduction_git_dirty", "smoke")}
            record.update(file=path.name, sha256=digest(path), size=path.stat().st_size,
                          variant=self.config.mappo_variant if self.config.algorithm == "mappo" else "tdec")
            self.records.append(record)
            atomic_json(self.folder / "index.json", {"schema": SCHEMA, "role": ROLE, "interval": self.interval,
                                                     "smoke": self.smoke, "records": self.records})
        finally:
            restore_rng_state(rng)
            guard.unlink()


def load_periodic_policy(path, *, expected_seed=None, expected_episode=None, expected_config_hash=None):
    """Strict role + SHA256 + config/provenance validation, usable before training completes."""
    path = Path(path).resolve()
    index = read(path.parent / "index.json")
    require(index["schema"] == SCHEMA and index["role"] == ROLE, "wrong snapshot index role/schema")
    matches = [r for r in index["records"] if r["file"] == path.name]
    require(len(matches) == 1, "snapshot absent or duplicate in index")
    record = matches[0]
    require(record["sha256"] == digest(path) and record["size"] == path.stat().st_size, "snapshot SHA256/size mismatch")
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(data["config"])
    run = path.parent.parent
    require(cfg.canonical_hash() == data["config_hash"] == config_from_dict(read(run / "config.resolved.json")).canonical_hash(), "snapshot config hash mismatch")
    require(data["checkpoint_role"] == ROLE and data["snapshot_schema"] == SCHEMA and data["artifact_type"] == "policy_only" and
            data["policy_schema_version"] == "policy_artifact_v1" and data["resume_supported"] is False and
            data["algorithm"] == cfg.algorithm and data["training_seed"] == cfg.seed and
            data["episode"] > 0 and data["episode"] <= cfg.episodes and
            data["environment_steps"] == data["episode"] * cfg.steps_per_episode and
            path.name == f"policy_episode_{data['episode']:06d}.pt" and data["save_timing"] == timing(cfg) and
            data["policy_version"] == version_at(cfg, data["episode"]) and data["actor_sharing"] == shared(cfg) and
            data["periodic_interval"] == index["interval"] and data["episode"] % index["interval"] == 0 and
            data["smoke"] == index["smoke"] and data["training_complete_at_save"] == (data["episode"] == cfg.episodes),
            "snapshot role/seed/episode/version/sharing mismatch")
    require((index["smoke"] and cfg.smoke) or (index["interval"] == 500 and cfg.episodes == 2500 and not cfg.smoke), "snapshot interval/smoke contract mismatch")
    require(expected_seed is None or cfg.seed == expected_seed, "wrong snapshot seed")
    require(expected_episode is None or data["episode"] == expected_episode, "wrong snapshot episode")
    require(expected_config_hash is None or cfg.canonical_hash() == expected_config_hash, "wrong expected config hash")
    provenance = read(run / "provenance.json")
    for key in ("reproduction_git_commit", "reproduction_git_branch", "reproduction_git_dirty"):
        require(data[key] == provenance[key], "snapshot Git provenance mismatch")
    require(all(data[k] == v for k, v in record.items() if k not in {"file", "sha256", "size", "variant"}), "snapshot/index metadata mismatch")
    require(record["variant"] == (cfg.mappo_variant if cfg.algorithm == "mappo" else "tdec"), "snapshot variant mismatch")
    expected_count = 1 if shared(cfg) else cfg.number_agents
    require(len(data["actors"]) == data["actor_network_count"] == expected_count and
            all(s and all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in s.values()) for s in data["actors"]), "invalid snapshot actors")
    return cfg, data


def load_periodic_actors(path, device="cpu", **expected):
    cfg, data = load_periodic_policy(path, **expected)
    from ..algorithms.mappo.networks import HybridActor
    from ..algorithms.modified_maddpg.networks import ActorNetwork
    rng = capture_rng_state()
    try:
        actors = []
        for state in data["actors"]:
            actor = (HybridActor(cfg.state_dim, cfg.actor_hidden, cfg.n_rb, cfg.n_modes) if cfg.algorithm == "mappo" else
                     ActorNetwork(cfg.state_dim, cfg.actor_hidden, cfg.action_dim, cfg.actor_lr, "cpu", create_optimizer=False))
            actor.load_state_dict(state, strict=True)
            actors.append(actor.requires_grad_(False).to(device).eval())
        return cfg, torch.nn.ModuleList(actors)
    finally:
        restore_rng_state(rng)


def verify_periodic_run(run, expected_episodes, *, final_path=None):
    run = Path(run)
    index = read(run / "policies/index.json")
    require([r["episode"] for r in index["records"]] == list(expected_episodes), "snapshot episode coverage mismatch")
    require(sorted(p.name for p in (run / "policies").glob("*.pt")) == sorted(r["file"] for r in index["records"]), "unexpected/missing snapshot files")
    cfg = config_from_dict(read(run / "config.resolved.json"))
    for record in index["records"]:
        load_periodic_actors(run / "policies" / record["file"], expected_seed=cfg.seed,
                             expected_episode=record["episode"], expected_config_hash=cfg.canonical_hash())
    if final_path is not None:
        require(index["records"][-1]["episode"] == cfg.episodes, "missing final-episode periodic snapshot")
        _, last = load_periodic_policy(run / "policies" / index["records"][-1]["file"])
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        require(len(last["actors"]) == len(final["actors"]) and all(a.keys() == b.keys() and
                all(torch.equal(a[k], b[k]) for k in a) for a, b in zip(last["actors"], final["actors"])), "last periodic/final actor tensors differ")
    return index
