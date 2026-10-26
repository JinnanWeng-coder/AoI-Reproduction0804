"""Opt-in episode-boundary training recovery, separate from inference artifacts.

Same-config, same-code, same-device recovery only. Checkpoints contain all live
training state; restoring never performs a learning step or consumes a rollout.
Callers must hold the training unit lock before preparing/restoring a run.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .checkpointing import atomic_torch_save, capture_rng_state, restore_rng_state, _git_stamp
from .periodic_policy import digest, read, require, shared, timing, version_at

SCHEMA = "cam96k_full_training_state_v1"
ROLE = "periodic_full_training_checkpoint"
MODULES = ("actors", "critic", "value_norm", "global_critic", "task1_critics", "task2_critics",
           "global_value_norm", "task1_value_norm", "task2_value_norm")
OPTIMIZERS = ("critic_optimizer", "global_critic_optimizer", "task1_critic_optimizer", "task2_critic_optimizer")
COUNTERS = ("policy_version", "environment_steps", "update_count", "actor_optimizer_step_count")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def durable_torch_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def mappo_state(trainer):
    return {"modules": {k: getattr(trainer, k).state_dict() for k in MODULES if getattr(trainer, k) is not None},
            "optimizers": {k: getattr(trainer, k).state_dict() for k in OPTIMIZERS if getattr(trainer, k) is not None},
            "actor_optimizers": [o.state_dict() for o in trainer.actor_optimizers],
            "counters": {k: int(getattr(trainer, k)) for k in COUNTERS},
            "modes": {k: {name: module.training for name, module in getattr(trainer, k).named_modules()}
                      for k in MODULES if getattr(trainer, k) is not None}}


def load_mappo(trainer, state):
    require(set(state["modules"]) == {k for k in MODULES if getattr(trainer, k) is not None}, "MAPPO module identity mismatch")
    require(set(state["optimizers"]) == {k for k in OPTIMIZERS if getattr(trainer, k) is not None}, "MAPPO optimizer identity mismatch")
    require(len(state["actor_optimizers"]) == len(trainer.actor_optimizers), "MAPPO actor optimizer count mismatch")
    for key, value in state["modules"].items():
        getattr(trainer, key).load_state_dict(value, strict=True)
        modules = dict(getattr(trainer, key).named_modules())
        require(set(modules) == set(state["modes"][key]), "MAPPO mode structure mismatch")
        for name, mode in state["modes"][key].items():
            modules[name].training = mode
    for key, value in state["optimizers"].items():
        getattr(trainer, key).load_state_dict(value)
    for optimizer, value in zip(trainer.actor_optimizers, state["actor_optimizers"]):
        optimizer.load_state_dict(value)
    require(set(state["counters"]) == set(COUNTERS), "MAPPO counters missing")
    for key, value in state["counters"].items():
        setattr(trainer, key, int(value))


def maddpg_modules(agents, learner):
    modules = {}
    for i, agent in enumerate(agents):
        for key, value in vars(agent).items():
            if isinstance(value, torch.nn.Module):
                modules[f"agent{i}.{key}"] = value
    for key, value in vars(learner).items():
        if isinstance(value, torch.nn.Module):
            modules[f"learner.{key}"] = value
    return modules


def training_state(config, system):
    if config.algorithm == "mappo":
        env, trainer, rollout, metrics, _ = system
        state = {"trainer": mappo_state(trainer), "rollout": copy.deepcopy(vars(rollout))}
    else:
        env, agents, learner, replay, metrics, _ = system
        state = {"agents": [a.state_dict_full(include_actor=not config.maddpg_actor_sharing or i == 0)
                            for i, a in enumerate(agents)], "learner": learner.state_dict_full(), "replay": replay.state_dict(),
                 "modes": {k: m.training for k, m in maddpg_modules(agents, learner).items()}}
    state.update(environment=env.state_dict(), metrics=metrics.state_dict())
    return state


def restore_system(config, system, state):
    if config.algorithm == "mappo":
        env, trainer, rollout, metrics, _ = system
        load_mappo(trainer, state["trainer"])
        require(set(state["rollout"]) == set(vars(rollout)) and
                state["rollout"]["number_agents"] == rollout.number_agents and
                state["rollout"]["observation_dim"] == rollout.observation_dim, "rollout identity mismatch")
        vars(rollout).update(copy.deepcopy(state["rollout"]))
    else:
        env, agents, learner, replay, metrics, _ = system
        require(len(state["agents"]) == len(agents), "MADDPG agent count mismatch")
        for i, (agent, value) in enumerate(zip(agents, state["agents"])):
            agent.load_state_dict_full(value, load_actor=not config.maddpg_actor_sharing or i == 0)
        learner.load_state_dict_full(state["learner"])
        replay.load_state_dict(state["replay"])
        modules = maddpg_modules(agents, learner)
        require(set(modules) == set(state["modes"]), "MADDPG module modes mismatch")
        for key, mode in state["modes"].items():
            modules[key].training = mode
        require(len({id(a.actor) for a in agents}) == (1 if config.maddpg_actor_sharing else len(agents)), "shared actor alias lost")
    env.load_state_dict(state["environment"])
    metrics.load_state_dict(state["metrics"])


def selector_state(selector):
    if selector is None:
        return None
    from .training_selection import BEST
    best_path = selector.run / BEST
    return {"window": list(selector.window), "best": copy.deepcopy(selector.best), "episode": selector.episode,
            "rows": copy.deepcopy(selector.rows),
            "best_payload": torch.load(best_path, map_location="cpu", weights_only=False) if selector.best is not None else None}


def restore_selector(run, config, state):
    if state is None:
        return None
    from .training_selection import TrainingRewardBest
    selector = TrainingRewardBest.__new__(TrainingRewardBest)
    selector.run, selector.config = Path(run), config
    selector.window = deque(state["window"], maxlen=100)
    selector.best, selector.episode, selector.rows = copy.deepcopy(state["best"]), state["episode"], copy.deepcopy(state["rows"])
    return selector


def _index(records, interval, smoke):
    return {"schema": SCHEMA, "role": ROLE, "interval": interval, "smoke": smoke, "records": records}


def load_full_checkpoint(path, *, config=None, expected_seed=None, expected_episode=None, check_code=True):
    from ..config import config_from_dict, safe_run_dir
    path = Path(path).resolve()
    require(path.parent.name == "training_checkpoints" and path.name.startswith("training_episode_"), "wrong full checkpoint path/role")
    index = read(path.parent / "index.json")
    require(index.get("schema") == SCHEMA and index.get("role") == ROLE, "full checkpoint index role/schema mismatch")
    require([r["episode"] for r in index["records"]] == list(range(index["interval"], index["interval"] * (len(index["records"]) + 1), index["interval"])),
            "full checkpoint index coverage mismatch")
    entries = [r for r in index["records"] if r["file"] == path.name]
    require(len(entries) == 1, "full checkpoint is not uniquely indexed")
    record = entries[0]
    require(record["sha256"] == digest(path) and record["size"] == path.stat().st_size, "full checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(payload["config"])
    episode = payload["episode"]
    require(payload.get("schema") == SCHEMA and payload.get("checkpoint_role") == ROLE and payload.get("resume_supported") is True,
            "full checkpoint role/schema mismatch")
    require(type(episode) is int and 0 < episode <= cfg.episodes and episode % index["interval"] == 0 and
            path.name == f"training_episode_{episode:06d}.pt" and
            index["interval"] == payload["interval"] and index["smoke"] == payload["smoke"] == cfg.smoke and
            ((cfg.smoke and index["interval"] > 0) or (not cfg.smoke and index["interval"] == 500 and cfg.episodes == 2500)),
            "full checkpoint episode/interval mismatch")
    require(payload["config_hash"] == cfg.canonical_hash() and payload["algorithm"] == cfg.algorithm and
            payload["training_seed"] == cfg.seed and payload["actor_sharing"] == shared(cfg) and
            payload["policy_version"] == version_at(cfg, episode) and payload["environment_steps"] == episode * cfg.steps_per_episode and
            payload["save_timing"] == timing(cfg), "full checkpoint config/counters mismatch")
    require(config is None or config.canonical_hash() == cfg.canonical_hash(), "expected full checkpoint config mismatch")
    require(expected_seed is None or expected_seed == cfg.seed, "full checkpoint seed mismatch")
    require(expected_episode is None or expected_episode == episode, "full checkpoint episode mismatch")
    run = path.parent.parent
    require(run == safe_run_dir(cfg.output_root, cfg.run_name).resolve() and
            config_from_dict(read(run / "config.resolved.json")).canonical_hash() == cfg.canonical_hash(), "full checkpoint run identity mismatch")
    current = _git_stamp() if check_code else None
    provenance = read(run / "provenance.json")
    for key in ("reproduction_git_commit", "reproduction_git_branch", "reproduction_git_dirty"):
        require(payload.get(key) == provenance.get(key) == record.get(key), "full checkpoint provenance mismatch")
        if check_code:
            require(payload.get(key) == current[key], "full checkpoint current Git identity mismatch")
    require(cfg.smoke or payload["reproduction_git_dirty"] is False, "formal full checkpoint must use clean code")
    for key in ("episode", "config_hash", "training_seed", "policy_version", "environment_steps", "algorithm", "actor_sharing", "save_timing"):
        require(record[key] == payload[key], f"full checkpoint indexed {key} mismatch")
    state = payload["state"]
    require(len(state["metrics"]["episodes"]) == episode and state["environment"]["episode_index"] == episode - 1,
            "full checkpoint environment/metrics coverage mismatch")
    if cfg.algorithm == "mappo":
        require(state["trainer"]["counters"]["policy_version"] == payload["policy_version"] and
                state["trainer"]["counters"]["environment_steps"] + len(state["rollout"]["_records"]) == payload["environment_steps"], "full MAPPO counters mismatch")
        require(state["trainer"]["counters"]["update_count"] == payload["policy_version"] and
                state["trainer"]["counters"]["actor_optimizer_step_count"] == payload["policy_version"] * cfg.mappo_ppo_epochs * (1 if shared(cfg) else cfg.number_agents),
                "full MAPPO optimizer counters mismatch")
    else:
        require(state["replay"]["mem_cntr"] == payload["environment_steps"] and
                state["learner"]["learn_step_counter"] == max(0, payload["environment_steps"] - cfg.batch_size + 1), "full MADDPG counters mismatch")
        require(len(state["agents"]) == cfg.number_agents and
                sum("actor" in a for a in state["agents"]) == (1 if shared(cfg) else cfg.number_agents), "full MADDPG actor structure mismatch")
        require(state["replay"]["mem_size"] == cfg.replay_capacity and state["replay"]["n_agents"] == cfg.number_agents,
                "full replay capacity mismatch")
        if shared(cfg):
            require(state["learner"]["actor_optimizer_step_count"] == payload["policy_version"] and
                    state["learner"]["target_actor_update_count"] == payload["policy_version"], "full shared actor counters mismatch")
    selection = payload["selector"]
    require(selection is None or (selection["episode"] == episode and len(selection["rows"]) == episode and
            len(selection["window"]) == min(100, episode) and
            (episode < 100 or selection["best_payload"]["training_reward_selection"] == selection["best"])), "full selector state mismatch")
    if not cfg.smoke:
        require(selection is not None, "production full checkpoint requires training100 selector")
    if selection is not None and selection["best_payload"] is not None:
        best = selection["best_payload"]
        require(best["checkpoint_role"] == "best_training100" and best["training_completed"] is False and
                config_from_dict(best["config"]).canonical_hash() == cfg.canonical_hash() and
                best["episode"] == selection["best"]["selected_episode"] <= episode,
                "full embedded best100 identity mismatch")
    return cfg, payload


def verify_full_run(run, expected_episodes, *, final_path=None, check_code=True):
    run = Path(run)
    index = read(run / "training_checkpoints/index.json")
    require([r["episode"] for r in index["records"]] == list(expected_episodes), "full checkpoint episode set mismatch")
    require(sorted(p.name for p in (run/"training_checkpoints").glob("*.pt")) == sorted(r["file"] for r in index["records"]),
            "unexpected/missing full checkpoint files")
    for record in index["records"]:
        cfg, payload = load_full_checkpoint(run / "training_checkpoints" / record["file"], expected_episode=record["episode"], check_code=check_code)
        if final_path is not None and payload["episode"] == cfg.episodes:
            final = torch.load(final_path, map_location="cpu", weights_only=False)
            actors = payload["state"]["trainer"]["modules"]["actors"] if cfg.algorithm == "mappo" else None
            for i, actor in enumerate(final["actors"]):
                saved = {k.removeprefix(f"{i}."): t for k, t in actors.items() if k.startswith(f"{i}.")} if actors is not None else payload["state"]["agents"][i]["actor"]
                require(set(actor) == set(saved) and all(torch.equal(actor[k], saved[k]) for k in actor), "full final actor tensors differ")
    return index


def verify_full_index(run, expected_episodes):
    """Recheck accepted immutable bytes without allocating replay/metrics in eval workers.

    Deep state validation is performed before acceptance and at verify-training;
    exact accepted record equality binds this index to that deep verification.
    """
    run = Path(run)
    index = read(run / "training_checkpoints/index.json")
    require(index.get("schema") == SCHEMA and index.get("role") == ROLE and
            [r["episode"] for r in index["records"]] == list(expected_episodes), "full checkpoint index mismatch")
    require(sorted(p.name for p in (run/"training_checkpoints").glob("*.pt")) == sorted(r["file"] for r in index["records"]),
            "unexpected/missing full checkpoint files")
    for record in index["records"]:
        require(record["file"] == f"training_episode_{record['episode']:06d}.pt", "unsafe full checkpoint filename")
        path = run / "training_checkpoints" / record["file"]
        require(path.stat().st_size == record["size"] and digest(path) == record["sha256"], "accepted full checkpoint SHA256 mismatch")
    return index


class FullTrainingCheckpoints:
    def __init__(self, run, config, interval, smoke=False, records=None):
        require(config.checkpoint_mode == "policy_only" and type(interval) is int and interval > 0 and
                ((smoke and config.smoke) or (not smoke and not config.smoke and interval == 500 and config.episodes == 2500)),
                "full production checkpoints locked to 500/2500; explicit smoke required for short checkpoints")
        self.run, self.config, self.interval, self.smoke = Path(run), config, interval, smoke
        self.folder = self.run / "training_checkpoints"
        self.records = [] if records is None else records
        if records is None:
            self.folder.mkdir(exist_ok=False)
            atomic_json(self.folder / "index.json", _index([], interval, smoke))

    def observe(self, episode, system, selector, periodic):
        if episode % self.interval:
            return
        path = self.folder / f"training_episode_{episode:06d}.pt"
        require(not path.exists(), "refusing to overwrite full checkpoint")
        guard = path.with_suffix(".lock")
        with guard.open("x"):
            pass
        rng = capture_rng_state()
        try:
            cfg = self.config
            payload = {"schema": SCHEMA, "checkpoint_role": ROLE, "resume_supported": True,
                       "algorithm": cfg.algorithm, "actor_sharing": shared(cfg), "config": cfg.to_dict(), "config_hash": cfg.canonical_hash(),
                       "training_seed": cfg.seed, "episode": episode, "environment_steps": episode * cfg.steps_per_episode,
                       "policy_version": version_at(cfg, episode), "save_timing": timing(cfg), "interval": self.interval, "smoke": self.smoke,
                       "state": training_state(cfg, system), "selector": selector_state(selector), "rng": rng,
                       "periodic_index": read(periodic.folder / "index.json"), "full_records_before": copy.deepcopy(self.records),
                       "runtime": {"torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda}}
            payload.update(_git_stamp())
            durable_torch_save(payload, path)
            keys = ("episode", "environment_steps", "policy_version", "config_hash", "training_seed", "algorithm", "actor_sharing", "save_timing",
                    "reproduction_git_commit", "reproduction_git_branch", "reproduction_git_dirty")
            self.records.append({**{k: payload[k] for k in keys}, "file": path.name, "size": path.stat().st_size, "sha256": digest(path)})
            atomic_json(self.folder / "index.json", _index(self.records, self.interval, self.smoke))
        finally:
            restore_rng_state(rng)
            guard.unlink()


def prepare_full_resume(config, path):
    """Validate before mutation, then quarantine discarded tail; repeatable journal.

    A crash during rollback leaves RECOVERY_PENDING.json. Reissue the same resume
    to finish its file operations. Retained checkpoint files are never replaced.
    """
    cfg, payload = load_full_checkpoint(path, config=config)
    run = Path(path).resolve().parent.parent
    require(not (run / "COMPLETE.json").exists(), "refusing to resume a completed run")
    require(payload["runtime"] == {"torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda}, "resume runtime mismatch")
    episode = payload["episode"]
    full_records = payload["full_records_before"] + [next(r for r in read(run / "training_checkpoints/index.json")["records"] if r["episode"] == episode)]
    periodic_index = payload["periodic_index"]
    for folder, records in (("training_checkpoints", full_records), ("policies", periodic_index["records"])):
        for r in records:
            p = run / folder / r["file"]
            require(p.is_file() and digest(p) == r["sha256"], "resume prefix file SHA256 mismatch")
    pending = run / "RECOVERY_PENDING.json"
    if pending.exists():
        journal = read(pending)
        require(journal["checkpoint"] == str(Path(path).resolve()) and journal["sha256"] == digest(path), "pending recovery requires the same checkpoint")
    else:
        archive = run / "recoveries" / f"rollback_{episode:06d}_{time.time_ns()}"
        archive.mkdir(parents=True)
        moves = []
        for folder, records in (("training_checkpoints", full_records), ("policies", periodic_index["records"])):
            retained = {r["file"] for r in records}
            for p in (run / folder).iterdir():
                if p.name == "index.json":
                    shutil.copy2(p, archive / f"{folder}_index_before.json")
                elif p.is_file() and p.name not in retained:
                    moves.append(str(p.relative_to(run)))
        for name in ("policy_best_training.pt", "policy_final.pt", "training_reward_selection.csv", "train_metrics.npz"):
            if (run / name).exists():
                moves.append(name)
        journal = {"checkpoint": str(Path(path).resolve()), "sha256": digest(path), "episode": episode,
                   "archive": str(archive), "moves": moves}
        atomic_json(pending, journal)
    archive = Path(journal["archive"])
    require(archive.resolve().is_relative_to((run / "recoveries").resolve()), "unsafe recovery archive")
    for name in journal["moves"]:
        source, dest = run / name, archive / name
        require(source.resolve().is_relative_to(run.resolve()) and dest.resolve().is_relative_to(archive.resolve()), "unsafe recovery path")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() and source.exists():
            shutil.move(str(source), str(dest))
    atomic_json(run / "training_checkpoints/index.json", _index(full_records, payload["interval"], payload["smoke"]))
    atomic_json(run / "policies/index.json", periodic_index)
    if payload["selector"] is not None and payload["selector"]["best_payload"] is not None:
        atomic_torch_save(payload["selector"]["best_payload"], run / "policy_best_training.pt")
    atomic_json(archive / "recovery.json", {**journal, "status": "rollback_prepared"})
    pending.unlink()
    return run, payload, full_records


def resumed_periodic(run, config, payload):
    from .periodic_policy import PeriodicPolicySnapshots
    obj = PeriodicPolicySnapshots.__new__(PeriodicPolicySnapshots)
    obj.run, obj.config, obj.interval, obj.smoke = Path(run), config, payload["interval"], payload["smoke"]
    obj.folder, obj.records = Path(run) / "policies", payload["periodic_index"]["records"]
    return obj
