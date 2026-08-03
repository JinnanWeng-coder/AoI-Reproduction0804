"""Training, evaluation, checkpoint, and provenance orchestration."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from checkpointing import atomic_torch_save, build_payload, capture_rng_state, restore_rng_state
from Classes.buffer import ReplayBuffer
from Classes.Environment_Platoon import PaperEnviron
from Classes.legacy_adapter import LegacyEnviron
from config import ExperimentConfig, config_from_dict, resolve_config, safe_run_dir
from global_critic import Global_Critic
from local_critic import Agent
from metrics import MetricStore


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        requested = "cuda:0"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is not available")
    return device


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _make_environment(config):
    if config.profile == "legacy_release":
        return LegacyEnviron(config)
    return PaperEnviron(config)


def _make_system(config):
    device = resolve_device(config.device)
    config.device_resolved = str(device)
    seed_everything(config.seed, device)
    environment = _make_environment(config)
    agents = [Agent(config, index) for index in range(config.number_agents)]
    learner = Global_Critic(config, agents)
    replay = ReplayBuffer(config.replay_capacity, config.state_dim, config.action_dim, config.number_agents)
    metrics = MetricStore(config.number_agents, config.steps_per_episode)
    return environment, agents, learner, replay, metrics, device


def _source_manifest_digest() -> Optional[str]:
    path = Path(__file__).resolve().parent / "SOURCE_MANIFEST.json"
    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_run(config: ExperimentConfig, resume: Optional[str]) -> Tuple[Path, bool]:
    if resume:
        checkpoint = Path(resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        run_dir = checkpoint.parent.parent
        return run_dir, True
    run_dir = safe_run_dir(config.output_root, config.run_name or "unnamed")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    _write_json(run_dir / "config.resolved.json", config.to_dict())
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": torch.version.cuda,
        "source_manifest_sha256": _source_manifest_digest(),
        "profile": config.profile,
        "scenario": config.scenario.id,
        "seed": config.seed,
        "is_formal_result": bool(config.is_formal_result),
    }
    _write_json(run_dir / "provenance.json", provenance)
    (run_dir / "stdout.log").write_text("run started\n", encoding="utf-8")
    return run_dir, False


def _load_checkpoint(path: Path, config, agents, learner, replay, environment, metrics):
    payload = torch.load(path, map_location=learner.device, weights_only=False)
    if payload.get("config_hash") != config.canonical_hash():
        raise ValueError("checkpoint config hash does not match resolved config")
    for agent, state in zip(agents, payload["agents"]):
        agent.load_state_dict_full(state)
    learner.load_state_dict_full(payload["learner"])
    replay.load_state_dict(payload["replay"])
    if payload.get("environment") is not None and hasattr(environment, "load_state_dict"):
        environment.load_state_dict(payload["environment"])
    metrics.load_state_dict(payload["metrics"])
    restore_rng_state(payload["rng"])
    return payload


def _record_step(info: Dict[str, Any]) -> Dict[str, Any]:
    return {key: np.asarray(value).copy() for key, value in info.items() if key not in {"actions_decoded"}}


def train(config: ExperimentConfig, resume: Optional[str] = None, max_episodes: Optional[int] = None) -> Dict[str, Any]:
    run_dir, is_resume = _prepare_run(config, resume)
    environment, agents, learner, replay, metrics, device = _make_system(config)
    start_episode = 0
    if is_resume:
        payload = _load_checkpoint(Path(resume).expanduser().resolve(), config, agents, learner, replay, environment, metrics)
        start_episode = int(payload["episode"])
        if payload.get("completed"):
            raise RuntimeError("refusing to resume a completed run")

    stop_episode = config.episodes if max_episodes is None else min(config.episodes, int(max_episodes))
    for episode in range(start_episode, stop_episode):
        observations = environment.reset_episode(episode)
        task1_steps: List[np.ndarray] = []
        task2_steps: List[np.ndarray] = []
        global_steps: List[float] = []
        step_records: List[Dict[str, Any]] = []
        for _step in range(config.steps_per_episode):
            actions = np.asarray([agent.choose_action(observations[index], explore=True) for index, agent in enumerate(agents)], dtype=np.float32)
            next_observations, reward_global, reward_task1, reward_task2, terminated, info = environment.step(actions)
            replay.store_transition(observations.reshape(-1), actions.reshape(-1), reward_global, reward_task1, reward_task2, next_observations.reshape(-1), terminated)
            if replay.size >= config.batch_size:
                diagnostics = learner.learn(replay.sample_buffer(config.batch_size))
                metrics.append_learning(diagnostics)
            task1_steps.append(np.asarray(reward_task1, dtype=np.float32))
            task2_steps.append(np.asarray(reward_task2, dtype=np.float32))
            global_steps.append(float(reward_global))
            step_records.append(_record_step(info))
            observations = next_observations
        metrics.append_episode(step_records, task1_steps, task2_steps, global_steps)
        if ((episode + 1) % config.checkpoint_every == 0) or episode == stop_episode - 1:
            payload = build_payload(config, agents, learner, replay, environment, metrics, episode + 1, completed=False)
            atomic_torch_save(payload, run_dir / "checkpoints" / "latest.pt")

    if stop_episode < config.episodes:
        shapes = metrics.save(run_dir)
        return {"run_dir": str(run_dir), "episodes_completed": stop_episode, "interrupted": True, "metrics_shapes": shapes, "device": str(device)}

    shapes = metrics.save(run_dir)
    final_payload = build_payload(config, agents, learner, replay, environment, metrics, config.episodes, completed=True)
    atomic_torch_save(final_payload, run_dir / "checkpoints" / "best.pt")
    atomic_torch_save(final_payload, run_dir / "checkpoints" / "latest.pt")
    from analysis.audit_results import audit_run

    audit = audit_run(run_dir, require_complete=False)
    if not audit["ok"]:
        raise RuntimeError("result audit failed before completion marker: " + json.dumps(audit, sort_keys=True))
    complete = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_formal_result": bool(config.is_formal_result),
        "profile": config.profile,
        "scenario": config.scenario.id,
        "seed": config.seed,
        "episodes": config.episodes,
        "metrics_shapes": shapes,
        "audit": audit,
    }
    _write_json(run_dir / "COMPLETE.json", complete)
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as handle:
        handle.write("run completed\n")
    return {"run_dir": str(run_dir), "episodes": config.episodes, "device": str(device), "audit": audit}


def _eval_id(seeds: Iterable[int], episodes: int) -> str:
    return "eval_" + "-".join(str(int(seed)) for seed in seeds) + f"_ep{int(episodes)}"


def evaluate_from_checkpoint(config: ExperimentConfig, checkpoint: str, eval_episodes: int, eval_seeds: Optional[List[int]] = None) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    run_dir = checkpoint_path.parent.parent
    caller_rng = capture_rng_state()
    checkpoint_preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_config = config_from_dict(checkpoint_preview["config"])
    if config.device != "auto":
        saved_config.device = config.device
    config = saved_config
    if eval_seeds is None:
        eval_seeds = [config.seed + 1000, config.seed + 1001]
    eval_id = _eval_id(eval_seeds, eval_episodes)
    eval_dir = run_dir / "eval" / eval_id
    if eval_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing eval directory: {eval_dir}")
    eval_dir.mkdir(parents=True, exist_ok=False)
    environment, agents, learner, replay, metrics, device = _make_system(config)
    payload = _load_checkpoint(checkpoint_path, config, agents, learner, replay, environment, metrics)
    for agent in agents:
        agent.actor.eval()
    raw_aoi = []
    raw_success = []
    raw_demand = []
    raw_v2i = []
    raw_v2v = []
    raw_interference = []
    raw_power = []
    raw_rb = []
    raw_mode = []
    try:
        for seed in eval_seeds:
            seed_aoi = []
            seed_success = []
            seed_demand = []
            seed_v2i = []
            seed_v2v = []
            seed_interference = []
            seed_power = []
            seed_rb = []
            seed_mode = []
            for episode in range(int(eval_episodes)):
                environment.reset(int(seed) + episode)
                observations = environment.get_observations()
                episode_aoi = []
                episode_success = []
                episode_demand = []
                episode_v2i = []
                episode_v2v = []
                episode_interference = []
                episode_power = []
                episode_rb = []
                episode_mode = []
                for _step in range(config.steps_per_episode):
                    actions = np.asarray([agent.choose_action(observations[index], explore=False) for index, agent in enumerate(agents)], dtype=np.float32)
                    observations, _rg, _t1, _t2, _done, info = environment.step(actions)
                    episode_aoi.append(info["aoi_ms"])
                    episode_success.append(info["success"])
                    episode_demand.append(info["remaining_demand"])
                    episode_v2i.append(info["v2i_rate"])
                    episode_v2v.append(info["v2v_rate"])
                    episode_interference.append(info["interference_db"])
                    episode_power.append(info["power_dbm"])
                    episode_rb.append(info["rb"])
                    episode_mode.append(info["mode"])
                seed_aoi.append(episode_aoi)
                seed_success.append(episode_success)
                seed_demand.append(episode_demand)
                seed_v2i.append(episode_v2i)
                seed_v2v.append(episode_v2v)
                seed_interference.append(episode_interference)
                seed_power.append(episode_power)
                seed_rb.append(episode_rb)
                seed_mode.append(episode_mode)
            raw_aoi.append(seed_aoi)
            raw_success.append(seed_success)
            raw_demand.append(seed_demand)
            raw_v2i.append(seed_v2i)
            raw_v2v.append(seed_v2v)
            raw_interference.append(seed_interference)
            raw_power.append(seed_power)
            raw_rb.append(seed_rb)
            raw_mode.append(seed_mode)
    finally:
        # Evaluation uses private environment generators and must not perturb
        # the caller's training RNG streams.
        restore_rng_state(caller_rng)

    arrays = {
        "aoi_ms": np.asarray(raw_aoi, dtype=np.float32),
        "success": np.asarray(raw_success, dtype=np.float32),
        "remaining_demand": np.asarray(raw_demand, dtype=np.float32),
        "v2i_rate": np.asarray(raw_v2i, dtype=np.float32),
        "v2v_rate": np.asarray(raw_v2v, dtype=np.float32),
        "interference_db": np.asarray(raw_interference, dtype=np.float32),
        "power_dbm": np.asarray(raw_power, dtype=np.float32),
        "rb": np.asarray(raw_rb, dtype=np.int64),
        "mode": np.asarray(raw_mode, dtype=np.int64),
    }
    np.savez_compressed(eval_dir / "metrics.npz", **arrays)
    try:
        import scipy.io

        scipy.io.savemat(eval_dir / "metrics.mat", arrays)
    except ImportError:
        pass
    per_seed_aoi = arrays["aoi_ms"].mean(axis=(1, 2, 3))
    per_seed_success = arrays["success"][:, :, -1, :].mean(axis=(1, 2))
    summary = {
        "eval_id": eval_id,
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "eval_episodes": int(eval_episodes),
        "mean_AoI_ms_per_seed": per_seed_aoi.tolist(),
        "CAM_success_probability_per_seed": per_seed_success.tolist(),
        "mean_AoI_ms": float(per_seed_aoi.mean()),
        "CAM_success_probability": float(per_seed_success.mean()),
        "is_frozen_eval": True,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(eval_dir / "summary.json", summary)
    _write_json(eval_dir / "EVAL_COMPLETE.json", summary)
    return {"eval_dir": str(eval_dir), **summary}
