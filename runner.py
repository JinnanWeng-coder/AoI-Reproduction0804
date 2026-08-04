"""Training, evaluation, checkpoint, and provenance orchestration."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import random
import subprocess
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


EVAL_PURPOSE_SEEDS = {
    "validation": [201, 202, 203, 204, 205, 206],
    "final_test": [101, 102, 103, 104, 105, 106],
}
EVAL_STATISTICS_SCHEMA_VERSION = "eval_seed_cluster_v1"


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
    metrics = MetricStore(config.number_agents, config.steps_per_episode, config.global_actor_weight)
    return environment, agents, learner, replay, metrics, device


def _source_manifest_digest() -> Optional[str]:
    path = Path(__file__).resolve().parent / "SOURCE_MANIFEST.json"
    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_metadata() -> Dict[str, Any]:
    """Return reproducibility metadata without making Git state changes."""
    root = Path(__file__).resolve().parent

    def _git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    tracked = _git("ls-files", "-z")
    digest = hashlib.sha256()
    if tracked:
        for name in sorted(item for item in tracked.split("\x00") if item):
            path = root / name
            digest.update(name.replace(os.sep, "/").encode("utf-8"))
            digest.update(b"\x00")
            if path.is_file():
                digest.update(path.read_bytes())
    device_names = []
    cuda_driver = None
    if torch.cuda.is_available():
        device_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        try:
            driver_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            cuda_driver = driver_result.stdout.strip().splitlines()[0] if driver_result.stdout.strip() else None
        except (OSError, subprocess.CalledProcessError, IndexError):
            cuda_driver = None
    return {
        "reproduction_git_commit": _git("rev-parse", "HEAD") or None,
        "reproduction_git_branch": _git("branch", "--show-current") or None,
        "reproduction_git_dirty": bool(_git("status", "--porcelain", "--untracked-files=all")),
        "reproduction_tracked_tree_sha256": digest.hexdigest() if tracked else None,
        "gpu_names": device_names,
        "cuda_driver": cuda_driver,
    }


def _prepare_run(config: ExperimentConfig, resume: Optional[str]) -> Tuple[Path, bool]:
    if resume:
        checkpoint = Path(resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if config.is_formal_result and _git_metadata().get("reproduction_git_dirty"):
            raise RuntimeError("formal results require a clean reproduction Git worktree")
        run_dir = checkpoint.parent.parent
        return run_dir, True
    run_dir = safe_run_dir(config.output_root, config.run_name or "unnamed")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    git = _git_metadata()
    if config.is_formal_result and git.get("reproduction_git_dirty"):
        raise RuntimeError("formal results require a clean reproduction Git worktree")
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
        "semantic_version": config.semantic_version,
        "config_hash": config.canonical_hash(),
        "scenario": config.scenario.id,
        "seed": config.seed,
        "is_formal_result": bool(config.is_formal_result),
        "smoke": bool(config.smoke),
        "eval_protocol": config.eval_protocol,
        "eval_warmup_episodes": int(config.eval_warmup_episodes),
        "global_reward_normalization": config.global_reward_normalization,
        "mobility_model": config.mobility_model,
        "gap_definition": config.gap_definition,
        "vehicle_length_m": float(config.vehicle_length_m),
        "effective_center_spacing_m": float(config.effective_center_spacing_m),
        "statistics_schema_version": config.statistics_schema_version,
    }
    provenance.update(git)
    _write_json(run_dir / "provenance.json", provenance)
    (run_dir / "stdout.log").write_text("run started\n", encoding="utf-8")
    return run_dir, False


def _load_checkpoint(path: Path, config, agents, learner, replay, environment, metrics):
    payload = torch.load(path, map_location=learner.device, weights_only=False)
    checkpoint_semantic_version = payload.get("semantic_version")
    checkpoint_version = int(payload.get("checkpoint_version", 0))
    legacy_compat = False
    if checkpoint_semantic_version is None:
        if config.profile == "legacy_release" and checkpoint_version in {1, 2}:
            # Pre-remediation legacy checkpoints predate the semantic marker;
            # their byte-preserved environment remains an explicit supported
            # compatibility path.  Paper checkpoints never receive this
            # fallback.
            checkpoint_semantic_version = "legacy_release_v1"
            legacy_compat = True
        else:
            raise ValueError("checkpoint semantic_version is missing; paper checkpoints are rejected")
    if checkpoint_semantic_version != config.semantic_version:
        raise ValueError(
            "checkpoint semantic_version mismatch: "
            f"checkpoint={checkpoint_semantic_version!r}, resolved={config.semantic_version!r}"
        )
    if config.profile == "legacy_release" and checkpoint_version < 3:
        legacy_compat = legacy_compat or not isinstance(payload.get("config"), dict) or "gap_definition" not in payload.get("config", {})
    if config.profile == "paper_faithful" and checkpoint_version != 3:
        raise ValueError(
            "paper_faithful_v3 requires checkpoint_version=3; "
            f"received {checkpoint_version}"
        )
    if config.profile == "legacy_release" and checkpoint_version not in {1, 2, 3}:
        raise ValueError(f"unsupported legacy checkpoint_version={checkpoint_version}")
    if payload.get("config_hash") != config.canonical_hash():
        raw_config = payload.get("config")
        raw_hash = None
        if legacy_compat and isinstance(raw_config, dict):
            raw_hash = hashlib.sha256(json.dumps(raw_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
        if raw_hash != payload.get("config_hash"):
            raise ValueError("checkpoint config hash does not match resolved config")
    if checkpoint_version >= 3:
        if payload.get("gap_definition") != config.gap_definition:
            raise ValueError("checkpoint gap_definition does not match resolved config")
        if float(payload.get("vehicle_length_m", -1.0)) != float(config.vehicle_length_m):
            raise ValueError("checkpoint vehicle_length_m does not match resolved config")
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
    # String metadata belongs in config/provenance, not in numeric NPZ
    # tensors.  The raw per-RB arrays remain available for audit/smoke while
    # this filter prevents accidental object/string arrays.
    return {
        key: np.asarray(value).copy()
        for key, value in info.items()
        if key not in {"actions_decoded", "global_reward_normalization"}
    }


def train(config: ExperimentConfig, resume: Optional[str] = None, max_episodes: Optional[int] = None) -> Dict[str, Any]:
    run_dir, is_resume = _prepare_run(config, resume)
    environment, agents, learner, replay, metrics, device = _make_system(config)
    start_episode = 0
    if is_resume:
        payload = _load_checkpoint(Path(resume).expanduser().resolve(), config, agents, learner, replay, environment, metrics)
        start_episode = int(payload["episode"])
        if payload.get("completed"):
            raise RuntimeError("refusing to resume a completed run")
    elif hasattr(environment, "reset_world") and not getattr(environment, "_world_initialized", False):
        environment.reset_world(config.seed)

    stop_episode = config.episodes if max_episodes is None else min(config.episodes, int(max_episodes))
    for episode in range(start_episode, stop_episode):
        if hasattr(environment, "start_episode"):
            observations = environment.start_episode(episode)
        else:
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
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_formal_result": bool(config.is_formal_result),
        "profile": config.profile,
        "semantic_version": config.semantic_version,
        "config_hash": config.canonical_hash(),
        "reproduction_git_commit": json.loads((run_dir / "provenance.json").read_text(encoding="utf-8")).get("reproduction_git_commit"),
        "reproduction_git_branch": json.loads((run_dir / "provenance.json").read_text(encoding="utf-8")).get("reproduction_git_branch"),
        "gap_definition": config.gap_definition,
        "vehicle_length_m": float(config.vehicle_length_m),
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eval_id(checkpoint_hash: str, purpose: str, protocol: str, warmup_episodes: int, seeds: Iterable[int], episodes: int, noise: float) -> str:
    seed_token = "-".join(str(int(seed)) for seed in seeds)
    noise_token = str(noise).replace(".", "p")
    return f"eval_{purpose}_ckpt{checkpoint_hash[:12]}_{protocol}_warm{int(warmup_episodes)}_s{seed_token}_ep{int(episodes)}_noise{noise_token}"


def _mean_sd_ci95(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    means = values.mean(axis=1)
    if values.shape[1] > 1:
        sd = values.std(axis=1, ddof=1)
    else:
        sd = np.zeros(values.shape[0], dtype=np.float64)
    ci = 1.96 * sd / np.sqrt(max(1, values.shape[1]))
    return means, sd, ci


def evaluate_from_checkpoint(
    config: ExperimentConfig,
    checkpoint: str,
    eval_episodes: int,
    eval_seeds: Optional[List[int]] = None,
    eval_purpose: str = "final_test",
) -> Dict[str, Any]:
    if eval_purpose not in EVAL_PURPOSE_SEEDS:
        raise ValueError(f"eval_purpose must be one of {sorted(EVAL_PURPOSE_SEEDS)}")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    run_dir = checkpoint_path.parent.parent
    caller_rng = capture_rng_state()
    checkpoint_hash = _sha256_file(checkpoint_path)
    checkpoint_preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    requested_device = config.device
    saved_config = config_from_dict(checkpoint_preview["config"])
    if requested_device != "auto":
        saved_config.device = requested_device
    config = saved_config
    if eval_seeds is None:
        eval_seeds = list(EVAL_PURPOSE_SEEDS[eval_purpose])
    eval_seeds = [int(seed) for seed in eval_seeds]
    if not eval_seeds or len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("eval_seeds must be non-empty and unique")
    if config.is_formal_result and eval_seeds != EVAL_PURPOSE_SEEDS[eval_purpose]:
        raise ValueError(f"formal {eval_purpose} evaluation requires seeds {EVAL_PURPOSE_SEEDS[eval_purpose]}")
    if int(eval_episodes) < 1:
        raise ValueError("eval_episodes must be positive")
    warmup_episodes = int(config.eval_warmup_episodes)
    eval_noise = 0.0
    eval_id = _eval_id(checkpoint_hash, eval_purpose, config.eval_protocol, warmup_episodes, eval_seeds, eval_episodes, eval_noise)
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

            # One cold reset per held-out seed.  Warm-up and scored episodes
            # then advance the same world sequentially, preserving AoI,
            # previous interference, mobility, and slow fading history.
            environment.reset_world(seed)
            for warmup_index in range(warmup_episodes):
                observations = environment.start_episode(warmup_index)
                for _step in range(config.steps_per_episode):
                    actions = np.asarray([agent.choose_action(observations[index], explore=False) for index, agent in enumerate(agents)], dtype=np.float32)
                    observations, _rg, _t1, _t2, _done, _info = environment.step(actions)

            for episode in range(int(eval_episodes)):
                episode_index = warmup_episodes + episode
                observations = environment.start_episode(episode_index)
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
    # Episodes are repeated frames within one held-out world, not independent
    # inferential units.  Keep their raw values and report only descriptive
    # within-seed SD.  Study-level CI is computed later across training runs.
    aoi_episode_seed_agent = arrays["aoi_ms"].mean(axis=2)  # seed x episode x agent
    endpoint_episode_seed_agent = arrays["success"][:, :, -1, :]
    per_seed_aoi_agent = aoi_episode_seed_agent.mean(axis=1)
    per_seed_success_agent = endpoint_episode_seed_agent.mean(axis=1)
    per_seed_aoi = per_seed_aoi_agent.mean(axis=1)
    per_seed_success = per_seed_success_agent.mean(axis=1)
    sd_aoi = aoi_episode_seed_agent.mean(axis=2).std(axis=1, ddof=1) if int(eval_episodes) > 1 else np.zeros(len(eval_seeds))
    sd_success = endpoint_episode_seed_agent.mean(axis=2).std(axis=1, ddof=1) if int(eval_episodes) > 1 else np.zeros(len(eval_seeds))
    checkpoint_reference = os.path.relpath(checkpoint_path, run_dir).replace(os.sep, "/")
    formal_eval = bool(config.is_formal_result and eval_purpose == "final_test")
    summary = {
        "eval_id": eval_id,
        "eval_purpose": eval_purpose,
        "statistics_schema_version": EVAL_STATISTICS_SCHEMA_VERSION,
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "eval_episodes": int(eval_episodes),
        "eval_protocol": config.eval_protocol,
        "eval_warmup_episodes": warmup_episodes,
        "eval_noise": eval_noise,
        "semantic_version": config.semantic_version,
        "profile": config.profile,
        "scenario": config.scenario.id,
        "training_seed": int(config.seed),
        "config_hash": config.canonical_hash(),
        "global_reward_normalization": config.global_reward_normalization,
        "mobility_model": config.mobility_model,
        "gap_definition": config.gap_definition,
        "vehicle_length_m": float(config.vehicle_length_m),
        "effective_center_spacing_m": float(config.effective_center_spacing_m),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": checkpoint_reference,
        "checkpoint_path_is_relative_to_run": True,
        "raw_metric_axes": {
            "aoi_ms": ["eval_seed", "scored_episode", "slot", "agent"],
            "success": ["eval_seed", "scored_episode", "slot", "agent"],
            "remaining_demand": ["eval_seed", "scored_episode", "slot", "agent"],
        },
        "mean_AoI_ms_per_seed_agent": per_seed_aoi_agent.tolist(),
        "mean_AoI_ms_per_seed": per_seed_aoi.tolist(),
        "sd_AoI_ms_per_seed": sd_aoi.tolist(),
        "sd_AoI_ms_per_seed_semantics": "descriptive_across_scored_episodes_within_eval_seed",
        "ci95_AoI_ms_per_seed": [None for _ in eval_seeds],
        "ci95_AoI_ms_per_seed_semantics": "not_computed_episodes_are_not_independent_units",
        "CAM_success_probability_per_seed_agent": per_seed_success_agent.tolist(),
        "CAM_success_probability_per_seed": per_seed_success.tolist(),
        "sd_CAM_success_probability_per_seed": sd_success.tolist(),
        "sd_CAM_success_probability_per_seed_semantics": "descriptive_across_scored_episodes_within_eval_seed",
        "ci95_CAM_success_probability_per_seed": [None for _ in eval_seeds],
        "ci95_CAM_success_probability_per_seed_semantics": "not_computed_episodes_are_not_independent_units",
        "mean_AoI_ms": float(per_seed_aoi.mean()),
        "CAM_success_probability": float(per_seed_success.mean()),
        "endpoint_success_probability_per_seed": per_seed_success.tolist(),
        "training_seed_is_inferential_unit": True,
        "is_frozen_eval": True,
        "status": "complete",
        "is_formal_result": formal_eval,
    }
    eval_provenance = {
        **_git_metadata(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": config.profile,
        "semantic_version": config.semantic_version,
        "config_hash": config.canonical_hash(),
        "scenario": config.scenario.id,
        "training_seed": int(config.seed),
        "eval_purpose": eval_purpose,
        "statistics_schema_version": EVAL_STATISTICS_SCHEMA_VERSION,
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_hash,
        "is_formal_result": formal_eval,
        "gap_definition": config.gap_definition,
        "vehicle_length_m": float(config.vehicle_length_m),
    }
    _write_json(eval_dir / "provenance.json", eval_provenance)
    _write_json(eval_dir / "summary.json", summary)
    _write_json(eval_dir / "EVAL_COMPLETE.json", summary)
    return {"eval_dir": str(eval_dir), **summary}
