"""Isolated frozen MADDPG evaluation for the E1 unified baseline."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from analysis.e1_contract import (
    ACTION_NOISE_TAG,
    ALGORITHMS,
    CHECKPOINT_EPISODE,
    EVAL_EPISODES,
    EVAL_SEEDS,
    EVAL_WARMUP,
    N_AGENTS,
    N_RB,
    NOISE_LEVELS,
    SCENARIO,
    STEPS_PER_EPISODE,
    cell_to_algorithm_noise_seed,
    maddpg_eval_id,
    resolve_maddpg_source_run,
    validate_latest500_payload,
)
from aoi_v2x_reproduction.config import config_from_dict
from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state, restore_rng_state
from aoi_v2x_reproduction.runtime.runner import _git_metadata, _make_system, _write_json


def _cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return copy.deepcopy(value)


def _nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_nested_equal(item, other) for item, other in zip(left, right))
    return left == right


def _parameter_snapshot(agents, learner) -> dict[str, Any]:
    return {
        "agents": [_cpu_clone(agent.state_dict_full()) for agent in agents],
        "learner": _cpu_clone(learner.state_dict_full()),
    }


def _parameters_unchanged(agents, learner, snapshot: Mapping[str, Any]) -> bool:
    return _nested_equal(_parameter_snapshot(agents, learner), snapshot)


def reduce_heldout_metrics(
    *,
    aoi: np.ndarray,
    success: np.ndarray,
    remaining: np.ndarray,
    power_dbm: np.ndarray,
    reward_global: np.ndarray,
    reward_task1: np.ndarray,
    reward_task2: np.ndarray,
    cam_bits: float,
    global_actor_weight: float,
) -> dict[str, float | int]:
    aoi = np.asarray(aoi, dtype=np.float64)
    success = np.asarray(success, dtype=np.float64)
    remaining = np.asarray(remaining, dtype=np.float64)
    power_dbm = np.asarray(power_dbm, dtype=np.float64)
    reward_global = np.asarray(reward_global, dtype=np.float64)
    reward_task1 = np.asarray(reward_task1, dtype=np.float64)
    reward_task2 = np.asarray(reward_task2, dtype=np.float64)
    if aoi.ndim != 4 or aoi.shape[-2:] != (STEPS_PER_EPISODE, N_AGENTS):
        raise ValueError(f"unexpected AoI shape: {aoi.shape}")
    if success.shape != aoi.shape or remaining.shape != aoi.shape or power_dbm.shape != aoi.shape:
        raise ValueError("AoI, success, remaining demand, and power shapes differ")
    if reward_task1.shape != aoi.shape or reward_task2.shape != aoi.shape:
        raise ValueError("task rewards must match the [world,episode,slot,agent] AoI shape")
    if reward_global.shape != aoi.shape[:-1]:
        raise ValueError("global reward must be [world,episode,slot]")
    endpoint_cam = success[..., -1, :]
    endpoint_payload = np.clip(1.0 - remaining[..., -1, :] / float(cam_bits), 0.0, 1.0)
    per_agent_aoi = aoi.mean(axis=(0, 1, 2))
    per_agent_cam = endpoint_cam.mean(axis=(0, 1))
    per_agent_payload = endpoint_payload.mean(axis=(0, 1))
    combined = reward_task1 + reward_task2 + float(global_actor_weight) * reward_global[..., None]
    power_mw = np.power(10.0, power_dbm / 10.0)
    gt50 = aoi > 50.0
    at_cap = np.isclose(aoi, 100.0, rtol=0.0, atol=1e-6)
    return {
        "mean_aoi_ms": float(per_agent_aoi.mean()),
        "worst_agent_aoi_ms": float(per_agent_aoi.max()),
        "aoi_gt50_count": int(gt50.sum()),
        "aoi_gt50_denominator": int(aoi.size),
        "aoi_gt50_fraction": float(gt50.mean()),
        "aoi_at_cap_count": int(at_cap.sum()),
        "aoi_at_cap_denominator": int(aoi.size),
        "aoi_at_cap_fraction": float(at_cap.mean()),
        "mean_binary_cam": float(per_agent_cam.mean()),
        "worst_agent_binary_cam": float(per_agent_cam.min()),
        "mean_payload_completion": float(per_agent_payload.mean()),
        "worst_agent_payload_completion": float(per_agent_payload.min()),
        "mean_reward_global": float(reward_global.mean()),
        "mean_reward_task1": float(reward_task1.mean()),
        "mean_reward_task2": float(reward_task2.mean()),
        "mean_reward_combined": float(combined.mean()),
        "mean_power_mw": float(power_mw.mean()),
        "power_mw_sum": float(power_mw.sum()),
        "power_mw_count": int(power_mw.size),
    }


def _stack(records: Sequence[Sequence[Sequence[Mapping[str, Any]]]], key: str, dtype) -> np.ndarray:
    return np.asarray(
        [[[step[key] for step in episode] for episode in world] for world in records],
        dtype=dtype,
    )


def _source_fingerprint(run_dir: Path) -> dict[str, Any]:
    fingerprint: dict[str, Any] = {"resolved_run_dir": str(run_dir.resolve())}
    for relative in (
        "COMPLETE.json",
        "config.resolved.json",
        "provenance.json",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
    ):
        path = run_dir / relative
        if path.exists():
            stat = path.stat()
            fingerprint[relative] = {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
        else:
            fingerprint[relative] = None
    eval_root = run_dir / "eval"
    fingerprint["eval_exists"] = eval_root.exists()
    fingerprint["eval_names"] = sorted(child.name for child in eval_root.iterdir()) if eval_root.exists() else []
    return fingerprint


def evaluate_maddpg_e1_cell(
    cell_id: int,
    diagnostics_root: Path,
    output_root: Path,
    device: str = "cpu",
    eval_seeds: Iterable[int] = EVAL_SEEDS,
    eval_episodes: int = EVAL_EPISODES,
    warmup_episodes: int = EVAL_WARMUP,
    expected_episode: int = CHECKPOINT_EPISODE,
) -> dict[str, Any]:
    algorithm, noise, seed = cell_to_algorithm_noise_seed(cell_id)
    return evaluate_maddpg_e1(
        algorithm,
        noise,
        seed,
        diagnostics_root,
        output_root,
        device=device,
        eval_seeds=eval_seeds,
        eval_episodes=eval_episodes,
        warmup_episodes=warmup_episodes,
        expected_episode=expected_episode,
    )


def evaluate_maddpg_e1(
    algorithm: str,
    noise: float,
    seed: int,
    diagnostics_root: Path,
    output_root: Path,
    device: str = "cpu",
    eval_seeds: Iterable[int] = EVAL_SEEDS,
    eval_episodes: int = EVAL_EPISODES,
    warmup_episodes: int = EVAL_WARMUP,
    expected_episode: int = CHECKPOINT_EPISODE,
) -> dict[str, Any]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unsupported MADDPG algorithm: {algorithm}")
    noise = float(noise)
    if noise not in NOISE_LEVELS:
        raise ValueError(f"unsupported E1 noise: {noise}")
    eval_seeds = tuple(int(value) for value in eval_seeds)
    if not eval_seeds or len(eval_seeds) != len(set(eval_seeds)):
        raise ValueError("eval_seeds must be unique and non-empty")
    if any(value in {101, 102, 103, 104, 105, 106, 151, 152, 153, 154, 155, 156} for value in eval_seeds):
        raise ValueError("refusing reserved final-test or locked worlds")
    if int(eval_episodes) < 1 or int(warmup_episodes) < 0:
        raise ValueError("eval_episodes must be positive")

    source = resolve_maddpg_source_run(diagnostics_root, algorithm, seed)
    output_root = Path(output_root).resolve()
    eval_name = maddpg_eval_id(noise) if tuple(eval_seeds) == EVAL_SEEDS and int(eval_episodes) == EVAL_EPISODES and int(warmup_episodes) == EVAL_WARMUP else (
        f"eval_e1_latest{int(expected_episode)}_noise{'0p0' if noise == 0.0 else '0p3'}_"
        f"sequential_warm_warm{int(warmup_episodes)}_s{'-'.join(str(v) for v in eval_seeds)}_ep{int(eval_episodes)}"
    )
    eval_dir = output_root / "evaluations" / source["canonical_run_name"] / eval_name
    if eval_dir.exists() and (eval_dir / "EVAL_COMPLETE.json").is_file():
        existing = json.loads((eval_dir / "EVAL_COMPLETE.json").read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("training_seed") == int(seed):
            existing["skipped_complete"] = True
            existing["eval_dir"] = str(eval_dir)
            return existing

    source_before = _source_fingerprint(source["run_dir"])
    caller_rng = capture_rng_state()
    try:
        payload = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
        validate_latest500_payload(payload, algorithm, seed, expected_episode=expected_episode)
        raw_config = payload["config"]
        training_config = config_from_dict(raw_config)
        if training_config.algorithm != algorithm:
            raise ValueError(
                f"resolved algorithm {training_config.algorithm!r} does not match source {algorithm!r}"
            )
        if training_config.algorithm == "mappo":
            raise ValueError("old TDec compatibility must resolve to modified_maddpg_tdec, not MAPPO")
        if int(training_config.n_rb) != N_RB or training_config.scenario.id != SCENARIO:
            raise ValueError("checkpoint config is not P5/N4/gap25 with n_rb=3")
        if int(training_config.number_agents) != N_AGENTS or int(training_config.steps_per_episode) != STEPS_PER_EPISODE:
            raise ValueError("checkpoint config agent/step contract mismatch")
        if payload.get("config_hash") != training_config.canonical_hash():
            raise ValueError("checkpoint config hash does not match reconstructed config")

        runtime_config = copy.deepcopy(training_config)
        runtime_config.device = str(device)
        runtime_config.is_formal_result = False
        environment, agents, learner, _replay, _metrics, resolved_device = _make_system(runtime_config)
        for agent, state in zip(agents, payload["agents"]):
            agent.load_state_dict_full(state)
        learner.load_state_dict_full(payload["learner"])
        for agent in agents:
            agent.actor.eval()
            for parameter in agent.actor.parameters():
                parameter.requires_grad_(False)
        snapshot = _parameter_snapshot(agents, learner)

        eval_dir.parent.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=False, exist_ok=False)
        raw_records = []
        for world in eval_seeds:
            environment.reset_world(int(world))
            action_noise_rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(world), ACTION_NOISE_TAG])
            )
            for warmup_index in range(int(warmup_episodes)):
                observations = environment.start_episode(warmup_index)
                for _slot in range(int(runtime_config.steps_per_episode)):
                    actions = np.asarray(
                        [
                            agent.choose_action(
                                observations[index],
                                explore=False,
                                noise_std=noise,
                                rng=action_noise_rng,
                            )
                            for index, agent in enumerate(agents)
                        ],
                        dtype=np.float32,
                    )
                    observations, _rg, _t1, _t2, _done, _info = environment.step(actions)
            world_records = []
            for episode in range(int(eval_episodes)):
                observations = environment.start_episode(int(warmup_episodes) + episode)
                episode_records = []
                for _slot in range(int(runtime_config.steps_per_episode)):
                    actions = np.asarray(
                        [
                            agent.choose_action(
                                observations[index],
                                explore=False,
                                noise_std=noise,
                                rng=action_noise_rng,
                            )
                            for index, agent in enumerate(agents)
                        ],
                        dtype=np.float32,
                    )
                    observations, reward_global, reward_task1, reward_task2, _done, info = environment.step(actions)
                    episode_records.append(
                        {
                            "aoi_ms": np.asarray(info["aoi_ms"], dtype=np.float32),
                            "success": np.asarray(info["success"], dtype=np.float32),
                            "remaining_demand": np.asarray(info["remaining_demand"], dtype=np.float32),
                            "power_dbm": np.asarray(info["power_dbm"], dtype=np.float32),
                            "rb": np.asarray(info["rb"], dtype=np.int64),
                            "mode": np.asarray(info["mode"], dtype=np.int64),
                            "action_normalized": np.asarray(actions, dtype=np.float32),
                            "reward_global": np.float32(reward_global),
                            "reward_task1": np.asarray(reward_task1, dtype=np.float32),
                            "reward_task2": np.asarray(reward_task2, dtype=np.float32),
                        }
                    )
                world_records.append(episode_records)
            raw_records.append(world_records)
        if not _parameters_unchanged(agents, learner, snapshot):
            raise RuntimeError("MADDPG parameters or optimizer state changed during frozen evaluation")
    finally:
        restore_rng_state(caller_rng)

    arrays = {
        "aoi_ms": _stack(raw_records, "aoi_ms", np.float32),
        "success": _stack(raw_records, "success", np.float32),
        "remaining_demand": _stack(raw_records, "remaining_demand", np.float32),
        "power_dbm": _stack(raw_records, "power_dbm", np.float32),
        "rb": _stack(raw_records, "rb", np.int64),
        "mode": _stack(raw_records, "mode", np.int64),
        "action_normalized": _stack(raw_records, "action_normalized", np.float32),
        "reward_global": np.asarray(
            [[[step["reward_global"] for step in episode] for episode in world] for world in raw_records],
            dtype=np.float32,
        ),
        "reward_task1": _stack(raw_records, "reward_task1", np.float32),
        "reward_task2": _stack(raw_records, "reward_task2", np.float32),
    }
    metrics = reduce_heldout_metrics(
        aoi=arrays["aoi_ms"],
        success=arrays["success"],
        remaining=arrays["remaining_demand"],
        power_dbm=arrays["power_dbm"],
        reward_global=arrays["reward_global"],
        reward_task1=arrays["reward_task1"],
        reward_task2=arrays["reward_task2"],
        cam_bits=float(runtime_config.cam_bits),
        global_actor_weight=float(runtime_config.global_actor_weight),
    )
    git = _git_metadata()
    summary = {
        "status": "complete",
        "algorithm": algorithm,
        "algorithm_label": source["algorithm_label"],
        "actor_sharing": False,
        "training_seed": int(seed),
        "training_run_name": source["canonical_run_name"],
        "historical_run_name": source["historical_run_name"],
        "source_run_dir": str(source["run_dir"]),
        "source_run_dir_resolved": str(source["resolved_run_dir"]),
        "source_is_symlink": bool(source["is_symlink"]),
        "checkpoint": str(source["checkpoint"]),
        "checkpoint_name": "latest.pt",
        "checkpoint_episode": int(payload.get("episode", -1)),
        "checkpoint_completed": True,
        "noise_std": noise,
        "explore": False,
        "eval_protocol": "sequential_warm",
        "eval_seeds": list(eval_seeds),
        "eval_episodes": int(eval_episodes),
        "eval_warmup_episodes": int(warmup_episodes),
        "steps_per_episode": int(runtime_config.steps_per_episode),
        "n_rb": int(runtime_config.n_rb),
        "number_agents": int(runtime_config.number_agents),
        "scenario": runtime_config.scenario.id,
        "cam_bits": float(runtime_config.cam_bits),
        "global_actor_weight": float(runtime_config.global_actor_weight),
        "parameter_snapshot_unchanged": True,
        "optimizer_steps": 0,
        "normalization_updated": False,
        "wrote_to_source_run": False,
        "data_role": "new",
        "worlds_are_reused_development_worlds": True,
        "reproduction_git_commit": git.get("reproduction_git_commit"),
        "reproduction_git_branch": git.get("reproduction_git_branch"),
        "reproduction_git_dirty": git.get("reproduction_git_dirty"),
        "evaluation_device_requested": str(device),
        "evaluation_device_resolved": str(resolved_device),
        "raw_metric_axes": {
            "aoi_ms": ["eval_seed", "scored_episode", "slot", "agent"],
            "success": ["eval_seed", "scored_episode", "slot", "agent"],
            "remaining_demand": ["eval_seed", "scored_episode", "slot", "agent"],
            "reward_global": ["eval_seed", "scored_episode", "slot"],
            "reward_task1": ["eval_seed", "scored_episode", "slot", "agent"],
        },
        **metrics,
    }
    provenance = {
        **git,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "training_seed": int(seed),
        "canonical_run_name": source["canonical_run_name"],
        "historical_run_name": source["historical_run_name"],
        "source_run_dir": str(source["run_dir"]),
        "source_run_dir_resolved": str(source["resolved_run_dir"]),
        "checkpoint": str(source["checkpoint"]),
        "checkpoint_name": "latest.pt",
        "checkpoint_episode": int(payload.get("episode", -1)),
        "noise_std": noise,
        "eval_id": eval_name,
        "eval_seeds": list(eval_seeds),
        "output_dir": str(eval_dir),
        "source_fingerprint_before": source_before,
        "manifest": source["manifest"],
    }
    np.savez_compressed(eval_dir / "metrics.npz", **arrays)
    _write_json(eval_dir / "summary.json", summary)
    _write_json(eval_dir / "EVAL_COMPLETE.json", summary)
    _write_json(eval_dir / "provenance.json", provenance)
    source_after = _source_fingerprint(source["run_dir"])
    if source_after != source_before:
        raise RuntimeError("E1 evaluation wrote back to a historical MADDPG source run")
    summary["eval_dir"] = str(eval_dir)
    summary["skipped_complete"] = False
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True, type=int)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = evaluate_maddpg_e1_cell(
        args.cell_id,
        args.diagnostics_root,
        args.output_root,
        device=args.device,
    )
    print(json.dumps({key: result[key] for key in (
        "status", "algorithm", "training_seed", "noise_std", "eval_dir",
        "checkpoint_episode", "parameter_snapshot_unchanged", "wrote_to_source_run",
        "reproduction_git_commit", "reproduction_git_branch", "reproduction_git_dirty",
        "mean_aoi_ms", "mean_binary_cam", "mean_payload_completion", "mean_reward_combined",
    ) if key in result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
