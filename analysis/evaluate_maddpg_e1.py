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
from aoi_v2x_reproduction.config import config_from_dict, load_scenario
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
        "learner": _cpu_clone(learner.state_dict_full()) if learner is not None else None,
    }


def _parameters_unchanged(agents, learner, snapshot: Mapping[str, Any]) -> bool:
    return _nested_equal(_parameter_snapshot(agents, learner), snapshot)


def _training_policy_system(path, runtime_config):
    """Portable actors only; use the original Agent.choose_action unchanged."""
    from aoi_v2x_reproduction.algorithms.modified_maddpg.agent import Agent
    from aoi_v2x_reproduction.envs.platoon import PaperEnviron
    from aoi_v2x_reproduction.runtime.runner import resolve_device, seed_everything
    from aoi_v2x_reproduction.runtime.training_selection import load_training_actors

    class FrozenAgent:
        choose_action = Agent.choose_action

        def __init__(self, actor, config, device):
            self.actor, self.config, self.device = actor, config, device

        def state_dict_full(self):
            return {"actor": self.actor.state_dict()}

    device = resolve_device(runtime_config.device)
    seed_everything(runtime_config.seed, device)
    _, actors = load_training_actors(path, device=device)
    per_leader = [actors[0]] * runtime_config.number_agents if runtime_config.maddpg_actor_sharing else actors
    return PaperEnviron(runtime_config), [FrozenAgent(a, runtime_config, device) for a in per_leader], None, device


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


def build_frozen_maddpg_target_config(source_config, target_scenario: str, device: str):
    """Retarget an already validated frozen checkpoint, not a new training run.

    Historical paper_faithful payloads are intentionally read-only compatible.
    Do not relabel their profile or apply the active-training validator here.
    The caller validates latest500/source identity/hash before using this copy.
    """
    if source_config.algorithm not in ALGORITHMS:
        raise ValueError("frozen MADDPG target requires a MADDPG source")
    if target_scenario not in {"p05_n04_g05", "p05_n04_g15", "p05_n04_g25", "p05_n04_g35"}:
        raise ValueError("deployment supports P5 gaps 5/15/25/35 only")
    target = copy.deepcopy(source_config)
    target.scenario = load_scenario(target_scenario)
    target.device = str(device)
    target.is_formal_result = False
    mismatches = [field for field in (
        "state_dim", "action_dim", "number_agents", "n_rb", "n_modes",
        "power_min_dbm", "power_max_dbm", "power_continuous",
    ) if getattr(source_config, field) != getattr(target, field)]
    if mismatches:
        raise ValueError("source MADDPG and target spaces are incompatible: " + ", ".join(mismatches))
    return target


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
    *,
    action_repeat: int = 0,
    target_scenario: str | None = None,
    source_override: dict | None = None,
    deployment: bool = False,
    world_role: str | None = None,
    training_policy: bool = False,
) -> dict[str, Any]:
    from analysis.deployment_rng import action_seed_words, exogenous_probe
    if action_repeat < 0:
        raise ValueError("action_repeat must be non-negative")
    if training_policy and (not deployment or source_override is None):
        raise ValueError("training100 policies require explicit deployment/source override")
    allowed_roles = {None, "locked_new_worlds", "reused_development_worlds"}
    if training_policy:
        allowed_roles.add("user_fixed_exploratory_worlds")
    if world_role not in allowed_roles:
        raise ValueError("invalid deployment world role")
    if not deployment and (action_repeat != 0 or target_scenario is not None or source_override is not None or world_role is not None):
        raise ValueError("non-default execution requires deployment=True for unambiguous output identity")
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

    source = source_override or resolve_maddpg_source_run(diagnostics_root, algorithm, seed)
    output_root = Path(output_root).resolve()
    eval_name = maddpg_eval_id(noise) if tuple(eval_seeds) == EVAL_SEEDS and int(eval_episodes) == EVAL_EPISODES and int(warmup_episodes) == EVAL_WARMUP else (
        f"eval_e1_latest{int(expected_episode)}_noise{'0p0' if noise == 0.0 else '0p3'}_"
        f"sequential_warm_warm{int(warmup_episodes)}_s{'-'.join(str(v) for v in eval_seeds)}_ep{int(eval_episodes)}"
    )
    if deployment:
        eval_name += f"_deployment_{target_scenario or SCENARIO}_r{action_repeat}"
    if training_policy:
        eval_name = f"training100_{Path(source['checkpoint']).stem}_episode{expected_episode}_noise{noise}_r{action_repeat}"
    eval_dir = (output_root / "native" if training_policy else
                output_root / "evaluations" / source["canonical_run_name"] / eval_name)
    if not deployment and eval_dir.exists() and (eval_dir / "EVAL_COMPLETE.json").is_file():
        existing = json.loads((eval_dir / "EVAL_COMPLETE.json").read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("training_seed") == int(seed):
            existing["skipped_complete"] = True
            existing["eval_dir"] = str(eval_dir)
            return existing

    source_before = _source_fingerprint(source["run_dir"])
    caller_rng = capture_rng_state()
    try:
        if training_policy:
            from aoi_v2x_reproduction.runtime.training_selection import load_training_policy
            cfg, payload = load_training_policy(source["checkpoint"])
            if cfg.seed != seed or cfg.algorithm != algorithm or payload["episode"] != expected_episode:
                raise ValueError("training100 source seed/algorithm/selected episode mismatch")
        else:
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
        if not training_policy and payload.get("config_hash") != training_config.canonical_hash():
            raise ValueError("checkpoint config hash does not match reconstructed config")

        runtime_config = copy.deepcopy(training_config)
        if target_scenario is not None:
            runtime_config = build_frozen_maddpg_target_config(training_config, target_scenario, device)
        runtime_config.device = str(device)
        runtime_config.is_formal_result = False
        if training_policy:
            environment, agents, learner, resolved_device = _training_policy_system(source["checkpoint"], runtime_config)
        else:
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
        exogenous = []
        for world in eval_seeds:
            environment.reset_world(int(world))
            initial_probe = exogenous_probe(environment) if deployment else None
            action_noise_rng = np.random.default_rng(
                np.random.SeedSequence(action_seed_words(seed, world, action_repeat, "maddpg"))
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
                            **({"v2i_rate": np.asarray(info["v2i_rate"], dtype=np.float32)} if deployment else {}),
                        }
                    )
                world_records.append(episode_records)
            raw_records.append(world_records)
            if deployment:
                exogenous.append({"world": world, "initial": initial_probe,
                                  "final": exogenous_probe(environment)})
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
    if deployment:
        arrays["v2i_rate"] = _stack(raw_records, "v2i_rate", np.float32)
        arrays["reset_event"] = np.isclose(arrays["aoi_ms"], 1.0, rtol=0.0, atol=1e-6)
        arrays["v2i_attempt"] = arrays["mode"] == 0
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
        **({"checkpoint_role": payload["checkpoint_role"], "policy_version": payload["policy_version"],
            "training_budget_episodes": training_config.episodes, "critic_constructed": False,
            "replay_constructed": False} if training_policy else {}),
        "status": "complete",
        "algorithm": algorithm,
        "algorithm_label": source["algorithm_label"],
        "actor_sharing": bool(training_config.maddpg_actor_sharing),
        **({"model_id": "shared_maddpg_tdec", "actor_network_count": 1,
            "actor_structure": "shared", "sharing_schema": "shared_maddpg_tdec_v1",
            "optimizer_constructed": False, "runtime_config": runtime_config.to_dict()}
           if training_config.maddpg_actor_sharing else {}),
        "training_seed": int(seed),
        "training_run_name": source["canonical_run_name"],
        "historical_run_name": source["historical_run_name"],
        "source_run_dir": str(source["run_dir"]),
        "source_run_dir_resolved": str(source["resolved_run_dir"]),
        "source_is_symlink": bool(source["is_symlink"]),
        "checkpoint": str(source["checkpoint"]),
        "checkpoint_name": Path(source["checkpoint"]).name if training_policy else "latest.pt",
        "checkpoint_episode": int(payload.get("episode", -1)),
        "checkpoint_completed": True,
        "noise_std": noise,
        "action_repeat": int(action_repeat),
        "action_seed_words": [action_seed_words(seed, w, action_repeat, "maddpg") for w in eval_seeds],
        "environment_seeds": list(eval_seeds),
        "exogenous_probes": exogenous if deployment else None,
        "event_source": "post_step_aoi_ms_isclose_1_verified_env_step" if deployment else None,
        "v2i_min": float(environment.v2i_min) if deployment else None,
        "target_scenario": runtime_config.scenario.id,
        "source_scenario": training_config.scenario.id,
        "source_config_hash": training_config.canonical_hash(),
        "training_source_commit": source["complete"].get("reproduction_git_commit"),
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
        "world_role": world_role if deployment else "reused_development_worlds",
        "worlds_are_reused_development_worlds": (None if training_policy else
            ((world_role == "reused_development_worlds" if world_role else None) if deployment else True)),
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
        **({"checkpoint_role": payload["checkpoint_role"], "policy_version": payload["policy_version"]}
           if training_policy else {}),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "training_seed": int(seed),
        "canonical_run_name": source["canonical_run_name"],
        "historical_run_name": source["historical_run_name"],
        "source_run_dir": str(source["run_dir"]),
        "source_run_dir_resolved": str(source["resolved_run_dir"]),
        "checkpoint": str(source["checkpoint"]),
        "checkpoint_name": Path(source["checkpoint"]).name if training_policy else "latest.pt",
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
