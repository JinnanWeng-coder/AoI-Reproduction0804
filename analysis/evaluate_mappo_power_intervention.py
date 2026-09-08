"""Frozen-policy stochastic MAPPO evaluation with mode-specific power interventions."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state, restore_rng_state
from aoi_v2x_reproduction.runtime.runner import (
    _git_metadata,
    _make_mappo_system,
    _validate_mappo_policy_artifact,
    _write_json,
)


ARMS = ("baseline", "v2i_plus3db", "v2v_minus3db")
DEFAULT_EVAL_SEEDS = (201, 202, 203, 204, 205, 206)
POLICY_SEED_TAG = 0x4D415050


@dataclass(frozen=True)
class IntervenedAction:
    policy_actions: np.ndarray
    environment_actions: np.ndarray
    rb: np.ndarray
    mode: np.ndarray
    policy_power_dbm: np.ndarray
    executed_power_dbm: np.ndarray
    power_clipped_low: np.ndarray
    power_clipped_high: np.ndarray
    power_at_boundary: np.ndarray


def intervention_eval_id(
    arm: str,
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    eval_episodes: int = 100,
    warmup_episodes: int = 5,
) -> str:
    if arm not in ARMS:
        raise ValueError(f"unsupported intervention arm: {arm}")
    seed_token = "-".join(str(int(seed)) for seed in eval_seeds)
    return (
        f"eval_validation_policy_final_stochastic_sequential_warm_"
        f"warm{int(warmup_episodes)}_s{seed_token}_ep{int(eval_episodes)}_arm_{arm}"
    )


def apply_power_intervention(
    policy_actions: np.ndarray,
    arm: str,
    power_min_dbm: float,
    power_max_dbm: float,
    n_rb: int,
    n_modes: int,
) -> IntervenedAction:
    """Apply a +/-3 dB intervention after decoding mode, preserving RB and mode."""

    if arm not in ARMS:
        raise ValueError(f"unsupported intervention arm: {arm}")
    actions = np.asarray(policy_actions)
    if actions.ndim != 2 or actions.shape[1] != 3 or not np.all(np.isfinite(actions)):
        raise ValueError("policy_actions must be a finite [agent, 3] array")
    if power_max_dbm <= power_min_dbm or int(n_rb) < 1 or int(n_modes) < 2:
        raise ValueError("invalid action-space bounds")

    normalized = np.clip(actions.astype(np.float64, copy=False), -1.0, 1.0)
    rb = np.minimum(int(n_rb) - 1, np.floor((normalized[:, 0] + 1.0) * 0.5 * int(n_rb))).astype(np.int64)
    mode = np.minimum(int(n_modes) - 1, np.floor((normalized[:, 1] + 1.0) * 0.5 * int(n_modes))).astype(np.int64)
    policy_power = float(power_min_dbm) + (normalized[:, 2] + 1.0) * 0.5 * (
        float(power_max_dbm) - float(power_min_dbm)
    )

    requested_power = policy_power.copy()
    affected = np.zeros(mode.shape, dtype=bool)
    if arm == "v2i_plus3db":
        affected = mode == 0
        requested_power[affected] += 3.0
    elif arm == "v2v_minus3db":
        affected = mode == 1
        requested_power[affected] -= 3.0
    clipped_low = affected & (requested_power < float(power_min_dbm))
    clipped_high = affected & (requested_power > float(power_max_dbm))
    target_power = np.clip(requested_power, float(power_min_dbm), float(power_max_dbm))

    if arm == "baseline":
        executed_actions = actions
    else:
        executed_actions = actions.copy()
        mapped = 2.0 * (target_power - float(power_min_dbm)) / (
            float(power_max_dbm) - float(power_min_dbm)
        ) - 1.0
        executed_actions[:, 2] = mapped.astype(executed_actions.dtype, copy=False)

    executed_normalized = np.clip(np.asarray(executed_actions, dtype=np.float64), -1.0, 1.0)
    executed_power = float(power_min_dbm) + (executed_normalized[:, 2] + 1.0) * 0.5 * (
        float(power_max_dbm) - float(power_min_dbm)
    )
    tolerance = 1e-5
    at_boundary = (
        np.isclose(executed_power, float(power_min_dbm), rtol=0.0, atol=tolerance)
        | np.isclose(executed_power, float(power_max_dbm), rtol=0.0, atol=tolerance)
    )
    executed_discrete = np.clip(np.asarray(executed_actions, dtype=np.float64), -1.0, 1.0)
    executed_rb = np.minimum(
        int(n_rb) - 1, np.floor((executed_discrete[:, 0] + 1.0) * 0.5 * int(n_rb))
    ).astype(np.int64)
    executed_mode = np.minimum(
        int(n_modes) - 1, np.floor((executed_discrete[:, 1] + 1.0) * 0.5 * int(n_modes))
    ).astype(np.int64)
    if not np.array_equal(executed_rb, rb) or not np.array_equal(executed_mode, mode):
        raise AssertionError("power intervention changed RB or mode")
    return IntervenedAction(
        policy_actions=actions,
        environment_actions=executed_actions,
        rb=rb,
        mode=mode,
        policy_power_dbm=policy_power.astype(np.float32),
        executed_power_dbm=executed_power.astype(np.float32),
        power_clipped_low=clipped_low,
        power_clipped_high=clipped_high,
        power_at_boundary=at_boundary,
    )


def _actor_snapshot(trainer) -> list[list[torch.Tensor]]:
    return [
        [parameter.detach().cpu().clone() for parameter in actor.parameters()]
        for actor in trainer.actors
    ]


def _actors_unchanged(trainer, snapshot: Sequence[Sequence[torch.Tensor]]) -> bool:
    return all(
        torch.equal(parameter.detach().cpu(), expected)
        for actor, expected_actor in zip(trainer.actors, snapshot)
        for parameter, expected in zip(actor.parameters(), expected_actor)
    )


def _record_step(info: dict, intervened: IntervenedAction) -> dict:
    """Copy environment and intervention observations without sampling RNG."""

    return {
        "aoi_ms": np.asarray(info["aoi_ms"], dtype=np.float32).copy(),
        "success": np.asarray(info["success"], dtype=np.float32).copy(),
        "remaining_demand": np.asarray(info["remaining_demand"], dtype=np.float32).copy(),
        "rb": np.asarray(info["rb"], dtype=np.int64).copy(),
        "mode": np.asarray(info["mode"], dtype=np.int64).copy(),
        "power_dbm": np.asarray(info["power_dbm"], dtype=np.float32).copy(),
        "v2i_rate": np.asarray(info["v2i_rate"], dtype=np.float32).copy(),
        "selected_interference_db": np.asarray(info["selected_interference_db"], dtype=np.float32).copy(),
        "policy_action_normalized": np.asarray(intervened.policy_actions, dtype=np.float32).copy(),
        "action_normalized": np.asarray(intervened.environment_actions, dtype=np.float32).copy(),
        "policy_power_dbm": intervened.policy_power_dbm.copy(),
        "executed_power_dbm": intervened.executed_power_dbm.copy(),
        "power_clipped_low": intervened.power_clipped_low.copy(),
        "power_clipped_high": intervened.power_clipped_high.copy(),
        "power_at_boundary": intervened.power_at_boundary.copy(),
        "reset_event": np.isclose(np.asarray(info["aoi_ms"]), 1.0, rtol=0.0, atol=1e-6),
    }


def _stack_trajectory(raw_by_seed: Sequence[Sequence[Sequence[dict]]]) -> dict[str, np.ndarray]:
    keys = tuple(raw_by_seed[0][0][0])
    arrays = {
        key: np.asarray([
            [[record[key] for record in episode] for episode in seed]
            for seed in raw_by_seed
        ])
        for key in keys
    }
    return arrays


def _summary(arrays: dict[str, np.ndarray], cam_bits: float) -> dict:
    aoi = arrays["aoi_ms"].astype(np.float64)
    endpoint_success = arrays["success"][:, :, -1, :].astype(np.float64)
    payload = np.clip(
        1.0 - arrays["remaining_demand"][:, :, -1, :].astype(np.float64) / float(cam_bits),
        0.0,
        1.0,
    )
    per_agent_aoi = aoi.mean(axis=(0, 1, 2))
    per_agent_cam = endpoint_success.mean(axis=(0, 1))
    per_agent_payload = payload.mean(axis=(0, 1))
    mode = arrays["mode"]
    attempts = mode == 0
    resets = arrays["reset_event"].astype(bool)
    attempt_count = int(attempts.sum())
    reset_count = int(resets.sum())
    power_mw = np.power(10.0, arrays["executed_power_dbm"].astype(np.float64) / 10.0)
    return {
        "mean_AoI_ms": float(per_agent_aoi.mean()),
        "worst_agent_mean_AoI_ms": float(per_agent_aoi.max()),
        "CAM_success_probability": float(per_agent_cam.mean()),
        "worst_agent_CAM_success_probability": float(per_agent_cam.min()),
        "payload_completion": float(per_agent_payload.mean()),
        "worst_agent_payload_completion": float(per_agent_payload.min()),
        "mean_executed_power_mw": float(power_mw.mean()),
        "v2i_attempt_count": attempt_count,
        "aoi_reset_count": reset_count,
        "v2i_attempt_rate": float(attempt_count / mode.size),
        "v2i_success_given_attempt": float(reset_count / attempt_count) if attempt_count else None,
        "aoi_reset_rate": float(reset_count / mode.size),
        "aoi_gt50_fraction": float((aoi > 50.0).mean()),
        "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, rtol=0.0, atol=1e-6).mean()),
        "power_at_boundary_fraction": float(arrays["power_at_boundary"].mean()),
        "power_clipped_low_fraction": float(arrays["power_clipped_low"].mean()),
        "power_clipped_high_fraction": float(arrays["power_clipped_high"].mean()),
    }


def evaluate_power_intervention(
    policy: Path,
    output_root: Path,
    arm: str,
    device: str = "auto",
    eval_seeds: Iterable[int] = DEFAULT_EVAL_SEEDS,
    eval_episodes: int = 100,
) -> dict:
    """Evaluate one frozen TDec policy under one stochastic intervention arm."""

    if arm not in ARMS:
        raise ValueError(f"unsupported intervention arm: {arm}")
    eval_seeds = tuple(int(seed) for seed in eval_seeds)
    if not eval_seeds or len(eval_seeds) != len(set(eval_seeds)) or int(eval_episodes) < 1:
        raise ValueError("eval_seeds must be unique and eval_episodes must be positive")
    policy_path = Path(policy).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    caller_rng = capture_rng_state()
    try:
        payload = torch.load(policy_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError("MAPPO policy payload must be a dictionary")
        run_dir, training_config, training_complete = _validate_mappo_policy_artifact(policy_path, payload)
        if training_config.algorithm != "mappo" or training_config.mappo_variant != "tdec":
            raise RuntimeError("mode-power intervention requires a TDec MAPPO policy")
        if not training_config.power_continuous:
            raise RuntimeError("mode-power intervention requires continuous dBm power mapping")
        runtime_config = copy.deepcopy(training_config)
        runtime_config.device = device
        warmup_episodes = int(runtime_config.eval_warmup_episodes)
        if runtime_config.eval_protocol != "sequential_warm" or warmup_episodes != 5:
            raise RuntimeError("unexpected source evaluation protocol")
        eval_name = intervention_eval_id(arm, eval_seeds, int(eval_episodes), warmup_episodes)
        eval_dir = output_root / run_dir.name / eval_name
        if eval_dir.exists():
            raise FileExistsError(f"refusing to overwrite intervention evaluation: {eval_dir}")

        environment, trainer, _rollout, _metrics, resolved_device = _make_mappo_system(runtime_config)
        for actor, state_dict in zip(trainer.actors, payload["actors"]):
            actor.load_state_dict(state_dict, strict=True)
        trainer.actors.eval()
        actor_snapshot = _actor_snapshot(trainer)
        eval_dir.mkdir(parents=True, exist_ok=False)
        raw_by_seed = []
        for heldout_seed in eval_seeds:
            policy_seed = int(
                np.random.SeedSequence([int(training_config.seed), int(heldout_seed), POLICY_SEED_TAG])
                .generate_state(1, dtype=np.uint32)[0]
            )
            torch.manual_seed(policy_seed)
            if resolved_device.type == "cuda":
                torch.cuda.manual_seed_all(policy_seed)
            environment.reset_world(heldout_seed)
            for warmup_index in range(warmup_episodes):
                observations = environment.start_episode(warmup_index)
                for _slot in range(runtime_config.steps_per_episode):
                    sampled = trainer.act(observations, deterministic=False)
                    intervention = apply_power_intervention(
                        sampled.environment_actions,
                        arm,
                        runtime_config.power_min_dbm,
                        runtime_config.power_max_dbm,
                        runtime_config.n_rb,
                        runtime_config.n_modes,
                    )
                    observations, _rg, _t1, _t2, _done, _info = environment.step(
                        intervention.environment_actions
                    )
            seed_records = []
            for episode in range(int(eval_episodes)):
                observations = environment.start_episode(warmup_episodes + episode)
                episode_records = []
                for _slot in range(runtime_config.steps_per_episode):
                    sampled = trainer.act(observations, deterministic=False)
                    intervention = apply_power_intervention(
                        sampled.environment_actions,
                        arm,
                        runtime_config.power_min_dbm,
                        runtime_config.power_max_dbm,
                        runtime_config.n_rb,
                        runtime_config.n_modes,
                    )
                    observations, _rg, _t1, _t2, _done, info = environment.step(
                        intervention.environment_actions
                    )
                    episode_records.append(_record_step(info, intervention))
                seed_records.append(episode_records)
            raw_by_seed.append(seed_records)
        if not _actors_unchanged(trainer, actor_snapshot):
            raise RuntimeError("frozen-policy evaluation modified actor parameters")
    finally:
        restore_rng_state(caller_rng)

    arrays = _stack_trajectory(raw_by_seed)
    expected_prefix = (len(eval_seeds), int(eval_episodes), runtime_config.steps_per_episode, runtime_config.number_agents)
    if arrays["aoi_ms"].shape != expected_prefix:
        raise RuntimeError(f"unexpected intervention metric shape: {arrays['aoi_ms'].shape}")
    np.savez_compressed(eval_dir / "metrics.npz", **arrays)
    git_metadata = _git_metadata()
    policy_reference = os.path.relpath(policy_path, eval_dir).replace(os.sep, "/")
    summary = {
        **git_metadata,
        **_summary(arrays, runtime_config.cam_bits),
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": "tdec",
        "intervention_arm": arm,
        "intervention_db": 0.0 if arm == "baseline" else (3.0 if arm == "v2i_plus3db" else -3.0),
        "intervention_semantics": "mode-conditioned shift in decoded dBm, clipped to source power bounds",
        "mappo_eval_mode": "stochastic",
        "external_action_noise_applicable": False,
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": warmup_episodes,
        "eval_seeds": list(eval_seeds),
        "eval_episodes": int(eval_episodes),
        "training_seed": int(training_config.seed),
        "training_run_name": run_dir.name,
        "scenario": training_config.scenario.id,
        "policy_name": policy_path.name,
        "policy": policy_reference,
        "policy_path_is_relative_to_eval": True,
        "policy_schema_version": payload.get("policy_schema_version"),
        "policy_episode": int(payload.get("episode", -1)),
        "source_config_hash": training_config.canonical_hash(),
        "policy_parameters_unchanged": True,
        "is_frozen_eval": True,
        "diagnostic_evaluation": True,
        "is_formal_result": False,
        "aoi_cap_ms": float(runtime_config.steps_per_episode),
        "cam_bits": float(runtime_config.cam_bits),
        "power_min_dbm": float(runtime_config.power_min_dbm),
        "power_max_dbm": float(runtime_config.power_max_dbm),
        "raw_metric_axes": {
            key: ["eval_seed", "scored_episode", "slot", "agent"]
            + (["action_dim"] if arrays[key].ndim == 5 else [])
            for key in arrays
        },
        "eval_id": eval_name,
    }
    provenance = {
        **git_metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "mappo",
        "mappo_variant": "tdec",
        "intervention_arm": arm,
        "training_seed": int(training_config.seed),
        "training_source_commit": training_complete.get("reproduction_git_commit"),
        "source_config_hash": training_config.canonical_hash(),
        "policy": policy_reference,
        "policy_schema_version": payload.get("policy_schema_version"),
        "eval_seeds": list(eval_seeds),
        "eval_episodes": int(eval_episodes),
        "eval_warmup_episodes": warmup_episodes,
        "mappo_eval_mode": "stochastic",
        "release_status": "diagnostic_evaluation",
    }
    _write_json(eval_dir / "provenance.json", provenance)
    _write_json(eval_dir / "summary.json", summary)
    _write_json(eval_dir / "EVAL_COMPLETE.json", summary)
    return {"eval_dir": str(eval_dir), **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-seeds", default=",".join(str(seed) for seed in DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()
    result = evaluate_power_intervention(
        policy=args.policy,
        output_root=args.output_root,
        arm=args.arm,
        device=args.device,
        eval_seeds=[int(token) for token in args.eval_seeds.split(",") if token.strip()],
        eval_episodes=args.eval_episodes,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
