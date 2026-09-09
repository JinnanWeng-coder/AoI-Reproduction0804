"""Frozen TDec MAPPO feasibility and reward-ledger evaluation.

The diagnostic is deliberately read-only with respect to the policy and the
environment implementation.  Feasibility is reconstructed from the current
slot immediately before ``env.step`` because the environment renews fast
fading at the end of that call.
"""

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

from analysis.evaluate_mappo_power_intervention import (
    DEFAULT_EVAL_SEEDS,
    POLICY_SEED_TAG,
    IntervenedAction,
    _actor_snapshot,
    _actors_unchanged,
    _stack_trajectory,
)
from aoi_v2x_reproduction.envs.platoon import power_penalty
from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state, restore_rng_state
from aoi_v2x_reproduction.runtime.runner import (
    _git_metadata,
    _make_mappo_system,
    _validate_mappo_policy_artifact,
    _write_json,
)


ARMS = (
    "baseline",
    "v2i_plus3db",
    "v2i_minus3db",
    "v2i_plus6db",
    "v2i_plus9db",
)
ARM_DELTAS_DB = {
    "baseline": 0.0,
    "v2i_plus3db": 3.0,
    "v2i_minus3db": -3.0,
    "v2i_plus6db": 6.0,
    "v2i_plus9db": 9.0,
}
REFERENCE_TRAJECTORY_ARRAYS = (
    "aoi_ms",
    "success",
    "remaining_demand",
    "power_dbm",
    "rb",
    "mode",
    "action_normalized",
    "policy_action_normalized",
    "policy_power_dbm",
    "executed_power_dbm",
    "v2i_rate",
    "selected_interference_db",
    "reset_event",
)


def task_to_arm_seed(task_id: int) -> tuple[str, int]:
    """Return the fixed 30-cell mapping requested by the audit protocol."""

    task_id = int(task_id)
    if task_id < 0 or task_id >= 30:
        raise ValueError("task_id must satisfy 0 <= task_id < 30")
    return ARMS[task_id // 6], (8, 9, 10, 11, 12, 13)[task_id % 6]


def feasibility_eval_id(
    arm: str,
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    eval_episodes: int = 100,
    warmup_episodes: int = 5,
) -> str:
    if arm not in ARMS:
        raise ValueError(f"unsupported feasibility arm: {arm}")
    seed_token = "-".join(str(int(seed)) for seed in eval_seeds)
    return (
        "eval_validation_policy_final_stochastic_sequential_warm_"
        f"warm{int(warmup_episodes)}_s{seed_token}_ep{int(eval_episodes)}_"
        f"feasibility_reward_arm_{arm}"
    )


def apply_v2i_power_offset(
    policy_actions: np.ndarray,
    arm: str,
    power_min_dbm: float,
    power_max_dbm: float,
    n_rb: int,
    n_modes: int,
) -> IntervenedAction:
    """Shift decoded dBm only for mode 0, preserving policy RB and mode."""

    if arm not in ARMS:
        raise ValueError(f"unsupported feasibility arm: {arm}")
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
    delta = ARM_DELTAS_DB[arm]
    affected = mode == 0
    requested_power = policy_power.copy()
    requested_power[affected] += delta
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
    executed_rb = np.minimum(
        int(n_rb) - 1,
        np.floor((executed_normalized[:, 0] + 1.0) * 0.5 * int(n_rb)),
    ).astype(np.int64)
    executed_mode = np.minimum(
        int(n_modes) - 1,
        np.floor((executed_normalized[:, 1] + 1.0) * 0.5 * int(n_modes)),
    ).astype(np.int64)
    if not np.array_equal(executed_rb, rb) or not np.array_equal(executed_mode, mode):
        raise AssertionError("power intervention changed RB or mode")
    tolerance = 1e-5
    at_boundary = (
        np.isclose(executed_power, float(power_min_dbm), rtol=0.0, atol=tolerance)
        | np.isclose(executed_power, float(power_max_dbm), rtol=0.0, atol=tolerance)
    )
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


@dataclass(frozen=True)
class FeasibilitySnapshot:
    leader_bs_distance_m: np.ndarray
    v2i_channel_loss_db: np.ndarray
    v2i_interference_plus_noise_linear: np.ndarray
    required_power_selected_dbm: np.ndarray
    required_power_best_current_interference_dbm: np.ndarray
    required_power_noise_only_best_dbm: np.ndarray
    best_current_interference_rate_at_power_max: np.ndarray
    noise_only_best_rate_at_power_max: np.ndarray
    selected_feasible: np.ndarray
    best_current_interference_feasible: np.ndarray
    noise_only_best_feasible: np.ndarray
    reconstructed_v2i_rate: np.ndarray
    gamma_required: float


def required_power_dbm(
    channel_loss_db: np.ndarray,
    interference_plus_noise_linear: np.ndarray,
    gamma_required: float,
    veh_ant_gain_db: float,
    bs_ant_gain_db: float,
    bs_noise_figure_db: float,
) -> np.ndarray:
    """Required transmit dBm under the environment's exact V2I SINR units."""

    loss = np.asarray(channel_loss_db, dtype=np.float64)
    interference = np.asarray(interference_plus_noise_linear, dtype=np.float64)
    if loss.shape != interference.shape:
        raise ValueError("channel loss and interference must have identical shapes")
    if gamma_required <= 0 or np.any(interference <= 0) or not np.all(np.isfinite(interference)):
        raise ValueError("gamma and interference must be finite and positive")
    return (
        10.0 * np.log10(float(gamma_required) * interference)
        + loss
        - float(veh_ant_gain_db)
        - float(bs_ant_gain_db)
        + float(bs_noise_figure_db)
    )


def compute_feasibility_snapshot(environment, environment_actions: np.ndarray) -> FeasibilitySnapshot:
    """Pure current-slot V2I feasibility reconstruction; no state or RNG writes."""

    decoded = environment.decode_actions(environment_actions)
    p, n_rb, n = environment.n_platoon, environment.n_rb, environment.size_platoon
    rb_indices = decoded[:, 0].astype(np.int64)
    modes = decoded[:, 1].astype(np.int64)
    leaders = np.arange(p, dtype=np.int64) * n
    channel_loss = np.asarray(environment.v2i_channels_fast[leaders, :], dtype=np.float64).copy()
    interference = np.full((p, n_rb), float(environment.sig2), dtype=np.float64)
    for receiver in range(p):
        for rb in range(n_rb):
            for transmitter in np.flatnonzero(rb_indices == rb):
                if transmitter == receiver:
                    continue
                leader_tx = int(leaders[transmitter])
                interference[receiver, rb] += 10.0 ** (
                    (
                        decoded[transmitter, 2]
                        - environment.v2i_channels_fast[leader_tx, rb]
                        + environment.veh_ant_gain
                        + environment.bs_ant_gain
                        - environment.bs_noise_figure
                    )
                    / 10.0
                )

    spectral_threshold = float(environment.v2i_min) / (
        float(environment.time_fast) * float(environment.bandwidth)
    )
    gamma_required = float(np.exp2(spectral_threshold) - 1.0)
    required_all = required_power_dbm(
        channel_loss,
        interference,
        gamma_required,
        environment.veh_ant_gain,
        environment.bs_ant_gain,
        environment.bs_noise_figure,
    )
    noise_only = np.full_like(interference, float(environment.sig2))
    required_noise = required_power_dbm(
        channel_loss,
        noise_only,
        gamma_required,
        environment.veh_ant_gain,
        environment.bs_ant_gain,
        environment.bs_noise_figure,
    )
    selected_required = required_all[np.arange(p), rb_indices]
    selected_interference = interference[np.arange(p), rb_indices]
    selected_loss = channel_loss[np.arange(p), rb_indices]
    signal = np.zeros(p, dtype=np.float64)
    v2i = modes == 0
    signal[v2i] = 10.0 ** (
        (
            decoded[v2i, 2]
            - selected_loss[v2i]
            + environment.veh_ant_gain
            + environment.bs_ant_gain
            - environment.bs_noise_figure
        )
        / 10.0
    )
    rate = np.log2(1.0 + signal / selected_interference) * environment.time_fast * environment.bandwidth
    signal_at_power_max = 10.0 ** (
        (
            float(environment.config.power_max_dbm)
            - channel_loss
            + environment.veh_ant_gain
            + environment.bs_ant_gain
            - environment.bs_noise_figure
        )
        / 10.0
    )
    rate_at_max_current = (
        np.log2(1.0 + signal_at_power_max / interference)
        * environment.time_fast
        * environment.bandwidth
    )
    rate_at_max_noise = (
        np.log2(1.0 + signal_at_power_max / noise_only)
        * environment.time_fast
        * environment.bandwidth
    )
    rsu = np.asarray(environment.config.rsu_position, dtype=np.float64)
    positions = np.asarray([environment.vehicles[int(leader)].position for leader in leaders], dtype=np.float64)
    distance = np.linalg.norm(positions - rsu[None, :], axis=1)
    power_max = float(environment.config.power_max_dbm)
    return FeasibilitySnapshot(
        leader_bs_distance_m=distance.astype(np.float32),
        v2i_channel_loss_db=channel_loss.astype(np.float32),
        v2i_interference_plus_noise_linear=interference,
        required_power_selected_dbm=selected_required.astype(np.float32),
        required_power_best_current_interference_dbm=required_all.min(axis=1).astype(np.float32),
        required_power_noise_only_best_dbm=required_noise.min(axis=1).astype(np.float32),
        best_current_interference_rate_at_power_max=rate_at_max_current.max(axis=1).astype(np.float32),
        noise_only_best_rate_at_power_max=rate_at_max_noise.max(axis=1).astype(np.float32),
        selected_feasible=selected_required <= power_max,
        best_current_interference_feasible=required_all.min(axis=1) <= power_max,
        noise_only_best_feasible=required_noise.min(axis=1) <= power_max,
        reconstructed_v2i_rate=rate.astype(np.float32),
        gamma_required=gamma_required,
    )


def reward_ledger(
    info: dict,
    reward_global: float,
    reward_task1: np.ndarray,
    reward_task2: np.ndarray,
    cam_bits: float,
    global_actor_weight: float,
) -> dict[str, np.ndarray | float]:
    """Record source rewards once and expose their directly auditable drivers."""

    mode = np.asarray(info["mode"], dtype=np.int64)
    aoi = np.asarray(info["aoi_ms"], dtype=np.float64)
    demand = np.asarray(info["remaining_demand"], dtype=np.float64)
    power = np.asarray(info["power_dbm"], dtype=np.float64)
    reset = np.isclose(aoi, 1.0, rtol=0.0, atol=1e-6)
    task1 = np.asarray(reward_task1, dtype=np.float64)
    task2 = np.asarray(reward_task2, dtype=np.float64)
    power_cost = np.asarray([power_penalty(value) for value in power], dtype=np.float64)
    demand_term = -4.95 * demand / float(cam_bits)
    aoi_term = -aoi / 20.0
    revenue_term = 0.05 * reset.astype(np.float64)
    expected_task1 = demand_term - power_cost * (mode == 1)
    expected_task2 = revenue_term + aoi_term - power_cost * (mode == 0)
    if not np.allclose(task1, expected_task1, rtol=1e-5, atol=2e-6):
        raise RuntimeError("task1 reward ledger does not match environment reward")
    if not np.allclose(task2, expected_task2, rtol=1e-5, atol=2e-6):
        raise RuntimeError("task2 reward ledger does not match environment reward")
    combined = task1 + task2 + float(global_actor_weight) * float(reward_global)
    return {
        "reward_global": float(reward_global),
        "reward_task1": task1.astype(np.float32),
        "reward_task2": task2.astype(np.float32),
        "reward_combined": combined.astype(np.float32),
        "reward_remaining_demand_term": demand_term.astype(np.float32),
        "reward_aoi_term": aoi_term.astype(np.float32),
        "reward_v2i_revenue_term": revenue_term.astype(np.float32),
        "reward_power_penalty": power_cost.astype(np.float32),
    }


def _record_step(
    info: dict,
    intervened: IntervenedAction,
    feasibility: FeasibilitySnapshot,
    reward_global: float,
    reward_task1: np.ndarray,
    reward_task2: np.ndarray,
    cam_bits: float,
    global_actor_weight: float,
) -> dict:
    """Copy aligned observations without sampling RNG or advancing state."""

    if not np.allclose(
        feasibility.reconstructed_v2i_rate,
        np.asarray(info["v2i_rate"], dtype=np.float32),
        rtol=1e-5,
        atol=1e-3,
    ):
        raise RuntimeError("pre-step V2I rate reconstruction is not aligned with env.step")
    record = {
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
        "leader_bs_distance_m": feasibility.leader_bs_distance_m.copy(),
        "v2i_channel_loss_db": feasibility.v2i_channel_loss_db.copy(),
        "v2i_interference_plus_noise_linear": feasibility.v2i_interference_plus_noise_linear.copy(),
        "required_power_selected_dbm": feasibility.required_power_selected_dbm.copy(),
        "required_power_best_current_interference_dbm": feasibility.required_power_best_current_interference_dbm.copy(),
        "required_power_noise_only_best_dbm": feasibility.required_power_noise_only_best_dbm.copy(),
        "best_current_interference_rate_at_power_max": feasibility.best_current_interference_rate_at_power_max.copy(),
        "noise_only_best_rate_at_power_max": feasibility.noise_only_best_rate_at_power_max.copy(),
        "selected_feasible": feasibility.selected_feasible.copy(),
        "best_current_interference_feasible": feasibility.best_current_interference_feasible.copy(),
        "noise_only_best_feasible": feasibility.noise_only_best_feasible.copy(),
    }
    record.update(
        reward_ledger(
            info,
            reward_global,
            reward_task1,
            reward_task2,
            cam_bits,
            global_actor_weight,
        )
    )
    return record


def _summary(arrays: dict[str, np.ndarray], cam_bits: float) -> dict:
    aoi = arrays["aoi_ms"].astype(np.float64)
    endpoint_success = arrays["success"][:, :, -1, :].astype(np.float64)
    payload = np.clip(1.0 - arrays["remaining_demand"][:, :, -1, :] / float(cam_bits), 0.0, 1.0)
    per_agent_aoi = aoi.mean(axis=(0, 1, 2))
    per_agent_cam = endpoint_success.mean(axis=(0, 1))
    per_agent_payload = payload.mean(axis=(0, 1))
    attempts = arrays["mode"] == 0
    resets = arrays["reset_event"].astype(bool)
    attempt_count, reset_count = int(attempts.sum()), int(resets.sum())
    return {
        "mean_AoI_ms": float(per_agent_aoi.mean()),
        "worst_agent_mean_AoI_ms": float(per_agent_aoi.max()),
        "CAM_success_probability": float(per_agent_cam.mean()),
        "worst_agent_CAM_success_probability": float(per_agent_cam.min()),
        "payload_completion": float(per_agent_payload.mean()),
        "worst_agent_payload_completion": float(per_agent_payload.min()),
        "mean_executed_power_mw": float(np.power(10.0, arrays["executed_power_dbm"] / 10.0).mean()),
        "v2i_attempt_count": attempt_count,
        "aoi_reset_count": reset_count,
        "v2i_attempt_rate": float(attempt_count / attempts.size),
        "v2i_success_given_attempt": float(reset_count / attempt_count) if attempt_count else None,
        "aoi_reset_rate": float(reset_count / attempts.size),
        "aoi_gt50_fraction": float((aoi > 50.0).mean()),
        "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, rtol=0.0, atol=1e-6).mean()),
        "selected_infeasible_fraction": float((~arrays["selected_feasible"].astype(bool)).mean()),
        "best_current_interference_infeasible_fraction": float((~arrays["best_current_interference_feasible"].astype(bool)).mean()),
        "noise_only_best_infeasible_fraction": float((~arrays["noise_only_best_feasible"].astype(bool)).mean()),
        "mean_reward_global": float(arrays["reward_global"].mean()),
        "mean_reward_task1": float(arrays["reward_task1"].mean()),
        "mean_reward_task2": float(arrays["reward_task2"].mean()),
        "mean_reward_combined": float(arrays["reward_combined"].mean()),
    }


def compare_reference_trajectory(reference_metrics: Path, new_metrics: Path) -> dict:
    """Require exact equality for arrays shared with the previous evaluator."""

    reference_metrics = Path(reference_metrics).expanduser().resolve()
    new_metrics = Path(new_metrics).expanduser().resolve()
    result = {
        "reference_metrics": str(reference_metrics),
        "new_metrics": str(new_metrics),
    }
    exact_all = True
    with np.load(reference_metrics, allow_pickle=False) as reference, np.load(
        new_metrics, allow_pickle=False
    ) as new:
        for key in REFERENCE_TRAJECTORY_ARRAYS:
            if key not in reference or key not in new:
                raise ValueError(f"trajectory equivalence array is missing: {key}")
            exact = bool(np.array_equal(reference[key], new[key]))
            result[f"{key}_exact"] = exact
            exact_all = exact_all and exact
    result["all_common_arrays_exact"] = exact_all
    return result


def evaluate_feasibility_reward(
    policy: Path,
    output_root: Path,
    arm: str,
    device: str = "auto",
    eval_seeds: Iterable[int] = DEFAULT_EVAL_SEEDS,
    eval_episodes: int = 100,
    reference_metrics: Path | None = None,
) -> dict:
    """Evaluate one frozen TDec policy under one V2I power-offset arm."""

    if arm not in ARMS:
        raise ValueError(f"unsupported feasibility arm: {arm}")
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
            raise RuntimeError("feasibility-reward audit requires a TDec MAPPO policy")
        if not training_config.power_continuous:
            raise RuntimeError("feasibility-reward audit requires continuous dBm power mapping")
        runtime_config = copy.deepcopy(training_config)
        runtime_config.device = device
        warmup_episodes = int(runtime_config.eval_warmup_episodes)
        if runtime_config.eval_protocol != "sequential_warm" or warmup_episodes != 5:
            raise RuntimeError("unexpected source evaluation protocol")
        eval_name = feasibility_eval_id(arm, eval_seeds, int(eval_episodes), warmup_episodes)
        eval_dir = output_root / run_dir.name / eval_name
        if eval_dir.exists():
            raise FileExistsError(f"refusing to overwrite feasibility evaluation: {eval_dir}")

        environment, trainer, _rollout, _metrics, resolved_device = _make_mappo_system(runtime_config)
        for actor, state_dict in zip(trainer.actors, payload["actors"]):
            actor.load_state_dict(state_dict, strict=True)
        trainer.actors.eval()
        actor_snapshot = _actor_snapshot(trainer)
        eval_dir.mkdir(parents=True, exist_ok=False)
        raw_by_seed = []
        gamma_required_values = []
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
                    intervention = apply_v2i_power_offset(
                        sampled.environment_actions,
                        arm,
                        runtime_config.power_min_dbm,
                        runtime_config.power_max_dbm,
                        runtime_config.n_rb,
                        runtime_config.n_modes,
                    )
                    observations, _rg, _t1, _t2, _done, _info = environment.step(intervention.environment_actions)
            seed_records = []
            for episode in range(int(eval_episodes)):
                observations = environment.start_episode(warmup_episodes + episode)
                episode_records = []
                for _slot in range(runtime_config.steps_per_episode):
                    sampled = trainer.act(observations, deterministic=False)
                    intervention = apply_v2i_power_offset(
                        sampled.environment_actions,
                        arm,
                        runtime_config.power_min_dbm,
                        runtime_config.power_max_dbm,
                        runtime_config.n_rb,
                        runtime_config.n_modes,
                    )
                    feasibility = compute_feasibility_snapshot(environment, intervention.environment_actions)
                    observations, reward_global, reward_task1, reward_task2, _done, info = environment.step(
                        intervention.environment_actions
                    )
                    gamma_required_values.append(feasibility.gamma_required)
                    episode_records.append(
                        _record_step(
                            info,
                            intervention,
                            feasibility,
                            reward_global,
                            reward_task1,
                            reward_task2,
                            runtime_config.cam_bits,
                            runtime_config.global_actor_weight,
                        )
                    )
                seed_records.append(episode_records)
            raw_by_seed.append(seed_records)
        if not _actors_unchanged(trainer, actor_snapshot):
            raise RuntimeError("frozen-policy evaluation modified actor parameters")
    finally:
        restore_rng_state(caller_rng)

    arrays = _stack_trajectory(raw_by_seed)
    expected_prefix = (
        len(eval_seeds),
        int(eval_episodes),
        runtime_config.steps_per_episode,
        runtime_config.number_agents,
    )
    if arrays["aoi_ms"].shape != expected_prefix:
        raise RuntimeError(f"unexpected feasibility metric shape: {arrays['aoi_ms'].shape}")
    if arrays["v2i_channel_loss_db"].shape != expected_prefix + (runtime_config.n_rb,):
        raise RuntimeError("unexpected per-RB feasibility metric shape")
    metrics_path = eval_dir / "metrics.npz"
    np.savez_compressed(metrics_path, **arrays)
    equivalence = None
    if reference_metrics is not None:
        if arm not in {"baseline", "v2i_plus3db"}:
            raise ValueError("reference comparison is only defined for baseline and v2i_plus3db")
        equivalence = compare_reference_trajectory(reference_metrics, metrics_path)
        _write_json(eval_dir / "trajectory_equivalence.json", equivalence)
        if not equivalence["all_common_arrays_exact"]:
            raise RuntimeError("new evaluator does not reproduce the reference trajectory exactly")
    git_metadata = _git_metadata()
    policy_reference = os.path.relpath(policy_path, eval_dir).replace(os.sep, "/")
    summary = {
        **git_metadata,
        **_summary(arrays, runtime_config.cam_bits),
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": "tdec",
        "intervention_arm": arm,
        "intervention_db": ARM_DELTAS_DB[arm],
        "intervention_semantics": "mode-0 decoded-dBm shift clipped to source power bounds",
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
        "source_gamma": float(runtime_config.gamma),
        "global_actor_weight": float(runtime_config.global_actor_weight),
        "gamma_required": float(np.mean(gamma_required_values)),
        "gamma_required_is_derived": True,
        "v2i_min_bits_per_step": float(runtime_config.v2i_min_bits_per_step),
        "bandwidth_hz": int(runtime_config.bandwidth_hz),
        "slot_ms": float(runtime_config.slot_ms),
        "thermal_noise_dbm": float(environment.sig2_db),
        "veh_antenna_gain_db": float(environment.veh_ant_gain),
        "bs_antenna_gain_db": float(environment.bs_ant_gain),
        "bs_noise_figure_db": float(environment.bs_noise_figure),
        "v2i_channel_loss_semantics": "path loss plus shadowing minus fast-fading gain, in dB",
        "reference_common_arrays_exact": (
            equivalence["all_common_arrays_exact"] if equivalence is not None else None
        ),
        "policy_parameters_unchanged": True,
        "is_frozen_eval": True,
        "diagnostic_evaluation": True,
        "is_formal_result": False,
        "aoi_cap_ms": float(runtime_config.steps_per_episode),
        "cam_bits": float(runtime_config.cam_bits),
        "power_min_dbm": float(runtime_config.power_min_dbm),
        "power_max_dbm": float(runtime_config.power_max_dbm),
        "raw_metric_axes": {
            key: (
                ["eval_seed", "scored_episode", "slot"]
                if arrays[key].ndim == 3
                else ["eval_seed", "scored_episode", "slot", "agent"]
                + (
                    ["action_dim"]
                    if key in {"policy_action_normalized", "action_normalized"}
                    else (["rb"] if arrays[key].ndim == 5 else [])
                )
            )
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
        "feasibility_alignment": "pre-step current fast fading and actions, validated against returned V2I rate",
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
    parser.add_argument("--reference-metrics", type=Path)
    args = parser.parse_args()
    result = evaluate_feasibility_reward(
        policy=args.policy,
        output_root=args.output_root,
        arm=args.arm,
        device=args.device,
        eval_seeds=[int(token) for token in args.eval_seeds.split(",") if token.strip()],
        eval_episodes=args.eval_episodes,
        reference_metrics=args.reference_metrics,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
