"""Frozen TDec MAPPO evaluation with a centralized strict-improvement RB oracle."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from analysis.evaluate_mappo_feasibility_reward import (
    _record_step,
    _summary,
    compute_feasibility_snapshot,
)
from analysis.evaluate_mappo_power_intervention import (
    POLICY_SEED_TAG,
    IntervenedAction,
    _actor_snapshot,
    _actors_unchanged,
    _stack_trajectory,
)
from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state, restore_rng_state
from aoi_v2x_reproduction.runtime.runner import (
    _git_metadata,
    _make_mappo_system,
    _validate_mappo_policy_artifact,
    _write_json,
)


ARMS = ("baseline", "joint_rb_strict_improve")
TRAINING_SEEDS = (8, 9, 10, 11, 12, 13)
DEFAULT_EVAL_SEEDS = (207, 208, 209, 210, 211, 212)
JOINT_RB_ASSIGNMENTS = np.asarray(tuple(itertools.product(range(3), repeat=5)), dtype=np.int64)


@dataclass(frozen=True)
class RBDecision:
    intervened: IntervenedAction
    policy_rb: np.ndarray
    executed_rb: np.ndarray
    original_success_count: int
    oracle_success_count: int
    strict_improvement_applied: bool
    rb_changed: np.ndarray
    oracle_evaluated: bool


def task_to_arm_seed(task_id: int) -> tuple[str, int]:
    task_id = int(task_id)
    if task_id < 0 or task_id >= 12:
        raise ValueError("task_id must satisfy 0 <= task_id < 12")
    return ARMS[task_id // 6], TRAINING_SEEDS[task_id % 6]


def rb_intervention_eval_id(
    arm: str,
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    eval_episodes: int = 100,
    warmup_episodes: int = 5,
) -> str:
    if arm not in ARMS:
        raise ValueError(f"unsupported RB intervention arm: {arm}")
    seed_token = "-".join(str(int(seed)) for seed in eval_seeds)
    return (
        "eval_validation_policy_final_stochastic_sequential_warm_"
        f"warm{int(warmup_episodes)}_s{seed_token}_ep{int(eval_episodes)}_rb_arm_{arm}"
    )


def _decode_normalized_actions(
    actions: np.ndarray,
    power_min_dbm: float,
    power_max_dbm: float,
    n_rb: int,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(actions)
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.all(np.isfinite(raw)):
        raise ValueError("actions must be a finite [agent, 3] array")
    normalized = np.clip(raw.astype(np.float64, copy=False), -1.0, 1.0)
    rb = np.minimum(int(n_rb) - 1, np.floor((normalized[:, 0] + 1.0) * 0.5 * int(n_rb))).astype(np.int64)
    mode = np.minimum(
        int(n_modes) - 1,
        np.floor((normalized[:, 1] + 1.0) * 0.5 * int(n_modes)),
    ).astype(np.int64)
    power = float(power_min_dbm) + (normalized[:, 2] + 1.0) * 0.5 * (
        float(power_max_dbm) - float(power_min_dbm)
    )
    return rb, mode, power


def batched_joint_v2i_rates(environment, decoded_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate all 3^5 joint RB assignments without stepping the environment."""

    decoded = np.asarray(decoded_actions, dtype=np.float64)
    if environment.n_platoon != 5 or environment.n_rb != 3 or decoded.shape != (5, 3):
        raise ValueError("RB intervention is fixed to P=5 and three RBs")
    candidates = JOINT_RB_ASSIGNMENTS
    mode = decoded[:, 1].astype(np.int64)
    power = decoded[:, 2]
    leaders = np.arange(5, dtype=np.int64) * int(environment.size_platoon)
    channel = np.asarray(environment.v2i_channels_fast[leaders, :], dtype=np.float64)
    candidate_loss = channel[np.arange(5)[None, :], candidates]
    received = np.power(
        10.0,
        (
            power[None, :]
            - candidate_loss
            + float(environment.veh_ant_gain)
            + float(environment.bs_ant_gain)
            - float(environment.bs_noise_figure)
        )
        / 10.0,
    )
    same_rb = candidates[:, :, None] == candidates[:, None, :]
    not_self = ~np.eye(5, dtype=bool)[None, :, :]
    interference = float(environment.sig2) + np.sum(
        same_rb * not_self * received[:, None, :], axis=2
    )
    signal = np.where(mode[None, :] == 0, received, 0.0)
    rates = (
        np.log2(1.0 + signal / interference)
        * float(environment.time_fast)
        * float(environment.bandwidth)
    )
    return rates, interference


def select_joint_rb_strict_improvement(environment, policy_actions: np.ndarray, arm: str) -> RBDecision:
    """Change only RB when the centralized oracle strictly increases V2I successes."""

    if arm not in ARMS:
        raise ValueError(f"unsupported RB intervention arm: {arm}")
    actions = np.asarray(policy_actions)
    rb, mode, power = _decode_normalized_actions(
        actions,
        environment.config.power_min_dbm,
        environment.config.power_max_dbm,
        environment.n_rb,
        environment.config.n_modes,
    )
    zeros = np.zeros(5, dtype=bool)
    boundary = (
        np.isclose(power, float(environment.config.power_min_dbm), rtol=0.0, atol=1e-5)
        | np.isclose(power, float(environment.config.power_max_dbm), rtol=0.0, atol=1e-5)
    )
    if arm == "baseline":
        intervened = IntervenedAction(
            policy_actions=actions,
            environment_actions=actions,
            rb=rb.copy(),
            mode=mode.copy(),
            policy_power_dbm=power.astype(np.float32),
            executed_power_dbm=power.astype(np.float32),
            power_clipped_low=zeros.copy(),
            power_clipped_high=zeros.copy(),
            power_at_boundary=boundary,
        )
        return RBDecision(intervened, rb, rb.copy(), -1, -1, False, zeros.copy(), False)

    decoded = np.column_stack((rb, mode, power))
    rates, _interference = batched_joint_v2i_rates(environment, decoded)
    success = (mode[None, :] == 0) & (rates >= float(environment.v2i_min))
    original_index = int(np.flatnonzero(np.all(JOINT_RB_ASSIGNMENTS == rb[None, :], axis=1))[0])
    original_success = success[original_index]
    success_count = success.sum(axis=1)
    preserve_count = (success & original_success[None, :]).sum(axis=1)
    changed_count = (JOINT_RB_ASSIGNMENTS != rb[None, :]).sum(axis=1)
    eligible = np.flatnonzero(success_count == success_count.max())
    eligible = eligible[preserve_count[eligible] == preserve_count[eligible].max()]
    eligible = eligible[changed_count[eligible] == changed_count[eligible].min()]
    best_index = int(eligible[0])
    original_count = int(success_count[original_index])
    best_count = int(success_count[best_index])
    apply = best_count > original_count
    executed_rb = JOINT_RB_ASSIGNMENTS[best_index].copy() if apply else rb.copy()
    changed = executed_rb != rb
    executed_actions = actions.copy() if apply else actions
    if apply:
        executed_actions[:, 0] = (
            -1.0 + 2.0 * (executed_rb.astype(np.float64) + 0.5) / float(environment.n_rb)
        ).astype(executed_actions.dtype, copy=False)
    decoded_executed = environment.decode_actions(executed_actions)
    if not np.array_equal(decoded_executed[:, 0].astype(np.int64), executed_rb):
        raise AssertionError("encoded oracle RB does not decode correctly")
    if not np.array_equal(decoded_executed[:, 1].astype(np.int64), mode):
        raise AssertionError("RB intervention changed mode")
    if not np.allclose(decoded_executed[:, 2], power, rtol=0.0, atol=2e-5):
        raise AssertionError("RB intervention changed power")
    intervened = IntervenedAction(
        policy_actions=actions,
        environment_actions=executed_actions,
        rb=executed_rb,
        mode=mode,
        policy_power_dbm=power.astype(np.float32),
        executed_power_dbm=power.astype(np.float32),
        power_clipped_low=zeros.copy(),
        power_clipped_high=zeros.copy(),
        power_at_boundary=boundary,
    )
    return RBDecision(
        intervened,
        rb,
        executed_rb,
        original_count,
        best_count,
        bool(apply),
        changed,
        True,
    )


def _record_rb_step(
    info: dict,
    decision: RBDecision,
    feasibility,
    reward_global: float,
    reward_task1: np.ndarray,
    reward_task2: np.ndarray,
    cam_bits: float,
    global_actor_weight: float,
) -> dict:
    record = _record_step(
        info,
        decision.intervened,
        feasibility,
        reward_global,
        reward_task1,
        reward_task2,
        cam_bits,
        global_actor_weight,
    )
    record.update({
        "policy_rb": decision.policy_rb.copy(),
        "executed_rb": decision.executed_rb.copy(),
        "rb_changed": decision.rb_changed.copy(),
        "oracle_original_v2i_success_count": decision.original_success_count,
        "oracle_best_v2i_success_count": decision.oracle_success_count,
        "oracle_strict_improvement_applied": decision.strict_improvement_applied,
        "oracle_evaluated": decision.oracle_evaluated,
    })
    return record


def evaluate_rb_intervention(
    policy: Path,
    output_root: Path,
    arm: str,
    device: str = "auto",
    eval_seeds: Iterable[int] = DEFAULT_EVAL_SEEDS,
    eval_episodes: int = 100,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unsupported RB intervention arm: {arm}")
    eval_seeds = tuple(int(seed) for seed in eval_seeds)
    if not eval_seeds or len(eval_seeds) != len(set(eval_seeds)) or int(eval_episodes) < 1:
        raise ValueError("eval_seeds must be unique and eval_episodes positive")
    policy_path = Path(policy).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    caller_rng = capture_rng_state()
    try:
        payload = torch.load(policy_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError("MAPPO policy payload must be a dictionary")
        run_dir, training_config, training_complete = _validate_mappo_policy_artifact(policy_path, payload)
        if training_config.algorithm != "mappo" or training_config.mappo_variant != "tdec":
            raise RuntimeError("RB intervention requires a TDec MAPPO policy")
        runtime_config = copy.deepcopy(training_config)
        runtime_config.device = device
        warmup_episodes = int(runtime_config.eval_warmup_episodes)
        if runtime_config.eval_protocol != "sequential_warm" or warmup_episodes != 5:
            raise RuntimeError("unexpected source evaluation protocol")
        if runtime_config.number_agents != 5 or runtime_config.n_rb != 3:
            raise RuntimeError("RB intervention v1 requires P=5 and three RBs")
        eval_name = rb_intervention_eval_id(arm, eval_seeds, int(eval_episodes), warmup_episodes)
        eval_dir = output_root / run_dir.name / eval_name
        if eval_dir.exists():
            raise FileExistsError(f"refusing to overwrite RB intervention evaluation: {eval_dir}")

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
                    decision = select_joint_rb_strict_improvement(
                        environment, sampled.environment_actions, arm
                    )
                    observations, *_unused = environment.step(decision.intervened.environment_actions)
            seed_records = []
            for episode in range(int(eval_episodes)):
                observations = environment.start_episode(warmup_episodes + episode)
                episode_records = []
                for _slot in range(runtime_config.steps_per_episode):
                    sampled = trainer.act(observations, deterministic=False)
                    decision = select_joint_rb_strict_improvement(
                        environment, sampled.environment_actions, arm
                    )
                    feasibility = compute_feasibility_snapshot(
                        environment, decision.intervened.environment_actions
                    )
                    observations, rg, task1, task2, _done, info = environment.step(
                        decision.intervened.environment_actions
                    )
                    episode_records.append(
                        _record_rb_step(
                            info,
                            decision,
                            feasibility,
                            rg,
                            task1,
                            task2,
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
    expected = (
        len(eval_seeds), int(eval_episodes), runtime_config.steps_per_episode, runtime_config.number_agents
    )
    if arrays["aoi_ms"].shape != expected:
        raise RuntimeError(f"unexpected RB intervention metric shape: {arrays['aoi_ms'].shape}")
    if not np.array_equal(arrays["mode"], _decode_action_component(arrays["policy_action_normalized"], 1, runtime_config.n_modes)):
        raise RuntimeError("recorded executed mode differs from policy mode")
    if not np.allclose(
        arrays["power_dbm"],
        runtime_config.power_min_dbm
        + (np.clip(arrays["policy_action_normalized"][..., 2], -1.0, 1.0) + 1.0)
        * 0.5
        * (runtime_config.power_max_dbm - runtime_config.power_min_dbm),
        rtol=0.0,
        atol=2e-5,
    ):
        raise RuntimeError("recorded executed power differs from policy power")
    np.savez_compressed(eval_dir / "metrics.npz", **arrays)
    git_metadata = _git_metadata()
    policy_reference = os.path.relpath(policy_path, eval_dir).replace(os.sep, "/")
    applied = arrays["oracle_strict_improvement_applied"].astype(bool)
    summary = {
        **git_metadata,
        **_summary(arrays, runtime_config.cam_bits),
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": "tdec",
        "intervention_arm": arm,
        "intervention_semantics": (
            "centralized 3^5 joint-RB oracle; apply only when instantaneous V2I success count strictly increases"
        ),
        "oracle_tie_break": "preserve original V2I successes, then minimize changed RBs, then enumeration order",
        "oracle_candidate_count": 243,
        "oracle_applied_slot_count": int(applied.sum()),
        "oracle_applied_slot_fraction": float(applied.mean()),
        "mean_changed_agents_when_applied": (
            float(arrays["rb_changed"][applied].sum(axis=-1).mean()) if applied.any() else 0.0
        ),
        "baseline_action_passthrough": (
            bool(np.array_equal(arrays["policy_action_normalized"], arrays["action_normalized"]))
            if arm == "baseline" else None
        ),
        "mappo_eval_mode": "stochastic",
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": warmup_episodes,
        "eval_seeds": list(eval_seeds),
        "eval_episodes": int(eval_episodes),
        "training_seed": int(training_config.seed),
        "training_run_name": run_dir.name,
        "scenario": training_config.scenario.id,
        "policy": policy_reference,
        "policy_name": policy_path.name,
        "policy_schema_version": payload.get("policy_schema_version"),
        "policy_episode": int(payload.get("episode", -1)),
        "source_config_hash": training_config.canonical_hash(),
        "policy_parameters_unchanged": True,
        "is_frozen_eval": True,
        "diagnostic_evaluation": True,
        "is_formal_result": False,
        "cam_bits": float(runtime_config.cam_bits),
        "aoi_cap_ms": float(runtime_config.steps_per_episode),
        "power_min_dbm": float(runtime_config.power_min_dbm),
        "power_max_dbm": float(runtime_config.power_max_dbm),
        "external_action_noise_applicable": False,
        "raw_metric_axes": {
            key: (
                ["eval_seed", "scored_episode", "slot"]
                if arrays[key].ndim == 3
                else ["eval_seed", "scored_episode", "slot", "agent"]
                + (
                    ["action_dim"]
                    if arrays[key].ndim == 5 and key.endswith("normalized")
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
        "eval_seeds": list(eval_seeds),
        "eval_episodes": int(eval_episodes),
        "eval_warmup_episodes": warmup_episodes,
        "mappo_eval_mode": "stochastic",
        "release_status": "centralized_diagnostic_evaluation",
    }
    _write_json(eval_dir / "provenance.json", provenance)
    _write_json(eval_dir / "summary.json", summary)
    _write_json(eval_dir / "EVAL_COMPLETE.json", summary)
    return {"eval_dir": str(eval_dir), **summary}


def _decode_action_component(actions: np.ndarray, index: int, cardinality: int) -> np.ndarray:
    normalized = np.clip(np.asarray(actions, dtype=np.float64)[..., index], -1.0, 1.0)
    return np.minimum(
        int(cardinality) - 1,
        np.floor((normalized + 1.0) * 0.5 * int(cardinality)),
    ).astype(np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-seeds", default=",".join(str(seed) for seed in DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()
    result = evaluate_rb_intervention(
        args.policy,
        args.output_root,
        args.arm,
        args.device,
        [int(token) for token in args.eval_seeds.split(",") if token.strip()],
        args.eval_episodes,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
