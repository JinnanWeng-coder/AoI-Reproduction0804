"""Summarize the minimal independent-versus-shared TDec MAPPO experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.evaluate_mappo_feasibility_reward import feasibility_eval_id


SEEDS = (8, 9, 10, 11, 12, 13)
EVAL_SEEDS = (213, 214, 215, 216, 217, 218)
STRUCTURES = ("independent", "shared")
SCENARIO = "p05_n04_g25"
N_AGENTS = 5
N_RB = 3
TRAIN_EPISODES = 500
STEPS_PER_EPISODE = 100
LAST_WINDOW = 100
PPO_UPDATES = 100
PPO_EPOCHS = 10
EVAL_EPISODES = 100
EVAL_WARMUP = 5

METRIC_FIELDS = (
    "mean_aoi_ms",
    "worst_agent_aoi_ms",
    "aoi_gt50_fraction",
    "aoi_at_cap_fraction",
    "mean_binary_cam",
    "worst_agent_binary_cam",
    "mean_payload_completion",
    "worst_agent_payload_completion",
    "mean_reward_global",
    "mean_reward_task1",
    "mean_reward_task2",
    "mean_reward_combined",
    "mean_power_mw",
    "same_rb_pair_fraction",
    "v2i_pair_same_rb_fraction",
    "rb0_fraction",
    "rb1_fraction",
    "rb2_fraction",
)


def training_run_name(structure: str, seed: int) -> str:
    if structure == "independent":
        return f"mappo_tdec_ab_tdec_{SCENARIO}_seed{int(seed):02d}"
    if structure == "shared":
        return f"mappo_shared_actor_tdec_{SCENARIO}_seed{int(seed):02d}"
    raise ValueError(f"unsupported actor structure: {structure}")


def eval_task_to_structure_seed(task_id: int) -> tuple[str, int]:
    task_id = int(task_id)
    if task_id < 0 or task_id >= 12:
        raise ValueError("task_id must satisfy 0 <= task_id < 12")
    return STRUCTURES[task_id // 6], SEEDS[task_id % 6]


def shared_eval_id() -> str:
    return feasibility_eval_id("baseline", EVAL_SEEDS, EVAL_EPISODES, EVAL_WARMUP)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_power_mw(power_dbm: np.ndarray) -> float:
    return float(np.power(10.0, np.asarray(power_dbm, dtype=np.float64) / 10.0).mean())


def _rb_reuse_metrics(rb: np.ndarray, mode: np.ndarray) -> dict[str, float]:
    rb = np.asarray(rb, dtype=np.int64)
    mode = np.asarray(mode, dtype=np.int64)
    if rb.shape != mode.shape or rb.shape[-1] != N_AGENTS:
        raise ValueError("RB and mode arrays must share an [...,agent] shape")
    same, pair_count = 0, 0
    v2i_same, v2i_pairs = 0, 0
    for left in range(N_AGENTS):
        for right in range(left + 1, N_AGENTS):
            equal = rb[..., left] == rb[..., right]
            same += int(equal.sum())
            pair_count += int(equal.size)
            both_v2i = (mode[..., left] == 0) & (mode[..., right] == 0)
            v2i_same += int((equal & both_v2i).sum())
            v2i_pairs += int(both_v2i.sum())
    result = {
        "same_rb_pair_fraction": float(same / pair_count),
        "v2i_pair_same_rb_fraction": float(v2i_same / v2i_pairs) if v2i_pairs else float("nan"),
    }
    for rb_index in range(N_RB):
        result[f"rb{rb_index}_fraction"] = float((rb == rb_index).mean())
    return result


def _metric_block(
    *,
    aoi: np.ndarray,
    success: np.ndarray,
    remaining: np.ndarray,
    rb: np.ndarray,
    mode: np.ndarray,
    power_dbm: np.ndarray,
    reward_global: np.ndarray,
    reward_task1: np.ndarray,
    reward_task2: np.ndarray,
    cam_bits: float,
    global_actor_weight: float,
) -> dict[str, float | bool]:
    aoi = np.asarray(aoi, dtype=np.float64)
    success = np.asarray(success, dtype=np.float64)
    remaining = np.asarray(remaining, dtype=np.float64)
    if aoi.shape[-2:] != (STEPS_PER_EPISODE, N_AGENTS):
        raise ValueError(f"unexpected AoI shape: {aoi.shape}")
    if success.shape != aoi.shape or remaining.shape != aoi.shape:
        raise ValueError("AoI, success, and remaining demand shapes differ")
    endpoint_cam = success[..., -1, :]
    endpoint_payload = np.clip(1.0 - remaining[..., -1, :] / float(cam_bits), 0.0, 1.0)
    reduction_axes = tuple(range(aoi.ndim - 1))
    per_agent_aoi = aoi.mean(axis=reduction_axes)
    per_agent_cam = endpoint_cam.mean(axis=tuple(range(endpoint_cam.ndim - 1)))
    per_agent_payload = endpoint_payload.mean(axis=tuple(range(endpoint_payload.ndim - 1)))
    global_reward = np.asarray(reward_global, dtype=np.float64)
    task1 = np.asarray(reward_task1, dtype=np.float64)
    task2 = np.asarray(reward_task2, dtype=np.float64)
    expected_local_shape = global_reward.shape + (N_AGENTS,)
    if task1.shape != expected_local_shape or task2.shape != expected_local_shape:
        raise ValueError("reward component shapes do not match")
    combined = task1 + task2 + float(global_actor_weight) * global_reward[..., None]
    result = {
        "mean_aoi_ms": float(per_agent_aoi.mean()),
        "worst_agent_aoi_ms": float(per_agent_aoi.max()),
        "aoi_gt50_fraction": float((aoi > 50.0).mean()),
        "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, rtol=0.0, atol=1e-6).mean()),
        "mean_binary_cam": float(per_agent_cam.mean()),
        "worst_agent_binary_cam": float(per_agent_cam.min()),
        "mean_payload_completion": float(per_agent_payload.mean()),
        "worst_agent_payload_completion": float(per_agent_payload.min()),
        "mean_reward_global": float(global_reward.mean()),
        "mean_reward_task1": float(task1.mean()),
        "mean_reward_task2": float(task2.mean()),
        "mean_reward_combined": float(combined.mean()),
        "mean_power_mw": _mean_power_mw(power_dbm),
    }
    result.update(_rb_reuse_metrics(rb, mode))
    result["screen_success"] = bool(
        result["worst_agent_aoi_ms"] < 50.0
        and result["worst_agent_binary_cam"] >= 0.5
    )
    return result


def _validate_training(
    run_dir: Path, structure: str, seed: int
) -> tuple[dict, dict, list[dict], dict[str, np.ndarray]]:
    config = _read_json(run_dir / "config.resolved.json")
    complete = _read_json(run_dir / "COMPLETE.json")
    diagnostics = json.loads((run_dir / "learning_diagnostics.json").read_text(encoding="utf-8"))
    if not isinstance(diagnostics, list):
        raise ValueError(f"expected diagnostic list: {run_dir}")
    expected_sharing = structure == "shared"
    scenario = config.get("scenario", {})
    checks = {
        "algorithm": (config.get("algorithm"), "mappo"),
        "scenario": (scenario.get("id"), SCENARIO),
        "seed": (config.get("seed"), seed),
        "episodes": (config.get("episodes"), TRAIN_EPISODES),
        "steps_per_episode": (config.get("steps_per_episode"), STEPS_PER_EPISODE),
        "mappo_variant": (config.get("mappo_variant"), "tdec"),
        "mappo_actor_sharing": (bool(config.get("mappo_actor_sharing", False)), expected_sharing),
        "mappo_rollout_episodes": (config.get("mappo_rollout_episodes"), 5),
        "mappo_ppo_epochs": (config.get("mappo_ppo_epochs"), PPO_EPOCHS),
        "mappo_value_clip_mode": (config.get("mappo_value_clip_mode"), "normalized"),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{run_dir.name}: {key}={actual!r}, expected {expected!r}")
    for key, expected in (
        ("mappo_actor_lr", 0.0005),
        ("mappo_critic_lr", 0.0005),
        ("mappo_entropy_coef_rb", 0.02),
        ("mappo_entropy_coef_mode", 0.02),
        ("mappo_entropy_coef_power", 0.002),
        ("tau", 0.005),
    ):
        if not np.isclose(float(config.get(key)), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"{run_dir.name}: {key} mismatch")
    if complete.get("status") != "complete" or int(complete.get("update_count", -1)) != PPO_UPDATES:
        raise ValueError(f"{run_dir.name}: incomplete training")
    if bool(complete.get("actor_sharing", False)) != expected_sharing:
        raise ValueError(f"{run_dir.name}: completion actor-sharing mismatch")
    expected_actor_count = 1 if expected_sharing else N_AGENTS
    if int(complete.get("actor_network_count", expected_actor_count)) != expected_actor_count:
        raise ValueError(f"{run_dir.name}: actor count mismatch")
    if len(diagnostics) != PPO_UPDATES:
        raise ValueError(f"{run_dir.name}: expected {PPO_UPDATES} PPO diagnostics")
    for row in diagnostics:
        if row.get("mappo_variant") != "tdec":
            raise ValueError(f"{run_dir.name}: diagnostic variant mismatch")
        if bool(row.get("actor_sharing", False)) != expected_sharing:
            raise ValueError(f"{run_dir.name}: diagnostic actor-sharing mismatch")
    if not (run_dir / "policy_final.pt").is_file():
        raise ValueError(f"{run_dir.name}: policy_final.pt is missing")
    with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return config, complete, diagnostics, arrays


def _training_rows(independent_root: Path, result_root: Path):
    cells, episodes = [], []
    for structure in STRUCTURES:
        base = independent_root if structure == "independent" else result_root
        for seed in SEEDS:
            name = training_run_name(structure, seed)
            run_dir = base / "training" / "runs" / name
            config, complete, diagnostics, arrays = _validate_training(run_dir, structure, seed)
            required = ("aoi_ms", "success", "remaining_demand", "rb", "mode", "power_dbm", "global_step", "task1_step", "task2_step")
            missing = [key for key in required if key not in arrays]
            if missing:
                raise ValueError(f"{name}: missing training arrays {missing}")
            if arrays["aoi_ms"].shape != (TRAIN_EPISODES, STEPS_PER_EPISODE, N_AGENTS):
                raise ValueError(f"{name}: unexpected training trajectory shape")
            window = slice(TRAIN_EPISODES - LAST_WINDOW, TRAIN_EPISODES)
            metrics = _metric_block(
                aoi=arrays["aoi_ms"][window], success=arrays["success"][window],
                remaining=arrays["remaining_demand"][window], rb=arrays["rb"][window],
                mode=arrays["mode"][window], power_dbm=arrays["power_dbm"][window],
                reward_global=arrays["global_step"][window], reward_task1=arrays["task1_step"][window],
                reward_task2=arrays["task2_step"][window], cam_bits=float(config["cam_bits"]),
                global_actor_weight=float(config["global_actor_weight"]),
            )
            actor_count = 1 if structure == "shared" else N_AGENTS
            expected_steps = PPO_UPDATES * PPO_EPOCHS * actor_count
            recorded_steps = complete.get("actor_optimizer_step_count")
            optimizer_steps = int(recorded_steps) if recorded_steps is not None else expected_steps
            if optimizer_steps != expected_steps:
                raise ValueError(f"{name}: actor optimizer step count mismatch")
            parameter_counts = complete.get("parameter_counts", {})
            cells.append({
                "structure": structure,
                "training_seed": seed,
                "run_name": name,
                "actor_sharing": structure == "shared",
                "actor_network_count": actor_count,
                "actor_parameter_count": int(parameter_counts.get("actors", -1)),
                "total_parameter_count": int(parameter_counts.get("total", -1)),
                "ppo_update_count": int(complete["update_count"]),
                "ppo_epochs": int(config["mappo_ppo_epochs"]),
                "actor_optimizer_step_count": optimizer_steps,
                "actor_optimizer_step_count_source": "recorded" if recorded_steps is not None else "derived_from_legacy_metadata",
                **metrics,
            })
            for episode in range(TRAIN_EPISODES):
                episode_metrics = _metric_block(
                    aoi=arrays["aoi_ms"][episode : episode + 1],
                    success=arrays["success"][episode : episode + 1],
                    remaining=arrays["remaining_demand"][episode : episode + 1],
                    rb=arrays["rb"][episode : episode + 1],
                    mode=arrays["mode"][episode : episode + 1],
                    power_dbm=arrays["power_dbm"][episode : episode + 1],
                    reward_global=arrays["global_step"][episode : episode + 1],
                    reward_task1=arrays["task1_step"][episode : episode + 1],
                    reward_task2=arrays["task2_step"][episode : episode + 1],
                    cam_bits=float(config["cam_bits"]),
                    global_actor_weight=float(config["global_actor_weight"]),
                )
                episodes.append({
                    "structure": structure,
                    "training_seed": seed,
                    "episode": episode + 1,
                    **episode_metrics,
                })
    return cells, episodes


def _validate_evaluation(eval_dir: Path, structure: str, seed: int, run_name: str):
    complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
    expected_sharing = structure == "shared"
    checks = {
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": "tdec",
        "actor_sharing": expected_sharing,
        "actor_network_count": 1 if expected_sharing else N_AGENTS,
        "intervention_arm": "baseline",
        "intervention_db": 0.0,
        "mappo_eval_mode": "stochastic",
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": EVAL_WARMUP,
        "eval_seeds": list(EVAL_SEEDS),
        "eval_episodes": EVAL_EPISODES,
        "training_seed": seed,
        "training_run_name": run_name,
        "policy_parameters_unchanged": True,
    }
    for key, expected in checks.items():
        if complete.get(key) != expected:
            raise ValueError(f"{eval_dir}: {key}={complete.get(key)!r}, expected {expected!r}")
    with np.load(eval_dir / "metrics.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    required = ("aoi_ms", "success", "remaining_demand", "rb", "mode", "executed_power_dbm", "reward_global", "reward_task1", "reward_task2")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{eval_dir}: missing evaluation arrays {missing}")
    if arrays["aoi_ms"].shape != (len(EVAL_SEEDS), EVAL_EPISODES, STEPS_PER_EPISODE, N_AGENTS):
        raise ValueError(f"{eval_dir}: unexpected evaluation trajectory shape")
    return complete, arrays


def _world_agent_rows(arrays: Mapping[str, np.ndarray], structure: str, seed: int, complete: Mapping) -> list[dict]:
    rows = []
    weight = float(complete["global_actor_weight"])
    cam_bits = float(complete["cam_bits"])
    for world_index, eval_seed in enumerate(EVAL_SEEDS):
        for agent in range(N_AGENTS):
            aoi = arrays["aoi_ms"][world_index, :, :, agent].astype(np.float64)
            endpoint_success = arrays["success"][world_index, :, -1, agent].astype(np.float64)
            endpoint_remaining = arrays["remaining_demand"][world_index, :, -1, agent].astype(np.float64)
            rb = arrays["rb"][world_index, :, :, agent].astype(np.int64)
            global_reward = arrays["reward_global"][world_index].astype(np.float64)
            task1 = arrays["reward_task1"][world_index, :, :, agent].astype(np.float64)
            task2 = arrays["reward_task2"][world_index, :, :, agent].astype(np.float64)
            all_rb = arrays["rb"][world_index].astype(np.int64)
            peer_same = np.delete(all_rb, agent, axis=-1) == rb[..., None]
            row = {
                "structure": structure,
                "training_seed": seed,
                "eval_seed": eval_seed,
                "agent": agent,
                "mean_aoi_ms": float(aoi.mean()),
                "aoi_gt50_fraction": float((aoi > 50.0).mean()),
                "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, rtol=0.0, atol=1e-6).mean()),
                "binary_cam": float(endpoint_success.mean()),
                "payload_completion": float(np.clip(1.0 - endpoint_remaining / cam_bits, 0.0, 1.0).mean()),
                "mean_reward_global": float(global_reward.mean()),
                "mean_reward_task1": float(task1.mean()),
                "mean_reward_task2": float(task2.mean()),
                "mean_reward_combined": float((task1 + task2 + weight * global_reward).mean()),
                "mean_power_mw": _mean_power_mw(arrays["executed_power_dbm"][world_index, :, :, agent]),
                "same_rb_peer_fraction": float(peer_same.mean()),
            }
            for rb_index in range(N_RB):
                row[f"rb{rb_index}_fraction"] = float((rb == rb_index).mean())
            rows.append(row)
    return rows


def _evaluation_rows(result_root: Path):
    cells, streams = [], []
    for structure in STRUCTURES:
        for seed in SEEDS:
            name = training_run_name(structure, seed)
            eval_dir = result_root / "evaluations" / name / shared_eval_id()
            complete, arrays = _validate_evaluation(eval_dir, structure, seed, name)
            metrics = _metric_block(
                aoi=arrays["aoi_ms"], success=arrays["success"],
                remaining=arrays["remaining_demand"], rb=arrays["rb"], mode=arrays["mode"],
                power_dbm=arrays["executed_power_dbm"], reward_global=arrays["reward_global"],
                reward_task1=arrays["reward_task1"], reward_task2=arrays["reward_task2"],
                cam_bits=float(complete["cam_bits"]),
                global_actor_weight=float(complete["global_actor_weight"]),
            )
            cells.append({
                "structure": structure,
                "training_seed": seed,
                "training_run_name": name,
                "actor_sharing": structure == "shared",
                "actor_network_count": int(complete["actor_network_count"]),
                **metrics,
            })
            streams.extend(_world_agent_rows(arrays, structure, seed, complete))
    return cells, streams


def _aggregate(rows: Sequence[Mapping], structure: str) -> dict:
    selected = [row for row in rows if row["structure"] == structure]
    if len(selected) != len(SEEDS):
        raise ValueError(f"expected six seed rows for {structure}")
    result = {"structure": structure, "training_seed_count": len(selected)}
    for key in METRIC_FIELDS:
        values = np.asarray([row[key] for row in selected], dtype=np.float64)
        result[key] = float(np.nanmean(values))
        result[f"{key}_sd_across_training_seeds"] = float(np.nanstd(values, ddof=1))
    result["screen_success_count"] = int(sum(bool(row["screen_success"]) for row in selected))
    result["actor_network_count"] = int(selected[0]["actor_network_count"])
    if "actor_parameter_count" in selected[0]:
        for key in (
            "actor_parameter_count", "total_parameter_count", "ppo_update_count",
            "ppo_epochs", "actor_optimizer_step_count",
        ):
            values = {int(row[key]) for row in selected}
            if len(values) != 1:
                raise ValueError(f"{structure}: {key} differs across training seeds")
            result[key] = values.pop()
    return result


def _paired_seed_rows(training_rows: Sequence[Mapping], evaluation_rows: Sequence[Mapping]) -> list[dict]:
    rows = []
    for phase, source in (("train_last100", training_rows), ("heldout_stochastic", evaluation_rows)):
        indexed = {(row["structure"], int(row["training_seed"])): row for row in source}
        for seed in SEEDS:
            independent = indexed[("independent", seed)]
            shared = indexed[("shared", seed)]
            row = {"phase": phase, "training_seed": seed, "delta_definition": "shared_minus_independent"}
            for key in METRIC_FIELDS:
                row[f"delta_{key}"] = float(shared[key]) - float(independent[key])
            rows.append(row)
    return rows


def _paired_stream_rows(streams: Sequence[Mapping]) -> list[dict]:
    indexed = {
        (row["structure"], int(row["training_seed"]), int(row["eval_seed"]), int(row["agent"])): row
        for row in streams
    }
    fields = (
        "mean_aoi_ms", "aoi_gt50_fraction", "aoi_at_cap_fraction", "binary_cam",
        "payload_completion", "mean_reward_global", "mean_reward_task1", "mean_reward_task2",
        "mean_reward_combined", "mean_power_mw", "same_rb_peer_fraction",
        "rb0_fraction", "rb1_fraction", "rb2_fraction",
    )
    rows = []
    for seed in SEEDS:
        for eval_seed in EVAL_SEEDS:
            for agent in range(N_AGENTS):
                independent = indexed[("independent", seed, eval_seed, agent)]
                shared = indexed[("shared", seed, eval_seed, agent)]
                row = {
                    "training_seed": seed,
                    "eval_seed": eval_seed,
                    "agent": agent,
                    "delta_definition": "shared_minus_independent",
                }
                for key in fields:
                    row[f"delta_{key}"] = float(shared[key]) - float(independent[key])
                rows.append(row)
    return rows


def summarize(independent_root: Path, result_root: Path) -> dict:
    independent_root = independent_root.expanduser().resolve()
    result_root = result_root.expanduser().resolve()
    training_rows, episode_rows = _training_rows(independent_root, result_root)
    evaluation_rows, stream_rows = _evaluation_rows(result_root)
    paired_seed = _paired_seed_rows(training_rows, evaluation_rows)
    paired_stream = _paired_stream_rows(stream_rows)
    return {
        "status": "PASS",
        "contract": {
            "scenario": SCENARIO,
            "training_seeds": list(SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
            "actor_structures": list(STRUCTURES),
            "mappo_variant": "tdec",
            "mappo_eval_mode": "stochastic",
            "eval_protocol": "sequential_warm",
            "eval_warmup_episodes": EVAL_WARMUP,
            "eval_episodes": EVAL_EPISODES,
            "paired_delta_definition": "shared_minus_independent",
            "independent_training_reused": True,
            "shared_training_cells_planned": 6,
            "evaluation_cells_planned": 12,
        },
        "shared_training_cells": sum(row["structure"] == "shared" for row in training_rows),
        "evaluation_cells": len(evaluation_rows),
        "training_per_seed_rows": len(training_rows),
        "training_per_episode_rows": len(episode_rows),
        "evaluation_per_seed_rows": len(evaluation_rows),
        "eval_world_agent_rows": len(stream_rows),
        "paired_training_seed_rows": len(paired_seed),
        "paired_eval_world_agent_rows": len(paired_stream),
        "training_summary": [_aggregate(training_rows, structure) for structure in STRUCTURES],
        "evaluation_summary": [_aggregate(evaluation_rows, structure) for structure in STRUCTURES],
        "training_per_seed": training_rows,
        "evaluation_per_seed": evaluation_rows,
        "paired_per_seed": paired_seed,
        "training_per_episode": episode_rows,
        "eval_world_agent": stream_rows,
        "paired_eval_world_agent": paired_stream,
    }


def _write_markdown(path: Path, report: Mapping) -> None:
    lines = [
        "# TDec MAPPO independent versus shared actor",
        "",
        "Independent training is reused from `tdec-ab-v1`; shared training and both frozen evaluations use the current experiment. Held-out evaluation is stochastic on worlds 213--218 with no action intervention.",
        "",
    ]
    for title, key in (("Training last 100 episodes", "training_summary"), ("Frozen held-out evaluation", "evaluation_summary")):
        lines.extend([
            f"## {title}",
            "",
            "| structure | AoI mean±SD | worst AoI | AoI>50 | cap | binary CAM mean/worst | payload mean/worst | combined reward | power mW | same-RB | screen | actors |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in report[key]:
            lines.append(
                f"| {row['structure']} | {row['mean_aoi_ms']:.3f}±{row['mean_aoi_ms_sd_across_training_seeds']:.3f} "
                f"| {row['worst_agent_aoi_ms']:.3f} | {row['aoi_gt50_fraction']:.4f} | {row['aoi_at_cap_fraction']:.4f} "
                f"| {row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} "
                f"| {row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} "
                f"| {row['mean_reward_combined']:.4f} | {row['mean_power_mw']:.3f} "
                f"| {row['same_rb_pair_fraction']:.4f} | {row['screen_success_count']}/6 | {row['actor_network_count']} |"
            )
        lines.append("")
        lines.extend([
            "Reward components are the original environment outputs; global is included once per agent when composing the combined reward.",
            "",
            "| structure | global reward | task1 | task2 | combined | power mW | RB 0/1/2 | V2I-pair same-RB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in report[key]:
            lines.append(
                f"| {row['structure']} | {row['mean_reward_global']:.4f} | {row['mean_reward_task1']:.4f} "
                f"| {row['mean_reward_task2']:.4f} | {row['mean_reward_combined']:.4f} "
                f"| {row['mean_power_mw']:.3f} | {row['rb0_fraction']:.3f}/{row['rb1_fraction']:.3f}/{row['rb2_fraction']:.3f} "
                f"| {row['v2i_pair_same_rb_fraction']:.4f} |"
            )
        lines.append("")
    lines.extend([
        "## Training structure and update counts",
        "",
        "| structure | actors | actor parameters | total parameters | PPO updates | epochs/update | actor optimizer steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["training_summary"]:
        lines.append(
            f"| {row['structure']} | {row['actor_network_count']} | {row['actor_parameter_count']} "
            f"| {row['total_parameter_count']} | {row['ppo_update_count']} | {row['ppo_epochs']} "
            f"| {row['actor_optimizer_step_count']} |"
        )
    lines.extend([
        "",
        "The independent policies predate the actor-step metadata, so their total optimizer-step count is derived as updates × PPO epochs × actor count; shared counts are recorded by training.",
        "",
    ])
    lines.extend([
        "Paired deltas are shared minus independent. RB reuse is reported only as a mechanism diagnostic; a smaller reuse fraction is not itself a success criterion.",
        "",
        "## Per-seed paired deltas",
        "",
        "| phase | seed | ΔAoI | Δbinary CAM | Δpayload | Δreward | Δpower mW | Δsame-RB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["paired_per_seed"]:
        lines.append(
            f"| {row['phase']} | {row['training_seed']} | {row['delta_mean_aoi_ms']:+.3f} "
            f"| {row['delta_mean_binary_cam']:+.4f} | {row['delta_mean_payload_completion']:+.4f} "
            f"| {row['delta_mean_reward_combined']:+.4f} | {row['delta_mean_power_mw']:+.3f} "
            f"| {row['delta_same_rb_pair_fraction']:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(result_root: Path, report: Mapping) -> Path:
    output = result_root.expanduser().resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "shared_actor_training_per_seed.csv", report["training_per_seed"])
    _write_csv(output / "shared_actor_training_per_episode.csv", report["training_per_episode"])
    _write_csv(output / "shared_actor_eval_per_seed.csv", report["evaluation_per_seed"])
    _write_csv(output / "shared_actor_eval_world_agent.csv", report["eval_world_agent"])
    _write_csv(output / "shared_actor_paired_per_seed.csv", report["paired_per_seed"])
    _write_csv(output / "shared_actor_paired_eval_world_agent.csv", report["paired_eval_world_agent"])
    _write_csv(output / "shared_actor_training_summary.csv", report["training_summary"])
    _write_csv(output / "shared_actor_eval_summary.csv", report["evaluation_summary"])
    compact = {
        key: value
        for key, value in report.items()
        if key not in {"training_per_episode", "eval_world_agent", "paired_eval_world_agent"}
    }
    (output / "shared_actor_summary.json").write_text(
        json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_markdown(output / "shared_actor_comparison.md", report)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.independent_root, args.result_root)
    output = write_report(args.result_root, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "shared_training_cells": report["shared_training_cells"],
        "evaluation_cells": report["evaluation_cells"],
        "training_summary": report["training_summary"],
        "evaluation_summary": report["evaluation_summary"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def write_report(result_root: Path, report: Mapping) -> Path:
    output = result_root.expanduser().resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "shared_actor_training_per_seed.csv", report["training_per_seed"])
    _write_csv(output / "shared_actor_training_per_episode.csv", report["training_per_episode"])
    _write_csv(output / "shared_actor_eval_per_seed.csv", report["evaluation_per_seed"])
    _write_csv(output / "shared_actor_eval_world_agent.csv", report["eval_world_agent"])
    _write_csv(output / "shared_actor_paired_per_seed.csv", report["paired_per_seed"])
    _write_csv(output / "shared_actor_paired_eval_world_agent.csv", report["paired_eval_world_agent"])
    compact = {key: value for key, value in report.items() if key not in {"training_per_episode", "eval_world_agent", "paired_eval_world_agent"}}
    (output / "shared_actor_summary.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output / "shared_actor_comparison.md", report)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.independent_root, args.result_root)
    output = write_report(args.result_root, report)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "training_summary": report["training_summary"],
        "evaluation_summary": report["evaluation_summary"],
        "paired_per_seed": report["paired_per_seed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
