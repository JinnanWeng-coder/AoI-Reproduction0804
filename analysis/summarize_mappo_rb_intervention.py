"""Summarize the two-arm frozen MAPPO joint-RB intervention diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.audit_mappo_service_power import power_mw
from analysis.evaluate_mappo_rb_intervention import (
    ARMS,
    DEFAULT_EVAL_SEEDS,
    TRAINING_SEEDS,
    rb_intervention_eval_id,
)
from analysis.summarize_mappo_power_intervention import interval_statistics


SCENARIO = "p05_n04_g25"


def _run_name(seed: int) -> str:
    return f"mappo_tdec_ab_tdec_{SCENARIO}_seed{int(seed):02d}"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _decoded_component(actions: np.ndarray, index: int, cardinality: int) -> np.ndarray:
    values = np.clip(np.asarray(actions, dtype=np.float64)[..., index], -1.0, 1.0)
    return np.minimum(
        int(cardinality) - 1,
        np.floor((values + 1.0) * 0.5 * int(cardinality)),
    ).astype(np.int64)


def validate_actions(arrays: Mapping[str, np.ndarray], complete: Mapping, arm: str) -> dict:
    policy = arrays["policy_action_normalized"]
    executed = arrays["action_normalized"]
    policy_rb = _decoded_component(policy, 0, 3)
    policy_mode = _decoded_component(policy, 1, 2)
    low, high = float(complete["power_min_dbm"]), float(complete["power_max_dbm"])
    policy_power = low + (np.clip(policy[..., 2], -1.0, 1.0) + 1.0) * 0.5 * (high - low)
    rb_direct = np.array_equal(arrays["policy_rb"], policy_rb)
    mode_direct = np.array_equal(arrays["mode"], policy_mode)
    power_direct = np.allclose(arrays["power_dbm"], policy_power, rtol=0.0, atol=2e-5)
    executed_rb = np.array_equal(arrays["executed_rb"], arrays["rb"])
    no_gain = ~arrays["oracle_strict_improvement_applied"].astype(bool)
    unchanged_without_gain = np.array_equal(arrays["executed_rb"][no_gain], arrays["policy_rb"][no_gain])
    applied = arrays["oracle_strict_improvement_applied"].astype(bool)
    strict = bool(np.all(
        arrays["oracle_best_v2i_success_count"][applied]
        > arrays["oracle_original_v2i_success_count"][applied]
    ))
    baseline_passthrough = (
        bool(np.array_equal(policy, executed)) if arm == "baseline" else True
    )
    baseline_no_oracle = (
        bool(not arrays["oracle_evaluated"].astype(bool).any()) if arm == "baseline" else True
    )
    joint_oracle_every_slot = (
        bool(arrays["oracle_evaluated"].astype(bool).all())
        if arm == "joint_rb_strict_improve" else True
    )
    passed = all((
        rb_direct,
        mode_direct,
        power_direct,
        executed_rb,
        unchanged_without_gain,
        strict,
        baseline_passthrough,
        baseline_no_oracle,
        joint_oracle_every_slot,
    ))
    return {
        "arm": arm,
        "training_seed": int(complete["training_seed"]),
        "policy_rb_decodes_exactly": rb_direct,
        "executed_mode_equals_policy_mode": mode_direct,
        "executed_power_equals_policy_power": bool(power_direct),
        "recorded_executed_rb_matches_environment": executed_rb,
        "rb_unchanged_without_strict_gain": unchanged_without_gain,
        "applied_changes_have_strict_gain": strict,
        "baseline_action_passthrough": baseline_passthrough,
        "baseline_bypasses_oracle": baseline_no_oracle,
        "joint_oracle_evaluated_every_slot": joint_oracle_every_slot,
        "passed": passed,
    }


def _episode_agent_rows(
    arrays: Mapping[str, np.ndarray], arm: str, training_seed: int, cam_bits: float
) -> list[dict]:
    rows = []
    for eval_index, eval_seed in enumerate(DEFAULT_EVAL_SEEDS):
        for episode in range(100):
            global_reward = float(arrays["reward_global"][eval_index, episode].mean())
            for agent in range(5):
                aoi = arrays["aoi_ms"][eval_index, episode, :, agent].astype(np.float64)
                remaining = float(arrays["remaining_demand"][eval_index, episode, -1, agent])
                rows.append({
                    "arm": arm,
                    "training_seed": training_seed,
                    "eval_seed": eval_seed,
                    "episode": episode,
                    "agent": agent,
                    "mean_aoi_ms": float(aoi.mean()),
                    "p95_aoi_ms": float(np.quantile(aoi, 0.95)),
                    "aoi_gt50_fraction": float((aoi > 50.0).mean()),
                    "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, atol=1e-6).mean()),
                    "strict_binary_cam": float(arrays["success"][eval_index, episode, -1, agent]),
                    "payload_completion": float(np.clip(1.0 - remaining / cam_bits, 0.0, 1.0)),
                    "mean_reward_global": global_reward,
                    "mean_reward_task1": float(arrays["reward_task1"][eval_index, episode, :, agent].mean()),
                    "mean_reward_task2": float(arrays["reward_task2"][eval_index, episode, :, agent].mean()),
                    "mean_reward_combined": float(arrays["reward_combined"][eval_index, episode, :, agent].mean()),
                    "mean_executed_power_mw": float(
                        power_mw(arrays["executed_power_dbm"][eval_index, episode, :, agent]).mean()
                    ),
                    "rb_changed_fraction": float(arrays["rb_changed"][eval_index, episode, :, agent].mean()),
                    "reset_count": int(arrays["reset_event"][eval_index, episode, :, agent].sum()),
                })
    return rows


def _stream_rows(
    arrays: Mapping[str, np.ndarray], arm: str, training_seed: int, cam_bits: float
) -> list[dict]:
    rows = []
    for eval_index, eval_seed in enumerate(DEFAULT_EVAL_SEEDS):
        for agent in range(5):
            aoi = arrays["aoi_ms"][eval_index, :, :, agent].astype(np.float64)
            endpoint_success = arrays["success"][eval_index, :, -1, agent].astype(np.float64)
            remaining = arrays["remaining_demand"][eval_index, :, -1, agent].astype(np.float64)
            intervals = interval_statistics(aoi)
            rows.append({
                "arm": arm,
                "training_seed": training_seed,
                "eval_seed": eval_seed,
                "agent": agent,
                "mean_aoi_ms": float(aoi.mean()),
                "p95_aoi_ms": float(np.quantile(aoi, 0.95)),
                "p99_aoi_ms": float(np.quantile(aoi, 0.99)),
                "aoi_gt50_fraction": float((aoi > 50.0).mean()),
                "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, atol=1e-6).mean()),
                "strict_binary_cam": float(endpoint_success.mean()),
                "payload_completion": float(np.clip(1.0 - remaining / cam_bits, 0.0, 1.0).mean()),
                "mean_reward_global": float(arrays["reward_global"][eval_index].mean()),
                "mean_reward_task1": float(arrays["reward_task1"][eval_index, :, :, agent].mean()),
                "mean_reward_task2": float(arrays["reward_task2"][eval_index, :, :, agent].mean()),
                "mean_reward_combined": float(arrays["reward_combined"][eval_index, :, :, agent].mean()),
                "mean_executed_power_mw": float(
                    power_mw(arrays["executed_power_dbm"][eval_index, :, :, agent]).mean()
                ),
                "rb_changed_fraction": float(arrays["rb_changed"][eval_index, :, :, agent].mean()),
                "complete_interval_count": intervals["complete_interval_count"],
                "complete_interval_p95_slots": intervals["complete_interval_p95_slots"],
                "complete_interval_gt100_count": intervals["complete_interval_gt100_count"],
                "censored_segment_slots": intervals["censored_segment_slots"],
            })
    return rows


def _cell_row(arrays: Mapping[str, np.ndarray], arm: str, seed: int, cam_bits: float) -> dict:
    per_agent_aoi = arrays["aoi_ms"].mean(axis=(0, 1, 2))
    per_agent_cam = arrays["success"][:, :, -1, :].mean(axis=(0, 1))
    payload = np.clip(
        1.0 - arrays["remaining_demand"][:, :, -1, :] / cam_bits, 0.0, 1.0
    )
    per_agent_payload = payload.mean(axis=(0, 1))
    aoi = arrays["aoi_ms"].astype(np.float64)
    applied = arrays["oracle_strict_improvement_applied"].astype(bool)
    improvement = (
        arrays["oracle_best_v2i_success_count"]
        - arrays["oracle_original_v2i_success_count"]
    )
    complete_intervals = []
    for eval_index in range(6):
        for agent in range(5):
            complete_intervals.extend(
                interval_statistics(aoi[eval_index, :, :, agent])["complete_intervals"].tolist()
            )
    complete_intervals = np.asarray(complete_intervals, dtype=np.int64)
    return {
        "arm": arm,
        "training_seed": seed,
        "mean_aoi_ms": float(per_agent_aoi.mean()),
        "worst_agent_aoi_ms": float(per_agent_aoi.max()),
        "p95_aoi_ms": float(np.quantile(aoi, 0.95)),
        "p99_aoi_ms": float(np.quantile(aoi, 0.99)),
        "aoi_gt50_fraction": float((aoi > 50.0).mean()),
        "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, atol=1e-6).mean()),
        "mean_binary_cam": float(per_agent_cam.mean()),
        "worst_agent_binary_cam": float(per_agent_cam.min()),
        "mean_payload_completion": float(per_agent_payload.mean()),
        "worst_agent_payload_completion": float(per_agent_payload.min()),
        "mean_reward_global": float(arrays["reward_global"].mean()),
        "mean_reward_task1": float(arrays["reward_task1"].mean()),
        "mean_reward_task2": float(arrays["reward_task2"].mean()),
        "mean_reward_combined": float(arrays["reward_combined"].mean()),
        "mean_executed_power_mw": float(power_mw(arrays["executed_power_dbm"]).mean()),
        "oracle_applied_fraction": float(applied.mean()),
        "mean_v2i_success_increment_when_applied": (
            float(improvement[applied].mean()) if applied.any() else 0.0
        ),
        "mean_changed_agents_when_applied": (
            float(arrays["rb_changed"][applied].sum(axis=-1).mean()) if applied.any() else 0.0
        ),
        "complete_interval_count": int(complete_intervals.size),
        "complete_interval_p95_slots": (
            float(np.quantile(complete_intervals, 0.95)) if complete_intervals.size else ""
        ),
        "complete_interval_gt100_count": int((complete_intervals > 100).sum()),
    }


def _paired_rows(cell_rows: Sequence[Mapping]) -> tuple[list[dict], dict]:
    metrics = (
        "mean_aoi_ms",
        "worst_agent_aoi_ms",
        "p95_aoi_ms",
        "p99_aoi_ms",
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
        "mean_executed_power_mw",
        "complete_interval_p95_slots",
        "complete_interval_gt100_count",
    )
    rows = []
    for seed in TRAINING_SEEDS:
        baseline = next(row for row in cell_rows if row["arm"] == "baseline" and row["training_seed"] == seed)
        intervention = next(
            row for row in cell_rows
            if row["arm"] == "joint_rb_strict_improve" and row["training_seed"] == seed
        )
        row = {"training_seed": seed}
        for metric in metrics:
            row[f"delta_{metric}"] = (
                float(intervention[metric]) - float(baseline[metric])
                if intervention[metric] != "" and baseline[metric] != "" else ""
            )
        rows.append(row)
    summary = {"paired_seed_count": len(rows)}
    for metric in metrics:
        values = np.asarray([float(row[f"delta_{metric}"]) for row in rows if row[f"delta_{metric}"] != ""])
        summary[f"mean_delta_{metric}"] = float(values.mean()) if values.size else ""
        summary[f"sd_delta_{metric}"] = float(values.std(ddof=1)) if values.size > 1 else ""
    summary["aoi_improved_seeds"] = sum(row["delta_mean_aoi_ms"] < 0 for row in rows)
    summary["cam_not_worse_seeds"] = sum(row["delta_mean_binary_cam"] >= 0 for row in rows)
    summary["payload_not_worse_seeds"] = sum(row["delta_mean_payload_completion"] >= 0 for row in rows)
    return rows, summary


def _report(path: Path, cell_rows: Sequence[Mapping], paired: Mapping) -> None:
    lines = [
        "# MAPPO frozen-policy strict joint-RB intervention",
        "",
        "Worlds 207-212 are new held-out trajectories. Baseline regression here means exact policy-action passthrough, not equality to an older-world trajectory.",
        "",
        "| arm | AoI | worst AoI | p95 AoI | CAM | worst CAM | payload | reward | power mW | apply rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        rows = [row for row in cell_rows if row["arm"] == arm]
        mean = lambda key: float(np.mean([float(row[key]) for row in rows]))
        lines.append(
            f"| {arm} | {mean('mean_aoi_ms'):.3f} | {mean('worst_agent_aoi_ms'):.3f} "
            f"| {mean('p95_aoi_ms'):.3f} | {mean('mean_binary_cam'):.4f} "
            f"| {mean('worst_agent_binary_cam'):.4f} | {mean('mean_payload_completion'):.4f} "
            f"| {mean('mean_reward_combined'):.4f} | {mean('mean_executed_power_mw'):.3f} "
            f"| {mean('oracle_applied_fraction'):.4f} |"
        )
    lines.extend([
        "",
        "## Paired joint-RB minus baseline",
        "",
        f"- AoI: {paired['mean_delta_mean_aoi_ms']:+.3f} ms; wins {paired['aoi_improved_seeds']}/6.",
        f"- Strict binary CAM: {paired['mean_delta_mean_binary_cam']:+.4f}; non-worse {paired['cam_not_worse_seeds']}/6.",
        f"- Payload: {paired['mean_delta_mean_payload_completion']:+.4f}; non-worse {paired['payload_not_worse_seeds']}/6.",
        "- Source reward deltas (global/task1/task2/combined): "
        f"{paired['mean_delta_mean_reward_global']:+.4f}/"
        f"{paired['mean_delta_mean_reward_task1']:+.4f}/"
        f"{paired['mean_delta_mean_reward_task2']:+.4f}/"
        f"{paired['mean_delta_mean_reward_combined']:+.4f}.",
        f"- Actual linear power: {paired['mean_delta_mean_executed_power_mw']:+.3f} mW.",
        f"- AoI>50 fraction: {paired['mean_delta_aoi_gt50_fraction']:+.4f}; cap fraction: {paired['mean_delta_aoi_at_cap_fraction']:+.4f}.",
        "",
        "The oracle changes only current RBs and applies only for a strict instantaneous V2I-success gain.",
        "It is centralized diagnostic control, not a learned or decentralized algorithm.",
        "Subsequent observations differ between arms, so later policy-selected mode and power need not match across trajectories.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(result_root: Path, output_root: Path) -> dict:
    result_root = result_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    episode_rows, stream_rows, cell_rows, validation_rows, manifest = [], [], [], [], []
    required = {
        "aoi_ms", "success", "remaining_demand", "rb", "mode", "power_dbm",
        "policy_action_normalized", "action_normalized", "policy_rb", "executed_rb",
        "executed_power_dbm", "reward_global", "reward_task1", "reward_task2",
        "reward_combined", "reset_event", "rb_changed", "oracle_evaluated",
        "oracle_original_v2i_success_count", "oracle_best_v2i_success_count",
        "oracle_strict_improvement_applied",
    }
    for arm in ARMS:
        for seed in TRAINING_SEEDS:
            run = _run_name(seed)
            eval_dir = result_root / "evaluations" / run / rb_intervention_eval_id(arm)
            complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
            expected = {
                "status": "complete", "algorithm": "mappo", "mappo_variant": "tdec",
                "intervention_arm": arm, "training_seed": seed,
                "eval_seeds": list(DEFAULT_EVAL_SEEDS), "eval_episodes": 100,
                "eval_warmup_episodes": 5, "mappo_eval_mode": "stochastic",
                "policy_parameters_unchanged": True,
            }
            for key, value in expected.items():
                if complete.get(key) != value:
                    raise ValueError(f"{eval_dir}: {key}={complete.get(key)!r}, expected {value!r}")
            with np.load(eval_dir / "metrics.npz", allow_pickle=False) as data:
                missing = required - set(data.files)
                if missing:
                    raise ValueError(f"{eval_dir}: missing arrays {sorted(missing)}")
                arrays = {key: np.asarray(data[key]) for key in data.files}
            if arrays["aoi_ms"].shape != (6, 100, 100, 5):
                raise ValueError(f"{eval_dir}: unexpected trajectory shape")
            validation = validate_actions(arrays, complete, arm)
            if not validation["passed"]:
                raise ValueError(f"{eval_dir}: action validation failed: {validation}")
            validation_rows.append(validation)
            episode_rows.extend(_episode_agent_rows(arrays, arm, seed, float(complete["cam_bits"])))
            stream_rows.extend(_stream_rows(arrays, arm, seed, float(complete["cam_bits"])))
            cell_rows.append(_cell_row(arrays, arm, seed, float(complete["cam_bits"])))
            manifest.append({
                "arm": arm,
                "training_seed": seed,
                "eval_dir": str(eval_dir),
                "status": complete["status"],
                "source_config_hash": complete.get("source_config_hash"),
                "reproduction_git_commit": complete.get("reproduction_git_commit"),
            })
    paired_rows, paired_summary = _paired_rows(cell_rows)
    _write_csv(output_root / "per_episode_agent.csv", episode_rows)
    _write_csv(output_root / "per_eval_seed_agent.csv", stream_rows)
    _write_csv(output_root / "per_training_seed.csv", cell_rows)
    _write_csv(output_root / "paired_seed_deltas.csv", paired_rows)
    _write_csv(output_root / "action_validation.csv", validation_rows)
    _write_csv(output_root / "completion_manifest.csv", manifest)
    (output_root / "paired_summary.json").write_text(
        json.dumps(paired_summary, indent=2) + "\n", encoding="utf-8"
    )
    _report(output_root / "rb_intervention.md", cell_rows, paired_summary)
    result = {
        "status": "PASS",
        "completed_cells": len(cell_rows),
        "arms": list(ARMS),
        "training_seeds": list(TRAINING_SEEDS),
        "eval_seeds": list(DEFAULT_EVAL_SEEDS),
        "per_episode_agent_rows": len(episode_rows),
        "per_eval_seed_agent_rows": len(stream_rows),
        "per_training_seed_rows": len(cell_rows),
        "paired_seed_rows": len(paired_rows),
        "action_validation_rows": len(validation_rows),
        "paired_summary": paired_summary,
        "interpretation_limit": (
            "centralized current-slot RB diagnostic; closed-loop trajectories may change later mode and power"
        ),
    }
    (output_root / "rb_intervention_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.result_root, args.output_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
