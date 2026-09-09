"""Summarize the MAPPO frozen-policy feasibility and reward-ledger audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.audit_mappo_service_power import power_mw
from analysis.evaluate_mappo_feasibility_reward import (
    ARMS,
    REFERENCE_TRAJECTORY_ARRAYS,
    feasibility_eval_id,
)
from analysis.evaluate_mappo_power_intervention import DEFAULT_EVAL_SEEDS, intervention_eval_id
from analysis.summarize_mappo_power_intervention import interval_statistics


SEEDS = (8, 9, 10, 11, 12, 13)
INTERVENTIONS = ARMS[1:]
SCENARIO = "p05_n04_g25"
COMMON_TRAJECTORY_ARRAYS = REFERENCE_TRAJECTORY_ARRAYS
FEASIBILITY_LAYERS = {
    "selected": "selected_feasible",
    "best_current_interference": "best_current_interference_feasible",
    "noise_only_best": "noise_only_best_feasible",
}


def _run_name(seed: int) -> str:
    return f"mappo_tdec_ab_tdec_{SCENARIO}_seed{seed:02d}"


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


def _safe_ratio(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else ""


def lack_of_opportunity_statistics(opportunity: np.ndarray) -> dict:
    """Summarize false runs in one continuous scored world trajectory."""

    values = np.asarray(opportunity, dtype=bool).reshape(-1)
    lack = ~values
    padded = np.concatenate(([False], lack, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = (ends - starts).astype(np.int64)
    left = starts == 0
    right = ends == values.size
    complete = lengths[~(left | right)]
    long = complete[complete > 100]
    return {
        "complete_lack_intervals": complete,
        "complete_lack_interval_count": int(complete.size),
        "complete_lack_interval_slots": int(complete.sum()),
        "complete_lack_interval_median_slots": float(np.median(complete)) if complete.size else "",
        "complete_lack_interval_p90_slots": float(np.quantile(complete, 0.90)) if complete.size else "",
        "complete_lack_interval_max_slots": int(complete.max()) if complete.size else "",
        "complete_lack_interval_gt100_count": int(long.size),
        "complete_lack_interval_gt100_duration_fraction": (
            float(long.sum() / complete.sum()) if complete.size and complete.sum() else ""
        ),
        "left_censored_lack_interval_count": int(left.sum()),
        "left_censored_lack_interval_slots": int(lengths[left].sum()),
        "right_censored_lack_interval_count": int(right.sum()),
        "right_censored_lack_interval_slots": int(lengths[right].sum()),
        "entire_world_lacks_opportunity": int(values.size > 0 and not values.any()),
    }


def _discounted_return(values: np.ndarray, gamma: float) -> np.ndarray:
    """Discount within each scored episode, never across episode boundaries."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (3, 4):
        raise ValueError("rewards must have [eval,episode,slot,(agent)] shape")
    weights = np.power(float(gamma), np.arange(values.shape[2], dtype=np.float64))
    shape = (1, 1, values.shape[2]) + ((1,) if values.ndim == 4 else ())
    return np.sum(values * weights.reshape(shape), axis=2)


def _mean_mask(values: np.ndarray, mask: np.ndarray):
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    return float(selected.mean()) if selected.size else ""


def validate_physical_reconstruction(
    arrays: Mapping[str, np.ndarray], complete: Mapping, arm: str, training_seed: int
) -> dict:
    """Recompute required power, actual V2I rate, and combined reward from logs."""

    bandwidth = float(complete["bandwidth_hz"])
    slot_seconds = float(complete["slot_ms"]) / 1000.0
    gamma = float(np.exp2(float(complete["v2i_min_bits_per_step"]) / (bandwidth * slot_seconds)) - 1.0)
    if not np.isclose(gamma, float(complete["gamma_required"]), rtol=0.0, atol=1e-12):
        raise ValueError("recorded gamma_required is inconsistent with the source configuration")
    channel = arrays["v2i_channel_loss_db"].astype(np.float64)
    interference = arrays["v2i_interference_plus_noise_linear"].astype(np.float64)
    gain = float(complete["veh_antenna_gain_db"]) + float(complete["bs_antenna_gain_db"])
    noise_figure = float(complete["bs_noise_figure_db"])
    required_all = 10.0 * np.log10(gamma * interference) + channel - gain + noise_figure
    rb = arrays["rb"].astype(np.int64)
    selected_required = np.take_along_axis(required_all, rb[..., None], axis=-1)[..., 0]
    required_noise = (
        10.0 * np.log10(gamma * 10.0 ** (float(complete["thermal_noise_dbm"]) / 10.0))
        + channel
        - gain
        + noise_figure
    )
    selected_loss = np.take_along_axis(channel, rb[..., None], axis=-1)[..., 0]
    selected_interference = np.take_along_axis(interference, rb[..., None], axis=-1)[..., 0]
    signal = 10.0 ** (
        (arrays["executed_power_dbm"].astype(np.float64) - selected_loss + gain - noise_figure) / 10.0
    )
    signal = np.where(arrays["mode"] == 0, signal, 0.0)
    rate = np.log2(1.0 + signal / selected_interference) * slot_seconds * bandwidth
    combined = (
        arrays["reward_task1"].astype(np.float64)
        + arrays["reward_task2"].astype(np.float64)
        + float(complete["global_actor_weight"]) * arrays["reward_global"].astype(np.float64)[..., None]
    )
    checks = {
        "required_power_selected_max_abs_error": float(
            np.max(np.abs(selected_required - arrays["required_power_selected_dbm"]))
        ),
        "required_power_best_current_max_abs_error": float(
            np.max(np.abs(required_all.min(axis=-1) - arrays["required_power_best_current_interference_dbm"]))
        ),
        "required_power_noise_only_max_abs_error": float(
            np.max(np.abs(required_noise.min(axis=-1) - arrays["required_power_noise_only_best_dbm"]))
        ),
        "v2i_rate_max_abs_error": float(np.max(np.abs(rate - arrays["v2i_rate"]))),
        "combined_reward_max_abs_error": float(np.max(np.abs(combined - arrays["reward_combined"]))),
    }
    if max(
        checks["required_power_selected_max_abs_error"],
        checks["required_power_best_current_max_abs_error"],
        checks["required_power_noise_only_max_abs_error"],
    ) > 2e-4:
        raise ValueError("required-power reconstruction does not match recorded metrics")
    if not np.allclose(rate, arrays["v2i_rate"], rtol=1e-5, atol=1e-3):
        raise ValueError("V2I rate reconstruction does not match recorded metrics")
    if not np.allclose(combined, arrays["reward_combined"], rtol=1e-5, atol=2e-6):
        raise ValueError("combined reward repeats or omits a reward component")
    return {"arm": arm, "training_seed": training_seed, "gamma_required": gamma, **checks}


def _stream_row(
    arrays: Mapping[str, np.ndarray],
    eval_index: int,
    agent: int,
    arm: str,
    training_seed: int,
    cam_bits: float,
    gamma: float,
) -> dict:
    slicer = (eval_index, slice(None), slice(None), agent)
    aoi = arrays["aoi_ms"][slicer].astype(np.float64)
    mode = arrays["mode"][slicer].astype(np.int64)
    attempts = mode == 0
    resets = arrays["reset_event"][slicer].astype(bool)
    success = arrays["success"][eval_index, :, -1, agent].astype(np.float64)
    remaining = arrays["remaining_demand"][eval_index, :, -1, agent].astype(np.float64)
    executed_power = arrays["executed_power_dbm"][slicer].astype(np.float64)
    policy_power = arrays["policy_power_dbm"][slicer].astype(np.float64)
    rb = arrays["rb"][slicer].astype(np.int64)
    v2i_interference = np.take_along_axis(
        arrays["v2i_interference_plus_noise_linear"][eval_index, :, :, agent],
        rb[..., None],
        axis=-1,
    )[..., 0]
    selected_mode_interference = np.power(
        10.0, arrays["selected_interference_db"][slicer].astype(np.float64) / 10.0
    )
    selected = arrays["selected_feasible"][slicer].astype(bool)
    best_current = arrays["best_current_interference_feasible"][slicer].astype(bool)
    noise_best = arrays["noise_only_best_feasible"][slicer].astype(bool)
    update_stats = interval_statistics(aoi)
    complete_update_intervals = update_stats["complete_intervals"]
    long_update_intervals = complete_update_intervals[complete_update_intervals > 100]
    discounted_global = _discounted_return(arrays["reward_global"][eval_index : eval_index + 1], gamma)[0]
    discounted_task1 = _discounted_return(arrays["reward_task1"][eval_index : eval_index + 1], gamma)[0, :, agent]
    discounted_task2 = _discounted_return(arrays["reward_task2"][eval_index : eval_index + 1], gamma)[0, :, agent]
    discounted_combined = _discounted_return(arrays["reward_combined"][eval_index : eval_index + 1], gamma)[0, :, agent]
    attempt_count, reset_count = int(attempts.sum()), int(resets.sum())
    return {
        "arm": arm,
        "training_seed": training_seed,
        "eval_seed": DEFAULT_EVAL_SEEDS[eval_index],
        "agent": agent,
        "slot_count": int(aoi.size),
        "mean_aoi_ms": float(aoi.mean()),
        "binary_cam": float(success.mean()),
        "payload_completion": float(np.clip(1.0 - remaining / float(cam_bits), 0.0, 1.0).mean()),
        "mean_policy_power_mw": float(power_mw(policy_power).mean()),
        "mean_executed_power_mw": float(power_mw(executed_power).mean()),
        "mean_v2i_executed_power_mw": _mean_mask(power_mw(executed_power), attempts),
        "mean_v2v_executed_power_mw": _mean_mask(power_mw(executed_power), ~attempts),
        "mean_v2i_selected_interference_linear": _mean_mask(v2i_interference, attempts),
        "mean_v2v_selected_interference_linear": _mean_mask(selected_mode_interference, ~attempts),
        "power_at_boundary_fraction": float(arrays["power_at_boundary"][slicer].mean()),
        "power_clipped_low_fraction": float(arrays["power_clipped_low"][slicer].mean()),
        "power_clipped_high_fraction": float(arrays["power_clipped_high"][slicer].mean()),
        "v2i_attempt_count": attempt_count,
        "aoi_reset_count": reset_count,
        "zero_v2i_attempt_denominator": int(attempt_count == 0),
        "v2i_attempt_rate": float(attempt_count / aoi.size),
        "v2i_success_given_attempt": _safe_ratio(reset_count, attempt_count),
        "aoi_reset_rate": float(reset_count / aoi.size),
        "aoi_gt50_fraction": float((aoi > 50.0).mean()),
        "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, rtol=0.0, atol=1e-6).mean()),
        "selected_infeasible_fraction": float((~selected).mean()),
        "best_current_interference_infeasible_fraction": float((~best_current).mean()),
        "noise_only_best_infeasible_fraction": float((~noise_best).mean()),
        "selected_feasible_but_not_reset_fraction": float((selected & ~resets).mean()),
        "best_current_feasible_but_not_reset_fraction": float((best_current & ~resets).mean()),
        "noise_only_feasible_but_not_reset_fraction": float((noise_best & ~resets).mean()),
        "selected_feasible_attempt_failure_count": int((selected & attempts & ~resets).sum()),
        "mean_required_power_selected_dbm": float(arrays["required_power_selected_dbm"][slicer].mean()),
        "mean_required_power_best_current_interference_dbm": float(
            arrays["required_power_best_current_interference_dbm"][slicer].mean()
        ),
        "mean_required_power_noise_only_best_dbm": float(arrays["required_power_noise_only_best_dbm"][slicer].mean()),
        "mean_best_current_interference_rate_at_power_max": float(
            arrays["best_current_interference_rate_at_power_max"][slicer].mean()
        ),
        "mean_noise_only_best_rate_at_power_max": float(
            arrays["noise_only_best_rate_at_power_max"][slicer].mean()
        ),
        "mean_leader_bs_distance_m": float(arrays["leader_bs_distance_m"][slicer].mean()),
        "mean_v2i_rate_bits_per_step": _mean_mask(arrays["v2i_rate"][slicer], attempts),
        "mean_reward_global": float(arrays["reward_global"][eval_index].mean()),
        "mean_reward_task1": float(arrays["reward_task1"][slicer].mean()),
        "mean_reward_task2": float(arrays["reward_task2"][slicer].mean()),
        "mean_reward_combined": float(arrays["reward_combined"][slicer].mean()),
        "mean_reward_remaining_demand_term": float(arrays["reward_remaining_demand_term"][slicer].mean()),
        "mean_reward_aoi_term": float(arrays["reward_aoi_term"][slicer].mean()),
        "mean_reward_v2i_revenue_term": float(arrays["reward_v2i_revenue_term"][slicer].mean()),
        "mean_reward_power_penalty": float(arrays["reward_power_penalty"][slicer].mean()),
        "mean_discounted_global_return": float(discounted_global.mean()),
        "mean_discounted_task1_return": float(discounted_task1.mean()),
        "mean_discounted_task2_return": float(discounted_task2.mean()),
        "mean_discounted_combined_return": float(discounted_combined.mean()),
        "complete_interval_slots": int(complete_update_intervals.sum()),
        "complete_interval_gt100_slots": int(long_update_intervals.sum()),
        **{key: value for key, value in update_stats.items() if key != "complete_intervals"},
    }


def _opportunity_rows(
    arrays: Mapping[str, np.ndarray], arm: str, training_seed: int
) -> list[dict]:
    rows = []
    for eval_index, eval_seed in enumerate(DEFAULT_EVAL_SEEDS):
        for agent in range(arrays["aoi_ms"].shape[-1]):
            for layer, key in FEASIBILITY_LAYERS.items():
                stats = lack_of_opportunity_statistics(arrays[key][eval_index, :, :, agent])
                rows.append({
                    "arm": arm,
                    "training_seed": training_seed,
                    "eval_seed": eval_seed,
                    "agent": agent,
                    "feasibility_layer": layer,
                    **{name: value for name, value in stats.items() if name != "complete_lack_intervals"},
                })
    return rows


def _episode_rows(
    arrays: Mapping[str, np.ndarray], arm: str, training_seed: int, cam_bits: float, gamma: float
) -> list[dict]:
    weights = np.power(float(gamma), np.arange(arrays["aoi_ms"].shape[2], dtype=np.float64))
    rows = []
    for eval_index, eval_seed in enumerate(DEFAULT_EVAL_SEEDS):
        for episode in range(arrays["aoi_ms"].shape[1]):
            aoi = arrays["aoi_ms"][eval_index, episode]
            endpoint_success = arrays["success"][eval_index, episode, -1]
            endpoint_remaining = arrays["remaining_demand"][eval_index, episode, -1]
            combined = arrays["reward_combined"][eval_index, episode]
            task1 = arrays["reward_task1"][eval_index, episode]
            task2 = arrays["reward_task2"][eval_index, episode]
            global_reward = arrays["reward_global"][eval_index, episode]
            rows.append({
                "arm": arm,
                "training_seed": training_seed,
                "eval_seed": eval_seed,
                "scored_episode": episode,
                "mean_aoi_ms": float(aoi.mean()),
                "worst_agent_mean_aoi_ms": float(aoi.mean(axis=0).max()),
                "mean_binary_cam": float(endpoint_success.mean()),
                "mean_payload_completion": float(
                    np.clip(1.0 - endpoint_remaining / float(cam_bits), 0.0, 1.0).mean()
                ),
                "mean_reward_combined": float(combined.mean()),
                "mean_reward_global": float(global_reward.mean()),
                "mean_reward_task1": float(task1.mean()),
                "mean_reward_task2": float(task2.mean()),
                "mean_discounted_combined_return": float((combined * weights[:, None]).sum(axis=0).mean()),
                "discounted_global_return": float((global_reward * weights).sum()),
                "mean_discounted_task1_return": float((task1 * weights[:, None]).sum(axis=0).mean()),
                "mean_discounted_task2_return": float((task2 * weights[:, None]).sum(axis=0).mean()),
                "selected_infeasible_fraction": float((~arrays["selected_feasible"][eval_index, episode]).mean()),
                "best_current_interference_infeasible_fraction": float(
                    (~arrays["best_current_interference_feasible"][eval_index, episode]).mean()
                ),
                "noise_only_best_infeasible_fraction": float(
                    (~arrays["noise_only_best_feasible"][eval_index, episode]).mean()
                ),
            })
    return rows


def _cell_summary(rows: Sequence[Mapping], arm: str, seed: int) -> dict:
    selected = [row for row in rows if row["arm"] == arm and int(row["training_seed"]) == seed]
    if len(selected) != 30:
        raise ValueError(f"expected 30 world-agent rows for {arm}/seed{seed}, found {len(selected)}")
    agent_rows = []
    for agent in range(5):
        subset = [row for row in selected if int(row["agent"]) == agent]
        agent_rows.append({
            "aoi": float(np.mean([row["mean_aoi_ms"] for row in subset])),
            "cam": float(np.mean([row["binary_cam"] for row in subset])),
            "payload": float(np.mean([row["payload_completion"] for row in subset])),
        })
    sum_keys = (
        "slot_count", "v2i_attempt_count", "aoi_reset_count", "complete_interval_count",
        "complete_interval_slots", "complete_interval_gt100_count", "complete_interval_gt100_slots",
        "censored_segment_count", "censored_segment_slots", "both_censored_stream",
    )
    mean_keys = (
        "mean_policy_power_mw", "mean_executed_power_mw", "mean_v2i_executed_power_mw",
        "mean_v2v_executed_power_mw", "power_at_boundary_fraction", "power_clipped_low_fraction",
        "mean_v2i_selected_interference_linear", "mean_v2v_selected_interference_linear",
        "power_clipped_high_fraction", "aoi_gt50_fraction", "aoi_at_cap_fraction",
        "selected_infeasible_fraction", "best_current_interference_infeasible_fraction",
        "noise_only_best_infeasible_fraction", "selected_feasible_but_not_reset_fraction",
        "best_current_feasible_but_not_reset_fraction", "noise_only_feasible_but_not_reset_fraction",
        "mean_required_power_selected_dbm", "mean_required_power_best_current_interference_dbm",
        "mean_required_power_noise_only_best_dbm", "mean_leader_bs_distance_m", "mean_reward_global",
        "mean_best_current_interference_rate_at_power_max", "mean_noise_only_best_rate_at_power_max",
        "mean_reward_task1", "mean_reward_task2", "mean_reward_combined",
        "mean_reward_remaining_demand_term", "mean_reward_aoi_term", "mean_reward_v2i_revenue_term",
        "mean_reward_power_penalty", "mean_discounted_global_return", "mean_discounted_task1_return",
        "mean_discounted_task2_return", "mean_discounted_combined_return",
    )
    result = {
        "arm": arm,
        "training_seed": seed,
        "mean_aoi_ms": float(np.mean([row["aoi"] for row in agent_rows])),
        "worst_agent_aoi_ms": float(np.max([row["aoi"] for row in agent_rows])),
        "mean_binary_cam": float(np.mean([row["cam"] for row in agent_rows])),
        "worst_agent_binary_cam": float(np.min([row["cam"] for row in agent_rows])),
        "mean_payload_completion": float(np.mean([row["payload"] for row in agent_rows])),
        "worst_agent_payload_completion": float(np.min([row["payload"] for row in agent_rows])),
    }
    for key in sum_keys:
        result[key] = int(sum(int(row[key]) for row in selected))
    for key in mean_keys:
        values = [float(row[key]) for row in selected if row[key] != ""]
        result[key] = float(np.mean(values)) if values else ""
    result["v2i_attempt_rate"] = float(result["v2i_attempt_count"] / result["slot_count"])
    result["v2i_success_given_attempt"] = _safe_ratio(result["aoi_reset_count"], result["v2i_attempt_count"])
    result["zero_v2i_attempt_stream_count"] = sum(int(row["zero_v2i_attempt_denominator"]) for row in selected)
    result["aoi_reset_rate"] = float(result["aoi_reset_count"] / result["slot_count"])
    result["complete_interval_gt100_duration_fraction"] = (
        float(result["complete_interval_gt100_slots"] / result["complete_interval_slots"])
        if result["complete_interval_slots"] else ""
    )
    return result


def _paired_rows(cell_rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    metrics = tuple(key for key in cell_rows[0] if key not in {"arm", "training_seed"})
    rows = []
    for arm in INTERVENTIONS:
        for seed in SEEDS:
            baseline = next(row for row in cell_rows if row["arm"] == "baseline" and row["training_seed"] == seed)
            intervention = next(row for row in cell_rows if row["arm"] == arm and row["training_seed"] == seed)
            row = {"arm": arm, "training_seed": seed}
            for key in metrics:
                row[f"delta_{key}"] = (
                    float(intervention[key]) - float(baseline[key])
                    if intervention[key] != "" and baseline[key] != "" else ""
                )
            rows.append(row)
    summaries = []
    for arm in INTERVENTIONS:
        selected = [row for row in rows if row["arm"] == arm]
        summary = {"arm": arm, "paired_seed_count": len(selected)}
        for key in metrics:
            values = np.asarray([row[f"delta_{key}"] for row in selected if row[f"delta_{key}"] != ""], dtype=np.float64)
            summary[f"mean_delta_{key}"] = float(values.mean()) if values.size else ""
            summary[f"sd_delta_{key}"] = float(values.std(ddof=1)) if values.size > 1 else ""
        summary["aoi_improved_seed_count"] = sum(row["delta_mean_aoi_ms"] < 0 for row in selected)
        summary["cam_not_worse_seed_count"] = sum(row["delta_mean_binary_cam"] >= 0 for row in selected)
        summary["reward_improved_seed_count"] = sum(row["delta_mean_reward_combined"] > 0 for row in selected)
        summaries.append(summary)
    return rows, summaries


def _paired_stream_rows(stream_rows: Sequence[Mapping]) -> list[dict]:
    """Pair every held-out world and agent within each training seed."""

    metrics = (
        "mean_aoi_ms", "binary_cam", "payload_completion", "mean_executed_power_mw",
        "v2i_attempt_rate", "v2i_success_given_attempt", "aoi_reset_rate",
        "aoi_gt50_fraction", "aoi_at_cap_fraction", "selected_infeasible_fraction",
        "best_current_interference_infeasible_fraction", "noise_only_best_infeasible_fraction",
        "selected_feasible_but_not_reset_fraction", "mean_reward_global", "mean_reward_task1",
        "mean_reward_task2", "mean_reward_combined", "mean_discounted_combined_return",
        "complete_interval_gt100_count", "complete_interval_gt100_duration_fraction",
    )
    indexed = {
        (row["arm"], int(row["training_seed"]), int(row["eval_seed"]), int(row["agent"])): row
        for row in stream_rows
    }
    rows = []
    for arm in INTERVENTIONS:
        for seed in SEEDS:
            for eval_seed in DEFAULT_EVAL_SEEDS:
                for agent in range(5):
                    baseline = indexed[("baseline", seed, eval_seed, agent)]
                    intervention = indexed[(arm, seed, eval_seed, agent)]
                    row = {
                        "arm": arm,
                        "training_seed": seed,
                        "eval_seed": eval_seed,
                        "agent": agent,
                    }
                    for key in metrics:
                        row[f"delta_{key}"] = (
                            float(intervention[key]) - float(baseline[key])
                            if intervention[key] != "" and baseline[key] != "" else ""
                        )
                    rows.append(row)
    return rows


def _equivalence_rows(old_root: Path, result_root: Path) -> list[dict]:
    rows = []
    for arm in ("baseline", "v2i_plus3db"):
        for seed in SEEDS:
            run = _run_name(seed)
            old = old_root / "evaluations" / run / intervention_eval_id(arm) / "metrics.npz"
            new = result_root / "evaluations" / run / feasibility_eval_id(arm) / "metrics.npz"
            row = {"arm": arm, "training_seed": seed, "old_metrics": str(old), "new_metrics": str(new)}
            all_exact = True
            with np.load(old, allow_pickle=False) as old_data, np.load(new, allow_pickle=False) as new_data:
                for key in COMMON_TRAJECTORY_ARRAYS:
                    exact = bool(np.array_equal(old_data[key], new_data[key]))
                    row[f"{key}_exact"] = exact
                    all_exact = all_exact and exact
            row["all_common_arrays_exact"] = all_exact
            rows.append(row)
    return rows


def _write_report(path: Path, cells: Sequence[Mapping], paired: Sequence[Mapping], special: Sequence[Mapping], exact: bool) -> None:
    lines = [
        "# MAPPO frozen-policy feasibility and reward audit",
        "",
        f"Baseline/+3 reproduction against mode-power-intervention-v1: **{'EXACT' if exact else 'REVIEW REQUIRED'}**.",
        "",
        "| arm | AoI | worst AoI | binary CAM | payload | power mW | rho | q | selected infeasible | best-RB infeasible | noise-only infeasible | reward | >100 intervals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        rows = [row for row in cells if row["arm"] == arm]
        mean = lambda key: float(np.mean([float(row[key]) for row in rows if row[key] != ""]))
        lines.append(
            f"| {arm} | {mean('mean_aoi_ms'):.3f} | {mean('worst_agent_aoi_ms'):.3f} "
            f"| {mean('mean_binary_cam'):.4f} | {mean('mean_payload_completion'):.4f} "
            f"| {mean('mean_executed_power_mw'):.3f} | {mean('v2i_attempt_rate'):.4f} "
            f"| {mean('v2i_success_given_attempt'):.4f} | {mean('selected_infeasible_fraction'):.4f} "
            f"| {mean('best_current_interference_infeasible_fraction'):.4f} "
            f"| {mean('noise_only_best_infeasible_fraction'):.4f} | {mean('mean_reward_combined'):.4f} "
            f"| {mean('complete_interval_gt100_count'):.1f} |"
        )
    lines.extend([
        "",
        "## Paired mean changes from baseline",
        "",
        "| arm | delta AoI | delta CAM | delta payload | delta power mW | delta q | delta noise-only infeasible | delta reward | AoI wins | CAM non-worse | reward wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in paired:
        lines.append(
            f"| {row['arm']} | {row['mean_delta_mean_aoi_ms']:.3f} | {row['mean_delta_mean_binary_cam']:.4f} "
            f"| {row['mean_delta_mean_payload_completion']:.4f} | {row['mean_delta_mean_executed_power_mw']:.3f} "
            f"| {row['mean_delta_v2i_success_given_attempt']:.4f} "
            f"| {row['mean_delta_noise_only_best_infeasible_fraction']:.4f} "
            f"| {row['mean_delta_mean_reward_combined']:.4f} | {row['aoi_improved_seed_count']}/6 "
            f"| {row['cam_not_worse_seed_count']}/6 | {row['reward_improved_seed_count']}/6 |"
        )
    lines.extend([
        "",
        "## Prespecified training seed 9 / eval seed 205 / agent 2",
        "",
        "| arm | AoI | CAM | payload | rho | q | selected infeasible | best-RB infeasible | noise-only infeasible | reward |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in special:
        q = "NA" if row["v2i_success_given_attempt"] == "" else f"{row['v2i_success_given_attempt']:.4f}"
        lines.append(
            f"| {row['arm']} | {row['mean_aoi_ms']:.3f} | {row['binary_cam']:.4f} "
            f"| {row['payload_completion']:.4f} | {row['v2i_attempt_rate']:.4f} | {q} "
            f"| {row['selected_infeasible_fraction']:.4f} "
            f"| {row['best_current_interference_infeasible_fraction']:.4f} "
            f"| {row['noise_only_best_infeasible_fraction']:.4f} | {row['mean_reward_combined']:.4f} |"
        )
    lines.extend([
        "",
        "Required power above 30 dBm under current interference is not a claim that every algorithm is infeasible.",
        "A higher frozen-policy reward is not evidence of a PPO defect or proof that a trained intervention will improve.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(source_root: Path, old_result_root: Path, result_root: Path, output_root: Path) -> dict:
    source_root = source_root.expanduser().resolve()
    old_result_root = old_result_root.expanduser().resolve()
    result_root = result_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stream_rows, episode_rows, opportunity_rows, completion_rows, reconstruction_rows = [], [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            run = _run_name(seed)
            eval_dir = result_root / "evaluations" / run / feasibility_eval_id(arm)
            complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
            checks = {
                "status": "complete", "algorithm": "mappo", "mappo_variant": "tdec",
                "intervention_arm": arm, "mappo_eval_mode": "stochastic", "training_seed": seed,
                "eval_seeds": list(DEFAULT_EVAL_SEEDS), "eval_episodes": 100,
                "eval_warmup_episodes": 5, "policy_parameters_unchanged": True,
            }
            for key, expected in checks.items():
                if complete.get(key) != expected:
                    raise ValueError(f"{eval_dir}: {key}={complete.get(key)!r}, expected {expected!r}")
            with np.load(eval_dir / "metrics.npz", allow_pickle=False) as data:
                arrays = {key: np.asarray(data[key]) for key in data.files}
            required = {
                "aoi_ms", "success", "remaining_demand", "rb", "mode", "power_dbm",
                "action_normalized", "policy_action_normalized", "executed_power_dbm",
                "policy_power_dbm", "power_clipped_low", "power_clipped_high", "power_at_boundary",
                "reset_event", "v2i_rate", "selected_interference_db", "required_power_selected_dbm",
                "required_power_best_current_interference_dbm", "required_power_noise_only_best_dbm",
                "selected_feasible", "best_current_interference_feasible", "noise_only_best_feasible",
                "best_current_interference_rate_at_power_max", "noise_only_best_rate_at_power_max",
                "leader_bs_distance_m", "v2i_channel_loss_db",
                "v2i_interference_plus_noise_linear", "reward_global", "reward_task1", "reward_task2",
                "reward_combined", "reward_remaining_demand_term", "reward_aoi_term",
                "reward_v2i_revenue_term", "reward_power_penalty",
            }
            if required - arrays.keys():
                raise ValueError(f"{eval_dir}: missing arrays {sorted(required - arrays.keys())}")
            expected = (6, 100, 100, 5)
            if arrays["aoi_ms"].shape != expected or arrays["reward_global"].shape != expected[:3]:
                raise ValueError(f"{eval_dir}: unexpected metric shape")
            if arrays["v2i_channel_loss_db"].shape != expected + (3,):
                raise ValueError(f"{eval_dir}: unexpected per-RB metric shape")
            if not np.array_equal(
                np.isclose(arrays["aoi_ms"], 1.0, rtol=0.0, atol=1e-6),
                arrays["reset_event"].astype(bool),
            ):
                raise ValueError(f"{eval_dir}: reset mismatch")
            gamma = float(complete["source_gamma"])
            cam_bits = float(complete["cam_bits"])
            reconstruction_rows.append(validate_physical_reconstruction(arrays, complete, arm, seed))
            rows = [
                _stream_row(arrays, eval_index, agent, arm, seed, cam_bits, gamma)
                for eval_index in range(6) for agent in range(5)
            ]
            stream_rows.extend(rows)
            episode_rows.extend(_episode_rows(arrays, arm, seed, cam_bits, gamma))
            opportunity_rows.extend(_opportunity_rows(arrays, arm, seed))
            provenance = _read_json(eval_dir / "provenance.json")
            completion_rows.append({
                "arm": arm, "training_seed": seed, "status": complete["status"],
                "eval_dir": str(eval_dir), "policy": provenance.get("policy"),
                "source_config_hash": complete["source_config_hash"],
                "reproduction_git_commit": provenance.get("reproduction_git_commit"),
            })

    cell_rows = [_cell_summary(stream_rows, arm, seed) for arm in ARMS for seed in SEEDS]
    paired_rows, paired_summary = _paired_rows(cell_rows)
    paired_stream_rows = _paired_stream_rows(stream_rows)
    equivalence = _equivalence_rows(old_result_root, result_root)
    exact = all(row["all_common_arrays_exact"] for row in equivalence)
    special = [
        row for row in stream_rows
        if row["training_seed"] == 9 and row["eval_seed"] == 205 and row["agent"] == 2
    ]
    lowest_q = sorted(
        [row for row in stream_rows if row["arm"] == "baseline"],
        key=lambda row: (
            row["v2i_success_given_attempt"] == "",
            float(row["v2i_success_given_attempt"]) if row["v2i_success_given_attempt"] != "" else 1.0,
        ),
    )[:20]
    _write_csv(output_root / "per_training_seed.csv", cell_rows)
    _write_csv(output_root / "per_eval_seed_agent.csv", stream_rows)
    _write_csv(output_root / "per_episode_summary.csv", episode_rows)
    _write_csv(output_root / "service_opportunity_intervals.csv", opportunity_rows)
    _write_csv(output_root / "paired_seed_deltas.csv", paired_rows)
    _write_csv(output_root / "paired_eval_seed_agent_deltas.csv", paired_stream_rows)
    _write_csv(output_root / "paired_summary.csv", paired_summary)
    _write_csv(output_root / "baseline_plus3_equivalence.csv", equivalence)
    _write_csv(output_root / "special_seed9_eval205_agent2.csv", special)
    _write_csv(output_root / "lowest_q_baseline_streams.csv", lowest_q)
    _write_csv(output_root / "completion_manifest.csv", completion_rows)
    _write_csv(output_root / "physical_reconstruction_checks.csv", reconstruction_rows)
    _write_report(output_root / "feasibility_reward_audit.md", cell_rows, paired_summary, special, exact)
    result = {
        "status": "PASS" if exact else "TRAJECTORY_REVIEW_REQUIRED",
        "source_root": str(source_root),
        "completed_cells": len(cell_rows),
        "training_seed_rows": len(cell_rows),
        "eval_seed_agent_rows": len(stream_rows),
        "per_episode_rows": len(episode_rows),
        "service_opportunity_rows": len(opportunity_rows),
        "paired_eval_seed_agent_rows": len(paired_stream_rows),
        "special_case_rows": len(special),
        "arms": list(ARMS),
        "training_seeds": list(SEEDS),
        "eval_seeds": list(DEFAULT_EVAL_SEEDS),
        "baseline_plus3_common_arrays_exact": exact,
        "equivalence_rows": len(equivalence),
        "physical_reconstruction_rows": len(reconstruction_rows),
        "paired_summary": paired_summary,
    }
    (output_root / "feasibility_reward_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--old-result-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.source_root, args.old_result_root, args.result_root, args.output_root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
