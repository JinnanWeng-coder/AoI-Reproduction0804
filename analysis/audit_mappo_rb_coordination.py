"""Pure post-processing audit of MAPPO RB selection and coordination space."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.evaluate_mappo_feasibility_reward import feasibility_eval_id, required_power_dbm


TRAINING_SEEDS = (8, 9, 10, 11, 12, 13)
EVAL_SEEDS = (201, 202, 203, 204, 205, 206)
SAMPLE_EPISODES = (0, 49, 99)
SAMPLE_SLOTS = (0, 49, 99)
REQUIRED_FIELDS = {
    "aoi_ms",
    "reset_event",
    "rb",
    "mode",
    "executed_power_dbm",
    "v2i_rate",
    "v2i_channel_loss_db",
    "v2i_interference_plus_noise_linear",
    "required_power_selected_dbm",
}


def _run_name(seed: int) -> str:
    return f"mappo_tdec_ab_tdec_p05_n04_g25_seed{int(seed):02d}"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dbm_to_mw(values):
    return np.power(10.0, np.asarray(values, dtype=np.float64) / 10.0)


def _take_selected(values: np.ndarray, rb: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, np.asarray(rb, dtype=np.int64)[..., None], axis=-1)[..., 0]


def physical_parameters(complete: Mapping) -> dict:
    required = (
        "bandwidth_hz",
        "slot_ms",
        "v2i_min_bits_per_step",
        "gamma_required",
        "veh_antenna_gain_db",
        "bs_antenna_gain_db",
        "bs_noise_figure_db",
        "thermal_noise_dbm",
        "power_max_dbm",
    )
    missing = [key for key in required if key not in complete]
    if missing:
        raise KeyError(f"EVAL_COMPLETE is missing physical metadata: {missing}")
    bandwidth = float(complete["bandwidth_hz"])
    slot_seconds = float(complete["slot_ms"]) / 1000.0
    gamma_required = float(
        np.exp2(float(complete["v2i_min_bits_per_step"]) / (bandwidth * slot_seconds)) - 1.0
    )
    if not np.isclose(gamma_required, float(complete["gamma_required"]), rtol=0.0, atol=1e-12):
        raise ValueError("gamma_required does not match source configuration")
    return {
        "bandwidth_hz": bandwidth,
        "slot_seconds": slot_seconds,
        "v2i_min_bits_per_step": float(complete["v2i_min_bits_per_step"]),
        "gamma_required": gamma_required,
        "combined_gain_db": float(complete["veh_antenna_gain_db"])
        + float(complete["bs_antenna_gain_db"]),
        "bs_noise_figure_db": float(complete["bs_noise_figure_db"]),
        "noise_linear": float(10.0 ** (float(complete["thermal_noise_dbm"]) / 10.0)),
        "power_max_dbm": float(complete["power_max_dbm"]),
    }


def counterfactual_v2i_quantities(arrays: Mapping[str, np.ndarray], complete: Mapping) -> dict:
    """Reconstruct selected and unilateral-best V2I quantities for all slots."""

    missing = sorted(REQUIRED_FIELDS - set(arrays))
    if missing:
        raise KeyError(f"metrics.npz is missing required fields: {missing}")
    parameters = physical_parameters(complete)
    channel = np.asarray(arrays["v2i_channel_loss_db"], dtype=np.float64)
    interference = np.asarray(arrays["v2i_interference_plus_noise_linear"], dtype=np.float64)
    rb = np.asarray(arrays["rb"], dtype=np.int64)
    mode = np.asarray(arrays["mode"], dtype=np.int64)
    power = np.asarray(arrays["executed_power_dbm"], dtype=np.float64)
    if channel.shape != interference.shape or channel.shape[:-1] != rb.shape:
        raise ValueError("unexpected V2I channel/interference/RB shapes")
    if channel.shape[-1] != 3 or rb.shape[-1] != 5:
        raise ValueError("RB coordination audit requires five agents and three RBs")
    if np.any((rb < 0) | (rb >= channel.shape[-1])) or np.any(interference <= 0):
        raise ValueError("invalid RB index or non-positive interference")

    required_all = required_power_dbm(
        channel,
        interference,
        parameters["gamma_required"],
        float(complete["veh_antenna_gain_db"]),
        float(complete["bs_antenna_gain_db"]),
        parameters["bs_noise_figure_db"],
    )
    selected_required = _take_selected(required_all, rb)
    best_rb = np.argmin(required_all, axis=-1).astype(np.int64)
    best_required = _take_selected(required_all, best_rb)
    received_signal_all = dbm_to_mw(
        power[..., None]
        - channel
        + parameters["combined_gain_db"]
        - parameters["bs_noise_figure_db"]
    )
    rate_all = (
        np.log2(1.0 + received_signal_all / interference)
        * parameters["slot_seconds"]
        * parameters["bandwidth_hz"]
    )
    selected_rate_hypothetical = _take_selected(rate_all, rb)
    selected_rate = np.where(mode == 0, selected_rate_hypothetical, 0.0)
    best_rate = np.max(rate_all, axis=-1)
    logged_rate = np.asarray(arrays["v2i_rate"], dtype=np.float64)
    if not np.allclose(selected_rate, logged_rate, rtol=1e-5, atol=1e-3):
        raise ValueError("selected-RB V2I rate reconstruction does not match metrics.npz")
    if not np.allclose(
        selected_required,
        np.asarray(arrays["required_power_selected_dbm"], dtype=np.float64),
        rtol=0.0,
        atol=2e-4,
    ):
        raise ValueError("selected required-power reconstruction does not match metrics.npz")
    v2i = mode == 0
    reconstructed_success = v2i & (selected_rate_hypothetical >= parameters["v2i_min_bits_per_step"])
    reset = np.asarray(arrays["reset_event"], dtype=bool)
    ambiguous = np.abs(selected_rate_hypothetical - parameters["v2i_min_bits_per_step"]) <= 1e-3
    if np.any((reconstructed_success != reset) & ~ambiguous):
        raise ValueError("reconstructed V2I success does not match reset_event")
    current_success = reset
    current_failure = v2i & ~current_success
    unilateral_rescue = current_failure & (best_rate >= parameters["v2i_min_bits_per_step"])

    selected_channel = _take_selected(channel, rb)
    best_channel = _take_selected(channel, best_rb)
    selected_interference_db = 10.0 * np.log10(_take_selected(interference, rb))
    best_interference_db = 10.0 * np.log10(_take_selected(interference, best_rb))
    required_improvement = selected_required - best_required
    channel_contribution = selected_channel - best_channel
    interference_contribution = selected_interference_db - best_interference_db
    if not np.allclose(
        required_improvement,
        channel_contribution + interference_contribution,
        rtol=0.0,
        atol=3e-4,
    ):
        raise ValueError("same-pair channel/interference decomposition is inconsistent")
    return {
        **parameters,
        "required_all_dbm": required_all,
        "selected_required_dbm": selected_required,
        "best_required_dbm": best_required,
        "best_rb": best_rb,
        "rate_all": rate_all,
        "selected_rate": selected_rate,
        "best_rate": best_rate,
        "v2i_mask": v2i,
        "current_success": current_success,
        "current_failure": current_failure,
        "unilateral_rescue": unilateral_rescue,
        "selected_minus_best_required_db": required_improvement,
        "channel_loss_contribution_db": channel_contribution,
        "interference_contribution_db": interference_contribution,
    }


def recompute_joint_v2i_rates(
    rb_assignment: Sequence[int],
    mode: np.ndarray,
    power_dbm: np.ndarray,
    channel_loss_db: np.ndarray,
    noise_linear: float,
    combined_gain_db: float,
    noise_figure_db: float,
    slot_seconds: float,
    bandwidth_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute selected V2I rates and interference for one joint RB action."""

    rb = np.asarray(rb_assignment, dtype=np.int64)
    mode = np.asarray(mode, dtype=np.int64)
    power = np.asarray(power_dbm, dtype=np.float64)
    channel = np.asarray(channel_loss_db, dtype=np.float64)
    n_agents, n_rb = channel.shape
    if rb.shape != (n_agents,) or mode.shape != (n_agents,) or power.shape != (n_agents,):
        raise ValueError("joint action arrays have inconsistent agent dimensions")
    if np.any((rb < 0) | (rb >= n_rb)) or noise_linear <= 0:
        raise ValueError("invalid joint RB assignment or noise")
    own_received = dbm_to_mw(
        power - channel[np.arange(n_agents), rb] + float(combined_gain_db) - float(noise_figure_db)
    )
    selected_interference = np.full(n_agents, float(noise_linear), dtype=np.float64)
    for receiver in range(n_agents):
        for transmitter in range(n_agents):
            if transmitter != receiver and rb[transmitter] == rb[receiver]:
                selected_interference[receiver] += own_received[transmitter]
    if np.any(selected_interference <= 0):
        raise ValueError("joint interference must retain positive thermal noise")
    signal = np.where(mode == 0, own_received, 0.0)
    rates = np.log2(1.0 + signal / selected_interference) * float(slot_seconds) * float(bandwidth_hz)
    return rates, selected_interference


def joint_rb_oracle(
    original_rb: Sequence[int],
    mode: np.ndarray,
    power_dbm: np.ndarray,
    channel_loss_db: np.ndarray,
    parameters: Mapping,
    unilateral_rescue_mask: np.ndarray,
) -> dict:
    """Enumerate all 3^5 joint assignments and maximize instantaneous V2I successes."""

    original = tuple(int(value) for value in original_rb)
    candidates = tuple(itertools.product(range(3), repeat=5))
    if original not in candidates:
        raise AssertionError("original RB assignment is absent from oracle enumeration")
    mode = np.asarray(mode, dtype=np.int64)
    unilateral = np.asarray(unilateral_rescue_mask, dtype=bool)
    threshold = float(parameters["v2i_min_bits_per_step"])
    original_rates, _ = recompute_joint_v2i_rates(
        original,
        mode,
        power_dbm,
        channel_loss_db,
        parameters["noise_linear"],
        parameters["combined_gain_db"],
        parameters["bs_noise_figure_db"],
        parameters["slot_seconds"],
        parameters["bandwidth_hz"],
    )
    original_success = (mode == 0) & (original_rates >= threshold)
    best = None
    for candidate in candidates:
        rates, interference = recompute_joint_v2i_rates(
            candidate,
            mode,
            power_dbm,
            channel_loss_db,
            parameters["noise_linear"],
            parameters["combined_gain_db"],
            parameters["bs_noise_figure_db"],
            parameters["slot_seconds"],
            parameters["bandwidth_hz"],
        )
        success = (mode == 0) & (rates >= threshold)
        score = (
            int(success.sum()),
            int((success & unilateral).sum()),
            int((success & original_success).sum()),
        )
        if best is None or score > best[0]:
            best = (score, candidate, rates, interference, success)
    score, assignment, rates, interference, success = best
    if score[0] < int(original_success.sum()):
        raise AssertionError("joint oracle is worse than an enumerated original assignment")
    return {
        "candidate_count": len(candidates),
        "original_assignment_included": True,
        "original_rates": original_rates,
        "original_success": original_success,
        "oracle_assignment": assignment,
        "oracle_rates": rates,
        "oracle_interference": interference,
        "oracle_success": success,
        "original_success_count": int(original_success.sum()),
        "oracle_success_count": int(success.sum()),
        "oracle_additional_success_count": int(success.sum() - original_success.sum()),
        "unilateral_rescue_count": int(unilateral.sum()),
        "oracle_retained_unilateral_count": int((success & unilateral).sum()),
        "original_success_preserved_count": int((success & original_success).sum()),
    }


def _conditional_mean(values: np.ndarray, mask: np.ndarray):
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return float(selected.mean()) if selected.size else ""


def stream_summary(
    arrays: Mapping[str, np.ndarray], quantities: Mapping, eval_index: int, agent: int, training_seed: int
) -> dict:
    selector = (eval_index, slice(None), slice(None), agent)
    mask = quantities["v2i_mask"][selector]
    failures = quantities["current_failure"][selector]
    rescued = quantities["unilateral_rescue"][selector]
    actual = np.asarray(arrays["executed_power_dbm"], dtype=np.float64)[selector]
    selected_required = quantities["selected_required_dbm"][selector]
    best_required = quantities["best_required_dbm"][selector]
    v2i_count = int(mask.sum())
    failure_count = int(failures.sum())
    rescue_count = int(rescued.sum())
    resets = np.asarray(arrays["reset_event"], dtype=bool)[selector]
    reset_count = int(resets.sum())
    selected_margin = actual - selected_required
    best_margin = actual - best_required
    return {
        "training_seed": training_seed,
        "eval_seed": EVAL_SEEDS[eval_index],
        "agent": agent,
        "total_slots": int(mask.size),
        "v2i_slots": v2i_count,
        "v2i_fraction": float(v2i_count / mask.size),
        "mean_aoi_ms": float(np.asarray(arrays["aoi_ms"], dtype=np.float64)[selector].mean()),
        "aoi_gt50_fraction": float(
            (np.asarray(arrays["aoi_ms"], dtype=np.float64)[selector] > 50.0).mean()
        ),
        "reset_count": reset_count,
        "v2i_success_given_attempt": float(reset_count / v2i_count) if v2i_count else "",
        "aoi_reset_rate": float(reset_count / mask.size),
        "current_v2i_failure_count": failure_count,
        "unilateral_rescue_count": rescue_count,
        "unilateral_rescue_fraction_of_v2i_failures": float(rescue_count / failure_count) if failure_count else "",
        "unilateral_rescue_fraction_of_v2i_slots": float(rescue_count / v2i_count) if v2i_count else "",
        "mean_actual_power_dbm": _conditional_mean(actual, mask),
        "mean_selected_required_power_dbm": _conditional_mean(selected_required, mask),
        "mean_best_required_power_dbm": _conditional_mean(best_required, mask),
        "mean_actual_power_mw": _conditional_mean(dbm_to_mw(actual), mask),
        "mean_selected_required_power_mw": _conditional_mean(dbm_to_mw(selected_required), mask),
        "mean_best_required_power_mw": _conditional_mean(dbm_to_mw(best_required), mask),
        "mean_selected_power_margin_db": _conditional_mean(selected_margin, mask),
        "mean_best_power_margin_db": _conditional_mean(best_margin, mask),
        "mean_selected_positive_shortfall_db": _conditional_mean(np.maximum(-selected_margin, 0.0), mask),
        "mean_best_positive_shortfall_db": _conditional_mean(np.maximum(-best_margin, 0.0), mask),
        "mean_selected_minus_best_required_db": _conditional_mean(
            quantities["selected_minus_best_required_db"][selector], mask
        ),
        "mean_channel_loss_contribution_db": _conditional_mean(
            quantities["channel_loss_contribution_db"][selector], mask
        ),
        "mean_interference_contribution_db": _conditional_mean(
            quantities["interference_contribution_db"][selector], mask
        ),
    }


def aggregate_streams(rows: Sequence[Mapping], training_seed: int) -> dict:
    selected = [row for row in rows if int(row["training_seed"]) == int(training_seed)]
    if len(selected) != 30:
        raise ValueError(f"expected 30 eval-world/agent rows for seed {training_seed}")
    v2i_slots = sum(int(row["v2i_slots"]) for row in selected)
    failures = sum(int(row["current_v2i_failure_count"]) for row in selected)
    rescues = sum(int(row["unilateral_rescue_count"]) for row in selected)
    resets = sum(int(row["reset_count"]) for row in selected)
    all_slot_mean_keys = ("mean_aoi_ms", "aoi_gt50_fraction")
    weighted_keys = [
        key for key in selected[0]
        if key.startswith("mean_") and key not in all_slot_mean_keys
    ]
    result = {
        "training_seed": int(training_seed),
        "stream_count": len(selected),
        "total_slots": sum(int(row["total_slots"]) for row in selected),
        "v2i_slots": v2i_slots,
        "current_v2i_failure_count": failures,
        "unilateral_rescue_count": rescues,
        "reset_count": resets,
        "unilateral_rescue_fraction_of_v2i_failures": float(rescues / failures) if failures else "",
        "unilateral_rescue_fraction_of_v2i_slots": float(rescues / v2i_slots) if v2i_slots else "",
    }
    result["v2i_fraction"] = float(v2i_slots / result["total_slots"])
    result["v2i_success_given_attempt"] = float(resets / v2i_slots) if v2i_slots else ""
    result["aoi_reset_rate"] = float(resets / result["total_slots"])
    for key in all_slot_mean_keys:
        result[key] = float(
            sum(float(row[key]) * int(row["total_slots"]) for row in selected) / result["total_slots"]
        )
    for key in weighted_keys:
        available = [row for row in selected if row[key] != "" and int(row["v2i_slots"]) > 0]
        denominator = sum(int(row["v2i_slots"]) for row in available)
        result[key] = (
            float(sum(float(row[key]) * int(row["v2i_slots"]) for row in available) / denominator)
            if denominator else ""
        )
    return result


def equal_seed_summary(seed_rows: Sequence[Mapping]) -> dict:
    if {int(row["training_seed"]) for row in seed_rows} != set(TRAINING_SEEDS):
        raise ValueError("equal-seed summary requires exactly training seeds 8-13")
    result = {"training_seed_count": len(seed_rows), "aggregation": "equal weight per training seed"}
    keys = [key for key in seed_rows[0] if key not in {"training_seed", "stream_count"}]
    for key in keys:
        values = [float(row[key]) for row in seed_rows if row[key] != ""]
        result[f"mean_{key}"] = float(np.mean(values)) if values else None
        result[f"sd_{key}"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return result


def rb_usage_rows(arrays: Mapping[str, np.ndarray], training_seed: int) -> list[dict]:
    rows = []
    rb = np.asarray(arrays["rb"], dtype=np.int64)
    mode = np.asarray(arrays["mode"], dtype=np.int64)
    for eval_index, eval_seed in enumerate(EVAL_SEEDS):
        for agent in range(5):
            values = rb[eval_index, :, :, agent].reshape(-1)
            modes = mode[eval_index, :, :, agent].reshape(-1)
            for scope, mask in (
                ("all", np.ones(values.size, dtype=bool)),
                ("v2i", modes == 0),
                ("v2v", modes == 1),
            ):
                denominator = int(mask.sum())
                for resource_block in range(3):
                    count = int(((values == resource_block) & mask).sum())
                    rows.append({
                        "training_seed": training_seed,
                        "eval_seed": eval_seed,
                        "agent": agent,
                        "mode_scope": scope,
                        "rb": resource_block,
                        "count": count,
                        "denominator": denominator,
                        "fraction": float(count / denominator) if denominator else "",
                    })
    return rows


def _rb_distribution(values: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None
    return np.bincount(values[mask], minlength=3).astype(np.float64) / int(mask.sum())


def pairwise_reuse_rows(arrays: Mapping[str, np.ndarray], training_seed: int) -> list[dict]:
    rows = []
    rb = np.asarray(arrays["rb"], dtype=np.int64)
    mode = np.asarray(arrays["mode"], dtype=np.int64)
    for eval_index, eval_seed in enumerate(EVAL_SEEDS):
        flat_rb = rb[eval_index].reshape(-1, 5)
        flat_mode = mode[eval_index].reshape(-1, 5)
        for first, second in itertools.combinations(range(5), 2):
            categories = {
                "all": np.ones(flat_rb.shape[0], dtype=bool),
                "both_v2i": (flat_mode[:, first] == 0) & (flat_mode[:, second] == 0),
                "both_v2v": (flat_mode[:, first] == 1) & (flat_mode[:, second] == 1),
                "mixed": flat_mode[:, first] != flat_mode[:, second],
            }
            for category, mask in categories.items():
                denominator = int(mask.sum())
                observed = float((flat_rb[mask, first] == flat_rb[mask, second]).mean()) if denominator else ""
                left = _rb_distribution(flat_rb[:, first], mask)
                right = _rb_distribution(flat_rb[:, second], mask)
                reference = float(np.dot(left, right)) if left is not None and right is not None else ""
                rows.append({
                    "training_seed": training_seed,
                    "eval_seed": eval_seed,
                    "agent_i": first,
                    "agent_j": second,
                    "mode_scope": category,
                    "slot_count": denominator,
                    "observed_same_rb_fraction": observed,
                    "independent_marginal_reference": reference,
                    "same_rb_excess_over_reference": (
                        float(observed - reference) if observed != "" and reference != "" else ""
                    ),
                })
    return rows


def joint_occupancy_rows(arrays: Mapping[str, np.ndarray], training_seed: int) -> list[dict]:
    rows = []
    rb = np.asarray(arrays["rb"], dtype=np.int64)
    mode = np.asarray(arrays["mode"], dtype=np.int64)
    for eval_index, eval_seed in enumerate(EVAL_SEEDS):
        flat_rb = rb[eval_index].reshape(-1, 5)
        flat_mode = mode[eval_index].reshape(-1, 5)
        for scope, desired_mode in (("all", None), ("v2i", 0), ("v2v", 1)):
            counts = {}
            for slot_rb, slot_mode in zip(flat_rb, flat_mode):
                selected = slot_rb if desired_mode is None else slot_rb[slot_mode == desired_mode]
                occupancy = tuple(int((selected == resource_block).sum()) for resource_block in range(3))
                counts[occupancy] = counts.get(occupancy, 0) + 1
            denominator = flat_rb.shape[0]
            for occupancy, count in sorted(counts.items()):
                rows.append({
                    "training_seed": training_seed,
                    "eval_seed": eval_seed,
                    "mode_scope": scope,
                    "rb0_count": occupancy[0],
                    "rb1_count": occupancy[1],
                    "rb2_count": occupancy[2],
                    "slot_count": count,
                    "fraction": float(count / denominator),
                    "same_rb_pair_count": int(sum(value * (value - 1) // 2 for value in occupancy)),
                })
    return rows


def oracle_rows(
    arrays: Mapping[str, np.ndarray], quantities: Mapping, training_seed: int
) -> list[dict]:
    rows = []
    for eval_index, eval_seed in enumerate(EVAL_SEEDS):
        for episode in SAMPLE_EPISODES:
            for slot in SAMPLE_SLOTS:
                original_rb = arrays["rb"][eval_index, episode, slot]
                result = joint_rb_oracle(
                    original_rb,
                    arrays["mode"][eval_index, episode, slot],
                    arrays["executed_power_dbm"][eval_index, episode, slot],
                    arrays["v2i_channel_loss_db"][eval_index, episode, slot],
                    quantities,
                    quantities["unilateral_rescue"][eval_index, episode, slot],
                )
                logged_rate = arrays["v2i_rate"][eval_index, episode, slot]
                if not np.allclose(result["original_rates"], logged_rate, rtol=1e-5, atol=1e-3):
                    raise ValueError("oracle original assignment does not reconstruct logged V2I rates")
                unilateral_count = result["unilateral_rescue_count"]
                rows.append({
                    "training_seed": training_seed,
                    "eval_seed": eval_seed,
                    "episode": episode,
                    "slot": slot,
                    "sample_stage": f"ep{episode}_slot{slot}",
                    "candidate_count": result["candidate_count"],
                    "original_assignment_included": result["original_assignment_included"],
                    "original_rb": "|".join(str(value) for value in original_rb),
                    "oracle_rb": "|".join(str(value) for value in result["oracle_assignment"]),
                    "v2i_agent_count": int((arrays["mode"][eval_index, episode, slot] == 0).sum()),
                    "original_success_count": result["original_success_count"],
                    "oracle_success_count": result["oracle_success_count"],
                    "oracle_additional_success_count": result["oracle_additional_success_count"],
                    "unilateral_rescue_count": unilateral_count,
                    "oracle_retained_unilateral_count": result["oracle_retained_unilateral_count"],
                    "retained_unilateral_fraction": (
                        float(result["oracle_retained_unilateral_count"] / unilateral_count)
                        if unilateral_count else ""
                    ),
                    "original_success_preserved_count": result["original_success_preserved_count"],
                })
    return rows


def _weighted_pair_summary(rows: Sequence[Mapping]) -> list[dict]:
    result = []
    for scope in ("all", "both_v2i", "both_v2v", "mixed"):
        selected = [row for row in rows if row["mode_scope"] == scope and row["observed_same_rb_fraction"] != ""]
        denominator = sum(int(row["slot_count"]) for row in selected)
        result.append({
            "mode_scope": scope,
            "pair_slot_count": denominator,
            "observed_same_rb_fraction": float(
                sum(float(row["observed_same_rb_fraction"]) * int(row["slot_count"]) for row in selected)
                / denominator
            ) if denominator else "",
            "independent_marginal_reference": float(
                sum(float(row["independent_marginal_reference"]) * int(row["slot_count"]) for row in selected)
                / denominator
            ) if denominator else "",
        })
    for row in result:
        row["same_rb_excess_over_reference"] = (
            float(row["observed_same_rb_fraction"] - row["independent_marginal_reference"])
            if row["observed_same_rb_fraction"] != "" else ""
        )
    return result


def _oracle_seed_rows(rows: Sequence[Mapping]) -> list[dict]:
    result = []
    for seed in TRAINING_SEEDS:
        selected = [row for row in rows if int(row["training_seed"]) == seed]
        unilateral = sum(int(row["unilateral_rescue_count"]) for row in selected)
        retained = sum(int(row["oracle_retained_unilateral_count"]) for row in selected)
        result.append({
            "training_seed": seed,
            "sample_count": len(selected),
            "candidate_count_per_sample": 243,
            "mean_original_success_count": float(np.mean([row["original_success_count"] for row in selected])),
            "mean_oracle_success_count": float(np.mean([row["oracle_success_count"] for row in selected])),
            "mean_oracle_additional_success_count": float(
                np.mean([row["oracle_additional_success_count"] for row in selected])
            ),
            "unilateral_rescue_count": unilateral,
            "oracle_retained_unilateral_count": retained,
            "retained_unilateral_fraction": float(retained / unilateral) if unilateral else "",
        })
    return result


def _report(path: Path, overall: Mapping, pair_summary: Sequence[Mapping], oracle_seed: Sequence[Mapping]) -> None:
    oracle_improvement = float(np.mean([row["mean_oracle_additional_success_count"] for row in oracle_seed]))
    retained_values = [float(row["retained_unilateral_fraction"]) for row in oracle_seed if row["retained_unilateral_fraction"] != ""]
    retained_text = f"{np.mean(retained_values):.4f}" if retained_values else "NA"
    def formatted(value):
        return f"{float(value):.4f}" if value != "" else "NA"
    lines = [
        "# MAPPO RB selection and coordination-space audit",
        "",
        "Only baseline frozen-policy scored slots are analyzed; warmup is absent from the input NPZ.",
        "",
        "## V2I power requirement and unilateral RB opportunity",
        "",
        f"- Mean selected power margin (actual minus required): {overall['mean_mean_selected_power_margin_db']:.3f} dB.",
        f"- Mean best-RB power margin: {overall['mean_mean_best_power_margin_db']:.3f} dB.",
        f"- Mean selected positive shortfall: {overall['mean_mean_selected_positive_shortfall_db']:.3f} dB.",
        f"- Mean best-RB positive shortfall: {overall['mean_mean_best_positive_shortfall_db']:.3f} dB.",
        f"- Same-mask mean actual/selected-required/best-required power: "
        f"{overall['mean_mean_actual_power_dbm']:.3f}/"
        f"{overall['mean_mean_selected_required_power_dbm']:.3f}/"
        f"{overall['mean_mean_best_required_power_dbm']:.3f} dBm.",
        f"- Same-mask linear means: {overall['mean_mean_actual_power_mw']:.3f}/"
        f"{overall['mean_mean_selected_required_power_mw']:.3f}/"
        f"{overall['mean_mean_best_required_power_mw']:.3f} mW.",
        f"- Selected-to-best required-power reduction: "
        f"{overall['mean_mean_selected_minus_best_required_db']:.3f} dB = "
        f"channel {overall['mean_mean_channel_loss_contribution_db']:.3f} dB + "
        f"interference {overall['mean_mean_interference_contribution_db']:.3f} dB.",
        f"- Failed V2I slots rescued by a unilateral RB switch: {overall['mean_unilateral_rescue_fraction_of_v2i_failures']:.4f}.",
        "",
        "All dBm and linear-mW means use the same mode=V2I denominator and are reported separately in CSV.",
        "The selected-to-best required-power difference is decomposed on that exact RB pair into channel-loss and interference terms.",
        "",
        "## RB reuse versus independent agent marginals",
        "",
        "| scope | observed same RB | independent marginal reference | excess |",
        "|---|---:|---:|---:|",
    ]
    for row in pair_summary:
        lines.append(
            f"| {row['mode_scope']} | {formatted(row['observed_same_rb_fraction'])} "
            f"| {formatted(row['independent_marginal_reference'])} "
            f"| {formatted(row['same_rb_excess_over_reference'])} |"
        )
    lines.extend([
        "",
        "## Joint 3^5 oracle on the prespecified sample",
        "",
        f"- Samples: {sum(int(row['sample_count']) for row in oracle_seed)}; 243 assignments per sample.",
        f"- Mean instantaneous additional V2I successes: {oracle_improvement:.4f}.",
        f"- Mean retained fraction of unilateral rescue opportunities: {retained_text}.",
        "",
        "The oracle fixes mode and power, recomputes joint interference, and optimizes only instantaneous V2I success count.",
        "It is not a CAM, reward, long-run AoI, decentralized-policy, shared-actor, or IPPO upper bound.",
        "With five agents and three RBs, reuse is unavoidable; same-RB occupancy alone is not a coordination failure.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(input_root: Path, output_root: Path) -> dict:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stream_rows, usage_rows, pair_rows, occupancy_rows, all_oracle_rows, manifest = [], [], [], [], [], []
    for training_seed in TRAINING_SEEDS:
        run = _run_name(training_seed)
        eval_dir = input_root / "evaluations" / run / feasibility_eval_id("baseline")
        complete_path = eval_dir / "EVAL_COMPLETE.json"
        metrics_path = eval_dir / "metrics.npz"
        if not complete_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"missing baseline feasibility input for seed {training_seed}: {eval_dir}")
        complete = _read_json(complete_path)
        expected = {
            "status": "complete",
            "intervention_arm": "baseline",
            "training_seed": training_seed,
            "eval_seeds": list(EVAL_SEEDS),
            "eval_episodes": 100,
            "eval_warmup_episodes": 5,
            "mappo_eval_mode": "stochastic",
        }
        for key, value in expected.items():
            if complete.get(key) != value:
                raise ValueError(f"{complete_path}: {key}={complete.get(key)!r}, expected {value!r}")
        with np.load(metrics_path, allow_pickle=False) as data:
            missing = sorted(REQUIRED_FIELDS - set(data.files))
            if missing:
                raise KeyError(f"{metrics_path}: missing fields {missing}; do not rerun evaluation automatically")
            arrays = {key: np.asarray(data[key]) for key in REQUIRED_FIELDS}
        if arrays["aoi_ms"].shape != (6, 100, 100, 5):
            raise ValueError(f"{metrics_path}: unexpected scored trajectory shape {arrays['aoi_ms'].shape}")
        quantities = counterfactual_v2i_quantities(arrays, complete)
        stream_rows.extend(
            stream_summary(arrays, quantities, eval_index, agent, training_seed)
            for eval_index in range(6) for agent in range(5)
        )
        usage_rows.extend(rb_usage_rows(arrays, training_seed))
        pair_rows.extend(pairwise_reuse_rows(arrays, training_seed))
        occupancy_rows.extend(joint_occupancy_rows(arrays, training_seed))
        all_oracle_rows.extend(oracle_rows(arrays, quantities, training_seed))
        manifest.append({
            "training_seed": training_seed,
            "eval_dir": str(eval_dir),
            "metrics": str(metrics_path),
            "status": complete["status"],
            "source_config_hash": complete.get("source_config_hash"),
            "input_fields": "|".join(sorted(REQUIRED_FIELDS)),
        })

    seed_rows = [aggregate_streams(stream_rows, seed) for seed in TRAINING_SEEDS]
    overall = equal_seed_summary(seed_rows)
    pair_summary = _weighted_pair_summary(pair_rows)
    oracle_seed = _oracle_seed_rows(all_oracle_rows)
    special = [
        row for row in stream_rows
        if row["training_seed"] == 9 and row["eval_seed"] == 205 and row["agent"] == 2
    ]
    sample_plan = {
        "selection": "cartesian product of fixed early/middle/late scored episode and slot indices",
        "oracle_objective": "maximize instantaneous V2I success count",
        "oracle_tie_break": (
            "among equal-success assignments maximize retained unilateral rescues, then preserved "
            "original successes, then use enumeration order"
        ),
        "episode_indices": list(SAMPLE_EPISODES),
        "slot_indices": list(SAMPLE_SLOTS),
        "training_seeds": list(TRAINING_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "samples_per_eval_world": len(SAMPLE_EPISODES) * len(SAMPLE_SLOTS),
        "total_samples": len(all_oracle_rows),
        "joint_assignments_per_sample": 243,
    }
    _write_csv(output_root / "per_eval_seed_agent.csv", stream_rows)
    _write_csv(output_root / "per_training_seed.csv", seed_rows)
    _write_csv(output_root / "rb_usage_by_agent.csv", usage_rows)
    _write_csv(output_root / "pairwise_rb_reuse.csv", pair_rows)
    _write_csv(output_root / "pairwise_rb_reuse_summary.csv", pair_summary)
    _write_csv(output_root / "joint_rb_occupancy.csv", occupancy_rows)
    _write_csv(output_root / "joint_oracle_samples.csv", all_oracle_rows)
    _write_csv(output_root / "joint_oracle_by_training_seed.csv", oracle_seed)
    _write_csv(output_root / "special_seed9_eval205_agent2.csv", special)
    _write_csv(output_root / "input_manifest.csv", manifest)
    _write_json(output_root / "oracle_sample_plan.json", sample_plan)
    result = {
        "status": "PASS",
        "input_root": str(input_root),
        "training_seeds": list(TRAINING_SEEDS),
        "eval_seeds": list(EVAL_SEEDS),
        "input_cells": len(manifest),
        "per_eval_seed_agent_rows": len(stream_rows),
        "per_training_seed_rows": len(seed_rows),
        "rb_usage_rows": len(usage_rows),
        "pairwise_reuse_rows": len(pair_rows),
        "oracle_sample_rows": len(all_oracle_rows),
        "special_case_rows": len(special),
        "sample_plan": sample_plan,
        "equal_seed_summary": overall,
        "pairwise_reuse_summary": pair_summary,
        "joint_oracle_by_training_seed": oracle_seed,
        "limitations": [
            "oracle fixes current mode and power and measures instantaneous V2I success only",
            "logged data do not reconstruct counterfactual V2V links or CAM under changed RB assignments",
            "180 streams are not treated as independent training replicates",
        ],
    }
    _write_json(output_root / "rb_coordination_summary.json", result)
    _report(output_root / "rb_coordination_audit.md", overall, pair_summary, oracle_seed)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.input_root, args.output_root)
    except (FileNotFoundError, KeyError, ValueError) as error:
        args.output_root.mkdir(parents=True, exist_ok=True)
        failure = {"status": "INPUT_REVIEW_REQUIRED", "error": str(error)}
        _write_json(args.output_root / "INPUT_REVIEW_REQUIRED.json", failure)
        print(json.dumps(failure, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
