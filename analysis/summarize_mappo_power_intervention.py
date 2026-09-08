"""Summarize the frozen-policy MAPPO mode-conditioned power intervention."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.audit_mappo_service_power import power_mw
from analysis.evaluate_mappo_power_intervention import ARMS, DEFAULT_EVAL_SEEDS, intervention_eval_id


SEEDS = (8, 9, 10, 11, 12, 13)
INTERVENTIONS = ("v2i_plus3db", "v2v_minus3db")
SCENARIO = "p05_n04_g25"
COMMON_BASELINE_ARRAYS = (
    "aoi_ms", "success", "remaining_demand", "power_dbm", "rb", "mode", "action_normalized",
)


def _run_name(seed: int) -> str:
    return f"mappo_tdec_ab_tdec_{SCENARIO}_seed{seed:02d}"


def _old_eval_id() -> str:
    token = "-".join(str(seed) for seed in DEFAULT_EVAL_SEEDS)
    return f"eval_validation_policy_final_stochastic_sequential_warm_warm5_s{token}_ep100"


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


def interval_statistics(aoi: np.ndarray) -> dict:
    """Compute complete intervals and separate left/right censoring for one stream."""

    series = np.asarray(aoi, dtype=np.float64).reshape(-1)
    resets = np.flatnonzero(np.isclose(series, 1.0, rtol=0.0, atol=1e-6))
    complete = np.diff(resets).astype(np.int64) if resets.size > 1 else np.empty(0, dtype=np.int64)
    if resets.size:
        censored = [int(resets[0]), int(series.size - 1 - resets[-1])]
        censored = [length for length in censored if length > 0]
        both_censored = 0
    else:
        censored = [int(series.size)]
        both_censored = 1
    long = complete[complete > 100]
    return {
        "complete_intervals": complete,
        "complete_interval_count": int(complete.size),
        "complete_interval_median_slots": float(np.median(complete)) if complete.size else "",
        "complete_interval_p90_slots": float(np.quantile(complete, 0.90)) if complete.size else "",
        "complete_interval_p95_slots": float(np.quantile(complete, 0.95)) if complete.size else "",
        "complete_interval_max_slots": int(complete.max()) if complete.size else "",
        "complete_interval_gt100_count": int(long.size),
        "complete_interval_gt100_duration_fraction": (
            float(long.sum() / complete.sum()) if complete.size and complete.sum() else ""
        ),
        "censored_segment_count": len(censored),
        "censored_segment_slots": int(sum(censored)),
        "both_censored_stream": both_censored,
    }


def _stream_row(
    arrays: Mapping[str, np.ndarray],
    eval_index: int,
    agent: int,
    arm: str,
    training_seed: int,
    cam_bits: float,
) -> dict:
    aoi = arrays["aoi_ms"][eval_index, :, :, agent].astype(np.float64)
    mode = arrays["mode"][eval_index, :, :, agent].astype(np.int64)
    resets = arrays["reset_event"][eval_index, :, :, agent].astype(bool)
    attempts = mode == 0
    success = arrays["success"][eval_index, :, -1, agent].astype(np.float64)
    remaining = arrays["remaining_demand"][eval_index, :, -1, agent].astype(np.float64)
    executed_power = arrays["executed_power_dbm"][eval_index, :, :, agent].astype(np.float64)
    policy_power = arrays["policy_power_dbm"][eval_index, :, :, agent].astype(np.float64)
    rate = arrays["v2i_rate"][eval_index, :, :, agent].astype(np.float64)
    interference = arrays["selected_interference_db"][eval_index, :, :, agent].astype(np.float64)
    intervals = interval_statistics(aoi)
    attempt_count = int(attempts.sum())
    reset_count = int(resets.sum())

    def mean_mask(values, mask):
        selected = np.asarray(values)[mask]
        return float(selected.mean()) if selected.size else ""

    row = {
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
        "mean_executed_power_dbm": float(executed_power.mean()),
        "power_at_boundary_fraction": float(arrays["power_at_boundary"][eval_index, :, :, agent].mean()),
        "power_clipped_low_fraction": float(arrays["power_clipped_low"][eval_index, :, :, agent].mean()),
        "power_clipped_high_fraction": float(arrays["power_clipped_high"][eval_index, :, :, agent].mean()),
        "v2i_attempt_count": attempt_count,
        "aoi_reset_count": reset_count,
        "v2i_attempt_rate": float(attempt_count / aoi.size),
        "v2i_success_given_attempt": _safe_ratio(reset_count, attempt_count),
        "aoi_reset_rate": float(reset_count / aoi.size),
        "aoi_gt50_fraction": float((aoi > 50.0).mean()),
        "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, rtol=0.0, atol=1e-6).mean()),
        "v2i_mean_power_mw": mean_mask(power_mw(executed_power), attempts),
        "v2v_mean_power_mw": mean_mask(power_mw(executed_power), ~attempts),
        "v2i_mean_rate_bits_per_step": mean_mask(rate, attempts),
        "v2i_success_mean_rate_bits_per_step": mean_mask(rate, attempts & resets),
        "v2i_failure_mean_rate_bits_per_step": mean_mask(rate, attempts & ~resets),
        "v2i_mean_selected_interference_db": mean_mask(interference, attempts),
        "v2i_mean_selected_interference_linear": mean_mask(power_mw(interference), attempts),
        "v2v_mean_selected_interference_db": mean_mask(interference, ~attempts),
        "v2v_mean_selected_interference_linear": mean_mask(power_mw(interference), ~attempts),
        **{key: value for key, value in intervals.items() if key != "complete_intervals"},
    }
    return row


def _cell_summary(rows: Sequence[Mapping], arrays: Mapping[str, np.ndarray], arm: str, seed: int) -> dict:
    attempts = sum(int(row["v2i_attempt_count"]) for row in rows)
    resets = sum(int(row["aoi_reset_count"]) for row in rows)
    slots = sum(int(row["slot_count"]) for row in rows)
    complete = []
    censored_count = 0
    censored_slots = 0
    both_censored = 0
    for eval_index in range(len(DEFAULT_EVAL_SEEDS)):
        for agent in range(5):
            stats = interval_statistics(arrays["aoi_ms"][eval_index, :, :, agent])
            complete.extend(stats["complete_intervals"].tolist())
            censored_count += int(stats["censored_segment_count"])
            censored_slots += int(stats["censored_segment_slots"])
            both_censored += int(stats["both_censored_stream"])
    complete = np.asarray(complete, dtype=np.int64)
    long = complete[complete > 100]
    all_mode = arrays["mode"].astype(np.int64)
    all_attempts = all_mode == 0
    all_power_mw = power_mw(arrays["executed_power_dbm"])
    all_rate = arrays["v2i_rate"].astype(np.float64)
    all_interference_db = arrays["selected_interference_db"].astype(np.float64)

    def pooled_mean(values, mask):
        selected = np.asarray(values)[mask]
        return float(selected.mean()) if selected.size else ""
    agent_aoi = []
    agent_cam = []
    agent_payload = []
    for agent in range(5):
        selected = [row for row in rows if int(row["agent"]) == agent]
        agent_aoi.append(float(np.mean([row["mean_aoi_ms"] for row in selected])))
        agent_cam.append(float(np.mean([row["binary_cam"] for row in selected])))
        agent_payload.append(float(np.mean([row["payload_completion"] for row in selected])))
    numeric_mean = lambda key: float(np.mean([float(row[key]) for row in rows if row[key] != ""]))
    return {
        "arm": arm,
        "training_seed": seed,
        "mean_aoi_ms": float(np.mean(agent_aoi)),
        "worst_agent_aoi_ms": float(np.max(agent_aoi)),
        "mean_binary_cam": float(np.mean(agent_cam)),
        "worst_agent_binary_cam": float(np.min(agent_cam)),
        "mean_payload_completion": float(np.mean(agent_payload)),
        "worst_agent_payload_completion": float(np.min(agent_payload)),
        "mean_policy_power_mw": numeric_mean("mean_policy_power_mw"),
        "mean_executed_power_mw": numeric_mean("mean_executed_power_mw"),
        "power_at_boundary_fraction": numeric_mean("power_at_boundary_fraction"),
        "v2i_attempt_count": attempts,
        "aoi_reset_count": resets,
        "v2i_attempt_rate": float(attempts / slots),
        "v2i_success_given_attempt": _safe_ratio(resets, attempts),
        "aoi_reset_rate": float(resets / slots),
        "aoi_gt50_fraction": numeric_mean("aoi_gt50_fraction"),
        "aoi_at_cap_fraction": numeric_mean("aoi_at_cap_fraction"),
        "complete_interval_count": int(complete.size),
        "complete_interval_gt100_count": int(long.size),
        "complete_interval_gt100_duration_fraction": (
            float(long.sum() / complete.sum()) if complete.size and complete.sum() else ""
        ),
        "complete_interval_median_slots": float(np.median(complete)) if complete.size else "",
        "complete_interval_p90_slots": float(np.quantile(complete, 0.90)) if complete.size else "",
        "complete_interval_p95_slots": float(np.quantile(complete, 0.95)) if complete.size else "",
        "complete_interval_max_slots": int(complete.max()) if complete.size else "",
        "censored_segment_count": censored_count,
        "censored_segment_slots": censored_slots,
        "both_censored_streams": both_censored,
        "v2i_mean_power_mw": pooled_mean(all_power_mw, all_attempts),
        "v2v_mean_power_mw": pooled_mean(all_power_mw, ~all_attempts),
        "v2i_mean_rate_bits_per_step": pooled_mean(all_rate, all_attempts),
        "v2i_mean_selected_interference_db": pooled_mean(all_interference_db, all_attempts),
        "v2i_mean_selected_interference_linear": pooled_mean(power_mw(all_interference_db), all_attempts),
        "v2v_mean_selected_interference_db": pooled_mean(all_interference_db, ~all_attempts),
        "v2v_mean_selected_interference_linear": pooled_mean(power_mw(all_interference_db), ~all_attempts),
    }


def _baseline_equivalence(source_root: Path, result_root: Path, seed: int) -> dict:
    run = _run_name(seed)
    old_path = source_root / "evaluations" / run / _old_eval_id() / "metrics.npz"
    new_path = result_root / "evaluations" / run / intervention_eval_id("baseline") / "metrics.npz"
    row = {"training_seed": seed, "old_metrics": str(old_path), "new_metrics": str(new_path)}
    exact_all = True
    with np.load(old_path, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
        for key in COMMON_BASELINE_ARRAYS:
            exact = np.array_equal(old[key], new[key])
            exact_all = exact_all and exact
            row[f"{key}_exact"] = exact
            if np.issubdtype(old[key].dtype, np.number):
                row[f"{key}_max_abs_difference"] = float(
                    np.max(np.abs(old[key].astype(np.float64) - new[key].astype(np.float64)))
                )
            else:
                row[f"{key}_max_abs_difference"] = ""
    row["all_common_arrays_exact"] = exact_all
    return row


def _paired_rows(cell_rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    metrics = (
        "mean_aoi_ms", "worst_agent_aoi_ms", "mean_binary_cam", "worst_agent_binary_cam",
        "mean_payload_completion", "worst_agent_payload_completion", "mean_executed_power_mw",
        "v2i_attempt_rate", "v2i_success_given_attempt", "aoi_reset_rate", "aoi_gt50_fraction",
        "aoi_at_cap_fraction", "complete_interval_gt100_count",
        "complete_interval_gt100_duration_fraction", "censored_segment_count", "censored_segment_slots",
    )
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
            values = np.asarray([float(row[f"delta_{key}"]) for row in selected if row[f"delta_{key}"] != ""])
            summary[f"mean_delta_{key}"] = float(values.mean()) if values.size else ""
            summary[f"sd_delta_{key}"] = float(values.std(ddof=1)) if values.size > 1 else ""
        summary["aoi_improved_seed_count"] = sum(row["delta_mean_aoi_ms"] < 0 for row in selected)
        summary["cam_not_worse_seed_count"] = sum(row["delta_mean_binary_cam"] >= 0 for row in selected)
        summary["long_interval_not_worse_seed_count"] = sum(
            row["delta_complete_interval_gt100_count"] <= 0 for row in selected
        )
        summaries.append(summary)
    return rows, summaries


def _write_report(
    path: Path,
    cell_rows: Sequence[Mapping],
    paired: Sequence[Mapping],
    special_rows: Sequence[Mapping],
    baseline_exact: bool,
) -> None:
    lines = [
        "# MAPPO frozen-policy mode/power intervention",
        "",
        f"Baseline reproduction against the original TDec stochastic NPZ: **{'EXACT' if baseline_exact else 'REVIEW REQUIRED'}**.",
        "",
        "| arm | AoI | worst AoI | binary CAM | worst CAM | payload | power mW | rho | q | reset rate | AoI>50 | cap | >100 intervals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        rows = [row for row in cell_rows if row["arm"] == arm]
        mean = lambda key: float(np.mean([float(row[key]) for row in rows if row[key] != ""]))
        lines.append(
            f"| {arm} | {mean('mean_aoi_ms'):.3f} | {mean('worst_agent_aoi_ms'):.3f} "
            f"| {mean('mean_binary_cam'):.4f} | {mean('worst_agent_binary_cam'):.4f} "
            f"| {mean('mean_payload_completion'):.4f} | {mean('mean_executed_power_mw'):.3f} "
            f"| {mean('v2i_attempt_rate'):.4f} | {mean('v2i_success_given_attempt'):.4f} "
            f"| {mean('aoi_reset_rate'):.4f} | {mean('aoi_gt50_fraction'):.4f} "
            f"| {mean('aoi_at_cap_fraction'):.4f} | {mean('complete_interval_gt100_count'):.1f} |"
        )
    lines.extend([
        "",
        "## Paired change from baseline",
        "",
        "Negative AoI/power/tail changes and non-negative CAM/payload changes are favorable; no single metric defines success.",
        "",
        "| arm | delta AoI | delta CAM | delta payload | delta power mW | delta reset rate | delta AoI>50 | delta >100 intervals | AoI wins | CAM non-worse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in paired:
        lines.append(
            f"| {row['arm']} | {row['mean_delta_mean_aoi_ms']:.3f} "
            f"| {row['mean_delta_mean_binary_cam']:.4f} | {row['mean_delta_mean_payload_completion']:.4f} "
            f"| {row['mean_delta_mean_executed_power_mw']:.3f} | {row['mean_delta_aoi_reset_rate']:.4f} "
            f"| {row['mean_delta_aoi_gt50_fraction']:.4f} | {row['mean_delta_complete_interval_gt100_count']:.2f} "
            f"| {row['aoi_improved_seed_count']}/6 | {row['cam_not_worse_seed_count']}/6 |"
        )
    lines.extend([
        "",
        "## Prespecified seed 9 / held-out seed 205 / agent 2",
        "",
        "| arm | AoI | CAM | payload | power mW | rho | q | reset rate | AoI>50 | >100 intervals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in special_rows:
        q = "NA" if row["v2i_success_given_attempt"] == "" else f"{row['v2i_success_given_attempt']:.4f}"
        lines.append(
            f"| {row['arm']} | {row['mean_aoi_ms']:.3f} | {row['binary_cam']:.4f} "
            f"| {row['payload_completion']:.4f} | {row['mean_executed_power_mw']:.3f} "
            f"| {row['v2i_attempt_rate']:.4f} | {q} | {row['aoi_reset_rate']:.4f} "
            f"| {row['aoi_gt50_fraction']:.4f} | {row['complete_interval_gt100_count']} |"
        )
    lines.extend([
        "",
        "These are paired descriptive diagnostics over six training seeds, not a trained algorithm or a complete causal proof.",
        "AoI remains capped at 100; intervals longer than 100 are inferred only from observed reset timing.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(source_root: Path, result_root: Path, output_root: Path) -> dict:
    source_root = source_root.expanduser().resolve()
    result_root = result_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stream_rows = []
    cell_rows = []
    completion_rows = []
    for arm in ARMS:
        for seed in SEEDS:
            run = _run_name(seed)
            eval_dir = result_root / "evaluations" / run / intervention_eval_id(arm)
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
            expected_shape = (6, 100, 100, 5)
            required = {
                "aoi_ms", "success", "remaining_demand", "rb", "mode", "power_dbm", "v2i_rate",
                "selected_interference_db", "policy_action_normalized", "action_normalized",
                "policy_power_dbm", "executed_power_dbm", "power_clipped_low", "power_clipped_high",
                "power_at_boundary", "reset_event",
            }
            if required - arrays.keys():
                raise ValueError(f"{eval_dir}: missing arrays {sorted(required - arrays.keys())}")
            if any(arrays[key].shape != expected_shape for key in required if "action_normalized" not in key):
                raise ValueError(f"{eval_dir}: unexpected metric shape")
            if arrays["action_normalized"].shape != expected_shape + (3,) or arrays["policy_action_normalized"].shape != expected_shape + (3,):
                raise ValueError(f"{eval_dir}: unexpected action shape")
            reset_from_aoi = np.isclose(arrays["aoi_ms"], 1.0, rtol=0.0, atol=1e-6)
            if not np.array_equal(reset_from_aoi, arrays["reset_event"].astype(bool)):
                raise ValueError(f"{eval_dir}: reset event mismatch")
            if np.any(reset_from_aoi & (arrays["mode"] != 0)):
                raise ValueError(f"{eval_dir}: reset outside V2I mode")
            if not np.allclose(arrays["power_dbm"], arrays["executed_power_dbm"], rtol=0.0, atol=2e-5):
                raise ValueError(f"{eval_dir}: environment and recorded executed power differ")
            rows = [
                _stream_row(arrays, eval_index, agent, arm, seed, float(complete["cam_bits"]))
                for eval_index in range(6) for agent in range(5)
            ]
            stream_rows.extend(rows)
            cell_rows.append(_cell_summary(rows, arrays, arm, seed))
            provenance = _read_json(eval_dir / "provenance.json")
            completion_rows.append({
                "arm": arm,
                "training_seed": seed,
                "eval_dir": str(eval_dir),
                "reproduction_git_commit": provenance.get("reproduction_git_commit"),
                "policy": provenance.get("policy"),
                "status": complete["status"],
            })

    baseline_rows = [_baseline_equivalence(source_root, result_root, seed) for seed in SEEDS]
    baseline_exact = all(row["all_common_arrays_exact"] for row in baseline_rows)
    paired_rows, paired_summary = _paired_rows(cell_rows)
    special_rows = [
        row for row in stream_rows
        if row["training_seed"] == 9 and row["eval_seed"] == 205 and row["agent"] == 2
    ]
    _write_csv(output_root / "per_heldout_seed_agent.csv", stream_rows)
    _write_csv(output_root / "per_training_seed.csv", cell_rows)
    _write_csv(output_root / "paired_seed_deltas.csv", paired_rows)
    _write_csv(output_root / "paired_summary.csv", paired_summary)
    _write_csv(output_root / "baseline_equivalence.csv", baseline_rows)
    _write_csv(output_root / "special_seed9_eval205_agent2.csv", special_rows)
    _write_csv(output_root / "completion_manifest.csv", completion_rows)
    _write_report(
        output_root / "mode_power_intervention.md",
        cell_rows,
        paired_summary,
        special_rows,
        baseline_exact,
    )
    result = {
        "status": "PASS" if baseline_exact else "BASELINE_REVIEW_REQUIRED",
        "completed_cells": len(cell_rows),
        "training_seeds": list(SEEDS),
        "eval_seeds": list(DEFAULT_EVAL_SEEDS),
        "arms": list(ARMS),
        "baseline_common_arrays_exact": baseline_exact,
        "per_heldout_seed_agent_rows": len(stream_rows),
        "special_case_rows": len(special_rows),
        "paired_summary": paired_summary,
    }
    (output_root / "intervention_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.source_root, args.result_root, args.output_root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
