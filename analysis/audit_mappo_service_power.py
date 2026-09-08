"""Audit update timing, service success, power, and interference in MAPPO TDec A/B runs.

This module is deliberately post-processing only.  It reads the existing
``tdec-ab-v1`` NPZ/JSON artifacts and never imports the trainer or environment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


SEEDS = (8, 9, 10, 11, 12, 13)
EVAL_SEEDS = (201, 202, 203, 204, 205, 206)
VARIANTS = ("combined", "tdec")
MODES = ("deterministic", "stochastic")
SCENARIO = "p05_n04_g25"
TRAIN_REQUIRED = (
    "aoi_ms",
    "success",
    "remaining_demand",
    "power_dbm",
    "mode",
    "v2i_rate",
    "selected_interference_db",
)
EVAL_REQUIRED = ("aoi_ms", "success", "remaining_demand", "power_dbm", "mode")


def run_name(variant: str, seed: int) -> str:
    return f"mappo_tdec_ab_{variant}_{SCENARIO}_seed{seed:02d}"


def eval_id(mode: str) -> str:
    seed_token = "-".join(str(seed) for seed in EVAL_SEEDS)
    return f"eval_validation_policy_final_{mode}_sequential_warm_warm5_s{seed_token}_ep100"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _training_paths(root: Path, variant: str, seed: int) -> Dict[str, Path]:
    directory = root / "training" / "runs" / run_name(variant, seed)
    return {
        "directory": directory,
        "config": directory / "config.resolved.json",
        "complete": directory / "COMPLETE.json",
        "metrics": directory / "train_metrics.npz",
    }


def _evaluation_paths(root: Path, variant: str, seed: int, mode: str) -> Dict[str, Path]:
    directory = root / "evaluations" / run_name(variant, seed) / eval_id(mode)
    return {
        "directory": directory,
        "complete": directory / "EVAL_COMPLETE.json",
        "metrics": directory / "metrics.npz",
    }


def inventory(source_root: Path) -> List[dict]:
    """Return one coverage row for every expected training/evaluation cell."""

    source_root = source_root.expanduser().resolve()
    rows: List[dict] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            paths = _training_paths(source_root, variant, seed)
            rows.append(_inventory_row("training", variant, seed, "train", paths, TRAIN_REQUIRED))
            for mode in MODES:
                paths = _evaluation_paths(source_root, variant, seed, mode)
                rows.append(_inventory_row("evaluation", variant, seed, mode, paths, EVAL_REQUIRED))
    return rows


def _inventory_row(
    phase: str,
    variant: str,
    seed: int,
    mode: str,
    paths: Mapping[str, Path],
    required: Sequence[str],
) -> dict:
    missing_files = [name for name, path in paths.items() if name != "directory" and not path.is_file()]
    missing_arrays: List[str] = []
    array_shapes: Dict[str, List[int]] = {}
    error = ""
    if not missing_files:
        try:
            with np.load(paths["metrics"], allow_pickle=False) as data:
                missing_arrays = sorted(set(required) - set(data.files))
                array_shapes = {name: list(data[name].shape) for name in required if name in data.files}
            marker = _read_json(paths["complete"])
            if marker.get("status") != "complete":
                error = f"completion status={marker.get('status')!r}"
            elif marker.get("algorithm") != "mappo" or marker.get("mappo_variant") != variant:
                error = "completion identity mismatch"
            elif int(marker.get("training_seed", marker.get("seed", -1))) != seed:
                error = "completion seed mismatch"
            elif phase == "evaluation" and marker.get("mappo_eval_mode") != mode:
                error = "evaluation mode mismatch"
        except Exception as exc:  # inventory must preserve all failures in one report
            error = f"{type(exc).__name__}: {exc}"
    status = "ok" if not missing_files and not missing_arrays and not error else "missing_or_invalid"
    return {
        "phase": phase,
        "variant": variant,
        "training_seed": seed,
        "mode": mode,
        "status": status,
        "directory": str(paths["directory"]),
        "missing_files": ";".join(missing_files),
        "missing_arrays": ";".join(missing_arrays),
        "array_shapes": json.dumps(array_shapes, sort_keys=True),
        "error": error,
    }


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_inventory(output_root: Path, rows: Sequence[Mapping]) -> bool:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "inventory.csv", rows)
    ok = all(row["status"] == "ok" for row in rows)
    payload = {
        "status": "PASS" if ok else "INCOMPLETE",
        "expected_training": 12,
        "expected_evaluation": 24,
        "valid_training": sum(row["phase"] == "training" and row["status"] == "ok" for row in rows),
        "valid_evaluation": sum(row["phase"] == "evaluation" and row["status"] == "ok" for row in rows),
    }
    (output_root / "inventory.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ok


def power_mw(power_dbm: np.ndarray) -> np.ndarray:
    """Convert each dBm observation before aggregation."""

    return np.power(10.0, np.asarray(power_dbm, dtype=np.float64) / 10.0)


def update_interval_rows(
    aoi: np.ndarray,
    identity: Mapping[str, object],
) -> List[dict]:
    """Histogram complete and boundary-censored update intervals for one stream."""

    series = np.asarray(aoi, dtype=np.float64).reshape(-1)
    reset_indices = np.flatnonzero(np.isclose(series, 1.0, rtol=0.0, atol=1e-6))
    observations: List[tuple[str, int]] = []
    if reset_indices.size == 0:
        observations.append(("both_censored", int(series.size)))
    else:
        if reset_indices[0] > 0:
            observations.append(("left_censored", int(reset_indices[0])))
        observations.extend(("complete", int(length)) for length in np.diff(reset_indices))
        right = int(series.size - 1 - reset_indices[-1])
        if right > 0:
            observations.append(("right_censored", right))
    counts = Counter(observations)
    return [
        {**identity, "interval_kind": kind, "length_slots": length, "count": count}
        for (kind, length), count in sorted(counts.items())
    ]


def aoi_distribution_rows(aoi: np.ndarray, identity: Mapping[str, object]) -> List[dict]:
    values, counts = np.unique(np.asarray(aoi, dtype=np.float64).reshape(-1), return_counts=True)
    return [
        {**identity, "aoi_ms": float(value), "count": int(count)}
        for value, count in zip(values, counts)
    ]


def _safe_ratio(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else ""


def _mean_or_blank(values: np.ndarray, mask: np.ndarray | None = None):
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    return float(array.mean()) if array.size else ""


def _aggregate(
    aoi: np.ndarray,
    success: np.ndarray,
    remaining: np.ndarray,
    mode: np.ndarray,
    power_dbm_values: np.ndarray,
    cam_bits: float,
    v2i_rate: np.ndarray | None = None,
    selected_interference_db: np.ndarray | None = None,
) -> dict:
    aoi = np.asarray(aoi, dtype=np.float64)
    success = np.asarray(success, dtype=np.float64)
    remaining = np.asarray(remaining, dtype=np.float64)
    mode = np.asarray(mode, dtype=np.int64)
    power_dbm_values = np.asarray(power_dbm_values, dtype=np.float64)
    attempts = mode == 0
    resets = np.isclose(aoi, 1.0, rtol=0.0, atol=1e-6)
    if np.any(resets & ~attempts):
        raise ValueError("AoI reset observed outside V2I mode")
    reset_count = int(resets.sum())
    attempt_count = int(attempts.sum())
    slot_count = int(aoi.size)
    payload = np.clip(1.0 - remaining[..., -1, :] / float(cam_bits), 0.0, 1.0)
    result = {
        "slot_count": slot_count,
        "attempt_count": attempt_count,
        "reset_count": reset_count,
        "v2i_attempt_rate": float(attempt_count / slot_count),
        "v2i_success_given_attempt": _safe_ratio(reset_count, attempt_count),
        "aoi_reset_rate": float(reset_count / slot_count),
        "mean_aoi_ms": float(aoi.mean()),
        "p90_aoi_ms": float(np.quantile(aoi, 0.90)),
        "p95_aoi_ms": float(np.quantile(aoi, 0.95)),
        "p99_aoi_ms": float(np.quantile(aoi, 0.99)),
        "max_aoi_ms": float(aoi.max()),
        "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, atol=1e-6).mean()),
        "binary_cam": float(success[..., -1, :].mean()),
        "payload_completion": float(payload.mean()),
        "mean_power_dbm": float(power_dbm_values.mean()),
        "mean_power_mw": float(power_mw(power_dbm_values).mean()),
    }
    if v2i_rate is not None:
        rate = np.asarray(v2i_rate, dtype=np.float64)
        result.update({
            "mean_v2i_rate_bits_per_step": float(rate.mean()),
            "mean_v2i_rate_on_attempt_bits_per_step": _mean_or_blank(rate, attempts),
        })
    else:
        result.update({"mean_v2i_rate_bits_per_step": "", "mean_v2i_rate_on_attempt_bits_per_step": ""})
    if selected_interference_db is not None:
        interference = np.asarray(selected_interference_db, dtype=np.float64)
        result["mean_selected_interference_db"] = float(interference.mean())
        result["mean_selected_interference_on_attempt_db"] = _mean_or_blank(interference, attempts)
        result["mean_selected_interference_linear"] = float(power_mw(interference).mean())
        result["mean_selected_interference_on_attempt_linear"] = _mean_or_blank(power_mw(interference), attempts)
    else:
        result.update({
            "mean_selected_interference_db": "",
            "mean_selected_interference_on_attempt_db": "",
            "mean_selected_interference_linear": "",
            "mean_selected_interference_on_attempt_linear": "",
        })
    return result


def _power_groups(
    power: np.ndarray,
    mode: np.ndarray,
    aoi: np.ndarray,
    identity: Mapping[str, object],
    rate: np.ndarray | None,
    interference: np.ndarray | None,
) -> List[dict]:
    attempts = mode == 0
    resets = np.isclose(aoi, 1.0, atol=1e-6)
    groups = {
        "all": np.ones_like(attempts, dtype=bool),
        "v2i_attempt": attempts,
        "v2i_success": attempts & resets,
        "v2i_failure": attempts & ~resets,
        "v2v": ~attempts,
    }
    rows = []
    for name, mask in groups.items():
        selected = np.asarray(power, dtype=np.float64)[mask]
        rows.append({
            **identity,
            "group": name,
            "count": int(mask.sum()),
            "mean_power_dbm": _mean_or_blank(selected),
            "mean_power_mw": _mean_or_blank(power_mw(selected)),
            "p50_power_mw": float(np.quantile(power_mw(selected), 0.50)) if selected.size else "",
            "p90_power_mw": float(np.quantile(power_mw(selected), 0.90)) if selected.size else "",
            "mean_v2i_rate_bits_per_step": _mean_or_blank(rate, mask) if rate is not None else "",
            "mean_selected_interference_db": _mean_or_blank(interference, mask) if interference is not None else "",
            "mean_selected_interference_linear": _mean_or_blank(power_mw(interference), mask) if interference is not None else "",
        })
    return rows


def _identity(phase: str, variant: str, mode: str, seed: int, eval_seed, agent: int) -> dict:
    return {
        "phase": phase,
        "variant": variant,
        "mode": mode,
        "training_seed": seed,
        "eval_seed": eval_seed,
        "agent": agent,
    }


def _validate_training_config(config: Mapping, complete: Mapping, variant: str, seed: int) -> None:
    scenario = config.get("scenario", {})
    checks = {
        "algorithm": (config.get("algorithm"), "mappo"),
        "scenario": (scenario.get("id"), SCENARIO),
        "seed": (int(config.get("seed", -1)), seed),
        "episodes": (int(config.get("episodes", -1)), 500),
        "steps": (int(config.get("steps_per_episode", -1)), 100),
        "variant": (config.get("mappo_variant"), variant),
        "value clipping": (config.get("mappo_value_clip_mode"), "normalized"),
        "rollout episodes": (int(config.get("mappo_rollout_episodes", -1)), 5),
        "PPO epochs": (int(config.get("mappo_ppo_epochs", -1)), 10),
        "slow update": (int(config.get("slow_update_every_episodes", -1)), 1),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{variant}/seed{seed}: {label}={actual!r}, expected {expected!r}")
    if complete.get("status") != "complete" or complete.get("mappo_variant") != variant:
        raise ValueError(f"{variant}/seed{seed}: invalid COMPLETE.json")
    expected_floats = {
        "tau": 0.005,
        "mappo_actor_lr": 0.0005,
        "mappo_critic_lr": 0.0005,
        "mappo_entropy_coef_rb": 0.02,
        "mappo_entropy_coef_mode": 0.02,
        "mappo_entropy_coef_power": 0.002,
    }
    for name, expected in expected_floats.items():
        if not np.isclose(float(config.get(name, np.nan)), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"{variant}/seed{seed}: {name} does not match the frozen tdec-ab protocol")


def _training_periods() -> Iterable[tuple[str, slice]]:
    for start in range(0, 500, 100):
        end = start + 100
        suffix = "_last100" if end == 500 else ""
        yield f"block_{start + 1:03d}_{end:03d}{suffix}", slice(start, end)


def _consume_training(root: Path, variant: str, seed: int, outputs: Dict[str, List[dict]]) -> None:
    paths = _training_paths(root, variant, seed)
    config = _read_json(paths["config"])
    complete = _read_json(paths["complete"])
    _validate_training_config(config, complete, variant, seed)
    with np.load(paths["metrics"], allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in TRAIN_REQUIRED}
    expected = (500, 100, 5)
    if any(arrays[name].shape != expected for name in TRAIN_REQUIRED):
        raise ValueError(f"{variant}/seed{seed}: training arrays must all have shape {expected}")
    threshold = float(config["v2i_min_bps_per_hz"]) * float(config["bandwidth_hz"]) * float(config["slot_ms"]) / 1000.0
    observed_reset = np.isclose(arrays["aoi_ms"], 1.0, atol=1e-6)
    expected_reset = arrays["v2i_rate"] >= threshold
    mismatch_count = int(np.count_nonzero(observed_reset != expected_reset))
    mismatch = observed_reset != expected_reset
    near_tolerance = max(1e-3, threshold * 1e-6)
    near_mismatch_count = int(np.count_nonzero(mismatch & (np.abs(arrays["v2i_rate"] - threshold) <= near_tolerance)))
    hard_mismatch_count = mismatch_count - near_mismatch_count
    reset_outside_attempt_count = int(np.count_nonzero(observed_reset & (arrays["mode"] != 0)))
    outputs["integrity"].append({
        "phase": "training",
        "variant": variant,
        "mode": "train",
        "training_seed": seed,
        "reset_outside_v2i_attempt": reset_outside_attempt_count,
        "reset_rate_threshold_mismatch": mismatch_count,
        "near_threshold_mismatch": near_mismatch_count,
        "hard_threshold_mismatch": hard_mismatch_count,
        "v2i_threshold_bits_per_step": threshold,
    })
    if reset_outside_attempt_count or hard_mismatch_count:
        raise ValueError(
            f"{variant}/seed{seed}: reset integrity failed; outside-attempt={reset_outside_attempt_count}, "
            f"hard rate mismatch={hard_mismatch_count}"
        )
    for agent in range(5):
        identity = _identity("training", variant, "train", seed, "", agent)
        outputs["intervals"].extend(update_interval_rows(arrays["aoi_ms"][:, :, agent], identity))
        outputs["aoi_distribution"].extend(aoi_distribution_rows(arrays["aoi_ms"][:, :, agent], identity))
        outputs["power_groups"].extend(_power_groups(
            arrays["power_dbm"][:, :, agent], arrays["mode"][:, :, agent], arrays["aoi_ms"][:, :, agent],
            identity, arrays["v2i_rate"][:, :, agent], arrays["selected_interference_db"][:, :, agent],
        ))
        for label, selection in _training_periods():
            sliced = {name: value[selection, :, agent][..., None] for name, value in arrays.items()}
            row = {
                **identity,
                "period": label,
                **_aggregate(
                    sliced["aoi_ms"], sliced["success"], sliced["remaining_demand"], sliced["mode"],
                    sliced["power_dbm"], float(config["cam_bits"]), sliced["v2i_rate"],
                    sliced["selected_interference_db"],
                ),
            }
            outputs["seed_agent"].append(row)
    for episode in range(500):
        for agent in range(5):
            sliced = {name: value[episode : episode + 1, :, agent][..., None] for name, value in arrays.items()}
            outputs["episode_agent"].append({
                **_identity("training", variant, "train", seed, "", agent),
                "episode": episode + 1,
                **_aggregate(
                    sliced["aoi_ms"], sliced["success"], sliced["remaining_demand"], sliced["mode"],
                    sliced["power_dbm"], float(config["cam_bits"]), sliced["v2i_rate"],
                    sliced["selected_interference_db"],
                ),
            })


def _consume_evaluation(root: Path, variant: str, seed: int, mode_name: str, outputs: Dict[str, List[dict]]) -> None:
    paths = _evaluation_paths(root, variant, seed, mode_name)
    complete = _read_json(paths["complete"])
    if complete.get("status") != "complete" or complete.get("mappo_variant") != variant:
        raise ValueError(f"{variant}/seed{seed}/{mode_name}: invalid EVAL_COMPLETE.json")
    if complete.get("mappo_eval_mode") != mode_name or int(complete.get("training_seed", -1)) != seed:
        raise ValueError(f"{variant}/seed{seed}/{mode_name}: evaluation identity mismatch")
    if complete.get("eval_seeds") != list(EVAL_SEEDS) or int(complete.get("eval_episodes", -1)) != 100:
        raise ValueError(f"{variant}/seed{seed}/{mode_name}: wrong held-out protocol")
    cam_bits = float(_read_json(_training_paths(root, variant, seed)["config"])["cam_bits"])
    with np.load(paths["metrics"], allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in EVAL_REQUIRED}
    expected = (6, 100, 100, 5)
    if any(arrays[name].shape != expected for name in EVAL_REQUIRED):
        raise ValueError(f"{variant}/seed{seed}/{mode_name}: evaluation arrays must all have shape {expected}")
    reset_outside_attempt_count = int(np.count_nonzero(
        np.isclose(arrays["aoi_ms"], 1.0, atol=1e-6) & (arrays["mode"] != 0)
    ))
    outputs["integrity"].append({
        "phase": "evaluation",
        "variant": variant,
        "mode": mode_name,
        "training_seed": seed,
        "reset_outside_v2i_attempt": reset_outside_attempt_count,
        "reset_rate_threshold_mismatch": "",
        "near_threshold_mismatch": "",
        "hard_threshold_mismatch": "",
        "v2i_threshold_bits_per_step": "",
    })
    if reset_outside_attempt_count:
        raise ValueError(f"{variant}/seed{seed}/{mode_name}: AoI reset outside V2I mode")
    for eval_index, heldout_seed in enumerate(EVAL_SEEDS):
        for agent in range(5):
            identity = _identity("evaluation", variant, mode_name, seed, heldout_seed, agent)
            series = {name: value[eval_index, :, :, agent][..., None] for name, value in arrays.items()}
            outputs["seed_agent"].append({
                **identity,
                "period": "scored_all",
                **_aggregate(
                    series["aoi_ms"], series["success"], series["remaining_demand"], series["mode"],
                    series["power_dbm"], cam_bits,
                ),
            })
            outputs["intervals"].extend(update_interval_rows(series["aoi_ms"], identity))
            outputs["aoi_distribution"].extend(aoi_distribution_rows(series["aoi_ms"], identity))
            outputs["power_groups"].extend(_power_groups(
                series["power_dbm"], series["mode"], series["aoi_ms"], identity, None, None,
            ))
            for episode in range(100):
                episode_data = {name: value[eval_index, episode : episode + 1, :, agent][..., None] for name, value in arrays.items()}
                outputs["episode_agent"].append({
                    **identity,
                    "episode": episode + 1,
                    **_aggregate(
                        episode_data["aoi_ms"], episode_data["success"], episode_data["remaining_demand"],
                        episode_data["mode"], episode_data["power_dbm"], cam_bits,
                    ),
                })


def _numeric_mean(rows: Sequence[Mapping], key: str):
    values = [float(row[key]) for row in rows if row.get(key) != ""]
    return float(np.mean(values)) if values else ""


def _cohort_rows(seed_agent_rows: Sequence[Mapping]) -> List[dict]:
    keys = sorted({(row["phase"], row["variant"], row["mode"], row["training_seed"], row["period"]) for row in seed_agent_rows})
    result = []
    for phase, variant, mode_name, seed, period in keys:
        selected = [row for row in seed_agent_rows if (
            row["phase"], row["variant"], row["mode"], row["training_seed"], row["period"]
        ) == (phase, variant, mode_name, seed, period)]
        slot_count = sum(int(row["slot_count"]) for row in selected)
        attempt_count = sum(int(row["attempt_count"]) for row in selected)
        reset_count = sum(int(row["reset_count"]) for row in selected)
        result.append({
            "phase": phase,
            "variant": variant,
            "mode": mode_name,
            "training_seed": seed,
            "period": period,
            "slot_count": slot_count,
            "attempt_count": attempt_count,
            "reset_count": reset_count,
            "v2i_attempt_rate": float(attempt_count / slot_count),
            "v2i_success_given_attempt": _safe_ratio(reset_count, attempt_count),
            "aoi_reset_rate": float(reset_count / slot_count),
            **{key: _numeric_mean(selected, key) for key in (
                "mean_aoi_ms", "binary_cam", "payload_completion", "mean_power_dbm", "mean_power_mw",
                "mean_v2i_rate_on_attempt_bits_per_step", "mean_selected_interference_on_attempt_db",
                "mean_selected_interference_on_attempt_linear",
            )},
        })
    return result


def _expanded_intervals(rows: Sequence[Mapping]) -> np.ndarray:
    return np.asarray([
        row["length_slots"] for row in rows if row["interval_kind"] == "complete"
        for _ in range(int(row["count"]))
    ], dtype=np.float64)


def _format_mean(rows: Sequence[Mapping], key: str, digits: int) -> str:
    value = _numeric_mean(rows, key)
    return "NA" if value == "" else f"{float(value):.{digits}f}"


def _write_report(path: Path, cohort_rows: Sequence[Mapping], interval_rows: Sequence[Mapping]) -> None:
    final_rows = [row for row in cohort_rows if row["period"] in {"block_401_500_last100", "scored_all"}]
    lines = [
        "# MAPPO service and power audit",
        "",
        "Read-only post-processing of the existing tdec-ab-v1 runs. Training last-100 and held-out modes are reported separately.",
        "",
        "| phase | mode | variant | AoI | binary CAM | payload | attempt rho | conditional q | reset rate | power mW |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in ("training", "evaluation"):
        for mode_name in (("train",) if phase == "training" else MODES):
            for variant in VARIANTS:
                rows = [row for row in final_rows if row["phase"] == phase and row["mode"] == mode_name and row["variant"] == variant]
                if not rows:
                    continue
                lines.append(
                    f"| {phase} | {mode_name} | {variant} | {_format_mean(rows, 'mean_aoi_ms', 3)} "
                    f"| {_format_mean(rows, 'binary_cam', 4)} | {_format_mean(rows, 'payload_completion', 4)} "
                    f"| {_format_mean(rows, 'v2i_attempt_rate', 4)} | {_format_mean(rows, 'v2i_success_given_attempt', 4)} "
                    f"| {_format_mean(rows, 'aoi_reset_rate', 4)} | {_format_mean(rows, 'mean_power_mw', 3)} |"
                )
    lines.extend([
        "",
        "## Complete inter-update intervals",
        "",
        "| phase | mode | variant | count | median | p90 | p95 | max | censored segments |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for phase in ("training", "evaluation"):
        for mode_name in (("train",) if phase == "training" else MODES):
            for variant in VARIANTS:
                selected = [row for row in interval_rows if row["phase"] == phase and row["mode"] == mode_name and row["variant"] == variant]
                expanded = _expanded_intervals(selected)
                censored = sum(int(row["count"]) for row in selected if row["interval_kind"] != "complete")
                if expanded.size:
                    lines.append(
                        f"| {phase} | {mode_name} | {variant} | {expanded.size} | {np.quantile(expanded, 0.5):.1f} "
                        f"| {np.quantile(expanded, 0.9):.1f} | {np.quantile(expanded, 0.95):.1f} "
                        f"| {expanded.max():.0f} | {censored} |"
                    )
                else:
                    lines.append(f"| {phase} | {mode_name} | {variant} | 0 |  |  |  |  | {censored} |")
    lines.extend([
        "",
        "Evaluation NPZ files do not contain V2I rate or interference; those columns are intentionally blank for held-out rows.",
        "AoI is capped at 100 slots, so this audit cannot identify the unconstrained tail beyond that cap.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(source_root: Path, output_root: Path) -> Dict[str, int]:
    outputs: Dict[str, List[dict]] = {
        "episode_agent": [],
        "seed_agent": [],
        "intervals": [],
        "aoi_distribution": [],
        "power_groups": [],
        "integrity": [],
    }
    for variant in VARIANTS:
        for seed in SEEDS:
            _consume_training(source_root, variant, seed, outputs)
            for mode_name in MODES:
                _consume_evaluation(source_root, variant, seed, mode_name, outputs)
    cohort = _cohort_rows(outputs["seed_agent"])
    for name, rows in outputs.items():
        _write_csv(output_root / f"{name}.csv", rows)
    _write_csv(output_root / "cohort_seed_summary.csv", cohort)
    _write_report(output_root / "service_power_audit.md", cohort, outputs["intervals"])
    summary = {
        "status": "PASS",
        "training_runs": 12,
        "evaluation_runs": 24,
        "episode_agent_rows": len(outputs["episode_agent"]),
        "seed_agent_rows": len(outputs["seed_agent"]),
        "interval_histogram_rows": len(outputs["intervals"]),
        "aoi_distribution_rows": len(outputs["aoi_distribution"]),
        "power_group_rows": len(outputs["power_groups"]),
        "cohort_seed_rows": len(cohort),
        "rate_and_interference_scope": "training_only",
        "aoi_cap_ms": 100,
    }
    (output_root / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    rows = inventory(source_root)
    if not write_inventory(output_root, rows):
        print((output_root / "inventory.json").read_text(encoding="utf-8"))
        return 2
    if args.inventory_only:
        print((output_root / "inventory.json").read_text(encoding="utf-8"))
        return 0
    summary = audit(source_root, output_root)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
