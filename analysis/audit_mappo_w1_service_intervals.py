"""W1: read-only audit of complete V2I reset intervals in existing dev evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis.mappo_e5_contract import EVAL_SEEDS, SCENARIO, SEEDS, independent_run_name, shared_run_name


CONDITIONS = (("independent", "combined"), ("shared", "combined"),
              ("independent", "tdec"), ("shared", "tdec"))
SHAPE = (6, 100, 100, 5)
SLOTS_PER_FLOW = 10000
T95_DF5 = 2.570581835636314
METRICS = ("reset_rate", "event_mean_L", "event_cv2", "flow_equal_mean_L",
           "flow_equal_within_cv2", "between_flow_mean_cv2", "long_gt100_count",
           "long_gt100_per_10000_slots", "long_gt100_fraction_intervals",
           "event_ccdf_gt100", "flow_equal_ccdf_gt100")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(value, expected, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label}: {value!r}, expected {expected!r}")


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source(study_root: Path, structure: str, variant: str, seed: int) -> tuple[Path, str]:
    base = (study_root / "E5-sharing-critic" / "P5_N4_gap25" if variant == "combined"
            else study_root / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25")
    run_name = (independent_run_name(variant, seed) if structure == "independent"
                else shared_run_name(variant, seed))
    # Same frozen directory identifier as feasibility_eval_id, without importing
    # the evaluator (which imports torch and policy/environment code).
    eval_id = ("eval_validation_policy_final_stochastic_sequential_warm_"
               f"warm5_s{'-'.join(map(str, EVAL_SEEDS))}_ep100_feasibility_reward_arm_baseline")
    return base / "evaluations" / run_name / eval_id, run_name


def _training_dir(study_root: Path, structure: str, variant: str, run_name: str) -> Path:
    if structure == "independent":
        root = study_root / "existing-evidence" / "tdec-ab-v1" / "P5_N4_gap25"
    elif variant == "tdec":
        root = study_root / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25"
    else:
        root = study_root / "E5-sharing-critic" / "P5_N4_gap25"
    return root / "training" / "runs" / run_name


def _load_events(eval_dir: Path, training_dir: Path, structure: str, variant: str, seed: int,
                 run_name: str) -> tuple[np.ndarray, dict]:
    complete = _json(eval_dir / "EVAL_COMPLETE.json")
    config = _json(training_dir / "config.resolved.json")
    for field, actual, expected_value in (
        ("algorithm", config.get("algorithm"), "mappo"),
        ("scenario.id", config.get("scenario", {}).get("id"), SCENARIO),
        ("seed", config.get("seed"), seed),
        ("mappo_variant", config.get("mappo_variant", "combined"), variant),
        ("mappo_actor_sharing", bool(config.get("mappo_actor_sharing", False)), structure == "shared"),
        ("n_rb", config.get("n_rb"), 3),
        ("episodes", config.get("episodes"), 500),
        ("steps_per_episode", config.get("steps_per_episode"), 100),
    ):
        _require(actual, expected_value, f"{training_dir}/config/{field}")
    training_complete = _json(training_dir / "COMPLETE.json")
    _require(training_complete.get("status"), "complete", f"{training_dir}/COMPLETE/status")
    _require(training_complete.get("mappo_variant", "combined"), variant,
             f"{training_dir}/COMPLETE/mappo_variant")
    _require(bool(training_complete.get("actor_sharing", False)), structure == "shared",
             f"{training_dir}/COMPLETE/actor_sharing")
    expected = {
        "status": "complete", "algorithm": "mappo", "mappo_variant": variant,
        "actor_sharing": structure == "shared",
        "actor_network_count": 1 if structure == "shared" else 5,
        "training_seed": seed, "training_run_name": run_name,
        "scenario": SCENARIO, "eval_seeds": list(EVAL_SEEDS), "eval_episodes": 100,
        "eval_warmup_episodes": 5, "mappo_eval_mode": "stochastic",
        "eval_protocol": "sequential_warm", "intervention_arm": "baseline",
        "policy_name": "policy_final.pt", "policy_episode": 500,
        "policy_parameters_unchanged": True, "is_frozen_eval": True,
    }
    for field, expected_value in expected.items():
        _require(complete.get(field), expected_value, f"{eval_dir}/{field}")
    _require(complete.get("eval_id"), eval_dir.name, f"{eval_dir}/eval_id")
    if complete.get("policy_path_is_relative_to_eval") is not True:
        raise ValueError(f"{eval_dir}: policy path is not relative to evaluation")
    policy_ref = complete.get("policy")
    if not isinstance(policy_ref, str) or (eval_dir / policy_ref).resolve() != (training_dir / "policy_final.pt").resolve():
        raise ValueError(f"{eval_dir}: policy reference does not resolve to seed-matched source")
    axes = complete.get("raw_metric_axes", {})
    if not isinstance(axes, dict):
        raise ValueError(f"{eval_dir}: raw_metric_axes missing")
    path = eval_dir / "metrics.npz"
    with np.load(path, allow_pickle=False) as loaded:
        keys = set(loaded.files)
        if "reset_event" not in keys and "aoi_ms" not in keys:
            raise ValueError(f"{path}: neither reset_event nor aoi_ms exists")
        event_source = "reset_event" if "reset_event" in keys else "aoi_ms_isclose_1"
        need = ("reset_event", "aoi_ms") if {"reset_event", "aoi_ms"} <= keys else (
            "reset_event" if "reset_event" in keys else "aoi_ms",)
        # NPZ is lazy: only the two requested members are decompressed.
        arrays = {key: loaded[key] for key in need}
    for key, array in arrays.items():
        _require(tuple(array.shape), SHAPE, f"{path}/{key} shape")
        _require(axes.get(key), ["eval_seed", "scored_episode", "slot", "agent"],
                 f"{eval_dir}/raw_metric_axes/{key}")
    if "reset_event" in arrays:
        raw = arrays["reset_event"]
        if raw.dtype != np.bool_:
            if not np.issubdtype(raw.dtype, np.integer) or not np.isin(raw, [0, 1]).all():
                raise ValueError(f"{path}: reset_event must be boolean/0-1")
        events = raw.astype(bool)
    else:
        events = None
    if "aoi_ms" in arrays:
        aoi = arrays["aoi_ms"]
        if not np.isfinite(aoi).all():
            raise ValueError(f"{path}: non-finite aoi_ms")
        from_aoi = np.isclose(aoi, 1.0, rtol=0.0, atol=1e-6)
        if events is None:
            events = from_aoi
        elif not np.array_equal(events, from_aoi):
            mismatch = np.argwhere(events != from_aoi)[0].tolist()
            raise ValueError(f"{path}: reset_event/aoi_ms mismatch at {mismatch}")
    assert events is not None
    inventory = {
        "actor_structure": structure, "value_configuration": variant,
        "training_seed": seed, "run_name": run_name, "evaluation_dir": str(eval_dir),
        "metrics_npz": str(path), "worlds": list(EVAL_SEEDS), "world_count": 6,
        "training_config": str(training_dir / "config.resolved.json"),
        "scored_episodes_per_world": 100, "slots_per_episode": 100, "agents": 5,
        "scenario": SCENARIO, "n_rb": 3, "policy_name": "policy_final.pt",
        "policy_episode": 500, "evaluation_mode": "stochastic", "intervention_arm": "baseline",
        "evaluation_protocol": "sequential_warm", "warmup_episodes_per_world": 5,
        "array_shape": list(SHAPE), "available_event_fields": sorted(keys & {"reset_event", "aoi_ms"}),
        "event_source": event_source,
        "scored_axis_confirmed": "raw_metric_axes:scored_episode; warmup excluded by evaluator",
        "evaluation_commit": complete.get("reproduction_git_commit"),
        "training_commit": complete.get("training_source_commit"),
    }
    return events, inventory


def flow_statistics(reset: np.ndarray) -> tuple[dict, Counter[int], list[int]]:
    """One world-agent stream; episodes may be 2-D, but worlds must be separate calls."""
    array = np.asarray(reset)
    if array.ndim not in (1, 2):
        raise ValueError("one flow must be a 1-D slot or 2-D episode x slot array")
    if array.dtype != np.bool_:
        raise ValueError("one flow must contain boolean reset events")
    flat = array.reshape(-1)
    positions = np.flatnonzero(flat)
    intervals = np.diff(positions).astype(np.int64)
    freq = Counter(int(length) for length in intervals)
    count, resets, slots = int(intervals.size), int(positions.size), int(flat.size)
    sum_l, sum_l2 = int(intervals.sum()), int(np.square(intervals).sum())
    mean = sum_l / count if count else None
    cv2 = (sum_l2 / count) / mean**2 - 1 if count else None
    long = intervals[intervals > 100]
    row = {
        "slot_count": slots, "reset_count": resets, "reset_rate": resets / slots,
        "complete_interval_count": count, "sum_L": sum_l, "sum_L2": sum_l2,
        "complete_interval_mean_L": mean, "complete_interval_cv2": cv2,
        "long_gt100_count": int(long.size), "long_gt100_sum_L": int(long.sum()),
        "left_boundary_visible_wait_slots": int(positions[0]) if resets else None,
        "right_boundary_visible_wait_slots": int(slots - 1 - positions[-1]) if resets else None,
        "no_reset_window_slots": slots if not resets else None,
        "no_reset_flow": not bool(resets), "no_complete_interval_flow": count == 0,
    }
    return row, freq, [int(p) for p in positions[:3]]


def _cv2(count: int, sum_l: int, sum_l2: int) -> float | None:
    if not count:
        return None
    mean = sum_l / count
    return (sum_l2 / count) / mean**2 - 1


def _ccdf(rows: list[dict], frequencies: dict[tuple, Counter[int]], key: tuple,
          thresholds: list[int]) -> list[dict]:
    eligible = [row for row in rows if row["complete_interval_count"] > 0]
    pooled = Counter()
    for row in rows:
        pooled.update(frequencies[key + (row["world"], row["agent"])])
    n = sum(pooled.values())
    result = []
    for threshold in thresholds:
        event = sum(count for length, count in pooled.items() if length > threshold) / n if n else None
        equal = (sum(sum(count for length, count in frequencies[key + (row["world"], row["agent"])].items()
                         if length > threshold) / row["complete_interval_count"] for row in eligible) / len(eligible)
                 if eligible else None)
        result.append({"actor_structure": key[0], "value_configuration": key[1],
                       "training_seed": key[2], "threshold_L": threshold,
                       "event_ccdf_P_L_gt_threshold": event,
                       "flow_equal_ccdf_P_L_gt_threshold": equal,
                       "event_denominator_intervals": n, "flow_equal_denominator_flows": len(eligible),
                       "all_flow_count": len(rows), "no_complete_interval_flow_count": len(rows) - len(eligible)})
    return result


def seed_statistics(rows: list[dict], frequencies: dict[tuple, Counter[int]], key: tuple) -> tuple[dict, list[dict]]:
    if len(rows) != 30:
        raise ValueError(f"{key}: expected 30 flows, got {len(rows)}")
    slots = sum(row["slot_count"] for row in rows)
    resets = sum(row["reset_count"] for row in rows)
    count = sum(row["complete_interval_count"] for row in rows)
    sum_l = sum(row["sum_L"] for row in rows)
    sum_l2 = sum(row["sum_L2"] for row in rows)
    long = sum(row["long_gt100_count"] for row in rows)
    eligible = [row for row in rows if row["complete_interval_count"]]
    flow_means = np.array([row["complete_interval_mean_L"] for row in eligible], dtype=float)
    within_cv2 = [row["complete_interval_cv2"] for row in eligible]
    support = sorted({0, 100} | {length for row in rows
                                for length in frequencies[key + (row["world"], row["agent"]) ]})
    ccdf = _ccdf(rows, frequencies, key, support)
    at100 = next(item for item in ccdf if item["threshold_L"] == 100)
    seed_row = {
        "actor_structure": key[0], "value_configuration": key[1], "training_seed": key[2],
        "flow_count": len(rows), "eligible_flow_count": len(eligible),
        "no_complete_interval_flow_count": len(rows) - len(eligible),
        "no_reset_flow_count": sum(row["no_reset_flow"] for row in rows),
        "slot_count": slots, "reset_count": resets, "reset_rate": resets / slots,
        "complete_interval_count": count, "sum_L": sum_l, "sum_L2": sum_l2,
        "event_mean_L": sum_l / count if count else None,
        "event_cv2": _cv2(count, sum_l, sum_l2),
        "flow_equal_mean_L": float(flow_means.mean()) if len(flow_means) else None,
        "flow_equal_within_cv2": float(np.mean(within_cv2)) if within_cv2 else None,
        "between_flow_mean_cv2": float(np.var(flow_means) / np.mean(flow_means)**2) if len(flow_means) else None,
        "long_gt100_count": long, "long_gt100_per_10000_slots": long / slots * 10000,
        "long_gt100_fraction_intervals": long / count if count else None,
        "event_ccdf_gt100": at100["event_ccdf_P_L_gt_threshold"],
        "flow_equal_ccdf_gt100": at100["flow_equal_ccdf_P_L_gt_threshold"],
    }
    return seed_row, ccdf


def _describe(values: list[float | None]) -> dict:
    available = np.array([value for value in values if value is not None], dtype=float)
    if len(available) != 6:
        return {"mean": None, "sample_sd": None, "ci95_low": None, "ci95_high": None,
                "defined_seed_count": len(available)}
    mean = float(available.mean())
    sd = float(available.std(ddof=1))
    margin = T95_DF5 * sd / np.sqrt(6)
    return {"mean": mean, "sample_sd": sd, "ci95_low": float(mean - margin),
            "ci95_high": float(mean + margin), "defined_seed_count": 6}


def _summary_rows(seed_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    indexed = {(row["actor_structure"], row["value_configuration"], row["training_seed"]): row
               for row in seed_rows}
    conditions, effects, effect_summary = [], [], []
    for structure, variant in CONDITIONS:
        selected = [indexed[structure, variant, seed] for seed in SEEDS]
        row = {"actor_structure": structure, "value_configuration": variant,
               "training_seed_count": 6, "flow_count": sum(item["flow_count"] for item in selected),
               "eligible_flow_count": sum(item["eligible_flow_count"] for item in selected),
               "no_complete_interval_flow_count": sum(item["no_complete_interval_flow_count"] for item in selected),
               "no_reset_flow_count": sum(item["no_reset_flow_count"] for item in selected),
               "slot_count": sum(item["slot_count"] for item in selected),
               "complete_interval_count": sum(item["complete_interval_count"] for item in selected),
               "long_gt100_total_count": sum(item["long_gt100_count"] for item in selected)}
        for metric in METRICS:
            row.update({f"{metric}_{name}": value for name, value in
                        _describe([item[metric] for item in selected]).items()})
        conditions.append(row)
    for variant in ("combined", "tdec"):
        selected = []
        for seed in SEEDS:
            independent, shared = indexed["independent", variant, seed], indexed["shared", variant, seed]
            row = {"value_configuration": variant, "training_seed": seed,
                   "delta_definition": "shared_minus_independent"}
            for metric in METRICS:
                left, right = independent[metric], shared[metric]
                row[f"delta_{metric}"] = right - left if left is not None and right is not None else None
            effects.append(row)
            selected.append(row)
        summary = {"value_configuration": variant, "training_seed_count": 6,
                   "delta_definition": "shared_minus_independent"}
        for metric in METRICS:
            summary.update({f"delta_{metric}_{name}": value for name, value in
                            _describe([row[f"delta_{metric}"] for row in selected]).items()})
        effect_summary.append(summary)
    return conditions, effects, effect_summary


def _historical_dir(study_root: Path) -> Path:
    return (study_root / "existing-evidence" / "zero-shot-v1" / "P5_N4_gap25"
            / "analysis" / "service_regularity")


def _ccdf_summaries(ccdf_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    indexed = {(row["actor_structure"], row["value_configuration"], row["training_seed"],
                row["threshold_L"]): row for row in ccdf_rows}
    thresholds = sorted({row["threshold_L"] for row in ccdf_rows})
    condition_rows, paired_rows, paired_summary = [], [], []
    fields = ("event_ccdf_P_L_gt_threshold", "flow_equal_ccdf_P_L_gt_threshold")
    for structure, variant in CONDITIONS:
        for threshold in thresholds:
            result = {"actor_structure": structure, "value_configuration": variant,
                      "threshold_L": threshold, "training_seed_count": 6}
            for field in fields:
                result.update({f"{field}_{name}": value for name, value in
                               _describe([indexed[structure, variant, seed, threshold][field]
                                          for seed in SEEDS]).items()})
            condition_rows.append(result)
    for variant in ("combined", "tdec"):
        for threshold in thresholds:
            at_threshold = []
            for seed in SEEDS:
                independent = indexed["independent", variant, seed, threshold]
                shared = indexed["shared", variant, seed, threshold]
                result = {"value_configuration": variant, "training_seed": seed,
                          "threshold_L": threshold, "delta_definition": "shared_minus_independent"}
                for field in fields:
                    left, right = independent[field], shared[field]
                    result[f"delta_{field}"] = right - left if left is not None and right is not None else None
                paired_rows.append(result)
                at_threshold.append(result)
            summary = {"value_configuration": variant, "threshold_L": threshold,
                       "training_seed_count": 6, "delta_definition": "shared_minus_independent"}
            for field in fields:
                summary.update({f"delta_{field}_{name}": value for name, value in
                                _describe([row[f"delta_{field}"] for row in at_threshold]).items()})
            paired_summary.append(summary)
    return condition_rows, paired_rows, paired_summary


def _historical_check(study_root: Path, flow_rows: list[dict], seed_rows: list[dict]) -> dict:
    directory = _historical_dir(study_root)
    with (directory / "service_regularity_world_agent.csv").open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    with (directory / "service_regularity_summary.csv").open(newline="", encoding="utf-8") as handle:
        old_summaries = list(csv.DictReader(handle))
    old_index = {(row["structure"], int(row["training_seed"]), int(row["eval_seed"]), int(row["agent"])): row
                 for row in old_rows}
    tdec = [row for row in flow_rows if row["value_configuration"] == "tdec"]
    _require(len(tdec), 360, "TDec flow count")
    _require(len(old_index), 360, "historical unique flow count")
    compared = 0
    for row in tdec:
        key = row["actor_structure"], row["training_seed"], row["world"], row["agent"]
        old = old_index[key]
        for new_field, old_field in (("slot_count", "slot_count"), ("reset_count", "reset_count"),
                                     ("complete_interval_count", "complete_interval_count"),
                                     ("sum_L", "complete_interval_sum_L"),
                                     ("sum_L2", "complete_interval_sum_L2"),
                                     ("long_gt100_count", "complete_long_gt100_count")):
            _require(row[new_field], int(old[old_field]), f"historical {key}/{new_field}")
            compared += 1
        if not row["no_reset_flow"]:
            for new_field, old_field in (("left_boundary_visible_wait_slots", "left_censored_slots"),
                                         ("right_boundary_visible_wait_slots", "right_censored_slots")):
                _require(row[new_field], int(old[old_field]), f"historical {key}/{new_field}")
    by_structure = {row["structure"]: row for row in old_summaries}
    _require(set(by_structure), {"independent", "shared"}, "historical summary structures")
    summaries = []
    for structure in ("independent", "shared"):
        old = by_structure[structure]
        selected = [row for row in seed_rows if row["actor_structure"] == structure and row["value_configuration"] == "tdec"]
        count = sum(row["complete_interval_count"] for row in selected)
        sum_l = sum(row["sum_L"] for row in selected)
        flow_means = [row["complete_interval_mean_L"] for row in tdec
                      if row["actor_structure"] == structure and row["complete_interval_mean_L"] is not None]
        _require(count, int(old["pooled_complete_interval_count"]), f"historical {structure}/interval count")
        _require(sum(row["long_gt100_count"] for row in selected), int(old["complete_long_gt100_count"]),
                 f"historical {structure}/long count")
        for label, actual in (("pooled_complete_interval_mean", sum_l / count),
                              ("equal_flow_mean_interval_mean", float(np.mean(flow_means)))):
            if not np.isclose(actual, float(old[label]), rtol=0, atol=1e-10):
                raise ValueError(f"historical {structure}/{label}: {actual} versus {old[label]}")
        summaries.append({"actor_structure": structure, "historical_flow_count": int(old["flow_count"]),
                          "historical_event_mean_L": float(old["pooled_complete_interval_mean"]),
                          "historical_flow_equal_mean_L": float(old["equal_flow_mean_interval_mean"]),
                          "difference_note": "seed-level averages are not historical pooled event/equal-flow summaries; weights differ"})
    return {"status": "PASS", "historical_dir": str(directory), "flow_fields_compared": compared,
            "summary": summaries, "boundary_note": "historical no-reset left/right both M; W1 stores one unique window M"}


def run(study_root: Path, preflight_only: bool = False) -> dict:
    study_root = study_root.resolve()
    output = study_root / "W1-service-intervals" / "P5_N4_gap25"
    if not output.resolve().is_relative_to(study_root):
        raise ValueError(f"output escapes study root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    historical_dir = _historical_dir(study_root)
    for name in ("service_regularity_world_agent.csv", "service_regularity_summary.csv"):
        if not (historical_dir / name).is_file():
            raise ValueError(f"missing historical reconciliation input: {historical_dir / name}")
    inventory, flow_rows, frequency_rows, examples = [], [], [], []
    frequencies: dict[tuple, Counter[int]] = {}
    source_paths = set()
    for structure, variant in CONDITIONS:
        for seed in SEEDS:
            eval_dir, run_name = _source(study_root, structure, variant, seed)
            training_dir = _training_dir(study_root, structure, variant, run_name)
            events, item = _load_events(eval_dir, training_dir, structure, variant, seed, run_name)
            path = item["metrics_npz"]
            if path in source_paths:
                raise ValueError(f"duplicate source NPZ: {path}")
            source_paths.add(path)
            inventory.append(item)
            if preflight_only:
                continue
            for world_index, world in enumerate(EVAL_SEEDS):
                for agent in range(5):
                    row, freq, positions = flow_statistics(events[world_index, :, :, agent])
                    key = structure, variant, seed, world, agent
                    frequencies[key] = freq
                    identity = {"actor_structure": structure, "value_configuration": variant,
                                "training_seed": seed, "world": world, "agent": agent}
                    flow_rows.append({**identity, "event_source": item["event_source"], **row})
                    frequency_rows.extend({**identity, "length_L": length, "frequency": count}
                                          for length, count in sorted(freq.items()))
                    if len(examples) < 8:
                        examples.append({**identity, "first_reset_flat_slot_indices_zero_based": positions,
                                         "first_complete_lengths": list(np.diff(positions))})
    _require(len(inventory), 24, "evaluation cell count")
    _require(len(source_paths), 24, "unique source count")
    _csv(output / "source_inventory.csv", inventory)
    if preflight_only:
        payload = {"status": "PASS", "stage": "preflight", "evaluation_cells": 24,
                   "policy_world_runs": 144, "expected_flows": 720,
                   "event_source_counts": dict(Counter(row["event_source"] for row in inventory)),
                   "note": "Only source/protocol/array/event-consistency checked; no W1 result computed"}
        (output / "preflight.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return payload
    _require(len(flow_rows), 720, "flow count")
    _require(sum(row["slot_count"] for row in flow_rows), 7200000, "agent-slot count")
    for row in flow_rows:
        _require(row["slot_count"], SLOTS_PER_FLOW, "scored slot count")
        key = row["actor_structure"], row["value_configuration"], row["training_seed"], row["world"], row["agent"]
        freq = frequencies[key]
        for field, actual in (("complete_interval_count", sum(freq.values())),
                              ("sum_L", sum(length * count for length, count in freq.items())),
                              ("sum_L2", sum(length**2 * count for length, count in freq.items()))):
            _require(row[field], actual, f"{key}/{field} frequency reconstruction")
        _require(row["complete_interval_count"], max(row["reset_count"] - 1, 0), f"{key}/reset interval identity")
        if row["no_reset_flow"]:
            _require(row["no_reset_window_slots"], SLOTS_PER_FLOW, f"{key}/unique window")
        else:
            _require(row["left_boundary_visible_wait_slots"] + row["sum_L"] +
                     row["right_boundary_visible_wait_slots"], SLOTS_PER_FLOW - 1,
                     f"{key}/boundary identity")
    seed_rows = []
    for structure, variant in CONDITIONS:
        for seed in SEEDS:
            key = structure, variant, seed
            selected = [row for row in flow_rows if (row["actor_structure"], row["value_configuration"],
                                                     row["training_seed"]) == key]
            seed_row, _ = seed_statistics(selected, frequencies, key)
            seed_rows.append(seed_row)
    global_support = sorted({0, 100} | {length for freq in frequencies.values() for length in freq})
    ccdf_rows = []
    for structure, variant in CONDITIONS:
        for seed in SEEDS:
            key = structure, variant, seed
            selected = [row for row in flow_rows if (row["actor_structure"], row["value_configuration"],
                                                     row["training_seed"]) == key]
            ccdf = _ccdf(selected, frequencies, key, global_support)
            ccdf_rows.extend(ccdf)
            for field in ("event_ccdf_P_L_gt_threshold", "flow_equal_ccdf_P_L_gt_threshold"):
                values = [row[field] for row in ccdf if row[field] is not None]
                if any(left + 1e-12 < right for left, right in zip(values, values[1:])):
                    raise ValueError(f"{key}: non-monotone {field}")
    historical = _historical_check(study_root, flow_rows, seed_rows)
    conditions, effects, effect_summary = _summary_rows(seed_rows)
    condition_ccdf, paired_ccdf, paired_ccdf_summary = _ccdf_summaries(ccdf_rows)
    _csv(output / "flow_statistics.csv", flow_rows)
    _csv(output / "flow_interval_frequency.csv", frequency_rows)
    _csv(output / "seed_statistics.csv", seed_rows)
    _csv(output / "seed_ccdf.csv", ccdf_rows)
    _csv(output / "condition_ccdf_summary.csv", condition_ccdf)
    _csv(output / "sharing_ccdf_effect_per_seed.csv", paired_ccdf)
    _csv(output / "sharing_ccdf_effect_summary.csv", paired_ccdf_summary)
    _csv(output / "condition_summary.csv", conditions)
    _csv(output / "sharing_effect_per_seed.csv", effects)
    _csv(output / "sharing_effect_summary.csv", effect_summary)
    verification = {"status": "PASS", "scope": "technical_source_and_calculation_only",
                    "evaluation_cells": 24, "unique_evaluation_cells": 24,
                    "condition_cell_counts": {f"{s}/{v}": sum(item["actor_structure"] == s and
                                                              item["value_configuration"] == v for item in inventory)
                                              for s, v in CONDITIONS},
                    "policy_world_runs": 144, "flow_rows": 720, "agent_slots": 7200000,
                    "condition_seed_rows": 24, "seed_ccdf_rows": len(ccdf_rows),
                    "frequency_rows": len(frequency_rows), "historical_reconciliation": historical,
                    "checks": ["protocol_and_shape", "event_field_consistency", "source_deduplication",
                               "frequency_moments", "reset_boundary_identity", "ccdf_monotonicity",
                               "historical_tdec_dev"], "new_training": 0, "new_evaluations": 0}
    (output / "index_examples.json").write_text(json.dumps(examples, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = ["# W1 更新间隔：技术后处理", "", "仅复用开发世界 213–218 的冻结评估轨迹；无训练、无策略评估。",
              "两种价值学习配置为 combined/TDec；此处只描述观察到的 V2I 复位更新过程。", "",
              "每条流按连续 scored episodes 展平，world 间不连接。完整间隔为相邻复位槽索引之差；边界另列。",
              "事件加权按完整间隔频数；流等权只对有完整间隔的流平均，并显式报告覆盖数。",
              "这两种分布均是有限观察窗内的完整间隔分布，不能当作总体无偏等待分布。", "",
              "| actor | value | 有效流/总流 | seed均值：事件 L | seed均值：流等权 L | seed均值：>100/万槽 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in conditions:
        report.append(f"| {row['actor_structure']} | {row['value_configuration']} | {row['eligible_flow_count']}/{row['flow_count']} | "
                      f"{row['event_mean_L_mean']} | {row['flow_equal_mean_L_mean']} | "
                      f"{row['long_gt100_per_10000_slots_mean']} |")
    report += ["", "技术 PASS 仅说明来源和计算验收，不要求共享优于独立。",
               "历史 TDec/dev 对账按逐流整数矩和旧 pooled/equal-flow 定义完成；seed 均值与旧全流 pooled 数不可直接等同。", ""]
    (output / "report_cn.md").write_text("\n".join(report), encoding="utf-8")
    (output / "fields_and_boundaries.md").write_text(FIELD_NOTES, encoding="utf-8")
    (output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return verification


FIELD_NOTES = """# W1 字段、权重与边界

- `actor_structure` 为 independent/shared；`value_configuration` 为 combined/TDec 两种价值学习配置；`training_seed` 8–13 是独立重复，`world` 213–218 与 `agent` 0–4 是嵌套观察。
- `reset_event` 优先，若缺失则用逐槽 post-step `aoi_ms` 与 1 的 `isclose(rtol=0,atol=1e-6)`；两字段都有时要求逐元素一致。`event_source` 逐流记录。连续复位会产生 L=1，AoI cap 不截断 L。
- 源数组的 `scored_episode` 已排除五个 warmup episodes。每世界按 episode/slot 展平，M=10000；不跨 world，不先筛 mode/V2V。
- 完整间隔 `L=diff(reset_flat_slot_index)`；`complete_interval_count=max(reset_count-1,0)`，`sum_L` 与 `sum_L2` 为整数矩。流均值和 CV² 只在有完整间隔时定义，CV²=`(sum_L2/n)/(sum_L/n)^2-1`。
- 左边界可见等待为首个复位的零基索引，右边界为 `M-1-末次复位索引`；它们不是完整间隔。无复位时左右留空，仅 `no_reset_window_slots=M`，避免重复记成 2M；单复位保留左右但无完整间隔。
- `flow_interval_frequency.csv` 的 `length_L,frequency` 可准确还原完整间隔分布；无完整间隔的流无频数行，但保留在 `flow_statistics.csv` 与覆盖计数。
- `seed_ccdf.csv` 的 `P(L>threshold_L)` 含阈值 0、100 和所有观测长度。事件加权分母是该 seed 30 流中全部完整间隔数；流等权分母是该 seed 有完整间隔的流数，每流先求比例。未覆盖流单列，不宣称全流覆盖。
- `long_gt100_count` 是严格 L>100 的完整间隔数；同时给出每 10000 agent-slots 频度和在完整间隔中的占比，不随结果改阈值。
- 条件和共享效应先按 seed 聚合，再以六个 seed 求 mean、样本 SD 与描述性 95% t 区间（df=5，t*=2.570581835636314）。流内 CV²、流均值之间 CV² 只是描述，不可相加或作因果分解；不筛 screen_success，不做显著性星号。
- 技术 PASS 只代表来源与计算通过。reset 间隔不可解释 CAM 完成机制，也不可推断服务再分配因果机制。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.study_root, args.preflight_only)
    except (OSError, KeyError, ValueError, AssertionError) as error:
        failure = {"status": "FAIL", "stage": "preflight" if args.preflight_only else "analysis",
                   "error": str(error)}
        output = args.study_root.resolve() / "W1-service-intervals" / "P5_N4_gap25"
        if output.resolve().is_relative_to(args.study_root.resolve()):
            output.mkdir(parents=True, exist_ok=True)
            target = output / ("preflight.json" if args.preflight_only else "verification.json")
            target.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
