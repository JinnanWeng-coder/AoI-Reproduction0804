"""Summarize the 30-cell MAPPO actor-only zero-shot experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.evaluate_mappo_feasibility_reward import feasibility_eval_id
from analysis.evaluate_mappo_zero_shot import DEFAULT_EVAL_SEEDS, TRAINING_SEEDS, zero_shot_eval_id
from analysis.summarize_mappo_shared_actor import training_run_name


EVAL_EPISODES = 100
WARMUP = 5
TARGET_STRUCTURES = (
    ("p05_n04_g05", "independent"),
    ("p05_n04_g05", "shared"),
    ("p05_n04_g35", "independent"),
    ("p05_n04_g35", "shared"),
    ("p07_n04_g25", "shared"),
)
METRICS = (
    "mean_aoi_ms", "worst_agent_aoi_ms", "aoi_gt50_fraction", "aoi_at_cap_fraction",
    "mean_binary_cam", "worst_agent_binary_cam", "mean_payload_completion",
    "worst_agent_payload_completion", "mean_reward_global", "mean_reward_task1",
    "mean_reward_task2", "mean_reward_combined", "mean_power_mw", "same_rb_pair_fraction",
)


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


def _same_rb_pair_fraction(rb: np.ndarray) -> float:
    agents = rb.shape[-1]
    values = [rb[..., left] == rb[..., right] for left in range(agents) for right in range(left + 1, agents)]
    return float(np.stack(values, axis=-1).mean()) if values else 0.0


def _metrics(arrays: Mapping[str, np.ndarray], cam_bits: float) -> dict:
    aoi = arrays["aoi_ms"].astype(np.float64)
    cam = arrays["success"][:, :, -1, :].astype(np.float64)
    payload = np.clip(1.0 - arrays["remaining_demand"][:, :, -1, :] / float(cam_bits), 0.0, 1.0)
    paoi = aoi.mean(axis=(0, 1, 2))
    pcam = cam.mean(axis=(0, 1))
    ppayload = payload.mean(axis=(0, 1))
    return {
        "mean_aoi_ms": float(paoi.mean()),
        "worst_agent_aoi_ms": float(paoi.max()),
        "aoi_gt50_fraction": float((aoi > 50.0).mean()),
        "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, atol=1e-6, rtol=0.0).mean()),
        "mean_binary_cam": float(pcam.mean()),
        "worst_agent_binary_cam": float(pcam.min()),
        "mean_payload_completion": float(ppayload.mean()),
        "worst_agent_payload_completion": float(ppayload.min()),
        "mean_reward_global": float(arrays["reward_global"].mean()),
        "mean_reward_task1": float(arrays["reward_task1"].mean()),
        "mean_reward_task2": float(arrays["reward_task2"].mean()),
        "mean_reward_combined": float(arrays["reward_combined"].mean()),
        "mean_power_mw": float(np.power(10.0, arrays["executed_power_dbm"] / 10.0).mean()),
        "same_rb_pair_fraction": _same_rb_pair_fraction(arrays["rb"]),
        "screen_success": bool(paoi.max() < 50.0 and pcam.min() >= 0.5),
    }


def _stream_rows(arrays, complete, target, structure, seed):
    rows = []
    cam_bits = float(complete["cam_bits"])
    weight = float(complete["global_actor_weight"])
    n_agents = int(complete["target_number_agents"])
    for world_index, world in enumerate(DEFAULT_EVAL_SEEDS):
        for agent in range(n_agents):
            aoi = arrays["aoi_ms"][world_index, :, :, agent]
            cam = arrays["success"][world_index, :, -1, agent]
            remaining = arrays["remaining_demand"][world_index, :, -1, agent]
            global_reward = arrays["reward_global"][world_index]
            task1 = arrays["reward_task1"][world_index, :, :, agent]
            task2 = arrays["reward_task2"][world_index, :, :, agent]
            rb = arrays["rb"][world_index, :, :, agent]
            peers = np.delete(arrays["rb"][world_index], agent, axis=-1)
            rows.append({
                "target_scenario": target, "structure": structure, "training_seed": seed,
                "eval_seed": world, "agent": agent, "is_new_p7_agent": bool(agent >= 5),
                "mean_aoi_ms": float(aoi.mean()), "aoi_gt50_fraction": float((aoi > 50).mean()),
                "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, atol=1e-6, rtol=0.0).mean()),
                "binary_cam": float(cam.mean()),
                "payload_completion": float(np.clip(1.0 - remaining / cam_bits, 0.0, 1.0).mean()),
                "mean_reward_global": float(global_reward.mean()), "mean_reward_task1": float(task1.mean()),
                "mean_reward_task2": float(task2.mean()),
                "mean_reward_combined": float((task1 + task2 + weight * global_reward).mean()),
                "mean_power_mw": float(np.power(10.0, arrays["executed_power_dbm"][world_index, :, :, agent] / 10.0).mean()),
                "same_rb_peer_fraction": float((peers == rb[..., None]).mean()),
            })
    return rows


def _load_cell(eval_dir: Path, target: str, structure: str, seed: int):
    complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
    expected_agents = 7 if target == "p07_n04_g25" else 5
    checks = {
        "status": "complete", "source_scenario": "p05_n04_g25", "target_scenario": target,
        "actor_structure": structure, "training_seed": seed, "target_number_agents": expected_agents,
        "eval_seeds": list(DEFAULT_EVAL_SEEDS), "eval_episodes": EVAL_EPISODES,
        "eval_warmup_episodes": WARMUP, "mappo_eval_mode": "stochastic",
        "action_intervention": "none", "policy_parameters_unchanged": True,
    }
    for key, expected in checks.items():
        if complete.get(key) != expected:
            raise ValueError(f"{eval_dir}: {key}={complete.get(key)!r}, expected {expected!r}")
    with np.load(eval_dir / "metrics.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    expected_shape = (6, 100, 100, expected_agents)
    if arrays["aoi_ms"].shape != expected_shape:
        raise ValueError(f"{eval_dir}: unexpected trajectory shape {arrays['aoi_ms'].shape}")
    return complete, arrays


def _reference_cell(shared_root: Path, structure: str, seed: int):
    run_name = training_run_name(structure, seed)
    eval_id = feasibility_eval_id("baseline", DEFAULT_EVAL_SEEDS, EVAL_EPISODES, WARMUP)
    eval_dir = shared_root / "evaluations" / run_name / eval_id
    complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
    with np.load(eval_dir / "metrics.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return _metrics(arrays, float(complete["cam_bits"]))


def summarize(zero_root: Path, shared_root: Path) -> dict:
    zero_root, shared_root = zero_root.resolve(), shared_root.resolve()
    cells, streams = [], []
    references = {}
    for structure in ("independent", "shared"):
        for seed in TRAINING_SEEDS:
            references[(structure, seed)] = _reference_cell(shared_root, structure, seed)
    for target, structure in TARGET_STRUCTURES:
        for seed in TRAINING_SEEDS:
            run_name = training_run_name(structure, seed)
            eval_dir = zero_root / target / structure / run_name / zero_shot_eval_id(target)
            complete, arrays = _load_cell(eval_dir, target, structure, seed)
            row = {
                "target_scenario": target, "structure": structure, "training_seed": seed,
                "training_run_name": run_name, "target_number_agents": int(complete["target_number_agents"]),
                **_metrics(arrays, float(complete["cam_bits"])),
            }
            cells.append(row)
            streams.extend(_stream_rows(arrays, complete, target, structure, seed))

    indexed = {(r["target_scenario"], r["structure"], r["training_seed"]): r for r in cells}
    cross_structure, reference_deltas = [], []
    for target in ("p05_n04_g05", "p05_n04_g35"):
        for seed in TRAINING_SEEDS:
            ind, shared = indexed[(target, "independent", seed)], indexed[(target, "shared", seed)]
            row = {"target_scenario": target, "training_seed": seed, "delta_definition": "shared_minus_independent"}
            for metric in METRICS:
                row[f"delta_{metric}"] = float(shared[metric]) - float(ind[metric])
            cross_structure.append(row)
            for structure, current in (("independent", ind), ("shared", shared)):
                reference = references[(structure, seed)]
                delta = {"target_scenario": target, "structure": structure, "training_seed": seed,
                         "reference_scenario": "p05_n04_g25", "delta_definition": "target_minus_gap25_reference"}
                for metric in METRICS:
                    delta[f"delta_{metric}"] = float(current[metric]) - float(reference[metric])
                reference_deltas.append(delta)

    summaries = []
    for target, structure in TARGET_STRUCTURES:
        selected = [r for r in cells if r["target_scenario"] == target and r["structure"] == structure]
        summary = {"target_scenario": target, "structure": structure, "training_seed_count": 6,
                   "screen_success_count": sum(bool(r["screen_success"]) for r in selected)}
        for metric in METRICS:
            values = np.asarray([r[metric] for r in selected], dtype=np.float64)
            summary[metric] = float(values.mean())
            summary[f"{metric}_sd_across_training_seeds"] = float(values.std(ddof=1))
        summaries.append(summary)
    return {
        "status": "PASS", "evaluation_cells": len(cells), "world_agent_rows": len(streams),
        "contract": {"source_scenario": "p05_n04_g25", "targets": [list(x) for x in TARGET_STRUCTURES],
                     "training_seeds": list(TRAINING_SEEDS), "eval_seeds": list(DEFAULT_EVAL_SEEDS),
                     "worlds_are_reused": True, "new_world_confirmation": False},
        "scenario_summary": summaries, "per_seed": cells, "world_agent": streams,
        "shared_minus_independent": cross_structure, "target_minus_gap25_reference": reference_deltas,
        "p7_new_agents": [r for r in streams if r["target_scenario"] == "p07_n04_g25" and r["agent"] >= 5],
    }


def write_report(zero_root: Path, report: Mapping) -> Path:
    output = zero_root.resolve() / "analysis" / "zero_shot"
    output.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("zero_shot_per_seed.csv", "per_seed"), ("zero_shot_world_agent.csv", "world_agent"),
        ("zero_shot_summary.csv", "scenario_summary"),
        ("zero_shot_shared_minus_independent.csv", "shared_minus_independent"),
        ("zero_shot_target_minus_gap25.csv", "target_minus_gap25_reference"),
        ("zero_shot_p7_agents5_6.csv", "p7_new_agents"),
    ):
        _write_csv(output / name, report[key])
    compact = {k: v for k, v in report.items() if k not in {"world_agent", "p7_new_agents"}}
    (output / "zero_shot_summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# MAPPO actor-only zero-shot evaluation", "",
             "Policies were trained only at P5/N4/gap25. Worlds 213--218 are reused diagnostic worlds, not a new-world confirmation.", "",
             "| target | actor | AoI mean±SD | worst AoI | binary CAM mean/worst | payload mean/worst | reward | power mW | same-RB | screen |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["scenario_summary"]:
        lines.append(
            f"| {row['target_scenario']} | {row['structure']} | {row['mean_aoi_ms']:.3f}±{row['mean_aoi_ms_sd_across_training_seeds']:.3f} "
            f"| {row['worst_agent_aoi_ms']:.3f} | {row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} "
            f"| {row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} "
            f"| {row['mean_reward_combined']:.4f} | {row['mean_power_mw']:.3f} | {row['same_rb_pair_fraction']:.4f} | {row['screen_success_count']}/6 |"
        )
    lines += ["", "P7 has no independent-actor comparator. Agents 5 and 6 are retained in a separate CSV. Same-RB reuse is diagnostic, not an objective.", ""]
    (output / "zero_shot_report.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-root", required=True, type=Path)
    parser.add_argument("--shared-reference-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.zero_root, args.shared_reference_root)
    output = write_report(args.zero_root, report)
    print(json.dumps({"status": report["status"], "output": str(output), "evaluation_cells": report["evaluation_cells"],
                      "world_agent_rows": report["world_agent_rows"], "scenario_summary": report["scenario_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
