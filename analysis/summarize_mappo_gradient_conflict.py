"""Summarize the diagnostic-only TDec-MAPPO objective-gradient audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


SEEDS = (8, 9, 10, 11, 12, 13)
SCENARIO = "p05_n04_g25"
OBJECTIVES = ("global", "task1", "task2")
PAIRS = ("global_task1", "global_task2", "task1_task2")
BLOCKS = ("trunk", "rb_head", "mode_head", "power_head")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _run_name(seed: int) -> str:
    return f"mappo_gradient_conflict_{SCENARIO}_seed{seed:02d}"


def _validate_run(run_dir: Path, seed: int, config, complete, learning) -> None:
    checks = {
        "algorithm": (config.get("algorithm"), "mappo"),
        "scenario": (config.get("scenario", {}).get("id"), SCENARIO),
        "seed": (int(config.get("seed", -1)), seed),
        "episodes": (int(config.get("episodes", -1)), 500),
        "mappo_variant": (config.get("mappo_variant"), "tdec"),
        "mappo_actor_lr": (float(config.get("mappo_actor_lr", np.nan)), 0.0005),
        "mappo_value_clip_mode": (config.get("mappo_value_clip_mode"), "normalized"),
        "mappo_objective_gradient_diagnostics": (
            config.get("mappo_objective_gradient_diagnostics"), True
        ),
    }
    for name, (actual, expected) in checks.items():
        if isinstance(expected, float):
            matches = np.isclose(actual, expected, rtol=0.0, atol=1e-12)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"{run_dir.name}: {name}={actual!r}, expected {expected!r}")
    for key, expected in (
        ("mappo_entropy_coef_rb", 0.02),
        ("mappo_entropy_coef_mode", 0.02),
        ("mappo_entropy_coef_power", 0.002),
    ):
        if not np.isclose(float(config.get(key, np.nan)), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"{run_dir.name}: {key} mismatch")
    if complete.get("status") != "complete" or int(complete.get("update_count", -1)) != 100:
        raise ValueError(f"{run_dir.name}: incomplete training")
    if len(learning) != 100:
        raise ValueError(f"{run_dir.name}: expected 100 PPO updates")


def _flatten_record(seed: int, update_record, gradient_record) -> Dict[str, object]:
    agent = int(gradient_record["agent"])
    row: Dict[str, object] = {
        "seed": seed,
        "update": int(update_record["update"]),
        "episode": int(update_record["episode"]),
        "ppo_epoch": int(gradient_record["ppo_epoch"]),
        "agent": agent,
        "cancellation_ratio": float(gradient_record["cancellation_ratio"]),
        "cancellation_valid": bool(gradient_record["cancellation_valid"]),
        "adv_corr_global_task1": float(update_record["global_task1_advantage_correlation_per_agent"][agent]),
        "adv_corr_global_task2": float(update_record["global_task2_advantage_correlation_per_agent"][agent]),
        "adv_corr_task1_task2": float(update_record["task1_task2_advantage_correlation_per_agent"][agent]),
    }
    for objective in OBJECTIVES:
        row[f"{objective}_grad_norm"] = float(gradient_record["objective_grad_norm"][objective])
        row[f"{objective}_policy_loss"] = float(gradient_record["objective_policy_loss"][objective])
        row[f"{objective}_effective_clip_fraction"] = float(
            gradient_record["effective_clip_fraction"][objective]
        )
    for pair in PAIRS:
        geometry = gradient_record["pairs"][pair]
        row[f"{pair}_dot"] = float(geometry["dot"])
        row[f"{pair}_cosine"] = float(geometry["cosine"])
        row[f"{pair}_valid"] = bool(geometry["valid"])
        row[f"{pair}_conflict"] = bool(geometry["conflict"])
        for block in BLOCKS:
            block_geometry = gradient_record["block_pairs"][block][pair]
            row[f"{block}_{pair}_cosine"] = float(block_geometry["cosine"])
            row[f"{block}_{pair}_valid"] = bool(block_geometry["valid"])
    numeric = [value for value in row.values() if isinstance(value, (int, float))]
    if not np.all(np.isfinite(np.asarray(numeric, dtype=np.float64))):
        raise FloatingPointError("non-finite flattened objective-gradient diagnostic")
    return row


def _mean_valid(rows, value_key: str, valid_key: str) -> float:
    values = [float(row[value_key]) for row in rows if bool(row[valid_key])]
    return float(np.mean(values)) if values else 0.0


def _conflict_rate(rows, pair: str) -> float:
    valid = [row for row in rows if bool(row[f"{pair}_valid"])]
    return float(np.mean([bool(row[f"{pair}_conflict"]) for row in valid])) if valid else 0.0


def _summarize_seed(seed: int, rows, train_summary) -> Dict[str, object]:
    result: Dict[str, object] = {
        "seed": seed,
        "diagnostic_rows": len(rows),
        "mean_cancellation_ratio": _mean_valid(rows, "cancellation_ratio", "cancellation_valid"),
        "last100_mean_aoi_ms": float(train_summary["final_window_mean_AoI_ms"]),
        "last100_worst_agent_aoi_ms": float(train_summary["final_window_worst_agent_mean_AoI_ms"]),
        "last100_mean_binary_cam": float(train_summary["final_window_endpoint_CAM_probability"]),
        "last100_worst_agent_binary_cam": float(
            train_summary["final_window_worst_agent_endpoint_CAM_probability"]
        ),
    }
    for objective in OBJECTIVES:
        result[f"mean_{objective}_grad_norm"] = float(
            np.mean([row[f"{objective}_grad_norm"] for row in rows])
        )
        result[f"mean_{objective}_effective_clip_fraction"] = float(
            np.mean([row[f"{objective}_effective_clip_fraction"] for row in rows])
        )
    for pair in PAIRS:
        result[f"{pair}_valid_fraction"] = float(np.mean([row[f"{pair}_valid"] for row in rows]))
        result[f"{pair}_mean_cosine"] = _mean_valid(rows, f"{pair}_cosine", f"{pair}_valid")
        result[f"{pair}_conflict_rate"] = _conflict_rate(rows, pair)
        result[f"{pair}_mean_advantage_correlation"] = float(
            np.mean([row[f"adv_corr_{pair}"] for row in rows])
        )
        for block in BLOCKS:
            result[f"{block}_{pair}_mean_cosine"] = _mean_valid(
                rows, f"{block}_{pair}_cosine", f"{block}_{pair}_valid"
            )
    for epoch in (0, 9):
        selected = [row for row in rows if row["ppo_epoch"] == epoch]
        if not selected:
            raise ValueError(f"seed {seed}: missing PPO epoch {epoch} diagnostics")
        prefix = f"epoch{epoch}"
        result[f"{prefix}_mean_cancellation_ratio"] = _mean_valid(
            selected, "cancellation_ratio", "cancellation_valid"
        )
        for objective in OBJECTIVES:
            result[f"{prefix}_mean_{objective}_effective_clip_fraction"] = float(
                np.mean([row[f"{objective}_effective_clip_fraction"] for row in selected])
            )
        for pair in PAIRS:
            result[f"{prefix}_{pair}_valid_fraction"] = float(
                np.mean([row[f"{pair}_valid"] for row in selected])
            )
            result[f"{prefix}_{pair}_mean_cosine"] = _mean_valid(
                selected, f"{pair}_cosine", f"{pair}_valid"
            )
            result[f"{prefix}_{pair}_conflict_rate"] = _conflict_rate(selected, pair)
    return result


def summarize(result_root: Path) -> Dict[str, object]:
    result_root = result_root.expanduser().resolve()
    per_update: List[Dict[str, object]] = []
    per_seed: List[Dict[str, object]] = []
    for seed in SEEDS:
        run_dir = result_root / "training" / "runs" / _run_name(seed)
        config = _read_json(run_dir / "config.resolved.json")
        complete = _read_json(run_dir / "COMPLETE.json")
        learning = _read_json(run_dir / "learning_diagnostics.json")
        train_summary = _read_json(run_dir / "train_metrics_summary.json")
        _validate_run(run_dir, seed, config, complete, learning)
        seed_rows: List[Dict[str, object]] = []
        for update_record in learning:
            diagnostic = update_record.get("objective_gradient_diagnostics")
            if not isinstance(diagnostic, dict):
                raise ValueError(f"{run_dir.name}: update lacks objective-gradient diagnostics")
            if diagnostic.get("schema_version") != "mappo_objective_gradient_v1":
                raise ValueError(f"{run_dir.name}: unexpected objective-gradient schema")
            records = diagnostic.get("records")
            if not isinstance(records, list) or len(records) != 10:
                raise ValueError(f"{run_dir.name}: expected first/last epoch for all five agents")
            seed_rows.extend(_flatten_record(seed, update_record, record) for record in records)
        if len(seed_rows) != 1000:
            raise ValueError(f"{run_dir.name}: expected 1000 diagnostic rows")
        per_update.extend(seed_rows)
        per_seed.append(_summarize_seed(seed, seed_rows, train_summary))

    cohort: Dict[str, object] = {
        "seeds_complete": len(per_seed),
        "diagnostic_rows": len(per_update),
        "mean_last100_aoi_ms": float(np.mean([row["last100_mean_aoi_ms"] for row in per_seed])),
        "mean_last100_binary_cam": float(np.mean([row["last100_mean_binary_cam"] for row in per_seed])),
        "mean_cancellation_ratio": float(np.mean([row["mean_cancellation_ratio"] for row in per_seed])),
    }
    for pair in PAIRS:
        cohort[f"{pair}_mean_seed_conflict_rate"] = float(
            np.mean([row[f"{pair}_conflict_rate"] for row in per_seed])
        )
        cohort[f"{pair}_mean_seed_cosine"] = float(
            np.mean([row[f"{pair}_mean_cosine"] for row in per_seed])
        )
        cohort[f"{pair}_seeds_majority_conflict"] = int(
            sum(row[f"{pair}_conflict_rate"] > 0.5 for row in per_seed)
        )
    cohort["by_ppo_epoch"] = []
    for epoch in (0, 9):
        prefix = f"epoch{epoch}"
        epoch_row: Dict[str, object] = {
            "ppo_epoch": epoch,
            "position": "first" if epoch == 0 else "last",
            "mean_seed_cancellation_ratio": float(np.mean([
                row[f"{prefix}_mean_cancellation_ratio"] for row in per_seed
            ])),
        }
        for objective in OBJECTIVES:
            epoch_row[f"mean_seed_{objective}_effective_clip_fraction"] = float(np.mean([
                row[f"{prefix}_mean_{objective}_effective_clip_fraction"] for row in per_seed
            ]))
        for pair in PAIRS:
            epoch_row[f"mean_seed_{pair}_cosine"] = float(np.mean([
                row[f"{prefix}_{pair}_mean_cosine"] for row in per_seed
            ]))
            epoch_row[f"mean_seed_{pair}_conflict_rate"] = float(np.mean([
                row[f"{prefix}_{pair}_conflict_rate"] for row in per_seed
            ]))
            epoch_row[f"seeds_majority_{pair}_conflict"] = int(sum(
                row[f"{prefix}_{pair}_conflict_rate"] > 0.5 for row in per_seed
            ))
        cohort["by_ppo_epoch"].append(epoch_row)
    return {
        "contract": {
            "algorithm": "mappo",
            "variant": "tdec",
            "scenario": SCENARIO,
            "seeds": list(SEEDS),
            "episodes": 500,
            "actor_lr": 0.0005,
            "entropy": {"rb": 0.02, "mode": 0.02, "power": 0.002},
            "value_clip_mode": "normalized",
            "diagnostic_epochs": [0, 9],
            "actor_update_changed": False,
        },
        "per_update": per_update,
        "per_seed": per_seed,
        "cohort": cohort,
    }


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(result_root: Path, report: Dict[str, object]) -> Path:
    output = result_root.expanduser().resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "gradient_conflict_per_update.csv", report["per_update"])
    _write_csv(output / "gradient_conflict_per_seed.csv", report["per_seed"])
    compact = {key: value for key, value in report.items() if key != "per_update"}
    (output / "gradient_conflict_summary.json").write_text(
        json.dumps(compact, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# TDec-MAPPO objective-gradient audit",
        "",
        "This is a diagnostic-only run. The composed PPO actor update is unchanged.",
        "Task1/task2 are the existing reward streams, not relabelled pure CAM/AoI objectives.",
        "",
        "| seed | AoI | binary CAM | cancellation | g-task1 conflict | g-task2 conflict | task1-task2 conflict |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['last100_mean_aoi_ms']:.3f} | {row['last100_mean_binary_cam']:.4f} "
            f"| {row['mean_cancellation_ratio']:.3f} | {row['global_task1_conflict_rate']:.3f} "
            f"| {row['global_task2_conflict_rate']:.3f} | {row['task1_task2_conflict_rate']:.3f} |"
        )
    cohort = report["cohort"]
    lines.extend([
        "",
        "## Cohort",
        "",
        f"- Complete seeds: {cohort['seeds_complete']}/6",
        f"- Diagnostic rows: {cohort['diagnostic_rows']}",
        f"- Mean last-100 AoI/CAM: {cohort['mean_last100_aoi_ms']:.3f}/{cohort['mean_last100_binary_cam']:.4f}",
        f"- Mean per-seed cancellation ratio: {cohort['mean_cancellation_ratio']:.3f}",
        "- Conflict/performance associations are descriptive and do not by themselves establish causality.",
        "",
        "## First versus last PPO epoch",
        "",
        "| position | epoch | cancellation | g-task1 conflict | g-task2 conflict | task1-task2 conflict | global clip | task1 clip | task2 clip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in cohort["by_ppo_epoch"]:
        lines.append(
            f"| {row['position']} | {row['ppo_epoch']} | {row['mean_seed_cancellation_ratio']:.3f} "
            f"| {row['mean_seed_global_task1_conflict_rate']:.3f} "
            f"| {row['mean_seed_global_task2_conflict_rate']:.3f} "
            f"| {row['mean_seed_task1_task2_conflict_rate']:.3f} "
            f"| {row['mean_seed_global_effective_clip_fraction']:.3f} "
            f"| {row['mean_seed_task1_effective_clip_fraction']:.3f} "
            f"| {row['mean_seed_task2_effective_clip_fraction']:.3f} |"
        )
    (output / "gradient_conflict_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.result_root)
    output = write_report(args.result_root, report)
    print(json.dumps({"output": str(output), "cohort": report["cohort"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
