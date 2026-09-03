"""Validate and summarize the three-arm MAPPO gradient-conflict Phase 2 run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ARMS = ("composed_clip", "separate_sum_clip", "pcgrad")
SEEDS = (8, 9, 10, 11, 12, 13)
MODES = ("deterministic", "stochastic")
EVAL_SEEDS = (201, 202, 203, 204, 205, 206)
OBJECTIVES = ("global", "task1", "task2")
PAIRS = ("global_task1", "global_task2", "task1_task2")
BLOCKS = ("trunk", "rb_head", "mode_head", "power_head")
SCENARIO = "p05_n04_g25"


def _run_name(arm: str, seed: int) -> str:
    return f"mappo_gradient_phase2_{arm}_{SCENARIO}_seed{seed:02d}"


def _eval_id(mode: str) -> str:
    return f"eval_validation_policy_final_{mode}_sequential_warm_warm5_s201-202-203-204-205-206_ep100"


def _read_object(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_config(config, complete, arm: str, seed: int, run_dir: Path) -> None:
    exact = {
        "algorithm": "mappo",
        "episodes": 500,
        "seed": seed,
        "mappo_variant": "tdec",
        "mappo_actor_update_mode": arm,
        "mappo_rollout_episodes": 5,
        "mappo_ppo_epochs": 10,
        "mappo_value_clip_mode": "normalized",
        "mappo_objective_gradient_diagnostics": True,
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"{run_dir.name}: {key}={config.get(key)!r}, expected {expected!r}")
    if config.get("scenario", {}).get("id") != SCENARIO:
        raise ValueError(f"{run_dir.name}: scenario mismatch")
    for key, expected in (
        ("mappo_actor_lr", 0.0005),
        ("mappo_critic_lr", 0.0005),
        ("mappo_entropy_coef_rb", 0.02),
        ("mappo_entropy_coef_mode", 0.02),
        ("mappo_entropy_coef_power", 0.002),
    ):
        if not np.isclose(float(config.get(key)), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"{run_dir.name}: {key} mismatch")
    for key, expected in (
        ("status", "complete"),
        ("algorithm", "mappo"),
        ("mappo_variant", "tdec"),
        ("mappo_actor_update_mode", arm),
        ("scenario", SCENARIO),
        ("seed", seed),
        ("update_count", 100),
        ("policy_final", "policy_final.pt"),
    ):
        if complete.get(key) != expected:
            raise ValueError(f"{run_dir.name}: COMPLETE {key} mismatch")


def _mean(values):
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    return float(array.mean()) if array.size else None


def _format_optional(value, digits: int) -> str:
    """Format undefined tiny-gradient geometry without aborting the report."""

    return "NA" if value is None else f"{float(value):.{digits}f}"


def _geometry_fields(records, source: str):
    fields = {}
    geometries = []
    for record in records:
        if source == "pre":
            geometries.append(record)
        else:
            projection = record.get("pcgrad_projection")
            if not isinstance(projection, dict):
                continue
            geometries.append(projection["after"])
    if not geometries:
        for objective in OBJECTIVES:
            fields[f"{source}_{objective}_grad_norm"] = None
        fields[f"{source}_aggregate_grad_norm"] = None
        fields[f"{source}_cancellation_ratio"] = None
        for pair in PAIRS:
            fields[f"{source}_{pair}_cosine"] = None
            fields[f"{source}_{pair}_conflict_rate"] = None
        for block in BLOCKS:
            for pair in PAIRS:
                fields[f"{source}_{block}_{pair}_conflict_rate"] = None
        return fields
    for objective in OBJECTIVES:
        fields[f"{source}_{objective}_grad_norm"] = _mean(
            geometry["objective_grad_norm"][objective] for geometry in geometries
        )
    fields[f"{source}_aggregate_grad_norm"] = _mean(
        geometry["aggregate_grad_norm"] for geometry in geometries
    )
    fields[f"{source}_cancellation_ratio"] = _mean(
        geometry["cancellation_ratio"] for geometry in geometries if geometry["cancellation_valid"]
    )
    for pair in PAIRS:
        valid = [geometry["pairs"][pair] for geometry in geometries if geometry["pairs"][pair]["valid"]]
        fields[f"{source}_{pair}_cosine"] = _mean(value["cosine"] for value in valid)
        fields[f"{source}_{pair}_conflict_rate"] = _mean(float(value["conflict"]) for value in valid)
        for block in BLOCKS:
            valid_block = [
                geometry["block_pairs"][block][pair]
                for geometry in geometries
                if geometry["block_pairs"][block][pair]["valid"]
            ]
            fields[f"{source}_{block}_{pair}_conflict_rate"] = _mean(
                float(value["conflict"]) for value in valid_block
            )
    return fields


def _gradient_rows(arm: str, seed: int, diagnostics):
    rows = []
    selected_updates = diagnostics[-20:]
    for epoch in (0, 9):
        records = []
        for update in selected_updates:
            audit = update.get("objective_gradient_diagnostics")
            if not isinstance(audit, dict) or audit.get("actor_update_mode") != arm:
                raise ValueError(f"{arm}/seed{seed}: invalid objective diagnostic")
            if len(audit.get("records", [])) != 10:
                raise ValueError(f"{arm}/seed{seed}: incomplete objective diagnostic")
            records.extend(record for record in audit["records"] if record.get("ppo_epoch") == epoch)
        if len(records) != 20 * 5:
            raise ValueError(f"{arm}/seed{seed}: incomplete last-20 epoch {epoch} diagnostics")
        row = {"arm": arm, "seed": seed, "ppo_epoch": epoch, **_geometry_fields(records, "pre")}
        row.update(_geometry_fields(records, "post"))
        if arm == "pcgrad":
            projections = [record["pcgrad_projection"] for record in records]
            row["aggregate_projection_magnitude"] = _mean(
                value["aggregate_projection_magnitude"] for value in projections
            )
            for objective in OBJECTIVES:
                row[f"{objective}_projection_magnitude"] = _mean(
                    value["projection_magnitude"][objective] for value in projections
                )
                row[f"{objective}_relative_projection_magnitude"] = _mean(
                    value["relative_projection_magnitude"][objective] for value in projections
                )
                row[f"{objective}_projection_count"] = _mean(
                    value["projection_count"][objective] for value in projections
                )
        else:
            row["aggregate_projection_magnitude"] = None
            for objective in OBJECTIVES:
                row[f"{objective}_projection_magnitude"] = None
                row[f"{objective}_relative_projection_magnitude"] = None
                row[f"{objective}_projection_count"] = None
        numeric = [value for value in row.values() if isinstance(value, (int, float))]
        if not np.isfinite(numeric).all():
            raise FloatingPointError(f"{arm}/seed{seed}: non-finite gradient summary")
        rows.append(row)
    return rows


def _training_row(root: Path, arm: str, seed: int):
    run_name = _run_name(arm, seed)
    run_dir = root / "training" / "runs" / run_name
    config = _read_object(run_dir / "config.resolved.json")
    complete = _read_object(run_dir / "COMPLETE.json")
    _validate_config(config, complete, arm, seed, run_dir)
    if not (run_dir / "policy_final.pt").is_file():
        raise ValueError(f"{run_name}: missing policy_final.pt")
    with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as data:
        aoi = np.asarray(data["mean_aoi_ms_episode_agent"], dtype=np.float64)
        cam = np.asarray(data["endpoint_cam_episode_agent"], dtype=np.float64)
        remaining = np.asarray(data["remaining_demand"], dtype=np.float64)
    if aoi.shape != (500, 5) or cam.shape != (500, 5) or remaining.shape != (500, 100, 5):
        raise ValueError(f"{run_name}: unexpected training metric shapes")
    payload = np.clip(1.0 - remaining[:, -1, :] / float(config["cam_bits"]), 0.0, 1.0)
    diagnostics = json.loads((run_dir / "learning_diagnostics.json").read_text(encoding="utf-8"))
    if not isinstance(diagnostics, list) or len(diagnostics) != 100:
        raise ValueError(f"{run_name}: expected 100 PPO updates")
    if any(update.get("mappo_actor_update_mode") != arm for update in diagnostics):
        raise ValueError(f"{run_name}: update-mode diagnostic mismatch")
    last = diagnostics[-20:]
    agent_aoi = aoi[-100:].mean(axis=0)
    agent_cam = cam[-100:].mean(axis=0)
    agent_payload = payload[-100:].mean(axis=0)
    row = {
        "arm": arm,
        "seed": seed,
        "run_name": run_name,
        "mean_aoi_ms": float(agent_aoi.mean()),
        "worst_agent_aoi_ms": float(agent_aoi.max()),
        "mean_binary_cam": float(agent_cam.mean()),
        "worst_agent_binary_cam": float(agent_cam.min()),
        "mean_payload_completion": float(agent_payload.mean()),
        "worst_agent_payload_completion": float(agent_payload.min()),
        "last20_approx_kl": _mean(value for update in last for value in update["approx_kl_per_agent"]),
        "last20_clip_fraction": _mean(value for update in last for value in update["clip_fraction_per_agent"]),
        "last20_actor_grad_norm": _mean(value for update in last for value in update["actor_grad_norm_per_agent"]),
        "last20_explained_variance": _mean(update["explained_variance"] for update in last),
    }
    row["screen_success"] = bool(
        row["worst_agent_aoi_ms"] < 50.0 and row["worst_agent_binary_cam"] >= 0.5
    )
    return row, _gradient_rows(arm, seed, diagnostics)


def _evaluation_row(root: Path, arm: str, seed: int, mode: str):
    run_name = _run_name(arm, seed)
    path = root / "evaluations" / run_name / _eval_id(mode) / "EVAL_COMPLETE.json"
    complete = _read_object(path)
    exact = {
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": "tdec",
        "mappo_actor_update_mode": arm,
        "scenario": SCENARIO,
        "training_seed": seed,
        "training_run_name": run_name,
        "mappo_eval_mode": mode,
        "eval_seeds": list(EVAL_SEEDS),
        "eval_episodes": 100,
        "eval_warmup_episodes": 5,
        "diagnostic_evaluation": True,
        "is_formal_result": False,
    }
    for key, expected in exact.items():
        if complete.get(key) != expected:
            raise ValueError(f"{path.parent}: {key} mismatch")
    row = {
        "arm": arm,
        "mode": mode,
        "training_seed": seed,
        "run_name": run_name,
        "mean_aoi_ms": float(complete["mean_AoI_ms"]),
        "worst_agent_aoi_ms": float(complete["worst_agent_mean_AoI_ms"]),
        "mean_binary_cam": float(complete["CAM_success_probability"]),
        "worst_agent_binary_cam": float(complete["worst_agent_CAM_success_probability"]),
        "mean_payload_completion": float(complete["payload_completion"]),
        "worst_agent_payload_completion": float(complete["worst_agent_payload_completion"]),
    }
    row["screen_success"] = bool(
        row["worst_agent_aoi_ms"] < 50.0 and row["worst_agent_binary_cam"] >= 0.5
    )
    return row


def _cohort_summary(rows, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    summaries = []
    metric_keys = (
        "mean_aoi_ms", "worst_agent_aoi_ms", "mean_binary_cam",
        "worst_agent_binary_cam", "mean_payload_completion",
        "worst_agent_payload_completion",
    )
    for key, selected in groups.items():
        summary = dict(zip(group_keys, key))
        for metric in metric_keys:
            summary[metric] = _mean(row[metric] for row in selected)
        aoi = np.asarray([row["mean_aoi_ms"] for row in selected], dtype=np.float64)
        summary["sd_aoi_across_training_seeds"] = float(aoi.std(ddof=1))
        summary["success_count"] = int(sum(row["screen_success"] for row in selected))
        summaries.append(summary)
    return summaries


def _gradient_summary(rows):
    numeric_keys = [key for key in rows[0] if key not in {"arm", "seed", "ppo_epoch"}]
    summaries = []
    for arm in ARMS:
        for epoch in (0, 9):
            selected = [row for row in rows if row["arm"] == arm and row["ppo_epoch"] == epoch]
            summary = {"arm": arm, "ppo_epoch": epoch}
            for key in numeric_keys:
                values = [row[key] for row in selected if row[key] is not None]
                summary[key] = _mean(values)
            summaries.append(summary)
    return summaries


def _paired(training, evaluations):
    rows = []
    train_index = {(row["arm"], row["seed"]): row for row in training}
    eval_index = {(row["arm"], row["mode"], row["training_seed"]): row for row in evaluations}
    for challenger in ARMS[1:]:
        comparisons = [("training", "last100", train_index)]
        for mode in MODES:
            comparisons.append(("heldout", mode, eval_index))
        for phase, mode, index in comparisons:
            if phase == "training":
                left = [index[(challenger, seed)] for seed in SEEDS]
                right = [index[("composed_clip", seed)] for seed in SEEDS]
            else:
                left = [index[(challenger, mode, seed)] for seed in SEEDS]
                right = [index[("composed_clip", mode, seed)] for seed in SEEDS]
            delta_aoi = np.asarray([a["mean_aoi_ms"] - b["mean_aoi_ms"] for a, b in zip(left, right)])
            delta_cam = np.asarray([a["mean_binary_cam"] - b["mean_binary_cam"] for a, b in zip(left, right)])
            rows.append({
                "challenger": challenger,
                "reference": "composed_clip",
                "phase": phase,
                "mode": mode,
                "mean_delta_aoi_ms": float(delta_aoi.mean()),
                "aoi_wins": int((delta_aoi < 0.0).sum()),
                "mean_delta_binary_cam": float(delta_cam.mean()),
                "binary_cam_wins": int((delta_cam > 0.0).sum()),
            })
    return rows


def summarize(result_root: Path):
    root = result_root.expanduser().resolve()
    training, gradients = [], []
    for arm in ARMS:
        for seed in SEEDS:
            row, gradient = _training_row(root, arm, seed)
            training.append(row)
            gradients.extend(gradient)
    evaluations = [
        _evaluation_row(root, arm, seed, mode)
        for arm in ARMS for seed in SEEDS for mode in MODES
    ]
    return {
        "contract": {
            "scenario": SCENARIO,
            "arms": list(ARMS),
            "training_seeds": list(SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
            "train_cells": 18,
            "eval_cells": 36,
            "episodes": 500,
            "eval_episodes": 100,
            "mappo_variant": "tdec",
            "mappo_actor_lr": 0.0005,
            "mappo_entropy_coefficients": [0.02, 0.02, 0.002],
            "mappo_value_clip_mode": "normalized",
        },
        "training_per_seed": training,
        "training_summary": _cohort_summary(training, ("arm",)),
        "evaluation_per_seed": evaluations,
        "evaluation_summary": _cohort_summary(evaluations, ("arm", "mode")),
        "gradient_per_seed_epoch": gradients,
        "gradient_summary_by_arm_epoch": _gradient_summary(gradients),
        "paired_vs_composed": _paired(training, evaluations),
    }


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report) -> None:
    lines = [
        "# MAPPO gradient-conflict Phase 2",
        "",
        "All arms use the same TDec critics, common-scale component advantages, joint hybrid-action ratio, and frozen hyperparameters.",
        "",
        "## Training last 100 episodes",
        "",
        "| arm | AoI mean±SD | worst AoI | binary CAM mean/worst | payload mean/worst | success |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["training_summary"]:
        lines.append(
            f"| {row['arm']} | {row['mean_aoi_ms']:.3f}±{row['sd_aoi_across_training_seeds']:.3f} "
            f"| {row['worst_agent_aoi_ms']:.3f} | {row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} "
            f"| {row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} "
            f"| {row['success_count']}/6 |"
        )
    lines.extend([
        "", "## Frozen held-out", "",
        "| arm | mode | AoI mean±SD | worst AoI | binary CAM mean/worst | payload mean/worst | success |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["evaluation_summary"]:
        lines.append(
            f"| {row['arm']} | {row['mode']} | {row['mean_aoi_ms']:.3f}±{row['sd_aoi_across_training_seeds']:.3f} "
            f"| {row['worst_agent_aoi_ms']:.3f} | {row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} "
            f"| {row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} "
            f"| {row['success_count']}/6 |"
        )
    lines.extend([
        "", "## PCGrad projection audit (last 20 updates)", "",
        "| PPO epoch | pre cancellation | post cancellation | pre conflicts g-t1/g-t2/t1-t2 | post conflicts | projection magnitude |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["gradient_summary_by_arm_epoch"]:
        if row["arm"] != "pcgrad":
            continue
        lines.append(
            f"| {row['ppo_epoch']} | {_format_optional(row['pre_cancellation_ratio'], 4)} "
            f"| {_format_optional(row['post_cancellation_ratio'], 4)} "
            f"| {_format_optional(row['pre_global_task1_conflict_rate'], 3)}/"
            f"{_format_optional(row['pre_global_task2_conflict_rate'], 3)}/"
            f"{_format_optional(row['pre_task1_task2_conflict_rate'], 3)} "
            f"| {_format_optional(row['post_global_task1_conflict_rate'], 3)}/"
            f"{_format_optional(row['post_global_task2_conflict_rate'], 3)}/"
            f"{_format_optional(row['post_task1_task2_conflict_rate'], 3)} "
            f"| {_format_optional(row['aggregate_projection_magnitude'], 4)} |"
        )
    lines.extend([
        "", "Paired deltas are challenger minus composed_clip; negative AoI and positive CAM favor the challenger.", "",
        "| challenger | phase/mode | ΔAoI | AoI wins | Δbinary CAM | CAM wins |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in report["paired_vs_composed"]:
        lines.append(
            f"| {row['challenger']} | {row['phase']}/{row['mode']} | {row['mean_delta_aoi_ms']:+.3f} "
            f"| {row['aoi_wins']}/6 | {row['mean_delta_binary_cam']:+.4f} | {row['binary_cam_wins']}/6 |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(result_root: Path, report) -> Path:
    output = result_root.expanduser().resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("phase2_training_per_seed.csv", "training_per_seed"),
        ("phase2_training_summary.csv", "training_summary"),
        ("phase2_eval_per_seed.csv", "evaluation_per_seed"),
        ("phase2_eval_summary.csv", "evaluation_summary"),
        ("phase2_gradient_per_seed_epoch.csv", "gradient_per_seed_epoch"),
        ("phase2_gradient_summary_by_arm_epoch.csv", "gradient_summary_by_arm_epoch"),
        ("phase2_paired_vs_composed.csv", "paired_vs_composed"),
    ):
        _write_csv(output / name, report[key])
    (output / "phase2_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output / "phase2_comparison.md", report)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.result_root)
    output = write_report(args.result_root, report)
    print(json.dumps({
        "output": str(output),
        "training_summary": report["training_summary"],
        "evaluation_summary": report["evaluation_summary"],
        "paired_vs_composed": report["paired_vs_composed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
