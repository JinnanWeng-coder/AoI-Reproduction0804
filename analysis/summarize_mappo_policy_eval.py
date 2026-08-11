"""Summarize the four-arm deterministic/stochastic MAPPO policy diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (8, 9, 10, 11, 12, 13)
MODES = ("deterministic", "stochastic")
EVAL_SEEDS = (201, 202, 203, 204, 205, 206)
ARM_SPECS = {
    "baseline": {"actor_lr": 0.0005, "entropy_rb": 0.01, "entropy_mode": 0.01, "entropy_power": 0.001},
    "actor_lr1e4": {"actor_lr": 0.0001, "entropy_rb": 0.01, "entropy_mode": 0.01, "entropy_power": 0.001},
    "entropy2x": {"actor_lr": 0.0005, "entropy_rb": 0.02, "entropy_mode": 0.02, "entropy_power": 0.002},
    "actor_lr1e4_entropy2x": {"actor_lr": 0.0001, "entropy_rb": 0.02, "entropy_mode": 0.02, "entropy_power": 0.002},
}
PAIRS = (
    ("actor_lr1e4", "baseline"),
    ("entropy2x", "baseline"),
    ("actor_lr1e4_entropy2x", "baseline"),
    ("actor_lr1e4_entropy2x", "actor_lr1e4"),
    ("actor_lr1e4_entropy2x", "entropy2x"),
)


def _run_name(arm: str, seed: int) -> str:
    if arm == "baseline":
        return f"mappo_default_p05_n04_g25_seed{seed:02d}"
    if arm == "actor_lr1e4_entropy2x":
        return f"mappo_combined_{arm}_p05_n04_g25_seed{seed:02d}"
    return f"mappo_stability_{arm}_p05_n04_g25_seed{seed:02d}"


def _eval_id(mode: str) -> str:
    seed_token = "-".join(str(seed) for seed in EVAL_SEEDS)
    return f"eval_validation_policy_final_{mode}_sequential_warm_warm5_s{seed_token}_ep100"


def _read_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def summarize(result_root: Path):
    result_root = result_root.expanduser().resolve()
    rows = []
    for arm, spec in ARM_SPECS.items():
        for seed in SEEDS:
            run_name = _run_name(arm, seed)
            for mode in MODES:
                eval_dir = result_root / "evaluations" / run_name / _eval_id(mode)
                complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
                checks = {
                    "algorithm": "mappo",
                    "status": "complete",
                    "diagnostic_evaluation": True,
                    "scenario": "p05_n04_g25",
                    "training_seed": seed,
                    "training_run_name": run_name,
                    "mappo_eval_mode": mode,
                    "eval_seeds": list(EVAL_SEEDS),
                    "eval_episodes": 100,
                    "eval_warmup_episodes": 5,
                    "policy_name": "policy_final.pt",
                }
                for key, expected in checks.items():
                    if complete.get(key) != expected:
                        raise ValueError(f"{eval_dir}: {key}={complete.get(key)!r}, expected {expected!r}")
                for key, expected in (
                    ("mappo_actor_lr", spec["actor_lr"]),
                    ("mappo_entropy_coef_rb", spec["entropy_rb"]),
                    ("mappo_entropy_coef_mode", spec["entropy_mode"]),
                    ("mappo_entropy_coef_power", spec["entropy_power"]),
                ):
                    if not np.isclose(float(complete.get(key)), expected, rtol=0.0, atol=1e-12):
                        raise ValueError(f"{eval_dir}: {key} mismatch")
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
                rows.append(row)

    summary = []
    for arm, spec in ARM_SPECS.items():
        for mode in MODES:
            selected = [row for row in rows if row["arm"] == arm and row["mode"] == mode]
            aoi = np.asarray([row["mean_aoi_ms"] for row in selected], dtype=np.float64)
            summary.append({
                "arm": arm,
                "mode": mode,
                **spec,
                "mean_aoi_ms": float(aoi.mean()),
                "sd_aoi_across_training_seeds": float(aoi.std(ddof=1)),
                "worst_agent_aoi_ms": float(np.mean([row["worst_agent_aoi_ms"] for row in selected])),
                "mean_binary_cam": float(np.mean([row["mean_binary_cam"] for row in selected])),
                "worst_agent_binary_cam": float(np.mean([row["worst_agent_binary_cam"] for row in selected])),
                "mean_payload_completion": float(np.mean([row["mean_payload_completion"] for row in selected])),
                "worst_agent_payload_completion": float(np.mean([row["worst_agent_payload_completion"] for row in selected])),
                "success_count": int(sum(row["screen_success"] for row in selected)),
            })

    indexed = {(row["arm"], row["mode"], row["training_seed"]): row for row in rows}
    paired = []
    for mode in MODES:
        for left, right in PAIRS:
            left_rows = [indexed[(left, mode, seed)] for seed in SEEDS]
            right_rows = [indexed[(right, mode, seed)] for seed in SEEDS]
            delta_aoi = np.asarray(
                [left_row["mean_aoi_ms"] - right_row["mean_aoi_ms"] for left_row, right_row in zip(left_rows, right_rows)],
                dtype=np.float64,
            )
            delta_cam = np.asarray(
                [left_row["mean_binary_cam"] - right_row["mean_binary_cam"] for left_row, right_row in zip(left_rows, right_rows)],
                dtype=np.float64,
            )
            paired.append({
                "mode": mode,
                "left_arm": left,
                "right_arm": right,
                "mean_delta_aoi_ms_left_minus_right": float(delta_aoi.mean()),
                "aoi_wins_left": int(np.sum(delta_aoi < 0.0)),
                "mean_delta_binary_cam_left_minus_right": float(delta_cam.mean()),
                "cam_wins_left": int(np.sum(delta_cam > 0.0)),
            })
    return {"per_training_seed": rows, "summary": summary, "paired": paired}


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report) -> None:
    lines = [
        "# MAPPO final-policy held-out diagnostic",
        "",
        "All rows use policy_final.pt, held-out seeds 201--206, 5 warm-up episodes, and 100 scored episodes.",
        "",
        "| arm | mode | AoI mean±SD | worst AoI | binary CAM mean/worst | payload mean/worst | success |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['arm']} | {row['mode']} | {row['mean_aoi_ms']:.3f}±{row['sd_aoi_across_training_seeds']:.3f} "
            f"| {row['worst_agent_aoi_ms']:.3f} | {row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} "
            f"| {row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} "
            f"| {row['success_count']}/6 |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.result_root)
    output = args.result_root.expanduser().resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "mappo_policy_eval_per_training_seed.csv", report["per_training_seed"])
    _write_csv(output / "mappo_policy_eval_summary.csv", report["summary"])
    _write_csv(output / "mappo_policy_eval_paired.csv", report["paired"])
    (output / "mappo_policy_eval_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output / "mappo_policy_eval.md", report)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
