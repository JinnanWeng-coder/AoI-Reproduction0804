"""Extract a compact action/reward audit from the existing MAPPO default NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPECTED_SEEDS = (8, 9, 10, 11, 12, 13)
BLOCKS = ((1, 100), (101, 200), (201, 300), (301, 400), (401, 500))


def _read_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _normalized_entropy(values: np.ndarray, categories: int) -> float:
    counts = np.bincount(values.astype(np.int64), minlength=categories).astype(np.float64)
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0.0]
    return float(-(nonzero * np.log(nonzero)).sum() / np.log(float(categories)))


def audit(result_root: Path):
    result_root = result_root.expanduser().resolve()
    rows = []
    for seed in EXPECTED_SEEDS:
        run_name = f"mappo_default_p05_n04_g25_seed{seed:02d}"
        run_dir = result_root / "runs" / run_name
        config = _read_json(run_dir / "config.resolved.json")
        complete = _read_json(run_dir / "COMPLETE.json")
        if config.get("algorithm") != "mappo" or config.get("seed") != seed:
            raise ValueError(f"seed {seed}: wrong resolved configuration")
        if complete.get("status") != "complete" or complete.get("update_count") != 100:
            raise ValueError(f"seed {seed}: run is not complete")
        with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as data:
            metrics = {name: np.asarray(data[name]) for name in (
                "mean_aoi_ms_episode_agent",
                "endpoint_cam_episode_agent",
                "remaining_demand",
                "task1_step",
                "task2_step",
                "global_step",
                "power_dbm",
                "mode",
                "rb",
            )}
        if metrics["mean_aoi_ms_episode_agent"].shape != (500, 5):
            raise ValueError(f"seed {seed}: unexpected episode/agent shape")
        if metrics["power_dbm"].shape != (500, 100, 5):
            raise ValueError(f"seed {seed}: unexpected action shape")
        payload = np.clip(
            1.0 - metrics["remaining_demand"][:, -1, :] / float(config["cam_bits"]),
            0.0,
            1.0,
        )
        for start, end in BLOCKS:
            selection = slice(start - 1, end)
            global_mean = float(metrics["global_step"][selection].mean())
            for agent in range(5):
                power = metrics["power_dbm"][selection, :, agent].reshape(-1)
                mode = metrics["mode"][selection, :, agent].reshape(-1).astype(np.int64)
                rb = metrics["rb"][selection, :, agent].reshape(-1).astype(np.int64)
                task1 = float(metrics["task1_step"][selection, :, agent].mean())
                task2 = float(metrics["task2_step"][selection, :, agent].mean())
                rows.append({
                    "seed": seed,
                    "episode_start": start,
                    "episode_end": end,
                    "agent": agent,
                    "mean_aoi_ms": float(metrics["mean_aoi_ms_episode_agent"][selection, agent].mean()),
                    "binary_cam": float(metrics["endpoint_cam_episode_agent"][selection, agent].mean()),
                    "payload_completion": float(payload[selection, agent].mean()),
                    "task1_reward": task1,
                    "task2_reward": task2,
                    "global_reward": global_mean,
                    "combined_reward": task1 + task2 + global_mean,
                    "power_mean_dbm": float(power.mean()),
                    "power_p10_dbm": float(np.quantile(power, 0.10)),
                    "power_p90_dbm": float(np.quantile(power, 0.90)),
                    "power_near_min_fraction": float((power <= float(config["power_min_dbm"]) + 0.29).mean()),
                    "power_near_max_fraction": float((power >= float(config["power_max_dbm"]) - 0.29).mean()),
                    "mode1_fraction": float((mode == 1).mean()),
                    "mode_entropy_normalized": _normalized_entropy(mode, 2),
                    "rb0_fraction": float((rb == 0).mean()),
                    "rb1_fraction": float((rb == 1).mean()),
                    "rb2_fraction": float((rb == 2).mean()),
                    "rb_entropy_normalized": _normalized_entropy(rb, 3),
                })
    if not np.all(np.isfinite([
        float(value)
        for row in rows
        for key, value in row.items()
        if key not in {"seed", "episode_start", "episode_end", "agent"}
    ])):
        raise FloatingPointError("non-finite action/reward audit value")

    cohort = []
    for start, end in BLOCKS:
        selected = [row for row in rows if row["episode_start"] == start]
        cohort.append({
            "episode_start": start,
            "episode_end": end,
            "mean_aoi_ms": float(np.mean([row["mean_aoi_ms"] for row in selected])),
            "mean_binary_cam": float(np.mean([row["binary_cam"] for row in selected])),
            "mean_payload_completion": float(np.mean([row["payload_completion"] for row in selected])),
            "mean_combined_reward": float(np.mean([row["combined_reward"] for row in selected])),
            "mean_power_dbm": float(np.mean([row["power_mean_dbm"] for row in selected])),
            "mean_mode1_fraction": float(np.mean([row["mode1_fraction"] for row in selected])),
            "mean_rb_entropy_normalized": float(np.mean([row["rb_entropy_normalized"] for row in selected])),
            "mean_mode_entropy_normalized": float(np.mean([row["mode_entropy_normalized"] for row in selected])),
        })
    final_failures = sorted(
        (row for row in rows if row["episode_start"] == 401),
        key=lambda row: (row["binary_cam"], row["payload_completion"]),
    )[:10]
    return {"rows": rows, "cohort_blocks": cohort, "lowest_final_agents": final_failures}


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report) -> None:
    lines = [
        "# MAPPO default action/reward audit",
        "",
        "This is read-only post-processing of the existing six default runs.",
        "",
        "| episodes | AoI | binary CAM | payload | combined reward | power dBm | mode-1 fraction | RB entropy | mode entropy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cohort_blocks"]:
        lines.append(
            f"| {row['episode_start']}–{row['episode_end']} | {row['mean_aoi_ms']:.3f} "
            f"| {row['mean_binary_cam']:.4f} | {row['mean_payload_completion']:.4f} "
            f"| {row['mean_combined_reward']:.4f} | {row['mean_power_dbm']:.3f} "
            f"| {row['mean_mode1_fraction']:.4f} | {row['mean_rb_entropy_normalized']:.4f} "
            f"| {row['mean_mode_entropy_normalized']:.4f} |"
        )
    lines.extend([
        "",
        "## Lowest final-block agent CAM",
        "",
        "| seed | agent | CAM | payload | AoI | power | mode-1 | RB entropy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["lowest_final_agents"]:
        lines.append(
            f"| {row['seed']} | {row['agent']} | {row['binary_cam']:.4f} "
            f"| {row['payload_completion']:.4f} | {row['mean_aoi_ms']:.3f} "
            f"| {row['power_mean_dbm']:.3f} | {row['mode1_fraction']:.4f} "
            f"| {row['rb_entropy_normalized']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.result_root)
    output = args.result_root.expanduser().resolve() / "analysis" / "action_reward_audit"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "block_agent.csv", report["rows"])
    _write_csv(output / "cohort_blocks.csv", report["cohort_blocks"])
    (output / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output / "audit.md", report)
    print(json.dumps(report["cohort_blocks"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
