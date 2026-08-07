"""Summarize the Algorithm 1 intra-platoon-gap extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

if __package__:
    from .summarize_modified_maddpg_default import _read_json, _window_metrics
else:
    from summarize_modified_maddpg_default import _read_json, _window_metrics


EXPECTED_ALGORITHM = "modified_maddpg"
EXPECTED_GAPS = (5, 15, 25, 35)
EXTENSION_GAPS = (5, 15, 35)
EXPECTED_SEEDS = (8, 9, 10, 11, 12, 13)


def _scenario(gap: int) -> str:
    return f"p05_n04_g{gap:02d}"


def _run_location(extension_root: Path, default_root: Path, gap: int, seed: int) -> Tuple[Path, str, str]:
    scenario = _scenario(gap)
    if gap == 25:
        run_name = f"modified_maddpg_default_{scenario}_seed{seed:02d}"
        return default_root / "runs" / run_name, run_name, "default_reuse"
    run_name = f"modified_maddpg_gap_{scenario}_seed{seed:02d}"
    return extension_root / "runs" / run_name, run_name, "gap_extension"


def _validate_config(config: Dict[str, object], gap: int, seed: int) -> None:
    expected = {
        "algorithm": EXPECTED_ALGORITHM,
        "profile": "paper_faithful",
        "seed": seed,
        "episodes": 500,
        "slow_update_every_episodes": 1,
        "global_update_mode": "synchronous_joint",
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError(
                f"gap {gap} seed {seed}: {key}={config.get(key)!r}, expected {wanted!r}"
            )
    scenario = config.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError(f"gap {gap} seed {seed}: scenario must be an object")
    scenario_expected = {
        "id": _scenario(gap),
        "number_platoons": 5,
        "platoon_size": 4,
        "gap_m": float(gap),
    }
    for key, wanted in scenario_expected.items():
        if scenario.get(key) != wanted:
            raise ValueError(
                f"gap {gap} seed {seed}: scenario.{key}={scenario.get(key)!r}, expected {wanted!r}"
            )
    if not np.isclose(float(config.get("tau", np.nan)), 0.005, rtol=0.0, atol=1e-12):
        raise ValueError(f"gap {gap} seed {seed}: tau must be 0.005")
    if not np.isclose(float(config.get("exploration_noise", np.nan)), 0.3, rtol=0.0, atol=1e-12):
        raise ValueError(f"gap {gap} seed {seed}: training noise must be fixed at 0.3")


def _aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_fields = [key for key in rows[0] if key.startswith("last")]
    summaries: List[Dict[str, object]] = []
    for gap in EXPECTED_GAPS:
        gap_rows = [row for row in rows if row["gap_m"] == gap]
        if len(gap_rows) != len(EXPECTED_SEEDS):
            raise ValueError(f"gap {gap}: expected six rows, found {len(gap_rows)}")
        summary: Dict[str, object] = {
            "gap_m": gap,
            "seed_count": len(gap_rows),
            "screen_success_last100": int(sum(bool(row["screen_success_last100"]) for row in gap_rows)),
        }
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in gap_rows], dtype=np.float64)
            summary[field] = float(values.mean())
            summary[f"{field}_seed_sd"] = float(values.std(ddof=1))
        summaries.append(summary)
    return summaries


def _trend(rows: List[Dict[str, object]], window: int) -> Dict[str, object]:
    prefix = f"last{window}"
    per_seed: List[Dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        seed_rows = sorted(
            (row for row in rows if row["seed"] == seed),
            key=lambda row: int(row["gap_m"]),
        )
        if [row["gap_m"] for row in seed_rows] != list(EXPECTED_GAPS):
            raise ValueError(f"seed {seed}: incomplete gap sequence")
        aoi = [float(row[f"{prefix}_mean_aoi_ms"]) for row in seed_rows]
        binary = [float(row[f"{prefix}_mean_binary_cam"]) for row in seed_rows]
        payload = [float(row[f"{prefix}_mean_payload_completion"]) for row in seed_rows]
        per_seed.append({
            "seed": seed,
            "aoi_by_gap": dict(zip(map(str, EXPECTED_GAPS), aoi)),
            "binary_cam_by_gap": dict(zip(map(str, EXPECTED_GAPS), binary)),
            "payload_by_gap": dict(zip(map(str, EXPECTED_GAPS), payload)),
            "aoi_nondecreasing": all(left <= right for left, right in zip(aoi, aoi[1:])),
            "aoi_endpoint_rise": aoi[-1] > aoi[0],
            "binary_endpoint_decline": binary[-1] < binary[0],
            "payload_endpoint_decline": payload[-1] < payload[0],
        })
    return {
        "window": f"last{window}",
        "per_seed": per_seed,
        "aoi_nondecreasing_count": int(sum(bool(row["aoi_nondecreasing"]) for row in per_seed)),
        "aoi_endpoint_rise_count": int(sum(bool(row["aoi_endpoint_rise"]) for row in per_seed)),
        "binary_endpoint_decline_count": int(sum(bool(row["binary_endpoint_decline"]) for row in per_seed)),
        "payload_endpoint_decline_count": int(sum(bool(row["payload_endpoint_decline"]) for row in per_seed)),
    }


def summarize(extension_root: Path, default_root: Path) -> Dict[str, object]:
    extension_root = extension_root.expanduser().resolve()
    default_root = default_root.expanduser().resolve()
    rows: List[Dict[str, object]] = []
    per_episode: List[Dict[str, object]] = []

    for gap in EXPECTED_GAPS:
        for seed in EXPECTED_SEEDS:
            run_dir, run_name, source = _run_location(extension_root, default_root, gap, seed)
            config = _read_json(run_dir / "config.resolved.json")
            complete = _read_json(run_dir / "COMPLETE.json")
            _validate_config(config, gap, seed)
            if complete.get("status") != "complete" or complete.get("algorithm") != EXPECTED_ALGORITHM:
                raise ValueError(f"gap {gap} seed {seed}: incomplete or wrong-algorithm COMPLETE.json")

            with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as data:
                aoi = np.asarray(data["mean_aoi_ms_episode_agent"], dtype=np.float64)
                cam = np.asarray(data["endpoint_cam_episode_agent"], dtype=np.float64)
                remaining = np.asarray(data["remaining_demand"], dtype=np.float64)
            if aoi.shape != (500, 5) or cam.shape != (500, 5) or remaining.shape != (500, 100, 5):
                raise ValueError(
                    f"gap {gap} seed {seed}: unexpected metric shapes "
                    f"aoi={aoi.shape}, cam={cam.shape}, remaining={remaining.shape}"
                )
            payload = np.clip(1.0 - remaining[:, -1, :] / float(config["cam_bits"]), 0.0, 1.0)
            selection = complete.get("checkpoint_selection")
            best_episode = selection.get("best_episode") if isinstance(selection, dict) else None
            row: Dict[str, object] = {
                "gap_m": gap,
                "seed": seed,
                "source": source,
                "run_name": run_name,
                "best_episode": best_episode,
            }
            row.update(_window_metrics(aoi, cam, payload, 100))
            row.update(_window_metrics(aoi, cam, payload, 50))
            row["screen_success_last100"] = bool(
                float(row["last100_worst_agent_aoi_ms"]) < 50.0
                and float(row["last100_worst_agent_binary_cam"]) >= 0.5
            )
            rows.append(row)

            for episode in range(500):
                per_episode.append({
                    "gap_m": gap,
                    "seed": seed,
                    "source": source,
                    "episode": episode + 1,
                    "mean_aoi_ms": float(aoi[episode].mean()),
                    "worst_agent_aoi_ms": float(aoi[episode].max()),
                    "mean_binary_cam": float(cam[episode].mean()),
                    "worst_agent_binary_cam": float(cam[episode].min()),
                    "mean_payload_completion": float(payload[episode].mean()),
                    "worst_agent_payload_completion": float(payload[episode].min()),
                })

    return {
        "experiment": {
            "algorithm": EXPECTED_ALGORITHM,
            "label": "gap-extension",
            "P": 5,
            "N": 4,
            "gaps_m": list(EXPECTED_GAPS),
            "new_gaps_m": list(EXTENSION_GAPS),
            "reused_gap_m": 25,
            "tau": 0.005,
            "slow_update_every_episodes": 1,
            "global_update_mode": "synchronous_joint",
            "episodes": 500,
            "training_noise": 0.3,
            "seeds": list(EXPECTED_SEEDS),
            "primary_window": "last100 training episodes",
            "secondary_window": "last50 training episodes",
            "default_root": str(default_root),
        },
        "rows": rows,
        "by_gap": _aggregate(rows),
        "trend_last100": _trend(rows, 100),
        "trend_last50": _trend(rows, 50),
        "per_episode": per_episode,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: Dict[str, object]) -> None:
    by_gap = report["by_gap"]
    rows = report["rows"]
    trend100 = report["trend_last100"]
    trend50 = report["trend_last50"]
    lines = [
        "# Modified MADDPG gap extension",
        "",
        "Algorithm 1; P=5, N=4; tau=0.005; slow=1; synchronous_joint; seeds 8--13.",
        "Gaps 5, 15, and 35 m are new training cells; gap 25 m is reused from the validated default run.",
        "Primary evidence is train-last100. Strict binary CAM and continuous payload completion remain separate.",
        "",
        "| gap (m) | AoI mean +/- seed SD | worst-agent AoI | binary CAM | payload | screen |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in by_gap:
        lines.append(
            f"| {row['gap_m']} | {row['last100_mean_aoi_ms']:.3f} +/- "
            f"{row['last100_mean_aoi_ms_seed_sd']:.3f} | "
            f"{row['last100_worst_agent_aoi_ms']:.3f} | "
            f"{row['last100_mean_binary_cam']:.4f} | "
            f"{row['last100_mean_payload_completion']:.4f} | "
            f"{row['screen_success_last100']}/6 |"
        )
    lines.extend([
        "",
        "## Per-seed AoI (train-last100)",
        "",
        "| seed | gap5 | gap15 | gap25 | gap35 | nondecreasing | endpoint rise |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ])
    for seed in EXPECTED_SEEDS:
        seed_rows = {int(row["gap_m"]): row for row in rows if row["seed"] == seed}
        trend = next(row for row in trend100["per_seed"] if row["seed"] == seed)
        lines.append(
            f"| {seed} | {seed_rows[5]['last100_mean_aoi_ms']:.3f} | "
            f"{seed_rows[15]['last100_mean_aoi_ms']:.3f} | "
            f"{seed_rows[25]['last100_mean_aoi_ms']:.3f} | "
            f"{seed_rows[35]['last100_mean_aoi_ms']:.3f} | "
            f"{'yes' if trend['aoi_nondecreasing'] else 'no'} | "
            f"{'yes' if trend['aoi_endpoint_rise'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Trend counts",
        "",
        f"- last100 AoI nondecreasing: {trend100['aoi_nondecreasing_count']}/6",
        f"- last100 AoI gap5 -> gap35 rise: {trend100['aoi_endpoint_rise_count']}/6",
        f"- last100 strict binary CAM gap5 -> gap35 decline: {trend100['binary_endpoint_decline_count']}/6",
        f"- last100 payload gap5 -> gap35 decline: {trend100['payload_endpoint_decline_count']}/6",
        f"- last50 AoI nondecreasing: {trend50['aoi_nondecreasing_count']}/6",
        f"- last50 AoI gap5 -> gap35 rise: {trend50['aoi_endpoint_rise_count']}/6",
        "",
        "The screen is descriptive only: worst-agent AoI < 50 ms and worst-agent strict binary CAM >= 0.5.",
        "No held-out evaluation or checkpoint selection is used for the Fig. 5 trend judgment.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--default-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.extension_root, args.default_root)
    output_dir = args.extension_root.expanduser().resolve() / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in report.items() if key != "per_episode"}
    (output_dir / "modified_maddpg_gap_extension_summary.json").write_text(
        json.dumps(compact, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "modified_maddpg_gap_extension.csv", report["rows"])
    _write_csv(output_dir / "modified_maddpg_gap_extension_by_gap.csv", report["by_gap"])
    _write_csv(output_dir / "modified_maddpg_gap_extension_per_episode.csv", report["per_episode"])
    _write_markdown(output_dir / "modified_maddpg_gap_extension.md", report)
    print(json.dumps({
        "by_gap": compact["by_gap"],
        "trend_last100": {
            key: value for key, value in compact["trend_last100"].items() if key != "per_seed"
        },
        "trend_last50": {
            key: value for key, value in compact["trend_last50"].items() if key != "per_seed"
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
