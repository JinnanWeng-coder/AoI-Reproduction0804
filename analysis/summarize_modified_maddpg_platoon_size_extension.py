"""Summarize the Algorithm 1 platoon-size extension."""

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
EXPECTED_SIZES = (4, 6, 8, 10)
EXTENSION_SIZES = (6, 8, 10)
EXPECTED_SEEDS = (8, 9, 10, 11, 12, 13)


def _scenario(size: int) -> str:
    return f"p05_n{size:02d}_g25"


def _run_location(
    extension_root: Path, default_root: Path, size: int, seed: int
) -> Tuple[Path, str, str]:
    scenario = _scenario(size)
    if size == 4:
        run_name = f"modified_maddpg_default_{scenario}_seed{seed:02d}"
        return default_root / "runs" / run_name, run_name, "default_reuse"
    run_name = f"modified_maddpg_platoon_{scenario}_seed{seed:02d}"
    return extension_root / "runs" / run_name, run_name, "platoon_size_extension"


def _validate_config(config: Dict[str, object], size: int, seed: int) -> None:
    expected = {
        "algorithm": EXPECTED_ALGORITHM,
        "seed": seed,
        "episodes": 500,
        "slow_update_every_episodes": 1,
        "global_update_mode": "synchronous_joint",
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError(
                f"N={size} seed {seed}: {key}={config.get(key)!r}, expected {wanted!r}"
            )
    if config.get("profile") not in {"paper_faithful", "reproduction_baseline"}:
        raise ValueError(f"N={size} seed {seed}: unsupported profile={config.get('profile')!r}")
    scenario = config.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError(f"N={size} seed {seed}: scenario must be an object")
    scenario_expected = {
        "id": _scenario(size),
        "number_platoons": 5,
        "platoon_size": size,
        "gap_m": 25.0,
    }
    for key, wanted in scenario_expected.items():
        if scenario.get(key) != wanted:
            raise ValueError(
                f"N={size} seed {seed}: scenario.{key}={scenario.get(key)!r}, expected {wanted!r}"
            )
    if not np.isclose(float(config.get("tau", np.nan)), 0.005, rtol=0.0, atol=1e-12):
        raise ValueError(f"N={size} seed {seed}: tau must be 0.005")
    if not np.isclose(float(config.get("exploration_noise", np.nan)), 0.3, rtol=0.0, atol=1e-12):
        raise ValueError(f"N={size} seed {seed}: training noise must be fixed at 0.3")


def _aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_fields = [key for key in rows[0] if key.startswith("last")]
    summaries: List[Dict[str, object]] = []
    for size in EXPECTED_SIZES:
        size_rows = [row for row in rows if row["platoon_size"] == size]
        if len(size_rows) != len(EXPECTED_SEEDS):
            raise ValueError(f"N={size}: expected six rows, found {len(size_rows)}")
        summary: Dict[str, object] = {
            "platoon_size": size,
            "seed_count": len(size_rows),
            "screen_success_last100": int(
                sum(bool(row["screen_success_last100"]) for row in size_rows)
            ),
        }
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in size_rows], dtype=np.float64)
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
            key=lambda row: int(row["platoon_size"]),
        )
        if [row["platoon_size"] for row in seed_rows] != list(EXPECTED_SIZES):
            raise ValueError(f"seed {seed}: incomplete platoon-size sequence")
        aoi = [float(row[f"{prefix}_mean_aoi_ms"]) for row in seed_rows]
        binary = [float(row[f"{prefix}_mean_binary_cam"]) for row in seed_rows]
        payload = [float(row[f"{prefix}_mean_payload_completion"]) for row in seed_rows]
        per_seed.append({
            "seed": seed,
            "aoi_by_size": dict(zip(map(str, EXPECTED_SIZES), aoi)),
            "binary_cam_by_size": dict(zip(map(str, EXPECTED_SIZES), binary)),
            "payload_by_size": dict(zip(map(str, EXPECTED_SIZES), payload)),
            "aoi_nondecreasing": all(left <= right for left, right in zip(aoi, aoi[1:])),
            "aoi_endpoint_rise": aoi[-1] > aoi[0],
            "binary_endpoint_decline": binary[-1] < binary[0],
            "payload_endpoint_decline": payload[-1] < payload[0],
        })
    return {
        "window": f"last{window}",
        "per_seed": per_seed,
        "aoi_nondecreasing_count": int(
            sum(bool(row["aoi_nondecreasing"]) for row in per_seed)
        ),
        "aoi_endpoint_rise_count": int(sum(bool(row["aoi_endpoint_rise"]) for row in per_seed)),
        "binary_endpoint_decline_count": int(
            sum(bool(row["binary_endpoint_decline"]) for row in per_seed)
        ),
        "payload_endpoint_decline_count": int(
            sum(bool(row["payload_endpoint_decline"]) for row in per_seed)
        ),
    }


def summarize(extension_root: Path, default_root: Path) -> Dict[str, object]:
    extension_root = extension_root.expanduser().resolve()
    default_root = default_root.expanduser().resolve()
    rows: List[Dict[str, object]] = []
    per_episode: List[Dict[str, object]] = []

    for size in EXPECTED_SIZES:
        for seed in EXPECTED_SEEDS:
            run_dir, run_name, source = _run_location(extension_root, default_root, size, seed)
            config = _read_json(run_dir / "config.resolved.json")
            complete = _read_json(run_dir / "COMPLETE.json")
            _validate_config(config, size, seed)
            if complete.get("status") != "complete" or complete.get("algorithm") != EXPECTED_ALGORITHM:
                raise ValueError(f"N={size} seed {seed}: incomplete or wrong-algorithm COMPLETE.json")

            with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as data:
                aoi = np.asarray(data["mean_aoi_ms_episode_agent"], dtype=np.float64)
                cam = np.asarray(data["endpoint_cam_episode_agent"], dtype=np.float64)
                remaining = np.asarray(data["remaining_demand"], dtype=np.float64)
            if aoi.shape != (500, 5) or cam.shape != (500, 5) or remaining.shape != (500, 100, 5):
                raise ValueError(
                    f"N={size} seed {seed}: unexpected metric shapes "
                    f"aoi={aoi.shape}, cam={cam.shape}, remaining={remaining.shape}"
                )
            payload = np.clip(1.0 - remaining[:, -1, :] / float(config["cam_bits"]), 0.0, 1.0)
            selection = complete.get("checkpoint_selection")
            best_episode = selection.get("best_episode") if isinstance(selection, dict) else None
            row: Dict[str, object] = {
                "platoon_size": size,
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
                    "platoon_size": size,
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
            "label": "platoon-size-extension",
            "P": 5,
            "gap_m": 25,
            "platoon_sizes": list(EXPECTED_SIZES),
            "new_platoon_sizes": list(EXTENSION_SIZES),
            "reused_platoon_size": 4,
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
        "by_size": _aggregate(rows),
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
    by_size = report["by_size"]
    rows = report["rows"]
    trend100 = report["trend_last100"]
    trend50 = report["trend_last50"]
    lines = [
        "# Modified MADDPG platoon-size extension",
        "",
        "Algorithm 1; P=5, gap=25 m; tau=0.005; slow=1; synchronous_joint; seeds 8--13.",
        "Sizes 6, 8, and 10 are new training cells; size 4 is reused from the validated default run.",
        "Primary evidence is train-last100. Strict binary CAM and continuous payload completion remain separate.",
        "",
        "| N | AoI mean +/- seed SD | worst-agent AoI | binary CAM | payload | screen |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in by_size:
        lines.append(
            f"| {row['platoon_size']} | {row['last100_mean_aoi_ms']:.3f} +/- "
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
        "| seed | N4 | N6 | N8 | N10 | nondecreasing | endpoint rise |",
        "|---:|---:|---:|---:|---:|:---:|:---:|",
    ])
    for seed in EXPECTED_SEEDS:
        seed_rows = {int(row["platoon_size"]): row for row in rows if row["seed"] == seed}
        trend = next(row for row in trend100["per_seed"] if row["seed"] == seed)
        lines.append(
            f"| {seed} | {seed_rows[4]['last100_mean_aoi_ms']:.3f} | "
            f"{seed_rows[6]['last100_mean_aoi_ms']:.3f} | "
            f"{seed_rows[8]['last100_mean_aoi_ms']:.3f} | "
            f"{seed_rows[10]['last100_mean_aoi_ms']:.3f} | "
            f"{'yes' if trend['aoi_nondecreasing'] else 'no'} | "
            f"{'yes' if trend['aoi_endpoint_rise'] else 'no'} |"
        )
    lines.extend([
        "",
        "## Trend counts",
        "",
        f"- last100 AoI nondecreasing: {trend100['aoi_nondecreasing_count']}/6",
        f"- last100 AoI N4 -> N10 rise: {trend100['aoi_endpoint_rise_count']}/6",
        f"- last100 strict binary CAM N4 -> N10 decline: {trend100['binary_endpoint_decline_count']}/6",
        f"- last100 payload N4 -> N10 decline: {trend100['payload_endpoint_decline_count']}/6",
        f"- last50 AoI nondecreasing: {trend50['aoi_nondecreasing_count']}/6",
        f"- last50 AoI N4 -> N10 rise: {trend50['aoi_endpoint_rise_count']}/6",
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
    (output_dir / "modified_maddpg_platoon_size_extension_summary.json").write_text(
        json.dumps(compact, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "modified_maddpg_platoon_size_extension.csv", report["rows"])
    _write_csv(output_dir / "modified_maddpg_platoon_size_extension_by_size.csv", report["by_size"])
    _write_csv(
        output_dir / "modified_maddpg_platoon_size_extension_per_episode.csv",
        report["per_episode"],
    )
    _write_markdown(output_dir / "modified_maddpg_platoon_size_extension.md", report)
    print(json.dumps({
        "by_size": compact["by_size"],
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
