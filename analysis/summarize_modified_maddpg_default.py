"""Summarize the six-seed Algorithm 1 default confirmation experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


EXPECTED_SEEDS = (8, 9, 10, 11, 12, 13)
EXPECTED_SCENARIO = "p05_n04_g25"
EXPECTED_ALGORITHM = "modified_maddpg"


def _read_json(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_config(config: Dict[str, object], seed: int) -> None:
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
            raise ValueError(f"seed {seed}: {key}={config.get(key)!r}, expected {wanted!r}")
    scenario = config.get("scenario")
    if not isinstance(scenario, dict) or scenario.get("id") != EXPECTED_SCENARIO:
        raise ValueError(f"seed {seed}: unexpected scenario")
    if not np.isclose(float(config.get("tau", np.nan)), 0.005, rtol=0.0, atol=1e-12):
        raise ValueError(f"seed {seed}: tau must be 0.005")
    if not np.isclose(float(config.get("exploration_noise", np.nan)), 0.3, rtol=0.0, atol=1e-12):
        raise ValueError(f"seed {seed}: training noise must be fixed at 0.3")


def _window_metrics(aoi: np.ndarray, cam: np.ndarray, payload: np.ndarray, count: int) -> Dict[str, float]:
    per_agent_aoi = aoi[-count:].mean(axis=0)
    per_agent_cam = cam[-count:].mean(axis=0)
    per_agent_payload = payload[-count:].mean(axis=0)
    return {
        f"last{count}_mean_aoi_ms": float(per_agent_aoi.mean()),
        f"last{count}_worst_agent_aoi_ms": float(per_agent_aoi.max()),
        f"last{count}_mean_binary_cam": float(per_agent_cam.mean()),
        f"last{count}_worst_agent_binary_cam": float(per_agent_cam.min()),
        f"last{count}_mean_payload_completion": float(per_agent_payload.mean()),
        f"last{count}_worst_agent_payload_completion": float(per_agent_payload.min()),
    }


def summarize(result_root: Path) -> Dict[str, object]:
    result_root = result_root.expanduser().resolve()
    runs_root = result_root / "runs"
    rows: List[Dict[str, object]] = []
    per_episode: List[Dict[str, object]] = []

    for seed in EXPECTED_SEEDS:
        run_name = f"modified_maddpg_default_{EXPECTED_SCENARIO}_seed{seed:02d}"
        run_dir = runs_root / run_name
        config = _read_json(run_dir / "config.resolved.json")
        complete = _read_json(run_dir / "COMPLETE.json")
        _validate_config(config, seed)
        if complete.get("status") != "complete" or complete.get("algorithm") != EXPECTED_ALGORITHM:
            raise ValueError(f"seed {seed}: incomplete or wrong-algorithm COMPLETE.json")

        with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as data:
            aoi = np.asarray(data["mean_aoi_ms_episode_agent"], dtype=np.float64)
            cam = np.asarray(data["endpoint_cam_episode_agent"], dtype=np.float64)
            remaining = np.asarray(data["remaining_demand"], dtype=np.float64)
        if aoi.shape != (500, 5) or cam.shape != (500, 5) or remaining.shape != (500, 100, 5):
            raise ValueError(
                f"seed {seed}: unexpected metric shapes aoi={aoi.shape}, cam={cam.shape}, remaining={remaining.shape}"
            )
        cam_bits = float(config["cam_bits"])
        payload = np.clip(1.0 - remaining[:, -1, :] / cam_bits, 0.0, 1.0)
        selection = complete.get("checkpoint_selection")
        best_episode = selection.get("best_episode") if isinstance(selection, dict) else None
        row: Dict[str, object] = {
            "seed": seed,
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
                "seed": seed,
                "episode": episode + 1,
                "mean_aoi_ms": float(aoi[episode].mean()),
                "worst_agent_aoi_ms": float(aoi[episode].max()),
                "mean_binary_cam": float(cam[episode].mean()),
                "worst_agent_binary_cam": float(cam[episode].min()),
                "mean_payload_completion": float(payload[episode].mean()),
                "worst_agent_payload_completion": float(payload[episode].min()),
            })

    aggregate_fields = [
        key for key in rows[0]
        if key.startswith("last") and key not in {"screen_success_last100"}
    ]
    cohort = {key: float(np.mean([float(row[key]) for row in rows])) for key in aggregate_fields}
    cohort["success_count_last100"] = int(sum(bool(row["screen_success_last100"]) for row in rows))
    return {
        "experiment": {
            "algorithm": EXPECTED_ALGORITHM,
            "label": "default",
            "scenario": EXPECTED_SCENARIO,
            "P": 5,
            "N": 4,
            "gap_m": 25,
            "tau": 0.005,
            "slow_update_every_episodes": 1,
            "global_update_mode": "synchronous_joint",
            "episodes": 500,
            "training_noise": 0.3,
            "seeds": list(EXPECTED_SEEDS),
            "primary_window": "last100 training episodes",
            "secondary_window": "last50 training episodes",
        },
        "rows": rows,
        "per_episode": per_episode,
        "cohort": cohort,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: Dict[str, object]) -> None:
    rows = report["rows"]
    cohort = report["cohort"]
    lines = [
        "# Modified MADDPG default confirmation",
        "",
        "Algorithm 1; P=5, N=4, gap=25 m; tau=0.005; slow=1; synchronous_joint; seeds 8--13.",
        "Primary evidence is the final 100 training episodes. Strict binary CAM and continuous payload completion are reported separately.",
        "",
        "| seed | AoI mean/worst | binary CAM mean/worst | payload mean/worst | best ep | screen |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['last100_mean_aoi_ms']:.3f}/{row['last100_worst_agent_aoi_ms']:.3f} "
            f"| {row['last100_mean_binary_cam']:.4f}/{row['last100_worst_agent_binary_cam']:.4f} "
            f"| {row['last100_mean_payload_completion']:.4f}/{row['last100_worst_agent_payload_completion']:.4f} "
            f"| {row['best_episode']} | {'ok' if row['screen_success_last100'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Six-seed mean",
        "",
        f"- AoI mean/worst-agent: {cohort['last100_mean_aoi_ms']:.3f}/{cohort['last100_worst_agent_aoi_ms']:.3f} ms",
        f"- Strict binary CAM mean/worst-agent: {cohort['last100_mean_binary_cam']:.4f}/{cohort['last100_worst_agent_binary_cam']:.4f}",
        f"- Payload completion mean/worst-agent: {cohort['last100_mean_payload_completion']:.4f}/{cohort['last100_worst_agent_payload_completion']:.4f}",
        f"- Screen success: {cohort['success_count_last100']}/6",
        "",
        "The screen is descriptive only: worst-agent AoI < 50 ms and worst-agent strict binary CAM >= 0.5.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.result_root)
    output_dir = args.result_root.expanduser().resolve() / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    compact = {key: value for key, value in report.items() if key != "per_episode"}
    (output_dir / "modified_maddpg_default_summary.json").write_text(
        json.dumps(compact, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "modified_maddpg_default.csv", report["rows"])
    _write_csv(output_dir / "modified_maddpg_default_per_episode.csv", report["per_episode"])
    _write_markdown(output_dir / "modified_maddpg_default.md", report)
    print(json.dumps(compact["cohort"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
