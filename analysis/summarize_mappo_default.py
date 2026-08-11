"""Summarize the six-seed MAPPO default training confirmation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


EXPECTED_SEEDS = (8, 9, 10, 11, 12, 13)
EXPECTED_SCENARIO = "p05_n04_g25"


def _read_json(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate(config: Dict[str, object], complete: Dict[str, object], seed: int) -> None:
    expected = {
        "algorithm": "mappo",
        "seed": seed,
        "episodes": 500,
        "mappo_rollout_episodes": 5,
        "mappo_ppo_epochs": 10,
        "checkpoint_mode": "policy_only",
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError(f"seed {seed}: {key}={config.get(key)!r}, expected {wanted!r}")
    scenario = config.get("scenario")
    if not isinstance(scenario, dict) or scenario.get("id") != EXPECTED_SCENARIO:
        raise ValueError(f"seed {seed}: unexpected scenario")
    if not np.isclose(float(config.get("tau", np.nan)), 0.005, rtol=0.0, atol=1e-12):
        raise ValueError(f"seed {seed}: shared reproduction baseline must retain tau=0.005")
    if complete.get("status") != "complete" or complete.get("algorithm") != "mappo":
        raise ValueError(f"seed {seed}: incomplete or wrong-algorithm COMPLETE.json")
    applicability = complete.get("algorithm_applicability")
    if not isinstance(applicability, dict) or any(bool(value) for value in applicability.values()):
        raise ValueError(f"seed {seed}: MAPPO applicability metadata is missing or incorrect")


def _window_metrics(aoi: np.ndarray, cam: np.ndarray, payload: np.ndarray, reward: np.ndarray, count: int):
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
        f"last{count}_mean_reward_proxy": float(reward[-count:].mean()),
    }


def summarize(result_root: Path) -> Dict[str, object]:
    result_root = result_root.expanduser().resolve()
    rows: List[Dict[str, object]] = []
    per_episode: List[Dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        run_name = f"mappo_default_{EXPECTED_SCENARIO}_seed{seed:02d}"
        run_dir = result_root / "runs" / run_name
        config = _read_json(run_dir / "config.resolved.json")
        complete = _read_json(run_dir / "COMPLETE.json")
        _validate(config, complete, seed)
        with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as data:
            aoi = np.asarray(data["mean_aoi_ms_episode_agent"], dtype=np.float64)
            cam = np.asarray(data["endpoint_cam_episode_agent"], dtype=np.float64)
            remaining = np.asarray(data["remaining_demand"], dtype=np.float64)
            reward = np.asarray(data["immediate_reward_proxy"], dtype=np.float64)
            rb_entropy = np.asarray(data["rb_entropy_normalized_episode_agent"], dtype=np.float64)
            mode_entropy = np.asarray(data["mode_entropy_normalized_episode_agent"], dtype=np.float64)
        expected_agent_shape = (500, 5)
        if aoi.shape != expected_agent_shape or cam.shape != expected_agent_shape:
            raise ValueError(f"seed {seed}: unexpected AoI/CAM shapes {aoi.shape}/{cam.shape}")
        if remaining.shape != (500, 100, 5) or reward.shape != expected_agent_shape:
            raise ValueError(f"seed {seed}: unexpected remaining/reward shapes {remaining.shape}/{reward.shape}")
        payload = np.clip(1.0 - remaining[:, -1, :] / float(config["cam_bits"]), 0.0, 1.0)
        row: Dict[str, object] = {"seed": seed, "run_name": run_name}
        row.update(_window_metrics(aoi, cam, payload, reward, 100))
        row.update(_window_metrics(aoi, cam, payload, reward, 50))
        row["last100_rb_entropy"] = float(rb_entropy[-100:].mean())
        row["last100_mode_entropy"] = float(mode_entropy[-100:].mean())
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
                "mean_reward_proxy": float(reward[episode].mean()),
            })

    numeric_fields = [key for key in rows[0] if key.startswith("last")]
    cohort = {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric_fields}
    cohort["success_count_last100"] = int(sum(bool(row["screen_success_last100"]) for row in rows))
    return {
        "experiment": {
            "algorithm": "mappo",
            "scenario": EXPECTED_SCENARIO,
            "P": 5,
            "N": 4,
            "gap_m": 25,
            "episodes": 500,
            "seeds": list(EXPECTED_SEEDS),
            "primary_window": "last100 training episodes",
            "policy": "separate local actors; categorical RB/mode; Beta power",
            "critic": "centralized per-agent state values",
            "polyak_tau": "not applicable (shared baseline metadata remains 0.005)",
            "external_action_noise": "not applicable",
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
        "# MAPPO default confirmation",
        "",
        "P=5, N=4, gap=25 m; 500 episodes; seeds 8--13. Primary evidence is the final 100 training episodes.",
        "Strict binary CAM and continuous payload completion remain separate metrics.",
        "",
        "| seed | AoI mean/worst | binary CAM mean/worst | payload mean/worst | reward | screen |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['last100_mean_aoi_ms']:.3f}/{row['last100_worst_agent_aoi_ms']:.3f} "
            f"| {row['last100_mean_binary_cam']:.4f}/{row['last100_worst_agent_binary_cam']:.4f} "
            f"| {row['last100_mean_payload_completion']:.4f}/{row['last100_worst_agent_payload_completion']:.4f} "
            f"| {row['last100_mean_reward_proxy']:.4f} "
            f"| {'ok' if row['screen_success_last100'] else 'FAIL'} |"
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
    (output_dir / "mappo_default_summary.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "mappo_default.csv", report["rows"])
    _write_csv(output_dir / "mappo_default_per_episode.csv", report["per_episode"])
    _write_markdown(output_dir / "mappo_default.md", report)
    print(json.dumps(compact["cohort"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
