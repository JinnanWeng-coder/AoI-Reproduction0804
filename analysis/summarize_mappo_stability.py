"""Compare the existing MAPPO baseline with two single-factor stability arms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (8, 9, 10, 11, 12, 13)
BLOCKS = ((1, 100), (101, 200), (201, 300), (301, 400), (401, 500))
ARMS = {
    "baseline": {"actor_lr": 0.0005, "entropy_rb": 0.01, "entropy_mode": 0.01, "entropy_power": 0.001},
    "actor_lr1e4": {"actor_lr": 0.0001, "entropy_rb": 0.01, "entropy_mode": 0.01, "entropy_power": 0.001},
    "entropy2x": {"actor_lr": 0.0005, "entropy_rb": 0.02, "entropy_mode": 0.02, "entropy_power": 0.002},
}


def _read_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_name(arm: str, seed: int) -> str:
    if arm == "baseline":
        return f"mappo_default_p05_n04_g25_seed{seed:02d}"
    return f"mappo_stability_{arm}_p05_n04_g25_seed{seed:02d}"


def _mean_nested(records, key: str) -> float:
    return float(np.mean([value for record in records for value in record[key]]))


def _summarize_run(run_dir: Path, arm: str, seed: int):
    config = _read_json(run_dir / "config.resolved.json")
    complete = _read_json(run_dir / "COMPLETE.json")
    expected = ARMS[arm]
    checks = {
        "algorithm": "mappo",
        "seed": seed,
        "episodes": 500,
        "mappo_actor_lr": expected["actor_lr"],
        "mappo_entropy_coef_rb": expected["entropy_rb"],
        "mappo_entropy_coef_mode": expected["entropy_mode"],
        "mappo_entropy_coef_power": expected["entropy_power"],
        "mappo_rollout_episodes": 5,
        "mappo_ppo_epochs": 10,
    }
    for key, wanted in checks.items():
        actual = config.get(key)
        if isinstance(wanted, float):
            if not np.isclose(float(actual), wanted, rtol=0.0, atol=1e-12):
                raise ValueError(f"{run_dir.name}: {key}={actual!r}, expected {wanted!r}")
        elif actual != wanted:
            raise ValueError(f"{run_dir.name}: {key}={actual!r}, expected {wanted!r}")
    if complete.get("status") != "complete" or complete.get("update_count") != 100:
        raise ValueError(f"{run_dir.name}: incomplete training")
    with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as data:
        aoi = np.asarray(data["mean_aoi_ms_episode_agent"], dtype=np.float64)
        cam = np.asarray(data["endpoint_cam_episode_agent"], dtype=np.float64)
        remaining = np.asarray(data["remaining_demand"], dtype=np.float64)
        reward = np.asarray(data["immediate_reward_proxy"], dtype=np.float64)
    if aoi.shape != (500, 5) or cam.shape != (500, 5) or remaining.shape != (500, 100, 5):
        raise ValueError(f"{run_dir.name}: unexpected training metric shapes")
    payload = np.clip(1.0 - remaining[:, -1, :] / float(config["cam_bits"]), 0.0, 1.0)
    diagnostics = json.loads((run_dir / "learning_diagnostics.json").read_text(encoding="utf-8"))
    if not isinstance(diagnostics, list) or len(diagnostics) != 100:
        raise ValueError(f"{run_dir.name}: expected 100 PPO diagnostics")
    last20 = diagnostics[-20:]

    def window(count: int):
        agent_aoi = aoi[-count:].mean(axis=0)
        agent_cam = cam[-count:].mean(axis=0)
        agent_payload = payload[-count:].mean(axis=0)
        return {
            f"last{count}_mean_aoi_ms": float(agent_aoi.mean()),
            f"last{count}_worst_agent_aoi_ms": float(agent_aoi.max()),
            f"last{count}_mean_binary_cam": float(agent_cam.mean()),
            f"last{count}_worst_agent_binary_cam": float(agent_cam.min()),
            f"last{count}_mean_payload_completion": float(agent_payload.mean()),
            f"last{count}_worst_agent_payload_completion": float(agent_payload.min()),
            f"last{count}_mean_reward_proxy": float(reward[-count:].mean()),
        }

    row = {
        "arm": arm,
        "seed": seed,
        "source": "baseline_reuse" if arm == "baseline" else "new_stability_run",
        "run_name": run_dir.name,
        **window(100),
        **window(50),
        "last20_approx_kl": _mean_nested(last20, "approx_kl_per_agent"),
        "last20_clip_fraction": _mean_nested(last20, "clip_fraction_per_agent"),
        "last20_entropy_rb": _mean_nested(last20, "entropy_rb_per_agent"),
        "last20_entropy_mode": _mean_nested(last20, "entropy_mode_per_agent"),
        "last20_entropy_power": _mean_nested(last20, "entropy_power_per_agent"),
        "last20_actor_grad_norm": _mean_nested(last20, "actor_grad_norm_per_agent"),
        "last20_critic_grad_norm": float(np.mean([record["critic_grad_norm"] for record in last20])),
        "last20_explained_variance": float(np.mean([record["explained_variance"] for record in last20])),
    }
    row["screen_success_last100"] = bool(
        row["last100_worst_agent_aoi_ms"] < 50.0
        and row["last100_worst_agent_binary_cam"] >= 0.5
    )
    blocks = []
    for start, end in BLOCKS:
        selection = slice(start - 1, end)
        blocks.append({
            "arm": arm,
            "seed": seed,
            "episode_start": start,
            "episode_end": end,
            "mean_aoi_ms": float(aoi[selection].mean()),
            "worst_agent_aoi_ms": float(aoi[selection].mean(axis=0).max()),
            "mean_binary_cam": float(cam[selection].mean()),
            "worst_agent_binary_cam": float(cam[selection].mean(axis=0).min()),
            "mean_payload_completion": float(payload[selection].mean()),
            "mean_reward_proxy": float(reward[selection].mean()),
        })
    return row, blocks


def summarize(stability_root: Path, baseline_root: Path):
    stability_root = stability_root.expanduser().resolve()
    baseline_root = baseline_root.expanduser().resolve()
    rows, blocks = [], []
    for arm in ARMS:
        source_root = baseline_root if arm == "baseline" else stability_root
        for seed in SEEDS:
            row, run_blocks = _summarize_run(source_root / "runs" / _run_name(arm, seed), arm, seed)
            rows.append(row)
            blocks.extend(run_blocks)

    summaries = []
    numeric_fields = [key for key in rows[0] if key.startswith("last")]
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        summary = {"arm": arm, **ARMS[arm]}
        summary.update({key: float(np.mean([row[key] for row in selected])) for key in numeric_fields})
        summary["success_count_last100"] = int(sum(row["screen_success_last100"] for row in selected))
        first = [row for row in blocks if row["arm"] == arm and row["episode_start"] == 1]
        final = [row for row in blocks if row["arm"] == arm and row["episode_start"] == 401]
        summary["block1_to_block5_delta_aoi_ms"] = float(
            np.mean([row["mean_aoi_ms"] for row in final]) - np.mean([row["mean_aoi_ms"] for row in first])
        )
        summary["block1_to_block5_delta_binary_cam"] = float(
            np.mean([row["mean_binary_cam"] for row in final]) - np.mean([row["mean_binary_cam"] for row in first])
        )
        summaries.append(summary)
    return {"per_seed": rows, "blocks": blocks, "summary": summaries}


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report) -> None:
    lines = [
        "# MAPPO stability comparison",
        "",
        "The existing six-seed baseline is reused; actor_lr1e4 and entropy2x are single-factor arms.",
        "",
        "| arm | AoI mean/worst | binary CAM mean/worst | payload mean/worst | success | late ΔAoI | late ΔCAM | last20 RB/mode entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['arm']} | {row['last100_mean_aoi_ms']:.3f}/{row['last100_worst_agent_aoi_ms']:.3f} "
            f"| {row['last100_mean_binary_cam']:.4f}/{row['last100_worst_agent_binary_cam']:.4f} "
            f"| {row['last100_mean_payload_completion']:.4f}/{row['last100_worst_agent_payload_completion']:.4f} "
            f"| {row['success_count_last100']}/6 | {row['block1_to_block5_delta_aoi_ms']:+.3f} "
            f"| {row['block1_to_block5_delta_binary_cam']:+.4f} "
            f"| {row['last20_entropy_rb']:.3f}/{row['last20_entropy_mode']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.stability_root, args.baseline_root)
    output = args.stability_root.expanduser().resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "mappo_stability_per_seed.csv", report["per_seed"])
    _write_csv(output / "mappo_stability_blocks.csv", report["blocks"])
    _write_csv(output / "mappo_stability_summary.csv", report["summary"])
    (output / "mappo_stability_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(output / "mappo_stability.md", report)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
