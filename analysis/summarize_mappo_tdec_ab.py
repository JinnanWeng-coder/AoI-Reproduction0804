"""Summarize the corrected-value-clipping MAPPO combined/TDec A/B."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (8, 9, 10, 11, 12, 13)
VARIANTS = ("combined", "tdec")
MODES = ("deterministic", "stochastic")
EVAL_SEEDS = (201, 202, 203, 204, 205, 206)
SCENARIO = "p05_n04_g25"
ACTOR_LR = 0.0005
ENTROPY_RB = 0.02
ENTROPY_MODE = 0.02
ENTROPY_POWER = 0.002
VALUE_CLIP_MODE = "normalized"


def _run_name(variant: str, seed: int) -> str:
    return f"mappo_tdec_ab_{variant}_{SCENARIO}_seed{seed:02d}"


def _eval_id(mode: str) -> str:
    seed_token = "-".join(str(seed) for seed in EVAL_SEEDS)
    return f"eval_validation_policy_final_{mode}_sequential_warm_warm5_s{seed_token}_ep100"


def _read_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _mean_nested(records, key: str) -> float:
    return float(np.mean([value for record in records for value in record[key]]))


def _mean_field(records, key: str) -> float:
    return float(np.mean([record[key] for record in records]))


def _validate_training_identity(config, complete, variant: str, seed: int, run_dir: Path) -> None:
    scenario = config.get("scenario", {})
    checks = {
        "algorithm": (config.get("algorithm"), "mappo"),
        "scenario": (scenario.get("id"), SCENARIO),
        "seed": (config.get("seed"), seed),
        "episodes": (config.get("episodes"), 500),
        "mappo_variant": (config.get("mappo_variant"), variant),
        "mappo_value_clip_mode": (config.get("mappo_value_clip_mode"), VALUE_CLIP_MODE),
        "mappo_rollout_episodes": (config.get("mappo_rollout_episodes"), 5),
        "mappo_ppo_epochs": (config.get("mappo_ppo_epochs"), 10),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{run_dir.name}: {key}={actual!r}, expected {expected!r}")
    for key, expected in (
        ("mappo_actor_lr", ACTOR_LR),
        ("mappo_entropy_coef_rb", ENTROPY_RB),
        ("mappo_entropy_coef_mode", ENTROPY_MODE),
        ("mappo_entropy_coef_power", ENTROPY_POWER),
    ):
        if not np.isclose(float(config.get(key)), expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"{run_dir.name}: {key} mismatch")
    if complete.get("status") != "complete" or complete.get("update_count") != 100:
        raise ValueError(f"{run_dir.name}: incomplete training")
    if complete.get("algorithm") != "mappo" or complete.get("mappo_variant") != variant:
        raise ValueError(f"{run_dir.name}: completion identity mismatch")


def _training_row(result_root: Path, variant: str, seed: int):
    run_name = _run_name(variant, seed)
    run_dir = result_root / "training" / "runs" / run_name
    config = _read_json(run_dir / "config.resolved.json")
    complete = _read_json(run_dir / "COMPLETE.json")
    _validate_training_identity(config, complete, variant, seed, run_dir)

    with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as data:
        aoi = np.asarray(data["mean_aoi_ms_episode_agent"], dtype=np.float64)
        cam = np.asarray(data["endpoint_cam_episode_agent"], dtype=np.float64)
        remaining = np.asarray(data["remaining_demand"], dtype=np.float64)
    if aoi.shape != (500, 5) or cam.shape != (500, 5) or remaining.shape != (500, 100, 5):
        raise ValueError(f"{run_dir.name}: unexpected training metric shapes")
    payload = np.clip(1.0 - remaining[:, -1, :] / float(config["cam_bits"]), 0.0, 1.0)

    diagnostics = json.loads((run_dir / "learning_diagnostics.json").read_text(encoding="utf-8"))
    if not isinstance(diagnostics, list) or len(diagnostics) != 100:
        raise ValueError(f"{run_dir.name}: expected 100 PPO diagnostics")
    if any(record.get("mappo_variant") != variant for record in diagnostics):
        raise ValueError(f"{run_dir.name}: diagnostic variant mismatch")
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
        }

    row = {
        "variant": variant,
        "seed": seed,
        "run_name": run_name,
        **window(100),
        **window(50),
        "last20_approx_kl": _mean_nested(last20, "approx_kl_per_agent"),
        "last20_clip_fraction": _mean_nested(last20, "clip_fraction_per_agent"),
        "last20_entropy_rb": _mean_nested(last20, "entropy_rb_per_agent"),
        "last20_entropy_mode": _mean_nested(last20, "entropy_mode_per_agent"),
        "last20_critic_loss": _mean_field(last20, "critic_loss"),
        "last20_explained_variance": _mean_field(last20, "explained_variance"),
        "last20_global_explained_variance": None,
        "last20_task1_explained_variance": None,
        "last20_task2_explained_variance": None,
    }
    if variant == "tdec":
        row.update({
            "last20_global_explained_variance": _mean_field(last20, "global_explained_variance"),
            "last20_task1_explained_variance": _mean_field(last20, "task1_explained_variance"),
            "last20_task2_explained_variance": _mean_field(last20, "task2_explained_variance"),
        })
    row["screen_success_last100"] = bool(
        row["last100_worst_agent_aoi_ms"] < 50.0
        and row["last100_worst_agent_binary_cam"] >= 0.5
    )
    return row


def _evaluation_row(result_root: Path, variant: str, seed: int, mode: str):
    run_name = _run_name(variant, seed)
    eval_dir = result_root / "evaluations" / run_name / _eval_id(mode)
    complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
    checks = {
        "algorithm": "mappo",
        "status": "complete",
        "mappo_variant": variant,
        "diagnostic_evaluation": True,
        "scenario": SCENARIO,
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
    row = {
        "variant": variant,
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


def _paired_rows(training_rows, evaluation_rows):
    rows = []

    def append_pair(phase: str, mode: str, left_rows, right_rows, aoi_key: str, cam_key: str, payload_key: str):
        delta_aoi = np.asarray([left[aoi_key] - right[aoi_key] for left, right in zip(left_rows, right_rows)])
        delta_cam = np.asarray([left[cam_key] - right[cam_key] for left, right in zip(left_rows, right_rows)])
        delta_payload = np.asarray([left[payload_key] - right[payload_key] for left, right in zip(left_rows, right_rows)])
        rows.append({
            "phase": phase,
            "mode": mode,
            "left_variant": "tdec",
            "right_variant": "combined",
            "mean_delta_aoi_ms_tdec_minus_combined": float(delta_aoi.mean()),
            "aoi_wins_tdec": int(np.sum(delta_aoi < 0.0)),
            "mean_delta_binary_cam_tdec_minus_combined": float(delta_cam.mean()),
            "binary_cam_wins_tdec": int(np.sum(delta_cam > 0.0)),
            "mean_delta_payload_tdec_minus_combined": float(delta_payload.mean()),
            "payload_wins_tdec": int(np.sum(delta_payload > 0.0)),
        })

    indexed_train = {(row["variant"], row["seed"]): row for row in training_rows}
    append_pair(
        "training", "last100",
        [indexed_train[("tdec", seed)] for seed in SEEDS],
        [indexed_train[("combined", seed)] for seed in SEEDS],
        "last100_mean_aoi_ms", "last100_mean_binary_cam", "last100_mean_payload_completion",
    )
    indexed_eval = {(row["variant"], row["mode"], row["training_seed"]): row for row in evaluation_rows}
    for mode in MODES:
        append_pair(
            "heldout", mode,
            [indexed_eval[("tdec", mode, seed)] for seed in SEEDS],
            [indexed_eval[("combined", mode, seed)] for seed in SEEDS],
            "mean_aoi_ms", "mean_binary_cam", "mean_payload_completion",
        )
    return rows


def summarize(result_root: Path):
    result_root = result_root.expanduser().resolve()
    training_rows = [
        _training_row(result_root, variant, seed)
        for variant in VARIANTS
        for seed in SEEDS
    ]
    training_summary = []
    for variant in VARIANTS:
        selected = [row for row in training_rows if row["variant"] == variant]
        training_summary.append({
            "variant": variant,
            "mean_aoi_ms": float(np.mean([row["last100_mean_aoi_ms"] for row in selected])),
            "sd_aoi_across_training_seeds": float(np.std([row["last100_mean_aoi_ms"] for row in selected], ddof=1)),
            "worst_agent_aoi_ms": float(np.mean([row["last100_worst_agent_aoi_ms"] for row in selected])),
            "mean_binary_cam": float(np.mean([row["last100_mean_binary_cam"] for row in selected])),
            "worst_agent_binary_cam": float(np.mean([row["last100_worst_agent_binary_cam"] for row in selected])),
            "mean_payload_completion": float(np.mean([row["last100_mean_payload_completion"] for row in selected])),
            "worst_agent_payload_completion": float(np.mean([row["last100_worst_agent_payload_completion"] for row in selected])),
            "success_count": int(sum(row["screen_success_last100"] for row in selected)),
            "last20_approx_kl": float(np.mean([row["last20_approx_kl"] for row in selected])),
            "last20_clip_fraction": float(np.mean([row["last20_clip_fraction"] for row in selected])),
            "last20_explained_variance": float(np.mean([row["last20_explained_variance"] for row in selected])),
        })

    evaluation_rows = [
        _evaluation_row(result_root, variant, seed, mode)
        for variant in VARIANTS
        for seed in SEEDS
        for mode in MODES
    ]
    evaluation_summary = []
    for variant in VARIANTS:
        for mode in MODES:
            selected = [
                row for row in evaluation_rows
                if row["variant"] == variant and row["mode"] == mode
            ]
            aoi = np.asarray([row["mean_aoi_ms"] for row in selected], dtype=np.float64)
            evaluation_summary.append({
                "variant": variant,
                "mode": mode,
                "mean_aoi_ms": float(aoi.mean()),
                "sd_aoi_across_training_seeds": float(aoi.std(ddof=1)),
                "worst_agent_aoi_ms": float(np.mean([row["worst_agent_aoi_ms"] for row in selected])),
                "mean_binary_cam": float(np.mean([row["mean_binary_cam"] for row in selected])),
                "worst_agent_binary_cam": float(np.mean([row["worst_agent_binary_cam"] for row in selected])),
                "mean_payload_completion": float(np.mean([row["mean_payload_completion"] for row in selected])),
                "worst_agent_payload_completion": float(np.mean([row["worst_agent_payload_completion"] for row in selected])),
                "success_count": int(sum(row["screen_success"] for row in selected)),
            })
    return {
        "contract": {
            "scenario": SCENARIO,
            "training_seeds": list(SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
            "episodes": 500,
            "eval_episodes": 100,
            "mappo_actor_lr": ACTOR_LR,
            "mappo_entropy_coef_rb": ENTROPY_RB,
            "mappo_entropy_coef_mode": ENTROPY_MODE,
            "mappo_entropy_coef_power": ENTROPY_POWER,
            "mappo_value_clip_mode": VALUE_CLIP_MODE,
        },
        "training_per_seed": training_rows,
        "training_summary": training_summary,
        "evaluation_per_seed": evaluation_rows,
        "evaluation_summary": evaluation_summary,
        "paired": _paired_rows(training_rows, evaluation_rows),
    }


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report) -> None:
    lines = [
        "# MAPPO combined vs TDec default-scenario A/B",
        "",
        "Both variants use normalized value clipping and the frozen entropy2x hyperparameters. Only task decomposition changes.",
        "",
        "## Training last 100 episodes",
        "",
        "| variant | AoI mean±SD | worst AoI | binary CAM mean/worst | payload mean/worst | success | KL / clip |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["training_summary"]:
        lines.append(
            f"| {row['variant']} | {row['mean_aoi_ms']:.3f}±{row['sd_aoi_across_training_seeds']:.3f} "
            f"| {row['worst_agent_aoi_ms']:.3f} | {row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} "
            f"| {row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} "
            f"| {row['success_count']}/6 | {row['last20_approx_kl']:.4f}/{row['last20_clip_fraction']:.4f} |"
        )
    lines.extend([
        "",
        "## Frozen-policy held-out validation",
        "",
        "| variant | mode | AoI mean±SD | worst AoI | binary CAM mean/worst | payload mean/worst | success |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["evaluation_summary"]:
        lines.append(
            f"| {row['variant']} | {row['mode']} | {row['mean_aoi_ms']:.3f}±{row['sd_aoi_across_training_seeds']:.3f} "
            f"| {row['worst_agent_aoi_ms']:.3f} | {row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} "
            f"| {row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} "
            f"| {row['success_count']}/6 |"
        )
    lines.extend([
        "",
        "Paired deltas are TDec minus combined; negative AoI and positive CAM/payload favor TDec.",
        "",
        "| phase | mode | ΔAoI | TDec AoI wins | Δbinary CAM | TDec CAM wins | Δpayload |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in report["paired"]:
        lines.append(
            f"| {row['phase']} | {row['mode']} | {row['mean_delta_aoi_ms_tdec_minus_combined']:+.3f} "
            f"| {row['aoi_wins_tdec']}/6 | {row['mean_delta_binary_cam_tdec_minus_combined']:+.4f} "
            f"| {row['binary_cam_wins_tdec']}/6 | {row['mean_delta_payload_tdec_minus_combined']:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(result_root: Path, report) -> Path:
    output = result_root.expanduser().resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "mappo_tdec_ab_training_per_seed.csv", report["training_per_seed"])
    _write_csv(output / "mappo_tdec_ab_training_summary.csv", report["training_summary"])
    _write_csv(output / "mappo_tdec_ab_eval_per_seed.csv", report["evaluation_per_seed"])
    _write_csv(output / "mappo_tdec_ab_eval_summary.csv", report["evaluation_summary"])
    _write_csv(output / "mappo_tdec_ab_paired.csv", report["paired"])
    (output / "mappo_tdec_ab_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output / "mappo_tdec_ab.md", report)
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
        "paired": report["paired"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
