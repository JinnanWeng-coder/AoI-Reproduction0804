"""Summarize E1 unified frozen MADDPG evaluations plus reused MAPPO cells."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.e1_contract import (
    ALGORITHMS,
    CHECKPOINT_EPISODE,
    EVAL_EPISODES,
    EVAL_SEEDS,
    EVAL_WARMUP,
    FORBIDDEN_EVAL_SEEDS,
    N_AGENTS,
    NOISE_LEVELS,
    SCENARIO,
    SEEDS,
    STEPS_PER_EPISODE,
    canonical_run_name,
    maddpg_eval_id,
    mappo_eval_root,
    mappo_run_name,
    mappo_training_root,
    MAPPO_EVAL_ID,
)
from analysis.evaluate_maddpg_e1 import reduce_heldout_metrics


METRIC_FIELDS = (
    "mean_aoi_ms",
    "worst_agent_aoi_ms",
    "aoi_gt50_fraction",
    "aoi_at_cap_fraction",
    "mean_binary_cam",
    "worst_agent_binary_cam",
    "mean_payload_completion",
    "worst_agent_payload_completion",
    "mean_reward_combined",
    "mean_power_mw",
)
COUNT_FIELDS = (
    "aoi_gt50_count",
    "aoi_gt50_denominator",
    "aoi_at_cap_count",
    "aoi_at_cap_denominator",
    "power_mw_sum",
    "power_mw_count",
)


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _condition_id(family: str, **parts: Any) -> str:
    if family == "maddpg":
        noise = "0p0" if float(parts["noise"]) == 0.0 else "0p3"
        return f"maddpg_{parts['algorithm_label']}_noise{noise}"
    return f"mappo_{parts['structure']}_{parts['variant']}"


def _world_agent_rows(arrays: Mapping[str, np.ndarray], complete: Mapping[str, Any], extra: Mapping[str, Any]) -> list[dict]:
    aoi = np.asarray(arrays["aoi_ms"], dtype=np.float64)
    success = np.asarray(arrays["success"], dtype=np.float64)
    remaining = np.asarray(arrays["remaining_demand"], dtype=np.float64)
    power_key = "power_dbm" if "power_dbm" in arrays else "executed_power_dbm"
    power = np.asarray(arrays[power_key], dtype=np.float64)
    reward_global = np.asarray(arrays["reward_global"], dtype=np.float64)
    reward_task1 = np.asarray(arrays["reward_task1"], dtype=np.float64)
    reward_task2 = np.asarray(arrays["reward_task2"], dtype=np.float64)
    cam_bits = float(complete["cam_bits"])
    weight = float(complete["global_actor_weight"])
    worlds = list(complete["eval_seeds"])
    if aoi.shape[0] != len(worlds) or aoi.shape[-1] != N_AGENTS:
        raise ValueError("world/agent axes do not match the E1 contract")
    rows = []
    for world_index, world in enumerate(worlds):
        if int(world) in FORBIDDEN_EVAL_SEEDS:
            raise ValueError("refusing MAPPO/MADDPG cells that used worlds 201-206")
        for agent in range(N_AGENTS):
            agent_aoi = aoi[world_index, :, :, agent]
            cam = success[world_index, :, -1, agent]
            payload = np.clip(1.0 - remaining[world_index, :, -1, agent] / cam_bits, 0.0, 1.0)
            combined = (
                reward_task1[world_index, :, :, agent]
                + reward_task2[world_index, :, :, agent]
                + weight * reward_global[world_index]
            )
            power_mw = np.power(10.0, power[world_index, :, :, agent] / 10.0)
            rows.append({
                **extra,
                "eval_seed": int(world),
                "agent": int(agent),
                "mean_aoi_ms": float(agent_aoi.mean()),
                "aoi_gt50_count": int((agent_aoi > 50.0).sum()),
                "aoi_gt50_denominator": int(agent_aoi.size),
                "aoi_gt50_fraction": float((agent_aoi > 50.0).mean()),
                "aoi_at_cap_count": int(np.isclose(agent_aoi, 100.0, rtol=0.0, atol=1e-6).sum()),
                "aoi_at_cap_denominator": int(agent_aoi.size),
                "aoi_at_cap_fraction": float(np.isclose(agent_aoi, 100.0, rtol=0.0, atol=1e-6).mean()),
                "binary_cam": float(cam.mean()),
                "payload_completion": float(payload.mean()),
                "mean_reward_combined": float(combined.mean()),
                "mean_power_mw": float(power_mw.mean()),
                "power_mw_sum": float(power_mw.sum()),
                "power_mw_count": int(power_mw.size),
            })
    return rows


def _load_arrays(eval_dir: Path) -> dict[str, np.ndarray]:
    with np.load(eval_dir / "metrics.npz", allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _verify_mappo_training(diagnostics_root: Path, structure: str, variant: str, seed: int) -> dict[str, Any]:
    root = mappo_training_root(diagnostics_root, structure, variant)
    run_name = mappo_run_name(structure, variant, seed)
    run_dir = root / "training" / "runs" / run_name
    if not (run_dir / "COMPLETE.json").is_file() or not (run_dir / "config.resolved.json").is_file():
        raise ValueError(f"missing reused MAPPO training run: {run_dir}")
    config = _read_json(run_dir / "config.resolved.json")
    complete = _read_json(run_dir / "COMPLETE.json")
    if config.get("algorithm") != "mappo" or complete.get("status") != "complete":
        raise ValueError(f"{run_name}: not a complete MAPPO training run")
    if int(config.get("seed", -1)) != int(seed) or config.get("scenario", {}).get("id") != SCENARIO:
        raise ValueError(f"{run_name}: seed/scenario mismatch")
    if int(config.get("episodes", -1)) != CHECKPOINT_EPISODE:
        raise ValueError(f"{run_name}: MAPPO training is not final-500")
    if complete.get("mappo_variant", variant) != variant:
        raise ValueError(f"{run_name}: MAPPO variant mismatch")
    sharing = bool(config.get("mappo_actor_sharing", False))
    if sharing != (structure == "shared"):
        raise ValueError(f"{run_name}: actor-sharing mismatch")
    if not (run_dir / "policy_final.pt").is_file():
        raise FileNotFoundError(f"{run_name}: missing policy_final.pt")
    return {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "training_commit": complete.get("reproduction_git_commit"),
        "data_role": "reused",
    }


def _maddpg_cell(e1_root: Path, diagnostics_root: Path, algorithm: str, noise: float, seed: int) -> tuple[dict, list[dict]]:
    run_name = canonical_run_name(algorithm, seed)
    eval_dir = e1_root / "evaluations" / run_name / maddpg_eval_id(noise)
    if not (eval_dir / "EVAL_COMPLETE.json").is_file():
        raise ValueError(f"missing MADDPG cell: {eval_dir}")
    complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
    provenance = _read_json(eval_dir / "provenance.json")
    required = {
        "status": "complete",
        "algorithm": algorithm,
        "training_seed": int(seed),
        "checkpoint_name": "latest.pt",
        "checkpoint_episode": CHECKPOINT_EPISODE,
        "eval_protocol": "sequential_warm",
        "eval_episodes": EVAL_EPISODES,
        "eval_warmup_episodes": EVAL_WARMUP,
        "n_rb": 3,
        "explore": False,
        "wrote_to_source_run": False,
        "parameter_snapshot_unchanged": True,
    }
    for key, expected in required.items():
        if complete.get(key) != expected:
            raise ValueError(f"{eval_dir}: {key}={complete.get(key)!r}, expected {expected!r}")
    if complete.get("eval_seeds") != list(EVAL_SEEDS):
        raise ValueError(f"{eval_dir}: eval_seeds are not worlds 213-218")
    if not np.isclose(float(complete.get("noise_std", -1)), float(noise), atol=1e-12):
        raise ValueError(f"{eval_dir}: noise mismatch")
    arrays = _load_arrays(eval_dir)
    if arrays["aoi_ms"].shape != (len(EVAL_SEEDS), EVAL_EPISODES, STEPS_PER_EPISODE, N_AGENTS):
        raise ValueError(f"{eval_dir}: unexpected metrics shape {arrays['aoi_ms'].shape}")
    metrics = reduce_heldout_metrics(
        aoi=arrays["aoi_ms"],
        success=arrays["success"],
        remaining=arrays["remaining_demand"],
        power_dbm=arrays["power_dbm"],
        reward_global=arrays["reward_global"],
        reward_task1=arrays["reward_task1"],
        reward_task2=arrays["reward_task2"],
        cam_bits=float(complete["cam_bits"]),
        global_actor_weight=float(complete["global_actor_weight"]),
    )
    extra = {
        "family": "maddpg",
        "condition": _condition_id("maddpg", algorithm_label=complete["algorithm_label"], noise=noise),
        "algorithm": algorithm,
        "algorithm_label": complete["algorithm_label"],
        "structure": "independent",
        "variant": complete["algorithm_label"],
        "noise_std": float(noise),
        "training_seed": int(seed),
        "run_name": run_name,
        "data_role": "new",
        "source_root": str(Path(complete["source_run_dir"]).resolve()),
        "training_root": str(Path(complete["source_run_dir"]).resolve()),
        "evaluation_commit": complete.get("reproduction_git_commit"),
        "evaluation_branch": complete.get("reproduction_git_branch"),
        "evaluation_dirty": complete.get("reproduction_git_dirty"),
        "training_commit": complete.get("reproduction_git_commit"),
        "checkpoint_episode": CHECKPOINT_EPISODE,
    }
    row = {**extra, **metrics, "phase": "heldout_stochastic"}
    streams = _world_agent_rows(arrays, complete, extra)
    if provenance.get("checkpoint_name") != "latest.pt":
        raise ValueError(f"{eval_dir}: provenance is not latest.pt")
    return row, streams


def _mappo_cell(diagnostics_root: Path, structure: str, variant: str, seed: int) -> tuple[dict, list[dict]]:
    training = _verify_mappo_training(diagnostics_root, structure, variant, seed)
    eval_root = mappo_eval_root(diagnostics_root, structure, variant)
    run_name = mappo_run_name(structure, variant, seed)
    eval_dir = eval_root / "evaluations" / run_name / MAPPO_EVAL_ID
    if not (eval_dir / "EVAL_COMPLETE.json").is_file():
        raise ValueError(f"missing reused MAPPO cell: {eval_dir}")
    complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
    if complete.get("status") != "complete" or complete.get("algorithm") != "mappo":
        raise ValueError(f"{eval_dir}: not a complete MAPPO evaluation")
    if complete.get("eval_seeds") != list(EVAL_SEEDS):
        raise ValueError(f"{eval_dir}: MAPPO eval is not worlds 213-218")
    if any(int(world) in FORBIDDEN_EVAL_SEEDS for world in complete.get("eval_seeds", [])):
        raise ValueError(f"{eval_dir}: refusing reserved 201-206 eval worlds")
    if int(complete.get("eval_episodes", -1)) != EVAL_EPISODES or int(complete.get("eval_warmup_episodes", -1)) != EVAL_WARMUP:
        raise ValueError(f"{eval_dir}: MAPPO warmup/episode protocol mismatch")
    if complete.get("mappo_eval_mode") != "stochastic" or complete.get("eval_protocol", "sequential_warm") not in {"sequential_warm", None}:
        # feasibility evals record sequential_warm via training config; require stochastic held-out
        if complete.get("mappo_eval_mode") != "stochastic":
            raise ValueError(f"{eval_dir}: MAPPO eval is not stochastic")
    arrays = _load_arrays(eval_dir)
    if arrays["aoi_ms"].shape != (len(EVAL_SEEDS), EVAL_EPISODES, STEPS_PER_EPISODE, N_AGENTS):
        raise ValueError(f"{eval_dir}: unexpected MAPPO metrics shape")
    power = arrays["executed_power_dbm"] if "executed_power_dbm" in arrays else arrays["power_dbm"]
    metrics = reduce_heldout_metrics(
        aoi=arrays["aoi_ms"],
        success=arrays["success"],
        remaining=arrays["remaining_demand"],
        power_dbm=power,
        reward_global=arrays["reward_global"],
        reward_task1=arrays["reward_task1"],
        reward_task2=arrays["reward_task2"],
        cam_bits=float(complete["cam_bits"]),
        global_actor_weight=float(complete["global_actor_weight"]),
    )
    extra = {
        "family": "mappo",
        "condition": _condition_id("mappo", structure=structure, variant=variant),
        "algorithm": "mappo",
        "algorithm_label": "mappo",
        "structure": structure,
        "variant": variant,
        "noise_std": None,
        "training_seed": int(seed),
        "run_name": run_name,
        "data_role": "reused",
        "source_root": str(eval_root),
        "training_root": training["run_dir"],
        "evaluation_commit": complete.get("reproduction_git_commit"),
        "evaluation_branch": complete.get("reproduction_git_branch"),
        "evaluation_dirty": complete.get("reproduction_git_dirty"),
        "training_commit": training["training_commit"],
        "checkpoint_episode": CHECKPOINT_EPISODE,
    }
    row = {**extra, **metrics, "phase": "heldout_stochastic"}
    complete_for_streams = {**complete, "eval_seeds": list(EVAL_SEEDS)}
    arrays_for_streams = {**arrays, "power_dbm": power}
    return row, _world_agent_rows(arrays_for_streams, complete_for_streams, extra)


def _aggregate(rows: Sequence[Mapping], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    if len(selected) != 6:
        raise ValueError(f"expected n=6 independent training repeats for {condition}, found {len(selected)}")
    seeds = [int(row["training_seed"]) for row in selected]
    if sorted(seeds) != list(SEEDS):
        raise ValueError(f"{condition}: training seeds must be 8-13 without dropping any seed")
    result = {
        "condition": condition,
        "family": selected[0]["family"],
        "algorithm": selected[0]["algorithm"],
        "structure": selected[0]["structure"],
        "variant": selected[0]["variant"],
        "noise_std": selected[0]["noise_std"],
        "data_role": selected[0]["data_role"],
        "training_seed_count": 6,
    }
    for metric in METRIC_FIELDS + COUNT_FIELDS:
        values = np.asarray([row[metric] for row in selected], dtype=np.float64)
        result[metric] = float(values.mean())
        result[f"{metric}_sd_across_training_seeds"] = float(values.std(ddof=1))
    return result


def _noise_pairs(rows: Sequence[Mapping]) -> list[dict[str, Any]]:
    pairs = []
    indexed = {
        (row["algorithm"], float(row["noise_std"]), int(row["training_seed"])): row
        for row in rows if row["family"] == "maddpg"
    }
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            quiet = indexed[(algorithm, 0.0, seed)]
            noisy = indexed[(algorithm, 0.3, seed)]
            if quiet["run_name"] != noisy["run_name"]:
                raise ValueError("noise pairing must use the same checkpoint/run")
            row = {
                "algorithm": algorithm,
                "algorithm_label": quiet["algorithm_label"],
                "training_seed": seed,
                "run_name": quiet["run_name"],
                "delta_definition": "noise0p3_minus_noise0",
                "same_checkpoint": True,
            }
            for metric in METRIC_FIELDS:
                row[f"delta_{metric}"] = float(noisy[metric]) - float(quiet[metric])
            pairs.append(row)
    return pairs


def _pair_summary(pairs: Sequence[Mapping]) -> list[dict[str, Any]]:
    rows = []
    for algorithm in ALGORITHMS:
        selected = [row for row in pairs if row["algorithm"] == algorithm]
        if len(selected) != 6:
            raise ValueError(f"expected six noise pairs for {algorithm}")
        row = {
            "algorithm": algorithm,
            "algorithm_label": selected[0]["algorithm_label"],
            "training_seed_count": 6,
            "delta_definition": "noise0p3_minus_noise0",
        }
        for metric in METRIC_FIELDS:
            values = np.asarray([item[f"delta_{metric}"] for item in selected], dtype=np.float64)
            row[f"mean_delta_{metric}"] = float(values.mean())
            row[f"sd_delta_{metric}"] = float(values.std(ddof=1))
        rows.append(row)
    return rows


def summarize_actor_sharing_e1(diagnostics_root: Path, e1_root: Path) -> dict[str, Any]:
    diagnostics_root = Path(diagnostics_root).resolve()
    e1_root = Path(e1_root).resolve()
    cells: list[dict[str, Any]] = []
    streams: list[dict[str, Any]] = []
    identities: set[tuple] = set()

    for algorithm in ALGORITHMS:
        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                row, world_rows = _maddpg_cell(e1_root, diagnostics_root, algorithm, noise, seed)
                identity = ("maddpg", algorithm, noise, seed)
                if identity in identities:
                    raise ValueError(f"duplicate MADDPG cell: {identity}")
                identities.add(identity)
                cells.append(row)
                streams.extend(world_rows)
    for structure in ("independent", "shared"):
        for variant in ("combined", "tdec"):
            for seed in SEEDS:
                row, world_rows = _mappo_cell(diagnostics_root, structure, variant, seed)
                identity = ("mappo", structure, variant, seed)
                if identity in identities:
                    raise ValueError(f"duplicate MAPPO cell: {identity}")
                identities.add(identity)
                cells.append(row)
                streams.extend(world_rows)

    if len(cells) != 48:
        raise ValueError(f"expected 48 cells, found {len(cells)}")
    new_cells = [row for row in cells if row["data_role"] == "new"]
    reused_cells = [row for row in cells if row["data_role"] == "reused"]
    if len(new_cells) != 24 or len(reused_cells) != 24:
        raise ValueError("expected 24 new MADDPG cells and 24 reused MAPPO cells")
    maddpg_streams = [row for row in streams if row["family"] == "maddpg"]
    if len(maddpg_streams) != 720:
        raise ValueError(f"MADDPG world×agent table must have 720 rows, found {len(maddpg_streams)}")
    if len(streams) != 1440:
        raise ValueError(f"merged world×agent table must have 1440 rows, found {len(streams)}")

    conditions = sorted({row["condition"] for row in cells})
    if len(conditions) != 8:
        raise ValueError(f"expected eight conditions, found {conditions}")
    summaries = [_aggregate(cells, condition) for condition in conditions]
    pairs = _noise_pairs(cells)
    pair_summaries = _pair_summary(pairs)
    return {
        "status": "PASS",
        "new_maddpg_cells": 24,
        "reused_mappo_cells": 24,
        "total_cells": 48,
        "conditions": 8,
        "repeats_per_condition": 6,
        "maddpg_world_agent_rows": 720,
        "merged_world_agent_rows": 1440,
        "contract": {
            "scenario": SCENARIO,
            "training_seeds": list(SEEDS),
            "eval_seeds": list(EVAL_SEEDS),
            "worlds_are_reused_development_worlds": True,
            "checkpoint": "latest.pt",
            "checkpoint_episode": CHECKPOINT_EPISODE,
            "noise_levels": list(NOISE_LEVELS),
        },
        "summary": summaries,
        "evaluation_per_seed": cells,
        "eval_world_agent": streams,
        "maddpg_eval_world_agent": maddpg_streams,
        "noise_pairs_per_seed": pairs,
        "noise_pair_summary": pair_summaries,
    }


def _write_figure(e1_root: Path, summaries: Sequence[Mapping]) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    figure_dir = e1_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["condition"]) for row in summaries]
    means = [float(row["mean_aoi_ms"]) for row in summaries]
    sds = [float(row["mean_aoi_ms_sd_across_training_seeds"]) for row in summaries]
    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.bar(range(len(labels)), means, yerr=sds, capsize=4, color="#4C72B0")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylabel("Mean AoI (ms)")
    axis.set_title("E1 held-out worlds 213-218 (development eval, n=6 training seeds)")
    fig.tight_layout()
    path = figure_dir / "e1_mean_aoi_by_condition.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_report(e1_root: Path, report: Mapping[str, Any]) -> Path:
    output = Path(e1_root).resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("e1_summary.csv", "summary"),
        ("e1_evaluation_per_seed.csv", "evaluation_per_seed"),
        ("e1_eval_world_agent.csv", "eval_world_agent"),
        ("e1_maddpg_eval_world_agent.csv", "maddpg_eval_world_agent"),
        ("e1_noise_pairs_per_seed.csv", "noise_pairs_per_seed"),
        ("e1_noise_pair_summary.csv", "noise_pair_summary"),
    ):
        _write_csv(output / name, report[key])
    compact = {key: value for key, value in report.items() if key not in {"eval_world_agent", "maddpg_eval_world_agent"}}
    (output / "e1_summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# E1 unified frozen evaluation",
        "",
        "24 new MADDPG cells plus 24 reused MAPPO cells on development worlds 213-218.",
        "This is not a locked or final test. Training seeds 8-13 are the independent repeats;",
        "world, agent, episode, and slot are not independent training repeats.",
        "",
        "| condition | family | role | AoI mean±SD | worst AoI | AoI>50/cap | binary CAM mean/worst | payload mean/worst | reward | power mW |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['condition']} | {row['family']} | {row['data_role']} | "
            f"{row['mean_aoi_ms']:.3f}±{row['mean_aoi_ms_sd_across_training_seeds']:.3f} | "
            f"{row['worst_agent_aoi_ms']:.3f} | {row['aoi_gt50_fraction']:.4f}/{row['aoi_at_cap_fraction']:.4f} | "
            f"{row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} | "
            f"{row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} | "
            f"{row['mean_reward_combined']:.4f} | {row['mean_power_mw']:.3f} |"
        )
    lines += [
        "",
        "## MADDPG noise pairing (0.3 − 0, same latest.pt)",
        "",
        "| algorithm | ΔAoI | Δbinary CAM | Δpayload | Δreward | Δpower mW |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["noise_pair_summary"]:
        lines.append(
            f"| {row['algorithm_label']} | {row['mean_delta_mean_aoi_ms']:+.3f} | "
            f"{row['mean_delta_mean_binary_cam']:+.4f} | {row['mean_delta_mean_payload_completion']:+.4f} | "
            f"{row['mean_delta_mean_reward_combined']:+.4f} | {row['mean_delta_mean_power_mw']:+.3f} |"
        )
    lines.append("")
    (output / "e1_report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_figure(e1_root, report["summary"])
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--e1-root", required=True, type=Path)
    args = parser.parse_args(argv)
    report = summarize_actor_sharing_e1(args.diagnostics_root, args.e1_root)
    output = write_report(args.e1_root, report)
    print(json.dumps({
        "status": report["status"],
        "new_maddpg_cells": report["new_maddpg_cells"],
        "reused_mappo_cells": report["reused_mappo_cells"],
        "total_cells": report["total_cells"],
        "maddpg_world_agent_rows": report["maddpg_world_agent_rows"],
        "merged_world_agent_rows": report["merged_world_agent_rows"],
        "output": str(output),
        "summary": report["summary"],
        "noise_pair_summary": report["noise_pair_summary"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
