"""Summarize the E5 actor-sharing x combined/TDec 2x2 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.evaluate_mappo_feasibility_reward import feasibility_eval_id
from analysis.mappo_e5_contract import EVAL_SEEDS, SCENARIO, SEEDS, independent_run_name, shared_run_name
from analysis.summarize_mappo_shared_actor import _metric_block


VARIANTS = ("combined", "tdec")
STRUCTURES = ("independent", "shared")
METRICS = (
    "mean_aoi_ms", "worst_agent_aoi_ms", "aoi_gt50_fraction", "aoi_at_cap_fraction",
    "mean_binary_cam", "worst_agent_binary_cam", "mean_payload_completion",
    "worst_agent_payload_completion", "mean_reward_combined", "mean_power_mw",
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _run_name(structure: str, variant: str, seed: int) -> str:
    return independent_run_name(variant, seed) if structure == "independent" else shared_run_name(variant, seed)


def _training_root(tdec_root: Path, shared_tdec_root: Path, e5_root: Path, structure: str, variant: str) -> Path:
    if structure == "independent":
        return tdec_root
    return shared_tdec_root if variant == "tdec" else e5_root


def _evaluation_root(shared_tdec_root: Path, e5_root: Path, variant: str) -> Path:
    return shared_tdec_root if variant == "tdec" else e5_root


def _validate_config(config, complete, structure: str, variant: str, seed: int, run_dir: Path) -> None:
    checks = {
        "algorithm": (config.get("algorithm"), "mappo"),
        "scenario": (config.get("scenario", {}).get("id"), SCENARIO),
        "seed": (int(config.get("seed", -1)), seed),
        "variant": (config.get("mappo_variant", "combined"), variant),
        "sharing": (bool(config.get("mappo_actor_sharing", False)), structure == "shared"),
        "episodes": (int(config.get("episodes", -1)), 500),
        "steps": (int(config.get("steps_per_episode", -1)), 100),
        "rollout": (int(config.get("mappo_rollout_episodes", -1)), 5),
        "epochs": (int(config.get("mappo_ppo_epochs", -1)), 10),
        "clip": (config.get("mappo_value_clip_mode"), "normalized"),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"{run_dir.name}: {label}={actual!r}, expected {expected!r}")
    if complete.get("status") != "complete" or complete.get("mappo_variant", "combined") != variant:
        raise ValueError(f"{run_dir.name}: invalid completion marker")
    if bool(complete.get("actor_sharing", False)) != (structure == "shared"):
        raise ValueError(f"{run_dir.name}: completion actor-sharing mismatch")


def _training_cell(tdec_root, shared_tdec_root, e5_root, structure, variant, seed):
    base = _training_root(tdec_root, shared_tdec_root, e5_root, structure, variant)
    run_name = _run_name(structure, variant, seed)
    run_dir = base / "training" / "runs" / run_name
    config, complete = _read_json(run_dir / "config.resolved.json"), _read_json(run_dir / "COMPLETE.json")
    _validate_config(config, complete, structure, variant, seed, run_dir)
    with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    required = ("aoi_ms", "success", "remaining_demand", "rb", "mode", "power_dbm", "global_step", "task1_step", "task2_step")
    missing = [key for key in required if key not in arrays]
    if missing or arrays["aoi_ms"].shape != (500, 100, 5):
        raise ValueError(f"{run_name}: invalid training arrays; missing={missing}")
    metrics = _metric_block(
        aoi=arrays["aoi_ms"][-100:], success=arrays["success"][-100:],
        remaining=arrays["remaining_demand"][-100:], rb=arrays["rb"][-100:], mode=arrays["mode"][-100:],
        power_dbm=arrays["power_dbm"][-100:], reward_global=arrays["global_step"][-100:],
        reward_task1=arrays["task1_step"][-100:], reward_task2=arrays["task2_step"][-100:],
        cam_bits=float(config["cam_bits"]), global_actor_weight=float(config["global_actor_weight"]),
    )
    actor_count = 1 if structure == "shared" else 5
    expected_steps = 100 * 10 * actor_count
    recorded_steps = complete.get("actor_optimizer_step_count")
    actor_steps = int(recorded_steps) if recorded_steps is not None else expected_steps
    if actor_steps != expected_steps:
        raise ValueError(f"{run_name}: actor optimizer steps {actor_steps}, expected {expected_steps}")
    counts = complete.get("parameter_counts", {})
    row = {
        "phase": "train_last100", "structure": structure, "variant": variant, "training_seed": seed,
        "run_name": run_name, "data_role": "new" if (structure, variant) == ("shared", "combined") else "reused",
        "source_root": str(base), "training_commit": complete.get("reproduction_git_commit"),
        "actor_network_count": actor_count, "actor_parameter_count": int(counts.get("actors", -1)),
        "total_parameter_count": int(counts.get("total", -1)), "ppo_update_count": int(complete.get("update_count", -1)),
        "actor_optimizer_step_count": actor_steps,
        "actor_optimizer_step_count_source": "recorded" if recorded_steps is not None else "derived_updates_x_epochs_x_actors",
        **metrics,
    }
    episodes = []
    for index in range(500):
        episode_metrics = _metric_block(
            aoi=arrays["aoi_ms"][index:index + 1], success=arrays["success"][index:index + 1],
            remaining=arrays["remaining_demand"][index:index + 1], rb=arrays["rb"][index:index + 1],
            mode=arrays["mode"][index:index + 1], power_dbm=arrays["power_dbm"][index:index + 1],
            reward_global=arrays["global_step"][index:index + 1], reward_task1=arrays["task1_step"][index:index + 1],
            reward_task2=arrays["task2_step"][index:index + 1], cam_bits=float(config["cam_bits"]),
            global_actor_weight=float(config["global_actor_weight"]),
        )
        episodes.append({"structure": structure, "variant": variant, "training_seed": seed, "episode": index + 1, **episode_metrics})
    return row, episodes


def _evaluation_cell(shared_tdec_root, e5_root, structure, variant, seed):
    base = _evaluation_root(shared_tdec_root, e5_root, variant)
    run_name = _run_name(structure, variant, seed)
    eval_id = feasibility_eval_id("baseline", EVAL_SEEDS, 100, 5)
    eval_dir = base / "evaluations" / run_name / eval_id
    complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
    checks = {
        "status": "complete", "algorithm": "mappo", "mappo_variant": variant,
        "actor_sharing": structure == "shared", "actor_network_count": 1 if structure == "shared" else 5,
        "training_seed": seed, "training_run_name": run_name, "eval_seeds": list(EVAL_SEEDS),
        "eval_episodes": 100, "eval_warmup_episodes": 5, "mappo_eval_mode": "stochastic",
        "intervention_arm": "baseline", "policy_parameters_unchanged": True,
    }
    for key, expected in checks.items():
        if complete.get(key) != expected:
            raise ValueError(f"{eval_dir}: {key}={complete.get(key)!r}, expected {expected!r}")
    with np.load(eval_dir / "metrics.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    if arrays["aoi_ms"].shape != (6, 100, 100, 5):
        raise ValueError(f"{eval_dir}: invalid evaluation array shape")
    metrics = _metric_block(
        aoi=arrays["aoi_ms"], success=arrays["success"], remaining=arrays["remaining_demand"],
        rb=arrays["rb"], mode=arrays["mode"], power_dbm=arrays["executed_power_dbm"],
        reward_global=arrays["reward_global"], reward_task1=arrays["reward_task1"], reward_task2=arrays["reward_task2"],
        cam_bits=float(complete["cam_bits"]), global_actor_weight=float(complete["global_actor_weight"]),
    )
    row = {
        "phase": "heldout_stochastic", "structure": structure, "variant": variant, "training_seed": seed,
        "actor_network_count": int(complete["actor_network_count"]),
        "run_name": run_name, "data_role": "new" if variant == "combined" else "reused",
        "source_root": str(base), "training_commit": complete.get("training_source_commit"),
        "evaluation_commit": complete.get("reproduction_git_commit"), **metrics,
    }
    streams = []
    weight, cam_bits = float(complete["global_actor_weight"]), float(complete["cam_bits"])
    for wi, world in enumerate(EVAL_SEEDS):
        for agent in range(5):
            aoi = arrays["aoi_ms"][wi, :, :, agent]
            cam = arrays["success"][wi, :, -1, agent]
            remaining = arrays["remaining_demand"][wi, :, -1, agent]
            rg, t1, t2 = arrays["reward_global"][wi], arrays["reward_task1"][wi, :, :, agent], arrays["reward_task2"][wi, :, :, agent]
            streams.append({
                "structure": structure, "variant": variant, "training_seed": seed, "eval_seed": world, "agent": agent,
                "mean_aoi_ms": float(aoi.mean()), "aoi_gt50_fraction": float((aoi > 50).mean()),
                "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, atol=1e-6, rtol=0).mean()),
                "binary_cam": float(cam.mean()), "payload_completion": float(np.clip(1.0 - remaining / cam_bits, 0, 1).mean()),
                "mean_reward_combined": float((t1 + t2 + weight * rg).mean()),
                "mean_power_mw": float(np.power(10.0, arrays["executed_power_dbm"][wi, :, :, agent] / 10.0).mean()),
            })
    return row, streams


def _aggregate(rows, phase, structure, variant):
    selected = [row for row in rows if row["phase"] == phase and row["structure"] == structure and row["variant"] == variant]
    if len(selected) != 6:
        raise ValueError(f"expected six rows for {phase}/{structure}/{variant}")
    result = {"phase": phase, "structure": structure, "variant": variant, "training_seed_count": 6,
              "screen_success_count": sum(bool(row["screen_success"]) for row in selected)}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in selected], dtype=np.float64)
        result[metric] = float(values.mean()); result[f"{metric}_sd_across_training_seeds"] = float(values.std(ddof=1))
    for key in ("actor_network_count", "actor_parameter_count", "total_parameter_count", "ppo_update_count", "actor_optimizer_step_count"):
        if key in selected[0]:
            values = {int(row[key]) for row in selected}
            result[key] = values.pop() if len(values) == 1 else -1
    return result


def _effects(rows, phase):
    indexed = {(row["structure"], row["variant"], row["training_seed"]): row for row in rows if row["phase"] == phase}
    effects, interactions = [], []
    for seed in SEEDS:
        by_variant = {}
        for variant in VARIANTS:
            independent, shared = indexed[("independent", variant, seed)], indexed[("shared", variant, seed)]
            effect = {"phase": phase, "variant": variant, "training_seed": seed, "delta_definition": "shared_minus_independent"}
            for metric in METRICS:
                effect[f"delta_{metric}"] = float(shared[metric]) - float(independent[metric])
            effects.append(effect); by_variant[variant] = effect
        interaction = {"phase": phase, "training_seed": seed,
                       "delta_definition": "(shared_minus_independent)_tdec_minus_(shared_minus_independent)_combined"}
        for metric in METRICS:
            interaction[f"interaction_{metric}"] = by_variant["tdec"][f"delta_{metric}"] - by_variant["combined"][f"delta_{metric}"]
        interactions.append(interaction)
    return effects, interactions


def _effect_summaries(effects, interactions):
    sharing_summary = []
    phases = [phase for phase in ("train_last100", "heldout_stochastic") if any(row["phase"] == phase for row in effects)]
    for phase in phases:
        for variant in VARIANTS:
            selected = [row for row in effects if row["phase"] == phase and row["variant"] == variant]
            row = {"phase": phase, "variant": variant, "training_seed_count": len(selected),
                   "delta_definition": "shared_minus_independent"}
            for metric in METRICS:
                values = np.asarray([item[f"delta_{metric}"] for item in selected], dtype=np.float64)
                row[f"mean_delta_{metric}"] = float(values.mean())
                row[f"sd_delta_{metric}"] = float(values.std(ddof=1))
            sharing_summary.append(row)
    interaction_summary = []
    for phase in phases:
        selected = [row for row in interactions if row["phase"] == phase]
        row = {"phase": phase, "training_seed_count": len(selected),
               "delta_definition": "(shared_minus_independent)_tdec_minus_(shared_minus_independent)_combined"}
        for metric in METRICS:
            values = np.asarray([item[f"interaction_{metric}"] for item in selected], dtype=np.float64)
            row[f"mean_interaction_{metric}"] = float(values.mean())
            row[f"sd_interaction_{metric}"] = float(values.std(ddof=1))
        interaction_summary.append(row)
    return sharing_summary, interaction_summary


def summarize(tdec_root: Path, shared_tdec_root: Path, e5_root: Path) -> dict:
    tdec_root, shared_tdec_root, e5_root = tdec_root.resolve(), shared_tdec_root.resolve(), e5_root.resolve()
    training, episodes, evaluation, streams = [], [], [], []
    source_ids = set()
    for structure in STRUCTURES:
        for variant in VARIANTS:
            for seed in SEEDS:
                trow, erows = _training_cell(tdec_root, shared_tdec_root, e5_root, structure, variant, seed)
                vrow, srows = _evaluation_cell(shared_tdec_root, e5_root, structure, variant, seed)
                identity = (trow["phase"], structure, variant, seed, trow["run_name"])
                if identity in source_ids:
                    raise ValueError(f"duplicate training cell: {identity}")
                source_ids.add(identity); training.append(trow); episodes.extend(erows); evaluation.append(vrow); streams.extend(srows)
    rows = training + evaluation
    effects, interactions = [], []
    for phase in ("train_last100", "heldout_stochastic"):
        e, i = _effects(rows, phase); effects.extend(e); interactions.extend(i)
    sharing_summary, interaction_summary = _effect_summaries(effects, interactions)
    summaries = [_aggregate(rows, phase, structure, variant) for phase in ("train_last100", "heldout_stochastic") for structure in STRUCTURES for variant in VARIANTS]
    return {
        "status": "PASS", "training_cells": len(training), "new_training_cells": sum(r["data_role"] == "new" for r in training),
        "evaluation_cells": len(evaluation), "new_evaluation_cells": sum(r["data_role"] == "new" for r in evaluation),
        "reused_evaluation_cells": sum(r["data_role"] == "reused" for r in evaluation),
        "training_episode_rows": len(episodes), "eval_world_agent_rows": len(streams),
        "contract": {"scenario": SCENARIO, "training_seeds": list(SEEDS), "eval_seeds": list(EVAL_SEEDS),
                     "evaluation_mode": "stochastic", "worlds_are_reused_development_worlds": True,
                     "interaction_definition": "(shared-independent)_tdec - (shared-independent)_combined"},
        "summary": summaries, "training_per_seed": training, "evaluation_per_seed": evaluation,
        "sharing_effect_per_seed": effects, "interaction_per_seed": interactions,
        "sharing_effect_summary": sharing_summary, "interaction_summary": interaction_summary,
        "training_per_episode": episodes, "eval_world_agent": streams,
    }


def write_report(e5_root: Path, report: Mapping) -> Path:
    output = e5_root.resolve() / "analysis"; output.mkdir(parents=True, exist_ok=True)
    for name, key in (
        ("e5_summary.csv", "summary"), ("e5_training_per_seed.csv", "training_per_seed"),
        ("e5_evaluation_per_seed.csv", "evaluation_per_seed"), ("e5_sharing_effect_per_seed.csv", "sharing_effect_per_seed"),
        ("e5_interaction_per_seed.csv", "interaction_per_seed"),
        ("e5_sharing_effect_summary.csv", "sharing_effect_summary"), ("e5_interaction_summary.csv", "interaction_summary"),
        ("e5_training_per_episode.csv", "training_per_episode"),
        ("e5_eval_world_agent.csv", "eval_world_agent"),
    ):
        _write_csv(output / name, report[key])
    compact = {key: value for key, value in report.items() if key not in {"training_per_episode", "eval_world_agent"}}
    (output / "e5_summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# E5 actor sharing x value structure", "",
             "A 2x2 descriptive comparison on reused development worlds 213--218; this is not a locked or final test.", "",
             "| phase | actor | value | AoI mean±SD | worst AoI | AoI>50/cap | binary CAM mean/worst | payload mean/worst | reward | power mW | screen | actors/steps |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        lines.append(f"| {row['phase']} | {row['structure']} | {row['variant']} | {row['mean_aoi_ms']:.3f}±{row['mean_aoi_ms_sd_across_training_seeds']:.3f} | {row['worst_agent_aoi_ms']:.3f} | {row['aoi_gt50_fraction']:.4f}/{row['aoi_at_cap_fraction']:.4f} | {row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} | {row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} | {row['mean_reward_combined']:.4f} | {row['mean_power_mw']:.3f} | {row['screen_success_count']}/6 | {row.get('actor_network_count', '-')}/{row.get('actor_optimizer_step_count', '-')} |")
    lines += ["", "Sharing effects are shared minus independent. The interaction is `(shared-independent)_TDec - (shared-independent)_combined`; training seeds are the six independent repeats.", ""]
    lines += ["## Sharing effects and descriptive interaction", "",
              "| phase | value structure | ΔAoI | Δbinary CAM | Δpayload | Δreward | Δpower mW |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in report["sharing_effect_summary"]:
        lines.append(f"| {row['phase']} | {row['variant']} | {row['mean_delta_mean_aoi_ms']:+.3f} | {row['mean_delta_mean_binary_cam']:+.4f} | {row['mean_delta_mean_payload_completion']:+.4f} | {row['mean_delta_mean_reward_combined']:+.4f} | {row['mean_delta_mean_power_mw']:+.3f} |")
    lines += ["", "| phase | interaction ΔAoI | interaction Δbinary CAM | interaction Δpayload | interaction Δreward |",
              "|---|---:|---:|---:|---:|"]
    for row in report["interaction_summary"]:
        lines.append(f"| {row['phase']} | {row['mean_interaction_mean_aoi_ms']:+.3f} | {row['mean_interaction_mean_binary_cam']:+.4f} | {row['mean_interaction_mean_payload_completion']:+.4f} | {row['mean_interaction_mean_reward_combined']:+.4f} |")
    lines.append("")
    (output / "e5_report.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdec-ab-root", required=True, type=Path)
    parser.add_argument("--shared-tdec-root", required=True, type=Path)
    parser.add_argument("--e5-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.tdec_ab_root, args.shared_tdec_root, args.e5_root)
    output = write_report(args.e5_root, report)
    print(json.dumps({key: report[key] for key in ("status", "training_cells", "new_training_cells", "evaluation_cells", "new_evaluation_cells", "reused_evaluation_cells", "training_episode_rows", "eval_world_agent_rows")} | {"output": str(output), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
