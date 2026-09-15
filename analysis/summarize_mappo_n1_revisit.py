"""Validate and summarize N1 training-world revisits against E5 development worlds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analysis.evaluate_mappo_feasibility_reward import feasibility_eval_id
from analysis.mappo_e5_contract import EVAL_SEEDS, SEEDS
from analysis.mappo_n1_contract import (
    CONDITIONS,
    EVAL_EPISODES,
    EVAL_WARMUP_EPISODES,
    development_eval_root,
    n1_cell,
    n1_eval_dir,
    validate_n1_cell,
)
from analysis.summarize_mappo_e5_sharing_critic import _evaluation_cell
from analysis.summarize_mappo_shared_actor import _metric_block


METRICS = (
    "mean_aoi_ms",
    "worst_agent_aoi_ms",
    "aoi_gt50_fraction",
    "aoi_at_cap_fraction",
    "mean_binary_cam",
    "worst_agent_binary_cam",
    "mean_payload_completion",
    "worst_agent_payload_completion",
    "mean_reward_global",
    "mean_reward_task1",
    "mean_reward_task2",
    "mean_reward_combined",
    "mean_power_mw",
)


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def _metrics(arrays: Mapping[str, np.ndarray], complete: Mapping) -> dict:
    return _metric_block(
        aoi=arrays["aoi_ms"],
        success=arrays["success"],
        remaining=arrays["remaining_demand"],
        rb=arrays["rb"],
        mode=arrays["mode"],
        power_dbm=arrays["executed_power_dbm"],
        reward_global=arrays["reward_global"],
        reward_task1=arrays["reward_task1"],
        reward_task2=arrays["reward_task2"],
        cam_bits=float(complete["cam_bits"]),
        global_actor_weight=float(complete["global_actor_weight"]),
    )


def _world_agent_rows(
    arrays: Mapping[str, np.ndarray],
    complete: Mapping,
    *,
    structure: str,
    variant: str,
    training_seed: int,
    run_name: str,
    evaluation_role: str,
    data_role: str,
    source_root: Path,
) -> list[dict]:
    worlds = [int(value) for value in complete["eval_seeds"]]
    if arrays["aoi_ms"].shape[0] != len(worlds):
        raise ValueError(f"{run_name}: world count does not match metrics")
    cam_bits = float(complete["cam_bits"])
    weight = float(complete["global_actor_weight"])
    cap = float(complete.get("aoi_cap_ms", 100.0))
    rows = []
    for world_index, world in enumerate(worlds):
        for agent in range(5):
            aoi = np.asarray(arrays["aoi_ms"][world_index, :, :, agent], dtype=np.float64)
            endpoint_cam = np.asarray(
                arrays["success"][world_index, :, -1, agent], dtype=np.float64
            )
            if not np.all(np.isclose(endpoint_cam, 0.0) | np.isclose(endpoint_cam, 1.0)):
                raise ValueError(f"{run_name}: endpoint CAM is not binary")
            remaining = np.asarray(
                arrays["remaining_demand"][world_index, :, -1, agent], dtype=np.float64
            )
            payload = np.clip(1.0 - remaining / cam_bits, 0.0, 1.0)
            power_mw = np.power(
                10.0,
                np.asarray(
                    arrays["executed_power_dbm"][world_index, :, :, agent],
                    dtype=np.float64,
                )
                / 10.0,
            )
            reward_global = np.asarray(
                arrays["reward_global"][world_index], dtype=np.float64
            )
            reward_task1 = np.asarray(
                arrays["reward_task1"][world_index, :, :, agent], dtype=np.float64
            )
            reward_task2 = np.asarray(
                arrays["reward_task2"][world_index, :, :, agent], dtype=np.float64
            )
            reward_combined = reward_task1 + reward_task2 + weight * reward_global
            rows.append(
                {
                    "experiment": "N1",
                    "evaluation_role": evaluation_role,
                    "data_role": data_role,
                    "structure": structure,
                    "variant": variant,
                    "training_seed": training_seed,
                    "eval_world": world,
                    "agent": agent,
                    "run_name": run_name,
                    "source_root": str(source_root),
                    "aoi_sum_ms": float(aoi.sum()),
                    "aoi_sample_count": int(aoi.size),
                    "mean_aoi_ms": float(aoi.mean()),
                    "aoi_gt50_count": int((aoi > 50.0).sum()),
                    "aoi_at_cap_count": int(
                        np.isclose(aoi, cap, rtol=0.0, atol=1e-6).sum()
                    ),
                    "aoi_gt50_fraction": float((aoi > 50.0).mean()),
                    "aoi_at_cap_fraction": float(
                        np.isclose(aoi, cap, rtol=0.0, atol=1e-6).mean()
                    ),
                    "binary_cam_success_count": int(np.rint(endpoint_cam.sum())),
                    "binary_cam_episode_count": int(endpoint_cam.size),
                    "binary_cam": float(endpoint_cam.mean()),
                    "payload_sum": float(payload.sum()),
                    "payload_episode_count": int(payload.size),
                    "payload_completion": float(payload.mean()),
                    "power_mw_sum": float(power_mw.sum()),
                    "power_sample_count": int(power_mw.size),
                    "mean_power_mw": float(power_mw.mean()),
                    "mean_reward_global": float(reward_global.mean()),
                    "mean_reward_task1": float(reward_task1.mean()),
                    "mean_reward_task2": float(reward_task2.mean()),
                    "mean_reward_combined": float(reward_combined.mean()),
                }
            )
    return rows


def _cell_row(
    *,
    cell_id: int,
    evaluation_role: str,
    data_role: str,
    source_root: Path,
    complete: Mapping,
    arrays: Mapping[str, np.ndarray],
    training_commit: str | None = None,
) -> dict:
    cell = n1_cell(cell_id)
    return {
        "experiment": "N1",
        "evaluation_role": evaluation_role,
        "data_role": data_role,
        "cell_id": cell_id,
        "structure": cell.structure,
        "variant": cell.variant,
        "training_seed": cell.training_seed,
        "eval_worlds": ",".join(str(value) for value in complete["eval_seeds"]),
        "eval_world_count": len(complete["eval_seeds"]),
        "run_name": cell.run_name,
        "source_root": str(source_root),
        "training_commit": training_commit,
        "evaluation_commit": complete.get("reproduction_git_commit"),
        **_metrics(arrays, complete),
    }


def pair_revisit_and_development(
    revisit_rows: Sequence[Mapping], development_rows: Sequence[Mapping]
) -> list[dict]:
    def index(rows: Sequence[Mapping], role: str) -> dict:
        result = {}
        for row in rows:
            key = (row["structure"], row["variant"], int(row["training_seed"]))
            if key in result:
                raise ValueError(f"duplicate {role} cell: {key}")
            result[key] = row
        if len(result) != 24:
            raise ValueError(f"expected 24 {role} cells, got {len(result)}")
        return result

    revisit = index(revisit_rows, "revisit")
    development = index(development_rows, "development")
    if set(revisit) != set(development):
        raise ValueError("revisit and development cell identities differ")
    paired = []
    for structure, variant in CONDITIONS:
        for seed in SEEDS:
            key = (structure, variant, seed)
            left, right = revisit[key], development[key]
            row = {
                "structure": structure,
                "variant": variant,
                "training_seed": seed,
                "run_name": left["run_name"],
                "revisit_world": seed,
                "development_worlds": ",".join(str(value) for value in EVAL_SEEDS),
                "gap_definition": "development_worlds_minus_training_world_revisit",
            }
            for metric in METRICS:
                row[f"revisit_{metric}"] = float(left[metric])
                row[f"development_{metric}"] = float(right[metric])
                row[f"gap_{metric}"] = float(right[metric]) - float(left[metric])
            paired.append(row)
    return paired


def sharing_gap_rows(paired_rows: Sequence[Mapping]) -> list[dict]:
    indexed = {
        (row["structure"], row["variant"], int(row["training_seed"])): row
        for row in paired_rows
    }
    if len(indexed) != 24 or len(paired_rows) != 24:
        raise ValueError("expected 24 unique paired condition/seed rows")
    result = []
    for variant in ("combined", "tdec"):
        for seed in SEEDS:
            independent = indexed[("independent", variant, seed)]
            shared = indexed[("shared", variant, seed)]
            row = {
                "variant": variant,
                "training_seed": seed,
                "gap_definition": "G=development_minus_revisit",
                "d_definition": "D=G_shared_minus_G_independent",
            }
            for metric in METRICS:
                independent_gap = float(independent[f"gap_{metric}"])
                shared_gap = float(shared[f"gap_{metric}"])
                row[f"independent_G_{metric}"] = independent_gap
                row[f"shared_G_{metric}"] = shared_gap
                row[f"D_{metric}"] = shared_gap - independent_gap
            result.append(row)
    return result


def _condition_summary(rows: Sequence[Mapping]) -> list[dict]:
    result = []
    for role in ("training_world_revisit", "development_worlds"):
        for structure, variant in CONDITIONS:
            selected = [
                row
                for row in rows
                if row["evaluation_role"] == role
                and row["structure"] == structure
                and row["variant"] == variant
            ]
            if len(selected) != 6:
                raise ValueError(f"expected six cells for {role}/{structure}/{variant}")
            item = {
                "evaluation_role": role,
                "structure": structure,
                "variant": variant,
                "training_seed_count": 6,
            }
            for metric in METRICS:
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                item[metric] = float(values.mean())
                item[f"{metric}_sd_across_training_seeds"] = float(values.std(ddof=1))
            result.append(item)
    return result


def _gap_summary(rows: Sequence[Mapping]) -> list[dict]:
    result = []
    for variant in ("combined", "tdec"):
        selected = [row for row in rows if row["variant"] == variant]
        if len(selected) != 6:
            raise ValueError(f"expected six sharing-gap rows for {variant}")
        item = {
            "variant": variant,
            "training_seed_count": 6,
            "gap_definition": "G=development_minus_revisit",
            "d_definition": "D=G_shared_minus_G_independent",
            "aoi_interpretation": "D<0 means the shared policy has a smaller cross-world AoI increment",
        }
        for metric in METRICS:
            values = np.asarray([row[f"D_{metric}"] for row in selected], dtype=np.float64)
            item[f"mean_D_{metric}"] = float(values.mean())
            item[f"sd_D_{metric}"] = float(values.std(ddof=1))
        result.append(item)
    return result


def summarize(study_root: Path, n1_root: Path) -> dict:
    study_root = Path(study_root).expanduser().resolve()
    n1_root = Path(n1_root).expanduser().resolve()
    e5_root = study_root / "E5-sharing-critic" / "P5_N4_gap25"
    shared_tdec_root = (
        study_root / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25"
    )
    revisit_cells, development_cells = [], []
    revisit_streams, development_streams = [], []
    for cell_id in range(24):
        cell = n1_cell(cell_id)
        marker = validate_n1_cell(study_root, n1_root, cell_id)
        revisit_dir = n1_eval_dir(n1_root, cell)
        revisit_complete = _read_json(revisit_dir / "EVAL_COMPLETE.json")
        revisit_arrays = _load_arrays(revisit_dir / "metrics.npz")
        revisit_cells.append(
            _cell_row(
                cell_id=cell_id,
                evaluation_role="training_world_revisit",
                data_role="new",
                source_root=n1_root,
                complete=revisit_complete,
                arrays=revisit_arrays,
                training_commit=(
                    marker.get("training_source_commit") or marker.get("training_commit")
                ),
            )
        )
        revisit_streams.extend(
            _world_agent_rows(
                revisit_arrays,
                revisit_complete,
                structure=cell.structure,
                variant=cell.variant,
                training_seed=cell.training_seed,
                run_name=cell.run_name,
                evaluation_role="training_world_revisit",
                data_role="new",
                source_root=Path(marker["eval_dir"]),
            )
        )

        _evaluation_cell(
            shared_tdec_root,
            e5_root,
            cell.structure,
            cell.variant,
            cell.training_seed,
        )
        development_root = development_eval_root(study_root, cell)
        development_dir = (
            development_root
            / "evaluations"
            / cell.run_name
            / feasibility_eval_id(
                "baseline", EVAL_SEEDS, EVAL_EPISODES, EVAL_WARMUP_EPISODES
            )
        )
        development_complete = _read_json(development_dir / "EVAL_COMPLETE.json")
        development_provenance = _read_json(development_dir / "provenance.json")
        provenance_expected = {
            "algorithm": "mappo",
            "mappo_variant": cell.variant,
            "actor_sharing": cell.structure == "shared",
            "training_seed": cell.training_seed,
            "eval_seeds": list(EVAL_SEEDS),
            "eval_episodes": EVAL_EPISODES,
            "eval_warmup_episodes": EVAL_WARMUP_EPISODES,
            "mappo_eval_mode": "stochastic",
            "intervention_arm": "baseline",
        }
        for key, expected_value in provenance_expected.items():
            if development_provenance.get(key) != expected_value:
                raise ValueError(
                    f"{development_dir}/provenance.json: {key}="
                    f"{development_provenance.get(key)!r}, expected {expected_value!r}"
                )
        development_arrays = _load_arrays(development_dir / "metrics.npz")
        development_cells.append(
            _cell_row(
                cell_id=cell_id,
                evaluation_role="development_worlds",
                data_role="reused",
                source_root=development_root,
                complete=development_complete,
                arrays=development_arrays,
                training_commit=development_provenance.get("training_source_commit"),
            )
        )
        development_streams.extend(
            _world_agent_rows(
                development_arrays,
                development_complete,
                structure=cell.structure,
                variant=cell.variant,
                training_seed=cell.training_seed,
                run_name=cell.run_name,
                evaluation_role="development_worlds",
                data_role="reused",
                source_root=development_dir,
            )
        )

    paired = pair_revisit_and_development(revisit_cells, development_cells)
    gap_rows = sharing_gap_rows(paired)
    all_cells = revisit_cells + development_cells
    all_streams = revisit_streams + development_streams
    condition_summary = _condition_summary(all_cells)
    gap_summary = _gap_summary(gap_rows)
    counts = {
        "new_training_cells": 0,
        "new_revisit_evaluation_cells": len(revisit_cells),
        "reused_development_evaluation_cells": len(development_cells),
        "total_policy_evaluation_cells": len(all_cells),
        "condition_count": len(CONDITIONS),
        "training_seed_count_per_condition": len(SEEDS),
        "revisit_world_agent_rows": len(revisit_streams),
        "development_world_agent_rows": len(development_streams),
        "combined_world_agent_rows": len(all_streams),
        "paired_condition_seed_rows": len(paired),
    }
    expected = {
        "new_training_cells": 0,
        "new_revisit_evaluation_cells": 24,
        "reused_development_evaluation_cells": 24,
        "total_policy_evaluation_cells": 48,
        "condition_count": 4,
        "training_seed_count_per_condition": 6,
        "revisit_world_agent_rows": 120,
        "development_world_agent_rows": 720,
        "combined_world_agent_rows": 840,
        "paired_condition_seed_rows": 24,
    }
    if counts != expected:
        raise ValueError(f"N1 acceptance counts mismatch: {counts}, expected {expected}")
    return {
        "status": "PASS",
        "experiment": "N1",
        "contract": {
            "scenario": "p05_n04_g25",
            "training_seeds": list(SEEDS),
            "revisit_world_rule": "eval_world=training_seed",
            "development_worlds": list(EVAL_SEEDS),
            "evaluation_mode": "stochastic",
            "eval_protocol": "sequential_warm",
            "eval_warmup_episodes": EVAL_WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": 100,
            "development_worlds_are_reused_not_final_test": True,
            "revisit_is_not_training_last100_state_replay": True,
        },
        "counts": counts,
        "condition_summary": condition_summary,
        "paired_per_seed": paired,
        "sharing_gap_per_seed": gap_rows,
        "sharing_gap_summary": gap_summary,
        "revisit_cells": revisit_cells,
        "development_cells": development_cells,
        "revisit_world_agent": revisit_streams,
        "development_world_agent": development_streams,
        "all_world_agent": all_streams,
    }


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(n1_root: Path, report: Mapping) -> Path:
    output = Path(n1_root).expanduser().resolve() / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    for filename, key in (
        ("n1_condition_summary.csv", "condition_summary"),
        ("n1_revisit_development_per_seed.csv", "paired_per_seed"),
        ("n1_sharing_gap_per_seed.csv", "sharing_gap_per_seed"),
        ("n1_sharing_gap_summary.csv", "sharing_gap_summary"),
        ("n1_revisit_world_agent.csv", "revisit_world_agent"),
        ("n1_e5_development_world_agent.csv", "development_world_agent"),
        ("n1_all_world_agent.csv", "all_world_agent"),
    ):
        _write_csv(output / filename, report[key])
    compact = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "revisit_world_agent",
            "development_world_agent",
            "all_world_agent",
        }
    }
    (output / "n1_summary.json").write_text(
        json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# N1 training-world revisit",
        "",
        "Frozen final policies revisit only their own training-initialization world. "
        "The comparison reuses E5 development worlds 213--218; neither side is a final test.",
        "",
        "A revisit is not a replay of the training last100 state trajectory, and matching the "
        "world initialization does not imply an identical closed-loop observation distribution.",
        "",
        "| role | actor | critic | AoI mean±SD | worst AoI | CAM mean/worst | payload mean/worst | power mW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["condition_summary"]:
        lines.append(
            f"| {row['evaluation_role']} | {row['structure']} | {row['variant']} | "
            f"{row['mean_aoi_ms']:.3f}±{row['mean_aoi_ms_sd_across_training_seeds']:.3f} | "
            f"{row['worst_agent_aoi_ms']:.3f} | "
            f"{row['mean_binary_cam']:.4f}/{row['worst_agent_binary_cam']:.4f} | "
            f"{row['mean_payload_completion']:.4f}/{row['worst_agent_payload_completion']:.4f} | "
            f"{row['mean_power_mw']:.3f} |"
        )
    lines += [
        "",
        "`G = development worlds - training-world revisit`; "
        "`D = G_shared - G_independent`. For AoI only, D<0 means sharing has a "
        "smaller cross-world increment. Other metrics retain their own directions.",
        "",
        "| critic | mean D AoI±SD | mean D worst AoI | mean D CAM | mean D payload | mean D power mW |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["sharing_gap_summary"]:
        lines.append(
            f"| {row['variant']} | {row['mean_D_mean_aoi_ms']:+.3f}±{row['sd_D_mean_aoi_ms']:.3f} | "
            f"{row['mean_D_worst_agent_aoi_ms']:+.3f} | "
            f"{row['mean_D_mean_binary_cam']:+.4f} | "
            f"{row['mean_D_mean_payload_completion']:+.4f} | "
            f"{row['mean_D_mean_power_mw']:+.3f} |"
        )
    lines += [
        "",
        "Technical PASS means source, protocol, completeness, frozen-policy checks, counts, "
        "and metric calculations passed. It does not require the scientific hypothesis to hold.",
        "",
    ]
    (output / "n1_report.md").write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--n1-root", required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.study_root, args.n1_root)
    output = write_report(args.n1_root, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "experiment": report["experiment"],
                "counts": report["counts"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
