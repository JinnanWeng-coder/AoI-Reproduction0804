"""Postprocess the 24 locked E3 evaluations into C1 service and C2 interval evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analysis import audit_mappo_w1_service_intervals as w1
from analysis.mappo_e3_contract import ARRAY_FIELDS, cell, protocol, result_root, eval_dir, validate_cell, validate_lock, current_commit

CONDITIONS = (("independent", "combined"), ("shared", "combined"),
              ("independent", "tdec"), ("shared", "tdec"))
SEEDS = (8, 9, 10, 11, 12, 13)
C1_METRICS = ("mean_aoi_ms", "worst_agent_aoi_ms", "aoi_gt50_fraction", "aoi_at_cap_fraction",
              "mean_binary_cam", "worst_agent_binary_cam", "mean_payload_completion",
              "worst_agent_payload_completion", "mean_reward_combined", "mean_power_mw")


def _csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def world_agent_rows(arrays: dict, complete: dict, item: dict) -> list[dict]:
    """Exact numerator/denominator per world-agent; CAM only at episode endpoints."""
    cam_bits = float(complete["cam_bits"])
    weight = float(complete["global_actor_weight"])
    if cam_bits <= 0 or not np.isfinite(weight):
        raise ValueError("invalid CAM bits or global actor weight")
    rows = []
    for wi, world in enumerate(protocol()["candidate_worlds"]):
        for agent in range(5):
            aoi = np.asarray(arrays["aoi_ms"][wi, :, :, agent], dtype=np.float64)
            cam = np.asarray(arrays["success"][wi, :, -1, agent], dtype=np.float64)
            remaining = np.asarray(arrays["remaining_demand"][wi, :, -1, agent], dtype=np.float64)
            if not np.all(np.isclose(cam, 0.0) | np.isclose(cam, 1.0)):
                raise ValueError(f"{item['cell_id']}/{world}/{agent}: CAM endpoint is not binary")
            payload = np.clip(1.0 - remaining / cam_bits, 0.0, 1.0)
            power = np.power(10.0, np.asarray(arrays["executed_power_dbm"][wi, :, :, agent],
                                                dtype=np.float64) / 10.0)
            reward = (np.asarray(arrays["reward_task1"][wi, :, :, agent], dtype=np.float64)
                      + np.asarray(arrays["reward_task2"][wi, :, :, agent], dtype=np.float64)
                      + weight * np.asarray(arrays["reward_global"][wi], dtype=np.float64))
            rows.append({"actor_structure": item["actor_structure"],
                         "value_configuration": item["value_configuration"],
                         "training_seed": item["training_seed"], "world": world, "agent": agent,
                         "aoi_sum_ms": float(aoi.sum()), "aoi_slot_count": int(aoi.size),
                         "aoi_gt50_count": int((aoi > 50).sum()),
                         "aoi_at_cap_count": int(np.isclose(aoi, 100, rtol=0, atol=1e-6).sum()),
                         "binary_cam_success_count": int(np.rint(cam).sum()), "binary_cam_episode_count": int(cam.size),
                         "payload_sum": float(payload.sum()), "payload_episode_count": int(payload.size),
                         "reward_combined_sum": float(reward.sum()), "reward_slot_count": int(reward.size),
                         "power_mw_sum": float(power.sum()), "power_slot_count": int(power.size)})
    return rows


def c1_seed_metrics(rows: list[dict], structure: str, variant: str, seed: int) -> dict:
    if len(rows) != 30 or {r["world"] for r in rows} != set(protocol()["candidate_worlds"]):
        raise ValueError("C1 seed requires six worlds x five agents")
    agents = []
    for agent in range(5):
        selected = [row for row in rows if row["agent"] == agent]
        if len(selected) != 6:
            raise ValueError("C1 agent/world coverage mismatch")
        aoi_sum = sum(row["aoi_sum_ms"] for row in selected)
        aoi_n = sum(row["aoi_slot_count"] for row in selected)
        cam_n = sum(row["binary_cam_episode_count"] for row in selected)
        agents.append((aoi_sum / aoi_n, sum(row["binary_cam_success_count"] for row in selected) / cam_n))
        if aoi_n != 60000 or cam_n != 600:
            raise ValueError("C1 AoI/CAM denominator mismatch")
    total_slots = sum(row["aoi_slot_count"] for row in rows)
    total_cam = sum(row["binary_cam_episode_count"] for row in rows)
    if total_slots != 300000 or total_cam != 3000:
        raise ValueError("C1 seed denominators mismatch")
    return {"actor_structure": structure, "value_configuration": variant, "training_seed": seed,
            "world_count": 6, "agent_count": 5, "aoi_slot_count": total_slots,
            "binary_cam_episode_count": total_cam,
            "mean_aoi_ms": sum(row["aoi_sum_ms"] for row in rows) / total_slots,
            "worst_agent_aoi_ms": max(value[0] for value in agents),
            "aoi_gt50_fraction": sum(row["aoi_gt50_count"] for row in rows) / total_slots,
            "aoi_at_cap_fraction": sum(row["aoi_at_cap_count"] for row in rows) / total_slots,
            "mean_binary_cam": sum(row["binary_cam_success_count"] for row in rows) / total_cam,
            "worst_agent_binary_cam": min(value[1] for value in agents),
            "mean_payload_completion": sum(row["payload_sum"] for row in rows) / total_cam,
            "worst_agent_payload_completion": min(
                sum(row["payload_sum"] for row in rows if row["agent"] == agent) / 600 for agent in range(5)),
            "mean_reward_combined": sum(row["reward_combined_sum"] for row in rows) / total_slots,
            "mean_power_mw": sum(row["power_mw_sum"] for row in rows) / total_slots}


def _direction_counts(values: list[float | None]) -> dict:
    defined = [value for value in values if value is not None]
    return {"negative_count": sum(value < 0 for value in defined),
            "zero_count": sum(value == 0 for value in defined),
            "positive_count": sum(value > 0 for value in defined),
            "undefined_count": len(values) - len(defined)}


def _summaries(seed_rows: list[dict], metrics: tuple[str, ...]) -> tuple[list[dict], list[dict], list[dict]]:
    indexed = {(r["actor_structure"], r["value_configuration"], r["training_seed"]): r for r in seed_rows}
    if len(indexed) != 24:
        raise ValueError("expected 24 distinct condition/seed rows")
    conditions, effects, effect_summaries = [], [], []
    for structure, variant in CONDITIONS:
        selected = [indexed[structure, variant, seed] for seed in SEEDS]
        summary = {"actor_structure": structure, "value_configuration": variant, "training_seed_count": 6}
        for metric in metrics:
            summary.update({f"{metric}_{key}": value for key, value in
                            w1._describe([r[metric] for r in selected]).items()})
        conditions.append(summary)
    for variant in ("combined", "tdec"):
        paired = []
        for seed in SEEDS:
            left, right = indexed["independent", variant, seed], indexed["shared", variant, seed]
            effect = {"value_configuration": variant, "training_seed": seed,
                      "delta_definition": "shared_minus_independent"}
            for metric in metrics:
                a, b = left[metric], right[metric]
                effect[f"delta_{metric}"] = b - a if a is not None and b is not None else None
            effects.append(effect)
            paired.append(effect)
        summary = {"value_configuration": variant, "training_seed_count": 6,
                   "delta_definition": "shared_minus_independent"}
        for metric in metrics:
            values = [row[f"delta_{metric}"] for row in paired]
            summary.update({f"delta_{metric}_{key}": value for key, value in w1._describe(values).items()})
            summary.update({f"delta_{metric}_{key}": value for key, value in _direction_counts(values).items()})
        effect_summaries.append(summary)
    return conditions, effects, effect_summaries


def analyze(study_root: Path) -> dict:
    locked = validate_lock(study_root)
    root = result_root(study_root)
    output = root / "analysis"
    if (output / "verification.json").exists():
        raise FileExistsError("E3 analysis verification already exists; do not overwrite")
    c1_rows, flow_rows, freq_rows, inventory, examples = [], [], [], [], []
    frequencies: dict[tuple, Counter[int]] = {}
    for cell_id in range(24):
        marker = validate_cell(study_root, cell_id)
        item = cell(cell_id)
        directory = eval_dir(study_root, item)
        complete = json.loads((directory / "EVAL_COMPLETE.json").read_text(encoding="utf-8"))
        with np.load(directory / "metrics.npz", allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in ARRAY_FIELDS}
        c1_rows.extend(world_agent_rows(arrays, complete, item))
        inventory.append({"cell_id": cell_id, "actor_structure": item["actor_structure"],
                          "value_configuration": item["value_configuration"], "training_seed": item["training_seed"],
                          "training_commit": marker["training_commit"], "evaluation_commit": marker["evaluation_commit"],
                          "evaluation_dir": str(directory), "worlds": ",".join(map(str, marker["worlds"])),
                          "policy_episode": 500, "event_source": "reset_event"})
        for wi, world in enumerate(protocol()["candidate_worlds"]):
            for agent in range(5):
                identity = {"actor_structure": item["actor_structure"],
                            "value_configuration": item["value_configuration"],
                            "training_seed": item["training_seed"], "world": world, "agent": agent}
                row, freq, positions = w1.flow_statistics(arrays["reset_event"][wi, :, :, agent].astype(bool))
                row["one_reset_flow"] = row["reset_count"] == 1
                row["interval_na_reason"] = ("no_reset" if row["reset_count"] == 0 else
                                             "one_reset" if row["reset_count"] == 1 else None)
                flow_rows.append({**identity, "event_source": "reset_event", **row})
                frequencies[(item["actor_structure"], item["value_configuration"],
                             item["training_seed"], world, agent)] = freq
                freq_rows.extend({**identity, "length_L": length, "frequency": count}
                                 for length, count in sorted(freq.items()))
                if len(examples) < 8:
                    examples.append(w1._index_example(identity, positions))
    if len(c1_rows) != 720 or len(flow_rows) != 720:
        raise ValueError("E3 world-agent row count mismatch")
    if sum(r["aoi_slot_count"] for r in c1_rows) != 7200000:
        raise ValueError("E3 agent-slot count mismatch")
    if sum(r["binary_cam_episode_count"] for r in c1_rows) != 72000:
        raise ValueError("E3 agent-episode count mismatch")
    c1_seeds, c2_seeds = [], []
    for structure, variant in CONDITIONS:
        for seed in SEEDS:
            key = structure, variant, seed
            selected = [r for r in c1_rows if (r["actor_structure"], r["value_configuration"], r["training_seed"]) == key]
            c1_seeds.append(c1_seed_metrics(selected, *key))
            flow_selected = [r for r in flow_rows if (r["actor_structure"], r["value_configuration"], r["training_seed"]) == key]
            seed_row, _ = w1.seed_statistics(flow_selected, frequencies, key)
            seed_row["one_reset_flow_count"] = sum(r["one_reset_flow"] for r in flow_selected)
            c2_seeds.append(seed_row)
    c1_conditions, c1_effects, c1_effect_summary = _summaries(c1_seeds, C1_METRICS)
    c2_conditions, c2_effects, c2_effect_summary = _summaries(c2_seeds, w1.METRICS)
    for row in c2_conditions:
        selected = [seed_row for seed_row in c2_seeds if
                    (seed_row["actor_structure"], seed_row["value_configuration"]) ==
                    (row["actor_structure"], row["value_configuration"])]
        row.update({"raw_long_gt100_total_count": sum(item["long_gt100_count"] for item in selected),
                    "eligible_flow_count": sum(item["eligible_flow_count"] for item in selected),
                    "no_reset_flow_count": sum(item["no_reset_flow_count"] for item in selected),
                    "one_reset_flow_count": sum(item["one_reset_flow_count"] for item in selected)})
    global_support = sorted({0, 5, 50, 100} | {length for freq in frequencies.values() for length in freq})
    c2_ccdf = []
    for structure, variant in CONDITIONS:
        for seed in SEEDS:
            key = structure, variant, seed
            selected = [r for r in flow_rows if (r["actor_structure"], r["value_configuration"], r["training_seed"]) == key]
            ccdf = w1._ccdf(selected, frequencies, key, global_support)
            for field in ("event_ccdf_P_L_gt_threshold", "flow_equal_ccdf_P_L_gt_threshold"):
                values = [r[field] for r in ccdf if r[field] is not None]
                if any(a + 1e-12 < b for a, b in zip(values, values[1:])):
                    raise ValueError(f"{key}: nonmonotone CCDF")
            c2_ccdf.extend(ccdf)
    c2_ccdf_conditions, c2_ccdf_effects, c2_ccdf_effect_summary = w1._ccdf_summaries(c2_ccdf)
    for row in c2_ccdf_effect_summary:
        matched = [effect for effect in c2_ccdf_effects
                   if effect["value_configuration"] == row["value_configuration"]
                   and effect["threshold_L"] == row["threshold_L"]]
        for field in ("event_ccdf_P_L_gt_threshold", "flow_equal_ccdf_P_L_gt_threshold"):
            row.update({f"delta_{field}_{key}": value for key, value in
                        _direction_counts([effect[f"delta_{field}"] for effect in matched]).items()})
    for row in flow_rows:
        key = (row["actor_structure"], row["value_configuration"], row["training_seed"], row["world"], row["agent"])
        freq = frequencies[key]
        for field, actual in (("complete_interval_count", sum(freq.values())),
                              ("sum_L", sum(n * count for n, count in freq.items())),
                              ("sum_L2", sum(n*n * count for n, count in freq.items()))):
            if row[field] != actual:
                raise ValueError(f"{key}: frequency reconstruction {field}")
        if row["complete_interval_count"] != max(row["reset_count"] - 1, 0):
            raise ValueError(f"{key}: reset/interval mismatch")
        if row["no_reset_flow"]:
            if row["no_reset_window_slots"] != 10000:
                raise ValueError(f"{key}: no-reset window mismatch")
        elif row["left_boundary_visible_wait_slots"] + row["sum_L"] + row["right_boundary_visible_wait_slots"] != 9999:
            raise ValueError(f"{key}: boundary mismatch")
    output.mkdir(parents=True, exist_ok=True)
    tables = {"source_inventory.csv": inventory, "c1_world_agent.csv": c1_rows,
              "c1_seed_statistics.csv": c1_seeds, "c1_condition_summary.csv": c1_conditions,
              "c1_sharing_effect_per_seed.csv": c1_effects, "c1_sharing_effect_summary.csv": c1_effect_summary,
              "c2_flow_statistics.csv": flow_rows, "c2_interval_frequency.csv": freq_rows,
              "c2_seed_statistics.csv": c2_seeds, "c2_seed_ccdf.csv": c2_ccdf,
              "c2_condition_summary.csv": c2_conditions, "c2_sharing_effect_per_seed.csv": c2_effects,
              "c2_sharing_effect_summary.csv": c2_effect_summary,
              "c2_condition_ccdf_summary.csv": c2_ccdf_conditions,
              "c2_sharing_ccdf_effect_per_seed.csv": c2_ccdf_effects,
              "c2_sharing_ccdf_effect_summary.csv": c2_ccdf_effect_summary}
    for name, rows in tables.items():
        columns = (["actor_structure", "value_configuration", "training_seed", "world", "agent",
                    "length_L", "frequency"] if name == "c2_interval_frequency.csv" else None)
        _csv(output / name, rows, columns)
    (output / "index_examples.json").write_text(json.dumps(examples, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "fields_and_boundaries.md").write_text(FIELD_NOTES, encoding="utf-8")
    report = ["# E3 预锁定新世界确认：技术报告", "",
              f"锁定世界：{protocol()['candidate_worlds']}；24个既有final-500策略、144个策略×世界运行。",
              "技术验收不以共享效应方向为条件；科学结论由分析会话复核。", "",
              f"完整间隔有效流：{sum(not r['no_complete_interval_flow'] for r in flow_rows)}/720；"
              f"零复位流：{sum(r['no_reset_flow'] for r in flow_rows)}；"
              f"单复位流：{sum(r['one_reset_flow'] for r in flow_rows)}。",
              "C1最差车队按同agent先跨六世界汇总，再取max AoI/min CAM；C2不跨世界连接。", "",
              "| actor | value | seed均值：最差AoI ms | seed均值：最差CAM | seed均值：L>100/万槽 | 完整间隔有效流 |",
              "|---|---|---:|---:|---:|---:|"]
    for c1row, c2row in zip(c1_conditions, c2_conditions):
        structure, variant = c1row["actor_structure"], c1row["value_configuration"]
        eligible = sum(row["eligible_flow_count"] for row in c2_seeds
                       if row["actor_structure"] == structure and row["value_configuration"] == variant)
        report.append(f"| {structure} | {variant} | {c1row['worst_agent_aoi_ms_mean']} | "
                      f"{c1row['worst_agent_binary_cam_mean']} | {c2row['long_gt100_per_10000_slots_mean']} | "
                      f"{eligible}/180 |")
    report.append("")
    (output / "report_cn.md").write_text("\n".join(report), encoding="utf-8")
    verification = {"status": "PASS", "scope": "technical_only", "implementation_commit": locked["implementation_commit"],
                    "validation_commit": current_commit(),
                    "evaluation_commits": sorted({row["evaluation_commit"] for row in inventory}),
                    "locked_at_utc": locked["locked_at_utc"], "worlds": protocol()["candidate_worlds"],
                    "evaluation_cells": 24, "policy_world_runs": 144, "world_agent_flows": 720,
                    "scored_agent_slots": 7200000, "cam_agent_episode_events": 72000,
                    "condition_seed_rows": 24, "eligible_interval_flows": sum(not r["no_complete_interval_flow"] for r in flow_rows),
                    "no_reset_flows": sum(r["no_reset_flow"] for r in flow_rows),
                    "one_reset_flows": sum(r["one_reset_flow"] for r in flow_rows),
                    "ccdf_threshold_count": len(global_support), "output_files": list(tables),
                    "checks": ["24_frozen_source_cells", "locked_protocol", "source_commit_separation",
                               "scored_shapes", "endpoint_cam_denominators", "worst_agent_reduction",
                               "frequency_moments", "boundary_identity", "ccdf_monotonicity"]}
    (output / "verification.json").write_text(json.dumps(verification, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return verification


FIELD_NOTES = """# E3 字段与统计合同

- 新世界 303–308 为修订后预先固定候选，排除源配置预留的301/302；只有 `lock.json` 完成历史使用核查后才生效。训练独立重复仍仅 seeds 8–13（n=6）。combined/TDec 是两种价值学习配置。
- C1 `c1_world_agent.csv` 保存逐世界逐agent精确 sum/count。AoI为100×100槽；binary CAM与payload为每个scored episode末的100个agent-episode事件。每seed同agent先合并六世界，再取最差AoI最大值和CAM最小值。功率先逐槽从dBm转线性mW。
- C2复用W1完整间隔定义；`reset_event`优先且与逐槽post-step AoI≈1逐元素核对。episode间连续、world间不连接、不滤V2V。连续复位 L=1；AoI cap不截断L。首次/末次复位外仅边界可见等待，不当完整间隔。无复位流只计一个M=10000观察窗；单复位无完整间隔，NA不填0。
- 严格主阈值 L>100。报告原始长间隔计数、每万agent-slots频度、完整间隔占比以及流等权超阈值比例。完整间隔频数与共同CCDF阈值网格（0、5、50、100及全部观测L）可精确重建。事件加权按完整间隔合并；流等权先逐有完整间隔流算再等权，覆盖分母始终显式报告。两者均仅为有限窗内完整间隔分布。
- 所有条件摘要先按training seed汇总，再跨六seed等权，sample SD(ddof=1)、描述性95% t区间df=5、t*=2.570581835636314。同seed的shared−independent差单列负/零/正个数。world、agent、episode、slot、interval不作为新增训练重复；未定义seed统计以空值与有效数报告。
- 技术PASS只检验来源、协议、冻结和计算；不要求共享效应方向，不作显著性星号、因果分解或CAM机制推断。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.study_root), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
