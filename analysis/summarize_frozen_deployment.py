"""D1 scene-first and legacy deployment levels and distinct variability axes."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import deployment_contract as dc
from analysis.evaluate_frozen_deployment import (SUM_METRICS, accept, csv_write, csv_read, unit_dir, reused_pilot_unit)
from analysis.mappo_b1_accounting import T95

GROUP = ("condition", "model_id", "target_scenario", "training_seed")
PRIMARY = ("worst_aoi_ms", "worst_cam_success", "mean_aoi_ms", "mean_cam_success",
           "mean_power_mw", "mean_reward_combined", "worst_aoi_gt50", "mean_aoi_gt50",
           "mean_v2i_attempt_rate", "mean_v2i_reset_rate")
PRIMARY = tuple(dict.fromkeys((*PRIMARY, "worst_payload_completion", *(f"mean_{m}" for m in SUM_METRICS))))


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def number(x):
    return None if x in (None, "") else float(x)


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def sd(values):
    values = list(values)
    return float(np.std(values, ddof=1)) if len(values) > 1 and all(v is not None for v in values) else None


def pooled(rows):
    fields = {field for pair in SUM_METRICS.values() for field in pair}
    return {field: (sum(number(r[field]) for r in rows) if all(number(r.get(field)) is not None for r in rows) else None)
            for field in fields}


def ratios(row):
    return {metric: (number(row[n]) / number(row[d]) if number(row.get(n)) is not None and
                    number(row.get(d)) not in (None, 0) else None)
            for metric, (n, d) in SUM_METRICS.items()}


def endpoints(platoons):
    values = [ratios(r) for r in platoons]
    result = {f"mean_{metric}": mean(v[metric] for v in values) for metric in SUM_METRICS}
    for label, metric, maximum in (("worst_aoi_ms", "aoi_ms", True), ("worst_cam_success", "cam_success", False),
                                   ("worst_aoi_gt50", "aoi_gt50", True), ("worst_payload_completion", "payload_completion", False)):
        data = [v[metric] for v in values]
        result[label] = (max(data) if maximum else min(data)) if all(x is not None for x in data) else None
        result[label + "_platoons"] = ",".join(str(platoons[i]["platoon"]) for i, x in enumerate(data) if x == result[label]) if result[label] is not None else None
    return result


def aggregate(rows):
    """Pool repeat sums/counts per world/platoon before selecting own extrema."""
    worlds, pooled_platoons, seeds, legacy, repeat_rows = [], [], [], [], []
    for key, seed_rows in grouped(rows, GROUP).items():
        ident = dict(zip(GROUP, key))
        world_items = []
        for (world,), world_rows in grouped(seed_rows, ("world",)).items():
            platoons = []
            for (platoon,), platoon_rows in grouped(world_rows, ("platoon",)).items():
                dc.require(len({r["repeat"] for r in platoon_rows}) == len(platoon_rows), "duplicate world/repeat/platoon")
                p = {**ident, "world": world, "platoon": platoon, "repeat_count": len(platoon_rows), **pooled(platoon_rows)}
                platoons.append(p)
                pooled_platoons.append(p)
            dc.require({int(p["platoon"]) for p in platoons} == set(range(5)), "world missing platoons")
            ep = endpoints(platoons)
            item = {**ident, "world": world, "repeat_count": platoons[0]["repeat_count"], **ep}
            by_repeat = []
            for (repeat,), rep in grouped(world_rows, ("repeat",)).items():
                dc.require(len(rep) == 5, "repeat missing platoons")
                data = {**ident, "world": world, "repeat": repeat, **endpoints(rep)}
                repeat_rows.append(data)
                by_repeat.append(data)
            for metric in PRIMARY:
                item[metric + "_repeat_sd_reselected"] = sd(r[metric] for r in by_repeat)
            # Also measure repeats at the fixed scene-first weakest platoon.
            # Ties are kept; their mean repeat SD is a descriptive convention.
            for label, metric in (("worst_aoi_ms", "aoi_ms"), ("worst_cam_success", "cam_success")):
                ids = {int(p) for p in ep[label + "_platoons"].split(",")}
                dispersions = [sd(ratios(r)[metric] for r in world_rows if int(r["platoon"]) == p) for p in ids]
                item[label + "_repeat_sd_fixed_platoon"] = mean(dispersions)
            worlds.append(item)
            world_items.append(item)
        seed = {**ident, "world_count": len(world_items)}
        for metric in PRIMARY:
            seed["scene_first_" + metric] = mean(r[metric] for r in world_items)
            seed[metric + "_world_sd"] = sd(r[metric] for r in world_items)
            seed[metric + "_repeat_sd_mean"] = mean(r[metric + "_repeat_sd_reselected"] for r in world_items)
        for label in ("worst_aoi_ms", "worst_cam_success"):
            seed[label + "_repeat_sd_fixed_platoon_mean"] = mean(r[label + "_repeat_sd_fixed_platoon"] for r in world_items)
        seeds.append(seed)
        agent_pool = [{**ident, "platoon": p, **pooled(rs)} for (p,), rs in grouped(seed_rows, ("platoon",)).items()]
        legacy_ep = endpoints(agent_pool)
        legacy.append({**ident, "world_count": len(world_items), **legacy_ep,
                       "legacy_max_agent_mean_aoi_ms": legacy_ep["worst_aoi_ms"],
                       "legacy_min_agent_binary_cam_rate": legacy_ep["worst_cam_success"],
                       "worst_agent_aoi_ms": legacy_ep["worst_aoi_ms"],
                       "worst_agent_binary_cam": legacy_ep["worst_cam_success"],
                       "definition": "legacy: pool worlds and repeats per platoon, then own max AoI/min CAM"})
    conditions = []
    for key, values in grouped(seeds, GROUP[:-1]).items():
        ident = dict(zip(GROUP[:-1], key))
        dc.require(len({v["training_seed"] for v in values}) == len(values), "duplicate training seed")
        for field in sorted(k for k in values[0] if k not in GROUP and k != "world_count"):
            data = [r[field] for r in values]
            center, spread, n = mean(data), sd(data), len(values)
            half = T95 * spread / math.sqrt(6) if n == 6 and spread is not None else None
            conditions.append({**ident, "metric": field, "n_training_seeds": n,
                               "defined_training_seeds": sum(x is not None for x in data), "mean": center,
                               "training_seed_sd": spread, "descriptive_t95_lower": center - half if half is not None else None,
                               "descriptive_t95_upper": center + half if half is not None else None,
                               "df": 5 if n == 6 else None, "claim_support": "NOT_ASSESSED"})
    legacy_conditions = []
    for key, values in grouped(legacy, GROUP[:-1]).items():
        for metric in PRIMARY:
            data = [r[metric] for r in values]
            center, spread, n = mean(data), sd(data), len(values)
            half = T95 * spread / math.sqrt(6) if n == 6 and spread is not None else None
            legacy_conditions.append({**dict(zip(GROUP[:-1], key)), "metric": metric, "mean": center,
                                      "training_seed_sd": spread, "n_training_seeds": n,
                                      "defined_training_seeds": sum(x is not None for x in data),
                                      "descriptive_t95_lower": center - half if half is not None else None,
                                      "descriptive_t95_upper": center + half if half is not None else None,
                                      "df": 5 if n == 6 else None, "claim_support": "NOT_ASSESSED"})
    return {"world_repeat_platoon": rows, "world_platoon_repeat_pooled": pooled_platoons,
            "world_endpoints": worlds, "repeat_endpoints": repeat_rows, "seed_endpoints": seeds,
            "legacy_seed_endpoints": legacy, "condition_summary": conditions, "legacy_condition_summary": legacy_conditions}


def evidence_tables(pack):
    with zipfile.ZipFile(pack) as archive:
        def get(suffix, compressed=False):
            names = [n for n in archive.namelist() if n.endswith("/" + suffix)]
            dc.require(len(names) == 1, f"evidence table ambiguous/missing: {suffix}")
            raw = archive.read(names[0])
            if compressed:
                raw = gzip.decompress(raw)
            return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        return get("tables/eval_world_agent.csv.gz", True), get("tables/legacy_seed_endpoints.csv"), get("tables/eval_world.csv"), get("inventory/evaluations.csv")


def evidence_regression(pack):
    """Regression on compact evidence sums. Does not load any trajectory NPZ."""
    agents, legacy, worlds, _ = evidence_tables(pack)
    by_eval = grouped(agents, ("eval_id",))
    comparisons, errors = 0, []
    for endpoint in legacy:
        rs = by_eval[(endpoint["eval_id"],)]
        per_agent = [(sum(float(r["aoi_sum_ms"]) for r in group) / sum(int(r["aoi_slot_count"]) for r in group),
                      sum(float(r["binary_cam_success_count"]) for r in group) / sum(int(r["binary_cam_episode_count"]) for r in group))
                     for group in grouped(rs, ("agent",)).values()]
        for actual, field, tol in ((max(x[0] for x in per_agent), "legacy_max_agent_mean_aoi_ms", 1e-10),
                                   (min(x[1] for x in per_agent), "legacy_min_agent_binary_cam_rate", 1e-12)):
            comparisons += 1
            if abs(actual - float(endpoint[field])) > tol:
                errors.append({"eval_id": endpoint["eval_id"], "field": field, "actual": actual, "expected": endpoint[field]})
    lookup = grouped(agents, ("eval_id", "world"))
    for endpoint in worlds:
        rs = lookup[(endpoint["eval_id"], endpoint["world"])]
        for actual, field, tol in ((max(float(r["aoi_sum_ms"]) / int(r["aoi_slot_count"]) for r in rs), "world_max_agent_mean_aoi_ms", 1e-10),
                                   (min(float(r["binary_cam_success_count"]) / int(r["binary_cam_episode_count"]) for r in rs), "world_min_agent_binary_cam_rate", 1e-12)):
            comparisons += 1
            if abs(actual - float(endpoint[field])) > tol:
                errors.append({"eval_id": endpoint["eval_id"], "field": field})
    return {"status": "PASS" if not errors else "FAIL", "world_agent_rows": len(agents),
            "legacy_endpoints": len(legacy), "world_endpoints": len(worlds), "comparisons": comparisons,
            "errors": errors, "arrays_loaded": False, "tolerance_aoi": 1e-10, "tolerance_cam": 1e-12}


def bridge(rows, sources, pack, diagnostics):
    """repeat-0 sums versus the exact historical source/evaluation mapping."""
    from analysis.e1_contract import mappo_eval_root, maddpg_eval_id
    agents, _, _, evaluations = evidence_tables(pack)
    outputs, index = [], grouped(agents, ("eval_id", "world", "agent"))
    for row in rows:
        if row["target_scenario"] != dc.SCENARIO or row["repeat"] != 0 or row["execution"] == "deterministic" and row["family"] == "mappo":
            continue
        source = next(s for s in sources if (s["model_id"], s["training_seed"]) == (row["model_id"], row["training_seed"]))
        if row["family"] == "mappo":
            root = str(mappo_eval_root(diagnostics, source["structure"], source["variant"]))
            matches = [r for r in evaluations if r["output_path"].startswith(root + "/") and
                       r["eval_dir_name"].endswith("feasibility_reward_arm_baseline") and
                       int(r["training_seed"]) == row["training_seed"] and
                       r["actor_sharing"].lower() == str(source["structure"] == "shared").lower()]
        else:
            matches = [r for r in evaluations if r["stage"] == "E1" and
                       r["algorithm"] == "modified_maddpg_tdec" and int(r["training_seed"]) == row["training_seed"] and
                       r["eval_dir_name"] == maddpg_eval_id(row["noise_std"])]
        old = index.get((matches[0]["eval_id"], str(row["world"]), str(row["platoon"])), []) if len(matches) == 1 else []
        status = "NO_UNIQUE_HISTORICAL_UNIT" if len(old) != 1 else "COMPARED"
        for field, old_field, atol in (("aoi_sum_ms", "aoi_sum_ms", 0.),
                                     ("cam_deadline_success_count", "binary_cam_success_count", 0.),
                                     ("power_mw_sum", "power_mw_sum", 1e-8)):
            before = number(old[0][old_field]) if len(old) == 1 else None
            difference = row[field] - before if before is not None else None
            outputs.append({k: row[k] for k in ("condition", "training_seed", "target_scenario", "world", "repeat", "platoon")} |
                           {"metric": field, "new_sum": row[field], "historical_sum": before, "new_minus_historical": difference,
                            "atol": atol, "status": ("MATCH" if abs(difference) <= atol else "MISMATCH_REVIEW_DEVICE_AND_PROTOCOL") if difference is not None else status,
                            "changes_technical_pass": False})
    return outputs


def historical_gap_rows(pack):
    """Reuse exactly 24 MAPPO gap cells as compact tables, never as new runs."""
    agents, _, _, evaluations = evidence_tables(pack)
    selected = [r for r in evaluations if r["stage"] == "zero-shot-v1" and
                r["target_scenario"] in {"p05_n04_g05", "p05_n04_g35"} and r["algorithm"] == "mappo" and
                r["critic_variant"] == "tdec" and r["action_mode"] == "stochastic" and
                int(r["training_seed"]) in dc.SEEDS]
    keys = [(r["actor_sharing"], r["training_seed"], r["target_scenario"]) for r in selected]
    dc.require(len(selected) == len(set(keys)) == 24, "historical MAPPO gap evaluation coverage")
    by_eval, result = grouped(agents, ("eval_id",)), []
    for evaluation in selected:
        dc.require(evaluation["world_ids"] == "213,214,215,216,217,218" and
                   evaluation["warmup_episodes"] == "5" and evaluation["scored_episodes_recorded"] == "100" and
                   evaluation["eval_protocol"] == "sequential_warm" and evaluation["actual_status"] == "complete",
                   "historical MAPPO gap protocol")
        group = by_eval[(evaluation["eval_id"],)]
        dc.require(len(group) == 30 and {(int(r["world"]), int(r["agent"])) for r in group} ==
                   {(w, p) for w in range(213, 219) for p in range(5)}, "historical gap world/platoon coverage")
        model = "shared_tdec" if evaluation["actor_sharing"] == "true" else "independent_tdec"
        for old in group:
            row = {"condition": model + "_stochastic", "model_id": model,
                   "training_seed": int(evaluation["training_seed"]), "target_scenario": evaluation["target_scenario"],
                   "world": int(old["world"]), "repeat": 0, "platoon": int(old["agent"]), "data_role": "reused",
                   "source_eval_id": evaluation["eval_id"], "observational_extension": True}
            row.update({field: None for pair in SUM_METRICS.values() for field in pair})
            mapping = {"slot_count": "aoi_slot_count", "aoi_sum_ms": "aoi_sum_ms", "aoi_gt50_count": "aoi_gt50_count",
                       "aoi_at_cap_count": "aoi_at_cap_count", "cam_deadline_success_count": "binary_cam_success_count",
                       "cam_episode_count": "binary_cam_episode_count", "power_mw_sum": "power_mw_sum",
                       "power_slot_count": "power_slot_count", "reward_slot_count": "reward_slot_count",
                       "reward_global_sum": "reward_global_sum", "reward_task1_sum": "reward_task1_sum", "reward_task2_sum": "reward_task2_sum",
                       "reward_combined_sum": "reward_combined_recomputed_sum", "v2i_attempt_count": "V2I_attempt_count"}
            for new, before in mapping.items():
                row[new] = number(old[before])
            row["payload_completion_sum"] = number(old["payload_completion_mean"]) * row["cam_episode_count"] if number(old["payload_completion_mean"]) is not None else None
            row["v2i_slot_count"] = row["slot_count"] if row["v2i_attempt_count"] is not None else None
            dc.require(row["slot_count"] == 10000 and row["cam_episode_count"] == 100, "historical gap denominators")
            result.append(row)
    return result


def plots(tables, output):
    """One-panel PNGs: raw seed points + descriptive mean/t95; no winner styling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    summaries, points = tables["condition_summary"], tables["seed_endpoints"]
    labels = {"shared_tdec_stochastic": "Shared TDec\nstochastic", "independent_tdec_stochastic": "Independent TDec\nstochastic",
              "shared_combined_stochastic": "Shared Combined\nstochastic", "algorithm2_noise0": "Algorithm 2\nnoise 0",
              "algorithm2_noise0p3": "Algorithm 2\nnoise 0.3", "shared_tdec_deterministic": "Shared TDec\ndeterministic",
              "independent_tdec_deterministic": "Independent TDec\ndeterministic"}
    for target in sorted({r["target_scenario"] for r in summaries}):
        for field, ylabel in (("scene_first_worst_aoi_ms", "Scene-first worst-platoon AoI (ms)"),
                              ("scene_first_worst_cam_success", "Scene-first worst-platoon CAM deadline success"),
                              ("worst_aoi_ms_world_sd", "Within-seed across-world SD of worst AoI (ms)"),
                              ("worst_aoi_ms_repeat_sd_fixed_platoon_mean", "Within-world repeat SD at fixed weakest platoon (ms)")):
            cells = [r for r in summaries if r["target_scenario"] == target and r["metric"] == field and r["mean"] is not None]
            if not cells:
                continue
            fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
            colors = {"shared_tdec": "#4477AA", "independent_tdec": "#888888", "shared_combined": "#AA3377", "algorithm2": "#228833"}
            for i, row in enumerate(cells):
                raw = [p[field] for p in points if p["target_scenario"] == target and p["condition"] == row["condition"]]
                color = colors[row["model_id"]]
                ax.scatter(i + np.linspace(-.09, .09, len(raw)), raw, color=color, s=18, alpha=.8)
                if row["descriptive_t95_lower"] is not None:
                    ax.errorbar(i, row["mean"], yerr=row["mean"] - row["descriptive_t95_lower"], color=color, capsize=3, fmt="D", markersize=5)
                else:
                    ax.plot(i, row["mean"], "D", color=color, markersize=5)
            ax.set_xticks(range(len(cells)), [labels[r["condition"]] for r in cells], fontsize=8)
            ax.set_xlim(-.5, len(cells) - .5)
            ax.set_ylabel(ylabel)
            ax.set_title(target + "; points = training seeds; diamonds = means")
            fig.savefig(output / f"{target}_{field}.png", dpi=300)
            csv_write(output / f"{target}_{field}_source.csv",
                      [{"condition": p["condition"], "model_id": p["model_id"], "training_seed": p["training_seed"],
                        "target_scenario": target, "metric": field, "value": p[field]}
                       for p in points if p["target_scenario"] == target])
            dc.write(output / f"{target}_{field}_qa.json",
                     {"single_panel_alignment": "NOT_APPLICABLE", "export": "PNG per task contract",
                      "uncertainty": "descriptive t95 df5 only at n=6; otherwise raw seed points and mean",
                      "plot_area_bounds": list(ax.get_position().bounds),
                      "undefined_conditions_retained_in_csv": [r["condition"] for r in summaries if
                          r["target_scenario"] == target and r["metric"] == field and r["mean"] is None],
                      "pdf_collision_audit": "NOT_APPLICABLE_PNG_ONLY"})
            plt.close(fig)


def summarize(batch, phase, attempt="a01", evidence_pack=None, make_plots=True, *, contract=dc, extension=None):
    dc.require(re.fullmatch(r"a[0-9]{2,4}", attempt), "unsafe analysis attempt")
    manifest, sources = contract.load_batch(batch)
    data = contract.plan(batch, phase)
    output = Path(batch) / "analysis" / phase / attempt
    output.mkdir(parents=True, exist_ok=False)
    rows, records, probes = [], [], {}
    try:
        if evidence_pack and manifest.get("input_packs", {}).get("evidence_pack"):
            dc.require(dc.file_identity(evidence_pack) == manifest["input_packs"]["evidence_pack"], "historical evidence pack identity changed")
        for unit in data["units"]:
            source = next(s for s in sources if (s["model_id"], s["training_seed"]) == (unit["model_id"], unit["training_seed"]))
            dc.source_check(source)
            reused = reused_pilot_unit(batch, unit, source, manifest)
            if reused is None:
                parent = unit_dir(batch, unit).parent
                completed = list(parent.glob("a*/DEPLOYMENT_COMPLETE.json"))
                dc.require(len(completed) == 1, f"unit {unit['unit_id']} requires exactly one accepted attempt")
                record, unit_rows, probe = accept(batch, unit, source, completed[0].parent.name, manifest)
            else:
                record, unit_rows, probe = reused
            rows.extend(unit_rows)
            records.append({**unit, "attempt": record["attempt"], "technical_status": "PASS",
                            "native_evaluation_dir": record["native_evaluation_dir"],
                            "source_checkpoint": source["checkpoint"]["path"], "training_commit": source["training_commit"],
                            "implementation_commit": manifest["implementation_commit"],
                            "evaluation_implementation_commit": record.get("implementation_commit", manifest["implementation_commit"]),
                            "data_role": "reused" if record.get("reused_from_batch") else "new",
                            "reused_from_batch": record.get("reused_from_batch"),
                            **{k: record.get(k) for k in ("slurm_job_id", "slurm_array_job_id", "slurm_array_task_id")}})
            key = (unit["target_scenario"], unit["world"])
            if key in probes:
                dc.require(probes[key] == probe, f"exogenous process changed across actions/models/seeds: {key}")
            else:
                probes[key] = probe
        dc.require(len(rows) == data["platoon_row_count"] and
                   sum(r["slot_count"] for r in rows) == len(data["units"]) * 50000 and
                   sum(r["cam_episode_count"] for r in rows) == len(data["units"]) * 500, "scored denominators")
        dc.require(phase != "gap-transfer" or evidence_pack is not None, "gap-transfer summary requires --evidence-pack for historical MAPPO alignment")
        tables = aggregate(rows)
        for name, values in tables.items():
            csv_write(output / f"{name}.csv", values)
        csv_write(output / "completion_records.csv", records)
        csv_write(output / "exogenous_verification.csv", [{"target_scenario": k[0], "world": k[1], "exact_initial_and_final_probes_match": True} for k in probes])
        if evidence_pack:
            regression = evidence_regression(evidence_pack)
            dc.write(output / "evidence_table_regression.json", regression)
            dc.require(regression["status"] == "PASS", "historical table regression failed")
            bridges = bridge(rows, sources, evidence_pack, Path(manifest["diagnostics_root"]))
            if bridges:
                csv_write(output / "historical_repeat0_bridge.csv", bridges)
            if phase == "gap-transfer":
                historical = historical_gap_rows(evidence_pack)
                csv_write(output / "historical_gap_world_repeat_platoon.csv", historical)
                aligned = aggregate([{**r, "data_role": "new"} for r in rows] + historical)
                for name, values in aligned.items():
                    csv_write(output / f"gap_alignment_{name}.csv", values)
                tables_for_plots = aligned
            else:
                tables_for_plots = tables
        else:
            tables_for_plots = tables
        if make_plots:
            plots(tables_for_plots, output)
        (output / "report_cn.md").write_text(
            f"# 冻结部署技术汇总\n\n阶段：{phase}；policy-world-repeat={len(data['units'])}；车队累积量行数={len(rows)}。\n\n"
            f"本批新执行={sum(r['data_role'] == 'new' for r in records)}；只读复用={sum(r['data_role'] == 'reused' for r in records)}。"
            "复用单元保留原评估commit、路径和完成记录，不复制原始数组、不新增策略执行计数。\n\n"
            "scene-first 在同一 world/车队合并 repeats 的 sum/count，再分别取最大 AoI 和最小 CAM；两者可来自不同车队。"
            "legacy 先跨 world 合并每个车队，再取各自极值。所有训练 seed 保留；训练 seed 是推断重复单位。\n\n"
            "业务水平、跨 world sample SD、跨 seed sample SD、动作 repeat sample SD 单独报告，不能相加或作为因果分解。"
            "确定性条件只有一次 repeat，repeat SD 留空。只有六训练 seed 的摘要提供 df=5 描述性 t95 区间。"
            "技术 PASS 不依赖比较方向；主张未自动解读。gap-transfer 是旧 world 的观察性扩展。\n",
            encoding="utf-8")
        (output / "fields_and_boundaries.md").write_text(
            "# 字段与边界\n\n逐流键：model_id/training_seed/condition/target_scenario/world/repeat/platoon。"
            "每流10000 scored slots、100个CAM deadline；warmup不在数组中，禁止再删除前五episode。"
            "AoI post-step；gt50严格>50；CAM=success的每episode最后一槽；功率逐槽dBm转mW后累加。"
            "reward_global在每车队账本重复表示同一world标量，不能把五份当五个独立奖励。"
            "reward_combined=task1+task2+source global_actor_weight×global。任务分量保持源环境定义。\n\n"
            "V2I attempt来自mode==0；reset来自经代码语义核验的post-step AoI≈1；空值写空单元格，不补0。"
            "v2i_success_given_attempt在attempt=0时未定义。各指标sum/count明确列出；辅助指标未定义则完整保留缺失覆盖。\n\n"
            "world_endpoints：repeat先按车队合并。repeat_endpoints：每repeat重新选择最弱车队，仅供波动描述。"
            "repeat_sd_fixed_platoon：scene-first选定车队后跨repeat sample SD，极值并列时等权平均SD。"
            "world_sd：各训练seed内跨world sample SD；condition_summary.training_seed_sd是该行指标跨seed的sample SD。"
            "重复数1时SD留空，CI跨零不代表等效。PNG均附有同名指标的CSV源，未采用显著性筛选。\n",
            encoding="utf-8")
        result = {"technical_status": "PASS", "claim_support": "NOT_ASSESSED", "phase": phase,
                  "implementation_commit": manifest["implementation_commit"], "new_training_runs": 0,
                  "policy_world_repeats": len(data["units"]), "platoon_rows": len(rows),
                  "new_policy_world_repeats": sum(r["data_role"] == "new" for r in records),
                  "reused_policy_world_repeats": sum(r["data_role"] == "reused" for r in records),
                  "scored_agent_slots": sum(r["slot_count"] for r in rows),
                  "cam_deadlines": sum(r["cam_episode_count"] for r in rows),
                  "training_seeds": data["training_seeds"], "worlds": data["worlds"],
                  "complete_unit_coverage": True, "exogenous_process_probes_exact": True,
                  "plots_generated": make_plots, "historical_bridge_requested": evidence_pack is not None}
        if phase == "gap-transfer":
            result.update({"reused_mappo_gap_evaluation_cells": 24, "reused_mappo_policy_worlds": 144,
                           "reused_mappo_platoon_rows": 720, "new_mappo_evaluations": 0})
        if extension is not None:
            result.update(extension(rows, tables, output, manifest, sources))
        dc.code_check(manifest["implementation_commit"])
        dc.write(output / "verification.json", result)
        return result
    except Exception as exc:
        dc.write(output / "verification.json", {"technical_status": "FAIL", "claim_support": "NOT_ASSESSED", "error": str(exc)})
        raise


def export(batch, phase, attempt="a01"):
    dc.require(re.fullmatch(r"a[0-9]{2,4}", attempt), "unsafe analysis attempt")
    dc.load_batch(batch)
    batch = Path(batch).resolve()
    analysis = batch / "analysis" / phase / attempt
    dc.require(dc.read(analysis / "verification.json")["technical_status"] == "PASS", "no technical PASS")
    dest = analysis / "core_evidence.tar.zst"
    dc.require(not dest.exists(), "evidence archive exists")
    files = [p for root in (batch / "protocol", analysis) for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".csv", ".md", ".png"}]
    require_names = {"verification.json", "completion_records.csv", "world_repeat_platoon.csv", "condition_summary.csv"}
    dc.require(require_names <= {p.name for p in files}, "incomplete evidence list")
    subprocess.run(["tar", "-I", "zstd", "-cf", str(dest), "-C", str(batch), *[str(p.relative_to(batch)) for p in files]], check=True)
    return {"archive": str(dest), "files": len(files), "raw_data_included": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("summary", "export", "evidence-regression"))
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--phase", choices=("pilot", "main", "gap-transfer"))
    parser.add_argument("--attempt", default="a01")
    parser.add_argument("--evidence-pack", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evidence-regression":
        dc.require(args.evidence_pack is not None, "--evidence-pack required")
        result = evidence_regression(args.evidence_pack)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1
    dc.require(args.batch_root is not None and args.phase is not None, "--batch-root/--phase required")
    result = summarize(args.batch_root, args.phase, args.attempt, args.evidence_pack, not args.no_plots) if args.command == "summary" else export(args.batch_root, args.phase, args.attempt)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
