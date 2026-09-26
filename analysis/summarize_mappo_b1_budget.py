"""B1 budget summaries: six seed replicates; historical compact tables only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis import mappo_b1_accounting as acc
from analysis import mappo_b1_contract as b1
from analysis.audit_mappo_w1_service_intervals import flow_statistics


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows, columns=None):
    columns = columns or sorted({k for row in rows for k in row})
    b1.require(bool(columns), f"missing CSV schema: {path}")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def historical_inputs(study, sources):
    """Reuse W1/N1 and E3 small tables, never read historical NPZ or rerun a policy."""
    w1 = study / "W1-service-intervals/P5_N4_gap25"
    e3 = study / "E3-locked-test/P5_N4_gap25/analysis"
    n1 = study / "N1-training-world-revisit/P5_N4_gap25/analysis/n1_e5_development_world_agent.csv"
    e5 = study / "E5-sharing-critic/P5_N4_gap25/analysis/e5_training_per_episode.csv"
    b1.require(b1.read(w1 / "verification.json")["status"] == "PASS", "W1 not verified")
    b1.require(b1.read(e3 / "verification.json")["status"] == "PASS", "E3 not verified")
    index = {(s["actor_structure"], s["value_configuration"], s["training_seed"]): s for s in sources}
    inventory = acc.unique(csv_read(w1 / "source_inventory.csv"), lambda r: acc.key({**r, "world": 0, "agent": 0})[:3])
    b1.require(set(inventory) == set(index), "W1 inventory coverage")
    for key, row in inventory.items():
        b1.require(Path(row["evaluation_dir"]).resolve() == Path(index[key]["historical_evaluations"]["development"]).resolve(), "W1 source mapping")
    dev = []
    for row in csv_read(n1):
        mapped = {**row, "actor_structure": row["structure"], "value_configuration": row["variant"],
                  "world": row["eval_world"], "aoi_slot_count": row["aoi_sample_count"],
                  "power_slot_count": row["power_sample_count"],
                  "reward_combined_sum": float(row["mean_reward_combined"]) * int(row["aoi_sample_count"]),
                  "reward_slot_count": row["aoi_sample_count"]}
        src = index[acc.key(mapped)[:3]]
        b1.require(row["evaluation_role"] == "development_worlds" and row["data_role"] == "reused"
                   and row["run_name"] == src["historical_run_name"]
                   and Path(row["source_root"]).resolve() == Path(src["historical_evaluations"]["development"]).resolve(), "N1/W1 source mismatch")
        dev.append(mapped)
    e3_inventory = acc.unique(csv_read(e3 / "source_inventory.csv"), lambda r: acc.key({**r, "world": 0, "agent": 0})[:3])
    b1.require(set(e3_inventory) == set(index), "E3 source coverage")
    for key, row in e3_inventory.items():
        b1.require(Path(row["evaluation_dir"]).resolve() == Path(index[key]["historical_evaluations"]["e3_reused"]).resolve(), "E3 source mapping")
    inputs = {"development": (dev, csv_read(w1 / "flow_statistics.csv"), csv_read(w1 / "flow_interval_frequency.csv")),
              "e3_reused": (csv_read(e3 / "c1_world_agent.csv"), csv_read(e3 / "c2_flow_statistics.csv"), csv_read(e3 / "c2_interval_frequency.csv"))}
    for world_set, tables in inputs.items():
        acc.analyze(*tables, b1.WORLDS[world_set], world_set)  # exact keys, moments, boundaries
    episodes = csv_read(e5)
    keys = [(r["structure"], r["variant"], int(r["training_seed"]), int(r["episode"])) for r in episodes]
    expected = {(a, v, s, e) for a, v in b1.CONDITIONS for s in range(8, 14) for e in range(1, 501)}
    b1.require(len(keys) == 12000 and set(keys) == expected, "historical episode table duplicate/missing rows")
    return inputs, episodes


def evaluation_tables(arrays, complete, item, world_set):
    service, flows, frequencies = [], [], []
    for wi, world in enumerate(b1.WORLDS[world_set]):
        for agent in range(5):
            identity = {k: item[k] for k in ("actor_structure", "value_configuration", "training_seed")}
            identity.update(world=world, agent=agent)
            aoi = arrays["aoi_ms"][wi, :, :, agent].astype(np.float64)
            cam = arrays["success"][wi, :, -1, agent]
            payload = np.clip(1 - arrays["remaining_demand"][wi, :, -1, agent].astype(np.float64) / complete["cam_bits"], 0, 1)
            reward = (arrays["reward_task1"][wi, :, :, agent].astype(np.float64)
                      + arrays["reward_task2"][wi, :, :, agent].astype(np.float64)
                      + complete["global_actor_weight"] * arrays["reward_global"][wi].astype(np.float64))
            power = np.power(10., arrays["executed_power_dbm"][wi, :, :, agent].astype(np.float64) / 10)
            service.append({**identity, "aoi_sum_ms": int(np.rint(aoi.sum())), "aoi_slot_count": aoi.size,
                            "aoi_gt50_count": int((aoi > 50).sum()), "aoi_at_cap_count": int((aoi == 100).sum()),
                            "binary_cam_success_count": int(cam.sum()), "binary_cam_episode_count": cam.size,
                            "payload_sum": float(payload.sum()), "payload_episode_count": payload.size,
                            "reward_combined_sum": float(reward.sum()), "reward_slot_count": reward.size,
                            "power_mw_sum": float(power.sum()), "power_slot_count": power.size})
            flow, freq, _ = flow_statistics(arrays["reset_event"][wi, :, :, agent].astype(bool))
            flows.append({**identity, "event_source": "reset_event", **flow})
            frequencies.extend({**identity, "length_L": int(length), "frequency": int(n)} for length, n in sorted(freq.items()))
    return service, flows, frequencies


METRICS = ("aoi_ms", "worst_binary_cam", "reset_rate", "contrib_le50", "contrib_mid50", "contrib_gt100",
           "contrib_boundary", "contrib_long_complete", "group_mean_aoi_ms", "group_reset_rate", "mean_binary_cam",
           "mean_payload_completion", "worst_agent_payload_completion", "mean_power_mw", "mean_reward_combined",
           "aoi_gt50_fraction", "aoi_at_cap_fraction")


def endpoints_and_aux(tables, world_set, budget):
    result = acc.analyze(*tables, b1.WORLDS[world_set], world_set)
    services = tables[0]
    for row in result["endpoints"]:
        key = (row["actor_structure"], row["value_configuration"], row["training_seed"])
        group = [r for r in services if acc.key(r)[:3] == key]
        total = lambda name: sum(float(r[name]) for r in group)
        row["worst_binary_cam"] = next(r["worst_binary_cam"] for r in result["cams"] if acc.key({**r, "world": 0, "agent": 0})[:3] == key)
        cam_info = next(r for r in result["cams"] if acc.key({**r, "world": 0, "agent": 0})[:3] == key)
        row["cam_argmin_set"] = cam_info["cam_argmin_set"]
        row["mean_binary_cam"] = total("binary_cam_success_count") / total("binary_cam_episode_count")
        row["mean_payload_completion"] = total("payload_sum") / total("payload_episode_count")
        row["worst_agent_payload_completion"] = min(sum(float(r["payload_sum"]) for r in group if int(r["agent"]) == a) / 600 for a in range(5))
        row["mean_power_mw"] = total("power_mw_sum") / total("power_slot_count")
        row["mean_reward_combined"] = total("reward_combined_sum") / total("reward_slot_count")
        row["aoi_gt50_fraction"] = total("aoi_gt50_count") / total("aoi_slot_count")
        row["aoi_at_cap_fraction"] = total("aoi_at_cap_count") / total("aoi_slot_count")
    for name in result:
        for row in result[name]:
            row["budget_env_steps"] = budget
    return result


def budget_effects(endpoints):
    index = acc.unique(endpoints, lambda r: (r["world_set"], r["actor_structure"], r["value_configuration"], r["training_seed"], r["budget_env_steps"]))
    paired, effects, levels = [], [], []
    comparisons = {"C-a": (("shared", 50000), ("independent", 250000)),
                   "C-b": (("shared", 250000), ("independent", 250000)),
                   "C-c": (("independent", 250000), ("independent", 50000))}
    for ws in b1.WORLDS:
        for variant in ("combined", "tdec"):
            for comparison, (left, right) in comparisons.items():
                for metric in METRICS:
                    values = []
                    for seed in range(8, 14):
                        a, b = [index[(ws, arm, variant, seed, budget)] for arm, budget in (left, right)]
                        scale = 100 if metric in ("worst_binary_cam", "mean_binary_cam") else 1
                        delta = scale * (a[metric] - b[metric])
                        values.append(delta)
                        paired.append({"world_set": ws, "value_configuration": variant, "comparison": comparison,
                                       "training_seed": seed, "quantity": metric, "unit": "percentage_points" if scale == 100 else "native",
                                       "left_structure": left[0], "left_budget": left[1], "right_structure": right[0], "right_budget": right[1],
                                       "left_selected_aoi_agent": a["agent"], "right_selected_aoi_agent": b["agent"],
                                       "left_value": a[metric], "right_value": b[metric], "delta": delta})
                    effects.append({"world_set": ws, "value_configuration": variant, "comparison": comparison,
                                    "quantity": metric, "unit": "percentage_points" if metric in ("worst_binary_cam", "mean_binary_cam") else "native",
                                    **acc.summary(values), "claim_support": "NOT_ASSESSED"})
            for arm in ("independent", "shared"):
                for budget in (50000, 250000):
                    for metric in METRICS:
                        levels.append({"world_set": ws, "actor_structure": arm, "value_configuration": variant,
                                       "budget_env_steps": budget, "quantity": metric,
                                       **acc.summary([index[(ws, arm, variant, s, budget)][metric] for s in range(8, 14)])})
    return paired, effects, levels


def training_curves(batch, sources, historical_episodes):
    from analysis.summarize_mappo_shared_actor import _metric_block
    episodes, updates, inventory = [], [], []
    for row in historical_episodes:
        identity = {"actor_structure": row["structure"], "value_configuration": row["variant"],
                    "training_seed": int(row["training_seed"]), "budget_env_steps": 50000,
                    "episode": int(row["episode"]), "environment_steps": int(row["episode"]) * 100}
        metrics = {}
        for k, v in row.items():
            if k in ("structure", "variant", "training_seed", "episode"):
                continue
            if k == "screen_success":
                continue  # screening never enters B1 accounting
            number = float(v) if v else float("nan")
            metrics[k] = number if math.isfinite(number) else None
        episodes.append({**identity, **metrics})
    for item in sources:
        run_id = item["run_id"]
        accepted = b1.read(b1.record_path(batch, "train", run_id))
        for budget, run, count in ((50000, Path(item["historical_run"]), 500), (250000, Path(accepted["run_dir"]), 2500)):
            identity = {k: item[k] for k in ("actor_structure", "value_configuration", "training_seed")}
            identity["budget_env_steps"] = budget
            diagnostics = b1.read(run / "learning_diagnostics.json")
            counts = b1.diagnostics_check(diagnostics, item, count, budget == 250000)
            for row in diagnostics:
                actor_steps = row.get("actor_optimizer_step_count")
                updates.append({**identity, **row, "actor_adam_steps_total": actor_steps if actor_steps is not None else row["update"] * 10 * counts["actor_network_count"],
                                "actor_adam_count_source": "recorded" if actor_steps is not None else "derived_updates_x_epochs_x_actors"})
            inventory.append({**identity, "run_dir": str(run), "training_commit": b1.read(run / "COMPLETE.json")["reproduction_git_commit"],
                              "episode_rows": count, "update_rows": len(diagnostics), **counts})
            if budget == 50000:
                continue
            config = b1.read(run / "config.resolved.json")
            with np.load(run / "train_metrics.npz", allow_pickle=False) as data:
                arrays = {name: data[name] for name in b1.TRAIN_FIELDS}
            for e in range(count):
                at = lambda name: arrays[name][e:e + 1]
                metrics = _metric_block(aoi=at("aoi_ms"), success=at("success"), remaining=at("remaining_demand"),
                                        rb=at("rb"), mode=at("mode"), power_dbm=at("power_dbm"), reward_global=at("global_step"),
                                        reward_task1=at("task1_step"), reward_task2=at("task2_step"),
                                        cam_bits=config["cam_bits"], global_actor_weight=config["global_actor_weight"])
                metrics.pop("screen_success", None)
                episodes.append({**identity, "episode": e + 1, "environment_steps": (e + 1) * 100,
                                 **{k: v if math.isfinite(v) else None for k, v in metrics.items()}})
    b1.require(len(episodes) == 72000 and len(updates) == 14400, "combined curve counts")
    return episodes, updates, inventory


def analyze(batch, attempt):
    manifest = b1.check_batch(batch)
    b1.require(bool(re.fullmatch(r"a[0-9]{2,4}", attempt)), "unsafe analysis attempt")
    output = Path(batch) / "analysis" / attempt
    output.mkdir(exist_ok=False)
    try:
        sources = b1.read(Path(batch) / "protocol/source_inventory.json")
        historical, old_episodes = historical_inputs(Path(manifest["study_root"]), sources)
        for i in range(24):
            r = b1.read(b1.record_path(batch, "train", i))
            b1.check_training(batch, i, r["attempt"])
        all_endpoints, tables_out, source_rows = [], {}, []
        for ws in b1.WORLDS:
            new_tables = ([], [], [])
            for i in range(24):
                identity = 2 * i + tuple(b1.WORLDS).index(ws)
                record = b1.read(b1.record_path(batch, "eval", identity))
                b1.check_evaluation(batch, identity, record["attempt"])
                directory = Path(record["evaluation_dir"])
                extracted = evaluation_tables(b1.evaluation_arrays(directory), b1.read(directory / "EVAL_COMPLETE.json"), b1.cell(i), ws)
                for dest, rows in zip(new_tables, extracted):
                    dest.extend(rows)
                source_rows.append({**b1.cell(i), "world_set": ws, "evaluation_dir": str(directory),
                                    "training_run_dir": record["training_run_dir"], "evaluation_commit": record["implementation_commit"]})
            for budget, source_tables in ((50000, historical[ws]), (250000, new_tables)):
                result = endpoints_and_aux(source_tables, ws, budget)
                all_endpoints.extend(result["endpoints"])
                for name in ("flows", "agents", "endpoints", "cams"):
                    tables_out[f"{ws}_b{budget}_{name}.csv"] = result[name]
                tables_out[f"{ws}_b{budget}_world_agent.csv"] = [{**r, "world_set": ws, "budget_env_steps": budget} for r in source_tables[0]]
            tables_out[f"{ws}_b250000_interval_frequency.csv"] = new_tables[2]
        paired, effects, levels = budget_effects(all_endpoints)
        episodes, updates, training_inventory = training_curves(batch, sources, old_episodes)
        for budget, ep_count, update_count in ((50000, 12000, 2400), (250000, 60000, 12000)):
            b1.require(sum(r["budget_env_steps"] == budget for r in episodes) == ep_count and
                       sum(r["budget_env_steps"] == budget for r in updates) == update_count, "curve budget coverage")
        tables_out.update({"paired_effects.csv": paired, "effect_summary.csv": effects, "condition_summary.csv": levels,
                           "training_episode_curves.csv": episodes, "training_update_curves.csv": updates,
                           "training_curve_inventory.csv": training_inventory, "new_source_inventory.csv": source_rows})
        for name, rows in tables_out.items():
            csv_write(output / name, rows, list(acc.KEYS) + ["length_L", "frequency"] if name.endswith("interval_frequency.csv") else None)
        result = {"technical_status": "PASS", "claim_support": "NOT_ASSESSED", "implementation_commit": manifest["code_audit"]["implementation_commit"],
                  "new_training_runs": 24, "reused_training_runs": 24, "new_evaluation_cells": 48, "reused_evaluation_cells": 48,
                  "new_training_environment_steps": 6000000, "new_policy_world_runs": 288,
                  "new_eval_environment_steps_including_warmup": 3024000,
                  "new_scored_agent_slots": 14400000, "new_cam_agent_episode_events": 144000,
                  "new_training_episode_rows": 60000, "historical_training_episode_rows": 12000,
                  "new_update_rows": 12000, "historical_update_rows": 2400,
                  "new_endpoint_rows": sum(len(tables_out[f"{ws}_b250000_endpoints.csv"]) for ws in b1.WORLDS),
                  "new_world_agent_rows": sum(len(tables_out[f"{ws}_b250000_world_agent.csv"]) for ws in b1.WORLDS),
                  "new_agent_rows": sum(len(tables_out[f"{ws}_b250000_agents.csv"]) for ws in b1.WORLDS),
                  "world_sets_analyzed_separately": True, "all_six_seeds_retained": True,
                  "df": 5, "checks": ["source_config_derivation", "recorded_optimizer_counts", "frozen_protocol",
                                       "post_step_recurrence", "exact_key_coverage", "four_area_closure", "own_extrema", "seed_pairing"],
                  "output_files": list(tables_out)}
        with (output / "report_cn.md").open("x", encoding="utf-8") as handle:
            handle.write("# B1 Tier 1 技术验收\n\ntechnical_status: PASS\nclaim_support: NOT_ASSESSED\n\n"
                         "24 个250k从头训练、48个新评估cell；历史24个50k训练及48个评估仅复用。\n"
                         "开发世界与E3历史锁定复用世界分别分析；303–308已被查看，不是新的未见确认。\n"
                         "C-a=shared50k−independent250k；C-b=shared250k−independent250k；C-c=independent250k−independent50k。\n"
                         "CAM效应单位百分点；AoI/四贡献单位ms；复位率无预设有利方向。每比较n=6，描述性95% t区间df=5，未校正。\n"
                         "技术通过不判定追上、等效、差距消失或因果机制。训练episode曲线不是固定世界冻结策略的学习曲线。\n")
        with (output / "fields_and_boundaries.md").open("x", encoding="utf-8") as handle:
            handle.write("# 字段和边界\n\n每臂每预算先按agent跨6世界合并sum/count，再独立选择AoI最大与CAM最小。"
                         "整数有理数确定完整并列集合，AoI并列取最小索引且输出复位率及贡献范围。\n"
                         "完整周期[u_j,u_(j+1))，S100(L)=L(L+1)/2 (L≤100)，否则5050+100(L−100)。"
                         "le50、mid50(50<L≤100)、gt100按整个完整周期分箱；boundary独立。long_complete=mid50+gt100。\n"
                         "无复位/单次复位流保留，最大完整间隔无定义时留空。跨episode连续，不跨world；scored数组不再删warmup。\n"
                         "训练旧episode只取E5的12000行，旧update按唯一训练路径读取2400行；新数据60000/12000行。"
                         "曲线保留原序列；数值无定义的辅助指标留空，screen_success不参与筛选。"
                         "独立/共享相同预算每actor Adam步数相同；总actor步数不同，C-a也不匹配critic训练量或轨迹分布。\n")
        b1.write(output / "verification.json", result)
        return result
    except Exception as exc:
        if not (output / "verification.json").exists():
            b1.write(output / "verification.json", {"technical_status": "FAIL", "claim_support": "NOT_ASSESSED", "error": str(exc)})
        raise


def package(batch, attempt):
    b1.check_batch(batch)
    b1.require(bool(re.fullmatch(r"a[0-9]{2,4}", attempt)), "unsafe analysis attempt")
    output = Path(batch) / "analysis" / attempt
    b1.require(b1.read(output / "verification.json")["technical_status"] == "PASS", "analysis not PASS")
    archive = Path(batch) / "core_evidence.tar.zst"
    b1.require(not archive.exists(), "evidence archive exists")
    files = []
    for sub in ("protocol", "configs", "records", f"analysis/{attempt}"):
        files.extend(p for p in (Path(batch) / sub).rglob("*") if p.is_file() and p.suffix in (".csv", ".json", ".md"))
    for i in range(24):
        run = Path(b1.read(b1.record_path(batch, "train", i))["run_dir"])
        files.extend(run / name for name in ("config.resolved.json", "COMPLETE.json", "provenance.json"))
    for i in range(48):
        directory = Path(b1.read(b1.record_path(batch, "eval", i))["evaluation_dir"])
        files.extend(directory / name for name in ("EVAL_COMPLETE.json", "provenance.json"))
    b1.require(all(p.resolve().is_relative_to(Path(batch).resolve()) and p.is_file() for p in files), "package path outside batch/missing")
    subprocess.run(["tar", "-I", "zstd", "-cf", str(archive), "-C", str(batch),
                    *sorted({str(p.relative_to(batch)) for p in files})], check=True)
    return {"archive": str(archive), "files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze", "package"))
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--attempt", default="a01")
    args = parser.parse_args()
    print(json.dumps((analyze if args.command == "analyze" else package)(args.batch_root.resolve(), args.attempt), allow_nan=False))


if __name__ == "__main__":
    main()
