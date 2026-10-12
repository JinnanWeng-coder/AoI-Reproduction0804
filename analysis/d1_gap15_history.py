"""Read-only same-world references and Fig.4 tables; no model or NPZ loading."""
from __future__ import annotations

import math
import posixpath
from pathlib import Path

from analysis import deployment_contract as dc
from analysis import summarize_frozen_deployment as su
from analysis.evaluate_frozen_deployment import SUM_METRICS, csv_read, csv_write

OLD_COMMIT = "75098b00714daf315a1d82298deb563209da9ac8"
CONDITIONS = ("shared_tdec_stochastic", "independent_tdec_stochastic", "algorithm2_noise0", "algorithm2_noise0p3")
WORLDS = list(range(213, 219))
MODELS = ("shared_tdec", "independent_tdec", "algorithm2")
FIG_METRICS = ("scene_first_worst_aoi_ms", "scene_first_worst_cam_success", "scene_first_mean_aoi_ms",
               "scene_first_mean_cam_success", "scene_first_mean_power_mw", "scene_first_mean_reward_combined")


def evidence_checkpoint(e):
    # Evidence paths are HPC POSIX paths even when reviewed on Windows.
    return posixpath.normpath(posixpath.join(e["output_path"], e["source_policy"]))


def normalize(rows):
    """Parse only identifiers and cumulative numeric fields, without rounding."""
    fields = {v for pair in SUM_METRICS.values() for v in pair}
    result = []
    for original in rows:
        row = dict(original)
        for key in ("training_seed", "world", "repeat", "platoon"):
            row[key] = int(row[key])
        for key in fields:
            row[key] = su.number(row.get(key))
        result.append(row)
    return result


def coverage(rows, targets):
    keys = [(r["condition"], r["target_scenario"], int(r["training_seed"]), int(r["world"]), int(r["repeat"]), int(r["platoon"])) for r in rows]
    expected = {(c, t, s, w, 0, p) for c in CONDITIONS for t in targets for s in dc.SEEDS for w in WORLDS for p in range(5)}
    dc.require(len(keys) == len(set(keys)) and set(keys) == expected, "historical/new same-world matrix coverage")
    dc.require(all(r["slot_count"] == 10000 and r["cam_episode_count"] == 100 and r["power_slot_count"] == 10000 for r in rows),
               "historical/new scored denominators")
    for r in rows:
        dc.require(r["aoi_sum_ms"] is not None and math.isfinite(r["aoi_sum_ms"]) and
                   10000 <= r["aoi_sum_ms"] <= 1000000 and
                   r["cam_deadline_success_count"] is not None and 0 <= r["cam_deadline_success_count"] <= 100 and
                   float(r["cam_deadline_success_count"]).is_integer() and
                   r["power_mw_sum"] is not None and math.isfinite(r["power_mw_sum"]) and r["power_mw_sum"] > 0,
                   "invalid historical/new AoI/CAM/power cumulative values")


def baseline_rows(pack, sources):
    """Use E1 Algorithm 2 and the exact reused MAPPO dev evaluation, never main."""
    from analysis.e1_contract import maddpg_eval_id
    agents, _, _, evaluations = su.evidence_tables(pack)
    by_eval = su.grouped(agents, ("eval_id",))
    result, inventory = [], []
    for condition in CONDITIONS:
        model_id, _execution, noise, _repeats = dc.CONDITIONS[condition]
        for seed in dc.SEEDS:
            source = next(s for s in sources if s["model_id"] == model_id and s["training_seed"] == seed)
            matches = [e for e in evaluations if int(e["training_seed"]) == seed and
                       evidence_checkpoint(e) == source["checkpoint"]["path"] and
                       (e["target_scenario"] or e["scenario"]) == dc.SCENARIO and
                       e["world_ids"] == ",".join(map(str, WORLDS)) and
                       (e["eval_dir_name"].endswith("feasibility_reward_arm_baseline") and
                        e["algorithm"] == "mappo" and e["action_mode"] == "stochastic" if model_id != "algorithm2" else
                        e["stage"] == "E1" and e["algorithm"] == "modified_maddpg_tdec" and e["eval_dir_name"] == maddpg_eval_id(noise))]
            dc.require(len(matches) == 1, f"gap25 same-world reference missing/ambiguous: {condition}/seed{seed}")
            e = matches[0]
            dc.require(e["actual_status"] == "complete" and e["frozen_check"] == "true" and
                       e["eval_protocol"] == "sequential_warm" and e["warmup_episodes"] == "5" and
                       e["scored_episodes_recorded"] == "100" and e["slots_recorded"] in {"", "100"},
                       "gap25 reference protocol/frozen mismatch")
            inventory.append({"condition": condition, "training_seed": seed, "target_scenario": dc.SCENARIO,
                              "source_eval_id": e["eval_id"], "source_evaluation_path": e["output_path"],
                              "evaluation_implementation_commit": e["evaluator_commit"],
                              "model_training_commit": source["training_commit"],
                              "training_source_commit_recorded": e.get("training_source_commit"),
                              "source_checkpoint": source["checkpoint"]["path"], "data_role": "reused",
                              "slots_metadata_recorded": e["slots_recorded"],
                              "slots_per_episode_verified_from_cumulative_denominators": 100})
            for old in by_eval[(e["eval_id"],)]:
                row = {"condition": condition, "model_id": model_id, "training_seed": seed, "target_scenario": dc.SCENARIO,
                       "world": int(old["world"]), "platoon": int(old["agent"]), "repeat": 0, "data_role": "reused",
                       "source_eval_id": e["eval_id"], "source_evaluation_path": e["output_path"],
                       "evaluation_implementation_commit": e["evaluator_commit"], "source_checkpoint": source["checkpoint"]["path"],
                       "model_training_commit": source["training_commit"]}
                row.update({f: None for pair in SUM_METRICS.values() for f in pair})
                mapping = {"slot_count": "aoi_slot_count", "aoi_sum_ms": "aoi_sum_ms", "aoi_gt50_count": "aoi_gt50_count",
                           "aoi_at_cap_count": "aoi_at_cap_count", "cam_episode_count": "binary_cam_episode_count",
                           "cam_deadline_success_count": "binary_cam_success_count", "power_mw_sum": "power_mw_sum",
                           "power_slot_count": "power_slot_count", "reward_slot_count": "reward_slot_count",
                           "reward_global_sum": "reward_global_sum", "reward_task1_sum": "reward_task1_sum",
                           "reward_task2_sum": "reward_task2_sum", "reward_combined_sum": "reward_combined_recomputed_sum",
                           "v2i_attempt_count": "V2I_attempt_count"}
                row.update({new: su.number(old[before]) for new, before in mapping.items()})
                row["v2i_slot_count"] = row["slot_count"] if row["v2i_attempt_count"] is not None else None
                payload = su.number(old["payload_completion_mean"])
                row["payload_completion_sum"] = payload * row["cam_episode_count"] if payload is not None else None
                result.append(row)
    coverage(result, [dc.SCENARIO])
    return result, inventory


def historical(old_batch, pack, sources):
    """Validate the old 75098b0 snapshot; current execution checks remain enabled."""
    old_batch = Path(old_batch).resolve()
    old_manifest, old_sources = dc.load_batch(old_batch, check_code=False)  # Historical read only.
    dc.require(old_manifest["implementation_commit"] == OLD_COMMIT, "historical D1 implementation mismatch")
    expected_sources = [s for s in old_sources if s["model_id"] in MODELS]
    dc.require(sorted(sources, key=lambda x: (x["model_id"], x["training_seed"])) ==
               sorted(expected_sources, key=lambda x: (x["model_id"], x["training_seed"])), "historical 18-model identity mismatch")
    base = old_batch / "analysis/gap-transfer/a01"
    verification = dc.read(base / "verification.json")
    dc.require(verification["technical_status"] == "PASS" and verification["implementation_commit"] == OLD_COMMIT and
               verification["policy_world_repeats"] == 144 and verification["worlds"] == WORLDS,
               "historical gap-transfer is not accepted")
    raw_path, seed_path = base / "gap_alignment_world_repeat_platoon.csv", base / "gap_alignment_seed_endpoints.csv"
    rows = normalize(csv_read(raw_path))
    coverage(rows, ["p05_n04_g05", "p05_n04_g35"])
    recalculated = su.aggregate(rows)
    old_seeds = csv_read(seed_path)
    index = {(r["condition"], r["target_scenario"], int(r["training_seed"])): r for r in old_seeds}
    dc.require(len(old_seeds) == len(index) == 48, "historical gap seed coverage")
    for row in recalculated["seed_endpoints"]:
        before = index[(row["condition"], row["target_scenario"], row["training_seed"])]
        for field in FIG_METRICS:
            atol = 1e-12 if "cam" in field else 1e-10
            dc.require(math.isclose(row[field], float(before[field]), rel_tol=0, abs_tol=atol), f"historical scene-first conflict: {field}")
        # Retain exactly the recorded full-precision endpoint after reconciliation.
        row.update({field: float(before[field]) for field in FIG_METRICS})
    completions = csv_read(base / "completion_records.csv")
    expected = {(c, s, t, w) for c in CONDITIONS[2:] for s in dc.SEEDS for t in ("p05_n04_g05", "p05_n04_g35") for w in WORLDS}
    keys = [(r["condition"], int(r["training_seed"]), r["target_scenario"], int(r["world"])) for r in completions]
    dc.require(len(keys) == len(set(keys)) and set(keys) == expected and all(
        r["technical_status"] == "PASS" and r["evaluation_implementation_commit"] == OLD_COMMIT and int(r["repeat"]) == 0 and
        r["source_checkpoint"] == next(s["checkpoint"]["path"] for s in sources if s["model_id"] == "algorithm2" and s["training_seed"] == int(r["training_seed"]))
        for r in completions), "historical MADDPG gap provenance mismatch")
    gap25, inventory = baseline_rows(pack, sources)
    # Historical MAPPO source_eval_id must resolve to the very same trained model.
    _, _, _, evaluations = su.evidence_tables(pack)
    ei = {e["eval_id"]: e for e in evaluations}
    for r in rows:
        r["data_role"] = "reused"
        model_source = next(s for s in sources if (s["model_id"], s["training_seed"]) == (r["model_id"], r["training_seed"]))
        r["model_training_commit"] = model_source["training_commit"]
        r["source_checkpoint"] = model_source["checkpoint"]["path"]
        if r["model_id"] != "algorithm2":
            e = ei[r["source_eval_id"]]
            s = next(s for s in sources if (s["model_id"], s["training_seed"]) == (r["model_id"], r["training_seed"]))
            dc.require(evidence_checkpoint(e) == s["checkpoint"]["path"] and e["target_scenario"] == r["target_scenario"], "historical MAPPO model/target mismatch")
            r["evaluation_implementation_commit"], r["source_evaluation_path"] = e["evaluator_commit"], e["output_path"]
        else:
            c = next(c for c in completions if (c["condition"], int(c["training_seed"]), c["target_scenario"], int(c["world"])) ==
                     (r["condition"], r["training_seed"], r["target_scenario"], r["world"]))
            r["evaluation_implementation_commit"], r["source_evaluation_path"] = OLD_COMMIT, c["native_evaluation_dir"]
    files = [old_batch / "protocol/manifest.json", old_batch / "protocol/sources.json", base / "verification.json",
             raw_path, seed_path, base / "completion_records.csv"]
    return rows + gap25, recalculated["seed_endpoints"] + su.aggregate(gap25)["seed_endpoints"], inventory, files


def fig4_tables(seeds):
    dc.require(len(seeds) == 96 and len({(r["condition"], r["target_scenario"], r["training_seed"]) for r in seeds}) == 96,
               "Fig.4 requires 96 unique seed endpoints")
    wide = []
    for r in seeds:
        dc.require(r["condition"] in CONDITIONS and r["training_seed"] in dc.SEEDS and r["world_count"] == 6,
                   "Fig.4 seed/world identity")
        gap = int(r["target_scenario"].rsplit("g", 1)[1])
        dc.require(gap in {5, 15, 25, 35}, "Fig.4 gap set")
        dc.require(all(r[k] is not None and math.isfinite(r[k]) for k in FIG_METRICS), "undefined Fig.4 endpoint")
        wide.append({k: r[k] for k in ("condition", "model_id", "training_seed", "target_scenario", *FIG_METRICS)} |
                    {"gap_m": gap, "gap_definition": "intra-platoon vehicle gap; bumper-to-bumper",
                     "worlds": ",".join(map(str, WORLDS)), "world_count": 6, "repeat": 0,
                     "data_role": "new" if gap == 15 else "reused",
                     "worst_cam_success_percent": 100 * r["scene_first_worst_cam_success"]})
    dc.require({(r["condition"], r["gap_m"], r["training_seed"]) for r in wide} ==
               {(c, g, s) for c in CONDITIONS for g in (5, 15, 25, 35) for s in dc.SEEDS}, "Fig.4 full matrix")
    means, changes, paired = [], [], []
    baseline = {(r["condition"], r["training_seed"]): r for r in wide if r["gap_m"] == 25}
    for r in wide:
        b = baseline[(r["condition"], r["training_seed"])]
        change = {k: r[k] for k in ("condition", "training_seed", "target_scenario", "gap_m", "data_role")}
        for field in FIG_METRICS:
            change[field + "_minus_gap25"] = r[field] - b[field]
        change["worst_cam_change_pp"] = 100 * change["scene_first_worst_cam_success_minus_gap25"]
        change["mean_cam_change_pp"] = 100 * change["scene_first_mean_cam_success_minus_gap25"]
        changes.append(change)
    for (condition, gap), group in su.grouped(wide, ("condition", "gap_m")).items():
        m = {"condition": condition, "gap_m": gap, "n_training_seeds": 6, "df": 5, "data_role": "new" if gap == 15 else "reused"}
        for field in (*FIG_METRICS, "worst_cam_success_percent"):
            values = [r[field] for r in group]
            center, spread = su.mean(values), su.sd(values)
            half = su.T95 * spread / math.sqrt(6)
            m.update({field: center, field + "_sample_sd": spread, field + "_t95_lower": center - half, field + "_t95_upper": center + half})
        means.append(m)
    for (condition, gap), group in su.grouped(changes, ("condition", "gap_m")).items():
        for field in [k + "_minus_gap25" for k in FIG_METRICS] + ["worst_cam_change_pp", "mean_cam_change_pp"]:
            values = [r[field] for r in group]
            center, spread = su.mean(values), su.sd(values)
            half = su.T95 * spread / math.sqrt(6)
            unit = "percentage points" if field.endswith("_pp") else "fraction" if "cam" in field else "mW" if "power" in field else "ms" if "aoi" in field else "reward"
            paired.append({"condition": condition, "gap_m": gap, "metric": field, "unit": unit, "n_training_seeds": 6, "df": 5,
                           "mean_change": center, "sample_sd": spread, "t95_lower": center - half, "t95_upper": center + half,
                           "negative": sum(v < 0 for v in values), "zero": sum(v == 0 for v in values), "positive": sum(v > 0 for v in values),
                           "claim_support": "NOT_ASSESSED"})
    power = [{k: r[k] for k in ("condition", "training_seed", "gap_m", "data_role", "scene_first_mean_power_mw")} |
             {"gap25_mean_power_mw": baseline[(r["condition"], r["training_seed"])]["scene_first_mean_power_mw"],
              "power_change_mw": r["scene_first_mean_power_mw"] - baseline[(r["condition"], r["training_seed"])]["scene_first_mean_power_mw"]} for r in wide]
    return {"fig4_seed_endpoints": wide, "fig4_condition_means": means, "fig4_paired_seed_changes": changes,
            "fig4_paired_summary": paired, "fig4_power": power,
            "gap15_paired_seed_changes": [r for r in changes if r["gap_m"] == 15],
            "gap15_paired_summary": [r for r in paired if r["gap_m"] == 15],
            "historical_gap25_seed_reference": [r for r in wide if r["gap_m"] == 25]}
