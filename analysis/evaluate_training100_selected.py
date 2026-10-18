"""Selected-seed best100/final exploratory evaluation: 5 x 30 x 2, no training."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analysis import deployment_contract as dc, training100_contract as tc
from analysis.deployment_rng import action_seed_words, policy_seed
from analysis.evaluate_frozen_deployment import csv_read, csv_write, load_arrays, reduce_arrays, _check_csv

WORLDS = list(range(1001, 1031))
ROOT = Path("Training100-selected-eval/P5_N4_gap25")
ROLES = {"best100": "best_training100", "final": "final_training100"}
WORLD_ROLE = "user_fixed_exploratory_worlds"
CONDITIONS = dict(list(dc.CONDITIONS.items())[:5])


def candidates(training_batch):
    """Metadata/CSV only: preserves all 24 candidates, never consults test results."""
    training_batch = Path(training_batch).resolve()
    manifest = dc.read(training_batch / "protocol/manifest.json")
    summary = training_batch / "analysis/a01"
    verification = dc.read(summary / "verification.json")
    dc.require(verification["technical_status"] == "PASS" and verification["accepted_runs"] == 24 and
               verification["episode_rows"] == 60000 and verification["training_env_steps"] == 6000000,
               "training batch must have accepted 24-run summary")
    dc.require(verification["implementation_commit"] == manifest["implementation_commit"] and
               manifest["rule"] == tc.RULE, "training manifest/summary mismatch")
    selected = csv_read(summary / "selected_policies.csv")
    dc.require(len(selected) == 24 and {int(r["run_id"]) for r in selected} == set(range(24)), "selection CSV coverage")
    curves = defaultdict(list)
    for row in csv_read(summary / "training_episode_curves.csv"):
        curves[int(row["run_id"])].append(row)
    dc.require(set(curves) == set(range(24)), "training curve run coverage")
    records = []
    for i in range(24):
        record = dc.read(training_batch / f"records/train_{i:02d}.json")
        dc.require(all(record[k] == v for k, v in tc.cell(i).items()) and
                   record["technical_status"] == "PASS" and record["from_scratch"] is True and
                   record["implementation_commit"] == manifest["implementation_commit"], "accepted training record mismatch")
        old = next(r for r in selected if int(r["run_id"]) == i)
        for key in ("model_id", "run_dir", "implementation_commit"):
            dc.require(old[key] == record[key], f"selection/record {key} mismatch")
        rows = sorted(curves[i], key=lambda r: int(r["episode"]))
        dc.require([int(r["episode"]) for r in rows] == list(range(1, 2501)) and all(
            r["model_id"] == record["model_id"] and int(r["training_seed"]) == record["training_seed"] for r in rows),
            "training curve episode/model/seed coverage")
        rewards = np.array([float(r["episode_reward"]) for r in rows], dtype=np.float64)
        dc.require(np.isfinite(rewards).all(), "nonfinite training reward")
        episode, score, _ = tc.best_index(rewards)
        for key, value in {"best_episode": episode, "best_training100_score": score,
                           "last100_training_score": float(np.mean(rewards[-100:], dtype=np.float64))}.items():
            dc.require(record[key] == value and float(old[key]) == value, f"recomputed {key} mismatch")
        records.append(record)
    dc.require(len({r["run_dir"] for r in records}) == 24, "duplicate accepted training runs")
    return records


def select(records):
    dc.require(len(records) == 24 and {(r["model_id"], r["training_seed"]) for r in records} ==
               {(m, s) for m in dc.MODELS for s in dc.SEEDS}, "four models x six seeds required")
    result = []
    for role, field in (("best100", "best_training100_score"), ("final", "last100_training_score")):
        for model in dc.MODELS:
            group = [r for r in records if r["model_id"] == model]
            dc.require(all(np.isfinite(float(r[field])) for r in group), "nonfinite selection score")
            # Exact ties choose the lowest training seed, fixed before evaluation.
            winner = min(group, key=lambda r: (-float(r[field]), int(r["training_seed"])))
            result.append({"source_id": f"{model}__{role}", "checkpoint_kind": role,
                           "selection_metric": field, "selection_score": float(winner[field]),
                           "candidate_count": 6, "record": winner})
    return result


def make_units(sources):
    units = []
    for kind in ROLES:
        for condition, (model, execution, noise, _) in CONDITIONS.items():
            source = next(s for s in sources if s["source_id"] == f"{model}__{kind}")
            seed = source["record"]["training_seed"]
            for world in WORLDS:
                units.append({"unit_id": len(units), "source_id": source["source_id"], "model_id": model,
                              "condition": f"{condition}__{kind}", "execution_condition": condition,
                              "checkpoint_kind": kind, "training_seed": seed, "world": world, "repeat": 0,
                              "execution": execution, "noise_std": noise, "target_scenario": dc.SCENARIO,
                              "action_seed_words": action_seed_words(seed, world, 0, dc.MODELS[model]["family"])})
    return units


def source_check(source, payload=False):
    from aoi_v2x_reproduction.config import config_from_dict
    record = source["record"]
    dc.require(dc.read(source["record_file"]["path"]) == record, "accepted source record changed")
    for identity in [source["checkpoint"], source["record_file"], *source["metadata_files"]]:
        dc.require(dc.file_identity(identity["path"]) == identity, f"source file changed: {identity['path']}")
    run = Path(record["run_dir"]).resolve()
    cfg = config_from_dict(dc.read(run / "config.resolved.json"))
    complete = dc.read(run / "COMPLETE.json")
    spec = dc.MODELS[record["model_id"]]
    dc.require(cfg.episodes == 2500 and cfg.steps_per_episode == 100 and cfg.seed == record["training_seed"] and
               cfg.scenario.id == dc.SCENARIO and cfg.n_rb == 3 and cfg.number_agents == 5 and
               cfg.global_actor_weight == 1 and cfg.eval_protocol == "sequential_warm" and cfg.eval_warmup_episodes == 5,
               "source training/evaluation configuration mismatch")
    dc.require(cfg.algorithm == ("mappo" if spec["family"] == "mappo" else "modified_maddpg_tdec"), "source algorithm")
    if spec["family"] == "mappo":
        dc.require(bool(cfg.mappo_actor_sharing) == (spec["structure"] == "shared") and
                   cfg.mappo_variant == spec["variant"], "source MAPPO configuration")
    dc.require(complete["status"] == "complete" and complete["config_hash"] == cfg.canonical_hash() == source["config_hash"] and
               complete["reproduction_git_commit"] == record["implementation_commit"] and
               complete["training_reward_selection"]["rule"] == tc.RULE and
               complete["training_reward_selection"]["final_episode"] == 2500, "source completion/config mismatch")
    selection = complete["training_reward_selection"]
    dc.require(selection["best"]["selected_episode"] == record["best_episode"] and
               selection["best"]["score"] == record["best_training100_score"] and
               selection["best"]["policy_version"] == record["best_policy_version"] and
               selection["final_policy_version"] == record["final_policy_version"], "source selection metadata mismatch")
    if payload:
        from aoi_v2x_reproduction.runtime.training_selection import load_training_policy, load_training_actors
        _, data = load_training_policy(source["checkpoint"]["path"])
        load_training_actors(source["checkpoint"]["path"], device="cpu")
        dc.require(data["episode"] == source["policy_episode"] and data["policy_version"] == source["policy_version"] and
                   data["checkpoint_role"] == ROLES[source["checkpoint_kind"]], "selected payload identity")
    return cfg


def init(training_batch, batch_id, expected_commit):
    commit = tc.code_check(expected_commit)
    training_batch = Path(training_batch).resolve()
    original = dc.read(training_batch / "protocol/manifest.json")
    dc.require(original["batch_root"] == str(training_batch), "use original lab training batch, not relocated evidence")
    dc.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id), "unsafe batch ID")
    study = Path(original["study_root"]).resolve()
    dc.require(training_batch.parent == study / tc.ROOT, "training batch outside study root")
    batch = study / ROOT / batch_id
    dc.require(not batch.exists(), "new unique evaluation batch required")
    records = candidates(training_batch)
    sources = select(records)
    for source in sources:
        record, role = source["record"], source["checkpoint_kind"]
        key = "best" if role == "best100" else "final"
        run = Path(record["run_dir"]).resolve()
        dc.require(run.is_relative_to(training_batch / "training"), "run outside original training batch")
        expected_name = tc.BEST if role == "best100" else "policy_final.pt"
        dc.require(Path(record[key]["path"]).resolve() == run / expected_name, "wrong accepted checkpoint path")
        complete = dc.read(run / "COMPLETE.json")
        source.update(checkpoint=record[key], config_hash=complete["config_hash"],
                      policy_episode=record["best_episode"] if role == "best100" else 2500,
                      policy_version=record["best_policy_version" if role == "best100" else "final_policy_version"],
                      record_file=dc.file_identity(training_batch / f"records/train_{record['run_id']:02d}.json"),
                      metadata_files=[dc.file_identity(run / n) for n in
                                      ("config.resolved.json", "COMPLETE.json", "training_reward_selection.csv")])
        source_check(source)
    configs = [source_check(s).to_dict() for s in sources]
    dc.require(all({k: c[k] for k in dc.ENV_FIELDS} == {k: configs[0][k] for k in dc.ENV_FIELDS} for c in configs),
               "selected models use different effective environments")
    batch.mkdir(parents=True, exist_ok=False)
    for name in ("protocol", "units", "records", "logs", "tmp"):
        (batch / name).mkdir()
    manifest = {"schema": "training100_selected_eval_v1", "batch_root": str(batch), "study_root": str(study),
                "training_batch": str(training_batch), "implementation_commit": commit, "training_commit": original["implementation_commit"],
                "worlds": WORLDS, "world_role": WORLD_ROLE, "new_unseen_confirmation": False,
                "selection_ties": "lowest_training_seed", "selection_uses_evaluation_results": False,
                "n_selected_training_seeds_per_condition": 1, "candidate_training_seeds": dc.SEEDS,
                "protocol": {"name": "sequential_warm", "warmup": 5, "scored": 100, "slots": 100, "repeat": 0},
                "policy_world_repeat_count": 300, "platoon_rows": 1500, "scored_agent_slots": 15000000,
                "cam_deadlines": 150000, "new_training": 0, "pilot_required": False}
    dc.write(batch / "protocol/manifest.json", manifest)
    dc.write(batch / "protocol/sources.json", sources)
    dc.write(batch / "protocol/candidates.json", records)
    dc.write(batch / "protocol/units.json", make_units(sources))
    csv_write(batch / "protocol/selected_models.csv", [
        {k: s[k] for k in ("source_id", "checkpoint_kind", "selection_metric", "selection_score", "candidate_count", "policy_episode", "policy_version")} |
        {"model_id": s["record"]["model_id"], "training_seed": s["record"]["training_seed"],
         "checkpoint": s["checkpoint"]["path"], "training_commit": s["record"]["implementation_commit"]} for s in sources])
    return manifest


def load(batch):
    batch = Path(batch).resolve()
    m = dc.read(batch / "protocol/manifest.json")
    tc.code_check(m["implementation_commit"])
    dc.require(m["schema"] == "training100_selected_eval_v1" and m["batch_root"] == str(batch) and
               batch.parent == Path(m["study_root"]).resolve() / ROOT and m["worlds"] == WORLDS and
               m["world_role"] == WORLD_ROLE and m["policy_world_repeat_count"] == 300, "evaluation batch identity changed")
    sources = dc.read(batch / "protocol/sources.json")
    frozen_selection = select(dc.read(batch / "protocol/candidates.json"))
    dc.require(len(sources) == 8 and all(all(s[k] == v for k, v in frozen.items())
               for s, frozen in zip(sources, frozen_selection)), "source selection changed")
    units = dc.read(batch / "protocol/units.json")
    dc.require(units == make_units(sources) and len(units) == 300, "fixed matrix changed")
    return m, sources, units


def preflight(batch, device="cuda:0"):
    import torch
    m, sources, _ = load(batch)
    dc.require(candidates(m["training_batch"]) == dc.read(Path(batch) / "protocol/candidates.json"), "training candidate evidence changed")
    for source in sources:
        source_check(source, payload=True)
    dc.require(device in {"cpu", "cuda:0"}, "use cpu or cuda:0")
    if device == "cuda:0":
        dc.require(torch.cuda.is_available(), "CUDA unavailable in selected interpreter")
        x = torch.ones((128, 128), device=device)
        dc.require(float((x @ x)[0, 0]) == 128, "CUDA probe failed")
        torch.cuda.synchronize()
    report = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"],
              "loaded_selected_payloads": 8, "candidate_runs": 24, "policy_world_repeat_count": 300,
              "device": device, "python": sys.version, "executable": sys.executable,
              "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
              "new_training": 0, "real_rollouts_in_preflight": 0}
    path = Path(batch) / "protocol/preflight.json"
    if path.exists():
        dc.require(dc.read(path) == report, "preflight environment changed")
    else:
        dc.write(path, report)
    return report


def native_check(metadata, unit, source, commit):
    record = source["record"]
    expected = {"status": "complete", "training_seed": unit["training_seed"], "eval_seeds": [unit["world"]],
                "environment_seeds": [unit["world"]], "action_repeat": 0, "action_seed_words": [unit["action_seed_words"]],
                "eval_protocol": "sequential_warm", "eval_warmup_episodes": 5, "eval_episodes": 100,
                "target_scenario": dc.SCENARIO, "source_scenario": dc.SCENARIO,
                "source_config_hash": source["config_hash"], "training_source_commit": record["implementation_commit"],
                "reproduction_git_commit": commit, "reproduction_git_dirty": False, "world_role": WORLD_ROLE,
                "checkpoint_role": ROLES[source["checkpoint_kind"]], "policy_version": source["policy_version"],
                "training_budget_episodes": 2500}
    if dc.MODELS[unit["model_id"]]["family"] == "mappo":
        expected.update(policy_episode=source["policy_episode"], policy_parameters_unchanged=True,
                        mappo_eval_mode="stochastic", actor_sharing=dc.MODELS[unit["model_id"]]["structure"] == "shared",
                        mappo_variant=dc.MODELS[unit["model_id"]]["variant"],
                        policy_sampling_seeds=[policy_seed(unit["training_seed"], unit["world"], 0)])
    else:
        expected.update(checkpoint_episode=source["policy_episode"], parameter_snapshot_unchanged=True,
                        optimizer_steps=0, normalization_updated=False, noise_std=unit["noise_std"],
                        checkpoint_name=Path(source["checkpoint"]["path"]).name)
    for key, value in expected.items():
        dc.require(metadata.get(key) == value, f"native {key}: {metadata.get(key)!r} != {value!r}")
    dc.require(metadata.get("raw_metric_axes", {}).get("aoi_ms") == ["eval_seed", "scored_episode", "slot", "agent"], "native axes")
    dc.require(len(metadata.get("exogenous_probes", [])) == 1 and
               metadata["exogenous_probes"][0]["world"] == unit["world"], "missing exogenous probe")


def accept(batch, unit, source, manifest, attempt, create=False):
    dest = Path(batch) / "units" / f"unit_{unit['unit_id']:03d}" / attempt
    started = dc.read(dest / "STARTED.json")
    dc.require(started["unit"] == unit and started["source"] == source and
               started["implementation_commit"] == manifest["implementation_commit"], "started identity changed")
    markers = list(dest.rglob("EVAL_COMPLETE.json"))
    dc.require(len(markers) == 1, "expected one native evaluation completion")
    metadata = dc.read(markers[0])
    native_check(metadata, unit, source, manifest["implementation_commit"])
    provenance = dc.read(markers[0].with_name("provenance.json"))
    dc.require(provenance["reproduction_git_commit"] == manifest["implementation_commit"], "native provenance commit")
    policy_path = ((markers[0].parent / metadata["policy"]).resolve() if "policy" in metadata else
                   Path(metadata["checkpoint"]).resolve())
    dc.require(str(policy_path) == source["checkpoint"]["path"], "native policy path changed")
    identity = {**unit, "source_checkpoint": str(policy_path), "policy_episode": source["policy_episode"],
                "policy_version": source["policy_version"], "training_environment_steps": source["policy_episode"] * 100,
                "training_budget_environment_steps": 250000,
                "training_commit": source["record"]["implementation_commit"],
                "implementation_commit": manifest["implementation_commit"], "data_role": "new"}
    identity.pop("action_seed_words")
    rows = reduce_arrays(load_arrays(markers[0].with_name("metrics.npz")), metadata, identity)
    record = {"technical_status": "PASS", "unit": unit, "source": source, "attempt": attempt,
              "implementation_commit": manifest["implementation_commit"], "row_count": 5,
              "metrics": dc.file_identity(markers[0].with_name("metrics.npz")), "native_completion": str(markers[0])}
    out = Path(batch) / f"records/unit_{unit['unit_id']:03d}.json"
    table = dest / "world_repeat_platoon.csv"
    if create:
        if table.exists():
            _check_csv(table, rows)
        else:
            csv_write(table, rows)
        dc.write(out, record)
    else:
        dc.require(dc.read(out) == record, "accepted identity changed")
        _check_csv(table, rows)
    return rows, metadata["exogenous_probes"][0]


def execute(batch, unit_id, device="cuda:0", attempt="a01", reason=None):
    m, sources, units = load(batch)
    dc.require(0 <= unit_id < 300 and re.fullmatch(r"a\d{2,4}", attempt), "invalid unit/attempt")
    pre = dc.read(Path(batch) / "protocol/preflight.json")
    dc.require(pre["technical_status"] == "PASS" and pre["implementation_commit"] == m["implementation_commit"] and
               pre["device"] == device and pre["executable"] == sys.executable, "run preflight with same interpreter/device")
    unit = units[unit_id]
    source = next(s for s in sources if s["source_id"] == unit["source_id"])
    source_check(source)
    accepted = Path(batch) / f"records/unit_{unit_id:03d}.json"
    if accepted.exists():
        accept(batch, unit, source, m, dc.read(accepted)["attempt"])
        return {"unit_id": unit_id, "skipped_verified": True}
    parent = Path(batch) / "units" / f"unit_{unit_id:03d}"
    dest = parent / attempt
    if dest.exists():
        # Completed rollout + failed post-check: inspect arrays; never run it again.
        dc.require(len(list(dest.rglob("EVAL_COMPLETE.json"))) == 1,
                   "partial attempt preserved; retry only failed IDs with new attempt and reason")
        accept(batch, unit, source, m, attempt, create=True)
        return {"unit_id": unit_id, "accepted_existing_native": True}
    dc.require(not list(parent.rglob("EVAL_COMPLETE.json")), "native completion exists in another attempt; accept its arrays first")
    dc.require(not parent.exists() or bool(reason), "retry requires explicit reason")
    dest.mkdir(parents=True, exist_ok=False)
    dc.write(dest / "STARTED.json", {"unit": unit, "source": source, "implementation_commit": m["implementation_commit"],
                                    "pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat(), "reason": reason})
    policy = Path(source["checkpoint"]["path"])
    if dc.MODELS[unit["model_id"]]["family"] == "mappo":
        from analysis.evaluate_mappo_zero_shot import evaluate_zero_shot
        evaluate_zero_shot(policy, dest, dc.SCENARIO, device, [unit["world"]], 100,
                           deployment=True, world_role=WORLD_ROLE, training_policy=True)
    else:
        from analysis.evaluate_maddpg_e1 import evaluate_maddpg_e1
        run = policy.parent
        override = {"run_dir": run, "resolved_run_dir": run, "checkpoint": policy,
                    "canonical_run_name": run.name, "historical_run_name": run.name, "algorithm_label": "algorithm2",
                    "is_symlink": run.is_symlink(), "complete": dc.read(run / "COMPLETE.json"),
                    "manifest": {"selected_training100_source": source["source_id"]}}
        evaluate_maddpg_e1("modified_maddpg_tdec", unit["noise_std"], unit["training_seed"], Path(m["study_root"]).parent,
                          dest, device, [unit["world"]], 100, 5, source["policy_episode"],
                          target_scenario=dc.SCENARIO, source_override=override, deployment=True,
                          world_role=WORLD_ROLE, training_policy=True)
    source_check(source)
    tc.code_check(m["implementation_commit"])
    accept(batch, unit, source, m, attempt, create=True)
    return {"unit_id": unit_id, "technical_status": "PASS"}


def summarize(batch, attempt="a01"):
    from analysis.summarize_frozen_deployment import aggregate
    m, sources, units = load(batch)
    dc.require(re.fullmatch(r"a\d{2,4}", attempt), "invalid analysis attempt")
    for source in sources:
        source_check(source)
    rows, probes, native_records = [], {}, []
    for unit in units:
        record = dc.read(Path(batch) / f"records/unit_{unit['unit_id']:03d}.json")
        source = next(s for s in sources if s["source_id"] == unit["source_id"])
        part, probe = accept(batch, unit, source, m, record["attempt"])
        rows.extend(part)
        if "native_completion" in record:
            completion = Path(record["native_completion"])
            native_records.extend((completion, completion.with_name("provenance.json")))
        if unit["world"] in probes:
            dc.require(probe == probes[unit["world"]], "exogenous world process differs between selected policies")
        probes[unit["world"]] = probe
    dc.require(len(rows) == 1500 and sum(r["slot_count"] for r in rows) == 15000000 and
               sum(r["cam_episode_count"] for r in rows) == 150000, "evaluation coverage/count mismatch")
    tables = aggregate(rows)  # Distinct condition suffix prevents pooling best100 and final.
    dc.require(len(tables["seed_endpoints"]) == 10 and all(r["world_count"] == 30 for r in tables["seed_endpoints"]),
               "each selected condition requires 30 worlds")
    dc.require(all(r["n_training_seeds"] == 1 and r["descriptive_t95_lower"] is None
                   for r in tables["condition_summary"]), "selected seeds must not masquerade as six independent repeats")
    out = Path(batch) / "analysis" / attempt
    out.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        csv_write(out / f"{name}.csv", table)
    verification = {"technical_status": "PASS", "claim_support": "NOT_ASSESSED", "accepted_units": 300,
                    "new_units": 300, "reused_units": 0, "platoon_rows": len(rows), "scored_agent_slots": 15000000,
                    "cam_deadlines": 150000, "worlds": WORLDS, "exogenous_probe_match": True,
                    "selected_training_seeds_per_condition": 1, "action_repeats": 1,
                    "training_seed_uncertainty": None, "repeat_variability": None,
                    "implementation_commit": m["implementation_commit"], "new_training": 0}
    dc.write(out / "verification.json", verification)
    (out / "REPORT_CN.md").write_text(
        "# 选中模型的探索性冻结评估\n\n300单元、1500逐车队行；technical PASS与性能方向无关。\n"
        "每种配置从六seed按训练回报分别选best100/final，final依据最后100集训练回报；并非测试集择优。\n"
        "这不是六seed平均算法比较，也不是best与final的同seed配对因果比较。\n"
        "world1001–1030由用户固定，历史未使用性未认证，不称新的未见确认。\n"
        "每world冷启动一次，5 warmup+100 scored×100 slots；仅repeat0。MADDPG噪声0/0.3共享相同模型。\n"
        "scene-first先逐world/车队归约再取该world最弱车队，legacy先跨world按车队合并再取最弱；AoI/CAM分别取极值。\n"
        "输出保留sum/count、严格AoI>50、CAM末槽deadline成功、线性mW功率、reward与可信V2I事件。\n"
        "world SD只是选定模型在这30world的描述；训练seed SD、执行repeat SD及训练seed置信区间为空。\n"
        "模型/episode/version见protocol/selected_models.csv。缺失量不填零；原始NPZ/权重留执行机。\n",
        encoding="utf-8")
    archive = out / "core_evidence.zip"
    files = [p for root in (out, Path(batch) / "protocol", Path(batch) / "records") for p in root.rglob("*")
             if p.is_file() and p.suffix in {".csv", ".json", ".md"}]
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(batch).as_posix())
        for path in native_records:
            z.write(path, path.relative_to(batch).as_posix())
        for source in sources:
            for identity in source["metadata_files"]:
                path = Path(identity["path"])
                if path.suffix == ".json":
                    z.write(path, f"selected_source_metadata/{source['source_id']}/{path.name}")
    with zipfile.ZipFile(archive) as z:
        dc.require(z.testzip() is None, "ZIP integrity failed")
    return {**verification, "archive": str(archive)}


def run(batch, device="cuda:0", workers=2, ids=None, attempt="a01", reason=None):
    """Bounded independent Python children; no sbatch, no in-process global RNG races."""
    dc.require(workers in (1, 2), "one or two concurrent evaluation processes only")
    m, _, _ = load(batch)
    ids = list(range(300)) if ids is None else ids
    dc.require(ids and len(ids) == len(set(ids)) and all(0 <= i < 300 for i in ids), "invalid/duplicate unit IDs")
    preflight(batch, device)
    lock = Path(batch) / "RUNNING.json"
    dc.write(lock, {"pid": os.getpid(), "implementation_commit": m["implementation_commit"]})
    active, failed, pending = [], [], iter(ids)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               TMP=str(Path(batch) / "tmp"), TEMP=str(Path(batch) / "tmp"), TMPDIR=str(Path(batch) / "tmp"))
    try:
        exhausted = False
        while active or not exhausted:
            while not failed and not exhausted and len(active) < workers:
                try:
                    i = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                if (Path(batch) / f"records/unit_{i:03d}.json").exists():
                    execute(batch, i, device, attempt, reason)
                    continue
                stdout = (Path(batch) / f"logs/unit_{i:03d}_{attempt}.out").open("x")
                stderr = (Path(batch) / f"logs/unit_{i:03d}_{attempt}.err").open("x")
                cmd = [sys.executable, "-B", "-u", "-m", __name__ if __name__ != "__main__" else
                       "analysis.evaluate_training100_selected", "unit", "--batch-root", str(batch),
                       "--id", str(i), "--device", device, "--attempt", attempt]
                if reason:
                    cmd += ["--reason", reason]
                start = time.time()
                process = subprocess.Popen(cmd, cwd=dc.REPO, env=env, stdout=stdout, stderr=stderr)
                active.append((i, process, stdout, stderr, start, cmd))
            for task in list(active):
                i, process, stdout, stderr, start, cmd = task
                code = process.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                dc.write(Path(batch) / f"records/process_{i:03d}_{attempt}.json",
                         {"unit_id": i, "pid": process.pid, "command": cmd, "exit_code": code,
                          "elapsed_seconds": time.time() - start, "started_unix": start,
                          "MaxRSS": None, "implementation_commit": m["implementation_commit"]})
                active.remove(task)
                if code != 0 or not (Path(batch) / f"records/unit_{i:03d}.json").exists():
                    failed.append(i)
                print(json.dumps({"unit_id": i, "exit_code": code, "remaining_active": len(active)}), flush=True)
            if failed:
                exhausted = True  # Let already running evaluations finish; schedule nothing else.
            if active:
                time.sleep(1)
        dc.require(not failed, f"failed units {failed}; preserved, do not change seed/model/config")
    finally:
        # On interruption never leave an untracked child using a supposedly free batch.
        for _, process, stdout, stderr, _, _ in active:
            if process.poll() is None:
                process.terminate()
                process.wait()
            stdout.close()
            stderr.close()
        lock.unlink()
    if all((Path(batch) / f"records/unit_{i:03d}.json").exists() for i in range(300)):
        return summarize(batch, attempt)
    return {"technical_status": "PARTIAL", "submitted_ids": ids}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preview", help="CSV/metadata-only dry run; no weights, output or rollout")
    p.add_argument("--training-batch", type=Path, required=True)
    p = sub.add_parser("init")
    p.add_argument("--training-batch", type=Path, required=True)
    p.add_argument("--batch-id", required=True)
    p.add_argument("--expected-commit", required=True)
    for name in ("preflight", "run", "unit", "summary"):
        p = sub.add_parser(name)
        p.add_argument("--batch-root", type=Path, required=True)
        if name != "summary":
            p.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
        if name in {"run", "unit", "summary"}:
            p.add_argument("--attempt", default="a01")
        if name in {"run", "unit"}:
            p.add_argument("--reason")
        if name == "run":
            p.add_argument("--workers", type=int, default=2, choices=(1, 2))
            p.add_argument("--ids", help="explicit failed/unrun IDs, comma separated")
        if name == "unit":
            p.add_argument("--id", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "preview":
        sources = select(candidates(args.training_batch))
        result = {"units": len(make_units(sources)), "worlds": WORLDS, "new_training": 0,
                  "selected": [{"model": s["record"]["model_id"], "kind": s["checkpoint_kind"],
                                "seed": s["record"]["training_seed"], "score": s["selection_score"]} for s in sources]}
    elif args.command == "init":
        result = init(args.training_batch, args.batch_id, args.expected_commit)
    elif args.command == "preflight":
        result = preflight(args.batch_root, args.device)
    elif args.command == "unit":
        result = execute(args.batch_root, args.id, args.device, args.attempt, args.reason)
    elif args.command == "summary":
        result = summarize(args.batch_root, args.attempt)
    else:
        result = run(args.batch_root, args.device, args.workers,
                     [int(x) for x in args.ids.split(",")] if args.ids else None, args.attempt, args.reason)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
