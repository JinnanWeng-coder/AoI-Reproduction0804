"""Uniform Random frozen lifecycle, two action streams x thirty exploratory worlds; CPU only."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import tempfile
import time
import zipfile

import numpy as np

from analysis import deployment_contract as dc, run_random_training_baseline as random
from analysis.deployment_rng import exogenous_probe, random_action_seed_words
from analysis.evaluate_frozen_deployment import csv_read, csv_write, load_arrays, reduce_arrays
from analysis.frozen_rollout import sequential_warm, arrays_from_records
from analysis.summarize_frozen_deployment import pooled, endpoints, mean, sd
from aoi_v2x_reproduction.config import config_from_dict
from aoi_v2x_reproduction.envs.platoon import PaperEnviron

CLI_MODULE = "analysis.evaluate_random_frozen"
BASELINE = "46f3b3554ccd66779a8bcd1272cba3013c7b5372"
ROOT = Path("Random-frozen-eval/P5_N4_gap25")
BATCH_ID = "random_uniform_stream12_13_world1001_1030_20261021_v1"
SCHEMA = "uniform_random_frozen_v1"
STREAMS = [12, 13]
WORLDS = list(range(1001, 1031))
WORLD_ROLE = "user_fixed_exploratory_worlds"
PROTOCOL = {"eval_protocol": "sequential_warm", "eval_warmup_episodes": 5, "eval_episodes": 100,
            "steps_per_episode": 100, "number_agents": 5, "n_rb": 3, "target_scenario": "p05_n04_g25"}
RNG_RULE = {"bit_generator": "PCG64", "seed_sequence": "[action_stream_id, world, 0x52414E44, 1]",
            "version": 1, "draw_order_per_slot": ["rb[P]", "mode[P]", "power_fraction[P]"],
            "warmup_consumes_same_action_stream": True, "physical_rng": "PaperEnviron.reset_world(world) private generator",
            "action_stream_ids_are_training_seeds": False, "action_repeats": None}


def code_check(expected=None):
    sha = dc.git("rev-parse", "HEAD")
    dc.require(dc.git("branch", "--show-current") in ("research/mappo", "") and
               not dc.git("status", "--porcelain", "--untracked-files=all"), "clean pinned research checkout required")
    dc.require(expected is None or sha == expected, "implementation commit changed")
    dc.require(not dc.git("diff", BASELINE, "HEAD", "--", "aoi_v2x_reproduction", random.BUNDLE,
                          "analysis/run_random_training_baseline.py"), "environment/reward/source bundle/Random sampler changed")
    return sha


def units():
    return [{"unit_id": i, "model_id": "uniform_random", "execution": "uniform_random",
             "condition": "uniform_random", "action_stream_id": stream, "world": world,
             "target_scenario": dc.SCENARIO, "training_seed": None, "training_model_count": 0,
             "unit_key": ["uniform_random", dc.SCENARIO, world, stream],
             "action_seed_words": random_action_seed_words(stream, world)}
            for i, (stream, world) in enumerate((s, w) for s in STREAMS for w in WORLDS)]


def resolved_config(bundle):
    cfg = random.derive(bundle, 0, 100)
    # seed0 is an unused template placeholder: all actual worlds are explicitly reset_world(world).
    cfg.run_name = "uniform_random_frozen_no_training"
    cfg.output_root = "unused_no_training_or_checkpoint_output"
    dc.require(cfg.device == "cpu" and cfg.checkpoint_mode == "none" and cfg.number_agents == 5 and
               cfg.steps_per_episode == 100 and cfg.global_actor_weight == 1 and cfg.eval_protocol == "sequential_warm" and
               cfg.eval_warmup_episodes == 5, "frozen environment/protocol mismatch")
    return cfg


def preview():
    bundle, environment = random.source_bundle()
    cfg = resolved_config(bundle)
    return {"schema": SCHEMA, "units": units(), "unit_count": 60, "action_stream_ids": STREAMS,
            "worlds": WORLDS, "protocol": PROTOCOL, "action_rule": random.ACTION_RULE, "rng_rule": RNG_RULE,
            "world_role": WORLD_ROLE, "new_unseen_confirmation": False, "training_models": 0, "training_runs": 0,
            "platoon_rows": 300, "scored_agent_slots": 3000000, "cam_deadlines": 30000,
            "warmup_agent_slots_excluded": 150000, "resolved_config": cfg.to_dict(), "environment_config": environment}


def init(study_root, expected_commit, batch_id=BATCH_ID):
    sha = code_check(expected_commit)
    dc.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id), "unsafe batch ID")
    study = Path(study_root).resolve()
    batch = study / ROOT / batch_id
    dc.require(not batch.exists(), "unique new output root required; do not re-init existing batch")
    p = preview()
    bundle, _ = random.source_bundle()
    m = {k: v for k, v in p.items() if k not in ("units", "resolved_config")}
    m.update(batch_root=str(batch), study_root=str(study), implementation_commit=sha,
             environment_baseline_commit=BASELINE, source_bundle=random.BUNDLE,
             source_bundle_origin="configuration-only recorded D1/Training100 environment; no old arrays or weights",
             policy_selection=None, network_constructed=False, optimizer_constructed=False)
    batch.mkdir(parents=True, exist_ok=False)
    for name in ("protocol", "records", "units", "logs", "tmp"):
        (batch / name).mkdir()
    for name, value in (("manifest", m), ("units", p["units"]), ("source_bundle", bundle), ("config.resolved", p["resolved_config"])):
        dc.write(batch / f"protocol/{name}.json", value)
    return m


def load(batch):
    batch = Path(batch).resolve()
    m = dc.read(batch / "protocol/manifest.json")
    code_check(m["implementation_commit"])
    p = preview()
    for k, value in p.items():
        if k not in ("units", "resolved_config"):
            dc.require(m[k] == value, f"Random frozen manifest drift: {k}")
    dc.require(m["batch_root"] == str(batch) and batch.parent == Path(m["study_root"]) / ROOT and
               m["policy_selection"] is None and m["network_constructed"] is False and m["optimizer_constructed"] is False,
               "batch root/nonlearning identity changed")
    matrix = dc.read(batch / "protocol/units.json")
    bundle, _ = random.source_bundle()
    dc.require(matrix == p["units"] and dc.read(batch / "protocol/config.resolved.json") == p["resolved_config"] and
               dc.read(batch / "protocol/source_bundle.json") == bundle, "matrix/resolved environment/sampler source changed")
    return m, matrix


def preflight(batch, device="cpu"):
    m, matrix = load(batch)
    dc.require(device == "cpu" and sys.version_info[:2] == (3, 10) and np.__version__ == "1.23.5",
               "use established Python3.10/NumPy1.23.5 CPU environment; do not upgrade")
    cfg = config_from_dict(dc.read(Path(batch) / "protocol/config.resolved.json"))
    env = PaperEnviron(cfg)  # Decoder only, never reset/roll out a world in preflight.
    rng = np.random.default_rng(np.random.SeedSequence(matrix[0]["action_seed_words"]))
    actions, decoded = random.draw_actions(cfg, rng)
    dc.require(np.allclose(env.decode_actions(actions), decoded, rtol=0, atol=1e-12), "Random/native decoder mismatch")
    result = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"], "device": "cpu",
              "executable": sys.executable, "python": sys.version, "numpy": np.__version__,
              "runtime_package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "PyYAML")},
              "new_training": 0, "checkpoint_loads": 0, "environment_rollouts": 0, "expected_units": 60,
              "network_constructed": False, "optimizer_constructed": False}
    path = Path(batch) / "protocol/preflight.json"
    if path.exists(): dc.require(dc.read(path) == result, "preflight interpreter/runtime changed")
    else: dc.write(path, result)
    return result


def sample_rollout(cfg, stream, world, scored_episodes=100, warmup_episodes=5):
    """Same lifecycle as MADDPG evaluation, exact existing Random sampler; private RNG only."""
    env = PaperEnviron(cfg)
    sampled = {"action_count": 0, "environment_step_count": 0, "decoder_checks": 0, "physical_rng_unchanged_by_actions": True}
    boundary = {}
    def factory(w):
        rng = np.random.default_rng(np.random.SeedSequence(random_action_seed_words(stream, w)))
        def act(_observations):
            before = copy.deepcopy(env.rng.bit_generator.state)
            actions, decoded = random.draw_actions(cfg, rng)  # Original draw order, continuous uniform dBm.
            dc.require(env.rng.bit_generator.state == before, "action draw consumed physical RNG")
            dc.require(np.allclose(env.decode_actions(actions), decoded, rtol=0, atol=1e-12), "executed decoder differs from sampled action")
            sampled["environment_step_count"] += 1
            sampled["action_count"] += cfg.number_agents
            sampled["decoder_checks"] += 1
            return actions  # Preserve original float64 encoded actions; do not clip at neural actor +/-0.999.
        return act
    records, probes = sequential_warm(env, [world], factory, steps_per_episode=cfg.steps_per_episode,
        scored_episodes=scored_episodes, warmup_episodes=warmup_episodes, deployment=True, probe=exogenous_probe,
        after_warmup=lambda w, e: boundary.update(warmup_end_aoi=e.aoi.copy().tolist()))
    arrays = arrays_from_records(records, deployment=True)
    metadata = {"event_source": "post_step_aoi_ms_isclose_1_verified_env_step", "v2i_min": float(env.v2i_min),
        "global_actor_weight": float(cfg.global_actor_weight), "cam_bits": float(cfg.cam_bits), "eval_seeds": [world],
        "eval_episodes": scored_episodes, "eval_warmup_episodes": warmup_episodes, "steps_per_episode": cfg.steps_per_episode,
        "number_agents": cfg.number_agents, "n_rb": cfg.n_rb, "target_scenario": dc.SCENARIO,
        "eval_protocol": "sequential_warm", "exogenous_probes": probes, "sampling_checks": sampled, **boundary}
    dc.require(np.array_equal(arrays["success"][:, :, -1, :].astype(bool), arrays["remaining_demand"][:, :, -1, :] == 0),
               "deadline CAM/remaining demand mismatch")
    return arrays, metadata


def reduce_unit(arrays, metadata, unit, commit, scored_episodes=100):
    identity = {k: v for k, v in unit.items() if k not in ("unit_key", "action_seed_words")}
    identity.update(implementation_commit=commit, data_role="new", training_seed=None,
                    training_model_count=0, world_role=WORLD_ROLE)
    rows = reduce_arrays(arrays, metadata, identity, expected=(1, scored_episodes, 100, 5))
    dc.require(np.all((arrays["power_dbm"] >= 1) & (arrays["power_dbm"] <= 30)), "decoded power outside1..30dBm")
    if "warmup_end_aoi" in metadata:
        old = np.asarray(metadata["warmup_end_aoi"], dtype=np.float64)
        first = np.asarray(arrays["aoi_ms"])[0, 0, 0]
        dc.require(old.shape == (5,) and np.array_equal(first, np.where(first == 1, 1, np.minimum(100, old + 1))),
                   "warmup/scored AoI continuity mismatch")
    for p, row in enumerate(rows):
        mode = arrays["mode"][0, ..., p]
        row.update(mode_v2i_slot_count=int((mode == 0).sum()), mode_v2v_slot_count=int((mode == 1).sum()))
        for rb in range(3): row[f"rb{rb}_slot_count"] = int((arrays["rb"][0, ..., p] == rb).sum())
    return rows


def _same_csv(path, rows):
    dc.require(csv_read(path) == [{k: "" if v is None else str(v) for k, v in r.items()} for r in rows],
               f"accepted CSV changed: {path}")


def accept(batch, unit, manifest, attempt, create=False):
    dest = Path(batch) / f"units/unit_{unit['unit_id']:03d}/{attempt}"
    started = dc.read(dest / "STARTED.json")
    dc.require(started["unit"] == unit and started["implementation_commit"] == manifest["implementation_commit"], "started unit identity changed")
    meta = dc.read(dest / "EVAL_COMPLETE.json")
    provenance = dc.read(dest / "provenance.json")
    dc.require(provenance["implementation_commit"] == manifest["implementation_commit"] and provenance["unit"] == unit and
               provenance["runtime_config"] == dc.read(Path(batch) / "protocol/config.resolved.json"), "native provenance changed")
    for key, value in {**PROTOCOL, "status": "complete", "unit": unit, "action_rng_rule": RNG_RULE,
            "action_rule": random.ACTION_RULE, "runtime_config": dc.read(Path(batch) / "protocol/config.resolved.json"),
            "implementation_commit": manifest["implementation_commit"], "world_role": WORLD_ROLE,
            "new_unseen_confirmation": False, "eval_seeds": [unit["world"]], "network_constructed": False,
            "optimizer_constructed": False, "checkpoint_loads": 0, "training_updates": 0}.items():
        dc.require(meta.get(key) == value, f"native Random {key} mismatch")
    sample = meta["sampling_checks"]
    dc.require("warmup_end_aoi" in meta, "missing native warmup/scored boundary")
    dc.require(len(meta["exogenous_probes"]) == 1 and meta["exogenous_probes"][0]["world"] == unit["world"] and
               {"initial", "final"} <= set(meta["exogenous_probes"][0]), "physical probe world identity mismatch")
    dc.require(sample == {"action_count": 52500, "environment_step_count": 10500, "decoder_checks": 10500,
                         "physical_rng_unchanged_by_actions": True}, "warmup/scored action sampling counts")
    arrays = load_arrays(dest / "metrics.npz")
    # load_arrays intentionally loads only reducer fields; the native Random writer also needs RB coverage.
    with np.load(dest / "metrics.npz", allow_pickle=False) as z:
        arrays["rb"] = z["rb"]
    dc.require(arrays["rb"].shape == (1, 100, 100, 5) and np.isin(arrays["rb"], [0, 1, 2]).all(), "invalid scored RB array")
    dc.require(np.array_equal(arrays["success"][:, :, -1, :].astype(bool), arrays["remaining_demand"][:, :, -1, :] == 0),
               "CAM deadline not payload completion")
    rows = reduce_unit(arrays, meta, unit, manifest["implementation_commit"])
    record = {"technical_status": "PASS", "unit": unit, "attempt": attempt, "row_count": 5,
              "implementation_commit": manifest["implementation_commit"], "metrics": dc.file_identity(dest / "metrics.npz"),
              "native_completion": str((dest / "EVAL_COMPLETE.json").resolve()),
              "completion_source": dc.file_identity(dest / "EVAL_COMPLETE.json"),
              "provenance_source": dc.file_identity(dest / "provenance.json"),
              "scored_agent_slots": 50000, "warmup_agent_slots_excluded": 2500, "cam_deadlines": 500}
    table = dest / "world_stream_platoon.csv"
    path = Path(batch) / f"records/unit_{unit['unit_id']:03d}.json"
    if create:
        if table.exists(): _same_csv(table, rows)
        else: csv_write(table, rows)
        dc.write(path, record)
    else:
        dc.require(dc.read(path) == record, "accepted full-key/source identity changed")
        _same_csv(table, rows)
    return rows, meta["exogenous_probes"][0]


def execute(batch, unit_id, device="cpu", attempt="a01", reason=None):
    m, matrix = load(batch)
    dc.require(device == "cpu" and type(unit_id) is int and 0 <= unit_id < 60 and re.fullmatch(r"a\d{2,4}", attempt),
               "invalid device/unit ID/attempt")
    pre = dc.read(Path(batch) / "protocol/preflight.json")
    dc.require(pre["technical_status"] == "PASS" and pre["implementation_commit"] == m["implementation_commit"] and
               pre["executable"] == sys.executable and pre["device"] == device, "same-interpreter preflight required")
    smoke_report = dc.read(Path(batch) / "protocol/smoke.json")
    dc.require(smoke_report["technical_status"] == "PASS" and smoke_report["implementation_commit"] == m["implementation_commit"],
               "short technical smoke PASS required")
    unit = matrix[unit_id]
    record = Path(batch) / f"records/unit_{unit_id:03d}.json"
    if record.exists():
        accept(batch, unit, m, dc.read(record)["attempt"])
        return {"unit_id": unit_id, "skipped_verified": True}
    parent = Path(batch) / f"units/unit_{unit_id:03d}"
    dest = parent / attempt
    if dest.exists():
        dc.require((dest / "EVAL_COMPLETE.json").is_file(), "partial attempt retained; use new attempt with explicit reason")
        accept(batch, unit, m, attempt, create=True)
        return {"unit_id": unit_id, "accepted_existing_native": True}
    dc.require(not list(parent.glob("a*/EVAL_COMPLETE.json")), "native completion exists; accept its arrays in ORIGINAL attempt first")
    dc.require(attempt == "a01" or reason, "new attempt requires reason")
    dc.require(not parent.exists() or reason, "failed attempt preserved; explicit retry reason required")
    dest.mkdir(parents=True, exist_ok=False)
    dc.write(dest / "STARTED.json", {"unit": unit, "implementation_commit": m["implementation_commit"], "reason": reason})
    cfg = config_from_dict(dc.read(Path(batch) / "protocol/config.resolved.json"))
    started = time.perf_counter()
    arrays, meta = sample_rollout(cfg, unit["action_stream_id"], unit["world"])
    rows = reduce_unit(arrays, meta, unit, m["implementation_commit"])
    code_check(m["implementation_commit"])
    np.savez_compressed(dest / "metrics.npz", **arrays)
    meta.update(status="complete", unit=unit, runtime_config=cfg.to_dict(), action_rule=random.ACTION_RULE,
        action_rng_rule=RNG_RULE, world_role=WORLD_ROLE, new_unseen_confirmation=False, network_constructed=False,
        optimizer_constructed=False, checkpoint_loads=0, training_updates=0,
        implementation_commit=m["implementation_commit"], elapsed_seconds=time.perf_counter() - started,
        global_reward_axes=["world", "scored_episode", "slot"], scored_array_shape=[1, 100, 100, 5])
    dc.require(len(rows) == 5, "one unit must produce five platoon rows")
    dc.write(dest / "provenance.json", {"implementation_commit": m["implementation_commit"], "unit": unit,
        "sampler": "analysis.run_random_training_baseline.draw_actions", "lifecycle": "analysis.frozen_rollout.sequential_warm",
        "physical_rng_probe": "analysis.deployment_rng.exogenous_probe", "python_executable": sys.executable,
        "runtime_config": cfg.to_dict(), "raw_arrays_stay_local": True})
    dc.write(dest / "EVAL_COMPLETE.json", meta)  # Publish only after all native arrays are written/checked.
    accept(batch, unit, m, attempt, create=True)
    return {"unit_id": unit_id, "technical_status": "PASS"}


METRICS = {"mean_aoi_ms": "mean_aoi_ms", "mean_cam_percent": "mean_cam_success",
    "mean_power_mw": "mean_power_mw", "mean_reward": "mean_reward_combined", "mean_global_reward": "mean_reward_global",
    "mean_task1_reward": "mean_reward_task1", "mean_task2_reward": "mean_reward_task2",
    "scene_first_worst_aoi_ms": "worst_aoi_ms", "scene_first_lowest_cam_percent": "worst_cam_success",
    "aoi_gt50_fraction": "mean_aoi_gt50", "aoi_cap_fraction": "mean_aoi_at_cap",
    "v2i_attempt_rate": "mean_v2i_attempt_rate", "v2i_reset_rate": "mean_v2i_reset_rate"}


def world_endpoint(rows, stream):
    dc.require(len(rows) == 5 and {r["platoon"] for r in rows} == set(range(5)), "world missing/duplicate platoons")
    ep = endpoints(rows)
    # Global stream is common to all platoons. Count it ONCE in joint/unit tables.
    global_sum = rows[0]["reward_global_sum"]
    global_count = rows[0]["reward_slot_count"]
    dc.require(all(r["reward_global_sum"] == global_sum and r["reward_slot_count"] == global_count for r in rows),
               "platoon copies of joint global reward disagree")
    return {"world": rows[0]["world"], "action_stream_id": stream, "stream_count": 1 if stream is not None else 2,
        "unit_id": rows[0].get("unit_id") if stream is not None else None,
        "condition": "uniform_random", "model_id": "uniform_random", "target_scenario": dc.SCENARIO,
        "training_seed": None, "training_model_count": 0,
        "joint_global_reward_sum": global_sum, "joint_global_reward_slot_count": global_count,
        "scored_agent_slots": sum(r["slot_count"] for r in rows), "cam_deadlines": sum(r["cam_episode_count"] for r in rows),
        **{label: (100 * ep[key] if label.endswith("_percent") and ep[key] is not None else ep[key]) for label, key in METRICS.items()}}


def summarize_worlds(worlds, stream, scope):
    dc.require(len(worlds) == 30 and {r["world"] for r in worlds} == set(WORLDS), "thirty distinct worlds required")
    return [{"action_stream_id": stream, "scope": scope, "metric": metric,
             "condition": "uniform_random", "model_id": "uniform_random", "target_scenario": dc.SCENARIO,
             "mean": mean(r[metric] for r in worlds), "world_sample_sd": sd(r[metric] for r in worlds),
             "world_count": 30, "sample_sd_ddof": 1, "n_training_seeds": 0, "training_seed_sd": None,
             "confidence_interval": None, "stream_count": 1 if stream is not None else 2}
            for metric in METRICS]


def tables_from_rows(rows):
    dc.require(len(rows) == 300 and len({(r["action_stream_id"], r["world"], r["platoon"]) for r in rows}) == 300 and
               {(r["action_stream_id"], r["world"], r["platoon"]) for r in rows} ==
               {(s, w, p) for s in STREAMS for w in WORLDS for p in range(5)}, "full stream/world/platoon coverage required")
    worlds, summaries, legacy, combined_p, combined_w = [], [], [], [], []
    for stream in STREAMS:
        group = [r for r in rows if r["action_stream_id"] == stream]
        stream_worlds = [world_endpoint([r for r in group if r["world"] == w], stream) for w in WORLDS]
        worlds.extend(stream_worlds)
        summaries.extend(summarize_worlds(stream_worlds, stream, "single_action_stream_primary"))
        legacy.append({"action_stream_id": stream, "scope": "legacy_pool_worlds_per_platoon_then_extrema",
            **endpoints([{"platoon": p, **pooled([r for r in group if r["platoon"] == p])} for p in range(5)])})
    for world in WORLDS:
        platoons = [{"world": world, "platoon": p, "action_stream_id": None, "stream_count": 2,
                     "model_id": "uniform_random", "condition": "uniform_random", "target_scenario": dc.SCENARIO,
                     "training_seed": None,
                     **pooled([r for r in rows if r["world"] == world and r["platoon"] == p])} for p in range(5)]
        combined_p.extend(platoons)
        combined_w.append(world_endpoint(platoons, None))
    combined_summary = summarize_worlds(combined_w, None, "two_streams_pooled_per_world_per_platoon_supplementary")
    legacy.append({"action_stream_id": None, "scope": "legacy_two_stream_pool_worlds_per_platoon_then_extrema",
        **endpoints([{"platoon": p, **pooled([r for r in rows if r["platoon"] == p])} for p in range(5)])})
    return {"world_stream_platoon": rows, "unit_world_endpoints": worlds, "stream_summary": summaries,
            "pooled_world_platoon": combined_p, "pooled_world_endpoints": combined_w,
            "pooled_world_summary": combined_summary, "legacy_endpoints": legacy}


def all_rows(batch):
    m, matrix = load(batch)
    rows, probes = [], {}
    for unit in matrix:
        record = dc.read(Path(batch) / f"records/unit_{unit['unit_id']:03d}.json")
        part, probe = accept(batch, unit, m, record["attempt"])
        rows.extend(part)
        world = unit["world"]
        dc.require(world not in probes or probes[world] == probe, "physical process differs across action streams")
        probes[world] = probe
    dc.require(sum(r["slot_count"] for r in rows) == 3000000 and sum(r["cam_episode_count"] for r in rows) == 30000,
               "scored denominator/warmup exclusion mismatch")
    return m, rows, probes


def summarize(batch, attempt="a01"):
    dc.require(re.fullmatch(r"a\d{2,4}", attempt), "unsafe analysis attempt")
    out = Path(batch) / "analysis" / attempt
    if out.exists():
        dc.require((out / "verification.json").exists(), "partial analysis preserved; choose new analysis attempt")
        return export(batch, attempt)
    m, rows, probes = all_rows(batch)
    tables = tables_from_rows(rows)
    out.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items(): csv_write(out / f"{name}.csv", table)
    dc.write(out / "exogenous_world_probes.json", probes)
    dc.write(out / "fields_and_protocol.json", {"action_rule": random.ACTION_RULE, "action_rng": RNG_RULE, "protocol": PROTOCOL,
        "statistics": "both streams separately; +/- is ddof1 sample SD over30 worlds, NOT training-seed SD or CI",
        "supplementary": "pool two streams within each world/platoon by sums/counts BEFORE own AoI/CAM extrema;30worlds,not60",
        "CAM": "binary success at each scored episode final slot; not payload completion;percent endpoints*100",
        "global_reward": "raw[world,episode,slot] once; repeated platoon component columns share same global sum; joint unit table counts it once",
        "power": "draw uniform dBm1..30; mean power is sum(10**(logged_float32_dbm/10))/count in float64",
        "theoretical_mean_power_mw_diagnostic_only": 149.56818535992474,
        "warmup": "5 episodes excluded; scored array episodes are already100, never drop first5 again",
        "world_role": WORLD_ROLE, "new_unseen_confirmation": False, "missing": "null/empty, never filled with0"})
    (out / "REPORT_CN.md").write_text("# Uniform Random 冻结协议评估\n\n动作流12和13各30world，非训练seed或训练模型。主表分别报告，两流合并仅作补充；±为30world样本SD，不计算训练seed区间。world1001–1030已被分析，不称新的未见确认。\n\n60单元、300车队行、3000000 scored agent-slots、30000 CAM deadline事件；5warmup不计分。Global联合流按slot计一次；power用线性mW；CAM仅deadline二元成功。无网络、checkpoint或更新。technical PASS不要求优于任何学习算法。未载入历史学习结果，不阻塞运行或自动重评。\n", encoding="utf-8")
    v = {"technical_status": "PASS", "claim_support": "NOT_ASSESSED", "accepted_units": 60, "new_units": 60,
         "reused_units": 0, "platoon_rows": 300, "worlds": WORLDS, "action_stream_ids": STREAMS,
         "action_streams_are_training_seeds": False, "training_models": 0, "training_runs": 0,
         "scored_agent_slots": 3000000, "cam_deadlines": 30000, "warmup_agent_slots_excluded": 150000,
         "supplementary_pooled_world_count": 30, "supplementary_pooled_platoon_rows": 150,
         "global_joint_slot_count": 600000, "physical_probes_match_across_streams": True,
         "new_unseen_confirmation": False, "implementation_commit": m["implementation_commit"]}
    dc.write(out / "verification.json", v)
    return export(batch, attempt)


def export(batch, attempt="a01"):
    dc.require(re.fullmatch(r"a\d{2,4}", attempt), "unsafe analysis attempt")
    m, rows, probes = all_rows(batch)
    out = Path(batch) / "analysis" / attempt
    v = dc.read(out / "verification.json")
    dc.require(v["technical_status"] == "PASS" and v["accepted_units"] == 60 and
               v["implementation_commit"] == m["implementation_commit"], "complete same-commit verification required")
    for key, value in {"platoon_rows": 300, "scored_agent_slots": 3000000, "cam_deadlines": 30000,
            "global_joint_slot_count": 600000, "warmup_agent_slots_excluded": 150000,
            "physical_probes_match_across_streams": True, "training_models": 0, "training_runs": 0,
            "worlds": WORLDS, "action_stream_ids": STREAMS, "new_unseen_confirmation": False}.items():
        dc.require(v[key] == value, f"verification coverage changed: {key}")
    dc.require(dc.read(out / "exogenous_world_probes.json") == {str(w): p for w, p in probes.items()}, "cached physical probes changed")
    for name, table in tables_from_rows(rows).items(): _same_csv(out / f"{name}.csv", table)
    dc.read(out / "fields_and_protocol.json")
    dc.require((out / "REPORT_CN.md").is_file(), "missing report")
    archive = out / "core_evidence.zip"
    files = {p for root in (Path(batch) / "protocol", Path(batch) / "records", out) for p in root.rglob("*")
             if p.is_file() and p.suffix in (".csv", ".json", ".md")}
    for unit in units():
        record = dc.read(Path(batch) / f"records/unit_{unit['unit_id']:03d}.json")
        dest = Path(record["native_completion"]).parent
        files.update(dest / name for name in ("EVAL_COMPLETE.json", "provenance.json", "STARTED.json", "world_stream_platoon.csv"))
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(files): z.write(p, p.relative_to(batch).as_posix())
            for p in sorted(p for p in (Path(batch) / "logs").glob("unit_*.*") if p.suffix in (".out", ".err")):
                # Small failure tails only; all complete logs and raw NPZ remain local.
                if p.stat().st_size:
                    with p.open("rb") as f:
                        f.seek(max(0, p.stat().st_size - 2048)); z.writestr("log_excerpts/" + p.name, f.read())
    with zipfile.ZipFile(archive) as z:
        dc.require(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) and
                   not any(Path(n).suffix in (".npz", ".pt") for n in z.namelist()), "invalid/oversize evidence members")
        dc.require(all(z.read(p.relative_to(batch).as_posix()) == p.read_bytes() for p in files), "existing archive differs from accepted outputs")
        dc.require(len([n for n in z.namelist() if re.fullmatch(r"records/unit_\d{3}.json", n)]) == 60, "ZIP completion count")
    return {**v, "archive": str(archive)}


def status(batch, verify=False):
    m, matrix = load(batch)
    accepted, pending_native, partial, unrun = [], [], [], []
    for unit in matrix:
        i = unit["unit_id"]
        path = Path(batch) / f"records/unit_{i:03d}.json"
        if path.exists():
            r = dc.read(path)
            dc.require(r["technical_status"] == "PASS" and r["unit"] == unit, "accepted unit key changed")
            if verify: accept(batch, unit, m, r["attempt"])
            accepted.append(i)
        else:
            parent = Path(batch) / f"units/unit_{i:03d}"
            native = list(parent.glob("a*/EVAL_COMPLETE.json"))
            if native: pending_native.append({"unit_id": i, "attempt": native[0].parent.name})
            elif parent.exists() or list((Path(batch) / "logs").glob(f"unit_{i:03d}_*")): partial.append(i)
            else: unrun.append(i)
    return {"technical_status": "COMPLETE" if len(accepted) == 60 else "PARTIAL", "accepted_units": len(accepted),
            "expected_units": 60, "accepted_ids": accepted, "native_completion_pending_acceptance": pending_native,
            "failed_or_partial_ids": partial, "unrun_ids": unrun,
            "running": dc.read(Path(batch) / "RUNNING.json") if (Path(batch) / "RUNNING.json").exists() else None}


def smoke(batch):
    m, _ = load(batch)
    path = Path(batch) / "protocol/smoke.json"
    if path.exists():
        result = dc.read(path)
        dc.require(result["technical_status"] == "PASS" and result["implementation_commit"] == m["implementation_commit"], "smoke identity changed")
        return result
    cfg = config_from_dict(dc.read(Path(batch) / "protocol/config.resolved.json"))
    root = Path(tempfile.mkdtemp(prefix="technical_smoke_", dir=Path(batch) / "tmp"))
    checks, probe, initial = [], None, None
    for stream in STREAMS:
        arrays, meta = sample_rollout(cfg, stream, 910001, scored_episodes=2, warmup_episodes=5)
        identity = {**units()[0], "unit_id": -stream, "world": 910001, "action_stream_id": stream,
                    "unit_key": ["technical_uniform_random", dc.SCENARIO, 910001, stream],
                    "action_seed_words": random_action_seed_words(stream, 910001)}
        rows = reduce_unit(arrays, meta, identity, m["implementation_commit"], 2)
        dc.require(sum(r["slot_count"] for r in rows) == 1000 and sum(r["cam_episode_count"] for r in rows) == 10,
                   "technical warmup/scored denominators")
        if probe is not None: dc.require(meta["exogenous_probes"][0] == probe, "technical stream physics differs")
        probe = meta["exogenous_probes"][0]
        np.savez_compressed(root / f"stream{stream}.npz", **arrays)
        csv_write(root / f"stream{stream}_platoon.csv", rows)
        if initial is None: initial = arrays["action_normalized"]
        else: dc.require(not np.array_equal(initial, arrays["action_normalized"]), "streams accidentally identical")
        checks.append({"action_stream_id": stream, "world": 910001, "scored_agent_slots": 1000,
                       "cam_deadlines": 10, "sampled_environment_steps_with_warmup": meta["sampling_checks"]["environment_step_count"]})
    result = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"],
              "scope": "short real-environment 5+2 episodes, technical world910001; NOT formal60 units",
              "checks": checks, "physical_probes_match": True, "artifacts": str(root), "new_training": 0}
    dc.write(path, result)
    return result


def run(batch, device="cpu", workers=2, ids=None, attempt="a01", reason=None):
    # Adapt ONLY the queue's historical (manifest,sources,units) return signature.
    # No fake policy/checkpoint/training-seed source records are manufactured.
    from types import SimpleNamespace
    from analysis.evaluate_training100_selected import run as queue
    def queue_load(path):
        m, matrix = load(path)
        return m, [], matrix
    workflow = SimpleNamespace(CLI_MODULE=CLI_MODULE, load=queue_load, preflight=preflight,
                               execute=execute, summarize=summarize)
    return queue(batch, device, workers, ids, attempt, reason, workflow=workflow)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preview", "init", "preflight", "smoke", "run", "unit", "status", "summary", "export"))
    p.add_argument("--study-root", type=Path)
    p.add_argument("--batch-id", default=BATCH_ID)
    p.add_argument("--expected-commit")
    p.add_argument("--batch-root", type=Path)
    p.add_argument("--device", choices=("cpu",), default="cpu")
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--id", type=int)
    p.add_argument("--ids")
    p.add_argument("--attempt", default="a01")
    p.add_argument("--reason")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args(argv)
    if a.command == "preview": result = preview()
    elif a.command == "init":
        dc.require(a.study_root and a.expected_commit, "init requires study-root and expected-commit")
        result = init(a.study_root, a.expected_commit, a.batch_id)
    else:
        dc.require(a.batch_root, "batch-root required")
        if a.command == "preflight": result = preflight(a.batch_root, a.device)
        elif a.command == "smoke": result = smoke(a.batch_root)
        elif a.command == "unit": result = execute(a.batch_root, a.id, a.device, a.attempt, a.reason)
        elif a.command == "status": result = status(a.batch_root, a.verify)
        elif a.command == "summary": result = summarize(a.batch_root, a.attempt)
        elif a.command == "export": result = export(a.batch_root, a.attempt)
        else:
            ids = [int(v) for v in a.ids.split(",")] if a.ids else None
            result = run(a.batch_root, a.device, a.workers, ids, a.attempt, a.reason)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
