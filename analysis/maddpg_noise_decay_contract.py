"""Four fresh seed10/12 TDec runs, environment-step noise decay; no frozen evaluation."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

from analysis import deployment_contract as dc, training100_contract as tc
from analysis.evaluate_frozen_deployment import csv_read, csv_write
from analysis.mappo_e5_contract import config_differences
from aoi_v2x_reproduction.config import config_from_dict
from aoi_v2x_reproduction.runtime import maddpg_noise as noise
from aoi_v2x_reproduction.runtime.training_selection import load_training_actors

CLI_MODULE = "analysis.maddpg_noise_decay_contract"
BASELINE = "dbf3f8c1f22d702e72fde2e8573b875a8c224c9c"
ROOT = Path("MADDPG-NoiseDecay-TDec/P5_N4_gap25")
SCHEMA = "maddpg_noise_decay_training_v1"
DEFAULT_BATCH = "noise03_to005_ep1500_seed10_12_20261021_v1"
CORE = {"aoi_v2x_reproduction/config.py", "aoi_v2x_reproduction/algorithms/modified_maddpg/agent.py",
        "aoi_v2x_reproduction/runtime/runner.py", "aoi_v2x_reproduction/runtime/shared_resume.py",
        "aoi_v2x_reproduction/runtime/training_selection.py", "aoi_v2x_reproduction/runtime/maddpg_noise.py"}
DIFF_FIELDS = {"exploration_noise_schedule", "exploration_noise_end", "exploration_noise_floor",
               "exploration_noise_decay_environment_steps", "output_root", "run_name"}


def code_check(expected=None):
    commit = dc.git("rev-parse", "HEAD")
    dc.require(dc.git("branch", "--show-current") in ("research/mappo", "") and
               not dc.git("status", "--porcelain", "--untracked-files=all"), "clean pinned research checkout required")
    dc.require(expected is None or expected == commit, "implementation commit changed")
    changed = set(dc.git("diff", "--name-only", BASELINE, "HEAD", "--", "aoi_v2x_reproduction").splitlines())
    dc.require(changed <= CORE, f"outside reviewed noise/snapshot/recovery extension: {changed - CORE}")
    return commit


def cell(i):
    dc.require(type(i) is int and 0 <= i < 4, "run ID must be 0..3")
    return {"run_id": i, "model_id": "algorithm2" if i < 2 else "shared_maddpg_tdec",
            "training_seed": (10, 12)[i % 2], "episodes": 2500, "slots": 100, "environment_steps": 250000}


def accepted_source(batch, item):
    batch = Path(batch).resolve()
    manifest = dc.read(batch / "protocol/manifest.json")
    verification = dc.read(batch / "analysis/a01/verification.json")
    n = 24 if item["model_id"] == "algorithm2" else 6
    dc.require(verification["technical_status"] == "PASS" and verification["accepted_runs"] == n and
               verification["episode_rows"] == n * 2500 and verification["training_env_steps"] == n * 250000 and
               verification["implementation_commit"] == manifest["implementation_commit"] and manifest["rule"] == tc.RULE,
               "accepted original training summary required")
    matches = []
    for path in (batch / "records").glob("train_*.json"):
        r = dc.read(path)
        if (r["model_id"], r["training_seed"]) == (item["model_id"], item["training_seed"]):
            matches.append((path, r))
    dc.require(len(matches) == 1, "one accepted original seed record required")
    path, r = matches[0]
    run = Path(r["run_dir"]).resolve()
    dc.require(run.is_relative_to(batch / "training") and r["technical_status"] == "PASS" and
               r["episodes"] == 2500 and r["environment_steps"] == 250000 and
               r["implementation_commit"] == manifest["implementation_commit"], "original accepted seed mismatch")
    s = {**item, "parent_record": r, "parent_batch": str(batch), "config": dc.read(run / "config.resolved.json"),
         "parent_files": [dc.file_identity(p) for p in
                          (path, run / "config.resolved.json", run / "COMPLETE.json")],
         "source_sharing_interpretation": "missing_or_null_is_false; true_is_shared",
         "parent_manifest": manifest, "parent_verification": verification}
    check_source(s)
    return s


def check_source(s):
    for identity in s["parent_files"]:
        dc.require(dc.file_identity(identity["path"]) == identity, "read-only accepted source changed")
    dc.require(dc.read(s["parent_files"][0]["path"]) == s["parent_record"] and
               dc.read(s["parent_files"][1]["path"]) == s["config"], "parent record/config changed")
    cfg = source_config(s["config"])
    complete = dc.read(s["parent_files"][2]["path"])
    dc.require(cfg.algorithm == "modified_maddpg_tdec" and cfg.seed == s["training_seed"] and
               cfg.maddpg_actor_sharing == (s["model_id"] == "shared_maddpg_tdec") and
               cfg.scenario.id == dc.SCENARIO and cfg.number_agents == 5 and cfg.scenario.platoon_size == 4 and
               cfg.n_rb == 3 and cfg.state_dim == 22 and cfg.episodes == 2500 and cfg.steps_per_episode == 100 and
               cfg.exploration_noise == .3 and cfg.exploration_noise_schedule == "fixed" and
               cfg.target_noise_sigma == .2 and cfg.target_noise_clip == .5 and cfg.target_action_clip == .999 and
               cfg.global_actor_weight == 1 and cfg.global_update_mode == "synchronous_joint" and
               cfg.checkpoint_mode == "policy_only" and not cfg.is_formal_result and
               complete["status"] == "complete" and complete["config_hash"] == cfg.canonical_hash() and
               complete["reproduction_git_commit"] == s["parent_record"]["implementation_commit"] and
               complete["training_reward_selection"]["final_episode"] == 2500, "fixed0.3 source treatment mismatch")


def source_config(raw):
    # Historical independent configs may omit the sharing field or store null.
    effective = dict(raw)
    if effective.get("maddpg_actor_sharing") is None:
        effective.pop("maddpg_actor_sharing", None)
    return config_from_dict(effective)


def derive(s, output_root):
    cfg = source_config(s["config"])
    before = cfg.to_dict()
    cfg.exploration_noise_schedule = noise.SCHEDULE["name"]
    cfg.exploration_noise_end = noise.SCHEDULE["end"]
    cfg.exploration_noise_floor = noise.SCHEDULE["floor"]
    cfg.exploration_noise_decay_environment_steps = noise.SCHEDULE["decay_environment_steps"]
    cfg.run_name = f"noise_decay_{s['model_id']}_seed{cfg.seed:02d}"
    cfg.output_root = str(Path(output_root).resolve())
    delta = config_differences(before, cfg.to_dict())
    dc.require(set(delta) == DIFF_FIELDS, "unintended change outside exploration/logging/output")
    noise.validate_request(cfg, True)
    return cfg, delta


def init(independent_batch, shared_batch, study_root, expected_commit, batch_id=DEFAULT_BATCH):
    commit = code_check(expected_commit)
    study = Path(study_root).resolve()
    dc.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id), "unsafe batch ID")
    batch = study / ROOT / batch_id
    dc.require(not batch.exists(), "new unique batch required; never re-init existing batch")
    sources = [accepted_source(independent_batch if i < 2 else shared_batch, cell(i)) for i in range(4)]
    derived = [derive(s, batch / "training/a01/runs")[0].to_dict() for s in sources]
    batch.mkdir(parents=True, exist_ok=False)
    for folder in ("protocol", "records", "training", "analysis", "logs", "tmp", "cache"):
        (batch / folder).mkdir()
    m = {"schema": SCHEMA, "batch_root": str(batch), "study_root": str(study), "implementation_commit": commit,
         "baseline_commit": BASELINE, "runs": [cell(i) for i in range(4)], "schedule": noise.SCHEDULE,
         "rule": tc.RULE, "policy_snapshot_interval": 100, "full_state_interval": 100,
         "periodic_policies_per_run": 25, "concurrent_training_processes": 2,
         "new_training_runs": 4, "training_env_steps": 1000000, "new_evaluations": 0,
         "initialization": "fresh_actor_critic_optimizer_replay_no_old_weight_load"}
    dc.write(batch / "protocol/manifest.json", m)
    dc.write(batch / "protocol/sources.json", sources)
    dc.write(batch / "protocol/derived_configs.json", derived)
    return m


def load(batch):
    batch = Path(batch).resolve()
    m = dc.read(batch / "protocol/manifest.json")
    code_check(m["implementation_commit"])
    sources = dc.read(batch / "protocol/sources.json")
    dc.require(m["schema"] == SCHEMA and m["batch_root"] == str(batch) and
               batch.parent == Path(m["study_root"]) / ROOT and m["schedule"] == noise.SCHEDULE and
               m["rule"] == tc.RULE and m["runs"] == [cell(i) for i in range(4)] and
               m["policy_snapshot_interval"] == m["full_state_interval"] == 100 and
               m["new_evaluations"] == 0 and m["concurrent_training_processes"] == 2 and
               [s["run_id"] for s in sources] == list(range(4)), "noise-decay protocol changed")
    for s in sources:
        dc.require(all(s[k] == v for k, v in cell(s["run_id"]).items()), "source matrix drift")
        check_source(s)
    dc.require(dc.read(batch / "protocol/derived_configs.json") ==
               [derive(s, batch / "training/a01/runs")[0].to_dict() for s in sources], "derived config snapshot changed")
    return m, sources


def attempt_run(batch, i, attempt):
    dc.require(re.fullmatch(r"a\d{2,4}", attempt), "unsafe attempt")
    item = cell(i)
    return Path(batch) / "training" / attempt / "runs" / f"noise_decay_{item['model_id']}_seed{item['training_seed']:02d}"


def validate_noise_rows(rows, cfg):
    dc.require([int(r["episode"]) for r in rows] == list(range(1, cfg.episodes + 1)), "noise ledger episode coverage")
    for e, row in enumerate(rows):
        k = e * cfg.steps_per_episode
        expected = {"sigma_first": noise.sigma(cfg, k), "sigma_last": noise.sigma(cfg, k + cfg.steps_per_episode - 1),
                    "sigma_mean": float(np.mean([noise.sigma(cfg, k + j) for j in range(cfg.steps_per_episode)]))}
        dc.require(int(row["environment_step_start"]) == k and
                   int(row["environment_step_end_exclusive"]) == k + cfg.steps_per_episode, "noise step counter mismatch")
        dc.require(all(float(row[key]) == v for key, v in expected.items()), "applied sigma differs from protocol")
        dc.require(int(row["action_count"]) == cfg.number_agents * cfg.steps_per_episode and
                   int(row["component_count"]) == 3 * int(row["action_count"]), "noise diagnostic denominator")
        for count, denominator, fraction in (("clipped_component_count", "component_count", "clip_rate"),
                ("rb_changed_count", "action_count", "rb_changed_fraction"), ("mode_changed_count", "action_count", "mode_changed_fraction")):
            value, den = int(row[count]), int(row[denominator])
            dc.require(0 <= value <= den and float(row[fraction]) == value / den, "noise diagnostic counts/rates mismatch")


def verify_run(batch, i, run):
    m, sources = load(batch)
    run = Path(run).resolve()
    dc.require(run == attempt_run(batch, i, run.parents[1].name).resolve(), "wrong attempt/seed run path")
    cfg, _ = derive(sources[i], run.parent)
    dc.require(config_from_dict(dc.read(run / "config.resolved.json")).to_dict() == cfg.to_dict(), "actual config differs from derivation")
    record = tc.verify_run(batch, i, run, contract=sys.modules[__name__])
    complete = dc.read(run / "COMPLETE.json")
    dc.require(complete["final_episode"] == 2500 and complete["environment_steps"] == 250000 and
               complete["episode_resume"]["schema"] == noise.SCHEMA and complete["episode_resume"]["interval"] == 100 and
               complete["actor_network_count"] == (1 if cfg.maddpg_actor_sharing else 5), "budget/architecture/recovery mismatch")
    learning = dc.read(run / "learning_diagnostics.json")
    learns = 250000 - cfg.batch_size + 1
    updates = learns // cfg.policy_delay
    dc.require(len(learning) == 2500 and sum(r["learner_update_count"] for r in learning) == learns and
               sum(r["actor_update_count"] for r in learning) == updates and complete["learn_step_count"] == learns,
               "learner/actor update counts changed")
    if cfg.maddpg_actor_sharing:
        dc.require(complete["actor_optimizer_step_count"] == complete["target_actor_update_count"] == updates,
                   "shared actor optimizer alias/update count")
    validate_noise_rows(csv_read(run / "exploration_noise.csv"), cfg)
    index = dc.read(run / "policy_snapshot_index.json")
    dc.require(complete["policy_snapshot_count"] == len(index) == 25 and
               [r["episode"] for r in index] == list(range(100, 2501, 100)), "25 periodic policies required")
    for row in index:
        noise.check_identity(row)
        path = Path(row["path"]).resolve()
        dc.require(path.is_relative_to(Path(batch).resolve() / "training") and
                   path.parent.parent.name == run.name and row["actor_network_count"] == complete["actor_network_count"] and
                   row["policy_version"] == max(0, row["episode"] * 100 - cfg.batch_size + 1) // cfg.policy_delay,
                   "snapshot source/episode/update count mismatch")
        original, policy = noise.load_periodic_policy(path)
        a, b = original.to_dict(), cfg.to_dict()
        for key in ("output_root", "run_name"):
            a.pop(key, None); b.pop(key, None)
        dc.require(a == b and policy["reproduction_git_commit"] == m["implementation_commit"], "snapshot training provenance mismatch")
        load_training_actors(path)  # Existing strict frozen loader, no environment/optimizer.
    inherited = int(complete["episode_resume"].get("resume_episode", 0))
    source = complete["episode_resume"].get("resume_source")
    started = dc.read(Path(batch) / f"records/started_{i:02d}_{run.parents[1].name}.json")
    dc.require(started["resume_source"] == source and (not source or
               started["resume_source_identity"] == dc.file_identity(source)), "resume lineage source changed")
    if source:
        dc.require(Path(source).resolve().is_relative_to(Path(batch).resolve() / "training") and 0 < inherited < 2500,
                   "only same-batch checkpoints may resume")
    return {**record, "from_scratch": not bool(source), "initialization_from_old_weights": False,
            "resume_source": source, "inherited_episodes": inherited, "attempt_environment_steps": (2500 - inherited) * 100,
            "actor_structure": "shared" if cfg.maddpg_actor_sharing else "independent",
            "actor_network_count": complete["actor_network_count"], "actor_update_events": updates,
            "total_actor_adam_steps": updates * complete["actor_network_count"], "periodic_policy_count": 25,
            "metrics_source": dc.file_identity(run / "train_metrics.npz"),
            "noise_ledger": dc.file_identity(run / "exploration_noise.csv"),
            "snapshot_index_source": dc.file_identity(run / "policy_snapshot_index.json")}


def preflight(batch):
    from analysis.run_training100_local import cuda_probe
    m, sources = load(batch)
    dc.require(sys.version_info[:2] == (3, 10) and torch.__version__.split("+")[0] == "2.11.0" and
               torch.version.cuda == "12.6", "reuse established Python3.10/torch2.11.0/cu126, do not upgrade")
    versions = {}
    for line in (dc.REPO / "requirements.cuda.lock.txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        name, expected = line.strip().split("==")
        versions[name] = importlib.metadata.version(name)
        dc.require(versions[name] == expected, f"dependency mismatch: {name}")
    for s in sources:
        dc.require(derive(s, Path(batch) / "training/a01/runs")[0].device == "cuda:0", "retain source CUDA device")
    result = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"], "python_executable": sys.executable,
              "dependencies": versions, "gpu": cuda_probe(), "source_configs": 4, "old_weights_loaded": 0,
              "formal_training_in_preflight": 0, "new_evaluations": 0}
    path = Path(batch) / "protocol/preflight.json"
    if path.exists(): dc.require(dc.read(path) == result, "preflight runtime changed")
    else: dc.write(path, result)
    return result


def train_run(batch, i, attempt="a01", reason=None, *, resume=None):
    from aoi_v2x_reproduction.runtime.runner import train
    m, sources = load(batch)
    pre = dc.read(Path(batch) / "protocol/preflight.json")
    dc.require(pre["technical_status"] == "PASS" and pre["python_executable"] == sys.executable and
               pre["implementation_commit"] == m["implementation_commit"], "same-runtime preflight required")
    smoke_check = dc.read(Path(batch) / "protocol/smoke_cuda_0.json")
    dc.require(smoke_check["technical_status"] == "PASS" and smoke_check["implementation_commit"] == m["implementation_commit"],
               "same-commit CUDA technical smoke required before formal training")
    record = Path(batch) / f"records/train_{i:02d}.json"
    if record.exists():
        saved = dc.read(record)
        dc.require(resume is None and verify_run(batch, i, saved["run_dir"]) == saved, "accepted run changed/cannot rerun")
        return {"skipped_complete": True, **saved}
    run = attempt_run(batch, i, attempt)
    dc.require(attempt == "a01" or reason, "retry/resume requires recorded reason")
    completed = list((Path(batch) / "training").glob(f"a*/runs/{run.name}/COMPLETE.json"))
    dc.require(len(completed) <= 1, "duplicate completed runs")
    if completed:
        dc.require(resume is None, "accept existing native COMPLETE, do not resume/retrain")
        result = verify_run(batch, i, completed[0].parent)
    else:
        dc.require(not run.exists(), "partial attempt preserved; resume into a NEW attempt")
        cfg, delta = derive(sources[i], run.parent)
        if resume:
            path = Path(resume).resolve()
            dc.require(reason and path.is_relative_to((Path(batch) / "training").resolve()) and
                       path.parent.name == "checkpoints" and path.parent.parent.name == run.name,
                       "resume only from this run/batch full state; never old fixed0.3 policies")
            noise.read_resume(path, cfg)
        dc.write(Path(batch) / f"records/started_{i:02d}_{attempt}.json", {**cell(i), "attempt": attempt, "reason": reason,
            "implementation_commit": m["implementation_commit"], "derivation": delta,
            "resume_source": str(Path(resume).resolve()) if resume else None,
            "resume_source_identity": dc.file_identity(resume) if resume else None})
        train(cfg, training_reward_best=True, noise_decay_training=True, resume=str(resume) if resume else None)
        result = verify_run(batch, i, run)
    code_check(m["implementation_commit"])
    dc.write(record, result)
    return result


def business_rows(run, record, batch):
    """Same float32-logged -> float64 sum/count convention as accepted training business exports."""
    run, batch = Path(run), Path(batch)
    ledger = csv_read(run / "training_reward_selection.csv")
    with np.load(run / "train_metrics.npz", allow_pickle=False) as z:
        fields = ("aoi_ms", "success", "remaining_demand", "power_dbm", "task1_step", "task2_step", "global_step", "mode")
        data = {k: z[k].astype(np.float64) for k in fields}
        data["reset_event"] = z["reset_event"].astype(np.float64) if "reset_event" in z else None
    aoi = data["aoi_ms"]
    dc.require(aoi.shape == (2500, 100, 5) and all(data[k].shape == aoi.shape for k in fields if k != "global_step") and
               data["global_step"].shape == (2500, 100) and all(np.isfinite(data[k]).all() for k in fields),
               "training business array shape/nonfinite error")
    cam = data["success"][:, -1, :]
    dc.require(np.isin(data["mode"], [0, 1]).all(), "invalid decoded execution mode")
    dc.require(np.isin(cam, [0, 1]).all() and np.array_equal(cam.astype(bool), data["remaining_demand"][:, -1, :] == 0),
               "CAM endpoint/remaining demand mismatch")
    reset = data["reset_event"]
    fallback = np.isclose(aoi, 1, rtol=0, atol=1e-6)
    dc.require(reset is None or (reset.shape == aoi.shape and np.isin(reset, [0, 1]).all() and
               np.array_equal(reset.astype(bool), fallback)), "reset event semantics disagree")
    event = fallback if reset is None else reset.astype(bool)
    dc.require(not np.any(event & (data["mode"] != 0)), "V2I resets on non-V2I slots")
    sums = {"aoi_sum_ms": aoi.sum(axis=1), "aoi_gt_50_slot_count": (aoi > 50).sum(axis=1),
            "power_mw_sum": np.power(10., data["power_dbm"] / 10).sum(axis=1),
            "task1_reward_sum": data["task1_step"].sum(axis=1), "task2_reward_sum": data["task2_step"].sum(axis=1),
            "v2i_attempt_count": (data["mode"] == 0).sum(axis=1), "v2i_reset_count": event.sum(axis=1)}
    global_sums = data["global_step"].sum(axis=1)
    for e in range(2500):
        reward = float((data["task1_step"][e] + data["task2_step"][e] + data["global_step"][e, :, None]).mean(dtype=np.float64))
        dc.require(reward == float(ledger[e]["episode_reward"]), "business/reward ledger differs")
        for p in range(5):
            yield {"run_id": record["run_id"], "model_id": record["model_id"], "training_seed": record["training_seed"],
                "attempt": run.parents[1].name, "source_run": run.relative_to(batch).as_posix(), "episode": e + 1,
                "platoon_index": p, **{k: float(v[e, p]) if k.endswith("_sum") or k == "aoi_sum_ms" else int(v[e, p]) for k, v in sums.items()},
                "aoi_slot_count": 100, "cam_deadline_success_count": int(cam[e, p]), "cam_deadline_event_count": 1,
                "episode_reward": reward, "task1_slot_count": 100, "task2_slot_count": 100,
                "global_reward_sum": float(global_sums[e]), "global_reward_slot_count": 100, "power_slot_count": 100,
                "event_source": "reset_event" if reset is not None else "post_step_aoi_isclose_1"}


def window_summary(rows, record, name, start, end):
    subset = [r for r in rows if start <= r["episode"] <= end]
    slots = sum(r["aoi_slot_count"] for r in subset)
    cam = sum(r["cam_deadline_event_count"] for r in subset)
    result = {"run_id": record["run_id"], "model_id": record["model_id"], "training_seed": record["training_seed"],
              "window": name, "start": start, "end": end, "agent_slots": slots, "cam_events": cam}
    for label, field, den in (("mean_aoi_ms", "aoi_sum_ms", slots), ("cam_success_rate", "cam_deadline_success_count", cam),
            ("mean_power_mw", "power_mw_sum", slots), ("task1_reward_mean", "task1_reward_sum", slots),
            ("task2_reward_mean", "task2_reward_sum", slots), ("global_reward_mean", "global_reward_sum", slots),
            ("v2i_reset_rate", "v2i_reset_count", slots), ("aoi_gt50_fraction", "aoi_gt_50_slot_count", slots)):
        result[label] = sum(r[field] for r in subset) / den
    result["combined_reward_mean"] = float(np.mean([subset[i]["episode_reward"] for i in range(0, len(subset), 5)]))
    return result


def episode_totals(rows):
    """Global reward has ONE joint stream; do not count its five repeated copies as five measurements."""
    dc.require(len(rows) == 5 and {r["platoon_index"] for r in rows} == set(range(5)), "episode platoon coverage")
    first = rows[0]
    row = {k: first[k] for k in ("run_id", "model_id", "training_seed", "attempt", "source_run", "episode", "episode_reward")}
    for key in ("aoi_sum_ms", "aoi_slot_count", "cam_deadline_success_count", "cam_deadline_event_count",
                "power_mw_sum", "power_slot_count", "task1_reward_sum", "task1_slot_count", "task2_reward_sum", "task2_slot_count",
                "aoi_gt_50_slot_count", "v2i_attempt_count", "v2i_reset_count"):
        row[key] = sum(r[key] for r in rows)
    dc.require(all(r["global_reward_sum"] == first["global_reward_sum"] for r in rows), "global stream copies disagree")
    row.update(global_reward_sum=first["global_reward_sum"], global_reward_slot_count=first["global_reward_slot_count"])
    row.update(mean_aoi_ms=row["aoi_sum_ms"] / row["aoi_slot_count"],
               cam_deadline_completion_rate=row["cam_deadline_success_count"] / row["cam_deadline_event_count"],
               mean_power_mw=row["power_mw_sum"] / row["power_slot_count"],
               task1_reward_mean=row["task1_reward_sum"] / row["task1_slot_count"],
               task2_reward_mean=row["task2_reward_sum"] / row["task2_slot_count"],
               global_reward_mean=row["global_reward_sum"] / row["global_reward_slot_count"])
    row["local_reward_mean"] = row["task1_reward_mean"] + row["task2_reward_mean"]
    dc.require(np.isclose(row["local_reward_mean"] + row["global_reward_mean"], row["episode_reward"], rtol=0, atol=1e-12),
               "episode component means fail reward reconstruction")
    return row


def summary(batch, attempt="a01"):
    m, _ = load(batch)
    dc.require(re.fullmatch(r"a\d{2,4}", attempt), "unsafe analysis attempt")
    out = Path(batch) / "analysis" / attempt
    if (out / "verification.json").exists():
        verify_outputs(batch, attempt)
        return dc.read(out / "verification.json")
    dc.require(not out.exists(), "partial analysis preserved; use a new analysis attempt")
    records, curves, noise_rows, snapshots, business, windows, episodes = [], [], [], [], [], [], []
    for i in range(4):
        r = dc.read(Path(batch) / f"records/train_{i:02d}.json")
        dc.require(verify_run(batch, i, r["run_dir"]) == r, "accepted training changed")
        run = Path(r["run_dir"])
        records.append(r)
        curves.extend({**cell(i), **row} for row in csv_read(run / "training_reward_selection.csv"))
        noise_rows.extend({**cell(i), **row} for row in csv_read(run / "exploration_noise.csv"))
        snapshots.extend({**cell(i), **row} for row in dc.read(run / "policy_snapshot_index.json"))
        rows = list(business_rows(run, r, batch)); business.extend(rows)
        episodes.extend(episode_totals(rows[j:j + 5]) for j in range(0, len(rows), 5))
        for name, start, end in (("best100", r["window_start"], r["window_end"]), ("last100", 2401, 2500), ("all", 1, 2500)):
            windows.append(window_summary(rows, r, name, start, end))
    dc.require(len(curves) == len(noise_rows) == 10000 and len(snapshots) == 100 and len(business) == 50000 and
               len({(r["run_id"], r["episode"], r["platoon_index"]) for r in business}) == 50000 and
               sum(r["aoi_slot_count"] for r in business) == 5000000 and
               sum(r["cam_deadline_event_count"] for r in business) == 50000, "aggregate coverage/denominators")
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("selected_policies", records), ("training_episode_curves", curves), ("exploration_noise", noise_rows),
                       ("policy_snapshot_index", snapshots), ("training_episode_platoon", business),
                       ("training_episode_metrics", episodes), ("training_window_summary", windows)):
        csv_write(out / f"{name}.csv", rows)
    dc.write(out / "fields_and_boundaries.json", {"schema": SCHEMA, "reward": tc.RULE, "schedule": noise.SCHEDULE,
        "training_episode_platoon": "one row/run/episode/platoon; episode_reward is the run-wide episode combined mean repeated five times, not five independent rewards",
        "global_reward": "same global stream repeated per platoon; pooled divisor is agent-slots, not environment-slots",
        "power": "sum(10**(logged_float32_dbm/10)) in float64; mean mW is NOT mean dBm transformed",
        "CAM": "success at final slot == remaining_demand zero; one deadline/platoon/episode",
        "AoI": "post-step capped AoI; strict >50; no warmup/test worlds", "best100": "rolling100 strict improvements every episode, not checkpoint average",
        "snapshot": "lightweight actor-only every100; NEVER a resume state; inherited snapshot paths retain original attempts",
        "repeat_unit": "two training seeds per structure; no significance screen or deployment claim"})
    (out / "report_cn.md").write_text("4 个从头噪声递减训练 run 的技术验收。best100 仍使用训练历史逐集选优；末100集受执行噪声变化影响，不能据此宣称冻结部署改善。未执行策略评估。\n", encoding="utf-8")
    result = {"technical_status": "PASS", "claim_support": "NOT_ASSESSED", "accepted_runs": 4,
        "episode_rows": 10000, "episode_platoon_rows": 50000, "training_env_steps": 1000000,
        "agent_slots": 5000000, "cam_deadline_events": 50000, "periodic_policy_snapshots": 100,
        "best_policies": 4, "final_policies": 4, "new_evaluations": 0, "implementation_commit": m["implementation_commit"]}
    dc.write(out / "verification.json", result)  # Last: never publish a premature PASS.
    return result


def verify_outputs(batch, attempt):
    m, _ = load(batch)
    out = Path(batch) / "analysis" / attempt
    v = dc.read(out / "verification.json")
    dc.require(v["technical_status"] == "PASS" and v["accepted_runs"] == 4 and
               v["implementation_commit"] == m["implementation_commit"], "complete same-commit summary required")
    expected = {name: [] for name in ("selected_policies", "training_episode_curves", "exploration_noise",
                "policy_snapshot_index", "training_episode_platoon", "training_episode_metrics", "training_window_summary")}
    for i in range(4):
        r = dc.read(Path(batch) / f"records/train_{i:02d}.json")
        dc.require(verify_run(batch, i, r["run_dir"]) == r, "accepted output/source changed")
        run = Path(r["run_dir"])
        expected["selected_policies"].append(r)
        for name, filename in (("training_episode_curves", "training_reward_selection.csv"), ("exploration_noise", "exploration_noise.csv")):
            expected[name].extend({**cell(i), **row} for row in csv_read(run / filename))
        expected["policy_snapshot_index"].extend({**cell(i), **row} for row in dc.read(run / "policy_snapshot_index.json"))
        rows = list(business_rows(run, r, batch))
        expected["training_episode_platoon"].extend(rows)
        expected["training_episode_metrics"].extend(episode_totals(rows[j:j + 5]) for j in range(0, len(rows), 5))
        for name, start, end in (("best100", r["window_start"], r["window_end"]), ("last100", 2401, 2500), ("all", 1, 2500)):
            expected["training_window_summary"].append(window_summary(rows, r, name, start, end))
    for name, n in (("selected_policies", 4), ("training_episode_curves", 10000), ("exploration_noise", 10000),
                    ("policy_snapshot_index", 100), ("training_episode_platoon", 50000),
                    ("training_episode_metrics", 10000), ("training_window_summary", 12)):
        actual = csv_read(out / f"{name}.csv")
        dc.require(len(actual) == len(expected[name]) == n and all(a == {k: "" if v is None else str(v) for k, v in b.items()}
                   for a, b in zip(actual, expected[name])), f"incomplete or altered {name} output")
    dc.read(out / "fields_and_boundaries.json")
    dc.require((out / "report_cn.md").is_file(), "missing report")
    return v


def verify_archive(path, attempt="a01"):
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        dc.require(z.testzip() is None and len(names) == len(set(names)) and
                   not any(Path(n).suffix in (".pt", ".npz") for n in names), "ZIP corrupt/duplicate or contains large weights/arrays")
        if path.name == "core_evidence.zip":
            v = json.loads(z.read(f"analysis/{attempt}/verification.json"))
            dc.require(v["technical_status"] == "PASS" and v["accepted_runs"] == 4 and v["periodic_policy_snapshots"] == 100,
                       "ZIP verification incomplete")
            dc.require(len([n for n in names if re.fullmatch(r"records/train_\d{2}.json", n)]) == 4, "ZIP accepted record coverage")
        else:
            rows = list(csv.DictReader(io.StringIO(z.read("training_episode_platoon.csv").decode("utf-8-sig"))))
            dc.require(len(rows) == 50000 and len({(r["run_id"], r["episode"], r["platoon_index"]) for r in rows}) == 50000 and
                       sum(int(r["aoi_slot_count"]) for r in rows) == 5000000 and
                       sum(int(r["cam_deadline_event_count"]) for r in rows) == 50000, "business ZIP denominator/coverage error")
    return True


def export(batch, attempt="a01"):
    verify_outputs(batch, attempt)
    out = Path(batch) / "analysis" / attempt
    files = set()
    for folder in (Path(batch) / "protocol", Path(batch) / "records", out):
        files.update(p for p in folder.rglob("*") if p.is_file() and p.suffix in (".csv", ".json", ".md") and p.name != "training_episode_platoon.csv")
    for i in range(4):
        r = dc.read(Path(batch) / f"records/train_{i:02d}.json")
        run = Path(r["run_dir"])
        for name in ("COMPLETE.json", "config.resolved.json", "provenance.json", "learning_diagnostics.json",
                     "training_reward_selection.csv", "exploration_noise.csv", "policy_snapshot_index.json", "progress.json"):
            files.add(run / name)
    archive = out / "core_evidence.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(files): z.write(p, p.relative_to(batch).as_posix())
    verify_archive(archive, attempt)
    # Existing compact archives must reflect the immutable current protocol/records, not just a PASS bit.
    with zipfile.ZipFile(archive) as z:
        dc.require(set(z.namelist()) == {p.relative_to(batch).as_posix() for p in files} and
                   all(z.read(p.relative_to(batch).as_posix()) == p.read_bytes() for p in files), "existing evidence differs from accepted outputs")
    business = out / "training_episode_platoon_metrics.zip"
    mapping = {"training_episode_platoon.csv": out / "training_episode_platoon.csv",
               "training_episode_metrics.csv": out / "training_episode_metrics.csv",
               "training_window_summary.csv": out / "training_window_summary.csv",
               "verification.json": out / "verification.json", "schema.json": out / "fields_and_boundaries.json"}
    if not business.exists():
        with zipfile.ZipFile(business, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for name, p in mapping.items(): z.write(p, name)
    verify_archive(business, attempt)
    with zipfile.ZipFile(business) as z:
        dc.require(set(z.namelist()) == set(mapping) and all(z.read(n) == p.read_bytes() for n, p in mapping.items()),
                   "existing business evidence differs from accepted outputs")
    return {"technical_status": "PASS", "archive": str(archive), "business_archive": str(business),
            "accepted_runs": 4, "new_evaluations": 0}


def status(batch):
    m, _ = load(batch)
    result = {"implementation_commit": m["implementation_commit"], "accepted_ids": [], "runs": []}
    for i in range(4):
        path = Path(batch) / f"records/train_{i:02d}.json"
        if path.exists(): result["accepted_ids"].append(i)
        attempts = []
        for run in sorted((Path(batch) / "training").glob(f"a*/runs/{attempt_run(batch, i, 'a01').name}")):
            progress = dc.read(run / "progress.json") if (run / "progress.json").exists() else None
            latest = dc.read(run / "checkpoints/latest.json") if (run / "checkpoints/latest.json").exists() else None
            attempts.append({"run_dir": str(run), "native_complete": (run / "COMPLETE.json").exists(),
                             "progress": progress, "latest_full_checkpoint": latest})
        result["runs"].append({**cell(i), "accepted": path.exists(), "attempts": attempts})
    result["accepted_runs"] = len(result["accepted_ids"])
    lock = Path(m["study_root"]) / ".training100_local_gpu.lock"
    result["gpu_queue_lock"] = dc.read(lock) if lock.exists() else None
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init", "preflight", "smoke", "run", "train", "status", "summary", "export"))
    p.add_argument("--batch-root", type=Path)
    p.add_argument("--independent-batch", type=Path)
    p.add_argument("--shared-batch", type=Path)
    p.add_argument("--study-root", type=Path)
    p.add_argument("--expected-commit")
    p.add_argument("--batch-id", default=DEFAULT_BATCH)
    p.add_argument("--id", type=int)
    p.add_argument("--ids")
    p.add_argument("--workers", type=int, choices=(1, 2), default=2)
    p.add_argument("--attempt", default="a01")
    p.add_argument("--reason")
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0", help="technical smoke only")
    a = p.parse_args(argv)
    dc.require(a.resume is None or a.command == "train", "resume only explicit train ID/new attempt")
    if a.command == "init":
        dc.require(a.independent_batch and a.shared_batch and a.study_root and a.expected_commit, "init source/study/commit required")
        result = init(a.independent_batch, a.shared_batch, a.study_root, a.expected_commit, a.batch_id)
    else:
        dc.require(a.batch_root, "batch-root required")
        if a.command == "preflight": result = preflight(a.batch_root)
        elif a.command == "train": result = train_run(a.batch_root, a.id, a.attempt, a.reason, resume=a.resume)
        elif a.command == "status": result = status(a.batch_root)
        elif a.command == "summary": result = summary(a.batch_root, a.attempt)
        elif a.command == "export": result = export(a.batch_root, a.attempt)
        elif a.command == "smoke":
            from analysis.maddpg_noise_decay_smoke import smoke
            result = smoke(a.batch_root, a.device)
        else:
            from analysis.run_shared_maddpg_local import run_queue
            result = run_queue(a.batch_root, a.ids, a.attempt, a.reason, workflow=sys.modules[__name__], workers=a.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
