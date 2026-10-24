"""CAM96k: three MAPPO methods x two seeds, paired local training and frozen evaluation.

The smoke batch has five episodes per seed and two scored episodes per world.
Only formal batches use the unchanged Training100 selector. Smoke artifacts never
qualify as formal training or evaluation; no policy-only training resume exists.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from analysis import deployment_contract as dc, training100_contract as tc
from analysis.deployment_rng import action_seed_words, policy_seed
from analysis.evaluate_frozen_deployment import csv_read, csv_write, load_arrays, reduce_arrays
from analysis.mappo_e5_contract import config_differences
from aoi_v2x_reproduction.config import config_from_dict
from aoi_v2x_reproduction.runtime.training_selection import BEST, RULE, load_training_actors, load_training_policy

REPO = dc.REPO
BASELINE = "ee05154233c9122f5cf6e57e13cf372a60bdf843"
MODELS = ("shared_tdec", "independent_tdec", "shared_combined")
SEEDS = (12, 13)
WORLDS = (1001, 1002, 1003, 1004, 1005)
ROOT = Path("CAM96k-training-pilot")
WORLD_ROLE = "user_fixed_exploratory_worlds"
SCHEMA = "cam96k_training_pilot_v1"
require, read, write = dc.require, dc.read, dc.write


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def code_check(expected=None, smoke=False):
    commit = dc.git("rev-parse", "HEAD")
    require(expected is None or commit == expected, "HEAD differs from pinned implementation commit")
    require(dc.git("branch", "--show-current").startswith("research/cam96k-training-pilot"), "wrong development/execution branch")
    if not smoke:
        require(not dc.git("status", "--porcelain", "--untracked-files=all"), "formal execution requires a clean checkout")
    changes = dc.git("diff", "--name-only", BASELINE, "--", "aoi_v2x_reproduction").splitlines()
    require(set(changes) <= {"aoi_v2x_reproduction/runtime/runner.py"}, "unexpected algorithm/environment/config changes")
    return commit


def cells():
    return [{"run_id": i, "model_id": model, "training_seed": seed}
            for i, (model, seed) in enumerate((m, s) for m in MODELS for s in SEEDS)]


def cell(i):
    require(type(i) is int and 0 <= i < 6, "training ID must be 0..5")
    return cells()[i]


def sources():
    all_sources = tc.bundled_sources()
    return [next(s for s in all_sources if (s["model_id"], s["training_seed"]) ==
                 (c["model_id"], c["training_seed"])) for c in cells()]


def derive(source, output_root, smoke=False):
    # Reuse the accepted Training100 derivation, including explicit independent actors.
    cfg, _ = tc.derive(source, output_root)
    before = cfg.to_dict()
    cfg.cam_bits = 96000
    cfg.run_name = f"cam96k_{source['model_id']}_s{cfg.seed}"
    if smoke:
        cfg.episodes = 5
        cfg.smoke = True
    delta = config_differences(source["config"], cfg.to_dict())
    require(set(config_differences(before, cfg.to_dict())) <= {"cam_bits", "run_name", "episodes", "smoke"}, "CAM96k derivation drift")
    require(cfg.algorithm == "mappo" and cfg.seed in SEEDS and cfg.cam_bits == 96000 and
            cfg.steps_per_episode == 100 and cfg.number_agents == 5 and cfg.scenario.platoon_size == 4 and
            cfg.scenario.gap_m == 25 and cfg.n_rb == 3 and cfg.slot_ms == 1 and
            cfg.mappo_rollout_episodes == 5, "effective scenario/budget mismatch")
    return cfg, delta


def run_path(batch, i, attempt):
    require(re.fullmatch(r"a[0-9]{2,4}", attempt) is not None, "unsafe attempt name")
    c = cell(i)
    return Path(batch) / "t" / attempt / f"cam96k_{c['model_id']}_s{c['training_seed']}"


def units(smoke=False):
    worlds = WORLDS[:1] if smoke else WORLDS
    return [{"unit_id": len(worlds) * c["run_id"] + j, **c, "world": world, "repeat": 0,
             "action_seed_words": action_seed_words(c["training_seed"], world, 0, "mappo")}
            for c in cells() for j, world in enumerate(worlds)]


def init(study, batch_id, expected_commit, smoke=False):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,49}", batch_id) is not None, "unsafe batch ID")
    require(("smoke" in batch_id.lower()) == smoke, "smoke identity must appear only in smoke batch names")
    commit = code_check(expected_commit, smoke)
    study = Path(study).resolve()
    batch = study / ROOT / batch_id
    require(not batch.exists(), "new unique batch required; never overwrite an existing batch")
    for c, source in zip(cells(), sources()):
        derive(source, run_path(batch, c["run_id"], "a01").parent, smoke)
    batch.mkdir(parents=True)
    for folder in ("protocol", "records", "logs", "locks", "t", "e", "tmp"):
        (batch / folder).mkdir()
    manifest = {"schema": SCHEMA, "batch_root": str(batch), "study_root": str(study),
                "implementation_commit": commit, "baseline_commit": BASELINE, "smoke": smoke,
                "training_episodes": 5 if smoke else 2500, "slots": 100, "cam_bits": 96000,
                "from_scratch": True, "resume_supported": False, "rule": RULE,
                "runs": cells(), "eval_units": units(smoke), "sources": sources(),
                "source_bundle": identity(tc.BUNDLE), "world_role": WORLD_ROLE,
                "new_unseen_confirmation": False, "warmup": 5, "scored": 2 if smoke else 100,
                "max_training_processes": 2, "pairing": "same_method_seed12_seed13_then_next_method",
                "created_at_utc": now()}
    write(batch / "protocol/manifest.json", manifest)
    return manifest


def load(batch):
    batch = Path(batch).resolve()
    m = read(batch / "protocol/manifest.json")
    smoke = m["smoke"]
    code_check(m["implementation_commit"], smoke)
    require(type(smoke) is bool and m["schema"] == SCHEMA and m["batch_root"] == str(batch) and
            batch.parent == Path(m["study_root"]).resolve() / ROOT and
            ("smoke" in batch.name.lower()) == smoke and m["baseline_commit"] == BASELINE and
            m["training_episodes"] == (5 if smoke else 2500) and m["slots"] == 100 and m["cam_bits"] == 96000 and
            m["runs"] == cells() and m["eval_units"] == units(smoke) and m["sources"] == sources() and
            m["source_bundle"] == identity(tc.BUNDLE) and m["rule"] == RULE and
            m["from_scratch"] is True and m["resume_supported"] is False and
            m["world_role"] == WORLD_ROLE and m["new_unseen_confirmation"] is False and
            m["warmup"] == 5 and m["scored"] == (2 if smoke else 100) and m["max_training_processes"] == 2 and
            m["pairing"] == "same_method_seed12_seed13_then_next_method", "batch contract changed")
    return m


def preview(batch):
    m = load(batch)
    return {"smoke": m["smoke"], "training_runs": 6, "training_environment_steps": m["training_episodes"] * 600,
            "eval_units": len(m["eval_units"]), "world_role": WORLD_ROLE, "max_training_processes": 2,
            "pairs": [[0, 1], [2, 3], [4, 5]], "configs": [
                {**c, "effective_config": derive(m["sources"][c["run_id"]], run_path(batch, c["run_id"], "a01").parent, m["smoke"])[0].to_dict(),
                 "differences_from_original_template": derive(m["sources"][c["run_id"]], run_path(batch, c["run_id"], "a01").parent, m["smoke"])[1]}
                for c in cells()]}


def preflight(batch):
    from analysis.run_training100_local import cuda_probe
    from aoi_v2x_reproduction.envs.platoon import PaperEnviron
    m = load(batch)
    require(sys.version_info[:2] == (3, 10) and torch.__version__.split("+")[0] == "2.11.0" and
            torch.version.cuda == "12.6", "use existing Python3.10 torch2.11.0 cu126 environment")
    versions = {}
    for line in (REPO / "requirements.cuda.lock.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, expected = line.strip().split("==")
            versions[name] = importlib.metadata.version(name)
            require(versions[name] == expected, f"dependency mismatch: {name}")
    # Environment RNG is private; this verifies the actual initial demand, not just JSON.
    for source in m["sources"]:
        cfg, _ = derive(source, Path(batch) / "tmp", m["smoke"])
        require(cfg.device == "cuda:0", "retain historical cuda:0 device")
        env = PaperEnviron(cfg)
        env.reset_world(cfg.seed)
        env.start_episode(0)
        require(np.all(env.v2v_demand == 96000), "environment initial CAM demand differs from 96000")
    report = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"], "smoke": m["smoke"],
              "python": sys.executable, "python_version": sys.version, "torch": torch.__version__,
              "cuda": torch.version.cuda, "dependencies": versions, "gpu": cuda_probe(),
              "initial_demand_verified": 96000, "configs_checked": 6, "max_training_processes": 2,
              "new_training": 0, "new_evaluation": 0, "created_at_utc": now()}
    write(Path(batch) / f"protocol/preflight_{time.time_ns()}.json", report)
    return report


@contextmanager
def lock(path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "python": sys.executable, "created_at_utc": now(), **metadata}, handle)
    try:
        yield
    finally:
        path.unlink()  # Only a lock acquired by this context; never delete stale locks automatically.


@contextmanager
def training_slot(study, batch):
    # Also limits direct `train-unit` commands to two processes across this study.
    for i in range(2):
        path = Path(study) / f".cam96k-training-slot{i}.lock"
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "batch": str(batch), "created_at_utc": now()}, handle)
        except FileExistsError:
            continue
        try:
            yield
        finally:
            path.unlink()
        return
    raise ValueError("two CAM96k training slots occupied; inspect locks and active PIDs")


def verify_run(batch, i, run):
    m = load(batch)
    run = Path(run).resolve()
    require(run.parent.parent == Path(batch).resolve() / "t" and run == run_path(batch, i, run.parent.name), "run outside expected unit/attempt")
    cfg, delta = derive(m["sources"][i], run.parent, m["smoke"])
    complete = read(run / "COMPLETE.json")
    episodes = m["training_episodes"]
    expected_updates = episodes // 5
    require(complete["status"] == "complete" and complete["config_hash"] == cfg.canonical_hash() ==
            config_from_dict(read(run / "config.resolved.json")).canonical_hash() and
            complete["reproduction_git_commit"] == m["implementation_commit"] and
            (m["smoke"] or complete["reproduction_git_dirty"] is False) and complete["seed"] == cfg.seed and
            complete["episodes"] == episodes and complete["environment_steps"] == episodes * 100 and
            complete["update_count"] == expected_updates and complete["actor_network_count"] == (1 if cfg.mappo_actor_sharing else 5) and
            complete["actor_optimizer_step_count"] == expected_updates * cfg.mappo_ppo_epochs * (1 if cfg.mappo_actor_sharing else 5),
            "training config/provenance/budget/update count mismatch")
    expected_critic = ("joint_observation_per_agent_value" if cfg.mappo_variant == "combined" else
                       "joint_observation_scalar_global_plus_independent_local_task1_task2_values")
    require(complete["mappo_critic_structure"] == expected_critic, "critic structure mismatch")
    provenance = read(run / "provenance.json")
    require(provenance["reproduction_git_commit"] == m["implementation_commit"] and
            (m["smoke"] or provenance["reproduction_git_dirty"] is False), "training provenance mismatch")
    rewards = tc.rewards_from_npz(run / "train_metrics.npz", episodes)
    require(np.isfinite(rewards).all(), "nonfinite training rewards")
    result = {**cell(i), "technical_status": "PASS", "smoke": m["smoke"], "run_dir": str(run),
              "implementation_commit": m["implementation_commit"], "episodes": episodes, "environment_steps": episodes * 100,
              "update_count": expected_updates, "actor_network_count": complete["actor_network_count"],
              "critic_structure": complete["mappo_critic_structure"], "from_scratch": True, "derivation": delta,
              "final": identity(run / "policy_final.pt"), "metrics": identity(run / "train_metrics.npz"),
              "complete": identity(run / "COMPLETE.json"), "config": identity(run / "config.resolved.json")}
    if m["smoke"]:
        from aoi_v2x_reproduction.runtime.runner import _validate_mappo_policy_artifact
        payload = torch.load(run / "policy_final.pt", map_location="cpu", weights_only=False)
        _validate_mappo_policy_artifact(run / "policy_final.pt", payload)
        return result
    rows = csv_read(run / "training_reward_selection.csv")
    episode, score, scores = tc.best_index(rewards)
    require(len(rows) == episodes and [int(r["episode"]) for r in rows] == list(range(1, episodes + 1)), "selection ledger coverage")
    maximum = None
    for e, row in enumerate(rows, 1):
        value = scores[e - 100] if e >= 100 else None
        better = value is not None and (maximum is None or value > maximum)
        require(float(row["episode_reward"]) == rewards[e - 1] and
                (row["rolling100"] == "" if value is None else float(row["rolling100"]) == value) and
                row["new_best"] == str(better) and row["save_timing"] == "after_episode_before_pending_ppo_update" and
                int(row["policy_version_before_episode"]) == int(row["policy_version_saved"]) == (e - 1) // 5,
                "selection rewards/window/policy timing mismatch")
        if better:
            maximum = value
    _, best = load_training_policy(run / BEST)
    _, final = load_training_policy(run / "policy_final.pt")
    load_training_actors(run / BEST)
    load_training_actors(run / "policy_final.pt")
    selected = best["training_reward_selection"]
    require(selected["selected_episode"] == episode and selected["score"] == score and
            selected["window_start"] == episode - 99 and selected["window_end"] == episode and
            selected["window_policy_version_min"] == (episode - 100) // 5 and
            selected["window_policy_version_max"] == (episode - 1) // 5 and
            best["policy_version"] == (episode - 1) // 5 and final["policy_version"] == 500,
            "best100/final selection mismatch")
    return {**result, "best": identity(run / BEST), "selection_ledger": identity(run / "training_reward_selection.csv"),
            "best_episode": episode, "best_score": score, "best_policy_version": best["policy_version"], "final_policy_version": 500}


def accepted_run(batch, i):
    saved = read(Path(batch) / f"records/train_{i}.json")
    require(verify_run(batch, i, saved["run_dir"]) == saved, "accepted training artifact identity changed")
    return saved


def train_unit(batch, i, attempt="a01", reason=None):
    from aoi_v2x_reproduction.runtime.runner import train, _write_json
    m = load(batch)
    run = run_path(batch, i, attempt)
    require(attempt == "a01" or reason, "new attempt requires technical failure reason; no low-score retries")
    with lock(Path(batch) / f"locks/train_{i}.lock", batch=str(batch), unit=i):
        record = Path(batch) / f"records/train_{i}.json"
        if record.exists():
            return {"skipped_verified": True, **accepted_run(batch, i)}
        # Pair barrier also applies to manual single-unit execution.
        for previous in range((i // 2) * 2):
            accepted_run(batch, previous)
        completed = list((Path(batch) / "t").glob(f"a*/{run.name}/COMPLETE.json"))
        require(len(completed) <= 1, "multiple completed attempts require investigation")
        if completed:
            result = verify_run(batch, i, completed[0].parent)
        else:
            require(not run.exists(), "partial attempt preserved; diagnose then use a fresh attempt")
            cfg, _ = derive(m["sources"][i], run.parent, m["smoke"])
            started = {**cell(i), "pid": os.getpid(), "attempt": attempt, "reason": reason,
                       "created_at_utc": now(), "from_scratch": True, "resume": False, "smoke": m["smoke"]}
            write(Path(batch) / f"records/started_{i}_{attempt}.json", started)
            def progress(data):
                report = {**started, **data, "updated_at_utc": now(),
                          "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                          "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                _write_json(Path(batch) / f"records/heartbeat_{i}_{attempt}.json", report)
                if data["episode"] % 5 == 0:
                    print(json.dumps(report), flush=True)
            with training_slot(m["study_root"], batch):
                try:
                    train(cfg, training_reward_best=not m["smoke"], progress_callback=progress)
                except BaseException as exc:
                    write(Path(batch) / f"records/failed_{i}_{attempt}.json", {**started, "failed_at_utc": now(), "error": repr(exc)})
                    raise
            result = verify_run(batch, i, run)
        write(record, result)
        return result


def child_env(batch):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)}
    for name in ("AOI_TASK_OFFSET", "AOI_RETRY_REASON", "AOI_ACCEPT_EXISTING", "AOI_CHECK_ONLY"):
        env.pop(name, None)
    for name in ("TMP", "TEMP", "TMPDIR"):
        env[name] = str(Path(batch) / "tmp")
    return env


def launch_group(batch, command, ids, attempt, reason):
    active = []
    outcomes = []
    try:
        for i in ids:
            cmd = [sys.executable, "-B", "-u", "-m", "analysis.cam96k_training_pilot", command,
                   "--batch-root", str(batch), "--id", str(i), "--attempt", attempt]
            if reason:
                cmd += ["--reason", reason]
            log = Path(batch) / f"logs/{command}_{i}_{attempt}_{time.time_ns()}.out"
            handle = log.open("x", encoding="utf-8")
            try:
                process = subprocess.Popen(cmd, cwd=REPO, env=child_env(batch), stdout=handle, stderr=subprocess.STDOUT)
            except BaseException:
                handle.close()
                raise
            metadata = {"unit_id": i, "pid": process.pid, "command": cmd, "log": str(log), "started_at_utc": now()}
            active.append((process, handle, metadata, time.perf_counter()))
            print(json.dumps(metadata), flush=True)
        last_report = time.perf_counter()
        while any(p.poll() is None for p, *_ in active):
            if time.perf_counter() - last_report >= 15:
                print(json.dumps({"queue_heartbeat_utc": now(), "processes": [
                    {"pid": p.pid, "unit_id": meta["unit_id"], "exit_code": p.poll(),
                     "elapsed_seconds": time.perf_counter() - start} for p, _, meta, start in active]}), flush=True)
                last_report = time.perf_counter()
            time.sleep(0.5)
        for p, _, meta, start in active:
            outcome = {**meta, "exit_code": p.returncode, "elapsed_seconds": time.perf_counter() - start, "finished_at_utc": now()}
            write(Path(batch) / f"records/process_{command}_{meta['unit_id']}_{time.time_ns()}.json", outcome)
            outcomes.append(outcome)
        require(all(o["exit_code"] == 0 for o in outcomes), f"group failed; preserve attempts and inspect logs: {outcomes}")
        return outcomes
    finally:
        for p, handle, _, _ in active:
            if p.poll() is None:  # Only this launcher's children, on interrupt/error.
                p.terminate()
                p.wait()
            handle.close()


def train_all(batch, attempt="a01", reason=None):
    m = load(batch)
    preflight(batch)
    with lock(Path(m["study_root"]) / ".cam96k-training-supervisor.lock", batch=str(batch)):
        for pair in ((0, 1), (2, 3), (4, 5)):
            pending = []
            for i in pair:
                if (Path(batch) / f"records/train_{i}.json").exists():
                    accepted_run(batch, i)
                    print(f"training {i}: accepted, reverified, skipped", flush=True)
                else:
                    pending.append(i)
            if pending:
                launch_group(batch, "train-unit", pending, attempt, reason)
            for i in pair:
                accepted_run(batch, i)
    return verify_training(batch)


def verify_training(batch):
    m = load(batch)
    records = [accepted_run(batch, i) for i in range(6)]
    report = {"technical_status": "PASS", "smoke": m["smoke"], "accepted_runs": 6,
              "training_environment_steps": sum(r["environment_steps"] for r in records), "records": records}
    path = Path(batch) / "records/training_verification.json"
    if path.exists():
        require(read(path) == report, "training verification changed")
    else:
        write(path, report)
    return report


def reduce_payload(arrays, metadata, unit, scored):
    rows = reduce_arrays(arrays, metadata, {k: v for k, v in unit.items() if k != "action_seed_words"}, expected=(1, scored, 100, 5))
    remaining = np.asarray(arrays["remaining_demand"], dtype=np.float64)[:, :, -1, :]
    success = np.asarray(arrays["success"])[:, :, -1, :].astype(bool)
    require(np.all((remaining >= 0) & (remaining <= 96000)) and np.array_equal(success, remaining == 0), "deadline success/zero remaining mismatch")
    for p, row in enumerate(rows):
        failed = remaining[0, :, p][~success[0, :, p]]
        row.update(deadline_remaining_bits_sum=float(remaining[0, :, p].sum()),
                   failed_remaining_bits_sum=float(failed.sum()), failed_episode_count=int(len(failed)))
    return rows


def verify_eval(batch, unit_id, attempt):
    m = load(batch)
    require(0 <= unit_id < len(m["eval_units"]) and re.fullmatch(r"a[0-9]{2,4}", attempt) is not None, "invalid eval ID/attempt")
    unit = m["eval_units"][unit_id]
    record = accepted_run(batch, unit["run_id"])
    cfg, _ = derive(m["sources"][unit["run_id"]], Path(record["run_dir"]).parent, m["smoke"])
    dest = Path(batch) / "e" / str(unit_id) / attempt
    started = read(dest / "STARTED.json")
    require(started["source"] == record and started["unit"] == unit, "evaluation training source/unit changed")
    meta = read(dest / "native/EVAL_COMPLETE.json")
    expected = {"status": "complete", "training_seed": unit["training_seed"], "eval_seeds": [unit["world"]],
                "environment_seeds": [unit["world"]], "action_repeat": 0, "action_seed_words": [unit["action_seed_words"]],
                "policy_sampling_seeds": [policy_seed(unit["training_seed"], unit["world"], 0)],
                "eval_protocol": "sequential_warm", "eval_warmup_episodes": 5, "eval_episodes": m["scored"],
                "target_scenario": dc.SCENARIO, "source_scenario": dc.SCENARIO, "source_config_hash": cfg.canonical_hash(),
                "training_source_commit": m["implementation_commit"], "reproduction_git_commit": m["implementation_commit"],
                "world_role": WORLD_ROLE, "cam_bits": 96000, "global_actor_weight": 1,
                "policy_parameters_unchanged": True, "critic_constructed": False, "optimizer_constructed": False,
                "rollout_constructed": False, "mappo_eval_mode": "stochastic", "mappo_variant": cfg.mappo_variant,
                "actor_sharing": bool(cfg.mappo_actor_sharing), "actor_network_count": 1 if cfg.mappo_actor_sharing else 5,
                "policy_episode": m["training_episodes"] if m["smoke"] else record["best_episode"],
                "target_config_hash": cfg.canonical_hash(), "target_config": cfg.to_dict()}
    if not m["smoke"]:
        expected.update(reproduction_git_dirty=False, checkpoint_role="best_training100", policy_version=record["best_policy_version"], training_budget_episodes=2500)
    for key, value in expected.items():
        require(meta.get(key) == value, f"native evaluation {key} mismatch")
    require(len(meta.get("exogenous_probes", [])) == 1 and meta["exogenous_probes"][0]["world"] == unit["world"], "missing physical RNG probe")
    policy = record["final" if m["smoke"] else "best"]
    require((dest / "native" / meta["policy"]).resolve() == Path(policy["path"]) and identity(policy["path"]) == policy, "evaluation policy identity changed")
    rows = reduce_payload(load_arrays(dest / "native/metrics.npz"), meta, unit, m["scored"])
    result = {"technical_status": "PASS", "unit": unit, "attempt": attempt, "smoke": m["smoke"], "source": policy,
              "completion": identity(dest / "native/EVAL_COMPLETE.json"), "metrics": identity(dest / "native/metrics.npz"), "rows": rows}
    return result


def eval_unit(batch, unit_id, attempt="a01", reason=None):
    from analysis.evaluate_mappo_zero_shot import evaluate_zero_shot
    m = load(batch)
    require(type(unit_id) is int and 0 <= unit_id < len(m["eval_units"]), "invalid eval unit")
    require(re.fullmatch(r"a[0-9]{2,4}", attempt) is not None and (attempt == "a01" or reason), "invalid attempt or missing retry reason")
    # Full six-run gate before even a single frozen rollout.
    verify_training(batch)
    unit = m["eval_units"][unit_id]
    with lock(Path(batch) / f"locks/eval_{unit_id}.lock", unit=unit_id):
        out = Path(batch) / f"records/eval_{unit_id}.json"
        if out.exists():
            saved = read(out)
            require(verify_eval(batch, unit_id, saved["attempt"]) == saved, "accepted evaluation identity changed")
            return {"skipped_verified": True, "unit_id": unit_id}
        parent = Path(batch) / "e" / str(unit_id)
        completed = list(parent.glob("a*/native/EVAL_COMPLETE.json"))
        require(len(completed) <= 1, "duplicate evaluation completions")
        if completed:
            attempt = completed[0].parent.parent.name
        else:
            dest = parent / attempt
            require(not dest.exists(), "partial evaluation preserved; fresh attempt and failure reason required")
            require(not parent.exists() or reason, "retry requires technical failure reason")
            source = accepted_run(batch, unit["run_id"])
            policy = Path(source["final" if m["smoke"] else "best"]["path"])
            dest.mkdir(parents=True)
            write(dest / "STARTED.json", {"unit": unit, "source": source, "pid": os.getpid(), "reason": reason, "created_at_utc": now()})
            try:
                evaluate_zero_shot(policy, dest, dc.SCENARIO, "cuda:0", [unit["world"]], m["scored"],
                                   deployment=True, world_role=WORLD_ROLE, training_policy=not m["smoke"], compact_output=True)
            except BaseException as exc:
                write(dest / "FAILED.json", {"failed_at_utc": now(), "error": repr(exc), "reason": reason})
                raise
        result = verify_eval(batch, unit_id, attempt)
        write(out, result)
        return {"unit_id": unit_id, "technical_status": "PASS"}


def eval_all(batch, workers=2, attempt="a01", reason=None):
    m = load(batch)
    require(1 <= workers <= 8, "evaluation workers must be 1..8; training concurrency remains 2")
    verify_training(batch)
    preflight(batch)
    with lock(Path(batch) / "locks/eval_supervisor.lock", batch=str(batch)):
        pending = []
        for unit in m["eval_units"]:
            i = unit["unit_id"]
            out = Path(batch) / f"records/eval_{i}.json"
            if out.exists():
                saved = read(out)
                require(verify_eval(batch, i, saved["attempt"]) == saved, "accepted evaluation changed")
            else:
                pending.append(i)
        for offset in range(0, len(pending), workers):
            launch_group(batch, "eval-unit", pending[offset:offset + workers], attempt, reason)
    return summary(batch)


def aggregate(rows):
    result = {"cam_success_count": sum(r["cam_deadline_success_count"] for r in rows),
              "cam_count": sum(r["cam_episode_count"] for r in rows),
              "deadline_remaining_bits_sum": sum(r["deadline_remaining_bits_sum"] for r in rows),
              "failed_remaining_bits_sum": sum(r["failed_remaining_bits_sum"] for r in rows),
              "failed_count": sum(r["failed_episode_count"] for r in rows)}
    result.update(cam_rate=result["cam_success_count"] / result["cam_count"],
                  deadline_remaining_bits=result["deadline_remaining_bits_sum"] / result["cam_count"],
                  failed_remaining_bits=result["failed_remaining_bits_sum"] / result["failed_count"] if result["failed_count"] else "NA")
    for key, numerator, denominator in (("aoi_ms", "aoi_sum_ms", "slot_count"),
                                        ("power_mw", "power_mw_sum", "power_slot_count"),
                                        ("payload_delivery_ratio", "payload_completion_sum", "cam_episode_count"),
                                        *[(k.removesuffix("_sum"), k, "reward_slot_count") for k in rows[0] if k.startswith("reward_") and k.endswith("_sum")]):
        if all(r.get(numerator) is not None for r in rows):
            result[key] = sum(r[numerator] for r in rows) / sum(r[denominator] for r in rows)
    return result


def summary(batch):
    m = load(batch)
    verify_training(batch)
    rows = []
    records = []
    for unit in m["eval_units"]:
        saved = read(Path(batch) / f"records/eval_{unit['unit_id']}.json")
        require(verify_eval(batch, unit["unit_id"], saved["attempt"]) == saved, "accepted eval changed")
        records.append(saved)
        rows.extend(saved["rows"])
    seeds, worlds, overall = [], [], []
    for model in MODELS:
        model_rows = [r for r in rows if r["model_id"] == model]
        for seed in SEEDS:
            group = [r for r in model_rows if r["training_seed"] == seed]
            seeds.append({"model_id": model, "training_seed": seed, **aggregate(group)})
            for world in sorted({r["world"] for r in group}):
                group_world = [r for r in group if r["world"] == world]
                worlds.append({"model_id": model, "training_seed": seed, "world": world, **aggregate(group_world),
                               "worst_platoon_cam_rate": min(r["cam_deadline_success_count"] / r["cam_episode_count"] for r in group_world),
                               "worst_platoon_aoi_ms": max(r["aoi_sum_ms"] / r["slot_count"] for r in group_world)})
        across = [r for r in seeds if r["model_id"] == model]
        overall.append({"model_id": model, "n_training_seeds": 2, **aggregate(model_rows),
                        "cam_rate_seed_sd": float(np.std([r["cam_rate"] for r in across], ddof=1)),
                        "aoi_ms_seed_sd": float(np.std([r["aoi_ms"] for r in across], ddof=1)),
                        "power_mw_seed_sd": float(np.std([r["power_mw"] for r in across], ddof=1))})
    report = {"technical_status": "PASS", "smoke": m["smoke"], "accepted_training_runs": 6,
              "accepted_eval_units": len(records), "platoon_rows": len(rows), "n_training_seeds_per_method": 2,
              "cam_deadlines": sum(r["cam_episode_count"] for r in rows),
              "scored_agent_slots": sum(r["slot_count"] for r in rows), "world_role": WORLD_ROLE,
              "new_unseen_confirmation": False, "selection_uses_evaluation": False,
              "implementation_commit": m["implementation_commit"]}
    out = Path(batch) / "tables"
    out.mkdir(exist_ok=True)
    for name, table in (("world_platoon.csv", rows), ("world_summary.csv", worlds), ("per_seed.csv", seeds), ("across_seeds.csv", overall)):
        path = out / name
        if path.exists():
            expected = [{k: "" if row.get(k) is None else str(row[k]) for k in sorted({k for r in table for k in r})} for row in table]
            require(csv_read(path) == expected, f"existing summary changed: {name}")
        else:
            csv_write(path, table)
    if (out / "verification.json").exists():
        require(read(out / "verification.json") == report, "summary verification changed")
    else:
        write(out / "verification.json", report)
    return report


def status(batch):
    m = load(batch)
    result = {"smoke": m["smoke"], "implementation_commit": m["implementation_commit"], "training": [], "evaluation": []}
    for c in cells():
        i = c["run_id"]
        accepted = Path(batch) / f"records/train_{i}.json"
        attempts = []
        failures = [read(p) for p in Path(batch).glob(f"records/failed_{i}_*.json")]
        for run in sorted((Path(batch) / "t").glob(f"a*/{run_path(batch, i, 'a01').name}")):
            hb = Path(batch) / f"records/heartbeat_{i}_{run.parent.name}.json"
            failed = Path(batch) / f"records/failed_{i}_{run.parent.name}.json"
            complete = (run / "COMPLETE.json").exists()
            attempts.append({"attempt": run.parent.name, "path": str(run), "complete": complete,
                             "state": "COMPLETE" if complete else "FAILED" if failed.exists() else "INCOMPLETE",
                             "heartbeat": read(hb) if hb.exists() else None})
        started = list(Path(batch).glob(f"records/started_{i}_*.json"))
        result["training"].append({**c, "state": "ACCEPTED" if accepted.exists() else "FAILED" if failures else
                                   "INCOMPLETE" if attempts or started else "PENDING", "attempts": attempts, "failures": failures})
    for u in m["eval_units"]:
        attempts = []
        for folder in sorted((Path(batch) / "e" / str(u["unit_id"])).glob("a*")):
            complete = (folder / "native/EVAL_COMPLETE.json").exists()
            failed = folder / "FAILED.json"
            attempts.append({"attempt": folder.name, "path": str(folder),
                             "state": "COMPLETE" if complete else "FAILED" if failed.exists() else "INCOMPLETE",
                             "failure": read(failed) if failed.exists() else None})
        result["evaluation"].append({**u, "state": "ACCEPTED" if (Path(batch) / f"records/eval_{u['unit_id']}.json").exists() else
                                      "FAILED" if any(a["state"] == "FAILED" for a in attempts) else
                                      "INCOMPLETE" if attempts else "PENDING", "attempts": attempts})
    result["locks"] = [{"path": str(p), "metadata": read(p)} for p in sorted((Path(batch) / "locks").glob("*.lock"))]
    result["study_locks"] = [{"path": str(p), "metadata": read(p)} for p in sorted(Path(m["study_root"]).glob(".cam96k-*.lock"))]
    return result


def reuse_uniform(batch, uniform_root):
    """Recheck existing 96k random arrays and protocol; performs zero rollouts."""
    m = load(batch)
    root = Path(uniform_root).resolve()
    original = read(root / "protocol/manifest.json")
    verified = read(root / "verification.json")
    require(original["schema"] == "uniform_random_cam_payload_v1" and verified["technical_status"] == "PASS" and
            original["worlds"] == verified["worlds"] == list(WORLDS) and
            original["action_stream_ids"] == verified["action_stream_ids"] == list(SEEDS) and
            96000 in original["payloads"] and original["action_stream_ids_are_training_seeds"] is False and
            original["payload_in_action_seed"] is False and original["seed_sequence"] == "[action_stream_id, world, 0x52414E44, 1]" and
            original["warmup_episodes"] == 5 and original["scored_episodes"] == 100 and original["slots"] == 100 and
            original["world_role"] == WORLD_ROLE and original["new_unseen_confirmation"] is False,
            "Uniform source protocol/identity mismatch")
    cfg = derive(m["sources"][0], Path(batch) / "tmp")[0].to_dict()
    files = [identity(root / "protocol/manifest.json"), identity(root / "verification.json")]
    rows = []
    for world in WORLDS:
        for stream in SEEDS:
            folder = root / f"runs/cam96000/world{world}/stream{stream}"
            meta = read(folder / "rollout.json")
            complete = read(folder / "unit_complete.json")
            expected = {"technical_status": "PASS", "cam_bits": 96000, "world": world, "action_stream_id": stream,
                        "action_seed_words": [stream, world, 0x52414E44, 1], "payload_in_action_seed": False,
                        "initial_demand_episode_starts": 105, "warmup_episodes_excluded": 5,
                        "scored_episodes": 100, "cam_deadlines": 500}
            require(all(meta.get(k) == complete.get(k) == v for k, v in expected.items()), "Uniform unit budget/identity mismatch")
            runtime = config_from_dict(meta["runtime_config"]).to_dict()
            require(all(runtime[k] == cfg[k] for k in dc.ENV_FIELDS) and
                    meta["eval_seeds"] == [world] and meta["eval_warmup_episodes"] == 5 and
                    meta["eval_episodes"] == 100 and meta["eval_protocol"] == "sequential_warm" and
                    meta["world_role"] == WORLD_ROLE, "Uniform effective environment mismatch")
            unit = {"model_id": "uniform_random", "action_stream_id": stream, "world": world}
            rows.extend(reduce_payload(load_arrays(folder / "metrics.npz"), meta, unit, 100))
            files.extend(identity(folder / n) for n in ("rollout.json", "unit_complete.json", "metrics.npz"))
    result = {"technical_status": "PASS", "source_root": str(root), "source_files": files,
              "source_runtime_commit": original["implementation_commit"], "source_runtime_worktree_status": original["worktree_status"],
              "world_role": WORLD_ROLE, "new_unseen_confirmation": False, "new_rollouts": 0, "new_training": 0,
              "n_training_models": 0, "action_stream_ids": list(SEEDS), "cam_bits": 96000,
              "platoon_rows": len(rows), "metrics": aggregate(rows)}
    out = Path(batch) / "protocol/uniform_reuse.json"
    if out.exists():
        require(read(out) == result, "reused Uniform evidence changed")
    else:
        write(out, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "preview", "preflight", "train-unit", "train", "verify-training", "eval-unit", "evaluate", "summary", "status", "reuse-uniform"))
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--expected-commit")
    parser.add_argument("--smoke", action="store_true", help="init a separate 5-episode smoke batch; never formal")
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--id", type=int)
    parser.add_argument("--attempt", default="a01")
    parser.add_argument("--reason")
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--uniform-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "init":
        require(args.study_root and args.batch_id and args.expected_commit, "init requires study root, batch ID, expected commit")
        result = init(args.study_root, args.batch_id, args.expected_commit, args.smoke)
    else:
        require(args.batch_root is not None, "batch root required")
        require(not args.smoke, "--smoke only applies to init; later commands use the pinned manifest")
        if args.command in {"train-unit", "eval-unit"}:
            require(args.id is not None, "unit ID required")
            func = train_unit if args.command == "train-unit" else eval_unit
            result = func(args.batch_root, args.id, args.attempt, args.reason)
        elif args.command == "train":
            result = train_all(args.batch_root, args.attempt, args.reason)
        elif args.command == "evaluate":
            result = eval_all(args.batch_root, args.eval_workers, args.attempt, args.reason)
        elif args.command == "reuse-uniform":
            require(args.uniform_root is not None, "Uniform source root required")
            result = reuse_uniform(args.batch_root, args.uniform_root)
        else:
            result = {"preview": preview, "preflight": preflight, "verify-training": verify_training,
                      "summary": summary, "status": status}[args.command](args.batch_root)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
