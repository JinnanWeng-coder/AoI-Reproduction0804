"""24 fresh 2500-episode runs with per-seed training100 best + final; no eval."""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch

from analysis import deployment_contract as dc
from analysis.mappo_e5_contract import config_differences
from aoi_v2x_reproduction.config import config_from_dict
from aoi_v2x_reproduction.runtime.training_selection import RULE, BEST, episode_reward, load_training_policy, load_training_actors

BASELINE = "2ce572b9c7fe487fc7a0898cdd8330fdab04bc4a"
MODELS = ("shared_tdec", "independent_tdec", "shared_combined", "algorithm2")
PILOTS = (0, 6, 12, 18)
ROOT = Path("Training100-best/P5_N4_gap25")
EPISODES = 2500
BUNDLE = dc.REPO / "configs/training100_fresh_sources.json"


def bundled_sources():
    """Expand exact historical configs, not historical payloads or machine paths."""
    bundle = dc.read(BUNDLE)
    dc.require(bundle["schema"] == "training100_config_bundle_v1" and
               bundle["origin_implementation_commit"] == "75098b00714daf315a1d82298deb563209da9ac8" and
               set(bundle["templates"]) == set(MODELS), "configuration bundle identity mismatch")
    sources = []
    for entry in bundle["sources"]:
        cfg = copy.deepcopy(bundle["templates"][entry["model_id"]])
        cfg.update(seed=entry["training_seed"], run_name=entry["run_name"])
        dc.require(config_from_dict(cfg).canonical_hash() == entry["source_config_hash"], "bundled historical config identity mismatch")
        sources.append({"source_kind": "bundled_config", "model_id": entry["model_id"], "training_seed": entry["training_seed"],
                        "config": cfg, "source_config_hash": entry["source_config_hash"], "training_commit": entry["training_commit"],
                        "historical_run_reference_only": entry["source_run"]})
    dc.require(len(sources) == 24 and {(s["model_id"], s["training_seed"]) for s in sources} ==
               {(m, seed) for m in MODELS for seed in dc.SEEDS}, "24 bundled configurations required")
    return sources


def check_source(source):
    if source.get("source_kind") == "bundled_config":
        dc.require(source in bundled_sources(), "bundled source changed")
    else:
        dc.source_check(source)


def code_check(expected=None):
    commit = dc.git("rev-parse", "HEAD")
    dc.require(dc.git("branch", "--show-current") in {"research/mappo", ""} and
               not dc.git("status", "--porcelain", "--untracked-files=all"), "pinned clean research checkout required")
    dc.require(expected is None or expected == commit, "training100 implementation changed")
    changes = dc.git("diff", "--name-only", BASELINE, "HEAD", "--", "aoi_v2x_reproduction").splitlines()
    dc.require(set(changes) <= {"aoi_v2x_reproduction/runtime/runner.py", "aoi_v2x_reproduction/runtime/training_selection.py"},
               f"unexpected algorithm/environment/config change: {changes}")
    return commit


def cell(i):
    dc.require(0 <= i < 24, "training100 ID must be 0..23")
    return {"run_id": i, "model_id": MODELS[i // 6], "training_seed": 8 + i % 6,
            "episodes": EPISODES, "slots": 100, "environment_steps": 250000}


def derive(source, output_root):
    cfg = config_from_dict(source["config"])
    before = cfg.to_dict()
    dc.require(cfg.seed == source["training_seed"] and cfg.scenario.id == dc.SCENARIO and cfg.episodes == 500 and
               cfg.steps_per_episode == 100 and cfg.number_agents == 5 and cfg.n_rb == 3 and cfg.global_actor_weight == 1 and
               cfg.checkpoint_mode == "policy_only" and not cfg.is_formal_result, "source training configuration mismatch")
    if source["model_id"] == "algorithm2":
        dc.require(cfg.algorithm == "modified_maddpg_tdec" and cfg.exploration_noise == .3, "MADDPG training noise/algorithm mismatch")
    else:
        spec = dc.MODELS[source["model_id"]]
        dc.require(cfg.algorithm == "mappo" and cfg.mappo_variant == spec["variant"] and
                   bool(cfg.mappo_actor_sharing) == (spec["structure"] == "shared") and cfg.mappo_rollout_episodes == 5,
                   "MAPPO source treatment mismatch")
    cfg.episodes = EPISODES
    cfg.run_name = f"train100_{source['model_id']}_seed{cfg.seed:02d}"
    cfg.output_root = str(Path(output_root).resolve())
    if cfg.algorithm == "mappo" and cfg.mappo_actor_sharing is None:
        cfg.mappo_actor_sharing = False
    differences = config_differences(before, cfg.to_dict())
    dc.require(set(differences) <= {"episodes", "run_name", "output_root", "mappo_actor_sharing"}, "training derivation drift")
    return cfg, differences


def init(study, batch_id, old_batch, b1_root):
    dc.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id), "unsafe batch ID")
    commit = code_check()
    study, old_batch, b1_root = [Path(p).resolve() for p in (study, old_batch, b1_root)]
    old, sources = dc.load_batch(old_batch, check_code=False)
    dc.require(old["implementation_commit"] == "75098b00714daf315a1d82298deb563209da9ac8", "wrong historical model ledger")
    dc.require(len(sources) == 24 and {(s["model_id"], s["training_seed"]) for s in sources} ==
               {(m, s) for m in MODELS for s in dc.SEEDS}, "24 original models required")
    batch = study / ROOT / batch_id
    dc.require(not batch.exists(), "unique new batch required")
    for source in sources:
        derive(source, batch / "training/a01/runs")
    batch.mkdir(parents=True)
    for folder in ("protocol", "records", "training", "analysis", "slurm_logs", "tmp", "cache"):
        (batch / folder).mkdir()
    dc.write(batch / "protocol/manifest.json", {"schema": "training100_batch_v1", "batch_root": str(batch), "study_root": str(study),
             "implementation_commit": commit, "baseline_commit": BASELINE, "rule": RULE, "runs": [cell(i) for i in range(24)],
             "old_batch": str(old_batch), "b1_root": str(b1_root), "source_ledger": dc.file_identity(old_batch / "protocol/sources.json"),
             "new_evaluations": 0, "new_worlds": [], "no_legacy_resume_implemented": True})
    dc.write(batch / "protocol/sources.json", sources)
    return {"batch_root": str(batch), "new_from_scratch_runs_if_no_qualified_reuse": 24, "environment_steps": 6000000}


def init_local(study, batch_id):
    dc.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id), "unsafe batch ID")
    commit = code_check()
    study = Path(study).resolve()
    batch = study / ROOT / batch_id
    dc.require(not batch.exists(), "unique new batch required")
    sources = bundled_sources()
    for source in sources:
        derive(source, batch / "training/a01/runs")
    batch.mkdir(parents=True)
    for folder in ("protocol", "records", "training", "analysis", "logs", "tmp", "cache"):
        (batch / folder).mkdir()
    dc.write(batch / "protocol/manifest.json", {"schema": "training100_batch_v1", "batch_root": str(batch), "study_root": str(study),
             "implementation_commit": commit, "baseline_commit": BASELINE, "rule": RULE, "runs": [cell(i) for i in range(24)],
             "source_mode": "bundled_config_fresh", "source_ledger": dc.file_identity(BUNDLE),
             "historical_payload_audit": "NOT_RUN_NOT_REQUIRED_FOR_AUTHORIZED_FRESH_TRAINING",
             "new_evaluations": 0, "new_worlds": [], "no_legacy_resume_implemented": True})
    dc.write(batch / "protocol/sources.json", sources)
    dc.write(batch / "protocol/execution_decision.json", {"implementation_commit": commit, "decision": "24_FROM_SCRATCH", "reuse": 0,
             "approved_runs": [cell(i) for i in range(24)], "evaluations": 0, "execution_mode": "direct24", "pilot_required": False,
             "max_concurrent_gpus": 1, "authorization": "fresh local training; configurations only, no old weights/arrays required"})
    return {"batch_root": str(batch), "from_scratch_runs": 24, "environment_steps": 6000000, "pilot_required": False}


def load(batch):
    batch = Path(batch).resolve()
    m, ss = dc.read(batch / "protocol/manifest.json"), dc.read(batch / "protocol/sources.json")
    code_check(m["implementation_commit"])
    dc.require(m["schema"] == "training100_batch_v1" and m["batch_root"] == str(batch) and
               batch.parent == Path(m["study_root"]).resolve() / ROOT and m["rule"] == RULE and
               m["runs"] == [cell(i) for i in range(24)] and len(ss) == 24 and
               {(s["model_id"], s["training_seed"]) for s in ss} == {(k, s) for k in MODELS for s in dc.SEEDS}, "batch contract changed")
    dc.require(dc.file_identity(m["source_ledger"]["path"]) == m["source_ledger"], "historical source ledger changed")
    if m.get("source_mode") == "bundled_config_fresh":
        dc.require(m["source_ledger"] == dc.file_identity(BUNDLE) and ss == bundled_sources(), "local configuration snapshot changed")
    return m, ss


def rewards_from_npz(path, episodes):
    with np.load(path, allow_pickle=False) as z:
        a, b, c = [z[k] for k in ("task1_step", "task2_step", "global_step")]
    dc.require(a.shape == b.shape == (episodes, 100, 5) and c.shape == (episodes, 100), "training metric shape mismatch")
    return [episode_reward(a[e], b[e], c[e]) for e in range(episodes)]


def best_index(rewards):
    scores = [float(np.mean(rewards[e - 100:e], dtype=np.float64)) for e in range(100, len(rewards) + 1)]
    dc.require(scores and np.isfinite(scores).all(), "invalid training100 scores")
    index = int(np.argmax(scores))  # Earliest tie.
    return index + 100, scores[index], scores


def audit(batch):
    m, ss = load(batch)
    dc.require(m.get("source_mode") != "bundled_config_fresh", "local fresh mode uses bundled config checks, not historical payload audit")
    rows = []
    for source in ss:
        dc.source_check(source, payload=True)
        run = Path(source["run_dir"])
        rewards = rewards_from_npz(run / "train_metrics.npz", 500)
        prefix_episode, prefix_score, _ = best_index(rewards)
        payload = None
        if source["model_id"] == "algorithm2":
            payload = torch.load(source["checkpoint"]["path"], map_location="cpu", weights_only=False)
        full_state = payload is not None and all(k in payload for k in ("agents", "learner", "replay", "environment", "metrics", "rng"))
        prefix_candidate = run / "checkpoints/periodic" / f"episode_{prefix_episode:06d}_actor.pt"
        b1_path = None
        if source["model_id"] != "algorithm2":
            spec = dc.MODELS[source["model_id"]]
            name = f"mappo_budget_{spec['structure']}_{spec['variant']}_p05_n04_g25_seed{source['training_seed']:02d}_b250k"
            b1_path = Path(m["b1_root"]) / name
            complete = dc.read(b1_path / "COMPLETE.json")
            dc.require(complete["status"] == "complete" and complete["episodes"] == 2500 and
                       complete["reproduction_git_commit"] == "09c38b48e98bae66f2c4ade3e3dee2f7b5d44a54" and
                       (b1_path / "policy_final.pt").is_file(), "historical B1250k final missing/mismatch")
            dc.require(not (b1_path / BEST).exists(), "unexpected historical training100 best: review before retraining")
            old_policy = torch.load(b1_path / "policy_final.pt", map_location="cpu", weights_only=False)
            old_cfg = config_from_dict(old_policy["config"])
            dc.require(old_policy.get("episode") == 2500 and old_cfg.episodes == 2500 and old_cfg.seed == source["training_seed"] and
                       old_cfg.mappo_variant == spec["variant"] and bool(old_cfg.mappo_actor_sharing) == (spec["structure"] == "shared") and
                       old_cfg.canonical_hash() == complete["config_hash"], "B1 final policy/config mismatch")
        rows.append({"model_id": source["model_id"], "training_seed": source["training_seed"], "source_run": str(run),
                     "prefix_100_best_episode": prefix_episode, "prefix_100_best_score": prefix_score,
                     "full_payload_keys_present_not_equivalence_proof": full_state,
                     "prefix_periodic_candidate_exists_not_verified": prefix_candidate.exists(),
                     "candidate_path": str(prefix_candidate), "legacy_complete_run": True,
                     "historical_best_semantics": "validation_or_unknown_not_training100",
                     "reusable_metadata_curves_final_reference": True, "b1_reference": str(b1_path) if b1_path else None,
                     "reuse_as_new_best": False, "equivalent_resume_verified": False,
                     "decision": "FROM_SCRATCH", "reason": "No verified all-candidate training100 artifact or equivalent fork-resume; do not rename old final/best"})
    result = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"], "source_models": 24,
              "fresh_mappo_runs": 18, "fresh_maddpg_runs": 6, "new_training_env_steps": 6000000,
              "legacy_resume_allowed": False, "rows": rows}
    dc.write(Path(batch) / "protocol/historical_audit.json", result)
    return result


def approve(batch):
    m, _ = load(batch)
    report = dc.read(Path(batch) / "protocol/historical_audit.json")
    dc.require(report["technical_status"] == "PASS" and report["implementation_commit"] == m["implementation_commit"], "remote audit required")
    result = {"implementation_commit": m["implementation_commit"], "decision": "24_FROM_SCRATCH", "reuse": 0,
              "approved_runs": [cell(i) for i in range(24)], "evaluations": 0,
              "execution_mode": "direct24", "pilot_required": False, "max_concurrent_gpus": 6}
    dc.write(Path(batch) / "protocol/execution_decision.json", result)
    return result


def attempt_run(batch, i, attempt):
    dc.require(re.fullmatch(r"a[0-9]{2,4}", attempt), "unsafe attempt")
    item = cell(i)
    return Path(batch) / "training" / attempt / "runs" / f"train100_{item['model_id']}_seed{item['training_seed']:02d}"


def verify_run(batch, i, run):
    m, ss = load(batch)
    run = Path(run).resolve()
    item = cell(i)
    dc.require(run.parent.parent.parent == Path(batch).resolve() / "training", "run outside new training root")
    source = next(s for s in ss if (s["model_id"], s["training_seed"]) == (item["model_id"], item["training_seed"]))
    check_source(source)
    cfg, expected_delta = derive(source, run.parent)
    complete = dc.read(run / "COMPLETE.json")
    dc.require(complete["reproduction_git_commit"] == m["implementation_commit"] and
               cfg.canonical_hash() == complete["config_hash"] and complete["episodes"] == 2500,
               "run config/commit/budget mismatch")
    rewards = rewards_from_npz(run / "train_metrics.npz", EPISODES)
    episode, score, scores = best_index(rewards)
    with (run / "training_reward_selection.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dc.require(len(rows) == 2500 and [int(r["episode"]) for r in rows] == list(range(1, 2501)), "selection ledger coverage")
    dc.require(all(float(r["episode_reward"]) == value for r, value in zip(rows, rewards)) and
               all(r["rolling100"] == "" for r in rows[:99]) and
               all(float(r["rolling100"]) == s for r, s in zip(rows[99:], scores)), "training score recomputation differs")
    maximum = None
    timing = "after_episode_before_pending_ppo_update" if cfg.algorithm == "mappo" else "after_episode_online_actor_updates"
    for e, row in enumerate(rows, 1):
        value = scores[e - 100] if e >= 100 else None
        better = value is not None and (maximum is None or value > maximum)
        dc.require(row["new_best"] == str(better), "incorrect strict-improvement ledger")
        if better: maximum = value
        before = (e - 1) // 5 if cfg.algorithm == "mappo" else max(0, (e - 1) * 100 - cfg.batch_size + 1) // cfg.policy_delay
        after = before if cfg.algorithm == "mappo" else max(0, e * 100 - cfg.batch_size + 1) // cfg.policy_delay
        dc.require(int(row["policy_version_before_episode"]) == before and int(row["policy_version_saved"]) == after,
                   "training policy-version ledger mismatch")
        dc.require(row["save_timing"] == timing, "training save-timing ledger mismatch")
    _, best = load_training_policy(run / BEST)
    _, final = load_training_policy(run / "policy_final.pt")
    load_training_actors(run / BEST)
    load_training_actors(run / "policy_final.pt")
    selection = best["training_reward_selection"]
    dc.require(selection["selected_episode"] == episode and selection["score"] == score and
               selection["window_start"] == episode - 99 and selection["window_end"] == episode, "wrong best window")
    dc.require(selection["window_policy_version_min"] == int(rows[episode - 100]["policy_version_before_episode"]) and
               selection["window_policy_version_max"] == int(rows[episode - 1]["policy_version_saved"]), "wrong window policy-version bounds")
    if cfg.algorithm == "mappo":
        dc.require(complete["update_count"] == 500 and complete["environment_steps"] == 250000 and
                   best["policy_version"] == (episode - 1) // 5 and final["policy_version"] == 500 and
                   selection["save_timing"] == "after_episode_before_pending_ppo_update", "MAPPO update/save order mismatch")
        dc.require(complete["actor_optimizer_step_count"] == (5000 if cfg.mappo_actor_sharing else 25000), "actor update count mismatch")
    else:
        dc.require(best["policy_version"] == max(0, episode * 100 - cfg.batch_size + 1) // cfg.policy_delay and
                   final["policy_version"] == max(0, 250000 - cfg.batch_size + 1) // cfg.policy_delay and
                   selection["save_timing"] == "after_episode_online_actor_updates", "MADDPG update/save order mismatch")
    final_score = float(np.mean(rewards[-100:], dtype=np.float64))
    return {**item, "technical_status": "PASS", "run_dir": str(run), "implementation_commit": m["implementation_commit"],
            "best_episode": episode, "window_start": episode - 99, "window_end": episode,
            "best_training100_score": score, "last100_training_score": final_score,
            "best_policy_version": best["policy_version"], "final_policy_version": final["policy_version"],
            "save_timing": selection["save_timing"], "derivation": expected_delta,
            "best": dc.file_identity(run / BEST), "final": dc.file_identity(run / "policy_final.pt"), "from_scratch": True}


def train_run(batch, i, attempt="a01", reason=None):
    from aoi_v2x_reproduction.runtime.runner import train
    m, ss = load(batch)
    decision = dc.read(Path(batch) / "protocol/execution_decision.json")
    dc.require(decision["decision"] == "24_FROM_SCRATCH" and decision["implementation_commit"] == m["implementation_commit"], "audit/approval required")
    direct = decision.get("execution_mode") == "direct24" and decision.get("pilot_required") is False
    if not direct and i not in PILOTS:
        dc.require(dc.read(Path(batch) / "protocol/pilot_verification.json")["technical_status"] == "PASS", "pilot PASS required")
    record = Path(batch) / "records" / f"train_{i:02d}.json"
    if record.exists():
        saved = dc.read(record)
        checked = verify_run(batch, i, saved["run_dir"])
        dc.require(all(checked[k] == saved[k] for k in checked), "accepted run identity changed")
        return {"skipped_complete": True, **saved}
    dc.require(attempt == "a01" or reason, "new attempt requires reason")
    # COMPLETE but missing post-check record: accept existing, never train again.
    possible = list((Path(batch) / "training").glob(f"a*/runs/train100_{cell(i)['model_id']}_seed{cell(i)['training_seed']:02d}/COMPLETE.json"))
    dc.require(len(possible) <= 1, "duplicate completed runs")
    if possible:
        result = verify_run(batch, i, possible[0].parent)
    else:
        item = cell(i)
        source = next(s for s in ss if (s["model_id"], s["training_seed"]) == (item["model_id"], item["training_seed"]))
        check_source(source)
        run = attempt_run(batch, i, attempt)
        dc.require(not run.exists(), "partial attempt preserved; use a fresh attempt, no resume")
        cfg, _ = derive(source, run.parent)
        dc.write(Path(batch) / "records" / f"started_{i:02d}_{attempt}.json", {**item, "attempt": attempt, "reason": reason,
                 "implementation_commit": m["implementation_commit"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"), "from_scratch": True, "resume": False})
        train(cfg, training_reward_best=True)
        result = verify_run(batch, i, run)
    code_check(m["implementation_commit"])
    dc.write(record, result)
    return result


def pilot_check(batch):
    m, _ = load(batch)
    for i in PILOTS:
        r = dc.read(Path(batch) / "records" / f"train_{i:02d}.json")
        checked = verify_run(batch, i, r["run_dir"])
        dc.require(all(checked[k] == r[k] for k in checked), "accepted pilot artifact identity changed")
    result = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"], "run_ids": list(PILOTS),
              "accepted_runs": 4, "environment_steps": 1000000, "part_of_formal24": True}
    dc.write(Path(batch) / "protocol/pilot_verification.json", result)
    return result


def summary(batch, attempt="a01"):
    m, _ = load(batch)
    dc.require(re.fullmatch(r"a[0-9]{2,4}", attempt), "unsafe summary attempt")
    out = Path(batch) / "analysis" / attempt
    out.mkdir(parents=True, exist_ok=False)
    try:
        records, curves = [], []
        for i in range(24):
            record = dc.read(Path(batch) / "records" / f"train_{i:02d}.json")
            checked = verify_run(batch, i, record["run_dir"])
            dc.require(all(checked[k] == record[k] for k in checked), "accepted artifact identity changed")
            records.append(checked)
            with (Path(record["run_dir"]) / "training_reward_selection.csv").open(newline="", encoding="utf-8") as f:
                curves.extend({**cell(i), **r, "budget_env_steps": 250000, "environment_steps": int(r["episode"]) * 100} for r in csv.DictReader(f))
        from analysis.evaluate_frozen_deployment import csv_write
        csv_write(out / "selected_policies.csv", records)
        csv_write(out / "training_episode_curves.csv", curves)
        (out / "README.md").write_text(
            "# Per-seed training100 selection\n\n24 fresh runs, 2500 episodes x100slots; 6M training environment steps, zero policy evaluations.\n"
            "R=mean_slot_platoon(task1+task2+global), float64 over logged float32 streams. S=mean last100 R; start100; earliest tie.\n"
            "Six seeds retained per model. MADDPG noise0/0.3 use the same six trained models; training exploration0.3.\n"
            "MAPPO best captured before pending episode-end PPO update; final after update500.\n"
            "MADDPG best captured after online updates of the episode. Actor update count is policy version.\n"
            "Windows span behavior versions; these scores are not frozen-policy validation or test scores. No causal performance claim.\n"
            "Best/final separate portable actors, not resumable; historical validation best is not reused.\n"
            "Old configs/curves/final references remain read-only. NPZ and policies stay on the execution host.\n", encoding="utf-8")
        result = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"], "accepted_runs": 24,
                  "from_scratch_runs": 24, "training_env_steps": 6000000, "episode_rows": len(curves),
                  "best_policies": 24, "final_policies": 24, "new_evaluations": 0, "claim_support": "NOT_ASSESSED"}
        dc.require(len(curves) == 60000, "training curve coverage")
        dc.write(out / "verification.json", result)
        return result
    except Exception as exc:
        dc.write(out / "verification.json", {"technical_status": "FAIL", "error": str(exc)})
        raise


def export(batch, attempt="a01"):
    m, _ = load(batch)
    dc.require(re.fullmatch(r"a[0-9]{2,4}", attempt), "unsafe export attempt")
    out = Path(batch) / "analysis" / attempt
    dc.require(dc.read(out / "verification.json")["technical_status"] == "PASS", "summary PASS required")
    runtime_path = out / "runtime_report.json"
    if not runtime_path.exists() and m.get("source_mode") == "bundled_config_fresh":
        processes = [dc.read(p) for p in sorted((Path(batch) / "records").glob("local_process_*.json"))]
        dc.write(runtime_path, {"implementation_commit": m["implementation_commit"], "backend": "local_python_serial",
                 "processes": processes, "MaxRSS": None, "MaxRSS_note": "not measured; no sacct on local host",
                 "raw_training_bytes": sum(p.stat().st_size for p in (Path(batch) / "training").rglob("*") if p.is_file()),
                 "historical_payload_audit": m["historical_payload_audit"], "new_evaluations": 0})
    elif not runtime_path.exists():
        jobs = [dc.read(p) for p in sorted((Path(batch) / "protocol").glob("submission_*.json"))]
        dc.require(jobs, "submission records required for runtime accounting")
        args = ["sacct", "-P", "-j", ",".join(str(j["job_id"]) for j in jobs),
                "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,ElapsedRaw,MaxRSS,AllocCPUS,ReqMem"]
        accounting = subprocess.check_output(args, text=True)
        rows = [{k: (None if v == "" else v) for k, v in row.items() if k}
                for row in csv.DictReader(io.StringIO(accounting), delimiter="|")]
        from analysis.evaluate_frozen_deployment import csv_write
        csv_write(out / "sacct.csv", rows)
        dc.write(runtime_path, {"implementation_commit": m["implementation_commit"], "jobs": jobs, "command": args,
                 "sacct": rows, "raw_training_bytes": sum(p.stat().st_size for p in (Path(batch) / "training").rglob("*") if p.is_file()),
                 "MaxRSS_missing_is_null": True, "packaging_job_id": os.environ.get("SLURM_JOB_ID"),
                 "packaging_state_at_snapshot": "RUNNING; verify final exit separately", "new_evaluations": 0})
    path = out / "core_evidence.zip"
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for root in (Path(batch) / "protocol", Path(batch) / "records", out):
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix in {".json", ".csv", ".md"}:
                    z.write(p, p.relative_to(batch).as_posix())
        for p in sorted((Path(batch) / "training").glob("a*/runs/*")):
            if (p / "COMPLETE.json").exists():
                for name in ("COMPLETE.json", "provenance.json", "config.resolved.json"):
                    z.write(p / name, (p / name).relative_to(batch).as_posix())
        for folder in ("slurm_logs", "logs"):
            for p in sorted((Path(batch) / folder).glob("*")):
                if p.is_file() and p.suffix in {".out", ".err"}:
                    with p.open("rb") as f:
                        f.seek(max(0, p.stat().st_size - 16384))
                        z.writestr("log_excerpts/" + folder + "/" + p.name, f.read())
    verify_archive(path, attempt)
    return {"archive": str(path.resolve()), "bytes": path.stat().st_size, "zip_integrity": "PASS"}


def verify_archive(path, attempt="a01"):
    with zipfile.ZipFile(path) as z:
        dc.require(z.testzip() is None, "ZIP integrity failure")
        result = json.loads(z.read(f"analysis/{attempt}/verification.json"))
        dc.require(result["accepted_runs"] == 24 and result["technical_status"] == "PASS", "ZIP readback failed")
        dc.require(len(list(csv.DictReader(io.StringIO(z.read(f"analysis/{attempt}/selected_policies.csv").decode("utf-8-sig"))))) == 24 and
                   len(list(csv.DictReader(io.StringIO(z.read(f"analysis/{attempt}/training_episode_curves.csv").decode("utf-8-sig"))))) == 60000,
                   "ZIP CSV readback counts")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init", "init-local", "audit", "approve-fresh", "train", "verify", "pilot-check", "summary", "export"))
    p.add_argument("--batch-root", type=Path)
    p.add_argument("--study-root", type=Path)
    p.add_argument("--batch-id")
    p.add_argument("--old-batch", type=Path)
    p.add_argument("--b1-runs", type=Path)
    p.add_argument("--id", type=int)
    p.add_argument("--attempt", default="a01")
    p.add_argument("--reason")
    a = p.parse_args(argv)
    if a.command == "init-local":
        dc.require(a.study_root is not None and a.batch_id, "--study-root and --batch-id required")
        r = init_local(a.study_root, a.batch_id)
    elif a.command == "init":
        dc.require(all((a.study_root, a.batch_id, a.old_batch, a.b1_runs)), "init paths required")
        r = init(a.study_root, a.batch_id, a.old_batch, a.b1_runs)
    else:
        dc.require(a.batch_root is not None, "--batch-root required")
        if a.command in {"train", "verify"}: dc.require(a.id is not None, "--id required")
        if a.command == "audit": r = audit(a.batch_root)
        elif a.command == "approve-fresh": r = approve(a.batch_root)
        elif a.command == "train": r = train_run(a.batch_root, a.id, a.attempt, a.reason)
        elif a.command == "verify": r = verify_run(a.batch_root, a.id, attempt_run(a.batch_root, a.id, a.attempt))
        elif a.command == "pilot-check": r = pilot_check(a.batch_root)
        elif a.command == "summary": r = summary(a.batch_root, a.attempt)
        else: r = export(a.batch_root, a.attempt)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
