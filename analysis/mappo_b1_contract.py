"""B1 Tier 1: source-derived 250k runs, immutable batch and attempt records.

Only train/evaluate commands instantiate the existing runtime. All checks are
metadata/array checks; historical policies are only loaded during preflight.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aoi_v2x_reproduction.config import config_from_dict
from analysis.mappo_e5_contract import config_differences, independent_run_name, shared_run_name

BASELINE = "d6121e876777841dc3d63800f7fc62ebb8fe9b41"
CONDITIONS = (("independent", "combined"), ("shared", "combined"),
              ("independent", "tdec"), ("shared", "tdec"))
WORLDS = {"development": list(range(213, 219)), "e3_reused": list(range(303, 309))}
PILOTS = (0, 6, 12, 18)
RELATIVE_ROOT = Path("B1-budget-sensitivity/P5_N4_gap25")
ARRAY_FIELDS = ("aoi_ms", "success", "remaining_demand", "reset_event", "executed_power_dbm",
                "reward_global", "reward_task1", "reward_task2")
TRAIN_FIELDS = ("aoi_ms", "success", "remaining_demand", "rb", "mode", "power_dbm",
                "global_step", "task1_step", "task2_step")
TRAIN_PROTOCOL = {"episodes": 2500, "slots_per_episode": 100, "environment_steps": 250000,
                  "ppo_updates": 500, "ppo_epochs": 10, "rollout_episodes": 5, "from_scratch": True,
                  "shared_actor_adam_steps": 5000, "independent_total_actor_adam_steps": 25000,
                  "per_actor_adam_steps": 5000}
EVAL_PROTOCOL = {"policy_episode": 2500, "action_mode": "stochastic", "arm": "baseline",
                 "episode_protocol": "sequential_warm", "world_cold_starts": 1,
                 "warmup_episodes": 5, "scored_episodes": 100, "slots_per_episode": 100,
                 "agent_shape": [6, 100, 100, 5], "policy_rng_tag": 0x4D415050,
                 "policy_rng_rule": "SeedSequence([training_seed, world, tag]).generate_state(1,uint32)[0]"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def code_audit():
    require(git("branch", "--show-current") == "research/mappo", "B1 requires research/mappo")
    require(not git("status", "--porcelain", "--untracked-files=all"), "B1 requires a clean checkout")
    paths = git("diff", "--name-only", BASELINE, "HEAD", "--", "aoi_v2x_reproduction",
                "analysis/evaluate_mappo_feasibility_reward.py").splitlines()
    require(not paths, f"Training/evaluator drift from audited baseline: {paths}")
    return {"implementation_commit": git("rev-parse", "HEAD"), "baseline_commit": BASELINE,
            "algorithm_and_evaluator_changed_paths": paths}


def cell(i):
    require(0 <= i < 24, "run id outside 0..23")
    structure, variant = CONDITIONS[i // 6]
    seed = 8 + i % 6
    old_name = independent_run_name(variant, seed) if structure == "independent" else shared_run_name(variant, seed)
    source_root = ("existing-evidence/tdec-ab-v1" if structure == "independent" else
                   "E5-sharing-critic" if variant == "combined" else "existing-evidence/shared-actor-v1")
    source_commit = ("f0226ab33c165c5e841e4ca85c7878d6a8d9f7f2" if structure == "independent" else
                     "b66278991ceba8c86c9141e23cd17250f6e8aa31" if variant == "combined" else
                     "81c6a59583c4bd040daf91179f19d9ce4d1f37fe")
    return {"run_id": i, "actor_structure": structure, "value_configuration": variant,
            "training_seed": seed, "budget_env_steps": 250000,
            "run_name": f"mappo_budget_{structure}_{variant}_p05_n04_g25_seed{seed:02d}_b250k",
            "historical_run_name": old_name, "historical_training_commit": source_commit,
            "historical_training_relative": f"{source_root}/P5_N4_gap25/training/runs/{old_name}"}


def eval_cell(i):
    require(0 <= i < 48, "evaluation id outside 0..47")
    return {"eval_id": i, "run_id": i // 2, "world_set": tuple(WORLDS)[i % 2]}


def eval_name(world_set):
    worlds = "-".join(map(str, WORLDS[world_set]))
    return f"eval_validation_policy_final_stochastic_sequential_warm_warm5_s{worlds}_ep100_feasibility_reward_arm_baseline"


def historical_eval(study, item, world_set):
    root = ("E3-locked-test" if world_set == "e3_reused" else "E5-sharing-critic"
            if item["value_configuration"] == "combined" else "existing-evidence/shared-actor-v1")
    return Path(study) / root / "P5_N4_gap25/evaluations" / item["historical_run_name"] / eval_name(world_set)


def batch_path(study, batch_id):
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id)), "unsafe batch id")
    study = Path(study).resolve()
    path = study / RELATIVE_ROOT / batch_id
    require(path.resolve().is_relative_to(study / RELATIVE_ROOT), "batch escapes B1 root")
    return path


def check_batch(batch):
    batch = Path(batch).resolve()
    manifest = read(batch / "protocol/manifest.json")
    require(batch == batch_path(manifest["study_root"], manifest["batch_id"]), "batch path mismatch")
    require(manifest["code_audit"] == code_audit(), "batch implementation commit changed")
    require(manifest["world_sets"] == WORLDS and manifest["runs"] == [cell(i) for i in range(24)]
            and manifest["evaluations"] == [eval_cell(i) for i in range(48)], "manifest protocol drift")
    require(manifest["training_protocol"] == TRAIN_PROTOCOL and manifest["evaluation_protocol"] == EVAL_PROTOCOL,
            "manifest training/evaluation protocol drift")
    require(read(batch / "protocol/preflight.json")["technical_status"] == "PASS", "preflight not PASS")
    return manifest


def source_config(raw, item):
    config = config_from_dict(raw)
    reconstruction = config_differences(raw, config.to_dict())
    require(set(reconstruction) <= {"mappo_actor_sharing"}, f"historical config reconstruction drift: {reconstruction}")
    expected = {"algorithm": "mappo", "episodes": 500, "steps_per_episode": 100,
                "seed": item["training_seed"], "mappo_variant": item["value_configuration"],
                "mappo_actor_sharing": item["actor_structure"] == "shared", "n_rb": 3,
                "mappo_rollout_episodes": 5, "mappo_ppo_epochs": 10, "mappo_num_minibatches": 1,
                "checkpoint_mode": "policy_only", "diagnostics": True, "device": "cuda:0",
                "is_formal_result": False, "run_name": item["historical_run_name"],
                "eval_protocol": "sequential_warm", "eval_warmup_episodes": 5,
                "mappo_actor_lr": .0005, "mappo_critic_lr": .0005,
                "mappo_entropy_coef_rb": .02, "mappo_entropy_coef_mode": .02, "mappo_entropy_coef_power": .002}
    for key, value in expected.items():
        actual = bool(getattr(config, key)) if key == "mappo_actor_sharing" else getattr(config, key)
        require(actual == value, f"source config {item['run_id']}/{key}: {actual!r} != {value!r}")
    require(config.scenario.id == "p05_n04_g25" and config.number_agents == 5, "source scenario")
    return config


def derive(raw, item, output_root):
    parent = source_config(raw, item)
    derived = copy.deepcopy(parent)
    derived.episodes = 2500
    derived.run_name = item["run_name"]
    derived.output_root = str(Path(output_root).resolve())
    # Historical omission/None has false semantics. Preserve the existing hash
    # compatibility path; do not insert current default hyperparameters.
    if derived.mappo_actor_sharing is None:
        derived.mappo_actor_sharing = False
    differences = config_differences(parent.to_dict(), derived.to_dict())
    require(set(differences) <= {"episodes", "run_name", "output_root", "mappo_actor_sharing"}, "config derivation drift")
    require(bool(parent.mappo_actor_sharing) == bool(derived.mappo_actor_sharing), "sharing treatment changed")
    return derived, {"differences": differences, "source_sharing_raw": raw.get("mappo_actor_sharing"),
                     "source_sharing_present": "mappo_actor_sharing" in raw,
                     "effective_actor_sharing": bool(derived.mappo_actor_sharing)}


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def diagnostics_check(records, item, episodes, recorded):
    count = episodes // 5
    actors = 1 if item["actor_structure"] == "shared" else 5
    require(len(records) == count and finite(records), "diagnostics count/nonfinite")
    for u, row in enumerate(records, 1):
        for key, value in {"update": u, "episode": u * 5, "environment_steps": u * 500,
                           "rollout_steps": 500, "ppo_epochs": 10}.items():
            require(row.get(key) == value, f"diagnostic {u}/{key}")
        for key, value in {"actor_optimizer_steps_this_update": 10 * actors,
                           "actor_optimizer_step_count": u * 10 * actors,
                           "actor_network_count": actors}.items():
            require((not recorded and key not in row) or row.get(key) == value, f"diagnostic {u}/{key}")
    return {"ppo_updates": count, "actor_network_count": actors, "actor_optimizer_count": actors,
            "actor_adam_steps_total": count * 10 * actors, "actor_adam_steps_per_actor": count * 10,
            "actor_adam_count_source": "recorded" if recorded else "derived_updates_x_epochs_x_actors",
            "critic_optimizer_count": 3 if item["value_configuration"] == "tdec" else 1,
            "critic_adam_steps_per_optimizer": count * 10, "critic_count_source": "derived_from_unchanged_trainer",
            "agent_transition_records_per_actor": episodes * 100 * (5 if actors == 1 else 1)}


def policy_check(run):
    import torch
    from aoi_v2x_reproduction.runtime.runner import _validate_mappo_policy_artifact
    payload = torch.load(Path(run) / "policy_final.pt", map_location="cpu", weights_only=False)
    _validate_mappo_policy_artifact(Path(run) / "policy_final.pt", payload)


def eval_metadata(directory, run, item, world_set, episodes, expected_commit=None):
    done, prov, training = read(directory / "EVAL_COMPLETE.json"), read(directory / "provenance.json"), read(run / "COMPLETE.json")
    expected = {"status": "complete", "algorithm": "mappo", "mappo_variant": item["value_configuration"],
                "training_seed": item["training_seed"], "training_run_name": run.name, "scenario": "p05_n04_g25",
                "policy_name": "policy_final.pt", "policy_episode": episodes, "eval_seeds": WORLDS[world_set],
                "eval_episodes": 100, "eval_warmup_episodes": 5, "eval_protocol": "sequential_warm",
                "mappo_eval_mode": "stochastic", "intervention_arm": "baseline", "is_frozen_eval": True,
                "policy_parameters_unchanged": True, "reproduction_git_dirty": False,
                "policy_path_is_relative_to_eval": True, "external_action_noise_applicable": False,
                "intervention_db": 0., "aoi_cap_ms": 100., "eval_id": eval_name(world_set),
                "actor_network_count": 1 if item["actor_structure"] == "shared" else 5,
                "source_config_hash": training["config_hash"]}
    for key, value in expected.items():
        require(done.get(key) == value, f"{directory}/{key}: {done.get(key)!r} != {value!r}")
    require(bool(done.get("actor_sharing")) == (item["actor_structure"] == "shared"), "eval sharing")
    require((directory / done["policy"]).resolve() == (run / "policy_final.pt").resolve(), "eval policy source")
    for key in ("algorithm", "mappo_variant", "training_seed", "eval_seeds", "eval_episodes",
                "eval_warmup_episodes", "mappo_eval_mode", "intervention_arm", "policy", "reproduction_git_commit", "source_config_hash"):
        require(prov.get(key) == done.get(key), f"eval provenance/{key}")
    require(prov.get("reproduction_git_dirty") is False and isinstance(prov.get("created_at_utc"), str), "eval provenance status/time")
    require(bool(prov.get("actor_sharing")) == (item["actor_structure"] == "shared"), "provenance sharing")
    if episodes == 2500 or prov.get("training_source_commit") is not None:
        require(prov.get("training_source_commit") == training["reproduction_git_commit"], "training source commit")
    if expected_commit:
        require(done["reproduction_git_commit"] == expected_commit, "evaluation implementation commit")
    require((directory / "metrics.npz").is_file(), "missing evaluation NPZ")
    return done


def initialize(study, batch_id):
    audit = code_audit()
    batch = batch_path(study, batch_id)
    batch.mkdir(parents=True, exist_ok=False)
    for name in ("protocol", "configs", "records", "training", "evaluations", "analysis", "slurm_logs", "cache", "tmp"):
        (batch / name).mkdir()
    for name, relative in {"TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "TORCH_HOME": "cache/torch",
                           "CUDA_CACHE_PATH": "cache/cuda", "MPLCONFIGDIR": "cache/matplotlib"}.items():
        os.environ[name] = str(batch / relative)
    manifest = {"experiment": "B1-budget-sensitivity", "tier": 1, "batch_id": batch_id,
                "study_root": str(Path(study).resolve()), "created_at_utc": utc(), "code_audit": audit,
                "training_protocol": TRAIN_PROTOCOL, "evaluation_protocol": EVAL_PROTOCOL,
                "statistics": {"independent_replicates": "training seeds 8-13", "n": 6, "df": 5,
                               "t95": 2.570581835636314, "own_extrema": "pool sum/count over worlds per agent before max/min",
                               "contributions": ["le50", "mid50", "gt100", "boundary"],
                               "long_complete": "mid50+gt100", "automatic_claim_classification": False},
                "world_sets": WORLDS, "e3_role": "historically_locked_now_observed_reuse_not_unseen_confirmation",
                "runs": [cell(i) for i in range(24)], "evaluations": [eval_cell(i) for i in range(48)],
                "comparisons": {"C-a": "shared50k-independent250k", "C-b": "shared250k-independent250k",
                                "C-c": "independent250k-independent50k"},
                "new_training_runs": 24, "new_evaluation_cells": 48, "reused_training_runs": 24,
                "reused_evaluation_cells": 48, "claim_support": "NOT_ASSESSED"}
    write(batch / "protocol/manifest.json", manifest)
    try:
        sources = []
        for item in manifest["runs"]:
            run = Path(study).resolve() / item["historical_training_relative"]
            raw, complete = read(run / "config.resolved.json"), read(run / "COMPLETE.json")
            source_config(raw, item)
            require(complete.get("status") == "complete" and complete.get("episodes") == 500
                    and complete.get("environment_steps") == 50000 and complete.get("update_count") == 100,
                    f"historical completion {run}")
            require(complete.get("reproduction_git_commit") == item["historical_training_commit"], "historical training commit")
            policy_check(run)
            diagnostics_check(read(run / "learning_diagnostics.json"), item, 500, False)
            config, comparison = derive(raw, item, batch / "training/a01/runs")
            write(batch / "configs" / f"run_{item['run_id']:02d}_a01.json", config.to_dict())
            dirs = {world_set: historical_eval(study, item, world_set) for world_set in WORLDS}
            for world_set, directory in dirs.items():
                eval_metadata(directory, run, item, world_set, 500)
            sources.append({**item, "source_config": raw, "source_complete": complete,
                            "historical_run": str(run), "historical_evaluations": {k: str(v) for k, v in dirs.items()},
                            "derivation": comparison})
        require(len({s["historical_run"] for s in sources}) == 24 and
                len({p for s in sources for p in s["historical_evaluations"].values()}) == 48, "duplicate sources")
        write(batch / "protocol/source_inventory.json", sources)
        from analysis.summarize_mappo_b1_budget import historical_inputs
        historical_inputs(Path(study), sources)  # compact CSV coverage/accounting, no historical NPZ
        write(batch / "protocol/preflight.json", {"technical_status": "PASS", "claim_support": "NOT_ASSESSED",
              "historical_policies": 24, "historical_eval_cells": 48, "historical_npz_loaded": False, **audit})
    except Exception as exc:
        write(batch / "protocol/preflight.json", {"technical_status": "FAIL", "claim_support": "NOT_ASSESSED", "error": str(exc)})
        raise
    return {"batch_root": str(batch), "technical_status": "PASS"}


def attempt_path(batch, kind, identity, attempt):
    require(bool(re.fullmatch(r"a[0-9]{2,4}", attempt)), "attempt must be a01, a02, ...")
    if kind == "train":
        path = Path(batch) / "training" / attempt / "runs" / cell(identity)["run_name"]
    else:
        ec = eval_cell(identity)
        path = Path(batch) / "evaluations" / ec["world_set"] / attempt / cell(ec["run_id"])["run_name"] / eval_name(ec["world_set"])
    require(path.resolve().is_relative_to(Path(batch).resolve()), "attempt path escapes batch")
    return path


def record_path(batch, kind, identity):
    return Path(batch) / "records" / f"{kind}_{identity:02d}.json"


def check_training(batch, run_id, attempt, record=False):
    manifest = check_batch(batch)
    item = cell(run_id)
    run = attempt_path(batch, "train", run_id, attempt)
    source = read(Path(batch) / "protocol/source_inventory.json")[run_id]
    require(read(Path(source["historical_run"]) / "config.resolved.json") == source["source_config"], "historical config drift")
    expected, comparison = derive(source["source_config"], item, run.parent)
    config = config_from_dict(read(run / "config.resolved.json"))
    require(config.to_dict() == expected.to_dict(), "new training config drift")
    done = read(run / "COMPLETE.json")
    actors = 1 if item["actor_structure"] == "shared" else 5
    for key, value in {"status": "complete", "episodes": 2500, "steps_per_episode": 100, "environment_steps": 250000,
                       "update_count": 500, "actor_network_count": actors, "actor_optimizer_step_count": 5000 * actors,
                       "actor_sharing": actors == 1, "mappo_variant": item["value_configuration"],
                       "reproduction_git_commit": manifest["code_audit"]["implementation_commit"], "reproduction_git_dirty": False}.items():
        require(done.get(key) == value, f"new COMPLETE/{key}")
    counts = diagnostics_check(read(run / "learning_diagnostics.json"), item, 2500, True)
    policy_check(run)
    require(not (run / "checkpoints").exists(), "unexpected intermediate checkpoints")
    with np.load(run / "train_metrics.npz", allow_pickle=False) as data:
        for name in TRAIN_FIELDS:
            array = data[name]
            require(array.shape == ((2500, 100) if name == "global_step" else (2500, 100, 5))
                    and np.isfinite(array).all(), f"train metrics/{name}")
    payload = {"technical_status": "PASS", "claim_support": "NOT_ASSESSED", **item, **counts,
               "attempt": attempt, "run_dir": str(run), "policy_episode": 2500, "environment_steps": 250000,
               "implementation_commit": manifest["code_audit"]["implementation_commit"], "derivation": comparison}
    if record:
        write(record_path(batch, "train", run_id), payload)
    else:
        require(read(record_path(batch, "train", run_id)) == payload, "training acceptance record mismatch")
    return payload


def evaluation_arrays(directory):
    done = read(Path(directory) / "EVAL_COMPLETE.json")
    with np.load(Path(directory) / "metrics.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in ARRAY_FIELDS}
    for name, a in arrays.items():
        axes = ["eval_seed", "scored_episode", "slot"] + ([] if name == "reward_global" else ["agent"])
        require(a.shape == ((6, 100, 100) if name == "reward_global" else (6, 100, 100, 5))
                and np.isfinite(a).all() and done["raw_metric_axes"][name] == axes, f"eval array/{name}")
    aoi = arrays["aoi_ms"].astype(np.float64)
    require(np.all((aoi >= 1) & (aoi <= 100)) and np.all(np.abs(aoi - np.rint(aoi)) <= 1e-6), "AoI integer/cap")
    require(np.isin(arrays["reset_event"], [0, 1]).all(), "nonbinary reset")
    reset = arrays["reset_event"].astype(bool)
    require(np.array_equal(reset, np.isclose(aoi, 1, rtol=0, atol=1e-6)), "reset/AoI mismatch")
    flat, events = aoi.reshape(6, -1, 5), reset.reshape(6, -1, 5)
    require(np.array_equal(flat[:, 1:], np.where(events[:, 1:], 1, np.minimum(flat[:, :-1] + 1, 100))), "post-step AoI recurrence")
    require(np.isin(arrays["success"][:, :, -1, :], [0, 1]).all(), "nonbinary CAM endpoints")
    return arrays


def check_evaluation(batch, identity, attempt, record=False):
    manifest = check_batch(batch)
    ec = eval_cell(identity)
    train_record = read(record_path(batch, "train", ec["run_id"]))
    item, run = cell(ec["run_id"]), Path(train_record["run_dir"])
    directory = attempt_path(batch, "eval", identity, attempt)
    eval_metadata(directory, run, item, ec["world_set"], 2500, manifest["code_audit"]["implementation_commit"])
    evaluation_arrays(directory)
    payload = {"technical_status": "PASS", "claim_support": "NOT_ASSESSED", **ec, "attempt": attempt,
               "evaluation_dir": str(directory), "training_run_dir": str(run), "policy_episode": 2500,
               "implementation_commit": manifest["code_audit"]["implementation_commit"],
               "worlds": WORLDS[ec["world_set"]], "scored_agent_slots": 300000, "cam_agent_episode_events": 3000,
               "policy_rng_seeds": {str(w): int(np.random.SeedSequence([item["training_seed"], w, 0x4D415050]).generate_state(1, dtype=np.uint32)[0]) for w in WORLDS[ec["world_set"]]}}
    if record:
        write(record_path(batch, "eval", identity), payload)
    else:
        require(read(record_path(batch, "eval", identity)) == payload, "evaluation acceptance record mismatch")
    return payload


def execute(batch, kind, identity, attempt, reason):
    manifest = check_batch(batch)
    require(not record_path(batch, kind, identity).exists(), "already accepted; do not repeat")
    destination = attempt_path(batch, kind, identity, attempt)
    require(not destination.exists(), "attempt exists; preserve it and use a new attempt path")
    require(attempt == "a01" or bool(reason.strip()), "retry requires an explicit technical failure reason")
    # A completed artifact without a B1 record must be checked/recorded, not rerun.
    pattern = (f"training/*/runs/{cell(identity)['run_name']}/COMPLETE.json" if kind == "train" else
               f"evaluations/{eval_cell(identity)['world_set']}/*/{cell(eval_cell(identity)['run_id'])['run_name']}/{eval_name(eval_cell(identity)['world_set'])}/EVAL_COMPLETE.json")
    require(not list(Path(batch).glob(pattern)), "existing completed artifact: use check --record, not another run")
    write(Path(batch) / "records" / f"{kind}_{identity:02d}_{attempt}_started.json",
          {"created_at_utc": utc(), "reason": reason, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
           "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"), "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
           "destination": str(destination)})
    if kind == "train":
        source = read(Path(batch) / "protocol/source_inventory.json")[identity]
        config, _ = derive(source["source_config"], cell(identity), destination.parent)
        if attempt != "a01":
            write(Path(batch) / "configs" / f"run_{identity:02d}_{attempt}.json", config.to_dict())
        from aoi_v2x_reproduction.runtime.runner import train
        train(config)
        return check_training(batch, identity, attempt, True)
    ec = eval_cell(identity)
    tr = read(record_path(batch, "train", ec["run_id"]))
    check_training(batch, ec["run_id"], tr["attempt"])
    from analysis.evaluate_mappo_feasibility_reward import evaluate_feasibility_reward
    evaluate_feasibility_reward(Path(tr["run_dir"]) / "policy_final.pt", destination.parent.parent,
                                "baseline", device="cuda:0", eval_seeds=WORLDS[ec["world_set"]], eval_episodes=100)
    return check_evaluation(batch, identity, attempt, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "preflight", "train", "evaluate", "check-train", "check-eval"))
    parser.add_argument("--study-root", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--id", type=int)
    parser.add_argument("--attempt", default="a01")
    parser.add_argument("--reason", default="")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        require(args.study_root is not None and args.batch_id, "init requires study root and batch id")
        result = initialize(args.study_root, args.batch_id)
    else:
        require(args.batch_root is not None, "batch root required")
        if args.command == "preflight":
            manifest = check_batch(args.batch_root)
            result = {"technical_status": "PASS", "batch_root": str(args.batch_root),
                      "implementation_commit": manifest["code_audit"]["implementation_commit"]}
        else:
            require(args.id is not None, "id required")
            if args.command in ("train", "evaluate"):
                result = execute(args.batch_root, "train" if args.command == "train" else "eval", args.id, args.attempt, args.reason)
            else:
                result = (check_training if args.command == "check-train" else check_evaluation)(args.batch_root, args.id, args.attempt, args.record)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
