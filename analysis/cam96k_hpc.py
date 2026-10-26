"""HPC CAM96k:36 fresh runs -> six-policy selection ->180 learned+60 Uniform CPU units.

One process executes one Slurm array element. No local worker pools, Windows
batch dependencies, performance-based retry or automatic configuration fallback.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from analysis import cam96k_seed8_11 as old
from analysis import deployment_contract as dc
from analysis.evaluate_frozen_deployment import csv_read, csv_write, load_arrays
from aoi_v2x_reproduction.config import config_from_dict, validate_config
from aoi_v2x_reproduction.runtime import runner
from aoi_v2x_reproduction.runtime.full_training_checkpoint import verify_full_run, load_full_checkpoint
from aoi_v2x_reproduction.runtime.periodic_policy import verify_periodic_run, version_at
from aoi_v2x_reproduction.runtime.training_selection import BEST, RULE, load_training_policy, load_training_actors

MODELS = (*old.MODELS, "shared_no_tdec")
SEEDS = tuple(range(8, 14))
WORLDS = tuple(range(1001, 1031))
SCHEMA = "cam96k_hpc36_cpu240_v1"
ROOT = Path("CAM96k-HPC-six-methods/P5_N4_gap25")
LAUNCHER_FILES={"hpc/cam96k_hpc_common.sh","hpc/aoi_cam96k_hpc_train.sbatch",
                "hpc/aoi_cam96k_hpc_eval_cpu.sbatch","hpc/aoi_cam96k_hpc_analyze.sbatch","hpc/submit_cam96k_hpc.py"}
read, require = old.read, old.require


def identity(path):
    return {**old.identity(path), "mtime_ns": Path(path).stat().st_mtime_ns}


def write(path, data):
    """Immutable protocol/completion files. Failures retain their whole attempt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == data, f"refusing to replace existing record: {path}")
    else:
        with path.open("x", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)


def code_check(algorithm_commit, launcher_commit=None, smoke=False):
    head = dc.git("rev-parse", "HEAD")
    require(launcher_commit is None or head == launcher_commit, "running checkout differs from fixed launcher commit")
    require(dc.git("branch", "--show-current").startswith("research/HPC") or not dc.git("branch", "--show-current"),
            "use research/HPC or its fixed detached snapshot")
    if not smoke:
        require(not dc.git("status", "--porcelain", "--untracked-files=all"), "clean fixed checkout required")
    dc.git("merge-base", "--is-ancestor", algorithm_commit, head)
    paths = dc.git("diff", "--name-only", algorithm_commit, head).splitlines()
    require(set(paths) <= LAUNCHER_FILES, f"launcher descendant changed protected algorithm/protocol files: {paths}")
    return head


def cells():
    return [{"run_id": i, "model_id": m, "training_seed": s}
            for i, (m, s) in enumerate((m, s) for m in MODELS for s in SEEDS)]


def plan():
    return [{"unit_id": i, "model_id": m, "world": w, "repeat": 0}
            for i, (w, m) in enumerate((w, m) for w in WORLDS for m in MODELS)]


def all_plan():
    from analysis.cam96k_hpc_uniform import units
    return plan()+units()


def sources():
    """Accepted embedded CONFIGS only; historical paths remain reference strings."""
    from analysis.cam96k_training_pilot_v2 import sources as originals
    templates = originals()
    result = []
    for c in cells():
        base = "shared_tdec" if c["model_id"] == "shared_no_tdec" else c["model_id"]
        source = copy.deepcopy(next(s for s in templates if s["model_id"] == base and s["training_seed"] == 12))
        source.update(**c, source_training_seed=12, source_model_id=base)
        result.append(source)
    return result


def derive(source, parent, *, smoke=False, device=None):
    cfg = config_from_dict(source["config"])
    before = cfg.to_dict()
    cfg.seed = source["training_seed"]
    cfg.cam_bits = 96000
    cfg.run_name = f"cam96k_{source['model_id']}_s{cfg.seed}"
    cfg.output_root = str(Path(parent).resolve())
    if source["model_id"] == "shared_no_tdec":
        cfg.mappo_variant = "no_tdec"
    if smoke:
        cfg.smoke = True
        cfg.episodes = 6
        cfg.device = device or "cpu"
        # Development-only small networks/replay; NEVER used by a formal manifest.
        cfg.actor_hidden = [16, 8]
        cfg.local_critic_hidden = [16, 8]
        cfg.global_critic_hidden = [16, 8, 4]
        cfg.batch_size, cfg.replay_capacity = 4, 64
    delta = old.config_differences(before, cfg.to_dict())
    allowed = {"seed", "cam_bits", "run_name", "output_root"}
    if source["model_id"] == "shared_no_tdec": allowed.add("mappo_variant")
    if smoke: allowed |= {"smoke", "episodes", "device", "actor_hidden", "local_critic_hidden", "global_critic_hidden", "batch_size", "replay_capacity"}
    require(set(delta) <= allowed, "effective training derivation drift")
    require(cfg.seed in SEEDS and cfg.cam_bits == 96000 and cfg.number_agents == 5 and cfg.n_rb == 3 and
            cfg.steps_per_episode == 100 and cfg.slot_ms == 1 and cfg.scenario.id == dc.SCENARIO,
            "scenario/seed/budget mismatch")
    if cfg.profile == "reproduction_baseline":
        validate_config(cfg)
    else:
        # Accepted MADDPG configs carry their historical profile. The original
        # CAM96k runner consumes those exact objects; do not relabel provenance
        # or apply the new-config profile gate to already-audited templates.
        require(cfg.profile == "paper_faithful" and cfg.algorithm == "modified_maddpg_tdec",
                "unsupported historical source profile")
    return cfg, delta


def init(study, batch_id, algorithm_commit, smoke=False):
    launcher = code_check(algorithm_commit, smoke=smoke)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", batch_id) is not None and
            ("smoke" in batch_id.lower()) == smoke, "unsafe or mislabeled batch ID")
    batch = Path(study).resolve() / ROOT / batch_id
    require(not batch.exists(), "new unique batch required")
    ss = sources()
    for s in ss: derive(s, batch / "t/a01", smoke=smoke)
    m = {"schema": SCHEMA, "batch_root": str(batch), "algorithm_commit": algorithm_commit,
         "launcher_commit": launcher, "smoke": smoke, "sources": ss, "runs": cells(), "eval_plan": all_plan(),
         "training_env_steps": 36*250000, "training_episodes": 2500, "slots": 100, "cam_bits": 96000,
         "snapshot_episodes": [500, 1000, 1500, 2000, 2500], "eval_device": "cpu",
         "eval_protocol": "hpc_cpu_stochastic_repeat0_sequential_warm_5_100_v1",
         "world_role": old.WORLD_ROLE, "worlds": list(WORLDS), "unseen_confirmation": False,
         "initial_concurrency": 2, "max_concurrency": 8, "selection_rule": RULE,
         "cross_seed_ties": "lowest_training_seed", "all_training_from_scratch": True,
         "config_source_only": "embedded accepted32k templates, not lab policy payloads"}
    for n in ("protocol", "records", "t", "e", "logs", "locks", "tables", "tmp"): (batch/n).mkdir(parents=True)
    write(batch / "protocol/manifest.json", m)
    return m


def load(batch):
    batch = Path(batch).resolve(); m = read(batch / "protocol/manifest.json")
    code_check(m["algorithm_commit"], m["launcher_commit"], m["smoke"])
    require(m["schema"] == SCHEMA and m["batch_root"] == str(batch) and m["sources"] == sources() and
            m["runs"] == cells() and m["eval_plan"] == all_plan() and m["eval_device"] == "cpu" and
            m["worlds"] == list(WORLDS) and m["snapshot_episodes"] == [500,1000,1500,2000,2500] and
            m["initial_concurrency"] == 2 and m["max_concurrency"] == 8, "HPC protocol changed")
    require(m["training_episodes"]==2500 and m["training_env_steps"]==9000000 and m["slots"]==100 and m["cam_bits"]==96000 and
            m["selection_rule"]==RULE and m["cross_seed_ties"]=="lowest_training_seed" and m["all_training_from_scratch"] is True and
            m["eval_protocol"]=="hpc_cpu_stochastic_repeat0_sequential_warm_5_100_v1" and m["unseen_confirmation"] is False and
            m["world_role"]==old.WORLD_ROLE,"budget/selection/CPU RNG protocol changed")
    return m


def preflight(batch, storage_check=True):
    m = load(batch)
    versions = {n: importlib.metadata.version(n) for n in ("torch", "numpy", "scipy", "matplotlib", "PyYAML", "pytest")}
    require(sys.version_info[:2] == (3,10) and torch.__version__.split("+")[0] == "2.11.0" and
            torch.version.cuda == "12.6", "expected existing Python3.10 torch2.11.0 cu126; do not auto-upgrade")
    for line in (dc.REPO / "requirements.cuda.lock.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("=="); require(versions[name] == version, f"dependency mismatch: {name}")
    replay={}
    for s in m["sources"]:
        cfg,_=derive(s, Path(batch)/"t/a01", smoke=m["smoke"])
        if "maddpg" in s["model_id"]:
            replay[str(s["run_id"])]=cfg.replay_capacity*(4*(2*cfg.state_dim*5+cfg.action_dim*5+1+2*5)+1)
    result = {"technical_status": "PASS", "algorithm_commit": m["algorithm_commit"], "launcher_commit": m["launcher_commit"],
              "python": sys.executable, "versions": versions, "smoke": m["smoke"], "training_cells": 36,
              "cpu_eval_units": 240, "learning_eval_units":180,"random_eval_units":60,
              "external_lab_weights_required": False, "new_training": 0, "new_evaluation": 0}
    result.update(replay_bytes_per_maddpg_run=replay,
                  full_checkpoint_replay_bytes_lower_bound=sum(replay.values())*5,
                  disk_budget_note="lower bound excludes models, Adam, arrays and recovery copies; check real filesystem/account quota before submission")
    if storage_check:
        require(shutil.disk_usage(batch).free>sum(replay.values())*5,"insufficient free disk even for raw replay snapshot lower bound")
    write(Path(batch)/"protocol/preflight.json", result)
    return result


def run_path(batch, i, attempt):
    require(type(i) is int and 0 <= i < 36 and re.fullmatch(r"a\d{2,4}", attempt), "invalid training key")
    c = cells()[i]
    return Path(batch)/"t"/attempt/f"cam96k_{c['model_id']}_s{c['training_seed']}"


def verify_train(batch, i, run, deep=True):
    m = load(batch); run = Path(run).resolve()
    require(run == run_path(batch, i, run.parent.name), "wrong training source path")
    cfg, delta = derive(m["sources"][i], run.parent, smoke=m["smoke"])
    complete = read(run/"COMPLETE.json"); provenance = read(run/"provenance.json")
    require(complete["status"] == "complete" and complete["config_hash"] == cfg.canonical_hash() ==
            config_from_dict(read(run/"config.resolved.json")).canonical_hash() and
            complete["episodes"] == cfg.episodes and complete["environment_steps"] == cfg.episodes*100 and
            complete["reproduction_git_commit"] == provenance["reproduction_git_commit"] == m["launcher_commit"] and
            (m["smoke"] or not complete["reproduction_git_dirty"]), "training completion/config/budget mismatch")
    expected = list(range(2 if m["smoke"] else 500, cfg.episodes+1, 2 if m["smoke"] else 500))
    if deep:
        verify_full_run(run, expected, final_path=run/"policy_final.pt")
        verify_periodic_run(run, expected, final_path=run/"policy_final.pt")
    for folder in ("policies", "training_checkpoints"):
        require([r["episode"] for r in read(run/folder/"index.json")["records"]] == expected, "periodic coverage mismatch")
    if cfg.algorithm == "mappo":
        count = 1 if cfg.mappo_actor_sharing else 5
        require(complete["update_count"] == (cfg.episodes+4)//5 and complete["actor_optimizer_step_count"] ==
                ((cfg.episodes+4)//5)*cfg.mappo_ppo_epochs*count, "PPO/actor update count mismatch")
    else:
        count = 1 if cfg.maddpg_actor_sharing else 5
        learns = max(0, cfg.episodes*100-cfg.batch_size+1)
        require(complete["critic_learning_steps"] == learns and complete["policy_version"] == learns//cfg.policy_delay and
                complete["actor_optimizer_step_count"] == learns//cfg.policy_delay*count, "MADDPG counts mismatch")
    require(complete["actor_network_count"] == count, "actor structure mismatch")
    files = {n: identity(run/n) for n in ("config.resolved.json", "COMPLETE.json", "provenance.json",
              "train_metrics.npz", "policy_final.pt", "policies/index.json", "training_checkpoints/index.json")}
    for folder in ("policies", "training_checkpoints"):
        for record in read(run/folder/"index.json")["records"]:
            files[f"{folder}/{record['file']}"] = identity(run/folder/record["file"])
    record = {**cells()[i], "run_dir": str(run), "technical_status": "PASS", "smoke": m["smoke"],
              "from_scratch": True, "launcher_commit": m["launcher_commit"], "algorithm_commit": m["algorithm_commit"],
              "files": files, "derivation": delta, "parameter_counts": complete.get("parameter_counts"),
              "training_wall_seconds": complete.get("training_wall_seconds"), "actor_network_count": count}
    if not m["smoke"]:
        rewards = old.tc.rewards_from_npz(run/"train_metrics.npz", 2500)
        episode, score, scores = old.tc.best_index(rewards)
        ledger = csv_read(run/"training_reward_selection.csv")
        maximum = None
        require(len(ledger) == 2500, "training selector coverage mismatch")
        for e, row in enumerate(ledger, 1):
            v = scores[e-100] if e >= 100 else None
            better = v is not None and (maximum is None or v > maximum)
            require(int(row["episode"]) == e and float(row["episode_reward"]) == rewards[e-1] and
                    (row["rolling100"] == "" if v is None else float(row["rolling100"]) == v) and
                    row["new_best"] == str(better), "training-only rolling100 selector mismatch")
            if better: maximum = v
        _, payload = load_training_policy(run/BEST); load_training_actors(run/BEST)
        require(payload["episode"] == episode and payload["training_reward_selection"]["score"] == score and
                payload["policy_version"] == (version_at(cfg, episode-1) if cfg.algorithm == "mappo" else version_at(cfg,episode)),
                "best100 role/timing/version mismatch")
        load_training_policy(run/"policy_final.pt")
        record.update(best=identity(run/BEST), score=score, best_episode=episode, best_policy_version=payload["policy_version"])
        files[BEST] = record["best"]; files["training_reward_selection.csv"] = identity(run/"training_reward_selection.csv")
    return record


def verify_inputs(record):
    for k, saved in record["files"].items():
        require(identity(saved["path"]) == saved, f"accepted input changed: {k}")


def train_unit(batch, i, attempt="a01", reason=None, resume_episode=None):
    m = load(batch); preflight(batch,storage_check=False)
    saved = Path(batch)/f"records/train_{i}.json"
    with old.lock(Path(batch)/f"locks/train_{i}.lock", batch=str(batch), run_id=i):
        if saved.exists():
            record = read(saved); verify_inputs(record)
            require(all(record[k]==v for k,v in cells()[i].items()),"accepted training record key changed")
            return record
        require(attempt == "a01" or reason, "retry requires technical reason")
        run = run_path(batch, i, attempt)
        cfg, _ = derive(m["sources"][i], run.parent, smoke=m["smoke"])
        checkpoint = None
        if run.exists() and (run/"COMPLETE.json").exists():
            record = verify_train(batch, i, run)
            write(saved, record); return record  # Post-rollout failure: validate, never overwrite/retrain.
        if run.exists():
            require(bool(reason) and resume_episode is not None, "partial attempt retained: explicit reason and full boundary required")
            checkpoint = run/"training_checkpoints"/f"training_episode_{resume_episode:06d}.pt"
            load_full_checkpoint(checkpoint, config=cfg, expected_episode=resume_episode)
        else:
            require(resume_episode is None, "cannot resume absent attempt")
        started = time.monotonic()
        try:
            runner.train(cfg, training_reward_best=not m["smoke"], periodic_policy_interval=2 if m["smoke"] else 500,
                         periodic_policy_smoke=m["smoke"], full_checkpoint_interval=2 if m["smoke"] else 500,
                         full_checkpoint_smoke=m["smoke"], resume_training_checkpoint=checkpoint)
            record = verify_train(batch, i, run)
            write(saved, record)
            return record
        except BaseException as exc:
            write(Path(batch)/f"records/failure_train_{i}_{time.time_ns()}.json",
                  {"error": repr(exc), "attempt": attempt, "reason": reason, "elapsed": time.monotonic()-started,
                   "resume_episode": resume_episode, "run_dir": str(run)})
            raise


def choose(records):
    require(len(records) == 36 and {(r["model_id"],r["training_seed"]) for r in records} ==
            {(m,s) for m in MODELS for s in SEEDS}, "36 unique candidates required")
    require(all(np.isfinite(r["score"]) and r["technical_status"] == "PASS" for r in records), "invalid candidate")
    return [min((r for r in records if r["model_id"] == m), key=lambda r:(-r["score"],r["training_seed"])) for m in MODELS]


def select(batch):
    m = load(batch); preflight(batch,storage_check=False)
    require(not m["smoke"], "short smoke cannot select formal best100")
    with old.lock(Path(batch)/"locks/selection.lock", batch=str(batch)):
        records = [read(Path(batch)/f"records/train_{i}.json") for i in range(36)]
        for i,r in enumerate(records):
            require(all(r[k]==v for k,v in cells()[i].items()),"candidate record position/key mismatch")
            verify_inputs(r)
            require(verify_train(batch,r["run_id"],r["run_dir"],deep=False) == r, "candidate validation changed")
        result = {"schema": SCHEMA, "candidates": records, "selected": choose(records),
                  "selection_uses_evaluation": False, "evaluated_training_seeds_per_method": 1}
        write(Path(batch)/"records/selection.json", result)
    return result


def selection(batch):
    m = load(batch); p = read(Path(batch)/"records/selection.json")
    require(choose(p["candidates"]) == p["selected"] and not p["selection_uses_evaluation"], "sealed selection changed")
    # Full states and original metrics were deeply checked when sealing. Frozen
    # tasks do not reread hundreds of full replay buffers. Check every accepted
    # identity's stat and rehash the actually used six policy/config/COMPLETE files.
    for r in p["candidates"]:
        require(read(Path(batch)/f"records/train_{r['run_id']}.json") == r, "candidate record changed")
        for saved in r["files"].values():
            stat = Path(saved["path"]).stat()
            require(stat.st_size == saved["size"] and stat.st_mtime_ns == saved["mtime_ns"], "sealed source stat changed")
    for r in p["selected"]:
        for name in (BEST,"config.resolved.json","COMPLETE.json","provenance.json"):
            require(identity(r["files"][name]["path"]) == r["files"][name], "selected source bytes changed")
    require({r["launcher_commit"] for r in p["candidates"]} == {m["launcher_commit"]}, "candidate implementation mismatch")
    return p


def evaluate_native(source, dest, world, scored=100, *, periodic=False):
    """Existing frozen samplers, explicit CPU, no replay/learner construction."""
    policy = Path(source["best"]["path"])
    if "maddpg" not in source["model_id"]:
        from analysis.evaluate_mappo_zero_shot import evaluate_zero_shot
        return evaluate_zero_shot(policy, dest, dc.SCENARIO, "cpu", [world], scored,
                                  deployment=True, world_role=old.WORLD_ROLE, training_policy=not periodic, compact_output=True)
    from analysis.evaluate_maddpg_e1 import evaluate_maddpg_e1
    run = policy.parent.parent if periodic else policy.parent
    override = {"run_dir":run,"resolved_run_dir":run,"checkpoint":policy,"canonical_run_name":run.name,
                "historical_run_name":run.name,"algorithm_label":"algorithm2","is_symlink":False,
                "complete":read(run/"COMPLETE.json"),"manifest":{}}
    return evaluate_maddpg_e1("modified_maddpg_tdec",0.,source["training_seed"],run,dest,"cpu",[world],scored,5,
                             source["best_episode"],source_override=override,deployment=True,world_role=old.WORLD_ROLE,
                             target_scenario=dc.SCENARIO,training_policy=not periodic,periodic_policy=periodic)


def verify_eval(batch, i, dest):
    m = load(batch); unit = plan()[i]; proof = selection(batch)
    source = next(r for r in proof["selected"] if r["model_id"] == unit["model_id"])
    meta = read(dest/"native/EVAL_COMPLETE.json")
    expected = {"status":"complete", "training_seed":source["training_seed"], "eval_seeds":[unit["world"]],
                "eval_episodes":100,"eval_warmup_episodes":5,"eval_protocol":"sequential_warm","action_repeat":0,
                "training_source_commit":m["launcher_commit"],"reproduction_git_commit":m["launcher_commit"],
                "source_config_hash":read(Path(source["run_dir"])/"COMPLETE.json")["config_hash"],
                "cam_bits":96000,"actor_network_count":source["actor_network_count"],"checkpoint_role":"best_training100"}
    for k,v in expected.items(): require(meta.get(k) == v, f"native evaluation identity mismatch: {k}")
    actual_policy = ((dest/"native"/meta["policy"]).resolve() if "maddpg" not in source["model_id"]
                     else Path(meta["checkpoint"]).resolve())
    require(actual_policy == Path(source["best"]["path"]) and identity(actual_policy) == source["best"], "wrong selected policy bytes/path")
    require(meta.get("policy_version") == source["best_policy_version"], "frozen policy version mismatch")
    require(meta["world_role"] == old.WORLD_ROLE and meta["reproduction_git_dirty"] is False, "native protocol/provenance mismatch")
    if "maddpg" in source["model_id"]:
        require(meta["noise_std"] == 0 and meta["optimizer_steps"] == 0 and not meta["critic_constructed"] and
                not meta["optimizer_constructed"] and not meta["replay_constructed"] and meta["parameter_snapshot_unchanged"] and
                not meta["normalization_updated"] and meta["explore"] is False and
                meta["runtime_config"]["device"] == "cpu", "MADDPG frozen CPU contract mismatch")
    else:
        require(meta["policy_parameters_unchanged"] and not meta["critic_constructed"] and not meta["optimizer_constructed"] and
                not meta["rollout_constructed"] and meta["mappo_eval_mode"] == "stochastic" and meta["target_config"]["device"] == "cpu" and
                meta["policy_episode"] == source["best_episode"], "MAPPO frozen CPU contract mismatch")
    require(meta["exogenous_probes"][0]["world"] == unit["world"], "missing external RNG probes")
    arrays = load_arrays(dest/"native/metrics.npz")
    rows = old.reduce_payload(arrays,meta,{**unit,"training_seed":source["training_seed"]},100)
    return {"unit":unit,"selected_training_seed":source["training_seed"],"selected_checkpoint":source["best"],
            "rows":rows,"metrics":identity(dest/"native/metrics.npz"),"native_complete":identity(dest/"native/EVAL_COMPLETE.json")}


def eval_unit(batch,i,attempt="a01",reason=None):
    load(batch); preflight(batch,storage_check=False)
    require(type(i) is int and 0 <= i < 240 and re.fullmatch(r"a\d{2,4}",attempt), "invalid eval key")
    if i>=180:
        from analysis.cam96k_hpc_uniform import execute
        return execute(batch,i,attempt,reason)
    with old.lock(Path(batch)/f"locks/eval_{i}.lock",batch=str(batch),unit_id=i):
        saved = Path(batch)/f"records/eval_{i}.json"; dest = Path(batch)/"e"/str(i)/attempt
        if saved.exists():
            r = read(saved); require(verify_eval(batch,i,Path(r["attempt_dir"])) == {k:v for k,v in r.items() if k!="attempt_dir"},"accepted eval changed"); return r
        require(attempt=="a01" or reason,"technical retry reason required")
        if not (dest/"native/EVAL_COMPLETE.json").exists():
            require(not dest.exists(),"partial evaluation retained; inspect arrays or use new attempt")
            p=selection(batch); source=next(r for r in p["selected"] if r["model_id"]==plan()[i]["model_id"])
            evaluate_native(source,dest,plan()[i]["world"])
        result={**verify_eval(batch,i,dest),"attempt_dir":str(dest)}
        write(saved,result);return result


def summary(batch):
    m=load(batch); preflight(batch,storage_check=False); p=selection(batch); rows=[]
    for i in range(180):
        saved=read(Path(batch)/f"records/eval_{i}.json")
        require(verify_eval(batch,i,Path(saved["attempt_dir"])) == {k:v for k,v in saved.items() if k!="attempt_dir"},"evaluation changed")
        rows.extend(saved["rows"])
    require(len(rows)==900 and sum(r["slot_count"] for r in rows)==9000000 and
            sum(r["cam_episode_count"] for r in rows)==90000,"full matrix denominator mismatch")
    worlds=[]; policies=[]
    for s in p["selected"]:
        group=[r for r in rows if r["model_id"]==s["model_id"]]
        require({(r["world"],r["platoon"]) for r in group} == {(w,a) for w in WORLDS for a in range(5)},"method/world/platoon set mismatch")
        for w in WORLDS:
            g=[r for r in group if r["world"]==w]
            worlds.append({"model_id":s["model_id"],"world":w,"selected_training_seed":s["training_seed"],**old.aggregate(g),
                           "worst_aoi_ms":max(r["aoi_sum_ms"]/r["slot_count"] for r in g),
                           "worst_cam_rate":min(r["cam_deadline_success_count"]/r["cam_episode_count"] for r in g)})
        policies.append({"model_id":s["model_id"],"selected_training_seed":s["training_seed"],"best100_training_reward":s["score"],
                         "candidate_training_seeds":6,"evaluated_training_seeds":1,**old.aggregate(group)})
    candidates=[{k:r[k] for k in ("run_id","model_id","training_seed","score","best_episode","training_wall_seconds")}
                | {"selected":r in p["selected"],"parameter_counts_json":json.dumps(r["parameter_counts"],sort_keys=True)} for r in p["candidates"]]
    for name,table in (("world_platoon.csv",rows),("world_summary.csv",worlds),("selected_policy_summary.csv",policies),("training_candidates.csv",candidates)):
        file=Path(batch)/"tables"/name
        if not file.exists():csv_write(file,table)
        else:
            expected=[{k:"" if row.get(k) is None else str(row[k]) for k in sorted({k for r in table for k in r})} for row in table]
            require(csv_read(file) == expected, "existing summary table changed")
    from analysis.cam96k_hpc_uniform import summary as random_summary
    random_report=random_summary(batch)
    result={"technical_status":"PASS","claim_support":"NOT_ASSESSED","algorithm_commit":m["algorithm_commit"],
            "launcher_commit":m["launcher_commit"],"training_runs":36,"training_episodes":90000,"training_env_steps":9000000,
            "full_checkpoints":180,"periodic_policies":180,"best100":36,"final":36,"selected_policies":6,"eval_units":240,
            "learning_eval_units":180,"random_eval_units":60,"random_verification":random_report,
            "platoon_rows":900,"cam_deadlines":90000,"scored_agent_slots":9000000,"eval_device":"cpu",
            "world_role":old.WORLD_ROLE,"evaluated_training_seeds_per_method":1,"smoke":False}
    write(Path(batch)/"tables/verification.json",result)
    report=["# CAM96k HPC冻结评估", "", "技术验收PASS，claim_support=NOT_ASSESSED；不以性能方向决定验收。",
            "36个全新训练候选，六个训练回报选定策略；每方法实际评估训练seed数为1。",
            "world1001–1030为探索集，不是新的未见确认集。CPU与历史CUDA随机采样流不宣称相同。",
            "学习180单元/900车队行/90000截止事件/9000000计分车队时隙；Random60/300/30000/3000000独立计数。", ""]
    for row in policies:
        report.append(f"- {row['model_id']}: seed{row['selected_training_seed']}, best100训练reward={row['best100_training_reward']:.9g}; "
                      f"AoI={row['aoi_ms']:.9g}ms, CAM成功率={row['cam_rate']:.9g}, 平均功率={row['power_mw']:.9g}mW。")
    report += ["", "Uniform stream12/13不是训练seed，不训练、不选模，功率在dBm而不是mW空间均匀。",
               "NoTDec与TDec的Critic数量/参数量不同，见training_candidates.csv；这里不是参数量匹配实验。",
               "逐world/车队、失败条件剩余bit、payload交付比例与reward分量见CSV。无失败时为NA，不补0。"]
    path=Path(batch)/"tables/report_cn.md";text="\n".join(report)+"\n"
    if path.exists():require(path.read_text(encoding="utf-8")==text,"existing report changed")
    else:path.write_text(text,encoding="utf-8")
    return result


def export(batch):
    summary(batch); out=Path(batch)/"core_evidence.tar.gz"
    require(not out.exists(),"evidence archive already exists")
    runtime_report(batch)
    with tarfile.open(out,"x:gz") as tar:
        for directory in ("protocol","records","tables"):
            for path in sorted((Path(batch)/directory).rglob("*")):
                if path.is_file() and (path.suffix in {".json",".csv",".md",".txt",".patch"} or
                    path.is_relative_to(Path(batch)/"protocol/launcher_snapshot") or
                    path.is_relative_to(Path(batch)/"protocol/scheduler_snapshots")):tar.add(path,arcname=path.relative_to(batch))
        for path in sorted((Path(batch)/"logs").glob("sacct_*.csv")):tar.add(path,arcname=path.relative_to(batch))
    return {"archive":str(out),"raw_arrays_and_policies_included":False}


def runtime_report(batch):
    m=load(batch)
    previous=Path(batch)/"records/runtime_report.json"
    if previous.exists():return read(previous)
    submission=read(Path(batch)/"records/submission_chain.json") if (Path(batch)/"records/submission_chain.json").exists() else None
    accounting=[]
    if submission is not None:
        raw=subprocess.check_output(["sacct","-j",",".join(submission["jobs"].values()),"--parsable2",
                                     "--format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES"],text=True)
        (Path(batch)/"logs/sacct_export_snapshot.csv").write_text(raw)
        accounting=[{k:(None if k=="MaxRSS" and v=="" else v) for k,v in row.items()} for row in csv.DictReader(raw.splitlines(),delimiter="|")]
    result={"algorithm_commit":m["algorithm_commit"],"launcher_commit":m["launcher_commit"],"worlds":list(WORLDS),
            "python":sys.executable,"runtime":{k:importlib.metadata.version(k) for k in ("torch","numpy","scipy")},
            "submission":submission,"sacct_snapshot":accounting,"snapshot_note":"finish job can still be RUNNING; final sacct supplied independently by monitor",
            "failure_records":[read(x) for x in sorted((Path(batch)/"records").glob("failure_*.json"))],
            "artifact_bytes":sum(x.stat().st_size for d in ("t","e") for x in (Path(batch)/d).rglob("*") if x.is_file()),
            "max_rss_policy":"empty sacct MaxRSS remains null; see archived sacct CSV, never estimated",
            "cpu_cuda_streams_equal":False,"evaluated_training_seeds_per_learning_method":1,
            "uniform_streams_are_training_seeds":False,"no_additional_hpc_pilot":True}
    write(Path(batch)/"records/runtime_report.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("init","preflight","train-unit","verify-training","select","eval-unit","summary","export","status","runtime-report"))
    p.add_argument("--study-root",type=Path);p.add_argument("--batch-id");p.add_argument("--algorithm-commit")
    p.add_argument("--batch-root",type=Path);p.add_argument("--id",type=int);p.add_argument("--attempt",default="a01")
    p.add_argument("--reason");p.add_argument("--resume-episode",type=int);p.add_argument("--smoke",action="store_true")
    p.add_argument("--ids",nargs="+",type=int)
    a=p.parse_args()
    if a.command=="init": result=init(a.study_root,a.batch_id,a.algorithm_commit,a.smoke)
    elif a.command=="preflight": result=preflight(a.batch_root)
    elif a.command=="train-unit": result=train_unit(a.batch_root,a.id,a.attempt,a.reason,a.resume_episode)
    elif a.command=="eval-unit": result=eval_unit(a.batch_root,a.id,a.attempt,a.reason)
    elif a.command=="verify-training":
        load(a.batch_root); ids=a.ids if a.ids is not None else list(range(36))
        result={"technical_status":"PASS","ids":ids,"records":[verify_train(a.batch_root,i,Path(read(a.batch_root/f"records/train_{i}.json")["run_dir"])) for i in ids]}
    elif a.command=="status":
        load(a.batch_root); result={"accepted_training":[i for i in range(36) if (a.batch_root/f"records/train_{i}.json").exists()],
                                   "accepted_evaluation":[i for i in range(240) if (a.batch_root/f"records/eval_{i}.json").exists()]}
    else: result={"select":select,"summary":summary,"export":export,"runtime-report":runtime_report}[a.command](a.batch_root)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
