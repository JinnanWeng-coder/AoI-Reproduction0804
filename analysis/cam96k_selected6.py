"""Twenty fresh seed8-11 runs, then one best100 per method across seed8-13.

An explicit combined schema delegates unchanged full-state training to its own
training batch. Its historical per-seed evaluation matrix is never executed.
Only this driver selects five policies and schedules 150 frozen world units.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from analysis import cam96k_seed8_11 as t
from analysis.deployment_rng import action_seed_words, policy_seed
from analysis.evaluate_frozen_deployment import csv_read, csv_write, load_arrays
from aoi_v2x_reproduction.config import config_from_dict
from aoi_v2x_reproduction.runtime.training_selection import BEST, RULE, load_training_policy, load_training_actors

SCHEMA = "cam96k_selected_seed8_13_v1"
RULE6 = {"score": "best100_rolling_training_reward", "maximize": True,
         "candidate_training_seeds": list(range(8,14)), "ties": "lowest_training_seed",
         "selection_uses_evaluation": False, "policies_per_method": 1}
MODELS, SEEDS, WORLDS = t.MODELS, tuple(range(8,14)), tuple(range(1001,1031))
read, write, require, identity = t.read, t.write, t.require, t.identity


def plan(smoke=False):
    return [{"unit_id": i, "model_id": model, "world": world, "repeat": 0,
             "family": "maddpg" if "maddpg" in model else "mappo"}
            for i,(world,model) in enumerate((w,m) for w in (WORLDS[:2] if smoke else WORLDS) for m in MODELS)]


def old_report(legacy):
    m = read(Path(legacy)/"protocol/manifest.json")
    proof = read(Path(legacy)/"records/training_verification.json")
    require(m["schema"] == "cam96k_training_pilot_v2" and m["smoke"] is False and
            m["implementation_commit"] == t.BASELINE and m["cam_bits"] == 96000 and m["training_episodes"] == 2500 and
            m["slots"] == 100 and m["rule"] == RULE and
            proof["technical_status"] == "PASS" and proof["smoke"] is False and proof["accepted_training_runs"] == 10 and
            len(proof["records"]) == 10 and
            {(r["model_id"],r["training_seed"]) for r in proof["records"]} == {(m,s) for m in MODELS for s in (12,13)},
            "legacy source is not the accepted ten CAM96k seed12/13 runs")
    return m,proof


def init(study,batch_id,commit,*,smoke=False,legacy=None):
    study=Path(study).resolve();legacy=Path(legacy or study/t.ROOT/"cam96k_20261025_v2").resolve()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}",batch_id) is not None and ("smoke" in batch_id.lower()) == smoke,"unsafe selected batch ID")
    t.code_check(commit,smoke)
    old_report(legacy)
    batch=study/t.ROOT/batch_id
    child=study/t.ROOT/(batch_id+"_train")
    require(not batch.exists() and (smoke or not child.exists()),"new unique batch required")
    if not smoke:
        t.init(study,child.name,commit)
    batch.mkdir(parents=True)
    for name in ("protocol","records","logs","locks","e","tmp","tables"):(batch/name).mkdir()
    m={"schema":SCHEMA,"batch_root":str(batch),"study_root":str(study),"implementation_commit":commit,"smoke":smoke,
       "training_batch":None if smoke else str(child),"training_manifest":None if smoke else identity(child/"protocol/manifest.json"),
       "legacy_batch":str(legacy),"legacy_manifest":identity(legacy/"protocol/manifest.json"),
       "legacy_training_verification":identity(legacy/"records/training_verification.json"),
       "selection_rule":RULE6,"candidate_training_seeds":[12,13] if smoke else list(SEEDS),
       "smoke_candidate_subset_only":smoke,"eval_plan":plan(smoke),"warmup":5,"scored":2 if smoke else 100,
       "slots":100,"cam_bits":96000,"max_training_processes":2,"default_eval_workers":8,
       "world_role":t.WORLD_ROLE,"new_unseen_confirmation":False,"created_at_utc":t.now()}
    write(batch/"protocol/manifest.json",m)
    return m


def load(batch):
    batch=Path(batch).resolve();m=read(batch/"protocol/manifest.json")
    t.code_check(m["implementation_commit"],m["smoke"])
    require(m["schema"] == SCHEMA and m["batch_root"] == str(batch) and m["eval_plan"] == plan(m["smoke"]) and
            m["selection_rule"] == RULE6 and m["candidate_training_seeds"] == ([12,13] if m["smoke"] else list(SEEDS)) and
            m["smoke_candidate_subset_only"] == m["smoke"] and m["warmup"] == 5 and m["scored"] == (2 if m["smoke"] else 100) and
            m["slots"] == 100 and m["cam_bits"] == 96000 and m["max_training_processes"] == 2 and m["default_eval_workers"] == 8 and
            m["world_role"] == t.WORLD_ROLE and m["new_unseen_confirmation"] is False,"selected-six batch contract changed")
    legacy=Path(m["legacy_batch"])
    require(identity(legacy/"protocol/manifest.json") == m["legacy_manifest"] and
            identity(legacy/"records/training_verification.json") == m["legacy_training_verification"],"historical source identities changed")
    if not m["smoke"]:
        require(identity(Path(m["training_batch"])/"protocol/manifest.json") == m["training_manifest"],"training delegation changed")
        child=t.load(m["training_batch"])
        require(child["implementation_commit"] == m["implementation_commit"] and child["smoke"] is False,"wrong training child batch")
    return m


def preview(batch):
    m=load(batch)
    training=None if m["smoke"] else t.preview(m["training_batch"])
    return {"schema":SCHEMA,"smoke":m["smoke"],"new_training_runs":0 if m["smoke"] else 20,
            "existing_training_runs":10,"candidate_runs":10 if m["smoke"] else 30,"candidate_training_seeds":m["candidate_training_seeds"],
            "selected_policies":5,"evaluated_training_seeds_per_method":1,"eval_units":len(m["eval_plan"]),
            "worlds":list(WORLDS[:2] if m["smoke"] else WORLDS),"selection_rule":RULE6,
            "max_training_processes":2,"default_eval_workers":8,"training_batch":m["training_batch"],
            "training_configs":None if training is None else training["configs"],
            "training_episodes_total":0 if training is None else training["training_episodes_total"],
            "training_environment_steps_total":0 if training is None else training["training_environment_steps_total"],
            "new_periodic_policy_records":0 if training is None else training["periodic_snapshot_records"],
            "new_full_checkpoint_records":0 if training is None else training["full_checkpoint_records"],
            "child_per_seed_eval_matrix_is_not_executed":True}


def audit_candidate(record,batch,commit):
    run=Path(record["run_dir"]).resolve();batch=Path(batch).resolve()
    model,seed=record["model_id"],record["training_seed"]
    require(model in MODELS and seed in SEEDS and run.parent.parent == batch/"t" and
            run.name == f"cam96k_{model}_s{seed}","candidate run identity mismatch")
    files={"config":"config.resolved.json","complete":"COMPLETE.json","provenance":"provenance.json",
           "metrics":"train_metrics.npz","best":BEST,"final":"policy_final.pt","selection_ledger":"training_reward_selection.csv"}
    artifacts={}
    for key,name in files.items():
        actual=identity(run/name);require(actual == record[key],f"accepted candidate bytes changed: {key}");artifacts[key]=actual
    cfg,payload=load_training_policy(run/BEST)
    load_training_actors(run/BEST)
    complete=read(run/"COMPLETE.json");provenance=read(run/"provenance.json")
    source=next(s for s in t.sources() if s["model_id"] == model)
    differences=t.config_differences(source["config"],cfg.to_dict())
    require(set(differences) <= {"seed","cam_bits","run_name","output_root"} and cfg.seed == seed and cfg.cam_bits == 96000 and
            cfg.episodes == 2500 and cfg.steps_per_episode == 100 and cfg.smoke is False and
            payload["checkpoint_role"] == "best_training100" and
            complete["reproduction_git_commit"] == provenance["reproduction_git_commit"] == payload["reproduction_git_commit"] == commit and
            complete["reproduction_git_dirty"] is provenance["reproduction_git_dirty"] is payload["reproduction_git_dirty"] is False,
            "candidate effective treatment/provenance mismatch")
    rewards=t.tc.rewards_from_npz(run/"train_metrics.npz",2500)
    episode,score,scores=t.tc.best_index(rewards)
    selected=payload["training_reward_selection"]
    require(selected["rule"] == RULE and selected["selected_episode"] == episode == record["best_episode"] and
            selected["score"] == score == record["best_score"] and payload["policy_version"] == record["best_policy_version"],
            "candidate best100 score/episode differs from training history")
    ledger=csv_read(run/"training_reward_selection.csv")
    require(len(ledger) == 2500,"candidate selection ledger coverage mismatch")
    maximum=None
    for e,row in enumerate(ledger,1):
        rolling=scores[e-100] if e>=100 else None
        improved=rolling is not None and (maximum is None or rolling>maximum)
        require(int(row["episode"]) == e and float(row["episode_reward"]) == rewards[e-1] and
                (row["rolling100"] == "" if rolling is None else float(row["rolling100"]) == rolling) and row["new_best"] == str(improved),
                "candidate rolling100 ledger changed")
        if improved:maximum=rolling
    return {"model_id":model,"training_seed":seed,"checkpoint_role":"best_training100","score":score,
            "best_episode":episode,"best_policy_version":payload["policy_version"],"best":artifacts["best"],"run_dir":str(run),
            "training_commit":commit,"config_hash":cfg.canonical_hash(),"actor_network_count":len(payload["actors"]),"artifacts":artifacts}


def candidates(batch):
    m=load(batch);old,proof=old_report(m["legacy_batch"])
    result=[audit_candidate(r,m["legacy_batch"],old["implementation_commit"]) for r in proof["records"]]
    if not m["smoke"]:
        training=t.verify_training(m["training_batch"])
        result += [audit_candidate(r,m["training_batch"],m["implementation_commit"]) for r in training["records"]]
    return sorted(result,key=lambda r:(MODELS.index(r["model_id"]),r["training_seed"]))


def choose(records,seeds=SEEDS):
    require(len(records) == len(MODELS)*len(seeds) and
            {(r["model_id"],r["training_seed"]) for r in records} == {(m,s) for m in MODELS for s in seeds},"complete unique candidate matrix required")
    require(all(r["checkpoint_role"] == "best_training100" and math.isfinite(r["score"]) for r in records),"invalid best100 training reward")
    return [min((r for r in records if r["model_id"] == model),key=lambda r:(-r["score"],r["training_seed"])) for model in MODELS]


def select(batch):
    m=load(batch)
    with t.lock(Path(batch)/"locks/selection.lock",batch=str(batch)):
        rows=candidates(batch)
        result={"schema":"cam96k_selected6_policy_selection_v1","smoke":m["smoke"],"rule":RULE6,
                "candidate_training_seeds":m["candidate_training_seeds"],"candidates":rows,"selected":choose(rows,m["candidate_training_seeds"])}
        path=Path(batch)/"records/selection.json"
        if path.exists():require(read(path) == result,"sealed training-only selection changed")
        else:write(path,result)
    return result


def selection(batch):
    m=load(batch);proof=read(Path(batch)/"records/selection.json")
    require(proof["schema"] == "cam96k_selected6_policy_selection_v1" and proof["rule"] == RULE6 and proof["smoke"] == m["smoke"] and
            proof["candidate_training_seeds"] == m["candidate_training_seeds"] and
            choose(proof["candidates"],m["candidate_training_seeds"]) == proof["selected"],"selection proof changed")
    for row in proof["candidates"]:
        for item in row["artifacts"].values():require(identity(item["path"]) == item,"candidate input changed after selection")
    return proof


def units(batch):
    m=load(batch);chosen={r["model_id"]:r for r in selection(batch)["selected"]}
    return [{**u,"training_seed":chosen[u["model_id"]]["training_seed"],
             "checkpoint_sha256":chosen[u["model_id"]]["best"]["sha256"],
             "noise_std":0. if u["family"] == "maddpg" else None,
             "action_seed_words":action_seed_words(chosen[u["model_id"]]["training_seed"],u["world"],0,u["family"])} for u in m["eval_plan"]]


def source_for(batch,unit):
    return next(r for r in selection(batch)["selected"] if r["model_id"] == unit["model_id"])


def native(batch,unit,dest):
    m=load(batch);s=source_for(batch,unit)
    # Even development smoke uses genuine best100, never disguises final or periodic policies.
    return t.evaluate_native({"smoke":False,"study_root":m["study_root"],"scored":m["scored"]},unit,
        {**s,"episodes":2500},dest)


def verify_eval(batch,i,attempt):
    m=load(batch);unit=units(batch)[i];s=source_for(batch,unit);run=Path(s["run_dir"])
    cfg=config_from_dict(read(run/"config.resolved.json"));dest=Path(batch)/"e"/str(i)/attempt
    started=read(dest/"STARTED.json");require(started["unit"] == unit and started["source"] == s,"evaluation unit/source changed")
    meta=read(dest/"native/EVAL_COMPLETE.json")
    expected={"status":"complete","training_seed":unit["training_seed"],"eval_seeds":[unit["world"]],"environment_seeds":[unit["world"]],
        "action_repeat":0,"action_seed_words":[unit["action_seed_words"]],"eval_protocol":"sequential_warm","eval_warmup_episodes":5,
        "eval_episodes":m["scored"],"target_scenario":t.dc.SCENARIO,"source_scenario":t.dc.SCENARIO,"source_config_hash":cfg.canonical_hash(),
        "training_source_commit":s["training_commit"],"reproduction_git_commit":m["implementation_commit"],"world_role":t.WORLD_ROLE,
        "cam_bits":96000,"global_actor_weight":1,"actor_network_count":s["actor_network_count"],
        "checkpoint_role":"best_training100","policy_version":s["best_policy_version"],"training_budget_episodes":2500}
    if not m["smoke"]:expected["reproduction_git_dirty"]=False
    if unit["family"] == "mappo":
        expected.update(policy_sampling_seeds=[policy_seed(unit["training_seed"],unit["world"],0)],policy_parameters_unchanged=True,
            critic_constructed=False,optimizer_constructed=False,rollout_constructed=False,mappo_eval_mode="stochastic",actor_sharing=bool(cfg.mappo_actor_sharing),
            mappo_variant=cfg.mappo_variant,policy_episode=s["best_episode"],target_config=cfg.to_dict(),target_config_hash=cfg.canonical_hash())
        actual=(dest/"native"/meta["policy"]).resolve()
    else:
        expected.update(noise_std=0.,explore=False,parameter_snapshot_unchanged=True,optimizer_steps=0,optimizer_constructed=False,
            critic_constructed=False,replay_constructed=False,normalization_updated=False,actor_sharing=bool(cfg.maddpg_actor_sharing),
            runtime_config=cfg.to_dict(),checkpoint_episode=s["best_episode"])
        actual=Path(meta["checkpoint"]).resolve()
    for k,v in expected.items():require(meta.get(k) == v,f"selected native evaluation {k} mismatch")
    require(actual == Path(s["best"]["path"]) and identity(actual) == s["best"],"selected checkpoint changed")
    require(len(meta.get("exogenous_probes",[])) == 1 and meta["exogenous_probes"][0]["world"] == unit["world"],"missing private physical RNG probe")
    rows=t.reduce_payload(load_arrays(dest/"native/metrics.npz"),meta,unit,m["scored"])
    return {"technical_status":"PASS","unit":unit,"attempt":attempt,"smoke":m["smoke"],"source":s["best"],
            "completion":identity(dest/"native/EVAL_COMPLETE.json"),"metrics":identity(dest/"native/metrics.npz"),"rows":rows}


@contextmanager
def eval_slot(batch):
    m=load(batch)
    for i in range(8):
        path=Path(m["study_root"])/f".cam96k-selected6-eval-slot{i}.lock"
        try:
            handle=path.open("x",encoding="utf-8")
        except FileExistsError:continue
        with handle:json.dump({"pid":os.getpid(),"batch":str(batch),"created_at_utc":t.now()},handle)
        try:yield
        finally:path.unlink()
        return
    raise ValueError("eight selected-six evaluation slots occupied; inspect holder PIDs")


def eval_unit(batch,i,attempt="a01",reason=None):
    m=load(batch);require(type(i) is int and 0<=i<len(m["eval_plan"]),"invalid unit ID")
    require(re.fullmatch(r"a[0-9]{2,4}",attempt) is not None and (attempt == "a01" or reason),"unsafe attempt/missing technical reason")
    unit=units(batch)[i];s=source_for(batch,unit)
    with t.lock(Path(batch)/f"locks/eval_{i}.lock",unit=i):
        record=Path(batch)/f"records/eval_{i}.json"
        if record.exists():
            saved=read(record);require(verify_eval(batch,i,saved["attempt"]) == saved,"accepted result changed")
            return {"unit_id":i,"skipped_verified":True}
        parent=Path(batch)/"e"/str(i);completed=list(parent.glob("a*/native/EVAL_COMPLETE.json"))
        require(len(completed)<=1,"multiple successful native attempts require investigation")
        if completed:attempt=completed[0].parent.parent.name
        else:
            dest=parent/attempt;require(not dest.exists() and (not parent.exists() or reason),"preserve partial attempt; new attempt and reason required")
            dest.mkdir(parents=True);write(dest/"STARTED.json",{"unit":unit,"source":s,"pid":os.getpid(),"reason":reason})
            with eval_slot(batch):
                try:native(batch,unit,dest)
                except BaseException as exc:
                    write(dest/"FAILED.json",{"error":repr(exc),"reason":reason,"failed_at_utc":t.now()});raise
        result=verify_eval(batch,i,attempt);write(record,result);return {"unit_id":i,"technical_status":"PASS"}


def launch(batch,ids,attempt,reason):
    active=[];samples=[]
    try:
        for i in ids:
            command=[sys.executable,"-B","-u","-m","analysis.cam96k_selected6","eval-unit","--batch-root",str(batch),"--id",str(i),"--attempt",attempt]
            if reason:command += ["--reason",reason]
            log=Path(batch)/f"logs/eval_{i}_{attempt}_{time.time_ns()}.out";handle=log.open("x",encoding="utf-8")
            try:process=subprocess.Popen(command,cwd=t.REPO,env=t.child_env(batch),stdout=handle,stderr=subprocess.STDOUT)
            except BaseException:handle.close();raise
            active.append((process,handle,i,command,str(log)))
            print(json.dumps({"unit_id":i,"pid":process.pid,"command":command,"log":str(log)}),flush=True)
        last=0;last_heartbeat=0
        while any(p.poll() is None for p,*_ in active):
            if time.perf_counter()-last>=2:
                from analysis.local_process_resources import sample
                samples.append(sample([p.pid for p,*_ in active if p.poll() is None]));last=time.perf_counter()
            if time.perf_counter()-last_heartbeat>=15:
                from aoi_v2x_reproduction.runtime.runner import _write_json
                heartbeat={"updated_at_utc":t.now(),"supervisor_pid":os.getpid(),"processes":[
                    {"pid":p.pid,"unit_id":i,"exit_code":p.poll()} for p,_,i,*_ in active]}
                _write_json(Path(batch)/"records/eval_supervisor_heartbeat.json",heartbeat)
                print(json.dumps(heartbeat),flush=True);last_heartbeat=time.perf_counter()
            time.sleep(.5)
        write(Path(batch)/f"records/resources_eval_{time.time_ns()}.json",samples)
        outcomes=[{"unit_id":i,"pid":p.pid,"exit_code":p.returncode,"command":c,"log":log} for p,_,i,c,log in active]
        write(Path(batch)/f"records/process_group_{time.time_ns()}.json",outcomes)
        require(all(o["exit_code"] == 0 for o in outcomes),f"evaluation group failed; preserve attempts: {outcomes}")
    finally:
        for p,h,*_ in active:
            if p.poll() is None:p.terminate();p.wait()
            h.close()


def evaluate(batch,workers=8,attempt="a01",reason=None):
    m=load(batch);require(1<=workers<=8,"evaluation concurrency must be 1..8; training stays 2")
    preflight(batch)
    select(batch) # Recompute every training reward before frozen scheduling, never score-based eval selection.
    with t.lock(Path(m["study_root"])/".cam96k-selected6-eval-supervisor.lock",batch=str(batch)):
        pending=[]
        for u in units(batch):
            file=Path(batch)/f"records/eval_{u['unit_id']}.json"
            if file.exists():
                r=read(file);require(verify_eval(batch,u["unit_id"],r["attempt"]) == r,"accepted evaluation changed")
            else:pending.append(u["unit_id"])
        for offset in range(0,len(pending),workers):launch(batch,pending[offset:offset+workers],attempt,reason)
    return summary(batch)


def summary(batch):
    m=load(batch);proof=selection(batch);rows=[]
    for u in units(batch):
        saved=read(Path(batch)/f"records/eval_{u['unit_id']}.json")
        require(verify_eval(batch,u["unit_id"],saved["attempt"]) == saved,"accepted result changed")
        rows+=saved["rows"]
    policies=[];worlds=[]
    for s in proof["selected"]:
        group=[r for r in rows if r["model_id"] == s["model_id"]]
        platoons=[t.aggregate([r for r in group if r["platoon"] == i]) for i in range(5)]
        policies.append({"model_id":s["model_id"],"selected_training_seed":s["training_seed"],"best100_training_reward":s["score"],
                         "n_evaluated_training_seeds":1,**t.aggregate(group),
                         "worst_platoon_aoi_ms":max(r["aoi_ms"] for r in platoons),
                         "worst_platoon_cam_rate":min(r["cam_rate"] for r in platoons)})
        for w in (WORLDS[:2] if m["smoke"] else WORLDS):
            g=[r for r in group if r["world"] == w]
            worlds.append({"model_id":s["model_id"],"training_seed":s["training_seed"],"world":w,**t.aggregate(g),
                "worst_platoon_cam_rate":min(r["cam_deadline_success_count"]/r["cam_episode_count"] for r in g),
                "worst_platoon_aoi_ms":max(r["aoi_sum_ms"]/r["slot_count"] for r in g)})
    report={"technical_status":"PASS","smoke":m["smoke"],"implementation_commit":m["implementation_commit"],
        "accepted_candidate_training_runs":len(proof["candidates"]),"n_candidate_training_seeds_per_method":len(m["candidate_training_seeds"]),
        "selected_policy_count":5,"n_evaluated_training_seeds_per_method":1,"accepted_eval_units":len(m["eval_plan"]),"platoon_rows":len(rows),
        "cam_deadlines":sum(r["cam_episode_count"] for r in rows),"scored_agent_slots":sum(r["slot_count"] for r in rows),
        "selection_uses_evaluation":False,"world_role":t.WORLD_ROLE,"new_unseen_confirmation":False}
    for name,table in (("world_platoon.csv",rows),("world_summary.csv",worlds),("selected_policy_summary.csv",policies)):
        file=Path(batch)/"tables"/name
        if not file.exists():csv_write(file,table)
        else:
            expected=[{k:"" if row.get(k) is None else str(row[k]) for k in sorted({k for r in table for k in r})} for row in table]
            require(csv_read(file) == expected,"existing selected-policy summary changed")
    file=Path(batch)/"tables/verification.json"
    if file.exists():require(read(file) == report,"summary counts changed")
    else:write(file,report)
    return report


def preflight(batch):
    m=load(batch);legacy,proof=old_report(m["legacy_batch"])
    old=[audit_candidate(r,m["legacy_batch"],legacy["implementation_commit"]) for r in proof["records"]]
    from analysis.local_process_resources import sample
    resources=sample();require(resources["gpu_total_mib"]-resources["gpu_used_mib"]>6144 and
        resources["available_physical_bytes"]>4*1024**3,"resource shortage; report, no automatic concurrency fallback")
    child=None if m["smoke"] else t.preflight(m["training_batch"])
    result={"technical_status":"PASS","smoke":m["smoke"],"legacy_best100_candidates_checked":len(old),"resources":resources,
        "training_preflight":child,"new_training_started":False,"new_evaluation_started":False}
    write(Path(batch)/f"protocol/preflight_{time.time_ns()}.json",result);return result


def status(batch):
    m=load(batch);child=None if m["smoke"] else t.status(m["training_batch"])
    return {"smoke":m["smoke"],"implementation_commit":m["implementation_commit"],"training_batch":m["training_batch"],
        "new_training":[] if child is None else child["training"],"selection_sealed":(Path(batch)/"records/selection.json").exists(),
        "evaluation":[{**u,"state":"ACCEPTED" if (Path(batch)/f"records/eval_{u['unit_id']}.json").exists() else
            "INCOMPLETE" if (Path(batch)/"e"/str(u["unit_id"])).exists() else "PENDING"} for u in m["eval_plan"]],
        "locks":[{"path":str(p),"metadata":read(p)} for p in (Path(batch)/"locks").glob("*.lock")],
        "study_locks":[{"path":str(p),"metadata":read(p)} for p in Path(m["study_root"]).glob(".cam96k-*.lock")]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("init","preview","preflight","status","train","train-unit","verify-training","select","evaluate","eval-unit","summary","compare-smoke"))
    parser.add_argument("--batch-root",type=Path);parser.add_argument("--study-root",type=Path);parser.add_argument("--batch-id")
    parser.add_argument("--expected-commit");parser.add_argument("--legacy-root",type=Path);parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--eval-workers",type=int,default=8,help="evaluation concurrency default8; independent training cap2")
    parser.add_argument("--id",type=int);parser.add_argument("--attempt",default="a01");parser.add_argument("--reason")
    parser.add_argument("--resume-incomplete",action="store_true");parser.add_argument("--resume-episode",type=int)
    args=parser.parse_args(argv)
    require(not args.resume_incomplete or args.command in {"train","train-unit"},"resume is training only")
    require(args.resume_episode is None or args.command == "train-unit","resume episode is train-unit only")
    if args.command == "init":
        require(args.study_root and args.batch_id and args.expected_commit,"init requires paths and pinned commit")
        result=init(args.study_root,args.batch_id,args.expected_commit,smoke=args.smoke,legacy=args.legacy_root)
    else:
        require(args.batch_root and not args.smoke,"batch root required; --smoke is init only")
        m=load(args.batch_root)
        if args.command in {"train","train-unit","verify-training"}:
            require(not m["smoke"],"selected eval smoke never starts training")
            if args.command == "train":
                result=t.train_all(m["training_batch"],args.attempt,args.reason,args.resume_incomplete);select(args.batch_root)
            elif args.command == "train-unit":
                result=t.train_unit(m["training_batch"],args.id,args.attempt,args.reason,args.resume_incomplete,args.resume_episode)
            else:result=t.verify_training(m["training_batch"])
        elif args.command == "evaluate":result=evaluate(args.batch_root,args.eval_workers,args.attempt,args.reason)
        elif args.command == "eval-unit":result=eval_unit(args.batch_root,args.id,args.attempt,args.reason)
        elif args.command == "compare-smoke":result=compare_smoke(args.batch_root)
        else:result={"preview":preview,"preflight":preflight,"status":status,"select":select,"summary":summary}[args.command](args.batch_root)
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),flush=True)
    return 0


def compare_smoke(batch):
    m=load(batch);require(m["smoke"],"comparison is development smoke only")
    proof=[]
    for u in units(batch)[:8]:
        saved=read(Path(batch)/f"records/eval_{u['unit_id']}.json")
        require(verify_eval(batch,u["unit_id"],saved["attempt"]) == saved,"concurrent result changed")
        dest=Path(batch)/"comparison"/str(u["unit_id"])
        require(not dest.exists(),"new comparison directory required")
        native(batch,u,dest)
        a=load_arrays(saved["metrics"]["path"]);b=load_arrays(dest/"native/metrics.npz")
        require(set(a) == set(b) and all(np.array_equal(a[k],b[k]) for k in a),"serial/concurrent arrays differ")
        original=read(saved["completion"]["path"]);other=read(dest/"native/EVAL_COMPLETE.json")
        require(original["exogenous_probes"] == other["exogenous_probes"],"private physical RNG changed")
        proof.append({"unit":u,"all_arrays_exact":True,"concurrent_metrics":saved["metrics"],"serial_metrics":identity(dest/"native/metrics.npz")})
    result={"technical_status":"PASS","smoke":True,"comparison_units":8,"records":proof}
    write(Path(batch)/"records/serial_eight_worker_comparison.json",result);return result


if __name__ == "__main__":raise SystemExit(main())
