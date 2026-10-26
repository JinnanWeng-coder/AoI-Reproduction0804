"""Explicit tiny development smoke: fresh-process resume + eight-way CPU equality.

Six episodes,100 slots, small networks/replay; NEVER formal result/candidate.
Formal resource calibration must use the production seed8 and CPU pilots.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from analysis import cam96k_hpc as p
from analysis.smoke_cam96k_full_resume import compare
from analysis.evaluate_frozen_deployment import reduce_arrays
from aoi_v2x_reproduction.runtime import runner as r, full_training_checkpoint as fc
from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state


def worker(root,model,stage,device,unit=0):
    if model=="uniform_random":
        from analysis import cam96k_hpc_uniform as uniform
        src=next(s for s in p.sources() if s["model_id"]=="shared_tdec")
        cfg,_=p.derive(src,root/"unused_random",smoke=True,device="cpu");cfg.checkpoint_mode="none"
        arrays,result=uniform.sample(cfg,12+(unit%2),1001,1)
        dest=root/stage/str(unit);(dest/"native").mkdir(parents=True)
        np.savez_compressed(dest/"native/metrics.npz",**arrays)
        p.write(dest/"native/EVAL_COMPLETE.json",result)
        rows=reduce_arrays(arrays,result,{"model_id":model,"world":1001,"training_seed":None},expected=(1,1,100,5))
        p.write(dest/"smoke_verification.json",{"technical_status":"PASS","smoke":True,"rows":rows,"device":"cpu"})
        return 0
    source=next(s for s in p.sources() if s["model_id"]==model and s["training_seed"]==8)
    mode="continuous" if stage=="continuous" else "restarted"
    cfg,_=p.derive(source,root/model/mode,smoke=True,device=device)
    cfg.diagnostics=False
    run=Path(cfg.output_root)/cfg.run_name
    if stage in {"serial","parallel"}:
        run=root/model/"restarted"/cfg.run_name
        policy=(run/"policies/policy_episode_000006.pt") if "maddpg" in model else run/"policy_final.pt"
        src={"model_id":model,"training_seed":8,"best_episode":6,"best":p.identity(policy)}
        dest=root/stage/str(unit)
        begin=time.perf_counter()
        result=p.evaluate_native(src,dest,1001,1,periodic=True)
        arrays=p.load_arrays(dest/"native/metrics.npz")
        rows=reduce_arrays(arrays,result,{"model_id":model,"world":1001,"training_seed":8,"repeat":0},expected=(1,1,100,5))
        p.require(len(rows)==5 and result["critic_constructed"] is False and result["optimizer_constructed"] is False,
                  "CPU policy-only smoke contract mismatch")
        p.write(dest/"smoke_verification.json",{"smoke":True,"technical_status":"PASS","model_id":model,
                "device":"cpu","elapsed_seconds":time.perf_counter()-begin,"rows":rows,"max_rss":None})
        return 0
    name="_make_system" if "maddpg" in model else "_make_mappo_system"
    factory=getattr(r,name);systems=[]
    def capture(c):
        system=factory(c);systems.append(system);return system
    setattr(r,name,capture)
    kwargs=dict(periodic_policy_interval=2,periodic_policy_smoke=True,full_checkpoint_interval=2,full_checkpoint_smoke=True)
    if stage=="interrupt":
        def stop(d):
            if d["episode"]==3:raise RuntimeError("intentional development cut")
        try:r.train(cfg,progress_callback=stop,**kwargs)
        except RuntimeError as exc:
            if str(exc)=="intentional development cut":return 43
            raise
        raise ValueError("cut not triggered")
    checkpoint=run/"training_checkpoints/training_episode_000002.pt" if stage=="resume" else None
    r.train(cfg,resume_training_checkpoint=checkpoint,**kwargs)
    fc.verify_full_run(run,[2,4,6],final_path=run/"policy_final.pt")
    torch.save({"state":fc.training_state(cfg,systems[-1]),"rng":capture_rng_state()},run/"smoke_state.pt")
    return 0


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("--root",required=True,type=Path)
    a.add_argument("--model",choices=(*p.MODELS,"uniform_random"));a.add_argument("--stage",choices=("continuous","interrupt","resume","serial","parallel"))
    a.add_argument("--training-device",choices=("cpu","cuda:0"),default="cpu");a.add_argument("--unit",type=int,default=0)
    n=a.parse_args();root=n.root.resolve();p.require("smoke" in root.name,"dedicated smoke directory required")
    if n.stage:return worker(root,n.model,n.stage,n.training_device,n.unit)
    p.require(not root.exists(),"new smoke root required");root.mkdir(parents=True)
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
    def command(model,stage,unit=0):
        return [sys.executable,"-B","-u","-m","analysis.smoke_cam96k_hpc","--root",str(root),"--model",model,
                "--stage",stage,"--training-device",n.training_device,"--unit",str(unit)]
    records=[]
    for model in p.MODELS:
        for stage in ("continuous","interrupt","resume"):
            with (root/f"{model}_{stage}.log").open("x",encoding="utf-8") as f:
                out=subprocess.run(command(model,stage),cwd=p.dc.REPO,env=environment,stdout=f,stderr=subprocess.STDOUT)
            p.require(out.returncode==(43 if stage=="interrupt" else 0),f"smoke failed: {model}/{stage}; see retained log")
        runs=[root/model/mode/f"cam96k_{model}_s8" for mode in ("continuous","restarted")]
        compare(*[torch.load(run/"smoke_state.pt",map_location="cpu",weights_only=False) for run in runs])
        with np.load(runs[0]/"train_metrics.npz") as x,np.load(runs[1]/"train_metrics.npz") as y:compare(dict(x),dict(y))
        records.append({"model_id":model,"fresh_process_resume_state_rng_metrics_exact":True})
        print(json.dumps(records[-1]),flush=True)
    # Eight independent CPU processes: six learned structures and both Random
    # streams; all inputs have a separately executed serial reference.
    models=[*p.MODELS,"uniform_random","uniform_random"]
    for i,model in enumerate(models):
        with (root/f"serial_{i}.log").open("x",encoding="utf-8") as f:
            out=subprocess.run(command(model,"serial",i),cwd=p.dc.REPO,env=environment,stdout=f,stderr=subprocess.STDOUT)
        p.require(out.returncode==0,f"CPU serial smoke failed: {model}")
    active=[]
    try:
        for i,model in enumerate(models):
            f=(root/f"parallel_{i}.log").open("x",encoding="utf-8")
            active.append((subprocess.Popen(command(model,"parallel",i),cwd=p.dc.REPO,env=environment,stdout=f,stderr=subprocess.STDOUT),f))
        for process,f in active:p.require(process.wait()==0,"CPU parallel smoke failed; retain logs")
    finally:
        for process,f in active:
            if process.poll() is None:process.terminate();process.wait()
            f.close()
    for i in range(8):
        x,y=[root/stage/str(i)/"native" for stage in ("serial","parallel")]
        with np.load(x/"metrics.npz") as a,np.load(y/"metrics.npz") as b:compare(dict(a),dict(b))
        p.require(p.read(x/"EVAL_COMPLETE.json")["exogenous_probes"]==p.read(y/"EVAL_COMPLETE.json")["exogenous_probes"],"CPU parallel physical RNG mismatch")
    result={"technical_status":"PASS","smoke":True,"methods":6,"records":records,"cpu_parallel_processes":8,
            "serial_parallel_all_arrays_and_probes_exact":True,"training_device":n.training_device,"eval_device":"cpu",
            "formal_training_started":False,"formal_evaluation_started":False,"implementation_commit":p.dc.git("rev-parse","HEAD"),
            "dirty":bool(p.dc.git("status","--porcelain")),"production_runtime_estimate":"not inferred from tiny smoke"}
    p.write(root/"verification.json",result);print(json.dumps(result,indent=2),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
