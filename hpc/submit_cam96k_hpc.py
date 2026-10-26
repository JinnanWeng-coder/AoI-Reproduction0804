"""Submit36->gate->240CPU->summary/export once; no independent HPC pilot.

Both arrays start at throttle2. afterok serializes stages, preventing a
per-array multiplication of the combined maximum8. Slurm grants allocations.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time
from analysis import cam96k_hpc as p


def call(args):return subprocess.check_output(args,text=True).strip()


def tres(text):
    d={}
    for s in text.split(","):
        if "=" in s:
            k,v=s.split("=",1)
            try:d[k]=float(v.rstrip("MG"))*(1024 if v.endswith("G") else 1)
            except ValueError:pass
    return d


def capacity(nodes,partition,cpus,mem_mb,gpu):
    slots=0
    for line in nodes.splitlines():
        f=dict(re.findall(r"(\w+)=([^ ]*)",line))
        if partition not in f.get("Partitions","").split(",") or any(s in f.get("State","") for s in ("DOWN","DRAIN","FAIL","MAINT","RESERVED")):continue
        c,a=tres(f.get("CfgTRES","")),tres(f.get("AllocTRES",""))
        n=min(int(max(0,c.get("cpu",0)-a.get("cpu",0))//cpus),int(max(0,c.get("mem",0)-a.get("mem",0))//mem_mb))
        if gpu:n=min(n,int(max(0,c.get("gres/gpu:l20",0)-max(a.get("gres/gpu:l20",0),a.get("gres/gpu",0)))))
        slots+=n
    return min(8,slots)


def policy_bound(policy,partition,account,gpu,cpus,mem_mb):
    """Conservative static association bounds; Slurm also enforces live/QOS use."""
    bounds=[8];matched=False
    for line in policy.splitlines():
        row=line.split("|")
        if len(row)<7:continue
        if account and row[0]!=account:continue
        if row[1] and row[1]!=partition:continue
        matched=True
        if row[5].isdigit():bounds.append(int(row[5]))
        total=tres(row[3])
        for k,request in (("cpu",cpus),("mem",mem_mb)):
            if k in total:bounds.append(int(total[k]//request))
        if gpu:
            for k in ("gres/gpu","gres/gpu:l20"):
                if k in total:bounds.append(int(total[k]))
        per_job=tres(row[4])
        if any(k in per_job and per_job[k]<v for k,v in (("cpu",cpus),("mem",mem_mb))):bounds.append(0)
    p.require(matched,"no matching account/partition policy")
    return max(0,min(bounds))


def qos_bound(rows,name,gpu,cpus,mem_mb):
    matched=[r.split("|") for r in rows.splitlines() if r.split("|")[0]==name]
    p.require(len(matched)==1 and len(matched[0])>=5,"actual job QOS limits unavailable")
    row=matched[0];bounds=[8]
    if row[3].isdigit():bounds.append(int(row[3]))
    for text in row[1:3]:
        limits=tres(text)
        for key,n in (("cpu",cpus),("mem",mem_mb)):
            if key in limits:bounds.append(int(limits[key]//n))
        if gpu:
            for key in ("gres/gpu","gres/gpu:l20"):
                if key in limits:bounds.append(int(limits[key]))
    return max(0,min(bounds))


def commands(batch,partition="Q10",cpu_partition=None,account=None,train_hours=12,eval_hours=4):
    root=p.dc.REPO/"hpc"
    common=["sbatch","--parsable","--export=ALL",f"--partition={partition}",
            f"--output={batch}/logs/%x_%A_%a.out",f"--error={batch}/logs/%x_%A_%a.err"]
    if account:common.append(f"--account={account}")
    train=[*common,"--array=0-35%2","--cpus-per-task=4","--mem=32G",f"--time={train_hours}:00:00",str(root/"aoi_cam96k_hpc_train.sbatch")]
    cpu=[x if not x.startswith("--partition=") else f"--partition={cpu_partition or partition}" for x in common]
    gate=[*cpu,"--dependency=afterok:{train}","--cpus-per-task=4","--mem=32G","--time=04:00:00",str(root/"aoi_cam96k_hpc_analyze.sbatch")]
    evaluate=[*cpu,"--dependency=afterok:{gate}","--array=0-239%2","--cpus-per-task=4","--mem=16G",f"--time={eval_hours}:00:00",str(root/"aoi_cam96k_hpc_eval_cpu.sbatch")]
    finish=[*cpu,"--dependency=afterok:{evaluate}","--cpus-per-task=4","--mem=32G","--time=04:00:00",str(root/"aoi_cam96k_hpc_analyze.sbatch")]
    return dict(train=train,gate=gate,evaluate=evaluate,finish=finish)


def submit(batch,partition="Q10",cpu_partition=None,account=None,actually=False,train_hours=12,eval_hours=4):
    m=p.load(batch);p.preflight(batch)
    marker=Path(batch)/"records/submission_chain.json";intent=Path(batch)/"records/submission_chain_intent.json"
    p.require(not marker.exists() and not intent.exists(),"chain already submitted or uncertain outcome; inspect receipts/Slurm, never repeat full chain")
    cmd=commands(batch,partition,cpu_partition,account,train_hours,eval_hours)
    record={"commands":cmd,"initial_concurrency":2,"max_combined_concurrency":8,"no_hpc_pilot":True,
            "algorithm_commit":m["algorithm_commit"],"launcher_commit":m["launcher_commit"],
            "partition":partition,"cpu_partition":cpu_partition or partition,"account":account}
    if not actually:return record
    with p.old.lock(Path(batch)/"locks/submission.lock",batch=str(batch)):
        p.require(not marker.exists() and not intent.exists(),"concurrent/uncertain submission")
        p.write(intent,record)
        os.environ.update(PROJECT_DIR=str(p.dc.REPO),AOI_BATCH_ROOT=str(Path(batch).resolve()),AOI_LAUNCHER_COMMIT=m["launcher_commit"])
        os.environ.update(PYTHON_BIN=os.sys.executable,AOI_SCHEDULER_DIR=str(p.dc.REPO),AOI_SCHEDULER_COMMIT=m["launcher_commit"])
        os.environ["AOI_ATTEMPT"]="a01"
        for k in ("AOI_TASK_OFFSET","AOI_RETRY_REASON","AOI_RESUME_EPISODE"):os.environ.pop(k,None)
        jobs={}
        for name,args in cmd.items():
            resolved=[arg.format(**jobs) for arg in args]
            os.environ["AOI_ANALYZE_COMMAND"]="select" if name=="gate" else "export"
            job=call(resolved).split(";")[0];p.require(job.isdigit(),"uncertain submission outcome; inspect retained intent")
            jobs[name]=job
            p.write(Path(batch)/f"records/submitted_{name}.json",{"job_id":job,"command":resolved})
        record.update(jobs=jobs);p.write(marker,record)
    return record


def monitor(batch,interval=60):
    p.load(batch);r=p.read(Path(batch)/"records/submission_chain.json");jobs=r["jobs"]
    p.require(interval>=10,"monitor interval must be at least10 seconds")
    while True:
        q=call(["squeue","-h","-r","-j",",".join(jobs.values()),"-o","%F|%T|%R"])
        if not q:break
        acct=call(["sacct","-n","-P","-j",",".join(jobs.values()),"--format=JobID,State,ExitCode"])
        if "DependencyNeverSatisfied" in q or any(s in acct for s in ("FAIL","OUT_OF_ME","TIMEOUT","CANCELLED","PREEMPTED","DEADLINE")):
            p.write(Path(batch)/f"records/failure_monitor_{time.time_ns()}.json",{"squeue":q,"sacct":acct,"jobs":jobs,
                    "reason":"technical failure or impossible dependency; no automatic resubmission"})
            raise ValueError("chain technical failure; inspect retained failure record and exact IDs; do not repeat full chain")
        p.require(not any(s in acct for s in ("FAILED","OUT_OF_MEMORY","TIMEOUT","CANCELLED","NODE_FAIL")),
                  "chain technical failure; held dependencies must not bypass validation; inspect individual IDs")
        lines=[line.split("|") for line in q.splitlines()]
        event={"time":time.time(),"squeue":q,"stages":{}}
        for stage in ("train","evaluate"):
            job=jobs[stage];group=[v for v in lines if v[0]==job]
            if not group or all("Dependency" in v[2] for v in group):continue
            running=sum(v[1]=="RUNNING" for v in group);gpu=stage=="train";partition=r["partition"] if gpu else r["cpu_partition"]
            try:
                nodes=call(["scontrol","show","nodes","-o"])
                policy=call(["sacctmgr","-n","-P","show","assoc",f"user={os.environ.get('USER','')}","format=Account,Partition,QOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmitJobs"])
                qos=call(["sacctmgr","-n","-P","show","qos","format=Name,GrpTRES,MaxTRESPU,MaxJobsPU,MaxSubmitJobsPU"])
                job_info=call(["scontrol","show","job",job,"-o"])
                qmatch=re.search(r"\bQOS=([^ ]+)",job_info)
                p.require(qmatch is not None,"actual job QOS not visible")
                part=call(["scontrol","show","partition",partition,"-o"])
                p.require(bool(policy) and bool(qos) and "State=UP" in part,"policy/QOS/partition not visible")
                free=capacity(nodes,partition,4,32768 if gpu else 16384,gpu)
                bound=policy_bound(policy,partition,r["account"],gpu,4,32768 if gpu else 16384)
                bound=min(bound,qos_bound(qos,qmatch.group(1),gpu,4,32768 if gpu else 16384))
                throttle=max(1,min(bound,8,max(2,running+free)))
                call(["scontrol","update",f"JobId={job}",f"ArrayTaskThrottle={throttle}"])
                event["stages"][stage]={"running":running,"free_slots":free,"throttle":throttle,"association_policy":policy,"qos_policy":qos,
                                       "static_association_bound":bound,"partition":part}
            except (subprocess.CalledProcessError,ValueError) as exc:
                call(["scontrol","update",f"JobId={job}","ArrayTaskThrottle=2"])
                event["stages"][stage]={"throttle":2,"error":str(exc)}
        p.write(Path(batch)/f"records/monitor_{time.time_ns()}.json",event)
        time.sleep(interval)
    states={}
    for stage,job in jobs.items():
        raw=call(["sacct","-j",job,"--parsable2","--format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocTRES"])
        file=Path(batch)/f"logs/sacct_{stage}_{job}.csv"
        if not file.exists():file.write_text(raw+"\n")
        states.update({row["JobID"]:row for row in csv.DictReader(raw.splitlines(),delimiter="|")})
    expected=[*(f"{jobs['train']}_{i}" for i in range(36)),jobs["gate"],
              *(f"{jobs['evaluate']}_{i}" for i in range(240)),jobs["finish"]]
    p.require(all(i in states and states[i]["State"]=="COMPLETED" and states[i]["ExitCode"]=="0:0" for i in expected),
              "sacct not fully COMPLETED0:0 (or accounting still pending); report exact gaps, do not claim PASS")
    missing=[i for i in range(240) if not (Path(batch)/f"records/eval_{i}.json").exists()]
    p.require(not missing,f"incomplete evaluation IDs={missing}; no automatic resubmit")
    p.require((Path(batch)/"core_evidence.tar.gz").is_file(),"final export missing")
    return {"jobs":jobs,"technical_status":"PASS","verification":p.read(Path(batch)/"tables/verification.json")}


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument("command",choices=("submit","monitor"));a.add_argument("--batch-root",required=True,type=Path)
    a.add_argument("--partition",default="Q10");a.add_argument("--cpu-partition");a.add_argument("--account");a.add_argument("--submit",action="store_true")
    a.add_argument("--train-hours",type=int,default=12);a.add_argument("--eval-hours",type=int,default=4);a.add_argument("--interval",type=int,default=60)
    n=a.parse_args();print(json.dumps(submit(n.batch_root,n.partition,n.cpu_partition,n.account,n.submit,n.train_hours,n.eval_hours)
                         if n.command=="submit" else monitor(n.batch_root,n.interval),indent=2),flush=True)


if __name__=="__main__":main()
