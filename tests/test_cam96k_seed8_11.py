"""Full-state save neutrality and interrupted/continuous training equivalence."""
import copy
import inspect
from pathlib import Path
import numpy as np
import pytest
import torch
from analysis import cam96k_seed8_11 as p
from aoi_v2x_reproduction.runtime import runner as r, checkpointing as cp
from aoi_v2x_reproduction.runtime import full_training_checkpoint as fc
from aoi_v2x_reproduction.runtime import periodic_policy as pp
from tests.test_cam96k_v2 import tiny_config
from tests.test_training100 import assert_same_rng


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


@pytest.fixture
def batch(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"code_check",lambda expected=None,smoke=False:p.BASELINE)
    return Path(p.init(tmp_path/"study","seed8_11_test",p.BASELINE)["batch_root"])


def test_twenty_configs_and_one_hundred_units(batch):
    plan=p.preview(batch)
    assert (plan["training_runs"],plan["eval_units"],plan["training_episodes_total"],plan["training_environment_steps_total"])==(20,100,50000,5000000)
    assert plan["periodic_snapshot_records"]==plan["full_checkpoint_records"]==100
    assert plan["max_training_processes"]==2 and plan["default_eval_workers"]==8
    assert plan["pairs"]==[list(range(i,i+2)) for i in range(0,20,2)]
    assert inspect.signature(p.eval_all).parameters["workers"].default==8
    for item in plan["configs"]:
        assert item["training_seed"] in (8,9,10,11)
        assert set(item["differences_from_accepted_32k"]) <= {"seed","cam_bits","output_root","run_name"}
        assert item["periodic_episodes"]==[500,1000,1500,2000,2500]
        c=item["effective_config"]
        if "maddpg" in item["model_id"]:
            assert c["exploration_noise"]==.3 and c["target_noise_sigma"]==.2 and c["global_update_mode"]=="synchronous_joint"
    assert p.UNIFORM_STREAMS==(12,13)
    assert len(p.units(True))==40
    assert {u["family"] for u in p.units()[:8]}=={"mappo","maddpg"}


@pytest.mark.parametrize("model",p.MODELS)
def test_full_save_and_restart_are_exact(tmp_path,model,monkeypatch):
    systems=[]
    factory=r._make_system if "maddpg" in model else r._make_mappo_system
    def capture(cfg):
        system=factory(cfg);systems.append(system);return system
    monkeypatch.setattr(r,"_make_system" if "maddpg" in model else "_make_mappo_system",capture)
    cfg=tiny_config(tmp_path/"continuous",model,6)
    cfg.seed=8
    # Save at2 (pending PPO rollout),4,6 (final PPO residual consumed).
    kwargs=dict(periodic_policy_interval=2,periodic_policy_smoke=True,full_checkpoint_interval=2,full_checkpoint_smoke=True)
    runs=[];states=[];rngs=[]
    for mode in ("off","on","restart"):
        c=copy.deepcopy(cfg);c.output_root=str(tmp_path/mode)
        if mode=="restart":
            def stop(d):
                if d["episode"]==3: raise RuntimeError("test simulated interruption")
            with pytest.raises(RuntimeError,match="simulated"):
                r.train(c,progress_callback=stop,**kwargs)
            run=Path(c.output_root)/c.run_name
            # Simulate a periodic orphan and final artifacts beyond the committed boundary.
            (run/"policies/policy_episode_999999.pt").write_bytes(b"orphan")
            (run/"policy_final.pt").write_bytes(b"discarded incomplete final")
            result=r.train(c,resume_training_checkpoint=run/"training_checkpoints/training_episode_000002.pt",**kwargs)
        else:
            result=r.train(c,**(kwargs if mode=="on" else {}))
        runs.append(Path(result["run_dir"]))
        states.append(copy.deepcopy(fc.training_state(c,systems[-1])))
        rngs.append(cp.capture_rng_state())
    equal(states[0],states[1]);equal(states[0],states[2])
    assert_same_rng(rngs[0],rngs[1]);assert_same_rng(rngs[0],rngs[2])
    with np.load(runs[0]/"train_metrics.npz") as a,np.load(runs[2]/"train_metrics.npz") as b:
        equal({k:a[k] for k in a.files},{k:b[k] for k in b.files})
    for run in runs[1:]:
        fc.verify_full_run(run,[2,4,6],final_path=run/"policy_final.pt")
        pp.verify_periodic_run(run,[2,4,6],final_path=run/"policy_final.pt")
    assert list((runs[-1]/"recoveries").glob("*/policies/policy_episode_999999.pt"))
    with pytest.raises(ValueError,match="completed"):
        r.train(c,resume_training_checkpoint=runs[-1]/"training_checkpoints/training_episode_000002.pt",**kwargs)


@pytest.mark.parametrize("model",p.MODELS)
def test_resume_preserves_rolling100_best_and_ledger(tmp_path,model):
    cfg=tiny_config(tmp_path/"a",model,105)
    cfg.seed=9
    # Small real networks and two-slot episodes; eligible rolling100 exercised.
    kwargs=dict(training_reward_best=True,periodic_policy_interval=20,periodic_policy_smoke=True,
                full_checkpoint_interval=20,full_checkpoint_smoke=True)
    continuous=Path(r.train(cfg,**kwargs)["run_dir"])
    cfg.output_root=str(tmp_path/"b")
    def stop(d):
        if d["episode"]==103: raise RuntimeError("cut")
    with pytest.raises(RuntimeError,match="cut"):r.train(cfg,progress_callback=stop,**kwargs)
    resumed=Path(cfg.output_root)/cfg.run_name
    r.train(cfg,resume_training_checkpoint=resumed/"training_checkpoints/training_episode_000100.pt",**kwargs)
    assert (continuous/"training_reward_selection.csv").read_bytes()==(resumed/"training_reward_selection.csv").read_bytes()
    from aoi_v2x_reproduction.runtime.training_selection import load_training_policy
    for name in ("policy_best_training.pt","policy_final.pt"):
        _,a=load_training_policy(continuous/name);_,b=load_training_policy(resumed/name)
        equal(a["actors"],b["actors"]);equal(a["training_reward_selection"],b["training_reward_selection"])
        assert a["episode"]==b["episode"] and a["policy_version"]==b["policy_version"]


def test_loader_rejects_corrupt_seed_episode_config_and_role(tmp_path):
    cfg=tiny_config(tmp_path,"shared_tdec",5)
    def stop(d):
        if d["episode"]==2:raise RuntimeError("cut")
    kwargs=dict(periodic_policy_interval=1,periodic_policy_smoke=True,full_checkpoint_interval=1,full_checkpoint_smoke=True)
    with pytest.raises(RuntimeError):r.train(cfg,progress_callback=stop,**kwargs)
    path=Path(cfg.output_root)/cfg.run_name/"training_checkpoints/training_episode_000002.pt"
    with pytest.raises(ValueError,match="seed"):fc.load_full_checkpoint(path,expected_seed=13)
    with pytest.raises(ValueError,match="episode"):fc.load_full_checkpoint(path,expected_episode=3)
    other=copy.deepcopy(cfg);other.exploration_noise=.1
    with pytest.raises(ValueError,match="config"):fc.load_full_checkpoint(path,config=other)
    raw=path.read_bytes();path.write_bytes(raw+b"bad")
    with pytest.raises(ValueError,match="SHA256"):fc.load_full_checkpoint(path)
    path.write_bytes(raw)
    data=torch.load(path,map_location="cpu",weights_only=False);data["checkpoint_role"]="best_training100";torch.save(data,path)
    index=pp.read(path.parent/"index.json");index["records"][-1].update(sha256=pp.digest(path),size=path.stat().st_size);pp.atomic_json(path.parent/"index.json",index)
    with pytest.raises(ValueError,match="role"):fc.load_full_checkpoint(path)


def test_training_pairs_and_independent_eval_concurrency(batch,monkeypatch):
    calls=[]
    monkeypatch.setattr(p,"preflight",lambda *a:{})
    monkeypatch.setattr(p,"accepted_run",lambda b,i:{"run_id":i})
    monkeypatch.setattr(p,"verify_training",lambda *a:{"technical_status":"PASS"})
    monkeypatch.setattr(p,"summary",lambda *a:{"technical_status":"PASS"})
    monkeypatch.setattr(p,"launch_group",lambda b,c,ids,a,reason,*more:calls.append((c,tuple(ids))))
    p.train_all(batch)
    assert calls==[("train-unit",(i,i+1)) for i in range(0,20,2)]
    calls.clear()
    def forbidden(*a):pytest.fail("evaluation acquired training slot")
    monkeypatch.setattr(p,"training_slot",forbidden)
    p.eval_all(batch)
    assert [len(ids) for _,ids in calls]==[8]*12+[4]
    assert all(c=="eval-unit" for c,_ in calls)
    with pytest.raises(ValueError,match="reason"):p.train_all(batch,resume_incomplete=True)


@pytest.mark.parametrize("field,value",[("resume_supported",False),("default_eval_workers",2),("max_training_processes",8),
                                        ("periodic_interval",100),("schema","cam96k_training_pilot_v2")])
def test_contract_drift_rejected(batch,field,value):
    path=batch/"protocol/manifest.json";data=p.read(path);data[field]=value
    pp.atomic_json(path,data)
    with pytest.raises(ValueError):p.load(batch)


def test_hundred_unit_summary_counts(batch,monkeypatch):
    from tests.test_frozen_deployment import dump,synthetic_arrays,META
    monkeypatch.setattr(p,"verify_training",lambda *a:{"training_episodes_total":50000,"training_environment_steps_total":5000000,
        "periodic_snapshot_records":100,"full_checkpoint_records":100})
    records={}
    for u in p.units():
        arrays=synthetic_arrays(episodes=100)
        arrays["remaining_demand"]=np.where(arrays["success"],0,48000).astype(np.float32)
        arrays["reward_task1"]=-4.95*arrays["remaining_demand"]/96000
        rows=p.reduce_payload(arrays,{**META,"cam_bits":96000,"eval_seeds":[u["world"]]},u,100)
        record={"attempt":"a01","rows":rows};records[u["unit_id"]]=record
        dump(batch/f"records/eval_{u['unit_id']}.json",record)
    monkeypatch.setattr(p,"verify_eval",lambda b,i,a:records[i])
    report=p.summary(batch)
    for key,val in {"accepted_training_runs":20,"accepted_eval_units":100,"platoon_rows":500,"cam_deadlines":50000,
        "scored_agent_slots":5000000,"n_training_seeds_per_method":4,"periodic_snapshot_records":100,"full_checkpoint_records":100,
        "training_episodes_total":50000,"training_environment_steps_total":5000000,"smoke":False}.items():assert report[key]==val
    assert p.summary(batch)==report


def test_full_checkpoint_production_interval_dispatch_and_neutrality(tmp_path,monkeypatch):
    cfg=tiny_config(tmp_path,"shared_tdec",2500);cfg.smoke=False
    run,_=r._prepare_run(cfg,None)
    system=r._make_mappo_system(cfg);system[0].reset_world(cfg.seed)
    periodic=pp.PeriodicPolicySnapshots(run,cfg,500)
    writer=fc.FullTrainingCheckpoints(run,cfg,500)
    rng=cp.capture_rng_state()
    for ep in range(1,2501):
        if ep%500==0:
            system[1].policy_version=ep//5
        periodic.observe(ep,system[1],system[1].policy_version)
        writer.observe(ep,system,None,periodic)
    assert [row["episode"] for row in writer.records]==[500,1000,1500,2000,2500]
    assert_same_rng(rng,cp.capture_rng_state())
    assert [row["policy_version"] for row in writer.records]==[100,200,300,400,500]
    assert all(row["save_timing"]=="after_episode_after_scheduled_ppo_update" for row in writer.records)
    with pytest.raises(ValueError,match="locked"):fc.FullTrainingCheckpoints(run,cfg,1)
    with pytest.raises(ValueError,match="overwrite"):writer.observe(2500,system,None,periodic)


def test_rollback_journal_survives_interruption(tmp_path,monkeypatch):
    cfg=tiny_config(tmp_path,"shared_tdec",5)
    def stop(d):
        if d["episode"]==3:raise RuntimeError("cut")
    kwargs=dict(periodic_policy_interval=1,periodic_policy_smoke=True,full_checkpoint_interval=1,full_checkpoint_smoke=True)
    with pytest.raises(RuntimeError):r.train(cfg,progress_callback=stop,**kwargs)
    run=Path(cfg.output_root)/cfg.run_name;path=run/"training_checkpoints/training_episode_000002.pt"
    original=fc.shutil.move
    def fail_once(a,b):
        original(a,b);raise RuntimeError("rollback crash")
    monkeypatch.setattr(fc.shutil,"move",fail_once)
    with pytest.raises(RuntimeError,match="rollback crash"):r.train(cfg,resume_training_checkpoint=path,**kwargs)
    assert (run/"RECOVERY_PENDING.json").exists()
    monkeypatch.setattr(fc.shutil,"move",original)
    r.train(cfg,resume_training_checkpoint=path,**kwargs)
    assert not (run/"RECOVERY_PENDING.json").exists()
    fc.verify_full_run(run,range(1,6),final_path=run/"policy_final.pt")
    (run/"training_checkpoints/training_episode_999999.pt").write_bytes(b"unindexed")
    with pytest.raises(ValueError,match="unexpected"):
        fc.verify_full_run(run,range(1,6),final_path=run/"policy_final.pt")
    with pytest.raises(ValueError,match="unexpected"):
        fc.verify_full_index(run,range(1,6))


def test_unit_lock_slots_failure_evidence_and_verified_skip(batch,monkeypatch):
    m=p.load(batch)
    path=batch/"locks/train_0.lock"
    with p.lock(path,unit=0):
        with pytest.raises(FileExistsError):p.train_unit(batch,0)
    with p.training_slot(m["study_root"],batch):
        with p.training_slot(m["study_root"],batch):
            with pytest.raises(ValueError,match="occupied"):
                with p.training_slot(m["study_root"],batch):pass
    def fail(*a,**kw):raise RuntimeError("technical test failure")
    monkeypatch.setattr(r,"train",fail)
    with pytest.raises(RuntimeError,match="technical test failure"):p.train_unit(batch,0)
    assert len(list((batch/"records").glob("failed_0_a01_*.json")))==1
    assert not path.exists() and not list(Path(m["study_root"]).glob(".cam96k-training-slot*.lock"))
    p.write(batch/"records/train_0.json",{})
    monkeypatch.setattr(p,"accepted_run",lambda b,i:{"unit":i})
    assert p.train_unit(batch,0)=={"skipped_verified":True,"unit":0}


def test_explicit_resume_boundary_is_forwarded_without_extra_training(batch,monkeypatch):
    run=p.run_path(batch,0,"a01");(run/"training_checkpoints").mkdir(parents=True)
    p.write(run/"training_checkpoints/index.json",{"records":[{"episode":500,"file":"training_episode_000500.pt"},
        {"episode":1000,"file":"training_episode_001000.pt"}]})
    seen=[]
    monkeypatch.setattr(fc,"load_full_checkpoint",lambda path,**kw:seen.append(Path(path)))
    def inspect_train(cfg,**kw):
        assert kw["resume_training_checkpoint"]==run/"training_checkpoints/training_episode_000500.pt"
        assert kw["full_checkpoint_interval"]==kw["periodic_policy_interval"]==500
        assert kw["training_reward_best"] is True
        raise RuntimeError("inspected without training")
    monkeypatch.setattr(r,"train",inspect_train)
    with pytest.raises(ValueError,match="reason"):p.train_unit(batch,0,resume_incomplete=True)
    with pytest.raises(RuntimeError,match="inspected"):p.train_unit(batch,0,reason="technical interruption",resume_incomplete=True,resume_episode=500)
    assert seen==[run/"training_checkpoints/training_episode_000500.pt"]


def test_background_wrapper_negative_exit_and_no_receipt_overwrite(tmp_path):
    import subprocess,json
    receipt=tmp_path/"receipt.json"
    command=["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",str(p.REPO/"scripts/run_cam96k_seed8_11_supervisor.ps1"),
        "-Phase","train","-BatchRoot",str(tmp_path),"-ExpectedCommit","invalid","-Receipt",str(receipt)]
    result=subprocess.run(command,cwd=p.REPO,capture_output=True,text=True)
    assert result.returncode!=0
    data=json.loads(receipt.read_text(encoding="utf-8-sig"))
    assert data["exit_code"]!=0 and data["validation_exit_code"] is None and "HEAD mismatch" in data["error"]
    raw=receipt.read_bytes();assert subprocess.run(command,cwd=p.REPO,capture_output=True).returncode!=0
    assert receipt.read_bytes()==raw


@pytest.fixture
def attested_run(batch):
    # Synthetic deep-proof fixture: exercise only immutable-byte revalidation,
    # not algorithm semantics (those are tested by real deep acceptance/smoke).
    cfg,_=p.derive(p.load(batch)["sources"][0],p.run_path(batch,0,"a01").parent)
    run=p.run_path(batch,0,"a01");run.mkdir(parents=True)
    p.write(run/"config.resolved.json",cfg.to_dict())
    saved={**p.cell(0),"run_dir":str(run),"implementation_commit":p.BASELINE,"smoke":False}
    files={"config":"config.resolved.json","provenance":"provenance.json","complete":"COMPLETE.json","metrics":"train_metrics.npz",
        "final":"policy_final.pt","best":"policy_best_training.pt","selection_ledger":"training_reward_selection.csv"}
    for key,name in files.items():
        if key!="config":(run/name).write_bytes(b"synthetic attested bytes")
        saved[key]=p.identity(run/name)
    for folder,key,index_key in (("policies","periodic_snapshots","periodic_snapshot_index"),("training_checkpoints","full_checkpoints","full_checkpoint_index")):
        (run/folder).mkdir();records=[]
        for episode in (500,1000,1500,2000,2500):
            file=run/folder/f"test_{episode}.pt";file.write_bytes(str(episode).encode())
            records.append({"episode":episode,"file":file.name,"size":file.stat().st_size,"sha256":pp.digest(file)})
        p.write(run/folder/"index.json",{"records":records});saved[key]=records;saved[index_key]=p.identity(run/folder/"index.json")
    p.write(batch/"records/train_0.json",saved)
    records=[saved]+[{"run_id":i} for i in range(1,20)]
    p.write(batch/"records/training_verification.json",{"technical_status":"PASS","smoke":False,"accepted_training_runs":20,"records":records})
    return run,saved


def test_evaluation_attestation_rehashes_every_input_without_model_allocation(batch,attested_run,monkeypatch):
    def forbidden(*a,**kw):pytest.fail("worker allocated/loaded training state after deep gate")
    monkeypatch.setattr(torch,"load",forbidden)
    assert p.accepted_run(batch,0,evaluation_gate=True)==attested_run[1]
    (batch/"records/training_verification.json").unlink()
    with pytest.raises(FileNotFoundError):p.accepted_run(batch,0,evaluation_gate=True)


@pytest.mark.parametrize("name",["config.resolved.json","policy_best_training.pt","policy_final.pt","train_metrics.npz","COMPLETE.json",
    "policies/test_500.pt","training_checkpoints/test_1000.pt","policies/index.json"])
def test_evaluation_attestation_rejects_changed_bytes(batch,attested_run,name):
    run,_=attested_run
    with (run/name).open("ab") as f:f.write(b"corruption")
    with pytest.raises((ValueError,KeyError)):p.accepted_run(batch,0,evaluation_gate=True)
