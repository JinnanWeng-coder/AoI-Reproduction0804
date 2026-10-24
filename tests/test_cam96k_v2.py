"""Five-family contracts, real tiny training invariance and strict policy roles."""
import copy
import inspect
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis import cam96k_training_pilot_v2 as p, cam96k_training_pilot as v1
from aoi_v2x_reproduction.runtime import runner as r, checkpointing as cp, periodic_policy as pp
from aoi_v2x_reproduction.runtime.metrics import MetricStore
from tests.test_training100 import FakeEnv, FakeTrainer, FakeRollout, assert_same_rng
from tests.test_frozen_deployment import dump, synthetic_arrays, META


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setattr(p,"code_check",lambda expected=None,smoke=False:p.BASELINE)
    return Path(p.init(tmp_path/"study","v2_formal_test",p.BASELINE)["batch_root"])


def tiny_config(tmp_path, model, episodes=5):
    source=next(s for s in p.sources() if s["model_id"]==model and s["training_seed"]==12)
    cfg,_=p.derive(source,tmp_path,True)
    cfg.episodes=episodes
    cfg.steps_per_episode=2
    cfg.actor_hidden=[8,4]
    cfg.local_critic_hidden=[8,4]
    cfg.global_critic_hidden=[8,4,2]
    cfg.batch_size=2
    cfg.replay_capacity=16
    cfg.diagnostics=False
    cfg.device="cpu"
    return cfg


def test_ten_configs_fifty_units_and_separate_concurrency(batch):
    plan=p.preview(batch)
    assert plan["training_runs"]==10 and plan["eval_units"]==50
    assert plan["training_episodes_total"]==25000 and plan["training_environment_steps_total"]==2500000
    assert plan["periodic_snapshot_records"]==50
    assert plan["pairs"]==[[0,1],[2,3],[4,5],[6,7],[8,9]]
    assert plan["max_training_processes"]==2 and plan["default_eval_workers"]==8
    assert inspect.signature(p.eval_all).parameters["workers"].default==8
    for c in plan["configs"]:
        assert c["periodic_episodes"]==[500,1000,1500,2000,2500]
        assert set(c["differences_from_accepted_32k"])<={"cam_bits","output_root","run_name"}
        if "maddpg" in c["model_id"]:
            cfg=c["effective_config"]
            assert cfg["exploration_noise"]==.3 and cfg["target_noise_sigma"]==.2
            assert cfg["global_update_mode"]=="synchronous_joint"
    assert {u["family"] for u in p.units(True)[:8]}=={"mappo","maddpg"}
    assert len(p.units(True))==20 and len(v1.cells())==6 and len(v1.units())==30
    with pytest.raises(ValueError): p.cell(10)
    assert p.read(batch/"protocol/manifest.json")["schema"] != v1.SCHEMA


@pytest.mark.parametrize("field,value",[("default_eval_workers",2),("max_training_processes",8),("periodic_interval",50),
                                        ("schema",v1.SCHEMA),("training_episodes",500),("new_unseen_confirmation",True)])
def test_v2_contract_rejects_drift(batch,field,value):
    path=batch/"protocol/manifest.json"; data=p.read(path); data[field]=value; dump(path,data)
    with pytest.raises(ValueError): p.load(batch)


@pytest.mark.parametrize("model",p.MODELS)
def test_five_real_structures_and_periodic_saving_leaves_training_identical(tmp_path,model,monkeypatch):
    # Real algorithms and environment, tiny networks/two-slot development episodes only.
    outputs=[]; states=[]; systems=[]
    factory = r._make_mappo_system if "maddpg" not in model else r._make_system
    def record_system(cfg):
        system=factory(cfg); systems.append(system); return system
    monkeypatch.setattr(r,"_make_mappo_system" if "maddpg" not in model else "_make_system",record_system)
    for enabled in (False,True):
        cfg=tiny_config(tmp_path/str(enabled),model)
        result=r.train(cfg,periodic_policy_interval=1 if enabled else None,periodic_policy_smoke=enabled)
        outputs.append(Path(result["run_dir"])); states.append(cp.capture_rng_state())
    assert_same_rng(*states)
    with np.load(outputs[0]/"train_metrics.npz") as a,np.load(outputs[1]/"train_metrics.npz") as b:
        assert set(a.files)==set(b.files)
        for key in a.files: np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    data=[torch.load(o/"policy_final.pt",map_location="cpu",weights_only=False) for o in outputs]
    for a,b in zip(data[0]["actors"],data[1]["actors"]):
        for key in a: assert torch.equal(a[key],b[key])
    from analysis.evaluate_maddpg_e1 import _cpu_clone,_nested_equal
    if "maddpg" in model:
        full=[{"agents":[a.state_dict_full() for a in system[1]],"learner":system[2].state_dict_full(),
               "replay":system[3].state_dict()} for system in systems]
        # Replay has NumPy arrays; compare those explicitly, including sampling position.
        for key,value in full[0].pop("replay").items():
            other=full[1]["replay"][key]
            if isinstance(value,np.ndarray): np.testing.assert_array_equal(value,other)
            else: assert value==other
        full[1].pop("replay")
    else:
        full=[]
        for system in systems:
            trainer=system[1]
            full.append({key:(value.state_dict() if isinstance(value,(torch.nn.Module,torch.optim.Optimizer)) else
                              [v.state_dict() for v in value]) for key,value in vars(trainer).items()
                         if isinstance(value,(torch.nn.Module,torch.optim.Optimizer)) or
                         (isinstance(value,list) and value and isinstance(value[0],torch.optim.Optimizer))})
    assert _nested_equal(_cpu_clone(full[0]),_cpu_clone(full[1])),"critics/targets/optimizers changed with saving"
    count=1 if model.startswith("shared") else 5
    assert len(data[1]["actors"])==count
    index=pp.verify_periodic_run(outputs[1],range(1,6),final_path=outputs[1]/"policy_final.pt")
    assert all(record["actor_network_count"]==count for record in index["records"])
    for record in index["records"]:
        _,actors=pp.load_periodic_actors(outputs[1]/"policies"/record["file"])
        assert len(actors)==count and all(not a.training for a in actors)
    if "maddpg" not in model:
        assert [v["policy_version"] for v in index["records"]]==[0,0,0,0,1]
        assert all(v["save_timing"]=="after_episode_after_scheduled_ppo_update" for v in index["records"])
    else:
        assert [v["policy_version"] for v in index["records"]]==[0,1,2,3,4]


def test_production_500_trigger_and_best100_preupdate_timing_without_long_training(tmp_path,monkeypatch):
    cfg=tiny_config(tmp_path,"shared_tdec",2500); cfg.smoke=False
    def factory(c):
        r.seed_everything(c.seed,torch.device("cpu"))
        return FakeEnv(c),FakeTrainer(c),FakeRollout(),MetricStore(5,2,1.,algorithm="mappo",diagnostics=False),torch.device("cpu")
    monkeypatch.setattr(r,"_make_mappo_system",factory)
    result=r.train(cfg,training_reward_best=True,periodic_policy_interval=500)
    run=Path(result["run_dir"])
    index=pp.verify_periodic_run(run,[500,1000,1500,2000,2500],final_path=run/"policy_final.pt")
    assert [row["policy_version"] for row in index["records"]]==[100,200,300,400,500]
    from aoi_v2x_reproduction.runtime.training_selection import load_training_policy,BEST
    _,best=load_training_policy(run/BEST)
    assert best["policy_version"]==(best["episode"]-1)//5
    assert best["training_reward_selection"]["save_timing"]=="after_episode_before_pending_ppo_update"
    assert index["records"][-1]["checkpoint_role"]==pp.ROLE


@pytest.fixture
def snapshot_run(tmp_path):
    cfg=tiny_config(tmp_path,"shared_tdec")
    return Path(r.train(cfg,periodic_policy_interval=1,periodic_policy_smoke=True)["run_dir"])


def test_snapshot_loader_rejects_wrong_seed_episode_role_corruption_and_final_mismatch(snapshot_run):
    path=snapshot_run/"policies/policy_episode_000005.pt"
    with pytest.raises(ValueError,match="seed"): pp.load_periodic_policy(path,expected_seed=13)
    with pytest.raises(ValueError,match="episode"): pp.load_periodic_policy(path,expected_episode=500)
    with pytest.raises(ValueError,match="expected config"): pp.load_periodic_policy(path,expected_config_hash="bad")
    raw=path.read_bytes(); path.write_bytes(raw+b"corrupt")
    with pytest.raises(ValueError,match="SHA256"): pp.load_periodic_policy(path)
    path.write_bytes(raw)
    data=torch.load(path,map_location="cpu",weights_only=False); data["checkpoint_role"]="best_training100"; torch.save(data,path)
    index=pp.read(path.parent/"index.json"); index["records"][-1].update(sha256=pp.digest(path),size=path.stat().st_size)
    pp.atomic_json(path.parent/"index.json",index)
    with pytest.raises(ValueError,match="role"): pp.load_periodic_policy(path)


def test_periodic_writer_preserves_rng_modes_no_overwrite_and_requires_explicit_smoke(tmp_path):
    cfg=tiny_config(tmp_path,"independent_maddpg_tdec")
    run,_=r._prepare_run(cfg,None)
    env,agents,learner,replay,metrics,device=r._make_system(cfg)
    agents[0].actor.eval()
    modes=[a.actor.training for a in agents]
    selector=pp.PeriodicPolicySnapshots(run,cfg,1,smoke=True)
    rng=cp.capture_rng_state()
    selector.observe(1,agents,0)
    assert_same_rng(rng,cp.capture_rng_state())
    assert [a.actor.training for a in agents]==modes
    with pytest.raises(ValueError,match="overwrite"): selector.observe(1,agents,0)
    with pytest.raises(ValueError,match="locked"): pp.PeriodicPolicySnapshots(tmp_path/"bad",cfg,1)


def test_last_snapshot_final_tensor_mismatch_rejected(snapshot_run):
    path=snapshot_run/"policy_final.pt"
    data=torch.load(path,map_location="cpu",weights_only=False)
    key=next(iter(data["actors"][0])); data["actors"][0][key].add_(1); torch.save(data,path)
    with pytest.raises(ValueError,match="tensors differ"):
        pp.verify_periodic_run(snapshot_run,range(1,6),final_path=path)


def test_shared_implementation_matches_accepted_history():
    for file in ("agent.py","learner.py","networks.py"):
        path=f"aoi_v2x_reproduction/algorithms/modified_maddpg/{file}"
        old=subprocess.check_output(["git","show",f"c459115c:{path}"],cwd=p.REPO).decode().splitlines()
        assert (p.REPO/path).read_text(encoding="utf-8").splitlines()==old


@pytest.mark.parametrize("model",("independent_maddpg_tdec","shared_maddpg_tdec"))
def test_maddpg_critics_targets_and_fixed_noise_are_the_accepted_structure(tmp_path,model):
    cfg=tiny_config(tmp_path,model)
    env,agents,learner,replay,metrics,device=r._make_system(cfg)
    sharing=model.startswith("shared")
    assert len({id(a.actor) for a in agents})==(1 if sharing else 5)
    assert len({id(a.target_actor) for a in agents})==(1 if sharing else 5)
    assert len({id(critic) for a in agents for critic in a.local_critics})==10
    assert len({id(critic) for a in agents for critic in a.target_local_critics})==10
    assert learner.global_critic1 is not learner.global_critic2
    assert learner.global_target_critic1 is not learner.global_target_critic2
    assert cfg.exploration_noise==.3 and cfg.target_noise_sigma==.2 and cfg.target_noise_clip==.5


def test_background_wrapper_rejects_bad_head_and_publishes_nonzero_receipt(tmp_path):
    import os,json
    if os.name!="nt": pytest.skip("Windows wrapper")
    receipt=tmp_path/"receipt.json"
    command=["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",str(p.REPO/"scripts/run_cam96k_v2_supervisor.ps1"),
             "-Phase","train","-BatchRoot",str(tmp_path),"-ExpectedCommit","invalid","-Receipt",str(receipt)]
    result=subprocess.run(command,cwd=p.REPO,capture_output=True,text=True)
    assert result.returncode!=0
    data=json.loads(receipt.read_text(encoding="utf-8-sig"))
    assert data["exit_code"]!=0 and data["validation_exit_code"] is None and "HEAD mismatch" in data["error"]
    before=receipt.read_bytes()
    assert subprocess.run(command,cwd=p.REPO,capture_output=True).returncode!=0
    assert receipt.read_bytes()==before


def test_training_pairs_and_eval_eight_workers_do_not_acquire_training_slots(batch,monkeypatch):
    calls=[]
    monkeypatch.setattr(p,"preflight",lambda *a:{})
    monkeypatch.setattr(p,"accepted_run",lambda b,i:{"run_id":i})
    monkeypatch.setattr(p,"verify_training",lambda *a:{"technical_status":"PASS"})
    monkeypatch.setattr(p,"summary",lambda *a:{"technical_status":"PASS"})
    monkeypatch.setattr(p,"launch_group",lambda b,c,ids,a,reason:calls.append((c,tuple(ids))))
    assert p.train_all(batch)["technical_status"]=="PASS"
    assert calls==[("train-unit",(i,i+1)) for i in range(0,10,2)]
    calls.clear()
    def forbidden(*a): pytest.fail("evaluation acquired a training slot")
    monkeypatch.setattr(p,"training_slot",forbidden)
    p.eval_all(batch)
    assert [len(ids) for _,ids in calls]==[8,8,8,8,8,8,2]
    assert all(command=="eval-unit" for command,_ in calls)


def test_fifty_unit_summary_counts_seed_units_and_worst_aggregation(batch,monkeypatch):
    monkeypatch.setattr(p,"verify_training",lambda *a:{"training_episodes_total":25000,"training_environment_steps_total":2500000,"periodic_snapshot_records":50})
    records={}
    for unit in p.units():
        arrays=synthetic_arrays(episodes=100)
        arrays["remaining_demand"]=np.where(arrays["success"],0,48000).astype(np.float32)
        arrays["reward_task1"]=-4.95*arrays["remaining_demand"]/96000
        rows=p.reduce_payload(arrays,{**META,"cam_bits":96000,"eval_seeds":[unit["world"]]},unit,100)
        record={"attempt":"a01","rows":rows}; records[unit["unit_id"]]=record
        dump(batch/f"records/eval_{unit['unit_id']}.json",record)
    monkeypatch.setattr(p,"verify_eval",lambda b,i,a:records[i])
    result=p.summary(batch)
    for k,v in {"accepted_training_runs":10,"accepted_eval_units":50,"platoon_rows":250,"cam_deadlines":25000,
                "scored_agent_slots":2500000,"n_training_seeds_per_method":2,"periodic_snapshot_records":50,
                "training_episodes_total":25000,"training_environment_steps_total":2500000,"smoke":False}.items(): assert result[k]==v
    assert p.summary(batch)==result
