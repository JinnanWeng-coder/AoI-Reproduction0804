"""Six-structure synthetic/short CPU tests; no HPC jobs or formal training."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis import cam96k_hpc as p
from aoi_v2x_reproduction.algorithms.mappo.trainer import MAPPOTrainer
from aoi_v2x_reproduction.runtime import runner as r, checkpointing as cp, full_training_checkpoint as fc
from tests.test_cam96k_seed8_11 import equal


def tiny(root, model, episodes=6):
    source=next(s for s in p.sources() if s["model_id"]==model and s["training_seed"]==8)
    c,_=p.derive(source,root,smoke=True)
    c.steps_per_episode=2;c.episodes=episodes
    c.actor_hidden=[8,4];c.local_critic_hidden=[8,4];c.global_critic_hidden=[8,4,2]
    c.batch_size=2;c.replay_capacity=16;c.diagnostics=False
    return c


def test_no_tdec_is_one_shared_actor_global_and_five_independent_local_sum_critics(tmp_path):
    c=tiny(tmp_path,"shared_no_tdec")
    t=MAPPOTrainer(c,torch.device("cpu"))
    assert len(t.actors)==1 and len(t.task1_critics)==5
    assert t.task2_critics is t.task2_critic_optimizer is t.task2_value_norm is None
    assert len({id(n) for n in t.task1_critics})==5
    assert t.global_critic(torch.zeros(1,5*c.state_dim)).shape==(1,1)
    assert t.task1_critics[0](torch.zeros(1,c.state_dim)).shape==(1,)
    assert t.critic is None and t.critic_structure.endswith("local_sum_values")
    counts=t.parameter_counts();assert counts["local_critic_network_count"]==5 and "task2_critics" not in counts
    tdec=MAPPOTrainer(tiny(tmp_path,"shared_tdec"),torch.device("cpu"))
    assert counts["actors"]==tdec.parameter_counts()["actors"]
    assert counts["critic"] < tdec.parameter_counts()["critic"]
    obs=np.arange(5*c.state_dim,dtype=np.float32).reshape(5,-1)/100
    first=t.tdec_values(obs);obs[1,:]+=100
    second=t.tdec_values(obs)
    np.testing.assert_array_equal(first.task1_values[[0,2,3,4]],second.task1_values[[0,2,3,4]])
    np.testing.assert_array_equal(second.task2_values,np.zeros(5))


def test_runner_local_sum_has_no_global_duplication_and_gae_then_normalization(tmp_path,monkeypatch):
    seen=[];systems=[];factory=r._make_mappo_system
    def capture(c):
        system=factory(c);systems.append(system)
        rollout=system[2];append=rollout.append
        def observe(**kw):seen.append(copy.deepcopy(kw));return append(**kw)
        rollout.append=observe
        return system
    monkeypatch.setattr(r,"_make_mappo_system",capture)
    c=tiny(tmp_path,"shared_no_tdec",2)
    run=Path(r.train(c)["run_dir"])
    arrays=np.load(run/"train_metrics.npz")
    assert seen and all(np.count_nonzero(x["task2_rewards"])==0 and np.count_nonzero(x["task2_values"])==0 for x in seen)
    # Native reward logging is untouched. Packing local sum into one rollout
    # stream must not change the environment reward definition.
    for e in range(2):
        for slot in range(2):
            np.testing.assert_array_equal(seen[e*2+slot]["task1_rewards"],
                arrays["task1_step"][e,slot]+arrays["task2_step"][e,slot])
    from aoi_v2x_reproduction.algorithms.mappo.rollout import TDecOnPolicyRollout
    rollout=TDecOnPolicyRollout(5,c.state_dim)
    for x in seen:rollout.append(**x)
    batch=rollout.consume(c.gamma,c.mappo_gae_lambda,0)
    advantages=c.global_actor_weight*batch.global_advantages.expand(-1,5)+batch.task1_advantages
    trainer=MAPPOTrainer(c,torch.device("cpu"));report=trainer.update(batch)
    np.testing.assert_allclose(report["advantage_mean_per_agent_before_normalization"],advantages.mean(0).numpy())
    assert "local_critic_loss" in report and "task2_critic_loss" not in report


@pytest.mark.parametrize("model",p.MODELS)
def test_six_full_state_saving_and_interruption_resume_are_bitwise_identical(tmp_path,monkeypatch,model):
    factory_name="_make_system" if "maddpg" in model else "_make_mappo_system"
    factory=getattr(r,factory_name);systems=[]
    def capture(c):
        s=factory(c);systems.append(s);return s
    monkeypatch.setattr(r,factory_name,capture)
    states=[];paths=[]
    kwargs=dict(periodic_policy_interval=2,periodic_policy_smoke=True,full_checkpoint_interval=2,full_checkpoint_smoke=True)
    for mode in ("plain","saved","restarted"):
        c=tiny(tmp_path/mode,model)
        if mode=="restarted":
            def stop(d):
                if d["episode"]==3:raise RuntimeError("intentional development cut")
            with pytest.raises(RuntimeError,match="development cut"):r.train(c,progress_callback=stop,**kwargs)
            run=Path(c.output_root)/c.run_name
            r.train(c,resume_training_checkpoint=run/"training_checkpoints/training_episode_000002.pt",**kwargs)
        else:r.train(c,**({} if mode=="plain" else kwargs))
        paths.append(Path(c.output_root)/c.run_name)
        states.append({"state":fc.training_state(c,systems[-1]),"rng":cp.capture_rng_state()})
    equal(states[0],states[1]);equal(states[0],states[2])
    for q in paths[1:]:
        fc.verify_full_run(q,[2,4,6],final_path=q/"policy_final.pt")
        p.verify_periodic_run(q,[2,4,6],final_path=q/"policy_final.pt")
    with np.load(paths[0]/"train_metrics.npz") as a,np.load(paths[-1]/"train_metrics.npz") as b:equal(dict(a),dict(b))


def test_fresh_36_matrix_180_world_units_and_no_external_source_payload(tmp_path):
    assert len(p.cells())==36 and {(r["model_id"],r["training_seed"]) for r in p.cells()}=={(m,s) for m in p.MODELS for s in range(8,14)}
    assert len(p.plan())==180 and {(r["model_id"],r["world"]) for r in p.plan()}=={(m,w) for m in p.MODELS for w in range(1001,1031)}
    for s in p.sources():
        c,d=p.derive(s,tmp_path)
        assert c.episodes==2500 and c.cam_bits==96000 and c.device=="cuda:0"
        assert set(d)<={"seed","cam_bits","run_name","output_root","mappo_variant"}
        if "maddpg" in s["model_id"]:assert c.exploration_noise==.3 and c.target_noise_sigma==.2


def test_training_only_six_seed_selection_and_ties():
    rows=[{**c,"score":float(c["training_seed"]),"technical_status":"PASS"} for c in p.cells()]
    assert {r["training_seed"] for r in p.choose(rows)}=={13}
    for row in rows:row["score"]=42.
    assert {r["training_seed"] for r in p.choose(rows)}=={8}
    with pytest.raises(ValueError):p.choose(rows[:-1])
    rows[0]["score"]=float("nan")
    with pytest.raises(ValueError):p.choose(rows)


def test_launcher_changes_cannot_modify_algorithm_or_protocol(monkeypatch):
    changes=["hpc/aoi_cam96k_hpc_train.sbatch"]
    def git(*a):
        if a[:2]==("rev-parse","HEAD"):return "launcher"
        if a[:2]==("branch","--show-current"):return "research/HPC"
        if a[0]=="diff":return "\n".join(changes)
        return ""
    monkeypatch.setattr(p.dc,"git",git)
    assert p.code_check("algorithm","launcher")=="launcher"
    changes.append("aoi_v2x_reproduction/config.py")
    with pytest.raises(ValueError,match="protected"):p.code_check("algorithm","launcher")


def test_cpu_array_has_no_gpu_and_dynamic_capacity_respects_free_allocations():
    path=p.dc.REPO/"hpc/submit_cam96k_hpc.py"
    spec=importlib.util.spec_from_file_location("submit_cam96k_test",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    nodes="NodeName=n1 Partitions=Q10 State=MIXED CfgTRES=cpu=32,mem=128G,gres/gpu=4,gres/gpu:l20=4 AllocTRES=cpu=8,mem=32G,gres/gpu=2,gres/gpu:l20=2"
    assert module.capacity(nodes,"Q10",4,32768,True)==2
    assert module.capacity(nodes,"Q10",4,16384,False)==6
    assert module.capacity(nodes.replace("MIXED","DRAIN"),"Q10",4,16384,True)==0
    assert module.policy_bound("acct|Q10|normal|cpu=16,gres/gpu=2||4|100|","Q10","acct",True,4,32768)==2
    cmds=module.commands("/new/batch")
    assert "--array=0-35%2" in cmds["train"] and "--array=0-239%2" in cmds["evaluate"]
    assert "--dependency=afterok:{train}" in cmds["gate"]
    assert "--dependency=afterok:{gate}" in cmds["evaluate"]
    assert "--dependency=afterok:{evaluate}" in cmds["finish"]
    assert set(cmds)=={"train","gate","evaluate","finish"}
    script=(p.dc.REPO/"hpc/aoi_cam96k_hpc_eval_cpu.sbatch").read_text()
    assert "--gres" not in script and "CUDA_VISIBLE_DEVICES=''" in script


def test_uniform_exact_validated_sampler_seed_rule_private_rng_and_60_units(tmp_path):
    from analysis import cam96k_hpc_uniform as u
    c=tiny(tmp_path,"shared_tdec");c.steps_per_episode=100
    seed=u.seed_words(12,1001);assert seed==[12,1001,0x52414E44,1]
    a=np.random.default_rng(np.random.SeedSequence(seed));b=np.random.default_rng(np.random.SeedSequence(seed))
    for _ in range(100):
        actual,decoded=u.draw_actions(c,a)
        rb=b.integers(0,3,size=5);mode=b.integers(0,2,size=5);fraction=b.random(5)
        np.testing.assert_array_equal(actual,np.column_stack((2*(rb+.5)/3-1,2*(mode+.5)/2-1,2*fraction-1)))
        np.testing.assert_array_equal(decoded[:,2],1+29*fraction)
    assert len(u.units())==60 and [r["unit_id"] for r in u.units()]==list(range(180,240))
    assert all(r["training_seed"] is None for r in u.units())
    assert len(p.all_plan())==240
    before=cp.capture_rng_state();arrays,meta=u.sample(c,12,1001,1)
    equal(before,cp.capture_rng_state())
    assert arrays["aoi_ms"].shape==(1,1,100,5) and meta["power_distribution"]=="uniform_dBm_not_mW"
    rows=p.old.reduce_payload(arrays,meta,u.units()[0],1)
    assert len(rows)==5 and sum(x["cam_episode_count"] for x in rows)==5


def test_no_tdec_resume_preserves_real_rolling100_best_final_and_role(tmp_path):
    cfg=tiny(tmp_path/"continuous","shared_no_tdec",105)
    kwargs=dict(training_reward_best=True,periodic_policy_interval=20,periodic_policy_smoke=True,
                full_checkpoint_interval=20,full_checkpoint_smoke=True)
    left=Path(r.train(cfg,**kwargs)["run_dir"])
    cfg.output_root=str(tmp_path/"restarted")
    def cut(d):
        if d["episode"]==103:raise RuntimeError("rolling100 cut")
    with pytest.raises(RuntimeError,match="rolling100 cut"):r.train(cfg,progress_callback=cut,**kwargs)
    right=Path(cfg.output_root)/cfg.run_name
    r.train(cfg,resume_training_checkpoint=right/"training_checkpoints/training_episode_000100.pt",**kwargs)
    assert (left/"training_reward_selection.csv").read_bytes()==(right/"training_reward_selection.csv").read_bytes()
    for name in (p.BEST,"policy_final.pt"):
        c,a=p.load_training_policy(left/name);_,b=p.load_training_policy(right/name)
        assert c.mappo_variant=="no_tdec" and len(a["actors"])==1
        equal(a["actors"],b["actors"]);equal(a["training_reward_selection"],b["training_reward_selection"])
        p.load_training_actors(right/name,"cpu")


def test_uniform_matches_previously_validated_source_native_arrays(tmp_path):
    """Optional historical oracle; bundle-only machines honestly skip if absent."""
    import ast,subprocess
    from analysis import cam96k_hpc_uniform as u
    try:
        sampler=subprocess.check_output(["git","show","fd99f8810a834e8976eee1fd566159a4975d9dac:analysis/run_random_training_baseline.py"],text=True,encoding="utf-8",stderr=subprocess.DEVNULL)
        lifecycle=subprocess.check_output(["git","show","fd99f8810a834e8976eee1fd566159a4975d9dac:analysis/frozen_rollout.py"],text=True,encoding="utf-8",stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:pytest.skip("historical fd99 object not present in this bundle-only checkout")
    namespace={"np":np}
    for text,names in ((sampler,{"draw_actions"}),(lifecycle,{"sequential_warm","arrays_from_records"})):
        tree=ast.parse(text);tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
        exec(compile(tree,"validated_fd99_reference","exec"),namespace)
    c=tiny(tmp_path,"shared_tdec");c.steps_per_episode=100
    from aoi_v2x_reproduction.envs.platoon import PaperEnviron
    from analysis.deployment_rng import exogenous_probe
    env=PaperEnviron(c)
    def factory(w):
        rng=np.random.default_rng(np.random.SeedSequence([12,w,0x52414E44,1]))
        return lambda observations:namespace["draw_actions"](c,rng)[0]
    raw,probes=namespace["sequential_warm"](env,[1001],factory,steps_per_episode=100,scored_episodes=1,
                                         warmup_episodes=5,deployment=True,probe=exogenous_probe)
    actual,meta=u.sample(c,12,1001,1)
    equal(actual,namespace["arrays_from_records"](raw,True));assert meta["exogenous_probes"]==probes


def test_synthetic_full_36_180_60_summary_counts_and_separate_denominators(tmp_path,monkeypatch):
    from analysis import cam96k_hpc_uniform as u
    from tests.test_frozen_deployment import synthetic_arrays,META
    batch=tmp_path/"synthetic_summary";(batch/"tables").mkdir(parents=True)
    m={"algorithm_commit":"algorithm","launcher_commit":"launcher"}
    monkeypatch.setattr(p,"load",lambda b:m)
    monkeypatch.setattr(p,"preflight",lambda *a,**kw:{"technical_status":"PASS"})
    records=[{**c,"score":float(c["training_seed"]),"technical_status":"PASS","best_episode":2100,
              "training_wall_seconds":1.,"parameter_counts":{"synthetic":True}} for c in p.cells()]
    selected=p.choose(records);monkeypatch.setattr(p,"selection",lambda b:{"candidates":records,"selected":selected})
    a=synthetic_arrays(episodes=100);a["remaining_demand"]=np.where(a["success"],0,48000).astype(np.float32)
    a["reward_task1"]=-4.95*a["remaining_demand"]/96000
    all_records={}
    for unit in p.all_plan():
        i=unit["unit_id"];meta={**META,"cam_bits":96000,"eval_seeds":[unit["world"]]}
        key={**unit,"training_seed":None if i>=180 else 13}
        rows=p.old.reduce_payload(a,meta,key,100)
        rec={"unit":unit,"rows":rows,"synthetic_only":True,"attempt_dir":str(batch/"e"/str(i))}
        all_records[i]=rec;p.write(batch/f"records/eval_{i}.json",rec)
    monkeypatch.setattr(p,"verify_eval",lambda b,i,d:{k:v for k,v in all_records[i].items() if k!="attempt_dir"})
    monkeypatch.setattr(u,"verify",lambda b,i,d:{k:v for k,v in all_records[i].items() if k!="attempt_dir"})
    report=p.summary(batch)
    assert report["eval_units"]==240 and report["learning_eval_units"]==180 and report["random_eval_units"]==60
    assert report["platoon_rows"]==900 and report["scored_agent_slots"]==9000000 and report["cam_deadlines"]==90000
    assert report["random_verification"]["platoon_rows"]==300
    assert report["random_verification"]["cam_deadlines"]==30000 and report["random_verification"]["scored_agent_slots"]==3000000
    assert len(p.csv_read(batch/"tables/training_candidates.csv"))==36
    assert len(p.csv_read(batch/"tables/selected_policy_summary.csv"))==6
    assert report["evaluated_training_seeds_per_method"]==1
    assert p.summary(batch)==report


def test_single_submission_builds_all_dependencies_once_and_records_actual_ids(tmp_path,monkeypatch):
    path=p.dc.REPO/"hpc/submit_cam96k_hpc.py";spec=importlib.util.spec_from_file_location("submit_chain_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    batch=tmp_path/"batch";(batch/"records").mkdir(parents=True);(batch/"locks").mkdir()
    monkeypatch.setattr(p,"load",lambda b:{"algorithm_commit":"algorithm","launcher_commit":"launcher"})
    monkeypatch.setattr(p,"preflight",lambda b:{"technical_status":"PASS"})
    import os
    for k in ("PROJECT_DIR","AOI_BATCH_ROOT","AOI_LAUNCHER_COMMIT","AOI_SCHEDULER_DIR","AOI_SCHEDULER_COMMIT","PYTHON_BIN",
              "AOI_ATTEMPT","AOI_TASK_OFFSET","AOI_RETRY_REASON","AOI_RESUME_EPISODE","AOI_ANALYZE_COMMAND"):
        monkeypatch.setenv(k,"old")
    calls=[]
    def fake(args):
        calls.append((args,os.environ["AOI_ANALYZE_COMMAND"]))
        return str(100+len(calls)-1)
    monkeypatch.setattr(module,"call",fake)
    preview=module.submit(batch);assert not calls and preview["no_hpc_pilot"] is True
    result=module.submit(batch,actually=True)
    assert result["jobs"]==dict(train="100",gate="101",evaluate="102",finish="103")
    assert "--dependency=afterok:100" in calls[1][0] and calls[1][1]=="select"
    assert "--dependency=afterok:101" in calls[2][0] and "--array=0-239%2" in calls[2][0]
    assert "--dependency=afterok:102" in calls[3][0] and calls[3][1]=="export"
    assert "AOI_TASK_OFFSET" not in os.environ and "AOI_RESUME_EPISODE" not in os.environ
    with pytest.raises(ValueError,match="already submitted"):module.submit(batch,actually=True)
