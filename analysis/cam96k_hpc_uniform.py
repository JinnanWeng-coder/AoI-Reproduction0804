"""Validated Uniform sampler/RNG from fd99f8810a834e8, new CAM96k CPU outputs.

RB and mode drawn separately, followed by uniform power FRACTION -> uniform dBm.
No training, selection, policy checkpoint or learner. Streams12/13 are not seeds.
"""
import copy
from pathlib import Path
import numpy as np
from analysis import cam96k_hpc as p
from analysis.deployment_rng import exogenous_probe
from analysis.frozen_rollout import sequential_warm, arrays_from_records
from aoi_v2x_reproduction.envs.platoon import PaperEnviron

STREAMS=(12,13)


def seed_words(stream,world):
    if type(stream) is not int or type(world) is not int or min(stream,world)<0:raise ValueError("invalid Random stream/world")
    return [stream,world,0x52414E44,1]


def draw_actions(cfg,rng):
    rb=rng.integers(0,cfg.n_rb,size=cfg.number_agents)
    mode=rng.integers(0,cfg.n_modes,size=cfg.number_agents)
    fraction=rng.random(cfg.number_agents)
    normalized=np.column_stack((2*(rb+.5)/cfg.n_rb-1,2*(mode+.5)/cfg.n_modes-1,2*fraction-1))
    decoded=np.column_stack((rb,mode,cfg.power_min_dbm+(cfg.power_max_dbm-cfg.power_min_dbm)*fraction))
    return normalized,decoded


def units():
    return [{"unit_id":i+180,"model_id":"uniform_random","action_stream_id":s,"training_seed":None,
             "world":w,"repeat":None,"action_seed_words":seed_words(s,w)}
            for i,(s,w) in enumerate((s,w) for s in STREAMS for w in p.WORLDS)]


def config(batch):
    m=p.load(batch); cfg,_=p.derive(m["sources"][0],Path(batch)/"tmp",smoke=m["smoke"])
    cfg.device="cpu";cfg.checkpoint_mode="none"
    p.require(cfg.power_min_dbm==1 and cfg.power_max_dbm==30,"Uniform requires historical1-30dBm action range")
    return cfg


def sample(cfg,stream,world,scored=100):
    env=PaperEnviron(cfg);boundary={}
    def factory(w):
        rng=np.random.default_rng(np.random.SeedSequence(seed_words(stream,w)))
        def act(obs):
            before=copy.deepcopy(env.rng.bit_generator.state)
            a,decoded=draw_actions(cfg,rng)
            p.require(env.rng.bit_generator.state==before,"Random consumed physical RNG")
            p.require(np.allclose(env.decode_actions(a),decoded,rtol=0,atol=1e-12),"Uniform decoder mismatch")
            return a
        return act
    raw,probes=sequential_warm(env,[world],factory,steps_per_episode=100,scored_episodes=scored,warmup_episodes=5,
        deployment=True,probe=exogenous_probe,after_warmup=lambda w,e:boundary.update(warmup_end_aoi=e.aoi.copy().tolist()))
    arrays=arrays_from_records(raw,True)
    meta={"status":"complete","runtime_config":cfg.to_dict(),"event_source":"post_step_aoi_ms_isclose_1_verified_env_step",
          "v2i_min":float(env.v2i_min),"global_actor_weight":cfg.global_actor_weight,"cam_bits":96000,
          "eval_seeds":[world],"eval_episodes":scored,"eval_warmup_episodes":5,"steps_per_episode":100,"number_agents":5,
          "n_rb":3,"target_scenario":p.dc.SCENARIO,"eval_protocol":"sequential_warm","world_role":p.old.WORLD_ROLE,
          "exogenous_probes":probes,"action_seed_words":seed_words(stream,world),"action_stream_id":stream,
          "action_stream_ids_are_training_seeds":False,"training_seed":None,"trained_models":0,"device":"cpu",
          "sampler_source_commit":"fd99f8810a834e8976eee1fd566159a4975d9dac",
          "power_distribution":"uniform_dBm_not_mW","draw_order":["rb[P]","mode[P]","power_fraction[P]"],**boundary}
    return arrays,meta


def verify(batch,i,dest):
    m=p.load(batch);u=units()[i-180];meta=p.read(dest/"EVAL_COMPLETE.json")
    p.require(meta["runtime_config"]==config(batch).to_dict() and meta["eval_episodes"]==100 and meta["eval_warmup_episodes"]==5 and
              meta["action_stream_id"]==u["action_stream_id"] and meta["eval_seeds"]==[u["world"]] and meta["device"]=="cpu" and
              meta["launcher_commit"]==m["launcher_commit"] and meta["algorithm_commit"]==m["algorithm_commit"] and
              meta["action_seed_words"]==u["action_seed_words"],"Random identity/protocol changed")
    a=p.load_arrays(dest/"metrics.npz");rows=p.old.reduce_payload(a,meta,u,100)
    previous=np.asarray(meta["warmup_end_aoi"])
    first=a["aoi_ms"][0,0,0]
    p.require(np.array_equal(first,np.where(first==1,1,np.minimum(100,previous+1))),"Random warmup boundary discontinuity")
    return {"unit":u,"rows":rows,"metrics":p.identity(dest/"metrics.npz"),"native_complete":p.identity(dest/"EVAL_COMPLETE.json")}


def execute(batch,i,attempt="a01",reason=None):
    m=p.load(batch);p.require(180<=i<240 and (attempt=="a01" or reason),"invalid Random ID/retry")
    with p.old.lock(Path(batch)/f"locks/eval_{i}.lock",batch=str(batch),unit_id=i):
        saved=Path(batch)/f"records/eval_{i}.json";dest=Path(batch)/"e"/str(i)/attempt
        if saved.exists():
            r=p.read(saved);p.require(verify(batch,i,Path(r["attempt_dir"]))=={k:v for k,v in r.items() if k!="attempt_dir"},"accepted Random changed");return r
        if not (dest/"EVAL_COMPLETE.json").exists():
            p.require(not dest.exists(),"partial Random attempt retained; inspect or use new attempt")
            cfg=config(batch);u=units()[i-180];a,meta=sample(cfg,u["action_stream_id"],u["world"])
            meta.update(algorithm_commit=m["algorithm_commit"],launcher_commit=m["launcher_commit"])
            dest.mkdir(parents=True);np.savez_compressed(dest/"metrics.npz",**a)
            p.write(dest/"EVAL_COMPLETE.json",meta)
        r={**verify(batch,i,dest),"attempt_dir":str(dest)};p.write(saved,r);return r


def summary(batch):
    rows=[]
    for i in range(180,240):
        saved=p.read(Path(batch)/f"records/eval_{i}.json")
        p.require(verify(batch,i,Path(saved["attempt_dir"]))=={k:v for k,v in saved.items() if k!="attempt_dir"},"Random result changed")
        rows.extend(saved["rows"])
    p.require(len(rows)==300 and sum(r["slot_count"] for r in rows)==3000000 and sum(r["cam_episode_count"] for r in rows)==30000,
              "Random denominator mismatch")
    p.require({(r["action_stream_id"],r["world"],r["platoon"]) for r in rows}=={(s,w,a) for s in STREAMS for w in p.WORLDS for a in range(5)},
              "Random stream/world/platoon set mismatch")
    file=Path(batch)/"tables/random_world_platoon.csv"
    if not file.exists():p.csv_write(file,rows)
    else:
        expected=[{k:"" if row.get(k) is None else str(row[k]) for k in sorted({k for r in rows for k in r})} for row in rows]
        p.require(p.csv_read(file)==expected,"existing Random table changed")
    result={"technical_status":"PASS","eval_units":60,"platoon_rows":300,"cam_deadlines":30000,"scored_agent_slots":3000000,
            "trained_models":0,"action_stream_ids_are_training_seeds":False,"summary":p.old.aggregate(rows)}
    p.write(Path(batch)/"tables/random_verification.json",result);return result
