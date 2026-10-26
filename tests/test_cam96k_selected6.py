"""Training-only six-seed selection and the fixed five-policy/30-world contract."""
import copy
import inspect
from pathlib import Path

import numpy as np
import pytest

from analysis import cam96k_selected6 as p
from tests.test_frozen_deployment import synthetic_arrays, META


def records():
    return [{"model_id":model,"training_seed":seed,"score":float(seed),"checkpoint_role":"best_training100"}
            for model in p.MODELS for seed in p.SEEDS]


@pytest.fixture
def batch(tmp_path,monkeypatch):
    monkeypatch.setattr(p.t,"code_check",lambda expected=None,smoke=False:"testcommit")
    study=tmp_path/"study";legacy=study/p.t.ROOT/"legacy";(legacy/"protocol").mkdir(parents=True);(legacy/"records").mkdir()
    p.write(legacy/"protocol/manifest.json",{"schema":"cam96k_training_pilot_v2","smoke":False,"implementation_commit":p.t.BASELINE,
        "cam_bits":96000,"training_episodes":2500,"slots":100,"rule":p.RULE})
    p.write(legacy/"records/training_verification.json",{"technical_status":"PASS","smoke":False,"accepted_training_runs":10,
        "records":[{"model_id":m,"training_seed":s} for m in p.MODELS for s in (12,13)]})
    return Path(p.init(study,"selected6_test","testcommit",legacy=legacy)["batch_root"])


@pytest.fixture
def selected(batch,tmp_path,monkeypatch):
    rows=records()
    for i,row in enumerate(rows):
        file=tmp_path/f"policy_{i}.pt";file.write_bytes(f"synthetic accepted {i}".encode())
        row.update(best=p.identity(file),artifacts={"best":p.identity(file)})
    monkeypatch.setattr(p,"candidates",lambda b:copy.deepcopy(rows))
    p.select(batch)
    return batch,rows


@pytest.mark.parametrize("winner",p.SEEDS)
def test_each_of_the_six_training_seeds_can_win_using_only_best100_reward(winner):
    rows=records()
    for row in rows:
        row["score"]=100. if row["training_seed"] == winner else -100.
        row["final_episode_reward"]=1e9 if row["training_seed"] != winner else -1e9
        row["evaluation_reward"]=row["final_episode_reward"]
    assert {r["training_seed"] for r in p.choose(rows)} == {winner}
    assert len(p.choose(rows)) == 5


def test_exact_score_tie_chooses_lowest_seed_independent_of_input_order():
    rows=records()
    for row in rows:row["score"]=42.
    assert [r["training_seed"] for r in p.choose(rows[::-1])] == [8]*5


@pytest.mark.parametrize("mutation",["missing","duplicate","nonfinite","final_role"])
def test_reject_incomplete_or_invalid_candidates(mutation):
    rows=records()
    if mutation == "missing":rows.pop()
    elif mutation == "duplicate":rows[-1]=copy.deepcopy(rows[0])
    elif mutation == "nonfinite":rows[0]["score"]=float("nan")
    else:rows[0]["checkpoint_role"]="final_training100"
    with pytest.raises(ValueError):p.choose(rows)


def test_preview_delegates_twenty_unchanged_training_runs_and_plans_150(batch):
    plan=p.preview(batch)
    assert plan["new_training_runs"] == 20 and plan["existing_training_runs"] == 10 and plan["candidate_runs"] == 30
    assert plan["candidate_training_seeds"] == [8,9,10,11,12,13]
    assert plan["selected_policies"] == 5 and plan["evaluated_training_seeds_per_method"] == 1 and plan["eval_units"] == 150
    assert plan["worlds"] == list(range(1001,1031))
    assert plan["training_episodes_total"] == 50000 and plan["training_environment_steps_total"] == 5000000
    assert plan["new_periodic_policy_records"] == plan["new_full_checkpoint_records"] == 100
    assert plan["max_training_processes"] == 2 and plan["default_eval_workers"] == 8
    assert inspect.signature(p.evaluate).parameters["workers"].default == 8
    assert p.load(batch)["schema"] != p.t.SCHEMA


def test_no_selection_until_all_new_training_passes(batch,monkeypatch):
    monkeypatch.setattr(p,"audit_candidate",lambda row,*a:row)
    def incomplete(*a):raise FileNotFoundError("missing new training")
    monkeypatch.setattr(p.t,"verify_training",incomplete)
    with pytest.raises(FileNotFoundError):p.select(batch)
    assert not (batch/"records/selection.json").exists()
    assert not (batch/"locks/selection.lock").exists()


def test_selection_is_sealed_and_changed_candidate_bytes_are_rejected(selected):
    batch,rows=selected
    first=(batch/"records/selection.json").read_bytes()
    p.select(batch)
    assert (batch/"records/selection.json").read_bytes() == first
    rows[0]["score"]=1000.
    with pytest.raises(ValueError,match="sealed"):p.select(batch)
    assert (batch/"records/selection.json").read_bytes() == first
    file=Path(rows[-1]["best"]["path"]);file.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="input changed"):p.selection(batch)


def test_each_method_uses_one_checkpoint_on_all_thirty_worlds(selected):
    batch,_=selected;units=p.units(batch)
    assert len(units) == 150 and [u["unit_id"] for u in units] == list(range(150))
    for model in p.MODELS:
        group=[u for u in units if u["model_id"] == model]
        assert [u["world"] for u in group] == list(range(1001,1031))
        assert len({u["training_seed"] for u in group}) == len({u["checkpoint_sha256"] for u in group}) == 1
        assert all(u["repeat"] == 0 for u in group)
        if "maddpg" in model:assert all(u["noise_std"] == 0 for u in group)
    assert {u["family"] for u in units[:8]} == {"maddpg","mappo"}


def test_eight_worker_queue_and_unit_slots_do_not_consume_training_slots(selected,monkeypatch):
    batch,_=selected;calls=[]
    monkeypatch.setattr(p,"preflight",lambda b:{"technical_status":"PASS"})
    monkeypatch.setattr(p,"launch",lambda b,ids,a,r:calls.append(list(ids)))
    monkeypatch.setattr(p,"summary",lambda b:{"technical_status":"PASS"})
    p.evaluate(batch)
    assert [len(c) for c in calls] == [8]*18+[6]
    assert sum(calls,[]) == list(range(150))
    study=Path(p.load(batch)["study_root"])
    assert not list(study.glob(".cam96k-training-slot*.lock"))
    with p.eval_slot(batch):
        assert len(list(study.glob(".cam96k-selected6-eval-slot*.lock"))) == 1
    assert not list(study.glob(".cam96k-selected6-eval-slot*.lock"))


def test_selected_summary_150_units_and_no_six_seed_average(selected,monkeypatch):
    batch,_=selected;accepted={}
    for u in p.units(batch):
        a=synthetic_arrays(episodes=100)
        a["remaining_demand"]=np.where(a["success"],0,48000).astype(np.float32)
        a["reward_task1"]=-4.95*a["remaining_demand"]/96000
        rows=p.t.reduce_payload(a,{**META,"cam_bits":96000,"eval_seeds":[u["world"]]},u,100)
        saved={"rows":rows,"attempt":"a01"};accepted[u["unit_id"]]=saved;p.write(batch/f"records/eval_{u['unit_id']}.json",saved)
    monkeypatch.setattr(p,"verify_eval",lambda b,i,a:accepted[i])
    report=p.summary(batch)
    for key,value in {"accepted_candidate_training_runs":30,"n_candidate_training_seeds_per_method":6,"selected_policy_count":5,
        "n_evaluated_training_seeds_per_method":1,"accepted_eval_units":150,"platoon_rows":750,"cam_deadlines":75000,
        "scored_agent_slots":7500000,"selection_uses_evaluation":False,"smoke":False}.items():assert report[key] == value
    assert not (batch/"tables/across_seeds.csv").exists()
    assert p.summary(batch) == report
