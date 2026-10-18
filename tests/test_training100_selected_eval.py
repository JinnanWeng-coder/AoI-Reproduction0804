"""Synthetic actors/environments only; never loads lab weights or runs real training."""
import copy
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis import evaluate_training100_selected as se
from analysis import evaluate_mappo_zero_shot as zs, evaluate_maddpg_e1 as md
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime import checkpointing as cp, training_selection as ts
from tests.test_training100 import FakeTrainer, FakeAgent
from tests.test_frozen_deployment import dump, FakeEnvironment, synthetic_arrays, META


def fake_training_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(se.tc, "code_check", lambda expected=None: "eval_commit")
    monkeypatch.setattr(cp, "_git_stamp", lambda: {"reproduction_git_commit": "training_commit"})
    study = tmp_path / "study"
    batch = study / se.tc.ROOT / "lab_synthetic"
    dump(batch / "protocol/manifest.json", {"batch_root": str(batch), "study_root": str(study),
         "implementation_commit": "training_commit", "rule": ts.RULE})
    records, curves = [], []
    for i in range(24):
        item = se.tc.cell(i)
        spec = se.dc.MODELS[item["model_id"]]
        run = se.tc.attempt_run(batch, i, "a02" if i == 2 else "a01")
        cfg = resolve_config(algorithm="mappo" if spec["family"] == "mappo" else "modified_maddpg_tdec",
                             scenario=se.dc.SCENARIO, seed=item["training_seed"], episodes=2500, steps_per_episode=100,
                             mappo_variant=spec["variant"], mappo_actor_sharing=spec["structure"] == "shared",
                             actor_hidden=[8, 4], device="cpu", output_root=str(run.parent), run_name=run.name)
        if spec["family"] == "maddpg":
            cfg.profile = "paper_faithful"  # Historical profile must not be relabeled.
        dump(run / "config.resolved.json", cfg.to_dict())
        score = float(cfg.seed)
        bv = 19 if spec["family"] == "mappo" else max(0, 10000 - cfg.batch_size + 1) // cfg.policy_delay
        fv = 500 if spec["family"] == "mappo" else max(0, 250000 - cfg.batch_size + 1) // cfg.policy_delay
        best = {"rule": ts.RULE, "selected_episode": 100, "window_start": 1, "window_end": 100,
                "score": score, "policy_version": bv, "save_timing": "synthetic_only"}
        selection = {"rule": ts.RULE, "best": best, "final_episode": 2500, "final_policy_version": fv}
        dump(run / "COMPLETE.json", {"status": "complete", "algorithm": cfg.algorithm,
             "config_hash": cfg.canonical_hash(), "reproduction_git_commit": "training_commit",
             "training_reward_selection": selection})
        policy = FakeTrainer(cfg) if spec["family"] == "mappo" else [FakeAgent(cfg) for _ in range(5)]
        for role, episode, version, name in (("best_training100", 100, bv, ts.BEST),
                                              ("final_training100", 2500, fv, "policy_final.pt")):
            data = cp.build_policy_payload(cfg, policy, episode)
            data.update(training_completed=True, checkpoint_role=role, selected_episode=episode, policy_version=version,
                        training_reward_selection=best if episode == 100 else {"rule": ts.RULE, "policy_version": fv})
            torch.save(data, run / name)
        local_rows = [{"episode": e, "episode_reward": score if e <= 100 else -score} for e in range(1, 2501)]
        se.csv_write(run / "training_reward_selection.csv", local_rows)
        curves.extend({**item, **r} for r in local_rows)
        record = {**item, "technical_status": "PASS", "from_scratch": True,
                  "implementation_commit": "training_commit", "run_dir": str(run), "best_episode": 100,
                  "best_training100_score": score, "last100_training_score": -score,
                  "best_policy_version": bv, "final_policy_version": fv,
                  "best": se.dc.file_identity(run / ts.BEST), "final": se.dc.file_identity(run / "policy_final.pt")}
        dump(batch / f"records/train_{i:02d}.json", record)
        records.append(record)
    summary = batch / "analysis/a01"
    dump(summary / "verification.json", {"technical_status": "PASS", "accepted_runs": 24, "episode_rows": 60000,
         "training_env_steps": 6000000, "implementation_commit": "training_commit"})
    se.csv_write(summary / "selected_policies.csv", records)
    se.csv_write(summary / "training_episode_curves.csv", curves)
    return batch, records


@pytest.fixture
def training(tmp_path, monkeypatch):
    return fake_training_batch(tmp_path, monkeypatch)


def test_selection_two_distinct_rules_exact_ties_and_300_unique_keys(training):
    batch, records = training
    assert se.candidates(batch) == records
    selected = se.select(records)
    assert [s["record"]["training_seed"] for s in selected] == [13] * 4 + [8] * 4
    units = se.make_units(selected)
    assert len(units) == 300
    assert len({(u["condition"], u["world"], u["repeat"]) for u in units}) == 300
    assert {u["repeat"] for u in units} == {0}
    assert {u["world"] for u in units} == set(range(1001, 1031))
    assert sum(u["model_id"] == "algorithm2" for u in units) == 120
    for kind in se.ROLES:
        md_units = [u for u in units if u["checkpoint_kind"] == kind and u["model_id"] == "algorithm2"]
        assert len({u["source_id"] for u in md_units}) == 1
    tied = copy.deepcopy(records)
    for r in tied:
        r["best_training100_score"] = r["last100_training_score"] = 1
    assert all(s["record"]["training_seed"] == 8 for s in se.select(tied))
    tied[0]["best_training100_score"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        se.select(tied)


def test_reject_missing_seed_and_corrupt_curve(training):
    batch, records = training
    with pytest.raises(ValueError, match="six seeds"):
        se.select(records[:-1])
    path = batch / "analysis/a01/training_episode_curves.csv"
    rows = se.csv_read(path)
    rows[0]["episode_reward"] = "9999"
    path.unlink()
    se.csv_write(path, rows)
    with pytest.raises(ValueError, match="recomputed"):
        se.candidates(batch)


def test_init_strict_payload_preflight_fixed_sources_and_no_overwrite(training):
    batch, _ = training
    outcome = se.init(batch, "selected_test", "eval_commit")
    out = Path(outcome["batch_root"])
    report = se.preflight(out, "cpu")
    assert report["loaded_selected_payloads"] == 8 and report["real_rollouts_in_preflight"] == 0
    assert se.preflight(out, "cpu") == report
    m, sources, units = se.load(out)
    assert m["world_role"] == se.WORLD_ROLE and m["new_unseen_confirmation"] is False
    assert len(sources) == 8 and len(units) == 300
    assert se.candidates(batch)[2]["run_dir"].endswith("a02/runs/train100_shared_tdec_seed10") or "a02\\runs" in se.candidates(batch)[2]["run_dir"]
    with pytest.raises(ValueError, match="unique"):
        se.init(batch, "selected_test", "eval_commit")
    source = sources[0]
    Path(source["checkpoint"]["path"]).write_bytes(b"changed source must never be accepted")
    with pytest.raises(ValueError, match="source file changed"):
        se.source_check(source)


@pytest.mark.parametrize("model", list(se.dc.MODELS))
@pytest.mark.parametrize("kind", list(se.ROLES))
def test_reused_rollout_loops_load_real_synthetic_actor_artifacts(training, monkeypatch, tmp_path, model, kind):
    batch, _ = training
    record = se.select(se.candidates(batch))[[s["source_id"] for s in se.select(se.candidates(batch))].index(f"{model}__{kind}")]["record"]
    path = Path(record["best" if kind == "best100" else "final"]["path"])
    episode = 100 if kind == "best100" else 2500
    # All environment calls use a test double; strict actor loading stays real.
    monkeypatch.setattr(zs, "PaperEnviron", FakeEnvironment)
    import aoi_v2x_reproduction.envs.platoon as env_module
    monkeypatch.setattr(env_module, "PaperEnviron", FakeEnvironment)
    monkeypatch.setattr(md, "_make_system", lambda *a: pytest.fail("portable actors must not build learners/replay"))
    if model != "algorithm2":
        result = zs.evaluate_zero_shot(path, tmp_path / "evaluation", se.dc.SCENARIO, "cpu", [1001], 1,
                                      deployment=True, world_role=se.WORLD_ROLE, training_policy=True)
        assert result["policy_episode"] == episode and result["policy_parameters_unchanged"] is True
        assert result["critic_constructed"] is False
    else:
        run = path.parent
        override = {"run_dir": run, "resolved_run_dir": run, "checkpoint": path,
                    "canonical_run_name": run.name, "historical_run_name": run.name, "algorithm_label": "algorithm2",
                    "is_symlink": False, "complete": se.dc.read(run / "COMPLETE.json"), "manifest": {}}
        results = []
        for noise in (0., .3):
            result = md.evaluate_maddpg_e1("modified_maddpg_tdec", noise, record["training_seed"], tmp_path,
                                          tmp_path / f"evaluation_{noise}", "cpu", [1001], 1, 5, episode,
                                          target_scenario=se.dc.SCENARIO, source_override=override, deployment=True,
                                          world_role=se.WORLD_ROLE, training_policy=True)
            assert result["checkpoint_episode"] == episode and result["optimizer_steps"] == 0
            assert result["parameter_snapshot_unchanged"] and not result["critic_constructed"]
            results.append(result)
        assert results[0]["exogenous_probes"] == results[1]["exogenous_probes"]
        for field in ("config_hash", "agents", "learner"):
            assert field not in torch.load(path, map_location="cpu", weights_only=False)
    assert result["checkpoint_role"] == se.ROLES[kind] and result["training_budget_episodes"] == 2500
    source = {"record": record, "policy_episode": episode, "policy_version": result["policy_version"],
              "checkpoint_kind": kind, "checkpoint": record["best" if kind == "best100" else "final"],
              "config_hash": result["source_config_hash"]}
    unit = next(u for u in se.make_units(se.select(se.candidates(batch))) if u["model_id"] == model and
                u["checkpoint_kind"] == kind and u["world"] == 1001 and u["noise_std"] == result.get("noise_std"))
    native = {**result, "eval_episodes": 100, "reproduction_git_commit": "eval_commit", "reproduction_git_dirty": False}
    se.native_check(native, unit, source, "eval_commit")
    field = "policy_episode" if model != "algorithm2" else "checkpoint_episode"
    with pytest.raises(ValueError, match=field):
        se.native_check({**native, field: 500}, unit, source, "eval_commit")
    rows = se.reduce_arrays(se.load_arrays(Path(result["eval_dir"]) / "metrics.npz"), result,
                            {"world": 1001}, expected=(1, 1, 100, 5))
    assert len(rows) == 5 and sum(r["slot_count"] for r in rows) == 500
    assert result["source_config_hash"] == se.dc.read(path.parent / "COMPLETE.json")["config_hash"]
    assert se.dc.file_identity(path) == record["best" if kind == "best100" else "final"]


def test_explicit_opt_in_guards(tmp_path):
    with pytest.raises(ValueError, match="deployment"):
        zs.evaluate_zero_shot(tmp_path / "x", tmp_path, se.dc.SCENARIO, training_policy=True)
    with pytest.raises(ValueError, match="deployment/source"):
        md.evaluate_maddpg_e1("modified_maddpg_tdec", 0, 8, tmp_path, tmp_path, training_policy=True)


def test_complete_300_summary_is_n1_no_seed_ci_and_compact_export(training, monkeypatch):
    batch, _ = training
    out = Path(se.init(batch, "summary_test", "eval_commit")["batch_root"])
    m, sources, units = se.load(out)
    synthetic = synthetic_arrays(episodes=100)
    def accept(b, unit, source, manifest, attempt, create=False):
        rows = se.reduce_arrays(synthetic, {**META, "eval_seeds": [unit["world"]]}, unit)
        return rows, {"world": unit["world"], "initial": "identical", "final": "identical"}
    monkeypatch.setattr(se, "accept", accept)
    for u in units:
        dump(out / f"records/unit_{u['unit_id']:03d}.json", {"attempt": "a01"})
    result = se.summarize(out)
    assert result["accepted_units"] == 300 and result["scored_agent_slots"] == 15000000
    assert result["cam_deadlines"] == 150000
    with zipfile.ZipFile(result["archive"]) as z:
        assert z.testzip() is None and not any(n.endswith((".pt", ".npz", ".log")) for n in z.namelist())
        assert "protocol/selected_models.csv" in z.namelist()
    rows = se.csv_read(out / "analysis/a01/condition_summary.csv")
    assert all(r["n_training_seeds"] == "1" and not r["descriptive_t95_lower"] and not r["training_seed_sd"] for r in rows)
    assert len(se.csv_read(out / "analysis/a01/world_repeat_platoon.csv")) == 1500
    with pytest.raises(FileExistsError):
        se.summarize(out)


def test_wrong_native_episode_and_partial_attempt_are_rejected(training):
    batch, _ = training
    out = Path(se.init(batch, "partial_test", "eval_commit")["batch_root"])
    se.preflight(out, "cpu")
    m, sources, units = se.load(out)
    with pytest.raises(ValueError, match="native status"):
        se.native_check({"policy_episode": 500}, units[0], sources[0], m["implementation_commit"])
    (out / "units/unit_000/a01").mkdir(parents=True)
    with pytest.raises(ValueError, match="partial attempt preserved"):
        se.execute(out, 0, "cpu")


def test_preview_cli_is_read_only(training, capsys):
    batch, _ = training
    assert se.main(["preview", "--training-batch", str(batch)]) == 0
    assert json.loads(capsys.readouterr().out)["units"] == 300
    assert not (batch.parents[2] / se.ROOT).exists()


@pytest.mark.parametrize("kind", list(se.ROLES))
def test_native_array_acceptance_exact_skip_and_checkpoint_budget_distinction(training, kind):
    batch, _ = training
    out = Path(se.init(batch, "accept_test", "eval_commit")["batch_root"])
    m, sources, units = se.load(out)
    unit = next(u for u in units if u["model_id"] == "shared_tdec" and u["checkpoint_kind"] == kind)
    source = next(s for s in sources if s["source_id"] == unit["source_id"])
    dest = out / "units" / f"unit_{unit['unit_id']:03d}" / "a01"
    native = dest / "native"
    native.mkdir(parents=True)
    dump(dest / "STARTED.json", {"unit": unit, "source": source, "implementation_commit": "eval_commit"})
    metadata = {**META, "status": "complete", "training_seed": unit["training_seed"],
                "eval_seeds": [unit["world"]], "environment_seeds": [unit["world"]],
                "action_repeat": 0, "action_seed_words": [unit["action_seed_words"]],
                "eval_protocol": "sequential_warm", "eval_warmup_episodes": 5, "eval_episodes": 100,
                "target_scenario": se.dc.SCENARIO, "source_scenario": se.dc.SCENARIO,
                "source_config_hash": source["config_hash"], "training_source_commit": "training_commit",
                "reproduction_git_commit": "eval_commit", "reproduction_git_dirty": False,
                "world_role": se.WORLD_ROLE, "checkpoint_role": se.ROLES[kind],
                "policy_version": source["policy_version"], "training_budget_episodes": 2500,
                "policy_episode": source["policy_episode"], "policy_parameters_unchanged": True,
                "mappo_eval_mode": "stochastic", "actor_sharing": True, "mappo_variant": "tdec",
                "policy_sampling_seeds": [se.policy_seed(unit["training_seed"], unit["world"], 0)],
                "raw_metric_axes": {"aoi_ms": ["eval_seed", "scored_episode", "slot", "agent"]},
                "exogenous_probes": [{"world": unit["world"], "initial": {}, "final": {}}],
                "policy": os.path.relpath(source["checkpoint"]["path"], native)}
    dump(native / "EVAL_COMPLETE.json", metadata)
    dump(native / "provenance.json", {"reproduction_git_commit": "eval_commit"})
    np.savez_compressed(native / "metrics.npz", **synthetic_arrays(episodes=100))
    rows, _ = se.accept(out, unit, source, m, "a01", create=True)
    assert sum(r["slot_count"] for r in rows) == 50000
    assert rows[0]["training_environment_steps"] == source["policy_episode"] * 100
    assert rows[0]["training_budget_environment_steps"] == 250000
    assert se.accept(out, unit, source, m, "a01")[0] == rows
    se.preflight(out, "cpu")
    assert se.execute(out, unit["unit_id"], "cpu")["skipped_verified"]
    csv_path = dest / "world_repeat_platoon.csv"
    bad = se.csv_read(csv_path)
    bad[0]["slot_count"] = "9999"
    csv_path.unlink()
    se.csv_write(csv_path, bad)
    with pytest.raises(ValueError, match="CSV mismatch"):
        se.accept(out, unit, source, m, "a01")


@pytest.mark.parametrize("failure", [None, 0])
def test_local_queue_bounded_children_skip_and_stop_without_retry(training, monkeypatch, failure):
    batch, _ = training
    out = Path(se.init(batch, "queue_test", "eval_commit")["batch_root"])
    monkeypatch.setattr(se, "preflight", lambda *a: {"technical_status": "PASS"})
    monkeypatch.setattr(se.time, "sleep", lambda *a: None)
    state = {"live": 0, "max": 0, "launched": [], "verified": []}
    dump(out / "records/unit_003.json", {"synthetic_accepted": True})
    monkeypatch.setattr(se, "execute", lambda b, i, *a: state["verified"].append(i))
    class Child:
        def __init__(self, command, **kw):
            self.i = int(command[command.index("--id") + 1])
            self.pid = 100 + self.i
            self.code = 1 if self.i == failure else 0
            self.done = False
            state["live"] += 1
            state["max"] = max(state["max"], state["live"])
            state["launched"].append(self.i)
            assert kw["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        def poll(self):
            if not self.done:
                self.done = True
                state["live"] -= 1
                if self.code == 0:
                    dump(out / f"records/unit_{self.i:03d}.json", {"synthetic_accepted": True})
            return self.code
    monkeypatch.setattr(se.subprocess, "Popen", Child)
    if failure is not None:
        with pytest.raises(ValueError, match="failed units"):
            se.run(out, "cpu", 2, [0, 1, 2, 3])
        assert state["launched"] == [0, 1]
    else:
        result = se.run(out, "cpu", 2, [0, 1, 2, 3])
        assert result["technical_status"] == "PARTIAL"
        assert state["launched"] == [0, 1, 2] and state["verified"] == [3]
    assert state["max"] == 2 and state["live"] == 0
    assert not (out / "RUNNING.json").exists()
    assert se.dc.read(out / "records/process_000_a01.json")["MaxRSS"] is None
