"""Contract, scheduling, integrity and reductions; no long training or lab weights."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis import cam96k_training_pilot as p
from analysis import evaluate_mappo_zero_shot as zs
from aoi_v2x_reproduction.runtime import runner, checkpointing as cp, training_selection as ts
from tests.test_training100 import FakeTrainer, config, install_fake, assert_same_rng
from tests.test_frozen_deployment import synthetic_arrays, META, dump, FakeEnvironment, FakeFrozenPolicy


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "code_check", lambda expected=None, smoke=False: p.BASELINE)
    m = p.init(tmp_path / "study", "formal_test", p.BASELINE)
    return Path(m["batch_root"])


def test_six_exact_configs_and_30_units_preserve_source_fields(batch):
    plan = p.preview(batch)
    assert plan["training_environment_steps"] == 1500000
    assert plan["pairs"] == [[0, 1], [2, 3], [4, 5]] and plan["eval_units"] == 30
    assert [(c["model_id"], c["training_seed"]) for c in p.cells()] == [(m, s) for m in p.MODELS for s in (12, 13)]
    for source, c in zip(p.sources(), plan["configs"]):
        raw = c["effective_config"]
        expected, _ = p.tc.derive(source, Path(raw["output_root"]))
        before = expected.to_dict()
        assert set(p.config_differences(before, raw)) == {"cam_bits", "run_name"}
        assert raw["cam_bits"] == 96000 and raw["episodes"] == 2500 and raw["steps_per_episode"] == 100
    assert len({(u["model_id"], u["training_seed"], u["world"], u["repeat"]) for u in p.units()}) == 30
    assert {u["world"] for u in p.units()} == set(range(1001, 1006))
    assert set(p.config_differences(p.sources()[0]["config"], plan["configs"][0]["effective_config"])) <= {"episodes", "cam_bits", "run_name", "output_root"}


@pytest.mark.parametrize("key,value", [("training_episodes", 2499), ("scored", 99), ("warmup", 4),
                                       ("cam_bits", 32000), ("max_training_processes", 6),
                                       ("smoke", True), ("rule", {}), ("new_unseen_confirmation", True)])
def test_reject_manifest_drift(batch, key, value):
    path = batch / "protocol/manifest.json"
    m = p.read(path)
    m[key] = value
    dump(path, m)
    with pytest.raises(ValueError):
        p.load(batch)


def test_no_overwrite_smoke_separation_and_unsafe_ids(batch, tmp_path):
    with pytest.raises(ValueError, match="unique"):
        p.init(batch.parent.parent, batch.name, p.BASELINE)
    for name, smoke in (("smoke_wrong", False), ("formal_wrong", True), ("../escape", False)):
        with pytest.raises(ValueError):
            p.init(tmp_path, name, p.BASELINE, smoke)
    for i in (-1, 6, True):
        with pytest.raises(ValueError):
            p.cell(i)
    with pytest.raises(ValueError):
        p.run_path(batch, 0, "../../old")


def test_exclusive_locks_slots_and_stale_lock_preservation(tmp_path):
    path = tmp_path / "unit.lock"
    with p.lock(path):
        with pytest.raises(FileExistsError):
            with p.lock(path):
                pytest.fail("duplicate lock acquired")
        assert path.exists()
    assert not path.exists()
    with p.training_slot(tmp_path, tmp_path):
        with p.training_slot(tmp_path, tmp_path):
            with pytest.raises(ValueError, match="two CAM96k"):
                with p.training_slot(tmp_path, tmp_path):
                    pytest.fail("third training process admitted")
    assert not list(tmp_path.glob("*.lock"))


def test_pair_barrier_prevents_later_method_and_full_gate_prevents_eval(batch):
    with pytest.raises(FileNotFoundError):
        p.train_unit(batch, 2)
    with pytest.raises(FileNotFoundError):
        p.eval_unit(batch, 0)
    assert not list((batch / "t").glob("a*"))
    assert not list((batch / "e").iterdir())


def test_queue_same_method_pairs_skips_reverified_and_stops_on_failure(batch, monkeypatch):
    events = []
    accepted = {0}
    dump(batch / "records/train_0.json", {})
    monkeypatch.setattr(p, "preflight", lambda *a: {})
    monkeypatch.setattr(p, "accepted_run", lambda b, i: events.append(("verify", i)) or {"run_id": i})
    def launch(b, command, ids, attempt, reason):
        events.append(("launch", tuple(ids)))
        accepted.update(ids)
    monkeypatch.setattr(p, "launch_group", launch)
    monkeypatch.setattr(p, "verify_training", lambda *a: {"accepted": sorted(accepted)})
    assert p.train_all(batch)["accepted"] == list(range(6))
    assert [v for k, v in events if k == "launch"] == [(1,), (2, 3), (4, 5)]
    assert events.index(("verify", 1)) < events.index(("launch", (2, 3)))
    def failed(*args):
        raise ValueError("technical failure")
    monkeypatch.setattr(p, "launch_group", failed)
    with pytest.raises(ValueError, match="technical failure"):
        p.train_all(batch)
    assert not (batch.parent.parent / ".cam96k-training-supervisor.lock").exists()


def test_payload_reduction_linear_power_counts_na_and_zero_success():
    arrays = synthetic_arrays()
    arrays["remaining_demand"] = np.where(arrays["success"], 0, 48000).astype(np.float32)
    meta = {**META, "cam_bits": 96000}
    # Reconstructed task1 ledger must reflect the new payload normalization.
    arrays["reward_task1"] = -4.95 * arrays["remaining_demand"] / 96000
    rows = p.reduce_payload(arrays, meta, p.units()[0], 2)
    metrics = p.aggregate(rows)
    assert metrics["cam_count"] == 10
    assert metrics["power_mw"] == pytest.approx(np.power(10., arrays["power_dbm"].astype(np.float64) / 10).mean())
    assert metrics["deadline_remaining_bits"] == pytest.approx(arrays["remaining_demand"][:, :, -1].mean())
    arrays["remaining_demand"][:] = 0
    arrays["success"][:] = 1
    arrays["reward_task1"][:] = 0
    assert p.aggregate(p.reduce_payload(arrays, meta, p.units()[0], 2))["failed_remaining_bits"] == "NA"
    arrays["success"][:] = 0
    with pytest.raises(ValueError, match="zero remaining"):
        p.reduce_payload(arrays, meta, p.units()[0], 2)


def synthetic_formal_run(batch, monkeypatch):
    """Full budget metadata and selection with constant synthetic rewards; no env steps."""
    stamp = {"reproduction_git_commit": p.BASELINE, "reproduction_git_dirty": False,
             "reproduction_git_branch": "research/cam96k-training-pilot"}
    monkeypatch.setattr(cp, "_git_stamp", lambda: stamp)
    monkeypatch.setattr(runner, "_git_metadata", lambda: stamp)
    run = p.run_path(batch, 0, "a01")
    cfg, _ = p.derive(p.sources()[0], run.parent)
    runner._prepare_run(cfg, None)
    trainer = FakeTrainer(cfg)
    selector = ts.TrainingRewardBest(run, cfg)
    zeros = np.zeros((100, 5), np.float32)
    for e in range(1, 2501):
        trainer.policy_version = (e - 1) // 5
        selector.observe(zeros, zeros, np.zeros(100, np.float32), trainer, trainer.policy_version,
                         trainer.policy_version, "after_episode_before_pending_ppo_update")
    selection = selector.finish(trainer, 500)
    np.savez_compressed(run / "train_metrics.npz", task1_step=np.zeros((2500, 100, 5), np.float32),
                        task2_step=np.zeros((2500, 100, 5), np.float32), global_step=np.zeros((2500, 100), np.float32))
    complete = {**stamp, "status": "complete", "algorithm": "mappo", "config_hash": cfg.canonical_hash(),
                "seed": 12, "episodes": 2500, "environment_steps": 250000, "update_count": 500,
                "actor_network_count": 1, "actor_optimizer_step_count": 5000,
                "mappo_critic_structure": "joint_observation_scalar_global_plus_independent_local_task1_task2_values",
                "training_reward_selection": selection}
    dump(run / "COMPLETE.json", complete)
    return run


def test_full_formal_verifier_and_best100_strict_loader_integrity(batch, monkeypatch):
    run = synthetic_formal_run(batch, monkeypatch)
    checked = p.verify_run(batch, 0, run)
    assert checked["best_episode"] == 100 and checked["best_score"] == 0
    assert checked["best_policy_version"] == 19 and checked["final_policy_version"] == 500
    dump(batch / "records/train_0.json", checked)
    assert p.accepted_run(batch, 0) == checked
    with (run / "train_metrics.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="identity changed"):
        p.accepted_run(batch, 0)


def test_progress_callback_preserves_training_rng_and_trajectory(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    results, states, progress = [], [], []
    for i, callback in enumerate((None, progress.append)):
        cfg = config(tmp_path / str(i), mappo_actor_sharing=True)
        results.append(runner.train(cfg, training_reward_best=True, progress_callback=callback))
        states.append(cp.capture_rng_state())
    assert_same_rng(*states)
    assert len(progress) == 110 and progress[-1]["update_count"] == 22 and progress[-1]["episode"] == 110
    with np.load(Path(results[0]["run_dir"]) / "train_metrics.npz") as a, np.load(Path(results[1]["run_dir"]) / "train_metrics.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_formal_clean_pin_and_algorithm_changes_guard(monkeypatch):
    responses = {("rev-parse", "HEAD"): p.BASELINE, ("branch", "--show-current"): "research/cam96k-training-pilot",
                 ("status", "--porcelain", "--untracked-files=all"): "",
                 ("diff", "--name-only", p.BASELINE, "--", "aoi_v2x_reproduction"): ""}
    monkeypatch.setattr(p.dc, "git", lambda *args: responses[args])
    assert p.code_check(p.BASELINE) == p.BASELINE
    with pytest.raises(ValueError, match="pinned"):
        p.code_check("bad")
    responses[("status", "--porcelain", "--untracked-files=all")] = "M file"
    with pytest.raises(ValueError, match="clean"):
        p.code_check()
    assert p.code_check(smoke=True) == p.BASELINE
    responses[("diff", "--name-only", p.BASELINE, "--", "aoi_v2x_reproduction")] = "aoi_v2x_reproduction/envs/platoon.py"
    with pytest.raises(ValueError, match="algorithm/environment"):
        p.code_check(smoke=True)


def test_compact_smoke_accepts_exploratory_world_role_before_file_load(tmp_path):
    # A valid explicit smoke role reaches file validation; the old guard rejected it.
    with pytest.raises(FileNotFoundError):
        zs.evaluate_zero_shot(tmp_path / "missing.pt", tmp_path / "out", p.dc.SCENARIO,
                              deployment=True, compact_output=True, world_role=p.WORLD_ROLE)
    with pytest.raises(ValueError, match="invalid deployment world role"):
        zs.evaluate_zero_shot(tmp_path / "missing.pt", tmp_path / "out", p.dc.SCENARIO,
                              deployment=True, world_role="invalid")


def test_summary_30_units_two_seeds_worst_platoon_not_binary_min_and_idempotence(batch, monkeypatch):
    m = p.load(batch)
    monkeypatch.setattr(p, "verify_training", lambda *a: {})
    all_records = {}
    for unit in m["eval_units"]:
        arrays = synthetic_arrays(episodes=100)
        arrays["success"][:] = 0
        for e in range(100):
            arrays["success"][0, e, -1, :] = 1
            arrays["success"][0, e, -1, e % 5] = 0
            if unit["training_seed"] == 13:
                arrays["success"][0, e, -1, (e + 1) % 5] = 0
        arrays["remaining_demand"] = np.where(arrays["success"], 0, 48000).astype(np.float32)
        arrays["reward_task1"] = -4.95 * arrays["remaining_demand"] / 96000
        meta = {**META, "cam_bits": 96000, "eval_seeds": [unit["world"]]}
        record = {"attempt": "a01", "rows": p.reduce_payload(arrays, meta, unit, 100)}
        all_records[unit["unit_id"]] = record
        dump(batch / f"records/eval_{unit['unit_id']}.json", record)
    monkeypatch.setattr(p, "verify_eval", lambda b, i, attempt: all_records[i])
    result = p.summary(batch)
    assert result["accepted_eval_units"] == 30 and result["platoon_rows"] == 150
    assert result["cam_deadlines"] == 15000 and result["scored_agent_slots"] == 1500000
    assert result["n_training_seeds_per_method"] == 2
    world_rows = p.csv_read(batch / "tables/world_summary.csv")
    assert {float(r["worst_platoon_cam_rate"]) for r in world_rows} == {.6, .8}
    across = p.csv_read(batch / "tables/across_seeds.csv")
    assert all(float(r["cam_rate"]) == pytest.approx(.7) and r["n_training_seeds"] == "2" for r in across)
    assert p.summary(batch) == result
    with (batch / "tables/across_seeds.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(ValueError, match="existing summary changed"):
        p.summary(batch)


def test_96k_best100_frozen_path_warmup_rng_and_source_unchanged(batch, monkeypatch, tmp_path):
    run = synthetic_formal_run(batch, monkeypatch)
    policy = run / ts.BEST
    before = p.identity(policy)
    calls = []
    class Environment(FakeEnvironment):
        def reset_world(self, world):
            calls.append(("reset", world))
            return super().reset_world(world)
        def start_episode(self, index):
            calls.append(("episode", index))
            return super().start_episode(index)
        def step(self, actions):
            obs, g, t1, t2, done, info = super().step(actions)
            info["remaining_demand"][:] = 96000
            return obs, g, t1, t2, done, info
    class Frozen(FakeFrozenPolicy):
        def __init__(self, cfg, states, device):
            super().__init__(cfg, states, device)
            self.actors = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
    monkeypatch.setattr(zs, "PaperEnviron", Environment)
    monkeypatch.setattr(zs, "FrozenActorPolicy", Frozen)
    monkeypatch.setattr(zs, "_git_metadata", lambda: {"reproduction_git_commit": p.BASELINE, "reproduction_git_dirty": False})
    initial_rng = cp.capture_rng_state()
    result = zs.evaluate_zero_shot(policy, tmp_path / "frozen", p.dc.SCENARIO, "cpu", [1001], 2,
                                  deployment=True, training_policy=True, world_role=p.WORLD_ROLE)
    assert_same_rng(initial_rng, cp.capture_rng_state())
    assert calls.count(("reset", 1001)) == 1
    assert [v for k, v in calls if k == "episode"] == list(range(7))
    assert result["checkpoint_role"] == "best_training100" and result["cam_bits"] == 96000
    assert result["policy_episode"] == 100 and result["policy_version"] == 19
    assert result["policy_parameters_unchanged"] and not result["optimizer_constructed"] and not result["critic_constructed"]
    assert p.identity(policy) == before
    arrays = p.load_arrays(Path(result["eval_dir"]) / "metrics.npz")
    assert arrays["aoi_ms"].shape == (1, 2, 100, 5)
    assert len(p.reduce_payload(arrays, result, p.units()[0], 2)) == 5


def test_uniform_reuse_checks_env_arrays_and_preserves_old_inputs(batch, tmp_path):
    root = tmp_path / "uniform"
    cfg = p.derive(p.sources()[0], batch / "tmp")[0].to_dict()
    manifest = {"schema": "uniform_random_cam_payload_v1", "worlds": list(p.WORLDS),
                "action_stream_ids": [12, 13], "payloads": [64000, 96000],
                "action_stream_ids_are_training_seeds": False, "payload_in_action_seed": False,
                "seed_sequence": "[action_stream_id, world, 0x52414E44, 1]", "warmup_episodes": 5,
                "scored_episodes": 100, "slots": 100, "world_role": p.WORLD_ROLE,
                "new_unseen_confirmation": False, "implementation_commit": "historical_commit",
                "worktree_status": "historical_dirty_state"}
    dump(root / "protocol/manifest.json", manifest)
    dump(root / "verification.json", {**manifest, "technical_status": "PASS"})
    arrays = synthetic_arrays(episodes=100)
    arrays["remaining_demand"] = np.where(arrays["success"], 0, 48000).astype(np.float32)
    arrays["reward_task1"] = -4.95 * arrays["remaining_demand"] / 96000
    for world in p.WORLDS:
        for stream in p.SEEDS:
            folder = root / f"runs/cam96000/world{world}/stream{stream}"
            common = {"technical_status": "PASS", "cam_bits": 96000, "world": world, "action_stream_id": stream,
                      "action_seed_words": [stream, world, 0x52414E44, 1], "payload_in_action_seed": False,
                      "initial_demand_episode_starts": 105, "warmup_episodes_excluded": 5,
                      "scored_episodes": 100, "cam_deadlines": 500}
            dump(folder / "unit_complete.json", common)
            dump(folder / "rollout.json", {**META, **common, "eval_seeds": [world], "eval_warmup_episodes": 5,
                                          "eval_episodes": 100, "eval_protocol": "sequential_warm",
                                          "runtime_config": cfg, "world_role": p.WORLD_ROLE})
            np.savez_compressed(folder / "metrics.npz", **arrays)
    original = p.identity(root / "protocol/manifest.json")
    result = p.reuse_uniform(batch, root)
    assert result["new_rollouts"] == 0 and result["n_training_models"] == 0
    assert result["metrics"]["cam_count"] == 5000 and result["platoon_rows"] == 50
    assert p.reuse_uniform(batch, root) == result
    assert p.identity(root / "protocol/manifest.json") == original
    path = root / "runs/cam96000/world1001/stream12/rollout.json"
    changed = p.read(path)
    changed["runtime_config"]["bandwidth_hz"] += 1
    dump(path, changed)
    with pytest.raises(ValueError, match="effective environment"):
        p.reuse_uniform(batch, root)


def test_status_distinguishes_pending_failed_attempts_and_accepted(batch):
    run = p.run_path(batch, 0, "a01")
    run.mkdir(parents=True)
    dump(batch / "records/failed_0_a01.json", {"error": "technical failure"})
    dump(batch / "e/0/a01/FAILED.json", {"error": "evaluation failure"})
    result = p.status(batch)
    assert result["training"][0]["state"] == "FAILED"
    assert result["training"][0]["attempts"][0]["state"] == "FAILED"
    assert result["training"][1]["state"] == "PENDING"
    assert result["evaluation"][0]["state"] == "FAILED"
    dump(batch / "e/0/a02/native/EVAL_COMPLETE.json", {})
    dump(batch / "records/eval_0.json", {})
    result = p.status(batch)
    assert result["evaluation"][0]["state"] == "ACCEPTED"
    assert [a["state"] for a in result["evaluation"][0]["attempts"]] == ["FAILED", "COMPLETE"]
