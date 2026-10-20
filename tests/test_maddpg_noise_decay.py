"""Synthetic trajectories / real small networks; no formal 2500x100 training."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aoi_v2x_reproduction.config import config_from_dict, resolve_config, validate_config
from aoi_v2x_reproduction.runtime import maddpg_noise as n, runner as r, shared_resume as sr
from aoi_v2x_reproduction.runtime.training_selection import load_training_policy, load_training_actors, TrainingRewardBest
from tests.test_shared_maddpg_resume import StatefulSyntheticEnv, equal_tree, read
from tests.test_training100 import assert_same_rng
from analysis import maddpg_noise_decay_contract as c, deployment_contract as dc


def cfg(root, name="run", shared=False, linear=True):
    return resolve_config(algorithm="modified_maddpg_tdec", scenario="p05_n04_g25", seed=10, episodes=104,
        steps_per_episode=2, batch_size=4, replay_capacity=16, actor_hidden=[8, 4], local_critic_hidden=[8, 4],
        global_critic_hidden=[8, 4, 2], checkpoint_mode="policy_only", maddpg_actor_sharing=shared,
        device="cpu", output_root=str(root), run_name=name, diagnostics=True,
        exploration_noise_schedule=n.SCHEDULE["name"] if linear else "fixed", exploration_noise_decay_environment_steps=13)


@pytest.fixture
def synthetic(monkeypatch):
    from aoi_v2x_reproduction.algorithms.modified_maddpg.agent import make_agents
    from aoi_v2x_reproduction.algorithms.modified_maddpg.learner import Global_Critic
    from aoi_v2x_reproduction.algorithms.modified_maddpg.replay import ReplayBuffer
    from aoi_v2x_reproduction.runtime.metrics import MetricStore
    def factory(config):
        r.seed_everything(config.seed, torch.device("cpu"))
        agents = make_agents(config)
        assert len({id(a.actor) for a in agents}) == (1 if config.maddpg_actor_sharing else 5)
        assert len({id(a.actor.optimizer) for a in agents}) == (1 if config.maddpg_actor_sharing else 5)
        return (StatefulSyntheticEnv(config), agents, Global_Critic(config, agents),
                ReplayBuffer(config.replay_capacity, config.state_dim, config.action_dim, 5),
                MetricStore(5, config.steps_per_episode, diagnostics=config.diagnostics), torch.device("cpu"))
    monkeypatch.setattr(r, "_make_system", factory)
    monkeypatch.setattr(r, "_selection_validation", lambda *a: pytest.fail("no validation rollout"))
    return factory


def test_schedule_points_fixed_serialization_and_validation(tmp_path):
    config = cfg(tmp_path)
    config.exploration_noise_decay_environment_steps = 150000
    for k, value in ((0, .3), (50000, .21666666666666667), (100000, .13333333333333336),
                     (150000, .05), (249999, .05), (250000, .05)):
        assert n.sigma(config, k) == pytest.approx(value, abs=1e-15)
    old = cfg(tmp_path, linear=False)
    assert not any(k.startswith("exploration_noise_") for k in old.to_dict())
    assert config_from_dict(old.to_dict()).canonical_hash() == old.canonical_hash()
    assert [n.sigma(old, k) for k in (0, 150000, 250000)] == [.3] * 3
    for field, value in (("exploration_noise_end", -.1), ("exploration_noise_floor", .06),
                         ("exploration_noise_decay_environment_steps", 0), ("exploration_noise_schedule", "episode")):
        bad = copy.deepcopy(config); setattr(bad, field, value)
        with pytest.raises(ValueError): validate_config(bad)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("cut", [5, 100])
def test_resume_bitwise_schedule_rng_optimizer_replay_metrics_best_and_snapshots(tmp_path, synthetic, shared, cut):
    full, partial, resumed = [cfg(tmp_path, name, shared) for name in ("full", "partial", "resumed")]
    r.train(full, training_reward_best=True, noise_decay_training=True)
    r.train(partial, training_reward_best=True, noise_decay_training=True, max_episodes=cut)
    source = tmp_path / f"partial/checkpoints/episode_{cut:06d}.pt"
    original = source.read_bytes()
    r.train(resumed, training_reward_best=True, noise_decay_training=True, resume=str(source))
    assert source.read_bytes() == original
    a, b = [read(tmp_path / name / "checkpoints/episode_000104.pt") for name in ("full", "resumed")]
    for key in ("agents", "learner", "replay", "environment", "metrics", "completed_environment_steps"):
        equal_tree(a[key], b[key])
    if shared: equal_tree(a["shared_actor"], b["shared_actor"])
    assert_same_rng(a["rng"], b["rng"])
    assert a["completed_environment_steps"] == 208
    for name in ("policy_best_training.pt", "policy_final.pt"):
        _, pa = load_training_policy(tmp_path / "full" / name)
        _, pb = load_training_policy(tmp_path / "resumed" / name)
        equal_tree(pa["actors"], pb["actors"])
        equal_tree(pa["training_reward_selection"], pb["training_reward_selection"])
    for name in ("training_reward_selection.csv", "exploration_noise.csv"):
        assert (tmp_path / "full" / name).read_bytes() == (tmp_path / "resumed" / name).read_bytes()
    for name in ("full", "resumed"):
        index = json.loads((tmp_path / name / "policy_snapshot_index.json").read_text())
        assert len(index) == 1 and index[0]["episode"] == 100
        _, actors = load_training_actors(index[0]["path"])
        assert len(actors) == (1 if shared else 5)
        assert not hasattr(actors[0], "optimizer") or actors[0].optimizer is None
        _, policy = load_training_policy(index[0]["path"])
        assert policy["episode"] == 100 and policy["policy_version"] == 98 and policy["resume_supported"] is False
    c.validate_noise_rows(c.csv_read(tmp_path / "full/exploration_noise.csv"), full)
    assert a["metrics"]["episodes"][5]["exploration"]["sigma_first"] == n.sigma(full, 10)
    assert a["metrics"]["episodes"][6]["exploration"]["sigma_last"] == .05


@pytest.mark.parametrize("shared", [False, True])
def test_diagnostics_same_forward_noise_rng_and_eval_noise_independent(tmp_path, synthetic, shared):
    config = cfg(tmp_path, shared=shared)
    _, agents, learner, *_ = synthetic(config)
    obs = np.zeros(config.state_dim, np.float32)
    np.random.seed(42)
    a = agents[0].choose_action(obs, noise_std=.3)
    after_a = r.capture_rng_state()
    np.random.seed(42)
    diagnostic = {}
    b = agents[0].choose_action(obs, noise_std=.3, diagnostics=diagnostic)
    after_b = r.capture_rng_state()
    np.testing.assert_array_equal(a, b); assert_same_rng(after_a, after_b)
    assert diagnostic["action_count"] == 1 and diagnostic["component_count"] == 3
    assert learner.config.target_noise_sigma == .2 and learner.config.target_noise_clip == .5
    np.random.seed(25)
    initial = r.capture_rng_state()
    zero = agents[0].choose_action(obs, explore=False, noise_std=0)
    assert_same_rng(initial, r.capture_rng_state())
    fixed = copy.deepcopy(config); fixed.exploration_noise_schedule = "fixed"
    agents[0].config = fixed
    np.testing.assert_array_equal(zero, agents[0].choose_action(obs, explore=False, noise_std=0))


def test_best_peak_not_quantized_and_checkpointing_does_not_change_fixed_behavior(tmp_path, synthetic):
    config = cfg(tmp_path)
    (tmp_path / "selector").mkdir()
    (tmp_path / "selector/provenance.json").write_text("{}")
    _, agents, *_ = synthetic(config)
    selector = TrainingRewardBest(tmp_path / "selector", config, resume_supported=True)
    config.episodes = 104
    for e in range(1, 105):
        reward = 0 if e <= 3 or e == 104 else 1
        with torch.no_grad():
            for a in agents:
                next(a.actor.parameters()).fill_(e / 1000)
        selector.observe(np.full((2, 5), reward), np.zeros((2, 5)), np.zeros(2), agents, e - 1, e,
                         "after_episode_online_actor_updates")
    selector.finish(agents, 104)
    best = read(tmp_path / "selector/policy_best_training.pt")
    assert best["episode"] == 103 and best["training_reward_selection"]["window_start"] == 4
    assert next(iter(best["actors"][0].values())).flatten()[0] == pytest.approx(.103)
    # Fixed-noise old path and old shared checkpoint path still produce identical outcomes.
    old = cfg(tmp_path, "old", shared=True, linear=False)
    legacy = cfg(tmp_path, "legacy", shared=True, linear=False)
    r.train(old, training_reward_best=True)
    rng = r.capture_rng_state()
    r.train(legacy, training_reward_best=True, shared_resume_interval=50)
    assert_same_rng(rng, r.capture_rng_state())
    equal_tree(read(tmp_path / "old/policy_final.pt")["actors"], read(tmp_path / "legacy/policy_final.pt")["actors"])
    with pytest.raises(ValueError, match="every 50"):
        sr.validate_request(legacy, True, 100)


def test_resume_rejects_foreign_schedule_old_weights_runtime_and_completed(tmp_path, synthetic, monkeypatch):
    config = cfg(tmp_path)
    r.train(config, training_reward_best=True, noise_decay_training=True)
    target = cfg(tmp_path, "target")
    with pytest.raises(ValueError, match="matching noise-decay"):
        n.read_resume(tmp_path / "run/policy_best_training.pt", target)
    with pytest.raises(ValueError, match="final"):
        n.read_resume(tmp_path / "run/checkpoints/episode_000104.pt", target)
    target.exploration_noise_decay_environment_steps += 1
    with pytest.raises(ValueError, match="configuration"):
        n.read_resume(tmp_path / "run/checkpoints/episode_000100.pt", target)
    target.exploration_noise_decay_environment_steps -= 1
    monkeypatch.setattr(n, "_git_stamp", lambda: {"reproduction_git_commit": "different"})
    with pytest.raises(ValueError, match="commit changed"):
        n.read_resume(tmp_path / "run/checkpoints/episode_000100.pt", target)


@pytest.mark.parametrize("shared", [False, True])
def test_logging_and_periodic_save_do_not_perturb_constant_noise_training(tmp_path, synthetic, shared):
    old = cfg(tmp_path, "old", shared=shared, linear=False)
    new = cfg(tmp_path, "instrumented", shared=shared)
    new.exploration_noise_end = new.exploration_noise_floor = .3
    r.train(old, training_reward_best=True)
    state = r.capture_rng_state()
    r.train(new, training_reward_best=True, noise_decay_training=True)
    assert_same_rng(state, r.capture_rng_state())
    equal_tree(read(tmp_path / "old/policy_final.pt")["actors"], read(tmp_path / "instrumented/policy_final.pt")["actors"])
    with np.load(tmp_path / "old/train_metrics.npz") as a, np.load(tmp_path / "instrumented/train_metrics.npz") as b:
        assert a.files == b.files
        for key in a.files: np.testing.assert_array_equal(a[key], b[key])


def test_matrix_source_derivation_preserves_profile_network_reward_and_old_none_sharing(tmp_path):
    for i in range(4):
        item = c.cell(i)
        source = cfg(tmp_path, shared=i >= 2, linear=False)
        source.episodes, source.steps_per_episode, source.seed = 2500, 100, item["training_seed"]
        source.profile, source.semantic_version = "paper_faithful", "paper_faithful_v4"
        raw = source.to_dict()
        if i < 2: raw["maddpg_actor_sharing"] = None  # Historical absent/None means false.
        # config_from_dict currently normalizes only absent; the contract must explicitly normalize None.
        s = {**item, "config": raw}
        derived, delta = c.derive(s, tmp_path / "new")
        assert set(delta) == c.DIFF_FIELDS
        assert derived.profile == source.profile and derived.semantic_version == source.semantic_version
        assert derived.actor_hidden == source.actor_hidden and derived.target_noise_sigma == .2
        assert derived.maddpg_actor_sharing == (i >= 2)
        assert derived.seed in (10, 12)
    with pytest.raises(ValueError): c.cell(4)


def source_batches(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "code_check", lambda expected=None: "synthetic")
    study = tmp_path / "study"
    parents = [study / "Training100-best/P5_N4_gap25/original", study / "Shared-MADDPG-TDec/P5_N4_gap25/original"]
    for batch, shared in zip(parents, (False, True)):
        nrun = 6 if shared else 24
        dc.write(batch / "protocol/manifest.json", {"implementation_commit": "old", "rule": tc_rule()})
        dc.write(batch / "analysis/a01/verification.json", {"technical_status": "PASS", "accepted_runs": nrun,
            "episode_rows": nrun * 2500, "training_env_steps": nrun * 250000, "implementation_commit": "old"})
        for seed in (10, 12):
            config = cfg(batch, shared=shared, linear=False)
            config.episodes, config.steps_per_episode, config.seed = 2500, 100, seed
            config.profile, config.semantic_version = "paper_faithful", "paper_faithful_v4"
            run = batch / "training/a02/runs" / f"old_seed{seed}"
            record = {"model_id": "shared_maddpg_tdec" if shared else "algorithm2", "training_seed": seed,
                "technical_status": "PASS", "episodes": 2500, "environment_steps": 250000,
                "implementation_commit": "old", "run_dir": str(run)}
            dc.write(batch / f"records/train_{seed:02d}.json", record)
            dc.write(run / "config.resolved.json", config.to_dict())
            dc.write(run / "COMPLETE.json", {"status": "complete", "config_hash": config.canonical_hash(),
                "reproduction_git_commit": "old", "training_reward_selection": {"final_episode": 2500}})
    return study, parents


def tc_rule():
    from aoi_v2x_reproduction.runtime.training_selection import RULE
    return RULE


def test_init_readonly_sources_reject_config_change_and_unique_output(tmp_path, monkeypatch):
    study, parents = source_batches(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for root in parents for p in root.rglob("*") if p.is_file()}
    m = c.init(*parents, study, "synthetic", "test")
    batch = Path(m["batch_root"])
    assert c.load(batch)[0] == m
    assert {p: p.read_bytes() for p in before} == before
    assert m["new_training_runs"] == 4 and m["training_env_steps"] == 1000000
    with pytest.raises(ValueError, match="unique"):
        c.init(*parents, study, "synthetic", "test")
    s = c.load(batch)[1][0]
    path = Path(s["parent_files"][1]["path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="source changed"): c.load(batch)


def build_synthetic_finished_run(batch, i):
    """Construct small-actor, full-shape synthetic evidence; never train environment steps."""
    from aoi_v2x_reproduction.algorithms.modified_maddpg.agent import make_agents
    m, sources = c.load(batch)
    run = c.attempt_run(batch, i, "a01")
    config, delta = c.derive(sources[i], run.parent)
    config.device = "cpu"  # Source fixture also uses CPU.
    run.mkdir(parents=True)
    dc.write(run / "config.resolved.json", config.to_dict())
    dc.write(run / "provenance.json", {"reproduction_git_commit": "synthetic"})
    selector = TrainingRewardBest(run, config, resume_supported=True)
    agents = make_agents(config)
    values = np.zeros(2500, np.float32); values[129:229] = 1
    task1 = np.broadcast_to(values[:, None, None], (2500, 100, 5)).copy()
    task2 = np.zeros_like(task1); global_step = np.zeros((2500, 100), np.float32)
    index = []
    for e in range(1, 2501):
        before = max(0, (e - 1) * 100 - config.batch_size + 1) // config.policy_delay
        after = (e * 100 - config.batch_size + 1) // config.policy_delay
        selector.observe(task1[e - 1], task2[e - 1], global_step[e - 1], agents, before, after, "after_episode_online_actor_updates")
        if e % 100 == 0: index.append(n.snapshot(run, config, agents, e, after))
    selection = selector.finish(agents, (250000 - config.batch_size + 1) // config.policy_delay)
    aoi = np.full_like(task1, 100); cam = np.ones_like(task1)
    np.savez_compressed(run / "train_metrics.npz", task1_step=task1, task2_step=task2, global_step=global_step,
        aoi_ms=aoi, success=cam, remaining_demand=np.zeros_like(task1), power_dbm=np.full_like(task1, 10),
        mode=np.zeros_like(task1), reset_event=np.zeros_like(task1))
    noise_rows = []
    for e in range(2500):
        k = e * 100
        noise_rows.append({"episode": e + 1, "environment_step_start": k, "environment_step_end_exclusive": k + 100,
            "sigma_first": n.sigma(config, k), "sigma_last": n.sigma(config, k + 99),
            "sigma_mean": float(np.mean([n.sigma(config, k + j) for j in range(100)])), "action_count": 500,
            "component_count": 1500, "clipped_component_count": 3, "clip_rate": 3/1500,
            "rb_changed_count": 5, "rb_changed_fraction": .01, "mode_changed_count": 10, "mode_changed_fraction": .02})
    c.csv_write(run / "exploration_noise.csv", noise_rows)
    dc.write(run / "policy_snapshot_index.json", index)
    dc.write(run / "progress.json", {"episodes_completed": 2500})
    learns = 250000 - config.batch_size + 1
    updates = learns // config.policy_delay
    learning = [{"episode": e + 1, "learner_update_count": (100 - config.batch_size + 1 if e == 0 else 100),
                 "actor_update_count": (100 - config.batch_size + 1) // config.policy_delay if e == 0 else 50} for e in range(2500)]
    dc.write(run / "learning_diagnostics.json", learning)
    complete = {"status": "complete", "episodes": 2500, "final_episode": 2500, "environment_steps": 250000,
        "algorithm": "modified_maddpg_tdec", "config_hash": config.canonical_hash(), "reproduction_git_commit": "synthetic",
        "training_reward_selection": selection, "actor_network_count": len(agents) if not config.maddpg_actor_sharing else 1,
        "policy_snapshot_count": 25, "learn_step_count": learns,
        "episode_resume": {"schema": n.SCHEMA, "interval": 100, "resume_episode": 0, "resume_source": None}}
    if config.maddpg_actor_sharing:
        complete.update(actor_sharing=True, model_id="shared_maddpg_tdec", actor_optimizer_step_count=updates,
                        target_actor_update_count=updates)
    dc.write(run / "COMPLETE.json", complete)
    dc.write(Path(batch) / f"records/started_{i:02d}_a01.json", {"resume_source": None})
    return run


def test_full_shape_synthetic_acceptance_summary_export_and_boundary_denominators(tmp_path, monkeypatch):
    from aoi_v2x_reproduction.runtime import checkpointing as ck
    monkeypatch.setattr(ck, "_git_stamp", lambda: {"reproduction_git_commit": "synthetic"})
    study, parents = source_batches(tmp_path, monkeypatch)
    batch = Path(c.init(*parents, study, "synthetic", "complete")["batch_root"])
    for i in range(4):
        run = build_synthetic_finished_run(batch, i)
        record = c.verify_run(batch, i, run)
        assert record["best_episode"] == 229 and record["window_start"] == 130
        dc.write(batch / f"records/train_{i:02d}.json", record)
    v = c.summary(batch)
    assert v["technical_status"] == "PASS" and v["agent_slots"] == 5000000 and v["cam_deadline_events"] == 50000
    assert c.status(batch)["accepted_ids"] == [0, 1, 2, 3]
    result = c.export(batch)
    assert Path(result["archive"]).is_file() and Path(result["business_archive"]).is_file()
    assert c.export(batch) == result and c.summary(batch) == v
    rows = c.csv_read(batch / "analysis/a01/training_window_summary.csv")
    assert len(rows) == 12 and float(rows[0]["combined_reward_mean"]) == 1
    assert all(float(row["mean_power_mw"]) == 10 and float(row["cam_success_rate"]) == 1 for row in rows)
    index = c.csv_read(batch / "analysis/a01/policy_snapshot_index.csv")
    assert len(index) == 100 and len({r["path"] for r in index}) == 100
    # Partial output must not be silently overwritten or packed as success.
    (batch / "analysis/a02").mkdir()
    with pytest.raises(ValueError, match="partial analysis"):
        c.summary(batch, "a02")
    path = Path(json.loads((batch / "records/train_00.json").read_text())["run_dir"]) / "exploration_noise.csv"
    rows = c.csv_read(path); rows[1500]["sigma_first"] = .3
    path.unlink(); c.csv_write(path, rows)
    with pytest.raises(ValueError, match="sigma"):
        c.export(batch)
