"""Synthetic loops only: no PaperEnviron, real training, checkpoint or evaluator."""
import copy
import csv
import json
import random
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime import runner as r
from aoi_v2x_reproduction.runtime import training_selection as ts
from aoi_v2x_reproduction.runtime.metrics import MetricStore
from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state
from analysis import training100_contract as tc, submit_training100 as sj
from tests.test_frozen_deployment import source, dump


def config(root, name="run", model="mappo", **kw):
    return resolve_config(algorithm=model, scenario="p05_n04_g25", episodes=110, steps_per_episode=2,
                          actor_hidden=[8, 4], local_critic_hidden=[8, 4], global_critic_hidden=[8, 4, 2],
                          checkpoint_mode="policy_only", seed=8, batch_size=1, replay_capacity=10,
                          mappo_rollout_episodes=5, device="cpu", output_root=str(root), run_name=name,
                          diagnostics=False, **kw)


class FakeEnv:
    def __init__(self, cfg): self.cfg = cfg; self._world_initialized = False
    def reset_world(self, seed): self.rng = np.random.default_rng(seed); self._world_initialized = True
    def start_episode(self, e): self.e = e; self.step_index = 0; return np.zeros((5, self.cfg.state_dim), np.float32)
    def step(self, actions):
        self.step_index += 1
        t1 = np.full(5, min(self.e, 104) / 100, np.float32)
        t2 = np.asarray(actions[:, 0] * .01, np.float32)
        glob = float(self.rng.random())  # Non-float32 source deliberately tests logged precision.
        info = {"aoi_ms": np.ones(5), "success": np.zeros(5), "remaining_demand": np.full(5, 32000.),
                "power_dbm": np.ones(5), "mode": np.zeros(5), "rb": np.zeros(5)}
        return np.zeros_like(actions[:, :1]).repeat(self.cfg.state_dim, axis=1), glob, t1, t2, self.step_index == self.cfg.steps_per_episode, info


class FakeRollout:
    def __init__(self): self.terminal_count = 0; self.steps = 0
    def append(self, **kw): self.terminal_count += int(kw["done"]); self.steps += 1
    def consume(self, **kw): self.terminal_count = self.steps = 0; return None
    def __len__(self): return self.steps


class FakeTrainer:
    def __init__(self, cfg):
        from aoi_v2x_reproduction.algorithms.mappo.networks import HybridActor
        self.cfg = cfg
        self.actors = [HybridActor(cfg.state_dim, cfg.actor_hidden, cfg.n_rb, cfg.n_modes) for _ in range(1 if cfg.mappo_actor_sharing else 5)]
        self.policy_version = self.update_count = self.environment_steps = self.actor_optimizer_step_count = 0
        self.critic_structure = "synthetic_only"
    def act(self, obs, **kw):
        self.environment_steps += 1
        a = (torch.rand(5, 3) * 2 - 1).numpy()
        return SimpleNamespace(environment_actions=a, rb=np.zeros(5), mode=np.zeros(5), power=np.ones(5) / 2,
                               log_prob=np.zeros(5), values=np.zeros(5), tdec_values=self.tdec_values(obs))
    def tdec_values(self, obs): return SimpleNamespace(global_value=np.zeros(1), task1_values=np.zeros(5), task2_values=np.zeros(5))
    def values(self, obs): return np.zeros(5)
    def combined_rewards(self, g, a, b): return a + b + g
    def update(self, batch):
        self.policy_version += 1; self.update_count += 1
        self.actor_optimizer_step_count += len(self.actors) * 10
        with torch.no_grad():
            for actor in self.actors:
                for p in actor.parameters(): p.add_(random.random() * .001)
        return {"update": self.update_count}
    def policy_state_dicts(self): return [a.state_dict() for a in self.actors]
    def parameter_counts(self): return {}


class FakeAgent:
    def __init__(self, cfg):
        from aoi_v2x_reproduction.algorithms.modified_maddpg.networks import ActorNetwork
        self.actor = ActorNetwork(cfg.state_dim, cfg.actor_hidden, 3, cfg.actor_lr, "cpu")
    def choose_action(self, obs, **kw): return np.clip(np.random.normal(0, .3, 3), -1, 1).astype(np.float32)


class FakeLearner:
    def __init__(self, agents, cfg): self.agents = agents; self.learn_step_counter = 0; self.cfg = cfg
    def learn(self, batch):
        self.learn_step_counter += 1
        if self.learn_step_counter % self.cfg.policy_delay == 0:
            with torch.no_grad():
                for a in self.agents:
                    for p in a.actor.parameters(): p.add_(torch.rand_like(p) * .001)
        return {"learn_step": self.learn_step_counter}


class FakeReplay:
    def __init__(self): self.size = 0
    def store_transition(self, *a): self.size += 1
    def sample_buffer(self, *a): return None


def install_fake(monkeypatch):
    def factory(cfg):
        r.seed_everything(cfg.seed, torch.device("cpu"))
        env = FakeEnv(cfg)
        metrics = MetricStore(5, cfg.steps_per_episode, 1., algorithm=cfg.algorithm, diagnostics=False)
        if cfg.algorithm == "mappo": return env, FakeTrainer(cfg), FakeRollout(), metrics, torch.device("cpu")
        agents = [FakeAgent(cfg) for _ in range(5)]
        return env, agents, FakeLearner(agents, cfg), FakeReplay(), metrics, torch.device("cpu")
    monkeypatch.setattr(r, "_make_mappo_system", factory)
    monkeypatch.setattr(r, "_make_system", factory)
    monkeypatch.setattr(r, "_selection_validation", lambda *a: pytest.fail("training100 must never call validation"))


def assert_same_rng(a, b):
    assert a["python"] == b["python"]
    assert a["numpy"][0] == b["numpy"][0]
    np.testing.assert_array_equal(a["numpy"][1], b["numpy"][1])
    assert a["numpy"][2:] == b["numpy"][2:]
    assert torch.equal(a["torch"], b["torch"])
    if "torch_cuda" in a: assert all(torch.equal(x, y) for x, y in zip(a["torch_cuda"], b["torch_cuda"]))


@pytest.mark.parametrize("algorithm,variant,sharing", [("mappo", "tdec", True), ("mappo", "tdec", False), ("mappo", "combined", True), ("modified_maddpg_tdec", "combined", False)])
def test_opt_in_loop_identical_final_rng_updates_and_trajectories(tmp_path, monkeypatch, algorithm, variant, sharing):
    install_fake(monkeypatch)
    cfgs = [config(tmp_path / str(i), model=algorithm, mappo_variant=variant, mappo_actor_sharing=sharing) for i in range(2)]
    outcomes, states = [], []
    for enabled, cfg in zip((False, True), cfgs):
        outcomes.append(r.train(cfg, training_reward_best=enabled))
        states.append(capture_rng_state())
    assert_same_rng(*states)
    paths = [Path(x["run_dir"]) for x in outcomes]
    with np.load(paths[0] / "train_metrics.npz") as a, np.load(paths[1] / "train_metrics.npz") as b:
        assert set(a.files) == set(b.files)
        for k in a.files: np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    policies = [torch.load(p / "policy_final.pt", map_location="cpu", weights_only=False) for p in paths]
    for aa, bb in zip(policies[0]["actors"], policies[1]["actors"]):
        for k in aa: assert torch.equal(aa[k], bb[k])
    cfg, best = ts.load_training_policy(paths[1] / ts.BEST)
    _, actors = ts.load_training_actors(paths[1] / ts.BEST)
    assert len(actors) == (1 if algorithm == "mappo" and sharing else 5)
    assert all(not p.requires_grad for actor in actors for p in actor.parameters())
    with (paths[1] / "training_reward_selection.csv").open() as f: rows = list(csv.DictReader(f))
    assert all(x["rolling100"] == "" for x in rows[:99])
    rewards = [float(x["episode_reward"]) for x in rows]
    episode, score, _ = tc.best_index(rewards)
    assert best["episode"] == episode and best["training_reward_selection"]["score"] == score
    if algorithm == "mappo":
        assert best["policy_version"] == (episode - 1) // 5
        assert policies[1]["policy_version"] == 22
        assert best["training_reward_selection"]["window_policy_version_min"] <= best["policy_version"]
    # Best and final are separate actors/files, even if selected at final episode.
    assert (paths[1] / ts.BEST).resolve() != (paths[1] / "policy_final.pt").resolve()
    assert best["checkpoint_role"] == "best_training100" and policies[1]["checkpoint_role"] == "final_training100"
    assert not (paths[0] / ts.BEST).exists()


def make_selector(tmp_path, seed=8):
    cfg = config(tmp_path, mappo_actor_sharing=True)
    cfg.seed = seed
    run = tmp_path / "run"
    run.mkdir(parents=True)
    dump(run / "provenance.json", {})
    return ts.TrainingRewardBest(run, cfg), FakeTrainer(cfg)


def test_window100_exact_ties_and_seed_independence(tmp_path):
    a, pa = make_selector(tmp_path / "a", 8)
    b, pb = make_selector(tmp_path / "b", 9)
    for e in range(1, 111):
        for selector, policy, value in ((a, pa, 1.), (b, pb, float(e))):
            selector.observe(np.full((2, 5), value), np.zeros((2, 5)), np.zeros(2), policy, 0, 0, "synthetic")
        if e == 99: assert a.best is None and b.best is None
    assert a.best["selected_episode"] == 100 and a.best["score"] == 1
    assert b.best["selected_episode"] == 110 and b.best["window_start"] == 11
    # A rolling mean, not the largest episode in the window.
    rewards = [1000.] + [0.] * 99 + [11.] * 100
    e, score, _ = tc.best_index(rewards)
    assert e == 200 and score == 11


@pytest.mark.parametrize("kwargs", [{"resume": "old.pt"}, {"max_episodes": 100}, {"recover": True}])
def test_recovery_and_partial_rejected_before_writes(tmp_path, kwargs):
    with pytest.raises(ValueError, match="resume/partial/recovery"): ts.validate_request(config(tmp_path), **kwargs)
    assert not list(tmp_path.iterdir())


def test_nonfinite_shapes_and_averaging():
    assert ts.episode_reward(np.full((100, 5), 1.), np.full((100, 5), 2.), np.full(100, 3.)) == 6
    with pytest.raises(ValueError): ts.episode_reward(np.zeros((2, 5)), np.zeros((2, 5)), np.zeros(5))
    with pytest.raises(ValueError): ts.episode_reward(np.full((2, 5), np.nan), np.zeros((2, 5)), np.zeros(2))


@pytest.mark.parametrize("model", tc.MODELS)
def test_source_derivation_only_budget_names_and_none_sharing(tmp_path, model):
    src, _ = source(tmp_path, model)
    if model == "algorithm2":
        src["config"].update(profile="paper_faithful", semantic_version="paper_faithful_v4")
    before = copy.deepcopy(src)
    cfg, delta = tc.derive(src, tmp_path / "new")
    assert src == before and cfg.episodes == 2500 and cfg.steps_per_episode == 100
    assert set(delta) <= {"episodes", "output_root", "run_name"} and cfg.profile == src["config"]["profile"]
    assert cfg.exploration_noise == .3


@pytest.mark.parametrize("sharing", [None, "missing"])
def test_legacy_independent_sharing_is_effectively_false(tmp_path, sharing):
    src, _ = source(tmp_path, "independent_tdec")
    if sharing == "missing":
        src["config"].pop("mappo_actor_sharing")
    else:
        src["config"]["mappo_actor_sharing"] = None
    cfg, delta = tc.derive(src, tmp_path / "new")
    assert cfg.mappo_actor_sharing is False
    assert set(delta) <= {"episodes", "output_root", "run_name", "mappo_actor_sharing"}


def test_matrix_and_submission_arrays_no_test_or_noise_duplicate(tmp_path):
    cells = [tc.cell(i) for i in range(24)]
    assert len({(c["model_id"], c["training_seed"]) for c in cells}) == 24
    assert {tc.cell(i)["training_seed"] for i in tc.PILOTS} == {8}
    assert sum(c["environment_steps"] for c in cells) == 6000000
    assert sj.ARRAYS == {"pilot": "0,6,12,18", "main": "1-5,7-11,13-17,19-23"}
    cmd = sj.command(tmp_path, "main", dependency="afterok:123")
    assert "--array=1-5,7-11,13-17,19-23%6" in cmd and "afterok:123" in cmd
    assert "--gres" not in sj.command(tmp_path, "audit")
    with pytest.raises(ValueError): sj.command(tmp_path, "main", concurrency=7)


def test_completed_training100_resume_rejected_without_state_loading(tmp_path, monkeypatch):
    cfg = config(tmp_path, model="modified_maddpg_tdec")
    run = tmp_path / "old"
    dump(run / "provenance.json", {"training_reward_selection": ts.RULE})
    cp = run / "checkpoints/latest.pt"
    cp.parent.mkdir(); cp.write_bytes(b"not loaded")
    monkeypatch.setattr(r, "_make_system", lambda *a: pytest.fail("must reject before initializing systems"))
    cfg.checkpoint_mode = "resumable"
    with pytest.raises(ValueError, match="do not support resume"): r.train(cfg, resume=str(cp))


def test_loader_rejects_missing_completion_and_wrong_role(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    cfg = config(tmp_path, mappo_actor_sharing=True)
    run = Path(r.train(cfg, training_reward_best=True)["run_dir"])
    payload = torch.load(run / ts.BEST, map_location="cpu", weights_only=False)
    payload["checkpoint_role"] = "final_training100"
    torch.save(payload, run / ts.BEST)
    with pytest.raises(ValueError, match="role/episode"): ts.load_training_policy(run / ts.BEST)
    (run / "COMPLETE.json").unlink()
    with pytest.raises(FileNotFoundError): ts.load_training_policy(run / "policy_final.pt")


@pytest.mark.parametrize("model", ["shared_tdec", "algorithm2"])
def test_full_budget_artifact_verification_without_training(tmp_path, monkeypatch, model):
    """Construct logged arrays and actor artifacts, never run an environment/learner."""
    src, _ = source(tmp_path / "old", model, actor_hidden=[8, 4])
    batch = tmp_path / "batch"
    i = 0 if model == "shared_tdec" else 18
    run = tc.attempt_run(batch, i, "a01")
    cfg, _ = tc.derive(src, run.parent)
    run.mkdir(parents=True)
    dump(run / "config.resolved.json", cfg.to_dict())
    dump(run / "provenance.json", {})
    selector = ts.TrainingRewardBest(run, cfg)
    policy = FakeTrainer(cfg) if model != "algorithm2" else [FakeAgent(cfg) for _ in range(5)]
    timing = "after_episode_before_pending_ppo_update" if model != "algorithm2" else "after_episode_online_actor_updates"
    for e in range(1, 2501):
        before = (e - 1) // 5 if model != "algorithm2" else max(0, (e - 1) * 100 - cfg.batch_size + 1) // cfg.policy_delay
        after = before if model != "algorithm2" else max(0, e * 100 - cfg.batch_size + 1) // cfg.policy_delay
        selector.observe(np.ones((100, 5), np.float32), np.zeros((100, 5), np.float32),
                         np.zeros(100, np.float32), policy, before, after, timing)
    final_version = 500 if model != "algorithm2" else max(0, 250000 - cfg.batch_size + 1) // cfg.policy_delay
    selection = selector.finish(policy, final_version)
    payload = torch.load(run / ts.BEST, map_location="cpu", weights_only=False)
    manifest = {"implementation_commit": payload["reproduction_git_commit"]}
    dump(run / "COMPLETE.json", {**manifest, "reproduction_git_commit": manifest["implementation_commit"],
         "status": "complete", "algorithm": cfg.algorithm, "episodes": 2500, "config_hash": cfg.canonical_hash(),
         "training_reward_selection": selection, "update_count": 500, "environment_steps": 250000,
         "actor_optimizer_step_count": 5000})
    np.savez_compressed(run / "train_metrics.npz", task1_step=np.ones((2500, 100, 5), np.float32),
                        task2_step=np.zeros((2500, 100, 5), np.float32), global_step=np.zeros((2500, 100), np.float32))
    monkeypatch.setattr(tc, "load", lambda _: (manifest, [src]))
    monkeypatch.setattr(tc.dc, "source_check", lambda *a, **kw: None)
    checked = tc.verify_run(batch, i, run)
    assert checked["best_episode"] == 100 and checked["best_training100_score"] == 1.
    assert checked["final_policy_version"] == final_version
    # A superficially complete artifact is not sufficient: recompute the full ledger.
    ledger = run / "training_reward_selection.csv"
    text = ledger.read_text()
    ledger.write_text(text.replace("True", "False", 1))
    with pytest.raises(ValueError, match="strict-improvement"): tc.verify_run(batch, i, run)


def test_summary_export_readback_and_exclusion_without_trajectories(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    manifest = {"implementation_commit": "synthetic"}
    monkeypatch.setattr(tc, "load", lambda _: (manifest, []))
    fake_records = {}
    for i in range(24):
        run = tc.attempt_run(batch, i, "a01")
        run.mkdir(parents=True)
        with (run / "training_reward_selection.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "episode_reward"])
            writer.writeheader()
            writer.writerows({"episode": e, "episode_reward": 1.} for e in range(1, 2501))
        record = {**tc.cell(i), "run_dir": str(run), "best_episode": 100}
        dump(batch / f"records/train_{i:02d}.json", record)
        fake_records[i] = record
        for name in ("COMPLETE.json", "provenance.json", "config.resolved.json"):
            dump(run / name, {"synthetic": True})
        (run / "train_metrics.npz").write_bytes(b"must not be packed")
        (run / ts.BEST).write_bytes(b"must not be packed")
    monkeypatch.setattr(tc, "verify_run", lambda b, i, run: fake_records[i])
    result = tc.summary(batch)
    assert result["episode_rows"] == 60000 and result["training_env_steps"] == 6000000
    out = batch / "analysis/a01"
    dump(out / "runtime_report.json", {"synthetic": True})
    exported = tc.export(batch)
    with zipfile.ZipFile(exported["archive"]) as z:
        assert not any(n.endswith((".pt", ".npz")) for n in z.namelist())
        assert z.testzip() is None
        rows = list(csv.DictReader(io.StringIO(z.read("analysis/a01/training_episode_curves.csv").decode("utf-8-sig"))))
        assert len(rows) == 60000 and int(rows[0]["environment_steps"]) == 100
        assert int(rows[2499]["environment_steps"]) == 250000
    with pytest.raises(FileExistsError): tc.summary(batch)
    with pytest.raises(FileExistsError): tc.export(batch)


def test_submit_cli_dry_run_and_pilot_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tc, "load", lambda _: ({"implementation_commit": "fixed"}, []))
    monkeypatch.setattr(sj.subprocess, "check_output", lambda *a, **kw: pytest.fail("dry-run must not submit"))
    dump(tmp_path / "protocol/execution_decision.json", {"decision": "24_FROM_SCRATCH"})
    assert sj.main(["--batch-root", str(tmp_path), "--stage", "pilot"]) == 0
    assert json.loads(capsys.readouterr().out)["submitted"] is False
    with pytest.raises(FileNotFoundError): sj.main(["--batch-root", str(tmp_path), "--stage", "main"])
    dump(tmp_path / "protocol/pilot_verification.json", {"technical_status": "PASS"})
    assert sj.main(["--batch-root", str(tmp_path), "--stage", "main"]) == 0


def test_loader_preserves_rng_on_architecture_construction(tmp_path, monkeypatch):
    install_fake(monkeypatch)
    cfg = config(tmp_path, mappo_actor_sharing=True)
    run = Path(r.train(cfg, training_reward_best=True)["run_dir"])
    before = capture_rng_state()
    ts.load_training_actors(run / ts.BEST)
    assert_same_rng(before, capture_rng_state())


def test_historical_audit_is_read_only_and_requires_b1_finals(tmp_path, monkeypatch):
    sources = [source(tmp_path / "old", model, seed=seed)[0] for model in tc.MODELS for seed in range(8, 14)]
    batch = tmp_path / "new"
    b1 = tmp_path / "b1"
    manifest = {"implementation_commit": "synthetic", "b1_root": str(b1)}
    policy_payloads = {}
    for src in sources:
        if src["model_id"] == "algorithm2":
            policy_payloads[src["checkpoint"]["path"]] = dict.fromkeys(("agents", "learner", "replay", "environment", "metrics", "rng"))
            continue
        spec = tc.dc.MODELS[src["model_id"]]
        old_run = b1 / f"mappo_budget_{spec['structure']}_{spec['variant']}_p05_n04_g25_seed{src['training_seed']:02d}_b250k"
        cfg, _ = tc.derive(src, old_run.parent)
        dump(old_run / "COMPLETE.json", {"status": "complete", "episodes": 2500,
             "reproduction_git_commit": "09c38b48e98bae66f2c4ade3e3dee2f7b5d44a54", "config_hash": cfg.canonical_hash()})
        cp = old_run / "policy_final.pt"
        cp.write_bytes(b"not loaded: synthetic audit")
        policy_payloads[str(cp)] = {"episode": 2500, "config": cfg.to_dict()}
    calls = []
    monkeypatch.setattr(tc, "load", lambda _: (manifest, sources))
    monkeypatch.setattr(tc.dc, "source_check", lambda src, payload=False: calls.append(payload))
    monkeypatch.setattr(tc, "rewards_from_npz", lambda path, episodes: [1.] * episodes)
    monkeypatch.setattr(tc.torch, "load", lambda path, **kw: policy_payloads[str(path)])
    before = {p: p.read_bytes() for root in (tmp_path / "old", b1) for p in root.rglob("*") if p.is_file()}
    result = tc.audit(batch)
    assert len(calls) == 24 and all(calls)
    assert result["fresh_mappo_runs"] == 18 and result["fresh_maddpg_runs"] == 6
    assert result["legacy_resume_allowed"] is False and all(r["decision"] == "FROM_SCRATCH" for r in result["rows"])
    assert before == {p: p.read_bytes() for p in before}
    assert tc.approve(batch)["decision"] == "24_FROM_SCRATCH"
    next(b1.rglob("policy_final.pt")).unlink()
    with pytest.raises(ValueError, match="B1250k final missing"): tc.audit(batch)
