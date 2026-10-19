"""Small synthetic environment; real networks/Adam/replay, no formal rollout."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.algorithms.modified_maddpg.agent import make_agents
from aoi_v2x_reproduction.algorithms.modified_maddpg.learner import Global_Critic
from aoi_v2x_reproduction.algorithms.modified_maddpg.replay import ReplayBuffer
from aoi_v2x_reproduction.runtime import runner as r, shared_resume as sr
from aoi_v2x_reproduction.runtime.metrics import MetricStore
from aoi_v2x_reproduction.runtime.training_selection import load_training_policy
from tests.test_training100 import FakeEnv, assert_same_rng


class StatefulSyntheticEnv(FakeEnv):
    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "initialized": self._world_initialized,
                "episode": self.e, "step_index": self.step_index}

    def load_state_dict(self, state):
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng"]
        self._world_initialized = state["initialized"]
        self.e, self.step_index = state["episode"], state["step_index"]


def config(root, name="run"):
    return resolve_config(algorithm="modified_maddpg_tdec", scenario="p05_n04_g25", seed=8,
                          episodes=104, steps_per_episode=2, batch_size=4, replay_capacity=16,
                          actor_hidden=[8, 4], local_critic_hidden=[8, 4], global_critic_hidden=[8, 4, 2],
                          checkpoint_mode="policy_only", maddpg_actor_sharing=True,
                          device="cpu", output_root=str(root), run_name=name, diagnostics=True)


@pytest.fixture
def synthetic(monkeypatch):
    def factory(cfg):
        r.seed_everything(cfg.seed, torch.device("cpu"))
        agents = make_agents(cfg)
        assert len({id(a.actor) for a in agents}) == len({id(a.target_actor) for a in agents}) == 1
        return (StatefulSyntheticEnv(cfg), agents, Global_Critic(cfg, agents),
                ReplayBuffer(cfg.replay_capacity, cfg.state_dim, cfg.action_dim, cfg.number_agents),
                MetricStore(5, cfg.steps_per_episode, diagnostics=cfg.diagnostics), torch.device("cpu"))
    monkeypatch.setattr(r, "_make_system", factory)
    monkeypatch.setattr(r, "_selection_validation", lambda *a: pytest.fail("no validation worlds allowed"))
    return factory


def read(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal_tree(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("cut", [50, 100])
def test_fork_resume_exact_networks_adam_replay_rng_metrics_best100(tmp_path, synthetic, cut):
    full = config(tmp_path, "full")
    partial = config(tmp_path, "partial")
    resumed = config(tmp_path, "resumed")
    r.train(full, training_reward_best=True, shared_resume_interval=50)
    r.train(partial, training_reward_best=True, shared_resume_interval=50, max_episodes=cut)
    source = tmp_path / "partial" / "checkpoints" / f"episode_{cut:06d}.pt"
    before = source.read_bytes()
    payload = read(source)
    assert payload["training100_selector"]["episode"] == cut
    assert (payload["training100_selector"]["best_payload"] is None) == (cut < 100)
    assert all("actor" not in a and "actor_optimizer" not in a for a in payload["agents"])
    assert set(payload["shared_actor"]) == {"actor", "target_actor", "actor_optimizer"}
    r.train(resumed, resume=str(source), training_reward_best=True, shared_resume_interval=50)
    assert source.read_bytes() == before
    assert not (tmp_path / "partial" / "COMPLETE.json").exists()
    a, b = [read(tmp_path / run / "checkpoints/episode_000104.pt") for run in ("full", "resumed")]
    for key in ("agents", "shared_actor", "learner", "replay", "environment", "metrics"):
        equal_tree(a[key], b[key])
    assert_same_rng(a["rng"], b["rng"])
    for name in ("policy_best_training.pt", "policy_final.pt"):
        _, ap = load_training_policy(tmp_path / "full" / name)
        _, bp = load_training_policy(tmp_path / "resumed" / name)
        equal_tree(ap["actors"], bp["actors"])
        equal_tree(ap["training_reward_selection"], bp["training_reward_selection"])
    assert (tmp_path / "full/training_reward_selection.csv").read_bytes() == (tmp_path / "resumed/training_reward_selection.csv").read_bytes()
    assert a["learner"]["actor_optimizer_step_count"] == (104 * 2 - 4 + 1) // 2
    assert a["learner"]["target_actor_update_count"] == a["learner"]["actor_optimizer_step_count"]
    assert len(list((tmp_path / "full/checkpoints").glob("episode_*.pt"))) == 3


def test_checkpointing_does_not_change_training_or_rng(tmp_path, synthetic):
    states = []
    for interval, name in ((None, "old"), (50, "new")):
        r.train(config(tmp_path, name), training_reward_best=True, shared_resume_interval=interval)
        states.append(r.capture_rng_state())
    assert_same_rng(*states)
    equal_tree(read(tmp_path / "old/policy_final.pt")["actors"], read(tmp_path / "new/policy_final.pt")["actors"])
    with np.load(tmp_path / "old/train_metrics.npz") as a, np.load(tmp_path / "new/train_metrics.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])


def test_resume_gates_and_cli_dry_run(tmp_path, synthetic, monkeypatch, capsys):
    cfg = config(tmp_path, "source")
    r.train(cfg, training_reward_best=True, shared_resume_interval=50, max_episodes=50)
    source = tmp_path / "source/checkpoints/episode_000050.pt"
    target = config(tmp_path, "fork")
    changed = copy.deepcopy(target)
    changed.seed += 1
    with pytest.raises(ValueError, match="configuration"):
        sr.read_resume(source, changed, 50)
    changed = read(source)
    changed["runtime_identity"]["torch"] = "wrong"
    with monkeypatch.context() as m:
        m.setattr(sr.torch, "load", lambda *a, **k: changed)
        with pytest.raises(ValueError, match="runtime"):
            sr.read_resume(source, target, 50)
    with pytest.raises(FileExistsError):
        r.train(cfg, resume=str(source), training_reward_best=True, shared_resume_interval=50)
    with pytest.raises(ValueError, match="every 50"):
        r.train(target, training_reward_best=True, shared_resume_interval=20)
    with pytest.raises(ValueError, match="does not support"):
        r.train(target, resume=str(source), training_reward_best=True)
    sr.main(["--resume", str(source), "--output-root", str(tmp_path), "--run-name", "dry", "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert report["interval"] == 50
    assert (report["completed_episodes"], report["next_episode"], report["remaining_episodes"]) == (50, 51, 54)
    assert not (tmp_path / "dry").exists()


def test_resume_rejects_actor_only_and_changed_commit(tmp_path, synthetic, monkeypatch):
    r.train(config(tmp_path, "source"), training_reward_best=True, shared_resume_interval=50)
    target = config(tmp_path, "fork")
    with pytest.raises(ValueError, match="not a resumable"):
        sr.read_resume(tmp_path / "source/policy_best_training.pt", target, 50)
    with pytest.raises(ValueError, match="final"):
        sr.read_resume(tmp_path / "source/checkpoints/episode_000104.pt", target, 50)
    monkeypatch.setattr(sr, "_git_stamp", lambda: {"reproduction_git_commit": "different"})
    with pytest.raises(ValueError, match="commit changed"):
        sr.read_resume(tmp_path / "source/checkpoints/episode_000050.pt", target, 50)


def test_fresh_cli_dry_run_only_changes_shared_identity_and_output(tmp_path, capsys):
    cfg = config(tmp_path, "parent")
    cfg.maddpg_actor_sharing = False
    source = tmp_path / "config.resolved.json"
    source.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    before = source.read_bytes()
    sr.main(["--config", str(source), "--output-root", str(tmp_path / "new"), "--run-name", "shared", "--dry-run"])
    report = json.loads(capsys.readouterr().out)
    assert set(report["changed_fields"]) == {"maddpg_actor_sharing", "output_root", "run_name"}
    assert (report["completed_episodes"], report["next_episode"], report["remaining_episodes"]) == (0, 1, 104)
    assert source.read_bytes() == before and not (tmp_path / "new").exists()


def test_batch_acceptance_records_resume_instead_of_fresh_attempt(tmp_path, monkeypatch):
    from analysis import shared_maddpg_contract as sc
    run = tmp_path / "training/a02/runs/train100_shared_maddpg_tdec_seed08"
    source = tmp_path / "training/a01/runs/train100_shared_maddpg_tdec_seed08/checkpoints/episode_000050.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"synthetic checkpoint identity only")
    run.mkdir(parents=True)
    cfg = config(run.parent, run.name)
    cfg.episodes, cfg.steps_per_episode = 2500, 100
    updates = (250000 - cfg.batch_size + 1) // cfg.policy_delay
    complete = {"actor_sharing": True, "model_id": sc.MODEL, "actor_network_count": 1,
                "target_actor_network_count": 1, "actor_optimizer_step_count": updates,
                "target_actor_update_count": updates, "learn_step_count": 250000 - cfg.batch_size + 1,
                "episode_resume": {"schema": sr.SCHEMA, "interval": 50, "resume_source": str(source), "resume_episode": 50}}
    def dump(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    dump(run / "config.resolved.json", cfg.to_dict())
    dump(run / "COMPLETE.json", complete)
    dump(run / "learning_diagnostics.json", [{"actor_update_count": updates}] + [{"actor_update_count": 0}] * 2499)
    dump(tmp_path / "records/started_00_a02.json", {"resume_source": str(source),
                                                   "resume_source_identity": sc.dc.file_identity(source)})
    monkeypatch.setattr(sc.tc, "verify_run", lambda *a, **k: {"from_scratch": True})
    checked = sc.verify_run(tmp_path, 0, run)
    assert checked["from_scratch"] is False and checked["inherited_episodes"] == 50
    assert checked["attempt_environment_steps"] == 245000
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="lineage changed"):
        sc.verify_run(tmp_path, 0, run)


def test_failed_snapshot_preserves_previous_checkpoint_pointer_and_rng(tmp_path, monkeypatch):
    """A failed save must not advertise a new checkpoint or consume RNG."""
    from types import SimpleNamespace
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    previous = folder / "episode_000050.pt"
    previous.write_bytes(b"previous completed snapshot")
    pointer = folder / "latest.json"
    pointer.write_text('{"episode": 50}', encoding="utf-8")
    original = {p.name: p.read_bytes() for p in folder.iterdir()}
    env = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None)
    metrics = SimpleNamespace(episodes=[None] * 100)
    selector = SimpleNamespace(episode=100, state_dict=lambda: {})
    monkeypatch.setattr(sr, "build_payload", lambda *a, **k: {"environment": {}})
    monkeypatch.setattr(sr, "runtime_identity", lambda cfg: {})

    def failed_save(payload, path):
        np.random.random()
        torch.rand(1)
        path.with_suffix(".pt.tmp").write_bytes(b"incomplete snapshot")
        raise OSError("synthetic disk full")

    monkeypatch.setattr(sr, "atomic_torch_save", failed_save)
    before = r.capture_rng_state()
    with pytest.raises(OSError, match="disk full"):
        sr.save_checkpoint(tmp_path, config(tmp_path), [], None, None, env, metrics, selector, 100, 50)
    assert_same_rng(before, r.capture_rng_state())
    assert not (folder / "episode_000100.pt").exists()
    assert all((folder / name).read_bytes() == data for name, data in original.items())


def test_shared_code_guard_still_rejects_dirty_checkout(monkeypatch):
    from analysis import shared_maddpg_contract as sc
    responses = {("rev-parse", "HEAD"): "synthetic_commit",
                 ("branch", "--show-current"): "research/mappo",
                 ("status", "--porcelain", "--untracked-files=all"): " M tracked.py"}
    monkeypatch.setattr(sc.dc, "git", lambda *args: responses[args])
    with pytest.raises(ValueError, match="pinned clean"):
        sc.code_check("synthetic_commit")
