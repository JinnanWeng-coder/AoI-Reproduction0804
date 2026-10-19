"""Synthetic release checks: six fresh runs, sharing semantics, no rollout."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from analysis import shared_maddpg_contract as sc, run_training100_local as queue
from aoi_v2x_reproduction.algorithms.modified_maddpg.agent import make_agents
from aoi_v2x_reproduction.algorithms.modified_maddpg.learner import Global_Critic
from tests.test_shared_maddpg_resume import config
from tests.test_training100_selected_eval import fake_training_batch


def test_six_seed_derivation_and_no_evaluation(tmp_path, monkeypatch):
    training, _ = fake_training_batch(tmp_path, monkeypatch)
    monkeypatch.setattr(sc, "code_check", lambda expected=None: "shared_commit")
    before = {p: p.read_bytes() for p in training.rglob("*") if p.is_file()}
    m = sc.init(training, "shared_test", "shared_commit")
    batch = Path(m["batch_root"])
    loaded, sources = sc.load(batch)
    assert loaded == m and len(sources) == 6
    assert [sc.cell(i)["training_seed"] for i in range(6)] == list(range(8, 14))
    assert m["new_evaluation_units"] == 0 and m["training_env_steps"] == 1500000
    assert m["concurrent_training_processes"] == 2 and m["resume_interval"] == 50
    for source in sources:
        cfg, delta = sc.derive(source, batch / "training/a01/runs")
        assert set(delta) == {"maddpg_actor_sharing", "run_name", "output_root"}
        assert cfg.maddpg_actor_sharing and cfg.episodes == 2500 and cfg.steps_per_episode == 100
        assert cfg.seed == source["training_seed"] and cfg.exploration_noise == .3
    assert before == {p: p.read_bytes() for p in training.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="unique"):
        sc.init(training, "shared_test", "shared_commit")


def test_shared_actor_one_mean_gradient_step_and_one_target_update(tmp_path):
    torch.manual_seed(91)
    cfg = config(tmp_path)
    agents = make_agents(cfg)
    learner = Global_Critic(cfg, agents)
    actor = agents[0].actor
    assert len({id(a.actor.optimizer) for a in agents}) == 1
    assert len({id(m) for a in agents for m in a.local_critics}) == 10
    states = torch.randn(4, cfg.state_dim * cfg.number_agents)
    g, local, _, _ = learner._actor_component_terms(states)
    expected = torch.autograd.grad((g + local) / cfg.number_agents, list(actor.parameters()))
    target_before = [p.detach().clone() for p in agents[0].target_actor.parameters()]
    online_critics = [learner.global_critic1, learner.global_critic2] + list(learner._local_critic_modules())
    before = [p.detach().clone() for m in online_critics for p in m.parameters()]
    result = learner._shared_actor_step(states)
    for p, gradient in zip(actor.parameters(), expected):
        torch.testing.assert_close(p.grad, gradient, rtol=1e-5, atol=1e-7)
        assert actor.optimizer.state[p]["step"].item() == 1
    for old, source, target in zip(target_before, actor.parameters(), agents[0].target_actor.parameters()):
        torch.testing.assert_close(target, cfg.tau * source + (1-cfg.tau) * old, rtol=0, atol=0)
    for old, current in zip(before, [p for m in online_critics for p in m.parameters()]):
        torch.testing.assert_close(old, current, rtol=0, atol=0)
    assert result["actor_optimizer_step_count"] == result["target_actor_update_count"] == 1


def test_shared_serial_queue_starts_six_and_exports_once(tmp_path, monkeypatch):
    batch = tmp_path / "study/batch"
    for folder in ("records", "logs", "tmp", "analysis"):
        (batch / folder).mkdir(parents=True, exist_ok=True)
    started, completed = [], []
    manifest = {"study_root": str(batch.parent), "runs": [sc.cell(i) for i in range(6)],
                "implementation_commit": "synthetic"}
    workflow = SimpleNamespace(CLI_MODULE=sc.CLI_MODULE, load=lambda b: (manifest, []),
        preflight=lambda b: {"technical_status": "PASS"}, attempt_run=sc.attempt_run,
        summary=lambda *a: completed.append("summary"), export=lambda *a: completed.append("export"))
    class Process:
        def __init__(self, cmd, **kwargs):
            assert cmd[cmd.index("-m") + 1] == sc.CLI_MODULE
            self.i = int(cmd[cmd.index("--id") + 1])
            self.pid = 100 + self.i
            started.append(self.i)
        def wait(self):
            sc.dc.write(batch / f"records/train_{self.i:02d}.json", {"synthetic": True})
            return 0
        def poll(self): return 0
    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    result = queue.run_queue(batch, workflow=workflow)
    assert started == list(range(6)) and completed == ["summary", "export"]
    assert result["accepted_runs"] == 6 and result["technical_status"] == "PASS"
