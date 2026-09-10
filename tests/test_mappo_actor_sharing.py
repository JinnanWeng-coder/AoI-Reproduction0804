import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from aoi_v2x_reproduction.algorithms.mappo import MAPPOTrainer
from aoi_v2x_reproduction.algorithms.mappo.rollout import OnPolicyRollout
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime.runner import evaluate_from_checkpoint, train


def _config(root: Path | None = None, sharing: bool = False, variant: str = "combined"):
    values = dict(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=82,
        episodes=2,
        steps_per_episode=3,
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        mappo_rollout_episodes=1,
        mappo_ppo_epochs=3,
        mappo_variant=variant,
        mappo_actor_sharing=sharing,
        device="cpu",
        checkpoint_mode="policy_only",
        diagnostics=True,
    )
    if root is not None:
        values.update(output_root=str(root), run_name=f"mappo-{variant}-{'shared' if sharing else 'independent'}")
    return resolve_config(**values)


def _combined_rollout(trainer: MAPPOTrainer, seed: int = 82):
    config = trainer.config
    rollout = OnPolicyRollout(config.number_agents, config.state_dim)
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(config.number_agents, config.state_dim)).astype(np.float32)
    for step in range(6):
        sampled = trainer.act(observations)
        next_observations = rng.normal(size=observations.shape).astype(np.float32)
        done = step in {2, 5}
        rollout.append(
            observations=observations,
            rb=sampled.rb,
            mode=sampled.mode,
            power=sampled.power,
            old_log_prob=sampled.log_prob,
            values=sampled.values,
            rewards=(
                np.linspace(-1.0, 1.0, config.number_agents, dtype=np.float32)
                + np.float32(0.1 * step)
            ),
            done=done,
            next_values=(
                np.zeros(config.number_agents, dtype=np.float32)
                if done else trainer.values(next_observations)
            ),
            policy_version=trainer.policy_version,
        )
        observations = next_observations
    return rollout.consume(config.gamma, config.mappo_gae_lambda, trainer.policy_version)


def test_default_path_has_independent_unique_actors_and_optimizers():
    trainer = MAPPOTrainer(_config(sharing=False), torch.device("cpu"))
    assert trainer.actor_sharing is False
    assert len(trainer.actors) == len(trainer.actor_optimizers) == trainer.number_agents
    parameter_sets = [{id(parameter) for parameter in actor.parameters()} for actor in trainer.actors]
    for index, parameters in enumerate(parameter_sets):
        for other in parameter_sets[index + 1:]:
            assert parameters.isdisjoint(other)
    counts = trainer.parameter_counts()
    assert counts["actor_network_count"] == trainer.number_agents
    assert counts["actors"] == trainer.number_agents * counts["actor_parameters_per_network"]


def test_shared_path_has_one_actual_actor_and_one_optimizer_without_agent_id():
    trainer = MAPPOTrainer(_config(sharing=True), torch.device("cpu"))
    assert trainer.actor_sharing is True
    assert len(trainer.actors) == len(trainer.actor_optimizers) == 1
    assert all(trainer.actor_for_agent(index) is trainer.actors[0] for index in range(5))
    assert trainer.actors[0].fc1.in_features == trainer.config.state_dim
    counts = trainer.parameter_counts()
    assert counts["actor_network_count"] == 1
    assert counts["actors"] == counts["actor_parameters_per_network"]


def test_shared_actor_receives_each_agents_own_observation_and_samples_five_times():
    trainer = MAPPOTrainer(_config(sharing=True), torch.device("cpu"))
    observations = np.stack([
        np.full(trainer.config.state_dim, float(index), dtype=np.float32)
        for index in range(trainer.number_agents)
    ])
    actor = trainer.actors[0]
    with patch.object(actor, "sample", wraps=actor.sample) as sampled:
        result = trainer.act(observations, deterministic=False)
    assert sampled.call_count == trainer.number_agents
    for index, call in enumerate(sampled.call_args_list):
        torch.testing.assert_close(
            call.args[0], torch.as_tensor(observations[index:index + 1])
        )
    assert result.rb.shape == result.mode.shape == result.power.shape == result.log_prob.shape == (5,)
    assert result.environment_actions.shape == (5, 3)


@pytest.mark.parametrize("sharing, expected_steps", [(False, 15), (True, 3)])
def test_optimizer_step_count_and_ppo_shapes_match_actor_sharing_mode(sharing, expected_steps):
    torch.manual_seed(82)
    np.random.seed(82)
    trainer = MAPPOTrainer(_config(sharing=sharing), torch.device("cpu"))
    before = [parameter.detach().clone() for parameter in trainer.actors[0].parameters()]
    diagnostics = trainer.update(_combined_rollout(trainer))
    assert diagnostics["actor_sharing"] is sharing
    assert diagnostics["actor_network_count"] == (1 if sharing else 5)
    assert diagnostics["actor_optimizer_steps_this_update"] == expected_steps
    assert diagnostics["actor_optimizer_step_count"] == expected_steps
    assert diagnostics["ppo_epochs"] == 3
    assert diagnostics["rollout_steps"] == 6
    for key in (
        "actor_loss_per_agent", "entropy_rb_per_agent", "entropy_mode_per_agent",
        "entropy_power_per_agent", "approx_kl_per_agent", "clip_fraction_per_agent",
        "actor_grad_norm_per_agent",
    ):
        assert len(diagnostics[key]) == 5
        assert np.all(np.isfinite(diagnostics[key]))
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.actors[0].parameters())
    )


def test_shared_policy_checkpoint_roundtrip_and_frozen_eval_metadata(tmp_path):
    config = _config(tmp_path / "training", sharing=True, variant="tdec")
    result = train(config)
    run_dir = Path(result["run_dir"])
    complete = json.loads((run_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    payload = torch.load(run_dir / "policy_final.pt", map_location="cpu", weights_only=False)
    assert resolved["mappo_actor_sharing"] is True
    assert complete["actor_sharing"] is provenance["actor_sharing"] is True
    assert complete["actor_network_count"] == 1
    assert complete["parameter_counts"]["actor_network_count"] == 1
    assert payload["actor_sharing"] is True and len(payload["actors"]) == 1

    evaluation_config = _config(tmp_path / "evaluation", sharing=False, variant="tdec")
    evaluated = evaluate_from_checkpoint(
        evaluation_config,
        str(run_dir / "policy_final.pt"),
        eval_episodes=1,
        eval_seeds=[207],
        eval_purpose="validation",
        scope="validation",
        diagnostic_eval=True,
        mappo_eval_mode="stochastic",
    )
    assert evaluated["actor_sharing"] is True
    assert evaluated["actor_network_count"] == 1
    assert Path(evaluated["eval_dir"]).joinpath("EVAL_COMPLETE.json").is_file()
