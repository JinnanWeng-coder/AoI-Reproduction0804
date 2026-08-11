from pathlib import Path

import numpy as np
import pytest
import torch

from aoi_v2x_reproduction.algorithms.mappo import (
    MAPPOTrainer,
    encode_hybrid_actions,
)
from aoi_v2x_reproduction.algorithms.mappo.networks import HybridActor
from aoi_v2x_reproduction.algorithms.mappo.rollout import OnPolicyRollout, compute_gae
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.envs import PaperEnviron


def _mappo_config(root: Path | None = None, **overrides):
    values = {
        "algorithm": "mappo",
        "seed": 71,
        "episodes": 2,
        "steps_per_episode": 3,
        "actor_hidden": [16, 8],
        "global_critic_hidden": [16, 8, 4],
        "device": "cpu",
        "mappo_rollout_episodes": 1,
        "mappo_ppo_epochs": 2,
    }
    if root is not None:
        values.update(output_root=str(root), run_name="mappo-test")
    values.update(overrides)
    return resolve_config(scenario="p05_n04_g25", **values)


def test_native_hybrid_actions_round_trip_through_unchanged_environment():
    config = _mappo_config()
    environment = PaperEnviron(config)
    rb = np.asarray([0, 1, 2, 0, 2], dtype=np.int64)
    mode = np.asarray([0, 1, 0, 1, 1], dtype=np.int64)
    power_unit = np.asarray([0.01, 0.25, 0.5, 0.75, 0.99], dtype=np.float32)

    encoded = encode_hybrid_actions(rb, mode, power_unit, config.n_rb, config.n_modes)
    decoded = environment.decode_actions(encoded)

    np.testing.assert_array_equal(decoded[:, 0].astype(np.int64), rb)
    np.testing.assert_array_equal(decoded[:, 1].astype(np.int64), mode)
    expected_power = config.power_min_dbm + power_unit * (config.power_max_dbm - config.power_min_dbm)
    np.testing.assert_allclose(decoded[:, 2], expected_power, rtol=0.0, atol=2e-6)


def test_policy_log_probability_is_consistent_before_an_update():
    torch.manual_seed(4)
    actor = HybridActor(obs_dim=22, hidden_dims=[16, 8], n_rb=3, n_modes=2)
    observations = torch.randn(11, 22)
    sampled = actor.sample(observations)
    evaluated = actor.evaluate_actions(observations, sampled.rb, sampled.mode, sampled.power)

    torch.testing.assert_close(evaluated.log_prob, sampled.log_prob)
    torch.testing.assert_close(torch.exp(evaluated.log_prob - sampled.log_prob), torch.ones(11))
    assert torch.isfinite(sampled.entropy_rb).all()
    assert torch.isfinite(sampled.entropy_mode).all()
    assert torch.isfinite(sampled.entropy_power).all()
    deterministic_a = actor.sample(observations, deterministic=True)
    deterministic_b = actor.sample(observations, deterministic=True)
    torch.testing.assert_close(deterministic_a.rb, deterministic_b.rb)
    torch.testing.assert_close(deterministic_a.mode, deterministic_b.mode)
    torch.testing.assert_close(deterministic_a.power, deterministic_b.power)


def test_gae_resets_at_episode_boundaries():
    rewards = torch.tensor([[1.0], [1.0], [2.0], [2.0]])
    zeros = torch.zeros_like(rewards)
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0])

    advantages, returns = compute_gae(rewards, zeros, zeros, dones, gamma=1.0, gae_lambda=1.0)

    expected = torch.tensor([[2.0], [1.0], [4.0], [2.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_rollout_is_versioned_and_consumed_once():
    rollout = OnPolicyRollout(number_agents=2, observation_dim=3)
    zeros_agent = np.zeros(2, dtype=np.float32)
    for done in (False, True):
        rollout.append(
            observations=np.zeros((2, 3), dtype=np.float32),
            rb=np.zeros(2, dtype=np.int64),
            mode=np.ones(2, dtype=np.int64),
            power=np.full(2, 0.5, dtype=np.float32),
            old_log_prob=zeros_agent,
            values=zeros_agent,
            rewards=np.ones(2, dtype=np.float32),
            done=done,
            next_values=zeros_agent,
            policy_version=3,
        )
    with pytest.raises(RuntimeError, match="stale"):
        rollout.consume(gamma=0.99, gae_lambda=0.95, expected_policy_version=4)
    batch = rollout.consume(gamma=0.99, gae_lambda=0.95, expected_policy_version=3)
    assert batch.size == 2
    assert len(rollout) == 0
    with pytest.raises(RuntimeError, match="empty"):
        rollout.consume(gamma=0.99, gae_lambda=0.95, expected_policy_version=3)


def test_mappo_update_is_finite_and_changes_each_local_policy():
    torch.manual_seed(9)
    np.random.seed(9)
    config = _mappo_config()
    trainer = MAPPOTrainer(config, torch.device("cpu"))
    rollout = OnPolicyRollout(config.number_agents, config.state_dim)
    before = [[parameter.detach().clone() for parameter in actor.parameters()] for actor in trainer.actors]

    observations = np.random.default_rng(9).normal(size=(config.number_agents, config.state_dim)).astype(np.float32)
    for step_index in range(6):
        sampled = trainer.act(observations)
        next_observations = np.random.default_rng(100 + step_index).normal(
            size=(config.number_agents, config.state_dim)
        ).astype(np.float32)
        done = step_index in {2, 5}
        rollout.append(
            observations=observations,
            rb=sampled.rb,
            mode=sampled.mode,
            power=sampled.power,
            old_log_prob=sampled.log_prob,
            values=sampled.values,
            rewards=np.linspace(-1.0, 1.0, config.number_agents, dtype=np.float32) + 0.1 * step_index,
            done=done,
            next_values=np.zeros(config.number_agents, dtype=np.float32) if done else trainer.values(next_observations),
            policy_version=trainer.policy_version,
        )
        observations = next_observations

    batch = rollout.consume(config.gamma, config.mappo_gae_lambda, trainer.policy_version)
    diagnostics = trainer.update(batch)

    assert diagnostics["policy_version"] == 1
    assert diagnostics["rollout_steps"] == 6
    assert all(np.isfinite(value) for value in diagnostics["approx_kl_per_agent"])
    for actor_before, actor_after in zip(before, trainer.actors):
        assert any(not torch.equal(old, new) for old, new in zip(actor_before, actor_after.parameters()))


def test_actors_are_local_while_value_estimates_use_joint_observations():
    torch.manual_seed(12)
    config = _mappo_config()
    trainer = MAPPOTrainer(config, torch.device("cpu"))
    rng = np.random.default_rng(12)
    observations = rng.normal(size=(config.number_agents, config.state_dim)).astype(np.float32)
    changed_others = observations.copy()
    changed_others[1:] = rng.normal(size=changed_others[1:].shape).astype(np.float32) * 7.0

    first = trainer.act(observations, deterministic=True)
    second = trainer.act(changed_others, deterministic=True)

    assert first.rb[0] == second.rb[0]
    assert first.mode[0] == second.mode[0]
    assert first.power[0] == pytest.approx(second.power[0], abs=0.0)
    assert not np.allclose(first.values, second.values)
    first_parameters = {id(parameter) for parameter in trainer.actors[0].parameters()}
    for actor in trainer.actors[1:]:
        assert first_parameters.isdisjoint(id(parameter) for parameter in actor.parameters())
