from pathlib import Path

import numpy as np
import pytest
import torch

from aoi_v2x_reproduction.algorithms.mappo import (
    MAPPOTrainer,
    encode_hybrid_actions,
)
from aoi_v2x_reproduction.algorithms.mappo.networks import HybridActor, RunningValueNorm
from aoi_v2x_reproduction.algorithms.mappo.rollout import OnPolicyRollout, TDecOnPolicyRollout, compute_gae
from aoi_v2x_reproduction.algorithms.mappo.trainer import _mean_joint_gradient_rss, _value_loss_inputs
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.envs import PaperEnviron


def _mappo_config(root: Path | None = None, **overrides):
    values = {
        "algorithm": "mappo",
        "seed": 71,
        "episodes": 2,
        "steps_per_episode": 3,
        "actor_hidden": [16, 8],
        "local_critic_hidden": [16, 8],
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


def test_tdec_rollout_retains_three_raw_streams_and_computes_their_gae():
    rollout = TDecOnPolicyRollout(number_agents=2, observation_dim=3)
    zeros_agent = np.zeros(2, dtype=np.float32)
    zeros_global = np.zeros(1, dtype=np.float32)
    for done in (False, True):
        rollout.append(
            observations=np.zeros((2, 3), dtype=np.float32),
            rb=np.zeros(2, dtype=np.int64),
            mode=np.ones(2, dtype=np.int64),
            power=np.full(2, 0.5, dtype=np.float32),
            old_log_prob=zeros_agent,
            global_values=zeros_global,
            task1_values=zeros_agent,
            task2_values=zeros_agent,
            global_rewards=np.ones(1, dtype=np.float32),
            task1_rewards=np.asarray([1.0, 2.0], dtype=np.float32),
            task2_rewards=np.asarray([3.0, 4.0], dtype=np.float32),
            done=done,
            global_next_values=zeros_global,
            task1_next_values=zeros_agent,
            task2_next_values=zeros_agent,
            policy_version=4,
        )
    batch = rollout.consume(gamma=1.0, gae_lambda=1.0, expected_policy_version=4)

    assert batch.global_rewards.shape == (2, 1)
    assert batch.task1_rewards.shape == batch.task2_rewards.shape == (2, 2)
    assert batch.global_next_values.shape == (2, 1)
    assert batch.task1_next_values.shape == batch.task2_next_values.shape == (2, 2)
    torch.testing.assert_close(batch.global_advantages, torch.tensor([[2.0], [1.0]]))
    torch.testing.assert_close(batch.task1_advantages, torch.tensor([[2.0, 4.0], [1.0, 2.0]]))
    torch.testing.assert_close(batch.task2_advantages, torch.tensor([[6.0, 8.0], [3.0, 4.0]]))
    torch.testing.assert_close(batch.global_returns, batch.global_advantages)
    torch.testing.assert_close(batch.task1_returns, batch.task1_advantages)
    torch.testing.assert_close(batch.task2_returns, batch.task2_advantages)

    with pytest.raises(ValueError, match="global_values"):
        TDecOnPolicyRollout(2, 3).append(
            observations=np.zeros((2, 3), dtype=np.float32),
            rb=np.zeros(2, dtype=np.int64),
            mode=np.ones(2, dtype=np.int64),
            power=np.full(2, 0.5, dtype=np.float32),
            old_log_prob=zeros_agent,
            global_values=np.zeros(2, dtype=np.float32),
            task1_values=zeros_agent,
            task2_values=zeros_agent,
            global_rewards=np.ones(1, dtype=np.float32),
            task1_rewards=zeros_agent,
            task2_rewards=zeros_agent,
            done=True,
            global_next_values=zeros_global,
            task1_next_values=zeros_agent,
            task2_next_values=zeros_agent,
            policy_version=4,
        )


@pytest.mark.parametrize("value_clip_mode", ["normalized", "legacy_raw"])
def test_mappo_update_is_finite_and_changes_each_local_policy(value_clip_mode):
    torch.manual_seed(9)
    np.random.seed(9)
    config = _mappo_config(mappo_value_clip_mode=value_clip_mode)
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
    assert np.isfinite(diagnostics["critic_loss"])
    assert np.isfinite(diagnostics["critic_grad_norm"])
    assert diagnostics["critic_loss_aggregate_semantics"] == "mean_over_epochs_of_combined_critic_loss"
    assert diagnostics["critic_grad_norm_aggregate_semantics"] == (
        "mean_over_epochs_of_combined_critic_grad_l2_before_clipping"
    )
    assert diagnostics["explained_variance_aggregate_semantics"] == (
        "mean_over_valid_per_agent_combined_value_outputs"
    )
    assert all(np.isfinite(value) for value in diagnostics["approx_kl_per_agent"])
    for actor_before, actor_after in zip(before, trainer.actors):
        assert any(not torch.equal(old, new) for old, new in zip(actor_before, actor_after.parameters()))


def test_normalized_value_clipping_is_affine_scale_invariant():
    returns = torch.tensor([
        [-40.0, 10.0],
        [-10.0, 15.0],
        [20.0, 25.0],
        [50.0, 30.0],
    ])
    old_values = returns + torch.tensor([
        [2.0, -3.0],
        [-4.0, 1.0],
        [3.0, 2.0],
        [-1.0, -2.0],
    ])
    predictions = old_values + torch.tensor([
        [15.0, -8.0],
        [-12.0, 6.0],
        [9.0, 7.0],
        [-11.0, -5.0],
    ])

    base_norm = RunningValueNorm(number_agents=2)
    base_norm.update(returns)
    base = _value_loss_inputs(
        predictions, old_values, returns, base_norm, clip_param=0.2, clip_mode="normalized"
    )
    normalized_old = base_norm.normalize(old_values)
    expected_clipped = normalized_old + torch.clamp(base[0] - normalized_old, -0.2, 0.2)
    torch.testing.assert_close(base[1], expected_clipped)

    scale = torch.tensor([3.0, 0.5])
    shift = torch.tensor([100.0, -7.0])
    scaled_returns = returns * scale + shift
    scaled_norm = RunningValueNorm(number_agents=2)
    scaled_norm.update(scaled_returns)
    scaled = _value_loss_inputs(
        predictions * scale + shift,
        old_values * scale + shift,
        scaled_returns,
        scaled_norm,
        clip_param=0.2,
        clip_mode="normalized",
    )

    for base_tensor, scaled_tensor in zip(base, scaled):
        torch.testing.assert_close(base_tensor, scaled_tensor, rtol=1e-5, atol=1e-5)


def test_legacy_value_clipping_reproduces_raw_space_order():
    returns = torch.tensor([[-20.0], [0.0], [20.0], [40.0]])
    old_values = torch.tensor([[-5.0], [2.0], [12.0], [25.0]])
    predictions = old_values + torch.tensor([[5.0], [-4.0], [3.0], [-2.0]])
    value_norm = RunningValueNorm(number_agents=1)
    value_norm.update(returns)

    normalized_predictions, normalized_clipped, normalized_returns = _value_loss_inputs(
        predictions,
        old_values,
        returns,
        value_norm,
        clip_param=0.2,
        clip_mode="legacy_raw",
    )
    expected_raw_clipped = old_values + torch.clamp(predictions - old_values, -0.2, 0.2)

    torch.testing.assert_close(normalized_predictions, value_norm.normalize(predictions))
    torch.testing.assert_close(normalized_clipped, value_norm.normalize(expected_raw_clipped))
    torch.testing.assert_close(normalized_returns, value_norm.normalize(returns))


def test_tdec_gradient_aggregate_is_joint_rss_per_epoch_then_mean():
    aggregate = _mean_joint_gradient_rss(
        [3.0, 0.0],
        [4.0, 0.0],
        [0.0, 12.0],
    )
    assert aggregate == pytest.approx((5.0 + 12.0) / 2.0)


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


def test_tdec_critics_are_parameter_independent_and_update_with_composed_advantages():
    torch.manual_seed(19)
    np.random.seed(19)
    config = _mappo_config(mappo_variant="tdec")
    trainer = MAPPOTrainer(config, torch.device("cpu"))
    rollout = TDecOnPolicyRollout(config.number_agents, config.state_dim)

    modules = [trainer.global_critic, *trainer.task1_critics, *trainer.task2_critics]
    parameter_sets = [{id(parameter) for parameter in module.parameters()} for module in modules]
    for index, parameters in enumerate(parameter_sets):
        for other in parameter_sets[index + 1:]:
            assert parameters.isdisjoint(other)

    actor_before = [[parameter.detach().clone() for parameter in actor.parameters()] for actor in trainer.actors]
    critic_before = [[parameter.detach().clone() for parameter in module.parameters()] for module in modules]
    rng = np.random.default_rng(19)
    observations = rng.normal(size=(config.number_agents, config.state_dim)).astype(np.float32)
    for step_index in range(6):
        sampled = trainer.act(observations)
        assert sampled.tdec_values is not None
        assert sampled.tdec_values.global_value.shape == (1,)
        assert sampled.tdec_values.task1_values.shape == sampled.tdec_values.task2_values.shape == (config.number_agents,)
        next_observations = rng.normal(size=(config.number_agents, config.state_dim)).astype(np.float32)
        done = step_index in {2, 5}
        if done:
            next_global = np.zeros(1, dtype=np.float32)
            next_task1 = np.zeros(config.number_agents, dtype=np.float32)
            next_task2 = np.zeros(config.number_agents, dtype=np.float32)
        else:
            next_values = trainer.tdec_values(next_observations)
            next_global = next_values.global_value
            next_task1 = next_values.task1_values
            next_task2 = next_values.task2_values
        rollout.append(
            observations=observations,
            rb=sampled.rb,
            mode=sampled.mode,
            power=sampled.power,
            old_log_prob=sampled.log_prob,
            global_values=sampled.tdec_values.global_value,
            task1_values=sampled.tdec_values.task1_values,
            task2_values=sampled.tdec_values.task2_values,
            global_rewards=np.asarray([-0.2 + 0.05 * step_index], dtype=np.float32),
            task1_rewards=np.linspace(-1.0, 0.5, config.number_agents, dtype=np.float32) + 0.03 * step_index,
            task2_rewards=np.linspace(0.4, -0.8, config.number_agents, dtype=np.float32) - 0.02 * step_index,
            done=done,
            global_next_values=next_global,
            task1_next_values=next_task1,
            task2_next_values=next_task2,
            policy_version=trainer.policy_version,
        )
        observations = next_observations

    batch = rollout.consume(config.gamma, config.mappo_gae_lambda, trainer.policy_version)
    expected_advantages = (
        config.global_actor_weight * batch.global_advantages.expand(-1, config.number_agents)
        + batch.task1_advantages
        + batch.task2_advantages
    )
    diagnostics = trainer.update(batch)

    assert diagnostics["mappo_variant"] == "tdec"
    assert diagnostics["policy_version"] == 1
    assert diagnostics["rollout_steps"] == 6
    np.testing.assert_allclose(
        diagnostics["advantage_mean_per_agent_before_normalization"],
        expected_advantages.mean(dim=0).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    for key in (
        "global_critic_loss", "task1_critic_loss", "task2_critic_loss",
        "global_critic_grad_norm", "task1_critic_grad_norm", "task2_critic_grad_norm",
        "global_explained_variance", "task1_explained_variance", "task2_explained_variance",
    ):
        assert np.isfinite(diagnostics[key])
    assert diagnostics["critic_loss"] == pytest.approx(
        diagnostics["global_critic_loss"]
        + diagnostics["task1_critic_loss"]
        + diagnostics["task2_critic_loss"]
    )
    assert diagnostics["critic_loss_aggregate_semantics"] == (
        "mean_over_epochs_of_global_plus_task1_plus_task2_losses"
    )
    assert diagnostics["critic_grad_norm_aggregate_semantics"] == (
        "mean_over_epochs_of_joint_rss_global_task1_task2_grad_l2_before_clipping"
    )
    assert diagnostics["explained_variance_aggregate_semantics"] == (
        "unweighted_mean_of_global_task1_task2_explained_variance"
    )
    assert len(diagnostics["global_task1_advantage_correlation_per_agent"]) == config.number_agents
    assert len(diagnostics["global_task2_advantage_correlation_per_agent"]) == config.number_agents
    assert len(diagnostics["task1_task2_advantage_correlation_per_agent"]) == config.number_agents
    assert trainer.global_value_norm.count.item() == pytest.approx(6.0)
    assert trainer.task1_value_norm.count.item() == pytest.approx(6.0)
    assert trainer.task2_value_norm.count.item() == pytest.approx(6.0)
    for before, actor in zip(actor_before, trainer.actors):
        assert any(not torch.equal(old, new) for old, new in zip(before, actor.parameters()))
    for before, module in zip(critic_before, modules):
        assert any(not torch.equal(old, new) for old, new in zip(before, module.parameters()))
