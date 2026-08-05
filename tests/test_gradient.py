import copy

import numpy as np
import torch

from config import resolve_config
from global_critic import Global_Critic
from local_critic import Agent


def _learner(mode="synchronous_joint", diagnostics=False):
    config = resolve_config("paper_faithful", "p05_n04_g25", seed=4, episodes=2, steps_per_episode=3, device="cpu", actor_hidden=[16, 8], local_critic_hidden=[16, 8], global_critic_hidden=[16, 8, 4], batch_size=4, replay_capacity=32, global_update_mode=mode, diagnostics=diagnostics)
    config.device_resolved = "cpu"
    agents = [Agent(config, index) for index in range(config.number_agents)]
    return config, agents, Global_Critic(config, agents)


def _batch(config):
    rng = np.random.default_rng(8)
    states = rng.normal(size=(4, config.state_dim * config.number_agents)).astype(np.float32)
    next_states = rng.normal(size=states.shape).astype(np.float32)
    actions = rng.uniform(-0.5, 0.5, size=(4, 3 * config.number_agents)).astype(np.float32)
    return states, actions, np.zeros(4, dtype=np.float32), np.zeros((4, config.number_agents), dtype=np.float32), np.zeros((4, config.number_agents), dtype=np.float32), next_states, np.zeros(4, dtype=bool)


def test_global_only_gradient_is_nonzero_for_every_actor():
    config, _agents, learner = _learner("synchronous_joint")
    audit = learner.actor_global_gradient_audit(_batch(config)[0])
    assert audit["finite"] is True
    assert all(value > 0 for value in audit["global_gradient_norms"])


def test_legacy_detach_global_gradient_is_zero():
    config, _agents, learner = _learner("legacy_detach")
    audit = learner.actor_global_gradient_audit(_batch(config)[0])
    assert audit["global_gradient_norms"] == [0.0] * config.number_agents


def test_detached_actor_blocks_global_actor_gradient_but_updates_global_critic():
    config, _agents, learner = _learner("detached_actor", diagnostics=True)
    batch = _batch(config)
    before = [parameter.detach().clone() for parameter in learner.global_critic1.parameters()]
    learner.learn(batch)
    result = learner.learn(batch)
    assert any(not torch.equal(old, new) for old, new in zip(before, learner.global_critic1.parameters()))
    gradient = result["actor_gradient_diagnostics"]
    assert gradient["global_contributes_to_actor"] is False
    assert result["global_actor_gradient_norms"] == [0.0] * config.number_agents
    assert any(value > 0.0 for value in gradient["global_grad_l2"])
    assert all(value > 0 for value in result["actor_parameter_deltas"])


def test_external_eval_noise_rng_does_not_advance_environment_numpy_rng():
    config, agents, _learner_instance = _learner("synchronous_joint")
    observation = np.zeros(config.state_dim, dtype=np.float32)
    np.random.seed(12345)
    expected_next = np.random.random()
    np.random.seed(12345)
    agents[0].choose_action(
        observation,
        explore=False,
        noise_std=0.1,
        rng=np.random.default_rng(77),
    )
    assert np.random.random() == expected_next


def test_joint_update_changes_all_actors_and_delays_policy():
    config, agents, learner = _learner("synchronous_joint")
    batch = _batch(config)
    first = learner.learn(batch)
    assert first["actor_loss"] is None
    assert first["local_critic_loss"] == []
    second = learner.learn(batch)
    assert second["actor_loss"] is not None
    assert len(second["local_critic_loss"]) == config.number_agents
    assert len(second["global_actor_gradient_norms"]) == config.number_agents
    assert all(value > 0 for value in second["global_actor_gradient_norms"])
    assert all(value > 0 for value in second["actor_parameter_deltas"])


def test_joint_actor_update_is_invariant_to_optimizer_step_order():
    torch.manual_seed(19)
    config_forward, agents_forward, learner_forward = _learner("synchronous_joint")
    torch.manual_seed(19)
    config_reverse, agents_reverse, learner_reverse = _learner("synchronous_joint")
    states = torch.as_tensor(_batch(config_forward)[0], dtype=torch.float32)
    forward = learner_forward._actor_step(states)
    reverse = learner_reverse._actor_step(states, actor_step_order=reversed(range(config_reverse.number_agents)))
    for left, right in zip(agents_forward, agents_reverse):
        for left_parameter, right_parameter in zip(left.actor.parameters(), right.actor.parameters()):
            assert torch.equal(left_parameter, right_parameter)
    np.testing.assert_array_equal(forward["actor_parameter_deltas"], reverse["actor_parameter_deltas"])


def test_diagnostics_are_finite_complete_and_do_not_change_actor_step():
    torch.manual_seed(29)
    config_off, agents_off, learner_off = _learner("synchronous_joint", diagnostics=False)
    torch.manual_seed(29)
    config_on, agents_on, learner_on = _learner("synchronous_joint", diagnostics=True)
    states = torch.as_tensor(_batch(config_off)[0], dtype=torch.float32)
    without = learner_off._actor_step(states)
    with_diagnostics = learner_on._actor_step(states)
    for left, right in zip(agents_off, agents_on):
        for left_parameter, right_parameter in zip(left.actor.parameters(), right.actor.parameters()):
            assert torch.equal(left_parameter, right_parameter)
    assert without["actor_gradient_diagnostics"] is None
    gradient = with_diagnostics["actor_gradient_diagnostics"]
    assert gradient["finite"] is True
    expected = {
        "task1_grad_l2", "task2_grad_l2", "local_sum_grad_l2", "global_grad_l2",
        "global_to_local_ratio", "global_to_task1_ratio", "global_to_task2_ratio",
        "task1_to_task2_ratio", "global_vs_local_cosine", "global_vs_task1_cosine",
        "global_vs_task2_cosine", "task1_vs_task2_cosine",
    }
    assert expected <= set(gradient)
    for key in expected:
        values = np.asarray(gradient[key], dtype=np.float64)
        assert values.shape == (config_on.number_agents,)
        assert np.all(np.isfinite(values))


def test_global_target_critics_update_on_every_learner_step():
    config, _agents, learner = _learner("synchronous_joint")
    config.tau = 1.0
    batch = _batch(config)
    before = [parameter.detach().clone() for parameter in learner.global_target_critic1.parameters()]
    first = learner.learn(batch)
    assert first["actor_loss"] is None
    assert any(not torch.equal(old, new) for old, new in zip(before, learner.global_target_critic1.parameters()))
    for target, source in zip(learner.global_target_critic1.parameters(), learner.global_critic1.parameters()):
        torch.testing.assert_close(target, source)
    for target, source in zip(learner.global_target_critic2.parameters(), learner.global_critic2.parameters()):
        torch.testing.assert_close(target, source)
