import copy

import numpy as np

from config import resolve_config
from global_critic import Global_Critic
from local_critic import Agent


def _learner(mode="synchronous_joint"):
    config = resolve_config("paper_faithful", "p05_n04_g25", seed=4, episodes=2, steps_per_episode=3, device="cpu", actor_hidden=[16, 8], local_critic_hidden=[16, 8], global_critic_hidden=[16, 8, 4], batch_size=4, replay_capacity=32, global_update_mode=mode)
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


def test_joint_update_changes_all_actors_and_delays_policy():
    config, agents, learner = _learner("synchronous_joint")
    batch = _batch(config)
    first = learner.learn(batch)
    assert first["actor_loss"] is None
    second = learner.learn(batch)
    assert second["actor_loss"] is not None
    assert len(second["global_actor_gradient_norms"]) == config.number_agents
    assert all(value > 0 for value in second["global_actor_gradient_norms"])
    assert all(value > 0 for value in second["actor_parameter_deltas"])

