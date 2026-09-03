import copy

import torch

from aoi_v2x_reproduction.algorithms.mappo.gradient_conflict import (
    OBJECTIVES,
    actor_parameter_blocks,
    clipped_policy_loss,
    common_scale_component_advantages,
    effective_clipping_mask,
    objective_actor_gradient_diagnostics,
    pair_geometry,
)
from aoi_v2x_reproduction.algorithms.mappo.networks import HybridActor
from aoi_v2x_reproduction.algorithms.mappo.rollout import TDecRolloutBatch
from aoi_v2x_reproduction.algorithms.mappo.trainer import MAPPOTrainer
from aoi_v2x_reproduction.config import resolve_config


def _flatten(gradients):
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def _actor_inputs():
    actor = HybridActor(obs_dim=6, hidden_dims=[8, 4], n_rb=3, n_modes=2)
    observations = torch.linspace(-1.0, 1.0, 48).reshape(8, 6)
    rb = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    mode = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    power = torch.linspace(0.2, 0.8, 8)
    evaluated = actor.evaluate_actions(observations, rb, mode, power)
    ratio = torch.exp(evaluated.log_prob - evaluated.log_prob.detach())
    return actor, ratio


def test_common_scale_components_sum_to_current_normalized_advantage():
    weighted_global = torch.tensor([[1.0, -2.0], [3.0, 1.0], [-1.0, 4.0], [2.0, 0.0]])
    task1 = torch.tensor([[2.0, 1.0], [-2.0, 3.0], [4.0, -1.0], [0.0, 2.0]])
    task2 = torch.tensor([[-3.0, 5.0], [1.0, -2.0], [2.0, 0.0], [4.0, -1.0]])
    components, normalized = common_scale_component_advantages(weighted_global, task1, task2)
    composed = weighted_global + task1 + task2
    expected = (composed - composed.mean(dim=0, keepdim=True)) / (
        composed.std(dim=0, unbiased=False, keepdim=True) + 1e-8
    )
    torch.testing.assert_close(normalized, expected)
    torch.testing.assert_close(sum(components.values()), expected)


def test_component_gradient_sum_matches_composed_at_ratio_one():
    torch.manual_seed(5)
    actor, ratio = _actor_inputs()
    raw = {
        "global": torch.tensor([[1.0], [0.5], [-0.5], [-1.0], [0.2], [0.8], [-0.2], [-0.8]]),
        "task1": torch.tensor([[-0.4], [0.2], [0.7], [-0.5], [1.1], [-1.0], [0.3], [-0.1]]),
        "task2": torch.tensor([[0.3], [-0.9], [0.4], [0.2], [-0.7], [0.6], [-0.2], [0.3]]),
    }
    components, normalized = common_scale_component_advantages(
        raw["global"], raw["task1"], raw["task2"]
    )
    parameters = tuple(actor.parameters())
    component_gradients = []
    for name in OBJECTIVES:
        loss = clipped_policy_loss(ratio, components[name][:, 0], clip_param=0.2)
        component_gradients.append(_flatten(torch.autograd.grad(loss, parameters, retain_graph=True)))
    composed_loss = clipped_policy_loss(ratio, normalized[:, 0], clip_param=0.2)
    composed_gradient = _flatten(torch.autograd.grad(composed_loss, parameters))
    torch.testing.assert_close(sum(component_gradients), composed_gradient, rtol=2e-5, atol=2e-7)


def test_read_only_diagnostic_preserves_parameters_grads_optimizer_and_rng():
    torch.manual_seed(7)
    actor, ratio = _actor_inputs()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    components = {
        "global": torch.linspace(-1.0, 1.0, 8),
        "task1": torch.tensor([1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.8, -0.8]),
        "task2": torch.tensor([-0.3, 0.6, -0.9, 1.2, -1.5, 1.8, -2.1, 2.4]),
    }
    for parameter in actor.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    parameters_before = [parameter.detach().clone() for parameter in actor.parameters()]
    grads_before = [parameter.grad.detach().clone() for parameter in actor.parameters()]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng_before = torch.random.get_rng_state().clone()

    result = objective_actor_gradient_diagnostics(actor, ratio, components, clip_param=0.2)

    assert result["cancellation_valid"] is True
    for before, parameter in zip(parameters_before, actor.parameters()):
        torch.testing.assert_close(parameter, before, rtol=0.0, atol=0.0)
    for before, parameter in zip(grads_before, actor.parameters()):
        torch.testing.assert_close(parameter.grad, before, rtol=0.0, atol=0.0)
    assert optimizer.state_dict() == optimizer_before
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0.0, atol=0.0)


def test_pair_geometry_detects_aligned_conflicting_and_tiny_vectors():
    aligned = pair_geometry(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
    conflicting = pair_geometry(torch.tensor([1.0, 2.0]), torch.tensor([-2.0, -4.0]))
    tiny = pair_geometry(torch.zeros(2), torch.ones(2))
    assert aligned == {"dot": 10.0, "cosine": 1.0, "valid": True, "conflict": False}
    assert conflicting == {"dot": -10.0, "cosine": -1.0, "valid": True, "conflict": True}
    assert tiny == {"dot": 0.0, "cosine": 0.0, "valid": False, "conflict": False}


def test_effective_clip_mask_is_sign_aware():
    ratio = torch.tensor([1.3, 0.7, 1.1, 0.9])
    positive = effective_clipping_mask(ratio, torch.ones(4), clip_param=0.2)
    negative = effective_clipping_mask(ratio, -torch.ones(4), clip_param=0.2)
    torch.testing.assert_close(positive, torch.tensor([True, False, False, False]))
    torch.testing.assert_close(negative, torch.tensor([False, True, False, False]))


def test_actor_parameter_blocks_are_complete_and_disjoint():
    actor = HybridActor(obs_dim=6, hidden_dims=[8, 4], n_rb=3, n_modes=2)
    blocks = actor_parameter_blocks(actor)
    indices = [index for values in blocks.values() for index in values]
    assert set(blocks) == {"trunk", "rb_head", "mode_head", "power_head"}
    assert sorted(indices) == list(range(len(tuple(actor.parameters()))))
    assert len(indices) == len(set(indices))


def _small_tdec_config(enabled: bool):
    return resolve_config(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=77,
        episodes=2,
        steps_per_episode=4,
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        device="cpu",
        mappo_variant="tdec",
        mappo_rollout_episodes=1,
        mappo_ppo_epochs=2,
        mappo_objective_gradient_diagnostics=enabled,
    )


def _tdec_batch(trainer: MAPPOTrainer) -> TDecRolloutBatch:
    samples = 8
    agents = trainer.number_agents
    observations = torch.linspace(
        -1.0, 1.0, samples * agents * trainer.observation_dim
    ).reshape(samples, agents, trainer.observation_dim)
    rb = torch.stack([torch.arange(samples) % 3 for _ in range(agents)], dim=1)
    mode = torch.stack([torch.arange(samples) % 2 for _ in range(agents)], dim=1)
    power = torch.full((samples, agents), 0.5)
    with torch.no_grad():
        old_log_prob = torch.stack([
            actor.evaluate_actions(observations[:, index], rb[:, index], mode[:, index], power[:, index]).log_prob
            for index, actor in enumerate(trainer.actors)
        ], dim=1)
    global_advantages = torch.linspace(-1.0, 1.0, samples).reshape(-1, 1)
    task1_advantages = torch.linspace(-0.7, 1.3, samples * agents).reshape(samples, agents)
    task2_advantages = torch.linspace(1.1, -1.5, samples * agents).reshape(samples, agents)
    zeros_global = torch.zeros(samples, 1)
    zeros_local = torch.zeros(samples, agents)
    return TDecRolloutBatch(
        observations=observations,
        rb=rb,
        mode=mode,
        power=power,
        old_log_prob=old_log_prob,
        old_global_values=zeros_global,
        old_task1_values=zeros_local,
        old_task2_values=zeros_local,
        global_rewards=zeros_global,
        task1_rewards=zeros_local,
        task2_rewards=zeros_local,
        global_next_values=zeros_global,
        task1_next_values=zeros_local,
        task2_next_values=zeros_local,
        dones=torch.zeros(samples),
        global_advantages=global_advantages,
        task1_advantages=task1_advantages,
        task2_advantages=task2_advantages,
        global_returns=global_advantages,
        task1_returns=task1_advantages,
        task2_returns=task2_advantages,
    )


def _all_trainer_parameters(trainer: MAPPOTrainer):
    modules = [trainer.actors, trainer.global_critic, trainer.task1_critics, trainer.task2_critics]
    return [parameter.detach().clone() for module in modules for parameter in module.parameters()]


def test_enabling_diagnostics_does_not_change_tdec_update():
    torch.manual_seed(41)
    without = MAPPOTrainer(_small_tdec_config(False), torch.device("cpu"))
    torch.manual_seed(41)
    with_diagnostics = MAPPOTrainer(_small_tdec_config(True), torch.device("cpu"))
    batch = _tdec_batch(without)

    plain = without.update(batch)
    audited = with_diagnostics.update(batch)

    assert "objective_gradient_diagnostics" not in plain
    objective = audited["objective_gradient_diagnostics"]
    assert objective["training_update_unchanged"] is True
    assert objective["ppo_epochs_recorded"] == [0, 1]
    assert len(objective["records"]) == 10
    for left, right in zip(_all_trainer_parameters(without), _all_trainer_parameters(with_diagnostics)):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
