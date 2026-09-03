import copy
import random

import numpy as np
import torch

from aoi_v2x_reproduction.algorithms.mappo.gradient_conflict import (
    OBJECTIVES,
    actor_parameter_blocks,
    clipped_policy_loss,
    common_scale_component_advantages,
    effective_clipping_mask,
    objective_actor_gradient_diagnostics,
    pcgrad_project_objective_gradients,
    pcgrad_projection_seed,
    pair_geometry,
    sum_objective_gradients,
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


def _manual_objective_gradients(actor, scales):
    return {
        name: tuple(torch.full_like(parameter, float(scale)) for parameter in actor.parameters())
        for name, scale in zip(OBJECTIVES, scales)
    }


def test_pcgrad_no_conflict_degenerates_exactly_to_separate_gradient_sum():
    actor = HybridActor(obs_dim=6, hidden_dims=[8, 4], n_rb=3, n_modes=2)
    original = _manual_objective_gradients(actor, (1.0, 2.0, 3.0))
    projected, audit = pcgrad_project_objective_gradients(actor, original, projection_seed=19)
    assert audit["projection_count"] == {"global": 0, "task1": 0, "task2": 0}
    assert audit["aggregate_projection_magnitude"] == 0.0
    for name in OBJECTIVES:
        for expected, actual in zip(original[name], projected[name]):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for expected, actual in zip(sum_objective_gradients(original), sum_objective_gradients(projected)):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_pcgrad_projects_all_three_objectives_and_reports_finite_geometry():
    actor = HybridActor(obs_dim=6, hidden_dims=[8, 4], n_rb=3, n_modes=2)
    parameters = tuple(actor.parameters())
    global_grad = tuple(torch.ones_like(parameter) for parameter in parameters)
    task1_grad = tuple(-torch.ones_like(parameter) for parameter in parameters)
    task2_grad = tuple(
        torch.full_like(parameter, 0.5 if index % 2 == 0 else -0.25)
        for index, parameter in enumerate(parameters)
    )
    original = {"global": global_grad, "task1": task1_grad, "task2": task2_grad}
    projected, audit = pcgrad_project_objective_gradients(actor, original, projection_seed=23)

    assert set(projected) == set(OBJECTIVES)
    for name in OBJECTIVES:
        assert set(audit["projection_order"][name]) == set(OBJECTIVES) - {name}
    assert sum(audit["projection_count"].values()) >= 2
    assert audit["aggregate_projection_magnitude"] > 0.0
    for phase in ("before", "after"):
        assert np.isfinite(audit[phase]["cancellation_ratio"])
        assert all(np.isfinite(value) for value in audit[phase]["objective_grad_norm"].values())
        assert all(np.isfinite(value["cosine"]) for value in audit[phase]["pairs"].values())
    assert all(torch.isfinite(value).all() for values in projected.values() for value in values)


def test_pcgrad_projection_order_rng_is_reproducible_and_isolated():
    actor = HybridActor(obs_dim=6, hidden_dims=[8, 4], n_rb=3, n_modes=2)
    gradients = _manual_objective_gradients(actor, (1.0, -1.0, 0.5))
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    seed = pcgrad_projection_seed(8, 11, 9, 4)
    first, first_audit = pcgrad_project_objective_gradients(actor, gradients, seed)
    second, second_audit = pcgrad_project_objective_gradients(actor, gradients, seed)

    assert first_audit["projection_order"] == second_audit["projection_order"]
    for name in OBJECTIVES:
        for left, right in zip(first[name], second[name]):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_before, rtol=0.0, atol=0.0)


def _small_tdec_config(
    enabled: bool,
    actor_update_mode: str = "composed_clip",
    ppo_epochs: int = 2,
):
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
        mappo_ppo_epochs=ppo_epochs,
        mappo_objective_gradient_diagnostics=enabled,
        mappo_actor_update_mode=actor_update_mode,
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


def test_phase2_actor_update_modes_are_finite_and_record_pcgrad_projection():
    results = {}
    for mode in ("composed_clip", "separate_sum_clip", "pcgrad"):
        torch.manual_seed(211)
        trainer = MAPPOTrainer(_small_tdec_config(True, mode), torch.device("cpu"))
        diagnostics = trainer.update(_tdec_batch(trainer))
        assert diagnostics["mappo_actor_update_mode"] == mode
        values = [
            *diagnostics["actor_loss_per_agent"],
            *diagnostics["actor_grad_norm_per_agent"],
            diagnostics["critic_loss"],
        ]
        assert np.isfinite(values).all()
        records = diagnostics["objective_gradient_diagnostics"]["records"]
        assert len(records) == 10
        if mode == "pcgrad":
            assert all(record["pcgrad_projection"]["schema_version"] == "mappo_pcgrad_projection_v1" for record in records)
        else:
            assert all("pcgrad_projection" not in record for record in records)
        results[mode] = _all_trainer_parameters(trainer)

    assert any(
        not torch.equal(left, right)
        for left, right in zip(results["composed_clip"], results["separate_sum_clip"])
    )


def test_separate_sum_matches_composed_update_before_ratio_can_leave_one():
    torch.manual_seed(307)
    composed = MAPPOTrainer(
        _small_tdec_config(False, "composed_clip", ppo_epochs=1), torch.device("cpu")
    )
    torch.manual_seed(307)
    separate = MAPPOTrainer(
        _small_tdec_config(False, "separate_sum_clip", ppo_epochs=1), torch.device("cpu")
    )
    batch = _tdec_batch(composed)
    composed.update(batch)
    separate.update(batch)
    for left, right in zip(_all_trainer_parameters(composed), _all_trainer_parameters(separate)):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=2e-7)
