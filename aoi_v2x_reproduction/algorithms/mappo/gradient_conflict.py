"""Objective-wise actor gradients and diagnostics for task-decomposed MAPPO."""

from __future__ import annotations

import random
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch


OBJECTIVES = ("global", "task1", "task2")
OBJECTIVE_PAIRS = (
    ("global", "task1"),
    ("global", "task2"),
    ("task1", "task2"),
)
PARAMETER_BLOCKS = ("trunk", "rb_head", "mode_head", "power_head")


def common_scale_component_advantages(
    weighted_global: torch.Tensor,
    task1: torch.Tensor,
    task2: torch.Tensor,
    epsilon: float = 1e-8,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Center each reward stream but retain the composed advantage scale.

    The returned components sum to the normalized composed advantage (up to
    floating-point roundoff).  In particular, each component is *not*
    independently standardized, which would silently change the objective
    weights.
    """

    if weighted_global.shape != task1.shape or task1.shape != task2.shape:
        raise ValueError("TDec component advantages must have identical [sample, agent] shapes")
    if weighted_global.ndim != 2 or weighted_global.shape[0] < 2:
        raise ValueError("TDec component advantages require at least two samples")
    raw = {"global": weighted_global, "task1": task1, "task2": task2}
    composed = weighted_global + task1 + task2
    denominator = composed.std(dim=0, unbiased=False, keepdim=True) + float(epsilon)
    components = {
        name: (value - value.mean(dim=0, keepdim=True)) / denominator
        for name, value in raw.items()
    }
    normalized_composed = (
        composed - composed.mean(dim=0, keepdim=True)
    ) / denominator
    component_sum = sum(components.values())
    if not torch.allclose(component_sum, normalized_composed, rtol=2e-5, atol=2e-6):
        raise RuntimeError("common-scale objective advantages do not recover composed advantage")
    return components, normalized_composed


def effective_clipping_mask(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    clip_param: float,
) -> torch.Tensor:
    """Whether PPO's clipped surrogate branch strictly limits this objective."""

    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
    return clipped < unclipped


def clipped_policy_loss(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    clip_param: float,
) -> torch.Tensor:
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
    return -torch.minimum(unclipped, clipped).mean()


def actor_parameter_blocks(actor: torch.nn.Module) -> Dict[str, Tuple[int, ...]]:
    """Return a complete, non-overlapping partition of actor parameters."""

    blocks = {name: [] for name in PARAMETER_BLOCKS}
    for index, (name, _parameter) in enumerate(actor.named_parameters()):
        prefix = name.split(".", 1)[0]
        block = prefix if prefix in {"rb_head", "mode_head", "power_head"} else "trunk"
        blocks[block].append(index)
    flattened = [index for values in blocks.values() for index in values]
    expected = list(range(len(tuple(actor.parameters()))))
    if sorted(flattened) != expected or len(flattened) != len(set(flattened)):
        raise RuntimeError("actor parameter blocks must cover each parameter exactly once")
    if any(not blocks[name] for name in PARAMETER_BLOCKS):
        raise RuntimeError("actor parameter block is empty")
    return {name: tuple(values) for name, values in blocks.items()}


def pair_geometry(
    first: torch.Tensor,
    second: torch.Tensor,
    tiny: float = 1e-12,
) -> Dict[str, object]:
    """Return finite geometry, explicitly marking tiny-gradient comparisons."""

    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    first_norm = torch.linalg.vector_norm(first_flat)
    second_norm = torch.linalg.vector_norm(second_flat)
    dot = torch.dot(first_flat, second_flat)
    valid = bool((first_norm > tiny).item() and (second_norm > tiny).item())
    cosine = dot / (first_norm * second_norm) if valid else torch.zeros_like(dot)
    values = (first_norm, second_norm, dot, cosine)
    if not all(bool(torch.isfinite(value).item()) for value in values):
        raise FloatingPointError("non-finite objective-gradient geometry")
    return {
        "dot": float(dot.detach().cpu()),
        "cosine": float(cosine.detach().cpu()),
        "valid": valid,
        "conflict": bool(valid and dot.item() < 0.0),
    }


def _flatten_gradients(
    gradients: Sequence[torch.Tensor],
    indices: Iterable[int] | None = None,
) -> torch.Tensor:
    selected = range(len(gradients)) if indices is None else indices
    values = [gradients[index].reshape(-1) for index in selected]
    if not values:
        raise ValueError("gradient vector cannot be empty")
    return torch.cat(values)


def objective_policy_losses(
    ratio: torch.Tensor,
    component_advantages: Mapping[str, torch.Tensor],
    clip_param: float,
) -> Dict[str, torch.Tensor]:
    """Build one clipped surrogate loss per objective using one joint ratio."""

    if tuple(component_advantages) != OBJECTIVES:
        raise ValueError(f"objective advantages must be ordered as {OBJECTIVES}")
    return {
        name: clipped_policy_loss(ratio, component_advantages[name], clip_param)
        for name in OBJECTIVES
    }


def objective_policy_gradients(
    losses: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    retain_graph: bool = True,
) -> Dict[str, Tuple[torch.Tensor, ...]]:
    """Differentiate objective losses without writing ``parameter.grad``."""

    if tuple(losses) != OBJECTIVES:
        raise ValueError(f"objective losses must be ordered as {OBJECTIVES}")
    parameter_tuple = tuple(parameters)
    gradients: Dict[str, Tuple[torch.Tensor, ...]] = {}
    for objective_index, name in enumerate(OBJECTIVES):
        keep_graph = retain_graph or objective_index < len(OBJECTIVES) - 1
        raw = torch.autograd.grad(
            losses[name],
            parameter_tuple,
            retain_graph=keep_graph,
            create_graph=False,
            allow_unused=True,
        )
        gradients[name] = tuple(
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameter_tuple, raw)
        )
    return gradients


def sum_objective_gradients(
    gradients: Mapping[str, Sequence[torch.Tensor]],
) -> Tuple[torch.Tensor, ...]:
    """Sum objective gradients, preserving separate-surrogate scale."""

    if tuple(gradients) != OBJECTIVES:
        raise ValueError(f"objective gradients must be ordered as {OBJECTIVES}")
    widths = {len(tuple(gradients[name])) for name in OBJECTIVES}
    if len(widths) != 1 or not widths or next(iter(widths)) == 0:
        raise ValueError("objective gradients must share a non-empty parameter layout")
    return tuple(
        sum(gradients[name][index] for name in OBJECTIVES)
        for index in range(next(iter(widths)))
    )


def gradient_geometry_diagnostics(
    actor: torch.nn.Module,
    gradients: Mapping[str, Sequence[torch.Tensor]],
    tiny: float = 1e-12,
) -> Dict[str, object]:
    """Summarize norms, pair geometry, blocks, and cancellation."""

    if tuple(gradients) != OBJECTIVES:
        raise ValueError(f"objective gradients must be ordered as {OBJECTIVES}")
    blocks = actor_parameter_blocks(actor)
    vectors = {name: _flatten_gradients(gradients[name]) for name in OBJECTIVES}
    objective_norms = {
        name: float(torch.linalg.vector_norm(vector).detach().cpu())
        for name, vector in vectors.items()
    }
    if not all(torch.isfinite(torch.as_tensor(value)) for value in objective_norms.values()):
        raise FloatingPointError("non-finite objective-gradient norm")

    pairs: Dict[str, Dict[str, object]] = {}
    block_pairs: Dict[str, Dict[str, Dict[str, object]]] = {
        block: {} for block in PARAMETER_BLOCKS
    }
    for first, second in OBJECTIVE_PAIRS:
        key = f"{first}_{second}"
        pairs[key] = pair_geometry(vectors[first], vectors[second], tiny=tiny)
        for block, indices in blocks.items():
            first_block = _flatten_gradients(gradients[first], indices)
            second_block = _flatten_gradients(gradients[second], indices)
            block_pairs[block][key] = pair_geometry(first_block, second_block, tiny=tiny)

    norm_sum = sum(torch.linalg.vector_norm(vectors[name]) for name in OBJECTIVES)
    aggregate = sum(vectors.values())
    cancellation_valid = bool((norm_sum > tiny).item())
    if cancellation_valid:
        cancellation = 1.0 - torch.linalg.vector_norm(aggregate) / norm_sum
        cancellation = torch.clamp(cancellation, min=0.0, max=1.0)
    else:
        cancellation = torch.zeros((), device=aggregate.device, dtype=aggregate.dtype)
    aggregate_norm = torch.linalg.vector_norm(aggregate)
    if not bool(torch.isfinite(aggregate_norm).item()):
        raise FloatingPointError("non-finite aggregate objective-gradient norm")
    return {
        "objective_grad_norm": objective_norms,
        "aggregate_grad_norm": float(aggregate_norm.detach().cpu()),
        "pairs": pairs,
        "block_pairs": block_pairs,
        "cancellation_ratio": float(cancellation.detach().cpu()),
        "cancellation_valid": cancellation_valid,
    }


def pcgrad_projection_seed(
    training_seed: int,
    update_index: int,
    ppo_epoch: int,
    agent_index: int,
) -> int:
    """Derive a stable local seed without touching any process RNG state."""

    values = (training_seed, update_index, ppo_epoch, agent_index)
    if any(int(value) < 0 for value in values):
        raise ValueError("PCGrad seed coordinates must be non-negative")
    mask = (1 << 64) - 1
    result = (int(training_seed) + 0x9E3779B97F4A7C15) & mask
    for value, constant in zip(
        (update_index, ppo_epoch, agent_index),
        (0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0xD6E8FEB86659FD93),
    ):
        result = (result ^ ((int(value) + 1) * constant)) & mask
        result = ((result << 27) | (result >> 37)) & mask
    return result


def pcgrad_project_objective_gradients(
    actor: torch.nn.Module,
    gradients: Mapping[str, Sequence[torch.Tensor]],
    projection_seed: int,
    tiny: float = 1e-12,
) -> Tuple[Dict[str, Tuple[torch.Tensor, ...]], Dict[str, object]]:
    """Apply full three-objective PCGrad with isolated reproducible shuffles.

    Every objective is an anchor and is checked against both other original
    objective gradients.  The independently projected anchors are then summed,
    matching the scale of ``separate_sum_clip`` when no conflict is present.
    """

    if tuple(gradients) != OBJECTIVES:
        raise ValueError(f"objective gradients must be ordered as {OBJECTIVES}")
    originals = {
        name: tuple(value.detach().clone() for value in gradients[name])
        for name in OBJECTIVES
    }
    widths = {len(values) for values in originals.values()}
    if len(widths) != 1 or not widths or next(iter(widths)) == 0:
        raise ValueError("objective gradients must share a non-empty parameter layout")
    rng = random.Random(int(projection_seed))
    projected: Dict[str, Tuple[torch.Tensor, ...]] = {}
    orders: Dict[str, list[str]] = {}
    projection_counts: Dict[str, int] = {}
    projection_magnitudes: Dict[str, float] = {}
    relative_projection_magnitudes: Dict[str, float] = {}
    minimum_squared_norm = float(tiny) ** 2

    for name in OBJECTIVES:
        current = [value.clone() for value in originals[name]]
        other_names = [other for other in OBJECTIVES if other != name]
        rng.shuffle(other_names)
        orders[name] = list(other_names)
        count = 0
        for other in other_names:
            reference = originals[other]
            dot = sum((left * right).sum() for left, right in zip(current, reference))
            denominator = sum(value.square().sum() for value in reference)
            if not bool(torch.isfinite(dot).item() and torch.isfinite(denominator).item()):
                raise FloatingPointError("non-finite PCGrad projection geometry")
            if bool((dot < 0.0).item()) and bool((denominator > minimum_squared_norm).item()):
                coefficient = dot / denominator
                current = [left - coefficient * right for left, right in zip(current, reference)]
                count += 1
        projected[name] = tuple(current)
        difference = _flatten_gradients(tuple(
            left - right for left, right in zip(projected[name], originals[name])
        ))
        magnitude = torch.linalg.vector_norm(difference)
        original_norm = torch.linalg.vector_norm(_flatten_gradients(originals[name]))
        relative = magnitude / original_norm if bool((original_norm > tiny).item()) else torch.zeros_like(magnitude)
        if not bool(torch.isfinite(magnitude).item() and torch.isfinite(relative).item()):
            raise FloatingPointError("non-finite PCGrad projection magnitude")
        projection_counts[name] = count
        projection_magnitudes[name] = float(magnitude.detach().cpu())
        relative_projection_magnitudes[name] = float(relative.detach().cpu())

    before = gradient_geometry_diagnostics(actor, originals, tiny=tiny)
    after = gradient_geometry_diagnostics(actor, projected, tiny=tiny)
    original_sum = _flatten_gradients(sum_objective_gradients(originals))
    projected_sum = _flatten_gradients(sum_objective_gradients(projected))
    aggregate_projection = torch.linalg.vector_norm(projected_sum - original_sum)
    if not bool(torch.isfinite(aggregate_projection).item()):
        raise FloatingPointError("non-finite aggregate PCGrad projection magnitude")
    return projected, {
        "schema_version": "mappo_pcgrad_projection_v1",
        "projection_seed": int(projection_seed),
        "projection_order": orders,
        "projection_count": projection_counts,
        "before": before,
        "after": after,
        "projection_magnitude": projection_magnitudes,
        "relative_projection_magnitude": relative_projection_magnitudes,
        "aggregate_projection_magnitude": float(aggregate_projection.detach().cpu()),
    }


def objective_actor_gradient_diagnostics(
    actor: torch.nn.Module,
    ratio: torch.Tensor,
    component_advantages: Mapping[str, torch.Tensor],
    clip_param: float,
    tiny: float = 1e-12,
) -> Dict[str, object]:
    """Measure objective gradients without touching ``parameter.grad`` or stepping."""

    parameters = tuple(actor.parameters())
    losses = objective_policy_losses(ratio, component_advantages, clip_param)
    gradients = objective_policy_gradients(losses, parameters, retain_graph=True)
    geometry = gradient_geometry_diagnostics(actor, gradients, tiny=tiny)

    effective_clip_fraction = {
        name: float(
            effective_clipping_mask(ratio, component_advantages[name], clip_param)
            .to(dtype=torch.float32)
            .mean()
            .detach()
            .cpu()
        )
        for name in OBJECTIVES
    }
    return {
        **geometry,
        "objective_policy_loss": {
            name: float(loss.detach().cpu()) for name, loss in losses.items()
        },
        "effective_clip_fraction": effective_clip_fraction,
    }
