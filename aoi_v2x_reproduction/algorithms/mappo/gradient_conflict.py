"""Read-only actor-gradient diagnostics for task-decomposed MAPPO."""

from __future__ import annotations

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


def objective_actor_gradient_diagnostics(
    actor: torch.nn.Module,
    ratio: torch.Tensor,
    component_advantages: Mapping[str, torch.Tensor],
    clip_param: float,
    tiny: float = 1e-12,
) -> Dict[str, object]:
    """Measure objective gradients without touching ``parameter.grad`` or stepping."""

    if tuple(component_advantages) != OBJECTIVES:
        raise ValueError(f"objective advantages must be ordered as {OBJECTIVES}")
    parameters = tuple(actor.parameters())
    blocks = actor_parameter_blocks(actor)
    losses = {
        name: clipped_policy_loss(ratio, component_advantages[name], clip_param)
        for name in OBJECTIVES
    }
    gradients: Dict[str, Tuple[torch.Tensor, ...]] = {}
    for name in OBJECTIVES:
        raw = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        gradients[name] = tuple(
            torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, raw)
        )

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
        cancellation = torch.zeros((), device=ratio.device, dtype=ratio.dtype)

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
        "objective_grad_norm": objective_norms,
        "objective_policy_loss": {
            name: float(loss.detach().cpu()) for name, loss in losses.items()
        },
        "effective_clip_fraction": effective_clip_fraction,
        "pairs": pairs,
        "block_pairs": block_pairs,
        "cancellation_ratio": float(cancellation.detach().cpu()),
        "cancellation_valid": cancellation_valid,
    }
