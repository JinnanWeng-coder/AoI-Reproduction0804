"""Networks and native hybrid-action distributions used by MAPPO."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical


def _hidden_pair(hidden: Iterable[int]) -> Tuple[int, int]:
    values = tuple(int(value) for value in hidden)
    if len(values) != 2 or any(value < 1 for value in values):
        raise ValueError("MAPPO actor hidden sizes must contain two positive values")
    return values


def _hidden_triplet(hidden: Iterable[int]) -> Tuple[int, int, int]:
    values = tuple(int(value) for value in hidden)
    if len(values) != 3 or any(value < 1 for value in values):
        raise ValueError("MAPPO critic hidden sizes must contain three positive values")
    return values


@dataclass(frozen=True)
class HybridSample:
    rb: torch.Tensor
    mode: torch.Tensor
    power: torch.Tensor
    log_prob: torch.Tensor
    entropy_rb: torch.Tensor
    entropy_mode: torch.Tensor
    entropy_power: torch.Tensor


class HybridActor(nn.Module):
    """Local-observation actor with categorical RB/mode and Beta power heads."""

    def __init__(self, obs_dim: int, hidden_dims, n_rb: int, n_modes: int):
        super().__init__()
        h1, h2 = _hidden_pair(hidden_dims)
        self.n_rb = int(n_rb)
        self.n_modes = int(n_modes)
        self.fc1 = nn.Linear(int(obs_dim), h1)
        self.fc2 = nn.Linear(h1, h2)
        self.norm1 = nn.LayerNorm(h1)
        self.norm2 = nn.LayerNorm(h2)
        self.rb_head = nn.Linear(h2, self.n_rb)
        self.mode_head = nn.Linear(h2, self.n_modes)
        self.power_head = nn.Linear(h2, 2)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in (self.fc1, self.fc2):
            nn.init.orthogonal_(layer.weight, gain=sqrt(2.0))
            nn.init.zeros_(layer.bias)
        for layer in (self.rb_head, self.mode_head, self.power_head):
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.zeros_(layer.bias)

    def _distributions(self, observations: torch.Tensor):
        features = F.relu(self.norm1(self.fc1(observations)))
        features = F.relu(self.norm2(self.fc2(features)))
        rb_dist = Categorical(logits=self.rb_head(features))
        mode_dist = Categorical(logits=self.mode_head(features))
        raw_power = self.power_head(features)
        # Parameters strictly above one avoid singular density at either
        # physical power boundary while retaining the entire open interval.
        alpha_beta = F.softplus(raw_power) + 1.0
        power_dist = Beta(alpha_beta[..., 0], alpha_beta[..., 1])
        return rb_dist, mode_dist, power_dist

    def sample(self, observations: torch.Tensor, deterministic: bool = False) -> HybridSample:
        rb_dist, mode_dist, power_dist = self._distributions(observations)
        if deterministic:
            rb = torch.argmax(rb_dist.logits, dim=-1)
            mode = torch.argmax(mode_dist.logits, dim=-1)
            power = power_dist.mean
        else:
            rb = rb_dist.sample()
            mode = mode_dist.sample()
            power = power_dist.sample()
        if not torch.all((power > 0.0) & (power < 1.0)):
            raise FloatingPointError("Beta policy produced a boundary power action")
        log_prob = rb_dist.log_prob(rb) + mode_dist.log_prob(mode) + power_dist.log_prob(power)
        return HybridSample(
            rb=rb,
            mode=mode,
            power=power,
            log_prob=log_prob,
            entropy_rb=rb_dist.entropy(),
            entropy_mode=mode_dist.entropy(),
            entropy_power=power_dist.entropy(),
        )

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        rb: torch.Tensor,
        mode: torch.Tensor,
        power: torch.Tensor,
    ) -> HybridSample:
        if not torch.all((power > 0.0) & (power < 1.0)):
            raise ValueError("stored Beta actions must be strictly inside (0, 1)")
        rb_dist, mode_dist, power_dist = self._distributions(observations)
        log_prob = rb_dist.log_prob(rb) + mode_dist.log_prob(mode) + power_dist.log_prob(power)
        return HybridSample(
            rb=rb,
            mode=mode,
            power=power,
            log_prob=log_prob,
            entropy_rb=rb_dist.entropy(),
            entropy_mode=mode_dist.entropy(),
            entropy_power=power_dist.entropy(),
        )


class CentralValueCritic(nn.Module):
    """Centralized state-value network with one return estimate per agent."""

    def __init__(self, joint_obs_dim: int, hidden_dims, number_agents: int):
        super().__init__()
        h1, h2, h3 = _hidden_triplet(hidden_dims)
        self.fc1 = nn.Linear(int(joint_obs_dim), h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.norm1 = nn.LayerNorm(h1)
        self.norm2 = nn.LayerNorm(h2)
        self.norm3 = nn.LayerNorm(h3)
        self.value = nn.Linear(h3, int(number_agents))
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.orthogonal_(layer.weight, gain=sqrt(2.0))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, joint_observations: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.fc1(joint_observations)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.relu(self.norm3(self.fc3(x)))
        return self.value(x)


class RunningValueNorm(nn.Module):
    """Per-agent running return scale used only by the centralized critic."""

    def __init__(self, number_agents: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(int(number_agents), dtype=torch.float64))
        self.register_buffer("m2", torch.zeros(int(number_agents), dtype=torch.float64))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        batch = values.detach().to(dtype=torch.float64)
        if batch.ndim != 2 or batch.shape[1] != self.mean.numel() or batch.shape[0] < 1:
            raise ValueError("value-normalization input must have shape [batch, agent]")
        batch_count = float(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean.add_(delta * (batch_count / total))
        self.m2.add_(batch_m2 + delta.square() * self.count * batch_count / total)
        self.count.copy_(total)

    def std(self, dtype: torch.dtype) -> torch.Tensor:
        denominator = torch.clamp(self.count - 1.0, min=1.0)
        variance = self.m2 / denominator
        return torch.sqrt(torch.clamp(variance, min=self.epsilon)).to(dtype=dtype)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(dtype=values.dtype)) / self.std(values.dtype)
