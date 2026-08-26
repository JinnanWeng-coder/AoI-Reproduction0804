"""Single-use on-policy rollout storage and GAE computation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List

import numpy as np
import torch


@dataclass(frozen=True)
class RolloutBatch:
    observations: torch.Tensor
    rb: torch.Tensor
    mode: torch.Tensor
    power: torch.Tensor
    old_log_prob: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def to(self, device: torch.device) -> "RolloutBatch":
        return RolloutBatch(**{item.name: getattr(self, item.name).to(device) for item in fields(self)})

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True)
class TDecRolloutBatch:
    """Task-decomposed rollout with one global and two per-agent streams."""

    observations: torch.Tensor
    rb: torch.Tensor
    mode: torch.Tensor
    power: torch.Tensor
    old_log_prob: torch.Tensor
    old_global_values: torch.Tensor
    old_task1_values: torch.Tensor
    old_task2_values: torch.Tensor
    global_rewards: torch.Tensor
    task1_rewards: torch.Tensor
    task2_rewards: torch.Tensor
    global_next_values: torch.Tensor
    task1_next_values: torch.Tensor
    task2_next_values: torch.Tensor
    dones: torch.Tensor
    global_advantages: torch.Tensor
    task1_advantages: torch.Tensor
    task2_advantages: torch.Tensor
    global_returns: torch.Tensor
    task1_returns: torch.Tensor
    task2_returns: torch.Tensor

    def to(self, device: torch.device) -> "TDecRolloutBatch":
        return TDecRolloutBatch(**{item.name: getattr(self, item.name).to(device) for item in fields(self)})

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
):
    """Compute per-agent advantages, resetting recursion at episode ends."""

    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("rewards, values, and next_values must share shape [time, agent]")
    if dones.ndim != 1 or dones.shape[0] != rewards.shape[0]:
        raise ValueError("dones must have shape [time]")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.shape[0] - 1, -1, -1):
        mask = 1.0 - dones[index]
        delta = rewards[index] + float(gamma) * mask * next_values[index] - values[index]
        running = delta + float(gamma) * float(gae_lambda) * mask * running
        advantages[index] = running
    return advantages, advantages + values


class OnPolicyRollout:
    """Collect one policy version and consume it exactly once."""

    def __init__(self, number_agents: int, observation_dim: int):
        self.number_agents = int(number_agents)
        self.observation_dim = int(observation_dim)
        self._records: List[Dict[str, np.ndarray]] = []
        self._policy_version = None
        self._terminal_count = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def terminal_count(self) -> int:
        return int(self._terminal_count)

    def append(
        self,
        observations,
        rb,
        mode,
        power,
        old_log_prob,
        values,
        rewards,
        done: bool,
        next_values,
        policy_version: int,
    ) -> None:
        version = int(policy_version)
        if self._policy_version is None:
            self._policy_version = version
        elif self._policy_version != version:
            raise RuntimeError("rollout cannot mix data from different policy versions")

        record = {
            "observations": np.asarray(observations, dtype=np.float32),
            "rb": np.asarray(rb, dtype=np.int64),
            "mode": np.asarray(mode, dtype=np.int64),
            "power": np.asarray(power, dtype=np.float32),
            "old_log_prob": np.asarray(old_log_prob, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_values": np.asarray(next_values, dtype=np.float32),
            "done": np.asarray(bool(done), dtype=np.float32),
        }
        expected = {
            "observations": (self.number_agents, self.observation_dim),
            "rb": (self.number_agents,),
            "mode": (self.number_agents,),
            "power": (self.number_agents,),
            "old_log_prob": (self.number_agents,),
            "values": (self.number_agents,),
            "rewards": (self.number_agents,),
            "next_values": (self.number_agents,),
            "done": (),
        }
        for name, shape in expected.items():
            if record[name].shape != shape:
                raise ValueError(f"rollout {name} has shape {record[name].shape}, expected {shape}")
            if name not in {"rb", "mode"} and not np.all(np.isfinite(record[name])):
                raise FloatingPointError(f"non-finite rollout field: {name}")
        self._records.append(record)
        self._terminal_count += int(bool(done))

    def consume(self, gamma: float, gae_lambda: float, expected_policy_version: int) -> RolloutBatch:
        if not self._records:
            raise RuntimeError("cannot consume an empty rollout")
        if self._policy_version != int(expected_policy_version):
            raise RuntimeError("rollout policy version is stale")
        stacked = {
            name: np.stack([record[name] for record in self._records])
            for name in self._records[0]
        }
        observations = torch.from_numpy(stacked["observations"])
        rewards = torch.from_numpy(stacked["rewards"])
        values = torch.from_numpy(stacked["values"])
        next_values = torch.from_numpy(stacked["next_values"])
        dones = torch.from_numpy(stacked["done"])
        advantages, returns = compute_gae(rewards, values, next_values, dones, gamma, gae_lambda)
        batch = RolloutBatch(
            observations=observations,
            rb=torch.from_numpy(stacked["rb"]),
            mode=torch.from_numpy(stacked["mode"]),
            power=torch.from_numpy(stacked["power"]),
            old_log_prob=torch.from_numpy(stacked["old_log_prob"]),
            old_values=values,
            rewards=rewards,
            dones=dones,
            advantages=advantages,
            returns=returns,
        )
        self._records.clear()
        self._policy_version = None
        self._terminal_count = 0
        return batch


class TDecOnPolicyRollout:
    """Single-use rollout retaining raw global, task-1, and task-2 value streams."""

    def __init__(self, number_agents: int, observation_dim: int):
        self.number_agents = int(number_agents)
        self.observation_dim = int(observation_dim)
        self._records: List[Dict[str, np.ndarray]] = []
        self._policy_version = None
        self._terminal_count = 0

    def __len__(self) -> int:
        return len(self._records)

    @property
    def terminal_count(self) -> int:
        return int(self._terminal_count)

    def append(
        self,
        observations,
        rb,
        mode,
        power,
        old_log_prob,
        global_values,
        task1_values,
        task2_values,
        global_rewards,
        task1_rewards,
        task2_rewards,
        done: bool,
        global_next_values,
        task1_next_values,
        task2_next_values,
        policy_version: int,
    ) -> None:
        version = int(policy_version)
        if self._policy_version is None:
            self._policy_version = version
        elif self._policy_version != version:
            raise RuntimeError("rollout cannot mix data from different policy versions")

        record = {
            "observations": np.asarray(observations, dtype=np.float32),
            "rb": np.asarray(rb, dtype=np.int64),
            "mode": np.asarray(mode, dtype=np.int64),
            "power": np.asarray(power, dtype=np.float32),
            "old_log_prob": np.asarray(old_log_prob, dtype=np.float32),
            "global_values": np.asarray(global_values, dtype=np.float32),
            "task1_values": np.asarray(task1_values, dtype=np.float32),
            "task2_values": np.asarray(task2_values, dtype=np.float32),
            "global_rewards": np.asarray(global_rewards, dtype=np.float32),
            "task1_rewards": np.asarray(task1_rewards, dtype=np.float32),
            "task2_rewards": np.asarray(task2_rewards, dtype=np.float32),
            "global_next_values": np.asarray(global_next_values, dtype=np.float32),
            "task1_next_values": np.asarray(task1_next_values, dtype=np.float32),
            "task2_next_values": np.asarray(task2_next_values, dtype=np.float32),
            "done": np.asarray(bool(done), dtype=np.float32),
        }
        expected = {
            "observations": (self.number_agents, self.observation_dim),
            "rb": (self.number_agents,),
            "mode": (self.number_agents,),
            "power": (self.number_agents,),
            "old_log_prob": (self.number_agents,),
            "global_values": (1,),
            "task1_values": (self.number_agents,),
            "task2_values": (self.number_agents,),
            "global_rewards": (1,),
            "task1_rewards": (self.number_agents,),
            "task2_rewards": (self.number_agents,),
            "global_next_values": (1,),
            "task1_next_values": (self.number_agents,),
            "task2_next_values": (self.number_agents,),
            "done": (),
        }
        for name, shape in expected.items():
            if record[name].shape != shape:
                raise ValueError(f"TDec rollout {name} has shape {record[name].shape}, expected {shape}")
            if name not in {"rb", "mode"} and not np.all(np.isfinite(record[name])):
                raise FloatingPointError(f"non-finite TDec rollout field: {name}")
        self._records.append(record)
        self._terminal_count += int(bool(done))

    def consume(self, gamma: float, gae_lambda: float, expected_policy_version: int) -> TDecRolloutBatch:
        if not self._records:
            raise RuntimeError("cannot consume an empty rollout")
        if self._policy_version != int(expected_policy_version):
            raise RuntimeError("rollout policy version is stale")
        stacked = {
            name: np.stack([record[name] for record in self._records])
            for name in self._records[0]
        }
        dones = torch.from_numpy(stacked["done"])
        global_rewards = torch.from_numpy(stacked["global_rewards"])
        task1_rewards = torch.from_numpy(stacked["task1_rewards"])
        task2_rewards = torch.from_numpy(stacked["task2_rewards"])
        global_values = torch.from_numpy(stacked["global_values"])
        task1_values = torch.from_numpy(stacked["task1_values"])
        task2_values = torch.from_numpy(stacked["task2_values"])
        global_next_values = torch.from_numpy(stacked["global_next_values"])
        task1_next_values = torch.from_numpy(stacked["task1_next_values"])
        task2_next_values = torch.from_numpy(stacked["task2_next_values"])
        global_advantages, global_returns = compute_gae(
            global_rewards, global_values, global_next_values, dones, gamma, gae_lambda
        )
        task1_advantages, task1_returns = compute_gae(
            task1_rewards, task1_values, task1_next_values, dones, gamma, gae_lambda
        )
        task2_advantages, task2_returns = compute_gae(
            task2_rewards, task2_values, task2_next_values, dones, gamma, gae_lambda
        )
        batch = TDecRolloutBatch(
            observations=torch.from_numpy(stacked["observations"]),
            rb=torch.from_numpy(stacked["rb"]),
            mode=torch.from_numpy(stacked["mode"]),
            power=torch.from_numpy(stacked["power"]),
            old_log_prob=torch.from_numpy(stacked["old_log_prob"]),
            old_global_values=global_values,
            old_task1_values=task1_values,
            old_task2_values=task2_values,
            global_rewards=global_rewards,
            task1_rewards=task1_rewards,
            task2_rewards=task2_rewards,
            global_next_values=global_next_values,
            task1_next_values=task1_next_values,
            task2_next_values=task2_next_values,
            dones=dones,
            global_advantages=global_advantages,
            task1_advantages=task1_advantages,
            task2_advantages=task2_advantages,
            global_returns=global_returns,
            task1_returns=task1_returns,
            task2_returns=task2_returns,
        )
        self._records.clear()
        self._policy_version = None
        self._terminal_count = 0
        return batch
