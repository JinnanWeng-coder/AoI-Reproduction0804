"""Device-explicit actor and local critic networks."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _hidden_pair(hidden: Iterable[int]) -> List[int]:
    values = [int(value) for value in hidden]
    if len(values) != 2:
        raise ValueError("local actor/critic hidden sizes must have length 2")
    return values


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dims, n_actions, actor_lr, device="cpu"):
        super().__init__()
        h1, h2 = _hidden_pair(hidden_dims)
        self.fc1 = nn.Linear(int(input_dims), h1)
        self.fc2 = nn.Linear(h1, h2)
        self.norm1 = nn.LayerNorm(h1)
        self.norm2 = nn.LayerNorm(h2)
        self.mu = nn.Linear(h2, int(n_actions))
        self._init_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(actor_lr))
        self.device = torch.device(device)
        self.to(self.device)

    def _init_weights(self):
        for layer in (self.fc1, self.fc2):
            bound = 1.0 / np.sqrt(layer.weight.data.size(0))
            layer.weight.data.uniform_(-bound, bound)
            layer.bias.data.uniform_(-bound, bound)
        self.mu.weight.data.uniform_(-0.003, 0.003)
        self.mu.bias.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        x = F.relu(self.norm1(self.fc1(state)))
        x = F.relu(self.norm2(self.fc2(x)))
        return torch.tanh(self.mu(x))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dims, n_actions, critic_lr, device="cpu"):
        super().__init__()
        h1, h2 = _hidden_pair(hidden_dims)
        self.fc1 = nn.Linear(int(input_dims), h1)
        self.fc2 = nn.Linear(h1, h2)
        self.norm1 = nn.LayerNorm(h1)
        self.norm2 = nn.LayerNorm(h2)
        self.action_value = nn.Linear(int(n_actions), h2)
        self.q = nn.Linear(h2, 1)
        self._init_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(critic_lr), weight_decay=0.01)
        self.device = torch.device(device)
        self.to(self.device)

    def _init_weights(self):
        for layer in (self.fc1, self.fc2):
            bound = 1.0 / np.sqrt(layer.weight.data.size(0))
            layer.weight.data.uniform_(-bound, bound)
            layer.bias.data.uniform_(-bound, bound)
        self.q.weight.data.uniform_(-0.003, 0.003)
        self.q.bias.data.uniform_(-0.003, 0.003)
        bound = 1.0 / np.sqrt(self.action_value.weight.data.size(0))
        self.action_value.weight.data.uniform_(-bound, bound)
        self.action_value.bias.data.uniform_(-bound, bound)

    def forward(self, state, action):
        state_value = F.relu(self.norm1(self.fc1(state)))
        state_value = self.norm2(self.fc2(state_value))
        action_value = self.action_value(action)
        return self.q(F.relu(state_value + action_value))

