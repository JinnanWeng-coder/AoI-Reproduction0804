"""Global centralized critic network."""

from __future__ import annotations

from typing import Iterable, List

from .networks import CriticNetwork


class G_CriticNetwork(CriticNetwork):
    def __init__(self, input_dims, hidden_dims: Iterable[int], n_actions, n_agents, critic_lr, device="cpu"):
        hidden = [int(value) for value in hidden_dims]
        if len(hidden) != 3:
            raise ValueError("global critic hidden sizes must have length 3")
        super(CriticNetwork, self).__init__()
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self.fc1 = nn.Linear(int(input_dims) * int(n_agents), hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.norm1 = nn.LayerNorm(hidden[0])
        self.norm2 = nn.LayerNorm(hidden[1])
        self.norm3 = nn.LayerNorm(hidden[2])
        self.action_value = nn.Linear(int(n_actions) * int(n_agents), hidden[1])
        self.q = nn.Linear(hidden[2], 1)
        for layer in (self.fc1, self.fc2, self.fc3):
            bound = 1.0 / np.sqrt(layer.weight.data.size(0))
            layer.weight.data.uniform_(-bound, bound)
            layer.bias.data.uniform_(-bound, bound)
        self.q.weight.data.uniform_(-0.003, 0.003)
        self.q.bias.data.uniform_(-0.003, 0.003)
        bound = 1.0 / np.sqrt(self.action_value.weight.data.size(0))
        self.action_value.weight.data.uniform_(-bound, bound)
        self.action_value.bias.data.uniform_(-bound, bound)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(critic_lr), weight_decay=0.01)
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state, action):
        import torch.nn.functional as F
        state_value = F.relu(self.norm1(self.fc1(state)))
        state_value = F.relu(self.norm2(self.fc2(state_value)))
        action_value = self.action_value(action)
        x = F.relu(state_value + action_value)
        x = F.relu(self.norm3(self.fc3(x)))
        return self.q(x)

