"""Per-agent actor and Algorithm 1/2 local critics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, config, agent_index: int):
        self.config = config
        self.agent_index = int(agent_index)
        self.device = torch.device(config.device_resolved if hasattr(config, "device_resolved") else config.device)
        self.actor = ActorNetwork(config.state_dim, config.actor_hidden, config.action_dim, config.actor_lr, self.device)
        self.target_actor = ActorNetwork(config.state_dim, config.actor_hidden, config.action_dim, config.actor_lr, self.device)
        if config.algorithm == "modified_maddpg":
            self.critic = CriticNetwork(config.state_dim, config.local_critic_hidden, config.action_dim, config.critic_lr, self.device)
            self.target_critic = CriticNetwork(config.state_dim, config.local_critic_hidden, config.action_dim, config.critic_lr, self.device)
        else:
            self.critic_task1 = CriticNetwork(config.state_dim, config.local_critic_hidden, config.action_dim, config.critic_lr, self.device)
            self.critic_task2 = CriticNetwork(config.state_dim, config.local_critic_hidden, config.action_dim, config.critic_lr, self.device)
            self.target_critic_task1 = CriticNetwork(config.state_dim, config.local_critic_hidden, config.action_dim, config.critic_lr, self.device)
            self.target_critic_task2 = CriticNetwork(config.state_dim, config.local_critic_hidden, config.action_dim, config.critic_lr, self.device)
        self.update_network_parameters(tau=1.0)

    @property
    def local_critics(self):
        if self.config.algorithm == "modified_maddpg":
            return (self.critic,)
        return (self.critic_task1, self.critic_task2)

    @property
    def target_local_critics(self):
        if self.config.algorithm == "modified_maddpg":
            return (self.target_critic,)
        return (self.target_critic_task1, self.target_critic_task2)

    def choose_action(
        self,
        observation,
        explore=True,
        noise_std: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        was_training = self.actor.training
        self.actor.eval()
        state = torch.as_tensor(np.asarray(observation, dtype=np.float32), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        applied_noise = float(self.config.exploration_noise) if explore and noise_std is None else float(noise_std or 0.0)
        if applied_noise < 0.0:
            raise ValueError("noise_std must be non-negative")
        if applied_noise > 0.0:
            noise_source = np.random if rng is None else rng
            action = action + noise_source.normal(0.0, applied_noise, size=self.config.action_dim)
        if was_training:
            self.actor.train()
        return np.clip(action, -self.config.target_action_clip, self.config.target_action_clip).astype(np.float32)

    def update_network_parameters(self, tau=None):
        tau = self.config.tau if tau is None else float(tau)
        pairs = [(self.target_actor, self.actor)]
        pairs.extend(zip(self.target_local_critics, self.local_critics))
        for target, source in pairs:
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def state_dict_full(self) -> Dict[str, object]:
        state = {
            "algorithm": self.config.algorithm,
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "actor_optimizer": self.actor.optimizer.state_dict(),
        }
        if self.config.algorithm == "modified_maddpg":
            state.update({
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "critic_optimizer": self.critic.optimizer.state_dict(),
            })
        else:
            state.update({
                "critic_task1": self.critic_task1.state_dict(),
                "critic_task2": self.critic_task2.state_dict(),
                "target_critic_task1": self.target_critic_task1.state_dict(),
                "target_critic_task2": self.target_critic_task2.state_dict(),
                "critic_task1_optimizer": self.critic_task1.optimizer.state_dict(),
                "critic_task2_optimizer": self.critic_task2.optimizer.state_dict(),
            })
        return state

    def load_state_dict_full(self, state):
        checkpoint_algorithm = state.get("algorithm", "modified_maddpg_tdec")
        if checkpoint_algorithm != self.config.algorithm:
            raise ValueError(
                f"agent checkpoint algorithm mismatch: checkpoint={checkpoint_algorithm!r}, "
                f"resolved={self.config.algorithm!r}"
            )
        self.actor.load_state_dict(state["actor"])
        self.target_actor.load_state_dict(state["target_actor"])
        self.actor.optimizer.load_state_dict(state["actor_optimizer"])
        if self.config.algorithm == "modified_maddpg":
            self.critic.load_state_dict(state["critic"])
            self.target_critic.load_state_dict(state["target_critic"])
            self.critic.optimizer.load_state_dict(state["critic_optimizer"])
        else:
            self.critic_task1.load_state_dict(state["critic_task1"])
            self.critic_task2.load_state_dict(state["critic_task2"])
            self.target_critic_task1.load_state_dict(state["target_critic_task1"])
            self.target_critic_task2.load_state_dict(state["target_critic_task2"])
            self.critic_task1.optimizer.load_state_dict(state["critic_task1_optimizer"])
            self.critic_task2.optimizer.load_state_dict(state["critic_task2_optimizer"])
