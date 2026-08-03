"""Centralized TD3 critics and synchronized joint actor update."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

from Classes.G_network import G_CriticNetwork


@contextmanager
def _freeze_modules(modules):
    previous = []
    for module in modules:
        states = [parameter.requires_grad for parameter in module.parameters()]
        previous.append(states)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for module, states in zip(modules, previous):
            for parameter, requires_grad in zip(module.parameters(), states):
                parameter.requires_grad_(requires_grad)


class Global_Critic:
    def __init__(self, config, agents):
        self.config = config
        self.agents_networks = list(agents)
        self.device = torch.device(config.device_resolved if hasattr(config, "device_resolved") else config.device)
        self.number_agents = config.number_agents
        self.number_states = config.state_dim
        self.number_actions = config.action_dim
        self.learn_step_counter = 0
        self.global_critic1 = G_CriticNetwork(config.state_dim, config.global_critic_hidden, config.action_dim, config.number_agents, config.critic_lr, self.device)
        self.global_critic2 = G_CriticNetwork(config.state_dim, config.global_critic_hidden, config.action_dim, config.number_agents, config.critic_lr, self.device)
        self.global_target_critic1 = G_CriticNetwork(config.state_dim, config.global_critic_hidden, config.action_dim, config.number_agents, config.critic_lr, self.device)
        self.global_target_critic2 = G_CriticNetwork(config.state_dim, config.global_critic_hidden, config.action_dim, config.number_agents, config.critic_lr, self.device)
        self.update_global_network_parameters(tau=1.0)

    def _split_states(self, states):
        return [states[:, i * self.number_states:(i + 1) * self.number_states] for i in range(self.number_agents)]

    def _split_actions(self, actions):
        return [actions[:, i * self.number_actions:(i + 1) * self.number_actions] for i in range(self.number_agents)]

    def _target_joint_actions(self, next_states):
        with torch.no_grad():
            target_actions = torch.cat([agent.target_actor(local_state) for agent, local_state in zip(self.agents_networks, self._split_states(next_states))], dim=1)
            noise = torch.randn_like(target_actions) * float(self.config.target_noise_sigma)
            noise = torch.clamp(noise, -float(self.config.target_noise_clip), float(self.config.target_noise_clip))
            return torch.clamp(target_actions + noise, -float(self.config.target_action_clip), float(self.config.target_action_clip))

    def _critic_step(self, states, actions, rewards_g, rewards_t1, rewards_t2, next_states, done):
        with torch.no_grad():
            target_actions = self._target_joint_actions(next_states)
            q1_next = self.global_target_critic1(next_states, target_actions).view(-1)
            q2_next = self.global_target_critic2(next_states, target_actions).view(-1)
            target_global = rewards_g + self.config.gamma * (1.0 - done) * torch.minimum(q1_next, q2_next)

        self.global_critic1.optimizer.zero_grad(set_to_none=True)
        self.global_critic2.optimizer.zero_grad(set_to_none=True)
        q1 = self.global_critic1(states, actions).view(-1)
        q2 = self.global_critic2(states, actions).view(-1)
        loss1 = F.mse_loss(q1, target_global)
        loss2 = F.mse_loss(q2, target_global)
        (loss1 + loss2).backward()
        self.global_critic1.optimizer.step()
        self.global_critic2.optimizer.step()

        local_losses = []
        local_states = self._split_states(states)
        local_next_states = self._split_states(next_states)
        local_actions = self._split_actions(actions)
        for index, agent in enumerate(self.agents_networks):
            with torch.no_grad():
                next_action = agent.target_actor(local_next_states[index])
                next_q1 = agent.target_critic_task1(local_next_states[index], next_action).view(-1)
                next_q2 = agent.target_critic_task2(local_next_states[index], next_action).view(-1)
                target1 = rewards_t1[:, index] + self.config.gamma * (1.0 - done) * next_q1
                target2 = rewards_t2[:, index] + self.config.gamma * (1.0 - done) * next_q2
            agent.critic_task1.optimizer.zero_grad(set_to_none=True)
            agent.critic_task2.optimizer.zero_grad(set_to_none=True)
            current1 = agent.critic_task1(local_states[index], local_actions[index]).view(-1)
            current2 = agent.critic_task2(local_states[index], local_actions[index]).view(-1)
            local_loss1 = F.mse_loss(current1, target1)
            local_loss2 = F.mse_loss(current2, target2)
            local_loss1.backward()
            agent.critic_task1.optimizer.step()
            local_loss2.backward()
            agent.critic_task2.optimizer.step()
            local_losses.append(float((local_loss1.detach() + local_loss2.detach()).cpu()))

        return {
            "global_critic_loss": float((loss1.detach() + loss2.detach()).cpu()),
            "local_critic_loss": local_losses,
        }

    def _joint_actor_terms(self, states):
        local_states = self._split_states(states)
        live_actions = [agent.actor(local_state) for agent, local_state in zip(self.agents_networks, local_states)]
        joint_action = torch.cat(live_actions, dim=1)
        global_loss = -float(self.config.global_actor_weight) * self.global_critic1(states, joint_action).mean()
        local_loss = torch.zeros((), dtype=states.dtype, device=self.device)
        for agent, local_state, action in zip(self.agents_networks, local_states, live_actions):
            local_loss = local_loss - agent.critic_task1(local_state, action).mean() - agent.critic_task2(local_state, action).mean()
        return global_loss, local_loss, live_actions

    @staticmethod
    def _grad_norms(gradients) -> List[float]:
        values = []
        for gradient in gradients:
            if gradient is None:
                values.append(0.0)
            else:
                values.append(float(torch.linalg.vector_norm(gradient).detach().cpu()))
        return values

    def actor_global_gradient_audit(self, states: np.ndarray) -> Dict[str, object]:
        tensor_states = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        if self.config.global_update_mode == "legacy_detach":
            return {"mode": "legacy_detach", "global_gradient_norms": [0.0 for _ in self.agents_networks], "finite": True}
        critics = [self.global_critic1, *[agent.critic_task1 for agent in self.agents_networks], *[agent.critic_task2 for agent in self.agents_networks]]
        with _freeze_modules(critics):
            global_loss, _, _ = self._joint_actor_terms(tensor_states)
            gradients = []
            for agent in self.agents_networks:
                gradients.extend(list(agent.actor.parameters()))
            all_grads = torch.autograd.grad(global_loss, gradients, retain_graph=False, allow_unused=True)
        offsets = []
        cursor = 0
        for agent in self.agents_networks:
            count = sum(1 for _ in agent.actor.parameters())
            offsets.append(self._grad_norms(all_grads[cursor:cursor + count]))
            cursor += count
        norms = [float(np.sqrt(np.sum(np.square(values)))) for values in offsets]
        return {"mode": self.config.global_update_mode, "global_gradient_norms": norms, "per_parameter_norms": offsets, "finite": bool(np.all(np.isfinite(norms)))}

    def _actor_step(self, states):
        critics = [self.global_critic1, self.global_critic2]
        for agent in self.agents_networks:
            critics.extend([agent.critic_task1, agent.critic_task2])
        for agent in self.agents_networks:
            agent.actor.optimizer.zero_grad(set_to_none=True)
        before = [torch.cat([parameter.detach().flatten() for parameter in agent.actor.parameters()]) for agent in self.agents_networks]
        if self.config.global_update_mode == "legacy_detach":
            with _freeze_modules(critics):
                global_loss, _, _ = self._joint_actor_terms(states)
                for agent, local_state in zip(self.agents_networks, self._split_states(states)):
                    agent.actor.optimizer.zero_grad(set_to_none=True)
                    action = agent.actor(local_state)
                    local_loss = -agent.critic_task1(local_state, action).mean() - agent.critic_task2(local_state, action).mean()
                    (local_loss + float(self.config.global_actor_weight) * global_loss.detach()).backward()
                    agent.actor.optimizer.step()
            global_norms = [0.0 for _ in self.agents_networks]
            actor_loss = float("nan")
        else:
            with _freeze_modules(critics):
                global_loss, local_loss, _ = self._joint_actor_terms(states)
                gradients = []
                for agent in self.agents_networks:
                    gradients.extend(list(agent.actor.parameters()))
                global_grads = torch.autograd.grad(global_loss, gradients, retain_graph=True, allow_unused=True)
                total_loss = global_loss + local_loss
                total_loss.backward()
                for agent in self.agents_networks:
                    agent.actor.optimizer.step()
            per_agent = []
            cursor = 0
            for agent in self.agents_networks:
                count = sum(1 for _ in agent.actor.parameters())
                per_agent.append(self._grad_norms(global_grads[cursor:cursor + count]))
                cursor += count
            global_norms = [float(np.sqrt(np.sum(np.square(values)))) for values in per_agent]
            actor_loss = float(total_loss.detach().cpu())
        deltas = []
        for agent, old in zip(self.agents_networks, before):
            new = torch.cat([parameter.detach().flatten() for parameter in agent.actor.parameters()])
            deltas.append(float(torch.linalg.vector_norm(new - old).cpu()))
        for agent in self.agents_networks:
            agent.update_network_parameters()
        self.update_global_network_parameters()
        return {"actor_loss": actor_loss, "global_actor_gradient_norms": global_norms, "actor_parameter_deltas": deltas}

    def learn(self, batch):
        states, actions, rewards_g, rewards_t1, rewards_t2, next_states, done = batch
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_g_t = torch.as_tensor(rewards_g, dtype=torch.float32, device=self.device).view(-1)
        rewards_t1_t = torch.as_tensor(rewards_t1, dtype=torch.float32, device=self.device)
        rewards_t2_t = torch.as_tensor(rewards_t2, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1)
        diagnostics = self._critic_step(states_t, actions_t, rewards_g_t, rewards_t1_t, rewards_t2_t, next_states_t, done_t)
        self.learn_step_counter += 1
        if self.learn_step_counter % int(self.config.policy_delay) == 0:
            diagnostics.update(self._actor_step(states_t))
        else:
            diagnostics.update({"actor_loss": None, "global_actor_gradient_norms": [], "actor_parameter_deltas": []})
        diagnostics["learn_step"] = self.learn_step_counter
        return diagnostics

    def update_global_network_parameters(self, tau=None):
        tau = self.config.tau if tau is None else float(tau)
        for target, source in ((self.global_target_critic1, self.global_critic1), (self.global_target_critic2, self.global_critic2)):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def state_dict_full(self):
        return {
            "global_critic1": self.global_critic1.state_dict(),
            "global_critic2": self.global_critic2.state_dict(),
            "global_target_critic1": self.global_target_critic1.state_dict(),
            "global_target_critic2": self.global_target_critic2.state_dict(),
            "global_critic1_optimizer": self.global_critic1.optimizer.state_dict(),
            "global_critic2_optimizer": self.global_critic2.optimizer.state_dict(),
            "learn_step_counter": self.learn_step_counter,
        }

    def load_state_dict_full(self, state):
        self.global_critic1.load_state_dict(state["global_critic1"])
        self.global_critic2.load_state_dict(state["global_critic2"])
        self.global_target_critic1.load_state_dict(state["global_target_critic1"])
        self.global_target_critic2.load_state_dict(state["global_target_critic2"])
        self.global_critic1.optimizer.load_state_dict(state["global_critic1_optimizer"])
        self.global_critic2.optimizer.load_state_dict(state["global_critic2_optimizer"])
        self.learn_step_counter = int(state["learn_step_counter"])

