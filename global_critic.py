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

    def _local_critic_modules(self):
        return [critic for agent in self.agents_networks for critic in agent.local_critics]

    def _local_actor_loss(self, agent, local_state, action):
        return sum(
            (-critic(local_state, action).mean() for critic in agent.local_critics),
            torch.zeros((), dtype=local_state.dtype, device=self.device),
        )

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

    def _critic_step(self, states, actions, rewards_g, rewards_t1, rewards_t2, next_states, done, update_local=True):
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
        # Global target critics follow every global critic optimizer step.
        # Local target networks remain policy-delayed and are updated below
        # only when ``update_local`` is true.
        self.update_global_network_parameters()

        local_losses = []
        if update_local:
            local_states = self._split_states(states)
            local_next_states = self._split_states(next_states)
            local_actions = self._split_actions(actions)
            for index, agent in enumerate(self.agents_networks):
                if self.config.algorithm == "modified_maddpg":
                    with torch.no_grad():
                        next_action = agent.target_actor(local_next_states[index])
                        next_q = agent.target_critic(local_next_states[index], next_action).view(-1)
                        reward_local = rewards_t1[:, index] + rewards_t2[:, index]
                        target = reward_local + self.config.gamma * (1.0 - done) * next_q
                    agent.critic.optimizer.zero_grad(set_to_none=True)
                    current = agent.critic(local_states[index], local_actions[index]).view(-1)
                    local_loss = F.mse_loss(current, target)
                    local_loss.backward()
                    agent.critic.optimizer.step()
                    local_losses.append(float(local_loss.detach().cpu()))
                else:
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

    def _actor_component_terms(self, states):
        local_states = self._split_states(states)
        live_actions = [agent.actor(local_state) for agent, local_state in zip(self.agents_networks, local_states)]
        joint_action = torch.cat(live_actions, dim=1)
        global_loss = -float(self.config.global_actor_weight) * self.global_critic1(states, joint_action).mean()
        if self.config.algorithm == "modified_maddpg":
            components = {
                "local": [
                    -agent.critic(local_state, action).mean()
                    for agent, local_state, action in zip(self.agents_networks, local_states, live_actions)
                ]
            }
        else:
            components = {
                "task1": [
                    -agent.critic_task1(local_state, action).mean()
                    for agent, local_state, action in zip(self.agents_networks, local_states, live_actions)
                ],
                "task2": [
                    -agent.critic_task2(local_state, action).mean()
                    for agent, local_state, action in zip(self.agents_networks, local_states, live_actions)
                ],
            }
        local_loss = sum(
            (loss for losses in components.values() for loss in losses),
            torch.zeros((), dtype=states.dtype, device=self.device),
        )
        return global_loss, local_loss, live_actions, components

    def _joint_actor_terms(self, states):
        global_loss, local_loss, live_actions, _components = self._actor_component_terms(states)
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

    def _actor_gradient_diagnostics(self, component_losses, global_grads_by_agent, global_contributes: bool):
        parameters_by_agent = [list(agent.actor.parameters()) for agent in self.agents_networks]
        flat_parameters = [parameter for parameters in parameters_by_agent for parameter in parameters]
        component_grads = {}
        for name, losses in component_losses.items():
            total = sum(losses, torch.zeros((), device=self.device))
            component_grads[name] = torch.autograd.grad(total, flat_parameters, retain_graph=True, allow_unused=True)

        rows = []
        cursor = 0
        epsilon = 1e-30
        for index, parameters in enumerate(parameters_by_agent):
            count = len(parameters)

            def flatten(values):
                return torch.cat([
                    (torch.zeros_like(parameter) if value is None else value).reshape(-1)
                    for parameter, value in zip(parameters, values)
                ])

            vectors = {
                name: flatten(gradients[cursor:cursor + count])
                for name, gradients in component_grads.items()
            }
            global_vector = flatten(global_grads_by_agent[index])
            cursor += count
            local_vector = sum(vectors.values(), torch.zeros_like(global_vector))
            global_norm = torch.linalg.vector_norm(global_vector)
            local_norm = torch.linalg.vector_norm(local_vector)

            def ratio(numerator, denominator):
                return numerator / torch.clamp(denominator, min=epsilon)

            def cosine(left, right, left_norm, right_norm):
                denominator = left_norm * right_norm
                value = torch.dot(left, right) / torch.clamp(denominator, min=epsilon)
                return torch.clamp(value, -1.0, 1.0)

            if self.config.algorithm == "modified_maddpg":
                rows.append(torch.stack((
                    local_norm,
                    global_norm,
                    ratio(global_norm, local_norm),
                    cosine(global_vector, local_vector, global_norm, local_norm),
                )))
            else:
                task1_vector = vectors["task1"]
                task2_vector = vectors["task2"]
                task1_norm = torch.linalg.vector_norm(task1_vector)
                task2_norm = torch.linalg.vector_norm(task2_vector)
                rows.append(torch.stack((
                    task1_norm,
                    task2_norm,
                    local_norm,
                    global_norm,
                    ratio(global_norm, local_norm),
                    ratio(global_norm, task1_norm),
                    ratio(global_norm, task2_norm),
                    ratio(task1_norm, task2_norm),
                    cosine(global_vector, local_vector, global_norm, local_norm),
                    cosine(global_vector, task1_vector, global_norm, task1_norm),
                    cosine(global_vector, task2_vector, global_norm, task2_norm),
                    cosine(task1_vector, task2_vector, task1_norm, task2_norm),
                )))
        field_names = (
            ("local_grad_l2", "global_grad_l2", "global_to_local_ratio", "global_vs_local_cosine")
            if self.config.algorithm == "modified_maddpg"
            else (
                "task1_grad_l2", "task2_grad_l2", "local_sum_grad_l2", "global_grad_l2",
                "global_to_local_ratio", "global_to_task1_ratio", "global_to_task2_ratio",
                "task1_to_task2_ratio", "global_vs_local_cosine", "global_vs_task1_cosine",
                "global_vs_task2_cosine", "task1_vs_task2_cosine",
            )
        )
        values = torch.stack(rows).detach().cpu().numpy()
        fields = {name: values[:, column].astype(np.float64).tolist() for column, name in enumerate(field_names)}
        fields["mode"] = self.config.global_update_mode
        fields["global_contributes_to_actor"] = bool(global_contributes)
        fields["finite"] = bool(np.all(np.isfinite(values)))
        return fields

    def actor_global_gradient_audit(self, states: np.ndarray) -> Dict[str, object]:
        tensor_states = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        if self.config.global_update_mode in {"legacy_detach", "detached_actor"}:
            return {"mode": self.config.global_update_mode, "global_gradient_norms": [0.0 for _ in self.agents_networks], "finite": True}
        critics = [self.global_critic1, *self._local_critic_modules()]
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

    def _actor_step(self, states, actor_step_order=None):
        if actor_step_order is None:
            actor_step_order = list(range(len(self.agents_networks)))
        else:
            actor_step_order = [int(index) for index in actor_step_order]
            if sorted(actor_step_order) != list(range(len(self.agents_networks))):
                raise ValueError("actor_step_order must be a permutation of agent indices")
        critics = [self.global_critic1, self.global_critic2, *self._local_critic_modules()]
        for agent in self.agents_networks:
            agent.actor.optimizer.zero_grad(set_to_none=True)
        before = [torch.cat([parameter.detach().flatten() for parameter in agent.actor.parameters()]) for agent in self.agents_networks]
        gradient_diagnostics = None
        if self.config.global_update_mode in {"legacy_detach", "detached_actor"}:
            with _freeze_modules(critics):
                if bool(getattr(self.config, "diagnostics", False)):
                    counterfactual_global, _local, _actions, component_losses = self._actor_component_terms(states)
                    actor_parameters = [list(agent.actor.parameters()) for agent in self.agents_networks]
                    flat_parameters = [parameter for parameters in actor_parameters for parameter in parameters]
                    flat_global_grads = torch.autograd.grad(
                        counterfactual_global,
                        flat_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    global_grads_by_agent = []
                    cursor = 0
                    for parameters in actor_parameters:
                        global_grads_by_agent.append(flat_global_grads[cursor:cursor + len(parameters)])
                        cursor += len(parameters)
                    gradient_diagnostics = self._actor_gradient_diagnostics(
                        component_losses,
                        global_grads_by_agent,
                        global_contributes=False,
                    )
                global_loss, _, _ = self._joint_actor_terms(states)
                for agent, local_state in zip(self.agents_networks, self._split_states(states)):
                    agent.actor.optimizer.zero_grad(set_to_none=True)
                    action = agent.actor(local_state)
                    local_loss = self._local_actor_loss(agent, local_state, action)
                    (local_loss + float(self.config.global_actor_weight) * global_loss.detach()).backward()
                    agent.actor.optimizer.step()
            global_norms = [0.0 for _ in self.agents_networks]
            actor_loss = float("nan")
        else:
            with _freeze_modules(critics):
                global_loss, local_loss, _, component_losses = self._actor_component_terms(states)
                gradients = []
                for agent in self.agents_networks:
                    gradients.extend(list(agent.actor.parameters()))
                global_grads = torch.autograd.grad(global_loss, gradients, retain_graph=True, allow_unused=True)
                global_grads_by_agent = []
                cursor = 0
                for agent in self.agents_networks:
                    count = sum(1 for _ in agent.actor.parameters())
                    global_grads_by_agent.append(global_grads[cursor:cursor + count])
                    cursor += count
                if bool(getattr(self.config, "diagnostics", False)):
                    gradient_diagnostics = self._actor_gradient_diagnostics(
                        component_losses,
                        global_grads_by_agent,
                        global_contributes=True,
                    )
                total_loss = global_loss + local_loss
                total_loss.backward()
                for index in actor_step_order:
                    self.agents_networks[index].actor.optimizer.step()
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
        return {
            "actor_loss": actor_loss,
            "global_actor_gradient_norms": global_norms,
            "actor_parameter_deltas": deltas,
            "actor_gradient_diagnostics": gradient_diagnostics,
        }

    def learn(self, batch):
        states, actions, rewards_g, rewards_t1, rewards_t2, next_states, done = batch
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_g_t = torch.as_tensor(rewards_g, dtype=torch.float32, device=self.device).view(-1)
        rewards_t1_t = torch.as_tensor(rewards_t1, dtype=torch.float32, device=self.device)
        rewards_t2_t = torch.as_tensor(rewards_t2, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done, dtype=torch.float32, device=self.device).view(-1)
        next_learn_step = self.learn_step_counter + 1
        update_local = next_learn_step % int(self.config.policy_delay) == 0
        diagnostics = self._critic_step(
            states_t,
            actions_t,
            rewards_g_t,
            rewards_t1_t,
            rewards_t2_t,
            next_states_t,
            done_t,
            update_local=update_local,
        )
        self.learn_step_counter += 1
        if update_local:
            diagnostics.update(self._actor_step(states_t))
        else:
            diagnostics.update({"actor_loss": None, "global_actor_gradient_norms": [], "actor_parameter_deltas": [], "actor_gradient_diagnostics": None})
        diagnostics["learn_step"] = self.learn_step_counter
        diagnostics["global_target_update"] = True
        diagnostics["local_target_update"] = bool(update_local)
        return diagnostics

    def update_global_network_parameters(self, tau=None):
        tau = self.config.tau if tau is None else float(tau)
        for target, source in ((self.global_target_critic1, self.global_critic1), (self.global_target_critic2, self.global_critic2)):
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def state_dict_full(self):
        return {
            "algorithm": self.config.algorithm,
            "global_critic1": self.global_critic1.state_dict(),
            "global_critic2": self.global_critic2.state_dict(),
            "global_target_critic1": self.global_target_critic1.state_dict(),
            "global_target_critic2": self.global_target_critic2.state_dict(),
            "global_critic1_optimizer": self.global_critic1.optimizer.state_dict(),
            "global_critic2_optimizer": self.global_critic2.optimizer.state_dict(),
            "learn_step_counter": self.learn_step_counter,
        }

    def load_state_dict_full(self, state):
        checkpoint_algorithm = state.get("algorithm", "modified_maddpg_tdec")
        if checkpoint_algorithm != self.config.algorithm:
            raise ValueError(
                f"learner checkpoint algorithm mismatch: checkpoint={checkpoint_algorithm!r}, "
                f"resolved={self.config.algorithm!r}"
            )
        self.global_critic1.load_state_dict(state["global_critic1"])
        self.global_critic2.load_state_dict(state["global_critic2"])
        self.global_target_critic1.load_state_dict(state["global_target_critic1"])
        self.global_target_critic2.load_state_dict(state["global_target_critic2"])
        self.global_critic1.optimizer.load_state_dict(state["global_critic1_optimizer"])
        self.global_critic2.optimizer.load_state_dict(state["global_critic2_optimizer"])
        self.learn_step_counter = int(state["learn_step_counter"])
