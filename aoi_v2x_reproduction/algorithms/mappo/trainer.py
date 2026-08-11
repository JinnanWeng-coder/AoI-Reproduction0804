"""Feed-forward hybrid-action MAPPO policy and PPO update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .action_adapter import encode_hybrid_actions
from .networks import CentralValueCritic, HybridActor, RunningValueNorm
from .rollout import RolloutBatch


@dataclass(frozen=True)
class PolicyStep:
    rb: np.ndarray
    mode: np.ndarray
    power: np.ndarray
    log_prob: np.ndarray
    values: np.ndarray
    environment_actions: np.ndarray


class MAPPOTrainer:
    """Separate local policies with one centralized vector-value critic."""

    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = torch.device(device)
        self.number_agents = int(config.number_agents)
        self.observation_dim = int(config.state_dim)
        self.actors = torch.nn.ModuleList([
            HybridActor(config.state_dim, config.actor_hidden, config.n_rb, config.n_modes)
            for _ in range(self.number_agents)
        ]).to(self.device)
        self.critic = CentralValueCritic(
            config.state_dim * config.number_agents,
            config.global_critic_hidden,
            config.number_agents,
        ).to(self.device)
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=float(config.mappo_actor_lr), eps=float(config.mappo_adam_eps))
            for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(config.mappo_critic_lr),
            eps=float(config.mappo_adam_eps),
        )
        self.value_norm = RunningValueNorm(self.number_agents).to(self.device)
        self.policy_version = 0
        self.environment_steps = 0
        self.update_count = 0

    def _observations_tensor(self, observations) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
        expected = (self.number_agents, self.observation_dim)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"MAPPO observations have shape {tuple(tensor.shape)}, expected {expected}")
        return tensor

    @torch.no_grad()
    def act(self, observations, deterministic: bool = False) -> PolicyStep:
        observation_tensor = self._observations_tensor(observations)
        rb_values: List[torch.Tensor] = []
        mode_values: List[torch.Tensor] = []
        power_values: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        for index, actor in enumerate(self.actors):
            sample = actor.sample(observation_tensor[index:index + 1], deterministic=deterministic)
            rb_values.append(sample.rb.squeeze(0))
            mode_values.append(sample.mode.squeeze(0))
            power_values.append(sample.power.squeeze(0))
            log_probs.append(sample.log_prob.squeeze(0))
        rb = torch.stack(rb_values).cpu().numpy().astype(np.int64)
        mode = torch.stack(mode_values).cpu().numpy().astype(np.int64)
        power = torch.stack(power_values).cpu().numpy().astype(np.float32)
        log_prob = torch.stack(log_probs).cpu().numpy().astype(np.float32)
        values = self.critic(observation_tensor.reshape(1, -1)).squeeze(0).cpu().numpy().astype(np.float32)
        environment_actions = encode_hybrid_actions(rb, mode, power, self.config.n_rb, self.config.n_modes)
        return PolicyStep(rb, mode, power, log_prob, values, environment_actions)

    @torch.no_grad()
    def values(self, observations) -> np.ndarray:
        tensor = self._observations_tensor(observations)
        return self.critic(tensor.reshape(1, -1)).squeeze(0).cpu().numpy().astype(np.float32)

    def combined_rewards(self, global_reward: float, task1, task2) -> np.ndarray:
        local = np.asarray(task1, dtype=np.float32) + np.asarray(task2, dtype=np.float32)
        if local.shape != (self.number_agents,):
            raise ValueError("per-agent task rewards have the wrong shape")
        return local + float(self.config.global_actor_weight) * np.float32(global_reward)

    @staticmethod
    def _explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        target_variance = torch.var(targets, dim=0, unbiased=False)
        residual_variance = torch.var(targets - predictions, dim=0, unbiased=False)
        valid = target_variance > 1e-8
        if not torch.any(valid):
            return 0.0
        result = 1.0 - residual_variance[valid] / target_variance[valid]
        return float(result.mean().detach().cpu())

    def update(self, rollout: RolloutBatch) -> Dict[str, object]:
        batch = rollout.to(self.device)
        if batch.observations.ndim != 3 or batch.observations.shape[1:] != (self.number_agents, self.observation_dim):
            raise ValueError("MAPPO rollout observations must have shape [sample, agent, observation]")
        if batch.size < 2:
            raise ValueError("MAPPO requires at least two rollout samples")
        advantages = batch.advantages
        advantage_mean = advantages.mean(dim=0, keepdim=True)
        advantage_std = advantages.std(dim=0, unbiased=False, keepdim=True)
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)
        self.value_norm.update(batch.returns)

        actor_loss_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        entropy_rb_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        entropy_mode_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        entropy_power_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        kl_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        clip_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        actor_grad_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        critic_loss_records: List[float] = []
        critic_grad_records: List[float] = []
        clip_param = float(self.config.mappo_clip_param)

        for _epoch in range(int(self.config.mappo_ppo_epochs)):
            for index, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
                evaluated = actor.evaluate_actions(
                    batch.observations[:, index, :],
                    batch.rb[:, index],
                    batch.mode[:, index],
                    batch.power[:, index],
                )
                log_ratio = evaluated.log_prob - batch.old_log_prob[:, index]
                ratio = torch.exp(log_ratio)
                advantage = normalized_advantages[:, index]
                surrogate = torch.minimum(
                    ratio * advantage,
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage,
                )
                policy_loss = -surrogate.mean()
                entropy_bonus = (
                    float(self.config.mappo_entropy_coef_rb) * evaluated.entropy_rb.mean()
                    + float(self.config.mappo_entropy_coef_mode) * evaluated.entropy_mode.mean()
                    + float(self.config.mappo_entropy_coef_power) * evaluated.entropy_power.mean()
                )
                total_actor_loss = policy_loss - entropy_bonus
                optimizer.zero_grad(set_to_none=True)
                total_actor_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), float(self.config.mappo_max_grad_norm))
                optimizer.step()

                approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > clip_param).float().mean()
                actor_loss_records[index].append(float(policy_loss.detach().cpu()))
                entropy_rb_records[index].append(float(evaluated.entropy_rb.mean().detach().cpu()))
                entropy_mode_records[index].append(float(evaluated.entropy_mode.mean().detach().cpu()))
                entropy_power_records[index].append(float(evaluated.entropy_power.mean().detach().cpu()))
                kl_records[index].append(float(approximate_kl.detach().cpu()))
                clip_records[index].append(float(clip_fraction.detach().cpu()))
                actor_grad_records[index].append(float(torch.as_tensor(grad_norm).detach().cpu()))

            joint_observations = batch.observations.reshape(batch.size, -1)
            predictions = self.critic(joint_observations)
            clipped_predictions = batch.old_values + torch.clamp(
                predictions - batch.old_values,
                -clip_param,
                clip_param,
            )
            normalized_returns = self.value_norm.normalize(batch.returns)
            normalized_predictions = self.value_norm.normalize(predictions)
            normalized_clipped = self.value_norm.normalize(clipped_predictions)
            original_loss = F.smooth_l1_loss(
                normalized_predictions,
                normalized_returns,
                reduction="none",
                beta=float(self.config.mappo_huber_delta),
            )
            clipped_loss = F.smooth_l1_loss(
                normalized_clipped,
                normalized_returns,
                reduction="none",
                beta=float(self.config.mappo_huber_delta),
            )
            value_loss = torch.maximum(original_loss, clipped_loss).mean()
            weighted_value_loss = float(self.config.mappo_value_loss_coef) * value_loss
            self.critic_optimizer.zero_grad(set_to_none=True)
            weighted_value_loss.backward()
            critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.config.mappo_max_grad_norm))
            self.critic_optimizer.step()
            critic_loss_records.append(float(value_loss.detach().cpu()))
            critic_grad_records.append(float(torch.as_tensor(critic_grad).detach().cpu()))

        with torch.no_grad():
            final_values = self.critic(batch.observations.reshape(batch.size, -1))
            explained_variance = self._explained_variance(final_values, batch.returns)
        self.policy_version += 1
        self.update_count += 1
        self.environment_steps += int(batch.size)

        def per_agent_mean(records: List[List[float]]) -> List[float]:
            return [float(np.mean(values)) for values in records]

        diagnostics: Dict[str, object] = {
            "algorithm": "mappo",
            "update": int(self.update_count),
            "policy_version": int(self.policy_version),
            "rollout_steps": int(batch.size),
            "environment_steps": int(self.environment_steps),
            "ppo_epochs": int(self.config.mappo_ppo_epochs),
            "actor_loss_per_agent": per_agent_mean(actor_loss_records),
            "critic_loss": float(np.mean(critic_loss_records)),
            "entropy_rb_per_agent": per_agent_mean(entropy_rb_records),
            "entropy_mode_per_agent": per_agent_mean(entropy_mode_records),
            "entropy_power_per_agent": per_agent_mean(entropy_power_records),
            "approx_kl_per_agent": per_agent_mean(kl_records),
            "clip_fraction_per_agent": per_agent_mean(clip_records),
            "actor_grad_norm_per_agent": per_agent_mean(actor_grad_records),
            "critic_grad_norm": float(np.mean(critic_grad_records)),
            "explained_variance": explained_variance,
            "advantage_mean_per_agent_before_normalization": advantage_mean.squeeze(0).detach().cpu().tolist(),
            "advantage_std_per_agent_before_normalization": advantage_std.squeeze(0).detach().cpu().tolist(),
        }
        numeric = [
            value
            for key, value in diagnostics.items()
            if key not in {"algorithm"}
        ]
        flat_numeric: List[float] = []
        for value in numeric:
            if isinstance(value, list):
                flat_numeric.extend(float(item) for item in value)
            elif isinstance(value, (int, float)):
                flat_numeric.append(float(value))
        if not np.all(np.isfinite(flat_numeric)):
            raise FloatingPointError("non-finite MAPPO update diagnostics")
        return diagnostics

    def policy_state_dicts(self):
        return [
            {name: tensor.detach().cpu() for name, tensor in actor.state_dict().items()}
            for actor in self.actors
        ]

    def parameter_counts(self) -> Dict[str, int]:
        actor_count = sum(parameter.numel() for actor in self.actors for parameter in actor.parameters())
        critic_count = sum(parameter.numel() for parameter in self.critic.parameters())
        return {
            "actors": int(actor_count),
            "critic": int(critic_count),
            "total": int(actor_count + critic_count),
        }
