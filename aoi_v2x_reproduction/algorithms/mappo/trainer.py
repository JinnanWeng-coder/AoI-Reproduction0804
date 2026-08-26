"""Feed-forward hybrid-action MAPPO policy and PPO update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .action_adapter import encode_hybrid_actions
from .networks import CentralValueCritic, HybridActor, LocalValueCritic, RunningValueNorm
from .rollout import RolloutBatch, TDecRolloutBatch


@dataclass(frozen=True)
class TDecValueStep:
    global_value: np.ndarray
    task1_values: np.ndarray
    task2_values: np.ndarray


@dataclass(frozen=True)
class PolicyStep:
    rb: np.ndarray
    mode: np.ndarray
    power: np.ndarray
    log_prob: np.ndarray
    values: np.ndarray
    environment_actions: np.ndarray
    tdec_values: Optional[TDecValueStep] = None


def _value_loss_inputs(
    predictions: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    value_norm: RunningValueNorm,
    clip_param: float,
    clip_mode: str,
):
    """Return normalized current, clipped, and target values for critic loss.

    Rollout values and GAE remain in their raw reward scale.  The corrected
    mode maps both value predictions to the running normalized scale before
    applying PPO's dimensionless clipping threshold.  ``legacy_raw`` exactly
    retains the earlier raw-space clipping order for historical replication.
    """
    normalized_predictions = value_norm.normalize(predictions)
    normalized_old_values = value_norm.normalize(old_values)
    normalized_returns = value_norm.normalize(returns)
    if clip_mode == "normalized":
        normalized_clipped = normalized_old_values + torch.clamp(
            normalized_predictions - normalized_old_values,
            -clip_param,
            clip_param,
        )
    elif clip_mode == "legacy_raw":
        clipped_raw = old_values + torch.clamp(
            predictions - old_values,
            -clip_param,
            clip_param,
        )
        normalized_clipped = value_norm.normalize(clipped_raw)
    else:
        raise ValueError(f"unsupported MAPPO value clip mode: {clip_mode}")
    return normalized_predictions, normalized_clipped, normalized_returns


def _clipped_value_loss(
    predictions: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    value_norm: RunningValueNorm,
    clip_param: float,
    clip_mode: str,
    huber_delta: float,
) -> torch.Tensor:
    normalized_predictions, normalized_clipped, normalized_returns = _value_loss_inputs(
        predictions=predictions,
        old_values=old_values,
        returns=returns,
        value_norm=value_norm,
        clip_param=clip_param,
        clip_mode=clip_mode,
    )
    original_loss = F.smooth_l1_loss(
        normalized_predictions,
        normalized_returns,
        reduction="none",
        beta=float(huber_delta),
    )
    clipped_loss = F.smooth_l1_loss(
        normalized_clipped,
        normalized_returns,
        reduction="none",
        beta=float(huber_delta),
    )
    return torch.maximum(original_loss, clipped_loss).mean()


def _mean_joint_gradient_rss(*stream_records: List[float]) -> float:
    """Combine stream gradient norms within each epoch, then average epochs."""

    if not stream_records or not stream_records[0]:
        raise ValueError("gradient aggregation requires at least one recorded epoch")
    if any(len(records) != len(stream_records[0]) for records in stream_records):
        raise ValueError("gradient streams must have the same number of epochs")
    per_epoch = np.sqrt(np.square(np.asarray(stream_records, dtype=np.float64)).sum(axis=0))
    return float(per_epoch.mean())


class MAPPOTrainer:
    """Separate local policies with combined or task-decomposed value critics."""

    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = torch.device(device)
        self.number_agents = int(config.number_agents)
        self.observation_dim = int(config.state_dim)
        self.variant = str(config.mappo_variant)
        self.actors = torch.nn.ModuleList([
            HybridActor(config.state_dim, config.actor_hidden, config.n_rb, config.n_modes)
            for _ in range(self.number_agents)
        ]).to(self.device)
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=float(config.mappo_actor_lr), eps=float(config.mappo_adam_eps))
            for actor in self.actors
        ]
        self.critic = None
        self.critic_optimizer = None
        self.value_norm = None
        self.global_critic = None
        self.task1_critics = None
        self.task2_critics = None
        self.global_critic_optimizer = None
        self.task1_critic_optimizer = None
        self.task2_critic_optimizer = None
        self.global_value_norm = None
        self.task1_value_norm = None
        self.task2_value_norm = None
        if self.variant == "combined":
            self.critic = CentralValueCritic(
                config.state_dim * config.number_agents,
                config.global_critic_hidden,
                config.number_agents,
            ).to(self.device)
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(),
                lr=float(config.mappo_critic_lr),
                eps=float(config.mappo_adam_eps),
            )
            self.value_norm = RunningValueNorm(self.number_agents).to(self.device)
        elif self.variant == "tdec":
            self.global_critic = CentralValueCritic(
                config.state_dim * config.number_agents,
                config.global_critic_hidden,
                1,
            ).to(self.device)
            self.task1_critics = torch.nn.ModuleList([
                LocalValueCritic(config.state_dim, config.local_critic_hidden)
                for _ in range(self.number_agents)
            ]).to(self.device)
            self.task2_critics = torch.nn.ModuleList([
                LocalValueCritic(config.state_dim, config.local_critic_hidden)
                for _ in range(self.number_agents)
            ]).to(self.device)
            optimizer_kwargs = {
                "lr": float(config.mappo_critic_lr),
                "eps": float(config.mappo_adam_eps),
            }
            self.global_critic_optimizer = torch.optim.Adam(self.global_critic.parameters(), **optimizer_kwargs)
            self.task1_critic_optimizer = torch.optim.Adam(self.task1_critics.parameters(), **optimizer_kwargs)
            self.task2_critic_optimizer = torch.optim.Adam(self.task2_critics.parameters(), **optimizer_kwargs)
            self.global_value_norm = RunningValueNorm(1).to(self.device)
            self.task1_value_norm = RunningValueNorm(self.number_agents).to(self.device)
            self.task2_value_norm = RunningValueNorm(self.number_agents).to(self.device)
        else:
            raise ValueError(f"unsupported MAPPO variant: {self.variant}")
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
        tdec_values = None
        if self.variant == "combined":
            values = self.critic(observation_tensor.reshape(1, -1)).squeeze(0).cpu().numpy().astype(np.float32)
        else:
            global_value, task1_values, task2_values = self._tdec_values_tensor(observation_tensor)
            tdec_values = TDecValueStep(
                global_value=global_value.cpu().numpy().astype(np.float32),
                task1_values=task1_values.cpu().numpy().astype(np.float32),
                task2_values=task2_values.cpu().numpy().astype(np.float32),
            )
            values = (
                float(self.config.global_actor_weight) * global_value.expand(self.number_agents)
                + task1_values
                + task2_values
            ).cpu().numpy().astype(np.float32)
        environment_actions = encode_hybrid_actions(rb, mode, power, self.config.n_rb, self.config.n_modes)
        return PolicyStep(rb, mode, power, log_prob, values, environment_actions, tdec_values)

    @torch.no_grad()
    def values(self, observations) -> np.ndarray:
        tensor = self._observations_tensor(observations)
        if self.variant == "combined":
            return self.critic(tensor.reshape(1, -1)).squeeze(0).cpu().numpy().astype(np.float32)
        global_value, task1_values, task2_values = self._tdec_values_tensor(tensor)
        return (
            float(self.config.global_actor_weight) * global_value.expand(self.number_agents)
            + task1_values
            + task2_values
        ).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def tdec_values(self, observations) -> TDecValueStep:
        if self.variant != "tdec":
            raise RuntimeError("task-decomposed values require mappo_variant=tdec")
        tensor = self._observations_tensor(observations)
        global_value, task1_values, task2_values = self._tdec_values_tensor(tensor)
        return TDecValueStep(
            global_value=global_value.cpu().numpy().astype(np.float32),
            task1_values=task1_values.cpu().numpy().astype(np.float32),
            task2_values=task2_values.cpu().numpy().astype(np.float32),
        )

    def _tdec_values_tensor(self, observations: torch.Tensor):
        if self.variant != "tdec":
            raise RuntimeError("task-decomposed values require mappo_variant=tdec")
        global_value = self.global_critic(observations.reshape(1, -1)).squeeze(0)
        task1_values = torch.stack([
            critic(observations[index:index + 1]).squeeze(0)
            for index, critic in enumerate(self.task1_critics)
        ])
        task2_values = torch.stack([
            critic(observations[index:index + 1]).squeeze(0)
            for index, critic in enumerate(self.task2_critics)
        ])
        return global_value, task1_values, task2_values

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

    @staticmethod
    def _per_agent_correlation(first: torch.Tensor, second: torch.Tensor) -> List[float]:
        first_centered = first - first.mean(dim=0, keepdim=True)
        second_centered = second - second.mean(dim=0, keepdim=True)
        numerator = (first_centered * second_centered).sum(dim=0)
        denominator = torch.sqrt(
            first_centered.square().sum(dim=0) * second_centered.square().sum(dim=0)
        )
        correlation = torch.where(denominator > 1e-8, numerator / denominator, torch.zeros_like(numerator))
        return correlation.detach().cpu().tolist()

    def _local_value_predictions(self, critics: torch.nn.ModuleList, observations: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            critic(observations[:, index, :])
            for index, critic in enumerate(critics)
        ], dim=1)

    def _optimize_value_stream(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        predictions: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        value_norm: RunningValueNorm,
        clip_param: float,
    ):
        value_loss = _clipped_value_loss(
            predictions=predictions,
            old_values=old_values,
            returns=returns,
            value_norm=value_norm,
            clip_param=clip_param,
            clip_mode=self.config.mappo_value_clip_mode,
            huber_delta=self.config.mappo_huber_delta,
        )
        weighted_value_loss = float(self.config.mappo_value_loss_coef) * value_loss
        optimizer.zero_grad(set_to_none=True)
        weighted_value_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), float(self.config.mappo_max_grad_norm))
        optimizer.step()
        return float(value_loss.detach().cpu()), float(torch.as_tensor(grad_norm).detach().cpu())

    def update(self, rollout: Union[RolloutBatch, TDecRolloutBatch]) -> Dict[str, object]:
        if self.variant == "combined" and not isinstance(rollout, RolloutBatch):
            raise ValueError("combined MAPPO requires a combined-reward rollout")
        if self.variant == "tdec" and not isinstance(rollout, TDecRolloutBatch):
            raise ValueError("TDec MAPPO requires a task-decomposed rollout")
        batch = rollout.to(self.device)
        if batch.observations.ndim != 3 or batch.observations.shape[1:] != (self.number_agents, self.observation_dim):
            raise ValueError("MAPPO rollout observations must have shape [sample, agent, observation]")
        if batch.size < 2:
            raise ValueError("MAPPO requires at least two rollout samples")
        component_diagnostics: Dict[str, object] = {}
        if self.variant == "combined":
            advantages = batch.advantages
            self.value_norm.update(batch.returns)
        else:
            expected_global = (batch.size, 1)
            expected_local = (batch.size, self.number_agents)
            for name in ("global_advantages", "global_returns", "old_global_values"):
                if tuple(getattr(batch, name).shape) != expected_global:
                    raise ValueError(f"TDec {name} must have shape [sample, 1]")
            for name in (
                "task1_advantages", "task2_advantages", "task1_returns", "task2_returns",
                "old_task1_values", "old_task2_values",
            ):
                if tuple(getattr(batch, name).shape) != expected_local:
                    raise ValueError(f"TDec {name} must have shape [sample, agent]")
            weighted_global_advantages = (
                float(self.config.global_actor_weight)
                * batch.global_advantages.expand(-1, self.number_agents)
            )
            advantages = weighted_global_advantages + batch.task1_advantages + batch.task2_advantages
            self.global_value_norm.update(batch.global_returns)
            self.task1_value_norm.update(batch.task1_returns)
            self.task2_value_norm.update(batch.task2_returns)
            component_diagnostics = {
                "global_advantage_mean_per_agent_before_composition": weighted_global_advantages.mean(dim=0).detach().cpu().tolist(),
                "global_advantage_std_per_agent_before_composition": weighted_global_advantages.std(dim=0, unbiased=False).detach().cpu().tolist(),
                "task1_advantage_mean_per_agent_before_composition": batch.task1_advantages.mean(dim=0).detach().cpu().tolist(),
                "task1_advantage_std_per_agent_before_composition": batch.task1_advantages.std(dim=0, unbiased=False).detach().cpu().tolist(),
                "task2_advantage_mean_per_agent_before_composition": batch.task2_advantages.mean(dim=0).detach().cpu().tolist(),
                "task2_advantage_std_per_agent_before_composition": batch.task2_advantages.std(dim=0, unbiased=False).detach().cpu().tolist(),
                "global_task1_advantage_correlation_per_agent": self._per_agent_correlation(
                    weighted_global_advantages, batch.task1_advantages
                ),
                "global_task2_advantage_correlation_per_agent": self._per_agent_correlation(
                    weighted_global_advantages, batch.task2_advantages
                ),
                "task1_task2_advantage_correlation_per_agent": self._per_agent_correlation(
                    batch.task1_advantages, batch.task2_advantages
                ),
            }
        advantage_mean = advantages.mean(dim=0, keepdim=True)
        advantage_std = advantages.std(dim=0, unbiased=False, keepdim=True)
        normalized_advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

        actor_loss_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        entropy_rb_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        entropy_mode_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        entropy_power_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        kl_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        clip_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        actor_grad_records: List[List[float]] = [[] for _ in range(self.number_agents)]
        critic_loss_records: List[float] = []
        critic_grad_records: List[float] = []
        global_critic_loss_records: List[float] = []
        task1_critic_loss_records: List[float] = []
        task2_critic_loss_records: List[float] = []
        global_critic_grad_records: List[float] = []
        task1_critic_grad_records: List[float] = []
        task2_critic_grad_records: List[float] = []
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
            if self.variant == "combined":
                value_loss, critic_grad = self._optimize_value_stream(
                    self.critic,
                    self.critic_optimizer,
                    self.critic(joint_observations),
                    batch.old_values,
                    batch.returns,
                    self.value_norm,
                    clip_param,
                )
                critic_loss_records.append(value_loss)
                critic_grad_records.append(critic_grad)
            else:
                global_loss, global_grad = self._optimize_value_stream(
                    self.global_critic,
                    self.global_critic_optimizer,
                    self.global_critic(joint_observations),
                    batch.old_global_values,
                    batch.global_returns,
                    self.global_value_norm,
                    clip_param,
                )
                task1_loss, task1_grad = self._optimize_value_stream(
                    self.task1_critics,
                    self.task1_critic_optimizer,
                    self._local_value_predictions(self.task1_critics, batch.observations),
                    batch.old_task1_values,
                    batch.task1_returns,
                    self.task1_value_norm,
                    clip_param,
                )
                task2_loss, task2_grad = self._optimize_value_stream(
                    self.task2_critics,
                    self.task2_critic_optimizer,
                    self._local_value_predictions(self.task2_critics, batch.observations),
                    batch.old_task2_values,
                    batch.task2_returns,
                    self.task2_value_norm,
                    clip_param,
                )
                global_critic_loss_records.append(global_loss)
                task1_critic_loss_records.append(task1_loss)
                task2_critic_loss_records.append(task2_loss)
                global_critic_grad_records.append(global_grad)
                task1_critic_grad_records.append(task1_grad)
                task2_critic_grad_records.append(task2_grad)
                critic_loss_records.append(global_loss + task1_loss + task2_loss)

        with torch.no_grad():
            if self.variant == "combined":
                final_values = self.critic(batch.observations.reshape(batch.size, -1))
                explained_variance = self._explained_variance(final_values, batch.returns)
                critic_loss = float(np.mean(critic_loss_records))
                critic_grad_norm = float(np.mean(critic_grad_records))
                critic_diagnostics: Dict[str, object] = {}
            else:
                final_global = self.global_critic(batch.observations.reshape(batch.size, -1))
                final_task1 = self._local_value_predictions(self.task1_critics, batch.observations)
                final_task2 = self._local_value_predictions(self.task2_critics, batch.observations)
                global_explained_variance = self._explained_variance(final_global, batch.global_returns)
                task1_explained_variance = self._explained_variance(final_task1, batch.task1_returns)
                task2_explained_variance = self._explained_variance(final_task2, batch.task2_returns)
                global_critic_loss = float(np.mean(global_critic_loss_records))
                task1_critic_loss = float(np.mean(task1_critic_loss_records))
                task2_critic_loss = float(np.mean(task2_critic_loss_records))
                global_critic_grad_norm = float(np.mean(global_critic_grad_records))
                task1_critic_grad_norm = float(np.mean(task1_critic_grad_records))
                task2_critic_grad_norm = float(np.mean(task2_critic_grad_records))
                critic_loss = float(np.mean(critic_loss_records))
                critic_grad_norm = _mean_joint_gradient_rss(
                    global_critic_grad_records,
                    task1_critic_grad_records,
                    task2_critic_grad_records,
                )
                explained_variance = float(np.mean([
                    global_explained_variance,
                    task1_explained_variance,
                    task2_explained_variance,
                ]))
                critic_diagnostics = {
                    "global_critic_loss": global_critic_loss,
                    "task1_critic_loss": task1_critic_loss,
                    "task2_critic_loss": task2_critic_loss,
                    "global_critic_grad_norm": global_critic_grad_norm,
                    "task1_critic_grad_norm": task1_critic_grad_norm,
                    "task2_critic_grad_norm": task2_critic_grad_norm,
                    "global_explained_variance": global_explained_variance,
                    "task1_explained_variance": task1_explained_variance,
                    "task2_explained_variance": task2_explained_variance,
                }
        self.policy_version += 1
        self.update_count += 1
        self.environment_steps += int(batch.size)

        def per_agent_mean(records: List[List[float]]) -> List[float]:
            return [float(np.mean(values)) for values in records]

        diagnostics: Dict[str, object] = {
            "algorithm": "mappo",
            "mappo_variant": self.variant,
            "update": int(self.update_count),
            "policy_version": int(self.policy_version),
            "rollout_steps": int(batch.size),
            "environment_steps": int(self.environment_steps),
            "ppo_epochs": int(self.config.mappo_ppo_epochs),
            "actor_loss_per_agent": per_agent_mean(actor_loss_records),
            "critic_loss": critic_loss,
            "entropy_rb_per_agent": per_agent_mean(entropy_rb_records),
            "entropy_mode_per_agent": per_agent_mean(entropy_mode_records),
            "entropy_power_per_agent": per_agent_mean(entropy_power_records),
            "approx_kl_per_agent": per_agent_mean(kl_records),
            "clip_fraction_per_agent": per_agent_mean(clip_records),
            "actor_grad_norm_per_agent": per_agent_mean(actor_grad_records),
            "critic_grad_norm": critic_grad_norm,
            "explained_variance": explained_variance,
            "critic_loss_aggregate_semantics": (
                "mean_over_epochs_of_global_plus_task1_plus_task2_losses"
                if self.variant == "tdec"
                else "mean_over_epochs_of_combined_critic_loss"
            ),
            "critic_grad_norm_aggregate_semantics": (
                "mean_over_epochs_of_joint_rss_global_task1_task2_grad_l2_before_clipping"
                if self.variant == "tdec"
                else "mean_over_epochs_of_combined_critic_grad_l2_before_clipping"
            ),
            "explained_variance_aggregate_semantics": (
                "unweighted_mean_of_global_task1_task2_explained_variance"
                if self.variant == "tdec"
                else "mean_over_valid_per_agent_combined_value_outputs"
            ),
            "advantage_mean_per_agent_before_normalization": advantage_mean.squeeze(0).detach().cpu().tolist(),
            "advantage_std_per_agent_before_normalization": advantage_std.squeeze(0).detach().cpu().tolist(),
            **critic_diagnostics,
            **component_diagnostics,
        }
        numeric = [
            value
            for key, value in diagnostics.items()
            if key not in {"algorithm", "mappo_variant"}
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
        if self.variant == "combined":
            critic_count = sum(parameter.numel() for parameter in self.critic.parameters())
            return {
                "actors": int(actor_count),
                "critic": int(critic_count),
                "total": int(actor_count + critic_count),
            }
        global_count = sum(parameter.numel() for parameter in self.global_critic.parameters())
        task1_count = sum(parameter.numel() for parameter in self.task1_critics.parameters())
        task2_count = sum(parameter.numel() for parameter in self.task2_critics.parameters())
        critic_count = global_count + task1_count + task2_count
        return {
            "actors": int(actor_count),
            "critic": int(critic_count),
            "global_critic": int(global_count),
            "task1_critics": int(task1_count),
            "task2_critics": int(task2_count),
            "total": int(actor_count + critic_count),
        }

    @property
    def critic_structure(self) -> str:
        if self.variant == "combined":
            return "joint_observation_per_agent_value"
        return "joint_observation_scalar_global_plus_independent_local_task1_task2_values"
