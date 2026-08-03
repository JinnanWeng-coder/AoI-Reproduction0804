"""Emit a concrete global-gradient repair audit artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import resolve_config
from global_critic import Global_Critic
from local_critic import Agent


def build(mode):
    config = resolve_config("paper_faithful", "p05_n04_g25", seed=41, episodes=2, steps_per_episode=3, device="cpu", actor_hidden=[16, 8], local_critic_hidden=[16, 8], global_critic_hidden=[16, 8, 4], batch_size=4, replay_capacity=32, global_update_mode=mode)
    config.device_resolved = "cpu"
    agents = [Agent(config, index) for index in range(config.number_agents)]
    learner = Global_Critic(config, agents)
    rng = np.random.default_rng(41)
    states = rng.normal(size=(4, config.state_dim * config.number_agents)).astype(np.float32)
    return config, agents, learner, states


def flatten_parameters(modules):
    return torch.cat([parameter.detach().flatten() for module in modules for parameter in module.parameters()])


def main():
    sync_config, sync_agents, sync_learner, states = build("synchronous_joint")
    sync_audit = sync_learner.actor_global_gradient_audit(states)
    critic_modules = [sync_learner.global_critic1, sync_learner.global_critic2]
    for agent in sync_agents:
        critic_modules.extend([agent.critic_task1, agent.critic_task2])
    before_critics = flatten_parameters(critic_modules).clone()
    actor_step = sync_learner._actor_step(torch.as_tensor(states, dtype=torch.float32))
    after_critics = flatten_parameters(critic_modules).clone()
    legacy_config, _legacy_agents, legacy_learner, _legacy_states = build("legacy_detach")
    legacy_audit = legacy_learner.actor_global_gradient_audit(states)
    result = {
        "synchronous_joint": sync_audit,
        "legacy_detach": legacy_audit,
        "actor_step": actor_step,
        "critic_parameter_delta_during_actor_step": float(torch.linalg.vector_norm(after_critics - before_critics)),
        "pass": bool(
            sync_audit["finite"]
            and all(value > 0 for value in sync_audit["global_gradient_norms"])
            and all(value == 0 for value in legacy_audit["global_gradient_norms"])
            and all(value > 0 for value in actor_step["actor_parameter_deltas"])
            and torch.equal(before_critics, after_critics)
        ),
    }
    output = ROOT / "audit" / "gradient_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

