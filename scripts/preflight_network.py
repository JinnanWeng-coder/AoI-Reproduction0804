"""Instantiate the largest configured network and perform two learner updates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from global_critic import Global_Critic
from local_critic import Agent
from config import resolve_config
from runner import resolve_device, seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="p05_n10_g25")
    parser.add_argument("--profile", default="paper_faithful")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    config = resolve_config(
        args.profile,
        args.scenario,
        seed=97,
        episodes=1,
        steps_per_episode=2,
        batch_size=args.batch_size,
        replay_capacity=args.batch_size,
        device=args.device,
        smoke=True,
        is_formal_result=False,
    )
    device = resolve_device(config.device)
    config.device_resolved = str(device)
    seed_everything(config.seed, device)
    agents = [Agent(config, index) for index in range(config.number_agents)]
    learner = Global_Critic(config, agents)
    rng = np.random.default_rng(config.seed)
    state_width = config.state_dim * config.number_agents
    action_width = config.action_dim * config.number_agents
    batch = (
        rng.normal(size=(args.batch_size, state_width)).astype(np.float32),
        rng.uniform(-1.0, 1.0, size=(args.batch_size, action_width)).astype(np.float32),
        rng.normal(size=args.batch_size).astype(np.float32),
        rng.normal(size=(args.batch_size, config.number_agents)).astype(np.float32),
        rng.normal(size=(args.batch_size, config.number_agents)).astype(np.float32),
        rng.normal(size=(args.batch_size, state_width)).astype(np.float32),
        np.zeros(args.batch_size, dtype=np.float32),
    )
    diagnostics = [learner.learn(batch), learner.learn(batch)]
    if diagnostics[0]["global_target_update"] is not True or diagnostics[1]["global_target_update"] is not True:
        raise RuntimeError("global target critics were not updated on both learner steps")
    if diagnostics[0]["local_target_update"] is not False or diagnostics[1]["local_target_update"] is not True:
        raise RuntimeError("policy-delay cadence did not update local targets on step 2 only")
    result = {
        "status": "pass",
        "profile": config.profile,
        "semantic_version": config.semantic_version,
        "scenario": config.scenario.id,
        "batch_size": args.batch_size,
        "device": str(device),
        "learn_steps": [item["learn_step"] for item in diagnostics],
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        "global_target_updates": [item["global_target_update"] for item in diagnostics],
        "local_target_updates": [item["local_target_update"] for item in diagnostics],
        "gap_definition": config.gap_definition,
        "vehicle_length_m": config.vehicle_length_m,
    }
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite preflight report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
