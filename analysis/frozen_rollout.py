"""Shared sequential-warm lifecycle for existing MADDPG and nonlearning deployments."""
from __future__ import annotations

import numpy as np


def sequential_warm(environment, worlds, action_factory, *, steps_per_episode, scored_episodes,
                    warmup_episodes, deployment=False, probe=None, after_warmup=None):
    """Cold-reset once/world, keep native start_episode semantics, never score warmup.

    action_factory(world) returns an action callable(observations). It owns its
    private policy RNG; the lifecycle never reseeds a policy/global/physical RNG.
    """
    raw_records, exogenous = [], []
    for world in worlds:
        environment.reset_world(int(world))
        initial = probe(environment) if deployment else None
        act = action_factory(int(world))
        for warmup_index in range(int(warmup_episodes)):
            observations = environment.start_episode(warmup_index)
            for _slot in range(int(steps_per_episode)):
                observations, _rg, _t1, _t2, _done, _info = environment.step(act(observations))
        if after_warmup is not None:
            after_warmup(int(world), environment)
        world_records = []
        for episode in range(int(scored_episodes)):
            observations = environment.start_episode(int(warmup_episodes) + episode)
            episode_records = []
            for _slot in range(int(steps_per_episode)):
                actions = act(observations)
                observations, rg, t1, t2, _done, info = environment.step(actions)
                episode_records.append({
                    "aoi_ms": np.asarray(info["aoi_ms"], dtype=np.float32),
                    "success": np.asarray(info["success"], dtype=np.float32),
                    "remaining_demand": np.asarray(info["remaining_demand"], dtype=np.float32),
                    "power_dbm": np.asarray(info["power_dbm"], dtype=np.float32),
                    "rb": np.asarray(info["rb"], dtype=np.int64),
                    "mode": np.asarray(info["mode"], dtype=np.int64),
                    "action_normalized": np.asarray(actions, dtype=np.float32),
                    "reward_global": np.float32(rg),
                    "reward_task1": np.asarray(t1, dtype=np.float32),
                    "reward_task2": np.asarray(t2, dtype=np.float32),
                    **({"v2i_rate": np.asarray(info["v2i_rate"], dtype=np.float32)} if deployment else {}),
                })
            world_records.append(episode_records)
        raw_records.append(world_records)
        if deployment:
            exogenous.append({"world": world, "initial": initial, "final": probe(environment)})
    return raw_records, exogenous


def arrays_from_records(records, deployment=False):
    # Same axes/dtypes as evaluate_maddpg_e1's historical frozen writer.
    keys = {"aoi_ms": np.float32, "success": np.float32, "remaining_demand": np.float32,
            "power_dbm": np.float32, "rb": np.int64, "mode": np.int64, "action_normalized": np.float32,
            "reward_global": np.float32, "reward_task1": np.float32, "reward_task2": np.float32}
    if deployment: keys["v2i_rate"] = np.float32
    arrays = {key: np.asarray([[[s[key] for s in ep] for ep in world] for world in records], dtype=dtype)
              for key, dtype in keys.items()}
    if deployment:
        arrays["reset_event"] = np.isclose(arrays["aoi_ms"], 1, rtol=0, atol=1e-6)
        arrays["v2i_attempt"] = arrays["mode"] == 0
    return arrays
