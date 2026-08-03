"""Thin adapter around the byte-preserved public legacy environment.

The adapter deliberately calls the original methods in the original order so
that the golden trace remains a meaningful compatibility check. It only adds
seed/reset/state plumbing and avoids importing the original Main.py, whose
module-level code would start a 500-episode training run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _load_legacy_module():
    path = Path(__file__).resolve().parents[1] / "legacy_reference" / "Classes" / "Environment_Platoon.py"
    spec = importlib.util.spec_from_file_location("legacy_reference_environment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load legacy environment: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_lanes():
    up_lanes = [i / 2.0 for i in [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
    down_lanes = [i / 2.0 for i in [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2, 750 - 3.5 / 2]]
    left_lanes = [i / 2.0 for i in [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
    right_lanes = [i / 2.0 for i in [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2, 1299 - 3.5 / 2]]
    return down_lanes, up_lanes, left_lanes, right_lanes


class LegacyEnviron:
    """Compatibility profile using the exact copied source environment."""

    def __init__(self, config):
        module = _load_legacy_module()
        down, up, left, right = legacy_lanes()
        self.config = config
        self._module = module
        self._env = module.Environ(
            down,
            up,
            left,
            right,
            config.map_width_m,
            config.map_height_m,
            config.number_vehicles,
            config.scenario.platoon_size,
            config.n_rb,
            config.v2i_min_bits_per_step,
            config.bandwidth_hz,
            config.cam_bits,
            config.scenario.gap_m,
        )
        self.step_count = 0
        self.reset(config.seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def state_dim(self) -> int:
        return 1 + self.config.n_rb + (self.config.scenario.platoon_size - 1) + (self.config.scenario.platoon_size - 1) * self.config.n_rb + 1 + 1

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(int(seed))
        self._env.new_random_game()
        self.step_count = 0
        return self.get_observations()

    def reset_episode(self, episode_index: int):
        p = self.config.number_agents
        self._env.V2V_demand = self._env.V2V_demand_size * np.ones(p, dtype=np.float16)
        self._env.individual_time_limit = self._env.time_slow * np.ones(p, dtype=np.float16)
        self._env.active_links = np.ones(p, dtype=bool)
        if episode_index == 0:
            self._env.AoI = np.ones(p) * 100
        if episode_index % self.config.slow_update_every_episodes == 0:
            self._env.renew_positions()
            self._env.renew_channel(self.config.number_vehicles, self.config.scenario.platoon_size)
            self._env.renew_channels_fastfading()
        self.step_count = 0
        return self.get_observations()

    def decode_actions(self, actions: np.ndarray) -> np.ndarray:
        # Preserve the actor's float32 arithmetic; the public Main.py received
        # torch.float32 actions before applying np.clip and np.round.
        actions = np.asarray(actions).reshape(self.config.number_agents, 3)
        clipped = np.clip(actions, -0.999, 0.999)
        # Main.py's original action_all_training array used np.int, so every
        # discrete assignment was cast to integer before the environment saw it.
        decoded = np.zeros(clipped.shape, dtype=np.int64)
        decoded[:, 0] = ((clipped[:, 0] + 1.0) / 2.0) * self.config.n_rb
        decoded[:, 1] = ((clipped[:, 1] + 1.0) / 2.0) * self.config.n_modes
        decoded[:, 2] = np.round(np.clip(((clipped[:, 2] + 1.0) / 2.0) * self.config.power_max_dbm, 1, self.config.power_max_dbm))
        return decoded

    def _state_for_agent(self, idx: int) -> np.ndarray:
        n = self.config.scenario.platoon_size
        leader = idx * n
        followers = leader + 1 + np.arange(n - 1)
        env = self._env
        v2i_abs = (env.V2I_channels_abs[leader] - 60) / 60.0
        v2v_abs = (env.V2V_channels_abs[leader, followers] - 60) / 60.0
        v2i_fast = (env.V2I_channels_with_fastfading[leader, :] - env.V2I_channels_abs[leader] + 10) / 35
        v2v_fast = (env.V2V_channels_with_fastfading[leader, followers, :] - env.V2V_channels_abs[leader, followers].reshape(n - 1, 1) + 10) / 35
        interference = (env.Interference_all[idx] + 60) / 60.0
        aoi = env.AoI[idx] / int(env.time_slow / env.time_fast)
        load = env.V2V_demand[idx] / env.V2V_demand_size
        return np.concatenate((
            np.reshape(v2i_abs, -1),
            np.reshape(v2i_fast, -1),
            np.reshape(v2v_abs, -1),
            np.reshape(v2v_fast, -1),
            np.reshape(interference, -1),
            np.reshape(aoi, -1),
            np.asarray([load]),
        ))

    def get_observations(self) -> np.ndarray:
        return np.asarray([self._state_for_agent(i) for i in range(self.config.number_agents)])

    def step(self, actions: np.ndarray):
        decoded = self.decode_actions(actions)
        t1, t2, global_reward, aoi, c_rate, v_rate, demand, success = self._env.act_for_training(decoded.copy())
        self._env.renew_channels_fastfading()
        self._env.Compute_Interference(decoded.copy())
        self.step_count += 1
        terminated = self.step_count >= self.config.steps_per_episode
        info: Dict[str, Any] = {
            "actions_decoded": decoded.copy(),
            "aoi_ms": np.asarray(aoi).copy(),
            "remaining_demand": np.asarray(demand).copy(),
            "v2i_rate": np.asarray(c_rate).copy(),
            "v2v_rate": np.asarray(v_rate).copy(),
            "interference_db": np.asarray(self._env.Interference_all).copy(),
            # The public source exposes one aggregate success rate. Keep that
            # scalar under success_rate and broadcast it for the common raw
            # per-platoon metric schema used by the result auditor.
            "success": np.full(self.config.number_agents, float(success)),
            "success_rate": np.asarray(success),
        }
        return self.get_observations(), float(global_reward), np.asarray(t1), np.asarray(t2), terminated, info

    def state_dict(self):
        env = self._env
        vehicle_state = [(list(vehicle.position), vehicle.direction, vehicle.velocity) for vehicle in env.vehicles]
        state = {
            "vehicles": vehicle_state,
            "V2V_Shadowing": np.asarray(env.V2V_Shadowing).copy(),
            "V2I_Shadowing": np.asarray(env.V2I_Shadowing).copy(),
            "delta_distance": np.asarray(env.delta_distance).copy(),
            "V2V_pathloss": np.asarray(env.V2V_pathloss).copy(),
            "V2I_pathloss": np.asarray(env.V2I_pathloss).copy(),
            "V2V_channels_abs": np.asarray(env.V2V_channels_abs).copy(),
            "V2I_channels_abs": np.asarray(env.V2I_channels_abs).copy(),
            "V2V_channels_with_fastfading": np.asarray(env.V2V_channels_with_fastfading).copy(),
            "V2I_channels_with_fastfading": np.asarray(env.V2I_channels_with_fastfading).copy(),
            "V2V_demand": np.asarray(env.V2V_demand).copy(),
            "individual_time_limit": np.asarray(env.individual_time_limit).copy(),
            "active_links": np.asarray(env.active_links).copy(),
            "AoI": np.asarray(env.AoI).copy(),
            "Interference_all": np.asarray(env.Interference_all).copy(),
            "step_count": self.step_count,
        }
        return state

    def load_state_dict(self, state):
        env = self._env
        env.vehicles = [self._module.Vehicle(list(position), direction, velocity) for position, direction, velocity in state["vehicles"]]
        for key in ("V2V_Shadowing", "V2I_Shadowing", "delta_distance", "V2V_pathloss", "V2I_pathloss", "V2V_channels_abs", "V2I_channels_abs", "V2V_channels_with_fastfading", "V2I_channels_with_fastfading", "V2V_demand", "individual_time_limit", "active_links", "AoI", "Interference_all"):
            setattr(env, key, np.asarray(state[key]).copy())
        self.step_count = int(state["step_count"])


def make_legacy_environment(config):
    return LegacyEnviron(config)
