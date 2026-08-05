import importlib.util
from pathlib import Path

import numpy as np

from Classes.legacy_adapter import LegacyEnviron, legacy_lanes
from config import resolve_config


TARGET = Path(__file__).resolve().parents[1]
LEGACY_ENV_PATH = TARGET / "legacy_reference" / "Classes" / "Environment_Platoon.py"


def _load_original():
    spec = importlib.util.spec_from_file_location("golden_legacy_environment", LEGACY_ENV_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _original_trace(config, normalized_actions):
    module = _load_original()
    np.random.seed(config.seed)
    down, up, left, right = legacy_lanes()
    env = module.Environ(down, up, left, right, config.map_width_m, config.map_height_m, config.number_vehicles, config.scenario.platoon_size, config.n_rb, config.v2i_min_bits_per_step, config.bandwidth_hz, config.cam_bits, config.scenario.gap_m)
    env.new_random_game()
    env.V2V_demand = env.V2V_demand_size * np.ones(config.number_agents, dtype=np.float16)
    env.individual_time_limit = env.time_slow * np.ones(config.number_agents, dtype=np.float16)
    env.active_links = np.ones(config.number_agents, dtype=bool)
    env.AoI = np.ones(config.number_agents) * 100
    env.renew_positions()
    env.renew_channel(config.number_vehicles, config.scenario.platoon_size)
    env.renew_channels_fastfading()
    clipped = np.clip(normalized_actions, -0.999, 0.999)
    decoded = np.zeros(clipped.shape, dtype=np.int64)
    decoded[:, 0] = ((clipped[:, 0] + 1) / 2) * config.n_rb
    decoded[:, 1] = ((clipped[:, 1] + 1) / 2) * config.n_modes
    decoded[:, 2] = np.round(np.clip(((clipped[:, 2] + 1) / 2) * config.power_max_dbm, 1, config.power_max_dbm))
    t1, t2, global_reward, aoi, c_rate, v_rate, demand, success = env.act_for_training(decoded.copy())
    env.renew_channels_fastfading()
    env.Compute_Interference(decoded.copy())
    return env, decoded, (t1, t2, global_reward, aoi, c_rate, v_rate, demand, success)


def test_legacy_golden_trace_matches_original_source():
    config = resolve_config("legacy_release", "p05_n04_g25", seed=23, episodes=2, steps_per_episode=3)
    actions = np.asarray([
        [-0.8, -0.7, -0.5],
        [-0.2, 0.4, 0.1],
        [0.2, -0.3, 0.8],
        [0.7, 0.8, -0.1],
        [-0.4, 0.1, 0.3],
    ], dtype=np.float32)
    original_env, decoded, original = _original_trace(config, actions)
    np.random.seed(config.seed)
    adapter = LegacyEnviron(config)
    adapter.reset_episode(0)
    reproduced = adapter.step(actions)
    expected_state = np.asarray([adapter._state_for_agent(index) for index in range(config.number_agents)])
    np.testing.assert_array_equal(expected_state, reproduced[0])
    # The copied source and the adapter must expose the same post-step state,
    # including the original scalar-interference layout.
    original_state = np.asarray([
        (lambda idx: np.concatenate((
            np.reshape((original_env.V2I_channels_abs[idx * config.scenario.platoon_size] - 60) / 60.0, -1),
            np.reshape((original_env.V2I_channels_with_fastfading[idx * config.scenario.platoon_size, :] - original_env.V2I_channels_abs[idx * config.scenario.platoon_size] + 10) / 35, -1),
            np.reshape((original_env.V2V_channels_abs[idx * config.scenario.platoon_size, idx * config.scenario.platoon_size + 1 + np.arange(config.scenario.platoon_size - 1)] - 60) / 60.0, -1),
            np.reshape((original_env.V2V_channels_with_fastfading[idx * config.scenario.platoon_size, idx * config.scenario.platoon_size + 1 + np.arange(config.scenario.platoon_size - 1), :] - original_env.V2V_channels_abs[idx * config.scenario.platoon_size, idx * config.scenario.platoon_size + 1 + np.arange(config.scenario.platoon_size - 1)].reshape(config.scenario.platoon_size - 1, 1) + 10) / 35, -1),
            np.reshape((original_env.Interference_all[idx] + 60) / 60.0, -1),
            np.reshape(original_env.AoI[idx] / int(original_env.time_slow / original_env.time_fast), -1),
            np.asarray([original_env.V2V_demand[idx] / original_env.V2V_demand_size]),
        )))(idx)
        for idx in range(config.number_agents)
    ])
    np.testing.assert_array_equal(original_state, reproduced[0])
    for expected, actual in zip(original[:2], reproduced[2:4]):
        np.testing.assert_allclose(expected, actual, rtol=0, atol=0)
    np.testing.assert_allclose(original[2], reproduced[1], rtol=0, atol=0)
    np.testing.assert_allclose(original[3], reproduced[5]["aoi_ms"], rtol=0, atol=0)
    np.testing.assert_allclose(original[4], reproduced[5]["v2i_rate"], rtol=0, atol=0)
    np.testing.assert_allclose(original[5], reproduced[5]["v2v_rate"], rtol=0, atol=0)
    np.testing.assert_allclose(original[6], reproduced[5]["remaining_demand"], rtol=0, atol=0)
    np.testing.assert_allclose(original[7], reproduced[5]["success_rate"], rtol=0, atol=0)
    np.testing.assert_array_equal((~original_env.active_links).astype(np.float32), reproduced[5]["success"])
    np.testing.assert_allclose(original_env.Interference_all, adapter.Interference_all, rtol=0, atol=0)
