import numpy as np
import pytest

from Classes.Environment_Platoon import PaperEnviron, compute_global_reward, power_penalty
from config import resolve_config


def _config(**overrides):
    values = {"seed": 53, "episodes": 2, "steps_per_episode": 4}
    values.update(overrides)
    return resolve_config("paper_faithful", "p05_n04_g25", **values)


def test_urban_grid_lane_constants_shadowing_and_exit_behavior():
    env = PaperEnviron(_config())
    env.reset_world(7)
    lanes = env._lane_sets()
    assert lanes["u"] == [1.75, 5.25, 251.75, 255.25, 501.75, 505.25]
    assert lanes["d"] == [244.75, 248.25, 494.75, 498.25, 744.75, 748.25]
    assert lanes["l"] == [1.75, 5.25, 434.75, 438.25, 867.75, 871.25]
    assert lanes["r"] == [427.75, 431.25, 860.75, 864.75, 1293.75, 1297.25]

    initial_shadowing = env.v2v_shadowing.copy()
    env._renew_channel()
    assert not np.array_equal(initial_shadowing, env.v2v_shadowing)
    env.reset_world(7)
    np.testing.assert_array_equal(initial_shadowing, env.v2v_shadowing)

    leader = env.vehicles[0]
    leader.direction = "d"
    leader.position = [leader.position[0], 0.5]
    for follower in env.vehicles[1 : env.size_platoon]:
        follower.position = list(leader.position)
    env._renew_positions()
    assert leader.direction == "l"
    assert leader.position[1] == pytest.approx(lanes["l"][0])
    assert all(vehicle.direction == "l" for vehicle in env.vehicles[: env.size_platoon])
    assert all(np.isfinite(vehicle.position).all() for vehicle in env.vehicles[: env.size_platoon])


def test_per_rb_interference_and_mode_receiver_selection_are_real():
    env = PaperEnviron(_config())
    env.reset_world(8)
    actions = np.full((env.n_platoon, 3), -1.0, dtype=np.float32)
    actions[1, 0] = 0.0  # one other transmitter on RB 1
    decoded = env.decode_actions(actions)
    metrics = env._compute_metrics(decoded)
    assert metrics["v2i_interference_linear"].shape == (env.n_platoon, env.n_rb)
    assert metrics["v2v_interference_linear"].shape == (env.n_platoon, env.size_platoon - 1, env.n_rb)
    assert metrics["v2i_interference_linear"][0, 1] > env.sig2

    actions[0, 1] = 1.0  # victim 0 now selects V2V receiver mode
    decoded = env.decode_actions(actions)
    metrics = env._compute_metrics(decoded)
    expected = np.maximum(metrics["v2v_interference_linear"][0].max(axis=0), env.sig2)
    np.testing.assert_allclose(metrics["interference_linear"][0], expected)


def test_global_reward_hand_fixture_and_eq16_diagnostic():
    linear = np.asarray([[1e-6, 1e-5], [1e-4, 1e-3]], dtype=np.float64)
    db = 10.0 * np.log10(linear)
    normalized = (db + 60.0) / 60.0
    expected_mean = -float(normalized.mean())
    expected_sum = -float(normalized.sum())
    assert compute_global_reward(linear, "source_normalized_per_rb_mean") == pytest.approx(expected_mean)
    assert compute_global_reward(linear, "eq16_sum") == pytest.approx(expected_sum)


def test_power_penalty_is_finite_nonnegative_and_monotone():
    values = [power_penalty(power) for power in (0.0, 1.0, 5.0, 25.0, 30.0)]
    assert all(np.isfinite(values))
    assert all(value >= 0.0 for value in values)
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0)


def test_zero_dbm_diagnostic_interval_remains_supported():
    config = _config(power_min_dbm=0.0)
    env = PaperEnviron(config)
    decoded = env.decode_actions(np.full((config.number_agents, 3), -1.0, dtype=np.float32))
    np.testing.assert_allclose(decoded[:, 2], 0.0)
