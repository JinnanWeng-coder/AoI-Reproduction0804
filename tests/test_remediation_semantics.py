import numpy as np
import pytest

from aoi_v2x_reproduction.envs.platoon import PaperEnviron, compute_global_reward, power_penalty
from aoi_v2x_reproduction.config import resolve_config


def _config(**overrides):
    values = {"seed": 53, "episodes": 2, "steps_per_episode": 4}
    values.update(overrides)
    return resolve_config(scenario="p05_n04_g25", **values)


def test_urban_grid_lane_constants_shadowing_and_exit_behavior():
    env = PaperEnviron(_config())
    env.reset_world(7)
    lanes = env._lane_sets()
    assert lanes["u"] == [1.75, 5.25, 251.75, 255.25, 501.75, 505.25]
    assert lanes["d"] == [244.75, 248.25, 494.75, 498.25, 744.75, 748.25]
    assert lanes["l"] == [1.75, 5.25, 434.75, 438.25, 867.75, 871.25]
    assert lanes["r"] == [427.75, 431.25, 860.75, 864.25, 1293.75, 1297.25]

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
    expected_sum = -float(np.log10(linear).sum()) / linear.shape[0]
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


@pytest.mark.parametrize(
    "direction,turn,position",
    [
        ("u", "l", [501.75, 434.35]),
        ("u", "r", [501.75, 427.35]),
        ("d", "l", [501.75, 435.15]),
        ("d", "r", [501.75, 428.15]),
        ("r", "u", [501.35, 649.5]),
        ("r", "d", [244.35, 649.5]),
        ("l", "u", [501.95, 649.5]),
        ("l", "d", [248.65, 649.5]),
    ],
)
def test_all_turns_use_exact_manhattan_path_and_reanchor_followers(direction, turn, position):
    config = _config()
    env = PaperEnviron(config)
    env.reset_world(71)
    env.change_direction_prob = 1.0
    leader = env.vehicles[0]
    leader.direction = direction
    leader.position = list(position)
    for follower in env.vehicles[1 : env.size_platoon]:
        follower.velocity = 999.0

    before = np.asarray(leader.position, dtype=np.float64)
    distance = leader.velocity * env.time_slow
    env._renew_positions()
    after = np.asarray(leader.position, dtype=np.float64)
    trace = env._last_mobility_trace[0]

    assert leader.direction == turn
    assert abs(after[0] - before[0]) + abs(after[1] - before[1]) == pytest.approx(distance)
    assert trace["route_length_m"] == pytest.approx(distance, abs=1e-9)
    assert sum(segment["distance_m"] for segment in trace["route_segments"]) == pytest.approx(distance, abs=1e-9)
    assert all(vehicle.direction == turn for vehicle in env.vehicles[: env.size_platoon])
    assert all(vehicle.velocity == pytest.approx(leader.velocity) for vehicle in env.vehicles[: env.size_platoon])
    spacing = config.effective_center_spacing_m
    for index, follower in enumerate(env.vehicles[1 : env.size_platoon], start=1):
        delta = np.asarray(follower.position) - after
        assert np.linalg.norm(delta) == pytest.approx(index * spacing)


@pytest.mark.parametrize(
    "direction,position",
    [
        ("u", [501.75, 600.0]),
        ("d", [501.75, 600.0]),
        ("l", [600.0, 434.75]),
        ("r", [600.0, 431.25]),
    ],
)
def test_straight_lane_graph_routes_consume_exact_distance(direction, position):
    env = PaperEnviron(_config())
    env.reset_world(72)
    env.change_direction_prob = 0.0
    leader = env.vehicles[0]
    leader.direction = direction
    leader.position = list(position)
    distance = leader.velocity * env.time_slow
    env._renew_positions()
    trace = env._last_mobility_trace[0]
    assert not any(event["type"] == "exit" for event in trace["events"])
    assert trace["route_length_m"] == pytest.approx(distance, abs=1e-9)
    assert sum(segment["distance_m"] for segment in trace["route_segments"]) == pytest.approx(distance, abs=1e-9)
    assert leader.direction == direction


@pytest.mark.parametrize(
    "direction,position,expected",
    [
        ("u", [501.75, 1297.0], "r"),
        ("d", [501.75, 2.0], "l"),
        ("l", [2.0, 434.75], "u"),
        ("r", [748.0, 431.25], "d"),
    ],
)
def test_exit_routes_use_explicit_lane_graph_mapping(direction, position, expected):
    env = PaperEnviron(_config())
    env.reset_world(73)
    env.change_direction_prob = 0.0
    leader = env.vehicles[0]
    leader.direction = direction
    leader.position = list(position)
    distance = leader.velocity * env.time_slow
    env._renew_positions()
    trace = env._last_mobility_trace[0]
    assert leader.direction == expected
    assert any(event["type"] == "exit" and event["to"] == expected for event in trace["events"])
    assert trace["route_length_m"] == pytest.approx(distance, abs=1e-9)
    assert sum(segment["distance_m"] for segment in trace["route_segments"]) == pytest.approx(distance, abs=1e-9)
    graph = env._graph_bounds(env._lane_sets())
    assert graph["x_min"] <= leader.position[0] <= graph["x_max"]
    assert graph["y_min"] <= leader.position[1] <= graph["y_max"]


def test_lane_graph_mobility_is_fixed_seed_reproducible():
    first = PaperEnviron(_config(slow_update_every_episodes=1))
    second = PaperEnviron(_config(slow_update_every_episodes=1))
    first.reset_world(74)
    second.reset_world(74)
    first.start_episode(1)
    second.start_episode(1)
    np.testing.assert_allclose(
        np.asarray([vehicle.position for vehicle in first.vehicles]),
        np.asarray([vehicle.position for vehicle in second.vehicles]),
    )
    assert first._last_mobility_trace == second._last_mobility_trace
