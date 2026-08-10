import numpy as np

from aoi_v2x_reproduction.envs.platoon import PaperEnviron
from aoi_v2x_reproduction.config import resolve_config


def _config(**overrides):
    values = {"seed": 41, "episodes": 3, "steps_per_episode": 4}
    values.update(overrides)
    return resolve_config(scenario="p05_n04_g25", **values)


def test_semantic_version_and_lifecycle_defaults_are_explicit():
    baseline = _config()
    assert baseline.semantic_version == "reproduction_baseline_v1"
    assert baseline.mobility_revision == "lane_graph_exit_safe_v1"
    assert baseline.tau == 0.005
    assert baseline.initial_aoi_ms == 100.0
    assert baseline.eval_protocol == "sequential_warm"
    assert baseline.eval_warmup_episodes == 5
    assert baseline.global_reward_normalization == "source_normalized_per_rb_mean"
    assert baseline.mobility_model == "urban_grid_correlated"
    assert baseline.gap_definition == "bumper_to_bumper"
    assert baseline.vehicle_length_m == 4.0
    assert baseline.effective_center_spacing_m == 29.0


def test_start_episode_preserves_aoi_and_previous_interference():
    config = _config()
    env = PaperEnviron(config)
    env.reset_world(91)
    actions = np.zeros((config.number_agents, 3), dtype=np.float32)
    env.step(actions)
    aoi = env.aoi.copy()
    previous = env.previous_interference.copy()
    env.v2v_demand.fill(0.0)
    env.individual_time_limit.fill(0.0)
    env.active_links[:] = False

    env.start_episode(1, update_mobility=False)

    np.testing.assert_array_equal(env.aoi, aoi)
    np.testing.assert_array_equal(env.previous_interference, previous)
    np.testing.assert_array_equal(env.v2v_demand, config.cam_bits)
    np.testing.assert_array_equal(env.active_links, True)
    np.testing.assert_allclose(env.individual_time_limit, config.steps_per_episode * config.slot_ms / 1000.0)


def test_cold_reset_recreates_the_same_world_without_episode_double_advance():
    config = _config(slow_update_every_episodes=1)
    env = PaperEnviron(config)
    env.reset_world(123)
    initial = np.asarray([vehicle.position for vehicle in env.vehicles])
    env.start_episode(0)
    np.testing.assert_array_equal(np.asarray([vehicle.position for vehicle in env.vehicles]), initial)
    env.start_episode(1)
    advanced_once = np.asarray([vehicle.position for vehicle in env.vehicles])

    env.reset_world(123)
    np.testing.assert_array_equal(np.asarray([vehicle.position for vehicle in env.vehicles]), initial)
    env.start_episode(1)
    np.testing.assert_array_equal(np.asarray([vehicle.position for vehicle in env.vehicles]), advanced_once)
