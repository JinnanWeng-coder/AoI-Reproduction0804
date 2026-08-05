import numpy as np
import pytest

from metrics import MetricStore


def test_metrics_save_global_combined_and_immediate_reward_proxy():
    store = MetricStore(number_agents=2, steps_per_episode=2, global_actor_weight=2.0)
    task1 = [np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])]
    task2 = [np.asarray([0.5, 1.5]), np.asarray([2.5, 3.5])]
    global_rewards = [10.0, 20.0]
    info = [{"success": np.asarray([1.0, 0.0])}, {"success": np.asarray([0.0, 1.0])}]
    store.append_episode(info, task1, task2, global_rewards)
    arrays = store.arrays()
    np.testing.assert_array_equal(arrays["local_total_episode_mean"], [[3.5, 5.5]])
    np.testing.assert_array_equal(arrays["global_episode_sum"], [30.0])
    np.testing.assert_array_equal(arrays["global_episode_mean"], [15.0])
    np.testing.assert_array_equal(arrays["immediate_reward_proxy"], [[33.5, 35.5]])


def test_endpoint_worst_agent_and_post_map_action_diagnostics():
    store = MetricStore(
        number_agents=2,
        steps_per_episode=3,
        n_rb=2,
        n_modes=2,
        power_min_dbm=1.0,
        power_max_dbm=30.0,
    )
    info = [
        {"aoi_ms": np.asarray([10.0, 40.0]), "success": np.asarray([0.0, 0.0]), "mode": np.asarray([0, 1]), "rb": np.asarray([0, 1]), "power_dbm": np.asarray([1.0, 30.0]), "action_post_clip_normalized": np.asarray([[0.0, 0.0, -0.99], [0.0, 0.0, 0.99]])},
        {"aoi_ms": np.asarray([20.0, 50.0]), "success": np.asarray([1.0, 0.0]), "mode": np.asarray([1, 1]), "rb": np.asarray([1, 1]), "power_dbm": np.asarray([2.0, 30.0]), "action_post_clip_normalized": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.99]])},
        {"aoi_ms": np.asarray([30.0, 60.0]), "success": np.asarray([1.0, 1.0]), "mode": np.asarray([1, 0]), "rb": np.asarray([1, 0]), "power_dbm": np.asarray([30.0, 1.0]), "action_post_clip_normalized": np.asarray([[0.0, 0.0, 0.99], [0.0, 0.0, -0.99]])},
    ]
    zeros = [np.zeros(2, dtype=np.float32) for _ in range(3)]
    store.append_episode(info, zeros, zeros, [0.0, 0.0, 0.0])
    arrays = store.arrays()
    np.testing.assert_array_equal(arrays["mean_aoi_ms_episode_agent"], [[20.0, 50.0]])
    np.testing.assert_array_equal(arrays["endpoint_cam_episode_agent"], [[1.0, 1.0]])
    np.testing.assert_array_equal(arrays["worst_agent_mean_aoi_ms_episode"], [50.0])
    np.testing.assert_allclose(arrays["mode_switch_rate_episode_agent"], [[0.5, 0.5]])
    np.testing.assert_allclose(arrays["power_post_map_near_min_fraction_episode_agent"], [[1 / 3, 1 / 3]])
    np.testing.assert_allclose(arrays["power_action_post_clip_near_min_fraction_episode_agent"], [[1 / 3, 1 / 3]])


def test_gradient_diagnostics_are_episode_aggregated_and_finite(tmp_path):
    store = MetricStore(number_agents=2, steps_per_episode=1, diagnostics=True)
    info = [{"aoi_ms": np.asarray([1.0, 2.0]), "success": np.asarray([1.0, 0.0])}]
    store.append_episode(info, [np.zeros(2)], [np.zeros(2)], [0.0])
    gradient = {field: [1.0, 2.0] for field in store.GRADIENT_FIELDS}
    gradient.update({"finite": True, "mode": "synchronous_joint", "global_contributes_to_actor": True})
    store.append_learning_episode([{
        "actor_loss": 1.0,
        "global_critic_loss": 2.0,
        "local_critic_loss": [3.0, 4.0],
        "global_actor_gradient_norms": [1.0, 2.0],
        "actor_parameter_deltas": [0.1, 0.2],
        "actor_gradient_diagnostics": gradient,
        "learn_step": 1,
        "global_target_update": True,
        "local_target_update": True,
    }])
    store.save(tmp_path)
    assert not (tmp_path / "train_metrics.mat").exists()
    with np.load(tmp_path / "diagnostics" / "actor_gradient_episode.npz", allow_pickle=False) as arrays:
        assert arrays["actor_update_count"].tolist() == [1]
        assert arrays["all_finite"].tolist() == [True]
        assert arrays["finite_fraction"].tolist() == [1.0]
        assert np.all(np.isfinite(arrays["global_vs_local_cosine_mean"]))
        assert arrays["task1_grad_l2_mean"].shape == (1, 2)


def test_nonfinite_gradient_diagnostics_stop_the_diagnostic_run():
    store = MetricStore(number_agents=1, steps_per_episode=1, diagnostics=True)
    store.append_episode([{"aoi_ms": np.asarray([1.0]), "success": np.asarray([1.0])}], [np.zeros(1)], [np.zeros(1)], [0.0])
    gradient = {field: [1.0] for field in store.GRADIENT_FIELDS}
    gradient.update({"finite": False, "mode": "synchronous_joint", "global_contributes_to_actor": True})
    with pytest.raises(FloatingPointError, match="actor-gradient diagnostics"):
        store.append_learning_episode([{
            "actor_loss": 1.0,
            "actor_gradient_diagnostics": gradient,
            "learn_step": 1,
            "global_target_update": True,
            "local_target_update": True,
        }])
