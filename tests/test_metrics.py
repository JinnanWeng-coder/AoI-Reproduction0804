import numpy as np

from metrics import MetricStore


def test_metrics_save_global_combined_and_training_objective_proxy():
    store = MetricStore(number_agents=2, steps_per_episode=2, global_actor_weight=2.0)
    task1 = [np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])]
    task2 = [np.asarray([0.5, 1.5]), np.asarray([2.5, 3.5])]
    global_rewards = [10.0, 20.0]
    info = [{"success": np.asarray([1.0, 0.0])}, {"success": np.asarray([0.0, 1.0])}]
    store.append_episode(info, task1, task2, global_rewards)
    arrays = store.arrays()
    np.testing.assert_array_equal(arrays["local_total_episode_mean"], [[3.5, 5.5]])
    np.testing.assert_array_equal(arrays["global_episode_sum"], [30.0])
    np.testing.assert_array_equal(arrays["training_objective_proxy"], [[63.5, 65.5]])
