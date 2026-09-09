import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.evaluate_mappo_feasibility_reward import (
    ARMS,
    _record_step,
    apply_v2i_power_offset,
    compute_feasibility_snapshot,
    evaluate_feasibility_reward,
    required_power_dbm,
    reward_ledger,
    task_to_arm_seed,
)
from analysis.evaluate_mappo_power_intervention import (
    ARMS as OLD_ARMS,
    evaluate_power_intervention,
)
from analysis.summarize_mappo_feasibility_reward import (
    COMMON_TRAJECTORY_ARRAYS,
    _discounted_return,
    _paired_stream_rows,
    lack_of_opportunity_statistics,
    validate_physical_reconstruction,
)
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.envs.platoon import PaperEnviron
from aoi_v2x_reproduction.runtime.runner import train


def _normalized_power(dbm, low=1.0, high=30.0):
    return 2.0 * (np.asarray(dbm) - low) / (high - low) - 1.0


def _actions():
    return np.asarray([
        [-0.8, -0.5, _normalized_power(28.0)],
        [0.0, 0.5, _normalized_power(2.0)],
        [0.8, -0.5, _normalized_power(10.0)],
        [0.8, 0.5, _normalized_power(20.0)],
    ], dtype=np.float32)


def _environment_actions(number_agents):
    return np.resize(_actions(), (int(number_agents), 3)).astype(np.float32)


def test_five_arm_mapping_is_fixed_and_old_three_arm_interface_is_unchanged():
    assert ARMS == (
        "baseline", "v2i_plus3db", "v2i_minus3db", "v2i_plus6db", "v2i_plus9db"
    )
    assert OLD_ARMS == ("baseline", "v2i_plus3db", "v2v_minus3db")
    assert task_to_arm_seed(0) == ("baseline", 8)
    assert task_to_arm_seed(5) == ("baseline", 13)
    assert task_to_arm_seed(6) == ("v2i_plus3db", 8)
    assert task_to_arm_seed(29) == ("v2i_plus9db", 13)
    with pytest.raises(ValueError):
        task_to_arm_seed(30)


@pytest.mark.parametrize(
    "arm, expected, low_clip, high_clip",
    [
        ("baseline", [28.0, 2.0, 10.0, 20.0], [False] * 4, [False] * 4),
        ("v2i_plus3db", [30.0, 2.0, 13.0, 20.0], [False] * 4, [True, False, False, False]),
        ("v2i_minus3db", [25.0, 2.0, 7.0, 20.0], [False] * 4, [False] * 4),
        ("v2i_plus6db", [30.0, 2.0, 16.0, 20.0], [False] * 4, [True, False, False, False]),
        ("v2i_plus9db", [30.0, 2.0, 19.0, 20.0], [False] * 4, [True, False, False, False]),
    ],
)
def test_v2i_offsets_operate_in_dbm_clip_and_preserve_rb_mode(arm, expected, low_clip, high_clip):
    actions = _actions()
    result = apply_v2i_power_offset(actions, arm, 1.0, 30.0, 3, 2)
    np.testing.assert_array_equal(result.policy_actions, actions)
    np.testing.assert_array_equal(result.environment_actions[:, :2], actions[:, :2])
    np.testing.assert_allclose(result.executed_power_dbm, expected, atol=2e-5)
    assert result.power_clipped_low.tolist() == low_clip
    assert result.power_clipped_high.tolist() == high_clip
    if arm == "baseline":
        assert result.environment_actions is actions
    np.testing.assert_array_equal(actions, _actions())


def test_v2i_minus3_clips_a_low_mode0_power_without_changing_discrete_actions():
    actions = np.asarray([[0.1, -0.5, _normalized_power(2.0)]], dtype=np.float32)
    result = apply_v2i_power_offset(actions, "v2i_minus3db", 1.0, 30.0, 3, 2)
    assert result.executed_power_dbm.item() == pytest.approx(1.0, abs=2e-5)
    assert result.power_clipped_low.tolist() == [True]
    np.testing.assert_array_equal(result.environment_actions[:, :2], actions[:, :2])


def test_required_power_uses_linear_interference_and_environment_gain_units():
    value = required_power_dbm(
        np.asarray([[100.0]]),
        np.asarray([[1e-10]]),
        gamma_required=10.0,
        veh_ant_gain_db=3.0,
        bs_ant_gain_db=8.0,
        bs_noise_figure_db=5.0,
    )
    assert value.item() == pytest.approx(4.0)
    assert required_power_dbm(
        np.asarray([[100.0]]), np.asarray([[1e-9]]), 10.0, 3.0, 8.0, 5.0
    ).item() == pytest.approx(14.0)


class _Vehicle:
    def __init__(self, position):
        self.position = position


class _Config:
    power_min_dbm = 1.0
    power_max_dbm = 30.0
    rsu_position = [0.0, 0.0]


class _SyntheticEnvironment:
    n_platoon = 2
    n_rb = 2
    size_platoon = 1
    n_modes = 2
    sig2 = 1e-12
    veh_ant_gain = 3.0
    bs_ant_gain = 8.0
    bs_noise_figure = 5.0
    v2i_min = 3.0
    time_fast = 1.0
    bandwidth = 1.0
    config = _Config()
    vehicles = [_Vehicle([3.0, 4.0]), _Vehicle([6.0, 8.0])]
    v2i_channels_fast = np.asarray([[100.0, 115.0], [100.0, 115.0]])

    def decode_actions(self, actions):
        raw = np.asarray(actions, dtype=np.float64)
        rb = np.minimum(1, np.floor((np.clip(raw[:, 0], -1, 1) + 1) * 1)).astype(int)
        mode = np.minimum(1, np.floor((np.clip(raw[:, 1], -1, 1) + 1) * 1)).astype(int)
        power = 1.0 + (np.clip(raw[:, 2], -1, 1) + 1.0) * 0.5 * 29.0
        return np.column_stack((rb, mode, power))


def test_three_feasibility_layers_include_hypothetical_mode1_service():
    actions = np.asarray([
        [-0.5, -0.5, _normalized_power(10.0)],
        [-0.5, 0.5, _normalized_power(10.0)],
    ])
    snapshot = compute_feasibility_snapshot(_SyntheticEnvironment(), actions)
    assert snapshot.gamma_required == pytest.approx(7.0)
    assert snapshot.leader_bs_distance_m.tolist() == pytest.approx([5.0, 10.0])
    assert snapshot.reconstructed_v2i_rate[0] > 0
    assert snapshot.reconstructed_v2i_rate[1] == 0
    assert np.all(snapshot.required_power_best_current_interference_dbm <= snapshot.required_power_selected_dbm)
    assert np.all(snapshot.required_power_noise_only_best_dbm <= snapshot.required_power_best_current_interference_dbm)
    assert snapshot.noise_only_best_feasible.shape == (2,)
    assert np.all(
        snapshot.noise_only_best_rate_at_power_max
        >= snapshot.best_current_interference_rate_at_power_max
    )


def _tiny_config(root: Path, seed=71):
    return resolve_config(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=seed,
        episodes=2,
        steps_per_episode=3,
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        mappo_rollout_episodes=1,
        mappo_ppo_epochs=2,
        mappo_variant="tdec",
        device="cpu",
        output_root=str(root),
        run_name="tiny-feasibility-tdec",
        checkpoint_mode="policy_only",
        diagnostics=True,
    )


def test_feasibility_snapshot_matches_rate_and_does_not_change_rng_or_next_trajectory(tmp_path):
    config = _tiny_config(tmp_path / "unused", seed=91)
    first, second = PaperEnviron(copy.deepcopy(config)), PaperEnviron(copy.deepcopy(config))
    first.reset_world(222)
    second.reset_world(222)
    first.start_episode(0)
    second.start_episode(0)
    actions = _environment_actions(config.number_agents)
    rng_before = copy.deepcopy(first.rng.bit_generator.state)
    snapshot = compute_feasibility_snapshot(first, actions)
    assert first.rng.bit_generator.state == rng_before
    next_first, rg1, t11, t21, _, info1 = first.step(actions)
    next_second, rg2, t12, t22, _, info2 = second.step(actions)
    np.testing.assert_allclose(snapshot.reconstructed_v2i_rate, info1["v2i_rate"], rtol=1e-5, atol=1e-3)
    np.testing.assert_array_equal(next_first, next_second)
    assert rg1 == pytest.approx(rg2)
    np.testing.assert_array_equal(t11, t12)
    np.testing.assert_array_equal(t21, t22)
    np.testing.assert_array_equal(info1["aoi_ms"], info2["aoi_ms"])


def test_reward_ledger_reconstructs_components_and_global_is_added_once(tmp_path):
    config = _tiny_config(tmp_path / "unused", seed=92)
    environment = PaperEnviron(config)
    environment.reset_world(223)
    environment.start_episode(0)
    _, global_reward, task1, task2, _, info = environment.step(_environment_actions(config.number_agents))
    ledger = reward_ledger(info, global_reward, task1, task2, config.cam_bits, 0.25)
    np.testing.assert_allclose(ledger["reward_task1"], task1)
    np.testing.assert_allclose(ledger["reward_task2"], task2)
    np.testing.assert_allclose(
        ledger["reward_combined"],
        task1.astype(np.float64) + task2.astype(np.float64) + 0.25 * global_reward,
        atol=1e-6,
    )


def test_saved_quantities_reconstruct_rate_required_power_and_combined_reward(tmp_path):
    config = _tiny_config(tmp_path / "unused", seed=93)
    environment = PaperEnviron(config)
    environment.reset_world(224)
    environment.start_episode(0)
    intervention = apply_v2i_power_offset(
        _environment_actions(config.number_agents),
        "baseline",
        config.power_min_dbm,
        config.power_max_dbm,
        config.n_rb,
        config.n_modes,
    )
    feasibility = compute_feasibility_snapshot(environment, intervention.environment_actions)
    _, global_reward, task1, task2, _, info = environment.step(intervention.environment_actions)
    record = _record_step(
        info,
        intervention,
        feasibility,
        global_reward,
        task1,
        task2,
        config.cam_bits,
        config.global_actor_weight,
    )
    arrays = {key: np.asarray([[[value]]]) for key, value in record.items()}
    complete = {
        "bandwidth_hz": config.bandwidth_hz,
        "slot_ms": config.slot_ms,
        "v2i_min_bits_per_step": config.v2i_min_bits_per_step,
        "gamma_required": feasibility.gamma_required,
        "veh_antenna_gain_db": environment.veh_ant_gain,
        "bs_antenna_gain_db": environment.bs_ant_gain,
        "bs_noise_figure_db": environment.bs_noise_figure,
        "thermal_noise_dbm": environment.sig2_db,
        "global_actor_weight": config.global_actor_weight,
    }
    result = validate_physical_reconstruction(arrays, complete, "baseline", 93)
    assert result["required_power_selected_max_abs_error"] < 2e-4
    assert result["v2i_rate_max_abs_error"] < 1e-3
    assert result["combined_reward_max_abs_error"] < 2e-6


def test_recording_helpers_do_not_consume_process_rng():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    apply_v2i_power_offset(_actions(), "v2i_plus9db", 1.0, 30.0, 3, 2)
    lack_of_opportunity_statistics(np.asarray([[False, True], [False, False]]))
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_opportunity_intervals_connect_episodes_not_worlds_and_mark_censoring():
    stats = lack_of_opportunity_statistics(np.asarray([[False, False, True], [False, False, False]]))
    assert stats["complete_lack_interval_count"] == 0
    assert stats["left_censored_lack_interval_slots"] == 2
    assert stats["right_censored_lack_interval_slots"] == 3
    no_opportunity = lack_of_opportunity_statistics(np.zeros((2, 3), dtype=bool))
    assert no_opportunity["entire_world_lacks_opportunity"] == 1
    assert no_opportunity["left_censored_lack_interval_count"] == 1
    assert no_opportunity["right_censored_lack_interval_count"] == 1
    assert no_opportunity["left_censored_lack_interval_slots"] == 6
    assert no_opportunity["right_censored_lack_interval_slots"] == 6


def test_discounted_returns_reset_at_each_training_episode():
    rewards = np.ones((1, 2, 3, 1), dtype=np.float64)
    returns = _discounted_return(rewards, 0.5)
    np.testing.assert_allclose(returns[:, :, 0], [[1.75, 1.75]])


def test_world_agent_pairing_uses_matching_training_and_eval_seeds():
    rows = []
    metric_names = (
        "mean_aoi_ms", "binary_cam", "payload_completion", "mean_executed_power_mw",
        "v2i_attempt_rate", "v2i_success_given_attempt", "aoi_reset_rate",
        "aoi_gt50_fraction", "aoi_at_cap_fraction", "selected_infeasible_fraction",
        "best_current_interference_infeasible_fraction", "noise_only_best_infeasible_fraction",
        "selected_feasible_but_not_reset_fraction", "mean_reward_global", "mean_reward_task1",
        "mean_reward_task2", "mean_reward_combined", "mean_discounted_combined_return",
        "complete_interval_gt100_count", "complete_interval_gt100_duration_fraction",
    )
    for arm_index, arm in enumerate(ARMS):
        for seed in range(8, 14):
            for eval_seed in range(201, 207):
                for agent in range(5):
                    row = {"arm": arm, "training_seed": seed, "eval_seed": eval_seed, "agent": agent}
                    row.update({key: float(arm_index) for key in metric_names})
                    rows.append(row)
    paired = _paired_stream_rows(rows)
    assert len(paired) == 720
    assert paired[0]["arm"] == "v2i_plus3db"
    assert paired[0]["training_seed"] == 8 and paired[0]["eval_seed"] == 201 and paired[0]["agent"] == 0
    assert paired[0]["delta_mean_aoi_ms"] == pytest.approx(1.0)


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory):
    root = tmp_path_factory.mktemp("feasibility_policy")
    run = Path(train(_tiny_config(root / "training"))["run_dir"])
    return run / "policy_final.pt"


@pytest.mark.parametrize("arm", ["baseline", "v2i_plus3db"])
def test_new_entry_matches_old_common_trajectory_and_freezes_policy(tmp_path, tiny_policy, arm):
    old = evaluate_power_intervention(
        tiny_policy, tmp_path / "old", arm, device="cpu", eval_seeds=[201], eval_episodes=1
    )
    old_metrics = Path(old["eval_dir"]) / "metrics.npz"
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    new = evaluate_feasibility_reward(
        tiny_policy,
        tmp_path / "new",
        arm,
        device="cpu",
        eval_seeds=[201],
        eval_episodes=1,
        reference_metrics=old_metrics,
    )
    assert new["policy_parameters_unchanged"] is True
    assert new["gamma_required"] == pytest.approx(7.0)
    assert new["reference_common_arrays_exact"] is True
    assert (Path(new["eval_dir"]) / "trajectory_equivalence.json").is_file()
    assert torch.equal(torch.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    with np.load(old_metrics, allow_pickle=False) as old_data, np.load(
        Path(new["eval_dir"]) / "metrics.npz", allow_pickle=False
    ) as new_data:
        for key in COMMON_TRAJECTORY_ARRAYS:
            np.testing.assert_array_equal(new_data[key], old_data[key])


def test_hpc_array_maps_30_cells_and_keeps_old_18_cell_launcher():
    root = Path(__file__).resolve().parents[1]
    evaluate = (root / "hpc" / "aoi_mappo_feasibility_reward_eval_array.sbatch").read_text(encoding="utf-8")
    analyze = (root / "hpc" / "aoi_mappo_feasibility_reward_analyze.sbatch").read_text(encoding="utf-8")
    old = (root / "hpc" / "aoi_mappo_mode_power_intervention_eval_array.sbatch").read_text(encoding="utf-8")
    for required in (
        "#SBATCH --array=0-29%6",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --gres=gpu:l20:1",
        "seeds=(8 9 10 11 12 13)",
        "arms=(baseline v2i_plus3db v2i_minus3db v2i_plus6db v2i_plus9db)",
        "arm_index=$((SLURM_ARRAY_TASK_ID / 6))",
        "seed_index=$((SLURM_ARRAY_TASK_ID % 6))",
        "--eval-seeds \"201,202,203,204,205,206\"",
        "--eval-episodes 100",
    ):
        assert required in evaluate
    assert "--gres=gpu" not in analyze
    assert "analysis.summarize_mappo_feasibility_reward" in analyze
    assert "#SBATCH --array=0-17%6" in old
    assert "arms=(baseline v2i_plus3db v2v_minus3db)" in old
