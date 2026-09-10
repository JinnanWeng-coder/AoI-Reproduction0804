import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.evaluate_mappo_rb_intervention import (
    ARMS,
    JOINT_RB_ASSIGNMENTS,
    batched_joint_v2i_rates,
    evaluate_rb_intervention,
    select_joint_rb_strict_improvement,
    task_to_arm_seed,
)
from analysis.evaluate_mappo_feasibility_reward import evaluate_feasibility_reward
from analysis.summarize_mappo_rb_intervention import validate_actions
from analysis.summarize_mappo_rb_intervention import _paired_rows
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.envs.platoon import PaperEnviron
from aoi_v2x_reproduction.runtime.runner import train


def _power_normalized(dbm):
    return 2.0 * (float(dbm) - 1.0) / 29.0 - 1.0


class _Config:
    power_min_dbm = 1.0
    power_max_dbm = 30.0
    n_modes = 2


class _SyntheticEnvironment:
    n_platoon = 5
    n_rb = 3
    size_platoon = 1
    veh_ant_gain = 3.0
    bs_ant_gain = 8.0
    bs_noise_figure = 5.0
    sig2 = 1e-12
    time_fast = 1.0
    bandwidth = 1.0
    v2i_min = 2.0
    config = _Config()
    v2i_channels_fast = np.full((5, 3), 100.0)

    def decode_actions(self, actions):
        raw = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        rb = np.minimum(2, np.floor((raw[:, 0] + 1.0) * 1.5)).astype(np.int64)
        mode = np.minimum(1, np.floor((raw[:, 1] + 1.0))).astype(np.int64)
        power = 1.0 + (raw[:, 2] + 1.0) * 0.5 * 29.0
        return np.column_stack((rb, mode, power))


def _actions(rb=None, mode=None, power=20.0):
    rb = np.asarray([0, 0, 0, 0, 0] if rb is None else rb, dtype=np.int64)
    mode = np.asarray([0, 0, 0, 0, 0] if mode is None else mode, dtype=np.int64)
    return np.column_stack((
        -1.0 + 2.0 * (rb + 0.5) / 3.0,
        -1.0 + 2.0 * (mode + 0.5) / 2.0,
        np.full(5, _power_normalized(power)),
    )).astype(np.float32)


def test_two_arm_mapping_is_fixed():
    assert ARMS == ("baseline", "joint_rb_strict_improve")
    assert task_to_arm_seed(0) == ("baseline", 8)
    assert task_to_arm_seed(5) == ("baseline", 13)
    assert task_to_arm_seed(6) == ("joint_rb_strict_improve", 8)
    assert task_to_arm_seed(11) == ("joint_rb_strict_improve", 13)
    with pytest.raises(ValueError):
        task_to_arm_seed(12)


def test_baseline_is_exact_passthrough_and_does_not_evaluate_oracle():
    environment = _SyntheticEnvironment()
    actions = _actions()
    result = select_joint_rb_strict_improvement(environment, actions, "baseline")
    assert result.intervened.environment_actions is actions
    assert result.oracle_evaluated is False
    assert result.strict_improvement_applied is False
    np.testing.assert_array_equal(result.policy_rb, result.executed_rb)


def test_joint_oracle_strictly_improves_and_changes_only_rb():
    environment = _SyntheticEnvironment()
    actions = _actions()
    original = actions.copy()
    result = select_joint_rb_strict_improvement(
        environment, actions, "joint_rb_strict_improve"
    )
    assert result.oracle_evaluated is True
    assert result.oracle_success_count > result.original_success_count
    assert result.strict_improvement_applied is True
    assert result.rb_changed.any()
    np.testing.assert_array_equal(result.intervened.environment_actions[:, 1:], original[:, 1:])
    np.testing.assert_array_equal(actions, original)


def test_no_strict_gain_keeps_original_rb():
    environment = _SyntheticEnvironment()
    environment.v2i_min = 0.1
    actions = _actions()
    result = select_joint_rb_strict_improvement(
        environment, actions, "joint_rb_strict_improve"
    )
    assert result.oracle_success_count == result.original_success_count == 5
    assert result.strict_improvement_applied is False
    np.testing.assert_array_equal(result.executed_rb, result.policy_rb)
    np.testing.assert_array_equal(result.intervened.environment_actions, actions)


def test_batched_candidates_recompute_joint_interference_and_include_original():
    environment = _SyntheticEnvironment()
    decoded = environment.decode_actions(_actions())
    rates, interference = batched_joint_v2i_rates(environment, decoded)
    assert rates.shape == interference.shape == (243, 5)
    original_index = np.flatnonzero(np.all(JOINT_RB_ASSIGNMENTS == 0, axis=1))[0]
    separated_index = np.flatnonzero(
        np.all(JOINT_RB_ASSIGNMENTS == np.asarray([0, 1, 2, 0, 1]), axis=1)
    )[0]
    assert np.all(interference[original_index] > interference[separated_index])
    assert np.all(rates[original_index] < rates[separated_index])


def test_batched_original_assignment_matches_real_environment_snapshot(tmp_path):
    config = _tiny_config(tmp_path / "unused")
    environment = PaperEnviron(config)
    environment.reset_world(207)
    environment.start_episode(0)
    actions = _actions(rb=[0, 1, 2, 0, 1], mode=[0, 1, 0, 1, 0])
    decoded = environment.decode_actions(actions)
    rates, _ = batched_joint_v2i_rates(environment, decoded)
    original = decoded[:, 0].astype(np.int64)
    original_index = np.flatnonzero(np.all(JOINT_RB_ASSIGNMENTS == original[None, :], axis=1))[0]
    from analysis.evaluate_mappo_feasibility_reward import compute_feasibility_snapshot

    snapshot = compute_feasibility_snapshot(environment, actions)
    np.testing.assert_allclose(rates[original_index], snapshot.reconstructed_v2i_rate, rtol=1e-5, atol=1e-3)


def test_oracle_does_not_consume_process_or_environment_rng():
    environment = _SyntheticEnvironment()
    environment.rng = np.random.default_rng(41)
    env_before = copy.deepcopy(environment.rng.bit_generator.state)
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    select_joint_rb_strict_improvement(environment, _actions(), "joint_rb_strict_improve")
    assert environment.rng.bit_generator.state == env_before
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def _tiny_config(root: Path):
    return resolve_config(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=81,
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
        run_name="tiny-rb-intervention-tdec",
        checkpoint_mode="policy_only",
        diagnostics=True,
    )


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory):
    root = tmp_path_factory.mktemp("rb_intervention_policy")
    run = Path(train(_tiny_config(root / "training"))["run_dir"])
    return run / "policy_final.pt"


@pytest.mark.parametrize("arm", ARMS)
def test_entry_freezes_policy_and_preserves_direct_mode_power(tmp_path, tiny_policy, arm):
    torch_before = torch.get_rng_state().clone()
    result = evaluate_rb_intervention(
        tiny_policy,
        tmp_path / arm,
        arm,
        device="cpu",
        eval_seeds=[207],
        eval_episodes=1,
    )
    assert result["policy_parameters_unchanged"] is True
    assert result["baseline_action_passthrough"] is (True if arm == "baseline" else None)
    assert torch.equal(torch.get_rng_state(), torch_before)
    with np.load(Path(result["eval_dir"]) / "metrics.npz", allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    complete = {
        "training_seed": 81,
        "power_min_dbm": 1.0,
        "power_max_dbm": 30.0,
    }
    validation = validate_actions(arrays, complete, arm)
    assert validation["passed"] is True


def test_baseline_matches_existing_feasibility_evaluator_on_same_new_world(tmp_path, tiny_policy):
    reference = evaluate_feasibility_reward(
        tiny_policy,
        tmp_path / "reference",
        "baseline",
        device="cpu",
        eval_seeds=[207],
        eval_episodes=1,
    )
    candidate = evaluate_rb_intervention(
        tiny_policy,
        tmp_path / "candidate",
        "baseline",
        device="cpu",
        eval_seeds=[207],
        eval_episodes=1,
    )
    common = (
        "aoi_ms", "success", "remaining_demand", "rb", "mode", "power_dbm",
        "action_normalized", "policy_action_normalized", "v2i_rate",
        "selected_interference_db", "reward_global", "reward_task1", "reward_task2",
        "reward_combined",
    )
    with np.load(Path(reference["eval_dir"]) / "metrics.npz", allow_pickle=False) as old, np.load(
        Path(candidate["eval_dir"]) / "metrics.npz", allow_pickle=False
    ) as new:
        for key in common:
            np.testing.assert_array_equal(new[key], old[key])


def test_paired_summary_uses_six_matching_training_seeds():
    metrics = (
        "mean_aoi_ms", "worst_agent_aoi_ms", "p95_aoi_ms", "p99_aoi_ms",
        "aoi_gt50_fraction", "aoi_at_cap_fraction", "mean_binary_cam",
        "worst_agent_binary_cam", "mean_payload_completion",
        "worst_agent_payload_completion", "mean_reward_global", "mean_reward_task1",
        "mean_reward_task2", "mean_reward_combined", "mean_executed_power_mw",
        "complete_interval_p95_slots", "complete_interval_gt100_count",
    )
    rows = []
    for arm_index, arm in enumerate(ARMS):
        for seed in range(8, 14):
            row = {"arm": arm, "training_seed": seed}
            row.update({metric: float(arm_index) for metric in metrics})
            rows.append(row)
    paired, summary = _paired_rows(rows)
    assert len(paired) == summary["paired_seed_count"] == 6
    assert all(row["delta_mean_aoi_ms"] == pytest.approx(1.0) for row in paired)


def test_hpc_scripts_have_requested_resources_mapping_and_no_training():
    root = Path(__file__).resolve().parents[1]
    evaluation = (root / "hpc" / "aoi_mappo_rb_intervention_eval_array.sbatch").read_text(encoding="utf-8")
    analysis = (root / "hpc" / "aoi_mappo_rb_intervention_analyze.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=4" in evaluation
    assert "#SBATCH --gres=gpu:l20:1" in evaluation
    assert "#SBATCH --array=0-11%6" in evaluation
    assert "arms=(baseline joint_rb_strict_improve)" in evaluation
    assert "--eval-seeds 207,208,209,210,211,212" in evaluation
    assert "rb-intervention-v1/P5_N4_gap25" in evaluation
    assert "#SBATCH --cpus-per-task=4" in analysis and "--gres=gpu" not in analysis
    assert "train_mappo" not in evaluation and "Main.py" not in evaluation
