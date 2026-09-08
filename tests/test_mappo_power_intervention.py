import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.evaluate_mappo_power_intervention import (
    _record_step,
    _summary,
    apply_power_intervention,
    evaluate_power_intervention,
)
from analysis.summarize_mappo_power_intervention import (
    COMMON_BASELINE_ARRAYS,
    _paired_rows,
    interval_statistics,
)
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime.runner import evaluate_from_checkpoint, train


def _normalized_power(dbm, low=1.0, high=30.0):
    return 2.0 * (np.asarray(dbm) - low) / (high - low) - 1.0


def _actions():
    return np.asarray([
        [-0.8, -0.5, _normalized_power(28.0)],
        [0.0, 0.5, _normalized_power(2.0)],
        [0.8, -0.5, _normalized_power(10.0)],
        [0.8, 0.5, _normalized_power(20.0)],
    ], dtype=np.float32)


def test_baseline_is_exact_action_passthrough():
    actions = _actions()
    result = apply_power_intervention(actions, "baseline", 1.0, 30.0, 3, 2)
    assert result.environment_actions is actions
    np.testing.assert_array_equal(result.environment_actions, actions)
    np.testing.assert_allclose(result.executed_power_dbm, [28.0, 2.0, 10.0, 20.0], atol=2e-6)
    assert not result.power_clipped_low.any() and not result.power_clipped_high.any()


def test_mode_specific_dbm_interventions_preserve_rb_and_mode_and_clip_bounds():
    actions = _actions()
    plus = apply_power_intervention(actions, "v2i_plus3db", 1.0, 30.0, 3, 2)
    minus = apply_power_intervention(actions, "v2v_minus3db", 1.0, 30.0, 3, 2)
    np.testing.assert_array_equal(plus.environment_actions[:, :2], actions[:, :2])
    np.testing.assert_array_equal(minus.environment_actions[:, :2], actions[:, :2])
    np.testing.assert_array_equal(plus.rb, minus.rb)
    np.testing.assert_array_equal(plus.mode, minus.mode)
    np.testing.assert_allclose(plus.executed_power_dbm, [30.0, 2.0, 13.0, 20.0], atol=2e-5)
    np.testing.assert_allclose(minus.executed_power_dbm, [28.0, 1.0, 10.0, 17.0], atol=2e-5)
    assert plus.power_clipped_high.tolist() == [True, False, False, False]
    assert minus.power_clipped_low.tolist() == [False, True, False, False]
    np.testing.assert_array_equal(actions, _actions())


def test_intervention_and_logging_do_not_consume_rng():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    intervention = apply_power_intervention(_actions(), "v2i_plus3db", 1.0, 30.0, 3, 2)
    info = {
        "aoi_ms": np.asarray([1, 2, 3, 4]),
        "success": np.zeros(4),
        "remaining_demand": np.ones(4),
        "rb": intervention.rb,
        "mode": intervention.mode,
        "power_dbm": intervention.executed_power_dbm,
        "v2i_rate": np.ones(4),
        "selected_interference_db": np.full(4, -90.0),
    }
    _record_step(info, intervention)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_interval_statistics_connect_episode_slots_but_not_separate_seeds():
    episodes = np.asarray([[1, 2, 3], [4, 1, 2]], dtype=np.float32)
    connected = interval_statistics(episodes)
    assert connected["complete_intervals"].tolist() == [4]
    first_seed = interval_statistics(np.asarray([1, 2, 3]))
    second_seed = interval_statistics(np.asarray([4, 1, 2]))
    assert first_seed["complete_interval_count"] == 0
    assert second_seed["complete_interval_count"] == 0


def test_long_complete_intervals_and_censoring_are_separate():
    reset = np.zeros(310, dtype=bool)
    reset[[0, 150, 251]] = True
    aoi = np.empty(310, dtype=np.float32)
    age = 100.0
    for index, is_reset in enumerate(reset):
        age = 1.0 if is_reset else min(100.0, age + 1.0)
        aoi[index] = age
    stats = interval_statistics(aoi)
    assert stats["complete_intervals"].tolist() == [150, 101]
    assert stats["complete_interval_gt100_count"] == 2
    assert stats["complete_interval_gt100_duration_fraction"] == pytest.approx(1.0)
    assert stats["censored_segment_count"] == 1
    assert stats["censored_segment_slots"] == 58


def test_paired_summary_uses_training_seed_and_handles_zero_attempt_q():
    rows = []
    for arm, offset in (("baseline", 0.0), ("v2i_plus3db", -1.0), ("v2v_minus3db", 1.0)):
        for seed in range(8, 14):
            row = {
                "arm": arm,
                "training_seed": seed,
                "mean_aoi_ms": 10.0 + offset,
                "worst_agent_aoi_ms": 20.0 + offset,
                "mean_binary_cam": 0.9,
                "worst_agent_binary_cam": 0.8,
                "mean_payload_completion": 0.99,
                "worst_agent_payload_completion": 0.98,
                "mean_executed_power_mw": 100.0,
                "v2i_attempt_rate": 0.0,
                "v2i_success_given_attempt": "",
                "aoi_reset_rate": 0.0,
                "aoi_gt50_fraction": 0.0,
                "aoi_at_cap_fraction": 0.0,
                "complete_interval_gt100_count": 0,
                "complete_interval_gt100_duration_fraction": "",
                "censored_segment_count": 1,
                "censored_segment_slots": 10,
            }
            rows.append(row)
    paired, summary = _paired_rows(rows)
    assert len(paired) == 12 and len(summary) == 2
    assert summary[0]["paired_seed_count"] == 6
    assert summary[0]["mean_delta_mean_aoi_ms"] == pytest.approx(-1.0)
    assert summary[0]["mean_delta_v2i_success_given_attempt"] == ""


def test_evaluation_summary_handles_zero_attempt_and_averages_linear_power():
    shape = (1, 1, 2, 1)
    arrays = {
        "aoi_ms": np.full(shape, 3.0),
        "success": np.zeros(shape),
        "remaining_demand": np.full(shape, 32000.0),
        "mode": np.ones(shape, dtype=np.int64),
        "reset_event": np.zeros(shape, dtype=bool),
        "executed_power_dbm": np.asarray([[[[0.0], [10.0]]]]),
        "power_at_boundary": np.zeros(shape, dtype=bool),
        "power_clipped_low": np.zeros(shape, dtype=bool),
        "power_clipped_high": np.zeros(shape, dtype=bool),
    }
    summary = _summary(arrays, 32000.0)
    assert summary["v2i_attempt_count"] == 0
    assert summary["v2i_success_given_attempt"] is None
    assert summary["mean_executed_power_mw"] == pytest.approx(5.5)


def _tiny_tdec_config(root: Path):
    return resolve_config(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=71,
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
        run_name="tiny-tdec",
        checkpoint_mode="policy_only",
        diagnostics=True,
    )


def test_new_baseline_matches_original_stochastic_evaluation_and_freezes_policy(tmp_path):
    training = _tiny_tdec_config(tmp_path / "training")
    run_dir = Path(train(training)["run_dir"])
    policy = run_dir / "policy_final.pt"
    original_config = _tiny_tdec_config(tmp_path / "original")
    original = evaluate_from_checkpoint(
        original_config,
        str(policy),
        eval_episodes=1,
        eval_seeds=[201],
        eval_purpose="validation",
        scope="validation",
        diagnostic_eval=True,
        mappo_eval_mode="stochastic",
    )
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    evaluated = evaluate_power_intervention(
        policy,
        tmp_path / "intervention",
        "baseline",
        device="cpu",
        eval_seeds=[201],
        eval_episodes=1,
    )
    assert evaluated["policy_parameters_unchanged"] is True
    assert evaluated["reproduction_git_commit"]
    assert evaluated["policy_path_is_relative_to_eval"] is True
    assert torch.equal(torch.get_rng_state(), torch_before)
    numpy_after = np.random.get_state()
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    with np.load(Path(original["eval_dir"]) / "metrics.npz", allow_pickle=False) as old, np.load(
        Path(evaluated["eval_dir"]) / "metrics.npz", allow_pickle=False
    ) as new:
        for key in COMMON_BASELINE_ARRAYS:
            np.testing.assert_array_equal(new[key], old[key])


def test_hpc_array_maps_18_cells_and_requests_four_cpu_plus_l20():
    root = Path(__file__).resolve().parents[1]
    evaluate = (root / "hpc" / "aoi_mappo_mode_power_intervention_eval_array.sbatch").read_text(encoding="utf-8")
    analyze = (root / "hpc" / "aoi_mappo_mode_power_intervention_analyze.sbatch").read_text(encoding="utf-8")
    for required in (
        "#SBATCH --array=0-17%6",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --gres=gpu:l20:1",
        "seeds=(8 9 10 11 12 13)",
        "arms=(baseline v2i_plus3db v2v_minus3db)",
        "arm_index=$((SLURM_ARRAY_TASK_ID / 6))",
        "seed_index=$((SLURM_ARRAY_TASK_ID % 6))",
        "--eval-seeds \"201,202,203,204,205,206\"",
        "--eval-episodes 100",
    ):
        assert required in evaluate
    assert "Main.py" not in evaluate and "--eval-only" not in evaluate
    assert "--gres=gpu" not in analyze
    assert "analysis.summarize_mappo_power_intervention" in analyze
