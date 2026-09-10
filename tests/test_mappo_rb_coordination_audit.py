from pathlib import Path

import numpy as np
import pytest

from analysis.audit_mappo_rb_coordination import (
    TRAINING_SEEDS,
    aggregate_streams,
    counterfactual_v2i_quantities,
    dbm_to_mw,
    equal_seed_summary,
    joint_rb_oracle,
    pairwise_reuse_rows,
    recompute_joint_v2i_rates,
    stream_summary,
)
from analysis.evaluate_mappo_feasibility_reward import required_power_dbm


def _complete():
    return {
        "bandwidth_hz": 180000,
        "slot_ms": 1.0,
        "v2i_min_bits_per_step": 540.0,
        "gamma_required": 7.0,
        "veh_antenna_gain_db": 3.0,
        "bs_antenna_gain_db": 8.0,
        "bs_noise_figure_db": 5.0,
        "thermal_noise_dbm": -114.0,
        "power_max_dbm": 30.0,
    }


def _logged_arrays(rb_values=None, mode_values=None, power_values=None):
    complete = _complete()
    rb = np.asarray(rb_values or [0, 0, 1, 1, 2], dtype=np.int64)
    mode = np.asarray(mode_values or [0, 0, 1, 0, 1], dtype=np.int64)
    power = np.asarray(power_values or [10.0, 13.0, 16.0, 19.0, 22.0], dtype=np.float64)
    channel = np.asarray([
        [103.0, 108.0, 112.0],
        [105.0, 101.0, 109.0],
        [100.0, 106.0, 110.0],
        [111.0, 104.0, 102.0],
        [107.0, 109.0, 103.0],
    ])
    noise = 10.0 ** (-114.0 / 10.0)
    gain = 3.0 + 8.0
    noise_figure = 5.0
    interference = np.full((5, 3), noise, dtype=np.float64)
    for receiver in range(5):
        for resource_block in range(3):
            for transmitter in range(5):
                if transmitter != receiver and rb[transmitter] == resource_block:
                    interference[receiver, resource_block] += dbm_to_mw(
                        power[transmitter] - channel[transmitter, resource_block] + gain - noise_figure
                    )
    rates, _ = recompute_joint_v2i_rates(
        rb, mode, power, channel, noise, gain, noise_figure, 0.001, 180000
    )
    required_all = required_power_dbm(channel, interference, 7.0, 3.0, 8.0, 5.0)
    selected_required = required_all[np.arange(5), rb]
    shape = (1, 1, 1)
    arrays = {
        "aoi_ms": np.asarray([[[np.where((mode == 0) & (rates >= 540.0), 1.0, 2.0)]]]),
        "reset_event": np.asarray([[[((mode == 0) & (rates >= 540.0))]]]),
        "rb": rb.reshape(shape + (5,)),
        "mode": mode.reshape(shape + (5,)),
        "executed_power_dbm": power.reshape(shape + (5,)),
        "v2i_rate": rates.reshape(shape + (5,)),
        "v2i_channel_loss_db": channel.reshape(shape + (5, 3)),
        "v2i_interference_plus_noise_linear": interference.reshape(shape + (5, 3)),
        "required_power_selected_dbm": selected_required.reshape(shape + (5,)),
    }
    return arrays, complete


def test_selected_assignment_reconstructs_logged_v2i_rate_and_success():
    arrays, complete = _logged_arrays()
    quantities = counterfactual_v2i_quantities(arrays, complete)
    np.testing.assert_allclose(quantities["selected_rate"], arrays["v2i_rate"], rtol=1e-5, atol=1e-3)
    np.testing.assert_array_equal(quantities["current_success"], arrays["reset_event"])
    assert not quantities["v2i_mask"][0, 0, 0, 2]


def test_self_signal_is_excluded_from_own_interference():
    channel = np.full((2, 3), 100.0)
    mode = np.zeros(2, dtype=np.int64)
    rb = np.asarray([0, 0])
    low_rates, low_interference = recompute_joint_v2i_rates(
        rb, mode, np.asarray([1.0, 10.0]), channel, 1e-12, 11.0, 5.0, 0.001, 180000
    )
    high_rates, high_interference = recompute_joint_v2i_rates(
        rb, mode, np.asarray([30.0, 10.0]), channel, 1e-12, 11.0, 5.0, 0.001, 180000
    )
    assert high_interference[0] == pytest.approx(low_interference[0])
    assert high_interference[1] > low_interference[1]
    assert high_rates[0] > low_rates[0]


def test_joint_rb_change_recomputes_interference():
    channel = np.asarray([[100.0, 100.0, 100.0], [100.0, 110.0, 100.0]])
    mode = np.zeros(2, dtype=np.int64)
    power = np.asarray([20.0, 20.0])
    _, shared = recompute_joint_v2i_rates(
        [0, 0], mode, power, channel, 1e-12, 11.0, 5.0, 0.001, 180000
    )
    _, separated = recompute_joint_v2i_rates(
        [0, 1], mode, power, channel, 1e-12, 11.0, 5.0, 0.001, 180000
    )
    assert np.all(separated < shared)
    np.testing.assert_allclose(separated, 1e-12)


def test_joint_oracle_contains_original_and_cannot_reduce_success_count():
    arrays, complete = _logged_arrays()
    quantities = counterfactual_v2i_quantities(arrays, complete)
    result = joint_rb_oracle(
        arrays["rb"][0, 0, 0],
        arrays["mode"][0, 0, 0],
        arrays["executed_power_dbm"][0, 0, 0],
        arrays["v2i_channel_loss_db"][0, 0, 0],
        quantities,
        quantities["unilateral_rescue"][0, 0, 0],
    )
    assert result["candidate_count"] == 243
    assert result["original_assignment_included"] is True
    assert result["oracle_success_count"] >= result["original_success_count"]


def test_stream_metrics_use_only_v2i_slots_and_keep_dbm_and_mw_means_separate():
    arrays, complete = _logged_arrays()
    quantities = counterfactual_v2i_quantities(arrays, complete)
    row = stream_summary(arrays, quantities, 0, 0, 8)
    assert row["v2i_slots"] == 1
    assert row["mean_actual_power_dbm"] == pytest.approx(10.0)
    assert row["mean_actual_power_mw"] == pytest.approx(10.0)
    mode1 = stream_summary(arrays, quantities, 0, 2, 8)
    assert mode1["v2i_slots"] == 0
    assert mode1["mean_actual_power_dbm"] == ""


def test_training_seed_aggregation_is_equal_weight_not_180_independent_streams():
    rows = []
    for seed in TRAINING_SEEDS:
        for eval_seed in range(201, 207):
            for agent in range(5):
                rows.append({
                    "training_seed": seed,
                    "eval_seed": eval_seed,
                    "agent": agent,
                    "total_slots": 10,
                    "v2i_slots": 5,
                    "v2i_fraction": 0.5,
                    "mean_aoi_ms": float(seed),
                    "aoi_gt50_fraction": 0.0,
                    "reset_count": 3,
                    "v2i_success_given_attempt": 0.6,
                    "aoi_reset_rate": 0.3,
                    "current_v2i_failure_count": 2,
                    "unilateral_rescue_count": 1,
                    "unilateral_rescue_fraction_of_v2i_failures": 0.5,
                    "unilateral_rescue_fraction_of_v2i_slots": 0.2,
                    "mean_actual_power_dbm": float(seed),
                })
    seed_rows = [aggregate_streams(rows, seed) for seed in TRAINING_SEEDS]
    overall = equal_seed_summary(seed_rows)
    assert len(seed_rows) == 6
    assert overall["aggregation"] == "equal weight per training seed"
    assert overall["mean_mean_actual_power_dbm"] == pytest.approx(10.5)


def test_independent_rb_reference_preserves_marginals_inside_each_mode_subset():
    arrays, _ = _logged_arrays()
    arrays["rb"] = np.repeat(arrays["rb"], 6, axis=0)
    arrays["mode"] = np.repeat(arrays["mode"], 6, axis=0)
    rows = pairwise_reuse_rows(arrays, 8)
    pair01_both_v2i = next(
        row for row in rows
        if row["agent_i"] == 0 and row["agent_j"] == 1 and row["mode_scope"] == "both_v2i"
    )
    assert pair01_both_v2i["observed_same_rb_fraction"] == pytest.approx(1.0)
    assert pair01_both_v2i["independent_marginal_reference"] == pytest.approx(1.0)


def test_hpc_entry_is_cpu_only_and_points_to_dedicated_roots():
    root = Path(__file__).resolve().parents[1]
    script = (root / "hpc" / "aoi_mappo_rb_coordination_audit.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=4" in script
    assert "--gres=gpu" not in script
    assert "feasibility-reward-audit-v1/P5_N4_gap25" in script
    assert "rb-coordination-audit-v1/P5_N4_gap25" in script
    assert "analysis.audit_mappo_rb_coordination" in script
    assert "evaluate_mappo" not in script and "Main.py" not in script
