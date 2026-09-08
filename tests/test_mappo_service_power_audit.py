import json

import numpy as np
import pytest

from analysis.audit_mappo_service_power import (
    _aggregate,
    _cohort_rows,
    _consume_evaluation,
    _consume_training,
    _evaluation_paths,
    _training_paths,
    aoi_distribution_rows,
    power_mw,
    update_interval_rows,
)


def _ages(reset_mask):
    age = 100.0
    result = []
    for reset in reset_mask:
        age = 1.0 if reset else min(100.0, age + 1.0)
        result.append(age)
    return np.asarray(result)


def test_power_is_converted_to_linear_mw_before_averaging():
    values = np.asarray([0.0, 10.0])
    assert power_mw(values).mean() == pytest.approx(5.5)
    assert power_mw(np.asarray([values.mean()])).item() != pytest.approx(5.5)


def test_equal_reset_rate_does_not_determine_mean_aoi():
    evenly_spaced = _ages([True, False, False, False, False, True, False, False, False, False])
    clustered = _ages([True, True, False, False, False, False, False, False, False, False])
    assert np.count_nonzero(evenly_spaced == 1.0) == np.count_nonzero(clustered == 1.0) == 2
    assert evenly_spaced.mean() < clustered.mean()


def test_intervals_do_not_hide_boundary_censoring():
    identity = {"phase": "evaluation", "variant": "tdec", "mode": "deterministic", "training_seed": 8, "eval_seed": 201, "agent": 0}
    rows = update_interval_rows(np.asarray([5, 6, 1, 2, 3, 1, 2]), identity)
    observed = {(row["interval_kind"], row["length_slots"]): row["count"] for row in rows}
    assert observed == {("complete", 3): 1, ("left_censored", 2): 1, ("right_censored", 1): 1}


def test_no_reset_is_reported_as_doubly_censored():
    rows = update_interval_rows(np.full(20, 100.0), {"stream": "x"})
    assert rows == [{"stream": "x", "interval_kind": "both_censored", "length_slots": 20, "count": 1}]


def test_zero_v2i_attempt_keeps_conditional_success_missing():
    shape = (1, 4, 1)
    result = _aggregate(
        aoi=np.full(shape, 4.0),
        success=np.zeros(shape),
        remaining=np.ones(shape),
        mode=np.ones(shape, dtype=np.int64),
        power_dbm_values=np.full(shape, 10.0),
        cam_bits=10.0,
    )
    assert result["attempt_count"] == 0
    assert result["v2i_success_given_attempt"] == ""
    assert result["mean_power_mw"] == pytest.approx(10.0)


def test_cohort_q_is_ratio_of_counts_not_mean_of_agent_ratios():
    base = {
        "phase": "training", "variant": "tdec", "mode": "train", "training_seed": 8,
        "eval_seed": "", "period": "block_401_500_last100", "mean_aoi_ms": 3.0,
        "binary_cam": 1.0, "payload_completion": 1.0, "mean_power_dbm": 10.0,
        "mean_power_mw": 10.0, "mean_v2i_rate_on_attempt_bits_per_step": 600.0,
        "mean_selected_interference_on_attempt_db": -90.0,
        "mean_selected_interference_on_attempt_linear": 1e-9,
    }
    rows = [
        {**base, "agent": 0, "slot_count": 100, "attempt_count": 1, "reset_count": 1,
         "v2i_attempt_rate": 0.01, "v2i_success_given_attempt": 1.0, "aoi_reset_rate": 0.01},
        {**base, "agent": 1, "slot_count": 100, "attempt_count": 9, "reset_count": 0,
         "v2i_attempt_rate": 0.09, "v2i_success_given_attempt": 0.0, "aoi_reset_rate": 0.0},
    ]
    cohort = _cohort_rows(rows)[0]
    assert cohort["v2i_success_given_attempt"] == pytest.approx(0.1)
    assert cohort["v2i_attempt_rate"] == pytest.approx(0.05)


def test_aoi_distribution_preserves_cap_mass():
    rows = aoi_distribution_rows(np.asarray([1.0, 2.0, 100.0, 100.0]), {"agent": 0})
    assert rows[-1] == {"agent": 0, "aoi_ms": 100.0, "count": 2}


def test_training_and_evaluation_consumers_respect_expected_axes(tmp_path):
    training = _training_paths(tmp_path, "combined", 8)
    training["directory"].mkdir(parents=True)
    config = {
        "algorithm": "mappo",
        "scenario": {"id": "p05_n04_g25"},
        "seed": 8,
        "episodes": 500,
        "steps_per_episode": 100,
        "mappo_variant": "combined",
        "mappo_value_clip_mode": "normalized",
        "mappo_rollout_episodes": 5,
        "mappo_ppo_epochs": 10,
        "slow_update_every_episodes": 1,
        "tau": 0.005,
        "mappo_actor_lr": 0.0005,
        "mappo_critic_lr": 0.0005,
        "mappo_entropy_coef_rb": 0.02,
        "mappo_entropy_coef_mode": 0.02,
        "mappo_entropy_coef_power": 0.002,
        "v2i_min_bps_per_hz": 3.0,
        "bandwidth_hz": 180000,
        "slot_ms": 1.0,
        "cam_bits": 32000,
    }
    training["config"].write_text(json.dumps(config), encoding="utf-8")
    training["complete"].write_text(json.dumps({
        "status": "complete", "algorithm": "mappo", "mappo_variant": "combined", "seed": 8,
    }), encoding="utf-8")
    train_shape = (500, 100, 5)
    np.savez_compressed(
        training["metrics"],
        aoi_ms=np.ones(train_shape, dtype=np.float32),
        success=np.ones(train_shape, dtype=np.float32),
        remaining_demand=np.zeros(train_shape, dtype=np.float32),
        power_dbm=np.full(train_shape, 10.0, dtype=np.float32),
        mode=np.zeros(train_shape, dtype=np.int64),
        v2i_rate=np.full(train_shape, 600.0, dtype=np.float32),
        selected_interference_db=np.full(train_shape, -90.0, dtype=np.float32),
    )
    evaluation = _evaluation_paths(tmp_path, "combined", 8, "deterministic")
    evaluation["directory"].mkdir(parents=True)
    evaluation["complete"].write_text(json.dumps({
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": "combined",
        "training_seed": 8,
        "mappo_eval_mode": "deterministic",
        "eval_seeds": [201, 202, 203, 204, 205, 206],
        "eval_episodes": 100,
    }), encoding="utf-8")
    eval_shape = (6, 100, 100, 5)
    np.savez_compressed(
        evaluation["metrics"],
        aoi_ms=np.ones(eval_shape, dtype=np.float32),
        success=np.ones(eval_shape, dtype=np.float32),
        remaining_demand=np.zeros(eval_shape, dtype=np.float32),
        power_dbm=np.full(eval_shape, 10.0, dtype=np.float32),
        mode=np.zeros(eval_shape, dtype=np.int64),
    )
    outputs = {name: [] for name in (
        "episode_agent", "seed_agent", "intervals", "aoi_distribution", "power_groups", "integrity",
    )}
    _consume_training(tmp_path, "combined", 8, outputs)
    _consume_evaluation(tmp_path, "combined", 8, "deterministic", outputs)
    assert len(outputs["episode_agent"]) == 2500 + 3000
    assert len(outputs["seed_agent"]) == 25 + 30
    assert len(outputs["integrity"]) == 2
    assert all(row["reset_outside_v2i_attempt"] == 0 for row in outputs["integrity"])


def test_service_power_sbatch_is_cpu_only_and_read_only():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "hpc" / "aoi_mappo_service_power_audit_cpu.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=8G" in text
    assert "--gres=gpu" not in text
    assert "Main.py" not in text
    assert "--eval-only" not in text
    assert "audit_mappo_service_power.py" in text
    assert "MAPPO_results/tdec-ab-v1/P5_N4_gap25" in text
    assert "MAPPO_results/service-power-audit-v1/P5_N4_gap25" in text
