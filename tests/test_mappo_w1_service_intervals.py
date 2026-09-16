"""Small synthetic sequences only; no real W1 data or evaluator invocation."""

import json
from collections import Counter

import numpy as np
import pytest

import analysis.audit_mappo_w1_service_intervals as w1
from analysis.audit_mappo_w1_service_intervals import (
    _ccdf, _load_events, flow_statistics, seed_statistics,
)


def _resets(length, positions):
    values = np.zeros(length, dtype=bool)
    values[positions] = True
    return values


def test_consecutive_and_100_101_intervals():
    row, freq, _ = flow_statistics(_resets(205, [0, 1, 101, 202]))
    assert freq == Counter({1: 1, 100: 1, 101: 1})
    assert row["sum_L"] == 202 and row["sum_L2"] == 20202
    assert row["long_gt100_count"] == 1 and row["long_gt100_sum_L"] == 101
    assert row["left_boundary_visible_wait_slots"] == 0
    assert row["right_boundary_visible_wait_slots"] == 2


def test_episode_continuity_but_not_world_continuity():
    episodes = _resets(8, [3, 4]).reshape(2, 4)
    row, freq, _ = flow_statistics(episodes)
    assert row["complete_interval_count"] == 1 and freq == Counter({1: 1})
    first, _, _ = flow_statistics(_resets(4, [3]))
    second, _, _ = flow_statistics(_resets(4, [0]))
    assert first["complete_interval_count"] + second["complete_interval_count"] == 0


def test_no_reset_and_one_reset_are_not_zero_intervals():
    absent, frequency, _ = flow_statistics(_resets(10, []))
    assert absent["no_reset_flow"] and absent["no_complete_interval_flow"]
    assert absent["no_reset_window_slots"] == 10
    assert absent["left_boundary_visible_wait_slots"] is None
    assert absent["right_boundary_visible_wait_slots"] is None
    assert absent["complete_interval_mean_L"] is None and not frequency
    once, frequency, _ = flow_statistics(_resets(10, [4]))
    assert not once["no_reset_flow"] and once["no_complete_interval_flow"]
    assert once["left_boundary_visible_wait_slots"] == 4
    assert once["right_boundary_visible_wait_slots"] == 5
    assert once["complete_interval_mean_L"] is None and not frequency


def test_capped_aoi_does_not_cap_interval():
    aoi = np.minimum(np.arange(150, dtype=np.float32) + 1, 100)
    aoi[0] = 1
    aoi[120] = 1
    event = np.isclose(aoi, 1, rtol=0, atol=1e-6)
    row, freq, _ = flow_statistics(event)
    assert row["long_gt100_count"] == 1 and freq == Counter({120: 1})


def test_event_and_flow_weighting_differ_and_cover_missing():
    key = ("independent", "tdec", 8)
    rows = []
    freqs = {}
    for agent in range(30):
        reset = (_resets(10, list(range(10))) if agent == 0 else
                 _resets(10, [0, 9]) if agent == 1 else _resets(10, []))
        row, freq, _ = flow_statistics(reset)
        rows.append({"world": 213 + agent // 5, "agent": agent % 5, **row})
        freqs[key + (213 + agent // 5, agent % 5)] = freq
    ccdf = _ccdf(rows, freqs, key, [0, 1, 9])
    assert ccdf[1]["event_ccdf_P_L_gt_threshold"] == pytest.approx(1 / 10)
    assert ccdf[1]["flow_equal_ccdf_P_L_gt_threshold"] == pytest.approx(1 / 2)
    seed, distribution = seed_statistics(rows, freqs, key)
    assert seed["eligible_flow_count"] == 2 and seed["no_complete_interval_flow_count"] == 28
    assert seed["event_mean_L"] != seed["flow_equal_mean_L"]
    assert distribution[0]["flow_equal_denominator_flows"] == 2


def test_paired_ccdf_uses_six_seed_differences():
    rows = []
    for structure in ("independent", "shared"):
        for variant in ("combined", "tdec"):
            for seed in range(8, 14):
                rows.append({"actor_structure": structure, "value_configuration": variant,
                             "training_seed": seed, "threshold_L": 100,
                             "event_ccdf_P_L_gt_threshold": 0.1 if structure == "independent" else 0.2,
                             "flow_equal_ccdf_P_L_gt_threshold": 0.3 if structure == "independent" else 0.1})
    conditions, paired, summaries = w1._ccdf_summaries(rows)
    assert len(conditions) == 4 and len(paired) == 12 and len(summaries) == 2
    assert paired[0]["delta_event_ccdf_P_L_gt_threshold"] == pytest.approx(0.1)
    assert paired[0]["delta_flow_equal_ccdf_P_L_gt_threshold"] == pytest.approx(-0.2)
    assert summaries[0]["delta_event_ccdf_P_L_gt_threshold_defined_seed_count"] == 6


def test_invalid_flow_shape_or_dtype():
    with pytest.raises(ValueError, match="1-D slot or 2-D"):
        flow_statistics(np.zeros((2, 2, 2), dtype=bool))
    with pytest.raises(ValueError, match="boolean"):
        flow_statistics(np.zeros(5, dtype=float))


def test_historical_reconciliation_path_includes_scenario(tmp_path):
    assert w1._historical_dir(tmp_path) == (
        tmp_path / "existing-evidence" / "zero-shot-v1" / "P5_N4_gap25"
        / "analysis" / "service_regularity"
    )


def test_source_metadata_and_shape_rejected(tmp_path):
    root = tmp_path / "study"
    train = root / "train"
    eval_dir = root / "eval"
    train.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    run_name = "run"
    config = {"algorithm": "mappo", "scenario": {"id": "p05_n04_g25"},
              "seed": 8, "mappo_variant": "tdec", "mappo_actor_sharing": False,
              "n_rb": 3, "episodes": 500, "steps_per_episode": 100}
    (train / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    (train / "COMPLETE.json").write_text(json.dumps({"status": "complete", "mappo_variant": "tdec",
                                                      "actor_sharing": False}), encoding="utf-8")
    complete = {"status": "complete", "algorithm": "mappo", "mappo_variant": "tdec",
                "actor_sharing": False, "actor_network_count": 5, "training_seed": 8,
                "training_run_name": run_name, "scenario": "p05_n04_g25",
                "eval_seeds": [213, 214, 215, 216, 217, 218], "eval_episodes": 100,
                "eval_warmup_episodes": 5, "mappo_eval_mode": "stochastic",
                "eval_protocol": "sequential_warm", "intervention_arm": "baseline",
                "policy_name": "policy_final.pt", "policy_episode": 500,
                "policy_parameters_unchanged": True, "is_frozen_eval": True,
                "eval_id": "eval", "policy_path_is_relative_to_eval": True,
                "policy": "../train/policy_final.pt",
                "raw_metric_axes": {"reset_event": ["eval_seed", "scored_episode", "slot", "agent"],
                                    "aoi_ms": ["eval_seed", "scored_episode", "slot", "agent"]}}
    (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    np.savez_compressed(eval_dir / "metrics.npz", reset_event=np.zeros((1, 2, 3, 4), dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        _load_events(eval_dir, train, "independent", "tdec", 8, run_name)
    complete["policy_episode"] = 499
    (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(ValueError, match="policy_episode"):
        _load_events(eval_dir, train, "independent", "tdec", 8, run_name)
    complete["policy_episode"] = 500
    (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    old_shape = w1.SHAPE
    w1.SHAPE = (1, 2, 2, 1)
    try:
        aoi = np.array([1, 2, 1, 2], dtype=np.float32).reshape(w1.SHAPE)
        np.savez_compressed(eval_dir / "metrics.npz", aoi_ms=aoi)
        events, inventory = _load_events(eval_dir, train, "independent", "tdec", 8, run_name)
        assert inventory["event_source"] == "aoi_ms_isclose_1" and int(events.sum()) == 2
        np.savez_compressed(eval_dir / "metrics.npz", aoi_ms=aoi,
                            reset_event=np.zeros(w1.SHAPE, dtype=bool))
        with pytest.raises(ValueError, match="mismatch"):
            _load_events(eval_dir, train, "independent", "tdec", 8, run_name)
    finally:
        w1.SHAPE = old_shape
