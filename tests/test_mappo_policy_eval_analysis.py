import json

import pytest

from analysis.summarize_mappo_policy_eval import (
    ARM_SPECS,
    EVAL_SEEDS,
    MODES,
    SEEDS,
    _eval_id,
    _run_name,
    summarize,
)


def test_policy_eval_summary_requires_and_compares_all_48_cells(tmp_path):
    arm_offsets = {arm: float(index) for index, arm in enumerate(ARM_SPECS)}
    for arm, spec in ARM_SPECS.items():
        for seed in SEEDS:
            run_name = _run_name(arm, seed)
            for mode in MODES:
                eval_dir = tmp_path / "evaluations" / run_name / _eval_id(mode)
                eval_dir.mkdir(parents=True)
                mode_offset = 0.5 if mode == "stochastic" else 0.0
                value = float(seed) + arm_offsets[arm] + mode_offset
                summary = {
                    "algorithm": "mappo",
                    "status": "complete",
                    "diagnostic_evaluation": True,
                    "scenario": "p05_n04_g25",
                    "training_seed": seed,
                    "training_run_name": run_name,
                    "mappo_eval_mode": mode,
                    "eval_seeds": list(EVAL_SEEDS),
                    "eval_episodes": 100,
                    "eval_warmup_episodes": 5,
                    "policy_name": "policy_final.pt",
                    "mappo_actor_lr": spec["actor_lr"],
                    "mappo_entropy_coef_rb": spec["entropy_rb"],
                    "mappo_entropy_coef_mode": spec["entropy_mode"],
                    "mappo_entropy_coef_power": spec["entropy_power"],
                    "mean_AoI_ms": value,
                    "worst_agent_mean_AoI_ms": value + 1.0,
                    "CAM_success_probability": 0.9,
                    "worst_agent_CAM_success_probability": 0.8,
                    "payload_completion": 0.99,
                    "worst_agent_payload_completion": 0.98,
                }
                (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(summary), encoding="utf-8")

    report = summarize(tmp_path)

    assert len(report["per_training_seed"]) == 48
    assert len(report["summary"]) == 8
    assert len(report["paired"]) == 10
    indexed = {(row["arm"], row["mode"]): row for row in report["summary"]}
    assert indexed[("baseline", "deterministic")]["mean_aoi_ms"] == pytest.approx(10.5)
    assert indexed[("actor_lr1e4_entropy2x", "stochastic")]["mean_aoi_ms"] == pytest.approx(14.0)
    assert all(row["success_count"] == 6 for row in report["summary"])
    pair = next(
        row for row in report["paired"]
        if row["mode"] == "deterministic"
        and row["left_arm"] == "actor_lr1e4_entropy2x"
        and row["right_arm"] == "entropy2x"
    )
    assert pair["mean_delta_aoi_ms_left_minus_right"] == pytest.approx(1.0)
    assert pair["aoi_wins_left"] == 0
