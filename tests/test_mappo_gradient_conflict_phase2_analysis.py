import numpy as np

from analysis.summarize_mappo_gradient_conflict_phase2 import (
    BLOCKS,
    OBJECTIVES,
    PAIRS,
    _format_optional,
    _gradient_rows,
    _gradient_summary,
    _paired,
)


def _geometry(cosine, conflict):
    pair = {"dot": -0.25 if conflict else 0.25, "cosine": cosine, "valid": True, "conflict": conflict}
    return {
        "objective_grad_norm": {"global": 1.0, "task1": 2.0, "task2": 3.0},
        "aggregate_grad_norm": 4.0,
        "pairs": {name: dict(pair) for name in PAIRS},
        "block_pairs": {
            block: {name: dict(pair) for name in PAIRS}
            for block in BLOCKS
        },
        "cancellation_ratio": 0.4,
        "cancellation_valid": True,
    }


def _record(epoch, agent, pcgrad):
    record = {"ppo_epoch": epoch, "agent": agent, **_geometry(-0.5, True)}
    if pcgrad:
        record["pcgrad_projection"] = {
            "schema_version": "mappo_pcgrad_projection_v1",
            "projection_seed": 1,
            "projection_order": {
                "global": ["task1", "task2"],
                "task1": ["task2", "global"],
                "task2": ["global", "task1"],
            },
            "projection_count": {name: 1 for name in OBJECTIVES},
            "before": _geometry(-0.5, True),
            "after": _geometry(0.25, False),
            "projection_magnitude": {name: 0.2 for name in OBJECTIVES},
            "relative_projection_magnitude": {name: 0.1 for name in OBJECTIVES},
            "aggregate_projection_magnitude": 0.3,
        }
    return record


def _updates(arm):
    records = [
        _record(epoch, agent, arm == "pcgrad")
        for epoch in (0, 9)
        for agent in range(5)
    ]
    return [{
        "mappo_actor_update_mode": arm,
        "objective_gradient_diagnostics": {
            "actor_update_mode": arm,
            "records": records,
        },
    } for _ in range(100)]


def test_phase2_gradient_analysis_keeps_pre_and_post_projection_separate():
    pcgrad = _gradient_rows("pcgrad", 8, _updates("pcgrad"))
    assert len(pcgrad) == 2
    assert [row["ppo_epoch"] for row in pcgrad] == [0, 9]
    assert all(row["pre_global_task1_conflict_rate"] == 1.0 for row in pcgrad)
    assert all(row["post_global_task1_conflict_rate"] == 0.0 for row in pcgrad)
    assert all(np.isclose(row["aggregate_projection_magnitude"], 0.3) for row in pcgrad)

    composed = _gradient_rows("composed_clip", 8, _updates("composed_clip"))
    assert all(row["post_cancellation_ratio"] is None for row in composed)
    assert all(row["aggregate_projection_magnitude"] is None for row in composed)

    cohort = _gradient_summary(pcgrad + composed)
    selected = [row for row in cohort if row["arm"] == "pcgrad"]
    assert all(row["pre_global_task1_cosine"] == -0.5 for row in selected)
    assert all(row["post_global_task1_cosine"] == 0.25 for row in selected)


def test_phase2_report_formats_undefined_tiny_gradient_geometry_as_na():
    assert _format_optional(None, 3) == "NA"
    assert _format_optional(0.125, 3) == "0.125"


def test_phase2_paired_rows_isolate_clipping_and_projection_effects():
    training = []
    evaluations = []
    for arm_index, arm in enumerate(("composed_clip", "separate_sum_clip", "pcgrad")):
        for seed in range(8, 14):
            training.append({
                "arm": arm,
                "seed": seed,
                "mean_aoi_ms": 10.0 - arm_index,
                "mean_binary_cam": 0.8 + 0.01 * arm_index,
            })
            for mode in ("deterministic", "stochastic"):
                evaluations.append({
                    "arm": arm,
                    "mode": mode,
                    "training_seed": seed,
                    "mean_aoi_ms": 11.0 - arm_index,
                    "mean_binary_cam": 0.7 + 0.01 * arm_index,
                })

    rows = _paired(training, evaluations)
    assert len(rows) == 9
    assert {
        (row["challenger"], row["reference"])
        for row in rows
    } == {
        ("separate_sum_clip", "composed_clip"),
        ("pcgrad", "composed_clip"),
        ("pcgrad", "separate_sum_clip"),
    }
    assert all(row["aoi_wins"] == 6 and row["binary_cam_wins"] == 6 for row in rows)
