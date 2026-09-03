import json

from analysis.summarize_mappo_gradient_conflict import SEEDS, summarize, write_report


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _gradient_record(epoch: int, agent: int):
    geometry = {"dot": -0.25, "cosine": -0.5, "valid": True, "conflict": True}
    pairs = {
        pair: dict(geometry)
        for pair in ("global_task1", "global_task2", "task1_task2")
    }
    return {
        "ppo_epoch": epoch,
        "agent": agent,
        "objective_grad_norm": {"global": 1.0, "task1": 2.0, "task2": 3.0},
        "objective_policy_loss": {"global": 0.1, "task1": 0.2, "task2": 0.3},
        "effective_clip_fraction": {"global": 0.1, "task1": 0.2, "task2": 0.3},
        "pairs": pairs,
        "block_pairs": {
            block: {pair: dict(geometry) for pair in pairs}
            for block in ("trunk", "rb_head", "mode_head", "power_head")
        },
        "cancellation_ratio": 0.4,
        "cancellation_valid": True,
    }


def test_gradient_conflict_analysis_writes_the_four_core_artifacts(tmp_path):
    for seed in SEEDS:
        run_name = f"mappo_gradient_conflict_p05_n04_g25_seed{seed:02d}"
        run_dir = tmp_path / "training" / "runs" / run_name
        run_dir.mkdir(parents=True)
        _write_json(run_dir / "config.resolved.json", {
            "algorithm": "mappo",
            "scenario": {"id": "p05_n04_g25"},
            "seed": seed,
            "episodes": 500,
            "mappo_variant": "tdec",
            "mappo_actor_lr": 0.0005,
            "mappo_entropy_coef_rb": 0.02,
            "mappo_entropy_coef_mode": 0.02,
            "mappo_entropy_coef_power": 0.002,
            "mappo_value_clip_mode": "normalized",
            "mappo_objective_gradient_diagnostics": True,
        })
        _write_json(run_dir / "COMPLETE.json", {"status": "complete", "update_count": 100})
        records = [_gradient_record(epoch, agent) for epoch in (0, 9) for agent in range(5)]
        learning = [{
            "update": update + 1,
            "episode": (update + 1) * 5,
            "global_task1_advantage_correlation_per_agent": [-0.1] * 5,
            "global_task2_advantage_correlation_per_agent": [-0.2] * 5,
            "task1_task2_advantage_correlation_per_agent": [0.05] * 5,
            "objective_gradient_diagnostics": {
                "schema_version": "mappo_objective_gradient_v1",
                "records": records,
            },
        } for update in range(100)]
        _write_json(run_dir / "learning_diagnostics.json", learning)
        _write_json(run_dir / "train_metrics_summary.json", {
            "final_window_mean_AoI_ms": 7.0 + seed / 100.0,
            "final_window_worst_agent_mean_AoI_ms": 12.0,
            "final_window_endpoint_CAM_probability": 0.95,
            "final_window_worst_agent_endpoint_CAM_probability": 0.8,
        })

    report = summarize(tmp_path)
    output = write_report(tmp_path, report)

    assert report["cohort"]["seeds_complete"] == 6
    assert report["cohort"]["diagnostic_rows"] == 6000
    assert report["cohort"]["global_task1_seeds_majority_conflict"] == 6
    for name in (
        "gradient_conflict_per_update.csv",
        "gradient_conflict_per_seed.csv",
        "gradient_conflict_summary.json",
        "gradient_conflict_audit.md",
    ):
        assert (output / name).is_file()
