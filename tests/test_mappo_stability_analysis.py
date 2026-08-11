import json
from pathlib import Path

import numpy as np
import pytest

from analysis.audit_mappo_default_actions import audit
from analysis.summarize_mappo_stability import ARMS, SEEDS, summarize
from aoi_v2x_reproduction.config import resolve_config


def _run_name(arm: str, seed: int) -> str:
    if arm == "baseline":
        return f"mappo_default_p05_n04_g25_seed{seed:02d}"
    return f"mappo_stability_{arm}_p05_n04_g25_seed{seed:02d}"


def _write_run(root: Path, arm: str, seed: int, include_action_audit: bool) -> None:
    settings = ARMS[arm]
    run_dir = root / "runs" / _run_name(arm, seed)
    run_dir.mkdir(parents=True)
    config = resolve_config(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=seed,
        mappo_actor_lr=settings["actor_lr"],
        mappo_entropy_coef_rb=settings["entropy_rb"],
        mappo_entropy_coef_mode=settings["entropy_mode"],
        mappo_entropy_coef_power=settings["entropy_power"],
    )
    (run_dir / "config.resolved.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (run_dir / "COMPLETE.json").write_text(json.dumps({
        "status": "complete",
        "algorithm": "mappo",
        "update_count": 100,
    }), encoding="utf-8")
    diagnostics = [{
        "approx_kl_per_agent": [0.01] * 5,
        "clip_fraction_per_agent": [0.1] * 5,
        "entropy_rb_per_agent": [0.5] * 5,
        "entropy_mode_per_agent": [0.4] * 5,
        "entropy_power_per_agent": [-0.2] * 5,
        "actor_grad_norm_per_agent": [0.7] * 5,
        "critic_grad_norm": 0.03,
        "explained_variance": 0.6,
    } for _ in range(100)]
    (run_dir / "learning_diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")

    offset = {"baseline": 0.0, "actor_lr1e4": 1.0, "entropy2x": 2.0}[arm]
    metrics = {
        "mean_aoi_ms_episode_agent": np.full((500, 5), seed + offset, dtype=np.float32),
        "endpoint_cam_episode_agent": np.full((500, 5), 0.8, dtype=np.float32),
        "remaining_demand": np.full((500, 100, 5), 320.0, dtype=np.float32),
        "immediate_reward_proxy": np.full((500, 5), -1.25, dtype=np.float32),
    }
    if include_action_audit:
        metrics.update({
            "task1_step": np.full((500, 100, 5), -1.0, dtype=np.float32),
            "task2_step": np.full((500, 100, 5), -0.5, dtype=np.float32),
            "global_step": np.full((500, 100), 0.25, dtype=np.float32),
            "power_dbm": np.full((500, 100, 5), 15.0, dtype=np.float32),
            "mode": np.zeros((500, 100, 5), dtype=np.float32),
            "rb": np.zeros((500, 100, 5), dtype=np.float32),
        })
    np.savez_compressed(run_dir / "train_metrics.npz", **metrics)


def test_action_reward_audit_extracts_blocks_agents_and_reward_components(tmp_path):
    baseline_root = tmp_path / "baseline"
    for seed in SEEDS:
        _write_run(baseline_root, "baseline", seed, include_action_audit=True)

    report = audit(baseline_root)

    assert len(report["rows"]) == 150
    assert len(report["cohort_blocks"]) == 5
    assert len(report["lowest_final_agents"]) == 10
    assert report["cohort_blocks"][0]["mean_payload_completion"] == pytest.approx(0.99)
    assert report["cohort_blocks"][0]["mean_combined_reward"] == pytest.approx(-1.25)
    assert report["cohort_blocks"][0]["mean_power_dbm"] == pytest.approx(15.0)


def test_stability_summary_reuses_baseline_and_compares_twelve_new_cells(tmp_path):
    baseline_root = tmp_path / "baseline"
    stability_root = tmp_path / "stability"
    for seed in SEEDS:
        _write_run(baseline_root, "baseline", seed, include_action_audit=False)
        _write_run(stability_root, "actor_lr1e4", seed, include_action_audit=False)
        _write_run(stability_root, "entropy2x", seed, include_action_audit=False)

    report = summarize(stability_root, baseline_root)

    assert len(report["per_seed"]) == 18
    assert len(report["blocks"]) == 90
    assert len(report["summary"]) == 3
    by_arm = {row["arm"]: row for row in report["summary"]}
    assert by_arm["baseline"]["last100_mean_aoi_ms"] == pytest.approx(10.5)
    assert by_arm["actor_lr1e4"]["last100_mean_aoi_ms"] == pytest.approx(11.5)
    assert by_arm["entropy2x"]["last100_mean_aoi_ms"] == pytest.approx(12.5)
    assert all(row["success_count_last100"] == 6 for row in report["summary"])
