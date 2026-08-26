import json

import numpy as np
import pytest

from analysis.summarize_mappo_tdec_ab import (
    EVAL_SEEDS,
    MODES,
    SEEDS,
    VARIANTS,
    _eval_id,
    _run_name,
    summarize,
    write_report,
)
from aoi_v2x_reproduction.config import resolve_config


def _write_training(root, variant: str, seed: int) -> None:
    run_dir = root / "training" / "runs" / _run_name(variant, seed)
    run_dir.mkdir(parents=True)
    config = resolve_config(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=seed,
        mappo_variant=variant,
        mappo_actor_lr=0.0005,
        mappo_entropy_coef_rb=0.02,
        mappo_entropy_coef_mode=0.02,
        mappo_entropy_coef_power=0.002,
        mappo_value_clip_mode="normalized",
    )
    (run_dir / "config.resolved.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (run_dir / "COMPLETE.json").write_text(json.dumps({
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": variant,
        "update_count": 100,
    }), encoding="utf-8")
    diagnostics = []
    for _ in range(100):
        row = {
            "mappo_variant": variant,
            "approx_kl_per_agent": [0.01] * 5,
            "clip_fraction_per_agent": [0.1] * 5,
            "entropy_rb_per_agent": [0.5] * 5,
            "entropy_mode_per_agent": [0.4] * 5,
            "critic_loss": 0.2,
            "explained_variance": 0.6,
        }
        if variant == "tdec":
            row.update({
                "global_explained_variance": 0.5,
                "task1_explained_variance": 0.6,
                "task2_explained_variance": 0.7,
            })
        diagnostics.append(row)
    (run_dir / "learning_diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    aoi_value = float(seed) - (1.0 if variant == "tdec" else 0.0)
    cam_value = 0.85 if variant == "tdec" else 0.8
    np.savez_compressed(
        run_dir / "train_metrics.npz",
        mean_aoi_ms_episode_agent=np.full((500, 5), aoi_value, dtype=np.float32),
        endpoint_cam_episode_agent=np.full((500, 5), cam_value, dtype=np.float32),
        remaining_demand=np.full((500, 100, 5), 320.0, dtype=np.float32),
    )


def _write_evaluation(root, variant: str, seed: int, mode: str) -> None:
    run_name = _run_name(variant, seed)
    eval_dir = root / "evaluations" / run_name / _eval_id(mode)
    eval_dir.mkdir(parents=True)
    mode_offset = 0.5 if mode == "stochastic" else 0.0
    value = float(seed) + mode_offset - (1.0 if variant == "tdec" else 0.0)
    (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps({
        "algorithm": "mappo",
        "status": "complete",
        "mappo_variant": variant,
        "diagnostic_evaluation": True,
        "scenario": "p05_n04_g25",
        "training_seed": seed,
        "training_run_name": run_name,
        "mappo_eval_mode": mode,
        "eval_seeds": list(EVAL_SEEDS),
        "eval_episodes": 100,
        "eval_warmup_episodes": 5,
        "policy_name": "policy_final.pt",
        "mean_AoI_ms": value,
        "worst_agent_mean_AoI_ms": value + 1.0,
        "CAM_success_probability": 0.9 if variant == "tdec" else 0.8,
        "worst_agent_CAM_success_probability": 0.8 if variant == "tdec" else 0.7,
        "payload_completion": 0.99,
        "worst_agent_payload_completion": 0.98,
    }), encoding="utf-8")


def test_tdec_ab_summary_requires_and_compares_all_default_cells(tmp_path):
    for variant in VARIANTS:
        for seed in SEEDS:
            _write_training(tmp_path, variant, seed)
            for mode in MODES:
                _write_evaluation(tmp_path, variant, seed, mode)

    report = summarize(tmp_path)

    assert len(report["training_per_seed"]) == 12
    assert len(report["training_summary"]) == 2
    assert len(report["evaluation_per_seed"]) == 24
    assert len(report["evaluation_summary"]) == 4
    assert len(report["paired"]) == 3
    training_pair = report["paired"][0]
    assert training_pair["mean_delta_aoi_ms_tdec_minus_combined"] == pytest.approx(-1.0)
    assert training_pair["aoi_wins_tdec"] == 6
    assert training_pair["mean_delta_binary_cam_tdec_minus_combined"] == pytest.approx(0.05)
    tdec = next(row for row in report["training_per_seed"] if row["variant"] == "tdec")
    assert tdec["last20_global_explained_variance"] == pytest.approx(0.5)

    output = write_report(tmp_path, report)
    assert (output / "mappo_tdec_ab.md").is_file()
    assert (output / "mappo_tdec_ab_summary.json").is_file()
