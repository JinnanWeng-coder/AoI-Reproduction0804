import json
from pathlib import Path

import numpy as np
import pytest

from analysis.build_paper_figures import plot_fig3, plot_fig4, plot_fig5
from analysis.audit_results import audit_eval


def _write_run(root: Path, name: str, seed: int, with_eval: bool = True):
    run = root / name
    (run / "eval" / "eval1").mkdir(parents=True)
    config = {
        "profile": "paper_faithful",
        "semantic_version": "paper_faithful_v2",
        "seed": seed,
        "episodes": 3,
        "steps_per_episode": 2,
        "is_formal_result": False,
        "scenario": {"id": "p05_n06_g25", "number_platoons": 5, "platoon_size": 6, "gap_m": 25.0},
    }
    (run / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "COMPLETE.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    arrays = {
        "task1_episode_mean": np.ones((3, 5)) * seed,
        "task2_episode_mean": np.ones((3, 5)) * (seed + 1),
        "local_total_episode_mean": np.ones((3, 5)) * (seed + 2),
        "global_episode_mean": np.ones(3) * (seed + 3),
        "immediate_reward_proxy": np.ones((3, 5)) * (seed + 4),
    }
    np.savez_compressed(run / "train_metrics.npz", **arrays)
    eval_path = run / "eval" / "eval1"
    if with_eval:
        summary = {
            "status": "complete",
            "eval_id": "eval1",
            "mean_AoI_ms_per_seed": [float(seed), float(seed + 1)],
            "CAM_success_probability_per_seed": [0.1, 0.2],
        }
        (eval_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run, eval_path


def test_synthetic_fig3_and_fig5_aggregation(tmp_path):
    run1, eval1 = _write_run(tmp_path, "run1", 1)
    run2, eval2 = _write_run(tmp_path, "run2", 2)
    manifest = {
        "entries": [
            {"algorithm": "Modified_MADDPG_with_TDec", "scenario": "p05_n06_g25", "run_path": str(run1), "eval_path": str(eval1), "status": "complete"},
            {"algorithm": "Modified_MADDPG_with_TDec", "scenario": "p05_n06_g25", "run_path": str(run2), "eval_path": str(eval2), "status": "complete"},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fig3 = plot_fig3(manifest_path, tmp_path / "fig3.png")
    fig5 = plot_fig5(manifest_path, tmp_path / "fig5.png")
    assert fig3.is_file() and fig5.is_file()
    rows = json.loads((tmp_path / "fig5.json").read_text(encoding="utf-8"))["rows"]
    assert rows[0]["AoI_ms"]["count"] == 4
    assert rows[0]["AoI_ms"]["ci95"] >= 0


def test_fig4_missing_baseline_guard(tmp_path):
    run, _eval = _write_run(tmp_path, "run", 1, with_eval=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": [{"algorithm": "Modified_MADDPG_with_TDec", "scenario": "p05_n06_g25", "run_path": str(run), "status": "complete"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="baselines"):
        plot_fig4(manifest_path, tmp_path / "fig4.png")
    marker = json.loads((tmp_path / "INCOMPLETE_BASELINES.json").read_text(encoding="utf-8"))
    assert marker["status"] == "INCOMPLETE_BASELINES"


def test_negative_eval_audit_rejects_duplicate_seeds_and_wrong_protocol(tmp_path):
    eval_dir = tmp_path / "bad_eval"
    eval_dir.mkdir()
    np.savez_compressed(eval_dir / "metrics.npz", aoi_ms=np.zeros((2, 1, 1, 1)), success=np.zeros((2, 1, 1, 1)))
    summary = {
        "status": "complete",
        "eval_seeds": [101, 101],
        "eval_episodes": 1,
        "eval_protocol": "independent_reset",
        "eval_warmup_episodes": 0,
        "semantic_version": "paper_faithful_v2",
        "checkpoint": str(tmp_path / "missing.pt"),
        "checkpoint_sha256": "bad",
        "mean_AoI_ms_per_seed": [0, 0],
        "sd_AoI_ms_per_seed": [0, 0],
        "ci95_AoI_ms_per_seed": [0, 0],
        "endpoint_success_probability_per_seed": [0, 0],
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(summary), encoding="utf-8")
    report = audit_eval(eval_dir)
    assert report["ok"] is False
    assert "heldout_seed_uniqueness" in report["errors"]
    assert "eval_protocol" in report["errors"]
