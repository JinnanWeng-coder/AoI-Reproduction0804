import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis.build_paper_figures import plot_fig3, plot_fig4, plot_fig5
from analysis.audit_results import audit_eval, audit_study_manifest
from analysis.study_manifest import build_study_manifest
from config import resolve_config


def _write_run(root: Path, name: str, seed: int, with_eval: bool = True):
    run = root / name
    (run / "eval" / "eval1").mkdir(parents=True)
    config = {
        "profile": "paper_faithful",
        "semantic_version": "paper_faithful_v4",
        "mobility_revision": "lane_graph_exit_safe_v1",
        "seed": seed,
        "episodes": 3,
        "steps_per_episode": 2,
        "is_formal_result": False,
        "scenario": {"id": "p05_n06_g25", "number_platoons": 5, "platoon_size": 6, "gap_m": 25.0},
    }
    (run / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    (run / "COMPLETE.json").write_text(json.dumps({"status": "complete", "mobility_revision": "lane_graph_exit_safe_v1"}), encoding="utf-8")
    arrays = {
        "task1_episode_mean": seed + np.arange(15, dtype=np.float32).reshape(3, 5) * 0.1,
        "task2_episode_mean": seed + 1 + np.arange(15, dtype=np.float32).reshape(3, 5)[:, ::-1] * 0.1,
        "local_total_episode_mean": seed + 2 + np.arange(15, dtype=np.float32).reshape(3, 5) * 0.1,
        "global_episode_mean": np.ones(3) * (seed + 3),
        "immediate_reward_proxy": np.ones((3, 5)) * (seed + 4),
    }
    np.savez_compressed(run / "train_metrics.npz", **arrays)
    eval_path = run / "eval" / "eval1"
    if with_eval:
        summary = {
            "status": "complete",
            "eval_id": "eval1",
            "eval_purpose": "validation",
            "scope": "validation",
            "release_status": "validation_ready",
            "is_formal_result": False,
            "eval_protocol": "sequential_warm",
            "semantic_version": "paper_faithful_v4",
            "mobility_revision": "lane_graph_exit_safe_v1",
            "mean_AoI_ms_per_seed": [float(seed), float(seed + 1)],
            "CAM_success_probability_per_seed": [0.1, 0.2],
        }
        (eval_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run, eval_path


def _write_fig5_cell(root: Path, algorithm: str, scenario_id: str, seed: int, purpose: str):
    run = root / f"{algorithm}_{scenario_id}_{purpose}_{seed}"
    eval_path = run / "eval" / "eval1"
    eval_path.mkdir(parents=True)
    is_final = purpose == "final_test"
    config = resolve_config(
        "paper_faithful",
        scenario_id,
        seed=seed,
        episodes=500 if is_final else 3,
        steps_per_episode=100 if is_final else 2,
        run_name=run.name,
        output_root=str(root),
        is_formal_result=is_final,
    )
    (run / "config.resolved.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (run / "COMPLETE.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    summary = {
        "status": "complete",
        "eval_id": "eval1",
        "eval_purpose": purpose,
        "scope": "validation" if purpose == "validation" else "final_release",
        "release_status": "validation_ready" if purpose == "validation" else "final_release",
        "is_formal_result": purpose == "final_test",
        "mean_AoI_ms_per_seed": [float(seed)],
        "CAM_success_probability_per_seed": [0.5],
        "mean_AoI_ms": float(seed),
        "CAM_success_probability": 0.5,
    }
    (eval_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return {
        "algorithm": algorithm,
        "semantic_version": "paper_faithful_v4",
        "mobility_revision": "lane_graph_exit_safe_v1",
        "profile": "paper_faithful",
        "scenario": scenario_id,
        "training_seed": seed,
        "run_path": str(run),
        "eval_path": str(eval_path),
        "status": "complete",
    }


def test_synthetic_fig3_and_fig5_aggregation(tmp_path):
    run1, eval1 = _write_run(tmp_path, "run1", 1)
    run2, eval2 = _write_run(tmp_path, "run2", 2)
    manifest = {
        "entries": [
            {"algorithm": "Modified_MADDPG_with_TDec", "scenario": "p05_n06_g25", "training_seed": 1, "profile": "paper_faithful", "run_path": str(run1), "eval_path": str(eval1), "status": "complete"},
            {"algorithm": "Modified_MADDPG_with_TDec", "scenario": "p05_n06_g25", "training_seed": 2, "profile": "paper_faithful", "run_path": str(run2), "eval_path": str(eval2), "status": "complete"},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    fig3 = plot_fig3(manifest_path, tmp_path / "fig3.png")
    fig5 = plot_fig5(manifest_path, tmp_path / "fig5.png", x_field="platoon_size", eval_purpose="validation", expected_training_seeds=[1, 2])
    assert fig3.is_file() and fig5.is_file()
    rows = json.loads(fig5.with_suffix(".json").read_text(encoding="utf-8"))["rows"]
    assert rows[0]["AoI_ms"]["count"] == 2
    assert json.loads(fig5.with_suffix(".json").read_text(encoding="utf-8"))["partial"] is True
    fig3_meta = json.loads(fig3.with_suffix(".json").read_text(encoding="utf-8"))
    assert fig3_meta["agent_count"] == 5
    with np.load(fig3.with_suffix(".npz"), allow_pickle=False) as fig3_data:
        assert fig3_data["task1_raw"].shape == (2, 3, 5)
        assert not np.allclose(fig3_data["task1_mean"][:, 0], fig3_data["task1_mean"][:, 1])
    assert rows[0]["AoI_ms"]["ci95"] >= 0


def test_fig4_missing_baseline_guard(tmp_path):
    run, _eval = _write_run(tmp_path, "run", 1, with_eval=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": [{"algorithm": "Modified_MADDPG_with_TDec", "scenario": "p05_n06_g25", "run_path": str(run), "status": "complete"}]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="baselines"):
        plot_fig4(manifest_path, tmp_path / "fig4.png")
    marker = json.loads((tmp_path / "INCOMPLETE_BASELINES.json").read_text(encoding="utf-8"))
    assert marker["status"] == "INCOMPLETE_BASELINES"
    assert marker["required"] == ["Modified_MADDPG", "MADDPG_FDec", "DDPG"]


def test_fig4_saves_task_global_and_combined_panels(tmp_path):
    run1, _eval1 = _write_run(tmp_path, "run1", 1, with_eval=False)
    run2, _eval2 = _write_run(tmp_path, "run2", 2, with_eval=False)
    entries = [
        {"algorithm": "Modified_MADDPG_with_TDec", "scenario": "p05_n06_g25", "training_seed": 1, "run_path": str(run1), "status": "complete"},
        {"algorithm": "Modified_MADDPG_with_TDec", "scenario": "p05_n06_g25", "training_seed": 2, "run_path": str(run2), "status": "complete"},
    ]
    for algorithm in ("Modified_MADDPG", "MADDPG_FDec", "DDPG"):
        entries.extend([
            {"algorithm": algorithm, "scenario": "p05_n06_g25", "training_seed": 1, "run_path": str(run1), "status": "complete"},
            {"algorithm": algorithm, "scenario": "p05_n06_g25", "training_seed": 2, "run_path": str(run2), "status": "complete"},
        ])
    manifest = tmp_path / "fig4_manifest.json"
    manifest.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    output = plot_fig4(manifest, tmp_path / "fig4.png", scenario="p05_n06_g25", expected_training_seeds=[1, 2])
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as arrays:
        assert set(("task1", "task2", "global_reward", "combined", "training_objective_proxy")) <= set(arrays.files)
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["partial"] is False
    assert sidecar["drawn_panels"] == ["task1", "task2", "global", "combined"]


def test_fig5_controlled_grid_purpose_partial_and_full_negative_gates(tmp_path):
    gap_scenarios = ["p05_n04_g05", "p05_n04_g15", "p05_n04_g25", "p05_n04_g35"]
    pilot_entries = [
        _write_fig5_cell(tmp_path / "pilot", "Modified_MADDPG_with_TDec", scenario, seed, "validation")
        for scenario in gap_scenarios
        for seed in (2, 3, 4)
    ]
    pilot_manifest = tmp_path / "pilot_manifest.json"
    pilot_manifest.write_text(json.dumps({"entries": pilot_entries}), encoding="utf-8")
    pilot_output = plot_fig5(pilot_manifest, tmp_path / "pilot.png", "gap_m", eval_purpose="validation", expected_training_seeds=[2, 3, 4])
    pilot_sidecar = json.loads(pilot_output.with_suffix(".json").read_text(encoding="utf-8"))
    assert pilot_sidecar["partial"] is True
    assert pilot_sidecar["missing_cells"] == []
    assert "PARTIAL" in pilot_output.stem

    size_scenarios = ["p05_n04_g25", "p05_n06_g25", "p05_n08_g25", "p05_n10_g25"]
    size_entries = [
        _write_fig5_cell(tmp_path / "size", "Modified_MADDPG_with_TDec", scenario, 2, "validation")
        for scenario in size_scenarios
    ]
    size_manifest = tmp_path / "size_manifest.json"
    size_manifest.write_text(json.dumps({"entries": size_entries}), encoding="utf-8")
    size_output = plot_fig5(size_manifest, tmp_path / "size.png", "platoon_size", eval_purpose="validation", expected_training_seeds=[2])
    size_sidecar = json.loads(size_output.with_suffix(".json").read_text(encoding="utf-8"))
    assert size_sidecar["scenario_grid"] == size_scenarios
    assert size_sidecar["missing_cells"] == []

    final_entries = []
    algorithms = ("Modified_MADDPG_with_TDec", "Modified_MADDPG", "MADDPG_FDec", "DDPG")
    for algorithm in algorithms:
        for scenario in gap_scenarios:
            for seed in (2, 3):
                final_entries.append(_write_fig5_cell(tmp_path / "final", algorithm, scenario, seed, "final_test"))
    final_manifest = tmp_path / "final_manifest.json"
    final_manifest.write_text(json.dumps({"entries": final_entries}), encoding="utf-8")
    final_output = plot_fig5(final_manifest, tmp_path / "final.png", "gap_m", eval_purpose="final_test", expected_training_seeds=[2, 3])
    final_sidecar = json.loads(final_output.with_suffix(".json").read_text(encoding="utf-8"))
    assert final_sidecar["partial"] is False
    assert final_sidecar["missing_cells"] == []
    assert all(row["AoI_ms"]["count"] == 2 for row in final_sidecar["rows"])

    missing_manifest = tmp_path / "missing_manifest.json"
    missing_manifest.write_text(json.dumps({"entries": final_entries[:-1]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="grid is incomplete"):
        plot_fig5(missing_manifest, tmp_path / "missing.png", "gap_m", eval_purpose="final_test", expected_training_seeds=[2, 3])

    mixed_entries = [pilot_entries[0], final_entries[0]]
    mixed_manifest = tmp_path / "mixed_manifest.json"
    mixed_manifest.write_text(json.dumps({"entries": mixed_entries}), encoding="utf-8")
    mixed_output = plot_fig5(mixed_manifest, tmp_path / "mixed.png", "gap_m", eval_purpose="validation", expected_training_seeds=[2, 3, 4])
    mixed_sidecar = json.loads(mixed_output.with_suffix(".json").read_text(encoding="utf-8"))
    assert mixed_sidecar["eval_purpose"] == "validation"
    assert mixed_sidecar["reused_eval_artifacts"] == [Path(pilot_entries[0]["eval_path"]).resolve().as_posix()]

    duplicate_manifest = tmp_path / "duplicate_manifest.json"
    duplicate_cell = _write_fig5_cell(tmp_path / "duplicate", "Modified_MADDPG_with_TDec", "p05_n04_g05", 2, "final_test")
    duplicate_manifest.write_text(json.dumps({"entries": final_entries + [duplicate_cell]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate training-seed"):
        plot_fig5(duplicate_manifest, tmp_path / "duplicate.png", "gap_m", eval_purpose="final_test", expected_training_seeds=[2, 3])


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
        "semantic_version": "paper_faithful_v4",
        "mobility_revision": "lane_graph_exit_safe_v1",
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


def test_study_manifest_and_figure_loader_survive_artifact_relocation(tmp_path):
    run = tmp_path / "runs" / "run1"
    run.mkdir(parents=True)
    (run / "config.resolved.json").write_text(json.dumps({
        "profile": "paper_faithful",
        "semantic_version": "paper_faithful_v4",
        "mobility_revision": "lane_graph_exit_safe_v1",
        "seed": 2,
        "scenario": {"id": "p05_n04_g25"},
    }), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({"config_hash": "synthetic"}), encoding="utf-8")
    (run / "COMPLETE.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    manifest_path = tmp_path / "manifests" / "study.json"
    build_study_manifest(run.parent, manifest_path, run_paths=[run])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not Path(manifest["entries"][0]["run_path"]).is_absolute()
    assert audit_study_manifest(manifest_path)["ok"] is True

    relocated = tmp_path / "relocated"
    shutil.copytree(run.parent, relocated / "runs")
    (relocated / "manifests").mkdir(parents=True)
    shutil.copy2(manifest_path, relocated / "manifests" / "study.json")
    relocated_report = audit_study_manifest(relocated / "manifests" / "study.json")
    assert relocated_report["ok"] is True
