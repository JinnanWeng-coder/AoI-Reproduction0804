import json
from pathlib import Path

import numpy as np
import pytest

from analysis.build_paper_figures import (
    CURRENT_ALGORITHM,
    FIG5_GAP_SCENARIOS,
    REQUIRED_BASELINES,
    plot_fig3,
    plot_fig4,
    plot_fig5,
)


ALL_ALGORITHMS = (CURRENT_ALGORITHM,) + REQUIRED_BASELINES


def _write_training_cell(
    root: Path,
    algorithm: str,
    scenario: str,
    seed: int,
    value: float,
    suffix: str = "",
):
    run = root / f"{algorithm}_{scenario}_{seed}{suffix}"
    run.mkdir(parents=True)
    episodes = np.full(3, value, dtype=np.float64)
    if algorithm == CURRENT_ALGORITHM:
        agent_values = np.column_stack((episodes, episodes + 1.0))
        arrays = {
            "task1_episode_mean": agent_values,
            "task2_episode_mean": agent_values + 2.0,
            "local_total_episode_mean": agent_values + 4.0,
            "global_episode_mean": episodes + 6.0,
            "immediate_reward_proxy": agent_values + 8.0,
        }
    else:
        arrays = {"reward_episode_mean": episodes}
    np.savez_compressed(run / "train_metrics.npz", **arrays)
    return {
        "algorithm": algorithm,
        "scenario": scenario,
        "training_seed": seed,
        "run_path": str(run),
        "status": "complete",
    }


def _write_manifest(path: Path, entries):
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    return path


def _write_eval_cell(
    root: Path,
    algorithm: str,
    scenario_id: str,
    seed: int,
    purpose: str,
    suffix: str = "",
):
    run = root / f"{algorithm}_{scenario_id}_{seed}_{purpose}{suffix}"
    eval_path = run / "eval" / "eval1"
    eval_path.mkdir(parents=True)
    gap = float(scenario_id.split("_g", 1)[1])
    size = int(scenario_id.split("_n", 1)[1].split("_g", 1)[0])
    config = {
        "profile": "paper_faithful",
        "seed": seed,
        "scenario": {
            "id": scenario_id,
            "number_platoons": 5,
            "platoon_size": size,
            "gap_m": gap,
        },
    }
    (run / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    summary = {
        "status": "complete",
        "eval_id": "eval1",
        "eval_purpose": purpose,
        "scope": "validation" if purpose == "validation" else "final_release",
        "is_formal_result": purpose == "final_test",
        "mean_AoI_ms_per_seed": [float(seed), float(seed + 1)],
        "CAM_success_probability_per_seed": [0.8, 0.9],
        "mean_AoI_ms": float(seed) + 0.5,
        "CAM_success_probability": 0.85,
    }
    (eval_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return {
        "algorithm": algorithm,
        "profile": "paper_faithful",
        "scenario": scenario_id,
        "training_seed": seed,
        "run_path": str(run),
        "eval_path": str(eval_path),
        "eval_purpose": purpose,
        "status": "complete",
    }


def _attach_eval_row(training_entry, purpose: str):
    run = Path(training_entry["run_path"])
    scenario_id = training_entry["scenario"]
    seed = int(training_entry["training_seed"])
    eval_id = "validation_eval" if purpose == "validation" else "final_eval"
    eval_path = run / "eval" / eval_id
    eval_path.mkdir(parents=True, exist_ok=True)
    config = {
        "profile": "paper_faithful",
        "seed": seed,
        "scenario": {
            "id": scenario_id,
            "number_platoons": 5,
            "platoon_size": int(scenario_id.split("_n", 1)[1].split("_g", 1)[0]),
            "gap_m": float(scenario_id.split("_g", 1)[1]),
        },
    }
    (run / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    summary = {
        "status": "complete",
        "eval_id": eval_id,
        "eval_purpose": purpose,
        "scope": "validation" if purpose == "validation" else "final_release",
        "is_formal_result": purpose == "final_test",
        "mean_AoI_ms_per_seed": [float(seed), float(seed + 1)],
        "CAM_success_probability_per_seed": [0.8, 0.9],
        "mean_AoI_ms": float(seed) + 0.5,
        "CAM_success_probability": 0.85,
    }
    (eval_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return {
        **training_entry,
        "profile": "paper_faithful",
        "eval_path": str(eval_path),
        "eval_purpose": purpose,
    }


def test_fig4_baseline_seed_holes_fail_or_are_explicit_partial(tmp_path):
    scenario = "p05_n06_g25"
    entries = [
        _write_training_cell(tmp_path / "runs", CURRENT_ALGORITHM, scenario, seed, float(seed))
        for seed in (2, 3)
    ]
    entries.extend(
        _write_training_cell(tmp_path / "runs", algorithm, scenario, 2, 2.0)
        for algorithm in REQUIRED_BASELINES
    )
    manifest = _write_manifest(tmp_path / "manifest.json", entries)

    with pytest.raises(RuntimeError, match="grid is incomplete"):
        plot_fig4(manifest, tmp_path / "fig4.png", scenario, expected_training_seeds=[2, 3])

    output = plot_fig4(
        manifest,
        tmp_path / "fig4.png",
        scenario,
        allow_incomplete=True,
        expected_training_seeds=[2, 3],
    )
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert "PARTIAL" in output.stem
    assert sidecar["status"] == "PARTIAL"
    assert sidecar["partial"] is True
    assert sidecar["aggregation_unit"] == "training_seed"
    assert sidecar["missing_cells"] == [
        {"algorithm": algorithm, "scenario": scenario, "training_seed": 3}
        for algorithm in REQUIRED_BASELINES
    ]

    weakened_manifest = tmp_path / "weakened_manifest.json"
    weakened_manifest.write_text(json.dumps({"required_baselines": [], "entries": entries}), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot omit paper baselines"):
        plot_fig4(weakened_manifest, tmp_path / "weakened.png", scenario, expected_training_seeds=[2, 3])


def test_fig4_complete_grid_aggregates_each_algorithm_across_training_seeds(tmp_path):
    scenario = "p05_n06_g25"
    entries = [
        _write_training_cell(tmp_path / "runs", algorithm, scenario, seed, float(seed))
        for algorithm in ALL_ALGORITHMS
        for seed in (2, 4)
    ]
    manifest = _write_manifest(tmp_path / "manifest.json", entries)
    output = plot_fig4(manifest, tmp_path / "fig4.png", scenario, expected_training_seeds=[2, 4])
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))

    assert output.name == "fig4.png"
    assert sidecar["status"] == "complete"
    assert sidecar["partial"] is False
    assert sidecar["missing_cells"] == []
    assert all(seeds == [2, 4] for seeds in sidecar["per_algorithm_training_seeds"].values())
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as arrays:
        for baseline in REQUIRED_BASELINES:
            curve = arrays[sidecar["algorithm_curve_artifacts"][baseline]]
            assert np.allclose(curve, 3.0)


def test_fig4_rejects_duplicate_algorithm_scenario_seed_cell(tmp_path):
    scenario = "p05_n06_g25"
    entries = [
        _write_training_cell(tmp_path / "runs", algorithm, scenario, seed, float(seed))
        for algorithm in ALL_ALGORITHMS
        for seed in (2, 3)
    ]
    entries.append(_write_training_cell(tmp_path / "runs", CURRENT_ALGORITHM, scenario, 2, 99.0, "_duplicate"))
    manifest = _write_manifest(tmp_path / "manifest.json", entries)

    with pytest.raises(ValueError, match="duplicate algorithm/scenario/training-seed"):
        plot_fig4(manifest, tmp_path / "fig4.png", scenario, expected_training_seeds=[2, 3])


def test_fig5_complete_validation_pilot_is_still_paper_partial(tmp_path):
    entries = [
        _write_eval_cell(tmp_path / "runs", CURRENT_ALGORITHM, scenario, seed, "validation")
        for scenario in FIG5_GAP_SCENARIOS
        for seed in (2, 3)
    ]
    manifest = _write_manifest(tmp_path / "manifest.json", entries)
    output = plot_fig5(
        manifest,
        tmp_path / "fig5.png",
        x_field="gap_m",
        eval_purpose="validation",
        expected_training_seeds=[2, 3],
    )
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))

    assert "PARTIAL" in output.stem
    assert sidecar["status"] == "PARTIAL"
    assert sidecar["partial"] is True
    assert sidecar["validation_current_grid_complete"] is True
    assert sidecar["current_missing_cells"] == []
    assert sidecar["missing_cells"] == []
    assert sidecar["paper_grid_complete"] is False
    assert len(sidecar["paper_missing_cells"]) == len(REQUIRED_BASELINES) * len(FIG5_GAP_SCENARIOS) * 2
    assert {cell["algorithm"] for cell in sidecar["paper_missing_cells"]} == set(REQUIRED_BASELINES)


def test_fig5_final_requires_full_paper_grid_even_when_allow_incomplete(tmp_path):
    entries = [
        _write_eval_cell(tmp_path / "runs", algorithm, scenario, seed, "final_test")
        for algorithm in ALL_ALGORITHMS
        for scenario in FIG5_GAP_SCENARIOS
        for seed in (2, 3)
    ]
    manifest = _write_manifest(tmp_path / "manifest.json", entries)
    output = plot_fig5(
        manifest,
        tmp_path / "fig5.png",
        x_field="gap_m",
        eval_purpose="final_test",
        expected_training_seeds=[2, 3],
    )
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["status"] == "complete"
    assert sidecar["paper_grid_complete"] is True
    assert sidecar["paper_missing_cells"] == []
    assert sidecar["partial"] is False

    with pytest.raises(ValueError, match="cannot omit paper algorithms"):
        plot_fig5(
            manifest,
            tmp_path / "weakened.png",
            x_field="gap_m",
            required_algorithms=(CURRENT_ALGORITHM,),
            eval_purpose="final_test",
            expected_training_seeds=[2, 3],
        )

    incomplete = _write_manifest(tmp_path / "incomplete.json", entries[:-1])
    with pytest.raises(RuntimeError, match="final_test grid is incomplete"):
        plot_fig5(
            incomplete,
            tmp_path / "incomplete.png",
            x_field="gap_m",
            eval_purpose="final_test",
            expected_training_seeds=[2, 3],
            allow_incomplete=True,
        )


def test_complete_lifecycle_manifest_supports_fig4_and_each_fig5_purpose(tmp_path):
    entries = []
    for algorithm in ALL_ALGORITHMS:
        for scenario in FIG5_GAP_SCENARIOS:
            training_entry = _write_training_cell(
                tmp_path / "lifecycle_runs", algorithm, scenario, 2, 2.0
            )
            if algorithm == CURRENT_ALGORITHM:
                entries.append(_attach_eval_row(training_entry, "validation"))
            entries.append(_attach_eval_row(training_entry, "final_test"))
    manifest = _write_manifest(tmp_path / "lifecycle.json", entries)

    fig4 = plot_fig4(
        manifest,
        tmp_path / "lifecycle_fig4.png",
        scenario=FIG5_GAP_SCENARIOS[0],
        expected_training_seeds=[2],
    )
    assert json.loads(fig4.with_suffix(".json").read_text(encoding="utf-8"))["status"] == "complete"

    validation = plot_fig5(
        manifest,
        tmp_path / "lifecycle_validation.png",
        x_field="gap_m",
        eval_purpose="validation",
        expected_training_seeds=[2],
    )
    validation_sidecar = json.loads(validation.with_suffix(".json").read_text(encoding="utf-8"))
    assert validation_sidecar["status"] == "PARTIAL"
    assert validation_sidecar["validation_current_grid_complete"] is True
    assert all("validation_eval" in path for path in validation_sidecar["reused_eval_artifacts"])

    final = plot_fig5(
        manifest,
        tmp_path / "lifecycle_final.png",
        x_field="gap_m",
        eval_purpose="final_test",
        expected_training_seeds=[2],
    )
    final_sidecar = json.loads(final.with_suffix(".json").read_text(encoding="utf-8"))
    assert final_sidecar["status"] == "complete"
    assert all("final_eval" in path for path in final_sidecar["reused_eval_artifacts"])


def test_fig5_rejects_different_run_duplicate_and_entry_summary_purpose_mismatch(tmp_path):
    first = _write_eval_cell(
        tmp_path / "runs",
        CURRENT_ALGORITHM,
        FIG5_GAP_SCENARIOS[0],
        2,
        "validation",
    )
    duplicate = _write_eval_cell(
        tmp_path / "runs",
        CURRENT_ALGORITHM,
        FIG5_GAP_SCENARIOS[0],
        2,
        "validation",
        "_duplicate",
    )
    duplicate_manifest = _write_manifest(tmp_path / "duplicates.json", [first, duplicate])
    with pytest.raises(ValueError, match="duplicate training-seed"):
        plot_fig5(
            duplicate_manifest,
            tmp_path / "duplicate.png",
            x_field="gap_m",
            eval_purpose="validation",
            expected_training_seeds=[2],
        )

    mislabeled_entry = _write_eval_cell(
        tmp_path / "runs",
        CURRENT_ALGORITHM,
        FIG5_GAP_SCENARIOS[1],
        2,
        "final_test",
    )
    mislabeled_entry["eval_purpose"] = "validation"
    mismatch_manifest = _write_manifest(tmp_path / "mismatch.json", [first, mislabeled_entry])
    with pytest.raises(ValueError, match="entry/summary eval purpose mismatch"):
        plot_fig5(
            mismatch_manifest,
            tmp_path / "mismatch.png",
            x_field="gap_m",
            eval_purpose="validation",
            expected_training_seeds=[2],
        )


def test_fig3_agent_axis_and_training_seed_ci_regression(tmp_path):
    scenario = "p05_n06_g25"
    entries = [
        _write_training_cell(tmp_path / "runs", CURRENT_ALGORITHM, scenario, seed, float(seed))
        for seed in (2, 4)
    ]
    manifest = _write_manifest(tmp_path / "manifest.json", entries)
    output = plot_fig3(manifest, tmp_path / "fig3.png", scenario=scenario)
    sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(output.with_suffix(".npz"), allow_pickle=False) as arrays:
        assert arrays["task1_raw"].shape == (2, 3, 2)
        assert arrays["task1_mean"].shape == (3, 2)
        assert arrays["task1_ci95"].shape == (3, 2)
        assert np.all(arrays["task1_ci95"] > 0)
    assert sidecar["agent_count"] == 2
    assert sidecar["aggregation"] == "mean_and_CI_across_training_seeds_preserving_agent_axis"
