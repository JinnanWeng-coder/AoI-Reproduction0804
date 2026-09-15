import json
import os
from pathlib import Path

import numpy as np
import pytest

from analysis.mappo_e5_contract import EVAL_SEEDS, SEEDS
from analysis.mappo_n1_contract import (
    CONDITIONS,
    development_eval_root,
    n1_cell,
    n1_eval_dir,
    source_run_dir,
    validate_n1_cell,
)
from analysis.summarize_mappo_n1_revisit import (
    METRICS,
    _gap_summary,
    _world_agent_rows,
    pair_revisit_and_development,
    sharing_gap_rows,
    summarize,
    write_report,
)
from analysis.evaluate_mappo_feasibility_reward import feasibility_eval_id


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _arrays(world_count: int = 1) -> dict[str, np.ndarray]:
    shape = (world_count, 100, 100, 5)
    return {
        "aoi_ms": np.full(shape, 10.0, dtype=np.float32),
        "success": np.ones(shape, dtype=np.float32),
        "remaining_demand": np.zeros(shape, dtype=np.float32),
        "rb": np.zeros(shape, dtype=np.int64),
        "mode": np.zeros(shape, dtype=np.int64),
        "executed_power_dbm": np.full(shape, 10.0, dtype=np.float32),
        "reward_task1": np.full(shape, 1.0, dtype=np.float32),
        "reward_task2": np.full(shape, 2.0, dtype=np.float32),
        "reward_global": np.full(shape[:-1], 3.0, dtype=np.float32),
    }


def _write_valid_cell(study_root: Path, n1_root: Path, cell_id: int = 6):
    cell = n1_cell(cell_id)
    run_dir = source_run_dir(study_root, cell)
    _write_json(
        run_dir / "config.resolved.json",
        {
            "algorithm": "mappo",
            "scenario": {"id": "p05_n04_g25"},
            "seed": cell.training_seed,
            "episodes": 500,
            "steps_per_episode": 100,
            "mappo_rollout_episodes": 5,
            "mappo_ppo_epochs": 10,
            "mappo_value_clip_mode": "normalized",
            "mappo_variant": cell.variant,
            "mappo_actor_sharing": cell.structure == "shared",
        },
    )
    _write_json(
        run_dir / "COMPLETE.json",
        {
            "status": "complete",
            "mappo_variant": cell.variant,
            "actor_sharing": cell.structure == "shared",
            "reproduction_git_commit": "training-commit",
        },
    )
    (run_dir / "policy_final.pt").write_bytes(b"synthetic policy marker")
    eval_dir = n1_eval_dir(n1_root, cell)
    common = {
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": cell.variant,
        "actor_sharing": cell.structure == "shared",
        "actor_network_count": 1 if cell.structure == "shared" else 5,
        "intervention_arm": "baseline",
        "mappo_eval_mode": "stochastic",
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": 5,
        "eval_seeds": [cell.eval_world],
        "eval_episodes": 100,
        "training_seed": cell.training_seed,
        "training_run_name": cell.run_name,
        "scenario": "p05_n04_g25",
        "policy_name": "policy_final.pt",
        "policy_episode": 500,
        "policy_path_is_relative_to_eval": True,
        "policy_parameters_unchanged": True,
        "is_frozen_eval": True,
        "cam_bits": 32000.0,
        "global_actor_weight": 1.0,
        "aoi_cap_ms": 100.0,
        "reproduction_git_branch": "research/mappo",
        "reproduction_git_commit": "evaluation-commit",
        "reproduction_git_dirty": False,
    }
    policy_reference = os.path.relpath(run_dir / "policy_final.pt", eval_dir).replace(
        os.sep, "/"
    )
    common["policy"] = policy_reference
    _write_json(eval_dir / "EVAL_COMPLETE.json", common)
    _write_json(
        eval_dir / "provenance.json",
        {
            "algorithm": "mappo",
            "mappo_variant": cell.variant,
            "actor_sharing": cell.structure == "shared",
            "training_seed": cell.training_seed,
            "training_source_commit": "training-commit",
            "policy": policy_reference,
            "eval_seeds": [cell.eval_world],
            "eval_episodes": 100,
            "eval_warmup_episodes": 5,
            "mappo_eval_mode": "stochastic",
            "intervention_arm": "baseline",
        },
    )
    np.savez_compressed(eval_dir / "metrics.npz", **_arrays())
    return cell, eval_dir


def _write_development_cell(study_root: Path, cell_id: int) -> None:
    cell = n1_cell(cell_id)
    root = development_eval_root(study_root, cell)
    eval_dir = (
        root
        / "evaluations"
        / cell.run_name
        / feasibility_eval_id("baseline", EVAL_SEEDS, 100, 5)
    )
    complete = {
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": cell.variant,
        "actor_sharing": cell.structure == "shared",
        "actor_network_count": 1 if cell.structure == "shared" else 5,
        "intervention_arm": "baseline",
        "mappo_eval_mode": "stochastic",
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": 5,
        "eval_seeds": list(EVAL_SEEDS),
        "eval_episodes": 100,
        "training_seed": cell.training_seed,
        "training_run_name": cell.run_name,
        "scenario": "p05_n04_g25",
        "policy_name": "policy_final.pt",
        "policy_episode": 500,
        "policy_parameters_unchanged": True,
        "is_frozen_eval": True,
        "cam_bits": 32000.0,
        "global_actor_weight": 1.0,
        "aoi_cap_ms": 100.0,
        "reproduction_git_commit": "development-evaluation-commit",
    }
    _write_json(eval_dir / "EVAL_COMPLETE.json", complete)
    _write_json(
        eval_dir / "provenance.json",
        {
            "algorithm": "mappo",
            "mappo_variant": cell.variant,
            "actor_sharing": cell.structure == "shared",
            "training_seed": cell.training_seed,
            "training_source_commit": "development-training-commit",
            "eval_seeds": list(EVAL_SEEDS),
            "eval_episodes": 100,
            "eval_warmup_episodes": 5,
            "mappo_eval_mode": "stochastic",
            "intervention_arm": "baseline",
        },
    )
    np.savez_compressed(eval_dir / "metrics.npz", **_arrays(world_count=6))


def test_n1_mapping_is_complete_unique_and_each_policy_visits_only_its_own_world():
    cells = [n1_cell(cell_id) for cell_id in range(24)]
    assert len({(cell.structure, cell.variant, cell.training_seed) for cell in cells}) == 24
    assert [(cells[index * 6].structure, cells[index * 6].variant) for index in range(4)] == list(CONDITIONS)
    assert all(cell.eval_world == cell.training_seed for cell in cells)
    assert {cell.training_seed for cell in cells} == set(SEEDS)
    assert n1_cell(0).run_name == "mappo_tdec_ab_combined_p05_n04_g25_seed08"
    assert n1_cell(6).run_name == "mappo_e5_shared_combined_p05_n04_g25_seed08"
    assert n1_cell(12).run_name == "mappo_tdec_ab_tdec_p05_n04_g25_seed08"
    assert n1_cell(18).run_name == "mappo_shared_actor_tdec_p05_n04_g25_seed08"
    with pytest.raises(ValueError):
        n1_cell(-1)
    with pytest.raises(ValueError):
        n1_cell(24)


def test_n1_source_and_development_roots_follow_e5_provenance(tmp_path):
    study = tmp_path / "actor-sharing-study"
    assert "tdec-ab-v1" in str(source_run_dir(study, n1_cell(0)))
    assert "E5-sharing-critic" in str(source_run_dir(study, n1_cell(6)))
    assert "tdec-ab-v1" in str(source_run_dir(study, n1_cell(12)))
    assert "shared-actor-v1" in str(source_run_dir(study, n1_cell(18)))
    assert "E5-sharing-critic" in str(development_eval_root(study, n1_cell(0)))
    assert "E5-sharing-critic" in str(development_eval_root(study, n1_cell(6)))
    assert "shared-actor-v1" in str(development_eval_root(study, n1_cell(12)))
    assert "shared-actor-v1" in str(development_eval_root(study, n1_cell(18)))


def test_single_world_cell_validation_writes_and_revalidates_n1_marker(tmp_path):
    study, n1_root = tmp_path / "study", tmp_path / "n1"
    cell, eval_dir = _write_valid_cell(study, n1_root)
    marker = validate_n1_cell(study, n1_root, cell.cell_id, write_marker=True)
    assert marker["experiment"] == "N1"
    assert marker["evaluation_role"] == "training_world_revisit"
    assert marker["training_seed"] == marker["eval_world"] == 8
    assert "not the training last100" in marker["training_world_revisit_semantics"]
    assert validate_n1_cell(study, n1_root, cell.cell_id) == marker
    assert (eval_dir / "N1_COMPLETE.json").is_file()


def test_n1_validation_rejects_six_world_shape(tmp_path):
    study, n1_root = tmp_path / "study", tmp_path / "n1"
    cell, eval_dir = _write_valid_cell(study, n1_root)
    np.savez_compressed(eval_dir / "metrics.npz", **_arrays(world_count=6))
    with pytest.raises(ValueError, match="shape"):
        validate_n1_cell(study, n1_root, cell.cell_id, write_marker=True)


def test_world_agent_rows_preserve_counts_and_linear_power(tmp_path):
    arrays = _arrays()
    complete = {
        "eval_seeds": [8],
        "cam_bits": 32000.0,
        "global_actor_weight": 1.0,
        "aoi_cap_ms": 100.0,
    }
    rows = _world_agent_rows(
        arrays,
        complete,
        structure="shared",
        variant="combined",
        training_seed=8,
        run_name="synthetic",
        evaluation_role="training_world_revisit",
        data_role="new",
        source_root=tmp_path,
    )
    assert len(rows) == 5
    assert rows[0]["eval_world"] == 8
    assert rows[0]["aoi_sample_count"] == 10000
    assert rows[0]["binary_cam_success_count"] == 100
    assert rows[0]["binary_cam_episode_count"] == 100
    assert rows[0]["mean_power_mw"] == pytest.approx(10.0)
    assert rows[0]["mean_reward_combined"] == pytest.approx(6.0)


def _synthetic_cell_rows(role: str) -> list[dict]:
    rows = []
    for structure, variant in CONDITIONS:
        for seed in SEEDS:
            baseline = 1.0
            if role == "development_worlds":
                baseline += 5.0 if structure == "independent" else 2.0
            row = {
                "evaluation_role": role,
                "structure": structure,
                "variant": variant,
                "training_seed": seed,
                "run_name": n1_cell(CONDITIONS.index((structure, variant)) * 6 + seed - 8).run_name,
            }
            row.update({metric: baseline for metric in METRICS})
            rows.append(row)
    return rows


def test_pairing_and_gap_sign_use_development_minus_revisit_then_shared_minus_independent():
    revisit = _synthetic_cell_rows("training_world_revisit")
    development = _synthetic_cell_rows("development_worlds")
    paired = pair_revisit_and_development(revisit, development)
    assert len(paired) == 24
    assert paired[0]["development_worlds"] == ",".join(str(value) for value in EVAL_SEEDS)
    gaps = sharing_gap_rows(paired)
    assert len(gaps) == 12
    assert gaps[0]["independent_G_mean_aoi_ms"] == pytest.approx(5.0)
    assert gaps[0]["shared_G_mean_aoi_ms"] == pytest.approx(2.0)
    assert gaps[0]["D_mean_aoi_ms"] == pytest.approx(-3.0)
    summary = _gap_summary(gaps)
    assert len(summary) == 2
    assert summary[0]["mean_D_mean_aoi_ms"] == pytest.approx(-3.0)
    assert summary[0]["sd_D_mean_aoi_ms"] == pytest.approx(0.0)


def test_pairing_rejects_missing_and_duplicate_cells():
    revisit = _synthetic_cell_rows("training_world_revisit")
    development = _synthetic_cell_rows("development_worlds")
    with pytest.raises(ValueError, match="24"):
        pair_revisit_and_development(revisit[:-1], development)
    with pytest.raises(ValueError, match="duplicate"):
        pair_revisit_and_development(revisit + [dict(revisit[0])], development)


def test_end_to_end_synthetic_summary_enforces_n1_counts_and_outputs(tmp_path):
    study, n1_root = tmp_path / "study", tmp_path / "n1"
    for cell_id in range(24):
        _write_valid_cell(study, n1_root, cell_id)
        validate_n1_cell(study, n1_root, cell_id, write_marker=True)
        _write_development_cell(study, cell_id)
    report = summarize(study, n1_root)
    assert report["status"] == "PASS"
    assert report["counts"] == {
        "new_training_cells": 0,
        "new_revisit_evaluation_cells": 24,
        "reused_development_evaluation_cells": 24,
        "total_policy_evaluation_cells": 48,
        "condition_count": 4,
        "training_seed_count_per_condition": 6,
        "revisit_world_agent_rows": 120,
        "development_world_agent_rows": 720,
        "combined_world_agent_rows": 840,
        "paired_condition_seed_rows": 24,
    }
    assert len(report["paired_per_seed"]) == 24
    assert len(report["sharing_gap_per_seed"]) == 12
    assert all(row["training_commit"] for row in report["development_cells"])
    output = write_report(n1_root, report)
    assert (output / "n1_summary.json").is_file()
    assert sum(1 for _ in (output / "n1_revisit_world_agent.csv").open()) == 121
    assert sum(1 for _ in (output / "n1_e5_development_world_agent.csv").open()) == 721
    assert sum(1 for _ in (output / "n1_all_world_agent.csv").open()) == 841


def test_n1_sbatch_contracts_are_isolated_and_single_world():
    root = Path(__file__).resolve().parents[1]
    evaluate = (root / "hpc" / "aoi_mappo_n1_revisit_array.sbatch").read_text(encoding="utf-8")
    analyze = (root / "hpc" / "aoi_mappo_n1_analyze.sbatch").read_text(encoding="utf-8")
    n1_root = "actor-sharing-study/N1-training-world-revisit/P5_N4_gap25"
    for script in (evaluate, analyze):
        assert n1_root in script
        assert "MAPPO_results" not in script
    for token in (
        "#SBATCH --array=0-23%6",
        "#SBATCH --gres=gpu:l20:1",
        "structures=(independent shared independent shared)",
        "variants=(combined combined tdec tdec)",
        "--eval-seeds \"$seed\"",
        "analysis.evaluate_mappo_feasibility_reward",
        "analysis.mappo_n1_contract",
        "N1_COMPLETE.json",
    ):
        assert token in evaluate
    assert "213,214,215,216,217,218" not in evaluate
    assert 'eval_world=$seed' in evaluate
    assert '--output-root "$RESULT_ROOT/evaluations"' in evaluate
    assert "--gres=gpu" not in analyze
    assert "analysis.summarize_mappo_n1_revisit" in analyze
    assert '"revisit_world_agent_rows": 120' in analyze
    assert '"development_world_agent_rows": 720' in analyze
    assert '"combined_world_agent_rows": 840' in analyze
