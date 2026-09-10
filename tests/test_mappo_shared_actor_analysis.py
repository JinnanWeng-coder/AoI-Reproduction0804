import json
from pathlib import Path

import numpy as np
import pytest

from analysis.summarize_mappo_shared_actor import (
    EVAL_SEEDS,
    SEEDS,
    STRUCTURES,
    eval_task_to_structure_seed,
    shared_eval_id,
    summarize,
    training_run_name,
    write_report,
)
from analysis.evaluate_mappo_feasibility_reward import evaluate_feasibility_reward
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime.runner import train


def _trajectory(aoi_value: float, cam_value: float, remaining_value: float, power_dbm: float, leading):
    shape = tuple(leading) + (100, 5)
    rb = np.indices(shape[:-1], sparse=False)[-1][..., None]
    rb = np.broadcast_to(rb % 3, shape).astype(np.int16)
    mode = np.zeros(shape, dtype=np.int8)
    return {
        "aoi_ms": np.full(shape, aoi_value, dtype=np.float32),
        "success": np.full(shape, cam_value, dtype=np.float32),
        "remaining_demand": np.full(shape, remaining_value, dtype=np.float32),
        "rb": rb,
        "mode": mode,
        "power_dbm": np.full(shape, power_dbm, dtype=np.float32),
    }


def _write_training(independent_root: Path, result_root: Path, structure: str, seed: int) -> None:
    base = independent_root if structure == "independent" else result_root
    run_dir = base / "training" / "runs" / training_run_name(structure, seed)
    run_dir.mkdir(parents=True)
    sharing = structure == "shared"
    config = resolve_config(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=seed,
        mappo_variant="tdec",
        mappo_actor_sharing=sharing,
        mappo_actor_lr=0.0005,
        mappo_entropy_coef_rb=0.02,
        mappo_entropy_coef_mode=0.02,
        mappo_entropy_coef_power=0.002,
        mappo_value_clip_mode="normalized",
    ).to_dict()
    if not sharing:
        config.pop("mappo_actor_sharing")
    (run_dir / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    complete = {
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": "tdec",
        "update_count": 100,
        "parameter_counts": {
            "actors": 100 if sharing else 500,
            "critic": 200,
            "total": 300 if sharing else 700,
        },
    }
    if sharing:
        complete.update({
            "actor_sharing": True,
            "actor_network_count": 1,
            "actor_optimizer_step_count": 1000,
        })
    (run_dir / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    diagnostics = [{"mappo_variant": "tdec", **({"actor_sharing": True} if sharing else {})} for _ in range(100)]
    (run_dir / "learning_diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    (run_dir / "policy_final.pt").write_bytes(b"fixture")
    aoi = 8.0 if sharing else 10.0
    cam = 0.9 if sharing else 0.8
    remaining = 50.0 if sharing else 100.0
    power = 18.0 if sharing else 20.0
    arrays = _trajectory(aoi, cam, remaining, power, (500,))
    arrays.update({
        "global_step": np.full((500, 100), -0.8 if sharing else -1.0, dtype=np.float32),
        "task1_step": np.full((500, 100, 5), -0.4 if sharing else -0.5, dtype=np.float32),
        "task2_step": np.full((500, 100, 5), -0.3 if sharing else -0.5, dtype=np.float32),
    })
    np.savez_compressed(run_dir / "train_metrics.npz", **arrays)


def _write_evaluation(result_root: Path, structure: str, seed: int) -> None:
    run_name = training_run_name(structure, seed)
    eval_dir = result_root / "evaluations" / run_name / shared_eval_id()
    eval_dir.mkdir(parents=True)
    sharing = structure == "shared"
    complete = {
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": "tdec",
        "actor_sharing": sharing,
        "actor_network_count": 1 if sharing else 5,
        "intervention_arm": "baseline",
        "intervention_db": 0.0,
        "mappo_eval_mode": "stochastic",
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": 5,
        "eval_seeds": list(EVAL_SEEDS),
        "eval_episodes": 100,
        "training_seed": seed,
        "training_run_name": run_name,
        "policy_parameters_unchanged": True,
        "global_actor_weight": 1.0,
        "cam_bits": 4000.0,
    }
    (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    aoi = 12.0 if sharing else 15.0
    cam = 0.85 if sharing else 0.75
    remaining = 80.0 if sharing else 160.0
    power = 19.0 if sharing else 21.0
    arrays = _trajectory(aoi, cam, remaining, power, (6, 100))
    arrays["executed_power_dbm"] = arrays["power_dbm"]
    arrays.update({
        "reward_global": np.full((6, 100, 100), -0.7 if sharing else -1.0, dtype=np.float32),
        "reward_task1": np.full((6, 100, 100, 5), -0.4 if sharing else -0.5, dtype=np.float32),
        "reward_task2": np.full((6, 100, 100, 5), -0.3 if sharing else -0.5, dtype=np.float32),
    })
    np.savez_compressed(eval_dir / "metrics.npz", **arrays)


def test_shared_actor_mapping_and_hpc_contracts_are_fixed():
    assert eval_task_to_structure_seed(0) == ("independent", 8)
    assert eval_task_to_structure_seed(5) == ("independent", 13)
    assert eval_task_to_structure_seed(6) == ("shared", 8)
    assert eval_task_to_structure_seed(11) == ("shared", 13)
    with pytest.raises(ValueError):
        eval_task_to_structure_seed(12)

    root = Path(__file__).resolve().parents[1]
    train = (root / "hpc" / "aoi_mappo_shared_actor_train_array.sbatch").read_text(encoding="utf-8")
    evaluate = (root / "hpc" / "aoi_mappo_shared_actor_eval_array.sbatch").read_text(encoding="utf-8")
    analyze = (root / "hpc" / "aoi_mappo_shared_actor_analyze.sbatch").read_text(encoding="utf-8")
    for required in (
        "#SBATCH --cpus-per-task=4", "#SBATCH --gres=gpu:l20:1", "#SBATCH --array=0-5%6",
        "--mappo-actor-sharing shared", "--mappo-variant tdec", "--episodes 500",
    ):
        assert required in train
    for required in (
        "#SBATCH --cpus-per-task=4", "#SBATCH --gres=gpu:l20:1", "#SBATCH --array=0-11%6",
        "structures=(independent shared)", "--eval-seeds \"$EVAL_SEEDS\"", "--arm baseline",
    ):
        assert required in evaluate
    assert "--gres=gpu" not in analyze
    assert "analysis.summarize_mappo_shared_actor" in analyze


def test_no_intervention_frozen_entry_loads_one_actor_policy(tmp_path):
    config = resolve_config(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=8,
        episodes=2,
        steps_per_episode=3,
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        mappo_rollout_episodes=1,
        mappo_ppo_epochs=2,
        mappo_variant="tdec",
        mappo_actor_sharing=True,
        device="cpu",
        output_root=str(tmp_path / "training"),
        run_name="shared-policy-fixture",
        checkpoint_mode="policy_only",
    )
    run_dir = Path(train(config)["run_dir"])
    result = evaluate_feasibility_reward(
        run_dir / "policy_final.pt",
        tmp_path / "evaluations",
        "baseline",
        device="cpu",
        eval_seeds=[213],
        eval_episodes=1,
    )
    assert result["actor_sharing"] is True
    assert result["actor_network_count"] == 1
    assert result["intervention_db"] == 0.0
    assert result["policy_parameters_unchanged"] is True
    assert Path(result["eval_dir"]).joinpath("EVAL_COMPLETE.json").is_file()


def test_shared_actor_summary_pairs_six_training_seeds_and_world_agent_rows(tmp_path):
    independent_root = tmp_path / "tdec-ab"
    result_root = tmp_path / "shared-actor"
    for structure in STRUCTURES:
        for seed in SEEDS:
            _write_training(independent_root, result_root, structure, seed)
            _write_evaluation(result_root, structure, seed)

    report = summarize(independent_root, result_root)
    assert report["status"] == "PASS"
    assert report["shared_training_cells"] == 6
    assert report["evaluation_cells"] == 12
    assert report["training_per_seed_rows"] == 12
    assert report["training_per_episode_rows"] == 6000
    assert report["eval_world_agent_rows"] == 360
    assert report["paired_training_seed_rows"] == 12
    assert report["paired_eval_world_agent_rows"] == 180

    training = {row["structure"]: row for row in report["training_summary"]}
    assert training["independent"]["actor_network_count"] == 5
    assert training["shared"]["actor_network_count"] == 1
    shared_seed = next(row for row in report["training_per_seed"] if row["structure"] == "shared")
    independent_seed = next(row for row in report["training_per_seed"] if row["structure"] == "independent")
    assert shared_seed["actor_optimizer_step_count"] == 1000
    assert independent_seed["actor_optimizer_step_count"] == 5000
    assert independent_seed["actor_optimizer_step_count_source"] == "derived_from_legacy_metadata"

    heldout_pair = next(row for row in report["paired_per_seed"] if row["phase"] == "heldout_stochastic")
    assert heldout_pair["delta_mean_aoi_ms"] == pytest.approx(-3.0)
    assert heldout_pair["delta_mean_binary_cam"] == pytest.approx(0.1)
    assert heldout_pair["delta_mean_power_mw"] < 0.0
    assert len(report["paired_eval_world_agent"]) == 6 * 6 * 5

    output = write_report(result_root, report)
    assert (output / "shared_actor_comparison.md").is_file()
    assert (output / "shared_actor_summary.json").is_file()
    assert (output / "shared_actor_training_per_episode.csv").is_file()
    compact = json.loads((output / "shared_actor_summary.json").read_text(encoding="utf-8"))
    assert "training_per_episode" not in compact
    assert compact["evaluation_cells"] == 12
