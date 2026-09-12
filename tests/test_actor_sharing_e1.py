import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.e1_contract import (
    CHECKPOINT_EPISODE,
    cell_to_algorithm_noise_seed,
    canonical_run_name,
    historical_run_name,
    maddpg_eval_id,
    resolve_maddpg_source_run,
    validate_latest500_payload,
)
from analysis.evaluate_maddpg_e1 import evaluate_maddpg_e1, reduce_heldout_metrics
from analysis.summarize_actor_sharing_e1 import summarize_actor_sharing_e1
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state
from aoi_v2x_reproduction.runtime.runner import train


def test_e1_cell_mapping_and_names():
    assert cell_to_algorithm_noise_seed(0) == ("modified_maddpg", 0.0, 8)
    assert cell_to_algorithm_noise_seed(5) == ("modified_maddpg", 0.0, 13)
    assert cell_to_algorithm_noise_seed(6) == ("modified_maddpg", 0.3, 8)
    assert cell_to_algorithm_noise_seed(11) == ("modified_maddpg", 0.3, 13)
    assert cell_to_algorithm_noise_seed(12) == ("modified_maddpg_tdec", 0.0, 8)
    assert cell_to_algorithm_noise_seed(18) == ("modified_maddpg_tdec", 0.3, 8)
    assert cell_to_algorithm_noise_seed(23) == ("modified_maddpg_tdec", 0.3, 13)
    with pytest.raises(ValueError):
        cell_to_algorithm_noise_seed(24)
    assert canonical_run_name("modified_maddpg", 8) == "modified_maddpg_default_p05_n04_g25_seed08"
    assert canonical_run_name("modified_maddpg_tdec", 13) == "tdec_p05_n04_g25_seed13"
    assert historical_run_name("modified_maddpg_tdec", 8) == "gap_global_slow_sync_slow01_p05_n04_g25_seed08"
    assert "noise0p0" in maddpg_eval_id(0.0) and "213-214-215-216-217-218" in maddpg_eval_id(0.0)


def test_latest500_validation_rejects_best_and_wrong_episode():
    payload = {
        "algorithm": "modified_maddpg",
        "episode": 500,
        "completed": True,
        "config": {"algorithm": "modified_maddpg", "seed": 8, "n_rb": 3, "scenario": {"id": "p05_n04_g25"}},
    }
    validate_latest500_payload(payload, "modified_maddpg", 8)
    with pytest.raises(ValueError, match="episode"):
        validate_latest500_payload({**payload, "episode": 350}, "modified_maddpg", 8)
    with pytest.raises(ValueError, match="MAPPO"):
        validate_latest500_payload({**payload, "algorithm": "mappo"}, "mappo", 8)
    missing = {"episode": 500, "completed": True, "training_completed": True, "config": {"seed": 8, "n_rb": 3, "scenario": {"id": "p05_n04_g25"}}}
    validate_latest500_payload(missing, "modified_maddpg_tdec", 8)


def test_metric_reduction_separates_cam_payload_power_and_reward():
    worlds, episodes, steps, agents = 2, 3, 100, 5
    aoi = np.full((worlds, episodes, steps, agents), 10.0)
    aoi[0, 0, 0, 0] = 80.0
    aoi[1, 0, 1, 1] = 100.0
    success = np.zeros((worlds, episodes, steps, agents))
    success[..., -1, :] = 1.0
    success[..., -1, 0] = 0.0
    remaining = np.zeros((worlds, episodes, steps, agents))
    remaining[..., -1, 1] = 16000.0
    power_dbm = np.full((worlds, episodes, steps, agents), 10.0)
    reward_global = np.full((worlds, episodes, steps), 1.0)
    reward_task1 = np.full((worlds, episodes, steps, agents), 2.0)
    reward_task2 = np.full((worlds, episodes, steps, agents), 3.0)
    metrics = reduce_heldout_metrics(
        aoi=aoi, success=success, remaining=remaining, power_dbm=power_dbm,
        reward_global=reward_global, reward_task1=reward_task1, reward_task2=reward_task2,
        cam_bits=32000.0, global_actor_weight=1.0,
    )
    assert metrics["mean_binary_cam"] == pytest.approx(0.8)
    assert metrics["worst_agent_binary_cam"] == pytest.approx(0.0)
    assert metrics["mean_payload_completion"] != metrics["mean_binary_cam"]
    assert metrics["worst_agent_payload_completion"] == pytest.approx(0.5)
    assert metrics["aoi_gt50_count"] == 2
    assert metrics["aoi_gt50_denominator"] == aoi.size
    assert metrics["aoi_at_cap_count"] == 1
    assert metrics["mean_power_mw"] == pytest.approx(10.0)
    assert metrics["power_mw_sum"] == pytest.approx(10.0 * aoi.size)
    assert metrics["mean_reward_combined"] == pytest.approx(6.0)
    assert metrics["mean_reward_combined"] != pytest.approx(
        metrics["mean_reward_task1"] + metrics["mean_reward_task2"] + 2.0 * metrics["mean_reward_global"]
    )


def _tiny_config(tmp_path: Path, algorithm: str, seed: int = 8, run_name: str = "tiny"):
    return resolve_config(
        scenario="p05_n04_g25",
        algorithm=algorithm if algorithm != "modified_maddpg_tdec" else None,
        seed=seed,
        episodes=2,
        steps_per_episode=100,
        device="cpu",
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        batch_size=4,
        replay_capacity=32,
        checkpoint_every=1,
        checkpoint_mode="resumable",
        target_noise_sigma=0.0,
        diagnostics=False,
        selection_validation_seeds=[301],
        selection_validation_episodes=1,
        selection_validation_warmup_episodes=0,
        output_root=str(tmp_path),
        run_name=run_name,
        is_formal_result=False,
    )


def _write_complete(run_dir: Path, algorithm: str, seed: int) -> None:
    config = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    complete = {
        "status": "complete",
        "seed": seed,
        "scenario": "p05_n04_g25",
        "episodes": 500,
        "final_episode": 500,
    }
    if algorithm == "modified_maddpg":
        complete["algorithm"] = "modified_maddpg"
        assert config.get("algorithm") == "modified_maddpg"
    (run_dir / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    if not (run_dir / "provenance.json").is_file():
        (run_dir / "provenance.json").write_text(json.dumps({"status": "fixture"}), encoding="utf-8")


def _install_source(diagnostics: Path, algorithm: str, seed: int, trained: Path) -> Path:
    if algorithm == "modified_maddpg":
        root = diagnostics / "Modified_MADDPG_results" / "default" / "P5_N4_gap25" / "runs"
        dest = root / canonical_run_name(algorithm, seed)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("COMPLETE.json", "config.resolved.json", "provenance.json", "checkpoints"):
            source = trained / name
            target = dest / name
            if source.is_dir():
                target.mkdir(exist_ok=True)
                for child in source.iterdir():
                    if child.is_file() and not (target / child.name).exists():
                        (target / child.name).symlink_to(child.resolve())
            elif source.is_file() and not target.exists():
                target.symlink_to(source.resolve())
        return dest
    historical = diagnostics / "historical_raw_data" / "canonical_tdec_sources" / "gap-global-slow-42-v1" / "runs" / historical_run_name(algorithm, seed)
    historical.mkdir(parents=True, exist_ok=True)
    for name in ("COMPLETE.json", "config.resolved.json", "provenance.json", "checkpoints"):
        source = trained / name
        target = historical / name
        if source.is_dir():
            target.mkdir(exist_ok=True)
            for child in source.iterdir():
                if child.is_file() and not (target / child.name).exists():
                    (target / child.name).symlink_to(child.resolve())
        elif source.is_file() and not target.exists():
            target.symlink_to(source.resolve())
    runs = diagnostics / "Modified_MADDPG_with_TDec_results" / "default" / "P5_N4_gap25" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    dest = runs / canonical_run_name(algorithm, seed)
    if not dest.exists():
        dest.symlink_to(historical.resolve())
    manifest = diagnostics / "Modified_MADDPG_with_TDec_results" / "run_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file():
        manifest.write_text(
            "algorithm,experiment,P,N,gap_m,seed,run_name,source_role,canonical_name,source_path,canonical_link,episodes\n"
            f"modified_maddpg_tdec,default,5,4,25,{seed},{historical_run_name(algorithm, seed)},gap25_default,"
            f"{canonical_run_name(algorithm, seed)},{historical},{dest},500\n",
            encoding="utf-8",
        )
    return dest


def _train_source(tmp_path: Path, algorithm: str, seed: int = 8) -> Path:
    run_name = f"tiny-{algorithm}-{seed}"
    config = _tiny_config(tmp_path / "raw", algorithm, seed=seed, run_name=run_name)
    if algorithm == "modified_maddpg_tdec":
        config = resolve_config(
            scenario="p05_n04_g25",
            seed=seed,
            episodes=2,
            steps_per_episode=100,
            device="cpu",
            actor_hidden=[16, 8],
            local_critic_hidden=[16, 8],
            global_critic_hidden=[16, 8, 4],
            batch_size=4,
            replay_capacity=32,
            checkpoint_every=1,
            checkpoint_mode="resumable",
            target_noise_sigma=0.0,
            selection_validation_seeds=[301],
            selection_validation_episodes=1,
            selection_validation_warmup_episodes=0,
            output_root=str(tmp_path / "raw"),
            run_name=run_name,
            is_formal_result=False,
        )
        assert "algorithm" not in config.to_dict()
    result = train(config)
    run_dir = Path(result["run_dir"])
    _write_complete(run_dir, algorithm, seed)
    payload = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert int(payload["episode"]) == 2
    return run_dir


def test_source_resolution_uses_canonical_and_historical_names(tmp_path):
    trained = _train_source(tmp_path, "modified_maddpg_tdec", 8)
    diagnostics = tmp_path / "diagnostics"
    dest = _install_source(diagnostics, "modified_maddpg_tdec", 8, trained)
    source = resolve_maddpg_source_run(diagnostics, "modified_maddpg_tdec", 8)
    assert source["canonical_run_name"] == "tdec_p05_n04_g25_seed08"
    assert source["is_symlink"] is True
    assert historical_run_name("modified_maddpg_tdec", 8) in source["resolved_run_dir"].name
    assert dest.is_symlink()


def test_old_tdec_loads_and_stays_isolated_from_source_run(tmp_path):
    trained = _train_source(tmp_path, "modified_maddpg_tdec", 8)
    diagnostics = tmp_path / "diagnostics"
    _install_source(diagnostics, "modified_maddpg_tdec", 8, trained)
    before_complete = (trained / "COMPLETE.json").read_text(encoding="utf-8")
    before_eval = (trained / "eval").exists()
    output = tmp_path / "e1"
    rng_before = capture_rng_state()
    result = evaluate_maddpg_e1(
        "modified_maddpg_tdec", 0.0, 8, diagnostics, output, device="cpu",
        eval_seeds=[213], eval_episodes=1, warmup_episodes=1, expected_episode=2,
    )
    rng_after = capture_rng_state()
    assert result["algorithm"] == "modified_maddpg_tdec"
    assert result["parameter_snapshot_unchanged"] is True
    assert result["wrote_to_source_run"] is False
    assert Path(result["eval_dir"]).is_relative_to(output)
    assert not str(result["eval_dir"]).startswith(str(trained))
    assert (trained / "COMPLETE.json").read_text(encoding="utf-8") == before_complete
    assert (trained / "eval").exists() is before_eval
    assert rng_before["python"] == rng_after["python"]
    assert all(np.array_equal(left, right) for left, right in zip(rng_before["numpy"], rng_after["numpy"]) if isinstance(left, np.ndarray))
    assert torch.equal(rng_before["torch"], rng_after["torch"])
    assert (output / "evaluations" / "tdec_p05_n04_g25_seed08").is_dir()


def test_algorithm1_noise_rng_is_separated_and_reproducible(tmp_path):
    trained = _train_source(tmp_path, "modified_maddpg", 8)
    diagnostics = tmp_path / "diagnostics"
    _install_source(diagnostics, "modified_maddpg", 8, trained)
    output = tmp_path / "e1"
    first = evaluate_maddpg_e1(
        "modified_maddpg", 0.3, 8, diagnostics, output / "a", device="cpu",
        eval_seeds=[213], eval_episodes=1, warmup_episodes=1, expected_episode=2,
    )
    second = evaluate_maddpg_e1(
        "modified_maddpg", 0.3, 8, diagnostics, output / "b", device="cpu",
        eval_seeds=[213], eval_episodes=1, warmup_episodes=1, expected_episode=2,
    )
    with np.load(Path(first["eval_dir"]) / "metrics.npz") as left, np.load(Path(second["eval_dir"]) / "metrics.npz") as right:
        np.testing.assert_array_equal(left["action_normalized"], right["action_normalized"])
        np.testing.assert_array_equal(left["aoi_ms"], right["aoi_ms"])
    quiet = evaluate_maddpg_e1(
        "modified_maddpg", 0.0, 8, diagnostics, output / "c", device="cpu",
        eval_seeds=[213], eval_episodes=1, warmup_episodes=1, expected_episode=2,
    )
    with np.load(Path(first["eval_dir"]) / "metrics.npz") as noisy, np.load(Path(quiet["eval_dir"]) / "metrics.npz") as silent:
        assert not np.array_equal(noisy["action_normalized"], silent["action_normalized"])


def test_latest500_helper_and_hpc_scripts():
    root = Path(__file__).resolve().parents[1]
    eval_script = (root / "hpc" / "aoi_e1_maddpg_eval_array.sbatch").read_text()
    analyze = (root / "hpc" / "aoi_e1_maddpg_analyze.sbatch").read_text()
    assert "analysis.evaluate_maddpg_e1" in eval_script
    assert "analysis.summarize_actor_sharing_e1" in analyze
    assert "E1-p5-baselines/P5_N4_gap25" in eval_script
    assert "--gres=gpu" not in analyze


def _fake_cell_arrays(seed: int, offset: float) -> dict[str, np.ndarray]:
    shape = (6, 100, 100, 5)
    aoi = np.full(shape, 6.0 + offset + 0.01 * seed)
    success = np.ones(shape)
    remaining = np.zeros(shape)
    power = np.full(shape, 10.0)
    reward_global = np.full(shape[:3], -0.1)
    reward_task1 = np.full(shape, -0.2)
    reward_task2 = np.full(shape, -0.3)
    return {
        "aoi_ms": aoi.astype(np.float32),
        "success": success.astype(np.float32),
        "remaining_demand": remaining.astype(np.float32),
        "power_dbm": power.astype(np.float32),
        "executed_power_dbm": power.astype(np.float32),
        "reward_global": reward_global.astype(np.float32),
        "reward_task1": reward_task1.astype(np.float32),
        "reward_task2": reward_task2.astype(np.float32),
    }


def _write_eval_cell(eval_dir: Path, complete: dict, arrays: dict) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(eval_dir / "metrics.npz", **arrays)
    (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    (eval_dir / "provenance.json").write_text(json.dumps({"checkpoint_name": "latest.pt"}), encoding="utf-8")


def _write_mappo_training(root: Path, structure: str, variant: str, seed: int) -> None:
    from analysis.e1_contract import mappo_run_name
    run = root / "training" / "runs" / mappo_run_name(structure, variant, seed)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.resolved.json").write_text(json.dumps({
        "algorithm": "mappo",
        "seed": seed,
        "episodes": 500,
        "mappo_variant": variant,
        "mappo_actor_sharing": structure == "shared",
        "scenario": {"id": "p05_n04_g25"},
    }), encoding="utf-8")
    (run / "COMPLETE.json").write_text(json.dumps({
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": variant,
        "reproduction_git_commit": "deadbeef",
    }), encoding="utf-8")
    (run / "policy_final.pt").write_bytes(b"fixture")


def test_summarizer_rejects_missing_duplicate_and_row_errors(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    e1 = tmp_path / "e1"
    study = diagnostics / "actor-sharing-study"
    for algorithm, label, noise, seed in (
        ("modified_maddpg", "algorithm1", 0.0, 8),
    ):
        run_name = canonical_run_name(algorithm, seed)
        complete = {
            "status": "complete", "algorithm": algorithm, "algorithm_label": label,
            "training_seed": seed, "checkpoint_name": "latest.pt", "checkpoint_episode": 500,
            "eval_protocol": "sequential_warm", "eval_episodes": 100, "eval_warmup_episodes": 5,
            "n_rb": 3, "explore": False, "wrote_to_source_run": False,
            "parameter_snapshot_unchanged": True, "eval_seeds": [213, 214, 215, 216, 217, 218],
            "noise_std": noise, "cam_bits": 32000.0, "global_actor_weight": 1.0,
            "source_run_dir": str(tmp_path / "src"),
        }
        _write_eval_cell(e1 / "evaluations" / run_name / maddpg_eval_id(noise), complete, _fake_cell_arrays(seed, 0.0))
    with pytest.raises(ValueError):
        summarize_actor_sharing_e1(diagnostics, e1)

    # Build a nearly complete fake matrix then corrupt row counts.
    for algorithm, label in (("modified_maddpg", "algorithm1"), ("modified_maddpg_tdec", "algorithm2")):
        for noise in (0.0, 0.3):
            for seed in range(8, 14):
                run_name = canonical_run_name(algorithm, seed)
                complete = {
                    "status": "complete", "algorithm": algorithm, "algorithm_label": label,
                    "training_seed": seed, "checkpoint_name": "latest.pt", "checkpoint_episode": 500,
                    "eval_protocol": "sequential_warm", "eval_episodes": 100, "eval_warmup_episodes": 5,
                    "n_rb": 3, "explore": False, "wrote_to_source_run": False,
                    "parameter_snapshot_unchanged": True, "eval_seeds": [213, 214, 215, 216, 217, 218],
                    "noise_std": noise, "cam_bits": 32000.0, "global_actor_weight": 1.0,
                    "source_run_dir": str(tmp_path / "src" / run_name),
                }
                _write_eval_cell(e1 / "evaluations" / run_name / maddpg_eval_id(noise), complete, _fake_cell_arrays(seed, noise))
    for structure, variant, root in (
        ("independent", "combined", study / "E5-sharing-critic" / "P5_N4_gap25"),
        ("shared", "combined", study / "E5-sharing-critic" / "P5_N4_gap25"),
        ("independent", "tdec", study / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25"),
        ("shared", "tdec", study / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25"),
    ):
        train_root = (
            study / "existing-evidence" / "tdec-ab-v1" / "P5_N4_gap25" if structure == "independent"
            else study / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25" if variant == "tdec"
            else study / "E5-sharing-critic" / "P5_N4_gap25"
        )
        for seed in range(8, 14):
            from analysis.e1_contract import MAPPO_EVAL_ID, mappo_run_name
            _write_mappo_training(train_root, structure, variant, seed)
            complete = {
                "status": "complete", "algorithm": "mappo", "mappo_variant": variant,
                "actor_sharing": structure == "shared", "training_seed": seed,
                "eval_seeds": [213, 214, 215, 216, 217, 218], "eval_episodes": 100,
                "eval_warmup_episodes": 5, "mappo_eval_mode": "stochastic",
                "cam_bits": 32000.0, "global_actor_weight": 1.0,
                "reproduction_git_commit": "cafebabe",
            }
            arrays = _fake_cell_arrays(seed, 1.0)
            _write_eval_cell(root / "evaluations" / mappo_run_name(structure, variant, seed) / MAPPO_EVAL_ID, complete, arrays)

    report = summarize_actor_sharing_e1(diagnostics, e1)
    assert report["status"] == "PASS"
    assert report["total_cells"] == 48
    assert report["maddpg_world_agent_rows"] == 720
    assert report["merged_world_agent_rows"] == 1440

    # Duplicate / wrong-world rejection.
    bad = e1 / "evaluations" / canonical_run_name("modified_maddpg", 8) / maddpg_eval_id(0.0)
    complete = json.loads((bad / "EVAL_COMPLETE.json").read_text())
    complete["eval_seeds"] = [201, 202, 203, 204, 205, 206]
    (bad / "EVAL_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(ValueError, match="213-218"):
        summarize_actor_sharing_e1(diagnostics, e1)
