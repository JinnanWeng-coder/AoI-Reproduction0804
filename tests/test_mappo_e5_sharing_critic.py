import json
from pathlib import Path

import numpy as np
import pytest

from analysis.evaluate_mappo_feasibility_reward import evaluate_feasibility_reward
from analysis.mappo_e5_contract import (
    derive_shared_combined_config,
    eval_task_to_cell,
    independent_run_name,
    shared_run_name,
    validate_derived_against_parent,
)
from analysis.summarize_mappo_e5_sharing_critic import _effect_summaries, _effects
from aoi_v2x_reproduction.algorithms.mappo.trainer import MAPPOTrainer
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime.runner import train


def _write_parent(root: Path, seed: int = 8) -> Path:
    run = root / "training" / "runs" / independent_run_name("combined", seed)
    run.mkdir(parents=True)
    config = resolve_config(
        scenario="p05_n04_g25", algorithm="mappo", seed=seed, mappo_variant="combined",
        mappo_actor_sharing=False, mappo_actor_lr=0.0005, mappo_critic_lr=0.0005,
        mappo_entropy_coef_rb=0.02, mappo_entropy_coef_mode=0.02,
        mappo_entropy_coef_power=0.002, mappo_value_clip_mode="normalized",
        mappo_rollout_episodes=5, mappo_ppo_epochs=10, episodes=500,
        steps_per_episode=100, checkpoint_mode="policy_only", diagnostics=True,
        output_root=str(root / "training" / "runs"), run_name=run.name,
    )
    (run / "config.resolved.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (run / "COMPLETE.json").write_text(json.dumps({
        "status": "complete", "algorithm": "mappo", "mappo_variant": "combined", "actor_sharing": False,
    }), encoding="utf-8")
    (run / "policy_final.pt").write_bytes(b"fixture")
    np.savez_compressed(run / "train_metrics.npz", marker=np.asarray([1]))
    return run


def test_e5_fixed_names_and_eval_mapping():
    assert independent_run_name("combined", 8) == "mappo_tdec_ab_combined_p05_n04_g25_seed08"
    assert shared_run_name("combined", 13) == "mappo_e5_shared_combined_p05_n04_g25_seed13"
    assert shared_run_name("tdec", 8) == "mappo_shared_actor_tdec_p05_n04_g25_seed08"
    assert eval_task_to_cell(0) == ("independent", 8)
    assert eval_task_to_cell(5) == ("independent", 13)
    assert eval_task_to_cell(6) == ("shared", 8)
    assert eval_task_to_cell(11) == ("shared", 13)
    with pytest.raises(ValueError):
        eval_task_to_cell(12)


def test_e5_config_is_derived_from_seed_matched_parent_with_only_allowed_changes(tmp_path):
    parent_run = _write_parent(tmp_path / "existing")
    parent = json.loads((parent_run / "config.resolved.json").read_text())
    derived = derive_shared_combined_config(parent_run, tmp_path / "e5" / "training" / "runs", 8, "cpu")
    comparison = validate_derived_against_parent(parent, derived.to_dict())
    assert comparison["status"] == "PASS"
    assert set(comparison["differences"]).issubset({
        "device", "mappo_actor_sharing", "output_root", "run_name", "is_formal_result"
    })
    assert derived.mappo_variant == "combined" and derived.mappo_actor_sharing is True
    assert derived.mappo_actor_lr == derived.mappo_critic_lr == pytest.approx(0.0005)
    trainer = MAPPOTrainer(derived, "cpu")
    assert len(trainer.actors) == len(trainer.actor_optimizers) == 1
    assert trainer.critic is not None and trainer.global_critic is None


def _tiny_shared_combined(root: Path):
    return resolve_config(
        scenario="p05_n04_g25", algorithm="mappo", seed=87, episodes=1, steps_per_episode=2,
        actor_hidden=[16, 8], global_critic_hidden=[16, 8, 4], mappo_rollout_episodes=1,
        mappo_ppo_epochs=1, mappo_variant="combined", mappo_actor_sharing=True,
        mappo_value_clip_mode="normalized", device="cpu", output_root=str(root),
        run_name="tiny-e5-shared-combined", checkpoint_mode="policy_only",
    )


def test_frozen_evaluator_accepts_shared_combined_and_preserves_policy(tmp_path):
    run = Path(train(_tiny_shared_combined(tmp_path / "training"))["run_dir"])
    result = evaluate_feasibility_reward(
        run / "policy_final.pt", tmp_path / "evaluation", "baseline", device="cpu",
        eval_seeds=[213], eval_episodes=1,
    )
    assert result["mappo_variant"] == "combined"
    assert result["actor_sharing"] is True and result["actor_network_count"] == 1
    assert result["policy_parameters_unchanged"] is True
    assert Path(result["eval_dir"]).joinpath("metrics.npz").is_file()


def _effect_row(phase, structure, variant, seed, offset):
    row = {"phase": phase, "structure": structure, "variant": variant, "training_seed": seed}
    for metric in (
        "mean_aoi_ms", "worst_agent_aoi_ms", "aoi_gt50_fraction", "aoi_at_cap_fraction",
        "mean_binary_cam", "worst_agent_binary_cam", "mean_payload_completion",
        "worst_agent_payload_completion", "mean_reward_combined", "mean_power_mw",
    ):
        row[metric] = float(offset)
    return row


def test_interaction_sign_is_tdec_sharing_effect_minus_combined_sharing_effect():
    rows = []
    for seed in range(8, 14):
        rows += [
            _effect_row("heldout_stochastic", "independent", "combined", seed, 10),
            _effect_row("heldout_stochastic", "shared", "combined", seed, 13),
            _effect_row("heldout_stochastic", "independent", "tdec", seed, 20),
            _effect_row("heldout_stochastic", "shared", "tdec", seed, 25),
        ]
    effects, interactions = _effects(rows, "heldout_stochastic")
    assert len(effects) == 12 and len(interactions) == 6
    assert interactions[0]["interaction_mean_aoi_ms"] == pytest.approx((25 - 20) - (13 - 10))
    assert "tdec_minus" in interactions[0]["delta_definition"]
    sharing_summary, interaction_summary = _effect_summaries(effects, interactions)
    assert len(sharing_summary) == 2 and len(interaction_summary) == 1
    assert interaction_summary[0]["mean_interaction_mean_aoi_ms"] == pytest.approx(2.0)


def test_e5_hpc_contracts_and_paths_are_consistent():
    root = Path(__file__).resolve().parents[1]
    train_script = (root / "hpc" / "aoi_mappo_e5_shared_combined_train_array.sbatch").read_text()
    eval_script = (root / "hpc" / "aoi_mappo_e5_combined_eval_array.sbatch").read_text()
    analyze = (root / "hpc" / "aoi_mappo_e5_analyze.sbatch").read_text()
    expected_root = "actor-sharing-study/E5-sharing-critic/P5_N4_gap25"
    for script in (train_script, eval_script, analyze):
        assert expected_root in script
        assert "MAPPO_results/shared-combined-v1" not in script
    for token in ("#SBATCH --array=0-5%6", "#SBATCH --cpus-per-task=4", "#SBATCH --gres=gpu:l20:1",
                  "derive_shared_combined_config", "parent_config_comparison.json"):
        assert token in train_script
    assert "#SBATCH --array=0-11%6" in eval_script and "analysis.evaluate_mappo_feasibility_reward" in eval_script
    assert "--gres=gpu" not in analyze and "analysis.summarize_mappo_e5_sharing_critic" in analyze
