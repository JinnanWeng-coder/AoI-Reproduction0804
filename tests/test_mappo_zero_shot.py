from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.audit_mappo_service_regularity import interval_statistics
from analysis.evaluate_mappo_feasibility_reward import evaluate_feasibility_reward
from analysis.evaluate_mappo_zero_shot import (
    FrozenActorPolicy,
    build_target_config,
    evaluate_zero_shot,
    summarize_arrays,
    task_to_target_structure_seed,
)
from aoi_v2x_reproduction.algorithms.mappo.networks import HybridActor
from aoi_v2x_reproduction.config import DEFAULT_SCENARIOS, resolve_config
from aoi_v2x_reproduction.runtime.runner import train


def _tiny_config(root: Path, sharing: bool):
    return resolve_config(
        scenario="p05_n04_g25", algorithm="mappo", seed=83, episodes=1, steps_per_episode=2,
        actor_hidden=[16, 8], local_critic_hidden=[16, 8], global_critic_hidden=[16, 8, 4],
        mappo_rollout_episodes=1, mappo_ppo_epochs=1, mappo_variant="tdec",
        mappo_actor_sharing=sharing, device="cpu", output_root=str(root),
        run_name=f"tiny-zero-shot-{'shared' if sharing else 'independent'}", checkpoint_mode="policy_only",
    )


@pytest.fixture(scope="module")
def shared_policy(tmp_path_factory):
    root = tmp_path_factory.mktemp("zero_shot_shared")
    run = Path(train(_tiny_config(root, True))["run_dir"])
    return run / "policy_final.pt"


@pytest.fixture(scope="module")
def independent_policy(tmp_path_factory):
    root = tmp_path_factory.mktemp("zero_shot_independent")
    run = Path(train(_tiny_config(root, False))["run_dir"])
    return run / "policy_final.pt"


def test_fixed_30_cell_mapping_and_scenario_is_not_in_formal_defaults():
    assert task_to_target_structure_seed(0) == ("p05_n04_g05", "independent", 8)
    assert task_to_target_structure_seed(6) == ("p05_n04_g05", "shared", 8)
    assert task_to_target_structure_seed(12) == ("p05_n04_g35", "independent", 8)
    assert task_to_target_structure_seed(23) == ("p05_n04_g35", "shared", 13)
    assert task_to_target_structure_seed(24) == ("p07_n04_g25", "shared", 8)
    assert task_to_target_structure_seed(29) == ("p07_n04_g25", "shared", 13)
    with pytest.raises(ValueError):
        task_to_target_structure_seed(30)
    assert "p07_n04_g25" not in DEFAULT_SCENARIOS
    assert resolve_config(scenario="p07_n04_g25").number_agents == 7


def test_shared_policy_accepts_dynamic_p_but_independent_rejects_it():
    shared = _tiny_config(Path("unused"), True)
    independent = _tiny_config(Path("unused"), False)
    target = build_target_config(shared, "p07_n04_g25", "cpu")
    assert target.number_agents == 7 and target.scenario.gap_m == 25.0
    with pytest.raises(ValueError, match="requires a shared actor"):
        build_target_config(independent, "p07_n04_g25", "cpu")
    incompatible = resolve_config(
        scenario="p05_n06_g25", algorithm="mappo", mappo_variant="tdec", mappo_actor_sharing=True
    )
    with pytest.raises(ValueError, match="state_dim"):
        build_target_config(incompatible, "p07_n04_g25", "cpu")


def test_actor_only_sampler_uses_one_shared_actor_without_copying_actions():
    config = _tiny_config(Path("unused"), True)
    actor = HybridActor(config.state_dim, config.actor_hidden, config.n_rb, config.n_modes)
    frozen = FrozenActorPolicy(config, [actor.state_dict()], torch.device("cpu"))
    observations = np.stack([np.full(config.state_dim, index, dtype=np.float32) for index in range(7)])
    snapshot = frozen.snapshot()
    actions = frozen.act(observations)
    assert actions.shape == (7, 3) and len(frozen.actors) == 1
    assert frozen.unchanged(snapshot)
    assert not np.shares_memory(actions[0], actions[1])


def test_dynamic_summary_uses_all_seven_agents_and_linear_power():
    shape = (1, 2, 3, 7)
    arrays = {
        "aoi_ms": np.broadcast_to(np.arange(1, 8), shape).astype(np.float32),
        "success": np.ones(shape, dtype=np.float32),
        "remaining_demand": np.zeros(shape, dtype=np.float32),
        "reward_global": np.ones(shape[:-1], dtype=np.float32),
        "reward_task1": np.ones(shape, dtype=np.float32),
        "reward_task2": np.ones(shape, dtype=np.float32),
        "reward_combined": np.ones(shape, dtype=np.float32) * 3,
        "executed_power_dbm": np.broadcast_to(np.asarray([0, 10, 0, 10, 0, 10, 0]), shape),
        "rb": np.broadcast_to(np.arange(7) % 3, shape),
    }
    result = summarize_arrays(arrays, 4000.0)
    assert result["mean_AoI_ms"] == pytest.approx(4.0)
    assert result["worst_agent_mean_AoI_ms"] == pytest.approx(7.0)
    assert result["mean_executed_power_mw"] == pytest.approx(34 / 7)


def test_service_intervals_join_episodes_and_keep_censored_boundaries_separate():
    reset = np.zeros((2, 6), dtype=bool)
    reset.reshape(-1)[[2, 8, 11]] = True
    result = interval_statistics(reset)
    assert result["complete_interval_count"] == 2
    assert result["complete_interval_sum_L"] == 9
    assert result["complete_interval_sum_L2"] == 45
    assert result["left_censored_slots"] == 2
    assert result["right_censored_slots"] == 0
    no_reset = interval_statistics(np.zeros((2, 6), dtype=bool))
    assert no_reset["no_reset_stream"] is True
    assert no_reset["left_censored_slots"] == 12 and no_reset["right_censored_slots"] == 12


def test_actor_only_same_scenario_exactly_matches_existing_entry(tmp_path, shared_policy):
    old = evaluate_feasibility_reward(
        shared_policy, tmp_path / "old", "baseline", device="cpu", eval_seeds=[213], eval_episodes=1
    )
    result = evaluate_zero_shot(
        shared_policy, tmp_path / "new", "p05_n04_g25", device="cpu", eval_seeds=[213],
        eval_episodes=1, reference_metrics=Path(old["eval_dir"]) / "metrics.npz",
    )
    assert result["reference_common_arrays_exact"] is True
    assert result["critic_constructed"] is False and result["optimizer_constructed"] is False
    assert result["source_scenario"] == result["target_scenario"] == "p05_n04_g25"


def test_independent_policy_loads_and_target_gap_is_real_environment_config(tmp_path, independent_policy):
    result = evaluate_zero_shot(
        independent_policy, tmp_path / "gap", "p05_n04_g05", device="cpu",
        eval_seeds=[213], eval_episodes=1,
    )
    assert result["actor_structure"] == "independent" and result["actor_network_count"] == 5
    assert result["source_scenario"] == "p05_n04_g25" and result["target_scenario"] == "p05_n04_g05"
    assert result["target_config"]["scenario"]["gap_m"] == pytest.approx(5.0)
    assert "p05_n04_g05/independent" in result["eval_dir"].replace("\\", "/")


def test_shared_policy_runs_on_p7_with_seven_agent_metadata(tmp_path, shared_policy):
    result = evaluate_zero_shot(
        shared_policy, tmp_path / "p7", "p07_n04_g25", device="cpu", eval_seeds=[213], eval_episodes=1
    )
    assert result["source_number_agents"] == 5 and result["target_number_agents"] == 7
    assert result["actor_network_count"] == 1 and result["policy_parameters_unchanged"] is True
    with np.load(Path(result["eval_dir"]) / "metrics.npz", allow_pickle=False) as arrays:
        assert arrays["aoi_ms"].shape == (1, 1, 2, 7)
        assert arrays["action_normalized"].shape == (1, 1, 2, 7, 3)


def test_zero_shot_hpc_contracts():
    root = Path(__file__).resolve().parents[1]
    evaluate = (root / "hpc" / "aoi_mappo_zero_shot_eval_array.sbatch").read_text(encoding="utf-8")
    analyze = (root / "hpc" / "aoi_mappo_zero_shot_analyze.sbatch").read_text(encoding="utf-8")
    for token in ("#SBATCH --array=0-29%6", "#SBATCH --cpus-per-task=4", "#SBATCH --gres=gpu:l20:1",
                  "p05_n04_g05", "p05_n04_g35", "p07_n04_g25", "analysis.evaluate_mappo_zero_shot"):
        assert token in evaluate
    assert "--gres=gpu" not in analyze
    assert "analysis.summarize_mappo_zero_shot" in analyze
    assert "analysis.audit_mappo_service_regularity" in analyze
