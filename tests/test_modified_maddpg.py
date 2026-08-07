import json
from pathlib import Path

import numpy as np
import pytest
import torch

from Classes.Environment_Platoon import PaperEnviron, power_penalty
from config import DEFAULT_ALGORITHM, build_parser, config_from_args, resolve_config
from global_critic import Global_Critic
from local_critic import Agent
from runner import train
from analysis.summarize_modified_maddpg_gap_extension import (
    EXPECTED_GAPS,
    EXTENSION_GAPS,
    summarize as summarize_gap_extension,
)
from analysis.summarize_modified_maddpg_platoon_size_extension import (
    EXPECTED_SIZES,
    EXTENSION_SIZES,
    summarize as summarize_platoon_size_extension,
)
from analysis.summarize_modified_maddpg_default import EXPECTED_SEEDS, summarize


def _config(tmp_path=None, diagnostics=False):
    return resolve_config(
        "paper_faithful",
        "p05_n04_g25",
        algorithm="modified_maddpg",
        seed=8,
        episodes=2,
        steps_per_episode=3,
        device="cpu",
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        batch_size=4,
        replay_capacity=32,
        checkpoint_every=1,
        target_noise_sigma=0.0,
        diagnostics=diagnostics,
        selection_validation_seeds=[301],
        selection_validation_episodes=1,
        selection_validation_warmup_episodes=0,
        output_root=None if tmp_path is None else str(tmp_path),
        run_name=None if tmp_path is None else "modified-smoke",
        is_formal_result=False,
    )


def _system(config):
    config.device_resolved = "cpu"
    agents = [Agent(config, index) for index in range(config.number_agents)]
    return agents, Global_Critic(config, agents)


def _batch(config):
    rng = np.random.default_rng(81)
    states = rng.normal(size=(4, config.state_dim * config.number_agents)).astype(np.float32)
    next_states = rng.normal(size=states.shape).astype(np.float32)
    actions = rng.uniform(-0.5, 0.5, size=(4, config.action_dim * config.number_agents)).astype(np.float32)
    task1 = rng.normal(size=(4, config.number_agents)).astype(np.float32)
    task2 = rng.normal(size=(4, config.number_agents)).astype(np.float32)
    return states, actions, np.zeros(4, dtype=np.float32), task1, task2, next_states, np.zeros(4, dtype=bool)


def test_algorithm1_is_explicit_and_default_tdec_hash_remains_implicit():
    tdec = resolve_config("paper_faithful", "p05_n04_g25")
    modified = resolve_config("paper_faithful", "p05_n04_g25", algorithm="modified_maddpg")
    assert tdec.algorithm == DEFAULT_ALGORITHM
    assert "algorithm" not in tdec.to_dict()
    assert modified.to_dict()["algorithm"] == "modified_maddpg"
    assert modified.canonical_hash() != tdec.canonical_hash()
    assert modified.is_formal_result is False

    args = build_parser().parse_args(["--algorithm", "modified_maddpg", "--seed", "8"])
    assert config_from_args(args).algorithm == "modified_maddpg"
    with pytest.raises(ValueError, match="paper_faithful"):
        resolve_config("legacy_release", "p05_n04_g25", algorithm="modified_maddpg")


def test_algorithm1_has_exactly_one_local_critic_per_agent():
    config = _config()
    agents, _learner = _system(config)
    for agent in agents:
        assert len(agent.local_critics) == 1
        assert len(agent.target_local_critics) == 1
        assert hasattr(agent, "critic") and hasattr(agent, "target_critic")
        assert not hasattr(agent, "critic_task1")
        assert not hasattr(agent, "critic_task2")


def test_recorded_task_rewards_sum_to_the_algorithm1_holistic_reward():
    config = _config()
    environment = PaperEnviron(config)
    environment.reset_world(config.seed)
    environment.start_episode(0)
    actions = np.zeros((config.number_agents, config.action_dim), dtype=np.float32)
    actions[::2, 1] = -0.9
    actions[1::2, 1] = 0.9
    _next, _global, task1, task2, _done, info = environment.step(actions)
    expected = (
        -4.95 * np.asarray(info["remaining_demand"]) / float(config.cam_bits)
        - np.asarray(info["aoi_ms"]) / 20.0
        + 0.05 * (np.asarray(info["v2i_rate"]) >= environment.v2i_min)
        - np.asarray([power_penalty(value) for value in info["power_dbm"]])
    )
    np.testing.assert_allclose(task1 + task2, expected, rtol=0.0, atol=1e-6)


def test_algorithm1_local_td_update_depends_only_on_sum_of_task_rewards():
    torch.manual_seed(91)
    config_a = _config()
    agents_a, learner_a = _system(config_a)
    config_b = _config()
    agents_b, learner_b = _system(config_b)
    for left, right in zip(agents_a, agents_b):
        right.load_state_dict_full(left.state_dict_full())
    learner_b.load_state_dict_full(learner_a.state_dict_full())

    batch = _batch(config_a)
    summed_batch = (batch[0], batch[1], batch[2], batch[3] + batch[4], np.zeros_like(batch[4]), batch[5], batch[6])
    for _ in range(2):
        learner_a.learn(batch)
        learner_b.learn(summed_batch)

    for left, right in zip(agents_a, agents_b):
        for left_parameter, right_parameter in zip(left.critic.parameters(), right.critic.parameters()):
            torch.testing.assert_close(left_parameter, right_parameter, rtol=0.0, atol=0.0)


def test_algorithm1_synchronous_global_gradient_reaches_every_actor():
    config = _config(diagnostics=True)
    _agents, learner = _system(config)
    batch = _batch(config)
    audit = learner.actor_global_gradient_audit(batch[0])
    assert audit["finite"] is True
    assert all(value > 0.0 for value in audit["global_gradient_norms"])
    learner.learn(batch)
    result = learner.learn(batch)
    assert set(result["actor_gradient_diagnostics"]) >= {
        "local_grad_l2",
        "global_grad_l2",
        "global_to_local_ratio",
        "global_vs_local_cosine",
    }


def test_algorithm_checkpoints_cannot_cross_load_between_algorithms():
    modified = _config()
    modified_agents, _modified_learner = _system(modified)
    tdec = resolve_config(
        "paper_faithful",
        "p05_n04_g25",
        episodes=2,
        steps_per_episode=3,
        device="cpu",
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        batch_size=4,
        replay_capacity=32,
        is_formal_result=False,
    )
    tdec.device_resolved = "cpu"
    tdec_agent = Agent(tdec, 0)
    with pytest.raises(ValueError, match="algorithm mismatch"):
        tdec_agent.load_state_dict_full(modified_agents[0].state_dict_full())


def test_algorithm1_tiny_training_completes_and_records_identity(tmp_path):
    config = _config(tmp_path=tmp_path, diagnostics=True)
    partial = train(config, max_episodes=1)
    run_dir = Path(partial["run_dir"])
    assert partial["interrupted"] is True
    result = train(config, resume=str(run_dir / "checkpoints" / "latest.pt"))
    assert result["episodes"] == 2
    complete = json.loads((run_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["algorithm"] == "modified_maddpg"
    checkpoint = torch.load(run_dir / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["algorithm"] == "modified_maddpg"
    assert checkpoint["config"]["algorithm"] == "modified_maddpg"
    assert (run_dir / "diagnostics" / "actor_gradient_episode.npz").is_file()


def test_default_summarizer_validates_six_cells_and_separates_binary_from_payload(tmp_path):
    runs_root = tmp_path / "runs"
    for seed in EXPECTED_SEEDS:
        run_name = f"modified_maddpg_default_p05_n04_g25_seed{seed:02d}"
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True)
        config = resolve_config(
            "paper_faithful",
            "p05_n04_g25",
            algorithm="modified_maddpg",
            seed=seed,
            tau=0.005,
            slow_update_every_episodes=1,
            global_update_mode="synchronous_joint",
            is_formal_result=False,
        )
        (run_dir / "config.resolved.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
        (run_dir / "COMPLETE.json").write_text(json.dumps({
            "status": "complete",
            "algorithm": "modified_maddpg",
            "checkpoint_selection": {"best_episode": 450},
        }), encoding="utf-8")
        aoi = np.full((500, 5), float(seed), dtype=np.float32)
        cam = np.full((500, 5), 0.8, dtype=np.float32)
        remaining = np.full((500, 100, 5), 320.0, dtype=np.float32)
        np.savez_compressed(
            run_dir / "train_metrics.npz",
            mean_aoi_ms_episode_agent=aoi,
            endpoint_cam_episode_agent=cam,
            remaining_demand=remaining,
        )

    report = summarize(tmp_path)
    assert len(report["rows"]) == 6
    assert report["cohort"]["last100_mean_aoi_ms"] == pytest.approx(10.5)
    assert report["cohort"]["last100_mean_binary_cam"] == pytest.approx(0.8)
    assert report["cohort"]["last100_mean_payload_completion"] == pytest.approx(0.99)
    assert len(report["per_episode"]) == 3000


def test_gap_summarizer_combines_eighteen_new_cells_with_six_default_cells(tmp_path):
    extension_root = tmp_path / "gap-extension"
    default_root = tmp_path / "default"
    for gap in EXPECTED_GAPS:
        scenario = f"p05_n04_g{gap:02d}"
        for seed in EXPECTED_SEEDS:
            if gap in EXTENSION_GAPS:
                run_name = f"modified_maddpg_gap_{scenario}_seed{seed:02d}"
                run_dir = extension_root / "runs" / run_name
            else:
                run_name = f"modified_maddpg_default_{scenario}_seed{seed:02d}"
                run_dir = default_root / "runs" / run_name
            run_dir.mkdir(parents=True)
            config = resolve_config(
                "paper_faithful",
                scenario,
                algorithm="modified_maddpg",
                seed=seed,
                tau=0.005,
                slow_update_every_episodes=1,
                global_update_mode="synchronous_joint",
                is_formal_result=False,
            )
            (run_dir / "config.resolved.json").write_text(
                json.dumps(config.to_dict()), encoding="utf-8"
            )
            (run_dir / "COMPLETE.json").write_text(json.dumps({
                "status": "complete",
                "algorithm": "modified_maddpg",
                "checkpoint_selection": {"best_episode": 450},
            }), encoding="utf-8")
            aoi = np.full((500, 5), float(gap), dtype=np.float32)
            cam = np.full((500, 5), 1.0 - gap / 1000.0, dtype=np.float32)
            remaining = np.full((500, 100, 5), float(gap * 10), dtype=np.float32)
            np.savez_compressed(
                run_dir / "train_metrics.npz",
                mean_aoi_ms_episode_agent=aoi,
                endpoint_cam_episode_agent=cam,
                remaining_demand=remaining,
            )

    report = summarize_gap_extension(extension_root, default_root)
    assert len(report["rows"]) == 24
    assert len(report["per_episode"]) == 12000
    assert [row["gap_m"] for row in report["by_gap"]] == list(EXPECTED_GAPS)
    assert report["by_gap"][0]["last100_mean_aoi_ms"] == pytest.approx(5.0)
    assert report["by_gap"][-1]["last100_mean_aoi_ms"] == pytest.approx(35.0)
    assert report["trend_last100"]["aoi_nondecreasing_count"] == 6
    assert report["trend_last100"]["aoi_endpoint_rise_count"] == 6
    assert report["trend_last100"]["binary_endpoint_decline_count"] == 6
    assert report["trend_last100"]["payload_endpoint_decline_count"] == 6
    reused = [row for row in report["rows"] if row["gap_m"] == 25]
    assert {row["source"] for row in reused} == {"default_reuse"}


def test_platoon_summarizer_combines_eighteen_new_cells_with_six_default_cells(tmp_path):
    extension_root = tmp_path / "platoon-size-extension"
    default_root = tmp_path / "default"
    for size in EXPECTED_SIZES:
        scenario = f"p05_n{size:02d}_g25"
        for seed in EXPECTED_SEEDS:
            if size in EXTENSION_SIZES:
                run_name = f"modified_maddpg_platoon_{scenario}_seed{seed:02d}"
                run_dir = extension_root / "runs" / run_name
            else:
                run_name = f"modified_maddpg_default_{scenario}_seed{seed:02d}"
                run_dir = default_root / "runs" / run_name
            run_dir.mkdir(parents=True)
            config = resolve_config(
                "paper_faithful",
                scenario,
                algorithm="modified_maddpg",
                seed=seed,
                tau=0.005,
                slow_update_every_episodes=1,
                global_update_mode="synchronous_joint",
                is_formal_result=False,
            )
            (run_dir / "config.resolved.json").write_text(
                json.dumps(config.to_dict()), encoding="utf-8"
            )
            (run_dir / "COMPLETE.json").write_text(json.dumps({
                "status": "complete",
                "algorithm": "modified_maddpg",
                "checkpoint_selection": {"best_episode": 450},
            }), encoding="utf-8")
            aoi = np.full((500, 5), float(size), dtype=np.float32)
            cam = np.full((500, 5), 1.0 - size / 100.0, dtype=np.float32)
            remaining = np.full((500, 100, 5), float(size * 100), dtype=np.float32)
            np.savez_compressed(
                run_dir / "train_metrics.npz",
                mean_aoi_ms_episode_agent=aoi,
                endpoint_cam_episode_agent=cam,
                remaining_demand=remaining,
            )

    report = summarize_platoon_size_extension(extension_root, default_root)
    assert len(report["rows"]) == 24
    assert len(report["per_episode"]) == 12000
    assert [row["platoon_size"] for row in report["by_size"]] == list(EXPECTED_SIZES)
    assert report["by_size"][0]["last100_mean_aoi_ms"] == pytest.approx(4.0)
    assert report["by_size"][-1]["last100_mean_aoi_ms"] == pytest.approx(10.0)
    assert report["trend_last100"]["aoi_nondecreasing_count"] == 6
    assert report["trend_last100"]["aoi_endpoint_rise_count"] == 6
    assert report["trend_last100"]["binary_endpoint_decline_count"] == 6
    assert report["trend_last100"]["payload_endpoint_decline_count"] == 6
    reused = [row for row in report["rows"] if row["platoon_size"] == 4]
    assert {row["source"] for row in reused} == {"default_reuse"}
