import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis.summarize_mappo_default import EXPECTED_SEEDS, summarize
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime.runner import train


def _config(root: Path, checkpoint_mode: str = "policy_only"):
    return resolve_config(
        scenario="p05_n04_g25",
        algorithm="mappo",
        seed=71,
        episodes=2,
        steps_per_episode=3,
        actor_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        mappo_rollout_episodes=1,
        mappo_ppo_epochs=2,
        device="cpu",
        output_root=str(root),
        run_name=f"mappo-{checkpoint_mode}",
        checkpoint_mode=checkpoint_mode,
        diagnostics=True,
    )


def test_mappo_training_writes_metrics_completion_and_policy_only(tmp_path):
    config = _config(tmp_path)
    result = train(config)
    run_dir = Path(result["run_dir"])

    assert (run_dir / "COMPLETE.json").is_file()
    assert (run_dir / "train_metrics.npz").is_file()
    assert (run_dir / "learning_diagnostics.json").is_file()
    assert not (run_dir / "checkpoints").exists()
    complete = json.loads((run_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["algorithm"] == "mappo"
    assert complete["update_count"] == 2
    assert complete["algorithm_applicability"] == {
        "polyak_tau_applicable": False,
        "external_action_noise_applicable": False,
        "global_actor_update_mode_applicable": False,
    }

    payload = torch.load(run_dir / "policy_final.pt", map_location="cpu", weights_only=False)
    assert payload["artifact_type"] == "policy_only"
    assert payload["algorithm"] == "mappo"
    assert len(payload["actors"]) == config.number_agents
    assert "critic" not in payload
    assert "optimizer" not in payload
    assert "replay" not in payload
    assert "rng" not in payload
    with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as metrics:
        assert metrics["mean_aoi_ms_episode_agent"].shape == (2, config.number_agents)


def test_mappo_rejects_resumable_checkpoint_mode(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_mode"):
        _config(tmp_path, checkpoint_mode="resumable")


def test_mappo_rejects_an_unresumable_partial_run_before_creating_it(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="partial"):
        train(config, max_episodes=1)
    assert not (tmp_path / config.run_name).exists()


def test_mappo_default_summarizer_validates_six_training_cells(tmp_path):
    for seed in EXPECTED_SEEDS:
        run_name = f"mappo_default_p05_n04_g25_seed{seed:02d}"
        run_dir = tmp_path / "runs" / run_name
        run_dir.mkdir(parents=True)
        config = resolve_config(scenario="p05_n04_g25", algorithm="mappo", seed=seed)
        (run_dir / "config.resolved.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
        (run_dir / "COMPLETE.json").write_text(json.dumps({
            "status": "complete",
            "algorithm": "mappo",
            "algorithm_applicability": {
                "polyak_tau_applicable": False,
                "external_action_noise_applicable": False,
                "global_actor_update_mode_applicable": False,
            },
        }), encoding="utf-8")
        agent_metrics = np.full((500, 5), float(seed), dtype=np.float32)
        np.savez_compressed(
            run_dir / "train_metrics.npz",
            mean_aoi_ms_episode_agent=agent_metrics,
            endpoint_cam_episode_agent=np.full((500, 5), 0.8, dtype=np.float32),
            remaining_demand=np.full((500, 100, 5), 320.0, dtype=np.float32),
            immediate_reward_proxy=np.full((500, 5), 0.25, dtype=np.float32),
            rb_entropy_normalized_episode_agent=np.full((500, 5), 0.7, dtype=np.float32),
            mode_entropy_normalized_episode_agent=np.full((500, 5), 0.6, dtype=np.float32),
        )

    report = summarize(tmp_path)
    assert len(report["rows"]) == 6
    assert len(report["per_episode"]) == 3000
    assert report["cohort"]["last100_mean_aoi_ms"] == pytest.approx(10.5)
    assert report["cohort"]["last100_mean_binary_cam"] == pytest.approx(0.8)
    assert report["cohort"]["last100_mean_payload_completion"] == pytest.approx(0.99)
