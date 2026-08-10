from pathlib import Path

import torch

from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime.runner import train


def _tiny_config(root: Path, mode: str):
    return resolve_config(
        scenario="p05_n04_g25",
        seed=71,
        episodes=2,
        steps_per_episode=3,
        batch_size=4,
        replay_capacity=16,
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        device="cpu",
        output_root=str(root),
        run_name=f"artifact-{mode}",
        checkpoint_mode=mode,
    )


def test_default_policy_only_writes_one_actor_artifact_without_replay(tmp_path):
    config = _tiny_config(tmp_path, "policy_only")
    result = train(config)
    run_dir = Path(result["run_dir"])
    assert not (run_dir / "checkpoints").exists()
    artifacts = list(run_dir.glob("*.pt"))
    assert [path.name for path in artifacts] == ["policy_final.pt"]
    payload = torch.load(artifacts[0], map_location="cpu", weights_only=False)
    assert payload["artifact_type"] == "policy_only"
    assert payload["episode"] == config.episodes
    assert len(payload["actors"]) == config.number_agents
    assert "replay" not in payload
    assert "learner" not in payload
    assert "rng" not in payload


def test_none_mode_writes_no_torch_artifact(tmp_path):
    config = _tiny_config(tmp_path, "none")
    result = train(config)
    run_dir = Path(result["run_dir"])
    assert not (run_dir / "checkpoints").exists()
    assert list(run_dir.glob("*.pt")) == []
