import hashlib
import json
from pathlib import Path

import numpy as np

from config import resolve_config
from runner import evaluate_from_checkpoint, train


def _small_config(root: Path, name: str):
    return resolve_config(
        "paper_faithful",
        "p05_n04_g25",
        seed=31,
        episodes=4,
        steps_per_episode=4,
        batch_size=4,
        replay_capacity=32,
        checkpoint_every=1,
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        device="cpu",
        output_root=str(root),
        run_name=name,
        smoke=True,
        is_formal_result=False,
    )


def test_checkpoint_resume_matches_uninterrupted_run(tmp_path):
    uninterrupted = _small_config(tmp_path / "uninterrupted", "run")
    interrupted = _small_config(tmp_path / "interrupted", "run")
    train(uninterrupted)
    first_half = train(interrupted, max_episodes=2)
    assert first_half["interrupted"] is True
    checkpoint = Path(first_half["run_dir"]) / "checkpoints" / "latest.pt"
    resumed = train(interrupted, resume=str(checkpoint))
    assert resumed["episodes"] == 4
    with np.load(Path(uninterrupted.output_root) / "run" / "train_metrics.npz", allow_pickle=False) as expected, np.load(Path(interrupted.output_root) / "run" / "train_metrics.npz", allow_pickle=False) as actual:
        assert set(expected.files) == set(actual.files)
        for key in expected.files:
            np.testing.assert_array_equal(expected[key], actual[key], err_msg=key)


def test_frozen_eval_creates_new_artifact_without_overwriting_checkpoint(tmp_path):
    config = _small_config(tmp_path / "eval", "run")
    result = train(config)
    checkpoint = Path(result["run_dir"]) / "checkpoints" / "latest.pt"
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    evaluated = evaluate_from_checkpoint(config, str(checkpoint), eval_episodes=2, eval_seeds=[101, 102])
    after_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert before_hash == after_hash
    assert evaluated["is_frozen_eval"] is True
    eval_dir = Path(evaluated["eval_dir"])
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["eval_seeds"] == [101, 102]
    with np.load(eval_dir / "metrics.npz", allow_pickle=False) as arrays:
        assert arrays["aoi_ms"].shape[:2] == (2, 2)
        assert np.all(np.isfinite(arrays["aoi_ms"]))

