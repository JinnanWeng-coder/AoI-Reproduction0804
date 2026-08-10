import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state, restore_rng_state
from aoi_v2x_reproduction.envs.platoon import PaperEnviron
from analysis.audit_results import audit_eval
from aoi_v2x_reproduction.runtime import runner as runner_module
from aoi_v2x_reproduction.runtime.runner import _load_checkpoint
from aoi_v2x_reproduction.runtime.runner import evaluate_from_checkpoint, train


def _small_config(root: Path, name: str):
    return resolve_config(
        scenario="p05_n04_g25",
        seed=31,
        episodes=4,
        steps_per_episode=4,
        batch_size=4,
        replay_capacity=32,
        checkpoint_every=1,
        checkpoint_mode="resumable",
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


def test_resume_repairs_interruption_between_latest_and_best_writes(tmp_path):
    config = _small_config(tmp_path / "repair", "run")
    interrupted = train(config, max_episodes=2)
    run_dir = Path(interrupted["run_dir"])
    latest_path = run_dir / "checkpoints" / "latest.pt"
    best_path = run_dir / "checkpoints" / "best.pt"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    episode = int(latest["episode"])
    selection_state = dict(latest["selection_state"])
    selection_state.update({
        "best_episode": episode,
        "best_score": 1.0,
        "best_metrics": {
            "episode": episode,
            "score": 1.0,
            "criterion": runner_module.BEST_SELECTION_CRITERION,
            "seeds": list(config.selection_validation_seeds),
        },
    })
    latest["selection_state"] = selection_state
    best_path.unlink()

    runner_module._repair_selection_best_after_resume(run_dir, latest, selection_state)

    repaired = torch.load(best_path, map_location="cpu", weights_only=False)
    assert repaired["checkpoint_role"] == "best_selection_validation"
    assert repaired["selected_episode"] == repaired["episode"] == episode
    assert repaired["training_completed"] is False


def test_verified_empty_run_recovery_accepts_an_explicit_nonformal_diagnostic(monkeypatch, tmp_path):
    config = resolve_config(
        scenario="p05_n04_g25",
        seed=3,
        output_root=str(tmp_path),
        run_name="detached_diagnostic",
        global_update_mode="detached_actor",
        diagnostics=True,
        is_formal_result=False,
        checkpoint_mode="resumable",
        smoke=False,
    )
    git = {
        "reproduction_git_commit": "diagnostic-test",
        "reproduction_git_branch": "main",
        "reproduction_git_dirty": False,
        "gpu_names": [],
        "cuda_driver": None,
    }
    monkeypatch.setattr(runner_module, "_git_metadata", lambda: dict(git))
    run_dir, is_resume = runner_module._prepare_run(config, None)

    verification = runner_module._validate_empty_run_for_reinitialization(run_dir, config)

    assert is_resume is False
    assert verification["status"] == "verified_empty_run"
    assert verification["config_hash"] == config.canonical_hash()


def test_restore_rng_state_moves_generator_byte_tensors_to_cpu(monkeypatch):
    expected_cpu = torch.get_rng_state()
    observed = {}

    class DeviceMappedState:
        def __init__(self, name):
            self.name = name
            self.detached = False
            self.moved_to_cpu = False

        def detach(self):
            self.detached = True
            return self

        def cpu(self):
            self.moved_to_cpu = True
            return expected_cpu

    cpu_rng_mapped_to_device = DeviceMappedState("torch")
    cuda_rng_mapped_to_device = DeviceMappedState("torch_cuda")
    monkeypatch.setattr(torch, "set_rng_state", lambda value: observed.setdefault("torch", value))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda values: observed.setdefault("torch_cuda", values),
    )

    restore_rng_state(
        {
            "python": __import__("random").getstate(),
            "numpy": np.random.get_state(),
            "torch": cpu_rng_mapped_to_device,
            "torch_cuda": [cuda_rng_mapped_to_device],
        }
    )

    assert cpu_rng_mapped_to_device.detached and cpu_rng_mapped_to_device.moved_to_cpu
    assert cuda_rng_mapped_to_device.detached and cuda_rng_mapped_to_device.moved_to_cpu
    assert observed["torch"] is expected_cpu
    assert observed["torch_cuda"] == [expected_cpu]


def test_checkpoint_payload_is_deserialized_on_cpu(tmp_path, monkeypatch):
    checkpoint = tmp_path / "resume.pt"
    checkpoint.write_bytes(b"placeholder")
    observed = {}

    def stop_after_load(path, *, map_location, weights_only):
        observed.update(path=path, map_location=map_location, weights_only=weights_only)
        raise RuntimeError("stop after checking deserialization device")

    monkeypatch.setattr(runner_module.torch, "load", stop_after_load)
    learner = SimpleNamespace(device=torch.device("cuda:0"))
    with pytest.raises(RuntimeError, match="stop after checking"):
        _load_checkpoint(checkpoint, SimpleNamespace(), [], learner, None, None, None)

    assert observed == {
        "path": checkpoint,
        "map_location": "cpu",
        "weights_only": False,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_checkpoint_can_resume_from_episode_one_to_two(tmp_path):
    config = resolve_config(
        scenario="p05_n04_g25",
        seed=31,
        episodes=2,
        steps_per_episode=4,
        batch_size=4,
        replay_capacity=32,
        checkpoint_every=1,
        checkpoint_mode="resumable",
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        device="cuda:0",
        output_root=str(tmp_path),
        run_name="cuda_resume",
        smoke=True,
        is_formal_result=False,
    )

    interrupted = train(config, max_episodes=1)
    checkpoint = Path(interrupted["run_dir"]) / "checkpoints" / "latest.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["episode"] == 1
    assert saved["completed"] is False
    assert saved["rng"]["torch"].device.type == "cpu"
    assert all(value.device.type == "cpu" for value in saved["rng"]["torch_cuda"])

    resumed = train(config, resume=str(checkpoint))
    assert resumed["episodes"] == 2
    completed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert completed["episode"] == 2
    assert completed["completed"] is True


def test_frozen_eval_creates_new_artifact_without_overwriting_checkpoint(tmp_path):
    config = _small_config(tmp_path / "eval", "run")
    result = train(config)
    checkpoint = Path(result["run_dir"]) / "checkpoints" / "latest.pt"
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    before_rng = capture_rng_state()
    evaluated = evaluate_from_checkpoint(config, str(checkpoint), eval_episodes=2, eval_seeds=[201, 202], eval_purpose="validation", scope="validation")
    after_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    after_rng = capture_rng_state()
    assert before_hash == after_hash
    assert before_rng["python"] == after_rng["python"]
    assert np.array_equal(before_rng["numpy"][1], after_rng["numpy"][1])
    assert before_rng["numpy"][0] == after_rng["numpy"][0]
    assert before_rng["torch"].equal(after_rng["torch"])
    assert evaluated["is_frozen_eval"] is True
    eval_dir = Path(evaluated["eval_dir"])
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["eval_seeds"] == [201, 202]
    assert summary["eval_purpose"] == "validation"
    assert summary["statistics_schema_version"] == "eval_seed_cluster_v1"
    assert summary["checkpoint"] == "checkpoints/latest.pt"
    assert summary["eval_protocol"] == "sequential_warm"
    assert summary["eval_warmup_episodes"] == 5
    assert len(summary["sd_AoI_ms_per_seed"]) == 2
    assert len(summary["ci95_AoI_ms_per_seed"]) == 2
    with np.load(eval_dir / "metrics.npz", allow_pickle=False) as arrays:
        assert arrays["aoi_ms"].shape[:2] == (2, 2)
        assert np.all(np.isfinite(arrays["aoi_ms"]))


def test_validation_and_final_test_artifacts_are_separate(tmp_path):
    config = _small_config(tmp_path / "purpose", "run")
    result = train(config)
    checkpoint = Path(result["run_dir"]) / "checkpoints" / "latest.pt"
    validation = evaluate_from_checkpoint(config, str(checkpoint), eval_episodes=1, eval_seeds=[201, 202], eval_purpose="validation", scope="validation")
    with pytest.raises(ValueError, match="frozen reproduction baseline checkpoint"):
        evaluate_from_checkpoint(config, str(checkpoint), eval_episodes=1, eval_seeds=[101, 102], eval_purpose="final_test", scope="final_release")
    assert validation["eval_purpose"] == "validation"
    assert validation["is_formal_result"] is False


def test_v1_checkpoint_is_rejected_before_loading_state(tmp_path):
    config = _small_config(tmp_path / "v1", "run")
    path = tmp_path / "v1.pt"
    torch.save({"config_hash": config.canonical_hash(), "config": config.to_dict()}, path)
    with pytest.raises(ValueError, match="semantic_version"):
        _load_checkpoint(path, config, [], SimpleNamespace(device=torch.device("cpu")), None, None, None)


def test_paper_v3_checkpoint_is_rejected_by_v4_loader(tmp_path):
    config = _small_config(tmp_path / "v3", "run")
    path = tmp_path / "paper_v3.pt"
    torch.save({
        "checkpoint_version": 3,
        "semantic_version": "paper_faithful_v3",
        "config_hash": config.canonical_hash(),
        "config": config.to_dict(),
    }, path)
    with pytest.raises(ValueError, match="semantic_version"):
        _load_checkpoint(path, config, [], SimpleNamespace(device=torch.device("cpu")), None, None, None)


def test_eval_uses_one_cold_reset_and_sequential_episode_indices(tmp_path, monkeypatch):
    config = _small_config(tmp_path / "sequence", "run")
    result = train(config)
    checkpoint = Path(result["run_dir"]) / "checkpoints" / "latest.pt"
    resets = []
    starts = []

    class TrackingEnvironment(PaperEnviron):
        def reset_world(self, seed=None):
            resets.append(int(seed))
            return super().reset_world(seed)

        def start_episode(self, episode_index, update_mobility=True):
            starts.append(int(episode_index))
            return super().start_episode(episode_index, update_mobility)

    monkeypatch.setattr(runner_module, "PaperEnviron", TrackingEnvironment)
    runner_module.evaluate_from_checkpoint(config, str(checkpoint), eval_episodes=2, eval_seeds=[201, 202], eval_purpose="validation", scope="validation")
    assert resets == [201, 202]
    assert starts == list(range(7)) + list(range(7))


def test_best_checkpoint_is_selected_not_an_alias_of_latest(tmp_path, monkeypatch):
    config = _small_config(tmp_path / "selection", "run")
    calls = []

    def fake_selection(config, agents):
        episode = len(calls) + 1
        calls.append(episode)
        score = 10.0 if episode == 1 else float(-episode)
        return {
            "criterion": runner_module.BEST_SELECTION_CRITERION,
            "seeds": list(config.selection_validation_seeds),
            "scored_episodes": config.selection_validation_episodes,
            "warmup_episodes": config.selection_validation_warmup_episodes,
            "mean_aoi_ms_per_agent": [float(episode)] * config.number_agents,
            "endpoint_cam_per_agent": [0.5] * config.number_agents,
            "worst_agent_mean_aoi_ms": float(episode),
            "worst_agent_endpoint_cam": 0.5,
            "score": score,
        }

    monkeypatch.setattr(runner_module, "_selection_validation", fake_selection)
    result = train(config)
    run = Path(result["run_dir"])
    best = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    latest = torch.load(run / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    assert best["checkpoint_role"] == "best_selection_validation"
    assert best["selected_episode"] == best["episode"] == 1
    assert best["completed"] is True and best["training_completed"] is True
    assert latest["checkpoint_role"] == "latest_final"
    assert latest["episode"] == config.episodes
    assert len(list((run / "checkpoints" / "periodic").glob("*_actor.pt"))) == config.episodes


def test_eval_noise_sweep_can_coexist_and_is_bounded(tmp_path):
    config = _small_config(tmp_path / "noise", "run")
    result = train(config)
    checkpoint = Path(result["run_dir"]) / "checkpoints" / "best.pt"
    deterministic = evaluate_from_checkpoint(
        config, str(checkpoint), eval_episodes=1, eval_seeds=[201], eval_purpose="validation", scope="validation", eval_noise=0.0
    )
    noisy = evaluate_from_checkpoint(
        config, str(checkpoint), eval_episodes=1, eval_seeds=[201], eval_purpose="validation", scope="validation", eval_noise=0.05
    )
    assert deterministic["eval_noise"] == 0.0
    assert noisy["eval_noise"] == 0.05
    assert deterministic["eval_dir"] != noisy["eval_dir"]
    assert not (Path(deterministic["eval_dir"]) / "metrics.mat").exists()
    assert not (Path(noisy["eval_dir"]) / "metrics.mat").exists()
    assert noisy["release_status"] == "diagnostic_evaluation"
    with pytest.raises(ValueError, match="eval_noise"):
        evaluate_from_checkpoint(
            config, str(checkpoint), eval_episodes=1, eval_seeds=[201], eval_purpose="validation", scope="validation", eval_noise=1.01
        )


def test_diagnostic_noise_zero_evaluates_best_and_latest_without_release_marker(tmp_path):
    config = _small_config(tmp_path / "diagnostic_checkpoints", "run")
    config.diagnostics = True
    result = train(config)
    run_dir = Path(result["run_dir"])

    best = evaluate_from_checkpoint(
        config,
        str(run_dir / "checkpoints" / "best.pt"),
        eval_episodes=1,
        eval_seeds=[201],
        eval_purpose="validation",
        scope="validation",
        eval_noise=0.0,
        diagnostic_eval=True,
    )
    latest = evaluate_from_checkpoint(
        config,
        str(run_dir / "checkpoints" / "latest.pt"),
        eval_episodes=1,
        eval_seeds=[201],
        eval_purpose="validation",
        scope="validation",
        eval_noise=0.0,
        diagnostic_eval=True,
    )

    assert best["release_status"] == latest["release_status"] == "diagnostic_evaluation"
    assert best["diagnostic_evaluation"] is True
    assert latest["diagnostic_evaluation"] is True
    assert best["eval_dir"] != latest["eval_dir"]
    assert not (run_dir / "VALIDATION_READY.json").exists()
    assert Path(best["eval_dir"], "summary.json").is_file()
    assert Path(latest["eval_dir"], "summary.json").is_file()


def test_training_diagnostics_alone_keep_normal_validation_lifecycle(tmp_path):
    config = _small_config(tmp_path / "instrumented_validation", "run")
    config.diagnostics = True
    result = train(config)
    run_dir = Path(result["run_dir"])

    evaluated = evaluate_from_checkpoint(
        config,
        str(run_dir / "checkpoints" / "best.pt"),
        eval_episodes=1,
        eval_seeds=[201],
        eval_purpose="validation",
        scope="validation",
        eval_noise=0.0,
    )

    assert evaluated["release_status"] == "validation_ready"
    assert evaluated["diagnostic_evaluation"] is False
    assert (run_dir / "VALIDATION_READY.json").is_file()
