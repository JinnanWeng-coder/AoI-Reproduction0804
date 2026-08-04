import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import analysis.audit_results as audit_module
import runner as runner_module
from analysis.audit_results import audit_eval, audit_run
from config import CHECKPOINT_SCHEMA_VERSION, resolve_config


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_config(root: Path, name: str = "formal_hardening"):
    return resolve_config(
        "paper_faithful",
        "p05_n04_g25",
        seed=2,
        output_root=str(root),
        run_name=name,
        device="cpu",
        is_formal_result=True,
    )


def _current_metadata():
    return {
        "reproduction_git_commit": "commit-123",
        "reproduction_git_branch": "main",
        "reproduction_git_dirty": False,
        "reproduction_tracked_tree_sha256": "tree-123",
    }


def _runtime_provenance(config):
    manifest_hash = _digest(ROOT / "SOURCE_MANIFEST.json")
    return {
        **_current_metadata(),
        "source_manifest_sha256": manifest_hash,
        "python": "test-python",
        "numpy": "test-numpy",
        "torch": "test-torch",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "cuda_driver": None,
        "gpu_names": [],
        "profile": config.profile,
        "semantic_version": config.semantic_version,
        "config_hash": config.canonical_hash(),
        "scenario": config.scenario.id,
        "seed": config.seed,
        "is_formal_result": True,
        "smoke": False,
        "eval_protocol": config.eval_protocol,
        "eval_warmup_episodes": config.eval_warmup_episodes,
        "global_reward_normalization": config.global_reward_normalization,
        "mobility_model": config.mobility_model,
        "mobility_revision": config.mobility_revision,
        "gap_definition": config.gap_definition,
        "vehicle_length_m": config.vehicle_length_m,
        "effective_center_spacing_m": config.effective_center_spacing_m,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "statistics_schema_version": config.statistics_schema_version,
    }


def _write_formal_run(root: Path, *, episode: int = 500, completed: bool = True):
    config = _formal_config(root)
    run = root / config.run_name
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (run / "config.resolved.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    provenance = _runtime_provenance(config)
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    payload = {
        "checkpoint_version": 4,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "semantic_version": config.semantic_version,
        "mobility_revision": config.mobility_revision,
        "config_hash": config.canonical_hash(),
        "config": config.to_dict(),
        "episode": episode,
        "completed": completed,
        **{key: provenance[key] for key in (
            "reproduction_git_commit", "reproduction_git_branch", "reproduction_git_dirty",
            "reproduction_tracked_tree_sha256", "source_manifest_sha256",
        )},
    }
    for name in ("latest.pt", "best.pt"):
        torch.save(payload, checkpoint_dir / name)
    hashes = {name: _digest(checkpoint_dir / name) for name in ("latest.pt", "best.pt")}
    complete_marker = {
        **provenance,
        "status": "complete",
        "final_episode": config.episodes,
        "checkpoint_completed": True,
        "checkpoint_sha256": hashes,
    }
    (run / "COMPLETE.json").write_text(json.dumps(complete_marker), encoding="utf-8")
    return config, run, checkpoint_dir / "latest.pt", payload, provenance


def _write_minimal_eval(run: Path, config, checkpoint: Path, payload, provenance):
    eval_id = "eval_final_test_hardening"
    eval_dir = run / "eval" / eval_id
    eval_dir.mkdir(parents=True)
    np.savez_compressed(eval_dir / "metrics.npz")
    seeds = [101, 102, 103, 104, 105, 106]
    summary = {
        **provenance,
        "eval_id": eval_id,
        "eval_purpose": "final_test",
        "scope": "final_release",
        "release_status": "final_release",
        "statistics_schema_version": config.statistics_schema_version,
        "eval_seeds": seeds,
        "eval_episodes": 100,
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": 5,
        "eval_noise": 0.0,
        "training_seed": config.seed,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_sha256": _digest(checkpoint),
        "checkpoint_name": checkpoint.name,
        "checkpoint_episode": payload["episode"],
        "checkpoint_completed": payload["completed"],
        "checkpoint": "checkpoints/latest.pt",
        "checkpoint_path_is_relative_to_run": True,
        "mean_AoI_ms_per_seed": [0.0] * 6,
        "sd_AoI_ms_per_seed": [0.0] * 6,
        "endpoint_success_probability_per_seed": [1.0] * 6,
        "is_frozen_eval": True,
        "is_formal_result": True,
        "status": "complete",
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(summary), encoding="utf-8")
    (eval_dir / "provenance.json").write_text(json.dumps(summary), encoding="utf-8")
    marker = {key: summary[key] for key in ("eval_id", "eval_purpose", "scope", "checkpoint_sha256")}
    marker["status"] = "final_release"
    (run / "FINAL_RELEASE.json").write_text(json.dumps(marker), encoding="utf-8")
    return eval_dir


def test_diagnostic_override_is_nonformal_and_explicit_formal_is_rejected():
    diagnostic = resolve_config("paper_faithful", "p05_n04_g25", power_min_dbm=0.0)
    assert diagnostic.is_formal_result is False
    with pytest.raises(ValueError, match="power_min_dbm"):
        resolve_config(
            "paper_faithful", "p05_n04_g25", power_min_dbm=0.0, is_formal_result=True
        )


def test_formal_train_fails_before_run_creation_for_missing_git_or_manifest(tmp_path, monkeypatch):
    config = _formal_config(tmp_path, "missing_provenance")
    monkeypatch.setattr(runner_module, "_git_metadata", lambda: {
        "reproduction_git_commit": None,
        "reproduction_git_branch": None,
        "reproduction_git_dirty": False,
        "reproduction_tracked_tree_sha256": None,
    })
    with pytest.raises(RuntimeError, match="Git provenance field"):
        runner_module._prepare_run(config, None)
    assert not (tmp_path / config.run_name).exists()

    monkeypatch.setattr(runner_module, "_git_metadata", _current_metadata)
    monkeypatch.setattr(runner_module, "_source_manifest_digest", lambda: None)
    with pytest.raises(RuntimeError, match="SOURCE_MANIFEST"):
        runner_module._prepare_run(config, None)
    assert not (tmp_path / config.run_name).exists()


def test_eval_rejects_incomplete_or_external_checkpoint_before_eval_dir(tmp_path, monkeypatch):
    config, run, checkpoint, payload, _provenance = _write_formal_run(
        tmp_path, episode=17, completed=False
    )
    monkeypatch.setattr(runner_module, "_git_metadata", _current_metadata)
    with pytest.raises(RuntimeError, match="completed=true"):
        runner_module.evaluate_from_checkpoint(
            config, str(checkpoint), 100, [201, 202, 203, 204, 205, 206], "validation", "validation"
        )
    assert not (run / "eval").exists()

    external = tmp_path / "external.pt"
    torch.save(payload, external)
    with pytest.raises(RuntimeError, match="checkpoints/latest.pt or checkpoints/best.pt"):
        runner_module.evaluate_from_checkpoint(
            config, str(external), 100, [201, 202, 203, 204, 205, 206], "validation", "validation"
        )
    assert not (run / "eval").exists()


def test_completed_in_run_checkpoint_passes_the_precreation_gate(tmp_path, monkeypatch):
    config, run, checkpoint, payload, _provenance = _write_formal_run(tmp_path)
    monkeypatch.setattr(runner_module, "_git_metadata", _current_metadata)
    runner_module._validate_formal_eval_preconditions(run, checkpoint, payload, config)
    assert not (run / "eval").exists()


def test_audits_reject_incomplete_final_checkpoint_and_broken_binding(tmp_path, monkeypatch):
    config, run, checkpoint, payload, provenance = _write_formal_run(
        tmp_path, episode=17, completed=False
    )
    eval_dir = _write_minimal_eval(run, config, checkpoint, payload, provenance)
    eval_report = audit_eval(eval_dir)
    assert "checkpoint_not_completed" in eval_report["errors"]
    assert "checkpoint_final_episode" in eval_report["errors"]
    run_report = audit_run(run, scope="final_release")
    assert any(error.startswith("checkpoint_not_completed:") for error in run_report["errors"])
    assert any(error.startswith("checkpoint_final_episode:") for error in run_report["errors"])

    complete_path = eval_dir / "EVAL_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["eval_id"] = "wrong-eval-id"
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    assert "eval_complete_binding_mismatch:eval_id" in audit_eval(eval_dir)["errors"]

    no_manifest_root = tmp_path / "repository_without_manifest"
    no_manifest_root.mkdir()
    monkeypatch.setattr(audit_module, "ROOT", no_manifest_root)
    assert "formal_source_manifest_missing" in audit_run(run, scope="train")["errors"]


def _small_nonformal_config(root: Path, device: str):
    return resolve_config(
        "paper_faithful",
        "p05_n04_g25",
        seed=19,
        episodes=1,
        steps_per_episode=2,
        batch_size=2,
        replay_capacity=16,
        checkpoint_every=1,
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        output_root=str(root),
        run_name="runtime_device_override",
        device=device,
        is_formal_result=False,
    )


def test_runtime_device_override_preserves_training_identity_and_statistics_are_recomputed(tmp_path):
    training_config = _small_nonformal_config(tmp_path, "auto")
    trained = runner_module.train(training_config)
    run = Path(trained["run_dir"])
    checkpoint = run / "checkpoints" / "latest.pt"
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    caller_config = _small_nonformal_config(tmp_path, "cpu")
    assert caller_config.canonical_hash() != checkpoint_payload["config_hash"]
    evaluated = runner_module.evaluate_from_checkpoint(
        caller_config,
        str(checkpoint),
        eval_episodes=2,
        eval_seeds=[201, 202],
        eval_purpose="validation",
        scope="validation",
    )
    eval_dir = Path(evaluated["eval_dir"])
    assert evaluated["config_hash"] == checkpoint_payload["config_hash"]
    assert evaluated["training_device_config"] == "auto"
    assert evaluated["evaluation_device_requested"] == "cpu"
    assert evaluated["evaluation_device_resolved"] == "cpu"
    assert audit_eval(eval_dir)["ok"] is True

    summary_path = eval_dir / "summary.json"
    complete_path = eval_dir / "EVAL_COMPLETE.json"
    provenance_path = eval_dir / "provenance.json"
    original_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    original_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    tampered = dict(original_summary)
    tampered["mean_AoI_ms"] = float(tampered["mean_AoI_ms"]) + 10.0
    summary_path.write_text(json.dumps(tampered), encoding="utf-8")
    report = audit_eval(eval_dir)
    assert "eval_complete_not_identical_to_summary" in report["errors"]
    assert "summary_metric_mismatch:mean_AoI_ms" in report["errors"]

    complete_path.write_text(json.dumps(tampered), encoding="utf-8")
    report = audit_eval(eval_dir)
    assert "eval_complete_not_identical_to_summary" not in report["errors"]
    assert "summary_metric_mismatch:mean_AoI_ms" in report["errors"]

    formal_flag_tamper = dict(original_summary)
    formal_flag_tamper["is_formal_result"] = True
    summary_path.write_text(json.dumps(formal_flag_tamper), encoding="utf-8")
    complete_path.write_text(json.dumps(formal_flag_tamper), encoding="utf-8")
    assert "eval_formal_marker_mismatch" in audit_eval(eval_dir)["errors"]

    final_tamper = dict(original_summary)
    final_tamper.update({
        "eval_purpose": "final_test",
        "scope": "final_release",
        "release_status": "evaluation_complete",
    })
    final_provenance = dict(original_provenance)
    final_provenance.update({
        "eval_purpose": "final_test",
        "scope": "final_release",
        "release_status": "evaluation_complete",
    })
    summary_path.write_text(json.dumps(final_tamper), encoding="utf-8")
    complete_path.write_text(json.dumps(final_tamper), encoding="utf-8")
    provenance_path.write_text(json.dumps(final_provenance), encoding="utf-8")
    assert "final_test_requires_formal_training" in audit_eval(eval_dir)["errors"]
    assert "final_release_requires_formal_paper_training" in audit_run(run, scope="final_release")["errors"]


def test_legacy_profile_defaults_nonformal_trains_normally_and_cannot_final_test(tmp_path):
    legacy = resolve_config(
        "legacy_release",
        "p05_n04_g25",
        seed=23,
        episodes=1,
        steps_per_episode=2,
        batch_size=2,
        replay_capacity=16,
        checkpoint_every=1,
        actor_hidden=[16, 8],
        local_critic_hidden=[16, 8],
        global_critic_hidden=[16, 8, 4],
        output_root=str(tmp_path),
        run_name="legacy_nonformal",
        device="cpu",
    )
    assert legacy.is_formal_result is False
    result = runner_module.train(legacy)
    run = Path(result["run_dir"])
    checkpoint = run / "checkpoints" / "latest.pt"
    assert (run / "COMPLETE.json").is_file()
    with pytest.raises(ValueError, match="formal training checkpoint"):
        runner_module.evaluate_from_checkpoint(
            legacy,
            str(checkpoint),
            eval_episodes=1,
            eval_seeds=[101],
            eval_purpose="final_test",
            scope="final_release",
        )
    assert not (run / "eval").exists()
