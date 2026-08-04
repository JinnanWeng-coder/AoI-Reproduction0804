import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Main import main as main_cli
from analysis.audit_results import audit_run
from config import CHECKPOINT_SCHEMA_VERSION, PAPER_MOBILITY_REVISION, PAPER_SEMANTIC_VERSION, matrix_specs, resolve_config
import runner as runner_module
from runner import evaluate_from_checkpoint
from scripts.matrix_runner import _command, _recovery_state, main as matrix_main


ROOT = Path(__file__).resolve().parents[1]


def _runtime_fields():
    return {
        "python": "synthetic-python",
        "numpy": "synthetic-numpy",
        "torch": "synthetic-torch",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "cuda_driver": None,
        "gpu_names": [],
        "reproduction_git_commit": "synthetic-commit",
        "reproduction_git_branch": "master",
        "reproduction_git_dirty": False,
        "reproduction_tracked_tree_sha256": "synthetic-tree",
        "source_manifest_sha256": "synthetic-source-manifest",
    }


def _write_formal_audit_run(root: Path, purpose=None):
    run = root / ("formal_" + (purpose or "train_only"))
    (run / "checkpoints").mkdir(parents=True)
    config = resolve_config(
        "paper_faithful",
        "p05_n04_g25",
        seed=2,
        episodes=500,
        steps_per_episode=100,
        output_root=str(root),
        run_name=run.name,
        device="cpu",
        is_formal_result=True,
    )
    config_dict = config.to_dict()
    config_hash = config.canonical_hash()
    (run / "config.resolved.json").write_text(json.dumps(config_dict), encoding="utf-8")
    provenance = {
        **_runtime_fields(),
        "profile": "paper_faithful",
        "semantic_version": PAPER_SEMANTIC_VERSION,
        "config_hash": config_hash,
        "scenario": config.scenario.id,
        "seed": 2,
        "is_formal_result": True,
        "smoke": False,
        "eval_protocol": config.eval_protocol,
        "eval_warmup_episodes": 5,
        "global_reward_normalization": config.global_reward_normalization,
        "mobility_model": config.mobility_model,
        "mobility_revision": PAPER_MOBILITY_REVISION,
        "gap_definition": config.gap_definition,
        "vehicle_length_m": 4.0,
        "effective_center_spacing_m": config.effective_center_spacing_m,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "statistics_schema_version": config.statistics_schema_version,
    }
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    episodes, steps, agents, followers, n_rb = 500, 100, 5, 3, 3
    train_base = (episodes, steps, agents)
    train_arrays = {
        "task1_step": train_base,
        "task2_step": train_base,
        "global_step": (episodes, steps),
        "task1_episode_mean": (episodes, agents),
        "task2_episode_mean": (episodes, agents),
        "local_total_episode_mean": (episodes, agents),
        "immediate_reward_proxy": (episodes, agents),
        "global_episode_sum": (episodes,),
        "global_episode_mean": (episodes,),
        "aoi_ms": train_base,
        "remaining_demand": train_base,
        "success": train_base,
        "v2i_rate": train_base,
        "v2v_rate": train_base,
        "power_dbm": train_base,
        "rb": train_base,
        "mode": train_base,
        "remaining_time_ms": train_base,
        "interference_db": train_base + (n_rb,),
        "selected_interference_db": train_base,
        "interference_linear": train_base + (n_rb,),
        "v2i_interference_linear": train_base + (n_rb,),
        "I_mode_db": train_base + (n_rb,),
        "I_v2i_linear": train_base + (n_rb,),
        "v2v_interference_linear": train_base + (followers, n_rb),
        "I_v2v_linear": train_base + (followers, n_rb),
        "v2v_rate_all": train_base + (followers,),
    }
    np.savez_compressed(run / "train_metrics.npz", **{key: np.zeros(shape, dtype=np.float32) for key, shape in train_arrays.items()})

    payload = {
        "checkpoint_version": 4,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "semantic_version": PAPER_SEMANTIC_VERSION,
        "mobility_revision": PAPER_MOBILITY_REVISION,
        "config_hash": config_hash,
        "config": config_dict,
        "completed": True,
        "reproduction_git_commit": provenance["reproduction_git_commit"],
        "reproduction_git_branch": provenance["reproduction_git_branch"],
        "reproduction_git_dirty": False,
        "reproduction_tracked_tree_sha256": provenance["reproduction_tracked_tree_sha256"],
        "source_manifest_sha256": provenance["source_manifest_sha256"],
    }
    torch.save(payload, run / "checkpoints" / "latest.pt")
    torch.save(payload, run / "checkpoints" / "best.pt")
    checkpoint_hash = hashlib.sha256((run / "checkpoints" / "latest.pt").read_bytes()).hexdigest()
    complete = {
        "status": "complete",
        "is_formal_result": True,
        "profile": "paper_faithful",
        "semantic_version": PAPER_SEMANTIC_VERSION,
        "mobility_revision": PAPER_MOBILITY_REVISION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_hash": config_hash,
        "reproduction_git_commit": provenance["reproduction_git_commit"],
        "reproduction_git_branch": provenance["reproduction_git_branch"],
        "reproduction_git_dirty": False,
        "reproduction_tracked_tree_sha256": provenance["reproduction_tracked_tree_sha256"],
        "source_manifest_sha256": provenance["source_manifest_sha256"],
        "gap_definition": config.gap_definition,
        "vehicle_length_m": 4.0,
        "effective_center_spacing_m": config.effective_center_spacing_m,
    }
    (run / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")

    if purpose is None:
        return run
    seeds = [201, 202, 203, 204, 205, 206] if purpose == "validation" else [101, 102, 103, 104, 105, 106]
    scope = "validation" if purpose == "validation" else "final_release"
    eval_dir = run / "eval" / ("eval_" + purpose)
    eval_dir.mkdir(parents=True)
    eval_base = (len(seeds), 100, 100, agents)
    eval_arrays = {
        "aoi_ms": eval_base,
        "success": eval_base,
        "remaining_demand": eval_base,
        "v2i_rate": eval_base,
        "v2v_rate": eval_base,
        "power_dbm": eval_base,
        "rb": eval_base,
        "mode": eval_base,
        "interference_db": eval_base + (n_rb,),
        "selected_interference_db": eval_base,
        "interference_linear": eval_base + (n_rb,),
        "v2i_interference_linear": eval_base + (n_rb,),
        "v2v_interference_linear": eval_base + (followers, n_rb),
        "I_v2i_linear": eval_base + (n_rb,),
        "I_v2v_linear": eval_base + (followers, n_rb),
        "I_mode_db": eval_base + (n_rb,),
        "v2v_rate_all": eval_base + (followers,),
    }
    eval_data = {key: np.zeros(shape, dtype=np.float32) for key, shape in eval_arrays.items()}
    eval_data["success"] = np.ones(eval_base, dtype=np.float32)
    np.savez_compressed(eval_dir / "metrics.npz", **eval_data)
    formal_eval = purpose == "final_test"
    summary = {
        **_runtime_fields(),
        "eval_id": "eval_" + purpose,
        "eval_purpose": purpose,
        "scope": scope,
        "release_status": "validation_ready" if purpose == "validation" else "final_release",
        "statistics_schema_version": config.statistics_schema_version,
        "eval_seeds": seeds,
        "eval_episodes": 100,
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": 5,
        "eval_noise": 0.0,
        "semantic_version": PAPER_SEMANTIC_VERSION,
        "profile": "paper_faithful",
        "scenario": config.scenario.id,
        "training_seed": 2,
        "config_hash": config_hash,
        "global_reward_normalization": config.global_reward_normalization,
        "mobility_model": config.mobility_model,
        "mobility_revision": PAPER_MOBILITY_REVISION,
        "gap_definition": config.gap_definition,
        "vehicle_length_m": 4.0,
        "effective_center_spacing_m": config.effective_center_spacing_m,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": "checkpoints/latest.pt",
        "checkpoint_path_is_relative_to_run": True,
        "mean_AoI_ms_per_seed_agent": [[0.0] * agents for _ in seeds],
        "mean_AoI_ms_per_seed": [0.0] * len(seeds),
        "sd_AoI_ms_per_seed": [0.0] * len(seeds),
        "ci95_AoI_ms_per_seed": [None] * len(seeds),
        "CAM_success_probability_per_seed_agent": [[1.0] * agents for _ in seeds],
        "CAM_success_probability_per_seed": [1.0] * len(seeds),
        "sd_CAM_success_probability_per_seed": [0.0] * len(seeds),
        "ci95_CAM_success_probability_per_seed": [None] * len(seeds),
        "endpoint_success_probability_per_seed": [1.0] * len(seeds),
        "mean_AoI_ms": 0.0,
        "CAM_success_probability": 1.0,
        "is_frozen_eval": True,
        "status": "complete",
        "is_formal_result": formal_eval,
    }
    (eval_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(summary), encoding="utf-8")
    (eval_dir / "provenance.json").write_text(json.dumps(summary), encoding="utf-8")
    marker = "VALIDATION_READY.json" if purpose == "validation" else "FINAL_RELEASE.json"
    (run / marker).write_text(json.dumps({"status": "validation_ready" if purpose == "validation" else "final_release", "scope": scope, "eval_purpose": purpose}), encoding="utf-8")
    return run


def test_formal_train_validation_and_final_release_scope_gates(tmp_path):
    train_only = _write_formal_audit_run(tmp_path, None)
    assert audit_run(train_only, scope="train")["ok"] is True
    validation_missing = audit_run(train_only, scope="validation")
    assert validation_missing["ok"] is False
    assert "validation_artifact_not_unique" in validation_missing["errors"]
    final_missing = audit_run(train_only, scope="final_release")
    assert final_missing["ok"] is False
    validation_run = _write_formal_audit_run(tmp_path / "validation", "validation")
    assert audit_run(validation_run, scope="validation")["ok"] is True
    assert audit_run(validation_run, scope="final_release")["ok"] is False
    final_run = _write_formal_audit_run(tmp_path / "final", "final_test")
    assert audit_run(final_run, scope="final_release")["ok"] is True
    assert audit_run(final_run, scope="validation")["ok"] is False


def test_formal_audit_rejects_missing_provenance_wrong_shape_and_checkpoint_commit(tmp_path):
    missing = _write_formal_audit_run(tmp_path / "missing", None)
    provenance_path = missing / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.pop("python")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    assert any(error == "provenance_missing:python" for error in audit_run(missing, scope="train")["errors"])

    wrong_shape = _write_formal_audit_run(tmp_path / "shape", None)
    with np.load(wrong_shape / "train_metrics.npz", allow_pickle=False) as arrays:
        copied = {key: value for key, value in arrays.items()}
    copied["task1_step"] = copied["task1_step"][:, :, :-1]
    np.savez_compressed(wrong_shape / "train_metrics.npz", **copied)
    assert any(error.startswith("shape:train:task1_step") for error in audit_run(wrong_shape, scope="train")["errors"])

    wrong_checkpoint = _write_formal_audit_run(tmp_path / "checkpoint", None)
    payload = torch.load(wrong_checkpoint / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)
    payload["reproduction_git_commit"] = "wrong-commit"
    torch.save(payload, wrong_checkpoint / "checkpoints" / "latest.pt")
    report = audit_run(wrong_checkpoint, scope="train")
    assert any(error == "checkpoint_git_commit:latest.pt" for error in report["errors"])


def test_main_eval_only_requires_explicit_purpose():
    with pytest.raises(SystemExit, match="explicit --eval-purpose"):
        main_cli(["--eval-only", "--scope", "validation", "--resume", "missing.pt"])


def test_formal_eval_precondition_fails_before_eval_directory_creation(tmp_path, monkeypatch):
    config = resolve_config(
        "paper_faithful",
        "p05_n04_g25",
        seed=2,
        episodes=1,
        steps_per_episode=1,
        output_root=str(tmp_path),
        run_name="formal_precondition",
        device="cpu",
        is_formal_result=True,
    )
    run = tmp_path / "formal_precondition"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.resolved.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    provenance = {
        "reproduction_git_commit": "run-commit",
        "reproduction_git_branch": "master",
        "reproduction_git_dirty": False,
        "reproduction_tracked_tree_sha256": "run-tree",
    }
    (run / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (run / "COMPLETE.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    payload = {
        "checkpoint_version": 4,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "semantic_version": PAPER_SEMANTIC_VERSION,
        "mobility_revision": PAPER_MOBILITY_REVISION,
        "config_hash": config.canonical_hash(),
        "config": config.to_dict(),
        "reproduction_git_commit": "run-commit",
        "reproduction_git_branch": "master",
        "reproduction_git_dirty": False,
        "reproduction_tracked_tree_sha256": "run-tree",
    }
    checkpoint = run / "checkpoints" / "latest.pt"
    torch.save(payload, checkpoint)
    dirty_metadata = {
        "reproduction_git_commit": "run-commit",
        "reproduction_git_branch": "master",
        "reproduction_git_dirty": True,
        "reproduction_tracked_tree_sha256": "run-tree",
    }
    monkeypatch.setattr(runner_module, "_git_metadata", lambda: dirty_metadata)
    with pytest.raises(RuntimeError, match="clean reproduction Git"):
        evaluate_from_checkpoint(config, str(checkpoint), 100, [201, 202, 203, 204, 205, 206], "validation", scope="validation")
    assert not (run / "eval").exists()

    mismatch_metadata = dict(dirty_metadata, reproduction_git_dirty=False, reproduction_git_commit="different-commit")
    monkeypatch.setattr(runner_module, "_git_metadata", lambda: mismatch_metadata)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        evaluate_from_checkpoint(config, str(checkpoint), 100, [201, 202, 203, 204, 205, 206], "validation", scope="validation")
    assert not (run / "eval").exists()


def test_matrix_default_validation_commands_and_recovery_states(tmp_path, capsys):
    matrix_main(["--dry-run", "--profile", "paper_faithful"])
    output = capsys.readouterr().out
    assert "matrix_count=48" in output
    assert "unique_count=48" in output
    assert "--scope validation" in output
    assert "--eval-purpose validation" in output

    item = matrix_specs()[0]
    args = SimpleNamespace(device="cpu", output_root=str(tmp_path), eval_purpose="validation", eval_episodes=100, eval_seeds="201,202,203,204,205,206")
    eval_command = _command(item, args, "eval", tmp_path / "checkpoints" / "latest.pt")
    assert "--scope" in eval_command and "validation" in eval_command
    assert "--eval-purpose" in eval_command and "validation" in eval_command

    no_checkpoint = tmp_path / "no_checkpoint"
    no_checkpoint.mkdir()
    assert _recovery_state(no_checkpoint, item)["code"] == "RUN_DIR_WITHOUT_CHECKPOINT"
    checkpoint_dir = tmp_path / "resume" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    payload = {"config_hash": item["config_hash"], "semantic_version": item["semantic_version"], "completed": False}
    torch.save(payload, checkpoint_dir / "latest.pt")
    assert _recovery_state(checkpoint_dir.parent, item)["code"] == "INCOMPLETE_RESUME_AVAILABLE"
    (checkpoint_dir.parent / "COMPLETE.json").write_text("{}", encoding="utf-8")
    assert _recovery_state(checkpoint_dir.parent, item)["code"] == "COMPLETE"


def test_source_manifest_read_only_verification():
    manifest_path = ROOT / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = Path(manifest["source_root"])
    before = {entry["path"]: (source_root / entry["path"]).stat().st_mtime_ns for entry in manifest["files"]}
    for entry in manifest["files"]:
        source_path = source_root / entry["path"]
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert source_path.stat().st_size == entry["bytes"]
        assert digest == entry["sha256"]
    after = {entry["path"]: (source_root / entry["path"]).stat().st_mtime_ns for entry in manifest["files"]}
    assert before == after
    source_repo = source_root.parents[0]
    head = subprocess.run(["git", "-C", str(source_repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    assert head == manifest["source_git_commit"]
    assert subprocess.run(["git", "-C", str(source_repo), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip() == ""
