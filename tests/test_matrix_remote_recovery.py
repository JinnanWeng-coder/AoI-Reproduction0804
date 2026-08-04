import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from config import build_parser, config_from_args, config_from_dict, matrix_specs
import runner as runner_module
from scripts import matrix_runner


def _args(output_root, device="cuda:0", purpose="validation"):
    return SimpleNamespace(
        device=device,
        output_root=str(output_root),
        eval_purpose=purpose,
        eval_episodes=100,
        eval_seeds=(
            "201,202,203,204,205,206"
            if purpose == "validation"
            else "101,102,103,104,105,106"
        ),
        profile="paper_faithful",
        checkpoint_every=5,
        recover_empty_run=False,
    )


def _payload(item, *, completed, episode, config=None):
    config = item["_expected_config"] if config is None else config
    resolved = config_from_dict(config)
    return {
        "checkpoint_version": 4,
        "checkpoint_schema_version": item["checkpoint_schema_version"],
        "semantic_version": item["semantic_version"],
        "mobility_revision": item["mobility_revision"],
        "config_hash": resolved.canonical_hash(),
        "config": config,
        "episode": episode,
        "completed": completed,
    }


def _write_checkpoint(run_dir, item, *, completed, episode, marker=False, config=None):
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_payload(item, completed=completed, episode=episode, config=config), checkpoint)
    if marker:
        best = run_dir / "checkpoints" / "best.pt"
        torch.save(_payload(item, completed=completed, episode=episode, config=config), best)
        checkpoint_hashes = {
            "latest.pt": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "best.pt": hashlib.sha256(best.read_bytes()).hexdigest(),
        }
        complete = {
            "status": "complete",
            "profile": item["profile"],
            "scenario": item["scenario"],
            "seed": item["seed"],
            "episodes": item["_expected_config"]["episodes"],
            "final_episode": item["_expected_config"]["episodes"],
            "checkpoint_completed": True,
            "checkpoint_sha256": checkpoint_hashes,
            "config_hash": item["config_hash"],
            "semantic_version": item["semantic_version"],
            "mobility_revision": item["mobility_revision"],
            "checkpoint_schema_version": item["checkpoint_schema_version"],
        }
        (run_dir / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    return checkpoint


def _formal_git_metadata():
    return {
        "reproduction_git_commit": "test-commit",
        "reproduction_git_branch": "test-branch",
        "reproduction_git_dirty": False,
        "reproduction_tracked_tree_sha256": "test-tree",
        "gpu_names": [],
        "cuda_driver": None,
    }


def _write_empty_initialized_run(monkeypatch, item):
    manifest_hash = "test-manifest"
    git = _formal_git_metadata()
    monkeypatch.setattr(runner_module, "_git_metadata", lambda: dict(git))
    monkeypatch.setattr(runner_module, "_source_manifest_digest", lambda: manifest_hash)
    config = config_from_dict(item["_expected_config"])
    run_dir = Path(item["output_root"]) / item["run_name"]
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.resolved.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = runner_module._run_provenance(config, git)
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "stdout.log").write_text("run started\n", encoding="utf-8")
    return run_dir, config


def test_expected_hash_uses_exact_remote_command_config(tmp_path):
    args = _args((tmp_path / "absolute remote runs").resolve())
    raw = matrix_specs()[0]
    item = matrix_runner._resolved_item(raw, args)
    command = matrix_runner._command(item, args, "train")
    parsed = build_parser().parse_args(command[2:])
    commanded_config = config_from_args(parsed)

    assert commanded_config.device == "cuda:0"
    assert commanded_config.output_root == str((tmp_path / "absolute remote runs").resolve())
    assert commanded_config.canonical_hash() == item["config_hash"]
    assert item["config_hash"] != raw["config_hash"]


def test_checkpoint_every_is_positive_and_shared_by_train_eval_and_hash(tmp_path):
    args = _args(tmp_path.resolve())
    args.checkpoint_every = 5
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    checkpoint = Path(item["output_root"]) / item["run_name"] / "checkpoints" / "latest.pt"
    train_command = matrix_runner._command(item, args, "train")
    eval_command = matrix_runner._command(item, args, "eval", checkpoint)

    for command in (train_command, eval_command):
        assert command[command.index("--checkpoint-every") + 1] == "5"
        parsed_config = config_from_args(build_parser().parse_args(command[2:]))
        assert parsed_config.canonical_hash() == item["config_hash"]
    with pytest.raises(SystemExit):
        matrix_runner.main(["--dry-run", "--checkpoint-every", "0"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--checkpoint-every", "-1"])


def test_recover_empty_flag_is_not_combined_with_checkpoint_resume(tmp_path):
    args = _args(tmp_path.resolve())
    args.recover_empty_run = True
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    run_dir = Path(item["output_root"]) / item["run_name"]
    checkpoint = _write_checkpoint(run_dir, item, completed=False, episode=17)
    recovery = matrix_runner._recovery_state(run_dir, item, recover_empty_run=True)
    command = matrix_runner._command(item, args, "train", checkpoint)

    assert recovery["action"] == "resume"
    assert "--resume" in command
    assert "--recover-empty-run" not in command


def test_matrix_report_records_recovery_and_checkpoint_policy(tmp_path, capsys):
    report = tmp_path / "matrix-report.json"
    assert matrix_runner.main(
        [
            "--dry-run",
            "--checkpoint-every",
            "5",
            "--recover-empty-run",
            "--output-root",
            str((tmp_path / "runs").resolve()),
            "--report",
            str(report),
        ]
    ) == 0
    capsys.readouterr()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["checkpoint_every"] == 5
    assert data["recover_empty_run"] is True
    assert len(data["commands"]) == 48
    assert all(entry["checkpoint_every"] == 5 for entry in data["commands"])
    assert all(entry["recover_empty_run"] is True for entry in data["commands"])
    assert all("--checkpoint-every" in entry["train"] for entry in data["commands"])
    assert all("--checkpoint-every" in entry["eval"] for entry in data["commands"])
    assert all("--recover-empty-run" in entry["train"] for entry in data["commands"])


def test_remote_incomplete_resumes_and_completed_skips_without_overwrite(tmp_path):
    args = _args((tmp_path / "remote runs").resolve())
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    run_dir = Path(item["output_root"]) / item["run_name"]
    checkpoint = _write_checkpoint(run_dir, item, completed=False, episode=17)
    before = checkpoint.read_bytes()

    resume = matrix_runner._recovery_state(run_dir, item)
    assert resume["code"] == "INCOMPLETE_RESUME_AVAILABLE"
    assert resume["action"] == "resume"
    assert checkpoint.read_bytes() == before

    _write_checkpoint(run_dir, item, completed=True, episode=500, marker=True)
    complete_before = checkpoint.read_bytes()
    completed = matrix_runner._recovery_state(run_dir, item)
    assert completed["code"] == "COMPLETE"
    assert completed["action"] == "skip"
    assert completed["checkpoint_sha256"] == hashlib.sha256(complete_before).hexdigest()
    assert checkpoint.read_bytes() == complete_before


def test_verified_empty_formal_run_requires_explicit_opt_in_and_is_not_overwritten(monkeypatch, tmp_path):
    args = _args(tmp_path.resolve())
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    run_dir, config = _write_empty_initialized_run(monkeypatch, item)
    config_before = (run_dir / "config.resolved.json").read_bytes()
    provenance_before = (run_dir / "provenance.json").read_bytes()

    disabled = matrix_runner._recovery_state(run_dir, item, recover_empty_run=False)
    enabled = matrix_runner._recovery_state(run_dir, item, recover_empty_run=True)
    prepared, is_resume = runner_module._prepare_run(config, None, recover_empty_run=True)

    assert disabled["code"] == "EMPTY_RUN_RECOVERY_NOT_ENABLED"
    assert enabled["code"] == "EMPTY_RUN_REINITIALIZE_AVAILABLE"
    assert enabled["action"] == "reinitialize"
    assert prepared == run_dir.resolve()
    assert is_resume is False
    assert (run_dir / "config.resolved.json").read_bytes() == config_before
    assert (run_dir / "provenance.json").read_bytes() == provenance_before
    assert not any((run_dir / "checkpoints").iterdir())


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_stdout", "EMPTY_RUN_INITIALIZATION_INCOMPLETE"),
        ("bad_stdout", "EMPTY_RUN_STDOUT_INVALID"),
        ("extra_tmp", "EMPTY_RUN_DIRECTORY_NOT_EMPTY"),
        ("checkpoint", "EMPTY_RUN_CHECKPOINTS_NOT_EMPTY"),
        ("identity", "EMPTY_RUN_IDENTITY_MISMATCH"),
        ("numeric_json_type", "EMPTY_RUN_CONFIG_MISMATCH"),
        ("provenance", "EMPTY_RUN_PROVENANCE_MISMATCH"),
    ],
)
def test_empty_run_recovery_rejects_noncanonical_or_nonempty_state(
    monkeypatch, tmp_path, mutation, expected_code
):
    args = _args(tmp_path.resolve())
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    run_dir, _config = _write_empty_initialized_run(monkeypatch, item)
    if mutation == "missing_stdout":
        (run_dir / "stdout.log").unlink()
    elif mutation == "bad_stdout":
        (run_dir / "stdout.log").write_text("training started twice\n", encoding="utf-8")
    elif mutation == "extra_tmp":
        (run_dir / "config.resolved.json.tmp").write_text("partial", encoding="utf-8")
    elif mutation == "checkpoint":
        (run_dir / "checkpoints" / "latest.pt.tmp").write_bytes(b"partial")
    elif mutation in {"identity", "numeric_json_type"}:
        config_path = run_dir / "config.resolved.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if mutation == "identity":
            raw["seed"] = 7 if item["seed"] != 7 else 6
        else:
            raw["bandwidth_hz"] = float(raw["bandwidth_hz"])
        config_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        provenance_path = run_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["reproduction_git_commit"] = "wrong-commit"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    recovery = matrix_runner._recovery_state(run_dir, item, recover_empty_run=True)
    assert recovery["action"] == "error"
    assert recovery["code"] == expected_code


def test_atomic_initialization_never_publishes_partial_final_run(monkeypatch, tmp_path):
    args = _args(tmp_path.resolve())
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    config = config_from_dict(item["_expected_config"])
    run_dir = Path(item["output_root"]) / item["run_name"]
    git = _formal_git_metadata()
    monkeypatch.setattr(runner_module, "_git_metadata", lambda: dict(git))
    monkeypatch.setattr(runner_module, "_source_manifest_digest", lambda: "test-manifest")

    def interrupted_publish(_staging, _run_dir):
        raise RuntimeError("simulated preemption before atomic rename")

    monkeypatch.setattr(runner_module, "_publish_staged_run_no_replace", interrupted_publish)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        runner_module._prepare_run(config, None)
    assert not run_dir.exists()
    staging = list(run_dir.parent.glob(f".{run_dir.name}.init-*"))
    assert len(staging) == 1
    assert matrix_runner._recovery_state(run_dir, item, recover_empty_run=True)["code"] == "NEW_RUN"


def test_atomic_initialization_publishes_only_complete_whitelist(monkeypatch, tmp_path):
    args = _args(tmp_path.resolve())
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    config = config_from_dict(item["_expected_config"])
    run_dir = Path(item["output_root"]) / item["run_name"]
    git = _formal_git_metadata()
    monkeypatch.setattr(runner_module, "_git_metadata", lambda: dict(git))
    monkeypatch.setattr(runner_module, "_source_manifest_digest", lambda: "test-manifest")

    published, is_resume = runner_module._prepare_run(config, None)
    assert published == run_dir.resolve()
    assert is_resume is False
    assert {entry.name for entry in run_dir.iterdir()} == {
        "checkpoints",
        "config.resolved.json",
        "provenance.json",
        "stdout.log",
    }
    assert not any((run_dir / "checkpoints").iterdir())
    assert not list(run_dir.parent.glob(f".{run_dir.name}.init-*"))
    assert matrix_runner._recovery_state(run_dir, item, recover_empty_run=True)["action"] == "reinitialize"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("semantic", "CHECKPOINT_SEMANTIC_MISMATCH"),
        ("mobility", "CHECKPOINT_MOBILITY_MISMATCH"),
        ("schema", "CHECKPOINT_SCHEMA_MISMATCH"),
        ("identity", "CHECKPOINT_IDENTITY_MISMATCH"),
        ("config", "CHECKPOINT_CONFIG_MISMATCH"),
    ],
)
def test_checkpoint_mismatches_are_structured(tmp_path, mutation, expected_code):
    args = _args(tmp_path.resolve())
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    run_dir = Path(item["output_root"]) / item["run_name"]
    payload = _payload(item, completed=False, episode=17)
    if mutation == "semantic":
        payload["semantic_version"] = "wrong-semantic"
    elif mutation == "mobility":
        payload["mobility_revision"] = "wrong-mobility"
    elif mutation == "schema":
        payload["checkpoint_schema_version"] = "wrong-schema"
    else:
        raw_config = json.loads(json.dumps(item["_expected_config"]))
        if mutation == "identity":
            raw_config["seed"] = 999
        else:
            raw_config["gamma"] = 0.5
        payload["config"] = raw_config
        payload["config_hash"] = matrix_runner._canonical_dict_hash(raw_config)
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(payload, checkpoint)

    recovery = matrix_runner._recovery_state(run_dir, item)
    assert recovery["action"] == "error"
    assert recovery["code"] == expected_code
    if mutation == "identity":
        assert recovery["mismatches"]["seed"] == {"checkpoint": 999, "expected": item["seed"]}


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("latest_version", "COMPLETE_CHECKPOINT_VERSION_MISMATCH"),
        ("best_missing", "COMPLETE_CHECKPOINT_BEST_MISSING"),
        ("best_version", "COMPLETE_BEST_CHECKPOINT_VERSION_MISMATCH"),
        ("best_not_completed", "COMPLETE_BEST_CHECKPOINT_NOT_COMPLETED"),
        ("final_episode", "COMPLETE_MARKER_MISMATCH"),
        ("checkpoint_completed", "COMPLETE_MARKER_MISMATCH"),
        ("checkpoint_hash", "COMPLETE_MARKER_MISMATCH"),
    ],
)
def test_complete_skip_rejects_unbound_or_invalid_final_checkpoints(tmp_path, mutation, expected_code):
    args = _args(tmp_path.resolve())
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    run_dir = Path(item["output_root"]) / item["run_name"]
    latest = _write_checkpoint(run_dir, item, completed=True, episode=500, marker=True)
    best = run_dir / "checkpoints" / "best.pt"
    marker_path = run_dir / "COMPLETE.json"
    if mutation == "best_missing":
        best.unlink()
    elif mutation in {"latest_version", "best_version", "best_not_completed"}:
        target = latest if mutation == "latest_version" else best
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if mutation.endswith("version"):
            payload["checkpoint_version"] = 3
        else:
            payload["completed"] = False
        torch.save(payload, target)
    else:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if mutation == "final_episode":
            marker["final_episode"] = 499
        elif mutation == "checkpoint_completed":
            marker["checkpoint_completed"] = False
        else:
            marker["checkpoint_sha256"]["latest.pt"] = "0" * 64
        marker_path.write_text(json.dumps(marker), encoding="utf-8")

    recovery = matrix_runner._recovery_state(run_dir, item)
    assert recovery["action"] == "error"
    assert recovery["code"] == expected_code


@pytest.mark.parametrize("stage", ["eval", "audit"])
def test_eval_and_audit_stages_gate_on_completed_identity(monkeypatch, tmp_path, capsys, stage):
    error = {
        "code": "COMPLETE_CHECKPOINT_IDENTITY_MISMATCH",
        "action": "error",
        "run": str(tmp_path / "wrong"),
        "mismatches": {"scenario": {"checkpoint": "wrong", "expected": "expected"}},
    }
    monkeypatch.setattr(matrix_runner, "_recovery_state", lambda *_, **__: error)

    def forbidden_run(*_):
        raise AssertionError("eval/audit command ran before completed identity validation")

    monkeypatch.setattr(matrix_runner, "_run", forbidden_run)
    result = matrix_runner.main(
        [
            "--execute",
            "--stage",
            stage,
            "--device",
            "cuda:0",
            "--output-root",
            str(tmp_path.resolve()),
        ]
    )
    output = capsys.readouterr().out
    assert result == 1
    assert f'"code": "{stage.upper()}_REQUIRES_COMPLETED_RUN"' in output
    assert "COMPLETE_CHECKPOINT_IDENTITY_MISMATCH" in output


def test_stage_all_orders_train_validation_eval_and_matching_audit(monkeypatch, tmp_path, capsys):
    output_root = (tmp_path / "remote matrix").resolve()
    args = _args(output_root)
    expected_items = matrix_runner._matrix_specs_for_args(args)
    by_name = {item["run_name"]: item for item in expected_items}
    events = []

    def fake_run(command, _log_path):
        if Path(command[1]).name == "Main.py":
            run_name = command[command.index("--run-name") + 1]
            item = by_name[run_name]
            run_dir = Path(item["output_root"]) / run_name
            if "--eval-only" not in command:
                events.append((run_name, "train"))
                _write_checkpoint(run_dir, item, completed=True, episode=500, marker=True)
            else:
                events.append((run_name, "eval"))
                assert command[command.index("--eval-purpose") + 1] == "validation"
                assert command[command.index("--scope") + 1] == "validation"
                checkpoint = run_dir / "checkpoints" / "latest.pt"
                summary = {
                    "status": "complete",
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "eval_episodes": 100,
                    "eval_seeds": [201, 202, 203, 204, 205, 206],
                    "eval_purpose": "validation",
                    "scope": "validation",
                    "profile": item["profile"],
                    "scenario": item["scenario"],
                    "training_seed": item["seed"],
                    "config_hash": item["config_hash"],
                    "semantic_version": item["semantic_version"],
                    "mobility_revision": item["mobility_revision"],
                    "checkpoint_schema_version": item["checkpoint_schema_version"],
                }
                eval_dir = run_dir / "eval" / "synthetic_validation"
                eval_dir.mkdir(parents=True)
                (eval_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                (eval_dir / "EVAL_COMPLETE.json").write_text(json.dumps(summary), encoding="utf-8")
        else:
            run_dir = Path(command[2])
            events.append((run_dir.name, "audit"))
            assert command[command.index("--scope") + 1] == "validation"
            assert "--require-eval" in command
        return 0

    monkeypatch.setattr(matrix_runner, "_run", fake_run)
    result = matrix_runner.main(
        [
            "--execute",
            "--stage",
            "all",
            "--device",
            "cuda:0",
            "--output-root",
            str(output_root),
        ]
    )
    capsys.readouterr()
    assert result == 0
    assert len(events) == 48 * 3
    for item in expected_items:
        per_run = [stage for run_name, stage in events if run_name == item["run_name"]]
        assert per_run == ["train", "eval", "audit"]


def test_remote_matrix_has_48_unique_default_validation_commands(tmp_path):
    args = _args((tmp_path / "remote matrix").resolve())
    items = matrix_runner._matrix_specs_for_args(args)
    assert len(items) == 48
    assert len({(item["profile"], item["scenario"], item["seed"]) for item in items}) == 48
    commands = {
        tuple(
            matrix_runner._command(
                item,
                args,
                "eval",
                Path(item["output_root"]) / item["run_name"] / "checkpoints" / "latest.pt",
            )
        )
        for item in items
    }
    assert len(commands) == 48
    assert all("validation" in command for command in commands)


def test_remote_wrappers_forward_checkpoint_interval_and_empty_recovery_option():
    root = Path(__file__).resolve().parents[1]
    powershell = (root / "scripts" / "run_paper_matrix.ps1").read_text(encoding="utf-8")
    shell = (root / "scripts" / "run_paper_matrix.sh").read_text(encoding="utf-8")
    assert "[int]$CheckpointEvery = 5" in powershell
    assert '"--checkpoint-every", $CheckpointEvery' in powershell
    assert "[switch]$RecoverEmptyRun" in powershell
    assert '"--recover-empty-run"' in powershell
    assert 'CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-5}"' in shell
    assert '--checkpoint-every "$CHECKPOINT_EVERY"' in shell
    assert "[--recover-empty-run]" in shell
