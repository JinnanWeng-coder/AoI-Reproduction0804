"""Safety tests for atomic run-dir publish without TOCTOU overwrite."""

from __future__ import annotations

import errno
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import runner as runner_module
from config import config_from_dict, matrix_specs
from scripts import matrix_runner
from tests.test_matrix_remote_recovery import _args, _formal_git_metadata


INIT_WHITELIST = {"checkpoints", "config.resolved.json", "provenance.json", "stdout.log"}


def _write_complete_staging(parent: Path, name: str = ".run.init-staging") -> Path:
    staging = parent / name
    staging.mkdir()
    (staging / "checkpoints").mkdir()
    (staging / "config.resolved.json").write_text('{"ok": true}\n', encoding="utf-8")
    (staging / "provenance.json").write_text('{"ok": true}\n', encoding="utf-8")
    (staging / "stdout.log").write_text("run started\n", encoding="utf-8")
    return staging


def _force_renameat2_errno(monkeypatch, code: int):
    if not sys.platform.startswith("linux"):
        pytest.skip("renameat2(RENAME_NOREPLACE) tests require Linux")

    def _boom(staging: Path, run_dir: Path) -> None:
        raise OSError(code, os.strerror(code), str(run_dir))

    monkeypatch.setattr(runner_module, "_renameat2_noreplace", _boom)


def test_einval_fallback_publishes_complete_whitelist_only(monkeypatch, tmp_path):
    _force_renameat2_errno(monkeypatch, errno.EINVAL)
    staging = _write_complete_staging(tmp_path)
    run_dir = tmp_path / "run"
    runner_module._publish_staged_run_no_replace(staging, run_dir)
    assert {entry.name for entry in run_dir.iterdir()} == INIT_WHITELIST
    assert not any((run_dir / "checkpoints").iterdir())
    assert not staging.exists()


@pytest.mark.parametrize("code", [errno.ENOSYS, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)])
def test_enosys_and_eopnotsupp_also_use_fallback(monkeypatch, tmp_path, code):
    _force_renameat2_errno(monkeypatch, code)
    staging = _write_complete_staging(tmp_path)
    run_dir = tmp_path / f"run_{code}"
    runner_module._publish_staged_run_no_replace(staging, run_dir)
    assert {entry.name for entry in run_dir.iterdir()} == INIT_WHITELIST
    assert not staging.exists()


def test_eexist_does_not_enter_fallback(monkeypatch, tmp_path):
    calls = {"fallback": 0}
    real_fallback = runner_module._publish_via_mkdir_reservation

    def tracked(staging, run_dir):
        calls["fallback"] += 1
        return real_fallback(staging, run_dir)

    monkeypatch.setattr(runner_module, "_publish_via_mkdir_reservation", tracked)
    _force_renameat2_errno(monkeypatch, errno.EEXIST)
    staging = _write_complete_staging(tmp_path)
    run_dir = tmp_path / "run"
    with pytest.raises(FileExistsError):
        runner_module._publish_staged_run_no_replace(staging, run_dir)
    assert calls["fallback"] == 0
    assert staging.exists()
    assert not run_dir.exists()


@pytest.mark.parametrize("code", [errno.EIO, errno.EPERM, errno.EXDEV])
def test_hard_errnos_are_raised_without_fallback(monkeypatch, tmp_path, code):
    calls = {"fallback": 0}

    def tracked(staging, run_dir):
        calls["fallback"] += 1
        raise AssertionError("fallback must not run")

    monkeypatch.setattr(runner_module, "_publish_via_mkdir_reservation", tracked)
    _force_renameat2_errno(monkeypatch, code)
    staging = _write_complete_staging(tmp_path)
    run_dir = tmp_path / f"run_{code}"
    with pytest.raises(OSError) as excinfo:
        runner_module._publish_staged_run_no_replace(staging, run_dir)
    assert excinfo.value.errno == code
    assert calls["fallback"] == 0
    assert staging.exists()
    assert not run_dir.exists()


@pytest.mark.parametrize("prefill", [None, "marker"])
def test_preexisting_empty_and_nonempty_dirs_are_never_overwritten(monkeypatch, tmp_path, prefill):
    _force_renameat2_errno(monkeypatch, errno.EINVAL)
    staging = _write_complete_staging(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if prefill is not None:
        (run_dir / "keep.txt").write_text(prefill, encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner_module._publish_staged_run_no_replace(staging, run_dir)
    assert staging.exists()
    assert run_dir.is_dir()
    if prefill is None:
        assert list(run_dir.iterdir()) == []
    else:
        assert (run_dir / "keep.txt").read_text(encoding="utf-8") == prefill
        assert "config.resolved.json" not in {p.name for p in run_dir.iterdir()}


def test_concurrent_publishers_only_one_wins(monkeypatch, tmp_path):
    _force_renameat2_errno(monkeypatch, errno.EINVAL)
    run_dir = tmp_path / "run"
    barrier = threading.Barrier(2)
    results = []

    def worker(idx: int):
        staging = _write_complete_staging(tmp_path, name=f".run.init-worker{idx}")
        (staging / "stdout.log").write_text(f"worker-{idx}\n", encoding="utf-8")
        barrier.wait()
        try:
            runner_module._publish_staged_run_no_replace(staging, run_dir)
            results.append(("ok", idx, staging))
        except FileExistsError:
            results.append(("exists", idx, staging))

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, i) for i in (0, 1)]
        for future in futures:
            future.result()

    outcomes = sorted(item[0] for item in results)
    assert outcomes == ["exists", "ok"]
    winner = next(item for item in results if item[0] == "ok")
    loser = next(item for item in results if item[0] == "exists")
    assert {entry.name for entry in run_dir.iterdir()} == INIT_WHITELIST
    assert (run_dir / "stdout.log").read_text(encoding="utf-8") == f"worker-{winner[1]}\n"
    assert not winner[2].exists()
    assert loser[2].exists()
    assert {entry.name for entry in loser[2].iterdir()} == INIT_WHITELIST


def test_interrupted_after_mkdir_reservation_hard_rejects_recovery(monkeypatch, tmp_path):
    args = _args(tmp_path.resolve())
    item = matrix_runner._resolved_item(matrix_specs()[0], args)
    config = config_from_dict(item["_expected_config"])
    run_dir = Path(item["output_root"]) / item["run_name"]
    git = _formal_git_metadata()
    monkeypatch.setattr(runner_module, "_git_metadata", lambda: dict(git))
    monkeypatch.setattr(runner_module, "_source_manifest_digest", lambda: "test-manifest")

    staging = Path(item["output_root"]) / f".{run_dir.name}.init-interrupted"
    staging.mkdir(parents=True)
    (staging / "checkpoints").mkdir()
    (staging / "config.resolved.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "provenance.json").write_text(
        json.dumps(runner_module._run_provenance(config, git), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "stdout.log").write_text("run started\n", encoding="utf-8")

    # Simulate crash after mkdir reservation, before staging replace.
    os.mkdir(run_dir)
    assert list(run_dir.iterdir()) == []
    assert "config.resolved.json" not in {p.name for p in run_dir.iterdir()}
    assert "provenance.json" not in {p.name for p in run_dir.iterdir()}

    with pytest.raises(runner_module.EmptyRunRecoveryError) as excinfo:
        runner_module._validate_empty_run_for_reinitialization(run_dir, config)
    assert excinfo.value.code == "INITIALIZATION_INCOMPLETE"

    recovery = matrix_runner._recovery_state(run_dir, item, recover_empty_run=True)
    assert recovery["action"] == "error"
    assert recovery["code"] == "EMPTY_RUN_INITIALIZATION_INCOMPLETE"
    assert staging.exists()
    assert run_dir.exists()
    assert list(run_dir.iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real NFS probe requires Linux")
def test_real_nfs_mkdir_reservation_replace_under_eeedata():
    storage_root = Path("/eeedata/sgxjw2")
    if not storage_root.is_dir() or not os.access(storage_root, os.W_OK):
        pytest.skip("/eeedata/sgxjw2 is not an available writable HPC storage root")

    base = Path("/eeedata/sgxjw2/tmp/pytest_nfs_publish_probe")
    base.mkdir(parents=True, exist_ok=True)
    case = base / f"case_{os.getpid()}"
    if case.exists():
        for child in sorted(case.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                child.rmdir()
        case.rmdir()
    case.mkdir()
    staging = _write_complete_staging(case)
    run_dir = case / "run"

    os.mkdir(run_dir)
    assert list(run_dir.iterdir()) == []
    os.rename(staging, run_dir)
    assert {entry.name for entry in run_dir.iterdir()} == INIT_WHITELIST
    assert not staging.exists()

    # Conflict: empty pre-existing reservation must not be overwritten by mkdir.
    staging2 = _write_complete_staging(case, name=".run.init-second")
    with pytest.raises(FileExistsError):
        os.mkdir(run_dir)
    assert staging2.exists()
    assert {entry.name for entry in run_dir.iterdir()} == INIT_WHITELIST
