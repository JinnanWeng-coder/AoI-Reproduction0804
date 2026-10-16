"""Portable config, queue and export tests: no real environment, GPU or training."""
import copy
import csv
import json
import sys
import zipfile
from pathlib import Path

import pytest

from analysis import training100_contract as tc
from analysis import run_training100_local as local
from analysis import submit_training100 as slurm
from tests.test_frozen_deployment import dump


def local_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "code_check", lambda expected=None: "synthetic")
    result = tc.init_local(tmp_path / "study", "fresh_test")
    return Path(result["batch_root"])


def test_all_bundled_configs_preserve_original_identity_and_fresh_budget(tmp_path):
    sources = tc.bundled_sources()
    assert len(sources) == 24
    for i in range(24):
        item = tc.cell(i)
        s = next(s for s in sources if (s["model_id"], s["training_seed"]) == (item["model_id"], item["training_seed"]))
        before = copy.deepcopy(s)
        cfg, delta = tc.derive(s, tmp_path / "runs")
        assert s == before and cfg.episodes == 2500 and cfg.device == "cuda:0"
        assert set(delta) <= {"episodes", "run_name", "output_root", "mappo_actor_sharing"}
        if s["model_id"] == "algorithm2":
            assert cfg.profile == "paper_faithful" and cfg.exploration_noise == .3
        tc.check_source(s)  # Never attempts to open its historical Linux run path.


def test_init_local_needs_no_historical_payload_or_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(tc.dc, "source_check", lambda *a, **kw: pytest.fail("old artifacts must not be required"))
    batch = local_batch(tmp_path, monkeypatch)
    m, ss = tc.load(batch)
    assert m["source_mode"] == "bundled_config_fresh" and len(ss) == 24
    assert "NOT_RUN" in m["historical_payload_audit"]
    assert tc.dc.read(batch / "protocol/execution_decision.json")["pilot_required"] is False
    for s in ss: tc.check_source(s)
    with pytest.raises(ValueError, match="not historical payload audit"): tc.audit(batch)
    with pytest.raises(ValueError, match="unique new batch"): tc.init_local(tmp_path / "study", "fresh_test")
    mutated = copy.deepcopy(ss)
    mutated[0]["config"]["gamma"] = .8
    dump(batch / "protocol/sources.json", mutated)
    with pytest.raises(ValueError, match="configuration snapshot changed"): tc.load(batch)


def test_bundle_rejects_changed_config_hash(tmp_path, monkeypatch):
    data = tc.dc.read(tc.BUNDLE)
    data["templates"]["shared_tdec"]["gamma"] = .8
    path = tmp_path / "bad.json"
    dump(path, data)
    monkeypatch.setattr(tc, "BUNDLE", path)
    with pytest.raises(ValueError, match="config identity mismatch"): tc.bundled_sources()


def test_local_batch_cannot_use_slurm_submitter(tmp_path, monkeypatch):
    batch = local_batch(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not sbatch"):
        slurm.main(["--batch-root", str(batch), "--stage", "all", "--submit"])


def test_preflight_checks_environment_and_config_only(tmp_path, monkeypatch):
    batch = local_batch(tmp_path, monkeypatch)
    versions = dict(line.split("==") for line in (tc.dc.REPO / "requirements.cuda.lock.txt").read_text().splitlines()
                    if line and not line.startswith("#"))
    monkeypatch.setattr(local.importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(local.torch, "__version__", "2.11.0+cu126")
    monkeypatch.setattr(local.torch.version, "cuda", "12.6")
    monkeypatch.setattr(local, "cuda_probe", lambda: {"name": "synthetic GPU", "visible_device_count": 1})
    report = local.preflight(batch)
    assert report["technical_status"] == "PASS" and report["historical_payloads_required"] is False
    assert report["concurrent_training_processes"] == 1
    monkeypatch.setattr(local.torch.version, "cuda", None)
    with pytest.raises(ValueError, match="cu126"): local.preflight(batch)


def fake_queue(tmp_path, monkeypatch, failure=None):
    batch = local_batch(tmp_path, monkeypatch)
    monkeypatch.setattr(local, "preflight", lambda b: {"technical_status": "PASS", "synthetic": True})
    state = {"active": 0, "started": [], "summary": 0, "export": 0}
    class Process:
        def __init__(self, cmd, **kw):
            assert state["active"] == 0
            assert kw["env"]["PYTHONDONTWRITEBYTECODE"] == "1" and cmd[0] == sys.executable
            assert kw["env"]["TMP"].startswith(str(batch))
            self.i = int(cmd[cmd.index("--id") + 1])
            assert cmd[-1] == "a01"
            self.pid = 1000 + self.i
            state["active"] += 1
            state["started"].append(self.i)
        def wait(self):
            state["active"] -= 1
            if self.i == failure: return 1
            dump(batch / f"records/train_{self.i:02d}.json", {"synthetic": True})
            return 0
        def poll(self): return 0
    monkeypatch.setattr(local.subprocess, "Popen", Process)
    def summary(b, attempt):
        state["summary"] += 1
        dump(batch / f"analysis/{attempt}/verification.json", {"technical_status": "PASS"})
    def export(b, attempt): state["export"] += 1
    monkeypatch.setattr(tc, "summary", summary)
    monkeypatch.setattr(tc, "export", export)
    return batch, state


def test_all_24_queue_is_serial_and_auto_summarizes(tmp_path, monkeypatch):
    batch, state = fake_queue(tmp_path, monkeypatch)
    result = local.run_queue(batch)
    assert state["started"] == list(range(24)) and state["active"] == 0
    assert state["summary"] == state["export"] == 1 and result["accepted_runs"] == 24
    assert len(list((batch / "records").glob("local_process_*.json"))) == 24
    assert not (tmp_path / "study/.training100_local_gpu.lock").exists()


def test_failure_stops_queue_and_preserves_scene(tmp_path, monkeypatch):
    batch, state = fake_queue(tmp_path, monkeypatch, failure=0)
    with pytest.raises(ValueError, match="run 0 failed"): local.run_queue(batch)
    assert state["started"] == [0] and state["summary"] == state["export"] == 0
    assert (batch / "logs/train_00_a01.out").exists()
    assert tc.dc.read(batch / "records/local_process_00_a01.json")["exit_code"] == 1
    assert not (tmp_path / "study/.training100_local_gpu.lock").exists()


def test_existing_queue_lock_prevents_second_training_process(tmp_path, monkeypatch):
    batch, state = fake_queue(tmp_path, monkeypatch)
    lock = tmp_path / "study/.training100_local_gpu.lock"
    lock.write_text("existing launcher; not automatically removed")
    with pytest.raises(FileExistsError): local.run_queue(batch)
    assert state["started"] == [] and lock.exists()


@pytest.mark.parametrize("ids", ["0,0", "24", "-1", "", "bad"])
def test_reject_invalid_queue_ids(ids):
    with pytest.raises(ValueError): local.selected_ids(ids)


def test_local_export_without_slurm_and_with_bounded_log_tail(tmp_path, monkeypatch):
    batch = local_batch(tmp_path, monkeypatch)
    out = batch / "analysis/a01"
    dump(out / "verification.json", {"technical_status": "PASS", "accepted_runs": 24})
    for filename, count in (("selected_policies.csv", 24), ("training_episode_curves.csv", 60000)):
        with (out / filename).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["synthetic_index"])
            w.writerows([i] for i in range(count))
    dump(batch / "records/local_process_00_a01.json", {"exit_code": 0, "pid": 1, "MaxRSS": None})
    (batch / "logs/one.out").write_bytes(b"x" * 20000)
    monkeypatch.setattr(tc.subprocess, "check_output", lambda *a, **kw: pytest.fail("no sacct or sbatch on lab host"))
    result = tc.export(batch)
    assert tc.dc.read(out / "runtime_report.json")["backend"] == "local_python_serial"
    with zipfile.ZipFile(result["archive"]) as z:
        assert len(z.read("log_excerpts/logs/one.out")) == 16384
        assert not any(n.endswith((".pt", ".npz")) for n in z.namelist())
    tc.verify_archive(result["archive"])
