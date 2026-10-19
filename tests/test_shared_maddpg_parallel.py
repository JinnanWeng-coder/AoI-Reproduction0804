"""Fake subprocesses only: no CUDA, training or policy evaluation."""
from pathlib import Path

import pytest

from analysis import run_shared_maddpg_local as q


def setup_queue(tmp_path, monkeypatch, *, failure=None, missing_record=None, spawn_failure=None):
    batch = tmp_path / "study/batch"
    for folder in ("records", "logs", "tmp", "analysis"):
        (batch / folder).mkdir(parents=True, exist_ok=True)
    state = {"tick": 0, "active": set(), "max_active": 0, "started": [], "terminated": [],
             "summary": 0, "export": 0, "reverified": [], "envs": {}, "events": []}
    manifest = {"study_root": str(batch.parent), "runs": [q.sc.cell(i) for i in range(6)],
                "implementation_commit": "synthetic", "concurrent_training_processes": 2}
    monkeypatch.setattr(q.sc, "load", lambda b: (manifest, []))
    monkeypatch.setattr(q.sc, "preflight", lambda b: {"technical_status": "PASS"})
    monkeypatch.setattr(q.sc, "train_run", lambda b, i, *a: state["reverified"].append(i))
    def summary(b, attempt):
        state["summary"] += 1
        q.write(batch / f"analysis/{attempt}/verification.json", {"technical_status": "PASS"})
    def export(b, attempt):
        state["export"] += 1
    monkeypatch.setattr(q.sc, "summary", summary)
    monkeypatch.setattr(q.sc, "export", export)
    monkeypatch.setattr(q.time, "sleep", lambda seconds: state.update(tick=state["tick"] + 1))
    class Process:
        def __init__(self, cmd, **kw):
            assert cmd[cmd.index("-m") + 1] == "analysis.shared_maddpg_contract"
            assert "--resume" not in cmd
            self.i = int(cmd[cmd.index("--id") + 1])
            if self.i == spawn_failure:
                raise OSError("synthetic launch failure")
            self.pid = self.i + 100
            self.start = state["tick"]
            self.code = None
            state["active"].add(self.i)
            state["max_active"] = max(state["max_active"], len(state["active"]))
            assert len(state["active"]) <= 2
            state["started"].append(self.i)
            state["events"].append(("start", self.i, state["tick"]))
            state["envs"][self.i] = kw["env"]
        def poll(self):
            if self.code is None and state["tick"] - self.start >= (5 if self.i == 1 else 2):
                self.code = 1 if self.i == failure else 0
                state["active"].remove(self.i)
                state["events"].append(("finish", self.i, state["tick"]))
                if self.code == 0 and self.i != missing_record:
                    q.write(batch / f"records/train_{self.i:02d}.json", {"synthetic": True})
            return self.code
        def wait(self):
            assert self.poll() is not None
            return self.code
        def terminate(self):
            state["terminated"].append(self.i)
            state["active"].remove(self.i)
            self.code = -15
    monkeypatch.setattr(q.subprocess, "Popen", Process)
    return batch, state, manifest


def test_two_active_dynamic_refill_and_separate_outputs(tmp_path, monkeypatch):
    batch, state, _ = setup_queue(tmp_path, monkeypatch)
    result = q.run_queue(batch)
    assert state["max_active"] == 2 and not state["active"]
    assert state["started"] == list(range(6)) and not state["terminated"]
    assert state["events"].index(("start", 2, 2)) < state["events"].index(("finish", 1, 5))
    assert state["summary"] == state["export"] == 1
    assert result["accepted_runs"] == 6 and result["concurrent_training_processes"] == 2
    assert len(list((batch / "logs").glob("*.out"))) == 6
    records = [q.read(p) for p in (batch / "records").glob("local_process_*.json")]
    assert len(records) == 6 and all(r["exit_code"] == 0 and r["MaxRSS"] is None for r in records)
    assert {r["training_seed"] for r in records} == set(range(8, 14))
    assert all(r["device"] == "cuda:0" and r["concurrent_training_processes"] == 2 for r in records)
    assert len({e["TMP"] for e in state["envs"].values()}) == 6
    assert len({e["TORCH_HOME"] for e in state["envs"].values()}) == 6
    assert all(e["OMP_NUM_THREADS"] == "1" for e in state["envs"].values())
    assert not (batch.parent / ".training100_local_gpu.lock").exists()


@pytest.mark.parametrize("mode", ["failure", "missing_record", "spawn_failure"])
def test_failure_stops_later_seeds_but_drains_healthy_peer(tmp_path, monkeypatch, mode):
    kwargs = {mode: 1 if mode == "spawn_failure" else 0}
    batch, state, _ = setup_queue(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(ValueError, match="active peers drained"):
        q.run_queue(batch)
    assert state["started"] == ([0] if mode == "spawn_failure" else [0, 1])
    assert not state["active"] and not state["terminated"]
    assert state["summary"] == state["export"] == 0
    assert (batch / f"records/train_{0 if mode == 'spawn_failure' else 1:02d}.json").exists()
    assert not (batch.parent / ".training100_local_gpu.lock").exists()
    assert len(list((batch / "logs").glob("*.out"))) == 2


def test_interrupt_terminates_owned_children_and_preserves_records(tmp_path, monkeypatch):
    batch, state, _ = setup_queue(tmp_path, monkeypatch)
    def interrupt(seconds):
        raise KeyboardInterrupt()
    monkeypatch.setattr(q.time, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        q.run_queue(batch)
    assert state["started"] == state["terminated"] == [0, 1] and not state["active"]
    assert all(q.read(p)["supervisor_interrupted"] for p in (batch / "records").glob("local_process_*.json"))
    assert len(list((batch / "records").glob("local_started_*.json"))) == 2
    assert not (batch.parent / ".training100_local_gpu.lock").exists()


def test_second_supervisor_cannot_bypass_existing_study_lock(tmp_path, monkeypatch):
    batch, state, _ = setup_queue(tmp_path, monkeypatch)
    lock = batch.parent / ".training100_local_gpu.lock"
    lock.write_text("existing owner", encoding="utf-8")
    with pytest.raises(FileExistsError):
        q.run_queue(batch)
    assert not state["started"] and lock.read_text() == "existing owner"


@pytest.mark.parametrize("existing", ["partial", "log", "started", "process"])
def test_existing_attempt_not_overwritten_or_restarted(tmp_path, monkeypatch, existing):
    batch, state, _ = setup_queue(tmp_path, monkeypatch)
    paths = {"partial": q.sc.attempt_run(batch, 1, "a01") / "partial.txt",
             "log": batch / "logs/train_01_a01.out",
             "started": batch / "records/local_started_01_a01.json",
             "process": batch / "records/local_process_01_a01.json"}
    path = paths[existing]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preserved")
    with pytest.raises(ValueError, match="preserv"):
        q.run_queue(batch)
    assert not state["started"] and path.read_bytes() == b"preserved"


def test_accepted_seed_is_reverified_not_spawned(tmp_path, monkeypatch):
    batch, state, _ = setup_queue(tmp_path, monkeypatch)
    q.write(batch / "records/train_00.json", {"synthetic": True})
    q.run_queue(batch)
    assert state["reverified"] == [0] and state["started"] == list(range(1, 6))


def test_subset_and_wrong_concurrency(tmp_path, monkeypatch):
    batch, state, manifest = setup_queue(tmp_path, monkeypatch)
    result = q.run_queue(batch, ids="2,3")
    assert state["started"] == [2, 3] and result["technical_status"] == "PARTIAL"
    assert state["summary"] == state["export"] == 0
    manifest["concurrent_training_processes"] = 3
    with pytest.raises(ValueError, match="two-process"):
        q.run_queue(batch)
