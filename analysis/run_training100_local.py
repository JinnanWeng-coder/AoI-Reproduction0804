"""Single-GPU, serial fresh training queue; no Slurm, old payloads, or evaluation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from analysis import training100_contract as tc
from analysis.deployment_contract import REPO, read, require, write


def cuda_probe():
    require(torch.cuda.is_available(), "CUDA unavailable; never silently train on CPU")
    device = torch.device("cuda:0")
    a = torch.ones((256, 256), dtype=torch.float32, device=device)
    b = a @ a
    torch.cuda.synchronize(device)
    value = float(b.sum().item())
    require(value == 16777216.0, "CUDA matrix probe failed")
    props = torch.cuda.get_device_properties(device)
    return {"device": str(device), "name": props.name, "total_memory_bytes": props.total_memory,
            "visible_device_count": torch.cuda.device_count(), "probe_sum": value}


def preflight(batch):
    batch = Path(batch).resolve()
    manifest, sources = tc.load(batch)
    require(manifest.get("source_mode") == "bundled_config_fresh", "init-local batch required")
    require(sys.version_info[:2] == (3, 10), "use dedicated Python 3.10 environment")
    require(torch.__version__.split("+")[0] == "2.11.0" and torch.version.cuda == "12.6", "use torch2.11.0 cu126, as historical runtime")
    versions = {}
    for line in (REPO / "requirements.cuda.lock.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, expected = line.strip().split("==")
        actual = importlib.metadata.version(name)
        require(actual == expected, f"dependency mismatch: {name}={actual}; expected {expected}")
        versions[name] = actual
    for source in sources:
        tc.check_source(source)
        cfg, _ = tc.derive(source, batch / "training/a01/runs")
        require(cfg.device == "cuda:0", "configuration must retain historical cuda:0 device")
    result = {"technical_status": "PASS", "scope": "local_config_environment_only_not_historical_payload_audit",
              "implementation_commit": manifest["implementation_commit"], "source_configs": 24, "historical_payloads_required": False,
              "python_executable": sys.executable, "python": sys.version, "platform": platform.platform(),
              "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "dependencies": versions,
              "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": cuda_probe(),
              "concurrent_training_processes": 1, "new_evaluations": 0}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    write(batch / f"protocol/local_preflight_{stamp}.json", result)
    return result


def selected_ids(text):
    ids = list(range(24)) if text is None else [int(x) for x in text.split(",")]
    require(ids and len(ids) == len(set(ids)) and all(0 <= i < 24 for i in ids), "IDs must be unique 0..23")
    return ids


def child_command(batch, i, attempt, reason):
    cmd = [sys.executable, "-B", "-u", "-m", "analysis.training100_contract", "train",
           "--batch-root", str(batch), "--id", str(i), "--attempt", attempt]
    if reason:
        cmd += ["--reason", reason]
    return cmd


def run_queue(batch, ids=None, attempt="a01", reason=None):
    batch = Path(batch).resolve()
    ids = selected_ids(ids)
    require(attempt == "a01" or reason, "retry attempt requires a reason; no partial resume")
    tc.attempt_run(batch, ids[0], attempt)  # Validate before starting any process.
    report = preflight(batch)
    manifest, _ = tc.load(batch)
    lock = Path(manifest["study_root"]) / ".training100_local_gpu.lock"
    # One launcher for this study root. A stale lock is reported, never auto-deleted.
    with lock.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "batch_root": str(batch), "python": sys.executable}, f)
    try:
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO)}
        for key in ("AOI_TASK_OFFSET", "AOI_RETRY_REASON", "AOI_ACCEPT_EXISTING", "AOI_CHECK_ONLY"):
            env.pop(key, None)
        env.update(TMP=str(batch / "tmp"), TEMP=str(batch / "tmp"), TMPDIR=str(batch / "tmp"),
                   XDG_CACHE_HOME=str(batch / "cache"), MPLCONFIGDIR=str(batch / "cache/matplotlib"),
                   TORCH_HOME=str(batch / "cache/torch"), CUDA_CACHE_PATH=str(batch / "cache/cuda"))
        for key in ("TMP", "TEMP", "TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        print(json.dumps({"queue_pid": os.getpid(), "ids": ids, "preflight": report}, ensure_ascii=False), flush=True)
        for i in ids:
            accepted = batch / f"records/train_{i:02d}.json"
            if accepted.exists():
                tc.train_run(batch, i, attempt, reason)  # Full identity/array/artifact recheck, not blind skip.
                print(f"run {i}: accepted and reverified; skipped", flush=True)
                continue
            log = batch / f"logs/train_{i:02d}_{attempt}.out"
            cmd = child_command(batch, i, attempt, reason)
            stamp = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            with log.open("x", encoding="utf-8") as handle:
                process = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
                metadata = {"run_id": i, "attempt": attempt, "reason": reason, "pid": process.pid, "command": cmd,
                            "started_at_utc": stamp, "implementation_commit": manifest["implementation_commit"],
                            "log": str(log), "python": sys.executable, "MaxRSS": None}
                try:
                    write(batch / f"records/local_started_{i:02d}_{attempt}.json", metadata)
                    print(json.dumps(metadata, ensure_ascii=False), flush=True)
                    exit_code = process.wait()
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        process.wait()
                    raise
            write(batch / f"records/local_process_{i:02d}_{attempt}.json", {**metadata, "exit_code": exit_code,
                  "elapsed_seconds": time.perf_counter() - started, "finished_at_utc": datetime.now(timezone.utc).isoformat()})
            require(exit_code == 0 and accepted.is_file(), f"run {i} failed; preserve {log}; no automatic retry")
        if all((batch / f"records/train_{i:02d}.json").is_file() for i in range(24)):
            summary_path = batch / f"analysis/{attempt}/verification.json"
            if summary_path.exists():
                require(read(summary_path)["technical_status"] == "PASS", "failed summary preserved: use a new analysis attempt")
            else:
                tc.summary(batch, attempt)
            archive = summary_path.parent / "core_evidence.zip"
            if not archive.exists():
                tc.export(batch, attempt)
            else:
                tc.verify_archive(archive, attempt)
            return {"technical_status": "PASS", "accepted_runs": 24, "archive": str(archive)}
        return {"technical_status": "PARTIAL", "requested_ids": ids, "new_evaluations": 0}
    finally:
        lock.unlink()  # Only the lock acquired by this launcher; never remove run artifacts.


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "run"))
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--ids", help="default all 0..23; comma-separated IDs for explicit selective recovery")
    p.add_argument("--attempt", default="a01")
    p.add_argument("--reason")
    a = p.parse_args(argv)
    result = preflight(a.batch_root) if a.command == "preflight" else run_queue(a.batch_root, a.ids, a.attempt, a.reason)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
