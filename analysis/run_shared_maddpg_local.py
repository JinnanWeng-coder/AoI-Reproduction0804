"""Two independent seed processes on cuda:0; no training semantics changed."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from analysis import shared_maddpg_contract as sc
from analysis.deployment_contract import REPO, read, require, write
from analysis.run_training100_local import child_command, selected_ids

CONCURRENCY = 2


def child_environment(batch, i, attempt):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for key in ("AOI_TASK_OFFSET", "AOI_RETRY_REASON", "AOI_ACCEPT_EXISTING", "AOI_CHECK_ONLY"):
        env.pop(key, None)
    # Independent temporary/cache paths as well as independent run/log paths.
    root = Path(batch) / "tmp" / f"train_{i:02d}_{attempt}"
    env.update(TMP=str(root / "tmp"), TEMP=str(root / "tmp"), TMPDIR=str(root / "tmp"),
               XDG_CACHE_HOME=str(root / "cache"), MPLCONFIGDIR=str(root / "cache/matplotlib"),
               TORCH_HOME=str(root / "cache/torch"), CUDA_CACHE_PATH=str(root / "cache/cuda"))
    for key in ("TMP", "TEMP", "TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_HOME", "CUDA_CACHE_PATH"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def run_queue(batch, ids=None, attempt="a01", reason=None, *, workflow=None, workers=CONCURRENCY):
    c = workflow or sc
    require(type(workers) is int and 1 <= workers <= CONCURRENCY, "one or two training workers required")
    require(workflow is not None or workers == CONCURRENCY, "historical shared queue retains two workers")
    batch = Path(batch).resolve()
    manifest, _ = c.load(batch)
    require(manifest["concurrent_training_processes"] == CONCURRENCY, "two-process batch required")
    count = len(manifest["runs"])
    ids = selected_ids(ids, count)
    require(attempt == "a01" or reason, "retry attempt requires a reason")
    for i in ids:
        c.attempt_run(batch, i, attempt)
    report = c.preflight(batch)
    lock = Path(manifest["study_root"]) / ".training100_local_gpu.lock"
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "batch_root": str(batch), "python": sys.executable,
                   "concurrent_training_processes": workers}, handle)
    active = {}
    failures = []
    try:
        pending = deque()
        for i in ids:
            if (batch / f"records/train_{i:02d}.json").exists():
                c.train_run(batch, i, attempt, reason)  # Verify, never blindly skip.
            else:
                # Refuse previously used launcher paths before launching any seed.
                for path in (batch / f"logs/train_{i:02d}_{attempt}.out",
                             batch / f"records/local_started_{i:02d}_{attempt}.json",
                             batch / f"records/local_process_{i:02d}_{attempt}.json"):
                    require(not path.exists(), f"preserve prior attempt; choose explicit new attempt: {path}")
                run = c.attempt_run(batch, i, attempt)
                require(not run.exists() or (run / "COMPLETE.json").is_file(),
                        f"partial run preserved; explicit checkpoint recovery required: {run}")
                pending.append(i)
        print(json.dumps({"queue_pid": os.getpid(), "ids": ids, "preflight": report,
                          "concurrent_training_processes": workers}, ensure_ascii=False), flush=True)
        while pending or active:
            # Inspect every live child before replenishing capacity: an observed
            # failure prevents all later seeds, but does not kill a healthy peer.
            for i, job in list(active.items()):
                code = job["process"].poll()
                if code is None:
                    continue
                job["process"].wait()
                job["handle"].close()
                accepted = code == 0 and (batch / f"records/train_{i:02d}.json").is_file()
                write(batch / f"records/local_process_{i:02d}_{attempt}.json",
                      {**job["metadata"], "exit_code": code, "accepted_record_present": accepted,
                       "elapsed_seconds": time.perf_counter() - job["started"],
                       "finished_at_utc": datetime.now(timezone.utc).isoformat()})
                del active[i]
                if not accepted:
                    failures.append({"run_id": i, "exit_code": code, "log": job["metadata"]["log"]})
                    print(json.dumps({"stop_new_launches": True, "failure": failures[-1],
                                      "waiting_for_active_ids": list(active)}, ensure_ascii=False), flush=True)
            while pending and not failures and len(active) < workers:
                i = pending.popleft()
                log = batch / f"logs/train_{i:02d}_{attempt}.out"
                cmd = child_command(batch, i, attempt, reason, c.CLI_MODULE)
                metadata = {"run_id": i, "training_seed": c.cell(i)["training_seed"],
                            "attempt": attempt, "reason": reason, "command": cmd,
                            "started_at_utc": datetime.now(timezone.utc).isoformat(),
                            "implementation_commit": manifest["implementation_commit"],
                            "log": str(log), "python": sys.executable, "MaxRSS": None,
                            "concurrent_training_processes": workers, "device": "cuda:0"}
                started = time.perf_counter()
                handle = log.open("x", encoding="utf-8")
                try:
                    process = subprocess.Popen(cmd, cwd=REPO, env=child_environment(batch, i, attempt),
                                               stdout=handle, stderr=subprocess.STDOUT)
                except Exception as exc:
                    handle.close()
                    failure = {**metadata, "pid": None, "exit_code": None, "launch_error": str(exc)}
                    write(batch / f"records/local_process_{i:02d}_{attempt}.json", failure)
                    failures.append(failure)
                    break
                metadata["pid"] = process.pid
                active[i] = {"process": process, "handle": handle, "metadata": metadata, "started": started}
                write(batch / f"records/local_started_{i:02d}_{attempt}.json", metadata)
                print(json.dumps(metadata, ensure_ascii=False), flush=True)
            if failures and not active:
                break
            if active:
                time.sleep(1)
        require(not failures, f"training queue failed; active peers drained; preserve attempts; failures={failures}")
        if all((batch / f"records/train_{i:02d}.json").is_file() for i in range(count)):
            summary = batch / f"analysis/{attempt}/verification.json"
            if summary.exists():
                require(read(summary)["technical_status"] == "PASS", "failed summary preserved; use new analysis attempt")
            else:
                c.summary(batch, attempt)
            archive = summary.parent / "core_evidence.zip"
            if archive.exists():
                c.verify_archive(archive, attempt)
            else:
                c.export(batch, attempt)
            if workflow is not None:
                # New workflows may deliver a second business package even if core ZIP already exists.
                return {**c.export(batch, attempt), "concurrent_training_processes": workers}
            return {"technical_status": "PASS", "accepted_runs": count, "archive": str(archive),
                    "concurrent_training_processes": workers}
        return {"technical_status": "PARTIAL", "requested_ids": ids, "new_evaluations": 0}
    finally:
        # Interrupt/unexpected supervisor error: stop only children created here,
        # wait for them before releasing the lock; never erase partial artifacts.
        for job in active.values():
            process = job["process"]
            if process.poll() is None:
                process.terminate()
        interrupted = []
        for job in active.values():
            process = job["process"]
            code = process.wait()
            job["handle"].close()
            path = batch / f"records/local_process_{job['metadata']['run_id']:02d}_{attempt}.json"
            interrupted.append((path, {**job["metadata"], "exit_code": code, "supervisor_interrupted": True,
                                      "elapsed_seconds": time.perf_counter() - job["started"],
                                      "finished_at_utc": datetime.now(timezone.utc).isoformat()}))
        # Stop/wait for all children even if persisting a final record later fails.
        for path, record in interrupted:
            if not path.exists():
                write(path, record)
        lock.unlink()
