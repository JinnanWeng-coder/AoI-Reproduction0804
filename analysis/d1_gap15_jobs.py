"""Five afterok jobs: payload check -> formal16 -> check16 -> remaining128 -> ZIP.

Dry-run by default. Never submits training or original D1 stages.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from analysis import deployment_contract as dc
from analysis import d1_gap15 as g

STAGES = ("preflight", "first16", "first-check", "remaining128", "finalize")


def array_spec(ids):
    dc.require(ids and len(ids) == len(set(ids)) and all(0 <= i < 144 for i in ids), "invalid gap15 array IDs")
    ranges, start, last = [], min(ids), min(ids)
    for value in sorted(ids)[1:]:
        if value != last + 1:
            ranges.append(str(start) if start == last else f"{start}-{last}")
            start = value
        last = value
    ranges.append(str(start) if start == last else f"{start}-{last}")
    return ",".join(ranges)


def command(project, batch, stage, dependency=None, partition="Q10", account=None,
            concurrency=6, cpus=4, mem="16G", gpu_time="02:00:00", cpu_time="03:00:00", gres="gpu:l20:1"):
    dc.require(stage in STAGES and 1 <= concurrency <= 6 and cpus > 0, "stage/resources contract")
    gpu = stage in {"first16", "remaining128"}
    args = ["sbatch", "--parsable", "--partition", partition, "--job-name", f"d1_g15_{stage}",
            "--cpus-per-task", str(cpus), "--mem", mem, "--time", gpu_time if gpu else cpu_time,
            "--output", str(batch / "slurm_logs/%x_%A_%a.out"), "--error", str(batch / "slurm_logs/%x_%A_%a.err"),
            f"--export=ALL,AOI_GAP15_STAGE={stage}"]
    if gpu:
        ids = g.FIRST_IDS if stage == "first16" else g.REST_IDS
        args += [f"--array={array_spec(ids)}%{concurrency}"]
        if gres:
            args += ["--gres", gres]
    if account:
        args += ["--account", account]
    if dependency:
        args += ["--dependency", dependency]
    return args + [str(project / "hpc" / ("aoi_d1_gap15_eval.sbatch" if gpu else "aoi_d1_gap15_cpu.sbatch"))]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--partition", default="Q10")
    p.add_argument("--account")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem", default="16G")
    p.add_argument("--gpu-time", default="02:00:00")
    p.add_argument("--cpu-time", default="03:00:00")
    p.add_argument("--gres", default="gpu:l20:1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--submit", action="store_true")
    a = p.parse_args(argv)
    batch = a.batch_root.resolve()
    m, _ = g.load_batch(batch)
    g.plan(batch)
    dc.require(not list((batch / "protocol").glob("submission_*.json")),
               "workflow already submitted/partly submitted; inspect jobs and recover only failed/unrun stages")
    dc.require(not list((batch / "evaluations").rglob("STARTED.json")), "existing executions; do not resubmit full workflow")
    environment = {**os.environ, "PROJECT_DIR": str(dc.REPO), "PYTHON_BIN": sys.executable,
                   "AOI_BATCH_ROOT": str(batch), "AOI_EXPECTED_COMMIT": m["implementation_commit"],
                   "AOI_DEVICE": a.device, "AOI_ATTEMPT": "a01", "AOI_CONTRACT_MODULE": "analysis.d1_gap15"}
    for key in ("AOI_TASK_OFFSET", "AOI_RETRY_REASON", "AOI_ACCEPT_EXISTING", "AOI_CHECK_ONLY"):
        environment.pop(key, None)
    jobs, commands, previous = [], [], None
    for stage in STAGES:
        cmd = command(dc.REPO, batch, stage, previous, a.partition, a.account, a.concurrency, a.cpus, a.mem,
                      a.gpu_time, a.cpu_time, a.gres)
        commands.append(shlex.join(cmd))
        if a.submit:
            job = subprocess.check_output(cmd, text=True, env=environment).strip().split(";")[0]
            dc.require(job.isdigit(), f"unexpected sbatch response: {job}")
            record = {"job_id": job, "stage": stage, "command": cmd,
                      "unit_ids": g.FIRST_IDS if stage == "first16" else g.REST_IDS if stage == "remaining128" else [],
                      "implementation_commit": m["implementation_commit"], "submitted_utc": datetime.now(timezone.utc).isoformat(),
                      "environment": {k: environment[k] for k in ("PROJECT_DIR", "PYTHON_BIN", "AOI_BATCH_ROOT",
                                       "AOI_EXPECTED_COMMIT", "AOI_DEVICE", "AOI_ATTEMPT", "AOI_CONTRACT_MODULE")}}
            token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            dc.write(batch / f"protocol/submission_{token}_{stage}.json", record)
            jobs.append(record)
            previous = f"afterok:{job}"
        else:
            previous = f"afterok:<{stage}_job>"
    print(json.dumps({"submitted": a.submit, "jobs": jobs, "commands": commands, "new_units": 144,
                      "models": 18, "first16_are_part_of_144": True, "batch_root": str(batch)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
