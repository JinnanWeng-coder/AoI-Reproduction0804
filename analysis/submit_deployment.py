"""Print or submit frozen deployment arrays; submission is explicitly opt-in."""
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


def chunks(count, size=1000):
    dc.require(1 <= size <= 1000, "array chunk must respect known MaxArraySize=1001")
    return [(offset, min(size, count - offset)) for offset in range(0, count, size)]


def command(project, batch, phase, offset, count, partition="Q10", account=None,
            concurrency=6, cpus=4, mem="16G", time="02:00:00", gres="gpu:l20:1", dependency=None):
    dc.require(concurrency >= 1 and cpus >= 1, "invalid resources")
    launcher = {"pilot": "pilot", "main": "main", "gap-transfer": "gap_transfer"}[phase]
    args = ["sbatch", "--parsable", "--partition", partition, f"--array=0-{count-1}%{concurrency}",
            "--cpus-per-task", str(cpus), "--mem", mem, "--time", time,
            "--output", str(batch / "slurm_logs/%x_%A_%a.out"), "--error", str(batch / "slurm_logs/%x_%A_%a.err"),
            f"--export=ALL,AOI_TASK_OFFSET={offset}"]
    if account:
        args += ["--account", account]
    if gres:
        args += ["--gres", gres]
    if dependency:
        args += ["--dependency", dependency]
    return args + [str(project / "hpc" / f"aoi_deployment_{launcher}.sbatch")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "main", "gap-transfer"), required=True)
    parser.add_argument("--partition", default="Q10")
    parser.add_argument("--account")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--gres", default="gpu:l20:1", help="empty string for CPU execution")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dependency")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    manifest, _ = dc.load_batch(args.batch_root)
    dc.require(dc.read(args.batch_root / "protocol/preflight_payload.json")["technical_status"] == "PASS", "payload preflight required")
    if args.phase == "main":
        pilot = args.batch_root / "analysis/pilot"
        passes = [p for p in pilot.glob("a*/verification.json") if dc.read(p).get("technical_status") == "PASS"]
        dc.require(passes, "pilot technical PASS required before main submission")
    data = dc.plan(args.batch_root, args.phase)
    environment = {**os.environ, "PROJECT_DIR": str(dc.REPO), "PYTHON_BIN": sys.executable,
                   "AOI_BATCH_ROOT": str(args.batch_root.resolve()), "AOI_DEVICE": args.device,
                   "AOI_EXPECTED_COMMIT": manifest["implementation_commit"]}
    previous, jobs, commands = args.dependency, [], []
    # Sequential chunks keep the requested concurrency a batch-wide bound.
    for index, (offset, count) in enumerate(chunks(len(data["units"]), args.chunk_size)):
        cmd = command(dc.REPO, args.batch_root, args.phase, offset, count, args.partition, args.account,
                      args.concurrency, args.cpus, args.mem, args.time, args.gres, previous)
        commands.append(cmd)
        if args.submit:
            job = subprocess.check_output(cmd, text=True, env=environment).strip().split(";")[0]
            dc.require(job.isdigit(), f"unexpected sbatch response: {job}")
            jobs.append({"job_id": job, "offset": offset, "local_array_count": count, "command": cmd})
            # Persist immediately: later submission failure must not hide earlier jobs.
            token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            dc.write(args.batch_root / f"protocol/submission_{args.phase}_{token}.json", jobs[-1])
            previous = f"afterok:{job}"
        else:
            previous = f"afterok:<job_chunk_{index}>"
    print(json.dumps({"submitted": args.submit, "jobs": jobs, "unit_count": len(data["units"]),
                      "environment": {k: environment[k] for k in ("PROJECT_DIR", "PYTHON_BIN", "AOI_BATCH_ROOT", "AOI_DEVICE", "AOI_EXPECTED_COMMIT")},
                      "commands": [shlex.join(cmd) for cmd in commands]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
