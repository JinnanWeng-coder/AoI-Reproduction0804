"""One explicit stage per invocation, dry-run by default; no automatic eval."""
import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from analysis import deployment_contract as dc, training100_contract as tc

ARRAYS = {"all": "0-23", "pilot": "0,6,12,18", "main": "1-5,7-11,13-17,19-23"}


def command(batch, stage, partition="Q10", account=None, concurrency=6, cpus=8, mem="32G", time="08:00:00", dependency=None):
    dc.require(1 <= concurrency <= 6 and cpus > 0, "resource/concurrency contract")
    train = stage in ARRAYS
    cmd = ["sbatch", "--parsable", "--job-name", f"train100_{stage}", "--partition", partition,
           "--cpus-per-task", str(cpus), "--mem", mem, "--time", time,
           "--output", str(batch / "slurm_logs/%x_%A_%a.out"), "--error", str(batch / "slurm_logs/%x_%A_%a.err"),
           f"--export=ALL,AOI_TRAIN100_STAGE={stage}"]
    if train: cmd += [f"--array={ARRAYS[stage]}%{concurrency}", "--gres", "gpu:l20:1"]
    if account: cmd += ["--account", account]
    if dependency: cmd += ["--dependency", dependency]
    return cmd + [str(dc.REPO / "hpc" / ("aoi_training100_train.sbatch" if train else "aoi_training100_cpu.sbatch"))]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--stage", choices=("audit", "all", "pilot", "pilot-check", "main", "summary", "export"), required=True)
    p.add_argument("--partition", default="Q10")
    p.add_argument("--account")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="32G")
    p.add_argument("--time", default="08:00:00")
    p.add_argument("--dependency")
    p.add_argument("--submit", action="store_true")
    a = p.parse_args(argv)
    m, _ = tc.load(a.batch_root)
    if a.stage in ARRAYS:
        decision = dc.read(a.batch_root / "protocol/execution_decision.json")
        dc.require(decision["decision"] == "24_FROM_SCRATCH" and
                   decision.get("implementation_commit") == m["implementation_commit"], "audit and pinned decision required")
        if a.stage == "all":
            dc.require(decision.get("execution_mode") == "direct24" and decision.get("pilot_required") is False,
                       "direct24 approval required")
    if a.stage == "main":
        dc.require(dc.read(a.batch_root / "protocol/pilot_verification.json")["technical_status"] == "PASS", "pilot PASS required")
    if a.submit:
        previous = list((a.batch_root / "protocol").glob(f"submission_*_{a.stage}.json"))
        dc.require(not previous, "stage already submitted: inspect/recover only missing tasks, do not blindly resubmit")
        conflicts = ("pilot", "main") if a.stage == "all" else (("all",) if a.stage in {"pilot", "main"} else ())
        dc.require(not any(list((a.batch_root / "protocol").glob(f"submission_*_{s}.json")) for s in conflicts),
                   "do not mix direct24 and pilot/main submissions in the same batch")
    cmd = command(a.batch_root.resolve(), a.stage, a.partition, a.account, a.concurrency, a.cpus, a.mem, a.time, a.dependency)
    env = {**os.environ, "PROJECT_DIR": str(dc.REPO), "PYTHON_BIN": sys.executable,
           "BATCH_ROOT": str(a.batch_root.resolve()), "AOI_EXPECTED_COMMIT": m["implementation_commit"], "AOI_ATTEMPT": "a01"}
    for key in ("AOI_TASK_OFFSET", "AOI_RETRY_REASON", "AOI_ACCEPT_EXISTING", "AOI_CHECK_ONLY"):
        env.pop(key, None)
    report = {"submitted": a.submit, "stage": a.stage, "command": shlex.join(cmd)}
    if a.submit:
        job = subprocess.check_output(cmd, text=True, env=env).strip().split(";")[0]
        dc.require(job.isdigit(), "invalid sbatch job ID")
        report.update(job_id=job, implementation_commit=m["implementation_commit"], environment={k: env[k] for k in
                      ("PROJECT_DIR", "PYTHON_BIN", "BATCH_ROOT", "AOI_EXPECTED_COMMIT", "AOI_ATTEMPT")})
        dc.write(a.batch_root / f"protocol/submission_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{a.stage}.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
