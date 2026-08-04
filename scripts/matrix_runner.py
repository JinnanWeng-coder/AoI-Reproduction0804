"""Restart-safe 48-run train/eval/audit matrix orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import matrix_specs, safe_run_dir


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(item, args, stage: str, resume: Path = None):
    command = [sys.executable, str(ROOT / "Main.py"), "--profile", item["profile"], "--scenario", item["scenario"], "--seed", str(item["seed"]), "--device", args.device, "--run-name", item["run_name"], "--output-root", args.output_root]
    if stage == "train":
        if resume is not None:
            command.extend(["--resume", str(resume)])
    elif stage == "eval":
        command.extend(["--eval-only", "--eval-episodes", str(args.eval_episodes), "--eval-seeds", args.eval_seeds])
        command.extend(["--resume", str(resume)])
    return command


def _run(command, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"exit={completed.returncode}\n")
    return completed.returncode


def _has_matching_eval(run_dir: Path, checkpoint_hash: str, args) -> bool:
    eval_root = run_dir / "eval"
    if not eval_root.is_dir():
        return False
    expected_seeds = [int(item) for item in args.eval_seeds.split(",") if item.strip()]
    for child in eval_root.iterdir():
        summary_path = child / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("status") == "complete" and summary.get("checkpoint_sha256") == checkpoint_hash and summary.get("eval_episodes") == args.eval_episodes and summary.get("eval_seeds") == expected_seeds:
            return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="paper_faithful")
    parser.add_argument("--stage", choices=("train", "eval", "audit", "all"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="experiments/runs")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seeds", default="101,102,103,104,105,106")
    parser.add_argument("--log-dir", default="batch_logs")
    parser.add_argument("--report", default=None, help="write a JSON dry-run/execution report")
    args = parser.parse_args(argv)
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    specs = matrix_specs(args.profile)
    keys = {(item["profile"], item["scenario"], item["seed"]) for item in specs}
    print(json.dumps(specs, indent=2, sort_keys=True))
    print(f"matrix_count={len(specs)}")
    print(f"unique_count={len(keys)}")
    if len(specs) != 48 or len(keys) != 48:
        raise RuntimeError("matrix must contain exactly 48 unique tasks")
    if args.dry_run:
        report_commands = []
        for item in specs:
            run_dir = safe_run_dir(args.output_root, item["run_name"])
            latest = run_dir / "checkpoints" / "latest.pt"
            train_command = _command(item, args, "train", latest if latest.exists() else None)
            eval_command = _command(item, args, "eval", latest)
            audit_command = [sys.executable, str(ROOT / "analysis" / "audit_results.py"), str(run_dir), "--require-eval"]
            print("TRAIN", " ".join(train_command))
            print("EVAL", " ".join(eval_command))
            print("AUDIT", " ".join(audit_command))
            report_commands.append({"run_name": item["run_name"], "train": train_command, "eval": eval_command, "audit": audit_command})
        if args.report:
            report_path = Path(args.report).expanduser()
            if not report_path.is_absolute():
                report_path = ROOT / report_path
            if report_path.exists():
                raise FileExistsError(f"Refusing to overwrite matrix report: {report_path}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({"status": "dry_run", "matrix_count": len(specs), "unique_count": len(keys), "specs": specs, "commands": report_commands}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    stages = ("train", "eval", "audit") if args.stage == "all" else (args.stage,)
    failures = []
    log_dir = Path(args.log_dir).expanduser()
    for item in specs:
        run_dir = safe_run_dir(args.output_root, item["run_name"])
        latest = run_dir / "checkpoints" / "latest.pt"
        log_path = log_dir / f"{item['run_name']}.log"
        for stage in stages:
            if stage == "train":
                if (run_dir / "COMPLETE.json").is_file():
                    print(f"SKIP complete {item['run_name']}")
                    continue
                resume = latest if latest.is_file() else None
                if _run(_command(item, args, "train", resume), log_path) != 0:
                    failures.append({"stage": stage, "run": item["run_name"]})
                    break
            elif stage == "eval":
                if not latest.is_file():
                    failures.append({"stage": stage, "run": item["run_name"], "error": "missing latest checkpoint"})
                    break
                if _has_matching_eval(run_dir, _checkpoint_hash(latest), args):
                    print(f"SKIP eval {item['run_name']}")
                    continue
                if _run(_command(item, args, "eval", latest), log_path) != 0:
                    failures.append({"stage": stage, "run": item["run_name"]})
                    break
            else:
                command = [sys.executable, str(ROOT / "analysis" / "audit_results.py"), str(run_dir), "--require-eval"]
                if _run(command, log_path) != 0:
                    failures.append({"stage": stage, "run": item["run_name"]})
                    break
    report = {"status": "failed" if failures else "complete", "failures": failures, "matrix_count": len(specs), "unique_count": len(keys)}
    if args.report:
        report_path = Path(args.report).expanduser()
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite matrix report: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
