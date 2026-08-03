"""Restart-safe 48-run matrix listing and optional executor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import matrix_specs, safe_run_dir


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="paper_faithful")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    specs = matrix_specs(args.profile)
    print(json.dumps(specs, indent=2, sort_keys=True))
    print(f"matrix_count={len(specs)}")
    print(f"unique_count={len({(item['profile'], item['scenario'], item['seed']) for item in specs})}")
    if args.dry_run:
        return 0
    for item in specs:
        config = safe_run_dir("experiments/runs", item["run_name"])
        if config.exists():
            if (config / "COMPLETE.json").exists():
                print(f"SKIP complete {item['run_name']}")
                continue
            raise RuntimeError(f"refusing incomplete existing run: {config}")
        command = [
            sys.executable,
            str(ROOT / "Main.py"),
            "--profile", item["profile"],
            "--scenario", item["scenario"],
            "--seed", str(item["seed"]),
            "--device", args.device,
            "--run-name", item["run_name"],
        ]
        print("RUN", " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
