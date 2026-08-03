"""Summarize completed run metadata without rerunning experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(root: Path):
    rows = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        config_path = run_dir / "config.resolved.json"
        complete_path = run_dir / "COMPLETE.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows.append({"run_name": run_dir.name, "profile": config["profile"], "scenario": config["scenario"]["id"], "seed": config["seed"], "complete": complete_path.exists()})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    rows = summarize(Path(args.root))
    if args.output:
        with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_name", "profile", "scenario", "seed", "complete"])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"count": len(rows), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

