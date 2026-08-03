"""Build paper-facing curves from already saved result files."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.plot_training import plot_run


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    print(plot_run(Path(args.run_dir), Path(args.run_dir) / "figures" / "paper_curves.png"))


if __name__ == "__main__":
    main()

