"""Plot training metrics from files only; never re-executes the environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def plot_run(run_dir: Path, output: Optional[Path] = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    if output is None:
        output = run_dir / "figures" / "training_curves.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(run_dir / "train_metrics.npz", allow_pickle=False) as metrics:
        task1 = metrics["task1_episode_mean"].mean(axis=1)
        task2 = metrics["task2_episode_mean"].mean(axis=1)
        global_reward = metrics["global_episode_sum"]
    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(task1, label="task1")
    axes[1].plot(task2, label="task2")
    axes[2].plot(global_reward, label="global")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("episode")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    (output.with_suffix(".json")).write_text('{"source": "train_metrics.npz", "aggregation": "mean over agents"}\n', encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    print(plot_run(Path(args.run_dir)))


if __name__ == "__main__":
    main()
