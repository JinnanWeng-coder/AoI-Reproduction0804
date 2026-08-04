"""Build Fig.3/4/5 from saved artifacts without running training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


CURRENT_ALGORITHM = "Modified_MADDPG_with_TDec"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entries(manifest_or_root: Path) -> List[Dict[str, Any]]:
    path = Path(manifest_or_root).expanduser().resolve()
    if path.is_file():
        data = _load_json(path)
        return list(data.get("entries", []))
    if not path.is_dir():
        raise FileNotFoundError(path)
    result = []
    for run_dir in sorted(child for child in path.iterdir() if child.is_dir()):
        config_path = run_dir / "config.resolved.json"
        if not config_path.is_file():
            continue
        config = _load_json(config_path)
        eval_root = run_dir / "eval"
        eval_dirs = sorted(child for child in eval_root.iterdir() if child.is_dir()) if eval_root.is_dir() else [None]
        for eval_dir in eval_dirs:
            summary = _load_json(eval_dir / "summary.json") if eval_dir is not None and (eval_dir / "summary.json").is_file() else None
            result.append({
                "algorithm": CURRENT_ALGORITHM,
                "semantic_version": config.get("semantic_version"),
                "scenario": config.get("scenario", {}).get("id"),
                "training_seed": config.get("seed"),
                "run_path": str(run_dir),
                "eval_path": None if eval_dir is None else str(eval_dir),
                "eval_id": None if summary is None else summary.get("eval_id"),
                "status": "complete" if (run_dir / "COMPLETE.json").is_file() and (summary is None or summary.get("status") == "complete") else "incomplete",
            })
    return result


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values
    window = min(int(window), len(values))
    if window < 2:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _write_sidecar(output: Path, data: Dict[str, Any]) -> None:
    output.with_suffix(".json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unique_run_entries(entries: Iterable[Dict[str, Any]], scenario: str = None, algorithm: str = None) -> List[Dict[str, Any]]:
    selected = []
    seen = set()
    for entry in entries:
        if scenario is not None and entry.get("scenario") != scenario:
            continue
        if algorithm is not None and entry.get("algorithm") != algorithm:
            continue
        run_path = entry.get("run_path")
        if not run_path or run_path in seen:
            continue
        seen.add(run_path)
        selected.append(entry)
    return selected


def plot_fig3(manifest_or_root: Path, output: Path, scenario: str = "p05_n06_g25", smoothing_window: int = 1) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = _entries(manifest_or_root)
    runs = _unique_run_entries(entries, scenario=scenario, algorithm=CURRENT_ALGORITHM)
    if not runs:
        raise FileNotFoundError(f"no current-algorithm runs for Fig.3 scenario {scenario}")
    task1, task2 = [], []
    for entry in runs:
        with np.load(Path(entry["run_path"]) / "train_metrics.npz", allow_pickle=False) as metrics:
            task1.append(metrics["task1_episode_mean"].mean(axis=1))
            task2.append(metrics["task2_episode_mean"].mean(axis=1))
    task1_mean = np.mean(np.stack(task1), axis=0)
    task2_mean = np.mean(np.stack(task2), axis=0)
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(_moving_average(task1_mean, smoothing_window), label="task1")
    axis.plot(_moving_average(task2_mean, smoothing_window), label="task2")
    axis.set_xlabel("episode")
    axis.set_ylabel("episode mean reward")
    axis.set_title(f"Fig. 3 — {scenario}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    _write_sidecar(output, {"figure": "Fig.3", "scenario": scenario, "smoothing_window": int(smoothing_window), "runs": [item["run_path"] for item in runs], "aggregation": "mean over seeds then agents"})
    return output


def plot_fig4(manifest_or_root: Path, output: Path, scenario: str = "p05_n06_g25", required_baselines: Sequence[str] = ("MADDPG", "DQN"), allow_incomplete: bool = False) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_path = Path(manifest_or_root).expanduser().resolve()
    manifest_data = _load_json(manifest_path) if manifest_path.is_file() else {}
    entries = _entries(manifest_path)
    available = {str(entry.get("algorithm")) for entry in entries if entry.get("scenario") == scenario and entry.get("algorithm") != CURRENT_ALGORITHM}
    expected = list(manifest_data.get("required_baselines", required_baselines))
    missing = [name for name in expected if name not in available]
    output = Path(output).expanduser().resolve()
    if missing and not allow_incomplete:
        marker = output.parent / "INCOMPLETE_BASELINES.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"status": "INCOMPLETE_BASELINES", "scenario": scenario, "missing": missing, "available": sorted(available)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"Fig.4 baselines are incomplete: {', '.join(missing)}")

    runs = _unique_run_entries(entries, scenario=scenario, algorithm=CURRENT_ALGORITHM)
    if not runs:
        raise FileNotFoundError(f"no current-algorithm runs for Fig.4 scenario {scenario}")
    local, global_mean, combined = [], [], []
    for entry in runs:
        with np.load(Path(entry["run_path"]) / "train_metrics.npz", allow_pickle=False) as metrics:
            local.append(metrics["local_total_episode_mean"].mean(axis=1))
            global_mean.append(metrics["global_episode_mean"])
            combined.append(metrics["immediate_reward_proxy"].mean(axis=1))
    local = np.mean(np.stack(local), axis=0)
    global_mean = np.mean(np.stack(global_mean), axis=0)
    combined = np.mean(np.stack(combined), axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(local, label="local_total")
    axis.plot(global_mean, label="global_episode_mean")
    axis.plot(combined, label="immediate_reward_proxy")
    axis.set_xlabel("episode")
    axis.set_ylabel("same-time-reduction reward")
    axis.set_title(f"Fig. 4 — {scenario}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    _write_sidecar(output, {"figure": "Fig.4", "scenario": scenario, "runs": [item["run_path"] for item in runs], "baselines": sorted(available), "status": "INCOMPLETE_BASELINES" if missing else "complete", "combined_metric": "immediate_reward_proxy"})
    return output


def plot_fig5(manifest_or_root: Path, output: Path, x_field: str = "gap_m") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = _entries(manifest_or_root)
    rows: Dict[float, List[float]] = {}
    success_rows: Dict[float, List[float]] = {}
    used = set()
    for entry in entries:
        if entry.get("algorithm") != CURRENT_ALGORITHM or entry.get("profile", "paper_faithful") not in {None, "paper_faithful"} or entry.get("status") != "complete" or not entry.get("eval_path"):
            continue
        eval_path = Path(entry["eval_path"]).resolve()
        if str(eval_path) in used:
            continue
        used.add(str(eval_path))
        summary_path = eval_path / "summary.json"
        run_config_path = Path(entry["run_path"]) / "config.resolved.json"
        if not summary_path.is_file() or not run_config_path.is_file():
            continue
        summary = _load_json(summary_path)
        config = _load_json(run_config_path)
        scenario = config.get("scenario", {})
        value = float(scenario.get(x_field))
        rows.setdefault(value, []).extend(float(item) for item in summary.get("mean_AoI_ms_per_seed", []))
        success_rows.setdefault(value, []).extend(float(item) for item in summary.get("CAM_success_probability_per_seed", []))
    if not rows:
        raise FileNotFoundError("no complete frozen-eval artifacts for Fig.5")

    def stats(values):
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        ci = float(1.96 * sd / np.sqrt(len(array))) if len(array) else 0.0
        return {"mean": mean, "sd": sd, "ci95": ci, "count": int(len(array))}

    table = [{"x": float(value), "AoI_ms": stats(rows[value]), "CAM_success_probability": stats(success_rows[value])} for value in sorted(rows)]
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.asarray([row["x"] for row in table])
    for axis, key, label in ((axes[0], "AoI_ms", "mean AoI (ms)"), (axes[1], "CAM_success_probability", "CAM endpoint success")):
        mean = np.asarray([row[key]["mean"] for row in table])
        error = np.asarray([row[key]["ci95"] for row in table])
        axis.errorbar(x, mean, yerr=error, marker="o", capsize=3)
        axis.set_xlabel(x_field)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    figure.suptitle("Fig. 5 — frozen evaluation sweep")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    _write_sidecar(output, {"figure": "Fig.5", "x_field": x_field, "rows": table, "reused_eval_artifacts": sorted(used)})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_or_root")
    parser.add_argument("--figure", choices=("3", "4", "5", "all"), default="3")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scenario", default="p05_n06_g25")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--fig5-x", default="gap_m")
    parser.add_argument("--allow-incomplete-baselines", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.manifest_or_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (root.parent / "figures" if root.is_file() else root / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    if args.figure in {"3", "all"}:
        results.append(str(plot_fig3(root, output_dir / "fig3_training.png", args.scenario, args.smooth_window)))
    if args.figure in {"4", "all"}:
        results.append(str(plot_fig4(root, output_dir / "fig4_global_combined.png", args.scenario, allow_incomplete=args.allow_incomplete_baselines)))
    if args.figure in {"5", "all"}:
        results.append(str(plot_fig5(root, output_dir / "fig5_sweep.png", args.fig5_x)))
    print(json.dumps({"outputs": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
