"""Build paper figures from saved artifacts only.

The module deliberately contains no environment or training calls.  Training
seeds are the independent unit for confidence intervals; held-out evaluation
seeds are clustered within a training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


CURRENT_ALGORITHM = "Modified_MADDPG_with_TDec"
REQUIRED_BASELINES = ("Modified_MADDPG", "MADDPG_FDec", "DDPG")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entries(manifest_or_root: Path) -> List[Dict[str, Any]]:
    path = Path(manifest_or_root).expanduser().resolve()
    if path.is_file():
        data = _load_json(path)
        entries = []
        for raw in data.get("entries", []):
            entry = dict(raw)
            for key in ("run_path", "eval_path", "checkpoint_path"):
                value = entry.get(key)
                if value:
                    reference = Path(str(value))
                    if not reference.is_absolute():
                        reference = path.parent / reference
                    entry[key] = str(reference.resolve())
            entry["_manifest_path"] = str(path)
            entries.append(entry)
        return entries
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
                "profile": config.get("profile"),
                "scenario": config.get("scenario", {}).get("id"),
                "training_seed": config.get("seed"),
                "run_path": str(run_dir),
                "eval_path": None if eval_dir is None else str(eval_dir),
                "eval_id": None if summary is None else summary.get("eval_id"),
                "eval_purpose": None if summary is None else summary.get("eval_purpose"),
                "status": "complete" if (run_dir / "COMPLETE.json").is_file() and (summary is None or summary.get("status") == "complete") else "incomplete",
            })
    return result


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) < 2:
        return values.copy()
    window = min(int(window), len(values))
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _moving_average_axis0(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.shape[0] < 2:
        return values.copy()
    window = min(int(window), values.shape[0])
    padded = np.concatenate([np.repeat(values[[0]], window - 1, axis=0), values], axis=0)
    cumulative = np.cumsum(padded, axis=0)
    previous = np.vstack([np.zeros_like(cumulative[[0]]), cumulative[:-window]])
    return (cumulative[window - 1:] - previous) / float(window)


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
        if entry.get("status", "complete") != "complete":
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
    task1, task2, training_seeds = [], [], []
    for entry in runs:
        with np.load(Path(entry["run_path"]) / "train_metrics.npz", allow_pickle=False) as metrics:
            task1.append(np.asarray(metrics["task1_episode_mean"], dtype=np.float64))
            task2.append(np.asarray(metrics["task2_episode_mean"], dtype=np.float64))
        if entry.get("training_seed") is not None:
            training_seeds.append(int(entry["training_seed"]))
    task1_stack = np.stack(task1)
    task2_stack = np.stack(task2)
    task1_mean = task1_stack.mean(axis=0)
    task2_mean = task2_stack.mean(axis=0)
    task1_sd = task1_stack.std(axis=0, ddof=1) if task1_stack.shape[0] > 1 else np.zeros_like(task1_mean)
    task2_sd = task2_stack.std(axis=0, ddof=1) if task2_stack.shape[0] > 1 else np.zeros_like(task2_mean)
    task1_ci = 1.96 * task1_sd / np.sqrt(max(1, task1_stack.shape[0]))
    task2_ci = 1.96 * task2_sd / np.sqrt(max(1, task2_stack.shape[0]))
    task1_smoothed = _moving_average_axis0(task1_mean, smoothing_window)
    task2_smoothed = _moving_average_axis0(task2_mean, smoothing_window)

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    for agent in range(task1_mean.shape[1]):
        axis.plot(task1_smoothed[:, agent], label=f"agent{agent + 1} task1")
        axis.plot(task2_smoothed[:, agent], linestyle="--", label=f"agent{agent + 1} task2")
    axis.set_xlabel("episode")
    axis.set_ylabel("episode mean reward")
    axis.set_title(f"Fig. 3 - {scenario}")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)

    data_path = output.with_suffix(".npz")
    np.savez_compressed(
        data_path,
        task1_raw=task1_stack,
        task2_raw=task2_stack,
        task1_mean=task1_mean,
        task2_mean=task2_mean,
        task1_ci95=task1_ci,
        task2_ci95=task2_ci,
        task1_smoothed=task1_smoothed,
        task2_smoothed=task2_smoothed,
    )
    _write_sidecar(output, {
        "figure": "Fig.3",
        "scenario": scenario,
        "smoothing_window": int(smoothing_window),
        "runs": [item["run_path"] for item in runs],
        "training_seeds": training_seeds,
        "agent_count": int(task1_mean.shape[1]),
        "aggregation": "mean_and_CI_across_training_seeds_preserving_agent_axis",
        "raw_data": data_path.name,
        "raw_curve_preserved": True,
        "smoothed_curve_preserved": True,
    })
    return output


def plot_fig4(
    manifest_or_root: Path,
    output: Path,
    scenario: str = "p05_n04_g05",
    required_baselines: Sequence[str] = REQUIRED_BASELINES,
    allow_incomplete: bool = False,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_path = Path(manifest_or_root).expanduser().resolve()
    manifest_data = _load_json(manifest_path) if manifest_path.is_file() else {}
    entries = _entries(manifest_path)
    available = {
        str(entry.get("algorithm"))
        for entry in entries
        if entry.get("scenario") == scenario
        and entry.get("algorithm") != CURRENT_ALGORITHM
        and entry.get("status", "complete") == "complete"
    }
    expected = list(manifest_data.get("required_baselines", required_baselines))
    missing = [name for name in expected if name not in available]
    output = Path(output).expanduser().resolve()
    if missing and not allow_incomplete:
        marker = output.parent / "INCOMPLETE_BASELINES.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "status": "INCOMPLETE_BASELINES",
            "figure": "Fig.4",
            "scenario": scenario,
            "required": expected,
            "missing": missing,
            "available": sorted(available),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"Fig.4 baselines are incomplete: {', '.join(missing)}")

    current_runs = _unique_run_entries(entries, scenario=scenario, algorithm=CURRENT_ALGORITHM)
    if not current_runs:
        raise FileNotFoundError(f"no current-algorithm runs for Fig.4 scenario {scenario}")

    def _curve(entry):
        with np.load(Path(entry["run_path"]) / "train_metrics.npz", allow_pickle=False) as metrics:
            algorithm = str(entry.get("algorithm"))
            if algorithm == CURRENT_ALGORITHM:
                key = "local_total_episode_mean"
            else:
                key = "reward_episode_mean" if "reward_episode_mean" in metrics.files else "global_episode_mean"
            if key not in metrics.files:
                raise ValueError(f"missing reward curve {key} for {algorithm}")
            value = np.asarray(metrics[key])
            return value.mean(axis=1) if value.ndim > 1 else value

    curves = {CURRENT_ALGORITHM: np.mean(np.stack([_curve(entry) for entry in current_runs]), axis=0)}
    for algorithm in sorted(available):
        baseline_runs = _unique_run_entries(entries, scenario=scenario, algorithm=algorithm)
        if baseline_runs:
            curves[algorithm] = np.mean(np.stack([_curve(entry) for entry in baseline_runs]), axis=0)
    partial = bool(missing)
    if partial and allow_incomplete and "PARTIAL" not in output.stem:
        output = output.with_name(output.stem + "_PARTIAL" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for algorithm in sorted(curves):
        axis.plot(curves[algorithm], label=algorithm)
    axis.set_xlabel("episode")
    axis.set_ylabel("reward curve")
    axis.set_title(f"Fig. 4 {'PARTIAL ' if partial else ''}- {scenario}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    _write_sidecar(output, {
        "figure": "Fig.4",
        "scenario": scenario,
        "runs": [item["run_path"] for item in current_runs],
        "baselines": sorted(available),
        "required_baselines": expected,
        "status": "INCOMPLETE_BASELINES" if partial else "complete",
        "partial": partial,
        "curves": sorted(curves),
        "current_algorithm_metrics": ["local_total_episode_mean", "global_episode_mean", "immediate_reward_proxy"],
    })
    return output


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Fig.5 values must be finite and non-empty")
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "sd": sd,
        "ci95": float(1.96 * sd / np.sqrt(len(array))),
        "count": int(len(array)),
    }


def plot_fig5(
    manifest_or_root: Path,
    output: Path,
    x_field: str = "gap_m",
    required_algorithms: Sequence[str] = (CURRENT_ALGORITHM,) + REQUIRED_BASELINES,
    allow_incomplete: bool = True,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if x_field not in {"gap_m", "platoon_size"}:
        raise ValueError("Fig.5 x_field must be gap_m or platoon_size")
    entries = _entries(manifest_or_root)
    required = set(required_algorithms)
    rows: Dict[str, Dict[float, List[float]]] = {}
    success_rows: Dict[str, Dict[float, List[float]]] = {}
    used = set()
    identities = set()
    available_algorithms = set()
    for entry in entries:
        algorithm = str(entry.get("algorithm"))
        if algorithm not in required or entry.get("profile", "paper_faithful") not in {None, "paper_faithful"} or entry.get("status") != "complete" or not entry.get("eval_path"):
            continue
        eval_path = Path(entry["eval_path"]).resolve()
        if str(eval_path) in used:
            continue
        summary_path = eval_path / "summary.json"
        run_config_path = Path(entry["run_path"]) / "config.resolved.json"
        if not summary_path.is_file() or not run_config_path.is_file():
            continue
        summary = _load_json(summary_path)
        if summary.get("eval_purpose") not in {None, "final_test"}:
            continue
        config = _load_json(run_config_path)
        scenario = config.get("scenario", {})
        value = float(scenario.get(x_field))
        training_seed = entry.get("training_seed", config.get("seed"))
        identity = (algorithm, scenario.get("id"), int(training_seed), summary.get("eval_purpose", "final_test"))
        if identity in identities:
            raise ValueError(f"duplicate training-seed eval artifact: {identity}")
        identities.add(identity)
        used.add(str(eval_path))
        available_algorithms.add(algorithm)
        aoi_values = summary.get("mean_AoI_ms_per_seed", [])
        success_values = summary.get("CAM_success_probability_per_seed", [])
        aoi_value = summary.get("mean_AoI_ms", float(np.mean(aoi_values)))
        success_value = summary.get("CAM_success_probability", float(np.mean(success_values)))
        rows.setdefault(algorithm, {}).setdefault(value, []).append(float(aoi_value))
        success_rows.setdefault(algorithm, {}).setdefault(value, []).append(float(success_value))
    if not rows:
        raise FileNotFoundError("no complete frozen-eval artifacts for Fig.5")

    missing_algorithms = [algorithm for algorithm in required_algorithms if algorithm not in available_algorithms]
    partial = bool(missing_algorithms)
    if partial and not allow_incomplete:
        raise RuntimeError("Fig.5 algorithms are incomplete: " + ", ".join(missing_algorithms))
    table = []
    for algorithm in sorted(rows):
        for value in sorted(rows[algorithm]):
            table.append({
                "algorithm": algorithm,
                "x": float(value),
                "AoI_ms": _stats(rows[algorithm][value]),
                "CAM_success_probability": _stats(success_rows[algorithm][value]),
            })

    output = Path(output).expanduser().resolve()
    if partial and "PARTIAL" not in output.stem:
        output = output.with_name(output.stem + "_PARTIAL" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for algorithm in sorted(rows):
        algorithm_rows = [row for row in table if row["algorithm"] == algorithm]
        x = np.asarray([row["x"] for row in algorithm_rows])
        for axis, key, label in (
            (axes[0], "AoI_ms", "mean AoI (ms)"),
            (axes[1], "CAM_success_probability", "CAM endpoint success"),
        ):
            mean = np.asarray([row[key]["mean"] for row in algorithm_rows])
            error = np.asarray([row[key]["ci95"] for row in algorithm_rows])
            axis.errorbar(x, mean, yerr=error, marker="o", capsize=3, label=algorithm)
            axis.set_xlabel(x_field)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
    figure.suptitle(f"Fig. 5 {'PARTIAL ' if partial else ''}- frozen evaluation sweep")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    _write_sidecar(output, {
        "figure": "Fig.5",
        "x_field": x_field,
        "rows": table,
        "reused_eval_artifacts": sorted(used),
        "required_algorithms": list(required_algorithms),
        "available_algorithms": sorted(available_algorithms),
        "missing_algorithms": missing_algorithms,
        "partial": partial,
        "ci_independent_unit": "training_seed",
        "eval_seeds_are_clustered_within_training_seed": True,
    })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_or_root")
    parser.add_argument("--figure", choices=("3", "4", "5", "all"), default="3")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--fig5-x", choices=("gap_m", "platoon_size"), default="gap_m")
    parser.add_argument("--allow-incomplete-baselines", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.manifest_or_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (root.parent / "figures" if root.is_file() else root / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = args.scenario or (["p05_n04_g05", "p07_n04_g05"] if args.figure == "4" else ["p05_n06_g25"])
    results = []
    if args.figure in {"3", "all"}:
        results.append(str(plot_fig3(root, output_dir / "fig3_training.png", scenarios[0], args.smooth_window)))
    if args.figure in {"4", "all"}:
        fig4_scenarios = scenarios if args.scenario else ["p05_n04_g05", "p07_n04_g05"]
        for scenario in fig4_scenarios:
            suffix = "" if len(fig4_scenarios) == 1 else f"_{scenario}"
            results.append(str(plot_fig4(root, output_dir / f"fig4_global_combined{suffix}.png", scenario, allow_incomplete=args.allow_incomplete_baselines)))
    if args.figure in {"5", "all"}:
        # Fig.5 may be generated as an explicitly labelled current-algorithm
        # partial while the three baselines are still unavailable.
        results.append(str(plot_fig5(root, output_dir / "fig5_sweep.png", args.fig5_x, allow_incomplete=True)))
    print(json.dumps({"outputs": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
