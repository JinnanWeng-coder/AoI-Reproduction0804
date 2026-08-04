"""Build paper figures from saved artifacts only.

The module deliberately contains no environment or training calls.  Training
seeds are the independent unit for confidence intervals; held-out evaluation
seeds are clustered within a training run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


CURRENT_ALGORITHM = "Modified_MADDPG_with_TDec"
REQUIRED_BASELINES = ("Modified_MADDPG", "MADDPG_FDec", "DDPG")
FIG3_SCENARIO = "p05_n06_g25"
FIG5_GAP_SCENARIOS = ("p05_n04_g05", "p05_n04_g15", "p05_n04_g25", "p05_n04_g35")
FIG5_SIZE_SCENARIOS = ("p05_n04_g25", "p05_n06_g25", "p05_n08_g25", "p05_n10_g25")
FORMAL_TRAINING_SEEDS = tuple(range(2, 8))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_identity(value: Any) -> str:
    """Return a platform-correct identity for an artifact path."""
    return os.path.normcase(str(Path(str(value)).expanduser().resolve()))


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

    if scenario != FIG3_SCENARIO:
        raise ValueError(f"Fig.3 is fixed to scenario {FIG3_SCENARIO}")
    entries = _entries(manifest_or_root)
    runs = _unique_run_entries(entries, scenario=scenario, algorithm=CURRENT_ALGORITHM)
    if not runs:
        raise FileNotFoundError(f"no current-algorithm runs for Fig.3 scenario {scenario}")
    task1, task2, training_seeds = [], [], []
    for entry in runs:
        with np.load(Path(entry["run_path"]) / "train_metrics.npz", allow_pickle=False) as metrics:
            task1.append(np.asarray(metrics["task1_episode_mean"], dtype=np.float64))
            task2.append(np.asarray(metrics["task2_episode_mean"], dtype=np.float64))
        if entry.get("training_seed") is None:
            raise ValueError("Fig.3 requires training_seed in every entry")
        training_seeds.append(int(entry["training_seed"]))
    if len(training_seeds) != len(set(training_seeds)):
        raise ValueError("Fig.3 duplicate training seed")
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
        episodes = np.arange(task1_mean.shape[0])
        axis.fill_between(episodes, task1_smoothed[:, agent] - task1_ci[:, agent], task1_smoothed[:, agent] + task1_ci[:, agent], alpha=0.08)
        axis.fill_between(episodes, task2_smoothed[:, agent] - task2_ci[:, agent], task2_smoothed[:, agent] + task2_ci[:, agent], alpha=0.08)
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
        task1_sd=task1_sd,
        task2_sd=task2_sd,
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
    expected_training_seeds: Optional[Sequence[int]] = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_path = Path(manifest_or_root).expanduser().resolve()
    manifest_data = _load_json(manifest_path) if manifest_path.is_file() else {}
    entries = _entries(manifest_path)
    expected = list(dict.fromkeys(str(name) for name in manifest_data.get("required_baselines", required_baselines)))
    omitted_paper_baselines = [algorithm for algorithm in REQUIRED_BASELINES if algorithm not in expected]
    if omitted_paper_baselines:
        raise ValueError(f"Fig.4 required_baselines cannot omit paper baselines: {omitted_paper_baselines}")
    if CURRENT_ALGORITHM in expected:
        raise ValueError("Fig.4 required_baselines must not contain the current algorithm")
    expected_algorithms = (CURRENT_ALGORITHM,) + tuple(expected)
    expected_training_seeds = tuple(sorted({
        int(seed)
        for seed in (FORMAL_TRAINING_SEEDS if expected_training_seeds is None else expected_training_seeds)
    }))
    if not expected_training_seeds:
        raise ValueError("Fig.4 expected_training_seeds must be non-empty")

    # Preserve a fast, explicit missing-baseline diagnostic before validating
    # per-cell metadata.  This is more useful than reporting an unrelated
    # malformed current entry when an entire comparison algorithm is absent.
    raw_available_baselines = {
        str(entry.get("algorithm"))
        for entry in entries
        if entry.get("scenario") == scenario
        and entry.get("status", "complete") == "complete"
        and str(entry.get("algorithm")) in expected
    }
    entirely_missing_baselines = [algorithm for algorithm in expected if algorithm not in raw_available_baselines]
    output = Path(output).expanduser().resolve()
    if entirely_missing_baselines and not allow_incomplete:
        marker = output.parent / "INCOMPLETE_BASELINES.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "status": "INCOMPLETE_BASELINES",
            "figure": "Fig.4",
            "scenario": scenario,
            "required": expected,
            "required_algorithms": list(expected_algorithms),
            "expected_training_seeds": list(expected_training_seeds),
            "missing": entirely_missing_baselines,
            "available": sorted(raw_available_baselines),
            "missing_cells": [
                {"algorithm": algorithm, "scenario": scenario, "training_seed": seed}
                for algorithm in entirely_missing_baselines
                for seed in expected_training_seeds
            ],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"Fig.4 baselines are incomplete: {', '.join(entirely_missing_baselines)}")

    # A training seed is the independent experimental unit.  Do not silently
    # collapse duplicate manifest entries (for example, two eval rows pointing
    # at the same training run) because that can otherwise bias a curve while
    # still looking like a complete study.
    cell_entries: Dict[tuple, Dict[str, Any]] = {}
    cell_run_identities: Dict[tuple, str] = {}
    duplicate_cells = []
    unexpected_cells = []
    for entry in entries:
        if entry.get("scenario") != scenario or entry.get("status", "complete") != "complete":
            continue
        algorithm = str(entry.get("algorithm"))
        if algorithm not in expected_algorithms:
            continue
        seed = entry.get("training_seed")
        if seed is None:
            raise ValueError(f"Fig.4 requires training_seed for {algorithm} in {scenario}")
        seed = int(seed)
        cell = (algorithm, scenario, seed)
        if seed not in expected_training_seeds:
            unexpected_cells.append({"algorithm": algorithm, "scenario": scenario, "training_seed": seed})
            continue
        run_path = entry.get("run_path")
        if not run_path:
            raise ValueError(f"Fig.4 cell has no run_path: {cell}")
        run_identity = _path_identity(run_path)
        if cell in cell_entries:
            if cell_run_identities[cell] == run_identity:
                # A study manifest has one row per eval artifact.  Validation
                # and final-test rows from the same training run are therefore
                # legitimate aliases of one Fig.4 training cell.
                continue
            duplicate_cells.append({
                "algorithm": algorithm,
                "scenario": scenario,
                "training_seed": seed,
                "first_run_path": cell_entries[cell]["run_path"],
                "duplicate_run_path": run_path,
            })
            continue
        cell_entries[cell] = entry
        cell_run_identities[cell] = run_identity
    if unexpected_cells:
        raise ValueError(f"Fig.4 contains training seeds outside the expected grid: {unexpected_cells}")
    if duplicate_cells:
        raise ValueError(f"Fig.4 duplicate algorithm/scenario/training-seed cells: {duplicate_cells}")

    expected_cells = [
        (algorithm, scenario, seed)
        for algorithm in expected_algorithms
        for seed in expected_training_seeds
    ]
    missing_cells = [
        {"algorithm": algorithm, "scenario": scenario_id, "training_seed": seed}
        for algorithm, scenario_id, seed in expected_cells
        if (algorithm, scenario_id, seed) not in cell_entries
    ]
    available = {
        algorithm
        for algorithm in expected
        if any(cell[0] == algorithm for cell in cell_entries)
    }
    missing = [algorithm for algorithm in expected if algorithm not in available]
    per_algorithm_training_seeds = {
        algorithm: sorted(cell[2] for cell in cell_entries if cell[0] == algorithm)
        for algorithm in expected_algorithms
    }
    current_runs = [
        cell_entries[(CURRENT_ALGORITHM, scenario, seed)]
        for seed in expected_training_seeds
        if (CURRENT_ALGORITHM, scenario, seed) in cell_entries
    ]
    if not current_runs:
        raise FileNotFoundError(f"no current-algorithm runs for Fig.4 scenario {scenario}")

    if missing_cells and not allow_incomplete:
        marker = output.parent / "INCOMPLETE_BASELINES.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "status": "INCOMPLETE_BASELINES" if missing else "INCOMPLETE_GRID",
            "figure": "Fig.4",
            "scenario": scenario,
            "required": expected,
            "required_algorithms": list(expected_algorithms),
            "expected_training_seeds": list(expected_training_seeds),
            "missing": missing,
            "available": sorted(available),
            "missing_cells": missing_cells,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if missing:
            raise RuntimeError(f"Fig.4 baselines are incomplete: {', '.join(missing)}; grid has {len(missing_cells)} missing cells")
        raise RuntimeError(f"Fig.4 algorithm/scenario/training-seed grid is incomplete: {len(missing_cells)} missing cells")

    def _metric(entry, key, fallback=None):
        with np.load(Path(entry["run_path"]) / "train_metrics.npz", allow_pickle=False) as metrics:
            selected = key if key in metrics.files else fallback
            if selected is None or selected not in metrics.files:
                raise ValueError(f"missing training metric {key} for {entry.get('algorithm')}")
            value = np.asarray(metrics[selected], dtype=np.float64)
            return value.mean(axis=1) if value.ndim > 1 else value

    current_metrics = {
        "task1": np.mean(np.stack([_metric(entry, "task1_episode_mean") for entry in current_runs]), axis=0),
        "task2": np.mean(np.stack([_metric(entry, "task2_episode_mean") for entry in current_runs]), axis=0),
        "global": np.mean(np.stack([_metric(entry, "global_episode_mean") for entry in current_runs]), axis=0),
        "combined": np.mean(np.stack([_metric(entry, "local_total_episode_mean") for entry in current_runs]), axis=0),
        "objective_proxy": np.mean(np.stack([_metric(entry, "immediate_reward_proxy") for entry in current_runs]), axis=0),
    }
    curves = {CURRENT_ALGORITHM: current_metrics["combined"]}
    for algorithm in expected:
        baseline_runs = [
            cell_entries[(algorithm, scenario, seed)]
            for seed in expected_training_seeds
            if (algorithm, scenario, seed) in cell_entries
        ]
        if baseline_runs:
            curves[algorithm] = np.mean(np.stack([_metric(entry, "reward_episode_mean", "global_episode_mean") for entry in baseline_runs]), axis=0)
    partial = bool(missing_cells)
    if partial and allow_incomplete and "PARTIAL" not in output.stem:
        output = output.with_name(output.stem + "_PARTIAL" + output.suffix)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = ((axes[0, 0], "task1", "task1 reward"), (axes[0, 1], "task2", "task2 reward"), (axes[1, 0], "global", "global reward"), (axes[1, 1], "combined", "combined reward"))
    for axis, key, ylabel in panels:
        axis.plot(current_metrics[key], label=CURRENT_ALGORITHM)
        if key == "combined":
            for algorithm in expected:
                if algorithm not in curves:
                    continue
                axis.plot(curves[algorithm], label=algorithm)
        axis.set_xlabel("episode")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle(f"Fig. 4 {'PARTIAL ' if partial else ''}- {scenario}")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    saved_arrays = {
        "task1": current_metrics["task1"],
        "task2": current_metrics["task2"],
        "global_reward": current_metrics["global"],
        "combined": current_metrics["combined"],
        "training_objective_proxy": current_metrics["objective_proxy"],
    }
    algorithm_curve_artifacts = {}
    for index, algorithm in enumerate(expected_algorithms):
        if algorithm in curves:
            artifact_key = f"algorithm_curve_{index}"
            saved_arrays[artifact_key] = curves[algorithm]
            algorithm_curve_artifacts[algorithm] = artifact_key
    np.savez_compressed(output.with_suffix(".npz"), **saved_arrays)
    _write_sidecar(output, {
        "figure": "Fig.4",
        "scenario": scenario,
        "runs": [item["run_path"] for item in current_runs],
        "baselines": sorted(available),
        "required_baselines": expected,
        "required_algorithms": list(expected_algorithms),
        "status": "PARTIAL" if partial else "complete",
        "partial": partial,
        "curves": sorted(curves),
        "training_seeds": per_algorithm_training_seeds[CURRENT_ALGORITHM],
        "per_algorithm_training_seeds": per_algorithm_training_seeds,
        "expected_training_seeds": list(expected_training_seeds),
        "missing_training_seeds": sorted({
            cell["training_seed"]
            for cell in missing_cells
            if cell["algorithm"] == CURRENT_ALGORITHM
        }),
        "missing_cells": missing_cells,
        "aggregation_unit": "training_seed",
        "algorithm_curve_artifacts": algorithm_curve_artifacts,
        "current_algorithm_metrics": ["task1_episode_mean", "task2_episode_mean", "global_episode_mean", "local_total_episode_mean", "immediate_reward_proxy"],
        "saved_metric_artifact": output.with_suffix(".npz").name,
        "drawn_panels": ["task1", "task2", "global", "combined"],
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
    allow_incomplete: bool = False,
    eval_purpose: Optional[str] = None,
    expected_training_seeds: Optional[Sequence[int]] = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if x_field not in {"gap_m", "platoon_size"}:
        raise ValueError("Fig.5 x_field must be gap_m or platoon_size")
    if eval_purpose not in {"validation", "final_test"}:
        raise ValueError("Fig.5 requires an explicit eval_purpose")
    if expected_training_seeds is None:
        expected_training_seeds = FORMAL_TRAINING_SEEDS
    expected_training_seeds = tuple(sorted({int(seed) for seed in expected_training_seeds}))
    if not expected_training_seeds:
        raise ValueError("Fig.5 expected_training_seeds must be non-empty")
    entries = _entries(manifest_or_root)
    required_algorithms = tuple(dict.fromkeys(str(name) for name in required_algorithms))
    required = set(required_algorithms)
    paper_algorithms = (CURRENT_ALGORITHM,) + REQUIRED_BASELINES
    omitted_paper_algorithms = [algorithm for algorithm in paper_algorithms if algorithm not in required]
    if omitted_paper_algorithms:
        raise ValueError(f"Fig.5 required_algorithms cannot omit paper algorithms: {omitted_paper_algorithms}")
    expected_scenarios = FIG5_GAP_SCENARIOS if x_field == "gap_m" else FIG5_SIZE_SCENARIOS
    expected_algorithm_set = {CURRENT_ALGORITHM} if eval_purpose == "validation" else required
    rows: Dict[str, Dict[float, List[float]]] = {}
    success_rows: Dict[str, Dict[float, List[float]]] = {}
    used = set()
    identities = set()
    available_algorithms = set()
    observed_purposes = set()
    candidate_entries = [
        entry for entry in entries
        if str(entry.get("algorithm")) in required
        and entry.get("profile", "paper_faithful") in {None, "paper_faithful"}
        and entry.get("status") == "complete"
        and entry.get("eval_path")
    ]
    for entry in candidate_entries:
        algorithm = str(entry.get("algorithm"))
        eval_path = Path(entry["eval_path"]).resolve()
        summary_path = eval_path / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"Fig.5 artifact is missing summary: {eval_path}")
        summary = _load_json(summary_path)
        summary_purpose = summary.get("eval_purpose")
        if summary_purpose is None:
            raise ValueError(f"Fig.5 artifact has no eval_purpose: {eval_path}")
        summary_purpose = str(summary_purpose)
        entry_purpose = entry.get("eval_purpose")
        if entry_purpose is not None and str(entry_purpose) != summary_purpose:
            raise ValueError(
                f"Fig.5 entry/summary eval purpose mismatch for {eval_path}: "
                f"entry={entry_purpose}, summary={summary_purpose}"
            )
        # A complete lifecycle manifest legitimately contains both validation
        # and final-test rows.  Select the requested population only after the
        # row's own provenance fields have been cross-checked.
        if summary_purpose != eval_purpose:
            continue
        purpose = summary_purpose
        observed_purposes.add(str(purpose))
        run_config_path = Path(entry["run_path"]) / "config.resolved.json"
        if not run_config_path.is_file():
            raise ValueError(f"Fig.5 artifact is missing run config: {eval_path}")
        if summary.get("scope") != ("validation" if eval_purpose == "validation" else "final_release"):
            raise ValueError(f"Fig.5 scope mismatch for {eval_path}")
        if summary.get("is_formal_result") is not (False if eval_purpose == "validation" else True):
            raise ValueError(f"Fig.5 formal-result marker mismatch for {eval_path}")
        if eval_path.as_posix() in used:
            raise ValueError(f"Fig.5 duplicate eval artifact: {eval_path}")
        config = _load_json(run_config_path)
        scenario = config.get("scenario", {})
        scenario_id = scenario.get("id")
        if scenario_id not in expected_scenarios:
            raise ValueError(f"Fig.5 scenario is outside the controlled grid: {scenario_id}")
        if x_field == "gap_m":
            expected_gap = float(scenario_id.split("_g", 1)[1])
            if int(scenario.get("number_platoons", -1)) != 5 or int(scenario.get("platoon_size", -1)) != 4 or float(scenario.get("gap_m", -1)) != expected_gap:
                raise ValueError(f"Fig.5 gap scenario configuration mismatch: {scenario_id}")
        else:
            expected_size = int(scenario_id.split("_n", 1)[1].split("_g", 1)[0])
            if int(scenario.get("number_platoons", -1)) != 5 or int(scenario.get("platoon_size", -1)) != expected_size or float(scenario.get("gap_m", -1)) != 25.0:
                raise ValueError(f"Fig.5 size scenario configuration mismatch: {scenario_id}")
        if algorithm not in expected_algorithm_set:
            raise ValueError(f"Fig.5 {eval_purpose} artifacts contain an unexpected algorithm: {algorithm}")
        training_seed = entry.get("training_seed", config.get("seed"))
        if training_seed is None:
            raise ValueError(f"Fig.5 requires training_seed: {eval_path}")
        training_seed = int(training_seed)
        if training_seed not in expected_training_seeds:
            raise ValueError(f"Fig.5 training seed {training_seed} is outside expected set {expected_training_seeds}")
        identity = (algorithm, scenario_id, training_seed, str(purpose))
        if identity in identities:
            raise ValueError(f"duplicate training-seed eval artifact: {identity}")
        identities.add(identity)
        used.add(eval_path.as_posix())
        available_algorithms.add(algorithm)
        aoi_values = summary.get("mean_AoI_ms_per_seed", [])
        success_values = summary.get("CAM_success_probability_per_seed", [])
        if not aoi_values or not success_values:
            raise ValueError(f"Fig.5 artifact has no per-eval-seed statistics: {eval_path}")
        aoi_value = summary.get("mean_AoI_ms", float(np.mean(aoi_values)))
        success_value = summary.get("CAM_success_probability", float(np.mean(success_values)))
        rows.setdefault(algorithm, {}).setdefault(float(scenario.get(x_field)), []).append(float(aoi_value))
        success_rows.setdefault(algorithm, {}).setdefault(float(scenario.get(x_field)), []).append(float(success_value))
    if observed_purposes and observed_purposes != {eval_purpose}:
        raise ValueError(f"Fig.5 mixed eval purposes: {sorted(observed_purposes)}")
    if not rows:
        raise FileNotFoundError("no complete frozen-eval artifacts for Fig.5")

    missing_algorithms = [algorithm for algorithm in required_algorithms if algorithm not in available_algorithms]
    current_expected_cells = [
        (CURRENT_ALGORITHM, scenario_id, seed)
        for scenario_id in expected_scenarios
        for seed in expected_training_seeds
    ]
    paper_expected_cells = [
        (algorithm, scenario_id, seed)
        for algorithm in required_algorithms
        for scenario_id in expected_scenarios
        for seed in expected_training_seeds
    ]
    observed_cells = {
        (algorithm, scenario_id, seed)
        for algorithm, scenario_id, seed, _purpose in identities
    }
    current_missing_cells = [
        {"algorithm": algorithm, "scenario": scenario_id, "training_seed": seed}
        for algorithm, scenario_id, seed in current_expected_cells
        if (algorithm, scenario_id, seed) not in observed_cells
    ]
    paper_missing_cells = [
        {"algorithm": algorithm, "scenario": scenario_id, "training_seed": seed}
        for algorithm, scenario_id, seed in paper_expected_cells
        if (algorithm, scenario_id, seed) not in observed_cells
    ]
    # ``missing_cells`` retains the validation-pilot meaning used by existing
    # consumers.  ``paper_missing_cells`` is the full four-algorithm study grid
    # and makes explicit why validation output can never be a paper-complete
    # Fig.5, even when every current-algorithm pilot cell is present.
    missing_cells = current_missing_cells if eval_purpose == "validation" else paper_missing_cells
    validation_current_grid_complete = not current_missing_cells if eval_purpose == "validation" else None
    paper_grid_complete = not paper_missing_cells
    partial = eval_purpose == "validation" or not paper_grid_complete
    if eval_purpose == "final_test" and not paper_grid_complete:
        raise RuntimeError("Fig.5 final_test grid is incomplete; see paper missing cells")
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
    figure.suptitle(f"Fig. 5 {'PARTIAL ' if partial else ''}- {eval_purpose} frozen evaluation sweep")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
    _write_sidecar(output, {
        "figure": "Fig.5",
        "x_field": x_field,
        "rows": table,
        "reused_eval_artifacts": sorted(used),
        "required_algorithms": list(required_algorithms),
        "eval_purpose": eval_purpose,
        "expected_training_seeds": list(expected_training_seeds),
        "scenario_grid": list(expected_scenarios),
        "available_algorithms": sorted(available_algorithms),
        "missing_algorithms": missing_algorithms,
        "status": "PARTIAL" if partial else "complete",
        "missing_cells": missing_cells,
        "current_missing_cells": current_missing_cells,
        "paper_missing_cells": paper_missing_cells,
        "validation_current_grid_complete": validation_current_grid_complete,
        "paper_grid_complete": paper_grid_complete,
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
    parser.add_argument("--eval-purpose", choices=("validation", "final_test"), default=None)
    parser.add_argument("--expected-training-seeds", default=None, help="comma-separated independent training seeds for Fig.5")
    parser.add_argument("--allow-incomplete-baselines", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.manifest_or_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (root.parent / "figures" if root.is_file() else root / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_training_seeds = None
    if args.expected_training_seeds is not None:
        expected_training_seeds = tuple(int(item.strip()) for item in args.expected_training_seeds.split(",") if item.strip())
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
        if args.eval_purpose is None:
            parser.error("--eval-purpose is required for Fig.5")
        results.append(str(plot_fig5(
            root,
            output_dir / "fig5_sweep.png",
            args.fig5_x,
            allow_incomplete=args.eval_purpose == "validation",
            eval_purpose=args.eval_purpose,
            expected_training_seeds=expected_training_seeds,
        )))
    print(json.dumps({"outputs": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
