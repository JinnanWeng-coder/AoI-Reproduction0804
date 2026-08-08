"""Build a canonical TDec result view and compact Algorithm 1 comparison bundle.

This is a post-processing utility.  It never imports the environment, resumes
training, evaluates checkpoints, or modifies source run directories.  The
canonical ``runs`` directories contain symlinks to the original remote runs;
all downloadable numerical evidence is materialized as CSV/JSON/PNG files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TDEC = "modified_maddpg_tdec"
MODIFIED = "modified_maddpg"
SEEDS = (8, 9, 10, 11, 12, 13)
GAPS = (5, 15, 25, 35)
SIZES = (4, 6, 8, 10)
EPISODES = 500
AGENTS = 5
ROLLING_WINDOW = 20
RASTER_DPI = 300
METHOD_COLORS = {TDEC: "#18864B", MODIFIED: "#7A3E9D"}
METHOD_LABELS = {TDEC: "Modified MADDPG with TDec", MODIFIED: "Modified MADDPG"}
METRIC_ROWS = (
    ("mean_aoi_ms", "Mean AoI (ms)", None),
    ("mean_binary_cam", "Strict binary CAM", (0.0, 1.02)),
    ("mean_payload_completion", "Payload completion", (0.0, 1.02)),
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


@dataclass(frozen=True)
class RunRecord:
    algorithm: str
    experiment: str
    source_role: str
    p: int
    n: int
    gap_m: int
    seed: int
    run_name: str
    source_dir: Path
    config: Mapping[str, object]

    @property
    def canonical_name(self) -> str:
        prefix = "tdec" if self.algorithm == TDEC else "modified"
        return f"{prefix}_p{self.p:02d}_n{self.n:02d}_g{self.gap_m:02d}_seed{self.seed:02d}"

    @property
    def condition(self) -> int:
        if self.experiment == "gap-extension":
            return self.gap_m
        if self.experiment == "platoon-size-extension":
            return self.n
        return 0


@dataclass(frozen=True)
class RunMetrics:
    aoi_agent: np.ndarray
    binary_agent: np.ndarray
    payload_agent: np.ndarray

    @property
    def episodes(self) -> int:
        return int(self.aoi_agent.shape[0])

    def episode_series(self) -> Dict[str, np.ndarray]:
        return {
            "mean_aoi_ms": self.aoi_agent.mean(axis=1),
            "worst_agent_aoi_ms": self.aoi_agent.max(axis=1),
            "mean_binary_cam": self.binary_agent.mean(axis=1),
            "worst_agent_binary_cam": self.binary_agent.min(axis=1),
            "mean_payload_completion": self.payload_agent.mean(axis=1),
            "worst_agent_payload_completion": self.payload_agent.min(axis=1),
        }


def _read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _algorithm_from_config(config: Mapping[str, object]) -> str:
    value = config.get("algorithm")
    return TDEC if value in (None, "", TDEC) else str(value)


def _config_value(config: Mapping[str, object], key: str, expected: object, label: str) -> None:
    actual = config.get(key)
    if isinstance(expected, float):
        try:
            matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            matches = False
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(f"{label}: {key}={actual!r}, expected {expected!r}")


def _validate_run(
    run_dir: Path,
    config: Mapping[str, object],
    algorithm: str,
    p: int,
    n: int,
    gap_m: int,
    seed: int,
) -> None:
    label = str(run_dir)
    for key, expected in (
        ("profile", "paper_faithful"),
        ("seed", seed),
        ("episodes", EPISODES),
        ("slow_update_every_episodes", 1),
        ("global_update_mode", "synchronous_joint"),
        ("tau", 0.005),
        ("exploration_noise", 0.3),
    ):
        _config_value(config, key, expected, label)
    if _algorithm_from_config(config) != algorithm:
        raise ValueError(
            f"{label}: algorithm={_algorithm_from_config(config)!r}, expected {algorithm!r}"
        )
    scenario = config.get("scenario")
    if not isinstance(scenario, dict):
        raise ValueError(f"{label}: scenario must be an object")
    for key, expected in (
        ("number_platoons", p),
        ("platoon_size", n),
        ("gap_m", float(gap_m)),
    ):
        actual = scenario.get(key)
        if actual != expected:
            raise ValueError(f"{label}: scenario.{key}={actual!r}, expected {expected!r}")
    complete = _read_json(run_dir / "COMPLETE.json")
    if complete.get("status") != "complete":
        raise ValueError(f"{label}: COMPLETE.json does not report status=complete")
    complete_algorithm = complete.get("algorithm")
    if complete_algorithm not in (None, "", algorithm):
        raise ValueError(
            f"{label}: COMPLETE algorithm={complete_algorithm!r}, expected {algorithm!r}"
        )
    if not (run_dir / "train_metrics.npz").is_file():
        raise FileNotFoundError(run_dir / "train_metrics.npz")


def _index_runs(root: Path) -> List[Tuple[Path, Dict[str, object]]]:
    root = root.expanduser().resolve()
    runs_root = root / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(runs_root)
    indexed: List[Tuple[Path, Dict[str, object]]] = []
    for config_path in sorted(runs_root.glob("*/config.resolved.json")):
        indexed.append((config_path.parent, _read_json(config_path)))
    if not indexed:
        raise ValueError(f"no run configs found under {runs_root}")
    return indexed


def _select_run(
    index: Sequence[Tuple[Path, Mapping[str, object]]],
    algorithm: str,
    p: int,
    n: int,
    gap_m: int,
    seed: int,
) -> Tuple[Path, Mapping[str, object]]:
    matches: List[Tuple[Path, Mapping[str, object]]] = []
    for run_dir, config in index:
        scenario = config.get("scenario")
        if not isinstance(scenario, dict):
            continue
        if (
            _algorithm_from_config(config) == algorithm
            and config.get("profile") == "paper_faithful"
            and config.get("seed") == seed
            and config.get("episodes") == EPISODES
            and config.get("slow_update_every_episodes") == 1
            and config.get("global_update_mode") == "synchronous_joint"
            and math.isclose(float(config.get("tau", math.nan)), 0.005, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                float(config.get("exploration_noise", math.nan)), 0.3, rel_tol=0.0, abs_tol=1e-12
            )
            and scenario.get("number_platoons") == p
            and scenario.get("platoon_size") == n
            and scenario.get("gap_m") == float(gap_m)
        ):
            matches.append((run_dir, config))
    if len(matches) != 1:
        names = [str(item[0]) for item in matches]
        raise ValueError(
            f"expected one {algorithm} P={p} N={n} gap={gap_m} seed={seed} run; "
            f"found {len(matches)}: {names}"
        )
    run_dir, config = matches[0]
    _validate_run(run_dir, config, algorithm, p, n, gap_m, seed)
    return run_dir, config


def _record(
    index: Sequence[Tuple[Path, Mapping[str, object]]],
    algorithm: str,
    experiment: str,
    source_role: str,
    p: int,
    n: int,
    gap_m: int,
    seed: int,
) -> RunRecord:
    run_dir, config = _select_run(index, algorithm, p, n, gap_m, seed)
    return RunRecord(
        algorithm=algorithm,
        experiment=experiment,
        source_role=source_role,
        p=p,
        n=n,
        gap_m=gap_m,
        seed=seed,
        run_name=run_dir.name,
        source_dir=run_dir,
        config=config,
    )


def build_catalogs(
    *,
    tdec_gap_root: Path,
    tdec_platoon_root: Path,
    modified_root: Path,
) -> Tuple[Dict[str, List[RunRecord]], Dict[str, List[RunRecord]]]:
    tdec_gap_index = _index_runs(tdec_gap_root)
    tdec_platoon_index = _index_runs(tdec_platoon_root)
    modified_root = modified_root.expanduser().resolve()
    modified_default_index = _index_runs(modified_root / "default" / "P5_N4_gap25")
    modified_gap_index = _index_runs(modified_root / "gap-extension")
    modified_platoon_index = _index_runs(modified_root / "platoon-size-extension")

    tdec_default = [
        _record(tdec_gap_index, TDEC, "default", "gap25_default", 5, 4, 25, seed)
        for seed in SEEDS
    ]
    tdec_gap = [
        _record(tdec_gap_index, TDEC, "gap-extension", "gap_phase_a", 5, 4, gap, seed)
        for gap in GAPS
        for seed in SEEDS
    ]
    tdec_platoon = [
        (
            _record(tdec_gap_index, TDEC, "platoon-size-extension", "default_reuse", 5, 4, 25, seed)
            if size == 4
            else _record(
                tdec_platoon_index,
                TDEC,
                "platoon-size-extension",
                "platoon_size_trend",
                5,
                size,
                25,
                seed,
            )
        )
        for size in SIZES
        for seed in SEEDS
    ]

    modified_default = [
        _record(
            modified_default_index,
            MODIFIED,
            "default",
            "default",
            5,
            4,
            25,
            seed,
        )
        for seed in SEEDS
    ]
    modified_gap = [
        (
            _record(
                modified_default_index,
                MODIFIED,
                "gap-extension",
                "default_reuse",
                5,
                4,
                25,
                seed,
            )
            if gap == 25
            else _record(
                modified_gap_index,
                MODIFIED,
                "gap-extension",
                "gap_extension",
                5,
                4,
                gap,
                seed,
            )
        )
        for gap in GAPS
        for seed in SEEDS
    ]
    modified_platoon = [
        (
            _record(
                modified_default_index,
                MODIFIED,
                "platoon-size-extension",
                "default_reuse",
                5,
                4,
                25,
                seed,
            )
            if size == 4
            else _record(
                modified_platoon_index,
                MODIFIED,
                "platoon-size-extension",
                "platoon_size_extension",
                5,
                size,
                25,
                seed,
            )
        )
        for size in SIZES
        for seed in SEEDS
    ]
    return (
        {"default": tdec_default, "gap-extension": tdec_gap, "platoon-size-extension": tdec_platoon},
        {
            "default": modified_default,
            "gap-extension": modified_gap,
            "platoon-size-extension": modified_platoon,
        },
    )


def _load_metrics(record: RunRecord) -> RunMetrics:
    with np.load(record.source_dir / "train_metrics.npz", allow_pickle=False) as data:
        aoi = np.asarray(data["mean_aoi_ms_episode_agent"], dtype=np.float64)
        binary = np.asarray(data["endpoint_cam_episode_agent"], dtype=np.float64)
        remaining = np.asarray(data["remaining_demand"], dtype=np.float64)
    if aoi.shape != (EPISODES, AGENTS):
        raise ValueError(f"{record.source_dir}: unexpected AoI shape {aoi.shape}")
    if binary.shape != (EPISODES, AGENTS):
        raise ValueError(f"{record.source_dir}: unexpected binary CAM shape {binary.shape}")
    if remaining.shape != (EPISODES, 100, AGENTS):
        raise ValueError(f"{record.source_dir}: unexpected remaining-demand shape {remaining.shape}")
    if not (np.isfinite(aoi).all() and np.isfinite(binary).all() and np.isfinite(remaining).all()):
        raise ValueError(f"{record.source_dir}: non-finite training metric")
    endpoint = remaining[:, -1, :]
    derived_binary = (endpoint <= 0.0).astype(np.float64)
    if not np.array_equal(binary, derived_binary):
        mismatch = int(np.count_nonzero(binary != derived_binary))
        raise ValueError(f"{record.source_dir}: binary CAM mismatch at {mismatch} entries")
    payload = np.clip(1.0 - endpoint / float(record.config["cam_bits"]), 0.0, 1.0)
    return RunMetrics(aoi_agent=aoi, binary_agent=binary, payload_agent=payload)


def _metrics_for_catalogs(
    catalogs: Iterable[Mapping[str, Sequence[RunRecord]]]
) -> Dict[Path, RunMetrics]:
    cache: Dict[Path, RunMetrics] = {}
    for catalog in catalogs:
        for records in catalog.values():
            for record in records:
                key = record.source_dir.resolve()
                if key not in cache:
                    cache[key] = _load_metrics(record)
    return cache


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _record_prefix(record: RunRecord) -> Dict[str, object]:
    return {
        "algorithm": record.algorithm,
        "experiment": record.experiment,
        "P": record.p,
        "N": record.n,
        "gap_m": record.gap_m,
        "seed": record.seed,
        "run_name": record.run_name,
        "source_role": record.source_role,
    }


def _window_row(record: RunRecord, metrics: RunMetrics, window: int) -> Dict[str, object]:
    aoi = metrics.aoi_agent[-window:]
    binary = metrics.binary_agent[-window:]
    payload = metrics.payload_agent[-window:]
    per_agent_aoi = aoi.mean(axis=0)
    per_agent_binary = binary.mean(axis=0)
    per_agent_payload = payload.mean(axis=0)
    row = _record_prefix(record)
    row.update(
        {
            "window": f"last{window}",
            "mean_aoi_ms": float(per_agent_aoi.mean()),
            "worst_agent_aoi_ms": float(per_agent_aoi.max()),
            "mean_binary_cam": float(per_agent_binary.mean()),
            "worst_agent_binary_cam": float(per_agent_binary.min()),
            "mean_payload_completion": float(per_agent_payload.mean()),
            "worst_agent_payload_completion": float(per_agent_payload.min()),
        }
    )
    row["screen_success"] = bool(
        row["worst_agent_aoi_ms"] < 50.0 and row["worst_agent_binary_cam"] >= 0.5
    )
    return row


def _rows_for_records(
    records: Sequence[RunRecord], metrics_cache: Mapping[Path, RunMetrics]
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    episode_rows: List[Dict[str, object]] = []
    agent_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for record in records:
        metrics = metrics_cache[record.source_dir.resolve()]
        series = metrics.episode_series()
        prefix = _record_prefix(record)
        for episode in range(metrics.episodes):
            row = dict(prefix)
            row["episode"] = episode + 1
            for key, values in series.items():
                row[key] = float(values[episode])
            episode_rows.append(row)
            for agent in range(AGENTS):
                agent_row = dict(prefix)
                agent_row.update(
                    {
                        "episode": episode + 1,
                        "agent": agent + 1,
                        "mean_aoi_ms": float(metrics.aoi_agent[episode, agent]),
                        "binary_cam": float(metrics.binary_agent[episode, agent]),
                        "payload_completion": float(metrics.payload_agent[episode, agent]),
                    }
                )
                agent_rows.append(agent_row)
        summary_rows.extend((_window_row(record, metrics, 100), _window_row(record, metrics, 50)))
    return episode_rows, agent_rows, summary_rows


def _cohort_rows(summary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    metric_names = (
        "mean_aoi_ms",
        "worst_agent_aoi_ms",
        "mean_binary_cam",
        "worst_agent_binary_cam",
        "mean_payload_completion",
        "worst_agent_payload_completion",
    )
    groups: MutableMapping[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in summary_rows:
        key = (
            row["algorithm"],
            row["experiment"],
            row["P"],
            row["N"],
            row["gap_m"],
            row["window"],
        )
        groups.setdefault(key, []).append(row)
    result: List[Dict[str, object]] = []
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        rows = groups[key]
        if len(rows) != len(SEEDS):
            raise ValueError(f"cohort {key}: expected {len(SEEDS)} seeds, found {len(rows)}")
        out: Dict[str, object] = {
            "algorithm": key[0],
            "experiment": key[1],
            "P": key[2],
            "N": key[3],
            "gap_m": key[4],
            "window": key[5],
            "seed_count": len(rows),
            "screen_success_count": int(sum(bool(row["screen_success"]) for row in rows)),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            out[metric] = float(values.mean())
            out[f"{metric}_seed_sd"] = float(values.std(ddof=1))
        result.append(out)
    return result


def _ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"existing symlink points elsewhere: {link} -> {link.resolve()}")
        return
    if link.exists():
        raise FileExistsError(f"refusing to replace non-symlink path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link, target_is_directory=True)


def _copy_metadata(record: RunRecord, metadata_root: Path) -> None:
    destination = metadata_root / record.canonical_name
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("config.resolved.json", "COMPLETE.json", "provenance.json", "train_metrics_summary.json"):
        source = record.source_dir / name
        if source.is_file():
            shutil.copy2(source, destination / name)


def _manifest_rows(catalog: Mapping[str, Sequence[RunRecord]], output_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for experiment, records in catalog.items():
        experiment_root = _experiment_root(output_root, experiment)
        for record in records:
            rows.append(
                {
                    **_record_prefix(record),
                    "canonical_name": record.canonical_name,
                    "source_path": str(record.source_dir.resolve()),
                    "canonical_link": str((experiment_root / "runs" / record.canonical_name).absolute()),
                    "episodes": record.config["episodes"],
                    "tau": record.config["tau"],
                    "slow_update_every_episodes": record.config["slow_update_every_episodes"],
                    "global_update_mode": record.config["global_update_mode"],
                    "training_noise": record.config["exploration_noise"],
                }
            )
    return rows


def _comparison_manifest_rows(
    tdec_catalog: Mapping[str, Sequence[RunRecord]],
    modified_catalog: Mapping[str, Sequence[RunRecord]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for catalog in (tdec_catalog, modified_catalog):
        for records in catalog.values():
            for record in records:
                rows.append(
                    {
                        **_record_prefix(record),
                        "source_path": str(record.source_dir.resolve()),
                        "episodes": record.config["episodes"],
                        "tau": record.config["tau"],
                        "slow_update_every_episodes": record.config[
                            "slow_update_every_episodes"
                        ],
                        "global_update_mode": record.config["global_update_mode"],
                        "training_noise": record.config["exploration_noise"],
                    }
                )
    return rows


def _experiment_root(root: Path, experiment: str) -> Path:
    if experiment == "default":
        return root / "default" / "P5_N4_gap25"
    return root / experiment


def _condition_values(experiment: str) -> Tuple[int, ...]:
    if experiment == "gap-extension":
        return GAPS
    if experiment == "platoon-size-extension":
        return SIZES
    return (0,)


def _condition_title(experiment: str, value: int) -> str:
    if experiment == "gap-extension":
        return f"gap = {value} m"
    if experiment == "platoon-size-extension":
        return f"N = {value}"
    return "P=5, N=4, gap=25 m"


def _rolling_mean(values: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty_like(values)
    for index in range(values.size):
        left = max(0, index + 1 - window)
        result[index] = (cumulative[index + 1] - cumulative[left]) / (index + 1 - left)
    return result


def _save_png(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_seed_grid(
    records: Sequence[RunRecord],
    metrics_cache: Mapping[Path, RunMetrics],
    output: Path,
    dpi: int,
) -> None:
    experiment = records[0].experiment
    conditions = _condition_values(experiment)
    figure, axes = plt.subplots(
        len(METRIC_ROWS),
        len(conditions),
        figsize=(3.2 * len(conditions), 7.1),
        sharex=True,
        squeeze=False,
    )
    seed_colors = dict(zip(SEEDS, plt.get_cmap("tab10").colors[: len(SEEDS)]))
    for column, condition in enumerate(conditions):
        condition_records = [record for record in records if record.condition == condition]
        condition_records.sort(key=lambda record: record.seed)
        if [record.seed for record in condition_records] != list(SEEDS):
            raise ValueError(f"{experiment} condition {condition}: incomplete seed set")
        for row_index, (metric_name, ylabel, ylim) in enumerate(METRIC_ROWS):
            axis = axes[row_index, column]
            stack: List[np.ndarray] = []
            for record in condition_records:
                values = metrics_cache[record.source_dir.resolve()].episode_series()[metric_name]
                episodes = np.arange(1, values.size + 1)
                color = seed_colors[record.seed]
                axis.plot(episodes, values, color=color, linewidth=0.35, alpha=0.10)
                axis.plot(
                    episodes,
                    _rolling_mean(values),
                    color=color,
                    linewidth=0.9,
                    alpha=0.85,
                    label=f"seed {record.seed}" if row_index == 0 and column == 0 else None,
                )
                stack.append(values)
            cohort = np.mean(np.stack(stack), axis=0)
            axis.plot(
                episodes,
                _rolling_mean(cohort),
                color="#202020",
                linewidth=1.7,
                label="cohort mean" if row_index == 0 and column == 0 else None,
            )
            if row_index == 0:
                axis.set_title(_condition_title(experiment, condition), fontsize=9)
            if column == 0:
                axis.set_ylabel(ylabel)
            if row_index == len(METRIC_ROWS) - 1:
                axis.set_xlabel("Training episode")
            if ylim is not None:
                axis.set_ylim(*ylim)
            axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.65)
    title = f"{METHOD_LABELS[records[0].algorithm]} — {experiment} seed trajectories"
    figure.suptitle(title, fontsize=11, y=0.995)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 0.965), fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 0.93), pad=1.0)
    _save_png(figure, output, dpi)


def _metric_stack(
    records: Sequence[RunRecord],
    metrics_cache: Mapping[Path, RunMetrics],
    metric_name: str,
) -> np.ndarray:
    ordered = sorted(records, key=lambda record: record.seed)
    return np.stack(
        [metrics_cache[record.source_dir.resolve()].episode_series()[metric_name] for record in ordered]
    )


def _plot_default_comparison(
    tdec_records: Sequence[RunRecord],
    modified_records: Sequence[RunRecord],
    metrics_cache: Mapping[Path, RunMetrics],
    output: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.05), sharex=True)
    episodes = np.arange(1, EPISODES + 1)
    for axis, (metric_name, ylabel, ylim) in zip(axes, METRIC_ROWS):
        for algorithm, records in ((TDEC, tdec_records), (MODIFIED, modified_records)):
            stack = _metric_stack(records, metrics_cache, metric_name)
            smoothed = np.stack([_rolling_mean(values) for values in stack])
            mean = np.mean(smoothed, axis=0)
            sd = np.std(smoothed, axis=0, ddof=1)
            color = METHOD_COLORS[algorithm]
            axis.fill_between(episodes, mean - sd, mean + sd, color=color, alpha=0.13)
            axis.plot(episodes, mean, color=color, linewidth=1.7, label=METHOD_LABELS[algorithm])
        axis.set_xlabel("Training episode")
        axis.set_ylabel(ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.65)
    axes[0].legend(fontsize=7)
    figure.suptitle("Default configuration convergence (mean ± seed SD, 20-episode rolling mean)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.92), pad=1.0)
    _save_png(figure, output, dpi)


def _last100_value(record: RunRecord, metrics: RunMetrics, metric: str) -> float:
    if metric == "mean_aoi_ms":
        return float(metrics.aoi_agent[-100:].mean())
    if metric == "mean_binary_cam":
        return float(metrics.binary_agent[-100:].mean())
    if metric == "mean_payload_completion":
        return float(metrics.payload_agent[-100:].mean())
    raise KeyError(metric)


def _plot_trend_comparison(
    experiment: str,
    tdec_records: Sequence[RunRecord],
    modified_records: Sequence[RunRecord],
    metrics_cache: Mapping[Path, RunMetrics],
    output: Path,
    dpi: int,
) -> None:
    conditions = _condition_values(experiment)
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.15))
    offsets = {TDEC: -0.10, MODIFIED: 0.10}
    for axis, (metric_name, ylabel, ylim) in zip(axes, METRIC_ROWS):
        for algorithm, records in ((TDEC, tdec_records), (MODIFIED, modified_records)):
            means: List[float] = []
            sds: List[float] = []
            for condition in conditions:
                condition_records = sorted(
                    [record for record in records if record.condition == condition],
                    key=lambda record: record.seed,
                )
                values = np.asarray(
                    [
                        _last100_value(record, metrics_cache[record.source_dir.resolve()], metric_name)
                        for record in condition_records
                    ]
                )
                means.append(float(values.mean()))
                sds.append(float(values.std(ddof=1)))
                jitter_x = np.full(values.shape, float(condition) + offsets[algorithm])
                axis.scatter(jitter_x, values, s=12, color=METHOD_COLORS[algorithm], alpha=0.38, linewidths=0)
            axis.errorbar(
                conditions,
                means,
                yerr=sds,
                color=METHOD_COLORS[algorithm],
                marker="o",
                markersize=4,
                linewidth=1.6,
                capsize=3,
                label=METHOD_LABELS[algorithm],
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(conditions)
        axis.set_xlabel("Intra-platoon gap (m)" if experiment == "gap-extension" else "Platoon size (N)")
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.65)
    axes[0].legend(fontsize=7)
    figure.suptitle("Train-last100: individual seeds and mean ± seed SD", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.92), pad=1.0)
    _save_png(figure, output, dpi)


def _plot_paired_aoi(
    tdec_catalog: Mapping[str, Sequence[RunRecord]],
    modified_catalog: Mapping[str, Sequence[RunRecord]],
    metrics_cache: Mapping[Path, RunMetrics],
    output: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    seed_colors = dict(zip(SEEDS, plt.get_cmap("tab10").colors[: len(SEEDS)]))
    for axis, experiment in zip(axes, ("gap-extension", "platoon-size-extension")):
        conditions = _condition_values(experiment)
        all_deltas: List[np.ndarray] = []
        for seed in SEEDS:
            deltas: List[float] = []
            for condition in conditions:
                tdec_record = next(
                    record
                    for record in tdec_catalog[experiment]
                    if record.seed == seed and record.condition == condition
                )
                modified_record = next(
                    record
                    for record in modified_catalog[experiment]
                    if record.seed == seed and record.condition == condition
                )
                tdec_value = _last100_value(
                    tdec_record, metrics_cache[tdec_record.source_dir.resolve()], "mean_aoi_ms"
                )
                modified_value = _last100_value(
                    modified_record,
                    metrics_cache[modified_record.source_dir.resolve()],
                    "mean_aoi_ms",
                )
                deltas.append(modified_value - tdec_value)
            delta_array = np.asarray(deltas)
            all_deltas.append(delta_array)
            axis.plot(
                conditions,
                delta_array,
                marker="o",
                markersize=3,
                linewidth=0.9,
                color=seed_colors[seed],
                alpha=0.75,
                label=f"seed {seed}" if experiment == "gap-extension" else None,
            )
        axis.plot(
            conditions,
            np.mean(np.stack(all_deltas), axis=0),
            marker="o",
            color="#202020",
            linewidth=2.0,
            label="cohort mean" if experiment == "gap-extension" else None,
        )
        axis.axhline(0.0, color="#767676", linestyle="--", linewidth=0.9)
        axis.set_xticks(conditions)
        axis.set_xlabel("Gap (m)" if experiment == "gap-extension" else "Platoon size (N)")
        axis.set_ylabel("AoI difference: Modified − TDec (ms)")
        axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.65)
    axes[0].legend(fontsize=6, ncol=2)
    figure.suptitle("Paired train-last100 AoI differences", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.92), pad=1.0)
    _save_png(figure, output, dpi)


def _write_experiment_data(
    root: Path,
    records: Sequence[RunRecord],
    metrics_cache: Mapping[Path, RunMetrics],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    episode_rows, agent_rows, summary_rows = _rows_for_records(records, metrics_cache)
    cohort_rows = _cohort_rows(summary_rows)
    plot_data = root / "plot_data"
    _write_csv(plot_data / "per_episode.csv", episode_rows)
    _write_csv(plot_data / "per_episode_agent.csv", agent_rows)
    _write_csv(plot_data / "per_seed_summary.csv", summary_rows)
    _write_csv(plot_data / "cohort_summary.csv", cohort_rows)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    summary = {
        "algorithm": records[0].algorithm,
        "experiment": records[0].experiment,
        "protocol": {
            "training_episodes": EPISODES,
            "seeds": list(SEEDS),
            "primary_window": "last100",
            "secondary_window": "last50",
            "rolling_window_for_figures_only": ROLLING_WINDOW,
            "strict_binary_and_payload_reported_separately": True,
        },
        "logical_run_count": len(records),
        "unique_source_run_count": len({record.source_dir.resolve() for record in records}),
        "cohort_summary": cohort_rows,
    }
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return episode_rows, agent_rows, summary_rows


def _write_tdec_readme(root: Path, catalog: Mapping[str, Sequence[RunRecord]]) -> None:
    unique = {record.source_dir.resolve() for records in catalog.values() for record in records}
    text = f"""# Modified MADDPG with task decomposition results

Canonical post-processing view for Algorithm 2. Original remote runs are not moved or copied.
The `runs/` entries are symlinks; `plot_data/`, `analysis/`, metadata, and PNG figures are self-contained.

- configuration: paper_faithful, tau=0.005, slow=1, synchronous_joint, 500 episodes, noise=0.3
- training seeds: 8--13
- unique source runs: {len(unique)}
- logical default cells: {len(catalog['default'])}
- logical gap-extension cells: {len(catalog['gap-extension'])}
- logical platoon-size-extension cells: {len(catalog['platoon-size-extension'])}
- primary paper-facing window: train-last100
- strict binary CAM and continuous payload completion are never substituted for one another

The PNG curves show raw episode values faintly and a 20-episode rolling mean for readability.
CSV files always contain the unmodified per-episode values.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def _write_comparison_report(
    root: Path,
    tdec_catalog: Mapping[str, Sequence[RunRecord]],
    modified_catalog: Mapping[str, Sequence[RunRecord]],
    metrics_cache: Mapping[Path, RunMetrics],
) -> None:
    lines = [
        "# Modified MADDPG versus Modified MADDPG with TDec",
        "",
        "Descriptive comparison using matched training seeds 8--13 and train-last100.",
        "Seed is the independent unit; gap/N cells must not be treated as independent replicates.",
        "Strict binary CAM and continuous payload completion are reported separately.",
        "",
    ]
    for experiment in ("gap-extension", "platoon-size-extension"):
        condition_label = "gap (m)" if experiment == "gap-extension" else "N"
        lines.extend(
            [
                f"## {experiment}",
                "",
                f"| {condition_label} | TDec AoI | Modified AoI | delta | TDec binary | Modified binary | TDec payload | Modified payload |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition in _condition_values(experiment):
            values: Dict[str, Dict[str, float]] = {}
            for algorithm, catalog in ((TDEC, tdec_catalog), (MODIFIED, modified_catalog)):
                records = [record for record in catalog[experiment] if record.condition == condition]
                values[algorithm] = {
                    metric: float(
                        np.mean(
                            [
                                _last100_value(
                                    record, metrics_cache[record.source_dir.resolve()], metric
                                )
                                for record in records
                            ]
                        )
                    )
                    for metric, _label, _ylim in METRIC_ROWS
                }
            lines.append(
                f"| {condition} | {values[TDEC]['mean_aoi_ms']:.3f} | "
                f"{values[MODIFIED]['mean_aoi_ms']:.3f} | "
                f"{values[MODIFIED]['mean_aoi_ms'] - values[TDEC]['mean_aoi_ms']:+.3f} | "
                f"{values[TDEC]['mean_binary_cam']:.4f} | {values[MODIFIED]['mean_binary_cam']:.4f} | "
                f"{values[TDEC]['mean_payload_completion']:.4f} | "
                f"{values[MODIFIED]['mean_payload_completion']:.4f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Scope",
            "",
            "These are training-window figures, not held-out evaluation, formal matrix, or final_test results.",
            "The experiment uses tau=0.005 from the public code; Table II states 0.0005.",
        ]
    )
    report_root = root / "report"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    readme = """# Modified MADDPG versus TDec plot bundle

This directory is a self-contained, training-only comparison of Algorithm 1
and Algorithm 2 using seeds 8--13. `plot_data/` contains every unmodified
episode value needed to rebuild the figures; `figures/` contains PNG-only
renders; `report/` records the build counts and descriptive tables.

Seed-trajectory figures show faint raw values and a 20-episode rolling mean.
The rolling operation is visual only. Fig. 5 trend panels use train-last100,
show every seed, and report the cohort mean with seed SD. Strict binary CAM
and continuous payload completion are separate metrics.

The download bundle excludes checkpoints, replay buffers, full NPZ files, and
remote run links. It is not a held-out evaluation, formal matrix, or final_test
release.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def prepare(
    *,
    tdec_gap_root: Path,
    tdec_platoon_root: Path,
    modified_root: Path,
    tdec_output_root: Path,
    comparison_output_root: Path,
    create_links: bool = True,
    dpi: int = RASTER_DPI,
) -> Dict[str, object]:
    tdec_gap_root = tdec_gap_root.expanduser().resolve()
    tdec_platoon_root = tdec_platoon_root.expanduser().resolve()
    modified_root = modified_root.expanduser().resolve()
    tdec_output_root = tdec_output_root.expanduser().resolve()
    comparison_output_root = comparison_output_root.expanduser().resolve()
    for output in (tdec_output_root, comparison_output_root):
        for source in (tdec_gap_root, tdec_platoon_root, modified_root):
            try:
                output.relative_to(source)
            except ValueError:
                continue
            raise ValueError(f"output root must not be inside a source result root: {output}")
    tdec_catalog, modified_catalog = build_catalogs(
        tdec_gap_root=tdec_gap_root,
        tdec_platoon_root=tdec_platoon_root,
        modified_root=modified_root,
    )
    metrics_cache = _metrics_for_catalogs((tdec_catalog, modified_catalog))

    tdec_output_root.mkdir(parents=True, exist_ok=True)
    comparison_output_root.mkdir(parents=True, exist_ok=True)
    _write_tdec_readme(tdec_output_root, tdec_catalog)
    manifest = _manifest_rows(tdec_catalog, tdec_output_root)
    _write_csv(tdec_output_root / "run_manifest.csv", manifest)
    _write_csv(
        comparison_output_root / "run_manifest.csv",
        _comparison_manifest_rows(tdec_catalog, modified_catalog),
    )

    combined_episode: List[Dict[str, object]] = []
    combined_agent: List[Dict[str, object]] = []
    combined_summary: List[Dict[str, object]] = []

    for experiment, records in tdec_catalog.items():
        experiment_root = _experiment_root(tdec_output_root, experiment)
        experiment_root.mkdir(parents=True, exist_ok=True)
        for record in records:
            if create_links:
                _ensure_symlink(experiment_root / "runs" / record.canonical_name, record.source_dir)
            _copy_metadata(record, experiment_root / "metadata")
        episode_rows, agent_rows, summary_rows = _write_experiment_data(
            experiment_root, records, metrics_cache
        )
        combined_episode.extend(episode_rows)
        combined_agent.extend(agent_rows)
        combined_summary.extend(summary_rows)
        _plot_seed_grid(
            records,
            metrics_cache,
            experiment_root / "figures" / "seed_trajectories.png",
            dpi,
        )

    for catalog in (modified_catalog,):
        for records in catalog.values():
            episode_rows, agent_rows, summary_rows = _rows_for_records(records, metrics_cache)
            combined_episode.extend(episode_rows)
            combined_agent.extend(agent_rows)
            combined_summary.extend(summary_rows)

    comparison_data = comparison_output_root / "plot_data"
    _write_csv(comparison_data / "combined_per_episode.csv", combined_episode)
    _write_csv(comparison_data / "combined_per_episode_agent.csv", combined_agent)
    _write_csv(comparison_data / "combined_per_seed_summary.csv", combined_summary)
    _write_csv(comparison_data / "combined_cohort_summary.csv", _cohort_rows(combined_summary))

    paired_rows: List[Dict[str, object]] = []
    for experiment in ("gap-extension", "platoon-size-extension"):
        for condition in _condition_values(experiment):
            for seed in SEEDS:
                tdec_record = next(
                    record
                    for record in tdec_catalog[experiment]
                    if record.seed == seed and record.condition == condition
                )
                modified_record = next(
                    record
                    for record in modified_catalog[experiment]
                    if record.seed == seed and record.condition == condition
                )
                tdec_metrics = metrics_cache[tdec_record.source_dir.resolve()]
                modified_metrics = metrics_cache[modified_record.source_dir.resolve()]
                row: Dict[str, object] = {
                    "experiment": experiment,
                    "condition": condition,
                    "seed": seed,
                    "N": modified_record.n,
                    "gap_m": modified_record.gap_m,
                }
                for metric, _label, _ylim in METRIC_ROWS:
                    tdec_value = _last100_value(tdec_record, tdec_metrics, metric)
                    modified_value = _last100_value(modified_record, modified_metrics, metric)
                    row[f"tdec_{metric}"] = tdec_value
                    row[f"modified_{metric}"] = modified_value
                    row[f"delta_{metric}"] = modified_value - tdec_value
                paired_rows.append(row)
    _write_csv(comparison_data / "paired_differences.csv", paired_rows)

    comparison_figures = comparison_output_root / "figures"
    for algorithm, catalog in ((TDEC, tdec_catalog), (MODIFIED, modified_catalog)):
        for experiment, records in catalog.items():
            prefix = "tdec" if algorithm == TDEC else "modified"
            _plot_seed_grid(
                records,
                metrics_cache,
                comparison_figures / f"{prefix}_{experiment}_seed_trajectories.png",
                dpi,
            )
    _plot_default_comparison(
        tdec_catalog["default"],
        modified_catalog["default"],
        metrics_cache,
        comparison_figures / "default_convergence_comparison.png",
        dpi,
    )
    _plot_trend_comparison(
        "gap-extension",
        tdec_catalog["gap-extension"],
        modified_catalog["gap-extension"],
        metrics_cache,
        comparison_figures / "fig5_gap_comparison.png",
        dpi,
    )
    _plot_trend_comparison(
        "platoon-size-extension",
        tdec_catalog["platoon-size-extension"],
        modified_catalog["platoon-size-extension"],
        metrics_cache,
        comparison_figures / "fig5_platoon_size_comparison.png",
        dpi,
    )
    _plot_paired_aoi(
        tdec_catalog,
        modified_catalog,
        metrics_cache,
        comparison_figures / "paired_aoi_differences.png",
        dpi,
    )
    _write_comparison_report(
        comparison_output_root, tdec_catalog, modified_catalog, metrics_cache
    )

    unique_tdec = {record.source_dir.resolve() for records in tdec_catalog.values() for record in records}
    unique_modified = {
        record.source_dir.resolve() for records in modified_catalog.values() for record in records
    }
    result = {
        "status": "PASS",
        "tdec_unique_runs": len(unique_tdec),
        "modified_unique_runs": len(unique_modified),
        "tdec_logical_cells": {key: len(value) for key, value in tdec_catalog.items()},
        "modified_logical_cells": {key: len(value) for key, value in modified_catalog.items()},
        "combined_per_episode_rows": len(combined_episode),
        "combined_per_episode_agent_rows": len(combined_agent),
        "comparison_png_count": len(list(comparison_figures.glob("*.png"))),
        "tdec_output_root": str(tdec_output_root),
        "comparison_output_root": str(comparison_output_root),
        "run_links_created": create_links,
    }
    (comparison_output_root / "report" / "build_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create canonical TDec results, compact plot data, and PNG Algorithm 1 comparison figures."
    )
    parser.add_argument("--tdec-gap-root", required=True, type=Path)
    parser.add_argument("--tdec-platoon-root", required=True, type=Path)
    parser.add_argument("--modified-root", required=True, type=Path)
    parser.add_argument("--tdec-output-root", required=True, type=Path)
    parser.add_argument("--comparison-output-root", required=True, type=Path)
    parser.add_argument("--no-run-links", action="store_true")
    parser.add_argument("--dpi", type=int, default=RASTER_DPI)
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    result = prepare(
        tdec_gap_root=args.tdec_gap_root,
        tdec_platoon_root=args.tdec_platoon_root,
        modified_root=args.modified_root,
        tdec_output_root=args.tdec_output_root,
        comparison_output_root=args.comparison_output_root,
        create_links=not args.no_run_links,
        dpi=args.dpi,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
