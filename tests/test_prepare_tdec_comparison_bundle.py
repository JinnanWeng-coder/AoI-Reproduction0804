import csv
import json
from pathlib import Path

import numpy as np

from analysis.prepare_tdec_comparison_bundle import (
    GAPS,
    MODIFIED,
    SEEDS,
    SIZES,
    TDEC,
    prepare,
)


def _write_run(
    root: Path,
    *,
    algorithm: str,
    n: int,
    gap: int,
    seed: int,
    run_name: str,
) -> None:
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True)
    config = {
        "profile": "paper_faithful",
        "seed": seed,
        "episodes": 500,
        "slow_update_every_episodes": 1,
        "global_update_mode": "synchronous_joint",
        "tau": 0.005,
        "exploration_noise": 0.3,
        "cam_bits": 32000,
        "scenario": {
            "id": f"p05_n{n:02d}_g{gap:02d}",
            "number_platoons": 5,
            "platoon_size": n,
            "gap_m": float(gap),
        },
    }
    if algorithm == MODIFIED:
        config["algorithm"] = MODIFIED
    (run_dir / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    complete = {"status": "complete", "episodes": 500}
    if algorithm == MODIFIED:
        complete["algorithm"] = MODIFIED
    (run_dir / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    (run_dir / "train_metrics_summary.json").write_text(
        json.dumps({"status": "complete", "episodes": 500}), encoding="utf-8"
    )

    episode = np.arange(500, dtype=np.float32)[:, None]
    agent = np.arange(5, dtype=np.float32)[None, :]
    algorithm_penalty = 0.7 if algorithm == MODIFIED else 0.0
    load = 0.04 * gap + 0.25 * n
    aoi = (
        18.0 * np.exp(-episode / 65.0)
        + load
        + algorithm_penalty
        + (seed - 8) * 0.04
        + agent * 0.08
    ).astype(np.float32)
    fail_period = max(3, 20 - n)
    binary = np.ones((500, 5), dtype=np.float32)
    binary[(np.arange(500)[:, None] + np.arange(5)[None, :] + seed) % fail_period == 0] = 0.0
    endpoint = np.where(binary > 0.5, 0.0, 640.0 + 10.0 * n).astype(np.float32)
    remaining = np.zeros((500, 100, 5), dtype=np.float32)
    remaining[:, -1, :] = endpoint
    np.savez_compressed(
        run_dir / "train_metrics.npz",
        mean_aoi_ms_episode_agent=aoi,
        endpoint_cam_episode_agent=binary,
        remaining_demand=remaining,
    )


def _build_fixture(tmp_path: Path):
    tdec_gap = tmp_path / "source" / "gap-global-slow-42-v1"
    tdec_platoon = tmp_path / "source" / "platoon-size-trend-v1"
    modified = tmp_path / "source" / "Modified_MADDPG_results"
    modified_default = modified / "default" / "P5_N4_gap25"
    modified_gap = modified / "gap-extension"
    modified_platoon = modified / "platoon-size-extension"

    for gap in GAPS:
        for seed in SEEDS:
            _write_run(
                tdec_gap,
                algorithm=TDEC,
                n=4,
                gap=gap,
                seed=seed,
                run_name=f"gap_global_slow_sync_slow01_p05_n04_g{gap:02d}_seed{seed:02d}",
            )
    for size in SIZES[1:]:
        for seed in SEEDS:
            _write_run(
                tdec_platoon,
                algorithm=TDEC,
                n=size,
                gap=25,
                seed=seed,
                run_name=f"platoon_size_trend_p05_n{size:02d}_g25_seed{seed:02d}",
            )

    for seed in SEEDS:
        _write_run(
            modified_default,
            algorithm=MODIFIED,
            n=4,
            gap=25,
            seed=seed,
            run_name=f"modified_maddpg_default_p05_n04_g25_seed{seed:02d}",
        )
    for gap in (5, 15, 35):
        for seed in SEEDS:
            _write_run(
                modified_gap,
                algorithm=MODIFIED,
                n=4,
                gap=gap,
                seed=seed,
                run_name=f"modified_maddpg_gap_p05_n04_g{gap:02d}_seed{seed:02d}",
            )
    for size in SIZES[1:]:
        for seed in SEEDS:
            _write_run(
                modified_platoon,
                algorithm=MODIFIED,
                n=size,
                gap=25,
                seed=seed,
                run_name=f"modified_maddpg_platoon_p05_n{size:02d}_g25_seed{seed:02d}",
            )
    return tdec_gap, tdec_platoon, modified


def _csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_prepare_builds_canonical_data_and_png_only_figures(tmp_path):
    tdec_gap, tdec_platoon, modified = _build_fixture(tmp_path)
    tdec_output = tmp_path / "output" / "Modified_MADDPG_with_TDec_results"
    comparison = tmp_path / "output" / "algorithm-comparison" / "Modified_MADDPG_vs_TDec"

    result = prepare(
        tdec_gap_root=tdec_gap,
        tdec_platoon_root=tdec_platoon,
        modified_root=modified,
        tdec_output_root=tdec_output,
        comparison_output_root=comparison,
        create_links=False,
        dpi=72,
    )

    assert result["status"] == "PASS"
    assert result["tdec_unique_runs"] == 42
    assert result["modified_unique_runs"] == 42
    assert result["tdec_logical_cells"] == {
        "default": 6,
        "gap-extension": 24,
        "platoon-size-extension": 24,
    }
    assert result["combined_per_episode_rows"] == 54_000
    assert result["combined_per_episode_agent_rows"] == 270_000
    assert result["comparison_png_count"] == 10

    manifest = _csv_rows(tdec_output / "run_manifest.csv")
    assert len(manifest) == 54
    assert len({row["source_path"] for row in manifest}) == 42
    assert {row["algorithm"] for row in manifest} == {TDEC}
    comparison_manifest = _csv_rows(comparison / "run_manifest.csv")
    assert len(comparison_manifest) == 108
    assert len({row["source_path"] for row in comparison_manifest}) == 84

    default_episode = _csv_rows(
        tdec_output / "default" / "P5_N4_gap25" / "plot_data" / "per_episode.csv"
    )
    assert len(default_episode) == 3_000
    assert {row["seed"] for row in default_episode} == {str(seed) for seed in SEEDS}
    default_agents = _csv_rows(
        tdec_output
        / "default"
        / "P5_N4_gap25"
        / "plot_data"
        / "per_episode_agent.csv"
    )
    assert len(default_agents) == 15_000
    assert {row["agent"] for row in default_agents} == {"1", "2", "3", "4", "5"}

    paired = _csv_rows(comparison / "plot_data" / "paired_differences.csv")
    assert len(paired) == 48
    assert all(float(row["delta_mean_aoi_ms"]) > 0.0 for row in paired)

    pngs = list((tmp_path / "output").rglob("*.png"))
    assert len(pngs) == 13
    assert all(path.stat().st_size > 1000 for path in pngs)
    assert not list((tmp_path / "output").rglob("*.svg"))
    assert not list((tmp_path / "output").rglob("*.pdf"))
    assert not list((tmp_path / "output").rglob("*.pt"))
    assert not list((tmp_path / "output").rglob("*replay*"))
