"""Pure post-processing audit of service regularity in shared-actor-v1 NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.evaluate_mappo_feasibility_reward import feasibility_eval_id
from analysis.evaluate_mappo_zero_shot import DEFAULT_EVAL_SEEDS, TRAINING_SEEDS
from analysis.summarize_mappo_shared_actor import training_run_name


def interval_statistics(reset: np.ndarray) -> dict:
    """Compute intervals on one continuous world stream; AoI values are not used."""
    reset = np.asarray(reset, dtype=bool).reshape(-1)
    positions = np.flatnonzero(reset)
    intervals = np.diff(positions).astype(np.int64)
    count = int(intervals.size)
    total = int(intervals.sum()) if count else 0
    total2 = int(np.square(intervals, dtype=np.int64).sum()) if count else 0
    mean = float(total / count) if count else None
    cv2 = float((total2 / count) / (mean * mean) - 1.0) if count and mean else None
    left = int(positions[0]) if positions.size else int(reset.size)
    right = int(reset.size - 1 - positions[-1]) if positions.size else int(reset.size)
    long = intervals[intervals > 100]
    return {
        "complete_interval_count": count, "complete_interval_sum_L": total,
        "complete_interval_sum_L2": total2, "complete_interval_mean": mean,
        "complete_interval_cv2": cv2, "complete_long_gt100_count": int(long.size),
        "complete_long_gt100_sum_L": int(long.sum()) if long.size else 0,
        "left_censored_slots": left, "right_censored_slots": right,
        "no_reset_stream": bool(positions.size == 0),
    }


def audit(shared_root: Path) -> dict:
    shared_root = shared_root.resolve()
    rows = []
    eval_id = feasibility_eval_id("baseline", DEFAULT_EVAL_SEEDS, 100, 5)
    for structure in ("independent", "shared"):
        for seed in TRAINING_SEEDS:
            run = training_run_name(structure, seed)
            eval_dir = shared_root / "evaluations" / run / eval_id
            with np.load(eval_dir / "metrics.npz", allow_pickle=False) as loaded:
                required = {"mode", "reset_event", "executed_power_dbm"}
                if not required.issubset(loaded.files):
                    raise ValueError(f"{eval_dir}: missing {sorted(required - set(loaded.files))}")
                mode, reset, power = loaded["mode"], loaded["reset_event"], loaded["executed_power_dbm"]
            if mode.shape != (6, 100, 100, 5) or reset.shape != mode.shape or power.shape != mode.shape:
                raise ValueError(f"{eval_dir}: unexpected shared-actor trajectory shape")
            for wi, world in enumerate(DEFAULT_EVAL_SEEDS):
                for agent in range(5):
                    m = mode[wi, :, :, agent].reshape(-1)
                    r = reset[wi, :, :, agent].astype(bool).reshape(-1)
                    p = np.power(10.0, power[wi, :, :, agent].reshape(-1) / 10.0)
                    attempts = m == 0
                    attempt_count, reset_count = int(attempts.sum()), int(r.sum())
                    rows.append({
                        "structure": structure, "training_seed": seed, "eval_seed": world, "agent": agent,
                        "slot_count": int(m.size), "v2i_attempt_count": attempt_count, "reset_count": reset_count,
                        "rho": float(attempt_count / m.size), "q": float(reset_count / attempt_count) if attempt_count else None,
                        "zero_attempt_denominator": bool(attempt_count == 0), "reset_rate": float(reset_count / m.size),
                        "mean_power_v2i_mw": float(p[attempts].mean()) if attempts.any() else None,
                        "mean_power_v2v_mw": float(p[~attempts].mean()) if (~attempts).any() else None,
                        **interval_statistics(r),
                    })
    summaries = []
    for structure in ("independent", "shared"):
        selected = [row for row in rows if row["structure"] == structure]
        count = sum(row["complete_interval_count"] for row in selected)
        sum_l = sum(row["complete_interval_sum_L"] for row in selected)
        sum_l2 = sum(row["complete_interval_sum_L2"] for row in selected)
        mean = sum_l / count if count else None
        pooled_cv2 = (sum_l2 / count) / (mean * mean) - 1.0 if count and mean else None
        flow_means = [row["complete_interval_mean"] for row in selected if row["complete_interval_mean"] is not None]
        flow_cv2 = [row["complete_interval_cv2"] for row in selected if row["complete_interval_cv2"] is not None]
        attempts, resets, slots = sum(r["v2i_attempt_count"] for r in selected), sum(r["reset_count"] for r in selected), sum(r["slot_count"] for r in selected)
        v2v_count = slots - attempts
        mean_v2i_power = (
            sum(r["mean_power_v2i_mw"] * r["v2i_attempt_count"] for r in selected if r["mean_power_v2i_mw"] is not None) / attempts
            if attempts else None
        )
        mean_v2v_power = (
            sum(r["mean_power_v2v_mw"] * (r["slot_count"] - r["v2i_attempt_count"]) for r in selected if r["mean_power_v2v_mw"] is not None) / v2v_count
            if v2v_count else None
        )
        summaries.append({
            "structure": structure, "flow_count": len(selected), "weighting": "slot/count weighted pooled plus equal-flow diagnostics",
            "pooled_rho": attempts / slots, "pooled_q": resets / attempts if attempts else None,
            "pooled_reset_rate": resets / slots, "pooled_complete_interval_count": count,
            "pooled_mean_power_v2i_mw": mean_v2i_power,
            "pooled_mean_power_v2v_mw": mean_v2v_power,
            "pooled_complete_interval_mean": mean, "pooled_complete_interval_cv2": pooled_cv2,
            "equal_flow_mean_interval_mean": float(np.mean(flow_means)) if flow_means else None,
            "equal_flow_mean_within_cv2": float(np.mean(flow_cv2)) if flow_cv2 else None,
            "between_flow_cv2_of_interval_means": float(np.var(flow_means) / np.mean(flow_means) ** 2) if flow_means else None,
            "complete_long_gt100_count": sum(r["complete_long_gt100_count"] for r in selected),
            "complete_long_gt100_sum_L": sum(r["complete_long_gt100_sum_L"] for r in selected),
            "left_censored_slots": sum(r["left_censored_slots"] for r in selected),
            "right_censored_slots": sum(r["right_censored_slots"] for r in selected),
        })
    return {"status": "PASS", "source_cells": 12, "flow_rows": len(rows),
            "episode_continuity": "episodes flattened within each eval world; never across worlds",
            "interval_source": "reset_event, not capped AoI", "flow_rows_data": rows, "summary": summaries}


def write_report(output_root: Path, report: dict) -> Path:
    output = output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "service_regularity_world_agent.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["flow_rows_data"][0]))
        writer.writeheader(); writer.writerows(report["flow_rows_data"])
    with (output / "service_regularity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["summary"][0]))
        writer.writeheader(); writer.writerows(report["summary"])
    compact = {k: v for k, v in report.items() if k != "flow_rows_data"}
    (output / "service_regularity_summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# MAPPO service regularity audit", "", "Existing shared-actor-v1 trajectories only; no sampling was performed.", "",
             "Complete update intervals join consecutive scored episodes inside one world, but never cross a world boundary. Boundary-censored intervals are reported separately. Moments come from reset events, not capped AoI.", "",
             "| actor | rho | q | reset rate | V2I/V2V power mW | interval mean | pooled CV2 | equal-flow within CV2 | between-flow CV2 | >100 count |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        lines.append(f"| {row['structure']} | {row['pooled_rho']:.4f} | {row['pooled_q']:.4f} | {row['pooled_reset_rate']:.4f} | {row['pooled_mean_power_v2i_mw']:.3f}/{row['pooled_mean_power_v2v_mw']:.3f} | {row['pooled_complete_interval_mean']:.3f} | {row['pooled_complete_interval_cv2']:.3f} | {row['equal_flow_mean_within_cv2']:.3f} | {row['between_flow_cv2_of_interval_means']:.3f} | {row['complete_long_gt100_count']} |")
    (output / "service_regularity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.shared_result_root)
    output = write_report(args.output_root, report)
    print(json.dumps({"status": report["status"], "source_cells": report["source_cells"], "flow_rows": report["flow_rows"], "output": str(output), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
