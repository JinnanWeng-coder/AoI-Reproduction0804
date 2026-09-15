"""Contracts and completion validation for the N1 training-world revisit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from analysis.evaluate_mappo_feasibility_reward import feasibility_eval_id
from analysis.mappo_e5_contract import SCENARIO, SEEDS, independent_run_name, shared_run_name


CONDITIONS = (
    ("independent", "combined"),
    ("shared", "combined"),
    ("independent", "tdec"),
    ("shared", "tdec"),
)
EVAL_EPISODES = 100
EVAL_WARMUP_EPISODES = 5
STEPS_PER_EPISODE = 100
N_AGENTS = 5


@dataclass(frozen=True)
class N1Cell:
    cell_id: int
    structure: str
    variant: str
    training_seed: int
    eval_world: int
    run_name: str


def n1_cell(cell_id: int) -> N1Cell:
    """Map one of the fixed 24 array tasks to a policy and its own world."""

    cell_id = int(cell_id)
    if cell_id < 0 or cell_id >= 24:
        raise ValueError("cell_id must satisfy 0 <= cell_id < 24")
    structure, variant = CONDITIONS[cell_id // 6]
    seed = SEEDS[cell_id % 6]
    run_name = (
        independent_run_name(variant, seed)
        if structure == "independent"
        else shared_run_name(variant, seed)
    )
    return N1Cell(cell_id, structure, variant, seed, seed, run_name)


def source_root(study_root: Path, cell: N1Cell) -> Path:
    study_root = Path(study_root).expanduser().resolve()
    if cell.structure == "independent":
        return study_root / "existing-evidence" / "tdec-ab-v1" / "P5_N4_gap25"
    if cell.variant == "tdec":
        return study_root / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25"
    return study_root / "E5-sharing-critic" / "P5_N4_gap25"


def source_run_dir(study_root: Path, cell: N1Cell) -> Path:
    return source_root(study_root, cell) / "training" / "runs" / cell.run_name


def n1_eval_id(cell: N1Cell) -> str:
    return feasibility_eval_id(
        "baseline", [cell.eval_world], EVAL_EPISODES, EVAL_WARMUP_EPISODES
    )


def n1_eval_dir(n1_root: Path, cell: N1Cell) -> Path:
    return (
        Path(n1_root).expanduser().resolve()
        / "evaluations"
        / cell.run_name
        / n1_eval_id(cell)
    )


def development_eval_root(study_root: Path, cell: N1Cell) -> Path:
    study_root = Path(study_root).expanduser().resolve()
    if cell.variant == "combined":
        return study_root / "E5-sharing-critic" / "P5_N4_gap25"
    return study_root / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25"


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_equal(source: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(
                f"{label}: {key}={source.get(key)!r}, expected {value!r}"
            )


def validate_source_run(study_root: Path, cell: N1Cell) -> tuple[Path, dict, dict]:
    run_dir = source_run_dir(study_root, cell)
    if run_dir.name != cell.run_name:
        raise ValueError(f"unexpected source run name: {run_dir}")
    config = _read_json(run_dir / "config.resolved.json")
    complete = _read_json(run_dir / "COMPLETE.json")
    config_expected = {
        "algorithm": "mappo",
        "seed": cell.training_seed,
        "episodes": 500,
        "steps_per_episode": STEPS_PER_EPISODE,
        "mappo_rollout_episodes": 5,
        "mappo_ppo_epochs": 10,
        "mappo_value_clip_mode": "normalized",
    }
    _require_equal(config, config_expected, str(run_dir))
    if config.get("scenario", {}).get("id") != SCENARIO:
        raise ValueError(f"{run_dir}: scenario mismatch")
    if config.get("mappo_variant", "combined") != cell.variant:
        raise ValueError(f"{run_dir}: MAPPO variant mismatch")
    if bool(config.get("mappo_actor_sharing", False)) != (
        cell.structure == "shared"
    ):
        raise ValueError(f"{run_dir}: actor-sharing mismatch")
    if complete.get("status") != "complete":
        raise ValueError(f"{run_dir}: training is not complete")
    if complete.get("mappo_variant", "combined") != cell.variant:
        raise ValueError(f"{run_dir}: completion variant mismatch")
    if bool(complete.get("actor_sharing", False)) != (cell.structure == "shared"):
        raise ValueError(f"{run_dir}: completion actor-sharing mismatch")
    if not (run_dir / "policy_final.pt").is_file():
        raise ValueError(f"{run_dir}: policy_final.pt is missing")
    return run_dir, config, complete


def _validate_metrics(metrics_path: Path) -> None:
    required_agent = (
        "aoi_ms",
        "success",
        "remaining_demand",
        "rb",
        "mode",
        "executed_power_dbm",
        "reward_task1",
        "reward_task2",
    )
    with np.load(metrics_path, allow_pickle=False) as arrays:
        missing = [key for key in required_agent + ("reward_global",) if key not in arrays]
        if missing:
            raise ValueError(f"{metrics_path}: missing arrays {missing}")
        expected_agent = (1, EVAL_EPISODES, STEPS_PER_EPISODE, N_AGENTS)
        for key in required_agent:
            if arrays[key].shape != expected_agent:
                raise ValueError(
                    f"{metrics_path}: {key} shape={arrays[key].shape}, expected {expected_agent}"
                )
        expected_global = expected_agent[:-1]
        if arrays["reward_global"].shape != expected_global:
            raise ValueError(
                f"{metrics_path}: reward_global shape={arrays['reward_global'].shape}, "
                f"expected {expected_global}"
            )


def _marker_payload(
    cell: N1Cell,
    run_dir: Path,
    eval_dir: Path,
    training_complete: dict,
    evaluation_complete: dict,
    provenance: dict,
) -> dict:
    return {
        "status": "complete",
        "experiment": "N1",
        "evaluation_role": "training_world_revisit",
        **asdict(cell),
        "source_run": str(run_dir),
        "policy_name": "policy_final.pt",
        "policy_episode": 500,
        "training_commit": training_complete.get("reproduction_git_commit"),
        "evaluation_branch": evaluation_complete.get("reproduction_git_branch"),
        "evaluation_commit": evaluation_complete.get("reproduction_git_commit"),
        "evaluation_git_dirty": evaluation_complete.get("reproduction_git_dirty"),
        "training_source_commit": provenance.get("training_source_commit"),
        "eval_dir": str(eval_dir),
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": EVAL_WARMUP_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "mappo_eval_mode": "stochastic",
        "intervention_arm": "baseline",
        "policy_parameters_unchanged": True,
        "training_world_revisit_semantics": (
            "frozen final policy revisits the initial segment of its training-initialization "
            "world; this is not the training last100 state trajectory"
        ),
        "closed_loop_scope_note": (
            "matching the world initialization does not imply an identical training-time "
            "closed-loop observation distribution"
        ),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_n1_cell(
    study_root: Path,
    n1_root: Path,
    cell_id: int,
    *,
    write_marker: bool = False,
) -> dict:
    cell = n1_cell(cell_id)
    run_dir, _config, training_complete = validate_source_run(study_root, cell)
    eval_dir = n1_eval_dir(n1_root, cell)
    evaluation_complete = _read_json(eval_dir / "EVAL_COMPLETE.json")
    provenance = _read_json(eval_dir / "provenance.json")
    expected = {
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": cell.variant,
        "actor_sharing": cell.structure == "shared",
        "actor_network_count": 1 if cell.structure == "shared" else 5,
        "intervention_arm": "baseline",
        "mappo_eval_mode": "stochastic",
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": EVAL_WARMUP_EPISODES,
        "eval_seeds": [cell.eval_world],
        "eval_episodes": EVAL_EPISODES,
        "training_seed": cell.training_seed,
        "training_run_name": cell.run_name,
        "scenario": SCENARIO,
        "policy_name": "policy_final.pt",
        "policy_episode": 500,
        "policy_path_is_relative_to_eval": True,
        "policy_parameters_unchanged": True,
        "is_frozen_eval": True,
        "reproduction_git_dirty": False,
    }
    _require_equal(evaluation_complete, expected, str(eval_dir))
    policy_reference = evaluation_complete.get("policy")
    if not isinstance(policy_reference, str) or not policy_reference:
        raise ValueError(f"{eval_dir}: policy reference is missing")
    resolved_policy = (eval_dir / policy_reference).resolve()
    if resolved_policy != (run_dir / "policy_final.pt").resolve():
        raise ValueError(
            f"{eval_dir}: policy resolves to {resolved_policy}, expected "
            f"{(run_dir / 'policy_final.pt').resolve()}"
        )
    _require_equal(
        provenance,
        {
            "algorithm": "mappo",
            "mappo_variant": cell.variant,
            "actor_sharing": cell.structure == "shared",
            "training_seed": cell.training_seed,
            "eval_seeds": [cell.eval_world],
            "eval_episodes": EVAL_EPISODES,
            "eval_warmup_episodes": EVAL_WARMUP_EPISODES,
            "mappo_eval_mode": "stochastic",
            "intervention_arm": "baseline",
            "policy": policy_reference,
        },
        str(eval_dir / "provenance.json"),
    )
    training_commit = training_complete.get("reproduction_git_commit")
    if training_commit is not None and provenance.get("training_source_commit") != training_commit:
        raise ValueError(f"{eval_dir}: training source commit mismatch")
    _validate_metrics(eval_dir / "metrics.npz")
    marker = _marker_payload(
        cell, run_dir, eval_dir, training_complete, evaluation_complete, provenance
    )
    marker_path = eval_dir / "N1_COMPLETE.json"
    if write_marker:
        if marker_path.exists():
            raise FileExistsError(f"refusing to overwrite N1 marker: {marker_path}")
        _write_json_atomic(marker_path, marker)
    else:
        existing = _read_json(marker_path)
        _require_equal(existing, marker, str(marker_path))
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--n1-root", required=True, type=Path)
    parser.add_argument("--cell-id", required=True, type=int)
    parser.add_argument("--write-marker", action="store_true")
    args = parser.parse_args()
    result = validate_n1_cell(
        args.study_root, args.n1_root, args.cell_id, write_marker=args.write_marker
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
