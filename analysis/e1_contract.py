"""Fixed names and source-run contracts for the E1 unified frozen evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


SEEDS = (8, 9, 10, 11, 12, 13)
EVAL_SEEDS = (213, 214, 215, 216, 217, 218)
FORBIDDEN_EVAL_SEEDS = (201, 202, 203, 204, 205, 206)
SCENARIO = "p05_n04_g25"
N_RB = 3
N_AGENTS = 5
STEPS_PER_EPISODE = 100
EVAL_EPISODES = 100
EVAL_WARMUP = 5
CHECKPOINT_EPISODE = 500
NOISE_LEVELS = (0.0, 0.3)
ALGORITHMS = ("modified_maddpg", "modified_maddpg_tdec")
ALGORITHM_LABELS = {
    "modified_maddpg": "algorithm1",
    "modified_maddpg_tdec": "algorithm2",
}
HISTORICAL_ALG2_PREFIX = "gap_global_slow_sync_slow01"
ACTION_NOISE_TAG = 0xA01
MAPPO_EVAL_ID = (
    "eval_validation_policy_final_stochastic_sequential_warm_"
    "warm5_s213-214-215-216-217-218_ep100_feasibility_reward_arm_baseline"
)


def cell_to_algorithm_noise_seed(cell_id: int) -> tuple[str, float, int]:
    cell_id = int(cell_id)
    if cell_id < 0 or cell_id >= 24:
        raise ValueError("cell_id must satisfy 0 <= cell_id < 24")
    algorithm = ALGORITHMS[cell_id // 12]
    noise = NOISE_LEVELS[(cell_id % 12) // 6]
    seed = SEEDS[cell_id % 6]
    return algorithm, float(noise), int(seed)


def canonical_run_name(algorithm: str, seed: int) -> str:
    seed_token = f"{int(seed):02d}"
    if algorithm == "modified_maddpg":
        return f"modified_maddpg_default_{SCENARIO}_seed{seed_token}"
    if algorithm == "modified_maddpg_tdec":
        return f"tdec_{SCENARIO}_seed{seed_token}"
    raise ValueError(f"unsupported MADDPG algorithm: {algorithm}")


def historical_run_name(algorithm: str, seed: int) -> str:
    if algorithm == "modified_maddpg":
        return canonical_run_name(algorithm, seed)
    if algorithm == "modified_maddpg_tdec":
        return f"{HISTORICAL_ALG2_PREFIX}_{SCENARIO}_seed{int(seed):02d}"
    raise ValueError(f"unsupported MADDPG algorithm: {algorithm}")


def maddpg_eval_id(noise: float) -> str:
    noise = float(noise)
    if noise not in NOISE_LEVELS:
        raise ValueError(f"unsupported E1 eval noise: {noise}")
    token = "0p0" if noise == 0.0 else "0p3"
    seed_token = "-".join(str(seed) for seed in EVAL_SEEDS)
    return (
        f"eval_e1_latest{CHECKPOINT_EPISODE}_noise{token}_sequential_warm_"
        f"warm{EVAL_WARMUP}_s{seed_token}_ep{EVAL_EPISODES}"
    )


def algorithm_root(diagnostics_root: Path, algorithm: str) -> Path:
    diagnostics_root = Path(diagnostics_root)
    if algorithm == "modified_maddpg":
        return diagnostics_root / "Modified_MADDPG_results" / "default" / "P5_N4_gap25"
    if algorithm == "modified_maddpg_tdec":
        return diagnostics_root / "Modified_MADDPG_with_TDec_results" / "default" / "P5_N4_gap25"
    raise ValueError(f"unsupported MADDPG algorithm: {algorithm}")


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _algorithm_from_record(record: Mapping[str, Any], default: str) -> str:
    value = record.get("algorithm")
    if value in {None, ""}:
        return default
    return str(value)


def resolve_maddpg_source_run(diagnostics_root: Path, algorithm: str, seed: int) -> dict[str, Any]:
    """Locate and verify one completed Algorithm 1/2 training run."""

    if algorithm not in ALGORITHMS:
        raise ValueError(f"unsupported MADDPG algorithm: {algorithm}")
    if int(seed) not in SEEDS:
        raise ValueError(f"unsupported training seed: {seed}")
    if algorithm == "mappo" or "mappo" in algorithm:
        raise ValueError("E1 MADDPG source resolution refuses MAPPO-TDec runs")

    diagnostics_root = Path(diagnostics_root).resolve()
    root = algorithm_root(diagnostics_root, algorithm)
    canonical = canonical_run_name(algorithm, seed)
    historical = historical_run_name(algorithm, seed)
    run_dir = root / "runs" / canonical
    if not run_dir.exists():
        fallback = root / "runs" / historical
        if fallback.exists():
            run_dir = fallback
        else:
            raise FileNotFoundError(f"missing MADDPG source run: {run_dir}")
    if run_dir.is_symlink() and algorithm != "modified_maddpg_tdec":
        raise ValueError(f"Algorithm 1 run must be a real directory: {run_dir}")
    resolved = run_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"source run does not resolve to a directory: {run_dir}")
    if algorithm == "modified_maddpg_tdec" and historical not in resolved.name and canonical not in resolved.name:
        raise ValueError(f"Algorithm 2 resolved path is not the historical TDec run: {resolved}")

    complete_path = run_dir / "COMPLETE.json"
    config_path = run_dir / "config.resolved.json"
    provenance_path = run_dir / "provenance.json"
    checkpoint = (run_dir / "checkpoints" / "latest.pt").resolve()
    if checkpoint.name != "latest.pt" or checkpoint.parent.name != "checkpoints":
        raise ValueError("E1 requires checkpoints/latest.pt")
    if (run_dir / "checkpoints" / "best.pt").resolve() == checkpoint:
        raise ValueError("refusing to treat best.pt as latest.pt")
    for path in (complete_path, config_path, checkpoint):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"unreadable source artifact: {path}")

    complete = _read_json(complete_path)
    config = _read_json(config_path)
    provenance = _read_json(provenance_path) if provenance_path.is_file() else {}
    config_algorithm = _algorithm_from_record(config, "modified_maddpg_tdec")
    complete_algorithm = _algorithm_from_record(complete, config_algorithm)
    if config_algorithm != algorithm or complete_algorithm != algorithm:
        raise ValueError(
            f"{canonical}: algorithm mismatch config={config_algorithm!r} "
            f"complete={complete_algorithm!r} expected={algorithm!r}"
        )
    if algorithm == "modified_maddpg" and "algorithm" not in config:
        raise ValueError(f"{canonical}: Algorithm 1 must record an explicit algorithm field")
    if complete.get("status") != "complete":
        raise ValueError(f"{canonical}: COMPLETE.json is not complete")
    if int(complete.get("final_episode", complete.get("episodes", -1))) != CHECKPOINT_EPISODE:
        raise ValueError(f"{canonical}: training is not a finished {CHECKPOINT_EPISODE}-episode run")
    scenario = config.get("scenario")
    scenario_id = scenario.get("id") if isinstance(scenario, dict) else scenario
    if scenario_id != SCENARIO or int(config.get("seed", -1)) != int(seed):
        raise ValueError(f"{canonical}: scenario/seed mismatch")
    if int(config.get("n_rb", -1)) != N_RB or int(config.get("steps_per_episode", -1)) != STEPS_PER_EPISODE:
        raise ValueError(f"{canonical}: n_rb/steps_per_episode mismatch")
    if config.get("eval_protocol") != "sequential_warm":
        raise ValueError(f"{canonical}: source eval_protocol must be sequential_warm")

    manifest_info = _verify_manifest(diagnostics_root, algorithm, seed, canonical, historical, resolved)
    return {
        "algorithm": algorithm,
        "algorithm_label": ALGORITHM_LABELS[algorithm],
        "seed": int(seed),
        "canonical_run_name": canonical,
        "historical_run_name": historical,
        "run_dir": run_dir,
        "resolved_run_dir": resolved,
        "is_symlink": run_dir.is_symlink(),
        "checkpoint": checkpoint,
        "complete_path": complete_path,
        "config_path": config_path,
        "provenance_path": provenance_path if provenance_path.is_file() else None,
        "complete": complete,
        "config": config,
        "provenance": provenance,
        "manifest": manifest_info,
    }


def _verify_manifest(
    diagnostics_root: Path,
    algorithm: str,
    seed: int,
    canonical: str,
    historical: str,
    resolved: Path,
) -> dict[str, Any]:
    if algorithm == "modified_maddpg_tdec":
        path = diagnostics_root / "Modified_MADDPG_with_TDec_results" / "run_manifest.csv"
    else:
        path = diagnostics_root / "algorithm-comparison" / "Modified_MADDPG_vs_TDec" / "run_manifest.csv"
    info: dict[str, Any] = {"path": str(path) if path.is_file() else None, "matched": False}
    if not path.is_file():
        if algorithm == "modified_maddpg_tdec":
            raise FileNotFoundError(f"missing Algorithm 2 run_manifest.csv: {path}")
        return info
    matches = []
    for row in _read_manifest_rows(path):
        row_algorithm = _algorithm_from_record(row, "modified_maddpg_tdec")
        if row_algorithm != algorithm or int(float(row.get("seed", -1))) != int(seed):
            continue
        if str(row.get("canonical_name", row.get("run_name", ""))) not in {canonical, historical, ""}:
            if str(row.get("run_name", "")) not in {canonical, historical}:
                continue
        if str(row.get("experiment", "default")) not in {"default", ""}:
            continue
        matches.append(row)
    if not matches:
        if algorithm == "modified_maddpg_tdec":
            raise ValueError(f"manifest has no default Algorithm 2 row for seed {seed}")
        return info
    row = matches[0]
    if int(float(row.get("episodes", -1))) != CHECKPOINT_EPISODE:
        raise ValueError(f"manifest episodes for seed {seed} are not {CHECKPOINT_EPISODE}")
    if algorithm == "modified_maddpg_tdec" and str(row.get("canonical_name")) != canonical:
        raise ValueError(f"manifest canonical_name mismatch for seed {seed}")
    if algorithm == "modified_maddpg_tdec" and historical not in resolved.name:
        raise ValueError(f"resolved Algorithm 2 run is not {historical}")
    info.update({"matched": True, "row": row})
    return info


def validate_latest500_payload(payload: Mapping[str, Any], algorithm: str, seed: int, expected_episode: int = CHECKPOINT_EPISODE) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    raw_config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    payload_algorithm = payload.get("algorithm", raw_config.get("algorithm", "modified_maddpg_tdec"))
    if payload_algorithm != algorithm:
        raise ValueError(
            f"latest.pt algorithm={payload_algorithm!r}, expected {algorithm!r} "
            "(Algorithm 2 missing algorithm defaults to modified_maddpg_tdec, not MAPPO-TDec)"
        )
    if payload_algorithm == "mappo":
        raise ValueError("refusing a MAPPO checkpoint as an E1 MADDPG source")
    if int(payload.get("episode", -1)) != int(expected_episode):
        raise ValueError(f"latest.pt episode={payload.get('episode')!r}, expected {expected_episode}")
    if payload.get("completed") is not True and payload.get("training_completed") is not True:
        raise ValueError("latest.pt is not a completed training checkpoint")
    if int(raw_config.get("seed", -1)) != int(seed):
        raise ValueError("latest.pt config seed mismatch")
    scenario = raw_config.get("scenario")
    scenario_id = scenario.get("id") if isinstance(scenario, dict) else scenario
    if scenario_id != SCENARIO:
        raise ValueError("latest.pt scenario is not p05_n04_g25")
    if int(raw_config.get("n_rb", -1)) != N_RB:
        raise ValueError("latest.pt n_rb is not 3")


def mappo_run_name(structure: str, variant: str, seed: int) -> str:
    seed_token = f"{int(seed):02d}"
    if structure == "independent":
        return f"mappo_tdec_ab_{variant}_{SCENARIO}_seed{seed_token}"
    if structure == "shared" and variant == "tdec":
        return f"mappo_shared_actor_tdec_{SCENARIO}_seed{seed_token}"
    if structure == "shared" and variant == "combined":
        return f"mappo_e5_shared_combined_{SCENARIO}_seed{seed_token}"
    raise ValueError(f"unsupported MAPPO cell: {structure}/{variant}")


def mappo_eval_root(diagnostics_root: Path, structure: str, variant: str) -> Path:
    study = Path(diagnostics_root) / "actor-sharing-study"
    if variant == "combined":
        return study / "E5-sharing-critic" / "P5_N4_gap25"
    if variant == "tdec":
        return study / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25"
    raise ValueError(f"unsupported MAPPO variant: {variant}")


def mappo_training_root(diagnostics_root: Path, structure: str, variant: str) -> Path:
    study = Path(diagnostics_root) / "actor-sharing-study"
    if structure == "independent":
        return study / "existing-evidence" / "tdec-ab-v1" / "P5_N4_gap25"
    if structure == "shared" and variant == "tdec":
        return study / "existing-evidence" / "shared-actor-v1" / "P5_N4_gap25"
    if structure == "shared" and variant == "combined":
        return study / "E5-sharing-critic" / "P5_N4_gap25"
    raise ValueError(f"unsupported MAPPO training cell: {structure}/{variant}")
