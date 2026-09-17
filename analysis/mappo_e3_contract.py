"""E3 new-world candidate audit, immutable lock, source/cell validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analysis.mappo_e5_contract import independent_run_name, shared_run_name

PROTOCOL_FILE = Path(__file__).resolve().parents[1] / "hpc" / "e3_locked_test_protocol.json"
AGENT_SHAPE = (6, 100, 100, 5)
ARRAY_FIELDS = ("aoi_ms", "success", "remaining_demand", "reset_event",
                "executed_power_dbm", "reward_global", "reward_task1", "reward_task2")
LOCKED_E3_COMMIT = "c56425776ca6b4e71d0002f6192900e1475b5693"
REPAIR_ALLOWED_PATHS = {"analysis/mappo_e3_contract.py", "analysis/summarize_mappo_e3_locked_test.py",
                        "tests/test_mappo_e3_locked_test.py", "hpc/README_E3_LOCKED_TEST.md"}


def protocol() -> dict:
    data = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    assert data["experiment"] == "E3-locked-test" and len(data["policies"]) == 24
    assert len(data["candidate_worlds"]) == len(set(data["candidate_worlds"])) == 6
    assert set(data["candidate_worlds"]).isdisjoint(set(data["excluded_known_or_reserved_worlds"]))
    for i, cell in enumerate(data["policies"]):
        structure, variant, seed = cell["actor_structure"], cell["value_configuration"], cell["training_seed"]
        assert cell["cell_id"] == i and seed == data["training_seeds"][i % 6]
        assert (structure, variant) == (("independent", "combined"), ("shared", "combined"),
                                        ("independent", "tdec"), ("shared", "tdec"))[i // 6]
        run = independent_run_name(variant, seed) if structure == "independent" else shared_run_name(variant, seed)
        assert cell["training_run_name"] == run and cell["policy_episode"] == 500
    return data


def result_root(study_root: Path) -> Path:
    root = Path(study_root).resolve()
    result = root / protocol()["output_relative"]
    if not result.resolve().is_relative_to(root):
        raise ValueError("E3 result root escapes study root")
    return result


def _prelock_directory_check(output: Path) -> None:
    if output.exists():
        allowed = {"world_use_audit.json", "source_inventory.json", "slurm_logs", "tmp", "cache"}
        unexpected = sorted(path.name for path in output.iterdir()
                            if path.name not in allowed and not re.fullmatch(
                                r"world_use_audit_previous_[0-9a-f]{7}\.json", path.name))
        if unexpected:
            raise ValueError(f"E3 root has existing unreviewed content: {unexpected}")


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _require(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: got {actual!r}, expected {expected!r}")


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def cell(cell_id: int) -> dict:
    if not 0 <= int(cell_id) < 24:
        raise ValueError("cell-id must be 0..23")
    return protocol()["policies"][int(cell_id)]


def run_dir(study_root: Path, item: dict) -> Path:
    return Path(study_root).resolve() / item["training_root"] / "training" / "runs" / item["training_run_name"]


def eval_id() -> str:
    worlds = "-".join(map(str, protocol()["candidate_worlds"]))
    return f"eval_validation_policy_final_stochastic_sequential_warm_warm5_s{worlds}_ep100_feasibility_reward_arm_baseline"


def eval_dir(study_root: Path, item: dict) -> Path:
    return result_root(study_root) / "evaluations" / item["training_run_name"] / eval_id()


def _assert_source(study_root: Path, item: dict, check_payload: bool) -> dict:
    run = run_dir(study_root, item)
    config, complete = _json(run / "config.resolved.json"), _json(run / "COMPLETE.json")
    expected = {"algorithm": "mappo", "seed": item["training_seed"], "episodes": 500,
                "steps_per_episode": 100, "n_rb": 3, "mappo_rollout_episodes": 5,
                "mappo_ppo_epochs": 10, "mappo_value_clip_mode": "normalized",
                "eval_protocol": "sequential_warm", "eval_warmup_episodes": 5,
                "checkpoint_mode": "policy_only", "power_continuous": True}
    for key, value in expected.items():
        _require(config.get(key), value, f"{run}/{key}")
    _require(config.get("scenario", {}).get("id"), "p05_n04_g25", f"{run}/scenario")
    _require(config.get("mappo_variant", "combined"), item["value_configuration"], f"{run}/variant")
    selection_worlds = [int(world) for world in config.get("selection_validation_seeds", [])]
    if set(selection_worlds) & set(protocol()["candidate_worlds"]):
        raise ValueError(f"{run}: candidate worlds overlap source selection_validation_seeds")
    sharing = item["actor_structure"] == "shared"
    _require(bool(config.get("mappo_actor_sharing", False)), sharing, f"{run}/sharing")
    for key, value in {"status": "complete", "algorithm": "mappo", "policy_final": "policy_final.pt"}.items():
        _require(complete.get(key), value, f"{run}/COMPLETE/{key}")
    _require(complete.get("mappo_variant", "combined"), item["value_configuration"], f"{run}/complete variant")
    _require(bool(complete.get("actor_sharing", False)), sharing, f"{run}/complete sharing")
    policy = run / "policy_final.pt"
    if not policy.is_file():
        raise FileNotFoundError(policy)
    if check_payload:
        import torch  # HPC only; no policy load in local synthetic tests
        from aoi_v2x_reproduction.runtime.runner import _validate_mappo_policy_artifact

        payload = torch.load(policy, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid policy payload: {policy}")
        _, validated, _ = _validate_mappo_policy_artifact(policy, payload)
        _require(validated.mappo_variant, item["value_configuration"], f"{policy}/variant")
        _require(bool(validated.mappo_actor_sharing), sharing, f"{policy}/actor sharing")
        _require(payload.get("episode"), 500, f"{policy}/episode")
    return {"cell_id": item["cell_id"], "actor_structure": item["actor_structure"],
            "value_configuration": item["value_configuration"], "training_seed": item["training_seed"],
            "training_run_name": item["training_run_name"], "training_dir": str(run),
            "policy": str(policy), "policy_episode": 500,
            "training_commit": complete.get("reproduction_git_commit"),
            "selection_validation_seeds": selection_worlds,
            "source_config_hash": complete.get("config_hash"),
            "payload_checked": check_payload}


def source_preflight(study_root: Path, check_payload: bool = True) -> dict:
    output = result_root(study_root)
    _prelock_directory_check(output)
    rows = [_assert_source(study_root, item, check_payload) for item in protocol()["policies"]]
    if len({row["policy"] for row in rows}) != 24:
        raise ValueError("duplicate E3 source policy")
    result = {"status": "PASS", "stage": "source_preflight", "commit": current_commit(),
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "policy_count": 24, "all_payloads_checked": check_payload, "sources": rows}
    _write(output / "source_inventory.json", result)
    return {key: result[key] for key in ("status", "stage", "commit", "policy_count", "all_payloads_checked")}


WORLD_FIELDS = ("eval_seeds", "eval_worlds", "world_seeds", "selection_validation_seeds",
                "validation_seeds", "eval_world", "eval_seed", "world")
EVAL_ID_RE = re.compile(r"_s([0-9]+(?:-[0-9]+)*)_ep[0-9]+")
LOG_WORLD_RE = re.compile(r"(?:eval_world|eval_seed|eval_seeds|--eval-seeds)[=\s]+([0-9]+(?:[-,][0-9]+)*)")


def _world_values(value) -> set[int]:
    if isinstance(value, list):
        return {int(x) for x in value if str(x).strip().isdigit()}
    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        return {int(x) for x in re.findall(r"\b\d+\b", value)}
    return set()


def history_audit(study_root: Path, supersede_blocked_audit: bool = False) -> dict:
    """Structured evidence inventory; human must also review execution records."""
    output = result_root(study_root)
    _prelock_directory_check(output)
    prior_path = output / "world_use_audit.json"
    if prior_path.exists():
        prior = _json(prior_path)
        if not supersede_blocked_audit:
            raise ValueError("existing world-use audit; explicit supersede of blocked prelock audit required")
        if prior.get("status") != "BLOCKED" or prior.get("candidate_worlds") == protocol()["candidate_worlds"]:
            raise ValueError("only a blocked audit for a different prelock candidate list may be superseded")
        previous_commit = prior.get("commit")
        if not isinstance(previous_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", previous_commit):
            raise ValueError("prior audit has no valid commit identity")
        backup = output / f"world_use_audit_previous_{previous_commit[:7]}.json"
        if backup.exists():
            raise FileExistsError(f"refusing to overwrite prior audit backup: {backup}")
        shutil.copy2(prior_path, backup)
    candidates = set(protocol()["candidate_worlds"])
    study = Path(study_root).resolve()
    roots = [study, study.parent / "MAPPO_results"]
    roots += [candidate for candidate in (study.parents[1] / "MAPPO_results",
                                          study.parents[1] / "AoI-Reproduction0804" / "MAPPO_results")
              if candidate.exists()]
    inspected, hits, observed, errors = 0, [], set(), []
    for root in roots:
        if not root.exists():
            errors.append(f"history root missing: {root}")
            continue
        for path in root.rglob("*"):
            if output in path.parents:
                continue
            values = set()
            if path.is_dir() and "evaluations" in path.parts:
                match = EVAL_ID_RE.search(path.name)
                if not match:
                    continue
                values.update(int(x) for x in match.group(1).split("-"))
            elif not path.is_file():
                continue
            elif path.suffix.lower() == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        continue
                    for key in WORLD_FIELDS:
                        values.update(_world_values(data.get(key)))
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{path}: {exc}")
                    continue
            elif path.suffix.lower() == ".csv":
                try:
                    with path.open(newline="", encoding="utf-8-sig") as handle:
                        reader = csv.DictReader(handle)
                        fields = set(reader.fieldnames or []) & set(WORLD_FIELDS)
                        if not fields:
                            continue
                        for row in reader:
                            for key in fields:
                                values.update(_world_values(row[key]))
                except (OSError, ValueError) as exc:
                    errors.append(f"{path}: {exc}")
                    continue
            elif path.name == "metrics.npz" and "evaluations" in path.parts:
                match = EVAL_ID_RE.search(path.parent.name)
                if match:
                    values.update(int(x) for x in match.group(1).split("-"))
            elif path.suffix.lower() in {".out", ".err"} and "slurm_logs" in path.parts:
                try:
                    with path.open(encoding="utf-8", errors="replace") as handle:
                        for line in handle:
                            for match in LOG_WORLD_RE.finditer(line):
                                values.update(int(x) for x in re.findall(r"\d+", match.group(1)))
                except OSError as exc:
                    errors.append(f"{path}: {exc}")
                    continue
            else:
                continue
            inspected += 1
            observed.update(values)
            if values & candidates:
                hits.append({"path": str(path), "candidate_worlds": sorted(values & candidates)})
    report = {"status": "CANDIDATE_CLEAR_REQUIRES_MANUAL_REVIEW" if not hits and not errors else "BLOCKED",
              "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "candidate_worlds": sorted(candidates), "inspected_structured_sources": inspected,
              "observed_worlds": sorted(observed), "candidate_hits": hits, "scan_errors": errors,
              "scan_roots": [str(x) for x in roots],
              "manual_review_required": "Check all actual EVAL_COMPLETE/provenance, evaluation dirs, Slurm execution records and any other research roots before lock; no inference from filenames alone",
              "commit": current_commit()}
    _write(output / "world_use_audit.json", report)
    return report


def create_lock(study_root: Path, confirm_reviewed: bool) -> dict:
    output = result_root(study_root)
    if not confirm_reviewed:
        raise ValueError("manual historical execution-record review must be explicitly confirmed")
    _prelock_directory_check(output)
    audit = _json(output / "world_use_audit.json")
    sources = _json(output / "source_inventory.json")
    _require(audit.get("status"), "CANDIDATE_CLEAR_REQUIRES_MANUAL_REVIEW", "history audit")
    _require(audit.get("candidate_worlds"), protocol()["candidate_worlds"], "history world list")
    _require(audit.get("candidate_hits"), [], "history candidate hits")
    _require(audit.get("scan_errors"), [], "history scan errors")
    study = Path(study_root).resolve()
    required_roots = {str(study), str(study.parent / "MAPPO_results")}
    if not required_roots.issubset(set(audit.get("scan_roots", []))):
        raise ValueError("history audit did not cover study and diagnostic MAPPO_results roots")
    _require(sources.get("policy_count"), 24, "source policy count")
    _require(sources.get("all_payloads_checked"), True, "policy payload validation")
    if len(sources.get("sources", [])) != 24:
        raise ValueError("source inventory must enumerate all 24 policies")
    for item, record in zip(protocol()["policies"], sources["sources"]):
        for key in ("cell_id", "actor_structure", "value_configuration", "training_seed", "training_run_name"):
            _require(record.get(key), item[key], f"source inventory cell {item['cell_id']}/{key}")
    for record in (audit, sources):
        _require(record.get("commit"), current_commit(), "audit/source commit")
    locked = {"status": "LOCKED", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
              "implementation_commit": current_commit(), "protocol": protocol(),
              "history_review_attestation": "operator confirmed structured evidence and execution-record review; no candidate previously used in development",
              "source_inventory": str(output / "source_inventory.json"),
              "world_use_audit": str(output / "world_use_audit.json")}
    _write(output / "lock.json", locked)
    return {k: locked[k] for k in ("status", "locked_at_utc", "implementation_commit")}


def _repair_diff_paths(locked_commit: str, repair_commit: str) -> list[str]:
    subprocess.run(["git", "merge-base", "--is-ancestor", locked_commit, repair_commit], check=True)
    paths = subprocess.check_output(["git", "diff", "--name-only", f"{locked_commit}..{repair_commit}"],
                                    text=True).splitlines()
    if not paths or not set(paths).issubset(REPAIR_ALLOWED_PATHS):
        raise ValueError(f"E3 repair contains unexpected changes: {paths}")
    return paths


def amend_lock_for_technical_repair(study_root: Path, confirm: bool) -> dict:
    if not confirm:
        raise ValueError("explicit technical-repair confirmation required")
    output = result_root(study_root)
    locked = _json(output / "lock.json")
    _require(locked.get("status"), "LOCKED", "E3 lock status")
    _require(locked.get("protocol"), protocol(), "E3 protocol drift")
    _require(locked.get("implementation_commit"), LOCKED_E3_COMMIT, "original E3 lock commit")
    _require(_json(output / "world_use_audit.json").get("commit"), LOCKED_E3_COMMIT, "history audit commit")
    _require(_json(output / "source_inventory.json").get("commit"), LOCKED_E3_COMMIT, "source audit commit")
    repair_commit = current_commit()
    if repair_commit == LOCKED_E3_COMMIT:
        raise ValueError("repair checkout is still the original E3 commit")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise ValueError("repair checkout must be clean")
    paths = _repair_diff_paths(LOCKED_E3_COMMIT, repair_commit)
    path = output / "lock_technical_repair.json"
    if path.exists():
        raise FileExistsError(f"preserve existing E3 technical repair amendment: {path}")
    amendment = {"status": "TECHNICAL_REPAIR", "original_lock_commit": LOCKED_E3_COMMIT,
                 "repair_commit": repair_commit, "original_locked_at_utc": locked["locked_at_utc"],
                 "created_at_utc": datetime.now(timezone.utc).isoformat(), "changed_paths": paths,
                 "reason": "Read training_source_commit from validated provenance; preserve existing pilot evaluations and original protocol lock"}
    _write(path, amendment)
    return amendment


def validate_lock(study_root: Path) -> dict:
    locked = _json(result_root(study_root) / "lock.json")
    _require(locked.get("status"), "LOCKED", "E3 lock status")
    _require(locked.get("protocol"), protocol(), "E3 protocol drift")
    if locked.get("implementation_commit") != current_commit():
        _require(locked.get("implementation_commit"), LOCKED_E3_COMMIT, "original E3 lock commit")
        amendment = _json(result_root(study_root) / "lock_technical_repair.json")
        for key, value in {"status": "TECHNICAL_REPAIR", "original_lock_commit": LOCKED_E3_COMMIT,
                           "repair_commit": current_commit(), "original_locked_at_utc": locked["locked_at_utc"],
                           "changed_paths": _repair_diff_paths(LOCKED_E3_COMMIT, current_commit())}.items():
            _require(amendment.get(key), value, f"E3 repair/{key}")
    return locked


def validate_cell(study_root: Path, cell_id: int, write_marker: bool = False,
                  source_array_job_id: str | None = None) -> dict:
    locked = validate_lock(study_root)
    item = cell(cell_id)
    source = _assert_source(study_root, item, check_payload=False)
    directory = eval_dir(study_root, item)
    complete, provenance = _json(directory / "EVAL_COMPLETE.json"), _json(directory / "provenance.json")
    expected = {"status": "complete", "algorithm": "mappo", "mappo_variant": item["value_configuration"],
                "actor_sharing": item["actor_structure"] == "shared",
                "actor_network_count": 1 if item["actor_structure"] == "shared" else 5,
                "training_seed": item["training_seed"], "training_run_name": item["training_run_name"],
                "scenario": "p05_n04_g25", "policy_name": "policy_final.pt", "policy_episode": 500,
                "eval_seeds": protocol()["candidate_worlds"], "eval_episodes": 100,
                "eval_warmup_episodes": 5, "eval_protocol": "sequential_warm",
                "mappo_eval_mode": "stochastic", "intervention_arm": "baseline",
                "is_frozen_eval": True, "policy_parameters_unchanged": True,
                "reproduction_git_dirty": False,
                "eval_id": eval_id(), "policy_path_is_relative_to_eval": True,
                "external_action_noise_applicable": False, "intervention_db": 0.0,
                "aoi_cap_ms": 100.0}
    for key, value in expected.items():
        _require(complete.get(key), value, f"{directory}/EVAL_COMPLETE/{key}")
    evaluation_commit = complete.get("reproduction_git_commit")
    if evaluation_commit not in {locked["implementation_commit"], current_commit()}:
        raise ValueError(f"{directory}: evaluation commit not original lock or approved repair: {evaluation_commit!r}")
    _require((directory / complete["policy"]).resolve(), Path(source["policy"]).resolve(), "policy reference")
    for key, value in {"algorithm": "mappo", "mappo_variant": item["value_configuration"],
                       "actor_sharing": item["actor_structure"] == "shared",
                       "training_seed": item["training_seed"], "eval_seeds": protocol()["candidate_worlds"],
                       "eval_episodes": 100, "eval_warmup_episodes": 5,
                       "mappo_eval_mode": "stochastic", "intervention_arm": "baseline",
                       "policy": complete["policy"],
                       "reproduction_git_commit": evaluation_commit,
                       "reproduction_git_dirty": False}.items():
        _require(provenance.get(key), value, f"{directory}/provenance/{key}")
    if not isinstance(provenance.get("created_at_utc"), str) or not provenance["created_at_utc"]:
        raise ValueError(f"{directory}: missing evaluation creation timestamp")
    _require(provenance.get("training_source_commit"), source["training_commit"], "training commit")
    if "training_source_commit" in complete:
        _require(complete["training_source_commit"], source["training_commit"], "evaluation training commit")
    arrays = {}
    with np.load(directory / "metrics.npz", allow_pickle=False) as npz:
        for name in ARRAY_FIELDS:
            if name not in npz.files:
                raise ValueError(f"{directory}: missing {name}")
            arrays[name] = npz[name]
    for name, array in arrays.items():
        shape = (6, 100, 100) if name == "reward_global" else AGENT_SHAPE
        _require(tuple(array.shape), shape, f"{directory}/{name} shape")
        axes = ["eval_seed", "scored_episode", "slot"] + ([] if name == "reward_global" else ["agent"])
        _require(complete.get("raw_metric_axes", {}).get(name), axes, f"{directory}/{name} axes")
        if not np.isfinite(array).all():
            raise ValueError(f"{directory}/{name}: non-finite values")
    if not np.isin(arrays["reset_event"], [0, 1]).all():
        raise ValueError(f"{directory}: reset_event is not boolean/0-1")
    reset = arrays["reset_event"].astype(bool)
    if not np.array_equal(reset, np.isclose(arrays["aoi_ms"], 1, rtol=0, atol=1e-6)):
        raise ValueError(f"{directory}: reset_event/aoi_ms mismatch")
    endpoint_cam = arrays["success"][:, :, -1, :]
    if not (np.isclose(endpoint_cam, 0.0) | np.isclose(endpoint_cam, 1.0)).all():
        raise ValueError(f"{directory}: non-binary endpoint CAM")
    marker = {"status": "complete", "experiment": "E3-locked-test", "cell_id": cell_id,
              "locked_at_utc": locked["locked_at_utc"], "implementation_commit": locked["implementation_commit"],
              "validation_commit": current_commit(),
              "evaluation_commit": complete["reproduction_git_commit"],
              "training_commit": source["training_commit"], "training_seed": item["training_seed"],
              "actor_structure": item["actor_structure"], "value_configuration": item["value_configuration"],
              "worlds": protocol()["candidate_worlds"], "eval_dir": str(directory),
              "action_rng_rule": "numpy.SeedSequence([training_seed, world, 0x4D415050]).generate_state(1,uint32)[0]",
              "action_rng_seeds": {str(world): int(np.random.SeedSequence(
                  [item["training_seed"], world, protocol()["evaluation"]["policy_rng_tag"]]
              ).generate_state(1, dtype=np.uint32)[0]) for world in protocol()["candidate_worlds"]}}
    path = directory / "E3_COMPLETE.json"
    if write_marker:
        if path.exists():
            raise FileExistsError(path)
        marker.update({"evaluation_created_at_utc": provenance.get("created_at_utc"),
                       "slurm_array_job_id": source_array_job_id or os.environ.get("SLURM_ARRAY_JOB_ID"),
                       "slurm_array_task_id": str(cell_id) if source_array_job_id else os.environ.get("SLURM_ARRAY_TASK_ID")})
        _write(path, marker)
    else:
        existing = _json(path)
        for key, value in marker.items():
            _require(existing.get(key), value, f"E3 cell marker/{key}")
        _require(existing.get("evaluation_created_at_utc"), provenance.get("created_at_utc"),
                 "E3 evaluation timestamp")
        marker = existing
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("history-audit", "source-preflight", "lock", "amend-lock",
                                            "check-lock", "check-cell", "mark-cell"))
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--cell-id", type=int)
    parser.add_argument("--confirm-reviewed", action="store_true")
    parser.add_argument("--supersede-blocked-audit", action="store_true")
    parser.add_argument("--confirm-technical-repair", action="store_true")
    parser.add_argument("--source-array-job-id")
    args = parser.parse_args()
    if args.command == "history-audit":
        result = history_audit(args.study_root, args.supersede_blocked_audit)
    elif args.command == "source-preflight":
        result = source_preflight(args.study_root)
    elif args.command == "lock":
        result = create_lock(args.study_root, args.confirm_reviewed)
    elif args.command == "amend-lock":
        result = amend_lock_for_technical_repair(args.study_root, args.confirm_technical_repair)
    elif args.command == "check-lock":
        locked = validate_lock(args.study_root)
        result = {k: locked[k] for k in ("status", "locked_at_utc", "implementation_commit")}
    else:
        if args.cell_id is None:
            parser.error("--cell-id required")
        if args.source_array_job_id is not None and (args.command != "mark-cell" or not args.source_array_job_id.isdecimal()):
            parser.error("--source-array-job-id requires mark-cell and a numeric Slurm array job id")
        result = validate_cell(args.study_root, args.cell_id, args.command == "mark-cell",
                               args.source_array_job_id)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 2 if args.command == "history-audit" and result["status"] != "CANDIDATE_CLEAR_REQUIRES_MANUAL_REVIEW" else 0


if __name__ == "__main__":
    raise SystemExit(main())
