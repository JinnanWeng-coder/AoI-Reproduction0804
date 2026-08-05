"""Restart-safe 48-run train/eval/audit matrix orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    CHECKPOINT_SCHEMA_VERSION,
    config_from_dict,
    matrix_specs,
    resolve_config,
    safe_run_dir,
)


_IDENTITY_FIELDS = ("profile", "scenario", "seed", "run_name", "output_root", "device")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _resolved_item(item, args) -> dict:
    """Resolve exactly the config encoded by :func:`_command`.

    Matrix specifications are intentionally independent of a machine.  Device
    selection and the output root are supplied by the remote invocation, so a
    hash produced by ``config.matrix_specs`` cannot be used for recovery until
    those arguments have been applied.
    """
    config = resolve_config(
        profile=str(item["profile"]),
        scenario=str(item["scenario"]),
        seed=int(item["seed"]),
        device=str(args.device),
        run_name=str(item["run_name"]),
        output_root=str(args.output_root),
        checkpoint_every=int(getattr(args, "checkpoint_every", None) or 50),
    )
    resolved = dict(item)
    resolved.update(
        {
            "profile": config.profile,
            "scenario": config.scenario.id,
            "seed": int(config.seed),
            "run_name": config.run_name,
            "output_root": config.output_root,
            "device": config.device,
            "semantic_version": config.semantic_version,
            "mobility_revision": config.mobility_revision,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "config_hash": config.canonical_hash(),
            # Private-by-convention field used by strict recovery validation.
            # It remains JSON serialisable so dry-run reports are portable.
            "_expected_config": config.to_dict(),
        }
    )
    return resolved


def _matrix_specs_for_args(args) -> list:
    return [_resolved_item(item, args) for item in matrix_specs(args.profile)]


def _matrix_shard(specs: list, shard_count: int, shard_index: int) -> list:
    """Return one deterministic, disjoint operational shard of the matrix."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if shard_count > len(specs):
        raise ValueError("shard_count cannot exceed the full matrix size")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    return [item for offset, item in enumerate(specs) if offset % shard_count == shard_index]


def _identity_from_dict(config: dict) -> dict:
    scenario = config.get("scenario")
    scenario_id = scenario.get("id") if isinstance(scenario, dict) else scenario
    return {
        "profile": config.get("profile"),
        "scenario": scenario_id,
        "seed": config.get("seed"),
        "run_name": config.get("run_name"),
        "output_root": config.get("output_root"),
        "device": config.get("device"),
    }


def _canonical_dict_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(item, args, stage: str, resume: Path = None):
    item = _resolved_item(item, args)
    command = [
        sys.executable,
        str(ROOT / "Main.py"),
        "--profile",
        item["profile"],
        "--scenario",
        item["scenario"],
        "--seed",
        str(item["seed"]),
        "--device",
        item["device"],
        "--run-name",
        item["run_name"],
        "--output-root",
        item["output_root"],
        "--checkpoint-every",
        str(item["_expected_config"]["checkpoint_every"]),
    ]
    if stage == "train":
        command.extend(["--scope", "train"])
        if resume is None and bool(getattr(args, "recover_empty_run", False)):
            command.append("--recover-empty-run")
        if resume is not None:
            command.extend(["--resume", str(resume)])
    elif stage == "eval":
        if resume is None:
            raise ValueError("eval command requires a checkpoint")
        scope = "validation" if args.eval_purpose == "validation" else "final_release"
        command.extend(["--scope", scope, "--eval-only", "--eval-purpose", args.eval_purpose, "--eval-episodes", str(args.eval_episodes), "--eval-seeds", args.eval_seeds])
        command.extend(["--resume", str(resume)])
    else:
        raise ValueError(f"unsupported command stage: {stage}")
    return command


def _checkpoint_error(prefix: str, suffix: str, run_dir: Path, **details) -> dict:
    return {"code": f"{prefix}_{suffix}", "action": "error", "run": str(run_dir), **details}


def _validate_checkpoint_payload(payload, item, run_dir: Path, prefix: str):
    """Return a structured error, or ``None`` for a matching checkpoint."""
    if not isinstance(payload, dict):
        return _checkpoint_error(prefix, "PAYLOAD_INVALID", run_dir, payload_type=type(payload).__name__)

    # Keep compatibility for callers that pass a raw config.matrix_specs item.
    # The executable path always supplies the strict, machine-resolved item.
    strict = isinstance(item.get("_expected_config"), dict)
    if strict and payload.get("checkpoint_version") != 4:
        return _checkpoint_error(
            prefix,
            "VERSION_MISMATCH",
            run_dir,
            checkpoint_version=payload.get("checkpoint_version"),
            expected_checkpoint_version=4,
        )
    if payload.get("semantic_version") != item.get("semantic_version"):
        return _checkpoint_error(
            prefix,
            "SEMANTIC_MISMATCH",
            run_dir,
            checkpoint_semantic_version=payload.get("semantic_version"),
            expected_semantic_version=item.get("semantic_version"),
        )
    if not strict:
        if payload.get("config_hash") != item.get("config_hash"):
            return _checkpoint_error(
                prefix,
                "CONFIG_MISMATCH",
                run_dir,
                checkpoint_config_hash=payload.get("config_hash"),
                expected_config_hash=item.get("config_hash"),
            )
        return None

    if payload.get("mobility_revision") != item.get("mobility_revision"):
        return _checkpoint_error(
            prefix,
            "MOBILITY_MISMATCH",
            run_dir,
            checkpoint_mobility_revision=payload.get("mobility_revision"),
            expected_mobility_revision=item.get("mobility_revision"),
        )
    if payload.get("checkpoint_schema_version") != item.get("checkpoint_schema_version"):
        return _checkpoint_error(
            prefix,
            "SCHEMA_MISMATCH",
            run_dir,
            checkpoint_schema_version=payload.get("checkpoint_schema_version"),
            expected_checkpoint_schema_version=item.get("checkpoint_schema_version"),
        )

    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        return _checkpoint_error(prefix, "CONFIG_MISSING", run_dir)
    expected_identity = _identity_from_dict(item["_expected_config"])
    actual_identity = _identity_from_dict(raw_config)
    mismatches = {
        key: {"checkpoint": actual_identity[key], "expected": expected_identity[key]}
        for key in _IDENTITY_FIELDS
        if actual_identity[key] != expected_identity[key]
    }
    if mismatches:
        return _checkpoint_error(prefix, "IDENTITY_MISMATCH", run_dir, mismatches=mismatches)

    try:
        embedded_hash = _canonical_dict_hash(raw_config)
    except (TypeError, ValueError) as exc:
        return _checkpoint_error(prefix, "CONFIG_INVALID", run_dir, error=str(exc))
    if payload.get("config_hash") != embedded_hash:
        return _checkpoint_error(
            prefix,
            "EMBEDDED_CONFIG_HASH_MISMATCH",
            run_dir,
            checkpoint_config_hash=payload.get("config_hash"),
            embedded_config_hash=embedded_hash,
        )
    if payload.get("config_hash") != item.get("config_hash"):
        return _checkpoint_error(
            prefix,
            "CONFIG_MISMATCH",
            run_dir,
            checkpoint_config_hash=payload.get("config_hash"),
            expected_config_hash=item.get("config_hash"),
        )
    return None


def _validate_complete_marker(path: Path, item, run_dir: Path, checkpoint_hashes: dict):
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _checkpoint_error("COMPLETE_MARKER", "READ_ERROR", run_dir, error=str(exc))
    if not isinstance(marker, dict):
        return _checkpoint_error("COMPLETE_MARKER", "INVALID", run_dir, marker_type=type(marker).__name__)

    expected = {
        "status": "complete",
        "profile": item["profile"],
        "scenario": item["scenario"],
        "seed": int(item["seed"]),
        "episodes": int(item["_expected_config"]["episodes"]),
        "final_episode": int(item["_expected_config"]["episodes"]),
        "checkpoint_completed": True,
        "checkpoint_sha256": checkpoint_hashes,
        "config_hash": item["config_hash"],
        "semantic_version": item["semantic_version"],
        "mobility_revision": item["mobility_revision"],
        "checkpoint_schema_version": item["checkpoint_schema_version"],
    }
    mismatches = {
        key: {"marker": marker.get(key), "expected": value}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        return _checkpoint_error("COMPLETE_MARKER", "MISMATCH", run_dir, mismatches=mismatches)
    return None


def _recovery_state(run_dir: Path, item, recover_empty_run: bool = False) -> dict:
    """Describe an existing run without deleting or overwriting anything."""
    latest = run_dir / "checkpoints" / "latest.pt"
    complete = run_dir / "COMPLETE.json"
    strict = isinstance(item.get("_expected_config"), dict)
    if not run_dir.exists():
        return {"code": "NEW_RUN", "action": "create", "run": str(run_dir)}
    if not latest.is_file():
        code = "COMPLETE_WITHOUT_CHECKPOINT" if complete.is_file() else "RUN_DIR_WITHOUT_CHECKPOINT"
        if not complete.is_file() and recover_empty_run and strict:
            try:
                from runner import EmptyRunRecoveryError, _validate_empty_run_for_reinitialization

                config = config_from_dict(item["_expected_config"])
                verification = _validate_empty_run_for_reinitialization(run_dir, config)
            except EmptyRunRecoveryError as exc:
                return {
                    "code": f"EMPTY_RUN_{exc.code}",
                    "action": "error",
                    "run": str(run_dir),
                    "error": str(exc),
                    **exc.details,
                }
            except Exception as exc:
                return {
                    "code": "EMPTY_RUN_VALIDATION_ERROR",
                    "action": "error",
                    "run": str(run_dir),
                    "error": str(exc),
                }
            return {
                "code": "EMPTY_RUN_REINITIALIZE_AVAILABLE",
                "action": "reinitialize",
                "run": str(run_dir),
                "verification": verification,
            }
        if not complete.is_file() and strict:
            code = "EMPTY_RUN_RECOVERY_NOT_ENABLED"
        return {"code": code, "action": "error", "run": str(run_dir)}
    try:
        import torch

        payload = torch.load(latest, map_location="cpu", weights_only=False)
    except Exception as exc:
        prefix = "COMPLETE_CHECKPOINT" if complete.is_file() else "CHECKPOINT"
        return _checkpoint_error(prefix, "READ_ERROR", run_dir, error=str(exc))

    prefix = "COMPLETE_CHECKPOINT" if complete.is_file() else "CHECKPOINT"
    validation_error = _validate_checkpoint_payload(payload, item, run_dir, prefix)
    if validation_error is not None:
        return validation_error

    if complete.is_file():
        best = run_dir / "checkpoints" / "best.pt"
        if strict and not best.is_file():
            return _checkpoint_error("COMPLETE_CHECKPOINT", "BEST_MISSING", run_dir, checkpoint=str(best))
        if strict and payload.get("completed") is not True:
            return _checkpoint_error(prefix, "NOT_COMPLETED", run_dir, completed=payload.get("completed"))
        if strict:
            expected_episode = int(item["_expected_config"]["episodes"])
            if payload.get("episode") != expected_episode:
                return _checkpoint_error(
                    prefix,
                    "EPISODE_MISMATCH",
                    run_dir,
                    checkpoint_episode=payload.get("episode"),
                    expected_episode=expected_episode,
                )
            try:
                import torch

                best_payload = torch.load(best, map_location="cpu", weights_only=False)
            except Exception as exc:
                return _checkpoint_error("COMPLETE_BEST_CHECKPOINT", "READ_ERROR", run_dir, error=str(exc))
            best_error = _validate_checkpoint_payload(best_payload, item, run_dir, "COMPLETE_BEST_CHECKPOINT")
            if best_error is not None:
                return best_error
            if best_payload.get("completed") is not True:
                return _checkpoint_error(
                    "COMPLETE_BEST_CHECKPOINT",
                    "NOT_COMPLETED",
                    run_dir,
                    completed=best_payload.get("completed"),
                )
            selected_best = best_payload.get("checkpoint_role") == "best_selection_validation"
            if selected_best:
                selected_episode = int(best_payload.get("selected_episode", -1))
                checkpoint_episode = int(best_payload.get("episode", -2))
                if (
                    best_payload.get("training_completed") is not True
                    or selected_episode != checkpoint_episode
                    or not 1 <= checkpoint_episode <= expected_episode
                    or not isinstance(best_payload.get("selection_validation"), dict)
                ):
                    return _checkpoint_error(
                        "COMPLETE_BEST_CHECKPOINT",
                        "SELECTION_METADATA_INVALID",
                        run_dir,
                        checkpoint_episode=best_payload.get("episode"),
                        selected_episode=best_payload.get("selected_episode"),
                    )
            elif best_payload.get("episode") != expected_episode:
                return _checkpoint_error(
                    "COMPLETE_BEST_CHECKPOINT",
                    "EPISODE_MISMATCH",
                    run_dir,
                    checkpoint_episode=best_payload.get("episode"),
                    expected_episode=expected_episode,
                )
            checkpoint_hashes = {
                "latest.pt": _checkpoint_hash(latest),
                "best.pt": _checkpoint_hash(best),
            }
            marker_error = _validate_complete_marker(complete, item, run_dir, checkpoint_hashes)
            if marker_error is not None:
                return marker_error
        else:
            checkpoint_hashes = {"latest.pt": _checkpoint_hash(latest)}
        return {
            "code": "COMPLETE",
            "action": "skip",
            "run": str(run_dir),
            "checkpoint": str(latest),
            "checkpoint_sha256": checkpoint_hashes["latest.pt"],
            "checkpoint_sha256s": checkpoint_hashes,
        }

    if payload.get("completed") is True:
        return {
            "code": "CHECKPOINT_COMPLETE_WITHOUT_MARKER",
            "action": "error",
            "run": str(run_dir),
            "checkpoint": str(latest),
        }
    if strict:
        expected_episode = int(item["_expected_config"]["episodes"])
        episode = payload.get("episode")
        if not isinstance(episode, int) or isinstance(episode, bool) or not 0 <= episode < expected_episode:
            return _checkpoint_error(
                "CHECKPOINT",
                "PROGRESS_INVALID",
                run_dir,
                checkpoint_episode=episode,
                expected_episode_range=[0, expected_episode - 1],
            )
    return {
        "code": "INCOMPLETE_RESUME_AVAILABLE",
        "action": "resume",
        "run": str(run_dir),
        "checkpoint": str(latest),
        "checkpoint_sha256": _checkpoint_hash(latest),
    }


def _run(command, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"exit={completed.returncode}\n")
    return completed.returncode


def _audit_command(run_dir: Path, args) -> list:
    audit_scope = "validation" if args.eval_purpose == "validation" else "final_release"
    return [
        sys.executable,
        str(ROOT / "analysis" / "audit_results.py"),
        str(run_dir),
        "--scope",
        audit_scope,
        "--require-eval",
    ]


def _stage_recovery_error(stage: str, item, recovery: dict) -> dict:
    return {
        "code": f"{stage.upper()}_REQUIRES_COMPLETED_RUN",
        "stage": stage,
        "run": item["run_name"],
        "recovery": recovery,
    }


def _has_matching_eval(run_dir: Path, checkpoint_hash: str, args, item=None) -> bool:
    eval_root = run_dir / "eval"
    if not eval_root.is_dir():
        return False
    expected_seeds = [int(token) for token in args.eval_seeds.split(",") if token.strip()]
    for child in eval_root.iterdir():
        summary_path = child / "summary.json"
        complete_path = child / "EVAL_COMPLETE.json"
        if not summary_path.is_file() or not complete_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        expected_scope = "validation" if args.eval_purpose == "validation" else "final_release"
        matches = (
            summary.get("status") == "complete"
            and complete == summary
            and summary.get("checkpoint_sha256") == checkpoint_hash
            and summary.get("eval_episodes") == args.eval_episodes
            and summary.get("eval_seeds") == expected_seeds
            and summary.get("eval_purpose") == args.eval_purpose
            and summary.get("scope") == expected_scope
        )
        if matches and item is not None:
            matches = all(
                (
                    summary.get("profile") == item["profile"],
                    summary.get("scenario") == item["scenario"],
                    summary.get("training_seed") == int(item["seed"]),
                    summary.get("config_hash") == item["config_hash"],
                    summary.get("semantic_version") == item["semantic_version"],
                    summary.get("mobility_revision") == item["mobility_revision"],
                    summary.get("checkpoint_schema_version") == item["checkpoint_schema_version"],
                )
            )
        if matches:
            return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="paper_faithful")
    parser.add_argument("--stage", choices=("train", "eval", "audit", "all"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="experiments/runs")
    parser.add_argument("--checkpoint-every", type=_positive_int, default=5)
    parser.add_argument(
        "--recover-empty-run",
        action="store_true",
        help="explicitly reinitialize a provenance-verified run stopped before its first checkpoint",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seeds", default=None)
    parser.add_argument("--eval-purpose", choices=("validation", "final_test"), default="validation")
    parser.add_argument("--log-dir", default="batch_logs")
    parser.add_argument("--report", default=None, help="write a JSON dry-run/execution report")
    parser.add_argument(
        "--shard-count",
        type=_positive_int,
        default=1,
        help="split the complete 48-cell matrix into this many deterministic, disjoint workers",
    )
    parser.add_argument(
        "--shard-index",
        type=_non_negative_int,
        default=0,
        help="zero-based worker index within --shard-count",
    )
    args = parser.parse_args(argv)
    if args.eval_seeds is None:
        args.eval_seeds = "201,202,203,204,205,206" if args.eval_purpose == "validation" else "101,102,103,104,105,106"
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    full_specs = _matrix_specs_for_args(args)
    full_keys = {(item["profile"], item["scenario"], item["seed"]) for item in full_specs}
    if len(full_specs) != 48 or len(full_keys) != 48:
        raise RuntimeError("full matrix must contain exactly 48 unique tasks")
    try:
        specs = _matrix_shard(full_specs, args.shard_count, args.shard_index)
    except ValueError as exc:
        parser.error(str(exc))
    keys = {(item["profile"], item["scenario"], item["seed"]) for item in specs}
    print(json.dumps(specs, indent=2, sort_keys=True))
    print(f"full_matrix_count={len(full_specs)}")
    print(f"matrix_count={len(specs)}")
    print(f"unique_count={len(keys)}")
    if len(specs) != len(keys):
        raise RuntimeError("selected matrix shard contains duplicate tasks")
    if args.dry_run:
        report_commands = []
        for item in specs:
            run_dir = safe_run_dir(item["output_root"], item["run_name"])
            latest = run_dir / "checkpoints" / "latest.pt"
            train_command = _command(item, args, "train", latest if latest.exists() else None)
            eval_command = _command(item, args, "eval", latest)
            audit_scope = "validation" if args.eval_purpose == "validation" else "final_release"
            audit_command = _audit_command(run_dir, args)
            print("TRAIN", " ".join(train_command))
            print("EVAL", " ".join(eval_command))
            print("AUDIT", " ".join(audit_command))
            report_commands.append({"run_name": item["run_name"], "train": train_command, "eval": eval_command, "audit": audit_command, "scope": audit_scope, "eval_purpose": args.eval_purpose, "checkpoint_every": args.checkpoint_every, "recover_empty_run": bool(args.recover_empty_run)})
        for command_name in ("train", "eval", "audit"):
            unique_commands = {tuple(entry[command_name]) for entry in report_commands}
            if len(unique_commands) != len(specs):
                raise RuntimeError(f"matrix shard contains duplicate {command_name} commands")
        if args.report:
            report_path = Path(args.report).expanduser()
            if not report_path.is_absolute():
                report_path = ROOT / report_path
            if report_path.exists():
                raise FileExistsError(f"Refusing to overwrite matrix report: {report_path}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({"status": "dry_run", "profile": args.profile, "semantic_version": specs[0]["semantic_version"], "eval_purpose": args.eval_purpose, "scope": "validation" if args.eval_purpose == "validation" else "final_release", "checkpoint_every": args.checkpoint_every, "recover_empty_run": bool(args.recover_empty_run), "full_matrix_count": len(full_specs), "matrix_count": len(specs), "unique_count": len(keys), "shard_count": args.shard_count, "shard_index": args.shard_index, "specs": specs, "commands": report_commands}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    stages = ("train", "eval", "audit") if args.stage == "all" else (args.stage,)
    failures = []
    recovery_events = []
    log_dir = Path(args.log_dir).expanduser()
    for item in specs:
        run_dir = safe_run_dir(item["output_root"], item["run_name"])
        latest = run_dir / "checkpoints" / "latest.pt"
        log_path = log_dir / f"{item['run_name']}.log"
        for stage in stages:
            if stage == "train":
                recovery = _recovery_state(run_dir, item, recover_empty_run=args.recover_empty_run)
                recovery_events.append({"stage": stage, "phase": "before", **recovery})
                if recovery["action"] == "skip":
                    print(f"SKIP complete {item['run_name']}")
                    continue
                if recovery["action"] == "error":
                    failures.append({"stage": stage, "run": item["run_name"], "recovery": recovery})
                    break
                resume = latest if recovery["action"] == "resume" else None
                if _run(_command(item, args, "train", resume), log_path) != 0:
                    failures.append({"stage": stage, "run": item["run_name"]})
                    break
                recovery = _recovery_state(run_dir, item, recover_empty_run=args.recover_empty_run)
                recovery_events.append({"stage": stage, "phase": "after", **recovery})
                if recovery["action"] != "skip":
                    failures.append(
                        {
                            "code": "TRAIN_DID_NOT_PRODUCE_COMPLETED_RUN",
                            "stage": stage,
                            "run": item["run_name"],
                            "recovery": recovery,
                        }
                    )
                    break
            elif stage == "eval":
                recovery = _recovery_state(run_dir, item, recover_empty_run=args.recover_empty_run)
                recovery_events.append({"stage": stage, "phase": "before", **recovery})
                if recovery["action"] != "skip":
                    failures.append(_stage_recovery_error(stage, item, recovery))
                    break
                checkpoint_hash = recovery.get("checkpoint_sha256") or _checkpoint_hash(latest)
                if _has_matching_eval(run_dir, checkpoint_hash, args, item):
                    print(f"SKIP eval {item['run_name']}")
                    continue
                if _run(_command(item, args, "eval", latest), log_path) != 0:
                    failures.append({"stage": stage, "run": item["run_name"]})
                    break
                if not _has_matching_eval(run_dir, checkpoint_hash, args, item):
                    failures.append(
                        {
                            "code": "EVAL_ARTIFACT_MISSING_OR_MISMATCH",
                            "stage": stage,
                            "run": item["run_name"],
                            "checkpoint_sha256": checkpoint_hash,
                            "eval_purpose": args.eval_purpose,
                        }
                    )
                    break
            else:
                recovery = _recovery_state(run_dir, item, recover_empty_run=args.recover_empty_run)
                recovery_events.append({"stage": stage, "phase": "before", **recovery})
                if recovery["action"] != "skip":
                    failures.append(_stage_recovery_error(stage, item, recovery))
                    break
                if _run(_audit_command(run_dir, args), log_path) != 0:
                    failures.append({"stage": stage, "run": item["run_name"]})
                    break
    report = {"status": "failed" if failures else "complete", "failures": failures, "recovery_events": recovery_events, "eval_purpose": args.eval_purpose, "checkpoint_every": args.checkpoint_every, "recover_empty_run": bool(args.recover_empty_run), "full_matrix_count": len(full_specs), "matrix_count": len(specs), "unique_count": len(keys), "shard_count": args.shard_count, "shard_index": args.shard_index}
    if args.report:
        report_path = Path(args.report).expanduser()
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite matrix report: {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
