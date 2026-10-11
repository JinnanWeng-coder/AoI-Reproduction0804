"""D1 frozen deployment manifests, source checks and pre-performance world lock."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from analysis.deployment_rng import action_seed_words, policy_seed

BASELINE = "09c38b48e98bae66f2c4ade3e3dee2f7b5d44a54"
SEEDS = list(range(8, 14))
SCENARIO = "p05_n04_g25"
PROTOCOL = {"name": "sequential_warm", "warmup_episodes": 5,
            "scored_episodes": 100, "slots": 100, "platoons": 5,
            "training_episodes": 500, "training_environment_steps": 50000}
MODELS = {
    "shared_tdec": {"family": "mappo", "structure": "shared", "variant": "tdec"},
    "independent_tdec": {"family": "mappo", "structure": "independent", "variant": "tdec"},
    "shared_combined": {"family": "mappo", "structure": "shared", "variant": "combined"},
    "algorithm2": {"family": "maddpg", "structure": "independent", "variant": "tdec"},
}
CONDITIONS = {
    "shared_tdec_stochastic": ("shared_tdec", "stochastic", None, 3),
    "independent_tdec_stochastic": ("independent_tdec", "stochastic", None, 3),
    "shared_combined_stochastic": ("shared_combined", "stochastic", None, 3),
    "algorithm2_noise0": ("algorithm2", "deterministic", 0.0, 1),
    "algorithm2_noise0p3": ("algorithm2", "noisy", 0.3, 3),
    "shared_tdec_deterministic": ("shared_tdec", "deterministic", None, 1),
    "independent_tdec_deterministic": ("independent_tdec", "deterministic", None, 1),
}
# Fields read by PaperEnviron: a different effective environment must not silently
# masquerade as an action repeat. Algorithm and optimization fields are excluded.
ENV_FIELDS = ("scenario", "steps_per_episode", "slot_ms", "slow_fading_ms", "n_rb", "n_modes",
              "bandwidth_hz", "cam_bits", "power_min_dbm", "power_max_dbm", "v2i_min_bps_per_hz",
              "slow_update_every_episodes", "map_width_m", "map_height_m", "rsu_position",
              "speed_min_mps", "speed_max_mps", "speed_distribution", "power_continuous",
              "previous_interference_dim", "include_remaining_time", "initial_aoi_ms",
              "current_interference_reward", "global_reward_normalization", "mobility_model",
              "mobility_revision", "gap_definition", "vehicle_length_m")
WORLD_FIELDS = {"world", "worlds", "world_ids", "eval_world", "eval_worlds", "eval_seed", "eval_seeds",
                "world_seed", "world_seeds", "environment_seed", "environment_seeds", "training_seed",
                "selection_validation_seeds", "validation_seeds",
                "candidate_worlds", "locked_worlds"}
RESERVED = set(SEEDS + list(range(101, 107)) + list(range(151, 157)) +
               list(range(201, 219)) + list(range(301, 309)))
REPO = Path(__file__).resolve().parents[1]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def code_check(expected=None):
    commit = git("rev-parse", "HEAD")
    require(git("branch", "--show-current") in {"research/mappo", ""}, "wrong execution branch")
    require(not git("status", "--porcelain", "--untracked-files=all"), "dirty execution checkout")
    require(expected is None or expected == commit, "implementation commit changed")
    require(not git("diff", BASELINE, "HEAD", "--", "aoi_v2x_reproduction"),
            "algorithm/environment/config differs from the approved baseline")
    return commit


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    require(path.is_file() and stat.st_size > 0, f"missing/empty source: {path}")
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_entry(model_id, seed, checkpoint):
    from aoi_v2x_reproduction.config import config_from_dict
    model = MODELS[model_id]
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent if model["family"] == "mappo" else checkpoint.parent.parent
    raw, complete = read(run / "config.resolved.json"), read(run / "COMPLETE.json")
    cfg = config_from_dict(raw)
    require(cfg.seed == seed and cfg.scenario.id == SCENARIO and cfg.n_rb == 3 and
            cfg.number_agents == 5 and cfg.scenario.platoon_size == 4 and
            cfg.episodes == 500 and cfg.steps_per_episode == 100, f"source contract: {run}")
    require(cfg.eval_protocol == "sequential_warm" and cfg.eval_warmup_episodes == 5,
            f"source evaluation protocol: {run}")
    require(complete.get("status") == "complete" and
            int(complete.get("final_episode", complete.get("episodes", -1))) == 500,
            f"source training incomplete: {run}")
    require(bool(complete.get("reproduction_git_commit")), f"missing training commit: {run}")
    if model["family"] == "mappo":
        require(checkpoint.name == "policy_final.pt" and cfg.algorithm == "mappo" and
                cfg.mappo_variant == model["variant"] and
                bool(cfg.mappo_actor_sharing) == (model["structure"] == "shared"), f"wrong MAPPO source: {run}")
    else:
        require(checkpoint.name == "latest.pt" and cfg.algorithm == "modified_maddpg_tdec",
                f"Algorithm 2 latest.pt required: {run}")
    effective = cfg.to_dict()
    return {"model_id": model_id, "training_seed": seed, **model,
            "checkpoint": file_identity(checkpoint), "run_dir": str(run),
            "config": raw, "complete": complete, "source_config_hash": cfg.canonical_hash(),
            "training_commit": complete["reproduction_git_commit"],
            "environment_config": {k: effective[k] for k in ENV_FIELDS},
            "sharing_raw": raw.get("mappo_actor_sharing"),
            "sharing_effective": bool(cfg.mappo_actor_sharing) if model["family"] == "mappo" else False}


def default_sources(diagnostics):
    from analysis.e1_contract import (resolve_maddpg_source_run, mappo_training_root, mappo_run_name)
    result = []
    for model_id, model in MODELS.items():
        for seed in SEEDS:
            if model["family"] == "mappo":
                checkpoint = (mappo_training_root(diagnostics, model["structure"], model["variant"]) /
                              "training/runs" / mappo_run_name(model["structure"], model["variant"], seed) / "policy_final.pt")
            else:
                checkpoint = resolve_maddpg_source_run(diagnostics, "modified_maddpg_tdec", seed)["checkpoint"]
            result.append(source_entry(model_id, seed, checkpoint))
    return result


def source_check(source, payload=False):
    current = source_entry(source["model_id"], source["training_seed"], source["checkpoint"]["path"])
    require(current == source, f"source identity/config/COMPLETE changed: {source['run_dir']}")
    if payload:
        import torch
        data = torch.load(source["checkpoint"]["path"], map_location="cpu", weights_only=False)
        if source["family"] == "mappo":
            from aoi_v2x_reproduction.runtime.runner import _validate_mappo_policy_artifact
            _, cfg, _ = _validate_mappo_policy_artifact(Path(source["checkpoint"]["path"]), data)
            require(data.get("episode") == 500 and cfg.canonical_hash() == source["source_config_hash"], "MAPPO final500 payload")
        else:
            from analysis.e1_contract import validate_latest500_payload
            validate_latest500_payload(data, "modified_maddpg_tdec", source["training_seed"])
            require(data.get("config_hash") == source["source_config_hash"], "MADDPG config hash")


def phase_plan(phase, worlds=None):
    require(phase in {"pilot", "main", "gap-transfer"}, "unknown phase")
    if phase == "pilot":
        seeds, worlds, conditions, targets = [8, 9], [213, 214], list(CONDITIONS), [SCENARIO]
    elif phase == "main":
        require(worlds is not None and len(worlds) == len(set(worlds)) == 30 and
                all(isinstance(w, int) and w >= 0 for w in worlds) and not set(worlds) & RESERVED,
                "main requires 30 unique locked new worlds outside reserved history")
        seeds, conditions, targets = SEEDS, list(CONDITIONS)[:5], [SCENARIO]
    else:
        seeds, worlds, conditions, targets = SEEDS, list(range(213, 219)), list(CONDITIONS)[3:5], ["p05_n04_g05", "p05_n04_g35"]
    units = []
    for target in targets:
        for condition in conditions:
            model_id, execution, noise, repeats = CONDITIONS[condition]
            for seed in seeds:
                for world in worlds:
                    for repeat in range(1 if phase == "gap-transfer" else repeats):
                        family = MODELS[model_id]["family"]
                        units.append({"unit_id": len(units), "phase": phase, "condition": condition,
                                      "model_id": model_id, "training_seed": seed, "target_scenario": target,
                                      "world": world, "repeat": repeat, "execution": execution, "noise_std": noise,
                                      "action_seed_words": action_seed_words(seed, world, repeat, family),
                                      "policy_sampling_seed": policy_seed(seed, world, repeat) if family == "mappo" else None,
                                      "environment_seed": world})
    return {"phase": phase, "worlds": list(worlds), "training_seeds": list(seeds),
            "policy_world_repeat_count": len(units), "platoon_row_count": len(units) * 5,
            "observational_extension": phase == "gap-transfer", "units": units}


def pack_inventory(path):
    """Read compact inventory only, never checkpoint or trajectory samples."""
    with zipfile.ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if n.endswith("/inventory/runs.csv")]
        require(len(matches) == 1, "evidence pack run inventory missing/ambiguous")
        import io
        rows = list(csv.DictReader(io.StringIO(archive.read(matches[0]).decode("utf-8-sig"))))
    return rows


def init(diagnostics, batch_id, models_json=None, evidence_pack=None, addendum=None):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id), "unsafe batch ID")
    commit = code_check()
    diagnostics = Path(diagnostics).resolve()
    batch = diagnostics / "actor-sharing-study/D1-frozen-deployment/P5_N4_gap25" / batch_id
    require(not batch.exists(), f"batch exists: {batch}")
    if models_json:
        supplied = read(models_json)
        sources = [source_entry(x["model_id"], int(x["training_seed"]), x["checkpoint"]) for x in supplied]
    else:
        sources = default_sources(diagnostics)
    require({(s["model_id"], s["training_seed"]) for s in sources} ==
            {(m, seed) for m in MODELS for seed in SEEDS} and len(sources) == 24, "exactly 24 source models required")
    require(len({s["checkpoint"]["path"] for s in sources}) == 24, "duplicate source model")
    require(all(s["environment_config"] == sources[0]["environment_config"] for s in sources),
            "effective source environments differ; cannot attribute repeats to actions alone")
    packs = {}
    if evidence_pack:
        rows = pack_inventory(evidence_pack)
        for source in sources:
            # The initial export sometimes picked an actor-only periodic file
            # as policy_path for MADDPG. E1/addendum specify latest.pt instead.
            # Match the training run, then verify the actual latest500 payload.
            matches = [r for r in rows if Path(r["run_dir"]).as_posix() == Path(source["run_dir"]).as_posix()]
            require(len(matches) == 1 and int(matches[0]["checkpoint_episode"]) == 500,
                    f"source not uniquely represented in evidence pack: {source['run_dir']}")
        packs["evidence_pack"] = file_identity(evidence_pack)
    if addendum:
        with zipfile.ZipFile(addendum) as archive:
            import io
            names = [n for n in archive.namelist() if n.endswith("/e1/checkpoint_config_map.csv")]
            require(len(names) == 1, "metadata addendum E1 checkpoint map missing")
            mappings = list(csv.DictReader(io.StringIO(archive.read(names[0]).decode("utf-8-sig"))))
            for source in sources:
                if source["family"] != "maddpg":
                    continue
                matches = [r for r in mappings if r["checkpoint_recorded"] == source["checkpoint"]["path"] and
                           r["checkpoint_name_recorded"] == "latest.pt" and int(r["checkpoint_episode_recorded"]) == 500]
                require(len(matches) == 2 and {float(r["noise_std_recorded"]) for r in matches} == {0., .3},
                        f"Algorithm 2 latest500 mapping absent/ambiguous in addendum: {source['run_dir']}")
        packs["metadata_addendum"] = file_identity(addendum)
    batch.mkdir(parents=True, exist_ok=False)
    for name in ("protocol", "evaluations", "slurm_logs", "analysis", "tmp", "cache"):
        (batch / name).mkdir()
    write(batch / "protocol/manifest.json", {"schema": "deployment_v1", "implementation_commit": commit,
          "baseline": BASELINE, "diagnostics_root": str(diagnostics), "batch_root": str(batch),
          "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL,
          "models": MODELS, "conditions": CONDITIONS, "input_packs": packs,
          "statistics": "training seed is inferential unit; scene-first pools repeats before own worst platoon",
          "claim_support": "NOT_ASSESSED"})
    write(batch / "protocol/sources.json", sources)
    for phase in ("pilot", "gap-transfer"):
        write(batch / f"protocol/{phase}.json", phase_plan(phase))
    return {"batch_root": str(batch), "status": "METADATA_READY", "pilot_units": 60,
            "main_units_after_lock": 2340, "gap_transfer_units": 144, "source_models": 24}


def load_batch(batch, check_code=True):
    batch = Path(batch).resolve()
    manifest = read(batch / "protocol/manifest.json")
    require(manifest["batch_root"] == str(batch) and manifest["protocol"] == PROTOCOL and
            manifest["models"] == MODELS and manifest["conditions"] == json.loads(json.dumps(CONDITIONS)), "batch manifest contract changed")
    expected_parent = Path(manifest["diagnostics_root"]).resolve() / "actor-sharing-study/D1-frozen-deployment/P5_N4_gap25"
    require(batch.parent == expected_parent, "batch outside new deployment root")
    if check_code:
        code_check(manifest["implementation_commit"])
    return manifest, read(batch / "protocol/sources.json")


def plan(batch, phase):
    data = read(Path(batch) / f"protocol/{phase}.json")
    require(data == phase_plan(phase, data["worlds"] if phase == "main" else None), "phase plan modified")
    if phase == "main":
        lock = read(Path(batch) / "protocol/world_lock.json")
        manifest, _ = load_batch(batch)
        require(lock["status"] == "LOCKED" and lock["worlds"] == data["worlds"] and
                lock["implementation_commit"] == manifest["implementation_commit"] and
                lock["manual_execution_records_reviewed"] is True, "main worlds not locked for this commit")
        require(re.fullmatch(r"[A-Za-z0-9_-]+\.json", lock["audit_file"]), "unsafe world audit reference")
        audit = read(Path(batch) / "protocol" / lock["audit_file"])
        require(audit["status"] == "CLEAR_REVIEWED" and not audit["candidate_hits"] and not audit["scan_errors"] and
                audit["candidate_worlds"] == data["worlds"] and audit["implementation_commit"] == manifest["implementation_commit"],
                "world-use audit/lock mismatch")
    return data


def preflight(batch, load_payload=False):
    manifest, sources = load_batch(batch)
    for identity in manifest["input_packs"].values():
        require(file_identity(identity["path"]) == identity, f"input evidence pack changed: {identity['path']}")
    for source in sources:
        source_check(source, payload=load_payload)
    result = {"technical_status": "PASS", "implementation_commit": manifest["implementation_commit"],
              "source_models": len(sources), "payloads_checked": load_payload,
              "protocol": PROTOCOL, "no_training": True}
    if load_payload:
        dest = Path(batch) / "protocol/preflight_payload.json"
        if dest.exists():
            require(read(dest) == result, "conflicting payload preflight")
        else:
            write(dest, result)
    return result


def _values(value):
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple)):
        return set().union(*(_values(v) for v in value)) if value else set()
    if isinstance(value, str):
        return {int(v) for v in re.findall(r"\b\d+\b", value)}
    return set()


def _worlds(data):
    result = set()
    if isinstance(data, dict):
        if "seed" in data and "scenario" in data and "episodes" in data:
            result.update(_values(data["seed"]))  # source training world, not a policy sampling seed.
        for key, value in data.items():
            if key in WORLD_FIELDS:
                result.update(_values(value))
            if isinstance(value, (dict, list)):
                result.update(_worlds(value))
    elif isinstance(data, list):
        for value in data:
            if isinstance(value, (dict, list)):
                result.update(_worlds(value))
    return result


def scan_history(roots, exclude):
    """Metadata/CSV/directory/log scan only. Unknown or unreadable inputs block lock."""
    from analysis.mappo_e3_contract import EVAL_ID_RE, LOG_WORLD_RE
    observed, records, errors = set(RESERVED), [], []
    for root in roots:
        root = Path(root).resolve()
        if not root.is_dir():
            errors.append(f"history root missing: {root}")
            continue
        for path in root.rglob("*"):
            if path.is_relative_to(exclude) or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            values = set()
            match = EVAL_ID_RE.search(path.name)
            if match:
                values.update(_values(match.group(1)))
            try:
                if path.is_file() and path.suffix == ".json":
                    values.update(_worlds(read(path)))
                elif path.is_file() and path.suffix == ".csv":
                    with path.open(encoding="utf-8-sig", newline="") as handle:
                        reader = csv.DictReader(handle)
                        fields = WORLD_FIELDS & set(reader.fieldnames or [])
                        if fields:
                            for row in reader:
                                for field in fields:
                                    values.update(_values(row[field]))
                elif path.is_file() and path.suffix in {".out", ".err", ".sbatch", ".sh"}:
                    with path.open(encoding="utf-8", errors="replace") as handle:
                        for line in handle:
                            for hit in LOG_WORLD_RE.finditer(line):
                                values.update(_values(hit.group(1)))
            except (OSError, ValueError, TypeError, csv.Error) as exc:
                errors.append(f"{path}: {exc}")
            if values:
                observed.update(values)
                records.append({"path": str(path), "worlds": sorted(values)})
    return {"scan_roots": [str(Path(p).resolve()) for p in roots], "observed_worlds": sorted(observed),
            "records": records, "scan_errors": errors}


def lock_worlds(batch, roots, candidates=None, candidate_start=1001, reviewed=False, audit_name="world_use_audit"):
    manifest, _ = load_batch(batch)
    require(reviewed, "review execution records and pass --confirm-reviewed before locking")
    require(re.fullmatch(r"[a-zA-Z0-9_-]+", audit_name), "unsafe audit name")
    require(not (Path(batch) / "protocol/main.json").exists(), "main plan already fixed")
    diagnostics = Path(manifest["diagnostics_root"]).resolve()
    roots = list(dict.fromkeys([diagnostics, REPO / "hpc", *map(Path, roots)]))
    scan = scan_history(roots, Path(batch).resolve())
    if candidates is None:
        candidates = []
        value = candidate_start
        while len(candidates) < 30:
            if value not in scan["observed_worlds"]:
                candidates.append(value)
            value += 1
    require(len(candidates) == len(set(candidates)) == 30 and min(candidates) >= 0, "exactly 30 candidate worlds")
    hits = sorted(set(candidates) & set(scan["observed_worlds"]))
    scan.update({"candidate_worlds": candidates, "candidate_hits": hits,
                 "status": "BLOCKED" if hits or scan["scan_errors"] else "CLEAR_REVIEWED",
                 "selection_rule": "explicit candidate list or first 30 unused ascending IDs; no performance inputs",
                 "implementation_commit": manifest["implementation_commit"]})
    write(Path(batch) / f"protocol/{audit_name}.json", scan)
    require(scan["status"] == "CLEAR_REVIEWED", f"world lock blocked; see {audit_name}.json")
    write(Path(batch) / "protocol/world_lock.json", {"status": "LOCKED", "worlds": candidates,
          "implementation_commit": manifest["implementation_commit"], "audit_file": f"{audit_name}.json",
          "locked_at_utc": datetime.now(timezone.utc).isoformat(), "manual_execution_records_reviewed": True})
    write(Path(batch) / "protocol/main.json", phase_plan("main", candidates))
    return {"status": "LOCKED", "worlds": candidates, "main_units": 2340}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--diagnostics-root", type=Path, required=True)
    p.add_argument("--batch-id", required=True)
    p.add_argument("--models-json", type=Path)
    p.add_argument("--evidence-pack", type=Path)
    p.add_argument("--metadata-addendum", type=Path)
    p = sub.add_parser("preflight")
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--load-payload", action="store_true")
    p = sub.add_parser("dry-run")
    p.add_argument("--phase", choices=("pilot", "main", "gap-transfer"), required=True)
    p.add_argument("--worlds", help="30 synthetic/planned world IDs for main; does not lock or evaluate")
    p = sub.add_parser("lock-worlds")
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--scan-root", type=Path, action="append", default=[])
    p.add_argument("--candidate-worlds")
    p.add_argument("--candidate-start", type=int, default=1001)
    p.add_argument("--confirm-reviewed", action="store_true")
    p.add_argument("--audit-name", default="world_use_audit")
    args = parser.parse_args(argv)
    if args.command == "init":
        result = init(args.diagnostics_root, args.batch_id, args.models_json, args.evidence_pack, args.metadata_addendum)
    elif args.command == "preflight":
        result = preflight(args.batch_root, args.load_payload)
    elif args.command == "lock-worlds":
        worlds = list(map(int, args.candidate_worlds.split(","))) if args.candidate_worlds else None
        result = lock_worlds(args.batch_root, args.scan_root, worlds, args.candidate_start, args.confirm_reviewed, args.audit_name)
    else:
        worlds = list(map(int, args.worlds.split(","))) if args.worlds else None
        data = phase_plan(args.phase, worlds)
        result = {k: v for k, v in data.items() if k != "units"}
        result.update({"first_unit": data["units"][0], "last_unit": data["units"][-1], "executes_environment": False})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
