"""Thin D1 adapter: execute one manifest unit with existing frozen evaluators."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np

from analysis import deployment_contract as dc

SUM_METRICS = {
    "aoi_ms": ("aoi_sum_ms", "slot_count"),
    "cam_success": ("cam_deadline_success_count", "cam_episode_count"),
    "power_mw": ("power_mw_sum", "power_slot_count"),
    "reward_global": ("reward_global_sum", "reward_slot_count"),
    "reward_task1": ("reward_task1_sum", "reward_slot_count"),
    "reward_task2": ("reward_task2_sum", "reward_slot_count"),
    "reward_combined": ("reward_combined_sum", "reward_slot_count"),
    "reward_remaining_demand_term": ("reward_remaining_demand_term_sum", "reward_slot_count"),
    "reward_aoi_term": ("reward_aoi_term_sum", "reward_slot_count"),
    "reward_v2i_revenue_term": ("reward_v2i_revenue_term_sum", "reward_slot_count"),
    "reward_power_penalty": ("reward_power_penalty_sum", "reward_slot_count"),
    "aoi_gt50": ("aoi_gt50_count", "slot_count"),
    "aoi_at_cap": ("aoi_at_cap_count", "slot_count"),
    "payload_completion": ("payload_completion_sum", "cam_episode_count"),
    "v2i_attempt_rate": ("v2i_attempt_count", "v2i_slot_count"),
    "v2i_reset_rate": ("v2i_reset_count", "v2i_slot_count"),
    "v2i_success_given_attempt": ("v2i_reset_count", "v2i_attempt_count"),
}
REQUIRED = ("aoi_ms", "success", "remaining_demand", "power_dbm", "reward_global", "reward_task1", "reward_task2")
OPTIONAL = ("mode", "reset_event", "v2i_attempt", "v2i_rate", "reward_combined",
            "reward_remaining_demand_term", "reward_aoi_term", "reward_v2i_revenue_term", "reward_power_penalty")


def csv_write(path, rows):
    columns = sorted({k for row in rows for k in row})
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        dc.require(set(REQUIRED) <= set(archive.files), f"missing array fields: {set(REQUIRED) - set(archive.files)}")
        return {k: archive[k] for k in (*REQUIRED, *OPTIONAL) if k in archive.files}


def reduce_arrays(arrays, metadata, identity, expected=(1, 100, 100, 5)):
    """All sums float64; CAM deadline is final-slot success, not per-slot success."""
    shape = np.asarray(arrays["aoi_ms"]).shape
    dc.require(shape == expected, f"AoI shape {shape}, expected {expected}")
    for key in REQUIRED + tuple(k for k in OPTIONAL if k in arrays):
        value = np.asarray(arrays[key])
        dc.require(value.shape == (shape[:-1] if key == "reward_global" else shape) and
                   np.all(np.isfinite(value)), f"invalid shape/non-finite {key}")
    aoi = np.asarray(arrays["aoi_ms"], dtype=np.float64)
    dc.require(np.all((aoi >= 1) & (aoi <= 100) & (aoi == np.rint(aoi))), "invalid post-step capped AoI")
    success = np.asarray(arrays["success"], dtype=np.float64)
    dc.require(np.isin(success, [0, 1]).all(), "CAM success must be binary")
    dc.require(np.all(np.diff(success, axis=2) >= 0), "CAM completion must persist inside an episode")
    reset = np.isclose(aoi, 1.0, rtol=0, atol=1e-6)
    if "reset_event" in arrays:
        dc.require(np.isin(arrays["reset_event"], [0, 1]).all() and
                   np.array_equal(arrays["reset_event"].astype(bool), reset), "reset_event/AoI mismatch")
    flat = aoi.reshape(shape[0], -1, shape[-1])
    expected_next = np.where(reset.reshape(flat.shape)[:, 1:], 1, np.minimum(100, flat[:, :-1] + 1))
    dc.require(np.array_equal(flat[:, 1:], expected_next), "AoI continuity broken across scored episodes")
    mode = arrays.get("mode")
    if mode is not None:
        dc.require(np.isin(mode, [0, 1]).all(), "invalid mode encoding")
        dc.require(not np.any(reset & (mode != 0)), "V2I reset in V2V mode")
    trusted = metadata.get("event_source") == "post_step_aoi_ms_isclose_1_verified_env_step"
    attempts = (mode == 0) if mode is not None and trusted else None
    if "v2i_attempt" in arrays:
        dc.require(np.isin(arrays["v2i_attempt"], [0, 1]).all(), "non-binary attempt event")
        if mode is not None:
            dc.require(np.array_equal(arrays["v2i_attempt"].astype(bool), mode == 0), "attempt/mode mismatch")
    if trusted and "v2i_rate" in arrays and metadata.get("v2i_min") is not None:
        # Both legacy writers store rates as float32; the exact reset predicate
        # is evaluated in env.step before conversion. Do not guess at threshold ties.
        dc.require(np.all(np.asarray(arrays["v2i_rate"])[~reset] <= float(metadata["v2i_min"]) + 1e-3),
                   "rate/reset inconsistency")
    power_mw = np.power(10.0, np.asarray(arrays["power_dbm"], dtype=np.float64) / 10.0)
    t1, t2 = (np.asarray(arrays[k], dtype=np.float64) for k in ("reward_task1", "reward_task2"))
    rg = np.broadcast_to(np.asarray(arrays["reward_global"], dtype=np.float64)[..., None], shape)
    weight, cam_bits = float(metadata["global_actor_weight"]), float(metadata["cam_bits"])
    combined = t1 + t2 + weight * rg
    values = {"aoi_ms": aoi, "cam_success": success[:, :, -1, :], "power_mw": power_mw,
              "reward_global": rg, "reward_task1": t1, "reward_task2": t2, "reward_combined": combined,
              "aoi_gt50": aoi > 50, "aoi_at_cap": aoi == 100,
              "payload_completion": np.clip(1 - np.asarray(arrays["remaining_demand"], dtype=np.float64)[:, :, -1, :] / cam_bits, 0, 1)}
    if trusted:
        from aoi_v2x_reproduction.envs.platoon import power_penalty
        penalty = np.array([power_penalty(v) for v in np.asarray(arrays["power_dbm"]).ravel()]).reshape(shape)
        terms = {"reward_remaining_demand_term": -4.95 * np.asarray(arrays["remaining_demand"], dtype=np.float64) / cam_bits,
                 "reward_aoi_term": -aoi / 20, "reward_v2i_revenue_term": .05 * reset,
                 "reward_power_penalty": penalty}
        if mode is not None:
            dc.require(np.allclose(t1, terms["reward_remaining_demand_term"] - penalty * (mode == 1), rtol=1e-5, atol=2e-6), "task1 ledger")
            dc.require(np.allclose(t2, terms["reward_aoi_term"] + terms["reward_v2i_revenue_term"] - penalty * (mode == 0), rtol=1e-5, atol=2e-6), "task2 ledger")
        values.update(terms)
    if attempts is not None:
        values["v2i_attempt_rate"], values["v2i_reset_rate"] = attempts, reset
    for key in ("reward_combined", "reward_remaining_demand_term", "reward_aoi_term", "reward_v2i_revenue_term", "reward_power_penalty"):
        if key in arrays and key in values:
            dc.require(np.allclose(arrays[key], values[key], rtol=1e-5, atol=2e-6), f"stored/reconstructed {key}")
    rows = []
    for world_index in range(shape[0]):
        for platoon in range(shape[-1]):
            row = {**identity, "world": identity.get("world", metadata["eval_seeds"][world_index]),
                   "platoon": platoon, "event_source": metadata.get("event_source") if trusted else None,
                   "slot_count": shape[1] * shape[2], "cam_episode_count": shape[1],
                   "power_slot_count": shape[1] * shape[2], "reward_slot_count": shape[1] * shape[2],
                   "v2i_slot_count": shape[1] * shape[2] if attempts is not None else None,
                   "aoi_sum_squares_ms2": float(np.square(aoi[world_index, ..., platoon]).sum(dtype=np.float64))}
            for metric, (numerator, _denominator) in SUM_METRICS.items():
                if numerator in row:
                    continue
                row[numerator] = (float(values[metric][world_index, ..., platoon].sum(dtype=np.float64)) if metric in values else None)
                if numerator.endswith("_count") and row[numerator] is not None:
                    row[numerator] = int(row[numerator])
            # A reset may be known even when attempt semantics are unavailable.
            if trusted:
                row["v2i_reset_count"] = int(reset[world_index, ..., platoon].sum())
            rows.append(row)
    return rows


def source_identity(source):
    return {k: source[k] for k in ("model_id", "training_seed", "checkpoint", "source_config_hash", "training_commit")}


def native_check(metadata, unit, source, implementation_commit):
    from aoi_v2x_reproduction.config import config_from_dict
    cfg = config_from_dict(source["config"])
    common = {"status": "complete", "training_seed": unit["training_seed"], "eval_seeds": [unit["world"]],
              "eval_protocol": "sequential_warm", "eval_warmup_episodes": 5, "eval_episodes": 100,
              "target_scenario": unit["target_scenario"], "source_config_hash": source["source_config_hash"],
              "training_source_commit": source["training_commit"], "action_repeat": unit["repeat"],
              "environment_seeds": [unit["world"]], "action_seed_words": [unit["action_seed_words"]],
              "reproduction_git_commit": implementation_commit, "reproduction_git_dirty": False,
              "source_scenario": dc.SCENARIO, "cam_bits": float(cfg.cam_bits),
              "global_actor_weight": float(cfg.global_actor_weight),
              "world_role": "locked_new_worlds" if unit["phase"] == "main" else "reused_development_worlds"}
    for key, expected in common.items():
        dc.require(metadata.get(key) == expected, f"native evaluation {key}: {metadata.get(key)!r} != {expected!r}")
    if source["family"] == "mappo":
        dc.require(metadata.get("policy_episode") == 500 and metadata.get("mappo_eval_mode") == unit["execution"] and
                   metadata.get("policy_parameters_unchanged") is True and
                   metadata.get("policy_sampling_seeds") == [unit["policy_sampling_seed"]] and
                   metadata.get("actor_sharing") == (source["structure"] == "shared") and
                   metadata.get("mappo_variant") == source["variant"] and
                   metadata.get("actor_network_count") == (1 if source["structure"] == "shared" else 5), "frozen MAPPO execution contract")
    else:
        dc.require(metadata.get("checkpoint_episode") == 500 and metadata.get("noise_std") == unit["noise_std"] and
                   metadata.get("parameter_snapshot_unchanged") is True and metadata.get("optimizer_steps") == 0 and
                   metadata.get("algorithm") == "modified_maddpg_tdec" and metadata.get("normalization_updated") is False,
                   "frozen MADDPG execution contract")
    dc.require(metadata.get("raw_metric_axes", {}).get("aoi_ms") == ["eval_seed", "scored_episode", "slot", "agent"], "native array axis semantics")
    dc.require(len(metadata.get("exogenous_probes", [])) == 1 and
               metadata["exogenous_probes"][0]["world"] == unit["world"], "missing environment RNG/state probe")


def unit_dir(batch, unit, attempt="a01"):
    dc.require(re.fullmatch(r"a[0-9]{2,4}", attempt), "unsafe attempt")
    return Path(batch) / "evaluations" / unit["phase"] / f"unit_{unit['unit_id']:04d}" / attempt


def accept(batch, unit, source, attempt, manifest, create=False):
    dest = unit_dir(batch, unit, attempt)
    markers = list(dest.rglob("EVAL_COMPLETE.json"))
    dc.require(len(markers) == 1, f"expected one native completion in {dest}, found {len(markers)}")
    metadata = dc.read(markers[0])
    native_check(metadata, unit, source, manifest["implementation_commit"])
    provenance = dc.read(markers[0].with_name("provenance.json"))
    dc.require(provenance.get("reproduction_git_commit") == manifest["implementation_commit"], "native provenance commit")
    model_path = (markers[0].parent / metadata["policy"]).resolve() if source["family"] == "mappo" else Path(metadata["checkpoint"]).resolve()
    dc.require(str(model_path) == source["checkpoint"]["path"], "native source model path mismatch")
    metrics = markers[0].with_name("metrics.npz")
    identity = {**unit, "family": source["family"], "actor_structure": source["structure"],
                "value_configuration": source["variant"], "source_checkpoint": str(model_path),
                "source_config_hash": source["source_config_hash"], "training_commit": source["training_commit"],
                "implementation_commit": manifest["implementation_commit"], "data_role": "new",
                "training_environment_steps": 50000, "policy_episode": 500}
    identity.pop("action_seed_words")
    rows = reduce_arrays(load_arrays(metrics), metadata, identity)
    started_path = dest / "STARTED.json"
    started = dc.read(started_path) if started_path.exists() else {}
    if started:
        dc.require(started["unit"] == unit and started["source_identity"] == source_identity(source) and
                   started["implementation_commit"] == manifest["implementation_commit"], "started record identity changed")
    record = {"technical_status": "PASS", "unit": unit, "source_identity": source_identity(source),
              "implementation_commit": manifest["implementation_commit"], "attempt": attempt,
              "native_evaluation_dir": str(markers[0].parent), "metrics": dc.file_identity(metrics),
              "row_count": len(rows), "scored_agent_slots": 50000, "cam_deadlines": 500,
              "slurm_job_id": started.get("slurm_job_id"), "slurm_array_job_id": started.get("slurm_array_job_id"),
              "slurm_array_task_id": started.get("slurm_array_task_id")}
    marker_path = dest / "DEPLOYMENT_COMPLETE.json"
    if create:
        dc.require(not marker_path.exists(), "deployment completion already exists")
        csv_path = dest / "world_repeat_platoon.csv"
        if csv_path.exists():
            _check_csv(csv_path, rows)
        else:
            csv_write(csv_path, rows)
        dc.write(marker_path, record)
    else:
        dc.require(dc.read(marker_path) == record, "completion identity changed")
        _check_csv(dest / "world_repeat_platoon.csv", rows)
    return record, rows, metadata["exogenous_probes"][0]


def _check_csv(path, rows):
    old = csv_read(path)
    dc.require(len(old) == len(rows), "cumulative CSV row count")
    for left, right in zip(old, rows):
        dc.require(set(left) == set(right), "cumulative CSV schema")
        for key, value in right.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                dc.require(float(left[key]) == value, f"cumulative CSV mismatch: {key}")
            else:
                dc.require(left[key] == ("" if value is None else str(value)), f"cumulative CSV identity: {key}")


REUSABLE_PILOT_COMMIT = "3258d35b66dbc00f4a987a76159e8a7167cb6413"


def _reuse_inputs(batch, previous_batch):
    """Narrow repair bridge: old MAPPO execution code must be unchanged."""
    manifest, sources = dc.load_batch(batch)
    old_manifest, old_sources = dc.load_batch(previous_batch, check_code=False)
    dc.require(Path(batch).resolve() != Path(previous_batch).resolve(), "reuse requires a new batch")
    dc.require(old_manifest["implementation_commit"] == REUSABLE_PILOT_COMMIT,
               "reuse is limited to the original 3258d35 pilot")
    dc.require(not dc.git("diff", "--name-only", REUSABLE_PILOT_COMMIT, "HEAD", "--",
                         "aoi_v2x_reproduction", "analysis/evaluate_mappo_zero_shot.py",
                         "analysis/evaluate_mappo_feasibility_reward.py",
                         "analysis/evaluate_mappo_power_intervention.py", "analysis/deployment_rng.py"),
               "MAPPO execution/environment/RNG code changed; cannot reuse this pilot")
    by_key = lambda entries: {(s["model_id"], s["training_seed"]): s for s in entries}
    dc.require(by_key(sources) == by_key(old_sources), "pilot reuse source identity/config mismatch")
    dc.require(manifest["diagnostics_root"] == old_manifest["diagnostics_root"] and
               manifest["input_packs"] == old_manifest["input_packs"], "pilot reuse input mapping changed")
    dc.require(dc.plan(previous_batch, "pilot") == dc.phase_plan("pilot"), "original pilot plan changed")
    return manifest, sources, old_manifest


def register_reused_pilot(batch, previous_batch):
    manifest, sources, old_manifest = _reuse_inputs(batch, previous_batch)
    dc.require(not any((Path(batch) / "evaluations/pilot").glob("unit_*")), "register reuse before any new pilot unit")
    records = {}
    for unit in dc.phase_plan("pilot")["units"]:
        parent = unit_dir(previous_batch, unit).parent
        completed = list(parent.glob("a*/DEPLOYMENT_COMPLETE.json"))
        source = next(s for s in sources if (s["model_id"], s["training_seed"]) == (unit["model_id"], unit["training_seed"]))
        if source["family"] != "mappo":
            dc.require(not completed and not list(parent.rglob("EVAL_COMPLETE.json")),
                       "old MADDPG completion exists; this repair expects 16 pre-rollout failures")
            continue
        dc.require(len(completed) == 1, f"reused MAPPO unit {unit['unit_id']} requires one accepted attempt")
        dc.source_check(source)
        record, _, _ = accept(previous_batch, unit, source, completed[0].parent.name, old_manifest)
        records[str(unit["unit_id"])] = record
    dc.require(len(records) == 44, "exactly 44 original MAPPO pilot units required")
    dc.write(Path(batch) / "protocol/reused_pilot.json", {
        "previous_batch_root": str(Path(previous_batch).resolve()),
        "previous_implementation_commit": old_manifest["implementation_commit"],
        "implementation_commit": manifest["implementation_commit"], "records": records,
        "reused_policy_world_repeats": 44, "required_new_policy_world_repeats": 16,
        "raw_data_copied": False, "historical_provenance_modified": False})


def reused_pilot_unit(batch, unit, source, manifest):
    reference = Path(batch) / "protocol/reused_pilot.json"
    if unit["phase"] != "pilot" or not reference.exists():
        return None
    refs = dc.read(reference)
    expected_ids = {str(u["unit_id"]) for u in dc.phase_plan("pilot")["units"] if dc.MODELS[u["model_id"]]["family"] == "mappo"}
    dc.require(refs["implementation_commit"] == manifest["implementation_commit"] and
               refs["previous_implementation_commit"] == REUSABLE_PILOT_COMMIT and
               set(refs["records"]) == expected_ids, "invalid reused pilot mapping")
    if str(unit["unit_id"]) not in refs["records"]:
        return None
    previous_batch = Path(refs["previous_batch_root"])
    _, _, old_manifest = _reuse_inputs(batch, previous_batch)
    dc.require(not list(unit_dir(batch, unit).parent.glob("a*/DEPLOYMENT_COMPLETE.json")) and
               not list(unit_dir(batch, unit).parent.rglob("EVAL_COMPLETE.json")),
               "duplicate newly evaluated and reused MAPPO unit")
    saved = refs["records"][str(unit["unit_id"])]
    record, rows, probe = accept(previous_batch, unit, source, saved["attempt"], old_manifest)
    dc.require(record == saved, "original pilot completion changed after reuse registration")
    return ({**record, "reused_from_batch": str(previous_batch)},
            [{**r, "data_role": "reused"} for r in rows], probe)


def execute(batch, phase, unit_id, device="cpu", attempt="a01", reason=None, accept_existing=False, check_only=False, *, contract=dc):
    manifest, sources = contract.load_batch(batch)
    preflight = dc.read(Path(batch) / "protocol/preflight_payload.json")
    dc.require(preflight.get("technical_status") == "PASS" and preflight.get("payloads_checked") is True and
               preflight["implementation_commit"] == manifest["implementation_commit"], "payload preflight required")
    units = contract.plan(batch, phase)["units"]
    dc.require(0 <= unit_id < len(units), "unit ID out of range")
    unit = units[unit_id]
    source = next(s for s in sources if (s["model_id"], s["training_seed"]) == (unit["model_id"], unit["training_seed"]))
    dc.source_check(source)
    reused = reused_pilot_unit(batch, unit, source, manifest)
    if reused is not None:
        return {**reused[0], "skipped_complete": True}
    dest = unit_dir(batch, unit, attempt)
    # One accepted source/key across attempts; never count or run it twice.
    accepted = list(dest.parent.glob("a*/DEPLOYMENT_COMPLETE.json")) if dest.parent.exists() else []
    dc.require(len(accepted) <= 1, "duplicate completed attempts for one unit")
    if accepted:
        record, _, _ = accept(batch, unit, source, accepted[0].parent.name, manifest)
        return {**record, "skipped_complete": True}
    if check_only or accept_existing:
        return accept(batch, unit, source, attempt, manifest, create=accept_existing)[0]
    dc.require(not dest.exists(), f"incomplete attempt preserved: {dest}; accept existing completion or use new attempt with reason")
    dc.require(attempt == "a01" or bool(reason), "retry requires --reason")
    if dest.parent.exists():
        dc.require(not list(dest.parent.rglob("EVAL_COMPLETE.json")), "native completion exists; --accept-existing or report validation failure, no re-evaluation")
    dest.mkdir(parents=True, exist_ok=False)
    dc.write(dest / "STARTED.json", {"unit": unit, "source_identity": source_identity(source),
             "implementation_commit": manifest["implementation_commit"], "attempt": attempt, "reason": reason,
             "device_requested": device, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
             "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")})
    if source["family"] == "mappo":
        from analysis.evaluate_mappo_zero_shot import evaluate_zero_shot
        evaluate_zero_shot(Path(source["checkpoint"]["path"]), dest, unit["target_scenario"], device,
                           [unit["world"]], 100, action_repeat=unit["repeat"],
                           deterministic=unit["execution"] == "deterministic", deployment=True,
                           world_role="locked_new_worlds" if phase == "main" else "reused_development_worlds")
    else:
        from analysis.evaluate_maddpg_e1 import evaluate_maddpg_e1
        run = Path(source["run_dir"])
        override = {"run_dir": run, "resolved_run_dir": run.resolve(), "checkpoint": Path(source["checkpoint"]["path"]),
                    "canonical_run_name": run.name, "historical_run_name": run.name, "algorithm_label": "algorithm2",
                    "is_symlink": run.is_symlink(), "complete": source["complete"], "manifest": {"deployment_source_inventory": True}}
        evaluate_maddpg_e1("modified_maddpg_tdec", unit["noise_std"], unit["training_seed"],
                          Path(manifest["diagnostics_root"]), dest, device, [unit["world"]], 100, 5, 500,
                          action_repeat=unit["repeat"], target_scenario=unit["target_scenario"],
                          source_override=override, deployment=True,
                          world_role="locked_new_worlds" if phase == "main" else "reused_development_worlds")
    dc.source_check(source)
    record, _, _ = accept(batch, unit, source, attempt, manifest, create=True)
    dc.code_check(manifest["implementation_commit"])
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--phase", choices=("pilot", "main", "gap-transfer"), required=True)
    parser.add_argument("--unit-id", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attempt", default="a01")
    parser.add_argument("--reason")
    parser.add_argument("--accept-existing", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    result = execute(args.batch_root, args.phase, args.unit_id, args.device, args.attempt, args.reason, args.accept_existing, args.check_only)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
