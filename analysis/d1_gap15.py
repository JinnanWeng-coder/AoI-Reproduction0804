"""D1 gap25->gap15: independent 144-unit frozen extension, no new training."""
from __future__ import annotations

import argparse
import csv
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from analysis import deployment_contract as dc
from analysis import evaluate_frozen_deployment as ev
from analysis import summarize_frozen_deployment as su
from analysis import d1_gap15_history as hist
from analysis.deployment_rng import action_seed_words, policy_seed

PHASE = "gap15"
TARGET = "p05_n04_g15"
FIRST_IDS = [c * 36 + i for c in range(4) for i in (0, 1, 6, 7)]
REST_IDS = [i for i in range(144) if i not in FIRST_IDS]
SELF = lambda: importlib.import_module("analysis.d1_gap15")


def matrix():
    units = []
    for condition in hist.CONDITIONS:
        model_id, execution, noise, _ = dc.CONDITIONS[condition]
        family = dc.MODELS[model_id]["family"]
        for seed in dc.SEEDS:
            for world in hist.WORLDS:
                units.append({"unit_id": len(units), "phase": PHASE, "condition": condition, "model_id": model_id,
                              "training_seed": seed, "target_scenario": TARGET, "world": world, "repeat": 0,
                              "execution": execution, "noise_std": noise, "environment_seed": world,
                              "action_seed_words": action_seed_words(seed, world, 0, family),
                              "policy_sampling_seed": policy_seed(seed, world, 0) if family == "mappo" else None})
    return {"phase": PHASE, "worlds": hist.WORLDS, "training_seeds": dc.SEEDS, "policy_world_repeat_count": 144,
            "platoon_row_count": 720, "observational_extension": True, "units": units, "first_ids": FIRST_IDS, "remaining_ids": REST_IDS}


def init(diagnostics, batch_id, old_batch, pack, addendum):
    dc.require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id), "unsafe batch ID")
    commit = dc.code_check()
    diagnostics, old_batch = Path(diagnostics).resolve(), Path(old_batch).resolve()
    old_manifest, old_sources = dc.load_batch(old_batch, check_code=False)  # Read original provenance, not current execution.
    dc.require(old_manifest["implementation_commit"] == hist.OLD_COMMIT and old_manifest["diagnostics_root"] == str(diagnostics), "wrong old D1 batch")
    sources = [s for s in old_sources if s["model_id"] in hist.MODELS]
    dc.require(len(sources) == 18 and {(s["model_id"], s["training_seed"]) for s in sources} ==
               {(m, s) for m in hist.MODELS for s in dc.SEEDS} and len({s["checkpoint"]["path"] for s in sources}) == 18,
               "exactly 18 original gap25/final500 models required")
    dc.require(all(s["environment_config"] == sources[0]["environment_config"] for s in sources), "source environments differ")
    packs = {"evidence_pack": dc.file_identity(pack), "metadata_addendum": dc.file_identity(addendum)}
    dc.require(packs == old_manifest["input_packs"], "use the original evidence/addendum identities")
    batch = diagnostics / "actor-sharing-study/D1-gap15-extension" / batch_id
    dc.require(not batch.exists() and batch != old_batch, "new unique gap15 batch required")
    batch.mkdir(parents=True, exist_ok=False)
    for name in ("protocol", "evaluations", "analysis", "slurm_logs", "tmp", "cache"):
        (batch / name).mkdir()
    dc.write(batch / "protocol/manifest.json", {"schema": "d1_gap15_v1", "implementation_commit": commit,
             "batch_root": str(batch), "diagnostics_root": str(diagnostics), "old_batch_root": str(old_batch),
             "historical_implementation_commit": hist.OLD_COMMIT, "protocol": dc.PROTOCOL,
             "models": {m: dc.MODELS[m] for m in hist.MODELS}, "conditions": list(hist.CONDITIONS),
             "target_scenario": TARGET, "source_scenario": dc.SCENARIO, "input_packs": packs,
             "gap_definition": "intra-platoon vehicle gap; bumper-to-bumper", "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "claim_support": "NOT_ASSESSED", "new_training_runs": 0})
    dc.write(batch / "protocol/sources.json", sources)
    dc.write(batch / "protocol/gap15.json", matrix())
    return {"batch_root": str(batch), "models": 18, "new_units": 144, "mappo": 72, "maddpg": 72, "first_ids": FIRST_IDS}


def load_batch(batch, check_code=True):
    batch = Path(batch).resolve()
    m, sources = dc.read(batch / "protocol/manifest.json"), dc.read(batch / "protocol/sources.json")
    dc.require(m["schema"] == "d1_gap15_v1" and m["batch_root"] == str(batch) and m["protocol"] == dc.PROTOCOL and
               m["target_scenario"] == TARGET and m["source_scenario"] == dc.SCENARIO and m["conditions"] == list(hist.CONDITIONS) and
               m["models"] == {k: dc.MODELS[k] for k in hist.MODELS} and
               batch.parent == Path(m["diagnostics_root"]).resolve() / "actor-sharing-study/D1-gap15-extension", "gap15 manifest/path contract")
    dc.require(len(sources) == 18 and {(s["model_id"], s["training_seed"]) for s in sources} ==
               {(k, s) for k in hist.MODELS for s in dc.SEEDS}, "gap15 18-model coverage")
    if check_code:
        dc.code_check(m["implementation_commit"])
    return m, sources


def plan(batch, phase=PHASE):
    dc.require(phase == PHASE, "gap15 entry cannot run old D1 stages")
    data = dc.read(Path(batch) / "protocol/gap15.json")
    dc.require(data == matrix(), "gap15 matrix modified")
    return data


def write_once_equal(path, result, volatile=()):
    """Permit a read-only recheck, not a changed protocol or replacement record."""
    if Path(path).exists():
        previous = dc.read(path)
        stable = lambda x: {k: v for k, v in x.items() if k not in volatile}
        dc.require(stable(previous) == stable(result), f"conflicting prior check: {path}")
        return previous
    dc.write(path, result)
    return result


def historical_check(manifest, sources):
    for identity in manifest["input_packs"].values():
        dc.require(dc.file_identity(identity["path"]) == identity, "original input pack changed")
    return hist.historical(manifest["old_batch_root"], manifest["input_packs"]["evidence_pack"]["path"], sources)


def disk_check(batch, manifest, first=False):
    directory = Path(batch) / "evaluations/gap15" if first else Path(manifest["old_batch_root"]) / "evaluations/gap-transfer"
    sizes = [p.stat().st_size for p in directory.rglob("metrics.npz")]
    dc.require(sizes, f"raw NPZ sizes unavailable for capacity estimate: {directory}")
    # Deliberately conservative; real raw arrays, never the compressed evidence bundle.
    estimate = max(sizes) * (128 if first else 144) * 2
    free = shutil.disk_usage(batch).free
    result = {"free_bytes": free, "reference_raw_npz_count": len(sizes), "reference_max_npz_bytes": max(sizes),
              "estimated_additional_bytes_with_2x_margin": estimate, "estimate_source": str(directory)}
    dc.require(free >= estimate, f"insufficient free disk: {result}")
    return result


def preflight(batch):
    from aoi_v2x_reproduction.config import config_from_dict
    from analysis.evaluate_mappo_zero_shot import build_target_config
    from analysis.evaluate_maddpg_e1 import build_frozen_maddpg_target_config
    m, sources = load_batch(batch)
    plan(batch)
    for s in sources:
        dc.source_check(s, payload=True)
        cfg = config_from_dict(s["config"])
        builder = build_target_config if s["family"] == "mappo" else build_frozen_maddpg_target_config
        target = builder(cfg, TARGET, "cpu")
        dc.require(target.scenario.gap_m == 15 and cfg.scenario.gap_m == 25 and target.profile == cfg.profile,
                   "gap15 derivation modified the source/profile")
    old_rows, old_seeds, inventory, files = historical_check(m, sources)
    write_once_equal(Path(batch) / "protocol/historical_inputs.json", {"files": [dc.file_identity(p) for p in files],
             "input_packs": m["input_packs"], "historical_commit": hist.OLD_COMMIT})
    result = {"technical_status": "PASS", "payloads_checked": True, "implementation_commit": m["implementation_commit"],
              "models": 18, "target": TARGET, "new_units": 144, "historical_rows": len(old_rows),
              "historical_seed_endpoints": len(old_seeds), "disk": disk_check(batch, m), "quota": "must be reviewed in runtime/preflight logs"}
    return write_once_equal(Path(batch) / "protocol/preflight_payload.json", result, volatile=("disk",))


def run(batch, unit_id, device="cuda:0", attempt="a01", reason=None, accept_existing=False, check_only=False):
    dc.require(0 <= unit_id < 144, "unit ID out of range")
    if unit_id not in FIRST_IDS:
        dc.require(dc.read(Path(batch) / "protocol/first_stage_verification.json")["technical_status"] == "PASS", "first 16 technical PASS required")
    return ev.execute(batch, PHASE, unit_id, device, attempt, reason, accept_existing, check_only, contract=SELF())


def verify_first(batch):
    m, sources = load_batch(batch)
    data, rows, probes = plan(batch), [], {}
    for i in FIRST_IDS:
        unit = data["units"][i]
        source = next(s for s in sources if (s["model_id"], s["training_seed"]) == (unit["model_id"], unit["training_seed"]))
        dc.source_check(source)
        complete = list(ev.unit_dir(batch, unit).parent.glob("a*/DEPLOYMENT_COMPLETE.json"))
        dc.require(len(complete) == 1, f"first-stage unit {i} not uniquely accepted")
        _, rs, probe = ev.accept(batch, unit, source, complete[0].parent.name, m)
        rows.extend(rs)
        if unit["world"] in probes:
            dc.require(probes[unit["world"]] == probe, "gap15 exogenous process mismatch in first stage")
        probes[unit["world"]] = probe
    dc.require(len(rows) == 80 and sum(r["slot_count"] for r in rows) == 800000 and
               sum(r["cam_episode_count"] for r in rows) == 8000, "first-stage counters")
    result = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"], "accepted_units": 16,
              "unit_ids": FIRST_IDS, "platoon_rows": 80, "scored_agent_slots": 800000, "cam_deadlines": 8000,
              "exogenous_process_exact_within_gap15": True, "disk": disk_check(batch, m, first=True)}
    return write_once_equal(Path(batch) / "protocol/first_stage_verification.json", result, volatile=("disk",))


def extend_summary(rows, tables, output, manifest, sources):
    hist.coverage(rows, [TARGET])
    dc.require(len(tables["world_endpoints"]) == 144 and len(tables["seed_endpoints"]) == 24, "gap15 endpoint counts")
    dc.require(all(r["world_count"] == 6 and r["worst_aoi_ms_repeat_sd_mean"] is None for r in tables["seed_endpoints"]),
               "single-repeat SD must remain undefined")
    identity = dc.read(Path(manifest["batch_root"]) / "protocol/historical_inputs.json")
    dc.require(all(dc.file_identity(f["path"]) == f for f in identity["files"]), "historical table changed after preflight")
    old_rows, old_seeds, inventory, _ = historical_check(manifest, sources)
    for name, values in hist.fig4_tables(tables["seed_endpoints"] + old_seeds).items():
        csv_write = ev.csv_write
        csv_write(output / f"{name}.csv", values)
    ev.csv_write(output / "historical_world_repeat_platoon.csv", old_rows)
    ev.csv_write(output / "historical_gap25_inventory.csv", inventory)
    old_tables = su.aggregate(old_rows)
    for key in ("world_platoon_repeat_pooled", "world_endpoints", "seed_endpoints"):
        ev.csv_write(output / f"historical_{key}.csv", old_tables[key])
    with (output / "fields_and_boundaries.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\n## gap15 extension / Fig.4\n\n"
            "gap_m: intra-platoon vehicle gap, bumper-to-bumper, metres (not inter-platoon).\n"
            "fig4_seed_endpoints: 96 rows, 24 new gap15 + 72 reused gaps5/25/35; same worlds213–218, repeat0.\n"
            "scene_first_*cam_success is a fraction; *_percent multiplies by100; *_change_pp is percentage points.\n"
            "fig4_paired_*: target minus same-seed gap25; six training seeds, sample SD, descriptive t95 df5.\n"
            "fig4_power: mW, never mean dBm. Historical fields absent in old exports remain undefined.\n"
            "historical_gap25_inventory and historical_world_repeat_platoon preserve original evaluation paths/commits.\n"
            "historical_* tables are read-only reused evidence, not additional new accepted units.\n")
    (output / "README.md").write_text(
        "# D1 gap25→gap15\n\n144 new frozen evaluations, 18 gap25-trained final500 models; no new training.\n"
        "Old worlds 213–218: observational transfer, not main new-world confirmation.\n"
        "Gap = intra-platoon vehicle gap (bumper-to-bumper, metres).\n"
        "Fig.4: 96 seed endpoints / 16 means. Old gaps5/25/35 are read-only reused, not new rollouts.\n"
        "Scene-first: sum/count within world/platoon, own worst AoI/CAM, then worlds, then six training seeds.\n"
        "CAM fractions, percent and percentage-point changes are explicitly separate. Power uses per-slot mW.\n"
        "Descriptive paired t95 df5, n=6 training seeds. Single-repeat SD undefined, never zero-filled.\n"
        "Historical gap25 references use the same old worlds, never D1 main. Provenance retains original commits.\n"
        "No NPZ/checkpoint in this ZIP. Full arrays stay on HPC. Technical PASS does not assert a favorable result.\n", encoding="utf-8")
    return {"world_endpoints": 144, "seed_endpoints": 24, "conditions": 4, "source_models": 18,
            "fig4_seed_endpoints": 96, "fig4_condition_means": 16, "historical_reused_policy_worlds": 432,
            "historical_reused_platoon_rows": 2160, "gap_definition": "intra-platoon vehicle gap; bumper-to-bumper"}


def summary(batch, attempt="a01"):
    return su.summarize(batch, PHASE, attempt, make_plots=False, contract=SELF(), extension=extend_summary)


def _command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True)
    except FileNotFoundError as exc:
        return {"command": args, "returncode": None, "stdout": None, "stderr": str(exc), "available": False}
    return {"command": args, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def runtime_report(batch, output):
    m, _ = load_batch(batch)
    jobs = [dc.read(p) for p in sorted((Path(batch) / "protocol").glob("submission_*.json"))]
    ids = [str(r["job_id"]) for r in jobs]
    dc.require(ids, "no recorded job IDs for runtime accounting")
    sacct = _command(["sacct", "-P", "-j", ",".join(ids), "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,ElapsedRaw,MaxRSS,AllocCPUS,ReqMem"])
    dc.require(sacct["returncode"] == 0, "sacct failed; preserve analysis and collect runtime before export")
    accounting = list(csv.DictReader(io.StringIO(sacct["stdout"]), delimiter="|"))
    accounting = [{k: (None if v == "" else v) for k, v in r.items() if k} for r in accounting]
    ev.csv_write(output / "sacct.csv", accounting)
    size = sum(p.stat().st_size for p in (Path(batch) / "evaluations").rglob("*") if p.is_file())
    failures = [r for r in accounting if any(s in str(r.get("State")) for s in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"))]
    report = {"implementation_commit": m["implementation_commit"], "batch_root": str(Path(batch).resolve()), "worlds": hist.WORLDS,
              "source_scenario": dc.SCENARIO, "target_scenario": TARGET, "source_models": 18, "jobs": jobs,
              "sacct": accounting, "failures": failures, "raw_evaluation_bytes": size, "MaxRSS_missing_values_are_null": True,
              "snapshot_utc": datetime.now(timezone.utc).isoformat(), "packaging_job_id": os.environ.get("SLURM_JOB_ID"),
              "packaging_job_state_at_snapshot": "RUNNING; Grok must check final COMPLETED/0:0 afterwards",
              "disk": _command(["df", "-PB1", str(batch)]), "inodes": _command(["df", "-Pi", str(batch)]),
              "quota": _command(["quota", "-s"]), "quota_review": "see preflight/runtime logs; unavailable is not zero quota"}
    dc.write(output / "runtime_report.json", report)
    (output / "runtime_report.md").write_text(
        f"# Runtime\n\nCommit: {m['implementation_commit']}\n\nBatch: {batch}\n\n"
        f"Raw evaluations: {size} bytes. Jobs: {','.join(ids)}.\n\n"
        "See JSON/sacct CSV for exact commands/resources/states. Empty MaxRSS is null, not estimated.\n"
        "This snapshot precedes packaging-job exit; Grok must verify its final COMPLETED/0:0 state.\n", encoding="utf-8")


ZIP_REQUIRED = {"analysis/gap15/a01/README.md", "protocol/manifest.json", "protocol/sources.json", "protocol/gap15.json",
                "protocol/preflight_payload.json", "protocol/first_stage_verification.json", "protocol/historical_inputs.json",
                "analysis/gap15/a01/verification.json", "analysis/gap15/a01/completion_records.csv",
                "analysis/gap15/a01/world_repeat_platoon.csv", "analysis/gap15/a01/fig4_seed_endpoints.csv",
                "analysis/gap15/a01/fig4_condition_means.csv", "analysis/gap15/a01/fig4_paired_summary.csv",
                "analysis/gap15/a01/fig4_paired_seed_changes.csv", "analysis/gap15/a01/fig4_power.csv",
                "analysis/gap15/a01/gap15_paired_seed_changes.csv", "analysis/gap15/a01/gap15_paired_summary.csv",
                "analysis/gap15/a01/historical_gap25_inventory.csv", "analysis/gap15/a01/historical_world_repeat_platoon.csv",
                "analysis/gap15/a01/world_endpoints.csv", "analysis/gap15/a01/seed_endpoints.csv",
                "analysis/gap15/a01/condition_summary.csv", "analysis/gap15/a01/legacy_seed_endpoints.csv",
                "analysis/gap15/a01/runtime_report.json", "analysis/gap15/a01/sacct.csv",
                "analysis/gap15/a01/fields_and_boundaries.md"}


def check_zip(path, attempt="a01"):
    required = {n.replace("/a01/", f"/{attempt}/") for n in ZIP_REQUIRED}
    with zipfile.ZipFile(path) as z:
        dc.require(z.testzip() is None and required <= set(z.namelist()), "ZIP integrity/required files")
        dc.require(not any(Path(n).suffix in {".npz", ".pt", ".zst"} or ".." in Path(n).parts or
                           n.startswith(("/", "\\")) or "\\" in n for n in z.namelist()), "raw data/unsafe ZIP member")
        v = json.loads(z.read(f"analysis/gap15/{attempt}/verification.json"))
        m = json.loads(z.read("protocol/manifest.json"))
        dc.require(v["technical_status"] == "PASS" and v["implementation_commit"] == m["implementation_commit"], "ZIP verification/commit")
        dc.require(v["new_policy_world_repeats"] == 144 and v["reused_policy_world_repeats"] == 0 and
                   len(json.loads(z.read("protocol/sources.json"))) == 18, "ZIP new/source counts")
        for name, count in (("world_repeat_platoon", 720), ("world_endpoints", 144), ("seed_endpoints", 24),
                            ("fig4_seed_endpoints", 96), ("fig4_condition_means", 16), ("completion_records", 144)):
            rs = list(csv.DictReader(io.StringIO(z.read(f"analysis/gap15/{attempt}/{name}.csv").decode("utf-8-sig"))))
            dc.require(len(rs) == count, f"ZIP table count: {name}")
        # Physically extract a small CSV/JSON sample, then parse it again.
        with tempfile.TemporaryDirectory(prefix="zip_readback_", dir=Path(path).parent) as temporary:
            for name in ("protocol/manifest.json", f"analysis/gap15/{attempt}/fig4_seed_endpoints.csv"):
                z.extract(name, temporary)
            dc.require(dc.read(Path(temporary) / "protocol/manifest.json") == m, "extracted JSON differs")
            dc.require(len(ev.csv_read(Path(temporary) / f"analysis/gap15/{attempt}/fig4_seed_endpoints.csv")) == 96,
                       "extracted CSV count")
    return {"zip_integrity": "PASS", "required_csv_json_readback": "PASS", "path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size}


def export_zip(batch, attempt="a01"):
    m, sources = load_batch(batch)
    dc.require(re.fullmatch(r"a[0-9]{2,4}", attempt), "unsafe attempt")
    out = Path(batch) / "analysis/gap15" / attempt
    v = dc.read(out / "verification.json")
    dc.require(v["technical_status"] == "PASS" and v["implementation_commit"] == m["implementation_commit"] and
               v["new_policy_world_repeats"] == 144 and v["reused_policy_world_repeats"] == 0, "no accepted gap15 summary")
    for s in sources:
        dc.source_check(s)
    dc.require((out / "runtime_report.json").is_file(), "runtime report required before packaging")
    dest = out / "core_evidence.zip"
    with zipfile.ZipFile(dest, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root in (Path(batch) / "protocol", out):
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix in {".json", ".csv", ".md"}:
                    dc.require(not p.is_symlink(), "do not export symlink targets")
                    z.write(p, p.relative_to(batch).as_posix())
        for p in sorted((Path(batch) / "evaluations/gap15").rglob("*.json")):
            if p.name in {"STARTED.json", "DEPLOYMENT_COMPLETE.json", "EVAL_COMPLETE.json", "provenance.json"}:
                dc.require(not p.is_symlink(), "do not export symlink targets")
                z.write(p, p.relative_to(batch).as_posix())
        for p in sorted((Path(batch) / "slurm_logs").glob("*")):
            if p.is_file() and p.suffix in {".out", ".err"}:
                with p.open("rb") as h:
                    h.seek(max(0, p.stat().st_size - 32768))
                    z.writestr("log_excerpts/" + p.name, h.read())
    return check_zip(dest, attempt)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("init", "dry-run", "preflight", "run", "first-check", "summary", "runtime", "finalize", "export", "check-zip"))
    p.add_argument("--batch-root", type=Path)
    p.add_argument("--diagnostics-root", type=Path)
    p.add_argument("--batch-id")
    p.add_argument("--old-batch", type=Path)
    p.add_argument("--evidence-pack", type=Path)
    p.add_argument("--metadata-addendum", type=Path)
    p.add_argument("--unit-id", type=int)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--attempt", default="a01")
    p.add_argument("--reason")
    p.add_argument("--accept-existing", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--zip", type=Path)
    a = p.parse_args(argv)
    if a.command == "init":
        dc.require(all((a.diagnostics_root, a.batch_id, a.old_batch, a.evidence_pack, a.metadata_addendum)), "init paths required")
        result = init(a.diagnostics_root, a.batch_id, a.old_batch, a.evidence_pack, a.metadata_addendum)
    elif a.command == "dry-run":
        result = {k: v for k, v in matrix().items() if k != "units"}
        result.update({"models": 18, "mappo_units": 72, "maddpg_units": 72, "executes_environment": False})
    elif a.command == "check-zip":
        dc.require(a.zip, "--zip required")
        result = check_zip(a.zip, a.attempt)
    else:
        dc.require(a.batch_root, "--batch-root required")
        if a.command == "preflight": result = preflight(a.batch_root)
        elif a.command == "run":
            dc.require(a.unit_id is not None, "--unit-id required")
            result = run(a.batch_root, a.unit_id, a.device, a.attempt, a.reason, a.accept_existing, a.check_only)
        elif a.command == "first-check": result = verify_first(a.batch_root)
        elif a.command == "runtime":
            runtime_report(a.batch_root, a.batch_root / "analysis/gap15" / a.attempt)
            result = {"runtime_report": "written"}
        elif a.command in {"summary", "finalize"}:
            result = summary(a.batch_root, a.attempt)
            if a.command == "finalize":
                runtime_report(a.batch_root, a.batch_root / "analysis/gap15" / a.attempt)
                result.update(export_zip(a.batch_root, a.attempt))
        else: result = export_zip(a.batch_root, a.attempt)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
