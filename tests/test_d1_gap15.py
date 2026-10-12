"""No real model, training or PaperEnviron rollout; gap15 synthetic contracts."""
import copy
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest

from analysis import deployment_contract as dc, d1_gap15 as g, d1_gap15_history as h
from analysis import d1_gap15_jobs as jobs, evaluate_frozen_deployment as ev, summarize_frozen_deployment as su
from analysis.deployment_rng import action_seed_words, policy_seed
from aoi_v2x_reproduction.config import config_from_dict, validate_config
from tests.test_frozen_deployment import source, synthetic_arrays, META, FakeEnvironment, FakeFrozenPolicy, dump


def test_matrix_and_formal_first16_sources():
    units = g.matrix()["units"]
    assert len(units) == len({tuple(u[k] for k in ("condition", "training_seed", "world", "repeat")) for u in units}) == 144
    assert {(u["model_id"], u["training_seed"]) for u in units} == {(m, s) for m in h.MODELS for s in dc.SEEDS}
    assert sum(u["model_id"] == "algorithm2" for u in units) == 72
    assert {u["target_scenario"] for u in units} == {g.TARGET}
    assert {u["repeat"] for u in units} == {0}
    assert len(g.FIRST_IDS) == 16 and len(g.REST_IDS) == 128 and set(g.FIRST_IDS).isdisjoint(g.REST_IDS)
    assert set(g.FIRST_IDS + g.REST_IDS) == set(range(144))
    assert {(units[i]["condition"], units[i]["training_seed"], units[i]["world"]) for i in g.FIRST_IDS} == {
        (c, s, w) for c in h.CONDITIONS for s in (8, 9) for w in (213, 214)}
    for u in units:
        family = dc.MODELS[u["model_id"]]["family"]
        assert u["action_seed_words"] == action_seed_words(u["training_seed"], u["world"], 0, family)
        assert u["environment_seed"] == u["world"]
        if family == "mappo": assert u["policy_sampling_seed"] == policy_seed(u["training_seed"], u["world"], 0)
    assert [len(dc.phase_plan(p, list(range(1001, 1031)) if p == "main" else None)["units"])
            for p in ("pilot", "main", "gap-transfer")] == [60, 2340, 144]
    assert {u["target_scenario"] for u in dc.phase_plan("gap-transfer")["units"]} == {"p05_n04_g05", "p05_n04_g35"}


@pytest.mark.parametrize("model", h.MODELS)
def test_gap15_derivation_preserves_source_and_training_guard(tmp_path, model):
    from analysis.evaluate_mappo_zero_shot import build_target_config
    from analysis.evaluate_maddpg_e1 import build_frozen_maddpg_target_config
    src, cfg = source(tmp_path, model)
    if model == "algorithm2":
        raw = cfg.to_dict()
        raw.update(profile="paper_faithful", semantic_version="paper_faithful_v4", mappo_actor_sharing=None)
        raw.pop("algorithm", None)
        cfg = config_from_dict(raw)
    before = copy.deepcopy(cfg.to_dict())
    builder = build_frozen_maddpg_target_config if model == "algorithm2" else build_target_config
    target = builder(cfg, g.TARGET, "cpu")
    assert cfg.to_dict() == before and cfg.scenario.gap_m == 25 and target.scenario.gap_m == 15
    after = target.to_dict()
    assert {k: v for k, v in before.items() if k not in {"scenario", "derived", "device", "is_formal_result"}} == {
        k: v for k, v in after.items() if k not in {"scenario", "derived", "device", "is_formal_result"}}
    if model == "algorithm2":
        with pytest.raises(ValueError, match="active training requires profile"):
            validate_config(target)


def test_gap15_existing_mappo_loop_repeat0_frozen_fake_environment(tmp_path, monkeypatch):
    from analysis import evaluate_mappo_zero_shot as zs
    src, cfg = source(tmp_path)
    monkeypatch.setattr(zs.torch, "load", lambda *a, **k: {"actors": [], "episode": 500})
    monkeypatch.setattr(zs, "_validate_mappo_policy_artifact", lambda *a: (Path(src["run_dir"]), cfg, src["complete"]))
    monkeypatch.setattr(zs, "PaperEnviron", FakeEnvironment)
    monkeypatch.setattr(zs, "FrozenActorPolicy", FakeFrozenPolicy)
    results = [zs.evaluate_zero_shot(Path(src["checkpoint"]["path"]), tmp_path / f"fake_{i}", g.TARGET,
               "cpu", [213], 1, action_repeat=0, deployment=True) for i in range(2)]
    assert results[0]["exogenous_probes"] == results[1]["exogenous_probes"]
    assert results[0]["policy_episode"] == 500
    with np.load(Path(results[0]["eval_dir"]) / "metrics.npz") as a, np.load(Path(results[1]["eval_dir"]) / "metrics.npz") as b:
        assert all(np.array_equal(a[k], b[k]) for k in a.files)


def cumulative_rows(target=g.TARGET):
    rows = []
    for u in g.matrix()["units"]:
        for p in range(5):
            r = {**u, "target_scenario": target, "platoon": p}
            r.update({f: None for pair in ev.SUM_METRICS.values() for f in pair})
            # Worst AoI platoon changes with world, CAM worst is a different platoon.
            r.update(slot_count=10000, aoi_sum_ms=10000 * (90 if p == u["world"] % 5 else 10),
                     cam_episode_count=100, cam_deadline_success_count=20 if p == 4 else 90,
                     power_slot_count=10000, power_mw_sum=20000., reward_slot_count=10000, reward_combined_sum=-10000.)
            rows.append(r)
    return rows


def synthetic_seed_tables():
    return [r for target in ("p05_n04_g05", g.TARGET, dc.SCENARIO, "p05_n04_g35")
            for r in su.aggregate(cumulative_rows(target))["seed_endpoints"]]


def test_scene_first_counts_legacy_null_single_repeat_and_fig4():
    rows = cumulative_rows()
    h.coverage(rows, [g.TARGET])
    t = su.aggregate(rows)
    assert len(rows) == 720 and sum(r["slot_count"] for r in rows) == 7200000
    assert sum(r["cam_episode_count"] for r in rows) == 72000
    assert len(t["world_endpoints"]) == 144 and len(t["seed_endpoints"]) == 24
    assert all(r["scene_first_worst_aoi_ms"] == 90 and r["worst_aoi_ms_repeat_sd_mean"] is None for r in t["seed_endpoints"])
    assert t["legacy_seed_endpoints"][0]["worst_aoi_ms"] < 90
    seeds = synthetic_seed_tables()
    changed = next(r for r in seeds if r["condition"] == h.CONDITIONS[0] and r["training_seed"] == 8 and r["target_scenario"] == g.TARGET)
    changed["scene_first_worst_cam_success"] += .01
    changed["scene_first_mean_power_mw"] += 2
    f = h.fig4_tables(seeds)
    assert len(f["fig4_seed_endpoints"]) == 96 and len(f["fig4_condition_means"]) == 16
    assert len(f["gap15_paired_seed_changes"]) == 24 and len(f["gap15_paired_summary"]) == 32
    assert sum(r["data_role"] == "new" for r in f["fig4_seed_endpoints"]) == 24
    r = next(r for r in f["fig4_paired_summary"] if r["condition"] == h.CONDITIONS[0] and r["gap_m"] == 15 and r["metric"] == "worst_cam_change_pp")
    assert r["mean_change"] == pytest.approx(1 / 6) and r["unit"] == "percentage points" and r["df"] == 5
    assert r["positive"] == 1 and r["zero"] == 5
    half = su.T95 * r["sample_sd"] / math.sqrt(6)
    assert r["t95_lower"] == pytest.approx(r["mean_change"] - half)
    assert next(r for r in f["fig4_power"] if r["condition"] == h.CONDITIONS[0] and r["gap_m"] == 15 and r["training_seed"] == 8)["power_change_mw"] == 2
    with pytest.raises(ValueError): h.fig4_tables(seeds[:-1])
    with pytest.raises(ValueError): h.coverage(rows + rows[:1], [g.TARGET])


def test_batch_manifest_checks_still_require_current_commit(tmp_path, monkeypatch):
    ss = [source(tmp_path / "sources", m, s)[0] for m in dc.MODELS for s in dc.SEEDS]
    old = tmp_path / "old"
    packs = {}
    for name in ("evidence_pack", "metadata_addendum"):
        p = tmp_path / (name + ".zip")
        p.write_bytes(b"synthetic metadata identity only")
        packs[name] = dc.file_identity(p)
    diagnostics = tmp_path / "diagnostics"
    monkeypatch.setattr(dc, "load_batch", lambda *a, **k: ({"implementation_commit": h.OLD_COMMIT,
                         "diagnostics_root": str(diagnostics), "input_packs": packs}, ss))
    commits = []
    monkeypatch.setattr(dc, "code_check", lambda expected=None: commits.append(expected) or "new-commit")
    result = g.init(diagnostics, "synthetic", old, packs["evidence_pack"]["path"], packs["metadata_addendum"]["path"])
    batch = Path(result["batch_root"])
    m, sources = g.load_batch(batch)
    assert len(sources) == 18 and commits[-1] == "new-commit"
    assert m["historical_implementation_commit"] == h.OLD_COMMIT
    assert len({s["checkpoint"]["path"] for s in sources}) == 18
    assert g.plan(batch) == g.matrix()
    with pytest.raises(ValueError, match="unique"): g.init(diagnostics, "synthetic", old, packs["evidence_pack"]["path"], packs["metadata_addendum"]["path"])
    bad = g.matrix(); bad["units"][0]["world"] = 999
    dump(batch / "protocol/gap15.json", bad)
    with pytest.raises(ValueError, match="matrix modified"): g.plan(batch)
    with pytest.raises(FileNotFoundError): g.run(batch, g.REST_IDS[0])


def test_summary_full144_adapter_and_history_extension(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    batch.mkdir()
    ss = [source(tmp_path / "sources", m, s)[0] for m in h.MODELS for s in dc.SEEDS]
    m = {"implementation_commit": "new", "batch_root": str(batch), "diagnostics_root": str(tmp_path)}
    monkeypatch.setattr(g, "load_batch", lambda *a: (m, ss))
    monkeypatch.setattr(dc, "source_check", lambda *a, **k: None)
    monkeypatch.setattr(dc, "code_check", lambda *a: "new")
    rows = cumulative_rows()
    oldrows = [r for target in ("p05_n04_g05", dc.SCENARIO, "p05_n04_g35") for r in cumulative_rows(target)]
    monkeypatch.setattr(g, "historical_check", lambda *a: (oldrows, su.aggregate(oldrows)["seed_endpoints"], [{"synthetic": True}], []))
    dump(batch / "protocol/historical_inputs.json", {"files": []})
    dump(batch / "protocol/gap15.json", g.matrix())
    for unit in g.matrix()["units"]:
        dump(ev.unit_dir(batch, unit) / "DEPLOYMENT_COMPLETE.json", {})
    def accept(batch, unit, src, attempt, manifest):
        return {"attempt": attempt, "native_evaluation_dir": "synthetic", "implementation_commit": "new"}, [r for r in rows if r["unit_id"] == unit["unit_id"]], {"world": unit["world"]}
    monkeypatch.setattr(su, "accept", accept)
    result = g.summary(batch)
    assert result["technical_status"] == "PASS" and result["new_policy_world_repeats"] == 144 and result["reused_policy_world_repeats"] == 0
    assert result["historical_reused_policy_worlds"] == 432 and result["scored_agent_slots"] == 7200000
    out = batch / "analysis/gap15/a01"
    assert len(ev.csv_read(out / "fig4_seed_endpoints.csv")) == 96
    assert len(ev.csv_read(out / "historical_world_repeat_platoon.csv")) == 2160
    assert not list(out.glob("*.png"))
    with pytest.raises(FileExistsError): g.summary(batch)
    monkeypatch.setattr(g, "historical_check", lambda *a: (_ for _ in ()).throw(ValueError("history conflict")))
    with pytest.raises(ValueError, match="history conflict"): g.summary(batch, "a02")
    assert dc.read(batch / "analysis/gap15/a02/verification.json")["technical_status"] == "FAIL"


def zip_fixture(tmp_path):
    batch = tmp_path / "batch"
    out = batch / "analysis/gap15/a01"
    out.mkdir(parents=True)
    dump(batch / "protocol/manifest.json", {"implementation_commit": "new"})
    dump(batch / "protocol/sources.json", [{"model_id": m, "training_seed": s} for m in h.MODELS for s in dc.SEEDS])
    dump(batch / "protocol/gap15.json", g.matrix())
    dump(out / "verification.json", {"technical_status": "PASS", "implementation_commit": "new", "new_policy_world_repeats": 144, "reused_policy_world_repeats": 0})
    dump(out / "runtime_report.json", {"MaxRSS": None})
    for name in ("README.md", "fields_and_boundaries.md"): (out / name).write_text("test", encoding="utf-8")
    for name, count in (("world_repeat_platoon", 720), ("world_endpoints", 144), ("seed_endpoints", 24), ("fig4_seed_endpoints", 96), ("fig4_condition_means", 16), ("completion_records", 144), ("fig4_paired_summary", 128)):
        ev.csv_write(out / (name + ".csv"), [{"i": i} for i in range(count)])
    for member in g.ZIP_REQUIRED:
        path = batch / member
        if not path.exists():
            if path.suffix == ".json": dump(path, {"synthetic": True})
            else: ev.csv_write(path, [{"synthetic": True}])
    return batch, out


def test_true_zip_export_and_readback_no_raw_data(tmp_path, monkeypatch):
    batch, out = zip_fixture(tmp_path)
    (out / "not_for_export.npz").write_bytes(b"not real data")
    (out / "not_for_export.pt").write_bytes(b"not real model")
    monkeypatch.setattr(g, "load_batch", lambda *a: (dc.read(batch / "protocol/manifest.json"), dc.read(batch / "protocol/sources.json")))
    monkeypatch.setattr(dc, "source_check", lambda *a: None)
    result = g.export_zip(batch)
    assert result["zip_integrity"] == "PASS"
    with zipfile.ZipFile(result["path"]) as z:
        assert z.testzip() is None and "analysis/gap15/a01/README.md" in z.namelist()
        assert not any(n.endswith((".npz", ".pt", ".tar.zst")) for n in z.namelist())
    with pytest.raises(FileExistsError): g.export_zip(batch)
    with zipfile.ZipFile(out / "wrong.zip", "w") as z: z.writestr("README.md", "incomplete")
    with pytest.raises(ValueError, match="required files"): g.check_zip(out / "wrong.zip")


def test_array_commands_and_dry_run_never_submits(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(g, "load_batch", lambda *a: ({"implementation_commit": "new"}, []))
    monkeypatch.setattr(g, "plan", lambda *a: g.matrix())
    monkeypatch.setattr(jobs.subprocess, "check_output", lambda *a, **k: pytest.fail("dry-run submitted sbatch"))
    assert jobs.main(["--batch-root", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["submitted"] is False and len(report["commands"]) == 5
    first, rest = report["commands"][1], report["commands"][3]
    assert f"--array={jobs.array_spec(g.FIRST_IDS)}%6" in first
    assert f"--array={jobs.array_spec(g.REST_IDS)}%6" in rest
    assert "afterok:" in first and "afterok:" in rest
    assert all("--gres" not in report["commands"][i] for i in (0, 2, 4))
    assert "AOI_TASK_OFFSET" not in " ".join(report["commands"])
    for bad in ([], [0, 0], [144]):
        with pytest.raises(ValueError): jobs.array_spec(bad)
    with pytest.raises(ValueError): jobs.command(tmp_path, tmp_path, "first16", concurrency=7)


def test_readonly_check_record_no_overwrite_and_absent_quota(tmp_path, monkeypatch):
    p = tmp_path / "check.json"
    g.write_once_equal(p, {"technical_status": "PASS", "disk": 1})
    before = p.read_bytes(), p.stat().st_mtime_ns
    assert g.write_once_equal(p, {"technical_status": "PASS", "disk": 2}, volatile=("disk",))["disk"] == 1
    assert (p.read_bytes(), p.stat().st_mtime_ns) == before
    with pytest.raises(ValueError): g.write_once_equal(p, {"technical_status": "FAIL", "disk": 2}, volatile=("disk",))
    monkeypatch.setattr(g.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("quota absent")))
    assert g._command(["quota", "-s"])["returncode"] is None


def test_evidence_posix_relative_source_and_coverage_errors():
    assert h.evidence_checkpoint({"output_path": "/a/evaluations/run/eval", "source_policy": "../../../training/run/policy_final.pt"}) == "/a/training/run/policy_final.pt"
    rows = cumulative_rows()
    rows[0]["slot_count"] = 9999
    with pytest.raises(ValueError, match="denominators"): h.coverage(rows, [g.TARGET])


def test_historical_gap25_identity_relative_paths_protocol_and_blank_slots(monkeypatch):
    from analysis.e1_contract import maddpg_eval_id
    sources = [{"model_id": m, "training_seed": s, "training_commit": "original-training", "checkpoint": {"path": f"/source/{m}/{s}/policy_final.pt" if m != "algorithm2" else f"/source/{m}/{s}/checkpoints/latest.pt"}}
               for m in h.MODELS for s in dc.SEEDS]
    evaluations, agents = [], []
    mapping = {"slot_count": "aoi_slot_count", "aoi_sum_ms": "aoi_sum_ms", "aoi_gt50_count": "aoi_gt50_count",
               "aoi_at_cap_count": "aoi_at_cap_count", "cam_episode_count": "binary_cam_episode_count",
               "cam_deadline_success_count": "binary_cam_success_count", "power_mw_sum": "power_mw_sum",
               "power_slot_count": "power_slot_count", "reward_slot_count": "reward_slot_count",
               "reward_global_sum": "reward_global_sum", "reward_task1_sum": "reward_task1_sum",
               "reward_task2_sum": "reward_task2_sum", "reward_combined_sum": "reward_combined_recomputed_sum",
               "v2i_attempt_count": "V2I_attempt_count"}
    rows = cumulative_rows(dc.SCENARIO)
    for c in h.CONDITIONS:
        model, _, noise, _ = dc.CONDITIONS[c]
        for seed in dc.SEEDS:
            cp = next(s["checkpoint"]["path"] for s in sources if (s["model_id"], s["training_seed"]) == (model, seed))
            eid = f"{c}_{seed}"
            evaluation = {"eval_id": eid, "eval_dir_name": maddpg_eval_id(noise) if model == "algorithm2" else "feasibility_reward_arm_baseline",
                          "output_path": f"/source/{model}/{seed}/eval", "source_policy": cp if model == "algorithm2" else "../policy_final.pt",
                          "training_seed": str(seed), "target_scenario": "", "scenario": dc.SCENARIO,
                          "world_ids": "213,214,215,216,217,218", "stage": "E1" if model == "algorithm2" else "shared-actor-v1",
                          "algorithm": "modified_maddpg_tdec" if model == "algorithm2" else "mappo", "action_mode": "stochastic",
                          "actual_status": "complete", "frozen_check": "true", "eval_protocol": "sequential_warm",
                          "warmup_episodes": "5", "scored_episodes_recorded": "100", "slots_recorded": "",
                          "evaluator_commit": "original-not-new"}
            evaluations.append(evaluation)
            for r in rows:
                if r["condition"] == c and r["training_seed"] == seed:
                    a = {before: r.get(new) for new, before in mapping.items()}
                    a.update(eval_id=eid, world=str(r["world"]), agent=str(r["platoon"]), payload_completion_mean="")
                    agents.append(a)
    monkeypatch.setattr(su, "evidence_tables", lambda *a: (agents, [], [], evaluations))
    old, inv = h.baseline_rows("fake", sources)
    assert len(old) == 720 and len(inv) == 24
    assert {r["evaluation_implementation_commit"] for r in old} == {"original-not-new"}
    assert all(r["v2i_reset_count"] is None for r in old)
    evaluations[0]["warmup_episodes"] = "0"
    with pytest.raises(ValueError, match="protocol/frozen"): h.baseline_rows("fake", sources)


def test_first_stage_native_counts_and_probe_mismatch(tmp_path, monkeypatch):
    ss = [{"model_id": m, "training_seed": s} for m in h.MODELS for s in dc.SEEDS]
    monkeypatch.setattr(g, "load_batch", lambda *a: ({"implementation_commit": "new"}, ss))
    dump(tmp_path / "protocol/gap15.json", g.matrix())
    monkeypatch.setattr(dc, "source_check", lambda *a: None)
    monkeypatch.setattr(g, "disk_check", lambda *a, **k: {"free_bytes": 1000})
    rows = cumulative_rows()
    for i in g.FIRST_IDS: dump(ev.unit_dir(tmp_path, g.matrix()["units"][i]) / "DEPLOYMENT_COMPLETE.json", {})
    monkeypatch.setattr(ev, "accept", lambda batch, unit, *a: ({}, [r for r in rows if r["unit_id"] == unit["unit_id"]], {"world": unit["world"]}))
    result = g.verify_first(tmp_path)
    assert result["accepted_units"] == 16 and result["platoon_rows"] == 80 and result["scored_agent_slots"] == 800000
    assert g.verify_first(tmp_path) == result
    monkeypatch.setattr(ev, "accept", lambda batch, unit, *a: ({}, [r for r in rows if r["unit_id"] == unit["unit_id"]], {"different_model_probe": unit["model_id"]}))
    with pytest.raises(ValueError, match="exogenous"): g.verify_first(tmp_path)


def test_preflight_checks_all18_payloads_and_preserves_original_record(tmp_path, monkeypatch):
    ss = [source(tmp_path / "sources", m, s)[0] for m in h.MODELS for s in dc.SEEDS]
    m = {"implementation_commit": "new", "input_packs": {"synthetic": True}}
    monkeypatch.setattr(g, "load_batch", lambda *a: (m, ss))
    dump(tmp_path / "protocol/gap15.json", g.matrix())
    checked = []
    monkeypatch.setattr(dc, "source_check", lambda s, payload=False: checked.append((s["model_id"], s["training_seed"], payload)))
    monkeypatch.setattr(g, "disk_check", lambda *a, **k: {"free_bytes": 1000})
    monkeypatch.setattr(g, "historical_check", lambda *a: ([{}] * 2160, [{}] * 72, [], []))
    result = g.preflight(tmp_path)
    assert len(checked) == 18 and all(r[-1] is True for r in checked)
    assert result["payloads_checked"] and result["historical_seed_endpoints"] == 72
    assert g.preflight(tmp_path) == result
    assert all(s["config"]["scenario"]["gap_m"] == 25 for s in ss)
