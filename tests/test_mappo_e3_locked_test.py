"""E3 synthetic contracts; never load a real policy or run an environment."""

import json
from collections import Counter

import numpy as np
import pytest

from analysis import mappo_e3_contract as e3
from analysis import summarize_mappo_e3_locked_test as summary
from analysis.audit_mappo_w1_service_intervals import _ccdf, flow_statistics


def test_24_cell_mapping_sources_and_world_lock():
    p = e3.protocol()
    assert p["candidate_worlds"] == [303, 304, 305, 306, 307, 308]
    assert set(p["candidate_worlds"]).isdisjoint({301, 302})
    assert len(p["policies"]) == 24
    assert [e3.cell(i)["training_seed"] for i in (0, 6, 12, 18)] == [8] * 4
    assert e3.cell(0)["training_root"].startswith("existing-evidence/tdec-ab-v1")
    assert e3.cell(6)["training_root"].startswith("E5-sharing-critic")
    assert e3.cell(12)["training_root"].startswith("existing-evidence/tdec-ab-v1")
    assert e3.cell(18)["training_root"].startswith("existing-evidence/shared-actor-v1")
    assert "s303-304-305-306-307-308_ep100" in e3.eval_id()


def test_protocol_mismatch_and_lock_refusal(tmp_path, monkeypatch):
    source = e3.protocol()
    changed = json.loads(json.dumps(source))
    changed["candidate_worlds"] = [303, 304, 305, 306, 307, 307]
    file = tmp_path / "protocol.json"
    file.write_text(json.dumps(changed), encoding="utf-8")
    monkeypatch.setattr(e3, "PROTOCOL_FILE", file)
    with pytest.raises(AssertionError):
        e3.protocol()
    monkeypatch.undo()
    monkeypatch.setattr(e3, "current_commit", lambda: "a" * 40)
    root = tmp_path / "study"
    out = e3.result_root(root)
    out.mkdir(parents=True)
    e3._write(out / "world_use_audit.json", {
        "status": "CANDIDATE_CLEAR_REQUIRES_MANUAL_REVIEW", "candidate_worlds": source["candidate_worlds"],
        "candidate_hits": [], "scan_errors": [], "commit": "a" * 40,
        "scan_roots": [str(root.resolve()), str(root.resolve().parent / "MAPPO_results")]})
    e3._write(out / "source_inventory.json", {
        "policy_count": 24, "all_payloads_checked": True, "sources": [
            {key: item[key] for key in ("cell_id", "actor_structure", "value_configuration",
                                        "training_seed", "training_run_name")}
            for item in source["policies"]],
        "commit": "a" * 40})
    with pytest.raises(ValueError, match="manual"):
        e3.create_lock(root, False)
    lock = e3.create_lock(root, True)
    assert lock["status"] == "LOCKED" and e3.validate_lock(root)["protocol"] == source
    with pytest.raises(ValueError, match="existing unreviewed"):
        e3.create_lock(root, True)


def test_history_audit_distinguishes_world_fields_from_plain_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(e3, "current_commit", lambda: "b" * 40)
    study = tmp_path / "project" / "AoI-Reproduction-diagnostics" / "actor-sharing-study"
    study.mkdir(parents=True)
    diagnostic_mappo = study.parent / "MAPPO_results"
    diagnostic_mappo.mkdir()
    (study / "plan.json").write_text(json.dumps({"note": "303 was only proposed"}), encoding="utf-8")
    (study / "reserved.json").write_text(json.dumps({"selection_validation_seeds": [301, 302]}), encoding="utf-8")
    (study / "actual.json").write_text(json.dumps({"eval_seeds": [213, 303]}), encoding="utf-8")
    directory = study / "old-run" / "evaluations" / "run" / "eval_validation_s304-305_ep100"
    directory.mkdir(parents=True)
    logs = study / "old-run" / "slurm_logs"
    logs.mkdir(parents=True)
    (logs / "job.out").write_text("eval_world=306\n", encoding="utf-8")
    (diagnostic_mappo / "eval.csv").write_text("eval_seed\n307\n", encoding="utf-8")
    report = e3.history_audit(study)
    assert report["status"] == "BLOCKED"
    assert {world for hit in report["candidate_hits"] for world in hit["candidate_worlds"]} == {303, 304, 305, 306, 307}
    assert str(diagnostic_mappo) in report["scan_roots"]


def test_supersede_preserves_old_blocked_audit_before_new_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(e3, "current_commit", lambda: "b" * 40)
    study = tmp_path / "project" / "AoI-Reproduction-diagnostics" / "actor-sharing-study"
    study.mkdir(parents=True)
    (study.parent / "MAPPO_results").mkdir()
    output = e3.result_root(study)
    output.mkdir(parents=True)
    old = {"status": "BLOCKED", "candidate_worlds": [301, 302, 303, 304, 305, 306],
           "commit": "a" * 40, "candidate_hits": [{"candidate_worlds": [301, 302]}]}
    e3._write(output / "world_use_audit.json", old)
    with pytest.raises(ValueError, match="explicit supersede"):
        e3.history_audit(study)
    new = e3.history_audit(study, supersede_blocked_audit=True)
    assert new["status"] == "CANDIDATE_CLEAR_REQUIRES_MANUAL_REVIEW"
    assert new["candidate_worlds"] == [303, 304, 305, 306, 307, 308]
    assert e3._json(output / "world_use_audit_previous_aaaaaaa.json") == old
    assert len(new["scan_roots"]) == 2


def _c1_rows_for_one_seed():
    rows = []
    for world in e3.protocol()["candidate_worlds"]:
        for agent in range(5):
            aoi = (100 if world == 303 else 0) if agent == 0 else 20 if agent == 1 else 5
            cam = (0 if world == 303 else 100) if agent == 0 else 75 if agent == 1 else 100
            rows.append({"world": world, "agent": agent, "aoi_sum_ms": aoi * 10000,
                         "aoi_slot_count": 10000, "aoi_gt50_count": 0, "aoi_at_cap_count": 0,
                         "binary_cam_success_count": cam, "binary_cam_episode_count": 100,
                         "payload_sum": 100.0, "payload_episode_count": 100,
                         "reward_combined_sum": 0.0, "reward_slot_count": 10000,
                         "power_mw_sum": 10000.0, "power_slot_count": 10000})
    return rows


def test_worst_agent_aggregates_worlds_first_and_cam_uses_episodes():
    result = summary.c1_seed_metrics(_c1_rows_for_one_seed(), "independent", "combined", 8)
    assert result["worst_agent_aoi_ms"] == pytest.approx(20)
    assert result["worst_agent_binary_cam"] == pytest.approx(0.75)
    assert result["binary_cam_episode_count"] == 3000
    assert result["aoi_slot_count"] == 300000


def test_world_agent_power_conversion_and_endpoint_cam():
    shape = (6, 100, 100, 5)
    arrays = {name: np.zeros(shape, dtype=np.float32) for name in
              ("aoi_ms", "success", "remaining_demand", "executed_power_dbm", "reward_task1", "reward_task2")}
    arrays["reward_global"] = np.zeros((6, 100, 100), dtype=np.float32)
    arrays["executed_power_dbm"][:, :, 50:, :] = 10.0
    arrays["success"][:, :, -1, :] = 1.0
    rows = summary.world_agent_rows(arrays, {"cam_bits": 32000, "global_actor_weight": 1}, e3.cell(0))
    assert len(rows) == 30
    assert rows[0]["binary_cam_episode_count"] == 100
    assert rows[0]["binary_cam_success_count"] == 100
    assert rows[0]["power_mw_sum"] / rows[0]["power_slot_count"] == pytest.approx(5.5)


def test_seed_effect_signs_and_descriptive_interval():
    rows = []
    for structure, variant in summary.CONDITIONS:
        for seed in summary.SEEDS:
            rows.append({"actor_structure": structure, "value_configuration": variant,
                         "training_seed": seed, "metric": float(seed) + (1 if structure == "shared" else 0)})
    conditions, effects, effects_summary = summary._summaries(rows, ("metric",))
    assert len(conditions) == 4 and len(effects) == 12
    assert effects_summary[0]["delta_metric_positive_count"] == 6
    assert effects_summary[0]["delta_metric_negative_count"] == 0
    assert effects_summary[0]["delta_metric_ci95_low"] == pytest.approx(1)


def test_interval_weights_boundaries_ccdf_and_json_indices():
    key = ("independent", "combined", 8)
    reset_a = np.zeros((2, 5), dtype=bool)
    reset_a.reshape(-1)[[0, 1, 5]] = True  # complete L=1,4, including cross-episode
    reset_b = np.zeros((2, 5), dtype=bool)
    reset_b.reshape(-1)[[0, 9]] = True
    rows, freqs = [], {}
    for agent, reset in enumerate((reset_a, reset_b)):
        row, freq, positions = flow_statistics(reset)
        rows.append({"world": 303, "agent": agent, **row})
        freqs[key + (303, agent)] = freq
        json.dumps({"index": positions, "example": [int(x) for x in np.diff(positions)]})
    empty, _, _ = flow_statistics(np.zeros((2, 5), dtype=bool))
    once = np.zeros((2, 5), dtype=bool); once[0, 3] = True
    single, _, _ = flow_statistics(once)
    assert empty["no_reset_window_slots"] == 10 and empty["complete_interval_mean_L"] is None
    assert single["complete_interval_mean_L"] is None and single["left_boundary_visible_wait_slots"] == 3
    ccdf = _ccdf(rows, freqs, key, [0, 1, 4, 9])
    assert ccdf[1]["event_ccdf_P_L_gt_threshold"] == pytest.approx(2 / 3)
    assert ccdf[1]["flow_equal_ccdf_P_L_gt_threshold"] == pytest.approx(0.75)
    assert freqs[key + (303, 0)] == Counter({1: 1, 4: 1})


def test_empty_interval_frequency_is_valid_header_only(tmp_path):
    path = tmp_path / "frequency.csv"
    summary._csv(path, [], ["actor_structure", "length_L", "frequency"])
    assert path.read_text(encoding="utf-8").strip() == "actor_structure,length_L,frequency"


def test_completed_cell_marker_revalidates_for_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(e3, "current_commit", lambda: "c" * 40)
    study = tmp_path / "study"
    root = e3.result_root(study)
    root.mkdir(parents=True)
    locked = {"status": "LOCKED", "locked_at_utc": "2026-09-17T00:00:00+00:00",
              "implementation_commit": "c" * 40, "protocol": e3.protocol()}
    e3._write(root / "lock.json", locked)
    item = e3.cell(0)
    directory = e3.eval_dir(study, item)
    directory.mkdir(parents=True)
    source = {"policy": str(study / "source" / "policy_final.pt"), "training_commit": "training-commit"}
    monkeypatch.setattr(e3, "_assert_source", lambda *_args, **_kwargs: source)
    # A relative policy reference is resolved from the evaluation directory.
    relpolicy = __import__("os").path.relpath(source["policy"], directory)
    complete = {"status": "complete", "algorithm": "mappo", "mappo_variant": "combined",
                "actor_sharing": False, "actor_network_count": 5, "training_seed": 8,
                "training_run_name": item["training_run_name"], "scenario": "p05_n04_g25",
                "policy_name": "policy_final.pt", "policy_episode": 500,
                "eval_seeds": e3.protocol()["candidate_worlds"], "eval_episodes": 100,
                "eval_warmup_episodes": 5, "eval_protocol": "sequential_warm",
                "mappo_eval_mode": "stochastic", "intervention_arm": "baseline", "is_frozen_eval": True,
                "policy_parameters_unchanged": True, "reproduction_git_commit": "c" * 40,
                "reproduction_git_dirty": False, "eval_id": e3.eval_id(),
                "policy_path_is_relative_to_eval": True, "external_action_noise_applicable": False,
                "intervention_db": 0.0, "aoi_cap_ms": 100.0, "policy": relpolicy,
                "raw_metric_axes": {}}
    arrays = {}
    for name in e3.ARRAY_FIELDS:
        shape = (6, 100, 100) if name == "reward_global" else e3.AGENT_SHAPE
        arrays[name] = np.zeros(shape, dtype=np.float32)
        complete["raw_metric_axes"][name] = ["eval_seed", "scored_episode", "slot"] + ([] if name == "reward_global" else ["agent"])
    arrays["reset_event"] = np.zeros(e3.AGENT_SHAPE, dtype=bool)
    arrays["aoi_ms"][:] = 2.0
    e3._write(directory / "EVAL_COMPLETE.json", complete)
    e3._write(directory / "provenance.json", {**{key: complete[key] for key in
        ("algorithm", "mappo_variant", "actor_sharing", "training_seed", "eval_seeds",
         "eval_episodes", "eval_warmup_episodes", "mappo_eval_mode", "intervention_arm", "policy")},
        "training_source_commit": "training-commit", "reproduction_git_commit": "c" * 40,
        "reproduction_git_dirty": False, "created_at_utc": "2026-09-17T00:01:00+00:00"})
    np.savez_compressed(directory / "metrics.npz", **arrays)
    marker = e3.validate_cell(study, 0, write_marker=True)
    assert marker["training_commit"] == "training-commit"
    assert marker["evaluation_commit"] == "c" * 40
    assert e3.validate_cell(study, 0) == marker
    with pytest.raises(FileExistsError):
        e3.validate_cell(study, 0, write_marker=True)
    (directory / "E3_COMPLETE.json").unlink()
    recovered = e3.validate_cell(study, 0, write_marker=True, source_array_job_id="307326")
    assert recovered["slurm_array_job_id"] == "307326" and recovered["slurm_array_task_id"] == "0"
    # The evaluator puts this field in provenance, not EVAL_COMPLETE.
    provenance_path = directory / "provenance.json"
    provenance = e3._json(provenance_path)
    provenance["training_source_commit"] = "wrong-training"
    e3._write(provenance_path, provenance)
    with pytest.raises(ValueError, match="training commit"):
        e3.validate_cell(study, 0)
    provenance["training_source_commit"] = "training-commit"
    e3._write(provenance_path, provenance)
    complete["training_source_commit"] = "wrong-training"
    e3._write(directory / "EVAL_COMPLETE.json", complete)
    with pytest.raises(ValueError, match="evaluation training commit"):
        e3.validate_cell(study, 0)
    complete.pop("training_source_commit")
    # A later technical-only checkout may validate a newly evaluated cell while
    # retaining the original protocol lock and the earlier pilot's eval commit.
    provenance["training_source_commit"] = "training-commit"
    provenance["reproduction_git_commit"] = "d" * 40
    e3._write(provenance_path, provenance)
    complete["reproduction_git_commit"] = "d" * 40
    e3._write(directory / "EVAL_COMPLETE.json", complete)
    monkeypatch.setattr(e3, "LOCKED_E3_COMMIT", "c" * 40)
    monkeypatch.setattr(e3, "current_commit", lambda: "d" * 40)
    monkeypatch.setattr(e3, "_repair_diff_paths", lambda *_args: sorted(e3.REPAIR_ALLOWED_PATHS))
    e3._write(root / "lock_technical_repair.json", {
        "status": "TECHNICAL_REPAIR", "original_lock_commit": "c" * 40,
        "repair_commit": "d" * 40, "original_locked_at_utc": locked["locked_at_utc"],
        "changed_paths": sorted(e3.REPAIR_ALLOWED_PATHS)})
    (directory / "E3_COMPLETE.json").unlink()
    repaired = e3.validate_cell(study, 0, write_marker=True)
    assert repaired["implementation_commit"] == "c" * 40
    assert repaired["evaluation_commit"] == "d" * 40
    assert repaired["validation_commit"] == "d" * 40


def test_technical_repair_preserves_original_lock_and_accepts_both_eval_commits(tmp_path, monkeypatch):
    old, new = e3.LOCKED_E3_COMMIT, "d" * 40
    monkeypatch.setattr(e3, "current_commit", lambda: new)
    monkeypatch.setattr(e3, "_repair_diff_paths", lambda *_args: sorted(e3.REPAIR_ALLOWED_PATHS))
    study = tmp_path / "study"
    output = e3.result_root(study)
    output.mkdir(parents=True)
    locked = {"status": "LOCKED", "locked_at_utc": "2026-09-17T07:36:06+00:00",
              "implementation_commit": old, "protocol": e3.protocol()}
    e3._write(output / "lock.json", locked)
    with pytest.raises(FileNotFoundError):
        e3.validate_lock(study)
    e3._write(output / "world_use_audit.json", {"commit": old})
    e3._write(output / "source_inventory.json", {"commit": old})
    original_check_output = e3.subprocess.check_output
    monkeypatch.setattr(e3.subprocess, "check_output", lambda args, **kwargs:
                        "" if args[:2] == ["git", "status"] else original_check_output(args, **kwargs))
    amendment = e3.amend_lock_for_technical_repair(study, True)
    assert amendment["original_lock_commit"] == old and amendment["repair_commit"] == new
    assert e3._json(output / "lock.json") == locked
    assert e3.validate_lock(study) == locked
    with pytest.raises(FileExistsError):
        e3.amend_lock_for_technical_repair(study, True)


def test_full_synthetic_analysis_retains_all_no_reset_flows(tmp_path, monkeypatch):
    study = tmp_path / "study"
    locked = {"status": "LOCKED", "locked_at_utc": "2026-09-17T00:00:00+00:00",
              "implementation_commit": "d" * 40, "protocol": e3.protocol()}
    monkeypatch.setattr(summary, "validate_lock", lambda _root: locked)
    monkeypatch.setattr(summary, "current_commit", lambda: "d" * 40)
    monkeypatch.setattr(summary, "validate_cell", lambda _root, i: {
        "training_commit": f"training-{i}", "implementation_commit": "d" * 40,
        "evaluation_commit": "d" * 40,
        "worlds": e3.protocol()["candidate_worlds"]})
    arrays = {}
    for name in e3.ARRAY_FIELDS:
        shape = (6, 100, 100) if name == "reward_global" else e3.AGENT_SHAPE
        arrays[name] = np.zeros(shape, dtype=bool if name == "reset_event" else np.float32)
    arrays["aoi_ms"][:] = 2.0
    arrays["success"][:, :, -1, :] = 1.0
    for i in range(24):
        directory = e3.eval_dir(study, e3.cell(i))
        directory.mkdir(parents=True)
        e3._write(directory / "EVAL_COMPLETE.json", {"cam_bits": 32000, "global_actor_weight": 1.0})
        np.savez_compressed(directory / "metrics.npz", **arrays)
    result = summary.analyze(study)
    assert result["status"] == "PASS"
    assert result["world_agent_flows"] == 720 and result["eligible_interval_flows"] == 0
    out = e3.result_root(study) / "analysis"
    assert (out / "c2_interval_frequency.csv").read_text(encoding="utf-8").count("\n") == 1
    assert json.loads((out / "verification.json").read_text(encoding="utf-8"))["status"] == "PASS"
