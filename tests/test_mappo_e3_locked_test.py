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
    assert p["candidate_worlds"] == [301, 302, 303, 304, 305, 306]
    assert len(p["policies"]) == 24
    assert [e3.cell(i)["training_seed"] for i in (0, 6, 12, 18)] == [8] * 4
    assert e3.cell(0)["training_root"].startswith("existing-evidence/tdec-ab-v1")
    assert e3.cell(6)["training_root"].startswith("E5-sharing-critic")
    assert e3.cell(12)["training_root"].startswith("existing-evidence/tdec-ab-v1")
    assert e3.cell(18)["training_root"].startswith("existing-evidence/shared-actor-v1")
    assert "s301-302-303-304-305-306_ep100" in e3.eval_id()


def test_protocol_mismatch_and_lock_refusal(tmp_path, monkeypatch):
    source = e3.protocol()
    changed = json.loads(json.dumps(source))
    changed["candidate_worlds"] = [301, 302, 303, 304, 305, 305]
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
        "candidate_hits": [], "scan_errors": [], "commit": "a" * 40})
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
    (study / "plan.json").write_text(json.dumps({"note": "301 was only proposed"}), encoding="utf-8")
    (study / "actual.json").write_text(json.dumps({"eval_seeds": [213, 301]}), encoding="utf-8")
    directory = study / "old-run" / "evaluations" / "run" / "eval_validation_s302-303_ep100"
    directory.mkdir(parents=True)
    logs = study / "old-run" / "slurm_logs"
    logs.mkdir(parents=True)
    (logs / "job.out").write_text("eval_world=304\n", encoding="utf-8")
    report = e3.history_audit(study)
    assert report["status"] == "BLOCKED"
    assert {world for hit in report["candidate_hits"] for world in hit["candidate_worlds"]} == {301, 302, 303, 304}


def _c1_rows_for_one_seed():
    rows = []
    for world in e3.protocol()["candidate_worlds"]:
        for agent in range(5):
            aoi = (100 if world == 301 else 0) if agent == 0 else 20 if agent == 1 else 5
            cam = (0 if world == 301 else 100) if agent == 0 else 75 if agent == 1 else 100
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
        rows.append({"world": 301, "agent": agent, **row})
        freqs[key + (301, agent)] = freq
        json.dumps({"index": positions, "example": [int(x) for x in np.diff(positions)]})
    empty, _, _ = flow_statistics(np.zeros((2, 5), dtype=bool))
    once = np.zeros((2, 5), dtype=bool); once[0, 3] = True
    single, _, _ = flow_statistics(once)
    assert empty["no_reset_window_slots"] == 10 and empty["complete_interval_mean_L"] is None
    assert single["complete_interval_mean_L"] is None and single["left_boundary_visible_wait_slots"] == 3
    ccdf = _ccdf(rows, freqs, key, [0, 1, 4, 9])
    assert ccdf[1]["event_ccdf_P_L_gt_threshold"] == pytest.approx(2 / 3)
    assert ccdf[1]["flow_equal_ccdf_P_L_gt_threshold"] == pytest.approx(0.75)
    assert freqs[key + (301, 0)] == Counter({1: 1, 4: 1})


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
                "training_source_commit": "training-commit", "raw_metric_axes": {}}
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
    assert e3.validate_cell(study, 0) == marker
    with pytest.raises(FileExistsError):
        e3.validate_cell(study, 0, write_marker=True)


def test_full_synthetic_analysis_retains_all_no_reset_flows(tmp_path, monkeypatch):
    study = tmp_path / "study"
    locked = {"status": "LOCKED", "locked_at_utc": "2026-09-17T00:00:00+00:00",
              "implementation_commit": "d" * 40, "protocol": e3.protocol()}
    monkeypatch.setattr(summary, "validate_lock", lambda _root: locked)
    monkeypatch.setattr(summary, "validate_cell", lambda _root, i: {
        "training_commit": f"training-{i}", "implementation_commit": "d" * 40,
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
