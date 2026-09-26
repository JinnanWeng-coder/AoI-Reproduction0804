"""B1 unit/synthetic tests: no environment, training, or policy evaluation."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from aoi_v2x_reproduction.config import resolve_config
from analysis import mappo_b1_contract as b1
from analysis import mappo_b1_accounting as acc
from analysis import summarize_mappo_b1_budget as report
from analysis.audit_mappo_w1_service_intervals import flow_statistics


def config_for(i):
    item = b1.cell(i)
    return resolve_config(algorithm="mappo", scenario="p05_n04_g25", seed=item["training_seed"],
                          run_name=item["historical_run_name"], episodes=500, steps_per_episode=100,
                          output_root="historical/runs", device="cuda:0", checkpoint_mode="policy_only",
                          diagnostics=True, mappo_variant=item["value_configuration"],
                          mappo_actor_sharing=item["actor_structure"] == "shared",
                          mappo_entropy_coef_rb=.02, mappo_entropy_coef_mode=.02, mappo_entropy_coef_power=.002).to_dict()


@pytest.mark.parametrize("i", range(24))
def test_derive_all_configs_changes_budget_identity_only(tmp_path, i):
    raw = config_for(i)
    derived, info = b1.derive(raw, b1.cell(i), tmp_path / "runs")
    assert set(info["differences"]) == {"episodes", "run_name", "output_root"}
    assert derived.episodes == 2500 and derived.mappo_entropy_coef_rb == .02
    assert derived.checkpoint_every == raw["checkpoint_every"]
    assert derived.mappo_actor_sharing == (i // 6 in (1, 3))


@pytest.mark.parametrize("value", [None, "absent"])
def test_historical_missing_sharing_keeps_false_semantics(tmp_path, value):
    raw = config_for(0)
    if value == "absent":
        raw.pop("mappo_actor_sharing")
    else:
        raw["mappo_actor_sharing"] = value
    config, info = b1.derive(raw, b1.cell(0), tmp_path)
    assert not config.mappo_actor_sharing and info["effective_actor_sharing"] is False


def test_config_corruption_not_silently_defaulted(tmp_path):
    raw = config_for(0)
    raw.pop("mappo_entropy_coef_rb")
    with pytest.raises(ValueError):
        b1.derive(raw, b1.cell(0), tmp_path)
    raw = config_for(0)
    raw["scenario"]["number_platoons"] = 99
    with pytest.raises(ValueError, match="reconstruction"):
        b1.derive(raw, b1.cell(0), tmp_path)


def test_mapping_and_attempt_paths(tmp_path):
    assert [b1.cell(i)["training_seed"] for i in b1.PILOTS] == [8] * 4
    assert [i for i in range(48) if b1.eval_cell(i)["run_id"] in b1.PILOTS] == [0, 1, 12, 13, 24, 25, 36, 37]
    batch = b1.batch_path(tmp_path, "budget_test")
    assert b1.attempt_path(batch, "train", 0, "a01") != b1.attempt_path(batch, "train", 0, "a02")
    assert len({b1.historical_eval(tmp_path, b1.cell(i), w) for i in range(24) for w in b1.WORLDS}) == 48
    assert "shared-actor-v1" in str(b1.historical_eval(tmp_path, b1.cell(12), "development"))
    assert "tdec-ab-v1" in b1.cell(12)["historical_training_relative"]
    for bad in ("../old", "", "/absolute", "a/b"):
        with pytest.raises(ValueError):
            b1.batch_path(tmp_path, bad)
    with pytest.raises(ValueError):
        b1.attempt_path(batch, "train", 0, "../old")
    file = tmp_path / "marker.json"
    b1.write(file, {"original": True})
    with pytest.raises(FileExistsError):
        b1.write(file, {})
    assert b1.read(file) == {"original": True}


@pytest.mark.parametrize("i,steps", [(0, 25000), (6, 5000), (12, 25000), (18, 5000)])
def test_exact_optimizer_and_update_counts(i, steps):
    actors = 1 if b1.cell(i)["actor_structure"] == "shared" else 5
    records = [{"episode": u * 5, "update": u, "environment_steps": u * 500,
                "rollout_steps": 500, "ppo_epochs": 10, "actor_network_count": actors,
                "actor_optimizer_steps_this_update": 10 * actors, "actor_optimizer_step_count": u * 10 * actors}
               for u in range(1, 501)]
    result = b1.diagnostics_check(records, b1.cell(i), 2500, True)
    assert result["actor_adam_steps_total"] == steps and result["actor_adam_steps_per_actor"] == 5000
    records[-1]["actor_optimizer_step_count"] -= 1
    with pytest.raises(ValueError):
        b1.diagnostics_check(records, b1.cell(i), 2500, True)


@pytest.mark.parametrize("positions", [[0, 1, 101, 202], [], [17], [0, 900]])
def test_capped_complete_areas_and_boundaries(positions):
    n = 1000
    reset = np.zeros(n, dtype=bool)
    reset[positions] = True
    aoi = np.empty(n)
    age = 100
    for i in range(n):
        age = 1 if reset[i] else min(age + 1, 100)
        aoi[i] = age
    flow, freq, _ = flow_statistics(reset.reshape(10, 100))
    service = {"aoi_sum_ms": aoi.sum(), "aoi_slot_count": n,
               "binary_cam_success_count": 10, "binary_cam_episode_count": 10}
    result = acc.compute_flow(service, {**flow, "event_source": "reset_event"},
                              [{"length_L": k, "frequency": v} for k, v in freq.items()])
    assert sum(result["area_" + p] for p in acc.PARTS) == aoi.sum()
    assert result["area_boundary"] >= 1
    if positions == [0, 1, 101, 202]:
        assert result["area_le50"] == 1 and result["area_mid50"] == 5050 and result["area_gt100"] == 5150
    if not positions:
        assert result["boundary_slots"] == n and result["max_complete_interval"] == ""


def test_worlds_not_connected_and_shape_corruption(tmp_path):
    arrays = {name: np.ones((6, 100, 100) if name == "reward_global" else (6, 100, 100, 5), dtype=np.float32)
              for name in b1.ARRAY_FIELDS}
    arrays["reset_event"][:] = 1
    axes = {name: ["eval_seed", "scored_episode", "slot"] + ([] if name == "reward_global" else ["agent"])
            for name in b1.ARRAY_FIELDS}
    b1.write(tmp_path / "EVAL_COMPLETE.json", {"raw_metric_axes": axes})
    np.savez_compressed(tmp_path / "metrics.npz", **arrays)
    b1.evaluation_arrays(tmp_path)
    service, flow, freq = report.evaluation_tables(arrays, {"cam_bits": 32000, "global_actor_weight": 1}, b1.cell(0), "development")
    assert len(flow) == 30 and all(r["complete_interval_count"] == 9999 for r in flow)
    assert sum(r["frequency"] for r in freq) == 30 * 9999
    arrays["aoi_ms"][0, 1, 0, 0] = 2
    np.savez_compressed(tmp_path / "metrics.npz", **arrays)
    with pytest.raises(ValueError, match="reset/AoI"):
        b1.evaluation_arrays(tmp_path)
    arrays["aoi_ms"] = arrays["aoi_ms"][:, :95]
    np.savez_compressed(tmp_path / "metrics.npz", **arrays)
    with pytest.raises(ValueError, match="eval array"):
        b1.evaluation_arrays(tmp_path)


def test_world_pooled_own_extrema_and_ties():
    rows = []
    for agent in range(5):
        # Agent 0 wins world 1 only; pooled agent 1 is worse.
        sums = [1000000 if agent == 0 and w == 0 else 10000 if agent == 0 else
                200000 if agent == 1 else 10000 for w in range(6)]
        rows.append({"agent": agent, "aoi_sum_ms": sum(sums), "slot_count": 60000,
                     "binary_cam_success_count": 0 if agent == 2 else 600, "binary_cam_episode_count": 600})
    assert acc.extrema(rows, "aoi_sum_ms", "slot_count") == [1]
    assert acc.extrema(rows, "binary_cam_success_count", "binary_cam_episode_count", False) == [2]
    rows[3]["aoi_sum_ms"] = rows[1]["aoi_sum_ms"]
    assert acc.extrema(rows, "aoi_sum_ms", "slot_count") == [1, 3]


def test_budget_pairing_keeps_six_and_three_prespecified_contrasts():
    endpoints = []
    for ws in b1.WORLDS:
        for a, v in b1.CONDITIONS:
            for seed in range(8, 14):
                for budget in (50000, 250000):
                    value = (1 if a == "shared" else 10) + (2 if budget == 250000 else 0)
                    endpoints.append({"world_set": ws, "actor_structure": a, "value_configuration": v,
                                      "training_seed": seed, "budget_env_steps": budget, "agent": 0 if a == "independent" else 3,
                                      **{k: float(value) for k in report.METRICS}})
    paired, effects, levels = report.budget_effects(endpoints)
    expected = {"C-a": -11, "C-b": -9, "C-c": 2}
    for r in effects:
        assert r["n_seeds"] == 6 and r["claim_support"] == "NOT_ASSESSED"
        assert r["mean"] == expected[r["comparison"]] * (100 if r["unit"] == "percentage_points" else 1)
    assert all(r["n_seeds"] == 6 for r in levels)
    with pytest.raises((KeyError, ValueError)):
        report.budget_effects(endpoints[:-1])
    zero_crossing = acc.summary([-5, -4, -3, 3, 4, 5])
    assert zero_crossing["n_negative"] == zero_crossing["n_positive"] == 3
    assert "claim_support" not in zero_crossing


def test_refuse_repeat_before_runtime_import(tmp_path, monkeypatch):
    monkeypatch.setattr(b1, "check_batch", lambda _: {})
    b1.write(b1.record_path(tmp_path, "train", 0), {"technical_status": "PASS"})
    with pytest.raises(ValueError, match="already accepted"):
        b1.execute(tmp_path, "train", 0, "a02", "test")
    incomplete = b1.attempt_path(tmp_path, "train", 1, "a01")
    incomplete.mkdir(parents=True)
    with pytest.raises(ValueError, match="attempt exists"):
        b1.execute(tmp_path, "train", 1, "a01", "")
    with pytest.raises(ValueError, match="retry requires"):
        b1.execute(tmp_path, "train", 1, "a02", "")


def test_batch_manifest_requires_fixed_code_protocol_and_location(tmp_path, monkeypatch):
    batch = b1.batch_path(tmp_path, "budget_unit")
    audit = {"implementation_commit": "test", "baseline_commit": b1.BASELINE, "algorithm_and_evaluator_changed_paths": []}
    monkeypatch.setattr(b1, "code_audit", lambda: audit)
    manifest = {"study_root": str(tmp_path), "batch_id": "budget_unit", "code_audit": audit,
                "world_sets": b1.WORLDS, "runs": [b1.cell(i) for i in range(24)],
                "evaluations": [b1.eval_cell(i) for i in range(48)],
                "training_protocol": b1.TRAIN_PROTOCOL, "evaluation_protocol": b1.EVAL_PROTOCOL}
    b1.write(batch / "protocol/manifest.json", manifest)
    b1.write(batch / "protocol/preflight.json", {"technical_status": "PASS"})
    assert b1.check_batch(batch) == manifest
    monkeypatch.setattr(b1, "code_audit", lambda: {**audit, "implementation_commit": "other"})
    with pytest.raises(ValueError, match="implementation commit"):
        b1.check_batch(batch)


def test_cli_shell_entry_points_are_real():
    root = Path(__file__).resolve().parents[1]
    for path in ("hpc/aoi_mappo_b1_train_array.sbatch", "hpc/aoi_mappo_b1_eval_array.sbatch", "hpc/aoi_mappo_b1_analyze.sbatch"):
        text = (root / path).read_text()
        assert "b1_common.sh" in text and "BATCH_ROOT" in text
    assert "--array" not in (root / "hpc/aoi_mappo_b1_train_array.sbatch").read_text()


def test_real_2500_policy_metadata_and_training_acceptance(tmp_path, monkeypatch):
    import torch
    item = b1.cell(6)
    run = b1.attempt_path(tmp_path, "train", 6, "a01")
    run.mkdir(parents=True)
    source = tmp_path / "historical"
    raw = config_for(6)
    b1.write(source / "config.resolved.json", raw)
    sources = [{} for _ in range(24)]
    sources[6] = {"source_config": raw, "historical_run": str(source)}
    b1.write(tmp_path / "protocol/source_inventory.json", sources)
    monkeypatch.setattr(b1, "check_batch", lambda _: {"code_audit": {"implementation_commit": "new"}})
    config, _ = b1.derive(raw, item, run.parent)
    b1.write(run / "config.resolved.json", config.to_dict())
    b1.write(run / "COMPLETE.json", {"status": "complete", "algorithm": "mappo", "mappo_variant": "combined",
              "episodes": 2500, "steps_per_episode": 100, "environment_steps": 250000, "update_count": 500,
              "actor_network_count": 1, "actor_optimizer_step_count": 5000, "actor_sharing": True,
              "reproduction_git_commit": "new", "reproduction_git_dirty": False,
              "policy_final": "policy_final.pt", "config_hash": config.canonical_hash()})
    diagnostics = [{"episode": u * 5, "update": u, "environment_steps": u * 500, "rollout_steps": 500,
                    "ppo_epochs": 10, "actor_network_count": 1, "actor_optimizer_steps_this_update": 10,
                    "actor_optimizer_step_count": u * 10} for u in range(1, 501)]
    b1.write(run / "learning_diagnostics.json", diagnostics)
    payload = {"algorithm": "mappo", "artifact_type": "policy_only", "policy_schema_version": "policy_artifact_v1",
               "config": config.to_dict(), "episode": 2500, "actor_sharing": True, "actors": [{}]}
    torch.save(payload, run / "policy_final.pt")
    arrays = {name: np.zeros((2500, 100) if name == "global_step" else (2500, 100, 5), dtype=np.float32) for name in b1.TRAIN_FIELDS}
    np.savez_compressed(run / "train_metrics.npz", **arrays)
    record = b1.check_training(tmp_path, 6, "a01", True)
    assert record["policy_episode"] == 2500 and b1.check_training(tmp_path, 6, "a01") == record
    payload["episode"] = 500
    torch.save(payload, run / "policy_final.pt")
    with pytest.raises(RuntimeError, match="episode"):
        b1.policy_check(run)


def test_evaluation_provenance_binds_2500_not_old_500(tmp_path):
    import os
    run, directory = tmp_path / "run", tmp_path / "eval"
    directory.mkdir()
    item = b1.cell(0)
    b1.write(run / "COMPLETE.json", {"reproduction_git_commit": "new", "config_hash": "config"})
    done = {"status": "complete", "algorithm": "mappo", "mappo_variant": "combined", "actor_sharing": False,
            "training_seed": 8, "training_run_name": run.name, "scenario": "p05_n04_g25", "policy_name": "policy_final.pt",
            "policy_episode": 2500, "eval_seeds": b1.WORLDS["development"], "eval_episodes": 100,
            "eval_warmup_episodes": 5, "eval_protocol": "sequential_warm", "mappo_eval_mode": "stochastic",
            "intervention_arm": "baseline", "is_frozen_eval": True, "policy_parameters_unchanged": True,
            "reproduction_git_dirty": False, "policy_path_is_relative_to_eval": True, "external_action_noise_applicable": False,
            "intervention_db": 0., "aoi_cap_ms": 100., "eval_id": b1.eval_name("development"), "actor_network_count": 5,
            "source_config_hash": "config", "reproduction_git_commit": "new", "policy": os.path.relpath(run / "policy_final.pt", directory)}
    b1.write(directory / "EVAL_COMPLETE.json", done)
    b1.write(directory / "provenance.json", {**done, "training_source_commit": "new", "created_at_utc": "synthetic"})
    np.savez_compressed(directory / "metrics.npz", placeholder=np.array([1]))
    assert b1.eval_metadata(directory, run, item, "development", 2500, "new") == done
    with pytest.raises(ValueError, match="policy_episode"):
        b1.eval_metadata(directory, run, item, "development", 500, "new")


def synthetic_tables(world_set):
    services, flows, frequencies = [], [], []
    for a, v in b1.CONDITIONS:
        for seed in range(8, 14):
            for world in b1.WORLDS[world_set]:
                for agent in range(5):
                    identity = dict(actor_structure=a, value_configuration=v, training_seed=seed, world=world, agent=agent)
                    services.append({**identity, "aoi_sum_ms": 10000, "aoi_slot_count": 10000,
                        "binary_cam_success_count": 100, "binary_cam_episode_count": 100,
                        "aoi_gt50_count": 0, "aoi_at_cap_count": 0, "payload_sum": 100, "payload_episode_count": 100,
                        "power_mw_sum": 10000, "power_slot_count": 10000, "reward_combined_sum": 0, "reward_slot_count": 10000})
                    flows.append({**identity, "slot_count": 10000, "reset_count": 10000, "reset_rate": 1,
                        "complete_interval_count": 9999, "sum_L": 9999, "sum_L2": 9999, "long_gt100_count": 0,
                        "left_boundary_visible_wait_slots": 0, "right_boundary_visible_wait_slots": 0, "event_source": "reset_event"})
                    frequencies.append({**identity, "length_L": 1, "frequency": 9999})
    return services, flows, frequencies


def test_full_synthetic_two_world_set_analysis_writes_complete_package_inputs(tmp_path, monkeypatch):
    (tmp_path / "analysis").mkdir()
    sources = [b1.cell(i) for i in range(24)]
    b1.write(tmp_path / "protocol/source_inventory.json", sources)
    monkeypatch.setattr(b1, "check_batch", lambda _: {"study_root": str(tmp_path), "code_audit": {"implementation_commit": "synthetic"}})
    monkeypatch.setattr(report, "historical_inputs", lambda *_: ({ws: synthetic_tables(ws) for ws in b1.WORLDS}, []))
    monkeypatch.setattr(b1, "check_training", lambda *_: {})
    monkeypatch.setattr(b1, "check_evaluation", lambda *_: {})
    monkeypatch.setattr(b1, "evaluation_arrays", lambda _: {})
    def extract(arrays, complete, item, ws):
        return tuple([r for r in rows if r["actor_structure"] == item["actor_structure"] and
                      r["value_configuration"] == item["value_configuration"] and r["training_seed"] == item["training_seed"]]
                     for rows in synthetic_tables(ws))
    monkeypatch.setattr(report, "evaluation_tables", extract)
    monkeypatch.setattr(report, "training_curves", lambda *_: (
        [{"budget_env_steps": budget, "synthetic": i} for budget, count in ((50000, 12000), (250000, 60000)) for i in range(count)],
        [{"budget_env_steps": budget, "synthetic": i} for budget, count in ((50000, 2400), (250000, 12000)) for i in range(count)],
        [{"synthetic": 1}]))
    for i in range(24):
        b1.write(b1.record_path(tmp_path, "train", i), {"attempt": "a01"})
    for i in range(48):
        directory = tmp_path / "evaluations" / str(i)
        b1.write(directory / "EVAL_COMPLETE.json", {})
        b1.write(b1.record_path(tmp_path, "eval", i), {"attempt": "a01", "evaluation_dir": str(directory),
                  "training_run_dir": "synthetic", "implementation_commit": "synthetic"})
    result = report.analyze(tmp_path, "a01")
    assert result["technical_status"] == "PASS" and result["claim_support"] == "NOT_ASSESSED"
    out = tmp_path / "analysis/a01"
    assert len(report.csv_read(out / "development_b250000_endpoints.csv")) == 24
    assert len(report.csv_read(out / "e3_reused_b250000_world_agent.csv")) == 720
    assert len(report.csv_read(out / "e3_reused_b250000_agents.csv")) == 120
    assert all(float(r["mean"]) == 0 for r in report.csv_read(out / "effect_summary.csv"))
    assert (out / "fields_and_boundaries.md").is_file()
    with pytest.raises(FileExistsError):
        report.analyze(tmp_path, "a01")
