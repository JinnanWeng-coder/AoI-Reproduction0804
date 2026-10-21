"""Random frozen sampler/lifecycle/aggregation/recovery, synthetic matrix and short real smoke only."""
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
import zipfile

import numpy as np
import pytest

from analysis import evaluate_random_frozen as ev
from analysis.deployment_rng import random_action_seed_words, exogenous_probe
from analysis.frozen_rollout import sequential_warm, arrays_from_records
from aoi_v2x_reproduction.config import config_from_dict
from aoi_v2x_reproduction.envs.platoon import PaperEnviron, power_penalty


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "code_check", lambda expected=None: "random-test")
    m = ev.init(tmp_path, "random-test", "test")
    batch = Path(m["batch_root"])
    ev.preflight(batch)
    return batch


def config():
    return ev.resolved_config(ev.random.source_bundle()[0])


def test_preview_fixed60_keys_no_training_and_config_alignment():
    p = ev.preview()
    assert p["unit_count"] == 60 and p["training_models"] == p["training_runs"] == 0
    assert p["new_unseen_confirmation"] is False and p["action_stream_ids"] == [12, 13]
    matrix = p["units"]
    assert len({tuple(u["unit_key"]) for u in matrix}) == 60
    assert {(u["action_stream_id"], u["world"]) for u in matrix} == {(s, w) for s in [12, 13] for w in range(1001, 1031)}
    assert all(u["training_seed"] is None and u["training_model_count"] == 0 and "repeat" not in u for u in matrix)
    cfg = config()
    assert {k: cfg.to_dict()[k] for k in ev.dc.ENV_FIELDS} == p["environment_config"]
    assert cfg.global_actor_weight == 1 and cfg.device == "cpu"
    assert ev.units()[0]["action_seed_words"] == [12, 1001, 0x52414E44, 1]
    with pytest.raises(ValueError): random_action_seed_words(12, -1)


def test_exact_sampler_definition_dbm_uniform_reproducible_private_rng():
    cfg = config()
    env = PaperEnviron(cfg)
    env.reset_world(910001)
    physical = copy.deepcopy(env.rng.bit_generator.state)
    global_state = np.random.get_state()
    rng = np.random.default_rng(np.random.SeedSequence(random_action_seed_words(12, 1001)))
    draws = np.concatenate([ev.random.draw_actions(cfg, rng)[1] for _ in range(10000)])
    assert env.rng.bit_generator.state == physical
    np.testing.assert_equal(global_state, np.random.get_state())
    assert np.all(np.abs(np.bincount(draws[:, 0].astype(int)) / len(draws) - 1/3) < .01)
    assert np.all(np.abs(np.bincount(draws[:, 1].astype(int)) / len(draws) - .5) < .01)
    assert 1 <= draws[:, 2].min() < 1.01 and 29.99 < draws[:, 2].max() <= 30
    assert draws[:, 2].mean() == pytest.approx(15.5, abs=.15)
    assert draws[:, 2].var() == pytest.approx(29**2 / 12, abs=1)
    theoretical_mw = 10 * (1000 - 10**.1) / (29 * math.log(10))
    assert theoretical_mw == pytest.approx(149.56818536)
    assert np.power(10., draws[:, 2] / 10).mean() == pytest.approx(theoretical_mw, abs=3)
    a, b, different = [np.random.default_rng(np.random.SeedSequence(random_action_seed_words(s, 1001))) for s in (12, 12, 13)]
    for _ in range(10): np.testing.assert_array_equal(ev.random.draw_actions(cfg, a)[0], ev.random.draw_actions(cfg, b)[0])
    assert not np.array_equal(ev.random.draw_actions(cfg, a)[0], ev.random.draw_actions(cfg, different)[0])
    for _ in range(10):
        actions, decoded = ev.random.draw_actions(cfg, a)
        np.testing.assert_allclose(env.decode_actions(actions), decoded, rtol=0, atol=1e-12)


class TrackingEnv:
    """Same call shape, not physical research trajectories."""
    def __init__(self): self.resets = []; self.starts = []; self.count = 0
    def reset_world(self, world): self.resets.append(world); self.count = 0
    def start_episode(self, e): self.starts.append(e); self.e = e; return np.full((5, 1), e)
    def step(self, actions):
        self.count += 1
        info = {"aoi_ms": np.full(5, self.e), "success": np.zeros(5), "remaining_demand": np.full(5, 32000),
                "power_dbm": np.full(5, 10), "mode": np.zeros(5), "rb": np.zeros(5), "v2i_rate": np.zeros(5)}
        return np.full((5, 1), self.e), self.e, np.full(5, self.e), np.zeros(5), False, info


def test_shared_rollout_excludes_warmup_preserves_cold_and_episode_indices():
    env = TrackingEnv()
    draws = []
    def factory(w):
        rng = np.random.default_rng(w)
        def act(obs):
            value = rng.random((5, 3)); draws.append((w, env.e)); return value
        return act
    records, probes = sequential_warm(env, [910001, 910002], factory, steps_per_episode=3,
        scored_episodes=2, warmup_episodes=5, deployment=True, probe=lambda e: {"slot_count": e.count})
    a = arrays_from_records(records, True)
    assert env.resets == [910001, 910002] and env.starts == list(range(7)) * 2
    assert len(draws) == 42 and a["aoi_ms"].shape == (2, 2, 3, 5)
    assert np.array_equal(a["reward_global"], np.array([[5]*3, [6]*3])[None].repeat(2, axis=0))
    assert all(p["initial"]["slot_count"] == 0 and p["final"]["slot_count"] == 21 for p in probes)


def test_real_smoke_no_networks_and_same_world_physics(batch, monkeypatch):
    import aoi_v2x_reproduction.runtime.runner as runner
    monkeypatch.setattr(runner, "_make_system", lambda *a: pytest.fail("must not build training/policy system"))
    result = ev.smoke(batch)
    assert result["technical_status"] == "PASS" and result["physical_probes_match"]
    assert ev.smoke(batch) == result and not list((batch / "records").glob("unit_*.json"))
    assert all(r["scored_agent_slots"] == 1000 and r["cam_deadlines"] == 10 and
               r["sampled_environment_steps_with_warmup"] == 700 for r in result["checks"])
    cfg = config()
    a, am = ev.sample_rollout(cfg, 12, 910001, 2, 5)
    b, bm = ev.sample_rollout(cfg, 12, 910001, 2, 5)
    for key in a: np.testing.assert_array_equal(a[key], b[key])
    assert am == bm


def fake_arrays():
    shape = (1, 100, 100, 5)
    aoi = np.full(shape, 100, np.float32)
    power = np.full(shape, 10, np.float32)
    return {"aoi_ms": aoi, "success": np.zeros(shape, np.float32), "remaining_demand": np.full(shape, 32000, np.float32),
            "power_dbm": power, "mode": np.zeros(shape, np.int64), "rb": np.zeros(shape, np.int64),
            "reset_event": np.zeros(shape, bool), "v2i_attempt": np.ones(shape, bool), "v2i_rate": np.zeros(shape, np.float32),
            "reward_global": np.full(shape[:-1], .5, np.float32), "reward_task1": np.full(shape, -4.95, np.float32),
            "reward_task2": (-aoi / 20 - power_penalty(10)).astype(np.float32)}


def write_native(batch, unit, m, attempt="a01"):
    dest = batch / f"units/unit_{unit['unit_id']:03d}/{attempt}"
    dest.mkdir(parents=True)
    ev.dc.write(dest / "STARTED.json", {"unit": unit, "implementation_commit": m["implementation_commit"]})
    cfg = ev.dc.read(batch / "protocol/config.resolved.json")
    meta = {**ev.PROTOCOL, "status": "complete", "unit": unit, "runtime_config": cfg,
        "action_rng_rule": ev.RNG_RULE, "action_rule": ev.random.ACTION_RULE, "implementation_commit": m["implementation_commit"],
        "world_role": ev.WORLD_ROLE, "new_unseen_confirmation": False, "eval_seeds": [unit["world"]],
        "network_constructed": False, "optimizer_constructed": False, "checkpoint_loads": 0, "training_updates": 0,
        "sampling_checks": {"action_count": 52500, "environment_step_count": 10500, "decoder_checks": 10500,
                            "physical_rng_unchanged_by_actions": True},
        "event_source": "post_step_aoi_ms_isclose_1_verified_env_step", "v2i_min": 1, "global_actor_weight": 1,
        "warmup_end_aoi": [100] * 5,
        "cam_bits": 32000, "exogenous_probes": [{"world": unit["world"], "initial": {"world": unit["world"]}, "final": {"world": unit["world"]}}]}
    ev.dc.write(dest / "EVAL_COMPLETE.json", meta)
    ev.dc.write(dest / "provenance.json", {"implementation_commit": m["implementation_commit"], "unit": unit, "runtime_config": cfg})
    np.savez_compressed(dest / "metrics.npz", **fake_arrays())
    return dest


def mark_smoke(batch):
    ev.dc.write(batch / "protocol/smoke.json", {"technical_status": "PASS", "implementation_commit": "random-test"})


def test_native_acceptance_skip_recovery_full_key_and_partial_preservation(batch, monkeypatch):
    m, matrix = ev.load(batch)
    mark_smoke(batch)
    write_native(batch, matrix[0], m)
    monkeypatch.setattr(ev, "sample_rollout", lambda *a, **k: pytest.fail("accepted/native work must never reroll"))
    assert ev.execute(batch, 0)["accepted_existing_native"]
    assert ev.execute(batch, 0)["skipped_verified"]
    assert ev.status(batch, True)["accepted_ids"] == [0]
    # Stream13/world1001 is a distinct key, not covered by stream12/world1001.
    assert 30 in ev.status(batch)["unrun_ids"]
    dest = batch / "units/unit_001/a01"
    dest.mkdir(parents=True)
    with pytest.raises(ValueError, match="partial attempt"):
        ev.execute(batch, 1)
    with pytest.raises(ValueError, match="reason"):
        ev.execute(batch, 1, attempt="a02")
    assert dest.exists()
    path = batch / "records/unit_000.json"
    row = ev.dc.read(path); row["unit"]["action_stream_id"] = 13
    path.unlink(); ev.dc.write(path, row)
    with pytest.raises(ValueError, match="identity changed"):
        ev.execute(batch, 0)


def test_reducer_global_once_cam_deadlines_and_three_million_denominators():
    a = fake_arrays()
    meta = {"event_source": "post_step_aoi_ms_isclose_1_verified_env_step", "v2i_min": 1,
            "global_actor_weight": 1, "cam_bits": 32000, "eval_seeds": [1001]}
    rows = ev.reduce_unit(a, meta, ev.units()[0], "test")
    ep = ev.world_endpoint(rows, 12)
    assert ep["scored_agent_slots"] == 50000 and ep["cam_deadlines"] == 500
    assert ep["joint_global_reward_sum"] == 5000 and ep["joint_global_reward_slot_count"] == 10000
    assert ep["mean_global_reward"] == .5 and ep["mean_cam_percent"] == 0
    assert ep["mean_power_mw"] == 10 and ep["aoi_gt50_fraction"] == 1
    assert rows[0]["aoi_at_cap_count"] == rows[0]["v2i_attempt_count"] == 10000
    assert 60 * ep["scored_agent_slots"] == 3000000 and 60 * ep["cam_deadlines"] == 30000
    with pytest.raises(ValueError, match="shape"):
        ev.reduce_unit({**a, "aoi_ms": a["aoi_ms"][:, :99]}, meta, ev.units()[0], "test")


def stat_rows():
    base = fake_arrays()
    meta = {"event_source": "post_step_aoi_ms_isclose_1_verified_env_step", "v2i_min": 1,
            "global_actor_weight": 1, "cam_bits": 32000, "eval_seeds": [1001]}
    rows = []
    for unit in ev.units():
        part = ev.reduce_unit(base, meta, unit, "test")
        bad = (unit["world"] + unit["action_stream_id"]) % 2
        for p, row in enumerate(part):
            row["aoi_sum_ms"] = (90 if p == bad else 10) * row["slot_count"]
            row["cam_deadline_success_count"] = (10 if p == bad else 90)
        rows.extend(part)
    return rows


def test_scene_first_own_extrema_world_sd_no_training_seed_or_best_stream():
    rows = stat_rows()
    tables = ev.tables_from_rows(rows)
    assert len(tables["unit_world_endpoints"]) == 60 and len(tables["pooled_world_endpoints"]) == 30
    assert all(r["scene_first_worst_aoi_ms"] == 90 for r in tables["unit_world_endpoints"])
    assert all(r["scene_first_worst_aoi_ms"] == 50 for r in tables["pooled_world_endpoints"])
    assert all(r["scene_first_lowest_cam_percent"] == 10 for r in tables["unit_world_endpoints"])
    assert all(r["scene_first_lowest_cam_percent"] == 50 for r in tables["pooled_world_endpoints"])
    assert all(r["world_count"] == 30 and r["sample_sd_ddof"] == 1 and r["n_training_seeds"] == 0 and
               r["training_seed_sd"] is None and r["confidence_interval"] is None for r in tables["stream_summary"])
    assert all(r["world_sample_sd"] == 0 for r in tables["stream_summary"])
    assert all(r["worst_aoi_ms"] == 50 for r in tables["legacy_endpoints"])
    with pytest.raises(ValueError, match="coverage"):
        ev.tables_from_rows(rows[:-1] + [rows[0]])
    varied = copy.deepcopy(rows)
    varied[4]["aoi_sum_ms"] += 50000  # First world mean26 ->27; other29 worlds remain26.
    s = next(r for r in ev.tables_from_rows(varied)["stream_summary"] if r["action_stream_id"] == 12 and r["metric"] == "mean_aoi_ms")
    assert s["world_sample_sd"] == pytest.approx(1/math.sqrt(30))


def test_synthetic60_accept_summary_export_idempotent_and_no_large_members(batch):
    m, matrix = ev.load(batch)
    for unit in matrix:
        write_native(batch, unit, m)
        ev.accept(batch, unit, m, "a01", create=True)
    v = ev.summarize(batch)
    assert v["technical_status"] == "PASS" and v["accepted_units"] == 60 and v["global_joint_slot_count"] == 600000
    assert ev.summarize(batch) == v and ev.export(batch) == v
    with zipfile.ZipFile(v["archive"]) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith("records/unit_")]) == 60
        assert not any(n.endswith((".pt", ".npz")) for n in z.namelist())
    assert len(ev.csv_read(batch / "analysis/a01/world_stream_platoon.csv")) == 300
    assert len(ev.csv_read(batch / "analysis/a01/pooled_world_platoon.csv")) == 150
    (batch / "analysis/a02").mkdir()
    with pytest.raises(ValueError, match="partial analysis"):
        ev.summarize(batch, "a02")


def test_preflight_readonly_configuration_and_manifest_drift(batch, monkeypatch):
    assert ev.preflight(batch)["environment_rollouts"] == 0
    m, _ = ev.load(batch)
    with pytest.raises(ValueError, match="unique"):
        ev.init(m["study_root"], "random-test", "test")
    p = batch / "protocol/units.json"
    values = ev.dc.read(p); values[30] = values[0]
    p.unlink(); ev.dc.write(p, values)
    with pytest.raises(ValueError, match="matrix"):
        ev.load(batch)


def test_reused_queue_adapts_no_policy_sources_and_cpu_cli(batch, monkeypatch):
    import analysis.evaluate_training100_selected as selected
    seen = {}
    def queue(b, device, workers, ids, attempt, reason, *, workflow):
        m, sources, matrix = workflow.load(b)
        assert sources == [] and len(matrix) == 60
        assert workflow.CLI_MODULE == ev.CLI_MODULE and workers == 2 and device == "cpu"
        seen.update(ids=ids, module=workflow.CLI_MODULE)
        return {"technical_status": "PARTIAL"}
    monkeypatch.setattr(selected, "run", queue)
    assert ev.run(batch, ids=[0, 30])["technical_status"] == "PARTIAL"
    assert seen["ids"] == [0, 30]
