"""Synthetic tests only: no training, real checkpoint or PaperEnviron rollout."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis import deployment_contract as dc
from analysis import evaluate_frozen_deployment as ev
from analysis import summarize_frozen_deployment as su
from analysis.deployment_rng import action_seed_words, policy_seed, exogenous_probe
from analysis.submit_deployment import chunks, command
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.envs.platoon import power_penalty


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def source(tmp_path, model="shared_tdec", seed=8, **changes):
    spec = dc.MODELS[model]
    cfg = resolve_config(algorithm="mappo" if spec["family"] == "mappo" else "modified_maddpg_tdec",
                         scenario="p05_n04_g25", seed=seed, episodes=500, steps_per_episode=100,
                         mappo_variant=spec["variant"], mappo_actor_sharing=spec["structure"] == "shared",
                         **changes)
    run = tmp_path / f"{model}_{seed}"
    dump(run / "config.resolved.json", cfg.to_dict())
    dump(run / "COMPLETE.json", {"status": "complete", "final_episode": 500, "reproduction_git_commit": "source_commit"})
    checkpoint = run / ("policy_final.pt" if spec["family"] == "mappo" else "checkpoints/latest.pt")
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"synthetic checkpoint placeholder; never loaded")
    return dc.source_entry(model, seed, checkpoint), cfg


def synthetic_arrays(episodes=2, slots=100, agents=5):
    shape = (1, episodes, slots, agents)
    age = np.minimum(100, np.arange(1, episodes * slots + 1)).reshape(1, episodes, slots, 1)
    age = np.broadcast_to(age, shape).astype(np.float32).copy()
    # Agents with different reset rates; sequence continues across episodes.
    for p in range(1, agents):
        age[..., p] = ((np.arange(episodes * slots) % (p + 2)) + 1).reshape(1, episodes, slots)
    reset = age == 1
    demand = np.full(shape, 16000, dtype=np.float32)
    mode = np.zeros(shape, dtype=np.int64)
    power = np.broadcast_to(np.array([1, 10, 20, 30, 1], dtype=np.float32), shape).copy()
    penalty = np.array([power_penalty(p) for p in power.ravel()]).reshape(shape)
    success = np.zeros(shape, dtype=np.float32)
    success[:, :, -1, 1:] = 1
    return {"aoi_ms": age, "success": success, "remaining_demand": demand, "power_dbm": power,
            "reward_global": np.ones(shape[:-1], dtype=np.float32), "mode": mode,
            "reset_event": reset, "v2i_attempt": mode == 0,
            "reward_task1": (-4.95 * demand / 32000).astype(np.float32),
            "reward_task2": (.05 * reset - age / 20 - penalty).astype(np.float32)}


META = {"event_source": "post_step_aoi_ms_isclose_1_verified_env_step", "global_actor_weight": 1.,
        "cam_bits": 32000, "eval_seeds": [213]}


@pytest.mark.parametrize("phase,count,rows", [("pilot", 60, 300), ("main", 2340, 11700), ("gap-transfer", 144, 720)])
def test_locked_counts_and_unique_unit_keys(phase, count, rows):
    plan = dc.phase_plan(phase, list(range(1001, 1031)) if phase == "main" else None)
    assert len(plan["units"]) == count and plan["platoon_row_count"] == rows
    assert len({tuple(r[k] for k in ("model_id", "execution", "noise_std", "training_seed", "target_scenario", "world", "repeat")) for r in plan["units"]}) == count
    if phase == "pilot":
        assert set(plan["training_seeds"]) == {8, 9} and plan["worlds"] == [213, 214]
        assert sum(u["noise_std"] == 0 for u in plan["units"]) == 4
        assert sum(u["execution"] == "deterministic" and u["noise_std"] is None for u in plan["units"]) == 8
    if phase == "gap-transfer":
        assert {u["model_id"] for u in plan["units"]} == {"algorithm2"}
        assert {u["repeat"] for u in plan["units"]} == {0}


def test_bad_main_worlds():
    for worlds in ([1001] * 30, [213] + list(range(1001, 1030)), None):
        with pytest.raises(ValueError):
            dc.phase_plan("main", worlds)
    with pytest.raises(ValueError):
        dc.phase_plan("main", list(range(30)))


@pytest.mark.parametrize("family,tag", [("mappo", 0x4D415050), ("maddpg", 0xA01)])
def test_repeat0_exact_seed_bridge_and_new_streams(family, tag):
    assert action_seed_words(8, 213, 0, family) == [8, 213, tag]
    old = np.random.default_rng(np.random.SeedSequence([8, 213, tag])).normal(size=16)
    new = np.random.default_rng(np.random.SeedSequence(action_seed_words(8, 213, 0, family))).normal(size=16)
    assert np.array_equal(old, new)
    streams = [np.random.default_rng(np.random.SeedSequence(action_seed_words(8, 213, r, family))).normal(size=16) for r in range(3)]
    assert not np.array_equal(streams[0], streams[1]) and not np.array_equal(streams[1], streams[2])
    assert policy_seed(8, 213, 0) == int(np.random.SeedSequence([8, 213, 0x4D415050]).generate_state(1)[0])
    with pytest.raises(ValueError):
        action_seed_words(8, 213, -1, family)


@pytest.mark.parametrize("model", list(dc.MODELS))
def test_source_models_episode_and_identity(tmp_path, model):
    src, _ = source(tmp_path, model)
    dc.source_check(src)
    assert src["model_id"] == model
    dump(Path(src["run_dir"]) / "COMPLETE.json", {**src["complete"], "final_episode": 2500})
    with pytest.raises(ValueError, match="incomplete"):
        dc.source_check(src)


def test_batch_init_metadata_preflight_and_refuse_overwrite(tmp_path, monkeypatch):
    supplied = []
    for model in dc.MODELS:
        for seed in dc.SEEDS:
            src, _ = source(tmp_path / "sources", model, seed)
            supplied.append({"model_id": model, "training_seed": seed, "checkpoint": src["checkpoint"]["path"]})
    models = tmp_path / "models.json"
    dump(models, supplied)
    monkeypatch.setattr(dc, "code_check", lambda *a: "f" * 40)
    result = dc.init(tmp_path / "diagnostics", "synthetic", models_json=models)
    batch = Path(result["batch_root"])
    manifest, sources = dc.load_batch(batch)
    assert len(sources) == 24 and manifest["protocol"]["training_environment_steps"] == 50000
    assert dc.preflight(batch)["payloads_checked"] is False
    assert not (batch / "protocol/main.json").exists()
    assert dc.plan(batch, "pilot")["policy_world_repeat_count"] == 60
    with pytest.raises(ValueError, match="batch exists"):
        dc.init(tmp_path / "diagnostics", "synthetic", models_json=models)
    with pytest.raises(ValueError, match="unsafe batch"):
        dc.init(tmp_path / "diagnostics", "../historical", models_json=models)


def test_absent_none_sharing_semantics_and_wrong_identity(tmp_path):
    src, _ = source(tmp_path, "independent_tdec")
    path = Path(src["run_dir"]) / "config.resolved.json"
    for value in (None, "absent"):
        raw = copy.deepcopy(src["config"])
        if value == "absent":
            raw.pop("mappo_actor_sharing", None)
        else:
            raw["mappo_actor_sharing"] = value
        dump(path, raw)
        checked = dc.source_entry("independent_tdec", 8, src["checkpoint"]["path"])
        assert checked["sharing_effective"] is False
    with pytest.raises(ValueError, match="wrong MAPPO source"):
        dc.source_entry("shared_tdec", 8, src["checkpoint"]["path"])


def test_cumulative_metrics_denominators_gt50_cam_power_and_nulls():
    arrays = synthetic_arrays()
    rows = ev.reduce_arrays(arrays, META, {"world": 213}, expected=(1, 2, 100, 5))
    assert len(rows) == 5 and sum(r["slot_count"] for r in rows) == 1000
    assert rows[0]["aoi_gt50_count"] == 150  # strict >50, not >=50.
    assert rows[0]["aoi_at_cap_count"] == 101
    assert rows[0]["cam_deadline_success_count"] == 0 and rows[1]["cam_deadline_success_count"] == 2
    assert rows[1]["power_mw_sum"] == pytest.approx(2000)
    assert rows[0]["v2i_reset_count"] == 1 and rows[0]["v2i_attempt_count"] == 200
    assert rows[0]["reward_combined_sum"] == pytest.approx(sum(float(arrays[k][0, ..., 0].sum(dtype=np.float64)) for k in ("reward_task1", "reward_task2")) + 200)
    missing = ev.reduce_arrays(arrays, {**META, "event_source": None}, {"world": 213}, expected=(1, 2, 100, 5))
    assert missing[0]["v2i_reset_count"] is None and missing[0]["v2i_attempt_count"] is None
    assert missing[0]["reward_power_penalty_sum"] is None


@pytest.mark.parametrize("error", ["shape", "nonfinite", "reset", "continuity", "cam", "mode", "ledger"])
def test_array_errors_rejected(error):
    arrays = synthetic_arrays()
    if error == "shape": arrays["power_dbm"] = arrays["power_dbm"][:, :1]
    if error == "nonfinite": arrays["power_dbm"][0, 0, 0, 0] = np.nan
    if error == "reset": arrays["reset_event"][0, 0, 0, 0] = False
    if error == "continuity": arrays["aoi_ms"][0, 1, 0, 0] = 99
    if error == "cam": arrays["success"][0, 0, 0, 0] = 1
    if error == "mode": arrays["mode"][0, 0, 0, 0] = 1
    if error == "ledger": arrays["reward_task2"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError):
        ev.reduce_arrays(arrays, META, {"world": 213}, expected=(1, 2, 100, 5))


def rows_for_aggregation():
    rows = []
    for seed in range(8, 14):
        for world in (1001, 1002):
            for repeat in range(3):
                for p in range(5):
                    r = {"condition": "shared_tdec_stochastic", "model_id": "shared_tdec", "training_seed": seed,
                         "target_scenario": dc.SCENARIO, "world": world, "repeat": repeat, "platoon": p}
                    r.update({field: 100 for pair in ev.SUM_METRICS.values() for field in pair})
                    # Worst platoon changes by world and repeat. Different
                    # denominators distinguish pooling from averaging ratios.
                    r["slot_count"] = (repeat + 1) * 100
                    r["aoi_sum_ms"] = (90 if p == world - 1001 else 10) * r["slot_count"]
                    r["cam_deadline_success_count"] = 20 if p == 4 else 90
                    r["v2i_attempt_count"] = 0
                    r["v2i_reset_count"] = 0
                    rows.append(r)
    return rows


def test_scene_first_pool_before_weakest_world_seed_and_null_variance():
    rows = rows_for_aggregation()
    tables = su.aggregate(rows)
    assert len(tables["world_endpoints"]) == 12 and len(tables["seed_endpoints"]) == 6
    assert tables["seed_endpoints"][0]["scene_first_worst_aoi_ms"] == 90
    assert tables["legacy_seed_endpoints"][0]["worst_aoi_ms"] == 50
    assert tables["world_endpoints"][0]["worst_cam_success_platoons"] == "4"
    assert tables["world_endpoints"][0]["worst_aoi_ms_platoons"] == "0"
    assert tables["world_platoon_repeat_pooled"][0]["slot_count"] == 600
    assert tables["world_endpoints"][0]["mean_v2i_success_given_attempt"] is None
    condition = next(r for r in tables["condition_summary"] if r["metric"] == "scene_first_worst_aoi_ms")
    assert condition["n_training_seeds"] == 6 and condition["df"] == 5 and condition["mean"] == 90
    assert condition["descriptive_t95_lower"] == condition["descriptive_t95_upper"] == 90
    assert condition["claim_support"] == "NOT_ASSESSED"
    single = su.aggregate([r for r in rows if r["repeat"] == 0 and r["training_seed"] == 8])
    assert single["seed_endpoints"][0]["worst_aoi_ms_repeat_sd_mean"] is None
    assert all(r["descriptive_t95_lower"] is None for r in single["condition_summary"])


def test_repeat_variation_is_not_between_world_variation():
    rows = rows_for_aggregation()
    for r in rows:
        r["aoi_sum_ms"] = (10 + r["repeat"] * 5) * r["slot_count"]
    seed = su.aggregate(rows)["seed_endpoints"][0]
    assert seed["worst_aoi_ms_world_sd"] == 0
    assert seed["worst_aoi_ms_repeat_sd_mean"] == 5
    assert seed["worst_aoi_ms_repeat_sd_fixed_platoon_mean"] == 5
    assert seed["scene_first_worst_aoi_ms"] == pytest.approx(100 / 6)  # denominator-weighted, not 15.


def test_history_reserved_nested_metadata_and_missing_roots(tmp_path):
    root = tmp_path / "history"
    dump(root / "config.json", {"selection_validation_seeds": [301, 302], "seed": 9001})
    dump(root / "training.json", {"seed": 42, "scenario": "p05_n04_g25", "episodes": 500})
    dump(root / "nested.json", {"evaluations": [{"eval_seeds": [1100, 1101]}]})
    (root / "eval_s1200-1201_ep100").mkdir()
    (root / "execution.out").write_text("--eval-seeds 1300,1301\n", encoding="utf-8")
    report = dc.scan_history([root], tmp_path / "new_batch")
    assert set([42, 301, 302, 1100, 1101, 1200, 1201, 1300, 1301]) <= set(report["observed_worlds"])
    assert 9001 not in report["observed_worlds"] and not report["scan_errors"]
    (root / "broken.json").write_text("not JSON", encoding="utf-8")
    report = dc.scan_history([root, tmp_path / "missing"], tmp_path / "new_batch")
    assert len(report["scan_errors"]) == 2


def test_array_offsets_resources_and_safe_paths():
    assert chunks(2340) == [(0, 1000), (1000, 1000), (2000, 340)]
    with pytest.raises(ValueError): chunks(60, 1002)
    cmd = command(Path("/repo"), Path("/new"), "main", 1000, 1000, "cpu", "account", 3, 8, "10G", "01:00:00", "", "afterok:1")
    assert "--array=0-999%3" in cmd and "--export=ALL,AOI_TASK_OFFSET=1000" in cmd
    assert "--gres" not in cmd and "account" in cmd and "afterok:1" in cmd
    with pytest.raises(ValueError): ev.unit_dir(Path("/batch"), {"phase": "pilot", "unit_id": 0}, "../../old")


def test_native_protocol_source_and_repeat_validation(tmp_path):
    src, _ = source(tmp_path)
    unit = dc.phase_plan("pilot")["units"][0]
    metadata = {"status": "complete", "training_seed": 8, "eval_seeds": [213], "eval_protocol": "sequential_warm",
                "eval_warmup_episodes": 5, "eval_episodes": 100, "target_scenario": dc.SCENARIO,
                "source_config_hash": src["source_config_hash"], "training_source_commit": "source_commit", "action_repeat": 0,
                "environment_seeds": [213], "action_seed_words": [unit["action_seed_words"]],
                "reproduction_git_commit": "implementation", "reproduction_git_dirty": False,
                "policy_episode": 500, "mappo_eval_mode": "stochastic", "policy_parameters_unchanged": True,
                "policy_sampling_seeds": [unit["policy_sampling_seed"]], "exogenous_probes": [{"world": 213}],
                "source_scenario": dc.SCENARIO, "cam_bits": 32000., "global_actor_weight": 1.,
                "actor_sharing": True, "mappo_variant": "tdec", "actor_network_count": 1,
                "world_role": "reused_development_worlds",
                "raw_metric_axes": {"aoi_ms": ["eval_seed", "scored_episode", "slot", "agent"]}}
    ev.native_check(metadata, unit, src, "implementation")
    for key, value in (("action_repeat", 1), ("policy_episode", 2500), ("eval_warmup_episodes", 0),
                       ("training_source_commit", None), ("policy_sampling_seeds", [1]), ("reproduction_git_commit", "other")):
        with pytest.raises(ValueError): ev.native_check({**metadata, key: value}, unit, src, "implementation")


def test_accept_existing_complete_and_reject_source_or_array_change(tmp_path):
    src, _ = source(tmp_path)
    unit = dc.phase_plan("pilot")["units"][0]
    dest = ev.unit_dir(tmp_path, unit)
    native = dest / "native"
    native.mkdir(parents=True)
    arrays = synthetic_arrays(100)
    np.savez_compressed(native / "metrics.npz", **arrays)
    dump(native / "EVAL_COMPLETE.json", {**META, "policy": str(src["checkpoint"]["path"]), "exogenous_probes": [{"world": 213}]})
    dump(native / "provenance.json", {"reproduction_git_commit": "test"})
    # Native protocol is tested separately; this test isolates row/marker reuse.
    from unittest.mock import patch
    with patch.object(ev, "native_check"):
        record, rows, _ = ev.accept(tmp_path, unit, src, "a01", {"implementation_commit": "test"}, create=True)
        assert record["scored_agent_slots"] == 50000 and len(rows) == 5
        ev.accept(tmp_path, unit, src, "a01", {"implementation_commit": "test"})
        with pytest.raises(ValueError): ev.accept(tmp_path, unit, {**src, "training_commit": "changed"}, "a01", {"implementation_commit": "test"})
        arrays["reward_global"] += 1
        np.savez_compressed(native / "metrics.npz", **arrays)
        with pytest.raises(ValueError): ev.accept(tmp_path, unit, src, "a01", {"implementation_commit": "test"})


class FakeEnvironment:
    """Only an RNG/state test double, not the project simulation environment."""
    def __init__(self, config):
        self.config = config
        self.v2i_min = 540.
        self.vehicles = [SimpleNamespace(position=[0., 0.], direction="u", velocity=1.)]
        self.v2i_shadowing = np.zeros(1)
        self.v2v_shadowing = np.zeros((1, 1))
        self.reset_world(0)

    def reset_world(self, world):
        self.rng = np.random.default_rng(world)
        self.age = np.full(5, 100.)
        self.v2i_channels_fast = self.rng.normal(size=(1, 3))

    def start_episode(self, index):
        self.v2i_channels_fast = self.rng.normal(size=(1, 3))
        return np.zeros((5, self.config.state_dim), np.float32)

    def step(self, actions):
        external = self.rng.uniform(size=5)
        mode = (actions[:, 1] > 0).astype(int)
        reset = (mode == 0) & (external > .5)
        self.age = np.where(reset, 1, np.minimum(100, self.age + 1))
        power = 1 + (actions[:, 2] + 1) * 14.5
        info = {"aoi_ms": self.age.copy(), "success": np.zeros(5), "remaining_demand": np.full(5, 32000.),
                "power_dbm": power, "rb": np.zeros(5, int), "mode": mode,
                "v2i_rate": reset * 600., "selected_interference_db": np.zeros(5)}
        cost = np.array([power_penalty(v) for v in power])
        t1 = -4.95 * np.ones(5) - cost * (mode == 1)
        t2 = .05 * reset - self.age / 20 - cost * (mode == 0)
        self.v2i_channels_fast = self.rng.normal(size=(1, 3))
        return np.zeros((5, self.config.state_dim), np.float32), 1., t1, t2, False, info


class FakeFrozenPolicy:
    def __init__(self, config, states, device):
        self.device = device
        self.actors = []
        self.count = 1 if config.mappo_actor_sharing else 5
    def snapshot(self): return []
    def unchanged(self, snapshot): return True
    def act(self, observations, deterministic=False):
        return (np.zeros((5, 3), np.float32) if deterministic else (torch.rand((5, 3)) * 2 - 1).numpy())


def test_existing_mappo_loop_repeat_deterministic_and_combined_extension(tmp_path, monkeypatch):
    from analysis import evaluate_mappo_zero_shot as zs
    src, cfg = source(tmp_path, "shared_combined")
    monkeypatch.setattr(zs.torch, "load", lambda *a, **k: {"actors": [], "episode": 500})
    monkeypatch.setattr(zs, "_validate_mappo_policy_artifact", lambda *a: (Path(src["run_dir"]), cfg, src["complete"]))
    monkeypatch.setattr(zs, "PaperEnviron", FakeEnvironment)
    monkeypatch.setattr(zs, "FrozenActorPolicy", FakeFrozenPolicy)
    results = []
    for repeat, deterministic in ((0, False), (1, False), (0, True)):
        result = zs.evaluate_zero_shot(Path(src["checkpoint"]["path"]), tmp_path / f"eval_{repeat}_{deterministic}",
                                      dc.SCENARIO, "cpu", [213], 1, action_repeat=repeat, deterministic=deterministic, deployment=True)
        results.append(result)
        assert result["mappo_variant"] == "combined" and result["policy_episode"] == 500
    assert results[0]["exogenous_probes"] == results[1]["exogenous_probes"] == results[2]["exogenous_probes"]
    assert results[0]["policy_sampling_seeds"] != results[1]["policy_sampling_seeds"]
    assert results[2]["mappo_eval_mode"] == "deterministic"
    with np.load(Path(results[0]["eval_dir"]) / "metrics.npz") as a, np.load(Path(results[1]["eval_dir"]) / "metrics.npz") as b:
        assert not np.array_equal(a["action_normalized"], b["action_normalized"])


def test_existing_maddpg_loop_noise_repeat_and_gap_adapter(tmp_path, monkeypatch):
    from analysis import evaluate_maddpg_e1 as md
    src, cfg = source(tmp_path, "algorithm2")
    class Agent:
        def __init__(self): self.actor = torch.nn.Linear(1, 1)
        def load_state_dict_full(self, state): pass
        def state_dict_full(self): return {"actor": self.actor.state_dict()}
        def choose_action(self, observation, explore, noise_std, rng):
            assert explore is False
            return np.clip(rng.normal(0, noise_std, 3), -.999, .999).astype(np.float32) if noise_std else np.zeros(3, np.float32)
    class Learner:
        def load_state_dict_full(self, state): pass
        def state_dict_full(self): return {}
    payload = {"config": cfg.to_dict(), "config_hash": cfg.canonical_hash(), "episode": 500,
               "completed": True, "algorithm": "modified_maddpg_tdec", "agents": [{}] * 5, "learner": {}}
    monkeypatch.setattr(md.torch, "load", lambda *a, **k: payload)
    monkeypatch.setattr(md, "_make_system", lambda config: (FakeEnvironment(config), [Agent() for _ in range(5)], Learner(), None, None, torch.device("cpu")))
    run = Path(src["run_dir"])
    override = {"run_dir": run, "resolved_run_dir": run, "checkpoint": Path(src["checkpoint"]["path"]),
                "canonical_run_name": run.name, "historical_run_name": run.name, "algorithm_label": "algorithm2",
                "is_symlink": False, "complete": src["complete"], "manifest": {}}
    results = []
    for repeat, noise, gap in ((0, .3, dc.SCENARIO), (1, .3, dc.SCENARIO), (0, 0., dc.SCENARIO), (0, .3, "p05_n04_g05")):
        result = md.evaluate_maddpg_e1("modified_maddpg_tdec", noise, 8, tmp_path, tmp_path / f"md_{repeat}_{noise}_{gap}",
                                      "cpu", [213], 1, 5, 500, action_repeat=repeat, target_scenario=gap,
                                      source_override=override, deployment=True)
        assert result["checkpoint_episode"] == 500 and result["optimizer_steps"] == 0
        assert result["source_scenario"] == dc.SCENARIO and result["target_scenario"] == gap
        results.append(result)
    assert results[0]["exogenous_probes"] == results[1]["exogenous_probes"] == results[2]["exogenous_probes"]
    assert results[0]["action_seed_words"] != results[1]["action_seed_words"]
    with np.load(Path(results[0]["eval_dir"]) / "metrics.npz") as a:
        assert np.array_equal(a["reset_event"], a["aoi_ms"] == 1)


def test_world_lock_fixed_before_main_and_reserved_hits_preserved(tmp_path, monkeypatch):
    diagnostics = tmp_path / "diagnostics"
    batch = diagnostics / "actor-sharing-study/D1-frozen-deployment/P5_N4_gap25/test"
    batch.mkdir(parents=True)
    monkeypatch.setattr(dc, "load_batch", lambda *a: ({"diagnostics_root": str(diagnostics), "implementation_commit": "test"}, []))
    with pytest.raises(ValueError, match="confirm-reviewed"):
        dc.lock_worlds(batch, [], reviewed=False)
    dump(diagnostics / "existing-evidence/config.json", {"selection_validation_seeds": [1001]})
    with pytest.raises(ValueError, match="blocked"):
        dc.lock_worlds(batch, [], list(range(1001, 1031)), reviewed=True)
    assert dc.read(batch / "protocol/world_use_audit.json")["candidate_hits"] == [1001]
    assert not (batch / "protocol/main.json").exists()
    result = dc.lock_worlds(batch, [], candidate_start=1001, reviewed=True, audit_name="world_use_audit_02")
    assert result["worlds"] == list(range(1002, 1032))
    assert dc.plan(batch, "main")["policy_world_repeat_count"] == 2340
    with pytest.raises(ValueError, match="already fixed"):
        dc.lock_worlds(batch, [], reviewed=True, audit_name="world_use_audit_03")


def test_summary_pipeline_full_synthetic_pilot_and_no_missing_units(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    batch.mkdir()
    units = dc.phase_plan("pilot")["units"]
    sources = []
    for model in dc.MODELS:
        for seed in (8, 9):
            s, _ = source(tmp_path, model, seed)
            sources.append(s)
    monkeypatch.setattr(dc, "load_batch", lambda *a: ({"implementation_commit": "test", "diagnostics_root": str(tmp_path)}, sources))
    monkeypatch.setattr(dc, "plan", lambda *a: dc.phase_plan("pilot"))
    monkeypatch.setattr(dc, "source_check", lambda *a, **k: None)
    monkeypatch.setattr(dc, "code_check", lambda *a: "test")
    def fake_accept(batch, unit, src, attempt, manifest):
        rows = ev.reduce_arrays(synthetic_arrays(100), META, {**unit, "family": src["family"]})
        return {"attempt": attempt, "native_evaluation_dir": str(ev.unit_dir(batch, unit) / "native")}, rows, {"world": unit["world"], "rng_state": "fixed"}
    monkeypatch.setattr(su, "accept", fake_accept)
    for unit in units:
        dc.write(ev.unit_dir(batch, unit) / "DEPLOYMENT_COMPLETE.json", {})
    result = su.summarize(batch, "pilot", make_plots=False)
    assert result["technical_status"] == "PASS" and result["policy_world_repeats"] == 60
    assert result["scored_agent_slots"] == 3000000 and result["cam_deadlines"] == 30000
    assert len(ev.csv_read(batch / "analysis/pilot/a01/world_repeat_platoon.csv")) == 300
    assert len(ev.csv_read(batch / "analysis/pilot/a01/seed_endpoints.csv")) == 14
    with pytest.raises(FileExistsError): su.summarize(batch, "pilot", make_plots=False)
    (ev.unit_dir(batch, units[0]) / "DEPLOYMENT_COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="accepted attempt"):
        su.summarize(batch, "pilot", attempt="a02", make_plots=False)
    assert dc.read(batch / "analysis/pilot/a02/verification.json")["technical_status"] == "FAIL"


def test_plot_pipeline_png_only_and_csv_sources(tmp_path):
    base = rows_for_aggregation()
    full = []
    for condition, (model, execution, noise, repeats) in dc.CONDITIONS.items():
        full.extend({**r, "condition": condition, "model_id": model} for r in base if r["repeat"] < repeats)
    tables = su.aggregate(full)
    su.plots(tables, tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 4
    assert len(list(tmp_path.glob("*_source.csv"))) == 4
    assert len(list(tmp_path.glob("*_qa.json"))) == 4
    assert not list(tmp_path.glob("*.pdf"))
