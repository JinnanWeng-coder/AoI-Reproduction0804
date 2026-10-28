import copy
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from analysis import cam96k_random_training as rb
from aoi_v2x_reproduction.runtime.metrics import MetricStore
from aoi_v2x_reproduction.envs.platoon import PaperEnviron


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "code_check", lambda expected=None: "test-commit")
    result = rb.init(tmp_path, "smoke_test", "test-commit", smoke_episodes=3)
    return rb.Path(result["batch_root"])


def test_bundle_all_four_environments_and_fresh_budget():
    bundle, env = rb.source_bundle()
    assert len(bundle["sources"]) == 24 and env["cam_bits"] == 96000
    assert (env["power_min_dbm"], env["power_max_dbm"], env["steps_per_episode"]) == (1, 30, 100)
    for seed in rb.SEEDS:
        cfg = rb.derive(bundle, seed, 2500)
        assert cfg.seed == seed and cfg.episodes == 2500 and cfg.checkpoint_mode == "none"
        assert {k: cfg.to_dict()[k] for k in rb.ENV_FIELDS} == env


def test_draws_uniform_discrete_and_dbm_and_decoder():
    bundle, _ = rb.source_bundle()
    cfg = rb.derive(bundle, 12, 3)
    env = PaperEnviron(cfg)
    rng = rb.action_rng(12)
    initial = copy.deepcopy(env.rng.bit_generator.state)
    decoded = []
    for _ in range(6000):
        actions, expected = rb.draw_actions(cfg, rng)
        np.testing.assert_allclose(env.decode_actions(actions), expected, rtol=0, atol=1e-12)
        decoded.append(expected)
    a = np.concatenate(decoded)
    assert env.rng.bit_generator.state == initial
    assert np.all(np.abs(np.bincount(a[:, 0].astype(int)) / len(a) - 1/3) < .015)
    assert np.all(np.abs(np.bincount(a[:, 1].astype(int)) / len(a) - .5) < .015)
    assert 1 <= a[:, 2].min() < 1.1 and 29.9 < a[:, 2].max() <= 30
    assert a[:, 2].mean() == pytest.approx(15.5, abs=.15)
    assert a[:, 2].var() == pytest.approx(29**2 / 12, abs=1.5)
    for seed in rb.SEEDS:
        one, two = rb.action_rng(seed), rb.action_rng(seed)
        for _ in range(10):
            np.testing.assert_array_equal(rb.draw_actions(cfg, one)[0], rb.draw_actions(cfg, two)[0])
    assert not np.array_equal(rb.draw_actions(cfg, rb.action_rng(12))[0], rb.draw_actions(cfg, rb.action_rng(13))[0])


def test_actions_do_not_change_exogenous_stream_and_aoi_carries():
    bundle, _ = rb.source_bundle()
    cfg = rb.derive(bundle, 12, 3)
    a, b = PaperEnviron(cfg), PaperEnviron(cfg)
    a.reset_world(12); b.reset_world(12)
    rng = rb.action_rng(12)
    for episode in range(3):
        old_aoi = a.aoi.copy()
        a.start_episode(episode); b.start_episode(episode)
        np.testing.assert_array_equal(a.aoi, old_aoi)
        assert np.all(a.v2v_demand == 96000)
        for _ in range(100):
            a.step(rb.draw_actions(cfg, rng)[0])
            b.step(np.tile([0., -0.5, 1.], (5, 1)))
            assert rb.exogenous_probe(a) == rb.exogenous_probe(b)
            np.testing.assert_array_equal(a.v2v_channels_fast, b.v2v_channels_fast)


def test_reducer_matches_training_metrics_and_uses_deadline():
    infos = [{"aoi_ms": np.arange(1, 6, dtype=np.float32) + i,
              "power_dbm": np.full(5, 1 if i % 2 == 0 else 30, np.float32),
              "success": np.zeros(5) if i < 99 else np.array([1, 0, 1, 0, 1]),
              "remaining_demand": np.array([0, 7, 0, 7, 0]) if i == 99 else np.full(5, 7),
              "mode": np.full(5, i % 2)} for i in range(100)]
    # Keep AoI within its native cap for the synthetic comparison.
    for row in infos: row["aoi_ms"] = np.minimum(row["aoi_ms"], 100)
    t1 = np.full((100, 5), -.3, dtype=np.float32)
    t2 = np.full((100, 5), .1, dtype=np.float32)
    g = np.linspace(-.001, .001, 100)
    p, e = rb.reduce_episode(12, 1, infos, t1, t2, g)
    metrics = MetricStore(5, 100)
    metrics.append_episode(infos, t1, t2, g)
    arrays = metrics.arrays()
    assert e["cam_successes"] == 3 and e["cam_trials"] == 5 and e["cam_completion_percent"] == 60
    assert e["slot_count"] == 500
    np.testing.assert_allclose([r["mean_aoi_ms"] for r in p], arrays["mean_aoi_ms_episode_agent"][0], atol=1e-5)
    np.testing.assert_array_equal([r["cam_successes"] for r in p], arrays["endpoint_cam_episode_agent"][0])
    expected_reward = (t1.astype(float) + t2.astype(float) + g.astype(np.float32).astype(float)[:, None]).mean()
    assert e["mean_reward"] == pytest.approx(expected_reward, abs=1e-15)
    assert e["mean_power_mw"] == pytest.approx((10**.1 + 1000)/2)
    assert e["mean_power_mw"] != pytest.approx(10**(15.5/10))


def test_smoke_lifecycle_counts_export_and_skip(batch, monkeypatch):
    counts = {"reset": 0, "start": []}
    class Spy(PaperEnviron):
        def reset_world(self, seed=None):
            counts["reset"] += 1
            return super().reset_world(seed)
        def start_episode(self, episode_index, update_mobility=True):
            counts["start"].append(episode_index)
            return super().start_episode(episode_index, update_mobility)
    monkeypatch.setattr(rb, "PaperEnviron", Spy)
    rb.run(batch)
    report = rb.summarize(batch)
    assert (report["episode_rows"], report["platoon_rows"], report["environment_slots"],
            report["agent_slots"], report["cam_deadlines"]) == (18, 90, 1800, 9000, 90)
    assert report["smoke_only"] and report["learning_updates"] == 0
    assert counts == {"reset": 6, "start": [0, 1, 2] * 6}
    archive = rb.Path(report["archive"])
    before = archive.read_bytes()
    again = rb.run(batch)
    assert all(r["skipped_complete"] for r in again["accepted_requested"])
    assert rb.summarize(batch)["skipped_complete"] and archive.read_bytes() == before
    assert counts["reset"] == 6 and not list(batch.glob("*_RUNNING.lock"))
    assert not list(batch.rglob("*.pt")) and not list(batch.rglob("*.npz"))
    with zipfile.ZipFile(archive) as z:
        assert all(not name.endswith((".pt", ".npz", ".log")) for name in z.namelist())
    summary = rb.csv_read(batch / "analysis/a01/summary.csv")
    assert all(r["mean_aoi_ms"] == "" for r in summary if r["window"] == "401_500")
    with pytest.raises(ValueError, match="do not initialize twice"):
        rb.init(batch.parents[2], "smoke_test", "test-commit", 3)


def test_partial_failure_lock_and_explicit_retry(batch, monkeypatch):
    with rb.batch_lock(batch, 12):
        with pytest.raises(FileExistsError):
            rb.run(batch, seeds=[12])
    real_env = rb.PaperEnviron
    class Broken(PaperEnviron):
        def step(self, actions): raise RuntimeError("synthetic failure")
    monkeypatch.setattr(rb, "PaperEnviron", Broken)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        rb.run(batch, seeds=[12])
    assert (batch / "runs/seed12/a01/STARTED.json").exists()
    assert not (batch / "records/seed12.json").exists()
    monkeypatch.setattr(rb, "PaperEnviron", real_env)
    with pytest.raises(ValueError, match="partial attempt preserved"):
        rb.run(batch, seeds=[12])
    with pytest.raises(ValueError, match="requires a reason"):
        rb.run(batch, seeds=[12], attempt="a02")
    rb.run(batch, seeds=[12], attempt="a02", reason="synthetic technical failure")
    assert rb.read(batch / "records/seed12.json")["run_dir"].endswith("a02")
    assert (batch / "runs/seed12/a01").exists()


def test_corruption_and_protocol_changes_fail(batch):
    rb.run(batch)
    p = batch / "runs/seed12/a01/episode_metrics.csv"
    text = p.read_text()
    p.write_text(text.replace("uniform_random", "other", 1))
    with pytest.raises(ValueError, match="condition"):
        rb.run(batch, seeds=[12])
    m = rb.read(batch / "protocol/manifest.json")
    m["episodes"] = 20
    (batch / "protocol/manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="budget"):
        rb.load(batch)


def test_summary_windows_and_seed_sd_no_slot_pseudoreplication():
    rows = []
    for seed in rb.SEEDS:
        for episode in range(1, 2501):
            r = {k: 0 for k in rb.SUM_FIELDS}
            r.update(slot_count=500, cam_trials=5, cam_successes=seed % 6, aoi_sum_ms=500*(seed+episode), power_mw_sum=1000)
            rows.append({"seed": seed, "episode": episode, **r, **rb.means(r)})
    sums, across, curves = rb.summary_tables(rows, 2500)
    assert len(sums) == 24 and len(curves) == 2500
    window = next(r for r in sums if r["seed"] == 12 and r["window"] == "401_500")
    assert window["episodes_present"] == 100 and window["mean_aoi_ms"] == 12+450.5
    across_aoi = next(r for r in across if r["window"] == "all" and r["metric"] == "mean_aoi_ms")
    assert across_aoi["n_seeds"] == 6 and across_aoi["sample_sd"] == pytest.approx(np.std(rb.SEEDS, ddof=1))


def test_cli_help_and_import_no_torch_or_learning_modules():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; import analysis.cam96k_random_training; "
        "assert 'torch' not in sys.modules; assert not any('.algorithms.' in n for n in sys.modules)"],
        cwd=rb.REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-B", "-m", "analysis.cam96k_random_training", "--help"],
                            cwd=rb.REPO, capture_output=True, text=True)
    assert result.returncode == 0 and "preflight" in result.stdout


def test_source_disagreement_is_not_silently_defaulted(monkeypatch):
    original = rb.read(rb.REPO / rb.BUNDLE)
    changed = copy.deepcopy(original)
    changed["templates"]["algorithm2"]["cam_bits"] = 64000
    monkeypatch.setattr(rb, "read", lambda path: changed)
    with pytest.raises(ValueError, match="environments differ"):
        rb.source_bundle()


def test_native_complete_without_record_is_accepted_without_rerun(batch, monkeypatch):
    rb.run(batch, seeds=[12])
    (batch / "records/seed12.json").unlink()  # Synthetic post-check interruption.
    class NoRollout(PaperEnviron):
        def reset_world(self, *args): pytest.fail("existing complete unit must not roll out")
    monkeypatch.setattr(rb, "PaperEnviron", NoRollout)
    result = rb.run(batch, seeds=[12])
    assert result["accepted_requested"][0]["accepted_existing_native_completion"]


def test_formal_denominators_from_synthetic_tables_without_rollout():
    totals = np.zeros(4, dtype=np.int64)
    for seed in rb.SEEDS:
        p = {"condition": "uniform_random", "seed": seed, "episode": 1, "platoon": 0,
             **{k: 0 for k in rb.SUM_FIELDS}}
        p.update(aoi_sum_ms=1000., slot_count=100, cam_successes=1, cam_trials=1,
                 power_mw_sum=1000., power_dbm_sum=1000.)
        p.update(rb.means(p))
        pr = [{**p, "episode": e, "platoon": a} for e in range(1, 2501) for a in range(5)]
        er = [{"condition": "uniform_random", "seed": seed, "episode": e,
               **rb.merge_rows(pr[(e-1)*5:e*5])} for e in range(1, 2501)]
        er, pr = rb.validate_tables(er, pr, seed, 2500)
        totals += [len(er), len(pr), int(sum(r["slot_count"] for r in pr)), int(sum(r["cam_trials"] for r in pr))]
    np.testing.assert_array_equal(totals, [15000, 75000, 7500000, 75000])


def test_same_environment_contract_as_all_six_hpc_methods(tmp_path):
    from analysis import cam96k_hpc as learned
    bundle, env = rb.source_bundle()
    for source in learned.sources():
        cfg, _ = learned.derive(source, tmp_path)
        assert {k: cfg.to_dict()[k] for k in rb.ENV_FIELDS} == env
    for seed in rb.SEEDS:
        cfg = rb.derive(bundle, seed, 2500)
        assert {k: cfg.to_dict()[k] for k in rb.ENV_FIELDS} == env
        assert cfg.global_actor_weight == 1 and cfg.slot_ms == 1


def test_historical_rng_sampler_and_reducer_exactly_preserved():
    # Read the real historical implementation as a test oracle; never roll out it.
    import types
    oracle = types.ModuleType("uniform_historical_oracle")
    oracle.__file__ = str(rb.REPO / "analysis/run_random_training_baseline.py")
    exec(compile(rb.git("show", f"{rb.HISTORICAL_UNIFORM}:analysis/run_random_training_baseline.py"),
                 oracle.__file__, "exec"), oracle.__dict__)
    bundle, _ = rb.source_bundle()
    from analysis.cam96k_hpc_uniform import draw_actions as frozen_sampler
    for seed in rb.SEEDS:
        cfg = rb.derive(bundle, seed, 3)
        a, b, c = rb.action_rng(seed), oracle.action_rng(seed), rb.action_rng(seed)
        for _ in range(10):
            new, old, frozen = rb.draw_actions(cfg, a), oracle.draw_actions(cfg, b), frozen_sampler(cfg, c)
            for col in range(2):
                np.testing.assert_array_equal(new[col], old[col])
                np.testing.assert_array_equal(new[col], frozen[col])
        env = PaperEnviron(cfg)
        env.reset_world(seed)
        env.start_episode(0)
        records, t1, t2, g = [], [], [], []
        for _ in range(100):
            _, rg, r1, r2, _, info = env.step(rb.draw_actions(cfg, a)[0])
            records.append(copy.deepcopy(info)); t1.append(r1); t2.append(r2); g.append(rg)
        nr, ne = rb.reduce_episode(seed, 1, records, t1, t2, g)
        oldr, olde = oracle.reduce_episode(seed, 1, records, t1, t2, g)
        assert all({k: row[k] for k in oldrow} == oldrow for row, oldrow in zip(nr, oldr))
        assert {k: ne[k] for k in olde} == olde


def test_short_serial_vs_two_cpu_processes_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "code_check", lambda expected=None: "test-commit")
    serial = rb.Path(rb.init(tmp_path, "smoke_serial", "test-commit", 3)["batch_root"])
    parallel = rb.Path(rb.init(tmp_path, "smoke_parallel", "test-commit", 3)["batch_root"])
    rb.run(serial, [8, 9])
    rb.preflight(parallel)
    workers = []
    for seed in (8, 9):
        script = ("from analysis import cam96k_random_training as r; "
                  "r.code_check=lambda expected=None:'test-commit'; "
                  f"r.run({str(parallel)!r}, [{seed}])")
        workers.append(subprocess.Popen([sys.executable, "-B", "-c", script], cwd=rb.REPO,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    for worker in workers:
        out, err = worker.communicate(timeout=60)
        assert worker.returncode == 0, out + err
    for seed in (8, 9):
        for name in ("episode_metrics.csv", "episode_platoon_metrics.csv", "exogenous_probes.json"):
            assert (serial / f"runs/seed{seed}/a01" / name).read_bytes() == (parallel / f"runs/seed{seed}/a01" / name).read_bytes()


def test_submit_cpu_array_dependency_receipts_and_duplicate_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "code_check", lambda expected=None: "test-commit")
    batch = rb.Path(rb.init(tmp_path, "formal_config_only", "test-commit")["batch_root"])
    dry = rb.submit(batch)
    assert not (batch / "records/submission_intent.json").exists()
    assert dry["max_concurrency"] == 6 and dry["initial_concurrency"] == 2
    assert "--array=0-5%2" in dry["commands"]["rollouts"]
    assert all(not any("gres" in arg or "gpu" in arg for arg in args) for args in dry["commands"].values())
    seen = []
    def sbatch(argv, **kwargs):
        seen.append((argv, kwargs["env"].copy()))
        return str(100 + len(seen)) + ";test_cluster\n"
    monkeypatch.setattr(rb.subprocess, "check_output", sbatch)
    result = rb.submit(batch, actually=True)
    assert result["jobs"] == {"rollouts": "101", "summary": "102"}
    assert "--dependency=afterok:101" in seen[1][0]
    assert [env["AOI_RANDOM_STAGE"] for _, env in seen] == ["rollouts", "summary"]
    assert all(env["PROJECT_DIR"] == str(rb.REPO) and env["AOI_EXPECTED_COMMIT"] == "test-commit" for _, env in seen)
    with pytest.raises(ValueError, match="submission already"):
        rb.submit(batch, actually=True)


def test_unknown_sbatch_outcome_blocks_resubmission(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "code_check", lambda expected=None: "test-commit")
    batch = rb.Path(rb.init(tmp_path, "formal_uncertain", "test-commit")["batch_root"])
    monkeypatch.setattr(rb.subprocess, "check_output", lambda *a, **kw: "unexpected_response")
    with pytest.raises(ValueError, match="uncertain sbatch"):
        rb.submit(batch, actually=True)
    assert (batch / "records/submission_intent.json").exists()
    with pytest.raises(ValueError, match="submission already"):
        rb.submit(batch, actually=True)


def test_missing_seed_summary_refused_and_smoke_not_submitted(batch):
    rb.run(batch, [8])
    with pytest.raises(FileNotFoundError):
        rb.summarize(batch)
    with pytest.raises(ValueError, match="formal only"):
        rb.submit(batch)
