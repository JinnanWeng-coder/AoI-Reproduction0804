import copy
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from analysis import run_random_training_baseline as rb
from aoi_v2x_reproduction.runtime.metrics import MetricStore
from aoi_v2x_reproduction.envs.platoon import PaperEnviron


@pytest.fixture
def batch(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "code_check", lambda expected=None: "test-commit")
    result = rb.init(tmp_path, "smoke_test", "test-commit", smoke_episodes=3)
    return rb.Path(result["batch_root"])


def test_bundle_all_four_environments_and_fresh_budget():
    bundle, env = rb.source_bundle()
    assert len(bundle["sources"]) == 24 and env["cam_bits"] == 32000
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
        assert np.all(a.v2v_demand == 32000)
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
    report = rb.run(batch)
    assert (report["episode_rows"], report["platoon_rows"], report["environment_slots"],
            report["agent_slots"], report["cam_deadlines"]) == (6, 30, 600, 3000, 30)
    assert report["smoke_only"] and report["learning_updates"] == 0
    assert counts == {"reset": 2, "start": [0, 1, 2, 0, 1, 2]}
    archive = rb.Path(report["archive"])
    before = archive.read_bytes()
    again = rb.run(batch)
    assert again["skipped_complete"] and archive.read_bytes() == before
    assert counts["reset"] == 2 and not (batch / "RUNNING.lock").exists()
    assert not list(batch.rglob("*.pt")) and not list(batch.rglob("*.npz"))
    with zipfile.ZipFile(archive) as z:
        assert all(not name.endswith((".pt", ".npz", ".log")) for name in z.namelist())
    summary = rb.csv_read(batch / "analysis/a01/summary.csv")
    assert all(r["mean_aoi_ms"] == "" for r in summary if r["window"] == "401_500")
    with pytest.raises(ValueError, match="do not initialize twice"):
        rb.init(batch.parents[2], "smoke_test", "test-commit", 3)


def test_partial_failure_lock_and_explicit_retry(batch, monkeypatch):
    with rb.batch_lock(batch):
        with pytest.raises(FileExistsError):
            rb.run(batch)
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
            r.update(slot_count=500, cam_trials=5, cam_successes=seed-10, aoi_sum_ms=500*(seed+episode), power_mw_sum=1000)
            rows.append({"seed": seed, "episode": episode, **r, **rb.means(r)})
    sums, across, curves = rb.summary_tables(rows, 2500)
    assert len(sums) == 8 and len(curves) == 2500
    window = next(r for r in sums if r["seed"] == 12 and r["window"] == "401_500")
    assert window["episodes_present"] == 100 and window["mean_aoi_ms"] == 12+450.5
    across_aoi = next(r for r in across if r["window"] == "all" and r["metric"] == "mean_aoi_ms")
    assert across_aoi["n_seeds"] == 2 and across_aoi["sample_sd"] == pytest.approx(1/np.sqrt(2))


def test_cli_help_and_import_no_torch_or_learning_modules():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; import analysis.run_random_training_baseline; "
        "assert 'torch' not in sys.modules; assert not any('.algorithms.' in n for n in sys.modules)"],
        cwd=rb.REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-B", "-m", "analysis.run_random_training_baseline", "--help"],
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
    np.testing.assert_array_equal(totals, [5000, 25000, 2500000, 25000])
