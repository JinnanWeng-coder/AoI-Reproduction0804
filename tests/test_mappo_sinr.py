"""Synthetic policies + short real physics; no historical models or formal evaluation."""
import copy
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis import inspect_selected_mappo_sinr as cli
from analysis import evaluate_mappo_zero_shot as zs
from analysis.mappo_sinr import SinrEnvironment, SinrLogger
from aoi_v2x_reproduction.config import resolve_config
from aoi_v2x_reproduction.envs.platoon import PaperEnviron
from aoi_v2x_reproduction.runtime import checkpointing as cp, training_selection as ts
from tests.test_training100 import FakeTrainer
from tests.test_frozen_deployment import dump


def cfg(root):
    return resolve_config(algorithm="mappo", scenario="p05_n04_g25", seed=13,
                          mappo_variant="tdec", mappo_actor_sharing=True, device="cpu",
                          actor_hidden=[8, 4], episodes=2500, steps_per_episode=2,
                          output_root=str(root), run_name="synthetic_sinr")


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    elif isinstance(a, (np.ndarray, torch.Tensor)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    else: assert a == b


def test_observer_is_bitwise_passive_and_captures_executed_not_next_channels(tmp_path):
    c = cfg(tmp_path)
    plain, observed = PaperEnviron(c), SinrEnvironment(c)
    plain.reset_world(1001); observed.reset_world(1001)
    actions = np.array([[-.8, -1, -.3], [.0, 1, .2], [.8, 1, -.1], [-.8, -1, .8], [.8, 1, .9]])
    rng = cp.capture_rng_state()
    for e in range(3):
        equal(plain.start_episode(e), observed.start_episode(e))
        for _ in range(c.steps_per_episode):
            old = observed.v2i_channels_fast.copy()
            decoded = observed.decode_actions(actions)
            equal(plain.step(actions), observed.step(actions))
            equal(plain.rng.bit_generator.state, observed.rng.bit_generator.state)
            np.testing.assert_array_equal(plain.v2i_channels_fast, observed.v2i_channels_fast)
            np.testing.assert_array_equal(plain.v2v_channels_fast, observed.v2v_channels_fast)
            p, rb = 0, int(decoded[0, 0])
            expected = 10**((decoded[p, 2]-old[0, rb]+observed.veh_ant_gain+
                             observed.bs_ant_gain-observed.bs_noise_figure)/10)
            assert observed.sinr_last["v2i_signal_mw"][p] == expected
            assert not np.array_equal(old, observed.v2i_channels_fast)
    equal(rng, cp.capture_rng_state())


def test_known_signal_noise_and_three_receiver_bottleneck(tmp_path):
    e = SinrEnvironment(cfg(tmp_path)); e.reset_world(1001)
    e.sig2 = 1.
    e.v2i_channels_fast.fill(1000.)
    e.v2v_channels_fast.fill(1000.)  # Suppress cross interference below float64 precision.
    decoded = np.array([[0, 0, 10], [1, 1, 10], [2, 0, 10], [0, 1, 10], [2, 0, 10]], float)
    for p in (0, 2, 4):
        e.v2i_channels_fast[p*4, int(decoded[p, 0])] = 10+e.veh_ant_gain+e.bs_ant_gain-e.bs_noise_figure
    for p in (1, 3):
        for f, g in enumerate((.1, 1., 10.)):
            e.v2v_channels_fast[p*4, p*4+f+1, int(decoded[p, 0])] = (
                10+2*e.veh_ant_gain-e.veh_noise_figure-10*np.log10(g))
    native = e._compute_metrics(decoded)
    # The two V2I leaders sharing RB2 each see the other's unit signal plus unit noise.
    np.testing.assert_allclose(e.sinr_last["v2i_sinr_linear"][[0, 2, 4]], [1., .5, .5])
    np.testing.assert_allclose(e.sinr_last["v2i_interference_noise_mw"][[0, 2, 4]], [1., 2., 2.])
    np.testing.assert_allclose(e.sinr_last["v2v_sinr_linear"][[1, 3]], [[.1, 1., 10.]]*2)
    np.testing.assert_allclose(native["v2v_rate"][[1, 3]], np.log2(1.1)*e.bandwidth*e.time_fast)
    log = SinrLogger(tmp_path, e.config, console_episodes=0)
    log.record(e, 1001, "scored", 1, 1); log.close()
    rows = list(csv.DictReader((tmp_path / "sinr_slots.csv").open(encoding="utf-8")))
    assert float(rows[0]["v2i_sinr_db"]) == pytest.approx(0.)  # Active0dB is valid, not missing.
    assert rows[0]["v2v_sinr_db_follower1"] == ""
    assert rows[1]["v2i_sinr_db"] == ""
    assert float(rows[1]["v2v_bottleneck_sinr_db"]) == pytest.approx(-10.)


def synthetic_policy(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    run = tmp_path / "synthetic_sinr"; run.mkdir()
    monkeypatch.setattr(cp, "_git_stamp", lambda: {"reproduction_git_commit": cli.TRAINING_COMMIT})
    best = {"rule": ts.RULE, "selected_episode": 2109, "window_start": 2010, "window_end": 2109,
            "score": .027340542243357566, "policy_version": 421, "save_timing": "synthetic_only"}
    selection = {"rule": ts.RULE, "best": best, "final_episode": 2500, "final_policy_version": 500}
    dump(run / "config.resolved.json", c.to_dict())
    dump(run / "COMPLETE.json", {"status": "complete", "algorithm": "mappo",
         "config_hash": c.canonical_hash(), "reproduction_git_commit": cli.TRAINING_COMMIT,
         "training_reward_selection": selection})
    payload = cp.build_policy_payload(c, FakeTrainer(c), 2109)
    payload.update(training_completed=True, checkpoint_role="best_training100", selected_episode=2109,
                   policy_version=421, training_reward_selection=best)
    policy = run / ts.BEST
    torch.save(payload, policy)
    return policy


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_logged_evaluator_exactly_matches_unlogged_arrays_policy_and_rng(tmp_path, monkeypatch, capsys, device):
    if device == "cuda:0" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    policy = synthetic_policy(tmp_path, monkeypatch)
    identity = cli.dc.file_identity(policy)
    rng = cp.capture_rng_state()
    old = zs.evaluate_zero_shot(policy, tmp_path / "plain", cli.dc.SCENARIO, device, [1001], 2,
                               deployment=True, training_policy=True, world_role=cli.selected.WORLD_ROLE)
    new = zs.evaluate_zero_shot(policy, tmp_path / "logged", cli.dc.SCENARIO, device, [1001], 2,
                               deployment=True, training_policy=True, world_role=cli.selected.WORLD_ROLE,
                               sinr_log=True, sinr_console_episodes=1,
                               reference_metrics=Path(old["eval_dir"]) / "metrics.npz")
    equal(rng, cp.capture_rng_state())
    assert cli.dc.file_identity(policy) == identity
    assert new["reference_common_arrays_exact"] is True
    assert new["policy_parameters_unchanged"] and not new["optimizer_constructed"] and not new["critic_constructed"]
    assert new["exogenous_probes"] == old["exogenous_probes"]
    counts = new["sinr_summary"]["row_counts"]
    assert counts == {"warmup": 50, "scored": 20}
    stats = new["sinr_summary"]["scored_link_statistics"]
    assert stats["v2i_active"]["count"]+stats["v2v_bottleneck_active"]["count"] == 20
    assert stats["v2v_receiver_active"]["count"] == 3*stats["v2v_bottleneck_active"]["count"]
    root = Path(new["eval_dir"])
    rows = list(csv.DictReader((root / "sinr_slots.csv").open(encoding="utf-8")))
    assert len(rows) == 70
    for label, col, mode in (("v2i_active", "v2i_sinr_linear", "V2I"),
                             ("v2v_bottleneck_active", "v2v_bottleneck_sinr_linear", "V2V")):
        x = [float(r[col]) for r in rows if r["phase"] == "scored" and r["mode"] == mode]
        assert stats[label]["mean_linear"] == pytest.approx(np.mean(x))
        assert stats[label]["mean_db"] == pytest.approx(np.mean(10*np.log10(x)))
    assert sum(line.startswith("world=") for line in (root / "sinr.log").read_text().splitlines()) == 70
    printed = capsys.readouterr().out
    assert printed.count("world=1001") == 10 and "phase=warmup" not in printed
    with np.load(root / "metrics.npz") as a, np.load(Path(old["eval_dir"]) / "metrics.npz") as b:
        equal(dict(a), dict(b))


def test_empty_link_coverage_null_and_bad_console_count(tmp_path):
    c = cfg(tmp_path)
    e = SinrEnvironment(c); e.reset_world(1001)
    log = SinrLogger(tmp_path, c, 0)
    try:
        for slot in range(1, 3):
            e.step(np.full((5, 3), -1.))
            log.record(e, 1001, "scored", 1, slot)
        result = log.finish([1001], 1, 0)
        assert result["scored_link_statistics"]["v2v_receiver_active"]["mean_db"] is None
        with pytest.raises(ValueError, match="coverage"): log.finish([1001], 2, 0)
    finally: log.close()
    with pytest.raises(ValueError, match="nonnegative"): SinrLogger(tmp_path, c, -1)


def setup_cli(tmp_path, monkeypatch):
    batch = tmp_path / "study" / cli.selected.tc.ROOT / "lab_synthetic"
    dump(batch / "protocol/manifest.json", {"batch_root": str(batch), "study_root": str(tmp_path / "study")})
    run = batch / "training/a01/runs/train100_shared_tdec_seed13"
    dump(run / "COMPLETE.json", {"config_hash": "synthetic"})
    for name in ("config.resolved.json", "training_reward_selection.csv", "policy_best_training.pt"):
        (run / name).write_text("synthetic")
    r = {"run_id": 5, "run_dir": str(run), "training_seed": 13, "best_episode": 2109,
         "best_policy_version": 421, "implementation_commit": cli.TRAINING_COMMIT,
         "best": cli.dc.file_identity(run / "policy_best_training.pt")}
    dump(batch / "records/train_05.json", r)
    monkeypatch.setattr(cli, "code_check", lambda expected: "test_commit")
    monkeypatch.setattr(cli.selected, "candidates", lambda b: [])
    monkeypatch.setattr(cli.selected, "select", lambda records: [{"source_id": "shared_tdec__best100", "record": copy.deepcopy(r)}])
    monkeypatch.setattr(cli.selected, "source_check", lambda *a, **kw: None)
    return batch, r


def test_cli_dry_run_no_output_no_payload_and_strict_original_choice(tmp_path, monkeypatch):
    batch, r = setup_cli(tmp_path, monkeypatch)
    result = cli.run(batch, "test", "test_commit", dry_run=True)
    assert result["status"] == "DRY_RUN" and not result["payload_loaded"]
    assert not Path(result["output_root"]).exists()
    assert result["protocol"] == {"warmup": 5, "scored": 100, "slots": 100, "action_repeat": 0,
                                  "execution": "stochastic", "device": "cuda:0"}
    r["training_seed"] = 8
    with pytest.raises(ValueError, match="previously accepted"): cli.choose(batch)


@pytest.mark.parametrize("worlds", [[], [1001, 1001], [1031], [213]])
def test_cli_forbids_new_or_duplicate_worlds(tmp_path, monkeypatch, worlds):
    batch, _ = setup_cli(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="previously used"): cli.run(batch, "test", "test_commit", worlds)


def test_cli_export_allowlist_no_weights_or_arrays_and_no_overwrite(tmp_path, monkeypatch):
    batch, _ = setup_cli(tmp_path, monkeypatch)
    def fake_evaluate(policy, out, *a, **kw):
        assert kw == dict(deployment=True, world_role=cli.selected.WORLD_ROLE, training_policy=True,
                          sinr_log=True, sinr_console_episodes=1)
        n = out / "native"; n.mkdir()
        for name in ("sinr.log", "sinr_slots.csv", "sinr_summary.json", "EVAL_COMPLETE.json", "provenance.json",
                     "metrics.npz", "policy.pt"):
            (n / name).write_text("synthetic")
        return {"eval_dir": str(n), "status": "complete", "policy_parameters_unchanged": True,
                "sinr_summary": {"technical_status": "PASS", "row_counts": {"warmup": 2500, "scored": 50000}},
                "policy_episode": 2109, "training_seed": 13, "checkpoint_role": "best_training100",
                "critic_constructed": False, "optimizer_constructed": False}
    monkeypatch.setattr(zs, "evaluate_zero_shot", fake_evaluate)
    result = cli.run(batch, "test", "test_commit")
    assert result["technical_status"] == "PASS"
    with zipfile.ZipFile(result["archive"]) as z:
        assert set(z.namelist()) == {"manifest.json", "verification.json", "README.txt", "native/sinr.log",
            "native/sinr_slots.csv", "native/sinr_summary.json", "native/EVAL_COMPLETE.json", "native/provenance.json"}
    with pytest.raises(ValueError, match="unique"): cli.run(batch, "test", "test_commit")
