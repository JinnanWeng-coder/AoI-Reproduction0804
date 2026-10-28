"""CAM96k Uniform training-lifecycle rollouts; CPU, no learner or evaluation.

Port of historical run_random_training_baseline.py at 168028fa28a32f9d41a7f7697e27ee4a391b79b5.
Sampler, RNG, lifecycle and reducers retained; isolated six-seed CAM96k outputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import zipfile

import numpy as np

from aoi_v2x_reproduction.config import config_from_dict
from aoi_v2x_reproduction.envs.platoon import PaperEnviron
from analysis.deployment_contract import ENV_FIELDS
from analysis.deployment_rng import exogenous_probe

REPO = Path(__file__).resolve().parents[1]
BASELINE = "a7075a3d1d9cf6cb9deab9be1ae17a8b1ff57fed"
HISTORICAL_UNIFORM = "168028fa28a32f9d41a7f7697e27ee4a391b79b5"
BUNDLE = "configs/training100_fresh_sources.json"
SEEDS = list(range(8, 14))
MODELS = ["shared_tdec", "independent_tdec", "shared_combined", "algorithm2"]
SCHEMA = "cam96k_uniform_random_training_v1"
ROOT = Path("CAM96k-Random-training/P5_N4_gap25")
SMOKE_ROOT = Path("CAM96k-Random-training-smoke/P5_N4_gap25")
RNG_RULE = {"bit_generator": "PCG64", "seed_sequence": "[environment_seed, 0x52414E44, 1]",
            "tag": 0x52414E44, "version": 1, "draw_order_per_slot": ["rb[P]", "mode[P]", "power_fraction[P]"],
            "independent_of_environment_generator": True}
ACTION_RULE = {"rb": "discrete uniform 0..K-1", "mode": "discrete uniform 0..n_modes-1",
               "power_fraction": "Uniform[0,1) float64", "power_dbm": "p_min+(p_max-p_min)*u",
               "encoding": "category bin centers; normalized power=2*u-1; native environment decoder once",
               "power_distribution": "uniform dBm, NOT uniform mW", "neural_policy": False}
SUM_FIELDS = ["aoi_sum_ms", "slot_count", "cam_successes", "cam_trials", "power_mw_sum",
              "power_dbm_sum", "task1_reward_sum", "task2_reward_sum", "global_reward_sum", "reward_sum",
              "aoi_gt50_count", "v2i_attempt_count", "v2i_reset_count", "deadline_remaining_bits_sum"]
MEAN_FIELDS = ["mean_aoi_ms", "cam_completion_percent", "mean_power_mw", "mean_power_dbm",
               "mean_reward", "mean_task1_reward", "mean_task2_reward", "mean_global_reward", "mean_deadline_remaining_bits"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def csv_write(path, rows):
    require(bool(rows), "empty CSV")
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True, encoding="utf-8").strip()


def code_check(expected=None):
    sha = git("rev-parse", "HEAD")
    require(git("branch", "--show-current").startswith("research/HPC") or not git("branch", "--show-current"),
            "use research/HPC or a fixed detached checkout")
    require(not git("status", "--porcelain", "--untracked-files=all"), "clean fixed checkout required")
    require(expected is None or sha == expected, "implementation commit changed")
    require(not git("diff", "--name-only", BASELINE, "HEAD", "--", "aoi_v2x_reproduction", BUNDLE,
                    "analysis/deployment_contract.py", "analysis/deployment_rng.py"),
            "source environment/configuration changed from reviewed baseline")
    return sha


def source_bundle():
    bundle = read(REPO / BUNDLE)
    require(bundle["schema"] == "training100_config_bundle_v1" and set(bundle["templates"]) == set(MODELS),
            "wrong four-configuration source bundle")
    configs = {name: config_from_dict(raw).to_dict() for name, raw in bundle["templates"].items()}
    environment = {k: configs[MODELS[0]][k] for k in ENV_FIELDS}
    require(all({k: c[k] for k in ENV_FIELDS} == environment for c in configs.values()),
            "four source environments differ")
    require(environment["scenario"]["id"] == "p05_n04_g25" and
            environment["scenario"]["number_platoons"] == 5 and environment["scenario"]["platoon_size"] == 4 and
            environment["scenario"]["gap_m"] == 25 and environment["n_rb"] == 3 and environment["n_modes"] == 2 and
            environment["steps_per_episode"] == 100 and environment["slot_ms"] == 1 and
            environment["cam_bits"] == 32000 and environment["power_continuous"] is True and
            (environment["power_min_dbm"], environment["power_max_dbm"]) == (1, 30), "source scenario drift")
    require(len(bundle["sources"]) == 24 and {(r["model_id"], r["training_seed"]) for r in bundle["sources"]} ==
            {(m, s) for m in MODELS for s in range(8, 14)}, "source seed coverage")
    # Reuse historical identities as recorded by the existing Training100 bundle;
    # no old model, NPZ, or historical absolute directory is required.
    for row in bundle["sources"]:
        raw = dict(bundle["templates"][row["model_id"]], seed=row["training_seed"], run_name=row["run_name"])
        require(config_from_dict(raw).canonical_hash() == row["source_config_hash"], "source configuration identity mismatch")
    environment = dict(environment, cam_bits=96000)
    require(all(c["global_actor_weight"] == 1 for c in configs.values()), "reward weight drift")
    return bundle, environment


def derive(bundle, seed, episodes):
    cfg = config_from_dict(bundle["templates"]["shared_tdec"])
    cfg.seed, cfg.episodes, cfg.device, cfg.cam_bits = seed, episodes, "cpu", 96000
    cfg.checkpoint_mode, cfg.run_name = "none", f"uniform_random_seed{seed}"
    # cfg.algorithm remains the provenance template only; no algorithm is constructed.
    return cfg


def init(study_root, batch_id, expected_commit, smoke_episodes=None):
    sha = code_check(expected_commit)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", batch_id), "unsafe batch ID")
    require(smoke_episodes is None or 1 <= smoke_episodes <= 10, "smoke budget must be 1..10 episodes")
    bundle, environment = source_bundle()
    study = Path(study_root).resolve()
    batch = study / (SMOKE_ROOT if smoke_episodes is not None else ROOT) / batch_id
    require(not batch.exists(), "batch exists; do not initialize twice")
    episodes = smoke_episodes if smoke_episodes is not None else 2500
    m = {"schema": SCHEMA, "batch_root": str(batch), "study_root": str(study), "condition": "uniform_random",
         "implementation_commit": sha, "source_baseline_commit": BASELINE, "source_bundle": BUNDLE,
         "historical_uniform_commit": HISTORICAL_UNIFORM, "cam_bits": 96000, "trained_models": 0,
         "resume_supported": False, "policy_checkpoints": 0,
         "seeds": SEEDS, "episodes": episodes, "slots": 100, "platoons": 5, "smoke_only": smoke_episodes is not None,
         "warmup_episodes": 0, "learning_updates": 0, "new_frozen_evaluations": 0,
         "action_rule": ACTION_RULE, "rng_rule": RNG_RULE, "environment_config": environment,
         "pairing": {"same_environment_seed": True, "historical_training_trace_verified": False,
                     "scope": "same code/config/lifecycle permits exogenous pairing; closed-loop states differ"}}
    batch.mkdir(parents=True, exist_ok=False)
    for name, data in (("manifest", m), ("source_bundle", bundle),
                       ("resolved_configs", {str(s): derive(bundle, s, episodes).to_dict() for s in SEEDS})):
        write(batch / "protocol" / f"{name}.json", data)
    return m


def load(batch):
    batch = Path(batch).resolve()
    m = read(batch / "protocol/manifest.json")
    code_check(m["implementation_commit"])
    bundle, environment = source_bundle()
    require(m["schema"] == SCHEMA and m["batch_root"] == str(batch) and m["seeds"] == SEEDS and
            batch.parent == Path(m["study_root"]) / (SMOKE_ROOT if m["smoke_only"] else ROOT) and
            (m["slots"], m["platoons"], m["warmup_episodes"], m["learning_updates"], m["new_frozen_evaluations"]) == (100, 5, 0, 0, 0) and
            m["condition"] == "uniform_random" and m["action_rule"] == ACTION_RULE and m["rng_rule"] == RNG_RULE and
            m["environment_config"] == environment and m["cam_bits"] == 96000 and m["trained_models"] == 0 and
            m["resume_supported"] is False and m["policy_checkpoints"] == 0 and
            m["source_baseline_commit"] == BASELINE and m["source_bundle"] == BUNDLE and
            m["historical_uniform_commit"] == HISTORICAL_UNIFORM, "random baseline protocol changed")
    require((1 <= m["episodes"] <= 10) if m["smoke_only"] else m["episodes"] == 2500, "budget drift")
    require(read(batch / "protocol/source_bundle.json") == bundle and
            read(batch / "protocol/resolved_configs.json") == {str(s): derive(bundle, s, m["episodes"]).to_dict() for s in SEEDS},
            "source snapshot changed")
    return m, bundle


def action_rng(seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), RNG_RULE["tag"], 1])))


def draw_actions(cfg, rng):
    rb = rng.integers(0, cfg.n_rb, size=cfg.number_agents)
    mode = rng.integers(0, cfg.n_modes, size=cfg.number_agents)
    fraction = rng.random(cfg.number_agents)
    normalized = np.column_stack((2 * (rb + .5) / cfg.n_rb - 1,
                                  2 * (mode + .5) / cfg.n_modes - 1, 2 * fraction - 1))
    decoded = np.column_stack((rb, mode, cfg.power_min_dbm + (cfg.power_max_dbm - cfg.power_min_dbm) * fraction))
    return normalized, decoded


def means(row):
    n, events = row["slot_count"], row["cam_trials"]
    require(n > 0 and events > 0, "empty denominator")
    return {"mean_aoi_ms": row["aoi_sum_ms"] / n,
            "cam_completion_percent": 100 * row["cam_successes"] / events,
            "mean_power_mw": row["power_mw_sum"] / n, "mean_power_dbm": row["power_dbm_sum"] / n,
            "mean_reward": row["reward_sum"] / n,
            "mean_deadline_remaining_bits": row["deadline_remaining_bits_sum"] / events,
            **{f"mean_{k}_reward": row[f"{k}_reward_sum"] / n for k in ("task1", "task2", "global")}}


def merge_rows(rows):
    total = {k: sum(r[k] for r in rows) for k in SUM_FIELDS}
    return {**total, **means(total)}


def reduce_episode(seed, episode, infos, task1, task2, global_rewards):
    aoi = np.asarray([r["aoi_ms"] for r in infos], dtype=np.float32).astype(np.float64)
    dbm = np.asarray([r["power_dbm"] for r in infos], dtype=np.float32).astype(np.float64)
    success = np.asarray(infos[-1]["success"], dtype=np.float32)
    mode = np.asarray([r["mode"] for r in infos])
    t1, t2 = [np.asarray(x, dtype=np.float32).astype(np.float64) for x in (task1, task2)]
    g = np.asarray(global_rewards, dtype=np.float32).astype(np.float64)
    require(aoi.ndim == 2 and aoi.shape == dbm.shape == t1.shape == t2.shape == mode.shape and
            g.shape == aoi.shape[:1] and success.shape == aoi.shape[1:], "slot arrays have wrong shapes")
    require(all(np.isfinite(v).all() for v in (aoi, dbm, t1, t2, g, success)) and
            np.isin(success, [0, 1]).all(), "nonfinite data or invalid CAM indicator")
    require(np.isfinite(infos[-1]["remaining_demand"]).all() and
            (np.asarray(infos[-1]["remaining_demand"]) >= 0).all(), "invalid remaining bits")
    require(np.array_equal(success.astype(bool), np.asarray(infos[-1]["remaining_demand"]) == 0),
            "CAM success/deadline demand disagreement")
    combined = t1 + t2 + g[:, None]
    rows = []
    for agent in range(aoi.shape[1]):
        row = {"condition": "uniform_random", "seed": seed, "episode": episode, "platoon": agent,
               "aoi_sum_ms": float(aoi[:, agent].sum()), "slot_count": len(infos),
               "cam_successes": int(success[agent]), "cam_trials": 1,
               "power_mw_sum": float(np.power(10., dbm[:, agent] / 10).sum()),
               "power_dbm_sum": float(dbm[:, agent].sum()),
               "task1_reward_sum": float(t1[:, agent].sum()), "task2_reward_sum": float(t2[:, agent].sum()),
               "global_reward_sum": float(g.sum()), "reward_sum": float(combined[:, agent].sum()),
               "aoi_gt50_count": int((aoi[:, agent] > 50).sum()),
               "v2i_attempt_count": int((mode[:, agent] == 0).sum()),
               "v2i_reset_count": int(np.isclose(aoi[:, agent], 1, rtol=0, atol=1e-6).sum()),
               "deadline_remaining_bits_sum": float(np.asarray(infos[-1]["remaining_demand"], dtype=np.float32)[agent])}
        rows.append({**row, **means(row)})
    return rows, {"condition": "uniform_random", "seed": seed, "episode": episode, **merge_rows(rows)}


def preflight(batch):
    m, bundle = load(batch)
    require(sys.version_info[:2] == (3, 10) and np.__version__ == "1.23.5", "use the established Python3.10 / NumPy1.23.5 environment")
    cfg = derive(bundle, 8, m["episodes"])
    env = PaperEnviron(cfg)  # Decoder only; does not initialize or roll out a world.
    actions, decoded = draw_actions(cfg, action_rng(8))
    require(np.allclose(env.decode_actions(actions), decoded, rtol=0, atol=1e-12), "action adapter mismatch")
    result = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"],
              "python_executable": sys.executable, "python": platform.python_version(), "numpy": np.__version__,
              "device": "cpu", "gpu_required": False, "actor_critic_optimizer_created": False,
              "source_models_checked": MODELS, "environment_rollouts": 0, "cam_bits": cfg.cam_bits,
              "rollout_seeds": SEEDS, "neural_models_loaded": 0}
    path = Path(batch) / "protocol/preflight.json"
    if path.exists():
        require(read(path) == result, "preflight environment changed")
    else:
        write(path, result)
    return result


@contextmanager
def batch_lock(batch, seed=None):
    path = Path(batch) / ("SUMMARY_RUNNING.lock" if seed is None else f"seed{seed}_RUNNING.lock")
    with path.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "python": sys.executable, "started": datetime.now(timezone.utc).isoformat()}, f)
    try:
        yield
    finally:
        path.unlink()  # Only this process's successfully acquired lock.


def numeric(row):
    return {k: (v if k == "condition" else float(v)) for k, v in row.items()}


def validate_tables(episode_rows, platoon_rows, seed, episodes):
    require(len(episode_rows) == episodes and len(platoon_rows) == episodes * 5, "missing/duplicate episode rows")
    episodes_numeric = [numeric(r) for r in episode_rows]
    platoons_numeric = [numeric(r) for r in platoon_rows]
    require([(r["seed"], r["episode"]) for r in episodes_numeric] == [(seed, i) for i in range(1, episodes + 1)],
            "episode key order/coverage mismatch")
    require([(r["seed"], r["episode"], r["platoon"]) for r in platoons_numeric] ==
            [(seed, i, a) for i in range(1, episodes + 1) for a in range(5)], "platoon key order/coverage mismatch")
    pmin, pmax = 10 ** .1, 1000.
    for r in platoons_numeric:
        require(r["condition"] == "uniform_random" and all(math.isfinite(v) for k, v in r.items() if k != "condition"),
                "nonfinite/wrong-condition output")
        require(r["slot_count"] == 100 and r["cam_trials"] == 1 and r["cam_successes"] in (0, 1) and
                100 <= r["aoi_sum_ms"] <= 10000 and pmin - 1e-8 <= r["mean_power_mw"] <= pmax + 1e-8 and
                1 <= r["mean_power_dbm"] <= 30, "invalid AoI/CAM/power denominator or range")
        require(all(0 <= r[k] <= 100 and r[k] == int(r[k]) for k in
                    ("aoi_gt50_count", "v2i_attempt_count", "v2i_reset_count")), "invalid count")
        require(r["v2i_reset_count"] <= r["v2i_attempt_count"], "V2I resets exceed attempts")
        require(0 <= r["deadline_remaining_bits_sum"] <= 96000 and
                (r["deadline_remaining_bits_sum"] == 0) == (r["cam_successes"] == 1), "invalid deadline remaining bits")
        for k, v in means(r).items():
            require(math.isclose(r[k], v, rel_tol=1e-12, abs_tol=1e-10), f"derived mean mismatch: {k}")
        require(math.isclose(r["reward_sum"], sum(r[f"{k}_reward_sum"] for k in ("task1", "task2", "global")),
                             rel_tol=1e-12, abs_tol=1e-10), "reward components do not sum")
    for i, r in enumerate(episodes_numeric):
        expected = merge_rows(platoons_numeric[5 * i:5 * i + 5])
        require(r["condition"] == "uniform_random", "episode condition mismatch")
        for k, v in expected.items():
            require(math.isclose(r[k], v, rel_tol=1e-12, abs_tol=1e-10), f"episode/platoon mismatch: {k}")
    return episodes_numeric, platoons_numeric


def verify_run(batch, m, seed, run):
    run = Path(run).resolve()
    require(run.parent == Path(batch).resolve() / "runs" / f"seed{seed}" and re.fullmatch(r"a\d{2,4}", run.name),
            "run outside this seed/batch")
    c = read(run / "COMPLETE.json")
    expected = {"technical_status": "PASS", "seed": seed, "episodes": m["episodes"], "slots": 100,
                "platoons": 5, "condition": "uniform_random", "implementation_commit": m["implementation_commit"],
                "learning_updates": 0, "cam_bits": 96000, "trained_models": 0, "policy_checkpoints": 0,
                "environment_slots": m["episodes"] * 100,
                "agent_slots": m["episodes"] * 500, "cam_deadlines": m["episodes"] * 5}
    require(all(c.get(k) == v for k, v in expected.items()), "completion identity/count mismatch")
    require(read(run / "config.resolved.json") == read(Path(batch) / "protocol/resolved_configs.json")[str(seed)], "run config drift")
    require(c["action_rng_rule"] == RNG_RULE and c["action_rng_initial"] == action_rng(seed).bit_generator.state,
            "action RNG provenance mismatch")
    require(c["event_source"] == "post_step_aoi_ms_isclose_1_verified_env_step" and
            c["python"] == platform.python_version() and c["numpy"] == np.__version__,
            "event source/runtime provenance mismatch")
    hist = c["action_checks"]
    require(hist["slots_checked"] == expected["environment_slots"] and hist["decode_mismatches"] == 0 and
            sum(hist["rb_counts"]) == sum(hist["mode_counts"]) == expected["agent_slots"] and
            len(hist["rb_counts"]) == 3 and len(hist["mode_counts"]) == 2 and
            1 <= hist["power_dbm_min"] <= hist["power_dbm_max"] <= 30, "action verification mismatch")
    rows, platoons = validate_tables(csv_read(run / "episode_metrics.csv"), csv_read(run / "episode_platoon_metrics.csv"), seed, m["episodes"])
    require(sum(r["v2i_attempt_count"] for r in rows) == hist["mode_counts"][0], "action/attempt counts disagree")
    started = read(run / "STARTED.json")
    require(started["seed"] == seed and started["implementation_commit"] == m["implementation_commit"] and
            started["attempt"] == run.name, "started/completed source mismatch")
    probes = read(run / "exogenous_probes.json")
    require("cold_start" in probes and f"after_episode_{m['episodes']}" in probes, "missing exogenous probes")
    names = ("STARTED.json", "COMPLETE.json", "config.resolved.json", "exogenous_probes.json",
             "episode_metrics.csv", "episode_platoon_metrics.csv")
    return {**expected, "run_dir": str(run), "smoke_only": m["smoke_only"],
            "completion_file": str(run / "COMPLETE.json"),
            "files": [{"name": name, "size": (run/name).stat().st_size, "mtime_ns": (run/name).stat().st_mtime_ns}
                      for name in names]}, rows, platoons


def run_seed(batch, m, bundle, seed, attempt, reason):
    require(seed in SEEDS and re.fullmatch(r"a\d{2,4}", attempt), "invalid seed/attempt")
    require(attempt == "a01" or reason, "technical retry requires a reason")
    batch = Path(batch)
    accepted = batch / "records" / f"seed{seed}.json"
    if accepted.exists():
        record = read(accepted)
        checked, _, _ = verify_run(batch, m, seed, record["run_dir"])
        require(record == checked, "accepted record changed")
        return {**checked, "skipped_complete": True}
    found = list((batch / "runs" / f"seed{seed}").glob("a*/COMPLETE.json"))
    require(len(found) <= 1, "multiple completed attempts; do not select by performance")
    if found:
        result, _, _ = verify_run(batch, m, seed, found[0].parent)
        write(accepted, result)
        return {**result, "accepted_existing_native_completion": True}
    run = batch / "runs" / f"seed{seed}" / attempt
    require(not run.exists(), "partial attempt preserved; inspect failure, then use an explicit new attempt")
    cfg = derive(bundle, seed, m["episodes"])
    run.mkdir(parents=True, exist_ok=False)
    write(run / "config.resolved.json", cfg.to_dict())
    write(run / "STARTED.json", {"seed": seed, "attempt": attempt, "reason": reason, "pid": os.getpid(),
                                "implementation_commit": m["implementation_commit"], "started": datetime.now(timezone.utc).isoformat()})
    env, rng = PaperEnviron(cfg), action_rng(seed)
    initial_rng = rng.bit_generator.state
    env.reset_world(seed)  # Exactly once, matching the training runner.
    checks = {"slots_checked": 0, "decode_mismatches": 0, "rb_counts": [0] * cfg.n_rb,
              "mode_counts": [0] * cfg.n_modes, "power_dbm_min": 30., "power_dbm_max": 1.}
    probes = {"cold_start": exogenous_probe(env)}
    started = time.monotonic()
    with (run / "episode_metrics.csv").open("x", newline="", encoding="utf-8") as ef, \
            (run / "episode_platoon_metrics.csv").open("x", newline="", encoding="utf-8") as pf, \
            (run / "run.log").open("x", encoding="utf-8") as log:
        ew = pw = None
        for episode in range(cfg.episodes):
            env.start_episode(episode)
            infos, t1s, t2s, gs = [], [], [], []
            for slot in range(cfg.steps_per_episode):
                actions, expected = draw_actions(cfg, rng)
                require(np.allclose(env.decode_actions(actions), expected, rtol=0, atol=1e-12), "native decoder changed sampled actions")
                _, g, t1, t2, done, info = env.step(actions)
                require(bool(done) == (slot == 99), "training episode termination changed")
                require(np.array_equal(info["rb"], expected[:, 0]) and np.array_equal(info["mode"], expected[:, 1]) and
                        np.array_equal(info["power_dbm"], expected[:, 2].astype(np.float32)), "executed action differs from draw")
                checks["slots_checked"] += 1
                for key, col, n in (("rb_counts", 0, cfg.n_rb), ("mode_counts", 1, cfg.n_modes)):
                    checks[key] = (np.asarray(checks[key]) + np.bincount(expected[:, col].astype(int), minlength=n)).tolist()
                checks["power_dbm_min"] = min(checks["power_dbm_min"], float(expected[:, 2].min()))
                checks["power_dbm_max"] = max(checks["power_dbm_max"], float(expected[:, 2].max()))
                infos.append({k: np.asarray(info[k]).copy() for k in ("aoi_ms", "power_dbm", "mode", "success", "remaining_demand")})
                t1s.append(t1); t2s.append(t2); gs.append(g)
            platoons, row = reduce_episode(seed, episode + 1, infos, t1s, t2s, gs)
            if ew is None:
                ew, pw = csv.DictWriter(ef, list(row)), csv.DictWriter(pf, list(platoons[0]))
                ew.writeheader(); pw.writeheader()
            ew.writerow(row); pw.writerows(platoons)
            ef.flush(); pf.flush()
            if episode + 1 in (1, 100, cfg.episodes) or (episode + 1) % 500 == 0:
                probes[f"after_episode_{episode + 1}"] = exogenous_probe(env)
            if (episode + 1) % 100 == 0 or episode == 0 or episode + 1 == cfg.episodes:
                message = f"seed={seed} episode={episode+1}/{cfg.episodes} elapsed_s={time.monotonic()-started:.1f}"
                print(message, flush=True); log.write(message + "\n"); log.flush()
    code_check(m["implementation_commit"])
    write(run / "exogenous_probes.json", probes)
    # Validate CSV before publishing completion. No performance gate.
    validate_tables(csv_read(run / "episode_metrics.csv"), csv_read(run / "episode_platoon_metrics.csv"), seed, cfg.episodes)
    write(run / "COMPLETE.json", {"technical_status": "PASS", "condition": "uniform_random", "seed": seed,
          "episodes": cfg.episodes, "slots": 100, "platoons": 5, "implementation_commit": m["implementation_commit"],
          "environment_slots": cfg.episodes * 100, "agent_slots": cfg.episodes * 500, "cam_deadlines": cfg.episodes * 5,
          "learning_updates": 0, "cam_bits": 96000, "trained_models": 0, "policy_checkpoints": 0,
          "event_source": "post_step_aoi_ms_isclose_1_verified_env_step",
          "action_rng_rule": RNG_RULE, "action_rng_initial": initial_rng,
          "action_rng_final": rng.bit_generator.state, "action_checks": checks, "elapsed_seconds": time.monotonic() - started,
          "python": platform.python_version(), "numpy": np.__version__, "completed": datetime.now(timezone.utc).isoformat()})
    result, _, _ = verify_run(batch, m, seed, run)
    write(accepted, result)
    return result


def summary_tables(rows, episodes):
    windows = {"all": (1, episodes), "first100": (1, min(100, episodes)),
               "401_500": (401, 500), "last100": (max(1, episodes - 99), episodes)}
    summaries = []
    for seed in SEEDS:
        for label, (lo, hi) in windows.items():
            part = [r for r in rows if r["seed"] == seed and lo <= r["episode"] <= hi]
            summaries.append({"condition": "uniform_random", "seed": seed, "window": label, "first_episode": lo, "last_episode": hi,
                              "episodes_present": len(part), "status": "AVAILABLE" if part else "NOT_OBSERVED",
                              **(merge_rows(part) if part else {k: None for k in SUM_FIELDS + MEAN_FIELDS})})
    across = []
    for label in windows:
        part = [r for r in summaries if r["window"] == label]
        for metric in MEAN_FIELDS:
            values = [r[metric] for r in part if r[metric] is not None]
            across.append({"condition": "uniform_random", "window": label, "metric": metric, "n_seeds": len(values),
                           "mean": float(np.mean(values)) if values else None,
                           "sample_sd": float(np.std(values, ddof=1)) if len(values) >= 2 else None})
    curves = []
    for episode in range(1, episodes + 1):
        part = [r for r in rows if r["episode"] == episode]
        require(len(part) == len(SEEDS) and {r["seed"] for r in part} == set(SEEDS), "six-seed curve coverage")
        curves.append({"condition": "uniform_random", "episode": episode, "n_seeds": len(SEEDS),
                       **{f"{k}_{stat}": float(np.mean([r[k] for r in part]) if stat == "mean" else
                                               np.std([r[k] for r in part], ddof=1))
                          for k in MEAN_FIELDS for stat in ("mean", "sample_sd")}})
    return summaries, across, curves


def summarize(batch, attempt="a01"):
    m, _ = load(batch)
    require(re.fullmatch(r"a\d{2,4}", attempt), "invalid analysis attempt")
    rows, platoons, runs = [], [], []
    for seed in SEEDS:
        rec = read(Path(batch) / "records" / f"seed{seed}.json")
        checked, erows, prows = verify_run(batch, m, seed, rec["run_dir"])
        require(checked == rec, "accepted result changed")
        rows.extend(erows); platoons.extend(prows); runs.append(Path(rec["run_dir"]))
    n = m["episodes"]
    verification = {"technical_status": "PASS", "claim_support": "NOT_ASSESSED", "smoke_only": m["smoke_only"],
                    "implementation_commit": m["implementation_commit"], "seeds": SEEDS, "accepted_runs": len(SEEDS),
                    "cam_bits": 96000, "trained_models": 0, "policy_checkpoints": 0, "resume_supported": False,
                    "episode_rows": len(rows), "platoon_rows": len(platoons), "environment_slots": len(SEEDS) * n * 100,
                    "agent_slots": int(sum(r["slot_count"] for r in platoons)),
                    "cam_deadlines": int(sum(r["cam_trials"] for r in platoons)), "learning_updates": 0,
                    "actor_critic_optimizer_created": False, "new_frozen_evaluations": 0}
    require((len(rows), len(platoons), verification["agent_slots"], verification["cam_deadlines"]) ==
            (len(SEEDS)*n, len(SEEDS)*5*n, len(SEEDS)*500*n, len(SEEDS)*5*n), "batch denominator mismatch")
    out = Path(batch) / "analysis" / attempt
    sums, across, curves = summary_tables(rows, n)
    tables = {"episode_metrics": rows, "episode_platoon_metrics": platoons, "summary": sums,
              "summary_across_seeds": across, "episode_across_seeds": curves}
    if out.exists():
        require(read(out / "verification.json") == verification, "existing verification changed/incomplete")
        for name, values in tables.items():
            require(csv_read(out / f"{name}.csv") == [{k: "" if v is None else str(v) for k, v in r.items()} for r in values],
                    f"existing summary mismatch: {name}")
        archive = out / "core_evidence.zip"
        verify_archive(batch, out, runs, archive, n, attempt)
        return {**verification, "archive": str(archive), "seed_results": sums, "skipped_complete": True}
    out.mkdir(parents=True)
    for name, values in tables.items():
        csv_write(out / f"{name}.csv", values)
    write(out / "verification.json", verification)
    (out / "REPORT_CN.md").write_text(
        "# 均匀随机动作训练流程基线\n\n"
        "CAM96000 bit，六个独立rollout seed8–13，不学习、不筛选、不冻结评估。功率在dBm上均匀；不等于mW均匀或初始PPO策略。\n"
        "每seed只冷启动一次，随后连续start_episode；无warmup剔除。AoI为原环境逐槽post-step毫秒值；"
        "CAM只取每episode最终slot的deadline成功标志，不平均全slot的success。\n"
        "reward使用float32日志的task1+task2+global转float64归约，与Training100选择曲线口径一致。"
        "power_mw先逐槽由日志dBm转换，再求和/平均；global reward在各车队重复计入，与训练combined reward一致。\n"
        "v2i_attempt为mode0；reset由post-step AoI≈1恢复，只描述V2I更新。\n"
        "summary按各窗口sum/count；六seed均值及样本SD仅作描述，不给显著性结论。末100不是best100权重。"
        "smoke未观察的401–500窗口为空值，不补0。\n"
        "动作与环境各用独立PCG64。同代码、配置、生命周期下外生RNG可配对，已用小型测试检查；"
        "未与历史正式训练原始轨迹对账，不能声称实际逐轨迹完全一致，闭环AoI/观测本来可以不同。\n",
        encoding="utf-8")
    archive = out / "core_evidence.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for folder in (Path(batch) / "protocol", Path(batch) / "records", out):
            for p in sorted(folder.glob("*")):
                if p.is_file() and p.suffix in (".json", ".csv", ".md"):
                    z.write(p, p.relative_to(batch).as_posix())
        for seed, run in zip(SEEDS, runs):
            for name in ("STARTED.json", "COMPLETE.json", "config.resolved.json", "exogenous_probes.json"):
                z.write(run / name, f"run_metadata/seed{seed}/{name}")
    verify_archive(batch, out, runs, archive, n, attempt)
    return {**verification, "archive": str(archive), "seed_results": sums}


def verify_archive(batch, out, runs, archive, n, attempt):
    expected = {}
    for folder in (Path(batch) / "protocol", Path(batch) / "records", out):
        for p in folder.glob("*"):
            if p.is_file() and p.suffix in (".json", ".csv", ".md"):
                expected[p.relative_to(batch).as_posix()] = p
    for seed, run in zip(SEEDS, runs):
        for name in ("STARTED.json", "COMPLETE.json", "config.resolved.json", "exogenous_probes.json"):
            expected[f"run_metadata/seed{seed}/{name}"] = run / name
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None and len(z.namelist()) == len(set(z.namelist())), "ZIP integrity/duplicates")
        require(set(z.namelist()) == set(expected), "ZIP unexpected/missing members")
        for name, path in expected.items():
            require(z.read(name) == path.read_bytes(), f"ZIP/source mismatch: {name}")
        import io
        for name, count in (("episode_metrics", len(SEEDS)*n), ("episode_platoon_metrics", len(SEEDS)*5*n)):
            require(len(list(csv.DictReader(io.StringIO(z.read(f"analysis/{attempt}/{name}.csv").decode())))) == count,
                    "ZIP row count mismatch")


def run(batch, seeds=None, attempt="a01", reason=None):
    m, bundle = load(batch)
    preflight(batch)
    requested = SEEDS if seeds is None else seeds
    require(requested and len(requested) == len(set(requested)) and set(requested) <= set(SEEDS), "invalid seeds")
    result = []
    for seed in requested:
        with batch_lock(batch, seed):
            try:
                result.append(run_seed(batch, m, bundle, seed, attempt, reason))
            except Exception as exc:
                failure = Path(batch) / "records" / f"failure_seed{seed}_{attempt}_{time.time_ns()}.json"
                write(failure, {"seed": seed, "attempt": attempt, "error": repr(exc),
                                "timestamp": datetime.now(timezone.utc).isoformat(), "performance_retry": False})
                raise
    return {"accepted_requested": result, "all_seeds_complete":
            all((Path(batch) / "records" / f"seed{s}.json").exists() for s in SEEDS)}


def submit(batch, partition="Q10", account=None, actually=False, hours=4):
    """One six-unit CPU array and afterok summary; never touches the old chain."""
    m, _ = load(batch)
    preflight(batch)
    require(not m["smoke_only"], "HPC submission is formal only; no independent pilot")
    batch = Path(batch).resolve()
    script = REPO / "hpc/aoi_cam96k_random_training.sbatch"
    common = ["sbatch", "--parsable", "--export=ALL", f"--partition={partition}",
              "--cpus-per-task=2", "--mem=4G", f"--time={hours}:00:00",
              f"--output={batch}/logs/%x_%A_%a.out", f"--error={batch}/logs/%x_%A_%a.err"]
    if account:
        common.append(f"--account={account}")
    commands = {"rollouts": [*common, "--job-name=cam96k_random_train", "--array=0-5%2", str(script)],
                "summary": [*common, "--job-name=cam96k_random_sum", "--dependency=afterok:{rollouts}", str(script)]}
    intent = batch / "records/submission_intent.json"
    receipt = batch / "records/submission.json"
    require(not intent.exists() and not receipt.exists(), "submission already exists/uncertain; inspect receipts and Slurm, never repeat chain")
    result = {"commands": commands, "initial_concurrency": 2, "max_concurrency": 6,
              "implementation_commit": m["implementation_commit"], "cpu_only": True, "no_hpc_pilot": True}
    if not actually:
        return result
    (batch / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, PROJECT_DIR=str(REPO), PYTHON_BIN=sys.executable,
               AOI_RANDOM_BATCH=str(batch), AOI_EXPECTED_COMMIT=m["implementation_commit"],
               AOI_RANDOM_ATTEMPT="a01")
    for key in ("AOI_TASK_OFFSET", "AOI_RETRY_REASON", "SBATCH_GRES", "SBATCH_GPUS",
                "SBATCH_GPUS_PER_NODE", "SBATCH_GPUS_PER_TASK", "SBATCH_GPUS_PER_SOCKET"):
        env.pop(key, None)
    with batch_lock(batch, "submission"):
        require(not intent.exists() and not receipt.exists(), "concurrent submission")
        write(intent, result)
        jobs = {}
        for stage, args in commands.items():
            argv = [a.format(**jobs) for a in args]
            env["AOI_RANDOM_STAGE"] = stage
            raw = subprocess.check_output(argv, text=True, env=env).strip()
            job = raw.split(";")[0]
            require(job.isdigit(), "uncertain sbatch response; inspect Slurm before any retry")
            jobs[stage] = job
            write(batch / f"records/submitted_{stage}.json", {"job_id": job, "command": argv})
        result.update(jobs=jobs)
        write(receipt, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("init")
    q.add_argument("--study-root", type=Path, required=True)
    q.add_argument("--batch-id", required=True)
    q.add_argument("--expected-commit", required=True)
    q.add_argument("--smoke-episodes", type=int)
    for command in ("preflight", "run-unit", "summary", "submit"):
        q = sub.add_parser(command)
        q.add_argument("--batch-root", type=Path, required=True)
        if command in ("run-unit", "summary"):
            q.add_argument("--attempt", default="a01")
        if command == "run-unit":
            q.add_argument("--seed", type=int, choices=SEEDS, required=True)
            q.add_argument("--reason")
        if command == "submit":
            q.add_argument("--partition", default="Q10")
            q.add_argument("--account")
            q.add_argument("--hours", type=int, default=4)
            q.add_argument("--submit", action="store_true")
    a = p.parse_args(argv)
    if a.command == "init":
        result = init(a.study_root, a.batch_id, a.expected_commit, a.smoke_episodes)
    elif a.command == "preflight":
        result = preflight(a.batch_root)
    elif a.command == "run-unit":
        result = run(a.batch_root, [a.seed], a.attempt, a.reason)
    elif a.command == "submit":
        require(a.hours > 0, "positive walltime required")
        result = submit(a.batch_root, a.partition, a.account, a.submit, a.hours)
    else:
        with batch_lock(a.batch_root):
            result = summarize(a.batch_root, a.attempt)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
