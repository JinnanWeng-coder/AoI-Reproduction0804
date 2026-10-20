"""Short real-environment/small-network recovery smoke; never formal training evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from analysis import deployment_contract as dc
from aoi_v2x_reproduction.config import config_from_dict
from aoi_v2x_reproduction.runtime import runner


def assert_equal(a, b, label="root"):
    if isinstance(a, torch.Tensor):
        dc.require(isinstance(b, torch.Tensor) and torch.equal(a, b), f"resume tensor differs: {label}")
    elif isinstance(a, np.ndarray):
        dc.require(isinstance(b, np.ndarray) and np.array_equal(a, b), f"resume array differs: {label}")
    elif isinstance(a, dict):
        dc.require(a.keys() == b.keys(), f"resume keys differ: {label}")
        for key in a: assert_equal(a[key], b[key], f"{label}/{key}")
    elif isinstance(a, (list, tuple)):
        dc.require(len(a) == len(b), f"resume length differs: {label}")
        for i, (x, y) in enumerate(zip(a, b)): assert_equal(x, y, f"{label}/{i}")
    else:
        dc.require(a == b, f"resume value differs: {label}: {a!r} vs {b!r}")


def smoke(batch, device="cuda:0"):
    from analysis import maddpg_noise_decay_contract as c
    m, sources = c.load(batch)
    destination = Path(batch) / "protocol" / f"smoke_{device.replace(':', '_')}.json"
    if destination.exists():
        result = dc.read(destination)
        dc.require(result["technical_status"] == "PASS" and result["implementation_commit"] == m["implementation_commit"],
                   "previous smoke did not pass")
        return result
    root = Path(tempfile.mkdtemp(prefix="noise_smoke_", dir=Path(batch) / "tmp"))
    checks = []
    for shared, source in ((False, sources[0]), (True, sources[2])):
        def cfg(name):
            cfg, _ = c.derive(source, root)
            cfg.run_name = ("shared_" if shared else "independent_") + name
            cfg.seed = 910010  # Dedicated technical world, never a research/test cell.
            cfg.episodes, cfg.steps_per_episode, cfg.batch_size, cfg.replay_capacity = 104, 2, 4, 16
            cfg.actor_hidden, cfg.local_critic_hidden, cfg.global_critic_hidden = [8, 4], [8, 4], [8, 4, 2]
            cfg.device = device
            cfg.exploration_noise_decay_environment_steps = 13  # Cut5 -> resume crosses floor during ep7.
            return cfg
        full, partial, resumed = cfg("full"), cfg("partial"), cfg("resumed")
        runner.train(full, training_reward_best=True, noise_decay_training=True)
        runner.train(partial, training_reward_best=True, noise_decay_training=True, max_episodes=5)
        checkpoint = root / partial.run_name / "checkpoints/episode_000005.pt"
        identity = dc.file_identity(checkpoint)
        runner.train(resumed, training_reward_best=True, noise_decay_training=True, resume=str(checkpoint))
        dc.require(dc.file_identity(checkpoint) == identity, "source checkpoint changed")
        a, b = [torch.load(root / x.run_name / "checkpoints/episode_000104.pt", map_location="cpu", weights_only=False)
                for x in (full, resumed)]
        for key in ("agents", "learner", "replay", "environment", "metrics", "rng", "completed_environment_steps"):
            assert_equal(a[key], b[key], key)
        if shared: assert_equal(a["shared_actor"], b["shared_actor"], "shared_actor")
        for name in ("policy_best_training.pt", "policy_final.pt", "policies/policy_episode_000100.pt"):
            pa, pb = [torch.load(root / x.run_name / name, map_location="cpu", weights_only=False) for x in (full, resumed)]
            for key in ("actors", "episode", "policy_version", "checkpoint_role"):
                assert_equal(pa[key], pb[key], key)
        for name in ("exploration_noise.csv", "training_reward_selection.csv"):
            dc.require((root / full.run_name / name).read_bytes() == (root / resumed.run_name / name).read_bytes(),
                       f"resumed {name} differs")
        with np.load(root / full.run_name / "train_metrics.npz") as x, np.load(root / resumed.run_name / "train_metrics.npz") as y:
            dc.require(x.files == y.files, "resumed metric fields differ")
            for key in x.files: assert_equal(x[key], y[key], key)
        checks.append({"actor_structure": "shared" if shared else "independent", "technical_status": "PASS",
                       "bitwise_equal": True, "full_episodes": 104, "slots_per_episode": 2, "cut_episode": 5,
                       "floor_crossing_step": 13, "formal_runs": 0})
    result = {"technical_status": "PASS", "implementation_commit": m["implementation_commit"], "device": device,
              "temporary_artifacts": str(root), "checks": checks,
              "scope": "real environment, tiny networks, 104x2 only; NOT four formal 2500x100 runs"}
    dc.write(destination, result)
    return result
