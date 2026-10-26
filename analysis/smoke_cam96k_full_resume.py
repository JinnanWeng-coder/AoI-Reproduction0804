"""Development-only fresh-process CUDA recovery smoke; never a formal run.

Retains production networks/replay/update parameters/100 slots. Only seed, paths
and the explicit smoke episode budget differ. Run after other training exits.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from analysis import cam96k_seed8_11 as p
from aoi_v2x_reproduction.runtime import runner as r
from aoi_v2x_reproduction.runtime import full_training_checkpoint as fc
from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state


def compare(a, b, key="state"):
    if isinstance(a, torch.Tensor):
        p.require(isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu()), f"tensor mismatch: {key}")
    elif isinstance(a, np.ndarray):
        p.require(np.array_equal(a, b), f"array mismatch: {key}")
    elif isinstance(a, dict):
        p.require(set(a) == set(b), f"keys mismatch: {key}")
        for name in a:
            compare(a[name], b[name], f"{key}.{name}")
    elif isinstance(a, (list, tuple)):
        p.require(len(a) == len(b), f"length mismatch: {key}")
        for i, (left, right) in enumerate(zip(a, b)):
            compare(left, right, f"{key}.{i}")
    else:
        p.require(a == b, f"value mismatch: {key}")


def worker(root, model, stage):
    source = next(s for s in p.sources() if s["model_id"] == model and s["training_seed"] == 8)
    cfg, _ = p.derive(source, root / model / ("continuous" if stage == "continuous" else "restarted"), True)
    cfg.episodes = 3 if "maddpg" in model else 6
    run = Path(cfg.output_root) / cfg.run_name
    factory_name = "_make_system" if "maddpg" in model else "_make_mappo_system"
    factory = getattr(r, factory_name)
    systems = []

    def capture(c):
        system = factory(c)
        systems.append(system)
        return system

    setattr(r, factory_name, capture)
    kwargs = dict(periodic_policy_interval=1, periodic_policy_smoke=True, full_checkpoint_interval=1, full_checkpoint_smoke=True)
    if stage == "interrupt":
        def stop(data):
            if data["episode"] == (2 if "maddpg" in model else 3):
                raise RuntimeError("development simulated interruption")
        try:
            r.train(cfg, progress_callback=stop, **kwargs)
        except RuntimeError as exc:
            if str(exc) == "development simulated interruption":
                return 43
            raise
        raise RuntimeError("interruption did not occur")
    checkpoint = run / "training_checkpoints" / ("training_episode_000001.pt" if "maddpg" in model else "training_episode_000002.pt")
    r.train(cfg, resume_training_checkpoint=checkpoint if stage == "resume" else None, **kwargs)
    torch.save({"state": fc.training_state(cfg, systems[0]), "rng": capture_rng_state()}, run / "development_final_state.pt")
    fc.verify_full_run(run, range(1, cfg.episodes + 1), final_path=run / "policy_final.pt")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", choices=p.MODELS)
    parser.add_argument("--stage", choices=("continuous", "interrupt", "resume"))
    args = parser.parse_args()
    root = args.root.resolve()
    p.require("smoke" in root.name.lower(), "dedicated smoke root required")
    if args.stage:
        p.require(args.model is not None, "worker model required")
        return worker(root, args.model, args.stage)
    p.require(not root.exists(), "new unique CUDA recovery smoke directory required")
    root.mkdir(parents=True)
    results = []
    for model in p.MODELS:
        for stage in ("continuous", "interrupt", "resume"):
            command = [sys.executable, "-B", "-u", "-m", "analysis.smoke_cam96k_full_resume",
                       "--root", str(root), "--model", model, "--stage", stage]
            with (root / f"{model}_{stage}.out").open("x", encoding="utf-8") as out:
                result = subprocess.run(command, cwd=p.REPO, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, stdout=out, stderr=subprocess.STDOUT)
            p.require(result.returncode == (43 if stage == "interrupt" else 0), f"fresh-process {model}/{stage} failed: exit {result.returncode}")
        run_name = f"cam96k_{model}_s8"
        left, right = [root / model / mode / run_name for mode in ("continuous", "restarted")]
        a, b = [torch.load(folder / "development_final_state.pt", map_location="cpu", weights_only=False) for folder in (left, right)]
        compare(a, b)
        with np.load(left / "train_metrics.npz") as x, np.load(right / "train_metrics.npz") as y:
            compare({k: x[k] for k in x.files}, {k: y[k] for k in y.files}, "metrics")
        record = {"model_id": model, "technical_status": "PASS", "fresh_processes": 3, "full_state_and_rng_exact": True,
                  "all_metrics_exact": True, "continuous_run": str(left), "resumed_run": str(right)}
        results.append(record)
        print(json.dumps(record), flush=True)
    p.write(root / "verification.json", {"technical_status": "PASS", "smoke": True, "formal_training_started": False,
        "device": "cuda:0", "implementation_commit": p.dc.git("rev-parse", "HEAD"), "worktree_status": p.dc.git("status", "--porcelain"),
        "methods": 5, "records": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
