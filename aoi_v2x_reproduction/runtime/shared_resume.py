"""Opt-in, episode-boundary Shared MADDPG checkpoints; no validation rollout.

Every restore forks into a NEW run directory. Original attempts and immutable
episode checkpoints are never overwritten. Old policy-only files cannot resume.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

from ..config import config_from_dict
from .checkpointing import atomic_torch_save, build_payload, capture_rng_state, restore_rng_state, _git_stamp
from .training_selection import require

SCHEMA = "shared_training100_episode_resume_v1"


def runtime_identity(config):
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": config.device,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def validate_request(config, training_reward_best, interval, max_episodes=None, recover=False):
    require(config.algorithm == "modified_maddpg_tdec" and config.maddpg_actor_sharing is True and
            config.global_update_mode == "synchronous_joint" and config.checkpoint_mode == "policy_only",
            "episode resume requires shared synchronous TDec and policy_only frozen artifacts")
    require(training_reward_best and config.episodes >= 100 and config.global_actor_weight == 1 and
            not config.is_formal_result and not recover, "shared resume requires research best100 training")
    require(type(interval) is int and interval == 50, "this protocol saves every 50 episodes")
    require(max_episodes is None or type(max_episodes) is int and 0 < max_episodes <= config.episodes,
            "invalid partial episode limit")


def validate_resume_config(payload, config):
    require(payload.get("resume_schema") == SCHEMA and payload.get("sharing_schema") == "shared_maddpg_resume_v1",
            "not a resumable shared training100 checkpoint")
    old = config_from_dict(payload["config"])
    require(old.canonical_hash() == payload.get("config_hash"), "source config hash mismatch")
    a, b = old.to_dict(), config.to_dict()
    for key in ("output_root", "run_name"):
        a.pop(key, None)
        b.pop(key, None)
    require(a == b, "resume changes training configuration (only output_root/run_name may change)")


def read_resume(path, config, interval):
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_config(payload, config)
    require(payload.get("checkpoint_interval") == interval and not payload.get("completed") and
            0 < payload["episode"] < config.episodes, "checkpoint is final or has wrong interval/episode")
    require(payload.get("environment") is not None and isinstance(payload.get("training100_selector"), dict),
            "checkpoint missing environment or best100 selector")
    require(payload.get("runtime_identity") == runtime_identity(config), "resume runtime/device changed")
    require(payload.get("reproduction_git_commit") == _git_stamp()["reproduction_git_commit"],
            "resume implementation commit changed")
    require(path.name == f"episode_{payload['episode']:06d}.pt", "checkpoint episode filename mismatch")
    return payload


def prepare_run(config, resume, interval):
    from .runner import _prepare_run, _write_json
    payload = read_resume(resume, config, interval) if resume else None
    # The existing no-overwrite initializer is used even for resuming.
    run, _ = _prepare_run(config, None)
    provenance_path = run / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update(episode_resume_schema=SCHEMA, checkpoint_interval=interval,
                      resume_source=str(Path(resume).resolve()) if resume else None,
                      resume_episode=payload["episode"] if payload else 0,
                      resume_config_changes=["output_root", "run_name"] if payload else [])
    _write_json(provenance_path, provenance)
    return run, payload is not None


def save_checkpoint(run, config, agents, learner, replay, environment, metrics, selector, episode, interval):
    from .runner import _write_json
    path = Path(run) / "checkpoints" / f"episode_{episode:06d}.pt"
    require(not path.exists(), "refusing to overwrite an episode checkpoint")
    require(hasattr(environment, "state_dict") and hasattr(environment, "load_state_dict"),
            "environment is not checkpointable")
    rng = capture_rng_state()
    try:
        payload = build_payload(config, agents, learner, replay, environment, metrics, episode,
                                completed=episode == config.episodes)
        require(payload["environment"] is not None and len(metrics.episodes) == selector.episode == episode,
                "checkpoint environment/metrics/selector inconsistent")
        payload.update(resume_schema=SCHEMA, checkpoint_role="shared_training100_episode_boundary",
                       checkpoint_interval=interval, runtime_identity=runtime_identity(config),
                       training100_selector=selector.state_dict())
        atomic_torch_save(payload, path)
        # A small pointer, not another full checkpoint. No old numbered file is removed.
        _write_json(Path(run) / "checkpoints" / "latest.json",
                    {"schema": SCHEMA, "episode": episode, "checkpoint": str(path.resolve()),
                     "bytes": path.stat().st_size, "can_resume": episode < config.episodes})
    finally:
        restore_rng_state(rng)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="accepted parent Algorithm2 or explicit shared resolved config")
    group.add_argument("--resume", type=Path, help="full episode_NNNNNN.pt, never policy_best_training.pt")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.resume:
        raw = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
    else:
        raw = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = config_from_dict(raw)
    original = cfg.to_dict()
    cfg.maddpg_actor_sharing = True
    cfg.output_root = str(args.output_root.resolve())
    cfg.run_name = args.run_name
    if args.device:
        cfg.device = args.device
    validate_request(cfg, True, 50)
    resume_episode = 0
    if args.resume:
        resume_episode = int(read_resume(args.resume, cfg, 50)["episode"])
    from ..config import safe_run_dir
    destination = safe_run_dir(cfg.output_root, cfg.run_name)
    require(not destination.exists(), "output already exists; choose a new attempt name")
    report = {"run_dir": str(destination), "episodes": cfg.episodes, "interval": 50,
              "resume_source": str(args.resume) if args.resume else None,
              "completed_episodes": resume_episode, "next_episode": resume_episode + 1,
              "remaining_episodes": cfg.episodes - resume_episode,
              "changed_fields": [k for k, v in cfg.to_dict().items() if original.get(k) != v],
              "validation_rollouts": 0}
    if not args.dry_run:
        require(not _git_stamp()["reproduction_git_dirty"], "formal launch requires clean pinned code")
        from .runner import train
        report["result"] = train(cfg, resume=str(args.resume) if args.resume else None,
                                 training_reward_best=True, shared_resume_interval=50)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
