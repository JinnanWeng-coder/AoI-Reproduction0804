"""Actor-only frozen MAPPO evaluation in a target scenario.

The policy artifact and actor architecture come from the source training run;
only the environment is rebuilt from the explicitly requested target scenario.
No critic, optimizer, rollout buffer, or action intervention is constructed.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from analysis.evaluate_mappo_feasibility_reward import (
    compare_reference_trajectory,
    reward_ledger,
)
from analysis.evaluate_mappo_power_intervention import (
    POLICY_SEED_TAG,
    _record_step as _record_base_step,
    _stack_trajectory,
    apply_power_intervention,
)
from aoi_v2x_reproduction.algorithms.mappo.action_adapter import encode_hybrid_actions
from aoi_v2x_reproduction.algorithms.mappo.networks import HybridActor
from aoi_v2x_reproduction.config import load_scenario, validate_config
from aoi_v2x_reproduction.envs.platoon import PaperEnviron
from aoi_v2x_reproduction.runtime.checkpointing import capture_rng_state, restore_rng_state
from aoi_v2x_reproduction.runtime.runner import (
    _git_metadata,
    _validate_mappo_policy_artifact,
    _write_json,
    resolve_device,
    seed_everything,
)


DEFAULT_EVAL_SEEDS = (213, 214, 215, 216, 217, 218)
TRAINING_SEEDS = (8, 9, 10, 11, 12, 13)
TARGETS = ("p05_n04_g05", "p05_n04_g35", "p07_n04_g25")


def task_to_target_structure_seed(task_id: int) -> tuple[str, str, int]:
    """Fixed 30-cell map: gap5 (12), gap35 (12), then P7 shared (6)."""

    task_id = int(task_id)
    if task_id < 0 or task_id >= 30:
        raise ValueError("task_id must satisfy 0 <= task_id < 30")
    if task_id < 24:
        target = ("p05_n04_g05", "p05_n04_g35")[task_id // 12]
        within = task_id % 12
        structure = ("independent", "shared")[within // 6]
        return target, structure, TRAINING_SEEDS[within % 6]
    return "p07_n04_g25", "shared", TRAINING_SEEDS[task_id - 24]


def zero_shot_eval_id(
    target_scenario: str,
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    eval_episodes: int = 100,
    warmup_episodes: int = 5,
) -> str:
    seed_token = "-".join(str(int(seed)) for seed in eval_seeds)
    return (
        f"eval_zero_shot_target_{target_scenario}_policy_final_stochastic_sequential_warm_"
        f"warm{int(warmup_episodes)}_s{seed_token}_ep{int(eval_episodes)}"
    )


class FrozenActorPolicy:
    """Minimal policy-only MAPPO loader preserving per-agent sampling order."""

    def __init__(self, source_config, actor_states: Sequence[dict], device: torch.device):
        self.source_config = source_config
        self.device = torch.device(device)
        self.actor_sharing = bool(source_config.mappo_actor_sharing)
        expected = 1 if self.actor_sharing else int(source_config.number_agents)
        if len(actor_states) != expected:
            raise RuntimeError(f"actor state count {len(actor_states)} does not match expected {expected}")
        self.actors = torch.nn.ModuleList([
            HybridActor(
                source_config.state_dim,
                source_config.actor_hidden,
                source_config.n_rb,
                source_config.n_modes,
            )
            for _ in range(expected)
        ]).to(self.device)
        for actor, state in zip(self.actors, actor_states):
            actor.load_state_dict(state, strict=True)
        self.actors.eval()

    def snapshot(self) -> list[list[torch.Tensor]]:
        return [[parameter.detach().cpu().clone() for parameter in actor.parameters()] for actor in self.actors]

    def unchanged(self, snapshot: Sequence[Sequence[torch.Tensor]]) -> bool:
        return all(
            torch.equal(parameter.detach().cpu(), expected)
            for actor, expected_actor in zip(self.actors, snapshot)
            for parameter, expected in zip(actor.parameters(), expected_actor)
        )

    @torch.no_grad()
    def act(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != int(self.source_config.state_dim):
            raise ValueError(
                f"target observations have shape {observations.shape}; expected [agent,{self.source_config.state_dim}]"
            )
        target_agents = int(observations.shape[0])
        if not self.actor_sharing and target_agents != int(self.source_config.number_agents):
            raise ValueError("an independent-actor policy cannot change the number of agents")
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        rb, mode, power = [], [], []
        for index in range(target_agents):
            actor = self.actors[0] if self.actor_sharing else self.actors[index]
            sample = actor.sample(tensor[index:index + 1], deterministic=False)
            rb.append(sample.rb.squeeze(0))
            mode.append(sample.mode.squeeze(0))
            power.append(sample.power.squeeze(0))
        return encode_hybrid_actions(
            torch.stack(rb).cpu().numpy().astype(np.int64),
            torch.stack(mode).cpu().numpy().astype(np.int64),
            torch.stack(power).cpu().numpy().astype(np.float32),
            self.source_config.n_rb,
            self.source_config.n_modes,
        )


def build_target_config(source_config, target_scenario: str, device: str):
    target = copy.deepcopy(source_config)
    target.scenario = load_scenario(target_scenario)
    target.device = str(device)
    target.is_formal_result = False
    validate_config(target)
    mismatches = []
    for field in ("state_dim", "n_rb", "n_modes", "power_min_dbm", "power_max_dbm", "power_continuous"):
        if getattr(source_config, field) != getattr(target, field):
            mismatches.append(field)
    if mismatches:
        raise ValueError("source actor and target action/observation spaces are incompatible: " + ", ".join(mismatches))
    if not bool(source_config.mappo_actor_sharing) and target.number_agents != source_config.number_agents:
        raise ValueError("cross-platoon-count zero-shot evaluation requires a shared actor")
    return target


def _step_record(info, actions, reward_global, reward_task1, reward_task2, config) -> dict:
    intervention = apply_power_intervention(
        actions, "baseline", config.power_min_dbm, config.power_max_dbm, config.n_rb, config.n_modes
    )
    record = _record_base_step(info, intervention)
    record.update(reward_ledger(
        info, reward_global, reward_task1, reward_task2, config.cam_bits, config.global_actor_weight
    ))
    return record


def _same_rb_pair_fraction(rb: np.ndarray) -> float:
    rb = np.asarray(rb)
    agents = rb.shape[-1]
    comparisons = [rb[..., left] == rb[..., right] for left in range(agents) for right in range(left + 1, agents)]
    return float(np.stack(comparisons, axis=-1).mean()) if comparisons else 0.0


def summarize_arrays(arrays: dict[str, np.ndarray], cam_bits: float) -> dict:
    aoi = arrays["aoi_ms"].astype(np.float64)
    endpoint_cam = arrays["success"][:, :, -1, :].astype(np.float64)
    endpoint_payload = np.clip(
        1.0 - arrays["remaining_demand"][:, :, -1, :].astype(np.float64) / float(cam_bits), 0.0, 1.0
    )
    per_agent_aoi = aoi.mean(axis=(0, 1, 2))
    per_agent_cam = endpoint_cam.mean(axis=(0, 1))
    per_agent_payload = endpoint_payload.mean(axis=(0, 1))
    return {
        "mean_AoI_ms": float(per_agent_aoi.mean()),
        "worst_agent_mean_AoI_ms": float(per_agent_aoi.max()),
        "aoi_gt50_fraction": float((aoi > 50.0).mean()),
        "aoi_at_cap_fraction": float(np.isclose(aoi, 100.0, rtol=0.0, atol=1e-6).mean()),
        "CAM_success_probability": float(per_agent_cam.mean()),
        "worst_agent_CAM_success_probability": float(per_agent_cam.min()),
        "payload_completion": float(per_agent_payload.mean()),
        "worst_agent_payload_completion": float(per_agent_payload.min()),
        "mean_reward_global": float(arrays["reward_global"].mean()),
        "mean_reward_task1": float(arrays["reward_task1"].mean()),
        "mean_reward_task2": float(arrays["reward_task2"].mean()),
        "mean_reward_combined": float(arrays["reward_combined"].mean()),
        "mean_executed_power_mw": float(np.power(10.0, arrays["executed_power_dbm"] / 10.0).mean()),
        "same_rb_pair_fraction": _same_rb_pair_fraction(arrays["rb"]),
        "screen_success": bool(per_agent_aoi.max() < 50.0 and per_agent_cam.min() >= 0.5),
    }


def evaluate_zero_shot(
    policy: Path,
    output_root: Path,
    target_scenario: str,
    device: str = "auto",
    eval_seeds: Iterable[int] = DEFAULT_EVAL_SEEDS,
    eval_episodes: int = 100,
    reference_metrics: Path | None = None,
) -> dict:
    eval_seeds = tuple(int(seed) for seed in eval_seeds)
    if not eval_seeds or len(eval_seeds) != len(set(eval_seeds)) or int(eval_episodes) < 1:
        raise ValueError("eval_seeds must be unique and eval_episodes must be positive")
    policy_path = Path(policy).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    caller_rng = capture_rng_state()
    try:
        payload = torch.load(policy_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError("MAPPO policy payload must be a dictionary")
        run_dir, source_config, source_complete = _validate_mappo_policy_artifact(policy_path, payload)
        if source_config.algorithm != "mappo" or source_config.mappo_variant != "tdec":
            raise RuntimeError("zero-shot evaluation requires a TDec MAPPO policy")
        target_config = build_target_config(source_config, target_scenario, device)
        if target_scenario not in TARGETS and target_scenario != source_config.scenario.id:
            raise ValueError(f"unsupported zero-shot target scenario: {target_scenario}")
        if target_config.eval_protocol != "sequential_warm" or int(target_config.eval_warmup_episodes) != 5:
            raise RuntimeError("unexpected source evaluation protocol")
        structure = "shared" if source_config.mappo_actor_sharing else "independent"
        eval_name = zero_shot_eval_id(
            target_config.scenario.id, eval_seeds, int(eval_episodes), int(target_config.eval_warmup_episodes)
        )
        eval_dir = output_root / target_config.scenario.id / structure / run_dir.name / eval_name
        if eval_dir.exists():
            raise FileExistsError(f"refusing to overwrite zero-shot evaluation: {eval_dir}")

        resolved_device = resolve_device(device)
        target_config.device_resolved = str(resolved_device)
        seed_everything(source_config.seed, resolved_device)
        environment = PaperEnviron(target_config)
        frozen = FrozenActorPolicy(source_config, payload["actors"], resolved_device)
        snapshot = frozen.snapshot()
        eval_dir.mkdir(parents=True, exist_ok=False)
        raw_by_seed = []
        for heldout_seed in eval_seeds:
            policy_seed = int(np.random.SeedSequence([
                int(source_config.seed), int(heldout_seed), POLICY_SEED_TAG
            ]).generate_state(1, dtype=np.uint32)[0])
            torch.manual_seed(policy_seed)
            if resolved_device.type == "cuda":
                torch.cuda.manual_seed_all(policy_seed)
            environment.reset_world(heldout_seed)
            for warmup_index in range(int(target_config.eval_warmup_episodes)):
                observations = environment.start_episode(warmup_index)
                for _slot in range(target_config.steps_per_episode):
                    actions = frozen.act(observations)
                    observations, _rg, _t1, _t2, _done, _info = environment.step(actions)
            seed_records = []
            for episode in range(int(eval_episodes)):
                observations = environment.start_episode(int(target_config.eval_warmup_episodes) + episode)
                episode_records = []
                for _slot in range(target_config.steps_per_episode):
                    actions = frozen.act(observations)
                    observations, rg, t1, t2, _done, info = environment.step(actions)
                    episode_records.append(_step_record(info, actions, rg, t1, t2, target_config))
                seed_records.append(episode_records)
            raw_by_seed.append(seed_records)
        if not frozen.unchanged(snapshot):
            raise RuntimeError("frozen zero-shot evaluation modified actor parameters")
    finally:
        restore_rng_state(caller_rng)

    arrays = _stack_trajectory(raw_by_seed)
    expected = (len(eval_seeds), int(eval_episodes), target_config.steps_per_episode, target_config.number_agents)
    if arrays["aoi_ms"].shape != expected:
        raise RuntimeError(f"unexpected zero-shot metric shape: {arrays['aoi_ms'].shape}, expected {expected}")
    metrics_path = eval_dir / "metrics.npz"
    np.savez_compressed(metrics_path, **arrays)
    equivalence = None
    if reference_metrics is not None:
        equivalence = compare_reference_trajectory(reference_metrics, metrics_path)
        _write_json(eval_dir / "trajectory_equivalence.json", equivalence)
        if not equivalence["all_common_arrays_exact"]:
            raise RuntimeError("actor-only same-scenario trajectory differs from the existing evaluator")

    git_metadata = _git_metadata()
    policy_reference = os.path.relpath(policy_path, eval_dir).replace(os.sep, "/")
    summary = {
        **git_metadata,
        **summarize_arrays(arrays, target_config.cam_bits),
        "status": "complete",
        "algorithm": "mappo",
        "mappo_variant": "tdec",
        "actor_sharing": bool(source_config.mappo_actor_sharing),
        "actor_structure": structure,
        "actor_network_count": len(frozen.actors),
        "actor_parameter_count": int(sum(parameter.numel() for actor in frozen.actors for parameter in actor.parameters())),
        "training_seed": int(source_config.seed),
        "training_run_name": run_dir.name,
        "source_scenario": source_config.scenario.id,
        "target_scenario": target_config.scenario.id,
        "source_number_agents": int(source_config.number_agents),
        "target_number_agents": int(target_config.number_agents),
        "source_config_hash": source_config.canonical_hash(),
        "target_config": target_config.to_dict(),
        "target_config_hash": target_config.canonical_hash(),
        "policy": policy_reference,
        "policy_path_is_relative_to_eval": True,
        "policy_schema_version": payload.get("policy_schema_version"),
        "policy_episode": int(payload.get("episode", -1)),
        "training_source_commit": source_complete.get("reproduction_git_commit"),
        "mappo_eval_mode": "stochastic",
        "external_action_noise_applicable": False,
        "eval_protocol": "sequential_warm",
        "eval_warmup_episodes": int(target_config.eval_warmup_episodes),
        "eval_seeds": list(eval_seeds),
        "eval_episodes": int(eval_episodes),
        "policy_parameters_unchanged": True,
        "critic_constructed": False,
        "optimizer_constructed": False,
        "rollout_constructed": False,
        "action_intervention": "none",
        "diagnostic_evaluation": True,
        "is_formal_result": False,
        "world_reuse_note": "worlds 213-218 are reused diagnostic worlds; this is not a new-world confirmation",
        "cam_bits": float(target_config.cam_bits),
        "global_actor_weight": float(target_config.global_actor_weight),
        "raw_metric_axes": {
            key: (["eval_seed", "scored_episode", "slot"] if value.ndim == 3 else
                  ["eval_seed", "scored_episode", "slot", "agent"] + (["action_dim"] if value.ndim == 5 else []))
            for key, value in arrays.items()
        },
        "reference_common_arrays_exact": equivalence["all_common_arrays_exact"] if equivalence else None,
        "eval_id": eval_name,
    }
    provenance = {
        **git_metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seed": int(source_config.seed),
        "training_run_name": run_dir.name,
        "source_scenario": source_config.scenario.id,
        "target_scenario": target_config.scenario.id,
        "actor_structure": structure,
        "training_source_commit": source_complete.get("reproduction_git_commit"),
        "source_config_hash": source_config.canonical_hash(),
        "target_config_hash": target_config.canonical_hash(),
        "policy": policy_reference,
        "eval_seeds": list(eval_seeds),
        "eval_episodes": int(eval_episodes),
        "release_status": "diagnostic_zero_shot_evaluation",
    }
    _write_json(eval_dir / "provenance.json", provenance)
    _write_json(eval_dir / "summary.json", summary)
    _write_json(eval_dir / "EVAL_COMPLETE.json", summary)
    return {"eval_dir": str(eval_dir), **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-scenario", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-seeds", default=",".join(str(seed) for seed in DEFAULT_EVAL_SEEDS))
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--reference-metrics", type=Path)
    args = parser.parse_args()
    result = evaluate_zero_shot(
        args.policy,
        args.output_root,
        args.target_scenario,
        args.device,
        [int(token) for token in args.eval_seeds.split(",") if token.strip()],
        args.eval_episodes,
        args.reference_metrics,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
