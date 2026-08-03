"""Configuration, profile resolution, scenario registry, and safe run paths."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "configs" / "scenarios"


DEFAULT_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "p05_n04_g05": {"id": "p05_n04_g05", "number_platoons": 5, "platoon_size": 4, "gap_m": 5.0},
    "p07_n04_g05": {"id": "p07_n04_g05", "number_platoons": 7, "platoon_size": 4, "gap_m": 5.0},
    "p05_n04_g15": {"id": "p05_n04_g15", "number_platoons": 5, "platoon_size": 4, "gap_m": 15.0},
    "p05_n04_g25": {"id": "p05_n04_g25", "number_platoons": 5, "platoon_size": 4, "gap_m": 25.0},
    "p05_n04_g35": {"id": "p05_n04_g35", "number_platoons": 5, "platoon_size": 4, "gap_m": 35.0},
    "p05_n06_g25": {"id": "p05_n06_g25", "number_platoons": 5, "platoon_size": 6, "gap_m": 25.0},
    "p05_n08_g25": {"id": "p05_n08_g25", "number_platoons": 5, "platoon_size": 8, "gap_m": 25.0},
    "p05_n10_g25": {"id": "p05_n10_g25", "number_platoons": 5, "platoon_size": 10, "gap_m": 25.0},
}


COMMON_DEFAULTS: Dict[str, Any] = {
    "episodes": 500,
    "steps_per_episode": 100,
    "slot_ms": 1.0,
    "slow_fading_ms": 100.0,
    "n_rb": 3,
    "n_modes": 2,
    "bandwidth_hz": 180000,
    "cam_bits": 32000,
    "power_max_dbm": 30.0,
    "power_min_dbm": 1.0,
    "v2i_min_bps_per_hz": 3.0,
    "replay_capacity": 50000,
    "batch_size": 64,
    "gamma": 0.99,
    "tau": 0.0005,
    "actor_lr": 0.0001,
    "critic_lr": 0.001,
    "policy_delay": 2,
    "target_noise_sigma": 0.2,
    "target_noise_clip": 0.5,
    "target_action_clip": 0.999,
    "global_actor_weight": 1.0,
    "actor_hidden": [1024, 512],
    "local_critic_hidden": [512, 256],
    "global_critic_hidden": [1024, 512, 256],
    "exploration_noise": 0.3,
    "output_root": "experiments/runs",
    "device": "auto",
    "checkpoint_every": 50,
    "slow_update_every_episodes": 1,
    "map_width_m": 750.0,
    "map_height_m": 1299.0,
    "rsu_position": [375.0, 649.5],
    "speed_min_mps": 10.0,
    "speed_max_mps": 15.0,
    "speed_distribution": "continuous_uniform",
    "power_continuous": True,
    "previous_interference_dim": 3,
    "include_remaining_time": True,
    "current_interference_reward": True,
    "global_update_mode": "synchronous_joint",
}


PROFILE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "paper_faithful": {},
    "legacy_release": {
        "tau": 0.005,
        "global_actor_weight": 2.0,
        "global_update_mode": "legacy_detach",
        "map_width_m": 375.0,
        "map_height_m": 649.0,
        "rsu_position": [375.0, 649.5],
        "speed_distribution": "legacy_integer_exclusive_high",
        "speed_max_mps": 15.0,
        "slow_update_every_episodes": 20,
        "power_continuous": False,
        "previous_interference_dim": 1,
        "include_remaining_time": False,
        "current_interference_reward": False,
    },
}


@dataclass(frozen=True)
class ScenarioConfig:
    id: str
    number_platoons: int
    platoon_size: int
    gap_m: float

    @property
    def number_vehicles(self) -> int:
        return self.number_platoons * self.platoon_size


@dataclass
class ExperimentConfig:
    profile: str
    scenario: ScenarioConfig
    seed: int = 2
    episodes: int = 500
    steps_per_episode: int = 100
    slot_ms: float = 1.0
    slow_fading_ms: float = 100.0
    n_rb: int = 3
    n_modes: int = 2
    bandwidth_hz: int = 180000
    cam_bits: int = 32000
    power_min_dbm: float = 1.0
    power_max_dbm: float = 30.0
    v2i_min_bps_per_hz: float = 3.0
    replay_capacity: int = 50000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.0005
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    policy_delay: int = 2
    target_noise_sigma: float = 0.2
    target_noise_clip: float = 0.5
    target_action_clip: float = 0.999
    global_actor_weight: float = 1.0
    actor_hidden: List[int] = field(default_factory=lambda: [1024, 512])
    local_critic_hidden: List[int] = field(default_factory=lambda: [512, 256])
    global_critic_hidden: List[int] = field(default_factory=lambda: [1024, 512, 256])
    exploration_noise: float = 0.3
    output_root: str = "experiments/runs"
    device: str = "auto"
    checkpoint_every: int = 50
    slow_update_every_episodes: int = 1
    map_width_m: float = 750.0
    map_height_m: float = 1299.0
    rsu_position: List[float] = field(default_factory=lambda: [375.0, 649.5])
    speed_min_mps: float = 10.0
    speed_max_mps: float = 15.0
    speed_distribution: str = "continuous_uniform"
    power_continuous: bool = True
    previous_interference_dim: int = 3
    include_remaining_time: bool = True
    current_interference_reward: bool = True
    global_update_mode: str = "synchronous_joint"
    run_name: Optional[str] = None
    smoke: bool = False
    is_formal_result: bool = True

    @property
    def v2i_min_bits_per_step(self) -> float:
        return self.v2i_min_bps_per_hz * self.bandwidth_hz * (self.slot_ms / 1000.0)

    @property
    def state_dim(self) -> int:
        n = self.scenario.platoon_size
        return 1 + self.n_rb + (n - 1) + (n - 1) * self.n_rb + self.previous_interference_dim + 1 + 1 + int(self.include_remaining_time)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def number_agents(self) -> int:
        return self.scenario.number_platoons

    @property
    def number_vehicles(self) -> int:
        return self.scenario.number_vehicles

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["scenario"] = asdict(self.scenario)
        data["derived"] = {
            "number_agents": self.number_agents,
            "number_vehicles": self.number_vehicles,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "v2i_min_bits_per_step": self.v2i_min_bits_per_step,
        }
        return data

    def canonical_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.lower() in {"true", "yes"}:
        return True
    if value.lower() in {"false", "no"}:
        return False
    if value.lower() in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("'\"")


def _minimal_yaml(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = _scalar(value)
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return dict(loaded or {})
    except ImportError:
        return _minimal_yaml(text)


def load_scenario(identifier: Optional[str]) -> ScenarioConfig:
    if not identifier:
        raw = DEFAULT_SCENARIOS["p05_n04_g25"]
    else:
        candidate = Path(identifier)
        if candidate.exists():
            raw = load_yaml(candidate)
        else:
            raw = DEFAULT_SCENARIOS.get(identifier)
            if raw is None:
                candidate = SCENARIO_DIR / f"{identifier}.yaml"
                if not candidate.exists():
                    raise FileNotFoundError(f"Unknown scenario: {identifier}")
                raw = load_yaml(candidate)
    scenario = ScenarioConfig(
        id=str(raw["id"]),
        number_platoons=int(raw["number_platoons"]),
        platoon_size=int(raw["platoon_size"]),
        gap_m=float(raw["gap_m"]),
    )
    if scenario.number_platoons < 1 or scenario.platoon_size < 2 or scenario.gap_m <= 0:
        raise ValueError(f"Invalid scenario: {scenario}")
    return scenario


def _profile_file(profile: str) -> Path:
    return ROOT / "configs" / f"{profile}.yaml"


def resolve_config(profile: str = "paper_faithful", scenario: Optional[str] = None, **overrides: Any) -> ExperimentConfig:
    if profile not in PROFILE_OVERRIDES:
        raise ValueError(f"profile must be one of {sorted(PROFILE_OVERRIDES)}")
    values = dict(COMMON_DEFAULTS)
    values.update(PROFILE_OVERRIDES[profile])
    profile_file = _profile_file(profile)
    if profile_file.exists():
        values.update(load_yaml(profile_file))
    values.update({key: value for key, value in overrides.items() if value is not None})
    scenario_obj = load_scenario(scenario or values.pop("scenario", None))
    values.pop("profile", None)
    values.pop("id", None)
    values["profile"] = profile
    values["scenario"] = scenario_obj
    config = ExperimentConfig(**values)
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.episodes < 1 or config.steps_per_episode < 1:
        raise ValueError("episodes and steps_per_episode must be positive")
    if config.n_rb < 1 or config.n_modes < 2:
        raise ValueError("n_rb and n_modes are invalid")
    if config.power_min_dbm < 0 or config.power_max_dbm <= config.power_min_dbm:
        raise ValueError("invalid power interval")
    if config.profile == "paper_faithful" and config.global_actor_weight != 1.0:
        raise ValueError("paper_faithful default global_actor_weight must remain 1.0")
    if config.global_update_mode not in {"legacy_detach", "synchronous_joint", "sequential_agent"}:
        raise ValueError("unsupported global_update_mode")
    if config.previous_interference_dim not in {1, config.n_rb}:
        raise ValueError("previous_interference_dim must be 1 or n_rb")
    if len(config.rsu_position) != 2:
        raise ValueError("rsu_position must have two coordinates")
    if config.profile == "paper_faithful":
        expected = [config.map_width_m / 2.0, config.map_height_m / 2.0]
        if not all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6) for a, b in zip(config.rsu_position, expected)):
            raise ValueError("paper_faithful RSU must be centered")


_RUN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_run_name(run_name: str) -> str:
    if not _RUN_RE.fullmatch(run_name) or run_name in {".", ".."}:
        raise ValueError("run-name must be a single safe directory name")
    return run_name


def safe_run_dir(output_root: Union[str, Path], run_name: str) -> Path:
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    root = root.resolve()
    validate_run_name(run_name)
    result = (root / run_name).resolve()
    if os.path.commonpath([str(root), str(result)]) != str(root):
        raise ValueError("run path escapes output-root")
    return result


def all_scenarios() -> List[ScenarioConfig]:
    return [load_scenario(name) for name in DEFAULT_SCENARIOS]


def matrix_specs(profile: str = "paper_faithful", seeds: Iterable[int] = range(2, 8)) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for scenario in all_scenarios():
        for seed in seeds:
            name = f"{profile}_{scenario.id}_seed{int(seed):02d}"
            config = resolve_config(profile=profile, scenario=scenario.id, seed=int(seed), run_name=name)
            specs.append({"run_name": name, "profile": profile, "scenario": scenario.id, "seed": int(seed), "state_dim": config.state_dim, "action_dim": config.action_dim})
    return specs


def apply_smoke_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(overrides)
    result.update({
        "episodes": 4,
        "steps_per_episode": 10,
        "batch_size": 4,
        "replay_capacity": 64,
        "checkpoint_every": 1,
        "actor_hidden": [64, 32],
        "local_critic_hidden": [64, 32],
        "global_critic_hidden": [64, 32, 16],
        "exploration_noise": 0.05,
        "smoke": True,
        "is_formal_result": False,
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modified MADDPG with task decomposition reproduction runner")
    parser.add_argument("--profile", choices=sorted(PROFILE_OVERRIDES), default="paper_faithful")
    parser.add_argument("--scenario", default="p05_n04_g25")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="experiments/runs")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps-per-episode", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seeds", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="print the complete 48-run matrix with --dry-run")
    parser.add_argument("--power-min-dbm", type=float, default=None)
    parser.add_argument("--power-max-dbm", type=float, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    overrides: Dict[str, Any] = {
        "seed": args.seed,
        "device": args.device,
        "run_name": args.run_name,
        "output_root": args.output_root,
        "episodes": args.episodes,
        "steps_per_episode": args.steps_per_episode,
        "checkpoint_every": args.checkpoint_every,
        "power_min_dbm": args.power_min_dbm,
        "power_max_dbm": args.power_max_dbm,
    }
    if args.smoke:
        overrides = apply_smoke_overrides(overrides)
    config = resolve_config(profile=args.profile, scenario=args.scenario, **overrides)
    if args.smoke:
        if not config.run_name:
            config.run_name = f"smoke_{args.profile}_{config.scenario.id}_seed{args.seed:02d}"
        config.output_root = "scratch"
        config.is_formal_result = False
    if not config.run_name:
        config.run_name = f"{config.profile}_{config.scenario.id}_seed{config.seed:02d}"
    return config
