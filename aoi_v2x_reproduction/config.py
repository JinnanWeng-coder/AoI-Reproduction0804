"""Configuration, profile resolution, scenario registry, and safe run paths."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "configs" / "scenarios"

REPRODUCTION_PROFILE = "reproduction_baseline"
REPRODUCTION_SEMANTIC_VERSION = "reproduction_baseline_v1"
REPRODUCTION_MOBILITY_REVISION = "lane_graph_exit_safe_v1"
CHECKPOINT_SCHEMA_VERSION = "checkpoint_v4"
DEFAULT_ALGORITHM = "modified_maddpg_tdec"
SUPPORTED_ALGORITHMS = (DEFAULT_ALGORITHM, "modified_maddpg", "mappo")

MAPPO_CONFIG_FIELDS = (
    "mappo_actor_lr",
    "mappo_critic_lr",
    "mappo_rollout_episodes",
    "mappo_gae_lambda",
    "mappo_clip_param",
    "mappo_ppo_epochs",
    "mappo_num_minibatches",
    "mappo_value_loss_coef",
    "mappo_entropy_coef_rb",
    "mappo_entropy_coef_mode",
    "mappo_entropy_coef_power",
    "mappo_max_grad_norm",
    "mappo_adam_eps",
    "mappo_huber_delta",
)


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
    "algorithm": DEFAULT_ALGORITHM,
    "semantic_version": REPRODUCTION_SEMANTIC_VERSION,
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
    "tau": 0.005,
    "actor_lr": 0.0001,
    "critic_lr": 0.001,
    "policy_delay": 2,
    "target_noise_sigma": 0.2,
    "target_noise_clip": 0.5,
    "target_action_clip": 0.999,
    "global_actor_weight": 1.0,
    "mappo_actor_lr": 0.0005,
    "mappo_critic_lr": 0.0005,
    "mappo_rollout_episodes": 5,
    "mappo_gae_lambda": 0.95,
    "mappo_clip_param": 0.2,
    "mappo_ppo_epochs": 10,
    "mappo_num_minibatches": 1,
    "mappo_value_loss_coef": 1.0,
    "mappo_entropy_coef_rb": 0.01,
    "mappo_entropy_coef_mode": 0.01,
    "mappo_entropy_coef_power": 0.001,
    "mappo_max_grad_norm": 10.0,
    "mappo_adam_eps": 0.00001,
    "mappo_huber_delta": 10.0,
    "actor_hidden": [1024, 512],
    "local_critic_hidden": [512, 256],
    "global_critic_hidden": [1024, 512, 256],
    "exploration_noise": 0.3,
    "output_root": "experiments/runs",
    "device": "auto",
    "checkpoint_every": 50,
    "checkpoint_mode": "policy_only",
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
    "diagnostics": False,
    "selection_validation_seeds": [301, 302],
    "selection_validation_episodes": 5,
    "selection_validation_warmup_episodes": 1,
    "initial_aoi_ms": 100.0,
    "eval_protocol": "sequential_warm",
    "eval_warmup_episodes": 5,
    "global_reward_normalization": "source_normalized_per_rb_mean",
    "mobility_model": "urban_grid_correlated",
    "mobility_revision": REPRODUCTION_MOBILITY_REVISION,
    "gap_definition": "bumper_to_bumper",
    "vehicle_length_m": 4.0,
    "statistics_schema_version": "eval_seed_cluster_v1",
    "is_formal_result": False,
}


PROFILE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    REPRODUCTION_PROFILE: {},
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
    algorithm: str = DEFAULT_ALGORITHM
    semantic_version: str = REPRODUCTION_SEMANTIC_VERSION
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
    tau: float = 0.005
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    policy_delay: int = 2
    target_noise_sigma: float = 0.2
    target_noise_clip: float = 0.5
    target_action_clip: float = 0.999
    global_actor_weight: float = 1.0
    mappo_actor_lr: float = 0.0005
    mappo_critic_lr: float = 0.0005
    mappo_rollout_episodes: int = 5
    mappo_gae_lambda: float = 0.95
    mappo_clip_param: float = 0.2
    mappo_ppo_epochs: int = 10
    mappo_num_minibatches: int = 1
    mappo_value_loss_coef: float = 1.0
    mappo_entropy_coef_rb: float = 0.01
    mappo_entropy_coef_mode: float = 0.01
    mappo_entropy_coef_power: float = 0.001
    mappo_max_grad_norm: float = 10.0
    mappo_adam_eps: float = 0.00001
    mappo_huber_delta: float = 10.0
    actor_hidden: List[int] = field(default_factory=lambda: [1024, 512])
    local_critic_hidden: List[int] = field(default_factory=lambda: [512, 256])
    global_critic_hidden: List[int] = field(default_factory=lambda: [1024, 512, 256])
    exploration_noise: float = 0.3
    output_root: str = "experiments/runs"
    device: str = "auto"
    checkpoint_every: int = 50
    checkpoint_mode: str = "policy_only"
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
    diagnostics: bool = False
    selection_validation_seeds: List[int] = field(default_factory=lambda: [301, 302])
    selection_validation_episodes: int = 5
    selection_validation_warmup_episodes: int = 1
    initial_aoi_ms: float = 100.0
    eval_protocol: str = "sequential_warm"
    eval_warmup_episodes: int = 5
    global_reward_normalization: str = "source_normalized_per_rb_mean"
    mobility_model: str = "urban_grid_correlated"
    mobility_revision: str = REPRODUCTION_MOBILITY_REVISION
    gap_definition: str = "bumper_to_bumper"
    vehicle_length_m: float = 4.0
    statistics_schema_version: str = "eval_seed_cluster_v1"
    run_name: Optional[str] = None
    smoke: bool = False
    is_formal_result: bool = False

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

    @property
    def effective_center_spacing_m(self) -> float:
        """Center-to-center spacing used by the mobility model.

        The article reports bumper-to-bumper gap. The reproduction baseline
        adds the configured vehicle length to obtain center spacing.
        """
        if self.gap_definition == "bumper_to_bumper":
            return float(self.scenario.gap_m + self.vehicle_length_m)
        return float(self.scenario.gap_m)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # The original checkpoint_v4 files predate the explicit algorithm
        # field and are unambiguously TDec.  Omitting that default preserves
        # their canonical hashes, while Algorithm 1 is always explicit.
        if self.algorithm == DEFAULT_ALGORITHM:
            data.pop("algorithm", None)
        # Historical checkpoint_v4 configs predate checkpoint_mode.  Omitting
        # the new field preserves their canonical identity for read-only
        # analysis without re-enabling their training profiles.
        if self.semantic_version != REPRODUCTION_SEMANTIC_VERSION:
            data.pop("checkpoint_mode", None)
        # PPO-only controls must not alter the established Algorithm 1/2
        # config identities or their resolved artifacts.
        if self.algorithm != "mappo":
            for field_name in MAPPO_CONFIG_FIELDS:
                data.pop(field_name, None)
        data["scenario"] = asdict(self.scenario)
        data["derived"] = {
            "number_agents": self.number_agents,
            "number_vehicles": self.number_vehicles,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "v2i_min_bits_per_step": self.v2i_min_bits_per_step,
            "effective_center_spacing_m": self.effective_center_spacing_m,
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


def resolve_config(profile: str = REPRODUCTION_PROFILE, scenario: Optional[str] = None, **overrides: Any) -> ExperimentConfig:
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
    formal_was_explicit = "is_formal_result" in overrides
    config = ExperimentConfig(**values)
    validate_config(config)
    if config.profile == REPRODUCTION_PROFILE and config.is_formal_result:
        violations = baseline_contract_errors(config)
        if violations:
            if formal_was_explicit:
                raise ValueError(
                    "is_formal_result=True requires the exact reproduction baseline contract; "
                    "mismatched fields: " + ", ".join(violations)
                )
            # Unit tests, diagnostics, and smoke-sized configurations remain
            # supported, but can never silently retain a formal-result label.
            config.is_formal_result = False
    return config


def config_from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    """Reconstruct a current config or a historical config for read-only use."""
    allowed = {item.name for item in fields(ExperimentConfig)}
    values = {key: value for key, value in data.items() if key in allowed and key not in {"profile", "scenario"}}
    profile = str(data.get("profile", REPRODUCTION_PROFILE))
    scenario_data = data.get("scenario")
    if profile == REPRODUCTION_PROFILE:
        scenario = scenario_data.get("id") if isinstance(scenario_data, dict) else scenario_data
        return resolve_config(scenario=str(scenario), **values)
    if not isinstance(scenario_data, dict):
        raise ValueError("historical config requires an embedded scenario object")
    scenario = ScenarioConfig(
        id=str(scenario_data["id"]),
        number_platoons=int(scenario_data["number_platoons"]),
        platoon_size=int(scenario_data["platoon_size"]),
        gap_m=float(scenario_data["gap_m"]),
    )
    return ExperimentConfig(profile=profile, scenario=scenario, **values)


def validate_config(config: ExperimentConfig) -> None:
    if config.algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"unsupported algorithm: {config.algorithm}")
    if config.profile != REPRODUCTION_PROFILE:
        raise ValueError(f"active training requires profile={REPRODUCTION_PROFILE}")
    if config.episodes < 1 or config.steps_per_episode < 1:
        raise ValueError("episodes and steps_per_episode must be positive")
    if not math.isfinite(config.tau) or not 0.0 < config.tau <= 1.0:
        raise ValueError("tau must be finite and in (0, 1]")
    if not math.isclose(config.tau, 0.005, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("reproduction baseline requires tau=0.005")
    if config.checkpoint_mode not in {"none", "policy_only", "resumable"}:
        raise ValueError("checkpoint_mode must be none, policy_only, or resumable")
    if config.algorithm == "mappo" and config.checkpoint_mode == "resumable":
        raise ValueError("the first MAPPO baseline supports checkpoint_mode none or policy_only")
    if config.slow_update_every_episodes != 1:
        raise ValueError("reproduction baseline requires slow_update_every_episodes=1")
    if config.n_rb < 1 or config.n_modes < 2:
        raise ValueError("n_rb and n_modes are invalid")
    if config.power_min_dbm < 0 or config.power_max_dbm <= config.power_min_dbm:
        raise ValueError("invalid power interval")
    if config.global_actor_weight != 1.0:
        raise ValueError("reproduction baseline requires global_actor_weight=1.0")
    if config.global_update_mode not in {"detached_actor", "synchronous_joint"}:
        raise ValueError("unsupported global_update_mode")
    if not config.selection_validation_seeds or len(set(config.selection_validation_seeds)) != len(config.selection_validation_seeds):
        raise ValueError("selection_validation_seeds must be non-empty and unique")
    if set(int(seed) for seed in config.selection_validation_seeds) & set(range(101, 207)):
        raise ValueError("selection_validation_seeds must be disjoint from validation/final-test seeds 101..206")
    if config.selection_validation_episodes < 1 or config.selection_validation_warmup_episodes < 0:
        raise ValueError("selection validation episode counts are invalid")
    if config.semantic_version != REPRODUCTION_SEMANTIC_VERSION:
        raise ValueError(f"reproduction baseline requires semantic_version={REPRODUCTION_SEMANTIC_VERSION}")
    if config.initial_aoi_ms < 0:
        raise ValueError("initial_aoi_ms must be non-negative")
    if config.eval_protocol not in {"sequential_warm"}:
        raise ValueError("unsupported eval_protocol")
    if config.eval_warmup_episodes < 0:
        raise ValueError("eval_warmup_episodes must be non-negative")
    if config.global_reward_normalization not in {"source_normalized_per_rb_mean", "eq16_sum", "legacy_scalar"}:
        raise ValueError("unsupported global_reward_normalization")
    if config.mobility_model != "urban_grid_correlated":
        raise ValueError("unsupported mobility_model")
    if config.mobility_revision != REPRODUCTION_MOBILITY_REVISION:
        raise ValueError(f"reproduction baseline requires mobility_revision={REPRODUCTION_MOBILITY_REVISION}")
    if config.gap_definition not in {"bumper_to_bumper", "center_to_center"}:
        raise ValueError("unsupported gap_definition")
    if config.vehicle_length_m < 0:
        raise ValueError("vehicle_length_m must be non-negative")
    if config.statistics_schema_version != "eval_seed_cluster_v1":
        raise ValueError("unsupported statistics_schema_version")
    if config.previous_interference_dim not in {1, config.n_rb}:
        raise ValueError("previous_interference_dim must be 1 or n_rb")
    if len(config.rsu_position) != 2:
        raise ValueError("rsu_position must have two coordinates")
    if config.mappo_rollout_episodes < 1 or config.mappo_ppo_epochs < 1 or config.mappo_num_minibatches < 1:
        raise ValueError("MAPPO rollout episodes, PPO epochs, and minibatches must be positive")
    if config.mappo_num_minibatches != 1:
        raise ValueError("the first MAPPO baseline uses one full rollout minibatch")
    for name in ("mappo_actor_lr", "mappo_critic_lr", "mappo_max_grad_norm", "mappo_adam_eps", "mappo_huber_delta"):
        if not math.isfinite(float(getattr(config, name))) or float(getattr(config, name)) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 <= config.mappo_gae_lambda <= 1.0:
        raise ValueError("mappo_gae_lambda must be in [0, 1]")
    if not 0.0 < config.mappo_clip_param < 1.0:
        raise ValueError("mappo_clip_param must be in (0, 1)")
    for name in ("mappo_value_loss_coef", "mappo_entropy_coef_rb", "mappo_entropy_coef_mode", "mappo_entropy_coef_power"):
        if not math.isfinite(float(getattr(config, name))) or float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


# Runtime/path controls are deliberately excluded: device, output_root,
# run_name, checkpoint_every, and the resolved CUDA device do not change the
# scientific experiment.  Every scientific or statistical choice does.
_FORMAL_BASELINE_VALUES: Dict[str, Any] = {
    "algorithm": DEFAULT_ALGORITHM,
    "episodes": 500,
    "steps_per_episode": 100,
    "slot_ms": 1.0,
    "slow_fading_ms": 100.0,
    "n_rb": 3,
    "n_modes": 2,
    "bandwidth_hz": 180000,
    "cam_bits": 32000,
    "power_min_dbm": 1.0,
    "power_max_dbm": 30.0,
    "v2i_min_bps_per_hz": 3.0,
    "replay_capacity": 50000,
    "batch_size": 64,
    "gamma": 0.99,
    "tau": 0.005,
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
    "checkpoint_mode": "resumable",
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
    "selection_validation_seeds": [301, 302],
    "selection_validation_episodes": 5,
    "selection_validation_warmup_episodes": 1,
    "initial_aoi_ms": 100.0,
    "eval_protocol": "sequential_warm",
    "eval_warmup_episodes": 5,
    "global_reward_normalization": "source_normalized_per_rb_mean",
    "mobility_model": "urban_grid_correlated",
    "mobility_revision": REPRODUCTION_MOBILITY_REVISION,
    "gap_definition": "bumper_to_bumper",
    "vehicle_length_m": 4.0,
    "statistics_schema_version": "eval_seed_cluster_v1",
}


def _formal_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        return all(_formal_value_matches(item, wanted) for item, wanted in zip(actual, expected))
    return actual == expected


def baseline_contract_errors(config: ExperimentConfig) -> List[str]:
    """Return fields that prevent an artifact from being a frozen baseline run."""
    errors: List[str] = []
    if config.profile != REPRODUCTION_PROFILE:
        return ["profile"]
    if config.semantic_version != REPRODUCTION_SEMANTIC_VERSION:
        errors.append("semantic_version")
    for field_name, expected in _FORMAL_BASELINE_VALUES.items():
        if not _formal_value_matches(getattr(config, field_name, None), expected):
            errors.append(field_name)
    expected_scenario = DEFAULT_SCENARIOS.get(config.scenario.id)
    if expected_scenario is None:
        errors.append("scenario.id")
    else:
        for field_name in ("number_platoons", "platoon_size", "gap_m"):
            if not _formal_value_matches(getattr(config.scenario, field_name), expected_scenario[field_name]):
                errors.append(f"scenario.{field_name}")
    if int(config.seed) not in range(2, 8):
        errors.append("seed")
    if bool(config.smoke):
        errors.append("smoke")
    return errors


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


def matrix_specs(seeds: Iterable[int] = range(2, 8)) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for scenario in all_scenarios():
        for seed in seeds:
            name = f"{REPRODUCTION_PROFILE}_{scenario.id}_seed{int(seed):02d}"
            config = resolve_config(scenario=scenario.id, seed=int(seed), run_name=name)
            specs.append({
                "run_name": name,
                "profile": REPRODUCTION_PROFILE,
                "semantic_version": config.semantic_version,
                "scenario": scenario.id,
                "seed": int(seed),
                "state_dim": config.state_dim,
                "action_dim": config.action_dim,
                "config_hash": config.canonical_hash(),
            })
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
        "mappo_rollout_episodes": 2,
        "mappo_ppo_epochs": 2,
        "smoke": True,
        "is_formal_result": False,
    })
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AoI-V2X reproduction and MAPPO comparison runner")
    parser.add_argument("--algorithm", choices=SUPPORTED_ALGORITHMS, default=DEFAULT_ALGORITHM)
    parser.add_argument("--scenario", default="p05_n04_g25")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="experiments/runs")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--steps-per-episode", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--scope", choices=("train", "validation", "final_release"), default="train")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seeds", default=None)
    parser.add_argument("--eval-purpose", choices=("validation", "final_test"), default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-every", type=_positive_int, default=None)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("none", "policy_only", "resumable"),
        default=None,
        help="none, one final actor-only artifact, or full resumable checkpoints",
    )
    parser.add_argument("--diagnostics", action="store_true", help="record episode-aggregated actor-gradient diagnostics")
    parser.add_argument("--eval-noise", type=float, default=0.0, help="Gaussian action noise used only by --eval-only")
    parser.add_argument(
        "--diagnostic-eval",
        action="store_true",
        help="write an eval-only artifact without claiming a validation/final-release lifecycle marker",
    )
    parser.add_argument(
        "--recover-empty-run",
        action="store_true",
        help="reuse only a provenance-verified formal run initialized before its first checkpoint",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--matrix", action="store_true", help="print the complete 48-run matrix with --dry-run")
    parser.add_argument("--power-min-dbm", type=float, default=None)
    parser.add_argument("--power-max-dbm", type=float, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    overrides: Dict[str, Any] = {
        "algorithm": args.algorithm,
        "seed": args.seed,
        "device": args.device,
        "run_name": args.run_name,
        "output_root": args.output_root,
        "episodes": args.episodes,
        "steps_per_episode": args.steps_per_episode,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_mode": args.checkpoint_mode,
        "diagnostics": bool(args.diagnostics),
        "power_min_dbm": args.power_min_dbm,
        "power_max_dbm": args.power_max_dbm,
    }
    if args.smoke:
        overrides = apply_smoke_overrides(overrides)
    config = resolve_config(scenario=args.scenario, **overrides)
    if args.smoke:
        if not config.run_name:
            config.run_name = f"smoke_{config.scenario.id}_seed{args.seed:02d}"
        config.output_root = "scratch"
        config.is_formal_result = False
    if not config.run_name:
        config.run_name = f"{config.algorithm}_{config.scenario.id}_seed{config.seed:02d}"
    return config
