"""Training, evaluation, checkpoint, and provenance orchestration."""

from __future__ import annotations

import copy
import errno
import json
import hashlib
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .checkpointing import atomic_torch_save, build_payload, build_policy_payload, capture_rng_state, restore_rng_state
from ..algorithms.modified_maddpg.replay import ReplayBuffer
from ..envs.platoon import PaperEnviron
from ..config import (
    CHECKPOINT_SCHEMA_VERSION,
    ExperimentConfig,
    REPRODUCTION_PROFILE,
    REPRODUCTION_SEMANTIC_VERSION,
    baseline_contract_errors,
    config_from_dict,
    resolve_config,
    safe_run_dir,
)
from ..algorithms.modified_maddpg.learner import Global_Critic
from ..algorithms.modified_maddpg.agent import Agent
from ..algorithms.mappo.rollout import OnPolicyRollout
from ..algorithms.mappo.trainer import MAPPOTrainer
from .metrics import MetricStore


EVAL_PURPOSE_SEEDS = {
    "validation": [201, 202, 203, 204, 205, 206],
    "final_test": [101, 102, 103, 104, 105, 106],
}
EVAL_STATISTICS_SCHEMA_VERSION = "eval_seed_cluster_v1"
SCOPE_FOR_PURPOSE = {"validation": "validation", "final_test": "final_release"}
BEST_SELECTION_CRITERION = "maximize(min_agent_endpoint_cam - max_agent_mean_aoi_ms / initial_aoi_ms)"
FORMAL_PROVENANCE_KEYS = (
    "reproduction_git_commit",
    "reproduction_git_branch",
    "reproduction_git_dirty",
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        requested = "cuda:0"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device.index} is not available")
    return device


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _make_environment(config):
    return PaperEnviron(config)


def _make_system(config):
    device = resolve_device(config.device)
    config.device_resolved = str(device)
    seed_everything(config.seed, device)
    environment = _make_environment(config)
    agents = [Agent(config, index) for index in range(config.number_agents)]
    learner = Global_Critic(config, agents)
    replay = ReplayBuffer(config.replay_capacity, config.state_dim, config.action_dim, config.number_agents)
    metrics = MetricStore(
        config.number_agents,
        config.steps_per_episode,
        config.global_actor_weight,
        n_rb=config.n_rb,
        n_modes=config.n_modes,
        power_min_dbm=config.power_min_dbm,
        power_max_dbm=config.power_max_dbm,
        diagnostics=config.diagnostics,
        algorithm=config.algorithm,
    )
    return environment, agents, learner, replay, metrics, device


def _make_mappo_system(config):
    device = resolve_device(config.device)
    config.device_resolved = str(device)
    seed_everything(config.seed, device)
    environment = _make_environment(config)
    trainer = MAPPOTrainer(config, device)
    rollout = OnPolicyRollout(config.number_agents, config.state_dim)
    metrics = MetricStore(
        config.number_agents,
        config.steps_per_episode,
        config.global_actor_weight,
        n_rb=config.n_rb,
        n_modes=config.n_modes,
        power_min_dbm=config.power_min_dbm,
        power_max_dbm=config.power_max_dbm,
        diagnostics=config.diagnostics,
        algorithm=config.algorithm,
    )
    return environment, trainer, rollout, metrics, device


def _git_metadata() -> Dict[str, Any]:
    """Return reproducibility metadata without making Git state changes."""
    root = Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    device_names = []
    cuda_driver = None
    if torch.cuda.is_available():
        device_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        try:
            driver_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            cuda_driver = driver_result.stdout.strip().splitlines()[0] if driver_result.stdout.strip() else None
        except (OSError, subprocess.CalledProcessError, IndexError):
            cuda_driver = None
    return {
        "reproduction_git_commit": _git("rev-parse", "HEAD") or None,
        "reproduction_git_branch": _git("branch", "--show-current") or None,
        "reproduction_git_dirty": bool(_git("status", "--porcelain", "--untracked-files=all")),
        "gpu_names": device_names,
        "cuda_driver": cuda_driver,
    }


def _require_formal_scientific_contract(config: ExperimentConfig, context: str) -> None:
    violations = baseline_contract_errors(config)
    if violations:
        raise RuntimeError(
            f"{context} violates the frozen reproduction baseline contract: "
            + ", ".join(violations)
        )


def _require_current_formal_provenance(context: str) -> Dict[str, Any]:
    """Fail before a formal run/evaluation mutates its artifact directory."""
    current = _git_metadata()
    if current.get("reproduction_git_dirty") is not False:
        raise RuntimeError(f"{context} requires a clean reproduction Git worktree")
    for key in ("reproduction_git_commit", "reproduction_git_branch"):
        if not current.get(key):
            raise RuntimeError(f"{context} requires Git provenance field {key}")
    return current


def _require_matching_formal_provenance(
    record: Dict[str, Any],
    current: Dict[str, Any],
    label: str,
) -> None:
    if record.get("reproduction_git_dirty") is not False:
        raise RuntimeError(f"{label} must record reproduction_git_dirty=false")
    for key in FORMAL_PROVENANCE_KEYS:
        if record.get(key) is None:
            raise RuntimeError(f"{label} is missing provenance field {key}")
        if record.get(key) != current.get(key):
            raise RuntimeError(f"{label} provenance mismatch: {key}")


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{label} is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _require_checkpoint_in_run(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.name not in {"latest.pt", "best.pt"} or checkpoint_path.parent.name != "checkpoints":
        raise RuntimeError("evaluation/resume checkpoint must be checkpoints/latest.pt or checkpoints/best.pt")
    run_dir = checkpoint_path.parent.parent.resolve()
    allowed = {
        (run_dir / "checkpoints" / "latest.pt").resolve(),
        (run_dir / "checkpoints" / "best.pt").resolve(),
    }
    if checkpoint_path not in allowed:
        raise RuntimeError("checkpoint is outside this run's checkpoints directory")
    return run_dir


def _validate_formal_resume_preconditions(
    config: ExperimentConfig,
    checkpoint_path: Path,
    run_dir: Path,
    current: Dict[str, Any],
) -> None:
    if (run_dir / "COMPLETE.json").exists():
        raise RuntimeError("refusing to resume a run that already has COMPLETE.json")
    run_config_data = _read_json_object(run_dir / "config.resolved.json", "formal run config")
    provenance = _read_json_object(run_dir / "provenance.json", "formal run provenance")
    try:
        run_config = config_from_dict(run_config_data)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        payload_config = config_from_dict(payload["config"])
    except Exception as exc:
        raise RuntimeError(f"formal resume checkpoint/config is invalid: {exc}") from exc
    _require_formal_scientific_contract(run_config, "formal resume run config")
    if run_config.canonical_hash() != config.canonical_hash():
        raise RuntimeError("formal resume config does not match config.resolved.json")
    if payload_config.canonical_hash() != config.canonical_hash() or payload.get("config_hash") != config.canonical_hash():
        raise RuntimeError("formal resume checkpoint config mismatch")
    if payload.get("checkpoint_version") != 4 or payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("formal resume requires checkpoint_v4")
    if payload.get("semantic_version") != config.semantic_version or payload.get("mobility_revision") != config.mobility_revision:
        raise RuntimeError("formal resume checkpoint semantic/mobility mismatch")
    episode = int(payload.get("episode", -1))
    if payload.get("completed") is not False or episode < 0 or episode >= int(config.episodes):
        raise RuntimeError("formal resume requires an incomplete checkpoint before the final episode")
    _require_matching_formal_provenance(provenance, current, "formal run provenance")
    _require_matching_formal_provenance(payload, current, "formal resume checkpoint")
    for key, expected in (
        ("semantic_version", config.semantic_version),
        ("mobility_revision", config.mobility_revision),
        ("config_hash", config.canonical_hash()),
        ("checkpoint_schema_version", CHECKPOINT_SCHEMA_VERSION),
    ):
        if provenance.get(key) != expected:
            raise RuntimeError(f"formal run provenance mismatch: {key}")


class EmptyRunRecoveryError(RuntimeError):
    """Structured rejection for an initialized run with no checkpoint."""

    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code = code
        self.details = details


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return True
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _config_identity_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    scenario = data.get("scenario")
    scenario_id = scenario.get("id") if isinstance(scenario, dict) else scenario
    return {
        "algorithm": data.get("algorithm", "modified_maddpg_tdec"),
        "profile": data.get("profile"),
        "scenario": scenario_id,
        "seed": data.get("seed"),
        "run_name": data.get("run_name"),
        "output_root": data.get("output_root"),
        "device": data.get("device"),
    }


def _validate_empty_run_for_reinitialization(run_dir: Path, config: ExperimentConfig) -> Dict[str, Any]:
    """Verify a formal or explicitly diagnostic run before its first checkpoint.

    This function is read-only.  It accepts only the exact artifacts produced
    by the atomic initialization path and never removes or replaces anything.
    """
    run_dir = Path(run_dir).resolve()
    if config.checkpoint_mode != "resumable":
        raise EmptyRunRecoveryError("NOT_RESUMABLE", "empty-run recovery requires checkpoint_mode=resumable")
    expected_run_dir = safe_run_dir(config.output_root, config.run_name or "unnamed")
    if run_dir != expected_run_dir:
        raise EmptyRunRecoveryError(
            "IDENTITY_MISMATCH",
            "empty run path does not match the resolved output_root/run_name",
            run=str(run_dir),
            expected_run=str(expected_run_dir),
        )
    formal_run = bool(config.is_formal_result and config.profile == REPRODUCTION_PROFILE)
    diagnostic_run = bool(config.diagnostics and not config.smoke)
    if not formal_run and not diagnostic_run:
        raise EmptyRunRecoveryError("NOT_FORMAL", "empty-run recovery requires a formal baseline run or diagnostics=true")
    if formal_run:
        try:
            _require_formal_scientific_contract(config, "empty-run recovery")
        except RuntimeError as exc:
            raise EmptyRunRecoveryError("SCIENTIFIC_CONTRACT_MISMATCH", str(exc)) from exc
    if not run_dir.is_dir() or _is_link_or_reparse(run_dir):
        raise EmptyRunRecoveryError("DIRECTORY_INVALID", "empty run must be a real directory")

    allowed_names = {"checkpoints", "config.resolved.json", "provenance.json", "stdout.log"}
    entries = list(run_dir.iterdir())
    unexpected = sorted(entry.name for entry in entries if entry.name not in allowed_names)
    if unexpected:
        raise EmptyRunRecoveryError(
            "DIRECTORY_NOT_EMPTY",
            "empty run contains non-initialization artifacts",
            unexpected=unexpected,
        )
    by_name = {entry.name: entry for entry in entries}
    for required in ("checkpoints", "config.resolved.json", "provenance.json", "stdout.log"):
        if required not in by_name:
            raise EmptyRunRecoveryError("INITIALIZATION_INCOMPLETE", f"empty run is missing {required}", missing=required)
    if any(_is_link_or_reparse(entry) for entry in entries):
        raise EmptyRunRecoveryError("DIRECTORY_INVALID", "empty run initialization artifacts cannot be links or reparse points")
    checkpoints = by_name["checkpoints"]
    if not checkpoints.is_dir() or any(checkpoints.iterdir()):
        raise EmptyRunRecoveryError("CHECKPOINTS_NOT_EMPTY", "empty run checkpoints directory must be empty")
    if not by_name["config.resolved.json"].is_file() or not by_name["provenance.json"].is_file():
        raise EmptyRunRecoveryError("INITIALIZATION_INVALID", "config/provenance must be regular files")
    stdout = by_name["stdout.log"]
    if not stdout.is_file() or stdout.read_text(encoding="utf-8") != "run started\n":
        raise EmptyRunRecoveryError("STDOUT_INVALID", "empty run stdout.log is not the initialization marker")

    try:
        raw_config = _read_json_object(by_name["config.resolved.json"], "empty run config")
    except RuntimeError as exc:
        raise EmptyRunRecoveryError("CONFIG_INVALID", str(exc)) from exc
    expected_config = config.to_dict()
    expected_identity = _config_identity_from_dict(expected_config)
    actual_identity = _config_identity_from_dict(raw_config)
    identity_mismatches = {
        key: {"run": actual_identity.get(key), "expected": expected_identity.get(key)}
        for key in expected_identity
        if actual_identity.get(key) != expected_identity.get(key)
    }
    if identity_mismatches:
        raise EmptyRunRecoveryError(
            "IDENTITY_MISMATCH",
            "empty run config identity mismatch",
            mismatches=identity_mismatches,
        )
    try:
        raw_hash = hashlib.sha256(
            json.dumps(raw_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        reconstructed_hash = config_from_dict(raw_config).canonical_hash()
    except Exception as exc:
        raise EmptyRunRecoveryError("CONFIG_INVALID", f"empty run config cannot be resolved: {exc}") from exc
    if raw_hash != config.canonical_hash() or reconstructed_hash != config.canonical_hash():
        raise EmptyRunRecoveryError(
            "CONFIG_MISMATCH",
            "empty run config is not the complete canonical resolved config",
            run_config_hash=raw_hash,
            expected_config_hash=config.canonical_hash(),
        )

    try:
        provenance = _read_json_object(by_name["provenance.json"], "empty run provenance")
        current = _require_current_formal_provenance("empty-run recovery")
        _require_matching_formal_provenance(
            provenance,
            current,
            "empty run provenance",
        )
    except RuntimeError as exc:
        raise EmptyRunRecoveryError("PROVENANCE_MISMATCH", str(exc)) from exc
    expected_provenance = {
        "profile": config.profile,
        "semantic_version": config.semantic_version,
        "config_hash": config.canonical_hash(),
        "scenario": config.scenario.id,
        "seed": int(config.seed),
        "is_formal_result": bool(config.is_formal_result),
        "smoke": bool(config.smoke),
        "diagnostics_enabled": bool(config.diagnostics),
        "mobility_revision": config.mobility_revision,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
    }
    provenance_mismatches = {
        key: {"run": provenance.get(key), "expected": value}
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    if provenance_mismatches:
        raise EmptyRunRecoveryError(
            "PROVENANCE_MISMATCH",
            "empty run provenance metadata mismatch",
            mismatches=provenance_mismatches,
        )
    return {
        "status": "verified_empty_run",
        "run": str(run_dir),
        "config_hash": config.canonical_hash(),
        "identity": expected_identity,
    }


def _run_provenance(config: ExperimentConfig, git: Dict[str, Any]) -> Dict[str, Any]:
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": torch.version.cuda,
        "algorithm": config.algorithm,
        "profile": config.profile,
        "semantic_version": config.semantic_version,
        "config_hash": config.canonical_hash(),
        "scenario": config.scenario.id,
        "seed": config.seed,
        "is_formal_result": bool(config.is_formal_result),
        "smoke": bool(config.smoke),
        "eval_protocol": config.eval_protocol,
        "eval_warmup_episodes": int(config.eval_warmup_episodes),
        "global_reward_normalization": config.global_reward_normalization,
        "mobility_model": config.mobility_model,
        "mobility_revision": config.mobility_revision,
        "gap_definition": config.gap_definition,
        "vehicle_length_m": float(config.vehicle_length_m),
        "effective_center_spacing_m": float(config.effective_center_spacing_m),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "statistics_schema_version": config.statistics_schema_version,
        "diagnostics_enabled": bool(config.diagnostics),
        "checkpoint_mode": config.checkpoint_mode,
        "checkpoint_selection": (
            {
                "criterion": BEST_SELECTION_CRITERION,
                "seeds": [int(seed) for seed in config.selection_validation_seeds],
                "scored_episodes": int(config.selection_validation_episodes),
                "warmup_episodes": int(config.selection_validation_warmup_episodes),
                "disjoint_from_validation_and_final_test": True,
            }
            if config.checkpoint_mode == "resumable"
            else None
        ),
    }
    provenance.update(git)
    return provenance


_RENAMEAT2_NOREPLACE = 1  # Linux RENAME_NOREPLACE


def _unsupported_renameat2_errnos() -> frozenset:
    """Errnos that mean the filesystem cannot honor RENAME_NOREPLACE."""
    values = {errno.EINVAL, errno.ENOSYS}
    for name in ("EOPNOTSUPP", "ENOTSUP"):
        value = getattr(errno, name, None)
        if value is not None:
            values.add(value)
    return frozenset(values)


def _renameat2_noreplace(staging: Path, run_dir: Path) -> None:
    """Attempt renameat2(RENAME_NOREPLACE). Raises OSError on failure."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable", str(run_dir))
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    ctypes.set_errno(0)
    if renameat2(-100, os.fsencode(staging), -100, os.fsencode(run_dir), _RENAMEAT2_NOREPLACE) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(run_dir))


def _publish_via_mkdir_reservation(staging: Path, run_dir: Path) -> None:
    """Occupy ``run_dir`` with ``mkdir``, then replace the empty reservation.

    Never uses exists()+rename (TOCTOU). Never deletes conflicting or unknown
    directories. If the replace step fails after reservation, leave both
    ``staging`` and the empty reservation intact for inspection.
    """
    try:
        os.mkdir(run_dir)
    except FileExistsError:
        raise FileExistsError(
            errno.EEXIST,
            f"Refusing to publish over existing run directory: {run_dir}",
            str(run_dir),
        ) from None
    try:
        os.rename(staging, run_dir)
    except OSError as exc:
        raise OSError(
            exc.errno,
            (
                "Failed to replace empty run-dir reservation with staging "
                f"(staging and reservation preserved for inspection): {exc.strerror}"
            ),
            str(run_dir),
        ) from exc


def _publish_staged_run_no_replace(staging: Path, run_dir: Path) -> None:
    """Atomically publish a sibling staging directory without clobbering."""
    if os.name == "nt":
        os.rename(staging, run_dir)  # Windows rename fails if the target exists.
        return
    if sys.platform.startswith("linux"):
        try:
            _renameat2_noreplace(staging, run_dir)
            return
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FileExistsError(exc.errno, os.strerror(exc.errno), str(run_dir)) from exc
            if exc.errno not in _unsupported_renameat2_errnos():
                raise
            # Filesystem cannot honor RENAME_NOREPLACE (e.g. some NFS mounts).
    _publish_via_mkdir_reservation(staging, run_dir)


def _initialize_run_atomically(run_dir: Path, config: ExperimentConfig, git: Dict[str, Any]) -> None:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.init-", dir=str(run_dir.parent)))
    if config.checkpoint_mode == "resumable":
        (staging / "checkpoints").mkdir()
    _write_json(staging / "config.resolved.json", config.to_dict())
    _write_json(staging / "provenance.json", _run_provenance(config, git))
    (staging / "stdout.log").write_text("run started\n", encoding="utf-8")
    _publish_staged_run_no_replace(staging, run_dir)


def _prepare_run(
    config: ExperimentConfig,
    resume: Optional[str],
    recover_empty_run: bool = False,
) -> Tuple[Path, bool]:
    if resume:
        if recover_empty_run:
            raise ValueError("recover_empty_run cannot be combined with a checkpoint resume")
        checkpoint = Path(resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        run_dir = _require_checkpoint_in_run(checkpoint)
        if config.is_formal_result:
            _require_formal_scientific_contract(config, "formal resume")
            current = _require_current_formal_provenance("formal resume")
            _validate_formal_resume_preconditions(config, checkpoint, run_dir, current)
        return run_dir, True
    run_dir = safe_run_dir(config.output_root, config.run_name or "unnamed")
    if run_dir.exists():
        if recover_empty_run:
            _validate_empty_run_for_reinitialization(run_dir, config)
            return run_dir, False
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    if config.is_formal_result:
        _require_formal_scientific_contract(config, "formal training")
        git = _require_current_formal_provenance("formal training")
    else:
        git = _git_metadata()
    _initialize_run_atomically(run_dir, config, git)
    return run_dir, False


def _load_checkpoint(path: Path, config, agents, learner, replay, environment, metrics):
    # Deserialize portable checkpoint data on CPU.  Module/optimizer
    # load_state_dict calls below copy their tensors to the live parameter
    # device, while CPU-only RNG state remains valid for torch.set_rng_state.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_checkpoint_config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    checkpoint_algorithm = payload.get("algorithm", raw_checkpoint_config.get("algorithm", "modified_maddpg_tdec"))
    if checkpoint_algorithm != config.algorithm:
        raise ValueError(
            f"checkpoint algorithm mismatch: checkpoint={checkpoint_algorithm!r}, resolved={config.algorithm!r}"
        )
    checkpoint_semantic_version = payload.get("semantic_version")
    checkpoint_version = int(payload.get("checkpoint_version", 0))
    if checkpoint_semantic_version != config.semantic_version:
        raise ValueError(
            "checkpoint semantic_version mismatch: "
            f"checkpoint={checkpoint_semantic_version!r}, resolved={config.semantic_version!r}"
        )
    if checkpoint_version != 4:
        raise ValueError(
            "reproduction_baseline_v1 requires checkpoint_version=4; "
            f"received {checkpoint_version}"
        )
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is not checkpoint_v4")
    if payload.get("mobility_revision") != config.mobility_revision:
        raise ValueError("checkpoint mobility_revision does not match resolved config")
    if payload.get("config_hash") != config.canonical_hash():
        raise ValueError("checkpoint config hash does not match resolved config")
    if payload.get("gap_definition") != config.gap_definition:
        raise ValueError("checkpoint gap_definition does not match resolved config")
    if float(payload.get("vehicle_length_m", -1.0)) != float(config.vehicle_length_m):
        raise ValueError("checkpoint vehicle_length_m does not match resolved config")
    for agent, state in zip(agents, payload["agents"]):
        agent.load_state_dict_full(state)
    learner.load_state_dict_full(payload["learner"])
    replay.load_state_dict(payload["replay"])
    if payload.get("environment") is not None and hasattr(environment, "load_state_dict"):
        environment.load_state_dict(payload["environment"])
    metrics.load_state_dict(payload["metrics"])
    restore_rng_state(payload["rng"])
    return payload


def _record_step(info: Dict[str, Any], normalized_actions: Optional[np.ndarray] = None) -> Dict[str, Any]:
    # String metadata belongs in config/provenance, not in numeric NPZ
    # tensors.  The raw per-RB arrays remain available for audit/smoke while
    # this filter prevents accidental object/string arrays.
    record = {
        key: np.asarray(value).copy()
        for key, value in info.items()
        if key not in {"actions_decoded", "global_reward_normalization"}
    }
    if normalized_actions is not None:
        record["action_post_clip_normalized"] = np.asarray(normalized_actions, dtype=np.float32).copy()
    return record


def _selection_validation(config: ExperimentConfig, agents) -> Dict[str, Any]:
    """Evaluate the live policy on a fixed split without perturbing training state."""
    caller_rng = capture_rng_state()
    training_modes = [agent.actor.training for agent in agents]
    try:
        environment = _make_environment(config)
        for agent in agents:
            agent.actor.eval()
        seed_aoi = []
        seed_cam = []
        for seed in config.selection_validation_seeds:
            environment.reset_world(int(seed))
            for episode in range(int(config.selection_validation_warmup_episodes)):
                observations = environment.start_episode(episode)
                for _step in range(config.steps_per_episode):
                    actions = np.asarray(
                        [agent.choose_action(observations[index], explore=False, noise_std=0.0) for index, agent in enumerate(agents)],
                        dtype=np.float32,
                    )
                    observations, _rg, _t1, _t2, _done, _info = environment.step(actions)
            episode_aoi = []
            episode_cam = []
            for scored_episode in range(int(config.selection_validation_episodes)):
                episode_index = int(config.selection_validation_warmup_episodes) + scored_episode
                observations = environment.start_episode(episode_index)
                slot_aoi = []
                endpoint_cam = None
                for _step in range(config.steps_per_episode):
                    actions = np.asarray(
                        [agent.choose_action(observations[index], explore=False, noise_std=0.0) for index, agent in enumerate(agents)],
                        dtype=np.float32,
                    )
                    observations, _rg, _t1, _t2, _done, info = environment.step(actions)
                    slot_aoi.append(np.asarray(info["aoi_ms"], dtype=np.float64))
                    endpoint_cam = np.asarray(info["success"], dtype=np.float64)
                episode_aoi.append(np.mean(np.stack(slot_aoi), axis=0))
                episode_cam.append(endpoint_cam)
            seed_aoi.append(np.mean(np.stack(episode_aoi), axis=0))
            seed_cam.append(np.mean(np.stack(episode_cam), axis=0))
        per_agent_aoi = np.mean(np.stack(seed_aoi), axis=0)
        per_agent_cam = np.mean(np.stack(seed_cam), axis=0)
        worst_aoi = float(np.max(per_agent_aoi))
        worst_cam = float(np.min(per_agent_cam))
        aoi_scale = max(float(config.initial_aoi_ms), 1.0)
        score = float(worst_cam - worst_aoi / aoi_scale)
        return {
            "criterion": BEST_SELECTION_CRITERION,
            "seeds": [int(seed) for seed in config.selection_validation_seeds],
            "scored_episodes": int(config.selection_validation_episodes),
            "warmup_episodes": int(config.selection_validation_warmup_episodes),
            "mean_aoi_ms_per_agent": per_agent_aoi.tolist(),
            "endpoint_cam_per_agent": per_agent_cam.tolist(),
            "mean_aoi_ms": float(np.mean(per_agent_aoi)),
            "endpoint_cam": float(np.mean(per_agent_cam)),
            "worst_agent_mean_aoi_ms": worst_aoi,
            "worst_agent_endpoint_cam": worst_cam,
            "score": score,
        }
    finally:
        for agent, was_training in zip(agents, training_modes):
            agent.actor.train(was_training)
        restore_rng_state(caller_rng)


def _save_actor_snapshot(run_dir: Path, config: ExperimentConfig, agents, episode: int, selection: Dict[str, Any]) -> Path:
    """Keep a lightweight policy snapshot; latest.pt remains the resumable checkpoint."""
    path = run_dir / "checkpoints" / "periodic" / f"episode_{int(episode):06d}_actor.pt"
    atomic_torch_save({
        "snapshot_schema_version": "actor_snapshot_v1",
        "checkpoint_role": "periodic_actor_snapshot_not_resumable",
        "semantic_version": config.semantic_version,
        "config_hash": config.canonical_hash(),
        "episode": int(episode),
        "actors": [agent.actor.state_dict() for agent in agents],
        "selection_validation": selection,
    }, path)
    return path


def _selection_best_payload(latest_payload: Dict[str, Any], selection_state: Dict[str, Any], training_completed: bool) -> Dict[str, Any]:
    selected_episode = int(selection_state.get("best_episode") or -1)
    selection = selection_state.get("best_metrics")
    if selected_episode < 1 or not isinstance(selection, dict) or int(selection.get("episode", -1)) != selected_episode:
        raise RuntimeError("selection_state cannot produce a valid best checkpoint")
    payload = dict(latest_payload)
    payload.update({
        "checkpoint_role": "best_selection_validation",
        "selected_episode": selected_episode,
        "selection_validation": selection,
        "selection_state": selection_state,
        "training_completed": bool(training_completed),
    })
    return payload


def _repair_selection_best_after_resume(run_dir: Path, latest_payload: Dict[str, Any], selection_state: Dict[str, Any]) -> None:
    """Repair an interruption between atomic latest.pt and best.pt writes."""
    selected_episode = selection_state.get("best_episode")
    if selected_episode is None:
        return
    selected_episode = int(selected_episode)
    best_path = run_dir / "checkpoints" / "best.pt"
    consistent = False
    if best_path.is_file():
        try:
            existing = torch.load(best_path, map_location="cpu", weights_only=False)
            consistent = (
                existing.get("checkpoint_role") == "best_selection_validation"
                and int(existing.get("selected_episode", -1)) == selected_episode
                and int(existing.get("episode", -2)) == selected_episode
            )
        except Exception:
            consistent = False
    if consistent:
        return
    latest_episode = int(latest_payload.get("episode", -1))
    if selected_episode != latest_episode:
        raise RuntimeError(
            "latest.pt selection_state disagrees with best.pt and cannot be repaired from the resumed episode"
        )
    atomic_torch_save(
        _selection_best_payload(latest_payload, selection_state, training_completed=False),
        best_path,
    )


def _train_mappo(
    config: ExperimentConfig,
    resume: Optional[str] = None,
    max_episodes: Optional[int] = None,
    recover_empty_run: bool = False,
) -> Dict[str, Any]:
    if resume or recover_empty_run:
        raise ValueError("the first MAPPO baseline does not support training resume or empty-run recovery")
    if config.checkpoint_mode == "resumable":
        raise ValueError("the first MAPPO baseline supports checkpoint_mode none or policy_only")
    if max_episodes is not None and int(max_episodes) < int(config.episodes):
        raise ValueError("the first MAPPO baseline does not support partial training runs")
    run_dir, is_resume = _prepare_run(config, None, recover_empty_run=False)
    if is_resume:
        raise RuntimeError("MAPPO exploratory runs must start from a new run directory")
    environment, trainer, rollout, metrics, device = _make_mappo_system(config)
    if hasattr(environment, "reset_world") and not getattr(environment, "_world_initialized", False):
        environment.reset_world(config.seed)

    started = time.perf_counter()
    stop_episode = int(config.episodes)
    for episode in range(stop_episode):
        observations = environment.start_episode(episode)
        task1_steps: List[np.ndarray] = []
        task2_steps: List[np.ndarray] = []
        global_steps: List[float] = []
        step_records: List[Dict[str, Any]] = []
        for _step in range(config.steps_per_episode):
            sampled = trainer.act(observations, deterministic=False)
            next_observations, reward_global, reward_task1, reward_task2, terminated, info = environment.step(
                sampled.environment_actions
            )
            next_values = (
                np.zeros(config.number_agents, dtype=np.float32)
                if terminated
                else trainer.values(next_observations)
            )
            rollout.append(
                observations=observations,
                rb=sampled.rb,
                mode=sampled.mode,
                power=sampled.power,
                old_log_prob=sampled.log_prob,
                values=sampled.values,
                rewards=trainer.combined_rewards(reward_global, reward_task1, reward_task2),
                done=terminated,
                next_values=next_values,
                policy_version=trainer.policy_version,
            )
            task1_steps.append(np.asarray(reward_task1, dtype=np.float32))
            task2_steps.append(np.asarray(reward_task2, dtype=np.float32))
            global_steps.append(float(reward_global))
            step_records.append(_record_step(info, sampled.environment_actions))
            observations = next_observations
        metrics.append_episode(step_records, task1_steps, task2_steps, global_steps)

        rollout_ready = rollout.terminal_count >= int(config.mappo_rollout_episodes)
        final_episode_this_call = episode == stop_episode - 1
        if rollout_ready or final_episode_this_call:
            batch = rollout.consume(
                gamma=config.gamma,
                gae_lambda=config.mappo_gae_lambda,
                expected_policy_version=trainer.policy_version,
            )
            diagnostics = trainer.update(batch)
            metrics.append_mappo_update(episode + 1, diagnostics)

    if len(rollout) != 0:
        raise RuntimeError("MAPPO rollout was not consumed at the training boundary")
    shapes = metrics.save(run_dir)
    policy_saved = False
    if config.checkpoint_mode == "policy_only":
        atomic_torch_save(build_policy_payload(config, trainer, config.episodes), run_dir / "policy_final.pt")
        policy_saved = True
    parameter_counts = trainer.parameter_counts()
    wall_seconds = float(time.perf_counter() - started)
    audit = {
        "ok": True,
        "scope": "train",
        "algorithm": "mappo",
        "checkpoint_mode": config.checkpoint_mode,
        "metrics_shapes": shapes,
        "update_count": int(trainer.update_count),
        "environment_steps": int(trainer.environment_steps),
    }
    provenance_data = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    complete = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_formal_result": False,
        "algorithm": "mappo",
        "profile": config.profile,
        "semantic_version": config.semantic_version,
        "mobility_revision": config.mobility_revision,
        "config_hash": config.canonical_hash(),
        "reproduction_git_commit": provenance_data.get("reproduction_git_commit"),
        "reproduction_git_branch": provenance_data.get("reproduction_git_branch"),
        "reproduction_git_dirty": provenance_data.get("reproduction_git_dirty"),
        "python": provenance_data.get("python"),
        "numpy": provenance_data.get("numpy"),
        "torch": provenance_data.get("torch"),
        "cuda_version": provenance_data.get("cuda_version"),
        "cuda_driver": provenance_data.get("cuda_driver"),
        "gpu_names": provenance_data.get("gpu_names", []),
        "scenario": config.scenario.id,
        "seed": int(config.seed),
        "episodes": int(config.episodes),
        "steps_per_episode": int(config.steps_per_episode),
        "environment_steps": int(trainer.environment_steps),
        "update_count": int(trainer.update_count),
        "checkpoint_mode": config.checkpoint_mode,
        "policy_final": "policy_final.pt" if policy_saved else None,
        "reward_semantics": "global_plus_per_agent_task1_plus_task2",
        "actor_sharing": False,
        "central_critic_output": "per_agent_state_value",
        "power_distribution": "beta_open_unit_interval",
        "algorithm_applicability": {
            "polyak_tau_applicable": False,
            "external_action_noise_applicable": False,
            "global_actor_update_mode_applicable": False,
        },
        "parameter_counts": parameter_counts,
        "training_wall_seconds": wall_seconds,
        "metrics_shapes": shapes,
        "audit": audit,
    }
    _write_json(run_dir / "COMPLETE.json", complete)
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as handle:
        handle.write("MAPPO run completed\n")
    return {
        "run_dir": str(run_dir),
        "episodes": int(config.episodes),
        "device": str(device),
        "audit": audit,
        "parameter_counts": parameter_counts,
    }


def train(
    config: ExperimentConfig,
    resume: Optional[str] = None,
    max_episodes: Optional[int] = None,
    recover_empty_run: bool = False,
) -> Dict[str, Any]:
    if config.algorithm == "mappo":
        return _train_mappo(config, resume, max_episodes, recover_empty_run)
    if resume and config.checkpoint_mode != "resumable":
        raise ValueError("training resume requires checkpoint_mode=resumable")
    run_dir, is_resume = _prepare_run(config, resume, recover_empty_run=recover_empty_run)
    environment, agents, learner, replay, metrics, device = _make_system(config)
    start_episode = 0
    selection_state: Optional[Dict[str, Any]] = None
    if config.checkpoint_mode == "resumable":
        selection_state = {
            "criterion": BEST_SELECTION_CRITERION,
            "seeds": [int(seed) for seed in config.selection_validation_seeds],
            "scored_episodes": int(config.selection_validation_episodes),
            "warmup_episodes": int(config.selection_validation_warmup_episodes),
            "best_score": None,
            "best_episode": None,
            "best_metrics": None,
        }
    if is_resume:
        payload = _load_checkpoint(Path(resume).expanduser().resolve(), config, agents, learner, replay, environment, metrics)
        start_episode = int(payload["episode"])
        if isinstance(payload.get("selection_state"), dict):
            selection_state = dict(payload["selection_state"])
        if payload.get("completed"):
            raise RuntimeError("refusing to resume a completed run")
        if selection_state is None:
            raise RuntimeError("resumable checkpoint is missing selection_state")
        _repair_selection_best_after_resume(run_dir, payload, selection_state)
    elif hasattr(environment, "reset_world") and not getattr(environment, "_world_initialized", False):
        environment.reset_world(config.seed)

    stop_episode = config.episodes if max_episodes is None else min(config.episodes, int(max_episodes))
    for episode in range(start_episode, stop_episode):
        if hasattr(environment, "start_episode"):
            observations = environment.start_episode(episode)
        else:
            observations = environment.reset_episode(episode)
        task1_steps: List[np.ndarray] = []
        task2_steps: List[np.ndarray] = []
        global_steps: List[float] = []
        step_records: List[Dict[str, Any]] = []
        learning_records: List[Dict[str, Any]] = []
        for _step in range(config.steps_per_episode):
            actions = np.asarray([agent.choose_action(observations[index], explore=True) for index, agent in enumerate(agents)], dtype=np.float32)
            next_observations, reward_global, reward_task1, reward_task2, terminated, info = environment.step(actions)
            replay.store_transition(observations.reshape(-1), actions.reshape(-1), reward_global, reward_task1, reward_task2, next_observations.reshape(-1), terminated)
            if replay.size >= config.batch_size:
                diagnostics = learner.learn(replay.sample_buffer(config.batch_size))
                learning_records.append(diagnostics)
            task1_steps.append(np.asarray(reward_task1, dtype=np.float32))
            task2_steps.append(np.asarray(reward_task2, dtype=np.float32))
            global_steps.append(float(reward_global))
            step_records.append(_record_step(info, actions))
            observations = next_observations
        metrics.append_episode(step_records, task1_steps, task2_steps, global_steps)
        metrics.append_learning_episode(learning_records)
        if config.checkpoint_mode == "resumable" and (
            ((episode + 1) % config.checkpoint_every == 0) or episode == stop_episode - 1
        ):
            assert selection_state is not None
            selection = _selection_validation(config, agents)
            selection["episode"] = int(episode + 1)
            _save_actor_snapshot(run_dir, config, agents, episode + 1, selection)
            is_better = selection_state.get("best_score") is None or float(selection["score"]) > float(selection_state["best_score"])
            if is_better:
                selection_state["best_score"] = float(selection["score"])
                selection_state["best_episode"] = int(episode + 1)
                selection_state["best_metrics"] = selection
            payload = build_payload(config, agents, learner, replay, environment, metrics, episode + 1, completed=False)
            payload.update({"checkpoint_role": "latest_resumable", "selection_state": selection_state})
            atomic_torch_save(payload, run_dir / "checkpoints" / "latest.pt")
            if is_better:
                atomic_torch_save(
                    _selection_best_payload(payload, selection_state, training_completed=False),
                    run_dir / "checkpoints" / "best.pt",
                )

    if stop_episode < config.episodes:
        shapes = metrics.save(run_dir)
        return {"run_dir": str(run_dir), "episodes_completed": stop_episode, "interrupted": True, "metrics_shapes": shapes, "device": str(device)}

    shapes = metrics.save(run_dir)
    final_checkpoint_hashes: Dict[str, str] = {}
    policy_saved = False
    if config.checkpoint_mode == "resumable":
        assert selection_state is not None
        final_payload = build_payload(config, agents, learner, replay, environment, metrics, config.episodes, completed=True)
        final_payload.update({"checkpoint_role": "latest_final", "selection_state": selection_state, "training_completed": True})
        atomic_torch_save(final_payload, run_dir / "checkpoints" / "latest.pt")
        best_path = run_dir / "checkpoints" / "best.pt"
        if int(selection_state.get("best_episode") or -1) == int(config.episodes):
            best_payload = _selection_best_payload(final_payload, selection_state, training_completed=True)
        else:
            best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
            best_payload.update({
                "completed": True,
                "training_completed": True,
                "checkpoint_role": "best_selection_validation",
                "selected_episode": int(selection_state["best_episode"]),
                "selection_state": selection_state,
            })
        atomic_torch_save(best_payload, best_path)
        final_checkpoint_hashes = {
            name: _sha256_file(run_dir / "checkpoints" / name)
            for name in ("latest.pt", "best.pt")
        }
    elif config.checkpoint_mode == "policy_only":
        atomic_torch_save(build_policy_payload(config, agents, config.episodes), run_dir / "policy_final.pt")
        policy_saved = True

    audit = {
        "ok": True,
        "scope": "train",
        "checkpoint_mode": config.checkpoint_mode,
        "metrics_shapes": shapes,
    }
    provenance_data = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    complete = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_formal_result": bool(config.is_formal_result),
        "algorithm": config.algorithm,
        "profile": config.profile,
        "semantic_version": config.semantic_version,
        "mobility_revision": config.mobility_revision,
        "checkpoint_schema_version": "checkpoint_v4" if config.checkpoint_mode == "resumable" else None,
        "config_hash": config.canonical_hash(),
        "reproduction_git_commit": provenance_data.get("reproduction_git_commit"),
        "reproduction_git_branch": provenance_data.get("reproduction_git_branch"),
        "reproduction_git_dirty": provenance_data.get("reproduction_git_dirty"),
        "python": provenance_data.get("python"),
        "numpy": provenance_data.get("numpy"),
        "torch": provenance_data.get("torch"),
        "cuda_version": provenance_data.get("cuda_version"),
        "cuda_driver": provenance_data.get("cuda_driver"),
        "gpu_names": provenance_data.get("gpu_names", []),
        "gap_definition": config.gap_definition,
        "vehicle_length_m": float(config.vehicle_length_m),
        "effective_center_spacing_m": float(config.effective_center_spacing_m),
        "scenario": config.scenario.id,
        "seed": config.seed,
        "episodes": config.episodes,
        "final_episode": config.episodes,
        "checkpoint_mode": config.checkpoint_mode,
        "checkpoint_completed": config.checkpoint_mode == "resumable",
        "checkpoint_sha256": final_checkpoint_hashes,
        "policy_final": "policy_final.pt" if policy_saved else None,
        "metrics_shapes": shapes,
        "checkpoint_selection": selection_state,
        "audit": audit,
    }
    _write_json(run_dir / "COMPLETE.json", complete)
    with (run_dir / "stdout.log").open("a", encoding="utf-8") as handle:
        handle.write("run completed\n")
    return {"run_dir": str(run_dir), "episodes": config.episodes, "device": str(device), "audit": audit}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eval_id(checkpoint_hash: str, purpose: str, protocol: str, warmup_episodes: int, seeds: Iterable[int], episodes: int, noise: float) -> str:
    seed_token = "-".join(str(int(seed)) for seed in seeds)
    noise_token = str(noise).replace(".", "p")
    return f"eval_{purpose}_ckpt{checkpoint_hash[:12]}_{protocol}_warm{int(warmup_episodes)}_s{seed_token}_ep{int(episodes)}_noise{noise_token}"


def _validate_formal_eval_preconditions(run_dir: Path, checkpoint_path: Path, payload: Dict[str, Any], config: ExperimentConfig) -> None:
    """Validate the frozen final policy before an eval directory can exist."""
    if _require_checkpoint_in_run(checkpoint_path) != run_dir.resolve():
        raise RuntimeError("checkpoint does not belong to the resolved run")
    complete = _read_json_object(run_dir / "COMPLETE.json", "training completion marker")
    if complete.get("status") != "complete":
        raise RuntimeError("validation/final_release requires a complete training run")

    current = None
    provenance = None
    if config.is_formal_result:
        _require_formal_scientific_contract(config, "formal evaluation")
        current = _require_current_formal_provenance("formal evaluation")
        provenance = _read_json_object(run_dir / "provenance.json", "formal run provenance")
        _require_matching_formal_provenance(provenance, current, "formal run provenance")
        _require_matching_formal_provenance(payload, current, "formal checkpoint")
        _require_matching_formal_provenance(complete, current, "formal completion marker")

    run_config_data = _read_json_object(run_dir / "config.resolved.json", "training run config")
    try:
        run_config = config_from_dict(run_config_data)
        payload_config = config_from_dict(payload["config"])
    except Exception as exc:
        raise RuntimeError(f"training/checkpoint config is invalid: {exc}") from exc
    expected_hash = run_config.canonical_hash()
    if expected_hash != payload_config.canonical_hash() or payload.get("config_hash") != expected_hash:
        raise RuntimeError("checkpoint does not match this run's resolved config")
    if config.canonical_hash() != expected_hash:
        raise RuntimeError("resolved evaluation config does not match the training run")
    if payload.get("completed") is not True:
        raise RuntimeError("evaluation requires a checkpoint with completed=true")
    selected_best = checkpoint_path.name == "best.pt" and payload.get("checkpoint_role") == "best_selection_validation"
    if selected_best:
        selection = payload.get("selection_validation")
        if not isinstance(selection, dict):
            raise RuntimeError("selected best checkpoint is missing selection validation metadata")
        if payload.get("training_completed") is not True:
            raise RuntimeError("selected best checkpoint is not bound to a completed training run")
        if int(payload.get("selected_episode", -1)) != int(payload.get("episode", -2)):
            raise RuntimeError("selected best checkpoint episode metadata is inconsistent")
        if selection.get("criterion") != BEST_SELECTION_CRITERION:
            raise RuntimeError("selected best checkpoint criterion mismatch")
        if selection.get("seeds") != [int(seed) for seed in run_config.selection_validation_seeds]:
            raise RuntimeError("selected best checkpoint selection split mismatch")
        if int(payload.get("episode", -1)) < 1 or int(payload.get("episode", -1)) > int(run_config.episodes):
            raise RuntimeError("selected best checkpoint episode is outside training")
    elif int(payload.get("episode", -1)) != int(run_config.episodes):
        raise RuntimeError("evaluation latest checkpoint episode must equal config.episodes")
    if payload.get("semantic_version") != run_config.semantic_version:
        raise RuntimeError("checkpoint semantic_version mismatch")
    if payload.get("mobility_revision") != run_config.mobility_revision:
        raise RuntimeError("checkpoint mobility_revision mismatch")
    if payload.get("checkpoint_version") != 4 or payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("evaluation requires checkpoint_v4")

    if config.is_formal_result:
        if complete.get("config_hash") != expected_hash:
            raise RuntimeError("formal completion marker config_hash mismatch")
        if complete.get("semantic_version") != run_config.semantic_version or complete.get("mobility_revision") != run_config.mobility_revision:
            raise RuntimeError("formal completion marker semantic/mobility mismatch")
        if complete.get("checkpoint_completed") is not True or int(complete.get("final_episode", -1)) != int(run_config.episodes):
            raise RuntimeError("formal completion marker does not bind the final episode")
        checkpoint_hashes = complete.get("checkpoint_sha256")
        actual_hash = _sha256_file(checkpoint_path)
        if not isinstance(checkpoint_hashes, dict) or checkpoint_hashes.get(checkpoint_path.name) != actual_hash:
            raise RuntimeError("formal completion marker checkpoint hash mismatch")


def _mean_sd_ci95(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    means = values.mean(axis=1)
    if values.shape[1] > 1:
        sd = values.std(axis=1, ddof=1)
    else:
        sd = np.zeros(values.shape[0], dtype=np.float64)
    ci = 1.96 * sd / np.sqrt(max(1, values.shape[1]))
    return means, sd, ci


def evaluate_from_checkpoint(
    config: ExperimentConfig,
    checkpoint: str,
    eval_episodes: int,
    eval_seeds: Optional[List[int]] = None,
    eval_purpose: Optional[str] = None,
    scope: Optional[str] = None,
    eval_noise: float = 0.0,
    diagnostic_eval: bool = False,
) -> Dict[str, Any]:
    if config.algorithm == "mappo":
        raise ValueError("MAPPO held-out evaluation is deferred; the exploratory baseline writes policy_final.pt only")
    if eval_purpose not in EVAL_PURPOSE_SEEDS:
        raise ValueError("eval_purpose must be explicitly set to validation or final_test")
    expected_scope = SCOPE_FOR_PURPOSE[eval_purpose]
    if scope is None:
        scope = expected_scope
    if scope not in {"validation", "final_release"} or scope != expected_scope:
        raise ValueError(f"scope={scope!r} does not match eval_purpose={eval_purpose!r}")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    run_dir = _require_checkpoint_in_run(checkpoint_path)
    caller_rng = capture_rng_state()
    checkpoint_hash = _sha256_file(checkpoint_path)
    checkpoint_preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    requested_device = config.device
    training_config = config_from_dict(checkpoint_preview["config"])
    runtime_config = copy.deepcopy(training_config)
    if requested_device != "auto":
        runtime_config.device = requested_device
    config = training_config
    if eval_purpose == "final_test" and not (
        config.profile == REPRODUCTION_PROFILE and config.is_formal_result
    ):
        raise ValueError("final_test/final_release is reserved for a frozen reproduction baseline checkpoint")
    if eval_seeds is None:
        eval_seeds = list(EVAL_PURPOSE_SEEDS[eval_purpose])
    eval_seeds = [int(seed) for seed in eval_seeds]
    if not eval_seeds or len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError("eval_seeds must be non-empty and unique")
    if config.is_formal_result and eval_seeds != EVAL_PURPOSE_SEEDS[eval_purpose]:
        raise ValueError(f"formal {eval_purpose} evaluation requires seeds {EVAL_PURPOSE_SEEDS[eval_purpose]}")
    if int(eval_episodes) < 1:
        raise ValueError("eval_episodes must be positive")
    eval_noise = float(eval_noise)
    if not np.isfinite(eval_noise) or eval_noise < 0.0 or eval_noise > 1.0:
        raise ValueError("eval_noise must be finite and in [0, 1]")
    warmup_episodes = int(config.eval_warmup_episodes)
    if config.is_formal_result:
        if warmup_episodes != 5 or int(eval_episodes) != 100:
            raise ValueError("formal evaluation requires warmup=5 and scored episodes=100")
    eval_id = _eval_id(checkpoint_hash, eval_purpose, config.eval_protocol, warmup_episodes, eval_seeds, eval_episodes, eval_noise)
    eval_dir = run_dir / "eval" / eval_id
    if eval_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing eval directory: {eval_dir}")
    # Explicit diagnostic evaluations may compare more than one frozen
    # checkpoint at noise=0 without contending for a release lifecycle marker.
    # Training-time instrumentation alone does not change release semantics.
    is_diagnostic_eval = bool(diagnostic_eval) or eval_noise > 0.0
    marker_path = None if is_diagnostic_eval else run_dir / ("VALIDATION_READY.json" if scope == "validation" else "FINAL_RELEASE.json")
    if marker_path is not None and marker_path.exists():
        raise FileExistsError(f"Refusing to overwrite lifecycle marker: {marker_path}")
    _validate_formal_eval_preconditions(run_dir, checkpoint_path, checkpoint_preview, config)
    environment, agents, learner, replay, metrics, device = _make_system(runtime_config)
    payload = _load_checkpoint(checkpoint_path, config, agents, learner, replay, environment, metrics)
    eval_dir.mkdir(parents=True, exist_ok=False)
    for agent in agents:
        agent.actor.eval()
    raw_aoi = []
    raw_success = []
    raw_demand = []
    raw_v2i = []
    raw_v2v = []
    raw_interference = []
    raw_power = []
    raw_rb = []
    raw_mode = []
    raw_action = []
    raw_v2v_rate_all = []
    raw_selected_interference = []
    raw_interference_linear = []
    raw_v2i_interference_linear = []
    raw_v2v_interference_linear = []
    raw_I_v2i_linear = []
    raw_I_v2v_linear = []
    raw_I_mode_db = []
    try:
        for seed in eval_seeds:
            seed_aoi = []
            seed_success = []
            seed_demand = []
            seed_v2i = []
            seed_v2v = []
            seed_interference = []
            seed_power = []
            seed_rb = []
            seed_mode = []
            seed_action = []
            seed_v2v_rate_all = []
            seed_selected_interference = []
            seed_interference_linear = []
            seed_v2i_interference_linear = []
            seed_v2v_interference_linear = []
            seed_I_v2i_linear = []
            seed_I_v2v_linear = []
            seed_I_mode_db = []

            # One cold reset per held-out seed.  Warm-up and scored episodes
            # then advance the same world sequentially, preserving AoI,
            # previous interference, mobility, and slow fading history.
            environment.reset_world(seed)
            action_noise_rng = np.random.default_rng(np.random.SeedSequence([int(config.seed), int(seed), 0xA01]))
            for warmup_index in range(warmup_episodes):
                observations = environment.start_episode(warmup_index)
                for _step in range(config.steps_per_episode):
                    actions = np.asarray([agent.choose_action(observations[index], explore=False, noise_std=eval_noise, rng=action_noise_rng) for index, agent in enumerate(agents)], dtype=np.float32)
                    observations, _rg, _t1, _t2, _done, _info = environment.step(actions)

            for episode in range(int(eval_episodes)):
                episode_index = warmup_episodes + episode
                observations = environment.start_episode(episode_index)
                episode_aoi = []
                episode_success = []
                episode_demand = []
                episode_v2i = []
                episode_v2v = []
                episode_interference = []
                episode_power = []
                episode_rb = []
                episode_mode = []
                episode_action = []
                episode_v2v_rate_all = []
                episode_selected_interference = []
                episode_interference_linear = []
                episode_v2i_interference_linear = []
                episode_v2v_interference_linear = []
                episode_I_v2i_linear = []
                episode_I_v2v_linear = []
                episode_I_mode_db = []
                for _step in range(config.steps_per_episode):
                    actions = np.asarray([agent.choose_action(observations[index], explore=False, noise_std=eval_noise, rng=action_noise_rng) for index, agent in enumerate(agents)], dtype=np.float32)
                    observations, _rg, _t1, _t2, _done, info = environment.step(actions)
                    episode_action.append(actions.copy())
                    episode_aoi.append(info["aoi_ms"])
                    episode_success.append(info["success"])
                    episode_demand.append(info["remaining_demand"])
                    episode_v2i.append(info["v2i_rate"])
                    episode_v2v.append(info["v2v_rate"])
                    episode_interference.append(info["interference_db"])
                    episode_power.append(info["power_dbm"])
                    episode_rb.append(info["rb"])
                    episode_mode.append(info["mode"])
                    episode_v2v_rate_all.append(info.get("v2v_rate_all", np.zeros((config.number_agents, config.scenario.platoon_size - 1), dtype=np.float32)))
                    episode_selected_interference.append(info.get("selected_interference_db", info["interference_db"]))
                    episode_interference_linear.append(info.get("interference_linear", np.zeros((config.number_agents, config.n_rb), dtype=np.float32)))
                    episode_v2i_interference_linear.append(info.get("v2i_interference_linear", np.zeros((config.number_agents, config.n_rb), dtype=np.float32)))
                    episode_v2v_interference_linear.append(info.get("v2v_interference_linear", np.zeros((config.number_agents, config.scenario.platoon_size - 1, config.n_rb), dtype=np.float32)))
                    episode_I_v2i_linear.append(info.get("I_v2i_linear", episode_v2i_interference_linear[-1]))
                    episode_I_v2v_linear.append(info.get("I_v2v_linear", episode_v2v_interference_linear[-1]))
                    episode_I_mode_db.append(info.get("I_mode_db", episode_interference_linear[-1]))
                seed_aoi.append(episode_aoi)
                seed_success.append(episode_success)
                seed_demand.append(episode_demand)
                seed_v2i.append(episode_v2i)
                seed_v2v.append(episode_v2v)
                seed_interference.append(episode_interference)
                seed_power.append(episode_power)
                seed_rb.append(episode_rb)
                seed_mode.append(episode_mode)
                seed_action.append(episode_action)
                seed_v2v_rate_all.append(episode_v2v_rate_all)
                seed_selected_interference.append(episode_selected_interference)
                seed_interference_linear.append(episode_interference_linear)
                seed_v2i_interference_linear.append(episode_v2i_interference_linear)
                seed_v2v_interference_linear.append(episode_v2v_interference_linear)
                seed_I_v2i_linear.append(episode_I_v2i_linear)
                seed_I_v2v_linear.append(episode_I_v2v_linear)
                seed_I_mode_db.append(episode_I_mode_db)
            raw_aoi.append(seed_aoi)
            raw_success.append(seed_success)
            raw_demand.append(seed_demand)
            raw_v2i.append(seed_v2i)
            raw_v2v.append(seed_v2v)
            raw_interference.append(seed_interference)
            raw_power.append(seed_power)
            raw_rb.append(seed_rb)
            raw_mode.append(seed_mode)
            raw_action.append(seed_action)
            raw_v2v_rate_all.append(seed_v2v_rate_all)
            raw_selected_interference.append(seed_selected_interference)
            raw_interference_linear.append(seed_interference_linear)
            raw_v2i_interference_linear.append(seed_v2i_interference_linear)
            raw_v2v_interference_linear.append(seed_v2v_interference_linear)
            raw_I_v2i_linear.append(seed_I_v2i_linear)
            raw_I_v2v_linear.append(seed_I_v2v_linear)
            raw_I_mode_db.append(seed_I_mode_db)
    finally:
        # Evaluation uses private environment generators and must not perturb
        # the caller's training RNG streams.
        restore_rng_state(caller_rng)

    arrays = {
        "aoi_ms": np.asarray(raw_aoi, dtype=np.float32),
        "success": np.asarray(raw_success, dtype=np.float32),
        "remaining_demand": np.asarray(raw_demand, dtype=np.float32),
        "v2i_rate": np.asarray(raw_v2i, dtype=np.float32),
        "v2v_rate": np.asarray(raw_v2v, dtype=np.float32),
        "interference_db": np.asarray(raw_interference, dtype=np.float32),
        "power_dbm": np.asarray(raw_power, dtype=np.float32),
        "rb": np.asarray(raw_rb, dtype=np.int64),
        "mode": np.asarray(raw_mode, dtype=np.int64),
        "action_post_clip_normalized": np.asarray(raw_action, dtype=np.float32),
        "v2v_rate_all": np.asarray(raw_v2v_rate_all, dtype=np.float32),
        "selected_interference_db": np.asarray(raw_selected_interference, dtype=np.float32),
        "interference_linear": np.asarray(raw_interference_linear, dtype=np.float32),
        "v2i_interference_linear": np.asarray(raw_v2i_interference_linear, dtype=np.float32),
        "v2v_interference_linear": np.asarray(raw_v2v_interference_linear, dtype=np.float32),
        "I_v2i_linear": np.asarray(raw_I_v2i_linear, dtype=np.float32),
        "I_v2v_linear": np.asarray(raw_I_v2v_linear, dtype=np.float32),
        "I_mode_db": np.asarray(raw_I_mode_db, dtype=np.float32),
    }
    np.savez_compressed(eval_dir / "metrics.npz", **arrays)
    # Episodes are repeated frames within one held-out world, not independent
    # inferential units.  Keep their raw values and report only descriptive
    # within-seed SD.  Study-level CI is computed later across training runs.
    aoi_episode_seed_agent = arrays["aoi_ms"].mean(axis=2)  # seed x episode x agent
    endpoint_episode_seed_agent = arrays["success"][:, :, -1, :]
    per_seed_aoi_agent = aoi_episode_seed_agent.mean(axis=1)
    per_seed_success_agent = endpoint_episode_seed_agent.mean(axis=1)
    per_seed_aoi = per_seed_aoi_agent.mean(axis=1)
    per_seed_success = per_seed_success_agent.mean(axis=1)
    worst_agent_aoi_per_seed = per_seed_aoi_agent.max(axis=1)
    worst_agent_success_per_seed = per_seed_success_agent.min(axis=1)
    aggregate_aoi_agent = per_seed_aoi_agent.mean(axis=0)
    aggregate_success_agent = per_seed_success_agent.mean(axis=0)
    mode = arrays["mode"]
    rb = arrays["rb"]
    power = arrays["power_dbm"]
    action = arrays["action_post_clip_normalized"]
    mode_fraction_seed_agent = np.stack([(mode == value).mean(axis=(1, 2)) for value in range(config.n_modes)], axis=-1)
    rb_fraction_seed_agent = np.stack([(rb == value).mean(axis=(1, 2)) for value in range(config.n_rb)], axis=-1)
    mode_switch_seed_agent = (mode[:, :, 1:, :] != mode[:, :, :-1, :]).mean(axis=(1, 2)) if config.steps_per_episode > 1 else np.zeros((len(eval_seeds), config.number_agents))
    rb_switch_seed_agent = (rb[:, :, 1:, :] != rb[:, :, :-1, :]).mean(axis=(1, 2)) if config.steps_per_episode > 1 else np.zeros((len(eval_seeds), config.number_agents))
    power_tolerance = max((config.power_max_dbm - config.power_min_dbm) * 0.01, 1e-5)
    power_min_seed_agent = (power <= config.power_min_dbm + power_tolerance).mean(axis=(1, 2))
    power_max_seed_agent = (power >= config.power_max_dbm - power_tolerance).mean(axis=(1, 2))
    action_saturation_seed_agent_dim = (np.abs(action) >= 0.95).mean(axis=(1, 2))
    sd_aoi = aoi_episode_seed_agent.mean(axis=2).std(axis=1, ddof=1) if int(eval_episodes) > 1 else np.zeros(len(eval_seeds))
    sd_success = endpoint_episode_seed_agent.mean(axis=2).std(axis=1, ddof=1) if int(eval_episodes) > 1 else np.zeros(len(eval_seeds))
    checkpoint_reference = os.path.relpath(checkpoint_path, run_dir).replace(os.sep, "/")
    formal_eval = bool(config.is_formal_result and eval_purpose == "final_test" and not is_diagnostic_eval)
    eval_git = _git_metadata()
    summary = {
        "algorithm": config.algorithm,
        "eval_id": eval_id,
        "eval_purpose": eval_purpose,
        "scope": scope,
        "release_status": "diagnostic_evaluation" if is_diagnostic_eval else ("validation_ready" if scope == "validation" else ("final_release" if formal_eval else "evaluation_complete")),
        "diagnostic_evaluation": is_diagnostic_eval,
        "statistics_schema_version": EVAL_STATISTICS_SCHEMA_VERSION,
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "eval_episodes": int(eval_episodes),
        "eval_protocol": config.eval_protocol,
        "eval_warmup_episodes": warmup_episodes,
        "eval_noise": eval_noise,
        "semantic_version": config.semantic_version,
        "profile": config.profile,
        "scenario": config.scenario.id,
        "training_seed": int(config.seed),
        "config_hash": config.canonical_hash(),
        "training_device_config": config.device,
        "evaluation_device_requested": requested_device,
        "evaluation_device_resolved": str(device),
        "reproduction_git_commit": eval_git.get("reproduction_git_commit"),
        "reproduction_git_branch": eval_git.get("reproduction_git_branch"),
        "reproduction_git_dirty": eval_git.get("reproduction_git_dirty"),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": torch.version.cuda,
        "cuda_driver": eval_git.get("cuda_driver"),
        "gpu_names": eval_git.get("gpu_names", []),
        "global_reward_normalization": config.global_reward_normalization,
        "mobility_model": config.mobility_model,
        "mobility_revision": config.mobility_revision,
        "gap_definition": config.gap_definition,
        "vehicle_length_m": float(config.vehicle_length_m),
        "effective_center_spacing_m": float(config.effective_center_spacing_m),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_episode": int(payload.get("episode", -1)),
        "checkpoint_completed": bool(payload.get("completed", False)),
        "checkpoint": checkpoint_reference,
        "checkpoint_path_is_relative_to_run": True,
        "raw_metric_axes": {
            "aoi_ms": ["eval_seed", "scored_episode", "slot", "agent"],
            "success": ["eval_seed", "scored_episode", "slot", "agent"],
            "remaining_demand": ["eval_seed", "scored_episode", "slot", "agent"],
            "action_post_clip_normalized": ["eval_seed", "scored_episode", "slot", "agent", "action_dim"],
            "interference_db": ["eval_seed", "scored_episode", "slot", "agent", "rb"],
            "v2v_rate_all": ["eval_seed", "scored_episode", "slot", "agent", "follower"],
            "v2v_interference_linear": ["eval_seed", "scored_episode", "slot", "agent", "follower", "rb"],
        },
        "mean_AoI_ms_per_seed_agent": per_seed_aoi_agent.tolist(),
        "mean_AoI_ms_per_seed": per_seed_aoi.tolist(),
        "sd_AoI_ms_per_seed": sd_aoi.tolist(),
        "sd_AoI_ms_per_seed_semantics": "descriptive_across_scored_episodes_within_eval_seed",
        "ci95_AoI_ms_per_seed": [None for _ in eval_seeds],
        "ci95_AoI_ms_per_seed_semantics": "not_computed_episodes_are_not_independent_units",
        "CAM_success_probability_per_seed_agent": per_seed_success_agent.tolist(),
        "CAM_success_probability_per_seed": per_seed_success.tolist(),
        "sd_CAM_success_probability_per_seed": sd_success.tolist(),
        "sd_CAM_success_probability_per_seed_semantics": "descriptive_across_scored_episodes_within_eval_seed",
        "ci95_CAM_success_probability_per_seed": [None for _ in eval_seeds],
        "ci95_CAM_success_probability_per_seed_semantics": "not_computed_episodes_are_not_independent_units",
        "mean_AoI_ms": float(per_seed_aoi.mean()),
        "CAM_success_probability": float(per_seed_success.mean()),
        "endpoint_success_probability_per_seed": per_seed_success.tolist(),
        "worst_agent_mean_AoI_ms_per_seed": worst_agent_aoi_per_seed.tolist(),
        "worst_agent_CAM_success_probability_per_seed": worst_agent_success_per_seed.tolist(),
        "worst_agent_mean_AoI_ms": float(aggregate_aoi_agent.max()),
        "worst_agent_CAM_success_probability": float(aggregate_success_agent.min()),
        "action_diagnostics": {
            "mode_fraction_per_seed_agent": mode_fraction_seed_agent.tolist(),
            "rb_fraction_per_seed_agent": rb_fraction_seed_agent.tolist(),
            "mode_entropy_normalized_per_seed_agent": MetricStore._normalized_entropy(mode_fraction_seed_agent).tolist(),
            "rb_entropy_normalized_per_seed_agent": MetricStore._normalized_entropy(rb_fraction_seed_agent).tolist(),
            "mode_switch_rate_per_seed_agent": mode_switch_seed_agent.tolist(),
            "rb_switch_rate_per_seed_agent": rb_switch_seed_agent.tolist(),
            "action_post_clip_abs_ge_0p95_fraction_per_seed_agent_dim": action_saturation_seed_agent_dim.tolist(),
            "power_action_post_clip_near_min_fraction_per_seed_agent": (action[..., 2] <= -0.95).mean(axis=(1, 2)).tolist(),
            "power_action_post_clip_near_max_fraction_per_seed_agent": (action[..., 2] >= 0.95).mean(axis=(1, 2)).tolist(),
            "power_post_map_near_min_fraction_per_seed_agent": power_min_seed_agent.tolist(),
            "power_post_map_near_max_fraction_per_seed_agent": power_max_seed_agent.tolist(),
            "post_clip_saturation_threshold": 0.95,
            "post_map_near_bound_fraction_of_power_range": 0.01,
        },
        "training_seed_is_inferential_unit": True,
        "is_frozen_eval": True,
        "status": "complete",
        "is_formal_result": formal_eval,
    }
    eval_provenance = {
        **eval_git,
        "algorithm": config.algorithm,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_id": eval_id,
        "profile": config.profile,
        "semantic_version": config.semantic_version,
        "config_hash": config.canonical_hash(),
        "training_device_config": config.device,
        "evaluation_device_requested": requested_device,
        "evaluation_device_resolved": str(device),
        "scenario": config.scenario.id,
        "training_seed": int(config.seed),
        "eval_purpose": eval_purpose,
        "eval_noise": eval_noise,
        "statistics_schema_version": EVAL_STATISTICS_SCHEMA_VERSION,
        "mobility_revision": config.mobility_revision,
        "effective_center_spacing_m": float(config.effective_center_spacing_m),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_episode": int(payload.get("episode", -1)),
        "checkpoint_completed": bool(payload.get("completed", False)),
        "is_formal_result": formal_eval,
        "gap_definition": config.gap_definition,
        "vehicle_length_m": float(config.vehicle_length_m),
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_version": torch.version.cuda,
        "cuda_driver": eval_git.get("cuda_driver"),
        "gpu_names": eval_git.get("gpu_names", []),
        "scope": scope,
        "release_status": "diagnostic_evaluation" if is_diagnostic_eval else ("validation_ready" if scope == "validation" else ("final_release" if formal_eval else "evaluation_complete")),
        "diagnostic_evaluation": is_diagnostic_eval,
    }
    _write_json(eval_dir / "provenance.json", eval_provenance)
    _write_json(eval_dir / "summary.json", summary)
    _write_json(eval_dir / "EVAL_COMPLETE.json", summary)
    if marker_path is not None:
        _write_json(marker_path, {
            "algorithm": config.algorithm,
            "status": "validation_ready" if scope == "validation" else "final_release",
            "scope": scope,
            "eval_purpose": eval_purpose,
            "eval_id": eval_id,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_episode": int(payload.get("episode", -1)),
            "checkpoint_completed": bool(payload.get("completed", False)),
            "config_hash": config.canonical_hash(),
            "semantic_version": config.semantic_version,
            "mobility_revision": config.mobility_revision,
            "is_formal_result": formal_eval,
        })
    return {"eval_dir": str(eval_dir), **summary}
