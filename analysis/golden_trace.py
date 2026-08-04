"""Compare one deterministic legacy adapter trace with the copied source."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Classes.legacy_adapter import LegacyEnviron
from config import resolve_config
from tests.test_golden_trace import _original_trace


def _max_abs(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    try:
        left, right = np.broadcast_arrays(left, right)
    except ValueError:
        return {"shape_left": list(left.shape), "shape_right": list(right.shape), "max_abs": float("inf")}
    return {"shape": list(left.shape), "max_abs": float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0}


def main() -> int:
    config = resolve_config("legacy_release", "p05_n04_g25", seed=23, episodes=2, steps_per_episode=3)
    actions = np.asarray([
        [-0.8, -0.7, -0.5], [-0.2, 0.4, 0.1], [0.2, -0.3, 0.8],
        [0.7, 0.8, -0.1], [-0.4, 0.1, 0.3],
    ], dtype=np.float32)
    original_env, _decoded, original = _original_trace(config, actions)
    np.random.seed(config.seed)
    adapter = LegacyEnviron(config)
    adapter.reset_episode(0)
    reproduced = adapter.step(actions)
    original_state = np.asarray([
        np.concatenate((
            np.reshape((original_env.V2I_channels_abs[index * config.scenario.platoon_size] - 60) / 60.0, -1),
            np.reshape((original_env.V2I_channels_with_fastfading[index * config.scenario.platoon_size, :] - original_env.V2I_channels_abs[index * config.scenario.platoon_size] + 10) / 35, -1),
            np.reshape((original_env.V2V_channels_abs[index * config.scenario.platoon_size, index * config.scenario.platoon_size + 1 + np.arange(config.scenario.platoon_size - 1)] - 60) / 60.0, -1),
            np.reshape((original_env.V2V_channels_with_fastfading[index * config.scenario.platoon_size, index * config.scenario.platoon_size + 1 + np.arange(config.scenario.platoon_size - 1), :] - original_env.V2V_channels_abs[index * config.scenario.platoon_size, index * config.scenario.platoon_size + 1 + np.arange(config.scenario.platoon_size - 1)].reshape(config.scenario.platoon_size - 1, 1) + 10) / 35, -1),
            np.reshape((original_env.Interference_all[index] + 60) / 60.0, -1),
            np.reshape(original_env.AoI[index] / int(original_env.time_slow / original_env.time_fast), -1),
            np.asarray([original_env.V2V_demand[index] / original_env.V2V_demand_size]),
        )) for index in range(config.number_agents)
    ])
    checks = {
        "state": _max_abs(original_state, reproduced[0]),
        "task1": _max_abs(original[0], reproduced[2]),
        "task2": _max_abs(original[1], reproduced[3]),
        "global_reward": _max_abs(original[2], reproduced[1]),
        "aoi": _max_abs(original[3], reproduced[5]["aoi_ms"]),
        "v2i_rate": _max_abs(original[4], reproduced[5]["v2i_rate"]),
        "v2v_rate": _max_abs(original[5], reproduced[5]["v2v_rate"]),
        "demand": _max_abs(original[6], reproduced[5]["remaining_demand"]),
        "success": _max_abs(original[7], reproduced[5]["success"]),
        "interference": _max_abs(original_env.Interference_all, adapter.Interference_all),
    }
    passed = all(value["max_abs"] == 0.0 for value in checks.values())
    result = {
        "status": "pass" if passed else "fail",
        "profile": "legacy_release",
        "seed": config.seed,
        "semantic_version": config.semantic_version,
        "action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "checks": checks,
    }
    output = ROOT / "audit" / "golden_trace.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
