"""Encode native hybrid MAPPO actions for the unchanged environment codec."""

from __future__ import annotations

import numpy as np


def encode_hybrid_actions(rb, mode, power_unit, n_rb: int, n_modes: int) -> np.ndarray:
    """Return the normalized ``[agent, 3]`` action expected by the environment.

    Discrete indices are encoded at bin centers so the existing floor-based
    decoder maps them back exactly. ``power_unit`` is the native Beta action in
    the open unit interval and is mapped linearly to the environment's third
    normalized component.
    """

    rb_array = np.asarray(rb)
    mode_array = np.asarray(mode)
    power_array = np.asarray(power_unit, dtype=np.float64)
    if rb_array.ndim != 1 or mode_array.shape != rb_array.shape or power_array.shape != rb_array.shape:
        raise ValueError("rb, mode, and power_unit must be one-dimensional arrays with the same shape")
    if not np.issubdtype(rb_array.dtype, np.integer) or not np.issubdtype(mode_array.dtype, np.integer):
        raise TypeError("rb and mode actions must be integer indices")
    if int(n_rb) < 1 or int(n_modes) < 2:
        raise ValueError("invalid discrete action cardinality")
    if np.any(rb_array < 0) or np.any(rb_array >= int(n_rb)):
        raise ValueError("RB action is out of range")
    if np.any(mode_array < 0) or np.any(mode_array >= int(n_modes)):
        raise ValueError("mode action is out of range")
    if not np.all(np.isfinite(power_array)) or np.any(power_array <= 0.0) or np.any(power_array >= 1.0):
        raise ValueError("power_unit must be finite and strictly inside (0, 1)")

    rb_normalized = -1.0 + 2.0 * (rb_array.astype(np.float64) + 0.5) / float(n_rb)
    mode_normalized = -1.0 + 2.0 * (mode_array.astype(np.float64) + 0.5) / float(n_modes)
    power_normalized = 2.0 * power_array - 1.0
    return np.column_stack((rb_normalized, mode_normalized, power_normalized)).astype(np.float32)
