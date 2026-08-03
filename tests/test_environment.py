import numpy as np
import pytest

from Classes.Environment_Platoon import PaperEnviron
from config import resolve_config


def _paper_config(**overrides):
    values = {"seed": 13, "episodes": 2, "steps_per_episode": 5}
    values.update(overrides)
    return resolve_config("paper_faithful", "p05_n04_g25", **values)


def test_paper_state_time_power_and_geometry():
    config = _paper_config()
    env = PaperEnviron(config)
    observations = env.reset(13)
    assert observations.shape == (5, 22)
    assert env.v2i_channels.bs_position == [375.0, 649.5]
    assert all(10.0 <= vehicle.velocity <= 15.0 for vehicle in env.vehicles)
    decoded = env.decode_actions(np.zeros((5, 3), dtype=np.float32))
    assert np.all(decoded[:, 0] >= 0) and np.all(decoded[:, 0] < 3)
    assert np.all(decoded[:, 1] >= 0) and np.all(decoded[:, 1] < 2)
    assert np.all(decoded[:, 2] >= 1) and np.all(decoded[:, 2] <= 30)
    assert np.any(np.abs(decoded[:, 2] - np.round(decoded[:, 2])) > 1e-6)
    assert observations[0, -1] == pytest.approx(1.0)
    next_observations, _rg, _t1, _t2, _done, info = env.step(np.zeros((5, 3), dtype=np.float32))
    assert next_observations.shape == observations.shape
    assert info["interference_db"].shape == (5, 3)
    assert next_observations[0, -1] == pytest.approx(0.8, abs=1e-6)


def test_current_action_changes_current_interference_reward():
    config = _paper_config()
    first = PaperEnviron(config)
    second = PaperEnviron(config)
    first.reset(99)
    second.reset(99)
    low_power = np.full((5, 3), -1.0, dtype=np.float32)
    high_power = low_power.copy()
    high_power[1, 2] = 1.0
    result_low = first.step(low_power)
    result_high = second.step(high_power)
    assert not np.allclose(result_low[5]["interference_db"], result_high[5]["interference_db"])
    assert result_low[1] != pytest.approx(result_high[1])


def test_remaining_time_reaches_zero_at_deadline():
    config = _paper_config(steps_per_episode=3)
    env = PaperEnviron(config)
    env.reset(7)
    for step in range(3):
        _obs, _rg, _t1, _t2, terminated, info = env.step(np.zeros((5, 3), dtype=np.float32))
        assert info["remaining_time_ms"].min() == pytest.approx((2 - step) * 1.0)
    assert terminated is True
