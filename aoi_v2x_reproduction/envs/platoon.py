"""Reproduction-baseline platoon C-V2X environment.

The implementation keeps the path-loss equations and reward definitions from
the public source, while making the transition order explicit and exposing raw
per-RB metrics needed by evaluation and auditing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


def compute_global_reward(interference_linear, normalization_mode: str = "source_normalized_per_rb_mean"):
    """Compute the global interference reward from a linear P-by-K tensor.

    ``source_normalized_per_rb_mean`` is the source-compatible normalized
    mean over platoons and resource blocks.  ``eq16_sum`` is retained as an
    explicit diagnostic alternative and is never selected implicitly.
    """
    linear = np.asarray(interference_linear, dtype=np.float64)
    if linear.ndim != 2:
        raise ValueError(f"interference_linear must have shape [P,K], got {linear.shape}")
    if not np.all(np.isfinite(linear)) or np.any(linear <= 0):
        raise ValueError("interference_linear must be finite and strictly positive")
    interference_db = 10.0 * np.log10(linear)
    normalized = (interference_db + 60.0) / 60.0
    if normalization_mode == "source_normalized_per_rb_mean":
        return -float(np.mean(normalized))
    if normalization_mode == "eq16_sum":
        # Diagnostic-only Eq. (16): -(1/P) sum_j sum_k log10(I_jk).
        # The formal source-compatible reward above is intentionally unchanged.
        return -float(np.sum(np.log10(linear)) / linear.shape[0])
    if normalization_mode == "legacy_scalar":
        return -float(np.mean(normalized))
    raise ValueError(f"unsupported global reward normalization: {normalization_mode}")


def power_penalty(power_dbm: float) -> float:
    """Finite, non-negative, monotone power penalty used by paper rewards."""
    return 0.5 * math.log(max(float(power_dbm), 1.0), 5)


@dataclass
class Vehicle:
    position: List[float]
    direction: str
    velocity: float


class V2Vchannels:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2.0
        self.decorrelation_distance = 10.0

    def get_path_loss(self, position_a, position_b):
        d1 = abs(position_a[0] - position_b[0])
        d2 = abs(position_a[1] - position_b[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * 1e9 / 3e8

        def pl_los(distance):
            if distance <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            if distance < d_bp:
                return 22.7 * np.log10(distance) + 41 + 20 * np.log10(self.fc / 5)
            return 40.0 * np.log10(distance) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def pl_nlos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return pl_los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        return pl_los(d) if min(d1, d2) < 7 else min(pl_nlos(d1, d2), pl_nlos(d2, d1))

    def get_shadowing(self, delta_distance, shadowing):
        correlation = np.exp(-delta_distance / self.decorrelation_distance)
        return correlation * shadowing + math.sqrt(max(0.0, 1 - np.exp(-2 * delta_distance / self.decorrelation_distance))) * self.rng.normal(0, 3)


class V2Ichannels:
    def __init__(self, rng: np.random.Generator, bs_position):
        self.rng = rng
        self.h_bs = 25.0
        self.h_ms = 1.5
        self.decorrelation_distance = 50.0
        self.bs_position = list(bs_position)

    def get_path_loss(self, position):
        d1 = abs(position[0] - self.bs_position[0])
        d2 = abs(position[1] - self.bs_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)

    def get_shadowing(self, delta_distance, shadowing):
        correlation = np.exp(-np.asarray(delta_distance) / self.decorrelation_distance)
        innovation = self.rng.normal(0, 8, len(shadowing))
        return correlation * shadowing + np.sqrt(np.maximum(0.0, 1 - np.exp(-2 * np.asarray(delta_distance) / self.decorrelation_distance))) * innovation


class PaperEnviron:
    def __init__(self, config):
        self.config = config
        self.n_platoon = config.number_agents
        self.n_veh = config.number_vehicles
        self.n_rb = config.n_rb
        self.size_platoon = config.scenario.platoon_size
        self.width = float(config.map_width_m)
        self.height = float(config.map_height_m)
        self.bandwidth = int(config.bandwidth_hz)
        self.time_fast = config.slot_ms / 1000.0
        self.time_slow = config.slow_fading_ms / 1000.0
        self.v2v_demand_size = float(config.cam_bits)
        self.v2i_min = float(config.v2i_min_bps_per_hz * config.bandwidth_hz * self.time_fast)
        self.sig2_db = -114.0
        self.sig2 = 10 ** (self.sig2_db / 10.0)
        self.bs_ant_gain = 8.0
        self.bs_noise_figure = 5.0
        self.veh_ant_gain = 3.0
        self.veh_noise_figure = 9.0
        self.change_direction_prob = 0.4
        self.rng = np.random.default_rng(config.seed)
        self.v2v_channels = V2Vchannels(self.rng)
        self.v2i_channels = V2Ichannels(self.rng, config.rsu_position)
        self.vehicles: List[Vehicle] = []
        self.step_count = 0
        self.episode_index = 0
        self._world_initialized = False
        self._last_mobility_trace = []

    @property
    def state_dim(self):
        return self.config.state_dim

    def _lane_sets(self):
        up = [1.75, 5.25, 251.75, 255.25, 501.75, 505.25]
        down = [244.75, 248.25, 494.75, 498.25, 744.75, 748.25]
        left = [1.75, 5.25, 434.75, 438.25, 867.75, 871.25]
        right = [427.75, 431.25, 860.75, 864.25, 1293.75, 1297.25]
        return {"u": up, "d": down, "l": left, "r": right}

    @property
    def center_spacing_m(self) -> float:
        return float(self.config.effective_center_spacing_m)

    def _build_vehicles(self):
        lanes = self._lane_sets()
        graph = self._graph_bounds(lanes)
        self.vehicles = []
        directions = ["d", "u", "l", "r"]
        for platoon in range(self.n_platoon):
            direction = directions[platoon % len(directions)]
            lane_index = min(2 + 2 * (platoon // len(directions)), len(lanes[direction]) - 1)
            lane = lanes[direction][lane_index]
            velocity = float(self.rng.uniform(self.config.speed_min_mps, self.config.speed_max_mps))
            if direction in {"u", "d"}:
                if direction == "u":
                    low = graph["y_min"] + (self.size_platoon - 1) * self.center_spacing_m
                    high = graph["y_max"]
                else:
                    low = graph["y_min"]
                    high = graph["y_max"] - (self.size_platoon - 1) * self.center_spacing_m
                start = [lane, float(self.rng.uniform(low, high))]
            else:
                if direction == "r":
                    low = graph["x_min"] + (self.size_platoon - 1) * self.center_spacing_m
                    high = graph["x_max"]
                else:
                    low = graph["x_min"]
                    high = graph["x_max"] - (self.size_platoon - 1) * self.center_spacing_m
                start = [float(self.rng.uniform(low, high)), lane]
            for follower in range(self.size_platoon):
                offset = follower * self.center_spacing_m
                if direction == "u":
                    position = [start[0], start[1] - offset]
                elif direction == "d":
                    position = [start[0], start[1] + offset]
                elif direction == "l":
                    position = [start[0] + offset, start[1]]
                else:
                    position = [start[0] - offset, start[1]]
                self.vehicles.append(Vehicle(position, direction, velocity))

    @staticmethod
    def _graph_bounds(lanes):
        """Return the legal lane-graph segment bounds, not map projections."""
        return {
            "x_min": float(lanes["u"][0]),
            "x_max": float(lanes["d"][-1]),
            "y_min": float(lanes["l"][0]),
            "y_max": float(lanes["r"][-1]),
        }

    def _initialize_shadowing(self):
        """Draw initial shadowing once per cold world reset."""
        count = len(self.vehicles)
        self.v2v_shadowing = self.rng.normal(0, 3, (count, count))
        self.v2i_shadowing = self.rng.normal(0, 8, count)

    def initialize_shadowing(self):
        """Public semantic name for the cold-reset shadowing draw."""
        self._initialize_shadowing()

    def renew_slow_channel(self):
        """Correlate large-scale fading from the previous world state.

        This method never reinitializes shadowing.  A fresh random draw is
        reserved for :meth:`reset_world`; subsequent episode updates use the
        previous shadowing and the configured decorrelation distance.
        """
        count = len(self.vehicles)
        self.v2v_pathloss = np.zeros((count, count), dtype=np.float64) + 50 * np.identity(count)
        self.v2i_pathloss = np.zeros(count, dtype=np.float64)
        if not hasattr(self, "v2v_shadowing") or self.v2v_shadowing.shape != (count, count):
            self._initialize_shadowing()
        self.delta_distance = np.asarray([vehicle.velocity * self.time_slow for vehicle in self.vehicles])
        for i in range(count):
            for j in range(i + 1, count):
                shadow = self.v2v_channels.get_shadowing(self.delta_distance[i] + self.delta_distance[j], self.v2v_shadowing[i, j])
                self.v2v_shadowing[i, j] = self.v2v_shadowing[j, i] = shadow
                path = self.v2v_channels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position)
                self.v2v_pathloss[i, j] = self.v2v_pathloss[j, i] = path
        self.v2v_channels_abs = self.v2v_pathloss + self.v2v_shadowing
        self.v2i_shadowing = self.v2i_channels.get_shadowing(self.delta_distance, self.v2i_shadowing)
        for i, vehicle in enumerate(self.vehicles):
            self.v2i_pathloss[i] = self.v2i_channels.get_path_loss(vehicle.position)
        self.v2i_channels_abs = self.v2i_pathloss + self.v2i_shadowing

    def _renew_channel(self):
        """Backward-compatible private alias."""
        self.renew_slow_channel()

    def _renew_fast_fading(self):
        shape = (len(self.vehicles), len(self.vehicles), self.n_rb)
        fading = np.abs(self.rng.normal(0, 1, shape) + 1j * self.rng.normal(0, 1, shape)) / math.sqrt(2)
        self.v2v_channels_fast = np.repeat(self.v2v_channels_abs[:, :, None], self.n_rb, axis=2) - 20 * np.log10(fading)
        fading_i = np.abs(self.rng.normal(0, 1, (len(self.vehicles), self.n_rb)) + 1j * self.rng.normal(0, 1, (len(self.vehicles), self.n_rb))) / math.sqrt(2)
        self.v2i_channels_fast = np.repeat(self.v2i_channels_abs[:, None], self.n_rb, axis=1) - 20 * np.log10(fading_i)

    @staticmethod
    def _translate(x, y, direction, distance):
        if direction == "u":
            return x, y + distance
        if direction == "d":
            return x, y - distance
        if direction == "r":
            return x + distance, y
        return x - distance, y

    @staticmethod
    def _axis_value(x, y, direction):
        return y if direction in {"u", "d"} else x

    @staticmethod
    def _exit_spec(direction, lanes):
        return {
            "u": ("r", lanes["r"][-1]),
            "d": ("l", lanes["l"][0]),
            "l": ("u", lanes["u"][0]),
            "r": ("d", lanes["d"][-1]),
        }[direction]

    @staticmethod
    def _turn_targets(direction, lanes):
        if direction == "u":
            return [("l", value) for value in lanes["l"]] + [("r", value) for value in lanes["r"]]
        if direction == "d":
            return [("l", value) for value in lanes["l"]] + [("r", value) for value in lanes["r"]]
        if direction == "r":
            return [("u", value) for value in lanes["u"]] + [("d", value) for value in lanes["d"]]
        return [("u", value) for value in lanes["u"]] + [("d", value) for value in lanes["d"]]

    def _advance_leader_on_lane_graph(self, leader, distance, lanes):
        """Consume exactly ``distance`` along legal lane-graph segments."""
        remaining = float(distance)
        x, y = map(float, leader.position)
        initial = np.asarray([x, y], dtype=np.float64)
        events = []
        route_segments = []
        route_length = 0.0
        epsilon = 1e-12

        def consume(segment_distance):
            nonlocal x, y, route_length
            segment_distance = max(0.0, float(segment_distance))
            before = [float(x), float(y)]
            x, y = self._translate(x, y, leader.direction, segment_distance)
            after = [float(x), float(y)]
            actual = float(abs(after[0] - before[0]) + abs(after[1] - before[1]))
            route_length += actual
            route_segments.append({"from": before, "to": after, "distance_m": actual, "direction": leader.direction})
            return actual

        for _ in range(32):
            if remaining <= epsilon:
                break
            direction = leader.direction
            axis = self._axis_value(x, y, direction)
            sign = 1.0 if direction in {"u", "r"} else -1.0
            exit_direction, exit_axis = self._exit_spec(direction, lanes)
            exit_distance = (float(exit_axis) - axis) * sign

            # A manually injected pre-exit point is defensively re-anchored to
            # the lane graph.  Normal paper trajectories never enter here
            # because _build_vehicles samples only legal lane segments.
            if exit_distance < -epsilon:
                x, y = self._translate(x, y, direction, exit_distance)
                axis = float(exit_axis)
                exit_distance = 0.0
                events.append({"type": "defensive_exit_reanchor", "direction": direction})

            next_turn = None
            for target_direction, crossing in self._turn_targets(direction, lanes):
                turn_distance = (float(crossing) - axis) * sign
                if turn_distance <= epsilon or turn_distance > remaining + epsilon:
                    continue
                if next_turn is None or turn_distance < next_turn[0]:
                    next_turn = (turn_distance, target_direction, float(crossing))

            if exit_distance <= remaining + epsilon and (next_turn is None or exit_distance <= next_turn[0] + epsilon):
                consumed = consume(max(0.0, exit_distance))
                remaining -= consumed
                leader.direction = exit_direction
                events.append({"type": "exit", "from": direction, "to": exit_direction, "axis": float(exit_axis)})
                continue

            if next_turn is not None:
                turn_distance, target_direction, crossing = next_turn
                consumed = consume(turn_distance)
                remaining -= consumed
                if self.rng.uniform(0.0, 1.0) < self.change_direction_prob:
                    leader.direction = target_direction
                    events.append({"type": "turn", "from": direction, "to": target_direction, "axis": crossing})
                else:
                    events.append({"type": "straight_through_intersection", "direction": direction, "axis": crossing})
                continue

            remaining -= consume(remaining)

        if remaining > epsilon:
            raise RuntimeError("lane graph mobility could not consume the full route distance")
        if not np.isclose(route_length, float(distance), rtol=0.0, atol=1e-9):
            raise RuntimeError(f"lane graph route length mismatch: {route_length} != {distance}")
        leader.position = [float(x), float(y)]
        final = np.asarray(leader.position, dtype=np.float64)
        return {
            "before": initial.tolist(),
            "after": final.tolist(),
            "path_length_m": float(distance),
            "route_length_m": float(route_length),
            "endpoint_manhattan_m": float(np.abs(final - initial).sum()),
            "direction": leader.direction,
            "events": events,
            "route_segments": route_segments,
        }

    def _renew_positions(self):
        """Advance platoons by lane-graph path consumption without projection."""
        lanes = self._lane_sets()
        traces = []
        for platoon in range(self.n_platoon):
            leader_index = platoon * self.size_platoon
            leader = self.vehicles[leader_index]
            distance = float(leader.velocity * self.time_slow)
            trace = self._advance_leader_on_lane_graph(leader, distance, lanes)
            trace["platoon"] = platoon
            traces.append(trace)
            x, y = leader.position
            for follower in range(1, self.size_platoon):
                if leader.direction == "u":
                    position = [x, y - follower * self.center_spacing_m]
                elif leader.direction == "d":
                    position = [x, y + follower * self.center_spacing_m]
                elif leader.direction == "l":
                    position = [x + follower * self.center_spacing_m, y]
                else:
                    position = [x - follower * self.center_spacing_m, y]
                self.vehicles[leader_index + follower].direction = leader.direction
                self.vehicles[leader_index + follower].position = position
                self.vehicles[leader_index + follower].velocity = leader.velocity
        self._last_mobility_trace = traces
        return traces

    def _build_episode_world(self):
        self._build_vehicles()
        self.initialize_shadowing()
        self.renew_slow_channel()
        self._renew_fast_fading()
        self.v2v_demand = np.full(self.n_platoon, self.v2v_demand_size, dtype=np.float64)
        self.individual_time_limit = np.full(self.n_platoon, self.config.steps_per_episode * self.time_fast, dtype=np.float64)
        self.active_links = np.ones(self.n_platoon, dtype=bool)
        self.aoi = np.full(self.n_platoon, self.config.initial_aoi_ms, dtype=np.float64)
        self.previous_interference = np.full((self.n_platoon, self.n_rb), self.sig2_db, dtype=np.float64)
        self.step_count = 0
        self._last_mobility_trace = []

    def reset_world(self, seed: Optional[int] = None):
        """Cold-reset the world and its private RNG stream.

        A world reset is deliberately separate from an episode start.  It is
        the only operation that rebuilds vehicles, channels, AoI, and the
        initial interference history.
        """
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
            self.v2v_channels.rng = self.rng
            self.v2i_channels.rng = self.rng
        self._build_episode_world()
        self.episode_index = 0
        self._world_initialized = True
        return self.get_observations()

    def reset(self, seed: Optional[int] = None):
        """Backward-compatible alias for :meth:`reset_world`."""
        return self.reset_world(seed)

    def start_episode(self, episode_index: int, update_mobility: bool = True):
        """Start an episode without resetting persistent AoI/interference.

        Payload, deadline, and link activity are episode-local.  Vehicle
        positions, slow fading, AoI, and previous interference belong to the
        continuing world and are preserved unless the configured slow-update
        boundary advances the mobility/channel state.
        """
        if not self._world_initialized:
            self.reset_world(self.config.seed)
        episode_index = int(episode_index)
        if episode_index > 0 and update_mobility and episode_index % self.config.slow_update_every_episodes == 0:
            self._renew_positions()
            self.renew_slow_channel()
        if episode_index > 0:
            self._renew_fast_fading()
        self.v2v_demand.fill(self.v2v_demand_size)
        self.individual_time_limit.fill(self.config.steps_per_episode * self.time_fast)
        self.active_links[:] = True
        self.episode_index = episode_index
        self.step_count = 0
        return self.get_observations()

    def reset_episode(self, episode_index: int):
        """Compatibility alias retained for older callers."""
        return self.start_episode(episode_index)

    def decode_actions(self, actions):
        raw = np.asarray(actions, dtype=np.float64).reshape(self.n_platoon, 3)
        normalized = np.clip(raw, -1.0, 1.0)
        rb = np.minimum(self.n_rb - 1, np.floor((normalized[:, 0] + 1.0) * 0.5 * self.n_rb)).astype(np.int64)
        mode = np.minimum(self.config.n_modes - 1, np.floor((normalized[:, 1] + 1.0) * 0.5 * self.config.n_modes)).astype(np.int64)
        power_fraction = np.clip((normalized[:, 2] + 1.0) * 0.5, 0.0, 1.0)
        if self.config.power_continuous:
            power = self.config.power_min_dbm + power_fraction * (self.config.power_max_dbm - self.config.power_min_dbm)
        else:
            power = np.round(np.clip(power_fraction * self.config.power_max_dbm, 1.0, self.config.power_max_dbm))
        return np.column_stack((rb, mode, power)).astype(np.float64)

    def _compute_metrics(self, decoded):
        p = self.n_platoon
        n = self.size_platoon
        rb_indices = decoded[:, 0].astype(np.int64)
        modes = decoded[:, 1].astype(np.int64)
        self.v2i_interference = np.full((p, self.n_rb), self.sig2, dtype=np.float64)
        self.v2v_interference = np.full((p, n - 1, self.n_rb), self.sig2, dtype=np.float64)
        # Every victim and every RB receives contributions from all other
        # transmitters using that RB.  This deliberately includes RBs that no
        # victim selected, which must not silently remain at noise only.
        for receiver in range(p):
            receiver_start = receiver * n
            for rb in range(self.n_rb):
                for transmitter in np.flatnonzero(rb_indices == rb):
                    if transmitter == receiver:
                        continue
                    leader_tx = transmitter * n
                    self.v2i_interference[receiver, rb] += 10 ** (
                        (decoded[transmitter, 2] - self.v2i_channels_fast[leader_tx, rb] + self.veh_ant_gain + self.bs_ant_gain - self.bs_noise_figure) / 10.0
                    )
                    for follower in range(n - 1):
                        self.v2v_interference[receiver, follower, rb] += 10 ** (
                            (decoded[transmitter, 2] - self.v2v_channels_fast[leader_tx, receiver_start + follower + 1, rb] + 2 * self.veh_ant_gain - self.veh_noise_figure) / 10.0
                        )
        v2i_signal = np.zeros(p, dtype=np.float64)
        v2v_signal = np.zeros((p, n - 1), dtype=np.float64)
        for receiver in range(p):
            rb = int(rb_indices[receiver])
            leader = receiver * n
            if modes[receiver] == 0:
                v2i_signal[receiver] = 10 ** (
                    (decoded[receiver, 2] - self.v2i_channels_fast[leader, rb] + self.veh_ant_gain + self.bs_ant_gain - self.bs_noise_figure) / 10.0
                )
            else:
                for follower in range(n - 1):
                    v2v_signal[receiver, follower] = 10 ** (
                        (decoded[receiver, 2] - self.v2v_channels_fast[leader, leader + follower + 1, rb] + 2 * self.veh_ant_gain - self.veh_noise_figure) / 10.0
                    )
        v2i_selected_interference = self.v2i_interference[np.arange(p), rb_indices]
        v2v_selected_interference = self.v2v_interference[np.arange(p), :, rb_indices]
        v2i_rate = np.log2(1.0 + v2i_signal / v2i_selected_interference) * self.time_fast * self.bandwidth
        v2v_rate_all = np.log2(1.0 + v2v_signal / v2v_selected_interference) * self.time_fast * self.bandwidth
        v2v_rate = v2v_rate_all.min(axis=1)
        interference_linear = np.where(modes[:, None] == 0, self.v2i_interference, np.maximum(self.v2v_interference.max(axis=1), self.sig2))
        interference_db = 10.0 * np.log10(np.maximum(interference_linear, np.finfo(np.float64).tiny))
        selected_interference = interference_db[np.arange(p), rb_indices]
        self.I_v2i_linear = self.v2i_interference.copy()
        self.I_v2v_linear = self.v2v_interference.copy()
        self.I_mode_db = interference_db.copy()
        return {
            "v2i_rate": v2i_rate,
            "v2v_rate": v2v_rate,
            "v2v_rate_all": v2v_rate_all,
            "interference_db": interference_db,
            "selected_interference_db": selected_interference,
            "interference_linear": interference_linear,
            "v2i_interference_linear": self.v2i_interference.copy(),
            "v2v_interference_linear": self.v2v_interference.copy(),
            "I_v2i_linear": self.I_v2i_linear,
            "I_v2v_linear": self.I_v2v_linear,
            "I_mode_db": self.I_mode_db,
        }

    def _state_for_agent(self, idx: int):
        n = self.size_platoon
        leader = idx * n
        followers = leader + 1 + np.arange(n - 1)
        v2i_abs = (self.v2i_channels_abs[leader] - 60.0) / 60.0
        v2i_fast = (self.v2i_channels_fast[leader, :] - self.v2i_channels_abs[leader] + 10.0) / 35.0
        v2v_abs = (self.v2v_channels_abs[leader, followers] - 60.0) / 60.0
        v2v_fast = (self.v2v_channels_fast[leader, followers, :] - self.v2v_channels_abs[leader, followers, None] + 10.0) / 35.0
        if self.config.previous_interference_dim == self.n_rb:
            interference = (self.previous_interference[idx] + 60.0) / 60.0
        else:
            interference = np.asarray([(self.previous_interference[idx].mean() + 60.0) / 60.0])
        aoi = np.asarray([self.aoi[idx] / self.config.steps_per_episode])
        demand = np.asarray([self.v2v_demand[idx] / self.v2v_demand_size])
        values = [v2i_abs.reshape(-1), v2i_fast.reshape(-1), v2v_abs.reshape(-1), v2v_fast.reshape(-1), interference.reshape(-1), aoi, demand]
        if self.config.include_remaining_time:
            values.append(np.asarray([max(0.0, self.individual_time_limit[idx]) / (self.config.steps_per_episode * self.time_fast)]))
        return np.concatenate(values).astype(np.float32)

    def get_observations(self):
        return np.asarray([self._state_for_agent(i) for i in range(self.n_platoon)], dtype=np.float32)

    def step(self, actions):
        decoded = self.decode_actions(actions)
        metrics = self._compute_metrics(decoded)
        v2i_rate = metrics["v2i_rate"]
        v2v_rate = metrics["v2v_rate"]
        v2v_rate_all = metrics["v2v_rate_all"]
        interference_db = metrics["interference_db"]
        selected_interference = metrics["selected_interference_db"]
        for i in range(self.n_platoon):
            self.aoi[i] = 1.0 if v2i_rate[i] >= self.v2i_min else min(float(self.config.steps_per_episode), self.aoi[i] + 1.0)
        self.v2v_demand = np.maximum(0.0, self.v2v_demand - v2v_rate)
        self.individual_time_limit = np.maximum(0.0, self.individual_time_limit - self.time_fast)
        self.active_links = self.v2v_demand > 0
        success = ~self.active_links
        task1 = np.empty(self.n_platoon, dtype=np.float64)
        task2 = np.empty(self.n_platoon, dtype=np.float64)
        for i in range(self.n_platoon):
            power_cost = power_penalty(decoded[i, 2])
            revenue = 1.0 if v2i_rate[i] >= self.v2i_min else 0.0
            if int(decoded[i, 1]) == 0:
                task1[i] = -4.95 * (self.v2v_demand[i] / self.v2v_demand_size)
                task2[i] = 0.05 * revenue - power_cost - self.aoi[i] / 20.0
            else:
                task1[i] = -4.95 * (self.v2v_demand[i] / self.v2v_demand_size) - power_cost
                task2[i] = 0.05 * revenue - self.aoi[i] / 20.0
        global_reward = compute_global_reward(metrics["interference_linear"], self.config.global_reward_normalization)
        self.previous_interference = interference_db.copy()
        self._renew_fast_fading()
        self.step_count += 1
        terminated = self.step_count >= self.config.steps_per_episode
        info: Dict[str, Any] = {
            "actions_decoded": decoded.astype(np.float32),
            "rb": decoded[:, 0].astype(np.int64),
            "mode": decoded[:, 1].astype(np.int64),
            "power_dbm": decoded[:, 2].astype(np.float32),
            "aoi_ms": self.aoi.astype(np.float32).copy(),
            "remaining_demand": self.v2v_demand.astype(np.float32).copy(),
            "remaining_time_ms": (self.individual_time_limit * 1000.0).astype(np.float32).copy(),
            "v2i_rate": v2i_rate.astype(np.float32),
            "v2v_rate": v2v_rate.astype(np.float32),
            "v2v_rate_all": v2v_rate_all.astype(np.float32),
            "interference_db": interference_db.astype(np.float32),
            "selected_interference_db": selected_interference.astype(np.float32),
            "interference_linear": metrics["interference_linear"].astype(np.float32),
            "v2i_interference_linear": metrics["v2i_interference_linear"].astype(np.float32),
            "v2v_interference_linear": metrics["v2v_interference_linear"].astype(np.float32),
            "I_v2i_linear": metrics["I_v2i_linear"].astype(np.float32),
            "I_v2v_linear": metrics["I_v2v_linear"].astype(np.float32),
            "I_mode_db": metrics["I_mode_db"].astype(np.float32),
            "global_reward_normalization": self.config.global_reward_normalization,
            "success": success.astype(np.float32),
        }
        return self.get_observations(), global_reward, task1.astype(np.float32), task2.astype(np.float32), terminated, info

    def state_dict(self):
        return {
            "vehicles": [(list(v.position), v.direction, v.velocity) for v in self.vehicles],
            "v2v_pathloss": self.v2v_pathloss.copy(),
            "v2i_pathloss": self.v2i_pathloss.copy(),
            "v2v_shadowing": self.v2v_shadowing.copy(),
            "v2i_shadowing": self.v2i_shadowing.copy(),
            "delta_distance": self.delta_distance.copy(),
            "v2v_channels_abs": self.v2v_channels_abs.copy(),
            "v2i_channels_abs": self.v2i_channels_abs.copy(),
            "v2v_channels_fast": self.v2v_channels_fast.copy(),
            "v2i_channels_fast": self.v2i_channels_fast.copy(),
            "v2v_demand": self.v2v_demand.copy(),
            "individual_time_limit": self.individual_time_limit.copy(),
            "active_links": self.active_links.copy(),
            "aoi": self.aoi.copy(),
            "previous_interference": self.previous_interference.copy(),
            "step_count": self.step_count,
            "episode_index": self.episode_index,
            "world_initialized": self._world_initialized,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        self.vehicles = [Vehicle(list(pos), direction, velocity) for pos, direction, velocity in state["vehicles"]]
        for key in ["v2v_pathloss", "v2i_pathloss", "v2v_shadowing", "v2i_shadowing", "delta_distance", "v2v_channels_abs", "v2i_channels_abs", "v2v_channels_fast", "v2i_channels_fast", "v2v_demand", "individual_time_limit", "active_links", "aoi", "previous_interference"]:
            setattr(self, key, np.asarray(state[key]).copy())
        self.step_count = int(state["step_count"])
        self.episode_index = int(state["episode_index"])
        self._world_initialized = bool(state.get("world_initialized", True))
        self.rng.bit_generator.state = state["rng_state"]
        self.v2v_channels.rng = self.rng
        self.v2i_channels.rng = self.rng


def make_paper_environment(config):
    return PaperEnviron(config)
