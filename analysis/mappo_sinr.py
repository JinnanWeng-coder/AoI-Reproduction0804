"""Read-only pre-fading SINR instrumentation; no change to environment transitions or actions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from aoi_v2x_reproduction.envs.platoon import PaperEnviron


class SinrEnvironment(PaperEnviron):
    def _compute_metrics(self, decoded):
        metrics = super()._compute_metrics(decoded)  # Exactly ONE normal physics computation.
        # step() renews fading BEFORE returning info. Capture executed-slot S/I here,
        # not by reading the next slot's channels after step(). No RNG or state updates.
        p, n = self.n_platoon, self.size_platoon
        rb = decoded[:, 0].astype(int)
        mode = decoded[:, 1].astype(int)
        leaders = np.arange(p) * n
        i_i = metrics["v2i_interference_linear"][np.arange(p), rb]
        i_v = metrics["v2v_interference_linear"][np.arange(p), :, rb]
        s_i = np.power(10., (decoded[:, 2] - self.v2i_channels_fast[leaders, rb] +
                            self.veh_ant_gain + self.bs_ant_gain - self.bs_noise_figure) / 10)
        followers = leaders[:, None] + 1 + np.arange(n - 1)[None, :]
        losses = self.v2v_channels_fast[leaders[:, None], followers, rb[:, None]]
        s_v = np.power(10., (decoded[:, 2, None] - losses + 2*self.veh_ant_gain - self.veh_noise_figure) / 10)
        gamma_i, gamma_v = s_i / i_i, s_v / i_v
        for values in (s_i, i_i, s_v, i_v, gamma_i, gamma_v):
            if not np.isfinite(values).all() or np.any(values <= 0):
                raise FloatingPointError("nonpositive/nonfinite active-slot SINR components")
        scale = self.time_fast * self.bandwidth
        if not (np.allclose(np.log2(1 + gamma_i[mode == 0])*scale, metrics["v2i_rate"][mode == 0], rtol=1e-12, atol=1e-12) and
                np.allclose(np.log2(1 + gamma_v[mode == 1])*scale, metrics["v2v_rate_all"][mode == 1], rtol=1e-12, atol=1e-12)):
            raise ValueError("SINR and native Shannon bits/slot disagree")
        self.sinr_last = {"rb": rb.copy(), "mode": mode.copy(), "power_dbm": decoded[:, 2].copy(),
            "v2i_signal_mw": s_i, "v2i_interference_noise_mw": i_i.copy(), "v2i_sinr_linear": gamma_i,
            "v2v_signal_mw": s_v, "v2v_interference_noise_mw": i_v.copy(), "v2v_sinr_linear": gamma_v,
            "rate_v2i_bits": metrics["v2i_rate"].copy(), "rate_v2v_bits": metrics["v2v_rate_all"].copy()}
        return metrics


class SinrLogger:
    def __init__(self, directory, config, console_episodes=1):
        if type(console_episodes) is not int or console_episodes < 0:
            raise ValueError("sinr_console_episodes must be a nonnegative integer")
        self.root = Path(directory)
        self.config = config
        self.console_episodes = console_episodes
        self.csv = (self.root / "sinr_slots.csv").open("x", newline="", encoding="utf-8")
        self.log = (self.root / "sinr.log").open("x", encoding="utf-8")
        self.writer = None
        self.counts = {"warmup": 0, "scored": 0}
        self.values = {"v2i_active": [], "v2v_receiver_active": [], "v2v_bottleneck_active": []}
        self.log.write("SINR measured before next-slot fading renewal; linear=S/(I+N), dB=10log10(linear). Inactive mode=NA, NOT0dB. Indices: episode/slot1-based,platoon/RB0-based,follower1..3.\n")

    def record(self, environment, world, phase, episode, slot):
        v = environment.sinr_last
        for p in range(self.config.number_agents):
            is_i = int(v["mode"][p]) == 0
            row = {"world": int(world), "phase": phase, "episode": int(episode), "slot": int(slot),
                "platoon": p, "leader_vehicle_index": p*self.config.scenario.platoon_size,
                "rb": int(v["rb"][p]), "mode": "V2I" if is_i else "V2V", "power_dbm": float(v["power_dbm"][p])}
            for key in ("v2i_signal_mw", "v2i_interference_noise_mw", "v2i_sinr_linear"):
                row[key] = float(v[key][p]) if is_i else None
            row["v2i_sinr_db"] = float(10*np.log10(v["v2i_sinr_linear"][p])) if is_i else None
            for f in range(self.config.scenario.platoon_size - 1):
                for key in ("v2v_signal_mw", "v2v_interference_noise_mw", "v2v_sinr_linear"):
                    row[f"{key}_follower{f+1}"] = None if is_i else float(v[key][p, f])
                row[f"v2v_sinr_db_follower{f+1}"] = None if is_i else float(10*np.log10(v["v2v_sinr_linear"][p, f]))
            worst = None if is_i else float(v["v2v_sinr_linear"][p].min())
            row["v2v_bottleneck_sinr_linear"] = worst
            row["v2v_bottleneck_sinr_db"] = None if worst is None else float(10*np.log10(worst))
            if self.writer is None:
                self.writer = csv.DictWriter(self.csv, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.counts[phase] += 1
            if phase == "scored":
                if is_i: self.values["v2i_active"].append(row["v2i_sinr_linear"])
                else:
                    self.values["v2v_receiver_active"].extend(v["v2v_sinr_linear"][p].tolist())
                    self.values["v2v_bottleneck_active"].append(worst)
            show = (f"V2I={row['v2i_sinr_db']:.6f}dB (linear={row['v2i_sinr_linear']:.9g}); V2V=NA" if is_i else
                    "V2I=NA; V2V_dB=[" + ",".join(f"{row[f'v2v_sinr_db_follower{f+1}']:.6f}" for f in range(self.config.scenario.platoon_size - 1)) + f"]; worst={row['v2v_bottleneck_sinr_db']:.6f}dB")
            line = f"world={world} phase={phase} episode={episode} slot={slot} platoon={p} RB={row['rb']} mode={row['mode']} power={row['power_dbm']:.6f}dBm {show}"
            self.log.write(line + "\n")
            if phase == "scored" and episode <= self.console_episodes: print(line, flush=True)
        if slot == self.config.steps_per_episode:
            self.csv.flush(); self.log.flush()

    def finish(self, worlds, episodes, warmup):
        expected = {"warmup": len(worlds)*warmup*self.config.steps_per_episode*self.config.number_agents,
                    "scored": len(worlds)*episodes*self.config.steps_per_episode*self.config.number_agents}
        if self.counts != expected: raise ValueError("SINR logger phase coverage mismatch")
        summary = {"schema": "executed_slot_sinr_v1", "technical_status": "PASS", "row_counts": self.counts,
            "worlds": list(worlds), "capture": "native _compute_metrics before _renew_fast_fading; no extra physics/RNG/action calls",
            "inactive_links": "null/NA, not0dB; only actually selected V2I or V2V mode included",
            "v2i_required_sinr_linear": float(np.exp2(self.config.v2i_min_bits_per_step / (self.config.slot_ms/1000*self.config.bandwidth_hz))-1),
            "scored_link_statistics": {}}
        for key, vals in self.values.items():
            x = np.asarray(vals, dtype=np.float64)
            db = 10*np.log10(x)
            summary["scored_link_statistics"][key] = {"count": len(vals), "mean_linear": float(x.mean()) if len(x) else None,
                "mean_db": float(db.mean()) if len(x) else None, "p05_db": float(np.quantile(db, .05)) if len(x) else None,
                "median_db": float(np.median(db)) if len(x) else None, "p95_db": float(np.quantile(db, .95)) if len(x) else None}
        (self.root / "sinr_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return summary

    def close(self):
        self.csv.close(); self.log.close()
