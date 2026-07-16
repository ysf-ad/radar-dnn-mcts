from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
import pandas as pd

from pufferlib.ocean.radarxs import binding
from pufferlib.ocean.radarxs.engine import RadarEngine, get_obs_from_buf

from radar_dnn_mcts.env.config import BenchmarkConfig
from radar_dnn_mcts.env.factory import create_engine
from radar_dnn_mcts.env.reward import shaped_transition_reward
from radar_dnn_mcts.evaluation.metrics import observation_metrics, summarize_windows


class TimedPlanner:
    def __init__(self, planner):
        self.planner = planner
        self.last_plan: list[int] = []
        self.last_latency_ms = 0.0

    def plan(self, obs: dict, budget_ms: float = 200.0) -> list[int]:
        start = time.perf_counter_ns()
        self.last_plan = list(self.planner.plan(obs, budget_ms))
        self.last_latency_ms = (time.perf_counter_ns() - start) / 1e6
        return self.last_plan


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def _engine(self, planner, initial: int, rate: float, seed: int) -> RadarEngine:
        return create_engine(
            planner,
            initial_targets=initial,
            max_targets=self.config.max_targets,
            seed=seed,
            arrival_rate=rate,
            window_ms=self.config.window_ms,
            dual_sensor=len(self.config.sensors) > 1,
            reward=self.config.reward,
        )

    def _run_window(self, engine: RadarEngine, timed: TimedPlanner) -> tuple[float, int, int]:
        obs = get_obs_from_buf(engine.obs_buf, self.config.max_targets)
        plan = timed.plan(obs, self.config.window_ms)
        spent_ms = 0.0
        reward = 0.0
        searches = 0
        executed = 0
        for action in plan:
            row = int(action)
            if spent_ms >= self.config.window_ms or bool(engine.term_buf[0]):
                break
            before = get_obs_from_buf(engine.obs_buf, self.config.max_targets)
            if row < 0 or row > self.config.max_targets:
                continue
            if row > 0:
                idx = row - 1
                if not before["active_mask"][idx] or before["t_deadline"][idx] < 0.0:
                    continue
                estimated_duration = float(before["t_dwell"][idx])
            else:
                estimated_duration = 10.0
            engine.act_buf[0] = row
            binding.vec_step(engine.env)
            after = get_obs_from_buf(engine.obs_buf, self.config.max_targets)
            duration = 10.0 if row == 0 else self._elapsed_ms(before, after)
            if duration <= 0.0:
                duration = estimated_duration
            reward += shaped_transition_reward(
                float(engine.rew_buf[0]),
                duration,
                before,
                after,
                row,
                self.config.reward,
            )
            spent_ms += duration
            searches += int(row == 0)
            executed += 1
        return reward, searches, executed

    @staticmethod
    def _elapsed_ms(before: dict, after: dict) -> float:
        candidates: list[float] = []
        grid_delta = np.asarray(before["grid"]) - np.asarray(after["grid"])
        if np.any(grid_delta > 0.0):
            candidates.append(float(np.max(grid_delta)))
        active = np.asarray(before["active_mask"], dtype=bool) & np.asarray(
            after["active_mask"], dtype=bool
        )
        if np.any(active):
            for key in ("t_desired", "t_deadline"):
                delta = np.asarray(before[key])[active] - np.asarray(after[key])[active]
                if np.any(delta > 0.0):
                    candidates.append(float(np.max(delta)))
        return max(candidates, default=0.0)

    def run(self, planners: dict[str, object], warmup_windows: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[dict] = []
        latency_warmup = min(int(warmup_windows), max(0, self.config.windows // 4))
        for initial, rate, seed in self.config.cells():
            for method, planner in planners.items():
                timed = TimedPlanner(planner)
                engine = self._engine(timed, initial, rate, seed)
                cumulative = 0.0
                try:
                    for window in range(self.config.windows):
                        reward, searches, executed = self._run_window(engine, timed)
                        cumulative += reward
                        obs = get_obs_from_buf(engine.obs_buf, self.config.max_targets)
                        metrics = observation_metrics(obs)
                        row = {
                            "method": method,
                            "initial_targets": initial,
                            "arrival_rate": rate,
                            "seed": seed,
                            "window": window + 1,
                            "reward": reward,
                            "cumulative_reward": cumulative,
                            "search_fraction": searches / max(1, executed),
                            "latency_ms": np.nan if window < latency_warmup else timed.last_latency_ms,
                            **metrics,
                        }
                        rows.append(row)
                        if engine.term_buf[0]:
                            break
                finally:
                    engine.close()
        windows = pd.DataFrame(rows)
        return windows, summarize_windows(windows)

    def manifest(self) -> dict:
        return asdict(self.config)
