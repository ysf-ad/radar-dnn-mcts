"""Run schedulers against the radar simulator with shared timing and metrics."""

from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
import pandas as pd

from pufferlib.ocean.radarxs.engine import RadarEngine, get_obs_from_buf

from radar_dnn_mcts.env.config import BenchmarkConfig
from radar_dnn_mcts.env.factory import create_engine
from radar_dnn_mcts.evaluation.metrics import observation_metrics, summarize_windows


class TimedPlanner:
    """Measure planner latency without including environment execution."""

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
    """Evaluate each planner on the same configured load cells."""

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

    def run(self, planners: dict[str, object], warmup_windows: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return per-window traces and aggregate method summaries."""
        rows: list[dict] = []
        latency_warmup = min(int(warmup_windows), max(0, self.config.windows // 4))
        for initial, rate, seed in self.config.cells():
            for method, planner in planners.items():
                timed = TimedPlanner(planner)
                engine = self._engine(timed, initial, rate, seed)
                cumulative = 0.0
                try:
                    for window in range(self.config.windows):
                        reward = engine.step_window()
                        executed = len(engine.last_executed_plan)
                        searches = sum(row == 0 for row in engine.last_executed_plan)
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
                            "observations": getattr(
                                timed.planner, "last_observations", np.nan
                            ),
                            "h_calls": getattr(
                                timed.planner, "last_h_calls", np.nan
                            ),
                            "g_calls": getattr(
                                timed.planner, "last_g_calls", np.nan
                            ),
                            "f_calls": getattr(
                                timed.planner, "last_f_calls", np.nan
                            ),
                            "simulations": getattr(
                                timed.planner, "last_simulations", np.nan
                            ),
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
