from __future__ import annotations

from copy import deepcopy

import numpy as np

from radar_dnn_mcts.env.actions import action_duration_ms


class RadarObservationTransition:
    """Deterministic within-window shadow transition used by full re-encoding."""

    def clone(self, obs: dict) -> dict:
        return deepcopy(obs)

    def step(self, obs: dict, row: int) -> tuple[dict, float]:
        nxt = self.clone(obs)
        duration = action_duration_ms(nxt, row)
        active = np.asarray(nxt["active_mask"], dtype=bool)
        nxt["t_desired"] = np.asarray(nxt["t_desired"], dtype=np.float32) - duration
        nxt["t_deadline"] = np.asarray(nxt["t_deadline"], dtype=np.float32) - duration
        if row == 0:
            grid = np.asarray(nxt.get("grid", np.zeros(300)), dtype=np.float32).copy()
            if grid.size:
                start = int(np.argmin(grid))
                grid[start] = max(3000.0, float(grid[start]) + 3000.0)
            nxt["grid"] = grid
            nxt["search_debt_ms"] = 0.0
        else:
            idx = int(row) - 1
            if idx < active.size and active[idx]:
                desired_period = max(100.0, float(-nxt["t_desired"][idx] + 1000.0))
                deadline_period = max(desired_period + 100.0, float(-nxt["t_deadline"][idx] + 3000.0))
                nxt["t_desired"][idx] = desired_period
                nxt["t_deadline"][idx] = deadline_period
            nxt["search_debt_ms"] = float(nxt.get("search_debt_ms", 0.0)) + duration
        return nxt, duration
