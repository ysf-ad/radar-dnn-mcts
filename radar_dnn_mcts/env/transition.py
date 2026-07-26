from __future__ import annotations

import numpy as np

from radar_dnn_mcts.env.actions import action_duration_ms


class RadarObservationTransition:
    """Deterministic within-window shadow transition used by full re-encoding."""

    grid_width: int = 30
    grid_height: int = 10
    fresh_sector_ms: float = 3000.0

    def clone(self, obs: dict) -> dict:
        return dict(obs)

    def step(self, obs: dict, row: int) -> tuple[dict, float]:
        """Apply one search/track action without advancing the live C engine."""
        nxt = self.clone(obs)
        duration = action_duration_ms(nxt, row)
        active = np.asarray(nxt["active_mask"], dtype=bool)
        desired = np.asarray(nxt["t_desired"], dtype=np.float32).copy()
        deadline = np.asarray(nxt["t_deadline"], dtype=np.float32).copy()
        priority = np.asarray(nxt.get("priority", np.zeros_like(desired)), dtype=np.float32)
        grid = np.asarray(nxt.get("grid", np.zeros(300)), dtype=np.float32).copy()
        refreshed: tuple[int, ...] = ()
        if row == 0:
            # Search refreshes the stalest 2x2 sector represented by the grid.
            if grid.size == self.grid_width * self.grid_height:
                cells = grid.reshape(self.grid_height, self.grid_width)
                candidates = [
                    (float(cells[y : y + 2, x : x + 2].sum()), x, y)
                    for y in range(0, self.grid_height - 1, 2)
                    for x in range(0, self.grid_width - 1, 2)
                ]
                _, x, y = min(candidates, key=lambda item: item[0])
                refreshed = (
                    y * self.grid_width + x,
                    y * self.grid_width + x + 1,
                    (y + 1) * self.grid_width + x,
                    (y + 1) * self.grid_width + x + 1,
                )
        else:
            idx = int(row) - 1
            if idx < active.size and active[idx] and deadline[idx] >= 0.0:
                multiplier = max(1.0, 2.5 - 0.75 * float(priority[idx]))
                desired_period = max(
                    100.0, float(deadline[idx] - desired[idx]) / (multiplier - 1.0)
                )
                desired[idx] = desired_period
                deadline[idx] = desired_period * multiplier

        desired[active] -= duration
        deadline[active] -= duration
        grid -= duration
        if refreshed:
            grid[np.asarray(refreshed, dtype=np.int64)] = self.fresh_sector_ms
        nxt["t_desired"] = desired
        nxt["t_deadline"] = deadline
        nxt["tracked_mask"] = active & (deadline >= 0.0)
        nxt["grid"] = grid
        debt = 0.0 if row == 0 else float(nxt.get("search_debt_ms", 0.0))
        nxt["search_debt_ms"] = debt + duration
        return nxt, duration
