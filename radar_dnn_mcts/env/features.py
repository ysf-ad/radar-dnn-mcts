from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TOKEN_DIM = 13
CONTEXT_DIM = 11


@dataclass(frozen=True)
class FeatureBuilder:
    """Convert a variable radar observation into fixed-size model inputs."""

    max_targets: int = 100
    window_ms: float = 200.0

    def tokens(self, obs: dict, selected: set[int] | None = None, search_count: int = 0) -> np.ndarray:
        """Build row 0 for search/global state and rows 1..N for targets."""
        selected = selected or set()
        rows = self.max_targets + 1
        x = np.zeros((rows, TOKEN_DIM), dtype=np.float32)
        active = np.asarray(obs["active_mask"], dtype=bool)[: self.max_targets]
        tracked = np.asarray(obs.get("tracked_mask", active), dtype=bool)[: self.max_targets]
        desired = np.asarray(obs["t_desired"], dtype=np.float32)[: self.max_targets]
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)[: self.max_targets]
        dwell = np.asarray(obs["t_dwell"], dtype=np.float32)[: self.max_targets]
        priority = np.asarray(obs.get("priority", np.zeros_like(desired)), dtype=np.float32)[: self.max_targets]
        grid = np.asarray(obs.get("grid", np.zeros(300)), dtype=np.float32)
        az = np.asarray(obs.get("az_bin", np.zeros_like(desired)), dtype=np.float32)[: self.max_targets]
        el = np.asarray(obs.get("el_bin", np.zeros_like(desired)), dtype=np.float32)[: self.max_targets]
        sector = np.clip(np.rint(el * 9).astype(int) * 30 + np.rint(az * 29).astype(int), 0, max(0, grid.size - 1))
        local_delay = np.maximum(0.0, -desired)

        tracked_active = active & tracked
        tracked_delay = local_delay[tracked_active]
        macro_urgencies = np.empty(0, dtype=np.float32)
        if grid.size == 300:
            patches = grid.reshape(10, 30)
            macro_urgencies = patches.reshape(5, 2, 15, 2).min(axis=(1, 3)).reshape(-1)
        x[0, :8] = [
            float(tracked_active.mean()) if tracked_active.size else 0.0,
            float(np.clip(grid.min() / 3000.0, -2.0, 2.0)) if grid.size else 0.0,
            float(np.clip(tracked_delay.sum() / 20000.0, 0.0, 10.0)),
            float(np.clip(tracked_delay.mean() / 2000.0, 0.0, 10.0)) if tracked_delay.size else 0.0,
            float((desired[tracked_active] < 0).mean()) if tracked_active.any() else 0.0,
            float(np.clip(np.maximum(0.0, 100.0 - deadline[tracked_active]).sum() / 2000.0, 0.0, 10.0)),
            float(np.clip(float(obs.get("search_debt_ms", 0.0)) / 1000.0, 0.0, 10.0)),
            float(np.clip(local_delay[tracked_active].sum() / 20000.0, 0.0, 10.0)),
        ]
        n = len(desired)
        x[1 : n + 1, 0] = np.clip(desired / 3000.0, -2.0, 2.0)
        x[1 : n + 1, 1] = np.clip(deadline / 3000.0, -2.0, 2.0)
        x[1 : n + 1, 2] = np.clip(dwell / 100.0, 0.0, 2.0)
        x[1 : n + 1, 3] = priority
        x[1 : n + 1, 4] = (active & tracked).astype(np.float32)
        x[1 : n + 1, 5] = grid[sector] / 3000.0 if grid.size else 0.0
        x[1 : n + 1, 6] = np.clip(0.001 * local_delay * (1.0 + 2.0 * priority), 0.0, 10.0)
        x[1 : n + 1, 7] = x[0, 7]
        x[0, 8] = float(search_count) / 20.0
        if macro_urgencies.size:
            x[0, 9:12] = np.clip(
                np.quantile(macro_urgencies, [0.0, 0.25, 0.5]) / 3000.0,
                -2.0,
                2.0,
            )
        for row in selected:
            if 1 <= row <= self.max_targets:
                x[row, 8] = 1.0
        ranges = np.asarray(obs.get("target_range", np.zeros(self.max_targets)), dtype=np.float32)[:n]
        x[1 : n + 1, 9] = np.clip(ranges / 184_000_000.0, 0.0, 1.5)
        x[1 : n + 1, 10] = ((ranges > 10_000_000.0) & (ranges < 184_000_000.0)).astype(np.float32)
        x[1 : n + 1, 11] = ((ranges > 5_000_000.0) & (ranges < 100_000_000.0)).astype(np.float32)
        x[:, 12] = float(obs.get("sensor_id", 0.0))
        return x

    def context(self, obs: dict, elapsed_ms: float, search_count: int, track_count: int, last_row: int) -> np.ndarray:
        """Build window-level timing, load, and sensor-availability features."""
        active = np.asarray(obs["active_mask"], dtype=bool)[: self.max_targets]
        deadline = np.asarray(obs["t_deadline"], dtype=np.float32)[: self.max_targets]
        dwell = np.asarray(obs["t_dwell"], dtype=np.float32)[: self.max_targets]
        tracked = active & (deadline >= 0)
        positive = deadline[tracked & (deadline > 0)]
        return np.asarray(
            [
                elapsed_ms / self.window_ms,
                search_count / 20.0,
                track_count / max(1.0, float(self.max_targets)),
                float(last_row == 0),
                active.sum() / max(1.0, float(self.max_targets)),
                tracked.sum() / max(1.0, float(self.max_targets)),
                min(float(dwell[tracked].sum()) / (20.0 * self.window_ms), 2.0),
                float(positive.min()) / 3000.0 if positive.size else 0.0,
                np.clip(float(obs.get("s_band_busy_ms", 0.0)) / self.window_ms, 0.0, 5.0),
                np.clip(float(obs.get("x_band_busy_ms", 0.0)) / self.window_ms, 0.0, 5.0),
                np.clip(float(obs.get("arrival_rate", 0.0)) / 10.0, 0.0, 2.0),
            ],
            dtype=np.float32,
        )
