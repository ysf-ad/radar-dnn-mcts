from __future__ import annotations

import numpy as np

from final_radar_campaign import MAXT

TOKEN_DIM = 13
SLOT_DIM = 11


def _grid_summary(obs):
    grid = np.asarray(obs.get("grid", []), dtype=np.float32)
    if grid.size == 0:
        return 0.0, 0.0, 0.0
    age = 3000.0 - grid
    overdue = np.maximum(0.0, age - 3000.0) / 3000.0
    mean_overdue = float(np.mean(overdue))
    drop_frac = float(np.mean(age > 4500.0))
    max_age = float(np.max(age) / 4500.0)
    return np.clip(mean_overdue, 0.0, 5.0), np.clip(drop_frac, 0.0, 1.0), np.clip(max_age, 0.0, 5.0)


def slot_features(obs, elapsed, search_count, track_count, last_action, budget_ms=200.0):
    active = np.asarray(obs["active_mask"]).astype(bool)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    dwell = np.asarray(obs["t_dwell"], dtype=np.float32)
    tracked = active & (deadline >= 0.0)
    workload = float(np.sum(dwell[tracked]) / max(1.0, budget_ms))
    min_deadline = float(np.min(deadline[tracked & (deadline > 0)])) if np.any(tracked & (deadline > 0)) else 0.0
    s_busy = float(obs.get("s_band_busy_ms", 0.0))
    x_busy = float(obs.get("x_band_busy_ms", 0.0))
    x_enabled = float(obs.get("enable_x_band", 0.0))
    # Opt-in arrival intensity without changing SLOT_DIM/checkpoint shapes.
    # Keep legacy checkpoints comparable unless the caller explicitly enables
    # this feature in the observation.
    arrival_feature = x_enabled
    if float(obs.get("use_arrival_feature", 0.0)) > 0.5:
        arrival_feature += np.clip(float(obs.get("arrival_rate", 0.0)) / 10.0, 0.0, 2.0)
    feat = [
            float(elapsed) / budget_ms,
            float(search_count) / 20.0,
            float(track_count) / 100.0,
            1.0 if int(last_action) == 0 else 0.0,
            float(np.sum(active)) / 100.0,
            float(np.sum(tracked)) / 100.0,
            min(workload / 20.0, 2.0),
            min_deadline / 3000.0,
            np.clip(s_busy / 200.0, 0.0, 5.0),
            np.clip(x_busy / 200.0, 0.0, 5.0),
            arrival_feature,
        ]
    if float(obs.get("use_grid_feature", 0.0)) > 0.5:
        mean_overdue, drop_frac, max_age = _grid_summary(obs)
        feat[8] = mean_overdue
        feat[9] = drop_frac
        feat[10] = max_age
    return np.asarray(feat, dtype=np.float32)


def tokenize(adapt, obs, selected=None, search_count=0):
    x8 = adapt.adapt_obs(obs)[0]
    x = np.zeros((MAXT + 1, TOKEN_DIM), dtype=np.float32)
    x[:, :8] = x8
    x[:, 0] = np.clip(x[:, 0] / 3000.0, -2.0, 2.0)
    x[:, 1] = np.clip(x[:, 1] / 3000.0, -2.0, 2.0)
    x[:, 2] = np.clip(x[:, 2] / 100.0, 0.0, 2.0)
    x[:, 5] = np.clip(x[:, 5] / 3000.0, -2.0, 2.0)
    if selected:
        for a in selected:
            if 1 <= int(a) <= MAXT:
                x[int(a), 8] = 1.0
    x[0, 8] = float(search_count) / 20.0
    ranges = np.asarray(obs.get("target_range", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)
    n = min(MAXT, len(ranges))
    if n > 0:
        range_norm = np.clip(ranges[:n] / 184_000_000.0, 0.0, 1.5)
        x[1 : n + 1, 9] = range_norm
        x[1 : n + 1, 10] = ((ranges[:n] > 10_000_000.0) & (ranges[:n] < 184_000_000.0)).astype(np.float32)
        x[1 : n + 1, 11] = ((ranges[:n] > 5_000_000.0) & (ranges[:n] < 100_000_000.0)).astype(np.float32)
    x[:, 12] = float(obs.get("sensor_id", 0.0))
    x[0, 9] = np.clip(float(obs.get("s_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0)
    x[0, 10] = np.clip(float(obs.get("x_band_busy_ms", 0.0)) / 200.0, 0.0, 5.0)
    x[0, 11] = float(obs.get("enable_x_band", 0.0))
    if float(obs.get("use_grid_feature", 0.0)) > 0.5:
        mean_overdue, drop_frac, max_age = _grid_summary(obs)
        x[0, 9] = mean_overdue
        x[0, 10] = drop_frac
        x[0, 11] = max_age
    return x
