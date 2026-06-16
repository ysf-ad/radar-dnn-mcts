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


def slot_features_batch(observations, elapsed=None, search_count=None, track_count=None, last_action=None, budget_ms=200.0):
    n = len(observations)
    elapsed = np.zeros((n,), dtype=np.float32) if elapsed is None else np.asarray(list(elapsed), dtype=np.float32)
    search_count = np.zeros((n,), dtype=np.float32) if search_count is None else np.asarray(list(search_count), dtype=np.float32)
    track_count = np.zeros((n,), dtype=np.float32) if track_count is None else np.asarray(list(track_count), dtype=np.float32)
    last_action = -np.ones((n,), dtype=np.int32) if last_action is None else np.asarray(list(last_action), dtype=np.int32)
    active = np.stack([np.asarray(obs["active_mask"]).astype(bool) for obs in observations], axis=0)
    deadline = np.stack([np.asarray(obs["t_deadline"], dtype=np.float32) for obs in observations], axis=0)
    dwell = np.stack([np.asarray(obs["t_dwell"], dtype=np.float32) for obs in observations], axis=0)
    tracked = active & (deadline >= 0.0)
    workload = np.sum(np.where(tracked, dwell, 0.0), axis=1) / max(1.0, float(budget_ms))
    positive_deadline = tracked & (deadline > 0.0)
    min_deadline_arr = np.where(positive_deadline, deadline, np.inf).min(axis=1)
    min_deadline_arr = np.where(np.isfinite(min_deadline_arr), min_deadline_arr, 0.0)
    s_busy = np.asarray([float(obs.get("s_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32)
    x_busy = np.asarray([float(obs.get("x_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32)
    x_enabled = np.asarray([float(obs.get("enable_x_band", 0.0)) for obs in observations], dtype=np.float32)
    arrival_feature = x_enabled.copy()
    for i, obs in enumerate(observations):
        if float(obs.get("use_arrival_feature", 0.0)) > 0.5:
            arrival_feature[i] += np.clip(float(obs.get("arrival_rate", 0.0)) / 10.0, 0.0, 2.0)

    feat = np.empty((n, SLOT_DIM), dtype=np.float32)
    feat[:, 0] = elapsed / float(budget_ms)
    feat[:, 1] = search_count / 20.0
    feat[:, 2] = track_count / 100.0
    feat[:, 3] = (last_action == 0).astype(np.float32)
    feat[:, 4] = np.sum(active, axis=1).astype(np.float32) / 100.0
    feat[:, 5] = np.sum(tracked, axis=1).astype(np.float32) / 100.0
    feat[:, 6] = np.minimum(workload / 20.0, 2.0)
    feat[:, 7] = min_deadline_arr / 3000.0
    feat[:, 8] = np.clip(s_busy / 200.0, 0.0, 5.0)
    feat[:, 9] = np.clip(x_busy / 200.0, 0.0, 5.0)
    feat[:, 10] = arrival_feature
    for i, obs in enumerate(observations):
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            mean_overdue, drop_frac, max_age = _grid_summary(obs)
            feat[i, 8] = mean_overdue
            feat[i, 9] = drop_frac
            feat[i, 10] = max_age
    return feat


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


def _search_delay_penalty_batch(pure_mcts, search_debt_ms):
    search_debt_ms = np.asarray(search_debt_ms, dtype=np.float32)
    if float(pure_mcts.search_debt_penalty_weight) <= 0.0:
        return np.zeros_like(search_debt_ms, dtype=np.float32)
    positive = search_debt_ms > 0.0
    if int(pure_mcts.search_delay_mode) == 0:
        penalty = float(pure_mcts.search_debt_penalty_weight) * search_debt_ms
    else:
        arg = np.minimum(search_debt_ms / max(1e-3, float(pure_mcts.search_debt_tau_ms)), 20.0)
        penalty = float(pure_mcts.search_debt_penalty_weight) * (np.exp(arg) - 1.0)
    penalty = np.where(positive, penalty, 0.0)
    if float(pure_mcts.search_delay_penalty_cap) >= 0.0:
        penalty = np.minimum(penalty, float(pure_mcts.search_delay_penalty_cap))
    return penalty.astype(np.float32)


def tokenize_batch(adapt, observations, selected=None, search_count=None):
    n = len(observations)
    selected = [set() for _ in range(n)] if selected is None else [set(x) for x in selected]
    search_count = np.zeros((n,), dtype=np.float32) if search_count is None else np.asarray(list(search_count), dtype=np.float32)
    x = np.zeros((n, MAXT + 1, TOKEN_DIM), dtype=np.float32)
    if n <= 0:
        return x

    t_desired = np.stack([np.asarray(obs["t_desired"], dtype=np.float32)[:MAXT] for obs in observations], axis=0)
    deadline = np.stack([np.asarray(obs["t_deadline"], dtype=np.float32)[:MAXT] for obs in observations], axis=0)
    dwell = np.stack([np.asarray(obs["t_dwell"], dtype=np.float32)[:MAXT] for obs in observations], axis=0)
    active = np.stack([np.asarray(obs["active_mask"]).astype(bool)[:MAXT] for obs in observations], axis=0)
    tracked = np.stack(
        [
            np.asarray(obs.get("tracked_mask", np.asarray(obs["active_mask"]).astype(bool) & (np.asarray(obs["t_deadline"]) > 0)), dtype=bool)[:MAXT]
            for obs in observations
        ],
        axis=0,
    )
    priority = np.stack(
        [np.asarray(obs.get("priority", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)[:MAXT] for obs in observations],
        axis=0,
    )
    ranges = np.stack(
        [np.asarray(obs.get("target_range", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)[:MAXT] for obs in observations],
        axis=0,
    )
    grids = [np.asarray(obs.get("grid", np.zeros((300,), dtype=np.float32)), dtype=np.float32) for obs in observations]
    grid_min = np.asarray([float(np.min(grid)) for grid in grids], dtype=np.float32)
    search_debt_ms = np.asarray([float(obs.get("search_debt_ms", 0.0)) for obs in observations], dtype=np.float32)
    search_penalty_norm = np.clip(_search_delay_penalty_batch(adapt.pure_mcts, search_debt_ms), 0.0, 10.0)

    tracked_active = active & tracked
    tracked_n = np.sum(tracked_active, axis=1).astype(np.float32)
    tracked_delays = np.maximum(0.0, -t_desired) * tracked_active.astype(np.float32)
    tracked_delay_sum = np.sum(tracked_delays, axis=1)
    mean_tracked_delay_norm = np.divide(
        tracked_delay_sum,
        np.maximum(tracked_n, 1.0),
        out=np.zeros_like(tracked_delay_sum),
        where=tracked_n > 0,
    )
    mean_tracked_delay_norm = np.clip(mean_tracked_delay_norm / 2000.0, 0.0, 10.0)
    overdue_count = np.sum((t_desired < 0.0) & tracked_active, axis=1).astype(np.float32)
    overdue_frac = np.divide(overdue_count, np.maximum(tracked_n, 1.0), out=np.zeros_like(overdue_count), where=tracked_n > 0)
    global_tardiness_norm = np.clip(tracked_delay_sum / 20000.0, 0.0, 10.0)
    deadline_pressure = np.maximum(0.0, 100.0 - deadline) * tracked_active.astype(np.float32)
    global_deadline_pressure_norm = np.clip(np.sum(deadline_pressure, axis=1) / 2000.0, 0.0, 10.0)
    global_penalty_norm = np.clip(
        0.001
        * (
            float(adapt.pure_mcts.global_tardiness_weight) * global_tardiness_norm
            + float(adapt.pure_mcts.local_tardiness_weight) * mean_tracked_delay_norm
        ),
        0.0,
        10.0,
    )

    x[:, 0, :8] = np.stack(
        [
            tracked_n / max(1, int(adapt.max_trackers)),
            grid_min,
            global_tardiness_norm,
            mean_tracked_delay_norm,
            overdue_frac,
            global_deadline_pressure_norm,
            search_penalty_norm,
            global_penalty_norm,
        ],
        axis=1,
    )

    az_bin = np.stack([np.asarray(obs.get("az_bin", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)[:MAXT] for obs in observations], axis=0)
    el_bin = np.stack([np.asarray(obs.get("el_bin", np.zeros(MAXT, dtype=np.float32)), dtype=np.float32)[:MAXT] for obs in observations], axis=0)
    az_idx = np.clip(np.round(az_bin * 29.0).astype(np.int32), 0, 29)
    el_idx = np.clip(np.round(el_bin * 9.0).astype(np.int32), 0, 9)
    sector_idx = np.clip(el_idx * 30 + az_idx, 0, 299)
    sector_urgency = np.zeros((n, MAXT), dtype=np.float32)
    for i, grid in enumerate(grids):
        if len(grid) > 0:
            sector_urgency[i] = grid[np.clip(sector_idx[i], 0, len(grid) - 1)].astype(np.float32)

    target_tardiness = np.maximum(0.0, -t_desired).astype(np.float32)
    local_penalty_norm = np.clip(
        0.001 * target_tardiness * (1.0 + 2.0 * priority) * float(adapt.pure_mcts.local_tardiness_weight),
        0.0,
        10.0,
    ).astype(np.float32)
    x[:, 1 : MAXT + 1, 0] = t_desired
    x[:, 1 : MAXT + 1, 1] = deadline
    x[:, 1 : MAXT + 1, 2] = dwell
    x[:, 1 : MAXT + 1, 3] = priority
    x[:, 1 : MAXT + 1, 4] = (active & tracked).astype(np.float32)
    x[:, 1 : MAXT + 1, 5] = sector_urgency
    x[:, 1 : MAXT + 1, 6] = local_penalty_norm
    x[:, 1 : MAXT + 1, 7] = (global_penalty_norm + search_penalty_norm)[:, None]

    x[:, :, 0] = np.clip(x[:, :, 0] / 3000.0, -2.0, 2.0)
    x[:, :, 1] = np.clip(x[:, :, 1] / 3000.0, -2.0, 2.0)
    x[:, :, 2] = np.clip(x[:, :, 2] / 100.0, 0.0, 2.0)
    x[:, :, 5] = np.clip(x[:, :, 5] / 3000.0, -2.0, 2.0)

    for row, selected_row in enumerate(selected):
        for action in selected_row:
            idx = int(action)
            if 1 <= idx <= MAXT:
                x[row, idx, 8] = 1.0
    x[:, 0, 8] = search_count / 20.0
    range_norm = np.clip(ranges / 184_000_000.0, 0.0, 1.5)
    x[:, 1 : MAXT + 1, 9] = range_norm
    x[:, 1 : MAXT + 1, 10] = ((ranges > 10_000_000.0) & (ranges < 184_000_000.0)).astype(np.float32)
    x[:, 1 : MAXT + 1, 11] = ((ranges > 5_000_000.0) & (ranges < 100_000_000.0)).astype(np.float32)
    sensor_id = np.asarray([float(obs.get("sensor_id", 0.0)) for obs in observations], dtype=np.float32)
    x[:, :, 12] = sensor_id[:, None]
    x[:, 0, 9] = np.clip(np.asarray([float(obs.get("s_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32) / 200.0, 0.0, 5.0)
    x[:, 0, 10] = np.clip(np.asarray([float(obs.get("x_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32) / 200.0, 0.0, 5.0)
    x[:, 0, 11] = np.asarray([float(obs.get("enable_x_band", 0.0)) for obs in observations], dtype=np.float32)
    for i, obs in enumerate(observations):
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            mean_overdue, drop_frac, max_age = _grid_summary(obs)
            x[i, 0, 9] = mean_overdue
            x[i, 0, 10] = drop_frac
            x[i, 0, 11] = max_age
    return x.astype(np.float32, copy=False)
