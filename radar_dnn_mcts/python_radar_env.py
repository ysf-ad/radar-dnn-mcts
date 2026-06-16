
"""Vectorized Python radar planning environment.

This is not intended to replace the C environment for final execution yet.
It is a fast, inspectable planning model designed to remove Python<->C replay
from MPC/MCTS scoring. The migration path is:

1. initialize from a real radar observation,
2. score many candidate plans in Python/NumPy,
3. choose a plan,
4. execute only the chosen plan in the real env,
5. validate and tighten the model until C/Python traces match.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

GRID_AZ = 30
GRID_EL = 10
GRID_SIZE = GRID_AZ * GRID_EL
MAXT = 100
SEARCH_DWELL_MS = 10.0
ZERO_COST_SEARCH_TIME = GRID_SIZE * SEARCH_DWELL_MS


@dataclass
class PyRadarConfig:
    search_action_reward: float = 0.10
    track_update_reward: float = 0.30
    track_loss_penalty: float = 4.0
    searched_sector_reward_weight: float = 0.10
    search_frame_overdue_weight: float = 0.05
    search_frame_desired_ms: float = 3000.0
    search_frame_deadline_ms: float = 4500.0
    search_frame_drop_penalty: float = 4.0
    global_delay_weight: float = 1.0 / 1000.0
    min_revisit_ms: float = 100.0
    max_deadline_ms: float = 30000.0
    revisit_time_scale: float = 0.75
    penalize_hidden_targets: bool = True
    search_delay_mode: int = 1
    search_debt_penalty_weight: float = 0.058
    search_debt_tau_ms: float = 200.0
    search_delay_penalty_cap: float = 2.0

    @classmethod
    def from_env_cfg(cls, env_cfg: dict) -> "PyRadarConfig":
        """Map the C environment knobs into the Python planning model."""
        return cls(
            search_action_reward=float(env_cfg.get("search_action_reward", 0.10)),
            track_update_reward=float(env_cfg.get("track_update_reward", 0.30)),
            track_loss_penalty=float(env_cfg.get("track_loss_penalty", 1.0)),
            searched_sector_reward_weight=float(env_cfg.get("searched_sector_reward_weight", 0.0)),
            search_frame_overdue_weight=float(env_cfg.get("search_frame_overdue_weight", 0.0)),
            search_frame_desired_ms=float(env_cfg.get("search_frame_desired_ms", 3000.0)),
            search_frame_deadline_ms=float(env_cfg.get("search_frame_deadline_ms", 4500.0)),
            search_frame_drop_penalty=float(env_cfg.get("search_frame_drop_penalty", 0.0)),
            revisit_time_scale=float(env_cfg.get("revisit_time_scale", 0.75)),
            penalize_hidden_targets=bool(env_cfg.get("penalize_hidden_targets", 0)),
            search_delay_mode=int(env_cfg.get("search_delay_mode", 1)),
            search_debt_penalty_weight=float(env_cfg.get("search_debt_penalty_weight", 0.058)),
            search_debt_tau_ms=float(env_cfg.get("search_debt_tau_ms", 200.0)),
            search_delay_penalty_cap=float(env_cfg.get("search_delay_penalty_cap", 2.0)),
        )


@dataclass
class PyRadarState:
    grid: np.ndarray
    t_desired: np.ndarray
    t_deadline: np.ndarray
    t_dwell: np.ndarray
    priority: np.ndarray
    active: np.ndarray
    search_debt_ms: float = 0.0
    tick_ms: float = 0.0

    @classmethod
    def from_obs(cls, obs: dict, search_debt_ms: float | None = None) -> "PyRadarState":
        return cls(
            grid=np.asarray(obs["grid"], dtype=np.float32).copy(),
            t_desired=np.asarray(obs["t_desired"], dtype=np.float32).copy(),
            t_deadline=np.asarray(obs["t_deadline"], dtype=np.float32).copy(),
            t_dwell=np.asarray(obs["t_dwell"], dtype=np.float32).copy(),
            priority=np.asarray(obs.get("priority", np.zeros_like(obs["t_desired"])), dtype=np.float32).copy(),
            active=np.asarray(obs["active_mask"], dtype=bool).copy(),
            search_debt_ms=float(obs.get("search_debt_ms", 0.0) if search_debt_ms is None else search_debt_ms),
        )

    def clone(self) -> "PyRadarState":
        return PyRadarState(
            self.grid.copy(),
            self.t_desired.copy(),
            self.t_deadline.copy(),
            self.t_dwell.copy(),
            self.priority.copy(),
            self.active.copy(),
            float(self.search_debt_ms),
            float(self.tick_ms),
        )

    @property
    def tracked(self) -> np.ndarray:
        return self.active & (self.t_deadline >= 0.0)

    @property
    def dropped(self) -> np.ndarray:
        return self.active & (self.t_deadline < 0.0)

    def obs_like(self) -> dict:
        return {
            "grid": self.grid,
            "t_desired": self.t_desired,
            "t_deadline": self.t_deadline,
            "t_dwell": self.t_dwell,
            "priority": self.priority,
            "active_mask": self.active,
            "search_debt_ms": self.search_debt_ms,
        }


def _best_lru_macro_sector(grid: np.ndarray) -> np.ndarray:
    g = grid.reshape(GRID_EL, GRID_AZ)
    best_r = 0
    best_c = 0
    best_sum = float("inf")
    for r in range(0, GRID_EL - 1, 2):
        row0 = g[r]
        row1 = g[r + 1]
        for c in range(0, GRID_AZ - 1, 2):
            s = float(row0[c] + row0[c + 1] + row1[c] + row1[c + 1])
            if s < best_sum:
                best_sum = s
                best_r = r
                best_c = c
    return np.asarray(
        [best_r * GRID_AZ + best_c, best_r * GRID_AZ + best_c + 1, (best_r + 1) * GRID_AZ + best_c, (best_r + 1) * GRID_AZ + best_c + 1],
        dtype=np.int32,
    )


def _reset_track_timer(state: PyRadarState, idx: int, cfg: PyRadarConfig):
    # Observation-level approximation of compute_tracker_timers. We preserve
    # priority and dwell heterogeneity, but do not resample target kinematics.
    priority = float(state.priority[idx])
    dwell = max(1.0, float(state.t_dwell[idx]))
    desired = max(cfg.min_revisit_ms, min(cfg.max_deadline_ms, cfg.revisit_time_scale * (750.0 + 3.0 * dwell) / (1.0 + 0.25 * priority)))
    deadline = max(desired, min(cfg.max_deadline_ms, desired * (2.5 - 0.75 * priority)))
    state.t_desired[idx] = desired
    state.t_deadline[idx] = deadline


def advance_time(state: PyRadarState, dt: float, cfg: PyRadarConfig) -> float:
    if dt <= 0.0:
        return 0.0
    reward = 0.0
    state.tick_ms += dt
    state.search_debt_ms += dt

    # Search-frame cost before/after aging, matching the current objective shape.
    state.grid -= dt
    if cfg.search_frame_overdue_weight > 0.0 and cfg.search_frame_desired_ms > 0.0:
        age = ZERO_COST_SEARCH_TIME - state.grid
        overdue = np.maximum(0.0, age - cfg.search_frame_desired_ms)
        frame_cost = np.square(overdue / cfg.search_frame_desired_ms)
        if cfg.search_frame_drop_penalty > 0.0:
            frame_cost = frame_cost + cfg.search_frame_drop_penalty * (age > cfg.search_frame_deadline_ms)
        reward -= cfg.search_frame_overdue_weight * float(np.sum(frame_cost)) / GRID_SIZE

    tracked = state.tracked
    if np.any(tracked):
        overdue_before = np.maximum(0.0, -state.t_desired[tracked])
        state.t_desired[tracked] -= dt
        state.t_deadline[tracked] -= dt
        overdue_after = np.maximum(0.0, -state.t_desired[tracked])
        priority_scale = 1.0 + 2.0 * state.priority[tracked]
        reward -= cfg.global_delay_weight * float(np.sum((overdue_after - overdue_before) * priority_scale))

    expired = state.active & (state.t_deadline < 0.0)
    if np.any(expired):
        reward -= float(np.sum(cfg.track_loss_penalty * (1.0 + 2.0 * state.priority[expired])))
        # Keep active but untracked. If hidden targets are penalized, continue
        # aging their latent timers; otherwise clear them out of tracked mask.
        if not cfg.penalize_hidden_targets:
            state.t_desired[expired] = -1.0
            state.t_deadline[expired] = -1.0
    reward -= _search_delay_penalty(state.search_debt_ms, cfg)
    return reward


def _search_delay_penalty(debt_ms: float, cfg: PyRadarConfig) -> float:
    if cfg.search_debt_penalty_weight <= 0.0 or debt_ms <= 0.0:
        return 0.0
    if cfg.search_delay_mode == 0:
        penalty = cfg.search_debt_penalty_weight * debt_ms
    else:
        arg = min(20.0, debt_ms / max(1e-3, cfg.search_debt_tau_ms))
        penalty = cfg.search_debt_penalty_weight * (float(np.exp(arg)) - 1.0)
    if cfg.search_delay_penalty_cap >= 0.0:
        penalty = min(penalty, cfg.search_delay_penalty_cap)
    return float(penalty)


def step(state: PyRadarState, action: int, cfg: PyRadarConfig = PyRadarConfig()) -> tuple[float, float, int | None]:
    action = int(action)
    reward = 0.0
    if action == 0:
        idx = _best_lru_macro_sector(state.grid)
        # Potential reward for refreshing stale sectors.
        if cfg.searched_sector_reward_weight > 0.0 and cfg.search_frame_desired_ms > 0.0:
            sector_age = ZERO_COST_SEARCH_TIME - state.grid[idx]
            reward += cfg.searched_sector_reward_weight * float(np.mean(np.maximum(0.0, sector_age / cfg.search_frame_desired_ms)))
        state.grid[idx] = ZERO_COST_SEARCH_TIME
        reward += cfg.search_action_reward
        state.search_debt_ms = 0.0
        dt = SEARCH_DWELL_MS
    else:
        idx = action - 1
        if idx < 0 or idx >= len(state.active) or not state.active[idx] or state.t_deadline[idx] < 0.0:
            return -cfg.track_loss_penalty, 0.0, None
        reward += cfg.track_update_reward
        dt = max(1.0, float(state.t_dwell[idx]))
        _reset_track_timer(state, idx, cfg)
    reward += advance_time(state, dt, cfg)
    return float(reward), float(dt), action


def score_plan(root: PyRadarState, plan: Sequence[int], cfg: PyRadarConfig = PyRadarConfig(), budget_ms: float = 200.0) -> tuple[float, list[tuple[float, float, int]]]:
    state = root.clone()
    elapsed = 0.0
    total = 0.0
    executed: list[tuple[float, float, int]] = []
    for action in plan:
        if elapsed >= budget_ms:
            break
        reward, dt, actual = step(state, int(action), cfg)
        if actual is None or dt <= 0.0:
            break
        total += reward
        elapsed += dt
        executed.append((reward, dt, actual))
    return float(total), executed


def score_plans(root: PyRadarState, plans: Iterable[Sequence[int]], cfg: PyRadarConfig = PyRadarConfig(), budget_ms: float = 200.0):
    scores = []
    for i, plan in enumerate(plans):
        reward, executed = score_plan(root, plan, cfg, budget_ms)
        scores.append((i, reward, executed))
    return scores


def score_plans_vectorized(root: PyRadarState, plans: np.ndarray, cfg: PyRadarConfig = PyRadarConfig(), budget_ms: float = 200.0) -> np.ndarray:
    """Score a batch of fixed-length plans with vectorized state evolution.

    This is the path that can actually replace C replay inside planning. It is
    approximate but fast: no Python loop over candidate plans, only over slots.
    """
    plans = np.asarray(plans, dtype=np.int32)
    if plans.ndim != 2:
        raise ValueError("plans must have shape [num_plans, num_slots]")
    bsz, slots = plans.shape
    grid = np.broadcast_to(root.grid.astype(np.float32), (bsz, GRID_SIZE)).copy()
    t_desired = np.broadcast_to(root.t_desired.astype(np.float32), (bsz, root.t_desired.shape[0])).copy()
    t_deadline = np.broadcast_to(root.t_deadline.astype(np.float32), (bsz, root.t_deadline.shape[0])).copy()
    t_dwell = np.broadcast_to(root.t_dwell.astype(np.float32), (bsz, root.t_dwell.shape[0])).copy()
    priority = np.broadcast_to(root.priority.astype(np.float32), (bsz, root.priority.shape[0])).copy()
    active = np.broadcast_to(root.active.astype(bool), (bsz, root.active.shape[0])).copy()
    search_debt = np.full(bsz, float(root.search_debt_ms), dtype=np.float32)
    elapsed = np.zeros(bsz, dtype=np.float32)
    total = np.zeros(bsz, dtype=np.float32)
    alive = np.ones(bsz, dtype=bool)

    rows = np.arange(bsz)
    for slot in range(slots):
        if not np.any(alive):
            break
        action = plans[:, slot]
        in_budget = alive & (elapsed < budget_ms)
        if not np.any(in_budget):
            break
        reward = np.zeros(bsz, dtype=np.float32)
        dt = np.zeros(bsz, dtype=np.float32)

        do_search = in_budget & (action == 0)
        if np.any(do_search):
            lru = _best_lru_macro_sector_batch(grid[do_search])
            sr = rows[do_search]
            if cfg.searched_sector_reward_weight > 0.0 and cfg.search_frame_desired_ms > 0.0:
                sector_age = ZERO_COST_SEARCH_TIME - grid[sr[:, None], lru]
                reward[do_search] += cfg.searched_sector_reward_weight * np.mean(
                    np.maximum(0.0, sector_age / cfg.search_frame_desired_ms), axis=1
                ).astype(np.float32)
            grid[sr[:, None], lru] = ZERO_COST_SEARCH_TIME
            reward[do_search] += cfg.search_action_reward
            search_debt[do_search] = 0.0
            dt[do_search] = SEARCH_DWELL_MS

        do_track = in_budget & (action > 0)
        if np.any(do_track):
            idx = action[do_track] - 1
            tr = rows[do_track]
            valid = (idx >= 0) & (idx < active.shape[1]) & active[tr, idx] & (t_deadline[tr, idx] >= 0.0)
            valid_rows = tr[valid]
            valid_idx = idx[valid]
            invalid_mask = do_track.copy()
            invalid_rows = tr[~valid]
            invalid_mask[:] = False
            invalid_mask[invalid_rows] = True
            if invalid_rows.size:
                reward[invalid_rows] -= cfg.track_loss_penalty
                alive[invalid_rows] = False
            if valid_rows.size:
                reward[valid_rows] += cfg.track_update_reward
                dt[valid_rows] = np.maximum(1.0, t_dwell[valid_rows, valid_idx])
                p = priority[valid_rows, valid_idx]
                dwell = np.maximum(1.0, t_dwell[valid_rows, valid_idx])
                desired = cfg.revisit_time_scale * (750.0 + 3.0 * dwell) / (1.0 + 0.25 * p)
                desired = np.clip(desired, cfg.min_revisit_ms, cfg.max_deadline_ms)
                deadline = np.maximum(desired, desired * (2.5 - 0.75 * p))
                deadline = np.minimum(deadline, cfg.max_deadline_ms)
                t_desired[valid_rows, valid_idx] = desired
                t_deadline[valid_rows, valid_idx] = deadline

        advance = in_budget & (dt > 0.0)
        if np.any(advance):
            ar = rows[advance]
            adt = dt[advance]
            search_debt[advance] += adt
            grid[advance] -= adt[:, None]
            if cfg.search_frame_overdue_weight > 0.0 and cfg.search_frame_desired_ms > 0.0:
                age = ZERO_COST_SEARCH_TIME - grid[advance]
                overdue = np.maximum(0.0, age - cfg.search_frame_desired_ms)
                frame_cost = np.square(overdue / cfg.search_frame_desired_ms)
                if cfg.search_frame_drop_penalty > 0.0:
                    frame_cost = frame_cost + cfg.search_frame_drop_penalty * (age > cfg.search_frame_deadline_ms)
                reward[advance] -= cfg.search_frame_overdue_weight * np.mean(frame_cost, axis=1).astype(np.float32)

            tracked = active[advance] & (t_deadline[advance] >= 0.0)
            overdue_before = np.maximum(0.0, -t_desired[advance]) * tracked
            t_desired[advance] -= adt[:, None] * tracked
            t_deadline[advance] -= adt[:, None] * tracked
            overdue_after = np.maximum(0.0, -t_desired[advance]) * tracked
            priority_scale = 1.0 + 2.0 * priority[advance]
            reward[advance] -= cfg.global_delay_weight * np.sum((overdue_after - overdue_before) * priority_scale, axis=1).astype(np.float32)

            expired = active[advance] & (t_deadline[advance] < 0.0)
            if np.any(expired):
                reward[advance] -= np.sum(cfg.track_loss_penalty * (1.0 + 2.0 * priority[advance]) * expired, axis=1).astype(np.float32)
                if not cfg.penalize_hidden_targets:
                    t_desired[ar[:, None], np.arange(active.shape[1])[None, :]] = np.where(expired, -1.0, t_desired[advance])
                    t_deadline[ar[:, None], np.arange(active.shape[1])[None, :]] = np.where(expired, -1.0, t_deadline[advance])
            reward[advance] -= _search_delay_penalty_batch(search_debt[advance], cfg)
            elapsed[advance] += adt
            total[advance] += reward[advance]
    return total


def _best_lru_macro_sector_batch(grid_batch: np.ndarray) -> np.ndarray:
    g = grid_batch.reshape((-1, GRID_EL, GRID_AZ))
    sums = (
        g[:, 0:GRID_EL - 1:2, 0:GRID_AZ - 1:2]
        + g[:, 0:GRID_EL - 1:2, 1:GRID_AZ:2]
        + g[:, 1:GRID_EL:2, 0:GRID_AZ - 1:2]
        + g[:, 1:GRID_EL:2, 1:GRID_AZ:2]
    )
    flat = np.argmin(sums.reshape((grid_batch.shape[0], -1)), axis=1)
    cols_per = GRID_AZ // 2
    r = (flat // cols_per) * 2
    c = (flat % cols_per) * 2
    base = r * GRID_AZ + c
    return np.stack([base, base + 1, base + GRID_AZ, base + GRID_AZ + 1], axis=1).astype(np.int32)


def _search_delay_penalty_batch(debt_ms: np.ndarray, cfg: PyRadarConfig) -> np.ndarray:
    if cfg.search_debt_penalty_weight <= 0.0:
        return np.zeros_like(debt_ms, dtype=np.float32)
    if cfg.search_delay_mode == 0:
        penalty = cfg.search_debt_penalty_weight * debt_ms
    else:
        arg = np.minimum(20.0, debt_ms / max(1e-3, cfg.search_debt_tau_ms))
        penalty = cfg.search_debt_penalty_weight * (np.exp(arg) - 1.0)
    if cfg.search_delay_penalty_cap >= 0.0:
        penalty = np.minimum(penalty, cfg.search_delay_penalty_cap)
    return penalty.astype(np.float32)
