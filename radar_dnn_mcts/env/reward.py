"""Reward terms shared by the live simulator and shadow search state."""

from __future__ import annotations

import numpy as np

from radar_dnn_mcts.env.config import RewardConfig


def normalized_tardiness(obs: dict) -> float:
    """Sum normalized delay beyond each active track's desired time."""
    active = np.asarray(obs["active_mask"], dtype=bool)
    desired = np.asarray(obs["t_desired"], dtype=np.float32)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    tracked = active & (deadline >= 0.0)
    scale = np.maximum(1.0, deadline - desired)
    return float((np.maximum(0.0, -desired[tracked]) / scale[tracked]).sum())


def service_pressure(obs: dict) -> float:
    """Return total normalized urgency for active tracking tasks."""
    active = np.asarray(obs["active_mask"], dtype=bool)
    desired = np.asarray(obs["t_desired"], dtype=np.float32)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    known = active & np.isfinite(desired) & np.isfinite(deadline)
    if not np.any(known):
        return 0.0
    late = np.clip(-desired[known] / 1000.0, 0.0, 4.0)
    deadline_risk = np.clip((500.0 - deadline[known]) / 500.0, 0.0, 2.0)
    dropped = (deadline[known] < 0.0).astype(np.float32)
    return float(np.sum(late + 2.0 * deadline_risk + 8.0 * dropped))


def search_frame_pressure(obs: dict, config: RewardConfig) -> float:
    """Measure accumulated staleness across search sectors."""
    grid = np.asarray(obs.get("grid", []), dtype=np.float32)
    if not grid.size:
        return 0.0
    age = 3000.0 - grid
    overdue = np.maximum(0.0, age - config.search_frame_desired_ms) / max(
        1.0, config.search_frame_desired_ms
    )
    pressure = float(np.mean(overdue))
    if config.search_frame_drop_penalty > 0.0:
        pressure += config.search_frame_drop_penalty * float(
            np.mean(age > config.search_frame_deadline_ms)
        )
    return pressure


def dropped_count(obs: dict) -> int:
    """Count targets that have been dropped by the scheduler."""
    active = np.asarray(obs["active_mask"], dtype=bool)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    return int((active & (deadline < 0.0)).sum())


def _target_pressure(desired: float, deadline: float) -> float:
    late = float(np.clip(-desired / 1000.0, 0.0, 4.0))
    risk = float(np.clip((500.0 - deadline) / 500.0, 0.0, 2.0))
    dropped = 1.0 if deadline < 0.0 else 0.0
    return late + 2.0 * risk + 8.0 * dropped


def serviced_pressure_improvement(
    before: dict, after: dict, row: int, duration_ms: float
) -> float:
    """Return the pressure removed by servicing one target."""
    if int(row) <= 0:
        return 0.0
    idx = int(row) - 1
    active_before = np.asarray(before["active_mask"], dtype=bool)
    active_after = np.asarray(after["active_mask"], dtype=bool)
    if idx >= active_before.size or not active_before[idx] or not active_after[idx]:
        return 0.0
    desired_before = float(np.asarray(before["t_desired"])[idx])
    deadline_before = float(np.asarray(before["t_deadline"])[idx])
    desired_after = float(np.asarray(after["t_desired"])[idx])
    deadline_after = float(np.asarray(after["t_deadline"])[idx])
    if deadline_before < 0.0 or deadline_after < 0.0:
        return 0.0
    natural = _target_pressure(
        desired_before - duration_ms, deadline_before - duration_ms
    )
    updated = _target_pressure(desired_after, deadline_after)
    return max(0.0, natural - updated)


def shaped_transition_reward(
    base_reward: float,
    duration_ms: float,
    before: dict,
    after: dict,
    row: int,
    config: RewardConfig,
) -> float:
    """Canonical service reward used by the reference single-sensor harness."""
    reward = float(base_reward)
    pressure_delta = max(0.0, service_pressure(before) - service_pressure(after))
    frame_delta = max(
        0.0,
        search_frame_pressure(before, config) - search_frame_pressure(after, config),
    )
    reward += config.service_pressure_delta_reward_weight * pressure_delta
    reward += config.search_frame_delta_reward_weight * frame_delta
    reward -= (
        config.search_frame_state_penalty_weight
        * search_frame_pressure(after, config)
        * max(0.0, duration_ms)
        / 200.0
    )
    reward += config.serviced_pressure_improvement_reward_weight * (
        serviced_pressure_improvement(before, after, row, duration_ms)
    )
    discovered = max(
        0,
        int(np.sum(np.asarray(after["active_mask"], dtype=bool)))
        - int(np.sum(np.asarray(before["active_mask"], dtype=bool))),
    )
    reward += config.discovered_target_reward * discovered
    if config.tracked_count_delta_reward_weight:
        tracked_before = np.asarray(before["active_mask"], dtype=bool) & (
            np.asarray(before["t_deadline"], dtype=np.float32) >= 0.0
        )
        tracked_after = np.asarray(after["active_mask"], dtype=bool) & (
            np.asarray(after["t_deadline"], dtype=np.float32) >= 0.0
        )
        reward += config.tracked_count_delta_reward_weight * (
            int(tracked_after.sum()) - int(tracked_before.sum())
        )
    if config.tracked_target_ms_reward_weight and duration_ms > 0.0:
        tracked_after = np.asarray(after["active_mask"], dtype=bool) & (
            np.asarray(after["t_deadline"], dtype=np.float32) >= 0.0
        )
        reward += (
            config.tracked_target_ms_reward_weight
            * duration_ms
            / 1000.0
            * int(tracked_after.sum())
        )
    return float(reward)


def transition_reward(before: dict, after: dict, row: int, config: RewardConfig) -> float:
    """Mirror the live single-sensor reward for one deterministic shadow step."""
    duration = 10.0 if int(row) == 0 else float(np.asarray(before["t_dwell"])[int(row) - 1])
    base = config.search_action_reward if int(row) == 0 else config.track_action_reward
    active = np.asarray(before["active_mask"], dtype=bool)
    desired_before = np.asarray(before["t_desired"], dtype=np.float32)
    deadline_before = np.asarray(before["t_deadline"], dtype=np.float32)
    desired_after = np.asarray(after["t_desired"], dtype=np.float32)
    deadline_after = np.asarray(after["t_deadline"], dtype=np.float32)
    tracked_before = active & (deadline_before >= 0.0)

    denominator_before = np.maximum(1e-3, deadline_before - desired_before)
    denominator_after = np.maximum(1e-3, deadline_after - desired_after)
    tardiness_before = np.clip(
        np.maximum(0.0, -desired_before) / denominator_before, 0.0, 1.0
    )
    tardiness_after = np.clip(
        np.maximum(0.0, -desired_after) / denominator_after, 0.0, 1.0
    )
    overdue_increment = np.maximum(
        0.0,
        tardiness_after[tracked_before] - tardiness_before[tracked_before],
    ).sum()
    newly_dropped = tracked_before & (deadline_after < 0.0)
    reward = (
        base
        - float(overdue_increment)
        - config.drop_penalty * int(newly_dropped.sum())
    )

    grid = np.asarray(after.get("grid", []), dtype=np.float32)
    if (
        grid.size
        and config.search_frame_overdue_weight > 0.0
        and config.search_frame_deadline_ms > config.search_frame_desired_ms
    ):
        age = 3000.0 - grid
        overdue = np.maximum(0.0, age - config.search_frame_desired_ms)
        scale = config.search_frame_deadline_ms - config.search_frame_desired_ms
        frame_cost = np.clip(overdue / scale, 0.0, 1.0).sum()
        reward -= (
            config.search_frame_overdue_weight
            * float(frame_cost)
            * max(0.0, duration)
            / 200.0
        )
    return float(reward)
