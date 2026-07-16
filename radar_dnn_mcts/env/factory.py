from __future__ import annotations

from pufferlib.ocean.radarxs.engine import RadarEngine

from radar_dnn_mcts.env.config import RewardConfig


def create_engine(
    planner,
    *,
    initial_targets: int,
    max_targets: int,
    seed: int,
    arrival_rate: float,
    window_ms: float = 200.0,
    dual_sensor: bool = False,
    reward: RewardConfig | None = None,
) -> RadarEngine:
    """Construct the single canonical simulator used by training and evaluation."""
    reward = reward or RewardConfig()
    return RadarEngine(
        planner,
        initial_targets=initial_targets,
        max_trackers=max_targets,
        seed=seed,
        window_ms=int(window_ms),
        enable_global_delay=True,
        enable_local_delay=False,
        enable_x_band=dual_sensor,
        enable_search_refresh_tracked=False,
        search_refresh_gain=0.0,
        enable_priority=reward.enable_priority,
        enable_poisson_arrivals=True,
        activate_all_targets_without_poisson=True,
        poisson_rate_per_second=arrival_rate,
        search_action_reward=reward.search_action_reward,
        track_update_reward=reward.track_action_reward,
        track_loss_penalty=reward.drop_penalty,
        target_service_weight=0.0,
        target_service_horizon_ms=3000.0,
        sector_staleness_weight=0.0,
        searched_sector_reward_weight=0.0,
        search_frame_overdue_weight=reward.search_frame_overdue_weight,
        search_frame_desired_ms=reward.search_frame_desired_ms,
        search_frame_deadline_ms=reward.search_frame_deadline_ms,
        search_frame_drop_penalty=reward.search_frame_drop_penalty,
        search_task_cost_mode=1,
        revisit_time_scale=0.75,
        penalize_hidden_targets=reward.penalize_hidden_targets,
        episode_time_limit_ms=2_000_000_000,
        search_delay_mode=0,
        search_debt_penalty_weight=0.0,
        search_debt_tau_ms=200.0,
        search_delay_penalty_cap=-1.0,
    )
