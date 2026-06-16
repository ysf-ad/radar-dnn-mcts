from __future__ import annotations

from pathlib import Path

from final_radar_campaign import RES, env_with

OUT = RES / "load_adaptive"
OUT.mkdir(parents=True, exist_ok=True)

BASE_ENV = dict(
    search_delay_mode=0,
    search_debt_penalty_weight=0.001,
    search_delay_penalty_cap=-1.0,
    enable_search_refresh_tracked=0,
    search_refresh_gain=0.0,
    search_action_reward=0.0,
    penalize_hidden_targets=1,
    track_loss_penalty=8.0,
)


def make_env(rate):
    return env_with(poisson_rate_per_second=float(rate), **BASE_ENV)
