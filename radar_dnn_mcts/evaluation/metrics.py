from __future__ import annotations

import numpy as np
import pandas as pd


def observation_metrics(obs: dict) -> dict[str, float]:
    active = np.asarray(obs["active_mask"], dtype=bool)
    deadline = np.asarray(obs["t_deadline"], dtype=np.float32)
    desired = np.asarray(obs["t_desired"], dtype=np.float32)
    tracked = active & (deadline >= 0.0)
    dropped = active & ~tracked
    delays = np.maximum(0.0, -desired[active])
    return {
        "active_targets": int(active.sum()),
        "tracked_targets": int(tracked.sum()),
        "dropped_targets": int(dropped.sum()),
        "drop_pct_active": 100.0 * float(dropped.sum()) / max(1, int(active.sum())),
        "mean_delay_active": float(delays.mean()) if delays.size else 0.0,
    }


def summarize_windows(windows: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "initial_targets", "arrival_rate", "seed"]
    grouped = windows.groupby(keys, as_index=False)
    summary = grouped.agg(
        reward_per_window=("reward", "mean"),
        final_cumulative_reward=("cumulative_reward", "last"),
        dropped_targets=("dropped_targets", "mean"),
        drop_pct_active=("drop_pct_active", "mean"),
        tracked_targets=("tracked_targets", "mean"),
        active_targets=("active_targets", "mean"),
        mean_delay_active=("mean_delay_active", "mean"),
        search_fraction=("search_fraction", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
        latency_ms_median=("latency_ms", "median"),
    )
    p90 = grouped["latency_ms"].quantile(0.9).rename(columns={"latency_ms": "latency_ms_p90"})
    return summary.merge(p90, on=keys, how="left")


def aggregate_methods(windows: pd.DataFrame) -> pd.DataFrame:
    return windows.groupby("method", as_index=False).agg(
        reward_per_window=("reward", "mean"),
        drop_pct_active=("drop_pct_active", "mean"),
        dropped_targets=("dropped_targets", "mean"),
        tracked_targets=("tracked_targets", "mean"),
        active_targets=("active_targets", "mean"),
        mean_delay_active=("mean_delay_active", "mean"),
        search_fraction=("search_fraction", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
        windows=("window", "count"),
    )
