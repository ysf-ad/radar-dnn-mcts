from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repaired_campaign_tools import build_env, env_preset_cfg, get_obs
from strict_window_report import execute_plan_until_budget, sample_state_metrics, summarize_fixed, summarize_sliding

ROOT = Path(__file__).resolve().parent
RES = ROOT / "CreateValid1" / "results"
RES.mkdir(parents=True, exist_ok=True)

MAXT = 100
PRESET = "repaired_stress"
BASE_ENV = env_preset_cfg(PRESET)


def seedall(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_with(**kw):
    cfg = dict(BASE_ENV)
    cfg.update(kw)
    return cfg


def run_fixed(planner, planner_name, initial_targets, max_trackers, seed, num_windows, window_ms, env_cfg):
    eng = build_env(planner, initial_targets, max_trackers, seed, window_ms, env_cfg)
    eng.reset(seed=seed)
    search_debt_ms = 0.0
    cumulative_reward = 0.0
    window_rows = []
    action_rows = []
    if hasattr(planner, "warmup"):
        planner.warmup(get_obs(eng, search_debt_ms), budget_ms=window_ms)
    for window_idx in range(num_windows):
        if eng.term_buf[0]:
            break
        obs = get_obs(eng, search_debt_ms)
        t0 = time.perf_counter()
        plan = planner.plan(obs, budget_ms=window_ms)
        plan_ms = (time.perf_counter() - t0) * 1000.0
        reward, spent_ms, search_debt_ms, executed, search_actions, arows = execute_plan_until_budget(
            eng, plan, float(window_ms), search_debt_ms, planner_name, seed, window_idx
        )
        cumulative_reward += reward
        state = sample_state_metrics(eng, search_debt_ms)
        window_rows.append({
            "planner": planner_name,
            "seed": seed,
            "window": window_idx,
            "elapsed_ms": float((window_idx + 1) * window_ms),
            "window_reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "search_fraction": float(search_actions / max(1, executed)),
            "planning_ms_per_decision": float(plan_ms),
            "planning_ms_per_executed_action": float(plan_ms / max(1, executed)),
            "executed_actions": int(executed),
            "spent_ms": float(spent_ms),
            **state,
        })
        for row in arows:
            row.update(window=window_idx, elapsed_ms=float((window_idx + 1) * window_ms))
        action_rows.extend(arows)
    eng.close()
    return pd.DataFrame(window_rows), pd.DataFrame(action_rows)


def summarize_window_df(df, mode="fixed", total_time_ms=None):
    if df.empty:
        return {}
    if mode == "fixed":
        s = summarize_fixed(df)
    else:
        s = summarize_sliding(df, int(total_time_ms))
    s["final_active_targets"] = float(df["active_targets"].iloc[-1])
    s["final_tracked_targets"] = float(df["tracked_targets"].iloc[-1])
    s["final_cumulative_reward"] = float(df["cumulative_reward"].iloc[-1])
    return s
