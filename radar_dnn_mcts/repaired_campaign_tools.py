import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pufferlib.ocean.radarxs import binding, engine
from pufferlib.ocean.radarxs.models.edf import EDFPlanner
from pufferlib.ocean.radarxs.models.est import ESTPlanner
from pufferlib.ocean.radarxs.models.mcts import MCTSPlanner
from pufferlib.ocean.radarxs.models.transformer_mcts_policy import (
    PolicyOnlyMCTSPlanner,
    PolicyOnlyTransformer,
    PolicyValueTransformer,
)


def configure_fast_cpu_inference():
    """Small transformer inference is faster with low thread fanout on CPU."""
    threads = int(os.environ.get("RADARXS_TORCH_THREADS", "1"))
    if threads > 0:
        torch.set_num_threads(threads)


def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def env_preset_cfg(preset: str, revisit_time_scale_override: Optional[float] = None) -> Dict[str, float]:
    presets = {
        "repaired_stress": dict(
            enable_global_delay=1,
            enable_local_delay=0,
            enable_x_band=0,
            enable_search_refresh_tracked=1,
            search_refresh_gain=0.75,
            enable_priority=0,
            enable_poisson_arrivals=1,
            activate_all_targets_without_poisson=1,
            poisson_rate_per_second=2.0,
            search_action_reward=0.08,
            track_update_reward=0.30,
            track_loss_penalty=1.0,
            track_urgency_bonus_weight=0.0,
            sector_staleness_weight=0.0,
            revisit_time_scale=0.75,
            penalize_hidden_targets=0,
            enable_track_beam_scan=0,
            episode_time_limit_ms=2_000_000_000,
            search_delay_mode=1,
            search_debt_penalty_weight=0.058,
            search_debt_tau_ms=200.0,
            search_delay_penalty_cap=2.0,
        ),
        "v39_legacy_reconstructed": dict(
            enable_global_delay=0,
            enable_local_delay=1,
            enable_x_band=0,
            enable_search_refresh_tracked=1,
            search_refresh_gain=1.0,
            enable_priority=0,
            enable_poisson_arrivals=1,
            activate_all_targets_without_poisson=1,
            poisson_rate_per_second=2.0,
            search_action_reward=0.10,
            track_update_reward=0.10,
            track_loss_penalty=1.0,
            track_urgency_bonus_weight=0.0,
            sector_staleness_weight=0.0,
            revisit_time_scale=0.75,
            penalize_hidden_targets=0,
            enable_track_beam_scan=0,
            episode_time_limit_ms=2_000_000_000,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0001,
            search_debt_tau_ms=10.0,
            search_delay_penalty_cap=-1.0,
        ),
        "operational_linear_staleness": dict(
            enable_global_delay=1,
            enable_local_delay=0,
            enable_x_band=0,
            enable_search_refresh_tracked=1,
            search_refresh_gain=1.0,
            enable_priority=0,
            enable_poisson_arrivals=1,
            activate_all_targets_without_poisson=1,
            poisson_rate_per_second=2.0,
            search_action_reward=0.10,
            track_update_reward=0.10,
            track_loss_penalty=1.0,
            track_urgency_bonus_weight=0.0,
            sector_staleness_weight=0.001,
            revisit_time_scale=0.75,
            penalize_hidden_targets=0,
            enable_track_beam_scan=0,
            episode_time_limit_ms=2_000_000_000,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0,
            search_debt_tau_ms=10.0,
            search_delay_penalty_cap=-1.0,
        ),
        "original_like": dict(
            enable_global_delay=0,
            enable_local_delay=1,
            enable_x_band=0,
            enable_search_refresh_tracked=1,
            search_refresh_gain=1.0,
            enable_priority=0,
            enable_poisson_arrivals=0,
            activate_all_targets_without_poisson=1,
            poisson_rate_per_second=0.0,
            search_action_reward=0.10,
            track_update_reward=0.10,
            track_loss_penalty=1.0,
            track_urgency_bonus_weight=0.0,
            sector_staleness_weight=0.0,
            revisit_time_scale=1.0,
            penalize_hidden_targets=0,
            enable_track_beam_scan=0,
            episode_time_limit_ms=2_000_000_000,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0001,
            search_debt_tau_ms=10.0,
            search_delay_penalty_cap=-1.0,
        ),
        "original_semantics": dict(
            enable_global_delay=0,
            enable_local_delay=1,
            enable_x_band=0,
            enable_search_refresh_tracked=1,
            search_refresh_gain=1.0,
            enable_priority=0,
            enable_poisson_arrivals=0,
            activate_all_targets_without_poisson=0,
            poisson_rate_per_second=0.0,
            search_action_reward=0.10,
            track_update_reward=0.10,
            track_loss_penalty=1.0,
            track_urgency_bonus_weight=0.0,
            sector_staleness_weight=0.0,
            revisit_time_scale=1.0,
            penalize_hidden_targets=0,
            enable_track_beam_scan=0,
            episode_time_limit_ms=2_000_000_000,
            search_delay_mode=0,
            search_debt_penalty_weight=0.0001,
            search_debt_tau_ms=10.0,
            search_delay_penalty_cap=-1.0,
        ),
    }
    if preset not in presets:
        raise ValueError(f"Unsupported env preset: {preset}")
    cfg = dict(presets[preset])
    if revisit_time_scale_override is not None:
        cfg["revisit_time_scale"] = float(revisit_time_scale_override)
    return cfg


def planner_delay_cfg(preset: str) -> Dict[str, float]:
    if preset in {"original_like", "original_semantics"}:
        return dict(
            tardiness_mode="hybrid",
            local_tardiness_weight=1.0,
            global_tardiness_weight=1.0,
            normalize_delay_penalty=False,
            global_aggregation="sum",
            tardiness_accounting="legacy",
            settle_rollout_debt=False,
        )
    return dict(
        tardiness_mode="global",
        local_tardiness_weight=0.0,
        global_tardiness_weight=1.0,
        normalize_delay_penalty=True,
        global_aggregation="sum",
        tardiness_accounting="legacy",
        settle_rollout_debt=False,
    )


def get_obs(eng, search_debt_ms: float = 0.0):
    obs = engine.get_obs_from_buf(eng.obs_buf, max_trackers=eng.max_trackers)
    if hasattr(binding, "vec_aux"):
        try:
            aux = binding.vec_aux(eng.env)
            obs["s_band_busy_ms"] = float(aux.get("s_band_busy_ms", 0.0))
            obs["x_band_busy_ms"] = float(aux.get("x_band_busy_ms", 0.0))
            obs["enable_x_band"] = int(aux.get("enable_x_band", 0))
            obs["target_range"] = np.asarray(aux.get("target_range", []), dtype=np.float32)
        except Exception:
            pass
    obs["search_debt_ms"] = float(search_debt_ms)
    return obs


def infer_elapsed_ms(obs_before, obs_after) -> float:
    candidates = []
    grid_delta = obs_before["grid"] - obs_after["grid"]
    if np.any(grid_delta > 0):
        candidates.append(float(np.max(grid_delta)))
    mask = obs_before["active_mask"] & obs_after["active_mask"]
    if np.any(mask):
        desired_delta = obs_before["t_desired"][mask] - obs_after["t_desired"][mask]
        deadline_delta = obs_before["t_deadline"][mask] - obs_after["t_deadline"][mask]
        if np.any(desired_delta > 0):
            candidates.append(float(np.max(desired_delta)))
        if np.any(deadline_delta > 0):
            candidates.append(float(np.max(deadline_delta)))
    return max(candidates) if candidates else 0.0


SEARCH_DWELL_MS = 10.0  # matches SEARCH_DWELL_TIME in radarxs.h

def decode_sensor_action(action: int, max_trackers: int) -> Tuple[int, Optional[int]]:
    """Return (logical_action, requested_sensor), where sensor is None/S=0/X=1."""
    a = int(action)
    if a == int(max_trackers) + 1 or a == int(max_trackers) + 2:
        return -1, None
    s_search = int(max_trackers) + 3
    x_search = int(max_trackers) + 4
    s_track_base = int(max_trackers) + 5
    x_track_base = int(max_trackers) + 5 + int(max_trackers)
    if a == s_search:
        return 0, 0
    if a == x_search:
        return 0, 1
    if s_track_base <= a < s_track_base + int(max_trackers):
        return (a - s_track_base) + 1, 0
    if x_track_base <= a < x_track_base + int(max_trackers):
        return (a - x_track_base) + 1, 1
    return a, None

def execute_first_valid_action(eng, plan, remaining_ms: float):
    if eng.term_buf[0] or remaining_ms <= 0:
        return 0.0, 0.0, None
    for a in plan:
        obs_before = get_obs(eng)
        logical_action, requested_sensor = decode_sensor_action(int(a), eng.max_trackers)
        if logical_action > 0:
            idx = int(logical_action) - 1
            if idx < 0 or idx >= len(obs_before["active_mask"]) or not obs_before["active_mask"][idx]:
                continue
            # Skip targets whose deadline has already passed (is_tracked=False in C but
            # active_mask is still True because the observation buffer is never cleared).
            # Scheduling such a target gives reward=-track_loss_penalty with dt=0,
            # causing the window loop to break without accumulating any useful reward.
            if obs_before["t_deadline"][idx] < 0:
                continue
            if requested_sensor is not None and "target_range" in obs_before:
                rng = float(obs_before["target_range"][idx])
                if requested_sensor == 0 and not (obs_before.get("s_band_busy_ms", 0.0) <= 0.0 and 10_000_000.0 < rng < 184_000_000.0):
                    continue
                if requested_sensor == 1 and not (
                    obs_before.get("enable_x_band", 0)
                    and obs_before.get("x_band_busy_ms", 0.0) <= 0.0
                    and 5_000_000.0 < rng < 100_000_000.0
                ):
                    continue
        elif logical_action == 0 and requested_sensor is not None:
            if requested_sensor == 0 and obs_before.get("s_band_busy_ms", 0.0) > 0.0:
                continue
            if requested_sensor == 1 and not (obs_before.get("enable_x_band", 0) and obs_before.get("x_band_busy_ms", 0.0) <= 0.0):
                continue
        eng.act_buf[0] = int(a)
        binding.vec_step(eng.env)
        obs_after = get_obs(eng)
        if logical_action == 0:
            # Search always takes SEARCH_DWELL_TIME ms regardless of observation changes.
            # infer_elapsed_ms returns 0 for search because sector freshness increases
            # (obs_before - obs_after < 0), causing the window loop to break prematurely.
            dt = SEARCH_DWELL_MS
        else:
            dt = infer_elapsed_ms(obs_before, obs_after)
            if logical_action < 0 and dt <= 0.0:
                dt = SEARCH_DWELL_MS
            if logical_action > 0 and dt <= 0.0:
                continue
        return float(eng.rew_buf[0]), float(dt), int(a)
    return 0.0, 0.0, None


def build_env(planner, initial_targets: int, max_trackers: int, seed: int, window_ms: int, env_cfg: Dict[str, float]):
    return engine.RadarEngine(
        planner=planner,
        initial_targets=initial_targets,
        max_trackers=max_trackers,
        seed=seed,
        window_ms=window_ms,
        **env_cfg,
    )


def make_reference_planner(
    max_trackers: int,
    rollouts: int,
    c_value: float,
    env_cfg: Dict[str, float],
    preset: str,
    simulation_window_ms: float = 200.0,
):
    delay_cfg = planner_delay_cfg(preset)
    planner_search_delay_mode = int(env_cfg.get("planner_search_delay_mode", env_cfg["search_delay_mode"]))
    planner_search_debt_penalty_weight = float(env_cfg.get("planner_search_debt_penalty_weight", env_cfg["search_debt_penalty_weight"]))
    planner_search_debt_tau_ms = float(env_cfg.get("planner_search_debt_tau_ms", env_cfg["search_debt_tau_ms"]))
    planner_search_delay_penalty_cap = float(env_cfg.get("planner_search_delay_penalty_cap", env_cfg["search_delay_penalty_cap"]))
    return MCTSPlanner(
        max_trackers=max_trackers,
        num_rollouts=rollouts,
        exploration_constant=c_value,
        **delay_cfg,
        enable_search_refresh_tracked=bool(env_cfg["enable_search_refresh_tracked"]),
        search_refresh_gain=float(env_cfg["search_refresh_gain"]),
        search_action_reward=float(env_cfg["search_action_reward"]),
        track_update_reward=float(env_cfg["track_update_reward"]),
        track_loss_penalty=float(env_cfg["track_loss_penalty"]),
        track_urgency_bonus_weight=float(env_cfg["track_urgency_bonus_weight"]),
        target_service_weight=float(env_cfg.get("target_service_weight", 0.0)),
        target_service_horizon_ms=float(env_cfg.get("target_service_horizon_ms", 1000.0)),
        sector_staleness_weight=float(env_cfg["sector_staleness_weight"]),
        searched_sector_reward_weight=float(env_cfg.get("searched_sector_reward_weight", 0.0)),
        search_frame_overdue_weight=float(env_cfg.get("search_frame_overdue_weight", 0.0)),
        search_frame_desired_ms=float(env_cfg.get("search_frame_desired_ms", 3000.0)),
        search_frame_deadline_ms=float(env_cfg.get("search_frame_deadline_ms", 4500.0)),
        search_frame_drop_penalty=float(env_cfg.get("search_frame_drop_penalty", 0.0)),
        enable_track_beam_scan=bool(env_cfg["enable_track_beam_scan"]),
        revisit_time_scale=float(env_cfg["revisit_time_scale"]),
        search_delay_mode=planner_search_delay_mode,
        search_debt_penalty_weight=planner_search_debt_penalty_weight,
        search_debt_tau_ms=planner_search_debt_tau_ms,
        search_delay_penalty_cap=planner_search_delay_penalty_cap,
        penalize_hidden_targets=bool(env_cfg.get("penalize_hidden_targets", False)),
        simulation_window_ms=float(simulation_window_ms),
    )


def load_student_model(
    ckpt: str,
    max_trackers: int,
    device: str,
    policy_head_type: str = "linear",
    value_head_use_tanh: bool = True,
    q_head_use_tanh: bool = True,
):
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    if dev.type == "cpu":
        configure_fast_cpu_inference()
    state = torch.load(ckpt, map_location=dev)
    use_value_head = any(k.startswith("value_head.") for k in state.keys()) or any(k.startswith("q_head.") for k in state.keys())
    model_cls = PolicyValueTransformer if use_value_head else PolicyOnlyTransformer
    model_kwargs = dict(num_tasks=max_trackers + 1, policy_head_type=policy_head_type)
    if model_cls is PolicyValueTransformer:
        model_kwargs.update(
            value_head_use_tanh=bool(value_head_use_tanh),
            q_head_use_tanh=bool(q_head_use_tanh),
        )
    model = model_cls(**model_kwargs).to(dev)
    model_state = model.state_dict()
    model_state.update({k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape})
    model.load_state_dict(model_state)
    model.eval()
    return model


def make_student(
    model,
    max_trackers: int,
    rollouts: int,
    c_value: float,
    device: str,
    search_prior_scale: float,
    env_cfg: Dict[str, float],
    preset: str,
    expand_top_k: int,
    action_select_mode: str = "value",
    force_search_debt_ms: float = -1.0,
    use_value_head: bool = False,
    value_scale: float = 20.0,
    leaf_value_mix: float = 1.0,
    use_q_head: bool = False,
    q_scale: float = 20.0,
    q_utility_weight: float = 0.0,
    track_uncertainty_bonus_weight: float = 0.0,
    rollout_candidate_cap: int = 96,
    root_search_strategy: str = "puct",
    gumbel_considered_actions: int = 8,
    selection_q_mode: str = "raw",
    value_utility_weight: float = 0.0,
    completed_q_weight: float = 1.0,
    completed_q_transform: str = "minmax",
    completed_q_expand_all_root: bool = False,
    search_macro_len: int = 1,
    search_macro_min_margin: float = 0.0,
    value_head_use_tanh: bool = True,
    q_head_use_tanh: bool = True,
    policy_head_type: str = "linear",
    puct_parent_visits_power: float = 0.5,
    simulation_window_ms: float = 200.0,
):
    delay_cfg = planner_delay_cfg(preset)
    planner_search_delay_mode = int(env_cfg.get("planner_search_delay_mode", env_cfg["search_delay_mode"]))
    planner_search_debt_penalty_weight = float(env_cfg.get("planner_search_debt_penalty_weight", env_cfg["search_debt_penalty_weight"]))
    planner_search_debt_tau_ms = float(env_cfg.get("planner_search_debt_tau_ms", env_cfg["search_debt_tau_ms"]))
    planner_search_delay_penalty_cap = float(env_cfg.get("planner_search_delay_penalty_cap", env_cfg["search_delay_penalty_cap"]))
    return PolicyOnlyMCTSPlanner(
        model=model,
        max_trackers=max_trackers,
        num_rollouts=rollouts,
        exploration_constant=c_value,
        device=device,
        ucb_mode="additive",
        **delay_cfg,
        enable_search_refresh_tracked=bool(env_cfg["enable_search_refresh_tracked"]),
        search_refresh_gain=float(env_cfg["search_refresh_gain"]),
        search_action_reward=float(env_cfg["search_action_reward"]),
        track_update_reward=float(env_cfg["track_update_reward"]),
        track_loss_penalty=float(env_cfg["track_loss_penalty"]),
        track_urgency_bonus_weight=float(env_cfg["track_urgency_bonus_weight"]),
        track_uncertainty_bonus_weight=float(track_uncertainty_bonus_weight),
        target_service_weight=float(env_cfg.get("target_service_weight", 0.0)),
        target_service_horizon_ms=float(env_cfg.get("target_service_horizon_ms", 1000.0)),
        sector_staleness_weight=float(env_cfg["sector_staleness_weight"]),
        enable_track_beam_scan=bool(env_cfg["enable_track_beam_scan"]),
        revisit_time_scale=float(env_cfg["revisit_time_scale"]),
        search_delay_mode=planner_search_delay_mode,
        search_debt_penalty_weight=planner_search_debt_penalty_weight,
        search_debt_tau_ms=planner_search_debt_tau_ms,
        search_delay_penalty_cap=planner_search_delay_penalty_cap,
        penalize_hidden_targets=bool(env_cfg.get("penalize_hidden_targets", False)),
        training_mode=False,
        expand_top_k=int(expand_top_k),
        action_select_mode=str(action_select_mode),
        search_prior_scale=search_prior_scale,
        force_search_debt_ms=float(force_search_debt_ms),
        use_value_head=bool(use_value_head),
        value_scale=float(value_scale),
        leaf_value_mix=float(leaf_value_mix),
        use_q_head=bool(use_q_head),
        q_scale=float(q_scale),
        q_utility_weight=float(q_utility_weight),
        rollout_candidate_cap=int(rollout_candidate_cap),
        root_search_strategy=str(root_search_strategy),
        gumbel_considered_actions=int(gumbel_considered_actions),
        selection_q_mode=str(selection_q_mode),
        value_utility_weight=float(value_utility_weight),
        completed_q_weight=float(completed_q_weight),
        completed_q_transform=str(completed_q_transform),
        completed_q_expand_all_root=bool(completed_q_expand_all_root),
        search_macro_len=int(search_macro_len),
        search_macro_min_margin=float(search_macro_min_margin),
        value_head_use_tanh=bool(value_head_use_tanh),
        q_head_use_tanh=bool(q_head_use_tanh),
        policy_head_type=str(policy_head_type),
        puct_parent_visits_power=float(puct_parent_visits_power),
        simulation_window_ms=float(simulation_window_ms),
    )


def run_episode(planner, initial_targets: int, max_trackers: int, seed: int, num_windows: int, window_ms: int, env_cfg: Dict[str, float]):
    eng = build_env(planner, initial_targets, max_trackers, seed, window_ms, env_cfg)
    eng.reset(seed=seed)

    rows = []
    cumulative_reward = 0.0
    total_search = 0
    total_actions = 0
    action_latencies = []
    search_debt_ms = 0.0

    for window_idx in range(num_windows):
        window_reward = 0.0
        remaining_ms = float(window_ms)

        while remaining_ms > 0 and not eng.term_buf[0]:
            obs = get_obs(eng, search_debt_ms)
            t0 = time.perf_counter()
            plan = planner.plan(obs, budget_ms=int(max(1.0, remaining_ms)))
            plan_ms = (time.perf_counter() - t0) * 1000.0
            reward, dt, action = execute_first_valid_action(eng, plan, remaining_ms)
            if action is None or dt <= 0:
                break
            window_reward += reward
            remaining_ms -= dt
            total_search += int(action == 0)
            total_actions += 1
            action_latencies.append(plan_ms)
            if action == 0:
                search_debt_ms = 0.0
            else:
                search_debt_ms += max(dt, 0.0)

        cumulative_reward += window_reward
        obs_now = get_obs(eng, search_debt_ms)
        active_mask = obs_now["active_mask"]
        delay_per_active = float(np.mean(np.maximum(0.0, -obs_now["t_desired"][active_mask]))) if np.any(active_mask) else 0.0
        active_targets = float(np.sum(active_mask))

        rows.append(
            {
                "window": int(window_idx + 1),
                "window_reward": float(window_reward),
                "cumulative_reward": float(cumulative_reward),
                "delay_per_active": delay_per_active,
                "active_targets": active_targets,
                "search_fraction_running": float(total_search / max(1, total_actions)),
                "planner_latency_ms_running": float(np.mean(action_latencies)) if action_latencies else 0.0,
            }
        )
        if eng.term_buf[0]:
            break

    eng.close()
    df = pd.DataFrame(rows)
    return {
        "mean_window_reward": float(df["window_reward"].mean()) if not df.empty else 0.0,
        "mean_delay_per_active": float(df["delay_per_active"].mean()) if not df.empty else 0.0,
        "mean_active_targets": float(df["active_targets"].mean()) if not df.empty else 0.0,
        "mean_search_fraction": float(df["search_fraction_running"].mean()) if not df.empty else 0.0,
        "planner_latency_ms": float(df["planner_latency_ms_running"].mean()) if not df.empty else 0.0,
        "windows": int(len(df)),
    }, rows


def summarize_results(raw_rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw_rows)
    return (
        df.groupby(["planner", "task"], as_index=False)
        .agg(
            mean_window_reward=("mean_window_reward", "mean"),
            mean_delay_per_active=("mean_delay_per_active", "mean"),
            mean_active_targets=("mean_active_targets", "mean"),
            mean_search_fraction=("mean_search_fraction", "mean"),
            planner_latency_ms=("planner_latency_ms", "mean"),
        )
    )


def pick_best_candidate(
    summary: pd.DataFrame,
    reward_col: str = "mean_window_reward",
    low_load_tasks: Optional[List[int]] = None,
    search_fraction_bounds: Optional[Tuple[float, float]] = None,
    tie_pct: float = 0.01,
):
    candidate_rows = []
    for planner_name, grp in summary.groupby("planner"):
        overall_reward = float(grp[reward_col].mean())
        overall_latency = float(grp["planner_latency_ms"].mean())
        constraint_ok = True
        if low_load_tasks and search_fraction_bounds is not None:
            for task in low_load_tasks:
                task_rows = grp[grp["task"] == task]
                if task_rows.empty:
                    constraint_ok = False
                    break
                sf = float(task_rows["mean_search_fraction"].iloc[0])
                if sf < float(search_fraction_bounds[0]) or sf > float(search_fraction_bounds[1]):
                    constraint_ok = False
                    break
        candidate_rows.append(
            {
                "planner": planner_name,
                "overall_reward": overall_reward,
                "overall_latency": overall_latency,
                "constraint_ok": constraint_ok,
            }
        )
    cand = pd.DataFrame(candidate_rows)
    feasible = cand[cand["constraint_ok"]].copy()
    if feasible.empty:
        feasible = cand.copy()
    best_reward = float(feasible["overall_reward"].max())
    reward_tol = max(1.0, abs(best_reward)) * float(tie_pct)
    feasible["within_tie"] = feasible["overall_reward"] >= (best_reward - reward_tol)
    feasible = feasible.sort_values(["within_tie", "overall_reward", "overall_latency"], ascending=[False, False, True]).reset_index(drop=True)
    best = feasible.iloc[0].to_dict()
    return cand, best


def evaluate_Reference_candidates(
    tasks: List[int],
    seeds: List[int],
    windows: int,
    window_ms: int,
    max_trackers: int,
    env_cfg: Dict[str, float],
    preset: str,
):
    candidates = [(64, 0.5), (64, 1.0), (128, 1.0), (128, 2.0), (256, 2.0)]
    raw_rows = []
    total = len(tasks) * len(seeds) * len(candidates)
    counter = 0
    for rollouts, c_value in candidates:
        planner_name = f"Reference_r{rollouts}_c{c_value:g}"
        for task in tasks:
            for seed in seeds:
                counter += 1
                print(f"[{counter}/{total}] planner={planner_name} task={task} seed={seed}", flush=True)
                planner = make_reference_planner(max_trackers, rollouts, c_value, env_cfg, preset)
                summary, _ = run_episode(planner, task, max_trackers, seed, windows, window_ms, env_cfg)
                raw_rows.append({"planner": planner_name, "task": task, "seed": seed, **summary})
    raw_df = pd.DataFrame(raw_rows)
    summary_df = summarize_results(raw_rows)
    candidates_df, best = pick_best_candidate(summary_df, reward_col="mean_window_reward")
    return raw_df, summary_df, candidates_df, best


def run_reference_calibration(args):
    os.makedirs(args.out_dir, exist_ok=True)
    env_cfg = env_preset_cfg(args.env_preset, args.revisit_time_scale_override)
    tasks = parse_int_list(args.tasks)
    seeds = parse_int_list(args.seeds)
    raw_df, summary_df, candidates_df, best = evaluate_Reference_candidates(
        tasks, seeds, args.windows, args.window_ms, args.max_trackers, env_cfg, args.env_preset
    )
    raw_df.to_csv(os.path.join(args.out_dir, "raw.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    candidates_df.to_csv(os.path.join(args.out_dir, "selection_table.csv"), index=False)
    fallback_used = False
    best_task20 = summary_df[(summary_df["planner"] == best["planner"]) & (summary_df["task"] == 20)]
    best_task50 = summary_df[(summary_df["planner"] == best["planner"]) & (summary_df["task"] == 50)]
    if (
        args.env_preset == "original_like"
        and args.revisit_time_scale_override is None
        and not best_task20.empty
        and not best_task50.empty
        and float(best_task20["mean_search_fraction"].iloc[0]) > 0.8
        and float(best_task50["mean_search_fraction"].iloc[0]) > 0.8
    ):
        fallback_used = True
        fallback_cfg = env_preset_cfg(args.env_preset, 0.75)
        raw_fb, summary_fb, candidates_fb, best_fb = evaluate_Reference_candidates(
            tasks, seeds, args.windows, args.window_ms, args.max_trackers, fallback_cfg, args.env_preset
        )
        raw_fb.to_csv(os.path.join(args.out_dir, "raw_fallback.csv"), index=False)
        summary_fb.to_csv(os.path.join(args.out_dir, "summary_fallback.csv"), index=False)
        candidates_fb.to_csv(os.path.join(args.out_dir, "selection_table_fallback.csv"), index=False)
        best = dict(best_fb)
        best["fallback_used"] = True
        best["revisit_time_scale"] = 0.75
        with open(os.path.join(args.out_dir, "best_Reference_fallback.json"), "w") as f:
            json.dump(best, f, indent=2)
    else:
        best["fallback_used"] = False
        best["revisit_time_scale"] = float(env_cfg["revisit_time_scale"])
    best["env_preset"] = args.env_preset
    with open(os.path.join(args.out_dir, "BEST_REFERENCE.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("BEST_REFERENCE", json.dumps(best), flush=True)


def run_student_calibration(args):
    os.makedirs(args.out_dir, exist_ok=True)
    env_cfg = env_preset_cfg(args.env_preset, args.revisit_time_scale_override)
    tasks = parse_int_list(args.tasks)
    seeds = parse_int_list(args.seeds)
    rollouts = parse_int_list(args.rollouts)
    sps_values = parse_float_list(args.search_prior_scales)
    model = load_student_model(args.ckpt, args.max_trackers, args.device)

    raw_rows = []
    total = len(tasks) * len(seeds) * len(rollouts) * len(sps_values)
    counter = 0
    for r in rollouts:
        for sps in sps_values:
            planner_name = f"Student_r{r}_sps{sps:g}"
            for task in tasks:
                for seed in seeds:
                    counter += 1
                    print(f"[{counter}/{total}] planner={planner_name} task={task} seed={seed}", flush=True)
                    planner = make_student(
                        model,
                        args.max_trackers,
                        r,
                        args.student_c,
                        args.device,
                        sps,
                    env_cfg,
                    args.env_preset,
                        args.expand_top_k,
                    force_search_debt_ms=args.student_force_search_debt_ms,
                    use_value_head=bool(args.student_use_value_head),
                    value_scale=float(args.student_value_scale),
                    leaf_value_mix=float(args.student_leaf_value_mix),
                    rollout_candidate_cap=int(args.student_rollout_candidate_cap),
                    root_search_strategy=str(args.student_root_search_strategy),
                    gumbel_considered_actions=int(args.student_gumbel_considered_actions),
                )
                    summary, _ = run_episode(planner, task, args.max_trackers, seed, args.windows, args.window_ms, env_cfg)
                    raw_rows.append({"planner": planner_name, "task": task, "seed": seed, **summary})

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(os.path.join(args.out_dir, "raw.csv"), index=False)
    summary_df = summarize_results(raw_rows)
    summary_df.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    candidates_df, best = pick_best_candidate(summary_df, reward_col="mean_window_reward")
    best["env_preset"] = args.env_preset
    candidates_df.to_csv(os.path.join(args.out_dir, "selection_table.csv"), index=False)
    with open(os.path.join(args.out_dir, "best_student.json"), "w") as f:
        json.dump(best, f, indent=2)
    print("BEST_STUDENT", json.dumps(best), flush=True)


def save_task_plot(summary_df: pd.DataFrame, metric: str, ylabel: str, out_path: str):
    plt.figure(figsize=(9, 5.5))
    for planner_name, grp in summary_df.groupby("planner"):
        grp = grp.sort_values("task")
        plt.plot(grp["task"], grp[metric], marker="o", linewidth=2, label=planner_name)
    plt.xlabel("Initial Targets")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Task Count")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_window_plot(window_df: pd.DataFrame, task: int, metric: str, ylabel: str, out_path: str):
    plt.figure(figsize=(9, 5.5))
    focus = window_df[window_df["task"] == task].copy()
    for planner_name, grp in focus.groupby("planner"):
        grp = grp.groupby("window", as_index=False)[metric].mean().sort_values("window")
        plt.plot(grp["window"], grp[metric], marker="o", linewidth=2, label=planner_name)
    plt.xlabel("Window")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Window ({task} Targets)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def run_benchmark(args):
    os.makedirs(args.out_dir, exist_ok=True)
    env_cfg = env_preset_cfg(args.env_preset, args.revisit_time_scale_override)
    tasks = parse_int_list(args.tasks)
    seeds = parse_int_list(args.seeds)
    rep_tasks = parse_int_list(args.rep_tasks)
    # Load model once; architecture is flexible to any max_trackers at inference time
    model = load_student_model(args.ckpt, max(tasks), args.device)

    # Planner factories are now per-task so max_trackers matches the actual task size.
    def make_planners_for_task(task_mt):
        return [
            ("EDF", lambda mt=task_mt: EDFPlanner(max_trackers=mt)),
            ("EST", lambda mt=task_mt: ESTPlanner(max_trackers=mt)),
            (
                args.student_name,
                lambda mt=task_mt: make_student(
                    model,
                    mt,
                    args.student_rollouts,
                    args.student_c,
                    args.device,
                    args.student_search_prior_scale,
                    env_cfg,
                    args.env_preset,
                    args.expand_top_k,
                    force_search_debt_ms=args.student_force_search_debt_ms,
                    use_value_head=bool(args.student_use_value_head),
                    value_scale=float(args.student_value_scale),
                    leaf_value_mix=float(args.student_leaf_value_mix),
                    rollout_candidate_cap=int(args.student_rollout_candidate_cap),
                    root_search_strategy=str(args.student_root_search_strategy),
                    gumbel_considered_actions=int(args.student_gumbel_considered_actions),
                ),
            ),
        ]

    summary_rows = []
    window_rows = []
    # Build full list of (planner_name, task, seed) for progress counter
    planner_names = ["EDF", "EST", args.student_name]
    total = len(planner_names) * len(tasks) * len(seeds)
    counter = 0
    for planner_name in planner_names:
        for task in tasks:
            task_mt = task  # max_trackers = task so world has exactly 'task' targets
            planners_for_task = make_planners_for_task(task_mt)
            planner_factory = next(f for n, f in planners_for_task if n == planner_name)
            for seed in seeds:
                counter += 1
                print(f"[{counter}/{total}] planner={planner_name} task={task} seed={seed}", flush=True)
                planner = planner_factory()
                summary, rows = run_episode(planner, task, task_mt, seed, args.windows, args.window_ms, env_cfg)
                summary_rows.append({"planner": planner_name, "task": task, "seed": seed, **summary})
                for row in rows:
                    window_rows.append({"planner": planner_name, "task": task, "seed": seed, **row})

    raw_summary_df = pd.DataFrame(summary_rows)
    raw_summary_df.to_csv(os.path.join(args.out_dir, "raw_summary.csv"), index=False)
    raw_window_df = pd.DataFrame(window_rows)
    raw_window_df.to_csv(os.path.join(args.out_dir, "raw_windows.csv"), index=False)

    summary_df = summarize_results(summary_rows)
    summary_df.to_csv(os.path.join(args.out_dir, "summary_by_task.csv"), index=False)
    overall_df = (
        summary_df.groupby("planner", as_index=False)
        .agg(
            mean_window_reward=("mean_window_reward", "mean"),
            mean_delay_per_active=("mean_delay_per_active", "mean"),
            mean_active_targets=("mean_active_targets", "mean"),
            mean_search_fraction=("mean_search_fraction", "mean"),
            planner_latency_ms=("planner_latency_ms", "mean"),
        )
    )
    overall_df.to_csv(os.path.join(args.out_dir, "overall_summary.csv"), index=False)

    save_task_plot(summary_df, "mean_window_reward", "Mean Window Reward", os.path.join(args.out_dir, "task_reward.png"))
    save_task_plot(summary_df, "mean_delay_per_active", "Delay per Active Target", os.path.join(args.out_dir, "task_delay.png"))
    save_task_plot(summary_df, "mean_search_fraction", "Search Fraction", os.path.join(args.out_dir, "task_search_fraction.png"))
    save_task_plot(summary_df, "mean_active_targets", "Active Targets", os.path.join(args.out_dir, "task_active_targets.png"))
    save_task_plot(summary_df, "planner_latency_ms", "Planner Latency (ms/action)", os.path.join(args.out_dir, "task_latency.png"))

    for task in rep_tasks:
        save_window_plot(raw_window_df, task, "cumulative_reward", "Cumulative Reward", os.path.join(args.out_dir, f"window_cumulative_reward_task{task}.png"))


def run_old_methodology_compare(args):
    os.makedirs(args.out_dir, exist_ok=True)
    env_cfg = env_preset_cfg("original_like")
    tasks = parse_int_list(args.tasks)
    reference = {
        "EST": {10: 1.0000, 20: 1.0820, 30: 0.9920, 40: 0.9020, 50: 0.9255, 100: -0.9159, 200: -1.9543, 300: -3.1548, 400: -3.7150},
        "EDF": {10: 0.9996, 20: 1.0872, 30: 0.6345, 40: 0.7825, 50: 0.8196, 100: -0.3225, 200: -1.6163, 300: -2.4038, 400: -2.5206},
        "MCTS_FebReference": {10: 1.0000, 20: 1.1493, 30: 0.8869, 40: 0.8455, 50: 0.9118, 100: 0.8245, 200: -0.1682, 300: -0.7781, 400: -1.7871},
    }
    model = load_student_model(args.ckpt, args.max_trackers, args.device)
    planners = [
        ("EDF", lambda: EDFPlanner(max_trackers=args.max_trackers)),
        ("EST", lambda: ESTPlanner(max_trackers=args.max_trackers)),
        (
            args.student_name,
            lambda: make_student(
                model,
                args.max_trackers,
                args.student_rollouts,
                args.student_c,
                args.device,
                args.student_search_prior_scale,
                env_cfg,
                "original_like",
                args.expand_top_k,
                force_search_debt_ms=args.student_force_search_debt_ms,
                use_value_head=bool(args.student_use_value_head),
                value_scale=float(args.student_value_scale),
                leaf_value_mix=float(args.student_leaf_value_mix),
                rollout_candidate_cap=int(args.student_rollout_candidate_cap),
                root_search_strategy=str(args.student_root_search_strategy),
                gumbel_considered_actions=int(args.student_gumbel_considered_actions),
            ),
        ),
    ]
    rows = []
    for planner_name, planner_factory in planners:
        planner = planner_factory()
        for task in tasks:
            summary, _ = run_episode(planner, task, args.max_trackers, args.seed, args.windows, args.window_ms, env_cfg)
            rows.append(
                {
                    "planner": planner_name,
                    "task": task,
                    "mean_window_reward": summary["mean_window_reward"],
                    "reference_value": reference.get(planner_name, {}).get(task, np.nan),
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "old_methodology_compare.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    reference = sub.add_parser("reference_calibration")
    reference.add_argument("--out-dir", required=True)
    reference.add_argument("--tasks", default="20,50,100")
    reference.add_argument("--seeds", default="42,43,44")
    reference.add_argument("--windows", type=int, default=20)
    reference.add_argument("--window-ms", type=int, default=200)
    reference.add_argument("--max-trackers", type=int, default=100)
    reference.add_argument("--env-preset", choices=["original_like", "repaired_stress", "v39_legacy_reconstructed", "operational_linear_staleness"], default="repaired_stress")
    reference.add_argument("--revisit-time-scale-override", type=float, default=None)

    student = sub.add_parser("student_calibration")
    student.add_argument("--out-dir", required=True)
    student.add_argument("--ckpt", required=True)
    student.add_argument("--tasks", default="20,50,100")
    student.add_argument("--seeds", default="42,43")
    student.add_argument("--windows", type=int, default=20)
    student.add_argument("--window-ms", type=int, default=200)
    student.add_argument("--max-trackers", type=int, default=100)
    student.add_argument("--rollouts", default="6")
    student.add_argument("--search-prior-scales", default="1.0")
    student.add_argument("--student-c", type=float, default=0.5)
    student.add_argument("--expand-top-k", type=int, default=101)
    student.add_argument("--student-force-search-debt-ms", type=float, default=-1.0)
    student.add_argument("--student-use-value-head", type=int, default=0)
    student.add_argument("--student-value-scale", type=float, default=20.0)
    student.add_argument("--student-leaf-value-mix", type=float, default=1.0)
    student.add_argument("--student-rollout-candidate-cap", type=int, default=24)
    student.add_argument("--student-root-search-strategy", choices=["puct", "gumbel"], default="puct")
    student.add_argument("--student-gumbel-considered-actions", type=int, default=8)
    student.add_argument("--env-preset", choices=["original_like", "repaired_stress", "v39_legacy_reconstructed", "operational_linear_staleness"], default="repaired_stress")
    student.add_argument("--revisit-time-scale-override", type=float, default=None)
    student.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    bench = sub.add_parser("benchmark")
    bench.add_argument("--out-dir", required=True)
    bench.add_argument("--ckpt", required=True)
    bench.add_argument("--tasks", default="20,50,100")
    bench.add_argument("--rep-tasks", default="20,100")
    bench.add_argument("--seeds", default="42,43,44")
    bench.add_argument("--windows", type=int, default=20)
    bench.add_argument("--window-ms", type=int, default=200)
    bench.add_argument("--max-trackers", type=int, default=100)
    bench.add_argument("--student-name", default="Student_R6")
    bench.add_argument("--student-rollouts", type=int, default=6)
    bench.add_argument("--student-c", type=float, default=0.5)
    bench.add_argument("--student-search-prior-scale", type=float, default=0.05)
    bench.add_argument("--student-force-search-debt-ms", type=float, default=-1.0)
    bench.add_argument("--student-use-value-head", type=int, default=0)
    bench.add_argument("--student-value-scale", type=float, default=20.0)
    bench.add_argument("--student-leaf-value-mix", type=float, default=1.0)
    bench.add_argument("--student-rollout-candidate-cap", type=int, default=24)
    bench.add_argument("--student-root-search-strategy", choices=["puct", "gumbel"], default="puct")
    bench.add_argument("--student-gumbel-considered-actions", type=int, default=8)
    bench.add_argument("--expand-top-k", type=int, default=101)
    bench.add_argument("--env-preset", choices=["original_like", "repaired_stress", "v39_legacy_reconstructed", "operational_linear_staleness"], default="repaired_stress")
    bench.add_argument("--revisit-time-scale-override", type=float, default=None)
    bench.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    old = sub.add_parser("old_methodology_compare")
    old.add_argument("--out-dir", required=True)
    old.add_argument("--ckpt", required=True)
    old.add_argument("--tasks", default="10,20,30,40,50,100,200,300,400")
    old.add_argument("--windows", type=int, default=50)
    old.add_argument("--window-ms", type=int, default=200)
    old.add_argument("--max-trackers", type=int, default=500)
    old.add_argument("--seed", type=int, default=42)
    old.add_argument("--student-name", default="Student_R8")
    old.add_argument("--student-rollouts", type=int, default=6)
    old.add_argument("--student-c", type=float, default=0.5)
    old.add_argument("--student-search-prior-scale", type=float, default=1.0)
    old.add_argument("--student-force-search-debt-ms", type=float, default=-1.0)
    old.add_argument("--student-use-value-head", type=int, default=0)
    old.add_argument("--student-value-scale", type=float, default=20.0)
    old.add_argument("--student-leaf-value-mix", type=float, default=1.0)
    old.add_argument("--student-rollout-candidate-cap", type=int, default=24)
    old.add_argument("--student-root-search-strategy", choices=["puct", "gumbel"], default="puct")
    old.add_argument("--student-gumbel-considered-actions", type=int, default=8)
    old.add_argument("--expand-top-k", type=int, default=101)
    old.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    args = ap.parse_args()
    if args.cmd == "reference_calibration":
        run_reference_calibration(args)
    elif args.cmd == "student_calibration":
        run_student_calibration(args)
    elif args.cmd == "benchmark":
        run_benchmark(args)
    else:
        run_old_methodology_compare(args)


if __name__ == "__main__":
    main()

