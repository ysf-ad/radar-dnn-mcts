from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()) if arr.size else 0.0,
        "p50_ms": float(np.percentile(arr, 50)) if arr.size else 0.0,
        "p90_ms": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
    }


def add_profile_stage(buckets: dict[str, list[float]], name: str, value_ms: float) -> None:
    buckets.setdefault(name, []).append(float(value_ms))


def profile_summary(buckets: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return dict(sorted(((name, stats(values)) for name, values in buckets.items()), key=lambda item: item[1]["mean_ms"], reverse=True))


def time_stage(device: torch.device, enabled: bool, buckets: dict[str, list[float]], name: str, fn):
    if not enabled:
        return fn()
    sync(device)
    t0 = time.perf_counter()
    out = fn()
    sync(device)
    add_profile_stage(buckets, name, (time.perf_counter() - t0) * 1000.0)
    return out


def make_live_slots(slot_template: np.ndarray, live_pos: list[int], elapsed, search_count, track_count, last, budget_ms: float) -> np.ndarray:
    slots = np.asarray(slot_template, dtype=np.float32)[np.asarray(live_pos, dtype=np.int64)].copy()
    if not live_pos:
        return slots
    elapsed_arr = np.asarray([elapsed[p] for p in live_pos], dtype=np.float32)
    search_arr = np.asarray([search_count[p] for p in live_pos], dtype=np.float32)
    track_arr = np.asarray([track_count[p] for p in live_pos], dtype=np.float32)
    last_arr = np.asarray([last[p] for p in live_pos], dtype=np.int32)
    slots[:, 0] = elapsed_arr / float(budget_ms)
    slots[:, 1] = search_arr / 20.0
    slots[:, 2] = track_arr / 100.0
    slots[:, 3] = (last_arr == 0).astype(np.float32)
    return slots


def tokenize_root_batch_fast(adapt, observations, max_trackers: int, token_dim: int) -> np.ndarray:
    """Root-only tokenizer for cached multi-env windows.

    Equivalent to ``tokenize_batch(adapt, observations)`` for the root call
    where no targets are selected and search_count is zero, but avoids selected
    set construction and a few optional per-row branches in the hot benchmark.
    """
    n = len(observations)
    x = np.zeros((n, int(max_trackers) + 1, int(token_dim)), dtype=np.float32)
    if n <= 0:
        return x

    t_desired = np.stack([np.asarray(obs["t_desired"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    deadline = np.stack([np.asarray(obs["t_deadline"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    dwell = np.stack([np.asarray(obs["t_dwell"], dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    active = np.stack([np.asarray(obs["active_mask"], dtype=bool)[:max_trackers] for obs in observations], axis=0)
    tracked = np.stack(
        [
            np.asarray(obs.get("tracked_mask", np.asarray(obs["active_mask"], dtype=bool) & (np.asarray(obs["t_deadline"]) > 0)), dtype=bool)[
                :max_trackers
            ]
            for obs in observations
        ],
        axis=0,
    )
    priority = np.stack(
        [np.asarray(obs.get("priority", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers] for obs in observations],
        axis=0,
    )
    ranges = np.stack(
        [np.asarray(obs.get("target_range", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers] for obs in observations],
        axis=0,
    )
    grids = [np.asarray(obs.get("grid", np.zeros((300,), dtype=np.float32)), dtype=np.float32) for obs in observations]
    grid_min = np.asarray([float(np.min(grid)) for grid in grids], dtype=np.float32)
    search_debt_ms = np.asarray([float(obs.get("search_debt_ms", 0.0)) for obs in observations], dtype=np.float32)
    pure = adapt.pure_mcts
    if float(pure.search_debt_penalty_weight) <= 0.0:
        search_penalty_norm = np.zeros((n,), dtype=np.float32)
    elif int(pure.search_delay_mode) == 0:
        search_penalty_norm = (float(pure.search_debt_penalty_weight) * search_debt_ms).astype(np.float32)
    else:
        arg = np.minimum(search_debt_ms / max(1e-3, float(pure.search_debt_tau_ms)), 20.0)
        search_penalty_norm = (float(pure.search_debt_penalty_weight) * (np.exp(arg) - 1.0)).astype(np.float32)
    if float(pure.search_delay_penalty_cap) >= 0.0:
        search_penalty_norm = np.minimum(search_penalty_norm, float(pure.search_delay_penalty_cap))
    search_penalty_norm = np.clip(np.where(search_debt_ms > 0.0, search_penalty_norm, 0.0), 0.0, 10.0).astype(np.float32)

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
            float(pure.global_tardiness_weight) * global_tardiness_norm
            + float(pure.local_tardiness_weight) * mean_tracked_delay_norm
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
    x[:, 0, 0] = np.clip(x[:, 0, 0] / 3000.0, -2.0, 2.0)
    x[:, 0, 1] = np.clip(x[:, 0, 1] / 3000.0, -2.0, 2.0)
    x[:, 0, 2] = np.clip(x[:, 0, 2] / 100.0, 0.0, 2.0)
    x[:, 0, 5] = np.clip(x[:, 0, 5] / 3000.0, -2.0, 2.0)

    az_bin = np.stack([np.asarray(obs.get("az_bin", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    el_bin = np.stack([np.asarray(obs.get("el_bin", np.zeros(max_trackers, dtype=np.float32)), dtype=np.float32)[:max_trackers] for obs in observations], axis=0)
    sector_idx = np.clip(np.round(el_bin * 9.0).astype(np.int32) * 30 + np.round(az_bin * 29.0).astype(np.int32), 0, 299)
    sector_urgency = np.zeros((n, max_trackers), dtype=np.float32)
    for i, grid in enumerate(grids):
        if len(grid) > 0:
            sector_urgency[i] = grid[np.clip(sector_idx[i], 0, len(grid) - 1)].astype(np.float32)

    target_tardiness = np.maximum(0.0, -t_desired).astype(np.float32)
    local_penalty_norm = np.clip(
        0.001 * target_tardiness * (1.0 + 2.0 * priority) * float(pure.local_tardiness_weight),
        0.0,
        10.0,
    ).astype(np.float32)
    x[:, 1 : max_trackers + 1, 0] = np.clip(t_desired / 3000.0, -2.0, 2.0)
    x[:, 1 : max_trackers + 1, 1] = np.clip(deadline / 3000.0, -2.0, 2.0)
    x[:, 1 : max_trackers + 1, 2] = np.clip(dwell / 100.0, 0.0, 2.0)
    x[:, 1 : max_trackers + 1, 3] = priority
    x[:, 1 : max_trackers + 1, 4] = (active & tracked).astype(np.float32)
    x[:, 1 : max_trackers + 1, 5] = np.clip(sector_urgency / 3000.0, -2.0, 2.0)
    x[:, 1 : max_trackers + 1, 6] = local_penalty_norm
    x[:, 1 : max_trackers + 1, 7] = (global_penalty_norm + search_penalty_norm)[:, None]
    range_norm = np.clip(ranges / 184_000_000.0, 0.0, 1.5)
    x[:, 1 : max_trackers + 1, 9] = range_norm
    x[:, 1 : max_trackers + 1, 10] = ((ranges > 10_000_000.0) & (ranges < 184_000_000.0)).astype(np.float32)
    x[:, 1 : max_trackers + 1, 11] = ((ranges > 5_000_000.0) & (ranges < 100_000_000.0)).astype(np.float32)
    x[:, :, 12] = np.asarray([float(obs.get("sensor_id", 0.0)) for obs in observations], dtype=np.float32)[:, None]
    x[:, 0, 9] = np.clip(np.asarray([float(obs.get("s_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32) / 200.0, 0.0, 5.0)
    x[:, 0, 10] = np.clip(np.asarray([float(obs.get("x_band_busy_ms", 0.0)) for obs in observations], dtype=np.float32) / 200.0, 0.0, 5.0)
    x[:, 0, 11] = np.asarray([float(obs.get("enable_x_band", 0.0)) for obs in observations], dtype=np.float32)
    for i, obs in enumerate(observations):
        if float(obs.get("use_grid_feature", 0.0)) > 0.5:
            grid = grids[i]
            if grid.size == 0:
                mean_overdue, drop_frac, max_age = 0.0, 0.0, 0.0
            else:
                age = 3000.0 - grid
                overdue = np.maximum(0.0, age - 3000.0) / 3000.0
                mean_overdue = float(np.clip(float(np.mean(overdue)), 0.0, 5.0))
                drop_frac = float(np.clip(float(np.mean(age > 4500.0)), 0.0, 1.0))
                max_age = float(np.clip(float(np.max(age) / 4500.0), 0.0, 5.0))
            x[i, 0, 9] = mean_overdue
            x[i, 0, 10] = drop_frac
            x[i, 0, 11] = max_age
    return x


def run_serial(planner, envs, args, device: torch.device) -> dict:
    from final_radar_campaign import get_obs
    from strict_window_report import execute_plan_until_budget

    search_debt = [0.0 for _ in envs]
    plan_times: list[float] = []
    rewards = [0.0 for _ in envs]
    executed = [0 for _ in envs]
    windows_done = 0
    if envs and hasattr(planner, "warmup"):
        warm_obs = get_obs(envs[0], search_debt[0])
        planner.warmup(warm_obs, budget_ms=int(args.window_ms))
    sync(device)
    wall0 = time.perf_counter()
    for window_idx in range(int(args.windows)):
        active_ids = [i for i, eng in enumerate(envs) if not eng.term_buf[0]]
        if not active_ids:
            break
        for env_idx in active_ids:
            eng = envs[env_idx]
            obs = get_obs(eng, search_debt[env_idx])
            sync(device)
            t0 = time.perf_counter()
            plan = planner.plan(obs, budget_ms=int(args.window_ms))
            sync(device)
            plan_times.append((time.perf_counter() - t0) * 1000.0)
            reward, _spent, new_debt, n_exec, _search_actions, _rows = execute_plan_until_budget(
                eng,
                plan,
                float(args.window_ms),
                search_debt[env_idx],
                "serial_fast_graph_gpu_select",
                int(args.seed) + env_idx,
                int(window_idx),
            )
            search_debt[env_idx] = float(new_debt)
            rewards[env_idx] += float(reward)
            executed[env_idx] += int(n_exec)
        windows_done += 1
    sync(device)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    total_windows = int(windows_done * len(envs))
    return {
        "wall_ms": float(wall_ms),
        "windows_requested": int(args.windows * len(envs)),
        "window_rounds": int(windows_done),
        "envs": int(len(envs)),
        "planned_env_windows": int(total_windows),
        "window_throughput_per_s": float(1000.0 * total_windows / max(wall_ms, 1e-12)),
        "planning_ms_per_env_window": float(np.mean(plan_times)) if plan_times else 0.0,
        "planning_stats": stats(plan_times),
        "total_reward": float(sum(rewards)),
        "executed_actions": int(sum(executed)),
    }


def run_batched(scorer, envs, args, device: torch.device) -> dict:
    from exact_env_mutual import xs_decode_action
    from final_radar_campaign import get_obs
    from repaired_campaign_tools import decode_sensor_action, execute_first_valid_action
    from two_sensor_physical_head_eval import MAXT

    search_debt = [0.0 for _ in envs]
    rewards = [0.0 for _ in envs]
    executed = [0 for _ in envs]
    plan_round_times: list[float] = []
    batch_sizes: list[int] = []
    depth_counts: list[int] = []
    if envs:
        warm_obs = [get_obs(eng, 0.0) for eng in envs if not eng.term_buf[0]]
        if warm_obs:
            _ = scorer.best_actions_torch(warm_obs, budget_ms=float(args.window_ms))
    sync(device)
    wall0 = time.perf_counter()
    windows_done = 0
    for window_idx in range(int(args.windows)):
        selected = [set() for _ in envs]
        elapsed = [0.0 for _ in envs]
        search_count = [0 for _ in envs]
        track_count = [0 for _ in envs]
        last = [-1 for _ in envs]
        active_ids = [i for i, eng in enumerate(envs) if not eng.term_buf[0]]
        if not active_ids:
            break
        depth = 0
        while active_ids and depth < int(args.max_depth):
            obs_batch = [get_obs(envs[i], search_debt[i]) for i in active_ids]
            selected_batch = [selected[i] for i in active_ids]
            elapsed_batch = [elapsed[i] for i in active_ids]
            search_count_batch = [search_count[i] for i in active_ids]
            track_count_batch = [track_count[i] for i in active_ids]
            last_batch = [last[i] for i in active_ids]
            sync(device)
            t0 = time.perf_counter()
            actions = scorer.best_actions_torch(
                obs_batch,
                selected=selected_batch,
                elapsed=elapsed_batch,
                search_count=search_count_batch,
                track_count=track_count_batch,
                last=last_batch,
                budget_ms=float(args.window_ms),
            )
            sync(device)
            plan_round_times.append((time.perf_counter() - t0) * 1000.0)
            batch_sizes.append(len(active_ids))
            next_active: list[int] = []
            for local_idx, env_idx in enumerate(active_ids):
                eng = envs[env_idx]
                if eng.term_buf[0] or elapsed[env_idx] >= float(args.window_ms):
                    continue
                remaining = max(0.0, float(args.window_ms) - float(elapsed[env_idx]))
                reward, dt, executed_action = execute_first_valid_action(eng, [int(actions[local_idx])], remaining)
                if executed_action is None or dt <= 0.0:
                    continue
                logical_action, _sensor = decode_sensor_action(int(executed_action), eng.max_trackers)
                base, _ = xs_decode_action(int(executed_action), MAXT)
                if int(logical_action) == 0:
                    search_debt[env_idx] = 0.0
                    search_count[env_idx] += 1
                else:
                    search_debt[env_idx] += max(float(dt), 0.0)
                    if int(base) > 0:
                        selected[env_idx].add(int(base))
                    track_count[env_idx] += 1
                rewards[env_idx] += float(reward)
                elapsed[env_idx] += float(dt)
                executed[env_idx] += 1
                last[env_idx] = int(base)
                if not eng.term_buf[0] and elapsed[env_idx] < float(args.window_ms):
                    next_active.append(env_idx)
            active_ids = next_active
            depth += 1
        depth_counts.append(depth)
        windows_done += 1
    sync(device)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    total_env_windows = int(windows_done * len(envs))
    return {
        "wall_ms": float(wall_ms),
        "windows_requested": int(args.windows * len(envs)),
        "window_rounds": int(windows_done),
        "envs": int(len(envs)),
        "planned_env_windows": int(total_env_windows),
        "window_throughput_per_s": float(1000.0 * total_env_windows / max(wall_ms, 1e-12)),
        "neural_rounds": int(len(plan_round_times)),
        "mean_batch_size": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "mean_depth": float(np.mean(depth_counts)) if depth_counts else 0.0,
        "planning_round_stats": stats(plan_round_times),
        "planning_ms_per_env_action": float(sum(plan_round_times) / max(1, sum(batch_sizes))),
        "total_reward": float(sum(rewards)),
        "executed_actions": int(sum(executed)),
    }


def run_batched_cached(planner, envs, args, device: torch.device) -> dict:
    from exact_env_mutual import attach_env_obs, xs_decode_action
    from final_radar_campaign import get_obs
    from mutual_features import TOKEN_DIM, slot_features_batch
    from perf_fast_planner import physical_action_table_batch
    from realistic_reward_retrain import adapter
    from repaired_campaign_tools import decode_sensor_action, execute_first_valid_action
    from two_sensor_physical_head_eval import MAXT

    adapt = adapter()
    search_debt = [0.0 for _ in envs]
    rewards = [0.0 for _ in envs]
    executed = [0 for _ in envs]
    plan_round_times: list[float] = []
    encode_times: list[float] = []
    batch_sizes: list[int] = []
    depth_counts: list[int] = []
    profile_enabled = bool(getattr(args, "profile_stages", False))
    stage_buckets: dict[str, list[float]] = {}
    if envs and hasattr(planner, "warmup"):
        planner.warmup(get_obs(envs[0], 0.0), budget_ms=int(args.window_ms))
    sync(device)
    wall0 = time.perf_counter()
    windows_done = 0
    for window_idx in range(int(args.windows)):
        root_env_ids = [i for i, eng in enumerate(envs) if not eng.term_buf[0]]
        if not root_env_ids:
            break
        obs2 = time_stage(
            device,
            profile_enabled,
            stage_buckets,
            "root_obs_attach",
            lambda: [attach_env_obs(get_obs(envs[i], search_debt[i]), planner.env_cfg, True, True) for i in root_env_ids],
        )
        selected = [set() for _ in root_env_ids]
        elapsed = [0.0 for _ in root_env_ids]
        search_count = [0 for _ in root_env_ids]
        track_count = [0 for _ in root_env_ids]
        last = [-1 for _ in root_env_ids]
        slot_template = time_stage(
            device,
            profile_enabled,
            stage_buckets,
            "root_slot_template",
            lambda: slot_features_batch(
                obs2,
                elapsed=elapsed,
                search_count=search_count,
                track_count=track_count,
                last_action=last,
                budget_ms=float(args.window_ms),
            ),
        )
        root_tokens = time_stage(
            device,
            profile_enabled,
            stage_buckets,
            "root_tokenize_batch",
            lambda: tokenize_root_batch_fast(adapt, obs2, MAXT, TOKEN_DIM),
        )
        sync(device)
        t0 = time.perf_counter()
        def encode_root():
            with torch.inference_mode():
                root_x = torch.from_numpy(root_tokens).to(device, dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                    return planner.model.backbone.encode_tokens(root_x)

        cls_out, tok_out, selected_t_all, token_active = time_stage(
            device,
            profile_enabled,
            stage_buckets,
            "root_h2d_encode",
            encode_root,
        )
        sync(device)
        encode_times.append((time.perf_counter() - t0) * 1000.0)
        live_pos = list(range(len(root_env_ids)))
        depth = 0
        while live_pos and depth < int(args.max_depth):
            live_obs = [obs2[p] for p in live_pos]
            live_selected = [selected[p] for p in live_pos]
            slots = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "slot_context_update",
                lambda: make_live_slots(slot_template, live_pos, elapsed, search_count, track_count, last, float(args.window_ms)),
            )
            physical = time_stage(
                device,
                profile_enabled,
                stage_buckets,
                "physical_action_table_batch",
                lambda: physical_action_table_batch(live_obs, selected=live_selected, max_trackers=MAXT),
            )
            sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                def prep_tensors():
                    pos_t = torch.as_tensor(live_pos, device=device, dtype=torch.long)
                    slot_t = torch.from_numpy(slots).to(device, dtype=torch.float32)
                    selected_t = selected_t_all.index_select(0, pos_t)
                    cls_live = cls_out.index_select(0, pos_t)
                    tok_live = tok_out.index_select(0, pos_t)
                    active_live = token_active.index_select(0, pos_t)
                    actions_t = torch.as_tensor(physical.actions, device=device, dtype=torch.long)
                    flat_t = torch.as_tensor(physical.bases * 2 + physical.sensors, device=device, dtype=torch.long)
                    valid_t = torch.as_tensor(physical.valid, device=device, dtype=torch.bool)
                    return slot_t, selected_t, cls_live, tok_live, active_live, actions_t, flat_t, valid_t

                slot_t, selected_t, cls_live, tok_live, active_live, actions_t, flat_t, valid_t = time_stage(
                    device,
                    profile_enabled,
                    stage_buckets,
                    "decision_tensor_prep_h2d",
                    prep_tensors,
                )

                def score_forward():
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                        score = planner.score_slots_from_encoded(cls_live, tok_live, selected_t, active_live, slot_t).float()
                    score[:, 0, :] += planner.search_score_bias
                    return score

                score_t = time_stage(device, profile_enabled, stage_buckets, "decision_score_forward", score_forward)

                def select_actions():
                    candidate_scores = torch.gather(score_t.reshape(len(live_pos), -1), 1, flat_t)
                    candidate_scores = candidate_scores.masked_fill(~(valid_t & torch.isfinite(candidate_scores)), -torch.inf)
                    idx = torch.argmax(candidate_scores, dim=1)
                    best = torch.gather(actions_t, 1, idx[:, None]).squeeze(1)
                    has_valid = torch.any(torch.isfinite(candidate_scores), dim=1)
                    return torch.where(has_valid, best, torch.full_like(best, -1))

                best = time_stage(device, profile_enabled, stage_buckets, "decision_select_device", select_actions)
                actions = time_stage(
                    device,
                    profile_enabled,
                    stage_buckets,
                    "decision_action_d2h",
                    lambda: best.cpu().numpy().astype(np.int64, copy=False),
                )
            sync(device)
            plan_round_times.append((time.perf_counter() - t0) * 1000.0)
            batch_sizes.append(len(live_pos))
            next_live: list[int] = []
            def step_envs():
                next_ids: list[int] = []
                for local_idx, pos in enumerate(live_pos):
                    env_idx = root_env_ids[pos]
                    eng = envs[env_idx]
                    if eng.term_buf[0] or elapsed[pos] >= float(args.window_ms):
                        continue
                    remaining = max(0.0, float(args.window_ms) - float(elapsed[pos]))
                    reward, dt, executed_action = execute_first_valid_action(eng, [int(actions[local_idx])], remaining)
                    if executed_action is None or dt <= 0.0:
                        continue
                    logical_action, _sensor = decode_sensor_action(int(executed_action), eng.max_trackers)
                    base, _ = xs_decode_action(int(executed_action), MAXT)
                    if int(logical_action) == 0:
                        search_debt[env_idx] = 0.0
                        search_count[pos] += 1
                    else:
                        search_debt[env_idx] += max(float(dt), 0.0)
                        if int(base) > 0:
                            selected[pos].add(int(base))
                            if 0 <= int(base) < selected_t_all.shape[1]:
                                selected_t_all[pos, int(base)] = True
                        track_count[pos] += 1
                    rewards[env_idx] += float(reward)
                    elapsed[pos] += float(dt)
                    executed[env_idx] += 1
                    last[pos] = int(base)
                    if not eng.term_buf[0] and elapsed[pos] < float(args.window_ms):
                        next_ids.append(pos)
                return next_ids

            next_live = time_stage(device, profile_enabled, stage_buckets, "env_step_batch", step_envs)
            live_pos = next_live
            depth += 1
        depth_counts.append(depth)
        windows_done += 1
    sync(device)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    total_env_windows = int(windows_done * len(envs))
    return {
        "wall_ms": float(wall_ms),
        "windows_requested": int(args.windows * len(envs)),
        "window_rounds": int(windows_done),
        "envs": int(len(envs)),
        "planned_env_windows": int(total_env_windows),
        "window_throughput_per_s": float(1000.0 * total_env_windows / max(wall_ms, 1e-12)),
        "encode_stats": stats(encode_times),
        "neural_rounds": int(len(plan_round_times)),
        "mean_batch_size": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "mean_depth": float(np.mean(depth_counts)) if depth_counts else 0.0,
        "planning_round_stats": stats(plan_round_times),
        "planning_ms_per_env_action": float(sum(plan_round_times) / max(1, sum(batch_sizes))),
        "total_reward": float(sum(rewards)),
        "executed_actions": int(sum(executed)),
        "stage_profile": profile_summary(stage_buckets) if profile_enabled else {},
    }


def _build_score_graph(planner, cls_out, tok_out, selected_t, token_active, slot_t):
    if planner.device.type != "cuda":
        return None
    try:
        static_selected = torch.empty_like(selected_t)
        static_slot = torch.empty_like(slot_t)
        static_selected.copy_(selected_t, non_blocking=False)
        static_slot.copy_(slot_t, non_blocking=False)

        def compute_score():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                return planner.score_slots_from_encoded(
                    cls_out,
                    tok_out,
                    static_selected,
                    token_active,
                    static_slot,
                ).float()

        with torch.inference_mode():
            for _ in range(3):
                _ = compute_score()
            torch.cuda.synchronize(planner.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_score = compute_score()

        def replay(next_selected_t, next_slot_t):
            static_selected.copy_(next_selected_t, non_blocking=False)
            static_slot.copy_(next_slot_t, non_blocking=False)
            graph.replay()
            return static_score

        return replay
    except Exception:
        return None


def run_batched_cached_graph(planner, envs, args, device: torch.device) -> dict:
    from exact_env_mutual import attach_env_obs, xs_decode_action
    from final_radar_campaign import get_obs
    from mutual_features import TOKEN_DIM, slot_features_batch
    from perf_fast_planner import physical_action_table_batch
    from realistic_reward_retrain import adapter
    from repaired_campaign_tools import decode_sensor_action, execute_first_valid_action
    from two_sensor_physical_head_eval import MAXT

    adapt = adapter()
    search_debt = [0.0 for _ in envs]
    rewards = [0.0 for _ in envs]
    executed = [0 for _ in envs]
    plan_round_times: list[float] = []
    encode_times: list[float] = []
    graph_build_times: list[float] = []
    batch_sizes: list[int] = []
    depth_counts: list[int] = []
    graph_replay_rounds = 0
    raw_rounds = 0
    if envs and hasattr(planner, "warmup"):
        planner.warmup(get_obs(envs[0], 0.0), budget_ms=int(args.window_ms))
    sync(device)
    wall0 = time.perf_counter()
    windows_done = 0
    for window_idx in range(int(args.windows)):
        root_env_ids = [i for i, eng in enumerate(envs) if not eng.term_buf[0]]
        if not root_env_ids:
            break
        obs2 = [attach_env_obs(get_obs(envs[i], search_debt[i]), planner.env_cfg, True, True) for i in root_env_ids]
        selected = [set() for _ in root_env_ids]
        elapsed = [0.0 for _ in root_env_ids]
        search_count = [0 for _ in root_env_ids]
        track_count = [0 for _ in root_env_ids]
        last = [-1 for _ in root_env_ids]
        slot_template = slot_features_batch(
            obs2,
            elapsed=elapsed,
            search_count=search_count,
            track_count=track_count,
            last_action=last,
            budget_ms=float(args.window_ms),
        )
        root_tokens = tokenize_root_batch_fast(adapt, obs2, MAXT, TOKEN_DIM)
        sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            root_x = torch.from_numpy(root_tokens).to(device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                cls_out, tok_out, selected_t_all, token_active = planner.model.backbone.encode_tokens(root_x)
        sync(device)
        encode_times.append((time.perf_counter() - t0) * 1000.0)

        full_slots = make_live_slots(slot_template, list(range(len(root_env_ids))), elapsed, search_count, track_count, last, float(args.window_ms))
        full_slot_t = torch.from_numpy(full_slots).to(device, dtype=torch.float32)
        sync(device)
        t0 = time.perf_counter()
        graph_replay = _build_score_graph(planner, cls_out, tok_out, selected_t_all, token_active, full_slot_t)
        sync(device)
        graph_build_times.append((time.perf_counter() - t0) * 1000.0)

        live_pos = list(range(len(root_env_ids)))
        depth = 0
        while live_pos and depth < int(args.max_depth):
            live_obs = [obs2[p] for p in live_pos]
            live_selected = [selected[p] for p in live_pos]
            slots = make_live_slots(slot_template, live_pos, elapsed, search_count, track_count, last, float(args.window_ms))
            physical = physical_action_table_batch(live_obs, selected=live_selected, max_trackers=MAXT)
            sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                if graph_replay is not None and len(live_pos) == len(root_env_ids):
                    slot_t = torch.from_numpy(slots).to(device, dtype=torch.float32)
                    score_t = graph_replay(selected_t_all, slot_t)
                    graph_replay_rounds += 1
                else:
                    pos_t = torch.as_tensor(live_pos, device=device, dtype=torch.long)
                    slot_t = torch.from_numpy(slots).to(device, dtype=torch.float32)
                    selected_t = selected_t_all.index_select(0, pos_t)
                    cls_live = cls_out.index_select(0, pos_t)
                    tok_live = tok_out.index_select(0, pos_t)
                    active_live = token_active.index_select(0, pos_t)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=planner.use_amp):
                        score_t = planner.score_slots_from_encoded(cls_live, tok_live, selected_t, active_live, slot_t).float()
                    raw_rounds += 1
                score_t[:, 0, :] += planner.search_score_bias
                actions_t = torch.as_tensor(physical.actions, device=device, dtype=torch.long)
                flat_t = torch.as_tensor(physical.bases * 2 + physical.sensors, device=device, dtype=torch.long)
                valid_t = torch.as_tensor(physical.valid, device=device, dtype=torch.bool)
                candidate_scores = torch.gather(score_t.reshape(len(live_pos), -1), 1, flat_t)
                candidate_scores = candidate_scores.masked_fill(~(valid_t & torch.isfinite(candidate_scores)), -torch.inf)
                idx = torch.argmax(candidate_scores, dim=1)
                best = torch.gather(actions_t, 1, idx[:, None]).squeeze(1)
                has_valid = torch.any(torch.isfinite(candidate_scores), dim=1)
                best = torch.where(has_valid, best, torch.full_like(best, -1))
                actions = best.cpu().numpy().astype(np.int64, copy=False)
            sync(device)
            plan_round_times.append((time.perf_counter() - t0) * 1000.0)
            batch_sizes.append(len(live_pos))
            next_live: list[int] = []
            for local_idx, pos in enumerate(live_pos):
                env_idx = root_env_ids[pos]
                eng = envs[env_idx]
                if eng.term_buf[0] or elapsed[pos] >= float(args.window_ms):
                    continue
                remaining = max(0.0, float(args.window_ms) - float(elapsed[pos]))
                reward, dt, executed_action = execute_first_valid_action(eng, [int(actions[local_idx])], remaining)
                if executed_action is None or dt <= 0.0:
                    continue
                logical_action, _sensor = decode_sensor_action(int(executed_action), eng.max_trackers)
                base, _ = xs_decode_action(int(executed_action), MAXT)
                if int(logical_action) == 0:
                    search_debt[env_idx] = 0.0
                    search_count[pos] += 1
                else:
                    search_debt[env_idx] += max(float(dt), 0.0)
                    if int(base) > 0:
                        selected[pos].add(int(base))
                        if 0 <= int(base) < selected_t_all.shape[1]:
                            selected_t_all[pos, int(base)] = True
                    track_count[pos] += 1
                rewards[env_idx] += float(reward)
                elapsed[pos] += float(dt)
                executed[env_idx] += 1
                last[pos] = int(base)
                if not eng.term_buf[0] and elapsed[pos] < float(args.window_ms):
                    next_live.append(pos)
            live_pos = next_live
            depth += 1
        depth_counts.append(depth)
        windows_done += 1
    sync(device)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    total_env_windows = int(windows_done * len(envs))
    return {
        "wall_ms": float(wall_ms),
        "windows_requested": int(args.windows * len(envs)),
        "window_rounds": int(windows_done),
        "envs": int(len(envs)),
        "planned_env_windows": int(total_env_windows),
        "window_throughput_per_s": float(1000.0 * total_env_windows / max(wall_ms, 1e-12)),
        "encode_stats": stats(encode_times),
        "graph_build_stats": stats(graph_build_times),
        "neural_rounds": int(len(plan_round_times)),
        "graph_replay_rounds": int(graph_replay_rounds),
        "raw_rounds": int(raw_rounds),
        "mean_batch_size": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
        "mean_depth": float(np.mean(depth_counts)) if depth_counts else 0.0,
        "planning_round_stats": stats(plan_round_times),
        "planning_ms_per_env_action": float(sum(plan_round_times) / max(1, sum(batch_sizes))),
        "total_reward": float(sum(rewards)),
        "executed_actions": int(sum(executed)),
    }


def build_envs(args, env_cfg):
    from repaired_campaign_tools import EDFPlanner, build_env
    from two_sensor_physical_head_eval import MAXT

    envs = []
    for idx in range(int(args.envs)):
        seed = int(args.seed) + idx
        eng = build_env(EDFPlanner(MAXT), int(args.initial_targets), MAXT, seed, int(args.window_ms), env_cfg)
        eng.reset(seed=seed)
        envs.append(eng)
    return envs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--initial-targets", type=int, default=60)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/perf_lab_multi_env_online_batch.json"))
    args = parser.parse_args()

    from perf_fast_planner import BatchedActionAttentionScorer, FastActionAttentionPlanner
    from repaired_campaign_tools import env_preset_cfg
    from two_sensor_physical_head_eval import ActionAttentionFactorizedNet

    torch.manual_seed(123)
    np.random.seed(123)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    serial_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    batch_model = ActionAttentionFactorizedNet(48, 4, 2).eval()
    batch_model.load_state_dict(serial_model.state_dict())
    serial = FastActionAttentionPlanner(
        serial_model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
        use_cuda_graph=True,
        use_gpu_select=True,
    )
    batched = BatchedActionAttentionScorer(
        batch_model,
        env_cfg,
        device=device,
        use_amp=bool(args.amp),
    )

    serial_envs = build_envs(args, env_cfg)
    batch_envs = build_envs(args, env_cfg)
    cached_envs = build_envs(args, env_cfg)
    graph_envs = build_envs(args, env_cfg) if not bool(args.skip_graph) else []
    try:
        serial_report = run_serial(serial, serial_envs, args, device)
        batched_report = run_batched(batched, batch_envs, args, device)
        cached_report = run_batched_cached(serial, cached_envs, args, device)
        graph_report = run_batched_cached_graph(serial, graph_envs, args, device) if graph_envs else {}
    finally:
        for eng in [*serial_envs, *batch_envs, *cached_envs, *graph_envs]:
            eng.close()

    report = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "amp": bool(args.amp),
        "envs": int(args.envs),
        "windows": int(args.windows),
        "window_ms": int(args.window_ms),
        "max_depth": int(args.max_depth),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "seed": int(args.seed),
        "serial_fast_graph_gpu_select": serial_report,
        "batched_multi_env_reencode": batched_report,
        "batched_multi_env_cached_root": cached_report,
        "batched_multi_env_cached_root_graph": graph_report,
        "throughput_speedup": float(
            cached_report["window_throughput_per_s"] / max(serial_report["window_throughput_per_s"], 1e-12)
        ),
        "graph_throughput_speedup": float(
            graph_report.get("window_throughput_per_s", 0.0) / max(serial_report["window_throughput_per_s"], 1e-12)
        )
        if graph_report
        else None,
        "reward_delta_cached_minus_serial": float(cached_report["total_reward"] - serial_report["total_reward"]),
        "reward_delta_graph_minus_serial": float(graph_report.get("total_reward", serial_report["total_reward"]) - serial_report["total_reward"])
        if graph_report
        else None,
        "reward_delta_reencode_minus_serial": float(batched_report["total_reward"] - serial_report["total_reward"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
