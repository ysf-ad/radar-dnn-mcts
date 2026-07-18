from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from eval_action_attention_muzero_g import (
    LatentMuZeroPlanner,
    execute_plan_until_budget_joint_shaped,
    load_base_policy_model,
    sample_state_metrics,
    window_service_penalty,
    window_underuse_penalty,
)
from exact_env_mutual import MAXT, engine_env_cfg, env_cfg_for, xs_decode_action
from final_radar_campaign import get_obs
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env
from train_action_attention_muzero_g import LatentG


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def merge_repair(provisional: list[int], repaired: list[int], prefix: int) -> list[int]:
    if prefix <= 0:
        return list(provisional)
    # The provisional prefix is already available at the boundary and can
    # execute while the learned repair is computed.  Switch to the repaired
    # schedule once that prefix has covered the repair latency.
    merged = list(provisional[:prefix]) + list(repaired)
    out = []
    selected = set()
    for action in merged:
        row, sensor = xs_decode_action(int(action), MAXT)
        if int(sensor or 0) != 0:
            continue
        if int(row) > 0 and int(row) in selected:
            continue
        if int(row) > 0:
            selected.add(int(row))
        out.append(int(action))
    return out


def build_planner(args, env_cfg):
    model_args = SimpleNamespace(base_state=args.base_state, lean_base_load=False, variant=args.variant)
    model = load_base_policy_model(model_args, args.device).to(args.device).eval()
    checkpoint = torch.load(args.g_state, map_location=args.device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    seq_len = int(state["seq_pos"].shape[0])
    g = LatentG(d_model=int(args.d_model), seq_len=seq_len).to(args.device).eval()
    g.load_state_dict(state, strict=False)
    g.action_value_target_mean = float(checkpoint.get("action_value_target_mean", 0.0))
    g.action_value_target_std = max(1.0e-3, float(checkpoint.get("action_value_target_std", 1.0)))
    for module in (model, g):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return LatentMuZeroPlanner(
        model,
        g,
        env_cfg,
        policy_weight=1.0,
        q_weight=float(args.q_weight),
        search_score_bias=0.0,
        max_steps=20,
        tensor_loop=True,
        tensor_loop_factorized_decode=True,
        cuda_graph_tensor_loop=True,
        single_sensor_noop_action=True,
        single_sensor_s_only_choose=True,
        use_g_policy=True,
        device=args.device,
    )


def run_pipeline(planner, mode: str, repair_prefix: int, initial: int, rate: float, seed: int, windows: int, env_cfg: dict):
    eng = build_env(None, int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    next_plan = None
    rows = []
    try:
        # Compile/capture fixed-shape CUDA paths before deployment timing.
        warm_obs = get_obs(eng, debt)
        planner.plan(warm_obs, budget_ms=200.0)
        planner.predict_next_window_plan(warm_obs, [], [], 100.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for window in range(int(windows)):
            if bool(eng.term_buf[0]):
                break
            obs_start = get_obs(eng, debt)
            repair_ms = 0.0
            repair_overlap_ms = 0.0
            if next_plan is None:
                current_plan = planner.plan(obs_start, budget_ms=200.0)
            elif repair_prefix > 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                repaired = planner.plan(obs_start, budget_ms=200.0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                repair_ms = 1000.0 * (time.perf_counter() - t0)
                current_plan = merge_repair(next_plan, repaired, int(repair_prefix))
                dwell = np.asarray(obs_start.get("t_dwell", []), dtype=np.float32)
                for action in list(next_plan[: int(repair_prefix)]):
                    row, sensor = xs_decode_action(int(action), MAXT)
                    if int(sensor or 0) != 0:
                        continue
                    repair_overlap_ms += 10.0 if int(row) <= 0 else float(dwell[int(row) - 1])
            else:
                current_plan = list(next_plan)
            spent = 0.0
            reward = 0.0
            executed = []
            searches = 0
            predicted_plan = None
            prediction_ms = 0.0
            for action_index, action in enumerate(current_plan):
                remaining_budget = 200.0 - spent
                if remaining_budget <= 0.0 or bool(eng.term_buf[0]):
                    break
                r, dt, debt, ex, sea, _action_rows = execute_plan_until_budget_joint_shaped(
                    eng, [int(action)], remaining_budget, debt, mode, int(seed), int(window), env_cfg
                )
                if ex <= 0 or dt <= 0.0:
                    continue
                reward += float(r)
                spent += float(dt)
                searches += int(sea)
                executed.append(int(action))
                if predicted_plan is None and spent >= float(args_global.plan_start_ms):
                    obs_half = get_obs(eng, debt)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    predicted_plan = planner.predict_next_window_plan(
                        obs_half,
                        executed,
                        list(current_plan[action_index + 1 :]),
                        spent,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    prediction_ms = 1000.0 * (time.perf_counter() - t0)
            if predicted_plan is None:
                obs_half = get_obs(eng, debt)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                predicted_plan = planner.predict_next_window_plan(obs_half, executed, [], spent)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prediction_ms = 1000.0 * (time.perf_counter() - t0)
            next_plan = predicted_plan
            metrics = sample_state_metrics(eng, debt)
            reward += window_underuse_penalty(spent, 200.0, env_cfg)
            reward += window_service_penalty(metrics, env_cfg)
            cumulative += float(reward)
            overlap_available = max(0.0, 200.0 - float(args_global.plan_start_ms))
            rows.append(
                {
                    "method": mode,
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(seed),
                    "window": int(window),
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "drop_pct_active": float(metrics["drop_pct_active"]),
                    "tracked_targets": float(metrics["tracked_targets"]),
                    "mean_delay_active": float(metrics["mean_delay_active"]),
                    "search_fraction": float(searches / max(1, len(executed))),
                    "prediction_ms": float(prediction_ms),
                    "repair_ms": float(repair_ms),
                    "deadline_miss_ms": float(
                        max(0.0, prediction_ms - overlap_available)
                        + max(0.0, repair_ms - repair_overlap_ms)
                    ),
                    "spent_ms": float(spent),
                }
            )
    finally:
        eng.close()
    return pd.DataFrame(rows)


args_global = None


def main() -> None:
    global args_global
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-state", required=True)
    parser.add_argument("--g-state", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="two_row_action_attention")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--q-weight", type=float, default=0.5)
    parser.add_argument("--plan-start-ms", type=float, default=100.0)
    parser.add_argument("--repair-prefix", type=int, default=2)
    parser.add_argument("--initials", default="20,40,60")
    parser.add_argument("--rates", default="2,3,4")
    parser.add_argument("--seeds", default="916")
    parser.add_argument("--windows", type=int, default=100)
    args_global = parser.parse_args()

    reward_args = SimpleNamespace(
        windows=int(args_global.windows),
        env_mode="pufferlib_service",
        search_frame_overdue_weight=0.5,
        search_frame_drop_penalty=8.0,
        search_frame_state_penalty_weight=2.0,
        search_frame_delta_reward_weight=5.0,
        service_pressure_delta_reward_weight=0.30,
        serviced_pressure_improvement_reward_weight=0.15,
        discovered_target_reward=0.08,
    )
    exact_args = make_exact_args(reward_args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    frames = []
    for initial in parse_ints(args_global.initials):
        for rate in parse_floats(args_global.rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 0
            planner = build_planner(args_global, env_cfg)
            for seed in parse_ints(args_global.seeds):
                frames.append(run_pipeline(planner, "Async predictive", 0, initial, rate, seed, args_global.windows, env_cfg))
                frames.append(
                    run_pipeline(
                        planner,
                        f"Async + {args_global.repair_prefix}-action repair",
                        int(args_global.repair_prefix),
                        initial,
                        rate,
                        seed,
                        args_global.windows,
                        env_cfg,
                    )
                )
    out = pd.concat(frames, ignore_index=True)
    args_global.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args_global.out, index=False)
    summary = out.groupby("method", as_index=False).agg(
        reward_per_window=("window_reward", "mean"),
        drop_pct_active=("drop_pct_active", "mean"),
        tracked_targets=("tracked_targets", "mean"),
        mean_delay_active=("mean_delay_active", "mean"),
        search_fraction=("search_fraction", "mean"),
        prediction_ms=("prediction_ms", "mean"),
        repair_ms=("repair_ms", "mean"),
        deadline_miss_ms=("deadline_miss_ms", "mean"),
    )
    summary.to_csv(args_global.out.with_name(args_global.out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
