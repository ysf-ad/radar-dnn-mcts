from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from canonical_batch_decoder import BatchSchedulePlanner, load_model as load_batch_model
from canonical_scheduler_contract import add_canonical_reward_args
from distill_sparse64_sequence_decoder import Sparse64CudaGraphPlanner, Sparse64SequenceDecoder
from eval_action_attention_muzero_g import (
    execute_plan_until_budget_joint_shaped,
    sample_state_metrics,
    window_service_penalty,
    window_underuse_penalty,
)
from eval_async_predictive_pipeline import build_planner as build_muzero_planner
from eval_sonly_reencode_action_attention import SOnlyReencodeActionAttentionPlanner
from exact_env_mutual import MAXT, engine_env_cfg, env_cfg_for, get_obs, xs_decode_action
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env


def parse_ints(text: str) -> list[int]:
    return [int(value) for value in str(text).split(",") if value.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(value) for value in str(text).split(",") if value.strip()]


def load_ar_planner(path: str, env_cfg: dict, device: str, max_steps: int):
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("args", {}) if isinstance(payload, dict) else {}
    model = Sparse64SequenceDecoder(
        d_model=int(cfg.get("d_model", 96)),
        nhead=int(cfg.get("nhead", 4)),
        nlayers=int(cfg.get("nlayers", 2)),
    ).to(device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    return Sparse64CudaGraphPlanner(model.eval(), env_cfg, max_steps=max_steps)


def make_planner(method: str, args, env_cfg: dict):
    if method == "batch":
        return BatchSchedulePlanner(load_batch_model(args.batch_state, args.device), args.device, assignment="hungarian")
    if method == "ar":
        return load_ar_planner(args.ar_state, env_cfg, args.device, args.max_steps)
    if method == "muzero":
        ns = SimpleNamespace(
            base_state=args.base_state,
            g_state=args.g_state,
            variant=args.variant,
            device=args.device,
            d_model=args.g_d_model,
            q_weight=args.muzero_q_weight,
        )
        return build_muzero_planner(ns, env_cfg)
    if method == "reencode":
        return SOnlyReencodeActionAttentionPlanner(
            args.reencode_state,
            args.variant,
            env_cfg,
            device=args.device,
            policy_weight=1.0,
            q_weight=args.reencode_q_weight,
            cuda_graph=True,
        )
    raise ValueError(method)


def complete_plan(planner, obs: dict, budget_ms: float, one_step: bool, max_steps: int) -> list[int]:
    if not one_step:
        return [int(action) for action in planner.plan(obs, budget_ms=budget_ms)]
    plan: list[int] = []
    remaining = float(budget_ms)
    dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
    for step in range(int(max_steps)):
        proposed = planner.plan(obs, budget_ms=remaining if step else budget_ms)
        if not proposed:
            break
        action = int(proposed[0])
        row, sensor = xs_decode_action(action, MAXT)
        if int(sensor or 0) != 0:
            continue
        dt = 10.0 if int(row) <= 0 else float(dwell[int(row) - 1]) if int(row) - 1 < len(dwell) else 10.0
        plan.append(action)
        remaining -= max(1.0, dt)
        if remaining <= 0.0:
            break
    return plan


def merge_repair(provisional: list[int], repaired: list[int], depth: int) -> list[int]:
    merged = list(provisional[: max(0, int(depth))]) + list(repaired)
    output: list[int] = []
    selected: set[int] = set()
    for action in merged:
        row, sensor = xs_decode_action(int(action), MAXT)
        if int(sensor or 0) != 0 or (int(row) > 0 and int(row) in selected):
            continue
        if int(row) > 0:
            selected.add(int(row))
        output.append(int(action))
    return output


def run_episode(
    method: str,
    planner,
    args,
    env_cfg: dict,
    initial: int,
    rate: float,
    seed: int,
    repair_depth: int,
    preboundary_passes: int = -1,
):
    eng = build_env(None, initial, MAXT, seed, 200, engine_env_cfg(env_cfg))
    eng.reset(seed=seed)
    debt = 0.0
    next_plan: list[int] | None = None
    rows = []
    one_step = method == "reencode"
    try:
        warm = get_obs(eng, debt)
        complete_plan(planner, warm, 200.0, one_step, args.max_steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for window in range(args.windows):
            obs_start = get_obs(eng, debt)
            repair_ms = 0.0
            overlap_ms = 0.0
            if next_plan is None:
                current = complete_plan(planner, obs_start, 200.0, one_step, args.max_steps)
            elif preboundary_passes >= 0 or repair_depth <= 0:
                current = list(next_plan)
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                repaired = complete_plan(planner, obs_start, 200.0, one_step, args.max_steps)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                repair_ms = 1000.0 * (time.perf_counter() - start)
                current = merge_repair(next_plan, repaired, repair_depth)
                dwell = np.asarray(obs_start.get("t_dwell", []), dtype=np.float32)
                for action in next_plan[:repair_depth]:
                    row, _ = xs_decode_action(int(action), MAXT)
                    overlap_ms += 10.0 if row <= 0 else float(dwell[row - 1]) if row - 1 < len(dwell) else 10.0

            spent = reward = 0.0
            searches = 0
            executed: list[int] = []
            predicted = None
            prediction_ms = 0.0
            prediction_miss_ms = 0.0
            if preboundary_passes >= 0:
                latest_start = max(float(args.plan_start_ms), 200.0 - float(args.preboundary_guard_ms))
                prediction_times = np.linspace(
                    float(args.plan_start_ms), latest_start, max(1, int(preboundary_passes) + 1)
                ).tolist()
            else:
                prediction_times = [float(args.plan_start_ms)]
            prediction_index = 0
            for action_index, action in enumerate(current):
                remaining = 200.0 - spent
                if remaining <= 0.0:
                    break
                result = execute_plan_until_budget_joint_shaped(
                    eng, [int(action)], remaining, debt, f"Async {method}", seed, window, env_cfg
                )
                step_reward, dt, debt, count, search_count, _ = result
                if count <= 0 or dt <= 0.0:
                    continue
                reward += float(step_reward)
                spent += float(dt)
                searches += int(search_count)
                executed.append(int(action))
                while prediction_index < len(prediction_times) and spent >= prediction_times[prediction_index]:
                    mid_obs = get_obs(eng, debt)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    if (
                        method == "muzero"
                        and bool(args.use_muzero_boundary_projection)
                        and hasattr(planner, "predict_next_window_plan")
                    ):
                        predicted = planner.predict_next_window_plan(
                            mid_obs,
                            executed,
                            list(current[action_index + 1 :]),
                            spent,
                        )
                    else:
                        predicted = complete_plan(planner, mid_obs, 200.0, one_step, args.max_steps)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_prediction = 1000.0 * (time.perf_counter() - start)
                    prediction_ms += elapsed_prediction
                    prediction_miss_ms = max(
                        prediction_miss_ms,
                        max(0.0, elapsed_prediction - (200.0 - prediction_times[prediction_index])),
                    )
                    prediction_index += 1
            if predicted is None:
                mid_obs = get_obs(eng, debt)
                start = time.perf_counter()
                predicted = complete_plan(planner, mid_obs, 200.0, one_step, args.max_steps)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prediction_ms = 1000.0 * (time.perf_counter() - start)
            next_plan = predicted
            metrics = sample_state_metrics(eng, debt)
            reward += window_underuse_penalty(spent, 200.0, env_cfg)
            reward += window_service_penalty(metrics, env_cfg)
            available = max(0.0, 200.0 - args.plan_start_ms)
            rows.append({
                "method": method,
                "repair_depth": int(repair_depth),
                "repair_passes": int(preboundary_passes),
                "repair_mode": "preboundary" if preboundary_passes >= 0 else "boundary_prefix",
                "initial": initial,
                "rate": rate,
                "seed": seed,
                "window": window,
                "window_reward": reward,
                "drop_pct_active": metrics["drop_pct_active"],
                "tracked_targets": metrics["tracked_targets"],
                "mean_delay_active": metrics["mean_delay_active"],
                "search_fraction": searches / max(1, len(executed)),
                "prediction_ms": prediction_ms,
                "repair_ms": repair_ms,
                "deadline_miss_ms": (
                    prediction_miss_ms
                    if preboundary_passes >= 0
                    else max(0.0, prediction_ms - available) + max(0.0, repair_ms - overlap_ms)
                ),
            })
    finally:
        eng.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-state", required=True)
    parser.add_argument("--g-state", required=True)
    parser.add_argument("--ar-state", required=True)
    parser.add_argument("--batch-state", required=True)
    parser.add_argument("--reencode-state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", default="batch,ar,muzero,reencode")
    parser.add_argument("--repair-depths", default="0,1,2")
    parser.add_argument(
        "--preboundary-passes",
        default="",
        help="Comma-separated numbers of late repair passes. When set, repair completes before the boundary and repair-depths are ignored.",
    )
    parser.add_argument("--preboundary-guard-ms", type=float, default=30.0)
    parser.add_argument("--use-muzero-boundary-projection", action="store_true")
    parser.add_argument("--initials", default="20,40,60")
    parser.add_argument("--rates", default="2,3,4")
    parser.add_argument("--seeds", default="916")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--plan-start-ms", type=float, default=100.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", default="two_row_action_attention_qpolicy_factored_loss")
    parser.add_argument("--g-d-model", type=int, default=48)
    parser.add_argument("--muzero-q-weight", type=float, default=1.0)
    parser.add_argument("--reencode-q-weight", type=float, default=0.25)
    add_canonical_reward_args(parser)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    rows = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            env_cfg = env_cfg_for(rate, exact_args)
            env_cfg["enable_x_band"] = 0
            for seed in parse_ints(args.seeds):
                for method in [value.strip() for value in args.methods.split(",") if value.strip()]:
                    if str(args.preboundary_passes).strip():
                        configs = [(-1, passes) for passes in parse_ints(args.preboundary_passes)]
                    else:
                        configs = [(depth, -1) for depth in parse_ints(args.repair_depths)]
                    for depth, passes in configs:
                        planner = make_planner(method, args, env_cfg)
                        rows.extend(run_episode(method, planner, args, env_cfg, initial, rate, seed, depth, passes))
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    summary = frame.groupby(["method", "repair_mode", "repair_depth", "repair_passes"], as_index=False).agg(
        reward_per_window=("window_reward", "mean"),
        drop_pct_active=("drop_pct_active", "mean"),
        tracked_targets=("tracked_targets", "mean"),
        mean_delay_active=("mean_delay_active", "mean"),
        search_fraction=("search_fraction", "mean"),
        prediction_ms=("prediction_ms", "mean"),
        repair_ms=("repair_ms", "mean"),
        deadline_miss_ms=("deadline_miss_ms", "mean"),
    )
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
