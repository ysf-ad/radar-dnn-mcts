"""Evaluate predicted schedules with an in-flight observation-conditioned update."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from boundary_realtime_eval import (
    add_reward_args,
    load_ar,
    load_boundary,
    plan_from_tokens,
    predict_tokens,
    token_slot,
)
from collect_schedule_update_data import rows_from_plan
from eval_action_attention_muzero_g import (
    build_env,
    execute_plan_until_budget_joint_shaped,
    get_obs,
    sample_state_metrics,
    window_service_penalty,
    window_underuse_penalty,
)
from exact_env_mutual import MAXT, _DummyPlanner, engine_env_cfg, env_cfg_for, xs_decode_action, xs_s_search_action, xs_s_track_action
from foundation_mcts_fair_eval import parse_floats, parse_ints
from mutual_features import tokenize
from penalty_window_quota_learner_eval import make_exact_args
from realistic_reward_retrain import adapter
from schedule_update_model import ScheduleUpdateModel
from distill_sparse64_sequence_decoder import Sparse64CudaGraphRunner


def load_updater(path: Path, device: torch.device) -> ScheduleUpdateModel:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = ScheduleUpdateModel(
        token_dim=int(cfg["token_dim"]),
        max_tokens=int(cfg["max_tokens"]),
        max_steps=int(cfg["max_steps"]),
        d_model=int(cfg["d_model"]),
        layers=int(cfg["layers"]),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


@torch.inference_mode()
def update_plan(model: ScheduleUpdateModel, latest: np.ndarray, existing: list[int]) -> list[int]:
    device = next(model.parameters()).device
    rows = rows_from_plan(existing, model.max_steps)
    mask = rows >= 0
    copy_logits, type_logits, target_logits = model(
        torch.as_tensor(latest[None], dtype=torch.float32, device=device),
        torch.as_tensor(np.maximum(rows, 0)[None], dtype=torch.long, device=device),
        torch.as_tensor(mask[None], dtype=torch.bool, device=device),
    )
    root = torch.as_tensor(latest, dtype=torch.float32, device=device)
    active = root[:, 4] > 0.5
    active[0] = False
    dwell = root[:, 2].clamp(0.01, 2.0) * 100.0
    selected = torch.zeros_like(active)
    elapsed = 0.0
    actions: list[int] = []
    for slot in range(model.max_steps):
        valid = active & ~selected
        copied_row = int(rows[slot])
        use_copy = bool(copy_logits[0, slot] > 0) and copied_row >= 0
        if use_copy and copied_row > 0 and not bool(valid[copied_row]):
            use_copy = False
        if use_copy:
            best = copied_row
            choose_track = copied_row > 0
        else:
            target_logp = torch.log_softmax(target_logits[0, slot], dim=-1).masked_fill(~valid, -1.0e9)
            best = int(target_logp.argmax().item())
            type_logp = torch.log_softmax(type_logits[0, slot], dim=-1)
            choose_track = bool(valid.any()) and float(type_logp[1] + target_logp[best]) > float(type_logp[0])
        if choose_track:
            dt = float(max(1.0, dwell[best].item()))
            actions.append(xs_s_track_action(best, MAXT))
            selected[best] = True
        else:
            dt = 10.0
            actions.append(xs_s_search_action(MAXT))
        elapsed += dt
        if elapsed >= 200.0:
            break
    return actions or [xs_s_search_action(MAXT)]


def ar_prefix_update(ar, latest: np.ndarray, existing: list[int], prefix_steps: int) -> list[int]:
    """Replan a short prefix from fresh observations and retain the valid suffix."""
    prefix = plan_from_tokens(ar, latest, max_steps=max(1, int(prefix_steps)))
    active = np.asarray(latest[:, 4] > 0.5, dtype=np.bool_)
    dwell = np.asarray(np.clip(latest[:, 2], 0.01, 2.0) * 100.0, dtype=np.float32)
    selected: set[int] = set()
    actions: list[int] = []
    elapsed = 0.0
    for action in [*prefix, *existing]:
        row, sensor = xs_decode_action(int(action), MAXT)
        if int(sensor or 0) != 0:
            continue
        row = int(row)
        if row > 0 and (row in selected or row >= len(active) or not active[row]):
            continue
        dt = 10.0 if row <= 0 else float(max(1.0, dwell[row]))
        actions.append(int(action))
        if row > 0:
            selected.add(row)
        elapsed += dt
        if elapsed >= 200.0:
            break
    while elapsed < 200.0 and len(actions) < 32:
        actions.append(xs_s_search_action(MAXT))
        elapsed += 10.0
    return actions


@torch.inference_mode()
def graph_replan(runner: Sparse64CudaGraphRunner, latest: np.ndarray) -> list[int]:
    device = runner.root.device
    root = torch.as_tensor(latest[None], dtype=torch.float32, device=device)
    root_single = root[0]
    slot = token_slot(root_single, 0.0, 0, 0, -1)
    dwell = (root_single[:, 2].clamp(0.01, 2.0) * 100.0).unsqueeze(0)
    inactive = root_single[:, 4] <= 0.5
    dwell[0, inactive] = 1.0e9
    budget = torch.full((1,), 200.0, dtype=torch.float32, device=device)
    decoded = runner(root, slot, dwell, budget)[0].detach().cpu().tolist()
    return [
        xs_s_search_action(MAXT) if int(row) <= 0 else xs_s_track_action(int(row), MAXT)
        for row in decoded if int(row) >= 0
    ] or [xs_s_search_action(MAXT)]


@torch.inference_mode()
def should_revise_first(model: ScheduleUpdateModel, latest: np.ndarray, existing: list[int]) -> bool:
    device = next(model.parameters()).device
    rows = rows_from_plan(existing, model.max_steps)
    mask = rows >= 0
    copy_logits, _, _ = model(
        torch.as_tensor(latest[None], dtype=torch.float32, device=device),
        torch.as_tensor(np.maximum(rows, 0)[None], dtype=torch.long, device=device),
        torch.as_tensor(mask[None], dtype=torch.bool, device=device),
    )
    return bool(copy_logits[0, 0] <= 0.0)


def run_episode(mode, ar, predictor, updater, graph_runner, initial, rate, seed, windows, args):
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    env_cfg = env_cfg_for(float(rate), exact_args)
    env_cfg["enable_x_band"] = 0
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    adapt = adapter()
    debt = cumulative = 0.0
    current = plan_from_tokens(ar, tokenize(adapt, get_obs(eng, debt), set(), 0), args.max_steps)
    rows = []
    try:
        for window in range(int(windows)):
            spent = reward = 0.0
            searches = 0
            provisional = None
            update_ms = 0.0
            updated = False
            midpoint_active = None
            schedule_changed = 0.0
            new_arrival_targets = 0.0
            new_arrival_scheduled = 0.0
            for action_index, action in enumerate(current):
                result = execute_plan_until_budget_joint_shaped(
                    eng, [int(action)], 200.0 - spent, debt, f"Online update {mode}", seed, window, env_cfg
                )
                step_reward, dt, debt, count, search_count, _ = result
                if count <= 0 or dt <= 0:
                    continue
                reward += float(step_reward)
                spent += float(dt)
                searches += int(search_count)
                if provisional is None and spent >= float(args.plan_start_ms):
                    midpoint = tokenize(adapt, get_obs(eng, debt), set(), 0).astype(np.float32)
                    midpoint_active = midpoint[:, 4] > 0.5
                    suffix = [int(value) for value in current[action_index + 1 :]]
                    predicted, _ = predict_tokens(
                        predictor, midpoint, suffix, max(0.0, 200.0 - spent), (0, 1), False
                    )
                    provisional = plan_from_tokens(ar, predicted, args.max_steps)
                if not updated and provisional is not None and spent >= float(args.repair_start_ms):
                    latest = tokenize(adapt, get_obs(eng, debt), set(), 0).astype(np.float32)
                    before_update = list(provisional)
                    latest_active = latest[:, 4] > 0.5
                    newly_visible = latest_active & ~(midpoint_active if midpoint_active is not None else latest_active)
                    newly_visible[0] = False
                    device = next(ar.parameters()).device
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    if mode == "updater":
                        provisional = update_plan(updater, latest, provisional)
                    elif mode == "late_replan":
                        provisional = plan_from_tokens(ar, latest, args.max_steps)
                    elif mode == "graph_replan":
                        provisional = graph_replan(graph_runner, latest)
                    elif mode.startswith("prefix"):
                        provisional = ar_prefix_update(
                            ar, latest, provisional, int(mode.removeprefix("prefix"))
                        )
                    elif mode == "selective1":
                        if should_revise_first(updater, latest, provisional):
                            provisional = ar_prefix_update(ar, latest, provisional, 1)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    update_ms = 1000.0 * (time.perf_counter() - started)
                    schedule_changed = float(provisional != before_update)
                    scheduled_rows = {
                        int(xs_decode_action(int(value), MAXT)[0]) for value in provisional
                    }
                    new_rows = set(np.flatnonzero(newly_visible).tolist())
                    new_arrival_targets = float(len(new_rows))
                    new_arrival_scheduled = float(len(new_rows & scheduled_rows))
                    updated = True
            if provisional is None:
                latest = tokenize(adapt, get_obs(eng, debt), set(), 0).astype(np.float32)
                provisional = plan_from_tokens(ar, latest, args.max_steps)
            current = list(provisional)
            metrics = sample_state_metrics(eng, debt)
            reward += window_underuse_penalty(spent, 200.0, env_cfg)
            reward += window_service_penalty(metrics, env_cfg)
            cumulative += reward
            rows.append(
                {
                    "mode": mode,
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(seed),
                    "window": int(window),
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "drop_pct_active": float(metrics["drop_pct_active"]),
                    "tracked_targets": float(metrics["tracked_targets"]),
                    "mean_delay_active": float(metrics["mean_delay_active"]),
                    "search_fraction": float(searches / max(1, len(current))),
                    "update_ms": float(update_ms),
                    "deadline_miss_ms": float(max(0.0, update_ms - (200.0 - args.repair_start_ms))),
                    "schedule_changed": schedule_changed,
                    "new_arrival_targets": new_arrival_targets,
                    "new_arrival_scheduled": new_arrival_scheduled,
                }
            )
    finally:
        eng.close()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-checkpoint", type=Path, required=True)
    parser.add_argument("--updater-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", default="predictor,graph_replan,late_replan")
    parser.add_argument("--initials", default="20,40,60")
    parser.add_argument("--rates", default="2,3,4")
    parser.add_argument("--seeds", default="916")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--plan-start-ms", type=float, default=100.0)
    parser.add_argument("--repair-start-ms", type=float, default=170.0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_reward_args(parser)
    args = parser.parse_args()
    device = torch.device(args.device)
    ar = load_ar(args.ar_checkpoint, device)
    predictor = load_boundary(args.boundary_checkpoint, device)
    updater = load_updater(args.updater_checkpoint, device)
    graph_runner = Sparse64CudaGraphRunner(ar, args.max_steps, device) if device.type == "cuda" else None
    frames = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            for seed in parse_ints(args.seeds):
                for mode in [value.strip() for value in args.modes.split(",") if value.strip()]:
                    if mode == "graph_replan" and graph_runner is None:
                        raise ValueError("graph_replan requires CUDA")
                    frame = run_episode(
                        mode, ar, predictor, updater, graph_runner,
                        initial, rate, seed, args.windows, args
                    )
                    frames.append(frame)
                    print({"mode": mode, "initial": initial, "rate": rate,
                           "reward": float(frame.window_reward.mean()),
                           "drop": float(frame.drop_pct_active.mean()),
                           "delay": float(frame.mean_delay_active.mean()),
                           "update_ms": float(frame.update_ms.mean())}, flush=True)
    result = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    summary = result.groupby("mode", as_index=False).agg(
        reward_per_window=("window_reward", "mean"),
        drop_pct_active=("drop_pct_active", "mean"),
        mean_delay_active=("mean_delay_active", "mean"),
        search_fraction=("search_fraction", "mean"),
        update_ms=("update_ms", "mean"),
        deadline_miss_ms=("deadline_miss_ms", "mean"),
        schedule_changed=("schedule_changed", "mean"),
        new_arrival_targets=("new_arrival_targets", "mean"),
        new_arrival_scheduled=("new_arrival_scheduled", "mean"),
    )
    final = (
        result.sort_values("window").groupby(["mode", "initial", "rate", "seed"], as_index=False).tail(1)
        .groupby("mode", as_index=False).tracked_targets.mean().rename(columns={"tracked_targets": "final_tracked"})
    )
    summary = summary.merge(final, on="mode")
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
