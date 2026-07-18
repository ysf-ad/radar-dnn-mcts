"""Causal real-time scheduling evaluation with a learned boundary predictor."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from boundary_predictor import BoundaryStatePredictor
from distill_sparse64_sequence_decoder import Sparse64SequenceDecoder, action_to_prev_class
from eval_action_attention_muzero_g import (
    build_env,
    execute_plan_until_budget_joint_shaped,
    get_obs,
    sample_state_metrics,
    window_service_penalty,
    window_underuse_penalty,
)
from exact_env_mutual import MAXT, _DummyPlanner, engine_env_cfg, env_cfg_for, xs_s_search_action, xs_s_track_action
from foundation_mcts_fair_eval import parse_floats, parse_ints
from mutual_features import SLOT_DIM, tokenize
from penalty_window_quota_learner_eval import make_exact_args
from realistic_reward_retrain import adapter


def load_ar(path: Path, device: torch.device) -> Sparse64SequenceDecoder:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("args", {}) if isinstance(payload, dict) else {}
    model = Sparse64SequenceDecoder(
        d_model=int(cfg.get("d_model", 96)),
        nhead=int(cfg.get("nhead", 4)),
        nlayers=int(cfg.get("nlayers", 2)),
    ).to(device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=False)
    return model.eval()


def load_boundary(path: Path, device: torch.device) -> BoundaryStatePredictor:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["state_dict"]
    d_model = int(state["token_in.weight"].shape[0])
    layer_ids = {int(key.split(".")[2]) for key in state if key.startswith("encoder.layers.")}
    model = BoundaryStatePredictor(
        token_dim=int(payload["token_dim"]),
        max_action_id=int(payload["max_action_id"]),
        max_suffix=int(payload["max_suffix"]),
        max_tokens=int(payload.get("max_tokens", 101)),
        use_row_embeddings=bool(payload.get("use_row_embeddings", "row_embedding.weight" in state)),
        d_model=d_model,
        layers=max(layer_ids) + 1 if layer_ids else 1,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def token_slot(root: torch.Tensor, elapsed: float, searches: int, tracks: int, last: int) -> torch.Tensor:
    active = root[:, 4] > 0.5
    active[0] = False
    dwell = root[:, 2].clamp(0.0, 2.0) * 100.0
    deadline = root[:, 1] * 3000.0
    positive = deadline[(active) & (deadline > 0)]
    slot = torch.zeros((1, SLOT_DIM), dtype=root.dtype, device=root.device)
    slot[0, 0] = float(elapsed) / 200.0
    slot[0, 1] = float(searches) / 20.0
    slot[0, 2] = float(tracks) / 100.0
    slot[0, 3] = float(last == 0)
    slot[0, 4] = active.float().sum() / 100.0
    slot[0, 5] = slot[0, 4]
    slot[0, 6] = (dwell[active].sum() / 4000.0).clamp(max=2.0)
    slot[0, 7] = positive.min() / 3000.0 if positive.numel() else 0.0
    slot[0, 8:11] = root[0, 9:12]
    return slot


@torch.inference_mode()
def plan_from_tokens(model: Sparse64SequenceDecoder, tokens: np.ndarray, max_steps: int = 32) -> list[int]:
    device = next(model.parameters()).device
    root = torch.as_tensor(tokens, dtype=torch.float32, device=device)
    root_batch = root.unsqueeze(0)
    cls, target_tokens, token_active = model.encode(root_batch)
    dwell = root[:, 2].clamp(0.01, 2.0) * 100.0
    selected: set[int] = set()
    elapsed = 0.0
    searches = tracks = 0
    last = -1
    prev_class = torch.zeros(1, dtype=torch.long, device=device)
    prev_row = torch.zeros(1, dtype=torch.float32, device=device)
    hidden = None
    actions: list[int] = []
    while elapsed < 200.0 and len(actions) < int(max_steps):
        slot = token_slot(root, elapsed, searches, tracks, last)
        type_logits, target_logits, hidden = model.score_step(
            cls, target_tokens, slot, prev_class, prev_row, hidden
        )
        valid = token_active[0].clone()
        valid[0] = False
        for row in selected:
            valid[row] = False
        target_logp = torch.log_softmax(target_logits[0], dim=-1).masked_fill(~valid, -1.0e9)
        best = int(target_logp.argmax().item())
        type_logp = torch.log_softmax(type_logits[0], dim=-1)
        choose_track = bool(valid.any()) and float(type_logp[1] + target_logp[best]) > float(type_logp[0])
        row = best if choose_track else 0
        if row <= 0:
            actions.append(xs_s_search_action(MAXT))
            elapsed += 10.0
            searches += 1
            last = 0
        else:
            actions.append(xs_s_track_action(row, MAXT))
            elapsed += float(max(1.0, dwell[row].item()))
            tracks += 1
            selected.add(row)
            last = row
        prev_class.fill_(action_to_prev_class(row))
        prev_row.fill_(float(row) / float(MAXT))
    return actions or [xs_s_search_action(MAXT)]


def predict_tokens(
    predictor: BoundaryStatePredictor,
    midpoint: np.ndarray,
    suffix: list[int],
    remaining_ms: float,
    target_features: tuple[int, ...] = (0, 1, 5, 6, 7),
    forecast_search_row: bool = True,
) -> tuple[np.ndarray, float]:
    device = next(predictor.parameters()).device
    width = predictor.max_suffix
    actions = np.zeros((1, width), dtype=np.int64)
    count = min(width, len(suffix))
    if count:
        from exact_env_mutual import MAXT, xs_decode_action

        canonical = []
        for simulator_action in suffix[:count]:
            row, sensor = xs_decode_action(int(simulator_action), MAXT)
            if row == 0 and sensor == 0:
                canonical.append(0)
            elif row > 0 and sensor == 0:
                canonical.append(2 * int(row))
            else:
                raise ValueError(f"non-S action in S-only suffix: {simulator_action}")
        actions[0, :count] = np.asarray(canonical, dtype=np.int64)
    mask = np.arange(width)[None, :] < count
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = predictor(
            torch.as_tensor(midpoint[None], dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device),
            torch.tensor([remaining_ms], dtype=torch.float32, device=device),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = 1000.0 * (time.perf_counter() - start)
    raw = output[0].detach().cpu().numpy()
    # A causal boundary forecast must not invent target identities, visibility,
    # sensor validity, dwell, priority, or range.  Only continuous control
    # pressures for targets already visible at the midpoint are forecast.
    # Future stochastic arrivals remain unknown until the next observation.
    projected = np.asarray(midpoint, dtype=np.float32).copy()
    if forecast_search_row:
        projected[0, :8] = raw[0, :8]
    active = projected[1:, 4] > 0.5
    for feature in target_features:
        projected[1:, feature][active] = raw[1:, feature][active]
    return projected, latency


def run_episode(mode: str, model, predictor, initial: int, rate: float, seed: int, windows: int, args) -> pd.DataFrame:
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = False
    exact_args.single_sensor = True
    env_cfg = env_cfg_for(float(rate), exact_args)
    env_cfg["enable_x_band"] = 0
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    adapt = adapter()
    debt = cumulative = 0.0
    current = plan_from_tokens(model, tokenize(adapt, get_obs(eng, debt), set(), 0))
    rows = []
    try:
        for window in range(int(windows)):
            spent = reward = 0.0
            searches = 0
            next_plan = None
            prediction_ms = 0.0
            planning_ms = 0.0
            midpoint_tokens = None
            for action_index, action in enumerate(current):
                result = execute_plan_until_budget_joint_shaped(
                    eng, [int(action)], 200.0 - spent, debt, f"Realtime {mode}", seed, window, env_cfg
                )
                step_reward, dt, debt, count, search_count, _ = result
                if count <= 0 or dt <= 0:
                    continue
                reward += float(step_reward)
                spent += float(dt)
                searches += int(search_count)
                if next_plan is None and spent >= float(args.plan_start_ms):
                    mid_obs = get_obs(eng, debt)
                    midpoint_tokens = tokenize(adapt, mid_obs, set(), 0).astype(np.float32)
                    suffix = [int(value) for value in current[action_index + 1 :]]
                    if device := next(model.parameters()).device:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                    plan_started = time.perf_counter()
                    if mode == "naive":
                        next_plan = plan_from_tokens(model, midpoint_tokens, args.max_steps)
                    elif mode == "learned":
                        predicted, prediction_ms = predict_tokens(
                            predictor,
                            midpoint_tokens,
                            suffix,
                            max(0.0, 200.0 - spent),
                            tuple(int(value) for value in args.forecast_features.split(",") if value.strip()),
                            bool(args.forecast_search_row),
                        )
                        next_plan = plan_from_tokens(model, predicted, args.max_steps)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    planning_ms = 1000.0 * (time.perf_counter() - plan_started)
            boundary_obs = get_obs(eng, debt)
            boundary_tokens = tokenize(adapt, boundary_obs, set(), 0).astype(np.float32)
            if mode in ("synchronous", "oracle"):
                next_plan = plan_from_tokens(model, boundary_tokens, args.max_steps)
            elif next_plan is None:
                next_plan = plan_from_tokens(model, boundary_tokens, args.max_steps)
            metrics = sample_state_metrics(eng, debt)
            reward += window_underuse_penalty(spent, 200.0, env_cfg)
            reward += window_service_penalty(metrics, env_cfg)
            cumulative += reward
            rows.append({
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
                "prediction_ms": float(prediction_ms),
                "planning_ms": float(planning_ms),
                "deadline_miss_ms": float(max(0.0, planning_ms - max(0.0, 200.0 - args.plan_start_ms))),
                "spent_ms": float(spent),
            })
            current = list(next_plan)
    finally:
        eng.close()
    return pd.DataFrame(rows)


def add_reward_args(parser):
    from canonical_scheduler_contract import add_canonical_reward_args

    add_canonical_reward_args(parser)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ar-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", default="synchronous,naive,learned,oracle")
    parser.add_argument("--initials", default="20,40,60")
    parser.add_argument("--rates", default="2,3,4")
    parser.add_argument("--seeds", default="916")
    parser.add_argument("--windows", type=int, default=100)
    parser.add_argument("--plan-start-ms", type=float, default=100.0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--forecast-features", default="0,1,5,6,7")
    parser.add_argument("--forecast-search-row", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_reward_args(parser)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = load_ar(args.ar_checkpoint, device)
    predictor = load_boundary(args.boundary_checkpoint, device)
    frames = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            for seed in parse_ints(args.seeds):
                for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
                    frame = run_episode(mode, model, predictor, initial, rate, seed, args.windows, args)
                    frames.append(frame)
                    print({"mode": mode, "initial": initial, "rate": rate, "seed": seed,
                           "reward_per_window": float(frame.window_reward.mean()),
                           "drop_pct_active": float(frame.drop_pct_active.mean()),
                           "tracked_targets": float(frame.tracked_targets.iloc[-1]),
                           "mean_delay_active": float(frame.mean_delay_active.mean()),
                           "prediction_ms": float(frame.prediction_ms.mean())}, flush=True)
    result = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    summary = result.groupby("mode", as_index=False).agg(
        reward_per_window=("window_reward", "mean"),
        drop_pct_active=("drop_pct_active", "mean"),
        mean_delay_active=("mean_delay_active", "mean"),
        search_fraction=("search_fraction", "mean"),
        prediction_ms=("prediction_ms", "mean"),
        planning_ms=("planning_ms", "mean"),
        deadline_miss_ms=("deadline_miss_ms", "mean"),
    )
    final_tracked = (
        result.sort_values("window")
        .groupby(["mode", "initial", "rate", "seed"], as_index=False)
        .tail(1)
        .groupby("mode", as_index=False)["tracked_targets"]
        .mean()
        .rename(columns={"tracked_targets": "final_tracked"})
    )
    summary = summary.merge(final_tracked, on="mode", how="left")
    summary.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
