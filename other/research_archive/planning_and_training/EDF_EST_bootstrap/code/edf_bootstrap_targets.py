from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alphazero_orthodox import save_targets
from exact_env_mutual import EDFPlanner, MAXT, _DummyPlanner, attach_env_obs, env_cfg_for, xs_decode_action
from final_radar_campaign import get_obs
from foundation_mcts_fair_eval import apply_token_action_mask, parse_floats, parse_ints
from mutual_features import slot_features, tokenize
from mutual_foundation import SearchTarget
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env, execute_first_valid_action
from realistic_reward_retrain import adapter
from strict_window_report import sample_state_metrics


def dense_state_reward(eng, debt: float) -> float:
    metrics = sample_state_metrics(eng, float(debt))
    tracked = float(metrics.get("tracked_targets", 0.0))
    dropped_pct = float(metrics.get("drop_pct_active", 0.0))
    mean_delay = float(metrics.get("mean_delay_active", 0.0))
    return 0.10 * tracked - 0.25 * dropped_pct - 0.20 * mean_delay


def target_from_action(
    obs: dict,
    selected: set[int],
    spent: float,
    search_count: int,
    track_count: int,
    last: int,
    action: int,
    ret: float,
    initial: int,
    rate: float,
    seed: int,
    window: int,
    action_index: int,
    symmetrize_unspecified_sensor: bool,
) -> SearchTarget | None:
    base, sensor = xs_decode_action(int(action), MAXT)
    if int(base) < 0:
        return None
    base = int(np.clip(int(base), 0, MAXT))
    if sensor is None and bool(symmetrize_unspecified_sensor):
        sensor_weights = [(0, 0.5), (1, 0.5)]
    else:
        sidx = 0 if sensor is None else int(np.clip(int(sensor), 0, 1))
        sensor_weights = [(sidx, 1.0)]

    pi = np.zeros((MAXT + 1,), dtype=np.float32)
    q = np.zeros((MAXT + 1,), dtype=np.float32)
    q_mask = np.zeros((MAXT + 1,), dtype=np.float32)
    sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
    sensor_q = np.zeros((MAXT + 1, 2), dtype=np.float32)
    sensor_q_mask = np.zeros((MAXT + 1, 2), dtype=np.float32)
    pi[base] = 1.0
    q[base] = float(ret)
    q_mask[base] = 1.0
    for sidx, weight in sensor_weights:
        sensor_pi[base, int(sidx)] = float(weight)
        sensor_q[base, int(sidx)] = float(ret)
        sensor_q_mask[base, int(sidx)] = 1.0

    adapt = adapter()
    tok = tokenize(adapt, obs, selected=selected, search_count=int(search_count)).astype(np.float32)
    slot = slot_features(obs, float(spent), int(search_count), int(track_count), int(last), 200.0).astype(np.float32)
    sensor_pi, sensor_q_mask = apply_token_action_mask(tok, sensor_pi, sensor_q_mask)
    if float(sensor_pi.sum()) <= 0.0:
        return None
    return SearchTarget(
        tok,
        slot,
        pi,
        q,
        q_mask,
        int(search_count),
        int(track_count),
        reward=0.0,
        ret=float(ret),
        sensor_pi=sensor_pi,
        sensor_q=sensor_q,
        sensor_q_mask=sensor_q_mask,
        initial=int(initial),
        rate=float(rate),
        seed=int(seed),
        window=int(window),
        action_index=int(action_index),
    )


def collect_cell(args, exact_args, initial: int, rate: float, seed: int) -> tuple[list[SearchTarget], list[dict]]:
    env_cfg = env_cfg_for(float(rate), exact_args)
    env_cfg["enable_x_band"] = 1
    planner = EDFPlanner(MAXT)
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, env_cfg)
    eng.reset(seed=int(seed))
    debt = 0.0
    targets: list[SearchTarget] = []
    rows: list[dict] = []
    action_index = 0
    try:
        for window in range(int(args.windows)):
            if bool(eng.term_buf[0]) or len(targets) >= int(args.max_targets):
                break
            spent = 0.0
            selected: set[int] = set()
            search_count = 0
            track_count = 0
            last = -1
            while spent < 200.0 and not bool(eng.term_buf[0]) and len(targets) < int(args.max_targets):
                obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                plan = list(planner.plan(obs, budget_ms=int(max(1.0, 200.0 - spent))))
                reward, dt, executed = execute_first_valid_action(eng, plan, 200.0 - spent)
                if executed is None or float(dt) <= 0.0:
                    break
                base, _sensor = xs_decode_action(int(executed), MAXT)
                next_debt = 0.0 if int(base) == 0 else float(debt) + float(dt)
                ret = dense_state_reward(eng, next_debt) if str(args.return_mode) == "dense_state" else float(reward)
                target = target_from_action(
                    obs,
                    selected,
                    spent,
                    search_count,
                    track_count,
                    last,
                    int(executed),
                    ret,
                    int(initial),
                    float(rate),
                    int(seed),
                    int(window),
                    int(action_index),
                    bool(args.symmetrize_unspecified_sensor),
                )
                if target is not None and int(window) >= int(args.collect_start_window):
                    targets.append(target)
                    rows.append(
                        {
                            "initial": int(initial),
                            "rate": float(rate),
                            "seed": int(seed),
                            "window": int(window),
                            "action_index": int(action_index),
                            "base": int(base),
                            "ret": float(ret),
                            "search_mass": float(target.sensor_pi[0, :].sum()),
                            "x_mass": float(target.sensor_pi[:, 1].sum()),
                        }
                    )
                    action_index += 1
                debt = float(next_debt)
                spent += float(dt)
                if int(base) == 0:
                    search_count += 1
                elif int(base) > 0:
                    selected.add(int(base))
                    track_count += 1
                last = int(base)
    finally:
        eng.close()
    return targets, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-out", default="CreateValid1/results/edf_bootstrap_targets.pt")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="3")
    ap.add_argument("--eval-seeds", default="903")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--collect-start-window", type=int, default=5)
    ap.add_argument("--max-targets", type=int, default=512)
    ap.add_argument("--return-mode", choices=["env", "dense_state"], default="dense_state")
    ap.add_argument("--symmetrize-unspecified-sensor", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    all_targets: list[SearchTarget] = []
    all_rows: list[dict] = []
    for seed in parse_ints(args.eval_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                targets, rows = collect_cell(args, exact_args, int(initial), float(rate), int(seed))
                all_targets.extend(targets)
                all_rows.extend(rows)
                print({"seed": seed, "initial": initial, "rate": rate, "targets": len(targets)}, flush=True)
                if len(all_targets) >= int(args.max_targets):
                    all_targets = all_targets[: int(args.max_targets)]
                    all_rows = all_rows[: int(args.max_targets)]
                    break
            if len(all_targets) >= int(args.max_targets):
                break
        if len(all_targets) >= int(args.max_targets):
            break

    out = Path(args.targets_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_targets(out, all_targets)
    pd.DataFrame(all_rows).to_csv(out.with_suffix(".csv"), index=False)
    print({"saved": str(out), "targets": len(all_targets)}, flush=True)


if __name__ == "__main__":
    main()
