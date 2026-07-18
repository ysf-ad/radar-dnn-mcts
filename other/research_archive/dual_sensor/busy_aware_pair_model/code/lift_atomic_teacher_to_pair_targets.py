from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from alphazero_orthodox import save_targets
from exact_env_mutual import (
    MAXT,
    attach_env_obs,
    engine_env_cfg,
    env_cfg_for,
    xs_decode_action,
    xs_s_search_action,
    xs_s_track_action,
    xs_x_search_action,
    xs_x_track_action,
)
from final_radar_campaign import get_obs
from joint_action_experiment import encode_joint_action, execute_first_valid_action_joint
from mutual_foundation import SearchTarget
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env, execute_first_valid_action
from two_sensor_physical_head_eval import adapter, slot_features, tokenize


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def feasible_track_actions(obs: dict, base: int) -> list[tuple[int, int]]:
    if int(base) <= 0:
        return []
    ranges = np.asarray(obs.get("target_range", []), dtype=np.float32)
    rng = float(ranges[int(base) - 1]) if int(base) - 1 < len(ranges) else 1e30
    out = []
    if float(obs.get("s_band_busy_ms", 0.0)) <= 0.0 and 10_000_000.0 < rng < 184_000_000.0:
        out.append((0, xs_s_track_action(int(base), MAXT)))
    if (
        int(obs.get("enable_x_band", 0))
        and float(obs.get("x_band_busy_ms", 0.0)) <= 0.0
        and 5_000_000.0 < rng < 100_000_000.0
    ):
        out.append((1, xs_x_track_action(int(base), MAXT)))
    return out


def choose_primary(obs: dict, base: int) -> tuple[int, int]:
    if int(base) == 0:
        if float(obs.get("s_band_busy_ms", 0.0)) <= 0.0:
            return 0, xs_s_search_action(MAXT)
        return 1, xs_x_search_action(MAXT)
    feasible = feasible_track_actions(obs, int(base))
    if feasible:
        return feasible[0]
    return 0, xs_s_search_action(MAXT)


def edf_fill(obs: dict, sensor: int, selected_base: int) -> tuple[int, int]:
    if int(sensor) == 0 and float(obs.get("s_band_busy_ms", 0.0)) > 0.0:
        return 0, xs_s_search_action(MAXT)
    if int(sensor) == 1 and (not int(obs.get("enable_x_band", 0)) or float(obs.get("x_band_busy_ms", 0.0)) > 0.0):
        return 0, xs_x_search_action(MAXT)
    active = np.asarray(obs.get("active_mask", []), dtype=bool)
    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
    ranges = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
    candidates = []
    for idx, ok in enumerate(active[:MAXT]):
        base = idx + 1
        if not bool(ok) or base == int(selected_base) or idx >= len(deadline) or float(deadline[idx]) < 0.0:
            continue
        rng = float(ranges[idx]) if idx < len(ranges) else 1e30
        if int(sensor) == 0 and 10_000_000.0 < rng < 184_000_000.0:
            candidates.append((float(deadline[idx]), base, xs_s_track_action(base, MAXT)))
        if int(sensor) == 1 and 5_000_000.0 < rng < 100_000_000.0:
            candidates.append((float(deadline[idx]), base, xs_x_track_action(base, MAXT)))
    if candidates:
        _deadline, base, action = sorted(candidates, key=lambda x: (x[0], x[1]))[0]
        return int(base), int(action)
    return 0, xs_s_search_action(MAXT) if int(sensor) == 0 else xs_x_search_action(MAXT)


def make_pair_target(adapt, obs: dict, atomic_action: int, initial: int, rate: float, seed: int, window: int, slot_idx: int, elapsed: float, search_count: int, track_count: int, last: int) -> SearchTarget | None:
    if float(obs.get("s_band_busy_ms", 0.0)) > 0.0:
        return None
    if not int(obs.get("enable_x_band", 0)) or float(obs.get("x_band_busy_ms", 0.0)) > 0.0:
        return None
    base, _sensor = xs_decode_action(int(atomic_action), MAXT)
    if int(base) < 0:
        return None
    primary_sensor, primary_action = choose_primary(obs, int(base))
    primary_base, _ = xs_decode_action(int(primary_action), MAXT)
    other_sensor = 1 - int(primary_sensor)
    other_base, other_action = edf_fill(obs, other_sensor, int(primary_base))
    if int(primary_sensor) == 0:
        s_base, x_base = int(primary_base), int(other_base)
        s_action, x_action = int(primary_action), int(other_action)
    else:
        s_base, x_base = int(other_base), int(primary_base)
        s_action, x_action = int(other_action), int(primary_action)
    if s_base > 0 and s_base == x_base:
        return None

    tok = tokenize(adapt, obs, selected=set(), search_count=int(search_count)).astype(np.float32)
    slot = slot_features(obs, float(elapsed), int(search_count), int(track_count), int(last), 200.0).astype(np.float32)
    sensor_pi = np.zeros((MAXT + 1, 2), dtype=np.float32)
    pair_pi = np.zeros((MAXT + 1, MAXT + 1), dtype=np.float32)
    sensor_pi[s_base, 0] = 1.0
    sensor_pi[x_base, 1] = 1.0
    pair_pi[s_base, x_base] = 1.0
    pi = sensor_pi.sum(axis=1)
    pi = pi / max(1e-6, float(pi.sum()))
    zeros = np.zeros((MAXT + 1,), dtype=np.float32)
    sensor_zeros = np.zeros((MAXT + 1, 2), dtype=np.float32)
    target = SearchTarget(
        x=tok,
        slot=slot,
        pi=pi.astype(np.float32),
        q=zeros.copy(),
        q_mask=(pi > 0).astype(np.float32),
        search_count=int(search_count),
        track_count=int(track_count),
        reward=0.0,
        ret=0.0,
        sensor_pi=sensor_pi,
        sensor_q=sensor_zeros.copy(),
        sensor_q_mask=(sensor_pi > 0).astype(np.float32),
        initial=int(initial),
        rate=float(rate),
        seed=int(seed),
        window=int(window),
        action_index=int(slot_idx),
        pair_pi=pair_pi,
    )
    return target


def collect(args) -> list[SearchTarget]:
    actions = pd.read_csv(args.actions)
    actions = actions[actions["method"].astype(str) == str(args.method)]
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    adapt = adapter()
    targets: list[SearchTarget] = []
    rows = []
    for initial in parse_ints(args.initials):
        for rate in parse_floats(args.rates):
            cell = actions[(actions.initial.astype(int) == int(initial)) & (actions.rate.astype(float) == float(rate))].copy()
            if cell.empty:
                continue
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 1
            env_cfg["use_arrival_token_feature"] = 1.0
            eng = build_env(None, int(initial), MAXT, int(args.seed), 200, engine_env_cfg(env_cfg))
            eng.reset(seed=int(args.seed))
            debt = 0.0
            try:
                for window, wrows in cell.sort_values(["window", "slot"]).groupby("window"):
                    elapsed = 0.0
                    search_count = 0
                    track_count = 0
                    last = -1
                    for item in wrows.itertuples(index=False):
                        if bool(eng.term_buf[0]):
                            break
                        obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                        target = make_pair_target(
                            adapt,
                            obs,
                            int(getattr(item, "action")),
                            int(initial),
                            float(rate),
                            int(args.seed),
                            int(window),
                            int(getattr(item, "slot")),
                            float(elapsed),
                            int(search_count),
                            int(track_count),
                            int(last),
                        )
                        if target is not None:
                            targets.append(target)
                            p = np.asarray(target.pair_pi)
                            rows.append(
                                {
                                    "initial": int(initial),
                                    "rate": float(rate),
                                    "window": int(window),
                                    "slot": int(getattr(item, "slot")),
                                    "s_base": int(np.argwhere(p.sum(axis=1) > 0)[0][0]),
                                    "x_base": int(np.argwhere(p.sum(axis=0) > 0)[0][0]),
                                }
                            )
                        reward, dt, executed = execute_first_valid_action(eng, [int(getattr(item, "action"))], 200.0 - elapsed)
                        if executed is None or float(dt) <= 0.0:
                            continue
                        base_action, _ = xs_decode_action(int(executed), MAXT)
                        if int(base_action) == 0:
                            debt = 0.0
                            search_count += 1
                        else:
                            debt += float(dt)
                            if int(base_action) > 0:
                                track_count += 1
                        last = int(base_action)
                        elapsed += float(dt)
            finally:
                eng.close()
    out = Path(args.out)
    save_targets(out, targets)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    return targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="two_row_action_attention_factored_loss_PQ")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--seed", type=int, default=916)
    ap.add_argument("--windows", type=int, default=100)
    args = ap.parse_args()
    targets = collect(args)
    print({"saved": args.out, "targets": len(targets)}, flush=True)


if __name__ == "__main__":
    main()
