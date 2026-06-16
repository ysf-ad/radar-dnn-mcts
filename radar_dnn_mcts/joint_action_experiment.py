from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare_action_heads_smoke import usable_targets
from exact_env_mutual import (
    EDFPlanner,
    ESTPlanner,
    MAXT,
    attach_env_obs,
    engine_env_cfg,
    env_cfg_for,
    xs_decode_action,
    xs_s_search_action,
    xs_x_search_action,
)
from final_radar_campaign import get_obs, run_fixed, summarize_window_df
from foundation_mcts_fair_eval import parse_floats, parse_ints, physical_candidates
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env, execute_first_valid_action, infer_elapsed_ms
from strict_window_report import sample_state_metrics
from two_sensor_physical_head_eval import PhysicalHeadPlanner, train_head
from pufferlib.ocean.radarxs import binding


JOINT_ACTION_BASE = 1_000_000
JOINT_ACTION_STRIDE = 1_000


def encode_joint_action(s_action: int, x_action: int) -> int:
    return int(JOINT_ACTION_BASE + int(s_action) * JOINT_ACTION_STRIDE + int(x_action))


def is_joint_action(action: int) -> bool:
    return int(action) >= JOINT_ACTION_BASE


def split_joint_action(action: int) -> tuple[int, int]:
    encoded = int(action) - JOINT_ACTION_BASE
    return int(encoded // JOINT_ACTION_STRIDE), int(encoded % JOINT_ACTION_STRIDE)


def action_duration(obs: dict, action: int) -> float:
    base, sensor = xs_decode_action(int(action), MAXT)
    if int(base) == 0:
        return 10.0
    if int(base) <= 0:
        return 10.0
    dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
    dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
    if sensor == 1:
        dt *= 0.5
    return max(1.0, dt)


def joint_duration(obs: dict, action: int) -> float:
    if not is_joint_action(action):
        return action_duration(obs, action)
    s_action, x_action = split_joint_action(action)
    return max(1.0, min(action_duration(obs, s_action), action_duration(obs, x_action)))


def execute_first_valid_action_joint(eng, plan, remaining_ms: float):
    if eng.term_buf[0] or remaining_ms <= 0:
        return 0.0, 0.0, None
    for a in plan:
        a = int(a)
        if not is_joint_action(a):
            return execute_first_valid_action(eng, [a], remaining_ms)
        obs_before = get_obs(eng)
        s_action, x_action = split_joint_action(a)
        s_base, _ = xs_decode_action(s_action, MAXT)
        x_base, _ = xs_decode_action(x_action, MAXT)
        if int(s_base) > 0 and int(x_base) > 0 and int(s_base) == int(x_base):
            continue
        eng.act_buf[0] = int(a)
        binding.vec_step(eng.env)
        obs_after = get_obs(eng)
        dt = infer_elapsed_ms(obs_before, obs_after)
        if dt <= 0.0:
            dt = joint_duration(obs_before, a)
        executed = int(a)
        dt = min(float(remaining_ms), float(dt))
        if dt <= 0.0:
            continue
        return float(eng.rew_buf[0]), float(dt), int(executed)
    return 0.0, 0.0, None


def execute_plan_until_budget_joint(eng, plan, budget_ms: float, search_debt_ms: float, planner_name: str, seed: int, window_idx: int):
    spent_ms = 0.0
    total_reward = 0.0
    search_actions = 0
    executed = 0
    rows = []
    slot = 0
    for action in plan:
        if spent_ms >= float(budget_ms) or bool(eng.term_buf[0]):
            break
        reward, dt, executed_action = execute_first_valid_action_joint(eng, [int(action)], float(budget_ms) - spent_ms)
        if executed_action is None or dt <= 0.0:
            continue
        total_reward += float(reward)
        spent_ms += float(dt)
        atoms = split_joint_action(executed_action) if is_joint_action(executed_action) else (int(executed_action),)
        is_search = [xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms]
        if any(is_search):
            search_debt_ms = 0.0
        else:
            search_debt_ms += float(dt)
        search_actions += int(any(is_search))
        executed += 1
        rows.append(
            {
                "planner": planner_name,
                "seed": int(seed),
                "bucket": int(window_idx),
                "slot": int(slot),
                "action": int(executed_action),
                "s_action": int(atoms[0]) if len(atoms) > 1 else -1,
                "x_action": int(atoms[1]) if len(atoms) > 1 else -1,
                "action_type": "Joint" if len(atoms) > 1 else "Atomic",
                "reward": float(reward),
                "dt_ms": float(dt),
            }
        )
        slot += 1
    return total_reward, spent_ms, search_debt_ms, executed, search_actions, rows


def run_fixed_joint(planner, name: str, initial: int, seed: int, windows: int, env_cfg: dict):
    eng = build_env(planner, int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
    eng.reset(seed=int(seed))
    debt = 0.0
    cumulative = 0.0
    win_rows = []
    action_rows = []
    try:
        for w in range(int(windows)):
            if bool(eng.term_buf[0]):
                break
            obs = get_obs(eng, debt)
            t0 = time.perf_counter()
            plan = planner.plan(obs, budget_ms=200)
            plan_ms = (time.perf_counter() - t0) * 1000.0
            reward, spent, debt, executed, searches, arows = execute_plan_until_budget_joint(eng, plan, 200.0, debt, name, int(seed), int(w))
            cumulative += float(reward)
            state = sample_state_metrics(eng, debt)
            win_rows.append(
                {
                    "planner": name,
                    "seed": int(seed),
                    "window": int(w),
                    "elapsed_ms": float((w + 1) * 200),
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "search_fraction": float(searches / max(1, executed)),
                    "planning_ms_per_decision": float(plan_ms),
                    "planning_ms_per_executed_action": float(plan_ms / max(1, executed)),
                    "executed_actions": int(executed),
                    "spent_ms": float(spent),
                    **state,
                }
            )
            for row in arows:
                row.update(window=int(w), elapsed_ms=float((w + 1) * 200))
            action_rows.extend(arows)
    finally:
        eng.close()
    return pd.DataFrame(win_rows), pd.DataFrame(action_rows)


class JointPhysicalHeadPlanner:
    def __init__(self, base: PhysicalHeadPlanner, per_sensor_top: int = 1):
        self.base = base
        self.env_cfg = dict(base.env_cfg)
        self.per_sensor_top = int(per_sensor_top)

    def _ranked(self, obs, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        score = self.base.score_actions(obs, selected=selected, elapsed=elapsed, search_count=search_count, track_count=track_count, last=last)
        ranked = [[], []]
        for action in physical_candidates(obs, top_k=MAXT):
            base, sensor = xs_decode_action(int(action), MAXT)
            if sensor is None or int(sensor) not in {0, 1} or int(base) < 0:
                continue
            if int(base) > 0 and int(base) in selected:
                continue
            ranked[int(sensor)].append((float(score[int(base), int(sensor)]), int(action)))
        for i in (0, 1):
            ranked[i].sort(reverse=True, key=lambda x: x[0])
            ranked[i] = ranked[i][: max(1, self.per_sensor_top)]
        return ranked[0], ranked[1]

    def plan(self, obs, budget_ms=200):
        obs = attach_env_obs(obs, self.env_cfg, True, True)
        selected: set[int] = set()
        plan = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        while elapsed < float(budget_ms) and len(plan) < 64:
            s_ranked, x_ranked = self._ranked(obs, selected, elapsed, search_count, track_count, last)
            if not s_ranked and not x_ranked:
                break
            best = None
            best_score = -np.inf
            for s_score, s_action in (s_ranked or [(-1e9, -1)]):
                for x_score, x_action in (x_ranked or [(-1e9, -1)]):
                    if s_action < 0 and x_action < 0:
                        continue
                    if s_action < 0:
                        cand = int(x_action)
                        score = float(x_score)
                    elif x_action < 0:
                        cand = int(s_action)
                        score = float(s_score)
                    else:
                        s_base, _ = xs_decode_action(int(s_action), MAXT)
                        x_base, _ = xs_decode_action(int(x_action), MAXT)
                        if int(s_base) > 0 and int(s_base) == int(x_base):
                            continue
                        cand = encode_joint_action(int(s_action), int(x_action))
                        score = float(s_score) + float(x_score)
                    if score > best_score:
                        best = cand
                        best_score = score
            if best is None:
                break
            plan.append(int(best))
            atoms = split_joint_action(best) if is_joint_action(best) else (int(best),)
            dt = joint_duration(obs, int(best))
            for atom in atoms:
                base, _sensor = xs_decode_action(int(atom), MAXT)
                if int(base) == 0:
                    search_count += 1
                elif int(base) > 0:
                    selected.add(int(base))
                    track_count += 1
                last = int(base)
            elapsed += max(1.0, float(dt))
        return plan if plan else [encode_joint_action(xs_s_search_action(MAXT), xs_x_search_action(MAXT))]


def evaluate(args):
    torch.set_num_threads(1)
    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    targets = usable_targets(Path(args.targets))
    model = train_head(str(args.variant), targets, args, torch.device("cpu"))
    if str(getattr(args, "save_model", "")):
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print({"saved_model": str(model_path)}, flush=True)

    rows = []
    windows = []
    actions = []
    for seed in parse_ints(args.eval_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                if args.revisit_scale is not None:
                    env_cfg["revisit_time_scale"] = float(args.revisit_scale)
                if args.dwell_scale is not None:
                    env_cfg["dwell_time_scale"] = float(args.dwell_scale)
                planners = {"EDF": EDFPlanner(MAXT), "EST": ESTPlanner(MAXT)}
                for bias in parse_floats(getattr(args, "search_biases", str(args.search_bias))):
                    suffix = f"{float(bias):+g}".replace("+", "p").replace("-", "m").replace(".", "p")
                    atomic = PhysicalHeadPlanner(model, str(args.variant), env_cfg, policy_weight=1.0, q_weight=float(args.q_weight), search_score_bias=float(bias))
                    planners[f"Atomic_sb{suffix}"] = atomic
                    planners[f"Joint_sb{suffix}"] = JointPhysicalHeadPlanner(atomic, per_sensor_top=int(args.per_sensor_top))
                for name, planner in planners.items():
                    if name.startswith("Joint"):
                        w, a = run_fixed_joint(planner, name, int(initial), int(seed), int(args.eval_windows), env_cfg)
                    else:
                        w, a = run_fixed(planner, name, int(initial), MAXT, int(seed), int(args.eval_windows), 200, engine_env_cfg(env_cfg))
                    s = summarize_window_df(w, "fixed")
                    row = {
                        "method": name,
                        "initial": int(initial),
                        "rate": float(rate),
                        "seed": int(seed),
                        "reward": float(s.get("reward_per_200ms_eq", np.nan)),
                        "search": float(s.get("search_fraction", np.nan)),
                        "tracked": float(s.get("mean_tracked_targets", np.nan)),
                        "drop": float(s.get("mean_drop_pct_active", np.nan)),
                        "delay": float(s.get("mean_delay_active", np.nan)),
                        "latency_ms": float(s.get("planning_ms_per_decision", np.nan)),
                    }
                    rows.append(row)
                    windows.append(w.assign(method=name, initial=int(initial), rate=float(rate), seed=int(seed)))
                    actions.append(a.assign(method=name, initial=int(initial), rate=float(rate), seed=int(seed)) if not a.empty else a)
                    print(row, flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out, index=False)
    pd.concat(windows, ignore_index=True).to_csv(out.with_name(out.stem + "_windows.csv"), index=False)
    if actions:
        pd.concat(actions, ignore_index=True).to_csv(out.with_name(out.stem + "_actions.csv"), index=False)
    summary = raw.groupby("method", as_index=False).agg(
        reward=("reward", "mean"),
        search=("search", "mean"),
        tracked=("tracked", "mean"),
        drop=("drop", "mean"),
        delay=("delay", "mean"),
        latency_ms=("latency_ms", "mean"),
        n=("reward", "size"),
    ).sort_values("reward", ascending=False)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="CreateValid1/results/action_attention_exact_argmax_fullgrid_seed916_targets.pt")
    ap.add_argument("--out", default="CreateValid1/results/pq1_alphazero_r60_rates136/joint_action_smoke.csv")
    ap.add_argument("--variant", default="two_row_action_attention_qpolicy_factored_loss")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--eval-seeds", default="916")
    ap.add_argument("--eval-windows", type=int, default=20)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--train-steps", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--q-loss-weight", type=float, default=0.25)
    ap.add_argument("--value-loss-weight", type=float, default=0.25)
    ap.add_argument("--search-calibration-weight", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=45)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    ap.add_argument("--q-weight", type=float, default=1.0)
    ap.add_argument("--search-bias", type=float, default=0.0)
    ap.add_argument("--search-biases", default="0")
    ap.add_argument("--per-sensor-top", type=int, default=3)
    ap.add_argument("--revisit-scale", type=float, default=None)
    ap.add_argument("--dwell-scale", type=float, default=None)
    ap.add_argument("--save-model", default="")
    args = ap.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
