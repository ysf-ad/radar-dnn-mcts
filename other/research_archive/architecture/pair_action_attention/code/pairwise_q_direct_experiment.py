from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
CODE = ROOT / "CreateValid1" / "experiments" / "code" / "model_code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from best_model_joint_vs_seq_ablation import (
    WorkConservingAsyncBeamPlanner,
    WorkConservingAsyncCoupledPlanner,
    encode_joint_action,
    execute_plan_until_budget_joint_compatible,
    execute_first_valid_action_joint,
    run_exact_rescore_grid_joint,
)
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, _DummyPlanner, attach_env_obs, engine_env_cfg, env_cfg_for, xs_decode_action, xs_s_search_action, xs_x_search_action
from final_radar_campaign import get_obs, summarize_window_df
from foundation_mcts_fair_eval import physical_candidates, run_heuristic
from joint_action_experiment import is_joint_action, split_joint_action
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env
from strict_window_report import sample_state_metrics
from two_sensor_physical_head_eval import PhysicalHeadPlanner, make_physical_model, state_potential
from pufferlib.ocean.radarxs import binding


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


class PairQNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def action_duration(obs: dict, action: int) -> float:
    base, sensor = xs_decode_action(int(action), MAXT)
    if int(base) == 0:
        return 10.0
    if int(base) <= 0:
        return 1.0
    dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
    dt = float(dwell[int(base) - 1]) if int(base) - 1 < len(dwell) else 10.0
    if int(sensor) == 1:
        dt *= 0.5
    return max(1.0, float(dt))


def row_features(obs: dict, base: int, sensor: int, score: float) -> list[float]:
    active = np.asarray(obs.get("active_mask", []), dtype=bool)
    desired = np.asarray(obs.get("t_desired", []), dtype=np.float32)
    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
    dwell = np.asarray(obs.get("t_dwell", []), dtype=np.float32)
    ranges = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
    if int(base) <= 0:
        return [1.0, 0.0, 0.0, 1.0, float(sensor), float(score), 10.0 / 200.0, 0.0, 0.0, 0.0, 0.0]
    idx = int(base) - 1
    is_active = float(idx < len(active) and bool(active[idx]))
    t_des = float(desired[idx]) if idx < len(desired) else 0.0
    t_dead = float(deadline[idx]) if idx < len(deadline) else 0.0
    dt = float(dwell[idx]) if idx < len(dwell) else 10.0
    if int(sensor) == 1:
        dt *= 0.5
    rng = float(ranges[idx]) if idx < len(ranges) else 0.0
    return [
        0.0,
        is_active,
        float(base) / float(MAXT),
        float(sensor),
        float(score),
        max(-2.0, min(2.0, t_des / 1000.0)),
        max(-2.0, min(2.0, t_dead / 1000.0)),
        max(0.0, min(2.0, dt / 100.0)),
        max(0.0, min(2.0, rng / 100_000_000.0)),
        max(0.0, -t_des / max(1.0, t_dead - t_des + 1e-6)),
        max(0.0, 300.0 - t_dead) / 300.0,
    ]


def pair_features(obs: dict, s_score: float, s_base: int, s_action: int, x_score: float, x_base: int, x_action: int) -> np.ndarray:
    active = float(np.asarray(obs.get("active_mask", []), dtype=bool)[:MAXT].sum())
    s_feat = row_features(obs, int(s_base), 0, float(s_score))
    x_feat = row_features(obs, int(x_base), 1, float(x_score))
    same_target = float(int(s_base) > 0 and int(s_base) == int(x_base))
    both_search = float(int(s_base) == 0 and int(x_base) == 0)
    one_search = float((int(s_base) == 0) ^ (int(x_base) == 0))
    d_s = action_duration(obs, int(s_action))
    d_x = action_duration(obs, int(x_action))
    extra = [
        active / float(MAXT),
        float(obs.get("arrival_rate", 0.0)) / 10.0,
        float(obs.get("s_band_busy_ms", 0.0)) / 200.0,
        float(obs.get("x_band_busy_ms", 0.0)) / 200.0,
        same_target,
        both_search,
        one_search,
        min(d_s, d_x) / 100.0,
        max(d_s, d_x) / 100.0,
        abs(d_s - d_x) / 100.0,
        float(s_score) + float(x_score),
        float(s_score) * float(x_score),
    ]
    return np.asarray([*s_feat, *x_feat, *extra], dtype=np.float32)


def ranked_for_sensor(base_planner: PhysicalHeadPlanner, obs: dict, sensor: int, selected: set[int], per_sensor_top: int, force_search: bool):
    scores = base_planner.score_actions(obs, selected=selected)
    ranked = []
    for action in physical_candidates(obs, top_k=MAXT):
        base, sid = xs_decode_action(int(action), MAXT)
        if int(sid) != int(sensor) or int(base) < 0:
            continue
        if int(base) > 0 and int(base) in selected:
            continue
        ranked.append((float(scores[int(base), int(sensor)]), int(base), int(action)))
    ranked.sort(reverse=True, key=lambda x: x[0])
    out = ranked[: max(1, int(per_sensor_top))]
    search_action = xs_s_search_action(MAXT) if int(sensor) == 0 else xs_x_search_action(MAXT)
    if bool(force_search) and not any(int(a) == int(search_action) for _score, _base, a in out):
        for item in ranked:
            if int(item[2]) == int(search_action):
                out.append(item)
                break
    return out


def candidate_pairs(base_planner: PhysicalHeadPlanner, obs: dict, selected: set[int], per_sensor_top: int, force_search: bool):
    obs = attach_env_obs(dict(obs), base_planner.env_cfg, True, True)
    s_ranked = ranked_for_sensor(base_planner, obs, 0, selected, per_sensor_top, force_search)
    x_ranked = ranked_for_sensor(base_planner, obs, 1, selected, per_sensor_top, force_search) if int(obs.get("enable_x_band", 0)) else []
    out = []
    for s_score, s_base, s_action in s_ranked:
        for x_score, x_base, x_action in x_ranked:
            if int(s_base) > 0 and int(s_base) == int(x_base):
                continue
            out.append((s_score, s_base, s_action, x_score, x_base, x_action))
    return out


def exact_pair_label(eng, plan: list[int], debt: float, horizon_ms: float, potential_weight: float) -> float:
    root = binding.vec_snapshot(eng.env)
    try:
        reward, _spent, next_debt, _executed, _searches, _rows = execute_plan_until_budget_joint_compatible(
            eng,
            [int(a) for a in plan],
            float(horizon_ms),
            float(debt),
            "pair_label",
            0,
            0,
        )
        label = float(reward) + float(potential_weight) * state_potential(eng, float(next_debt))
    finally:
        binding.vec_restore(eng.env, root)
    return float(label)


def collect_pair_data(args, base_planner: PhysicalHeadPlanner, exact_args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    groups = []
    tail_planner = WorkConservingAsyncBeamPlanner(
        base_planner,
        per_sensor_top=int(args.per_sensor_top),
        beams=max(1, int(args.per_sensor_top) * int(args.per_sensor_top)),
        include_search_candidate=True,
    )
    for seed in parse_ints(args.train_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                eng.reset(seed=int(seed))
                debt = 0.0
                selected: set[int] = set()
                try:
                    for window in range(int(args.collect_windows)):
                        spent = 0.0
                        selected.clear()
                        while spent < 200.0 and len(xs) < int(args.max_pairs) and not bool(eng.term_buf[0]):
                            obs = attach_env_obs(get_obs(eng, debt), env_cfg, True, True)
                            pairs = candidate_pairs(base_planner, obs, selected, int(args.per_sensor_top), True)
                            scored = []
                            group_id = len(set(groups)) if groups else 0
                            for item in pairs[: int(args.max_pairs_per_state)]:
                                s_score, s_base, s_action, x_score, x_base, x_action = item
                                joint = encode_joint_action(int(s_action), int(x_action))
                                tail = tail_planner._tail_from(obs, int(joint), float(args.label_horizon_ms))
                                xs.append(pair_features(obs, s_score, s_base, s_action, x_score, x_base, x_action))
                                ys.append(exact_pair_label(eng, tail, debt, float(args.label_horizon_ms), float(args.potential_weight)))
                                groups.append(group_id)
                                scored.append((float(ys[-1]), int(joint), int(s_base), int(x_base)))
                            if not scored:
                                break
                            _best_label, best_action, s_base, x_base = max(scored, key=lambda x: x[0])
                            reward, dt, executed = execute_first_valid_action_joint(eng, [int(best_action)], 200.0 - spent)
                            if executed is None or float(dt) <= 0.0:
                                break
                            spent += float(dt)
                            atoms = split_joint_action(int(executed)) if is_joint_action(int(executed)) else (int(executed),)
                            is_search = False
                            for atom in atoms:
                                base, _sid = xs_decode_action(int(atom), MAXT)
                                if int(base) == 0:
                                    is_search = True
                                elif int(base) > 0:
                                    selected.add(int(base))
                            debt = 0.0 if is_search else float(debt) + float(dt)
                        if len(xs) >= int(args.max_pairs):
                            break
                finally:
                    eng.close()
                print({"pairs": len(xs), "initial": initial, "rate": rate, "seed": seed}, flush=True)
                if len(xs) >= int(args.max_pairs):
                    break
            if len(xs) >= int(args.max_pairs):
                break
        if len(xs) >= int(args.max_pairs):
            break
    if not xs:
        raise RuntimeError("no pair data collected")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), np.asarray(groups, dtype=np.int64)


def train_pair_net(x: np.ndarray, y: np.ndarray, groups: np.ndarray, steps: int, seed: int, rank_loss: bool = False) -> PairQNet:
    torch.manual_seed(int(seed))
    model = PairQNet(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    y_mean = yt.mean()
    y_std = yt.std(unbiased=False).clamp_min(1.0)
    yt_n = (yt - y_mean) / y_std
    model.register_buffer("label_mean", y_mean)
    model.register_buffer("label_std", y_std)
    n = x.shape[0]
    group_ids = np.unique(groups)
    group_to_idx = [torch.from_numpy(np.where(groups == gid)[0]) for gid in group_ids]
    for step in range(int(steps)):
        if bool(rank_loss):
            losses = []
            for _ in range(min(32, len(group_to_idx))):
                gidx = group_to_idx[int(torch.randint(0, len(group_to_idx), (1,)).item())]
                if int(gidx.numel()) < 2:
                    continue
                pred = model(xt[gidx])
                best = int(torch.argmax(yt[gidx]).item())
                losses.append(F.cross_entropy(pred[None, :], torch.tensor([best])))
            loss = torch.stack(losses).mean() if losses else torch.zeros((), requires_grad=True)
        else:
            idx = torch.randint(0, n, (min(256, n),))
            pred = model(xt[idx])
            loss = F.smooth_l1_loss(pred, yt_n[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in {0, int(steps) - 1}:
            print({"pair_step": step, "loss": float(loss.detach()), "pairs": int(n), "groups": int(len(group_to_idx)), "rank_loss": bool(rank_loss), "label_mean": float(y_mean), "label_std": float(y_std)}, flush=True)
    return model.eval()


class PairQCoupledPlanner(WorkConservingAsyncCoupledPlanner):
    def __init__(self, base_planner: PhysicalHeadPlanner, pair_model: PairQNet, per_sensor_top: int = 4, pair_weight: float = 1.0):
        super().__init__(base_planner, per_sensor_top=int(per_sensor_top), include_search_candidate=True)
        self.pair_model = pair_model.eval()
        self.pair_weight = float(pair_weight)

    def _candidate_pairs(self, obs: dict, selected: set[int], elapsed: float, search_count: int, track_count: int, last: int):
        obs = attach_env_obs(dict(obs), self.env_cfg, True, True)
        s_busy = float(obs.get("s_band_busy_ms", 0.0))
        x_busy = float(obs.get("x_band_busy_ms", 0.0))
        x_enabled = bool(int(obs.get("enable_x_band", 0)))
        s_free = s_busy <= 0.0
        x_free = x_enabled and x_busy <= 0.0
        s_dummy = xs_s_search_action(MAXT)
        x_dummy = xs_x_search_action(MAXT)
        out = []
        if s_free and x_free:
            s_ranked = self._ranked_for_sensor(obs, 0, selected, elapsed, search_count, track_count, last)
            x_ranked = self._ranked_for_sensor(obs, 1, selected, elapsed, search_count, track_count, last)
            items = []
            for s_score, s_base, s_action in s_ranked:
                for x_score, x_base, x_action in x_ranked:
                    if int(s_base) > 0 and int(s_base) == int(x_base):
                        continue
                    items.append((s_score, s_base, s_action, x_score, x_base, x_action))
            if items:
                feats = np.stack([pair_features(obs, *item) for item in items]).astype(np.float32)
                with torch.inference_mode():
                    pred = self.pair_model(torch.from_numpy(feats)).cpu().numpy()
                for item, pair_pred in zip(items, pred):
                    s_score, _s_base, s_action, x_score, _x_base, x_action = item
                    score = float(s_score) + float(x_score) + float(self.pair_weight) * float(pair_pred)
                    out.append((score, encode_joint_action(int(s_action), int(x_action))))
        elif s_free:
            for s_score, _s_base, s_action in self._ranked_for_sensor(obs, 0, selected, elapsed, search_count, track_count, last):
                out.append((float(s_score), encode_joint_action(int(s_action), int(x_dummy))))
        elif x_free:
            for x_score, _x_base, x_action in self._ranked_for_sensor(obs, 1, selected, elapsed, search_count, track_count, last):
                out.append((float(x_score), encode_joint_action(int(s_dummy), int(x_action))))
        elif s_busy > 0.0 or (x_enabled and x_busy > 0.0):
            out.append((-1e6, encode_joint_action(int(s_dummy), int(x_dummy))))
        deduped = {}
        for score, action in out:
            deduped[int(action)] = max(float(score), deduped.get(int(action), -np.inf))
        return sorted([(score, action) for action, score in deduped.items()], reverse=True, key=lambda x: x[0])


class PairQDirectPlanner:
    def __init__(self, base_planner: PhysicalHeadPlanner, pair_model: PairQNet, per_sensor_top: int = 4, pair_weight: float = 1.0):
        self.coupled = PairQCoupledPlanner(base_planner, pair_model, per_sensor_top=per_sensor_top, pair_weight=pair_weight)

    def plan(self, obs, budget_ms=200):
        return self.coupled.plan(obs, budget_ms=budget_ms)


class DirectAdapter:
    def __init__(self, planner):
        self.planner = planner

    def choose(self, eng, debt_ms: float, obs):
        t0 = time.perf_counter()
        plan = self.planner.plan(obs, budget_ms=200)
        return plan, {"planning_ms": float((time.perf_counter() - t0) * 1000.0)}


def eval_methods(args, base_planner: PhysicalHeadPlanner, pair_model: PairQNet, exact_args):
    rows = []
    windows = []
    actions = []
    for initial in parse_ints(args.eval_initials):
        for rate in parse_floats(args.eval_rates):
            env_cfg = env_cfg_for(float(rate), exact_args)
            env_cfg["enable_x_band"] = 1
            base_planner.env_cfg = dict(env_cfg)
            pair_planner = PairQDirectPlanner(base_planner, pair_model, per_sensor_top=int(args.per_sensor_top), pair_weight=float(args.pair_weight))
            planners = {
                "EDF": EDFPlanner(MAXT),
                "EST": ESTPlanner(MAXT),
                "PairQ_direct": DirectAdapter(pair_planner),
            }
            for name, planner in planners.items():
                if name in {"EDF", "EST"}:
                    w, a = run_heuristic(planner, name, int(initial), int(args.eval_seed), int(args.eval_windows), engine_env_cfg(env_cfg))
                else:
                    w, a = run_exact_rescore_grid_joint(planner, name, int(initial), int(args.eval_seed), int(args.eval_windows), env_cfg)
                s = summarize_window_df(w, "fixed")
                denom = max(1.0, float(w["spent_ms"].sum()) if "spent_ms" in w else float(len(w) * 200))
                has_busy = not a.empty and {"s_busy_ms", "x_busy_ms"}.issubset(a.columns)
                s_util = float(a["s_busy_ms"].sum() / denom) if has_busy else np.nan
                x_util = float(a["x_busy_ms"].sum() / denom) if has_busy else np.nan
                row = {
                    "method": name,
                    "initial": int(initial),
                    "rate": float(rate),
                    "reward": float(s.get("reward_per_200ms_eq", np.nan)),
                    "search": float(s.get("search_fraction", np.nan)),
                    "tracked": float(s.get("mean_tracked_targets", np.nan)),
                    "drop": float(s.get("mean_drop_pct_active", np.nan)),
                    "delay": float(s.get("mean_delay_active", np.nan)),
                    "latency_ms": float(s.get("planning_ms_per_decision", np.nan)),
                    "s_util": s_util,
                    "x_util": x_util,
                    "mean_sensor_util": float(np.nanmean([s_util, x_util])) if has_busy else np.nan,
                }
                print(row, flush=True)
                rows.append(row)
                windows.append(w.assign(method=name, initial=int(initial), rate=float(rate)))
                if not a.empty:
                    actions.append(a.assign(method=name, initial=int(initial), rate=float(rate)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).groupby("method").agg(
        reward=("reward", "mean"),
        search=("search", "mean"),
        tracked=("tracked", "mean"),
        drop=("drop", "mean"),
        delay=("delay", "mean"),
        latency_ms=("latency_ms", "mean"),
        mean_sensor_util=("mean_sensor_util", "mean"),
        n=("reward", "size"),
    ).reset_index().sort_values("reward", ascending=False).to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    if windows:
        pd.concat(windows, ignore_index=True).to_csv(out.with_name(out.stem + "_windows.csv"), index=False)
    if actions:
        pd.concat(actions, ignore_index=True).to_csv(out.with_name(out.stem + "_actions.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-state", default="CreateValid1/results/factorized_qstrong_edfonly_state.pt")
    ap.add_argument("--base-variant", default="two_row_action_attention_factored_loss")
    ap.add_argument("--out", default="CreateValid1/results/pairwise_q_direct_hard_60_4_20w.csv")
    ap.add_argument("--initials", default="60")
    ap.add_argument("--rates", default="4")
    ap.add_argument("--train-seeds", default="916")
    ap.add_argument("--collect-windows", type=int, default=20)
    ap.add_argument("--max-pairs", type=int, default=2048)
    ap.add_argument("--max-pairs-per-state", type=int, default=16)
    ap.add_argument("--label-horizon-ms", type=float, default=800.0)
    ap.add_argument("--potential-weight", type=float, default=1.0)
    ap.add_argument("--pair-steps", type=int, default=600)
    ap.add_argument("--pair-model-in", default="")
    ap.add_argument("--rank-loss", action="store_true")
    ap.add_argument("--per-sensor-top", type=int, default=4)
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--policy-weight", type=float, default=1.0)
    ap.add_argument("--q-weight", type=float, default=1.5)
    ap.add_argument("--search-bias", type=float, default=-12.0)
    ap.add_argument("--eval-initials", default="60")
    ap.add_argument("--eval-rates", default="4")
    ap.add_argument("--eval-seed", type=int, default=916)
    ap.add_argument("--eval-windows", type=int, default=20)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    args = ap.parse_args()
    args.windows = int(args.eval_windows)

    torch.set_num_threads(1)
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    env_cfg = env_cfg_for(float(parse_floats(args.rates)[0]), exact_args)
    env_cfg["enable_x_band"] = 1
    model = make_physical_model(str(args.base_variant), args)
    state = torch.load(args.base_state, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    base_planner = PhysicalHeadPlanner(
        model.eval(),
        str(args.base_variant),
        env_cfg,
        policy_weight=float(args.policy_weight),
        q_weight=float(args.q_weight),
        search_score_bias=float(args.search_bias),
    )
    if str(args.pair_model_in).strip():
        saved = torch.load(args.pair_model_in, map_location="cpu", weights_only=False)
        pair_model = PairQNet(int(saved["in_dim"]))
        pair_model.load_state_dict(saved["state_dict"], strict=False)
        pair_model.eval()
    else:
        x, y, groups = collect_pair_data(args, base_planner, exact_args)
        pair_model = train_pair_net(x, y, groups, int(args.pair_steps), int(args.eval_seed), rank_loss=bool(args.rank_loss))
        torch.save({"state_dict": pair_model.state_dict(), "in_dim": int(x.shape[1])}, Path(args.out).with_name(Path(args.out).stem + "_pair_model.pt"))
    eval_methods(args, base_planner, pair_model, exact_args)


if __name__ == "__main__":
    main()
