from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from alphazero_benchmark import load_state_into
from alphazero_orthodox import base_exact_args
from exact_env_mutual import (
    EDFPlanner,
    ESTPlanner,
    ExactEnvMCTS,
    MAXT,
    SnapshotSimulator,
    attach_env_obs,
    choose_root_action,
    env_cfg_for,
    load_model,
    run_fixed,
    run_snapshot_exact_episode,
    xs_decode_action,
    xs_s_search_action,
    xs_x_search_action,
)
from final_radar_campaign import get_obs, summarize_window_df
from joint_action_experiment import encode_joint_action, execute_first_valid_action_joint, is_joint_action, split_joint_action
from repaired_campaign_tools import build_env
from strict_window_report import sample_state_metrics


ROOT = Path(r"C:\Users\yousi\Downloads\Model1 1")
RES = ROOT / "CreateValid1" / "results" / "pq1_alphazero_r60_rates136"
CKPT = Path(r"C:\Users\yousi\Downloads\radar_outputs\exact_train_qw02_seeded_more\exact_mutual_latest.pt")
STATE = RES / "r16h4_behavior_distill_state.pt"


def exact_args(rollouts: int, horizon_windows: int = 2, windows: int = 20, select_mode: str = "load_gated_prior"):
    return base_exact_args(
        SimpleNamespace(
            ckpt=str(CKPT),
            device="cpu",
            head_arch="branch_context",
            windows=int(windows),
            max_targets_per_episode=64,
            rollouts=int(rollouts),
            c_puct=1.25,
            expand_top_k=48,
            horizon_windows=int(horizon_windows),
            prior_uniform_mix=0.03,
            root_dirichlet_alpha=0.3,
            root_dirichlet_frac=0.0,
            leaf_value_mix=0.5,
            rollout_policy="branch",
            branch_rollout_threshold=0.65,
            seed_rollout_policies="",
            skip_default_rollout_seed=False,
            prior_mode="physical_flat",
            sensor_action_mode="explicit",
            disable_x_search=False,
            canonical_search_only=False,
            search_alg="puct",
            plan_mode="atomic",
            window_extract="tree_fill",
            gumbel_scale=0.0,
            max_num_considered_actions=16,
            mctx_value_scale=0.1,
            mctx_maxvisit_init=50.0,
            select_mode=str(select_mode),
            load_gated_prior_threshold=80,
            visit_unvisited_first=True,
            head_mode="pq",
            q_utility_weight=0.0,
            q_utility_normalize=False,
            puct_q_transform="raw",
            prior_q_beta=0.0,
            prior_search_bias=0.0,
            adaptive_search_bias=0.0,
            adaptive_search_target_load=0.75,
            q_scale=100.0,
            self_play_sample_tau=0.0,
            gamma=0.99,
            env_mode="mcts_sched_v1",
            use_arrival_feature=True,
            use_grid_feature=True,
            single_sensor=False,
            zero_action_rewards=False,
            track_loss_penalty=8.0,
            target_service_weight=10.0,
            target_service_horizon_ms=3000.0,
            sector_staleness_weight=0.01,
            search_frame_overdue_weight=0.20,
            search_frame_drop_penalty=8.0,
            enable_x_band=True,
        )
    )


def load_distilled_model(args):
    model = load_model(args).to(torch.device("cpu"))
    load_state_into(model, str(STATE), torch.device("cpu"))
    model.eval()
    return model


def child_score(child, mode: str) -> float:
    if mode == "prior":
        return float(child.prior)
    if mode == "q":
        return float(child.edge_reward + child.mean_value)
    return float(child.visits) + 1e-3 * float(child.edge_reward + child.mean_value)


def select_mode_for_obs(args, obs: dict) -> str:
    if str(args.select_mode) != "load_gated_prior":
        return str(args.select_mode)
    active = np.asarray(obs.get("active_mask", []), dtype=bool)
    active_count = int(np.sum(active)) if active.size else 0
    return "prior" if active_count <= int(getattr(args, "load_gated_prior_threshold", 80)) else "visits"


def choose_joint_from_root(root, args, obs: dict) -> int:
    if not root.children:
        return encode_joint_action(xs_s_search_action(MAXT), xs_x_search_action(MAXT))
    mode = select_mode_for_obs(args, obs)
    ranked = {0: [], 1: []}
    for child in root.children:
        action = int(child.action)
        base, sensor = xs_decode_action(action, MAXT)
        if sensor is None or int(sensor) not in ranked or int(base) < 0:
            continue
        ranked[int(sensor)].append((child_score(child, mode), action))
    for sensor in ranked:
        ranked[sensor].sort(reverse=True, key=lambda x: x[0])
    best = None
    best_score = -float("inf")
    for s_score, s_action in ranked[0][:8]:
        for x_score, x_action in ranked[1][:8]:
            s_base, _ = xs_decode_action(int(s_action), MAXT)
            x_base, _ = xs_decode_action(int(x_action), MAXT)
            if int(s_base) > 0 and int(s_base) == int(x_base):
                continue
            score = float(s_score) + float(x_score)
            if score > best_score:
                best = encode_joint_action(int(s_action), int(x_action))
                best_score = score
    if best is not None:
        return int(best)
    return int(choose_root_action(root, mode))


def make_mcts(model, sim, args):
    return ExactEnvMCTS(
        model,
        sim,
        [],
        q_scale=float(getattr(args, "q_scale", 100.0)),
        rollouts=int(args.rollouts),
        c_puct=float(args.c_puct),
        expand_top_k=int(args.expand_top_k),
        horizon_windows=int(args.horizon_windows),
        rollout_policy=str(args.rollout_policy),
        prior_mode=str(args.prior_mode),
        epsilon=float(args.epsilon),
        policy_target=str(args.policy_target),
        policy_tau=float(args.policy_tau),
        branch_rollout_threshold=float(getattr(args, "branch_rollout_threshold", 0.65)),
        search_alg=str(args.search_alg),
        max_num_considered_actions=int(args.max_num_considered_actions),
        gumbel_scale=float(args.gumbel_scale),
        mctx_value_scale=float(args.mctx_value_scale),
        mctx_maxvisit_init=float(args.mctx_maxvisit_init),
        eager_edge_depth=int(args.eager_edge_depth),
        prior_uniform_mix=float(args.prior_uniform_mix),
        root_dirichlet_alpha=float(getattr(args, "root_dirichlet_alpha", 0.0)),
        root_dirichlet_frac=float(getattr(args, "root_dirichlet_frac", 0.0)),
        rollout_est_prob=float(args.rollout_est_prob),
        mask_selected=not bool(args.allow_retrack_in_window),
        stateless_tree_context=bool(args.stateless_tree_context),
        head_mode=str(args.head_mode),
        q_utility_weight=float(args.q_utility_weight),
        q_utility_normalize=bool(args.q_utility_normalize),
        leaf_value_mix=float(args.leaf_value_mix),
        seed_rollout_policies=args.seed_rollout_policies.split(",") if args.seed_rollout_policies else (),
        fast_zero_rollout=bool(args.fast_zero_rollout),
        skip_default_rollout_seed=bool(args.skip_default_rollout_seed),
        complete_root_q_with_value=bool(args.complete_root_q_with_value),
        visit_unvisited_first=bool(args.visit_unvisited_first),
        duration_normalize_q=bool(args.duration_normalize_q),
        prior_q_beta=float(args.prior_q_beta),
        prior_search_bias=float(args.prior_search_bias),
        adaptive_search_bias=float(getattr(args, "adaptive_search_bias", 0.0)),
        adaptive_search_target_load=float(getattr(args, "adaptive_search_target_load", 0.75)),
        sensor_action_mode=str(args.sensor_action_mode),
        disable_x_search=bool(args.disable_x_search),
        canonical_search_only=bool(args.canonical_search_only),
    )


def apply_stress_cfg(env_cfg: dict, cli=None) -> dict:
    out = dict(env_cfg)
    if cli is not None and hasattr(cli, "revisit_scale"):
        out["revisit_time_scale"] = float(cli.revisit_scale)
    if cli is not None and hasattr(cli, "dwell_scale"):
        out["dwell_time_scale"] = float(cli.dwell_scale)
    return out


def run_joint_mcts_episode(model, args, initial: int, rate: float, seed: int, stress_cli=None):
    env_cfg = env_cfg_for(float(rate), args)
    env_cfg = apply_stress_cfg(env_cfg, stress_cli)
    eng = build_env(None, int(initial), MAXT, int(seed), 200, env_cfg)
    eng.reset(seed=int(seed))
    rows = []
    actions = []
    debt = 0.0
    cumulative = 0.0
    try:
        for window in range(int(args.windows)):
            if bool(eng.term_buf[0]):
                break
            window_reward = 0.0
            window_ms = 0.0
            window_actions = []
            plan_ms_total = 0.0
            while window_ms < 200.0 and not bool(eng.term_buf[0]):
                obs = attach_env_obs(get_obs(eng, debt), env_cfg, bool(args.use_arrival_feature), bool(args.use_grid_feature))
                sim = SnapshotSimulator(eng, debt, env_cfg, bool(args.use_arrival_feature), bool(args.use_grid_feature), int(seed))
                t0 = time.perf_counter()
                mcts = make_mcts(model, sim, args)
                root = mcts.run()
                action = choose_joint_from_root(root, args, sim._cache[()].obs)
                fallback = [int(c.action) for c in root.children if int(c.action) != int(action)]
                reward, dt, executed = execute_first_valid_action_joint(eng, [int(action), *fallback], 200.0 - window_ms)
                plan_ms_total += (time.perf_counter() - t0) * 1000.0
                if executed is None or dt <= 0.0:
                    debt += max(0.0, 200.0 - window_ms)
                    window_ms = 200.0
                    break
                if is_joint_action(executed):
                    atoms = split_joint_action(executed)
                else:
                    atoms = (int(executed),)
                if any(xs_decode_action(int(a), MAXT)[0] == 0 for a in atoms):
                    debt = 0.0
                else:
                    debt += float(dt)
                window_reward += float(reward)
                window_ms += float(dt)
                window_actions.append(int(executed))
                actions.append(
                    {
                        "window": int(window + 1),
                        "action": int(executed),
                        "s_action": int(atoms[0]) if len(atoms) > 1 else -1,
                        "x_action": int(atoms[1]) if len(atoms) > 1 else -1,
                        "reward": float(reward),
                        "dt_ms": float(dt),
                    }
                )
            cumulative += float(window_reward)
            obs_after = get_obs(eng, debt)
            rows.append(
                {
                    "window": int(window + 1),
                    "window_reward": float(window_reward),
                    "cumulative_reward": float(cumulative),
                    "window_ms_used": float(window_ms),
                    "actions": int(len(window_actions)),
                    "search_fraction": float(np.mean([
                        any(xs_decode_action(int(a), MAXT)[0] == 0 for a in (split_joint_action(x) if is_joint_action(x) else (int(x),)))
                        for x in window_actions
                    ])) if window_actions else 0.0,
                    "planning_ms_per_decision": float(plan_ms_total / max(1, len(window_actions))),
                    **sample_state_metrics(eng, debt),
                }
            )
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(actions)


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def evaluate(cli):
    torch.set_num_threads(1)
    rows = []
    windows = []
    actions = []
    rollouts_list = parse_ints(cli.rollouts_list)
    models = {}
    for r in rollouts_list:
        args = exact_args(r, horizon_windows=int(cli.horizon_windows), windows=int(cli.windows), select_mode=str(cli.select_mode))
        models[r] = (args, load_distilled_model(args))
    heur_args = exact_args(1, horizon_windows=int(cli.horizon_windows), windows=int(cli.windows), select_mode=str(cli.select_mode))
    for seed in parse_ints(cli.seeds):
        for initial in parse_ints(cli.initials):
            for rate in parse_floats(cli.rates):
                env_cfg = apply_stress_cfg(env_cfg_for(float(rate), heur_args), cli)
                for name, planner in [("EDF", EDFPlanner(MAXT)), ("EST", ESTPlanner(MAXT))]:
                    w, a = run_fixed(planner, name, int(initial), MAXT, int(seed), int(cli.windows), 200, env_cfg)
                    s = summarize_window_df(w, "fixed")
                    rows.append(row_from_summary(name, initial, rate, seed, s, len(w)))
                    windows.append(w.assign(method=name, initial=int(initial), rate=float(rate), seed=int(seed)))
                    actions.append(a.assign(method=name, initial=int(initial), rate=float(rate), seed=int(seed)) if not a.empty else a)
                    print(rows[-1], flush=True)
                for r in rollouts_list:
                    args, model = models[r]
                    if bool(cli.include_atomic):
                        t0 = time.perf_counter()
                        # Atomic checkpoint reproduction uses the original env
                        # function, so leave stress sweeps to the joint path.
                        w, _targets = run_snapshot_exact_episode(model, args, int(initial), float(rate), int(seed), train=False)
                        s = summarize_any(w)
                        s["wall_seconds"] = time.perf_counter() - t0
                        rows.append(row_from_summary(f"Atomic_r{r}", initial, rate, seed, s, len(w)))
                        windows.append(w.assign(method=f"Atomic_r{r}", initial=int(initial), rate=float(rate), seed=int(seed)))
                        print(rows[-1], flush=True)
                    t0 = time.perf_counter()
                    w, a = run_joint_mcts_episode(model, args, int(initial), float(rate), int(seed), cli)
                    s = summarize_any(w)
                    s["wall_seconds"] = time.perf_counter() - t0
                    rows.append(row_from_summary(f"Joint_r{r}", initial, rate, seed, s, len(w)))
                    windows.append(w.assign(method=f"Joint_r{r}", initial=int(initial), rate=float(rate), seed=int(seed)))
                    actions.append(a.assign(method=f"Joint_r{r}", initial=int(initial), rate=float(rate), seed=int(seed)) if not a.empty else a)
                    print(rows[-1], flush=True)
    out = Path(cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out, index=False)
    if windows:
        pd.concat(windows, ignore_index=True).to_csv(out.with_name(out.stem + "_windows.csv"), index=False)
    if actions:
        pd.concat(actions, ignore_index=True).to_csv(out.with_name(out.stem + "_actions.csv"), index=False)
    summary = raw.groupby("method", as_index=False).agg(
        reward=("reward", "mean"),
        total_reward=("total_reward", "mean"),
        search=("search", "mean"),
        active=("active", "mean"),
        tracked=("tracked", "mean"),
        drop=("drop", "mean"),
        delay=("delay", "mean"),
        latency_ms=("latency_ms", "mean"),
        wall_seconds=("wall_seconds", "sum"),
        n=("reward", "size"),
    ).sort_values(["drop", "tracked"], ascending=[True, False])
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    print(summary.to_string(index=False), flush=True)


def row_from_summary(method: str, initial: int, rate: float, seed: int, s: dict, windows_completed: int):
    return {
        "method": str(method),
        "initial": int(initial),
        "rate": float(rate),
        "seed": int(seed),
        "reward": float(s.get("reward_per_200ms_eq", s.get("window_reward", np.nan))),
        "total_reward": float(s.get("total_reward", np.nan)),
        "search": float(s.get("search_fraction", np.nan)),
        "windows_completed": int(windows_completed),
        "latency_ms": float(s.get("planning_ms_per_decision", np.nan)),
        "active": float(s.get("mean_active_targets", np.nan)),
        "tracked": float(s.get("mean_tracked_targets", np.nan)),
        "drop": float(s.get("mean_drop_pct_active", np.nan)),
        "delay": float(s.get("mean_delay_active", np.nan)),
        "wall_seconds": float(s.get("wall_seconds", np.nan)),
    }


def summarize_any(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    latency_col = "planning_ms_per_decision" if "planning_ms_per_decision" in df.columns else None
    return {
        "total_reward": float(df["window_reward"].sum()) if "window_reward" in df.columns else np.nan,
        "reward_per_200ms_eq": float(df["window_reward"].mean()) if "window_reward" in df.columns else np.nan,
        "search_fraction": float(df["search_fraction"].mean()) if "search_fraction" in df.columns else np.nan,
        "mean_active_targets": float(df["active_targets"].mean()) if "active_targets" in df.columns else np.nan,
        "mean_tracked_targets": float(df["tracked_targets"].mean()) if "tracked_targets" in df.columns else np.nan,
        "mean_drop_pct_active": float(df["drop_pct_active"].mean()) if "drop_pct_active" in df.columns else np.nan,
        "mean_delay_active": float(df["mean_delay_active"].mean()) if "mean_delay_active" in df.columns else np.nan,
        "planning_ms_per_decision": float(df[latency_col].mean()) if latency_col else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "joint_mcts_checkpoint_eval.csv"))
    ap.add_argument("--initials", default="60,100")
    ap.add_argument("--rates", default="0,10,20")
    ap.add_argument("--seeds", default="931,936,938")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--rollouts-list", default="0,1,4,16")
    ap.add_argument("--horizon-windows", type=int, default=2)
    ap.add_argument("--select-mode", default="load_gated_prior")
    ap.add_argument("--revisit-scale", type=float, default=1.0)
    ap.add_argument("--dwell-scale", type=float, default=1.0)
    ap.add_argument("--include-atomic", action="store_true")
    evaluate(ap.parse_args())


if __name__ == "__main__":
    main()
