from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from collect_r16_behavior_targets import load_state_into, make_args
from exact_env_mutual import (
    EDFPlanner,
    ExactEnvMCTS,
    MAXT,
    SnapshotSimulator,
    _DummyPlanner,
    choose_root_action,
    choose_root_action_load_gated,
    env_cfg_for,
    load_model,
    xs_decode_action,
)
from final_radar_campaign import build_env, get_obs


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def action_kind(action: int) -> str:
    base, sensor = xs_decode_action(int(action), MAXT)
    if int(base) == 0:
        return "search_s" if int(sensor or 0) == 0 else "search_x"
    if int(base) > 0:
        return "track_s" if int(sensor or 0) == 0 else "track_x"
    return "idle"


def target_features(obs: dict, action: int, prefix: str) -> dict:
    base, sensor = xs_decode_action(int(action), MAXT)
    out = {
        f"{prefix}_action": int(action),
        f"{prefix}_base": int(base),
        f"{prefix}_sensor": -1 if sensor is None else int(sensor),
        f"{prefix}_kind": action_kind(int(action)),
    }
    if int(base) <= 0:
        out.update(
            {
                f"{prefix}_target_status": "search" if int(base) == 0 else "idle",
                f"{prefix}_deadline": np.nan,
                f"{prefix}_desired": np.nan,
                f"{prefix}_lateness": np.nan,
                f"{prefix}_dwell": np.nan,
                f"{prefix}_priority": np.nan,
                f"{prefix}_range": np.nan,
                f"{prefix}_deadline_rank": np.nan,
                f"{prefix}_lateness_rank": np.nan,
                f"{prefix}_range_bucket": "none",
                f"{prefix}_dwell_bucket": "none",
            }
        )
        return out

    idx = int(base) - 1
    active = np.asarray(obs.get("active_mask", []), dtype=bool)
    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
    desired = np.asarray(obs.get("t_desired", np.zeros_like(deadline)), dtype=np.float32)
    dwell = np.asarray(obs.get("t_dwell", np.zeros_like(deadline)), dtype=np.float32)
    priority = np.asarray(obs.get("priority", np.zeros_like(deadline)), dtype=np.float32)
    target_range = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
    n = min(len(active), len(deadline), len(desired), len(dwell), len(priority), len(target_range), MAXT)
    if not (0 <= idx < n):
        out[f"{prefix}_target_status"] = "invalid"
        return out

    active_idx = np.where(active[:n])[0]
    tracked_idx = np.where(active[:n] & (deadline[:n] >= 0.0))[0]
    deadline_rank = np.nan
    if idx in set(tracked_idx.tolist()):
        order = tracked_idx[np.argsort(deadline[tracked_idx])]
        deadline_rank = int(np.where(order == idx)[0][0]) + 1
    lateness = np.maximum(0.0, -desired[:n])
    lateness_rank = np.nan
    if idx in set(active_idx.tolist()):
        order = active_idx[np.argsort(-lateness[active_idx])]
        lateness_rank = int(np.where(order == idx)[0][0]) + 1

    def bucket(values: np.ndarray, value: float) -> str:
        vals = values[np.isfinite(values)]
        if vals.size < 3:
            return "unknown"
        q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
        if value <= q1:
            return "low"
        if value <= q2:
            return "mid"
        return "high"

    dline = float(deadline[idx])
    des = float(desired[idx])
    dw = float(dwell[idx])
    status = "dropped" if dline < 0.0 else ("urgent" if dline <= max(1.0, dw) + 50.0 else ("late_desired" if des < 0.0 else "healthy"))
    out.update(
        {
            f"{prefix}_target_status": status,
            f"{prefix}_deadline": dline,
            f"{prefix}_desired": des,
            f"{prefix}_lateness": max(0.0, -des),
            f"{prefix}_dwell": dw,
            f"{prefix}_priority": float(priority[idx]),
            f"{prefix}_range": float(target_range[idx]),
            f"{prefix}_deadline_rank": deadline_rank,
            f"{prefix}_lateness_rank": lateness_rank,
            f"{prefix}_range_bucket": bucket(target_range[active_idx], float(target_range[idx])) if active_idx.size else "unknown",
            f"{prefix}_dwell_bucket": bucket(dwell[active_idx], dw) if active_idx.size else "unknown",
        }
    )
    return out


def compare_category(model_action: int, edf_action: int) -> str:
    m_base, _ = xs_decode_action(int(model_action), MAXT)
    e_base, _ = xs_decode_action(int(edf_action), MAXT)
    if int(m_base) == int(e_base):
        return "same_search" if int(m_base) == 0 else "same_track"
    if int(m_base) == 0 and int(e_base) > 0:
        return "model_search_edf_track"
    if int(m_base) > 0 and int(e_base) == 0:
        return "model_track_edf_search"
    if int(m_base) > 0 and int(e_base) > 0:
        return "different_track"
    return "other"


def run_case(model, exact_args, initial: int, rate: float, seed: int) -> list[dict]:
    env_cfg = env_cfg_for(float(rate), exact_args)
    eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, env_cfg)
    eng.reset(seed=int(seed))
    edf = EDFPlanner(MAXT)
    debt = 0.0
    rows: list[dict] = []
    try:
        for window in range(int(exact_args.windows)):
            window_ms = 0.0
            search_count = 0
            track_count = 0
            last_action = -1
            while window_ms < 200.0 and not bool(eng.term_buf[0]):
                sim = SnapshotSimulator(
                    eng,
                    debt,
                    env_cfg,
                    bool(exact_args.use_arrival_feature),
                    bool(exact_args.use_grid_feature),
                    int(seed),
                )
                obs = sim._cache[()].obs
                mcts = ExactEnvMCTS(
                    model,
                    sim,
                    [],
                    q_scale=float(exact_args.q_scale),
                    rollouts=int(exact_args.rollouts),
                    c_puct=float(exact_args.c_puct),
                    expand_top_k=int(exact_args.expand_top_k),
                    horizon_windows=int(exact_args.horizon_windows),
                    rollout_policy=str(exact_args.rollout_policy),
                    prior_mode=str(exact_args.prior_mode),
                    epsilon=0.0,
                    policy_target=str(exact_args.policy_target),
                    policy_tau=float(exact_args.policy_tau),
                    branch_rollout_threshold=float(exact_args.branch_rollout_threshold),
                    search_alg=str(exact_args.search_alg),
                    max_num_considered_actions=int(exact_args.max_num_considered_actions),
                    prior_uniform_mix=float(exact_args.prior_uniform_mix),
                    root_dirichlet_alpha=float(exact_args.root_dirichlet_alpha),
                    root_dirichlet_frac=float(exact_args.root_dirichlet_frac),
                    rollout_est_prob=float(exact_args.rollout_est_prob),
                    mask_selected=not bool(exact_args.allow_retrack_in_window),
                    head_mode=str(exact_args.head_mode),
                    q_utility_weight=float(exact_args.q_utility_weight),
                    q_utility_normalize=bool(exact_args.q_utility_normalize),
                    leaf_value_mix=float(exact_args.leaf_value_mix),
                    visit_unvisited_first=bool(exact_args.visit_unvisited_first),
                    prior_q_beta=0.0,
                    prior_search_bias=0.0,
                    adaptive_search_bias=0.0,
                    adaptive_search_target_load=0.75,
                    sensor_action_mode=str(exact_args.sensor_action_mode),
                )
                root = mcts.run()
                if str(exact_args.select_mode) == "load_gated_prior":
                    model_action = choose_root_action_load_gated(root, obs, int(getattr(exact_args, "load_gated_prior_threshold", 80)))
                else:
                    model_action = choose_root_action(root, str(exact_args.select_mode))
                fallback_actions = [int(a) for a in mcts.valid_actions(obs) if int(a) != int(model_action)]
                edf_plan = list(edf.plan(obs, budget_ms=int(max(1.0, 200.0 - window_ms))))
                _er, _edt, _edebt, edf_exec = sim.commit_first_valid(edf_plan, 200.0 - window_ms)
                if edf_exec is None:
                    edf_exec = -1
                reward, dt, debt, model_exec = sim.commit_first_valid([int(model_action), *fallback_actions, MAXT + 1, MAXT + 2], 200.0 - window_ms)
                if model_exec is None or dt <= 0.0:
                    break

                row = {
                    "initial": int(initial),
                    "rate": float(rate),
                    "seed": int(seed),
                    "window": int(window + 1),
                    "elapsed_in_window_ms": float(window_ms),
                    "reward": float(reward),
                    "dt_ms": float(dt),
                    "active_targets": int(np.asarray(obs.get("active_mask", []), dtype=bool).sum()),
                    "search_debt_ms": float(debt),
                    "category": compare_category(int(model_exec), int(edf_exec)),
                }
                row.update(target_features(obs, int(model_exec), "model"))
                row.update(target_features(obs, int(edf_exec), "edf"))
                row["deadline_delta_model_minus_edf"] = row.get("model_deadline", np.nan) - row.get("edf_deadline", np.nan)
                row["lateness_delta_model_minus_edf"] = row.get("model_lateness", np.nan) - row.get("edf_lateness", np.nan)
                row["priority_delta_model_minus_edf"] = row.get("model_priority", np.nan) - row.get("edf_priority", np.nan)
                row["range_delta_model_minus_edf"] = row.get("model_range", np.nan) - row.get("edf_range", np.nan)
                rows.append(row)

                base, _ = xs_decode_action(int(model_exec), MAXT)
                window_ms += float(dt)
                if int(base) == 0:
                    search_count += 1
                elif int(base) > 0:
                    track_count += 1
                last_action = int(base)
    finally:
        eng.close()
    return rows


def plot_outputs(events: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = events["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    counts.to_csv(out_dir / "best_vs_edf_category_counts.csv", index=False)

    plt.figure(figsize=(8, 4.5))
    plt.bar(counts["category"], counts["count"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("decisions")
    plt.title("Best model vs EDF decision categories")
    plt.tight_layout()
    plt.savefig(out_dir / "best_vs_edf_category_counts.png", dpi=160)
    plt.close()

    diff = events[events["category"] == "different_track"].copy()
    if not diff.empty:
        metrics = ["deadline_delta_model_minus_edf", "lateness_delta_model_minus_edf", "priority_delta_model_minus_edf", "range_delta_model_minus_edf"]
        means = diff[metrics].mean(numeric_only=True).reset_index()
        means.columns = ["metric", "mean"]
        means.to_csv(out_dir / "different_track_feature_deltas.csv", index=False)
        plt.figure(figsize=(8, 4.5))
        plt.bar(means["metric"], means["mean"])
        plt.xticks(rotation=25, ha="right")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title("When both track different targets: model minus EDF")
        plt.tight_layout()
        plt.savefig(out_dir / "different_track_feature_deltas.png", dpi=160)
        plt.close()

        pivot = pd.crosstab(diff["model_target_status"], diff["edf_target_status"])
        pivot.to_csv(out_dir / "different_track_status_crosstab.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--max-targets-per-episode", type=int, default=64)
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--horizon-windows", type=int, default=2)
    ap.add_argument("--expand-top-k", type=int, default=48)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--rollout-policy", default="branch")
    ap.add_argument("--branch-rollout-threshold", type=float, default=0.65)
    ap.add_argument("--prior-uniform-mix", type=float, default=0.03)
    ap.add_argument("--leaf-value-mix", type=float, default=0.5)
    ap.add_argument("--head-mode", default="pq")
    ap.add_argument("--prior-mode", default="physical_flat")
    ap.add_argument("--sensor-action-mode", default="implicit")
    ap.add_argument("--select-mode", default="load_gated_prior")
    ap.add_argument("--load-gated-prior-threshold", type=int, default=80)
    ap.add_argument("--max-num-considered-actions", type=int, default=16)
    ap.add_argument("--q-utility-weight", type=float, default=0.0)
    ap.add_argument("--q-utility-normalize", action="store_true")
    ap.add_argument("--prior-q-beta", type=float, default=0.0)
    ap.add_argument("--q-scale", type=float, default=100.0)
    ap.add_argument("--self-play-sample-tau", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--initials", default="60,100")
    ap.add_argument("--rates", default="0,10,20")
    ap.add_argument("--seeds", default="931,936,938")
    ap.add_argument("--env-mode", default="mcts_sched_v1")
    ap.add_argument("--track-loss-penalty", type=float, default=8.0)
    ap.add_argument("--target-service-weight", type=float, default=10.0)
    ap.add_argument("--target-service-horizon-ms", type=float, default=3000.0)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.01)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.20)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=8.0)
    args = ap.parse_args()

    torch.set_num_threads(1)
    exact_args = make_args(args)
    exact_args.load_gated_prior_threshold = int(args.load_gated_prior_threshold)
    model = load_model(exact_args)
    load_state_into(model, args.state, torch.device(args.device))
    model.to(torch.device(args.device)).eval()

    rows: list[dict] = []
    for seed in parse_ints(args.seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                case = run_case(model, exact_args, int(initial), float(rate), int(seed))
                rows.extend(case)
                print({"initial": initial, "rate": rate, "seed": seed, "events": len(case)}, flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = pd.DataFrame(rows)
    events.to_csv(out_dir / "best_vs_edf_events.csv", index=False)
    summary = events.groupby(["category"]).agg(
        decisions=("category", "size"),
        reward=("reward", "mean"),
        active=("active_targets", "mean"),
        deadline_delta=("deadline_delta_model_minus_edf", "mean"),
        lateness_delta=("lateness_delta_model_minus_edf", "mean"),
        priority_delta=("priority_delta_model_minus_edf", "mean"),
        range_delta=("range_delta_model_minus_edf", "mean"),
    ).reset_index().sort_values("decisions", ascending=False)
    summary.to_csv(out_dir / "best_vs_edf_deviation_summary.csv", index=False)
    plot_outputs(events, out_dir)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
