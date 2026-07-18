from __future__ import annotations

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from eval_exact_rescore128 import ExactRescore128, run_exact_rescore
from exact_env_mutual import SnapshotSimulator, ExactEnvMCTS, best_window_plan, _DummyPlanner
from final_radar_campaign import build_env, get_obs, run_fixed, summarize_window_df
from mutual_foundation import MutualRadarNet, DEVICE
from repaired_campaign_tools import EDFPlanner, ESTPlanner, env_preset_cfg
from strict_window_report import sample_state_metrics


OUT = Path(r"C:\Users\yousi\Downloads\radar_outputs\direct_mcts_vs_rescore_fixed")
OUT.mkdir(parents=True, exist_ok=True)
CKPT = Path(r"C:\Users\yousi\Downloads\radar_outputs\exact_train_qw02_seeded_more\exact_mutual_latest.pt")


def load_model():
    torch.set_num_threads(1)
    model = MutualRadarNet(d_model=96, nhead=4, nlayers=2).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE), strict=False)
    model.eval()
    return model


class DirectExactMCTSPlanner:
    def __init__(self, model, rollouts: int = 4, horizon_windows: int = 1):
        self.model = model
        self.rollouts = int(rollouts)
        self.horizon_windows = int(horizon_windows)

    def plan_and_commit(self, eng, debt: float, budget_ms: float = 200.0):
        sim = SnapshotSimulator(eng, debt)
        t0 = time.perf_counter()
        mcts = ExactEnvMCTS(
            self.model,
            sim,
            [],
            rollouts=self.rollouts,
            c_puct=1.25,
            expand_top_k=12,
            horizon_windows=self.horizon_windows,
            rollout_policy="edf",
            prior_mode="factorized",
            policy_target="q_softmax",
            head_mode="pq",
            q_utility_weight=0.2,
            eager_edge_depth=1,
            seed_rollout_policies=("planner_edf", "planner_est", "edf", "est", "edge"),
        )
        root = mcts.run()
        plan = best_window_plan(mcts, root, "q", budget_ms)
        plan_ms = (time.perf_counter() - t0) * 1000.0
        steps, debt = sim.commit_sequence(plan, budget_ms)
        return plan, steps, debt, plan_ms


def run_direct(planner: DirectExactMCTSPlanner, name: str, seed: int, windows_n: int, env_cfg: dict):
    eng = build_env(_DummyPlanner(), 50, 100, seed, 200, env_cfg)
    eng.reset(seed=seed)
    debt = 0.0
    cumulative = 0.0
    rows = []
    try:
        for window in range(windows_n):
            plan, steps, debt, plan_ms = planner.plan_and_commit(eng, debt, 200.0)
            if not steps:
                break
            reward = float(sum(r for r, _, _ in steps))
            spent = float(sum(dt for _, dt, _ in steps))
            search_actions = int(sum(1 for _, _, a in steps if int(a) == 0))
            cumulative += reward
            rows.append(
                {
                    "planner": name,
                    "seed": seed,
                    "window": window,
                    "window_reward": reward,
                    "cumulative_reward": cumulative,
                    "search_fraction": float(search_actions / max(1, len(steps))),
                    "planning_ms_per_decision": float(plan_ms),
                    "planning_ms_per_200ms_eq": float(plan_ms),
                    "executed_actions": int(len(steps)),
                    "spent_ms": spent,
                    **sample_state_metrics(eng, debt),
                }
            )
    finally:
        eng.close()
    return pd.DataFrame(rows)


def plot(windows: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = [
        ("cumulative_reward", "Cumulative Reward"),
        ("search_fraction", "Search Fraction"),
        ("drop_pct_active", "Drop % Active"),
        ("tracked_targets", "Tracked Targets"),
        ("mean_delay_active", "Mean Delay"),
        ("planning_ms_per_decision", "Planning ms"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for planner, sub in windows.groupby("planner"):
            curve = sub.groupby("window", as_index=False)[metric].mean()
            ax.plot(curve["window"], curve[metric], label=planner, linewidth=1.4)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUT / "direct_mcts_vs_rescore_suite.png", dpi=180)
    plt.close(fig)


def main():
    env_cfg = env_preset_cfg("repaired_stress")
    model = load_model()
    seeds = [983, 984, 985]
    windows_n = 500
    planners = {
        "DirectMCTS_r0": DirectExactMCTSPlanner(model, rollouts=0, horizon_windows=1),
        "DirectMCTS_r4": DirectExactMCTSPlanner(model, rollouts=4, horizon_windows=1),
        "DirectMCTS_r8": DirectExactMCTSPlanner(model, rollouts=8, horizon_windows=1),
        "ExactRescore_k8_h200_fixed": ExactRescore128(env_cfg, top_k=8, score_horizon_ms=200.0, slots=32, generator="structured", seed=14008),
        "ExactRescore_k4_h800_fixed": ExactRescore128(env_cfg, top_k=4, score_horizon_ms=800.0, slots=96, generator="structured", seed=18004),
        "EDF": EDFPlanner(100),
        "EST": ESTPlanner(100),
    }
    all_w = []
    rows = []
    for seed in seeds:
        for name, planner in planners.items():
            print("running", name, seed, flush=True)
            if name.startswith("DirectMCTS"):
                w = run_direct(planner, name, seed, windows_n, env_cfg)
            elif name.startswith("ExactRescore"):
                w, _ = run_exact_rescore(planner, name, seed, windows_n, env_cfg)
            else:
                w, _ = run_fixed(planner, name, 50, 100, seed, windows_n, 200, env_cfg)
            all_w.append(w)
            s = summarize_window_df(w, "fixed")
            s.update(planner=name, seed=seed)
            rows.append(s)
    windows = pd.concat(all_w, ignore_index=True)
    raw = pd.DataFrame(rows)
    summary = raw.groupby("planner", as_index=False).mean(numeric_only=True)
    edf = raw[raw.planner == "EDF"].set_index("seed")["reward_per_200ms_eq"]
    gaps = []
    for _, r in raw.iterrows():
        gaps.append({**r.to_dict(), "gap_vs_edf": float(r.reward_per_200ms_eq - edf.loc[r.seed])})
    gap_df = pd.DataFrame(gaps)
    windows.to_csv(OUT / "windows.csv", index=False)
    raw.to_csv(OUT / "by_seed.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    gap_df.to_csv(OUT / "by_seed_with_gaps.csv", index=False)
    plot(windows)
    print(
        summary[
            [
                "planner",
                "reward_per_200ms_eq",
                "mean_drop_pct_active",
                "mean_tracked_targets",
                "mean_active_targets",
                "mean_delay_active",
                "search_fraction",
                "planning_ms_per_decision",
            ]
        ]
        .sort_values("reward_per_200ms_eq", ascending=False)
        .to_string(index=False)
    )
    print(
        gap_df.groupby("planner", as_index=False)
        .agg(avg_gap_vs_edf=("gap_vs_edf", "mean"), min_gap_vs_edf=("gap_vs_edf", "min"))
        .sort_values("avg_gap_vs_edf", ascending=False)
        .to_string(index=False)
    )
    print(OUT)


if __name__ == "__main__":
    main()
