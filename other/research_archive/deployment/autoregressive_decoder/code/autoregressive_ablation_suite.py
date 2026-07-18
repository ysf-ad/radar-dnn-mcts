"""Autoregressive-vs-batch ablation suite for the mutual radar model.

This isolates the main question:
    Is the batch gap caused by the learned model, or by open-loop sequence
    decoding against a stale root state?

Methods:
    BatchQScore_0r      one encoder pass, open-loop slots, learned Q + urgency
    AR_QScore_0r        same learned Q + urgency, but updates planning state
                        after every emitted action
    SequentialQ_0r      closed-loop single-action learned Q without urgency
    MCTS_PQ_r16         online model-guided MCTS
    EDF / EST           heuristics
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from final_radar_campaign import MAXT, run_fixed, seedall, summarize_window_df
from mutual_alpha_radar_loop import (
    OUT as MODEL_OUT,
    MutualArgmaxPolicyPlanner,
    MutualBatchArgmaxPlanner,
    MutualBatchQUrgencyPlanner,
    configured_env,
    make_mcts,
)
from mutual_features import slot_features, tokenize
from mutual_foundation import DEVICE, MutualRadarNet, _copy_plan_obs, advance_plan_obs
from realistic_reward_retrain import adapter
from repaired_campaign_tools import EDFPlanner, ESTPlanner, SEARCH_DWELL_MS


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "autoregressive_ablation_suite"
CLEAN = Path(r"C:\Users\yousi\Downloads\radar_outputs")
OUT.mkdir(parents=True, exist_ok=True)
CLEAN.mkdir(parents=True, exist_ok=True)


class ARQScorePlanner:
    """Closed-loop 0-rollout planner using the same Q-score as BatchQScore.

    It re-encodes after each planned action and advances a copied planning
    observation. This is the exact ablation for open-loop batch decoding.
    """

    def __init__(
        self,
        model: MutualRadarNet,
        deadline_weight: float = 8.0,
        overdue_weight: float = 8.0,
        allow_retrack: bool = False,
        search_refresh_tracked: bool = False,
        search_refresh_gain: float = 0.0,
    ):
        self.model = model.eval()
        self.deadline_weight = float(deadline_weight)
        self.overdue_weight = float(overdue_weight)
        self.allow_retrack = bool(allow_retrack)
        self.search_refresh_tracked = bool(search_refresh_tracked)
        self.search_refresh_gain = float(search_refresh_gain)
        self.adapt = adapter()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def warmup(self, obs, budget_ms=200):
        _ = self.plan(obs, budget_ms)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _scores(self, plan_obs: Dict[str, np.ndarray], elapsed: float, selected: set[int], search_count: int, track_count: int, last: int, budget_ms: float):
        x = tokenize(self.adapt, plan_obs, selected=set() if self.allow_retrack else selected, search_count=search_count)
        slot = slot_features(plan_obs, elapsed, search_count, track_count, last, float(budget_ms))
        with torch.inference_mode():
            _, _, _, type_q, track_q = self.model(
                torch.from_numpy(x).float().unsqueeze(0).to(self.device),
                torch.from_numpy(slot).float().unsqueeze(0).to(self.device),
            )
        tq = type_q[0].detach().cpu().numpy()
        trq = track_q[0].detach().cpu().numpy()
        scores = tq[0] + trq
        search_score = float(tq[1])

        visible = np.asarray(plan_obs["active_mask"], dtype=bool) & (np.asarray(plan_obs["t_deadline"], dtype=np.float32) > 0.0)
        deadline = np.asarray(plan_obs["t_deadline"], dtype=np.float32)
        desired = np.asarray(plan_obs["t_desired"], dtype=np.float32)
        urgency = (
            -self.deadline_weight * deadline / 1000.0
            + self.overdue_weight * np.maximum(0.0, -desired) / 500.0
        ).astype(np.float32)

        row = scores.copy()
        row[0] = -1e9
        upto = min(len(urgency), len(row) - 1)
        row[1 : upto + 1] += urgency[:upto]
        for action in range(1, len(row)):
            idx = action - 1
            if idx >= len(visible) or not visible[idx] or ((not self.allow_retrack) and action in selected):
                row[action] = -1e9
        best = int(np.argmax(row))
        best_score = float(row[best])
        if search_score >= best_score or best_score < -1e8:
            return 0, SEARCH_DWELL_MS
        dwell = np.asarray(plan_obs["t_dwell"], dtype=np.float32)
        dt = float(dwell[best - 1]) if 1 <= best <= len(dwell) else SEARCH_DWELL_MS
        return best, max(1.0, dt)

    def plan(self, obs, budget_ms=200):
        plan_obs = _copy_plan_obs(obs)
        selected: set[int] = set()
        plan: List[int] = []
        elapsed = 0.0
        search_count = 0
        track_count = 0
        last = -1
        while elapsed < float(budget_ms) and len(plan) < 64:
            action, dt = self._scores(plan_obs, elapsed, selected, search_count, track_count, last, float(budget_ms))
            plan.append(int(action))
            if action == 0:
                search_count += 1
            else:
                selected.add(int(action))
                track_count += 1
            advance_plan_obs(
                plan_obs,
                int(action),
                float(dt),
                search_refresh_tracked=self.search_refresh_tracked,
                search_refresh_gain=self.search_refresh_gain,
            )
            elapsed += float(dt)
            last = int(action)
        return plan if plan else [0]


def env_args():
    return SimpleNamespace(
        env_mode="operational",
        search_refresh_tracked=0,
        search_refresh_gain=0.0,
        track_update_reward=0.30,
        track_loss_penalty=8.0,
        penalize_hidden_targets=1,
        search_debt_penalty_weight=0.00025,
        mcts_rollout_policy="edf",
        mcts_rollout_search_period_ms=100.0,
        mcts_prior_uniform_mix=0.8,
        c_puct=1.25,
        expand_top_k=100,
        q_scale=100.0,
        q_utility_weight=0.15,
        leaf_value_mix=0.25,
        belief_search_weight=0.25,
        belief_search_cap=4.0,
    )


def load_model() -> MutualRadarNet:
    ckpt = MODEL_OUT / "mutual_alpha_model.pt"
    model = MutualRadarNet(d_model=96, nhead=4, nlayers=2).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def make_methods(model: MutualRadarNet, args, rollouts: int):
    env_placeholder = configured_env(3.0, args)
    return {
        "BatchPolicy_0r": lambda: MutualBatchArgmaxPlanner(model, mode="policy"),
        "AR_Policy_0r": lambda: MutualArgmaxPolicyPlanner(model, mode="policy"),
        "BatchQ_0r": lambda: MutualBatchArgmaxPlanner(model, mode="q"),
        "BatchQScore_w8_o8_0r": lambda: MutualBatchQUrgencyPlanner(model, deadline_weight=8.0, overdue_weight=8.0),
        "BatchQScore_w2_o2_0r": lambda: MutualBatchQUrgencyPlanner(model, deadline_weight=2.0, overdue_weight=2.0),
        "AR_QScore_w8_o8_0r": lambda: ARQScorePlanner(model, deadline_weight=8.0, overdue_weight=8.0),
        "AR_QScore_w2_o2_0r": lambda: ARQScorePlanner(model, deadline_weight=2.0, overdue_weight=2.0),
        "AR_QScore_w0_o0_0r": lambda: ARQScorePlanner(model, deadline_weight=0.0, overdue_weight=0.0),
        "SequentialQ_0r": lambda: MutualArgmaxPolicyPlanner(model, mode="q"),
        f"MCTS_PQ_r{rollouts}": lambda env=env_placeholder: make_mcts(model, env, args, training=False, rollouts=rollouts, mode="pq"),
        "EDF": lambda: EDFPlanner(MAXT),
        "EST": lambda: ESTPlanner(MAXT),
    }


def evaluate(model: MutualRadarNet, cells, seeds, windows: int, rollouts: int):
    args = env_args()
    rows = []
    wins = []
    for init, rate in cells:
        env = configured_env(float(rate), args)
        methods = make_methods(model, args, rollouts)
        # Rebuild MCTS lambdas with the correct env for this cell.
        methods[f"MCTS_PQ_r{rollouts}"] = lambda env=env: make_mcts(model, env, args, training=False, rollouts=rollouts, mode="pq")
        for name, factory in methods.items():
            for seed in seeds:
                seedall(int(seed))
                t0 = time.perf_counter()
                w, _ = run_fixed(factory(), name, int(init), MAXT, int(seed), int(windows), 200, env)
                s = summarize_window_df(w, "fixed")
                s.update(planner=name, initial_targets=int(init), rate=float(rate), seed=int(seed), wall_s=time.perf_counter() - t0)
                rows.append(s)
                ww = w.copy()
                ww["planner"] = name
                ww["initial_targets"] = int(init)
                ww["rate"] = float(rate)
                ww["seed"] = int(seed)
                wins.append(ww)
                print("abl_eval", init, rate, seed, name, json.dumps({k: round(float(s[k]), 4) for k in ["reward_per_200ms_eq", "planning_ms_per_200ms_eq", "mean_drop_pct_active", "mean_delay_active"]}), flush=True)
    raw = pd.DataFrame(rows)
    win = pd.concat(wins, ignore_index=True)
    return raw, win


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.groupby("planner", as_index=False).agg(
        reward=("reward_per_200ms_eq", "mean"),
        drop=("mean_drop_pct_active", "mean"),
        delay=("mean_delay_active", "mean"),
        search=("search_fraction", "mean"),
        latency=("planning_ms_per_200ms_eq", "mean"),
        final_cumulative=("final_cumulative_reward", "mean"),
        final_active=("final_active_targets", "mean"),
        final_tracked=("final_tracked_targets", "mean"),
    ).sort_values("reward", ascending=False)


def plot_suite(raw: pd.DataFrame, win: pd.DataFrame, tag: str):
    summary = summarize(raw)
    summary_path = OUT / f"{tag}_summary.csv"
    raw_path = OUT / f"{tag}_raw.csv"
    win_path = OUT / f"{tag}_windows.csv"
    summary.to_csv(summary_path, index=False)
    raw.to_csv(raw_path, index=False)
    win.to_csv(win_path, index=False)

    window_col = "window_idx" if "window_idx" in win.columns else "window"
    plot_order = summary["planner"].tolist()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()
    specs = [
        ("window_reward", "Cumulative reward", "cumsum"),
        ("mean_delay_active", "Mean active delay", "mean"),
        ("drop_pct_active", "Drop % active", "mean"),
        ("tracked_targets", "Tracked targets", "mean"),
        ("active_targets", "Active targets", "mean"),
        ("search_fraction", "Search fraction", "mean"),
    ]
    for ax, (col, title, mode) in zip(axes, specs):
        for name in plot_order:
            g = win[win["planner"] == name]
            if g.empty or col not in g:
                continue
            curve = g.groupby(window_col)[col].mean().sort_index()
            if mode == "cumsum":
                curve = curve.cumsum()
            ax.plot(curve.index * 0.2, curve.values, label=name, linewidth=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("episode time (s)")
    axes[0].set_ylabel("reward")
    axes[1].set_ylabel("ms")
    axes[2].set_ylabel("%")
    axes[3].set_ylabel("targets")
    axes[4].set_ylabel("targets")
    axes[5].set_ylabel("fraction")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    suite_path = OUT / f"{tag}_plot_suite.png"
    fig.savefig(suite_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in plot_order:
        g = win[win["planner"] == name]
        curve = g.groupby(window_col)["window_reward"].mean().sort_index().cumsum()
        ax.plot(curve.index * 0.2, curve.values, label=name, linewidth=2.2)
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("cumulative reward")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    cum_path = OUT / f"{tag}_cumulative.png"
    fig.savefig(cum_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(summary["latency"], summary["reward"], s=80)
    for _, r in summary.iterrows():
        ax.annotate(r["planner"], (r["latency"], r["reward"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("planning latency per 200ms window (ms, log)")
    ax.set_ylabel("reward per 200ms")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    frontier_path = OUT / f"{tag}_latency_reward_frontier.png"
    fig.savefig(frontier_path, dpi=180)
    plt.close(fig)

    for src in [summary_path, raw_path, win_path, suite_path, cum_path, frontier_path]:
        shutil.copy2(src, CLEAN / f"ar_ablation_{src.name}")
    return summary, suite_path, cum_path, frontier_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--seeds", default="82,83,84")
    ap.add_argument("--cells", default="50:3")
    ap.add_argument("--tag", default="heavy50_rate3_100w")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x]
    cells = []
    for item in args.cells.split(","):
        if not item:
            continue
        init, rate = item.split(":")
        cells.append((int(init), float(rate)))
    model = load_model()
    raw, win = evaluate(model, cells, seeds, args.windows, args.rollouts)
    summary, suite, cum, frontier = plot_suite(raw, win, args.tag)
    print(summary.to_string(index=False), flush=True)
    print("outputs", suite, cum, frontier, OUT / f"{args.tag}_summary.csv", flush=True)


if __name__ == "__main__":
    main()
