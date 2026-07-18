from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import torch

from final_radar_campaign import MAXT, run_fixed, seedall, summarize_window_df
from mutual_alpha_radar_loop import MutualArgmaxPolicyPlanner, MutualBatchQUrgencyPlanner, collect_mutual_targets, configured_env
from mutual_foundation import DEVICE, MutualRadarNet
from q_residual_training_ablation import base_args, q_scale_for, train_variant
from repaired_campaign_tools import EDFPlanner, ESTPlanner


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "reward_rebalance_suite"
CLEAN = Path(r"C:\Users\yousi\Downloads\radar_outputs")
OUT.mkdir(parents=True, exist_ok=True)
CLEAN.mkdir(parents=True, exist_ok=True)


def load_base() -> MutualRadarNet:
    ckpt = ROOT / "CreateValid1" / "results" / "mutual_alpha_radar_loop" / "mutual_alpha_model.pt"
    model = MutualRadarNet(d_model=96, nhead=4, nlayers=2).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


def cfg_for(name: str) -> SimpleNamespace:
    args = base_args()
    args.episodes_per_iter = 4
    args.windows_per_episode = 6
    args.rollouts = 8
    args.train_initials = "15,30,50,75,100"
    args.train_rates = "1,2,3,4"
    args.selfplay_replan_each_action = True
    args.search_refresh_tracked = 0
    args.search_refresh_gain = 0.0
    args.track_update_reward = 0.30
    args.track_loss_penalty = 4.0
    args.penalize_hidden_targets = 1
    args.search_debt_penalty_weight = 0.006
    args.sector_staleness_weight = 0.0
    if name == "current_strict":
        args.env_mode = "operational"
        args.track_loss_penalty = 8.0
        args.search_debt_penalty_weight = 0.00025
    elif name == "original_reward":
        args.env_mode = "original_reward"
    elif name == "balanced_linear_002":
        args.env_mode = "balanced_linear"
        args.search_debt_penalty_weight = 0.002
        args.track_loss_penalty = 4.0
    elif name == "balanced_linear_006":
        args.env_mode = "balanced_linear"
        args.search_debt_penalty_weight = 0.006
        args.track_loss_penalty = 4.0
    elif name == "balanced_linear_012":
        args.env_mode = "balanced_linear"
        args.search_debt_penalty_weight = 0.012
        args.track_loss_penalty = 4.0
    elif name == "balanced_linear_006_loss8":
        args.env_mode = "balanced_linear"
        args.search_debt_penalty_weight = 0.006
        args.track_loss_penalty = 8.0
    else:
        raise ValueError(name)
    return args


def quick_eval_existing(model: MutualRadarNet, reward_names: list[str]):
    rows = []
    for rname in reward_names:
        args = cfg_for(rname)
        for init, rate in [(15, 1.0), (30, 2.0), (50, 3.0), (75, 4.0), (100, 4.0)]:
            env = configured_env(rate, args)
            methods = {
                "ResidualQ_unretrained": lambda: MutualArgmaxPolicyPlanner(model, mode="q"),
                "EDF": lambda: EDFPlanner(MAXT),
                "EST": lambda: ESTPlanner(MAXT),
            }
            for name, factory in methods.items():
                seedall(82)
                w, _ = run_fixed(factory(), name, init, MAXT, 82, 24, 200, env)
                s = summarize_window_df(w, "fixed")
                s.update(reward_cfg=rname, planner=name, initial_targets=init, rate=rate)
                rows.append(s)
                print("quick_reward_eval", rname, init, name, round(float(s["reward_per_200ms_eq"]), 3), round(float(s["search_fraction"]), 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "quick_existing_reward_sweep.csv", index=False)
    return df


def train_one(base: MutualRadarNet, rname: str, steps: int):
    args = cfg_for(rname)
    rows, _, windows = collect_mutual_targets(base, args, iteration=0)
    stats = {
        "reward_cfg": rname,
        "targets": len(rows),
        "q_scale": q_scale_for(rows),
        "search_pi_mean": float(sum(float(r.pi[0]) for r in rows) / max(1, len(rows))),
        "mean_selfplay_reward": float(sum(float(w["reward"]) for w in windows) / max(1, len(windows))),
    }
    print("train_data_stats", json.dumps(stats), flush=True)
    tag, model = train_variant(base, rows, "residual_max", True, steps, lr=2e-4)
    path = OUT / f"{rname}_{tag}.pt"
    torch.save(model.cpu().state_dict(), path)
    model.to(DEVICE).eval()
    pd.DataFrame([stats]).to_csv(OUT / f"{rname}_target_stats.csv", index=False)
    return path, model, stats


def evaluate_trained(variants, windows: int):
    raw_rows = []
    win_rows = []
    for rname, model in variants:
        args = cfg_for(rname)
        for init, rate in [(15, 1.0), (30, 2.0), (50, 3.0), (75, 4.0), (100, 4.0)]:
            env = configured_env(rate, args)
            methods = {
                "SequentialQ_0r": lambda model=model: MutualArgmaxPolicyPlanner(model, mode="q"),
                "BatchQ_0r": lambda model=model: MutualBatchQUrgencyPlanner(model, deadline_weight=8.0, overdue_weight=8.0),
                "EDF": lambda: EDFPlanner(MAXT),
                "EST": lambda: ESTPlanner(MAXT),
            }
            for planner, factory in methods.items():
                for seed in [82, 83]:
                    seedall(seed)
                    w, _ = run_fixed(factory(), planner, init, MAXT, seed, windows, 200, env)
                    s = summarize_window_df(w, "fixed")
                    s.update(reward_cfg=rname, planner=planner, initial_targets=init, rate=rate, seed=seed)
                    raw_rows.append(s)
                    ww = w.copy()
                    ww["reward_cfg"] = rname
                    ww["planner"] = planner
                    ww["initial_targets"] = init
                    ww["rate"] = rate
                    ww["seed"] = seed
                    win_rows.append(ww)
                    print("trained_reward_eval", rname, init, seed, planner, round(float(s["reward_per_200ms_eq"]), 3), round(float(s["search_fraction"]), 3), flush=True)
    raw = pd.DataFrame(raw_rows)
    win = pd.concat(win_rows, ignore_index=True)
    raw.to_csv(OUT / "trained_reward_eval_raw.csv", index=False)
    win.to_csv(OUT / "trained_reward_eval_windows.csv", index=False)
    return raw, win


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["reward_cfg", "planner"], as_index=False)
        .agg(
            reward=("reward_per_200ms_eq", "mean"),
            drop=("mean_drop_pct_active", "mean"),
            delay=("mean_delay_active", "mean"),
            search=("search_fraction", "mean"),
            latency=("planning_ms_per_200ms_eq", "mean"),
            final_cumulative=("final_cumulative_reward", "mean"),
        )
        .sort_values(["reward_cfg", "reward"], ascending=[True, False])
    )
    summary.to_csv(OUT / "trained_reward_eval_summary.csv", index=False)
    return summary


def plot_search_by_load(raw: pd.DataFrame):
    seq = raw[raw["planner"].eq("SequentialQ_0r")]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for rname, g in seq.groupby("reward_cfg"):
        curve = g.groupby("initial_targets")["search_fraction"].mean()
        ax.plot(curve.index, curve.values, marker="o", linewidth=2, label=rname)
    ax.set_title("Residual Q search fraction by load")
    ax.set_xlabel("initial targets")
    ax.set_ylabel("search fraction")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / "search_fraction_by_load.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_cumulative(win: pd.DataFrame, reward_cfg: str):
    g = win[win["reward_cfg"].eq(reward_cfg)]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    for planner, h in g.groupby("planner"):
        curve = h.groupby("window")["window_reward"].mean().cumsum()
        ax.plot(curve.index * 0.2, curve.values, label=planner, linewidth=2)
    ax.set_title(f"{reward_cfg}: cumulative reward")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cumulative reward")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT / f"{reward_cfg}_cumulative.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def write_html(summary: pd.DataFrame, stats: list[dict], plots: list[Path]):
    cards = "\n".join(
        f'<section><h2>{p.stem}</h2><a href="{p.name}"><img src="{p.name}" style="max-width:560px;width:100%;border:1px solid #ddd"></a></section>'
        for p in plots
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Reward Rebalance Suite</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:white;color:#111}}table{{border-collapse:collapse;font-size:13px;margin-bottom:24px}}th,td{{border:1px solid #ddd;padding:6px 8px;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}section{{display:inline-block;vertical-align:top;margin:0 18px 24px 0}}h1{{font-size:24px}}h2{{font-size:16px}}</style></head>
<body><h1>Reward Rebalance Suite</h1>
<h2>Training Target Stats</h2>{pd.DataFrame(stats).to_html(index=False, float_format=lambda x: f"{x:.4f}")}
<h2>Evaluation Summary</h2>{summary.to_html(index=False, float_format=lambda x: f"{x:.4f}")}
{cards}</body></html>"""
    p = OUT / "reward_rebalance_suite.html"
    p.write_text(html, encoding="utf-8")
    return p


def main():
    torch.set_num_threads(1)
    base = load_base()
    reward_names = ["current_strict", "original_reward", "balanced_linear_002", "balanced_linear_006", "balanced_linear_012", "balanced_linear_006_loss8"]
    quick_eval_existing(base, reward_names)
    train_names = ["original_reward", "balanced_linear_002", "balanced_linear_006", "balanced_linear_012"]
    variants = []
    stats = []
    for rname in train_names:
        _, model, st = train_one(base, rname, steps=90)
        variants.append((rname, model))
        stats.append(st)
    raw, win = evaluate_trained(variants, windows=40)
    summary = summarize(raw)
    plots = [plot_search_by_load(raw)]
    for rname in train_names:
        plots.append(plot_cumulative(win, rname))
    html = write_html(summary, stats, plots)
    for p in [html, OUT / "trained_reward_eval_summary.csv", OUT / "trained_reward_eval_raw.csv", OUT / "trained_reward_eval_windows.csv", OUT / "quick_existing_reward_sweep.csv", *plots]:
        shutil.copy2(p, CLEAN / f"reward_rebalance_{p.name}")
    print(summary.to_string(index=False), flush=True)
    print("HTML", html, flush=True)


if __name__ == "__main__":
    main()
