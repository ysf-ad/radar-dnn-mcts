from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from alphazero_benchmark import load_state_into
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, env_cfg_for, load_model, run_fixed
from single_sensor_cadence_probe import (
    FrameAwareQuotaPlanner,
    QuotaPlanner,
    TreeMacroArmPlanner,
    make_direct_base,
    make_exact_args,
    summarize_df,
)


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "CreateValid1" / "results"
OUT = RESULTS / "best_model_plot_suite"

CKPT = Path(r"C:\Users\yousi\Downloads\radar_outputs\alphazero_orthodox\paper_factorized_pv_current_best.pt")
STATE = RESULTS / "az_hard1323_rawret_k16_probe_state.pt"
MACRO_JSON = RESULTS / "macro_arm_selector_learned_only_quota4568_v1.json"


def _base_args(windows: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        ckpt=str(CKPT),
        state=str(STATE),
        device="cpu",
        windows=int(windows),
        direct_threshold=0.5,
        direct_mode="branch",
        direct_alpha=0.0,
        direct_beta=0.0,
        q_residual_gate="off",
        q_gate_margin=0.0,
        env_mode="penalty_only_frame",
        use_arrival_feature=False,
        use_grid_feature=False,
        track_loss_penalty=8.0,
        search_frame_overdue_weight=1.0,
        search_frame_drop_penalty=16.0,
    )


def load_aggregate() -> pd.DataFrame:
    sources = [
        (
            RESULTS / "az_hard1323_rawret_k16_probe_quota_ood_150case.csv",
            {"PV_quota_4": "PV quota 4", "EDF": "EDF", "EST": "EST"},
        ),
        (
            RESULTS / "az_hard1323_rawret_k16_probe_tree_arm_learned_only_ood_150case.csv",
            {"PV_tree_arm_macro": "Learned macro"},
        ),
        (
            RESULTS / "frameaware_min4_fixed_ood_150case.csv",
            {"PV_frame_4_18_3000_4500_4": "Frame-aware macro"},
        ),
    ]
    frames = []
    for path, names in sources:
        df = pd.read_csv(path)
        df = df[df["method"].isin(names)].copy()
        df["method"] = df["method"].map(names)
        frames.append(df)
    agg = pd.concat(frames, ignore_index=True)
    agg = agg.drop_duplicates(subset=["method", "initial", "rate", "seed"], keep="last")
    return agg


def add_margins(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index=["initial", "rate", "seed"], columns="method", values="reward")
    best_heur = pivot[["EDF", "EST"]].max(axis=1)
    out = df.copy()
    out["best_heuristic"] = [
        float(best_heur.loc[(int(r.initial), float(r.rate), int(r.seed))])
        for r in out.itertuples(index=False)
    ]
    out["margin_vs_best_heuristic"] = out["reward"] - out["best_heuristic"]
    return out


def save_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        if method in {"EDF", "EST"}:
            rows.append(
                {
                    "method": method,
                    "reward": g["reward"].mean(),
                    "search": g["search"].mean(),
                    "latency_ms": g["latency_ms"].mean(),
                    "wins": np.nan,
                    "eps01": np.nan,
                    "mean_margin": np.nan,
                    "min_margin": np.nan,
                    "cases": len(g),
                }
            )
        else:
            margin = g["margin_vs_best_heuristic"]
            rows.append(
                {
                    "method": method,
                    "reward": g["reward"].mean(),
                    "search": g["search"].mean(),
                    "latency_ms": g["latency_ms"].mean(),
                    "wins": int((margin > 0.0).sum()),
                    "eps01": int((margin >= -0.1).sum()),
                    "mean_margin": margin.mean(),
                    "min_margin": margin.min(),
                    "cases": len(g),
                }
            )
    summary = pd.DataFrame(rows).sort_values("reward", ascending=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    return summary


def line_plot(df: pd.DataFrame, x: str, y: str, path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for method, g in df.groupby("method"):
        s = g.groupby(x)[y].mean().sort_index()
        ax.plot(s.index, s.values, marker="o", linewidth=2, label=method)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def bar_summary(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("reward", "Mean reward"),
        ("search", "Search fraction"),
        ("wins", "Wins vs best heuristic"),
        ("latency_ms", "Latency ms/window"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        s = summary.dropna(subset=[col]).sort_values(col, ascending=(col != "reward"))
        ax.barh(s["method"], s[col], color="#4169a8")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "summary_bars.png", dpi=180)
    plt.close(fig)


def heatmap(df: pd.DataFrame, method: str, value: str, path: Path, title: str) -> None:
    mat = df[df["method"] == method].pivot_table(index="initial", columns="rate", values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(mat.columns)), [str(c) for c in mat.columns])
    ax.set_yticks(range(len(mat.index)), [str(i) for i in mat.index])
    ax.set_xlabel("Arrival rate")
    ax.set_ylabel("Initial targets")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_planner(name: str, model, args: SimpleNamespace, rate: float):
    if name == "EDF":
        return EDFPlanner(MAXT)
    if name == "EST":
        return ESTPlanner(MAXT)
    base = make_direct_base(model, args)
    if name == "PV quota 4":
        return QuotaPlanner(base, 4)
    if name == "Frame-aware macro":
        return FrameAwareQuotaPlanner(base, 4, 18, 3000.0, 4500.0, 4)
    if name == "Learned macro":
        selector = json.loads(MACRO_JSON.read_text())
        return TreeMacroArmPlanner(base, selector, scenario_rate=float(rate))
    raise ValueError(name)


def trace_suite() -> None:
    args = _base_args(100)
    exact_args = make_exact_args(args)
    model = load_model(exact_args).to(torch.device("cpu"))
    load_state_into(model, str(STATE), torch.device("cpu"))
    model.eval()

    trace_cases = [
        (10, 0.0, 1401),
        (20, 4.0, 1404),
        (60, 8.0, 1401),
        (100, 4.0, 1405),
    ]
    methods = ["EDF", "EST", "PV quota 4", "Learned macro", "Frame-aware macro"]
    all_windows = []
    trace_summary = []
    for init, rate, seed in trace_cases:
        env_cfg = env_cfg_for(rate, exact_args)
        for method in methods:
            planner = make_planner(method, model, args, rate)
            df, _ = run_fixed(planner, method, int(init), MAXT, int(seed), 100, 200, env_cfg)
            df["initial"] = int(init)
            df["rate"] = float(rate)
            df["case"] = f"init={init}, rate={rate:g}, seed={seed}"
            all_windows.append(df)
            trace_summary.append({"case": df["case"].iloc[0], "method": method, **summarize_df(df)})
    windows = pd.concat(all_windows, ignore_index=True)
    windows.to_csv(OUT / "representative_window_traces.csv", index=False)
    pd.DataFrame(trace_summary).to_csv(OUT / "representative_trace_summary.csv", index=False)

    for case, gcase in windows.groupby("case"):
        safe = case.replace("=", "").replace(", ", "_").replace(" ", "_")
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for method, g in gcase.groupby("planner"):
            axes[0].plot(g["window"], g["cumulative_reward"], label=method, linewidth=2)
            roll = g["search_fraction"].rolling(10, min_periods=1).mean()
            axes[1].plot(g["window"], roll, label=method, linewidth=2)
        axes[0].set_title(case)
        axes[0].set_ylabel("Cumulative reward")
        axes[1].set_ylabel("Search fraction (10-window mean)")
        axes[1].set_xlabel("Window")
        for ax in axes:
            ax.grid(alpha=0.25)
        axes[0].legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"trace_{safe}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = add_margins(load_aggregate())
    df.to_csv(OUT / "aggregate_best_models.csv", index=False)
    summary = save_summary(df)

    bar_summary(summary)
    line_plot(df, "initial", "reward", OUT / "reward_vs_initial.png", "Mean reward")
    line_plot(df, "rate", "reward", OUT / "reward_vs_rate.png", "Mean reward")
    line_plot(df, "initial", "search", OUT / "search_vs_initial.png", "Search fraction")
    line_plot(df, "rate", "search", OUT / "search_vs_rate.png", "Search fraction")
    line_plot(df, "initial", "margin_vs_best_heuristic", OUT / "margin_vs_initial.png", "Margin vs best heuristic")
    line_plot(df, "rate", "margin_vs_best_heuristic", OUT / "margin_vs_rate.png", "Margin vs best heuristic")
    heatmap(df, "Learned macro", "margin_vs_best_heuristic", OUT / "learned_macro_margin_heatmap.png", "Learned macro margin")
    heatmap(df, "Frame-aware macro", "margin_vs_best_heuristic", OUT / "frameaware_margin_heatmap.png", "Frame-aware macro margin")
    trace_suite()
    print(f"wrote plot suite to {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
