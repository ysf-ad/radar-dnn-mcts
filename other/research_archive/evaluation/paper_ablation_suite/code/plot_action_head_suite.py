from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "flat_PQ": "Flat PQ",
    "two_row_factorized_PQ": "Factorized PQ",
    "two_row_action_attention_qpolicy_factored_loss_PQ": "Factorized Target PQ",
    "two_row_action_attention_qpolicy_factored_loss_PQ1": "Factorized Target PQ1",
    "two_sensor_type_PQ": "Best Factorized PQ",
    "two_sensor_type_m08": "Factorized Type m08",
    "two_sensor_type_m09": "Factorized Type m09",
    "two_sensor_type_m15": "Best Factorized m15",
    "two_sensor_type_m1": "Factorized Type m1",
    "flat_search20_PQ": "Flat Search20 PQ",
    "fair_exact": "Fair Exact",
    "EDF": "EDF",
    "EST": "EST",
}


FACTORIZED_PRIORITY = [
    "two_row_action_attention_qpolicy_factored_loss_PQ1",
    "two_row_action_attention_qpolicy_factored_loss_PQ",
    "two_sensor_type_m15",
    "two_sensor_type_m1",
    "two_sensor_type_m08",
    "two_sensor_type_PQ",
    "two_sensor_type_m09",
    "two_row_factorized_PQ",
]


WINDOW_PREFIX = [
    "planner",
    "seed",
    "window",
    "elapsed_ms",
    "window_reward",
    "cumulative_reward",
    "search_fraction",
    "planning_ms_per_decision",
    "planning_ms_per_executed_action",
    "executed_actions",
    "spent_ms",
    "active_targets",
    "tracked_targets",
    "drop_pct_active",
    "mean_delay_active",
    "search_debt_end_ms",
]


def read_windows(path: str | Path) -> pd.DataFrame:
    rows = []
    with Path(path).open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for parts in reader:
            if len(parts) < len(WINDOW_PREFIX) + 3:
                continue
            row = {name: parts[i] for i, name in enumerate(WINDOW_PREFIX)}
            row["method"] = parts[-3]
            row["initial"] = parts[-2]
            row["rate"] = parts[-1]
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in [c for c in df.columns if c not in {"planner", "method"}]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pivot(raw: pd.DataFrame, method: str, value: str) -> pd.DataFrame:
    sub = raw[raw["method"] == method]
    table = sub.pivot_table(index="initial", columns="rate", values=value, aggfunc="mean")
    return table.sort_index().sort_index(axis=1)


def draw_heatmaps(raw: pd.DataFrame, methods: list[str], value: str, title: str, path: Path, cmap: str) -> None:
    tables = [pivot(raw, method, value) for method in methods]
    vals = np.concatenate([t.to_numpy().reshape(-1) for t in tables if not t.empty])
    vals = vals[np.isfinite(vals)]
    vmin = float(vals.min()) if len(vals) else None
    vmax = float(vals.max()) if len(vals) else None
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8), constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, method, table in zip(axes, methods, tables):
        im = ax.imshow(table.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(METHOD_LABELS.get(method, method))
        ax.set_xlabel("arrival rate")
        ax.set_ylabel("initial targets")
        ax.set_xticks(range(len(table.columns)), [str(c).rstrip("0").rstrip(".") for c in table.columns])
        ax.set_yticks(range(len(table.index)), [str(int(i)) for i in table.index])
        for r in range(table.shape[0]):
            for c in range(table.shape[1]):
                val = table.iloc[r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_margin_heatmaps(raw: pd.DataFrame, baseline: str, methods: list[str], path: Path) -> pd.DataFrame:
    keys = ["initial", "rate", "seed"]
    base = raw[raw["method"] == baseline][[*keys, "reward"]].rename(columns={"reward": "baseline_reward"})
    rows = []
    for method in methods:
        sub = raw[raw["method"] == method][[*keys, "reward"]].rename(columns={"reward": "method_reward"})
        joined = sub.merge(base, on=keys, how="inner")
        joined["margin"] = joined["method_reward"] - joined["baseline_reward"]
        joined["method"] = method
        rows.append(joined)
    margins = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 3.8), constrained_layout=True)
    if len(methods) == 1:
        axes = [axes]
    vals = margins["margin"].to_numpy(dtype=float)
    lim = float(np.nanmax(np.abs(vals))) if len(vals) else 1.0
    for ax, method in zip(axes, methods):
        table = margins[margins["method"] == method].pivot_table(index="initial", columns="rate", values="margin", aggfunc="mean")
        table = table.sort_index().sort_index(axis=1)
        im = ax.imshow(table.to_numpy(), cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"{METHOD_LABELS.get(method, method)} - {METHOD_LABELS.get(baseline, baseline)}")
        ax.set_xlabel("arrival rate")
        ax.set_ylabel("initial targets")
        ax.set_xticks(range(len(table.columns)), [str(c).rstrip("0").rstrip(".") for c in table.columns])
        ax.set_yticks(range(len(table.index)), [str(int(i)) for i in table.index])
        for r in range(table.shape[0]):
            for c in range(table.shape[1]):
                val = table.iloc[r, c]
                if np.isfinite(val):
                    ax.text(c, r, f"{val:+.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.suptitle("Reward Margin Heatmaps")
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return margins


def draw_cumulative(win: pd.DataFrame, methods: list[str], rate: float, path: Path) -> None:
    initials = sorted(win.loc[win["rate"] == rate, "initial"].dropna().unique())
    fig, axes = plt.subplots(1, len(initials), figsize=(4.8 * len(initials), 3.8), constrained_layout=True)
    if len(initials) == 1:
        axes = [axes]
    for ax, initial in zip(axes, initials):
        for method in methods:
            sub = win[(win["method"] == method) & (win["rate"] == rate) & (win["initial"] == initial)]
            if sub.empty:
                continue
            curve = sub.groupby("window")["window_reward"].mean().sort_index().cumsum()
            ax.plot(curve.index + 1, curve.values, label=METHOD_LABELS.get(method, method), linewidth=2)
        ax.set_title(f"R{rate:g}, initial {int(initial)}")
        ax.set_xlabel("window")
        ax.set_ylabel("cumulative reward")
        ax.grid(alpha=0.25)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(f"Cumulative Reward Curves, Rate {rate:g}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def choose_focus_methods(raw: pd.DataFrame) -> list[str]:
    available = set(raw["method"].dropna().astype(str))
    focus = [m for m in ["flat_PQ", "flat_search20_PQ"] if m in available]
    factorized = next((m for m in FACTORIZED_PRIORITY if m in available), None)
    if factorized is not None:
        focus.append(factorized)
    return focus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    win = read_windows(args.windows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    focus = choose_focus_methods(raw)
    baselines = [m for m in ["fair_exact", "EDF", "EST"] if m in set(raw["method"].dropna().astype(str))]
    raw["method_label"] = raw["method"].map(lambda x: METHOD_LABELS.get(x, x))
    summary = raw.groupby("method", as_index=False).agg(
        reward=("reward", "mean"),
        search=("search", "mean"),
        latency_ms=("latency_ms", "mean"),
        n=("reward", "size"),
    ).sort_values("reward", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    raw.to_csv(out_dir / "cell_sweep.csv", index=False)

    draw_heatmaps(raw, focus, "reward", "Reward Per 200ms, Learned Heads", out_dir / "reward_heatmaps_learned.png", "YlGnBu")
    draw_heatmaps(raw, focus + baselines, "reward", "Reward Per 200ms, All Methods", out_dir / "reward_heatmaps_all.png", "YlGnBu")
    draw_heatmaps(raw, focus, "search", "Search Fraction, Learned Heads", out_dir / "search_heatmaps_learned.png", "PuBuGn")
    margin_methods = [m for m in focus if m != "flat_PQ"]
    margins = draw_margin_heatmaps(raw, "flat_PQ", margin_methods, out_dir / "margin_vs_flat.png")
    margins.to_csv(out_dir / "margins_vs_flat.csv", index=False)

    cumulative_methods = focus + [m for m in ["fair_exact", "EDF", "EST"] if m in set(win["method"].dropna().astype(str))]
    for rate in sorted(raw["rate"].dropna().unique()):
        draw_cumulative(win, cumulative_methods, float(rate), out_dir / f"cumulative_r{float(rate):g}.png")

    print("summary", (out_dir / "summary.csv").resolve())
    print("cell_sweep", (out_dir / "cell_sweep.csv").resolve())
    print("reward_heatmaps", (out_dir / "reward_heatmaps_learned.png").resolve())
    print("margin", (out_dir / "margin_vs_flat.png").resolve())


if __name__ == "__main__":
    main()
