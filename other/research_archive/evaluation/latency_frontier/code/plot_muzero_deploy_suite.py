from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "CreateValid1" / "results"
OUT = RESULTS / "muzero_deploy_suite"


def _load_windows(path: Path, label: str, planner: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if planner is not None:
        df = df[df["planner"].astype(str) == planner].copy()
    df["method"] = label
    if "initial" not in df.columns:
        df["initial"] = -1
    if "rate" not in df.columns:
        df["rate"] = -1.0
    return df


def _window_mean(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "window_reward",
        "search_fraction",
        "tracked_targets",
        "drop_pct_active",
        "mean_delay_active",
        "executed_actions",
    ]
    return df.groupby(["method", "window"], as_index=False)[metrics].mean()


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        rows.append(
            {
                "method": method,
                "reward_per_window": float(g["window_reward"].mean()),
                "tracked_targets": float(g["tracked_targets"].mean()),
                "drop_pct_active": float(g["drop_pct_active"].mean()),
                "drop_count_est": float(((g["drop_pct_active"] / 100.0) * g["active_targets"]).mean()),
                "mean_delay_active": float(g["mean_delay_active"].mean()),
                "search_fraction": float(g["search_fraction"].mean()),
                "planning_ms_per_window": float(g["planning_ms_per_decision"].mean()),
                "executed_actions": float(g["executed_actions"].mean()),
                "n_windows": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("reward_per_window", ascending=False)


def _plot_suite(mean_df: pd.DataFrame, out: Path) -> None:
    colors = {
        "0-rollout latent decoder": "#1f4ea8",
        "Original action-attention": "#7b3294",
        "Flat PQ": "#f28e2b",
        "Flat20 PQ": "#9467bd",
        "EDF": "#d95f02",
        "EST": "#1b9e77",
    }
    panels = [
        ("cum_reward", "window_reward", "Cumulative reward", "Cumulative reward", True),
        ("window_reward", "window_reward", "Window reward, 10-window avg.", "Reward/window", False),
        ("search", "search_fraction", "Search ratio", "Search ratio", False),
        ("tracked", "tracked_targets", "Tracked targets", "Targets", False),
        ("drop", "drop_pct_active", "Dropped / active", "Percent", False),
        ("delay", "mean_delay_active", "Mean delay", "Delay (ms)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), dpi=180)
    for ax, (_name, col, title, ylabel, cumulative) in zip(axes.ravel(), panels):
        for method, g in mean_df.groupby("method", sort=False):
            g = g.sort_values("window")
            y = g[col].cumsum() if cumulative else g[col].rolling(10, min_periods=1).mean()
            ax.plot(g["window"], y, label=method, linewidth=2.0, color=colors.get(method))
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xlabel("Window", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.015))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(summary: pd.DataFrame, out: Path) -> None:
    cols = [
        ("reward_per_window", "Reward / window", False),
        ("drop_pct_active", "Dropped / active (%)", True),
        ("mean_delay_active", "Delay (ms)", True),
        ("planning_ms_per_window", "Latency (ms/window)", True),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), dpi=180)
    order = summary["method"].tolist()
    for ax, (col, title, _lower_better) in zip(axes, cols):
        vals = summary.set_index("method").loc[order, col]
        ax.barh(order, vals, color="#1f4ea8")
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(True, axis="x", alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [
        _load_windows(RESULTS / "eval_validated_cap25_prefix1_3seed_9cell_100w.csv", "0-rollout latent decoder"),
        _load_windows(RESULTS / "eval_validated_cap35_prefix2_3seed_9cell_100w.csv", "Original action-attention", "ActionAttention_full"),
        _load_windows(RESULTS / "eval_validated_cap35_prefix2_3seed_9cell_100w.csv", "EDF", "EDF"),
        _load_windows(RESULTS / "eval_validated_cap35_prefix2_3seed_9cell_100w.csv", "EST", "EST"),
        _load_windows(RESULTS / "codex_flat_current_harness_9cell_100w_windows.csv", "Flat PQ"),
        _load_windows(RESULTS / "codex_flat20_current_harness_9cell_100w_windows.csv", "Flat20 PQ"),
    ]
    all_df = pd.concat(frames, ignore_index=True)
    mean_df = _window_mean(all_df)
    summary = _summarize(all_df)
    all_df.to_csv(OUT / "window_trace_inputs.csv", index=False)
    mean_df.to_csv(OUT / "window_mean_traces.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    _plot_suite(mean_df, OUT / "six_metric_suite.png")
    _plot_summary(summary, OUT / "summary_bars.png")
    (OUT / "README.txt").write_text(
        "\n".join(
            [
                "MuZero deployment comparison suite",
                "",
                "0-rollout latent decoder is the fast MuZero-style deployment path:",
                "  encode state once, recurrently decode joint S/X actions in latent space, no online PUCT.",
                "",
                "Original action-attention, EDF, and EST are from eval_validated_cap35_prefix2_3seed_9cell_100w.csv.",
                "0-rollout latent decoder is from eval_validated_cap25_prefix1_3seed_9cell_100w.csv.",
                "Flat PQ and Flat20 PQ are labelled current-harness references, not the same validated harness.",
            ]
        ),
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
