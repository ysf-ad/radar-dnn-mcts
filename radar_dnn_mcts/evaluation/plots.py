from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _mean_by_window(data: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "cumulative_reward",
        "tracked_targets",
        "dropped_targets",
        "mean_delay_active",
        "search_fraction",
        "latency_ms",
    ]
    return data.groupby(["method", "window"], as_index=False)[numeric].mean()


def plot_suite(windows: pd.DataFrame, output: str | Path) -> Path:
    data = _mean_by_window(windows)
    panels = [
        ("cumulative_reward", "Cumulative reward", "Reward"),
        ("latency_ms", "Warmed scheduling latency", "ms/window"),
        ("tracked_targets", "Tracked targets", "Targets"),
        ("dropped_targets", "Dropped targets", "Targets"),
        ("mean_delay_active", "Mean delay", "ms"),
        ("search_fraction", "Search ratio", "Ratio"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (column, title, ylabel) in zip(axes.flat, panels):
        for method, group in data.groupby("method"):
            ax.plot(group["window"], group[column], label=method, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Window")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
