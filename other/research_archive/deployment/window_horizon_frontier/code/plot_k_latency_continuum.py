from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.summary).sort_values("k")
    x = range(len(frame))

    fig, ax = plt.subplots(figsize=(10.8, 4.6), dpi=240)
    ax.plot(x, frame.latency_ms_per_window, color="#173f8a", linewidth=2.8, zorder=2)
    ax.scatter(x, frame.latency_ms_per_window, s=78, color="#2467b2", edgecolor="white", linewidth=1.2, zorder=3)

    for idx, row in frame.reset_index(drop=True).iterrows():
        ax.annotate(
            f"{row.latency_ms_per_window:.1f} ms",
            (idx, row.latency_ms_per_window),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#15345f",
        )
        ax.annotate(
            f"{row.replans_per_window:.2f} calls/window",
            (idx, row.latency_ms_per_window),
            xytext=(0, -19),
            textcoords="offset points",
            ha="left" if idx == 0 else ("right" if idx == len(frame) - 1 else "center"),
            fontsize=8.5,
            color="#596779",
        )

    ax.set_xticks(list(x), [f"K={int(k)}" for k in frame.k])
    ax.set_xlabel("Actions executed before replanning")
    ax.set_ylabel("Warmed planning latency (ms / 200 ms window)")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(frame.latency_ms_per_window) * 1.18)

    ax.text(0.0, 1.055, "Full re-encode", transform=ax.transAxes, ha="left", va="bottom",
            color="#173f8a", fontsize=11, fontweight="bold")
    ax.text(0.5, 1.055, "Receding-horizon continuum", transform=ax.transAxes, ha="center", va="bottom",
            color="#596779", fontsize=10)
    ax.text(1.0, 1.055, "Near-batched schedule", transform=ax.transAxes, ha="right", va="bottom",
            color="#173f8a", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(0.98, 1.025), xytext=(0.02, 1.025), xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "#8a98aa", "linewidth": 1.4})

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
