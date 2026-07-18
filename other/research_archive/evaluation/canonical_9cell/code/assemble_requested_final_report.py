from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
OUT = RESULTS / "requested_final_sonly_methods_20260707"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def maybe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def summarize_rows() -> pd.DataFrame:
    strict = read_csv(RESULTS / "final_sonly_requested_methods_20260707" / "canonical_summary.csv")
    fair = read_csv(RESULTS / "final_fair_reencode_best_current_sonly" / "summary.csv")
    puct4 = read_csv(RESULTS / "puct_current_stack_clean_smoke" / "exact_puct_terminal_sf8_svc025_s4k8_9cell_50w_summary.csv")
    puct_seq = read_csv(
        RESULTS
        / "puct_current_stack_clean_smoke"
        / "ar_histk4_terminal_puct_sequence_s4k8_9cell50w_800_eval_9cell100w_summary.csv"
    )
    puct_seq_search0 = read_csv(
        RESULTS
        / "puct_current_stack_clean_smoke"
        / "ar_histk4_terminal_puct_sequence_s4k8_search0_9cell50w_1000_eval_9cell100w_graphseq_summary.csv"
    )
    puct_seq_search0_freeze = read_csv(
        RESULTS
        / "puct_current_stack_clean_smoke"
        / "ar_histk4_terminal_puct_sequence_s4k8_search0_freezeseq_9cell50w_600_eval_9cell100w_graphseq_summary.csv"
    )
    puct_root_search0 = maybe_read_csv(
        RESULTS
        / "puct_current_stack_clean_smoke"
        / "rootdist_search0_9cell10w_800_eval_9cell100w"
        / "canonical_summary.csv"
    )

    rows = []

    def add(source: str, method: str, row: pd.Series, note: str) -> None:
        latency = row.get("planning_ms_per_window", row.get("latency_mean_ms", row.get("latency_ms", float("nan"))))
        if pd.isna(latency):
            latency = row.get("latency_ms", float("nan"))
        rows.append(
            {
                "source": source,
                "method": method,
                "reward_per_window": float(row["reward_per_window"] if "reward_per_window" in row else row["reward"]),
                "drop_pct_active": float(row["drop_pct_active"] if "drop_pct_active" in row else row["drop_pct"]),
                "tracked_targets": float(row["tracked_targets"] if "tracked_targets" in row else row["tracked"]),
                "mean_delay_ms": float(row["mean_delay_active"] if "mean_delay_active" in row else row["delay"]),
                "search_fraction": float(row["search_fraction"] if "search_fraction" in row else row["search"]),
                "latency_ms_per_window": float(latency),
                "windows": int(row.get("windows", row.get("n", 0))),
                "note": note,
            }
        )

    fair_map = {
        "Encode-each-step action-attention PQ": "Re-encode each action: S-band action-attention PQ",
        "Fast AR action-attention": "Fast AR sequence decoder: root encode + action-attention history",
        "MuZero latent g-loop": "MuZero-style latent g-loop 0R",
        "EDF": "EDF",
        "EST": "EST",
    }
    for old, new in fair_map.items():
        match = fair[fair["method"].eq(old)]
        if not match.empty:
            add("clean_9cell_100w_seed916", new, match.iloc[0], "Clean no forced ratios/search bias; 9 cells x 100 windows.")

    strict_map = {
        "S-only exact-trained AR PQ": "Root-only action-attention then prediction loop",
        "Direct FlatPQ": "Flat PQ direct physical head",
        "Fixed MuZero latent structured 0R": "Orthodox MuZero latent structured 0R",
    }
    for old, new in strict_map.items():
        match = strict[strict["method"].eq(old)]
        if not match.empty:
            add("strict_no_bias_9cell_100w_seed916", new, match.iloc[0], "Strict no inference shaping; 9 cells x 100 windows.")

    for planner in ["ExactEnvPUCT_s4_k8", "EDF", "EST"]:
        match = puct4[puct4["planner"].eq(planner)]
        if not match.empty:
            add(
                "online_exact_puct_4r_9cell_50w_seed916",
                "Online exact-env PUCT 4R" if planner == "ExactEnvPUCT_s4_k8" else f"{planner} on PUCT sweep",
                match.iloc[0],
                "Non-greedy rollout teacher; 4 simulations, top-8 expansion, 9 cells x 50 windows.",
            )

    puct_student = puct_seq[puct_seq["planner"].eq("LatentMuZero_greedy")]
    if not puct_student.empty:
        grouped = puct_student.mean(numeric_only=True)
        add(
            "puct_sequence_student_9cell_100w_seed916",
            "PUCT sequence-distilled AR student",
            grouped,
            "Failed transfer: PUCT sequence labels under-search; kept as diagnostic.",
        )

    corrected = puct_seq_search0[puct_seq_search0["planner"].eq("LatentMuZero_greedy")]
    if not corrected.empty:
        add(
            "corrected_puct_sequence_student_9cell_100w_seed916",
            "Corrected PUCT sequence AR student",
            corrected.mean(numeric_only=True),
            "Teacher search bias fixed; CUDA graph AR-seq decode; still underperforms clean AR/action-attention.",
        )

    corrected_freeze = puct_seq_search0_freeze[puct_seq_search0_freeze["planner"].eq("LatentMuZero_greedy")]
    if not corrected_freeze.empty:
        add(
            "corrected_puct_sequence_student_9cell_100w_seed916",
            "Corrected PUCT sequence AR student, seq-only finetune",
            corrected_freeze.mean(numeric_only=True),
            "Teacher search bias fixed and non-sequence weights frozen; still underperforms.",
        )

    if puct_root_search0 is not None:
        root_student = puct_root_search0[puct_root_search0["method"].eq("LatentMuZero_greedy")]
        if not root_student.empty:
            add(
                "corrected_puct_root_student_9cell_100w_seed916",
                "Corrected PUCT-root latent policy student",
                root_student.iloc[0],
                "PUCT root distribution distillation with teacher search bias 0; beats heuristics but trails action-attention/AR.",
            )

    return pd.DataFrame(rows)


def plot_suite(df: pd.DataFrame, out_png: Path) -> None:
    keep = [
        "S-only action-attention PQ",
        "Fast S-only sequence decoder",
        "Fixed MuZero latent structured 0R",
        "Direct FlatPQ",
        "EDF",
        "EST",
    ]
    display = {
        "S-only action-attention PQ": "S-band AA PQ",
        "Fast S-only sequence decoder": "Fast AR",
        "Fixed MuZero latent structured 0R": "MuZero 0R",
        "Direct FlatPQ": "Flat PQ",
        "EDF": "EDF",
        "EST": "EST",
    }
    win = read_csv(RESULTS / "final_sonly_requested_methods_20260707" / "canonical_windows.csv")
    win = win[win["method"].isin(keep)].copy()
    order = [m for m in keep if m in set(win["method"])]
    colors = {
        "S-only action-attention PQ": "#1f4ea8",
        "Fast S-only sequence decoder": "#6f3cc3",
        "Fixed MuZero latent structured 0R": "#159895",
        "Direct FlatPQ": "#f28e2b",
        "EDF": "#d95f02",
        "EST": "#1b9e77",
    }
    styles = {"EDF": ":", "EST": ":", "Fixed MuZero latent structured 0R": "--"}
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=False)
    metrics = [
        ("cumulative_reward", "Cumulative reward", "Reward"),
        ("window_reward", "Window reward, 10-window avg.", "Reward/window"),
        ("drop_pct_active", "Dropped / active (%)", "%"),
        ("tracked_targets", "Tracked targets", "Targets"),
        ("mean_delay_active", "Mean delay", "ms"),
        ("search_fraction", "Search ratio, 10-window avg.", "Ratio"),
    ]
    for ax, (col, title, ylabel) in zip(axs.flat, metrics):
        for method in order:
            g = win[win["method"].eq(method)].groupby("window", as_index=False)[col].mean()
            y = g[col]
            if col in {"window_reward", "search_fraction"}:
                y = y.rolling(10, min_periods=1).mean()
            if col == "cumulative_reward":
                y = win[win["method"].eq(method)].groupby(["initial", "rate", "seed"])["window_reward"].cumsum()
                tmp = win[win["method"].eq(method)].copy()
                tmp["_cum"] = y
                g = tmp.groupby("window", as_index=False)["_cum"].mean()
                y = g["_cum"]
            ax.plot(g["window"], y, label=display.get(method, method), color=colors.get(method), linestyle=styles.get(method, "-"), linewidth=2)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Window")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_puct(out_png: Path) -> None:
    win = read_csv(RESULTS / "puct_current_stack_clean_smoke" / "exact_puct_terminal_sf8_svc025_s4k8_9cell_50w.csv")
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=False)
    display = {"ExactEnvPUCT_s4_k8": "PUCT 4R", "EDF": "EDF", "EST": "EST"}
    metrics = [
        ("cumulative_reward", "Cumulative reward", "Reward"),
        ("window_reward", "Window reward, 10-window avg.", "Reward/window"),
        ("drop_pct_active", "Dropped / active (%)", "%"),
        ("tracked_targets", "Tracked targets", "Targets"),
        ("mean_delay_active", "Mean delay", "ms"),
        ("planning_ms_per_decision", "Planning latency", "ms/window"),
    ]
    for ax, (col, title, ylabel) in zip(axs.flat, metrics):
        for method in ["ExactEnvPUCT_s4_k8", "EDF", "EST"]:
            g = win[win["planner"].eq(method)].copy()
            if g.empty:
                continue
            if col == "cumulative_reward":
                g["_cum"] = g.groupby(["init", "rate", "seed"])["window_reward"].cumsum()
                s = g.groupby("window", as_index=False)["_cum"].mean()
                ycol = "_cum"
            else:
                s = g.groupby("window", as_index=False)[col].mean()
                ycol = col
                if col == "window_reward":
                    s[ycol] = s[ycol].rolling(10, min_periods=1).mean()
            ax.plot(s["window"], s[ycol], label=display.get(method, method), linewidth=2)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xlabel("Window")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows()
    summary.to_csv(OUT / "requested_final_method_summary.csv", index=False)
    plot_suite(summary, OUT / "clean_method_window_suite.png")
    plot_puct(OUT / "online_puct_teacher_window_suite.png")
    print({"summary": str(OUT / "requested_final_method_summary.csv"), "clean_plot": str(OUT / "clean_method_window_suite.png"), "puct_plot": str(OUT / "online_puct_teacher_window_suite.png")})


if __name__ == "__main__":
    main()
