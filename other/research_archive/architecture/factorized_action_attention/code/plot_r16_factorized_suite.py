from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from alphazero_benchmark import load_state_into
from alphazero_orthodox import base_exact_args
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, env_cfg_for, load_model, run_fixed, run_snapshot_exact_episode


ROOT = Path(r"C:\Users\yousi\Downloads\Model1 1")
RES = ROOT / "CreateValid1" / "results" / "pq1_alphazero_r60_rates136"
OUT = RES / "r16_factorized_plot_suite"
CKPT = Path(r"C:\Users\yousi\Downloads\radar_outputs\exact_train_qw02_seeded_more\exact_mutual_latest.pt")
FORCEDCF_STATE = RES / "typebias_forcedcf_h8_finetune_state.pt"
DISTILLED_STATE = RES / "r16h4_behavior_distill_state.pt"


def exact_args(rollouts: int, horizon_windows: int, windows: int = 20, select_mode: str = "visits"):
    return base_exact_args(
        SimpleNamespace(
            ckpt=str(CKPT),
            device="cpu",
            head_arch="branch_context",
            windows=int(windows),
            max_targets_per_episode=64,
            rollouts=int(rollouts),
            c_puct=1.25,
            expand_top_k=48,
            horizon_windows=int(horizon_windows),
            prior_uniform_mix=0.03,
            root_dirichlet_alpha=0.3,
            root_dirichlet_frac=0.0,
            leaf_value_mix=0.5,
            rollout_policy="branch",
            branch_rollout_threshold=0.65,
            seed_rollout_policies="",
            skip_default_rollout_seed=False,
            prior_mode="physical_flat",
            sensor_action_mode="implicit",
            disable_x_search=False,
            canonical_search_only=False,
            search_alg="puct",
            plan_mode="atomic",
            window_extract="tree_fill",
            gumbel_scale=0.0,
            max_num_considered_actions=16,
            mctx_value_scale=0.1,
            mctx_maxvisit_init=50.0,
            select_mode=str(select_mode),
            load_gated_prior_threshold=80,
            visit_unvisited_first=True,
            head_mode="pq",
            q_utility_weight=0.0,
            q_utility_normalize=False,
            puct_q_transform="raw",
            prior_q_beta=0.0,
            prior_search_bias=0.0,
            adaptive_search_bias=0.0,
            adaptive_search_target_load=0.75,
            q_scale=100.0,
            self_play_sample_tau=0.0,
            gamma=0.99,
            env_mode="mcts_sched_v1",
            use_arrival_feature=True,
            use_grid_feature=True,
            single_sensor=False,
            zero_action_rewards=False,
            track_loss_penalty=8.0,
            target_service_weight=10.0,
            target_service_horizon_ms=3000.0,
            sector_staleness_weight=0.01,
            search_frame_overdue_weight=0.20,
            search_frame_drop_penalty=8.0,
            enable_x_band=True,
        )
    )


def load_model_for(args, state_path: Path = FORCEDCF_STATE):
    model = load_model(args).to(torch.device("cpu"))
    load_state_into(model, str(state_path), torch.device("cpu"))
    model.eval()
    return model


def aggregate_suite() -> pd.DataFrame:
    frames = []
    for path, name in [
        (RES / "typebias_forcedcf_r16_h4_grid_18_eval.csv", "Forced-CF r16 h4"),
        (RES / "r16_distill_load_gated_t80_grid18_eval.csv", "Distilled load-gated"),
        (RES / "cal_pos2_prior_grid18_eval.csv", "Calibrated prior only"),
        (RES / "typebias_forcedcf_h8_grid_18_eval.csv", "Forced-CF r1"),
        (RES / "r16h4_behavior_distill_r1_grid18_eval.csv", "Distilled r1 visits"),
        (RES / "typebias_m05_grid_18_eval.csv", "Learned type-bias"),
        (RES / "baseline_grid_18_eval.csv", "Current best"),
        (RES / "bias_m05_vs_heur_grid_18_eval.csv", None),
    ]:
        df = pd.read_csv(path)
        if name:
            df = df.copy()
            df["method"] = name
        else:
            df = df[df["method"].isin(["EDF", "EST"])].copy()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["method", "initial", "rate", "seed"], keep="last")


def plot_bars(df: pd.DataFrame) -> None:
    summary = df.groupby("method", as_index=False).agg(
        reward=("reward", "mean"),
        total_reward=("total_reward", "mean"),
        search=("search", "mean"),
        tracked=("mean_tracked_targets", "mean"),
        drop=("mean_drop_pct_active", "mean"),
        delay=("mean_delay_active", "mean"),
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    order = ["Distilled load-gated", "Forced-CF r16 h4", "Forced-CF r1", "Distilled r1 visits", "Calibrated prior only", "Learned type-bias", "Current best", "EDF", "EST"]
    summary["method"] = pd.Categorical(summary["method"], order, ordered=True)
    summary = summary.sort_values("method")
    colors = ["#155f47", "#2f7d59", "#4f83b6", "#6a9c89", "#b35c62", "#6b62a8", "#8d8d8d", "#b17a41", "#b95f5f"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [
        ("reward", "Reward / window"),
        ("total_reward", "Episode total"),
        ("search", "Search ratio"),
        ("tracked", "Tracked targets"),
        ("drop", "Drop % active"),
        ("delay", "Mean delay"),
    ]
    for ax, (col, title) in zip(axes.ravel(), metrics):
        ax.bar(summary["method"].astype(str), summary[col], color=colors[: len(summary)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "summary_metrics.png", dpi=180)
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index=["initial", "rate", "seed"], columns="method", values="reward")
    for lhs, rhs, stem, title in [
        ("Forced-CF r16 h4", "Forced-CF r1", "reward_delta_r16_minus_r1", "Reward delta: r16 h4 minus r1"),
        ("Distilled load-gated", "Forced-CF r1", "reward_delta_gated_minus_r1", "Reward delta: load-gated distill minus r1"),
        ("Distilled load-gated", "Forced-CF r16 h4", "reward_delta_gated_minus_r16", "Reward delta: load-gated distill minus r16 h4"),
        ("Forced-CF r16 h4", "Current best", "reward_delta_r16_minus_current", "Reward delta: r16 h4 minus current best"),
    ]:
        cells = (pivot[lhs] - pivot[rhs]).reset_index().pivot_table(index="initial", columns="rate", values=0, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(7, 4.8))
        vmax = float(np.nanmax(np.abs(cells.values))) if np.isfinite(cells.values).any() else 1.0
        im = ax.imshow(cells.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cells.columns)), [str(c) for c in cells.columns])
        ax.set_yticks(range(len(cells.index)), [str(i) for i in cells.index])
        ax.set_xlabel("Arrival rate")
        ax.set_ylabel("Initial targets")
        ax.set_title(title)
        for i in range(cells.shape[0]):
            for j in range(cells.shape[1]):
                ax.text(j, i, f"{cells.values[i, j]:+.2f}", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.tight_layout()
        fig.savefig(OUT / f"{stem}.png", dpi=180)
        plt.close(fig)


def run_traces() -> pd.DataFrame:
    cases = [(60, 0.0, 931), (60, 20.0, 936), (100, 20.0, 931), (100, 20.0, 938)]
    args_r1 = exact_args(1, 2, 20)
    args_gated = exact_args(1, 2, 20, "load_gated_prior")
    args_r16 = exact_args(16, 4, 20)
    model_r1 = load_model_for(args_r1)
    model_gated = load_model_for(args_gated, DISTILLED_STATE)
    model_r16 = load_model_for(args_r16)
    rows = []
    for initial, rate, seed in cases:
        for name, model, args in [
            ("Distilled load-gated", model_gated, args_gated),
            ("Forced-CF r16 h4", model_r16, args_r16),
            ("Forced-CF r1", model_r1, args_r1),
        ]:
            df, _ = run_snapshot_exact_episode(model, args, int(initial), float(rate), int(seed), train=False)
            df["method"] = name
            df["initial"] = int(initial)
            df["rate"] = float(rate)
            df["seed"] = int(seed)
            rows.append(df)
        env_cfg = env_cfg_for(float(rate), args_r1)
        for name, planner in [("EDF", EDFPlanner(MAXT)), ("EST", ESTPlanner(MAXT))]:
            df, _ = run_fixed(planner, name, int(initial), MAXT, int(seed), 20, 200, env_cfg)
            df["method"] = name
            df["initial"] = int(initial)
            df["rate"] = float(rate)
            df["seed"] = int(seed)
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT / "representative_window_traces.csv", index=False)
    return out


def plot_traces(traces: pd.DataFrame) -> None:
    for (initial, rate, seed), gcase in traces.groupby(["initial", "rate", "seed"]):
        fig, axes = plt.subplots(5, 1, figsize=(10, 12.5), sharex=True)
        for method, g in gcase.groupby("method"):
            g = g.sort_values("window")
            axes[0].plot(g["window"], g["cumulative_reward"], label=method, linewidth=2)
            axes[1].plot(g["window"], g["window_reward"], label=method, linewidth=2)
            axes[2].plot(g["window"], g["search_fraction"], label=method, linewidth=2)
            axes[3].plot(g["window"], g["tracked_targets"], label=method, linewidth=2)
            axes[4].plot(g["window"], g["drop_pct_active"], label=method, linewidth=2)
        axes[0].set_title(f"initial={initial}, rate={rate:g}, seed={seed}")
        for ax, ylabel in zip(axes, ["Cumulative reward", "Window reward", "Search ratio", "Tracked", "Drop % active"]):
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Window")
        axes[0].legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"trace_i{initial}_r{rate:g}_s{seed}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = aggregate_suite()
    df.to_csv(OUT / "aggregate_eval.csv", index=False)
    plot_bars(df)
    plot_heatmaps(df)
    traces = run_traces()
    plot_traces(traces)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
