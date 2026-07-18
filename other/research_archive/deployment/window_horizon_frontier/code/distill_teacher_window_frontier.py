from __future__ import annotations

import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from balanced_q_retrain import BASE_Q, make_env, make_q_planner
from final_radar_campaign import MAXT, build_env, get_obs, run_fixed, seedall, summarize_window_df
from repaired_campaign_tools import EDFPlanner, ESTPlanner, load_student_model
from sequence_decoder_experiment import ParallelSequenceDecoder, SequenceDirectPlanner
from strict_window_report import execute_plan_until_budget, sample_state_metrics


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "distill_teacher_window_frontier"
OUT.mkdir(parents=True, exist_ok=True)
CLEAN = Path(r"C:\Users\yousi\Downloads\radar_outputs")
CLEAN.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


def single_env(rate: float):
    env = make_env(rate, refresh=1, strict=False)
    env["enable_x_band"] = 0
    return env


def load_qdistill():
    model = ParallelSequenceDecoder(seq_len=32).to(DEVICE)
    model.load_state_dict(
        torch.load(
            ROOT / "CreateValid1/results/single_sensor_qdistill_batch/qdistill_batch_decoder.pt",
            map_location=DEVICE,
        )
    )
    return model.eval()


def make_distill(model, threshold: float = 0.75):
    return SequenceDirectPlanner(model, threshold=threshold, mode="branch", allow_retrack=False)


def make_teacher(env):
    q_model = load_student_model(str(BASE_Q), MAXT, "cpu", q_head_use_tanh=False)
    return make_q_planner(q_model, env, rollouts=16, topk=8, search_prior_scale=2.0, q_weight=1.0, sim_h=600.0)


def fixed_table(model):
    raw_path = OUT / "fixed_teacher_student_raw.csv"
    win_path = OUT / "fixed_teacher_student_windows.csv"
    if raw_path.exists() and win_path.exists():
        return pd.read_csv(raw_path), pd.read_csv(win_path)

    rows, wins = [], []
    cells = [(15, 0.0), (15, 2.0), (50, 0.0), (50, 2.0), (75, 2.0), (100, 2.0)]
    for init, rate in cells:
        env = single_env(rate)
        methods = {
            "QDistillBatch_t0.75": lambda model=model: make_distill(model, 0.75),
            "LegacyQ_Teacher_r16": lambda env=env: make_teacher(env),
            "EDF": lambda: EDFPlanner(MAXT),
            "EST": lambda: ESTPlanner(MAXT),
        }
        for name, factory in methods.items():
            seedall(91)
            t0 = time.perf_counter()
            w, _ = run_fixed(factory(), name, init, MAXT, 91, 60, 200, env)
            s = summarize_window_df(w, "fixed")
            s.update(planner=name, initial_targets=init, rate=rate, seed=91, wall_s=time.perf_counter() - t0)
            rows.append(s)
            ww = w.copy()
            ww["planner"] = name
            ww["initial_targets"] = init
            ww["rate"] = rate
            ww["seed"] = 91
            wins.append(ww)
            print(
                "fixed",
                init,
                rate,
                name,
                round(s["reward_per_200ms_eq"], 3),
                round(s["mean_drop_pct_active"], 2),
                round(s["mean_delay_active"], 1),
                round(s["search_fraction"], 2),
                round(s["planning_ms_per_200ms_eq"], 2),
                flush=True,
            )
    raw = pd.DataFrame(rows)
    win = pd.concat(wins, ignore_index=True)
    raw.to_csv(raw_path, index=False)
    win.to_csv(win_path, index=False)
    return raw, win


def run_windowed(planner, name: str, init: int, rate: float, seed: int, plan_ms: float, stride_ms: float, duration_ms: float):
    env = single_env(rate)
    eng = build_env(planner, init, MAXT, seed, int(round(plan_ms)), env)
    eng.reset(seed=seed)
    debt, elapsed, cumulative, idx = 0.0, 0.0, 0.0, 0
    rows = []
    if hasattr(planner, "warmup"):
        planner.warmup(get_obs(eng, debt), budget_ms=int(round(plan_ms)))
    while elapsed < duration_ms and not eng.term_buf[0]:
        obs = get_obs(eng, debt)
        t0 = time.perf_counter()
        plan = planner.plan(obs, budget_ms=int(round(plan_ms)))
        lat = (time.perf_counter() - t0) * 1000.0
        reward, spent, debt, executed, search, _ = execute_plan_until_budget(
            eng, plan, float(stride_ms), debt, name, seed, idx
        )
        cumulative += float(reward)
        elapsed += float(stride_ms)
        scale = 200.0 / float(stride_ms)
        rows.append(
            dict(
                planner=name,
                config=f"plan{int(round(plan_ms))}_exec{int(round(stride_ms))}",
                plan_ms=float(plan_ms),
                stride_ms=float(stride_ms),
                initial_targets=init,
                rate=rate,
                seed=seed,
                elapsed_ms=elapsed,
                window_reward=float(reward),
                reward_per_200ms_eq=float(reward) * scale,
                cumulative_reward=float(cumulative),
                planning_ms=float(lat),
                planning_ms_per_200ms_eq=float(lat) * scale,
                executed_actions=int(executed),
                spent_ms=float(spent),
                search_fraction=float(search / max(1, executed)),
                **sample_state_metrics(eng, debt),
            )
        )
        idx += 1
    eng.close()
    return rows


def frontier(model):
    raw_path = OUT / "window_frontier_raw.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)

    rows = []
    # plan_ms is the horizon shown to the planner; stride_ms is how much of the
    # generated plan is executed before replanning.
    configs = [
        (100, 100),
        (150, 150),
        (200, 200),
        (200, 100),
        (200, 50),
        (200, 200.0 / 6.0),
        (300, 150),
        (400, 200),
        (400, 100),
        (600, 200),
        (800, 200),
        (1200, 200),
    ]
    cells = [(15, 2.0), (50, 2.0), (75, 2.0), (100, 2.0)]
    for init, rate in cells:
        for plan_ms, stride_ms in configs:
            for threshold in [0.50, 0.65, 0.75]:
                name = f"QDistill_t{threshold:.2f}"
                rows.extend(
                    run_windowed(
                        make_distill(model, threshold),
                        name,
                        init,
                        rate,
                        92,
                        plan_ms,
                        stride_ms,
                        duration_ms=20000.0,
                    )
                )
                print("frontier", init, rate, name, plan_ms, stride_ms, flush=True)
            # Baseline heuristics on every window setting to ensure the frontier
            # comparison is not hiding an execution-mode artifact.
            for hname, planner in [("EDF", EDFPlanner(MAXT)), ("EST", ESTPlanner(MAXT))]:
                rows.extend(run_windowed(planner, hname, init, rate, 92, plan_ms, stride_ms, duration_ms=20000.0))
                print("frontier", init, rate, hname, plan_ms, stride_ms, flush=True)
        # Teacher is expensive; keep it on the key fixed and best-practice
        # sliding settings for the teacher/student comparison.
        env = single_env(rate)
        for plan_ms, stride_ms in [(200, 200), (200, 100), (400, 200)]:
            rows.extend(
                run_windowed(
                    make_teacher(env),
                    "LegacyQ_Teacher_r16",
                    init,
                    rate,
                    92,
                    plan_ms,
                    stride_ms,
                    duration_ms=10000.0,
                )
            )
            print("frontier", init, rate, "LegacyQ_Teacher_r16", plan_ms, stride_ms, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)
    return df


def plot_outputs(fixed_raw: pd.DataFrame, fixed_win: pd.DataFrame, front: pd.DataFrame):
    fixed_summary = (
        fixed_raw.groupby("planner")
        .agg(
            reward_per_200ms=("reward_per_200ms_eq", "mean"),
            final_cumulative=("final_cumulative_reward", "mean"),
            drop_pct=("mean_drop_pct_active", "mean"),
            avg_delay_ms=("mean_delay_active", "mean"),
            tracked=("mean_tracked_targets", "mean"),
            active=("mean_active_targets", "mean"),
            search_fraction=("search_fraction", "mean"),
            latency_ms_per_200=("planning_ms_per_200ms_eq", "mean"),
        )
        .reset_index()
        .sort_values("reward_per_200ms", ascending=False)
    )
    fixed_summary.to_csv(OUT / "fixed_teacher_student_summary.csv", index=False)

    fpiv = fixed_win.pivot_table(
        index="elapsed_ms",
        columns=["planner", "initial_targets", "rate", "seed"],
        values="cumulative_reward",
        aggfunc="last",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for name in fixed_summary["planner"]:
        cols = [c for c in fpiv.columns if c[0] == name]
        if cols:
            ax.plot(fpiv.index / 1000.0, fpiv[cols].mean(axis=1), label=name, linewidth=2 if "QDistill" in name else 1.5)
    ax.set_title("Distilled Batch Model vs Original Teacher")
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    cumulative_plot = OUT / "distill_teacher_cumulative.png"
    fig.savefig(cumulative_plot)
    plt.close(fig)

    frontier_summary = (
        front.groupby(["planner", "config", "plan_ms", "stride_ms"])
        .agg(
            reward_per_200ms=("reward_per_200ms_eq", "mean"),
            final_cumulative=("cumulative_reward", "last"),
            drop_pct=("drop_pct_active", "mean"),
            avg_delay_ms=("mean_delay_active", "mean"),
            tracked=("tracked_targets", "mean"),
            active=("active_targets", "mean"),
            search_fraction=("search_fraction", "mean"),
            latency_ms_per_200=("planning_ms_per_200ms_eq", "mean"),
            executed_actions=("executed_actions", "mean"),
            spent_ms=("spent_ms", "mean"),
        )
        .reset_index()
        .sort_values("reward_per_200ms", ascending=False)
    )
    frontier_summary.to_csv(OUT / "window_frontier_summary.csv", index=False)

    # Compact student-only frontier plot.
    stu = frontier_summary[frontier_summary["planner"].str.startswith("QDistill")].copy()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for name, sub in stu.groupby("planner"):
        sub = sub.sort_values("latency_ms_per_200")
        ax.plot(sub["latency_ms_per_200"], sub["reward_per_200ms"], marker="o", label=name)
        for _, r in sub.iterrows():
            if r["config"] in {"plan200_exec200", "plan200_exec100", "plan200_exec50", "plan400_exec200", "plan1200_exec200"}:
                ax.annotate(r["config"], (r["latency_ms_per_200"], r["reward_per_200ms"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_title("Window/Sliding Frontier: Distilled Batch Model")
    ax.set_xlabel("planning latency per executed 200ms (ms, log)")
    ax.set_ylabel("reward per executed 200ms")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    frontier_plot = OUT / "window_frontier_reward_latency.png"
    fig.savefig(frontier_plot)
    plt.close(fig)

    # Per-window cumulative comparison for the best frontier configs.
    best_configs = (
        stu.groupby("config")["reward_per_200ms"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    keep = front[(front["planner"].eq("QDistill_t0.75") & front["config"].isin(["plan200_exec200"] + best_configs)) | front["planner"].isin(["EDF", "EST"])].copy()
    keep = keep[keep["config"].isin(["plan200_exec200"] + best_configs)]
    piv = keep.pivot_table(index="elapsed_ms", columns=["planner", "config", "initial_targets", "rate", "seed"], values="cumulative_reward", aggfunc="last").sort_index()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for (planner, config), _ in keep.groupby(["planner", "config"]):
        cols = [c for c in piv.columns if c[0] == planner and c[1] == config]
        if cols:
            ax.plot(piv.index / 1000.0, piv[cols].mean(axis=1), label=f"{planner} {config}", linewidth=2 if planner.startswith("QDistill") else 1.2)
    ax.set_title("Best Window Settings: Cumulative Reward")
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    best_cum_plot = OUT / "best_window_cumulative.png"
    fig.savefig(best_cum_plot)
    plt.close(fig)

    for src in [
        OUT / "fixed_teacher_student_summary.csv",
        OUT / "window_frontier_summary.csv",
        cumulative_plot,
        frontier_plot,
        best_cum_plot,
    ]:
        shutil.copy2(src, CLEAN / src.name)
    return fixed_summary, frontier_summary, cumulative_plot, frontier_plot, best_cum_plot


def main():
    model = load_qdistill()
    fixed_raw, fixed_win = fixed_table(model)
    front = frontier(model)
    fixed_summary, frontier_summary, cumulative_plot, frontier_plot, best_cum_plot = plot_outputs(fixed_raw, fixed_win, front)
    print("\nFIXED SUMMARY")
    print(fixed_summary.to_string(index=False))
    print("\nTOP FRONTIER")
    cols = ["planner", "config", "reward_per_200ms", "latency_ms_per_200", "drop_pct", "avg_delay_ms", "search_fraction"]
    print(frontier_summary[cols].head(25).to_string(index=False))
    print("OUT", OUT.resolve())
    print("CLEAN", CLEAN.resolve())
    print("PLOTS", cumulative_plot.resolve(), frontier_plot.resolve(), best_cum_plot.resolve())


if __name__ == "__main__":
    main()
