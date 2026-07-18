from __future__ import annotations

import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from balanced_q_retrain import BASE_Q, make_env, make_q_planner
from final_radar_campaign import MAXT, build_env, get_obs, seedall
from repaired_campaign_tools import EDFPlanner, ESTPlanner, load_student_model
from sequence_decoder_experiment import ParallelSequenceDecoder, SequenceDirectPlanner
from strict_window_report import execute_plan_until_budget, sample_state_metrics


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "proper_200ms_window_frontier"
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


def qdistill_planner(model, threshold: float = 0.75):
    return SequenceDirectPlanner(model, threshold=threshold, mode="branch", allow_retrack=False)


def legacy_q_planner(env):
    q_model = load_student_model(str(BASE_Q), MAXT, "cpu", q_head_use_tanh=False)
    return make_q_planner(q_model, env, rollouts=16, topk=8, search_prior_scale=2.0, q_weight=1.0, sim_h=600.0)


def run_windowed(planner, name: str, init: int, rate: float, seed: int, stride_ms: float, duration_ms: float):
    env = single_env(rate)
    plan_ms = 200.0
    eng = build_env(planner, init, MAXT, seed, int(plan_ms), env)
    eng.reset(seed=seed)
    debt, elapsed, cumulative, idx = 0.0, 0.0, 0.0, 0
    rows = []
    if hasattr(planner, "warmup"):
        planner.warmup(get_obs(eng, debt), budget_ms=int(plan_ms))
    while elapsed < duration_ms and not eng.term_buf[0]:
        obs = get_obs(eng, debt)
        t0 = time.perf_counter()
        plan = planner.plan(obs, budget_ms=int(plan_ms))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        reward, spent, debt, executed, search, _ = execute_plan_until_budget(
            eng, plan, float(stride_ms), debt, name, seed, idx
        )
        cumulative += float(reward)
        elapsed += float(stride_ms)
        eq_scale = 200.0 / float(stride_ms)
        rows.append(
            dict(
                planner=name,
                config=f"200/{int(round(stride_ms))}",
                plan_ms=200.0,
                stride_ms=float(stride_ms),
                initial_targets=init,
                rate=rate,
                seed=seed,
                elapsed_ms=float(elapsed),
                window_reward=float(reward),
                reward_per_200ms_eq=float(reward) * eq_scale,
                cumulative_reward=float(cumulative),
                planning_ms=float(latency_ms),
                planning_ms_per_200ms_eq=float(latency_ms) * eq_scale,
                executed_actions=int(executed),
                spent_ms=float(spent),
                search_fraction=float(search / max(1, executed)),
                **sample_state_metrics(eng, debt),
            )
        )
        idx += 1
    eng.close()
    return rows


def run_all():
    raw_path = OUT / "proper_200ms_frontier_raw.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)

    qdistill = load_qdistill()
    rows = []
    cells = [(15, 2.0), (50, 2.0), (75, 2.0), (100, 2.0)]
    strides = [200.0, 150.0, 100.0, 75.0, 50.0, 200.0 / 6.0]
    for init, rate in cells:
        env = single_env(rate)
        for stride in strides:
            for name, planner in [
                ("QDistillBatch", qdistill_planner(qdistill, 0.75)),
                ("EDF", EDFPlanner(MAXT)),
                ("EST", ESTPlanner(MAXT)),
            ]:
                seedall(96)
                print("start", init, rate, name, f"200/{stride}", flush=True)
                rows.extend(run_windowed(planner, name, init, rate, 96, stride, duration_ms=6000.0))
                pd.DataFrame(rows).to_csv(raw_path, index=False)
                print("frontier", init, rate, name, f"200/{stride}", flush=True)
            # LegacyQ is the teacher reference; the 33ms stride is extremely
            # expensive and operationally unrealistic for the rollout teacher,
            # but we include it at shorter duration for scale sanity.
            seedall(96)
            legacy_duration = 4000.0
            print("start", init, rate, "LegacyQ_r16", f"200/{stride}", flush=True)
            rows.extend(run_windowed(legacy_q_planner(env), "LegacyQ_r16", init, rate, 96, stride, duration_ms=legacy_duration))
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            print("frontier", init, rate, "LegacyQ_r16", f"200/{stride}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(raw_path, index=False)
    return df


def summarize_and_plot(df: pd.DataFrame):
    summary = (
        df.groupby(["planner", "config", "stride_ms"])
        .agg(
            reward_per_200ms=("reward_per_200ms_eq", "mean"),
            final_cumulative=("cumulative_reward", "last"),
            drop_pct=("drop_pct_active", "mean"),
            avg_delay_ms=("mean_delay_active", "mean"),
            tracked=("tracked_targets", "mean"),
            active=("active_targets", "mean"),
            search_fraction=("search_fraction", "mean"),
            latency_ms_per_200=("planning_ms_per_200ms_eq", "mean"),
            raw_planning_ms=("planning_ms", "mean"),
            executed_actions=("executed_actions", "mean"),
        )
        .reset_index()
        .sort_values(["planner", "stride_ms"], ascending=[True, False])
    )
    summary.to_csv(OUT / "proper_200ms_frontier_summary.csv", index=False)

    # Frontier: reward vs execution stride.
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=180)
    order = ["QDistillBatch", "LegacyQ_r16", "EDF", "EST"]
    for name in order:
        sub = summary[summary["planner"] == name].sort_values("stride_ms", ascending=False)
        if sub.empty:
            continue
        ax.plot(sub["stride_ms"], sub["reward_per_200ms"], marker="o", label=name, linewidth=2 if name == "QDistillBatch" else 1.5)
        for _, r in sub.iterrows():
            ax.annotate(r["config"], (r["stride_ms"], r["reward_per_200ms"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.invert_xaxis()
    ax.set_title("200ms Planning Horizon, Partial Execution Frontier")
    ax.set_xlabel("executed stride before replanning (ms)")
    ax.set_ylabel("reward per executed 200ms")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    stride_plot = OUT / "proper_200ms_frontier_reward_by_stride.png"
    fig.savefig(stride_plot)
    plt.close(fig)

    # Reward-latency frontier.
    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=180)
    for name in order:
        sub = summary[summary["planner"] == name].sort_values("latency_ms_per_200")
        if sub.empty:
            continue
        ax.plot(sub["latency_ms_per_200"], sub["reward_per_200ms"], marker="o", label=name, linewidth=2 if name == "QDistillBatch" else 1.5)
        for _, r in sub.iterrows():
            ax.annotate(r["config"], (r["latency_ms_per_200"], r["reward_per_200ms"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_title("200ms Planning Horizon, Reward-Latency Frontier")
    ax.set_xlabel("planning latency per executed 200ms (ms, log)")
    ax.set_ylabel("reward per executed 200ms")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    latency_plot = OUT / "proper_200ms_frontier_reward_latency.png"
    fig.savefig(latency_plot)
    plt.close(fig)

    # Cumulative reward for the important configs only.
    keep_configs = ["200/200", "200/150", "200/100"]
    keep = df[df["config"].isin(keep_configs)].copy()
    piv = keep.pivot_table(
        index="elapsed_ms",
        columns=["planner", "config", "initial_targets", "rate", "seed"],
        values="cumulative_reward",
        aggfunc="last",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for name in ["QDistillBatch", "LegacyQ_r16", "EDF", "EST"]:
        for cfg in keep_configs:
            cols = [c for c in piv.columns if c[0] == name and c[1] == cfg]
            if cols:
                ax.plot(
                    piv.index / 1000.0,
                    piv[cols].mean(axis=1),
                    label=f"{name} {cfg}",
                    linewidth=2 if name == "QDistillBatch" else 1.1,
                )
    ax.set_title("Cumulative Reward, Key 200ms Window Modes")
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    cumulative_plot = OUT / "proper_200ms_frontier_cumulative.png"
    fig.savefig(cumulative_plot)
    plt.close(fig)

    for src in [OUT / "proper_200ms_frontier_summary.csv", stride_plot, latency_plot, cumulative_plot]:
        shutil.copy2(src, CLEAN / src.name)
    return summary, stride_plot, latency_plot, cumulative_plot


def main():
    df = run_all()
    summary, stride_plot, latency_plot, cumulative_plot = summarize_and_plot(df)
    cols = [
        "planner",
        "config",
        "reward_per_200ms",
        "latency_ms_per_200",
        "drop_pct",
        "avg_delay_ms",
        "search_fraction",
        "tracked",
        "active",
    ]
    print(summary[cols].to_string(index=False))
    print("OUT", OUT.resolve())
    print("PLOTS", stride_plot.resolve(), latency_plot.resolve(), cumulative_plot.resolve())
    print("CLEAN", CLEAN.resolve())


if __name__ == "__main__":
    main()
