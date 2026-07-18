from __future__ import annotations

import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from balanced_q_retrain import BASE_Q, make_env, make_q_planner
from final_radar_campaign import MAXT, build_env, get_obs, run_fixed, seedall, summarize_window_df
from mutual_foundation import MutualRadarMCTSPlanner
from refresh_method_suite import load_mutual, make_legacy_policy
from repaired_campaign_tools import EDFPlanner, ESTPlanner, load_student_model
from sequence_decoder_experiment import ParallelSequenceDecoder, SequenceDirectPlanner
from strict_window_report import execute_plan_until_budget, sample_state_metrics


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CreateValid1" / "results" / "clean_qdistill_pvq_compare"
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
    model.load_state_dict(torch.load(ROOT / "CreateValid1/results/single_sensor_qdistill_batch/qdistill_batch_decoder.pt", map_location=DEVICE))
    return model.eval()


def tuned_q(env):
    q_model = load_student_model(str(BASE_Q), MAXT, "cpu", q_head_use_tanh=False)
    return make_q_planner(q_model, env, rollouts=16, topk=8, search_prior_scale=2.0, q_weight=1.0, sim_h=600.0)


def mutual_pvq(env, rollouts: int = 16):
    model_dir = ROOT / "CreateValid1/results/mutual_improvement/refresh_pvq_factorized"
    model = load_mutual(model_dir)
    return MutualRadarMCTSPlanner(
        model,
        env,
        rollouts=rollouts,
        c_puct=1.25,
        expand_top_k=12,
        prior_mode="factorized",
        use_q_head=True,
        q_utility_weight=1.0,
        q_prior_weight=0.75,
        use_value_head=True,
        leaf_value_mix=0.35,
        q_scale=100.0,
    )


def run_sliding(planner, name: str, init: int, rate: float, config: str, plan_ms: float, stride_ms: float, total_ms: float = 10000.0, seed: int = 84):
    env = single_env(rate)
    seedall(seed)
    eng = build_env(planner, init, MAXT, seed, int(plan_ms), env)
    eng.reset(seed=seed)
    debt, elapsed, cum, idx = 0.0, 0.0, 0.0, 0
    rows = []
    if hasattr(planner, "warmup"):
        planner.warmup(get_obs(eng, debt), budget_ms=int(plan_ms))
    while elapsed < total_ms and not eng.term_buf[0]:
        obs = get_obs(eng, debt)
        t0 = time.perf_counter()
        plan = planner.plan(obs, budget_ms=int(plan_ms))
        lat = (time.perf_counter() - t0) * 1000.0
        rew, spent, debt, executed, search, _ = execute_plan_until_budget(eng, plan, float(stride_ms), debt, name, seed, idx)
        cum += rew
        elapsed += float(stride_ms)
        rows.append(dict(
            planner=name,
            config=config,
            initial_targets=init,
            rate=rate,
            seed=seed,
            elapsed_ms=elapsed,
            window_reward=rew,
            reward_per_200ms_eq=rew * (200.0 / float(stride_ms)),
            cumulative_reward=cum,
            planning_ms_per_200ms_eq=lat * (200.0 / float(stride_ms)),
            executed_actions=executed,
            search_fraction=search / max(1, executed),
            **sample_state_metrics(eng, debt),
        ))
        idx += 1
    eng.close()
    return pd.DataFrame(rows)


def evaluate():
    raw_path = OUT / "clean_compare_raw.csv"
    win_path = OUT / "clean_compare_windows.csv"
    if raw_path.exists() and win_path.exists():
        return pd.read_csv(raw_path), pd.read_csv(win_path)

    qd = load_qdistill()
    rows, wins = [], []
    cells = [(15, 0.0), (15, 2.0), (50, 0.0), (50, 2.0), (75, 2.0), (100, 2.0)]
    for init, rate in cells:
        env = single_env(rate)
        fixed_methods = {
            "QDistill_fixed_t0.75": lambda: SequenceDirectPlanner(qd, threshold=0.75, mode="branch", allow_retrack=False),
            "LegacyQ_tuned_r16": lambda env=env: tuned_q(env),
            "MutualPVQ_r16": lambda env=env: mutual_pvq(env, 16),
            "LegacyPolicy_r8": lambda env=env: make_legacy_policy(env, 8, 16),
            "EDF": lambda: EDFPlanner(MAXT),
            "EST": lambda: ESTPlanner(MAXT),
        }
        for name, factory in fixed_methods.items():
            seedall(84)
            t0 = time.perf_counter()
            w, _ = run_fixed(factory(), name, init, MAXT, 84, 50, 200, env)
            s = summarize_window_df(w, "fixed")
            s.update(planner=name, config="fixed_200_200", initial_targets=init, rate=rate, seed=84, wall_s=time.perf_counter() - t0)
            rows.append(s)
            ww = w.copy()
            ww["planner"] = name
            ww["config"] = "fixed_200_200"
            ww["initial_targets"] = init
            ww["rate"] = rate
            ww["seed"] = 84
            ww["reward_per_200ms_eq"] = ww["window_reward"]
            wins.append(ww)
            print("fixed", init, rate, name, round(s["reward_per_200ms_eq"], 3), round(s["planning_ms_per_200ms_eq"], 2), flush=True)

        # Only keep one sliding variant: best previous Q-distill choice.
        name = "QDistill_slide100_t0.50"
        planner = SequenceDirectPlanner(qd, threshold=0.50, mode="branch", allow_retrack=False)
        w = run_sliding(planner, name, init, rate, "slide_200_100", 200, 100, total_ms=10000.0, seed=84)
        s = {
            "total_reward": float(w["window_reward"].sum()),
            "reward_per_200ms_eq": float(w["reward_per_200ms_eq"].mean()),
            "mean_active_targets": float(w["active_targets"].mean()),
            "mean_tracked_targets": float(w["tracked_targets"].mean()),
            "mean_drop_pct_active": float(w["drop_pct_active"].mean()),
            "mean_delay_active": float(w["mean_delay_active"].mean()),
            "search_fraction": float(w["search_fraction"].mean()),
            "planning_ms_total": float(w["planning_ms_per_200ms_eq"].sum()),
            "planning_ms_per_decision": float(w["planning_ms_per_200ms_eq"].mean()),
            "planning_ms_per_200ms_eq": float(w["planning_ms_per_200ms_eq"].mean()),
            "steps_or_windows": int(len(w)),
            "final_active_targets": float(w["active_targets"].iloc[-1]),
            "final_tracked_targets": float(w["tracked_targets"].iloc[-1]),
            "final_cumulative_reward": float(w["cumulative_reward"].iloc[-1]),
            "planner": name,
            "config": "slide_200_100",
            "initial_targets": init,
            "rate": rate,
            "seed": 84,
        }
        rows.append(s)
        wins.append(w)
        print("slide", init, rate, name, round(s["reward_per_200ms_eq"], 3), round(s["planning_ms_per_200ms_eq"], 2), flush=True)

    raw = pd.DataFrame(rows)
    win = pd.concat(wins, ignore_index=True)
    raw.to_csv(raw_path, index=False)
    win.to_csv(win_path, index=False)
    return raw, win


def plot(raw: pd.DataFrame, win: pd.DataFrame):
    summary = raw.groupby(["planner", "config"]).agg(
        reward_per_200ms=("reward_per_200ms_eq", "mean"),
        final_cumulative=("final_cumulative_reward", "mean"),
        drop_pct=("mean_drop_pct_active", "mean"),
        avg_delay_ms=("mean_delay_active", "mean"),
        tracked=("mean_tracked_targets", "mean"),
        active=("mean_active_targets", "mean"),
        search_fraction=("search_fraction", "mean"),
        latency_ms_per_200=("planning_ms_per_200ms_eq", "mean"),
    ).reset_index().sort_values("reward_per_200ms", ascending=False)
    summary.to_csv(OUT / "clean_compare_summary.csv", index=False)

    piv = win.pivot_table(index="elapsed_ms", columns=["planner", "config", "initial_targets", "rate", "seed"], values="cumulative_reward", aggfunc="last").sort_index()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    for _, r in summary.iterrows():
        name, cfg = r["planner"], r["config"]
        cols = [c for c in piv.columns if c[0] == name and c[1] == cfg]
        if cols:
            label = name if cfg == "fixed_200_200" else f"{name} ({cfg})"
            ax.plot(piv.index / 1000.0, piv[cols].mean(axis=1), label=label, linewidth=2 if name.startswith("QDistill") else 1.5)
    ax.set_title("Clean Single-Sensor Comparison: QDistill vs PVQ vs Legacy")
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("mean cumulative reward")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = OUT / "clean_compare_cumulative.png"
    fig.savefig(p)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=180)
    ax.scatter(summary["latency_ms_per_200"], summary["reward_per_200ms"], s=80)
    for _, r in summary.iterrows():
        ax.annotate(r["planner"].replace("_", " "), (r["latency_ms_per_200"], r["reward_per_200ms"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("latency per 200ms executed (ms, log)")
    ax.set_ylabel("reward / 200ms")
    ax.set_title("Reward-Latency Comparison")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pp = OUT / "clean_compare_reward_latency.png"
    fig.savefig(pp)
    plt.close(fig)

    for src in [p, pp, OUT / "clean_compare_summary.csv"]:
        shutil.copy2(src, CLEAN / src.name)
    return summary, p, pp


def main():
    raw, win = evaluate()
    summary, p, pp = plot(raw, win)
    print(summary.to_string(index=False))
    print("OUT", OUT.resolve())
    print("PLOT", p.resolve())
    print("LATENCY", pp.resolve())
    print("CLEAN", (CLEAN / p.name).resolve())


if __name__ == "__main__":
    main()
