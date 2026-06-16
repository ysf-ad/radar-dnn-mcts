from __future__ import annotations

import time
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_vectorized128_structured import StructuredVectorized128
from eval_vectorized128_vs_heuristics import VectorizedCandidate128Planner
from exact_env_mutual import SnapshotSimulator, _DummyPlanner
from final_radar_campaign import build_env, get_obs, summarize_window_df
from python_radar_env import PyRadarState, score_plans_vectorized
from repaired_campaign_tools import EDFPlanner, ESTPlanner, env_preset_cfg
from strict_window_report import execute_plan_until_budget, sample_state_metrics
from pufferlib.ocean.radarxs import binding


OUT = Path("results/exact_rescore128")
OUT.mkdir(parents=True, exist_ok=True)


class ExactRescore128:
    """Vectorized candidate pruning + exact C snapshot rescoring."""

    def __init__(
        self,
        env_cfg: dict,
        max_trackers: int = 100,
        n_plans: int = 128,
        slots: int = 96,
        top_k: int = 16,
        score_horizon_ms: float = 1200.0,
        generator: str = "structured",
        seed: int = 1234,
    ):
        self.env_cfg = dict(env_cfg)
        self.max_trackers = int(max_trackers)
        self.n_plans = int(n_plans)
        self.slots = int(slots)
        self.top_k = int(top_k)
        self.score_horizon_ms = float(score_horizon_ms)
        if generator == "structured":
            self.gen = StructuredVectorized128(max_trackers=max_trackers, n_plans=n_plans, slots=slots, seed=seed, env_cfg=env_cfg)
        else:
            self.gen = VectorizedCandidate128Planner(max_trackers=max_trackers, n_plans=n_plans, slots=slots, seed=seed, env_cfg=env_cfg)
        self.edf = EDFPlanner(max_trackers=max_trackers)
        self.est = ESTPlanner(max_trackers=max_trackers)

    def candidates(self, obs) -> np.ndarray:
        c = self.gen._make_candidates(obs)
        # Force exact heuristic baselines into the pool.
        edf = self.gen._repeat_to_slots(self.edf.plan(obs, self.score_horizon_ms))
        est = self.gen._repeat_to_slots(self.est.plan(obs, self.score_horizon_ms))
        c[-2, :] = np.asarray(edf[: self.slots], dtype=np.int32)
        c[-1, :] = np.asarray(est[: self.slots], dtype=np.int32)
        return c

    def choose(self, eng, debt_ms: float, obs) -> tuple[list[int], dict]:
        candidates = self.candidates(obs)
        t0 = time.perf_counter()
        approx = score_plans_vectorized(PyRadarState.from_obs(obs), candidates, self.gen.cfg, budget_ms=200.0)
        # Keep both vector top-K and heuristic-injected tail candidates.
        top = np.argsort(approx)[-self.top_k :].astype(int).tolist()
        top.extend([self.n_plans - 2, self.n_plans - 1])
        top = sorted(set(i for i in top if 0 <= i < len(candidates)))
        sim = SnapshotSimulator(eng, debt_ms)
        exact_scores = []
        for idx in top:
            score, elapsed = exact_score_plan(sim, candidates[idx].tolist(), self.score_horizon_ms)
            exact_scores.append((idx, score, elapsed))
        best_idx, best_score, best_elapsed = max(exact_scores, key=lambda x: x[1])
        plan_ms = (time.perf_counter() - t0) * 1000.0
        meta = {
            "planning_ms": plan_ms,
            "candidate_count": int(len(candidates)),
            "exact_rescored": int(len(top)),
            "exact_score": float(best_score),
            "exact_score_elapsed_ms": float(best_elapsed),
            "vector_score": float(approx[best_idx]),
            "best_candidate_idx": int(best_idx),
        }
        # Snapshot scoring leaves the live C env at the last scored prefix.
        # Restore the real root before the caller executes the chosen plan.
        binding.vec_restore(eng.env, sim.root)
        return candidates[best_idx].tolist(), meta


def exact_score_plan(sim: SnapshotSimulator, plan: list[int], horizon_ms: float) -> tuple[float, float]:
    prev = sim.replay(())
    elapsed = 0.0
    prefix: list[int] = []
    score = 0.0
    for action in plan:
        if elapsed >= horizon_ms:
            break
        prefix.append(int(action))
        st = sim.replay(prefix)
        dr = float(st.reward - prev.reward)
        dt = float(st.dt_ms - prev.dt_ms)
        if dt <= 0.0 or st.terminal:
            break
        score += dr
        elapsed += dt
        prev = st
    return score, elapsed


def run_exact_rescore(planner: ExactRescore128, name: str, seed: int, windows_n: int, env_cfg: dict):
    eng = build_env(_DummyPlanner(), 50, 100, seed, 200, env_cfg)
    eng.reset(seed=seed)
    debt = 0.0
    cumulative = 0.0
    rows = []
    actions = []
    try:
        for window in range(windows_n):
            obs = get_obs(eng, debt)
            plan, meta = planner.choose(eng, debt, obs)
            reward, spent, debt, executed, search_actions, arows = execute_plan_until_budget(
                eng, plan, 200.0, debt, name, seed, window
            )
            cumulative += reward
            state = sample_state_metrics(eng, debt)
            rows.append(
                {
                    "planner": name,
                    "seed": seed,
                    "window": window,
                    "window_reward": float(reward),
                    "cumulative_reward": float(cumulative),
                    "search_fraction": float(search_actions / max(1, executed)),
                    "planning_ms_per_decision": float(meta["planning_ms"]),
                    "planning_ms_per_200ms_eq": float(meta["planning_ms"]),
                    "executed_actions": int(executed),
                    "spent_ms": float(spent),
                    **state,
                    **meta,
                }
            )
            actions.extend(arows)
    finally:
        eng.close()
    return pd.DataFrame(rows), pd.DataFrame(actions)


def run_normal(planner, name: str, seed: int, windows_n: int, env_cfg: dict):
    from final_radar_campaign import run_fixed

    return run_fixed(planner, name, 50, 100, seed, windows_n, 200, env_cfg)


def plot(windows):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = [
        ("cumulative_reward", "Cumulative Reward"),
        ("search_fraction", "Search Fraction"),
        ("drop_pct_active", "Drop % Active"),
        ("tracked_targets", "Tracked Targets"),
        ("mean_delay_active", "Mean Delay"),
        ("planning_ms_per_decision", "Planning ms"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        for planner, sub in windows.groupby("planner"):
            curve = sub.groupby("window", as_index=False)[metric].mean()
            ax.plot(curve["window"], curve[metric], label=planner, linewidth=1.6)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(OUT / "exact_rescore128_suite.png", dpi=180)
    plt.close(fig)


def main():
    env_cfg = env_preset_cfg("repaired_stress")
    seeds = [int(x) for x in os.environ.get("EXACT_RESC_SEEDS", "983,984,985").split(",") if x.strip()]
    windows_n = int(os.environ.get("EXACT_RESC_WINDOWS", "500"))
    top_k_long = int(os.environ.get("EXACT_RESC_TOPK_LONG", "16"))
    top_k_short = int(os.environ.get("EXACT_RESC_TOPK_SHORT", "8"))
    exact_planners = {}
    if top_k_long > 0:
        exact_planners[f"ExactRescore128_k{top_k_long}_h1200"] = ExactRescore128(
            env_cfg, top_k=top_k_long, score_horizon_ms=1200.0, slots=96, generator="structured"
        )
    if top_k_short > 0:
        exact_planners[f"ExactRescore128_k{top_k_short}_h400"] = ExactRescore128(
            env_cfg, top_k=top_k_short, score_horizon_ms=400.0, slots=64, generator="structured"
        )
    normal_planners = {
        "EDF": EDFPlanner(100),
        "EST": ESTPlanner(100),
        "StructuredVec128": StructuredVectorized128(max_trackers=100, n_plans=128, slots=32, seed=1234, env_cfg=env_cfg),
    }
    all_w = []
    all_s = []
    for seed in seeds:
        for name, planner in exact_planners.items():
            print(f"running {name} seed={seed}", flush=True)
            wdf, _ = run_exact_rescore(planner, name, seed, windows_n, env_cfg)
            all_w.append(wdf)
            s = summarize_window_df(wdf, mode="fixed")
            s.update({"planner": name, "seed": seed})
            all_s.append(s)
        for name, planner in normal_planners.items():
            print(f"running {name} seed={seed}", flush=True)
            wdf, _ = run_normal(planner, name, seed, windows_n, env_cfg)
            all_w.append(wdf)
            s = summarize_window_df(wdf, mode="fixed")
            s.update({"planner": name, "seed": seed})
            all_s.append(s)
    windows = pd.concat(all_w, ignore_index=True)
    raw = pd.DataFrame(all_s)
    summary = raw.groupby("planner", as_index=False).mean(numeric_only=True)
    windows.to_csv(OUT / "windows.csv", index=False)
    raw.to_csv(OUT / "by_seed.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    plot(windows)
    print(summary.to_string(index=False))
    print(OUT)


if __name__ == "__main__":
    main()
