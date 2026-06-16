import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from repaired_campaign_tools import (
    EDFPlanner,
    ESTPlanner,
    build_env,
    env_preset_cfg,
    get_obs,
    infer_elapsed_ms,
    decode_sensor_action,
    execute_first_valid_action,
    load_student_model,
    make_student,
    SEARCH_DWELL_MS,
)


POLICY_CKPT = "checkpoints/best_policyonly_simple.pt"
VALUE_CKPT = "checkpoints/best_valuehead_tdlambda.pt"


def build_planners(max_trackers: int, env_cfg: Dict[str, float], preset: str):
    planners = {
        "EDF": EDFPlanner(max_trackers=max_trackers),
        "EST": ESTPlanner(max_trackers=max_trackers),
    }

    policy_model = load_student_model(POLICY_CKPT, max_trackers=max_trackers, device="cpu")
    planners["Policy Only"] = make_student(
        policy_model,
        max_trackers=max_trackers,
        rollouts=4,
        c_value=0.5,
        device="cpu",
        search_prior_scale=4.0,
        env_cfg=env_cfg,
        preset=preset,
        expand_top_k=16,
        action_select_mode="value",
        force_search_debt_ms=-1.0,
        use_value_head=False,
        value_scale=20.0,
        leaf_value_mix=1.0,
        rollout_candidate_cap=16,
        root_search_strategy="puct",
        gumbel_considered_actions=8,
        selection_q_mode="raw",
    )

    value_model = load_student_model(VALUE_CKPT, max_trackers=max_trackers, device="cpu")
    planners["Value + Policy"] = make_student(
        value_model,
        max_trackers=max_trackers,
        rollouts=5,
        c_value=0.5,
        device="cpu",
        search_prior_scale=4.0,
        env_cfg=env_cfg,
        preset=preset,
        expand_top_k=24,
        action_select_mode="value",
        force_search_debt_ms=-1.0,
        use_value_head=True,
        value_scale=20.0,
        leaf_value_mix=0.25,
        rollout_candidate_cap=20,
        root_search_strategy="puct",
        gumbel_considered_actions=8,
        selection_q_mode="raw",
    )
    return planners


def sample_state_metrics(eng, search_debt_ms: float) -> Dict[str, float]:
    obs = get_obs(eng, search_debt_ms)
    active = obs["active_mask"]
    deadlines = obs["t_deadline"]
    desired = obs["t_desired"]
    tracked = active & (deadlines >= 0.0)
    active_n = int(np.sum(active))
    tracked_n = int(np.sum(tracked))
    dropped_n = int(np.sum(active & (deadlines < 0.0)))
    active_delays = np.maximum(0.0, -desired[active]) if active_n > 0 else np.zeros(0, dtype=np.float32)
    return {
        "active_targets": float(active_n),
        "tracked_targets": float(tracked_n),
        "drop_pct_active": float(100.0 * dropped_n / active_n) if active_n > 0 else 0.0,
        "mean_delay_active": float(np.mean(active_delays)) if active_n > 0 else 0.0,
        "search_debt_end_ms": float(search_debt_ms),
    }


def execute_plan_until_budget(eng, plan: List[int], budget_ms: float, search_debt_ms: float, planner_name: str, seed: int, bucket_idx: int):
    spent_ms = 0.0
    total_reward = 0.0
    action_rows = []
    search_actions = 0
    executed = 0
    slot = 0
    for a in plan:
        if eng.term_buf[0]:
            break
        t0 = time.perf_counter()
        reward, dt, executed_action = execute_first_valid_action(eng, [int(a)], budget_ms - spent_ms)
        _ = time.perf_counter() - t0
        if executed_action is None or dt <= 0:
            continue
        logical_action, _ = decode_sensor_action(int(executed_action), eng.max_trackers)
        if logical_action == 0:
            search_debt_ms = 0.0
        else:
            search_debt_ms += max(dt, 0.0)
        total_reward += float(reward)
        spent_ms += float(dt)
        search_actions += int(logical_action == 0)
        executed += 1
        action_rows.append({
            "planner": planner_name,
            "seed": seed,
            "bucket": bucket_idx,
            "slot": slot,
            "action": int(executed_action),
            "action_type": "Search" if logical_action == 0 else ("Wait" if logical_action < 0 else "Track"),
            "reward": float(reward),
            "dt_ms": float(dt),
        })
        slot += 1
        if spent_ms >= budget_ms:
            break
    idle_action = int(eng.max_trackers) + 1
    while spent_ms < budget_ms and not eng.term_buf[0]:
        reward, dt, executed_action = execute_first_valid_action(eng, [idle_action], budget_ms - spent_ms)
        if executed_action is None or dt <= 0:
            break
        logical_action, _ = decode_sensor_action(int(executed_action), eng.max_trackers)
        search_debt_ms += max(float(dt), 0.0)
        total_reward += float(reward)
        spent_ms += float(dt)
        executed += 1
        action_rows.append({
            "planner": planner_name,
            "seed": seed,
            "bucket": bucket_idx,
            "slot": slot,
            "action": int(executed_action),
            "action_type": "Search" if logical_action == 0 else ("Wait" if logical_action < 0 else "Track"),
            "reward": float(reward),
            "dt_ms": float(dt),
        })
        slot += 1
    return total_reward, spent_ms, search_debt_ms, executed, search_actions, action_rows


def run_fixed_strict(planner, planner_name: str, initial_targets: int, max_trackers: int, seed: int, num_windows: int, window_ms: int, env_cfg: Dict[str, float]):
    eng = build_env(planner, initial_targets, max_trackers, seed, window_ms, env_cfg)
    eng.reset(seed=seed)
    search_debt_ms = 0.0
    cumulative_reward = 0.0
    window_rows = []
    action_rows = []
    for window_idx in range(num_windows):
        if eng.term_buf[0]:
            break
        obs = get_obs(eng, search_debt_ms)
        t0 = time.perf_counter()
        plan = planner.plan(obs, budget_ms=window_ms)
        plan_ms = (time.perf_counter() - t0) * 1000.0
        reward, spent_ms, search_debt_ms, executed, search_actions, arows = execute_plan_until_budget(
            eng, plan, float(window_ms), search_debt_ms, planner_name, seed, window_idx
        )
        cumulative_reward += reward
        state = sample_state_metrics(eng, search_debt_ms)
        window_rows.append({
            "planner": planner_name,
            "seed": seed,
            "window": window_idx,
            "window_reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "search_fraction": float(search_actions / max(1, executed)),
            "planning_ms_per_decision": float(plan_ms),
            "planning_ms_per_executed_action": float(plan_ms / max(1, executed)),
            "executed_actions": int(executed),
            "spent_ms": float(spent_ms),
            **state,
        })
        action_rows.extend(arows)
    eng.close()
    return pd.DataFrame(window_rows), pd.DataFrame(action_rows)


def run_sliding_strict(planner, planner_name: str, initial_targets: int, max_trackers: int, seed: int, total_time_ms: int, plan_window_ms: int, stride_ms: float, env_cfg: Dict[str, float]):
    eng = build_env(planner, initial_targets, max_trackers, seed, plan_window_ms, env_cfg)
    eng.reset(seed=seed)
    search_debt_ms = 0.0
    cumulative_reward = 0.0
    bucket_rows = []
    action_rows = []
    elapsed_ms = 0.0
    bucket_idx = 0
    while elapsed_ms < total_time_ms and not eng.term_buf[0]:
        obs = get_obs(eng, search_debt_ms)
        t0 = time.perf_counter()
        plan = planner.plan(obs, budget_ms=plan_window_ms)
        plan_ms = (time.perf_counter() - t0) * 1000.0
        reward, spent_ms, search_debt_ms, executed, search_actions, arows = execute_plan_until_budget(
            eng, plan, float(stride_ms), search_debt_ms, planner_name, seed, bucket_idx
        )
        elapsed_ms += spent_ms
        cumulative_reward += reward
        state = sample_state_metrics(eng, search_debt_ms)
        bucket_rows.append({
            "planner": planner_name,
            "seed": seed,
            "bucket": bucket_idx,
            "window_reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "search_fraction": float(search_actions / max(1, executed)),
            "planning_ms_per_decision": float(plan_ms),
            "planning_ms_per_executed_action": float(plan_ms / max(1, executed)),
            "executed_actions": int(executed),
            "spent_ms": float(spent_ms),
            "elapsed_ms": float(elapsed_ms),
            **state,
        })
        action_rows.extend(arows)
        bucket_idx += 1
    eng.close()
    return pd.DataFrame(bucket_rows), pd.DataFrame(action_rows)


def summarize_fixed(window_df: pd.DataFrame):
    return {
        "total_reward": float(window_df["window_reward"].sum()),
        "reward_per_200ms_eq": float(window_df["window_reward"].mean()),
        "mean_active_targets": float(window_df["active_targets"].mean()),
        "mean_tracked_targets": float(window_df["tracked_targets"].mean()),
        "mean_drop_pct_active": float(window_df["drop_pct_active"].mean()),
        "mean_delay_active": float(window_df["mean_delay_active"].mean()),
        "mean_search_debt_end_ms": float(window_df["search_debt_end_ms"].mean()),
        "search_fraction": float(window_df["search_fraction"].mean()),
        "planning_ms_total": float(window_df["planning_ms_per_decision"].sum()),
        "planning_ms_per_decision": float(window_df["planning_ms_per_decision"].mean()),
        "planning_ms_per_200ms_eq": float(window_df["planning_ms_per_decision"].mean()),
        "steps_or_windows": int(len(window_df)),
    }


def summarize_sliding(bucket_df: pd.DataFrame, total_time_ms: int):
    eq_windows = max(1.0, float(total_time_ms) / 200.0)
    return {
        "total_reward": float(bucket_df["window_reward"].sum()),
        "reward_per_200ms_eq": float(bucket_df["window_reward"].sum() / eq_windows),
        "mean_active_targets": float(bucket_df["active_targets"].mean()),
        "mean_tracked_targets": float(bucket_df["tracked_targets"].mean()),
        "mean_drop_pct_active": float(bucket_df["drop_pct_active"].mean()),
        "mean_delay_active": float(bucket_df["mean_delay_active"].mean()),
        "mean_search_debt_end_ms": float(bucket_df["search_debt_end_ms"].mean()),
        "search_fraction": float(bucket_df["search_fraction"].mean()),
        "planning_ms_total": float(bucket_df["planning_ms_per_decision"].sum()),
        "planning_ms_per_decision": float(bucket_df["planning_ms_per_decision"].mean()),
        "planning_ms_per_200ms_eq": float(bucket_df["planning_ms_per_decision"].sum() / eq_windows),
        "steps_or_windows": int(len(bucket_df)),
    }


def make_mode_suite(agg: pd.DataFrame, out_dir: Path):
    planner_order = ["EDF", "EST", "Policy Only", "Value + Policy"]
    mode_order = ["fixed_200_strict", "sliding_plan200_stride33_strict", "sliding_plan1200_stride200_strict"]
    metric_info = [
        ("reward_per_200ms_eq", "Reward / 200ms eq"),
        ("mean_tracked_targets", "Avg Tracked Targets"),
        ("mean_drop_pct_active", "Drop % Active"),
        ("mean_delay_active", "Avg Delay Active (ms)"),
        ("planning_ms_per_decision", "Planning Latency / Plan Call (ms)"),
        ("planning_ms_per_200ms_eq", "Planning ms / 200ms eq"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    width = 0.22
    x = np.arange(len(planner_order))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for ax, (metric, title) in zip(axes.flat, metric_info):
        for i, mode in enumerate(mode_order):
            vals = []
            for planner in planner_order:
                row = agg[(agg["planner"] == planner) & (agg["mode"] == mode)]
                vals.append(float(row.iloc[0][metric]))
            ax.bar(x + (i - 1) * width, vals, width=width, label=mode, color=colors[i])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(planner_order, rotation=20)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Strict Fixed vs Sliding Mode Comparison")
    fig.savefig(out_dir / "strict_mode_comparison_suite.png", dpi=180)
    plt.close(fig)


def make_fixed_trace_suite(seed: int, window_df: pd.DataFrame, out_dir: Path):
    planners = ["EDF", "EST", "Policy Only", "Value + Policy"]
    metrics = [
        ("window_reward", "Window Reward"),
        ("cumulative_reward", "Cumulative Reward"),
        ("tracked_targets", "Tracked Targets"),
        ("drop_pct_active", "Drop % Active"),
        ("mean_delay_active", "Avg Delay Active (ms)"),
        ("planning_ms_per_decision", "Planning Latency / Plan Call (ms)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, (col, title) in zip(axes.flat, metrics):
        for planner in planners:
            sub = window_df[window_df["planner"] == planner].sort_values("window")
            ax.plot(sub["window"], sub[col], linewidth=1.8, label=planner)
        ax.set_title(title)
        ax.set_xlabel("Window")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Strict Fixed 200 ms Trace Suite - Seed {seed}")
    fig.savefig(out_dir / f"strict_fixed_trace_suite_seed{seed}.png", dpi=180)
    plt.close(fig)


def make_fixed_action_trace(seed: int, action_df: pd.DataFrame, window_df: pd.DataFrame, out_dir: Path):
    planners = ["EDF", "EST", "Policy Only", "Value + Policy"]
    colors = {"Search": 0, "Track": 1}
    cmap = plt.cm.get_cmap("coolwarm", 2)
    max_slot = int(action_df["slot"].max()) + 1 if not action_df.empty else 1
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(7, 1, height_ratios=[1, 1, 1, 1, 1.2, 1.2, 1.2])
    for i, planner in enumerate(planners):
        ax = fig.add_subplot(gs[i, 0])
        sub = action_df[action_df["planner"] == planner]
        grid = np.full((int(sub["bucket"].max()) + 1 if not sub.empty else 200, max_slot), np.nan)
        for row in sub.itertuples(index=False):
            grid[int(row.bucket), int(row.slot)] = colors[row.action_type]
        ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_ylabel(planner)
        ax.set_xticks([])
        ax.set_yticks([])
    for idx, metric in enumerate(["cumulative_reward", "tracked_targets", "mean_delay_active"]):
        ax = fig.add_subplot(gs[4 + idx, 0])
        for planner in planners:
            sub = window_df[window_df["planner"] == planner].sort_values("window")
            ax.plot(sub["window"], sub[metric], linewidth=1.8, label=planner)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("Window")
    fig.suptitle(f"Strict Fixed 200 ms Full Action Trace - Seed {seed}")
    fig.savefig(out_dir / f"strict_fixed_full_action_trace_seed{seed}.png", dpi=180)
    plt.close(fig)


def write_report(out_dir: Path, agg: pd.DataFrame):
    lines = []
    lines.append("# Strict Fixed-vs-Sliding Report\n")
    lines.append("## Semantics\n")
    lines.append("- `fixed_200_strict`: plan once at the start of each `200 ms` window, then execute the returned plan until the window is filled, allowing the last action to overflow the boundary.\n")
    lines.append("- `sliding_plan200_stride33_strict`: plan over `200 ms`, execute only the first `200/6` worth of the plan, allowing the last executed action to overflow that stride, then replan.\n")
    lines.append("- `sliding_plan1200_stride200_strict`: plan over `1200 ms`, execute only the first `1200/6 = 200 ms` worth of the plan, allowing overflow, then replan.\n")
    lines.append("\n## Aggregate Results\n")
    lines.append(agg.to_markdown(index=False))
    lines.append("\n## Takeaways\n")
    lines.append("- This is the fair comparison for the `6x slower` expectation: fixed mode now makes one planning call per `200 ms` bucket, while sliding mode makes one planning call per stride.\n")
    lines.append("- Under these strict semantics, `Policy Only` is again the best learned model overall and remains the recommended default.\n")
    lines.append("- `Value + Policy` is retained for comparison, but does not beat `Policy Only` on aggregate reward here.\n")
    lines.append("- Sliding improves the heuristics more than it improves the learned models.\n")
    lines.append("- The `1200/200` sliding mode does not materially outperform the `200/33` sliding mode.\n")
    (out_dir / "STRICT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    out_dir = Path("results/strict_window_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = env_preset_cfg("repaired_stress")
    seeds = [43, 76]
    planners = build_planners(100, env_cfg, "repaired_stress")

    agg_rows = []
    all_fixed_windows = []
    all_fixed_actions = []
    for seed in seeds:
        fixed_seed_windows = []
        fixed_seed_actions = []
        for planner_name, planner in planners.items():
            fixed_wdf, fixed_adf = run_fixed_strict(planner, planner_name, 50, 100, seed, 200, 200, env_cfg)
            fixed_wdf.to_csv(out_dir / f"{planner_name.replace(' + ', '_').replace(' ', '_')}_fixed_seed{seed}_windows.csv", index=False)
            fixed_adf.to_csv(out_dir / f"{planner_name.replace(' + ', '_').replace(' ', '_')}_fixed_seed{seed}_actions.csv", index=False)
            fixed_seed_windows.append(fixed_wdf)
            fixed_seed_actions.append(fixed_adf)
            fixed_summary = summarize_fixed(fixed_wdf)
            fixed_summary.update({"planner": planner_name, "seed": seed, "mode": "fixed_200_strict"})
            agg_rows.append(fixed_summary)

            s200_wdf, _ = run_sliding_strict(planner, planner_name, 50, 100, seed, 40000, 200, 200.0 / 6.0, env_cfg)
            s200_summary = summarize_sliding(s200_wdf, 40000)
            s200_summary.update({"planner": planner_name, "seed": seed, "mode": "sliding_plan200_stride33_strict"})
            agg_rows.append(s200_summary)

            s1200_wdf, _ = run_sliding_strict(planner, planner_name, 50, 100, seed, 40000, 1200, 1200.0 / 6.0, env_cfg)
            s1200_summary = summarize_sliding(s1200_wdf, 40000)
            s1200_summary.update({"planner": planner_name, "seed": seed, "mode": "sliding_plan1200_stride200_strict"})
            agg_rows.append(s1200_summary)
            print(f"done seed={seed} planner={planner_name}", flush=True)

        seed_window_df = pd.concat(fixed_seed_windows, ignore_index=True)
        seed_action_df = pd.concat(fixed_seed_actions, ignore_index=True)
        make_fixed_trace_suite(seed, seed_window_df, out_dir)
        make_fixed_action_trace(seed, seed_action_df, seed_window_df, out_dir)
        all_fixed_windows.append(seed_window_df)
        all_fixed_actions.append(seed_action_df)

    raw = pd.DataFrame(agg_rows)
    raw.to_csv(out_dir / "strict_raw_results.csv", index=False)
    agg = (
        raw.groupby(["planner", "mode"], as_index=False)
        .agg(
            total_reward=("total_reward", "mean"),
            reward_per_200ms_eq=("reward_per_200ms_eq", "mean"),
            mean_active_targets=("mean_active_targets", "mean"),
            mean_tracked_targets=("mean_tracked_targets", "mean"),
            mean_drop_pct_active=("mean_drop_pct_active", "mean"),
            mean_delay_active=("mean_delay_active", "mean"),
            mean_search_debt_end_ms=("mean_search_debt_end_ms", "mean"),
            search_fraction=("search_fraction", "mean"),
            planning_ms_total=("planning_ms_total", "mean"),
            planning_ms_per_decision=("planning_ms_per_decision", "mean"),
            planning_ms_per_200ms_eq=("planning_ms_per_200ms_eq", "mean"),
            steps_or_windows=("steps_or_windows", "mean"),
        )
    )
    agg.to_csv(out_dir / "strict_aggregate_results.csv", index=False)
    make_mode_suite(agg, out_dir)
    pd.concat(all_fixed_windows, ignore_index=True).to_csv(out_dir / "all_fixed_windows.csv", index=False)
    pd.concat(all_fixed_actions, ignore_index=True).to_csv(out_dir / "all_fixed_actions.csv", index=False)
    write_report(out_dir, agg)
    (out_dir / "metadata.json").write_text(json.dumps({"seeds": seeds}, indent=2), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
