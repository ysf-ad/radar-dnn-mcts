from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "radar_dnn_mcts"))


def load_model_checkpoint(model, checkpoint: Path | None):
    if checkpoint is None:
        return model
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def action_kind(action: int, max_trackers: int) -> str:
    from repaired_campaign_tools import decode_sensor_action

    logical, sensor = decode_sensor_action(int(action), int(max_trackers))
    prefix = "S" if sensor == 0 else ("X" if sensor == 1 else "?")
    if logical == 0:
        return f"{prefix}:search"
    if logical > 0:
        return f"{prefix}:track:{logical}"
    return "idle"


def common_prefix_len(left: list[int], right: list[int]) -> int:
    n = min(len(left), len(right))
    for i in range(n):
        if int(left[i]) != int(right[i]):
            return i
    return n


def make_env(seed: int, initial_targets: int, max_trackers: int, window_ms: int, env_cfg: dict):
    from repaired_campaign_tools import EDFPlanner, build_env

    eng = build_env(EDFPlanner(max_trackers), int(initial_targets), int(max_trackers), int(seed), int(window_ms), env_cfg)
    eng.reset(seed=int(seed))
    return eng


def run_one(planner, name: str, args, env_cfg: dict, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from repaired_campaign_tools import get_obs
    from strict_window_report import execute_plan_until_budget, sample_state_metrics

    eng = make_env(args.seed, args.initial_targets, args.max_trackers, args.window_ms, env_cfg)
    debt = 0.0
    cumulative = 0.0
    windows = []
    actions = []
    if hasattr(planner, "warmup"):
        planner.warmup(get_obs(eng, debt), budget_ms=args.window_ms)
        sync(device)
    for w in range(int(args.windows)):
        if eng.term_buf[0]:
            break
        obs = get_obs(eng, debt)
        sync(device)
        t0 = time.perf_counter()
        plan = [int(a) for a in planner.plan(obs, budget_ms=args.window_ms)]
        sync(device)
        plan_ms = (time.perf_counter() - t0) * 1000.0
        reward, spent_ms, debt, executed, searches, arows = execute_plan_until_budget(
            eng, plan, float(args.window_ms), float(debt), name, int(args.seed), int(w)
        )
        cumulative += float(reward)
        state = sample_state_metrics(eng, debt)
        executed_actions = [int(row["action"]) for row in arows]
        windows.append(
            {
                "planner": name,
                "window": int(w),
                "window_reward": float(reward),
                "cumulative_reward": float(cumulative),
                "plan_ms": float(plan_ms),
                "executed": int(executed),
                "searches": int(searches),
                "search_fraction": float(searches / max(1, executed)),
                "spent_ms": float(spent_ms),
                "plan": json.dumps(plan),
                "executed_actions": json.dumps(executed_actions),
                "first_plan_action": int(plan[0]) if plan else -1,
                "first_plan_action_kind": action_kind(plan[0], args.max_trackers) if plan else "none",
                **state,
            }
        )
        for slot, row in enumerate(arows):
            out = dict(row)
            out["planner"] = name
            out["window"] = int(w)
            out["slot"] = int(slot)
            out["action_kind"] = action_kind(int(row["action"]), args.max_trackers)
            actions.append(out)
    eng.close()
    win_df = pd.DataFrame(windows)
    act_df = pd.DataFrame(actions)
    summary = {
        "planner": name,
        "windows": int(len(win_df)),
        "total_reward": float(win_df["window_reward"].sum()) if not win_df.empty else 0.0,
        "reward_per_window": float(win_df["window_reward"].mean()) if not win_df.empty else 0.0,
        "final_cumulative_reward": float(win_df["cumulative_reward"].iloc[-1]) if not win_df.empty else 0.0,
        "executed_actions": int(win_df["executed"].sum()) if not win_df.empty else 0,
        "search_fraction": float(win_df["searches"].sum() / max(1, win_df["executed"].sum())) if not win_df.empty else 0.0,
        "plan_ms_per_window_mean": float(win_df["plan_ms"].mean()) if not win_df.empty else 0.0,
        "plan_ms_per_executed_action": float(win_df["plan_ms"].sum() / max(1, win_df["executed"].sum())) if not win_df.empty else 0.0,
        "final_active_targets": float(win_df["active_targets"].iloc[-1]) if "active_targets" in win_df and not win_df.empty else 0.0,
        "final_tracked_targets": float(win_df["tracked_targets"].iloc[-1]) if "tracked_targets" in win_df and not win_df.empty else 0.0,
        "final_drop_pct_active": float(win_df["drop_pct_active"].iloc[-1]) if "drop_pct_active" in win_df and not win_df.empty else 0.0,
    }
    return win_df, act_df, summary


def pairwise_compare(base: pd.DataFrame, other: pd.DataFrame, base_name: str, other_name: str) -> dict:
    rows = []
    for _, brow in base.iterrows():
        w = int(brow["window"])
        match = other[other["window"] == w]
        if match.empty:
            continue
        orow = match.iloc[0]
        bplan = [int(x) for x in json.loads(brow["plan"])]
        oplan = [int(x) for x in json.loads(orow["plan"])]
        bexec = [int(x) for x in json.loads(brow["executed_actions"])]
        oexec = [int(x) for x in json.loads(orow["executed_actions"])]
        rows.append(
            {
                "window": w,
                "plan_exact": bplan == oplan,
                "executed_exact": bexec == oexec,
                "first_plan_match": (bplan[:1] == oplan[:1]),
                "common_plan_prefix": common_prefix_len(bplan, oplan),
                "base_first": action_kind(bplan[0], 100) if bplan else "none",
                "other_first": action_kind(oplan[0], 100) if oplan else "none",
                "reward_delta_other_minus_base": float(orow["window_reward"]) - float(brow["window_reward"]),
            }
        )
    cmp = pd.DataFrame(rows)
    if cmp.empty:
        return {"base": base_name, "other": other_name, "windows_compared": 0}
    first_mismatch = cmp[(~cmp["plan_exact"]) | (~cmp["executed_exact"])].head(1)
    out = {
        "base": base_name,
        "other": other_name,
        "windows_compared": int(len(cmp)),
        "plan_exact_fraction": float(cmp["plan_exact"].mean()),
        "executed_exact_fraction": float(cmp["executed_exact"].mean()),
        "first_action_match_fraction": float(cmp["first_plan_match"].mean()),
        "mean_common_plan_prefix": float(cmp["common_plan_prefix"].mean()),
        "total_reward_delta_other_minus_base": float(cmp["reward_delta_other_minus_base"].sum()),
    }
    if not first_mismatch.empty:
        row = first_mismatch.iloc[0]
        out["first_mismatch_window"] = int(row["window"])
        out["first_mismatch_base_first"] = str(row["base_first"])
        out["first_mismatch_other_first"] = str(row["other_first"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("../CreateValid1/results/critic_bootstrap_medium_eval_two_row_action_attention_qpolicy_factored_loss.pt"))
    parser.add_argument("--variant", default="two_row_action_attention_qpolicy_factored_loss")
    parser.add_argument("--out-dir", type=Path, default=Path("results/same_env_planner_compare"))
    parser.add_argument("--seed", type=int, default=916)
    parser.add_argument("--initial-targets", type=int, default=60)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--max-trackers", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--gpu-select", action="store_true")
    parser.add_argument("--manual-action-coupler", action="store_true")
    args = parser.parse_args()

    from perf_fast_planner import FastActionAttentionPlanner
    from repaired_campaign_tools import env_preset_cfg
    from two_sensor_physical_head_eval import (
        ActionAttentionFactorizedNet,
        CachedRootActionAttentionFactorizedNet,
        PhysicalHeadPlanner,
    )

    torch.set_num_threads(1)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    env_cfg = env_preset_cfg("repaired_stress")
    env_cfg["poisson_rate_per_second"] = float(args.rate)
    env_cfg["enable_x_band"] = 1

    model_cls = CachedRootActionAttentionFactorizedNet if str(args.variant) == "cached_root_action_attention_qpolicy_factored_loss" else ActionAttentionFactorizedNet
    old_model = load_model_checkpoint(model_cls(48, 4, 2).eval(), args.checkpoint)
    exact_fast_model = load_model_checkpoint(model_cls(48, 4, 2).eval(), args.checkpoint)
    graph_fast_model = load_model_checkpoint(model_cls(48, 4, 2).eval(), args.checkpoint)
    selected_fast_model = load_model_checkpoint(model_cls(48, 4, 2).eval(), args.checkpoint)

    planners = [
        (
            "old_physical_head",
            PhysicalHeadPlanner(
                old_model,
                str(args.variant),
                env_cfg,
                policy_weight=1.0,
                q_weight=1.0,
            ),
        ),
        (
            "fast_cached_cpu_select",
            FastActionAttentionPlanner(
                exact_fast_model,
                env_cfg,
                policy_weight=1.0,
                q_weight=1.0,
                device=device,
                use_amp=False,
                use_cuda_graph=False,
                use_gpu_select=False,
                use_manual_action_coupler=bool(args.manual_action_coupler),
            ),
        ),
        (
            "fast_graph_gpu_select",
            FastActionAttentionPlanner(
                graph_fast_model,
                env_cfg,
                policy_weight=1.0,
                q_weight=1.0,
                device=device,
                use_amp=bool(args.amp),
                use_cuda_graph=bool(args.graph),
                use_gpu_select=bool(args.gpu_select),
                use_manual_action_coupler=bool(args.manual_action_coupler),
            ),
        ),
        (
            "fast_reencode_selected",
            FastActionAttentionPlanner(
                selected_fast_model,
                env_cfg,
                policy_weight=1.0,
                q_weight=1.0,
                device=device,
                use_amp=bool(args.amp),
                use_cuda_graph=False,
                use_gpu_select=bool(args.gpu_select),
                use_manual_action_coupler=bool(args.manual_action_coupler),
                reencode_selected=True,
            ),
        ),
    ]

    all_windows = []
    all_actions = []
    summaries = []
    by_name = {}
    for name, planner in planners:
        win_df, act_df, summary = run_one(planner, name, args, env_cfg, device)
        all_windows.append(win_df)
        all_actions.append(act_df)
        summaries.append(summary)
        by_name[name] = win_df

    comparisons = [
        pairwise_compare(by_name["old_physical_head"], by_name["fast_cached_cpu_select"], "old_physical_head", "fast_cached_cpu_select"),
        pairwise_compare(by_name["old_physical_head"], by_name["fast_graph_gpu_select"], "old_physical_head", "fast_graph_gpu_select"),
        pairwise_compare(by_name["old_physical_head"], by_name["fast_reencode_selected"], "old_physical_head", "fast_reencode_selected"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    window_df = pd.concat(all_windows, ignore_index=True)
    action_df = pd.concat(all_actions, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    compare_df = pd.DataFrame(comparisons)
    window_path = args.out_dir / "same_env_windows.csv"
    action_path = args.out_dir / "same_env_actions.csv"
    summary_path = args.out_dir / "same_env_summary.csv"
    compare_path = args.out_dir / "same_env_action_compare.csv"
    report_path = args.out_dir / "same_env_report.json"
    window_df.to_csv(window_path, index=False)
    action_df.to_csv(action_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    compare_df.to_csv(compare_path, index=False)
    report = {
        "checkpoint": str(args.checkpoint),
        "seed": int(args.seed),
        "initial_targets": int(args.initial_targets),
        "rate": float(args.rate),
        "windows": int(args.windows),
        "device": str(device),
        "amp": bool(args.amp),
        "graph": bool(args.graph),
        "gpu_select": bool(args.gpu_select),
        "manual_action_coupler": bool(args.manual_action_coupler),
        "summary": summaries,
        "comparisons": comparisons,
        "files": {
            "windows": str(window_path),
            "actions": str(action_path),
            "summary": str(summary_path),
            "comparisons": str(compare_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
