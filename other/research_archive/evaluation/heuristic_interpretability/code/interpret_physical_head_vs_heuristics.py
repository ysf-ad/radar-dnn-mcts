from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from compare_action_heads_smoke import usable_targets
from exact_env_mutual import EDFPlanner, ESTPlanner, MAXT, _DummyPlanner, attach_env_obs, engine_env_cfg, env_cfg_for, xs_decode_action
from final_radar_campaign import get_obs
from foundation_mcts_fair_eval import FairExactRescore, parse_floats, parse_ints
from learned_proposal_fair_eval import LearnedProposalFairExact
from penalty_window_quota_learner_eval import make_exact_args
from repaired_campaign_tools import build_env
from strict_window_report import execute_plan_until_budget
from two_sensor_physical_head_eval import PhysicalHeadPlanner, train_head


def action_label(action: int) -> tuple[str, int, int]:
    if int(action) < 0:
        return "none", -1, -1
    base, sensor = xs_decode_action(int(action), MAXT)
    sensor_idx = -1 if sensor is None else int(sensor)
    if int(base) == 0:
        return "search", 0, sensor_idx
    if int(base) > 0:
        return "track", int(base), sensor_idx
    return "none", int(base), sensor_idx


def target_features(obs: dict, action: int, prefix: str) -> dict:
    kind, base, sensor = action_label(int(action))
    out = {
        f"{prefix}_action": int(action),
        f"{prefix}_kind": kind,
        f"{prefix}_base": int(base),
        f"{prefix}_sensor": int(sensor),
    }
    if kind != "track":
        out.update(
            {
                f"{prefix}_deadline": np.nan,
                f"{prefix}_desired": np.nan,
                f"{prefix}_lateness": np.nan,
                f"{prefix}_dwell": np.nan,
                f"{prefix}_priority": np.nan,
                f"{prefix}_range": np.nan,
                f"{prefix}_deadline_rank": np.nan,
                f"{prefix}_lateness_rank": np.nan,
                f"{prefix}_target_status": kind,
            }
        )
        return out

    idx = int(base) - 1
    active = np.asarray(obs.get("active_mask", []), dtype=bool)
    deadline = np.asarray(obs.get("t_deadline", []), dtype=np.float32)
    desired = np.asarray(obs.get("t_desired", np.zeros_like(deadline)), dtype=np.float32)
    dwell = np.asarray(obs.get("t_dwell", np.zeros_like(deadline)), dtype=np.float32)
    priority = np.asarray(obs.get("priority", np.zeros_like(deadline)), dtype=np.float32)
    target_range = np.asarray(obs.get("target_range", np.zeros_like(deadline)), dtype=np.float32)
    n = min(len(active), len(deadline), len(desired), len(dwell), len(priority), len(target_range), MAXT)
    if not (0 <= idx < n):
        out[f"{prefix}_target_status"] = "invalid"
        return out

    active_idx = np.where(active[:n])[0]
    tracked_idx = np.where(active[:n] & (deadline[:n] >= 0.0))[0]
    deadline_rank = np.nan
    if idx in set(tracked_idx.tolist()):
        order = tracked_idx[np.argsort(deadline[tracked_idx])]
        deadline_rank = int(np.where(order == idx)[0][0]) + 1
    lateness = np.maximum(0.0, -desired[:n])
    lateness_rank = np.nan
    if idx in set(active_idx.tolist()):
        order = active_idx[np.argsort(-lateness[active_idx])]
        lateness_rank = int(np.where(order == idx)[0][0]) + 1

    dline = float(deadline[idx])
    des = float(desired[idx])
    dw = float(dwell[idx])
    if dline < 0.0:
        status = "dropped"
    elif dline <= max(1.0, dw) + 50.0:
        status = "urgent"
    elif des < 0.0:
        status = "late_desired"
    else:
        status = "healthy"
    out.update(
        {
            f"{prefix}_deadline": dline,
            f"{prefix}_desired": des,
            f"{prefix}_lateness": max(0.0, -des),
            f"{prefix}_dwell": dw,
            f"{prefix}_priority": float(priority[idx]),
            f"{prefix}_range": float(target_range[idx]),
            f"{prefix}_deadline_rank": deadline_rank,
            f"{prefix}_lateness_rank": lateness_rank,
            f"{prefix}_target_status": status,
        }
    )
    return out


def compare_category(model_action: int, heuristic_action: int, hname: str) -> str:
    m_kind, m_base, _ = action_label(int(model_action))
    h_kind, h_base, _ = action_label(int(heuristic_action))
    if m_kind == "search" and h_kind == "search":
        return f"both_search"
    if m_kind == "track" and h_kind == "track" and int(m_base) == int(h_base):
        return f"same_track_target"
    if m_kind == "track" and h_kind == "track":
        return f"different_track_target"
    if m_kind == "search" and h_kind == "track":
        return f"model_search_{hname.lower()}_track"
    if m_kind == "track" and h_kind == "search":
        return f"model_track_{hname.lower()}_search"
    return "other"


def train_winning_model(args):
    train_args = argparse.Namespace(
        d_model=48,
        nhead=4,
        nlayers=2,
        lr=3e-4,
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        model_seed=int(args.model_seed),
        q_loss_weight=0.25,
        value_loss_weight=0.25,
        search_calibration_weight=0.0,
        log_every=max(1, int(args.train_steps)),
        cell_balanced_sampling=bool(args.cell_balanced_sampling),
    )
    torch.manual_seed(int(args.model_seed))
    np.random.seed(int(args.model_seed))
    model = train_head(str(args.variant), usable_targets(Path(args.targets)), train_args, torch.device("cpu"))
    if str(args.finetune_targets).strip():
        ft_args = argparse.Namespace(**vars(train_args))
        ft_args.train_steps = int(args.finetune_steps)
        ft_args.log_every = max(1, int(ft_args.train_steps))
        ft_args.value_loss_weight = 0.0
        model = train_head(str(args.variant), usable_targets(Path(args.finetune_targets)), ft_args, torch.device("cpu"), model=model)
    return model.eval()


def make_planner(args, model, env_cfg: dict):
    learned = []
    for bias in parse_floats(args.search_biases):
        for q_weight in parse_floats(args.q_weights):
            learned.append(
                PhysicalHeadPlanner(
                    model,
                    str(args.variant),
                    env_cfg,
                    policy_weight=1.0,
                    q_weight=float(q_weight),
                    search_score_bias=float(bias),
                )
            )
    return LearnedProposalFairExact(
        env_cfg,
        learned,
        top_k=int(args.top_k),
        score_horizon_ms=float(args.score_horizon_ms),
        slots=96,
        generator="structured",
        seed=15008,
        force_learned_rescore=bool(args.force_learned_rescore),
        learned_extra_top_k=int(args.learned_extra_top_k),
        preserve_base_topk=bool(args.preserve_base_topk),
    )


def run_trace(args, model) -> pd.DataFrame:
    exact_args = make_exact_args(args)
    exact_args.enable_x_band = True
    exact_args.single_sensor = False
    rows = []
    for seed in parse_ints(args.eval_seeds):
        for initial in parse_ints(args.initials):
            for rate in parse_floats(args.rates):
                env_cfg = env_cfg_for(float(rate), exact_args)
                env_cfg["enable_x_band"] = 1
                model_planner = make_planner(args, model, env_cfg)
                fair_planner = FairExactRescore(env_cfg, top_k=int(args.top_k), score_horizon_ms=float(args.score_horizon_ms), slots=96, generator="structured", seed=15008)
                edf = EDFPlanner(MAXT)
                est = ESTPlanner(MAXT)
                eng = build_env(_DummyPlanner(), int(initial), MAXT, int(seed), 200, engine_env_cfg(env_cfg))
                eng.reset(seed=int(seed))
                debt = 0.0
                try:
                    for window in range(int(args.eval_windows)):
                        if bool(eng.term_buf[0]):
                            break
                        raw_obs = get_obs(eng, debt)
                        obs = attach_env_obs(raw_obs, env_cfg, True, True)
                        model_plan, meta = model_planner.choose(eng, debt, raw_obs)
                        fair_plan, _fair_meta = fair_planner.choose(eng, debt, raw_obs)
                        edf_plan = list(edf.plan(obs, budget_ms=200))
                        est_plan = list(est.plan(obs, budget_ms=200))
                        actions = {
                            "model": int(model_plan[0]) if model_plan else -1,
                            "fair_exact": int(fair_plan[0]) if fair_plan else -1,
                            "EDF": int(edf_plan[0]) if edf_plan else -1,
                            "EST": int(est_plan[0]) if est_plan else -1,
                        }
                        base_row = {
                            "initial": int(initial),
                            "rate": float(rate),
                            "seed": int(seed),
                            "window": int(window),
                            "active_targets": int(np.asarray(obs.get("active_mask", []), dtype=bool).sum()),
                            "model_planning_ms": float(meta.get("planning_ms", np.nan)),
                        }
                        for name, action in actions.items():
                            base_row.update(target_features(obs, int(action), name))
                        for hname in ("EDF", "EST", "fair_exact"):
                            row = dict(base_row)
                            row["baseline"] = hname
                            row["category"] = compare_category(actions["model"], actions[hname], hname)
                            for metric in ("deadline", "lateness", "priority", "range", "deadline_rank", "lateness_rank"):
                                row[f"model_minus_{hname}_{metric}"] = row.get(f"model_{metric}", np.nan) - row.get(f"{hname}_{metric}", np.nan)
                            rows.append(row)
                        reward, _spent, debt, _executed, _searches, _arows = execute_plan_until_budget(
                            eng, [int(a) for a in model_plan], 200.0, float(debt), "interpret_model", int(seed), int(window)
                        )
                        if not np.isfinite(float(reward)):
                            break
                finally:
                    eng.close()
    return pd.DataFrame(rows)


def write_outputs(events: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "target_selection_events.csv", index=False)
    counts = events.groupby(["baseline", "category"], as_index=False).size().rename(columns={"size": "count"})
    totals = counts.groupby("baseline")["count"].transform("sum")
    counts["fraction"] = counts["count"] / totals.clip(lower=1)
    counts.to_csv(out_dir / "target_selection_category_counts.csv", index=False)

    diff = events[events["category"] == "different_track_target"].copy()
    delta_cols = [c for c in diff.columns if c.startswith("model_minus_")]
    if delta_cols:
        diff_summary = diff.groupby("baseline")[delta_cols].mean().reset_index()
    else:
        diff_summary = pd.DataFrame()
    diff_summary.to_csv(out_dir / "different_track_target_feature_deltas.csv", index=False)

    plt.figure(figsize=(9, 4.8))
    pivot = counts.pivot(index="category", columns="baseline", values="fraction").fillna(0.0)
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("fraction of window-start decisions")
    plt.title("Best factorized model: decision agreement vs baselines")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "target_selection_category_fractions.png", dpi=170)
    plt.close()

    if not diff_summary.empty:
        plot_cols = [c for c in diff_summary.columns if c.endswith("_deadline_rank") or c.endswith("_lateness_rank")]
        if plot_cols:
            plt.figure(figsize=(8, 4.5))
            x = np.arange(len(diff_summary["baseline"]))
            width = 0.35
            for i, col in enumerate(plot_cols[:2]):
                plt.bar(x + (i - 0.5) * width, diff_summary[col], width=width, label=col.replace("model_minus_", "").replace("_", " "))
            plt.axhline(0, color="black", linewidth=0.8)
            plt.xticks(x, diff_summary["baseline"])
            plt.ylabel("model minus baseline rank")
            plt.title("When both track different targets")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "different_track_rank_deltas.png", dpi=170)
            plt.close()

    notes = ["# Target-selection interpretation", ""]
    notes.append("Counts are measured at the start of each 200 ms window on the same 9-cell seed-916 grid.")
    notes.append("")
    notes.append(counts.to_markdown(index=False))
    if not diff_summary.empty:
        notes.append("")
        notes.append("## Different-track target feature deltas")
        notes.append("")
        notes.append(diff_summary.to_markdown(index=False))
    (out_dir / "target_selection_notes.md").write_text("\n".join(notes), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="CreateValid1/results/edf_bootstrap_r3_lmh_1024_targets.pt")
    ap.add_argument("--finetune-targets", default="CreateValid1/results/selfplay_adv_edf_owntail_factor_r3_lmh_512_targets.pt")
    ap.add_argument("--out-dir", default="CreateValid1/results/slide_ready_interpretation")
    ap.add_argument("--variant", default="two_row_action_attention_factored_loss")
    ap.add_argument("--initials", default="20,40,60")
    ap.add_argument("--rates", default="2,3,4")
    ap.add_argument("--eval-seeds", default="916")
    ap.add_argument("--eval-windows", type=int, default=100)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=120)
    ap.add_argument("--finetune-steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--model-seed", type=int, default=123)
    ap.add_argument("--cell-balanced-sampling", action="store_true")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--score-horizon-ms", type=float, default=800.0)
    ap.add_argument("--search-biases", default="-14,-12,-10,-8,-5")
    ap.add_argument("--q-weights", default="0.5")
    ap.add_argument("--force-learned-rescore", action="store_true")
    ap.add_argument("--preserve-base-topk", action="store_true")
    ap.add_argument("--learned-extra-top-k", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(1)
    model = train_winning_model(args)
    events = run_trace(args, model)
    write_outputs(events, Path(args.out_dir))
    print({"events": int(len(events)), "out_dir": str(args.out_dir)}, flush=True)


if __name__ == "__main__":
    main()
