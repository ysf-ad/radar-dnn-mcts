from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\yousi\Downloads\radar_outputs\alphazero_orthodox")


def env_args(args) -> list[str]:
    return [
        "--device", str(args.device),
        "--env-mode", str(args.env_mode),
        "--enable-x-band",
        "--track-loss-penalty", str(args.track_loss_penalty),
        "--target-service-weight", str(args.target_service_weight),
        "--target-service-horizon-ms", str(args.target_service_horizon_ms),
        "--sector-staleness-weight", str(args.sector_staleness_weight),
        "--search-frame-overdue-weight", str(args.search_frame_overdue_weight),
        "--search-frame-drop-penalty", str(args.search_frame_drop_penalty),
    ]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def unlink_outputs(path: Path) -> None:
    for p in [path, path.with_name(path.stem + "_summary.csv")]:
        if p.exists():
            p.unlink()


def eval_mean(path: Path, method: str) -> float:
    if not path.exists():
        return -1e18
    df = pd.read_csv(path)
    if df.empty:
        return -1e18
    if "method" in df.columns:
        df = df[df["method"] == method]
    return float(df["reward"].mean()) if not df.empty else -1e18


def gate_summary(path: Path, method: str) -> dict:
    if not path.exists():
        return {
            "method_score": -1e18,
            "best_heuristic_score": -1e18,
            "mean_margin_vs_best_heuristic": -1e18,
            "win_rate_vs_best_heuristic": 0.0,
            "cells": 0,
        }
    df = pd.read_csv(path)
    if df.empty:
        return {
            "method_score": -1e18,
            "best_heuristic_score": -1e18,
            "mean_margin_vs_best_heuristic": -1e18,
            "win_rate_vs_best_heuristic": 0.0,
            "cells": 0,
        }
    keys = ["initial", "rate", "seed"]
    model = df[df["method"] == method][[*keys, "reward"]].rename(columns={"reward": "model_reward"})
    heur = (
        df[df["method"].isin(["EDF", "EST"])]
        .groupby(keys, as_index=False)["reward"]
        .max()
        .rename(columns={"reward": "best_heuristic_reward"})
    )
    joined = model.merge(heur, on=keys, how="inner")
    method_score = float(model["model_reward"].mean()) if not model.empty else -1e18
    best_heuristic_score = float(heur["best_heuristic_reward"].mean()) if not heur.empty else -1e18
    if joined.empty:
        return {
            "method_score": method_score,
            "best_heuristic_score": best_heuristic_score,
            "mean_margin_vs_best_heuristic": -1e18,
            "win_rate_vs_best_heuristic": 0.0,
            "cells": 0,
        }
    margin = joined["model_reward"] - joined["best_heuristic_reward"]
    return {
        "method_score": method_score,
        "best_heuristic_score": best_heuristic_score,
        "mean_margin_vs_best_heuristic": float(margin.mean()),
        "win_rate_vs_best_heuristic": float((margin > 0.0).mean()),
        "cells": int(len(joined)),
    }


def model_cell_rewards(path: Path, method: str, reward_col: str) -> pd.DataFrame:
    keys = ["initial", "rate", "seed"]
    if not path.exists():
        return pd.DataFrame(columns=[*keys, reward_col])
    df = pd.read_csv(path)
    if df.empty or "method" not in df.columns:
        return pd.DataFrame(columns=[*keys, reward_col])
    cols = [*keys, "reward"]
    return df[df["method"] == method][cols].rename(columns={"reward": reward_col})


def compare_model_cells(
    candidate_path: Path,
    candidate_method: str,
    accepted_path: Path,
    accepted_method: str,
) -> dict:
    cand = model_cell_rewards(candidate_path, candidate_method, "candidate_reward")
    prev = model_cell_rewards(accepted_path, accepted_method, "accepted_reward")
    joined = cand.merge(prev, on=["initial", "rate", "seed"], how="inner")
    if joined.empty:
        return {
            "mean_margin_vs_accepted_cells": -1e18,
            "min_margin_vs_accepted_cells": -1e18,
            "win_rate_vs_accepted_cells": 0.0,
            "regression_cells": 999999,
            "compared_cells": 0,
        }
    margin = joined["candidate_reward"] - joined["accepted_reward"]
    return {
        "mean_margin_vs_accepted_cells": float(margin.mean()),
        "min_margin_vs_accepted_cells": float(margin.min()),
        "win_rate_vs_accepted_cells": float((margin > 0.0).mean()),
        "regression_cells": int((margin < 0.0).sum()),
        "compared_cells": int(len(joined)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--start-state", required=True)
    ap.add_argument("--accepted-state", default=str(OUT / "paper_az_accepted.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--env-mode", default="mcts_sched_v1")
    ap.add_argument("--use-arrival-feature", action="store_true")
    ap.add_argument("--track-loss-penalty", type=float, default=8.0)
    ap.add_argument("--target-service-weight", type=float, default=10.0)
    ap.add_argument("--target-service-horizon-ms", type=float, default=3000.0)
    ap.add_argument("--sector-staleness-weight", type=float, default=0.01)
    ap.add_argument("--search-frame-overdue-weight", type=float, default=0.20)
    ap.add_argument("--search-frame-drop-penalty", type=float, default=8.0)
    ap.add_argument("--generations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--inner-lr", type=float, default=1e-4)
    ap.add_argument("--rollouts", type=int, default=16)
    ap.add_argument("--eval-rollouts", type=int, default=1)
    ap.add_argument("--horizon-windows", type=int, default=3)
    ap.add_argument("--eval-horizon-windows", type=int, default=2)
    ap.add_argument("--expand-top-k", type=int, default=6)
    ap.add_argument("--eval-expand-top-k", type=int, default=48)
    ap.add_argument("--rollout-policy", choices=["model", "branch", "q", "pq", "random", "value", "edge", "edf", "est", "mixed"], default="branch")
    ap.add_argument("--branch-rollout-threshold", type=float, default=0.65)
    ap.add_argument("--plan-mode", choices=["atomic", "window", "first_window"], default="atomic")
    ap.add_argument("--window-extract", choices=["tree", "tree_fill", "best", "greedy_expand", "batched_value", "model_q", "edge_q"], default="tree_fill")
    ap.add_argument("--prior-mode", choices=["factorized", "flat", "branch_corrected", "physical_flat", "true_physical_flat"], default="factorized")
    ap.add_argument("--prior-uniform-mix", type=float, default=0.03)
    ap.add_argument("--prior-search-bias", type=float, default=0.0)
    ap.add_argument("--leaf-value-mix", type=float, default=1.0)
    ap.add_argument("--visit-unvisited-first", dest="visit_unvisited_first", action="store_true", default=True)
    ap.add_argument("--disable-visit-unvisited-first", dest="visit_unvisited_first", action="store_false")
    ap.add_argument("--self-play-sample-tau", type=float, default=0.0)
    ap.add_argument("--self-play-root-dirichlet-frac", type=float, default=0.0)
    ap.add_argument("--self-play-root-dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--c-puct", type=float, default=1.25)
    ap.add_argument("--value-target", choices=["mc_return", "none", "raw", "raw_diff", "continuous", "sign"], default="mc_return")
    ap.add_argument("--terminal-baseline-mode", choices=["max_heuristic", "accepted", "max_all", "edf", "est"], default="max_heuristic")
    ap.add_argument("--terminal-baseline-scale", type=float, default=1.0)
    ap.add_argument("--terminal-baseline-margin", type=float, default=0.0)
    ap.add_argument("--type-loss-weight", type=float, default=1.0)
    ap.add_argument("--track-loss-weight", type=float, default=1.0)
    ap.add_argument("--sensor-loss-weight", type=float, default=0.5)
    ap.add_argument("--joint-policy-loss-weight", type=float, default=0.5)
    ap.add_argument("--value-loss-weight", type=float, default=0.5)
    ap.add_argument("--factor-value-loss-weight", type=float, default=0.0)
    ap.add_argument("--policy-kl-weight", type=float, default=0.0)
    ap.add_argument("--replay-window", type=int, default=0)
    ap.add_argument("--extra-replay-targets", default="")
    ap.add_argument("--accepted-baseline-rollouts", type=int, default=0)
    ap.add_argument("--train-initials", default="20,60,100")
    ap.add_argument("--train-rates", default="4")
    ap.add_argument("--eval-initials", default="20,60,100")
    ap.add_argument("--eval-rates", default="4")
    ap.add_argument("--eval-seeds", default="930")
    ap.add_argument("--min-improvement", type=float, default=0.01)
    ap.add_argument("--require-heuristic-win", action="store_true")
    ap.add_argument("--require-no-regression", action="store_true")
    ap.add_argument("--max-regression", type=float, default=0.0)
    ap.add_argument("--run-prefix", default="paper_az")
    args = ap.parse_args()
    common_env_args = env_args(args)
    if bool(args.use_arrival_feature):
        common_env_args.append("--use-arrival-feature")

    accepted = Path(args.accepted_state)
    accepted.parent.mkdir(parents=True, exist_ok=True)
    if not accepted.exists():
        shutil.copyfile(args.start_state, accepted)

    accepted_eval_path = OUT / f"{args.run_prefix}_accepted_gate.csv"
    accepted_name = f"{args.run_prefix}_accepted_mcts_r1"
    unlink_outputs(accepted_eval_path)
    accepted_cmd = [
        sys.executable, "alphazero_benchmark.py",
        "--ckpt", args.ckpt,
        "--state", str(accepted),
        "--method-name", accepted_name,
        *common_env_args,
        "--windows", "20",
        "--initials", args.eval_initials,
        "--rates", args.eval_rates,
        "--seeds", args.eval_seeds,
        "--include-heuristics",
        "--include-model",
        "--rollouts", str(args.eval_rollouts),
        "--c-puct", str(args.c_puct),
        "--rollout-policy", str(args.rollout_policy),
        "--branch-rollout-threshold", str(args.branch_rollout_threshold),
        "--plan-mode", str(args.plan_mode),
        "--window-extract", str(args.window_extract),
        "--prior-uniform-mix", str(args.prior_uniform_mix),
        "--prior-search-bias", str(args.prior_search_bias),
        "--leaf-value-mix", str(args.leaf_value_mix),
        "--prior-mode", str(args.prior_mode),
        "--horizon-windows", str(args.eval_horizon_windows),
        "--expand-top-k", str(args.eval_expand_top_k),
        "--out", str(accepted_eval_path),
    ]
    if not bool(args.visit_unvisited_first):
        accepted_cmd.append("--disable-visit-unvisited-first")
    run(accepted_cmd)
    accepted_gate = gate_summary(accepted_eval_path, accepted_name)

    rows = []
    replay_paths: list[Path] = []
    best_score = float(accepted_gate["method_score"])
    best_eval_path = accepted_eval_path
    best_method_name = accepted_name
    for gen in range(1, int(args.generations) + 1):
        prefix = f"{args.run_prefix}_gen{gen:02d}"
        target_path = OUT / f"{prefix}_targets.pt"
        candidate_state = OUT / f"{prefix}_state.pt"
        append_targets = [str(p) for p in replay_paths[-int(args.replay_window):]] if int(args.replay_window) > 0 else []
        if str(args.extra_replay_targets).strip():
            append_targets.extend([part.strip() for part in str(args.extra_replay_targets).split(";") if part.strip()])
        train_cmd = [
            sys.executable, "alphazero_orthodox.py",
            "--ckpt", args.ckpt,
            "--load-state", str(accepted),
            *common_env_args,
            "--target-source", "mcts",
            "--episodes", str(args.episodes),
            "--seed", str(args.seed + 1000 * (gen - 1)),
            "--windows", "20",
            "--max-targets-per-episode", "1000000",
            "--train-initials", args.train_initials,
            "--train-rates", args.train_rates,
            "--rollouts", str(args.rollouts),
            "--c-puct", str(args.c_puct),
            "--rollout-policy", str(args.rollout_policy),
            "--branch-rollout-threshold", str(args.branch_rollout_threshold),
            "--plan-mode", str(args.plan_mode),
            "--window-extract", str(args.window_extract),
            "--prior-uniform-mix", str(args.prior_uniform_mix),
            "--prior-search-bias", str(args.prior_search_bias),
            "--leaf-value-mix", str(args.leaf_value_mix),
            "--prior-mode", str(args.prior_mode),
            "--horizon-windows", str(args.horizon_windows),
            "--expand-top-k", str(args.expand_top_k),
            "--self-play-sample-tau", str(args.self_play_sample_tau),
            "--root-dirichlet-frac", str(args.self_play_root_dirichlet_frac),
            "--root-dirichlet-alpha", str(args.self_play_root_dirichlet_alpha),
            "--terminal-baseline-target", "none" if str(args.value_target) == "mc_return" else str(args.value_target),
            "--terminal-score-metric", "reward",
            "--terminal-baseline-mode", str(args.terminal_baseline_mode),
            "--terminal-baseline-scale", str(args.terminal_baseline_scale),
            "--terminal-baseline-margin", str(args.terminal_baseline_margin),
            "--accepted-baseline-state", str(accepted),
            "--accepted-baseline-rollouts", str(args.accepted_baseline_rollouts),
            "--train-steps", str(args.train_steps),
            "--batch-size", "128",
            "--lr", str(args.inner_lr),
            "--type-loss-weight", str(args.type_loss_weight),
            "--track-loss-weight", str(args.track_loss_weight),
            "--sensor-loss-weight", str(args.sensor_loss_weight),
            "--joint-policy-loss-weight", str(args.joint_policy_loss_weight),
            "--value-loss-weight", str(args.value_loss_weight),
            "--factor-value-loss-weight", str(args.factor_value_loss_weight),
            "--policy-kl-weight", str(args.policy_kl_weight),
            "--skip-before-eval",
            "--skip-after-eval",
            "--save-targets", str(target_path),
            "--save-state", str(candidate_state),
            "--out-prefix", prefix,
        ]
        if not bool(args.visit_unvisited_first):
            train_cmd.append("--disable-visit-unvisited-first")
        if append_targets:
            train_cmd.extend(["--append-targets", ";".join(append_targets)])
        run(train_cmd)
        replay_paths.append(target_path)

        eval_path = OUT / f"{prefix}_gate.csv"
        candidate_name = f"{prefix}_mcts_r1"
        unlink_outputs(eval_path)
        candidate_cmd = [
            sys.executable, "alphazero_benchmark.py",
            "--ckpt", args.ckpt,
            "--state", str(candidate_state),
            "--method-name", candidate_name,
            *common_env_args,
            "--windows", "20",
            "--initials", args.eval_initials,
            "--rates", args.eval_rates,
            "--seeds", args.eval_seeds,
            "--include-heuristics",
            "--include-model",
            "--rollouts", str(args.eval_rollouts),
            "--c-puct", str(args.c_puct),
            "--rollout-policy", str(args.rollout_policy),
            "--branch-rollout-threshold", str(args.branch_rollout_threshold),
            "--plan-mode", str(args.plan_mode),
            "--window-extract", str(args.window_extract),
            "--prior-uniform-mix", str(args.prior_uniform_mix),
            "--prior-search-bias", str(args.prior_search_bias),
            "--leaf-value-mix", str(args.leaf_value_mix),
            "--prior-mode", str(args.prior_mode),
            "--horizon-windows", str(args.eval_horizon_windows),
            "--expand-top-k", str(args.eval_expand_top_k),
            "--out", str(eval_path),
        ]
        if not bool(args.visit_unvisited_first):
            candidate_cmd.append("--disable-visit-unvisited-first")
        run(candidate_cmd)
        candidate_gate = gate_summary(eval_path, candidate_name)
        accepted_cell_gate = compare_model_cells(eval_path, candidate_name, best_eval_path, best_method_name)
        candidate_score = float(candidate_gate["method_score"])
        accepted_score_before = float(best_score)
        beats_accepted = candidate_score > best_score + float(args.min_improvement)
        no_regression = float(accepted_cell_gate["min_margin_vs_accepted_cells"]) >= -float(args.max_regression)
        beats_heuristics = (
            float(candidate_gate["mean_margin_vs_best_heuristic"]) > float(args.min_improvement)
            and float(candidate_gate["win_rate_vs_best_heuristic"]) >= 1.0
        )
        accepted_candidate = (
            beats_accepted
            and (no_regression or not bool(args.require_no_regression))
            and (beats_heuristics or not bool(args.require_heuristic_win))
        )
        if accepted_candidate:
            shutil.copyfile(candidate_state, accepted)
            best_score = candidate_score
            accepted_gate = candidate_gate
            best_eval_path = eval_path
            best_method_name = candidate_name
        row = {
            "generation": gen,
            "candidate_score": candidate_score,
            "accepted_score_before": accepted_score_before,
            "mean_margin_vs_accepted_cells": float(accepted_cell_gate["mean_margin_vs_accepted_cells"]),
            "min_margin_vs_accepted_cells": float(accepted_cell_gate["min_margin_vs_accepted_cells"]),
            "win_rate_vs_accepted_cells": float(accepted_cell_gate["win_rate_vs_accepted_cells"]),
            "regression_cells": int(accepted_cell_gate["regression_cells"]),
            "compared_cells": int(accepted_cell_gate["compared_cells"]),
            "best_heuristic_score": float(candidate_gate["best_heuristic_score"]),
            "mean_margin_vs_best_heuristic": float(candidate_gate["mean_margin_vs_best_heuristic"]),
            "win_rate_vs_best_heuristic": float(candidate_gate["win_rate_vs_best_heuristic"]),
            "beats_accepted": bool(beats_accepted),
            "no_regression": bool(no_regression),
            "beats_heuristics": bool(beats_heuristics),
            "accepted": bool(accepted_candidate),
            "accepted_state": str(accepted),
            "candidate_state": str(candidate_state),
            "targets": str(target_path),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / f"{args.run_prefix}_loop_summary.csv", index=False)
        print(row, flush=True)


if __name__ == "__main__":
    main()
